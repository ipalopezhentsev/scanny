"""The camera worker: every PTP transaction happens on this one thread.

PTP allows a single outstanding transaction, and a capture can occupy the
camera for seconds. Running the frame grab and the commands on one worker
thread keeps the interface responsive without any lock contention, because Qt
queues the command slots in between frame grabs.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import comtypes
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from ..camera.nikon import CameraError, LiveViewFrame, NikonCamera
from ..wpd.device import MtpError, WpdCommandError

__all__ = ["CameraWorker"]

#: Interval between live-view grabs. The camera itself tops out near 30fps;
#: polling faster only re-reads frames that have not changed.
_FRAME_INTERVAL_MS = 33

#: Consecutive grab failures tolerated before concluding live view has ended.
_MAX_GRAB_ERRORS = 15

#: Magnification of each live-view zoom level, measured on a D750.
_MEASURED_MAGNIFICATION = {
    0: 1.0, 2: 2.35, 3: 3.13, 4: 4.7, 5: 6.27, 6: 9.4, 7: 18.8,
}


class CameraWorker(QObject):
    """Owns the camera connection and serialises all access to it."""

    connected = Signal(str)
    disconnected = Signal()
    frameReady = Signal(object)  # LiveViewFrame
    settingsReady = Signal(object)  # list[Setting]
    liveViewChanged = Signal(bool)
    zoomChanged = Signal(int)
    focusStateChanged = Signal(str)
    fpsChanged = Signal(float, int, int)  # frames per second, frame width, height
    exposurePreviewChanged = Signal(bool)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._camera: "NikonCamera | None" = None
        self._timer: "QTimer | None" = None
        self._grab_errors = 0
        self._zoom_level = 0
        self._com_ready = False
        self._last_frame: "LiveViewFrame | None" = None
        # Timestamps of recent frames, for the displayed rate.
        self._frame_times: "deque[float]" = deque(maxlen=60)
        # Magnification observed at each zoom level, seeded from measurements
        # on a D750 and corrected from live frames as levels are used, so
        # region zoom follows the body rather than a fitted curve.
        self._zoom_magnification: "dict[int, float]" = dict(_MEASURED_MAGNIFICATION)
        self._save_dir = Path.home() / "Pictures" / "scanny"

    # -- thread lifecycle --------------------------------------------------

    @Slot()
    def initialise(self) -> None:
        """Join a COM apartment on this thread, then open the camera.

        COM is per-thread, so the apartment the main thread set up does not
        carry over here. WPD's ``PortableDeviceFTM`` is the free-threaded
        variant, meant for the multithreaded apartment -- which also spares
        this thread from having to pump a Windows message queue alongside
        Qt's own event loop.
        """
        try:
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        except OSError as exc:
            self.failed.emit(f"Could not initialise COM on the camera thread: {exc}")
            return
        self._com_ready = True
        self.connect_camera()

    @Slot()
    def shutdown(self) -> None:
        self.disconnect_camera()
        if self._com_ready:
            comtypes.CoUninitialize()
            self._com_ready = False

    # -- connection --------------------------------------------------------

    @Slot()
    def connect_camera(self) -> None:
        if self._camera is not None:
            return
        try:
            camera = NikonCamera.open()
        except CameraError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover - driver-level failures
            self.failed.emit(f"Could not open the camera: {exc}")
            return
        self._camera = camera
        battery = camera.battery_level()
        suffix = f" - battery {battery}%" if battery is not None else ""
        self.connected.emit(f"{camera.model}  |  firmware {camera.firmware}{suffix}")
        self.status.emit("Connected")
        self.refresh_settings()

    @Slot()
    def disconnect_camera(self) -> None:
        self._stop_timer()
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:
                pass
            self._camera = None
        self.disconnected.emit()
        self.status.emit("Disconnected")

    # -- live view ---------------------------------------------------------

    @Slot()
    def start_live_view(self) -> None:
        camera = self._require()
        if camera is None:
            return
        self.status.emit("Starting live view...")
        try:
            camera.start_live_view()
        except CameraError as exc:
            self.failed.emit(str(exc))
            return
        self._grab_errors = 0
        self._frame_times.clear()
        self._zoom_level = camera.zoom_level()
        self.exposurePreviewChanged.emit(camera.exposure_preview)
        self.liveViewChanged.emit(True)
        self.zoomChanged.emit(self._zoom_level)
        self.status.emit("Live view running")
        self._start_timer()

    @Slot()
    def stop_live_view(self) -> None:
        self._stop_timer()
        if self._camera is not None:
            self._camera.stop_live_view()
        self._frame_times.clear()
        self._last_frame = None
        self.fpsChanged.emit(0.0, 0, 0)
        self.liveViewChanged.emit(False)
        self.status.emit("Live view stopped")

    def _start_timer(self) -> None:
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.setTimerType(Qt.TimerType.PreciseTimer)
            self._timer.timeout.connect(self._grab)
        self._timer.start(_FRAME_INTERVAL_MS)

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()

    @Slot()
    def _grab(self) -> None:
        camera = self._camera
        if camera is None or not camera.live_view_active:
            return
        try:
            frame = camera.live_view_frame()
        except (CameraError, MtpError, WpdCommandError):
            self._grab_errors += 1
            # The odd dropped frame is normal while the camera adjusts; a run
            # of them means live view really has ended, because someone pressed
            # a button on the body or the mirror dropped.
            if self._grab_errors >= _MAX_GRAB_ERRORS:
                self._stop_timer()
                camera.stop_live_view()
                self.liveViewChanged.emit(False)
                self.failed.emit("Live view stopped responding and was shut down.")
            return
        self._grab_errors = 0
        self._last_frame = frame
        self._note_magnification(frame)
        self._note_frame_rate(frame)
        self.frameReady.emit(frame)

    def _note_magnification(self, frame: LiveViewFrame) -> None:
        if frame.crop_width:
            self._zoom_magnification[self._zoom_level] = frame.magnification

    def _note_frame_rate(self, frame: LiveViewFrame) -> None:
        """Publish the rate measured over the last second of frames."""
        now = time.monotonic()
        self._frame_times.append(now)
        while len(self._frame_times) > 2 and now - self._frame_times[0] > 1.0:
            self._frame_times.popleft()
        if len(self._frame_times) < 2:
            return
        span = self._frame_times[-1] - self._frame_times[0]
        if span > 0:
            fps = (len(self._frame_times) - 1) / span
            self.fpsChanged.emit(fps, frame.width, frame.height)

    # -- focus -------------------------------------------------------------

    @Slot(float, float)
    def move_point(self, nx: float, ny: float) -> None:
        """Move the focus rectangle to the clicked point.

        Nothing else: no focus, no magnification. Moving the point is also
        what pans the camera's magnified view, so this doubles as the way to
        aim before zooming in.
        """
        camera = self._require()
        if camera is None:
            return
        frame = self._current_frame()
        if frame is None:
            return
        x, y = frame.to_af_coords(nx, ny)
        try:
            camera.set_af_area(x, y)
        except CameraError as exc:
            self.failed.emit(str(exc))
            return
        self.status.emit(f"Focus point at ({x}, {y})")

    @Slot(int)
    def drive_focus(self, steps: int) -> None:
        """Move focus manually by a signed number of drive steps."""
        camera = self._require()
        if camera is None:
            return
        try:
            moved = camera.drive_focus(steps)
        except CameraError as exc:
            self.failed.emit(str(exc))
            return
        where = "nearer" if steps < 0 else "further"
        count = abs(steps)
        self.status.emit(
            f"Focus {count} step{'' if count == 1 else 's'} {where}"
            if moved
            else f"Focus is already at its {where} limit"
        )

    @Slot()
    def autofocus(self) -> None:
        camera = self._require()
        if camera is None:
            return
        self.focusStateChanged.emit("busy")
        try:
            focused = camera.autofocus()
        except CameraError as exc:
            self.focusStateChanged.emit("idle")
            self.failed.emit(str(exc))
            return
        self.focusStateChanged.emit("focused" if focused else "idle")
        self.status.emit("Focus locked" if focused else "Could not find focus")

    # -- zoom --------------------------------------------------------------

    @Slot(int)
    def set_zoom(self, level: int) -> None:
        camera = self._require()
        if camera is None:
            return
        level = min(NikonCamera.ZOOM_LEVELS, key=lambda v: abs(v - int(level)))
        try:
            camera.set_zoom_level(level)
        except CameraError as exc:
            self.failed.emit(str(exc))
            return
        self._zoom_level = level
        self.zoomChanged.emit(level)
        self.status.emit("Zoom: full frame" if level == 0 else f"Zoom level {level}")

    @Slot(int)
    def step_zoom(self, delta: int) -> None:
        """Move one usable zoom level in or out, skipping ones the body rejects."""
        if self._camera is None:
            return
        levels = NikonCamera.ZOOM_LEVELS
        try:
            index = levels.index(self._zoom_level)
        except ValueError:
            index = 0
        self.set_zoom(levels[max(0, min(index + delta, len(levels) - 1))])

    @Slot(float, float, float, float)
    def zoom_to_region(self, nx: float, ny: float, nw: float, nh: float) -> None:
        """Magnify onto a dragged rectangle.

        The camera centres its magnified view on the focus point, so putting
        the focus point at the middle of the selection and choosing the
        strongest magnification that still shows the whole selection lands the
        region on screen.
        """
        camera = self._require()
        if camera is None:
            return
        frame = self._current_frame()
        if frame is None or nw <= 0 or nh <= 0:
            return
        x, y = frame.to_af_coords(nx + nw / 2, ny + nh / 2)
        # How much further in we need to go, relative to what is on screen now.
        wanted = frame.magnification / max(nw, nh)
        level = self._level_for_magnification(wanted)
        try:
            camera.set_af_area(x, y)
            camera.set_zoom_level(level)
        except CameraError as exc:
            self.failed.emit(str(exc))
            return
        self._zoom_level = level
        self.zoomChanged.emit(level)
        self.status.emit(f"Magnified {wanted:.1f}x on ({x}, {y}) - zoom level {level}")

    def _level_for_magnification(self, wanted: float) -> int:
        """The strongest usable level that does not overshoot the request."""
        usable = [
            (level, self._zoom_magnification.get(level, _MEASURED_MAGNIFICATION[level]))
            for level in NikonCamera.ZOOM_LEVELS
        ]
        fitting = [level for level, mag in usable if mag <= wanted + 1e-6]
        return max(fitting) if fitting else NikonCamera.ZOOM_LEVELS[0]

    @Slot()
    def reset_zoom(self) -> None:
        self.set_zoom(0)

    @Slot()
    def toggle_zoom(self) -> None:
        """Magnify fully, or come back out if already magnified.

        The state is taken from the live frame rather than a remembered level,
        so it still agrees with what is on screen if the zoom was changed on
        the camera body.
        """
        camera = self._require()
        if camera is None:
            return
        frame = self._last_frame or self._current_frame()
        magnified = frame is not None and frame.magnification > 1.01
        self.set_zoom(0 if magnified else max(NikonCamera.ZOOM_LEVELS))

    @Slot(int, int)
    def pan(self, dx: int, dy: int) -> None:
        """Scroll the magnified view by moving the focus point.

        A step is an eighth of whatever is currently on screen, so the view
        travels by the same visible amount at every magnification. Autofocus is
        deliberately not driven here: panning should not hunt.
        """
        camera = self._require()
        if camera is None:
            return
        frame = self._current_frame()
        if frame is None:
            return
        step_x = max(1, (frame.crop_width or frame.image_width) // 8)
        step_y = max(1, (frame.crop_height or frame.image_height) // 8)
        x, y = frame.af_x + dx * step_x, frame.af_y + dy * step_y
        # Reuse the frame's own clamping so the focus box cannot leave the sensor.
        half_w, half_h = frame.af_width // 2, frame.af_height // 2
        x = min(max(x, half_w), max(frame.image_width - half_w, half_w))
        y = min(max(y, half_h), max(frame.image_height - half_h, half_h))
        if (x, y) == (frame.af_x, frame.af_y):
            return
        try:
            camera.set_af_area(int(x), int(y))
        except CameraError as exc:
            self.failed.emit(str(exc))

    # -- exposure preview --------------------------------------------------

    @Slot(bool)
    def set_exposure_preview(self, enabled: bool) -> None:
        camera = self._require()
        if camera is None:
            return
        camera.set_exposure_preview(enabled)
        self.exposurePreviewChanged.emit(camera.exposure_preview)
        self.status.emit(
            "Live view shows the actual exposure"
            if enabled
            else "Live view brightness normalised by the camera"
        )

    # -- settings ----------------------------------------------------------

    @Slot()
    def refresh_settings(self) -> None:
        camera = self._require()
        if camera is None:
            return
        try:
            self.settingsReady.emit(camera.settings())
        except Exception as exc:
            self.failed.emit(f"Could not read camera settings: {exc}")

    @Slot(str, object)
    def set_setting(self, name: str, value: Any) -> None:
        camera = self._require()
        if camera is None:
            return
        try:
            camera.set_setting(name, value)
        except (CameraError, KeyError) as exc:
            self.failed.emit(str(exc))
        else:
            self.status.emit(f"{name} set")
        self.refresh_settings()

    # -- capture -----------------------------------------------------------

    @Slot(bool, bool)
    def capture(self, autofocus: bool, download: bool) -> None:
        camera = self._require()
        if camera is None:
            return
        self.status.emit("Releasing shutter...")
        try:
            handles = camera.capture(autofocus=autofocus)
        except CameraError as exc:
            self.failed.emit(str(exc))
            return
        if not handles:
            self.status.emit("Shutter fired (camera reported no new file)")
            self.refresh_settings()
            return

        names = []
        for handle in handles:
            try:
                info = camera.session.object_info(handle)
            except (MtpError, WpdCommandError):
                continue
            names.append(info.filename)
            if download:
                self._download(camera, handle, info.filename)
        if names and not download:
            self.status.emit("Captured " + ", ".join(names))
        self.refresh_settings()

    def _download(self, camera: NikonCamera, handle: int, filename: str) -> None:
        self.status.emit(f"Downloading {filename}...")
        try:
            _, data = camera.download(handle)
        except (MtpError, WpdCommandError, CameraError) as exc:
            self.failed.emit(f"Could not download {filename}: {exc}")
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        path = self._save_dir / filename
        stem, suffix, counter = path.stem, path.suffix, 1
        while path.exists():
            path = self._save_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        path.write_bytes(data)
        self.status.emit(f"Saved {path}")

    @Slot(str)
    def set_save_directory(self, path: str) -> None:
        self._save_dir = Path(path)

    @property
    def save_directory(self) -> Path:
        return self._save_dir

    # -- helpers -----------------------------------------------------------

    def _require(self) -> "NikonCamera | None":
        if self._camera is None:
            self.failed.emit("No camera is connected.")
            return None
        return self._camera

    def _current_frame(self) -> "LiveViewFrame | None":
        """The newest frame, preferring the one the grab loop just fetched.

        Its geometry is at most one frame old, which is close enough for
        mapping a click and saves a round trip on every gesture.
        """
        camera = self._camera
        if camera is None or not camera.live_view_active:
            self.failed.emit("Start live view first.")
            return None
        if self._last_frame is not None:
            return self._last_frame
        try:
            return camera.live_view_frame()
        except (CameraError, MtpError, WpdCommandError):
            return None
