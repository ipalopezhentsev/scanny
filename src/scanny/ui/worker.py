"""The camera worker: every PTP transaction happens on this one thread.

PTP allows a single outstanding transaction, and a capture can occupy the
camera for seconds. Running the frame grab and the commands on one worker
thread keeps the interface responsive without any lock contention, because Qt
queues the command slots in between frame grabs.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from pathlib import Path
from typing import Any

import comtypes
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage

from ..camera.nikon import CameraError, LiveViewFrame, NikonCamera
from ..wpd.device import MtpError, WpdCommandError
from .hunt import Walk
from .integration import FrameIntegrator
from .naming import NameSequence, unique
from .pixels import green
from .sharpness import SharpnessMeter, grain_reading, measure, variance_between

__all__ = ["CameraWorker"]

#: Interval between live-view grabs. Measured on a D750, the body draws a new
#: frame every 23ms -- about 44 a second, not the 30 this was originally set
#: for -- and a read plus its decode costs enough that the timer has to be set
#: well under that to keep up with it. Frames actually collected, against the
#: interval asked for:
#:
#: ===== =====  ===== =====  ===== =====
#: 33ms  30/s   25ms  34/s   20ms  40/s
#: 15ms  44/s   10ms  44/s    5ms  44/s
#: ===== =====  ===== =====  ===== =====
#:
#: 15ms is where that flattens: everything the camera draws, without spending
#: reads to find it. Polling harder is not wrong, only wasted -- the duplicates
#: it collects are discarded either way.
#:
#: The 44 is not a constant of the body. Magnification decides it: out to 3.13x
#: it is 44 with the exposure preview on and 30 with it off, and from 4.7x up
#: it is 16 whichever way the preview is set. The live-view frame size does not
#: enter into it -- 320x180 is drawn at the same rate as 640x360 everywhere.
#:
#: So no number here is load-bearing, which is the whole point: the timer
#: overshoots whatever the rate currently is -- by 4x, in that 16fps corner --
#: and :meth:`CameraWorker._grab` throws away the re-reads. The alternative,
#: a timer tracking the observed rate, would save reads that cost 6ms each and
#: would have to be got right at every zoom and preview transition.
_FRAME_INTERVAL_MS = 15

#: Consecutive grab failures tolerated before concluding live view has ended.
_MAX_GRAB_ERRORS = 15

#: How the settling after a focus move is judged. `MfDrive` has returned by
#: the time the call does, but live view is a frame or two behind the lens and
#: the lens may still be moving, so the picture is watched until it stops
#: changing rather than waited on for a fixed time -- a guessed wait that is
#: too short is not visibly wrong, it just feeds the hunt readings of where
#: focus used to be, and sends it the wrong way.
_SETTLE_FRAMES = 6  # never believe stillness before this many frames
_SETTLE_LIMIT = 30  # nor wait longer than this for it
_STILL = 0.05  # frames this close to each other are the same picture

#: The deepest pipeline that will be believed. Live view runs a handful of
#: frames behind the lens; a picture that only changes half a second after the
#: move did not change because of the move, and taking that for the pipeline
#: would have every probe after it wait the maximum for nothing.
_LAG_LIMIT = 12

#: How many frame-to-frame noise measurements to keep. The smallest of them is
#: the estimate, so this wants to be long enough to hold a spell of the picture
#: holding still -- a couple of seconds of frames.
_NOISE_MEMORY = 60

#: How much the picture has to change before a move counts as having arrived.
#: Far above the wander in a single frame on purpose: the move being watched
#: for is the first of a hunt, which is a coarse step and changes the reading
#: several times over. Mistaking grain for the move measures the pipeline as
#: shorter than it is, and everything after it is then read too early.
_MOVED = 0.25
#: Differences are judged against the best reading of the hunt as well as
#: against the readings being compared. Thoroughly defocused, the reading is
#: nearly zero and one grain of noise is a difference of hundreds of per cent
#: -- which would read as the picture changing, and as never being still.
_READING_FLOOR = 0.05

#: Magnification of each live-view zoom level, measured on a D750.
_MEASURED_MAGNIFICATION = {
    0: 1.0, 2: 2.35, 3: 3.13, 4: 4.7, 5: 6.27, 6: 9.4, 7: 18.8,
}


class CameraWorker(QObject):
    """Owns the camera connection and serialises all access to it."""

    connected = Signal(str)
    disconnected = Signal()
    #: A live-view frame, and the picture to show with it. The picture is a
    #: null QImage when this frame only moves the overlay -- which is what the
    #: frames in the middle of an integration stack do, since their pixels are
    #: still being added up.
    frameReady = Signal(object, object)  # LiveViewFrame, QImage
    settingsReady = Signal(object)  # list[Setting]
    liveViewChanged = Signal(bool)
    zoomChanged = Signal(int)
    focusStateChanged = Signal(str)
    fpsChanged = Signal(float, int, int)  # frames per second, frame width, height
    #: The sharpness of the displayed picture, and the best seen of this view.
    sharpnessChanged = Signal(float, float)
    #: Whether the focus hunt is running.
    huntChanged = Signal(bool)
    exposurePreviewChanged = Signal(bool)
    #: Whether shots are being written to the camera's card. Reported back
    #: rather than assumed: a body that refuses the choice keeps using it.
    saveToCardChanged = Signal(bool)
    #: The shutter delays the connected body offers, in seconds, and the one
    #: it is itself set to. Both are asked of the camera: what is on offer
    #: varies by model, and what the body is set to is where the choice starts.
    shutterDelaysAvailable = Signal(object, object)  # tuple[int, ...], int | None
    #: The number the next saved picture will be given, reported once a shot
    #: has used one. Only ever emitted when the counter moved by itself: an
    #: override comes from the window, and is not echoed back at it.
    nextNumber = Signal(int)
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
        # Digest of the last JPEG the camera sent, to recognise a re-read of a
        # frame it has not replaced yet. See _grab.
        self._last_digest: "bytes | None" = None
        # Whether those re-reads are discarded. On, unless someone wants to see
        # the raw poll rate or suspects the check itself of dropping frames.
        self._deduplicate = True
        # Timestamps of recent frames, for the displayed rate.
        self._frame_times: "deque[float]" = deque(maxlen=60)
        # Magnification observed at each zoom level, seeded from measurements
        # on a D750 and corrected from live frames as levels are used, so
        # region zoom follows the body rather than a fitted curve.
        self._zoom_magnification: "dict[int, float]" = dict(_MEASURED_MAGNIFICATION)
        # Noise averaging. Off by default: it costs frame rate, so it is the
        # user's to ask for.
        self._integrator = FrameIntegrator()
        # Focusing by hand against a number. Measured on whatever is displayed,
        # so integration cleans up the picture it reads.
        self._sharpness = SharpnessMeter()
        # Focusing by hunting the reading rather than by hand. Advanced by
        # frames, not by a loop of its own, so the grab keeps running.
        self._hunt: "FocusHunt | None" = None
        self._hunt_opening = False
        self._settling = False
        self._settle_seen = 0
        self._settle_last: "float | None" = None
        self._settle_before: "float | None" = None
        self._settle_changed = False
        self._settle_moving = False
        # How many frames live view runs behind the lens, measured from the
        # first move of each hunt rather than assumed.
        self._pipeline_lag: "int | None" = None
        # How much grain is on a single frame, measured from consecutive ones
        # rather than guessed at from within one. See sharpness.measure.
        self._noise_seen: "deque[float]" = deque(maxlen=_NOISE_MEMORY)
        self._last_pixels: "np.ndarray | None" = None
        # What a picture of nothing but grain would read, which is the scale
        # small readings are judged against.
        self._grain_scale = 0.0
        self._settle_frames = 0
        self._save_dir = Path.home() / "Pictures" / "scanny"
        # Tethered by default: the picture comes straight down the cable and
        # the card is left out of it. See NikonCamera.set_save_to_card.
        self._save_to_card = False
        # Seconds between the mirror lifting and the shutter firing. Kept here
        # as well as on the camera so it survives a reconnection.
        self._shutter_delay = 0
        # What downloaded pictures are called. Off by default: the camera's
        # own names are what someone expects until they ask for otherwise.
        self._namer = NameSequence()

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
        """Close the camera down, then end this thread's event loop.

        The loop is ended from in here rather than from the window, because
        quitting it from the outside races this slot: the exit flag is only
        looked at between events, so a quit posted straight after the request
        wins whenever the thread is busy with a frame grab, and the camera is
        left with its mirror up and live view running after the program has
        gone.
        """
        self.disconnect_camera()
        if self._com_ready:
            comtypes.CoUninitialize()
            self._com_ready = False
        thread = QThread.currentThread()
        if thread is not None:
            thread.quit()

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
        camera.set_save_to_card(self._save_to_card)
        # The camera has the last word: one that will not be told where to
        # record keeps using its card.
        self._save_to_card = camera.save_to_card
        self.saveToCardChanged.emit(self._save_to_card)
        camera.set_shutter_delay(self._shutter_delay)
        self.shutterDelaysAvailable.emit(
            camera.shutter_delay_choices(), camera.shutter_delay_on_body()
        )
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
        # Forget rather than reset: the canvas is blank, so the first frame
        # has to go up as it arrives instead of a stack later.
        self._integrator.forget()
        self._sharpness.reset()
        self._last_digest = None
        self._zoom_level = camera.zoom_level()
        self.exposurePreviewChanged.emit(camera.exposure_preview)
        self.liveViewChanged.emit(True)
        self.zoomChanged.emit(self._zoom_level)
        self.status.emit("Live view running")
        self._start_timer()

    @Slot()
    def stop_live_view(self) -> None:
        self._cancel_hunt("")
        self._stop_timer()
        if self._camera is not None:
            self._camera.stop_live_view()
        self._frame_times.clear()
        self._integrator.forget()
        self._sharpness.reset()
        self._last_frame = None
        self._last_digest = None
        self.fpsChanged.emit(0.0, 0, 0)
        self.sharpnessChanged.emit(0.0, 0.0)
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
                self._cancel_hunt("")
                camera.stop_live_view()
                self.liveViewChanged.emit(False)
                self.failed.emit("Live view stopped responding and was shut down.")
            return
        self._grab_errors = 0
        # Polling overshoots the camera's draw rate on purpose, so some reads
        # return the frame that was already here. Everything below this point
        # would be wrong or wasted on one: averaging a frame with itself
        # cancels no noise (duplicated noise adds coherently, so the mean of a
        # stack is the mean of its *distinct* frames, and counting a re-read
        # towards the stack only makes it finish early and grainier than it
        # claims), the grain meter reads the difference between consecutive
        # frames and would call an identical pair noise-free, and the JPEG
        # decode is the most expensive thing on this thread.
        if self._deduplicate:
            digest = hashlib.blake2b(frame.jpeg, digest_size=8).digest()
            if digest == self._last_digest:
                return
            self._last_digest = digest
        self._last_frame = frame
        self._note_magnification(frame)
        if self._settling:
            # The lens is moving. A frame caught mid-move belongs to no focus
            # position in particular, so nothing is shown or measured from
            # one: a stack completing here would blend two focus positions
            # into one picture, and the reading taken off it would describe
            # neither. Whatever these frames make of the stack is dropped when
            # the move lands.
            #
            # They still go through the integrator, because the grain meter
            # reads the frame it decoded and the settling is judged against
            # that grain -- starve it and a blank picture never reads as
            # still. The overlay wants the frame too: the focus box and the
            # level on it are current.
            self._integrator.add(frame)
            self._note_noise()
            self.frameReady.emit(frame, QImage())
            if self._hunt is not None:
                self._advance_hunt(frame, None)
            return
        # Every frame is published, so the focus box and the level readout stay
        # as responsive as the camera is; only the picture waits for its stack.
        image = self._integrator.add(frame)
        self._note_noise()
        reading = None
        if image is not None:
            self._note_frame_rate(frame)
            # Only a completed stack is measured. The odd one out is the
            # single frame shown the instant the view moves, which carries the
            # full grain of one frame: reading it would put a spike in the
            # trend and could hand the hunt a peak that no focus position can
            # be returned to.
            if self._integrator.last_image_was_whole:
                reading = self._note_sharpness(frame, image)
        self.frameReady.emit(frame, image if image is not None else QImage())
        if self._hunt is not None:
            self._advance_hunt(frame, reading)

    def _note_noise(self) -> None:
        """Measure the grain from the frame just decoded and the one before it.

        Only worth doing while something is going to read it, and only from a
        pair of frames: how much grain there is cannot be told from inside a
        single frame, because at best focus the finest detail in the picture
        looks exactly like grain.
        """
        if not self._sharpness.enabled:
            self._last_pixels = None
            return
        frame_image = self._integrator.last_frame
        if frame_image is None:
            return
        pixels = green(frame_image)[::2, ::2]  # every other one is plenty
        variance = variance_between(self._last_pixels, pixels) if (
            self._last_pixels is not None
        ) else None
        self._last_pixels = pixels
        if variance is not None:
            self._noise_seen.append(variance)

    @property
    def _noise_variance(self) -> float:
        """The quietest pair of frames lately: the one with no movement in it."""
        return min(self._noise_seen) if self._noise_seen else 0.0

    def _note_sharpness(
        self, frame: LiveViewFrame, image: QImage
    ) -> "float | None":
        """Read the picture that is about to be shown, if measuring is on."""
        # The picture is the mean of however many frames went into it, and
        # averaging frames divides their noise by as many. Only whole stacks
        # reach here, so that is always the full count.
        frames = self._integrator.frames if self._integrator.enabled else 1
        reading = self._sharpness.measure(
            frame, image, self._noise_variance / frames
        )
        if reading is None:
            return None
        self.sharpnessChanged.emit(*reading)
        return reading[0]

    def _note_magnification(self, frame: LiveViewFrame) -> None:
        if frame.crop_width:
            self._zoom_magnification[self._zoom_level] = frame.magnification

    def _note_frame_rate(self, frame: LiveViewFrame) -> None:
        """Publish the rate measured over the last second of displayed images.

        Displayed, not grabbed: while frames are being integrated the camera
        still draws forty-odd a second, but the picture only changes when a
        stack completes, and that is the number worth showing.
        """
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
        self._move_point_to(camera, *frame.to_af_coords(nx, ny))

    @Slot(float, float)
    def move_point_in_frame(self, fx: float, fy: float) -> None:
        """Move the focus rectangle to a fraction of the *whole* frame.

        What the navigator drags. Unlike :meth:`move_point`, whose coordinates
        are fractions of the picture on screen, these are fractions of the
        frame that picture was cut from -- the only coordinates the navigator
        has, since it is showing the whole frame while the live view shows a
        part of it. Moving the point is what pans the magnified view, so this
        is also how the navigator scrolls the picture.
        """
        camera = self._require()
        if camera is None:
            return
        frame = self._current_frame()
        if frame is None:
            return
        self._move_point_to(camera, *frame.to_af_coords_in_frame(fx, fy))

    def _move_point_to(self, camera: NikonCamera, x: int, y: int) -> None:
        try:
            camera.set_af_area(x, y)
        except CameraError as exc:
            self.failed.emit(str(exc))
            return
        self.status.emit(f"Focus point at ({x}, {y})")

    @Slot(int)
    def drive_focus(self, steps: int) -> None:
        """Move focus manually by a signed number of drive steps."""
        self._cancel_hunt("Focus hunt stopped - you took the focus")
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
        self._cancel_hunt("Focus hunt stopped - the camera's autofocus took over")
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
        self._cancel_hunt("Focus hunt stopped - the view changed")
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
        # The frame's own clamping, so the focus box cannot leave the sensor.
        x, y = frame.clamp_af_coords(
            frame.af_x + dx * step_x, frame.af_y + dy * step_y
        )
        if (x, y) == (frame.af_x, frame.af_y):
            return
        try:
            camera.set_af_area(x, y)
        except CameraError as exc:
            self.failed.emit(str(exc))

    # -- exposure preview --------------------------------------------------

    @Slot(bool)
    def set_exposure_preview(self, enabled: bool) -> None:
        camera = self._require()
        if camera is None:
            return
        camera.set_exposure_preview(enabled)
        self._forget_sharpness()
        self.exposurePreviewChanged.emit(camera.exposure_preview)
        # A D750 drops back to 1.0x when this is written, without saying so.
        # Left unread, the slider and the navigator would go on claiming a
        # magnification the camera is no longer at.
        self._zoom_level = camera.zoom_level()
        self.zoomChanged.emit(self._zoom_level)
        self.status.emit(
            "Live view shows the actual exposure"
            if enabled
            else "Live view brightness normalised by the camera"
        )

    # -- where shots are recorded ------------------------------------------

    @Slot(bool)
    def set_save_to_card(self, enabled: bool) -> None:
        # Remembered even with no camera attached, so the choice survives a
        # reconnection and is applied to whatever turns up next.
        self._save_to_card = bool(enabled)
        camera = self._camera
        if camera is not None and not camera.set_save_to_card(self._save_to_card):
            self._save_to_card = camera.save_to_card
            self.failed.emit(
                "This camera will not be told where to record, so shots keep "
                "going to its card."
            )
        else:
            self.status.emit(
                "Shots will be written to the camera's card"
                if self._save_to_card
                else "Shots will go straight to the computer"
            )
        self.saveToCardChanged.emit(self._save_to_card)

    @property
    def save_to_card(self) -> bool:
        return self._save_to_card

    # -- shutter delay -----------------------------------------------------

    @Slot(int)
    def set_shutter_delay(self, seconds: int) -> None:
        """Wait this many seconds, mirror up, before each shot is released."""
        # Held here as well as on the camera, for the same reason as the
        # recording media: the choice outlives any one connection.
        self._shutter_delay = max(0, int(seconds))
        camera = self._camera
        if camera is not None:
            camera.set_shutter_delay(self._shutter_delay)
        self.status.emit(
            f"The shutter will fire {self._shutter_delay}s after the mirror lifts"
            if self._shutter_delay
            else "The shutter will fire as soon as it is asked to"
        )

    @property
    def shutter_delay(self) -> int:
        return self._shutter_delay

    # -- noise integration -------------------------------------------------

    @Slot(bool, int)
    def set_integration(self, enabled: bool, frames: int) -> None:
        """Average *frames* live-view frames into each displayed image."""
        if not self._integrator.configure(enabled, frames):
            return
        self._cancel_hunt("Focus hunt stopped - the integration changed")
        # The rate is about to change, so the last second of it is no longer
        # anything worth averaging into the readout.
        self._frame_times.clear()
        # Nor is the best sharpness: how much noise is left in the picture is
        # part of what the reading measures.
        self._sharpness.reset()
        self.sharpnessChanged.emit(0.0, 0.0)
        self.status.emit(
            f"Integrating {self._integrator.frames} frames into each image"
            if self._integrator.enabled
            else "Showing every frame as it arrives"
        )

    @Slot(bool)
    def set_deduplicate(self, enabled: bool) -> None:
        """Whether a read returning the frame already in hand is discarded.

        Off, every read is treated as a frame, which is what the code did
        before the camera's draw rate was measured. It is worth having as a
        switch rather than a constant: it is the difference between the rate
        on the status bar meaning frames the camera drew and meaning times a
        second we asked, and seeing both is how the gap between them -- which
        is large, and moves with zoom -- can be seen at all.
        """
        if bool(enabled) == self._deduplicate:
            return
        self._deduplicate = bool(enabled)
        # A stack begun under the other rule is a mix of the two.
        self._last_digest = None
        self._integrator.reset()
        self._frame_times.clear()
        self._forget_sharpness()
        self.status.emit(
            "Counting only the frames the camera redraws"
            if enabled
            else "Counting every read, redrawn or not"
        )

    # -- sharpness ---------------------------------------------------------

    @Slot(bool)
    def set_sharpness(self, enabled: bool) -> None:
        """Measure the contrast in each displayed picture, or stop measuring."""
        if not self._sharpness.configure(enabled):
            return
        self._cancel_hunt("")
        self.sharpnessChanged.emit(0.0, 0.0)
        self.status.emit(
            "Measuring sharpness - drive focus and keep the direction that raises it"
            if enabled
            else "Sharpness measurement off"
        )

    @Slot(object)
    def set_sharpness_area(self, area: object) -> None:
        """Measure only this part of the picture, or all of it when None.

        The area arrives as fractions of the displayed picture, which is what
        makes it usable past the camera's strongest magnification: there is no
        more zooming to be had there, but a rectangle drawn on what is already
        on screen can be as small as the subject is.
        """
        if not self._sharpness.set_area(area):
            return
        self._cancel_hunt("Focus hunt stopped - the measured area moved")
        self.sharpnessChanged.emit(0.0, 0.0)
        self.status.emit(
            "Measuring sharpness over the selected area"
            if area is not None
            else "Measuring sharpness over the whole frame"
        )

    @Slot()
    def reset_sharpness_peak(self) -> None:
        self._sharpness.reset()
        self.sharpnessChanged.emit(0.0, 0.0)

    def _forget_sharpness(self) -> None:
        """Start the readings again, because the picture is no longer the same.

        Anything that changes what the camera sends changes what the reading
        means. Exposure is the one that bites: a darker live view is a grainier
        one and, with exposure preview on, a deeper one too, so readings either
        side of a change of aperture are measuring different pictures.
        """
        self._cancel_hunt("Focus hunt stopped - the picture changed under it")
        if self._sharpness.enabled:
            self._sharpness.reset()
            self.sharpnessChanged.emit(0.0, 0.0)

    # -- hunting the sharpest focus ----------------------------------------

    @Slot(int)
    def fine_tune(self, step: int) -> None:
        """Walk to focus in one increment, for focus that is already close."""
        step = abs(int(step)) or 1
        self._start_hunt(Walk(step), f"Fine tuning focus in steps of {step}...")

    def _start_hunt(self, hunt, announcement: str) -> None:
        camera = self._require()
        if camera is None or self._hunt is not None:
            return
        if not camera.live_view_active:
            self.failed.emit("Start live view before hunting for focus.")
            return
        if not self._sharpness.enabled:
            self.failed.emit("Switch sharpness measuring on before hunting for focus.")
            return
        self._hunt = hunt
        self._pipeline_lag = None
        # Whether the camera's own autofocus is needed first is decided on the
        # first settled reading rather than here: the meter may simply not have
        # read anything yet, and assuming that means "defocused" would throw
        # away good focus the moment the button was pressed.
        self._hunt_opening = True
        # A fresh line for the hunt, so what is plotted is the hunt itself.
        self._sharpness.reset()
        self.sharpnessChanged.emit(0.0, 0.0)
        self._settle()
        self.huntChanged.emit(True)
        self.status.emit(announcement)

    @Slot()
    def cancel_hunt(self) -> None:
        self._cancel_hunt("Focus hunt stopped")

    def _cancel_hunt(self, why: str = "") -> None:
        """Stop a hunt, if one is running. Silent when nothing is happening."""
        if self._hunt is None:
            return
        self._hunt = None
        self._settling = False
        self.huntChanged.emit(False)
        if why:
            self.status.emit(why)

    def _settle(self, before: "float | None" = None) -> None:
        """Watch frames until the picture has moved and stopped, then stack.

        *before* is what the picture read at the moment of the move, which is
        how the first settle of a hunt can tell that the move has come through
        the pipeline at all.
        """
        self._settling = True
        self._settle_seen = 0
        self._settle_last = None
        self._settle_before = before
        self._settle_changed = False
        self._settle_moving = False

    def _advance_hunt(
        self, frame: LiveViewFrame, reading: "float | None"
    ) -> None:
        """Give the hunt one settled reading, and carry out what it asks for.

        Two kinds of frame are refused here, and both matter. The frames while
        the picture is still changing are refused because live view lags the
        lens and the lens may still be moving; the frames in the middle of an
        integration stack are refused because that picture is still half of
        where focus used to be. What the hunt sees is only ever a whole picture
        of where the lens is now.
        """
        hunt, camera = self._hunt, self._camera
        if hunt is None or camera is None:
            return
        if self._settling:
            self._watch_for_stillness(frame)
            return
        if reading is None or not self._integrator.last_image_was_whole:
            return
        if self._hunt_opening:
            self._hunt_opening = False
            if reading <= 0.0:  # nothing above the grain to climb
                # Nothing to climb yet. Get roughly there and read again.
                self._open_with_autofocus(camera)
                return

        move = hunt.step(reading)
        if move is None:
            self._finish_hunt(hunt)
            return
        before = self._frame_reading(frame)
        try:
            camera.drive_focus(move)
        except CameraError as exc:
            self._cancel_hunt("")
            self.failed.emit(f"Focus hunt stopped: {exc}")
            return
        self._settle(before)
        probes = hunt.probes
        self.status.emit(
            f"Hunting focus: {probes} probe{'' if probes == 1 else 's'}, "
            f"best {hunt.best:.0f}, steps of {hunt.step_size}"
        )

    def _watch_for_stillness(self, frame: LiveViewFrame) -> None:
        """Wait for the move to arrive and the picture to stop, then stack.

        Each frame is read on its own -- not through the integrator, which is
        still holding frames from before the move -- and two frames that agree
        mean the lens has arrived and stopped.

        Two frames agreeing is not enough on its own, though, and the reason is
        worth keeping: for the first frames after a move the picture has not
        started changing yet, because live view runs behind the lens. Those
        frames agree with each other perfectly while showing exactly the focus
        position the hunt has just left. So the first move of a hunt waits for
        the picture to *change*, and how many frames that took is the depth of
        the pipeline; every move after it waits at least that long before
        stillness is allowed to mean anything.
        """
        self._settle_seen += 1
        previous, self._settle_last = self._settle_last, self._frame_reading(frame)

        # Two frames in a row showing the move, not one: a single frame can
        # read a quarter high on grain alone, and a pipeline measured short is
        # worse than one not measured at all. Every move is measured, not just
        # the first, and the longest wins: a hunt that begins thoroughly
        # defocused has nothing but noise to measure against, and the answer it
        # gets there must not be allowed to stand once there is a real picture
        # to measure with.
        moving = self._differs_by(self._settle_before, _MOVED)
        if moving and self._settle_moving and not self._settle_changed:
            # The frame the move first showed up in, and only that one: it
            # stays different from here on, and counting those later frames
            # would push the measurement out to wherever the settle ended.
            self._settle_changed = True
            if self._settle_seen - 1 <= _LAG_LIMIT:
                self._pipeline_lag = max(self._pipeline_lag or 0, self._settle_seen - 1)
        self._settle_moving = moving

        # Two frames after the move has shown up, so there are two frames of
        # the new focus position to compare with each other.
        floor = max(_SETTLE_FRAMES, (self._pipeline_lag or 0) + 2)
        # With nothing to compare against there is no move to wait for: that
        # is the settle at the start of a hunt, where the lens has not been
        # sent anywhere and only stillness is wanted.
        arrived = (
            self._settle_changed
            or self._pipeline_lag is not None
            or self._settle_before is None
        )
        settled = (
            self._settle_seen >= floor
            and arrived
            and not self._differs_by(previous, _STILL)
        )
        if settled or self._settle_seen >= _SETTLE_LIMIT:
            # Everything from here belongs to this focus position.
            self._settling = False
            self._integrator.reset()

    def _differs_by(self, against: "float | None", fraction: float) -> bool:
        """Whether the newest frame reads meaningfully differently from *against*.

        Meaningfully: measured against the best this hunt has seen as well as
        against the two readings themselves, so that grain on a reading of
        nearly nothing is not mistaken for the picture changing.
        """
        now = self._settle_last
        if against is None or now is None:
            return False
        best = self._hunt.best if self._hunt is not None else 0.0
        # The grain scale matters before the hunt has a reading of its own to
        # judge by: a picture with nothing in it wanders about near zero, and
        # without a floor those wanderings read as the picture never being
        # still, so the settling would run to its limit every time.
        scale = max(now, against, _READING_FLOOR * best, self._grain_scale, 1e-9)
        return abs(now - against) > fraction * scale

    def _frame_reading(self, frame: LiveViewFrame) -> "float | None":
        """One frame's sharpness on its own, for judging whether it has moved."""
        image = QImage.fromData(frame.jpeg, "JPG")
        if image.isNull():
            return None
        area = self._sharpness.area
        self._grain_scale = grain_reading(image, area, self._noise_variance)
        return measure(image, area, self._noise_variance)

    def _open_with_autofocus(self, camera: NikonCamera) -> None:
        """Let the camera get roughly there, when there is no hill to climb.

        A thoroughly defocused frame reads zero -- rightly, it has no detail
        above its own grain -- and zero is nothing to climb. The camera's own
        autofocus uses its own big box, which is exactly why it is the opening
        move and not the whole job.
        """
        self.status.emit("Nothing to measure yet - letting the camera get close first")
        try:
            camera.autofocus()
        except CameraError as exc:
            self._cancel_hunt("")
            self.failed.emit(str(exc))
            return
        self._settle()

    def _finish_hunt(self, hunt: Walk) -> None:
        self._hunt = None
        self._settling = False
        self.huntChanged.emit(False)
        self.status.emit(
            {
                "found": (
                    f"Focus found: {hunt.best:.0f} after {hunt.probes} probes, "
                    f"{hunt.best_position:+d} steps from where it started"
                ),
                "nothing": (
                    "Nothing in the measured area to focus on - put it on "
                    "something with detail in it, or magnify further"
                ),
                "exhausted": (
                    f"Gave up after {hunt.probes} probes; best was {hunt.best:.0f}"
                ),
                "lost": (
                    f"Walked back {hunt.probes} probes without finding the "
                    f"{hunt.best:.0f} it saw again -- the picture may have moved "
                    f"under it, or the lens may have more play than it can walk "
                    f"through"
                ),
                "limit": "Ran as far as it is allowed to without finding focus",
            }.get(hunt.outcome, "Focus hunt finished")
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
            self._forget_sharpness()
            self.status.emit(f"{name} set")
        self.refresh_settings()

    # -- capture -----------------------------------------------------------

    @Slot(bool, bool)
    def capture(self, autofocus: bool, download: bool) -> None:
        camera = self._require()
        if camera is None:
            return
        # The wait is the camera's, and nothing arrives while it runs, so say
        # what is being waited for rather than let the window look stuck.
        self.status.emit(self._shutter_message(camera))
        try:
            handles = camera.capture(autofocus=autofocus)
        except CameraError as exc:
            self.failed.emit(str(exc))
            return
        if not handles:
            self.status.emit("Shutter fired (camera reported no new file)")
            self.refresh_settings()
            return

        # A picture the camera is holding in SDRAM rather than writing to
        # the card is gone the moment the next one is taken, so there is no
        # such thing as choosing not to download it.
        download = download or not camera.save_to_card

        # One name for the whole release of the shutter, so a RAW and a JPEG
        # of the same picture stay a pair. Claimed once the shot is known to
        # exist, so a capture that produced nothing does not eat a number.
        naming = download and self._namer.enabled
        stem = self._namer.claim(self._save_dir) if naming else ""

        names = []
        for handle in handles:
            if not download:
                names.append(self._filename(camera, handle))
                continue
            if self._download(camera, handle, stem) and not camera.save_to_card:
                camera.release_from_sdram(handle)
        if names:
            self.status.emit("Captured " + ", ".join(names))
        if naming:
            self.nextNumber.emit(self._namer.number)
        self.refresh_settings()

    @staticmethod
    def _shutter_message(camera: NikonCamera) -> str:
        """Say what the camera is about to spend the wait on.

        A long exposure is the case that needs saying: with noise reduction
        the body is busy for twice the shutter speed, taking a dark frame it
        subtracts from the picture, and until that is over there is no picture
        to fetch. Silence for half a minute reads as a program that has hung.
        """
        seconds = camera.shot_seconds()
        delay = camera.shutter_delay
        if seconds - delay >= 2:
            return f"Shooting -- about {seconds:.0f}s, the camera's own time..."
        if delay:
            return f"Mirror up, shutter in {delay}s..."
        return "Releasing shutter..."

    def _filename(self, camera: NikonCamera, handle: int) -> str:
        try:
            return camera.session.object_info(handle).filename
        except (MtpError, WpdCommandError):
            return f"image {handle:#010x}"

    def _download(self, camera: NikonCamera, handle: int, stem: str = "") -> bool:
        """Fetch one captured picture and write it out. False if it failed.

        `stem` is the name this shot was given by the sequence; empty means
        the sequence is off and the camera's own name is kept. Either way the
        camera's extension is: it is what says whether the file is a NEF.
        """
        self.status.emit("Downloading...")
        try:
            filename, data = camera.download(handle)
        except (MtpError, WpdCommandError, CameraError) as exc:
            self.failed.emit(f"Could not download the picture: {exc}")
            return False
        self._save_dir.mkdir(parents=True, exist_ok=True)
        if stem:
            path = self._save_dir / f"{stem}{Path(filename).suffix}"
        else:
            path = self._save_dir / filename
        path = unique(path)
        path.write_bytes(data)
        self.status.emit(f"Saved {path}")
        return True

    @Slot(str)
    def set_save_directory(self, path: str) -> None:
        self._save_dir = Path(path)

    @property
    def save_directory(self) -> Path:
        return self._save_dir

    @Slot(bool, str, int)
    def set_naming(self, enabled: bool, prefix: str, number: int) -> None:
        """Take the naming the window is showing, including an override.

        Nothing is emitted back: the window already has these values, and
        answering with them would fight whatever is being typed right now.
        The counter only reports itself after it has moved on its own, which
        is when a shot has used a number.
        """
        self._namer.configure(enabled, prefix, number)

    @property
    def naming(self) -> NameSequence:
        return self._namer

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
