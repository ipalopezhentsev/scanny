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
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import comtypes
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage

from ..camera.nikon import CameraError, LiveViewFrame, NikonCamera
from ..wpd.device import MtpError, WpdCommandError
from .activity import stamped
from .hunt import FineTune
from .integration import FrameIntegrator
from .naming import NameSequence, unique
from .pixels import green
from .regions import OBJECTIVES, Calibration, Look, Region, View, cut_out
from .sharpness import (
    QUANTISATION,
    SharpnessMeter,
    format_reading,
    grain_reading,
    measure,
    variance_between,
)

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

#: How many times live view may be started again by itself within how many
#: seconds, when the body ends it. The body does end it: a D750 turns live
#: view off after its own monitor-off delay for live view -- custom setting
#: c4, ten minutes out of the box -- whoever is driving it, and a calibration
#: can take longer than that. Starting it again is harmless
#: and puts back everything the readings depend on; but a body that ends it
#: again straight away is refusing for some other reason, and saying so beats
#: fighting it.
_MOST_RESTARTS = 3
_RESTART_WINDOW = 120.0

#: How long to wait before each attempt at starting live view again. A body
#: that has just turned it off can refuse to turn it straight back on while
#: it is still putting the mirror down, so the first attempt is immediate and
#: the rest are patient.
_RESTART_WAITS = (0.0, 1.0, 2.0, 4.0)

#: How long to keep trying to put the zoom and the focus point back after a
#: restart. The body answers the first commands after one with busy.
_VIEW_BACK_PATIENCE = 4.0

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

#: How much more than the grain two frames have to differ by before what is
#: between them counts as the picture having changed -- which is how the
#: compromise tells the lens is still arriving after a move.
_STOP_MOVED = 2.5

#: How many new frames to wait for before believing that what is on screen is
#: where the lens now is. Frames rather than a length of time: the body draws
#: forty-four a second at full frame and sixteen magnified, and live view runs
#: a few of them behind the lens either way.
_STOP_FRESH = 4
_STOP_PATIENCE = 0.8  # seconds, in case the body has stopped drawing at all

#: Magnification of each live-view zoom level, measured on a D750.
_MEASURED_MAGNIFICATION = {
    0: 1.0, 2: 2.35, 3: 3.13, 4: 4.7, 5: 6.27, 6: 9.4, 7: 18.8,
}

#: How many redrawn frames to wait for after panning the magnified view to the
#: next region. The camera pans by moving its focus point, and live view is a
#: few frames behind that as it is behind everything else; a reading taken off
#: a frame that still shows the last region is a reading of the wrong subject,
#: which is worse than no reading at all because it looks like one.
#:
#: The same four frames the stop-finder waits for, and it costs the same:
#: magnified, the body draws sixteen a second, so a pan is a quarter of a
#: second.
_AIM_FRESH = 4

#: How long to wait for a region's view to come back and fill a stack before
#: giving up on it: a fixed allowance for the body to answer, and so much a
#: frame on top. Generous, because the body draws sixteen a second magnified
#: and a change of zoom level can take it a moment longer than a pan.
_VIEW_PATIENCE = 2.0
_FRAME_TIME = 0.1


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
    #: Whether a calibration of the focus regions is running, and what it made
    #: of them: a :class:`scanny.ui.regions.CalibrationReport`, or None when
    #: there is nothing to report -- the regions changed, or one has begun.
    calibrationChanged = Signal(bool)
    calibrationReady = Signal(object)  # CalibrationReport | None
    #: Which region is being fine tuned (-1 for none, or for all of them at
    #: once), and a line saying how far the calibration has got.
    calibrationProgress = Signal(int, str)
    #: One probe of the compromise, for the chart of every region: which
    #: regions by number, what the search was doing, each one's share of its
    #: best, and the number made of them.
    calibrationProbe = Signal(object, str, object, float)
    #: A line for the activity log: what is going on in more detail than the
    #: status bar holds, and everything that went wrong, with the camera's
    #: own words for it. See scanny.ui.activity.
    log = Signal(str)
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
        # When live view was last started again after the body ended it.
        self._restarts: "deque[float]" = deque()
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
        self._hunt: "FineTune | None" = None
        self._settling = False
        self._settle_seen = 0
        self._settle_last: "float | None" = None
        self._settle_before: "float | None" = None
        self._settle_changed = False
        self._settle_moving = False
        # The rectangles someone drew to be kept in focus together, and the
        # calibration finding the one focus position that serves them all.
        self._regions: "list[Region]" = []
        self._calibration: "Calibration | None" = None
        # The user's own sharpness meter, put aside while a calibration runs
        # one of its own over each region in turn, and handed back after.
        self._calibration_meter: "SharpnessMeter | None" = None
        # The zoom level and focus point to put back when a calibration ends,
        # since it pans and magnifies the view all over the frame.
        self._view_restore: "tuple[int, int, int] | None" = None
        # Everything said while a calibration runs -- status, failures and
        # the log -- as the activity log shows it, to go into its report.
        self._calibration_said: "list[str] | None" = None
        self.status.connect(lambda text: self._hear(text, "status"))
        self.failed.connect(lambda text: self._hear(text, "error"))
        self.log.connect(lambda text: self._hear(text, ""))
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
        self._cancel_hunt("")
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
        if self._calibration is not None and self._calibration.phase == "compromise":
            # That part of a calibration reads its own frames, one region's
            # view at a time; see _advance_compromise.
            self._advance_compromise(camera)
            return
        try:
            frame = camera.live_view_frame()
        except (CameraError, MtpError, WpdCommandError):
            self._grab_errors += 1
            # The odd dropped frame is normal while the camera adjusts; a run
            # of them means live view really has ended, because someone pressed
            # a button on the body or the mirror dropped.
            if self._grab_errors >= _MAX_GRAB_ERRORS:
                if self._recover_live_view(
                    camera, f"{_MAX_GRAB_ERRORS} live-view frames failed in a row"
                ):
                    return
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
            self._advance_hunt(frame, reading, image)

    def _recover_live_view(self, camera: NikonCamera, why: object = None) -> bool:
        """Start live view again after the body ended it, and put the view back.

        What the body ends is live view, not anything that depends on the
        lens: focus stays where it was, so a fine tune or a calibration
        running across the gap can carry on across it -- once the
        view is what it was, the zoom level and the focus point that the
        magnified crop is centred on, and once the picture has been let settle
        again, so that no reading is taken off frames from before the gap.

        *why* is what gave it away, for the log. Frames stopping is one sign;
        the body refusing to move the focus point or drive focus, "not in live
        view", is another, and the one a calibration between two probes meets
        first -- missing that one is what used to end calibrations at the
        body's ten minutes.

        The restart is tried more than once, with a wait between, because a
        body that has only just turned live view off can refuse to turn it
        back on while the mirror is still coming down. False when it will not
        come back, or has been brought back too often lately to be believed:
        then whatever was running stops, and says why.
        """
        now = time.monotonic()
        while self._restarts and now - self._restarts[0] > _RESTART_WINDOW:
            self._restarts.popleft()
        if len(self._restarts) >= _MOST_RESTARTS:
            self._log(
                f"Live view went off again ({why}), the {_MOST_RESTARTS + 1}th time "
                f"in {_RESTART_WINDOW:.0f} s: not starting it again"
            )
            return False
        self._restarts.append(now)
        self._log(f"Live view seems to have gone off ({why}); starting it again")
        frame = self._last_frame
        level = self._zoom_level
        trouble: "Exception | None" = None
        for attempt, wait in enumerate(_RESTART_WAITS, start=1):
            if wait:
                time.sleep(wait)
            try:
                camera.restart_live_view()
            except (CameraError, MtpError, WpdCommandError) as exc:
                trouble = exc
                self._log(f"  attempt {attempt}: the camera refused: {exc}")
                continue
            trouble = None
            self._log(f"  attempt {attempt}: live view is back")
            break
        if trouble is not None:
            self.failed.emit(f"Live view went off and could not be started again: {trouble}")
            return False
        if not self._put_view_back(camera, frame, level):
            self._log("  the view could not be put back exactly; carrying on")
        self._grab_errors = 0
        self._last_digest = None
        self._integrator.reset()
        if self._hunt is not None:
            self._settle()
        self.status.emit(
            "The camera turned live view off by itself -- its own monitor-off "
            "delay for live view, custom setting c4 on a D750 -- so it was "
            "started again and the view put back"
        )
        return True

    def _put_view_back(
        self, camera: NikonCamera, frame: "LiveViewFrame | None", level: int
    ) -> bool:
        """The focus point and the zoom level as they were, busy or not."""
        deadline = time.monotonic() + _VIEW_BACK_PATIENCE
        while True:
            try:
                if frame is not None:
                    # The point first: the body centres a magnified view on it.
                    camera.set_af_area(frame.af_x, frame.af_y)
                if level:
                    camera.set_zoom_level(level)
                return True
            except (CameraError, MtpError, WpdCommandError) as exc:
                if time.monotonic() >= deadline:
                    self._log(f"  putting the view back failed: {exc}")
                    try:
                        self._zoom_level = camera.zoom_level()
                    except (CameraError, MtpError, WpdCommandError):
                        pass
                    self.zoomChanged.emit(self._zoom_level)
                    return False
                time.sleep(0.2)

    def _drive_through(self, camera: NikonCamera, steps: int) -> bool:
        """Drive focus, starting live view again first if it has gone off.

        False, having said why, when it cannot be done either way.
        """
        try:
            camera.drive_focus(steps)
            return True
        except (CameraError, MtpError, WpdCommandError) as exc:
            if not self._recover_live_view(camera, exc):
                self.failed.emit(f"Focus could not be driven: {exc}")
                return False
        try:
            camera.drive_focus(steps)
            return True
        except (CameraError, MtpError, WpdCommandError) as exc:
            self.failed.emit(f"Focus could not be driven: {exc}")
            return False

    def _log(self, text: str) -> None:
        self.log.emit(text)

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
        if self._calibration_meter is not None:
            # A calibration has its own meter in place; this is about the one
            # it put aside, and only a real change to that one stops it.
            if bool(enabled) == self._calibration_meter.enabled:
                return
            self._end_calibration(
                "Calibration stopped - sharpness measuring was switched",
                stopped=True,
            )
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
        """Measure only this part of the frame, or all of it when None.

        The area arrives as fractions of the **whole frame** -- a place on the
        sensor -- which is what makes it usable past the camera's strongest
        magnification: there is no more zooming to be had there, but a
        rectangle drawn round what is already on screen can be as small as the
        subject is, and it stays on that subject when the view magnifies or
        pans instead of jumping to whatever the same fractions of the new crop
        happen to cover. See :meth:`scanny.ui.sharpness.SharpnessMeter.set_area`.
        """
        if self._calibration_meter is not None:
            wanted = tuple(float(v) for v in area) if area is not None else None
            if wanted == self._calibration_meter.area:
                return
            self._end_calibration(
                "Calibration stopped - the measured area moved", stopped=True
            )
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
        """Autofocus over the measured area, then better it one increment at a time."""
        # A calibration runs fine tunes of its own, and between them walks the
        # lens itself; one started by hand in the middle of that takes the
        # focus from it, like anything else that touches the focus does.
        self._end_calibration(
            "Calibration stopped - fine tuning took over", stopped=True
        )
        step = abs(int(step)) or 1
        self._start_hunt(
            FineTune(step),
            f"Fine tuning focus: magnified onto the measured area, autofocus "
            f"first, then steps of {step}...",
        )

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
        # Magnified before anything is read, and never again after: every
        # reading a search makes has to be of the same picture as the last.
        before = self._magnify_on_measured_area(camera)
        self._hunt = hunt
        self._pipeline_lag = None
        # A fresh line for the hunt, so what is plotted is the hunt itself.
        self._sharpness.reset()
        self.sharpnessChanged.emit(0.0, 0.0)
        self._settle(before)
        self.huntChanged.emit(True)
        self.status.emit(announcement)

    def _magnify_on_measured_area(self, camera: NikonCamera) -> "float | None":
        """Magnify as far as the body will go and still show the measured area.

        This is the single largest thing that can be done for the quality of
        the answer, and it costs one command. **A drive step moves the picture
        far more when the view is magnified**, so the same focus error that is
        lost in the grain at full frame is obvious at 18.8x -- which is the
        difference between a search that can tell one step from the next and
        one reading its own noise.

        As far as it will go *and still show the area*, rather than simply as
        far as it will go. Magnifying past the area would leave the reading
        taken over the part of it that happened to stay on screen, which is a
        different question from the one the rectangle was drawn to ask, and
        one nobody chose. With no area marked out there is nothing chosen to
        magnify onto and the view is left alone -- going to 18.8x there would
        silently replace "the whole frame" with a twentieth of it.

        Answers with what the picture read beforehand, which is what the
        settle that follows waits to see change. Magnifying rewrites the
        picture completely, and the frames still in flight from before it
        agree with each other perfectly.
        """
        area = self._sharpness.area
        frame = self._last_frame
        if area is None or frame is None:
            return None
        x, y, w, h = area
        # A level's magnification is the same in both axes, so the area fits
        # whole while the magnification is no more than one over its long side.
        level = self._level_for_magnification(1.0 / max(w, h, 1e-6))
        if level == self._zoom_level and _inside(area, frame.crop_normalised):
            return None
        before = self._frame_reading(frame)
        try:
            # The body centres its magnified view on the focus point, so the
            # point goes first: it is what decides where the magnified view
            # lands. It goes even when the level is already right, because
            # the view may be magnified onto somewhere else -- which is what
            # a calibration moving on to its next region finds.
            camera.set_af_area(*frame.to_af_coords_in_frame(x + w / 2, y + h / 2))
            if level != self._zoom_level:
                camera.set_zoom_level(level)
        except CameraError as exc:
            # Worth saying, not worth stopping for: the search still works at
            # whatever magnification the view is already at.
            self.failed.emit(str(exc))
            return None
        self._zoom_level = level
        self.zoomChanged.emit(level)
        return before

    @Slot()
    def cancel_hunt(self) -> None:
        self._cancel_hunt("Focus hunt stopped")

    def _cancel_hunt(self, why: str = "") -> None:
        """Stop a hunt, if one is running. Silent when nothing is happening.

        A calibration goes with it. Every caller of this is something that
        changed what the camera is showing or where the lens is -- the view
        magnified, the exposure altered, focus taken by hand -- and after any
        of those the readings are not comparable with the ones before; a
        calibration's peaks, besides, are only worth anything while the lens
        is walked from where they left it.
        """
        self._end_calibration(
            why.replace("Focus hunt stopped", "Calibration stopped"), stopped=True
        )
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
        self,
        frame: LiveViewFrame,
        reading: "float | None",
        image: "QImage | None" = None,
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

        move = hunt.step(reading)
        if move is None:
            self._finish_hunt(hunt, frame, image)
            return
        if move.autofocus:
            self._autofocus_on_measured_area(camera, frame)
            return
        before = self._frame_reading(frame)
        if not self._drive_through(camera, move.steps):
            self._cancel_hunt("Focus hunt stopped: focus could not be driven")
            return
        if self._calibration is not None:
            self._calibration.drove(move.steps)
        self._settle(before)
        probes = hunt.probes
        # Which half of the walk it is in, because they take very different
        # lengths of time and a long search that says nothing but a rising
        # probe count reads as a hang.
        doing = "coming back to" if hunt.coming_back else "looking, best"
        self.status.emit(
            f"Fine tuning focus: {probes} probe{'' if probes == 1 else 's'} in "
            f"steps of {hunt.step_size}, {doing} {format_reading(hunt.best)}"
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
        # The measured area is a place on the sensor, so where it falls on
        # this picture depends on the crop this frame was sent with.
        area = self._sharpness.shown_in(frame)
        if area is None and self._sharpness.area is not None:
            return None
        self._grain_scale = grain_reading(image, area, self._noise_variance)
        return measure(image, area, self._noise_variance)

    def _autofocus_on_measured_area(
        self, camera: NikonCamera, frame: LiveViewFrame
    ) -> None:
        """Put the camera's own focus box on the measured area, and focus there.

        The opening move of a fine tune, and its reset between the two
        directions it searches. The box is 324 sensor pixels wide, which is
        why this is only ever the opening move -- but it is a rough answer
        arrived at in one go, and somewhere to improve on beats somewhere to
        start walking from.

        Aiming it is worth the trouble rather than focusing wherever the box
        was left: the measured area is what the reading is about, and a camera
        focusing on something else in the frame hands the search a starting
        point with no relation to what it is climbing. Moving the focus point
        is also what pans a magnified view, so this has the second effect of
        bringing the measured area on screen.
        """
        self.status.emit("Fine tuning focus: letting the camera get close first")
        self.focusStateChanged.emit("busy")
        # What the picture read as the camera was let loose on it, so the
        # settling below waits for the move to come through the pipeline
        # rather than believing the frames that still show where focus was.
        before = self._frame_reading(frame)
        if self._calibration is not None:
            self._calibration.autofocused()
        try:
            self._aim_at_measured_area(camera)
            focused = camera.autofocus()
        except (CameraError, MtpError, WpdCommandError) as exc:
            # Live view may have gone off under it; once it is back, the
            # autofocus is simply asked for again.
            try:
                if not self._recover_live_view(camera, exc):
                    raise
                self._aim_at_measured_area(camera)
                focused = camera.autofocus()
            except (CameraError, MtpError, WpdCommandError) as again:
                self.focusStateChanged.emit("idle")
                self._cancel_hunt("")
                self.failed.emit(f"Autofocus failed: {again}")
                return
        self.focusStateChanged.emit("focused" if focused else "idle")
        self._settle(before)

    def _aim_at_measured_area(self, camera: NikonCamera) -> None:
        """Move the focus box to the middle of the measured area, if there is one."""
        area = self._sharpness.area
        frame = self._last_frame
        if area is None or frame is None:
            return
        x, y, w, h = area
        camera.set_af_area(*frame.to_af_coords_in_frame(x + w / 2, y + h / 2))

    def _finish_hunt(
        self,
        hunt: FineTune,
        frame: "LiveViewFrame | None" = None,
        image: "QImage | None" = None,
    ) -> None:
        """Say how the search ended, with the picture it ended on to hand.

        The picture matters to a calibration, which keeps it as what the
        region looked like at its best; see :meth:`_region_tuned`.
        """
        self._hunt = None
        self._settling = False
        self.huntChanged.emit(False)
        if self._calibration is not None and self._calibration.phase == "peaks":
            self._region_tuned(hunt, frame, image)
            return
        stopped_on = hunt.confirmed if hunt.confirmed is not None else hunt.best
        self.status.emit(
            {
                "found": (
                    f"Focus found: {format_reading(stopped_on)} against "
                    f"{format_reading(hunt.baseline)} "
                    f"from the camera's own autofocus, after {hunt.probes} probes"
                ),
                "nothing": (
                    "Nothing in the measured area to focus on - put it on "
                    "something with detail in it, or magnify further"
                ),
                "exhausted": (
                    f"Stopped at {format_reading(stopped_on)} after {hunt.probes} probes, "
                    f"the most it will take; the best it saw was "
                    f"{format_reading(hunt.best)}"
                ),
                "lost": (
                    f"Stopped at {format_reading(stopped_on)} against the "
                    f"{format_reading(hunt.best)} it "
                    f"saw: it was sent back and forth {hunt.turns} times, so the "
                    f"reading is too unsteady to walk by -- integrate more "
                    f"frames, or measure a patch with more detail in it"
                ),
                "restored": (
                    f"Nothing bettered the camera's own "
                    f"{format_reading(hunt.baseline)}, so "
                    f"focus was put back where its autofocus had it"
                ),
            }.get(hunt.outcome, "Fine tuning finished")
        )

    # -- focus regions: one focus position for several places --------------

    @Slot(object)
    def set_focus_regions(self, regions: object) -> None:
        """Take the rectangles on the sensor's frame to be kept in focus together.

        Fractions of the **whole frame**, not of what is on screen -- see
        :class:`scanny.ui.regions.Region`. Whatever a calibration found is
        thrown away: a region that moved, or one more region to serve, is a
        different compromise.
        """
        self._regions = [
            Region(*(float(part) for part in rect)) for rect in (regions or [])
        ]
        self._end_calibration(
            "Calibration stopped - the regions changed", stopped=True
        )
        self.calibrationReady.emit(None)

    @Slot(int, str, int, bool, object)
    def start_calibration(
        self,
        step: int,
        objective: str = "average",
        turn_back: int = 80,
        depths: bool = False,
        panel: object = None,
    ) -> None:
        """Find each region's best, then the one focus that does best by all.

        *step* is the one increment every walk is made in, the fine tunes and
        the compromise alike: the lens's minimum. *objective* is what "does
        best by all" means -- one of :data:`scanny.ui.regions.OBJECTIVES`.
        *turn_back* is the percentage of its best below which a walk of the
        search turns round; see :data:`scanny.ui.regions.TURN_BACK`. With
        *depths*, the walks also go far enough to place every region's peak,
        for the depths and the film's shape -- which costs walking the
        compromise itself does not need. See
        :mod:`scanny.ui.regions` for the two phases, and why the second one
        walks rather than drives.

        *panel* is the settings only the window knows, as ``(group, name,
        value)``, for the log; see :meth:`_log_settings`.
        """
        if objective not in OBJECTIVES:
            self.failed.emit(f"There is no compromise called {objective!r}.")
            return
        camera = self._require()
        if camera is None or self._calibration is not None:
            return
        if not camera.live_view_active:
            self.failed.emit("Start live view before calibrating.")
            return
        frame = self._last_frame
        if frame is None:
            self.failed.emit("Wait for a live-view frame before calibrating.")
            return
        if len(self._regions) < 2:
            self.failed.emit(
                "Draw at least two regions first: ctrl-drag on the picture "
                "round each place that has to be sharp."
            )
            return
        self._cancel_hunt("")
        self._calibration = Calibration(
            self._regions,
            step,
            objective=objective,
            turn_back=turn_back / 100.0,
            depths=depths,
        )
        self._view_restore = (self._zoom_level, frame.af_x, frame.af_y)
        # A meter of its own, pointed at each region in turn. The user's is
        # put aside rather than repointed, so that nothing about it -- on or
        # off, where its area is -- has to be remembered and put back.
        self._calibration_meter = self._sharpness
        self._sharpness = SharpnessMeter(enabled=True)
        self.calibrationReady.emit(None)
        self.calibrationChanged.emit(True)
        self._calibration_said = []
        self._log(
            f"Calibration started: {len(self._regions)} regions, aiming for "
            f"{OBJECTIVES[objective].lower()}, turning back below "
            f"{self._calibration.turn_back:.0%}, in steps of {step}"
            + (", measuring depths" if depths else "")
        )
        self._log_settings(camera, frame, panel)
        self._tune_region()

    @Slot()
    def cancel_calibration(self) -> None:
        self._end_calibration("Calibration stopped", stopped=True)

    def _log_settings(
        self, camera: NikonCamera, frame: LiveViewFrame, panel: object = None
    ) -> None:
        """Every setting the calibration runs under, one to a line, for its log.

        All of them, not the ones thought to matter: a report looked at weeks
        later is only comparable with another if what differed between them
        can be read off it, and which of them mattered is not known when it
        is written. The camera's are read from the body now rather than
        remembered, and what only the window knows -- the focus increments,
        the way the view is shown -- comes in *panel*. Nothing here may stop
        the calibration: a setting that cannot be read says so and the rest
        go on.
        """
        run = self._calibration
        said: "list[tuple[str, str, object]]" = []

        def put(group: str, name: str, value: object) -> None:
            said.append((group, name, value))

        if run is not None:
            put("Calibration", "regions", len(run.regions))
            for number, region in enumerate(run.regions, start=1):
                put(
                    "Calibration",
                    f"region {number}",
                    f"x {region.x:.4f}, y {region.y:.4f}, width {region.w:.4f}, "
                    f"height {region.h:.4f} of the frame",
                )
            put("Calibration", "aim for", OBJECTIVES[run.objective].lower())
            put("Calibration", "turn back below", f"{run.turn_back:.0%} of the best")
            put("Calibration", "walk in steps of", run.step)
            put("Calibration", "measure depths for levelling", _yes(run.measures_depths))

        put("Camera", "model", _ask(lambda: camera.model))
        put("Camera", "firmware", _ask(lambda: camera.firmware))
        put("Camera", "battery", _ask(lambda: f"{camera.battery_level()}%"))
        try:
            settings = camera.settings()
        except Exception as exc:  # noqa: BLE001 - a log line is not worth stopping for
            put("Exposure", "settings", f"could not be read ({exc})")
            settings = []
        for setting in settings:
            put(
                "Exposure",
                setting.name,
                setting.label + ("" if setting.writable else " (set on the body)"),
            )

        put("Live view", "exposure preview", _ask(lambda: _yes(camera.exposure_preview)))
        put("Live view", "frame", f"{frame.width}x{frame.height}")
        put("Live view", "sensor", f"{frame.image_width}x{frame.image_height}")
        magnification = self._zoom_magnification.get(self._zoom_level)
        put(
            "Live view",
            "zoom level",
            f"{self._zoom_level}"
            + (f" ({magnification:.2f}x)" if magnification is not None else ""),
        )
        put(
            "Live view",
            "view",
            f"{frame.crop_width}x{frame.crop_height} centred on "
            f"({frame.crop_center_x}, {frame.crop_center_y})",
        )
        put("Live view", "focus point", f"({frame.af_x}, {frame.af_y})")
        put(
            "Live view",
            "integrate",
            f"{self._integrator.frames} frames" if self._integrator.enabled else "off",
        )
        put("Live view", "skip repeated frames", _yes(self._deduplicate))
        put("Live view", "grain between frames", f"{self._noise_variance:.4f} (measured)")
        put(
            "Live view",
            "level",
            f"roll {frame.roll}, pitch {frame.pitch}"
            if frame.roll is not None or frame.pitch is not None
            else "no reading",
        )

        meter = self._calibration_meter or self._sharpness
        put("Sharpness", "measure sharpness", _yes(meter.enabled))
        area = meter.area
        put(
            "Sharpness",
            "measured area",
            "the whole picture"
            if area is None
            else "x {:.4f}, y {:.4f}, width {:.4f}, height {:.4f} of the frame".format(
                *area
            ),
        )

        for entry in panel or ():
            try:
                group, name, value = entry
            except (TypeError, ValueError):
                continue
            put(str(group), str(name), value)

        put("Capture", "write to the camera's card", _yes(self._save_to_card))
        put(
            "Capture",
            "mirror-up delay",
            f"{self._shutter_delay} s" if self._shutter_delay else "off",
        )
        put(
            "Capture",
            "number the files",
            f"yes, next {self._namer.name()}" if self._namer.enabled else "no",
        )
        put("Capture", "saving to", self._save_dir)

        self._log(f"Settings at the start, {len(said)} of them:")
        for group, name, value in said:
            self._log(f"Setting  {group} > {name}: {value}")

    def _tune_region(self) -> None:
        """Fine tune the next region on its own, to find what it reads at best.

        Exactly the search the Fine tune button runs -- magnified onto the
        region, autofocus aimed at it, a walk in the one increment -- with the
        region as its measured area. What it ends on is picked up again in
        :meth:`_region_tuned`.
        """
        run = self._calibration
        if run is None:
            return
        region = run.regions[run.index]
        number, total = run.index + 1, len(run.regions)
        self._sharpness.set_area(region.rect)
        self.calibrationProgress.emit(
            run.index, f"Region {number} of {total}: fine tuning it on its own"
        )
        self._log(f"Region {number} of {total}: fine tuning it on its own")
        self._start_hunt(
            FineTune(run.step),
            f"Calibrating region {number} of {total}: magnified onto it, "
            f"autofocus, then steps of {run.step}...",
        )
        if self._hunt is None:
            self._end_calibration(
                f"Calibration stopped: region {number} could not be fine tuned",
                stopped=True,
            )

    def _region_tuned(
        self,
        hunt: FineTune,
        frame: "LiveViewFrame | None",
        image: "QImage | None",
    ) -> None:
        """Keep one region's best, then go on to the next region or the compromise.

        The best is what the fine tune stood on at the end -- the reading it
        confirmed, and the picture that reading came off -- not the highest
        reading it passed on the way. That one is the luckiest of a noisy
        walk; this one is where the lens actually is, and the picture of the
        region there is what it looks like at its best.

        The view goes with it: the zoom level, the focus point, and the crop
        the body answered with. Every later reading of this region is taken
        through that same crop, because a reading through another one is not
        the same measurement.
        """
        run = self._calibration
        if run is None:
            return
        reading = hunt.confirmed if hunt.confirmed is not None else hunt.best
        region = run.regions[run.index]
        if frame is None or image is None:
            look, view = Look(0.0), None
        else:
            look = _look_at(region, frame, image, reading)
            view = View.of(frame, self._zoom_level)
        self._log(
            f"Region {run.index + 1}: best reading {format_reading(reading)} "
            f"({hunt.outcome}) "
            f"after {hunt.probes} probes, read at zoom level {self._zoom_level}"
        )
        if run.region_tuned(look, hunt.outcome, view, hunt.probes):
            self._tune_region()
            return
        if not run.begin_compromise():
            self._end_calibration()
            return
        self._log("Every region has its best; the search for the compromise begins")
        self.calibrationProgress.emit(
            -1, "Every region has its best: walking to the focus that serves all"
        )
        # A fresh trend line: what is plotted from here is the average.
        self.sharpnessChanged.emit(0.0, 0.0)
        self.status.emit(
            "Calibrating: every region has its best, now walking to the one "
            "focus position that does best by all of them..."
        )

    def _advance_compromise(self, camera: NikonCamera) -> None:
        """One probe of the compromise: read every region here, then step.

        One probe per call, and the call comes from the frame grab, so the
        timer keeps running between probes and a Stop pressed halfway through
        still lands. Inside a probe this blocks, because there is nothing else
        to be doing: the camera has to be panned to each region and a stack
        of its picture gathered before the next can be read.

        Live view going off in the middle of it -- which the body does after
        its own monitor-off delay, and which shows up here as the camera
        refusing to move the focus point or drive focus -- is not the end of
        the calibration. Live view is started again, and the probe is taken
        again, whole: focus has not moved. A move the search had already asked
        for when it went off is kept, and made first, so the search's idea of
        where it is stays true.
        """
        run = self._calibration
        if run is None or run.phase != "compromise":
            return
        try:
            if run.pending:
                camera.drive_focus(run.pending)
                run.drove(run.pending)
                run.pending = 0
                run.moved = True
            if run.moved:
                self._hold_still(camera)
            looks: "dict[int, Look]" = {}
            for index in run.order():
                look = self._read_region(camera, index)
                if look is None:
                    raise CameraError(
                        f"no picture of region {index + 1} the way it was measured"
                    )
                looks[index] = look
        except (CameraError, MtpError, WpdCommandError) as exc:
            self._log(f"Compromise probe {run.probes + 1} interrupted: {exc}")
            if self._recover_live_view(camera, exc):
                return  # the probe is taken again, whole, next time round
            self._end_calibration(f"Calibration stopped: {exc}", stopped=True)
            return
        doing = run.searching
        move = run.take(looks)
        if run.score is not None:
            self.calibrationProbe.emit(run.history_regions, doing, run.shares, run.score)
            self._log(
                f"Compromise probe {run.probes} ({doing}): "
                + ", ".join(
                    f"{number}: {share:.0%}"
                    for number, share in zip(run.history_regions, run.shares)
                )
                + f" -> {run.score:.0%}"
                + (f", then {move.steps:+d} steps" if move is not None else ", standing here")
            )
        if run.searching != doing and run.searching:
            self._log(f"Compromise: now {run.searching}")
        if move is None:
            self._end_calibration()
            return
        run.pending = move.steps
        try:
            camera.drive_focus(move.steps)
        except (CameraError, MtpError, WpdCommandError) as exc:
            self._log(f"Compromise move {move.steps:+d} refused: {exc}")
            if not self._recover_live_view(camera, exc):
                self._end_calibration(f"Calibration stopped: {exc}", stopped=True)
            return  # the move is still pending, and is made first next time
        run.drove(move.steps)
        run.pending = 0
        run.moved = True
        self.calibrationProgress.emit(-1, run.progress())
        # The trend line plots the average while the compromise is sought, as
        # a percentage of the regions' best, so the search can be watched the
        # way a fine tune can.
        if run.score is not None:
            self.sharpnessChanged.emit(100.0 * run.score, 100.0 * run.best)

    def _read_region(self, camera: NikonCamera, index: int) -> "Look | None":
        """Pan to a region's view, stack its picture, and read the region.

        Read exactly as its peak was: through the same crop, off a stack of
        as many frames as the integration is set to, with the grain taken off
        in the same proportion. Otherwise its fraction of that peak would be
        comparing two different measurements.
        """
        run = self._calibration
        view = run.view(index) if run is not None else None
        if run is None or view is None:
            return None
        frames = self._integrator.frames if self._integrator.enabled else 1
        frame, image = self._show_view(camera, view, frames)
        if frame is None or image is None:
            return None
        region = run.regions[index]
        shown = frame.area_normalised(region.rect)
        reading = (
            measure(image, shown, self._noise_variance / frames)
            if shown is not None
            else 0.0
        )
        return _look_at(region, frame, image, reading)

    def _show_view(
        self, camera: NikonCamera, view: View, frames: int
    ) -> "tuple[LiveViewFrame | None, QImage | None]":
        """Point the camera the way *view* says, and hand back a stack from there.

        The zoom level and the focus point are only sent when they differ
        from what the camera has, and the redrawn frames a pan needs are only
        waited for when something was sent -- the region read last at one
        probe is read first at the next, still on screen, for nothing.
        """
        moved = False
        if view.level != self._zoom_level:
            camera.set_zoom_level(view.level)
            self._zoom_level = view.level
            self.zoomChanged.emit(view.level)
            moved = True
        frame = self._last_frame
        if frame is None or (frame.af_x, frame.af_y) != view.af:
            camera.set_af_area(*view.af)
            moved = True
        return self._stack_through(camera, view, frames, _AIM_FRESH if moved else 0)

    def _stack_through(
        self, camera: NikonCamera, view: View, frames: int, skip: int
    ) -> "tuple[LiveViewFrame | None, QImage | None]":
        """Gather *frames* redrawn frames showing *view*, and average them.

        *skip* redrawn frames are let go first, for a view that has only just
        been asked for: live view is a few frames behind the camera, and the
        frames still in flight show wherever it was pointed before. Past
        those, a frame is only taken if its crop is the view's, so a stack is
        never a blend of two places.

        The averaging is a :class:`~scanny.ui.integration.FrameIntegrator`'s,
        the same arithmetic the peaks were read off. And the grain is
        measured on the way, from each frame against the one before it: the
        lens has stopped and so has the view, so what is between them is
        noise.
        """
        stacker = FrameIntegrator(frames > 1, max(frames, 2))
        deadline = time.monotonic() + _VIEW_PATIENCE + _FRAME_TIME * (skip + frames)
        skipped = 0
        previous: "np.ndarray | None" = None
        while time.monotonic() < deadline:
            try:
                latest = camera.live_view_frame()
            except (CameraError, MtpError, WpdCommandError):
                continue
            digest = hashlib.blake2b(latest.jpeg, digest_size=8).digest()
            if digest == self._last_digest:
                continue
            self._last_digest = digest
            self._last_frame = latest
            if skipped < skip:
                skipped += 1
                continue
            if not view.shows(latest):
                continue
            image = stacker.add(latest)
            single = stacker.last_frame
            if single is not None:
                pixels = green(single)[::2, ::2]
                if previous is not None:
                    variance = variance_between(previous, pixels)
                    if variance is not None:
                        self._noise_seen.append(variance)
                previous = pixels
            if image is not None and stacker.last_image_was_whole:
                self._note_magnification(latest)
                self.frameReady.emit(latest, image)
                return latest, image
        return None, None

    def _end_calibration(self, why: str = "", *, stopped: bool = False) -> None:
        """Stop calibrating, put everything back, and say what was found.

        Every way a calibration ends comes through here, the bad ones
        included: the user's own sharpness meter goes back, and so do the
        zoom and the focus point -- leaving someone at 18.8x on the last
        region is leaving them somewhere they did not ask to be. The lens is
        *not* put back. Standing on the compromise is the point of it.

        What was found is published even when it stopped part way: the
        peaks of the regions it got to are still worth looking at. It is
        published last, once the last line about it has been said, so that
        the report carries everything that was said while it ran.
        """
        run, self._calibration = self._calibration, None
        if run is None:
            return
        if self._hunt is not None:
            self._hunt = None
            self._settling = False
            self.huntChanged.emit(False)
        if self._calibration_meter is not None:
            self._sharpness, self._calibration_meter = self._calibration_meter, None
            self._sharpness.reset()
            self.sharpnessChanged.emit(0.0, 0.0)
        self._restore_view()
        report = run.report(stopped=stopped)
        message = why if stopped else (why or report.describe())
        if message:
            self.status.emit(message)
        self._log(
            f"Calibration {'stopped' if stopped else 'finished'}: "
            f"{why or report.describe()}"
            + (f". {report.cost()}" if report.cost() else "")
        )
        said, self._calibration_said = self._calibration_said or [], None
        self.calibrationReady.emit(replace(report, log=tuple(said)))
        self.calibrationProgress.emit(-1, "")
        self.calibrationChanged.emit(False)

    def _hear(self, text: str, kind: str) -> None:
        """Keep a line for the calibration's report, while one is running."""
        if self._calibration_said is not None and str(text).strip():
            self._calibration_said.append(stamped(text, kind))

    def _restore_view(self) -> None:
        """Put the zoom and the focus point back where the calibration found them."""
        remembered, self._view_restore = self._view_restore, None
        camera = self._camera
        if remembered is None or camera is None:
            return
        level, x, y = remembered
        try:
            camera.set_zoom_level(level)
            camera.set_af_area(x, y)
        except (CameraError, MtpError, WpdCommandError):
            return
        self._zoom_level = level
        self.zoomChanged.emit(level)

    def _hold_still(self, camera: NikonCamera) -> None:
        """Wait for the lens to arrive and the picture to stop changing.

        The frame-driven searches have :meth:`_watch_for_stillness` for this,
        and it cannot be used here: it judges stillness by the sharpness of
        the measured area, and the compromise has one per region and is about
        to pan across all of them.

        Same two parts as everywhere else in this file, and the order of them
        is the point: **first** wait for frames the body has redrawn since the
        drive, because live view runs behind the lens and the frames right
        after a move show the position it just left -- they agree with each
        other perfectly and mean nothing. Only then does two frames agreeing
        say the lens has stopped.
        """
        self._look_again(camera, _STOP_FRESH)
        grain = max(self._noise_variance, QUANTISATION)
        seen: "np.ndarray | None" = None
        still = 0
        for _ in range(_SETTLE_LIMIT):
            pixels = self._look(camera)
            if pixels is None:
                continue
            if seen is not None and np.array_equal(pixels, seen):
                continue
            change = variance_between(seen, pixels) if seen is not None else None
            seen = pixels
            if change is None:
                continue
            if change > _STOP_MOVED * grain:
                still = 0
                continue
            still += 1
            if still >= 2:
                return

    def _look(self, camera: NikonCamera) -> "np.ndarray | None":
        """One frame's pixels, decimated, for telling whether it moved."""
        try:
            frame = camera.live_view_frame()
        except (CameraError, MtpError, WpdCommandError):
            return None
        picture = QImage.fromData(frame.jpeg, "JPG")
        if picture.isNull():
            return None
        return green(picture)[::2, ::2]

    def _look_again(
        self, camera: NikonCamera, fresh: int = _STOP_FRESH
    ) -> "list[np.ndarray]":
        """Wait for *fresh* frames the body has actually redrawn, and keep them.

        Reads overshoot the draw rate on purpose, so most of them come back
        holding the frame that was already here; those say nothing about
        whether anything moved. Counting redrawn frames rather than waiting a
        length of time is what makes this right at the sixteen a second the
        body draws at when it is magnified as well as at forty-four.
        """
        deadline = time.monotonic() + _STOP_PATIENCE
        seen: "list[np.ndarray]" = []
        while len(seen) < fresh and time.monotonic() < deadline:
            pixels = self._look(camera)
            if pixels is None:
                continue
            if not seen or not np.array_equal(pixels, seen[-1]):
                seen.append(pixels)
        return seen

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
        # Releasing the shutter takes the mirror down and, if it is asked to
        # focus first, drives the lens. Neither a hunt nor a calibration
        # survives that, and finding out afterwards is worse than being told.
        self._cancel_hunt("Focus hunt stopped - the shutter was released")
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


def _look_at(
    region: Region, frame: LiveViewFrame, image: QImage, reading: float
) -> Look:
    """A reading of *region*, with the region cut out of the picture it came off.

    Cut from the picture exactly where the reading was taken: the region as
    it falls on this frame's crop, which is a place on the sensor turned into
    a place on the picture. A region off the picture altogether has no
    picture to keep.
    """
    shown = frame.area_normalised(region.rect)
    picture = cut_out(image, shown) if shown is not None else None
    return Look(float(reading), picture)


def _inside(
    area: "tuple[float, float, float, float]",
    crop: "tuple[float, float, float, float]",
) -> bool:
    """Whether the whole of *area* is inside *crop*, both fractions of the frame."""
    x, y, w, h = area
    left, top, width, height = crop
    slack = 1e-6
    return (
        x >= left - slack
        and y >= top - slack
        and x + w <= left + width + slack
        and y + h <= top + height + slack
    )


def _yes(on: bool) -> str:
    return "yes" if on else "no"


def _ask(question: "Callable[[], object]") -> str:
    """What the camera answers, or why it did not, for a line of the log."""
    try:
        return str(question())
    except Exception as exc:  # noqa: BLE001 - a log line is not worth stopping for
        return f"could not be read ({exc})"
