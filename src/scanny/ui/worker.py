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
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage

from ..camera.nikon import CameraError, LiveViewFrame, NikonCamera
from ..wpd.device import MtpError, WpdCommandError
from .depth import (
    LEVELS,
    QUANTISATION,
    Survey,
    Sweep,
    tile_sums,
    tiling_for,
)
from .hunt import FineTune
from .integration import FrameIntegrator
from .naming import NameSequence, unique
from .points import Point, PointSurvey, area_for, ordering
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

#: How the depth map drives the lens. It parks against the near stop before
#: every pass, because that is the one place a lens comes back to exactly, and
#: it is what makes one pass's step counts mean the same thing as another's.
#: **A body may keep saying yes to a lens that is already against its stop.**
#: `MfDrive` can answer `STEP_END` or `STEP_INSUFFICIENT`, and when it does
#: that is the end of it -- but a D750 driving a lens past its infinity stop
#: answers OK and moves nothing, over and over. A stop-finder that believes
#: the OK never finds a stop: it runs to whatever limit it is given, calls
#: that the travel, and hands the sweep a step twelve times too big. The
#: sweep then reaches infinity a tenth of the way through its stops and
#: spends the rest of them driving a lens that cannot move.
#:
#: So the picture is watched instead, which is the same answer
#: :mod:`scanny.ui.hunt` reaches about step counts: a chunk of travel that
#: does not change the picture at all is a chunk the optics did not make.
#: The refusal is still taken when it comes -- it is free and it is quicker.
#:
#: The chunk is how far it drives between looks. It wants to be **large**,
#: and that is not about saving round trips. The near end of the travel is
#: where an ordinary scene is at its most thoroughly defocused, so it is
#: exactly where the picture changes least per step -- which is the worst
#: possible place to be asking whether the picture changed. Measured against
#: a modelled lens, a chunk of 500 steps left the last real move reading 1.5
#: times the grain against 1.5 for no move at all, and a chunk of 1000 left
#: it reading 5.3 against 0.9. Twice the chunk is four times the margin.
_STOP_CHUNK = 1000

#: How much of the travel may go by with the picture not answering before
#: the far end of the useful range is taken to have been passed. Generous,
#: because a scene with something close and something far has a quiet
#: stretch between them, and cutting the sweep short there loses the far
#: half of it.
_STOP_QUIET = 6

#: How far a park drives at a time. Nothing is looked at while parking -- the
#: lens only has to end up against the stop, and over-driving into it is how
#: that is guaranteed -- so this is only about not asking the body for one
#: enormous move.
_PARK_CHUNK = 5000

#: How far the sweep's own backstop has to see the picture hold still before
#: it decides the lens has stopped moving. **In drive steps, not in stops**,
#: and that distinction is the whole of it.
#:
#: Counting stops was wrong and wrong in the worst way, because it goes wrong
#: in proportion to how good the rest of the machinery got. A sweep's step is
#: the range it was given divided by the stops asked for, so once the range
#: finder started handing it a narrow range the step became small -- a range
#: of 12000 steps over 400 stops is a step of 30 -- and five stops of that is
#: 150 steps of travel, which changes nothing anywhere. The backstop then
#: fired six stops into every pass, and the pass after it, narrowed to what
#: those six stops found, fired sooner still.
#:
#: A distance is scale-free, and this one is the same distance the range
#: finder is willing to sit through before it decides the far end has gone
#: by. Nothing inside the range it found can be quieter than that, because
#: quieter than that is how the range was defined.
_SWEEP_QUIET_STEPS = _STOP_QUIET * _STOP_CHUNK

#: How much more than the grain two frames have to differ by before what is
#: between them counts as the picture having changed.
#:
#: **The frame it is compared against only moves when the picture does.** A
#: reference that follows every chunk asks "did this chunk change anything",
#: which a chunk that moved the optics a little answers no to -- and three
#: little moves in a row then read as the lens having stopped when it has
#: travelled three chunks. Holding the reference asks the question that
#: matters instead: has the picture changed since the last place it
#: demonstrably changed? Small moves add up until it has, and a lens against
#: its stop never gets there.
_STOP_MOVED = 2.5

#: How many new frames to wait for before believing that what is on screen is
#: where the lens now is. Frames rather than a length of time: the body draws
#: forty-four a second at full frame and sixteen magnified, and live view runs
#: a few of them behind the lens either way.
_STOP_FRESH = 4
_STOP_PATIENCE = 0.8  # seconds, in case the body has stopped drawing at all

#: How far past the travel a park drives once the travel is known. One command
#: instead of a chunked hunt, and more certain than one: the lens ends against
#: the stop whether or not it says so.
_STOP_MARGIN = 2000

#: The most travel any of this will assume a lens has. A D750's kit zoom is
#: about 6000 steps end to end and a micro several times that; this is a
#: bound, not an estimate, and the only thing it costs when it is far too
#: large is the parking drives, which are quick.
_TRAVEL_LIMIT = 30000

#: A range shorter than this many of the lens's minimum increments is not
#: something a sweep can divide up.
_LEAST_TRAVEL = 8

#: Magnification of each live-view zoom level, measured on a D750.
_MEASURED_MAGNIFICATION = {
    0: 1.0, 2: 2.35, 3: 3.13, 4: 4.7, 5: 6.27, 6: 9.4, 7: 18.8,
}

#: How many redrawn frames to wait for after panning the magnified view to the
#: next point. The camera pans by moving its focus point, and live view is a
#: few frames behind that as it is behind everything else; a reading taken off
#: a frame that still shows the last point is a reading of the wrong subject,
#: which is worse than no reading at all because it looks like one.
#:
#: The same four frames the stop-finder waits for, and it costs the same:
#: magnified, the body draws sixteen a second, so a pan is a quarter of a
#: second and a stop with five points on it is a second and a quarter.
_AIM_FRESH = 4

#: How much more of the bracket a pass may take than it was asked for, when it
#: reaches its last stop with a point still getting sharper. Extending forward
#: is free of the play in the gearing -- the lens is already driving that way
#: -- so the only reason to bound it is that a point which never peaks is a
#: point that is not going to, and marching to the end of the travel to prove
#: it wastes a minute.
_MOST_EXTRA = 1.0


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
    #: Whether a depth-map sweep is running.
    depthChanged = Signal(bool)
    #: The map as it stands, or None when there is not one. Emitted after every
    #: sample as well as at the end, so the picture fills in while it is swept
    #: rather than appearing a minute later.
    depthMapReady = Signal(object)  # DepthMap | None
    #: What the sweep is doing, in one line that stays put: which stretch of
    #: the travel it decided to cover, in what step, and how far through it
    #: is. The status bar carries this too, but the status bar is where every
    #: other message lands as well, so the numbers that say whether the sweep
    #: was pointed at the right place scroll away as it runs. Shared by both
    #: the things a sweep can be filling in.
    sweeping = Signal(str)
    #: Whether a scan of the placed points is running, and what it made of
    #: them: a list of :class:`scanny.ui.points.Found`, or None for nothing
    #: measured yet. Emitted after every stop as well as at the end.
    pointScanChanged = Signal(bool)
    pointsFound = Signal(object)  # list[Found] | None
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
        self._hunt: "FineTune | None" = None
        self._settling = False
        self._settle_seen = 0
        self._settle_last: "float | None" = None
        self._settle_before: "float | None" = None
        self._settle_changed = False
        self._settle_moving = False
        # Mapping the depth of the scene by sweeping focus and reading every
        # part of the picture at each stop. The survey outlives the sweep, so
        # the map can be redrawn at another grid without driving anything.
        self._survey: "Survey | None" = None
        self._sweep: "Sweep | None" = None
        self._sweep_samples = 0
        self._sweep_pass = 0
        self._sweep_passes = 1
        self._sweep_detail = LEVELS - 1
        self._sweep_minimum = 1
        # How many steps this lens has between its stops, measured rather
        # than assumed, and the step the last pass used -- which is the
        # margin to put either side of what it found when asked to go again.
        self._sweep_travel = 0
        self._sweep_last_step = 0
        self._sweep_span = (0, 0)
        # Which of the two things a sweep can be filling in is running:
        # "depth" for the whole-frame grid, "points" for the few places
        # someone put on the picture, and empty for neither. Everything
        # about driving the lens is the same for both; all that differs is
        # what each settled picture is read into and what comes out at the
        # end, so they share one controller and branch twice.
        self._scan = ""
        self._points: "list[Point]" = []
        self._point_box = 0.12
        self._point_survey: "PointSurvey | None" = None
        # Whether the running point scan is the magnified kind: bracketed
        # around where autofocus landed, with the camera panned from point to
        # point at every stop because at that magnification no two of them are
        # on screen together. What it has to put back when it is done, and how
        # many extra stops it has already allowed itself.
        self._points_apart = False
        self._points_restore: "tuple[int, int, int] | None" = None
        self._points_extra = 0
        # Whether the body reported a stop rather than the picture having to
        # say so, and how much two frames of a still picture differ, which is
        # what the stop-finder judges a chunk of travel against.
        self._stops_reported = False
        self._stop_grain = QUANTISATION
        # The picture at the last stop that differed from the one before it,
        # how far has been driven since, and whether the picture has changed
        # at all this pass. Together they stop a sweep that has run out of
        # lens rather than let it count out the rest of its stops against a
        # lens that cannot move.
        self._sweep_pixels: "np.ndarray | None" = None
        self._sweep_silent = 0
        self._sweep_spoke = False
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
            elif self._sweep is not None:
                self._advance_sweep(frame, None)
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
        elif self._sweep is not None:
            self._advance_sweep(frame, image)

    def _note_noise(self) -> None:
        """Measure the grain from the frame just decoded and the one before it.

        Only worth doing while something is going to read it, and only from a
        pair of frames: how much grain there is cannot be told from inside a
        single frame, because at best focus the finest detail in the picture
        looks exactly like grain.
        """
        if not (self._sharpness.enabled or self._sweep is not None):
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
        """Measure only this part of the frame, or all of it when None.

        The area arrives as fractions of the **whole frame** -- a place on the
        sensor -- which is what makes it usable past the camera's strongest
        magnification: there is no more zooming to be had there, but a
        rectangle drawn round what is already on screen can be as small as the
        subject is, and it stays on that subject when the view magnifies or
        pans instead of jumping to whatever the same fractions of the new crop
        happen to cover. See :meth:`scanny.ui.sharpness.SharpnessMeter.set_area`.
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
        """Autofocus over the measured area, then better it one increment at a time."""
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
        one reading its own noise. It is the same reason the depth map
        magnifies before comparing two points that are close together.

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
        if level == self._zoom_level:
            return None
        before = self._frame_reading(frame)
        try:
            # The body centres its magnified view on the focus point, so the
            # point goes first: it is what decides where the magnified view
            # lands.
            camera.set_af_area(*frame.to_af_coords_in_frame(x + w / 2, y + h / 2))
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

        A depth-map sweep goes with it. Every caller of this is something that
        changed what the camera is showing or where the lens is -- the view
        magnified, the exposure altered, focus taken by hand -- and each of
        those invalidates a sweep for exactly the reason it invalidates a
        hunt: the readings after it are not comparable with the ones before.
        """
        self._cancel_sweep(why.replace("Focus hunt stopped", "Depth map stopped"))
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

        move = hunt.step(reading)
        if move is None:
            self._finish_hunt(hunt)
            return
        if move.autofocus:
            self._autofocus_on_measured_area(camera, frame)
            return
        before = self._frame_reading(frame)
        try:
            camera.drive_focus(move.steps)
        except CameraError as exc:
            self._cancel_hunt("")
            self.failed.emit(f"Focus hunt stopped: {exc}")
            return
        self._settle(before)
        probes = hunt.probes
        # Which half of the walk it is in, because they take very different
        # lengths of time and a long search that says nothing but a rising
        # probe count reads as a hang.
        doing = "coming back to" if hunt.coming_back else "looking, best"
        self.status.emit(
            f"Fine tuning focus: {probes} probe{'' if probes == 1 else 's'} in "
            f"steps of {hunt.step_size}, {doing} {hunt.best:.0f}"
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
        try:
            self._aim_at_measured_area(camera)
            focused = camera.autofocus()
        except CameraError as exc:
            self.focusStateChanged.emit("idle")
            self._cancel_hunt("")
            self.failed.emit(str(exc))
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

    def _finish_hunt(self, hunt: FineTune) -> None:
        self._hunt = None
        self._settling = False
        self.huntChanged.emit(False)
        stopped_on = hunt.confirmed if hunt.confirmed is not None else hunt.best
        self.status.emit(
            {
                "found": (
                    f"Focus found: {stopped_on:.0f} against {hunt.baseline:.0f} "
                    f"from the camera's own autofocus, after {hunt.probes} probes"
                ),
                "nothing": (
                    "Nothing in the measured area to focus on - put it on "
                    "something with detail in it, or magnify further"
                ),
                "exhausted": (
                    f"Stopped at {stopped_on:.0f} after {hunt.probes} probes, "
                    f"the most it will take; the best it saw was {hunt.best:.0f}"
                ),
                "lost": (
                    f"Stopped at {stopped_on:.0f} against the {hunt.best:.0f} it "
                    f"saw: it was sent back and forth {hunt.turns} times, so the "
                    f"reading is too unsteady to walk by -- integrate more "
                    f"frames, or measure a patch with more detail in it"
                ),
                "restored": (
                    f"Nothing bettered the camera's own {hunt.baseline:.0f}, so "
                    f"focus was put back where its autofocus had it"
                ),
            }.get(hunt.outcome, "Fine tuning finished")
        )

    # -- mapping the depth of the scene ------------------------------------

    @Slot(int, int, int, bool)
    def start_depth_map(
        self, samples: int, passes: int, minimum: int, narrowed: bool = False
    ) -> None:
        """Sweep focus across the travel and read every part of the picture.

        *samples* is how many stops each pass makes, *passes* how many times
        the travel is swept -- each one over the stretch the last one found
        something in, in a step that much finer -- and *minimum* the smallest
        step this lens answers to, which is where refining has to stop. With
        *narrowed*, the first pass covers only the stretch the last map found
        something in rather than the whole travel.

        Every pass parks against the near stop first and drives one way only.
        See :mod:`scanny.ui.depth` for why that is the whole basis of the
        positions meaning anything.
        """
        camera = self._require()
        if camera is None:
            return
        if not camera.live_view_active:
            self.failed.emit("Start live view before mapping depth.")
            return
        frame = self._last_frame
        if frame is None:
            self.failed.emit("Wait for a live-view frame before mapping depth.")
            return
        self._cancel_hunt("")
        self._sweep_samples = max(4, int(samples))
        self._sweep_passes = max(1, int(passes))
        self._sweep_minimum = max(1, int(minimum))
        self._sweep_pass = 0
        # Where the first pass should look, before the old survey is thrown
        # away: a second run over the stretch the first one found something
        # in is how a scene that lives in a corner of the travel is mapped
        # without spending most of the stops on empty air again.
        again = (
            self._survey.interesting(self._sweep_last_step or self._sweep_minimum)
            if narrowed and self._survey is not None and len(self._survey)
            else None
        )
        self._scan = "depth"
        self._survey = Survey(tiling_for(frame.width, frame.height))
        self.depthMapReady.emit(None)
        self.depthChanged.emit(True)
        self._sweep_over(camera, again)

    @Slot(object, float)
    def set_focus_points(self, points: object, box: float) -> None:
        """Take the places on the sensor's frame that are to be measured.

        Fractions of the **whole frame**, not of what is on screen, so they
        stay on the piece of the world someone pointed at when the camera
        magnifies or pans underneath them -- see :class:`scanny.ui.points.Point`.
        What was measured before is thrown away: a point that moved is a
        different question.
        """
        self._points = [Point(float(x), float(y)) for x, y in (points or [])]
        self._point_box = float(box)
        if self._scan == "points":
            self._cancel_sweep("Point scan stopped - the points moved")
        self._point_survey = None
        self.pointsFound.emit(None)

    @Slot(int, int, int, bool, bool, int)
    def start_point_scan(
        self,
        samples: int,
        passes: int,
        minimum: int,
        narrowed: bool = False,
        magnified: bool = False,
        around: int = 0,
    ) -> None:
        """Sweep focus once and read every placed point off the same frames.

        The same driving as the depth map, down to the parking and the
        bracket, and for the same reasons -- see :mod:`scanny.ui.depth`. What
        differs is only what each settled picture is read into: a handful of
        boxes someone chose rather than a grid over the whole frame, which is
        what makes the answers worth having. See :mod:`scanny.ui.points`.
        """
        camera = self._require()
        if camera is None:
            return
        if not camera.live_view_active:
            self.failed.emit("Start live view before measuring points.")
            return
        if len(self._points) < 2:
            self.failed.emit(
                "Put at least two points on the picture first: ctrl-click "
                "where you want them measured."
            )
            return
        self._cancel_hunt("")
        self._sweep_samples = max(4, int(samples))
        self._sweep_passes = max(1, int(passes))
        self._sweep_minimum = max(1, int(minimum))
        self._sweep_pass = 0
        if magnified:
            self._start_points_apart(camera, max(1, int(around)))
            return
        again = (
            self._point_survey.interesting(
                self._sweep_last_step or self._sweep_minimum
            )
            if narrowed and self._point_survey is not None
            and len(self._point_survey)
            else None
        )
        self._scan = "points"
        self._points_apart = False
        self._point_survey = PointSurvey(self._points)
        self.pointsFound.emit(None)
        self.pointScanChanged.emit(True)
        self._sweep_over(camera, again)

    # -- the same question at full magnification ----------------------------

    def _start_points_apart(self, camera: NikonCamera, around: int) -> None:
        """Measure the points magnified, one in view at a time.

        The whole-frame scan above reads every point off the same frame, and
        on an unmagnified frame that is the best there is: one pass driving
        one way puts every reading in one coordinate. What it cannot do is
        *resolve* anything small. A step of focus moves the picture in
        proportion to how much the view is magnified, so on a whole frame two
        things a hundred steps apart move the reading less than the grain
        does, and the honest answer -- which is the one the doubt gives -- is
        that they cannot be told apart.

        Magnifying fixes that and breaks something else: at 18.8x the screen
        shows a hundredth of the frame, so two points worth comparing are
        never on it together. What is kept, and what is given up:

        **Kept: one sweep, one coordinate.** The camera is panned from point
        to point at every stop and focus is not touched while it pans, so
        every reading taken at a stop still belongs to that one position, on
        one monotonic drive. Panning costs frames; it costs nothing in the
        coordinate.

        **Given up: the near stop as the datum.** This never parks. Parking
        and then sweeping the whole travel at a step fine enough to be worth
        magnifying for would be thousands of stops, so instead the camera's
        own autofocus is pointed at the first point and the sweep is a
        bracket around where that landed: back off *around* steps, then drive
        forward through twice that. Backing off first is not a detail -- it is
        what puts the play in the gearing behind the sweep instead of inside
        it, so that every stop of the pass that follows is honest travel. The
        positions that come out are counted from where the bracket began
        rather than from the near stop, and the gaps between them -- which are
        the answer -- are unaffected either way.

        Points whose focus lies past the far end of the bracket say so, and
        the pass keeps driving to reach them; see :meth:`_reach_further`. A
        point past the *near* end cannot be reached that way, because reaching
        back is a reversal, so that one is reported rather than chased.
        """
        frame = self._current_frame()
        if frame is None:
            return
        self._scan = "points"
        self._points_apart = True
        self._points_extra = 0
        self._sweep_passes = 1
        self._point_survey = PointSurvey(self._points)
        self.pointsFound.emit(None)
        self.pointScanChanged.emit(True)
        self._points_restore = (self._zoom_level, frame.af_x, frame.af_y)
        # The grain is a property of the view, and the view is about to change
        # out of all recognition: what a whole frame reads says nothing about
        # what a hundredth of one does.
        self._noise_seen.clear()
        self._last_pixels = None
        try:
            self._magnify_fully(camera)
            self._aim_at(camera, self._points[0])
            self.status.emit(
                "Autofocusing on point 1, to find the stretch of travel worth "
                "sweeping..."
            )
            self.focusStateChanged.emit("busy")
            locked = camera.autofocus()
            self.focusStateChanged.emit("focused" if locked else "idle")
            if not locked:
                self._finish_scan(
                    "Point scan stopped: autofocus could not find point 1, so "
                    "there is nothing to bracket around. Put point 1 on "
                    "something with an edge in it, or focus by hand and use "
                    "the whole-frame scan."
                )
                return
            # Behind where the sweep will start, so that the first stops of it
            # are the optics moving rather than the gearing taking up its play.
            camera.drive_focus(-around)
        except CameraError as exc:
            self._finish_scan(f"Point scan stopped: {exc}")
            return
        span = 2 * around
        step = max(self._sweep_minimum, span // self._sweep_samples)
        self._sweep_span = (0, span)
        self._sweep = Sweep(0, step, self._sweep_samples)
        self._settling = False
        self._sweep_pixels = None
        self._sweep_silent = 0
        self._sweep_spoke = False
        self.status.emit(
            f"Sweeping {span} steps around where autofocus landed, in "
            f"{self._sweep_samples} stops of {step}, panning to each of the "
            f"{len(self._points)} points at every one of them"
        )
        self._say_sweeping(0)

    def _magnify_fully(self, camera: NikonCamera) -> None:
        """Go to the strongest magnification the body has."""
        level = max(NikonCamera.ZOOM_LEVELS)
        camera.set_zoom_level(level)
        self._zoom_level = level
        self.zoomChanged.emit(level)

    def _restore_view(self) -> None:
        """Put the zoom and the focus point back where the scan found them.

        Everything that ends a magnified scan comes through here, including
        the ways it ends badly: leaving someone at 18.8x on the last point of
        a scan that failed is leaving them somewhere they did not ask to be
        and cannot easily get back from.
        """
        remembered, self._points_restore = self._points_restore, None
        self._points_apart = False
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

    def _aim_at(
        self, camera: NikonCamera, point: Point
    ) -> "tuple[LiveViewFrame | None, QImage | None]":
        """Pan the magnified view onto *point* and hand back what it shows.

        The camera centres its magnified view on the focus point, so moving
        the focus point is how the view is panned -- the same mechanism the
        arrow keys and the navigator use, and it touches nothing about focus.
        Near the edge of the frame the focus point stops before the crop does,
        which is why where the point actually landed is read back off the
        frame that comes out rather than assumed to be the middle of it.
        """
        frame = self._last_frame
        if frame is None:
            frame, _image = self._fresh(camera, 1)
            if frame is None:
                return None, None
        x, y = frame.to_af_coords_in_frame(point.x, point.y)
        if (x, y) == (frame.af_x, frame.af_y):
            return self._fresh(camera, 1)
        camera.set_af_area(x, y)
        return self._fresh(camera, _AIM_FRESH)

    def _fresh(
        self, camera: NikonCamera, fresh: int = _AIM_FRESH
    ) -> "tuple[LiveViewFrame | None, QImage | None]":
        """Wait for *fresh* frames the body has really redrawn; keep the last.

        Redrawn frames rather than a length of time, for the reason
        :meth:`_look_again` gives: reads overshoot the draw rate on purpose,
        and magnified the body draws sixteen a second rather than forty-four.

        The grain is measured here too, off the last two frames, because by
        then the view has stopped moving and two frames of a still picture
        differ by nothing but noise. It has to be measured on this view: the
        grain is what :func:`~scanny.ui.sharpness.measure` subtracts before it
        answers, and a figure carried over from an unmagnified frame is a
        figure about a different picture.
        """
        deadline = time.monotonic() + _STOP_PATIENCE
        seen = 0
        frame: "LiveViewFrame | None" = None
        image: "QImage | None" = None
        before: "np.ndarray | None" = None
        while seen < fresh and time.monotonic() < deadline:
            try:
                latest = camera.live_view_frame()
            except (CameraError, MtpError, WpdCommandError):
                continue
            digest = hashlib.blake2b(latest.jpeg, digest_size=8).digest()
            if digest == self._last_digest:
                continue
            decoded = QImage.fromData(latest.jpeg, "JPG")
            if decoded.isNull():
                continue
            self._last_digest = digest
            self._last_frame = latest
            before = green(image)[::2, ::2] if image is not None else None
            frame, image = latest, decoded
            seen += 1
        if frame is None or image is None:
            return None, None
        if before is not None:
            variance = variance_between(before, green(image)[::2, ::2])
            if variance is not None:
                self._noise_seen.append(variance)
        self._note_magnification(frame)
        self.frameReady.emit(frame, image)
        return frame, image

    def _advance_points_apart(self, camera: NikonCamera) -> None:
        """One stop of the magnified scan: read every point, then drive on.

        One stop per call, and the call comes from the frame grab, so the
        timer keeps running between stops and a Stop pressed halfway through
        still lands. Inside a stop this blocks, because there is nothing else
        to be doing: the camera has to be panned and its answer waited for
        before the next point can be read.
        """
        sweep, survey = self._sweep, self._point_survey
        if sweep is None or survey is None:
            return
        try:
            self._hold_still(camera)
            readings = self._read_points_apart(camera)
        except CameraError as exc:
            self._finish_scan(f"Point scan stopped: {exc}")
            return
        if readings is None:
            self._finish_scan(
                "Point scan stopped: live view stopped sending pictures"
            )
            return
        survey.add(sweep.position, readings, self._sweep_pass)
        self.pointsFound.emit(survey.found())
        self._say_sweeping(sweep.taken + 1)
        if sweep.taken + 1 >= sweep.samples:
            self._reach_further(sweep)
        move = sweep.took_one() if not sweep.done else None
        if move is not None:
            try:
                if not camera.drive_focus(move):
                    sweep.blocked()
            except CameraError as exc:
                self._finish_scan(f"Point scan stopped: {exc}")
                return
        if sweep.done:
            self._sweep_pass += 1
            self._finish_scan()

    def _read_points_apart(self, camera: NikonCamera) -> "list[float] | None":
        """Pan to each point in turn and read the box around it.

        Every one of these belongs to the focus position the lens is at now:
        panning moves the view and nothing else. What it costs is frames --
        four redrawn ones a point, so a quarter of a second each at the
        sixteen a second the body draws magnified.

        A point the camera cannot bring into view reads zero, which is what
        :func:`~scanny.ui.sharpness.measure` answers for anything it cannot
        tell from the grain, and what the peak finder treats as "nothing
        here" rather than as a low reading to be fitted.
        """
        survey = self._point_survey
        if survey is None:
            return None
        readings: "list[float]" = []
        for point in survey.points:
            frame, image = self._aim_at(camera, point)
            if frame is None or image is None:
                return None
            seen = point.seen_in(frame.crop_normalised)
            if seen is None:
                readings.append(0.0)
                continue
            aspect = image.width() / max(image.height(), 1)
            readings.append(
                measure(
                    image,
                    area_for(seen[0], seen[1], self._point_box, aspect),
                    self._noise_variance,
                )
            )
        return readings

    def _hold_still(self, camera: NikonCamera) -> None:
        """Wait for the lens to arrive and the picture to stop changing.

        The frame-driven sweep has :meth:`_watch_for_stillness` for this and
        cannot be used here, because that judges stillness by the sharpness of
        the whole displayed picture and this view is about to be panned across
        five different subjects.

        Same two parts as everywhere else in this file, and the order of them
        is the point: **first** wait for frames the body has redrawn since the
        drive, because live view runs behind the lens and the frames right
        after a move show the position it just left -- they agree with each
        other perfectly and mean nothing. Only then does two frames agreeing
        say the lens has stopped.
        """
        self._look_again(camera, _STOP_FRESH)
        grain = max(self._noise_variance, self._stop_grain, QUANTISATION)
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

    def _reach_further(self, sweep: Sweep) -> None:
        """Add stops when a point is still getting sharper at the last one.

        Only ever forward, and only so far. Forward because that is the way
        the lens is already driving, so the stops added are in the same
        coordinate as the ones before them; a reversal would not be. So far
        because a point that has not peaked by twice as many stops is not
        going to -- there is nothing in its box to peak -- and marching the
        rest of the travel to prove it costs a minute and answers nothing.

        Two things ask for more stops, and the second is the ordinary one. A
        point still getting sharper at the last stop obviously has its peak
        further on. But a point far outside the bracket does not read as
        rising at all: thoroughly defocused it reads nothing, flat, all the
        way across -- so having no answer for a point counts as a reason to
        keep driving too. It is a guess in that case, because a point nearer
        than the bracket reads exactly the same flat nothing and driving
        further goes away from it; it is a bounded guess, and the way to not
        need it is to put point 1 on the nearest of the subjects.
        """
        survey = self._point_survey
        if survey is None:
            return
        rising, _falling = survey.escaping()
        if not rising and all(one.known for one in survey.found()):
            return
        allowed = int(_MOST_EXTRA * self._sweep_samples)
        if self._points_extra >= allowed:
            return
        more = min(max(4, self._sweep_samples // 4), allowed - self._points_extra)
        self._points_extra += more
        sweep.keep_going(more)
        self.status.emit(
            f"A point has not shown its best yet at the end of the bracket, so "
            f"the sweep is carrying on for {more} more stops"
        )

    def _sweep_over(
        self, camera: NikonCamera, again: "tuple[int, int] | None"
    ) -> None:
        """Decide which stretch of travel to sweep, and start the first pass."""
        what = "Depth map" if self._scan == "depth" else "Point scan"
        try:
            # A second run over what the first one found is following it
            # immediately, on the same lens and the same scene, so the range
            # the first one found still stands and is not looked for again.
            span = again if again is not None else self._useful_range(camera)
        except CameraError as exc:
            self._finish_scan(f"{what} stopped: {exc}")
            return
        if span is None:
            self._finish_scan(
                "Driving focus from one end of its travel to the other changed "
                "nothing in the picture, so there is nothing here to measure. "
                "Check that live view is showing the scene and that the lens is "
                "on manual focus, or magnify onto something with detail in it."
            )
            return
        low, high = max(0, span[0]), span[1]
        if high - low < _LEAST_TRAVEL * self._sweep_minimum:
            self._finish_scan(
                f"Only {high - low} steps of the travel changed the picture at "
                f"all, which is not a range a sweep can divide up. The scene may "
                f"be flat enough that one focus position covers all of it."
            )
            return
        step = max(self._sweep_minimum, (high - low) // self._sweep_samples)
        # Whether the body owned up to a stop is worth saying: it is the one
        # thing about a lens this cannot find out for itself, and a body that
        # never does is what the whole picture-watching apparatus is for.
        told = "" if self._stops_reported else ", which it never admitted to reaching"
        self.status.emit(
            f"The picture answers focus between {low} and {high} steps from the "
            f"near stop{told}; sweeping that in {self._sweep_samples} stops of {step}"
        )
        self._sweep_span = (low, high)
        self._begin_pass(camera, low, step)

    @Slot()
    def cancel_depth_map(self) -> None:
        self._cancel_sweep("Depth map stopped")

    @Slot()
    def cancel_point_scan(self) -> None:
        self._cancel_sweep("Point scan stopped")

    @Slot(int)
    def set_depth_detail(self, level: int) -> None:
        """Redraw the map at a coarser or finer grid, without sweeping again.

        Free, and that is the point of keeping the survey rather than the map:
        a coarse zone's numbers are the sum of the fine zones inside it, so
        every grid was already measured by the pass that ran.
        """
        self._sweep_detail = max(0, int(level))
        survey = self._survey
        if survey is not None and len(survey):
            self.depthMapReady.emit(survey.map(self._sweep_detail))

    def _cancel_sweep(self, why: str = "") -> None:
        """Stop a sweep, if one is running. Silent when nothing is happening.

        What was read is kept. Half a map is still worth looking at, the
        detail control still works on it, and points that were placed before
        the sweep gave up keep the answers they got.
        """
        if self._sweep is None:
            return
        self._sweep = None
        self._settling = False
        kind, self._scan = self._scan, ""
        self._restore_view()
        if kind == "points":
            self.pointScanChanged.emit(False)
        else:
            self.depthChanged.emit(False)
        if why:
            self.status.emit(why)

    def _begin_pass(self, camera: NikonCamera, low: int, step: int) -> None:
        """Park, drive to where this pass starts, and begin sampling.

        The jump forward is never shorter than one step, even for the pass
        that starts at the stop. It is what takes the play in the gearing up
        in the direction the pass will drive, and doing it identically at the
        start of every pass is what makes the passes agree with each other.
        """
        try:
            self._park(camera)
            jump = max(int(low), int(step))
            camera.drive_focus(jump)
        except CameraError as exc:
            self._finish_scan(f"Stopped: {exc}")
            return
        self._sweep = Sweep(jump, step, self._sweep_samples)
        self._pipeline_lag = None
        self._sweep_pixels = None
        self._sweep_silent = 0
        self._sweep_spoke = False
        self._settle()
        what = "Depth map" if self._scan == "depth" else "Point scan"
        self.status.emit(
            f"{what} pass {self._sweep_pass + 1} of {self._sweep_passes}: "
            f"{self._sweep_samples} stops in steps of {step}..."
        )
        self._say_sweeping(0)

    def _park(self, camera: NikonCamera) -> None:
        """Drive the lens against its near stop, which is the datum.

        Blind, and that is the point: nothing is watched and no answer is
        wanted, so there is nothing here to get wrong. Driving further than
        the lens can go is what guarantees it arrives, whether or not the
        body admits to the stop -- and a body that does admit to it ends the
        parking early for nothing but speed.
        """
        self.status.emit("Parking focus against its near stop...")
        far = (self._sweep_travel or _TRAVEL_LIMIT) + _STOP_MARGIN
        driven = 0
        while driven < far:
            if not camera.drive_focus(-min(_PARK_CHUNK, far - driven)):
                self._stops_reported = True
                return
            driven += _PARK_CHUNK

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

    def _measure_grain(self, camera: NikonCamera) -> float:
        """How much two frames of a still picture differ, on this scene now.

        The stop-finder needs it before a sweep has measured anything, and it
        is the whole basis of the finder's one question: did that chunk of
        travel change the picture by more than the picture changes anyway?
        """
        seen = self._look_again(camera, fresh=8)
        pairs = [
            variance_between(earlier, later)
            for earlier, later in zip(seen, seen[1:])
        ]
        real = [value for value in pairs if value]
        return min(real) if real else QUANTISATION

    def _picture_moved(
        self, earlier: "np.ndarray | None", later: "np.ndarray | None"
    ) -> bool:
        """Whether what is between two frames is more than the grain on them.

        Not being able to tell counts as movement: a stop-finder that stops
        because it could not read a frame stops in the middle of the travel
        and calls it the end of it.
        """
        if earlier is None or later is None:
            return True
        variance = variance_between(earlier, later)
        if variance is None:
            return True
        return variance > _STOP_MOVED * max(self._stop_grain, QUANTISATION)

    def _useful_range(
        self, camera: NikonCamera
    ) -> "tuple[int, int] | None":
        """Which part of the travel the picture answers focus at all in.

        Not the mechanical travel. The mechanical travel is what the first
        attempt at this went looking for, and it is both hard to measure and
        not the thing wanted. Hard to measure, because a body will keep
        saying yes to a drive it is not making -- a D750 refuses at the near
        stop and accepts for ever past infinity -- so the refusal cannot be
        relied on. And not the thing wanted, because most of a macro lens's
        travel is the first few centimetres in front of it, where an
        ordinary scene is a uniform wash that no amount of focusing brings
        into anything. Stops spent there measure nothing.

        So this drives from the near stop to the far one in chunks and
        watches, and answers with the stretch between the first chunk that
        changed the picture and the last one that did. Focus moves outside
        that and the picture does not follow; inside it is the whole of what
        there is to map, and dividing *it* by the stops asked for is what
        makes a stop worth taking.

        Silence before the first change is not the end of anything -- it is
        the wash in front of the lens -- which is why the early finish waits
        for the picture to have spoken once.
        """
        self.status.emit("Finding the part of the travel the picture answers in...")
        self._stops_reported = False
        self._stop_grain = self._measure_grain(camera)
        self._park(camera)
        # Waiting for redrawn frames rather than taking the next one: parking
        # drove the whole travel and live view is still showing where the lens
        # was, so the frame on the wire belongs to the other end of the range.
        opening = self._look_again(camera)
        seen = opening[-1] if opening else None
        travelled = 0
        first: "int | None" = None
        last = 0
        quiet = 0
        while travelled < _TRAVEL_LIMIT:
            if not camera.drive_focus(_STOP_CHUNK):
                self._stops_reported = True
                break
            was, travelled = travelled, travelled + _STOP_CHUNK
            fresh = self._look_again(camera)
            now = fresh[-1] if fresh else None
            if not self._picture_moved(seen, now):
                # The reference stays where it last moved; see _STOP_MOVED.
                quiet += 1
                if first is not None and quiet >= _STOP_QUIET:
                    break
                continue
            if first is None:
                first = was
            last = travelled
            quiet = 0
            seen = now
        self._sweep_travel = travelled
        self._park(camera)
        if first is None:
            return None
        # A chunk of margin either side: the change was seen somewhere
        # within the chunk, not at the end of it.
        return max(0, first - _STOP_CHUNK), last + _STOP_CHUNK

    def _advance_sweep(self, frame: LiveViewFrame, image: "QImage | None") -> None:
        """Take one settled picture into the survey and drive to the next stop.

        The same two refusals the hunt makes, for the same two reasons: a
        frame caught while the lens is still moving belongs to no focus
        position, and a half-filled integration stack is a blend of two.
        """
        sweep, camera = self._sweep, self._camera
        if sweep is None or camera is None:
            return
        if self._points_apart:
            # Nothing below this applies: that scan reads its own frames, at
            # its own magnification, one point at a time. See
            # _start_points_apart.
            self._advance_points_apart(camera)
            return
        if self._settling:
            self._watch_for_stillness(frame)
            return
        if image is None or image.isNull():
            return
        if not self._integrator.last_image_was_whole:
            return
        frames = self._integrator.frames if self._integrator.enabled else 1
        if not self._record(frame, sweep.position, image, frames):
            return
        if self._out_of_lens(image, sweep.step, frames):
            sweep.blocked()
        before = self._frame_reading(frame)
        move = sweep.took_one() if not sweep.done else None
        if move is not None:
            try:
                if not camera.drive_focus(move):
                    # The far end of the travel: there is nowhere left to
                    # sweep, and what was read up to here is the map.
                    sweep.blocked()
            except CameraError as exc:
                self._cancel_sweep(f"Depth map stopped: {exc}")
                return
        if sweep.done:
            self._finish_pass(camera, sweep)
            return
        self._settle(before)
        self._say_sweeping(sweep.taken)

    def _record(
        self, frame: LiveViewFrame, position: int, image: QImage, frames: int
    ) -> bool:
        """Read one settled picture into whichever survey is being filled.

        The one place the two kinds of sweep part company, along with
        :meth:`_interesting` and :meth:`_finish_scan`. Everything about
        driving the lens to get here was the same for both.

        The frame comes in as well as the picture because the points are
        places on the sensor, and it takes the frame's crop rectangle to say
        where on *this* picture each of them falls. A point outside the crop
        reads zero: on an unmagnified sweep that means someone placed it while
        magnified somewhere else, and a zero is the reading that says "nothing
        here" rather than a small one that would be fitted like an answer.

        Answers False when the sweep cannot go on, having said why.
        """
        noise = self._noise_variance / max(frames, 1)
        if self._scan == "points":
            survey = self._point_survey
            if survey is None:
                return False
            aspect = image.width() / max(image.height(), 1)
            crop = frame.crop_normalised
            readings = []
            for point in survey.points:
                seen = point.seen_in(crop)
                readings.append(
                    0.0
                    if seen is None
                    else measure(
                        image,
                        area_for(seen[0], seen[1], self._point_box, aspect),
                        noise,
                    )
                )
            survey.add(position, readings, self._sweep_pass)
            self.pointsFound.emit(survey.found())
            return True
        survey = self._survey
        if survey is None:
            return False
        try:
            survey.add(position, tile_sums(image, survey.tiling), noise)
        except ValueError:
            self._cancel_sweep(
                "Depth map stopped - the picture changed size under it"
            )
            return False
        self.depthMapReady.emit(survey.map(self._sweep_detail))
        return True

    def _interesting(self, margin: int) -> "tuple[int, int] | None":
        """The stretch worth sweeping again, from whichever survey is running."""
        survey = (
            self._point_survey if self._scan == "points" else self._survey
        )
        return survey.interesting(margin) if survey is not None else None

    def _out_of_lens(self, image: QImage, step: int, frames: int) -> bool:
        """Whether the picture has held still over a long stretch of driving.

        Which means the lens is against a stop and the drives are going
        nowhere. This is the backstop under the range the sweep was given
        rather than a replacement for it: however wrong that range turns out
        to be, a pass must not count out four hundred stops against a lens
        that cannot move, which is what a body that answers OK to a drive it
        did not make will otherwise have it do for several minutes.

        What it counts is **how far it has driven** since the picture last
        changed, not how many stops ago that was; see
        :data:`_SWEEP_QUIET_STEPS`. And it says nothing until the picture has
        changed once, because a pass begins in the margin the range finder
        put either side of what it found, where quiet is expected.

        It is judged on the stacked picture that was just read, so nothing is
        grabbed for it, and against the grain divided by the stack depth,
        because averaging frames divides their noise by as many.
        """
        pixels = green(image)[::2, ::2]
        before = self._sweep_pixels
        variance = variance_between(before, pixels) if before is not None else None
        # Both estimates are of a single frame, and averaging a stack divides
        # its noise by as many frames as went into it.
        grain = max(self._noise_variance, self._stop_grain, QUANTISATION)
        if variance is None or variance > _STOP_MOVED * grain / max(frames, 1):
            # The reference only moves when the picture does; see _STOP_MOVED.
            self._sweep_pixels = pixels
            self._sweep_silent = 0
            self._sweep_spoke = True
            return False
        if not self._sweep_spoke:
            return False
        self._sweep_silent += max(1, int(step))
        if self._sweep_silent < _SWEEP_QUIET_STEPS:
            return False
        self.status.emit(
            f"The picture has not changed in {self._sweep_silent} steps of "
            f"driving - the lens is against a stop, so there is no more to sweep"
        )
        return True

    def _say_sweeping(self, taken: int) -> None:
        """The one line that says where the sweep was pointed and how far in."""
        sweep = self._sweep
        if sweep is None:
            return
        low, high = self._sweep_span
        self.sweeping.emit(
            f"Pass {self._sweep_pass + 1}/{self._sweep_passes}, "
            f"stop {taken}/{sweep.samples} at {sweep.position}: "
            f"{low} to {high} in steps of {sweep.step}"
        )

    def _finish_pass(self, camera: NikonCamera, sweep: Sweep) -> None:
        """Plan the next pass over what this one found, or call the map done.

        A pass is only worth running if it can be finer than the one before
        it, and it can only be finer if the stretch worth sweeping is
        narrower than what was just swept. When it is not, saying so is the
        whole of the value: a map that stops after one pass because the
        answers it found are spread over the entire travel has not converged
        on anything, and looks exactly like one that has.
        """
        self._sweep = None
        self._sweep_pass += 1
        self._sweep_last_step = sweep.step
        if self._sweep_pass >= self._sweep_passes:
            self._finish_scan()
            return
        span = self._interesting(sweep.step)
        if span is None:
            self._finish_scan(
                "Nothing anywhere in the travel came into focus to refine"
            )
            return
        low = max(0, span[0])
        high = span[1]
        self._sweep_span = (low, high)
        step = max(self._sweep_minimum, (high - low) // self._sweep_samples)
        if step >= sweep.step:
            covered = sweep.step * self._sweep_samples
            self._finish_scan(
                f"Stopped after {self._sweep_pass} pass"
                f"{'' if self._sweep_pass == 1 else 'es'}: the answers cover "
                f"{high - low} steps of the {covered} just swept, so a finer "
                f"pass would not fit. Raise the stops per pass, or run it "
                f"again over what this one found."
            )
            return
        self._begin_pass(camera, low, step)

    def _finish_scan(self, why: str = "") -> None:
        """Stop the sweep and publish what it found, whichever kind it was."""
        self._sweep = None
        self._settling = False
        kind, self._scan = self._scan, ""
        apart = self._points_apart
        self._restore_view()
        passes = self._sweep_pass
        spent = f"{passes} pass" + ("" if passes == 1 else "es")
        if kind == "points":
            self.pointScanChanged.emit(False)
            survey = self._point_survey
            if survey is None or not len(survey):
                self.status.emit(why or "Point scan finished with nothing read")
                return
            results = survey.found()
            self.pointsFound.emit(results)
            if why:
                self.status.emit(why)
                return
            if apart:
                # A point whose best reading was the first one taken was
                # already past its best when the sweep began, and the sweep
                # cannot reach back for it: that would be a reversal, and the
                # play the reversal takes up is the one thing none of this can
                # measure. Say so instead of quietly reporting the edge of the
                # bracket as an answer.
                _rising, falling = survey.escaping()
                missing = [one for one in results if not one.known]
                if falling:
                    short = (
                        " One point was already at its best when the bracket "
                        "began, so its real peak is nearer than anything "
                        "swept -- raise 'Around AF' and measure again."
                    )
                elif missing:
                    short = (
                        f" {len(missing)} of them never came into focus "
                        f"anywhere in the bracket. Raise 'Around AF', or put "
                        f"point 1 on the nearest of the subjects -- the sweep "
                        f"can drive further out to reach a point but never "
                        f"back to reach one nearer than where it began."
                    )
                else:
                    short = ""
                self.status.emit(
                    f"Points measured at full magnification: "
                    f"{ordering(results)}.{short}"
                )
                return
            self.status.emit(f"Points measured in {spent}: {ordering(results)}")
            return
        self.depthChanged.emit(False)
        survey = self._survey
        if survey is None or not len(survey):
            self.status.emit(why or "Depth map finished with nothing measured")
            return
        depth_map = survey.map(self._sweep_detail)
        self.depthMapReady.emit(depth_map)
        self.status.emit(
            why or f"Depth map done in {spent}: {depth_map.describe()}"
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
        # Releasing the shutter takes the mirror down and, if it is asked to
        # focus first, drives the lens. Neither a hunt nor a depth sweep
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
