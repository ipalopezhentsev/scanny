"""Tests for hunting focus by driving it to the top of the sharpness reading.

Two halves. The walk itself has no camera in it, so it is run against
modelled focus curves -- a peak somewhere, the reading falling away either
side of it, and a gearing with play in it. Then the whole thing through the
worker against a simulated lens, because the part that is easy to get wrong is
not the arithmetic but the waiting: live view lags the lens, and an integrated
picture is a stack of frames that may straddle a focus move.

Nothing here checks that focus was driven to a remembered position, because
nothing does that any more. What is checked is where the *optics* ended up,
which on a lens with play in its gearing is not the same as where the step
counts say it is.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtGui")

from PySide6.QtCore import QBuffer, QByteArray  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402

from scanny.camera.nikon import CameraError, LiveViewFrame  # noqa: E402
from scanny.ui.hunt import Walk  # noqa: E402

#: What the panel's minimum increment is by default.
STEP = 6


# -- the walk ----------------------------------------------------------------


def curve(position: float, peak: int, width: float = 120.0, height: float = 400.0,
          floor: float = 0.0) -> float:
    """Sharpness against focus position: one hill, and not much either side."""
    return floor + height * float(np.exp(-((position - peak) / width) ** 2))


def _drive(hunt, start_offset, *, slack=0, noise=0.0, seed=0, width=30.0, limit=80):
    """Run *hunt* against a modelled lens; answer it and where the optics went.

    The optics are carried between two faces of the driver, so a reversal
    moves nothing until the play is taken up -- which is the whole thing the
    walk is built not to care about.
    """
    rng = np.random.default_rng(seed)
    driver = optics = 0.0
    while True:
        reading = curve(optics - start_offset, 0, width=width)
        if noise:
            reading *= 1 + rng.normal(0, noise)
        move = hunt.step(reading)
        if move is None:
            return hunt, abs(optics - start_offset)
        driver += move
        optics = min(max(optics, driver), driver + slack)
        assert hunt.probes < limit, "it has to stop on its own"


def _walk(start_offset, *, unit=6, patience=12, **modelled):
    return _drive(Walk(unit, patience=patience), start_offset, **modelled)


def test_the_walk_goes_out_until_the_reading_turns_over_and_comes_back_to_it():
    """The shape of it: out one way, over the top, back to the best of it."""
    walk, error = _walk(24)
    positions = [position for position, _ in walk.trail]
    readings = [reading for _, reading in walk.trail]
    top = positions[readings.index(walk.best)]
    assert error == 0
    assert positions[-1] == top, "it has to finish where the best reading was"
    assert max(positions) > top, "having gone past it first"


def test_the_walk_turns_round_when_it_starts_out_the_wrong_way():
    walk, error = _walk(-24)
    assert walk.outcome == "found"
    assert error == 0


@pytest.mark.parametrize("offset", [6, -6, 18, -30, 48])
def test_the_walk_lands_on_focus_from_either_side(offset):
    assert _walk(offset)[1] == 0


@pytest.mark.parametrize("slack", [30, 90])
def test_play_in_the_gearing_is_simply_walked_through(slack):
    """The whole reason it comes back by reading rather than by step count:
    the play is taken up by walking, and nothing has to know how much of it
    there is -- only how long to keep walking before giving up on it."""
    walk, error = _walk(24, slack=slack, patience=slack // 6 + 3)
    assert error == 0
    # Somewhere in there are the steps that moved nothing at all.
    readings = [reading for _, reading in walk.trail]
    assert any(a == b for a, b in zip(readings, readings[1:]))


def test_more_play_than_it_will_walk_through_is_said_rather_than_hidden():
    walk, error = _walk(24, slack=120, patience=4)
    assert walk.outcome == "lost"
    assert error > 0


def test_the_walk_drives_a_fraction_of_what_counting_steps_would():
    walk, _ = _walk(24, slack=30, patience=8)
    positions = [position for position, _ in walk.trail]
    driven = sum(abs(b - a) for a, b in zip(positions, positions[1:]))
    assert driven < 400


def test_the_step_grows_through_the_play_on_the_way_out():
    """Every probe in the play reads exactly what the last one did, so there is
    nothing to be learnt by taking them one small step at a time."""
    # Focus behind where it starts, so the walk turns round and then has the
    # play to take up on its way out.
    walk, error = _walk(-24, slack=90, patience=20)
    strides = [abs(b - a) for (a, _), (b, _) in zip(walk.trail, walk.trail[1:])]
    assert max(strides) > 6, "it should have lengthened its step in the play"
    assert error == 0


def test_the_step_never_grows_on_the_way_back():
    """The step that finally takes up the last of the play also moves the
    optics by whatever is left of it, so a long step on the way back can carry
    the lens clean past the reading it came back for -- which is the error the
    walk exists to avoid."""
    walk, error = _walk(24, slack=90, patience=20)
    turned = [reading for _, reading in walk.trail].index(walk.best)
    coming_back = [
        abs(b - a)
        for (a, _), (b, _) in zip(walk.trail[turned:], walk.trail[turned + 1 :])
    ]
    assert coming_back, "it has to have come back at all"
    assert set(coming_back) == {6}, "in the step it was asked for, every one"
    assert error == 0


@pytest.mark.parametrize("seed", range(6))
def test_a_reading_that_wanders_does_not_send_the_walk_off(seed):
    """One falling reading is as likely to be the noise as the lens going the
    wrong way, so it takes two in a row to turn round."""
    assert _walk(18, slack=30, noise=0.01, seed=seed, patience=8)[1] <= 18


def test_a_walk_that_is_getting_nowhere_still_lands_where_it_looked_best():
    """Whatever happens, it ends at the best reading it saw rather than
    wherever the last probe left it."""
    walk, _ = _walk(24, slack=200, patience=6)
    positions = [position for position, _ in walk.trail]
    assert abs(walk.position - walk.best_position) <= abs(
        max(positions) - min(positions)
    )
    assert walk.outcome in {"found", "lost", "exhausted"}


def test_the_walk_gives_up_rather_than_walking_for_ever():
    """A picture that says the same thing however far it is driven."""
    walk = Walk(6, patience=4)
    while walk.step(100.0) is not None:
        assert walk.probes < 60
    assert walk.outcome in {"found", "nothing"}
    assert abs(walk.position) <= 6 * 20, "and it does not wander off doing it"


def test_a_reading_of_nothing_at_all_is_said_to_be_nothing():
    walk = Walk(6, patience=3)
    while walk.step(0.0) is not None:
        assert walk.probes < 60
    assert walk.outcome == "nothing"


def test_a_coarse_walk_gives_up_on_distance_rather_than_on_probes():
    """A step of two hundred and fifty spends the travel of a whole lens in
    the dozen probes a fine walk takes to cross its own play."""
    walk, _ = _walk(0, unit=250, width=30.0)
    assert max(abs(position) for position, _ in walk.trail) <= 1000


# -- the whole thing, through the worker -------------------------------------

W, H = 96, 72


def _texture() -> np.ndarray:
    """Detail that does not repeat, which is what real subjects look like."""
    if not hasattr(_texture, "cached"):
        field = np.random.default_rng(1).normal(0, 1, (H, W))
        for _ in range(2):
            field = (
                field
                + np.roll(field, 1, 0) + np.roll(field, -1, 0)
                + np.roll(field, 1, 1) + np.roll(field, -1, 1)
            ) / 5
        _texture.cached = np.clip(120 + 55 * field / field.std(), 0, 255)
    return _texture.cached


def _blurred(pixels: np.ndarray, passes: int) -> np.ndarray:
    for _ in range(passes):
        pixels = (
            pixels
            + np.roll(pixels, 1, 0) + np.roll(pixels, -1, 0)
            + np.roll(pixels, 1, 1) + np.roll(pixels, -1, 1)
        ) / 5
    return pixels


#: The subject at whole numbers of blur passes, worked out once.
_STAGES = [_blurred(_texture(), n) for n in range(13)]


def _defocused(steps: float, depth: float) -> np.ndarray:
    """The subject as many drive steps from focus, blurred smoothly.

    *depth* is how many steps it takes to go visibly soft -- a few hundred on
    an ordinary scene, a few tens on a magnified macro one. Smoothly matters:
    blur in whole passes would make the reading a staircase, and a staircase
    hides a lens that comes back to a step count but not to the same place.
    """
    softness = min(abs(steps) / depth, 11.999)
    lower = int(softness)
    weight = softness - lower
    return _STAGES[lower] * (1 - weight) + _STAGES[lower + 1] * weight


class _Lens:
    """A camera whose live view goes soft as focus leaves its best position.

    Everything about it that matters is real: frames come out of a pipeline
    and lag the lens, the gearing has play in it so the first steps after a
    reversal move nothing at all, and the reading has grain on it.
    """

    live_view_active = True
    exposure_preview = True

    def __init__(self, best: int = 0, start: int = 40, lag: int = 2,
                 noise: float = 6.0, slack: int = 0, depth: float = 20.0) -> None:
        self.blank = False
        self.best = best
        self.slack = slack
        self.depth = depth
        #: Where the optics actually are, as against where the steps say.
        self.optics = start
        self.position = start
        self.lag = lag
        self.noise = noise
        self.autofocused = 0
        self.drives: "list[int]" = []
        self._history = [start] * (lag + 1)
        self._rng = np.random.default_rng(17)

    def drive_focus(self, steps: int) -> bool:
        self.position += int(steps)
        self.drives.append(int(steps))
        # The optics are carried between two faces of the driver, and move
        # only when one of them catches up with them.
        self.optics = min(max(self.optics, self.position), self.position + self.slack)
        return True

    def set_zoom_level(self, level: int) -> None:
        pass

    def zoom_level(self) -> int:
        return 0

    def autofocus(self) -> bool:
        self.autofocused += 1
        return True

    def stop_live_view(self) -> None:
        pass

    def set_setting(self, name, value) -> None:
        pass

    def settings(self) -> list:
        return []

    def set_exposure_preview(self, enabled: bool) -> None:
        self.exposure_preview = enabled

    @property
    def error(self) -> int:
        """How far the optics -- not the step count -- are from focus."""
        return abs(self.optics - self.best)

    def live_view_frame(self) -> LiveViewFrame:
        self._history.append(self.optics)
        shown = self._history[-(self.lag + 1)]
        if self.blank:
            pixels = np.full((H, W), 90.0)  # an empty wall: nothing to focus on
        else:
            pixels = _defocused(shown - self.best, self.depth)
        pixels = pixels + self._rng.normal(0, self.noise, pixels.shape)
        return _frame(np.clip(pixels, 0, 255))


def _frame(pixels: np.ndarray) -> LiveViewFrame:
    grey = pixels.astype(np.uint8)
    buffer = np.zeros((H, W, 4), np.uint8)
    for channel in range(3):
        buffer[:, :, channel] = grey
    buffer[:, :, 3] = 255
    image = QImage(buffer.tobytes(), W, H, W * 4, QImage.Format.Format_RGB32).copy()
    data = QByteArray()
    sink = QBuffer(data)
    sink.open(QBuffer.OpenModeFlag.WriteOnly)
    image.save(sink, "JPG", 95)
    return LiveViewFrame(
        jpeg=bytes(data.data()), width=W, height=H,
        image_width=6016, image_height=4016, crop_width=6016, crop_height=4016,
        crop_center_x=3008, crop_center_y=2008,
        af_width=324, af_height=270, af_x=3008, af_y=2008,
    )


@pytest.fixture
def worker():
    pytest.importorskip("PySide6.QtWidgets")
    from scanny.ui.worker import CameraWorker

    return CameraWorker()


def _ready(worker, lens, *, frames: int = 4, area=None):
    """A camera, a reading to start from, and nothing walking yet."""
    worker._camera = lens
    if frames:
        worker.set_integration(True, frames)
    worker.set_sharpness(True)
    if area is not None:
        worker.set_sharpness_area(area)
    for _ in range(frames * 2 + 8):
        worker._grab()
    return lens


def _run(worker, lens, *, frames: int = 4, grabs: int = 3000, area=None) -> None:
    """Watch a whole walk, one grab at a time."""
    _ready(worker, lens, frames=frames, area=area)
    worker.fine_tune(STEP)
    for _ in range(grabs):
        if worker._hunt is None:
            break
        worker._grab()
    assert worker._hunt is None, "the walk has to finish on its own"


@pytest.mark.parametrize("start", [12, -12, 40, -40])
def test_it_finds_focus_through_the_whole_machinery(worker, start):
    lens = _Lens(start=start)
    _run(worker, lens)
    assert lens.error <= 12, f"the walk left the optics at {lens.optics}, not near 0"


def test_it_finds_focus_without_integration(worker):
    lens = _Lens(start=30)
    _run(worker, lens, frames=0)
    assert lens.error <= 12


@pytest.mark.parametrize("slack", [40, 90])
def test_play_in_the_gearing_costs_the_walk_nothing(worker, slack):
    """No allowance, no compensation, nothing driven past its target and back:
    the walk takes the play up itself, by walking."""
    lens = _Lens(start=30, slack=slack)
    _run(worker, lens)
    assert lens.error <= 12


def test_no_reading_is_believed_until_the_lens_has_stopped(worker):
    """Live view lags the lens, so the frames straight after a drive show the
    focus it used to have. Believing one would send the walk the wrong way."""
    lens = _ready(worker, _Lens(start=40))
    worker.fine_tune(STEP)
    settled_at = []
    for _ in range(3000):
        if worker._hunt is None:
            break
        before = len(lens.drives)
        worker._grab()
        if len(lens.drives) > before:
            settled_at.append(worker._integrator.pending)
    assert settled_at, "the walk never probed"
    assert all(pending == 0 for pending in settled_at)


def test_the_stack_a_reading_comes_from_starts_after_the_drive(worker):
    """Otherwise the bottom of the stack is the focus position before it."""
    lens = _ready(worker, _Lens(start=40), frames=8)
    worker.fine_tune(STEP)
    checked = 0
    for _ in range(3000):
        if worker._hunt is None:
            break
        drives = len(lens.drives)
        worker._grab()
        if len(lens.drives) == drives:
            continue
        assert worker._settling
        while worker._settling:
            worker._grab()
        assert worker._integrator.pending == 0
        checked += 1
    assert checked > 3, "the walk has to have driven a few times to mean anything"


def test_the_settling_waits_for_the_picture_not_for_a_fixed_time(worker):
    """Read too soon and the walk is told about the focus position it has just
    left, so it walks away from focus rather than towards it."""
    slow = _ready(worker, _Lens(start=40, lag=9))
    worker.fine_tune(STEP)
    waits = []
    for _ in range(3000):
        if worker._hunt is None:
            break
        drives = len(slow.drives)
        worker._grab()
        if len(slow.drives) == drives:
            continue
        frames = 0
        while worker._settling:
            worker._grab()
            frames += 1
        waits.append(frames)
    assert waits, "the walk never drove"
    assert max(waits) > 3, "a lens this far behind has to be waited for"
    assert slow.error <= 12


def test_how_far_live_view_runs_behind_is_measured_not_assumed(worker):
    lens = _Lens(start=40, lag=6)
    _run(worker, lens)
    assert worker._pipeline_lag is not None
    assert worker._pipeline_lag >= lens.lag, (
        "waiting less than the pipeline is deep reads the focus it just left"
    )


def test_grain_is_not_mistaken_for_the_move_arriving(worker):
    """One grain of noise on a reading of nearly nothing is a difference of
    hundreds of per cent. Measuring the pipeline as shorter than it is would
    have every reading after it taken too early."""
    lens = _Lens(start=40, lag=6, noise=14.0)
    _run(worker, lens, frames=0)
    assert lens.error <= 24


def test_the_walk_plots_its_own_readings(worker):
    """The line starts again with the walk, so what is drawn is the walk."""
    _ready(worker, _Lens(start=30))
    told = []
    worker.sharpnessChanged.connect(lambda value, peak: told.append((value, peak)))
    worker.fine_tune(STEP)
    assert told[-1] == (0.0, 0.0)


def test_the_camera_gets_it_close_when_there_is_nothing_to_climb(worker):
    """A reading of zero is no hill at all, so the camera's own autofocus is
    the opening move -- and only the opening move."""
    lens = _Lens()
    lens.blank = True
    worker._camera = lens
    worker.set_sharpness(True)
    worker.fine_tune(STEP)
    for _ in range(40):
        worker._grab()
    assert lens.autofocused == 1


def test_the_camera_is_left_alone_when_there_is_something_to_climb(worker):
    lens = _Lens(start=30)
    worker._camera = lens
    worker.set_sharpness(True)
    worker.fine_tune(STEP)
    for _ in range(12):
        worker._grab()
    assert lens.autofocused == 0


def test_pressing_the_button_before_the_first_reading_keeps_the_focus(worker):
    """The meter not having read anything yet is not the same as defocused,
    and treating it as such would throw away good focus on the first press."""
    lens = _Lens(start=30)
    worker._camera = lens
    worker.set_integration(True, 8)
    worker.set_sharpness(True)
    worker.fine_tune(STEP)
    for _ in range(20):
        worker._grab()
    assert lens.autofocused == 0


def test_it_will_not_walk_what_it_cannot_measure(worker):
    worker._camera = _Lens()
    said = []
    worker.failed.connect(said.append)
    worker.fine_tune(STEP)
    assert worker._hunt is None
    assert "sharpness" in said[-1]


def test_it_will_not_walk_without_live_view(worker):
    lens = _Lens()
    lens.live_view_active = False
    worker._camera = lens
    worker.set_sharpness(True)
    said = []
    worker.failed.connect(said.append)
    worker.fine_tune(STEP)
    assert worker._hunt is None
    assert "live view" in said[-1]


def test_taking_the_focus_by_hand_stops_the_walk(worker):
    _ready(worker, _Lens(start=30), frames=0)
    worker.fine_tune(STEP)
    assert worker._hunt is not None
    worker.drive_focus(50)
    assert worker._hunt is None


@pytest.mark.parametrize(
    "interrupt",
    [
        lambda w: w.set_zoom(2),
        lambda w: w.set_sharpness_area((0.1, 0.1, 0.3, 0.3)),
        lambda w: w.set_integration(True, 8),
        lambda w: w.stop_live_view(),
        lambda w: w.cancel_hunt(),
    ],
)
def test_anything_that_changes_the_picture_stops_the_walk(worker, interrupt):
    _ready(worker, _Lens(start=30), frames=0)
    worker.fine_tune(STEP)
    told = []
    worker.huntChanged.connect(told.append)
    interrupt(worker)
    assert worker._hunt is None
    assert told[-1] is False


class _Stuck(_Lens):
    """A lens that seizes partway through, which is what a jam looks like."""

    def __init__(self, *args, after: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.after = after

    def drive_focus(self, steps: int) -> bool:
        if len(self.drives) >= self.after:
            raise CameraError("focus is stuck")
        return super().drive_focus(steps)


def test_a_camera_that_refuses_to_drive_stops_the_walk(worker):
    lens = _Stuck(start=30, after=3)
    _ready(worker, lens, frames=0)
    said = []
    worker.failed.connect(said.append)
    worker.fine_tune(STEP)
    for _ in range(300):
        if worker._hunt is None:
            break
        worker._grab()
    assert worker._hunt is None
    assert "stuck" in said[-1]


def test_the_measured_area_is_read_where_it_was_put(worker):
    """Nothing moves it about: a walk decides on one reading against the one
    before it, so anything that changed *what* was measured between them would
    corrupt the only comparison it has.
    """
    area = (0.25, 0.25, 0.4, 0.4)
    lens = _Lens(start=30)
    _run(worker, lens, area=area)
    assert worker._sharpness.area == area
    assert lens.error <= 12


def test_nothing_is_shown_or_measured_while_the_lens_is_moving(worker):
    """A frame caught mid-move belongs to no focus position. Stacking one
    blends two positions into a picture and then measures the blend; showing
    one on its own puts a flash of grain on screen where a clean image was."""
    lens = _ready(worker, _Lens(start=30))
    pictures, readings = [], []
    worker.frameReady.connect(
        lambda frame, image: pictures.append(image) if not image.isNull() else None
    )
    worker.sharpnessChanged.connect(lambda value, peak: readings.append(value))
    worker.fine_tune(STEP)
    for _ in range(3000):
        if worker._hunt is None:
            break
        drives = len(lens.drives)
        worker._grab()
        if len(lens.drives) == drives:
            continue
        pictures.clear()
        readings.clear()
        while worker._settling:
            worker._grab()
        assert pictures == [], "a picture went up while the lens was moving"
        assert readings == [], "a reading was taken while the lens was moving"


def test_the_first_frame_after_a_move_is_not_shown_on_its_own(worker):
    """The stack is dropped when the move lands, because it holds frames from
    the focus position just left. The frame that starts the new one is a
    single unaveraged frame: it must not go to the screen in place of the
    clean picture that is already there."""
    lens = _ready(worker, _Lens(start=30))
    pictures = []
    worker.frameReady.connect(
        lambda frame, image: pictures.append(image) if not image.isNull() else None
    )
    worker.fine_tune(STEP)
    for _ in range(3000):
        if worker._hunt is None:
            break
        drives = len(lens.drives)
        worker._grab()
        if len(lens.drives) == drives:
            continue
        while worker._settling:
            worker._grab()
        pictures.clear()
        # The frame that restarts the stack, and the rest of that stack.
        for _ in range(worker._integrator.frames - 1):
            worker._grab()
        assert pictures == [], "an unaveraged frame went up after the move"
        worker._grab()
        assert len(pictures) == 1, "the completed stack did not go up"
