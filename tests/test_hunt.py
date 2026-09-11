"""Tests for fine tuning focus: autofocus, then better it a step at a time.

Two halves. The search itself has no camera in it, so it is run against
modelled focus curves -- a peak somewhere, the reading falling away either
side of it, a camera autofocus that gets roughly there, and a gearing with
play in it. Then the whole thing through the worker against a simulated lens,
because the part that is easy to get wrong is not the arithmetic but the
waiting: live view lags the lens, and an integrated picture is a stack of
frames that may straddle a focus move.

Nothing here checks that focus was driven to a remembered position, because
nothing does that. What is checked is where the *optics* ended up, which on a
lens with play in its gearing is not the same as where the step counts say it
is -- and, in the two places it matters most, what the picture actually read
when the search stopped against the best it ever saw.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtGui")

from PySide6.QtCore import QBuffer, QByteArray  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402

from scanny.camera.nikon import CameraError, LiveViewFrame  # noqa: E402
from scanny.ui.hunt import FineTune, Move  # noqa: E402

#: What the panel's minimum increment is by default.
STEP = 6


# -- the search --------------------------------------------------------------


def curve(position: float, width: float = 120.0, height: float = 400.0,
          floor: float = 0.0) -> float:
    """Sharpness against focus position: one hill at zero, and not much else."""
    return floor + height * float(np.exp(-(position / width) ** 2))


def _run(tune, *, af_error=12, slack=0, noise=0.0, seed=0, width=30.0,
         limit=400):
    """Run *tune* against a modelled lens; answer it and where the optics went.

    Focus is at zero. The optics are carried between two faces of the driver,
    so a reversal moves nothing until the play is taken up -- which is the
    whole thing the search is built not to care about -- and the camera's own
    autofocus drives them itself, to *af_error* steps off, leaving the driver
    with nothing useful to say about where they are.
    """
    rng = np.random.default_rng(seed)
    optics = float(af_error) * 3.0  # wherever the button was pressed
    driver = optics
    autofocused = 0
    while True:
        reading = curve(optics, width=width)
        if noise:
            reading *= 1 + rng.normal(0, noise)
        move = tune.step(max(reading, 0.0))
        if move is None:
            return tune, abs(optics), autofocused
        if move.autofocus:
            autofocused += 1
            optics = float(af_error)
            # The camera drove the optics itself: the play is wherever it left
            # it, and the step count knows nothing about any of it.
            driver = optics - slack / 2
        else:
            driver += move.steps
            optics = min(max(optics, driver), driver + slack)
        assert tune.probes < limit, "it has to stop on its own"


def _tune(**modelled):
    kept = {k: modelled.pop(k) for k in ("first", "patience") if k in modelled}
    return _run(FineTune(STEP, **kept), **modelled)


def test_the_first_thing_it_does_is_autofocus():
    """It improves on the camera's answer rather than on wherever it was left."""
    tune = FineTune(STEP)
    assert tune.step(10.0) == Move(autofocus=True)


@pytest.mark.parametrize("af_error", [12, -12, 30, -30, 6, -6])
def test_it_lands_on_focus_from_either_side_of_the_autofocus(af_error):
    """The peak may be on either side of where the camera left it, and which
    one it is is exactly what a search that only looks one way cannot tell."""
    tune, error, _ = _tune(af_error=af_error)
    assert tune.outcome == "found"
    assert error <= STEP


def test_it_searches_both_ways_round_without_autofocusing_again():
    """One autofocus, at the start. There used to be a second one to reset
    between searching one way and the other, and it was a mistake: autofocus
    on the same patch does not land on the same place twice, so everything the
    walk had learnt about which way things lay was worthless the moment it
    ran. A walk that turns itself round needs no datum to return to."""
    tune, _, autofocused = _tune(af_error=-18)
    assert autofocused == 1
    # It still saw both sides of the best: that is what turning round is for.
    assert any(where > tune.best_position for where in tune._fell_at)
    assert any(where < tune.best_position for where in tune._fell_at)


def test_it_never_takes_a_longer_step_than_the_one_it_was_given():
    """A step that grows through a lens's play is also the step that takes up
    the last of it, and it moves the optics by whatever was left. That is how
    a search walks over the peak it is looking for."""
    tune, _, _ = _tune(af_error=-30, slack=60, patience=20)
    strides = {abs(b - a) for (a, _), (b, _) in zip(tune.trail, tune.trail[1:])}
    # Zero is the pair of readings either side of an autofocus, which moves
    # the optics without moving the step count.
    assert strides <= {0, STEP}


def test_it_stops_on_the_best_reading_it_saw_and_not_a_slope_below_it():
    """The whole complaint about what was here before: it came back to within
    three per cent of the peak, which on a magnified subject is a focus error
    anybody can better by hand."""
    tune, _, _ = _tune(af_error=-24)
    assert tune.confirmed is not None
    assert tune.confirmed >= tune.best * 0.99


@pytest.mark.parametrize("af_error", [18, -18])
def test_it_ends_no_worse_than_the_camera_managed_on_its_own(af_error):
    tune, _, _ = _tune(af_error=af_error)
    assert tune.confirmed is not None
    assert tune.confirmed >= tune.baseline


@pytest.mark.parametrize("slack", [30, 90])
def test_play_in_the_gearing_is_simply_walked_through(slack):
    """The reason it comes back by reading rather than by step count: the play
    is taken up by walking, and nothing has to know how much of it there is --
    only how long to keep walking before giving up on it."""
    tune, error, _ = _tune(af_error=-24, slack=slack, patience=slack // STEP + 4)
    assert error <= STEP
    # Somewhere in there are the steps that moved nothing at all.
    readings = [reading for _, reading in tune.trail]
    assert any(a == b for a, b in zip(readings, readings[1:]))


@pytest.mark.parametrize("seed", range(6))
def test_a_reading_that_wanders_does_not_send_the_search_off(seed):
    """One falling reading is as likely to be the noise as the lens going the
    wrong way, so it takes two in a row to turn round."""
    tune, error, _ = _tune(af_error=-18, slack=30, noise=0.01, seed=seed,
                           patience=10)
    assert error <= 3 * STEP


def test_the_direction_that_says_nothing_at_all_is_given_up_on():
    """A lens with more play than the search will walk through, one way."""
    tune, _, _ = _tune(af_error=-24, slack=300, patience=4)
    assert tune.outcome in {"found", "lost", "restored"}


def test_a_picture_that_never_changes_stops_rather_than_walking_for_ever():
    """It looks further and further for something to climb -- that is the
    point of the growing reach -- but it is still bounded, and it still ends
    on the best reading it saw rather than wherever it ran out."""
    tune = FineTune(STEP, patience=4, max_probes=60)
    while True:
        move = tune.step(100.0)
        if move is None:
            break
        assert tune.probes < 120
    assert tune.outcome == "found"
    assert tune.confirmed == 100.0


def test_a_reading_of_nothing_at_all_is_said_to_be_nothing():
    """Zero is not a hill to climb, but it is not a reason to refuse to look.

    It used to stop on the spot, and that was wrong: zero means the picture
    has no detail above its own grain, which is true of anything far enough
    out of focus -- including a subject whose focus is a couple of hundred
    steps from wherever the camera's autofocus stopped. So it walks its reach
    both ways first, and only then says there is nothing there.
    """
    tune = FineTune(STEP, patience=3, max_probes=40)
    assert tune.step(0.0) == Move(autofocus=True)
    walked = 0
    while tune.step(0.0) is not None:
        walked += 1
        assert walked < 100
    assert walked > 6, "it has to have looked before saying there is nothing"
    assert tune.outcome == "nothing"


def test_focus_is_put_back_when_nothing_it_found_beat_the_autofocus():
    """The floor enforced rather than hoped for: a picture that reads worse
    every time it is looked at leaves the search standing somewhere worse than
    it started, and the only way back is the camera's own autofocus."""
    tune = FineTune(STEP, patience=3, max_probes=40)
    readings = iter([100.0, 100.0] + [40.0] * 400)
    autofocused = 0
    reading = next(readings)
    while True:
        move = tune.step(reading)
        if move is None:
            break
        if move.autofocus:
            autofocused += 1
            reading = 100.0 if autofocused >= 2 else next(readings)
        else:
            reading = next(readings)
        assert tune.probes < 200
    assert tune.outcome == "restored"
    assert autofocused == 2, "one to start with, one to put focus back"
    assert tune.confirmed == 100.0


def test_one_reading_off_a_cliff_is_enough_to_turn_it_round():
    """What waiting for a second fall costs on a subject with depth to it.

    A steep peak: the reading goes over the top at the best it will ever read,
    and is a fifth of that one step later. Taking a confirming step from there
    means standing somewhere nothing could be sharp, and then walking back
    through the whole of the lens's play to undo it.
    """
    tune = FineTune(STEP)
    assert tune.step(100.0) == Move(autofocus=True)
    assert tune.step(100.0) == Move(steps=STEP), "the camera's answer, then out"
    assert tune.step(400.0) == Move(steps=STEP), "climbing, so keep climbing"
    assert tune.step(80.0) == Move(steps=-STEP), "off a cliff: back, at once"


def test_a_cliff_on_the_very_first_step_turns_it_round_as_well():
    """It does not need to have climbed anything first. A direction that has
    already halved the reading holds nothing worth walking towards."""
    tune = FineTune(STEP)
    tune.step(100.0)
    assert tune.step(100.0) == Move(steps=STEP)
    assert tune.step(45.0) == Move(steps=-STEP)


def _legs(trail):
    """The trail split wherever the walk turned round or autofocused.

    What the rule below is about is one stretch of walking the same way: a
    walk that has turned round is *meant* to be reading far below the best it
    has seen, because that is what walking back across a lens's play looks
    like. Every step here is one increment or none, so a step that differs
    from the one before it is a reversal or an autofocus either way.
    """
    steps = [b - a for (a, _), (b, _) in zip(trail, trail[1:])]
    start = 0
    for at in range(1, len(steps) + 1):
        if at == len(steps) or steps[at] != steps[at - 1]:
            yield [reading for _, reading in trail[start : at + 1]]
            start = at


@pytest.mark.parametrize("af_error", [-30, -18, 18, 30])
@pytest.mark.parametrize("width", [8.0, 15.0, 30.0])
def test_it_never_walks_on_from_a_picture_that_has_fallen_apart(af_error, width):
    """The complaint this answers, as a property of every walk it makes.

    One reading a tenth below the best of the stretch it is on is allowed --
    that is how it finds out -- and the step after it has to be a step back.
    Two in a row means it went on walking while the picture it was reading was
    already unusable.
    """
    tune, _, _ = _run(FineTune(STEP), af_error=af_error, width=width)
    for leg in _legs(tune.trail):
        best, collapsed = 0.0, 0
        for reading in leg:
            best = max(best, reading)
            collapsed = collapsed + 1 if reading < 0.9 * best else 0
            assert collapsed <= 1, (
                f"it kept walking at {reading:.0f} against {best:.0f} in "
                f"{[f'{r:.0f}' for r in leg]}"
            )


def test_a_direction_that_is_not_working_turns_round_rather_than_giving_up():
    """The complaint this answers. A reading getting worse does not mean the
    search has failed; it means the other way, and that is the only thing it
    can mean. It used to end the whole search wherever it was standing --
    which, the direction being the wrong one, was as far from focus as that
    direction had managed to drag it."""
    tune = FineTune(STEP)
    tune.step(100.0)
    assert tune.step(100.0) == Move(steps=STEP), "the camera's answer, then out"
    assert tune.step(60.0) == Move(steps=-STEP), "worse: the other way"
    assert not tune.done, "and it is nowhere near finished"


def test_it_keeps_turning_round_for_as_long_as_it_is_allowed_to():
    """However often it is sent the wrong way, it answers with a step. What
    ends a walk is standing on the best reading it has seen, not running out
    of patience with a direction."""
    tune = FineTune(STEP, most_turns=6, autofocus=False)
    reading, moves = 100.0, 0
    while tune.step(reading) is not None:
        moves += 1
        reading *= 0.8  # every step the walk takes makes it worse
        assert moves < 60
    assert tune.turns > 6, "it went on turning rather than stopping"
    assert moves > 10, "and it kept walking while it did"


@pytest.mark.parametrize("away", [30, 60, 90, 150])
@pytest.mark.parametrize("sign", [1, -1])
def test_a_peak_outside_its_first_look_is_found_by_looking_further(away, sign):
    """The complaint this answers.

    The reach used to be fixed -- sixteen probes of the minimum increment,
    which is ninety drive steps on a lens with six thousand of travel. A peak
    further out than that sat outside the box, and the walk went back and
    forth inside the box a few times and announced it was done. It doubles its
    reach now, every time a direction is walked to the end of it without the
    reading ever falling away: a direction that said nothing is a direction
    not yet explored.
    """
    tune, error, _ = _run(
        FineTune(STEP), af_error=sign * away, width=15.0, limit=500
    )
    assert tune.outcome == "found"
    assert error <= STEP, f"a peak {away} steps out left it {error} steps away"


def test_a_direction_that_said_nothing_is_walked_further_next_time():
    """What "explored" means. Falling away is evidence; saying nothing is not,
    and the answer to a direction that said nothing is to go further in it."""
    tune = FineTune(STEP, patience=4)
    tune.step(10.0)
    reach = dict(tune._reach)
    for _ in range(60):
        if tune.step(10.0) is None:  # a picture that says the same thing always
            break
    assert min(tune._reach.values()) > min(reach.values())


def test_a_peak_on_the_far_side_of_the_autofocus_is_still_the_one_it_stands_on():
    """Whichever way it sets off, the walk turns round and finds it."""
    for first in (1, -1):
        tune, error, _ = _tune(af_error=-24, first=first)
        assert error <= STEP, f"setting off {first:+d} left it {error} steps out"
        assert tune.confirmed is not None
        assert tune.confirmed >= tune.best * 0.99


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
    reversal move nothing at all, the reading has grain on it, and the body's
    own autofocus gets close rather than right.
    """

    live_view_active = True
    exposure_preview = True

    def __init__(self, best: int = 0, start: int = 40, lag: int = 2,
                 noise: float = 6.0, slack: int = 0, depth: float = 20.0,
                 af_error: int = 14) -> None:
        self.blank = False
        self.best = best
        self.slack = slack
        self.depth = depth
        #: Where the optics actually are, as against where the steps say.
        self.optics = start
        self.position = start
        self.lag = lag
        self.noise = noise
        self.af_error = af_error
        self.autofocused = 0
        self.aimed: "list[tuple[int, int]]" = []
        self.zoomed: "list[int]" = []
        self.magnification = 1.0
        self.centre = (3008, 2008)
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
        self.zoomed.append(int(level))
        self.magnification = _MAGNIFICATION[int(level)]

    def zoom_level(self) -> int:
        return self.zoomed[-1] if self.zoomed else 0

    def set_af_area(self, x: int, y: int) -> None:
        self.aimed.append((int(x), int(y)))
        # The body centres its magnified view on the focus point.
        self.centre = (int(x), int(y))

    def autofocus(self) -> bool:
        self.autofocused += 1
        if self.blank:
            return False
        # The body drives the optics itself, close but not right, and leaves
        # the play wherever it happens to leave it.
        self.optics = self.best + self.af_error
        self.position = self.optics - self.slack // 2
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
        return _frame(
            np.clip(pixels, 0, 255), self.magnification, self.centre
        )


#: What each zoom level magnifies by, as measured on a D750.
_MAGNIFICATION = {0: 1.0, 2: 2.35, 3: 3.13, 4: 4.7, 5: 6.27, 6: 9.4, 7: 18.8}


def _frame(
    pixels: np.ndarray, magnification: float = 1.0, centre=(3008, 2008)
) -> LiveViewFrame:
    """One frame, reporting the crop the camera would be showing.

    The crop fields are what everything that keeps a place on the sensor reads
    -- the measured area above all -- so a fake that always claims the whole
    frame would test the mapping by never using it.
    """
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
    crop_w = int(6016 / magnification)
    crop_h = int(4016 / magnification)
    # The body will not show a crop that runs off the sensor.
    x = min(max(centre[0], crop_w // 2), 6016 - crop_w // 2)
    y = min(max(centre[1], crop_h // 2), 4016 - crop_h // 2)
    return LiveViewFrame(
        jpeg=bytes(data.data()), width=W, height=H,
        image_width=6016, image_height=4016,
        crop_width=crop_w, crop_height=crop_h,
        crop_center_x=x, crop_center_y=y,
        af_width=324, af_height=270, af_x=centre[0], af_y=centre[1],
    )


@pytest.fixture
def worker():
    pytest.importorskip("PySide6.QtWidgets")
    from scanny.ui.worker import CameraWorker

    return CameraWorker()


def _ready(worker, lens, *, frames: int = 4, area=None):
    """A camera, a reading to start from, and nothing searching yet."""
    worker._camera = lens
    if frames:
        worker.set_integration(True, frames)
    worker.set_sharpness(True)
    if area is not None:
        worker.set_sharpness_area(area)
    for _ in range(frames * 2 + 8):
        worker._grab()
    return lens


def _tuned(worker, lens, *, frames: int = 4, grabs: int = 9000, area=None) -> None:
    """Watch a whole fine tune, one grab at a time."""
    _ready(worker, lens, frames=frames, area=area)
    worker.fine_tune(STEP)
    for _ in range(grabs):
        if worker._hunt is None:
            break
        worker._grab()
    assert worker._hunt is None, "the search has to finish on its own"


@pytest.mark.parametrize("af_error", [14, -14, 30, -30])
def test_it_finds_focus_through_the_whole_machinery(worker, af_error):
    lens = _Lens(start=90, af_error=af_error)
    _tuned(worker, lens)
    assert lens.error <= 12, f"it left the optics at {lens.optics}, not near 0"


@pytest.mark.parametrize("af_error", [12, -12])
def test_it_finds_focus_on_a_subject_with_almost_no_depth_to_it(worker, af_error):
    """The subject this button is for: a magnified macro scene where a couple
    of steps is the whole of the depth of focus, so the reading goes over the
    top and falls off a cliff rather than down a slope."""
    lens = _Lens(start=60, depth=6.0, af_error=af_error)
    _tuned(worker, lens)
    assert lens.error <= 6


def test_it_does_not_walk_on_once_the_picture_has_fallen_apart(worker):
    """What was wrong with it: on a steep subject it took a confirming step
    past the peak, and one step past the peak there is a fifth of the reading
    left. Every probe it takes has to be worth taking."""
    lens = _Lens(start=60, depth=6.0, af_error=-18)
    readings = []
    _ready(worker, lens)
    worker.sharpnessChanged.connect(
        lambda value, peak: readings.append(value) if value else None
    )
    worker.fine_tune(STEP)
    tune = worker._hunt
    for _ in range(9000):
        if worker._hunt is None:
            break
        worker._grab()
    for leg in _legs(tune.trail):
        best, collapsed = 0.0, 0
        for reading in leg:
            best = max(best, reading)
            collapsed = collapsed + 1 if reading < 0.9 * best else 0
            assert collapsed <= 1, f"kept walking at {reading:.0f} of {best:.0f}"


def test_it_betters_what_the_camera_managed_on_its_own(worker):
    """The point of the button. The camera's own autofocus is the starting
    point and the floor, and what is measured at the end has to clear it."""
    lens = _Lens(start=90, af_error=24)
    _tuned(worker, lens)
    assert lens.error < 24


def test_it_finds_focus_without_integration(worker):
    lens = _Lens(start=60, af_error=-18)
    _tuned(worker, lens, frames=0)
    assert lens.error <= 12


@pytest.mark.parametrize("slack", [40, 90])
def test_play_in_the_gearing_costs_the_search_nothing(worker, slack):
    """No allowance, no compensation, nothing driven past its target and back:
    the search takes the play up itself, by walking."""
    lens = _Lens(start=60, slack=slack, af_error=-20)
    _tuned(worker, lens)
    assert lens.error <= 12


def test_it_starts_by_autofocusing_on_the_measured_area(worker):
    """Not wherever the focus box was left: the measured area is what the
    reading is about, and it is also what has to be on screen to be read."""
    area = (0.25, 0.25, 0.4, 0.4)
    lens = _Lens(start=60, af_error=-14)
    _tuned(worker, lens, area=area)
    assert lens.autofocused >= 1
    assert lens.aimed, "it never moved the focus box"
    # The middle of the area, in the frame's own coordinates.
    assert lens.aimed[0] == (int(0.45 * 6016), int(0.45 * 4016))


def test_it_magnifies_onto_the_measured_area_before_it_reads_anything(worker):
    """The single largest thing that can be done for the answer, for one
    command: a drive step moves the picture far more when the view is
    magnified, so the focus error that is lost in the grain at full frame is
    obvious at 18.8x."""
    lens = _Lens(start=60, af_error=-14)
    told = []
    worker.zoomChanged.connect(told.append)
    _tuned(worker, lens, area=(0.48, 0.48, 0.04, 0.04))
    assert lens.zoomed, "it never magnified"
    assert lens.zoomed[0] == 7, "a twenty-fifth of the frame fits at full zoom"
    assert told[0] == 7, "and the panel has to be told where the view went"
    # Once, before the search starts, and never again while it is running.
    assert set(lens.zoomed) == {7}


@pytest.mark.parametrize(
    "side, level",
    [(0.02, 7), (0.06, 6), (0.12, 5), (0.2, 4), (0.4, 2), (0.9, 0)],
)
def test_it_magnifies_as_far_as_it_can_and_still_show_the_whole_area(
    worker, side, level
):
    """As far as it will go *and still show the area*. Magnifying past the
    rectangle would read whatever part of it stayed on screen, which is a
    different question from the one the rectangle was drawn to ask."""
    lens = _ready(worker, _Lens(start=60), frames=0)
    worker.set_sharpness_area((0.5 - side / 2, 0.5 - side / 2, side, side))
    worker.fine_tune(STEP)
    assert (lens.zoomed[-1] if lens.zoomed else 0) == level


def test_the_measured_area_is_on_screen_once_it_has_magnified(worker):
    """Magnifying is only worth anything if the rectangle is still being read,
    and the focus point going to its middle first is what puts it there."""
    area = (0.2, 0.3, 0.05, 0.05)
    lens = _ready(worker, _Lens(start=60), frames=0)
    worker.set_sharpness_area(area)
    worker.fine_tune(STEP)
    worker._grab()
    shown = worker._sharpness.shown_in(worker._last_frame)
    assert shown is not None, "it magnified away from what it is measuring"
    x, y, w, h = shown
    assert 0.0 <= x and 0.0 <= y and x + w <= 1.0 and y + h <= 1.0
    assert w > 0.5 and h > 0.5, "and the area should fill most of the picture"


def test_with_no_area_marked_out_the_view_is_left_alone(worker):
    """There is nothing chosen to magnify onto, and going to 18.8x anyway
    would quietly replace "the whole frame" with a twentieth of it."""
    lens = _ready(worker, _Lens(start=60), frames=0)
    worker.fine_tune(STEP)
    assert lens.zoomed == []


def test_it_autofocuses_once_and_only_once(worker):
    """Autofocus on the same patch does not land on the same place twice, so a
    second one half way through would throw away everything the walk had
    learnt about which way things lie."""
    lens = _Lens(start=60, af_error=-14)
    _tuned(worker, lens)
    assert lens.autofocused == 1


def test_every_step_it_drives_is_the_one_increment_it_was_given(worker):
    lens = _Lens(start=60, slack=40, af_error=-20)
    _tuned(worker, lens)
    assert {abs(steps) for steps in lens.drives} == {STEP}


def test_no_reading_is_believed_until_the_lens_has_stopped(worker):
    """Live view lags the lens, so the frames straight after a drive show the
    focus it used to have. Believing one would send the search the wrong way."""
    lens = _ready(worker, _Lens(start=60))
    worker.fine_tune(STEP)
    settled_at = []
    for _ in range(9000):
        if worker._hunt is None:
            break
        before = len(lens.drives)
        worker._grab()
        if len(lens.drives) > before:
            settled_at.append(worker._integrator.pending)
    assert settled_at, "the search never probed"
    assert all(pending == 0 for pending in settled_at)


def test_the_stack_a_reading_comes_from_starts_after_the_drive(worker):
    """Otherwise the bottom of the stack is the focus position before it."""
    lens = _ready(worker, _Lens(start=60), frames=8)
    worker.fine_tune(STEP)
    checked = 0
    for _ in range(9000):
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
    assert checked > 3, "it has to have driven a few times to mean anything"


def test_the_settling_waits_for_the_picture_not_for_a_fixed_time(worker):
    """Read too soon and the search is told about the focus position it has
    just left, so it walks away from focus rather than towards it."""
    slow = _ready(worker, _Lens(start=60, lag=9))
    worker.fine_tune(STEP)
    waits = []
    for _ in range(9000):
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
    assert waits, "it never drove"
    assert max(waits) > 3, "a lens this far behind has to be waited for"
    assert slow.error <= 12


def test_how_far_live_view_runs_behind_is_measured_not_assumed(worker):
    lens = _Lens(start=60, lag=6)
    _tuned(worker, lens)
    assert worker._pipeline_lag is not None
    assert worker._pipeline_lag >= lens.lag, (
        "waiting less than the pipeline is deep reads the focus it just left"
    )


def test_grain_is_not_mistaken_for_the_move_arriving(worker):
    """One grain of noise on a reading of nearly nothing is a difference of
    hundreds of per cent. Measuring the pipeline as shorter than it is would
    have every reading after it taken too early."""
    lens = _Lens(start=60, lag=6, noise=14.0)
    _tuned(worker, lens, frames=0)
    assert lens.error <= 24


def test_the_search_plots_its_own_readings(worker):
    """The line starts again with the search, so what is drawn is the search."""
    _ready(worker, _Lens(start=60))
    told = []
    worker.sharpnessChanged.connect(lambda value, peak: told.append((value, peak)))
    worker.fine_tune(STEP)
    assert told[-1] == (0.0, 0.0)


def test_it_says_so_when_there_is_nothing_in_the_area_to_focus_on(
    worker, monkeypatch
):
    """A reading of zero is no hill at all -- but it is a reason to go and
    look, not a reason to refuse. It walks its reach both ways first, and says
    there is nothing only once that has come back empty.

    The reach is cut down here so the test does not have to sit through three
    hundred probes of an empty wall; what is being checked is what the worker
    does with the answer.
    """
    from scanny.ui import worker as wk

    monkeypatch.setattr(
        wk, "FineTune", lambda step: FineTune(step, patience=2, max_probes=6)
    )
    lens = _Lens()
    lens.blank = True
    worker._camera = lens
    worker.set_sharpness(True)
    said = []
    worker.status.connect(said.append)
    worker.fine_tune(STEP)
    for _ in range(3000):
        if worker._hunt is None:
            break
        worker._grab()
    assert lens.autofocused == 1
    assert lens.drives, "it never went to look"
    assert worker._hunt is None
    assert "Nothing in the measured area" in said[-1]


def test_it_will_not_search_what_it_cannot_measure(worker):
    worker._camera = _Lens()
    said = []
    worker.failed.connect(said.append)
    worker.fine_tune(STEP)
    assert worker._hunt is None
    assert "sharpness" in said[-1]


def test_it_will_not_search_without_live_view(worker):
    lens = _Lens()
    lens.live_view_active = False
    worker._camera = lens
    worker.set_sharpness(True)
    said = []
    worker.failed.connect(said.append)
    worker.fine_tune(STEP)
    assert worker._hunt is None
    assert "live view" in said[-1]


def test_taking_the_focus_by_hand_stops_the_search(worker):
    _ready(worker, _Lens(start=60), frames=0)
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
def test_anything_that_changes_the_picture_stops_the_search(worker, interrupt):
    _ready(worker, _Lens(start=60), frames=0)
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


def test_a_camera_that_refuses_to_drive_stops_the_search(worker):
    lens = _Stuck(start=60, after=3)
    _ready(worker, lens, frames=0)
    said = []
    worker.failed.connect(said.append)
    worker.fine_tune(STEP)
    for _ in range(600):
        if worker._hunt is None:
            break
        worker._grab()
    assert worker._hunt is None
    assert "stuck" in said[-1]


def test_the_measured_area_is_read_where_it_was_put(worker):
    """Nothing moves it about: a search decides on one reading against the one
    before it, so anything that changed *what* was measured between them would
    corrupt the only comparison it has.
    """
    area = (0.25, 0.25, 0.4, 0.4)
    lens = _Lens(start=60, af_error=-16)
    _tuned(worker, lens, area=area)
    assert worker._sharpness.area == area
    assert lens.error <= 12


def test_nothing_is_shown_or_measured_while_the_lens_is_moving(worker):
    """A frame caught mid-move belongs to no focus position. Stacking one
    blends two positions into a picture and then measures the blend; showing
    one on its own puts a flash of grain on screen where a clean image was."""
    lens = _ready(worker, _Lens(start=60))
    pictures, readings = [], []
    worker.frameReady.connect(
        lambda frame, image: pictures.append(image) if not image.isNull() else None
    )
    worker.sharpnessChanged.connect(lambda value, peak: readings.append(value))
    worker.fine_tune(STEP)
    for _ in range(9000):
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
    lens = _ready(worker, _Lens(start=60))
    pictures = []
    worker.frameReady.connect(
        lambda frame, image: pictures.append(image) if not image.isNull() else None
    )
    worker.fine_tune(STEP)
    for _ in range(9000):
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
