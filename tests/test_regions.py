"""Tests for focus regions: each one's best, and the one focus that serves all.

Three parts. The bookkeeping has no camera in it and is run against readings
made up to a known shape. The window's side is checked for the gestures -- a
region is drawn with ctrl held and taken away the same way -- and for the
report, which has to show the pictures the way the view is being shown. And
then the whole of it through the worker against a simulated rig: regions at
three different depths across the frame, a lens with play in its gearing,
and a camera that has to be panned and magnified onto each region to read it.

What is checked at the end is where the *optics* are, against the true
compromise worked out from the model itself -- not against step counts,
which on a lens with play in it are not the same thing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import (  # noqa: E402
    QBuffer,
    QByteArray,
    QPoint,
    QPointF,
    QSettings,
    Qt,
)
from PySide6.QtGui import QImage, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.camera.nikon import CameraError, LiveViewFrame  # noqa: E402
from scanny.ui import main_window as mw  # noqa: E402
from scanny.ui.orientation import Orientation  # noqa: E402
from scanny.ui.regions import (  # noqa: E402
    HURRY_BELOW,
    HURRY_STRIDE,
    MAX_REGIONS,
    TURN_BACK,
    Calibration,
    CalibrationReport,
    Look,
    Reading,
    Region,
    RegionResult,
    View,
    average_of_best,
    clock,
    colour_for,
    combine,
    cut_out,
    describe_depth,
    summarise,
)
from scanny.ui.sharpness import measure  # noqa: E402

#: The lens's minimum increment, as the panel has it by default.
STEP = 6


# -- pictures ----------------------------------------------------------------


def _texture(height: int, width: int, seed: int = 1) -> np.ndarray:
    """Detail that does not repeat, which is what real subjects look like."""
    field = np.random.default_rng(seed).normal(0, 1, (height, width))
    for _ in range(2):
        field = (
            field
            + np.roll(field, 1, 0) + np.roll(field, -1, 0)
            + np.roll(field, 1, 1) + np.roll(field, -1, 1)
        ) / 5
    return np.clip(128 + 55 * field / field.std(), 0, 255)


def _blur(pixels: np.ndarray, passes: float) -> np.ndarray:
    """Soften by a smooth number of passes, so the reading is not a staircase."""
    stages = [pixels]
    for _ in range(int(passes) + 1):
        last = stages[-1]
        stages.append(
            (
                last
                + np.roll(last, 1, 0) + np.roll(last, -1, 0)
                + np.roll(last, 1, 1) + np.roll(last, -1, 1)
            ) / 5
        )
    whole = int(passes)
    weight = passes - whole
    return stages[whole] * (1 - weight) + stages[whole + 1] * weight


def _image(pixels: np.ndarray) -> QImage:
    height, width = pixels.shape
    grey = np.clip(pixels, 0, 255).astype(np.uint8)
    buffer = np.zeros((height, width, 4), np.uint8)
    for channel in range(3):
        buffer[:, :, channel] = grey
    buffer[:, :, 3] = 255
    return QImage(
        buffer.tobytes(), width, height, width * 4, QImage.Format.Format_RGB32
    ).copy()


# -- regions, as places on the sensor ----------------------------------------


def test_a_region_on_the_whole_frame_is_where_it_was_drawn():
    region = Region(0.2, 0.3, 0.1, 0.1)
    assert region.seen_in((0.0, 0.0, 1.0, 1.0)) == pytest.approx((0.2, 0.3, 0.1, 0.1))


def test_magnifying_onto_a_region_fills_the_picture_with_it():
    region = Region(0.2, 0.3, 0.1, 0.1)
    # A fifth of the frame, centred on the region.
    shown = region.seen_in((0.15, 0.25, 0.2, 0.2))
    assert shown == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_a_region_off_the_view_is_not_on_the_picture():
    assert Region(0.8, 0.8, 0.1, 0.1).seen_in((0.0, 0.0, 0.3, 0.3)) is None


def test_a_region_half_on_the_view_is_clipped_to_it():
    shown = Region(0.25, 0.25, 0.1, 0.1).seen_in((0.0, 0.0, 0.3, 0.3))
    assert shown is not None
    assert shown[0] + shown[2] == pytest.approx(1.0)


def test_cutting_a_region_out_keeps_its_pixels():
    pixels = np.zeros((40, 80))
    pixels[10:20, 40:60] = 255.0
    picture = cut_out(_image(pixels), (0.5, 0.25, 0.25, 0.25))
    assert (picture.width(), picture.height()) == (20, 10)
    assert QImage(picture).pixelColor(5, 5).red() == 255


# -- the arithmetic of a compromise ------------------------------------------


def test_the_average_is_of_each_region_against_its_own_best():
    """A reading has no units: a region with ten times the texture must not
    have ten times the say."""
    assert average_of_best([90.0, 9.0], [100.0, 10.0]) == pytest.approx(0.9)
    assert average_of_best([100.0, 5.0], [100.0, 10.0]) == pytest.approx(0.75)


def test_a_region_with_no_peak_has_no_say():
    assert average_of_best([90.0, 0.0], [100.0, 0.0]) == pytest.approx(0.9)
    assert average_of_best([], []) == 0.0


def _view(index: int) -> View:
    return View(7, (1000 + 100 * index, 1000), (1000 + 100 * index, 1000, 320, 213))


def _three() -> Calibration:
    return Calibration(
        [Region(0.1 * n, 0.2, 0.05, 0.05) for n in range(1, 4)], STEP
    )


def test_the_regions_are_tuned_one_after_another_then_the_compromise_begins():
    run = _three()
    assert run.phase == "peaks" and run.index == 0
    assert run.region_tuned(Look(100.0), "found", _view(0))
    assert run.region_tuned(Look(40.0), "found", _view(1))
    assert not run.region_tuned(Look(70.0), "found", _view(2))
    assert run.begin_compromise()
    assert run.phase == "compromise"


def test_the_compromise_walks_without_the_camera_s_autofocus():
    """There is no one place for the camera to focus on, and the peaks are
    only worth anything while the lens is walked from where they left it."""
    run = _three()
    for index in range(3):
        run.region_tuned(Look(100.0), "found", _view(index))
    run.begin_compromise()
    move = run.take({index: Look(80.0) for index in range(3)})
    assert move is not None and not move.autofocus
    assert abs(move.steps) == STEP


def test_the_region_on_screen_is_read_first_and_the_order_snakes():
    """The region read last at one probe is read first at the next: it is
    still on screen, and a pan saved is a quarter of a second."""
    run = _three()
    for index in range(3):
        run.region_tuned(Look(100.0), "found", _view(index))
    run.begin_compromise()
    first = run.order()
    assert first[0] == 2, "the one tuned last is the one on screen"
    second = run.order()
    assert second == list(reversed(first))


def test_a_region_with_nothing_in_it_is_left_out_of_the_compromise():
    run = _three()
    run.region_tuned(Look(100.0), "found", _view(0))
    run.region_tuned(Look(0.0), "nothing", _view(1))
    run.region_tuned(Look(50.0), "found", _view(2))
    assert run.begin_compromise()
    assert sorted(run.order()) == [0, 2]


def test_one_region_worth_focusing_on_is_nothing_to_compromise_between():
    run = _three()
    run.region_tuned(Look(100.0), "found", _view(0))
    run.region_tuned(Look(0.0), "nothing", _view(1))
    run.region_tuned(Look(0.0), "nothing", _view(2))
    assert not run.begin_compromise()
    report = run.report()
    assert report.outcome == "single"
    assert "Only region 1" in report.describe()


def _walked(
    curves,
    start: float,
    *,
    slack: int = 0,
    noise: float = 0.0,
    seed: int = 0,
    objective: str = "average",
    places=None,
    turn_back: float = TURN_BACK,
    depths: bool = False,
    hurry: "bool | int" = False,
    tuned=None,
    moves=None,
    watch=None,
) -> "tuple[Calibration, float]":
    """Run the compromise against modelled regions; answer it and where it ended.

    *curves* is one function per region, reading against focus position. The
    optics are carried between two faces of the driver, *slack* steps apart,
    so a reversal moves nothing until the play is taken up -- and it is where
    the optics end up that is answered, not where the steps say. *tuned* is
    what share of its true peak each region's fine tune read, if not all of
    it. *moves* is filled in, if a list is given, with every move the search
    asked for as (what it was doing, the steps, what it read there, the best
    it had read) -- for saying how a walk was paced as well as where it ended.
    """
    rng = np.random.default_rng(seed)
    regions = places or [Region(0.2 * n, 0.2, 0.05, 0.05) for n in range(len(curves))]
    run = Calibration(
        regions,
        STEP,
        objective=objective,
        turn_back=turn_back,
        depths=depths,
        hurry=HURRY_STRIDE if hurry is True else int(hurry or 1),
    )
    peaks = [max(curve(p) for p in range(-400, 400)) for curve in curves]
    for index, peak in enumerate(peaks):
        run.region_tuned(
            Look(peak * (tuned[index] if tuned else 1.0)), "found", _view(index)
        )
    run.begin_compromise()
    optics = driver = start
    for _ in range(1000):
        looks = {
            i: Look(curve(optics) * (1 + rng.normal(0, noise)) if noise else curve(optics))
            for i, curve in enumerate(curves)
        }
        move = run.take(looks)
        if move is None:
            return run, optics
        if moves is not None:
            moves.append((run.searching, move.steps, run.score, run.best))
        if watch is not None:
            watch(run)
        driver += move.steps
        optics = min(max(optics, driver), driver + slack)
    raise AssertionError("it has to stop on its own")


def _best_average(curves) -> "tuple[float, float]":
    """Where the average of the fractions is highest, by brute force, and its value."""
    grid = np.arange(-300.0, 300.0)
    peaks = [max(curve(p) for p in grid) for curve in curves]
    truth = [average_of_best([c(p) for c in curves], peaks) for p in grid]
    return float(grid[int(np.argmax(truth))]), max(truth)


def _hill(peak: float, width: float = 60.0, height: float = 100.0):
    return lambda p: height * float(np.exp(-((p - peak) / width) ** 2))


def test_it_walks_from_one_region_s_peak_to_where_the_average_is_best():
    curves = [_hill(-50, height=300.0), _hill(10, height=20.0), _hill(40)]
    run, optics = _walked(curves, start=40.0)
    best, value = _best_average(curves)
    assert abs(optics - best) <= STEP, f"ended at {optics}, best is {best}"
    report = run.report()
    assert report.outcome == "found"
    assert report.score == pytest.approx(value, abs=0.01)


@pytest.mark.parametrize("start", [-40.0, 40.0])
def test_a_broad_flat_top_is_met_in_the_middle_and_not_at_its_edge(start):
    """What was wrong with walking home the way a fine tune does. Two regions
    eighty steps apart add up to a top so flat that a per cent below its best
    is eighteen steps from the middle -- where one region is at 85% of its
    best and the other at 42%, against 64% each in the middle."""
    curves = [_hill(-40), _hill(40)]
    run, optics = _walked(curves, start=start)
    assert abs(optics) <= STEP, f"ended at {optics}, not in the middle"
    shares = [one.fraction for one in run.report().results]
    assert abs(shares[0] - shares[1]) < 0.12, shares


@pytest.mark.parametrize("start", [-60.0, 60.0])
def test_of_several_hills_it_finds_the_one_that_serves_all_of_them_best(start):
    """Regions further apart in focus than each is deep give the average a
    hill per region. Started on an outside region's own peak, a climb stands
    on that hill and calls it the compromise; the hill in the middle serves
    all three better."""
    curves = [_hill(-60, width=35), _hill(0, width=35), _hill(60, width=35)]
    # Hills this far apart are reached through valleys deeper than the
    # default turns back at; that is what lowering it is for.
    run, optics = _walked(curves, start=start, turn_back=0.3)
    best, value = _best_average(curves)
    assert abs(best) < STEP, "the middle hill is the best one"
    assert abs(optics - best) <= STEP, f"ended at {optics}, best is {best}"


def test_a_top_one_increment_wide_is_still_come_home_to():
    """Magnified onto a subject with no depth to it, one increment off the top
    loses a quarter of the reading, and a way home that needs two readings on
    the top to recognise it never recognises this one."""
    curves = [_hill(-9, width=10), _hill(9, width=10)]
    run, optics = _walked(curves, start=-9.0, slack=20)
    best, value = _best_average(curves)
    grid = np.arange(-300.0, 300.0)
    peaks = [max(curve(p) for p in grid) for curve in curves]
    achieved = average_of_best([curve(optics) for curve in curves], peaks)
    assert achieved >= value - 0.05, f"ended at {optics}"


@pytest.mark.parametrize("slack", [20, 50])
def test_play_in_the_gearing_does_not_shift_where_it_comes_home_to(slack):
    """The walk home counts steps only from the place the reading climbs back
    onto the top, which is the same place on the lens whatever the play was."""
    curves = [_hill(-40), _hill(40), _hill(10, height=30.0)]
    run, optics = _walked(curves, start=-40.0, slack=slack)
    best, _value = _best_average(curves)
    assert abs(optics - best) <= STEP + 2, f"ended at {optics}, best is {best}"
    assert run.report().outcome == "found"


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_grain_on_the_readings_does_not_send_it_off_the_top(seed):
    curves = [_hill(-40), _hill(40)]
    run, optics = _walked(curves, start=-40.0, noise=0.01, seed=seed, slack=20)
    best, value = _best_average(curves)
    assert abs(optics - best) <= STEP, f"ended at {optics}"


def _scenes(count: int, spread: float, widths=(12, 90), seed: int = 123):
    """Made-up scenes: regions at random depths, of random depths of their
    own, started on one region's peak as a search always is, through random
    play and grain. Yields (curves, start, slack, noise, seed)."""
    rng = np.random.default_rng(seed)
    for trial in range(count):
        regions = int(rng.integers(2, 6))
        peaks = rng.uniform(-spread, spread, regions)
        depth = rng.uniform(widths[0], widths[1], regions)
        start = float(peaks[int(rng.integers(regions))]) + float(rng.uniform(-3, 3))
        slack = int(rng.integers(0, 60))
        noise = float(rng.choice([0.0, 0.01, 0.02]))
        curves = [_hill(float(p), width=float(w)) for p, w in zip(peaks, depth)]
        yield curves, start, slack, noise, trial


def _loss(curves, optics: float, objective: str = "average") -> float:
    """What the search's answer cost the thing it was asked to maximise."""
    grid = np.arange(-500.0, 500.0)
    tops = [max(curve(p) for p in grid) for curve in curves]
    value = lambda p: combine([c(p) / t for c, t in zip(curves, tops)], objective)  # noqa: E731
    return max(value(p) for p in grid) - value(optics)


def test_across_many_made_up_scenes_it_gives_away_almost_nothing():
    """Regions anywhere in 240 steps of focus, some of them further apart
    than they are deep, with the walks let reach as far as they like: the
    search's own arithmetic, held to what it can do. What is measured is
    what it cost the thing it was asked to maximise -- the average share of
    the regions' best, against the best any focus position gives."""
    # Within 120 steps of each other and at least 25 deep: regions further
    # apart than that, or narrower, have a flat valley of nothing between
    # them -- every other region reads nought where one is sharp -- and
    # nothing on the way says there is a hill beyond it. No walk that turns
    # back, however late, sees across one; only a sweep of everything could.
    losses = []
    for curves, start, slack, noise, seed in _scenes(60, spread=60.0, widths=(25, 90)):
        run, optics = _walked(
            curves, start=start, slack=slack, noise=noise, seed=seed, turn_back=0.3
        )
        assert run.report().outcome == "found"
        losses.append(_loss(curves, optics))
    assert np.mean(losses) < 0.005, np.mean(losses)
    assert max(losses) < 0.05, max(losses)


def test_on_a_frame_of_film_turning_back_early_costs_nothing_but_saves_time():
    """What the turn-back share is for. Regions on one frame of film are
    within a few depths of field of each other, and there the compromise is
    found as well by walks that turn back at four fifths of their best as by
    walks that go on into the soft stretch -- in far fewer probes."""
    early, late = [], []
    for curves, start, slack, noise, seed in _scenes(40, spread=30.0, widths=(25, 90)):
        for turn_back, kept in ((TURN_BACK, early), (0.3, late)):
            run, optics = _walked(
                curves, start=start, slack=slack, noise=noise, seed=seed, turn_back=turn_back
            )
            kept.append((_loss(curves, optics), run.probes))
    assert np.mean([loss for loss, _probes in early]) < 0.004
    assert max(loss for loss, _probes in early) < 0.03
    assert np.mean([probes for _loss, probes in early]) < 0.8 * np.mean(
        [probes for _loss, probes in late]
    )


@pytest.mark.parametrize("turn_back", [0.8, 0.6])
def test_no_walk_goes_further_into_the_soft_stretch_than_it_is_let(turn_back):
    """Once the number the search climbs has fallen below the turn-back share
    of the best of the walk it is on, the walk turns -- waiting a few
    increments at most for a region still climbing -- rather than walking on
    through readings nothing is going to come of."""
    from scanny.ui.regions import _CLIMB_PATIENCE

    for curves, start, slack, noise, seed in _scenes(20, spread=30.0, widths=(25, 90)):
        run, _optics = _walked(
            curves, start=start, slack=slack, noise=noise, seed=seed, turn_back=turn_back
        )
        history = run.report().history
        for leg in ("out", "across"):
            scores = [combined for stage, _s, combined in history if stage == leg]
            best, soft = 0.0, 0
            for score in scores:
                best = max(best, score)
                soft = soft + 1 if score < turn_back * best else 0
                assert soft <= _CLIMB_PATIENCE + 1, (leg, scores)

def test_the_worst_region_is_the_least_of_the_shares():
    assert combine([0.9, 0.5, 0.7], "worst") == pytest.approx(0.5)
    assert combine([0.9, 0.5, 0.7], "average") == pytest.approx(0.7)
    assert combine([], "worst") == 0.0


def test_aiming_for_the_worst_region_sacrifices_none_of_them():
    """A sharp region and a broad one: the best average leans towards the
    sharp one's peak, which a broad region can afford to lose more of; the
    best worst region leans the other way, until neither is softer than it
    has to be."""
    curves = [_hill(-30, width=20), _hill(30, width=60)]
    grid = np.arange(-300.0, 300.0)
    tops = [max(curve(p) for p in grid) for curve in curves]
    softest = lambda p: min(c(p) / t for c, t in zip(curves, tops))  # noqa: E731
    # Where the two curves cross is a point, and with no play the lens only
    # stops a whole number of increments from where it started: the best it
    # can be asked for is the best of those.
    reachable = 30.0 + STEP * np.arange(-50, 50)
    best = max(softest(p) for p in reachable)
    by_average, at_average = _walked(curves, start=30.0)
    by_worst, at_worst = _walked(curves, start=30.0, objective="worst")
    assert softest(at_worst) >= best - 0.01
    assert softest(at_worst) > softest(at_average) + 0.05
    report = by_worst.report()
    assert report.objective == "worst"
    assert "every region at least" in report.describe()


def test_across_many_made_up_scenes_the_worst_region_is_looked_after_too():
    """The same made-up scenes, by the other objective. Its top is a point
    where two regions' curves cross rather than a hill, so an increment off
    it -- which the play can make unavoidable -- costs more than it does on
    the average; this is held to what that allows."""
    losses = []
    for curves, start, slack, noise, seed in _scenes(40, spread=120.0, seed=7):
        run, optics = _walked(
            curves,
            start=start,
            slack=slack,
            noise=noise,
            seed=seed,
            objective="worst",
            turn_back=0.3,
        )
        losses.append(_loss(curves, optics, "worst"))
    assert np.mean(losses) < 0.015, np.mean(losses)
    assert max(losses) < 0.1, max(losses)


# -- how far apart in focus the regions are ----------------------------------


@pytest.mark.parametrize("slack", [0, 30])
def test_the_walk_across_says_how_far_apart_in_focus_the_regions_are(slack):
    """What levelling the film needs: not the compromise but the depths. The
    walk across crosses every region's peak in one direction, so where each
    peaked on it is honest against the others, however much play was taken
    up at its start."""
    curves = [_hill(-40), _hill(10), _hill(40)]
    run, _optics = _walked(
        curves, start=40.0, slack=slack, noise=0.01, seed=3, depths=True
    )
    report = run.report()
    depths = [one.depth for one in report.results]
    assert depths == pytest.approx([0.0, 50.0, 80.0], abs=3.0)
    assert all(one.doubt < 5 for one in report.results)
    assert not any(one.edge for one in report.results)
    import re

    assert re.fullmatch(
        r"Depth, in drive steps: 1 nearest,  2 \+5\d ±\d,  3 \+[78]\d ±\d",
        report.ordering(),
    ), report.ordering()


@pytest.mark.parametrize("slack", [0, 30])
def test_the_walk_across_also_says_where_the_compromise_was_placed(slack):
    """The depths say where each region focuses; this says where the one focus
    position they all ended at sits among them, in the same steps counted from
    the same nearest region -- the plane the compromise brings into focus, and
    the thing the film's shape is to be judged against. It is read off the same
    walk, as the top of the very number the search climbs, and so lands where
    the search went home to and climbed from: within an increment of where the
    lens finally stood, play taken up or not."""
    curves = [_hill(-40), _hill(10), _hill(40)]
    run, optics = _walked(
        curves, start=40.0, slack=slack, noise=0.01, seed=3, depths=True
    )
    report = run.report()
    # The nearest region's peak is the zero the depths are counted from, and
    # it is the hill at -40; where the lens ended is that much further on.
    ended = optics - (-40)
    assert report.focus_depth == pytest.approx(ended, abs=STEP)
    depths = [one.depth for one in report.results]
    assert min(depths) < report.focus_depth < max(depths), (
        "a compromise between them is somewhere among them"
    )


def test_where_the_compromise_was_placed_is_not_said_without_a_walk_across():
    """No walk across, no axis to say it on: the depths and the focus plane
    are read off that one walk or not at all."""
    curves = [_hill(-40), _hill(10), _hill(40)]
    run, _optics = _walked(curves, start=40.0, depths=False)
    report = run.report()
    assert report.focus_depth is None
    assert all(one.depth is None for one in report.results)


def test_a_region_a_little_outside_the_others_is_still_measured():
    """A region the compromise can do without is still one the film has to be
    levelled by, so the walk across waits a few increments for its peak."""
    curves = [_hill(-40), _hill(10), _hill(75, width=40)]
    run, _optics = _walked(curves, start=10.0, depths=True)
    results = run.report().results
    assert not any(one.edge for one in results)
    assert results[2].depth - results[0].depth == pytest.approx(115.0, abs=4.0)


def test_a_region_far_outside_the_others_is_not_chased_and_says_so():
    """Not through the whole soft stretch, which is minutes of walking: its
    depth is reported as a bound -- at least so far -- rather than measured,
    and it is left out of the film's shape rather than bending it."""
    curves = [_hill(-40), _hill(10), _hill(320, width=40)]
    run, _optics = _walked(curves, start=10.0, depths=True)
    report = run.report()
    far = report.results[2]
    assert far.depth is None or far.edge
    if far.depth is not None:
        assert "at least" in describe_depth(far)


def _levelled(depth_at, orientation=None, corners=None):
    """A report of four regions at the corners, their depths given by *depth_at*."""
    corners = corners or [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    raw = [depth_at(x, y) for x, y in corners]
    results = tuple(
        RegionResult(
            number + 1,
            Region(x - 0.05, y - 0.05, 0.1, 0.1),
            Look(100.0),
            "found",
            Look(90.0),
            depth=value - min(raw),
            doubt=1.0,
        )
        for number, ((x, y), value) in enumerate(zip(corners, raw))
    )
    return CalibrationReport(results, 0.9, "found").tilt(orientation)


def test_the_regions_depths_are_fitted_with_a_plane_the_film_can_be_levelled_by():
    tilt = _levelled(lambda x, y: 100 * x + 30 * y)
    assert tilt.across == pytest.approx(100.0, abs=0.5)
    assert tilt.down == pytest.approx(30.0, abs=0.5)
    assert tilt.off_the_plane == pytest.approx(0.0, abs=0.1)
    said = tilt.describe()
    assert "right edge focuses 100 steps further than the left edge" in said
    assert "bottom edge focuses 30 steps further than the top edge" in said


def test_the_lean_is_said_the_way_the_picture_is_shown():
    """Levelling is done looking at the picture as shown, so left and right
    are the screen's, not the sensor's."""
    mirrored = _levelled(lambda x, y: 100 * x + 30 * y, Orientation(mirrored=True))
    assert mirrored.across == pytest.approx(-100.0, abs=0.5)
    assert mirrored.down == pytest.approx(30.0, abs=0.5)
    # A quarter turn right puts the frame's top on the screen's right.
    turned = _levelled(lambda x, y: 100 * x + 30 * y, Orientation(turns=1))
    assert turned.across == pytest.approx(-30.0, abs=0.5)
    assert turned.down == pytest.approx(100.0, abs=0.5)


def test_what_levelling_cannot_take_out_is_said_separately():
    """Four corners on a tilted plane and a middle standing proud of it: the
    tilt is still the plane's, and the middle is what is left over -- curl,
    which no amount of levelling removes."""
    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.5, 0.5)]
    tilt = _levelled(
        lambda x, y: 60 * x + (20 if (x, y) == (0.5, 0.5) else 0), corners=corners
    )
    assert tilt.across == pytest.approx(60.0, abs=3.0)
    assert tilt.off_the_plane > 5.0


def test_a_lean_inside_the_doubt_is_called_level():
    tilt = _levelled(lambda x, y: 1.5 * x)
    assert "level" in tilt.describe().splitlines()[0]


def test_a_lean_needs_three_regions_not_in_a_line():
    assert _levelled(lambda x, y: 50 * x, corners=[(0.2, 0.5), (0.8, 0.5)]) is None
    assert _levelled(lambda x, y: 50 * x, corners=[(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)]) is None


def test_the_report_says_what_each_region_gave_up():
    curves = [_hill(-40), _hill(40)]
    run, _optics = _walked(curves, start=-40.0)
    report = run.report()
    assert all(one.fraction is not None for one in report.results)
    assert all(0.5 < one.fraction <= 1.01 for one in report.results)
    assert report.worst in report.results
    line = report.describe()
    assert "Compromise" in line and "region" in line


def test_a_region_read_higher_on_the_walk_has_that_for_its_best():
    """What a real calibration showed: three regions of four read up to 13%
    over "their best" at the compromise, because the best they were measured
    against was what their fine tunes had walked back to and stood on -- a
    little off each region's top. The walk across crosses every region's
    peak, so whatever it reads higher than a region's best is that region's
    best from then on, and no share of anything is over the whole of it."""
    curves = [_hill(-40), _hill(10), _hill(40)]
    grid = np.arange(-400, 400)
    peaks = [max(curve(p) for p in grid) for curve in curves]
    run, _optics = _walked(curves, start=40.0, tuned=[0.93, 1.0, 0.9])
    report = run.report()
    first, second, third = report.results
    assert first.tuned == pytest.approx(0.93 * peaks[0])
    assert first.best.reading == pytest.approx(peaks[0], rel=0.01)
    assert first.bettered == pytest.approx(1 / 0.93 - 1, abs=0.01)
    assert third.bettered == pytest.approx(1 / 0.9 - 1, abs=0.01)
    assert second.bettered is None, "read at its best by its fine tune already"
    assert all(one.fraction <= 1.0 for one in report.results)
    for _stage, shares, _combined in report.history:
        assert max(shares) <= 1.0
    assert report.score == pytest.approx(np.mean([one.fraction for one in report.results]))
    assert "read on the walk for the compromise" in summarise(first, 1)
    assert "7% above" in summarise(first, 1)


def test_a_best_read_low_does_not_move_the_compromise():
    """Worse than a report that says 108%. Each region's say in the compromise
    is its share of its own best, so a best read low gives that region more
    say than it should have -- and aiming for the best worst region, the one
    looked after is not the one that needs it. With the best put right on the
    walk across, the compromise is where it would have been."""
    curves = [_hill(-30, width=20), _hill(30, width=60)]
    grid = np.arange(-300.0, 300.0)
    tops = [max(curve(p) for p in grid) for curve in curves]
    softest = lambda p: min(c(p) / t for c, t in zip(curves, tops))  # noqa: E731
    _run, exact = _walked(curves, start=30.0, objective="worst")
    # Measured against a best a fifth short, the sharp region looked well
    # enough off to be given up for the broad one: it stood an increment
    # the broad one's way, the sharp one at 45% of its best rather than 53%.
    _run, low = _walked(curves, start=30.0, objective="worst", tuned=[0.8, 1.0])
    assert low == exact
    assert softest(low) == pytest.approx(softest(exact))


def test_a_calibration_stopped_part_way_keeps_the_peaks_it_found():
    run = _three()
    run.region_tuned(Look(100.0, _image(np.full((4, 4), 9.0))), "found", _view(0))
    report = run.report(stopped=True)
    assert report.stopped
    assert report.results[0].best is not None and report.results[0].best.picture
    assert report.results[1].best is None
    assert all(one.compromise is None for one in report.results)
    assert "stopped" in report.describe()


def test_regions_are_coloured_by_what_the_compromise_left_them():
    region = Region(0.1, 0.1, 0.1, 0.1)
    near = RegionResult(1, region, Look(100.0), "found", Look(97.0))
    fair = RegionResult(2, region, Look(100.0), "found", Look(88.0))
    poor = RegionResult(3, region, Look(100.0), "found", Look(60.0))
    empty = RegionResult(4, region, Look(0.0), "nothing", None)
    colours = {colour_for(one) for one in (near, fair, poor, empty, None)}
    assert len(colours) == 5, "each state its own colour"


def test_a_region_s_tooltip_has_both_numbers():
    result = RegionResult(2, Region(0.1, 0.1, 0.1, 0.1), Look(150.0), "found", Look(120.0))
    said = summarise(result, 2)
    assert "150" in said and "120" in said and "80%" in said


# -- the window's side -------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    made = QApplication.instance() or QApplication([])
    QApplication.setOrganizationName("scanny-tests")
    QApplication.setApplicationName("scanny-tests")
    QSettings().clear()
    yield made
    QSettings().clear()


@pytest.fixture
def window(app, monkeypatch):
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    QSettings().clear()
    made = mw.MainWindow()
    made.show()
    QApplication.processEvents()
    yield made
    made.hide()
    made.deleteLater()
    QApplication.processEvents()


def _whole_frame() -> LiveViewFrame:
    return _frame(np.full((TALL, WIDE), 128.0), (3008, 2008, 6016, 4016), (3008, 2008))


def _showing(window, frame: "LiveViewFrame | None" = None) -> None:
    """Give the picture a frame, so its gestures have somewhere to land."""
    frame = frame or _whole_frame()
    window.view.show_frame(frame, QImage.fromData(frame.jpeg, "JPG"))
    window.view.resize(640, 400)
    window.view.repaint()


def _at(view, x: float, y: float) -> QPoint:
    return QPoint(
        view._target.x() + int(x * view._target.width()),
        view._target.y() + int(y * view._target.height()),
    )


def _ctrl_drag(view, one, other) -> None:
    ctrl = Qt.KeyboardModifier.ControlModifier
    left = Qt.MouseButton.LeftButton
    start, end = _at(view, *one), _at(view, *other)
    for kind, where in (
        (QMouseEvent.Type.MouseButtonPress, start),
        (QMouseEvent.Type.MouseMove, end),
        (QMouseEvent.Type.MouseButtonRelease, end),
    ):
        event = QMouseEvent(
            kind, QPointF(where), QPointF(view.mapToGlobal(where)), left,
            left if kind != QMouseEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton,
            ctrl,
        )
        {
            QMouseEvent.Type.MouseButtonPress: view.mousePressEvent,
            QMouseEvent.Type.MouseMove: view.mouseMoveEvent,
            QMouseEvent.Type.MouseButtonRelease: view.mouseReleaseEvent,
        }[kind](event)


def _ctrl_click(view, x: float, y: float) -> None:
    _ctrl_drag(view, (x, y), (x, y))


def test_ctrl_dragging_the_picture_draws_a_numbered_region(window):
    _showing(window)
    _ctrl_drag(window.view, (0.2, 0.3), (0.4, 0.5))
    assert len(window._regions) == 1
    assert window._regions[0] == pytest.approx((0.2, 0.3, 0.2, 0.2), abs=0.01)
    drawn = window.view._regions
    assert len(drawn) == 1 and drawn[0][1] == 1, "drawn, and numbered one"


def test_ctrl_clicking_inside_a_region_takes_it_away(window):
    _showing(window)
    _ctrl_drag(window.view, (0.2, 0.3), (0.4, 0.5))
    _ctrl_drag(window.view, (0.6, 0.3), (0.8, 0.5))
    _ctrl_click(window.view, 0.3, 0.4)
    assert len(window._regions) == 1
    assert window._regions[0][0] == pytest.approx(0.6, abs=0.01)


def test_a_plain_drag_still_magnifies_and_draws_no_region(window):
    _showing(window)
    zoomed = []
    window.view.regionSelected.connect(lambda *rect: zoomed.append(rect))
    start, end = _at(window.view, 0.2, 0.3), _at(window.view, 0.4, 0.5)
    left = Qt.MouseButton.LeftButton
    none = Qt.KeyboardModifier.NoModifier
    window.view.mousePressEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(start), QPointF(start), left, left, none))
    window.view.mouseMoveEvent(QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(end), QPointF(end), left, left, none))
    window.view.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease, QPointF(end), QPointF(end), left,
        Qt.MouseButton.NoButton, none))
    assert window._regions == []
    assert zoomed, "an ordinary drag is still a magnification"


def test_a_region_drawn_while_magnified_is_kept_where_the_sensor_is(window):
    """So it stays on its subject when the view goes back out."""
    magnified = _frame(np.full((TALL, WIDE), 128.0), (1504, 3012, 1504, 1004), (1504, 3012))
    _showing(window, magnified)
    _ctrl_drag(window.view, (0.25, 0.25), (0.75, 0.75))
    # The middle half of a quarter-frame view centred at (0.25, 0.75).
    assert window._regions[0] == pytest.approx((0.1875, 0.6875, 0.125, 0.125), abs=0.01)


def test_it_will_not_take_more_regions_than_it_calibrates(window):
    for index in range(MAX_REGIONS + 2):
        window._on_region_drawn(0.05 + 0.13 * index, 0.1, 0.1, 0.1)
    assert len(window._regions) == MAX_REGIONS


def test_calibrating_takes_two_regions_and_sends_the_minimum_step(window):
    asked = []
    window.requestCalibration.connect(lambda step, aim: asked.append((step, aim)))
    window._focus_steps["minimum"].setValue(18)
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    assert not window.calibrate_button.isEnabled()
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    assert window.calibrate_button.isEnabled()
    window.calibrate_button.click()
    assert asked == [(18, "average")]


def test_what_to_aim_for_is_chosen_before_calibrating_and_remembered(window):
    asked = []
    window.requestCalibration.connect(lambda step, aim: asked.append(aim))
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    window.regions_objective.setCurrentIndex(window.regions_objective.findData("worst"))
    window.calibrate_button.click()
    assert asked == ["worst"]
    assert QSettings().value("regions/objective") == "worst"
    # Settled before it starts: the search climbs it, and a report made by one
    # cannot be read as if it had been made by the other.
    window._on_calibration_changed(True)
    assert not window.regions_objective.isEnabled()
    window._on_calibration_changed(False)
    assert window.regions_objective.isEnabled()


def test_the_regions_are_sent_to_the_worker_and_a_change_forgets_the_report(window):
    told = []
    window.requestFocusRegions.connect(told.append)
    window._report = object()
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    assert told[-1] == [(0.1, 0.1, 0.1, 0.1)]
    assert window._report is None


def test_the_regions_are_remembered_for_next_time(app, monkeypatch):
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    QSettings().clear()
    first = mw.MainWindow()
    first._on_region_drawn(0.1, 0.2, 0.1, 0.1)
    first._on_region_drawn(0.5, 0.6, 0.2, 0.1)
    first.deleteLater()
    second = mw.MainWindow()
    assert second._regions == pytest.approx([(0.1, 0.2, 0.1, 0.1), (0.5, 0.6, 0.2, 0.1)])
    second.deleteLater()


def test_the_shape_of_a_calibration_is_settled_while_it_runs(window):
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    window._on_calibration_changed(True)
    assert window.calibrate_button.text() == "Stop"
    assert not window.regions_clear.isEnabled()
    window._on_calibration_changed(False)
    assert window.calibrate_button.text() == "Calibrate"


def _report(inverted_left: bool = True) -> CalibrationReport:
    """A report whose pictures are black on the left and white on the right."""
    pixels = np.zeros((10, 20))
    pixels[:, 10:] = 255.0
    picture = _image(pixels)
    region = Region(0.1, 0.1, 0.1, 0.1)
    return CalibrationReport(
        results=(
            RegionResult(1, region, Look(150.0, picture), "found", Look(140.0, picture)),
            RegionResult(2, region, Look(60.0, picture), "found", Look(51.0, picture)),
        ),
        score=0.89,
        outcome="found",
        probes=31,
    )


def test_a_report_colours_the_regions_and_opens_a_window_of_pictures(window):
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    window._on_calibration_report(_report())
    assert window.report_button.isEnabled()
    assert "89%" in window.regions_result.text()
    window.report_button.click()
    dialog = window._report_dialog
    assert dialog is not None and dialog.isVisible()
    # Two regions, a best and a compromise each.
    assert len(dialog._pictures) == 4
    assert all(not label.pixmap().isNull() for label in dialog._pictures)


def test_no_line_of_the_report_is_cut_off(window):
    """A wrapped label in a grid is given the height its text needs at the
    wrong width, and the heading over the pictures came out half a line tall,
    clipped top and bottom. Every wrapped line has to get the room its text
    takes at the width it was actually given."""
    from PySide6.QtWidgets import QLabel

    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    window._on_calibration_report(_report())
    window.report_button.click()
    dialog = window._report_dialog
    QApplication.processEvents()
    wrapped = [
        label
        for label in dialog._content.findChildren(QLabel)
        if label.wordWrap() and label.text()
    ]
    assert wrapped, "the headings and captions wrap"
    for label in wrapped:
        assert label.height() >= label.heightForWidth(label.width()), label.text()


def test_the_report_says_how_far_apart_the_regions_are_and_how_the_frame_leans(window):
    from PySide6.QtWidgets import QLabel

    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8)]
    for x, y in corners:
        window._on_region_drawn(x - 0.05, y - 0.05, 0.1, 0.1)
    picture = _image(np.full((10, 10), 128.0))
    results = tuple(
        RegionResult(
            number + 1,
            Region(x - 0.05, y - 0.05, 0.1, 0.1),
            Look(100.0, picture),
            "found",
            Look(90.0, picture),
            depth=depth,
            doubt=1.0,
        )
        for number, ((x, y), depth) in enumerate(zip(corners, (0.0, 60.0, 20.0)))
    )
    window._on_calibration_report(CalibrationReport(results, 0.9, "found"))
    assert "1 nearest" in window.regions_result.text()
    window.report_button.click()
    said = " ".join(
        label.text() for label in window._report_dialog.findChildren(QLabel)
    )
    assert "Region 2: 60 steps further" in said
    assert "right edge focuses 100 steps further than the left edge" in said
    assert "bottom edge focuses 33 steps further than the top edge" in said
    # Turned, the lean is said again the way the picture now is.
    window._flip_view(False)
    said = " ".join(
        label.text() for label in window._report_dialog.findChildren(QLabel)
    )
    assert "left edge focuses 100 steps further than the right edge" in said


def test_the_report_shows_the_pictures_the_way_the_view_is_shown(window):
    """Mirrored, the white half of every picture is on the left -- the way the
    picture on screen is being looked at, not the way the sensor sees it."""
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    window._on_calibration_report(_report())
    window.report_button.click()
    dialog = window._report_dialog

    def left_is_white() -> bool:
        shown = dialog._pictures[0].pixmap().toImage()
        return shown.pixelColor(1, shown.height() // 2).red() > 200

    assert not left_is_white()
    window._flip_view(False)
    assert left_is_white(), "the open report follows the view"
    window._on_invert_toggled(True)
    assert not left_is_white(), "and inverting it inverts the pictures too"


# -- the whole thing, through the worker -------------------------------------


WIDE, TALL = 160, 120
FRAME = (6016, 4016)
MAGNIFICATION = {0: 1.0, 2: 2.35, 3: 3.13, 4: 4.7, 5: 6.27, 6: 9.4, 7: 18.8}


def _frame(pixels, crop, af) -> LiveViewFrame:
    """One frame, reporting the crop the camera would be showing."""
    data = QByteArray()
    sink = QBuffer(data)
    sink.open(QBuffer.OpenModeFlag.WriteOnly)
    _image(pixels).save(sink, "JPG", 95)
    cx, cy, w, h = crop
    return LiveViewFrame(
        jpeg=bytes(data.data()), width=pixels.shape[1], height=pixels.shape[0],
        image_width=FRAME[0], image_height=FRAME[1],
        crop_width=w, crop_height=h, crop_center_x=cx, crop_center_y=cy,
        af_width=324, af_height=270, af_x=af[0], af_y=af[1],
    )


class _Rig:
    """A body over a curled frame of film: places at different depths.

    What the picture shows depends on where the camera is aimed and how far
    it is magnified -- the view is a crop of the frame centred on the focus
    point -- and it is softened by how far the optics are from the depth of
    whichever place the view is over. The gearing has play in it, so the
    first steps after a reversal move nothing; live view runs a frame behind
    the lens; the picture has grain on it; and autofocus gets close to the
    place it is aimed at rather than onto it.
    """

    live_view_active = True
    exposure_preview = True

    def __init__(self, places, start: int = 3000, softness: float = 30.0,
                 slack: int = 0, noise: float = 1.5, af_error: int = 12,
                 lag: int = 1) -> None:
        #: (x, y, best) each: where on the frame, and where on the travel.
        self.places = list(places)
        self.optics = float(start)
        self.position = float(start)
        self.softness = softness
        self.slack = slack
        self.noise = noise
        self.af_error = af_error
        self.lag = lag
        self.zoom = 0
        self.af = (FRAME[0] // 2, FRAME[1] // 2)
        self.zooms: "list[int]" = []
        self.aimed: "list[tuple[int, int]]" = []
        self.focused = 0
        self._history = [self.optics] * (lag + 1)
        self._textures: "dict[tuple, np.ndarray]" = {}
        self._rng = np.random.default_rng(3)
        #: After how many more frames the body ends live view by itself, as a
        #: D750 does after its own monitor-off delay; None for never. And how
        #: often it has been started again since.
        self.ends_after: "int | None" = None
        self.ended = False
        self.restarts = 0
        #: How many attempts at starting live view again it refuses, as a body
        #: still putting its mirror down does, and how many commands after a
        #: restart it answers busy to.
        self.refuses_restart = 0
        self.busy_after_restart = 0

    # -- the camera's side ---------------------------------------------------

    def _in_live_view(self, what: str) -> None:
        """Refuse, as the body does, what cannot be done out of live view."""
        if self.ended:
            raise CameraError(f"{what} can only be done in live view")
        if self.busy_after_restart:
            self.busy_after_restart -= 1
            raise CameraError(f"could not {what} (0x2019)")

    def drive_focus(self, steps: int) -> bool:
        self._in_live_view("drive focus")
        self.position += int(steps)
        # The optics are carried between two faces of the driver.
        self.optics = min(max(self.optics, self.position), self.position + self.slack)
        return True

    def set_zoom_level(self, level: int) -> None:
        self._in_live_view("zoom")
        self.zoom = int(level)
        self.zooms.append(int(level))

    def zoom_level(self) -> int:
        return self.zoom

    def set_af_area(self, x: int, y: int) -> None:
        self._in_live_view("move the focus point")
        self.af = (int(x), int(y))
        self.aimed.append(self.af)

    def autofocus(self, timeout: float = 8.0) -> bool:
        self._in_live_view("autofocus")
        self.focused += 1
        self.optics = float(self._place()[2] + self.af_error)
        self.position = self.optics - self.slack // 2
        return True

    def live_view_frame(self) -> LiveViewFrame:
        if self.ends_after is not None:
            self.ends_after -= 1
            if self.ends_after <= 0:
                self.ended, self.ends_after = True, None
        if self.ended:
            raise CameraError("live view has ended")
        self._history.append(self.optics)
        shown = self._history[-(self.lag + 1)]
        crop = self._crop()
        pixels = self.picture(crop, shown) + self._rng.normal(0, self.noise, (TALL, WIDE))
        return _frame(pixels, crop, self.af)

    def stop_live_view(self) -> None:
        self.live_view_active = False

    def restart_live_view(self) -> None:
        """Live view again -- at the whole frame, as a body starts it."""
        if self.refuses_restart:
            self.refuses_restart -= 1
            raise CameraError("camera will not enter live view: busy")
        self.restarts += 1
        self.ended = False
        self.live_view_active = True
        self.zoom = 0

    def set_setting(self, name, value) -> None:
        pass

    def settings(self) -> list:
        return []

    def set_exposure_preview(self, enabled: bool) -> None:
        self.exposure_preview = enabled

    # -- the scene -----------------------------------------------------------

    def _crop(self, zoom: "int | None" = None, af=None):
        magnification = MAGNIFICATION[self.zoom if zoom is None else zoom]
        ax, ay = self.af if af is None else af
        w, h = int(FRAME[0] / magnification), int(FRAME[1] / magnification)
        cx = min(max(ax, w // 2), FRAME[0] - w // 2)
        cy = min(max(ay, h // 2), FRAME[1] - h // 2)
        return cx, cy, w, h

    def _place(self, crop=None):
        cx, cy, _w, _h = crop or self._crop()
        here = (cx / FRAME[0], cy / FRAME[1])
        return min(self.places, key=lambda p: (p[0] - here[0]) ** 2 + (p[1] - here[1]) ** 2)

    def picture(self, crop, optics: float) -> np.ndarray:
        """The view's own detail, softened by how far focus is off its depth.

        The blur grows as the square of the distance near focus and in
        proportion to it further out, which gives the reading the rounded top
        and long tails a real lens does. Blur in proportion to the distance
        all the way in makes the top a cusp, which no lens has and which
        makes every increment the gearing's play leaves the lattice off by
        look like a failure to find focus.
        """
        base = self._textures.get(crop)
        if base is None:
            base = _texture(TALL, WIDE, seed=abs(hash(crop)) % 9973)
            self._textures[crop] = base
        away = (optics - self._place(crop)[2]) / self.softness
        return _blur(base, min(2.0 * (np.hypot(1.0, away) - 1.0), 7.99))


def _regions_round(places, side: float = 0.06):
    return [(x - side / 2, y - side / 2, side, side) for x, y, _best in places]


@pytest.fixture
def worker():
    from scanny.ui.worker import CameraWorker

    return CameraWorker()


def _calibrated(
    worker, rig, regions, *, frames: int = 2, limit: int = 20000, depths: bool = False
):
    worker._camera = rig
    if frames:
        worker.set_integration(True, frames)
    for _ in range(8):
        worker._grab()
    worker.set_focus_regions(regions)
    reports = []
    worker.calibrationReady.connect(lambda report: reports.append(report))
    worker.start_calibration(STEP, "average", 80, depths)
    for _ in range(limit):
        if worker._calibration is None:
            break
        worker._grab()
    assert worker._calibration is None, "the calibration has to finish on its own"
    assert reports and reports[-1] is not None
    return reports[-1]


def _true_average(rig, report, position: float) -> float:
    """What the model says the average of each region's fraction is, there.

    Read without grain, through each region's own view, against the best the
    model can give that region anywhere -- the thing the walk was trying to
    find, worked out by brute force.
    """
    shares = []
    for result in report.results:
        crop = rig._crop(7 if result.region.w < 0.05 else 6, _af_for(result.region))
        shown = result.region.seen_in(_normalised(crop))
        read = lambda p: measure(_image(rig.picture(crop, p)), shown)  # noqa: E731
        best = rig._place(crop)[2]
        shares.append(read(position) / read(best))
    return float(np.mean(shares))


def _af_for(region: Region) -> "tuple[int, int]":
    return (
        int(round((region.x + region.w / 2) * FRAME[0])),
        int(round((region.y + region.h / 2) * FRAME[1])),
    )


def _normalised(crop):
    cx, cy, w, h = crop
    return ((cx - w / 2) / FRAME[0], (cy - h / 2) / FRAME[1], w / FRAME[0], h / FRAME[1])


PLACES = [(0.2, 0.25, 3000), (0.8, 0.3, 3060), (0.5, 0.75, 3120)]


@pytest.mark.parametrize("slack", [0, 40])
def test_three_regions_at_three_depths_meet_at_the_best_compromise(worker, slack):
    """The whole of what was asked for. Three places across a curled frame,
    sixty steps apart in focus each: every one fine tuned on its own, then
    one focus position walked to that does best by all three -- through the
    play in the gearing, and panning between regions the camera can only
    show one of at a time."""
    rig = _Rig(PLACES, slack=slack)
    report = _calibrated(worker, rig, _regions_round(PLACES))

    assert report.outcome == "found", report.describe()
    assert all(one.usable for one in report.results)
    # Every region was fine tuned on its own, magnified onto it.
    assert rig.focused == 3, "one autofocus per region, and none for the compromise"
    assert 6 in rig.zooms

    grid = np.arange(2900, 3221, 10)
    truth = [_true_average(rig, report, float(p)) for p in grid]
    achieved = _true_average(rig, report, rig.optics)
    assert achieved >= max(truth) - 0.03, (
        f"it stood at {rig.optics:.0f} worth {achieved:.3f}; the best there is "
        f"{max(truth):.3f} at {grid[int(np.argmax(truth))]}"
    )
    # And the middle one is where it should be: between the other two.
    assert 3000 < rig.optics < 3120


def test_each_region_s_best_is_remembered_with_its_picture(worker):
    rig = _Rig(PLACES)
    report = _calibrated(worker, rig, _regions_round(PLACES))
    for result in report.results:
        assert result.best.reading > 0.0
        assert result.best.picture is not None and not result.best.picture.isNull()
        assert result.compromise.picture is not None
        # The same view both times, so the same size of picture.
        assert result.best.picture.size() == result.compromise.picture.size()
        assert 0.0 < result.fraction <= 1.1
    # These three are further apart in focus than each is deep, so the best
    # average stands on the middle one and the outer two pay for it.
    middle = report.results[1]
    assert middle.fraction > 0.9
    assert report.worst is not middle
    assert report.score == pytest.approx(
        np.mean([one.fraction for one in report.results]), abs=1e-6
    )


@pytest.mark.parametrize("slack", [0, 40])
def test_each_region_s_best_is_the_best_its_fine_tune_saw(worker, slack):
    """Not what it walked back to and stood on, which it no longer does: the
    lens goes somewhere else next anyway, and walking back lands a little off
    the top -- which then became the yardstick for every share of the region.
    And every reading it and the compromise took is in the report, with how
    long after the lens last moved it was taken."""
    rig = _Rig(PLACES, slack=slack)
    report = _calibrated(worker, rig, _regions_round(PLACES))
    assert report.outcome == "found"
    for result in report.results:
        walk = [
            one.value
            for one in report.readings
            if one.region == result.number and one.stage == "tune"
        ]
        # The first is the one taken before its autofocus, somewhere else.
        assert result.tuned == pytest.approx(max(walk[1:]))
        assert result.best.reading >= result.tuned
        assert result.best.picture is not None
        assert 0.0 < result.fraction <= 1.0
    stages = {one.stage for one in report.readings}
    assert {"tune", "out", "across"} <= stages
    times = [one.at for one in report.readings]
    assert times == sorted(times)
    known = [one.after for one in report.readings if not np.isnan(one.after)]
    assert len(known) > 0.9 * len(report.readings)
    assert all(after >= 0.0 for after in known)
    said = "\n".join(report.log)
    assert "the compromise walks on from there" in said, "it did not walk back"
    assert "this one" in said, "each probe of a fine tune says what it read"


class _Shaded(_Rig):
    """The rig with each place as bright as *shade* says, and grain to match.

    A darker picture is a quieter one: grain's variance goes with how much
    light there is. Which is what made a calibration's regions incomparable
    with themselves, when one grain was subtracted from all of them.
    """

    def __init__(self, places, shade, **kwargs) -> None:
        super().__init__(places, **kwargs)
        self.shade = list(shade)

    def _shade(self, crop) -> float:
        return self.shade[self.places.index(self._place(crop))]

    def live_view_frame(self) -> LiveViewFrame:
        if self.ended:
            raise CameraError("live view has ended")
        self._history.append(self.optics)
        shown = self._history[-(self.lag + 1)]
        crop = self._crop()
        shade = self._shade(crop)
        pixels = self.picture(crop, shown) * shade + self._rng.normal(
            0, self.noise * np.sqrt(shade), (TALL, WIDE)
        )
        return _frame(pixels, crop, self.af)


def test_each_region_is_read_with_its_own_grain_taken_off(worker):
    """What two real calibrations showed. Panning between four regions for the
    compromise, the quietest pair of frames lately -- the grain every reading
    had taken off it -- came from the darkest region, a third as bright as the
    others. So the bright ones had too little grain taken off, and read up to
    a tenth higher than they had while fine tuned on their own: the same detail
    in the picture, 42.8 of gradient energy against 41.9, read 3.62 against
    3.26. Each region's grain is its own now, measured on its own frames, the
    same in its fine tune and in the compromise."""
    rig = _Shaded(PLACES, shade=(1.0, 0.3, 1.0), noise=7.0)
    report = _calibrated(worker, rig, _regions_round(PLACES))
    assert report.outcome == "found", report.describe()
    grain = {}
    for result in report.results:
        mine = [one for one in report.readings if one.region == result.number]
        tuned = np.median([one.grain for one in mine if one.stage == "tune"][2:])
        sought = np.median([one.grain for one in mine if one.stage != "tune"])
        assert sought == pytest.approx(tuned, rel=0.15), result.number
        grain[result.number] = sought
        # Nothing the compromise read of it is above its fine tune's best by
        # more than the grain on a reading.
        assert result.bettered is None or result.bettered < 0.02, (
            result.number,
            result.bettered,
        )
    assert grain[2] < 0.5 * grain[1] and grain[2] < 0.5 * grain[3]


@pytest.mark.parametrize("slack", [0, 40])
def test_the_rig_s_regions_come_back_at_their_depths(worker, slack):
    """Three places across the frame, sixty steps apart in focus each, read
    through a lens with play in it and a camera panned from one to the next:
    the depths are the ones the rig was built with."""
    rig = _Rig(PLACES, slack=slack)
    report = _calibrated(worker, rig, _regions_round(PLACES), depths=True)
    depths = [one.depth for one in report.results]
    assert None not in depths
    assert depths == pytest.approx([0.0, 60.0, 120.0], abs=6.0)
    tilt = report.tilt()
    assert tilt is not None
    # The rig is a plane tilted mostly left to right: (0.2, 0.25) at 0,
    # (0.8, 0.3) at 60 and (0.5, 0.75) at 120.
    assert tilt.off_the_plane < 3.0


def test_what_to_aim_for_reaches_the_search(worker):
    rig = _Rig(PLACES)
    worker._camera = rig
    for _ in range(8):
        worker._grab()
    reports = []
    worker.calibrationReady.connect(reports.append)
    worker.set_focus_regions(_regions_round(PLACES))
    worker.start_calibration(STEP, "worst")
    for _ in range(20000):
        if worker._calibration is None:
            break
        worker._grab()
    assert reports[-1].objective == "worst"
    assert reports[-1].outcome == "found"


def test_an_unknown_aim_is_refused(worker):
    rig = _Rig(PLACES)
    worker._camera = rig
    for _ in range(4):
        worker._grab()
    failures = []
    worker.failed.connect(failures.append)
    worker.set_focus_regions(_regions_round(PLACES))
    worker.start_calibration(STEP, "sharpest")
    assert worker._calibration is None
    assert failures


def test_it_puts_the_view_and_the_user_s_meter_back(worker):
    """Leaving someone at 9.4x on the last region is leaving them somewhere
    they did not ask to be; and the sharpness meter is theirs."""
    rig = _Rig(PLACES)
    worker._camera = rig
    worker.set_sharpness_area((0.4, 0.4, 0.2, 0.2))
    was_af = rig.af
    report = _calibrated(worker, rig, _regions_round(PLACES))
    assert report.outcome == "found"
    assert rig.zoom == 0
    assert rig.af == was_af
    assert not worker._sharpness.enabled
    assert worker._sharpness.area == (0.4, 0.4, 0.2, 0.2)


def test_changing_the_regions_stops_a_calibration(worker):
    rig = _Rig(PLACES)
    worker._camera = rig
    for _ in range(8):
        worker._grab()
    worker.set_focus_regions(_regions_round(PLACES))
    worker.start_calibration(STEP)
    for _ in range(40):
        worker._grab()
    assert worker._calibration is not None
    worker.set_focus_regions(_regions_round(PLACES[:2]))
    assert worker._calibration is None
    assert worker._hunt is None


def test_taking_the_focus_by_hand_stops_a_calibration(worker):
    rig = _Rig(PLACES)
    worker._camera = rig
    for _ in range(8):
        worker._grab()
    reports = []
    worker.calibrationReady.connect(reports.append)
    worker.set_focus_regions(_regions_round(PLACES))
    worker.start_calibration(STEP)
    for _ in range(40):
        worker._grab()
    worker.drive_focus(50)
    assert worker._calibration is None
    assert reports[-1] is not None and reports[-1].stopped
    assert rig.zoom == 0, "and it still put the view back"


def test_fine_tuning_by_hand_during_the_compromise_stops_the_calibration(worker):
    """Between the fine tunes no search of the button's own is running, so
    there is nothing for it to collide with -- except the calibration, which
    is walking the lens itself."""
    rig = _Rig(PLACES)
    worker._camera = rig
    for _ in range(8):
        worker._grab()
    worker.set_focus_regions(_regions_round(PLACES))
    worker.start_calibration(STEP)
    for _ in range(20000):
        if worker._calibration is None or worker._calibration.phase == "compromise":
            break
        worker._grab()
    assert worker._calibration is not None and worker._calibration.phase == "compromise"
    worker.fine_tune(STEP)
    assert worker._calibration is None
    assert worker._calibration_meter is None, "the user's meter is back"


def test_it_refuses_fewer_than_two_regions(worker):
    rig = _Rig(PLACES)
    worker._camera = rig
    for _ in range(4):
        worker._grab()
    failures = []
    worker.failed.connect(failures.append)
    worker.set_focus_regions(_regions_round(PLACES[:1]))
    worker.start_calibration(STEP)
    assert worker._calibration is None
    assert failures and "two regions" in failures[0]


# -- what it took ---------------------------------------------------------------


def test_the_report_says_how_long_it_took_and_how_much_it_drove():
    now = [100.0]
    run = Calibration(
        [Region(0.2 * n, 0.2, 0.05, 0.05) for n in range(2)],
        STEP,
        clock=lambda: now[0],
    )
    run.autofocused()
    run.drove(6)
    run.drove(-6)
    run.region_tuned(Look(100.0), "found", _view(0), probes=41)
    run.autofocused()
    run.drove(12)
    run.region_tuned(Look(100.0), "found", _view(1), probes=37)
    now[0] += 372.0
    report = run.report()
    assert report.seconds == pytest.approx(372.0)
    assert report.tune_probes == (41, 37)
    assert (report.moves, report.travel, report.autofocuses) == (3, 24, 2)
    said = report.cost()
    assert said.startswith("Took 6 min 12 s")
    assert "fine tuning 41, 37 probes" in said
    assert "3 focus moves covering 24 drive steps" in said
    assert "2 autofocuses" in said


def test_a_running_clock_reads_like_one():
    assert clock(7) == "0:07"
    assert clock(252.9) == "4:12"
    assert clock(3753) == "1:02:33"


def test_turning_back_at_four_fifths_means_four_fifths():
    """What was reported from a real calibration. The worst region started at
    81% of its best and fell a few per cent a step, while three broad regions
    around it stayed near theirs. Told to turn back at 80%, the walk out went
    on to where the worst was at 38% -- because it was waiting, for the
    depths, for the broad ones to fall as well, and they barely fell at all.

    Without depths asked for, it turns at the first reading below four fifths
    of the best of the walk. With them, it goes further, and says it will."""
    curves = [_hill(-63, width=137), _hill(0, width=150), _hill(-5, width=150),
              _hill(5, width=150)]

    def out_leg(depths: bool) -> "list[float]":
        run, _optics = _walked(
            curves, start=0.0, objective="worst", turn_back=0.8, depths=depths
        )
        return [combined for stage, _s, combined in run.report().history if stage == "out"]

    quick = out_leg(False)
    assert quick[0] == pytest.approx(0.81, abs=0.02)
    # One reading below the line, and it turns.
    below = [value for value in quick if value < 0.8 * max(quick)]
    assert len(below) == 1, quick
    assert min(quick) > 0.6, "nowhere near a third of its best"
    thorough = out_leg(True)
    assert len(thorough) > len(quick), "depths walk further, when asked for"


# -- hurrying through focus no compromise can be in ---------------------------


def _paced(moves) -> "dict[str, set[int]]":
    """How big a step each leg of a walk was made of, in increments."""
    paced: "dict[str, set[int]]" = {}
    for stage, steps, _score, _best in moves:
        paced.setdefault(stage, set()).add(abs(steps) // STEP)
    return paced


def test_hurrying_crosses_the_soft_ground_in_strides_and_the_top_in_increments():
    """Two regions twenty steps apart, so their average clears the line at the
    top, started a long way out on one side. The valley in between is crossed
    in strides of two increments and the top is walked in single ones, which
    is the whole of what the option is."""
    curves = [_hill(-10), _hill(10)]
    plain_moves, quick_moves = [], []
    plain, plain_optics = _walked(curves, start=140.0, moves=plain_moves)
    quick, quick_optics = _walked(curves, start=140.0, hurry=True, moves=quick_moves)
    assert quick.probes < plain.probes, "hurrying has to save probes"
    assert abs(quick_optics - plain_optics) <= STEP, (
        f"hurried to {quick_optics}, walked to {plain_optics}"
    )
    paced = _paced(quick_moves)
    assert HURRY_STRIDE in paced["out"] | paced["across"], "no strides at all"
    assert _paced(plain_moves) == {stage: {1} for stage in _paced(plain_moves)}, (
        "without hurrying, every step is one increment"
    )
    # On the two walks, which are paced by what they read: at or above the
    # line, every step is a single increment.
    near = [
        abs(steps) // STEP
        for stage, steps, score, _best in quick_moves
        if stage in ("out", "across") and score is not None and score >= HURRY_BELOW
    ]
    assert near and set(near) == {1}, f"strode where the answer is decided: {near}"


def test_a_scene_that_never_clears_the_line_is_strided_throughout_all_the_same():
    """The price of a fixed line, and why it is affordable at two increments.

    Regions further apart in focus than each is deep have no focus position
    where the average is four fifths -- eighty steps apart it tops out at
    two thirds -- so a hurrying walk strides the whole way across, top
    included. What saves it is that a stride is two increments and that
    nothing which lands on the answer is hurried: the way home walks the last
    of it an increment at a time, and it ends where walking ended.
    """
    curves = [_hill(-40), _hill(40)]
    moves = []
    plain, plain_optics = _walked(curves, start=40.0)
    quick, quick_optics = _walked(curves, start=40.0, hurry=True, moves=moves)
    assert max(one for _st, steps, _s, _b in moves for one in [abs(steps) // STEP]) > 1
    assert abs(quick_optics - plain_optics) <= STEP, (
        f"hurried to {quick_optics}, walked to {plain_optics}"
    )
    assert quick.probes <= plain.probes


def test_the_way_home_strides_the_dead_travel_and_lands_an_increment_at_a_time():
    """The way home is paced by distance, not by what it reads.

    It has to be paced by something: the two walks are two thirds of the
    probes and the strides cut them to a quarter, and a real calibration gave
    every bit of that back on a way home that retraced at single increments
    what the strides had crossed four at a time -- seventy probes home against
    twenty-four for both walks. What the beginning of the way home is, on a
    lens with play in it, is dead travel: the optics have not moved yet and
    every reading says what the far end of the stretch said. What the end of
    it is, is the answer, so the last of it is walked as it always was.
    """
    curves = [_hill(-40), _hill(10, height=40.0), _hill(40)]
    plain_moves, quick_moves = [], []
    plain, plain_optics = _walked(curves, start=40.0, slack=18, moves=plain_moves)
    quick, quick_optics = _walked(
        curves, start=40.0, slack=18, hurry=True, moves=quick_moves
    )
    home = [abs(steps) // STEP for stage, steps, _s, _b in quick_moves if stage == "home"]
    assert home, "it never went home"
    assert max(home) > 1, f"the way home never strided: {home}"
    assert home[-1] == 1 and home[-2:] == [1, 1], f"strode into the answer: {home}"
    walked = [abs(steps) // STEP for stage, steps, _s, _b in plain_moves if stage == "home"]
    assert len(home) < len(walked), (
        f"home took {len(home)} probes hurried and {len(walked)} walked"
    )
    assert abs(quick_optics - plain_optics) <= STEP
    # The climb, when the way home is lost, is a fine tune: never hurried.
    assert _paced(quick_moves).get("climb", {1}) == {1}


def test_one_reading_is_not_enough_to_change_the_pace():
    """Readings that will not hold still, which is what a real rig had: a
    region swinging by half between probes put the pace anywhere -- striding
    over good ground and creeping across dead ground in the same leg -- when a
    single reading decided it. Two in a row have to agree."""
    curves = [_hill(-40), _hill(40)]
    moves = []
    _run, _optics = _walked(
        curves, start=40.0, noise=0.25, seed=11, hurry=True, moves=moves
    )
    walks = [
        (steps, score)
        for stage, steps, score, _best in moves
        if stage in ("out", "across") and score is not None
    ]
    # No stride is ever taken from a reading that was not itself under the
    # line, however much the readings swing either side of it.
    for steps, score in walks:
        if abs(steps) // STEP > 1:
            assert score < HURRY_BELOW, score
    assert any(abs(steps) // STEP > 1 for steps, _score in walks), "no strides at all"


@pytest.mark.parametrize("start", [-40.0, 40.0])
def test_a_hurried_walk_meets_a_broad_top_in_the_middle_all_the_same(start):
    curves = [_hill(-40), _hill(40)]
    _run, optics = _walked(curves, start=start, hurry=True)
    assert abs(optics) <= STEP, f"ended at {optics}, not in the middle"


def test_hurrying_costs_nothing_it_can_be_held_to_across_many_made_up_scenes():
    """The same scenes the unhurried search is held to, hurried: what each
    one gives away against the true best compromise, and what it cost in
    probes. It has to save real time and lose almost nothing."""
    given_up, saved = [], []
    for seed in range(24):
        rng = np.random.default_rng(seed)
        curves = [
            _hill(float(rng.uniform(-70, 70)), width=float(rng.uniform(40, 160)),
                  height=float(rng.uniform(20, 300)))
            for _ in range(int(rng.integers(2, 5)))
        ]
        start = float(rng.choice([-1, 1]) * rng.uniform(20, 80))
        slack = int(rng.integers(0, 12))
        best, value = _best_average(curves)
        plain, _optics = _walked(curves, start=start, slack=slack, turn_back=0.3)
        quick, optics = _walked(
            curves, start=start, slack=slack, turn_back=0.3, hurry=True
        )
        shares = [curve(optics) / max(curve(p) for p in np.arange(-300.0, 300.0))
                  for curve in curves]
        given_up.append(value - float(np.mean(shares)))
        saved.append(plain.probes - quick.probes)
    assert float(np.mean(given_up)) < 0.02, f"gave away {np.mean(given_up):.3f}"
    assert max(given_up) < 0.1, f"worst gave away {max(given_up):.3f}"
    assert float(np.mean(saved)) > 0, f"saved {np.mean(saved):.1f} probes on average"


def test_a_walk_climbing_from_soft_focus_strides_even_though_every_step_is_its_best():
    """The stretch the relative test cannot see. A fine tune that left the lens
    where every region reads badly has the walk climbing ground it has never
    seen, reading its own best at every step -- so nothing is ever four fifths
    of the best, and it is the absolute half that strides there."""
    curves = [_hill(-40), _hill(40)]
    moves = []
    # Started far outside both hills, so the walk out climbs all the way in.
    _run, _optics = _walked(curves, start=-200.0, hurry=True, moves=moves)
    climbing = [
        abs(steps) // STEP
        for stage, steps, score, best in moves
        if stage == "out" and score is not None and score >= best and score < HURRY_BELOW
    ]
    # All of them strides but the first, which is the probe the two-readings
    # rule costs at the top of every soft stretch.
    assert climbing and set(climbing) <= {1, HURRY_STRIDE}, climbing
    assert climbing.count(HURRY_STRIDE) >= len(climbing) - 1, climbing


def test_a_hurrying_search_says_what_pace_it_is_walking_at():
    """Someone watching a search that has slowed down wants to know whether it
    is on to something or merely hurrying, so the line says which."""
    said = []
    _run, _optics = _walked(
        [_hill(-40), _hill(40)],
        start=40.0,
        hurry=True,
        watch=lambda run: said.append(run.progress()),
    )
    assert any(f"strides of {HURRY_STRIDE * STEP}" in line for line in said), said
    assert any(f"steps of {STEP}," in line for line in said), said


def _travel(moves) -> "dict[str, int]":
    """How far each leg of a walk went, in drive steps."""
    per: "dict[str, int]" = {}
    for stage, steps, _score, _best in moves:
        per[stage] = per.get(stage, 0) + abs(steps)
    return per


def _dead(at: float = -250.0, height: float = 0.2):
    """A region whose fine tune found a little and which reads nothing after.

    Region 2 of the calibration that found the bug below: a peak of 0.2 where
    the others read 2.4 to 4.6, and nothing above its own grain anywhere the
    compromise walked -- so its share was zero at every probe.
    """
    return lambda p: height if abs(p - at) < 3 else 0.0


def _five_with_a_dead_one():
    return [
        _hill(-30, width=45, height=4.6),
        _dead(),
        _hill(0, width=45, height=3.5),
        _hill(20, width=50, height=3.2),
        _hill(-10, width=45, height=2.4),
    ]


def test_hurrying_covers_the_same_ground_and_not_four_times_as_much():
    """What a real run found, and why every limit here is a distance.

    Five regions with one that reads nothing where the walks go, and depths
    asked for: "fallen a quarter below its best" can never be true of a region
    reading nothing, so both walks run to the end of their patience every
    time. With that patience counted by the probe, a stride of four made it
    four times as long -- the rig walked 672 steps out where it had walked 156
    and 1230 across where it had walked 228, out into focus where nothing
    reads at all, and then crawled the whole of it back an increment at a
    time, 206 probes against 86. Hurrying may cross the same ground in fewer
    probes. It may not cross more ground.
    """
    curves = _five_with_a_dead_one()
    walked, hurried = [], []
    plain, plain_optics = _walked(
        curves, start=-30.0, slack=8, depths=True, moves=walked
    )
    quick, quick_optics = _walked(
        curves, start=-30.0, slack=8, depths=True, hurry=True, moves=hurried
    )
    there, and_back = _travel(walked), _travel(hurried)
    for leg in ("out", "across"):
        # A stride at either end of a leg, since the turn-back line is only
        # looked at where a reading is taken -- and nothing like four times.
        assert and_back[leg] <= there[leg] + 2 * HURRY_STRIDE * STEP, (
            f"hurrying went {and_back[leg]} steps {leg} where walking went {there[leg]}"
        )
    assert quick.probes < plain.probes, (quick.probes, plain.probes)
    assert abs(quick_optics - plain_optics) <= STEP


def test_a_walk_stops_waiting_for_a_picture_with_nothing_in_it():
    """The other half of that run: what the walks were waiting for.

    A region that reads nothing can never be seen to climb and has no peak to
    place, so both of the things a walk goes on past the turn-back line for
    are waits for something that cannot happen -- and it walked on to where
    the number it was climbing was at a fiftieth of its best, a pan to every
    region and a wait for a picture at each step. Now a collapsed reading ends
    the wait wherever it happens, hurried or not.
    """
    curves = _five_with_a_dead_one()
    for hurry in (False, True):
        moves = []
        run, _optics = _walked(
            curves, start=-30.0, slack=8, depths=True, hurry=hurry, moves=moves
        )
        spent = [
            score
            for _stage, _steps, score, best in moves
            if score is not None and best > 0.0 and score < 0.1 * best
        ]
        assert len(spent) <= 10, f"{len(spent)} probes on nothing (hurry={hurry})"
        assert run.report().outcome == "found"


def test_the_leg_across_still_crosses_the_soft_ground_it_sets_off_from():
    """What the give-up line must not do. The leg across starts at the far end
    of the leg out, which is the softest reading there is -- and if a collapsed
    reading were judged against the best of the whole search rather than of the
    leg it is on, the walk would turn round on the spot and never cross back
    over the regions at all."""
    curves = _five_with_a_dead_one()
    moves = []
    run, optics = _walked(
        curves, start=-30.0, slack=8, depths=True, hurry=True, moves=moves
    )
    across = [steps for stage, steps, _s, _b in moves if stage == "across"]
    assert len(across) >= 8, f"the leg across gave up after {len(across)} probes"
    best, _value = _best_average(curves)
    assert abs(optics - best) <= 2 * STEP, f"ended at {optics}, best is {best}"


def test_without_depths_asked_for_the_report_has_none_and_says_how_to_get_them(window):
    from PySide6.QtWidgets import QLabel

    from scanny.ui.report import CalibrationReportDialog

    run, _optics = _walked([_hill(-40), _hill(10), _hill(40)], start=40.0)
    report = run.report()
    assert not report.depths_measured
    assert all(one.depth is None for one in report.results)
    assert report.ordering() == ""
    dialog = CalibrationReportDialog(report, Orientation(), window._save_directory(), window)
    said = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "Measure depths for levelling" in said


def test_measuring_depths_is_asked_for_before_calibrating_and_remembered(window):
    asked = []
    window.requestCalibration.connect(
        lambda step, aim, turn, depths: asked.append(depths)
    )
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    assert not window.regions_depths.isChecked(), "off, out of the box"
    window.calibrate_button.click()
    window.regions_depths.setChecked(True)
    window.calibrate_button.click()
    assert asked == [False, True]
    assert QSettings().value("regions/depths", False, bool) is True
    window._on_calibration_changed(True)
    assert not window.regions_depths.isEnabled()
    window._on_calibration_changed(False)


def test_how_long_a_stride_is_set_before_calibrating_and_remembered(window):
    asked = []
    window.requestCalibration.connect(
        lambda step, aim, turn, depths, hurry: asked.append(hurry)
    )
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    assert window.regions_hurry.value() == 1, "off, out of the box"
    assert window.regions_hurry.specialValueText() == "off", "and it says so"
    window.calibrate_button.click()
    window.regions_hurry.setValue(3)
    window.calibrate_button.click()
    assert asked == [1, 3]
    assert int(QSettings().value("regions/hurry")) == 3
    # Settled before it starts, like everything else the search is shaped by.
    window._on_calibration_changed(True)
    assert not window.regions_hurry.isEnabled()
    window._on_calibration_changed(False)
    assert window.regions_hurry.isEnabled()


def test_the_tick_hurrying_used_to_be_becomes_the_stride_it_stood_for(app):
    from scanny.ui.regions import HURRY_STRIDE as stride

    settings = QSettings()
    settings.setValue("regions/hurry", True)
    assert mw.MainWindow._stored_hurry() == stride
    settings.setValue("regions/hurry", False)
    assert mw.MainWindow._stored_hurry() == 1
    settings.setValue("regions/hurry", 4)
    assert mw.MainWindow._stored_hurry() == 4
    settings.remove("regions/hurry")
    assert mw.MainWindow._stored_hurry() == 1


def test_hurrying_reaches_the_search_and_is_said_while_it_runs(worker):
    rig = _Rig(PLACES)
    worker._camera = rig
    for _ in range(4):
        worker._grab()
    worker.set_focus_regions(_regions_round(PLACES))
    worker.start_calibration(STEP, "average", 80, False, 3)
    assert worker._calibration.hurries
    assert worker._calibration.hurry == 3
    worker.cancel_calibration()


def test_the_rig_s_calibration_is_timed_and_counted(worker):
    rig = _Rig(PLACES)
    report = _calibrated(worker, rig, _regions_round(PLACES))
    assert report.seconds > 0.0
    assert len(report.tune_probes) == 3 and all(report.tune_probes)
    assert report.autofocuses == 3
    assert report.moves > report.probes
    assert report.travel >= STEP * report.moves
    assert report.cost().startswith("Took ")


def test_the_panel_shows_how_long_the_calibration_has_run(window, monkeypatch):
    started = [1000.0]
    monkeypatch.setattr(mw.time, "monotonic", lambda: started[0])
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    window._on_calibration_changed(True)
    window._on_calibration_progress(0, "Region 1 of 2: fine tuning it on its own")
    started[0] += 125.0
    window._show_progress()
    assert window.regions_progress.text().startswith("2:05")
    assert "Region 1 of 2" in window.regions_progress.text()
    assert window._clock.isActive(), "it keeps ticking between messages"
    window._on_calibration_changed(False)
    assert not window._clock.isActive()


# -- live view the body turns off ----------------------------------------------


@pytest.fixture
def quick_restarts(monkeypatch):
    """No waiting between attempts at starting live view again."""
    from scanny.ui import worker as module

    monkeypatch.setattr(module, "_RESTART_WAITS", (0.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(module, "_VIEW_BACK_PATIENCE", 0.5)


def test_live_view_the_body_turns_off_is_started_again_and_the_view_put_back(
    worker, quick_restarts
):
    """A D750 ends live view after its own monitor-off delay whoever is
    driving it, and a calibration can take longer than the ten minutes it
    comes set to. Focus does not move when it does, so the calibration goes
    on once the view is back the way it was.

    The body in this is the real one's kind: out of live view it refuses to
    move the focus point, zoom or drive focus as well as to send frames, its
    first attempt at starting again is refused, and the first command after
    it is answered busy. A calibration between two probes meets the refusals
    first, and that is the path that used to end it."""
    rig = _Rig(PLACES)
    rig.refuses_restart = 1
    worker._camera = rig
    for _ in range(8):
        worker._grab()
    said = []
    worker.status.connect(said.append)
    reports = []
    worker.calibrationReady.connect(reports.append)
    worker.set_focus_regions(_regions_round(PLACES))
    worker.start_calibration(STEP, "average", 80, True)
    ended_in = []
    for count in range(20000):
        run = worker._calibration
        if run is None:
            break
        # Once while a region is being fine tuned, once in the compromise.
        early = count == 150 and not ended_in
        later = run.phase == "compromise" and run.probes == 5 and len(ended_in) == 1
        if early or later:
            ended_in.append(run.phase)
            rig.ends_after = 1
            rig.refuses_restart = 1
            rig.busy_after_restart = 1
        worker._grab()
    assert rig.restarts == 2
    assert ended_in == ["peaks", "compromise"]
    assert reports[-1].outcome == "found", reports[-1].describe()
    assert any("turned live view off by itself" in line for line in said)
    depths = [one.depth for one in reports[-1].results]
    assert depths == pytest.approx([0.0, 60.0, 120.0], abs=6.0)


def test_a_compromise_move_the_body_refused_is_made_once_live_view_is_back(
    worker, quick_restarts
):
    """Live view going off as a move is sent loses the move, and the search
    would then be one increment out about where it is for the rest of the
    walk. The move is kept, and made first thing once live view is back."""
    rig = _Rig(PLACES)
    worker._camera = rig
    for _ in range(8):
        worker._grab()
    worker.set_focus_regions(_regions_round(PLACES))
    worker.start_calibration(STEP)
    for _ in range(20000):
        run = worker._calibration
        if run is None or (run.phase == "compromise" and run.probes == 3):
            break
        worker._grab()
    run = worker._calibration
    assert run is not None
    driven = run._moves
    original = rig.drive_focus

    def goes_off_as_it_drives(steps: int) -> bool:
        rig.drive_focus = original
        rig.ended = True
        return original(steps)

    rig.drive_focus = goes_off_as_it_drives
    worker._grab()  # the probe, then the move -- refused
    assert rig.restarts == 1
    refused = run.pending
    assert refused != 0, "the refused move is kept"
    assert run._moves == driven, "and not counted as made"
    drives = []
    made = rig.drive_focus
    rig.drive_focus = lambda steps: drives.append(steps) or made(steps)
    worker._grab()
    assert drives[0] == refused, "it is the first thing made once live view is back"
    assert run.pending == 0


def test_a_body_that_keeps_ending_live_view_is_not_fought(worker, quick_restarts):
    rig = _Rig(PLACES)
    worker._camera = rig
    for _ in range(4):
        worker._grab()
    failures = []
    worker.failed.connect(failures.append)
    changed = []
    worker.liveViewChanged.connect(changed.append)
    for _ in range(5):
        rig.ends_after = 1
        for _ in range(20):
            worker._grab()
    assert rig.restarts == 3, "three in a couple of minutes, and no more"
    assert changed == [False]
    assert failures and "shut down" in failures[-1]


def test_every_restart_is_in_the_activity_log_with_the_camera_s_own_words(
    worker, quick_restarts
):
    rig = _Rig(PLACES)
    worker._camera = rig
    for _ in range(4):
        worker._grab()
    logged = []
    worker.log.connect(logged.append)
    rig.ends_after = 1
    rig.refuses_restart = 1
    for _ in range(20):
        worker._grab()
    said = "\n".join(logged)
    assert "gone off" in said
    assert "attempt 1: the camera refused: camera will not enter live view: busy" in said
    assert "attempt 2: live view is back" in said


# -- every region through the search ------------------------------------------


def test_the_report_keeps_every_region_through_the_search():
    curves = [_hill(-40), _hill(10), _hill(40)]
    run, _optics = _walked(curves, start=40.0)
    report = run.report()
    assert report.history_regions == (1, 2, 3)
    assert len(report.history) == report.probes
    stages = [stage for stage, _shares, _combined in report.history]
    assert stages[0] == "out" and "across" in stages and stages[-1] == "home"
    # Region 3 is where it starts: at its own best on the first probe.
    assert report.history[0][1][2] == pytest.approx(1.0)
    for _stage, shares, combined in report.history:
        assert combined == pytest.approx(np.mean(shares))


def test_the_panel_charts_every_region_as_the_search_goes(window):
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    window._on_calibration_changed(True)
    assert not window.regions_chart.isVisible()
    window._on_calibration_probe((1, 2), "out", (1.0, 0.3), 0.65)
    window._on_calibration_probe((1, 2), "out", (0.8, 0.5), 0.65)
    assert window.regions_chart.isVisible()
    assert window.regions_chart.probes == 2
    window.regions_chart.repaint()


def test_the_worker_hands_every_probe_to_the_chart(worker):
    rig = _Rig(PLACES)
    probes = []
    worker.calibrationProbe.connect(lambda *probe: probes.append(probe))
    report = _calibrated(worker, rig, _regions_round(PLACES))
    assert len(probes) == report.probes
    regions, stage, shares, combined = probes[0]
    assert tuple(regions) == (1, 2, 3) and stage == "out" and len(shares) == 3


# -- the film's shape ----------------------------------------------------------


def test_three_regions_make_a_plane_and_no_bulge():
    from scanny.ui.film import FilmSurface

    points = [(0.2, 0.2), (0.8, 0.3), (0.4, 0.8)]
    surface = FilmSurface.fit(points, [100 * x + 30 * y for x, y in points])
    assert surface.plane[0] == pytest.approx(100.0) and surface.plane[1] == pytest.approx(30.0)
    assert abs(surface.bulge()[0]) < 1e-6


def test_a_bowed_frame_bulges_between_edges_that_stay_on_their_plane():
    """Four corners on a leaning plane and the middle standing proud of it:
    the film passes through every measured depth, its edges keep the lean,
    and all of the rest is bulge -- most in the middle, none at the edges."""
    from scanny.ui.film import FilmSurface

    points = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.5, 0.5)]
    depths = [60 * x + (20 if (x, y) == (0.5, 0.5) else 0) for x, y in points]
    surface = FilmSurface.fit(points, depths, [1.0] * 5)
    for (x, y), depth in zip(points, depths):
        assert float(surface.at(x, y)) == pytest.approx(depth, abs=1e-6)
    assert surface.plane[0] == pytest.approx(60.0, abs=0.5)
    bulge, x, y = surface.bulge()
    assert bulge > 20.0 and (x, y) == pytest.approx((0.5, 0.5), abs=0.05)
    for edge in ((0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0)):
        assert float(surface.bulge_at(*edge)) == pytest.approx(0.0, abs=1e-9)


def test_the_bulge_says_which_way_the_film_bows():
    report = _levelled_report(
        lambda x, y: 50 * x - (15 if (x, y) == (0.5, 0.5) else 0),
        corners=[(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.5, 0.5)],
    )
    said = report.tilt().describe()
    assert "nearer the camera than they are" in said
    assert "right edge focuses 50 steps further than the left edge" in said


def _levelled_report(depth_at, corners):
    raw = [depth_at(x, y) for x, y in corners]
    results = tuple(
        RegionResult(
            number + 1,
            Region(x - 0.05, y - 0.05, 0.1, 0.1),
            Look(100.0),
            "found",
            Look(90.0),
            depth=value - min(raw),
            doubt=1.0,
        )
        for number, ((x, y), value) in enumerate(zip(corners, raw))
    )
    return CalibrationReport(results, 0.9, "found")


def test_the_film_is_drawn_over_the_sensor_and_can_be_turned_round(app):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    from scanny.ui.film import FilmView, Mark

    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.5, 0.5)]
    report = _levelled_report(lambda x, y: 40 * x + (25 if x == 0.5 else 0), corners)
    view = FilmView()
    view.resize(500, 380)
    view.set_scene(
        report.surface(),
        [Mark(one.number, one.region.rect, one.depth) for one in report.placed],
        Orientation(turns=1, mirrored=True),
    )
    view.show()
    QApplication.processEvents()
    before = view.grab().toImage()
    turned_from = view.angles
    left = Qt.MouseButton.LeftButton
    for kind, x in (
        (QMouseEvent.Type.MouseButtonPress, 100),
        (QMouseEvent.Type.MouseMove, 180),
        (QMouseEvent.Type.MouseButtonRelease, 180),
    ):
        event = QMouseEvent(kind, QPointF(x, 150), QPointF(x, 150), left, left,
                            Qt.KeyboardModifier.NoModifier)
        {
            QMouseEvent.Type.MouseButtonPress: view.mousePressEvent,
            QMouseEvent.Type.MouseMove: view.mouseMoveEvent,
            QMouseEvent.Type.MouseButtonRelease: view.mouseReleaseEvent,
        }[kind](event)
    assert view.angles != turned_from
    after = view.grab().toImage()
    assert after != before, "turning it round redraws it from the new side"


def test_what_the_film_covers_of_a_region_is_drawn_faintly_and_not_in_full(app):
    """A post runs from the sensor up to the film, so wherever the film is
    between it and the eye it has to give way to it. Drawn at full strength
    all the way up it reads as standing in front of the film it is really
    behind, and turning the view round does not say otherwise. Faintly is
    still enough to follow it down to the region it belongs to."""
    from scanny.ui.film import _MARK, FilmView, Mark

    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.5, 0.5)]
    report = _levelled_report(lambda x, y: 40 * x + (25 if x == 0.5 else 0), corners)
    view = FilmView()
    view.resize(500, 380)
    view.set_scene(
        report.surface(),
        [Mark(one.number, one.region.rect, one.depth) for one in report.placed],
    )
    view.show()

    def in_full() -> int:
        """How much of the marks is drawn in their colour and nothing else."""
        view.update()
        QApplication.processEvents()
        shown = view.grab().toImage()
        return sum(
            shown.pixelColor(x, y) == _MARK
            for x in range(shown.width())
            for y in range(shown.height())
        )

    low, high, scale, gap = view._heights()
    heights = (low, scale, gap, gap + (high - low) * scale)
    one = report.placed[0]
    x, y, w, h = one.region.rect
    up_to = gap + (one.depth - low) * scale
    halfway = view._world(x + w / 2, y + h / 2, up_to / 2)
    on_the_film = view._world(x + w / 2, y + h / 2, up_to)

    view._turn, view._tilt = 0.0, 80.0
    from_above = in_full()
    assert view._covered([halfway], *heights)[0], "the film is over the post"
    assert not view._covered([on_the_film], *heights)[0], (
        "but not over the mark at the top of it, which lies on the film: a "
        "way out that rises off the film has not gone through it"
    )
    view._tilt = 5.0
    assert in_full() > from_above, (
        "and from the side, what is under the film's edge is in the open"
    )


def test_the_film_is_vivid_from_above_and_dull_from_underneath(app):
    """Both sides of the sheet are the same shape at the same heights, so with
    nothing to tell them apart a view from below passes for a view from above
    -- and every lean read off it is backwards. The underside is dulled and
    darkened instead, so which side is being looked at needs no working out."""
    from scanny.ui.film import FilmView, Mark

    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.5, 0.5)]
    report = _levelled_report(lambda x, y: 40 * x + (25 if x == 0.5 else 0), corners)
    view = FilmView()
    view.resize(500, 380)
    view.set_scene(
        report.surface(),
        [Mark(one.number, one.region.rect, one.depth) for one in report.placed],
    )
    view.show()

    def vivid() -> int:
        """How much of the drawing is in the strong colours of the depth ramp."""
        view.update()
        QApplication.processEvents()
        shown = view.grab().toImage()
        looked = [
            shown.pixelColor(x, y)
            for x in range(0, shown.width(), 3)
            for y in range(0, shown.height(), 3)
        ]
        return sum(one.saturation() > 120 and one.value() > 140 for one in looked)

    view._tilt = 28.0
    from_above = vivid()
    view._tilt = -28.0
    from_below = vivid()
    assert from_above > 10 * from_below, "the far side of the film is not its face"


def test_the_film_page_keeps_its_notes_under_the_drawing(window):
    """Beside it they took a third of the page's width off the drawing, which
    is the thing on the page meant to be looked at. Under it they are as tall
    as they need and no taller, and the drawing has the whole width."""
    from scanny.ui.report import CalibrationReportDialog, _NOTES_TALL

    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    report = _levelled_report(lambda x, y: 30 * x + 10 * y, corners)
    dialog = CalibrationReportDialog(report, Orientation(), window._save_directory(), window)
    dialog.show()
    # A page that is not the one on show is not laid out, and has nothing to
    # say about where anything on it sits.
    dialog._tabs.setCurrentWidget(dialog._film_page)
    QApplication.processEvents()
    view, notes = dialog._film_view, dialog._film_notes
    assert notes.height() <= _NOTES_TALL
    assert notes.y() >= view.y() + view.height(), "under the drawing, not beside it"
    assert view.width() >= notes.width(), "and the drawing has the width of the page"
    dialog.close()


def test_the_drawing_fills_a_page_wider_than_it_is_tall(app):
    """The two sheets seen from the side are far wider than they are tall.
    Fitted by their width against the shorter side of the widget, they came out
    small in the middle of a wide page with the room going to waste."""
    from scanny.ui.film import FilmView, Mark

    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    report = _levelled_report(lambda x, y: 30 * x + 10 * y, corners)
    view = FilmView()
    view.set_scene(
        report.surface(),
        [Mark(one.number, one.region.rect, one.depth) for one in report.placed],
    )
    view.show()

    def drawn_across() -> int:
        """How much of the width the drawing covers, in pixels."""
        QApplication.processEvents()
        shown = view.grab().toImage()
        ground = shown.pixelColor(0, 0)
        columns = [
            x
            for x in range(shown.width())
            for y in range(0, shown.height(), 4)
            if shown.pixelColor(x, y) != ground
        ]
        return max(columns) - min(columns)

    view.resize(400, 400)
    square = drawn_across()
    view.resize(900, 400)
    assert drawn_across() > 1.8 * square, "a wider page draws a wider picture"


def test_the_focus_plane_is_drawn_among_the_depths_it_was_chosen_for(app):
    """One focus position for all of the regions is one plane, level with the
    sensor, and the whole of the compromise is what the film does either side
    of it. Drawn see-through, so that the film shows through it where it is
    behind it and over it where it is in front."""
    from scanny.ui.film import FilmView, Mark

    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.5, 0.5)]
    report = _levelled_report(lambda x, y: 40 * x + (25 if x == 0.5 else 0), corners)
    view = FilmView()
    view.resize(500, 380)
    marks = [Mark(one.number, one.region.rect, one.depth) for one in report.placed]

    def drawn():
        view.update()
        QApplication.processEvents()
        return view.grab().toImage()

    view.set_scene(report.surface(), marks)
    view.show()
    without = drawn()
    deepest = max(one.depth for one in report.placed)
    view.set_scene(report.surface(), marks, focus=deepest / 2)
    assert drawn() != without, "the plane is drawn"

    # Put it above everything and the drawing has to make room for it: the
    # heights are stretched to take it in, as they are for any depth.
    view.set_scene(report.surface(), marks, focus=deepest * 3)
    assert view._heights()[1] == pytest.approx(deepest * 3)


def test_the_sensor_says_which_of_its_sides_is_being_looked_at(app):
    """It lies flat, so there is no shading on it to tell its face from its
    back, and two sides alike make turning right under the rig look like
    turning back over it. Its back is drawn as the back of a plate instead:
    darker, and without the grid its face is ruled into."""
    from scanny.ui.film import _SENSOR, _SENSOR_BACK, FilmView, Mark

    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    report = _levelled_report(lambda x, y: 30 * x + 10 * y, corners)
    view = FilmView()
    view.resize(500, 380)
    view.set_scene(
        report.surface(),
        [Mark(one.number, one.region.rect, one.depth) for one in report.placed],
    )
    view.show()

    def sides() -> "tuple[int, int]":
        """How much of the drawing is the sensor's face and how much its back."""
        view.update()
        QApplication.processEvents()
        shown = view.grab().toImage()
        looked = [
            shown.pixelColor(x, y)
            for x in range(0, shown.width(), 3)
            for y in range(0, shown.height(), 3)
        ]
        return (
            sum(one == _SENSOR for one in looked),
            sum(one == _SENSOR_BACK for one in looked),
        )

    view._tilt = 30.0
    face, back = sides()
    assert face > 0 and back == 0, "from over the rig, the sensor's face"
    view._tilt = -30.0
    face, back = sides()
    assert back > 0 and face == 0, "from under it, the back of the same plate"


def test_the_report_has_the_film_and_the_search_on_pages_of_their_own(window):
    from scanny.ui.report import CalibrationReportDialog

    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    report = _levelled_report(lambda x, y: 30 * x + 10 * y, corners)
    dialog = CalibrationReportDialog(report, Orientation(), window._save_directory(), window)
    assert [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())] == [
        "Regions", "Film shape", "The search", "Activity log",
    ]
    assert dialog._film_view._surface is not None
    assert len(dialog._film_view._marks) == 4
    dialog._tabs.setCurrentIndex(1)
    widget, word = dialog._page_to_save()
    assert word == "film"


# -- the settings and the log --------------------------------------------------


def test_how_far_a_walk_goes_is_set_before_calibrating_and_remembered(window):
    asked = []
    window.requestCalibration.connect(lambda step, aim, turn: asked.append(turn))
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    assert window.regions_turn_back.value() == 80, "four fifths, out of the box"
    window.regions_turn_back.setValue(60)
    window.calibrate_button.click()
    assert asked == [60]
    assert int(QSettings().value("regions/turn_back")) == 60
    window._on_calibration_changed(True)
    assert not window.regions_turn_back.isEnabled()
    window._on_calibration_changed(False)
    assert window.regions_turn_back.isEnabled()


def test_the_turn_back_share_reaches_the_search(worker):
    rig = _Rig(PLACES)
    worker._camera = rig
    for _ in range(4):
        worker._grab()
    worker.set_focus_regions(_regions_round(PLACES))
    worker.start_calibration(STEP, "worst", 65)
    assert worker._calibration.turn_back == pytest.approx(0.65)
    worker.cancel_calibration()


def test_the_activity_log_keeps_what_was_said_with_the_time(tmp_path):
    from scanny.ui.activity import ActivityLog

    log = ActivityLog()
    log.write_to(tmp_path / "activity.log")
    log.add("Calibration started", "status")
    log.add("attempt 1: the camera refused: busy")
    log.add("Live view could not be started again", "error")
    lines = log.lines
    assert len(lines) == 3
    assert lines[2].split("  ", 1)[1].startswith("!! Live view")
    written = (tmp_path / "activity.log").read_text(encoding="utf-8").splitlines()
    assert len(written) == 3 and "attempt 1" in written[1]


def test_the_activity_window_follows_the_log_as_it_grows(window):
    window._show_activity()
    shown = window._activity_window
    window.activity.add("Region 1 of 3: fine tuning it on its own")
    assert "Region 1 of 3" in shown.text
    # And what the worker says reaches it, when there is a worker.
    from scanny.ui.worker import CameraWorker

    worker = CameraWorker()
    worker.log.connect(window.activity.add)
    worker.log.emit("Compromise probe 4 (across): 1: 90%, 2: 71% -> 71%")
    assert "Compromise probe 4" in shown.text


# -- the report as a document, and the bell ------------------------------------


def _documented_report() -> CalibrationReport:
    """A report with everything in it a file has to carry: pictures, depths,
    an unknown doubt, a region never reached, the search and its log."""
    from dataclasses import replace

    pixels = np.zeros((10, 20))
    pixels[:, 10:] = 255.0
    picture = _image(pixels)
    places = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.5, 0.5)]
    levelled = _levelled_report(lambda x, y: 30 * x + 10 * y, places)
    results = list(levelled.results)
    results[0] = replace(
        results[0],
        best=Look(152.345, picture),
        compromise=Look(140.5, picture),
        tuned=148.0,
    )
    results[1] = replace(results[1], doubt=float("inf"), edge=True)
    results[3] = replace(results[3], best=None, compromise=None, depth=None)
    return replace(
        levelled,
        results=tuple(results),
        probes=2,
        objective="worst",
        seconds=372.5,
        tune_probes=(41, 37, 12),
        autofocuses=3,
        moves=90,
        travel=540,
        history=(("out", (0.9, 0.8, 0.7), 0.7), ("across", (0.95, 0.85, 0.75), 0.75)),
        history_regions=(1, 2, 3),
        depths_measured=True,
        focus_depth=17.5,
        began=1_789_000_000.0,
        log=("14:03:01  Calibration started", "14:03:02  !! the camera refused: busy"),
        readings=(
            Reading(1, "tune", 0, 120.5, 1.25, 0.5, 1.25, 110.5),
            Reading(1, "tune", 12, 148.0, 2.5, 0.75, 1.25, 111.0),
            Reading(2, "out", -6, 90.0, 5.0, 1.5, 0.375, 48.25),
        ),
    )


def test_the_whole_report_can_be_saved_and_opened_again(app, tmp_path):
    from scanny.ui.reportfile import load_report, save_report

    report = _documented_report()
    path = tmp_path / "levelling.focusreport"
    save_report(path, report, aspect=1.5)
    saved = load_report(path)
    assert saved.report == report, "every number comes back as it went"
    assert saved.report.results[0].bettered == pytest.approx(152.345 / 148.0 - 1)
    assert saved.aspect == pytest.approx(1.5)
    best = saved.report.results[0].best.picture
    assert (best.width(), best.height()) == (20, 10)
    assert QImage(best).pixelColor(15, 5).red() == 255
    assert saved.report.results[1].best.picture is None
    assert saved.report.tilt() == report.tilt(), "and so does the film's shape"
    assert not list(tmp_path.glob("*.part")), "nothing left over from writing it"


def test_every_reading_goes_into_the_file_as_a_table_too(app, tmp_path):
    """For a spreadsheet, which is where a question about the readings --
    why the compromise read a region higher than its fine tune did -- gets
    looked into. A time not known is an empty cell, and comes back as one."""
    import zipfile
    from dataclasses import replace

    from scanny.ui.reportfile import load_report, save_report

    report = _documented_report()
    report = replace(
        report, readings=report.readings + (Reading(3, "home", 6, 80.25, 9.5),)
    )
    path = tmp_path / "levelling.focusreport"
    save_report(path, report, aspect=1.5)
    with zipfile.ZipFile(path) as archive:
        table = archive.read("readings.csv").decode("utf-8").splitlines()
    assert table[0] == "region,stage,position,value,at,after,grain,level"
    assert table[1] == "1,tune,0,120.5,1.250,0.500,1.25,110.5"
    assert table[-1] == "3,home,6,80.25,9.500,,,"
    back = load_report(path).report.readings
    assert back[:3] == report.readings[:3]
    assert np.isnan(back[3].after) and back[3].value == 80.25
    assert np.isnan(back[3].grain) and np.isnan(back[3].level)


def test_a_report_saved_before_readings_were_kept_still_opens(app, tmp_path):
    import json
    import zipfile

    from scanny.ui.reportfile import load_report, save_report

    path = tmp_path / "older.focusreport"
    save_report(path, _documented_report(), aspect=1.5)
    older = tmp_path / "oldest.focusreport"
    with zipfile.ZipFile(path) as archive, zipfile.ZipFile(older, "w") as kept:
        numbers = json.loads(archive.read("report.json"))
        del numbers["report"]["readings"]
        for one in numbers["report"]["results"]:
            del one["tuned"]
        kept.writestr("report.json", json.dumps(numbers))
        for name in archive.namelist():
            if name not in ("report.json", "readings.csv"):
                kept.writestr(name, archive.read(name))
    report = load_report(older).report
    assert report.readings == ()
    assert all(one.tuned is None and one.bettered is None for one in report.results)


def test_the_report_window_says_when_a_best_was_read_on_the_walk(window):
    from dataclasses import replace

    from PySide6.QtWidgets import QLabel

    from scanny.ui.report import CalibrationReportDialog

    report = _report()
    first, second = report.results
    report = replace(report, results=(replace(first, tuned=140.0), second))
    dialog = CalibrationReportDialog(report, Orientation(), window._save_directory(), window)
    said = [label.text() for label in dialog._content.findChildren(QLabel)]
    assert any("read on the walk for the compromise" in text for text in said)
    assert any("fine tuning on its own found 140" in text for text in said)
    assert any("7% higher than fine tuning did" in text for text in said)


def test_a_file_that_is_not_a_report_is_refused(app, tmp_path):
    import zipfile

    from scanny.ui.reportfile import ReportFileError, load_report

    text = tmp_path / "notes.focusreport"
    text.write_text("not a report", encoding="utf-8")
    with pytest.raises(ReportFileError):
        load_report(text)
    other = tmp_path / "other.focusreport"
    with zipfile.ZipFile(other, "w") as archive:
        archive.writestr("readme.txt", "hello")
    with pytest.raises(ReportFileError, match="not a focus report"):
        load_report(other)
    with pytest.raises(ReportFileError):
        load_report(tmp_path / "missing.focusreport")


def test_the_report_window_saves_the_whole_document(window, tmp_path, monkeypatch):
    from scanny.ui import report as module
    from scanny.ui.report import CalibrationReportDialog
    from scanny.ui.reportfile import load_report

    report = _documented_report()
    dialog = CalibrationReportDialog(report, Orientation(), tmp_path, window)
    offered = []

    def choose(parent, title, suggested, kinds):
        offered.append(suggested)
        return str(tmp_path / "kept"), kinds

    monkeypatch.setattr(module.QFileDialog, "getSaveFileName", choose)
    dialog._save_report()
    assert offered[0].endswith(".focusreport")
    assert "focus-report-" in offered[0]
    # The suffix is put on when it was left off.
    assert load_report(tmp_path / "kept.focusreport").report == report


def test_a_report_opened_from_a_file_has_a_window_of_its_own(window, tmp_path, monkeypatch):
    from scanny.ui.reportfile import save_report

    path = tmp_path / "levelling.focusreport"
    save_report(path, _documented_report(), aspect=1.5)
    monkeypatch.setattr(
        mw.QFileDialog, "getOpenFileName", lambda *args: (str(path), "")
    )
    window._open_report_file()
    assert len(window._opened_reports) == 1
    dialog = window._opened_reports[0]
    assert "levelling.focusreport" in dialog.windowTitle()
    assert "Calibration started" in dialog._log_text.toPlainText()
    assert "Calibrated " in dialog._cost.text()
    assert dialog._film_view._surface is not None
    # It follows the view like the last calibration's report does.
    window._turn_view(1)
    assert dialog._orientation.turns == 1
    dialog.close()
    QApplication.processEvents()
    assert window._opened_reports == []


def test_a_file_that_will_not_open_is_said_so(window, tmp_path, monkeypatch):
    broken = tmp_path / "broken.focusreport"
    broken.write_text("nothing", encoding="utf-8")
    monkeypatch.setattr(
        mw.QFileDialog, "getOpenFileName", lambda *args: (str(broken), "")
    )
    window._open_report_file()
    assert window._opened_reports == []
    assert "broken.focusreport" in window.statusBar().currentMessage()


def test_what_was_said_while_calibrating_goes_into_its_report(worker):
    rig = _Rig(PLACES)
    report = _calibrated(worker, rig, _regions_round(PLACES))
    said = "\n".join(report.log)
    assert report.log[0].split("  ", 1)[1].startswith("   Calibration started")
    assert "Region 1 of 3" in said
    assert "Compromise probe 1" in said
    assert "Calibration finished" in report.log[-1]
    assert report.began > 0.0
    assert worker._calibration_said is None, "and nothing after it is kept"


def test_the_bell_sounds_when_a_calibration_ends_by_itself(window, monkeypatch):
    rung = []
    monkeypatch.setattr(QApplication, "beep", lambda: rung.append(True))
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    window._on_calibration_changed(True)
    window._on_calibration_changed(False)
    assert rung == [True]
    # Stopped from here, whoever stopped it is looking.
    window._on_calibration_changed(True)
    window.calibrate_button.click()
    window._on_calibration_changed(False)
    assert rung == [True]
    # And the next one that finishes rings again.
    window._on_calibration_changed(True)
    window._on_calibration_changed(False)
    assert rung == [True, True]


def test_every_setting_is_logged_when_a_calibration_starts(worker, monkeypatch):
    """What a report was made under has to be readable off it later, all of
    it: the calibration's own choices, the camera's, live view's, and the
    panel's, one to a line."""
    from scanny.camera.nikon import Setting

    rig = _Rig(PLACES)
    rig.model = "Nikon D750"
    rig.exposure_preview = True
    monkeypatch.setattr(
        rig,
        "settings",
        lambda: [
            Setting(0x500D, "Shutter speed", 60, "1/60", True, ()),
            Setting(0x500E, "Exposure mode", 1, "Manual", False, ()),
        ],
        raising=False,
    )
    worker._camera = rig
    worker.set_integration(True, 3)
    for _ in range(4):
        worker._grab()
    worker.set_focus_regions(_regions_round(PLACES))
    reports = []
    worker.calibrationReady.connect(reports.append)
    panel = [("Focus", "coarse increment", "500 steps"), ("View", "shown", "Rotated")]
    worker.start_calibration(STEP, "worst", 65, True, 1, "", panel)
    worker.cancel_calibration()
    said = [line.split("  ", 1)[1].strip() for line in reports[-1].log]
    settings = [line for line in said if line.startswith("Setting  ")]
    for wanted in (
        "Setting  Calibration > regions: 3",
        "Setting  Calibration > aim for: the best worst region",
        "Setting  Calibration > turn back below: 65% of the best",
        f"Setting  Calibration > walk in steps of: {STEP}",
        "Setting  Calibration > measure depths for levelling: yes",
        "Setting  Calibration > hurry through soft focus: off",
        "Setting  Camera > model: Nikon D750",
        "Setting  Exposure > Shutter speed: 1/60",
        "Setting  Exposure > Exposure mode: Manual (set on the body)",
        "Setting  Live view > exposure preview: yes",
        "Setting  Live view > integrate: 3 frames",
        "Setting  Live view > skip repeated frames: yes",
        "Setting  Focus > coarse increment: 500 steps",
        "Setting  View > shown: Rotated",
        "Setting  Capture > mirror-up delay: off",
    ):
        assert wanted in settings, wanted
    assert any(line.startswith("Setting  Calibration > region 3: x ") for line in settings)
    # A body that cannot answer one of them does not stop the calibration.
    assert any(
        line.startswith("Setting  Camera > firmware: could not be read") for line in settings
    )
    assert f"Settings at the start, {len(settings)} of them:" in said


def test_the_panel_sends_every_focus_increment_and_the_view_with_a_calibration(window):
    asked = []
    window.requestCalibration.connect(lambda *args: asked.append(args[6]))
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    window._turn_view(1)
    window.calibrate_button.click()
    sent = {(group, name): value for group, name, value in asked[0]}
    for increment in ("minimum", "fine", "medium", "coarse"):
        assert ("Focus", f"{increment} increment") in sent
    assert sent[("View", "shown")] == window._orientation.describe()
    assert ("Capture", "autofocus before shooting") in sent
