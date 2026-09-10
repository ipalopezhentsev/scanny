"""Tests for measuring how far apart, in focus, a few chosen places are.

Three halves, if that is allowed. The arithmetic has no camera in it and runs
against curves built to a known shape. The window's side is checked for the
one gesture that has to be hard to make by accident and the one thing that has
to be forgotten when a point moves. And then the whole of it through the
worker against the simulated lens from :mod:`tests.test_depth`, because the
part worth checking is not the peak-finding -- that is shared with the depth
map and tested there -- but that five boxes read off one pass come back in an
order, with the gaps between them right.
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

from scanny.ui import main_window as mw  # noqa: E402
from scanny.ui.points import (  # noqa: E402
    MAX_POINTS,
    Point,
    PointSurvey,
    area_for,
    ordering,
    summarise,
)
from test_depth import (  # noqa: E402
    TALL,
    WIDE,
    _blur,
    _image,
    _Lens,
    _ready,
    _texture,
)

from scanny.camera.nikon import LiveViewFrame  # noqa: E402
from scanny.ui.depth import Sweep  # noqa: E402


# -- the boxes ---------------------------------------------------------------


def test_a_box_is_square_on_screen_and_not_in_fractions():
    """A box that is 0.12 of the width and 0.12 of the height of a 16:9 frame
    is half as tall as it is wide, and reads a strip rather than a place."""
    left, top, across, down = area_for(0.5, 0.5, 0.12, aspect=16 / 9)
    assert across == pytest.approx(0.12)
    assert down == pytest.approx(0.12 * 16 / 9)
    # In pixels on a 640x360 frame, that is square.
    assert across * 640 == pytest.approx(down * 360)
    assert (left, top) == pytest.approx((0.5 - across / 2, 0.5 - down / 2))


def test_a_box_at_the_edge_stays_inside_the_picture():
    for x, y in ((0.0, 0.0), (1.0, 1.0), (0.02, 0.98)):
        left, top, across, down = area_for(x, y, 0.2, aspect=16 / 9)
        assert 0.0 <= left and left + across <= 1.0 + 1e-9
        assert 0.0 <= top and top + down <= 1.0 + 1e-9


# -- where a point lives -----------------------------------------------------

_WHOLE = (0.0, 0.0, 1.0, 1.0)


def test_a_point_on_the_whole_frame_is_where_it_was_put():
    assert Point(0.1, 0.9).seen_in(_WHOLE) == pytest.approx((0.1, 0.9))


def test_magnifying_moves_a_point_across_the_screen_rather_than_with_it():
    """The failure this was rebuilt for. Two points near opposite corners,
    magnified onto the first of them: the first is now the middle of the
    picture and the second is not on it at all. Screen fractions said both
    were exactly where they had been, which is to say on two places nobody
    chose."""
    near_corner, far_corner = Point(0.1, 0.1), Point(0.9, 0.9)
    onto_the_first = (0.05, 0.05, 0.1, 0.1)
    assert near_corner.seen_in(onto_the_first) == pytest.approx((0.5, 0.5))
    assert far_corner.seen_in(onto_the_first) is None


def test_a_point_lands_where_the_crop_says_wherever_the_crop_is():
    crop = (0.4, 0.25, 0.2, 0.5)
    assert Point(0.5, 0.5).seen_in(crop) == pytest.approx((0.5, 0.5))
    assert Point(0.4, 0.25).seen_in(crop) == pytest.approx((0.0, 0.0))
    assert Point(0.6, 0.75).seen_in(crop) == pytest.approx((1.0, 1.0))
    assert Point(0.61, 0.5).seen_in(crop) is None


# -- placing and ordering ----------------------------------------------------


def _hill(position: float, peak: float, height: float = 200.0) -> float:
    return height * float(np.exp(-((position - peak) / 300.0) ** 2))


def _survey(peaks, positions=range(500, 3600, 100), heights=None) -> PointSurvey:
    points = [Point(0.1 + 0.2 * index, 0.5) for index in range(len(peaks))]
    survey = PointSurvey(points)
    heights = heights or [200.0] * len(peaks)
    for position in positions:
        survey.add(
            position,
            [_hill(position, peak, tall) for peak, tall in zip(peaks, heights)],
        )
    return survey


def test_each_point_is_placed_where_its_reading_peaked():
    found = _survey([1000, 1600, 2600]).found()
    assert [round(one.steps) for one in found] == [1000, 1600, 2600]
    assert all(one.known for one in found)


def test_they_come_back_ranked_nearest_first_with_the_gaps():
    """Which is the whole of what was asked for: the order, and how many steps
    are between them."""
    found = _survey([2600, 1000, 1600]).found()
    assert [one.rank for one in found] == [2, 0, 1]
    assert [round(one.behind_nearest) for one in found] == [1600, 0, 600]
    assert [one.behind_previous for one in found][1] is None
    assert round(found[2].behind_previous) == 600
    assert ordering(found).startswith("2: nearest")


def test_a_point_with_nothing_to_focus_on_is_left_out_of_the_order():
    survey = PointSurvey([Point(0.2, 0.5), Point(0.8, 0.5)])
    rng = np.random.default_rng(3)
    for position in range(500, 3600, 100):
        survey.add(position, [_hill(position, 1500), float(rng.normal(0, 0.0))])
    found = survey.found()
    assert found[0].known and found[0].rank == 0
    assert not found[1].known and found[1].rank is None
    assert "unplaced" in ordering(found)
    assert "left out of the order" in summarise(found[1], 2)


def test_one_faint_point_among_bright_ones_is_still_believed():
    """The opposite of what a grid wants. A zone of a grid that reads a
    hundredth of the rest of the frame has nothing in it; a place a person
    pointed at that reads a hundredth of the others is a dim thing they
    pointed at on purpose."""
    found = _survey([1000, 2000], heights=[200.0, 2.0]).found()
    assert all(one.known for one in found)
    assert round(found[1].behind_nearest) == 1000


def test_the_answer_is_finer_than_the_step_it_was_sampled_at():
    found = _survey([1450, 2350], positions=range(500, 3600, 200)).found()
    assert abs(found[0].steps - 1450) < 40
    assert not any(one.steps % 200 == 0 for one in found)


def test_a_peak_at_the_end_of_the_readings_says_the_sweep_stopped_short():
    """A hill can only be placed by seeing it fall away on both sides, so a
    best reading at either end of a pass is a bracket that missed."""
    over = _survey([1500, 2200], positions=range(500, 3600, 100))
    assert over.escaping() == (False, False)
    # Cut the readings off before the second peak, and after the first.
    short = _survey([1500, 2200], positions=range(500, 1900, 100))
    assert short.escaping()[0] is True
    late = _survey([1500, 2200], positions=range(1600, 3600, 100))
    assert late.escaping()[1] is True


def test_a_pass_can_take_more_stops_but_only_forward():
    sweep = Sweep(0, 50, 4)
    for _ in range(3):
        assert sweep.took_one() == 50
    sweep.keep_going(2)
    assert sweep.took_one() == 50
    assert not sweep.done
    assert sweep.position == 200
    # And not once it is over: what ended it was the lens, not the plan.
    sweep.blocked()
    sweep.keep_going(5)
    assert sweep.done


def test_the_stretch_worth_sweeping_again_covers_every_point():
    survey = _survey([1000, 2600])
    span = survey.interesting(150)
    assert span is not None
    assert span[0] <= 1000 and span[1] >= 2600


def test_nothing_read_is_no_answer_rather_than_a_wrong_one():
    survey = PointSurvey([Point(0.2, 0.5), Point(0.8, 0.5)])
    assert [one.steps for one in survey.found()] == [None, None]
    assert survey.interesting(100) is None
    survey.add(1000, [1.0, 2.0])
    assert [one.steps for one in survey.found()] == [None, None]


def test_a_reading_per_point_and_no_more():
    survey = PointSurvey([Point(0.2, 0.5), Point(0.8, 0.5)])
    with pytest.raises(ValueError):
        survey.add(100, [1.0])


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
def window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    QSettings().clear()
    made = mw.MainWindow()
    made.show()
    QApplication.processEvents()
    yield made
    made.hide()
    made.deleteLater()
    QApplication.processEvents()


def _showing(window) -> None:
    """Give the picture a frame, so its gestures have somewhere to land."""
    window.view.set_frame(_Lens().live_view_frame())
    window.view.resize(640, 400)
    window.view.repaint()


def _click(view, x: float, y: float, ctrl: bool) -> None:
    """A click at a fraction of the displayed picture, with ctrl or without."""
    spot = QPoint(
        view._target.x() + int(x * view._target.width()),
        view._target.y() + int(y * view._target.height()),
    )
    view.mousePressEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(spot),
            QPointF(view.mapToGlobal(spot)),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier
            if ctrl
            else Qt.KeyboardModifier.NoModifier,
        )
    )


def test_a_point_needs_two_before_it_will_measure_anything(window):
    assert not window.points_button.isEnabled()
    window._on_point_placed(0.3, 0.5)
    assert not window.points_button.isEnabled()
    window._on_point_placed(0.7, 0.5)
    assert window.points_button.isEnabled()


def test_ctrl_clicking_the_picture_places_a_point_and_again_removes_it(window):
    """Through the gesture itself, because the gesture is the feature."""
    _showing(window)
    _click(window.view, 0.3, 0.5, ctrl=True)
    _click(window.view, 0.7, 0.5, ctrl=True)
    assert len(window._points) == 2
    _click(window.view, 0.302, 0.503, ctrl=True)
    assert len(window._points) == 1
    assert window._points[0][0] == pytest.approx(0.7, abs=0.01)


def test_it_will_not_take_more_points_than_it_can_measure(window):
    for index in range(MAX_POINTS + 3):
        window._on_point_placed(0.1 + 0.1 * index, 0.5)
    assert len(window._points) == MAX_POINTS


def test_moving_a_point_forgets_what_was_measured(window):
    """A point that moved is a different question, and readings either side of
    that are of different parts of the picture."""
    window._on_point_placed(0.3, 0.5)
    window._on_point_placed(0.7, 0.5)
    window._found = [object(), object()]
    window._on_point_placed(0.5, 0.5)
    assert window._found is None


def test_the_points_are_sent_to_the_worker_with_the_box_they_are_read_in(window):
    asked = []
    window.requestFocusPoints.connect(lambda points, box: asked.append((points, box)))
    window.points_box_size.setValue(20)
    window._on_point_placed(0.3, 0.5)
    assert asked[-1][0] == [(0.3, 0.5)]
    assert asked[-1][1] == pytest.approx(0.20)


def _magnified_onto(window, x: float, y: float, part: int = 10) -> None:
    """Show the window a frame magnified onto ``(x, y)`` of the whole frame."""
    frame = _cropped_frame(
        np.full((TALL, WIDE), 128.0),
        int(x * _Apart.FRAME[0]),
        int(y * _Apart.FRAME[1]),
        _Apart.FRAME[0] // part,
        _Apart.FRAME[1] // part,
        (int(x * _Apart.FRAME[0]), int(y * _Apart.FRAME[1])),
    )
    window.view.show_frame(frame, QImage.fromData(frame.jpeg, "JPG"))
    window.view.resize(640, 400)
    window.view.repaint()


def test_a_point_keeps_its_subject_when_the_view_magnifies(window):
    """The whole complaint this answers. Two points near opposite corners of an
    unmagnified frame; magnify onto the first and it is the middle of the
    picture, not a tenth of the way across it, and the second is off screen
    rather than sitting on whatever the corner of the magnified strip is."""
    _showing(window)
    _click(window.view, 0.1, 0.1, ctrl=True)
    _click(window.view, 0.9, 0.9, ctrl=True)
    assert window._points[0] == pytest.approx((0.1, 0.1), abs=0.01)

    _magnified_onto(window, 0.1, 0.1)
    window._draw_points()
    drawn = window.view._points
    assert [one[2] for one in drawn] == [1], "only the first should be on screen"
    assert drawn[0][0] == pytest.approx(0.5, abs=0.03)
    assert drawn[0][1] == pytest.approx(0.5, abs=0.03)
    # Both are still points, and the navigator still shows both -- magnified it
    # is the only place the one that is off screen can be seen at all.
    assert len(window._points) == 2
    assert len(window.navigator._points) == 2


def test_a_point_placed_while_magnified_is_kept_where_the_sensor_is(window):
    """So that zooming back out puts it back on the thing it was put on."""
    _showing(window)
    _magnified_onto(window, 0.25, 0.75, part=4)
    _click(window.view, 0.5, 0.5, ctrl=True)
    assert window._points[0] == pytest.approx((0.25, 0.75), abs=0.01)
    # And it can be taken away again by the ring that is under the pointer,
    # which is what reach means when the view is magnified.
    _click(window.view, 0.5, 0.5, ctrl=True)
    assert window._points == []


def test_the_readout_says_how_many_points_the_magnification_hid(window):
    _showing(window)
    _click(window.view, 0.1, 0.1, ctrl=True)
    _click(window.view, 0.9, 0.9, ctrl=True)
    _magnified_onto(window, 0.1, 0.1)
    window._show_points()
    assert "off screen" in window.points_result.text()


def test_a_plain_click_places_nothing(window):
    """A stray click that moved a point would silently invalidate a
    measurement that took a minute to make, which is why the gesture wants a
    modifier on it."""
    _showing(window)
    _click(window.view, 0.3, 0.5, ctrl=False)
    assert window._points == []
    # And it is still the ordinary click it always was: it opens a drag.
    assert window.view._drag_origin is not None


def test_the_shape_of_a_scan_is_settled_before_it_starts(window):
    window._on_point_placed(0.3, 0.5)
    window._on_point_placed(0.7, 0.5)
    window._on_point_scan_changed(True)
    assert window.points_button.text() == "Stop"
    assert not window.points_stops.isEnabled()
    assert not window.points_box_size.isEnabled()
    assert not window.points_clear.isEnabled()
    window._on_point_scan_changed(False)
    assert window.points_stops.isEnabled()


def test_asking_for_a_scan_sends_the_lens_minimum_step(window):
    asked = []
    window.requestPointScan.connect(lambda *args: asked.append(args))
    window._focus_steps["minimum"].setValue(18)
    window.points_stops.setValue(24)
    window.points_passes.setValue(2)
    window._on_point_placed(0.3, 0.5)
    window._on_point_placed(0.7, 0.5)
    window.points_magnified.setChecked(False)
    window.points_button.click()
    assert asked == [(24, 2, 18, False, False, window.points_around.value())]


def test_asking_for_a_magnified_scan_sends_the_bracket_around_autofocus(window):
    asked = []
    window.requestPointScan.connect(lambda *args: asked.append(args))
    window.points_magnified.setChecked(True)
    window.points_around.setValue(650)
    window.points_stops.setValue(30)
    window._on_point_placed(0.3, 0.5)
    window._on_point_placed(0.7, 0.5)
    window.points_button.click()
    assert asked[-1][4] is True and asked[-1][5] == 650
    # And it says so about its own numbers: a magnified scan never parks, so
    # it has no near stop to count from.
    assert window._points_datum == "where the sweep began"


# -- the whole thing, through the worker -------------------------------------


@pytest.fixture
def worker():
    from scanny.ui.worker import CameraWorker

    return CameraWorker()


def test_five_points_come_back_in_the_right_order_with_the_right_gaps(worker):
    """The whole of what was asked for, through the whole of the machinery: a
    lens whose useful travel is in the far half of it, five places across a
    scene that is nearer on one side, and one sweep to place them all."""
    lens = _Lens(
        travel=24000, start=9000, scene=(14000.0, 22000.0), softness=900.0
    )
    _ready(worker, lens)
    # On the middles of five of the scene's eight bands, so that each box of
    # 0.12 sits inside one band of 0.125 rather than straddling two. A box
    # holding two depths gets an answer between them -- see the test below.
    places = [(0.1875, 0.5), (0.3125, 0.5), (0.5625, 0.5), (0.6875, 0.5), (0.8125, 0.5)]
    worker.set_focus_points(places, 0.12)
    found: "list[object]" = []
    worker.pointsFound.connect(found.append)
    worker.start_point_scan(40, 3, 6, False)
    for _ in range(200000):
        if worker._sweep is None:
            break
        worker._grab()
    assert worker._sweep is None, "the scan has to finish on its own"

    results = found[-1]
    assert all(one.known for one in results), "a point was left unplaced"
    # The scene runs from near on the left to far on the right, so the points
    # come back in the order they were put down.
    assert [one.rank for one in results] == [0, 1, 2, 3, 4]
    for index, one in enumerate(results):
        middle = int(places[index][0] * WIDE)
        want = lens.scene.best[lens.scene.band(middle)]
        assert abs(one.steps - want) < 250, (
            f"point {index + 1} read {one.steps:.0f}, not {want:.0f}"
        )


def test_it_refuses_a_scan_of_fewer_than_two_points(worker):
    lens = _ready(worker, _Lens())
    failures: "list[str]" = []
    worker.failed.connect(failures.append)
    worker.set_focus_points([(0.5, 0.5)], 0.12)
    worker.start_point_scan(20, 1, 6, False)
    assert worker._sweep is None
    assert failures and "two points" in failures[0]


def test_moving_the_points_stops_a_scan_and_forgets_it(worker):
    lens = _ready(worker, _Lens(travel=6000, start=3000))
    worker.set_focus_points([(0.3, 0.5), (0.7, 0.5)], 0.12)
    worker.start_point_scan(60, 1, 6, False)
    for _ in range(60):
        worker._grab()
    assert worker._sweep is not None
    worker.set_focus_points([(0.3, 0.5), (0.7, 0.5), (0.5, 0.5)], 0.12)
    assert worker._sweep is None
    assert worker._point_survey is None


def test_a_scan_stops_when_something_else_moves_the_view(worker):
    lens = _ready(worker, _Lens(travel=6000, start=3000))
    worker.set_focus_points([(0.3, 0.5), (0.7, 0.5)], 0.12)
    worker.start_point_scan(60, 1, 6, False)
    for _ in range(60):
        worker._grab()
    assert worker._sweep is not None
    worker.set_zoom(4)
    assert worker._sweep is None


# -- the doubt, and refusing to order what cannot be told apart --------------


def _noisy(peaks, spread=900.0, grain=0.0, seed=5, step=20):
    """Two broad peaks with grain on the readings, as a real lens gives."""
    rng = np.random.default_rng(seed)
    points = [Point(0.1 + 0.2 * index, 0.5) for index in range(len(peaks))]
    survey = PointSurvey(points)
    for position in range(2000, 4001, step):
        survey.add(
            position,
            [
                200 * float(np.exp(-((position - peak) / spread) ** 2))
                + float(rng.normal(0, grain))
                for peak in peaks
            ],
        )
    return survey


def test_a_clean_curve_comes_with_a_small_doubt_on_it():
    found = _noisy([2900, 3100], grain=0.0).found()
    assert all(one.doubt < 5 for one in found)
    assert found[0].apart_from(found[1])
    assert "give or take" in summarise(found[0], 1, found)


def test_two_points_inside_the_noise_are_not_put_in_an_order():
    """The failure this was rebuilt for: a broad peak read through grain, two
    things closer together than the reading can resolve, and an answer given
    in the same confident voice as one it is sure of."""
    found = _noisy([2990, 3010], grain=25.0).found()
    assert all(one.known for one in found), "they should still be placed"
    assert max(one.doubt for one in found) > 20, "the doubt should be large"
    assert not found[0].apart_from(found[1])
    assert "=" in ordering(found), ordering(found)
    assert "Too close to call" in summarise(found[0], 1, found)


def test_the_doubt_grows_with_the_grain_on_the_readings():
    quiet = _noisy([2900, 3100], grain=1.0).found()
    loud = _noisy([2900, 3100], grain=30.0).found()
    assert max(one.doubt for one in loud) > 4 * max(one.doubt for one in quiet)


def test_a_curve_with_too_few_samples_admits_it_has_no_idea():
    survey = PointSurvey([Point(0.2, 0.5), Point(0.8, 0.5)])
    for position in (1000, 1100, 1200):
        survey.add(position, [1.0, 2.0])
    assert all(not np.isfinite(one.doubt) or not one.known for one in survey.found())


def test_the_peak_is_found_from_the_whole_top_and_not_three_samples():
    """A broad peak finely sampled has a summit twenty samples wide, all within
    the noise of each other, so which is highest is the grain's choice."""
    from scanny.ui.depth import locate

    rng = np.random.default_rng(7)
    where = np.arange(2000.0, 4001.0, 20.0)
    truth = 3000.0
    errors = {}
    for label in ("centroid",):
        stack = np.stack(
            [
                200 * np.exp(-((where - truth) / 900.0) ** 2)
                + rng.normal(0, 8.0, where.shape)
                for _ in range(40)
            ],
            axis=1,
        )
        steps, _strength, _limit, width = locate(where, stack)
        errors[label] = float(np.std(steps[np.isfinite(steps)]))
        assert width[0] > 500, "a broad peak should report a broad top"
    # Argmax alone, on the same curves, wanders by hundreds of steps.
    rough = where[stack.argmax(axis=0)]
    assert errors["centroid"] < 0.5 * float(np.std(rough))


def test_the_next_pass_is_never_narrower_than_the_peak_is_wide():
    """A pass that fits inside the top of a hill is looking at a flat noisy
    line, and it answers with the noise: that is how refining makes a coarse
    answer worse instead of better."""
    survey = _noisy([2950, 3050], spread=900.0, grain=2.0)
    span = survey.interesting(20)
    assert span is not None
    assert span[1] - span[0] > 1000, span


def test_a_box_holding_two_depths_answers_with_something_between_them():
    """Not a fault, and worth knowing. The peak is the middle of the top of the
    curve, so a box straddling two things at different distances reports a
    reading-weighted average of the two rather than picking one. That is the
    right answer to "how far away is the stuff in this box" and the wrong
    answer to "how far away is that edge", and the box is the thing to move.
    """
    survey = PointSurvey([Point(0.5, 0.5)])
    for position in range(2000, 4001, 20):
        near = 200 * float(np.exp(-((position - 2800) / 900.0) ** 2))
        far = 200 * float(np.exp(-((position - 3200) / 900.0) ** 2))
        survey.add(position, [near + far])
    found = survey.found()[0]
    assert found.known
    assert 2900 < found.steps < 3100, found.steps


# -- the same question at full magnification ---------------------------------


class _Apart:
    """A body that magnifies and pans, over subjects at different distances.

    The one thing it has that :class:`test_depth._Lens` has not, and the whole
    reason it exists: **what the picture shows depends on where the camera is
    aimed**. The magnified view is a crop of the frame centred on the focus
    point, so a reading is a reading of whatever place that crop is over -- and
    a scan that forgets to pan reads the same subject five times and calls it
    five answers.

    Everything else is the same lens: the drive is bounded at both ends, live
    view runs a frame behind it, and the picture has grain on it, so the
    stillness watching and the grain measuring have something real to do.
    """

    live_view_active = True
    exposure_preview = True

    #: The body's own magnifications, and its frame in autofocus units.
    MAGNIFICATION = {0: 1.0, 2: 2.35, 3: 3.13, 4: 4.7, 5: 6.27, 6: 9.4, 7: 18.8}
    FRAME = (6016, 4016)

    def __init__(
        self,
        places,
        travel: int = 6000,
        start: int = 3000,
        softness: float = 40.0,
        lag: int = 1,
        noise: float = 1.5,
        af_error: int = 0,
    ) -> None:
        #: (x, y, best) each: where on the frame, and where on the travel it
        #: comes into focus.
        self.places = list(places)
        self.travel = travel
        self.position = start
        self.softness = softness
        self.noise = noise
        self.af_error = af_error
        self.lag = lag
        self.zoom = 0
        self.af = (self.FRAME[0] // 2, self.FRAME[1] // 2)
        #: What was asked of it, for tests that care that it was asked.
        self.drives: "list[int]" = []
        self.aimed: "list[tuple[int, int]]" = []
        self.zooms: "list[int]" = []
        self.focused = 0
        self._history = [start] * (lag + 1)
        self._textures: "dict[tuple, np.ndarray]" = {}
        self._rng = np.random.default_rng(11)

    # -- the camera's side ---------------------------------------------------

    def drive_focus(self, steps: int) -> bool:
        wanted = self.position + int(steps)
        self.position = min(max(wanted, 0), self.travel)
        self.drives.append(int(steps))
        return wanted == self.position

    def set_zoom_level(self, level: int) -> None:
        self.zoom = int(level)
        self.zooms.append(int(level))

    def zoom_level(self) -> int:
        return self.zoom

    def set_af_area(self, x: int, y: int) -> None:
        self.af = (int(x), int(y))
        self.aimed.append(self.af)

    def autofocus(self, timeout: float = 8.0) -> bool:
        """Land on whichever place the focus point is nearest, as AF does."""
        self.focused += 1
        self.position = min(
            max(int(self._nearest_place()[2]) + self.af_error, 0), self.travel
        )
        return True

    def live_view_frame(self) -> LiveViewFrame:
        self._history.append(self.position)
        shown = self._history[-(self.lag + 1)]
        cx, cy, w, h = self._crop()
        return _cropped_frame(self._picture(cx, cy, w, h, shown), cx, cy, w, h, self.af)

    def stop_live_view(self) -> None:
        pass

    def set_setting(self, name, value) -> None:
        pass

    def settings(self) -> list:
        return []

    def set_exposure_preview(self, enabled: bool) -> None:
        self.exposure_preview = enabled

    # -- the scene -----------------------------------------------------------

    def _crop(self) -> "tuple[int, int, int, int]":
        """The magnified view: centred on the focus point, kept on the frame."""
        magnification = self.MAGNIFICATION[self.zoom]
        w = int(self.FRAME[0] / magnification)
        h = int(self.FRAME[1] / magnification)
        cx = min(max(self.af[0], w // 2), self.FRAME[0] - w // 2)
        cy = min(max(self.af[1], h // 2), self.FRAME[1] - h // 2)
        return cx, cy, w, h

    def _nearest_place(self):
        cx, cy, _w, _h = self._crop()
        here = (cx / self.FRAME[0], cy / self.FRAME[1])
        return min(
            self.places,
            key=lambda place: (place[0] - here[0]) ** 2 + (place[1] - here[1]) ** 2,
        )

    def _picture(self, cx, cy, w, h, position) -> np.ndarray:
        """Detail belonging to this view, softened by how far focus is off it.

        The texture is remembered per view rather than made afresh, because two
        frames of a scene holding still have to differ by the grain and by
        nothing else -- that is what the grain is measured from.
        """
        key = (cx, cy, w, h)
        base = self._textures.get(key)
        if base is None:
            base = _texture(TALL, WIDE, seed=abs(hash(key)) % 9973)
            self._textures[key] = base
        best = self._nearest_place()[2]
        passes = min(abs(position - best) / self.softness, 7.99)
        return _blur(base, passes) + self._rng.normal(0, self.noise, (TALL, WIDE))


def _cropped_frame(pixels, cx, cy, w, h, af) -> LiveViewFrame:
    data = QByteArray()
    sink = QBuffer(data)
    sink.open(QBuffer.OpenModeFlag.WriteOnly)
    _image(pixels).save(sink, "JPG", 95)
    return LiveViewFrame(
        jpeg=bytes(data.data()), width=WIDE, height=TALL,
        image_width=_Apart.FRAME[0], image_height=_Apart.FRAME[1],
        crop_width=w, crop_height=h, crop_center_x=cx, crop_center_y=cy,
        af_width=324, af_height=270, af_x=af[0], af_y=af[1],
    )


def _run(worker, limit: int = 4000) -> None:
    """Turn the frame grab until the scan finishes on its own."""
    for _ in range(limit):
        if worker._sweep is None:
            return
        worker._grab()
    raise AssertionError("the scan did not finish")


def test_two_corners_are_measured_apart_with_the_camera_panning_between_them():
    """The whole of what was asked for. Two points near opposite corners of the
    frame, two hundred steps apart in focus, measured at the strongest
    magnification the body has -- where they are never on screen together."""
    from scanny.ui.worker import CameraWorker

    worker = CameraWorker()
    lens = _ready(worker, _Apart([(0.1, 0.15, 2600), (0.9, 0.85, 2800)]))
    worker.set_focus_points([(0.1, 0.15), (0.9, 0.85)], 0.2)
    found = []
    worker.pointsFound.connect(found.append)
    worker.start_point_scan(20, 1, 6, False, True, 500)
    _run(worker)

    results = found[-1]
    assert all(one.known for one in results), "a point was left unplaced"
    assert [one.rank for one in results] == [0, 1]
    # Counted from where the bracket began, which is 500 steps behind where
    # autofocus put the first point.
    assert abs(results[0].steps - 500) < 80, results[0].steps
    assert abs(results[1].behind_nearest - 200) < 80, results[1].behind_nearest


def test_it_magnifies_onto_each_point_in_turn_and_reads_it_there():
    """Panning is the whole mechanism, so it is worth checking it happened:
    both points were aimed at, at the strongest magnification, and neither was
    read off a frame showing the other."""
    from scanny.ui.worker import CameraWorker

    worker = CameraWorker()
    lens = _ready(worker, _Apart([(0.1, 0.15, 2600), (0.9, 0.85, 2800)]))
    worker.set_focus_points([(0.1, 0.15), (0.9, 0.85)], 0.2)
    worker.start_point_scan(12, 1, 6, False, True, 400)
    _run(worker)

    assert 7 in lens.zooms, "it never magnified"
    aimed = {spot for spot in lens.aimed}
    assert len(aimed) >= 3, aimed  # both points, and the one put back
    left = min(x for x, _y in aimed)
    right = max(x for x, _y in aimed)
    assert right - left > lens.FRAME[0] // 2, "it never panned across the frame"


def test_it_puts_the_view_back_where_it_found_it():
    """Leaving someone at 18.8x on the last point of a scan is leaving them
    somewhere they did not ask to be."""
    from scanny.ui.worker import CameraWorker

    worker = CameraWorker()
    lens = _ready(worker, _Apart([(0.2, 0.5, 2600), (0.8, 0.5, 2800)]))
    was = lens.af
    worker.set_focus_points([(0.2, 0.5), (0.8, 0.5)], 0.2)
    worker.start_point_scan(12, 1, 6, False, True, 400)
    _run(worker)
    assert lens.zoom == 0
    assert lens.af == was


def test_a_point_past_the_end_of_the_bracket_makes_the_sweep_carry_on():
    """Forward is free: the lens is already driving that way, so the stops
    added are in the same coordinate as the ones before them."""
    from scanny.ui.worker import CameraWorker

    worker = CameraWorker()
    # The second point is 900 steps behind the first, and the bracket is 500
    # either side of where autofocus lands on the first: it is outside.
    lens = _ready(worker, _Apart([(0.2, 0.5, 2400), (0.8, 0.5, 3300)]))
    worker.set_focus_points([(0.2, 0.5), (0.8, 0.5)], 0.2)
    found = []
    worker.pointsFound.connect(found.append)
    worker.start_point_scan(20, 1, 6, False, True, 500)
    _run(worker)

    results = found[-1]
    assert all(one.known for one in results), "a point was left unplaced"
    assert abs(results[1].behind_nearest - 900) < 120, results[1].behind_nearest


def test_a_scan_that_cannot_autofocus_says_so_rather_than_bracketing_nothing():
    from scanny.ui.worker import CameraWorker

    worker = CameraWorker()
    lens = _ready(worker, _Apart([(0.2, 0.5, 2600), (0.8, 0.5, 2800)]))
    lens.autofocus = lambda timeout=8.0: False
    said = []
    worker.status.connect(said.append)
    worker.set_focus_points([(0.2, 0.5), (0.8, 0.5)], 0.2)
    worker.start_point_scan(12, 1, 6, False, True, 400)
    assert worker._sweep is None
    assert any("autofocus could not find point 1" in line for line in said)
    assert lens.zoom == 0, "and it still put the view back"
