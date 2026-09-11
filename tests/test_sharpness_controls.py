"""The window's side of the sharpness meter.

The reading on its own is a number with no units and no scale; what makes it
usable is the comparison with the best this view has managed. So the panel's
job is to show that comparison, and to stop showing anything at all when the
meter is off.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.ui import main_window as mw  # noqa: E402


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


def test_measuring_is_off_until_it_is_asked_for(window):
    assert not window.measure_sharpness.isChecked()
    assert not window.sharpness_trend.isVisible()
    assert not window.sharpness_label.isVisible()


def test_switching_it_on_asks_the_worker_and_shows_the_readout(window):
    asked = []
    window.requestSharpness.connect(asked.append)
    window.measure_sharpness.setChecked(True)
    QApplication.processEvents()
    assert asked == [True]
    assert window.sharpness_trend.isVisible()


def test_the_numbers_say_where_the_reading_sits_against_the_best(window):
    window.measure_sharpness.setChecked(True)
    window._on_sharpness(80.0, 100.0)
    assert "80" in window.sharpness_label.text()
    assert "100" in window.sharpness_label.text()
    assert "80%" in window.sharpness_label.text()


def test_a_new_best_says_so_rather_than_reading_a_hundred_percent(window):
    """While focus is improving every reading is the best one, and a bar
    pinned at the top is exactly the readout that says nothing."""
    window.measure_sharpness.setChecked(True)
    window._on_sharpness(140.0, 140.0)
    assert "best so far" in window.sharpness_label.text()


def test_every_reading_is_plotted_even_between_the_written_ones(window):
    """The line gets all of them; the text is rewritten five times a second."""
    window.measure_sharpness.setChecked(True)
    for value in (100.0, 110.0, 120.0, 130.0):
        window._on_sharpness(value, 130.0)
    assert window.sharpness_trend.values == [100.0, 110.0, 120.0, 130.0]


def test_starting_again_clears_the_line(window):
    window.measure_sharpness.setChecked(True)
    window._on_sharpness(100.0, 100.0)
    window._on_sharpness(0.0, 0.0)
    assert window.sharpness_trend.values == []


def test_a_view_with_no_reading_yet_says_so(window):
    window.measure_sharpness.setChecked(True)
    window._on_sharpness(0.0, 0.0)
    assert "aiting" in window.sharpness_label.text()


def test_the_best_can_be_forgotten_from_the_panel(window):
    asked = []
    window.requestSharpnessReset.connect(lambda: asked.append(True))
    window.measure_sharpness.setChecked(True)
    window.reset_peak_button.click()
    assert asked == [True]


def test_the_reset_button_does_nothing_while_the_meter_is_off(window):
    assert not window.reset_peak_button.isEnabled()


def test_the_choice_is_remembered_for_next_time(window):
    window.measure_sharpness.setChecked(True)
    assert QSettings().value("focus/sharpness", False, bool) is True
    again = mw.MainWindow()
    try:
        assert again.measure_sharpness.isChecked()
    finally:
        again.deleteLater()


def test_the_readout_does_not_take_the_keyboard_from_the_image(window):
    from PySide6.QtCore import Qt

    assert window.measure_sharpness.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert window.reset_peak_button.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert window.sharpness_trend.focusPolicy() == Qt.FocusPolicy.NoFocus


# -- the measured area -------------------------------------------------------


def test_no_area_is_measured_until_one_is_asked_for(window):
    assert not window.measure_area.isChecked()
    assert window.view.measure_area is None


def test_the_area_only_applies_while_measuring(window):
    """Its checkbox is dead until there is a reading for it to narrow."""
    assert not window.measure_area.isEnabled()
    window.measure_sharpness.setChecked(True)
    assert window.measure_area.isEnabled()


def test_switching_the_area_on_puts_one_up_to_start_from(window):
    """Better than an empty setting the user then has to discover a gesture for."""
    window.measure_sharpness.setChecked(True)
    window.measure_area.setChecked(True)
    area = window.view.measure_area
    assert area is not None
    x, y, w, h = area
    assert 0 < w < 1 and 0 < h < 1
    assert x + w / 2 == pytest.approx(0.5), "it should start in the middle"


def test_the_worker_is_told_which_part_to_read(window):
    asked = []
    window.requestSharpnessArea.connect(asked.append)
    window.measure_sharpness.setChecked(True)
    window.measure_area.setChecked(True)
    assert asked[-1] == window.view.measure_area


def test_shift_dragging_an_area_switches_measuring_on_by_itself(window):
    """Drawing the box is unambiguous enough to be the whole request."""
    asked = []
    window.requestSharpnessArea.connect(asked.append)
    window.view.measureAreaSelected.emit(0.2, 0.3, 0.4, 0.25)
    assert window.measure_sharpness.isChecked()
    assert window.measure_area.isChecked()
    assert window.view.measure_area == (0.2, 0.3, 0.4, 0.25)
    assert asked[-1] == (0.2, 0.3, 0.4, 0.25)


def test_switching_the_area_off_goes_back_to_the_whole_frame(window):
    asked = []
    window.view.measureAreaSelected.emit(0.2, 0.3, 0.4, 0.25)
    window.requestSharpnessArea.connect(asked.append)
    window.measure_area.setChecked(False)
    assert asked[-1] is None
    assert window.view.measure_area is None


def test_switching_measuring_off_takes_the_box_off_the_image_too(window):
    window.view.measureAreaSelected.emit(0.2, 0.3, 0.4, 0.25)
    window.measure_sharpness.setChecked(False)
    assert window.view.measure_area is None


def test_the_area_is_remembered_for_next_time(window):
    window.view.measureAreaSelected.emit(0.2, 0.3, 0.4, 0.25)
    again = mw.MainWindow()
    try:
        assert again.measure_area.isChecked()
        assert again.view.measure_area == pytest.approx((0.2, 0.3, 0.4, 0.25))
    finally:
        again.deleteLater()


def test_a_nonsense_stored_area_is_ignored(window):
    QSettings().setValue(mw._MEASURE_AREA_KEY, "not, an, area")
    again = mw.MainWindow()
    try:
        again.measure_sharpness.setChecked(True)
        again.measure_area.setChecked(True)
        assert again.view.measure_area == mw._DEFAULT_MEASURE_AREA
    finally:
        again.deleteLater()


# -- the trend line ----------------------------------------------------------


@pytest.fixture
def trend(app):
    from scanny.ui.trend import TrendGraph

    made = TrendGraph()
    made.resize(280, 52)
    made.show()
    QApplication.processEvents()
    yield made
    made.hide()
    made.deleteLater()


def test_the_line_keeps_only_what_it_can_draw(trend):
    for value in range(500):
        trend.add(float(value))
    kept = trend.values
    assert 0 < len(kept) <= 200
    assert kept[-1] == 499.0, "the newest reading has to be one of them"


def test_a_flat_reading_does_not_divide_by_its_own_range(trend):
    """Nothing changing is a perfectly ordinary thing for it to be shown."""
    for _ in range(5):
        trend.add(120.0)
    trend.repaint()


def test_one_reading_is_not_yet_a_line(trend):
    trend.add(120.0)
    trend.repaint()


def test_an_empty_line_draws_the_placeholder(trend):
    trend.clear()
    trend.repaint()
    assert trend.values == []


# -- hunting for focus -------------------------------------------------------


def test_walking_is_offered_only_once_there_is_a_reading_to_walk_against(window):
    assert not window.fine_tune_button.isEnabled()
    window.measure_sharpness.setChecked(True)
    assert window.fine_tune_button.isEnabled()


def test_it_walks_in_the_minimum_increment_from_the_panel(window):
    """Nothing here has a step count of its own: it is the user's number, so
    making the walk finer is a matter of changing what "minimum" means."""
    asked = []
    window.requestFineTune.connect(asked.append)
    window.measure_sharpness.setChecked(True)
    window._focus_steps["minimum"].setValue(4)
    window.fine_tune_button.click()
    assert asked == [4]


def test_the_button_becomes_the_way_to_stop_it(window):
    window.measure_sharpness.setChecked(True)
    started = window.fine_tune_button.text()

    window.fine_tune_button.click()
    window._on_hunt_changed(True)
    assert window.fine_tune_button.text() == "Stop"

    stopped = []
    window.requestHuntCancel.connect(lambda: stopped.append(True))
    window.fine_tune_button.click()
    assert stopped == [True]

    window._on_hunt_changed(False)
    assert window.fine_tune_button.text() == started


def test_a_walk_that_ends_on_its_own_gives_the_button_back(window):
    window.measure_sharpness.setChecked(True)
    window._on_hunt_changed(True)
    asked = []
    window.requestFineTune.connect(asked.append)
    window._on_hunt_changed(False)
    window.fine_tune_button.click()
    assert asked, "the button has to start a walk again, not stop one"


# -- what the worker is told at startup --------------------------------------


class _Slot:
    """Stands in for one of the worker's slots or signals.

    Connected to as a slot it records the calls; connected to as a signal it
    swallows them, which is all the window's own slots need from it here.
    """

    def __init__(self, calls: list, name: str) -> None:
        self._calls = calls
        self._name = name

    def __call__(self, *args) -> None:
        self._calls.append((self._name, args))

    def connect(self, *_args) -> None:
        pass


class _RecordingWorker:
    """A worker that only remembers what the window asked it to do."""

    def __init__(self) -> None:
        self.calls: list = []

    def __getattr__(self, name: str) -> _Slot:
        return _Slot(self.calls, name)


class _IdleThread:
    """A thread that never runs: the fake worker has nothing to run there."""

    def __init__(self, _parent=None) -> None:
        pass

    def __getattr__(self, name: str) -> _Slot:
        return _Slot([], name)


@pytest.fixture
def started(app, monkeypatch):
    """A window with the real worker wiring, over a worker that only listens."""
    worker = _RecordingWorker()
    monkeypatch.setattr(mw, "CameraWorker", lambda: worker)
    monkeypatch.setattr(mw, "QThread", _IdleThread)
    QSettings().clear()
    yield worker
    QSettings().clear()


def test_a_remembered_choice_reaches_the_worker_at_startup(started):
    """Otherwise the panel says it is measuring while nothing measures.

    The checkbox is restored as the panel is built, which is before there is a
    worker to hear about it, so the request has to go out again once there is.
    """
    QSettings().setValue("focus/sharpness", True)
    made = mw.MainWindow()
    try:
        assert made.measure_sharpness.isChecked()
        assert ("set_sharpness", (True,)) in started.calls
    finally:
        made.deleteLater()


def test_a_remembered_area_reaches_the_worker_at_startup(started):
    QSettings().setValue("focus/sharpness", True)
    QSettings().setValue("focus/sharpness_area", True)
    QSettings().setValue(mw._MEASURE_AREA_KEY, "0.2,0.3,0.4,0.25")
    made = mw.MainWindow()
    try:
        assert ("set_sharpness_area", ((0.2, 0.3, 0.4, 0.25),)) in started.calls
    finally:
        made.deleteLater()


def test_measuring_left_off_is_not_switched_on_at_startup(started):
    made = mw.MainWindow()
    try:
        assert ("set_sharpness", (False,)) in started.calls
        assert ("set_sharpness_area", (None,)) in started.calls
    finally:
        made.deleteLater()
