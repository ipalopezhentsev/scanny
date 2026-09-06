"""The window's side of the mirror-up delay.

The delay is remembered between runs like the other capture choices are, but
unlike them it is not the window's to promise: how many seconds are on offer
is the connected body's, so the control follows what the camera says it has.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.camera.nikon import NikonCamera  # noqa: E402
from scanny.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    made = QApplication.instance() or QApplication([])
    # Keep the test's settings out of the real ones.
    QApplication.setOrganizationName("scanny-tests")
    QApplication.setApplicationName("scanny-tests")
    QSettings().clear()
    yield made
    QSettings().clear()


@pytest.fixture
def window(app):
    QSettings().clear()
    made = MainWindow()
    made.show()
    QApplication.processEvents()
    yield made
    made.close()


def offered(window) -> "list[int]":
    box = window.shutter_delay
    return [box.itemData(i) for i in range(box.count())]


def test_no_delay_until_it_is_asked_for(window):
    assert window.shutter_delay.currentData() == 0
    assert window.shutter_delay.currentText() == "Off"


def test_every_delay_the_body_has_is_offered(window):
    assert offered(window) == list(NikonCamera.SHUTTER_DELAYS)


def test_choosing_a_delay_asks_the_worker_for_it(window):
    asked = []
    window.requestShutterDelay.connect(asked.append)
    window.shutter_delay.setCurrentIndex(offered(window).index(2))
    assert asked == [2]


def test_the_delay_is_remembered_between_runs(app):
    QSettings().clear()
    first = MainWindow()
    first.shutter_delay.setCurrentIndex(offered(first).index(3))
    first.close()

    second = MainWindow()
    try:
        assert second._stored_shutter_delay() == 3
        # Driven rather than waited for: a window makes its own connection to
        # whatever camera is plugged into this machine, and the test is about
        # what is remembered, not about what that camera happens to answer.
        second._on_shutter_delays(NikonCamera.SHUTTER_DELAYS, 0)
        assert second.shutter_delay.currentData() == 3
    finally:
        second.close()


def test_the_camera_says_which_delays_it_has(window):
    window._on_shutter_delays((0, 1), 0)
    assert offered(window) == [0, 1]
    assert window.shutter_delay.isEnabled()


def test_a_camera_already_set_up_for_this_keeps_its_delay(window):
    # Nothing has ever been chosen here, and the body has three seconds on it.
    asked = []
    window.requestShutterDelay.connect(asked.append)
    window._on_shutter_delays(NikonCamera.SHUTTER_DELAYS, 3)
    assert window.shutter_delay.currentData() == 3
    assert asked == [3]


def test_a_delay_chosen_here_outranks_the_one_on_the_camera(window):
    window.shutter_delay.setCurrentIndex(offered(window).index(1))
    window._on_shutter_delays(NikonCamera.SHUTTER_DELAYS, 3)
    assert window.shutter_delay.currentData() == 1


def test_a_body_that_cannot_be_asked_leaves_the_choice_for_the_next_one(window):
    window.shutter_delay.setCurrentIndex(offered(window).index(2))
    window._on_shutter_delays((), None)
    window._on_shutter_delays(NikonCamera.SHUTTER_DELAYS, 0)
    assert window.shutter_delay.currentData() == 2


def test_a_delay_this_body_lacks_is_not_left_selected(window):
    window.shutter_delay.setCurrentIndex(offered(window).index(3))
    asked = []
    window.requestShutterDelay.connect(asked.append)
    window._on_shutter_delays((0, 1), 0)
    # Nothing is silently shot with a delay the camera never took: the choice
    # falls back, and the worker is told what it fell back to.
    assert window.shutter_delay.currentData() == 0
    assert asked == [0]


def test_a_body_with_no_delay_at_all_cannot_be_asked_for_one(window):
    window._on_shutter_delays((), None)
    assert not window.shutter_delay.isEnabled()
    assert "no exposure delay mode" in window.shutter_delay_hint.text()
