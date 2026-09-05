"""The window's side of frame integration: the switch, the count, and the cost.

Two things are worth holding onto here. The count is the user's, remembered
between runs like the focus increments are, and the frame rate the status bar
shows has to be the rate the *picture* updates at -- the camera still sends
thirty frames a second while they are being stacked, and reporting that would
be a lie about what is on screen.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.ui.integration import DEFAULT_FRAMES, MAX_FRAMES, MIN_FRAMES  # noqa: E402
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


def test_integration_is_off_until_it_is_asked_for(window):
    assert not window.integrate.isChecked()
    assert window.integrate_frames.value() == DEFAULT_FRAMES


def test_the_count_only_matters_while_it_is_on(window):
    assert not window.integrate_frames.isEnabled()
    window.integrate.setChecked(True)
    assert window.integrate_frames.isEnabled()


def test_switching_it_on_asks_the_worker_for_it(window):
    asked = []
    window.requestIntegration.connect(lambda on, n: asked.append((on, n)))
    window.integrate_frames.setValue(6)
    window.integrate.setChecked(True)
    assert asked[-1] == (True, 6)


def test_changing_the_count_asks_again(window):
    asked = []
    window.integrate.setChecked(True)
    window.requestIntegration.connect(lambda on, n: asked.append((on, n)))
    window.integrate_frames.setValue(12)
    assert asked[-1] == (True, 12)


def test_switching_it_off_asks_for_that_too(window):
    window.integrate.setChecked(True)
    asked = []
    window.requestIntegration.connect(lambda on, n: asked.append((on, n)))
    window.integrate.setChecked(False)
    assert asked[-1][0] is False


def test_the_count_cannot_be_set_past_what_the_integrator_accepts(window):
    window.integrate_frames.setValue(MAX_FRAMES + 100)
    assert window.integrate_frames.value() == MAX_FRAMES
    window.integrate_frames.setValue(0)
    assert window.integrate_frames.value() == MIN_FRAMES


def test_the_cost_is_shown_before_it_is_switched_on(window):
    """Ten frames is a third of a second an image; say so rather than let it
    be discovered."""
    window.integrate_frames.setValue(10)
    assert "3.0 fps" in window.integrate_hint.text()
    # Averaging ten frames divides the noise by the square root of ten.
    assert "3.2x less noise" in window.integrate_hint.text()


def test_the_choice_is_remembered_for_next_time(window):
    window.integrate_frames.setValue(16)
    window.integrate.setChecked(True)
    assert QSettings().value("liveview/integrate", False, bool) is True
    assert int(QSettings().value("liveview/frames")) == 16

    again = MainWindow()
    try:
        assert again.integrate.isChecked()
        assert again.integrate_frames.value() == 16
    finally:
        again.close()


def test_a_nonsense_stored_count_falls_back_to_the_default(window):
    QSettings().setValue("liveview/frames", "not a number")
    again = MainWindow()
    try:
        assert again.integrate_frames.value() == DEFAULT_FRAMES
    finally:
        again.close()


def test_the_status_bar_reports_the_rate_the_picture_updates_at(window):
    window._on_fps(7.5, 640, 424)
    assert window.fps_label.text() == "640x424  -  7.5 fps"

    window.integrate_frames.setValue(4)
    window.integrate.setChecked(True)
    window._on_fps(7.5, 640, 424)
    assert "7.5 fps" in window.fps_label.text()
    assert "4 frames integrated" in window.fps_label.text()


def test_the_rate_from_the_old_setting_is_not_left_on_screen(window):
    """It was measured before the change and is about to be wrong."""
    window._on_fps(30.0, 640, 424)
    window.integrate.setChecked(True)
    assert window.fps_label.text() == ""


def test_a_change_that_alters_nothing_leaves_the_rate_alone(window):
    # The count while integration is off: nothing on screen changes, so the
    # measured rate is still the truth.
    window._on_fps(30.0, 640, 424)
    window.integrate_frames.setValue(12)
    assert "30.0 fps" in window.fps_label.text()
