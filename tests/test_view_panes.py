"""The window's two read-off-the-picture panes: the navigator and the histogram.

Both are fed from the same frames the image is, and both have to be emptied
when the picture goes away: a map or a set of curves left over from the last
session looks exactly as live as a real one.

The other thing checked here is the rule the whole sidebar follows -- nothing
in it may take the keyboard away from the image, or the focus, pan and zoom
keys go dead until the image is clicked again.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.camera.nikon import LiveViewFrame  # noqa: E402
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
    # No worker: this is about what the window does with a frame, and there is
    # no reason to go near a camera to find out.
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    made = mw.MainWindow()
    made.worker = type("Worker", (), {"save_directory": "."})()
    made._live = True
    made.resize(1000, 800)
    made.show()
    QApplication.processEvents()
    # Not closed at the end: closing waits on a worker thread that was never
    # started, and the window goes out of scope with the test anyway.
    return made


def picture(shade: int = 0x606060) -> QImage:
    image = QImage(64, 42, QImage.Format.Format_RGB32)
    image.fill(shade)
    return image


def frame(*, crop=(6016, 3376), centre=(3008, 1688)) -> LiveViewFrame:
    return LiveViewFrame(
        jpeg=b"",
        width=640,
        height=424,
        image_width=6016,
        image_height=3376,
        crop_width=crop[0],
        crop_height=crop[1],
        crop_center_x=centre[0],
        crop_center_y=centre[1],
        af_width=324,
        af_height=270,
        af_x=centre[0],
        af_y=centre[1],
    )


def test_a_frame_reaches_both_panes(window):
    window._on_frame(frame(), picture())
    assert window.navigator.has_whole_frame
    assert window.histogram.counts is not None


def test_the_caption_says_what_the_curves_showed(window):
    window._on_frame(frame(), picture(0x000000))
    assert "crushed" in window.histogram_label.text()


def test_the_navigator_follows_a_magnified_frame(window):
    window._on_frame(frame(), picture())
    window._on_frame(frame(crop=(1504, 844), centre=(1504, 844)), picture(0x101010))
    assert window.navigator.crop == pytest.approx((0.125, 0.125, 0.25, 0.25))


def test_stopping_live_view_empties_them(window):
    window._on_frame(frame(), picture())
    window._on_live_view_changed(False)
    assert not window.navigator.has_whole_frame
    assert window.histogram.counts is None
    assert window.histogram_label.text() == "Waiting for a frame..."


def test_losing_the_camera_empties_them(window):
    window._on_frame(frame(), picture())
    window._on_disconnected()
    assert not window.navigator.has_whole_frame
    assert window.histogram.counts is None


def test_a_drag_on_the_navigator_becomes_a_request(window):
    asked = []
    window.requestMovePointInFrame.connect(lambda x, y: asked.append((x, y)))
    window._on_view_centre_moved(0.25, 0.75)
    assert asked == [(0.25, 0.75)]


def test_nothing_is_asked_for_while_live_view_is_off(window):
    """The same rule a click on the image follows: there is no view to move."""
    window._live = False
    asked = []
    window.requestMovePointInFrame.connect(lambda x, y: asked.append((x, y)))
    window._on_view_centre_moved(0.25, 0.75)
    assert asked == []


def test_neither_pane_takes_the_keyboard_from_the_image(window):
    for pane in (window.navigator, window.histogram):
        assert pane.focusPolicy() == Qt.FocusPolicy.NoFocus
