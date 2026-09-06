"""How big the window opens, and where it opens.

A first run has no answer to go on, so it works one out: the picture is drawn
as large as fits with its shape kept, and any mismatch between the window's
shape and the frame's comes back as black strips down two sides. Sizing the
image area to the frame starts those strips at nothing.

After that the question is settled by the user rather than by arithmetic --
whatever they dragged the window to is what they get back next time.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QByteArray, QSettings, QSize  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.ui import main_window as mw  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def app():
    made = QApplication.instance() or QApplication([])
    QApplication.setOrganizationName("scanny-tests")
    QApplication.setApplicationName("scanny-tests")
    yield made
    QSettings().clear()


@pytest.fixture(autouse=True)
def settings():
    """A clean store either side: a geometry left behind would size the
    windows the other test modules build."""
    QSettings().clear()
    yield QSettings()
    QSettings().clear()


@pytest.fixture
def build(monkeypatch):
    """Build windows without going near a camera."""
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)

    def make():
        window = mw.MainWindow()
        window.worker = type("Worker", (), {"save_directory": "."})()
        # What closeEvent waits on, which is normally the camera's thread.
        window._thread = type("Thread", (), {"wait": lambda self, ms: True})()
        return window

    return make


def test_the_image_area_opens_the_shape_of_a_frame(build):
    """No black strips: the picture fills what it is given, both ways."""
    window = build()
    beside, above = window._chrome()
    picture = QSize(window.width() - beside, window.height() - above)
    assert picture.width() / picture.height() == pytest.approx(
        mw._LIVE_VIEW_ASPECT, rel=0.01
    )


def test_the_image_widget_really_ends_up_that_shape(build):
    """The arithmetic above is only worth anything if the layout agrees."""
    window = build()
    window.show()
    QApplication.processEvents()
    assert window.view.width() / window.view.height() == pytest.approx(
        mw._LIVE_VIEW_ASPECT, rel=0.01
    )
    window.hide()


def test_the_window_is_not_opened_smaller_than_it_can_be_drawn(build):
    """The pinned panes and the image both have a size they insist on.

    Meeting either by stretching the window alone would put the strips back;
    they are answered in the picture's own height instead, so the shape
    survives.
    """
    window = build()
    smallest = window.minimumSizeHint()
    assert window.width() >= smallest.width()
    assert window.height() >= smallest.height()


def test_the_window_leaves_some_screen_around_it(build):
    """Short of the whole screen, so its own edges and the taskbar stay
    grabbable.

    Unless the screen is smaller than the window can be drawn at all, which is
    the one case where there is nothing to give: the minimum wins and the
    desktop loses.
    """
    window = build()
    room = QApplication.primaryScreen().availableGeometry()
    smallest = window.minimumSizeHint()
    assert window.width() <= max(room.width() * mw._SCREEN_SHARE, smallest.width())
    assert window.height() <= max(room.height() * mw._SCREEN_SHARE, smallest.height())


def test_closing_saves_the_size(build, settings):
    window = build()
    window.resize(1004, 668)
    window.closeEvent(QCloseEvent())
    stored = settings.value("window/geometry")
    assert isinstance(stored, (QByteArray, bytes))
    assert len(stored)


def test_the_size_is_saved_even_if_the_camera_will_not_let_go(build, settings):
    """The wait for the worker can time out; the geometry is written first."""
    window = build()
    window._thread = type("Stuck", (), {"wait": lambda self, ms: False, "quit": lambda self: None, "terminate": lambda self: None})()
    window.closeEvent(QCloseEvent())
    assert settings.value("window/geometry")


def test_a_saved_window_comes_back_instead_of_being_sized_again(build, settings):
    saved = build()
    saved.resize(1004, 668)
    saved.closeEvent(QCloseEvent())

    worked_one_out = []
    window = build()
    # Whatever the platform makes of the stored geometry -- an offscreen
    # screen may clamp it -- the point is that the fallback was not reached.
    window._size_that_fits_the_frame = lambda: worked_one_out.append(1) or QSize(800, 600)
    window._restore_geometry()
    assert not worked_one_out


def test_nonsense_in_the_store_is_not_fatal(build, settings):
    """A settings file from another version, or a truncated one."""
    settings.setValue("window/geometry", QByteArray(b"not a geometry"))
    window = build()
    beside, above = window._chrome()
    # Sized from the frame, exactly as on a first run.
    assert (window.width() - beside) / (window.height() - above) == pytest.approx(
        mw._LIVE_VIEW_ASPECT, rel=0.01
    )
