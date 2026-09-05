"""The controls panel must never be squeezed to make itself fit.

With a camera connected the exposure form fills with eight rows, and by then
the panel wants more height than the window has. A plain layout answers that
by handing every widget less than it asked for, which flattens the spin boxes
and cuts the last line off every wrapped hint -- damage that reads as a broken
control rather than a panel that is simply too long for the window. The panel
scrolls instead.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QScrollArea,
    QSpinBox,
)

from scanny.camera.nikon import Setting  # noqa: E402
from scanny.ui import main_window as mw  # noqa: E402

#: What the sidebar looks like with a camera on the end of the cable: this is
#: the state the panel runs out of room in.
_CONNECTED = [
    Setting(0, name, 0, label, True, ((0, label), (1, other)))
    for name, label, other in (
        ("Shutter", "1/125", "1/250"), ("Aperture", "f/8", "f/11"),
        ("ISO", "400", "800"), ("Exp. comp.", "0.0 EV", "+0.3 EV"),
        ("White balance", "Daylight", "Auto"), ("Mode", "Manual", "Aperture"),
        ("Focus mode", "AF-S", "MF"), ("Drive", "Single", "Continuous"),
    )
]


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
    # No worker: this is about layout, and there is no reason to go near a
    # camera to measure it.
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    made = mw.MainWindow()
    # Deliberately short, so the panel is well past what the window can show.
    made.resize(1000, 560)
    made.integrate.setChecked(True)
    made.show()
    made._on_settings(_CONNECTED)
    for _ in range(5):
        QApplication.processEvents()
    yield made
    # Not close(): closing shuts the worker thread down, and there is no
    # worker here to shut down.
    made.hide()
    made.deleteLater()
    QApplication.processEvents()


def _scroller(window) -> QScrollArea:
    return window.findChild(QScrollArea)


def test_the_panel_is_taller_than_the_room_it_has(window):
    """Otherwise the rest of these tests are not testing anything."""
    scroller = _scroller(window)
    assert scroller.widget().height() > scroller.viewport().height()


def test_the_shortfall_becomes_scrolling(window):
    assert _scroller(window).verticalScrollBar().maximum() > 0


def test_no_control_is_squeezed_below_the_size_it_asked_for(window):
    for kind in (QSpinBox, QCheckBox):
        for control in window.findChildren(kind):
            assert control.height() >= control.sizeHint().height(), (
                f"{kind.__name__} {control.objectName() or control.__class__.__name__} "
                "was squeezed to make the panel fit"
            )


def test_every_wrapped_hint_shows_all_of_its_text(window):
    for label in window.findChildren(mw.WrappedLabel):
        if not label.text():
            continue
        assert label.height() >= label.heightForWidth(label.width()), (
            f"the last line of {label.text()[:40]!r} would be cut off"
        )


def test_the_controls_keep_their_width_when_the_scrollbar_appears(window):
    scroller = _scroller(window)
    assert scroller.verticalScrollBar().isVisible()
    assert scroller.widget().width() == mw._SIDEBAR_WIDTH


def test_scrolling_the_panel_does_not_take_the_keyboard_from_the_image(window):
    assert _scroller(window).focusPolicy() == Qt.FocusPolicy.NoFocus


def test_a_wrapped_hint_takes_back_the_height_its_text_needs(app):
    """The mechanism itself, at the point where it has to hold.

    Squeezed to four pixels -- which is what a layout short of room does to it
    -- the label has to answer with the height its text actually occupies at
    the width it has been given, or its last line is lost.
    """
    label = mw.WrappedLabel(
        "Manual focus: < is nearer, > is further. Set how many drive steps "
        "each increment is worth. Keys [ ] , . < > drive the first three."
    )
    label.resize(140, 4)
    # Shown, because Qt holds resize events back until a widget is visible.
    label.show()
    QApplication.processEvents()
    wanted = label.heightForWidth(140)
    assert wanted > 4, "the text has to need more than one line for this to mean anything"
    assert label.height() == wanted
    label.hide()


def test_a_wrapped_hint_follows_a_change_of_text(app):
    """The integration hint is rewritten every time the count changes."""
    label = mw.WrappedLabel("short")
    label.resize(140, 4)
    QApplication.processEvents()
    label.setText(
        "About 0.5 fps, with roughly 8.0x less noise, which is a good deal "
        "longer than the line it replaced and must not be cut off."
    )
    QApplication.processEvents()
    assert label.height() == label.heightForWidth(140)
