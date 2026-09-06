"""The controls panel must never be squeezed to make itself fit.

With a camera connected the exposure form fills with nine rows, and by then
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

from PySide6.QtCore import QPoint, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QScrollArea,
    QSpinBox,
)

from scanny.camera.nikon import Setting, Span  # noqa: E402
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
] + [
    # The one row that is typed into rather than picked from.
    Setting(0, "Colour temp.", 5000, "5000 K", True, (),
            span=Span(2500, 10000, 10, " K")),
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


# -- what does not scroll ---------------------------------------------------


def test_the_readout_panes_are_not_in_the_scrolling_part(window):
    """Both are read while a hand is busy with something else.

    A readout that has to be scrolled back to is one that gets looked at once
    and then forgotten about, so the navigator and the histogram keep their
    place whatever the controls below them are doing.
    """
    scroller = _scroller(window)
    for pane in (window.navigator, window.histogram):
        assert not scroller.isAncestorOf(pane)


def test_the_readout_panes_stay_put_while_the_controls_scroll(window):
    """The panel here is well past what the window can show, so this is the
    state the question actually arises in."""
    scroller = _scroller(window)
    bar = scroller.verticalScrollBar()
    before = [window.navigator.pos(), window.histogram.pos()]
    bar.setValue(bar.maximum())
    QApplication.processEvents()
    assert bar.value() > 0, "nothing scrolled, so nothing was tested"
    assert [window.navigator.pos(), window.histogram.pos()] == before


def test_the_panes_line_up_with_the_controls_under_them(window):
    """One column, not a column and two things beside it."""
    scroller = _scroller(window)
    for pane in (window.navigator, window.histogram):
        box = pane.parentWidget()  # its group box
        assert box.width() == scroller.width()


def test_the_pinned_panes_leave_the_controls_room_to_be_scrolled(window):
    """Pinning is only worth having if what is left is still usable.

    At the shortest window these tests use, the two panes must not take so
    much of the column that the controls have nowhere to appear at all.
    """
    assert _scroller(window).viewport().height() > 120


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


# -- the wheel over the panel ----------------------------------------------


def _wheel(widget, notches: int = -1) -> None:
    """One notch of the wheel over *widget*, as the mouse would deliver it."""
    at = QPointF(4, 4)
    QApplication.sendEvent(
        widget,
        QWheelEvent(
            at,
            QPointF(widget.mapToGlobal(QPoint(4, 4))),
            QPoint(0, 0),
            QPoint(0, 120 * notches),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        ),
    )


def test_the_wheel_over_a_combo_scrolls_rather_than_setting_the_camera(window):
    """The exposure combos are the dangerous ones: a notch that lands on the
    shutter speed while the panel is being scrolled past changes the exposure
    of the next photograph, and says nothing about having done it.

    These are also the combos built after the panel was guarded -- they arrive
    with the camera's answer -- so this covers the later ones too.
    """
    combo = window._setting_widgets["Shutter"]
    bar = _scroller(window).verticalScrollBar()
    was, where = combo.currentIndex(), bar.value()
    _wheel(combo)
    assert combo.currentIndex() == was
    assert bar.value() > where


def test_the_wheel_over_a_spin_box_scrolls_rather_than_changing_it(window):
    spin = window._focus_steps["fine"]
    bar = _scroller(window).verticalScrollBar()
    was, where = spin.value(), bar.value()
    _wheel(spin)
    assert spin.value() == was
    assert bar.value() > where


def test_the_wheel_over_the_zoom_slider_scrolls_rather_than_zooming(window):
    slider = window.zoom_slider
    bar = _scroller(window).verticalScrollBar()
    was, where = slider.value(), bar.value()
    _wheel(slider)
    assert slider.value() == was
    assert bar.value() > where


def test_the_panel_scrolls_both_ways(window):
    combo = window._setting_widgets["ISO"]
    bar = _scroller(window).verticalScrollBar()
    _wheel(combo, notches=-3)
    down = bar.value()
    assert down > 0
    _wheel(combo, notches=1)
    assert bar.value() < down


def test_every_control_that_reads_the_wheel_is_guarded(window):
    """Nothing in the panel may answer the wheel itself.

    Named by type rather than one by one, so a control added later is covered
    by this the day it is added.
    """
    scroller = _scroller(window)
    for kind in (QComboBox, QSpinBox):
        for control in scroller.widget().findChildren(kind):
            bar = scroller.verticalScrollBar()
            bar.setValue(0)
            before = control.property("value") if isinstance(control, QSpinBox) else None
            index = control.currentIndex() if isinstance(control, QComboBox) else None
            _wheel(control)
            assert bar.value() > 0, f"{kind.__name__} swallowed the wheel"
            if isinstance(control, QSpinBox):
                assert control.value() == before
            else:
                assert control.currentIndex() == index
