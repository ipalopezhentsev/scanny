"""Tests for pointer input on the live-view widget, driven by real Qt events.

The gestures share one surface, so the risk is one firing another's action.
Qt delivers press, release, then double-click, which means the first click of
a double click has already been acted on: a single click magnifies, and the
double click that follows focuses on the point the first click just set.

Note that QTest injects events straight into the widget and skips the platform
layer, so these tests say nothing about whether a real click would be
delivered at all -- see the README.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def double_click(widget, pos: QPoint) -> None:
    """Deliver the event sequence a real double click produces.

    ``QTest.mouseDClick`` sends only the double-click event, omitting the
    ordinary click that precedes it. Windows actually delivers
    press, release, double-click, release -- verified against the real input
    queue -- and that leading click is the whole reason the focus gesture can
    rely on the point already having been set.
    """
    left = Qt.MouseButton.LeftButton
    none = Qt.MouseButton.NoButton
    mods = Qt.KeyboardModifier.NoModifier
    for kind, buttons in (
        (QEvent.Type.MouseButtonPress, left),
        (QEvent.Type.MouseButtonRelease, none),
        (QEvent.Type.MouseButtonDblClick, left),
        (QEvent.Type.MouseButtonRelease, none),
    ):
        QApplication.sendEvent(
            widget,
            QMouseEvent(
                kind,
                QPointF(pos),
                QPointF(widget.mapToGlobal(pos)),
                left,
                buttons,
                mods,
            ),
        )

from scanny.camera.nikon import LiveViewFrame  # noqa: E402
from scanny.ui.liveview import LiveViewWidget  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _jpeg() -> bytes:
    """A tiny but real JPEG, so the widget has a pixmap to lay out against."""
    from PySide6.QtGui import QImage

    image = QImage(64, 48, QImage.Format.Format_RGB32)
    image.fill(0x202020)
    from PySide6.QtCore import QBuffer, QByteArray

    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    image.save(buffer, "JPG")
    return bytes(data.data())


@pytest.fixture
def widget(app):
    made = LiveViewWidget()
    made.resize(640, 480)
    made.set_frame(
        LiveViewFrame(
            jpeg=_jpeg(),
            width=640, height=424,
            image_width=6016, image_height=4016,
            crop_width=6016, crop_height=4016,
            crop_center_x=3008, crop_center_y=2008,
            af_width=324, af_height=270,
            af_x=3008, af_y=2008,
        )
    )
    made.show()
    QApplication.processEvents()
    return made


@pytest.fixture
def events(widget):
    seen = {
        "clicked": [], "focus": 0, "region": [], "reset": 0, "toggle": 0,
        "zoom": [], "measure": [],
    }
    widget.pointSelected.connect(lambda x, y: seen["clicked"].append((x, y)))
    widget.focusRequested.connect(lambda: seen.update(focus=seen["focus"] + 1))
    widget.regionSelected.connect(lambda *r: seen["region"].append(r))
    widget.zoomReset.connect(lambda: seen.update(reset=seen["reset"] + 1))
    widget.zoomToggled.connect(lambda: seen.update(toggle=seen["toggle"] + 1))
    widget.zoomStepped.connect(lambda d: seen["zoom"].append(d))
    widget.measureAreaSelected.connect(lambda *r: seen["measure"].append(r))
    return seen


def test_single_click_selects_a_point_and_does_not_focus(widget, events):
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=QPoint(320, 240))
    assert len(events["clicked"]) == 1
    assert events["focus"] == 0
    assert events["reset"] == 0


def test_double_click_requests_focus(widget, events):
    double_click(widget, QPoint(320, 240))
    assert events["focus"] == 1
    # The opening click still moves the rectangle, which is what puts it where
    # the user is pointing before focus runs. It must not magnify.
    assert len(events["clicked"]) == 1
    assert events["toggle"] == 0
    assert events["reset"] == 0


def test_focus_request_carries_no_coordinates(widget):
    """Focus runs where the first click already put the point.

    Re-mapping the second click would be wrong: magnifying changes the crop,
    so the same screen position means a different part of the sensor by then.
    """
    received = []
    widget.focusRequested.connect(lambda *args: received.append(args))
    double_click(widget, QPoint(320, 240))
    assert received == [()]


def test_right_click_toggles_zoom_without_selecting_or_focusing(widget, events):
    QTest.mouseClick(widget, Qt.MouseButton.RightButton, pos=QPoint(320, 240))
    assert events["toggle"] == 1
    assert events["clicked"] == []
    assert events["focus"] == 0
    assert events["reset"] == 0


def test_repeated_right_clicks_each_toggle(widget, events):
    for _ in range(3):
        QTest.mouseClick(widget, Qt.MouseButton.RightButton, pos=QPoint(300, 200))
    assert events["toggle"] == 3
    assert events["clicked"] == []


def test_drag_selects_a_region_instead_of_a_point(widget, events):
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=QPoint(200, 150))
    QTest.mouseMove(widget, QPoint(400, 330))
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=QPoint(400, 330))
    assert len(events["region"]) == 1
    assert events["clicked"] == []
    assert events["focus"] == 0
    x, y, w, h = events["region"][0]
    assert 0.0 <= x < 1.0 and 0.0 <= y < 1.0 and w > 0 and h > 0


def test_shift_drag_marks_out_an_area_to_measure_instead_of_magnifying(
    widget, events
):
    """At full magnification there is nowhere further to zoom, so the same
    gesture with shift held picks out a part of what is already on screen."""
    shift = Qt.KeyboardModifier.ShiftModifier
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, shift, QPoint(200, 150))
    QTest.mouseMove(widget, QPoint(400, 330))
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, shift, QPoint(400, 330))
    assert len(events["measure"]) == 1
    assert events["region"] == [], "it must not magnify as well"
    assert events["clicked"] == []
    x, y, w, h = events["measure"][0]
    assert 0.0 <= x < 1.0 and 0.0 <= y < 1.0 and w > 0 and h > 0


def test_letting_go_of_shift_midway_does_not_turn_a_measurement_into_a_zoom(
    widget, events
):
    """What the gesture is gets decided when the button goes down."""
    shift = Qt.KeyboardModifier.ShiftModifier
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, shift, QPoint(200, 150))
    QTest.mouseMove(widget, QPoint(400, 330))
    QTest.mouseRelease(
        widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        QPoint(400, 330),
    )
    assert len(events["measure"]) == 1
    assert events["region"] == []


def test_a_shift_click_too_small_to_be_an_area_moves_nothing(widget, events):
    """It is a slip of the hand, not a request to move the focus point."""
    shift = Qt.KeyboardModifier.ShiftModifier
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, shift, QPoint(300, 240))
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, shift, QPoint(302, 242))
    assert events["measure"] == []
    assert events["clicked"] == []


def test_the_measured_area_is_drawn_until_it_is_taken_away(widget):
    widget.set_measure_area((0.25, 0.25, 0.5, 0.5))
    assert widget.measure_area == (0.25, 0.25, 0.5, 0.5)
    widget.set_measure_area(None)
    assert widget.measure_area is None


def test_tiny_drag_still_counts_as_a_click(widget, events):
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=QPoint(300, 240))
    QTest.mouseMove(widget, QPoint(304, 243))
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=QPoint(304, 243))
    assert len(events["clicked"]) == 1
    assert events["region"] == []


def test_clicks_outside_the_image_are_ignored(widget, events):
    # Make the widget taller than the frame's aspect so it is genuinely
    # letterboxed, then click in the bar above the image.
    widget.resize(640, 600)
    QApplication.processEvents()
    widget.repaint()
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=QPoint(320, 5))
    assert events["clicked"] == []
    # ...while the middle of the image still focuses.
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=QPoint(320, 300))
    assert len(events["clicked"]) == 1


def test_input_is_inert_before_any_frame_arrives(app, events):
    empty = LiveViewWidget()
    empty.resize(640, 480)
    seen = []
    empty.pointSelected.connect(lambda x, y: seen.append((x, y)))
    resets = []
    empty.zoomReset.connect(lambda: resets.append(1))
    empty.zoomToggled.connect(lambda: resets.append(1))
    QTest.mouseClick(empty, Qt.MouseButton.LeftButton, pos=QPoint(320, 240))
    QTest.mouseClick(empty, Qt.MouseButton.RightButton, pos=QPoint(320, 240))
    assert seen == [] and resets == []


def test_right_button_events_are_not_stolen_by_a_context_menu(widget):
    """The reset must not depend on the platform's context-menu behaviour.

    With the default policy Qt turns a right click into a context-menu event
    and does not guarantee the widget sees the mouse press at all, so the
    reset silently never fired on Windows.
    """
    assert widget.contextMenuPolicy() == Qt.ContextMenuPolicy.PreventContextMenu


def test_right_press_alone_toggles(widget, events):
    """Fires on press, so a swallowed release cannot lose the gesture."""
    QTest.mousePress(widget, Qt.MouseButton.RightButton, pos=QPoint(320, 240))
    assert events["toggle"] == 1
    QTest.mouseRelease(widget, Qt.MouseButton.RightButton, pos=QPoint(320, 240))
    assert events["toggle"] == 1
    assert events["clicked"] == []


def test_widget_takes_keyboard_focus_for_arrow_keys(widget):
    assert widget.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_arrow_keys_pan(widget):
    steps = []
    widget.panStepped.connect(lambda dx, dy: steps.append((dx, dy)))
    for key, expected in (
        (Qt.Key.Key_Left, (-1, 0)),
        (Qt.Key.Key_Right, (1, 0)),
        (Qt.Key.Key_Up, (0, -1)),
        (Qt.Key.Key_Down, (0, 1)),
    ):
        QTest.keyClick(widget, key)
        assert steps[-1] == expected
    assert len(steps) == 4


@pytest.mark.parametrize("key", [Qt.Key.Key_Escape, Qt.Key.Key_0])
def test_keyboard_also_resets_zoom(widget, events, key):
    """A keyboard route to the reset, immune to any mouse-focus quirk."""
    QTest.keyClick(widget, key)
    assert events["reset"] == 1
    assert events["clicked"] == []


# -- keyboard: focus ---------------------------------------------------------


@pytest.mark.parametrize("key", [Qt.Key.Key_Return, Qt.Key.Key_Enter])
def test_enter_requests_autofocus(widget, events, key):
    QTest.keyClick(widget, key)
    assert events["focus"] == 1
    assert events["clicked"] == []
    assert events["toggle"] == 0


def test_focus_keys_name_an_increment_and_a_direction(widget):
    """The widget must not know how many steps an increment is -- that is the
    user's setting, so it names the increment and lets the window resolve it."""
    steps = []
    widget.focusStepped.connect(lambda name, d: steps.append((name, d)))
    for key, expected in (
        (Qt.Key.Key_BracketLeft, ("minimum", -1)),
        (Qt.Key.Key_BracketRight, ("minimum", 1)),
        (Qt.Key.Key_Comma, ("fine", -1)),
        (Qt.Key.Key_Period, ("fine", 1)),
        (Qt.Key.Key_Less, ("coarse", -1)),
        (Qt.Key.Key_Greater, ("coarse", 1)),
    ):
        QTest.keyClick(widget, key)
        assert steps[-1] == expected
    assert len(steps) == 6


def test_focus_keys_do_not_pan_or_zoom(widget, events):
    pans = []
    widget.panStepped.connect(lambda dx, dy: pans.append((dx, dy)))
    for key in (Qt.Key.Key_Comma, Qt.Key.Key_Period):
        QTest.keyClick(widget, key)
    assert pans == []
    assert events["toggle"] == 0 and events["reset"] == 0
