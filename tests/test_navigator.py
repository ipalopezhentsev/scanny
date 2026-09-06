"""Tests for the navigator pane.

Two things make it work at all. It keeps the last *whole* frame rather than
whatever came past last, because once the view is magnified the camera stops
sending the rest of the picture -- take the crop as the map and the pane shows
a piece of the territory instead. And it draws the rectangle where it has been
dragged to straight away, because the camera answers a few frames a second
with a stack to fill, and a rectangle that waits for the answer will not
follow the pointer.

The coordinates it emits are fractions of the *whole frame*, not of the
picture on screen: those are two different rectangles the moment anything is
magnified, and confusing them would move the view by a factor of the zoom.
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.camera.nikon import LiveViewFrame  # noqa: E402
from scanny.ui.navigator import NavigatorWidget  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication([])


def picture(shade: int = 0x404040, width: int = 60, height: int = 40) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(shade)
    return image


def frame(*, crop=(6016, 4016), centre=(3008, 2008)) -> LiveViewFrame:
    return LiveViewFrame(
        jpeg=b"",
        width=640,
        height=424,
        image_width=6016,
        image_height=4016,
        crop_width=crop[0],
        crop_height=crop[1],
        crop_center_x=centre[0],
        crop_center_y=centre[1],
        af_width=324,
        af_height=270,
        af_x=centre[0],
        af_y=centre[1],
    )


#: A quarter of the frame across, in the top left quadrant of it.
MAGNIFIED = dict(crop=(1504, 1004), centre=(1504, 1004))


@pytest.fixture
def widget():
    """A pane 300x168, already holding a whole frame to be magnified out of."""
    made = NavigatorWidget()
    made.resize(300, 168)
    made.show_frame(frame(), picture())
    made.show()
    QApplication.processEvents()
    return made


@pytest.fixture
def moves(widget):
    seen = []
    widget.viewCentreMoved.connect(lambda x, y: seen.append((x, y)))
    return seen


def drag(widget, *points, release=True):
    """Press at the first point, move through the rest, and let go."""
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent

    left = Qt.MouseButton.LeftButton
    mods = Qt.KeyboardModifier.NoModifier
    for index, point in enumerate(points):
        kind = QEvent.Type.MouseButtonPress if index == 0 else QEvent.Type.MouseMove
        QApplication.sendEvent(
            widget,
            QMouseEvent(kind, QPointF(point), QPointF(point), left, left, mods),
        )
    if release:
        QApplication.sendEvent(
            widget,
            QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                QPointF(points[-1]),
                QPointF(points[-1]),
                left,
                Qt.MouseButton.NoButton,
                mods,
            ),
        )


def test_the_whole_frame_fills_the_rectangle(widget):
    assert widget.has_whole_frame
    assert widget.crop == (0.0, 0.0, 1.0, 1.0)


def test_a_magnified_frame_marks_the_part_on_screen(widget):
    widget.show_frame(frame(**MAGNIFIED), picture(0x808080))
    x, y, w, h = widget.crop
    assert (w, h) == pytest.approx((0.25, 0.25))
    assert (x, y) == pytest.approx((0.125, 0.125))


def test_the_map_is_the_last_whole_frame_not_the_last_frame(widget):
    """A magnified picture is a piece of the territory, not the map."""
    magnified = picture(0xFFFFFF, width=20, height=12)
    widget.show_frame(frame(**MAGNIFIED), magnified)
    # Still the picture from the whole frame, at its own size.
    assert widget._whole.size().toTuple() == (60, 40)


def test_a_frame_without_a_picture_still_moves_the_rectangle(widget):
    """What arrives while an integration stack is filling."""
    widget.show_frame(frame(**MAGNIFIED), QImage())
    assert widget.crop[2] == pytest.approx(0.25)
    assert widget.has_whole_frame


def test_nothing_is_drawn_before_a_whole_frame_has_been_seen():
    """Live view started while the body was already magnified."""
    made = NavigatorWidget()
    made.resize(300, 168)
    made.show_frame(frame(**MAGNIFIED), picture())
    assert not made.has_whole_frame


def test_dragging_the_rectangle_asks_for_a_new_centre(widget, moves):
    widget.show_frame(frame(**MAGNIFIED), QImage())
    target = widget._target
    # A third of the way across the picture, halfway down it.
    point = QPoint(
        int(target.x() + target.width() / 3), int(target.y() + target.height() / 2)
    )
    drag(widget, point)
    assert moves
    assert moves[-1] == pytest.approx((1 / 3, 0.5), abs=0.02)


def test_a_press_outside_the_rectangle_sends_the_view_there(widget, moves):
    """Pressing away from the rectangle means "show me that", not nothing."""
    widget.show_frame(frame(**MAGNIFIED), QImage())
    target = widget._target
    point = QPoint(
        int(target.x() + target.width() * 0.8), int(target.y() + target.height() * 0.8)
    )
    drag(widget, point)
    assert moves[-1] == pytest.approx((0.8, 0.8), abs=0.02)


def test_picking_the_rectangle_up_keeps_it_under_the_pointer(widget, moves):
    """Grabbed by its corner, it stays grabbed by its corner.

    Otherwise the rectangle jumps so its middle is under the pointer the
    moment it is touched, which moves the view before the drag has begun.
    """
    widget.show_frame(frame(**MAGNIFIED), QImage())
    target = widget._target
    x, y, w, h = widget.crop
    # Just inside the top-left corner of the rectangle.
    grab = QPoint(
        int(target.x() + (x + 0.02) * target.width()),
        int(target.y() + (y + 0.02) * target.height()),
    )
    drag(widget, grab, release=False)
    # The centre asked for is still the one it already had, near enough.
    assert moves[-1] == pytest.approx((x + w / 2, y + h / 2), abs=0.03)


def test_the_rectangle_cannot_be_dragged_off_the_frame(widget, moves):
    widget.show_frame(frame(**MAGNIFIED), QImage())
    target = widget._target
    drag(widget, QPoint(target.x() + 1, target.y() + 1))
    x, y = moves[-1]
    # Half the rectangle's width in from the edge, since the view cannot show
    # what is past it.
    assert (x, y) == pytest.approx((0.125, 0.125), abs=0.01)


def test_a_pointer_leaving_the_pane_pins_the_rectangle_rather_than_dropping_it(
    widget, moves
):
    widget.show_frame(frame(**MAGNIFIED), QImage())
    target = widget._target
    start = QPoint(target.center())
    drag(widget, start, QPoint(-40, target.center().y()))
    assert moves[-1][0] == pytest.approx(0.125, abs=0.01)


def test_moves_during_a_drag_are_not_sent_one_per_pointer_event(widget, moves):
    """Each one is a command down the cable, answered one at a time."""
    widget.show_frame(frame(**MAGNIFIED), QImage())
    target = widget._target
    points = [
        QPoint(target.x() + 60 + step, target.center().y()) for step in range(0, 40, 2)
    ]
    drag(widget, *points, release=False)
    assert len(moves) < len(points)


def test_letting_go_is_always_sent(widget, moves):
    """Wherever it was dropped is where the view is meant to end up, even if
    the last move landed inside the gap between sends."""
    widget.show_frame(frame(**MAGNIFIED), QImage())
    target = widget._target
    drag(widget, QPoint(target.center()), QPoint(target.center().x() + 30, target.center().y()))
    assert len(moves) >= 2
    assert moves[-1][0] > moves[0][0]


def test_the_dragged_rectangle_is_drawn_before_the_camera_answers(widget, moves):
    widget.show_frame(frame(**MAGNIFIED), QImage())
    target = widget._target
    was = widget.crop
    drag(widget, QPoint(int(target.x() + target.width() * 0.7), target.center().y()))
    assert widget.crop[0] > was[0]


def test_the_next_frame_after_the_drag_is_what_is_drawn(widget, moves):
    """The camera's answer is the truth, wherever the rectangle was let go."""
    widget.show_frame(frame(**MAGNIFIED), QImage())
    target = widget._target
    drag(widget, QPoint(int(target.x() + target.width() * 0.7), target.center().y()))
    widget.show_frame(frame(**MAGNIFIED), QImage())
    assert widget.crop == pytest.approx((0.125, 0.125, 0.25, 0.25))


def test_a_drag_still_in_progress_is_not_overwritten_by_a_frame(widget, moves):
    """Frames keep arriving while the pointer is down, and the rectangle has
    to stay where it is being held rather than flicking back and forth."""
    widget.show_frame(frame(**MAGNIFIED), QImage())
    target = widget._target
    drag(widget, QPoint(int(target.x() + target.width() * 0.7), target.center().y()), release=False)
    held = widget.crop
    widget.show_frame(frame(**MAGNIFIED), QImage())
    assert widget.crop == pytest.approx(held)


def test_clearing_forgets_the_frame(widget):
    widget.clear("Live view stopped")
    assert not widget.has_whole_frame
    assert widget.crop == (0.0, 0.0, 1.0, 1.0)


def test_a_pane_with_nothing_in_it_ignores_the_pointer():
    made = NavigatorWidget()
    made.resize(300, 168)
    seen = []
    made.viewCentreMoved.connect(lambda x, y: seen.append((x, y)))
    drag(made, QPoint(150, 84))
    assert seen == []


def test_painting_does_not_fall_over(widget):
    widget.show_frame(frame(**MAGNIFIED), QImage())
    widget.grab()
    widget.clear()
    widget.grab()


def test_the_pane_keeps_up_with_a_run_of_frames(widget):
    """Cheap enough to feed every frame, which is how it stays live."""
    started = time.monotonic()
    for _ in range(100):
        widget.show_frame(frame(**MAGNIFIED), picture())
    assert time.monotonic() - started < 1.0
