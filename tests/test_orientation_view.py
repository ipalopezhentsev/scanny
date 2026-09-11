"""The view transform where it meets the widgets and the window.

:mod:`test_orientation` checks the arithmetic. What is checked here is that
the boundary holds: the picture is turned on the way to the screen, and
*nothing* the panes hand back is turned with it. A click on a picture that is
being shown upside down still has to ask the camera to focus on the thing
under the pointer, or the transform has quietly broken aiming -- which is the
one failure that would look like the camera misbehaving rather than like the
view option being wrong.

The other thing here is that one arrangement is shared. Three panes draw the
picture and two controls describe it, and they are all driven from a single
value, so the check is that they cannot be caught disagreeing.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPoint, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.camera.nikon import LiveViewFrame  # noqa: E402
from scanny.ui import main_window as mw  # noqa: E402
from scanny.ui.liveview import LiveViewWidget  # noqa: E402
from scanny.ui.navigator import NavigatorWidget  # noqa: E402
from scanny.ui.orientation import Orientation  # noqa: E402

#: A quarter turn with a mirror on it: the arrangement a copy stand actually
#: ends up in, and the one where a transform that is only half applied still
#: looks plausible.
TURNED = Orientation(turns=1, mirrored=True)


@pytest.fixture(scope="module")
def app():
    made = QApplication.instance() or QApplication([])
    QApplication.setOrganizationName("scanny-tests")
    QApplication.setApplicationName("scanny-tests")
    QSettings().clear()
    yield made
    QSettings().clear()


def picture(width: int = 64, height: int = 42, shade: int = 0x606060) -> QImage:
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


# -- the image ------------------------------------------------------------


@pytest.fixture
def view(app):
    made = LiveViewWidget()
    made.resize(640, 480)
    made.show_frame(frame(), picture())
    made.show()
    QApplication.processEvents()
    return made


def test_the_picture_is_shown_turned_and_takes_the_turned_shape(view):
    """A landscape frame quarter-turned is a portrait picture on the screen."""
    was = view._fitted_rect()
    assert was.width() > was.height()
    view.set_orientation(TURNED)
    QApplication.processEvents()
    now = view._fitted_rect()
    assert now.height() > now.width()
    assert now.height() / now.width() == pytest.approx(64 / 42, rel=0.05)


def test_the_last_frame_is_turned_without_waiting_for_another(view):
    """Live view may be stopped, or a stack may be filling for a second.

    Either way the button has to answer at once, so the picture already in
    hand is turned again rather than the next one being waited for.
    """
    view.set_orientation(TURNED)
    QApplication.processEvents()
    assert view._pixmap is not None
    assert (view._pixmap.width(), view._pixmap.height()) == (42, 64)


@pytest.mark.parametrize(
    "orientation",
    [Orientation(turns=t, mirrored=m) for t in range(4) for m in (False, True)],
)
def test_a_click_aims_at_what_is_under_the_pointer(view, orientation):
    """The click has to come back in the frame's coordinates, not the screen's.

    Worked the long way round on purpose: where on the *screen* the answer
    puts the point is compared with where the click actually was. Comparing
    against ``from_view`` instead would only be restating the widget's own
    arithmetic back at it.
    """
    clicked = []
    view.pointSelected.connect(lambda x, y: clicked.append((x, y)))
    view.set_orientation(orientation)
    QApplication.processEvents()

    target = view._fitted_rect()
    where = QPoint(target.x() + 20, target.y() + 30)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=where)

    assert len(clicked) == 1
    back = orientation.to_view(*clicked[0])
    assert back[0] * target.width() == pytest.approx(20, abs=1.0)
    assert back[1] * target.height() == pytest.approx(30, abs=1.0)


@pytest.mark.parametrize(
    "orientation",
    [Orientation(turns=t, mirrored=m) for t in range(4) for m in (False, True)],
)
def test_the_arrow_keys_pan_the_way_they_point_on_the_screen(view, orientation):
    """The camera pans across the sensor, so the step goes out in the frame's
    coordinates -- and must still carry the view the way the key points."""
    steps = []
    view.panStepped.connect(lambda dx, dy: steps.append((dx, dy)))
    view.set_orientation(orientation)
    QApplication.processEvents()

    start = (0.3, 0.6)
    for key, screen in (
        (Qt.Key.Key_Left, (-1, 0)),
        (Qt.Key.Key_Right, (1, 0)),
        (Qt.Key.Key_Up, (0, -1)),
        (Qt.Key.Key_Down, (0, 1)),
    ):
        QTest.keyClick(view, key)
        dx, dy = steps[-1]
        a = orientation.to_view(*start)
        b = orientation.to_view(start[0] + 0.1 * dx, start[1] + 0.1 * dy)
        moved = ((b[0] - a[0]) / 0.1, (b[1] - a[1]) / 0.1)
        assert moved == pytest.approx(screen, abs=1e-9)
    assert len(steps) == 4


def test_the_overlays_are_turned_with_the_picture(view):
    """They arrive in the frame's coordinates and are drawn on a turned picture.

    The focus box is the one that matters: it is the camera's own answer about
    where it is focusing, so a box drawn without the turn applied would sit in
    the wrong corner of a picture that otherwise looks perfectly right.
    """
    view.set_measure_area((0.0, 0.0, 0.25, 0.25))
    view.set_orientation(Orientation(turns=1))
    QApplication.processEvents()
    target = view._fitted_rect()

    # A quarter turn clockwise puts the frame's top left corner at the
    # picture's top right one.
    corner = view._box((0.0, 0.0, 0.25, 0.25))
    assert corner.right() == pytest.approx(target.right(), abs=2.0)
    assert corner.top() == pytest.approx(target.top(), abs=2.0)
    spot = view._at(0.0, 0.0)
    assert spot.x() == pytest.approx(target.right(), abs=2.0)
    assert spot.y() == pytest.approx(target.top(), abs=2.0)


def test_a_dragged_rectangle_comes_back_the_right_way_round(view):
    """Mapped corner-wise, so a turn cannot hand on a negative width.

    A rectangle with a negative width draws as nothing and contains nothing:
    the magnification or the measured area would simply stop working, with no
    error to say why.
    """
    dragged = []
    view.regionSelected.connect(lambda *r: dragged.append(r))
    view.set_orientation(TURNED)
    QApplication.processEvents()

    target = view._fitted_rect()
    start = QPoint(target.x() + 10, target.y() + 10)
    end = QPoint(target.x() + 90, target.y() + 70)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view, end)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=end)

    assert len(dragged) == 1
    x, y, w, h = dragged[0]
    assert w > 0 and h > 0
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    # Turned back to the screen, it is the rectangle that was drawn.
    vx, vy, vw, vh = TURNED.rect_to_view((x, y, w, h))
    assert vx * target.width() == pytest.approx(10, abs=1.5)
    assert vy * target.height() == pytest.approx(10, abs=1.5)
    assert vw * target.width() == pytest.approx(80, abs=1.5)
    assert vh * target.height() == pytest.approx(60, abs=1.5)


def test_a_region_is_taken_away_by_clicking_where_it_is_drawn(view):
    """A ctrl-click comes back in the frame, which is where regions are kept.

    The window decides which region a ctrl-click takes away by whether the
    click is inside one it holds. Both have to be in the same space or a
    turned picture makes regions impossible to remove.
    """
    clicked = []
    view.regionClicked.connect(lambda x, y: clicked.append((x, y)))
    view.set_orientation(TURNED)
    QApplication.processEvents()

    target = view._fitted_rect()
    where = QPoint(target.x() + 15, target.y() + 25)
    QTest.mouseClick(
        view,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        pos=where,
    )
    assert len(clicked) == 1
    # Drawn back where it was clicked, so the pointer can find it again.
    spot = view._at(*clicked[0])
    assert spot.x() == pytest.approx(where.x(), abs=2.0)
    assert spot.y() == pytest.approx(where.y(), abs=2.0)


# -- the focus regions, which are places on the sensor ---------------------


def _shown_at(widget, fx: float, fy: float) -> QPoint:
    """A fraction of the picture as it is being *shown*, as a pixel."""
    target = widget._fitted_rect()
    return QPoint(
        target.x() + round(fx * target.width()),
        target.y() + round(fy * target.height()),
    )


def _ctrl_drag(widget, one, other) -> "tuple[QPoint, QPoint]":
    """Ctrl-drag between two fractions of the picture as it is being shown."""
    widget.repaint()  # so the rect the drag lands in is the one drawn
    start, end = _shown_at(widget, *one), _shown_at(widget, *other)
    ctrl = Qt.KeyboardModifier.ControlModifier
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, ctrl, start)
    QTest.mouseMove(widget, end)
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, ctrl, end)
    return start, end


def _ctrl_click(widget, fx: float, fy: float) -> None:
    widget.repaint()
    QTest.mouseClick(
        widget,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        pos=_shown_at(widget, fx, fy),
    )


def _magnified(window) -> LiveViewFrame:
    """A quarter-sized view centred a quarter across and three quarters down."""
    shown = frame(crop=(1504, 1004), centre=(1504, 3012))
    window.view.show_frame(shown, picture())
    QApplication.processEvents()
    window.view.repaint()
    return shown


def test_a_region_on_a_turned_magnified_picture_lands_where_the_sensor_is(window):
    """Where the two transforms meet, and the one place either could hide the
    other's absence.

    A ctrl-drag is two places on the screen. Getting from there to places on
    the sensor is the arrangement undone *and then* the crop applied, and
    dropping either leaves a region that still looks plausible on the picture
    and is on the wrong piece of the world.
    """
    window._apply_orientation(TURNED)
    _magnified(window)
    _ctrl_drag(window.view, (0.2, 0.4), (0.3, 0.6))

    # Mirrored and quarter-turned, those corners are (0.6, 0.8) and (0.4, 0.7)
    # of the picture -- and the picture is the quarter of the frame around
    # (0.25, 0.75).
    assert len(window._regions) == 1
    assert window._regions[0] == pytest.approx((0.225, 0.8, 0.05, 0.025), abs=0.005)


def test_a_region_is_still_taken_away_by_clicking_inside_what_you_can_see(window):
    window._apply_orientation(TURNED)
    _magnified(window)
    _ctrl_drag(window.view, (0.2, 0.4), (0.4, 0.6))
    assert len(window._regions) == 1
    _ctrl_click(window.view, 0.3, 0.5)
    assert window._regions == []


def test_the_region_is_drawn_back_under_the_pointer_that_drew_it(window):
    window._apply_orientation(TURNED)
    _magnified(window)
    start, end = _ctrl_drag(window.view, (0.2, 0.3), (0.45, 0.6))
    window._draw_regions()
    drawn = window.view._regions
    assert len(drawn) == 1
    box = window.view._box(drawn[0][0])
    assert box.left() == pytest.approx(start.x(), abs=3.0)
    assert box.top() == pytest.approx(start.y(), abs=3.0)
    assert box.right() == pytest.approx(end.x(), abs=3.0)
    assert box.bottom() == pytest.approx(end.y(), abs=3.0)


def test_turning_the_picture_does_not_move_the_regions_or_spoil_a_calibration(window):
    """Rotating the display is not a change to what was calibrated.

    The regions are on the sensor and the transform is on the way to the
    screen, so turning it moves nothing behind the screen -- and the worker is
    not told anything, which is what stops a rotation throwing away a
    calibration that took minutes to make.
    """
    window.view.show_frame(frame(), picture())
    QApplication.processEvents()
    _ctrl_drag(window.view, (0.1, 0.2), (0.3, 0.4))
    _ctrl_drag(window.view, (0.6, 0.6), (0.8, 0.8))
    was = list(window._regions)
    assert len(was) == 2

    told = []
    window.requestFocusRegions.connect(told.append)
    window._turn_view(1)
    window._flip_view(False)
    assert window._regions == was
    assert told == [], "a turn is not a region moving"


def test_the_navigator_marks_a_region_where_the_turned_map_puts_it(window):
    """Magnified onto one region the others are off screen, so this is the
    only place they can be seen -- and it draws the whole frame turned like
    every other pane."""
    window._apply_orientation(TURNED)
    window.navigator.show_frame(frame(), picture(60, 40))
    window._regions = [(0.1, 0.2, 0.1, 0.1)]
    window._draw_regions()
    QApplication.processEvents()

    assert window.navigator._regions[0][0] == (0.1, 0.2, 0.1, 0.1), "kept in the frame"
    target = window.navigator._fitted_rect()
    spot = window.navigator._at(0.1, 0.2)
    # Mirrored then quarter-turned, (0.1, 0.2) is drawn at (0.8, 0.9) of the
    # picture: near the far corner from the one it names.
    assert (spot.x() - target.x()) / target.width() == pytest.approx(0.8, abs=0.02)
    assert (spot.y() - target.y()) / target.height() == pytest.approx(0.9, abs=0.02)


# -- the navigator --------------------------------------------------------


@pytest.fixture
def navigator(app):
    made = NavigatorWidget()
    made.resize(300, 168)
    made.show_frame(frame(), picture(60, 40))
    made.show()
    QApplication.processEvents()
    return made


def test_the_map_is_shown_the_same_way_round_as_the_picture(navigator):
    navigator.set_orientation(TURNED)
    QApplication.processEvents()
    fitted = navigator._fitted_rect()
    assert fitted.height() > fitted.width()


def test_the_crop_rectangle_is_drawn_where_the_turned_map_puts_it(navigator):
    """The rectangle is kept in the frame and turned only for drawing."""
    # A quarter of the frame across, centred a quarter in from its top left:
    # (0.125, 0.125, 0.25, 0.25) of the frame.
    navigator.show_frame(frame(crop=(1504, 1004), centre=(1504, 1004)), QImage())
    navigator.set_orientation(Orientation(turns=1))
    QApplication.processEvents()
    fitted = navigator._fitted_rect()
    box = navigator._crop_rect()
    # Turned clockwise, what was 0.125 in from the left is 0.125 down from the
    # top, and what was 0.375 in from the left is 0.375 in from the right --
    # so the rectangle is drawn at (0.625, 0.125) of the picture.
    assert box.left() - fitted.left() == pytest.approx(0.625 * fitted.width(), abs=3.0)
    assert box.top() - fitted.top() == pytest.approx(0.125 * fitted.height(), abs=3.0)
    assert box.width() == pytest.approx(0.25 * fitted.width(), abs=3.0)


def test_dragging_the_map_moves_the_view_in_the_frame_s_coordinates(navigator):
    """What the camera is told is a place in the frame, however the map is shown."""
    moved = []
    navigator.viewCentreMoved.connect(lambda x, y: moved.append((x, y)))
    navigator.show_frame(frame(crop=(1504, 1004), centre=(1504, 1004)), QImage())
    navigator.set_orientation(Orientation(turns=1))
    QApplication.processEvents()

    fitted = navigator._fitted_rect()
    # The top left of the *screen*, which a clockwise turn makes the frame's
    # bottom left: near x=0, near y=1.
    QTest.mousePress(
        navigator,
        Qt.MouseButton.LeftButton,
        pos=QPoint(fitted.x() + 3, fitted.y() + 3),
    )
    QTest.mouseRelease(
        navigator,
        Qt.MouseButton.LeftButton,
        pos=QPoint(fitted.x() + 3, fitted.y() + 3),
    )
    assert moved
    x, y = moved[-1]
    assert x < 0.25
    assert y > 0.75


# -- the window -----------------------------------------------------------


@pytest.fixture
def window(app, monkeypatch):
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    QSettings().clear()
    made = mw.MainWindow()
    made.worker = type("Worker", (), {"save_directory": "."})()
    made._live = True
    made.resize(1000, 800)
    made.show()
    QApplication.processEvents()
    return made


def panes(window):
    return (window.view, window.navigator, window.depth_view)


def test_every_pane_that_draws_the_picture_is_shown_the_same_arrangement(window):
    window._turn_view(1)
    window._flip_view(False)
    wanted = window._orientation
    assert not wanted.is_identity
    for pane in panes(window):
        assert pane._orientation == wanted.geometry


def test_the_depth_map_is_turned_but_never_inverted(window):
    """It is false colour, and its ramp is the whole of the readout.

    Inverting it would paint near in far's colour with the scale underneath
    still saying otherwise -- a readout that is confidently backwards.
    """
    window.invert_colours.setChecked(True)
    window._turn_view(1)
    assert window.view._orientation.inverted
    assert window.depth_view._orientation.turns == 1
    assert not window.depth_view._orientation.inverted


def test_the_histogram_reads_the_frame_and_not_the_inverted_picture(window):
    """Clipping is a fact about what the camera is recording.

    Inverted, a blown highlight would be counted as a blocked shadow, which is
    the one thing the readout exists to catch.
    """
    window.invert_colours.setChecked(True)
    blown = picture(shade=0xFFFFFFFF)
    window._on_frame(frame(), blown)
    QApplication.processEvents()
    counts = window.histogram.describe()
    assert "blown" in counts
    assert "crushed" not in counts
    # And the picture it was handed is the camera's, unmodified: the transform
    # copies rather than writing back into the frame it was given.
    assert blown.pixel(0, 0) == 0xFFFFFFFF


def test_the_controls_and_the_readout_cannot_disagree(window):
    """Whatever changed the arrangement, everything is set from the result."""
    window.invert_action.trigger()
    assert window._orientation.inverted
    assert window.invert_colours.isChecked()
    assert window.invert_action.isChecked()
    assert "inverted" in window.orientation_label.text().lower()

    window.invert_colours.setChecked(False)
    assert not window._orientation.inverted
    assert not window.invert_action.isChecked()
    assert window.orientation_label.text() == "As the camera sends it"


def test_resetting_puts_the_frame_back_up_and_leaves_the_colours_alone(window):
    """Two different jobs, and the tick box is the one that says so."""
    window.invert_colours.setChecked(True)
    window._turn_view(1)
    window._flip_view(True)
    window._reset_view_geometry()
    assert window._orientation == Orientation(inverted=True)
    assert window.invert_colours.isChecked()


def test_the_arrangement_is_remembered(window, app, monkeypatch):
    """A copy stand is not rebuilt between sessions."""
    window._turn_view(-1)
    window._flip_view(False)
    window.invert_colours.setChecked(True)
    wanted = window._orientation

    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    again = mw.MainWindow()
    assert again._orientation == wanted
    assert again.view._orientation == wanted
    assert again.invert_colours.isChecked()


def test_a_sideways_rig_opens_a_window_the_shape_of_what_it_shows(window):
    """The opening size exists to start the black strips at nothing.

    A quarter turn swaps the frame's two sides, so a window opened at the
    sensor's own shape would put the strips back down the sides.
    """
    upright = window._frame_aspect()
    window._turn_view(1)
    assert window._frame_aspect() == pytest.approx(1 / upright)
    size = window._size_that_fits_the_frame()
    assert size.height() > size.width()
