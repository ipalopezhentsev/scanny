"""Aiming at a turned picture: does the camera get told the right place?

The transform is a display transform, so every gesture has to survive it. This
file checks the two that could go wrong without looking wrong -- focusing, and
marking out the area the sharpness is read from -- and it checks them the only
honest way: by finding where a **marked patch of the picture is actually drawn
on the screen**, clicking exactly there, and asserting the camera is told the
coordinate that patch really occupies on the sensor.

That is what makes these tests worth having. Comparing the answer against
``from_view`` would only restate the widget's own arithmetic back at it, and
would pass just as happily if the whole convention were mirrored. Rendering the
picture and finding the patch takes the screen's word for what is under the
pointer, which is also what the user takes.

Both are checked at magnification as well, because the click mapping goes
through the crop rectangle: a turned view *and* an offset crop is where a
half-applied transform hides.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPoint, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.camera.nikon import LiveViewFrame  # noqa: E402
from scanny.ui import main_window as mw  # noqa: E402
from scanny.ui.orientation import Orientation  # noqa: E402

#: Every arrangement, so no test can pass by symmetry.
ARRANGEMENTS = [
    Orientation(turns=t, mirrored=m) for t in range(4) for m in (False, True)
]

#: The picture the camera is pretending to send, and the patch marked on it.
#: Off centre in both axes and not square, so a transform that is out by a
#: quarter turn or a mirror cannot land on it by accident.
PICTURE = (640, 424)
PATCH = (96, 40, 40, 24)  # x, y, w, h in the camera's own pixels
_MARK = 0xFFFF0000
_GROUND = 0xFF203040


@pytest.fixture(scope="module")
def app():
    made = QApplication.instance() or QApplication([])
    QApplication.setOrganizationName("scanny-tests")
    QApplication.setApplicationName("scanny-tests")
    QSettings().clear()
    yield made
    QSettings().clear()


def marked_picture() -> QImage:
    image = QImage(*PICTURE, QImage.Format.Format_RGB32)
    image.fill(_GROUND)
    x, y, w, h = PATCH
    for py in range(y, y + h):
        for px in range(x, x + w):
            image.setPixel(px, py, _MARK)
    return image


def patch_centre_in_frame() -> "tuple[float, float]":
    """The middle of the patch, in fractions of the picture the camera sent."""
    x, y, w, h = PATCH
    return ((x + w / 2) / PICTURE[0], (y + h / 2) / PICTURE[1])


def frame(*, crop=(6016, 4016), centre=(3008, 2008)) -> LiveViewFrame:
    return LiveViewFrame(
        jpeg=b"",
        width=PICTURE[0],
        height=PICTURE[1],
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


#: Magnified onto the frame's top left quadrant, so the click mapping has a
#: crop offset to get wrong as well as a turn.
MAGNIFIED = dict(crop=(1504, 1004), centre=(1504, 1004))


@pytest.fixture
def window(app, monkeypatch):
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    QSettings().clear()
    made = mw.MainWindow()
    made.worker = type("Worker", (), {"save_directory": "."})()
    made._live = True
    made.resize(1100, 800)
    made.show()
    QApplication.processEvents()
    return made


def where_the_patch_is_drawn(view) -> QPoint:
    """The widget pixel the marked patch appears at, found by rendering it.

    The screen's own answer to "what is under the pointer", which is the whole
    reason these tests are worth more than restating the mapping.
    """
    canvas = QPixmap(view.size())
    canvas.fill(QColor(0, 0, 0))
    view.render(canvas)
    shown = canvas.toImage()
    # The patch is drawn scaled and smoothed, so its colour is approached
    # rather than matched: anything strongly red and not the ground is it.
    xs, ys = [], []
    for py in range(0, shown.height(), 2):
        for px in range(0, shown.width(), 2):
            pixel = QColor(shown.pixel(px, py))
            red, green, blue = pixel.red(), pixel.green(), pixel.blue()
            if red > 150 and red > green + 60 and red > blue + 60:
                xs.append(px)
                ys.append(py)
    assert xs, "the marked patch was not drawn anywhere"
    return QPoint(round(sum(xs) / len(xs)), round(sum(ys) / len(ys)))


# -- focusing --------------------------------------------------------------


@pytest.mark.parametrize("orientation", ARRANGEMENTS)
@pytest.mark.parametrize("view_of", ["whole frame", "magnified"])
def test_clicking_the_patch_aims_the_camera_at_the_patch(window, orientation, view_of):
    """Where the camera is told to put its focus box.

    A click is the only thing that moves the box, so this is the whole of
    aiming -- and the coordinate that leaves for the camera has to be the one
    the clicked-on thing really occupies on the sensor.
    """
    aimed = []
    window.requestMovePoint.connect(lambda x, y: aimed.append((x, y)))
    shown = frame(**(MAGNIFIED if view_of == "magnified" else {}))
    window._apply_orientation(orientation)
    window._on_frame(shown, marked_picture())
    QApplication.processEvents()

    QTest.mouseClick(
        window.view, Qt.MouseButton.LeftButton, pos=where_the_patch_is_drawn(window.view)
    )
    assert len(aimed) == 1

    # Through the same mapping the worker uses, and compared with the patch's
    # real place on the sensor.
    got = shown.to_af_coords(*aimed[0])
    wanted = shown.to_af_coords(*patch_centre_in_frame())
    assert got[0] == pytest.approx(wanted[0], abs=shown.af_width / 2)
    assert got[1] == pytest.approx(wanted[1], abs=shown.af_height / 2)


@pytest.mark.parametrize("orientation", ARRANGEMENTS)
def test_double_clicking_focuses_where_the_click_just_aimed(window, orientation):
    """The double click carries no coordinates, and does not need to.

    Qt delivers press, release, then double-click, so the opening click has
    already moved the box to where the user is pointing -- and that click goes
    through the transform. The double click then simply focuses there, which
    is what makes this path immune to the view having been turned at all.
    """
    aimed, focused = [], []
    window.requestMovePoint.connect(lambda x, y: aimed.append((x, y)))
    window.requestAutofocus.connect(lambda: focused.append(True))
    shown = frame()
    window._apply_orientation(orientation)
    window._on_frame(shown, marked_picture())
    QApplication.processEvents()

    at = where_the_patch_is_drawn(window.view)
    QTest.mouseClick(window.view, Qt.MouseButton.LeftButton, pos=at)
    QTest.mouseDClick(window.view, Qt.MouseButton.LeftButton, pos=at)
    QApplication.processEvents()

    assert focused, "the double click did not ask for focus"
    assert aimed, "the opening click did not aim"
    wanted = shown.to_af_coords(*patch_centre_in_frame())
    got = shown.to_af_coords(*aimed[0])
    assert got[0] == pytest.approx(wanted[0], abs=shown.af_width / 2)
    assert got[1] == pytest.approx(wanted[1], abs=shown.af_height / 2)


@pytest.mark.parametrize("orientation", ARRANGEMENTS)
def test_the_focus_box_is_drawn_back_where_the_camera_says_it_is(window, orientation):
    """The round trip: click, the camera answers, the box lands under the pointer.

    The box comes off the header of the next frame in the frame's own
    coordinates, so this is the other half of the transform -- and a mismatch
    between the two halves is exactly the failure that looks like the camera
    misbehaving rather than like the view being turned.
    """
    window._apply_orientation(orientation)
    window._on_frame(frame(), marked_picture())
    QApplication.processEvents()
    at = where_the_patch_is_drawn(window.view)

    aimed = []
    window.requestMovePoint.connect(lambda x, y: aimed.append((x, y)))
    QTest.mouseClick(window.view, Qt.MouseButton.LeftButton, pos=at)
    af_x, af_y = frame().to_af_coords(*aimed[0])

    # The camera's answer: the same frame with the box moved to where it was
    # asked to go.
    answered = frame()
    answered = LiveViewFrame(
        jpeg=b"",
        width=PICTURE[0],
        height=PICTURE[1],
        image_width=answered.image_width,
        image_height=answered.image_height,
        crop_width=answered.crop_width,
        crop_height=answered.crop_height,
        crop_center_x=answered.crop_center_x,
        crop_center_y=answered.crop_center_y,
        af_width=answered.af_width,
        af_height=answered.af_height,
        af_x=af_x,
        af_y=af_y,
    )
    window._on_frame(answered, marked_picture())
    QApplication.processEvents()

    nx, ny, nw, nh = answered.af_box_normalised
    drawn = window.view._box((nx, ny, nw, nh))
    assert drawn.center().x() == pytest.approx(at.x(), abs=drawn.width())
    assert drawn.center().y() == pytest.approx(at.y(), abs=drawn.height())


# -- the area the sharpness is read from -----------------------------------


@pytest.mark.parametrize("orientation", ARRANGEMENTS)
@pytest.mark.parametrize("view_of", ["whole frame", "magnified"])
def test_shift_dragging_the_patch_measures_the_patch(window, orientation, view_of):
    """What fine tuning drives against.

    The meter reads the picture the camera sent, never the turned copy, so the
    rectangle it is given has to be in the camera's own axes. Drawn round the
    patch on screen, it has to come out as the rectangle that holds the patch
    -- and with a positive width, since a right angle swaps which corner is
    the top left one.

    In fractions of the **whole frame**, which is why the expected answer goes
    through the crop: the measured area is a place on the sensor, so at
    magnification the fractions it comes out as are not the fractions of the
    picture the patch occupies.
    """
    areas = []
    window.requestSharpnessArea.connect(lambda area: areas.append(area))
    shown = frame(**(MAGNIFIED if view_of == "magnified" else {}))
    window._apply_orientation(orientation)
    window._on_frame(shown, marked_picture())
    QApplication.processEvents()

    at = where_the_patch_is_drawn(window.view)
    start = QPoint(at.x() - 40, at.y() - 30)
    end = QPoint(at.x() + 40, at.y() + 30)
    QTest.mousePress(
        window.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, start
    )
    QTest.mouseMove(window.view, end)
    QTest.mouseRelease(
        window.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, end
    )
    QApplication.processEvents()

    assert areas, "shift-dragging did not hand on an area"
    x, y, w, h = areas[-1]
    assert w > 0 and h > 0
    # The patch's middle is inside the rectangle the meter was given, in the
    # coordinates the meter keeps its area in: the whole sensor frame.
    px, py = shown.to_frame_fraction(*patch_centre_in_frame())
    assert x <= px <= x + w, f"{px} not within {x}..{x + w}"
    assert y <= py <= y + h, f"{py} not within {y}..{y + h}"


def test_the_meter_reads_the_camera_s_picture_and_not_the_turned_one(window):
    """Nothing on the measuring path ever sees the transform.

    Which is why fine tuning needs no coordinate work of its own: it drives
    focus and reads a rectangle of the frame the camera sent, both of which
    the view transform leaves entirely alone.
    """
    from scanny.ui.sharpness import measure

    window._apply_orientation(Orientation(turns=1, mirrored=True, inverted=True))
    sent = marked_picture()
    window._on_frame(frame(), sent)
    QApplication.processEvents()

    # The window handed the pane its own copy to turn; the picture itself is
    # untouched, so the same reading comes off it as before there was a view
    # transform at all.
    assert sent.size() == QImage(*PICTURE, QImage.Format.Format_RGB32).size()
    assert sent.pixel(PATCH[0] + 1, PATCH[1] + 1) == _MARK
    assert measure(sent) == pytest.approx(measure(marked_picture()))
