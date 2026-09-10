"""The view transform: the eight arrangements, and what the buttons do to them.

Two things here are easy to get wrong and quiet when they are wrong.

The first is that the picture and the coordinates must agree. The pixels are
turned by Qt and the overlays are turned by arithmetic in
:mod:`scanny.ui.orientation`, and if those two disagree about which way
``rotate(90)`` goes then the focus box sits in the wrong corner of a picture
that looks perfectly correct -- so the test that matters most here marks a
pixel, turns the image, and checks the pixel landed where ``to_view`` said it
would.

The second is composition. Every control is a *relative* move: "mirror
left-right" mirrors whatever is on the screen at the moment it is pressed, and
that is a different thing from mirroring the frame once the picture has already
been turned. So each control is checked against an independent statement of
what it is meant to do to the screen -- from every one of the eight
arrangements it could be pressed in, not just from the identity, which is the
one arrangement in which the mistake does not show.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtGui")

from PySide6.QtGui import QImage  # noqa: E402

from scanny.ui.orientation import Orientation  # noqa: E402

#: Places in the frame to compare mappings at. Deliberately not symmetric:
#: (0.5, 0.5) is fixed by every one of the eight arrangements and so agrees
#: with any wrong answer.
_SPOTS = ((0.0, 0.0), (1.0, 0.0), (0.25, 0.75), (0.9, 0.1), (0.5, 0.5))


def arrangements():
    """All eight of them, as (turns, mirrored)."""
    return [
        Orientation(turns=turns, mirrored=mirrored)
        for turns in range(4)
        for mirrored in (False, True)
    ]


# Independent statements of the three moves, on fractions of the screen with
# y downwards. Written out here rather than taken from the module, since the
# module is what they are checking.
def rot90(spot):
    """A quarter turn clockwise: the top left corner goes to the top right."""
    x, y = spot
    return (1.0 - y, x)


def mirror_h(spot):
    x, y = spot
    return (1.0 - x, y)


def mirror_v(spot):
    x, y = spot
    return (x, 1.0 - y)


def close(a, b):
    return a == pytest.approx(b, abs=1e-9)


def picture(width: int = 8, height: int = 4) -> QImage:
    """A picture with one pixel marked, so the marked one can be found again."""
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(0xFF202020)
    return image


def marked(image: QImage, x: int, y: int) -> QImage:
    image = image.copy()
    image.setPixel(x, y, 0xFFFF0000)
    return image


def mark_of(image: QImage):
    found = [
        (x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixel(x, y) == 0xFFFF0000
    ]
    assert len(found) == 1, f"expected one marked pixel, found {found}"
    return found[0]


# -- the arrangement itself ------------------------------------------------


def test_the_camera_s_own_way_up_changes_nothing():
    plain = Orientation()
    assert plain.is_identity
    assert not plain.swaps_axes
    image = picture()
    # The same object, not merely an equal one: this is the path every frame
    # takes, and copying each one to change nothing about it is waste.
    assert plain.apply(image) is image
    for spot in _SPOTS:
        assert close(plain.to_view(*spot), spot)


def test_turns_outside_the_quarter_turns_are_refused():
    with pytest.raises(ValueError):
        Orientation(turns=4)


@pytest.mark.parametrize("orientation", arrangements())
def test_every_arrangement_round_trips(orientation):
    """Screen back to frame is the inverse of frame to screen.

    Pointer input goes one way and the overlays go the other, so a mismatch
    means a click that lands somewhere other than where it was aimed.
    """
    for spot in _SPOTS:
        there = orientation.to_view(*spot)
        assert close(orientation.from_view(*there), spot)


@pytest.mark.parametrize("orientation", arrangements())
def test_the_pixels_go_where_the_coordinates_say(orientation):
    """Qt turns the picture; arithmetic turns the overlays. They must agree.

    Checked at a corner of a picture that is not square, so a transform that
    is out by a quarter turn cannot pass by symmetry.
    """
    source = marked(picture(8, 4), 0, 0)
    shown = orientation.apply(source)
    if orientation.swaps_axes:
        assert (shown.width(), shown.height()) == (4, 8)
    else:
        assert (shown.width(), shown.height()) == (8, 4)

    # The middle of the marked pixel, so the comparison is not sitting on a
    # boundary between two of them.
    x, y = orientation.to_view(0.5 / 8, 0.5 / 4)
    assert mark_of(shown) == (int(x * shown.width()), int(y * shown.height()))


@pytest.mark.parametrize("orientation", arrangements())
def test_a_quarter_turn_is_what_swaps_the_axes(orientation):
    assert orientation.swaps_axes == (orientation.turns in (1, 3))


# -- what the controls do --------------------------------------------------


@pytest.mark.parametrize("orientation", arrangements())
def test_rotate_right_turns_what_is_on_the_screen(orientation):
    turned = orientation.turned(1)
    for spot in _SPOTS:
        assert close(turned.to_view(*spot), rot90(orientation.to_view(*spot)))


@pytest.mark.parametrize("orientation", arrangements())
def test_rotate_left_is_its_undoing(orientation):
    assert orientation.turned(1).turned(-1) == orientation
    assert orientation.turned(4) == orientation


@pytest.mark.parametrize("orientation", arrangements())
def test_mirroring_mirrors_what_is_on_the_screen(orientation):
    """Both mirrors are about the *screen's* axes, not the frame's.

    This is the one that fails if mirroring is implemented as a flag: from a
    quarter-turned picture, flipping the frame left-right shows up on screen
    as a top-to-bottom flip, which is not what the button says.
    """
    across = orientation.flipped()
    over = orientation.flipped(vertical=True)
    for spot in _SPOTS:
        was = orientation.to_view(*spot)
        assert close(across.to_view(*spot), mirror_h(was))
        assert close(over.to_view(*spot), mirror_v(was))


@pytest.mark.parametrize("orientation", arrangements())
def test_a_mirror_is_its_own_undoing(orientation):
    assert orientation.flipped().flipped() == orientation
    assert orientation.flipped(True).flipped(True) == orientation


@pytest.mark.parametrize("orientation", arrangements())
def test_the_two_mirrors_together_are_a_half_turn(orientation):
    """Which is why they cannot be two independent tick boxes."""
    assert orientation.flipped().flipped(True) == orientation.turned(2)


def test_turning_and_mirroring_reach_exactly_eight_arrangements():
    """No twelfth spelling of the same picture, and none missing.

    The state is a canonical form, so the set the buttons can reach has to
    close at the eight arrangements there are -- if two routes to the same
    picture left different states behind, the readout and the tick box would
    end up describing a picture that is not on the screen.
    """
    seen = {Orientation()}
    while True:
        grown = set(seen)
        for one in seen:
            grown |= {one.turned(1), one.turned(-1), one.flipped(), one.flipped(True)}
        if grown == seen:
            break
        seen = grown
    assert len(seen) == 8
    assert seen == set(arrangements())


# -- the inversion ---------------------------------------------------------


def test_inverting_complements_the_picture_and_leaves_the_frame_alone():
    """The frame is the histogram's too, and the next stack's.

    An in-place inversion would show up as a picture that flickers between
    positive and negative, and as a histogram that disagrees with the screen.
    """
    source = picture(4, 2)
    source.setPixel(0, 0, 0xFF102030)
    shown = Orientation(inverted=True).apply(source)
    assert shown.pixel(0, 0) == 0xFFEFDFCF
    assert source.pixel(0, 0) == 0xFF102030
    # Opaque still: the alpha byte of a live-view frame is padding, and
    # inverting it would turn the picture transparent.
    assert (shown.pixel(0, 0) >> 24) == 0xFF


def test_inverting_is_not_geometry():
    inverted = Orientation(turns=1, mirrored=True, inverted=True)
    assert not inverted.is_identity
    assert inverted.geometry == Orientation(turns=1, mirrored=True)
    assert inverted.with_inversion(False) == inverted.geometry
    for spot in _SPOTS:
        assert close(inverted.to_view(*spot), inverted.geometry.to_view(*spot))


def test_the_inversion_survives_every_turn_and_mirror():
    """It is the one flag the geometry buttons must not disturb."""
    one = Orientation(inverted=True)
    for step in (one.turned(1), one.turned(-1), one.flipped(), one.flipped(True)):
        assert step.inverted


# -- rectangles ------------------------------------------------------------


@pytest.mark.parametrize("orientation", arrangements())
def test_a_rectangle_comes_back_the_right_way_round(orientation):
    """A turn swaps which corner is the top left one.

    Handing on a rectangle with a negative width is the quiet version of this
    bug: it draws as nothing and contains nothing, so the measured area simply
    stops working with no error anywhere.
    """
    rect = (0.1, 0.2, 0.3, 0.4)
    shown = orientation.rect_to_view(rect)
    assert shown[2] > 0 and shown[3] > 0
    if orientation.swaps_axes:
        assert close(shown[2:], (0.4, 0.3))
    else:
        assert close(shown[2:], (0.3, 0.4))
    assert close(orientation.rect_from_view(shown), rect)


@pytest.mark.parametrize("orientation", arrangements())
def test_a_rectangle_holds_the_corners_it_covered(orientation):
    """The turned rectangle covers the turned corners of the original."""
    x, y, w, h = 0.1, 0.2, 0.3, 0.4
    vx, vy, vw, vh = orientation.rect_to_view((x, y, w, h))
    for corner in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
        cx, cy = orientation.to_view(*corner)
        assert vx - 1e-9 <= cx <= vx + vw + 1e-9
        assert vy - 1e-9 <= cy <= vy + vh + 1e-9


# -- the readout -----------------------------------------------------------


def test_the_readout_names_the_moves_that_were_asked_for():
    """The two mirrors are described as mirrors, not as their canonical form.

    Mirror top-to-bottom is stored as "mirrored, turned 180 degrees", which is
    accurate and unrecognisable as the thing the user just pressed.
    """
    assert Orientation().describe() == "As the camera sends it"
    assert Orientation().flipped().describe() == "Mirrored left to right"
    assert Orientation().flipped(True).describe() == "Mirrored top to bottom"
    assert Orientation().turned(1).describe() == "Turned 90° right"
    assert Orientation().turned(-1).describe() == "Turned 90° left"
    assert Orientation(inverted=True).describe() == "Inverted"
    assert Orientation().turned(1).with_inversion(True).describe() == (
        "Turned 90° right, inverted"
    )


@pytest.mark.parametrize("orientation", arrangements())
def test_every_arrangement_has_something_to_say(orientation):
    for inverted in (False, True):
        line = orientation.with_inversion(inverted).describe()
        assert line and line[0].isupper()
