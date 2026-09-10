"""How the picture is turned, mirrored and toned on its way to the screen.

A copy stand is built the way the room allows, not the way the sensor is
wired: the camera ends up on its side because that is how the frame fits the
film, and the film ends up emulsion-towards the lens because that is the way
round it lies flat. Negative film adds a third mismatch that is not geometry
at all -- what is on the screen is the complement of what was in front of the
lens, and judging a face by its negative is guesswork.

None of that is the camera's business, and none of it is the measurements'.
So this is a **display transform**: the picture is turned and toned on the way
to the screen, and everything behind the screen -- focus coordinates, the
sharpness reading, the depth sweep, the picture the camera actually saves --
stays in the frame's own coordinates and the frame's own tones. The price of
that boundary is that the widget drawing a turned picture has to turn the
overlays with it and turn pointer input back again, which is what
:meth:`Orientation.to_view` and :meth:`Orientation.from_view` are for.

Both the geometry and the way it composes are worth spelling out.

**Canonical form.** Turning by right angles and mirroring in either axis
generate eight arrangements in all, and three independent flags cannot name
them: mirror left-to-right and then top-to-bottom and you have not got a
doubly mirrored picture, you have got a picture turned through 180 degrees.
So the state kept here is the eight arrangements themselves, written as *a
mirror across the vertical middle, then some number of quarter turns* --
:attr:`mirrored` and :attr:`turns`. Every arrangement has exactly one such
spelling, so two routes to the same picture cannot leave the state disagreeing
with itself.

**Composition.** A button has to do the obvious thing to what is on screen at
the moment it is pressed: *rotate right* turns whatever is showing a quarter
turn clockwise, and *mirror left-right* mirrors whatever is showing about the
screen's vertical middle -- and that has to hold when the picture is already
turned. Mirroring the screen when the picture is turned is not the same as
mirroring the frame, so :meth:`flipped` does not simply flip a flag: it
rewrites the pair, using ``H . R^k = R^-k . H``, and that identity is where
its arithmetic comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QTransform

__all__ = ["Orientation"]

#: The quarter turns, named for the readout. Clockwise, because that is the
#: direction :meth:`turned` counts in and the direction Qt rotates in once its
#: y axis is pointing down.
_TURNS = ("", "turned 90° right", "turned 180°", "turned 90° left")


@dataclass(frozen=True)
class Orientation:
    """One of the eight arrangements of the picture, and whether it is inverted.

    Frozen, and every change returns a new one: the arrangement is pushed to
    three panes and written to the settings, and a value that cannot be
    modified in place is one that cannot be pushed to two of them and left
    stale in the third.
    """

    #: Quarter turns clockwise, applied *after* the mirror. 0 to 3.
    turns: int = 0
    #: Mirrored across the vertical middle, before those turns.
    mirrored: bool = False
    #: Tones and colours reversed. Nothing to do with geometry, and carried
    #: here because it is the same kind of thing: a change to the picture on
    #: the screen that leaves the frame behind it alone.
    inverted: bool = False

    def __post_init__(self) -> None:
        if self.turns not in (0, 1, 2, 3):
            raise ValueError(f"turns must be 0, 1, 2 or 3, not {self.turns!r}")

    # -- what it is --------------------------------------------------------

    @property
    def is_identity(self) -> bool:
        """Whether the picture goes to the screen untouched."""
        return not (self.turns or self.mirrored or self.inverted)

    @property
    def swaps_axes(self) -> bool:
        """Whether a landscape frame is shown as a portrait one.

        The window opens at the shape of a frame, so it has to be asked before
        there is a frame to measure.
        """
        return self.turns % 2 == 1

    @property
    def geometry(self) -> "Orientation":
        """The same arrangement with the inversion dropped.

        For pictures that are already false colour -- the depth map is drawn
        in a ramp whose two ends are the whole of its meaning -- where turning
        it to match the view is right and inverting it would leave near
        looking like far.
        """
        return replace(self, inverted=False)

    def describe(self) -> str:
        """A line for the readout: what is being done to the picture."""
        parts = []
        if self.mirrored and self.turns in (0, 2):
            # The two the user actually asked for by name; spelling these as
            # "mirrored, turned 180 degrees" is accurate and unrecognisable.
            parts.append(
                "mirrored left to right"
                if self.turns == 0
                else "mirrored top to bottom"
            )
        else:
            if self.mirrored:
                parts.append("mirrored")
            if self.turns:
                parts.append(_TURNS[self.turns])
        if self.inverted:
            parts.append("inverted")
        if not parts:
            return "As the camera sends it"
        line = ", ".join(parts)
        return line[:1].upper() + line[1:]

    # -- changing it -------------------------------------------------------

    def turned(self, steps: int) -> "Orientation":
        """Turned *steps* quarter turns clockwise from what is on screen now."""
        return replace(self, turns=(self.turns + steps) % 4)

    def flipped(self, vertical: bool = False) -> "Orientation":
        """Mirrored about the screen's vertical middle, or its horizontal one.

        The turns are rewritten as well as the mirror flag, because mirroring
        the *screen* is not mirroring the frame once the picture is turned:
        ``H . R^k`` is ``R^-k . H``, and a vertical flip is a horizontal one
        turned through 180 degrees.
        """
        turns = (2 - self.turns if vertical else -self.turns) % 4
        return replace(self, turns=turns, mirrored=not self.mirrored)

    def with_inversion(self, inverted: bool) -> "Orientation":
        return replace(self, inverted=bool(inverted))

    # -- the picture -------------------------------------------------------

    def apply(self, image: QImage) -> QImage:
        """*image* as it is to be shown.

        The one given is never written to: it is also the histogram's, the
        navigator's and -- while a stack is filling -- the next frame's, and a
        transform that reached back into it would invert the same frame twice.
        """
        if image.isNull() or self.is_identity:
            return image
        shown = image
        if self.mirrored:
            shown = shown.flipped(Qt.Orientation.Horizontal)
        if self.turns:
            shown = shown.transformed(QTransform().rotate(90 * self.turns))
        if self.inverted:
            # Both calls above hand back a new image, so by here it is usually
            # ours to write to already; copy only when neither ran.
            if shown is image:
                shown = image.copy()
            # RGB and not RGBA: the alpha byte of a live-view frame is padding,
            # and inverting it turns an opaque picture transparent.
            shown.invertPixels(QImage.InvertMode.InvertRgb)
        return shown

    # -- the coordinates ---------------------------------------------------

    def to_view(self, x: float, y: float) -> "tuple[float, float]":
        """A place in the frame as a place on the screen, both in fractions.

        What the overlays are drawn through: the focus box, the measured area
        and the placed points all arrive in the frame's coordinates and have
        to land on the picture where the picture has been turned to.
        """
        if self.mirrored:
            x = 1.0 - x
        for _ in range(self.turns):
            x, y = 1.0 - y, x
        return x, y

    def from_view(self, x: float, y: float) -> "tuple[float, float]":
        """A place on the screen as a place in the frame: the inverse.

        What pointer input is read through, so that a click on a turned
        picture still asks the camera to focus on the thing under the pointer.
        """
        for _ in range(self.turns):
            x, y = y, 1.0 - x
        if self.mirrored:
            x = 1.0 - x
        return x, y

    def rect_to_view(
        self, rect: "tuple[float, float, float, float]"
    ) -> "tuple[float, float, float, float]":
        """An (x, y, w, h) in the frame as one on the screen."""
        return _mapped_rect(self.to_view, rect)

    def rect_from_view(
        self, rect: "tuple[float, float, float, float]"
    ) -> "tuple[float, float, float, float]":
        """An (x, y, w, h) on the screen as one in the frame."""
        return _mapped_rect(self.from_view, rect)


def _mapped_rect(mapping, rect):
    """*rect* through *mapping*, put back the right way round.

    Right angles and mirrors keep an upright rectangle upright, so mapping two
    opposite corners is enough -- but they also swap which corner is which,
    and a rectangle handed on with a negative width is one that silently
    misses everything it is tested against.
    """
    x, y, w, h = rect
    ax, ay = mapping(x, y)
    bx, by = mapping(x + w, y + h)
    return (min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay))
