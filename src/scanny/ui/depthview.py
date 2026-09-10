"""The depth map as a picture, in the panel, while it is being swept.

Small, and letterboxed into a fixed height for the same reason the navigator
is: the panel's layout must not move when a map turns up, or every control
under it jumps down the moment a sweep starts.

The colour is the whole of the readout, so the scale is drawn under it -- near
at the left, far at the right -- and the two ends are labelled in drive steps,
because that is the only unit the map has. Without the scale a rainbow says
"these are different depths" and nothing at all about which way round.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from .depth import DepthMap, colourise, ramp_colour
from .orientation import Orientation

__all__ = ["DepthView"]

_BACKDROP = QColor(24, 24, 27)
_LABEL = QColor(150, 150, 155)

#: How tall the pane is, the scale underneath included.
_HEIGHT = 168
#: How tall the colour scale is, and the gap above it.
_SCALE = 10
_GAP = 4
#: Room under the scale for the two numbers on it.
_CAPTION = 13


class DepthView(QWidget):
    """Draws whatever map it was last given, and a scale to read it by."""

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Read, never typed into: it must not take the keyboard from the image.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._map: "DepthMap | None" = None
        self._pixmap: "QPixmap | None" = None
        self._orientation = Orientation()
        self._placeholder = "No depth map yet"

    # -- content -----------------------------------------------------------

    def show_map(self, depth_map: "DepthMap | None") -> None:
        """Take a map, or None to go back to the placeholder."""
        self._map = depth_map
        self._redraw()
        self.update()

    def clear(self, message: str = "No depth map yet") -> None:
        self._placeholder = message
        self.show_map(None)

    def set_orientation(self, orientation: Orientation) -> None:
        """Show the map the same way round the picture is being shown.

        Only the geometry of it. The map is false colour and the ramp is the
        whole of the readout, so inverting it would leave near painted in far's
        colour with the scale underneath still saying otherwise.
        """
        orientation = orientation.geometry
        if orientation == self._orientation:
            return
        self._orientation = orientation
        self._redraw()
        self.update()

    def _redraw(self) -> None:
        self._pixmap = (
            None
            if self._map is None
            else QPixmap.fromImage(self._orientation.apply(colourise(self._map)))
        )

    @property
    def depth_map(self) -> "DepthMap | None":
        return self._map

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKDROP)
        if self._pixmap is None or self._map is None:
            painter.setPen(_LABEL)
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder
            )
            return
        picture = QRect(
            0, 0, self.width(), max(1, self.height() - _SCALE - _GAP - _CAPTION)
        )
        # Nearest neighbour: the map has the resolution it has, and smoothing
        # it would draw detail between zones that was never measured.
        painter.drawPixmap(
            _fitted(picture, self._pixmap.width(), self._pixmap.height()),
            self._pixmap,
        )
        self._draw_scale(painter)

    def _draw_scale(self, painter: QPainter) -> None:
        near, far = self._map.range
        bar = QRect(0, self.height() - _SCALE - _CAPTION, self.width(), _SCALE)
        ramp = QLinearGradient(bar.left(), 0, bar.right(), 0)
        # The same ramp the map is drawn with, sampled straight off a one-zone
        # gradient rather than restated here.
        for stop in range(11):
            fraction = stop / 10
            ramp.setColorAt(fraction, QColor(*ramp_colour(fraction)))
        painter.fillRect(bar, ramp)
        painter.setPen(_LABEL)
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
        painter.setFont(font)
        captions = QRect(2, bar.bottom() + 1, self.width() - 4, _CAPTION)
        painter.drawText(
            captions, Qt.AlignmentFlag.AlignLeft, f"near {near:.0f}"
        )
        painter.drawText(
            captions, Qt.AlignmentFlag.AlignRight, f"far {far:.0f} steps"
        )


def _fitted(inside: QRect, width: int, height: int) -> QRect:
    """The largest rectangle of this aspect that fits, centred."""
    if width <= 0 or height <= 0:
        return inside
    scale = min(inside.width() / width, inside.height() / height)
    drawn_w, drawn_h = max(1, int(width * scale)), max(1, int(height * scale))
    return QRect(
        inside.left() + (inside.width() - drawn_w) // 2,
        inside.top() + (inside.height() - drawn_h) // 2,
        drawn_w,
        drawn_h,
    )
