"""The whole frame, with the magnified view marked on it and draggable.

Magnified, the live view is a window onto a picture the camera never sends in
full: at level 7 the screen shows a hundredth of the frame, with nothing to
say where in the frame that hundredth is. Panning by arrow keys then becomes a
walk in the dark, and the way back to a particular corner is to zoom out,
aim, and zoom in again.

So this keeps **the last whole frame that came past** and draws the crop
rectangle on it. The picture goes stale while the view is magnified -- it has
to, since the camera is no longer sending the rest of the frame -- but the
rectangle does not: it comes off the header of every frame, so it follows the
camera live even though the picture under it is a memory. Dragging it moves
the focus point, which is what the camera pans its magnified view with, so a
drag here scrolls the live view for the same reason the arrow keys do.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..camera.nikon import LiveViewFrame
from .orientation import Orientation

__all__ = ["NavigatorWidget"]

_BACKDROP = QColor(24, 24, 27)
_OUTSIDE = QColor(0, 0, 0, 120)
_RECT = QColor(120, 190, 255)

#: Above this magnification the picture is a crop, not the whole frame. The
#: same threshold the worker uses to decide whether the view is magnified.
_WHOLE = 1.01

#: The shortest gap between two moves sent while dragging. Each one is a
#: command down the cable, and the camera answers them one at a time; sending
#: one per mouse move would queue up work the drag has already made obsolete.
_MIN_INTERVAL = 0.1

#: How big the number beside a focus region is drawn here, in pixels. The
#: region's own rectangle is drawn to scale, which on a thumbnail can be a
#: few pixels across -- the number is what says which one it is.
_TAG = 11

#: How tall the pane is. Fixed rather than derived from the frame's aspect,
#: because a height that follows the width would make the sidebar's own layout
#: depend on it -- the picture is letterboxed inside instead.
_HEIGHT = 168


class NavigatorWidget(QWidget):
    """The whole frame, the part of it on screen, and a handle to move it by.

    Shown the same way round as the image is -- see
    :mod:`scanny.ui.orientation`. It has to be: the two are read together, and
    a map that disagrees with the picture about which way is up turns dragging
    the rectangle into a puzzle. The rectangle and the drag are kept in the
    frame's own coordinates, so only the drawing and the pointer change.
    """

    #: Where the middle of the magnified view should go, in fractions of the
    #: whole frame. Not of the displayed image: the whole point of this widget
    #: is that it is looking at the frame the displayed image was cut from.
    viewCentreMoved = Signal(float, float)

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # It is dragged, not typed into: it must never take the keyboard away
        # from the image.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # The last whole frame as it arrived, and as it is shown; see
        # LiveViewWidget for why both are kept.
        self._source: "QImage | None" = None
        self._whole: "QPixmap | None" = None
        self._orientation = Orientation()
        self._crop = (0.0, 0.0, 1.0, 1.0)
        self._target = QRect()
        self._dragging = False
        self._grab_offset = (0.0, 0.0)
        self._pending: "tuple[float, float] | None" = None
        self._sent_at = 0.0
        self._placeholder = "Not connected"
        # The focus regions, in fractions of the whole frame -- which is the
        # coordinate this widget is already in.
        self._regions: "list[tuple[tuple[float, ...], int, QColor]]" = []

    # -- content -----------------------------------------------------------

    def show_frame(self, frame: LiveViewFrame, image: QImage) -> None:
        """Take a frame, and the picture that goes with it if there is one.

        A **null** *image* moves the rectangle and leaves the picture alone,
        which is what arrives while an integration stack is filling. A picture
        that is itself a crop is not kept: it would replace the map with a
        piece of the territory.
        """
        self._crop = frame.crop_normalised
        if not image.isNull() and frame.magnification <= _WHOLE:
            self._source = image
            self._redraw_picture()
        # A frame arriving after the drag has ended is the camera's answer to
        # it. Whether it landed exactly where the rectangle was let go or not,
        # it is now the truth, and holding the drawn rectangle anywhere else
        # would be showing the user a view they have not got.
        if not self._dragging:
            self._pending = None
        self.update()

    def clear(self, message: str = "Live view stopped") -> None:
        self._source = None
        self._whole = None
        self._crop = (0.0, 0.0, 1.0, 1.0)
        self._pending = None
        self._dragging = False
        self._placeholder = message
        self.update()

    def set_orientation(self, orientation: Orientation) -> None:
        """Show the map the same way round the picture is being shown."""
        if orientation == self._orientation:
            return
        self._orientation = orientation
        self._redraw_picture()
        self.update()

    def _redraw_picture(self) -> None:
        self._whole = (
            None
            if self._source is None
            else QPixmap.fromImage(self._orientation.apply(self._source))
        )

    def set_regions(self, regions) -> None:
        """Mark the focus regions: ((x, y, w, h), number, colour) each.

        All of them, whatever the live view is showing, and that is the whole
        reason they are here. Magnified onto one region the others are off
        screen entirely, and this is the only place left that can say where
        they went -- which matters most exactly then, since a calibration
        magnified onto each region in turn is what cannot show them together.
        """
        self._regions = [
            (tuple(float(v) for v in rect), int(number), QColor(colour))
            for rect, number, colour in regions
        ]
        self.update()

    @property
    def crop(self) -> "tuple[float, float, float, float]":
        """The rectangle as drawn: the camera's, or the one being dragged."""
        x, y, w, h = self._crop
        if self._pending is None:
            return (x, y, w, h)
        cx, cy = self._pending
        return (cx - w / 2, cy - h / 2, w, h)

    @property
    def has_whole_frame(self) -> bool:
        return self._whole is not None

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKDROP)

        if self._whole is None:
            painter.setPen(QColor(140, 140, 150))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._placeholder,
            )
            return

        self._target = self._fitted_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(self._target, self._whole)

        box = self._crop_rect()
        # Everything outside the view is dimmed rather than the view being
        # brightened: it keeps the part that matters at the tone it really is,
        # and says without a caption that the rest is not on screen.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_OUTSIDE)
        for shade in self._surrounding(QRectF(self._target), box):
            painter.drawRect(shade)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(_RECT, 2))
        painter.drawRect(box)

        if self._regions:
            self._draw_regions(painter)

    def _draw_regions(self, painter: QPainter) -> None:
        """The focus regions, numbered, over the whole frame.

        Drawn after the dimming and over it, so that a region outside the
        magnified view is still legible: being outside it is exactly what
        someone is looking here to find out.
        """
        font = QFont(painter.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1))
        font.setBold(True)
        painter.setFont(font)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for rect, number, colour in self._regions:
            x, y, w, h = self._orientation.rect_to_view(rect)
            box = QRectF(
                self._target.x() + x * self._target.width(),
                self._target.y() + y * self._target.height(),
                max(w * self._target.width(), 3.0),
                max(h * self._target.height(), 3.0),
            )
            painter.setPen(QPen(QColor(0, 0, 0, 160), 3))
            painter.drawRect(box)
            painter.setPen(QPen(colour, 1))
            painter.drawRect(box)
            # Beside the rectangle rather than in it, which on a thumbnail may
            # be too small to hold a number.
            label = QRectF(box.right() + 1, box.top() - _TAG / 2, _TAG, _TAG)
            painter.setPen(colour)
            painter.drawText(label, Qt.AlignmentFlag.AlignCenter, str(number))

    def _at(self, x: float, y: float) -> QPoint:
        """A place in the frame as a pixel on the widget.

        Through the arrangement, like every other overlay: they arrive in the
        frame's coordinates and the map under them is drawn turned.
        """
        vx, vy = self._orientation.to_view(x, y)
        return QPoint(
            self._target.x() + int(vx * self._target.width()),
            self._target.y() + int(vy * self._target.height()),
        )

    def _fitted_rect(self) -> QRect:
        """The largest centred rect of the frame's aspect that fits the widget."""
        assert self._whole is not None
        pw, ph = self._whole.width(), self._whole.height()
        if not pw or not ph:
            return self.rect()
        scale = min(self.width() / pw, self.height() / ph)
        w, h = int(pw * scale), int(ph * scale)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _crop_rect(self) -> QRectF:
        # The rectangle is kept in the frame's coordinates and turned only for
        # drawing, so the arithmetic that clamps and drags it never has to know
        # which way round the map is being shown.
        x, y, w, h = self._orientation.rect_to_view(self.crop)
        return QRectF(
            self._target.x() + x * self._target.width(),
            self._target.y() + y * self._target.height(),
            max(w * self._target.width(), 3.0),
            max(h * self._target.height(), 3.0),
        )

    @staticmethod
    def _surrounding(whole: QRectF, box: QRectF) -> "list[QRectF]":
        """The four strips of *whole* that *box* does not cover."""
        return [
            QRectF(whole.left(), whole.top(), whole.width(), box.top() - whole.top()),
            QRectF(
                whole.left(),
                box.bottom(),
                whole.width(),
                whole.bottom() - box.bottom(),
            ),
            QRectF(whole.left(), box.top(), box.left() - whole.left(), box.height()),
            QRectF(box.right(), box.top(), whole.right() - box.right(), box.height()),
        ]

    # -- input -------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._whole is None or event.button() != Qt.MouseButton.LeftButton:
            return
        spot = self._normalise(event.position().toPoint())
        if spot is None:
            return
        _, _, w, h = self._crop
        cx, cy = self.crop[0] + w / 2, self.crop[1] + h / 2
        inside = abs(spot[0] - cx) <= w / 2 and abs(spot[1] - cy) <= h / 2
        # Picking the rectangle up keeps it under the pointer where it was
        # taken hold of; pressing anywhere else means "show me that", so the
        # rectangle jumps to the pointer and is dragged from its middle.
        self._grab_offset = (cx - spot[0], cy - spot[1]) if inside else (0.0, 0.0)
        self._dragging = True
        self._drag_to(spot)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._dragging:
            return
        spot = self._normalise(event.position().toPoint(), clamped=True)
        if spot is not None:
            self._drag_to(spot)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if not self._dragging or event.button() != Qt.MouseButton.LeftButton:
            return
        spot = self._normalise(event.position().toPoint(), clamped=True)
        if spot is not None:
            # Unthrottled: wherever the rectangle was let go is where the view
            # is meant to end up, even if the last move was inside the gap.
            self._drag_to(spot, force=True)
        self._dragging = False
        self.unsetCursor()

    def _drag_to(self, spot: "tuple[float, float]", force: bool = False) -> None:
        """Put the rectangle's middle under the pointer, and say so.

        The rectangle is redrawn where it has been dragged to straight away,
        rather than waiting for the camera to answer: at a few frames a second
        with a stack to fill, waiting reads as a rectangle that will not
        follow the pointer.
        """
        centre = self._clamp_centre(
            spot[0] + self._grab_offset[0], spot[1] + self._grab_offset[1]
        )
        if centre == self._pending and not force:
            return
        self._pending = centre
        self.update()
        now = time.monotonic()
        if force or now - self._sent_at >= _MIN_INTERVAL:
            self._sent_at = now
            self.viewCentreMoved.emit(*centre)

    def _clamp_centre(self, cx: float, cy: float) -> "tuple[float, float]":
        """Keep the view inside the frame; it cannot show what is off the edge."""
        _, _, w, h = self._crop
        half_w, half_h = min(w, 1.0) / 2, min(h, 1.0) / 2
        return (
            min(max(cx, half_w), 1 - half_w),
            min(max(cy, half_h), 1 - half_h),
        )

    def _normalise(
        self, point: QPoint, clamped: bool = False
    ) -> "tuple[float, float] | None":
        """A point on the widget as fractions of the frame, or None if outside.

        Dragging passes *clamped*: the pointer leaving the picture partway
        through a drag should pin the rectangle to the edge, not abandon it.

        Clamped on the screen's axes before being turned back, not after: the
        edge the pointer has gone past is a screen edge, and clamping in the
        frame's coordinates instead would pin the rectangle to whichever edge
        that happens to be once the picture is turned.
        """
        if self._target.isEmpty():
            return None
        if not clamped and not self._target.contains(point):
            return None
        fx = (point.x() - self._target.x()) / self._target.width()
        fy = (point.y() - self._target.y()) / self._target.height()
        return self._orientation.from_view(
            min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)
        )
