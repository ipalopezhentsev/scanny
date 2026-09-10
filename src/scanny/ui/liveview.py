"""The live-view display: aspect-correct painting, focus overlay, drag-to-zoom."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from ..camera.nikon import LiveViewFrame
from .orientation import Orientation

__all__ = ["LiveViewWidget"]

_BACKDROP = QColor(24, 24, 27)
_AF_IDLE = QColor(235, 235, 235)
_AF_FOCUSED = QColor(80, 220, 120)
_AF_BUSY = QColor(250, 190, 70)
_SELECTION = QColor(120, 190, 255)
_MEASURE = QColor(255, 120, 200)
#: A placed point that has not been measured yet. The measured ones are drawn
#: in the depth ramp's colours instead, so near and far read the same way they
#: do on the map -- and both ends of that ramp are dark, which is why the
#: number on a point picks its ink from what it is sitting on rather than
#: having one.
_POINT = QColor(235, 235, 235)
_POINT_DARK_INK = QColor(20, 20, 24)
_POINT_LIGHT_INK = QColor(250, 250, 252)

#: How big a placed point is drawn, and how near the pointer has to be for
#: its tooltip -- in fractions of the displayed picture, so a click lands
#: the same way whatever size the window is.
_POINT_RADIUS = 11
_POINT_REACH = 0.05


class LiveViewWidget(QWidget):
    """Shows live-view frames and turns pointer input into camera coordinates.

    Clicks and drags are reported in fractions of the displayed image, which
    :meth:`LiveViewFrame.to_af_coords` maps to sensor coordinates. That keeps
    this widget independent of how the camera happens to be magnified.

    The picture may be turned, mirrored or inverted on the way to the screen
    -- see :mod:`scanny.ui.orientation` -- and this widget is where that stops:
    the overlays it draws arrive in the frame's coordinates and are turned with
    the picture, and pointer input is turned back before it leaves, so every
    signal below still speaks in fractions **of the frame** whichever way round
    it is being shown.
    """

    #: A click at (nx, ny) in fractions of the displayed image: move the focus
    #: rectangle there. Does not focus and does not magnify.
    pointSelected = Signal(float, float)
    #: A double click: focus where the rectangle now is.
    #:
    #: Carries no coordinates on purpose. Qt delivers press, release, then
    #: double-click, so the click that opens the gesture has already moved the
    #: rectangle to exactly where the user is pointing. Focusing there is both
    #: simpler than re-mapping the second click and immune to the view having
    #: changed in between.
    focusRequested = Signal()
    #: A dragged rectangle (nx, ny, nw, nh), in the same fractions.
    regionSelected = Signal(float, float, float, float)
    #: A rectangle dragged with shift held: the part of the picture whose
    #: sharpness is to be measured. Separate from magnifying because at full
    #: magnification there is nowhere further to zoom, and picking out
    #: something smaller than the frame is exactly what is wanted there.
    measureAreaSelected = Signal(float, float, float, float)
    #: Mouse wheel: +1 to magnify, -1 to pull back.
    zoomStepped = Signal(int)
    #: Right click: back to the whole frame.
    #:
    #: Right click: magnify if the view is whole, or go back to whole if it is
    #: already magnified. Fires on *press*, so a swallowed release cannot lose
    #: the gesture.
    zoomToggled = Signal()
    #: Esc or 0: always back to the whole frame, whatever the current state.
    zoomReset = Signal()
    #: Arrow keys: pan by (dx, dy) steps of -1, 0 or +1.
    panStepped = Signal(int, int)
    #: Ctrl-click: put a point to be measured here, or take away the one
    #: already here. Ctrl because every other button and modifier on the
    #: image is spoken for, and because it is the one gesture that has to be
    #: impossible to make by accident: a stray click that moved a point
    #: would silently invalidate a measurement that took a minute to make.
    pointPlaced = Signal(float, float)
    #: Manual focus, as the name of an increment and a direction of -1 for
    #: nearer or +1 for further. The widget deliberately does not know how many
    #: steps an increment is: that is the user's setting, resolved by the
    #: window.
    focusStepped = Signal(str, int)

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        # Needed to receive arrow keys. Click focus comes with StrongFocus, so
        # clicking the image also hands it the keyboard.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Without this the platform turns a right click into a context-menu
        # event and the widget is not guaranteed the mouse events at all, so
        # the zoom reset never fires. PreventContextMenu is the documented way
        # to say "no menu here, give me the button presses instead".
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        # The picture as the camera sent it, and the picture as it is shown.
        # Both, because the orientation can change with no new frame behind it
        # -- live view stopped, or a stack still filling -- and re-turning the
        # last frame is the only way the screen can answer that straight away.
        self._source: "QImage | None" = None
        self._pixmap: "QPixmap | None" = None
        self._orientation = Orientation()
        self._frame: "LiveViewFrame | None" = None
        self._target = QRect()
        self._drag_origin: "QPoint | None" = None
        self._drag_current: "QPoint | None" = None
        self._drag_measures = False
        self._measure_area: "tuple[float, float, float, float] | None" = None
        # The places someone asked about, each with the colour to draw it in
        # and the line to show when the pointer is over it.
        self._points: "list[tuple[float, float, int, QColor, str]]" = []
        self._focus_state = "idle"
        self._placeholder = "Not connected"

    # -- content -----------------------------------------------------------

    def set_frame(self, frame: LiveViewFrame) -> None:
        """Show a frame, decoding its own JPEG."""
        self.show_frame(frame, QImage.fromData(frame.jpeg, "JPG"))

    def show_frame(self, frame: LiveViewFrame, image: QImage) -> None:
        """Show a frame whose picture has already been decoded.

        A **null** *image* updates the overlay and leaves the picture alone,
        which is how frames arriving in the middle of an integration stack are
        published: the focus box and the level readout follow the camera at its
        full rate while the picture waits for its stack to finish.
        """
        if not image.isNull():
            self._source = image
            self._redraw_picture()
        elif self._pixmap is None:
            return
        self._frame = frame
        self.update()

    def clear(self, message: str = "Live view stopped") -> None:
        self._source = None
        self._pixmap = None
        self._frame = None
        self._placeholder = message
        self.update()

    def set_orientation(self, orientation: Orientation) -> None:
        """Which way round, and which way up, to show the picture.

        The last frame is turned again on the spot rather than waiting for the
        next one: at a few frames a second with a stack to fill, waiting reads
        as a button that did nothing.
        """
        if orientation == self._orientation:
            return
        self._orientation = orientation
        self._redraw_picture()
        self.update()

    @property
    def orientation(self) -> Orientation:
        return self._orientation

    def _redraw_picture(self) -> None:
        self._pixmap = (
            None
            if self._source is None
            else QPixmap.fromImage(self._orientation.apply(self._source))
        )

    def set_measure_area(
        self, area: "tuple[float, float, float, float] | None"
    ) -> None:
        """Outline the part of the picture being measured, or None for all of it."""
        if area != self._measure_area:
            self._measure_area = area
            self.update()

    @property
    def measure_area(self) -> "tuple[float, float, float, float] | None":
        return self._measure_area

    def set_points(self, points) -> None:
        """Show the places being measured: (x, y, number, colour, tooltip) each.

        In fractions of the picture on screen, worked out afresh for every
        frame by whoever holds the points, because the points themselves live
        on the sensor and the picture on screen is a crop of it that moves.
        The number comes with them for the same reason: at magnification only
        some of them are on screen, and the one that is has to keep the number
        the readout calls it by rather than being renumbered from one.
        """
        self._points = [
            (float(x), float(y), int(number), QColor(colour), str(text))
            for x, y, number, colour, text in points
        ]
        self.update()

    def set_focus_state(self, state: str) -> None:
        """One of ``idle``, ``busy`` or ``focused``, which tints the focus box."""
        if state != self._focus_state:
            self._focus_state = state
            self.update()

    @property
    def frame(self) -> "LiveViewFrame | None":
        return self._frame

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKDROP)

        if self._pixmap is None:
            painter.setPen(QColor(140, 140, 150))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder
            )
            return

        self._target = self._fitted_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(self._target, self._pixmap)

        if self._frame is not None:
            self._draw_focus_box(painter)
        if self._measure_area is not None:
            self._draw_measure_area(painter)
        if self._points:
            self._draw_points(painter)
        if self._drag_origin is not None and self._drag_current is not None:
            colour = _MEASURE if self._drag_measures else _SELECTION
            painter.setPen(QPen(colour, 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(colour.red(), colour.green(), colour.blue(), 40))
            painter.drawRect(QRect(self._drag_origin, self._drag_current).normalized())

    def _draw_points(self, painter: QPainter) -> None:
        """The places being measured, numbered, in the colour they came back.

        Numbered because the numbers are what the readout and the tooltips
        refer to, and a ring rather than a filled blob because whatever is
        under a point is the thing being measured and covering it up would
        hide exactly what someone needs to see to judge the answer.
        """
        font = QFont(painter.font())
        font.setPointSizeF(max(8.0, font.pointSizeF()))
        font.setBold(True)
        painter.setFont(font)
        for x, y, number, colour, _text in self._points:
            centre = self._at(x, y)
            box = QRect(0, 0, _POINT_RADIUS * 2, _POINT_RADIUS * 2)
            box.moveCenter(centre)
            painter.setPen(QPen(QColor(0, 0, 0, 150), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(box)
            painter.setPen(QPen(colour, 2))
            painter.drawEllipse(box)
            # The number sits on a disc of its own colour so it stays legible
            # over whatever the picture happens to be.
            tag = QRect(0, 0, _POINT_RADIUS + 4, _POINT_RADIUS + 4)
            tag.moveCenter(centre)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colour)
            painter.drawEllipse(tag)
            painter.setPen(_ink_for(colour))
            painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, str(number))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    # -- tooltips ----------------------------------------------------------

    def event(self, happening) -> bool:  # noqa: N802 - Qt naming
        """Answer a tooltip request with whatever the point under it says.

        Per-point rather than one tooltip for the widget, which is why this
        is done by hand: the widget is the picture, and what someone wants
        to read is what *that* point came back with.
        """
        if happening.type() == QEvent.Type.ToolTip:
            said = self._said_at(happening.pos())
            if said:
                QToolTip.showText(happening.globalPos(), said, self)
            else:
                QToolTip.hideText()
                happening.ignore()
            return True
        return super().event(happening)

    def _said_at(self, position: QPoint) -> str:
        """What the nearest point to *position* says, if one is near enough."""
        spot = self._normalise(position)
        if spot is None:
            return ""
        nearest, best = "", _POINT_REACH * _POINT_REACH
        for x, y, _number, _colour, text in self._points:
            gap = (x - spot[0]) ** 2 + (y - spot[1]) ** 2
            if gap <= best:
                nearest, best = text, gap
        return nearest

    def _at(self, x: float, y: float) -> QPoint:
        """A place in the frame as a pixel on the widget."""
        vx, vy = self._orientation.to_view(x, y)
        return QPoint(
            self._target.x() + int(vx * self._target.width()),
            self._target.y() + int(vy * self._target.height()),
        )

    def _box(self, rect: "tuple[float, float, float, float]") -> QRectF:
        """A rectangle in the frame as one on the widget."""
        nx, ny, nw, nh = self._orientation.rect_to_view(rect)
        return QRectF(
            self._target.x() + nx * self._target.width(),
            self._target.y() + ny * self._target.height(),
            nw * self._target.width(),
            nh * self._target.height(),
        )

    def _fitted_rect(self) -> QRect:
        """The largest centred rect of the image's aspect that fits the widget."""
        assert self._pixmap is not None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if not pw or not ph:
            return self.rect()
        scale = min(self.width() / pw, self.height() / ph)
        w, h = int(pw * scale), int(ph * scale)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _draw_focus_box(self, painter: QPainter) -> None:
        assert self._frame is not None
        nx, ny, nw, nh = self._frame.af_box_normalised
        if nw <= 0 or nh <= 0:
            return
        box = self._box((nx, ny, nw, nh))
        colour = {
            "focused": _AF_FOCUSED,
            "busy": _AF_BUSY,
        }.get(self._focus_state, _AF_IDLE)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(colour, 2))
        painter.drawRect(box)
        # Corner ticks, so the box stays readable against a busy subject.
        tick = min(box.width(), box.height()) / 4
        painter.setPen(QPen(colour, 3))
        for cx, sx in ((box.left(), 1), (box.right(), -1)):
            for cy, sy in ((box.top(), 1), (box.bottom(), -1)):
                painter.drawLine(cx, cy, cx + sx * tick, cy)
                painter.drawLine(cx, cy, cx, cy + sy * tick)

    def _draw_measure_area(self, painter: QPainter) -> None:
        """The measured region, in its own colour and dashed.

        Deliberately unlike the focus box: one is where the camera will focus,
        the other is where the sharpness is being read, and they are usually
        not the same rectangle.
        """
        assert self._measure_area is not None
        box = self._box(self._measure_area)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(_MEASURE, 2, Qt.PenStyle.DashLine))
        painter.drawRect(box)

    # -- input -------------------------------------------------------------

    def _normalise(self, point: QPoint) -> "tuple[float, float] | None":
        """A pixel on the widget as a place in the frame, or None if off it.

        In the *frame's* coordinates and not the screen's, so that everything
        this widget emits means the same thing however the picture is being
        shown -- and so that the tooltip reach and the point-removal reach are
        still measured in the space the points are kept in.
        """
        if self._target.isEmpty() or not self._target.contains(point):
            return None
        return self._orientation.from_view(
            (point.x() - self._target.x()) / self._target.width(),
            (point.y() - self._target.y()) / self._target.height(),
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Handled on the press and not the release, and it opens no
                # drag: a ctrl-click is never the start of a rectangle.
                spot = self._normalise(event.position().toPoint())
                if spot is not None:
                    self.pointPlaced.emit(*spot)
                return
            self._drag_origin = event.position().toPoint()
            self._drag_current = self._drag_origin
            # Held at the moment of the press, so letting go of shift midway
            # through cannot turn a measurement into a magnification.
            self._drag_measures = bool(
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            )
        elif event.button() == Qt.MouseButton.RightButton:
            self.zoomToggled.emit()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_origin is not None:
            self._drag_current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:
            return
        if event.button() != Qt.MouseButton.LeftButton or self._drag_origin is None:
            return
        origin, self._drag_origin = self._drag_origin, None
        current, self._drag_current = event.position().toPoint(), None
        measuring, self._drag_measures = self._drag_measures, False
        self.update()

        rect = QRect(origin, current).normalized()
        # A drag has to be deliberate before it counts as a region rather than
        # a click; a few pixels of travel while pressing is still a click.
        if rect.width() < 12 or rect.height() < 12:
            # A shift-click is a slip of the hand, not a request to move the
            # focus point somewhere the user was trying to draw a box.
            spot = None if measuring else self._normalise(current)
            if spot is not None:
                self.pointSelected.emit(*spot)
            return
        top_left = self._normalise(rect.topLeft())
        bottom_right = self._normalise(rect.bottomRight())
        if top_left is None or bottom_right is None:
            return
        # Both corners come back in the frame's coordinates, where a turn or a
        # mirror may have swapped which of them is the top left one, so the
        # rectangle is rebuilt from the two rather than subtracted.
        signal = self.measureAreaSelected if measuring else self.regionSelected
        signal.emit(
            min(top_left[0], bottom_right[0]),
            min(top_left[1], bottom_right[1]),
            abs(bottom_right[0] - top_left[0]),
            abs(bottom_right[1] - top_left[1]),
        )

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._frame is not None:
            self.focusRequested.emit()
            event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Arrow keys pan the magnified view.

        The camera has no separate pan control: it centres the magnified view
        on the focus point, so moving that point is what scrolls the view.
        Held keys repeat, which is what makes this feel like panning rather
        than nudging.
        """
        key = Qt.Key(event.key())
        if key in (Qt.Key.Key_Escape, Qt.Key.Key_0):
            self.zoomReset.emit()
            event.accept()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.focusRequested.emit()
            event.accept()
            return
        # Comma and full stop nudge focus, with shift for a big step; the
        # bracket keys take the smallest increment, for critical focus when
        # magnified.
        focus_step = {
            Qt.Key.Key_BracketLeft: ("minimum", -1),
            Qt.Key.Key_BracketRight: ("minimum", 1),
            Qt.Key.Key_Comma: ("fine", -1),
            Qt.Key.Key_Period: ("fine", 1),
            Qt.Key.Key_Less: ("coarse", -1),
            Qt.Key.Key_Greater: ("coarse", 1),
        }.get(key)
        if focus_step is not None:
            self.focusStepped.emit(*focus_step)
            event.accept()
            return
        step = {
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
        }.get(key)
        if step is None or self._frame is None:
            super().keyPressEvent(event)
            return
        self.panStepped.emit(*step)
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta:
            self.zoomStepped.emit(1 if delta > 0 else -1)


def _ink_for(colour: QColor) -> QColor:
    """Dark or light, whichever the number will be readable on.

    The depth ramp is dark at both ends -- deep violet at the near one, deep
    red at the far -- so a single ink colour makes the first and last points
    unreadable, which are exactly the two the answer is about.
    """
    brightness = (
        0.299 * colour.red() + 0.587 * colour.green() + 0.114 * colour.blue()
    )
    return _POINT_DARK_INK if brightness > 140 else _POINT_LIGHT_INK
