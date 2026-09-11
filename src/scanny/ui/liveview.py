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
#: A focus region being drawn. Its own colour, like the other two drags, so
#: that which of the three a drag is going to be is plain before it is let go.
_REGION_DRAG = QColor(255, 200, 80)
#: The number on a region picks its ink from the colour it is drawn in, since
#: the colours run from pale to dark and one ink is unreadable on half of them.
_DARK_INK = QColor(20, 20, 24)
_LIGHT_INK = QColor(250, 250, 252)

#: How big the number tag on a region is, in pixels on screen.
_TAG = 16


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
    #:
    #: In fractions of the **whole frame**, unlike every other rectangle this
    #: emits: the measured area is a place on the subject that has to survive
    #: the view magnifying and panning, not a place on the screen. See
    #: :meth:`scanny.ui.sharpness.SharpnessMeter.set_area`.
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
    #: Arrow keys: pan by (dx, dy) steps of -1, 0 or +1, across the frame.
    panStepped = Signal(int, int)
    #: A rectangle dragged with ctrl held: a focus region, to be kept in focus
    #: along with the others. In fractions of the **whole frame**, like the
    #: measured area, because a region is a place on the subject and has to
    #: stay on it while a calibration magnifies onto every region in turn.
    #:
    #: Ctrl because every other button and modifier on the image is spoken
    #: for, and because it is a gesture that has to be impossible to make by
    #: accident: a stray drag that moved a region would silently throw away a
    #: calibration that took minutes to make.
    regionDrawn = Signal(float, float, float, float)
    #: A ctrl-click without a drag, at (nx, ny) in fractions of the displayed
    #: picture: take away the region under it, if there is one.
    regionClicked = Signal(float, float)
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
        # What the drag in progress will be when it is let go: a magnification,
        # a measured area, or a focus region. Decided by the modifier held at
        # the press, so letting go of it midway changes nothing.
        self._drag_kind = "zoom"
        self._measure_area: "tuple[float, float, float, float] | None" = None
        # The focus regions on this picture: each a rectangle in fractions of
        # it, its number, the colour to draw it in, the line to show when the
        # pointer is over it, and whether it is the one being worked on.
        self._regions: "list[tuple[tuple[float, ...], int, QColor, str, bool]]" = []
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
        """Outline the part being measured, or None for all of it.

        In fractions of the whole frame; where that lands on the picture on
        screen is worked out afresh for every frame, because the picture on
        screen is a crop of the frame that moves.
        """
        if area != self._measure_area:
            self._measure_area = area
            self.update()

    @property
    def measure_area(self) -> "tuple[float, float, float, float] | None":
        return self._measure_area

    def set_regions(self, regions) -> None:
        """Show the focus regions: (rect, number, colour, tooltip, active) each.

        The rectangle is in fractions of the picture on screen, worked out
        afresh for every frame by whoever holds the regions, because the
        regions themselves live on the sensor and the picture on screen is a
        crop of it that moves. The number comes with them for the same reason:
        magnified, only one of them is on screen, and it has to keep the
        number the report calls it by rather than being renumbered from one.
        """
        self._regions = [
            (
                tuple(float(v) for v in rect),
                int(number),
                QColor(colour),
                str(text),
                bool(active),
            )
            for rect, number, colour, text, active in regions
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
            self._draw_measure_area(painter)
        if self._regions:
            self._draw_regions(painter)
        if self._drag_origin is not None and self._drag_current is not None:
            colour = {"measure": _MEASURE, "region": _REGION_DRAG}.get(
                self._drag_kind, _SELECTION
            )
            painter.setPen(QPen(colour, 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(colour.red(), colour.green(), colour.blue(), 40))
            painter.drawRect(QRect(self._drag_origin, self._drag_current).normalized())

    def _draw_regions(self, painter: QPainter) -> None:
        """The focus regions, numbered, in the colour the report left them.

        An outline rather than a filled box because whatever is inside a
        region is the thing being judged, and covering it would hide exactly
        what someone needs to see. The number sits on a tag at the region's
        top left corner on screen -- whichever corner of the frame that is
        once the picture has been turned -- so it covers as little as it can.
        """
        font = QFont(painter.font())
        font.setPointSizeF(max(8.0, font.pointSizeF()))
        font.setBold(True)
        painter.setFont(font)
        for rect, number, colour, _text, active in self._regions:
            box = self._box(rect)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 150), 4 if active else 3))
            painter.drawRect(box)
            style = Qt.PenStyle.DashLine if active else Qt.PenStyle.SolidLine
            painter.setPen(QPen(colour, 3 if active else 2, style))
            painter.drawRect(box)
            tag = QRectF(box.left(), box.top(), _TAG, _TAG)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colour)
            painter.drawRect(tag)
            painter.setPen(_ink_for(colour))
            painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, str(number))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    # -- tooltips ----------------------------------------------------------

    def event(self, happening) -> bool:  # noqa: N802 - Qt naming
        """Answer a tooltip request with whatever the region under it says.

        Per-region rather than one tooltip for the widget, which is why this
        is done by hand: the widget is the picture, and what someone wants
        to read is what *that* region came back with.
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
        """What the region under *position* says, the smallest if they overlap."""
        spot = self._normalise(position)
        if spot is None:
            return ""
        under = [
            (rect[2] * rect[3], text)
            for rect, _number, _colour, text, _active in self._regions
            if rect[0] <= spot[0] <= rect[0] + rect[2]
            and rect[1] <= spot[1] <= rect[1] + rect[3]
        ]
        return min(under)[1] if under else ""

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

        It is kept in the frame's coordinates, so where it falls on the
        picture depends on the crop this frame arrived with -- and nothing is
        drawn at all when the view has been magnified somewhere else, because
        then it is not on the picture.
        """
        assert self._frame is not None
        if self._measure_area is None:
            return
        shown = self._frame.area_normalised(self._measure_area)
        if shown is None:
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(_MEASURE, 2, Qt.PenStyle.DashLine))
        painter.drawRect(self._box(shown))

    # -- input -------------------------------------------------------------

    def _normalise(self, point: QPoint) -> "tuple[float, float] | None":
        """A pixel on the widget as a place in the frame, or None if off it.

        In the *frame's* coordinates and not the screen's, so that everything
        this widget emits means the same thing however the picture is being
        shown -- and so that the regions are hit-tested in the space they are
        drawn from.
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
            self._drag_origin = event.position().toPoint()
            self._drag_current = self._drag_origin
            # Held at the moment of the press, so letting go of a modifier
            # midway through cannot turn one kind of rectangle into another.
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                self._drag_kind = "region"
            elif modifiers & Qt.KeyboardModifier.ShiftModifier:
                self._drag_kind = "measure"
            else:
                self._drag_kind = "zoom"
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
        kind, self._drag_kind = self._drag_kind, "zoom"
        self.update()

        rect = QRect(origin, current).normalized()
        # A drag has to be deliberate before it counts as a rectangle rather
        # than a click; a few pixels of travel while pressing is still a click.
        if rect.width() < 12 or rect.height() < 12:
            spot = self._normalise(current)
            if spot is None:
                return
            if kind == "region":
                self.regionClicked.emit(*spot)
            elif kind == "zoom":
                # A shift-click is a slip of the hand, not a request to move
                # the focus point somewhere the user was trying to draw a box.
                self.pointSelected.emit(*spot)
            return
        top_left = self._normalise(rect.topLeft())
        bottom_right = self._normalise(rect.bottomRight())
        if top_left is None or bottom_right is None:
            return
        # Both corners come back in the displayed picture's coordinates, where
        # a turn or a mirror may have swapped which of them is the top left
        # one, so the rectangle is rebuilt from the two rather than subtracted.
        if kind == "zoom":
            self.regionSelected.emit(*_rect_between(top_left, bottom_right))
            return
        # A measured area and a focus region are both places on the sensor,
        # so the two corners go out through the crop this frame was sent
        # with. Without a frame there is no crop to go through and nothing
        # that could be said.
        if self._frame is None:
            return
        on_the_sensor = _rect_between(
            self._frame.to_frame_fraction(*top_left),
            self._frame.to_frame_fraction(*bottom_right),
        )
        if kind == "region":
            self.regionDrawn.emit(*on_the_sensor)
        else:
            self.measureAreaSelected.emit(*on_the_sensor)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._frame is not None:
            self.focusRequested.emit()
            event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Arrow keys pan the magnified view.

        The camera has no separate pan control: it centres the magnified view
        on the focus point, so moving that point is what scrolls the view.
        Held keys repeat, which is what makes this feel like panning rather
        than nudging. The keys name directions on the screen and the step goes
        out in the frame's, like every other signal here, so *right* still
        pans towards what is to the right of the picture however it is turned.
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
        self.panStepped.emit(*self._orientation.step_from_view(*step))
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta:
            self.zoomStepped.emit(1 if delta > 0 else -1)


def _rect_between(
    one: "tuple[float, float]", other: "tuple[float, float]"
) -> "tuple[float, float, float, float]":
    """An (x, y, w, h) rectangle from two corners, whichever way round they are."""
    return (
        min(one[0], other[0]),
        min(one[1], other[1]),
        abs(other[0] - one[0]),
        abs(other[1] - one[1]),
    )


def _ink_for(colour: QColor) -> QColor:
    """Dark or light, whichever the number will be readable on."""
    brightness = (
        0.299 * colour.red() + 0.587 * colour.green() + 0.114 * colour.blue()
    )
    return _DARK_INK if brightness > 140 else _LIGHT_INK
