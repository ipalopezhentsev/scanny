"""The live-view display: aspect-correct painting, focus overlay, drag-to-zoom."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..camera.nikon import LiveViewFrame

__all__ = ["LiveViewWidget"]

_BACKDROP = QColor(24, 24, 27)
_AF_IDLE = QColor(235, 235, 235)
_AF_FOCUSED = QColor(80, 220, 120)
_AF_BUSY = QColor(250, 190, 70)
_SELECTION = QColor(120, 190, 255)


class LiveViewWidget(QWidget):
    """Shows live-view frames and turns pointer input into camera coordinates.

    Clicks and drags are reported in fractions of the displayed image, which
    :meth:`LiveViewFrame.to_af_coords` maps to sensor coordinates. That keeps
    this widget independent of how the camera happens to be magnified.
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
        self._pixmap: "QPixmap | None" = None
        self._frame: "LiveViewFrame | None" = None
        self._target = QRect()
        self._drag_origin: "QPoint | None" = None
        self._drag_current: "QPoint | None" = None
        self._focus_state = "idle"
        self._placeholder = "Not connected"

    # -- content -----------------------------------------------------------

    def set_frame(self, frame: LiveViewFrame) -> None:
        image = QImage.fromData(frame.jpeg, "JPG")
        if not image.isNull():
            self._pixmap = QPixmap.fromImage(image)
            self._frame = frame
            self.update()

    def clear(self, message: str = "Live view stopped") -> None:
        self._pixmap = None
        self._frame = None
        self._placeholder = message
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
        if self._drag_origin is not None and self._drag_current is not None:
            painter.setPen(QPen(_SELECTION, 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(120, 190, 255, 40))
            painter.drawRect(QRect(self._drag_origin, self._drag_current).normalized())

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
        box = QRectF(
            self._target.x() + nx * self._target.width(),
            self._target.y() + ny * self._target.height(),
            nw * self._target.width(),
            nh * self._target.height(),
        )
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

    # -- input -------------------------------------------------------------

    def _normalise(self, point: QPoint) -> "tuple[float, float] | None":
        if self._target.isEmpty() or not self._target.contains(point):
            return None
        return (
            (point.x() - self._target.x()) / self._target.width(),
            (point.y() - self._target.y()) / self._target.height(),
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.position().toPoint()
            self._drag_current = self._drag_origin
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
        self.update()

        rect = QRect(origin, current).normalized()
        # A drag has to be deliberate before it counts as a region rather than
        # a click; a few pixels of travel while pressing is still a click.
        if rect.width() < 12 or rect.height() < 12:
            spot = self._normalise(current)
            if spot is not None:
                self.pointSelected.emit(*spot)
            return
        top_left = self._normalise(rect.topLeft())
        bottom_right = self._normalise(rect.bottomRight())
        if top_left is None or bottom_right is None:
            return
        self.regionSelected.emit(
            top_left[0],
            top_left[1],
            bottom_right[0] - top_left[0],
            bottom_right[1] - top_left[1],
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
