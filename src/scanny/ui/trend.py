"""A small plot of the last so-many readings, for focusing against.

A bar showing the reading as a fraction of the best it has seen is no use
during the half of the job that matters: while focus is improving, every
reading *is* the best one, so the bar sits at the top and says nothing. What
the hand needs to see is the direction the number is moving in, and how far it
moved for the step just taken.

So this draws the readings themselves, scaled to whatever range they have
lately covered rather than to zero. That is what makes a small change visible:
when the reading wanders between 410 and 430, the line uses the whole height
for that twenty rather than drawing it as a flat line near the top.
"""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

__all__ = ["TrendGraph"]

#: How many readings are kept. Integrating sixteen frames they arrive about
#: twice a second, so this is a minute or so of hunting; without integration
#: it is the last few seconds, which is the right window either way.
_SAMPLES = 120


class TrendGraph(QWidget):
    """The recent readings, drawn as a line that fills the height it is given."""

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self._values: "deque[float]" = deque(maxlen=_SAMPLES)
        self.setMinimumHeight(52)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # It is a readout, not a control: it must never take the keyboard away
        # from the image.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    # -- content -----------------------------------------------------------

    def add(self, value: float) -> None:
        self._values.append(float(value))
        self.update()

    def clear(self) -> None:
        self._values.clear()
        self.update()

    @property
    def values(self) -> "list[float]":
        return list(self._values)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))

        values = list(self._values)
        if len(values) < 2:
            painter.setPen(palette.color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for readings..."
            )
            return

        area = self.rect().adjusted(1, 4, -1, -4)
        bottom, top = min(values), max(values)
        span = top - bottom
        if span <= 0:
            # Nothing is changing. Draw it down the middle rather than dividing
            # by zero or magnifying the last digit of a steady number.
            span, bottom = 2.0, top - 1.0

        step = area.width() / (len(values) - 1)
        points = [
            QPointF(
                area.left() + index * step,
                area.bottom() - (value - bottom) / span * area.height(),
            )
            for index, value in enumerate(values)
        ]

        accent = palette.color(QPalette.ColorRole.Highlight)
        # Under the line, so the shape reads at a glance rather than needing to
        # be traced.
        under = QPolygonF(
            points
            + [
                QPointF(points[-1].x(), self.rect().bottom()),
                QPointF(points[0].x(), self.rect().bottom()),
            ]
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 40))
        painter.drawPolygon(under)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(accent, 1.5))
        painter.drawPolyline(QPolygonF(points))

        # The newest reading, so the eye knows which end it is watching.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawEllipse(points[-1], 2.5, 2.5)
