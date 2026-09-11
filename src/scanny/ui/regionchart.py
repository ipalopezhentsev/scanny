"""Every region's sharpness through the search for the compromise, one line each.

The sharpness trend shows one number, and during the compromise that number is
the average or the worst of the regions -- which is what the search climbs,
and exactly what hides what is going on underneath it. What makes the search
make sense is the regions one by one: region 1 coming up to its best and going
over it while region 2 is still climbing towards its own, the combined number
peaking somewhere between. So each region gets a line here, as a share of its
own best, on the same scale -- a share has the same meaning for every region,
which a raw reading does not -- with the combined number drawn over them.

The background says what the search was doing at the time, because the shape
only makes sense against it: walking out past every region's best, walking
back across all of them (the leg the depths are read from), then going home to
the best of it.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

__all__ = ["RegionChart", "line_colour"]

#: A colour per region number, told apart at a glance. The regions on the
#: picture are coloured by how they fared, which is a different question, so
#: these are matched to them by the number beside each line instead.
_LINES = ("#4e9af1", "#f28e2b", "#59a14f", "#e15759", "#b07aa1")

#: How the stretches of the search are shaded behind the lines, and named.
_STAGES = {
    "out": ("walking out", QColor(120, 120, 140, 28)),
    "across": ("across all of them", QColor(80, 160, 255, 34)),
    "home": ("home to the best", QColor(90, 200, 120, 34)),
    "climb": ("climbing", QColor(240, 180, 60, 34)),
}


def line_colour(number: int) -> str:
    """The colour region *number* is drawn in on the chart."""
    return _LINES[(int(number) - 1) % len(_LINES)]


class RegionChart(QWidget):
    """Each region's share of its own best at every probe, and the combined one."""

    def __init__(self, parent: "QWidget | None" = None, *, compact: bool = True) -> None:
        super().__init__(parent)
        self._compact = compact
        self._regions: "tuple[int, ...]" = ()
        self._stages: "list[str]" = []
        self._shares: "list[tuple[float, ...]]" = []
        self._combined: "list[float]" = []
        self._combined_name = "average"
        self.setMinimumHeight(96 if compact else 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # A readout, not a control: it must never take the keyboard away from
        # the image.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    # -- content -----------------------------------------------------------

    def clear(self) -> None:
        self._regions = ()
        self._stages, self._shares, self._combined = [], [], []
        self.update()

    def set_combined_name(self, name: str) -> None:
        """What the combined line is: ``average`` or ``worst``."""
        self._combined_name = name
        self.update()

    def add(self, regions, stage: str, shares, combined: float) -> None:
        """One probe: the regions by number, the stage, their shares, and the one number."""
        regions = tuple(int(number) for number in regions)
        if regions != self._regions:
            self.clear()
            self._regions = regions
        self._stages.append(str(stage))
        self._shares.append(tuple(float(share) for share in shares))
        self._combined.append(float(combined))
        self.update()

    def set_history(self, regions, history) -> None:
        """A whole search at once: ``(stage, shares, combined)`` per probe."""
        self.clear()
        for stage, shares, combined in history:
            self.add(regions, stage, shares, combined)

    @property
    def probes(self) -> int:
        return len(self._combined)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))
        text = palette.color(QPalette.ColorRole.Text)
        faint = palette.color(QPalette.ColorRole.PlaceholderText)
        if len(self._combined) < 2:
            painter.setPen(faint)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                "Every region's sharpness through the search appears here "
                "once the compromise is being sought",
            )
            return

        font = QFont(painter.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - (2 if self._compact else 0)))
        painter.setFont(font)
        left = 6 if self._compact else 40
        area = QRectF(self.rect()).adjusted(left, 14, -18, -6 if self._compact else -22)
        count = len(self._combined)
        top = max(1.0, max(max(shares, default=0.0) for shares in self._shares))
        top = max(top, max(self._combined)) * 1.04

        def at(index: int, value: float) -> QPointF:
            return QPointF(
                area.left() + index * area.width() / (count - 1),
                area.bottom() - max(value, 0.0) / top * area.height(),
            )

        # What the search was doing, behind everything, each stretch named once.
        start = 0
        for index in range(1, count + 1):
            if index < count and self._stages[index] == self._stages[start]:
                continue
            name, shade = _STAGES.get(self._stages[start], ("", QColor(0, 0, 0, 0)))
            x0 = at(start, 0).x() - (area.width() / (count - 1)) / 2
            x1 = at(index - 1, 0).x() + (area.width() / (count - 1)) / 2
            band = QRectF(max(x0, area.left()), area.top() - 12, 0, 0)
            band.setRight(min(x1, area.right()))
            band.setBottom(area.bottom())
            painter.fillRect(band, shade)
            # Named only where there is room: in the panel the shading is
            # enough, and the names would sit on top of the line's own label.
            if name and band.width() > 40 and not self._compact:
                painter.setPen(faint)
                painter.drawText(
                    band.adjusted(3, 0, -2, 0),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                    name,
                )
            start = index

        # Every region's best is one: the line it is measured against.
        painter.setPen(QPen(faint, 1, Qt.PenStyle.DotLine))
        full = at(0, 1.0).y()
        painter.drawLine(QPointF(area.left(), full), QPointF(area.right(), full))
        if not self._compact:
            painter.setPen(faint)
            for share in (0.0, 0.25, 0.5, 0.75, 1.0):
                y = at(0, share).y()
                painter.drawText(
                    QRectF(0, y - 8, left - 4, 16),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    f"{share:.0%}",
                )
            painter.drawText(
                QRectF(area.left(), area.bottom() + 4, area.width(), 16),
                Qt.AlignmentFlag.AlignHCenter,
                f"probe, 1 to {count} -- each one every region read at one focus",
            )

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for column, number in enumerate(self._regions):
            colour = QColor(line_colour(number))
            points = [
                at(index, shares[column])
                for index, shares in enumerate(self._shares)
                if column < len(shares)
            ]
            painter.setPen(QPen(colour, 1.5))
            painter.drawPolyline(QPolygonF(points))
            painter.setPen(colour)
            end = points[-1]
            painter.drawText(
                QRectF(end.x() + 2, end.y() - 8, 16, 16),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                str(number),
            )

        # The one number the search climbs, over the regions it is made of.
        combined = [at(index, value) for index, value in enumerate(self._combined)]
        painter.setPen(QPen(text, 2.2, Qt.PenStyle.DashLine))
        painter.drawPolyline(QPolygonF(combined))
        painter.setPen(text)
        painter.drawText(
            QRectF(area.left() + 2, area.top() - 13, area.width(), 13),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            f"- - {self._combined_name}   now {self._combined[-1]:.0%}",
        )
