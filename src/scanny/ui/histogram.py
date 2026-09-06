"""The distribution of levels in the picture on screen, as three curves.

Focus has a number to drive against; exposure does not, and the eye is a poor
judge of it on a screen that is itself a guess at the room's brightness. The
histogram is what a photographer reads instead: where the levels sit, and
whether anything has been pushed off either end of the scale, which no amount
of later work brings back.

Two decisions worth knowing about:

- **The height is scaled to the middle of the range**, not to the whole of it.
  A frame that clips -- or one of a light box, which is most of the job here
  -- puts an enormous spike in bin 0 or bin 255, and scaling to that flattens
  everything else into the floor. Leaving the two end bins out of the scale
  keeps the shape of the picture readable, and the clipping is reported as a
  number underneath instead, where it is more use than a tall bar.
- **The count is sampled, not exhaustive.** Counting every pixel thirty times
  a second buys nothing: a few tens of thousands of them already describe the
  shape to well within the width of a drawn line.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from .pixels import view

__all__ = ["HistogramWidget"]

_BINS = 256

#: The channels, in the byte order an RGB32 buffer has on a little-endian
#: machine, each with the colour it is drawn in. Red goes down first so that
#: blue, which usually carries the most noise and so the widest curve, ends up
#: on top rather than buried under the other two.
_CHANNELS = (
    (2, QColor(255, 96, 96)),
    (1, QColor(96, 220, 120)),
    (0, QColor(110, 160, 255)),
)

#: Roughly how many pixels are counted. The sample is a regular grid over the
#: whole picture, so it describes all of it rather than one corner of it.
_SAMPLE_TARGET = 60_000

#: The shortest gap between two counts. Frames arrive at up to thirty a second
#: and no one reads a histogram that fast.
_MIN_INTERVAL = 0.1

#: Below this fraction of the picture, a count in an end bin is the odd
#: specular highlight or a fleck of true black, not lost detail.
_CLIPPING_FLOOR = 0.0005


class HistogramWidget(QWidget):
    """Counts the levels in each displayed picture and draws them.

    Feed it every picture with :meth:`set_image`; it decides for itself how
    often that is worth acting on.
    """

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self._counts: "np.ndarray | None" = None
        self._clipped = (0.0, 0.0)
        self._counted_at = 0.0
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # A readout, not a control: it must never take the keyboard away from
        # the image.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    # -- content -----------------------------------------------------------

    def set_image(self, image: QImage) -> bool:
        """Count *image*, unless one was counted a moment ago.

        Answers whether it actually counted, so a caller with a readout to
        write alongside can leave it alone the rest of the time.
        """
        now = time.monotonic()
        if image.isNull() or now - self._counted_at < _MIN_INTERVAL:
            return False
        self._counted_at = now
        if image.format() != QImage.Format.Format_RGB32:
            # Kept in a local: the array below borrows this image's buffer,
            # and a converted temporary would be freed on the way out.
            image = image.convertToFormat(QImage.Format.Format_RGB32)
        step = _sample_step(image.width(), image.height())
        sample = view(image)[::step, ::step]
        total = sample.shape[0] * sample.shape[1]
        if not total:
            return False
        self._counts = np.stack(
            [
                np.bincount(sample[:, :, index].ravel(), minlength=_BINS)
                for index, _ in _CHANNELS
            ]
        )
        self._clipped = (
            float(self._counts[:, 0].max()) / total,
            float(self._counts[:, _BINS - 1].max()) / total,
        )
        self.update()
        return True

    def clear(self) -> None:
        self._counts = None
        self._clipped = (0.0, 0.0)
        self._counted_at = 0.0
        self.update()

    @property
    def counts(self) -> "np.ndarray | None":
        """The last counts, as a ``(3, 256)`` array in red, green, blue order."""
        return self._counts

    @property
    def clipping(self) -> "tuple[float, float]":
        """What fraction of the picture sits at each end of the scale.

        Shadows first, then highlights, each the worst of the three channels:
        a single blown channel is already a colour that cannot be recovered,
        however comfortable the other two are.
        """
        return self._clipped

    def describe(self) -> str:
        """The clipping, in the words the panel puts under the curves."""
        if self._counts is None:
            return "Waiting for a frame..."
        shadows, highlights = self._clipped
        parts = []
        if shadows > _CLIPPING_FLOOR:
            parts.append(f"{100 * shadows:.1f}% crushed")
        if highlights > _CLIPPING_FLOOR:
            parts.append(f"{100 * highlights:.1f}% blown")
        return "   ".join(parts) if parts else "Nothing clipped"

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))

        if self._counts is None:
            painter.setPen(palette.color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for a frame..."
            )
            return

        area = self.rect().adjusted(1, 3, -1, -1)
        self._draw_quarters(painter, area)
        peak = self._peak()
        if peak <= 0:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for row, (_, colour) in enumerate(_CHANNELS):
            self._draw_channel(painter, area, self._counts[row], peak, colour)

    def _peak(self) -> float:
        """The count the tallest drawn bin stands for."""
        assert self._counts is not None
        peak = float(self._counts[:, 1 : _BINS - 1].max())
        # Everything at one end or the other: a lens cap, or a light box with
        # nothing on it. Scale to the whole range instead, so the spike that
        # says so is at least drawn.
        return peak if peak > 0 else float(self._counts.max())

    def _draw_channel(
        self,
        painter: QPainter,
        area: QRect,
        counts: "np.ndarray",
        peak: float,
        colour: QColor,
    ) -> None:
        step = area.width() / (_BINS - 1)
        points = [
            QPointF(
                area.left() + index * step,
                area.bottom() - min(count / peak, 1.0) * area.height(),
            )
            for index, count in enumerate(counts)
        ]
        # Filled, because three bare lines over one another are hard to tell
        # apart; translucent, because where they overlap is the interesting
        # part -- that is what a neutral grey looks like.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(colour.red(), colour.green(), colour.blue(), 70))
        painter.drawPolygon(
            QPolygonF(
                points
                + [
                    QPointF(points[-1].x(), area.bottom()),
                    QPointF(points[0].x(), area.bottom()),
                ]
            )
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(colour, 1.0))
        painter.drawPolyline(QPolygonF(points))

    def _draw_quarters(self, painter: QPainter, area: QRect) -> None:
        """Faint marks at the quarter tones, to read a level against."""
        grid = self.palette().color(QPalette.ColorRole.PlaceholderText)
        grid.setAlpha(60)
        painter.setPen(QPen(grid, 1, Qt.PenStyle.DotLine))
        for quarter in (0.25, 0.5, 0.75):
            x = int(area.left() + quarter * area.width())
            painter.drawLine(x, area.top(), x, area.bottom())


def _sample_step(width: int, height: int) -> int:
    """Take every nth pixel in both directions, for about the target count."""
    return max(1, int((max(1, width * height) / _SAMPLE_TARGET) ** 0.5))
