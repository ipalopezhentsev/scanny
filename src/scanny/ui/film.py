"""The shape of the film, from the depths of the regions, drawn in three dimensions.

A frame of film on a copy stand is not a plane. The holder grips it along its
four sides, and those sides lie more or less on a plane -- one that may lean
against the sensor, which is what levelling is for. Inside them the film is
foil: free to bow towards the lens or away from it, most in the middle and
not at all at the edges it is held by.

So that is the shape fitted to the regions' depths (:class:`FilmSurface`): a
plane for the edges, plus a bulge that is nothing at the edges and as smooth
as it can be while passing through every measured depth. Smooth in the sense
a thin sheet is -- the bulge is made of the shapes a sheet clamped at its
edges takes, the gentlest weighted as the cheapest -- so nothing is invented
between the regions that the regions do not ask for. With three regions there
is no bulge to see, three points being a plane; with four or more there is,
and the plane of the edges, which is the part levelling can correct, comes
apart from the bulge, which is the part it cannot.

The edges are taken to be the edges of the picture. That is true when the
film frame fills the camera's, which is how a copy stand is set up; where it
does not, the bulge is being measured against the wrong edges.

:class:`FilmView` draws it: the sensor flat underneath, the film above it
bent through the depths, each region's rectangle on the sensor joined by a
line to where that region is on the film -- which is how a place on the
picture becomes a place on the film to draw. Drag to turn it round, scroll to
come closer. The height is exaggerated, and has to be: the depths are drive
steps, which have no length in common with the frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from .orientation import Orientation

__all__ = ["FilmSurface", "FilmView", "Mark"]

#: The shapes a bulge is made of: sin(m pi x) sin(n pi y) for m and n up to
#: this. Each is nothing along all four edges, which is what being held by
#: them means. Nine is far more than five regions can pin down; the rest are
#: settled by being as gentle as possible, which is the point.
_MODES = 3

#: How finely the bulge is searched for its largest point.
_SAMPLES = 41


@dataclass(frozen=True)
class FilmSurface:
    """The film's shape: a plane for its edges and a bulge inside them.

    In fractions of the whole frame for position and drive steps for depth,
    positive being the way the drive calls further -- away from the camera.
    """

    #: The edges' plane: depth = a x + b y + c.
    plane: "tuple[float, float, float]"
    #: The bulge: (m, n, amount) per shape.
    modes: "tuple[tuple[int, int, float], ...]"
    #: How sure the plane is, from how sure each depth was: its covariance.
    plane_covariance: "tuple[tuple[float, ...], ...]"

    @classmethod
    def fit(cls, points, depths, doubts=None) -> "FilmSurface | None":
        """The gentlest held-at-the-edges sheet through every depth, or None.

        None for fewer than three regions, or three or more in a line: a
        plane needs three that are not, and without a plane there are no edges
        to hold the bulge against.

        The sheet passes through every depth exactly -- they are measured to a
        step or two, and five of them are not so many that any can be spared
        -- and among all the sheets that do, it is the one with the least
        bending in it: each shape weighted by the square of how sharply it
        curves, which is what bending a thin sheet costs. The plane costs
        nothing, so all of the lean the depths ask for goes into the plane and
        none of it is mistaken for a bulge.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        depths = np.asarray(depths, dtype=float)
        count = len(depths)
        if count < 3 or len(points) != count:
            return None
        plane_part = np.column_stack([points[:, 0], points[:, 1], np.ones(count)])
        if np.linalg.matrix_rank(plane_part, tol=1e-6) < 3:
            return None
        shapes = [(m, n) for m in range(1, _MODES + 1) for n in range(1, _MODES + 1)]
        bulge_part = np.column_stack(
            [_shape(m, n, points[:, 0], points[:, 1]) for m, n in shapes]
        )
        # Bending a sheet into sin(m pi x) sin(n pi y) costs as the square of
        # (m^2 + n^2): the gentle shapes are cheap and the rippled ones dear.
        cost = np.array([(m * m + n * n) ** 2 for m, n in shapes], dtype=float)
        # Least bending through every depth: minimise the cost of the bulge
        # subject to plane + bulge = depth at each region. Its solution is
        # linear in the depths, which is also how their doubt is carried.
        kernel = bulge_part @ np.diag(1.0 / cost) @ bulge_part.T
        system = np.block(
            [[kernel, plane_part], [plane_part.T, np.zeros((3, 3))]]
        )
        rhs = np.concatenate([np.eye(count), np.zeros((3, count))])
        solved = np.linalg.lstsq(system, rhs, rcond=None)[0]
        weights_of, plane_of = solved[:count], solved[count:]
        plane = plane_of @ depths
        amounts = np.diag(1.0 / cost) @ bulge_part.T @ (weights_of @ depths)
        if doubts is None:
            doubts = np.ones(count)
        spread = np.diag(np.asarray(doubts, dtype=float) ** 2)
        covariance = plane_of @ spread @ plane_of.T
        return cls(
            plane=tuple(float(v) for v in plane),
            modes=tuple(
                (m, n, float(amount)) for (m, n), amount in zip(shapes, amounts)
            ),
            plane_covariance=tuple(tuple(float(v) for v in row) for row in covariance),
        )

    # -- the shape ---------------------------------------------------------

    def edges_at(self, x, y):
        """The depth the edges' plane has at (x, y)."""
        a, b, c = self.plane
        return a * np.asarray(x, dtype=float) + b * np.asarray(y, dtype=float) + c

    def bulge_at(self, x, y):
        """How far the film stands off its edges' plane at (x, y)."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        out = np.zeros(np.broadcast(x, y).shape)
        for m, n, amount in self.modes:
            out = out + amount * _shape(m, n, x, y)
        return out

    def at(self, x, y):
        """The depth of the film at (x, y), in fractions of the frame."""
        return self.edges_at(x, y) + self.bulge_at(x, y)

    def bulge(self) -> "tuple[float, float, float]":
        """The bulge at its largest: how much, signed, and where (x, y)."""
        grid = np.linspace(0.0, 1.0, _SAMPLES)
        x, y = np.meshgrid(grid, grid)
        off = self.bulge_at(x, y)
        where = np.unravel_index(int(np.argmax(np.abs(off))), off.shape)
        return float(off[where]), float(x[where]), float(y[where])

    def lean(
        self, orientation: "Orientation | None" = None
    ) -> "tuple[float, float, float, float]":
        """How the edges' plane leans, in the picture's directions as shown.

        Answers how much further the right edge is than the left and the
        bottom than the top, in steps, and one standard deviation on each.
        Worked out in the directions the picture is shown in, because that is
        how the film is looked at while it is levelled.
        """
        orientation = orientation or Orientation()
        a, b, _c = self.plane
        covariance = np.array(self.plane_covariance)[:2, :2]
        # A step across the picture as shown is some step across the frame;
        # the plane's slope along it is the gradient taken along that step.
        answers = []
        for step in ((1.0, 0.0), (0.0, 1.0)):
            x0, y0 = orientation.from_view(0.5 - step[0] / 2, 0.5 - step[1] / 2)
            x1, y1 = orientation.from_view(0.5 + step[0] / 2, 0.5 + step[1] / 2)
            along = np.array([x1 - x0, y1 - y0])
            answers.append(
                (
                    float(along @ np.array([a, b])),
                    float(np.sqrt(max(along @ covariance @ along, 0.0))),
                )
            )
        (across, across_doubt), (down, down_doubt) = answers
        return across, down, across_doubt, down_doubt


def _shape(m: int, n: int, x, y):
    return np.sin(m * math.pi * np.asarray(x)) * np.sin(n * math.pi * np.asarray(y))


def _rim(i: int, j: int, columns: int, rows: int) -> "list[tuple[int, int]]":
    """Which sides of the mesh's face (i, j) lie on the frame's edge, as pairs
    of its corners: top-left, top-right, bottom-right, bottom-left."""
    sides = []
    if j == 0:
        sides.append((0, 1))
    if i == columns - 1:
        sides.append((1, 2))
    if j == rows - 1:
        sides.append((2, 3))
    if i == 0:
        sides.append((3, 0))
    return sides


# -- drawing it --------------------------------------------------------------


@dataclass(frozen=True)
class Mark:
    """One region, for the drawing: where on the frame, how deep, called what."""

    number: int
    #: The region's rectangle, in fractions of the whole frame.
    rect: "tuple[float, float, float, float]"
    depth: float


#: Where the view starts from, and returns to on a double click: turned a
#: little, and looking down at the film from a little above its edge.
_TURN, _TILT = -32.0, 28.0

#: How finely the film is drawn: across and down the picture.
_MESH = (30, 20)

#: How tall the film's shape is drawn against the frame's longer side, and
#: how far above the sensor it floats. Both are for the eye: the depths are in
#: drive steps and have no length in common with the frame.
_RELIEF = 0.35
_GAP = 0.3

_SENSOR = QColor(70, 74, 86)
_SENSOR_EDGE = QColor(150, 155, 170)

#: How the film is coloured by height: Turbo, at seven stops. A scale where
#: neighbouring heights are obviously different colours, because the
#: question asked of the drawing is "is that nearer than this".
_RAMP = (
    (48, 18, 59),
    (70, 134, 251),
    (39, 203, 158),
    (170, 232, 58),
    (249, 177, 50),
    (223, 64, 17),
    (122, 4, 3),
)


def ramp_colour(fraction: float) -> "tuple[int, int, int]":
    """The colour a height *fraction* of the way from lowest to highest is drawn."""
    place = min(max(float(fraction), 0.0), 1.0) * (len(_RAMP) - 1)
    lower = min(int(place), len(_RAMP) - 2)
    weight = place - lower
    return tuple(
        int(round(a * (1 - weight) + b * weight))
        for a, b in zip(_RAMP[lower], _RAMP[lower + 1])
    )


class FilmView(QWidget):
    """The sensor, the film above it bent through the regions' depths, and a
    line from each region's place on the sensor to its place on the film."""

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._surface: "FilmSurface | None" = None
        self._marks: "list[Mark]" = []
        self._orientation = Orientation()
        self._aspect = 1.5
        self._turn, self._tilt, self._zoom = _TURN, _TILT, 1.0
        self._dragging: "QPointF | None" = None

    # -- content -----------------------------------------------------------

    def set_scene(
        self,
        surface: "FilmSurface | None",
        marks: "list[Mark]",
        orientation: "Orientation | None" = None,
        aspect: float = 1.5,
    ) -> None:
        """What to draw. *aspect* is the frame's width over its height."""
        self._surface = surface
        self._marks = list(marks)
        self._orientation = orientation or Orientation()
        self._aspect = float(aspect) if aspect > 0 else 1.5
        self.update()

    def set_orientation(self, orientation: Orientation) -> None:
        self._orientation = orientation
        self.update()

    @property
    def angles(self) -> "tuple[float, float]":
        """Turned by, and looking down at, in degrees."""
        return self._turn, self._tilt

    # -- the geometry ------------------------------------------------------

    def _size(self) -> "tuple[float, float]":
        """The frame's width and depth in the drawing, as shown."""
        wide, tall = self._aspect, 1.0
        if self._orientation.swaps_axes:
            wide, tall = tall, wide
        return wide, tall

    def _heights(self) -> "tuple[float, float, float, float]":
        """The shallowest and deepest depth, the scale from steps to height,
        and how far above the sensor the shallowest is drawn."""
        long_side = max(self._size())
        depths = [mark.depth for mark in self._marks] or [0.0]
        if self._surface is not None:
            grid = np.linspace(0.0, 1.0, 12)
            x, y = np.meshgrid(grid, grid)
            sampled = self._surface.at(x, y)
            depths += [float(sampled.min()), float(sampled.max())]
        low, high = min(depths), max(depths)
        scale = _RELIEF * long_side / max(high - low, 1.0)
        return low, high, scale, _GAP * long_side

    def _world(self, u: float, v: float, height: float) -> "tuple[float, float, float]":
        """A place on the picture as shown, and a height, as a point in the drawing."""
        wide, tall = self._size()
        return (u - 0.5) * wide, (v - 0.5) * tall, height

    def _film(self, u: float, v: float, low: float, scale: float, gap: float):
        x, y = self._orientation.from_view(u, v)
        depth = float(self._surface.at(x, y)) if self._surface is not None else low
        return self._world(u, v, gap + (depth - low) * scale)

    def _project(self, point) -> "tuple[float, float, float]":
        """Screen x and y before scaling, and how far from the eye."""
        x, y, z = point
        turn, tilt = math.radians(self._turn), math.radians(self._tilt)
        across = x * math.cos(turn) - y * math.sin(turn)
        away = x * math.sin(turn) + y * math.cos(turn)
        screen_y = away * math.sin(tilt) - z * math.cos(tilt)
        # Square to the screen: what is drawn lower is nearer, as is what is
        # higher, when looking down. Otherwise the sensor behind the film's
        # near edge sorts in front of it and shows through.
        distance = -away * math.cos(tilt) - z * math.sin(tilt)
        return across, screen_y, distance

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))
        faint = palette.color(QPalette.ColorRole.PlaceholderText)
        text = palette.color(QPalette.ColorRole.Text)
        if self._surface is None:
            painter.setPen(faint)
            painter.drawText(
                self.rect().adjusted(20, 20, -20, -20),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                "The film's shape needs the depths of three regions or more, "
                "not all in a line",
            )
            return
        low, high, scale, gap = self._heights()
        top = gap + (high - low) * scale
        columns, rows = _MESH

        # Every face, sensor and film alike, so that whichever is nearer the
        # eye is drawn over the other from above and from below. The sensor's
        # outline and the film's edges -- what the holder grips -- go with the
        # faces along them, so they are hidden by what is in front as well.
        faces: "list[tuple[float, list, QColor, list, QPen]]" = []
        sensor_rim = QPen(_SENSOR_EDGE, 1.2)
        film_rim = QPen(text, 1.6)
        sensor_step = 6
        for i in range(sensor_step):
            for j in range(sensor_step):
                corners = [
                    self._world(u, v, 0.0)
                    for u, v in (
                        (i / sensor_step, j / sensor_step),
                        ((i + 1) / sensor_step, j / sensor_step),
                        ((i + 1) / sensor_step, (j + 1) / sensor_step),
                        (i / sensor_step, (j + 1) / sensor_step),
                    )
                ]
                rim = _rim(i, j, sensor_step, sensor_step)
                faces.append((0.0, corners, _SENSOR, rim, sensor_rim))
        for i in range(columns):
            for j in range(rows):
                corners = [
                    self._film(u, v, low, scale, gap)
                    for u, v in (
                        (i / columns, j / rows),
                        ((i + 1) / columns, j / rows),
                        ((i + 1) / columns, (j + 1) / rows),
                        (i / columns, (j + 1) / rows),
                    )
                ]
                height = sum(corner[2] for corner in corners) / 4.0
                colour = self._shade(corners, height, gap, top)
                faces.append((height, corners, colour, _rim(i, j, columns, rows), film_rim))

        projected = [
            (
                sum(self._project(corner)[2] for corner in corners) / 4.0,
                [self._project(corner) for corner in corners],
                colour,
                rim,
                pen,
            )
            for _h, corners, colour, rim, pen in faces
        ]
        # Fit everything to the widget, then come closer by the zoom.
        xs = [p[0] for _d, points, *_rest in projected for p in points]
        ys = [p[1] for _d, points, *_rest in projected for p in points]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
        size = min(self.width(), self.height()) * 0.86 * self._zoom / span
        middle = QPointF(self.width() / 2, self.height() / 2 + 10)
        centre = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2)

        def to_screen(p) -> QPointF:
            return QPointF(
                middle.x() + (p[0] - centre[0]) * size,
                middle.y() + (p[1] - centre[1]) * size,
            )

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for _distance, points, colour, rim, pen in sorted(projected, key=lambda f: -f[0]):
            screen = [to_screen(p) for p in points]
            painter.setBrush(colour)
            painter.setPen(QPen(colour.darker(115), 0.6))
            painter.drawPolygon(QPolygonF(screen))
            painter.setPen(pen)
            for start, end in rim:
                painter.drawLine(screen[start], screen[end])
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Each region: its rectangle on the sensor, a line up to the film, and
        # a mark on the film where it was measured.
        font = QFont(painter.font())
        font.setBold(True)
        painter.setFont(font)
        for mark in self._marks:
            x, y, w, h = self._orientation.rect_to_view(mark.rect)
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            painter.setPen(QPen(QColor(255, 200, 80), 1.6))
            painter.drawPolyline(
                QPolygonF(
                    [to_screen(self._project(self._world(u, v, 0.0))) for u, v in corners]
                )
            )
            u, v = x + w / 2, y + h / 2
            below = to_screen(self._project(self._world(u, v, 0.0)))
            above = to_screen(
                self._project(self._world(u, v, gap + (mark.depth - low) * scale))
            )
            painter.setPen(QPen(QColor(255, 200, 80), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(below, above)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 200, 80))
            painter.drawEllipse(above, 4.0, 4.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(text)
            further = mark.depth - low
            label = f"{mark.number}  +{further:.0f}" if further >= 0.5 else f"{mark.number}"
            painter.drawText(
                QRectF(above.x() + 6, above.y() - 18, 90, 16),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

        painter.setPen(faint)
        small = QFont(painter.font())
        small.setBold(False)
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1))
        painter.setFont(small)
        painter.drawText(
            QRectF(8, 6, self.width() - 16, 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            "Film above, sensor below; higher is further from the camera. "
            "Height exaggerated: drive steps, not to scale.",
        )
        painter.drawText(
            QRectF(8, self.height() - 22, self.width() - 16, 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            "Drag to turn it, scroll to come closer, double-click to start again",
        )

    def _shade(self, corners, height: float, gap: float, top: float) -> QColor:
        """A face's colour: its height on the depth ramp, lit from above-left."""
        fraction = 0.0 if top <= gap + 1e-9 else (height - gap) / (top - gap)
        red, green, blue = ramp_colour(min(max(fraction, 0.0), 1.0))
        (x0, y0, z0), (x1, y1, z1), _c, (x3, y3, z3) = corners
        normal = np.cross([x1 - x0, y1 - y0, z1 - z0], [x3 - x0, y3 - y0, z3 - z0])
        length = float(np.linalg.norm(normal)) or 1.0
        light = np.array([-0.4, -0.5, 0.77])
        lit = 0.55 + 0.45 * abs(float(normal @ light)) / length
        return QColor(int(red * lit), int(green * lit), int(blue * lit))

    # -- turning it --------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging is None:
            return
        moved = event.position() - self._dragging
        self._dragging = event.position()
        self._turn = (self._turn - moved.x() * 0.5) % 360.0
        self._tilt = min(max(self._tilt + moved.y() * 0.5, -85.0), 89.0)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._dragging = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self._turn, self._tilt, self._zoom = _TURN, _TILT, 1.0
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        notches = event.angleDelta().y() / 120.0
        self._zoom = min(max(self._zoom * (1.12**notches), 0.4), 6.0)
        self.update()
