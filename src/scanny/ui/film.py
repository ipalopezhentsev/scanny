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
picture becomes a place on the film to draw. All of that lies under the film,
so it is drawn faintly wherever the film is in front of it and fully where
nothing is; at full strength throughout, a line would read as standing in
front of the film it is really behind. Turned far enough over it is the
film's underside that is on show, and that is drawn dulled and darkened,
because two sides alike mean a view from below passes for a view from above
and every lean read off it is backwards. Drag to turn it round, scroll to come
closer. The height is exaggerated, and has to be: the depths are drive steps,
which have no length in common with the frame.
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


def _faded(colour: QColor) -> QColor:
    """*colour* as it is drawn where something stands in front of it."""
    out = QColor(colour)
    out.setAlpha(_COVERED)
    return out


def _subdivided(corners, pieces: int):
    """The way round *corners*, cut into *pieces* along each side.

    Whether the film covers a mark changes along it and not only at its
    corners, so it is drawn as many short pieces rather than a few long ones.
    """
    out = []
    for start, end in zip(corners, corners[1:]):
        out += [
            tuple(a + (b - a) * k / pieces for a, b in zip(start, end))
            for k in range(pieces)
        ]
    if corners:
        out.append(tuple(corners[-1]))
    return out


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

#: Where the light comes from for the film's top, and for its underside: the
#: same light turned round, so that the far side of the sheet has a shape to
#: it too rather than going flat.
_LIGHT = np.array([-0.4, -0.5, 0.77])
_UNDER_LIGHT = np.array([0.4, 0.5, -0.77])

#: What a face lying flat takes of that light. Shading is reckoned against it
#: rather than against a face turned square to the light, which the film never
#: is: so film lying flat is drawn in the ramp's own colour at full strength,
#: and only what slopes away from the light falls short of it. The colours are
#: the one thing the drawing is read by, and a light that dims all of them
#: costs more than the little relief it adds.
_FLAT = float(_LIGHT[2])

#: How the underside is drawn against the top: this much of the height's own
#: colour and the rest this slate, then darkened by this much. The heights are
#: still to be read off it, but at a glance it is plainly the back of the
#: sheet and not the face of it.
_UNDER_HUE = 0.45
_UNDER_SLATE = (96, 100, 116)
_UNDER_LIT = 0.62

#: What a region is drawn in: its rectangle on the sensor, the post up to the
#: film, and the mark on the film at the top of it.
_MARK = QColor(255, 200, 80)

#: How much of itself any of that keeps where the film or the sensor is in
#: front of it. Faint enough to read as behind, plain enough to follow.
_COVERED = 80

#: How finely a mark is cut up to be drawn part covered and part not, along a
#: post and along each side of a rectangle.
_PIECES = 24

#: How many steps the way out towards the eye is tried in, looking for the
#: film or the sensor in front of a point.
_RAYS = 40

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
    line from each region's place on the sensor to its place on the film.

    The faces are sorted and drawn back to front, which is enough for two
    sheets that do not cut through each other. The marks are drawn over all
    of them afterwards, and so are shown faintly wherever the film or the
    sensor stands between them and the eye (:meth:`_covered`)."""

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

    def _towards(self):
        """The way to the eye, the same for every point: the view is square-on.

        Going this way takes a point straight towards the eye, which is what
        both the side of a face being looked at and what is in front of a mark
        are worked out along.
        """
        turn, tilt = math.radians(self._turn), math.radians(self._tilt)
        return np.array(
            [
                math.cos(tilt) * math.sin(turn),
                math.cos(tilt) * math.cos(turn),
                math.sin(tilt),
            ]
        )

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
        # a mark on the film where it was measured. All of it is under the
        # film, so it is drawn faintly wherever the film is in front of it and
        # fully where nothing is. Drawn at full strength throughout, a post
        # reads as standing in front of the film it is really behind, and
        # turning the view round does not say otherwise.
        font = QFont(painter.font())
        font.setBold(True)
        painter.setFont(font)
        heights = (low, scale, gap, top)
        for mark in self._marks:
            x, y, w, h = self._orientation.rect_to_view(mark.rect)
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            self._trace(
                painter,
                _subdivided([self._world(u, v, 0.0) for u, v in corners], _PIECES),
                to_screen,
                heights,
                QPen(_MARK, 1.6),
            )
            u, v = x + w / 2, y + h / 2
            reaches = gap + (mark.depth - low) * scale
            post = [
                self._world(u, v, reaches * k / _PIECES) for k in range(_PIECES + 1)
            ]
            self._trace(
                painter, post, to_screen, heights,
                QPen(_MARK, 1.2, Qt.PenStyle.DashLine),
            )
            above = to_screen(self._project(post[-1]))
            # The mark itself sits on the film, so it is behind it only when
            # the film is being looked at from underneath.
            hidden = bool(self._covered(post[-1:], *heights)[0])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_faded(_MARK) if hidden else _MARK)
            painter.drawEllipse(above, 4.0, 4.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(_faded(text) if hidden else text)
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

    def _covered(self, points, low: float, scale: float, gap: float, top: float):
        """Which of the world *points* the film or the sensor is in front of.

        The view is square-on, so every point is looked at along the same
        direction: the way to the eye is followed from each of them, and what
        it goes through on the way is what is in front of that point. It is
        followed only as far as the frame's sides, because past them there is
        neither film nor sensor to go through, which is what looking in under
        the film's edge is.

        The film is gone through where the way crosses it, which is not the
        same as ending up above it: a mark lying on the film starts out on it
        and rises off it, and that is the film being looked at rather than the
        film being in the way. It counts only once the way has been under the
        film and come out over it -- or straight away, for a mark lying on the
        film whose way out goes under it, because then what is being looked at
        there is the film's underside and the mark is on the other face of it.
        The sensor is flat, so under it is under it.
        """
        count = len(points)
        if self._surface is None or count == 0:
            return np.zeros(count, dtype=bool)
        wide, tall = self._size()
        reach = wide + tall + top
        # From the points themselves, so that which side of the film each one
        # starts on is read off the same way as the rest.
        steps = np.linspace(0.0, reach, _RAYS + 1)
        way = np.asarray(points, dtype=float)[:, None, :] + steps[:, None] * self._towards()
        u = way[..., 0] / wide + 0.5
        v = way[..., 1] / tall + 0.5
        inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        # Once out past the frame's side it is out for good, so nothing met
        # after that is between the point and the eye.
        before = np.cumprod(inside, axis=1).astype(bool)
        x, y = self._orientation.from_view(u, v)
        film = gap + (self._surface.at(x, y) - low) * scale
        # A hair of room, so that lying on the film counts as neither side.
        skin = 1e-3 * max(wide, tall)
        under = way[..., 2] < film - skin
        over = way[..., 2] > film + skin
        was_under = np.zeros_like(under)
        was_under[:, 1:] = np.maximum.accumulate(under, axis=1)[:, :-1]
        crosses = over & was_under
        # On the film and going under it: its underside is what faces the eye
        # there, and the mark is on the face turned away.
        turned = (np.abs(way[:, 0, 2] - film[:, 0]) <= skin) & under[:, 1]
        return turned | np.any(before & (crosses | (way[..., 2] < -skin)), axis=1)

    def _trace(self, painter, points, to_screen, heights, pen: QPen) -> None:
        """Draw the way through the world *points*, faint where it is covered.

        Faintly the whole way first, then fully over the stretches nothing is
        in front of, each carrying on the dashes of the faint line under it so
        that a dashed line does not come out doubled where the two meet.
        """
        screen = [to_screen(self._project(point)) for point in points]
        covered = self._covered(points, *heights)
        faint = QPen(pen)
        faint.setColor(_faded(pen.color()))
        painter.setPen(faint)
        painter.drawPolyline(QPolygonF(screen))
        gone, start, clear = 0.0, 0.0, []
        for i, point in enumerate(screen):
            if i:
                was = screen[i - 1]
                gone += math.hypot(point.x() - was.x(), point.y() - was.y())
            if covered[i]:
                self._stroke(painter, pen, clear, start)
                clear = []
                continue
            if not clear:
                start = gone
            clear.append(point)
        self._stroke(painter, pen, clear, start)

    @staticmethod
    def _stroke(painter, pen: QPen, run, start: float) -> None:
        """One unhidden stretch, its dashes taken up *start* along the way."""
        if len(run) < 2:
            return
        full = QPen(pen)
        full.setDashOffset(start / max(pen.widthF(), 0.1))
        painter.setPen(full)
        painter.drawPolyline(QPolygonF(run))

    def _shade(self, corners, height: float, gap: float, top: float) -> QColor:
        """A face's colour: its height on the depth ramp, lit from above-left.

        Dulled and darkened when it is its underside that is being looked at.
        The two sides of the sheet are the same shape and the same heights, so
        with nothing to tell them apart a view from below passes for a view
        from above and every lean in it is read backwards.
        """
        fraction = 0.0 if top <= gap + 1e-9 else (height - gap) / (top - gap)
        red, green, blue = ramp_colour(min(max(fraction, 0.0), 1.0))
        (x0, y0, z0), (x1, y1, z1), _c, (x3, y3, z3) = corners
        normal = np.cross([x1 - x0, y1 - y0, z1 - z0], [x3 - x0, y3 - y0, z3 - z0])
        length = float(np.linalg.norm(normal)) or 1.0
        # The corners go round so that the normal is the way the top faces.
        underside = float(normal @ self._towards()) < 0.0
        seen = -normal if underside else normal
        light = _UNDER_LIGHT if underside else _LIGHT
        lit = min(0.55 + 0.45 * max(float(seen @ light), 0.0) / (length * _FLAT), 1.0)
        if underside:
            red, green, blue = (
                _UNDER_HUE * own + (1 - _UNDER_HUE) * slate
                for own, slate in zip((red, green, blue), _UNDER_SLATE)
            )
            lit *= _UNDER_LIT
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
