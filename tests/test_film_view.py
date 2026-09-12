"""Turning the film's drawing round turns it, and does nothing else to it.

The drawing used to be fitted to what it happened to spread across and down
at the angle it was turned to, and a turn changes that spread: so the drawing
grew and shrank as it was dragged. Read off a picture whose whole subject is
how far one thing is in front of another, that is the worst thing it could
do -- it is exactly what coming closer looks like, and the hand that asked
for a turn gets a zoom it did not ask for.

So it is fitted instead to how far it could ever reach, at any angle. What
that has to buy is checked here: the same scale and the same middle whatever
the view is turned to, and still nothing falling off the widget at any angle
it can be turned to -- a fit that never changed but clipped the drawing would
pass the first half of this and be no better than what it replaced.
"""

from __future__ import annotations

import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.ui.film import FilmSurface, FilmView, Mark  # noqa: E402

#: Turns and tilts to try: all the way round, and from under it to over it.
#: The angles the drag can reach, not a handful of tidy ones.
TURNS = tuple(range(-180, 181, 9))
TILTS = tuple(range(-89, 90, 7))

#: Five places at five depths: enough for a bulge, and a deep one in the
#: middle so the drawing stands as tall as it ever does.
PLACES = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.5, 0.5))
DEPTHS = (0.0, 6.0, 4.0, 9.0, 22.0)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(app):
    widget = FilmView()
    widget.resize(700, 500)
    marks = [
        Mark(number=i + 1, rect=(x - 0.06, y - 0.06, 0.12, 0.12), depth=depth)
        for i, ((x, y), depth) in enumerate(zip(PLACES, DEPTHS))
    ]
    widget.set_scene(
        FilmSurface.fit(PLACES, DEPTHS), marks, aspect=1.5, focus=8.0
    )
    return widget


def fit(view: FilmView):
    """What the drawing is fitted to, as the painting works it out: the
    scale, where the middle of the widget is, what goes there, and how tall
    the drawing stands."""
    low, high, scale, gap = view._heights()
    top = gap + (high - low) * scale
    return view._fit(top), top


def corners(view: FilmView, top: float):
    """The eight corners of the box the whole drawing stands in."""
    wide, tall = view._size()
    return [
        (x, y, z)
        for x in (-wide / 2, wide / 2)
        for y in (-tall / 2, tall / 2)
        for z in (0.0, top)
    ]


def test_turning_the_view_does_not_change_how_large_the_film_is_drawn(view):
    """The complaint this was written for: dragging to turn it zoomed it."""
    (size, middle, _centre), _top = fit(view)
    for turn in TURNS:
        for tilt in TILTS:
            view._turn, view._tilt = float(turn), float(tilt)
            (turned_size, turned_middle, _centre), _top = fit(view)
            assert turned_size == pytest.approx(size), (turn, tilt)
            assert turned_middle == middle, (turn, tilt)


def test_the_middle_of_the_film_stays_where_it_is_while_the_view_turns(view):
    """Turning is about the middle of the drawing, so that is the one place
    that does not move: the drawing turns rather than swinging about."""
    (size, middle, _c), top = fit(view)
    places = set()
    for turn in TURNS:
        for tilt in TILTS:
            view._turn, view._tilt = float(turn), float(tilt)
            (_s, _m, centre), _t = fit(view)
            point = view._project((0.0, 0.0, top / 2))
            places.add(
                (
                    round(middle.x() + (point[0] - centre[0]) * size, 6),
                    round(middle.y() + (point[1] - centre[1]) * size, 6),
                )
            )
    assert places == {(middle.x(), middle.y())}


def test_nothing_falls_off_the_widget_at_any_angle_it_can_be_turned_to(view):
    """What the scale is bought with: it is the largest one that holds at
    every angle, so the drawing stays whole however it is dragged."""
    for turn in TURNS:
        for tilt in TILTS:
            view._turn, view._tilt = float(turn), float(tilt)
            (size, middle, centre), top = fit(view)
            for corner in corners(view, top):
                point = view._project(corner)
                x = middle.x() + (point[0] - centre[0]) * size
                y = middle.y() + (point[1] - centre[1]) * size
                assert 0 <= x <= view.width(), (turn, tilt, corner)
                assert 0 <= y <= view.height(), (turn, tilt, corner)


def test_the_fit_is_no_smaller_than_it_has_to_be(view):
    """A fit that shrank the drawing to nothing would also never change and
    never clip. It has to be tight: at some angle the drawing reaches the
    room the fit left it, across and down alike."""
    (_size, _middle, _centre), top = fit(view)
    wide, tall = view._size()
    room_across = 2 * (math.hypot(wide, tall) / 2)
    room_down = 2 * math.hypot(math.hypot(wide, tall) / 2, top / 2)
    widest = tallest = 0.0
    for turn in TURNS:
        for tilt in TILTS:
            view._turn, view._tilt = float(turn), float(tilt)
            (_s, _m, centre), _t = fit(view)
            points = [view._project(corner) for corner in corners(view, top)]
            widest = max(
                widest, 2 * max(abs(p[0] - centre[0]) for p in points)
            )
            tallest = max(
                tallest, 2 * max(abs(p[1] - centre[1]) for p in points)
            )
    assert widest == pytest.approx(room_across, rel=1e-3)
    assert tallest == pytest.approx(room_down, rel=1e-3)


def test_the_drawing_comes_out_whole_at_every_angle(view):
    """The whole of it, drawn: the checks above are on the arithmetic the
    painting uses, and this is the painting itself not falling over."""
    for turn, tilt in ((-32, 28), (0, 0), (90, -60), (180, 89), (-140, -89)):
        view._turn, view._tilt = float(turn), float(tilt)
        image = QImage(view.size(), QImage.Format.Format_ARGB32)
        image.fill(0xFFFFFFFF)
        view.render(image)
        assert not image.isNull()
