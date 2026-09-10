"""How far apart, in focus, are the few places someone pointed at.

The depth map answers the same question about every part of the picture at
once, and pays for it: a zone of a grid is small, it is wherever the grid put
it rather than on anything in particular, and most of a frame is not worth
measuring. This asks the question about a handful of places a person chose,
and almost everything that made the map fragile goes away with that choice.
The boxes are large, they have subject in them because someone looked before
clicking, and there are five of them rather than two thousand -- so a test
that has to hold for every one of them can be strict without throwing away
most of the answer.

**It is still one sweep, not a hunt per point.** Pointing the hunt in
:mod:`scanny.ui.hunt` at each box in turn is the obvious construction and it
is worse in both directions at once. It costs five hunts of twenty to forty
probes where a sweep costs its stops once, because every box is read off the
same frame. And its answers do not compare: a hunt walks back and forth and
finishes wherever the reading told it to, so what separates the resting places
of two hunts is the focus difference plus whatever play the gearing took up on
the way, and the play is the thing nothing here can measure. A single pass
driving one way has every reading in one coordinate by construction. See
:mod:`scanny.ui.depth` for why that coordinate means anything.

**Magnification is what makes a small gap readable, and it is why the points
live in sensor coordinates.** A step of focus moves the picture far more when
the view is magnified, so two things a hundred steps apart that are
indistinguishable on a whole frame are obvious at 18.8x. But at 18.8x the
screen shows a hundredth of the frame, so the points that are to be compared
are not on it together -- and reading them off the same frame, which is the
whole basis of the paragraph above, stops being possible. What replaces it is
reading them off the same *stop*: the sweep pans the camera from point to
point without touching focus, so every reading at a given position still
belongs to one place on the travel, driven to one way, from one datum. Panning
is free in the coordinate that matters. See
:meth:`scanny.ui.worker.CameraWorker._read_points_apart`.

So the whole of this module is: which part of the picture belongs to each
point, and what to make of the readings once they are in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .depth import _TOP, _spacing, locate

__all__ = [
    "BOX",
    "Found",
    "MAX_POINTS",
    "MIN_BOX",
    "Point",
    "PointSurvey",
    "area_for",
    "ordering",
    "summarise",
]

#: How many places may be asked about at once. Not a limit of the arithmetic
#: -- every one of them is read off the same frame, so the tenth costs what the
#: second does. It is a limit of the picture: the boxes have to be big enough
#: to read and far enough apart to mean different things, and past about half a
#: dozen on a live-view frame they are neither.
MAX_POINTS = 5

#: How wide the box read around a point is, as a fraction of the frame, and the
#: least it may be set to. Wide enough to hold some subject, narrow enough that
#: two points a third of the frame apart are reading different things.
BOX = 0.12
MIN_BOX = 0.03

#: How much bigger than the doubt on it a gap has to be before the two
#: points either side of it are called apart rather than the same.
#:
#: There has to be a number like this, and its absence is what made the
#: first version of this worse than useless. Whether two things a hundred
#: steps apart can be told apart at all is not a property of the arithmetic;
#: it is a property of the lens, the magnification and what is in the boxes.
#: Read the whole frame of an unmagnified scene and a hundred steps may move
#: the reading less than the grain does, in which case the honest answer is
#: 'these two are the same as far as I can tell' -- and the answer that
#: comes out instead, if nobody works out the doubt, is a confident few
#: steps in whichever order the noise picked.
#:
#: The doubt is one standard deviation, so this is how many of those a gap has
#: to clear. Two and a half of them is about one wrong call in a hundred, and
#: the asymmetry is deliberate: being told two things cannot be told apart
#: costs a magnified second look, and being told the wrong one is nearer costs
#: whatever was built on believing it.
SURE = 2.5

#: How far past the top of a peak the next pass has to reach, in multiples of
#: how wide that top is. A sweep can only place a hill by seeing it fall away
#: on both sides, so refining down to the peaks themselves -- which is what
#: narrowing by one step of margin does -- leaves the finest pass looking at a
#: plateau and answering with the grain on it.
_SHOULDERS = 1.5


@dataclass(frozen=True)
class Point:
    """Somewhere on the sensor's frame, in fractions of **the whole frame**.

    Fractions of the frame and not of what is on screen, and that is the one
    thing about this module that was rebuilt rather than added to. Screen
    fractions look right until the view is magnified: a point put a tenth of
    the way across an unmagnified frame stays a tenth of the way across the
    strip the camera shows at 18.8x, which is a different piece of the world
    entirely -- so a pair of points placed on opposite corners to be compared
    end up, at the magnification that would actually separate them, on two
    places nobody chose.

    Frame fractions cannot do that. They are the coordinate the camera's own
    focus point lives in, they do not move when the view magnifies or pans,
    and they are what :meth:`seen_in` turns back into screen fractions for
    whichever crop happens to be on show. What was a place in the picture is
    now a place on the sensor, which is what someone pointing at their subject
    meant in the first place.
    """

    x: float
    y: float

    def near(self, x: float, y: float, within: float) -> bool:
        return (self.x - x) ** 2 + (self.y - y) ** 2 <= within * within

    def seen_in(
        self, crop: "tuple[float, float, float, float]"
    ) -> "tuple[float, float] | None":
        """Where this lands on a view showing *crop* of the frame, or None.

        *crop* is ``(x, y, w, h)`` in fractions of the whole frame -- what
        :attr:`scanny.camera.nikon.LiveViewFrame.crop_normalised` reports. None
        when the point is outside it, which is the ordinary case once the view
        is magnified: four of five points are off screen and the answer to
        "where is it on this picture" is that it is not.
        """
        left, top, width, height = crop
        if width <= 0.0 or height <= 0.0:
            return None
        x = (self.x - left) / width
        y = (self.y - top) / height
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        return x, y


@dataclass(frozen=True)
class Found:
    """What the sweep made of one point."""

    point: Point
    #: Where it was sharpest, in drive steps from the near stop, or None if it
    #: never showed a peak worth believing.
    steps: "float | None"
    #: The reading at that peak: how much subject the answer rests on.
    strength: float
    #: Whether its best reading was at an end of what was swept, which says the
    #: sweep did not contain its peak rather than that it is the furthest away.
    at_the_limit: bool
    #: How far out this could be, in steps: how much the answer moved when it
    #: was worked out twice from halves of the same readings. See
    #: :meth:`PointSurvey.found`.
    doubt: float = 0.0
    #: Its place in the order, nearest first, or None when it has no answer.
    rank: "int | None" = None
    #: How far behind the nearest point it is, and behind the one before it.
    behind_nearest: "float | None" = None
    behind_previous: "float | None" = None

    @property
    def known(self) -> bool:
        return self.steps is not None

    def apart_from(self, other: "Found") -> bool:
        """Whether the gap to *other* is bigger than the doubt on it."""
        if not (self.known and other.known):
            return False
        doubt = np.hypot(self.doubt, other.doubt)
        return abs(self.steps - other.steps) > SURE * max(doubt, 1e-9)


def area_for(
    x: float, y: float, box: float = BOX, aspect: float = 1.0
) -> "tuple[float, float, float, float]":
    """The rectangle read around ``(x, y)``, in fractions of what is on screen.

    Screen fractions in and screen fractions out, unlike :class:`Point`, which
    is a place on the sensor: by the time there is a box to read, the caller
    has a picture in front of it and has already asked the point where it
    falls on that picture. See :meth:`Point.seen_in`.

    The box is what gets an answer, not the point at its middle: a box
    straddling two things at different distances comes back with a
    reading-weighted average of the two rather than one of them, which is the
    right answer to "how far away is the stuff in this box" and the wrong
    answer to "how far away is that edge". Move the box, or make it smaller.

    Square on the screen rather than square in fractions, which is why the
    aspect comes into it: a box that is 0.12 of the width and 0.12 of the
    height on a 16:9 frame is half as tall as it is wide, and reads a strip
    rather than a place. *aspect* is width over height.
    """
    across = max(MIN_BOX, float(box))
    down = min(1.0, across * max(aspect, 1e-6))
    left = min(max(x - across / 2, 0.0), 1.0 - across)
    top = min(max(y - down / 2, 0.0), 1.0 - down)
    return left, top, across, down


class PointSurvey:
    """Every reading taken at every point, and what the order of them is.

    Readings rather than sums, unlike :class:`scanny.ui.depth.Survey`: there is
    no pyramid to build here, because the boxes are where a person put them and
    not a division of anything, so nothing is gained by keeping what a coarser
    version of them would have read.
    """

    def __init__(self, points: "list[Point]") -> None:
        self._points = list(points)
        self._samples: "dict[int, np.ndarray]" = {}
        self._passes: "dict[int, int]" = {}

    @property
    def points(self) -> "list[Point]":
        return list(self._points)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def positions(self) -> "tuple[int, ...]":
        return tuple(sorted(self._samples))

    def add(
        self, position: int, readings: "list[float]", pass_number: int = 0
    ) -> None:
        """Record what each point read with focus at *position*.

        The pass it came from is kept with it. Positions from different
        passes are not quite the same coordinate -- each pass parks against
        the stop again, and what the play in the gearing gives back varies --
        and the whole question here is a difference of a few tens of steps.
        Against a modelled lens whose parking landed within forty steps of
        the same place, merging the passes turned a gap of a hundred steps
        into a hundred and fifty.
        """
        if len(readings) != len(self._points):
            raise ValueError("a reading per point, and no more")
        self._samples[int(position)] = np.asarray(readings, dtype=float)
        self._passes[int(position)] = int(pass_number)

    def curve(
        self, pass_number: "int | None" = None
    ) -> "tuple[np.ndarray, np.ndarray]":
        """The positions swept, and ``(samples, points)`` of readings.

        Of one pass, or of all of them together when no pass is named.
        """
        order = [
            position
            for position in self.positions
            if pass_number is None or self._passes.get(position) == pass_number
        ]
        stack = np.zeros((len(order), len(self._points)))
        for index, position in enumerate(order):
            stack[index] = self._samples[position]
        return np.array(order, dtype=float), stack

    @property
    def last_pass(self) -> int:
        return max(self._passes.values()) if self._passes else 0

    def found(self) -> "list[Found]":
        """Each point placed, with the doubt on it, ranked nearest focus first.

        Nearest in *focus travel*, which is nearest to the camera: the datum is
        the near stop, so a smaller number is a shorter subject distance. What
        it cannot say is how much shorter -- steps are not metres and the two
        are not even proportional -- only which and by how many steps.

        **The doubt is the point of this, and its absence was what made the
        first version of it worse than useless.** Whether two things a hundred
        steps apart can be told apart is not a property of this arithmetic. It
        is a property of the lens, of how far the view is magnified, and of
        what happens to be in the boxes: a hundred steps may move the reading
        less than the grain does, and then there is nothing to find. Asked
        anyway, the old code answered with the few steps of noise between two
        curves it could not separate, in whichever order the grain fell -- and
        said it in the same voice it uses for an answer it is sure of.

        So the peak is worked out twice, from the odd samples and from the even
        ones, and how far apart those two answers land is what is reported as
        the doubt. It needs no model of the noise and it measures the thing
        that actually matters: whether reading the same scene again would say
        the same thing.

        Only the last pass's readings are used when it placed as many points as
        the whole survey does. Positions from different passes are not quite
        the same coordinate -- each parks against the stop again, and the play
        in the gearing gives back a little more or less each time.
        """
        if len(self._samples) < 2:
            return [Found(point, None, 0.0, False) for point in self._points]
        where, stack = self.curve(self.last_pass)
        if len(where) < 4 or self._places(where, stack) < self._places(*self.curve()):
            # That pass cannot see as much as everything together can; better a
            # coarse answer than a missing one.
            where, stack = self.curve()
        # Not against the frame: one of a handful of places a person chose
        # being fainter than the others is a fact about the scene, not a reason
        # to disbelieve it. See depth.locate.
        steps, strength, limit, _width = locate(where, stack, against_the_frame=False)
        doubt = self._doubt(where, stack, steps)
        answered = sorted(
            (index for index in range(len(self._points)) if np.isfinite(steps[index])),
            key=lambda index: steps[index],
        )
        places = {index: rank for rank, index in enumerate(answered)}
        nearest = float(steps[answered[0]]) if answered else None
        out: "list[Found]" = []
        for index, point in enumerate(self._points):
            rank = places.get(index)
            here = float(steps[index]) if rank is not None else None
            previous = (
                float(steps[answered[rank - 1]])
                if rank is not None and rank > 0
                else None
            )
            out.append(
                Found(
                    point=point,
                    steps=here,
                    strength=float(strength[index]),
                    at_the_limit=bool(limit[index]),
                    doubt=float(doubt[index]),
                    rank=rank,
                    behind_nearest=(
                        here - nearest
                        if here is not None and nearest is not None
                        else None
                    ),
                    behind_previous=(
                        here - previous
                        if here is not None and previous is not None
                        else None
                    ),
                )
            )
        return out

    @staticmethod
    def _places(where: np.ndarray, stack: np.ndarray) -> int:
        """How many of the points these readings can place at all."""
        if len(where) < 2:
            return 0
        steps, _strength, _limit, _width = locate(where, stack, against_the_frame=False)
        return int(np.isfinite(steps).sum())

    @staticmethod
    def _doubt(
        where: np.ndarray, stack: np.ndarray, steps: np.ndarray
    ) -> np.ndarray:
        """How far out the position of each peak could be, in steps.

        The grain on the readings comes from the readings themselves: on a
        curve that is smooth apart from noise, each sample less the average of
        its two neighbours is noise and nothing else, and the median of those
        is a measure of it that a few wild samples cannot inflate.

        That is then carried through the arithmetic that found the peak. The
        peak is a weighted mean of the positions on the top of the curve, so
        shifting one reading by *e* shifts the answer by ``e`` times that
        sample's leverage -- how far it sits from the middle, over the total
        weight -- and the shifts add in quadrature.

        Splitting the samples in two and comparing the halves was tried first
        and is worth recording as wrong: neighbouring samples of a broad peak
        read almost the same thing, so the two halves are not two looks at the
        curve but very nearly the same look twice. They agree beautifully and
        say nothing about whether the answer is real.
        """
        doubt = np.full(steps.shape, np.inf)
        if len(where) < 5:
            return doubt
        # What the reading wanders by, from the curve itself.
        bumps = stack[1:-1] - 0.5 * (stack[:-2] + stack[2:])
        # A second difference of white noise has one and a half times its
        # variance; the median of the sizes is 0.6745 of a standard deviation.
        grain = np.median(np.abs(bumps), axis=0) / (0.6745 * np.sqrt(1.5))
        peak = stack.max(axis=0)
        floor = stack.min(axis=0)
        line = floor + _TOP * (peak - floor)
        above = np.clip(stack - line, 0.0, None)
        span = _spacing(where).reshape((len(where),) + (1,) * (stack.ndim - 1))
        weight = above * span
        total = weight.sum(axis=0)
        middle = np.divide(
            (weight * where.reshape(span.shape)).sum(axis=0),
            total,
            out=np.zeros_like(total),
            where=total > 0.0,
        )
        leverage = np.sqrt(
            ((span * (where.reshape(span.shape) - middle)) ** 2 * (above > 0)).sum(
                axis=0
            )
        )
        real = (total > 0.0) & np.isfinite(steps)
        np.divide(grain * leverage, total, out=doubt, where=real)
        return np.where(real, doubt, np.inf)

    def interesting(self, margin: int) -> "tuple[int, int] | None":
        """The stretch of travel worth sweeping again, with *margin* either side.

        Everything the points found, plus room either side, because a finer
        pass has to contain every one of them and not just the average of
        them.
        """
        results = self.found()
        placed = [one.steps for one in results if one.known]
        if not placed:
            return None
        where, stack = self.curve()
        _steps, _strength, _limit, width = locate(
            where, stack, against_the_frame=False
        )
        # Never narrower than the peaks themselves are wide. A pass that fits
        # inside the top of a hill is looking at a flat noisy line, and it will
        # answer with the noise: that is how a fine pass makes a coarse one's
        # answer worse instead of better.
        widest = float(np.nanmax(width)) if np.isfinite(width).any() else 0.0
        room = max(int(margin), int(_SHOULDERS * widest))
        return int(np.floor(min(placed))) - room, int(np.ceil(max(placed))) + room

    def escaping(self) -> "tuple[bool, bool]":
        """Whether the last pass ran out of travel before it ran out of peaks.

        Answers ``(rising, falling)``: whether some point's best reading is
        the **last** one taken, and whether some point's best is the
        **first**. A sweep can only place a hill by seeing it fall away on
        both sides, so either of those means the bracket did not contain
        that point rather than that the point is at the end of the world.

        The two are not the same problem and that is why they are answered
        separately. A pass that is still rising can simply take more stops --
        it is already driving that way, and going further costs nothing but
        the stops. A pass whose best reading is its first cannot be helped
        the same way: reaching back means reversing the lens, and a reversal
        takes up the play in the gearing, which is the one quantity none of
        this can measure. That one has to start again from further back.
        """
        where, stack = self.curve(self.last_pass)
        if len(where) < 3:
            return False, False
        best = stack.argmax(axis=0)
        return bool((best >= len(where) - 1).any()), bool((best <= 0).any())


def summarise(
    found: Found,
    number: int,
    others: "list[Found] | None" = None,
    datum: str = "the near stop",
) -> str:
    """What to say about one point, for the tooltip on it.

    *datum* is what a position is counted from, because that is not always the
    same place. A sweep of the whole travel parks against the near stop first
    and counts from there. A sweep bracketed around where autofocus landed
    never parks -- parking is what it exists to avoid -- so its positions are
    counted from where that bracket began, and saying "from the near stop"
    about them would be saying something untrue. Neither datum changes the
    gaps between the points, which are the answer either way.
    """
    if not found.known:
        return (
            f"Point {number}: nothing here has enough contrast to say where it "
            f"comes into focus, so it is left out of the order. Try a bigger "
            f"box, or put the point on something with detail in it."
        )
    lines = [
        f"Point {number}: sharpest at {found.steps:,.0f} steps from "
        f"{datum}, give or take {_doubt(found.doubt)}"
    ]
    same = [
        (index + 1, abs(other.steps - found.steps))
        for index, other in enumerate(others or [])
        if other is not found and other.known and not found.apart_from(other)
    ]
    # The doubt goes before the rank, not after it. Saying "the nearest of
    # them" and then taking it back reads as a quibble about an answer; saying
    # it cannot be told apart, and then where it landed anyway, is the answer.
    if not same:
        if found.rank == 0:
            lines.append("The nearest of the points measured")
        else:
            lines.append(
                f"{found.rank + 1}{_ordinal(found.rank + 1)} nearest: "
                f"{found.behind_nearest:,.0f} steps behind point one"
                + (
                    f", {found.behind_previous:,.0f} behind the one before it"
                    if found.behind_previous is not None
                    else ""
                )
            )
    if same:
        which = ", ".join(str(number) for number, _gap in same)
        widest = max(gap for _number, gap in same)
        lines.append(
            f"Too close to call against point{'' if len(same) == 1 else 's'} "
            f"{which}: the {widest:,.0f} steps between them is inside the "
            f"doubt, so which is nearer is not to be believed. Magnify onto "
            f"the subject and measure again -- a step of focus moves the "
            f"picture far more when the view is magnified, and that is the "
            f"whole of what makes a small gap readable."
        )
    lines.append(f"Reading {found.strength:.0f} at its best")
    if found.at_the_limit:
        lines.append(
            "Its best reading was at the end of what was swept, so its real "
            "peak may lie outside -- treat the number as a bound"
        )
    return "\n".join(lines)


def _doubt(value: float) -> str:
    if not np.isfinite(value):
        return "no telling how much"
    return f"{value:,.0f} step" + ("" if 0.5 <= value < 1.5 else "s")


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")


def ordering(results: "list[Found]") -> str:
    """The whole answer in one line: which point is where, nearest first.

    Points whose gap is inside the doubt on it are joined by ``=`` rather than
    ordered, because between those two the order is whichever way the grain
    fell and saying it as an order is saying something untrue.
    """
    placed = sorted(
        (index for index, found in enumerate(results) if found.known),
        key=lambda index: results[index].steps,
    )
    if not placed:
        return "No point could be placed"
    parts = []
    for spot, index in enumerate(placed):
        behind = results[index].behind_nearest or 0.0
        label = "nearest" if spot == 0 else f"+{behind:,.0f}"
        if spot and not results[index].apart_from(results[placed[spot - 1]]):
            parts.append(f"= {index + 1}: {label}")
        else:
            parts.append(f"{index + 1}: {label}")
    missing = len(results) - len(placed)
    if missing:
        parts.append(f"({missing} unplaced)")
    return "   ".join(parts)
