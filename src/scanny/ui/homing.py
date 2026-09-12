"""Back to the best place on a stretch of walking, through the gearing's play.

Both searches end the same way. A walk in one direction has gone over the top
of the reading and down the far side, and the lens has to go back and stand on
the top. It cannot be driven there by step count: the gearing has play in it,
and the first steps back after a reversal move nothing until the play is taken
up -- how many of them is not known. And it cannot be walked there by watching
for the reading to be as good as the best one again: the best reading of a
walk is the luckiest of its readings, and a walk home that waits to match it
wanders, while one that stops at the first reading that has *gone over the
top* stops wherever the grain on the readings first says so -- which, crossing
the play, is anywhere.

What *can* be done is to treat the stretch walked in one direction as a
profile. Within one direction the steps are honest: the play was taken up at
the start and stayed taken up. So the profile says where on it the reading was
best -- the middle of its top, from every reading on the top rather than the
luckiest one -- and the way home is the same stretch in reverse. The one
number missing is how much play the way home took up before the optics began
to move, and that is *measured*: the readings on the way home are matched
against the profile, and the play is whatever lines them up. Then the rest of
the way is honest travel, an increment at a time, matching again with every
reading, until the best is less than half an increment away.

:class:`WayHome` is that, with no camera in it: it is told each reading on the
way, and says whether to keep going, whether it is home, or whether it has
lost the way -- which the searches using it answer each in their own way.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["WayHome", "best_of", "middle_of_top", "run_around_top"]

#: Where, between the lower end of the stretch and the top of it, the line is
#: drawn that marks out the top whose middle is wanted.
#:
#: High, because the middle of a top is only the best of it where the top is
#: symmetrical, and the average of several regions seldom is: a line half way
#: down took in enough of a lopsided top to pull its middle several steps off
#: the best, and once the walk across had to take in every region's whole
#: curve for their depths, the half-way line took in more still. Across a
#: few hundred made-up scenes, drawing it near the top halved what the
#: compromise gave away. But a parabola needs three readings and the top of
#: a magnified subject can be a single increment wide, so the line comes down
#: until the top holds that many, or as far as it will go.
ON_TOP = (0.85, 0.7, 0.5, 0.25)
TOP_READINGS = 3

#: How far the top has to rise above the ends of the stretch before it is
#: worth going home to, as a fraction of it. Less than that and the shape of
#: it is the grain's, and there is nothing to match the way home against.
TALL_ENOUGH = 0.03

#: How far a reading on the way home has to be from what the far end of the
#: stretch read before it says the optics are moving, as a share of
#: everything the stretch read, top to bottom -- and how many of those it
#: takes before the play is worked out from them. The readings before the play
#: is taken up are all the same, and are no help at all in saying how much of
#: it there was.
#:
#: *From*, not *above*: the stretch may end on another region's hill, and then
#: the way home goes down before it goes up. Waiting for readings above the
#: far end waited through the whole of that valley for evidence that was
#: there from its first step.
#:
#: And one of them is enough. The best may be a single increment from where
#: the stretch ended, and waiting for three walked straight past it. The
#: readings that have not moved yet still count in the fit, and they are
#: what keeps one grainy reading from being taken for the play running out.
_ON_THE_WAY = 0.25
_ENOUGH_ON_THE_WAY = 1

#: How finely the play is worked out, in drive steps. Far finer than an
#: increment, so that rounding it is not where the error comes from.
_PLAY_RESOLUTION = 0.5

#: How many increments of the way home are always walked one at a time,
#: however far there is to go: see :meth:`WayHome.stride`. Two, because the
#: arrival is judged by the reading against the profile, and one increment of
#: warning is one reading of warning.
_CLOSE_IN = 2

#: How far below the top the stretch saw the reading may be where the way
#: home ends, before it is not believed to be there. Well outside the grain;
#: well inside what one increment off the top of a steep hill costs.
_ARRIVED = 0.05


def best_of(where: np.ndarray, read: np.ndarray) -> "float | None":
    """Where along a stretch the reading is best, or None if it cannot say.

    None when the stretch is too short to have a shape, or its top stands so
    little above its ends that the shape is the grain's.
    """
    if len(read) < 5:
        return None
    peak = float(read.max())
    ends = max(float(read[0]), float(read[-1]))
    if peak <= 0.0 or peak - ends < TALL_ENOUGH * peak:
        return None
    for share in ON_TOP:
        level = ends + share * (peak - ends)
        first, last = run_around_top(list(read), level)
        if last - first + 1 >= TOP_READINGS:
            break
    return middle_of_top(list(where), list(read), level)


def run_around_top(read: "list[float]", level: float) -> "tuple[int, int]":
    """The first and last index of the run of readings above *level* that
    holds the highest of them."""
    first = last = int(np.argmax(read))
    while first > 0 and read[first - 1] >= level:
        first -= 1
    while last + 1 < len(read) and read[last + 1] >= level:
        last += 1
    return first, last


def _spacing(where: np.ndarray) -> np.ndarray:
    """How much travel each reading stands for: half way to either neighbour.

    One when there is nothing to compare with, so a single reading or an
    evenly walked stretch is weighted exactly as it was before this existed.
    """
    if len(where) < 2:
        return np.ones(len(where))
    span = np.empty(len(where))
    span[0] = abs(where[1] - where[0])
    span[-1] = abs(where[-1] - where[-2])
    if len(where) > 2:
        span[1:-1] = np.abs(where[2:] - where[:-2]) / 2.0
    average = float(np.mean(span)) or 1.0
    # Relative to the average, so that the weights stay of the same size as
    # the readings' own and the arithmetic below is unchanged by the units.
    return np.maximum(span / average, 1e-9)


def middle_of_top(
    where: "list[int]", read: "list[float]", level: float
) -> "float | None":
    """Where the highest hill of the stretch is highest, from every reading on it.

    Only the run of readings above *level* that holds the highest of them:
    another hill that also clears the line is another answer, and averaging
    the two would put the result in the valley between.

    Over that run, one of two estimators, chosen by how many readings are on
    the top. A top of several readings is found as
    their middle, weighted by how far above the line each stands, which
    divides the grain on them between them -- on a broad top every reading is
    within the grain of the others, and which is highest is the grain's
    choice. A top of three is found as the parabola through them, which is
    the better estimator when the hill is narrow against the step.

    **And weighted by how much travel each reading stands for**, which is one
    where the stretch was walked evenly and is not where it was not. A walk
    told to hurry takes strides through soft focus and single increments near
    the best, so a leg can arrive here with one spacing on one side of its top
    and another on the other -- and a plain middle of such a run is pulled
    towards whichever side was walked finely, because three closely spaced
    readings outvote one stride-spaced reading that stands for the same
    ground. Where the spacing is even this changes nothing at all.
    """
    if not read:
        return None
    first, last = run_around_top(read, level)
    xs = np.array(where[first : last + 1], dtype=float)
    ys = np.array(read[first : last + 1], dtype=float)
    weights = (ys - level) * _spacing(xs)
    total = float(weights.sum())
    centre = float((xs * weights).sum() / total) if total > 0.0 else float(xs.mean())
    if len(xs) != 3 or np.ptp(xs) <= 0.0:
        return centre
    # Fitted about the centre, which keeps the arithmetic well conditioned
    # however far from zero the step counts have wandered.
    curvature, slope, _height = np.polyfit(xs - centre, ys, 2)
    if curvature >= 0.0:
        return centre
    vertex = centre - slope / (2.0 * curvature)
    if not xs.min() <= vertex <= xs.max():
        return centre
    return float(vertex)


class WayHome:
    """Back along one stretch walked in one direction, to the best place on it.

    Made with :meth:`along`, from the stretch as the walk made it: where each
    reading was taken, in step counts, and what it read. The walk home starts
    where the stretch ended, standing on its last reading, and goes the other
    way. Before each increment home, :meth:`drive`; after it, :meth:`heard`
    with what it read there.
    """

    def __init__(
        self, back: np.ndarray, profile: np.ndarray, best_at: float, unit: int
    ) -> None:
        self._unit = abs(int(unit)) or 1
        #: The stretch as the way home meets it: steps back from its far end,
        #: against what was read there.
        self._back = back
        self._profile = profile
        #: How far back from the far end the best of the stretch is.
        self.best_at = float(best_at)
        #: The highest reading on the stretch, which is what arriving home is
        #: checked against.
        self.peak = float(profile.max())
        self._far_end = float(profile[0])
        self._moving = _ON_THE_WAY * (self.peak - self._far_end)
        # Every reading on the way home, against how far it had driven.
        self._way: "list[tuple[float, float]]" = [(0.0, self._far_end)]
        self.travel = 0.0
        #: How far to go before giving up on the way: twice the stretch -- the
        #: play can be as long as the stretch itself -- and a little more.
        self.reach = 2.0 * float(back[-1]) + 4 * self._unit

    @classmethod
    def along(
        cls, where: "Sequence[float]", read: "Sequence[float]", unit: int
    ) -> "WayHome | None":
        """The way home along a stretch, or None if it has no top to go to."""
        where = np.asarray(where, dtype=float)
        read = np.asarray(read, dtype=float)
        best = best_of(where, read)
        if best is None:
            return None
        end = where[-1]
        # Steps back from the far end, which is the way home counts, in the
        # order it will meet them.
        back = np.abs(where - end)[::-1]
        return cls(back, read[::-1], abs(best - end), unit)

    def drive(self, stride: int = 1) -> None:
        """*stride* increments further home are being driven."""
        self.travel += self._unit * max(1, int(stride))

    def stride(self, most: int = 1) -> int:
        """How many increments to drive next, taking at most *most*.

        One, always, unless there is plainly a long way still to go. The way
        home is what lands on the answer, so the end of it is walked an
        increment at a time whatever the caller would like -- but the
        beginning of it need not be. What the beginning is, on a lens with
        play in it, is dead travel: until the play is taken up the optics have
        not moved, every reading says what the far end of the stretch said,
        and there is nothing in any of them to be had by taking them one
        increment apart.

        *left* is what the play worked out so far says there is still to go,
        and it is honest in both halves of the journey: while the optics have
        not moved the fit puts the play at everything driven so far, so left
        stays at the whole distance home; once they are moving it is pinned,
        and left counts down. Two increments are kept in hand so that the
        last of the way, where the reading is matched against the profile
        step by step, is walked the way it always was.
        """
        if most <= 1:
            return 1
        left = self.play() + self.best_at - self.travel
        room = int(left // self._unit) - _CLOSE_IN
        return max(1, min(int(most), room))

    def heard(self, reading: float) -> str:
        """What it read after the last increment: ``on``, ``home`` or ``lost``.

        ``home`` means the play worked out from the readings so far puts the
        best less than half an increment away -- whether the reading here
        agrees is :meth:`agrees`. ``lost`` means it has gone as far as the
        stretch could possibly have been without getting there.
        """
        self._way.append((self.travel, float(reading)))
        moving = sum(
            1 for _t, value in self._way if abs(value - self._far_end) > self._moving
        )
        if moving >= _ENOUGH_ON_THE_WAY:
            left = self.play() + self.best_at - self.travel
            if left < self._unit / 2:
                return "home"
        if self.travel >= self.reach:
            return "lost"
        return "on"

    def agrees(self, reading: float) -> bool:
        """Whether *reading* is as near the top of the stretch as home should be.

        It will not be when the top is a single increment wide and the play
        was placed a step out -- a steep enough hill loses a quarter of its
        reading in one increment -- or when the walk home went past the best
        before it could tell how much play there had been.
        """
        return reading >= (1.0 - _ARRIVED) * self.peak

    def play(self) -> float:
        """How much play was taken up before the way home started moving.

        Whatever lines the readings on the way home up with the profile the
        stretch made. Until the play is taken up the optics are at the far end
        of the stretch, reading what it read there; after it, they are as far
        back along it as the walk has driven, less the play. Every reading
        counts, so the grain on any one of them is shared out.
        """
        travel = np.array([t for t, _value in self._way])
        seen = np.array([value for _t, value in self._way])
        # Largest first, so that of two that fit as well the larger is taken:
        # readings that have not changed say the optics have not moved, and
        # the play that says that is the one that is at least as long as the
        # walk that read them.
        candidates = np.arange(self.travel, -_PLAY_RESOLUTION, -_PLAY_RESOLUTION)
        misses = [
            float(
                np.sum(
                    (
                        seen
                        - np.interp(
                            np.maximum(travel - play, 0.0), self._back, self._profile
                        )
                    )
                    ** 2
                )
            )
            for play in candidates
        ]
        return float(max(candidates[int(np.argmin(misses))], 0.0))
