"""One focus position for several places at once, and what it cost each of them.

Fine tuning (:mod:`scanny.ui.hunt`) answers the question for one rectangle:
where is *this* sharpest. A frame of film is not one rectangle. It curls, it
sags in the carrier, the lens has a curved field -- so the corners and the
middle come into focus at slightly different places, and there is only one
focus position to give them. What is wanted is not how far apart they are but
the position that does best by all of them together, and an honest account of
what each of them gave up for it.

So a calibration is two phases, and the second only means anything because of
the first.

**First, each region's best, by fine tuning on it alone.** The same search the
button runs -- magnified onto the region, the camera's autofocus aimed at it,
then walked one increment at a time by what the picture reads -- once per
region. What it stood on at the end is that region's *peak*: the reading, and
the picture of the region at that moment. Along with where the camera was
pointed when it read it, because a reading is only comparable with another
taken through the same crop at the same magnification: the same piece of
sensor magnified further has its detail spread over more pixels, and reads
differently for it.

**Then one position for all of them.** At every probe the camera is panned to
each region's view in turn -- panning moves the focus point and nothing else,
so every region is read at the one focus position -- and each region is read
against its own peak. Those shares are combined into the one number the search
climbs, in whichever of two ways was asked for (:data:`OBJECTIVES`): their
average, or the worst of them. The search itself is :class:`_Search`, and like
everything else here it drives to no remembered step count.

**Shares of each region's own best, not the readings themselves.** A reading
has no units; its size is set by how much detail is in the box and how bright
it is. Combining raw readings would hand the compromise to whichever region
has the most texture in it, and a faint region would count for nothing. As
shares, each region has the same say: one at its own peak, less as it softens.

**And, on the way, how far apart in focus the regions are.** The search walks
once across every region's peak in a single direction, and within one
direction the steps are honest -- so where on that walk each region peaked is
its depth, in drive steps, against the others. Fitted with a plane, that is
how far the frame is tilted against the sensor and how much of the rest is
curl, which is what levelling the film needs.

What comes out is a :class:`CalibrationReport`: per region, what it read at
its best and at the compromise, the picture of it at both, and its depth --
so a person can look at what was possible and what was chosen, rather than
take a percentage's word for it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage

from ..camera.nikon import LiveViewFrame
from .film import FilmSurface
from .hunt import FineTune, Move
from .orientation import Orientation
from .sharpness import format_reading

__all__ = [
    "MAX_REGIONS",
    "OBJECTIVES",
    "TURN_BACK",
    "TURN_BACK_RANGE",
    "Calibration",
    "CalibrationReport",
    "Look",
    "Region",
    "RegionResult",
    "Tilt",
    "View",
    "average_of_best",
    "clock",
    "colour_for",
    "combine",
    "cut_out",
    "describe_depth",
    "summarise",
]

#: What "the focus that does best by all of them" can be taken to mean, and
#: how each is put to the person choosing.
#:
#: *average* is every region's share of its own best, averaged. It can be
#: bought with one region's sharpness: where the regions are further apart in
#: focus than each is deep, the best average may stand on one region's peak
#: with the others well short of theirs. *worst* is the least of those shares,
#: which no region can be sacrificed to -- it finds where the softest region is
#: as sharp as it can be made, at whatever cost to the sharpest.
OBJECTIVES = {
    "average": "The best average",
    "worst": "The best worst region",
}

#: How many regions may be calibrated at once. Not a limit of the arithmetic
#: but of the time: every probe of the compromise pans to every region and
#: waits for the picture there, so each one adds most of a second to a probe.
MAX_REGIONS = 5

#: How far off the crop recorded for a region a frame's may be and still be
#: the same view, as a fraction of the crop's size. The body reports the crop
#: in whole sensor pixels, and a view put back is not always put back to the
#: pixel.
_SAME_VIEW = 0.01

#: The colours a region is drawn in once there is a report: near its best,
#: noticeably short of it, and well short. And the two before there is one.
_UNMEASURED = "#ebebeb"
_NOTHING = "#8c8c8c"
_GOOD, _FAIR, _POOR = "#50dc78", "#fabe46", "#f05a5a"
_GOOD_ABOVE, _FAIR_ABOVE = 0.95, 0.85


@dataclass(frozen=True)
class Region:
    """A rectangle on the sensor's frame, in fractions of **the whole frame**.

    Fractions of the frame and not of what is on screen, for the reason the
    measured area gives in :meth:`scanny.ui.sharpness.SharpnessMeter.set_area`:
    a region is a place on the subject, and has to stay on it when the view
    magnifies or pans underneath it -- which calibrating it does, to every
    region in turn.
    """

    x: float
    y: float
    w: float
    h: float

    @property
    def rect(self) -> "tuple[float, float, float, float]":
        return (self.x, self.y, self.w, self.h)

    def contains(self, x: float, y: float) -> bool:
        """Whether a place in the frame is inside this region."""
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def seen_in(
        self, crop: "tuple[float, float, float, float]"
    ) -> "tuple[float, float, float, float] | None":
        """Where this lands on a view showing *crop* of the frame, or None.

        Clipped to the view, like
        :meth:`scanny.camera.nikon.LiveViewFrame.area_normalised`, and None
        only when none of it is on screen -- which is the ordinary case for
        all but one region once the view is magnified.
        """
        left, top, width, height = crop
        if width <= 0.0 or height <= 0.0:
            return None
        nx, ny = (self.x - left) / width, (self.y - top) / height
        right = min(nx + self.w / width, 1.0)
        bottom = min(ny + self.h / height, 1.0)
        nx, ny = max(nx, 0.0), max(ny, 0.0)
        if right <= nx or bottom <= ny:
            return None
        return (nx, ny, right - nx, bottom - ny)


@dataclass(frozen=True)
class View:
    """Where the camera was pointed when a region's peak was read.

    Kept so that every later reading of the region is taken through the same
    crop: the zoom level and the focus point are what the body is told, and
    the crop is what it answered with, which is what a frame is checked
    against before it is believed to be showing this view again.
    """

    level: int
    af: "tuple[int, int]"
    #: Centre x, centre y, width, height, in sensor pixels.
    crop: "tuple[int, int, int, int]"

    @classmethod
    def of(cls, frame: LiveViewFrame, level: int) -> "View":
        return cls(int(level), (frame.af_x, frame.af_y), _crop_of(frame))

    def shows(self, frame: LiveViewFrame) -> bool:
        """Whether *frame* is a picture through this view."""
        cx, cy, w, h = _crop_of(frame)
        want_x, want_y, want_w, want_h = self.crop
        if (w, h) != (want_w, want_h):
            return False
        slack = _SAME_VIEW * max(w, h, 1)
        return abs(cx - want_x) <= slack and abs(cy - want_y) <= slack


def _crop_of(frame: LiveViewFrame) -> "tuple[int, int, int, int]":
    return (
        frame.crop_center_x,
        frame.crop_center_y,
        frame.crop_width or frame.image_width,
        frame.crop_height or frame.image_height,
    )


@dataclass(frozen=True)
class Look:
    """One reading of one region, and the picture it was read off.

    The picture is the region cut out of the displayed image, in the frame's
    own orientation and tones. Turning, mirroring and inverting it is left to
    whoever shows it, so that it is shown the way the view is set *then*
    rather than the way it was set while calibrating.
    """

    reading: float
    picture: "QImage | None" = field(default=None, compare=False, repr=False)


def cut_out(
    image: QImage, rect: "tuple[float, float, float, float]"
) -> "QImage | None":
    """The part of *image* that *rect* covers, in fractions of it, as its own image."""
    if image.isNull():
        return None
    width, height = image.width(), image.height()
    x, y, w, h = rect
    left = min(max(int(round(x * width)), 0), width - 1)
    top = min(max(int(round(y * height)), 0), height - 1)
    right = min(max(int(round((x + w) * width)), left + 1), width)
    bottom = min(max(int(round((y + h) * height)), top + 1), height)
    return image.copy(QRect(left, top, right - left, bottom - top))


def average_of_best(readings: Sequence[float], peaks: Sequence[float]) -> float:
    """How near its own best each region is, on average, as a fraction.

    Regions with no peak to be measured against are left out: they had
    nothing in them to focus on, and no focus position is better or worse for
    them.
    """
    shares = [
        reading / peak for reading, peak in zip(readings, peaks) if peak > 0.0
    ]
    return sum(shares) / len(shares) if shares else 0.0


def combine(shares: Sequence[float], objective: str = "average") -> float:
    """The regions' shares of their own best, made into the one number climbed.

    Both ways of doing it only ever go up when any one share goes up, which
    is what lets :meth:`_Search._past` bound what is still to be found ahead
    the same way for either. See :data:`OBJECTIVES`.
    """
    if not shares:
        return 0.0
    if objective == "worst":
        return float(min(shares))
    return float(sum(shares)) / len(shares)


@dataclass(frozen=True)
class RegionResult:
    """What a calibration made of one region."""

    number: int
    region: Region
    #: What it read when fine tuned on its own, or None if it was never tuned.
    best: "Look | None"
    #: How that fine tune ended; see :attr:`scanny.ui.hunt.FineTune.outcome`.
    outcome: str
    #: What it read at the compromise, or None if there is no compromise.
    compromise: "Look | None"
    #: Where it came into focus, in drive steps further than the nearest of
    #: the regions, or None if the walk across did not see its peak. Positive
    #: is the way the drive calls further; see :meth:`CalibrationReport.tilt`.
    depth: "float | None" = None
    #: How far out that could be, in steps: one standard deviation.
    doubt: float = float("inf")
    #: Whether its best reading on the walk across was at an end of it, which
    #: makes the depth a bound rather than an answer.
    edge: bool = False

    @property
    def usable(self) -> bool:
        """Whether it had a peak to be measured against."""
        return self.best is not None and self.best.reading > 0.0

    @property
    def fraction(self) -> "float | None":
        """How near its best the compromise left it: 1 is its best."""
        if not self.usable or self.compromise is None:
            return None
        return self.compromise.reading / self.best.reading


@dataclass(frozen=True)
class Tilt:
    """How the film's edges lean, and how far its middle stands off them.

    The edges are the part levelling corrects: the holder grips the film along
    them, so tipping the holder tips them. The bulge is the part it cannot --
    the film bowing between its edges. See :mod:`scanny.ui.film` for how the
    two are told apart.

    In drive steps from one edge of the picture to the other, in the picture's
    directions *as it is shown* -- turned and mirrored as the view has it --
    because that is how whoever levels the film is looking at it.
    """

    #: How much further the right edge focuses than the left.
    across: float
    #: How much further the bottom edge focuses than the top.
    down: float
    #: One standard deviation on each of those, from the doubt on the depths.
    across_doubt: float
    down_doubt: float
    #: How far the film's middle stands off the plane of its edges at the most:
    #: what levelling cannot take out. Positive is further from the camera.
    bulge: float = 0.0
    #: How many regions the shape was fitted through.
    regions: int = 3

    @property
    def off_the_plane(self) -> float:
        return abs(self.bulge)

    def describe(self) -> str:
        if self.regions <= 3:
            # Three points always make a plane, so nothing is left over to
            # say how much of it is bulge.
            rest = (
                "Three regions always lie on one plane: draw a fourth -- one in "
                "the middle says most -- to see how far the film bows between "
                "its edges"
            )
        elif abs(self.bulge) < 0.5:
            rest = "Between its edges the film lies flat, as far as can be told"
        else:
            way = "further from the camera" if self.bulge > 0 else "nearer the camera"
            rest = (
                f"Between its edges the film bows up to {abs(self.bulge):.0f} "
                f"steps {way} than they are, which levelling cannot take out"
            )
        return "\n".join(
            [
                _lean(self.across, self.across_doubt, "right edge", "left edge"),
                _lean(self.down, self.down_doubt, "bottom edge", "top edge"),
                rest,
            ]
        )


def _lean(steps: float, doubt: float, far_side: str, near_side: str) -> str:
    """One line of a tilt: which side focuses further, by how much."""
    plus = _give_or_take(doubt)
    if abs(steps) < 0.5 or (np.isfinite(doubt) and abs(steps) <= 2.0 * doubt):
        return (
            f"The {far_side} and the {near_side} focus within {abs(steps):.0f} "
            f"steps of each other{plus} -- level, as far as can be told"
        )
    if steps < 0:
        far_side, near_side = near_side, far_side
    return f"The {far_side} focuses {abs(steps):.0f} steps further than the {near_side}{plus}"


def _duration(seconds: float) -> str:
    """A length of time the way a clock on the wall would say it."""
    whole = int(round(seconds))
    minutes, rest = divmod(whole, 60)
    if not minutes:
        return f"{rest} s"
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes} min"
    return f"{minutes} min {rest} s"


def clock(seconds: float) -> str:
    """Elapsed time as a running clock reads it: 4:07, or 1:02:33."""
    whole = max(0, int(seconds))
    minutes, rest = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{rest:02d}" if hours else f"{minutes}:{rest:02d}"


def _give_or_take(doubt: float) -> str:
    return f", give or take {doubt:.0f}" if np.isfinite(doubt) else ""


@dataclass(frozen=True)
class CalibrationReport:
    """Every region's best, and what the one focus position left each of them."""

    results: "tuple[RegionResult, ...]"
    #: What the compromise achieved, combined the way :attr:`objective` says,
    #: or None without one.
    score: "float | None"
    #: How the compromise walk ended -- ``found``, ``exhausted`` or ``lost``,
    #: as a fine tune's does -- or ``single`` or ``nothing`` when there were
    #: fewer than two regions to compromise between, or ``stopped``.
    outcome: str
    probes: int = 0
    #: Which of :data:`OBJECTIVES` the compromise was sought by.
    objective: str = "average"
    #: How long the whole calibration took, in seconds.
    seconds: float = 0.0
    #: How many readings each region's fine tune took, in order.
    tune_probes: "tuple[int, ...]" = ()
    #: How many times the camera's own autofocus was run.
    autofocuses: int = 0
    #: How many times focus was driven, and how many drive steps that came
    #: to, the fine tunes and the compromise together.
    moves: int = 0
    travel: int = 0
    #: The compromise as it went: at every probe, what the search was doing
    #: (``out``, ``across``, ``home`` or ``climb``), every usable region's
    #: share of its best in the order of :attr:`history_regions`, and the one
    #: number made of them.
    history: "tuple[tuple[str, tuple[float, ...], float], ...]" = ()
    #: Which regions, by number, the shares in :attr:`history` belong to.
    history_regions: "tuple[int, ...]" = ()
    #: Whether the calibration was asked to walk far enough to place every
    #: region's peak. Without it there are no depths, rather than some.
    depths_measured: bool = False
    #: When it began, in seconds since the epoch, or 0 if not known.
    began: float = 0.0
    #: Everything that was said while it ran, as the activity log has it.
    log: "tuple[str, ...]" = ()

    @property
    def stopped(self) -> bool:
        return self.outcome == "stopped"

    def cost(self) -> str:
        """What the calibration took, in time and in driving."""
        if self.seconds <= 0.0:
            return ""
        parts = [f"Took {_duration(self.seconds)}"]
        if self.tune_probes:
            each = ", ".join(str(count) for count in self.tune_probes)
            parts.append(f"fine tuning {each} probes")
        if self.probes:
            parts.append(f"the compromise {self.probes}")
        line = parts[0] + (": " + ", ".join(parts[1:]) if len(parts) > 1 else "")
        driving = f"{self.moves:,} focus moves covering {self.travel:,} drive steps"
        if self.autofocuses:
            driving += (
                f", and {self.autofocuses} autofocus"
                + ("" if self.autofocuses == 1 else "es")
            )
        return f"{line}; {driving}"

    @property
    def worst(self) -> "RegionResult | None":
        """The region the compromise cost the most."""
        placed = [one for one in self.results if one.fraction is not None]
        return min(placed, key=lambda one: one.fraction) if placed else None

    @property
    def average(self) -> "float | None":
        """The average share of their best the regions have at the compromise."""
        placed = [one.fraction for one in self.results if one.fraction is not None]
        return sum(placed) / len(placed) if placed else None

    # -- depth -------------------------------------------------------------

    @property
    def placed(self) -> "list[RegionResult]":
        """The regions whose depth is known, nearest first."""
        return sorted(
            (one for one in self.results if one.depth is not None),
            key=lambda one: one.depth,
        )

    def ordering(self) -> str:
        """The regions by depth in one line: nearest first, then how much further."""
        placed = self.placed
        if len(placed) < 2:
            return ""
        parts = [f"{placed[0].number} nearest"]
        for one in placed[1:]:
            mark = "≥" if one.edge else "+"
            parts.append(
                f"{one.number} {mark}{one.depth:.0f}{_plus_minus(one.doubt)}"
            )
        return "Depth, in drive steps: " + ",  ".join(parts)

    def surface(self) -> "FilmSurface | None":
        """The film's shape through the regions' depths; see :mod:`scanny.ui.film`.

        A region whose peak was at an end of the walk across has a depth that
        is only a bound, and a bound pinned into a shape bends it the wrong
        way, so those are left out of it.
        """
        placed = [one for one in self.placed if not one.edge]
        if len(placed) < 3:
            return None
        return FilmSurface.fit(
            [
                (one.region.x + one.region.w / 2, one.region.y + one.region.h / 2)
                for one in placed
            ],
            [one.depth for one in placed],
            [one.doubt if np.isfinite(one.doubt) else 1e3 for one in placed],
        )

    def tilt(self, orientation: "Orientation | None" = None) -> "Tilt | None":
        """How the film's edges lean and how far it bows, as the picture is shown.

        A plane needs three regions that are not in a line; with fewer there
        is a difference between two of them and no telling which way the rest
        of the frame leans, so there is no answer rather than a guess. The
        doubt on each depth is carried into the lean, so a lean inside it is
        said to be level.

        With four regions or more, the lean is the *edges'*, not a plane's
        through all of the regions. A region in the middle of a bowed frame
        stands off any plane the others make, and a plane pulled towards it
        leans the wrong amount: the holder grips the film by its edges, and
        it is their plane that levelling tips.

        Further and nearer are the drive's own words for its two directions,
        and :mod:`scanny.camera.nikon` could not confirm from live view which
        of them a lens actually turns: before levelling by this, drive focus a
        few steps *further* and check that the far edge is what sharpens.
        """
        surface = self.surface()
        if surface is None:
            return None
        across, down, across_doubt, down_doubt = surface.lean(orientation)
        bulge, _x, _y = surface.bulge()
        return Tilt(
            across=across,
            down=down,
            across_doubt=across_doubt,
            down_doubt=down_doubt,
            bulge=bulge,
            regions=len([one for one in self.placed if not one.edge]),
        )

    def describe(self) -> str:
        """The whole answer in one line."""
        tuned = sum(1 for one in self.results if one.best is not None)
        if self.stopped:
            return (
                f"Calibration stopped with {tuned} of {len(self.results)} "
                f"regions tuned and no compromise found"
            )
        usable = [one for one in self.results if one.usable]
        if not usable:
            return (
                "None of the regions had anything in them to focus on: draw "
                "them round something with detail in it"
            )
        if len(usable) == 1 or self.score is None:
            return (
                f"Only region {usable[0].number} had anything in it to focus "
                f"on, so there was nothing to find a compromise between"
            )
        worst = self.worst
        average = self.average
        if self.objective == "worst" and worst is not None:
            line = (
                f"Compromise: every region at least {worst.fraction:.0%} of its "
                f"best (region {worst.number} is the softest)"
            )
            if average is not None:
                line += f"; {average:.0%} on average"
        else:
            line = f"Compromise: {self.score:.0%} of each region's best on average"
            if worst is not None:
                line += (
                    f"; region {worst.number} gives up most, at {worst.fraction:.0%}"
                )
        if self.outcome == "lost":
            line += (
                ". The average was too unsteady to walk by -- integrate more "
                "frames and calibrate again"
            )
        elif self.outcome == "exhausted":
            line += ". It stopped at the most probes it will take"
        return line


def summarise(result: "RegionResult | None", number: int) -> str:
    """What to say about one region, for the tooltip on it."""
    if result is None:
        return (
            f"Region {number}: not calibrated yet.\n"
            f"Press Calibrate, or ctrl-click inside it to remove it."
        )
    lines = [f"Region {number}"]
    if result.best is None:
        lines.append("Not reached before the calibration stopped")
        return "\n".join(lines)
    if not result.usable:
        lines.append(
            "Nothing in it to focus on, so it was left out of the compromise. "
            "Draw it round something with detail in it."
        )
        return "\n".join(lines)
    lines.append(
        f"Best: {format_reading(result.best.reading)} when fine tuned on its own"
        + _HOW.get(result.outcome, "")
    )
    if result.fraction is not None:
        lines.append(
            f"At the compromise: {format_reading(result.compromise.reading)}, "
            f"{result.fraction:.0%} of its best"
        )
    if result.depth is not None:
        lines.append(describe_depth(result))
    return "\n".join(lines)


def describe_depth(result: RegionResult) -> str:
    """Where a region comes into focus against the nearest of them, in words."""
    if result.depth is None:
        return "Depth: its peak was not on the walk across, so not measured"
    plus = _give_or_take(result.doubt)
    if result.depth < 0.5:
        return f"Depth: the nearest of the regions{plus}"
    bound = " at least" if result.edge else ""
    return (
        f"Depth:{bound} {result.depth:.0f} steps further than the nearest "
        f"region{plus}"
    )


def _plus_minus(doubt: float) -> str:
    return f" ±{doubt:.0f}" if np.isfinite(doubt) else ""


#: What each way a fine tune ends says about the best it found.
_HOW = {
    "restored": " (nothing bettered the camera's own autofocus)",
    "exhausted": " (fine tuning ran out of probes)",
    "lost": " (the reading was too unsteady to walk by)",
}


def colour_for(result: "RegionResult | None") -> str:
    """How to draw a region: by how near its best the compromise left it."""
    if result is None or result.best is None:
        return _UNMEASURED
    if not result.usable:
        return _NOTHING
    fraction = result.fraction
    if fraction is None:
        return _UNMEASURED
    if fraction >= _GOOD_ABOVE:
        return _GOOD
    return _FAIR if fraction >= _FAIR_ABOVE else _POOR


# -- the search for the compromise -------------------------------------------

#: Where a walk turns back, by default: once the number the search climbs has
#: fallen below this share of the best it reached on the walk. Settable
#: before a calibration, as **Turn back below**. Strictly: the only things a
#: walk goes on past it for are a region visibly climbing (see
#: :data:`_CLIMB_PATIENCE`) and depths, when they are asked for.
#:
#: It was a bound once -- walk on while anything still ahead could in
#: principle beat the best so far -- and that was right in theory and a waste
#: of minutes in practice: it walked, and walked back, through stretches where
#: the softest region was at a third of its best, which no compromise is ever
#: going to be found in. A reading four fifths of the best is already plainly
#: the wrong way. Set lower, the walks reach further, which is what regions
#: very far apart in focus need -- their average can have a hill in the middle
#: that is only reached through a valley.
TURN_BACK = 0.8

#: The least and most the turn-back share may be set to.
TURN_BACK_RANGE = (0.3, 0.95)

#: The most increments one leg may take, whatever the regions are doing.
_LONGEST_LEG = 150

#: How near its own best a region has to have come on the walk across for its
#: peak to be counted as on it. Not all the way: the best a fine tune stood on
#: is its luckiest reading, and the walk across steps past the peak in whole
#: increments rather than landing on it.
_NEAR_ITS_BEST = 0.7

#: How many increments further a walk may go, once its number has fallen
#: below the turn-back share, for a region visibly climbing towards its own
#: best -- one just beyond a valley between two regions' hills, whose hill
#: may serve every region better than the one behind. A handful, and no
#: more.
_CLIMB_PATIENCE = 10

#: What *visibly climbing* is: a region at the best it has read on this walk
#: that has risen by this share of its own best over the last few readings.
#: A region out in the flat tail of its focus creeps up by less than that,
#: and is not walked through the soft stretch for.
_CLIMBING = 0.03
_CLIMBING_OVER = 3

#: When depths are asked for, how far below its best on a walk every region
#: has to fall -- on both sides of its peak -- before the walk has enough of
#: its curve to place the peak by. And how many increments past the turn-back
#: point a walk may go for that. A quarter, because across made-up scenes a
#: sixth left the worst depth nearly five steps out and a quarter two, for a
#: few more probes. Asked for separately, because it is the expensive part:
#: a broad region takes a long way to fall a quarter, and all
#: that way a sharp one is softening, which is walking the compromise does
#: not need.
_DEPTH_FALL = 0.25
_DEPTH_PATIENCE = 25

#: Where, between the lower end of the leg across and the top of it, the
#: line is drawn that marks out the top whose middle is wanted.
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
_ON_TOP = (0.85, 0.7, 0.5, 0.25)
_TOP_READINGS = 3

#: How far a reading on the way home has to be from what the far end of the
#: leg across read before it says the optics are moving, as a share of
#: everything the leg read, top to bottom -- and how many of those it takes
#: before the play is worked out from them. The readings before the play is
#: taken up are all the same, and are no help at all in saying how much of it
#: there was.
#:
#: *From*, not *above*: the leg may end on another region's hill, and then
#: the way home goes down before it goes up. Waiting for readings above the
#: far end waited through the whole of that valley for evidence that was
#: there from its first step.
#:
#: And one of them is enough. The best may be a single increment from where
#: the leg ended, and waiting for three walked straight past it. The
#: readings that have not moved yet still count in the fit, and they are
#: what keeps one grainy reading from being taken for the play running out.
_ON_THE_WAY = 0.25
_ENOUGH_ON_THE_WAY = 1

#: How many times the walk home may turn round and go home again, from the
#: profile the last walk home made, before giving the search to a climb.
_HOMECOMINGS = 2

#: How finely the play is worked out, in drive steps. Far finer than an
#: increment, so that rounding it is not where the error comes from.
_PLAY_RESOLUTION = 0.5

#: How far below the top the leg across saw the reading may be where the
#: walk home ends, before it is not believed to be there. Well outside the
#: grain on an average; well inside what one increment off the top of a
#: steep hill costs.
_ARRIVED = 0.05

#: How far the top has to rise above the ends of the leg before it is worth
#: going home to, as a fraction of it. Less than that and the shape of it is
#: the grain's, and there is nothing to match the way home against.
_TALL_ENOUGH = 0.03


class _Search:
    """Across every region's peak in one direction, then back to the best of it.

    Why not :class:`~scanny.ui.hunt.FineTune` on the average, which is what
    this was to begin with: two things about the average that are not true of
    one region's reading, and each of them cost the answer something real.

    **It can have more than one hill.** A region's reading against focus has
    one peak. The average of several has one per region wherever the regions
    are further apart in focus than each is deep -- which magnified onto the
    curl of a frame of film they easily are -- and a climb finds the hill it
    starts on. Started, as it has to be, on the last region's own peak, it
    stands there and calls that the compromise, when the hill in the middle
    serves all of them better.

    **Its top is flat.** Several peaks side by side add up to a broad top whose
    readings are within a per cent of each other over many steps, and a walk
    home that stops at the first reading within a hair of the best meets the
    edge of that, not the middle. An average of 0.63 against a best of 0.64
    there hid one region at 85% and the other at 42%, where the middle had
    them both at 64%.

    So there are three parts to it, and not one of them counts a step across
    a reversal -- which is the one thing nothing here does.

    - **Out**: walk one way until the number being climbed has fallen below
      the turn-back share of the best the walk reached; see :meth:`_past`.
    - **Across**: turn round and walk back until the same is true the other
      way. That one leg crosses the peak of every region in a single
      direction, and within one direction the steps are honest: the play in
      the gearing was taken up at the start of the leg and stayed taken up.
      So it is a true profile of the average against focus, and it says
      where on it the average is best -- the best of all of it, not the
      nearest.
    - **Home**: turn round again. The way home retraces the leg across, but
      how much play was taken up first is not known, and that is the one
      number the walk needs. So it is *measured*, the way everything else
      here is: the readings on the way home are matched against the profile
      the leg across made, and the play is whatever lines them up. Then the
      rest of the way to the best is honest travel, and the walk goes on an
      increment at a time, matching again with every reading, until the best
      is less than half an increment away.

    If the way home loses the top -- the scene moved, or the readings would
    not hold still -- it is not the end of the search: a
    :class:`~scanny.ui.hunt.FineTune` takes over from wherever it is and
    climbs the hill it is on, which is what it is good at.
    """

    def __init__(
        self,
        unit: int,
        objective: str = "average",
        *,
        longest: int = _LONGEST_LEG,
        turn_back: float = TURN_BACK,
        depths: bool = False,
    ) -> None:
        self._unit = abs(int(unit)) or 1
        self._objective = objective
        #: Whether the walks go on far enough to place every region's peak.
        self._depths = bool(depths)
        self._longest = max(4, int(longest))
        self._turn_back = min(max(float(turn_back), TURN_BACK_RANGE[0]), TURN_BACK_RANGE[1])
        self._direction = 1
        self._state = "out"
        # Increments walked since the leg's number fell below the turn-back
        # share, waiting for a region still climbing.
        self._waited = 0
        #: The walk across, as every region's share against its position in
        #: steps -- the only record of all their peaks in one direction, and
        #: so the only one their depths can be read from.
        self.survey: "tuple[np.ndarray, np.ndarray] | None" = None
        #: Step counts, for saying where a reading was within a leg. Never
        #: driven to, and never compared across a reversal.
        self._position = 0
        self._leg: "list[tuple[int, float, tuple[float, ...]]]" = []
        self._leg_best: "list[float]" = []
        self._leg_top = 0.0
        # The profile the leg across made, as steps back from its far end
        # against the average there; where on it the best is; and how high a
        # reading on the way home has to be to say the optics are moving.
        self._profile: "tuple[np.ndarray, np.ndarray]" = (np.zeros(1), np.zeros(1))
        self._best_at = 0.0
        self._peak = 0.0
        self._far_end = 0.0
        self._moving = 0.0
        # Every reading on the way home, against how far it had driven.
        self._way_home: "list[tuple[float, float]]" = []
        self._travel = 0.0
        self._reach = 0.0
        self._homecomings = _HOMECOMINGS
        self._climb: "FineTune | None" = None
        self.probes = 0
        self.best = 0.0
        self.outcome = ""

    @property
    def state(self) -> str:
        """``out``, ``across``, ``home``, ``climb`` or ``done``."""
        return self._state

    @property
    def step_size(self) -> int:
        return self._unit

    @property
    def done(self) -> bool:
        return self._state == "done"

    def step(self, shares: "Sequence[float]") -> "Move | None":
        """Every region's share of its own peak, here; answer the next move."""
        if self._state == "done":
            return None
        self.probes += 1
        score = combine(shares, self._objective)
        self.best = max(self.best, score)
        if self._state == "home":
            return self._home(score)
        if self._state == "climb":
            return self._climbing(score)
        return self._walk(score, tuple(float(share) for share in shares))

    # -- out and across ----------------------------------------------------

    def _walk(self, score: float, shares: "tuple[float, ...]") -> "Move | None":
        self._leg.append((self._position, score, shares))
        if len(self._leg) == 1:
            self._leg_best, self._leg_top = list(shares), score
        else:
            self._leg_best = [max(a, b) for a, b in zip(self._leg_best, shares)]
            self._leg_top = max(self._leg_top, score)
        if not (self._past(score, shares) or len(self._leg) > self._longest):
            return self._drive()
        if self._state == "out":
            # The leg across begins where this one ended, with this reading.
            self._state = "across"
            self._direction = -self._direction
            self._leg = [self._leg[-1]]
            self._leg_best, self._leg_top = list(shares), score
            self._waited = 0
            return self._drive()
        self.survey = (
            np.array([position for position, _s, _r in self._leg], dtype=float),
            np.array([seen for _p, _s, seen in self._leg], dtype=float),
        )
        return self._turn_for_home(score)

    def _past(self, score: float, shares: "tuple[float, ...]") -> bool:
        """Whether this walk has gone far enough the way it is going.

        Far enough is when the number the search climbs has fallen below the
        turn-back share of the best it reached on this walk: it is plainly
        going the wrong way, and the walk turns there. That is the whole of
        the rule unless one of two things is asked of the walk.

        A region **visibly climbing** is waited for, a few increments at most
        (:data:`_CLIMB_PATIENCE`): one just beyond a valley between two
        regions' hills, whose hill may serve every region better than the one
        behind. A walk that turned back at the bottom of the valley would
        never know it was there.

        And when **depths** are asked for, the walk goes on until every region
        has fallen a quarter below its best on it -- far enough down both sides
        of each peak to place the peak by -- for up to :data:`_DEPTH_PATIENCE`
        increments. This is the part that used to happen whether it was asked
        for or not, and it is why a walk told to turn back at four fifths went
        on to where a sharp region was at a third of its best: the broad ones
        around it had barely begun to fall.
        """
        if len(self._leg) < 2 or score >= self._turn_back * self._leg_top:
            self._waited = 0
            return False
        self._waited += 1
        if self._depths:
            if all(self._seen(share, best) for share, best in zip(shares, self._leg_best)):
                return True
            return self._waited > _DEPTH_PATIENCE
        if self._rising(shares):
            return self._waited > _CLIMB_PATIENCE
        return True

    def _seen(self, share: float, best: float) -> bool:
        """Whether this walk has gone far enough past a region to place its peak.

        Past its peak and a quarter down its far side. On the walk across, the
        region has to have come near its best as well: that is the walk its
        depth is read from, and a region that never came up on it has no peak
        on it to read.
        """
        fallen = share < (1.0 - _DEPTH_FALL) * best
        if self._state == "across":
            return fallen and best >= _NEAR_ITS_BEST
        return fallen

    def _rising(self, shares: "tuple[float, ...]") -> bool:
        """Whether some region is visibly on its way up to its own best."""
        recent = [seen for _p, _s, seen in self._leg[-_CLIMBING_OVER - 1 :]]
        for region, (share, best) in enumerate(zip(shares, self._leg_best)):
            lowest = min(seen[region] for seen in recent)
            if share >= best - 0.01 and share - lowest >= _CLIMBING:
                return True
        return False

    def _turn_for_home(self, score: float) -> "Move | None":
        """Make a profile of the leg across, find its best, and start back."""
        where = np.array([position for position, _s, _r in self._leg], dtype=float)
        read = np.array([value for _p, value, _r in self._leg], dtype=float)
        best = _best_of(where, read)
        if best is None:
            return self._start_climb(score)
        end = where[-1]
        # Steps back from the far end, which is the way the walk home counts,
        # in the order it will meet them.
        back = np.abs(where - end)[::-1]
        self._profile = (back, read[::-1])
        self._best_at = abs(best - end)
        self._peak = float(read.max())
        self._far_end = float(read[-1])
        self._moving = _ON_THE_WAY * (self._peak - float(read.min()))
        self._way_home = [(0.0, score)]
        self._travel = 0.0
        self._reach = 2.0 * float(back[-1]) + 4 * self._unit
        self._state = "home"
        self._direction = -self._direction
        # The walk home is a leg of its own, and if it has to it will be the
        # profile for another way home; see _arrived.
        self._leg = [(self._position, score, ())]
        return self._drive_home()

    # -- home --------------------------------------------------------------

    def _home(self, score: float) -> "Move | None":
        self._way_home.append((self._travel, score))
        self._leg.append((self._position, score, ()))
        moving = sum(
            1 for _t, value in self._way_home
            if abs(value - self._far_end) > self._moving
        )
        if moving >= _ENOUGH_ON_THE_WAY:
            left = self._play() + self._best_at - self._travel
            if left < self._unit / 2:
                return self._arrived(score)
        if self._travel >= self._reach:
            return self._start_climb(score)
        return self._drive_home()

    def _play(self) -> float:
        """How much play was taken up before the way home started moving.

        Whatever lines the readings on the way home up with the profile the
        leg across made. Until the play is taken up the optics are at the far
        end of that leg, reading what it read there; after it, they are as
        far back along it as the walk has driven, less the play. Every
        reading counts, so the grain on any one of them is shared out.
        """
        back, profile = self._profile
        travel = np.array([t for t, _value in self._way_home])
        seen = np.array([value for _t, value in self._way_home])
        # Largest first, so that of two that fit as well the larger is taken:
        # readings that have not changed say the optics have not moved, and
        # the play that says that is the one that is at least as long as the
        # walk that read them.
        candidates = np.arange(self._travel, -_PLAY_RESOLUTION, -_PLAY_RESOLUTION)
        misses = [
            float(
                np.sum(
                    (seen - np.interp(np.maximum(travel - play, 0.0), back, profile))
                    ** 2
                )
            )
            for play in candidates
        ]
        return float(max(candidates[int(np.argmin(misses))], 0.0))

    def _arrived(self, score: float) -> "Move | None":
        """Home, if the reading here agrees; otherwise, go home again.

        It will not agree when the top is a single increment wide and the
        play was placed a step out -- a steep enough hill loses a quarter of
        its reading in one increment -- or when the walk home went past the
        best before it could tell how much play there had been.

        The walk home is an honest leg in one direction once its play was
        taken up, and it has usually crossed the best by then: so it is a
        profile as good as the leg across was, and the way home from it is
        the same arithmetic again. Only when that has been tried and still
        does not agree -- the scene moved, or the readings will not hold
        still -- is the search handed to a climb, which is what a fine tune
        does best and which climbs whatever hill it is next to.
        """
        if score >= (1.0 - _ARRIVED) * self._peak:
            return self._finish("found")
        if self._homecomings > 0 and len(self._leg) >= 5:
            self._homecomings -= 1
            return self._turn_for_home(score)
        return self._start_climb(score)

    def _drive_home(self) -> Move:
        self._travel += self._unit
        return self._drive()

    # -- when the way home is lost -----------------------------------------

    def _start_climb(self, score: float) -> "Move | None":
        self._state = "climb"
        self._climb = FineTune(self._unit, autofocus=False)
        return self._climbing(score)

    def _climbing(self, score: float) -> "Move | None":
        assert self._climb is not None
        move = self._climb.step(score)
        if move is None:
            return self._finish(self._climb.outcome or "found")
        self._position += move.steps
        return move

    # -- driving -----------------------------------------------------------

    def _drive(self) -> Move:
        steps = self._direction * self._unit
        self._position += steps
        return Move(steps=steps)

    def _finish(self, outcome: str) -> None:
        self._state = "done"
        self.outcome = outcome
        return None


def _best_of(where: np.ndarray, read: np.ndarray) -> "float | None":
    """Where along a leg across the average is best, or None if it cannot say.

    None when the leg is too short to have a shape, or its top stands so
    little above its ends that the shape is the grain's.
    """
    if len(read) < 5:
        return None
    peak = float(read.max())
    ends = max(float(read[0]), float(read[-1]))
    if peak <= 0.0 or peak - ends < _TALL_ENOUGH * peak:
        return None
    for share in _ON_TOP:
        level = ends + share * (peak - ends)
        first, last = _run_around_top(list(read), level)
        if last - first + 1 >= _TOP_READINGS:
            break
    return _middle_of_top(list(where), list(read), level)


def _run_around_top(read: "list[float]", level: float) -> "tuple[int, int]":
    """The first and last index of the run of readings above *level* that
    holds the highest of them."""
    first = last = int(np.argmax(read))
    while first > 0 and read[first - 1] >= level:
        first -= 1
    while last + 1 < len(read) and read[last + 1] >= level:
        last += 1
    return first, last


def _middle_of_top(
    where: "list[int]", read: "list[float]", level: float
) -> "float | None":
    """Where the highest hill of the leg is highest, from every reading on it.

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
    """
    if not read:
        return None
    first, last = _run_around_top(read, level)
    xs = np.array(where[first : last + 1], dtype=float)
    ys = np.array(read[first : last + 1], dtype=float)
    weights = ys - level
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


# -- the calibration as a whole ----------------------------------------------


class Calibration:
    """Where a calibration has got to, and everything it has found on the way.

    No camera in it, in the same way :class:`~scanny.ui.hunt.FineTune` has
    none. The worker runs the fine tunes and does the panning and reading;
    this keeps what came of them, decides what comes next, and says what it
    all amounted to.
    """

    def __init__(
        self,
        regions: "Sequence[Region]",
        step: int,
        *,
        objective: str = "average",
        longest: int = _LONGEST_LEG,
        clock: "Callable[[], float]" = time.monotonic,
        turn_back: float = TURN_BACK,
        depths: bool = False,
    ) -> None:
        if not regions:
            raise ValueError("nothing to calibrate")
        if objective not in OBJECTIVES:
            raise ValueError(f"no such objective as {objective!r}")
        self._regions = tuple(regions)
        self._step = abs(int(step)) or 1
        self._objective = objective
        self._longest = longest
        self._turn_back = min(max(float(turn_back), TURN_BACK_RANGE[0]), TURN_BACK_RANGE[1])
        self._measure_depths = bool(depths)
        count = len(self._regions)
        self._best: "list[Look | None]" = [None] * count
        self._outcomes = [""] * count
        self._views: "list[View | None]" = [None] * count
        self._index = 0
        self._search: "_Search | None" = None
        self._order: "list[int]" = []
        self._latest: "dict[int, Look]" = {}
        self._score: "float | None" = None
        # What it has cost so far, for the report.
        self._clock = clock
        self._started = clock()
        self._began = time.time()
        self._tune_probes: "list[int]" = []
        self._autofocuses = 0
        self._moves = 0
        self._travel = 0
        self._history: "list[tuple[str, tuple[float, ...], float]]" = []
        #: Whether the compromise has driven the lens yet, which is what says
        #: the next probe has to wait for the picture to settle first.
        self.moved = False
        #: A move the search asked for that the camera has not yet made --
        #: because live view went off as it was sent. It is made before the
        #: next probe, so where the search thinks it is stays true.
        self.pending = 0

    # -- what it is doing --------------------------------------------------

    @property
    def regions(self) -> "tuple[Region, ...]":
        return self._regions

    @property
    def step(self) -> int:
        return self._step

    @property
    def index(self) -> int:
        """Which region is being fine tuned, while the peaks are being found."""
        return self._index

    @property
    def phase(self) -> str:
        """``peaks``, then ``compromise``, then ``done``."""
        if self._search is not None:
            return "done" if self._search.done else "compromise"
        return "peaks" if self._index < len(self._regions) else "done"

    @property
    def probes(self) -> int:
        """How many times every region has been read for the compromise."""
        return self._search.probes if self._search is not None else 0

    @property
    def score(self) -> "float | None":
        """The average share of their best the regions read at the last probe."""
        return self._score

    @property
    def best(self) -> float:
        """The best average the compromise has read anywhere."""
        return self._search.best if self._search is not None else 0.0

    def view(self, index: int) -> "View | None":
        return self._views[index]

    @property
    def usable(self) -> "list[int]":
        """The regions with a peak to be measured against."""
        return [
            index
            for index, look in enumerate(self._best)
            if look is not None
            and look.reading > 0.0
            and self._views[index] is not None
        ]

    def progress(self) -> str:
        """One line on what the compromise is doing, for whoever is watching."""
        search = self._search
        if search is None:
            return ""
        doing = {
            "out": "walking out past every region's best",
            "across": "walking back across all of them",
            "home": "going back to the best of them",
            "climb": "climbing the hill it is on",
        }.get(search.state, "done")
        here = f", {self._score:.0%} here" if self._score is not None else ""
        probes = search.probes
        return (
            f"Compromise: {probes} probe{'' if probes == 1 else 's'} in steps "
            f"of {search.step_size}, {doing} -- best average so far "
            f"{search.best:.0%}{here}"
        )

    # -- the peaks ---------------------------------------------------------

    @property
    def measures_depths(self) -> bool:
        """Whether the walks were asked to go far enough to place every peak."""
        return self._measure_depths

    @property
    def turn_back(self) -> float:
        """The share of a walk's best below which the walk turns back."""
        return self._turn_back

    @property
    def objective(self) -> str:
        """Which of :data:`OBJECTIVES` the compromise is sought by."""
        return self._objective

    @property
    def elapsed(self) -> float:
        """Seconds since the calibration began."""
        return self._clock() - self._started

    @property
    def searching(self) -> str:
        """What the compromise is doing: ``out``, ``across``, ``home``, ``climb``."""
        return self._search.state if self._search is not None else ""

    @property
    def shares(self) -> "tuple[float, ...]":
        """Every usable region's share of its best at the last probe."""
        return self._history[-1][1] if self._history else ()

    @property
    def history_regions(self) -> "tuple[int, ...]":
        """Which regions, by number, the compromise reads -- in its order."""
        return tuple(index + 1 for index in self.usable)

    def drove(self, steps: int) -> None:
        """Focus was driven, by the fine tunes or the compromise."""
        self._moves += 1
        self._travel += abs(int(steps))

    def autofocused(self) -> None:
        """The camera's own autofocus was run, which drives an unknown way."""
        self._autofocuses += 1

    def region_tuned(
        self, look: Look, outcome: str, view: "View | None", probes: int = 0
    ) -> bool:
        """Keep what the fine tune on this region found; say if another follows."""
        if self._index >= len(self._regions):
            raise RuntimeError("every region has been tuned already")
        self._best[self._index] = look
        self._outcomes[self._index] = outcome
        self._views[self._index] = view
        self._tune_probes.append(int(probes))
        self._index += 1
        return self._index < len(self._regions)

    # -- the compromise ----------------------------------------------------

    def begin_compromise(self) -> bool:
        """Start looking for the one position; False if there is nothing to find.

        It takes two regions with a peak each to have anything to compromise
        between. The search starts wherever the last fine tune left the lens,
        on that region's peak, and without an autofocus: there is no one
        place for the camera to focus on. See :class:`_Search`.
        """
        usable = self.usable
        if len(usable) < 2:
            return False
        # The region tuned last is the one still on screen, so it is read
        # first and the first probe costs one pan fewer.
        self._order = sorted(usable, reverse=True)
        self._search = _Search(
            self._step,
            self._objective,
            longest=self._longest,
            turn_back=self._turn_back,
            depths=self._measure_depths,
        )
        return True

    def order(self) -> "list[int]":
        """Which regions to read at the next probe, in the order to read them.

        Reversed every time, so the region read last at one probe is read
        first at the next -- it is still on screen, and a pan saved is a
        quarter of a second and a stack.
        """
        now = list(self._order)
        self._order.reverse()
        return now

    def take(self, looks: "dict[int, Look]") -> "Move | None":
        """Every usable region read at one focus position; answer the next move.

        None means it is standing on the compromise, and *looks* are what
        every region reads there.
        """
        search = self._search
        if search is None:
            raise RuntimeError("the compromise has not begun")
        if search.done:
            return None
        usable = self.usable
        readings = [
            looks[index].reading if index in looks else 0.0 for index in usable
        ]
        peaks = [self._best[index].reading for index in usable]
        shares = [reading / peak for reading, peak in zip(readings, peaks)]
        self._latest = dict(looks)
        self._score = combine(shares, self._objective)
        self._history.append((search.state, tuple(shares), self._score))
        return search.step(shares)

    # -- what it all came to -----------------------------------------------

    def depths(self) -> "dict[int, tuple[float, float, bool]]":
        """Each region's depth, doubt and whether it is only a bound, by index.

        Read off the walk across, the one record of every region in a single
        direction, and relative to the nearest of them: the play taken up at
        the start of that walk moves every position on it by the same amount,
        so the differences between them are honest and the positions
        themselves mean nothing.
        """
        search = self._search
        if not self._measure_depths or search is None or search.survey is None:
            return {}
        where, shares = search.survey
        found = _depths(where, shares)
        usable = self.usable
        known = [position for position, _doubt, _edge in found if position is not None]
        if not known:
            return {}
        nearest = min(known)
        return {
            usable[column]: (position - nearest, doubt, edge)
            for column, (position, doubt, edge) in enumerate(found)
            if position is not None
        }

    def report(self, stopped: bool = False) -> CalibrationReport:
        search = self._search
        settled = search is not None and search.done and not stopped
        depths = self.depths()
        results = tuple(
            RegionResult(
                number=index + 1,
                region=region,
                best=self._best[index],
                outcome=self._outcomes[index],
                compromise=self._latest.get(index) if settled else None,
                depth=depths[index][0] if index in depths else None,
                doubt=depths[index][1] if index in depths else float("inf"),
                edge=depths[index][2] if index in depths else False,
            )
            for index, region in enumerate(self._regions)
        )
        if stopped:
            outcome = "stopped"
        elif search is not None:
            outcome = search.outcome
        else:
            outcome = "single" if len(self.usable) == 1 else "nothing"
        return CalibrationReport(
            results=results,
            score=self._score if settled else None,
            outcome=outcome,
            probes=self.probes,
            objective=self._objective,
            seconds=self.elapsed,
            tune_probes=tuple(self._tune_probes),
            autofocuses=self._autofocuses,
            moves=self._moves,
            travel=self._travel,
            history=tuple(self._history),
            history_regions=self.history_regions if search is not None else (),
            depths_measured=self._measure_depths,
            began=self._began,
        )


#: Where the line is drawn that marks out the top of one region's curve, for
#: its depth: between the higher of the curve's two ends on the walk across
#: and its peak, half way first, and lower if that leaves fewer than three
#: readings on the top. Half way, unlike the compromise's own line, because a
#: single region's curve is near enough symmetrical about its peak that the
#: middle of its top is the peak -- and the more readings the middle is taken
#: over, the less the grain on any of them moves it.
_DEPTH_TOP = (0.5, 0.25, 0.125)


def _depths(
    where: np.ndarray, shares: np.ndarray
) -> "list[tuple[float | None, float, bool]]":
    """Where on the walk across each region peaked, how sure, and whether at an end.

    The top of each region's curve is drawn from its two ends on the walk,
    not from the bottom of the curve: the walk turns back once the regions
    have fallen a fifth or so below their best, so what it has of each is the
    top and a little of each side, and a line drawn half way down to nothing
    would call every one of them cut off. The same line on both sides of the
    peak keeps the top symmetrical, so its middle is the peak. A region whose
    best reading is at an end of the walk had its peak outside it, and says
    so.
    """
    if shares.ndim != 2 or len(where) < 3:
        return []
    order = np.argsort(where, kind="stable")
    where, shares = where[order], shares[order]
    # A position read twice -- the reading the walk turned on -- keeps the
    # later reading, which is the one taken from the direction of the walk.
    unique = np.concatenate([where[1:] != where[:-1], [True]])
    where, shares = where[unique], shares[unique]
    if len(where) < 3:
        return []
    return [_peak_of(where, shares[:, column]) for column in range(shares.shape[1])]


def _peak_of(
    where: np.ndarray, read: np.ndarray
) -> "tuple[float | None, float, bool]":
    """One region's peak on the walk: where, give or take how much, and if cut off."""
    top = int(np.argmax(read))
    peak = float(read[top])
    ends = max(float(read[0]), float(read[-1]))
    if peak <= 0.0 or peak - ends < _TALL_ENOUGH * peak:
        # Flat, or its best at an end: the peak is outside the walk.
        if peak > 0.0 and top in (0, len(read) - 1):
            return float(where[top]), float("inf"), True
        return None, float("inf"), False
    for share in _DEPTH_TOP:
        level = ends + share * (peak - ends)
        first, last = _run_around_top(list(read), level)
        if last - first + 1 >= _TOP_READINGS:
            break
    position = _middle_of_top(list(where), list(read), level)
    if position is None:
        return None, float("inf"), False
    return position, _doubt(where, read, level), False


def _doubt(where: np.ndarray, read: np.ndarray, level: float) -> float:
    """How far out the position of a peak could be, in steps.

    The grain on the readings comes from the readings themselves: on a curve
    that is smooth apart from noise, each sample less the average of its two
    neighbours is noise and nothing else, and the median of those is a
    measure of it that a few wild samples cannot inflate. That is then
    carried through the arithmetic that found the peak -- a weighted middle
    of the positions on the top of the curve, so shifting one reading by *e*
    shifts the answer by *e* times that sample's leverage -- and the shifts
    add in quadrature.

    Kept from the point scan this replaced, where it was the difference
    between an answer and a guess given in the same voice. Splitting the
    samples in two and comparing the halves was tried there first, and is
    worth remembering as wrong: neighbouring samples of a broad peak read
    almost the same thing, so the halves agree beautifully and say nothing.
    """
    # Sampled every increment, a peak is not placed closer than a fraction of
    # one however clean the readings: the doubt is never less than that.
    least = 0.3 * float(np.median(np.diff(where))) if len(where) > 1 else 0.0
    if len(where) < 5:
        return float("inf")
    bumps = read[1:-1] - 0.5 * (read[:-2] + read[2:])
    # A second difference of white noise has one and a half times its
    # variance; the median of the sizes is 0.6745 of a standard deviation.
    grain = float(np.median(np.abs(bumps))) / (0.6745 * np.sqrt(1.5))
    first, last = _run_around_top(list(read), level)
    xs = where[first : last + 1]
    ys = read[first : last + 1] - level
    span = _spacing(where)[first : last + 1]
    weight = np.clip(ys, 0.0, None) * span
    total = float(weight.sum())
    if total <= 0.0:
        return float("inf")
    middle = float((weight * xs).sum()) / total
    leverage = float(np.sqrt(((span * (xs - middle)) ** 2).sum()))
    return max(grain * leverage / total, least)


def _spacing(where: np.ndarray) -> np.ndarray:
    """How much travel each sample stands for: half way to either neighbour."""
    if len(where) < 2:
        return np.ones(len(where))
    span = np.empty(len(where))
    span[0] = where[1] - where[0]
    span[-1] = where[-1] - where[-2]
    if len(where) > 2:
        span[1:-1] = (where[2:] - where[:-2]) / 2.0
    return np.maximum(span, 1e-9)
