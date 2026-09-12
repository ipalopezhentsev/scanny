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
region. The best reading the walk saw is that region's *peak*: the reading,
and the picture of the region at that moment. The best it *saw*, not the one
it would walk back to and stand on: where the lens is left does not matter,
since the compromise walks it somewhere else next, and walking back only ever
lands a little off the top -- seven per cent off, on a real calibration, which
every share of it was then wrongly measured against. Along with where the
camera was pointed when it read it, because a reading is only comparable with
another taken through the same crop at the same magnification: the same piece
of sensor magnified further has its detail spread over more pixels, and reads
differently for it.

**A peak can still go up.** The search for the compromise reads every region at
many focus positions, and a region it reads higher than its peak has that for
its peak from then on -- see :class:`_Search`.

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

**And then, if it is asked to, the aperture.** One focus position is only half
the compromise; the other half is how deep that position reaches, which is the
aperture's to decide -- against diffraction, which takes back from every
region what depth gives the ones off the plane. With the regions' bests frozen
it can simply be tried, a stop at a time, and the readings say where the trade
turns. That is :mod:`scanny.ui.aperture`.

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
from .aperture import STOP_SIZES
from .aperture import ApertureSearch, Ladder
from .aperture import Next as ApertureNext
from .aperture import Probe as ApertureProbe
from .film import FilmSurface
from .homing import TALL_ENOUGH as _TALL_ENOUGH
from .homing import TOP_READINGS as _TOP_READINGS
from .homing import WayHome
from .homing import middle_of_top as _middle_of_top
from .homing import run_around_top as _run_around_top
from .hunt import FineTune, Move
from .orientation import Orientation
from .sharpness import format_reading

__all__ = [
    "HURRY_BELOW",
    "HURRY_RANGE",
    "HURRY_STRIDE",
    "MAX_REGIONS",
    "OBJECTIVES",
    "TURN_BACK",
    "TURN_BACK_RANGE",
    "ApertureNext",
    "ApertureProbe",
    "Calibration",
    "CalibrationReport",
    "Look",
    "Reading",
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
class Reading:
    """One reading of one region during a calibration, kept for looking into.

    Every reading a calibration takes, in the order it takes them -- the fine
    tunes' walks as well as the compromise's -- with when it was taken and how
    long after the lens last moved. The report is made of the best of these,
    and a best that is out of line with the rest is only explained by the
    rest: a fine tune that reads a region lower than the compromise later
    does, say, is either reading too soon after its moves or watching
    something change over the minutes between them, and which of those it is
    is in :attr:`after` and :attr:`at`.
    """

    #: Which region, by number from 1.
    region: int
    #: ``tune`` for its fine tune; the search's ``out``, ``across``, ``home``
    #: or ``climb`` for the compromise; ``aperture`` for the search for the
    #: best aperture, which does not move the lens at all.
    stage: str
    #: Where, in drive steps: a fine tune's from where its autofocus left the
    #: lens, the compromise's from where it began. Comparable within one
    #: stretch of walking in one direction, and nowhere else. For an
    #: ``aperture`` reading it is not a position but the aperture itself, in
    #: PTP's hundredths of an f-stop.
    position: int
    value: float
    #: Seconds since the calibration began.
    at: float
    #: Seconds since the lens was last driven or autofocused, or not a number
    #: when that is not known.
    after: float = float("nan")
    #: The grain taken off it: the noise variance of one pixel of the picture
    #: read, of which four times came off the gradient energy. Not a number
    #: when not known.
    grain: float = float("nan")
    #: The mean level of the region in the picture read, which the energy was
    #: divided by the square of. With the reading and the grain, what the
    #: picture's gradient energy was, and so what the reading would have been
    #: with other grain taken off: ``value * level**2 / 1000 + 4 * grain``.
    level: float = float("nan")


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
    #: What its fine tune read at best, on its own. The same as :attr:`best`
    #: unless the search for the compromise read it higher since; None if
    #: not known.
    tuned: "float | None" = None
    #: What it read at the compromise **at the aperture the calibration ran
    #: at**, before the search for a better one changed it. None when no
    #: aperture was looked for -- :attr:`compromise` is then that same
    #: reading, since nothing about the aperture moved.
    before_aperture: "Look | None" = None

    @property
    def usable(self) -> bool:
        """Whether it had a peak to be measured against."""
        return self.best is not None and self.best.reading > 0.0

    @property
    def bettered(self) -> "float | None":
        """How far above its fine tune's best the compromise's search read it.

        As a fraction -- 0.06 is six per cent higher -- or None when it did
        not, which is the ordinary case. When it did, the fine tune stopped
        short of the region's peak, or read it lower than the search could.
        """
        if not self.usable or self.tuned is None or self.tuned <= 0.0:
            return None
        above = self.best.reading / self.tuned - 1.0
        return above if above > 0.0 else None

    @property
    def fraction(self) -> "float | None":
        """How near its best the compromise left it: 1 is its best."""
        if not self.usable or self.compromise is None:
            return None
        return self.compromise.reading / self.best.reading

    @property
    def before_fraction(self) -> "float | None":
        """The same share, at the aperture the calibration ran at.

        What this region had before the aperture was changed, so that what the
        change was worth to *it* can be read off rather than inferred from the
        one combined number the search climbed.
        """
        if not self.usable or self.before_aperture is None:
            return None
        return self.before_aperture.reading / self.best.reading

    @property
    def aperture_gain(self) -> "float | None":
        """What the change of aperture was worth to this region, as a fraction.

        0.32 is a third more sharpness than it had at the aperture the
        calibration ran at; negative is a region that gave something up for
        the others. None when no aperture was looked for.
        """
        if self.before_aperture is None or self.compromise is None:
            return None
        if self.before_aperture.reading <= 0.0:
            return None
        return self.compromise.reading / self.before_aperture.reading - 1.0


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
    #: Where the compromise was placed, in the depths' own steps -- so many
    #: further than the nearest region -- or None when there is no walk across
    #: to read it off, or the compromise settled at an end of it. The plane it
    #: brings into focus, which the film's shape is to be judged against.
    focus_depth: "float | None" = None
    #: Every aperture the search for the best one tried, in the order it
    #: tried them; empty when it was not asked for or could not be run. See
    #: :mod:`scanny.ui.aperture`.
    apertures: "tuple[ApertureProbe, ...]" = ()
    #: The aperture the calibration ran at, and the one it chose, both in
    #: PTP's hundredths of an f-stop; 0 for neither.
    aperture_started: int = 0
    aperture_chosen: int = 0
    #: How the aperture search ended -- ``found``, ``bounded`` or ``single``
    #: as :attr:`scanny.ui.aperture.ApertureSearch.outcome` has them -- or ""
    #: when there was none, and why there was none when something stopped it.
    aperture_outcome: str = ""
    aperture_note: str = ""
    #: When it began, in seconds since the epoch, or 0 if not known.
    began: float = 0.0
    #: Everything that was said while it ran, as the activity log has it.
    log: "tuple[str, ...]" = ()
    #: Every reading of every region it took; see :class:`Reading`.
    readings: "tuple[Reading, ...]" = ()

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
        if self.apertures:
            parts.append(f"{len(self.apertures)} apertures")
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

    # -- the aperture ------------------------------------------------------

    @property
    def aperture_probe(self) -> "ApertureProbe | None":
        """The probe at the aperture it chose, or None without a search."""
        for probe in self.apertures:
            if probe.aperture == self.aperture_chosen:
                return probe
        return None

    @property
    def aperture_gain(self) -> "float | None":
        """How much better the chosen aperture reads than the one it started at.

        As a fraction of the starting aperture's score -- 0.08 is eight per
        cent more of every region's best, on average -- or None when there is
        nothing to compare. Zero when it chose the aperture it started at,
        which is an answer: nothing on the ladder beat what was already set.
        """
        chosen, started = self.aperture_probe, None
        for probe in self.apertures:
            if probe.aperture == self.aperture_started:
                started = probe
        if chosen is None or started is None or started.score <= 0.0:
            return None
        return chosen.score / started.score - 1.0

    def describe_aperture(self) -> str:
        """What the search for the aperture came to, in one line."""
        if not self.apertures:
            return self.aperture_note
        chosen = self.aperture_probe
        if chosen is None:
            return self.aperture_note
        started = self.aperture_started
        gain = self.aperture_gain
        if chosen.aperture == started:
            line = (
                f"Aperture: {chosen.label} is the best of the "
                f"{len(self.apertures)} tried -- the one it started at"
            )
        else:
            way = "stopped down" if chosen.aperture > started else "opened up"
            line = (
                f"Aperture: {way} {abs(chosen.stops):.1f} stops to {chosen.label}"
            )
            if gain:
                line += f", worth {gain:+.0%} against the aperture it started at"
        if chosen.shutter_label:
            line += f", at {chosen.shutter_label}"
        if self.aperture_outcome == "bounded":
            line += (
                ". It is as far as the search was let go, so a better one may "
                "lie beyond it"
            )
        return line

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
        aperture = self.describe_aperture()
        return f"{line}. {aperture}" if aperture else line


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
    if result.bettered is not None:
        lines.append(
            f"Best: {format_reading(result.best.reading)}, read on the walk for "
            f"the compromise -- {result.bettered:.0%} above the "
            f"{format_reading(result.tuned)} fine tuning it on its own found"
        )
    else:
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

#: How long a stride to start from, in increments: what the panel recommends
#: under **Hurry in strides of**, and what the tick this setting used to be
#: turns into. How long a stride actually is, is the user's -- see
#: :data:`HURRY_RANGE`.
#:
#: A stride is a multiple of whatever increment the walk is made in -- the
#: lens's minimum, the user's setting -- so it scales with the lens rather
#: than with a number chosen here. Two of them is twelve drive steps on the
#: six-step increment a D750 kit lens has.
#:
#: **Two, and it was four.** What a stride can do that an increment cannot is
#: step clean over a hill: the average of several regions is broader than any
#: one of them, but a magnified region's top can be a single increment wide
#: and two regions nearly on top of each other add up to little more. On a
#: real five-region calibration four was plainly too much -- it strode into
#: the peak from one side and away from it on the other, leaving the leg
#: sampled at one increment around the top and four everywhere else. Two
#: halved the walking, read the same compromise as walking it did (91% both
#: ways, 61 probes against 86), and is near enough to an increment that the
#: profile a leg leaves behind is still an honest one.
HURRY_STRIDE = 2

#: The least and most a stride may be set to, as **Hurry in strides of**.
#:
#: One is no hurrying at all, which is what the search does unless asked. The
#: top is loose rather than a judgement: past two the strides start to tell on
#: the answer -- at four, readings a third grainy cost twice what walking them
#: cost -- and somebody with a lens whose minimum increment is finer than a
#: D750's, or a subject far deeper than a frame of film, has every right to a
#: longer one. What it costs is written down beside it.
HURRY_RANGE = (1, 8)

#: The average sharpness below which a hurrying walk strides, and at or above
#: which it walks single increments again.
#:
#: An **absolute** share of what the regions can each do -- the number the
#: panel shows as a percentage while the search runs -- and not a share of the
#: best the search has read so far. Both were tried against a real rig, and
#: the moving reference is what made the pace flap: the best read anywhere
#: goes up while the walk is walking, so the line it is judged against moves
#: under it, and a walk strode into a peak from one side and away from it on
#: the other in the same leg. A fixed line does not move, which is the whole
#: point: below it the walk is in a valley and hurries out, at or above it the
#: walk is where the answer is decided and steps as it always did.
#:
#: Four fifths is where a person watching the trend line puts it, and it is
#: what was asked for. Where a scene tops out under it the walks stride
#: throughout -- at two increments, and with the way home and the climb still
#: walking the answer an increment at a time.
HURRY_BELOW = 0.8

#: How many readings in a row have to call the ground soft before a walk
#: strides across it.
#:
#: One reading is not enough, and this is what a real calibration on unsteady
#: readings showed: where a region's reading swings by half between one probe
#: and the next, a single reading puts the pace anywhere, and the walk strides
#: over good ground and creeps across dead ground in the same leg. Two in a
#: row costs one probe at the top of every soft stretch and makes the pace
#: mean something.
_SOFT_ENOUGH = 2

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

#: How far below the best of its own leg the number being climbed has to
#: collapse for a walk to turn back at once, whatever it was waiting for.
#:
#: The two waits in :meth:`_Search._past` -- a region visibly climbing, and
#: the depths -- are both waits for something to be *read*, and neither can be
#: read in a picture with nothing above its own grain in it. A tenth of what
#: the leg has seen is that: a region at a tenth of its best is not climbing
#: towards anything and has no peak to place, so walking further into it buys
#: nothing and costs a pan to every region and a wait for the picture at each
#: step. A real calibration spent twenty probes at a fiftieth of its best
#: because a region that read nothing at all could never satisfy the depths.
#:
#: Against the best of *this leg*, which is what makes it safe on the leg
#: across: that leg sets off from the collapsed far end of the leg out, where
#: the best it has read is the reading it is standing on, so nothing is under
#: a tenth of it and the walk crosses the peaks as it must.
_GIVE_UP = 0.1

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

#: How many times the walk home may turn round and go home again, from the
#: profile the last walk home made, before giving the search to a climb.
_HOMECOMINGS = 2


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
    - **Home**: turn round again, and go back along the leg across to the
      best of it, measuring the play on the way; see
      :class:`~scanny.ui.homing.WayHome`.

    If the way home loses the top -- the scene moved, or the readings would
    not hold still -- it is not the end of the search: a
    :class:`~scanny.ui.hunt.FineTune` takes over from wherever it is and
    climbs the hill it is on, which is what it is good at.

    **What each region's share is a share of can go up while it walks.** It is
    handed every region's reading, not its share, and each share is of the
    best that region has read *anywhere* -- its fine tune or this walk. The
    walk across crosses every region's peak, read the way every later reading
    will be read, and it can read a region higher than its fine tune did: a
    real calibration had one region a sixth over. Its old best would leave it
    counting for more than its share in the average, and being protected when
    it needs no protecting in the worst -- so its best is raised, and
    everything the walk judges by is worked out again from the readings.
    Only the way home is judged by the bests as they stood when it set out:
    it matches readings against a profile, and a profile has to hold still.

    **It may be told to hurry through focus no compromise can be in.** Asked
    to, the two walks take a stride of several increments (*hurry*) while the
    number they are climbing is well below the best read anywhere, and go back
    to single increments as they come up to it -- so the ground where the
    answer is decided is walked exactly as it always was, and the soft ground
    either side of it is crossed in a quarter of the probes. The way home and
    the climb never hurry; see :meth:`_hurrying`.
    """

    def __init__(
        self,
        unit: int,
        peaks: "Sequence[float]",
        objective: str = "average",
        *,
        longest: int = _LONGEST_LEG,
        turn_back: float = TURN_BACK,
        depths: bool = False,
        hurry: int = 1,
    ) -> None:
        self._unit = abs(int(unit)) or 1
        self._objective = objective
        #: Every region's best reading: what its share is a share of. Raised
        #: whenever a reading beats it.
        self.peaks = [float(peak) for peak in peaks]
        #: The bests the way home and the climb are judged by, held as they
        #: were when they set out.
        self._held: "list[float]" = list(self.peaks)
        #: Whether the walks go on far enough to place every region's peak.
        self._depths = bool(depths)
        self._longest = max(4, int(longest))
        self._turn_back = min(max(float(turn_back), TURN_BACK_RANGE[0]), TURN_BACK_RANGE[1])
        #: How many increments a step of the walks is worth while what they
        #: are climbing is plainly soft; one for no hurrying at all.
        self._hurry = max(1, int(hurry))
        #: How many increments the last step was worth, for saying so while
        #: it runs.
        self._stride = 1
        #: How many readings in a row have said the ground here is soft.
        self._soft = 0
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
        #: The leg being walked: where, and what every region read there.
        self._leg: "list[tuple[int, tuple[float, ...]]]" = []
        #: What every region read at every probe, for the best of them.
        self._probed: "list[tuple[float, ...]]" = []
        self._way: "WayHome | None" = None
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
    def stride(self) -> int:
        """How many increments the last step was worth: above one is hurrying."""
        return self._stride

    @property
    def position(self) -> int:
        """Where the next reading is taken, in step counts from where it began."""
        return self._position

    @property
    def done(self) -> bool:
        return self._state == "done"

    def shares(
        self, readings: "Sequence[float]", peaks: "Sequence[float] | None" = None
    ) -> "tuple[float, ...]":
        """Every region's reading as a share of its best -- as it is now, or *peaks*."""
        peaks = self.peaks if peaks is None else peaks
        return tuple(
            reading / peak if peak > 0.0 else 0.0
            for reading, peak in zip(readings, peaks)
        )

    def score(
        self, readings: "Sequence[float]", peaks: "Sequence[float] | None" = None
    ) -> float:
        """The one number climbed, for what every region read at one probe."""
        return combine(self.shares(readings, peaks), self._objective)

    def step(self, readings: "Sequence[float]") -> "Move | None":
        """What every region read here; answer the next move."""
        if self._state == "done":
            return None
        readings = tuple(float(reading) for reading in readings)
        self.probes += 1
        self.peaks = [max(peak, reading) for peak, reading in zip(self.peaks, readings)]
        self._probed.append(readings)
        self.best = max(self.score(seen) for seen in self._probed)
        if self._state == "home":
            return self._home(readings)
        if self._state == "climb":
            return self._climbing(readings)
        return self._walk(readings)

    # -- out and across ----------------------------------------------------

    def _walk(self, readings: "tuple[float, ...]") -> "Move | None":
        self._leg.append((self._position, readings))
        if not (self._past(readings) or self._walked() > self._longest):
            return self._drive(self._hurrying(readings))
        if self._state == "out":
            # The leg across begins where this one ended, with this reading.
            self._state = "across"
            self._direction = -self._direction
            self._leg = [self._leg[-1]]
            self._waited = 0
            return self._drive(self._hurrying(readings))
        self.survey = (
            np.array([position for position, _r in self._leg], dtype=float),
            np.array([self.shares(seen) for _p, seen in self._leg], dtype=float),
        )
        return self._turn_for_home(readings)

    def _walked(self) -> int:
        """How far this leg has come, in increments.

        In increments and not in probes, which is the whole difference between
        a walk that hurries and one that simply goes four times as far. Every
        limit that stops a walk -- this one, and the two patiences in
        :meth:`_past` -- is a distance, and each of them was being counted by
        the probe because until there were strides the two were the same
        number. They are not: told to hurry, a real calibration walked 672
        steps out where it had walked 156, and 1230 across where it had walked
        228, straight out into focus where nothing reads at all -- and then
        crawled the whole of it back an increment at a time, because the way
        home never hurries. It took 206 probes where the same rig had taken
        86. The strides are meant to cross the same ground in fewer probes,
        never to cross more of it.
        """
        return abs(self._position - self._leg[0][0]) // self._unit

    def _leg_top(self) -> float:
        """The best number this leg has read, by every region's best as it is now."""
        return max(self.score(seen) for _p, seen in self._leg)

    def _leg_best(self) -> "list[float]":
        """Each region's best share on this leg, of its best as it is now."""
        return list(
            self.shares([max(column) for column in zip(*(seen for _p, seen in self._leg))])
        )

    def _past(self, readings: "tuple[float, ...]") -> bool:
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

        **Both of those waits end at once where the reading has collapsed.**
        Under :data:`_GIVE_UP` of what this leg has read there is nothing
        there to wait for: a region cannot be seen to climb in a picture with
        nothing above its own grain in it, and a peak cannot be placed from
        readings of nothing either, so the walk turns instead of walking on
        into it. Measured against the best of *this leg*, which is what makes
        it safe to apply to the leg across as well: that leg sets off from the
        soft far end of the leg out, where the best it has read is the
        collapsed reading it is standing on -- so nothing is under a tenth of
        it, and the walk goes on and crosses the peaks as it must. Only once
        the leg has seen something does falling to a tenth of it mean
        anything.

        A calibration whose walks had been carried a long way out reported
        this as the real complaint: the number had been at a fiftieth of its
        best for twenty probes, and every one of them was a pan to each region
        and a wait for a picture with nothing in it.
        """
        top = self._leg_top()
        if len(self._leg) < 2 or self.score(readings) >= self._turn_back * top:
            self._waited = 0
            return False
        if self.score(readings) < _GIVE_UP * top:
            return True
        # In increments of travel, not in probes: see _walked.
        self._waited += self._stride
        shares, best = self.shares(readings), self._leg_best()
        if self._depths:
            if all(self._seen(share, most) for share, most in zip(shares, best)):
                return True
            return self._waited > _DEPTH_PATIENCE
        if self._rising(shares, best):
            return self._waited > _CLIMB_PATIENCE
        return True

    def _hurrying(self, readings: "tuple[float, ...]") -> int:
        """How many increments the next step of a walk is worth.

        Two things decide it and nothing else: whether hurrying was asked for,
        and whether the average sharpness here is under :data:`HURRY_BELOW`.
        Under it the walk is in a valley, where no compromise is going to be
        found and the only thing to be had is travel, so it strides; at or
        above it the walk is where the answer is decided, and it steps exactly
        as it would have without hurrying.

        **One line, fixed, and not a share of the best read so far.** The
        moving reference is what made the pace flap on a real rig: the best
        read anywhere goes up while the walk walks, so the line moved under
        it, and one leg strode into a peak from one side and away from it on
        the other. Going back to single increments takes one reading at or
        above the line; striding takes :data:`_SOFT_ENOUGH` under it, since
        slowing down where it might matter is cheap and hurrying where it
        might matter is not.

        Only the walks out and across. The way home is paced by how far it
        has left to go rather than by what it reads
        (:meth:`~scanny.ui.homing.WayHome.stride`), and the climb that takes
        over when the way home is lost is a fine tune, which takes the one
        increment it was given on purpose -- see :mod:`scanny.ui.hunt`.
        """
        if self._hurry <= 1:
            return 1
        self._soft = self._soft + 1 if self.score(readings) < HURRY_BELOW else 0
        return self._hurry if self._soft >= _SOFT_ENOUGH else 1

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

    def _rising(self, shares: "tuple[float, ...]", best: "list[float]") -> bool:
        """Whether some region is visibly on its way up to its own best."""
        recent = [self.shares(seen) for _p, seen in self._leg[-_CLIMBING_OVER - 1 :]]
        for region, (share, most) in enumerate(zip(shares, best)):
            lowest = min(seen[region] for seen in recent)
            if share >= most - 0.01 and share - lowest >= _CLIMBING:
                return True
        return False

    def _turn_for_home(self, readings: "tuple[float, ...]") -> "Move | None":
        """Make a profile of the leg just walked, find its best, and start back."""
        self._held = list(self.peaks)
        way = WayHome.along(
            [position for position, _r in self._leg],
            [self.score(seen, self._held) for _p, seen in self._leg],
            self._unit,
        )
        if way is None:
            return self._start_climb(readings)
        self._way = way
        self._state = "home"
        self._direction = -self._direction
        # The walk home is a leg of its own, and if it has to it will be the
        # profile for another way home; see _arrived.
        self._leg = [(self._position, readings)]
        return self._drive_home()

    # -- home --------------------------------------------------------------

    def _home(self, readings: "tuple[float, ...]") -> "Move | None":
        assert self._way is not None
        self._leg.append((self._position, readings))
        score = self.score(readings, self._held)
        verdict = self._way.heard(score)
        if verdict == "home":
            return self._arrived(readings, score)
        if verdict == "lost":
            return self._start_climb(readings)
        return self._drive_home()

    def _arrived(self, readings: "tuple[float, ...]", score: float) -> "Move | None":
        """Home, if the reading here agrees; otherwise, go home again.

        The walk home is an honest leg in one direction once its play was
        taken up, and it has usually crossed the best by then: so it is a
        profile as good as the leg across was, and the way home from it is
        the same arithmetic again. Only when that has been tried and still
        does not agree -- the scene moved, or the readings will not hold
        still -- is the search handed to a climb, which is what a fine tune
        does best and which climbs whatever hill it is next to.
        """
        assert self._way is not None
        if self._way.agrees(score):
            return self._finish("found")
        if self._homecomings > 0 and len(self._leg) >= 5:
            self._homecomings -= 1
            return self._turn_for_home(readings)
        return self._start_climb(readings)

    def _drive_home(self) -> Move:
        """One step of the way home, which is where the strides are worth most.

        The walks out and across are two thirds of the probes and the strides
        cut them to a quarter -- and it was all given back here: a real
        calibration's way home took seventy probes against the twenty-four the
        two walks had taken, because it retraced at single increments a
        stretch the strides had crossed four at a time, and then, on readings
        that would not hold still, went home again twice more.

        So the way home strides too, but on distance rather than on what it
        reads: :meth:`~scanny.ui.homing.WayHome.stride` gives back what the
        play worked out so far says there is still to go, less a couple of
        increments kept in hand, and the last of the way -- where the reading
        is matched against the profile and the answer is landed on -- is
        walked exactly as it always was.
        """
        assert self._way is not None
        stride = self._way.stride(self._hurry)
        self._way.drive(stride)
        return self._drive(stride)

    # -- when the way home is lost -----------------------------------------

    def _start_climb(self, readings: "tuple[float, ...]") -> "Move | None":
        self._state = "climb"
        self._held = list(self.peaks)
        # A fine tune drives itself, in the one increment it was given.
        self._stride = 1
        self._climb = FineTune(self._unit, autofocus=False)
        return self._climbing(readings)

    def _climbing(self, readings: "tuple[float, ...]") -> "Move | None":
        assert self._climb is not None
        move = self._climb.step(self.score(readings, self._held))
        if move is None:
            return self._finish(self._climb.outcome or "found")
        self._position += move.steps
        return move

    # -- driving -----------------------------------------------------------

    def _drive(self, stride: int = 1) -> Move:
        self._stride = max(1, int(stride))
        steps = self._direction * self._unit * self._stride
        self._position += steps
        return Move(steps=steps)

    def _finish(self, outcome: str) -> None:
        self._state = "done"
        self.outcome = outcome
        return None


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
        hurry: int = 1,
        apertures: str = "",
    ) -> None:
        apertures = apertures if isinstance(apertures, str) else ""
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
        #: How many increments a stride is worth where the compromise's walks
        #: are in soft focus; one for no hurrying at all. See
        #: :meth:`_Search._hurrying`.
        self._hurry = min(max(int(hurry), HURRY_RANGE[0]), HURRY_RANGE[1])
        #: Which of :data:`scanny.ui.aperture.STOP_SIZES` the aperture is to
        #: be walked in once there is a compromise, or "" not to look for one.
        self._aperture_key = apertures if apertures in STOP_SIZES else ""
        self._apertures: "ApertureSearch | None" = None
        #: Why there is no aperture search, when something stopped one that
        #: was asked for; and the score at the aperture it settled on.
        self._aperture_note = ""
        self._aperture_score: "float | None" = None
        #: Every region as it was read at the **first** aperture tried, which
        #: is the one the calibration ran at. Kept whole, pictures and all,
        #: because what the change of aperture did to a region is a thing to
        #: be looked at rather than taken on the word of a percentage.
        self._aperture_first: "dict[int, Look]" = {}
        #: An aperture the search has asked for that the camera has not been
        #: put to yet. Set before the probe that reads it, like
        #: :attr:`pending`, so that a live view that ends in the middle of one
        #: takes the whole probe again rather than reading the wrong aperture.
        self.pending_aperture: "ApertureNext | None" = None
        count = len(self._regions)
        #: Each region's best reading and the picture of it: its fine tune's
        #: to begin with, then any the compromise reads higher.
        self._best: "list[Look | None]" = [None] * count
        #: What each region's fine tune found on its own, kept for saying so
        #: when the compromise has read it higher since.
        self._tuned: "list[float | None]" = [None] * count
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
        #: The compromise probe by probe: what the search was doing, and what
        #: every usable region read. Readings rather than shares, because
        #: what a share is a share of can go up while it runs.
        self._history: "list[tuple[str, tuple[float, ...]]]" = []
        #: Every reading of every region, for the report; see :class:`Reading`.
        self._readings: "list[Reading]" = []
        #: The regions, by index, whose best the last probe of the compromise
        #: read higher than, so the worker can say so.
        self.bettered: "tuple[int, ...]" = ()
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
        """``peaks``, then ``compromise``, then ``apertures``, then ``done``.

        The aperture phase only exists when one was asked for and the
        compromise got far enough to be worth one; see :meth:`begin_apertures`.
        """
        if self._apertures is not None:
            return "done" if self._apertures.done else "apertures"
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

    def best_of(self, index: int) -> float:
        """Region *index*'s best reading so far, or 0 if it has none."""
        look = self._best[index]
        return look.reading if look is not None else 0.0

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
        # Said as it is being walked, since that is the question someone
        # watching a hurried search has: is it striding, or has it slowed
        # down because it is on to something?
        pace = (
            f"strides of {search.stride * search.step_size}"
            if search.stride > 1
            else f"steps of {search.step_size}"
        )
        return (
            f"Compromise: {probes} probe{'' if probes == 1 else 's'} in {pace}"
            f", {doing} -- best average so far "
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
    def hurry(self) -> int:
        """How many increments a stride is, in soft focus; one for not hurrying."""
        return self._hurry

    @property
    def hurries(self) -> bool:
        """Whether the walks stride through focus no compromise can be in."""
        return self._hurry > 1

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
        if not self._history or self._search is None:
            return ()
        return self._search.shares(self._history[-1][1])

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

    def noted(
        self,
        index: int,
        stage: str,
        position: int,
        value: float,
        after: float = float("nan"),
        *,
        grain: float = float("nan"),
        level: float = float("nan"),
    ) -> None:
        """One reading of region *index*, for the report's record of them all.

        The compromise's readings are noted by :meth:`take`; this is for the
        fine tunes', which only the worker sees. See :class:`Reading`.
        """
        self._readings.append(
            Reading(
                region=index + 1,
                stage=stage,
                position=int(position),
                value=float(value),
                at=self.elapsed,
                after=float(after),
                grain=float(grain),
                level=float(level),
            )
        )

    def region_tuned(
        self, look: Look, outcome: str, view: "View | None", probes: int = 0
    ) -> bool:
        """Keep what the fine tune on this region found; say if another follows.

        *look* is its best reading and the picture that came with it, and
        *view* the view it was read through -- which every later reading of
        the region is taken through.
        """
        if self._index >= len(self._regions):
            raise RuntimeError("every region has been tuned already")
        self._best[self._index] = look
        self._tuned[self._index] = look.reading
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
        near that region's peak, and without an autofocus: there is no one
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
            [self._best[index].reading for index in usable],
            self._objective,
            longest=self._longest,
            turn_back=self._turn_back,
            depths=self._measure_depths,
            hurry=self._hurry,
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

    def take(
        self,
        looks: "dict[int, Look]",
        after: "dict[int, float] | None" = None,
        grain: "dict[int, float] | None" = None,
        level: "dict[int, float] | None" = None,
    ) -> "Move | None":
        """Every usable region read at one focus position; answer the next move.

        None means it is standing on the compromise, and *looks* are what
        every region reads there. *after* is how long after the lens last
        moved each was read, *grain* the grain taken off it and *level* its
        mean level, for the record; see :class:`Reading`.

        A region read higher than its best has that for its best from here
        on, picture and all -- see :class:`_Search` for why -- and is listed
        in :attr:`bettered` until the next probe.
        """
        search = self._search
        if search is None:
            raise RuntimeError("the compromise has not begun")
        if search.done:
            return None
        usable = self.usable
        readings = tuple(
            looks[index].reading if index in looks else 0.0 for index in usable
        )
        bettered = []
        for index in usable:
            look = looks.get(index)
            if look is not None and look.reading > self._best[index].reading:
                self._best[index] = look
                bettered.append(index)
        self.bettered = tuple(bettered)
        unknown = float("nan")
        for index, reading in zip(usable, readings):
            self.noted(
                index,
                search.state,
                search.position,
                reading,
                (after or {}).get(index, unknown),
                grain=(grain or {}).get(index, unknown),
                level=(level or {}).get(index, unknown),
            )
        self._latest = dict(looks)
        self._history.append((search.state, readings))
        move = search.step(readings)
        self._score = search.score(readings)
        return move

    # -- the aperture ------------------------------------------------------

    @property
    def seeks_aperture(self) -> bool:
        """Whether an aperture is to be looked for once there is a compromise."""
        return bool(self._aperture_key)

    @property
    def aperture_stops(self) -> float:
        """How big a step to take between apertures, in EV."""
        return STOP_SIZES.get(self._aperture_key, STOP_SIZES["whole"])

    @property
    def aperture(self) -> int:
        """The aperture being read now, or 0 when that phase is not running.

        What tells one aperture's readings of a region from another's --
        the grain on them differs, and so does the picture -- so it is part of
        what those readings are kept under.
        """
        return self._apertures.aperture if self._apertures is not None else 0

    @property
    def aperture_probes(self) -> "tuple[ApertureProbe, ...]":
        return self._apertures.probes if self._apertures is not None else ()

    def no_aperture(self, why: str) -> None:
        """There will be no aperture search, and this is what to say about it."""
        self._aperture_note = why

    def begin_apertures(
        self, ladder: Ladder, aperture: int, shutter: int = 0
    ) -> bool:
        """Start looking for the aperture; False if there is nothing to look for.

        It takes a compromise to judge an aperture by -- every region read at
        one focus position, against bests that are now frozen (see
        :mod:`scanny.ui.aperture`) -- so this only follows a compromise that
        finished, and only when the body offers more than one aperture to put
        the lens to.
        """
        if not self._aperture_key or self._apertures is not None:
            return False
        if self._search is None or not self._search.done:
            return False
        if len(self.usable) < 2 or not ladder.holds(aperture):
            return False
        if len(ladder.apertures) < 2:
            self._aperture_note = (
                "The body offered only one aperture, so there was nothing to try"
            )
            return False
        self._apertures = ApertureSearch(
            ladder,
            aperture,
            shutter,
            size=self.aperture_stops,
            objective=self._objective,
        )
        return True

    def take_aperture(
        self,
        looks: "dict[int, Look]",
        after: "dict[int, float] | None" = None,
        grain: "dict[int, float] | None" = None,
        level: "dict[int, float] | None" = None,
    ) -> "ApertureNext | None":
        """Every region read at one aperture; answer the next one to try.

        None means it is done and the camera stands at the aperture it chose.
        The readings are kept the way the compromise's are, but **nothing here
        may better a region's best**: those are what the apertures are being
        compared against, and a yardstick that grows with what it measures
        measures nothing. A region reading over one is the answer, not a
        correction to make.
        """
        search = self._apertures
        if search is None:
            raise RuntimeError("the aperture search has not begun")
        if search.done:
            return None
        if not search.probes:
            # The first probe is at the aperture the calibration ran at, and
            # it is the before to everything after it.
            self._aperture_first = dict(looks)
        usable = self.usable
        readings = tuple(
            looks[index].reading if index in looks else 0.0 for index in usable
        )
        unknown = float("nan")
        for index, reading in zip(usable, readings):
            self.noted(
                index,
                "aperture",
                search.aperture,
                reading,
                (after or {}).get(index, unknown),
                grain=(grain or {}).get(index, unknown),
                level=(level or {}).get(index, unknown),
            )
        shares = tuple(
            reading / self.best_of(index) if self.best_of(index) > 0.0 else 0.0
            for index, reading in zip(usable, readings)
        )
        score = combine(shares, self._objective)
        self._latest = dict(looks)
        self._aperture_score = score
        return search.step(score, shares)

    # -- what it all came to -----------------------------------------------

    def depths(self) -> "dict[int, tuple[float, float, bool]]":
        """Each region's depth, doubt and whether it is only a bound, by index.

        Read off the walk across, the one record of every region in a single
        direction, and relative to the nearest of them: the play taken up at
        the start of that walk moves every position on it by the same amount,
        so the differences between them are honest and the positions
        themselves mean nothing.
        """
        walked = self._walk_across()
        if walked is None:
            return {}
        _where, _shares, found, nearest = walked
        usable = self.usable
        return {
            usable[column]: (position - nearest, doubt, edge)
            for column, (position, doubt, edge) in enumerate(found)
            if position is not None
        }

    def _walk_across(self):
        """The walk across and what its peaks came to, or None without it.

        Its positions in order, every region's share at each of them, where
        each region peaked, and the nearest of those peaks -- which is what
        everything read off the walk is counted from.
        """
        search = self._search
        if not self._measure_depths or search is None or search.survey is None:
            return None
        where, shares = search.survey
        found = _depths(where, shares)
        known = [position for position, _doubt, _edge in found if position is not None]
        if not known:
            return None
        return (*_in_order(where, shares), found, min(known))

    def focus_depth(self) -> "float | None":
        """How deep the compromise was placed, in the depths' own steps.

        The one focus position all the regions were left at, said the way
        their depths are: so many steps further than the nearest of them. It
        is read off the same walk across, as the top of the very number the
        search climbs -- every region's share of its best, combined the way
        the objective says -- so it is on the depths' own axis and can be
        drawn among them.

        Not the drive position the lens finally stands at, which is on the
        far side of a reversal and comparable with nothing here. The search
        goes home to the top of this walk and climbs from there, so this is
        where it placed the focus, give or take the increment or two the
        climb moved it by.
        """
        walked = self._walk_across()
        if walked is None:
            return None
        where, shares, _found, nearest = walked
        climbed = np.array(
            [combine(tuple(row), self._objective) for row in shares], dtype=float
        )
        position, _doubt, edge = _peak_of(where, climbed)
        if position is None or edge:
            return None
        return position - nearest

    def report(self, stopped: bool = False) -> CalibrationReport:
        """What it all came to.

        Every share in it is of each region's best as it finally stood --
        the search's history included, which the search itself saw against
        bests that were still going up.
        """
        search = self._search
        hunted = self._apertures
        settled = search is not None and search.done and not stopped
        if settled and hunted is not None and not hunted.done:
            # An aperture search that never finished leaves the regions read
            # at whichever aperture it had got to, which is nobody's answer.
            settled = False
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
                tuned=self._tuned[index],
                before_aperture=(
                    self._aperture_first.get(index)
                    if settled and hunted is not None
                    else None
                ),
            )
            for index, region in enumerate(self._regions)
        )
        history: "tuple[tuple[str, tuple[float, ...], float], ...]" = ()
        score = None
        if search is not None:
            history = tuple(
                (stage, search.shares(readings), search.score(readings))
                for stage, readings in self._history
            )
            if settled and self._history:
                score = search.score(self._history[-1][1])
        if settled and self._aperture_score is not None:
            # The regions were read again at the aperture it chose, and those
            # are the readings the results carry, so this is their number too.
            score = self._aperture_score
        if stopped:
            outcome = "stopped"
        elif search is not None:
            outcome = search.outcome
        else:
            outcome = "single" if len(self.usable) == 1 else "nothing"
        return CalibrationReport(
            results=results,
            score=score,
            outcome=outcome,
            probes=self.probes,
            objective=self._objective,
            seconds=self.elapsed,
            tune_probes=tuple(self._tune_probes),
            autofocuses=self._autofocuses,
            moves=self._moves,
            travel=self._travel,
            history=history,
            history_regions=self.history_regions if search is not None else (),
            depths_measured=self._measure_depths,
            focus_depth=self.focus_depth() if settled else None,
            apertures=hunted.probes if hunted is not None else (),
            aperture_started=hunted.start if hunted is not None else 0,
            # An unfinished search chose nothing: the worker puts the aperture
            # back where it found it, so there is no answer to report.
            aperture_chosen=(
                hunted.chosen.aperture
                if hunted is not None and hunted.done and hunted.chosen is not None
                else 0
            ),
            aperture_outcome=(
                ""
                if hunted is None
                else (hunted.outcome if hunted.done else "stopped")
            ),
            aperture_note=self._aperture_note,
            began=self._began,
            readings=tuple(self._readings),
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
    where, shares = _in_order(where, shares)
    if len(where) < 3:
        return []
    return [_peak_of(where, shares[:, column]) for column in range(shares.shape[1])]


def _in_order(where: np.ndarray, shares: np.ndarray):
    """The walk across with its positions in order and each of them read once.

    A position read twice -- the reading the walk turned on -- keeps the later
    reading, which is the one taken from the direction of the walk.
    """
    order = np.argsort(where, kind="stable")
    where, shares = where[order], shares[order]
    unique = np.concatenate([where[1:] != where[:-1], [True]])
    return where[unique], shares[unique]


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
