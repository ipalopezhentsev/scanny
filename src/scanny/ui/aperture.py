"""Choosing the aperture, once there is one focus position for all the regions.

A compromise (:mod:`scanny.ui.regions`) leaves every region somewhere short of
its own best, because the film is not flat and one focus position cannot serve
all of it. The aperture is the other end of that trade, and it pulls both ways
at once:

- **Stopping down deepens the focus.** Every region off the compromise plane
  is off it by however many drive steps the film's tilt and bow put it there,
  and a smaller opening turns less of that distance into blur. The regions
  that gave up the most are the ones this helps.
- **Stopping down blurs everything.** Diffraction at the opening is the same
  everywhere in the frame, so what it costs, it costs the regions that were
  already sharp -- the ones the compromise served best.

So there is a *best* aperture, not a smallest one, and where it is depends on
how far apart the regions are in focus: a frame that lies nearly flat is best
served wide open, and a badly curled one is worth stopping down for until
diffraction takes back more than depth gives. Nobody can read that off the
film. But the calibration already has the one instrument that can answer it --
a sharpness reading on each region -- so it can simply be tried.

**The regions are measured against the bests they already have.** Those were
read at the aperture the calibration ran at, and they are frozen once this
begins: the whole question is whether another aperture reads *higher* than
what was possible at this one, and a best that goes up with the reading can
answer nothing. A region can read over its best here, and that is the shape of
the answer -- a defocused region coming good as the depth grows -- just as a
region reading under its best at a small opening is diffraction, plainly
visible as a number.

**Whether any of that is visible depends on the magnification.** Live view is
a downscaled picture of the sensor, and diffraction at f/16 is finer than a
live-view pixel at full frame; magnified onto a region the way a calibration
reads one, it is several pixels across and reads plainly. This is only ever
run on the regions' own magnified views, which is where it means something.

**The exposure is compensated by the shutter**, so that the picture the
readings come off differs by the aperture and nothing else. A stop down is a
stop of shutter back, snapped to what the body offers, and what it could not
match exactly is kept as :attr:`Probe.residual` -- readings divide by the
square of the mean level, so a small mismatch comes out in the wash and a
large one would not, which is why a change the shutter cannot cover is the end
of that direction rather than something to make do with. A body whose shutter
is not ours to set -- aperture priority, where it meters for itself -- is
compensated by the body, and nothing is sent.

The search itself is :class:`ApertureSearch`: from the aperture the camera is
already set to, a step at a time, whichever way reads better, until it stops
getting better or it runs out of ladder. Nothing here is a model of a lens;
the numbers are all read off the film in front of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "BETTER",
    "DEFAULT_STOPS",
    "MOST_RESIDUAL",
    "MOST_STOPS",
    "STOP_NAMES",
    "STOP_SIZES",
    "ApertureSearch",
    "Ladder",
    "Next",
    "Probe",
    "stops_between",
]

#: How big a step to take between apertures, in EV. Whole stops are the
#: default because the answer is broad -- a third of a stop either side of the
#: best reads within a per cent of it -- and every step costs a probe, which
#: is a pan and a stack of frames for every region.
STOP_SIZES = {"whole": 1.0, "half": 0.5, "third": 1.0 / 3.0}

#: The same, as the panel puts them.
STOP_NAMES = {
    "whole": "Whole stops",
    "half": "Half stops",
    "third": "Third stops",
}

DEFAULT_STOPS = "whole"

#: How far either way from the aperture it started at the search may wander,
#: in EV. Four stops is the whole useful range of a macro lens on a copy
#: stand -- from wide open to where diffraction has taken back everything
#: depth ever gave -- and a bound is wanted because each step is a probe.
MOST_STOPS = 4.0

#: How much better a reading has to be for the search to keep going the same
#: way, as a fraction. Apertures are far apart compared with focus increments,
#: so a step that gains nothing measurable has nothing more to give: half a
#: per cent is well clear of what an integrated stack's grain moves the
#: average by, and well under what a stop of depth is worth on a curled frame.
BETTER = 0.005

#: How far out the exposure may be left when the shutter cannot match a change
#: of aperture exactly, in EV. Within a third of a stop the readings still
#: compare -- they divide by the square of the mean level, which is what a
#: brightness difference mostly is -- and past it the grain and the body's own
#: tone curve start telling as well, so that direction is called exhausted
#: instead.
MOST_RESIDUAL = 0.35

#: PTP's "shutter speed is Bulb", which is no use for a live-view exposure.
_BULB = 0xFFFFFFFF


def stops_between(one: int, other: int) -> float:
    """How many EV apart two apertures are, positive when *other* is smaller.

    Both in PTP's hundredths of an f-stop, as the body reports them. The
    opening's area goes as the square of the f-number, so a stop is a factor
    of the square root of two in the number engraved on the ring.
    """
    if one <= 0 or other <= 0:
        return 0.0
    return 2.0 * math.log2(other / one)


class Ladder:
    """The apertures and shutter speeds this body offers, and the sums on them.

    Built from what the camera itself lists rather than from a table of
    standard values: which apertures exist depends on the lens, and how finely
    they are spaced depends on a custom setting on the body.

    An empty *shutters* means the shutter is not ours to set -- aperture
    priority, where the body meters for itself -- and then :meth:`compensate`
    answers that nothing need be sent, because the body is already doing it.
    """

    def __init__(
        self,
        apertures: "Sequence[tuple[int, str]]",
        shutters: "Sequence[tuple[int, str]]" = (),
    ) -> None:
        self._apertures, self._labels = _ordered(apertures)
        self._shutters, self._shutter_labels = _ordered(shutters, skip=(_BULB,))

    @property
    def apertures(self) -> "tuple[int, ...]":
        """Every aperture the body offers, widest first."""
        return self._apertures

    @property
    def shutters(self) -> "tuple[int, ...]":
        """Every shutter speed it offers, quickest first; empty when it meters."""
        return self._shutters

    @property
    def compensates(self) -> bool:
        """Whether the body is left to compensate the exposure itself."""
        return not self._shutters

    def label(self, aperture: int) -> str:
        return self._labels.get(int(aperture), f"f/{aperture / 100:g}")

    def shutter_label(self, shutter: int) -> str:
        return self._shutter_labels.get(int(shutter), "")

    def holds(self, aperture: int) -> bool:
        return int(aperture) in self._labels

    def next_aperture(self, at: int, direction: int, size: float) -> "int | None":
        """The aperture about *size* EV from *at*, stopping down for *direction* 1.

        The nearest the body has to that, which is the honest answer to "one
        stop down" on a lens that only offers whole stops as well as on one
        that offers thirds -- how far it actually went is :attr:`Probe.stops`,
        and that is read off the aperture rather than assumed.
        """
        beyond = [
            value
            for value in self._apertures
            if (value > at if direction > 0 else value < at)
        ]
        if not beyond:
            return None
        return min(beyond, key=lambda value: abs(abs(stops_between(at, value)) - size))

    def compensate(self, shutter: int, change: float) -> "tuple[int, float] | None":
        """The shutter that puts back *change* EV of light, and what it left over.

        *change* is how much light the aperture lost, so it is positive
        stopping down and the shutter goes the same way -- longer. The answer
        is ``(speed, residual)`` with the residual in EV, positive for a
        picture left brighter than it was. ``(0, 0.0)`` when the body is
        metering for itself, and None when the ladder has nothing near enough:
        the end of the ladder, which is the end of the search that way.
        """
        if self.compensates:
            return 0, 0.0
        if shutter <= 0 or shutter == _BULB:
            return None
        wanted = shutter * (2.0**change)
        nearest = min(self._shutters, key=lambda speed: abs(math.log2(speed / wanted)))
        residual = math.log2(nearest / wanted)
        if abs(residual) > MOST_RESIDUAL:
            return None
        return nearest, residual


def _ordered(
    pairs: "Sequence[tuple[int, str]]", skip: "tuple[int, ...]" = ()
) -> "tuple[tuple[int, ...], dict[int, str]]":
    """The values a setting offers in order, with the body's own labels for them."""
    labels: "dict[int, str]" = {}
    for value, label in pairs or ():
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number <= 0 or number in skip:
            continue
        labels[number] = str(label)
    return tuple(sorted(labels)), labels


@dataclass(frozen=True)
class Probe:
    """One aperture tried, and what every region read there."""

    #: The aperture, in PTP's hundredths of an f-stop, and how it is written.
    aperture: int
    label: str
    #: How far it is from the aperture the search started at, in EV. Positive
    #: is stopped down.
    stops: float = 0.0
    #: The shutter it was read at and what the body calls it, or 0 when the
    #: body was left to meter for itself.
    shutter: int = 0
    shutter_label: str = ""
    #: How far out the exposure was left, in EV, because the shutter ladder
    #: had nothing exact. Positive is brighter than the start.
    residual: float = 0.0
    #: Every usable region's share of its best, in the calibration's order,
    #: and those combined the way the objective says. A share over one is a
    #: region reading better than anything the starting aperture could give it.
    shares: "tuple[float, ...]" = ()
    score: float = 0.0


@dataclass(frozen=True)
class Next:
    """What to set on the camera before the next probe."""

    aperture: int
    #: The shutter to set with it, or 0 to leave the body metering.
    shutter: int = 0
    residual: float = 0.0
    #: Whether this is the last one: the aperture the search chose, set so
    #: that the regions are read and pictured where the camera is being left.
    confirming: bool = False


class ApertureSearch:
    """Walk the aperture ladder to the opening that serves the regions best.

    Like everything else here it is asked what to do next each time a reading
    arrives and answers with somewhere to go, so that the worker keeps
    grabbing frames in between.

    It stops down first, because that is the direction a compromise usually
    wants, and turns back to the wide side when stopping down stops paying.
    Either leg ends when a step fails to better the leg by :data:`BETTER`,
    when the ladder runs out, when the exposure can no longer be compensated,
    or at :data:`MOST_STOPS` from where it began. Unlike the focus searches
    there is no play to reason about: an aperture is set, not driven, so any
    of them can be gone back to exactly.
    """

    def __init__(
        self,
        ladder: Ladder,
        aperture: int,
        shutter: int = 0,
        *,
        size: float = STOP_SIZES[DEFAULT_STOPS],
        objective: str = "average",
        most: float = MOST_STOPS,
        better: float = BETTER,
    ) -> None:
        self._ladder = ladder
        self._start = int(aperture)
        self._start_shutter = int(shutter)
        self._size = max(float(size), 0.01)
        self._objective = objective
        self._most = float(most)
        self._better = float(better)
        self._probes: "list[Probe]" = []
        #: ``down`` while it is stopping down, ``up`` on the way back out,
        #: ``confirm`` while it reads the aperture it chose, then ``done``.
        self._state = "down"
        #: The aperture the next step is measured from: the last one probed on
        #: this leg, and the starting one again when a leg turns round.
        self._from = self._start
        self._leg_best = 0.0
        #: Where the next probe is, and how it is exposed.
        self._at = self._start
        self._shutter = self._start_shutter
        self._residual = 0.0
        #: Which directions ended because there was no more ladder, rather
        #: than because the readings said so.
        self._exhausted: "set[int]" = set()

    # -- what it is doing --------------------------------------------------

    @property
    def objective(self) -> str:
        """Which way the regions' shares were combined into the score it climbs."""
        return self._objective

    @property
    def state(self) -> str:
        return self._state

    @property
    def done(self) -> bool:
        return self._state == "done"

    @property
    def aperture(self) -> int:
        """The aperture the next reading is to be taken at."""
        return self._at

    @property
    def shutter(self) -> int:
        """The shutter that goes with it, or 0 when the body is metering."""
        return self._shutter

    @property
    def probes(self) -> "tuple[Probe, ...]":
        return tuple(self._probes)

    @property
    def start(self) -> int:
        return self._start

    @property
    def chosen(self) -> "Probe | None":
        """The best aperture tried; the wider one when two read the same.

        Wider on a tie because a tie is two readings that cannot be told
        apart, and the wider of them is the quicker exposure and the one with
        less diffraction still to come if anything about the subject changes.
        """
        if not self._probes:
            return None
        return max(self._probes, key=lambda probe: (probe.score, -probe.aperture))

    @property
    def outcome(self) -> str:
        """``found``, ``bounded``, ``single``, or ``running``.

        ``bounded`` means the best it found is the furthest it was let go in
        that direction -- the ladder ran out, the exposure could not be
        compensated any further, or :data:`MOST_STOPS` stopped it -- so the
        real best may lie beyond it. ``single`` is a body that would not offer
        it a second aperture to try.
        """
        if not self.done:
            return "running"
        if len(self._probes) < 2:
            return "single"
        chosen = self.chosen
        for direction in self._exhausted:
            furthest = max(
                (probe for probe in self._probes if _towards(probe.stops, direction)),
                key=lambda probe: abs(probe.stops),
                default=None,
            )
            if furthest is not None and furthest.aperture == chosen.aperture:
                return "bounded"
        return "found"

    # -- walking it --------------------------------------------------------

    def step(self, score: float, shares: "Sequence[float]" = ()) -> "Next | None":
        """Keep what this aperture read; answer where to go next, or None if done.

        The reading is of the aperture in :attr:`aperture`, which is where the
        camera was put for it. None means the search is over and the camera is
        standing at :attr:`chosen`.
        """
        if self.done:
            return None
        self._probes.append(
            Probe(
                aperture=self._at,
                label=self._ladder.label(self._at),
                stops=stops_between(self._start, self._at),
                shutter=self._shutter,
                shutter_label=self._ladder.shutter_label(self._shutter),
                residual=self._residual,
                shares=tuple(float(share) for share in shares),
                score=float(score),
            )
        )
        if self._state == "confirm":
            return self._finish()
        if score > self._leg_best * (1.0 + self._better):
            self._leg_best = float(score)
            return self._carry_on()
        return self._turn()

    def _carry_on(self) -> "Next | None":
        """One more step the way this leg is going, or stop if there is no more.

        A leg that runs out of ladder while it is still improving has said
        everything the other direction had to say: the readings rose on the
        way here, so the way back is downhill and worth no probes. The one
        exception is a leg that never got a step in at all -- the aperture it
        started at is the end of the ladder -- and then the other way is the
        only way there is.
        """
        going_on = self._at != self._start
        self._from = self._at
        moved = self._move(1 if self._state == "down" else -1)
        if moved is not None:
            return moved
        return self._confirm() if going_on else self._turn()

    def _turn(self) -> "Next | None":
        """The readings say this leg is over: try the other one, or stop."""
        if self._state == "up":
            return self._confirm()
        self._state = "up"
        self._from = self._start
        # The wide side is judged against the aperture it began at, which is
        # the one reading both legs share.
        self._leg_best = self._probes[0].score if self._probes else 0.0
        moved = self._move(-1)
        return moved if moved is not None else self._confirm()

    def _move(self, direction: int) -> "Next | None":
        """Step *direction* from where this leg has got to, if it is allowed to.

        None for every way a leg can run out: no more apertures, past
        :data:`MOST_STOPS`, or a change of light the shutter cannot put back.
        Each of those is remembered, because the best found at the end of one
        is a bound and not an answer.
        """
        where = self._ladder.next_aperture(self._from, direction, self._size)
        if where is None:
            self._exhausted.add(direction)
            return None
        stops = stops_between(self._start, where)
        if abs(stops) > self._most + 1e-9:
            self._exhausted.add(direction)
            return None
        compensated = self._ladder.compensate(self._start_shutter, stops)
        if compensated is None:
            self._exhausted.add(direction)
            return None
        shutter, residual = compensated
        self._at, self._shutter, self._residual = where, shutter, residual
        return Next(aperture=where, shutter=shutter, residual=residual)

    def _confirm(self) -> "Next | None":
        """Go back and read the aperture it chose, where the camera is to be left.

        Only when it is somewhere else: a search that walked past its best
        ends one step beyond it, and the report's pictures should be of the
        aperture the camera keeps -- read at it, rather than assumed from the
        step before.
        """
        chosen = self.chosen
        if chosen is None or chosen.aperture == self._at:
            return self._finish()
        self._state = "confirm"
        self._at = chosen.aperture
        self._shutter = chosen.shutter
        self._residual = chosen.residual
        return Next(
            aperture=chosen.aperture,
            shutter=chosen.shutter,
            residual=chosen.residual,
            confirming=True,
        )

    def _finish(self) -> None:
        self._state = "done"
        return None


def _towards(stops: float, direction: int) -> bool:
    return stops > 0 if direction > 0 else stops < 0
