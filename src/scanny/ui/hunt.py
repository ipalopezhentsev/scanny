"""Driving focus to the top of the sharpness reading: contrast autofocus, ours.

The camera's own contrast autofocus only hunts inside its focus box, which is
324 sensor pixels across. This hunts the number in :mod:`scanny.ui.sharpness`
instead, which means it hunts wherever the measured area has been put -- and
that can be as small as the subject is.

**Nothing here counts steps.** Focus is driven by making steps and watching
what the reading does, never by remembering that the best reading was so many
steps back and driving that far. The reason is the lens: focus gearing has
play in it, so the same number of steps moves the optics differently
depending on which way they were last driven, and a reversal moves nothing at
all until the play is taken up. A search that navigates by step count has to
know how much play there is and drive past every target and back to take it
up -- a great deal of machinery, all of it to make a number mean something it
does not naturally mean. Watching the reading instead needs none of it: the
play shows up as steps where nothing happens, and the walk simply keeps
walking until something does.

There is one algorithm here, :class:`Walk`, and it has no loop of its own: it
is asked what to do next each time a trustworthy reading arrives, and answers
with the steps to drive.
"""

from __future__ import annotations

__all__ = ["Walk"]

#: How much better a reading has to be before it counts as better.
#:
#: It has to clear the wander in a steady reading, or the walk chases noise --
#: but every bit above that is focus error it will not correct, because it
#: stops as soon as a step no longer clears the bar. Against a modelled focus
#: curve, two per cent left it a fine step short of focus on a broad peak
#: almost every time, where one per cent finds it.
_IMPROVEMENT = 0.01

#: How many falling readings in a row it takes to turn a walk round. Judging
#: one reading against the one before it is the noisiest comparison there is,
#: and one fall is as likely to be the reading wandering as the lens going the
#: wrong way.
_FALLS_TO_TURN = 2

#: How near the best reading counts as being back at it, in multiples of the
#: improvement threshold. The best of a noisy walk is the luckiest reading of
#: it, and asking to see that exact number again would walk straight past.
_CLOSE_ENOUGH = 3.0

#: How many readings that say nothing at all before the step starts growing,
#: and how many increments it may grow to.
_PATIENT_STEPS = 2
_LONGEST_STRIDE = 4

#: When to give up on a direction that is saying nothing: whichever of these
#: comes first. The probe count is what stops a fine walk crawling for ever
#: through a lens's play; the distance is what stops a coarse one marching off
#: to the end of the travel, where a single step is already hundreds.
_PATIENCE_PROBES = 12
_PATIENCE_TRAVEL = 400

#: Bounds, because the body does not report the end of its travel: a lens
#: driven into its stop keeps answering "moved". About 6000 steps covers the
#: whole range on a D750.
_MAX_PROBES = 40
_MAX_TRAVEL = 3000


class Walk:
    """Step until the reading turns over, then walk back to the best of it.

    What a hand does, and the whole of what is done here:

    - step in one increment until the reading **rises**; if it falls instead,
      turn round and walk the other way;
    - keep going while it rises, remembering the best reading seen;
    - when it **turns over**, walk back until the reading is as good as that
      best one again, and stop there.

    Four things about it are worth keeping.

    **It comes back by reading, not by step count.** That is what makes it
    indifferent to the play in the gearing: coming back, the first steps take
    up the play and the picture does not move at all, and the walk simply
    continues until it does. Nothing has to be known about how much play there
    is, and nothing is driven past its target and back to take it up.

    **Rising and falling are judged against the previous reading, not the best
    one.** It sounds like a detail and it is the difference between working
    and not: every reading after the first is below the best, so a walk that
    asks "is this below the best?" answers yes to everything and gives up the
    moment it turns round.

    **A reading that has not changed is not a reading that got worse.** Play
    shows up as readings that are identical, and those are walked through;
    going the wrong way shows up as readings that fall. Telling the two apart
    keeps a wrong first guess at the direction down to a probe or two while
    still walking however far the play requires.

    **On the way out the step grows while nothing is happening, and drops back
    the instant it does.** Crawling through the play a minimum step at a time
    is a probe a second, every one of them reading exactly what the last one
    read. The way *back* never grows: the step that finally takes up the last
    of the play also moves the optics by whatever is left of it, so a long
    step there can carry the lens clean past the reading it came back for.
    Speed where nothing is changing, and never where something is.
    """

    def __init__(
        self,
        step: int,
        *,
        improvement: float = _IMPROVEMENT,
        max_probes: int = _MAX_PROBES,
        max_travel: int = _MAX_TRAVEL,
        patience: int = _PATIENCE_PROBES,
        patience_travel: int = _PATIENCE_TRAVEL,
    ) -> None:
        self._unit = abs(int(step)) or 1
        self._improvement = improvement
        self._max_probes = max_probes
        self._max_travel = max_travel
        self._patience = max(1, int(patience))
        self._patience_travel = max(self._unit, int(patience_travel))
        self._stride = self._unit
        self._direction = 1
        self._back = -1
        self._position = 0
        self._best = 0.0
        self._best_position = 0
        self._previous = 0.0
        self._probes = 0
        self._unchanged = 0
        self._unchanged_travel = 0
        self._falls = 0
        self._rose = False
        self._turned = False
        self._budget = 0
        self._state = "looking"
        self._outcome = ""
        self._confirmed: "float | None" = None
        #: Every reading taken, as (position, reading). Positions are counted
        #: from where the walk began, and are for saying afterwards what it
        #: did -- nothing is ever driven by them.
        self.trail: "list[tuple[int, float]]" = []

    # -- what happened -----------------------------------------------------

    @property
    def probes(self) -> int:
        return self._probes

    @property
    def best(self) -> float:
        return self._best

    @property
    def best_position(self) -> int:
        return self._best_position

    @property
    def position(self) -> int:
        return self._position

    @property
    def step_size(self) -> int:
        return self._unit

    @property
    def done(self) -> bool:
        return self._state == "done"

    @property
    def outcome(self) -> str:
        """``found``, ``nothing``, ``lost`` or ``exhausted`` once it is done."""
        return self._outcome

    @property
    def confirmed(self) -> "float | None":
        """The reading it stopped on, which is what it came back looking for."""
        return self._confirmed

    # -- the walk ----------------------------------------------------------

    def step(self, reading: float) -> "int | None":
        """Take a settled reading; answer with the steps to walk, or None."""
        self.trail.append((self._position, reading))
        if self._state == "done":
            return None
        self._probes += 1
        if self._probes == 1:
            self._best, self._best_position = reading, self._position
            self._previous = reading
            return self._walk(self._direction)
        if self._probes >= self._max_probes:
            return self._stop("exhausted")
        previous, self._previous = self._previous, reading
        if reading > self._best:
            self._best, self._best_position = reading, self._position
        if self._state == "returning":
            return self._coming_back(reading, previous)
        return self._going_out(reading, previous)

    def _going_out(self, reading: float, previous: float) -> "int | None":
        if reading > previous * (1 + self._improvement):
            # Going the right way. Keep going, and in the step asked for.
            self._unchanged = self._unchanged_travel = self._falls = 0
            self._stride = self._unit
            self._state = "climbing"
            return self._walk(self._direction)

        if reading < previous * (1 - self._improvement):
            self._falls += 1
            self._stride = self._unit  # something is happening: fine steps again
            if self._falls < _FALLS_TO_TURN:
                return self._walk(self._direction)  # one fall may be the noise
            if self._state == "climbing":
                return self._turn_back()  # over the top
            if not self._turned:
                self._about_turn()  # walking away from it
                return self._walk(self._direction)
            return self._turn_back()

        # Nothing changed: the gearing taking up its play, or nothing there.
        self._unchanged += 1
        if (
            self._unchanged < self._patience
            and self._unchanged_travel < self._patience_travel
            and abs(self._position) < self._max_travel
        ):
            self._lengthen()
            return self._walk(self._direction)
        if not self._turned:
            self._about_turn()
            return self._walk(self._direction)
        return self._turn_back()

    def _about_turn(self) -> None:
        self._turned = True
        self._unchanged = self._unchanged_travel = self._falls = 0
        self._stride = self._unit
        self._direction = -self._direction

    def _lengthen(self) -> None:
        """Take longer steps while the picture is not answering at all."""
        if self._unchanged >= _PATIENT_STEPS:
            self._stride = min(self._stride * 2, self._unit * _LONGEST_STRIDE)

    def _turn_back(self) -> "int | None":
        """Start walking back to the best reading of the way out."""
        if self._position == self._best_position:
            return self._stop("found" if self._best > 0.0 else "nothing")
        self._state = "returning"
        self._back = -self._direction
        self._rose = False
        self._stride = self._unit
        # However far out it went, plus the play it may have to take up again.
        self._budget = (
            -(-abs(self._position - self._best_position) // self._unit)
            + self._patience
        )
        return self._walk(self._back)

    def _coming_back(self, reading: float, previous: float) -> "int | None":
        """Walk back until the reading is as good as the best of the way out.

        Two ways of arriving, because the best reading of a noisy walk is the
        luckiest of them and may not come again: near enough to it, or having
        gone over the top of it and started down the far side.
        """
        self._budget -= 1
        if reading > previous * (1 + self._improvement):
            self._rose = True
        arrived = reading >= self._best * (1 - _CLOSE_ENOUGH * self._improvement)
        passed = self._rose and reading < previous * (1 - self._improvement)
        if arrived or passed:
            self._confirmed = reading
            self._best = max(self._best, reading)
            return self._stop("found")
        if self._budget <= 0:
            self._confirmed = reading
            return self._stop("lost" if self._best > 0.0 else "nothing")
        return self._walk(self._back)

    def _walk(self, direction: int) -> int:
        if self._state != "returning":
            self._unchanged_travel += self._stride
        self._position += direction * self._stride
        return direction * self._stride

    def _stop(self, outcome: str) -> None:
        self._outcome = outcome
        self._state = "done"
        return None
