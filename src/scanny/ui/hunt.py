"""Driving focus to the top of the sharpness reading: contrast autofocus, ours.

The camera's own contrast autofocus only hunts inside its focus box, which is
324 sensor pixels across. This hunts the number in :mod:`scanny.ui.sharpness`
instead, which means it hunts wherever the measured area has been put -- and
that can be as small as the subject is.

**Nothing here counts steps to drive by.** Focus is driven by making steps and
watching what the reading does, never by remembering that the best reading was
so many steps back and driving that far. The reason is the lens: focus gearing
has play in it, so the same number of steps moves the optics differently
depending on which way they were last driven, and a reversal moves nothing at
all until the play is taken up. A search that navigates by step count has to
know how much play there is and drive past every target and back to take it up
-- a great deal of machinery, all of it to make a number mean something it does
not naturally mean. Watching the reading instead needs none of it: the play
shows up as steps where nothing happens, and the walk simply keeps walking
until something does.

Step counts are kept, but only to *reason* about: which side of the best
reading a place was on, and how far the walk has wandered. Nothing is ever
driven to one.

**Nor does anything here take a longer step to save time.** Every step this
makes is the one increment it was given. A step that grows while the picture
is quiet sounds free -- there is nothing to lose by hurrying through a lens's
play -- but the step that finally takes up the last of that play is also the
one that moves the optics, by however much of it was left, and a long step
there walks straight over the peak. What that costs is not a slower search
but a worse answer, and a worse answer is the one thing this button exists
not to give.

**And nothing here gives up on a direction because it is not working.** A
reading that is getting worse does not mean the search has failed; it means
the other way. That is the whole content of :meth:`FineTune._turned`, and it
is what a hand does without thinking about it.

There is one algorithm here, :class:`FineTune`, and it has no loop of its own:
it is asked what to do next each time a trustworthy reading arrives, and
answers with a :class:`Move`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .homing import WayHome

__all__ = ["FineTune", "Move"]

#: How much better a reading has to be before it counts as better.
#:
#: It has to clear the wander in a steady reading, or the search chases noise
#: -- but every bit above that is focus error it will not correct, because it
#: stops climbing as soon as a step no longer clears the bar. Against a
#: modelled focus curve, two per cent left it a fine step short of focus on a
#: broad peak almost every time, where one per cent finds it.
_IMPROVEMENT = 0.01

#: How many falling readings in a row it takes to turn the walk round.
#: Judging one reading against the one before it is the noisiest comparison
#: there is, and one *small* fall is as likely to be the reading wandering as
#: the lens going the wrong way. A large one is not -- see :data:`_COLLAPSED`.
_FALLS_TO_TURN = 2

#: How far below the best it has seen a single reading has to fall before that
#: one reading is enough, and the walk turns round without waiting for another.
#:
#: Everything else here is careful about noise, and being careful about noise
#: is what let this walk somewhere absurd. Waiting for a second fall means
#: always taking one more step in a direction that has already got worse --
#: and on a subject with any depth to it that second step is the expensive
#: one: the reading goes over the top at the best it will ever read, is a
#: fifth of that one step later, and a twentieth the step after.
#:
#: A tenth is far outside anything a steady reading wanders by -- five times
#: the worst grain measured on an integrated stack -- so nothing is lost by
#: acting on the first one. And it is the number a person watching the trend
#: line uses: a reading a tenth below the best is not in focus, and no amount
#: of walking further the same way is going to help.
_COLLAPSED = 0.10

#: How far below the best of the current leg a reading has to sag before a
#: fall is allowed to mean anything at all.
#:
#: Because a fall means two different things depending on whether the optics
#: are moving. Crossing a lens's play they are not: the reading wanders around
#: one value and does not trend, and at one per cent of noise against a one
#: per cent threshold about one pair of readings in sixteen falls twice in a
#: row by luck alone. A walk that turns round on that never gets out of the
#: play -- it turns, crosses back, turns again, and settles somewhere in the
#: middle of it having learnt nothing.
#:
#: A reading that has sagged a twentieth below the best of the leg it is on is
#: not wandering, it is going downhill. Above that, falls are counted; below
#: it, they are the same thing as nothing having changed. Note what this is
#: measured against: the best of *this leg*, since the last time the walk
#: turned round. The best of the whole walk is no use here, because a walk on
#: its way back from far out starts every leg a long way below that.
_SAGGED = 0.05

#: How near the best reading counts as being back at it, in multiples of the
#: improvement threshold.
#:
#: This number used to be three, and three was the whole complaint: it stops
#: the walk back three per cent below the peak, which on a magnified macro
#: subject is a visible focus error and one anybody can better by hand with
#: the minimum increment. One means the walk back has to actually get back to
#: what it saw.
_BACK_ON_TARGET = 1.0

#: How much worse than the baseline counts as worse, in multiples of the
#: improvement threshold, before focus is put back where the camera had it.
#:
#: Wider than everything else here judges by, and deliberately. The baseline
#: is one reading and so is the one it is compared with, and a reading
#: wanders: on a subject where it wanders a couple of per cent, a band as
#: narrow as the improvement threshold calls a tie a loss about half the time,
#: and every one of those throws away a real improvement to go back to a
#: rougher answer. What the floor is for is the search having gone somewhere
#: genuinely worse, which is not a thing that happens by one per cent.
_CLEARLY_WORSE = 5.0

#: How far a direction is walked before it is abandoned for saying nothing --
#: to begin with. It **doubles** every time that direction is abandoned
#: without ever having shown the reading fall away, which is the whole of how
#: the walk reaches past its first idea of where to look.
#:
#: It has to grow, and that is what was missing. A fixed reach is a box drawn
#: round wherever the autofocus happened to stop: sixteen probes of the
#: minimum increment is ninety-odd drive steps on a lens with six thousand of
#: travel, so a peak a couple of hundred steps away sat outside it, and the
#: walk went back and forth inside the box a few times and announced it was
#: done. Doubling costs the ordinary case nothing -- a peak near the autofocus
#: is found inside the first reach and the growth never happens -- and it
#: means a peak twice as far away costs one more round rather than being
#: invisible.
_PATIENCE_PROBES = 16
_PATIENCE_TRAVEL = 400

#: How many times a walk that has **never read anything at all** may turn
#: round before it says there is nothing there.
#:
#: Two, which is its reach walked out and then back across to the far side of
#: where it started: the whole of the first reach, both ways, which is exactly
#: the looking a reading of nothing earns and no more. Going further only
#: finds more of the same, and it is not free -- a picture that cannot change
#: also never settles, so every probe of it waits the full settling limit.
#: Measured on a real calibration of five regions, one region of blank film
#: ran to the probe cap: **253 probes at two seconds each, nine minutes**
#: spent proving what its first fifty had already said, with four more regions
#: waiting behind it.
#:
#: It applies only to a walk that has read *nothing* -- not a small reading,
#: zero, which is :func:`scanny.ui.sharpness.measure` refusing to tell the
#: picture from its own grain. One reading above the grain anywhere and the
#: ordinary bounds take over, because then there is a hill to climb and the
#: growing reach is what finds it.
_NOTHING_TURNS = 2

#: How many times the walk may turn round while it is still exploring, and
#: again while it is on its way back.
#:
#: Reaching it does not stop the walk. Exploration that runs out of turns
#: **goes back to the best reading it saw** rather than stopping where it
#: stands -- which is the difference between a search that ends on its best
#: answer and one that ends on whichever direction it was last walking down.
_MOST_TURNS = 10

#: How many times the way home may be set off on again, from the stretch the
#: last way home walked, when it arrives somewhere the reading says is not
#: the top -- before the rest of the way is walked by the reading alone.
_HOMECOMINGS = 2

#: Bounds, because the body does not report the end of its travel: a lens
#: driven into its stop keeps answering "moved". About 6000 steps covers the
#: whole range on a D750, and the travel here is measured from the best
#: reading rather than from where the walk began.
#:
#: The probe count is generous on purpose. A walk that has to double its reach
#: two or three times to find the hill is spending a minute or two, and that
#: is the right trade: it is a minute against an answer that is not the best
#: one, on a subject that took longer than that to set up. The **Stop** button
#: is there, and the trend line shows what it is doing while it does it.
_MAX_PROBES = 300
_MAX_TRAVEL = 3000


@dataclass(frozen=True)
class Move:
    """What to do to the camera before the next reading is taken.

    Two things rather than one, because the search opens with the camera's own
    autofocus and drives the lens by hand for everything after that. Both
    arrive here so that the caller has one thing to carry out and the
    algorithm has one place it decides.
    """

    #: Signed drive steps: negative is nearer, positive further.
    steps: int = 0
    #: Run the camera's own autofocus, over the measured area.
    autofocus: bool = False


class FineTune:
    """Autofocus once, then walk to the best reading there is and stand on it.

    The whole of it:

    - **Autofocus on the measured area.** That is the neighbourhood, and its
      reading is the *baseline*: a floor the rest is not allowed to end below.
    - **Walk**, one increment at a time, keeping the best reading seen and
      where it was seen. While the reading rises, keep going. When it gets
      worse, **turn round** -- because a reading getting worse means the other
      way, and that is the only thing it can mean.
    - Once the reading has fallen away on **both sides of the best**, the best
      is a peak rather than a slope, and there is nothing left to explore.
    - **Go back to it** along the stretch it last walked in one direction,
      which crossed the peak: see :class:`~scanny.ui.homing.WayHome`. Only if
      that loses the way is the rest walked by the reading alone -- turning
      round whenever it gets worse, and stopping when it is back to what it
      was.
    - If what it ends on is worse than the baseline after all, **autofocus
      once more** and say so.

    Asked not to *come back*, it stops as soon as there is nothing left to
    explore, wherever that leaves the lens: for a calibration, which wants
    each region's best reading and walks the lens somewhere else next anyway.

    Three things about it are worth keeping.

    **A direction that is not working is never a reason to stop.** It used to
    be: legs that walked towards a reading had a probe budget, and running out
    of it ended the whole search wherever it happened to be standing -- which,
    if the direction was the wrong one, was as far from focus as it had
    managed to get. The reading getting worse is information, not failure, and
    what it says is *go the other way*.

    **There is exactly one autofocus, at the start.** There used to be a second
    one to reset between searching one way and the other, and it was a mistake:
    autofocus on the same patch does not land on the same place twice, so
    everything the search had learnt about which way things lay was worthless
    the moment it ran. One walk that turns itself round needs no datum to
    return to and no second opinion about where it is.

    **Rising and falling are judged against the previous reading, not the best
    one.** It sounds like a detail and it is the difference between working
    and not: every reading after the first is below the best, so a walk that
    asks "is this below the best?" answers yes to everything and turns round on
    the spot, for ever.
    """

    def __init__(
        self,
        step: int,
        *,
        improvement: float = _IMPROVEMENT,
        first: int = 1,
        autofocus: bool = True,
        max_probes: int = _MAX_PROBES,
        max_travel: int = _MAX_TRAVEL,
        patience: int = _PATIENCE_PROBES,
        patience_travel: int = _PATIENCE_TRAVEL,
        most_turns: int = _MOST_TURNS,
        come_back: bool = True,
    ) -> None:
        self._unit = abs(int(step)) or 1
        #: Whether it goes back to stand on the best once it has found it, or
        #: stops where the finding left it.
        self._come_back = bool(come_back)
        self._improvement = improvement
        #: Which way the walk sets off. Nothing recommends one over the other
        #: -- it turns round of its own accord -- so it is a parameter for the
        #: tests rather than a decision.
        self._direction = 1 if first >= 0 else -1
        self._autofocus = bool(autofocus)
        self._max_probes = max_probes
        self._max_travel = max_travel
        self._patience = max(1, int(patience))
        self._patience_travel = max(self._unit, int(patience_travel))
        #: How far each direction is walked before it is abandoned for saying
        #: nothing, in probes. Per direction and growing, because a direction
        #: that said nothing has not been explored -- it has been glanced at.
        self._reach = {1: self._patience, -1: self._patience}
        self._most_turns = max(2, int(most_turns))

        self._state = "opening"
        self._position = 0
        self._probes = 0
        self._outcome = ""
        self._confirmed: "float | None" = None
        self._baseline = 0.0
        self._best = 0.0
        self._best_position = 0

        self._previous = 0.0
        self._falls = 0
        self._unchanged = 0
        self._unchanged_travel = 0
        self._turns = 0
        self._best_at_last_turn = 0.0
        self._rose = False
        #: Whether the walk is done exploring and is on its way back to the
        #: best reading it found.
        self._going_back = False
        #: The best reading since the last time it turned round.
        #:
        #: What "fallen off a cliff" is measured against, and it has to be
        #: this rather than the best of the whole walk. A walk coming back
        #: from far out starts every leg a long way below the best there is --
        #: that is what coming back *means* -- so judging a collapse against
        #: the best of the walk calls every step of the journey home a
        #: collapse, and turns the walk round on the spot, every time.
        self._leg_best = 0.0
        #: What it is walking back to, frozen when the walk turns for home.
        #:
        #: Frozen, and that matters. Letting it follow the best while the walk
        #: back is in progress -- carrying on because the reading is still
        #: rising and has already beaten what it set out for -- is the obvious
        #: improvement and it is wrong: the step that beats the target is
        #: usually the peak itself, so carrying on from there steps over it,
        #: and a step past a peak cannot be taken back without crossing the
        #: whole of the gearing's play again.
        self._target = 0.0
        #: Where in :attr:`trail` the stretch walked in one direction since the
        #: last reversal begins: the profile the way home is read off.
        self._leg_start = 0
        #: The way home, while it is being walked by the profile rather than
        #: by the reading alone.
        self._home: "WayHome | None" = None
        self._homecomings = _HOMECOMINGS
        #: Where the reading has been seen to fall away, in step counts from
        #: where the walk began. **Read, never driven to**: they answer one
        #: question, which is whether the best reading has been walked past on
        #: both sides or only one.
        self._fell_at: "list[int]" = []
        #: Every reading taken, as (position, reading). For saying afterwards
        #: what it did.
        self.trail: "list[tuple[int, float]]" = []

    # -- what happened -----------------------------------------------------

    @property
    def probes(self) -> int:
        return self._probes

    @property
    def best(self) -> float:
        """The best reading seen anywhere, which is what it walks back to."""
        return self._best

    @property
    def best_position(self) -> int:
        return self._best_position

    @property
    def baseline(self) -> float:
        """What the camera's own autofocus read: the floor this may not go under."""
        return self._baseline

    @property
    def coming_back(self) -> bool:
        """Whether it has finished looking and is going to stand on its best."""
        return self._going_back

    @property
    def turns(self) -> int:
        """How many times the reading getting worse sent it the other way."""
        return self._turns

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
        """``found``, ``nothing``, ``lost``, ``exhausted`` or ``restored``."""
        return self._outcome

    @property
    def confirmed(self) -> "float | None":
        """The reading it stopped on, which is what it walked back looking for."""
        return self._confirmed

    # -- the search --------------------------------------------------------

    def step(self, reading: float) -> "Move | None":
        """Take a settled reading; answer with what to do next, or None."""
        self.trail.append((self._position, reading))
        if self._state == "done":
            return None
        self._probes += 1
        # The cap is on the walking, not on putting focus back: a rescue
        # autofocus already asked for has to be allowed to finish.
        if self._state == "walking" and self._probes > self._max_probes:
            if not self._going_back:
                if not self._come_back:
                    return self._stop_here(reading, "exhausted")
                # Out of probes to explore with, but not out of the walk: it
                # still has to go and stand on the best reading it found.
                self._turn_for_home()
            elif self._probes > self._max_probes + self._max_probes // 2:
                return self._settle_on(reading, "exhausted")
        return {
            "opening": self._opening,
            "anchoring": self._anchoring,
            "walking": self._walking,
            "restoring": self._restoring,
        }[self._state](reading)

    # -- getting to a place worth improving on -----------------------------

    def _opening(self, reading: float) -> "Move | None":
        """The reading as the button was pressed, which is only a starting gun.

        The camera's own autofocus goes first, over the measured area rather
        than wherever its box happened to be left. It is a big box and a rough
        answer, which is exactly why it is the opening move and not the job --
        and why it happens once. It does not land on the same place twice, so
        running it again half way through would throw away everything the walk
        had learnt about which way things lie.
        """
        if not self._autofocus:
            return self._anchoring(reading)
        self._state = "anchoring"
        return Move(autofocus=True)

    def _anchoring(self, reading: float) -> "Move | None":
        """The reading the camera left behind: the baseline, and the floor.

        A reading of nothing here is not a reason to stop, and it used to be.
        Zero means the picture has no detail above its own grain, which is
        true of anything far enough out of focus -- including a subject whose
        focus is a couple of hundred steps from wherever the camera's own
        autofocus decided to stop. There is no hill to climb from here, but
        there is ground to cover, and the walk covers it: if the whole of its
        reach both ways is still nothing, *then* there is nothing to focus on,
        and that is what it says.
        """
        # Everything before the autofocus belongs to a focus position that
        # cannot be walked back to, so the best starts here.
        self._baseline = self._best = self._previous = reading
        self._leg_best = reading
        self._best_position = self._position
        self._leg_start = len(self.trail) - 1
        self._state = "walking"
        return self._drive()

    # -- the walk ----------------------------------------------------------

    def _walking(self, reading: float) -> "Move | None":
        """One settled reading, and what it says about where to step next."""
        if self._home is not None:
            return self._homing(reading)
        previous, self._previous = self._previous, reading
        if reading > self._best:
            self._best, self._best_position = reading, self._position

        if self._going_back and reading >= self._target * (
            1 - _BACK_ON_TARGET * self._improvement
        ):
            return self._settle_on(reading, "found")

        rising = reading > previous * (1 + self._improvement)
        falling = reading < previous * (1 - self._improvement)
        # How this reading stands against the best of the leg it is on, worked
        # out before that best is brought up to date -- otherwise every new
        # high compares itself with itself. Both of these are read further
        # down; they are up here so that the rise, which returns early, cannot
        # skip the bookkeeping and leave the bar where it was two steps ago.
        collapsed = reading < self._leg_best * (1 - _COLLAPSED)
        sagged = reading < self._leg_best * (1 - _SAGGED)
        self._leg_best = max(self._leg_best, reading)

        if rising:
            self._rose = True
            self._falls = self._unchanged = self._unchanged_travel = 0
            return self._drive()

        if (
            self._going_back
            and self._rose
            and falling
            and previous >= self._target * (1 - _SAGGED)
            and reading >= self._target * (1 - _COLLAPSED)
        ):
            # It climbed towards what it came back for and went over the top
            # of it without matching the number. The best reading of a noisy
            # walk is the luckiest of them and may not come again; this is as
            # near to it as stepping is going to get.
            #
            # Only when it is genuinely near, though, and both conditions are
            # doing real work. Crossing a lens's play the reading wanders, so
            # a rise of two per cent followed by a fall of three is ordinary
            # -- and without them it reads as "climbed to the target and
            # went over it" while standing anywhere in the play. The walk
            # turns for home once the reading has sagged a twentieth or so
            # below the best, so the play is crossed at around nine tenths of
            # the target: a test of the reading alone let a real calibration
            # stop there, seven per cent short of each region's best. The top
            # it went over has to have been near the target too.
            return self._settle_on(reading, "found")

        if collapsed:
            return self._turned(reading)  # off a cliff: one reading is enough

        if falling and sagged:
            # Going downhill, not merely wandering: the reading is below the
            # one before it *and* has sagged clear of the best this leg has
            # managed. Both, because inside a lens's play only the first of
            # those is ever true and turning round there is how a walk gets
            # stuck in the play instead of crossing it.
            self._falls += 1
            self._unchanged = self._unchanged_travel = 0
            if self._falls >= _FALLS_TO_TURN:
                return self._turned(reading)
            return self._drive()  # one fall may still be the reading wandering

        # Nothing that means anything: the gearing taking up its play, the
        # reading wandering, or nothing there at all. Keep walking, and how
        # far is measured **from the best reading, not from here**.
        #
        # That distinction is the whole of the reach. Counting quiet probes
        # from wherever the last leg happened to stop makes the legs cancel:
        # ninety steps out, ninety steps back, and the walk is where it began
        # having spent thirty probes. Measured from the best, the legs are a
        # bracket that opens -- sixteen increments either side of it, then
        # thirty-two, then sixty-four -- and a peak outside the first bracket
        # is inside the second.
        self._unchanged += 1
        beyond = abs(self._position - self._best_position)
        allowed = (
            self._max_travel
            if self._going_back
            else min(self._reach[self._direction] * self._unit, self._max_travel)
        )
        if beyond < allowed:
            return self._drive()
        return self._turned(reading)

    def _turned(self, reading: float) -> "Move | None":
        """The reading got worse, so go the other way.

        The one thing this must not do is stop. A direction that stops paying
        is not a search that has failed, it is a search that now knows which
        way to go -- and stopping there leaves the lens as far from focus as
        that direction managed to drag it, which is the worst place it visited
        rather than the best.

        What it does decide is whether there is any exploring left, and it is
        careful about that. Turning round is cheap and happens for all sorts of
        reasons -- a wobble in the reading, a direction that said nothing at
        all, the far side of the peak. Only the last of those is **evidence**,
        and only evidence ends a search.
        """
        proven = self._proven(reading)
        if proven:
            self._fell_at.append(self._position)
        elif not self._going_back:
            # This direction has been walked to the end of its reach without
            # ever showing the reading fall away, so its reach was too short.
            # Go twice as far in it.
            self._reach[self._direction] = min(
                self._reach[self._direction] * 2, self._max_travel // self._unit
            )

        if not self._going_back and (
            self._straddled() or self._turns >= self._most_turns
        ):
            if not self._come_back:
                return self._stop_here(reading, "found")
            self._turn_for_home()
            if self._set_off_home():
                return self._drive_home()
        elif self._going_back and self._turns >= self._most_turns + _MOST_TURNS:
            # Sent back and forth more than any single-humped reading can
            # account for: the reading is too unsteady to walk by.
            return self._settle_on(
                reading,
                "found"
                if reading >= self._best * (1 - _CLEARLY_WORSE * self._improvement)
                else "lost",
            )

        if not (proven or self._going_back) and self._fell_away(-self._direction):
            # This direction still owes an answer and the other one has
            # already given its own: the reading has been watched to fall away
            # over there, so there is nothing to go back for. Carry straight
            # on, further than last time, rather than walking the same ground
            # twice. Without this the two legs cancel each other out and the
            # bracket opens at half the rate for twice the probes.
            self._fresh_leg(reading)
            return self._drive()

        # Turns only count against the limit when the walk is getting nowhere.
        # The limit is there to catch a reading so unsteady that it sends the
        # walk back and forth for no reason; a walk that has found something
        # better since it last turned is not doing that, whatever the count.
        if self._best > self._best_at_last_turn:
            self._turns = 0
        else:
            self._turns += 1
        self._best_at_last_turn = self._best
        if (
            not self._going_back
            and self._best <= 0.0
            and self._turns >= _NOTHING_TURNS
        ):
            # The reach has been walked out and back across, and nothing
            # anywhere in it read above the grain. There is no hill here to
            # go further for; see _NOTHING_TURNS. It stops where it stands
            # rather than going home, because with no reading anywhere there
            # is no home to go to.
            return self._stop_here(reading, "nothing")
        self._reverse()
        self._fresh_leg(reading)
        return self._drive()

    def _fresh_leg(self, reading: float) -> None:
        """Start judging again from here: this is where this stretch begins."""
        self._falls = self._unchanged = self._unchanged_travel = 0
        self._rose = False
        self._leg_best = reading

    def _reverse(self) -> None:
        """Turn round: the reading just taken is where the next stretch starts."""
        self._direction = -self._direction
        self._leg_start = len(self.trail) - 1

    def _turn_for_home(self) -> None:
        """Stop exploring and go and stand on the best reading there was."""
        self._going_back = True
        self._target = self._best

    # -- the way home ------------------------------------------------------

    def _set_off_home(self) -> bool:
        """Turn round for home along the stretch just walked, if it has a top.

        The stretch since the last reversal was walked in one direction, and
        the walk turned because the reading had fallen away on the far side
        of the best -- so it went over the top, and its readings are an
        honest profile of it. False when they are not: too few, or no top
        standing clear of the grain, which leaves the way home to be walked by
        the reading alone.

        Or a top that is not the best: a stretch spent crossing a long play
        reads the same thing all the way along, give or take the grain, and
        the grain can make a hump in it that looks like a top. The best was
        somewhere else, and this is not the way to it. Not *exactly* the
        best, though: the best of a walk is its luckiest reading, and on a
        top one increment wide where the play leaves the increments across it
        decides how near the top any of them comes. A top within a tenth of
        the best is the top.
        """
        stretch = self.trail[self._leg_start :]
        home = WayHome.along(
            [where for where, _reading in stretch],
            [reading for _where, reading in stretch],
            self._unit,
        )
        if home is None or home.peak < self._best * (1 - _COLLAPSED):
            return False
        self._home = home
        self._reverse()
        return True

    def _homing(self, reading: float) -> "Move | None":
        """One reading on the way home, walked by the profile of the stretch.

        Arriving where the profile says the top is and reading what the top
        read there is home. Arriving and reading something else is a way
        home that was misjudged -- the play placed a step out on a top a
        step wide -- and the way just walked is an honest stretch in one
        direction of its own, so it is set off along again, from the other
        side. Only when that too has failed, or the way was lost altogether,
        is the rest of it walked by the reading alone.
        """
        home = self._home
        assert home is not None
        if reading > self._best:
            self._best, self._best_position = reading, self._position
        verdict = home.heard(reading)
        if verdict == "on":
            self._previous = reading
            return self._drive_home()
        if verdict == "home":
            if home.agrees(reading):
                return self._settle_on(reading, "found")
            if self._climbing_home(reading):
                # Arrived by the count, short of the top by the reading, and
                # still climbing towards it: the play was longer than it
                # looked. Carry on over the top; the stretch has one to go
                # back to once it has.
                self._previous = reading
                return self._drive_home()
            if self._homecomings > 0 and self._set_off_home():
                self._homecomings -= 1
                self._previous = reading
                return self._drive_home()
        # Walked by the reading from here, towards the best there was.
        self._home = None
        self._fresh_leg(reading)
        return self._walking(reading)

    def _climbing_home(self, reading: float) -> bool:
        """Whether this reading is the highest the way home has read, and a rise."""
        way = [value for _where, value in self.trail[self._leg_start :]]
        return (
            len(way) >= 2
            and reading >= max(way)
            and reading > self._previous * (1 + self._improvement)
        )

    def _drive_home(self) -> Move:
        assert self._home is not None
        self._home.drive()
        return self._drive()

    def _proven(self, reading: float) -> bool:
        """Whether this turn is evidence that the best has been walked past.

        Two things have to hold, and leaving either of them out is what let a
        search finish inside a box of its own drawing. The walk has to have
        been going **outward** -- away from the best reading, not back towards
        it across a lens's play, where every reading is below the best by
        construction. And the reading has to have **fallen clear of the best**,
        rather than merely wobbled or said nothing: a direction that said
        nothing is a direction not yet explored, and the answer to that is to
        go further, not to call it done.
        """
        outward = (self._position - self._best_position) * self._direction > 0
        fell_away = reading < self._best * (1 - _CLEARLY_WORSE * self._improvement)
        return outward and fell_away

    def _fell_away(self, direction: int) -> bool:
        """Whether the reading has been watched to fall away on that side."""
        return any(
            (where - self._best_position) * direction > 0 for where in self._fell_at
        )

    def _straddled(self) -> bool:
        """Whether the best reading has been walked past on both sides.

        The one place step counts are read. They are not trustworthy enough to
        drive to -- the play sees to that -- but they are quite trustworthy
        enough to say which side of something a place was on, which is all
        this asks.
        """
        return self._fell_away(1) and self._fell_away(-1)

    # -- ending no worse than it began -------------------------------------

    def _settle_on(self, reading: float, outcome: str) -> "Move | None":
        """Stop here, unless here is worse than the camera managed on its own."""
        self._confirmed = reading
        under = self._baseline * (1 - _CLEARLY_WORSE * self._improvement)
        if reading >= under or not self._autofocus:
            return self._stop(outcome)
        self._state = "restoring"
        return Move(autofocus=True)

    def _restoring(self, reading: float) -> "Move | None":
        self._confirmed = reading
        return self._stop("restored")

    def _stop_here(self, reading: float, outcome: str) -> None:
        """Stop without going back: what was wanted was the best, not to stand on it."""
        self._confirmed = reading
        return self._stop(outcome)

    # -- driving -----------------------------------------------------------

    def _drive(self) -> Move:
        self._unchanged_travel += self._unit
        self._position += self._direction * self._unit
        return Move(steps=self._direction * self._unit)

    def _stop(self, outcome: str) -> None:
        # A walk that never saw anything above the grain has nothing to report
        # but that, whichever way it ran out.
        self._outcome = "nothing" if self._best <= 0.0 else outcome
        self._state = "done"
        return None
