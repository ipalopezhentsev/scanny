"""A depth map of the scene, read off the live view by sweeping focus once.

The sharpness meter answers one question about one rectangle: how much
contrast is in it now. Ask that question of every part of the picture at once,
at a series of focus positions, and each part answers with the position where
it was sharpest -- which is how far away the thing in it is. That is the whole
idea, and it is old: shape from focus.

**One sweep, not a hunt per zone.** The obvious construction is to point the
existing hunt at each zone in turn and write down where it stopped. It does
not work, for two reasons and both are fatal. A hunt costs twenty to forty
probes and every probe is a focus move and a settle, so a modest 16x9 grid is
some thousands of camera round trips -- the better part of an hour. Worse, a
hunt *moves the lens*, so by the time the second zone is measured the first
zone's answer describes a lens position nothing else was measured against.
Sweeping instead costs one probe per focus position no matter how many zones
there are, because every zone is read off the same frame, and every zone's
answer is in the same coordinate.

**The coordinate is drive steps from the near stop, and it is honest only
because nothing here ever reverses.** :mod:`scanny.ui.hunt` refuses to count
steps at all, and it is right to: focus gearing has play in it, so the same
step count moves the optics differently depending on which way they were last
driven. That argument is about *reversals*. Within one run in a single
direction the play was taken up by the first move and stays taken up, so
cumulative steps are a faithful, if not linear, stand-in for distance. So a
pass only ever drives one way, and every pass starts by parking the lens
against its near stop -- the one position a lens can be returned to exactly,
because it is a mechanical stop rather than a remembered number. That is what
makes the second pass's positions comparable with the first's.

The map is therefore in **steps from the near stop**, not in metres. Nothing
here knows the lens, so nothing here can turn steps into distance; what it can
say is which parts of the scene are nearer than which, and by how much in the
only unit available.

**Resolution costs nothing across the picture and everything along focus.**
Splitting the frame into four times as many zones is the same single pass over
the same array -- the numbers are sums, and a coarse zone's sums are the sums
of the fine zones inside it. So the picture is read at the *finest* useful
grid from the first frame, and every coarser grid is derived from it for free.
What actually costs time is focus positions: each one is a move and a wait for
the picture to settle, half a second or so. That is where coarse-then-fine
earns its keep, and it is the only axis this iterates on:

- the first pass walks the whole travel in big steps, which finds roughly
  where in the travel the scene lives but is far too coarse to place a zone;
- each pass after it sweeps only the part of the travel the last one found
  anything in, in steps several times finer.

**The grids are not just resolutions; the coarse ones vet the fine ones.** A
zone's reading is a mean over its pixels, so a ten-pixel zone's curve wanders
where a forty-pixel zone's is steady, and a fine zone that clears its grain at
one stop out of twenty has produced noise, not a peak -- and noise passes a
"does it stand above the rest of the curve" test easily, precisely because the
rest of the curve is noisy too. So coarse decides *where there is a subject*
and fine decides *where in the travel it peaked*: a fine zone is believed only
where the zone containing it found something as well, and where a fine zone
found nothing it borrows the containing zone's answer and is marked as having
done so. It also takes more than a single reading to make a curve: sharpness
against focus is continuous, so a peak with nothing either side of it is not
a subject coming into focus but a zone's grain having a good day -- and that
is the test that does the most work, because the wander it is catching is
not Gaussian and so cannot be thresholded away. Live view arrives as JPEG,
and the blocking in a flat area is real contrast that comes and goes with
the frame; what it does not do is land on two stops in a row.
Everything left over is drawn as blank, which is the honest answer for a patch
of clear sky or a blank wall: it has no contrast to peak, and the position of
the largest of its noise readings would draw as convincing terrain.

The tests above are all local to a zone, and there is a failure they cannot
see between them: a whole region of nothing agrees with itself. Every zone in
it wanders, every wander has a largest value, and no amount of looking at one
zone tells that largest value from a subject. So there is one test that is
not local -- a peak far below what the rest of the picture managed is not a
reading (:data:`WORTH`) -- and one floor under the grain that no measurement
of it may go below, because an eight-bit picture cannot be flatter than its
own levels (:data:`QUANTISATION`). The second matters more than it sounds:
the grain is measured from how much consecutive frames differ, and a
thoroughly defocused live view is smooth enough that JPEG returns almost the
same frame twice. The measured grain then falls towards zero and takes the
whole "nothing here" test with it.

Where the next pass should look is read off the coarsest grid for the same
reason, with the whole frame -- steadier again by two orders of magnitude in
pixel count -- as the last resort when not one coarse zone could place itself.

**Two things about lenses put a hard limit on how wide a sweep can be, and
neither can be corrected for from here.**

*Focus breathing.* A lens changes how big the picture is as it focuses, so a
zone is a rectangle of the *screen* and the scene slides underneath it as the
sweep runs. Over the minimum step that :mod:`scanny.ui.hunt` walks in this is
about a pixel and is rightly ignored; over a whole travel it is a good
fraction of the frame, worst at the edges and nothing at the centre. So a
zone near the edge of the frame is not looking at the same part of the scene
at both ends of a full-travel pass, and the peak it reports is the position
at which whatever happened to be in it then was sharpest.

*Bokeh.* A point of light out of focus is not a faint point, it is a large
bright disc, and as focus comes towards it the disc shrinks. Its *edge* is
contrast, and that edge sweeps across zones as the disc closes -- so a zone
with nothing of its own in it can read a rising, falling response from a
highlight that belongs to another part of the scene entirely, and place
itself at a depth that is not its own.

Both get worse the wider the range swept, and both are small over a narrow
one. That is what the first pass is *for*: it is reconnaissance, and its job
is to say which stretch of the travel the scene lives in, not to place
anything. Sweeping that stretch again -- a fresh survey over a fraction of
the travel -- is what produces a map worth trusting, and the panel offers it
as a second run rather than trying to correct a lens it knows nothing about.
A zone whose best reading sat at an end of what was swept is counted
separately for the same reason: what that says is not "it is furthest away"
but "the sweep did not contain its peak".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

from .pixels import green
from .sharpness import ABOVE_THE_GRAIN, SCALE

__all__ = [
    "BASE_COLUMNS",
    "DepthMap",
    "LEVELS",
    "MIN_TILE",
    "PROMINENCE",
    "QUANTISATION",
    "Sweep",
    "Survey",
    "Tiling",
    "colourise",
    "locate",
    "ramp_colour",
    "as_steps_image",
    "tile_sums",
    "tiling_for",
    "WANDER",
    "WORTH",
]

#: How many zones the coarsest grid is across. Sixteen is about the width at
#: which a zone of an ordinary live-view frame still has enough subject in it
#: to peak reliably, and it divides an ordinary frame's aspect ratio into whole
#: rows.
BASE_COLUMNS = 16

#: How many grids the pyramid has, the coarsest being :data:`BASE_COLUMNS`
#: across and each one after it twice as fine. Three levels over a 640x360
#: frame is 16x9, 32x18 and 64x36 -- ten-pixel zones at the bottom, which is
#: about as small as a gradient reading means anything.
LEVELS = 3

#: The smallest zone, in pixels on a side, that the finest grid may have. A
#: grid finer than the frame can support is dropped a level rather than
#: measured, because a zone of a handful of pixels reads almost pure grain.
MIN_TILE = 8

#: How far a zone's best reading has to stand above the worst it read during
#: the sweep before the peak counts as a peak. Without it, a zone with no
#: subject in it reports whichever of its noise readings happened to be the
#: largest, and a map of noise looks exactly like a map of terrain.
PROMINENCE = 0.25

#: The variance a picture has when it has nothing in it at all. Rounding to
#: whole levels puts a twelfth of a level of variance into every pixel, and
#: no eight-bit picture can be flatter than that, so the grain a reading is
#: judged against is never taken as less.
#:
#: It matters because the grain is otherwise *measured*, from how much two
#: consecutive frames differ -- and a thoroughly defocused live view is a
#: smooth one, which JPEG compresses into near-identical frames. The measured
#: grain then falls towards zero, and with it the whole test that says a zone
#: has nothing in it: every scrap of blocking left in a blank wall clears a
#: threshold of nothing, and the sweep reports confident depths for a part of
#: the travel where the scene is not even nearly in focus.
QUANTISATION = 1.0 / 12.0

#: How far below the rest of the picture a zone's best reading may be and
#: still be believed, as a fraction of what the better-lit ninth of the frame
#: managed.
#:
#: The other tests are all local to a zone, which is what lets a whole region
#: of nothing agree with itself: every zone in it wanders, every zone's wander
#: has a largest value, and nothing local can tell that largest value from a
#: subject. Set against the picture as a whole it is obvious -- a defocused
#: patch reads a hundredth of what the parts with a subject in them read, not
#: a half.
WORTH = 0.02

#: How far down the peak counts as still being the top of it, for the
#: estimator that locates a peak from every sample on its top rather than
#: from the three around its highest.
#:
#: Three points is the textbook answer and it is the wrong one whenever the
#: depth of field is broad, which on an ordinary lens it usually is. A broad
#: peak finely sampled has a summit twenty samples wide, all of them within
#: the noise of each other, so *which* one is highest is decided by the
#: grain -- and a fit through it and its two neighbours inherits all of that
#: and adds nothing. Two things sitting a hundred steps apart then come back
#: a few steps apart, in whichever order the noise chose. Averaging the
#: whole top instead divides that wander by the square root of how many
#: samples are on it.
_TOP = 0.5

#: How many samples have to be on the top before averaging them beats
#: fitting a parabola to the three around the highest. Below this the peak
#: is narrow against the step, and three points either side of it is the
#: better estimator; above it, it is much the worse one.
_ENOUGH_TOP = 4

#: How many standard deviations of its own wander a small zone's signal has
#: to clear before it is a reading rather than luck.
#:
#: :data:`scanny.ui.sharpness.ABOVE_THE_GRAIN` is a flat fraction, which is
#: the right test for an area of any size -- and it is nowhere near enough
#: for the small ones. The energy of a zone is the mean of its squared
#: differences, and the mean of *n* of those wanders by about the square
#: root of two over *n* of itself; over a forty-pixel zone that is under
#: three per cent and the flat fraction is the binding test, but over a
#: ten-pixel one it is ten per cent, so a flat twenty per cent is a two-sigma
#: event -- which, over a few thousand zones and a few dozen stops, happens
#: hundreds of times and puts a peak in every one of them.
WANDER = 4.0

#: The three planes kept per focus position: the sum of the levels, and the
#: sums of the squared differences to the right and downwards.
LEVEL, ACROSS, DOWN = 0, 1, 2


# -- how the picture is divided ----------------------------------------------


@dataclass(frozen=True)
class Tiling:
    """The finest grid of zones, and the crop of the picture it covers exactly.

    The picture is cropped rather than the grid stretched, so that every zone
    is the same size in pixels and a coarse zone is exactly the four fine
    zones inside it. A handful of pixels are lost from the edges; a depth map
    of the very edge of the frame is not what anyone is after.
    """

    levels: int
    #: The finest grid, in zones.
    rows: int
    cols: int
    #: One zone of the finest grid, in pixels.
    tile_height: int
    tile_width: int
    #: Where the cropped part of the picture starts.
    top: int
    left: int

    @property
    def height(self) -> int:
        return self.rows * self.tile_height

    @property
    def width(self) -> int:
        return self.cols * self.tile_width

    def factor(self, level: int) -> int:
        """How many fine zones on a side make up one zone at *level*."""
        return 1 << (self.levels - 1 - max(0, min(level, self.levels - 1)))

    def shape(self, level: int) -> "tuple[int, int]":
        """The grid at *level*, in zones. Level 0 is the coarsest."""
        step = self.factor(level)
        return self.rows // step, self.cols // step

    def tile(self, level: int) -> "tuple[int, int]":
        """One zone at *level*, in pixels."""
        step = self.factor(level)
        return self.tile_height * step, self.tile_width * step

    def covers(self, width: int, height: int) -> bool:
        """Whether a picture this size is the one this tiling was made for."""
        return self.left + self.width <= width and self.top + self.height <= height


def tiling_for(
    width: int, height: int, columns: int = BASE_COLUMNS, levels: int = LEVELS
) -> Tiling:
    """Divide a picture this size, as finely as its zones can still be read.

    The coarsest grid is *columns* across with as many rows as the aspect
    ratio asks for, and each level after it halves the zone. Levels are given
    up -- coarsest grid first -- until the finest zone is at least
    :data:`MIN_TILE` pixels on a side, so a small live-view frame simply gets
    a shallower pyramid rather than zones made of noise.
    """
    columns = max(1, int(columns))
    base_rows = max(1, round(columns * max(height, 1) / max(width, 1)))
    for count in range(max(1, int(levels)), 0, -1):
        cols = columns << (count - 1)
        rows = base_rows << (count - 1)
        tile_width, tile_height = width // cols, height // rows
        if tile_width >= MIN_TILE and tile_height >= MIN_TILE:
            break
    else:  # pragma: no cover - the loop above always binds on its last turn
        count, cols, rows = 1, columns, base_rows
        tile_width, tile_height = width // cols, height // rows
    # A picture too small even for the coarsest grid gets fewer zones instead
    # of zones too small to read.
    cols = max(1, min(cols, width // MIN_TILE))
    rows = max(1, min(rows, height // MIN_TILE))
    # Whatever the clamp left, the pyramid can only halve as far as both sides
    # stay whole.
    while count > 1 and (rows % (1 << (count - 1)) or cols % (1 << (count - 1))):
        count -= 1
    tile_width, tile_height = max(1, width // cols), max(1, height // rows)
    return Tiling(
        levels=count,
        rows=rows,
        cols=cols,
        tile_height=tile_height,
        tile_width=tile_width,
        top=(height - rows * tile_height) // 2,
        left=(width - cols * tile_width) // 2,
    )


def tile_sums(image: QImage, tiling: Tiling) -> np.ndarray:
    """What one frame contributes, as three sums over each of the finest zones.

    Sums rather than readings, because sums add: a coarse zone's three planes
    are the three planes of the fine zones inside it added together, so one
    pass over the frame at the finest grid is a pass at every grid.

    The two gradient planes are padded to the shape of the picture -- the last
    difference in each row and column is taken against itself and so is zero --
    which keeps every zone the same number of terms and lets the count be
    worked out from the zone's size rather than carried alongside it. The
    zeroes are not an approximation: :func:`readings` knows how many there are
    and takes exactly that much off the grain it subtracts.
    """
    pixels = green(image)
    pixels = pixels[
        tiling.top : tiling.top + tiling.height,
        tiling.left : tiling.left + tiling.width,
    ]
    if pixels.shape != (tiling.height, tiling.width):
        raise ValueError("the picture is not the size this tiling was made for")
    across = np.diff(pixels, axis=1, append=pixels[:, -1:])
    down = np.diff(pixels, axis=0, append=pixels[-1:, :])
    return np.stack(
        [
            _zone_sums(pixels, tiling),
            _zone_sums(across * across, tiling),
            _zone_sums(down * down, tiling),
        ]
    )


def _zone_sums(plane: np.ndarray, tiling: Tiling) -> np.ndarray:
    return (
        plane.astype(np.float64)
        .reshape(tiling.rows, tiling.tile_height, tiling.cols, tiling.tile_width)
        .sum(axis=(1, 3))
    )


def coarsen(sums: np.ndarray, tiling: Tiling, level: int) -> np.ndarray:
    """The same sums over the grid at *level*, by adding fine zones together."""
    step = tiling.factor(level)
    if step == 1:
        return sums
    rows, cols = tiling.shape(level)
    return sums.reshape(sums.shape[0], rows, step, cols, step).sum(axis=(2, 4))


def readings(
    sums: np.ndarray, tiling: Tiling, level: int, noise_variance: float = 0.0
) -> np.ndarray:
    """Turn the sums into one sharpness reading per zone of the grid at *level*.

    The same number :func:`scanny.ui.sharpness.measure` produces, on the same
    scale, worked out per zone: the gradient energy with the grain taken off
    it, over the square of the mean level. A zone whose signal does not clear
    its own grain reads zero, which is what says "nothing here to focus on"
    rather than "focus is wherever the noise was loudest". How far it has to
    clear the grain by depends on how big the zone is, because a small zone's
    reading wanders where a large one's does not (:data:`WANDER`), and the
    grain itself is never taken as less than the picture's own quantisation,
    however quiet the frames it was measured from looked (:data:`QUANTISATION`).
    """
    tall, wide = tiling.tile(level)
    counted = tall * wide
    totals = coarsen(sums, tiling, level)
    level_mean = totals[LEVEL] / counted
    energy = (totals[ACROSS] + totals[DOWN]) / counted
    # White noise of variance v puts 2v into each squared difference. The
    # padding zeroed one difference per row and per column of the zone, so
    # what it puts into these sums is that much short of the 4v that a full
    # pair of gradient planes would carry.
    variance = max(noise_variance, QUANTISATION)
    grain = 2.0 * variance * ((1.0 - 1.0 / wide) + (1.0 - 1.0 / tall))
    signal = energy - grain
    # How far above the grain this zone's signal has to be before it counts as
    # a reading: the flat fraction, or the wander of a zone this small,
    # whichever asks for more. See WANDER.
    terms = 2 * counted - tall - wide
    margin = max(ABOVE_THE_GRAIN, WANDER * float(np.sqrt(2.0 / max(terms, 1))))
    out = np.zeros_like(signal)
    real = (signal > margin * grain) & (signal > 0.0) & (level_mean > 0.0)
    np.divide(SCALE * signal, level_mean * level_mean, out=out, where=real)
    return np.where(real, out, 0.0)


# -- one pass along the travel -----------------------------------------------


class Sweep:
    """Where one pass stops, and how far to drive to reach the next stop.

    It has no loop of its own, in the same way :class:`scanny.ui.hunt.Walk`
    has none: it is told that a sample has been taken and answers with the
    steps to drive, so the caller's frame grab keeps running throughout.

    It only ever drives one way. A pass that turned round would have to know
    how much play the gearing has before its step counts meant anything, and
    the whole point of parking against the stop first is to not have to know.
    """

    def __init__(self, start: int, step: int, samples: int) -> None:
        self._start = int(start)
        self._step = max(1, abs(int(step)))
        self._samples = max(2, int(samples))
        self._taken = 0
        self._position = int(start)
        self._done = False
        self._stopped = False

    @property
    def start(self) -> int:
        return self._start

    @property
    def step(self) -> int:
        return self._step

    @property
    def samples(self) -> int:
        """How many stops the pass is planned to make."""
        return self._samples

    @property
    def taken(self) -> int:
        """How many stops it has made so far."""
        return self._taken

    @property
    def position(self) -> int:
        """Where the lens is, in drive steps from the near stop."""
        return self._position

    @property
    def done(self) -> bool:
        return self._done

    @property
    def stopped(self) -> bool:
        """Whether the pass ended against the end of the lens's travel."""
        return self._stopped

    def took_one(self) -> "int | None":
        """A sample was taken here; answer with the steps to the next stop."""
        if self._done:
            return None
        self._taken += 1
        if self._taken >= self._samples:
            self._done = True
            return None
        self._position += self._step
        return self._step

    def keep_going(self, more: int) -> None:
        """Take *more* stops than were planned, having reached the end of them.

        Only ever forward, which is the whole reason it can be done at all: a
        pass that has not yet found the peak it was aimed at is already
        driving the right way, so carrying on adds stops in the coordinate it
        was already keeping. Reaching back the other way would be a reversal,
        and a reversal takes up the play in the gearing -- which is exactly
        what parking against the stop exists to avoid having to know about.
        """
        if self._done:
            return
        self._samples += max(0, int(more))

    def blocked(self) -> None:
        """The lens would not go any further: this pass is over."""
        self._done = True
        self._stopped = True


# -- everything that was read ------------------------------------------------


class Survey:
    """The sums read at every focus position, from however many passes.

    Kept as sums rather than as readings so that the map can be worked out at
    any grid, and re-worked at another one, without sweeping again -- which is
    what lets the panel offer a coarse map and a fine one from the same
    minute of driving.
    """

    def __init__(self, tiling: Tiling) -> None:
        self._tiling = tiling
        self._samples: "dict[int, tuple[np.ndarray, float]]" = {}

    @property
    def tiling(self) -> Tiling:
        return self._tiling

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def positions(self) -> "tuple[int, ...]":
        return tuple(sorted(self._samples))

    def add(self, position: int, sums: np.ndarray, noise_variance: float = 0.0) -> None:
        """Record what was read at *position*, in steps from the near stop.

        Each sample carries the grain estimate that was current when it was
        taken, because that estimate moves with the light and with how many
        frames the picture was stacked from, and a reading is only comparable
        with another once its own grain has come off it.
        """
        self._samples[int(position)] = (
            np.asarray(sums, dtype=np.float64),
            float(noise_variance),
        )

    def curve(self, level: int) -> "tuple[np.ndarray, np.ndarray]":
        """The positions swept, and the reading at each one for every zone.

        Answers ``(positions, readings)`` where readings is
        ``(samples, rows, cols)`` over the grid at *level*.
        """
        order = self.positions
        rows, cols = self._tiling.shape(level)
        stack = np.zeros((len(order), rows, cols))
        for index, position in enumerate(order):
            sums, noise = self._samples[position]
            stack[index] = readings(sums, self._tiling, level, noise)
        return np.array(order, dtype=float), stack

    def whole_frame(self) -> "tuple[np.ndarray, np.ndarray]":
        """The same, for the picture read as a single zone.

        The steadiest curve the pyramid has, by two orders of magnitude in
        pixel count over the coarsest grid, and so the last resort for
        pointing the next pass when no zone anywhere could place itself.
        """
        order = self.positions
        tall, wide = self._tiling.height, self._tiling.width
        counted = tall * wide
        out = np.zeros(len(order))
        for index, position in enumerate(order):
            sums, noise = self._samples[position]
            totals = sums.sum(axis=(1, 2))
            mean = totals[LEVEL] / counted
            energy = (totals[ACROSS] + totals[DOWN]) / counted
            variance = max(noise, QUANTISATION)
            grain = 2.0 * variance * ((1.0 - 1.0 / wide) + (1.0 - 1.0 / tall))
            signal = energy - grain
            if mean > 0.0 and signal > 0.0 and signal > ABOVE_THE_GRAIN * grain:
                out[index] = SCALE * signal / (mean * mean)
        return np.array(order, dtype=float), out

    def map(self, level: int = LEVELS - 1) -> "DepthMap":
        """Where every zone of the grid at *level* was sharpest.

        Built coarse first and refined downwards, each grid vetted by the one
        above it: see :meth:`DepthMap.against`. A zone that could not find a
        peak of its own takes the answer of the zone containing it and is
        marked as having borrowed it, which keeps a blank patch inside a
        textured subject from punching a hole in the map; a zone whose
        containing zone found nothing is dropped whatever it thought it saw,
        which keeps a region with no subject in it from drawing as terrain.
        """
        level = max(0, min(int(level), self._tiling.levels - 1))
        positions = self.positions
        if len(positions) < 2:
            rows, cols = self._tiling.shape(level)
            empty = np.full((rows, cols), np.nan)
            return DepthMap(
                depth=empty,
                strength=np.zeros((rows, cols)),
                borrowed=np.zeros((rows, cols), bool),
                edge=np.zeros((rows, cols), bool),
                swept=(float(positions[0]) if positions else 0.0,) * 2,
                level=level,
                tiling=self._tiling,
            )

        coarse: "DepthMap | None" = None
        for step in range(level + 1):
            here = self._peaks(step)
            if coarse is not None:
                here = here.against(coarse)
            coarse = here
        assert coarse is not None
        return coarse

    def _peaks(self, level: int) -> "DepthMap":
        where, stack = self.curve(level)
        depth, strength, edge, _width = locate(
            where, stack, against_the_frame=True
        )
        return DepthMap(
            depth=depth,
            strength=strength,
            borrowed=np.zeros(strength.shape, bool),
            edge=edge,
            swept=(float(where[0]), float(where[-1])),
            level=level,
            tiling=self._tiling,
        )

    def interesting(self, margin: int) -> "tuple[int, int] | None":
        """The part of the travel worth sweeping again, with *margin* either side.

        Read off the coarsest grid, whose curves are the steadiest of the
        pyramid, and off the whole frame when not one coarse zone could place
        itself. The whole frame is held to the same test as a zone: a curve
        with no peak in it says the travel has nothing in it worth sweeping
        again, and answering with the top of a flat noisy line would point the
        next pass at a stretch chosen by the grain.
        """
        found = self._peaks(0)
        depths = found.depth[np.isfinite(found.depth)]
        if depths.size == 0:
            where, curve = self.whole_frame()
            peak, floor = float(curve.max()), float(curve.min())
            if peak <= 0.0 or peak - floor < PROMINENCE * peak:
                return None
            depths = where[[int(curve.argmax())]]
        low = int(np.floor(depths.min())) - int(margin)
        high = int(np.ceil(depths.max())) + int(margin)
        return low, high


def locate(
    where: np.ndarray, stack: np.ndarray, against_the_frame: bool = False
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Where each curve in *stack* peaked, how strongly, whether at an end,
    and how wide the top of it was.

    *where* is the focus positions swept, and *stack* is ``(len(where), ...)``
    -- one curve per zone of a grid, or per point someone put on the picture.
    Answers ``(position, strength, at_the_limit, width)``, with ``NaN`` for a
    curve that has no peak worth reporting. The width is how much travel the
    top of the curve covers, which is what says whether a finer sweep of a
    narrower stretch would see anything at all: a pass that fits inside the
    top of a peak is looking at a flat noisy line.

    Three things have to hold before there is an answer: the curve read
    something above its grain, that reading stands above the rest of the
    curve, and the peak has **shoulders** -- the stops either side of it read
    something as well. The last does the most work of the three. Sharpness
    against focus is a continuous thing: a subject coming into focus lights up
    the stops around the best one too, where a patch of blank wall lights one
    stop out of twenty and nothing either side of it. The wander that gets a
    reading over its grain once is not Gaussian and cannot be thresholded away
    -- live view is JPEG, and a flat area's blocking is real contrast that
    comes and goes with the frame -- but it does not oblige by landing on two
    stops in a row.

    *against_the_frame* adds a fourth test that is not local to one curve: a
    peak far below what the rest of the picture managed is a zone with nothing
    in it that has agreed with itself. It is right for a grid, where the zones
    are just a division of the frame and most of them may hold nothing, and
    wrong for a handful of places a person pointed at, where one of them being
    much fainter than the others is a fact about the scene and not a reason to
    disbelieve it. See :data:`WORTH`.
    """
    best = stack.argmax(axis=0)
    peak = np.take_along_axis(stack, best[None], axis=0)[0]
    floor = stack.min(axis=0)
    count = len(where)
    below = np.take_along_axis(stack, np.clip(best - 1, 0, count - 1)[None], 0)
    above = np.take_along_axis(stack, np.clip(best + 1, 0, count - 1)[None], 0)
    # A peak at either end of the sweep has one neighbour, not two.
    shoulder = np.where(
        best == 0,
        above[0] > 0.0,
        np.where(
            best == count - 1,
            below[0] > 0.0,
            (below[0] > 0.0) & (above[0] > 0.0),
        ),
    )
    found = (peak > 0.0) & (peak - floor >= PROMINENCE * peak) & shoulder
    if against_the_frame:
        standing = peak[found] if found.any() else peak[peak > 0.0]
        if standing.size:
            found = found & (peak >= WORTH * float(np.quantile(standing, 0.9)))
    position = np.where(found, where[best], np.nan)
    # Two estimators, and which one is better depends on how much of the
    # curve is summit. See _TOP.
    middle, width, on_top, truncated = _top_of(where, stack, floor, peak)
    broad = (on_top >= _ENOUGH_TOP) & ~truncated & np.isfinite(middle)
    refined = _vertex(where, stack, best)
    position = np.where(found & np.isfinite(refined), refined, position)
    position = np.where(found & broad, middle, position)
    at_the_limit = found & (((best == 0) | (best == count - 1)) | truncated)
    return position, np.where(found, peak, 0.0), at_the_limit, width


def _top_of(
    where: np.ndarray, stack: np.ndarray, floor: np.ndarray, peak: np.ndarray
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """The middle and the width of the top of each curve, and how it was found.

    The top is everything within :data:`_TOP` of the way down from the peak to
    the lowest the curve got, and the middle of it is those samples' centroid,
    weighted by how far above that line they stand and by how much travel each
    of them stands for. The spacing matters because passes are merged: a coarse
    pass's samples are hundreds of steps apart and a fine one's are single
    figures, and without it the fine stretch would outvote the rest of the
    curve simply by being crowded.

    Also answers how many samples were on the top, and whether it ran off
    either end of what was swept -- a truncated top has a centroid pulled
    towards the middle of the sweep, which is a lie, so the caller uses the
    three-point fit there instead and says the peak was at the limit.
    """
    height = np.maximum(peak - floor, 1e-12)
    line = floor + _TOP * height
    above = np.clip(stack - line, 0.0, None)
    shape = (len(where),) + (1,) * (stack.ndim - 1)
    span = _spacing(where).reshape(shape)
    place = where.reshape(shape)
    weight = above * span
    total = weight.sum(axis=0)
    middle = np.full(total.shape, np.nan)
    real = total > 0.0
    np.divide((weight * place).sum(axis=0), total, out=middle, where=real)
    width = (span * (above > 0.0)).sum(axis=0)
    on_top = (above > 0.0).sum(axis=0)
    truncated = (above[0] > 0.0) | (above[-1] > 0.0)
    return middle, width, on_top, truncated


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


def _vertex(where: np.ndarray, stack: np.ndarray, best: np.ndarray) -> np.ndarray:
    """Where the peak really is, from the three readings around the best one.

    The sweep only stopped where it stopped, so a zone's answer is otherwise
    quantised to the step -- which on a coarse pass is most of the depth of
    the scene. Three points either side of a maximum fix a parabola, and its
    top is a better answer than the middle sample by about the amount the
    curve is not a straight line. Zones peaking at an end of the pass have no
    three points and keep the sample they had.
    """
    count = len(where)
    if count < 3:
        # Two stops is not three points, and while it is being swept a map is
        # published from as few as two.
        return np.full(stack.shape[1:], np.nan)
    middle = np.clip(best, 1, count - 2)
    interior = (best > 0) & (best < count - 1)
    x0, x1, x2 = where[middle - 1], where[middle], where[middle + 1]
    y0 = np.take_along_axis(stack, (middle - 1)[None], axis=0)[0]
    y1 = np.take_along_axis(stack, middle[None], axis=0)[0]
    y2 = np.take_along_axis(stack, (middle + 1)[None], axis=0)[0]
    left, right = x1 - x0, x1 - x2
    top = left * left * (y1 - y2) - right * right * (y1 - y0)
    bottom = left * (y1 - y2) - right * (y1 - y0)
    out = np.full(y1.shape, np.nan)
    usable = interior & (np.abs(bottom) > 1e-12)
    np.divide(top, 2.0 * bottom, out=out, where=usable)
    out = np.where(usable, x1 - out, np.nan)
    # A fit that lands outside the two samples either side is not a refinement
    # of anything; it is three noisy points making a shape they should not.
    inside = (out >= np.minimum(x0, x2)) & (out <= np.maximum(x0, x2))
    return np.where(inside, out, np.nan)


@dataclass(frozen=True)
class DepthMap:
    """Where each zone was sharpest, in drive steps from the near stop."""

    #: The answer, per zone. NaN where the zone had nothing to answer with.
    depth: np.ndarray
    #: The reading at the peak: how much subject the answer rests on.
    strength: np.ndarray
    #: Zones that took a coarser zone's answer for want of one of their own.
    borrowed: np.ndarray
    #: Zones whose best reading was at one end of what was swept. What that
    #: says is not that they are the furthest thing in the scene but that the
    #: sweep did not contain their peak -- which is also what a defocused
    #: zone reading a bokeh disc from somewhere else looks like.
    edge: np.ndarray
    #: The stretch of travel this was read from, in steps.
    swept: "tuple[float, float]"
    level: int
    tiling: Tiling

    @property
    def known(self) -> np.ndarray:
        return np.isfinite(self.depth)

    @property
    def coverage(self) -> float:
        """The fraction of the picture that could answer at all."""
        return float(self.known.mean()) if self.depth.size else 0.0

    @property
    def range(self) -> "tuple[float, float]":
        """The nearest and furthest answers, in steps from the near stop."""
        known = self.depth[self.known]
        if known.size == 0:
            return self.swept
        return float(known.min()), float(known.max())

    def against(self, coarser: "DepthMap") -> "DepthMap":
        """This map, held to what the zones of *coarser* found.

        Two things at once, and they are the same rule from either side. A
        zone with no answer takes the answer of the zone containing it, so a
        blank patch inside a subject is filled rather than punched out. A zone
        *with* an answer keeps it only if the zone containing it found
        something too -- because a small zone's curve wanders, and a wander
        that clears the grain once looks exactly like a peak, while the zone
        around it has sixteen times the pixels and a quarter of the wander to
        say whether there was ever anything there.
        """
        scale = self.depth.shape[0] // coarser.depth.shape[0]
        if scale < 1:
            return self
        spread = np.kron(coarser.depth, np.ones((scale, scale)))
        spread_edge = np.kron(coarser.edge, np.ones((scale, scale))).astype(bool)
        vouched = np.isfinite(spread)
        missing = ~self.known & vouched
        return DepthMap(
            depth=np.where(vouched, np.where(missing, spread, self.depth), np.nan),
            strength=np.where(vouched, self.strength, 0.0),
            borrowed=self.borrowed | missing,
            edge=np.where(missing, spread_edge, self.edge) & vouched,
            swept=self.swept,
            level=self.level,
            tiling=self.tiling,
        )

    def describe(self) -> str:
        """One line saying how much answered, over what depth, out of what."""
        swept = self.swept[1] - self.swept[0]
        if not self.known.any():
            return (
                f"Nothing in the picture could be focused on anywhere in the "
                f"{swept:.0f} steps swept"
            )
        near, far = self.range
        rows, cols = self.depth.shape
        borrowed = int(self.borrowed.sum())
        limited = int((self.edge & self.known).sum())
        note = f", {borrowed} filled in" if borrowed else ""
        note += f", {limited} at the limit of the sweep" if limited else ""
        return (
            f"{cols}x{rows} zones, {self.coverage:.0%} answered{note}; "
            f"{far - near:.0f} steps from nearest to furthest, out of "
            f"{swept:.0f} swept"
        )


# -- drawing it --------------------------------------------------------------

#: Turbo, at seven stops. A depth map wants a scale where neighbouring depths
#: are obviously different colours rather than one perceptually smooth ramp:
#: the question asked of it is "is that nearer than this", and a rainbow
#: answers that at a glance where a greyscale does not.
_RAMP = (
    (48, 18, 59),
    (70, 134, 251),
    (39, 203, 158),
    (170, 232, 58),
    (249, 177, 50),
    (223, 64, 17),
    (122, 4, 3),
)

#: What a zone with no answer is drawn as: darker than any of the ramp, and
#: obviously not part of it.
_UNKNOWN = (34, 34, 38)


def colourise(depth_map: DepthMap, width: int = 0, height: int = 0) -> QImage:
    """The map as a picture: near at one end of the ramp, far at the other.

    Scaled to the span the map actually covers rather than to the whole of the
    travel that was swept, because most of a sweep is usually empty air either
    side of the scene and scaling to it would put every zone within a shade of
    the same colour. Zones that borrowed a coarser answer are drawn dimmer, so
    that what was measured is told apart from what was filled in.
    """
    rows, cols = depth_map.depth.shape
    near, far = depth_map.range
    span = far - near
    fraction = np.zeros((rows, cols))
    known = depth_map.known
    if span > 0:
        fraction[known] = (depth_map.depth[known] - near) / span
    rgb = _ramp(fraction)
    rgb[depth_map.borrowed & known] = (
        rgb[depth_map.borrowed & known] * 0.62
    ).astype(np.uint8)
    rgb[~known] = _UNKNOWN
    buffer = np.empty((rows, cols, 4), np.uint8)
    # An RGB32 buffer is blue, green, red, unused in memory order; see pixels.
    for plane, channel in enumerate((2, 1, 0)):
        buffer[:, :, channel] = rgb[..., plane]
    buffer[:, :, 3] = 255
    image = QImage(
        buffer.tobytes(), cols, rows, cols * 4, QImage.Format.Format_RGB32
    ).copy()
    if width > 0 and height > 0:
        # Nearest neighbour on purpose: the map has the resolution it has, and
        # a smoothed one claims detail between zones that was never measured.
        image = image.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
    return image


def as_steps_image(depth_map: DepthMap) -> QImage:
    """The map as sixteen-bit greyscale, for reading the numbers back out.

    Level 0 means "no answer here"; everything else runs from 1 at the nearest
    answer to 65535 at the furthest, so a reader who knows the two ends -- and
    they are in the file's name -- gets the drive-step position back.
    """
    rows, cols = depth_map.depth.shape
    near, far = depth_map.range
    span = far - near
    values = np.zeros((rows, cols), np.uint16)
    known = depth_map.known
    if span > 0:
        scaled = 1.0 + 65534.0 * (depth_map.depth[known] - near) / span
    else:
        scaled = np.full(int(known.sum()), 1.0)
    values[known] = np.clip(np.round(scaled), 1, 65535).astype(np.uint16)
    return QImage(
        values.tobytes(), cols, rows, cols * 2, QImage.Format.Format_Grayscale16
    ).copy()


def ramp_colour(fraction: float) -> "tuple[int, int, int]":
    """The colour a depth *fraction* of the way through the map's range is drawn.

    So that a legend can be drawn from the same ramp the picture uses, rather
    than from a second copy of it that has to be kept in step.
    """
    red, green, blue = _ramp(np.array(float(fraction)))
    return int(red), int(green), int(blue)


def _ramp(fraction: np.ndarray) -> np.ndarray:
    """The colour ramp, sampled at each of *fraction*'s values in 0..1."""
    stops = np.array(_RAMP, dtype=float)
    place = np.clip(fraction, 0.0, 1.0) * (len(stops) - 1)
    lower = np.clip(place.astype(int), 0, len(stops) - 2)
    weight = (place - lower)[..., None]
    mixed = stops[lower] * (1 - weight) + stops[lower + 1] * weight
    return np.clip(mixed, 0, 255).astype(np.uint8)
