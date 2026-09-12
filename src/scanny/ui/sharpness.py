"""Measuring how sharp the picture on screen is, for focusing by hand.

A lens is in focus when the picture has the most contrast in it, which is what
a camera's contrast-detect autofocus hunts for. A D750 will only hunt inside
its own focus box, and that box is 324 sensor pixels wide -- far bigger than
the things worth focusing on this carefully. So this measures the picture
instead: drive focus a step, watch the number, and keep the direction that
raises it. Magnifying first is what makes it selective, because then the whole
frame *is* the small thing being focused on.

The measure is the mean squared difference between neighbouring pixels -- the
classic gradient energy that contrast autofocus uses -- with the noise taken
off it, over the square of the mean level.

Both corrections earn their place:

- **Dividing by the mean squared** makes readings comparable while focusing.
  Gradients scale with brightness, so without it the light changing would move
  the reading as much as focus does.
- **Subtracting the noise** is what stops the reading being mostly about
  exposure. Grain is contrast, and a darker live view is a grainier one, so a
  stop of exposure moved the raw reading by about a third -- as much as a
  visible focus error. White noise adds a known amount to gradient energy,
  four times its variance, so given that variance it can simply be taken off.

**How much noise there is has to be measured over time, not within a frame.**
Estimating it from a single frame -- the usual trick, a kernel that cancels
anything smooth and leaves what changes from pixel to pixel -- cannot work
here, and fails in the worst possible way. At the point of best focus, the
finest detail in the picture *is* what changes from pixel to pixel: the
estimator reads the subject as grain, subtracts it, and the sharpest frame of
all reads zero while a blurred one reads well above it. The hunt then walks
away from focus, having been told that focus is the worst place it visited.

Two consecutive frames of a scene that is holding still, on the other hand,
differ by nothing but noise. So the caller passes in a variance measured that
way, and the smallest recently seen is the one used -- when the picture is
moving the frames differ by more than noise, and the smallest is the estimate
that has no movement in it. Nothing is subtracted when nobody has measured it.

**And it has to be the grain of what is being read** -- the measured area, in
the view it is read through -- which :class:`GrainMemory` keeps, one estimate
for each. Grain is not one number for the camera: it grows with how bright
the picture is, and it differs between magnifications. One estimate shared by
everything, the quietest pair of frames from whatever had been on screen,
was what calibrations used to subtract, and it made regions incomparable with
themselves: panning between four regions, the quietest pair came from the
darkest one, so the grain of a region under a third as bright was subtracted
from all four, and the bright ones read up to a tenth higher than they had
while fine tuned on their own -- with exactly the same detail in the picture.


It is still measured on the picture being *displayed*, because integrating
removes the noise for real rather than estimating it, and less noise left in
the picture is less that has to be reasoned about.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Hashable

import numpy as np
from PySide6.QtGui import QImage

from ..camera.nikon import LiveViewFrame
from .pixels import green

__all__ = [
    "GrainMemory",
    "SharpnessMeter",
    "measure",
    "grain_reading",
    "level_of",
    "pixels_of",
    "variance_between",
    "Area",
    "ABOVE_THE_GRAIN",
    "SCALE",
    "MIN_AREA_PIXELS",
    "QUANTISATION",
    "format_reading",
]

#: Readings are only meaningful against each other, so the scale is chosen to
#: put an ordinary subject in the tens rather than at three decimal places.
SCALE = 1000.0

#: An area as (x, y, width, height) fractions.
#:
#: Two coordinate systems wear this type, and which one is meant is worth
#: being careful about. :func:`measure` and :func:`grain_reading` take
#: fractions of the **displayed picture**, because that is what they have in
#: their hands. :class:`SharpnessMeter` is given fractions of the **whole
#: frame** -- a place on the sensor -- and turns them into the other kind for
#: whatever crop each frame happens to be showing. See
#: :meth:`SharpnessMeter.set_area` for why.
Area = "tuple[float, float, float, float]"

#: How far above the grain the signal has to be before it is a reading at all.
#: Below this the picture cannot be told from its own noise, and a number that
#: is really noise is worse than no number: a hunt will chase it, lock onto
#: whichever grain read highest, and drive somewhere with nothing in it.
ABOVE_THE_GRAIN = 0.2

#: An area smaller than this is not measured but clamped up to it. A handful
#: of pixels is all noise and no subject, and the reading from one jumps about
#: far too much to focus against.
MIN_AREA_PIXELS = 16

#: The variance a picture has when it has nothing in it at all. Rounding to
#: whole levels puts a twelfth of a level of variance into every pixel, and
#: no eight-bit picture can be flatter than that, so grain measured from how
#: much consecutive frames differ is never taken as less. A thoroughly
#: defocused live view is smooth enough that JPEG returns almost the same
#: frame twice, and a grain measured from those falls towards nothing.
QUANTISATION = 1.0 / 12.0


def format_reading(value: float) -> str:
    """A reading as text, with more decimals the smaller it is.

    Readings are only compared with each other, and the steps a fine tune
    walks in move them by a fraction of a per cent near the top of focus, so
    the digits that change from one step to the next are kept on screen.
    """
    return f"{value:.2f}" if abs(value) >= 100 else f"{value:.3f}"


def measure(
    image: QImage, area: "Area | None" = None, noise_variance: float = 0.0
) -> float:
    """How much contrast *image* has, as a number to be maximised.

    With an *area*, only that part of the picture is read -- which is how a
    subject smaller than the frame gets focused on once the camera has run out
    of magnification. *noise_variance* is how much grain the picture has, from
    :func:`variance_between`; without it nothing is taken off.
    """
    pixels = pixels_of(image, area)
    if pixels.size < 4:
        return 0.0
    level = float(pixels.mean())
    if level <= 0.0:
        return 0.0
    across = np.diff(pixels, axis=1)
    down = np.diff(pixels, axis=0)
    energy = float(np.mean(across * across) + np.mean(down * down))
    # What is left once the grain is accounted for. White noise of variance v
    # contributes four times it to the energy, whatever the subject is.
    grain = 4.0 * max(noise_variance, 0.0)
    signal = energy - grain
    if signal <= ABOVE_THE_GRAIN * grain:
        # Nothing here that can be told from the grain. Nothing is the honest
        # answer, and it is the useful one: a hunt reads it as "no hill here"
        # and goes back where it came from rather than chasing the noise.
        return 0.0
    return SCALE * signal / (level * level)


def grain_reading(
    image: QImage, area: "Area | None" = None, noise_variance: float = 0.0
) -> float:
    """What this picture would read with nothing in it but its own grain.

    Which is the scale to judge a small reading against. There is no absolute
    number that means "nothing to focus on" -- the reading has no units and its
    size depends on the subject and the light -- but a reading well below what
    the grain alone accounts for means there is nothing there.
    """
    if noise_variance <= 0.0:
        return 0.0
    pixels = green(image)
    if area is not None:
        pixels = _crop(pixels, area)
    level = float(pixels.mean())
    if level <= 0.0:
        return 0.0
    return SCALE * 4.0 * noise_variance / (level * level)


def pixels_of(
    image: "QImage | np.ndarray", area: "Area | None" = None
) -> np.ndarray:
    """The pixels :func:`measure` reads: the green of the part the area covers.

    *image* may be the green already got out of a picture, to save doing it
    twice.
    """
    pixels = image if isinstance(image, np.ndarray) else green(image)
    return _crop(pixels, area) if area is not None else pixels


def level_of(image: QImage, area: "Area | None" = None) -> float:
    """The mean level :func:`measure` divides by, for the record of a reading."""
    pixels = pixels_of(image, area)
    return float(pixels.mean()) if pixels.size else 0.0


def variance_between(earlier: np.ndarray, later: np.ndarray) -> "float | None":
    """The noise variance implied by two frames of a scene holding still.

    They differ by the noise on each of them, so the variance of the
    difference is twice the variance of one. Anything that actually moved
    between them counts as well, which is why the caller keeps the smallest of
    these it has seen rather than the latest.
    """
    if earlier.shape != later.shape or earlier.size == 0:
        return None
    difference = later - earlier
    return 0.5 * float(np.mean(difference * difference))


#: Where in the pairs of frames of one thing its grain is read off: a quarter
#: of the way up the sorted ones. See :meth:`GrainMemory.variance`.
_QUIETEST = 0.25


class GrainMemory:
    """How much grain each thing that is read has, each from its own frames.

    One estimate per *key* -- whatever the caller says makes two readings the
    same measurement: a view and the area read in it, or a calibration's
    region -- from the variances lately seen between consecutive frames of it,
    for the reason the module gives. Kept apart because grain is not one
    number: a darker picture has less of it, a different magnification a
    different amount, and the smallest pair of frames from everything that has
    been on screen is the grain of the quietest of them, not of the one being
    read.

    **Only pairs of frames that were both of a picture holding still** should
    be given to it: while the lens is moving, frames differ by the move as
    well, and see :meth:`variance` for why the estimate cannot simply be the
    smallest of whatever it is given.

    How many keys are kept is bounded: the oldest used is forgotten first.
    """

    def __init__(self, memory: int = 60, keys: int = 32) -> None:
        self._memory = max(1, int(memory))
        self._keys = max(1, int(keys))
        self._seen: "OrderedDict[Hashable, deque[float]]" = OrderedDict()

    def note(self, key: Hashable, variance: float) -> None:
        """One pair of consecutive frames of *key* differed by *variance*."""
        seen = self._seen.get(key)
        if seen is None:
            seen = self._seen[key] = deque(maxlen=self._memory)
            while len(self._seen) > self._keys:
                self._seen.popitem(last=False)
        else:
            self._seen.move_to_end(key)
        seen.append(float(variance))

    def variance(self, key: Hashable) -> "float | None":
        """The grain of *key*, or None if no pair of its frames has been seen.

        The quietest quarter of the pairs, not the quietest one. Both are
        ways of ignoring the pairs that have movement in them as well as
        noise, which read high and never low -- but the smallest of them
        depends on *how many* there are, since more tries find a lower one,
        and that is a difference between measurements that have nothing else
        between them. A calibration's compromise gathers three still pairs a
        probe where its fine tunes gather a handful per region, and its
        estimates came out a sixth lower for it: every region read a per cent
        or two higher in the compromise than in its own fine tune, which is
        exactly the thing kept apart here. A quarter of the way up the sorted
        pairs is as blind to movement and says the same thing however many
        there are.
        """
        seen = self._seen.get(key)
        if not seen:
            return None
        return float(np.quantile(np.fromiter(seen, float, len(seen)), _QUIETEST))

    def pairs(self, key: Hashable) -> int:
        """How many pairs of *key*'s frames the estimate is the smallest of."""
        seen = self._seen.get(key)
        return len(seen) if seen else 0

    def forget(self) -> None:
        """Forget everything, as when the exposure changed under it."""
        self._seen.clear()


def _crop(pixels: np.ndarray, area: "Area") -> np.ndarray:
    """The part of *pixels* the area covers, never smaller than it is useful."""
    height, width = pixels.shape
    x, y, w, h = area
    left = int(round(x * width))
    top = int(round(y * height))
    right = int(round((x + w) * width))
    bottom = int(round((y + h) * height))
    left, right = _span(left, right, width)
    top, bottom = _span(top, bottom, height)
    return pixels[top:bottom, left:right]


def _span(start: int, end: int, limit: int) -> "tuple[int, int]":
    """One axis of the crop: inside the picture, and wide enough to mean something."""
    size = min(max(end - start, MIN_AREA_PIXELS), limit)
    start = min(max(start, 0), limit - size)
    return start, start + size


class SharpnessMeter:
    """Measures displayed frames, and remembers the best reading so far.

    The best is what makes the number usable: focus by hand and the reading
    wanders, so what matters is not its value but whether it is at the top of
    what this view has managed. It is forgotten whenever the view changes,
    since a reading from a different crop is not the same measurement.
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = bool(enabled)
        self._area: "Area | None" = None
        self._value = 0.0
        self._peak = 0.0
        self._key: "tuple | None" = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def value(self) -> float:
        return self._value

    @property
    def peak(self) -> float:
        """The best reading since the view last changed."""
        return self._peak

    @property
    def area(self) -> "Area | None":
        """The part of the **frame** being read, or None for all of it."""
        return self._area

    def set_area(self, area: "Area | None") -> bool:
        """Read only this part of the frame; say whether that was a change.

        **Fractions of the whole frame, not of the picture on screen**, and
        that is the one thing about this class that was rebuilt rather than
        added to. Screen fractions look right until the view is magnified: a
        rectangle drawn round one letter of a caption at full frame stays a
        third of the way across whatever strip the camera shows at 18.8x,
        which is a different piece of the world entirely -- so the area jumps
        to somewhere nobody chose every time the magnification changes, and
        the readings either side of that have nothing to do with each other.

        Sensor coordinates cannot do that. They are what the camera's own
        focus point lives in, they do not move when the view magnifies or
        pans, and :meth:`scanny.camera.nikon.LiveViewFrame.area_normalised`
        turns them back into screen fractions for whichever crop is on show.
        What was a place on the screen is now a place on the subject, which is
        what someone drawing a box round their subject meant.

        The best reading goes with it: a different region is a different
        measurement, and its numbers have nothing to do with the old ones.
        """
        area = tuple(float(v) for v in area) if area is not None else None
        if area == self._area:
            return False
        self._area = area
        self._value = 0.0
        self._peak = 0.0
        return True

    def configure(self, enabled: bool) -> bool:
        """Switch measuring on or off; say whether that was a change."""
        if bool(enabled) == self._enabled:
            return False
        self._enabled = bool(enabled)
        self.reset()
        return True

    def reset(self) -> None:
        """Forget the best reading, as when the picture is no longer the same."""
        self._value = 0.0
        self._peak = 0.0
        self._key = None

    def shown_in(self, frame: LiveViewFrame) -> "Area | None":
        """Where the measured area falls on *frame*'s picture, in its fractions.

        ``None`` means the whole picture, exactly as it does everywhere else
        here -- so the one case that has to be told apart from it, the area
        being off screen altogether, is answered by
        :meth:`measure` reading nothing rather than by a third value.
        """
        return None if self._area is None else frame.area_normalised(self._area)

    def measure(
        self, frame: LiveViewFrame, image: QImage, noise_variance: float = 0.0
    ) -> "tuple[float, float] | None":
        """Read one displayed image; return ``(value, best)``, or ``None`` if off."""
        if not self._enabled:
            return None
        key = (_view_key(frame), self._area)
        if key != self._key:
            # A different crop is a different measurement, and its numbers are
            # nothing to do with the ones before it: the same piece of sensor
            # magnified further has its detail spread over more pixels.
            self._key = key
            self._peak = 0.0
        area = self.shown_in(frame)
        if area is None and self._area is not None:
            # The area is a place on the sensor and the view has been
            # magnified somewhere else. Reading the whole picture instead
            # would answer a question nobody asked, and answer it with a
            # number that looks just like the ones being compared.
            self._value = 0.0
            return self._value, self._peak
        self._value = measure(image, area, noise_variance)
        self._peak = max(self._peak, self._value)
        return self._value, self._peak


def _view_key(frame: LiveViewFrame) -> "tuple[int, ...]":
    """What has to hold still for two readings to be comparable."""
    return (
        frame.crop_center_x,
        frame.crop_center_y,
        frame.crop_width,
        frame.crop_height,
    )
