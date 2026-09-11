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

It is still measured on the picture being *displayed*, because integrating
removes the noise for real rather than estimating it, and less noise left in
the picture is less that has to be reasoned about.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage

from ..camera.nikon import LiveViewFrame
from .pixels import green

__all__ = [
    "SharpnessMeter",
    "measure",
    "grain_reading",
    "variance_between",
    "Area",
    "ABOVE_THE_GRAIN",
    "SCALE",
    "MIN_AREA_PIXELS",
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


def measure(
    image: QImage, area: "Area | None" = None, noise_variance: float = 0.0
) -> float:
    """How much contrast *image* has, as a number to be maximised.

    With an *area*, only that part of the picture is read -- which is how a
    subject smaller than the frame gets focused on once the camera has run out
    of magnification. *noise_variance* is how much grain the picture has, from
    :func:`variance_between`; without it nothing is taken off.
    """
    pixels = green(image)
    if area is not None:
        pixels = _crop(pixels, area)
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
