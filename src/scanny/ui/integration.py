"""Averaging consecutive live-view frames, to trade frame rate for clarity.

Sensor noise is different in every frame while the scene is not, so the mean of
N frames keeps the picture and shrinks the noise by roughly the square root of
N. Live view cannot be asked for a longer exposure -- the camera sends what it
sends, about 44 frames a second on a D750 and only 16 of them once the view is
magnified past 4.7x -- so the only currency available is frame rate: show the
mean of each N frames and the picture arrives N times less often, with visibly
less grain in it.

The N frames have to be N *different* frames for any of that to hold. Averaging
a frame with a copy of itself cancels nothing, because the copy carries the
same noise and it adds coherently rather than averaging down -- the mean of a
stack is the mean of its distinct frames however many times each was counted.
The worker polls faster than the camera draws and drops the re-reads, so what
arrives here is only ever new; nothing in this file re-checks it.

Two choices worth knowing about:

- **The mean is taken in the JPEG's own gamma-encoded values**, not in linear
  light. Averaging linearly is the physically correct way to add exposures,
  but it also lifts the shadows and changes how the picture looks. The point
  here is the same picture with less noise in it, so the arithmetic stays in
  the values the camera sent.
- **A frame of a view we have nothing to show for is shown immediately**, and
  starts a new stack. Magnifying or panning changes the crop the camera
  renders, and waiting out the rest of a stack before showing the new view
  would make every pan feel like it had frozen. This way the first frame of a
  new view appears at once -- noisy -- and the integrated version replaces it
  a stack later.

  A stack thrown away by :meth:`FrameIntegrator.reset` is not that. The view
  has not moved, so the picture already on screen is still a picture of it,
  and a far cleaner one than the single frame that would replace it. Those
  restarts fill quietly, and the last good image stays up. Telling the two
  apart is what :attr:`_last_key` is for: it outlives a reset, where
  :attr:`_key` does not.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage

from ..camera.nikon import LiveViewFrame
from .pixels import view

__all__ = [
    "FrameIntegrator",
    "MAX_FRAMES",
    "MIN_FRAMES",
    "DEFAULT_FRAMES",
    "source_fps",
]

#: Fewer than two frames is not an average, and 64 frames is already a second
#: and a half of integration at the camera's 44fps -- long enough that the view
#: has stopped being live in any useful sense.
MIN_FRAMES = 2
MAX_FRAMES = 64
DEFAULT_FRAMES = 8


class FrameIntegrator:
    """Accumulates decoded live-view frames and yields their mean.

    Feed it every frame with :meth:`add`. It answers with the image to show,
    or ``None`` while a stack is still filling up. Disabled, it simply decodes
    each frame and hands it straight back, so the caller has one path.
    """

    def __init__(self, enabled: bool = False, frames: int = DEFAULT_FRAMES) -> None:
        self._enabled = bool(enabled)
        self._frames = _clamp(frames)
        self._sum: "np.ndarray | None" = None
        self._count = 0
        self._key: "tuple[int, ...] | None" = None
        # The view the picture on screen belongs to. Unlike _key it survives a
        # reset, so a restarted stack can be told from a moved view.
        self._last_key: "tuple[int, ...] | None" = None
        self._last_whole = True
        self._last_frame: "QImage | None" = None

    # -- configuration -----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def frames(self) -> int:
        """How many frames go into one displayed image."""
        return self._frames

    @property
    def pending(self) -> int:
        """Frames gathered so far towards the current stack."""
        return self._count

    @property
    def last_frame(self) -> "QImage | None":
        """The last frame decoded, as it arrived and before any averaging.

        Kept because it has been decoded already: whoever wants to look at a
        single frame -- to see how much grain is on it, say -- should not have
        to decode the same JPEG a second time.
        """
        return self._last_frame

    @property
    def last_image_was_whole(self) -> bool:
        """Whether the last picture handed back was a full stack.

        The odd one out is the single frame shown the instant the view changes,
        which is deliberately not integrated. Anything reading the picture to
        make a decision about it -- the focus hunt does -- has to be able to
        tell that one apart from a finished stack.
        """
        return self._last_whole

    def configure(self, enabled: bool, frames: int) -> bool:
        """Set the mode, and say whether that was actually a change.

        The caller has work to do only if it was: an unchanged setting must
        not throw away a stack that is halfway to being an image.
        """
        wanted = (bool(enabled), _clamp(frames))
        if wanted == (self._enabled, self._frames):
            return False
        self._enabled, self._frames = wanted
        self.reset()
        return True

    def reset(self) -> None:
        """Throw away the part-filled stack, the view being unchanged.

        What is on screen stays there while the stack fills again, because it
        is still a picture of this view and a cleaner one than any single
        frame. Use :meth:`forget` where that is not true.
        """
        self._sum = None
        self._count = 0
        self._key = None

    def forget(self) -> None:
        """Throw away the stack *and* what is on screen, as when live view
        stops: the next frame has nothing to be continuous with, so it is
        shown as soon as it arrives rather than a stack later."""
        self.reset()
        self._last_key = None

    # -- integration -------------------------------------------------------

    def add(self, frame: LiveViewFrame) -> "QImage | None":
        """Take one frame; return the image to display, or ``None`` to wait."""
        image = QImage.fromData(frame.jpeg, "JPG")
        if image.isNull():
            return None
        image = image.convertToFormat(QImage.Format.Format_RGB32)
        self._last_frame = image
        if not self._enabled:
            self._last_whole = True
            return image

        key = _stack_key(frame, image)
        if key != self._key:
            moved = key != self._last_key
            self._last_key = key
            self._start(key, image)
            if not moved:
                # A restarted stack, not a new view. Leave the picture that is
                # already up: showing this one frame instead would replace a
                # clean image with a noisy one for no reason, which is exactly
                # what it looks like on screen -- a flash of grain.
                return None
            self._last_whole = False
            return image
        self._accumulate(image)
        if self._count < self._frames:
            return None
        self._last_whole = True
        return self._mean(image.width(), image.height())

    def _start(self, key: "tuple[int, ...]", image: QImage) -> None:
        self._key = key
        self._sum = np.zeros((image.height(), image.width(), 4), dtype=np.uint32)
        self._count = 0
        self._accumulate(image)

    def _accumulate(self, image: QImage) -> None:
        assert self._sum is not None
        np.add(self._sum, view(image), out=self._sum)
        self._count += 1

    def _mean(self, width: int, height: int) -> QImage:
        assert self._sum is not None
        # Round rather than truncate: over a long stack, always rounding down
        # would darken the picture by half a level.
        mean = ((self._sum + self._count // 2) // self._count).astype(np.uint8)
        self._sum[:] = 0
        self._count = 0
        # QImage does not own the buffer it is handed, and this one is about to
        # go out of scope, so the copy is what makes the image safe to emit.
        return QImage(
            mean.tobytes(), width, height, width * 4, QImage.Format.Format_RGB32
        ).copy()


def source_fps(zoom_level: int, exposure_preview: bool) -> float:
    """How fast the camera draws, under the two conditions that decide it.

    Measured on a D750, and a hint rather than a promise: it is here so the
    controls can say what a setting will cost before it is switched on, while
    the status bar goes on reporting the rate actually arriving.

    Magnification dominates. Out to 3.13x the body draws 44 frames a second,
    or 30 with the exposure preview off; from 4.7x up it draws 16 whatever
    else is set, the preview included. The live-view frame size does not come
    into it -- 320x180 is drawn at exactly the rate 640x360 is, at every
    magnification -- so asking for the small frame buys detail away for
    nothing, and is not a way to go faster.
    """
    if zoom_level >= _SLOW_ZOOM_LEVEL:
        return 16.0
    return 44.0 if exposure_preview else 30.0


#: The magnification at which the body's draw rate falls away. Level 3 is
#: 3.13x and holds 44fps; level 4 is 4.7x and does not.
_SLOW_ZOOM_LEVEL = 4


def _clamp(frames: int) -> int:
    return max(MIN_FRAMES, min(int(frames), MAX_FRAMES))


def _stack_key(frame: LiveViewFrame, image: QImage) -> "tuple[int, ...]":
    """What has to stay the same for frames to belong in the same stack.

    The crop rectangle covers magnifying and panning, which are the two things
    that repoint the camera's rendering; the image size covers the body being
    switched between its photo and movie live-view positions.
    """
    return (
        frame.crop_center_x,
        frame.crop_center_y,
        frame.crop_width,
        frame.crop_height,
        image.width(),
        image.height(),
    )
