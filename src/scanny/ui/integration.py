"""Averaging consecutive live-view frames, to trade frame rate for clarity.

Sensor noise is different in every frame while the scene is not, so the mean of
N frames keeps the picture and shrinks the noise by roughly the square root of
N. Live view cannot be asked for a longer exposure -- the camera sends what it
sends, about 30 frames a second -- so the only currency available is frame
rate: show the mean of each N frames and the picture arrives 30/N times a
second with visibly less grain in it.

Two choices worth knowing about:

- **The mean is taken in the JPEG's own gamma-encoded values**, not in linear
  light. Averaging linearly is the physically correct way to add exposures,
  but it also lifts the shadows and changes how the picture looks. The point
  here is the same picture with less noise in it, so the arithmetic stays in
  the values the camera sent.
- **A frame that does not match the stack is shown immediately**, and starts a
  new stack. Magnifying or panning changes the crop the camera renders, and
  waiting out the rest of a stack before showing the new view would make every
  pan feel like it had frozen. This way the first frame of a new view appears
  at once -- noisy -- and the integrated version replaces it a stack later.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage

from ..camera.nikon import LiveViewFrame
from .pixels import view

__all__ = ["FrameIntegrator", "MAX_FRAMES", "MIN_FRAMES", "DEFAULT_FRAMES"]

#: Fewer than two frames is not an average, and 64 frames is already two
#: seconds of integration at the camera's 30fps -- long enough that the view
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
        """Throw away the part-filled stack, as when live view restarts."""
        self._sum = None
        self._count = 0
        self._key = None

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
            self._start(key, image)
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
