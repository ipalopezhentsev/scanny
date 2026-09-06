"""Reading a QImage's pixels as numpy, without copying the image first.

Both the frame integrator and the sharpness meter want the same thing from a
live-view frame -- its bytes, as an array -- and the incantation for that has
one trap in it worth keeping in a single place: the array borrows the image's
buffer rather than owning it, so it is only good for as long as the image is.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage

__all__ = ["view", "green"]

#: The one format these two functions understand. Four bytes a pixel, in
#: memory order blue, green, red, unused -- which is what a little-endian
#: machine makes of Qt's 0xffRRGGBB -- and scanlines that are exactly four
#: bytes a pixel, since that is already 32-bit aligned.
_FORMAT = QImage.Format.Format_RGB32

#: Where green sits in that byte order. Green is the channel the sharpness
#: meter reads: JPEG carries full-resolution detail in every channel, but
#: green is the one with the best signal to noise in it.
GREEN = 1


def view(image: QImage) -> np.ndarray:
    """A ``(height, width, 4)`` view of *image*'s own pixel buffer.

    Borrowed, not owned -- the array dies with the image, and writing through
    it writes to the image. The image must already be in RGB32; converting it
    here would return a view of a temporary that is freed on the way out.
    """
    if image.format() != _FORMAT:
        raise ValueError(f"expected an RGB32 image, got {image.format()}")
    return np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4
    )[:, : image.width()]


def green(image: QImage) -> np.ndarray:
    """The green channel of any image, as an owned float32 array.

    A copy, which is what makes it safe to hand an image of any format: the
    conversion this may need lives only as long as the call.
    """
    if image.format() != _FORMAT:
        image = image.convertToFormat(_FORMAT)
    return view(image)[:, :, GREEN].astype(np.float32)
