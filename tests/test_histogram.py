"""Tests for the level histogram.

The two things worth pinning down are the ones that took a decision: what the
curves are scaled to, and what counts as clipping. Scaling to the tallest bin
of all would let one blown highlight -- which a light box guarantees -- flatten
the rest of the picture into the floor, and clipping reported as an average
across the channels would call a blown red channel "a third clipped".
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.ui.histogram import HistogramWidget  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def app():
    """A widget cannot be built without one, and every test here builds one."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget():
    made = HistogramWidget()
    made.resize(280, 96)
    return made


def picture(red: int, green: int, blue: int, size: int = 64) -> QImage:
    """A flat picture of one colour, in the format live view arrives in."""
    pixels = np.zeros((size, size, 4), dtype=np.uint8)
    pixels[:, :, 0] = blue
    pixels[:, :, 1] = green
    pixels[:, :, 2] = red
    return QImage(
        pixels.tobytes(), size, size, size * 4, QImage.Format.Format_RGB32
    ).copy()


def halves(low: int, high: int, size: int = 64) -> QImage:
    """Half the picture at one level, half at another."""
    pixels = np.full((size, size, 4), low, dtype=np.uint8)
    pixels[: size // 2] = high
    return QImage(
        pixels.tobytes(), size, size, size * 4, QImage.Format.Format_RGB32
    ).copy()


def test_counts_land_in_the_bin_of_the_level(widget):
    assert widget.set_image(picture(10, 20, 30))
    counts = widget.counts
    assert counts is not None
    for row, level in enumerate((10, 20, 30)):
        assert counts[row].argmax() == level
        assert counts[row][level] == counts[row].sum()


def test_channels_are_counted_in_red_green_blue_order(widget):
    """The buffer is blue, green, red; the array must not be."""
    widget.set_image(picture(200, 100, 50))
    counts = widget.counts
    assert (counts[0].argmax(), counts[1].argmax(), counts[2].argmax()) == (
        200,
        100,
        50,
    )


def test_a_picture_of_any_format_is_accepted(widget):
    """Nothing published by live view is another format, but a caller that
    hands over a decoded JPEG straight from Qt would be."""
    image = picture(10, 20, 30).convertToFormat(QImage.Format.Format_RGB888)
    assert widget.set_image(image)
    assert widget.counts is not None


def test_clipping_is_the_worst_channel_not_their_average(widget):
    """One channel off the end is a colour that cannot be recovered."""
    widget.set_image(picture(255, 128, 128))
    shadows, highlights = widget.clipping
    assert highlights == pytest.approx(1.0)
    assert shadows == pytest.approx(0.0)


def test_clipping_counts_the_part_of_the_picture_that_clipped(widget):
    widget.set_image(halves(0, 255))
    shadows, highlights = widget.clipping
    assert shadows == pytest.approx(0.5)
    assert highlights == pytest.approx(0.5)


def test_the_odd_clipped_pixel_is_not_worth_saying_anything_about(widget):
    """A single specular highlight in a big picture is not lost detail."""
    pixels = np.full((256, 256, 4), 128, dtype=np.uint8)
    pixels[0, 0] = 255
    widget.set_image(
        QImage(pixels.tobytes(), 256, 256, 256 * 4, QImage.Format.Format_RGB32).copy()
    )
    assert widget.describe() == "Nothing clipped"


def test_clipping_is_described_at_both_ends(widget):
    widget.set_image(halves(0, 255))
    said = widget.describe()
    assert "50.0% crushed" in said
    assert "50.0% blown" in said


def test_the_height_is_scaled_past_the_end_bins(widget):
    """A light box fills bin 255; the picture on it must still be readable.

    The peak the curves are drawn against comes from the bins between the
    ends, so the enormous spike is drawn clipped off rather than squashing
    everything else to nothing.
    """
    pixels = np.full((100, 100, 4), 255, dtype=np.uint8)
    pixels[:10] = 120  # a tenth of the picture is the subject
    widget.set_image(
        QImage(pixels.tobytes(), 100, 100, 100 * 4, QImage.Format.Format_RGB32).copy()
    )
    counts = widget.counts
    assert counts[:, 255].max() > counts[:, 120].max()
    assert widget._peak() == pytest.approx(counts[:, 120].max())


def test_a_picture_entirely_at_one_end_still_has_a_scale(widget):
    """A lens cap: nothing between the end bins, so nothing to scale to."""
    widget.set_image(picture(0, 0, 0))
    assert widget._peak() > 0


def test_frames_arriving_faster_than_the_eye_are_not_all_counted(widget):
    assert widget.set_image(picture(10, 10, 10))
    assert not widget.set_image(picture(200, 200, 200))
    # The first count stands, rather than a half-updated one.
    assert widget.counts[0].argmax() == 10


def test_the_next_frame_after_the_gap_is_counted(widget):
    widget.set_image(picture(10, 10, 10))
    time.sleep(0.11)
    assert widget.set_image(picture(200, 200, 200))
    assert widget.counts[0].argmax() == 200


def test_a_frame_still_waiting_for_its_stack_is_not_counted(widget):
    """A null image is how a part-filled integration stack is published."""
    assert not widget.set_image(QImage())
    assert widget.counts is None


def test_clearing_forgets_the_picture(widget):
    widget.set_image(picture(10, 20, 30))
    widget.clear()
    assert widget.counts is None
    assert widget.clipping == (0.0, 0.0)
    assert widget.describe() == "Waiting for a frame..."
    # And is countable again at once, rather than waiting out the interval.
    assert widget.set_image(picture(10, 20, 30))


def test_a_big_picture_is_sampled_rather_than_counted_whole():
    """The shape is what matters, and it survives taking every nth pixel."""
    from scanny.ui.histogram import _SAMPLE_TARGET, _sample_step

    for width, height in ((6000, 4000), (640, 424), (320, 200)):
        step = _sample_step(width, height)
        counted = (width // step) * (height // step)
        # Never more than about the target, and never so coarse that a small
        # picture is counted from a handful of pixels.
        assert counted <= _SAMPLE_TARGET * 1.5
        assert step == 1 or counted >= _SAMPLE_TARGET / 2


def test_painting_a_counted_picture_does_not_fall_over(widget):
    widget.set_image(halves(0, 255))
    widget.show()
    QApplication.processEvents()
    widget.grab()  # paints
