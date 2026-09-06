"""Tests for the sharpness measure, which is the thing focus is set against.

The measure only has to do one thing well: rise as the picture comes into
focus and fall as it leaves, without moving for anything else. So the tests
are a simulated focus sweep -- the same subject at increasing amounts of blur
-- plus the two things that would otherwise move the reading on their own,
brightness and noise.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtGui")

from PySide6.QtGui import QImage  # noqa: E402

from scanny.camera.nikon import LiveViewFrame  # noqa: E402
from scanny.ui.integration import FrameIntegrator  # noqa: E402
from scanny.ui.sharpness import (  # noqa: E402
    MIN_AREA_PIXELS,
    SharpnessMeter,
    measure,
    variance_between,
)

W, H = 128, 96


def _subject() -> np.ndarray:
    """A chequerboard: detail everywhere, so blur has something to destroy."""
    yy, xx = np.mgrid[0:H, 0:W]
    return (40 + 120 * ((xx // 8 + yy // 8) % 2)).astype(np.float64)


def _blurred(pixels: np.ndarray, passes: int) -> np.ndarray:
    for _ in range(passes):
        pixels = (
            pixels
            + np.roll(pixels, 1, 0) + np.roll(pixels, -1, 0)
            + np.roll(pixels, 1, 1) + np.roll(pixels, -1, 1)
        ) / 5
    return pixels


def _image(pixels: np.ndarray) -> QImage:
    grey = np.clip(pixels, 0, 255).astype(np.uint8)
    buffer = np.zeros((H, W, 4), np.uint8)
    for channel in range(3):
        buffer[:, :, channel] = grey
    buffer[:, :, 3] = 255
    return QImage(
        buffer.tobytes(), W, H, W * 4, QImage.Format.Format_RGB32
    ).copy()


def _noisy(pixels: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    return pixels + np.random.default_rng(seed).normal(0, sigma, pixels.shape)


def _frame(jpeg: bytes = b"", *, crop_centre=(3008, 2008)) -> LiveViewFrame:
    return LiveViewFrame(
        jpeg=jpeg, width=W, height=H,
        image_width=6016, image_height=4016,
        crop_width=6016, crop_height=4016,
        crop_center_x=crop_centre[0], crop_center_y=crop_centre[1],
        af_width=324, af_height=270, af_x=3008, af_y=2008,
    )


# -- the measure itself ------------------------------------------------------


def test_the_reading_falls_all_the_way_through_a_focus_sweep():
    subject = _subject()
    readings = [measure(_image(_blurred(subject, n))) for n in (0, 1, 2, 4, 8, 16)]
    assert readings == sorted(readings, reverse=True), readings
    assert readings[0] > 20 * readings[-1]


def test_a_blank_wall_reads_as_nothing_at_all():
    assert measure(_image(np.full((H, W), 120.0))) == pytest.approx(0.0, abs=1e-6)


def test_brightness_does_not_move_the_reading():
    """Stopping down darkens live view; that must not read as going soft."""
    subject = _blurred(_subject(), 1)
    readings = [measure(_image(subject * gain)) for gain in (0.5, 1.0, 1.7)]
    assert max(readings) - min(readings) < 0.05 * max(readings)


def test_a_black_frame_is_not_a_division_by_zero():
    assert measure(_image(np.zeros((H, W)))) == 0.0


def _noise_variance(pixels: np.ndarray, sigma: float) -> float:
    """What the worker measures: two frames of a still scene, an instant apart."""
    return variance_between(_noisy(pixels, sigma, 11), _noisy(pixels, sigma, 12))


def test_grain_does_not_read_as_sharpness():
    """Noise is contrast, and it was most of what the reading measured.

    White noise adds a known amount to the gradient energy, so given how much
    noise there is it can be taken off. Without that, this subject read only 2
    to 1 between focused and thoroughly defocused on noisy frames -- less than
    a stop of exposure moved it, which made the number useless.
    """
    sharp, soft = _subject(), _blurred(_subject(), 16)
    noise = 25
    variance = _noise_variance(soft, noise)

    def ratio(readings):
        return readings[0] / max(readings[1], 1e-9)

    clean = ratio([measure(_image(p)) for p in (sharp, soft)])
    counted = ratio([measure(_image(_noisy(p, noise, 1))) for p in (sharp, soft)])
    removed = ratio(
        [measure(_image(_noisy(p, noise, 1)), None, variance) for p in (sharp, soft)]
    )

    assert counted < clean / 4, "the grain has to actually swamp the reading"
    assert removed > 4 * counted, "and taking it off has to put that back"


def test_the_grain_is_measured_from_two_frames_not_from_one():
    """From one frame it cannot be done, and the way it fails is the worst
    possible: at best focus the finest detail in the picture *is* what changes
    from pixel to pixel, so a within-frame estimate reads the subject as grain
    and the sharpest frame of all reads zero."""
    sharp = _subject()
    still = variance_between(_noisy(sharp, 8, 1), _noisy(sharp, 8, 2))
    assert still == pytest.approx(64, rel=0.3), "two still frames give the grain"
    # And a pair that moved between them reads high, which is why the worker
    # keeps the smallest it has seen rather than the latest.
    moved = variance_between(_noisy(sharp, 8, 1), _noisy(_blurred(sharp, 4), 8, 2))
    assert moved > 4 * still


def test_a_frame_of_nothing_but_grain_reads_as_nothing():
    flat = np.full((H, W), 110.0)
    variance = _noise_variance(flat, 20)
    assert measure(_image(_noisy(flat, 20, 4)), None, variance) < 0.01 * measure(
        _image(_subject())
    )


def test_the_sharpest_frame_reads_highest_however_fine_its_detail():
    """The failure this all came from: pixel-level detail read as grain, so
    the sharpest picture scored zero and a blurred one beat it."""
    fine = _fine_texture()
    variance = _noise_variance(fine, 6)
    readings = [
        measure(_image(_noisy(_blurred(fine, n), 6, 3)), None, variance)
        for n in (0, 1, 2, 4, 8)
    ]
    assert readings == sorted(readings, reverse=True), readings
    assert readings[0] > 3 * readings[1], "and by a decisive margin"


def _fine_texture() -> np.ndarray:
    """Detail right down at the pixel, which is what best focus looks like."""
    field = np.random.default_rng(7).normal(0, 1, (H, W))
    field = (
        field
        + np.roll(field, 1, 0) + np.roll(field, -1, 0)
        + np.roll(field, 1, 1) + np.roll(field, -1, 1)
    ) / 5
    return np.clip(120 + 55 * field / field.std(), 0, 255)


def test_exposure_barely_moves_the_reading():
    """The complaint this was built to answer: a stop of exposure used to move
    the reading by a third, which is as much as a visible focus error does."""
    subject = _blurred(_subject(), 1)
    readings = []
    for stops in (-1, 0, 1):
        # Darker exposure, grainier frame -- and the grain measured from two
        # frames of it, as the worker does.
        sigma = 3 * 2 ** (-stops / 2)
        exposed = _exposed(subject, stops)
        readings.append(
            measure(_image(_noisy(exposed, sigma, 7)), None, _noise_variance(exposed, sigma))
        )
    assert (max(readings) - min(readings)) < 0.25 * max(readings)


def _exposed(code: np.ndarray, stops: float) -> np.ndarray:
    """The same scene a stop brighter or darker, clipping and all.

    Exposure is a scaling of the light, not of the numbers in a JPEG, so it
    happens in linear light with the gamma taken off and put back.
    """
    linear = np.clip(code / 255.0, 0, 1) ** 2.2
    return 255.0 * np.clip(linear * 2.0**stops, 0, 1) ** (1 / 2.2)


def _integrate(pixels: np.ndarray, noise: float, frames: int = 16) -> QImage:
    """What the screen shows for this subject with integration switched on."""
    stack = FrameIntegrator(enabled=True, frames=frames)
    shown = None
    for seed in range(frames):
        got = stack.add(_jpeg_frame(_noisy(pixels, noise, seed)))
        if got is not None:
            shown = got
    assert shown is not None
    return shown


def _jpeg_frame(pixels: np.ndarray) -> LiveViewFrame:
    from PySide6.QtCore import QBuffer, QByteArray

    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    _image(pixels).save(buffer, "JPG", 95)
    return _frame(bytes(data.data()))


# -- the meter ---------------------------------------------------------------


def test_nothing_is_measured_until_it_is_switched_on():
    meter = SharpnessMeter()
    assert meter.measure(_frame(), _image(_subject())) is None


def test_the_best_reading_is_kept_while_the_reading_wanders():
    meter = SharpnessMeter(enabled=True)
    sharp, soft = _subject(), _blurred(_subject(), 8)
    best = meter.measure(_frame(), _image(sharp))[1]
    value, peak = meter.measure(_frame(), _image(soft))
    assert value < best
    assert peak == best


def test_the_best_is_forgotten_when_the_view_moves():
    """A reading from a different crop is a different measurement."""
    meter = SharpnessMeter(enabled=True)
    meter.measure(_frame(), _image(_subject()))
    value, peak = meter.measure(
        _frame(crop_centre=(1000, 1000)), _image(_blurred(_subject(), 8))
    )
    assert peak == value


def test_switching_it_on_and_off_starts_again():
    meter = SharpnessMeter(enabled=True)
    meter.measure(_frame(), _image(_subject()))
    assert meter.peak > 0
    assert meter.configure(False) is True
    assert meter.peak == 0.0
    assert meter.configure(False) is False


def test_the_best_can_be_forgotten_on_request():
    meter = SharpnessMeter(enabled=True)
    meter.measure(_frame(), _image(_subject()))
    meter.reset()
    assert (meter.peak, meter.value) == (0.0, 0.0)


# -- how the worker publishes it ---------------------------------------------


class _FakeCamera:
    live_view_active = True
    exposure_preview = True

    def __init__(self, frame: LiveViewFrame) -> None:
        self.frame = frame

    def live_view_frame(self) -> LiveViewFrame:
        return self.frame

    def stop_live_view(self) -> None:
        pass

    def set_setting(self, name, value) -> None:
        pass

    def settings(self) -> list:
        return []

    def set_exposure_preview(self, enabled: bool) -> None:
        self.exposure_preview = enabled


@pytest.fixture
def worker():
    pytest.importorskip("PySide6.QtWidgets")
    from scanny.ui.worker import CameraWorker

    made = CameraWorker()
    made._camera = _FakeCamera(_jpeg_frame(_subject()))
    return made


def _readings(worker, grabs: int) -> list:
    seen = []
    worker.sharpnessChanged.connect(lambda value, peak: seen.append((value, peak)))
    for _ in range(grabs):
        worker._grab()
    return seen


def test_nothing_is_published_until_measuring_is_asked_for(worker):
    assert _readings(worker, 3) == []


def test_a_reading_for_every_frame_that_is_displayed(worker):
    worker.set_sharpness(True)
    seen = _readings(worker, 4)
    assert len(seen) == 4
    assert all(value > 0 for value, _ in seen)


def test_only_the_frames_that_are_displayed_are_measured(worker):
    """Integrating, the picture changes once a stack; measuring the frames in
    between would be measuring pictures nobody sees."""
    worker.set_integration(True, 4)
    worker.set_sharpness(True)
    seen = _readings(worker, 8)
    # The first frame of a view is shown at once, then one per completed stack.
    assert len(seen) == 3


def test_changing_the_integration_forgets_the_best(worker):
    """How much noise is left in the picture is part of what the reading is."""
    worker.set_sharpness(True)
    _readings(worker, 2)
    assert worker._sharpness.peak > 0
    worker.set_integration(True, 8)
    assert worker._sharpness.peak == 0.0


def test_stopping_live_view_clears_the_readout(worker):
    worker.set_sharpness(True)
    _readings(worker, 2)
    seen = []
    worker.sharpnessChanged.connect(lambda value, peak: seen.append((value, peak)))
    worker.stop_live_view()
    assert seen[-1] == (0.0, 0.0)
    assert worker._sharpness.peak == 0.0


# -- measuring one part of the picture ---------------------------------------


def _split_subject() -> np.ndarray:
    """Sharp detail on the left half, a smooth gradient on the right."""
    yy, xx = np.mgrid[0:H, 0:W]
    sharp = 40 + 120 * ((xx // 4 + yy // 4) % 2)
    smooth = 40 + 120 * (xx / W)
    return np.where(xx < W // 2, sharp, smooth).astype(np.float64)


def test_an_area_reads_only_what_is_inside_it():
    image = _image(_split_subject())
    detailed = measure(image, (0.0, 0.0, 0.5, 1.0))
    plain = measure(image, (0.5, 0.0, 0.5, 1.0))
    whole = measure(image)
    assert detailed > whole > plain
    assert detailed > 50 * plain


def test_a_small_area_can_pick_out_what_the_whole_frame_averages_away():
    """The point of it: one small sharp thing in an otherwise soft frame."""
    pixels = _blurred(_subject(), 8)
    pixels[40:60, 40:60] = _subject()[40:60, 40:60]
    image = _image(pixels)
    assert measure(image, (40 / W, 40 / H, 20 / W, 20 / H)) > 5 * measure(image)


def test_an_area_outside_the_picture_is_pulled_back_inside_it():
    image = _image(_subject())
    assert measure(image, (0.9, 0.9, 0.5, 0.5)) > 0


def test_an_area_too_small_to_measure_is_grown_to_something_usable():
    """A few pixels is all noise and no subject, and jumps about far too much."""
    image = _image(_subject())
    tiny = measure(image, (0.5, 0.5, 0.001, 0.001))
    same = measure(image, (0.5, 0.5, MIN_AREA_PIXELS / W, MIN_AREA_PIXELS / H))
    assert tiny == pytest.approx(same)


def test_choosing_an_area_forgets_the_best_reading():
    meter = SharpnessMeter(enabled=True)
    meter.measure(_frame(), _image(_subject()))
    assert meter.peak > 0
    assert meter.set_area((0.25, 0.25, 0.5, 0.5)) is True
    assert meter.peak == 0.0
    assert meter.set_area((0.25, 0.25, 0.5, 0.5)) is False


def test_the_meter_reads_the_area_it_was_given():
    meter = SharpnessMeter(enabled=True)
    image = _image(_split_subject())
    meter.set_area((0.0, 0.0, 0.5, 1.0))
    detailed, _ = meter.measure(_frame(), image)
    meter.set_area((0.5, 0.0, 0.5, 1.0))
    plain, _ = meter.measure(_frame(), image)
    assert detailed > plain


def test_going_back_to_the_whole_frame(worker):
    worker.set_sharpness(True)
    worker.set_sharpness_area((0.0, 0.0, 0.5, 1.0))
    assert worker._sharpness.area == (0.0, 0.0, 0.5, 1.0)
    worker.set_sharpness_area(None)
    assert worker._sharpness.area is None


def test_changing_the_exposure_starts_the_readings_again(worker):
    """A darker live view is a grainier one, and with exposure preview on a
    deeper one: readings either side of the change are different pictures."""
    worker.set_sharpness(True)
    _readings(worker, 2)
    assert worker._sharpness.peak > 0

    told = []
    worker.sharpnessChanged.connect(lambda value, peak: told.append((value, peak)))
    worker.set_setting("Aperture", 8)
    assert worker._sharpness.peak == 0.0
    assert told[-1] == (0.0, 0.0)


def test_turning_exposure_preview_on_or_off_starts_them_again_too(worker):
    worker.set_sharpness(True)
    _readings(worker, 2)
    worker.set_exposure_preview(False)
    assert worker._sharpness.peak == 0.0


def test_a_setting_change_says_nothing_while_the_meter_is_off(worker):
    told = []
    worker.sharpnessChanged.connect(lambda value, peak: told.append((value, peak)))
    worker.set_setting("Aperture", 8)
    assert told == []
