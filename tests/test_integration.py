"""Tests for live-view frame integration.

Averaging consecutive frames is meant to do exactly one thing: keep the scene
and cancel the noise. The tests build frames whose "noise" is known -- a value
either side of the truth -- so the mean can be checked against the answer
rather than against a guess about how much cleaner it looks.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtGui")
pytest.importorskip("numpy")

from PySide6.QtCore import QBuffer, QByteArray  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402

from scanny.camera.nikon import LiveViewFrame  # noqa: E402
from scanny.ui.integration import (  # noqa: E402
    MAX_FRAMES,
    MIN_FRAMES,
    FrameIntegrator,
)


def _jpeg(grey: int, size=(32, 24)) -> bytes:
    """A flat grey JPEG. Flat, so the codec is not what the test measures."""
    image = QImage(*size, QImage.Format.Format_RGB32)
    image.fill(0xFF000000 | (grey << 16) | (grey << 8) | grey)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    image.save(buffer, "JPG", 100)
    return bytes(data.data())


def _frame(grey: int, *, crop_centre=(3008, 2008), size=(32, 24)) -> LiveViewFrame:
    return LiveViewFrame(
        jpeg=_jpeg(grey, size),
        width=size[0], height=size[1],
        image_width=6016, image_height=4016,
        crop_width=6016, crop_height=4016,
        crop_center_x=crop_centre[0], crop_center_y=crop_centre[1],
        af_width=324, af_height=270,
        af_x=3008, af_y=2008,
    )


def _grey(image: QImage) -> float:
    """The mean red channel of the middle of the image."""
    return sum(
        QImage.pixelColor(image, x, y).red()
        for x in range(8, 24)
        for y in range(6, 18)
    ) / (16 * 12)


def _feed(integrator: FrameIntegrator, greys) -> list:
    return [integrator.add(_frame(grey)) for grey in greys]


def test_disabled_shows_every_frame_as_it_arrives():
    integrator = FrameIntegrator(enabled=False)
    shown = _feed(integrator, [40, 80, 120])
    assert all(image is not None for image in shown)
    assert [round(_grey(image)) for image in shown] == pytest.approx([40, 80, 120], abs=2)


def test_a_stack_produces_one_image_and_nothing_in_between():
    integrator = FrameIntegrator(enabled=True, frames=4)
    shown = _feed(integrator, [100] * 9)
    # The first frame of a new view is shown at once, so the pattern is that
    # one, then an image every fourth frame.
    assert [image is not None for image in shown] == [
        True, False, False, True, False, False, False, True, False
    ]


def test_noise_either_side_of_the_truth_averages_out():
    integrator = FrameIntegrator(enabled=True, frames=4)
    # A scene at 100 with +-20 of noise on it, in equal measure.
    shown = _feed(integrator, [80, 120, 80, 120])
    assert shown[-1] is not None
    assert _grey(shown[-1]) == pytest.approx(100, abs=2)


def test_the_stack_is_the_frames_asked_for_and_no_others():
    integrator = FrameIntegrator(enabled=True, frames=4)
    # Four frames of 100 complete a stack; the 200 that follows belongs to the
    # next one and must not pull the answer up.
    shown = _feed(integrator, [100, 100, 100, 100, 200])
    assert _grey(shown[3]) == pytest.approx(100, abs=2)
    assert shown[4] is None


def test_a_new_view_is_shown_at_once_rather_than_a_stack_later():
    """Panning must not look like the picture has frozen."""
    integrator = FrameIntegrator(enabled=True, frames=8)
    integrator.add(_frame(100))
    for _ in range(3):
        assert integrator.add(_frame(100)) is None
    moved = integrator.add(_frame(200, crop_centre=(1000, 1000)))
    assert moved is not None
    assert _grey(moved) == pytest.approx(200, abs=2)
    assert integrator.pending == 1


def test_frames_from_the_old_view_do_not_leak_into_the_new_one():
    integrator = FrameIntegrator(enabled=True, frames=2)
    integrator.add(_frame(0))
    integrator.add(_frame(0))
    integrator.add(_frame(200, crop_centre=(1000, 1000)))
    averaged = integrator.add(_frame(200, crop_centre=(1000, 1000)))
    assert _grey(averaged) == pytest.approx(200, abs=2)


def test_a_change_of_size_starts_a_new_stack():
    """The body switching between its photo and movie positions resizes the frame."""
    integrator = FrameIntegrator(enabled=True, frames=2)
    integrator.add(_frame(100))
    resized = integrator.add(_frame(100, size=(32, 18)))
    assert resized is not None and resized.height() == 18


def test_changing_the_settings_throws_away_the_part_filled_stack():
    integrator = FrameIntegrator(enabled=True, frames=4)
    _feed(integrator, [100, 100])
    assert integrator.pending == 2
    assert integrator.configure(True, 8) is True
    assert integrator.pending == 0
    assert integrator.frames == 8


def test_repeating_the_same_settings_leaves_the_stack_alone():
    integrator = FrameIntegrator(enabled=True, frames=4)
    _feed(integrator, [100, 100])
    assert integrator.configure(True, 4) is False
    assert integrator.pending == 2


def test_the_frame_count_is_clamped_to_what_makes_sense():
    integrator = FrameIntegrator(enabled=True, frames=1)
    assert integrator.frames == MIN_FRAMES
    integrator.configure(True, 10_000)
    assert integrator.frames == MAX_FRAMES


def test_a_frame_that_is_not_a_picture_is_ignored():
    integrator = FrameIntegrator(enabled=True, frames=2)
    broken = LiveViewFrame(
        jpeg=b"\xff\xd8\xff not a jpeg",
        width=32, height=24,
        image_width=6016, image_height=4016,
        crop_width=6016, crop_height=4016,
        crop_center_x=3008, crop_center_y=2008,
        af_width=324, af_height=270,
        af_x=3008, af_y=2008,
    )
    assert integrator.add(broken) is None
    assert integrator.pending == 0


def test_a_long_stack_does_not_drift_darker():
    """Rounding down every time would cost half a level over a long stack."""
    integrator = FrameIntegrator(enabled=True, frames=MAX_FRAMES)
    shown = _feed(integrator, [101] * MAX_FRAMES)
    assert _grey(shown[-1]) == pytest.approx(101, abs=1)


# -- how the worker publishes an integrated stack ----------------------------


class _FakeCamera:
    """Just enough camera for the grab loop: it always has another frame.

    Another *different* frame. The worker discards a read that comes back
    holding the frame it already has, so a fake handing out identical bytes
    would exercise that path instead of the one under test here. The scene
    stays put and a single grey level moves either side of it, which is what
    a real frame does and what integration exists to average away.
    """

    live_view_active = True

    def __init__(self, grey: int = 100) -> None:
        self._grey = grey
        self._grabs = 0

    def live_view_frame(self) -> LiveViewFrame:
        self._grabs += 1
        return _frame(self._grey + (1 if self._grabs % 2 else -1))


@pytest.fixture
def worker(monkeypatch):
    from scanny.ui.worker import CameraWorker

    made = CameraWorker()
    made._camera = _FakeCamera(100)
    now = [1000.0]
    monkeypatch.setattr("scanny.ui.worker.time.monotonic", lambda: now[0])
    made._tick = lambda seconds=1 / 44: now.__setitem__(0, now[0] + seconds)
    return made


def _grabs(worker, count: int) -> list:
    published = []
    worker.frameReady.connect(lambda frame, image: published.append(image))
    for _ in range(count):
        worker._grab()
        worker._tick()
    return published


def test_every_frame_is_published_even_while_a_stack_fills(worker):
    """The focus box and the level readout must not wait for the picture."""
    worker.set_integration(True, 4)
    published = _grabs(worker, 8)
    assert len(published) == 8
    assert [not image.isNull() for image in published] == [
        True, False, False, True, False, False, False, True
    ]


def test_without_integration_every_frame_carries_its_picture(worker):
    published = _grabs(worker, 5)
    assert all(not image.isNull() for image in published)


def test_the_rate_shown_is_the_rate_the_picture_updates_at(worker):
    """Not the rate the camera sends at: that is unchanged by integrating."""
    worker.set_integration(True, 4)
    seen = []
    worker.fpsChanged.connect(lambda fps, w, h: seen.append(fps))
    _grabs(worker, 60)
    assert seen[-1] == pytest.approx(44.0 / 4, rel=0.15)


def test_stopping_live_view_throws_the_stack_away(worker):
    worker.set_integration(True, 4)
    _grabs(worker, 2)
    assert worker._integrator.pending == 2
    worker._camera = None
    worker.stop_live_view()
    assert worker._integrator.pending == 0


# -- reads that come back holding the frame we already have ------------------


class _StutteringCamera:
    """A camera polled twice as fast as it draws.

    Which is roughly the real situation: the worker's timer is set a little
    faster than the body's own frame rate on purpose, so that no frame is
    missed when that rate moves.
    """

    live_view_active = True

    def __init__(self) -> None:
        self._reads = 0
        self._drawn = 0

    def live_view_frame(self) -> LiveViewFrame:
        if self._reads % 2 == 0:
            self._drawn += 1
        self._reads += 1
        # Identical bytes for as long as the frame stands, which is what makes
        # the re-read recognisable as one.
        return _frame(100 + (1 if self._drawn % 2 else -1))


@pytest.fixture
def stuttering(monkeypatch):
    from scanny.ui.worker import CameraWorker

    made = CameraWorker()
    made._camera = _StutteringCamera()
    now = [1000.0]
    monkeypatch.setattr("scanny.ui.worker.time.monotonic", lambda: now[0])
    return made


def test_a_re_read_does_not_count_towards_the_stack(stuttering):
    """Averaging a frame with a copy of itself cancels nothing: the copy
    carries the same noise, which adds coherently instead of averaging down.
    Counting one in would only finish the stack early, and leave it grainier
    than the number of frames claims."""
    stuttering.set_integration(True, 4)
    for _ in range(6):
        stuttering._grab()
    # Six reads, three frames drawn; a stack that counted reads would have
    # completed at four and be two into the next one.
    assert stuttering._integrator.pending == 3


def test_a_re_read_publishes_nothing(stuttering):
    """There is no news in one -- not a new picture, and not a new focus box
    or level reading either, because they come off the same frame."""
    stuttering.set_integration(True, 4)
    published = []
    stuttering.frameReady.connect(lambda frame, image: published.append(image))
    for _ in range(8):
        stuttering._grab()
    assert [not image.isNull() for image in published] == [True, False, False, True]


def test_the_skip_can_be_switched_off(stuttering):
    """Off, every read is a frame again -- which is what the code did before
    the camera's draw rate was measured, and is how the gap between the poll
    rate and the draw rate can be seen at all."""
    stuttering.set_deduplicate(False)
    stuttering.set_integration(True, 4)
    published = []
    stuttering.frameReady.connect(lambda frame, image: published.append(image))
    for _ in range(8):
        stuttering._grab()
    assert len(published) == 8
    assert [not image.isNull() for image in published] == [
        True, False, False, True, False, False, False, True
    ]


def test_switching_the_skip_back_on_starts_a_clean_stack(stuttering):
    """A stack half filled under the other rule is a mix of the two."""
    stuttering.set_deduplicate(False)
    stuttering.set_integration(True, 8)
    for _ in range(3):
        stuttering._grab()
    assert stuttering._integrator.pending == 3
    stuttering.set_deduplicate(True)
    assert stuttering._integrator.pending == 0


# -- a thrown-away stack is not a moved view ---------------------------------


def test_restarting_a_stack_leaves_the_clean_picture_up():
    """The flash of grain: focus settles, the stack is dropped because it
    holds frames from the old focus position, and the next frame -- a single
    noisy one -- went straight to the screen in place of a clean image. The
    view has not moved, so there is nothing to catch up with: fill quietly."""
    integrator = FrameIntegrator(enabled=True, frames=4)
    _feed(integrator, [100] * 4)
    integrator.reset()
    assert integrator.add(_frame(100)) is None
    assert integrator.pending == 1


def test_a_moved_view_is_still_shown_at_once():
    """The other side of it: panning must not look like it has frozen."""
    integrator = FrameIntegrator(enabled=True, frames=4)
    _feed(integrator, [100] * 4)
    integrator.reset()
    moved = integrator.add(_frame(200, crop_centre=(1000, 1000)))
    assert moved is not None
    assert not integrator.last_image_was_whole


def test_forgetting_shows_the_next_frame_at_once():
    """Live view stopping and starting again leaves nothing on screen, so the
    first frame back has to go up rather than wait out a stack."""
    integrator = FrameIntegrator(enabled=True, frames=4)
    _feed(integrator, [100] * 4)
    integrator.forget()
    assert integrator.add(_frame(100)) is not None
