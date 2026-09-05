"""Tests for the measured live-view frame rate.

The worker publishes the rate over a rolling one-second window rather than an
instantaneous delta, so the readout does not flicker on a single slow frame.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtCore")

from scanny.ui.worker import CameraWorker  # noqa: E402


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def tick(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def worker(monkeypatch):
    made = CameraWorker()
    clock = _Clock()
    monkeypatch.setattr("scanny.ui.worker.time.monotonic", lambda: clock.now)
    made._clock = clock
    return made


class _Frame:
    def __init__(self, width=640, height=424, crop_width=6016):
        self.width = width
        self.height = height
        self.crop_width = crop_width


def _rates(worker, frame, intervals):
    seen = []
    worker.fpsChanged.connect(lambda fps, w, h: seen.append((fps, w, h)))
    worker._note_frame_rate(frame)
    for gap in intervals:
        worker._clock.tick(gap)
        worker._note_frame_rate(frame)
    return seen


def test_no_rate_reported_from_a_single_frame(worker):
    seen = _rates(worker, _Frame(), [])
    assert seen == []


def test_steady_frames_report_their_rate(worker):
    seen = _rates(worker, _Frame(), [1 / 30] * 20)
    fps, width, height = seen[-1]
    assert fps == pytest.approx(30.0, rel=1e-6)
    assert (width, height) == (640, 424)


def test_window_is_limited_to_recent_frames(worker):
    frame = _Frame()
    # A long stall, then a steady 30fps run. The stall must age out rather than
    # drag the reported rate down for the next minute.
    _rates(worker, frame, [5.0])
    seen = _rates(worker, frame, [1 / 30] * 40)
    assert seen[-1][0] == pytest.approx(30.0, rel=0.05)


def test_rate_reflects_a_slower_camera(worker):
    seen = _rates(worker, _Frame(), [0.1] * 12)
    assert seen[-1][0] == pytest.approx(10.0, rel=1e-6)


def test_frame_size_is_reported_alongside_the_rate(worker):
    seen = _rates(worker, _Frame(width=320, height=212), [1 / 30] * 5)
    assert seen[-1][1:] == (320, 212)
