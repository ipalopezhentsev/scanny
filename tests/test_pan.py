"""Tests for arrow-key panning.

Panning has no dedicated camera command: the body centres its magnified view on
the focus point, so scrolling the view means moving that point. A step is an
eighth of what is currently on screen, so the view travels by the same visible
amount however far in you are zoomed.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtCore")

from scanny.ui.worker import CameraWorker  # noqa: E402

from test_live_view_frame import make_frame  # noqa: E402


class _FakeCamera:
    live_view_active = True

    def __init__(self, frame):
        self._frame = frame
        self.moves = []
        self.zooms = []
        self.focused = 0

    def live_view_frame(self):
        return self._frame

    def set_af_area(self, x, y):
        self.moves.append((x, y))

    def set_zoom_level(self, level):
        self.zooms.append(level)

    def zoom_level(self):
        return self.zooms[-1] if self.zooms else 0

    def autofocus(self, timeout=8.0):
        self.focused += 1
        return True


def _worker(frame):
    worker = CameraWorker()
    worker._camera = _FakeCamera(frame)
    return worker, worker._camera


ZOOMED = dict(crop=(640, 480), crop_centre=(3000, 2000), af=(3000, 2000))


def test_step_is_an_eighth_of_the_visible_crop():
    worker, camera = _worker(make_frame(**ZOOMED))
    worker.pan(1, 0)
    assert camera.moves == [(3000 + 640 // 8, 2000)]


def test_vertical_step_uses_the_crop_height():
    worker, camera = _worker(make_frame(**ZOOMED))
    worker.pan(0, 1)
    assert camera.moves == [(3000, 2000 + 480 // 8)]


def test_negative_directions_move_back():
    worker, camera = _worker(make_frame(**ZOOMED))
    worker.pan(-1, -1)
    assert camera.moves == [(3000 - 80, 2000 - 60)]


def test_step_scales_with_magnification():
    """At full frame a step covers far more sensor than when magnified, but the
    same fraction of what is on screen."""
    wide, wide_cam = _worker(make_frame(crop=(6016, 4016), crop_centre=(3008, 2008), af=(3008, 2008)))
    wide.pan(1, 0)
    tight, tight_cam = _worker(make_frame(**ZOOMED))
    tight.pan(1, 0)
    wide_step = wide_cam.moves[0][0] - 3008
    tight_step = tight_cam.moves[0][0] - 3000
    assert wide_step == 6016 // 8
    assert tight_step == 640 // 8
    assert wide_step > tight_step


def test_panning_never_drives_autofocus():
    worker, camera = _worker(make_frame(**ZOOMED))
    for _ in range(5):
        worker.pan(1, 0)
    assert camera.focused == 0


def test_pan_clamps_at_the_sensor_edge():
    # Sitting one step short of the right-hand limit of 6016 - 162.
    worker, camera = _worker(
        make_frame(crop=(640, 480), crop_centre=(5800, 2000), af=(5800, 2000))
    )
    worker.pan(1, 0)
    assert camera.moves == [(6016 - 162, 2000)]


def test_pan_at_the_edge_is_not_sent_again():
    """Already hard against the edge, so there is nothing to ask the camera."""
    worker, camera = _worker(
        make_frame(crop=(640, 480), crop_centre=(6016 - 162, 2000), af=(6016 - 162, 2000))
    )
    worker.pan(1, 0)
    assert camera.moves == []


def test_pan_without_a_camera_is_harmless():
    worker = CameraWorker()
    failures = []
    worker.failed.connect(failures.append)
    worker.pan(1, 0)
    assert failures  # reported, not crashed


# -- single click moves the rectangle, and nothing else -----------------------

# The 3:2 photo-mode frame, so the clamp limits are the 4016-high ones.
FULL = dict(
    image=(6016, 4016),
    crop=(6016, 4016),
    crop_centre=(3008, 2008),
    af=(3008, 2008),
)


def test_click_moves_the_rectangle():
    worker, camera = _worker(make_frame(**FULL))
    worker.move_point(0.25, 0.75)
    assert camera.moves == [(1504, 3012)]


def test_click_does_not_magnify_or_focus():
    worker, camera = _worker(make_frame(**FULL))
    zooms = []
    worker.zoomChanged.connect(zooms.append)
    worker.move_point(0.3, 0.3)
    assert camera.zooms == []
    assert zooms == []
    assert camera.focused == 0


def test_click_clamps_to_where_the_focus_box_fits():
    worker, camera = _worker(make_frame(**FULL))
    worker.move_point(1.0, 1.0)
    assert camera.moves == [(6016 - 162, 4016 - 135)]


# -- right click toggles between whole frame and full magnification -----------


def test_toggle_magnifies_when_the_view_is_whole():
    worker, camera = _worker(make_frame(**FULL))
    worker._last_frame = camera._frame
    worker.toggle_zoom()
    assert camera.zooms == [7]


def test_toggle_returns_to_the_whole_frame_when_magnified():
    frame = make_frame(image=(6016, 4016), crop=(320, 240), crop_centre=(1000, 1000), af=(1000, 1000))
    worker, camera = _worker(frame)
    worker._last_frame = frame
    worker.toggle_zoom()
    assert camera.zooms == [0]


def test_toggle_reads_the_live_frame_not_a_remembered_level():
    """Zoom can be changed on the camera body, so the frame is the authority."""
    frame = make_frame(image=(6016, 4016), crop=(640, 480), crop_centre=(1000, 1000), af=(1000, 1000))
    worker, camera = _worker(frame)
    worker._last_frame = frame
    worker._zoom_level = 0  # stale: says whole frame, but the frame is magnified
    worker.toggle_zoom()
    assert camera.zooms == [0]


def test_toggle_never_focuses():
    worker, camera = _worker(make_frame(**FULL))
    worker._last_frame = camera._frame
    worker.toggle_zoom()
    assert camera.focused == 0


# -- manual focus -------------------------------------------------------------


class _FocusCamera(_FakeCamera):
    def __init__(self, frame, at_limit=False):
        super().__init__(frame)
        self.drives = []
        self._at_limit = at_limit

    def drive_focus(self, steps):
        self.drives.append(steps)
        return not self._at_limit


def test_drive_focus_passes_signed_steps_through():
    worker = CameraWorker()
    worker._camera = camera = _FocusCamera(make_frame())
    worker.drive_focus(-10)
    worker.drive_focus(1000)
    assert camera.drives == [-10, 1000]


def test_drive_focus_reports_direction():
    worker = CameraWorker()
    worker._camera = _FocusCamera(make_frame())
    said = []
    worker.status.connect(said.append)
    worker.drive_focus(-100)
    assert "nearer" in said[-1]
    worker.drive_focus(100)
    assert "further" in said[-1]


def test_drive_focus_says_when_the_lens_is_at_its_limit():
    worker = CameraWorker()
    worker._camera = _FocusCamera(make_frame(), at_limit=True)
    said = []
    worker.status.connect(said.append)
    worker.drive_focus(1000)
    assert "limit" in said[-1]


def test_default_focus_step_sizes_are_ordered():
    from scanny.camera.nikon import NikonCamera

    values = [
        NikonCamera.FOCUS_STEP_DEFAULTS[name]
        for name in NikonCamera.FOCUS_INCREMENTS
    ]
    assert values == sorted(values)
    assert all(v > 0 for v in values)
    # How fine the finest default can usefully be is a property of the lens,
    # not the body: six steps move an AF-S 60/2.8 micro visibly, while a
    # 24-120/4 needs about eighteen. So the requirement is only that it is a
    # genuinely small step -- the number itself is the user's to tune, and the
    # editor still goes down to the single step the body accepts.
    assert values[0] <= 20
