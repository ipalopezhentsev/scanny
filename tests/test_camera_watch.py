"""Connecting to a camera that was not there when the program started.

Someone who starts scanny before switching the body on used to be stuck: the
camera was opened once, at startup, and nothing looked again. The only way
back was to quit and start over -- with a body sitting right there, switched
on and plugged in.

So a failed connection leaves a watch running, and the watch opens the camera
as soon as Windows can see one. The awkward part is the second or two after
the switch, when the device has been enumerated but is not answering PTP yet:
an attempt then can open the transport and still fail on the first question,
and what must not happen is for that half-open state to count as connected
and stop the watch.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.camera.nikon import CameraError  # noqa: E402
from scanny.ui import worker as worker_module  # noqa: E402
from scanny.ui.worker import CameraWorker  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _FakeCamera:
    """A body that answers everything a connection asks it.

    Unless ``mute``, which is one that has been opened but is not answering
    questions yet -- the state a body is in for a moment after the switch.
    """

    def __init__(self, mute: bool = False) -> None:
        self.mute = mute
        self.save_to_card = False
        self.shutter_delay = 0
        self.closed = False

    def _answer(self) -> None:
        if self.mute:
            raise CameraError("device is busy")

    def set_save_to_card(self, on):
        self._answer()
        self.save_to_card = on
        return True

    def set_shutter_delay(self, seconds):
        self._answer()
        self.shutter_delay = seconds

    def shutter_delay_choices(self):
        self._answer()
        return (0, 1, 2, 3)

    def shutter_delay_on_body(self):
        self._answer()
        return 0

    def battery_level(self):
        self._answer()
        return 80

    @property
    def model(self):
        self._answer()
        return "Nikon D750"

    @property
    def firmware(self):
        self._answer()
        return "1.10"

    def settings(self):
        return []

    def close(self):
        self.closed = True


class _Body:
    """Stands in for the camera on the other end of the cable.

    ``present`` is the switch: nothing is discovered while it is off, which is
    what a body that is switched off looks like to Windows.
    """

    def __init__(self) -> None:
        self.present = False
        self.mute = False
        self.opened: "list[_FakeCamera]" = []

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(
            worker_module.NikonCamera, "discover", staticmethod(self._discover)
        )
        monkeypatch.setattr(
            worker_module.NikonCamera, "open", staticmethod(self._open)
        )

    def _discover(self):
        return ["a device"] if self.present else []

    def _open(self):
        if not self.present:
            raise CameraError("no Nikon camera found")
        camera = _FakeCamera(self.mute)
        self.opened.append(camera)
        return camera


@pytest.fixture
def body(monkeypatch):
    fake = _Body()
    fake.install(monkeypatch)
    return fake


def _worker(app):
    worker = CameraWorker()
    said = {"connected": [], "waiting": 0, "failed": []}
    worker.connected.connect(lambda text: said["connected"].append(text))
    worker.waiting.connect(lambda: said.update(waiting=said["waiting"] + 1))
    worker.failed.connect(lambda text: said["failed"].append(text))
    return worker, said


def test_a_camera_that_is_there_is_opened_and_nothing_is_watched(app, body):
    worker, said = _worker(app)
    body.present = True
    worker.connect_camera()
    assert said["connected"] == ["Nikon D750  |  firmware 1.10 - battery 80%"]
    assert said["waiting"] == 0
    assert worker._watch is None


def test_a_missing_camera_is_reported_once_and_then_watched_for(app, body):
    worker, said = _worker(app)
    worker.connect_camera()
    assert said["failed"] == ["no Nikon camera found"]
    assert said["waiting"] == 1
    assert worker._watch is not None and worker._watch.isActive()


def test_switching_the_body_on_connects_without_a_restart(app, body):
    worker, said = _worker(app)
    worker.connect_camera()
    # Still off: the watch keeps quiet rather than complaining every tick.
    worker._watch_tick()
    assert worker._camera is None
    assert said["failed"] == ["no Nikon camera found"]

    body.present = True
    worker._watch_tick()
    assert worker._camera is not None
    assert said["connected"] == ["Nikon D750  |  firmware 1.10 - battery 80%"]
    assert not worker._watch.isActive()


def test_a_body_that_is_still_waking_up_does_not_count_as_connected(app, body):
    """Opened, but not answering yet: let go and try again on the next tick."""
    worker, said = _worker(app)
    worker.connect_camera()

    body.present = True
    body.mute = True
    worker._watch_tick()
    assert worker._camera is None
    assert worker._watch.isActive()
    # And the half-open camera was handed back rather than left held open.
    assert body.opened[-1].closed is True
    # Nothing said about it: the watch is quiet until it has something.
    assert said["connected"] == []
    assert said["failed"] == ["no Nikon camera found"]

    body.mute = False
    worker._watch_tick()
    assert worker._camera is not None
    assert said["connected"] == ["Nikon D750  |  firmware 1.10 - battery 80%"]


def test_an_asked_for_disconnection_stops_the_watch(app, body):
    """Shutdown goes through here, and must not leave a timer running."""
    worker, _said = _worker(app)
    worker.connect_camera()
    assert worker._watch.isActive()
    worker.disconnect_camera()
    assert not worker._watch.isActive()


def test_reconnect_with_the_body_still_off_goes_back_to_watching(app, body):
    worker, said = _worker(app)
    worker.connect_camera()
    worker.disconnect_camera()
    worker.connect_camera()
    assert worker._watch.isActive()
    assert said["waiting"] == 2
