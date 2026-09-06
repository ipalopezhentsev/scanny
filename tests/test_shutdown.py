"""Leaving the program has to put the camera back down.

Live view holds the mirror up and keeps the sensor streaming, and the body
stays that way until it is told otherwise -- unplugging or closing the session
does not end it. So the worker has to get its shutdown in before its thread
stops, even though the thread is nearly always mid frame grab when the window
closes.
"""

from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.ui.worker import CameraWorker  # noqa: E402


class _Camera:
    """A camera that answers slowly, the way a real one on USB does."""

    def __init__(self, delay: float = 0.0) -> None:
        self.live_view = True
        self.live_view_active = True
        self.closed = False
        self._delay = delay

    def stop_live_view(self) -> None:
        self.live_view = False
        self.live_view_active = False

    def close(self) -> None:
        # NikonCamera.close ends live view on the way out.
        if self._delay:
            threading.Event().wait(self._delay)
        if self.live_view:
            self.stop_live_view()
        self.closed = True


class _Asker(QObject):
    """Stands in for the window: it only ever asks, from its own thread."""

    requestShutdown = Signal()


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _run_worker(app, camera, busy_for: float = 0.0):
    """Put a worker on a real thread, then shut it down the way the window does."""
    thread = QThread()
    worker = CameraWorker()
    worker.moveToThread(thread)
    worker._camera = camera
    asker = _Asker()
    asker.requestShutdown.connect(worker.shutdown)
    if busy_for:
        # Occupy the thread, so the shutdown request lands while it is inside a
        # camera call rather than idle in its event loop -- which is when the
        # window closing used to lose the request.
        thread.started.connect(lambda: threading.Event().wait(busy_for))
    thread.start()

    asker.requestShutdown.emit()
    finished = thread.wait(5000)
    if not finished:  # pragma: no cover - only on a failure
        thread.quit()
        thread.wait(1000)
    QCoreApplication.processEvents()
    return worker, thread, finished


def test_shutdown_stops_live_view_and_ends_the_thread(app):
    camera = _Camera()
    _worker, _thread, finished = _run_worker(app, camera)
    assert camera.closed
    assert not camera.live_view
    assert finished


def test_shutdown_still_arrives_when_the_thread_is_busy(app):
    camera = _Camera()
    _worker, _thread, finished = _run_worker(app, camera, busy_for=0.3)
    assert camera.closed
    assert not camera.live_view
    assert finished


def test_the_thread_is_not_quit_before_the_camera_is_closed(app):
    # The camera takes its time answering; the wait has to outlast it rather
    # than cut it off.
    camera = _Camera(delay=0.2)
    _worker, _thread, finished = _run_worker(app, camera)
    assert camera.closed
    assert finished


def test_closing_the_window_shuts_the_worker_down(app, monkeypatch):
    from scanny.ui.main_window import MainWindow

    camera = _Camera()
    # Hand the worker its camera instead of letting it hunt for a real one, so
    # the test says the same thing whether or not a body is plugged in.
    monkeypatch.setattr(
        CameraWorker, "connect_camera", lambda self: setattr(self, "_camera", camera)
    )
    window = MainWindow()
    window.show()
    QApplication.processEvents()

    window.close()

    assert camera.closed
    assert not camera.live_view
    assert window._thread.isFinished()
