"""Naming what lands on the computer, from the shot down to the spin box.

What these hold onto is the difference between the counter as a *setting* and
the counter as a *readout*. It is both: the user types over it to override the
sequence, and it moves by itself as shots use numbers up. So a change the user
made has to reach the worker, a change the worker reported must not be sent
back at it, and either way the box has to show the number the next shot will
actually get.

The worker's half is simpler to state: one release of the shutter takes one
number, whatever number of files it produced, and a shot that produced none
takes none.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.ui.main_window import MainWindow  # noqa: E402
from scanny.ui.naming import DEFAULT_PREFIX  # noqa: E402
from scanny.ui.worker import CameraWorker  # noqa: E402


# -- the worker ------------------------------------------------------------


class _FakeCamera:
    """A body that hands back the files it was told to, and nothing else."""

    shutter_delay = 0
    save_to_card = True

    @staticmethod
    def shot_seconds() -> float:
        """Nothing here is about how long a shot takes."""
        return 0.0

    def __init__(self, files: "list[str]") -> None:
        self._files = files
        self.session = self

    def capture(self, autofocus=False):
        return list(range(1, len(self._files) + 1))

    def object_info(self, handle):
        raise AssertionError("the name should come from the download")

    def download(self, handle):
        return self._files[handle - 1], b"a picture"

    def settings(self):
        return []


def _worker(tmp_path, files=("DSC_0001.NEF",), enabled=True, prefix="page_", number=1):
    worker = CameraWorker()
    worker._camera = _FakeCamera(list(files))
    worker._save_dir = tmp_path
    worker.set_naming(enabled, prefix, number)
    return worker


def _saved(tmp_path):
    return sorted(path.name for path in tmp_path.iterdir())


def test_a_saved_picture_takes_the_name_it_was_promised(tmp_path):
    worker = _worker(tmp_path, number=84)
    worker.capture(autofocus=False, download=True)
    assert _saved(tmp_path) == ["page_0084.NEF"]


def test_the_camera_still_says_what_kind_of_file_it_is(tmp_path):
    worker = _worker(tmp_path, files=("DSC_0001.JPG",))
    worker.capture(autofocus=False, download=True)
    assert _saved(tmp_path) == ["page_0001.JPG"]


def test_the_counter_moves_on_by_itself(tmp_path):
    worker = _worker(tmp_path)
    worker.capture(autofocus=False, download=True)
    worker.capture(autofocus=False, download=True)
    assert _saved(tmp_path) == ["page_0001.NEF", "page_0002.NEF"]
    assert worker.naming.number == 3


def test_a_raw_and_its_jpeg_are_one_picture_under_one_number(tmp_path):
    worker = _worker(tmp_path, files=("DSC_0001.NEF", "DSC_0001.JPG"))
    worker.capture(autofocus=False, download=True)
    assert _saved(tmp_path) == ["page_0001.JPG", "page_0001.NEF"]
    assert worker.naming.number == 2


def test_an_override_between_two_shots_is_taken(tmp_path):
    worker = _worker(tmp_path)
    worker.capture(autofocus=False, download=True)
    worker.set_naming(True, "plate_", 40)
    worker.capture(autofocus=False, download=True)
    assert _saved(tmp_path) == ["page_0001.NEF", "plate_0040.NEF"]
    assert worker.naming.number == 41


def test_shooting_into_a_folder_carries_on_past_what_is_in_it(tmp_path):
    (tmp_path / "page_0001.NEF").write_bytes(b"")
    worker = _worker(tmp_path)
    worker.capture(autofocus=False, download=True)
    assert _saved(tmp_path) == ["page_0001.NEF", "page_0002.NEF"]


def test_the_camera_keeps_naming_while_the_sequence_is_off(tmp_path):
    worker = _worker(tmp_path, enabled=False)
    worker.capture(autofocus=False, download=True)
    assert _saved(tmp_path) == ["DSC_0001.NEF"]


def test_where_the_counter_reached_is_reported_back(tmp_path):
    worker = _worker(tmp_path, number=84)
    seen = []
    worker.nextNumber.connect(seen.append)
    worker.capture(autofocus=False, download=True)
    assert seen == [85]


def test_a_shot_that_produced_nothing_does_not_eat_a_number(tmp_path):
    worker = _worker(tmp_path, files=())
    worker.capture(autofocus=False, download=True)
    assert worker.naming.number == 1


# -- the window ------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    made = QApplication.instance() or QApplication([])
    # Keep the test's settings out of the real ones.
    QApplication.setOrganizationName("scanny-tests")
    QApplication.setApplicationName("scanny-tests")
    QSettings().clear()
    yield made
    QSettings().clear()


@pytest.fixture
def window(app):
    QSettings().clear()
    made = MainWindow()
    made.show()
    QApplication.processEvents()
    yield made
    made.close()


def test_the_camera_keeps_naming_until_it_is_asked_not_to(window):
    assert not window.rename_downloads.isChecked()
    assert window.name_prefix.text() == DEFAULT_PREFIX
    assert window.name_number.value() == 1


def test_the_prefix_and_counter_only_matter_while_it_is_on(window):
    assert not window.name_prefix.isEnabled()
    assert not window.name_number.isEnabled()
    window.rename_downloads.setChecked(True)
    assert window.name_prefix.isEnabled()
    assert window.name_number.isEnabled()


def test_switching_it_on_tells_the_worker_the_whole_of_it(window):
    asked = []
    window.requestNaming.connect(lambda on, prefix, n: asked.append((on, prefix, n)))
    window.name_prefix.setText("page_")
    window.name_number.setValue(84)
    window.rename_downloads.setChecked(True)
    assert asked[-1] == (True, "page_", 84)


def test_an_override_typed_mid_batch_goes_out_at_once(window):
    window.rename_downloads.setChecked(True)
    asked = []
    window.requestNaming.connect(lambda on, prefix, n: asked.append((on, prefix, n)))
    window.name_number.setValue(40)
    assert asked[-1] == (True, DEFAULT_PREFIX, 40)


def test_the_next_name_is_shown_before_anything_is_shot(window):
    window.name_prefix.setText("page_")
    window.name_number.setValue(84)
    window.rename_downloads.setChecked(True)
    assert "page_0084" in window.name_preview.text()


def test_with_it_off_the_preview_says_whose_names_they_are(window):
    assert "camera" in window.name_preview.text()


def test_the_counter_follows_the_worker_after_a_shot(window):
    window.rename_downloads.setChecked(True)
    window._on_next_number(85)
    assert window.name_number.value() == 85
    assert "0085" in window.name_preview.text()


def test_the_counter_moving_by_itself_is_not_an_override(window):
    """Echoing it back would only hand the worker what it just reported."""
    window.rename_downloads.setChecked(True)
    asked = []
    window.requestNaming.connect(lambda on, prefix, n: asked.append((on, prefix, n)))
    window._on_next_number(85)
    assert asked == []


def test_where_the_counter_reached_is_remembered_for_the_next_run(window):
    window.rename_downloads.setChecked(True)
    window.name_prefix.setText("page_")
    window._on_next_number(85)

    resumed = MainWindow()
    try:
        assert resumed.rename_downloads.isChecked()
        assert resumed.name_prefix.text() == "page_"
        assert resumed.name_number.value() == 85
    finally:
        resumed.close()
