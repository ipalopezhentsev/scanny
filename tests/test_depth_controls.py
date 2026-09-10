"""The window's side of the depth map.

The panel's job is small: settle the shape of the sweep before it starts,
show the map as it fills in, and let the grid it is drawn at be changed
afterwards without sweeping again. What is worth testing is that the controls
that decide the sweep are settled once and then left alone, and that the one
that only decides the drawing is not.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.ui import main_window as mw  # noqa: E402
from scanny.ui.depth import LEVELS, DepthMap, Tiling  # noqa: E402


@pytest.fixture(scope="module")
def app():
    made = QApplication.instance() or QApplication([])
    QApplication.setOrganizationName("scanny-tests")
    QApplication.setApplicationName("scanny-tests")
    QSettings().clear()
    yield made
    QSettings().clear()


class _NoWorker:
    """Enough of the worker for the panel: where a save starts from."""

    def __init__(self, directory) -> None:
        self.save_directory = directory


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(
        mw.MainWindow,
        "_start_worker",
        lambda self: setattr(self, "worker", _NoWorker(tmp_path)),
    )
    QSettings().clear()
    made = mw.MainWindow()
    made.show()
    QApplication.processEvents()
    yield made
    made.hide()
    made.deleteLater()
    QApplication.processEvents()


def _map(rows: int = 9, cols: int = 16) -> DepthMap:
    depth = np.linspace(1000.0, 4000.0, rows * cols).reshape(rows, cols)
    # One zone with nothing in it, away from either end so the two ends of the
    # scale stay round numbers the tests can look for.
    depth[rows // 2, cols // 2] = np.nan
    return DepthMap(
        depth=depth,
        strength=np.ones((rows, cols)),
        borrowed=np.zeros((rows, cols), bool),
        edge=np.zeros((rows, cols), bool),
        swept=(0.0, 6000.0),
        level=0,
        tiling=Tiling(
            levels=1, rows=rows, cols=cols, tile_height=40, tile_width=40, top=0, left=0
        ),
    )


def test_there_is_nothing_to_save_until_something_was_swept(window):
    assert window.depth_button.text() == "Map depth"
    assert not window.depth_save_button.isEnabled()
    assert window.depth_view.depth_map is None


def test_asking_for_a_map_sends_the_sweep_the_lens_minimum_step(window):
    """The refinement has to stop where the lens stops answering, and what a
    step of this lens is worth is the user's setting, not the worker's guess."""
    asked = []
    window.requestDepthMap.connect(lambda *args: asked.append(args))
    window._focus_steps["minimum"].setValue(18)
    window.depth_samples.setValue(24)
    window.depth_passes.setValue(2)
    window.depth_button.click()
    assert asked == [(24, 2, 18, False)]


def test_the_button_stops_a_sweep_that_is_running(window):
    stopped = []
    window.requestDepthCancel.connect(lambda: stopped.append(True))
    window._on_depth_changed(True)
    assert window.depth_button.text() == "Stop"
    window.depth_button.click()
    assert stopped == [True]


def test_the_shape_of_the_sweep_is_settled_before_it_starts(window):
    """Changing how many stops a running sweep makes would mean nothing: the
    pass is already planned, and the map is the passes taken together."""
    window._on_depth_changed(True)
    assert not window.depth_samples.isEnabled()
    assert not window.depth_passes.isEnabled()
    # The grid the map is drawn at is not part of the sweep, and is worth
    # changing while watching one.
    assert window.depth_detail.isEnabled()
    window._on_depth_changed(False)
    assert window.depth_samples.isEnabled()


def test_a_map_arriving_is_shown_described_and_made_saveable(window):
    window._on_depth_map(_map())
    assert window.depth_view.depth_map is not None
    assert "16x9 zones" in window.depth_label.text()
    assert window.depth_save_button.isEnabled()


def test_a_map_going_away_takes_the_readout_with_it(window):
    window._on_depth_map(_map())
    window._on_depth_map(None)
    assert window.depth_view.depth_map is None
    assert window.depth_label.text() == ""
    assert not window.depth_save_button.isEnabled()


def test_the_detail_asked_for_is_a_level_of_the_pyramid_and_is_remembered(window):
    asked = []
    window.requestDepthDetail.connect(asked.append)
    window.depth_detail.setCurrentIndex(0)
    assert asked == [0]
    assert QSettings().value("depth/detail") in (0, "0")
    levels = [
        window.depth_detail.itemData(index)
        for index in range(window.depth_detail.count())
    ]
    assert levels == sorted(levels)
    assert levels[-1] == LEVELS - 1


def test_the_sweep_it_offers_is_remembered_between_runs(window, app, monkeypatch):
    window.depth_samples.setValue(96)
    window.depth_passes.setValue(4)
    window.depth_button.click()  # the values are stored when the sweep is asked for
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    again = mw.MainWindow()
    try:
        assert again.depth_samples.value() == 96
        assert again.depth_passes.value() == 4
    finally:
        again.deleteLater()
        QApplication.processEvents()


def test_a_stored_value_the_control_no_longer_offers_falls_back(window, monkeypatch):
    QSettings().setValue("depth/passes", 99)
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    again = mw.MainWindow()
    try:
        assert again.depth_passes.value() == 3
    finally:
        again.deleteLater()
        QApplication.processEvents()


def test_saving_writes_the_picture_and_the_numbers_beside_it(
    window, tmp_path, monkeypatch
):
    """Two files, because they are wanted for different things: one to look at
    and one to read the drive-step positions back out of."""
    window._on_depth_map(_map())
    monkeypatch.setattr(
        mw.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(tmp_path / "depth.png"), "")),
    )
    window._save_depth_map()
    picture, steps = tmp_path / "depth.png", tmp_path / "depth-steps.png"
    assert picture.exists() and steps.exists()
    # The greyscale one is the measurement, at the grid's own size and depth.
    header = steps.read_bytes()
    assert header[24] == 16 and header[25] == 0, "not a 16-bit greyscale PNG"
    said = window.statusBar().currentMessage()
    assert "1000" in said and "4000" in said, said


def test_there_is_nothing_to_narrow_to_until_a_map_has_been_made(window):
    """Offering the choice before then invites picking it and being given the
    whole travel anyway."""
    assert not window.depth_range.isEnabled()
    window._on_depth_map(_map())
    assert window.depth_range.isEnabled()


def test_asking_to_go_again_says_so_to_the_worker(window):
    asked = []
    window.requestDepthMap.connect(lambda *args: asked.append(args))
    window._on_depth_map(_map())
    window.depth_range.setCurrentIndex(1)
    window.depth_button.click()
    assert asked and asked[-1][-1] is True


def test_the_range_is_put_away_while_a_sweep_runs(window):
    window._on_depth_map(_map())
    window._on_depth_changed(True)
    assert not window.depth_range.isEnabled()
    window._on_depth_changed(False)
    assert window.depth_range.isEnabled()
