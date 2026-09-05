"""The user owns the focus increments, so the window must resolve them.

The live-view widget names an increment; how many drive steps that is worth is
a setting, remembered between runs. Nothing outside the window should have a
hardcoded step count.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QSpinBox  # noqa: E402

from scanny.camera.nikon import NikonCamera  # noqa: E402
from scanny.ui.main_window import MainWindow  # noqa: E402


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
    made = MainWindow()
    made.show()
    QApplication.processEvents()
    yield made
    made.close()


def test_every_increment_has_an_editor(window):
    assert set(window._focus_steps) == set(NikonCamera.FOCUS_INCREMENTS)
    for spin in window._focus_steps.values():
        assert isinstance(spin, QSpinBox)


def test_editors_start_at_the_defaults(window):
    for name, spin in window._focus_steps.items():
        assert spin.value() == NikonCamera.FOCUS_STEP_DEFAULTS[name]


def test_a_step_uses_the_value_the_user_set(window):
    sent = []
    window.requestDriveFocus.connect(sent.append)
    window._focus_steps["fine"].setValue(77)
    window._step_focus("fine", 1)
    window._step_focus("fine", -1)
    assert sent == [77, -77]


def test_every_increment_is_resolved_from_its_editor(window):
    sent = []
    window.requestDriveFocus.connect(sent.append)
    for index, name in enumerate(NikonCamera.FOCUS_INCREMENTS):
        window._focus_steps[name].setValue(index + 2)
        window._step_focus(name, 1)
    assert sent == [2, 3, 4, 5]


def test_keyboard_goes_through_the_same_setting(window):
    sent = []
    window.requestDriveFocus.connect(sent.append)
    window._focus_steps["minimum"].setValue(3)
    window.view.setFocus()
    QTest.keyClick(window.view, Qt.Key.Key_BracketRight)
    QApplication.processEvents()
    assert sent[-1] == 3


def test_values_are_remembered_between_runs(app):
    first = MainWindow()
    first._focus_steps["coarse"].setValue(4321)
    QApplication.processEvents()
    first.close()

    second = MainWindow()
    try:
        assert second._focus_steps["coarse"].value() == 4321
    finally:
        second.close()


def test_a_nonsense_stored_value_falls_back_to_the_default(app):
    QSettings().setValue("focus/fine", "not a number")
    window = MainWindow()
    try:
        assert window._focus_steps["fine"].value() == (
            NikonCamera.FOCUS_STEP_DEFAULTS["fine"]
        )
    finally:
        window.close()
        QSettings().remove("focus/fine")


def test_editors_keep_the_keyboard_available_for_typing(window):
    # They must be focusable, unlike the buttons, or they cannot be edited.
    for spin in window._focus_steps.values():
        assert spin.focusPolicy() != Qt.FocusPolicy.NoFocus
