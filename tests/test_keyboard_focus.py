"""The image must keep the keyboard, or every shortcut silently stops working.

Clicking a button or checkbox used to move keyboard focus off the live-view
widget, after which the focus, pan and zoom keys did nothing until the image
was clicked again -- which reads as "the shortcuts only work sometimes".
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QPushButton,
    QSlider,
)

from scanny.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def window(app):
    made = MainWindow()
    made.show()
    QApplication.processEvents()
    yield made
    made.close()


@pytest.mark.parametrize("kind", [QPushButton, QCheckBox, QSlider])
def test_pointer_controls_never_take_the_keyboard(window, kind):
    controls = window.findChildren(kind)
    assert controls, f"expected some {kind.__name__} in the window"
    for control in controls:
        assert control.focusPolicy() == Qt.FocusPolicy.NoFocus, (
            f"{kind.__name__} {control.text() if hasattr(control, 'text') else ''!r} "
            "would steal the keyboard from the image"
        )


def test_the_image_can_take_the_keyboard(window):
    assert window.view.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_clicking_a_button_leaves_the_keyboard_on_the_image(window):
    window.view.setFocus()
    QApplication.processEvents()
    assert window.view.hasFocus()
    window.af_button.click()
    QApplication.processEvents()
    assert window.view.hasFocus(), "autofocus button took the keyboard"


def test_combo_boxes_still_take_focus_for_their_popup(window):
    # They need it to be usable; they just have to give it back afterwards.
    combos = window.findChildren(QComboBox)
    if combos:
        assert combos[0].focusPolicy() != Qt.FocusPolicy.NoFocus
