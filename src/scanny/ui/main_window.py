"""The application window: live view on the left, camera controls on the right."""

from __future__ import annotations

import time

from PySide6.QtCore import QSettings, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..camera.nikon import NikonCamera, Setting
from .liveview import LiveViewWidget
from .worker import CameraWorker

__all__ = ["MainWindow"]

#: The slider steps through the levels the body accepts, not a raw 0-7 range,
#: so every position is reachable.
_ZOOM_LEVELS = NikonCamera.ZOOM_LEVELS


class MainWindow(QMainWindow):
    """Drives a :class:`CameraWorker` living on its own thread."""

    # Requests to the worker. Signals rather than direct calls, so the work
    # happens on the worker's thread instead of blocking the interface.
    requestConnect = Signal()
    requestDisconnect = Signal()
    requestStartLiveView = Signal()
    requestStopLiveView = Signal()
    requestAutofocus = Signal()
    requestMovePoint = Signal(float, float)
    requestToggleZoom = Signal()
    requestDriveFocus = Signal(int)
    requestZoom = Signal(int)
    requestZoomStep = Signal(int)
    requestZoomRegion = Signal(float, float, float, float)
    requestSetting = Signal(str, object)
    requestCapture = Signal(bool, bool)
    requestRefresh = Signal()
    requestShutdown = Signal()
    requestResetZoom = Signal()
    requestExposurePreview = Signal(bool)
    requestPan = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("scanny")
        self.resize(1280, 820)

        self._combos: "dict[str, QComboBox]" = {}
        self._level_shown = 0.0
        self._updating_settings = False
        self._live = False

        self._build_ui()
        self._start_worker()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        self.view = LiveViewWidget()
        self.view.pointSelected.connect(self._on_point_selected)
        self.view.focusRequested.connect(self._on_focus_requested)
        self.view.regionSelected.connect(self.requestZoomRegion)
        self.view.zoomStepped.connect(self.requestZoomStep)
        self.view.zoomReset.connect(self.requestResetZoom)
        self.view.zoomToggled.connect(self.requestToggleZoom)
        self.view.panStepped.connect(self.requestPan)
        self.view.focusStepped.connect(self._step_focus)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.view, 1)
        layout.addWidget(self._build_sidebar(), 0)
        self.setCentralWidget(central)

        # Buttons, checkboxes and the slider are all operated by pointer, so
        # none of them should take the keyboard away from the image: doing so
        # silently kills the focus, pan and zoom keys until the image is
        # clicked again.
        for kind in (QPushButton, QCheckBox, QSlider):
            for control in self.findChildren(kind):
                control.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.statusBar().showMessage("Starting...")
        self.level_label = QLabel("")
        self.level_label.setToolTip(
            "The camera's level sensor: roll, then pitch. Green when level."
        )
        self.statusBar().addPermanentWidget(self.level_label)
        self.fps_label = QLabel("")
        self.fps_label.setStyleSheet("color: #888;")
        self.statusBar().addPermanentWidget(self.fps_label)
        self._build_menu()

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(310)
        column = QVBoxLayout(sidebar)
        column.setContentsMargins(0, 0, 0, 0)

        self.camera_label = QLabel("Looking for a camera...")
        self.camera_label.setWordWrap(True)
        self.camera_label.setStyleSheet("color: #888;")
        column.addWidget(self.camera_label)

        column.addWidget(self._build_live_view_box())
        column.addWidget(self._build_focus_box())
        column.addWidget(self._build_exposure_box())
        column.addWidget(self._build_capture_box())
        column.addStretch(1)
        return sidebar

    def _build_live_view_box(self) -> QGroupBox:
        box = QGroupBox("Live view")
        layout = QVBoxLayout(box)

        self.live_button = QPushButton("Start live view")
        self.live_button.clicked.connect(self._toggle_live_view)
        layout.addWidget(self.live_button)

        self.zoom_label = QLabel("Zoom: full frame")
        layout.addWidget(self.zoom_label)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(0, len(_ZOOM_LEVELS) - 1)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(1)
        self.zoom_slider.sliderReleased.connect(
            lambda: self.requestZoom.emit(_ZOOM_LEVELS[self.zoom_slider.value()])
        )
        layout.addWidget(self.zoom_slider)

        self.reset_zoom_button = QPushButton("Reset zoom to full frame")
        self.reset_zoom_button.clicked.connect(self.requestResetZoom)
        layout.addWidget(self.reset_zoom_button)

        hint = QLabel(
            "Click to move the focus box, double-click to focus there. "
            "Right-click toggles full magnification. Drag a box to magnify it. "
            "Scroll to zoom, arrow keys pan, Enter focuses, Esc for the "
            "whole frame."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)

        self.exposure_preview = QCheckBox("Exposure preview")
        self.exposure_preview.setChecked(True)
        self.exposure_preview.setToolTip(
            "Stop the lens down to the chosen aperture and show the exposure "
            "that would actually be taken. Turn off and the camera normalises "
            "brightness instead, so aperture and shutter make no visible "
            "difference."
        )
        self.exposure_preview.toggled.connect(self.requestExposurePreview)
        layout.addWidget(self.exposure_preview)

        return box

    def _build_focus_box(self) -> QGroupBox:
        box = QGroupBox("Focus")
        layout = QVBoxLayout(box)

        self.af_button = QPushButton("Autofocus")
        self.af_button.setToolTip("Double-click the image, or press Enter, to focus")
        self.af_button.clicked.connect(self.requestAutofocus)
        layout.addWidget(self.af_button)
        layout.addWidget(self._build_manual_focus())

        hint = QLabel(
            "Manual focus: < is nearer, > is further. Set how many drive steps "
            "each increment is worth. Keys [ ] , . < > drive the first three."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)
        return box

    def _build_manual_focus(self) -> QWidget:
        """A column per increment: its name, its two buttons, and its size.

        Grouping by increment rather than laying all eight buttons out as one
        row keeps each value next to the buttons it drives, and leaves the
        labels room to render -- at eight across, a sidebar this wide elides
        them.
        """
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)

        self._focus_buttons: "dict[str, list[QPushButton]]" = {}
        self._focus_steps: "dict[str, QSpinBox]" = {}

        for index, name in enumerate(NikonCamera.FOCUS_INCREMENTS):
            left = index * 2
            caption = QLabel(name)
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption.setStyleSheet("color: #888; font-size: 10px;")
            grid.addWidget(caption, 0, left, 1, 2)

            for offset, direction, label in ((0, -1, "<"), (1, 1, ">")):
                grid.addWidget(self._focus_button(name, direction, label), 1, left + offset)

            spin = QSpinBox()
            spin.setRange(1, 30000)
            spin.setValue(self._stored_focus_step(name))
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            spin.setToolTip(f"Drive steps for the {name} increment")
            spin.valueChanged.connect(
                lambda value, n=name: self._on_focus_step_changed(n, value)
            )
            # A spin box has to take the keyboard to be typed into, so give it
            # back once the value is settled, or the focus keys stay dead.
            spin.editingFinished.connect(self.view.setFocus)
            grid.addWidget(spin, 2, left, 1, 2)
            self._focus_steps[name] = spin

        for name in NikonCamera.FOCUS_INCREMENTS:
            self._refresh_focus_tooltips(name)
        return holder

    def _focus_button(self, name: str, direction: int, label: str) -> QPushButton:
        button = QPushButton(label)
        button.setAutoRepeat(True)
        button.setAutoRepeatDelay(400)
        button.setAutoRepeatInterval(150)
        button.setMaximumWidth(34)
        button.setProperty("focusDirection", direction)
        button.clicked.connect(
            lambda _checked=False, n=name, d=direction: self._step_focus(n, d)
        )
        self._focus_buttons.setdefault(name, []).append(button)
        return button

    def _stored_focus_step(self, name: str) -> int:
        default = NikonCamera.FOCUS_STEP_DEFAULTS[name]
        try:
            stored = int(QSettings().value(f"focus/{name}", default))
        except (TypeError, ValueError):
            return default
        return stored if 1 <= stored <= 30000 else default

    def _on_focus_step_changed(self, name: str, value: int) -> None:
        QSettings().setValue(f"focus/{name}", value)
        self._refresh_focus_tooltips(name)

    def _refresh_focus_tooltips(self, name: str) -> None:
        steps = self._focus_steps[name].value()
        plural = "" if steps == 1 else "s"
        for button in self._focus_buttons.get(name, []):
            where = "nearer" if button.property("focusDirection") < 0 else "further"
            button.setToolTip(f"Focus {where}, {name} ({steps} step{plural})")

    @Slot(str, int)
    def _step_focus(self, name: str, direction: int) -> None:
        """Turn an increment name and a direction into signed drive steps."""
        spin = self._focus_steps.get(name)
        steps = spin.value() if spin else NikonCamera.FOCUS_STEP_DEFAULTS.get(name, 1)
        self.requestDriveFocus.emit(steps * (1 if direction >= 0 else -1))

    def _build_exposure_box(self) -> QGroupBox:
        box = QGroupBox("Exposure")
        self.exposure_form = QFormLayout(box)
        self.exposure_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        placeholder = QLabel("Not connected")
        placeholder.setStyleSheet("color: #888;")
        self.exposure_form.addRow(placeholder)
        return box

    def _build_capture_box(self) -> QGroupBox:
        box = QGroupBox("Capture")
        layout = QVBoxLayout(box)

        self.af_before_shot = QCheckBox("Autofocus before shooting")
        layout.addWidget(self.af_before_shot)

        self.download_after_shot = QCheckBox("Download to computer")
        self.download_after_shot.setChecked(True)
        layout.addWidget(self.download_after_shot)

        self.save_dir_label = QLabel()
        self.save_dir_label.setWordWrap(True)
        self.save_dir_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.save_dir_label)

        choose = QPushButton("Change folder...")
        choose.clicked.connect(self._choose_save_directory)
        layout.addWidget(choose)

        self.shoot_button = QPushButton("Take photo")
        self.shoot_button.setMinimumHeight(40)
        self.shoot_button.clicked.connect(self._shoot)
        layout.addWidget(self.shoot_button)
        return box

    def _build_menu(self) -> None:
        camera_menu = self.menuBar().addMenu("&Camera")

        reconnect = QAction("&Reconnect", self)
        reconnect.triggered.connect(self._reconnect)
        camera_menu.addAction(reconnect)

        refresh = QAction("Refresh &settings", self)
        refresh.setShortcut(QKeySequence.StandardKey.Refresh)
        refresh.triggered.connect(self.requestRefresh)
        camera_menu.addAction(refresh)

        camera_menu.addSeparator()
        shoot = QAction("Take &photo", self)
        shoot.setShortcut(QKeySequence("Ctrl+Return"))
        shoot.triggered.connect(self._shoot)
        camera_menu.addAction(shoot)

        focus = QAction("&Autofocus", self)
        focus.setShortcut(QKeySequence("Ctrl+F"))
        focus.triggered.connect(self.requestAutofocus)
        camera_menu.addAction(focus)

        camera_menu.addSeparator()
        for label, direction, shortcut in (
            ("Focus nearer", -1, "Ctrl+,"),
            ("Focus further", 1, "Ctrl+."),
        ):
            action = QAction(label, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(
                lambda _checked=False, d=direction: self._step_focus("medium", d)
            )
            camera_menu.addAction(action)

    # -- worker wiring -----------------------------------------------------

    def _start_worker(self) -> None:
        self._thread = QThread(self)
        self._thread.setObjectName("camera")
        self.worker = CameraWorker()
        self.worker.moveToThread(self._thread)

        self.requestConnect.connect(self.worker.connect_camera)
        self.requestDisconnect.connect(self.worker.disconnect_camera)
        self.requestStartLiveView.connect(self.worker.start_live_view)
        self.requestStopLiveView.connect(self.worker.stop_live_view)
        self.requestAutofocus.connect(self.worker.autofocus)
        self.requestMovePoint.connect(self.worker.move_point)
        self.requestToggleZoom.connect(self.worker.toggle_zoom)
        self.requestZoom.connect(self.worker.set_zoom)
        self.requestZoomStep.connect(self.worker.step_zoom)
        self.requestZoomRegion.connect(self.worker.zoom_to_region)
        self.requestSetting.connect(self.worker.set_setting)
        self.requestCapture.connect(self.worker.capture)
        self.requestRefresh.connect(self.worker.refresh_settings)
        self.requestShutdown.connect(self.worker.shutdown)
        self.requestResetZoom.connect(self.worker.reset_zoom)
        self.requestExposurePreview.connect(self.worker.set_exposure_preview)
        self.requestPan.connect(self.worker.pan)
        self.requestDriveFocus.connect(self.worker.drive_focus)

        self.worker.connected.connect(self._on_connected)
        self.worker.disconnected.connect(self._on_disconnected)
        self.worker.frameReady.connect(self.view.set_frame)
        self.worker.frameReady.connect(self._on_frame_level)
        self.worker.settingsReady.connect(self._on_settings)
        self.worker.liveViewChanged.connect(self._on_live_view_changed)
        self.worker.zoomChanged.connect(self._on_zoom_changed)
        self.worker.focusStateChanged.connect(self.view.set_focus_state)
        self.worker.fpsChanged.connect(self._on_fps)
        self.worker.exposurePreviewChanged.connect(self._on_exposure_preview)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.failed.connect(self._on_failed)

        # The worker sets up its own COM apartment before touching the camera.
        self._thread.started.connect(self.worker.initialise)
        self._thread.finished.connect(self.worker.deleteLater)

        self._thread.start()
        self.save_dir_label.setText(f"Saving to {self.worker.save_directory}")

    # -- slots -------------------------------------------------------------

    @Slot(str)
    def _on_connected(self, description: str) -> None:
        self.camera_label.setText(description)
        # Weight only, no colour: the window follows the system theme, and a
        # hardcoded light grey vanishes against a light background.
        self.camera_label.setStyleSheet("font-weight: 600;")
        self.view.clear("Press Start live view")

    @Slot()
    def _on_disconnected(self) -> None:
        self.camera_label.setText("No camera connected")
        self.camera_label.setStyleSheet("color: #888;")
        self.view.clear("Not connected")

    @Slot(bool)
    def _on_live_view_changed(self, active: bool) -> None:
        self._live = active
        self.live_button.setText("Stop live view" if active else "Start live view")
        if active:
            self.view.setFocus()
        if not active:
            self.view.clear("Live view stopped")
            self.view.set_focus_state("idle")

    @Slot(int)
    def _on_zoom_changed(self, level: int) -> None:
        index = _ZOOM_LEVELS.index(level) if level in _ZOOM_LEVELS else 0
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(index)
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(
            "Zoom: full frame" if level == 0 else f"Zoom: level {level}"
        )

    @Slot(object)
    def _on_frame_level(self, frame) -> None:
        """Show the level sensor, throttled -- 30 updates a second is unreadable."""
        now = time.monotonic()
        if now - self._level_shown < 0.2:
            return
        self._level_shown = now
        if frame.roll is None and frame.pitch is None:
            self.level_label.setText("")
            return
        parts = []
        for name, value in (("roll", frame.roll), ("pitch", frame.pitch)):
            parts.append(f"{name} {value:+d}°" if value is not None else f"{name} --")
        self.level_label.setText("  ".join(parts))
        self.level_label.setStyleSheet(
            "color: #2e9e4f; font-weight: 600;" if frame.is_level else "color: #888;"
        )

    @Slot(float, int, int)
    def _on_fps(self, fps: float, width: int, height: int) -> None:
        self.fps_label.setText(
            f"{width}x{height}  -  {fps:.1f} fps" if fps > 0 else ""
        )

    @Slot(bool)
    def _on_exposure_preview(self, enabled: bool) -> None:
        self.exposure_preview.blockSignals(True)
        self.exposure_preview.setChecked(enabled)
        self.exposure_preview.blockSignals(False)

    @Slot(object)
    def _on_settings(self, settings: "list[Setting]") -> None:
        self._updating_settings = True
        try:
            self._rebuild_exposure_form(settings)
        finally:
            self._updating_settings = False

    def _rebuild_exposure_form(self, settings: "list[Setting]") -> None:
        existing = {s.name for s in settings}
        if set(self._combos) != existing:
            while self.exposure_form.rowCount():
                self.exposure_form.removeRow(0)
            self._combos.clear()
            for setting in settings:
                combo = QComboBox()
                combo.setEnabled(setting.writable and bool(setting.choices))
                combo.activated.connect(
                    lambda _index, name=setting.name: self._on_setting_chosen(name)
                )
                self._combos[setting.name] = combo
                self.exposure_form.addRow(setting.name, combo)

        for setting in settings:
            combo = self._combos[setting.name]
            labels = setting.choice_labels or [setting.label]
            if [combo.itemText(i) for i in range(combo.count())] != labels:
                combo.blockSignals(True)
                combo.clear()
                for value, label in setting.choices or ((setting.value, setting.label),):
                    combo.addItem(label, value)
                combo.blockSignals(False)
            index = combo.findText(setting.label)
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
            combo.setToolTip(
                "Set on the camera body" if not setting.writable else setting.name
            )

    def _on_setting_chosen(self, name: str) -> None:
        if self._updating_settings:
            return
        combo = self._combos.get(name)
        if combo is None:
            return
        value = combo.currentData()
        if value is not None:
            self.requestSetting.emit(name, value)
        # The combo needed the keyboard for its popup; give it back to the
        # image so the shortcuts keep working.
        self.view.setFocus()

    @Slot(float, float)
    def _on_point_selected(self, nx: float, ny: float) -> None:
        if self._live:
            self.requestMovePoint.emit(nx, ny)

    @Slot()
    def _on_focus_requested(self) -> None:
        # The click that opened the double click has already put the rectangle
        # where the user clicked, so this focuses where it now is.
        if self._live:
            self.requestAutofocus.emit()

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)

    # -- actions -----------------------------------------------------------

    def _toggle_live_view(self) -> None:
        if self._live:
            self.requestStopLiveView.emit()
        else:
            self.requestStartLiveView.emit()

    def _shoot(self) -> None:
        self.requestCapture.emit(
            self.af_before_shot.isChecked(), self.download_after_shot.isChecked()
        )

    def _reconnect(self) -> None:
        self.requestDisconnect.emit()
        self.requestConnect.emit()

    def _choose_save_directory(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Where should photos be saved?", str(self.worker.save_directory)
        )
        if chosen:
            self.worker.set_save_directory(chosen)
            self.save_dir_label.setText(f"Saving to {chosen}")

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.requestShutdown.emit()
        self._thread.quit()
        if not self._thread.wait(5000):
            self._thread.terminate()
            self._thread.wait(1000)
        super().closeEvent(event)
