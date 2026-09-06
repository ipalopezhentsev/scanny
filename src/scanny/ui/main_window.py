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
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..camera.nikon import NikonCamera, Setting
from .integration import DEFAULT_FRAMES, MAX_FRAMES, MIN_FRAMES
from .liveview import LiveViewWidget
from .trend import TrendGraph
from .worker import CameraWorker

__all__ = ["MainWindow", "WrappedLabel"]


def _reading(value: float) -> str:
    """A reading, with a decimal only while it is small enough to want one."""
    return f"{value:.0f}" if value >= 100 else f"{value:.1f}"

#: The slider steps through the levels the body accepts, not a raw 0-7 range,
#: so every position is reachable.
_ZOOM_LEVELS = NikonCamera.ZOOM_LEVELS

#: How wide the controls are laid out to be. The scroll area around them is
#: this plus a scrollbar, so they keep the same width whether it shows or not.
_SIDEBAR_WIDTH = 310

#: The increment focus is walked in. Its step count is the user's, from the
#: Focus panel, so the walk moves in a size that means something for their
#: lens -- and making the walk finer or coarser is a matter of changing what
#: "minimum" means.
_FINE_INCREMENT = "minimum"

#: Where a freshly switched-on measurement area starts: a third of the frame,
#: in the middle of it. Small enough to be worth having over the whole frame,
#: big enough to have some subject in it before it is dragged anywhere.
_DEFAULT_MEASURE_AREA = (1 / 3, 1 / 3, 1 / 3, 1 / 3)

#: What the camera delivers, near enough, for working out what integrating a
#: given number of frames will cost in frame rate before it is switched on.
_SOURCE_FPS = 30.0


class WrappedLabel(QLabel):
    """A word-wrapped label that takes up exactly the height its text needs.

    Wrapping makes a label's height depend on its width, and Qt has two ways
    of dealing with that. The one a plain wrapped label asks for -- answer
    height-for-width questions and let the layout work it out -- goes wrong in
    a panel that is short of room: the layout is told a height measured at the
    label's *hint* width rather than the width it will be given, hands out the
    space it was told about, and the last line of every hint disappears. The
    shortfall is taken out of the controls around it too, which is what
    flattens a spin box to two thirds of its height.

    So the height is resolved here instead, against the width the label
    actually has, and fixed. The panel's minimum height then includes the room
    the text really occupies, which is the number a scroll area around it goes
    by when it decides how tall to make the panel.

    Safe because the column is a fixed width: height follows width, and
    nothing follows height, so there is no loop to fall into.
    """

    def __init__(self, text: str = "", parent: "QWidget | None" = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        policy = self.sizePolicy()
        policy.setHeightForWidth(False)
        self.setSizePolicy(policy)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        super().setText(text)
        self._fit()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if self.width() > 0:
            self.setFixedHeight(self.heightForWidth(self.width()))


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
    requestIntegration = Signal(bool, int)
    requestSharpness = Signal(bool)
    requestSharpnessReset = Signal()
    requestSharpnessArea = Signal(object)  # (x, y, w, h) fractions, or None
    requestFineTune = Signal(int)  # the one increment to walk in
    requestHuntCancel = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("scanny")
        self.resize(1280, 820)

        self._combos: "dict[str, QComboBox]" = {}
        self._level_shown = 0.0
        self._updating_settings = False
        self._live = False
        self._fps_shown = (0.0, 0, 0)
        self._sharpness_shown = 0.0
        self._hunting = False
        self._measure_area = _DEFAULT_MEASURE_AREA

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
        self.view.measureAreaSelected.connect(self._on_measure_area_selected)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.view, 1)
        layout.addWidget(self._build_sidebar_scroller(), 0)
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

    def _build_sidebar_scroller(self) -> QScrollArea:
        """The sidebar, free to be as tall as its contents ask to be.

        A connected camera fills the exposure form with eight rows, and by then
        the panel wants more height than the window has. A plain layout answers
        that by squeezing every widget below its size hint -- which flattens the
        spin boxes to two thirds of their height and cuts the last line off the
        wrapped hints, both of which read as controls that have been damaged
        rather than a panel that is too long. Scrolling gives each control the
        size it asked for and puts the shortfall somewhere the user can see and
        act on.
        """
        scroller = QScrollArea()
        scroller.setWidget(self._build_sidebar())
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Room for the scrollbar on top of the controls' own width, so they do
        # not shuffle sideways when it appears.
        scroller.setFixedWidth(
            _SIDEBAR_WIDTH
            + scroller.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        )
        # Like every other pointer-operated control here, it must not take the
        # keyboard away from the image.
        scroller.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return scroller

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setMinimumWidth(_SIDEBAR_WIDTH)
        column = QVBoxLayout(sidebar)
        column.setContentsMargins(0, 0, 0, 0)

        self.camera_label = WrappedLabel("Looking for a camera...")
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

        hint = WrappedLabel(
            "Click to move the focus box, double-click to focus there. "
            "Right-click toggles full magnification. Drag a box to magnify it. "
            "Scroll to zoom, arrow keys pan, Enter focuses, Esc for the "
            "whole frame."
        )
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

        layout.addWidget(self._build_integration())
        return box

    def _build_integration(self) -> QWidget:
        """The noise-integration control: a switch, a count, and what it costs.

        The cost is spelled out under the controls rather than left to be
        discovered, because it is the whole trade: every frame added to the
        stack divides the frame rate again.
        """
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.integrate = QCheckBox("Integrate")
        self.integrate.setToolTip(
            "Average several frames into each displayed image. The noise is "
            "different in every frame and the scene is not, so it cancels out "
            "and the picture gets cleaner -- at the cost of frame rate."
        )
        row.addWidget(self.integrate)

        self.integrate_frames = QSpinBox()
        self.integrate_frames.setRange(MIN_FRAMES, MAX_FRAMES)
        self.integrate_frames.setSuffix(" frames")
        self.integrate_frames.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.integrate_frames.setToolTip("How many frames go into each image")
        # A spin box has to take the keyboard to be typed into; hand it back
        # once the value is settled, or the image's shortcuts stay dead.
        self.integrate_frames.editingFinished.connect(self.view.setFocus)
        row.addWidget(self.integrate_frames, 1)
        column.addLayout(row)

        self.integrate_hint = WrappedLabel()
        self.integrate_hint.setStyleSheet("color: #888; font-size: 11px;")
        column.addWidget(self.integrate_hint)

        settings = QSettings()
        self.integrate.setChecked(settings.value("liveview/integrate", False, bool))
        self.integrate_frames.setValue(self._stored_integration_frames())
        self._integrating = self.integrate.isChecked()
        self._describe_integration()
        # Connected last, so restoring the stored values does not count as the
        # user asking for anything.
        self.integrate.toggled.connect(self._on_integration_changed)
        self.integrate_frames.valueChanged.connect(self._on_integration_changed)
        return holder

    def _stored_integration_frames(self) -> int:
        try:
            stored = int(QSettings().value("liveview/frames", DEFAULT_FRAMES))
        except (TypeError, ValueError):
            return DEFAULT_FRAMES
        return stored if MIN_FRAMES <= stored <= MAX_FRAMES else DEFAULT_FRAMES

    def _on_integration_changed(self) -> None:
        settings = QSettings()
        settings.setValue("liveview/integrate", self.integrate.isChecked())
        settings.setValue("liveview/frames", self.integrate_frames.value())
        self._describe_integration()
        # Whatever rate was measured belongs to the old setting. Blank it
        # rather than leave a stale number sitting under a new label -- unless
        # this was only the count changing while integration is switched off,
        # which changes nothing on screen.
        if self.integrate.isChecked() or self._integrating:
            self._fps_shown = (0.0, *self._fps_shown[1:])
        self._integrating = self.integrate.isChecked()
        self._show_fps()
        self.requestIntegration.emit(
            self.integrate.isChecked(), self.integrate_frames.value()
        )
        self.requestSharpness.emit(self.measure_sharpness.isChecked())
        self._apply_measure_area()

    def _describe_integration(self) -> None:
        """Spell out the trade at the number of frames currently chosen.

        Averaging n frames divides random noise by the square root of n, and
        the frame rate by n.
        """
        frames = self.integrate_frames.value()
        self.integrate_frames.setEnabled(self.integrate.isChecked())
        self.integrate_hint.setText(
            f"About {_SOURCE_FPS / frames:.1f} fps, with roughly "
            f"{frames ** 0.5:.1f}x less noise."
        )

    def _build_focus_box(self) -> QGroupBox:
        box = QGroupBox("Focus")
        layout = QVBoxLayout(box)

        self.af_button = QPushButton("Autofocus")
        self.af_button.setToolTip("Double-click the image, or press Enter, to focus")
        self.af_button.clicked.connect(self.requestAutofocus)
        layout.addWidget(self.af_button)
        layout.addWidget(self._build_manual_focus())

        hint = WrappedLabel(
            "Manual focus: < is nearer, > is further. Set how many drive steps "
            "each increment is worth. Keys [ ] , . < > drive the first three."
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)

        layout.addWidget(self._build_sharpness())
        return box

    def _build_sharpness(self) -> QWidget:
        """Focusing by hand against a number, rather than by eye.

        The reading on its own means nothing -- only whether it is higher than
        the last one does -- so what the panel shows is the reading against the
        best this view has managed, as a bar to aim at the top of.
        """
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.measure_sharpness = QCheckBox("Measure sharpness")
        self.measure_sharpness.setToolTip(
            "Score the contrast in the picture on screen, which is what focus "
            "maximises. Drive focus and keep the direction that raises it."
        )
        row.addWidget(self.measure_sharpness)
        row.addStretch(1)

        self.reset_peak_button = QPushButton("Reset best")
        self.reset_peak_button.setToolTip(
            "Forget the best reading and start again from here"
        )
        self.reset_peak_button.clicked.connect(self.requestSharpnessReset)
        row.addWidget(self.reset_peak_button)
        column.addLayout(row)

        self.measure_area = QCheckBox("Only a selected area")
        self.measure_area.setToolTip(
            "Read one rectangle of the picture instead of all of it. Shift-drag "
            "on the image to put it where you want it."
        )
        self.measure_area.toggled.connect(self._on_measure_area_toggled)
        column.addWidget(self.measure_area)

        self.sharpness_trend = TrendGraph()
        self.sharpness_trend.setToolTip(
            "The readings as they arrive, scaled to the range they have lately "
            "covered. Drive focus one step and watch which way the line goes; "
            "it starts again whenever the view, the area or the exposure change, "
            "since readings either side of those are not comparable."
        )
        column.addWidget(self.sharpness_trend)

        self.sharpness_label = QLabel()
        self.sharpness_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self.sharpness_label)

        self.fine_tune_button = QPushButton("Fine tune from here")
        self.fine_tune_button.setToolTip(
            "The same hunt, in minimum steps and nothing coarser, for when "
            "focus is already close. What a macro subject wants: the depth of "
            "focus is a hair, so a search in medium steps spends its time "
            "somewhere no part of the picture could be sharp."
        )
        self.fine_tune_button.clicked.connect(self._toggle_fine_tune)
        column.addWidget(self.fine_tune_button)

        self.sharpness_hint = WrappedLabel(
            "Magnify first, then shift-drag on the image to measure one part of "
            "it. That is how to focus on something smaller than the camera's own "
            "focus box, and smaller than its strongest magnification shows. "
            "Integrate while hunting: the hunt waits for a whole stack before "
            "believing a reading."
        )
        self.sharpness_hint.setStyleSheet("color: #888; font-size: 11px;")
        column.addWidget(self.sharpness_hint)

        settings = QSettings()
        self._measure_area = self._stored_measure_area()
        self.measure_sharpness.setChecked(settings.value("focus/sharpness", False, bool))
        self.measure_area.blockSignals(True)
        self.measure_area.setChecked(settings.value("focus/sharpness_area", False, bool))
        self.measure_area.blockSignals(False)
        self._show_sharpness(0.0, 0.0)
        # The overlay belongs to what was restored, not to the worker starting:
        # the box has to be on the image whether or not a camera turns up.
        self._apply_measure_area()
        self.measure_sharpness.toggled.connect(self._on_sharpness_toggled)
        return holder

    def _stored_measure_area(self) -> "tuple[float, float, float, float]":
        stored = QSettings().value("focus/sharpness_rect", None)
        try:
            area = tuple(float(part) for part in str(stored).split(","))
        except (TypeError, ValueError):
            return _DEFAULT_MEASURE_AREA
        if len(area) != 4 or area[2] <= 0 or area[3] <= 0:
            return _DEFAULT_MEASURE_AREA
        return area

    @Slot(float, float, float, float)
    def _on_measure_area_selected(self, x: float, y: float, w: float, h: float) -> None:
        """A rectangle shift-dragged on the image.

        Drawing one is a clear enough request that it also switches measuring
        on: there is nothing else the gesture could mean.
        """
        self._measure_area = (x, y, w, h)
        settings = QSettings()
        settings.setValue("focus/sharpness_rect", ",".join(f"{v:.5f}" for v in self._measure_area))
        settings.setValue("focus/sharpness", True)
        settings.setValue("focus/sharpness_area", True)
        # Ticked without their handlers running: the request they would send is
        # sent here, once, with the new area already in place.
        for box in (self.measure_sharpness, self.measure_area):
            box.blockSignals(True)
            box.setChecked(True)
            box.blockSignals(False)
        self.requestSharpness.emit(True)
        self._apply_measure_area()
        self._show_sharpness(0.0, 0.0)

    def _on_measure_area_toggled(self, enabled: bool) -> None:
        QSettings().setValue("focus/sharpness_area", enabled)
        self._apply_measure_area()

    def _apply_measure_area(self) -> None:
        """Push the area to the worker and the overlay, or take it away."""
        wanted = (
            self._measure_area
            if self.measure_sharpness.isChecked() and self.measure_area.isChecked()
            else None
        )
        self.view.set_measure_area(wanted)
        self.requestSharpnessArea.emit(wanted)

    def _toggle_fine_tune(self) -> None:
        """Walk to focus from close by, or stop the hunt that is running."""
        if self._hunting:
            self.requestHuntCancel.emit()
            return
        self.requestFineTune.emit(self._focus_steps[_FINE_INCREMENT].value())

    @Slot(bool)
    def _on_hunt_changed(self, hunting: bool) -> None:
        self._hunting = hunting
        self.fine_tune_button.setText("Stop" if hunting else "Fine tune from here")

    def _on_sharpness_toggled(self, enabled: bool) -> None:
        QSettings().setValue("focus/sharpness", enabled)
        self._show_sharpness(0.0, 0.0)
        self.requestSharpness.emit(enabled)
        self._apply_measure_area()

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

        self.save_dir_label = WrappedLabel()
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
        self.requestIntegration.connect(self.worker.set_integration)
        self.requestSharpness.connect(self.worker.set_sharpness)
        self.requestSharpnessReset.connect(self.worker.reset_sharpness_peak)
        self.requestSharpnessArea.connect(self.worker.set_sharpness_area)
        self.requestFineTune.connect(self.worker.fine_tune)
        self.requestHuntCancel.connect(self.worker.cancel_hunt)
        self.requestDriveFocus.connect(self.worker.drive_focus)

        self.worker.connected.connect(self._on_connected)
        self.worker.disconnected.connect(self._on_disconnected)
        self.worker.frameReady.connect(self._on_frame)
        self.worker.settingsReady.connect(self._on_settings)
        self.worker.liveViewChanged.connect(self._on_live_view_changed)
        self.worker.zoomChanged.connect(self._on_zoom_changed)
        self.worker.focusStateChanged.connect(self.view.set_focus_state)
        self.worker.fpsChanged.connect(self._on_fps)
        self.worker.sharpnessChanged.connect(self._on_sharpness)
        self.worker.huntChanged.connect(self._on_hunt_changed)
        self.worker.exposurePreviewChanged.connect(self._on_exposure_preview)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.failed.connect(self._on_failed)

        # The worker sets up its own COM apartment before touching the camera.
        self._thread.started.connect(self.worker.initialise)
        self._thread.finished.connect(self.worker.deleteLater)

        self._thread.start()
        self.save_dir_label.setText(f"Saving to {self.worker.save_directory}")
        # Tell the worker what was restored from the last run; the requests are
        # queued, so they arrive once the worker's thread is up. The sharpness
        # ones cannot go out while the panel is built, since that happens
        # before there is a worker to hear them.
        self.requestIntegration.emit(
            self.integrate.isChecked(), self.integrate_frames.value()
        )
        self.requestSharpness.emit(self.measure_sharpness.isChecked())
        self._apply_measure_area()

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

    @Slot(object, object)
    def _on_frame(self, frame, image) -> None:
        """A frame from the camera, and the picture that goes with it.

        The picture is null while an integration stack is still filling, in
        which case only the overlay and the level readout move.
        """
        self.view.show_frame(frame, image)
        self._on_frame_level(frame)

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
        self._fps_shown = (fps, width, height)
        self._show_fps()

    def _show_fps(self) -> None:
        """The rate the picture actually updates at, and why it is that rate."""
        fps, width, height = self._fps_shown
        if fps <= 0:
            self.fps_label.setText("")
            return
        integrated = (
            f"  -  {self.integrate_frames.value()} frames integrated"
            if self.integrate.isChecked()
            else ""
        )
        self.fps_label.setText(f"{width}x{height}  -  {fps:.1f} fps{integrated}")

    @Slot(float, float)
    def _on_sharpness(self, value: float, peak: float) -> None:
        """Plot every reading; write the numbers out less often than that.

        A peak of zero is how the worker says it has started again -- the view
        moved, or the exposure did -- so the line starts again with it.
        """
        if peak <= 0.0:
            self.sharpness_trend.clear()
        else:
            self.sharpness_trend.add(value)
        now = time.monotonic()
        # Thirty readings a second is more than the eye can read as text; the
        # line has already had all of them.
        if value > 0.0 and now - self._sharpness_shown < 0.2:
            return
        self._sharpness_shown = now
        self._show_sharpness(value, peak)

    def _show_sharpness(self, value: float, peak: float) -> None:
        """The reading against the best of this view, or nothing when off."""
        measuring = self.measure_sharpness.isChecked()
        # Off, the readout is not just blank but gone: it is two rows of panel
        # that mean nothing until it is switched on.
        self.sharpness_trend.setVisible(measuring)
        self.sharpness_label.setVisible(measuring)
        self.reset_peak_button.setEnabled(measuring)
        self.measure_area.setEnabled(measuring)
        self.fine_tune_button.setEnabled(measuring)
        if not measuring or peak <= 0.0:
            self.sharpness_label.setText("Waiting for a frame...")
            return
        if value >= peak:
            self.sharpness_label.setText(f"{_reading(value)}   -   best so far")
            return
        self.sharpness_label.setText(
            f"{_reading(value)}   -   {100 * value / peak:.0f}% of best "
            f"{_reading(peak)}"
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
        # Ask first and wait, rather than quitting the thread here: the worker
        # has to put the camera's mirror back down and end live view before its
        # event loop stops, or the body is left streaming after we are gone.
        # The worker ends its own loop once it is done, so the wait returns as
        # soon as the camera is closed.
        self.requestShutdown.emit()
        if not self._thread.wait(5000):
            # A camera command that never came back. Nothing left to do but
            # stop waiting on it.
            self._thread.quit()
            if not self._thread.wait(1000):
                self._thread.terminate()
                self._thread.wait(1000)
        super().closeEvent(event)
