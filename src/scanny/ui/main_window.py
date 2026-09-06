"""The application window: live view on the left, camera controls on the right."""

from __future__ import annotations

import time

from PySide6.QtCore import (
    QEvent,
    QObject,
    QRect,
    QSettings,
    QSize,
    Qt,
    QThread,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
from .histogram import HistogramWidget
from .integration import DEFAULT_FRAMES, MAX_FRAMES, MIN_FRAMES, source_fps
from .liveview import LiveViewWidget
from .naming import DEFAULT_PREFIX, MAX_NUMBER, format_name
from .navigator import NavigatorWidget
from .trend import TrendGraph
from .worker import _FRAME_INTERVAL_MS, CameraWorker

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

#: The shape of a live-view picture: what a D750 sends in its photo position,
#: 640x424. The window opens this shape so the picture fills the image area
#: instead of sitting in it between two black strips -- strips are not merely
#: untidy, they are room the picture could have been shown in. A body that
#: sends another size, or the movie position, gets thin ones back.
_LIVE_VIEW_ASPECT = 640 / 424

#: How much of the screen a first-run window takes. Short of the whole of it,
#: so the window's own edges and the taskbar are still there to grab.
_SCREEN_SHARE = 0.9

#: How long a naming prefix may be. Long enough for any label worth typing,
#: short enough that the path stays well inside what Windows will open.
_MAX_PREFIX = 64


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


class _WheelGuard(QObject):
    """Sends a wheel notch to the panel instead of to the control under it.

    A combo box or a spin box reads a notch as "change my value", which is
    what it should mean when the pointer went there to use the control. It is
    not what it means while the panel is being scrolled past: the pointer is
    over the control only because the control is in the way, and a shutter
    speed that quietly steps as the panel goes by is a real change to a real
    camera, announced by nothing. Scrolling becomes a thing to be afraid of,
    which is a poor state to leave a panel that has to be scrolled.

    Qt has no setting for this. The remedy is to take the wheel away from
    every control in the panel that reads one and give it to the panel, which
    is the gesture the hand was making.
    """

    def __init__(self, scroller: QScrollArea) -> None:
        super().__init__(scroller)
        self._scroller = scroller

    def watch(self, widget: QWidget) -> None:
        widget.installEventFilter(self)

    def watch_all(self, within: QWidget) -> None:
        """Guard every control under *within* that answers the wheel."""
        for kind in (QComboBox, QAbstractSpinBox, QSlider):
            for control in within.findChildren(kind):
                self.watch(control)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        if event.type() != QEvent.Type.Wheel:
            return False
        bar = self._scroller.verticalScrollBar()
        # A notch is 120 eighths of a degree, and the system setting says how
        # many lines it is worth. The line is floored at twenty pixels: a
        # scroll area whose widget resizes with it can report a single step of
        # one, and a panel that moves three pixels a notch has not scrolled.
        notches = event.angleDelta().y() / 120
        lines = QApplication.wheelScrollLines() or 3
        bar.setValue(bar.value() - round(notches * max(bar.singleStep(), 20) * lines))
        return True


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
    requestMovePointInFrame = Signal(float, float)
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
    #: Discard reads that come back with the frame already on screen.
    requestDeduplicate = Signal(bool)
    requestSaveToCard = Signal(bool)
    requestShutterDelay = Signal(int)  # seconds, mirror up, before the shot
    requestPan = Signal(int, int)
    requestIntegration = Signal(bool, int)
    requestSharpness = Signal(bool)
    requestSharpnessReset = Signal()
    requestSharpnessArea = Signal(object)  # (x, y, w, h) fractions, or None
    requestFineTune = Signal(int)  # the one increment to walk in
    requestHuntCancel = Signal()
    #: Naming for downloaded pictures: on, the prefix, and the next number.
    #: Sent whole on every change, since an override may touch any of them.
    requestNaming = Signal(bool, str, int)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("scanny")

        #: One control per exposure setting: a combo box for a setting that is
        #: a choice, a spin box for one that is a number.
        self._setting_widgets: "dict[str, QWidget]" = {}
        self._level_shown = 0.0
        self._updating_settings = False
        self._live = False
        self._fps_shown = (0.0, 0, 0)
        # The camera's magnification, which decides how fast it draws.
        self._zoom_level = 0
        self._sharpness_shown = 0.0
        self._hunting = False
        self._measure_area = _DEFAULT_MEASURE_AREA
        # Set while the counter is being written into the box by the worker
        # rather than by the user, so it is not mistaken for an override.
        self._updating_naming = False

        self._build_ui()
        self._restore_geometry()
        self._start_worker()

    # -- geometry ----------------------------------------------------------

    def _restore_geometry(self) -> None:
        """Where the window was left last time, or a first size that fits."""
        stored = QSettings().value("window/geometry")
        if stored is not None and self.restoreGeometry(stored):
            return
        self.resize(self._size_that_fits_the_frame())

    def _size_that_fits_the_frame(self) -> QSize:
        """A window whose image area is the shape of a live-view frame.

        The picture is drawn as large as fits with its shape kept, so any
        mismatch between the window and the frame comes back as black strips
        down two sides of it. Opening at the frame's own shape starts those
        strips at nothing. The user is then free to resize into whatever shape
        suits them -- and that, rather than this, is what gets saved.
        """
        screen = self.screen() or QApplication.primaryScreen()
        room = screen.availableGeometry() if screen else QRect(0, 0, 1280, 800)
        beside, above = self._chrome()
        picture_h = room.height() * _SCREEN_SHARE - above
        widest = room.width() * _SCREEN_SHARE - beside
        if picture_h * _LIVE_VIEW_ASPECT > widest:
            picture_h = widest / _LIVE_VIEW_ASPECT
        # Two floors the window cannot open under: the height the pinned panes
        # give the right-hand column, and the width the image insists on. Both
        # are answered in the picture's height, so that whichever of them wins
        # the picture is still the shape of a frame -- meet one of them by
        # stretching the window alone and the strips are back.
        smallest = self.minimumSizeHint()
        picture_h = max(
            picture_h,
            smallest.height() - above,
            (smallest.width() - beside) / _LIVE_VIEW_ASPECT,
        )
        return QSize(
            round(picture_h * _LIVE_VIEW_ASPECT + beside), round(picture_h + above)
        )

    def _chrome(self) -> "tuple[int, int]":
        """How much of the window is not the picture: beside it, and above it."""
        layout = self.centralWidget().layout()
        margins = layout.contentsMargins()
        beside = (
            self._right_column.width()
            + layout.spacing()
            + margins.left()
            + margins.right()
        )
        above = (
            self.menuBar().sizeHint().height()
            + self.statusBar().sizeHint().height()
            + margins.top()
            + margins.bottom()
        )
        return beside, above

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
        layout.addWidget(self._build_right_column(), 0)
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

    def _build_right_column(self) -> QWidget:
        """The panes that stay put, above the controls that scroll.

        The navigator and the histogram are read *while* something else is
        being done -- focus driven, an aperture chosen -- so a readout that
        has to be scrolled back to is a readout that gets looked at once and
        then forgotten about. They keep their place at the top; everything
        below them is a control, and controls can be scrolled to.
        """
        self._right_column = column = QWidget()
        stack = QVBoxLayout(column)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(6)
        stack.addWidget(self._build_navigator_box())
        stack.addWidget(self._build_histogram_box())
        scroller = self._build_sidebar_scroller()
        # The whole column takes the scroller's width, so the pinned panes
        # line up with the controls under them whether the bar is showing or
        # not.
        column.setFixedWidth(scroller.width())
        stack.addWidget(scroller, 1)
        return column

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
        self._scroller = scroller = QScrollArea()
        sidebar = self._build_sidebar()
        scroller.setWidget(sidebar)
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
        # Nothing in here may answer the wheel itself; see _WheelGuard.
        self._wheels = _WheelGuard(scroller)
        self._wheels.watch_all(sidebar)
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
        # It is one of the two things that set how fast the camera draws, so
        # the integration hint has to be redrawn with it -- straight away,
        # rather than waiting for the camera to confirm.
        self.exposure_preview.toggled.connect(
            lambda _: self._describe_integration()
        )
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

        # The two levers on where the frames come from, under the control that
        # spends them. Both change the rate rather than the picture, which is
        # why they live with integration rather than with the exposure.
        self.deduplicate = QCheckBox("Skip repeated frames")
        self.deduplicate.setToolTip(
            "The camera is polled faster than it draws, so some reads come "
            "back with the frame already on screen. Averaging one of those in "
            "cancels no noise -- it carries the same noise, which adds instead "
            "of averaging down -- so they are dropped. Turn off to see the "
            "raw poll rate instead of the rate the camera actually draws at."
        )
        column.addWidget(self.deduplicate)

        settings = QSettings()
        self.integrate.setChecked(settings.value("liveview/integrate", False, bool))
        self.integrate_frames.setValue(self._stored_integration_frames())
        self.deduplicate.setChecked(
            bool(settings.value("liveview/deduplicate", True, bool))
        )
        self._integrating = self.integrate.isChecked()
        self._describe_integration()
        # Connected last, so restoring the stored values does not count as the
        # user asking for anything.
        self.integrate.toggled.connect(self._on_integration_changed)
        self.integrate_frames.valueChanged.connect(self._on_integration_changed)
        self.deduplicate.toggled.connect(self._on_deduplicate_changed)
        return holder

    def _on_deduplicate_changed(self, enabled: bool) -> None:
        QSettings().setValue("liveview/deduplicate", enabled)
        self._describe_integration()
        self._fps_shown = (0.0, *self._fps_shown[1:])
        self._show_fps()
        self.requestDeduplicate.emit(enabled)

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
        source = source_fps(self._zoom_level, self.exposure_preview.isChecked())
        # Without the check, a re-read counts as a frame: the stack fills at
        # the rate we poll at rather than the rate the camera draws at, and
        # arrives that much sooner with that much less of the noise gone.
        if not self.deduplicate.isChecked():
            polled = 1000.0 / _FRAME_INTERVAL_MS
            distinct = max(frames * source / polled, 1.0)
            self.integrate_hint.setText(
                f"About {polled / frames:.1f} fps, but only ~{distinct:.1f} of "
                f"each {frames} frames are redrawn, so roughly "
                f"{distinct ** 0.5:.1f}x less noise."
            )
            return
        self.integrate_hint.setText(
            f"About {source / frames:.1f} fps, with roughly "
            f"{frames ** 0.5:.1f}x less noise."
        )

    def _build_navigator_box(self) -> QGroupBox:
        """Where the magnified view sits in the frame, and a way to move it.

        Once the view is magnified there is nothing on the picture itself that
        says which part of the frame is on screen, and the arrow keys pan
        without ever saying how much further there is to go.

        What would be a paragraph of hint underneath is in the tooltip
        instead. The pane is pinned, so every line under it is a line the
        controls do not get, and the one thing worth explaining -- why the
        picture stops moving when the rectangle does not -- is not worth
        reading more than once.
        """
        box = QGroupBox("Navigator")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

        self.navigator = NavigatorWidget()
        self.navigator.setToolTip(
            "The whole frame, with the part on screen marked on it. Drag the "
            "rectangle to move the view, or press anywhere to send it there."
            "\n\n"
            "The picture is the last whole frame the camera sent: while you "
            "are magnified it cannot send the rest of the frame, so the "
            "picture waits and only the rectangle moves."
        )
        self.navigator.viewCentreMoved.connect(self._on_view_centre_moved)
        layout.addWidget(self.navigator)
        return box

    def _build_histogram_box(self) -> QGroupBox:
        """The levels in the picture on screen, under the navigator.

        The pair belong together: both are read off the picture rather than
        set, and both are read while a hand is busy somewhere else -- which is
        why they are pinned above the controls instead of scrolling with them.
        """
        box = QGroupBox("Histogram")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(2)

        self.histogram = HistogramWidget()
        self.histogram.setToolTip(
            "How many pixels sit at each level, per channel, for the picture "
            "on screen. The two end bins are left out of the height, so one "
            "clipped highlight cannot flatten the rest of the curve; what is "
            "clipped is written underneath instead."
        )
        layout.addWidget(self.histogram)

        self.histogram_label = QLabel(self.histogram.describe())
        self.histogram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.histogram_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.histogram_label)
        return box

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

        self.save_to_card = QCheckBox("Write to the camera's card")
        self.save_to_card.setToolTip(
            "Off, the camera holds the picture in its own memory and hands it "
            "straight to the computer -- no card needed, and nothing left on "
            "one to clear out later. It is the same full-size file either way. "
            "On, the shot is written to the card instead, and downloading it "
            "afterwards becomes optional."
        )
        self.save_to_card.setChecked(
            QSettings().value("capture/save_to_card", False, bool)
        )
        layout.addWidget(self.save_to_card)

        self.download_after_shot = QCheckBox("Download to computer")
        self.download_after_shot.setChecked(True)
        layout.addWidget(self.download_after_shot)
        self._apply_card_coupling()
        # Connected after the stored value is in place, so restoring it does
        # not read as the user asking for anything.
        self.save_to_card.toggled.connect(self._on_save_to_card_toggled)

        self.save_dir_label = WrappedLabel()
        self.save_dir_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.save_dir_label)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(4)
        choose = QPushButton("Change folder...")
        choose.clicked.connect(self._choose_save_directory)
        folder_row.addWidget(choose, 1)
        reveal = QPushButton("Open folder")
        reveal.setToolTip("Show the folder photos are saved to in Explorer.")
        reveal.clicked.connect(self._open_save_directory)
        folder_row.addWidget(reveal, 0)
        layout.addLayout(folder_row)

        layout.addWidget(self._build_naming())

        layout.addWidget(self._build_shutter_delay())

        self.shoot_button = QPushButton("Take photo")
        self.shoot_button.setMinimumHeight(40)
        self.shoot_button.clicked.connect(self._shoot)
        layout.addWidget(self.shoot_button)
        return box

    def _build_naming(self) -> QWidget:
        """What downloaded pictures are called: a prefix and the next number.

        Both are live, which is what makes them an override rather than a
        setting: type over either between two shots and the next shot uses
        what was typed, then carries on counting from there. Nothing has to be
        switched off and on again, and there is no earlier sequence hiding
        behind the override to come back.

        The number shown is always the one the *next* shot will get, so the
        same box is the readout: after each save the worker sends back where
        the counter has reached, and this follows it.
        """
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        self.rename_downloads = QCheckBox("Number the files myself")
        self.rename_downloads.setToolTip(
            "Save each picture as the prefix and the next number instead of "
            "under the camera's own name, so the folder comes out in the "
            "order the pages were shot."
        )
        column.addWidget(self.rename_downloads)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.name_prefix = QLineEdit()
        self.name_prefix.setMaxLength(_MAX_PREFIX)
        self.name_prefix.setPlaceholderText("prefix")
        self.name_prefix.setToolTip(
            "The part before the number, exactly as typed -- the separator is "
            "yours to include or leave out. The extension stays the camera's, "
            "since it is what says whether the file is a NEF or a JPEG."
        )
        row.addWidget(self.name_prefix, 1)

        self.name_number = QSpinBox()
        self.name_number.setRange(1, MAX_NUMBER)
        self.name_number.setToolTip(
            "The number the next picture gets. Change it whenever you like: "
            "counting carries on from whatever it is set to. A number a file "
            "in the folder is already using is skipped, never overwritten."
        )
        row.addWidget(self.name_number, 0)
        column.addLayout(row)

        self.name_preview = WrappedLabel()
        self.name_preview.setStyleSheet("color: #888; font-size: 11px;")
        column.addWidget(self.name_preview)

        settings = QSettings()
        self.rename_downloads.setChecked(settings.value("save/rename", False, bool))
        self.name_prefix.setText(str(settings.value("save/prefix", DEFAULT_PREFIX)))
        self.name_number.setValue(self._stored_number())
        self._apply_naming_enabled()
        self._show_next_name()

        # Connected once the stored values are in place, so restoring them
        # does not read as an override being typed.
        self.rename_downloads.toggled.connect(self._on_naming_changed)
        self.name_prefix.textChanged.connect(self._on_naming_changed)
        self.name_number.valueChanged.connect(self._on_naming_changed)
        return holder

    def _stored_number(self) -> int:
        """Where the counter had reached when the program was last closed.

        Remembered so that a scan spread over two sittings is one sequence:
        the numbering the second session starts on is where the first one
        stopped, and it is still the user's to type over.
        """
        try:
            stored = int(QSettings().value("save/number", 1))
        except (TypeError, ValueError):
            return 1
        return stored if 1 <= stored <= MAX_NUMBER else 1

    def _apply_naming_enabled(self) -> None:
        on = self.rename_downloads.isChecked()
        self.name_prefix.setEnabled(on)
        self.name_number.setEnabled(on)

    def _show_next_name(self) -> None:
        if self.rename_downloads.isChecked():
            name = format_name(self.name_prefix.text(), self.name_number.value())
            self.name_preview.setText(f"Next: {name} + the camera's extension")
        else:
            self.name_preview.setText("Keeping the names the camera gives.")

    def _on_naming_changed(self) -> None:
        """Hand an override to the worker, and remember it for the next run."""
        self._apply_naming_enabled()
        self._show_next_name()
        if self._updating_naming:
            return
        settings = QSettings()
        settings.setValue("save/rename", self.rename_downloads.isChecked())
        settings.setValue("save/prefix", self.name_prefix.text())
        settings.setValue("save/number", self.name_number.value())
        self._request_naming()

    def _request_naming(self) -> None:
        self.requestNaming.emit(
            self.rename_downloads.isChecked(),
            self.name_prefix.text(),
            self.name_number.value(),
        )

    @Slot(int)
    def _on_next_number(self, number: int) -> None:
        """Follow the counter once a shot has taken a number from it.

        Guarded, because putting the value in the box is not the user typing
        an override: sending it back would only hand the worker what it just
        reported, and would stamp on a number typed while the picture was
        still downloading.
        """
        self._updating_naming = True
        try:
            self.name_number.setValue(number)
        finally:
            self._updating_naming = False
        QSettings().setValue("save/number", self.name_number.value())

    def _build_shutter_delay(self) -> QWidget:
        """The wait between the mirror lifting and the shutter firing.

        Nikon's own exposure delay mode, driven from here for the shot and
        handed back to the camera afterwards. What it is worth is spelled out
        underneath: on a copy stand the mirror is the largest thing that moves,
        and waiting for it to stop moving is free sharpness.
        """
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Mirror-up delay"))
        self.shutter_delay = QComboBox()
        self.shutter_delay.setToolTip(
            "Lift the mirror, wait, and only then release the shutter, so the "
            "vibration it made has died away before the exposure starts. This "
            "is the camera's own exposure delay mode, set for each shot taken "
            "from here and put back to whatever the camera had afterwards."
        )
        row.addWidget(self.shutter_delay, 1)
        column.addLayout(row)

        self.shutter_delay_hint = WrappedLabel()
        self.shutter_delay_hint.setStyleSheet("color: #888; font-size: 11px;")
        column.addWidget(self.shutter_delay_hint)

        # What a D750 offers, until a camera says otherwise: the panel is built
        # before anything is connected, and the control cannot be blank.
        self._offer_shutter_delays(NikonCamera.SHUTTER_DELAYS, None)
        self.shutter_delay.currentIndexChanged.connect(self._on_shutter_delay_changed)
        return holder

    def _offer_shutter_delays(self, delays, on_body) -> None:
        """Offer exactly the delays the body takes, keeping the chosen one.

        A body with no such setting sends nothing, and the control is disabled
        rather than left offering a delay that would never happen.

        Where the choice starts, on a camera never driven from here, is
        whatever that camera is set to: a rig already set up with a delay on
        the body should not lose it just because something else is now
        releasing the shutter. Once a delay has been chosen here, that is what
        is remembered and used.
        """
        wanted = self._stored_shutter_delay()
        if wanted is None:
            wanted = 0 if on_body is None else int(on_body)
        self.shutter_delay.blockSignals(True)
        self.shutter_delay.clear()
        for seconds in delays:
            self.shutter_delay.addItem("Off" if not seconds else f"{seconds} s", seconds)
        index = self.shutter_delay.findData(wanted)
        self.shutter_delay.setCurrentIndex(max(index, 0))
        self.shutter_delay.blockSignals(False)
        self.shutter_delay.setEnabled(bool(delays))
        # Whatever survived that -- the delay chosen here, or what the camera
        # was already set to -- is what the worker is now told to use. Not
        # stored, though: what is remembered between runs is what was chosen
        # here, and a body that has no three seconds to offer today should not
        # erase the three seconds that were asked for.
        self._describe_shutter_delay()
        self.requestShutterDelay.emit(self._chosen_shutter_delay())

    def _stored_shutter_delay(self) -> "int | None":
        """The delay chosen here last time, or None if none ever was."""
        stored = QSettings().value("capture/shutter_delay", None)
        try:
            seconds = int(stored)
        except (TypeError, ValueError):
            return None
        return seconds if seconds >= 0 else None

    def _chosen_shutter_delay(self) -> int:
        seconds = self.shutter_delay.currentData()
        return 0 if seconds is None else int(seconds)

    def _on_shutter_delay_changed(self) -> None:
        seconds = self._chosen_shutter_delay()
        QSettings().setValue("capture/shutter_delay", seconds)
        self._describe_shutter_delay()
        self.requestShutterDelay.emit(seconds)

    def _describe_shutter_delay(self) -> None:
        seconds = self._chosen_shutter_delay()
        self.shutter_delay_hint.setText(
            f"Each shot takes {seconds}s longer, and nothing moves in the "
            "picture while it waits."
            if seconds
            else "The shutter fires as soon as it is asked to."
            if self.shutter_delay.isEnabled()
            else "This camera has no exposure delay mode."
        )

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
        self.requestMovePointInFrame.connect(self.worker.move_point_in_frame)
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
        self.requestSaveToCard.connect(self.worker.set_save_to_card)
        self.requestShutterDelay.connect(self.worker.set_shutter_delay)
        self.requestPan.connect(self.worker.pan)
        self.requestIntegration.connect(self.worker.set_integration)
        self.requestDeduplicate.connect(self.worker.set_deduplicate)
        self.requestSharpness.connect(self.worker.set_sharpness)
        self.requestSharpnessReset.connect(self.worker.reset_sharpness_peak)
        self.requestSharpnessArea.connect(self.worker.set_sharpness_area)
        self.requestFineTune.connect(self.worker.fine_tune)
        self.requestHuntCancel.connect(self.worker.cancel_hunt)
        self.requestDriveFocus.connect(self.worker.drive_focus)
        self.requestNaming.connect(self.worker.set_naming)

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
        self.worker.saveToCardChanged.connect(self._on_save_to_card)
        self.worker.shutterDelaysAvailable.connect(self._on_shutter_delays)
        self.worker.nextNumber.connect(self._on_next_number)
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
        self.requestDeduplicate.emit(self.deduplicate.isChecked())
        self.requestSharpness.emit(self.measure_sharpness.isChecked())
        self._apply_measure_area()
        self.requestSaveToCard.emit(self.save_to_card.isChecked())
        self.requestShutterDelay.emit(self._chosen_shutter_delay())
        self._request_naming()

    # -- slots -------------------------------------------------------------

    @Slot(str)
    def _on_connected(self, description: str) -> None:
        self.camera_label.setText(description)
        # Weight only, no colour: the window follows the system theme, and a
        # hardcoded light grey vanishes against a light background.
        self.camera_label.setStyleSheet("font-weight: 600;")
        self._clear_view("Press Start live view")

    @Slot()
    def _on_disconnected(self) -> None:
        self.camera_label.setText("No camera connected")
        self.camera_label.setStyleSheet("color: #888;")
        self._clear_view("Not connected")

    @Slot(bool)
    def _on_live_view_changed(self, active: bool) -> None:
        self._live = active
        self.live_button.setText("Stop live view" if active else "Start live view")
        if active:
            self.view.setFocus()
        if not active:
            self._clear_view("Live view stopped")
            self.view.set_focus_state("idle")

    def _clear_view(self, message: str) -> None:
        """Take the picture away, and everything read off it with it.

        A histogram or a navigator map left behind from the last session is
        worse than an empty one: both look live, and neither is.
        """
        self.view.clear(message)
        self.navigator.clear(message)
        self.histogram.clear()
        self.histogram_label.setText(self.histogram.describe())

    @Slot(int)
    def _on_zoom_changed(self, level: int) -> None:
        # Kept because the rate the camera draws at depends on it: past 4.7x
        # the large frame falls to a third of its speed.
        self._zoom_level = level
        self._describe_integration()
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
        self.navigator.show_frame(frame, image)
        self._on_frame_histogram(image)
        self._on_frame_level(frame)

    def _on_frame_histogram(self, image) -> None:
        """Count the levels in the picture, as often as the widget wants to.

        It throttles itself, so the caption is only rewritten when there is
        something new behind it.
        """
        if self.histogram.set_image(image):
            self.histogram_label.setText(self.histogram.describe())

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
        # Forty-odd readings a second is more than the eye can read as text;
        # the line has already had all of them.
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

    def _on_save_to_card_toggled(self, enabled: bool) -> None:
        QSettings().setValue("capture/save_to_card", enabled)
        self._apply_card_coupling()
        self.requestSaveToCard.emit(enabled)

    @Slot(bool)
    def _on_save_to_card(self, enabled: bool) -> None:
        """Follow the camera, which may not have allowed what was asked of it."""
        self.save_to_card.blockSignals(True)
        self.save_to_card.setChecked(enabled)
        self.save_to_card.blockSignals(False)
        self._apply_card_coupling()

    def _apply_card_coupling(self) -> None:
        """With the card off, the picture only exists once it is downloaded."""
        to_card = self.save_to_card.isChecked()
        if not to_card:
            self.download_after_shot.setChecked(True)
        self.download_after_shot.setEnabled(to_card)
        self.download_after_shot.setToolTip(
            ""
            if to_card
            else "The camera is holding the picture in memory rather than "
            "writing it to a card, and the next shot replaces it -- so it has "
            "to come down to the computer."
        )

    @Slot(object, object)
    def _on_shutter_delays(self, delays, on_body) -> None:
        """The connected body's delays, which are not every body's."""
        self._offer_shutter_delays(tuple(delays), on_body)

    @Slot(bool)
    def _on_exposure_preview(self, enabled: bool) -> None:
        self.exposure_preview.blockSignals(True)
        self.exposure_preview.setChecked(enabled)
        self.exposure_preview.blockSignals(False)
        # It is one of the three things that decide how fast the camera draws.
        self._describe_integration()

    @Slot(object)
    def _on_settings(self, settings: "list[Setting]") -> None:
        self._updating_settings = True
        try:
            self._rebuild_exposure_form(settings)
        finally:
            self._updating_settings = False

    def _rebuild_exposure_form(self, settings: "list[Setting]") -> None:
        existing = {s.name for s in settings}
        if set(self._setting_widgets) != existing:
            while self.exposure_form.rowCount():
                self.exposure_form.removeRow(0)
            self._setting_widgets.clear()
            for setting in settings:
                widget = self._setting_widget(setting)
                self._wheels.watch(widget)
                self._setting_widgets[setting.name] = widget
                self.exposure_form.addRow(setting.name, widget)

        for setting in settings:
            widget = self._setting_widgets[setting.name]
            if isinstance(widget, QSpinBox):
                self._show_setting_number(widget, setting)
            elif isinstance(widget, QComboBox):
                self._show_setting_choices(widget, setting)
            widget.setEnabled(
                setting.writable and (setting.span is not None or bool(setting.choices))
            )
            tip = setting.note or (
                setting.name if setting.writable else "Set on the camera body"
            )
            widget.setToolTip(tip)
            # A disabled control is past the reach of the mouse, and a tooltip
            # it cannot be given is no way to say why it is disabled. Its row
            # label is still live, so the reason goes there as well.
            label = self.exposure_form.labelForField(widget)
            if label is not None:
                label.setToolTip(tip)

    def _setting_widget(self, setting: "Setting") -> QWidget:
        """A box to type the value into, or a list to pick it from."""
        if setting.span is None:
            combo = QComboBox()
            combo.activated.connect(
                lambda _index, name=setting.name: self._on_setting_chosen(name)
            )
            return combo
        spin = QSpinBox()
        spin.setSuffix(setting.span.unit)
        # Off, so that typing 5000 does not send 5, then 50, then 500 to the
        # camera on the way: the value goes when the number is finished.
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(
            lambda _value, name=setting.name: self._on_setting_chosen(name)
        )
        return spin

    def _show_setting_choices(self, combo: QComboBox, setting: "Setting") -> None:
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

    def _show_setting_number(self, spin: QSpinBox, setting: "Setting") -> None:
        span = setting.span
        try:
            value = int(setting.value)
        except (TypeError, ValueError):
            return
        spin.blockSignals(True)
        if span is not None:
            spin.setRange(span.minimum, span.maximum)
            spin.setSingleStep(span.step)
        if spin.value() != value:
            spin.setValue(value)
        spin.blockSignals(False)

    def _on_setting_chosen(self, name: str) -> None:
        if self._updating_settings:
            return
        widget = self._setting_widgets.get(name)
        if widget is None:
            return
        if isinstance(widget, QSpinBox):
            self.requestSetting.emit(name, widget.value())
            return
        if not isinstance(widget, QComboBox):
            return
        value = widget.currentData()
        if value is not None:
            self.requestSetting.emit(name, value)
        # The combo needed the keyboard for its popup; give it back to the
        # image so the shortcuts keep working. A spin box keeps it: the number
        # in it is likely still being adjusted.
        self.view.setFocus()

    @Slot(float, float)
    def _on_point_selected(self, nx: float, ny: float) -> None:
        if self._live:
            self.requestMovePoint.emit(nx, ny)

    @Slot(float, float)
    def _on_view_centre_moved(self, fx: float, fy: float) -> None:
        if self._live:
            self.requestMovePointInFrame.emit(fx, fy)

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

    def _open_save_directory(self) -> None:
        # The folder is only made when the first picture lands in it, so it may
        # not be there yet. Making it here is what the next save would have done
        # anyway, and it beats an "open" that quietly does nothing.
        directory = self.worker.save_directory
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.statusBar().showMessage(f"Cannot open {directory}: {exc}", 8000)
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory))):
            self.statusBar().showMessage(f"Cannot open {directory}", 8000)

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Ask first and wait, rather than quitting the thread here: the worker
        # has to put the camera's mirror back down and end live view before its
        # event loop stops, or the body is left streaming after we are gone.
        # The worker ends its own loop once it is done, so the wait returns as
        # soon as the camera is closed.
        # Saved first: a camera that hangs on the way out must not cost the
        # user the window they had arranged.
        QSettings().setValue("window/geometry", self.saveGeometry())
        self.requestShutdown.emit()
        if not self._thread.wait(5000):
            # A camera command that never came back. Nothing left to do but
            # stop waiting on it.
            self._thread.quit()
            if not self._thread.wait(1000):
                self._thread.terminate()
                self._thread.wait(1000)
        super().closeEvent(event)
