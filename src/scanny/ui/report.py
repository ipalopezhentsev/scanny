"""What a calibration found, laid out for a person to judge.

A percentage says how much sharpness a region gave up for the compromise; it
does not say whether what is left is good enough, and only looking can. So
each region is shown twice, side by side at the same size: as it was at its
own best, fine tuned on alone, and as it is at the one focus position chosen
for all of them. Two more pages hold what the rest of the calibration found:
the shape of the film the regions' depths make, drawn over the sensor, with
what levelling it needs (:mod:`scanny.ui.film`); and every region's
sharpness through the search, which is what makes the compromise it arrived
at make sense (:mod:`scanny.ui.regionchart`).

The pictures are shown the way the view is being shown -- turned, mirrored
and inverted as the View panel has it now, not as it was while calibrating --
because they are to be compared with the picture on screen and with each
other, and a negative judged as a negative is guesswork. They are scaled up
by whole numbers only, pixel for pixel: a smoothed enlargement would blur
the very thing the report exists to show.

The last page is what was said while the calibration ran, from the activity
log. The whole report, that included, can be saved as a document and opened
again later (:mod:`scanny.ui.reportfile`).
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .film import FilmView, Mark
from .orientation import Orientation
from .regionchart import RegionChart
from .regions import CalibrationReport, Look, RegionResult, describe_depth
from .reportfile import REPORT_FILTER, SUFFIX, ReportFileError, save_report
from .sharpness import format_reading

__all__ = ["CalibrationReportDialog", "enlarged"]

#: How big a picture of a region is shown, at most, on its longer side.
_SHOWN = 260

#: How wide the column saying which region a row is, and the whole grid.
_NAMES = 170
_SPACING = 12
_WHOLE = _NAMES + 2 * _SHOWN + 2 * _SPACING

#: How wide the notes beside the film's shape are.
_NOTES = 300

#: Said under the depths every time, because it is the one thing about them
#: that nothing here can find out, and levelling the wrong way doubles a tilt.
_WHICH_WAY = (
    "Further and nearer are the focus drive's own names for its two "
    "directions, and which of them a lens actually turns could not be "
    "confirmed from live view. Before levelling by these numbers, drive focus "
    "a few steps further with > and check that the edge said to be further "
    "is the one that sharpens."
)

#: What the fine tune on a region said about how it ended, in a few words.
_TUNED = {
    "found": "found by fine tuning it on its own",
    "restored": "the camera's own autofocus; fine tuning did not better it",
    "exhausted": "fine tuning ran out of probes here",
    "lost": "the reading was too unsteady to walk by",
    "nothing": "nothing in it to focus on",
}


def enlarged(picture: QImage, orientation: Orientation, longest: int = _SHOWN) -> QImage:
    """*picture* the way the view is shown, scaled to fit *longest* pixels.

    Up by a whole number of times, pixel for pixel, or down smoothly when it
    is bigger than that already -- a region drawn round half the frame at
    full frame is hundreds of pixels across.
    """
    shown = orientation.apply(picture)
    side = max(shown.width(), shown.height(), 1)
    if side > longest:
        return shown.scaled(
            longest,
            longest,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    times = max(1, longest // side)
    return shown.scaled(
        shown.width() * times,
        shown.height() * times,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )


def _wrapped(
    text: str,
    width: int,
    style: str = "",
    align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter,
) -> QLabel:
    """A word-wrapped label exactly as tall as its text is at *width*.

    Left to itself, a wrapped label in a grid is given the height its text
    needs at some other width than the one it gets, and the lines that do
    not fit are cut off top and bottom -- the same thing
    :class:`scanny.ui.main_window.WrappedLabel` is there to stop in the
    panel. Here the width is fixed, so the height can simply be worked out
    once and fixed too.
    """
    label = QLabel(text)
    label.setWordWrap(True)
    label.setAlignment(align)
    if style:
        label.setStyleSheet(style)
    # The style sheet can change the font, and the height is the font's.
    label.ensurePolished()
    label.setFixedWidth(width)
    label.setFixedHeight(label.heightForWidth(width))
    return label


class CalibrationReportDialog(QDialog):
    """What a calibration found, on four pages.

    **Regions**: every region at its best and at the compromise, side by side,
    with the numbers. **Film**: the shape the regions' depths make, in three
    dimensions over the sensor, with the lean of its edges and how far it bows
    between them -- what levelling needs. **Search**: every region's sharpness
    through the search, one line each, which is what makes the compromise it
    arrived at make sense. **Activity log**: what was said while it ran.

    *source* is the file the report was opened from, if it was.
    """

    def __init__(
        self,
        report: CalibrationReport,
        orientation: Orientation,
        save_to: Path,
        parent: "QWidget | None" = None,
        aspect: float = 1.5,
        source: "Path | None" = None,
    ) -> None:
        super().__init__(parent)
        self._source = Path(source) if source is not None else None
        self.setWindowTitle(
            f"Focus calibration - {self._source.name}"
            if self._source is not None
            else "Focus calibration"
        )
        self._report = report
        self._orientation = orientation
        self._save_to = save_to
        self._aspect = aspect
        #: The picture labels, best and compromise for each region in turn.
        self._pictures: "list[QLabel]" = []

        layout = QVBoxLayout(self)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("font-weight: 600;")
        layout.addWidget(self._summary)
        self._cost = QLabel()
        self._cost.setWordWrap(True)
        self._cost.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._cost)

        self._tabs = QTabWidget()
        self._scroller = QScrollArea()
        self._scroller.setWidgetResizable(True)
        self._tabs.addTab(self._scroller, "Regions")

        self._film_page = QWidget()
        film = QHBoxLayout(self._film_page)
        self._film_view = FilmView()
        film.addWidget(self._film_view, 1)
        self._film_notes = QScrollArea()
        self._film_notes.setWidgetResizable(True)
        self._film_notes.setFixedWidth(_NOTES + 24)
        film.addWidget(self._film_notes)
        self._tabs.addTab(self._film_page, "Film shape")

        self._search_page = QWidget()
        search = QVBoxLayout(self._search_page)
        self._chart = RegionChart(compact=False)
        search.addWidget(self._chart)
        search.addWidget(_wrapped(
            "Each region's sharpness as a share of its own best, at every probe "
            "of the search for the compromise -- every region read at one focus "
            "position -- with the number the search climbs drawn dashed over "
            "them. Shaded behind: walking out past every region's best, walking "
            "back across all of them (the walk the depths are read from), and "
            "going home to the best of it.",
            _WHOLE,
            "color: #888; font-size: 11px;",
            align=Qt.AlignmentFlag.AlignLeft,
        ))
        search.addStretch(1)
        self._tabs.addTab(self._search_page, "The search")

        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._log_text.setFont(font)
        self._tabs.addTab(self._log_text, "Activity log")
        layout.addWidget(self._tabs, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        keep = QPushButton("Save report...")
        keep.setToolTip(
            "Write the whole report out as one file -- every region's pictures "
            "and numbers, the film's shape, the search and the activity log -- "
            "to open again later with File > Open focus report."
        )
        keep.clicked.connect(self._save_report)
        buttons.addWidget(keep)
        save = QPushButton("Save page as picture...")
        save.setToolTip(
            "Write the page on show out as a PNG, the way it looks here, to keep "
            "alongside the scans it was made for."
        )
        save.clicked.connect(self._save)
        buttons.addWidget(save)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self._fill()
        self._fit_to_content()

    # -- what it shows -----------------------------------------------------

    def set_report(self, report: CalibrationReport) -> None:
        self._report = report
        self._fill()
        self._fit_to_content()

    def _fit_to_content(self) -> None:
        """Wide enough for both columns of pictures, and no taller than fits.

        Wide enough that nothing has to be scrolled sideways to compare a
        region at its best with the same region at the compromise, which is
        the one comparison the report is for. Tall enough for every region
        where the screen allows, and scrolling where it does not.
        """
        wanted = self._content.sizeHint()
        bar = self._scroller.verticalScrollBar().sizeHint().width()
        margins = self.layout().contentsMargins()
        width = max(wanted.width() + bar, 480 + _NOTES) + margins.left() + margins.right() + 16
        height = (
            max(wanted.height(), 460)
            + self._summary.sizeHint().height()
            + self._cost.sizeHint().height()
            + 120
            + margins.top()
            + margins.bottom()
        )
        screen = self.screen()
        if screen is not None:
            room = screen.availableGeometry()
            width = min(width, int(room.width() * 0.9))
            height = min(height, int(room.height() * 0.9))
        self.resize(width, height)

    def set_orientation(self, orientation: Orientation) -> None:
        """Show the pictures, and the film, the way the view is now being shown."""
        if orientation == self._orientation:
            return
        self._orientation = orientation
        self._fill()

    def _fill(self) -> None:
        report = self._report
        self._summary.setText(report.describe())
        cost = report.cost()
        if report.began > 0.0:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(report.began))
            cost = f"Calibrated {when}" + (f". {cost}" if cost else "")
        self._cost.setText(cost)
        self._cost.setVisible(bool(cost))
        self._fill_regions()
        self._fill_film()
        self._chart.set_combined_name(report.objective)
        self._chart.set_history(report.history_regions, report.history)
        self._log_text.setPlainText("\n".join(report.log))
        self._log_text.setPlaceholderText("Nothing was logged with this report.")

    def _fill_regions(self) -> None:
        report = self._report
        self._pictures = []
        content = QWidget()
        grid = QGridLayout(content)
        grid.setHorizontalSpacing(_SPACING)
        grid.setVerticalSpacing(6)
        for column, heading in enumerate(
            ("", "At its best, fine tuned on alone", "At the compromise"), start=0
        ):
            if heading:
                # No wider than the pictures under it, or it sets the width
                # of the column and pushes the other one off the side.
                title = _wrapped(heading, _SHOWN, "color: #888;")
                grid.addWidget(title, 0, column, Qt.AlignmentFlag.AlignHCenter)
        for row, result in enumerate(report.results, start=1):
            grid.addWidget(self._describe(result), row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(self._panel(result.best, "best"), row, 1, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(
                self._panel(result.compromise, "compromise", result.fraction),
                row,
                2,
                Qt.AlignmentFlag.AlignTop,
            )
        grid.setRowStretch(len(report.results) + 1, 1)
        self._content = content
        self._scroller.setWidget(content)

    def _fill_film(self) -> None:
        """The film's shape, and what levelling needs to know about it.

        The lean is worked out in the picture's directions as it is shown --
        turned and mirrored the way the View panel has it -- because that is
        how the film is being looked at while it is adjusted; it is worked out
        again, and the drawing turned, whenever the view is.
        """
        report = self._report
        self._film_view.set_scene(
            report.surface(),
            [
                Mark(one.number, one.region.rect, one.depth)
                for one in report.placed
                if not one.edge
            ],
            self._orientation,
            self._aspect,
        )
        notes = QWidget()
        column = QVBoxLayout(notes)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        placed = report.placed
        if not placed:
            said = (
                "No region's depth was measured, so there is no shape to draw."
                if report.depths_measured
                else "Depths were not asked for on this calibration. Tick "
                "<b>Measure depths for levelling</b> and calibrate again to see "
                "how far apart in focus the regions are, how the film's edges "
                "lean and how far it bows -- it walks further to find out."
            )
            column.addWidget(_wrapped(said, _NOTES, align=Qt.AlignmentFlag.AlignLeft))
            column.addStretch(1)
            self._film_notes.setWidget(notes)
            return
        lines = ["<b>Depth</b>, in drive steps, on one walk across all of them:"]
        for one in placed:
            if one.depth < 0.5:
                lines.append(f"Region {one.number}: the nearest")
                continue
            bound = "at least " if one.edge else ""
            plus = f" (± {one.doubt:.0f})" if one.doubt < float("inf") else ""
            lines.append(
                f"Region {one.number}: {bound}{one.depth:.0f} steps further{plus}"
            )
        for one in report.results:
            if one.usable and one.depth is None:
                lines.append(
                    f"Region {one.number}: its peak was not on the walk, so not measured"
                )
        column.addWidget(
            _wrapped("<br>".join(lines), _NOTES, align=Qt.AlignmentFlag.AlignLeft)
        )
        tilt = report.tilt(self._orientation)
        if tilt is not None:
            said = (
                "<b>The film's edges</b>, where the holder grips it, as the "
                "picture is shown:<br>" + tilt.describe().replace("\n", "<br>")
            )
        else:
            said = (
                "A lean needs three regions, not in a line, with their depths "
                "measured; with fewer, only the differences above can be said."
            )
        column.addWidget(_wrapped(said, _NOTES, align=Qt.AlignmentFlag.AlignLeft))
        column.addWidget(_wrapped(
            "The shape is a plane for the edges and the gentlest bulge between "
            "them that passes through every region, the edges being taken as "
            "the edges of the picture. " + _WHICH_WAY,
            _NOTES,
            "color: #888; font-size: 11px;",
            align=Qt.AlignmentFlag.AlignLeft,
        ))
        column.addStretch(1)
        self._film_notes.setWidget(notes)

    def _describe(self, result: RegionResult) -> QLabel:
        lines = [f"<b>Region {result.number}</b>"]
        if result.best is None:
            lines.append("not reached")
        else:
            lines.append(_TUNED.get(result.outcome, result.outcome or ""))
        if result.fraction is not None:
            lines.append(f"<b>{result.fraction:.0%}</b> of its best")
        if result.depth is not None:
            lines.append(describe_depth(result).removeprefix("Depth: "))
        return _wrapped("<br>".join(lines), _NAMES, align=Qt.AlignmentFlag.AlignLeft)

    def _panel(
        self, look: "Look | None", kind: str, fraction: "float | None" = None
    ) -> QWidget:
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        picture = QLabel()
        picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        picture.setStyleSheet("background: #18181b; color: #888;")
        if look is not None and look.picture is not None and not look.picture.isNull():
            shown = QPixmap.fromImage(enlarged(look.picture, self._orientation))
            picture.setPixmap(shown)
            # Exactly the picture's size: a label left to stretch pads it out
            # with background, which reads as part of the picture.
            picture.setFixedSize(shown.size())
        else:
            picture.setText("no picture" if look is not None else "--")
            picture.setFixedSize(_SHOWN // 2, _SHOWN // 3)
        self._pictures.append(picture)
        column.addWidget(picture, 0, Qt.AlignmentFlag.AlignHCenter)
        caption = _wrapped(
            self._caption(look, kind, fraction),
            _SHOWN,
            "color: #888; font-size: 11px;",
        )
        column.addWidget(caption, 0, Qt.AlignmentFlag.AlignHCenter)
        return holder

    @staticmethod
    def _caption(look: "Look | None", kind: str, fraction: "float | None") -> str:
        if look is None:
            return "no compromise found" if kind == "compromise" else "not tuned"
        reading = format_reading(look.reading)
        if kind == "compromise" and fraction is not None:
            return f"sharpness {reading}, {fraction:.0%} of its best"
        return f"sharpness {reading}"

    # -- keeping it --------------------------------------------------------

    def _page_to_save(self) -> "tuple[QWidget, str]":
        """The page on show, as a widget to draw, and a word for its file."""
        page = self._tabs.currentWidget()
        if page is self._scroller:
            # The whole of the content, not the part the scroll area shows.
            return self._content, "regions"
        if page is self._film_page:
            return self._film_page, "film"
        if page is self._log_text:
            return self._log_text, "log"
        return self._search_page, "search"

    def _stem(self) -> str:
        """What a file saved from this report is called, by when it was made."""
        when = self._report.began if self._report.began > 0.0 else time.time()
        return f"focus-report-{time.strftime('%Y%m%d-%H%M%S', time.localtime(when))}"

    def _folder(self) -> Path:
        """Where to offer to save: beside the file it came from, or the pictures."""
        if self._source is not None:
            return self._source.parent
        try:
            self._save_to.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return self._save_to

    def _save(self) -> None:
        widget, word = self._page_to_save()
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save the page as a picture",
            str(self._folder() / f"{self._stem()}-{word}.png"),
            "PNG (*.png)",
        )
        if not chosen:
            return
        picture = QWidget.grab(widget)
        if not picture.save(chosen, "PNG"):
            self._summary.setText(f"Could not write {chosen}")

    def _save_report(self) -> None:
        """The whole report, as a document that can be opened again."""
        suggested = (
            self._source
            if self._source is not None
            else self._folder() / f"{self._stem()}{SUFFIX}"
        )
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save the report", str(suggested), REPORT_FILTER
        )
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix.lower() != SUFFIX:
            path = path.with_name(path.name + SUFFIX)
        try:
            save_report(path, self._report, self._aspect)
        except ReportFileError as exc:
            self._summary.setText(str(exc))
