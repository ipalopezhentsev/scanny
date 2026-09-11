"""A running record of what the program did, to be read after the fact.

The status bar says one thing at a time and forgets it the moment the next
thing is said, which is right for watching and useless for finding out, a
quarter of an hour into a calibration, what it was doing when it stopped. So
everything the worker says -- what it is doing, what went wrong in the
camera's own words, every probe of a calibration and every attempt at getting
live view back -- is written here as well, with the time, and kept.

Kept twice: in memory for the window (**View > Activity log**), and in a file
beside the pictures, so that it survives the window being closed or the
program falling over, which is when it is wanted most. The file is started
afresh once it grows large, the last one kept beside it.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["ActivityLog", "ActivityWindow"]

#: How many lines the window keeps. A calibration writes a line a probe, a
#: few hundred at most, so this is many calibrations' worth.
_KEPT = 5000

#: How big the file may grow before it is started afresh, the old one kept.
_FILE_LIMIT = 2_000_000


class ActivityLog(QObject):
    """The lines, with their times, and the file they are written to."""

    #: A line has been added, as it will be shown.
    added = Signal(str)

    def __init__(self, parent: "QObject | None" = None) -> None:
        super().__init__(parent)
        self._lines: "deque[str]" = deque(maxlen=_KEPT)
        self._file: "Path | None" = None

    @property
    def lines(self) -> "list[str]":
        return list(self._lines)

    @property
    def file(self) -> "Path | None":
        return self._file

    def write_to(self, path: "Path | None") -> None:
        """Also write every line to *path* from now on, or to nowhere."""
        self._file = Path(path) if path is not None else None
        if self._file is not None:
            try:
                self._file.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._file = None

    def add(self, text: str, kind: str = "") -> None:
        """One line: the time, what kind of thing it is, and what was said."""
        text = str(text).strip()
        if not text:
            return
        mark = {"error": "!! ", "status": ""}.get(kind, "   ")
        line = f"{time.strftime('%H:%M:%S')}  {mark}{text}"
        self._lines.append(line)
        self._append_to_file(line)
        self.added.emit(line)

    def _append_to_file(self, line: str) -> None:
        if self._file is None:
            return
        try:
            if self._file.exists() and self._file.stat().st_size > _FILE_LIMIT:
                self._file.replace(self._file.with_suffix(".old.log"))
            with self._file.open("a", encoding="utf-8") as sink:
                sink.write(f"{time.strftime('%Y-%m-%d')} {line}\n")
        except OSError:
            # A log that cannot be written is not worth stopping for; the
            # window still has it.
            pass


class ActivityWindow(QDialog):
    """The log, as it grows: newest at the bottom, followed as it arrives."""

    def __init__(self, log: ActivityLog, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Activity log")
        self._log = log
        layout = QVBoxLayout(self)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._text.setMaximumBlockCount(_KEPT)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._text.setFont(font)
        self._text.setPlainText("\n".join(log.lines))
        layout.addWidget(self._text, 1)

        buttons = QHBoxLayout()
        copy = QPushButton("Copy all")
        copy.setToolTip("Put the whole log on the clipboard, to paste somewhere")
        copy.clicked.connect(self._copy)
        buttons.addWidget(copy)
        folder = QPushButton("Open log folder")
        folder.setToolTip(
            "Show the folder the log file is in. The file keeps everything, "
            "including what happened before this window was opened or after "
            "the program was closed."
        )
        folder.clicked.connect(self._open_folder)
        folder.setEnabled(log.file is not None)
        buttons.addWidget(folder)
        buttons.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        log.added.connect(self._append)
        self.resize(900, 480)
        self._scroll_to_end()

    def _append(self, line: str) -> None:
        bar = self._text.verticalScrollBar()
        following = bar.value() >= bar.maximum() - 2
        self._text.appendPlainText(line)
        if following:
            self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        self._text.moveCursor(QTextCursor.MoveOperation.End)
        self._text.ensureCursorVisible()

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self._text.toPlainText())

    def _open_folder(self) -> None:
        if self._log.file is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._log.file.parent)))

    @property
    def text(self) -> str:
        return self._text.toPlainText()

