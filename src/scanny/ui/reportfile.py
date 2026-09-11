"""A calibration's report as a file, to be kept and opened again later.

A picture of one page of the report keeps what that page looked like, and
nothing else: not the other pages, not the film's shape to be turned round,
not what was said while it ran. So the whole report is written out instead,
and opened again it is the same report -- every region at its best and at the
compromise, the depths and the film's shape if they were measured, the search
probe by probe, and the activity log of the run.

The file is a zip archive, so that what is in it can be got at without this
program as well: ``report.json`` holds the numbers, ``pictures/`` the regions
as they were cut out of the live view -- in the camera's own orientation and
tones, as :class:`scanny.ui.regions.Look` keeps them -- and ``activity.log``
the log.
"""

from __future__ import annotations

import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QImage

from .regions import CalibrationReport, Look, Region, RegionResult

__all__ = [
    "REPORT_FILTER",
    "SUFFIX",
    "ReportFileError",
    "SavedReport",
    "load_report",
    "save_report",
]

#: What a report file is called, and how the file dialogs offer it.
SUFFIX = ".focusreport"
REPORT_FILTER = f"Focus report (*{SUFFIX})"

#: What says a file is one of these, and which way of writing it. A file of a
#: later version than this program knows is refused rather than half read.
_FORMAT = "scanny focus report"
_VERSION = 1

_NUMBERS = "report.json"
_LOG = "activity.log"


class ReportFileError(Exception):
    """A report file that could not be written, or read back."""


@dataclass(frozen=True)
class SavedReport:
    """A report read back from a file, and the frame's shape it was made on."""

    report: CalibrationReport
    #: The sensor frame's width over its height, for drawing the film.
    aspect: float


def save_report(path: "str | Path", report: CalibrationReport, aspect: float) -> None:
    """Write *report* to *path*, whole.

    Written beside it first and put in its place once complete, so a save
    that fails half way leaves whatever was there before.
    """
    path = Path(path)
    pictures: "dict[str, bytes]" = {}

    def picture(look: "Look | None", name: str) -> "dict | None":
        if look is None:
            return None
        kept = None
        if look.picture is not None and not look.picture.isNull():
            kept = f"pictures/{name}.png"
            pictures[kept] = _png(look.picture)
        return {"reading": look.reading, "picture": kept}

    numbers = {
        "format": _FORMAT,
        "version": _VERSION,
        "saved": time.time(),
        "aspect": float(aspect),
        "report": {
            "score": report.score,
            "outcome": report.outcome,
            "probes": report.probes,
            "objective": report.objective,
            "seconds": report.seconds,
            "tune_probes": list(report.tune_probes),
            "autofocuses": report.autofocuses,
            "moves": report.moves,
            "travel": report.travel,
            "history": [
                [stage, list(shares), combined]
                for stage, shares, combined in report.history
            ],
            "history_regions": list(report.history_regions),
            "depths_measured": report.depths_measured,
            "began": report.began,
            "results": [
                {
                    "number": one.number,
                    "region": list(one.region.rect),
                    "outcome": one.outcome,
                    "best": picture(one.best, f"region-{one.number}-best"),
                    "compromise": picture(
                        one.compromise, f"region-{one.number}-compromise"
                    ),
                    "depth": one.depth,
                    # JSON has no infinity: an unknown doubt is written as none.
                    "doubt": one.doubt if math.isfinite(one.doubt) else None,
                    "edge": one.edge,
                }
                for one in report.results
            ],
        },
    }
    partial = path.with_name(path.name + ".part")
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(_NUMBERS, json.dumps(numbers, indent=1))
            archive.writestr(_LOG, "".join(f"{line}\n" for line in report.log))
            for name, data in pictures.items():
                # Already compressed; deflating them again only costs time.
                archive.writestr(name, data, compress_type=zipfile.ZIP_STORED)
        os.replace(partial, path)
    except OSError as exc:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise ReportFileError(f"Could not write {path}: {exc}") from exc


def load_report(path: "str | Path") -> SavedReport:
    """Read a report written by :func:`save_report` back from *path*."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            try:
                numbers = json.loads(archive.read(_NUMBERS).decode("utf-8"))
            except KeyError:
                raise ReportFileError(f"{path.name} is not a focus report") from None
            if not isinstance(numbers, dict) or numbers.get("format") != _FORMAT:
                raise ReportFileError(f"{path.name} is not a focus report")
            if int(numbers.get("version", 0)) > _VERSION:
                raise ReportFileError(
                    f"{path.name} was written by a newer version of scanny"
                )
            try:
                log = archive.read(_LOG).decode("utf-8").splitlines()
            except KeyError:
                log = []
            return SavedReport(
                report=_report_from(numbers["report"], archive, log),
                aspect=float(numbers.get("aspect", 1.5)) or 1.5,
            )
    except ReportFileError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReportFileError(f"Could not read {path.name}: {exc}") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportFileError(f"{path.name} is damaged: {exc}") from exc


def _report_from(
    saved: dict, archive: zipfile.ZipFile, log: "list[str]"
) -> CalibrationReport:
    def look(kept: "dict | None") -> "Look | None":
        if kept is None:
            return None
        picture = None
        if kept.get("picture"):
            picture = QImage.fromData(archive.read(kept["picture"]), "PNG")
            if picture.isNull():
                picture = None
        return Look(float(kept["reading"]), picture)

    results = tuple(
        RegionResult(
            number=int(one["number"]),
            region=Region(*(float(part) for part in one["region"])),
            best=look(one["best"]),
            outcome=str(one["outcome"]),
            compromise=look(one["compromise"]),
            depth=None if one["depth"] is None else float(one["depth"]),
            doubt=float("inf") if one["doubt"] is None else float(one["doubt"]),
            edge=bool(one["edge"]),
        )
        for one in saved["results"]
    )
    return CalibrationReport(
        results=results,
        score=None if saved["score"] is None else float(saved["score"]),
        outcome=str(saved["outcome"]),
        probes=int(saved["probes"]),
        objective=str(saved["objective"]),
        seconds=float(saved["seconds"]),
        tune_probes=tuple(int(count) for count in saved["tune_probes"]),
        autofocuses=int(saved["autofocuses"]),
        moves=int(saved["moves"]),
        travel=int(saved["travel"]),
        history=tuple(
            (str(stage), tuple(float(share) for share in shares), float(combined))
            for stage, shares, combined in saved["history"]
        ),
        history_regions=tuple(int(number) for number in saved["history_regions"]),
        depths_measured=bool(saved["depths_measured"]),
        began=float(saved.get("began", 0.0)),
        log=tuple(log),
    )


def _png(picture: QImage) -> bytes:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    picture.save(buffer, "PNG")
    buffer.close()
    return bytes(data.data())
