"""Tests for choosing the aperture: depth of field against diffraction.

Three parts, the same shape as the focus regions' own tests. The arithmetic --
the ladder the body offers, the shutter that puts the light back, and the walk
along it -- has no camera in it and is run against scores made up to a known
shape. Then the whole of it through the worker against a rig whose pictures
answer to the aperture the way a lens does: the blur from being off the focus
plane shrinks as it is stopped down, and diffraction grows. And then the panel
and the report, which have to offer it and say what it found.

What is checked of the end-to-end run is the aperture chosen, against the best
one worked out from the model itself by brute force -- not against a number
written down here, which would only be testing that the model has not changed.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

from scanny.camera.nikon import CameraError, Setting  # noqa: E402
from scanny.camera.values import format_aperture, format_shutter  # noqa: E402
from scanny.ui.aperture import (  # noqa: E402
    MOST_STOPS,
    STOP_SIZES,
    ApertureSearch,
    Ladder,
    stops_between,
)
from scanny.ui.orientation import Orientation  # noqa: E402
from scanny.ui.regions import CalibrationReport, Look  # noqa: E402
from scanny.ui.report import CalibrationReportDialog  # noqa: E402
from scanny.ui.reportfile import load_report, save_report  # noqa: E402

from test_regions import (  # noqa: E402
    PLACES,
    STEP,
    _af_for,
    _blur,
    _image,
    _normalised,
    _regions_round,
    _report,
    _Rig,
    app,  # noqa: F401 - a fixture, used by name
    window,  # noqa: F401 - a fixture, used by name
    worker,  # noqa: F401 - a fixture, used by name
)
from scanny.ui.sharpness import measure  # noqa: E402

#: A third-stop aperture ladder, as a body reports one: hundredths of an
#: f-stop. Third stops even when the search is asked for whole ones, because
#: that is the awkward case -- the step has to land on what the body has.
APERTURES = (
    280, 320, 360, 400, 450, 500, 560, 630, 710, 800, 900, 1000, 1100,
    1300, 1400, 1600, 1800, 2000, 2200, 2500, 2900, 3200,
)

#: And a third-stop shutter ladder, in PTP's tenths of a millisecond.
SHUTTERS = (
    2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 31, 40, 50, 62, 80, 100, 125,
    166, 200, 250, 333, 400, 500, 666, 769, 1000, 1250, 1666, 2000, 2500,
    3333, 4000, 5000, 6000, 8000, 10000, 13000, 16000, 20000,
)

#: Where a run starts from: f/5.6 at 1/125.
START, START_SHUTTER = 560, 80


def _ladder(apertures=APERTURES, shutters=SHUTTERS) -> Ladder:
    return Ladder(
        [(value, format_aperture(value)) for value in apertures],
        [(value, format_shutter(value)) for value in shutters],
    )


# -- the ladder --------------------------------------------------------------


def test_a_stop_is_a_factor_of_the_square_root_of_two_in_f_numbers():
    # f/5.6 is what the engraving says; the opening is f/5.657, so a stop
    # between the numbers the body reports is a stop less the rounding.
    assert stops_between(400, 560) == pytest.approx(1.0, abs=0.03)
    assert stops_between(560, 400) == pytest.approx(-1.0, abs=0.03)
    assert stops_between(400, 800) == pytest.approx(2.0, abs=0.01)
    assert stops_between(0, 800) == 0.0


@pytest.mark.parametrize(
    "size, expected", [("whole", 800), ("half", 630), ("third", 630)]
)
def test_the_next_aperture_is_the_nearest_the_body_has_to_the_step_asked_for(
    size, expected
):
    """A whole stop from f/5.6 is f/8; a half is f/6.3, which is the nearest
    the body has to it, and so is a third."""
    assert _ladder().next_aperture(START, 1, STOP_SIZES[size]) == expected


def test_opening_up_goes_the_other_way():
    assert _ladder().next_aperture(START, -1, 1.0) == 400


def test_a_lens_with_only_whole_stops_gives_whole_stops_however_finely_asked():
    coarse = _ladder(apertures=(400, 560, 800, 1100, 1600))
    assert coarse.next_aperture(560, 1, STOP_SIZES["third"]) == 800


def test_there_is_nothing_past_the_end_of_the_ladder():
    ladder = _ladder()
    assert ladder.next_aperture(APERTURES[-1], 1, 1.0) is None
    assert ladder.next_aperture(APERTURES[0], -1, 1.0) is None


def test_the_shutter_puts_back_exactly_the_light_the_aperture_lost():
    """Two stops down is two stops of shutter back: 1/125 becomes 1/30."""
    speed, residual = _ladder().compensate(80, 2.0)
    assert speed == 333  # 1/30 s
    # Not to the last hundredth of a stop: 1/30 is not exactly four times
    # 1/125, and the residual is the body's own rounding, said rather than
    # swallowed.
    assert abs(residual) < 0.1
    quicker, residual = _ladder().compensate(80, -1.0)
    assert quicker == 40  # 1/250 s
    assert abs(residual) < 0.1


def test_what_the_shutter_ladder_cannot_match_is_said_rather_than_hidden():
    """A body with only whole stops of shutter leaves a third of a stop over."""
    coarse = _ladder(shutters=(40, 80, 166, 333))
    speed, residual = coarse.compensate(80, 1.0 / 3.0)
    assert speed == 80  # nothing nearer than staying put
    assert residual == pytest.approx(-1.0 / 3.0, abs=0.02)


def test_a_change_the_shutter_cannot_cover_is_the_end_of_that_direction():
    short = _ladder(shutters=(40, 80, 166))
    assert short.compensate(166, 2.0) is None


def test_a_body_metering_for_itself_is_sent_no_shutter_at_all():
    metering = Ladder([(value, "") for value in APERTURES])
    assert metering.compensates
    assert metering.compensate(0, 2.0) == (0, 0.0)


# -- the walk along it -------------------------------------------------------


def _walk(scores, *, ladder=None, size=1.0, most=MOST_STOPS, start=START):
    """Run a search against *scores*, a reading for each aperture it tries.

    *scores* is keyed by aperture, so the shape of the hill is written down
    once and the search is free to visit it in whatever order it likes.
    """
    search = ApertureSearch(
        ladder or _ladder(), start, START_SHUTTER, size=size, most=most
    )
    visited = []
    while not search.done:
        visited.append(search.aperture)
        search.step(scores[search.aperture])
    return search, visited


def test_it_stops_down_while_that_reads_better_and_turns_when_it_does_not():
    scores = {400: 0.5, 560: 0.6, 800: 0.7, 1100: 0.8, 1600: 0.6, 2200: 0.4}
    search, visited = _walk(scores)
    assert visited[:4] == [560, 800, 1100, 1600]
    # Having fallen at f/16 it tries the wide side, which is worse than where
    # it began, so that leg ends at once -- and then it goes back to f/11.
    assert 400 in visited
    assert search.chosen.aperture == 1100
    assert search.outcome == "found"


def test_it_opens_up_when_stopping_down_is_worse_from_the_start():
    scores = {280: 0.5, 400: 0.9, 560: 0.8, 800: 0.6}
    search, visited = _walk(scores)
    assert visited[0] == 560 and visited[1] == 800
    assert 400 in visited and 280 in visited
    assert search.chosen.aperture == 400


def test_it_goes_back_to_read_the_aperture_it_chose():
    """The last probe of a walk that went one step too far is not the answer,
    and the report's pictures are of the answer, so it is read there."""
    scores = {400: 0.5, 560: 0.6, 800: 0.7, 1100: 0.8, 1600: 0.6, 2200: 0.4}
    _search, visited = _walk(scores)
    assert visited[-1] == 1100, visited


def test_it_does_not_go_back_when_it_is_standing_there_already():
    """A walk that ends on its own best has nothing to go back to, and does
    not spend a probe reading the same aperture twice."""
    short = _ladder(apertures=(400, 560, 800, 1100))
    scores = {400: 0.4, 560: 0.5, 800: 0.6, 1100: 0.7}
    _search, visited = _walk(scores, ladder=short)
    assert visited[-1] == 1100
    assert len(visited) == len(set(visited))


def test_the_aperture_it_began_at_is_gone_back_to_when_neither_way_paid():
    """Both legs read worse, so the answer is where it started -- and the
    camera is standing wherever the last probe left it, so it is put back."""
    scores = {280: 0.4, 400: 0.5, 560: 0.9, 800: 0.6}
    search, visited = _walk(scores)
    assert search.chosen.aperture == 560
    assert visited[-1] == 560


def test_it_goes_no_further_than_it_is_let():
    scores = {value: 0.1 * APERTURES.index(value) for value in APERTURES}
    search, visited = _walk(scores, most=2.0)
    assert max(abs(stops_between(START, value)) for value in visited) <= 2.01
    assert search.outcome == "bounded"


def test_a_best_at_the_end_of_the_ladder_says_so():
    """Every step better than the last, all the way to the smallest opening:
    the answer is a bound, because the ladder ran out and not the readings."""
    short = _ladder(apertures=(400, 560, 800, 1100))
    scores = {400: 0.4, 560: 0.5, 800: 0.6, 1100: 0.7}
    search, _visited = _walk(scores, ladder=short)
    assert search.chosen.aperture == 1100
    assert search.outcome == "bounded"


def test_a_tie_is_given_to_the_wider_aperture():
    scores = {400: 0.8, 560: 0.8, 800: 0.8, 1100: 0.5}
    search, _visited = _walk(scores)
    assert search.chosen.aperture == 400


def test_every_aperture_it_tried_is_kept_with_what_it_read():
    scores = {400: 0.5, 560: 0.6, 800: 0.7, 1100: 0.8, 1600: 0.6, 2200: 0.4}
    search, _visited = _walk(scores)
    tried = {probe.aperture: probe.score for probe in search.probes}
    assert tried[800] == 0.7
    stopped = next(one for one in search.probes if one.aperture == 800)
    assert stopped.stops == pytest.approx(1.0, abs=0.05)
    assert stopped.shutter_label == "1/60"  # a stop of shutter back from 1/125


def test_a_body_with_one_aperture_has_nothing_to_try():
    search = ApertureSearch(_ladder(apertures=(560,)), START, START_SHUTTER)
    assert search.step(0.5) is None
    assert search.outcome == "single"


# -- the whole of it, on a rig that answers to the aperture -------------------

#: How much of the blur at the starting aperture is diffraction, per unit of
#: how far it is stopped down, squared. Chosen so that the best aperture is
#: somewhere in the middle of the ladder rather than at one end of it, which
#: is the case worth testing.
DIFFRACTION = 0.05


class _Stopped(_Rig):
    """The same rig, with an aperture that means something.

    Two blurs rather than one, and the point of the whole feature is that they
    pull opposite ways. The blur from being off the focus plane is the
    circle of confusion, which shrinks in proportion to the opening, so
    stopping down divides the distance out of focus by however many times
    smaller the opening is. Diffraction grows with the same number instead.
    Both are added as variance, which is what a pass of the blur here is.
    """

    def __init__(self, places, *, aperture: int = START, shutter: int = START_SHUTTER,
                 diffraction: float = DIFFRACTION, refuses: bool = False, **rest):
        super().__init__(places, **rest)
        self.aperture = aperture
        self.shutter = shutter
        self.diffraction = diffraction
        #: A body that will not have its aperture set from here -- a lens with
        #: its aperture ring off the minimum, or shutter priority.
        self.refuses = refuses
        self.apertures_set: "list[int]" = []
        self.shutters_set: "list[int]" = []

    # -- the camera's side ---------------------------------------------------

    def setting(self, name: str) -> "Setting | None":
        if name == "Aperture":
            return Setting(
                code=0x5007,
                name=name,
                value=self.aperture,
                label=format_aperture(self.aperture),
                writable=not self.refuses,
                choices=tuple((v, format_aperture(v)) for v in APERTURES),
            )
        if name == "Shutter":
            return Setting(
                code=0x500D,
                name=name,
                value=self.shutter,
                label=format_shutter(self.shutter),
                writable=True,
                choices=tuple((v, format_shutter(v)) for v in SHUTTERS),
            )
        return None

    def set_setting(self, name, value) -> None:
        self._in_live_view(f"set {name}")
        if name == "Aperture":
            self.aperture = int(value)
            self.apertures_set.append(int(value))
        elif name == "Shutter":
            self.shutter = int(value)
            self.shutters_set.append(int(value))

    # -- the scene -----------------------------------------------------------

    def picture(self, crop, optics: float, aperture: "int | None" = None):
        opening = self.aperture if aperture is None else aperture
        times = opening / START
        base = self._textures.get(crop)
        if base is None:
            from test_regions import _texture

            base = _texture(*self._shape(), seed=abs(hash(crop)) % 9973)
            self._textures[crop] = base
        away = (optics - self._place(crop)[2]) / self.softness / times
        passes = 2.0 * (np.hypot(1.0, away) - 1.0) + self.diffraction * times * times
        return _blur(base, min(passes, 7.99))

    def _shape(self):
        """The size of a live-view frame here, as the base rig makes them."""
        from test_regions import TALL, WIDE

        return TALL, WIDE


def _calibrated(worker, rig, regions, *, apertures="whole", frames=2, limit=40000,
                stop_after: "int | None" = None):
    """Run a whole calibration on *rig* and hand back its report.

    With *stop_after*, it is stopped once that many apertures have been set,
    which is how the half-finished case is got at.
    """
    worker._camera = rig
    worker.set_integration(True, frames)
    for _ in range(8):
        worker._grab()
    worker.set_focus_regions(regions)
    reports = []
    worker.calibrationReady.connect(lambda report: reports.append(report))
    worker.start_calibration(STEP, "average", 80, False, 1, apertures)
    for _ in range(limit):
        if worker._calibration is None:
            break
        if stop_after is not None and len(rig.apertures_set) >= stop_after:
            worker.cancel_calibration()
            break
        worker._grab()
    assert worker._calibration is None, "the calibration has to finish on its own"
    assert reports and reports[-1] is not None
    return reports[-1]


def _true_scores(rig, report, position: float) -> "dict[int, float]":
    """What the model says every aperture on the ladder is worth, at *position*.

    Read without grain, through each region's own view, against the best the
    model can give that region at the aperture the calibration ran at -- which
    is what the search is measuring shares of.
    """
    scores: "dict[int, float]" = {}
    for aperture in APERTURES:
        shares = []
        for result in report.results:
            crop = rig._crop(7 if result.region.w < 0.05 else 6, _af_for(result.region))
            shown = result.region.seen_in(_normalised(crop))
            best = rig._place(crop)[2]
            peak = measure(_image(rig.picture(crop, best, START)), shown)
            here = measure(_image(rig.picture(crop, position, aperture)), shown)
            shares.append(here / peak if peak > 0 else 0.0)
        scores[aperture] = float(np.mean(shares))
    return scores


def test_it_finds_the_aperture_the_model_says_is_best(worker):
    """The whole of what was asked for: three places at three depths, one
    focus position found for all of them, and then the opening that serves
    them best -- against the best the model itself can give, by brute force."""
    rig = _Stopped(PLACES)
    report = _calibrated(worker, rig, _regions_round(PLACES))
    truth = _true_scores(rig, report, rig.optics)
    best = max(truth, key=lambda value: truth[value])
    assert report.aperture_chosen != 0
    # Within a per cent of the best the ladder holds. Not the same aperture
    # necessarily: neighbouring apertures can read within the grain of each
    # other, and what matters is what it gave away, not which rung it is on.
    chosen = truth[report.aperture_chosen]
    assert chosen >= truth[best] - 0.01, (
        f"chose {format_aperture(report.aperture_chosen)} worth {chosen:.1%}, "
        f"where {format_aperture(best)} is worth {truth[best]:.1%}"
    )
    # And it is not simply where it started: this rig is worth stopping down.
    assert report.aperture_chosen > report.aperture_started


def test_the_exposure_is_put_back_stop_for_stop(worker):
    rig = _Stopped(PLACES)
    report = _calibrated(worker, rig, _regions_round(PLACES))
    assert report.apertures
    for probe in report.apertures:
        lost = stops_between(START, probe.aperture)
        given = math.log2(probe.shutter / START_SHUTTER)
        assert abs(given - lost) < 0.2, (
            f"{probe.label} at {probe.shutter_label} is {given - lost:+.2f} EV out"
        )
    # And what the body was actually sent matches what the report says.
    assert rig.aperture == report.aperture_chosen
    assert rig.shutters_set, "the shutter has to move with the aperture"


def test_the_focus_is_left_where_the_compromise_put_it(worker):
    """Stopping down deepens the focus about the plane the compromise chose,
    so the lens has no business moving while the apertures are tried."""
    rig = _Stopped(PLACES)
    worker._camera = rig
    worker.set_integration(True, 2)
    for _ in range(8):
        worker._grab()
    worker.set_focus_regions(_regions_round(PLACES))
    worker.start_calibration(STEP, "average", 80, False, 1, "whole")
    settled: "float | None" = None
    for _ in range(40000):
        if worker._calibration is None:
            break
        if worker._calibration.phase == "apertures" and settled is None:
            settled = rig.optics
        worker._grab()
    assert settled is not None, "the aperture phase has to have been reached"
    assert rig.optics == settled


def test_without_asking_no_aperture_is_tried(worker):
    rig = _Stopped(PLACES)
    report = _calibrated(worker, rig, _regions_round(PLACES), apertures="")
    assert report.apertures == ()
    assert report.aperture_chosen == 0
    assert rig.apertures_set == []
    assert rig.aperture == START


def test_a_body_that_will_not_set_its_aperture_says_so_and_the_rest_stands(worker):
    rig = _Stopped(PLACES, refuses=True)
    report = _calibrated(worker, rig, _regions_round(PLACES))
    assert report.apertures == ()
    assert "aperture" in report.aperture_note.lower()
    assert rig.apertures_set == []
    # The compromise is still there, and still the answer.
    assert report.score is not None
    assert report.outcome != "stopped"


def test_a_calibration_stopped_part_way_puts_the_aperture_back(worker):
    rig = _Stopped(PLACES)
    report = _calibrated(worker, rig, _regions_round(PLACES), stop_after=2)
    assert rig.aperture == START, "the aperture it found has to be put back"
    assert rig.shutter == START_SHUTTER
    assert report.aperture_chosen == 0
    assert report.aperture_outcome == "stopped"


def test_every_reading_at_every_aperture_is_in_the_report(worker):
    rig = _Stopped(PLACES)
    report = _calibrated(worker, rig, _regions_round(PLACES))
    tried = {probe.aperture for probe in report.apertures}
    noted = {one.position for one in report.readings if one.stage == "aperture"}
    assert noted == tried
    # Every usable region, at every aperture.
    for aperture in tried:
        at = [one for one in report.readings
              if one.stage == "aperture" and one.position == aperture]
        assert {one.region for one in at} == set(report.history_regions)


# -- the panel and the report ------------------------------------------------


def test_the_option_is_asked_for_before_calibrating_and_remembered(window):
    QSettings().clear()
    window.regions_apertures.setChecked(True)
    window.regions_aperture_stops.setCurrentIndex(
        window.regions_aperture_stops.findData("half")
    )
    asked = []
    window.requestCalibration.connect(lambda *args: asked.append(args))
    window._on_region_drawn(0.1, 0.1, 0.1, 0.1)
    window._on_region_drawn(0.6, 0.6, 0.1, 0.1)
    window.calibrate_button.click()
    assert asked and asked[0][5] == "half"
    assert QSettings().value("regions/apertures", False, bool) is True
    assert str(QSettings().value("regions/aperture_stops")) == "half"


def test_the_stop_size_is_closed_off_until_apertures_are_asked_for(window):
    window.regions_apertures.setChecked(False)
    assert not window.regions_aperture_stops.isEnabled()
    window.regions_apertures.setChecked(True)
    assert window.regions_aperture_stops.isEnabled()


def _with_apertures() -> CalibrationReport:
    from dataclasses import replace

    from scanny.ui.aperture import Probe

    base = _report()
    # Each region also read at the aperture it started at, which is what the
    # before-and-after on the page is made of: region 1 gained, region 2 lost.
    results = (
        replace(base.results[0], before_aperture=Look(112.0, base.results[0].best.picture)),
        replace(base.results[1], before_aperture=Look(54.0, base.results[1].best.picture)),
    )
    return replace(
        base,
        results=results,
        history_regions=(1, 2),
        apertures=(
            Probe(560, "f/5.6", 0.0, 80, "1/125", 0.0, (0.8, 0.7), 0.75),
            Probe(800, "f/8", 1.0, 166, "1/60", 0.0, (0.9, 0.85), 0.875),
            Probe(1100, "f/11", 2.0, 333, "1/30", 0.0, (0.8, 0.8), 0.8),
        ),
        aperture_started=560,
        aperture_chosen=800,
        aperture_outcome="found",
    )


def test_a_report_with_no_aperture_search_has_no_page_for_one(window):
    dialog = CalibrationReportDialog(_report(), Orientation(), window._save_directory())
    pages = [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]
    assert "Aperture" not in pages
    dialog.close()


def test_a_report_with_one_shows_every_aperture_it_tried(window):
    dialog = CalibrationReportDialog(
        _with_apertures(), Orientation(), window._save_directory()
    )
    pages = [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]
    assert "Aperture" in pages
    shown = dialog._aperture_page.widget()
    text = " ".join(label.text() for label in shown.findChildren(QLabel))
    for wanted in ("f/5.6", "f/8", "f/11", "1/60", "chosen"):
        assert wanted in text, f"{wanted!r} is not on the page"
    dialog.close()


def test_what_it_found_is_said_in_one_line():
    line = _with_apertures().describe_aperture()
    assert "f/8" in line and "stopped down" in line and "1/60" in line
    assert "1.0 stops" in line
    # What it was worth against where it began: 87.5% where 75% was.
    assert "+17%" in line


def test_the_aperture_survives_being_saved_and_opened(tmp_path, app):
    path = tmp_path / "one.focusreport"
    save_report(path, _with_apertures(), 1.5)
    read = load_report(path).report
    assert read.aperture_chosen == 800
    assert read.aperture_started == 560
    assert read.aperture_outcome == "found"
    assert [probe.label for probe in read.apertures] == ["f/5.6", "f/8", "f/11"]
    assert read.apertures[1].shares == (0.9, 0.85)
    assert read.apertures[1].shutter_label == "1/60"


def test_a_report_from_before_there_were_apertures_reads_back_as_having_none(
    tmp_path, app
):
    path = tmp_path / "old.focusreport"
    save_report(path, _report(), 1.5)
    read = load_report(path).report
    assert read.apertures == ()
    assert read.aperture_chosen == 0
    assert read.describe_aperture() == ""


def test_live_view_the_body_ends_while_apertures_are_tried_is_not_the_end(worker):
    """The body turns live view off after its own monitor-off delay, whoever
    is driving it, and an aperture probe caught by that is taken again."""
    rig = _Stopped(PLACES)
    worker._camera = rig
    worker.set_integration(True, 2)
    for _ in range(8):
        worker._grab()
    worker.set_focus_regions(_regions_round(PLACES))
    worker.start_calibration(STEP, "average", 80, False, 1, "whole")
    ended = False
    for _ in range(40000):
        if worker._calibration is None:
            break
        if not ended and worker._calibration.phase == "apertures":
            rig.ends_after = 2
            ended = True
        worker._grab()
    assert ended and rig.restarts >= 1
    assert worker._calibration is None


def test_the_aperture_is_part_of_what_a_region_s_grain_is_kept_under():
    """Two apertures are two exposures, and two exposures are two grains."""
    from scanny.ui.worker import _region_grain_key

    assert _region_grain_key(0, 640, 360, 560) != _region_grain_key(0, 640, 360, 800)
    assert _region_grain_key(0, 640, 360) == _region_grain_key(0, 640, 360, 0)


def test_a_camera_that_refuses_mid_search_stops_the_calibration_rather_than_hanging(
    worker,
):
    rig = _Stopped(PLACES)
    worker._camera = rig
    worker.set_integration(True, 2)
    for _ in range(8):
        worker._grab()
    worker.set_focus_regions(_regions_round(PLACES))
    reports = []
    worker.calibrationReady.connect(lambda report: reports.append(report))
    worker.start_calibration(STEP, "average", 80, False, 1, "whole")
    broken = False
    for _ in range(40000):
        if worker._calibration is None:
            break
        if not broken and worker._calibration.phase == "apertures":
            def refuse(name, value):
                raise CameraError("the lens will not stop down")

            rig.set_setting = refuse
            broken = True
        worker._grab()
    assert broken
    assert worker._calibration is None
    assert reports and reports[-1] is not None


def test_the_page_shows_every_region_at_both_apertures(window):
    """The numbers say what the change of aperture was worth; the pictures say
    whether that is the picture somebody wants. Both columns are the same
    region at the same focus position, so the only thing that differs between
    them is the opening."""
    dialog = CalibrationReportDialog(
        _with_apertures(), Orientation(), window._save_directory()
    )
    shown = dialog._aperture_page.widget()
    text = " ".join(label.text() for label in shown.findChildren(QLabel))
    # The headings name both apertures, so a reader knows which column is which.
    assert "where it ran" in text and "where it was left" in text
    assert "f/5.6" in text and "f/8" in text
    # Region 1 went 112 -> 140, region 2 went 54 -> 51.
    assert "+25%" in text, text
    assert "-6%" in text, text
    assert "sharper" in text and "softer" in text
    # A picture each side for each region, on top of the Regions page's pair.
    pictures = [one for one in dialog._pictures if one.pixmap() is not None]
    assert len(pictures) >= 8
    dialog.close()


def test_a_region_not_read_at_both_apertures_is_left_off_that_page(window):
    from dataclasses import replace

    report = _with_apertures()
    bare = replace(report, results=tuple(
        replace(one, before_aperture=None) for one in report.results
    ))
    dialog = CalibrationReportDialog(bare, Orientation(), window._save_directory())
    shown = dialog._aperture_page.widget()
    text = " ".join(label.text() for label in shown.findChildren(QLabel))
    # The table of apertures is still there; the before-and-after is not.
    assert "f/11" in text
    assert "where it was left" not in text
    dialog.close()


def test_what_the_aperture_was_worth_to_each_region_survives_being_saved(
    tmp_path, app
):
    path = tmp_path / "both.focusreport"
    save_report(path, _with_apertures(), 1.5)
    read = load_report(path).report
    one, two = read.results
    assert one.before_aperture is not None
    assert one.before_aperture.reading == 112.0
    assert one.before_aperture.picture is not None
    assert one.aperture_gain == pytest.approx(140.0 / 112.0 - 1.0)
    assert two.aperture_gain == pytest.approx(51.0 / 54.0 - 1.0)
    assert one.before_fraction == pytest.approx(112.0 / 150.0)


def test_a_real_run_keeps_each_region_at_the_aperture_it_started_at(worker):
    """Through the worker: the pictures the page is made of are read off the
    camera at the first aperture tried and at the one it was left on."""
    rig = _Stopped(PLACES)
    report = _calibrated(worker, rig, _regions_round(PLACES))
    placed = [one for one in report.results if one.usable]
    assert placed
    for one in placed:
        assert one.before_aperture is not None, f"region {one.number}"
        assert one.before_aperture.picture is not None
        assert one.compromise is not None
        # It stopped down on this rig, so every region read better for it.
        assert one.aperture_gain is not None
    gained = [one.aperture_gain for one in placed]
    assert max(gained) > 0.05, gained
