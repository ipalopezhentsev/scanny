"""Tests for mapping the depth of the scene by sweeping focus.

Two halves, as with the hunt. The arithmetic has no camera in it and is run
against pictures built to a known shape -- a scene whose left side is focused
at one position and whose right side is focused at another, rendered at a
series of focus positions. Then the whole thing through the worker against a
simulated lens with stops at both ends of its travel and live view running
behind it, because what is easy to get wrong is not the peak finding but the
driving: where the datum is, and that a pass never turns round.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtGui")

from PySide6.QtCore import QBuffer, QByteArray  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402

from scanny.camera.nikon import LiveViewFrame  # noqa: E402
from scanny.ui.depth import (  # noqa: E402
    ACROSS,
    LEVELS,
    MIN_TILE,
    Survey,
    Sweep,
    as_steps_image,
    coarsen,
    colourise,
    readings,
    tile_sums,
    tiling_for,
)
from scanny.ui.sharpness import measure  # noqa: E402

#: The frame the arithmetic is tested against: big enough for the full pyramid.
BIG = (640, 360)


# -- pictures ----------------------------------------------------------------


def _texture(height: int, width: int, seed: int = 1) -> np.ndarray:
    """Detail that does not repeat, which is what real subjects look like."""
    field = np.random.default_rng(seed).normal(0, 1, (height, width))
    for _ in range(2):
        field = (
            field
            + np.roll(field, 1, 0) + np.roll(field, -1, 0)
            + np.roll(field, 1, 1) + np.roll(field, -1, 1)
        ) / 5
    return np.clip(128 + 55 * field / field.std(), 0, 255)


def _blur(pixels: np.ndarray, passes: float) -> np.ndarray:
    """Soften by a smooth number of passes, so the reading is not a staircase."""
    stages = [pixels]
    for _ in range(int(passes) + 1):
        last = stages[-1]
        stages.append(
            (
                last
                + np.roll(last, 1, 0) + np.roll(last, -1, 0)
                + np.roll(last, 1, 1) + np.roll(last, -1, 1)
            ) / 5
        )
    whole = int(passes)
    weight = passes - whole
    return stages[whole] * (1 - weight) + stages[whole + 1] * weight


def _image(pixels: np.ndarray) -> QImage:
    height, width = pixels.shape
    grey = np.clip(pixels, 0, 255).astype(np.uint8)
    buffer = np.zeros((height, width, 4), np.uint8)
    for channel in range(3):
        buffer[:, :, channel] = grey
    buffer[:, :, 3] = 255
    return QImage(
        buffer.tobytes(), width, height, width * 4, QImage.Format.Format_RGB32
    ).copy()


class Ramp:
    """A scene focused nearer on one side than the other, in vertical bands."""

    def __init__(
        self,
        width: int,
        height: int,
        bands: int = 8,
        near: float = 1000.0,
        far: float = 4000.0,
        depth: float = 300.0,
        blank: "int | None" = None,
        faint: "tuple[int, float] | None" = None,
        noise: float = 0.0,
        seed: int = 3,
    ) -> None:
        self.width, self.height, self.bands = width, height, bands
        self.best = np.linspace(near, far, bands)
        self.depth = depth
        self.blank = blank
        #: A band with the same detail in it at a hundredth of the contrast,
        #: for a subject that is really there but reads nothing like the rest
        #: of the picture.
        self.faint = faint
        self.noise = noise
        self._subject = _texture(height, width, seed)
        self._rng = np.random.default_rng(seed + 1)

    def band(self, column: int) -> int:
        return min(int(column * self.bands / self.width), self.bands - 1)

    def at(self, position: float) -> QImage:
        """The picture the camera would send with focus at *position*."""
        edges = np.linspace(0, self.width, self.bands + 1).astype(int)
        out = np.empty((self.height, self.width))
        for index in range(self.bands):
            left, right = edges[index], edges[index + 1]
            if index == self.blank:
                out[:, left:right] = 128.0
                continue
            softness = min(abs(position - self.best[index]) / self.depth, 7.99)
            band = _blur(self._subject, softness)[:, left:right]
            if self.faint is not None and index == self.faint[0]:
                band = 128.0 + (band - 128.0) * self.faint[1]
            out[:, left:right] = band
        if self.noise:
            out = out + self._rng.normal(0, self.noise, out.shape)
        return _image(out)


# -- how the picture is divided ----------------------------------------------


def test_the_pyramid_halves_the_zone_at_every_level():
    tiling = tiling_for(*BIG)
    assert tiling.levels == LEVELS
    assert [tiling.shape(level) for level in range(LEVELS)] == [
        (9, 16), (18, 32), (36, 64)
    ]
    assert [tiling.tile(level) for level in range(LEVELS)] == [
        (40, 40), (20, 20), (10, 10)
    ]


def test_a_frame_too_small_for_the_finest_grid_loses_a_level_instead():
    """Rather than measuring zones of four pixels, which read nearly pure grain."""
    for width, height in ((640, 360), (320, 180), (160, 90)):
        tiling = tiling_for(width, height)
        tall, wide = tiling.tile(tiling.levels - 1)
        assert min(tall, wide) >= MIN_TILE, (width, height, tiling)
        assert tiling.left + tiling.width <= width
        assert tiling.top + tiling.height <= height


def test_zones_are_whole_and_centred_on_the_picture():
    tiling = tiling_for(645, 363)
    assert tiling.rows * tiling.tile_height <= 363
    assert tiling.cols * tiling.tile_width <= 645
    assert tiling.left > 0 and tiling.top > 0


# -- the reading -------------------------------------------------------------


def test_a_coarse_zone_is_exactly_the_fine_zones_inside_it():
    """The whole reason the picture is read once and drawn at any resolution."""
    tiling = tiling_for(*BIG)
    sums = tile_sums(Ramp(*BIG).at(2500), tiling)
    fine = coarsen(sums, tiling, LEVELS - 1)
    coarse = coarsen(sums, tiling, 0)
    step = tiling.factor(0)
    folded = fine.reshape(3, coarse.shape[1], step, coarse.shape[2], step).sum((2, 4))
    assert np.allclose(folded, coarse)


def test_one_zone_over_the_whole_picture_reads_what_the_sharpness_meter_does():
    """The same number on the same scale, so a map and a hunt agree."""
    tiling = tiling_for(*BIG, columns=1, levels=1)
    picture = Ramp(*BIG).at(2500)
    mine = float(readings(tile_sums(picture, tiling), tiling, 0)[0, 0])
    theirs = measure(picture)
    assert mine == pytest.approx(theirs, rel=0.02)


def test_a_zone_with_nothing_in_it_reads_nothing_rather_than_its_noise():
    tiling = tiling_for(*BIG)
    scene = Ramp(*BIG, blank=0, noise=5.0)
    values = readings(tile_sums(scene.at(2500), tiling), tiling, 0, noise_variance=25.0)
    assert not values[:, 0].any(), "the blank band answered with its own grain"
    assert values[:, -1].any(), "the textured bands stopped answering"


# -- finding the peak --------------------------------------------------------


def _survey(scene: Ramp, positions, tiling=None) -> Survey:
    tiling = tiling or tiling_for(scene.width, scene.height)
    survey = Survey(tiling)
    for position in positions:
        survey.add(position, tile_sums(scene.at(position), tiling), scene.noise ** 2)
    return survey


def test_it_recovers_a_scene_whose_depth_is_known():
    scene = Ramp(*BIG)
    survey = _survey(scene, range(500, 4600, 100))
    depth_map = survey.map(LEVELS - 1)
    assert depth_map.coverage == 1.0
    tiling = survey.tiling
    for column in range(depth_map.depth.shape[1]):
        # The middle of this zone, in pixels, decides which band it is in.
        middle = tiling.left + int((column + 0.5) * tiling.tile_width)
        want = scene.best[scene.band(middle)]
        got = float(np.median(depth_map.depth[:, column]))
        assert abs(got - want) <= 60, f"zone {column} read {got:.0f}, not {want:.0f}"


def test_the_answer_is_finer_than_the_step_it_was_sampled_at():
    """Three readings round a maximum fix a parabola, and its top is the answer."""
    scene = Ramp(*BIG, bands=4)
    coarse = _survey(scene, range(400, 4800, 400)).map(0)
    answers = coarse.depth[np.isfinite(coarse.depth)]
    assert answers.size
    assert not np.allclose(answers % 400, 0), "every answer landed on a sample"
    off_grid = np.abs((answers - 400) % 400)
    assert np.median(np.minimum(off_grid, 400 - off_grid)) > 5


def test_a_fine_zone_is_only_believed_where_the_zone_around_it_answered():
    """A ten-pixel zone clearing its grain once in twenty stops is noise, and
    noise passes a prominence test easily because the floor is noisy too."""
    tiling = tiling_for(*BIG)
    survey = Survey(tiling)
    rng = np.random.default_rng(4)
    for position in range(0, 4000, 200):
        grain = 128 + rng.normal(0, 4.0, (BIG[1], BIG[0]))
        survey.add(position, tile_sums(_image(grain), tiling), 16.0)
    fine = survey._peaks(LEVELS - 1)
    assert fine.known.any(), "the grain never once fooled a fine zone"
    assert not survey.map(LEVELS - 1).known.any()


def test_a_blank_patch_borrows_from_the_zone_around_it_or_stays_blank():
    scene = Ramp(*BIG, blank=0, noise=4.0)
    survey = _survey(scene, range(500, 4600, 150))
    depth_map = survey.map(LEVELS - 1)
    wide = survey.tiling.factor(0)  # fine zones to a coarse one
    # The leftmost coarse zone lies wholly inside the blank band, so neither it
    # nor anything inside it has an answer to give or to inherit.
    assert not depth_map.known[:, :wide].any()
    # Nothing inside the band ever answers for itself. What is drawn there at
    # all is inherited from the coarse zone that reaches the band's edge, where
    # there is real contrast and it does go soft with focus -- and it is marked
    # as inherited, which is what lets the drawing tell the two apart.
    inside = slice(None, 2 * wide - 1)
    assert not (depth_map.known[:, inside] & ~depth_map.borrowed[:, inside]).any()
    assert depth_map.borrowed[:, inside].any(), "nothing was filled in at all"
    assert depth_map.coverage < 1.0


def test_a_picture_flatter_than_its_own_levels_is_not_a_reading():
    """The one floor under the grain that no measurement of it may go below.

    The grain is measured from how much two consecutive frames differ, and a
    thoroughly defocused live view is smooth enough that JPEG hands back
    almost the same frame twice. The measurement then falls towards zero and
    takes the whole "nothing here" test with it, because anything at all is
    above nothing. An eight-bit picture cannot be flatter than its own levels,
    and that is what the floor says.
    """
    tiling = tiling_for(*BIG)
    # A ramp of a twentieth of a level per pixel: after rounding, one pair of
    # neighbours in twenty differs, and by exactly one level.
    ramp = 128.0 + 0.05 * np.arange(BIG[0], dtype=float)[None, :]
    sums = tile_sums(_image(np.repeat(ramp, BIG[1], axis=0)), tiling)
    # There is gradient energy in it -- those rounding steps are real
    # differences, and with the grain measured as zero they would be readings.
    assert coarsen(sums, tiling, LEVELS - 1)[ACROSS].sum() > 0
    assert not readings(sums, tiling, LEVELS - 1, noise_variance=0.0).any()


def test_a_zone_far_below_the_rest_of_the_picture_is_not_believed():
    """Every test but this one is local to a zone, which is what lets a whole
    region of nothing agree with itself: each zone wanders, each wander has a
    largest value, and nothing local tells that from a subject."""
    scene = Ramp(*BIG, bands=8, faint=(0, 0.05))
    depth_map = _survey(scene, range(500, 4600, 150)).map(0)
    faint = depth_map.depth.shape[1] // scene.bands
    # All but the zone reaching the band's edge, where the step from a faint
    # band to a full-contrast one is real contrast and does go soft with focus.
    assert not depth_map.known[:, : faint - 1].any(), "the faint band was believed"
    assert depth_map.known[:, faint + 1 :].all()


def test_a_peak_at_the_end_of_the_sweep_is_reported_as_one():
    """It does not mean the zone is the furthest thing in the scene; it means
    the sweep did not contain its peak, which is also what a zone reading the
    edge of somebody else's bokeh disc looks like."""
    scene = Ramp(*BIG, bands=4, near=1000, far=4000)
    # Stopped well short of the far bands, so their readings are still rising.
    depth_map = _survey(scene, range(500, 2100, 100)).map(0)
    assert depth_map.edge.any()
    assert "at the limit of the sweep" in depth_map.describe()


def test_a_map_of_nothing_says_so_rather_than_drawing_the_noise():
    tiling = tiling_for(*BIG)
    survey = Survey(tiling)
    rng = np.random.default_rng(9)
    for position in range(0, 4000, 200):
        flat = 128 + rng.normal(0, 4.0, (BIG[1], BIG[0]))
        survey.add(position, tile_sums(_image(flat), tiling), 16.0)
    depth_map = survey.map(LEVELS - 1)
    assert depth_map.coverage == 0.0
    assert "Nothing" in depth_map.describe()
    assert survey.interesting(200) is None


def test_the_range_worth_sweeping_again_covers_the_scene():
    scene = Ramp(*BIG, near=1500, far=2500)
    span = _survey(scene, range(0, 6000, 400)).interesting(400)
    assert span is not None
    low, high = span
    assert low <= 1500 and high >= 2500
    assert high - low < 6000, "it did not narrow anything"


def test_the_whole_frame_reads_the_picture_as_one_zone():
    """The steadiest curve there is, and the last resort for pointing a pass."""
    scene = Ramp(*BIG, bands=1, near=2200, far=2200)
    where, curve = _survey(scene, range(1000, 3500, 100)).whole_frame()
    assert abs(where[int(curve.argmax())] - 2200) <= 100


def test_a_curve_with_no_peak_in_it_points_the_next_pass_nowhere():
    """Otherwise the grain chooses the stretch that gets swept again."""
    tiling = tiling_for(*BIG)
    survey = Survey(tiling)
    subject = _texture(BIG[1], BIG[0])
    rng = np.random.default_rng(11)
    for position in range(0, 4000, 200):
        # Sharp everywhere, so nothing anywhere has a peak to be placed by.
        flat = subject + rng.normal(0, 1.0, subject.shape)
        survey.add(position, tile_sums(_image(flat), tiling), 1.0)
    assert not np.isfinite(survey.map(0).depth).any()
    assert survey.interesting(200) is None


def test_readings_from_different_grids_can_be_asked_of_one_survey():
    scene = Ramp(*BIG)
    survey = _survey(scene, range(500, 4600, 200))
    shapes = [survey.map(level).depth.shape for level in range(LEVELS)]
    assert shapes == [(9, 16), (18, 32), (36, 64)]


# -- one pass ----------------------------------------------------------------


def test_a_sweep_stops_where_it_said_it_would_and_only_goes_one_way():
    sweep = Sweep(start=100, step=50, samples=4)
    seen, moves = [], []
    while not sweep.done:
        seen.append(sweep.position)
        move = sweep.took_one()
        if move is not None:
            moves.append(move)
    assert seen == [100, 150, 200, 250]
    assert moves == [50, 50, 50]
    assert sweep.taken == 4 and not sweep.stopped


def test_a_sweep_that_runs_into_the_stop_ends_there():
    sweep = Sweep(start=0, step=10, samples=100)
    sweep.took_one()
    sweep.blocked()
    assert sweep.done and sweep.stopped


# -- drawing it --------------------------------------------------------------


def test_the_picture_is_the_size_asked_for_and_the_numbers_survive_the_greyscale():
    depth_map = _survey(Ramp(*BIG, blank=0, noise=4.0), range(500, 4600, 200)).map(
        LEVELS - 1
    )
    drawn = colourise(depth_map, 640, 360)
    assert (drawn.width(), drawn.height()) == (640, 360)

    steps = as_steps_image(depth_map)
    assert steps.format() == QImage.Format.Format_Grayscale16
    assert (steps.width(), steps.height()) == depth_map.depth.shape[::-1]
    values = np.frombuffer(steps.constBits(), np.uint16).reshape(
        steps.height(), steps.bytesPerLine() // 2
    )[:, : steps.width()]
    known = depth_map.known
    assert not values[~known].any(), "an unanswered zone was given a depth"
    assert values[known].min() >= 1
    near, far = depth_map.range
    back = near + (values[known].astype(float) - 1) * (far - near) / 65534.0
    assert np.allclose(back, depth_map.depth[known], atol=0.1)


# -- the whole thing, through the worker -------------------------------------

WIDE, TALL = 256, 192


class _Lens:
    """A body with stops at both ends of its travel and a scene with depth.

    Everything about it that the sweep has to survive is real: live view runs
    behind the lens, the drive refuses to go past either end and says so, and
    the picture has grain on it.
    """

    live_view_active = True
    exposure_preview = True

    def __init__(
        self,
        travel: int = 6000,
        start: int = 3000,
        lag: int = 2,
        noise: float = 3.0,
        blank: "int | None" = None,
        scene: "tuple[float, float] | None" = None,
        softness: float = 260.0,
        reports_stops: bool = True,
    ) -> None:
        self.travel = travel
        #: Whether the body admits it could not make a drive. A D750 refuses
        #: at the near stop and accepts for ever past infinity, so nothing
        #: here may depend on being told.
        self.reports_stops = reports_stops
        self.position = start
        self.lag = lag
        self.drives: "list[int]" = []
        near, far = scene if scene is not None else (1200.0, 4200.0)
        self.scene = Ramp(
            WIDE, TALL, bands=8, near=near, far=far, depth=softness,
            blank=blank, noise=noise,
        )
        self._history = [start] * (lag + 1)

    # -- the camera's side ---------------------------------------------------

    def drive_focus(self, steps: int) -> bool:
        wanted = self.position + int(steps)
        self.position = min(max(wanted, 0), self.travel)
        self.drives.append(int(steps))
        return wanted == self.position or not self.reports_stops

    def live_view_frame(self) -> LiveViewFrame:
        self._history.append(self.position)
        shown = self._history[-(self.lag + 1)]
        return _frame(self.scene.at(shown))

    def set_zoom_level(self, level: int) -> None:
        pass

    def zoom_level(self) -> int:
        return 0

    def autofocus(self) -> bool:
        return True

    def stop_live_view(self) -> None:
        pass

    def set_setting(self, name, value) -> None:
        pass

    def settings(self) -> list:
        return []

    def set_exposure_preview(self, enabled: bool) -> None:
        self.exposure_preview = enabled


def _frame(image: QImage) -> LiveViewFrame:
    data = QByteArray()
    sink = QBuffer(data)
    sink.open(QBuffer.OpenModeFlag.WriteOnly)
    image.save(sink, "JPG", 95)
    return LiveViewFrame(
        jpeg=bytes(data.data()), width=WIDE, height=TALL,
        image_width=6016, image_height=4016, crop_width=6016, crop_height=4016,
        crop_center_x=3008, crop_center_y=2008,
        af_width=324, af_height=270, af_x=3008, af_y=2008,
    )


@pytest.fixture
def worker():
    pytest.importorskip("PySide6.QtWidgets")
    from scanny.ui.worker import CameraWorker

    return CameraWorker()


def _ready(worker, lens):
    """A camera, a few frames of grain measured, and nothing sweeping yet."""
    worker._camera = lens
    for _ in range(8):
        worker._grab()
    return lens


def _run(worker, lens, *, samples: int = 12, passes: int = 2, grabs: int = 20000):
    """Watch a whole map being made, one grab at a time."""
    _ready(worker, lens)
    maps: "list[object]" = []
    worker.depthMapReady.connect(maps.append)
    worker.start_depth_map(samples, passes, 6)
    for _ in range(grabs):
        if worker._sweep is None:
            break
        worker._grab()
    assert worker._sweep is None, "the sweep has to finish on its own"
    return maps[-1]


def test_it_parks_against_the_near_stop_before_it_sweeps(worker):
    lens = _Lens(start=3000)
    _run(worker, lens, samples=8, passes=1)
    driven = lens.drives
    parking = driven[: next(i for i, step in enumerate(driven) if step > 0)]
    assert parking, "it swept without parking"
    assert all(step < 0 for step in parking)
    assert sum(parking) <= -3000, "it gave up before it reached the stop"


def test_a_pass_never_turns_round(worker):
    """Which is the whole basis of its step counts meaning anything."""
    lens = _Lens(start=1500)
    _run(worker, lens, samples=8, passes=2)
    forward = False
    for step in lens.drives:
        if step > 0:
            forward = True
        elif forward and step < 0:
            # A reversal is only allowed as the parking that opens a pass, and
            # parking runs all the way back to the stop.
            forward = False
    # Every backward run in the record has to reach the stop.
    runs, current = [], 0
    for step in lens.drives:
        if step < 0:
            current += step
        elif current:
            runs.append(current)
            current = 0
    assert runs, "no parking happened"
    assert all(run <= -1200 for run in runs), f"a pass turned round: {runs}"


def test_the_map_gets_the_scene_the_right_way_round(worker):
    lens = _Lens(start=3000)
    depth_map = _run(worker, lens, samples=14, passes=2)
    assert depth_map is not None and depth_map.coverage > 0.8
    columns = depth_map.depth.shape[1]
    tiling = depth_map.tiling
    want, got = [], []
    for column in range(columns):
        middle = tiling.left + int((column + 0.5) * tiling.tile(depth_map.level)[1])
        answers = depth_map.depth[:, column]
        answers = answers[np.isfinite(answers)]
        if answers.size:
            want.append(lens.scene.best[lens.scene.band(middle)])
            got.append(float(np.median(answers)))
    assert len(got) > columns * 0.7
    order = np.corrcoef(np.argsort(np.argsort(want)), np.argsort(np.argsort(got)))[0, 1]
    assert order > 0.9, f"the depths came out in the wrong order ({order:.2f})"
    assert np.median(np.abs(np.array(got) - np.array(want))) < 260


def test_each_pass_is_finer_than_the_one_before_it(worker):
    lens = _Lens(start=3000)
    steps: "list[int]" = []
    _ready(worker, lens)
    worker.status.connect(
        lambda text: steps.append(int(text.rsplit(" ", 1)[-1].rstrip(".")))
        if "stops in steps of" in text
        else None
    )
    worker.start_depth_map(12, 3, 6)
    for _ in range(20000):
        if worker._sweep is None:
            break
        worker._grab()
    assert len(steps) >= 2, f"only one pass ran: {steps}"
    assert steps == sorted(steps, reverse=True) and steps[0] > steps[-1]


def test_it_sweeps_where_the_picture_answers_and_not_the_rest(worker):
    """The regression that mattered most, and the two halves of it.

    A guessed travel did not make a coarse map of the whole scene -- it made a
    map of however much of the travel the guess reached, and on a macro lens
    that is the first few centimetres, where an ordinary scene has nothing in
    focus at all. Nor is the mechanical travel the answer: most of it is that
    same wash, and a stop spent in it measures nothing.
    """
    lens = _Lens(travel=24000, start=9000, scene=(17000.0, 21000.0), softness=900.0)
    _run(worker, lens, samples=16, passes=1)
    positions = worker._survey.positions
    assert max(positions) > 21000, (
        f"the sweep stopped at {max(positions)}, short of the scene at 21000"
    )
    assert min(positions) > 6000, (
        f"the sweep started at {min(positions)}, in the wash in front of the lens"
    )
    # And so its stops are worth more than dividing the whole travel by them.
    step = positions[1] - positions[0]
    assert step < lens.travel // 16


def test_a_body_that_never_admits_to_a_stop_is_swept_anyway(worker):
    """The failure this was rebuilt for. `MfDrive` can answer STEP_END, and a
    D750 does at the near stop -- and past infinity it answers OK and moves
    nothing, for ever. A sweep that believes the OK reaches infinity a tenth of
    the way through its stops and spends the rest of them, minutes of them,
    driving a lens that cannot move."""
    lens = _Lens(
        travel=12000, start=6000, scene=(7000.0, 10000.0), softness=700.0,
        reports_stops=False,
    )
    depth_map = _run(worker, lens, samples=20, passes=1)
    assert depth_map is not None
    positions = worker._survey.positions
    # It found the range by watching the picture instead, so its stops are in
    # the lens and not miles past the end of it.
    assert max(positions) <= lens.travel + 2000, max(positions)
    answers = depth_map.depth[depth_map.known]
    assert answers.size and 6000 < float(np.median(answers)) < 11000


def test_a_fine_step_is_not_mistaken_for_a_lens_that_has_stopped(worker):
    """The backstop counted stops, and a sweep's step is the range it was given
    divided by the stops asked for -- so the better the range finder got, the
    smaller the step became and the sooner the backstop fired. At 400 stops over
    a 12000-step range the step is 30, five of those is 150 steps of travel, and
    150 steps of travel changes nothing anywhere. Every pass died six stops in.
    """
    lens = _Lens(travel=12000, start=6000, scene=(4000.0, 8000.0), softness=700.0)
    _ready(worker, lens)
    worker._sweep_samples = 120
    worker._sweep_passes = 1
    worker._sweep_minimum = 6
    worker._sweep_pass = 0
    worker._survey = Survey(tiling_for(WIDE, TALL))
    worker._stop_grain = worker._measure_grain(lens)
    worker.depthChanged.emit(True)
    # 120 stops of 100 over the range the picture answers in: fine enough that
    # neighbouring stops look alike, which is exactly the point of a fine pass.
    worker._begin_pass(lens, 0, 100)
    for _ in range(60000):
        if worker._sweep is None:
            break
        worker._grab()
    assert len(worker._survey) > 80, (
        f"the pass was cut off after {len(worker._survey)} of its 120 stops"
    )


def test_a_sweep_that_runs_out_of_lens_stops_instead_of_counting_it_out(worker):
    """The backstop under the measured range: however wrong the range turns out
    to be, a pass must not spend hundreds of stops on a pinned lens."""
    lens = _Lens(travel=6000, start=3000, reports_stops=False)
    _ready(worker, lens)
    # A range that runs a long way past the end of the travel, as a badly
    # measured one would.
    worker._sweep_samples = 200
    worker._sweep_passes = 1
    worker._sweep_minimum = 6
    worker._sweep_pass = 0
    worker._survey = Survey(tiling_for(WIDE, TALL))
    worker._stop_grain = worker._measure_grain(lens)
    worker.depthChanged.emit(True)
    worker._begin_pass(lens, 0, 1000)
    for _ in range(60000):
        if worker._sweep is None:
            break
        worker._grab()
    assert worker._sweep is None, "it never gave up on a lens that could not move"
    assert len(worker._survey) < 40, (
        f"it took {len(worker._survey)} stops against a pinned lens"
    )


def test_a_scene_at_the_far_end_of_a_long_travel_is_found(worker):
    lens = _Lens(travel=24000, start=9000, scene=(17000.0, 21000.0), softness=900.0)
    depth_map = _run(worker, lens, samples=16, passes=3)
    answers = depth_map.depth[depth_map.known]
    assert answers.size, "nothing was placed anywhere"
    assert 15000 < float(np.median(answers)) < 23000, float(np.median(answers))


def test_a_second_run_covers_only_what_the_first_one_found(worker):
    """Which is the answer to both of the lens effects a wide sweep suffers:
    breathing, and defocused highlights swelling into discs."""
    lens = _Lens(travel=24000, start=9000, scene=(17000.0, 21000.0), softness=900.0)
    _run(worker, lens, samples=16, passes=1)
    first = worker._survey.positions
    worker.start_depth_map(16, 1, 6, True)
    for _ in range(20000):
        if worker._sweep is None:
            break
        worker._grab()
    second = worker._survey.positions
    assert max(second) - min(second) < 0.5 * (max(first) - min(first))
    assert min(second) > 0.3 * lens.travel


def test_it_says_why_it_stopped_when_it_cannot_narrow(worker):
    """A map that stops after one pass because its answers are spread over the
    whole travel has converged on nothing, and looks exactly like one that
    has."""
    said: "list[str]" = []
    lens = _ready(worker, _Lens(travel=6000, start=3000))
    worker.status.connect(said.append)
    worker.start_depth_map(8, 4, 6, False)
    for _ in range(20000):
        if worker._sweep is None:
            break
        worker._grab()
    passes = [line for line in said if "stops in steps of" in line]
    if len(passes) < 4:
        assert any("would not fit" in line for line in said), said[-3:]


def test_it_stops_when_something_else_moves_the_view(worker):
    lens = _Lens(start=3000)
    _ready(worker, lens)
    worker.start_depth_map(12, 2, 6)
    for _ in range(60):
        worker._grab()
    assert worker._sweep is not None
    worker.set_zoom(4)
    assert worker._sweep is None


def test_what_was_read_survives_the_sweep_being_stopped(worker):
    """Half a map is still worth looking at, and still worth re-drawing."""
    lens = _Lens(start=3000)
    _ready(worker, lens)
    worker.start_depth_map(12, 1, 6)
    for _ in range(400):
        worker._grab()
    worker.cancel_depth_map()
    maps: "list[object]" = []
    worker.depthMapReady.connect(maps.append)
    worker.set_depth_detail(0)
    assert maps and maps[-1] is not None
    assert maps[-1].depth.shape == worker._survey.tiling.shape(0)


def test_the_map_fills_in_while_it_is_swept(worker):
    lens = _Lens(start=3000)
    _ready(worker, lens)
    maps: "list[object]" = []
    worker.depthMapReady.connect(maps.append)
    worker.start_depth_map(8, 1, 6)
    for _ in range(20000):
        if worker._sweep is None:
            break
        worker._grab()
    # One when it started saying there is nothing, one per stop, one at the end.
    assert maps[0] is None
    assert sum(1 for entry in maps if entry is not None) >= 8


def test_a_blank_band_is_left_blank_rather_than_invented(worker):
    lens = _Lens(start=3000, blank=0)
    depth_map = _run(worker, lens, samples=12, passes=1)
    tiling = depth_map.tiling
    # The blank band is a whole coarse zone wide plus a little; that first
    # coarse zone reaches none of the band's edge, so nothing inside it has
    # anything to read or to inherit.
    assert not depth_map.known[:, : tiling.factor(0)].any()
    band = depth_map.depth.shape[1] // lens.scene.bands
    assert depth_map.known[:, band + 1 :].mean() > 0.8


def test_it_will_not_sweep_without_live_view(worker):
    lens = _Lens()
    lens.live_view_active = False
    worker._camera = lens
    failures: "list[str]" = []
    worker.failed.connect(failures.append)
    worker.start_depth_map(12, 2, 6)
    assert worker._sweep is None
    assert failures and "live view" in failures[0]
