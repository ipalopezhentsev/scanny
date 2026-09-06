"""A shot is not over when the shutter closes.

With long exposure noise reduction on -- and Nikon applies it to anything of a
second or longer -- the body follows the exposure with a second one of the same
length, shutter closed, and subtracts that dark frame from the picture. A 15
second exposure is a 30 second shot, and the finished picture does not exist
until the end of it.

Two failures came out of that, one behind the other.

Waiting less than the shot takes broke every frame after the first, in a way
that took some finding: the wait ran out, the camera's events for that shot
arrived afterwards and sat in its queue, and the next shot read them the
instant it fired. Two events saying a picture was ready, so the buffer was
downloaded -- and the buffer still held the previous picture. Each shot handed
back the one before it, and the live view had been telling the truth all along.

Underneath that was the reason a wait could run out at all. The camera sends
two events, not one, and not together: CaptureComplete says the exposure is
over, ObjectAdded says the picture exists and can be fetched. After a long
exposure the first arrives while the image is still being written into the
buffer. Stopping at it means a finished exposure with nothing to show for
itself -- which is what the shot after the first fix did.

So: wait as long as the shot really takes, wait for the picture and not merely
for the exposure, and never believe an event that was already waiting when the
shutter was released.
"""

from __future__ import annotations

import struct

import pytest

from scanny.camera.nikon import CameraError, NikonCamera
from scanny.ptp.codes import Event, FormFlag, Op, Prop, Response
from scanny.wpd.device import MtpError

#: PTP ExposureTime counts in units of 0.1 ms.
SECOND = 10_000
OLD_HANDLE, NEW_HANDLE, JPEG_HANDLE = 0x100, 0x200, 0x201

SHOT_OVER = (Event.CAPTURE_COMPLETE, 0)
PICTURE = (Event.OBJECT_ADDED, NEW_HANDLE)
JPEG = (Event.OBJECT_ADDED, JPEG_HANDLE)
#: Both events of a shot, arriving together, as a short exposure's do.
SHOT = (PICTURE, SHOT_OVER)
#: The same for the shot before this one, still sitting in the queue.
LAST_SHOT = ((Event.OBJECT_ADDED, OLD_HANDLE), SHOT_OVER)


def _packed(events) -> bytes:
    """Events as the camera hands them over: a count, then code and parameter."""
    return struct.pack("<H", len(events)) + b"".join(
        struct.pack("<HI", int(code), int(param)) for code, param in events
    )


class _DelayDesc:
    """The exposure delay property as a D750 describes it: a range of 0 to 3."""

    form = FormFlag.RANGE
    writable = True
    minimum, maximum, step = 0, 3, 1

    @property
    def allowed_values(self):
        return [0, 1, 2, 3]


class _ShootingSession:
    """A body whose event queue behaves like the real one.

    Events are handed over once and then gone. `queued` is what is already in
    there when the shot starts -- the state that made every picture come back
    one late. `schedule` is what this shot produces, as {poll: events}, so a
    test can put the completion and the picture on different polls the way a
    long exposure really does.
    """

    def __init__(
        self,
        *,
        queued=(),
        schedule=None,
        exposure: int = 0,
        noise_reduction: bool = False,
    ) -> None:
        self._queue = [tuple(event) for event in queued]
        self._schedule = {
            int(poll): [tuple(event) for event in events]
            for poll, events in (schedule or {}).items()
        }
        self.props = {
            int(Prop.EXPOSURE_TIME): exposure,
            int(Prop.NIKON_LONG_EXPOSURE_NOISE_REDUCTION): int(noise_reduction),
            # 3 is how the exposure delay property encodes "off": the top of
            # its range. See NikonCamera.SHUTTER_DELAYS. A body with a delay
            # set is test_shutter_delay.py's business, not this file's.
            int(Prop.NIKON_EXPOSURE_DELAY_MODE): 3,
        }
        self.fired = False
        self.terminated = False
        self.queue_when_fired: "list | None" = None
        self.polls_after_firing = 0

    def prop_desc(self, code, refresh=True):
        if int(code) != int(Prop.NIKON_EXPOSURE_DELAY_MODE):
            raise MtpError(Op.GET_DEVICE_PROP_DESC, Response.DEVICE_PROP_NOT_SUPPORTED)
        return _DelayDesc()

    def set_prop(self, code, value):
        self.props[int(code)] = value

    def get_prop(self, code):
        return self.props.get(int(code), 0)

    def execute(self, opcode, params=()):
        if opcode == Op.NIKON_INITIATE_CAPTURE_REC_IN_MEDIA:
            self.queue_when_fired = list(self._queue)
            self.fired = True
        elif opcode == Op.NIKON_TERMINATE_CAPTURE:
            self.terminated = True
        return []

    def try_execute(self, opcode, params=()):
        self.execute(opcode, params)
        return Response.OK

    def read(self, opcode, params=()):
        if opcode != Op.NIKON_GET_EVENT:
            return b"", []
        if self.fired:
            self.polls_after_firing += 1
            self._queue.extend(self._schedule.pop(self.polls_after_firing, ()))
        events, self._queue = self._queue, []
        return _packed(events), []


def _camera(**kwargs) -> NikonCamera:
    camera = NikonCamera(None)
    camera.session = _ShootingSession(**kwargs)
    return camera


@pytest.fixture
def brief_grace(monkeypatch):
    """The wait for the second of the two events, shortened for the tests."""
    monkeypatch.setattr(NikonCamera, "_EVENT_GRACE", 0.05)


# -- waiting for the picture, not merely for the exposure ------------------


def test_a_finished_exposure_is_not_a_finished_picture():
    """The bug this file is named for. The camera says the exposure is over
    while the image is still going into the buffer; the picture follows."""
    camera = _camera(schedule={1: [SHOT_OVER], 3: [PICTURE]})
    assert camera.capture() == [NEW_HANDLE]


def test_the_picture_is_waited_for_after_the_exposure_ends():
    camera = _camera(schedule={1: [SHOT_OVER], 4: [PICTURE]})
    camera.capture()
    assert camera.session.polls_after_firing >= 4


def test_the_exposure_is_waited_for_after_the_picture_arrives():
    """The other order, which some bodies use. Both events, either way round."""
    camera = _camera(schedule={1: [PICTURE], 4: [SHOT_OVER]})
    assert camera.capture() == [NEW_HANDLE]
    assert camera.session.polls_after_firing >= 4


def test_a_picture_without_a_completion_is_not_waited_out(monkeypatch):
    """The picture is what the wait was for.

    Not every body sends CaptureComplete for every shot -- a short exposure
    held in SDRAM with live view running is the case this was found on. Once
    the handle is in hand the missing event would confirm nothing the picture
    does not already prove, and holding the shot open for one that is never
    coming is seconds of silence between the shutter and the download.
    """
    monkeypatch.setattr(NikonCamera, "_COMPANION_GRACE", 0.05)
    camera = _camera(schedule={1: [PICTURE]})
    assert camera.capture() == [NEW_HANDLE]
    # A handful of polls for the companion file, not the hundred that waiting
    # out the courtesy wait for the completion would take.
    assert camera.session.polls_after_firing < 10


def test_the_jpeg_beside_a_raw_is_caught_without_a_completion():
    """What the short wait after the picture is actually for: the second file
    of a raw-plus-jpeg shot, a poll behind the first."""
    camera = _camera(schedule={1: [PICTURE], 2: [JPEG]})
    assert camera.capture() == [NEW_HANDLE, JPEG_HANDLE]


def test_an_ordinary_shot_waits_for_nothing_extra():
    """Both events together, as a short exposure sends them: no grace spent."""
    camera = _camera(schedule={1: SHOT})
    assert camera.capture() == [NEW_HANDLE]
    # The poll that brought them, and the one last look after it.
    assert camera.session.polls_after_firing == 2


def test_a_raw_and_its_jpeg_a_poll_apart_are_both_kept():
    camera = _camera(schedule={1: [PICTURE, SHOT_OVER], 2: [JPEG]})
    assert camera.capture() == [NEW_HANDLE, JPEG_HANDLE]


def test_an_exposure_that_never_yields_a_picture_is_reported():
    """With the card off there is no second copy: saying "no new file" and
    moving on would be losing the picture quietly."""
    camera = _camera(schedule={1: [SHOT_OVER]})
    with pytest.raises(CameraError, match="never offered the picture"):
        camera.capture(timeout=0.1)


def test_the_picture_is_waited_for_as_long_as_the_shot_is_worth(brief_grace):
    """The camera can call the exposure over with the dark frame still to
    come, so the picture is not held to the courtesy wait: it has the whole
    budget, which was sized to the shot."""
    camera = _camera(schedule={1: [SHOT_OVER], 30: [PICTURE]})
    assert camera.capture(timeout=5.0) == [NEW_HANDLE]


def test_with_the_card_on_the_handle_is_only_a_convenience(brief_grace):
    """The file is on the card whether or not a handle turns up, so a
    completion on its own is enough to go on. Nothing is lost."""
    camera = _camera(schedule={1: [SHOT_OVER]})
    camera.set_save_to_card(True)
    assert camera.capture() == []


# -- not answering with the picture before ---------------------------------


def test_the_queue_is_emptied_before_the_shutter_is_released():
    camera = _camera(queued=LAST_SHOT, schedule={1: SHOT})
    camera.capture()
    assert camera.session.queue_when_fired == []


def test_a_shot_does_not_answer_with_the_one_before_it():
    """The whole bug, in one line: the handle has to be this shot's."""
    camera = _camera(queued=LAST_SHOT, schedule={1: SHOT})
    assert camera.capture() == [NEW_HANDLE]


def test_a_stale_completion_does_not_end_the_shot_early():
    """The events from the last shot said a picture was ready. It was not
    this one, so the wait has to carry on regardless."""
    camera = _camera(queued=LAST_SHOT, schedule={4: SHOT})
    assert camera.capture() == [NEW_HANDLE]
    assert camera.session.polls_after_firing >= 4


# -- waiting as long as the shot takes -------------------------------------


def test_a_short_exposure_costs_nothing_to_wait_for():
    assert _camera(exposure=SECOND // 125).shot_seconds() == pytest.approx(0.008)


def test_noise_reduction_doubles_a_long_exposure():
    camera = _camera(exposure=15 * SECOND, noise_reduction=True)
    assert camera.shot_seconds() == pytest.approx(30.0)


def test_noise_reduction_is_not_counted_where_the_camera_would_not_use_it():
    """Nikon applies it from a second up; a half second shot is a half second."""
    camera = _camera(exposure=SECOND // 2, noise_reduction=True)
    assert camera.shot_seconds() == pytest.approx(0.5)


def test_the_dark_frame_only_counts_when_it_is_switched_on():
    camera = _camera(exposure=15 * SECOND, noise_reduction=False)
    assert camera.shot_seconds() == pytest.approx(15.0)


def test_the_mirror_up_delay_is_part_of_the_shot():
    camera = _camera(exposure=15 * SECOND, noise_reduction=True)
    camera.set_shutter_delay(3)
    assert camera.shot_seconds() == pytest.approx(33.0)


def test_bulb_is_not_guessed_at():
    """Nobody knows how long it will be open, so nothing is added for it."""
    assert _camera(exposure=0xFFFFFFFF).shot_seconds() == 0.0


def test_the_wait_is_the_shot_plus_the_slack_allowed(monkeypatch):
    camera = _camera(exposure=15 * SECOND, noise_reduction=True)
    waits = []
    monkeypatch.setattr(
        camera, "_collect_capture_handles", lambda wait: waits.append(wait) or []
    )
    camera.capture(timeout=20.0)
    assert waits == [pytest.approx(50.0)]


# -- giving up ------------------------------------------------------------


def test_a_shot_that_never_arrives_is_reported_rather_than_passed_over():
    camera = _camera()
    with pytest.raises(CameraError, match="did not finish"):
        camera.capture(timeout=0.05)
    assert camera.session.terminated


def test_a_capture_that_did_arrive_is_left_alone():
    """TerminateCapture is for a shot that never happened. Sending it after
    one that did is how a body ends up refusing the next one."""
    camera = _camera(schedule={1: SHOT})
    camera.capture()
    assert not camera.session.terminated
