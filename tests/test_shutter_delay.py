"""The mirror-up delay: Nikon's exposure delay mode, driven for one shot.

The point of the tests below is not that the property is written -- it is
*when* it is written. The delay has to be on the camera before the shutter is
released and off it again afterwards, whatever the shot did, because a body
left with exposure delay mode on is a body that may refuse to start live view
next time.
"""

from __future__ import annotations

import struct

import pytest

from scanny.camera.nikon import CameraError, NikonCamera
from scanny.ptp.codes import Event, FormFlag, Op, Prop, Response
from scanny.wpd.device import MtpError

_DELAY = Prop.NIKON_EXPOSURE_DELAY_MODE


class FakeDesc:
    def __init__(self, values, writable=True):
        self.form = FormFlag.ENUMERATION
        self.enumeration = list(values)
        self.writable = writable

    @property
    def allowed_values(self):
        return list(self.enumeration)


class FakeSession:
    """Just enough of PtpSession to take a picture."""

    def __init__(self, *, delays=(0, 1, 2, 3), writable=True, refuse_delay=False):
        self.props = {_DELAY: 0}
        self.writes: "list[tuple[int, object]]" = []
        self.operations: "list[int]" = []
        self.desc = FakeDesc(delays, writable) if delays else None
        self.refuse_delay = refuse_delay
        self.shutter_response: "int | None" = None
        self.events = struct.pack(
            "<HHIHI", 2, Event.OBJECT_ADDED, 0x42, Event.CAPTURE_COMPLETE, 0
        )

    def prop_desc(self, code, refresh=True):
        if code == _DELAY and self.desc is None:
            raise MtpError(Op.GET_DEVICE_PROP_DESC, Response.DEVICE_PROP_NOT_SUPPORTED)
        return self.desc

    def get_prop(self, code):
        if code not in self.props:
            raise MtpError(Op.GET_DEVICE_PROP_VALUE, Response.DEVICE_PROP_NOT_SUPPORTED)
        return self.props[code]

    def set_prop(self, code, value):
        if code == _DELAY and self.refuse_delay:
            raise MtpError(Op.SET_DEVICE_PROP_VALUE, Response.NIKON_SET_PROPERTY_NOT_SUPPORTED)
        self.props[code] = value
        self.writes.append((code, value))

    def execute(self, opcode, params=()):
        self.operations.append(opcode)
        if opcode == Op.NIKON_INITIATE_CAPTURE_REC_IN_MEDIA and self.shutter_response:
            raise MtpError(opcode, self.shutter_response)
        return []

    def read(self, opcode, params=()):
        if opcode == Op.NIKON_GET_EVENT:
            return self.events, []
        return b"", []

    def try_execute(self, opcode, params=()):
        self.operations.append(opcode)
        return Response.OK


@pytest.fixture
def camera():
    made = NikonCamera(transport=None)
    made.session = FakeSession()
    return made


def delays_written(camera) -> "list[int]":
    return [value for code, value in camera.session.writes if code == _DELAY]


def test_a_body_already_off_is_not_written_to(camera):
    camera.capture()
    assert delays_written(camera) == []


def test_the_delay_is_set_for_the_shot_and_taken_off_after(camera):
    camera.set_shutter_delay(2)
    camera.capture()
    assert delays_written(camera) == [2, 0]


def test_the_cameras_own_delay_is_taken_off_when_none_was_asked_for(camera):
    # This D750 arrived with three seconds set on it. Off has to mean off.
    camera.session.props[_DELAY] = 3
    camera.capture()
    assert delays_written(camera) == [0, 3]


def test_a_camera_that_will_not_let_go_of_its_delay_still_takes_the_picture(camera):
    camera.session.props[_DELAY] = 3
    camera.session.refuse_delay = True
    camera.capture()
    assert Op.NIKON_INITIATE_CAPTURE_REC_IN_MEDIA in camera.session.operations


def test_the_shot_is_released_while_the_delay_is_on(camera):
    camera.set_shutter_delay(3)
    released = []
    original = camera.session.execute

    def watching(opcode, params=()):
        if opcode == Op.NIKON_INITIATE_CAPTURE_REC_IN_MEDIA:
            released.append(camera.session.props[_DELAY])
        return original(opcode, params)

    camera.session.execute = watching
    camera.capture()
    assert released == [3]


def test_the_body_gets_back_the_setting_it_had(camera):
    camera.session.props[_DELAY] = 1
    camera.set_shutter_delay(3)
    camera.capture()
    assert delays_written(camera) == [3, 1]
    assert camera.session.props[_DELAY] == 1


def test_a_body_already_set_the_way_it_is_wanted_is_left_alone(camera):
    camera.session.props[_DELAY] = 2
    camera.set_shutter_delay(2)
    camera.capture()
    assert delays_written(camera) == []


def test_a_body_with_no_delay_at_all_says_so_rather_than_shooting(camera):
    camera.set_shutter_delay(2)
    camera.session.props.pop(_DELAY)
    with pytest.raises(CameraError, match="no exposure delay mode"):
        camera.capture()
    assert Op.NIKON_INITIATE_CAPTURE_REC_IN_MEDIA not in camera.session.operations


def test_a_body_with_no_delay_at_all_still_shoots_without_one(camera):
    camera.session.props.pop(_DELAY)
    camera.capture()
    assert Op.NIKON_INITIATE_CAPTURE_REC_IN_MEDIA in camera.session.operations


def test_what_the_body_is_set_to_is_readable(camera):
    camera.session.props[_DELAY] = 3
    assert camera.shutter_delay_on_body() == 3


def test_a_body_with_no_such_setting_reports_no_delay_of_its_own(camera):
    camera.session.props.pop(_DELAY)
    assert camera.shutter_delay_on_body() is None


def test_a_shutter_that_would_not_fire_still_gives_the_delay_back(camera):
    camera.set_shutter_delay(2)
    camera.session.shutter_response = Response.NIKON_OUT_OF_FOCUS
    with pytest.raises(CameraError):
        camera.capture()
    assert delays_written(camera) == [2, 0]


def test_a_camera_that_refuses_the_delay_is_not_shot_with(camera):
    camera.set_shutter_delay(2)
    camera.session.refuse_delay = True
    with pytest.raises(CameraError, match="shutter delay"):
        camera.capture()
    assert Op.NIKON_INITIATE_CAPTURE_REC_IN_MEDIA not in camera.session.operations


def test_the_wait_is_added_to_the_time_the_shot_is_given(camera):
    given = []
    camera._collect_capture_handles = lambda timeout: given.append(timeout) or []
    camera.set_shutter_delay(3)
    camera.capture(timeout=20.0)
    assert given == [23.0]


def test_the_delays_on_offer_come_from_the_body(camera):
    camera.session.desc = FakeDesc((0, 1))
    assert camera.shutter_delay_choices() == (0, 1)


def test_a_body_with_no_such_setting_offers_nothing(camera):
    camera.session.desc = None
    assert camera.shutter_delay_choices() == ()


def test_a_setting_the_body_will_not_take_is_not_offered(camera):
    camera.session.desc = FakeDesc((0, 1, 2, 3), writable=False)
    assert camera.shutter_delay_choices() == ()


def test_a_body_that_will_not_enumerate_still_gets_nikons_own_values(camera):
    camera.session.desc = FakeDesc(())
    assert camera.shutter_delay_choices() == NikonCamera.SHUTTER_DELAYS


def test_a_delay_cannot_be_negative(camera):
    camera.set_shutter_delay(-2)
    assert camera.shutter_delay == 0
