"""Where a shot is recorded: the camera's card, or straight to the computer.

Tethered scanning wants the second, and gets it by putting the body in SDRAM
mode -- Nikon's RecordingMedia property -- so the file comes down the cable
and the card is never involved. That is the default here.

The catch is what makes the rest of this worth testing. SDRAM holds one
picture, and only until the next release of the shutter: a shot taken that way
exists nowhere else until it has been downloaded. So downloading stops being
optional whenever the card is off, and the choice has to be made in a way the
user cannot accidentally undo.
"""

from __future__ import annotations

import os
import struct

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings, QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanny.camera.nikon import NikonCamera  # noqa: E402
from scanny.ptp.codes import Event, FormFlag, Op, Prop, Response  # noqa: E402
from scanny.ui import main_window as mw  # noqa: E402
from scanny.ui.main_window import MainWindow  # noqa: E402
from scanny.ui.worker import CameraWorker  # noqa: E402
from scanny.wpd.device import MtpError  # noqa: E402

SDRAM_HANDLE = 0xFFFF0001

#: Where the window remembers the choice.
_STORED = "capture/save_to_card"


# -- the camera ------------------------------------------------------------


class _DelayDesc:
    """The exposure delay property as a D750 describes it: a range of 0 to 3."""

    form = FormFlag.RANGE
    writable = True
    minimum, maximum, step = 0, 3, 1

    @property
    def allowed_values(self):
        return [0, 1, 2, 3]


class _FakeSession:
    """Enough of a PTP session to watch what the camera asks of the body."""

    def __init__(self, refuse_media: bool = False) -> None:
        # 3 is how the exposure delay property encodes "off": the top of its
        # range. See NikonCamera.SHUTTER_DELAYS.
        self.props: "dict[int, int]" = {int(Prop.NIKON_EXPOSURE_DELAY_MODE): 3}
        self.refuse_media = refuse_media
        self.executed: "list[tuple[int, tuple[int, ...]]]" = []
        #: The shot only exists once the shutter has been released, and its
        #: events are handed over once. Anything read before that is the
        #: queue the capture empties on its way in.
        self.fired = False
        self.reported = False

    def prop_desc(self, code, refresh=True):
        if int(code) != int(Prop.NIKON_EXPOSURE_DELAY_MODE):
            raise MtpError(Op.GET_DEVICE_PROP_DESC, Response.DEVICE_PROP_NOT_SUPPORTED)
        return _DelayDesc()

    def set_prop(self, code, value):
        if code == Prop.NIKON_RECORDING_MEDIA and self.refuse_media:
            raise MtpError(Op.SET_DEVICE_PROP_VALUE, Response.DEVICE_PROP_NOT_SUPPORTED)
        self.props[int(code)] = value

    def get_prop(self, code):
        return self.props.get(int(code), 0)

    def execute(self, opcode, params=()):
        self.executed.append((int(opcode), tuple(params)))
        if opcode == Op.NIKON_INITIATE_CAPTURE_REC_IN_MEDIA:
            self.fired = True
        return []

    def try_execute(self, opcode, params=()):
        self.execute(opcode, params)
        return Response.OK

    def read(self, opcode, params=()):
        self.executed.append((int(opcode), tuple(params)))
        if opcode == Op.NIKON_GET_EVENT and self.fired and not self.reported:
            # One picture, then the shot is over -- so a capture in a test
            # returns rather than waiting out its timeout. Handed over once
            # and then gone, as the camera's own queue does it.
            self.reported = True
            return (
                struct.pack(
                    "<HHIHI",
                    2,
                    int(Event.NIKON_OBJECT_ADDED_IN_SDRAM),
                    SDRAM_HANDLE,
                    int(Event.NIKON_CAPTURE_COMPLETE_RECV_IN_SDRAM),
                    0,
                ),
                [],
            )
        return b"", []


def _camera(refuse_media: bool = False) -> NikonCamera:
    camera = NikonCamera(None)
    camera.session = _FakeSession(refuse_media)
    return camera


def _media(camera: NikonCamera):
    return camera.session.props.get(int(Prop.NIKON_RECORDING_MEDIA))


def test_a_camera_starts_out_shooting_to_the_computer():
    assert _camera().save_to_card is False


def test_the_choice_reaches_the_body_as_its_recording_media():
    camera = _camera()
    assert camera.set_save_to_card(False) is True
    assert _media(camera) == NikonCamera._MEDIA_SDRAM
    assert camera.set_save_to_card(True) is True
    assert _media(camera) == NikonCamera._MEDIA_CARD


def test_a_body_that_will_not_be_told_is_taken_at_its_word():
    """It is still writing to its card, whatever was asked for.

    Reporting otherwise would be worse than useless: the caller reads
    save_to_card to decide whether the SDRAM buffer is theirs to clear.
    """
    camera = _camera(refuse_media=True)
    assert camera.set_save_to_card(False) is False
    assert camera.save_to_card is True


def test_every_shot_says_again_where_it_is_to_go():
    camera = _camera()
    camera.set_save_to_card(False)
    # As if the body had forgotten -- a power cycle, or the card being set on
    # the camera itself.
    camera.session.props[int(Prop.NIKON_RECORDING_MEDIA)] = NikonCamera._MEDIA_CARD
    assert camera.capture() == [SDRAM_HANDLE]
    assert _media(camera) == NikonCamera._MEDIA_SDRAM


def test_the_buffer_is_cleared_by_handle():
    camera = _camera()
    camera.release_from_sdram(SDRAM_HANDLE)
    assert (int(Op.NIKON_DELETE_IMAGE_SDRAM), (SDRAM_HANDLE,)) in camera.session.executed


# -- the worker ------------------------------------------------------------


class _FakeObjectInfo:
    filename = "DSC_0001.NEF"


class _FakeCameraSession:
    def object_info(self, handle):
        return _FakeObjectInfo()


class _FakeCamera:
    shutter_delay = 0

    @staticmethod
    def shot_seconds() -> float:
        return 0.0

    def __init__(self, save_to_card: bool) -> None:
        self.save_to_card = save_to_card
        self.session = _FakeCameraSession()
        self.downloaded: "list[int]" = []
        self.released: "list[int]" = []

    def capture(self, autofocus=False):
        return [SDRAM_HANDLE]

    def download(self, handle):
        self.downloaded.append(handle)
        return "DSC_0001.NEF", b"a picture"

    def release_from_sdram(self, handle):
        self.released.append(handle)

    def settings(self):
        return []


def _worker(tmp_path, save_to_card: bool):
    worker = CameraWorker()
    worker._camera = _FakeCamera(save_to_card)
    worker._save_dir = tmp_path
    return worker, worker._camera


def test_a_shot_held_in_memory_comes_down_whatever_was_asked(tmp_path):
    """Downloading is not the user's to decline when the card is off."""
    worker, camera = _worker(tmp_path, save_to_card=False)
    worker.capture(autofocus=False, download=False)
    assert camera.downloaded == [SDRAM_HANDLE]
    assert [path.name for path in tmp_path.iterdir()] == ["DSC_0001.NEF"]


def test_the_buffer_is_freed_once_the_picture_is_safely_down(tmp_path):
    worker, camera = _worker(tmp_path, save_to_card=False)
    worker.capture(autofocus=False, download=True)
    assert camera.released == [SDRAM_HANDLE]


def test_a_picture_on_the_card_may_be_left_there(tmp_path):
    worker, camera = _worker(tmp_path, save_to_card=True)
    worker.capture(autofocus=False, download=False)
    assert camera.downloaded == []
    assert list(tmp_path.iterdir()) == []


def test_downloading_from_the_card_does_not_delete_anything(tmp_path):
    """DeleteImageSDRAM is for the buffer; the card's copy is the user's."""
    worker, camera = _worker(tmp_path, save_to_card=True)
    worker.capture(autofocus=False, download=True)
    assert camera.downloaded == [SDRAM_HANDLE]
    assert camera.released == []


def test_the_choice_survives_having_no_camera_to_apply_it_to():
    worker = CameraWorker()
    worker.set_save_to_card(True)
    assert worker.save_to_card is True


# -- the window ------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    made = QApplication.instance() or QApplication([])
    # Keep the test's settings out of the real ones.
    QApplication.setOrganizationName("scanny-tests")
    QApplication.setApplicationName("scanny-tests")
    return made


@pytest.fixture
def windows(app, monkeypatch):
    """Builds windows with no worker behind them, and closes them afterwards.

    No worker, because none of this needs a camera: what is under test is
    which box is ticked, what that does to the box below it, and what is
    written down for next time. Going near a real body would also let it
    answer back -- and a window that hears from a camera writes down what it
    heard, in this settings store, which the next test to run reads.
    """
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    # Only this one key is cleared, not the whole store, for the same reason.
    QSettings().remove(_STORED)
    made = []

    def build() -> MainWindow:
        window = MainWindow()
        # Stands in for the thread close() waits on. Never started, so the
        # wait returns at once.
        window._thread = QThread()
        window.show()
        QApplication.processEvents()
        made.append(window)
        return window

    yield build
    for window in made:
        window.close()
    QSettings().remove(_STORED)


@pytest.fixture
def window(windows):
    return windows()


def test_the_card_is_off_to_begin_with(window):
    assert not window.save_to_card.isChecked()


def test_with_the_card_off_downloading_is_not_a_choice(window):
    assert window.download_after_shot.isChecked()
    assert not window.download_after_shot.isEnabled()
    assert window.download_after_shot.toolTip()


def test_turning_the_card_on_hands_the_choice_back(window):
    window.save_to_card.setChecked(True)
    assert window.download_after_shot.isEnabled()
    window.download_after_shot.setChecked(False)
    window.save_to_card.setChecked(False)
    assert window.download_after_shot.isChecked()


def test_the_worker_is_told_where_to_record(window):
    asked = []
    window.requestSaveToCard.connect(asked.append)
    window.save_to_card.setChecked(True)
    window.save_to_card.setChecked(False)
    assert asked == [True, False]


def test_the_camera_has_the_last_word(window):
    """A body that will not be told reports the card back, and the box follows
    that rather than the click which did not take -- downloading is optional
    again, because the picture is on the card whether it is downloaded or not.
    """
    asked = []
    window.requestSaveToCard.connect(asked.append)
    window._on_save_to_card(True)
    assert window.save_to_card.isChecked()
    assert window.download_after_shot.isEnabled()
    assert asked == []


def test_the_choice_is_remembered_between_runs(windows):
    window = windows()
    window.save_to_card.setChecked(True)
    assert QSettings().value(_STORED, False, bool) is True
    assert windows().save_to_card.isChecked()
