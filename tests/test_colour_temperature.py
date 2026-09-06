"""Setting the white balance by colour temperature.

Choosing "Colour temperature" for white balance only names the method; the
number itself lives in a separate Nikon property, and until it can be typed in
the choice does nothing at all. The body describes that property as a range of
kelvin rather than a list of settings, so it arrives as a box to type into
instead of the combo every other exposure setting gets.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox  # noqa: E402

from scanny.camera.nikon import NikonCamera, Setting, Span  # noqa: E402
from scanny.ptp.codes import DataType, FormFlag, Op, Prop, Response  # noqa: E402
from scanny.ptp.parser import PropDesc  # noqa: E402
from scanny.ui import main_window as mw  # noqa: E402
from scanny.wpd.device import MtpError  # noqa: E402

#: The white balance value that means "use the temperature I typed in".
_BY_TEMPERATURE = 0x8012
_DAYLIGHT = 4

_TEMPERATURE = int(Prop.NIKON_WHITE_BALANCE_COLOUR_TEMP)
_BALANCE = int(Prop.WHITE_BALANCE)

#: What the row is called in the panel.
_ROW = "Colour temp."


# -- the camera ------------------------------------------------------------


class _FakeSession:
    """A body that has both white balance properties and nothing else.

    It advertises only the standard one, as a D750 does: that body lists its
    twenty-two standard properties in DeviceInfo and not one of the vendor
    codes it nevertheless answers for.
    """

    def __init__(self, white_balance: int = _BY_TEMPERATURE) -> None:
        self.props = {_BALANCE: white_balance, _TEMPERATURE: 5000}

    def device_info(self):
        return SimpleNamespace(device_properties_supported=[_BALANCE])

    def prop_desc(self, code, refresh=True):
        code = int(code)
        if code == _BALANCE:
            return PropDesc(
                code=code,
                datatype=DataType.UINT16,
                writable=True,
                factory_default=2,
                current=self.props[code],
                form=FormFlag.ENUMERATION,
                enumeration=[2, _DAYLIGHT, _BY_TEMPERATURE],
            )
        if code == _TEMPERATURE:
            # A D750's range: 2500 K to 10000 K in steps of ten.
            return PropDesc(
                code=code,
                datatype=DataType.UINT32,
                writable=True,
                factory_default=5000,
                current=self.props[code],
                form=FormFlag.RANGE,
                minimum=2500,
                maximum=10000,
                step=10,
            )
        raise MtpError(Op.GET_DEVICE_PROP_DESC, Response.DEVICE_PROP_NOT_SUPPORTED)

    def set_prop(self, code, value):
        self.props[int(code)] = value


def _camera(white_balance: int = _BY_TEMPERATURE) -> NikonCamera:
    camera = NikonCamera(None)
    camera.session = _FakeSession(white_balance)
    return camera


def _row(camera: NikonCamera, name: str = _ROW) -> Setting:
    found = {s.name: s for s in camera.settings()}
    assert name in found, f"{name} was not offered at all"
    return found[name]


def test_the_temperature_is_offered_alongside_white_balance():
    """Even though the body never says it has the property.

    Asking for a vendor property the body has not advertised is how the rest
    of this file talks to a D750: it lists none of them, and answers for all
    of them. Trusting the list here leaves the row out of the panel entirely,
    which is the whole of what was wrong before.
    """
    assert [s.name for s in _camera().settings()] == ["White balance", _ROW]


def test_it_arrives_as_a_range_to_type_a_number_into():
    temperature = _row(_camera())
    assert temperature.span == Span(2500, 10000, 10, " K")
    # A range of 751 values is not a list to pick from, so none is offered.
    assert temperature.choices == ()
    assert temperature.label == "5000 K"


def test_a_temperature_typed_in_reaches_the_body():
    camera = _camera()
    camera.set_setting(_ROW, 6500)
    assert camera.session.props[_TEMPERATURE] == 6500


def test_it_is_closed_while_white_balance_is_set_to_something_else():
    """A temperature set under Daylight changes nothing anyone can see."""
    temperature = _row(_camera(white_balance=_DAYLIGHT))
    assert temperature.writable is False
    assert "white balance" in temperature.note.lower()


def test_it_opens_as_soon_as_white_balance_asks_for_it():
    camera = _camera(white_balance=_DAYLIGHT)
    camera.set_setting("White balance", _BY_TEMPERATURE)
    temperature = _row(camera)
    assert temperature.writable is True
    assert temperature.note == ""


# -- the panel -------------------------------------------------------------


_OFFERED = [
    Setting(_BALANCE, "White balance", _BY_TEMPERATURE, "Colour temperature", True,
            ((_DAYLIGHT, "Daylight"), (_BY_TEMPERATURE, "Colour temperature"))),
    Setting(_TEMPERATURE, _ROW, 5000, "5000 K", True, (),
            span=Span(2500, 10000, 10, " K")),
]


@pytest.fixture(scope="module")
def app():
    made = QApplication.instance() or QApplication([])
    QApplication.setOrganizationName("scanny-tests")
    QApplication.setApplicationName("scanny-tests")
    QSettings().clear()
    yield made
    QSettings().clear()


@pytest.fixture
def window(app, monkeypatch):
    # No worker: nothing here needs to go near a camera.
    monkeypatch.setattr(mw.MainWindow, "_start_worker", lambda self: None)
    made = mw.MainWindow()
    made.show()
    made._on_settings(_OFFERED)
    QApplication.processEvents()
    yield made
    made.hide()
    made.deleteLater()
    QApplication.processEvents()


def _box(window) -> QSpinBox:
    widget = window._setting_widgets[_ROW]
    assert isinstance(widget, QSpinBox), "the temperature has nowhere to be typed"
    return widget


def test_the_temperature_gets_a_box_to_type_into(window):
    box = _box(window)
    assert (box.minimum(), box.maximum()) == (2500, 10000)
    assert box.singleStep() == 10
    assert box.value() == 5000
    assert box.suffix() == " K"


def test_the_settings_that_are_choices_still_get_a_list(window):
    assert isinstance(window._setting_widgets["White balance"], QComboBox)


def test_a_number_typed_in_is_sent_to_the_camera(window):
    sent = []
    window.requestSetting.connect(lambda name, value: sent.append((name, value)))
    _box(window).setValue(6500)
    assert sent == [(_ROW, 6500)]


def test_what_the_camera_reports_does_not_come_back_as_a_change(window):
    """The camera's own answer refills the box, and must not read as typing."""
    sent = []
    window.requestSetting.connect(lambda name, value: sent.append((name, value)))
    window._on_settings(
        [_OFFERED[0], Setting(_TEMPERATURE, _ROW, 7100, "7100 K", True, (),
                              span=Span(2500, 10000, 10, " K"))]
    )
    assert _box(window).value() == 7100
    assert sent == []


def test_a_closed_box_says_why_it_is_closed(window):
    note = "Set white balance to Colour temperature to use this"
    window._on_settings(
        [_OFFERED[0], Setting(_TEMPERATURE, _ROW, 5000, "5000 K", False, (),
                              span=Span(2500, 10000, 10, " K"), note=note)]
    )
    box = _box(window)
    assert not box.isEnabled()
    assert box.toolTip() == note
    # The box itself is out of the mouse's reach while it is disabled, so the
    # reason has to be readable from the row's label too.
    assert window.exposure_form.labelForField(box).toolTip() == note
