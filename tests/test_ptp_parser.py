"""Tests for the PTP dataset encoding, checked against bytes the D750 sent."""

from __future__ import annotations

import pytest

from scanny.ptp.codes import DataType, FormFlag
from scanny.ptp.parser import DeviceInfo, PropDesc, Reader, Writer


def test_string_round_trip():
    encoded = Writer().string("D750").bytes()
    # A count of characters including the terminator, then UCS-2.
    assert encoded[0] == 5
    assert Reader(encoded).string() == "D750"


def test_empty_string_round_trip():
    encoded = Writer().string("").bytes()
    assert encoded == b"\x00"
    assert Reader(encoded).string() == ""


@pytest.mark.parametrize(
    "datatype,value",
    [
        (DataType.INT8, -8),
        (DataType.UINT8, 200),
        (DataType.INT16, -3000),
        (DataType.UINT16, 65000),
        (DataType.INT32, -100000),
        (DataType.UINT32, 4000000000),
        (DataType.INT64, -2**40),
        (DataType.UINT64, 2**63),
        (DataType.STR, "hello"),
        (DataType.AUINT16, [1, 2, 3]),
    ],
)
def test_scalar_and_array_round_trip(datatype, value):
    encoded = Writer().value(datatype, value).bytes()
    assert Reader(encoded).value(datatype) == value


def test_reader_rejects_truncated_data():
    with pytest.raises(ValueError, match="truncated"):
        Reader(b"\x01").uint32()


#: The real reply to GetDevicePropDesc(FNumber) from a D750 with an f/4 lens.
#: Header and the first ten choices are the bytes the camera actually sent;
#: the remaining six are the rest of that lens's aperture scale.
_FNUMBER_APERTURES = [
    400, 450, 500, 560, 630, 710, 800, 900, 1000, 1100,
    1300, 1400, 1600, 1800, 2000, 2200,
]
_FNUMBER_DESC = (
    bytes.fromhex("0750" "0400" "01" "9001" "2003" "02" "1000")
    + b"".join(value.to_bytes(2, "little") for value in _FNUMBER_APERTURES)
)


def test_prop_desc_parses_real_aperture_descriptor():
    desc = PropDesc.parse(_FNUMBER_DESC)
    assert desc.code == 0x5007
    assert desc.datatype == DataType.UINT16
    assert desc.writable is True
    assert desc.factory_default == 400  # f/4.0, in hundredths of a stop
    assert desc.current == 800  # f/8.0
    assert desc.form == FormFlag.ENUMERATION
    assert len(desc.enumeration) == 16
    assert desc.enumeration[:4] == [400, 450, 500, 560]
    assert desc.allowed_values == desc.enumeration


def test_prop_desc_range_form_expands_to_allowed_values():
    # A UINT8 property ranging 0..7 in steps of 1, as live-view zoom reports.
    raw = bytes.fromhex("a3d1" "0200" "01" "00" "03" "01" "00" "07" "01")
    desc = PropDesc.parse(raw)
    assert desc.form == FormFlag.RANGE
    assert (desc.minimum, desc.maximum, desc.step) == (0, 7, 1)
    assert desc.allowed_values == [0, 1, 2, 3, 4, 5, 6, 7]


def test_prop_desc_without_form_field():
    raw = bytes.fromhex("a4d1" "0600" "00" "00000000" "00000000")
    desc = PropDesc.parse(raw)
    assert desc.form == FormFlag.NONE
    assert desc.allowed_values == []


def test_device_info_round_trip():
    payload = (
        Writer()
        .uint16(100)
        .uint32(6)
        .uint16(1)
        .string("Nikon")
        .uint16(0)
        .value(DataType.AUINT16, [0x1001, 0x9201])
        .value(DataType.AUINT16, [0x4002])
        .value(DataType.AUINT16, [0x5007, 0x500D])
        .value(DataType.AUINT16, [0x3801])
        .value(DataType.AUINT16, [0x3800])
        .string("Nikon Corporation")
        .string("D750")
        .string("V1.10")
        .string("00000006140780")
        .bytes()
    )
    info = DeviceInfo.parse(payload)
    assert info.model == "D750"
    assert info.device_version == "V1.10"
    assert info.supports(0x9201)
    assert not info.supports(0x9999)
    assert info.has_property(0x500D)
