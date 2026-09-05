"""Formatting of raw PTP property values into the labels photographers expect.

PTP encodes exposure settings in fixed-point integers -- aperture in hundredths
of an f-stop, shutter in tenths of a millisecond, exposure compensation in
thousandths of an EV. The shutter values a camera reports are rounded, so
``2`` means 1/4000s rather than a literal 1/5000s; the formatter snaps to the
nearest standard speed so the UI shows the number engraved on the dial.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "format_shutter",
    "format_aperture",
    "format_iso",
    "format_exposure_bias",
    "format_white_balance",
    "format_program_mode",
    "format_focus_mode",
    "format_capture_mode",
    "describe_lv_prohibit",
]

_BULB = 0xFFFFFFFF

#: Standard shutter speeds in seconds, from 1/8000 to 30s.
#:
#: 1/6400 and 1/5000 are deliberately absent. PTP reports exposure time in
#: whole tenths of a millisecond, so everything from 1/8000 to 1/4000 lands on
#: the integers 1, 2 and 3 -- there is not enough resolution to tell those two
#: apart from their neighbours. Leaving them out makes the common case exact: a
#: body whose fastest speed is 1/4000 reports 2, and 2 reads back as 1/4000.
_STANDARD_SHUTTER: "tuple[float, ...]" = (
    1 / 8000, 1 / 4000, 1 / 3200, 1 / 2500, 1 / 2000,
    1 / 1600, 1 / 1250, 1 / 1000, 1 / 800, 1 / 640, 1 / 500, 1 / 400,
    1 / 320, 1 / 250, 1 / 200, 1 / 160, 1 / 125, 1 / 100, 1 / 80, 1 / 60,
    1 / 50, 1 / 40, 1 / 30, 1 / 25, 1 / 20, 1 / 15, 1 / 13, 1 / 10, 1 / 8,
    1 / 6, 1 / 5, 1 / 4, 1 / 3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0,
    2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 15.0, 20.0, 25.0, 30.0,
)


def _snap_shutter(seconds: float) -> float:
    """The standard shutter speed closest to a reported duration.

    Compared in log space, so that being a third of a stop out counts the same
    whether the speed is fast or slow.
    """
    return min(_STANDARD_SHUTTER, key=lambda s: abs(math.log(s / seconds)))


def format_shutter(value: Any) -> str:
    """PTP ExposureTime (units of 0.1 ms) as a shutter speed."""
    try:
        raw = int(value)
    except (TypeError, ValueError):
        return str(value)
    if raw == _BULB:
        return "Bulb"
    if raw <= 0:
        return "--"
    seconds = _snap_shutter(raw / 10_000)
    if seconds >= 1:
        return f"{seconds:g}\""
    return f"1/{round(1 / seconds)}"


def format_aperture(value: Any) -> str:
    """PTP FNumber (hundredths of an f-stop) as f/N."""
    try:
        raw = int(value)
    except (TypeError, ValueError):
        return str(value)
    stop = raw / 100
    return f"f/{stop:.1f}".rstrip("0").rstrip(".") if stop % 1 else f"f/{stop:.0f}"


def format_iso(value: Any) -> str:
    try:
        return f"ISO {int(value)}"
    except (TypeError, ValueError):
        return str(value)


def format_exposure_bias(value: Any) -> str:
    """PTP ExposureBiasCompensation (thousandths of an EV)."""
    try:
        ev = int(value) / 1000
    except (TypeError, ValueError):
        return str(value)
    if abs(ev) < 0.005:
        return "0 EV"
    return f"{ev:+.1f} EV"


_WHITE_BALANCE = {
    1: "Manual",
    2: "Auto",
    3: "One-push auto",
    4: "Daylight",
    5: "Fluorescent",
    6: "Incandescent",
    7: "Flash",
    0x8010: "Cloudy",
    0x8011: "Shade",
    0x8012: "Colour temperature",
    0x8013: "Preset",
}

_PROGRAM_MODE = {
    1: "Manual",
    2: "Program",
    3: "Aperture priority",
    4: "Shutter priority",
    0x8010: "Auto",
    0x8016: "Portrait",
    0x8018: "Landscape",
    0x8019: "Close-up",
    0x8050: "Effects",
    0x8051: "Scene",
}

_FOCUS_MODE = {
    1: "Manual",
    2: "Automatic",
    3: "Macro",
    0x8010: "AF-S (single)",
    0x8011: "AF-C (continuous)",
    0x8012: "AF-A (auto)",
    0x8013: "Manual",
}

_CAPTURE_MODE = {
    1: "Single frame",
    2: "Continuous",
    0x8010: "Continuous low",
    0x8011: "Timer",
    0x8012: "Mirror up",
    0x8016: "Quiet",
    0x8018: "Continuous high",
}


def _lookup(table: "dict[int, str]", value: Any) -> str:
    try:
        raw = int(value)
    except (TypeError, ValueError):
        return str(value)
    return table.get(raw, f"0x{raw:04X}")


def format_white_balance(value: Any) -> str:
    return _lookup(_WHITE_BALANCE, value)


def format_program_mode(value: Any) -> str:
    return _lookup(_PROGRAM_MODE, value)


def format_focus_mode(value: Any) -> str:
    return _lookup(_FOCUS_MODE, value)


def format_capture_mode(value: Any) -> str:
    return _lookup(_CAPTURE_MODE, value)


#: Bits of NIKON_LiveViewProhibitCondition, as documented by libgphoto2.
_LV_PROHIBIT = (
    (1 << 0, "recording to card in progress"),
    (1 << 2, "sequence error"),
    (1 << 4, "fully pressed shutter button"),
    (1 << 5, "aperture ring is not at minimum"),
    (1 << 6, "battery is exhausted"),
    (1 << 8, "TTL error"),
    (1 << 9, "mirror is up"),
    (1 << 11, "camera is too hot"),
    (1 << 12, "card protected or missing"),
    (1 << 14, "non-CPU lens with no aperture set"),
    (1 << 15, "image area / crop conflict"),
    (1 << 17, "card is full"),
    (1 << 18, "custom white balance in progress"),
    (1 << 20, "exposure delay mode is on"),
)


def describe_lv_prohibit(mask: int) -> str:
    """Turn Nikon's prohibit bitmask into something a person can act on."""
    reasons = [text for bit, text in _LV_PROHIBIT if mask & bit]
    if not reasons:
        return f"unknown reason (0x{mask:08X})"
    return ", ".join(reasons)
