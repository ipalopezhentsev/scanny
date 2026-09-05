"""Tests for turning raw PTP property values into photographer-facing labels."""

from __future__ import annotations

import pytest

from scanny.camera import values as v


@pytest.mark.parametrize(
    "raw,expected",
    [
        (125, "1/80"),  # the value the D750 reported at 1/80s
        (1000, "1/10"),
        (10000, '1"'),
        (300000, '30"'),
        (0xFFFFFFFF, "Bulb"),
    ],
)
def test_shutter_formatting(raw, expected):
    assert v.format_shutter(raw) == expected


def test_shutter_snaps_rounded_values_to_the_marked_speed():
    # PTP rounds 1/4000s (0.25 units) to 2, and 1/3200s (0.3125) to 3. A naive
    # reciprocal would print 1/5000 and 1/3333.
    assert v.format_shutter(2) == "1/4000"
    assert v.format_shutter(3) == "1/3200"


@pytest.mark.parametrize(
    "raw,expected",
    [(400, "f/4"), (450, "f/4.5"), (560, "f/5.6"), (800, "f/8"), (2200, "f/22")],
)
def test_aperture_formatting(raw, expected):
    assert v.format_aperture(raw) == expected


def test_iso_formatting():
    assert v.format_iso(500) == "ISO 500"


@pytest.mark.parametrize(
    "raw,expected",
    [(0, "0 EV"), (-333, "-0.3 EV"), (1000, "+1.0 EV"), (-5000, "-5.0 EV")],
)
def test_exposure_bias_formatting(raw, expected):
    assert v.format_exposure_bias(raw) == expected


def test_enumerated_labels():
    assert v.format_white_balance(2) == "Auto"
    assert v.format_program_mode(1) == "Manual"
    assert v.format_focus_mode(0x8011) == "AF-C (continuous)"
    assert v.format_capture_mode(1) == "Single frame"


def test_unknown_enumerated_value_falls_back_to_its_code():
    assert v.format_white_balance(0x9999) == "0x9999"


def test_live_view_prohibit_reasons_are_explained():
    assert "battery is exhausted" in v.describe_lv_prohibit(1 << 6)
    combined = v.describe_lv_prohibit((1 << 6) | (1 << 9))
    assert "battery is exhausted" in combined and "mirror is up" in combined
    assert "unknown" in v.describe_lv_prohibit(0)
