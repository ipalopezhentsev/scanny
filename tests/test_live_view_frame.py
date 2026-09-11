"""Tests for the live-view header geometry.

The numbers here are the ones a D750 reported while the mapping was worked out
against the camera: a 6016x3376 focus space, a 324x270 focus box, and a crop
rectangle that shrinks and follows the focus point as the view is magnified.
"""

from __future__ import annotations

import struct

import pytest

from scanny.camera.nikon import CameraError, LiveViewFrame

_JPEG = b"\xff\xd8\xff" + b"\x00" * 32 + b"\xff\xd9"


def make_frame(
    *,
    lv=(640, 360),
    image=(6016, 3376),
    crop=(6016, 3376),
    crop_centre=(3008, 1688),
    af_box=(324, 270),
    af=(3008, 1688),
    header_size=384,
) -> LiveViewFrame:
    header = bytearray(header_size)
    for offset, value in (
        (8, lv[0]), (10, lv[1]),
        (12, image[0]), (14, image[1]),
        (16, crop[0]), (18, crop[1]),
        (20, crop_centre[0]), (22, crop_centre[1]),
        (24, af_box[0]), (26, af_box[1]),
        (28, af[0]), (30, af[1]),
    ):
        struct.pack_into(">H", header, offset, value)
    return LiveViewFrame.parse(bytes(header) + _JPEG)


def test_parses_geometry_and_extracts_jpeg():
    frame = make_frame()
    assert frame.jpeg == _JPEG
    assert (frame.width, frame.height) == (640, 360)
    assert (frame.image_width, frame.image_height) == (6016, 3376)
    assert (frame.af_x, frame.af_y) == (3008, 1688)
    assert frame.magnification == pytest.approx(1.0)


def test_rejects_response_without_an_image():
    with pytest.raises(CameraError, match="no JPEG"):
        LiveViewFrame.parse(b"\x00" * 384)


def test_click_maps_to_focus_coordinates_at_full_frame():
    frame = make_frame()
    assert frame.to_af_coords(0.5, 0.5) == (3008, 1688)
    # A quarter of the way across a full-frame view is a quarter of the sensor.
    assert frame.to_af_coords(0.25, 0.25) == (1504, 844)


def test_click_mapping_clamps_so_the_focus_box_stays_inside():
    frame = make_frame()
    # The camera clamps to half a box width from each edge: 324/2, 270/2.
    assert frame.to_af_coords(0.0, 0.0) == (162, 135)
    assert frame.to_af_coords(1.0, 1.0) == (6016 - 162, 3376 - 135)


def test_click_maps_through_the_crop_when_magnified():
    # Zoom level 6: a 640x480 window centred on the focus point.
    frame = make_frame(crop=(640, 480), crop_centre=(1500, 1000), af=(1500, 1000))
    assert frame.magnification == pytest.approx(6016 / 640)
    assert frame.to_af_coords(0.5, 0.5) == (1500, 1000)
    # Half a crop-width to the right is 320 sensor units, not half the sensor.
    assert frame.to_af_coords(1.0, 0.5) == (1820, 1000)
    assert frame.to_af_coords(0.0, 0.5) == (1180, 1000)


def test_focus_box_overlay_is_centred_when_the_view_follows_it():
    frame = make_frame(crop=(640, 480), crop_centre=(1500, 1000), af=(1500, 1000))
    x, y, w, h = frame.af_box_normalised
    assert x + w / 2 == pytest.approx(0.5)
    assert y + h / 2 == pytest.approx(0.5)
    assert w == pytest.approx(324 / 640)
    assert h == pytest.approx(270 / 480)


def test_focus_box_overlay_offsets_when_the_point_is_off_centre():
    frame = make_frame(af=(1504, 844))
    x, y, w, h = frame.af_box_normalised
    assert x + w / 2 == pytest.approx(0.25)
    assert y + h / 2 == pytest.approx(0.25)


def test_click_then_overlay_is_a_round_trip():
    frame = make_frame(crop=(1920, 1440), crop_centre=(2000, 1200), af=(2000, 1200))
    target_x, target_y = frame.to_af_coords(0.3, 0.7)
    moved = make_frame(
        crop=(1920, 1440), crop_centre=(2000, 1200), af=(target_x, target_y)
    )
    x, y, w, h = moved.af_box_normalised
    assert x + w / 2 == pytest.approx(0.3, abs=1e-3)
    assert y + h / 2 == pytest.approx(0.7, abs=1e-3)


def test_degenerate_header_does_not_divide_by_zero():
    frame = make_frame(image=(0, 0), crop=(0, 0), crop_centre=(0, 0), af_box=(0, 0))
    assert frame.af_box_normalised == (0.0, 0.0, 0.0, 0.0)
    assert frame.magnification == 1.0


# -- the level sensor ---------------------------------------------------------


def make_level_frame(roll_raw: int, pitch_raw: int) -> LiveViewFrame:
    header = bytearray(384)
    for offset, value in (
        (8, 640), (10, 424), (12, 6016), (14, 4016),
        (16, 6016), (18, 4016), (20, 3008), (22, 2008),
        (24, 324), (26, 270), (28, 3008), (30, 2008),
        (52, roll_raw), (56, pitch_raw),
    ):
        struct.pack_into(">H", header, offset, value)
    return LiveViewFrame.parse(bytes(header) + _JPEG)


@pytest.mark.parametrize(
    "raw,expected",
    [(0, 0), (9, 9), (90, 90), (180, 180), (181, -179), (351, -9), (313, -47), (359, -1)],
)
def test_angles_fold_to_signed_degrees(raw, expected):
    assert make_level_frame(raw, 0).roll == expected


def test_out_of_range_pitch_reads_as_no_value():
    frame = make_level_frame(10, 0xFFFF)
    assert frame.pitch is None
    assert frame.roll == 10  # the other axis still reads


def test_level_when_both_axes_are_within_a_degree():
    assert make_level_frame(0, 0).is_level
    assert make_level_frame(1, 359).is_level
    assert not make_level_frame(5, 0).is_level
    assert not make_level_frame(0, 355).is_level


def test_not_level_when_an_axis_has_no_reading():
    assert not make_level_frame(0, 0xFFFF).is_level


def test_level_fields_are_optional_for_frames_without_them():
    # The geometry-only frames the other tests build must still work.
    frame = make_frame()
    assert frame.roll == 0 and frame.pitch == 0


# -- where the picture sits in the frame -----------------------------------


def test_a_whole_frame_covers_all_of_itself():
    x, y, w, h = make_frame().crop_normalised
    assert (x, y, w, h) == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_a_magnified_frame_covers_the_part_it_shows():
    """What the navigator draws: a quarter of the frame, in its top left."""
    frame = make_frame(crop=(1504, 844), crop_centre=(752, 422))
    assert frame.crop_normalised == pytest.approx((0.0, 0.0, 0.25, 0.25))


def test_the_crop_follows_the_focus_point_across_the_frame():
    frame = make_frame(crop=(1504, 844), crop_centre=(4512, 2532))
    x, y, w, h = frame.crop_normalised
    assert (x + w / 2, y + h / 2) == pytest.approx((0.75, 0.75))


def test_a_fraction_of_the_frame_maps_to_the_sensor_whatever_is_on_screen():
    """The navigator's coordinates do not go through the crop rectangle.

    A click on the image does, and must: it is a fraction of what is on
    screen. These are fractions of the frame that picture was cut from, so
    they mean the same place at every magnification.
    """
    whole = make_frame()
    magnified = make_frame(crop=(1504, 844), crop_centre=(4512, 2532))
    assert whole.to_af_coords_in_frame(0.25, 0.5) == (1504, 1688)
    assert magnified.to_af_coords_in_frame(0.25, 0.5) == (1504, 1688)


def test_a_point_at_the_edge_is_pulled_in_to_where_the_box_fits():
    frame = make_frame()
    assert frame.to_af_coords_in_frame(0.0, 0.0) == (162, 135)
    assert frame.to_af_coords_in_frame(1.0, 1.0) == (6016 - 162, 3376 - 135)


# -- a place on the sensor, on whatever crop is on show ---------------------


def test_a_place_on_screen_becomes_a_place_on_the_sensor():
    """The inverse of the crop rectangle, and what a shift-drag goes out as."""
    whole = make_frame()
    assert whole.to_frame_fraction(0.5, 0.5) == pytest.approx((0.5, 0.5))
    quadrant = make_frame(crop=(1504, 844), crop_centre=(752, 422))
    assert quadrant.to_frame_fraction(0.5, 0.5) == pytest.approx((0.125, 0.125))
    assert quadrant.to_frame_fraction(1.0, 1.0) == pytest.approx((0.25, 0.25))


def test_a_sensor_rectangle_comes_back_as_the_part_of_it_that_is_on_screen():
    area = (0.1, 0.1, 0.1, 0.1)
    whole = make_frame()
    assert whole.area_normalised(area) == pytest.approx(area)
    # The same rectangle fills half of a view showing a quarter of the frame.
    quadrant = make_frame(crop=(1504, 844), crop_centre=(752, 422))
    assert quadrant.area_normalised(area) == pytest.approx((0.4, 0.4, 0.4, 0.4))


def test_a_sensor_rectangle_the_view_has_left_behind_is_not_on_the_picture():
    """Which has to be told apart from "the whole picture", because the meter
    answers one with a reading and the other with nothing."""
    far = make_frame(crop=(1504, 844), crop_centre=(4512, 2532))
    assert far.area_normalised((0.0, 0.0, 0.1, 0.1)) is None


def test_a_sensor_rectangle_half_off_the_picture_is_clipped_to_it():
    quadrant = make_frame(crop=(1504, 844), crop_centre=(752, 422))
    clipped = quadrant.area_normalised((0.2, 0.2, 0.2, 0.2))
    assert clipped == pytest.approx((0.8, 0.8, 0.2, 0.2))
