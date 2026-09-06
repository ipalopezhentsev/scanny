"""The naming sequence: where it starts, and what an override does to it.

The three things worth pinning down are the ones a scan depends on. A batch
starts on the number it was told to. Typing a different number mid-batch moves
the counter and everything after it follows from there, rather than the
sequence snapping back to where it would have been. And a number a file is
already using is stepped over, so a second pass at a half-scanned book adds to
it instead of writing over page one.
"""

from __future__ import annotations

from scanny.ui.naming import (
    DEFAULT_PREFIX,
    MAX_NUMBER,
    NameSequence,
    clean_prefix,
    format_name,
    unique,
)


def test_it_starts_on_the_number_it_was_given():
    namer = NameSequence(enabled=True, prefix="page_", number=84)
    assert namer.name(".NEF") == "page_0084.NEF"


def test_a_claim_moves_the_counter_on(tmp_path):
    namer = NameSequence(enabled=True, prefix="page_", number=84)
    assert namer.claim(tmp_path) == "page_0084"
    assert namer.number == 85
    assert namer.claim(tmp_path) == "page_0085"


def test_the_names_are_padded_so_they_sort():
    assert format_name("scan_", 7) == "scan_0007"
    assert format_name("scan_", 1234) == "scan_1234"


def test_past_the_padding_the_numbers_simply_grow():
    assert format_name("scan_", 12345) == "scan_12345"


def test_an_override_is_taken_from_the_next_shot_on(tmp_path):
    namer = NameSequence(enabled=True, prefix="page_", number=1)
    namer.claim(tmp_path)
    namer.configure(True, "plate_", 40)
    assert namer.claim(tmp_path) == "plate_0040"


def test_counting_carries_on_from_an_override(tmp_path):
    """No earlier sequence survives an override to snap back to."""
    namer = NameSequence(enabled=True, prefix="p", number=1)
    for _ in range(5):
        namer.claim(tmp_path)
    namer.configure(True, "p", 40)
    assert [namer.claim(tmp_path) for _ in range(3)] == ["p0040", "p0041", "p0042"]


def test_an_override_can_go_backwards_as_well(tmp_path):
    namer = NameSequence(enabled=True, prefix="p", number=90)
    namer.configure(True, "p", 12)
    assert namer.claim(tmp_path) == "p0012"


def test_configure_says_whether_anything_actually_changed():
    namer = NameSequence(enabled=True, prefix="scan_", number=3)
    assert not namer.configure(True, "scan_", 3)
    assert namer.configure(True, "scan_", 4)


def test_a_number_already_on_disk_is_skipped(tmp_path):
    (tmp_path / "page_0001.NEF").write_bytes(b"")
    (tmp_path / "page_0002.NEF").write_bytes(b"")
    namer = NameSequence(enabled=True, prefix="page_", number=1)
    assert namer.claim(tmp_path) == "page_0003"


def test_the_skip_ignores_which_extension_the_file_has(tmp_path):
    """A number is used up by the picture, not by one of its two files."""
    (tmp_path / "page_0001.JPG").write_bytes(b"")
    namer = NameSequence(enabled=True, prefix="page_", number=1)
    assert namer.claim(tmp_path) == "page_0002"


def test_a_folder_that_is_not_there_yet_has_nothing_taken(tmp_path):
    namer = NameSequence(enabled=True, prefix="page_", number=1)
    assert namer.claim(tmp_path / "not-made-yet") == "page_0001"


def test_only_this_sequence_is_looked_at(tmp_path):
    (tmp_path / "DSC_0001.NEF").write_bytes(b"")
    namer = NameSequence(enabled=True, prefix="page_", number=1)
    assert namer.claim(tmp_path) == "page_0001"


def test_the_prefix_keeps_only_what_a_file_name_can_hold():
    assert clean_prefix('pa:ge/*?') == "page"
    assert clean_prefix("page ") == "page"
    assert clean_prefix("page.") == "page"


def test_the_counter_is_kept_inside_its_range():
    assert NameSequence(number=0).number == 1
    assert NameSequence(number=-5).number == 1
    assert NameSequence(number=MAX_NUMBER + 10).number == MAX_NUMBER


def test_the_default_sequence_is_off_until_it_is_asked_for():
    namer = NameSequence()
    assert not namer.enabled
    assert namer.prefix == DEFAULT_PREFIX
    assert namer.number == 1


def test_a_camera_name_that_is_taken_is_decorated_rather_than_replaced(tmp_path):
    (tmp_path / "DSC_0001.NEF").write_bytes(b"")
    assert unique(tmp_path / "DSC_0001.NEF").name == "DSC_0001_2.NEF"


def test_the_decoration_keeps_going_until_it_finds_a_gap(tmp_path):
    (tmp_path / "DSC_0001.NEF").write_bytes(b"")
    (tmp_path / "DSC_0001_2.NEF").write_bytes(b"")
    assert unique(tmp_path / "DSC_0001.NEF").name == "DSC_0001_3.NEF"


def test_a_free_name_is_left_alone(tmp_path):
    assert unique(tmp_path / "DSC_0001.NEF").name == "DSC_0001.NEF"
