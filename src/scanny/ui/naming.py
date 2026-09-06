"""Naming the files that land on the computer: a prefix and a running number.

The camera's own names are no use for a scan. It hands out `DSC_1234.NEF` from
a counter that belongs to the body, wraps at 9999, and starts again from one
whenever the card is formatted -- so the order the pages were shot in is not
recoverable from the folder afterwards. What a scan wants is a name it chose:
`page_0007.NEF`, next to `page_0006.NEF`, counting from wherever this batch
starts.

So the sequence here is the user's, in three parts:

- **It starts where they say.** A book resumed at page 84 is set to 84, not
  renamed afterwards.
- **It can be overridden at any time.** The prefix and the next number are
  live: type over either between two shots and the shot after uses what was
  typed. Nothing is reset by that -- an override is a move of the counter, not
  a new sequence.
- **It carries on from the override.** Set 84 before a shot and the ones after
  it are 85, 86, ... The counter's only state is the next number, so there is
  no "original" sequence hiding behind an override to snap back to.

The numbers already in the folder are read as part of the sequence rather than
ignored: a number whose file exists is skipped over, and the counter lands
after it. Re-pointing at a half-scanned book therefore carries on where it
left off instead of overwriting page one, and no shot can ever land on top of
an earlier one.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "NameSequence",
    "DEFAULT_PREFIX",
    "DIGITS",
    "MAX_NUMBER",
    "clean_prefix",
    "format_name",
    "unique",
]

#: What an unconfigured sequence counts with. The trailing separator is part of
#: the prefix rather than added here, so a prefix is exactly what was typed:
#: someone who wants `page12.jpg` can have it.
DEFAULT_PREFIX = "scan_"

#: Zero padding, so the names sort in a file manager the way they were shot.
#: Four digits covers a book; past 9999 the numbers simply grow a digit, which
#: sorts wrongly against the shorter ones but is better than refusing to shoot.
DIGITS = 4

#: A counter has to stop somewhere, and the spin box needs a ceiling to offer.
MAX_NUMBER = 999_999

#: Characters Windows will not put in a file name, plus the separators. Struck
#: out of the prefix rather than rejected, because the prefix is edited a
#: keystroke at a time and half-typed text should not raise anything.
_ILLEGAL = set('<>:"/\\|?*') | {chr(code) for code in range(32)}


class NameSequence:
    """A prefix and a next number, which together name the next file saved.

    Nothing here writes anything. :meth:`claim` picks a name and moves the
    counter past it; the caller does the saving.
    """

    def __init__(
        self,
        enabled: bool = False,
        prefix: str = DEFAULT_PREFIX,
        number: int = 1,
    ) -> None:
        self._enabled = bool(enabled)
        self._prefix = clean_prefix(prefix)
        self._number = _clamp(number)

    # -- configuration -----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def number(self) -> int:
        """The number the next shot will be given, unless it is taken."""
        return self._number

    def configure(self, enabled: bool, prefix: str, number: int) -> bool:
        """Take an override, and say whether it actually changed anything.

        Every part of it is live, so this is the whole of what overriding
        means: the next shot uses these, and the shots after it carry on
        counting from here.
        """
        wanted = (bool(enabled), clean_prefix(prefix), _clamp(number))
        if wanted == (self._enabled, self._prefix, self._number):
            return False
        self._enabled, self._prefix, self._number = wanted
        return True

    # -- naming ------------------------------------------------------------

    def name(self, suffix: str = "") -> str:
        """What the next file would be called. Does not move the counter."""
        return format_name(self._prefix, self._number, suffix)

    def claim(self, directory: "Path") -> str:
        """The name one shot's files go under, with the counter left past it.

        A stem rather than a path, because one release of the shutter can
        produce two files -- a RAW and a JPEG -- and they are one picture, so
        they share one number and differ only in their extension.

        The number handed out is the first one no file in the folder is
        already using, whatever extension that file has.
        """
        taken = _stems(directory)
        while self.name().lower() in taken and self._number < MAX_NUMBER:
            self._number += 1
        stem = self.name()
        self._number = min(self._number + 1, MAX_NUMBER)
        return stem


def format_name(prefix: str, number: int, suffix: str = "") -> str:
    """One name of the sequence, for whoever needs to show it before it exists."""
    return f"{clean_prefix(prefix)}{_clamp(number):0{DIGITS}d}{suffix}"


def clean_prefix(prefix: str) -> str:
    """The prefix as a file name can carry it.

    Trailing dots and spaces go too: Windows drops them silently when the file
    is created, which would make the name on disk differ from the one the
    status bar just reported.
    """
    kept = "".join(char for char in str(prefix) if char not in _ILLEGAL)
    return kept.rstrip(". ")


def unique(path: "Path") -> "Path":
    """`path`, or the first `name_2.ext` beside it that no file has.

    This is for names that are not ours to choose -- the camera's own, when the
    sequence is switched off -- where there is no counter to advance and the
    only way past a collision is to decorate the name.
    """
    stem, suffix, counter = path.stem, path.suffix, 1
    while path.exists():
        counter += 1
        path = path.with_name(f"{stem}_{counter}{suffix}")
    return path


def _stems(directory: "Path") -> "set[str]":
    """Every name in the folder, extension dropped and folded to lower case.

    Lower case because the counter has to agree with the file system about
    what "already there" means, and Windows' does not care about case.
    """
    try:
        return {entry.stem.lower() for entry in directory.iterdir()}
    except OSError:
        # No folder yet is simply nothing taken; it is created on the way to
        # writing the first file.
        return set()


def _clamp(number: int) -> int:
    return max(1, min(int(number), MAX_NUMBER))
