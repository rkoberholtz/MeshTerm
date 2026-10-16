# SPDX-License-Identifier: Apache-2.0
"""Cut a message that is too long for one packet into parts that each fit.

A MeshCore message must fit in one LoRa packet (refer to
:data:`~meshterm.ui.tui.prompt.DM_BYTE_LIMIT`). When the text of a message is longer than
this limit, MeshTerm can send it as more than one message. Each of these messages is a
**part**, and each part ends with its **part number**: ``[1/3]``, ``[2/3]``, ``[3/3]``. The
part number goes at the end, so that the start of the first part (the ``@[mention]`` of a
reply) stays where a person reads first. Each part, with its part number, fits in the
limit.

A cut goes at a space when the window of the part has one. Thus each word stays complete.
A word that is longer than a part (a long URL) has no space to cut at, and MeshTerm cuts it
between two characters. Such a cut never divides a character, and it does not go into a
sequence that shows as one symbol: an accent after its letter, a skin tone or a variation
selector after its emoji, or the two sides of a zero-width joiner.

The receiver needs nothing to read the parts. Each part is an ordinary message.
"""

from __future__ import annotations

import re
import unicodedata

#: The marks that join to the character before them. A cut before one of these divides a
#: symbol that the user sees as one: U+200D zero-width joiner, U+FE0E and U+FE0F variation
#: selectors, U+20E3 combining keycap.
_JOINERS = frozenset("‍︎️⃣")

_SPACE = re.compile(r"\s")


def _bytes(text: str) -> int:
    """The UTF-8 length of ``text``: what counts against the limit of a packet."""
    return len(text.encode("utf-8"))


def part_number(index: int, total: int) -> str:
    """The text that ends part ``index`` of ``total``: a space, then ``[index/total]``."""
    return f" [{index}/{total}]"


def _joins_previous(ch: str) -> bool:
    """Whether ``ch`` joins to the character before it, so that a cut cannot go before it."""
    code = ord(ch)
    return (
        ch in _JOINERS
        or unicodedata.category(ch) in ("Mn", "Me", "Mc")
        or 0x1F3FB <= code <= 0x1F3FF  # skin tone modifiers
        or 0xE0020 <= code <= 0xE007F  # tag characters (subdivision flags)
    )


def _hard_cut(text: str, end: int) -> int:
    """Move a cut at ``end`` back, so that it does not divide a symbol.

    Returns the new cut. It is never less than 1, so that each part has text.
    """
    cut = end
    while cut > 1 and (_joins_previous(text[cut]) or text[cut - 1] == "‍"):
        cut -= 1
    return cut if cut > 1 else end


def _fit(text: str, start: int, budget: int) -> int:
    """The end of the longest run of ``text`` from ``start`` that fits in ``budget`` bytes."""
    used = 0
    end = start
    while end < len(text):
        used += _bytes(text[end])
        if used > budget:
            break
        end += 1
    return end


def _spans(text: str, budget: int) -> list[tuple[int, int]]:
    """Cut ``text`` into runs of ``budget`` bytes or less, as ``(start, end)`` indexes.

    The runs have no whitespace at their two ends. The whitespace between two runs is not
    in a run.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text) and text[start].isspace():
        start += 1
    while start < len(text):
        end = _fit(text, start, budget)
        if end == len(text):
            spans.append((start, end))
            break
        if end == start:  # one character is larger than the budget: it cannot fit at all
            end = start + 1
        # Look for the last space in the window, and also at the first character after it:
        # a space there means that the window ends at the end of a word.
        space = next(
            (i for i in range(end, start, -1) if _SPACE.match(text[i])),
            None,
        )
        if space is not None:
            stop = space
            while stop > start and text[stop - 1].isspace():
                stop -= 1
            spans.append((start, stop))
            start = space
        else:
            cut = start + _hard_cut(text[start:], end - start)
            spans.append((start, cut))
            start = cut
        while start < len(text) and text[start].isspace():
            start += 1
    return spans


def _plan(text: str, limit: int, max_parts: int) -> tuple[list[tuple[int, int]], int]:
    """Cut ``text`` for ``max_parts`` parts at most, as spans and the total of the cut.

    The size of a part number changes with the total (``[9/9]`` is shorter than
    ``[10/10]``). Thus the budget for the text of a part comes from a total, and the cut
    gives a new total. The plan starts at two parts and goes up until the two agree. The
    returned spans can be more than ``max_parts``: the caller then knows where the text
    that does not fit starts.
    """
    total = 2
    while True:
        budget = limit - _bytes(part_number(total, total))
        spans = _spans(text, max(1, budget))
        if len(spans) <= total or total >= max_parts:
            return spans, total
        total = min(max_parts, max(total + 1, len(spans)))


def split(text: str, limit: int, max_parts: int) -> list[str] | None:
    """Cut ``text`` into the parts that MeshTerm sends, each with its part number.

    Args:
        text: The text of the message.
        limit: The UTF-8 byte limit of one message.
        max_parts: The largest number of parts. With 1, MeshTerm never cuts a message.

    Returns:
        The texts to send, in order. A text that fits in ``limit`` is one part with no part
        number. ``None`` when the text needs more than ``max_parts`` parts.
    """
    if _bytes(text) <= limit:
        return [text]
    if max_parts < 2:
        return None
    spans, _ = _plan(text, limit, max_parts)
    if len(spans) > max_parts:
        return None
    total = len(spans)
    return [text[a:b] + part_number(i, total) for i, (a, b) in enumerate(spans, 1)]


def overflow_at(text: str, limit: int, max_parts: int) -> int | None:
    """The index of the first character of ``text`` that does not fit, or ``None``.

    With ``max_parts`` of 1, this is the first character whose bytes go past ``limit``.
    With more, it is the start of the text that would go in the part after the last one.
    """
    if _bytes(text) <= limit:
        return None
    if max_parts < 2:
        return _fit(text, 0, limit)
    spans, _ = _plan(text, limit, max_parts)
    if len(spans) <= max_parts:
        return None
    return spans[max_parts][0]


def unnumbered(part: str, index: int, total: int) -> str:
    """The text of ``part`` without its part number (``part`` itself when it has none)."""
    tail = part_number(index, total)
    return part[: -len(tail)] if total > 1 and part.endswith(tail) else part
