# SPDX-License-Identifier: Apache-2.0
"""Tests for the parts of a long message: the cut, and how the chat screen sends them."""

from __future__ import annotations

import asyncio

import pytest

from meshterm.core import parts
from meshterm.core.models import ChatMessage, Contact, Conversation
from meshterm.core.preferences import Preferences
from meshterm.ui.chat import ChatScreen

LIMIT = 130

_TEXT = " ".join(f"word{i:03d}" for i in range(40))  # 319 bytes, words of 7 characters


def _bodies(cut: list[str]) -> list[str]:
    return [parts.unnumbered(p, i, len(cut)) for i, p in enumerate(cut, 1)]


# --- the cut -----------------------------------------------------------------------------


def test_a_text_that_fits_is_one_part_with_no_part_number() -> None:
    """A text in the limit goes out as it is, with no part number."""
    assert parts.split("hello", LIMIT, 3) == ["hello"]
    assert parts.split("x" * LIMIT, LIMIT, 3) == ["x" * LIMIT]


def test_each_part_fits_and_ends_with_its_part_number() -> None:
    """Each part, with its part number, fits in the limit."""
    cut = parts.split(_TEXT, LIMIT, 3)
    assert cut is not None and len(cut) == 3
    for i, part in enumerate(cut, 1):
        assert len(part.encode("utf-8")) <= LIMIT
        assert part.endswith(f" [{i}/3]")


def test_the_parts_cut_at_spaces_and_keep_all_the_words() -> None:
    """Each cut is at a space, and the parts together hold all the words."""
    cut = parts.split(_TEXT, LIMIT, 3)
    assert cut is not None
    bodies = _bodies(cut)
    assert " ".join(bodies) == _TEXT
    for body in bodies:
        assert body == body.strip()
        assert all(word.startswith("word") and len(word) == 7 for word in body.split(" "))


def test_a_word_longer_than_a_part_is_cut_between_characters() -> None:
    """A word with no space to cut at is cut between two characters."""
    url = "https://example.com/" + "a" * 200
    cut = parts.split(url, LIMIT, 3)
    assert cut is not None and len(cut) == 2
    assert "".join(_bodies(cut)) == url
    assert all(len(p.encode("utf-8")) <= LIMIT for p in cut)


def test_a_cut_never_divides_an_emoji_sequence_or_an_accent() -> None:
    """A cut does not divide a joined emoji, or an accent from its letter."""
    family = "👨‍👩‍👧"  # three emoji and two zero-width joiners: one symbol
    cut = parts.split(family * 20, LIMIT, 9)
    assert cut is not None
    for body in _bodies(cut):
        assert body == family * (len(body) // len(family))
    accented = "é" * 100  # e and a combining acute accent, 3 bytes for each pair
    cut = parts.split(accented, LIMIT, 9)
    assert cut is not None
    for body in _bodies(cut):
        assert not body.startswith("́")
        assert body.endswith("́")


def test_a_text_that_needs_too_many_parts_is_refused() -> None:
    """A text that needs more than the largest number of parts has no cut."""
    assert parts.split("x" * 400, LIMIT, 3) is None
    assert parts.split("x" * (LIMIT + 1), LIMIT, 1) is None


def test_the_part_number_grows_with_the_total() -> None:
    """The part number shows the index and the total."""
    assert parts.part_number(1, 3) == " [1/3]"
    assert parts.part_number(2, 9) == " [2/9]"


def test_overflow_at_finds_the_text_after_the_last_part() -> None:
    """The overflow starts where the text after the last part starts."""
    assert parts.overflow_at("x" * LIMIT, LIMIT, 1) is None
    assert parts.overflow_at("x" * (LIMIT + 5), LIMIT, 1) == LIMIT  # the old rule
    assert parts.overflow_at(_TEXT, LIMIT, 3) is None
    long = "y" * 500
    cut_at = parts.overflow_at(long, LIMIT, 3)
    budget = LIMIT - len(parts.part_number(3, 3))
    assert cut_at == 3 * budget


def test_unnumbered_removes_only_its_own_part_number() -> None:
    """Only the part number of the part itself is removed."""
    assert parts.unnumbered("abc [2/3]", 2, 3) == "abc"
    assert parts.unnumbered("abc [2/3]", 1, 3) == "abc [2/3]"
    assert parts.unnumbered("abc", 1, 1) == "abc"


def test_the_preference_permits_three_parts_by_default() -> None:
    """The default of the preference is three parts."""
    assert Preferences().message_parts_max == 3


# --- the chat screen ---------------------------------------------------------------------


class _Session:
    def invalidate(self) -> None:
        pass

    def run_detached(self, work):  # noqa: ANN001, ANN201
        return asyncio.ensure_future(work)


def _screen(send, *, channel: bool, max_parts: int = 3, gap: float = 0.0) -> ChatScreen:
    if channel:
        conv = Conversation(label="#public", is_channel=True, channel_idx=0)
    else:
        conv = Conversation(
            label="Alice",
            is_channel=False,
            contact=Contact(name="Alice", public_key="d4" + "0" * 62, key_prefix="d4e5f6a7"),
        )
    return ChatScreen(
        conv,
        [],
        send=send,
        names={},
        session=_Session(),
        max_parts=max_parts,
        part_gap_s=gap,
    )


async def _enter(screen: ChatScreen, text: str) -> None:
    screen.handle("text", text)
    screen.handle("enter")
    await asyncio.gather(*list(screen._tasks))


async def test_a_channel_sends_each_part_in_order_with_the_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A channel sends the parts in order, with the gap between them."""
    sent: list[str] = []
    waits: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(delay: float) -> None:
        waits.append(delay)
        await real_sleep(0)

    async def send(text: str) -> ChatMessage:
        sent.append(text)
        return ChatMessage(text=text, outbound=True, is_channel=True, channel_idx=0)

    screen = _screen(send, channel=True, gap=5.0)
    monkeypatch.setattr("meshterm.ui.chat.asyncio.sleep", sleep)
    await _enter(screen, _TEXT)

    assert sent == parts.split(_TEXT, LIMIT, 3)
    assert waits == [5.0, 5.0]  # between the parts, not before the first
    assert [m.text for m in screen._messages] == sent
    assert screen._status == ""
    assert not screen._sending


async def test_a_direct_message_sends_each_part_after_the_one_before() -> None:
    """A direct part goes out only after the part before it."""
    sent: list[str] = []
    in_flight = 0

    async def send(text: str) -> ChatMessage:
        nonlocal in_flight
        in_flight += 1
        assert in_flight == 1  # one part at a time
        await asyncio.sleep(0)
        sent.append(text)
        in_flight -= 1
        return ChatMessage(text=text, outbound=True, peer="d4e5f6a7", acked=False)

    screen = _screen(send, channel=False, max_parts=4)
    await _enter(screen, _TEXT + " more")
    assert len(sent) == 3
    assert [m.text for m in screen._messages] == sent
    assert all(m.acked is False for m in screen._messages)  # an unacked part did not stop the rest


async def test_a_failed_part_stops_the_rest_and_keeps_their_text() -> None:
    """A part that fails stops the parts after it, and their text goes back."""
    calls = 0

    async def send(text: str) -> ChatMessage:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("radio busy")
        return ChatMessage(text=text, outbound=True, peer="d4e5f6a7", acked=True)

    screen = _screen(send, channel=False)
    await _enter(screen, _TEXT)
    cut = parts.split(_TEXT, 150, 3)  # the limit of a direct message
    assert cut is not None
    assert calls == 2
    assert len(screen._messages) == 1  # the failed bubble is gone
    assert "radio busy" in screen._status
    assert screen._editor.text == " ".join(_bodies(cut)[1:])  # parts 2 and 3, unnumbered


async def test_too_many_parts_is_refused_and_the_text_stays() -> None:
    """A text with too many parts is refused, and the text stays."""
    sent: list[str] = []

    async def send(text: str) -> ChatMessage:
        sent.append(text)
        return ChatMessage(text=text, outbound=True)

    screen = _screen(send, channel=True, max_parts=2)
    await _enter(screen, _TEXT)
    assert sent == []
    assert screen._editor.text == _TEXT
    assert screen._status == "Too long for 2 parts — trim to send."


def test_the_gauge_counts_the_parts() -> None:
    """The gauge shows the number of parts and the bytes of the last part."""
    screen = _screen(None, channel=True)
    screen.handle("text", _TEXT)
    gauge = screen._byte_counter(LIMIT).plain
    cut = parts.split(_TEXT, LIMIT, 3)
    assert cut is not None
    assert gauge == f"3 parts · {len(cut[-1].encode('utf-8'))}/{LIMIT}"
    assert screen._overflow_at(LIMIT) is None  # nothing is marked red


def test_one_part_keeps_the_old_gauge_and_the_old_overflow() -> None:
    """With one part, the gauge and the overflow are as before."""
    screen = _screen(None, channel=True, max_parts=1)
    screen.handle("text", "x" * (LIMIT + 2))
    assert screen._byte_counter(LIMIT).plain == f"{LIMIT + 2}/{LIMIT}"
    assert screen._overflow_at(LIMIT) == LIMIT
