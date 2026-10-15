# SPDX-License-Identifier: Apache-2.0
"""Emoji reactions: the wire format of MeshCore One, the fold, the send path, and the chat.

The parse cases come from the tests of MeshCore One (``ReactionParserTests.swift`` and
``MeshCoreOpenReactionParserTests.swift``). Thus MeshTerm reads the same bodies as the app
whose format it shares. The hash test calculates the expected value a second way, with the
Base32 of the standard library, so that the test does not only repeat the code.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from meshterm.core import reactions
from meshterm.core.config import Settings
from meshterm.core.models import ChatMessage, Contact, Conversation, Delivery, Message
from meshterm.core.preferences import Preferences
from meshterm.core.reactions import MC1, MCO, MCO_V1, ReactionFold, parse
from meshterm.persistence.repository import Repository
from meshterm.services.chat_service import ChatService
from meshterm.ui.chat import ChatScreen
from tests.conftest import plain

_T = 1_704_067_200  # 2024-01-01 00:00:00 UTC


def _at(seconds: int) -> datetime:
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


# -- the hash -----------------------------------------------------------------------------


def test_the_hash_is_crockford_base32_of_the_first_five_bytes_of_sha256() -> None:
    """SHA-256 of the UTF-8 body and the little-endian timestamp, five bytes, Crockford Base32."""
    digest = hashlib.sha256(b"Hello" + _T.to_bytes(4, "little")).digest()[:5]
    rfc = base64.b32encode(digest).decode()  # five bytes make exactly eight characters
    expected = rfc.translate(
        str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", "0123456789abcdefghjkmnpqrstvwxyz")
    )
    assert reactions.message_hash("Hello", _T) == expected
    assert reactions.message_hash("Hello", _T + 1) != expected
    assert reactions.message_hash("World", _T) != expected


def test_the_texts_that_meshterm_sends_name_their_target() -> None:
    """A channel reaction names the author first. A direct one has the emoji and the hash."""
    digest = reactions.message_hash("hi all", _T)
    assert reactions.channel_text("👍", "Alice", "hi all", _T) == f"@[Alice]👍\n{digest}"
    assert reactions.direct_text("👍", "hi all", _T) == f"👍\n{digest}"
    assert parse(reactions.channel_text("👍", "Alice", "hi all", _T), channel=True) == (
        reactions.Reaction("👍", MC1, digest, author="Alice")
    )


# -- the parse ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "emoji", "author", "digest"),
    [
        ("@[AlphaNode]👍\n7f3a9c12", "👍", "AlphaNode", "7f3a9c12"),
        ("❤️@[BetaNode]\ne4d8b1a0", "❤️", "BetaNode", "e4d8b1a0"),  # before 1.4.1
        ("@[Node]👍\nOOOOOOOO", "👍", "Node", "00000000"),
        ("@[Node]👍\niiiiiiii", "👍", "Node", "11111111"),
        ("@[Node]👍\nLLLLLLLL", "👍", "Node", "11111111"),
        ("@[Node:Alpha]👍\na1b2c3d4", "👍", "Node:Alpha", "a1b2c3d4"),
        ("@[Alice]👍\nreceived", "👍", "Alice", "rece1ved"),
        ("@[A]👍🏽\na1b2c3d4", "👍🏽", "A", "a1b2c3d4"),
        ("@[A]👨‍👩‍👧\na1b2c3d4", "👨‍👩‍👧", "A", "a1b2c3d4"),
        ("@[A]🇨🇦\na1b2c3d4", "🇨🇦", "A", "a1b2c3d4"),
    ],
)
def test_a_channel_reaction_parses(body: str, emoji: str, author: str, digest: str) -> None:
    """The two orders, the Crockford substitutions, a colon in a name, and each glyph shape."""
    assert parse(body, channel=True) == reactions.Reaction(emoji, MC1, digest, author=author)


@pytest.mark.parametrize(
    "body",
    [
        "Just a normal message",
        "@[Node]👍",
        "👍 [Node]\na1b2c3d4",
        "👍@Node\na1b2c3d4",
        "@[Node]👍\nabc",
        "@[Node]👍\nuuuuuuuu",
        "@[]👍\na1b2c3d4",
        "@[Node]\na1b2c3d4",
        "@[Alice]🎉 on my way\npassword",
        "🎉 on my way@[Alice]\npassword",
        "A@[Node]\na1b2c3d4",
    ],
)
def test_a_usual_channel_message_is_not_a_reaction(body: str) -> None:
    """A body that only starts with an emoji, or is short of a part, stays a message."""
    assert parse(body, channel=True) is None


def test_a_direct_reaction_has_no_author() -> None:
    """A direct reaction is the emoji and the hash. The channel shape is not one."""
    assert parse("👍\nb45pc4ek", channel=False) == reactions.Reaction("👍", MC1, "b45pc4ek")
    assert parse("@[Node]👍\na1b2c3d4", channel=False) is None
    assert parse("🎉 on my way\npassword", channel=False) is None
    assert parse("1\na1b2c3d4", channel=False) is None


def test_the_reactions_of_meshcore_open_parse() -> None:
    """The v3 form indexes a table of 184 emoji. The v1 form carries the time and two hashes."""
    assert len(reactions.MCO_EMOJI) == 184
    assert parse("r:a1b2:00", channel=True) == reactions.Reaction("👍", MCO, "a1b2")
    assert parse("r:0000:45", channel=False).emoji == "😶"  # the last smiley
    assert parse("r:0000:66", channel=False).emoji == "💪"  # the last gesture
    assert parse("r:0000:67", channel=False).emoji == "❤️"  # the first heart
    assert parse("r:0000:b7", channel=False).emoji == "🚀"  # the last object
    for bad in ("r:a1b2:b8", "r:A1B2:00", "r:a1b2:0A", "r:a1b:00", "x:a1b2:00", ""):
        assert parse(bad, channel=True) is None
    old = parse("r:1704067200000_12345_67890:👍", channel=True)
    assert old == reactions.Reaction(
        "👍", MCO_V1, "1704067200000_12345_67890", timestamp=_T, author_hash=12345, body_hash=67890
    )


def test_the_dart_hash_is_thirty_bits_and_never_zero() -> None:
    """The ``String.hashCode`` of Dart: 1 for no input, and always under 2**30."""
    assert reactions.dart_hash([]) == 1
    value = reactions.dart_hash([ord(c) for c in "test string"])
    assert 0 < value < 1 << 30
    assert reactions.dart_hash([ord(c) for c in "hello"]) != reactions.dart_hash(
        [ord(c) for c in "world"]
    )


def test_the_reaction_set_keeps_only_single_emoji() -> None:
    """A word that is not one emoji is ignored. With nothing left, the default set is used."""
    assert reactions.reaction_set("🔥 ok 👍 🔥 👍👍") == ["🔥", "👍"]
    assert reactions.reaction_set("") == reactions.DEFAULT_REACTIONS.split()
    assert reactions.reaction_set("abc") == reactions.DEFAULT_REACTIONS.split()


# -- the fold -----------------------------------------------------------------------------


def _channel(text: str, seconds: int = _T, *, outbound: bool = False) -> ChatMessage:
    return ChatMessage(text=text, outbound=outbound, is_channel=True, created_at=_at(seconds))


def test_a_channel_reaction_folds_onto_its_target_once_for_each_node() -> None:
    """The target stays. The reaction leaves the transcript. A copy from the same node is one."""
    fold = ReactionFold(channel=True, self_name="Me")
    target = _channel("Alice: hi all")
    assert fold.take(target) is False
    react = reactions.channel_text("👍", "Alice", "hi all", _T)
    assert fold.take(_channel(f"Bob: {react}", _T + 5)) is True
    assert fold.take(_channel(f"Bob: {react}", _T + 6)) is True  # a relayed copy
    assert fold.take(_channel(f"Carol: {react}", _T + 7)) is True
    assert fold.take(_channel(reactions.channel_text("❤️", "Alice", "hi all", _T), outbound=True))
    badges = fold.badges(target)
    assert [(b.emoji, [r.name for r in b.reactors]) for b in badges] == [
        ("👍", ["Bob", "Carol"]),
        ("❤️", [None]),
    ]
    assert badges[1].ours and fold.reacted(target, "❤️") and not fold.reacted(target, "👍")
    assert fold.count(target) == 3


def test_a_reaction_to_our_channel_message_names_our_node() -> None:
    """Our channel messages have no prefix in the history. Our name is their author."""
    fold = ReactionFold(channel=True, self_name="Me")
    ours = _channel("lunch?", outbound=True)
    fold.take(ours)
    fold.take(_channel("Bob: " + reactions.channel_text("😂", "Me", "lunch?", _T)))
    assert [b.emoji for b in fold.badges(ours)] == ["😂"]
    assert fold.target_of(ours) == ("Me", "lunch?", _T)


def test_a_reaction_that_arrives_first_waits_for_its_target() -> None:
    """On a mesh, a reaction can arrive before its target. It attaches when the target comes."""
    fold = ReactionFold(channel=True)
    assert fold.take(_channel("Bob: " + reactions.channel_text("👍", "Alice", "late", _T)))
    target = _channel("Alice: late")
    assert fold.take(target) is False
    assert [b.emoji for b in fold.badges(target)] == ["👍"]


def test_the_author_tells_two_identical_messages_apart() -> None:
    """Two nodes that send the same body in the same second get their own reactions."""
    fold = ReactionFold(channel=True)
    alice, carol = _channel("Alice: ok"), _channel("Carol: ok")
    fold.take(alice)
    fold.take(carol)
    fold.take(_channel("Bob: " + reactions.channel_text("👍", "Carol", "ok", _T)))
    assert fold.badges(alice) == [] and len(fold.badges(carol)) == 1


def test_a_direct_reaction_folds_by_hash_alone() -> None:
    """In a direct exchange, the peer is the only other node. Its reactor has no name."""
    fold = ReactionFold(channel=False)
    ours = ChatMessage(text="see you", outbound=True, created_at=_at(_T))
    fold.take(ours)
    assert fold.take(ChatMessage(text=reactions.direct_text("👍", "see you", _T), peer="d4e5"))
    (badge,) = fold.badges(ours)
    assert badge.emoji == "👍" and badge.reactors == [reactions.Reactor(None)]


def test_a_reaction_of_meshcore_open_finds_its_target_near_in_time() -> None:
    """The short hash of v3 is compared with the messages within the window of the reaction."""
    fold = ReactionFold(channel=True)
    target = _channel("Alice: hello there")
    fold.take(target)
    digest = reactions.mco_hash(_T, "Alice", "hello there")
    assert fold.take(_channel(f"Bob: r:{digest}:05", _T + 30))
    assert [b.emoji for b in fold.badges(target)] == ["🔥"]
    far = _channel("Carol: hello there", _T + 10_000)
    fold.take(far)
    late = reactions.mco_hash(_T + 10_000, "Carol", "hello there")
    assert fold.take(_channel(f"Bob: r:{late}:05", _T))  # out of the window: no target
    assert fold.badges(far) == []


# -- the send path ------------------------------------------------------------------------


class _Recorder:
    """A device that records the timestamp and the try number of each send."""

    def __init__(self, acks: list[bool]) -> None:
        self._acks = list(acks)
        self.direct: list[tuple[str, int | None, int]] = []
        self.channel: list[tuple[str, int | None]] = []

    async def send_direct_message(self, contact, text, *, timestamp=None, attempt=0):  # noqa: ANN001, ANN201
        self.direct.append((text, timestamp, attempt))
        acked = self._acks.pop(0) if self._acks else False
        code = f"{len(self.direct):08x}"
        return Delivery(code, object() if acked else None)

    async def send_channel_in_scope(self, index, text, scope, *, timestamp=None):  # noqa: ANN001, ANN201
        self.channel.append((text, timestamp))


class _Ctx:
    def __init__(self, device: _Recorder, repo: Repository) -> None:
        self._device = device
        self.repo = repo
        self.log = logging.getLogger("test.reactions")
        self.settings = Settings()
        self.preferences = Preferences()

    async def device(self) -> _Recorder:
        return self._device


@pytest.fixture()
def repo(tmp_path: Path) -> Repository:
    """A Repository on a scratch database. It closes when the test ends."""
    r = Repository(tmp_path / "reactions.db")
    yield r
    r.close()


_ALICE = Contact(name="Alice", public_key="d4e5" + "0" * 60, key_prefix="d4e5f6a7")


async def test_a_sent_message_stores_the_timestamp_that_went_out(repo: Repository) -> None:
    """The stored time is the wire time to the second. A reaction to it then names it right."""
    device = _Recorder([True])
    chat = ChatService(_Ctx(device, repo))
    sent = await chat.send_channel(0, "hello")
    assert device.channel[0][1] == int(sent.created_at.timestamp())
    assert sent.created_at.microsecond == 0
    dm = await chat.send_direct(_ALICE, "hey")
    assert device.direct[0][1] == int(dm.created_at.timestamp())
    stored = repo.recent_chat_messages(is_channel=False, peer=_ALICE.key_prefix)
    assert stored[-1].created_at == dm.created_at


async def test_retries_keep_the_timestamp_and_count_the_try(repo: Repository) -> None:
    """Soft retries and ^R send one message: the same timestamp, the next try number, up to 3."""
    device = _Recorder([False, False, False, False, False])
    ctx = _Ctx(device, repo)
    ctx.preferences.direct_message_soft_retries = 1
    chat = ChatService(ctx)
    sent = await chat.send_direct(_ALICE, "hey")
    await chat.resend_direct(_ALICE, sent)
    await chat.resend_direct(_ALICE, sent)
    stamps = {timestamp for _, timestamp, _ in device.direct}
    assert stamps == {int(sent.created_at.timestamp())}
    assert [attempt for *_, attempt in device.direct] == [0, 1, 2, 3, 3, 3]


async def test_a_reaction_does_not_raise_the_unread_badge(repo: Repository) -> None:
    """A reaction shows on its target, so a badge for it would point at nothing new."""
    chat = ChatService(_Ctx(_Recorder([]), repo))
    react = Message(text=reactions.direct_text("👍", "hey", _T), sender="d4e5f6a7")
    assert await chat._notifies(react, None) is False
    channel = Message(
        text="Bob: " + reactions.channel_text("👍", "Alice", "hi", _T), channel=0, is_channel=True
    )
    assert await chat._notifies(channel, "abc") is False


# -- the chat screen ----------------------------------------------------------------------


class _Session:
    """A session stand-in that answers the reaction dialog with ``answer``."""

    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.asked: list[str] = []
        self.items: list = []
        self.filterable: bool | None = None

    def invalidate(self) -> None:
        pass

    async def select(self, title, items, *, filterable=True, **_kw):  # noqa: ANN001, ANN003, ANN201
        self.asked.append(title)
        self.items, self.filterable = items, filterable
        return self.answer


def _screen(messages: list[ChatMessage], session: _Session, sent: list[str]) -> ChatScreen:
    async def send(text: str) -> ChatMessage:
        sent.append(text)
        return ChatMessage(text=text, outbound=True, is_channel=True, created_at=_at(_T + 60))

    return ChatScreen(
        Conversation(label="#public", is_channel=True, channel_idx=0),
        messages,
        send=send,
        names={},
        session=session,
        self_name="Me",
    )


def test_the_transcript_shows_reactions_under_their_target() -> None:
    """The reaction leaves the transcript. Its emoji and count show under the target."""
    react = "Bob: " + reactions.channel_text("👍", "Alice", "hi all", _T)
    screen = _screen([_channel("Alice: hi all"), _channel(react, _T + 5)], _Session(None), [])
    assert len(screen._messages) == 1
    body = plain(screen.render_body(60))
    assert "👍 1" in body
    assert reactions.message_hash("hi all", _T) not in body and "@[Alice]" not in body
    screen.handle("up")  # the selection names the nodes instead of the count
    assert "👍 Bob" in plain(screen.render_body(60))


async def test_ctrl_e_sends_the_reaction_and_folds_our_copy() -> None:
    """^E asks which emoji, sends the text of MeshCore One, and shows our reaction on the target."""
    session, sent = _Session("😂"), []
    target = _channel("Alice: hi all")
    screen = _screen([target], session, sent)
    screen.handle("up")
    assert "^E react" in screen.footer_hint
    screen.handle("react")
    await asyncio.sleep(0)
    await asyncio.gather(*screen._tasks)
    assert session.asked == ["React — Alice"]
    assert sent == [reactions.channel_text("😂", "Alice", "hi all", _T)]
    assert screen._messages == [target]
    assert screen._fold.reacted(target, "😂")


def test_a_room_shows_reaction_text_as_it_arrived() -> None:
    """A room has no reactions. Its messages stay messages, and ^E offers nothing."""
    body = reactions.direct_text("👍", "x", _T)
    room = Contact(name="Board", public_key="ab" * 32, key_prefix="ab" * 6, node_type=3)
    screen = ChatScreen(
        Conversation(label="Board", is_channel=False, contact=room),
        [ChatMessage(text=body, peer="abababababab", created_at=_at(_T))],
        send=None,
        names={},
        session=_Session(None),
    )
    assert len(screen._messages) == 1
    screen.handle("up")
    assert not screen._can_react()


def test_the_messages_list_previews_a_reaction_in_words() -> None:
    """The latest message of a conversation can be a reaction. Its preview says what it is."""
    from meshterm.tools.chat import _preview_text

    react = "Bob: " + reactions.channel_text("👍", "Alice", "hi", _T)
    preview = _preview_text(_channel(react), lambda name: None).plain
    assert "Bob" in preview and "reacted 👍" in preview and "@[" not in preview
    ours = ChatMessage(text=reactions.direct_text("😂", "hi", _T), outbound=True)
    assert "reacted 😂" in _preview_text(ours, lambda name: None).plain


def test_a_reaction_from_a_node_with_a_long_name_folds() -> None:
    """A node name can have 31 characters. A sender of 22 is still split off its reaction.

    The two messages are from the hardware log that found the fault: the name limit was 20,
    so the reaction of ``Hazelwood r0b3.Anduril`` showed as a new message.
    """
    fold = ReactionFold(channel=True)
    target = _channel("Hazelwood: ⚽ Union 2-1 at Chicago Fire (59')", 1_792_029_645)
    fold.take(target)
    react = _channel("Hazelwood r0b3.Anduril: @[Hazelwood]🔥\nmzjsjjpk", 1_792_029_654)
    assert fold.take(react) is True
    (badge,) = fold.badges(target)
    assert badge.emoji == "🔥" and badge.reactors == [reactions.Reactor("Hazelwood r0b3.Anduril")]


async def test_the_reaction_dialog_searches_every_emoji_by_name() -> None:
    """The preferred emoji come first, then each other emoji. A typed name filters them all."""
    from meshterm.ui.chat import emoji_catalog
    from meshterm.ui.tui import Choice, Separator

    session = _Session(None)
    screen = _screen([_channel("Alice: hi all")], session, [])
    screen.handle("up")
    screen.handle("react")
    await asyncio.sleep(0)
    await asyncio.gather(*screen._tasks)
    assert session.filterable is True
    values = [item.value for item in session.items if isinstance(item, Choice)]
    assert values[:6] == reactions.DEFAULT_REACTIONS.split()
    assert "🔥" in values and values.count("👍") == 1  # a preferred emoji is listed once
    assert isinstance(session.items[0], Separator)  # "Your reactions", then "All emoji"
    assert sum(isinstance(item, Separator) for item in session.items) == 2
    fire = next(i for i in session.items if isinstance(i, Choice) and i.value == "🔥")
    assert "fire" in fire.label.plain
    assert len(emoji_catalog()) > 1000
    assert all(reactions.is_reaction_emoji(emoji) for emoji in emoji_catalog())
