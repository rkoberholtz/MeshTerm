# SPDX-License-Identifier: Apache-2.0
"""Emoji reactions to chat messages, in the wire format of MeshCore One.

A reaction is a usual text message. The firmware and the repeaters do not know the format.
Only the apps do. MeshTerm writes the format of MeshCore One, and reads it and the two
formats of meshcore-open. The protocol is in ``docs/Reactions.md`` of MeshCore One. In short:

* A reaction names its target message by a hash of the body of the target and the
  timestamp of its sender: SHA-256 of the UTF-8 body and the timestamp as a little-endian
  32-bit integer, the first five bytes, in Crockford Base32 (eight characters). Each node
  that received the target calculates the same hash, because the sender's timestamp travels
  with the message.
* On a channel, the reaction also names the author of the target, because two nodes can
  send the same body in the same second: ``@[Author]👍`` then a newline and the hash.
  Before MeshCore One 1.4.1, the emoji came first (``👍@[Author]``). MeshTerm reads the two
  orders and writes the new one.
* In a direct exchange, there are two persons only, so the reaction is ``👍``, a newline,
  and the hash.

A body that only starts with an emoji is a usual message. The emoji part must be exactly
one emoji (one glyph) for the body to be a reaction.

MeshTerm stores a reaction as the usual message that it is. Thus the history keeps all that
was received. The chat screen folds the reactions of a transcript onto their targets when it
shows the transcript (refer to :class:`ReactionFold`).
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

from .channels import split_channel_sender
from .models import ChatMessage

#: The reactions that the reaction dialog offers when the preference is empty. They are the
#: default set of MeshCore One, in its order.
DEFAULT_REACTIONS = "👍 👎 ❤️ 😂 😮 😢"

#: How far apart in time, in seconds, a reaction of meshcore-open and its target can be. The
#: short hash of meshcore-open (16 bits) matches many messages in a long history. Thus
#: MeshTerm compares it only with the messages near the reaction in time. MeshCore One uses
#: the same window.
WINDOW_S = 300

#: The maximum number of reactions that wait for a target that did not arrive yet.
PENDING_CAP = 100

# -- Crockford Base32 ---------------------------------------------------------------------

_CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz"
_DECODE = {char: char for char in _CROCKFORD}
_DECODE.update({char.upper(): char for char in _CROCKFORD})
# The letters that a person can mistake for a digit read as that digit.
_DECODE.update({"o": "0", "O": "0", "i": "1", "I": "1", "l": "1", "L": "1"})


def _crockford(data: bytes) -> str:
    """Encode five bytes as eight Crockford Base32 characters, the most significant first."""
    bits = int.from_bytes(data[:5], "big")
    return "".join(_CROCKFORD[(bits >> shift) & 0x1F] for shift in range(35, -5, -5))


def _normal_hash(raw: str) -> str | None:
    """The canonical (lowercase) form of an eight-character hash, or ``None`` if it is not one."""
    if len(raw) != 8:
        return None
    out = [_DECODE.get(char) for char in raw]
    if None in out:
        return None
    return "".join(out)  # type: ignore[arg-type] - None is excluded above


def message_hash(body: str, timestamp: int) -> str:
    """The hash with which a reaction names its target message.

    Args:
        body: The body of the target. On a channel, this is the text after the ``Name: ``
            prefix of the sender.
        timestamp: The timestamp that the sender gave the target, in seconds.

    Returns:
        Eight lowercase Crockford Base32 characters.
    """
    data = body.encode("utf-8") + (timestamp & 0xFFFFFFFF).to_bytes(4, "little")
    return _crockford(hashlib.sha256(data).digest())


def wire_timestamp(when: datetime) -> int:
    """The timestamp of a message as its sender put it on the wire: whole seconds."""
    return int(when.timestamp())


# -- the emoji part -----------------------------------------------------------------------

_VS = frozenset("︎️")
_ZWJ = "‍"
_KEYCAP = "⃣"
_TONES = frozenset(chr(cp) for cp in range(0x1F3FB, 0x1F400))
_TAGS = frozenset(chr(cp) for cp in range(0xE0020, 0xE0080))


def _is_regional(char: str) -> bool:
    """Whether ``char`` is a Regional Indicator (one half of a flag)."""
    return 0x1F1E6 <= ord(char) <= 0x1F1FF


def _is_pictograph(char: str) -> bool:
    """Whether ``char`` can start an emoji: a symbol outside the Latin and punctuation blocks."""
    return ord(char) >= 0x2000 and unicodedata.category(char) == "So"


def is_reaction_emoji(value: str) -> bool:
    """Whether ``value`` is exactly one emoji: the emoji part of a reaction.

    One emoji is one glyph. It can be a flag (two Regional Indicators), a keycap (``1️⃣``),
    or pictographs joined with zero-width joiners (``❤️‍🔥``). Each pictograph can have a
    variation selector, a skin tone, and a tag sequence. Text, or an emoji with text after
    it, is not a reaction.
    """
    if not value or value.isascii():
        return False
    if len(value) == 2 and _is_regional(value[0]) and _is_regional(value[1]):
        return True
    if value[0] in "0123456789#*":
        return value[1:] in (_KEYCAP, "️" + _KEYCAP)
    for part in value.split(_ZWJ):
        if not part or not _is_pictograph(part[0]):
            return False
        if any(char not in _VS and char not in _TONES and char not in _TAGS for char in part[1:]):
            return False
    return True


def emoji_name(emoji: str) -> str:
    """A name for an emoji, for a person to read: its Unicode name in lowercase."""
    try:
        return unicodedata.name(emoji[0]).lower()
    except (ValueError, IndexError):
        return "emoji"


def reaction_set(preference: str) -> list[str]:
    """The emoji that the reaction dialog offers, from the ``reaction_emoji`` preference.

    The preference is a list of emoji with spaces between them. A word that is not one emoji
    is ignored, and so is a second copy of an emoji. An empty result gives
    :data:`DEFAULT_REACTIONS`, because a reaction dialog with no rows is of no use.
    """
    seen: list[str] = []
    for word in (preference or "").split():
        if is_reaction_emoji(word) and word not in seen:
            seen.append(word)
    return seen or DEFAULT_REACTIONS.split()


# -- the wire format ----------------------------------------------------------------------


#: The three forms of a reaction on the wire.
MC1 = "mc1"  # MeshCore One: an eight-character hash
MCO = "mco"  # meshcore-open v3: a 16-bit hash, and an index into its table of emoji
MCO_V1 = "mco1"  # meshcore-open v1 (before January 2026): the time and two 30-bit hashes


@dataclass(frozen=True, slots=True)
class Reaction:
    """A reaction, as read from the body of a message.

    Attributes:
        emoji: The emoji of the reaction.
        form: :data:`MC1`, :data:`MCO`, or :data:`MCO_V1`.
        hash: How the reaction names its target. For :data:`MC1`, the eight-character hash
            (refer to :func:`message_hash`). For :data:`MCO`, four lowercase hex digits.
            For :data:`MCO_V1`, the message id of meshcore-open.
        author: For a channel reaction of :data:`MC1`, the name of the author of the
            target. ``None`` in all other cases.
        timestamp: For :data:`MCO_V1`, the timestamp of the target, in seconds.
        author_hash: For :data:`MCO_V1`, the Dart hash of the name of the author.
        body_hash: For :data:`MCO_V1`, the Dart hash of the body of the target.
    """

    emoji: str
    form: str
    hash: str
    author: str | None = None
    timestamp: int | None = None
    author_hash: int | None = None
    body_hash: int | None = None


def parse(body: str, *, channel: bool) -> Reaction | None:
    """Read a message body as a reaction.

    Args:
        body: The body of the message. On a channel, this is the text after the
            ``Name: `` prefix of the sender.
        channel: Whether the message is on a channel. A channel reaction of MeshCore One
            names the author of its target, and a direct one does not.

    Returns:
        The reaction, or ``None`` when the body is a usual message.
    """
    if body.startswith("r:"):
        return _parse_mco(body) or _parse_mco_v1(body)
    head, newline, tail = body.rpartition("\n")
    if not newline:
        return None
    digest = _normal_hash(tail)
    if digest is None:
        return None
    if not channel:
        if "@[" in head or not is_reaction_emoji(head):
            return None
        return Reaction(head, MC1, digest)
    before, mark, after = head.partition("@[")
    if not mark:
        return None
    if before:
        # The order before MeshCore One 1.4.1: the emoji, then the mention.
        if not after.endswith("]"):
            return None
        emoji, author = before, after[:-1]
    else:
        author, close, emoji = after.partition("]")
        if not close:
            return None
    if not author or not is_reaction_emoji(emoji):
        return None
    return Reaction(emoji, MC1, digest, author=author)


def channel_text(emoji: str, author: str, body: str, timestamp: int) -> str:
    """The body of a channel reaction to the message ``body`` of ``author``.

    Args:
        emoji: The emoji of the reaction.
        author: The name of the author of the target, as the target carried it.
        body: The body of the target, without the ``Name: `` prefix.
        timestamp: The timestamp that the author gave the target.
    """
    return f"@[{author}]{emoji}\n{message_hash(body, timestamp)}"


def direct_text(emoji: str, body: str, timestamp: int) -> str:
    """The body of a direct reaction to the message ``body`` sent at ``timestamp``."""
    return f"{emoji}\n{message_hash(body, timestamp)}"


# -- meshcore-open ------------------------------------------------------------------------

#: The table of emoji of meshcore-open (its ``emoji_picker.dart``), in order: the quick set,
#: then the smileys, gestures, hearts, and objects. A reaction of v3 sends an index into it.
# fmt: off
MCO_EMOJI: tuple[str, ...] = (
    "👍", "❤️", "😂", "🎉", "👏", "🔥",
    "😀", "😃", "😄", "😁", "😅", "😂", "🤣", "😊",
    "😇", "🙂", "🙃", "😉", "😌", "😍", "🥰", "😘",
    "😗", "😙", "😚", "😋", "😛", "😝", "😜", "🤪",
    "🤨", "🧐", "🤓", "😎", "🥸", "🤩", "🥳", "😏",
    "😒", "😞", "😔", "😟", "😕", "🙁", "😣", "😖",
    "😫", "😩", "🥺", "😢", "😭", "😤", "😠", "😡",
    "🤬", "🤯", "😳", "🥵", "🥶", "😱", "😨", "😰",
    "😥", "😓", "🤗", "🤔", "🤭", "🤫", "🤥", "😶",
    "👍", "👎", "👊", "✊", "🤛", "🤜", "🤞", "✌️",
    "🤟", "🤘", "👌", "🤌", "🤏", "👈", "👉", "👆",
    "👇", "☝️", "👋", "🤚", "🖐️", "✋", "🖖", "👏",
    "🙌", "👐", "🤲", "🤝", "🙏", "✍️", "💅", "🤳",
    "💪",
    "❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍",
    "🤎", "💔", "❤️‍🔥", "❤️‍🩹", "💕", "💞", "💓", "💗",
    "💖", "💘", "💝", "💟", "💌", "💢", "💥", "💫",
    "💦", "💨", "🕳️", "💬", "👁️‍🗨️", "🗨️", "🗯️", "💭",
    "🎉", "🎊", "🎈", "🎁", "🎀", "🪅", "🪆", "🏆",
    "🥇", "🥈", "🥉", "⚽", "⚾", "🥎", "🏀", "🏐",
    "🏈", "🏉", "🎾", "🥏", "🎳", "🏏", "🏑", "🏒",
    "🥍", "🏓", "🏸", "🥊", "🥋", "🥅", "⛳", "🔥",
    "⭐", "🌟", "✨", "⚡", "💡", "🔦", "🏮", "🪔",
    "📱", "💻", "⌚", "📷", "📺", "📻", "🎵", "🎶",
    "🚀",
)
# fmt: on

_HEX = frozenset("0123456789abcdef")


def _parse_mco(body: str) -> Reaction | None:
    """Read ``r:{hash}:{index}``, a reaction of meshcore-open v3 (four and two hex digits)."""
    if len(body) != 9 or body[6] != ":":
        return None
    digest, index = body[2:6], body[7:]
    if not set(digest + index) <= _HEX:
        return None
    position = int(index, 16)
    if position >= len(MCO_EMOJI):
        return None
    return Reaction(MCO_EMOJI[position], MCO, digest)


def _parse_mco_v1(body: str) -> Reaction | None:
    """Read ``r:{millis}_{author hash}_{body hash}:{emoji}``, a reaction of meshcore-open v1."""
    ident, colon, emoji = body[2:].rpartition(":")
    if not colon or not ident or not emoji:
        return None
    parts = ident.split("_")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    seconds = int(parts[0]) // 1000
    if seconds > 0xFFFFFFFF:
        return None
    return Reaction(
        emoji,
        MCO_V1,
        ident,
        timestamp=seconds,
        author_hash=int(parts[1]),
        body_hash=int(parts[2]),
    )


def dart_hash(units: list[int]) -> int:
    """The ``String.hashCode`` of the Dart VM over UTF-16 code units: a 30-bit value, never 0."""
    mask = 0xFFFFFFFF
    value = 0
    for unit in units:
        value = (value + unit) & mask
        value = (value + (value << 10)) & mask
        value ^= value >> 6
    value = (value + (value << 3)) & mask
    value ^= value >> 11
    value = (value + (value << 15)) & mask
    value &= (1 << 30) - 1
    return value or 1


def _units(text: str) -> list[int]:
    """The UTF-16 code units of ``text``."""
    raw = text.encode("utf-16-le")
    return [int.from_bytes(raw[i : i + 2], "little") for i in range(0, len(raw), 2)]


def mco_hash(timestamp: int, author: str | None, body: str) -> str:
    """The 16-bit hash with which meshcore-open v3 names a target, as four hex digits.

    It is the Dart hash of the timestamp in decimal, the name of the author (on a channel
    only), and the first five UTF-16 code units of the body.
    """
    units = _units(str(timestamp)) + (_units(author) if author else []) + _units(body)[:5]
    return f"{dart_hash(units) & 0xFFFF:04x}"


# -- the fold -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reactor:
    """Who sent a reaction: our node, or another node by the name that it gave.

    Attributes:
        name: The name of the node. ``None`` for our node, and for the peer of a direct
            exchange (the conversation names the peer).
        ours: Whether our node sent the reaction.
    """

    name: str | None
    ours: bool = False


@dataclass(slots=True)
class Badge:
    """One emoji on one message, and the nodes that reacted with it, the first first."""

    emoji: str
    reactors: list[Reactor] = field(default_factory=list)

    @property
    def ours(self) -> bool:
        """Whether our node is one of the nodes that reacted with this emoji."""
        return any(reactor.ours for reactor in self.reactors)


@dataclass(slots=True)
class _Target:
    """A message of the transcript, as the fold matches reactions against it."""

    message: ChatMessage
    author: str | None
    body: str
    timestamp: int


class ReactionFold:
    """Fold the reactions of one conversation onto the messages that they react to.

    Give each message of the transcript to :meth:`take`, in the order of arrival. A message
    that is not a reaction becomes a target, and :meth:`take` returns ``False``. A reaction
    attaches to its target, and :meth:`take` returns ``True``. Then the transcript does not
    show it as a message. A reaction whose target is not known yet waits (at most
    :data:`PENDING_CAP` of them), because on a mesh a reaction can arrive before its target.
    :meth:`badges` gives the reactions on one message.

    The fold counts each emoji one time for each node. A node can send the same reaction
    again, and a repeater can deliver a copy two times.
    """

    def __init__(self, *, channel: bool, self_name: str | None = None) -> None:
        """Make an empty fold.

        Args:
            channel: Whether the conversation is a channel. A channel reaction names the
                author of its target, and a direct one does not.
            self_name: The name of our node. On a channel, it is the author of each message
                that we sent. Without it, the reactions to our channel messages of
                MeshCore One cannot find their target.
        """
        self._channel = channel
        self._self_name = self_name
        self._targets: list[_Target] = []
        self._by_id: dict[int, _Target] = {}
        self._by_hash: dict[tuple[str | None, str], _Target] = {}
        self._badges: dict[int, list[Badge]] = {}
        self._pending: list[tuple[Reaction, Reactor]] = []

    def take(self, message: ChatMessage) -> bool:
        """Fold one message in, and return whether it is a reaction.

        Args:
            message: The next message of the transcript.

        Returns:
            ``True`` when the message is a reaction. The transcript then does not show it.
            ``False`` when it is a usual message, which other reactions can now name.
        """
        author, body = self._author_and_body(message)
        reaction = parse(body, channel=self._channel)
        if reaction is None:
            self._index(message, author, body)
            return False
        reactor = Reactor(
            None if message.outbound else (author if self._channel else None), ours=message.outbound
        )
        target = self._find(reaction, wire_timestamp(message.created_at))
        if target is not None:
            self._attach(target.message, reaction.emoji, reactor)
        elif reaction.form == MC1:
            self._pending.append((reaction, reactor))
            del self._pending[:-PENDING_CAP]
        return True

    def badges(self, message: ChatMessage) -> list[Badge]:
        """The reactions on ``message``: the most frequent emoji first, then the earliest."""
        badges = self._badges.get(id(message), [])
        order = {id(badge): index for index, badge in enumerate(badges)}
        return sorted(badges, key=lambda badge: (-len(badge.reactors), order[id(badge)]))

    def count(self, message: ChatMessage) -> int:
        """The number of reactions on ``message``. A render cache compares it to see a change."""
        return sum(len(badge.reactors) for badge in self._badges.get(id(message), []))

    def reacted(self, message: ChatMessage, emoji: str) -> bool:
        """Whether our node already reacted to ``message`` with ``emoji``."""
        return any(
            badge.emoji == emoji and badge.ours for badge in self._badges.get(id(message), [])
        )

    def target_of(self, message: ChatMessage) -> tuple[str | None, str, int] | None:
        """The author, the body, and the timestamp with which a reaction names ``message``.

        ``None`` when ``message`` is not a target of this fold (a reaction, or a message
        that the fold never took).
        """
        target = self._by_id.get(id(message))
        if target is None or target.message is not message:
            return None
        return target.author, target.body, target.timestamp

    def _author_and_body(self, message: ChatMessage) -> tuple[str | None, str]:
        """The author and the body of a message, as the hash of a reaction sees them.

        On a channel, the body of a received message is the text after the ``Name: ``
        prefix, without the spaces at its two ends. Our channel messages have no prefix in
        the history, and their author is our node.
        """
        if not self._channel:
            return None, message.text
        if message.outbound:
            return self._self_name, message.text.strip(" \t")
        name, body = split_channel_sender(message.text)
        return name, body.strip(" \t")

    def _index(self, message: ChatMessage, author: str | None, body: str) -> None:
        """Make ``message`` a target, and attach the reactions that waited for it."""
        target = _Target(message, author, body, wire_timestamp(message.created_at))
        self._targets.append(target)
        self._by_id[id(message)] = target
        if self._channel and not author:
            return  # a channel reaction of MeshCore One must name the author
        key = (author if self._channel else None, message_hash(body, target.timestamp))
        self._by_hash[key] = target
        waiting = [entry for entry in self._pending if (entry[0].author, entry[0].hash) == key]
        if waiting:
            self._pending = [entry for entry in self._pending if entry not in waiting]
            for reaction, reactor in waiting:
                self._attach(message, reaction.emoji, reactor)

    def _find(self, reaction: Reaction, at: int) -> _Target | None:
        """The target that ``reaction`` names, or ``None``. The newest target wins a tie."""
        if reaction.form == MC1:
            return self._by_hash.get((reaction.author, reaction.hash))
        if reaction.form == MCO:
            anchor = at
        else:
            anchor = reaction.timestamp or 0
        for target in reversed(self._targets):
            if abs(target.timestamp - anchor) > WINDOW_S:
                continue
            if reaction.form == MCO:
                author = target.author if self._channel else None
                if mco_hash(target.timestamp, author, target.body) == reaction.hash:
                    return target
            elif dart_hash(_units(target.body)) == reaction.body_hash:
                if self._channel and target.author is not None:
                    if dart_hash(_units(target.author)) != reaction.author_hash:
                        continue
                return target
        return None

    def _attach(self, message: ChatMessage, emoji: str, reactor: Reactor) -> None:
        """Add one reaction to ``message``, if that node did not already send that emoji."""
        badges = self._badges.setdefault(id(message), [])
        badge = next((b for b in badges if b.emoji == emoji), None)
        if badge is None:
            badge = Badge(emoji)
            badges.append(badge)
        if reactor not in badge.reactors:
            badge.reactors.append(reactor)
