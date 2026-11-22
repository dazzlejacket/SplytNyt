"""ID3v2.4 tag writer.

Our own implementation, because Mutagen is GPL-2.0+ and linking it would
force this application's licence (see ``docs/ARCHITECTURE.md`` section 2).

The tag is written in front of the MPEG audio frames::

    "ID3" | 0x04 0x00 | 0x00 | syncsafe size | frames | terminator

Frame sizes are syncsafe (seven bits per byte), text is stored as UTF-8
(encoding byte 0x03) and every frame is terminated by a frame header whose id
is four zero bytes.
"""

from __future__ import annotations

from ..encode import atomic_output
from .flac import copy_range, describe_image

#: Canonical tag key -> ID3v2 frame id. ``TXXX:NAME`` means a TXXX frame
#: whose description is ``NAME``.
FRAME_MAP: dict[str, str] = {
    "TITLE": "TIT2",
    "ARTIST": "TPE1",
    "ALBUM": "TALB",
    "ALBUMARTIST": "TPE2",
    "COMPOSER": "TCOM",
    "GENRE": "TCON",
    "DATE": "TDRC",
    "TRACKNUMBER": "TRCK",
    "DISCNUMBER": "TPOS",
    "COMMENT": "COMM",
    "LYRICS": "USLT",
    "PUBLISHER": "TPUB",
    "CONDUCTOR": "TPE3",
    "PERFORMER": "TPE4",
    "PRODUCTION": "TXXX:PRODUCTION",
    "ACT": "TXXX:ACT",
    "SONG": "TXXX:SONG",
    "PERFORMANCE_DATE": "TXXX:PERFORMANCEDATE",
    "VENUE": "TXXX:VENUE",
    "CITY": "TXXX:CITY",
    "COUNTRY": "TXXX:COUNTRY",
    "TOUR": "TXXX:TOUR",
    "SOURCE_NOTES": "TXXX:SOURCENOTES",
    "EQUIPMENT": "TXXX:EQUIPMENT",
    "LINEAGE": "TXXX:LINEAGE",
    "IDENTIFIER": "TXXX:IDENTIFIER",
    "REPLAYGAIN_TRACK_GAIN": "TXXX:REPLAYGAIN_TRACK_GAIN",
    "REPLAYGAIN_TRACK_PEAK": "TXXX:REPLAYGAIN_TRACK_PEAK",
    "REPLAYGAIN_ALBUM_GAIN": "TXXX:REPLAYGAIN_ALBUM_GAIN",
    "REPLAYGAIN_ALBUM_PEAK": "TXXX:REPLAYGAIN_ALBUM_PEAK",
}

#: Text encoding byte for UTF-8.
_ENCODING_UTF8 = 0x03

#: Picture type 3 is "cover (front)".
_PICTURE_TYPE_FRONT = 0x03


def map_tag_key_to_frame(key: str) -> str | None:
    """Map a canonical tag key to an ID3v2 frame id.

    Returns ``None`` for keys that have no dedicated frame; the caller then
    falls back to a TXXX frame.
    """
    if not key:
        return None
    return FRAME_MAP.get(str(key).strip().upper())


def build_id3v2(tags: dict[str, str], artwork_path: str = "") -> bytes:
    """Build a complete ID3v2.4 tag.

    Args:
        tags: Comment fields, keyed by canonical tag key.
        artwork_path: Optional cover image, stored as an APIC frame.

    Returns:
        The tag bytes, ready to be written in front of the audio.
    """
    frames = bytearray()
    for key, value in tags.items():
        frame = _build_frame(key, value)
        if frame:
            frames += frame
    if artwork_path:
        frames += _build_apic(artwork_path)
    frames += b"\x00" * 10  # frame terminator

    header = b"ID3" + bytes([0x04, 0x00, 0x00]) + _syncsafe(len(frames))
    return header + bytes(frames)


def write_id3v2(
    mp3_path: str, tags: dict[str, str], artwork_path: str = ""
) -> None:
    """Write an ID3v2.4 tag in front of the MPEG audio frames.

    The file is rewritten atomically: the tag and the original audio are
    written to ``<path>.partial`` and renamed onto the original afterwards.
    An existing ID3v2 tag is replaced, not duplicated.

    Args:
        mp3_path: The MP3 file to tag.
        tags: Comment fields, keyed by canonical tag key.
        artwork_path: Optional cover image, stored as an APIC frame.
    """
    tag = build_id3v2(tags, artwork_path)
    audio_offset = _existing_tag_size(mp3_path)
    with atomic_output(mp3_path) as (_temp, handle):
        if handle is None:  # pragma: no cover - atomic_output always opens one
            raise RuntimeError("No output handle for the ID3v2 writer")
        handle.write(tag)
        copy_range(mp3_path, audio_offset, handle)


def _build_frame(key: str, value: str) -> bytes:
    """Build one ID3v2 frame for a tag field."""
    name = str(key).strip().upper()
    if not name:
        return b""
    text = "" if value is None else str(value)
    mapped = map_tag_key_to_frame(name)
    if mapped is None:
        return _frame("TXXX", _txxx_body(name, text))
    if mapped.startswith("TXXX:"):
        return _frame("TXXX", _txxx_body(mapped.split(":", 1)[1], text))
    if mapped in ("COMM", "USLT"):
        # language (3 bytes), empty content descriptor, then the text
        body = bytes([_ENCODING_UTF8]) + b"eng" + b"\x00" + text.encode("utf-8")
        return _frame(mapped, body)
    return _frame(mapped, bytes([_ENCODING_UTF8]) + text.encode("utf-8"))


def _txxx_body(description: str, text: str) -> bytes:
    """Body of a TXXX frame: a description and a value, both UTF-8."""
    return (
        bytes([_ENCODING_UTF8])
        + description.encode("utf-8")
        + b"\x00"
        + text.encode("utf-8")
    )


def _build_apic(artwork_path: str) -> bytes:
    """Build an APIC frame holding the cover image."""
    with open(artwork_path, "rb") as handle:
        data = handle.read()
    mime, _width, _height, _depth = describe_image(artwork_path, data)
    body = (
        bytes([_ENCODING_UTF8])
        + mime.encode("ascii")
        + b"\x00"
        + bytes([_PICTURE_TYPE_FRONT])
        + b"\x00"  # empty description
        + data
    )
    return _frame("APIC", body)


def _frame(frame_id: str, body: bytes) -> bytes:
    """Frame header plus body."""
    return frame_id.encode("ascii") + _syncsafe(len(body)) + b"\x00\x00" + body


def _syncsafe(value: int) -> bytes:
    """Encode a length as four syncsafe bytes (seven bits per byte)."""
    value = int(value)
    if value < 0 or value > 0x0FFFFFFF:
        raise ValueError(f"Value does not fit into a syncsafe integer: {value}")
    return bytes(
        [
            (value >> 21) & 0x7F,
            (value >> 14) & 0x7F,
            (value >> 7) & 0x7F,
            value & 0x7F,
        ]
    )


def _existing_tag_size(path: str) -> int:
    """Size of an ID3v2 tag at the start of the file, or 0 when there is none."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(10)
            if len(header) < 10 or header[:3] != b"ID3":
                return 0
            size = 0
            for byte in header[6:10]:
                size = (size << 7) | (byte & 0x7F)
            return 10 + size
    except OSError:
        return 0
