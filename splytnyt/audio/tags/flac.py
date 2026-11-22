"""Vorbis comment and picture blocks for FLAC files.

These writers are ours on purpose: Mutagen is GPL-2.0+, and linking it would
force this application's licence (see ``docs/ARCHITECTURE.md`` section 2).

The FLAC container is a chain of metadata blocks::

    "fLaC" | block header | block data | block header | block data | audio

Every block header is one byte (the last-block flag in bit 7, the block type
in bits 6..0) followed by a 24 bit big-endian length. The block types used
here are:

* 0 ``STREAMINFO`` - kept byte for byte, it describes the audio,
* 4 ``VORBIS_COMMENT`` - the tags,
* 6 ``PICTURE`` - the cover art, in the RFC 9639 layout.
"""

from __future__ import annotations

import os
import struct

from ..encode import atomic_output

#: Block type of the mandatory STREAMINFO block.
METADATA_STREAMINFO = 0
#: Block type of the Vorbis comment block.
METADATA_VORBIS_COMMENT = 4
#: Block type of the picture block.
METADATA_PICTURE = 6

#: Vendor string written into every comment block.
_VENDOR = "SplytNyt"

#: Picture type 3 is "cover (front)".
_PICTURE_TYPE_FRONT = 3


def write_vorbis_comments(
    flac_path: str, tags: dict[str, str], artwork_path: str = ""
) -> None:
    """Write Vorbis comments (and optionally a picture) into a FLAC file.

    The file is rewritten atomically: the new metadata is written to
    ``<path>.partial`` and renamed onto the original afterwards. The
    STREAMINFO block and the audio frames are copied unchanged.

    Args:
        flac_path: The FLAC file to tag.
        tags: Comment fields. Keys are uppercased; a key that is not printable
            ASCII, or that contains ``=``, is rejected.
        artwork_path: Optional cover image. When given, it replaces an
            existing picture block; when omitted, an existing picture block
            is kept.
    """
    blocks, audio_offset = _read_metadata_blocks(flac_path)
    comment = _build_vorbis_comment(tags)
    picture = _build_picture(artwork_path) if artwork_path else None

    new_blocks: list[tuple[int, bytes]] = []
    comment_written = False
    picture_written = False
    for block_type, payload in blocks:
        if block_type == METADATA_VORBIS_COMMENT:
            if not comment_written:
                new_blocks.append((METADATA_VORBIS_COMMENT, comment))
                comment_written = True
            continue
        if block_type == METADATA_PICTURE:
            if picture is not None:
                # A FLAC file may carry only one picture block; the new
                # artwork replaces the old one.
                if not picture_written:
                    new_blocks.append((METADATA_PICTURE, picture))
                    picture_written = True
                continue
            new_blocks.append((block_type, payload))
            continue
        new_blocks.append((block_type, payload))
    if not comment_written:
        new_blocks.append((METADATA_VORBIS_COMMENT, comment))
    if picture is not None and not picture_written:
        new_blocks.append((METADATA_PICTURE, picture))

    with atomic_output(flac_path, with_handle=False) as (temp, _handle):
        with open(temp, "wb") as out:
            out.write(b"fLaC")
            last = len(new_blocks) - 1
            for index, (block_type, payload) in enumerate(new_blocks):
                flag = 0x80 if index == last else 0x00
                out.write(bytes([flag | (block_type & 0x7F)]))
                out.write(len(payload).to_bytes(3, "big"))
                out.write(payload)
            copy_range(flac_path, audio_offset, out)


def _read_metadata_blocks(path: str) -> tuple[list[tuple[int, bytes]], int]:
    """Return the metadata blocks of a FLAC file and the audio offset."""
    blocks: list[tuple[int, bytes]] = []
    with open(path, "rb") as handle:
        if handle.read(4) != b"fLaC":
            raise ValueError(f"Not a FLAC file: {path}")
        while True:
            header = handle.read(4)
            if len(header) < 4:
                raise ValueError(f"Truncated FLAC metadata in {path}")
            is_last = bool(header[0] & 0x80)
            block_type = header[0] & 0x7F
            length = int.from_bytes(header[1:4], "big")
            payload = handle.read(length)
            if len(payload) < length:
                raise ValueError(f"Truncated FLAC metadata in {path}")
            blocks.append((block_type, payload))
            if is_last:
                break
        offset = handle.tell()
    return blocks, offset


def _build_vorbis_comment(tags: dict[str, str]) -> bytes:
    """Build the payload of a Vorbis comment block."""
    vendor = _VENDOR.encode("utf-8")
    comments: list[bytes] = []
    for key, value in tags.items():
        name = str(key).strip().upper()
        if not name:
            continue
        if not _is_printable_ascii(name):
            raise ValueError(
                f"Invalid Vorbis comment key {key!r}: keys must be printable ASCII."
            )
        if "=" in name:
            raise ValueError(
                f"Invalid Vorbis comment key {key!r}: keys must not contain '='."
            )
        comments.append(f"{name}={value}".encode("utf-8"))

    out = bytearray()
    out += struct.pack("<I", len(vendor)) + vendor
    out += struct.pack("<I", len(comments))
    for comment in comments:
        out += struct.pack("<I", len(comment)) + comment
    return bytes(out)


def _build_picture(artwork_path: str) -> bytes:
    """Build the payload of an RFC 9639 picture block."""
    with open(artwork_path, "rb") as handle:
        data = handle.read()
    mime, width, height, depth = describe_image(artwork_path, data)
    out = bytearray()
    out += struct.pack(">I", _PICTURE_TYPE_FRONT)
    out += struct.pack(">I", len(mime)) + mime.encode("ascii")
    out += struct.pack(">I", 0)  # empty description
    out += struct.pack(">I", width)
    out += struct.pack(">I", height)
    out += struct.pack(">I", depth)
    out += struct.pack(">I", 0)  # not an indexed-colour image
    out += struct.pack(">I", len(data)) + data
    return bytes(out)


def describe_image(path: str, data: bytes) -> tuple[str, int, int, int]:
    """Return ``(mime_type, width, height, colour_depth)`` of an image.

    The header of PNG and JPEG files is parsed directly, so no image library
    is needed. Unknown formats fall back to a generic MIME type and a zero
    size, which is what the containers expect for "unknown".
    """
    mime = _guess_mime(path, data)
    width = height = depth = 0
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 33:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        bit_depth = data[24]
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(data[25], 1)
        depth = bit_depth * channels
    elif data[:2] == b"\xff\xd8":
        width, height, depth = _jpeg_size(data)
    return mime, width, height, depth


def _jpeg_size(data: bytes) -> tuple[int, int, int]:
    """Return ``(width, height, colour_depth)`` of a JPEG file."""
    position = 2
    size = len(data)
    while position + 9 < size:
        if data[position] != 0xFF:
            position += 1
            continue
        marker = data[position + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            position += 2
            continue
        length = int.from_bytes(data[position + 2 : position + 4], "big")
        if marker in (
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        ):
            height = int.from_bytes(data[position + 5 : position + 7], "big")
            width = int.from_bytes(data[position + 7 : position + 9], "big")
            depth = data[position + 9] * 3
            return width, height, depth
        position += 2 + length
    return 0, 0, 0


def _guess_mime(path: str, data: bytes) -> str:
    """Best effort MIME type of an image file."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    extension = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(extension, "application/octet-stream")


def copy_range(source_path: str, offset: int, destination, limit: int | None = None) -> None:
    """Copy a file from ``offset`` into ``destination``.

    ``limit`` caps the number of bytes copied; without it the rest of the
    file is copied.
    """
    remaining = limit
    with open(source_path, "rb") as source:
        source.seek(offset)
        while remaining is None or remaining > 0:
            size = 1 << 16 if remaining is None else min(1 << 16, remaining)
            chunk = source.read(size)
            if not chunk:
                break
            destination.write(chunk)
            if remaining is not None:
                remaining -= len(chunk)


def _is_printable_ascii(text: str) -> bool:
    """True when every character is printable ASCII."""
    return all(0x20 <= ord(char) <= 0x7E for char in text)
