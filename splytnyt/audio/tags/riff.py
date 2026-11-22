"""RIFF INFO and ID3 chunk writer for WAV files.

Our own implementation, because Mutagen is GPL-2.0+ and linking it would
force this application's licence (see ``docs/ARCHITECTURE.md`` section 2).

A RIFF/WAVE file is a list of chunks::

    "RIFF" | size | "WAVE" | "fmt " chunk | ... | "data" chunk

This writer inserts two chunks right behind the ``fmt `` chunk:

* a ``LIST`` chunk of type ``INFO`` holding the classic INFO fields,
* an ``id3 `` chunk holding the full ID3v2.4 tag, so fields that RIFF has no
  room for (ReplayGain, custom fields, lyrics) survive as well.

The file is rewritten atomically through ``<path>.partial``.
"""

from __future__ import annotations

import os
import struct

from ..encode import atomic_output
from .flac import copy_range
from .id3v2 import build_id3v2

#: Canonical tag key -> RIFF INFO id.
_INFO_IDS: dict[str, str] = {
    "TITLE": "INAM",
    "ARTIST": "IART",
    "ALBUM": "IPRD",
    "COMMENT": "ICMT",
    "GENRE": "IGNR",
    "DATE": "ICRD",
    "TRACKNUMBER": "ITRK",
    "DISCNUMBER": "IPRT",
}


def write_riff_info(
    wav_path: str, tags: dict[str, str], artwork_path: str = ""
) -> None:
    """Insert a LIST/INFO chunk and an ``id3 `` chunk into a WAV file.

    Args:
        wav_path: The WAV file to tag.
        tags: Comment fields. The classic INFO fields are written into the
            LIST/INFO chunk, every field is additionally written into the
            ``id3 `` chunk.
        artwork_path: Optional cover image, written into the ``id3 `` chunk.
    """
    info_chunk = _build_info_chunk(tags)
    id3_chunk = _build_id3_chunk(build_id3v2(tags, artwork_path))
    chunks, size = _read_riff_chunks(wav_path)

    insert_at = 0
    for index, (chunk_id, _offset, _length) in enumerate(chunks):
        if chunk_id == b"fmt ":
            insert_at = index + 1
            break

    with atomic_output(wav_path, with_handle=False) as (temp, _handle):
        with open(temp, "wb") as out:
            out.write(b"RIFF")
            out.write(b"\x00\x00\x00\x00")  # size, patched below
            out.write(b"WAVE")
            for index, (_chunk_id, offset, length) in enumerate(chunks):
                if index == insert_at:
                    if info_chunk:
                        out.write(info_chunk)
                    if id3_chunk:
                        out.write(id3_chunk)
                copy_range(wav_path, offset, out, limit=8 + length + (length & 1))
            if insert_at >= len(chunks):
                if info_chunk:
                    out.write(info_chunk)
                if id3_chunk:
                    out.write(id3_chunk)
            total = out.tell()
            out.seek(4)
            out.write(struct.pack("<I", total - 8))


def _build_info_chunk(tags: dict[str, str]) -> bytes:
    """Build the ``LIST``/``INFO`` chunk, or empty bytes when there is none."""
    fields: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, value in tags.items():
        name = str(key).strip().upper()
        info_id = _INFO_IDS.get(name)
        if info_id is None or info_id in seen:
            continue
        seen.add(info_id)
        fields.append((info_id, str(value)))

    if not fields:
        return b""

    payload = bytearray(b"INFO")
    for info_id, value in fields:
        data = value.encode("utf-8") + b"\x00"
        payload += info_id.encode("ascii")
        payload += struct.pack("<I", len(data))
        payload += data
        if len(data) & 1:
            payload += b"\x00"
    return b"LIST" + struct.pack("<I", len(payload)) + bytes(payload)


def _build_id3_chunk(tag: bytes) -> bytes:
    """Build the ``id3 `` chunk holding an ID3v2.4 tag."""
    if not tag:
        return b""
    chunk = b"id3 " + struct.pack("<I", len(tag)) + tag
    if len(tag) & 1:
        chunk += b"\x00"
    return chunk


def _read_riff_chunks(path: str) -> tuple[list[tuple[bytes, int, int]], int]:
    """Return the chunks of a RIFF/WAVE file and the file size."""
    size = os.path.getsize(path)
    chunks: list[tuple[bytes, int, int]] = []
    with open(path, "rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"Not a RIFF/WAVE file: {path}")
        offset = 12
        while offset + 8 <= size:
            handle.seek(offset)
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            length = int.from_bytes(chunk_header[4:8], "little")
            chunks.append((chunk_id, offset, length))
            offset += 8 + length + (length & 1)
    return chunks, size
