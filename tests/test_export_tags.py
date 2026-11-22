"""Tests for the export encoders and the in-house tag writers.

The tests use real files in a temporary directory: a synthetic stereo
signal is written with libsndfile, exported with our own writers and read
back again. Nothing is mocked except the failure path of the atomic rename.
"""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np
import pytest
import soundfile as sf

from splytnyt.audio.decode import AudioSource
from splytnyt.audio.encode import (
    CueTrack,
    WriteResult,
    write_cue,
    write_flac,
    write_mp3,
    write_wav,
)
from splytnyt.audio.tags import (
    map_tag_key_to_frame,
    write_id3v2,
    write_riff_info,
    write_vorbis_comments,
)
from splytnyt.core.units import frames_to_tc

SAMPLE_RATE = 44100
CHANNELS = 2
SOURCE_FRAMES = SAMPLE_RATE  # one second


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _stereo_signal(frames: int = SOURCE_FRAMES) -> np.ndarray:
    """Two different sine tones, left and right."""
    time = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    left = 0.5 * np.sin(2.0 * np.pi * 440.0 * time)
    right = 0.25 * np.sin(2.0 * np.pi * 660.0 * time + 0.7)
    return np.stack([left, right], axis=1).astype(np.float32)


def _make_png(width: int = 8, height: int = 6, colour=(200, 30, 40)) -> bytes:
    """Build a small valid PNG (8 bit RGB) without an image library."""
    raw = b"".join(b"\x00" + bytes(colour) * width for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _syncsafe(data: bytes) -> int:
    """Decode four syncsafe bytes."""
    value = 0
    for byte in data:
        value = (value << 7) | (byte & 0x7F)
    return value


def _flac_blocks(path: str) -> list[tuple[int, bytes]]:
    """Parse the metadata block chain of a FLAC file."""
    blocks: list[tuple[int, bytes]] = []
    with open(path, "rb") as handle:
        assert handle.read(4) == b"fLaC"
        while True:
            header = handle.read(4)
            assert len(header) == 4
            is_last = bool(header[0] & 0x80)
            block_type = header[0] & 0x7F
            length = int.from_bytes(header[1:4], "big")
            blocks.append((block_type, handle.read(length)))
            if is_last:
                break
    return blocks


def _vorbis_comments(payload: bytes) -> dict[str, str]:
    """Parse a Vorbis comment block payload."""
    vendor_length = int.from_bytes(payload[0:4], "little")
    offset = 4 + vendor_length
    count = int.from_bytes(payload[offset : offset + 4], "little")
    offset += 4
    comments: dict[str, str] = {}
    for _ in range(count):
        length = int.from_bytes(payload[offset : offset + 4], "little")
        offset += 4
        text = payload[offset : offset + length].decode("utf-8")
        offset += length
        key, _, value = text.partition("=")
        comments[key] = value
    return comments


def _picture(payload: bytes) -> dict:
    """Parse an RFC 9639 picture block payload."""
    offset = 0
    picture_type = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    mime_length = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    mime = payload[offset : offset + mime_length].decode("ascii")
    offset += mime_length
    description_length = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    description = payload[offset : offset + description_length].decode("utf-8")
    offset += description_length
    width = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    height = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    depth = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    colours = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    data_length = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    return {
        "type": picture_type,
        "mime": mime,
        "description": description,
        "width": width,
        "height": height,
        "depth": depth,
        "colours": colours,
        "data": payload[offset : offset + data_length],
    }


def _riff_chunks(path: str) -> list[tuple[bytes, bytes]]:
    """Parse the chunk list of a RIFF/WAVE file."""
    chunks: list[tuple[bytes, bytes]] = []
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        header = handle.read(12)
        assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
        offset = 12
        while offset + 8 <= size:
            handle.seek(offset)
            chunk_header = handle.read(8)
            chunk_id = chunk_header[:4]
            length = int.from_bytes(chunk_header[4:8], "little")
            chunks.append((chunk_id, handle.read(length)))
            offset += 8 + length + (length & 1)
    return chunks


def _riff_info(path: str) -> dict[str, str]:
    """Read the INFO fields of a RIFF/WAVE file."""
    info: dict[str, str] = {}
    for chunk_id, payload in _riff_chunks(path):
        if chunk_id != b"LIST" or payload[:4] != b"INFO":
            continue
        offset = 4
        while offset + 8 <= len(payload):
            field_id = payload[offset : offset + 4].decode("ascii")
            length = int.from_bytes(payload[offset + 4 : offset + 8], "little")
            value = payload[offset + 8 : offset + 8 + length]
            info[field_id] = value.rstrip(b"\x00").decode("utf-8")
            offset += 8 + length + (length & 1)
    return info


def _id3v2_frames(raw: bytes) -> tuple[dict[str, list[bytes]], int]:
    """Parse an ID3v2.4 tag and return its frames and its size."""
    assert raw[:3] == b"ID3"
    assert raw[3:5] == b"\x04\x00", "expected an ID3v2.4 tag"
    assert raw[5] == 0x00, "expected no tag flags"
    size = _syncsafe(raw[6:10])
    body = raw[10 : 10 + size]
    frames: dict[str, list[bytes]] = {}
    offset = 0
    while offset + 10 <= len(body):
        frame_id = body[offset : offset + 4]
        if frame_id == b"\x00\x00\x00\x00":
            break
        frame_size = _syncsafe(body[offset + 4 : offset + 8])
        payload = body[offset + 10 : offset + 10 + frame_size]
        frames.setdefault(frame_id.decode("ascii"), []).append(payload)
        offset += 10 + frame_size
    return frames, size


@pytest.fixture()
def source_file(tmp_path) -> str:
    """A one second stereo PCM_16 recording."""
    path = str(tmp_path / "source.wav")
    sf.write(path, _stereo_signal(), SAMPLE_RATE, subtype="PCM_16")
    return path


@pytest.fixture()
def source(source_file) -> AudioSource:
    return AudioSource(source_file)


# ----------------------------------------------------------------------
# FLAC
# ----------------------------------------------------------------------
def test_flac_round_trip_is_bit_exact(source, source_file, tmp_path):
    target = str(tmp_path / "track.flac")
    start, end = 1000, 20000

    result = write_flac(target, [(source, start, end)])

    assert isinstance(result, WriteResult)
    assert result.frames_written == end - start
    assert result.sample_rate == SAMPLE_RATE
    assert result.channels == CHANNELS
    assert result.encoder_delay == 0 and result.encoder_padding == 0
    assert result.bytes_written == os.path.getsize(target)

    data, sample_rate = sf.read(target, always_2d=True, dtype="float32")
    assert sample_rate == SAMPLE_RATE
    assert data.shape == (end - start, CHANNELS)
    assert sf.info(target).subtype == "PCM_16"

    expected, _ = sf.read(
        source_file, start=start, frames=end - start, always_2d=True, dtype="float32"
    )
    # PCM_16 in, PCM_16 out: the samples are bit exact.
    assert np.array_equal(data, expected)
    assert float(np.abs(data - expected).max()) == 0.0


def test_flac_multi_part_joins_without_gap(source, source_file, tmp_path):
    target = str(tmp_path / "joined.flac")
    first = (source, 0, 5000)
    second = (source, 20000, 30000)

    result = write_flac(target, [first, second])

    assert result.frames_written == 5000 + 10000
    data, _ = sf.read(target, always_2d=True, dtype="float32")
    assert data.shape == (15000, CHANNELS)

    whole, _ = sf.read(source_file, always_2d=True, dtype="float32")
    expected = np.concatenate([whole[0:5000], whole[20000:30000]], axis=0)
    assert np.array_equal(data, expected)


def test_flac_preserves_the_source_bit_depth(source_file, tmp_path):
    path = str(tmp_path / "source24.wav")
    sf.write(path, _stereo_signal(1000), SAMPLE_RATE, subtype="PCM_24")
    source = AudioSource(path)

    target = str(tmp_path / "track24.flac")
    result = write_flac(target, [(source, 0, 1000)])

    assert sf.info(target).subtype == "PCM_24"
    assert not any("24-bit" in note for note in result.notes)


# ----------------------------------------------------------------------
# WAV and RIFF INFO
# ----------------------------------------------------------------------
def test_wav_round_trip_and_riff_info(source, source_file, tmp_path):
    target = str(tmp_path / "track.wav")
    start, end = 0, 12000

    result = write_wav(target, [(source, start, end)])

    assert result.frames_written == end - start
    assert result.sample_rate == SAMPLE_RATE
    assert result.channels == CHANNELS

    data, sample_rate = sf.read(target, always_2d=True, dtype="float32")
    assert sample_rate == SAMPLE_RATE
    assert data.shape == (end - start, CHANNELS)
    expected, _ = sf.read(
        source_file, start=start, frames=end - start, always_2d=True, dtype="float32"
    )
    assert np.array_equal(data, expected)

    write_riff_info(target, {"TITLE": "Song", "ARTIST": "Band", "ALBUM": "Record"})

    info = _riff_info(target)
    assert info["INAM"] == "Song"
    assert info["IART"] == "Band"
    assert info["IPRD"] == "Record"

    # The audio is untouched by the tagging.
    tagged, sample_rate = sf.read(target, always_2d=True, dtype="float32")
    assert sample_rate == SAMPLE_RATE
    assert np.array_equal(tagged, data)

    # The full tag also travels in the "id3 " chunk.
    id3_chunks = [payload for chunk_id, payload in _riff_chunks(target) if chunk_id == b"id3 "]
    assert len(id3_chunks) == 1
    frames, _size = _id3v2_frames(id3_chunks[0])
    assert frames["TIT2"][0][1:].decode("utf-8") == "Song"


# ----------------------------------------------------------------------
# MP3
# ----------------------------------------------------------------------
def test_mp3_export_reports_delay_and_padding(source, source_file, tmp_path):
    target = str(tmp_path / "track.mp3")

    result = write_mp3(target, [(source, 0, SOURCE_FRAMES)])

    assert os.path.exists(target)
    assert os.path.getsize(target) > 0
    assert result.bytes_written == os.path.getsize(target)
    assert result.frames_written == SOURCE_FRAMES
    assert result.sample_rate == SAMPLE_RATE
    assert result.channels == CHANNELS

    # The encoder delay and the padding residual are reported.
    assert result.encoder_delay == 576
    assert result.encoder_padding > 0
    notes = " ".join(result.notes).lower()
    assert "delay" in notes
    assert "padding" in notes
    assert "residual" in notes
    assert "not sample-accurate" in notes

    decoded, sample_rate = sf.read(target, always_2d=True, dtype="float32")
    assert sample_rate == SAMPLE_RATE

    # The decoded stream is the source plus the encoder delay and the
    # padding; within one MPEG frame (1152 samples, ~26 ms at 44.1 kHz).
    expected = result.frames_written + result.encoder_padding
    assert abs(len(decoded) - expected) <= 1152

    # The audio itself starts after the gapless offset.
    original, _ = sf.read(source_file, always_2d=True, dtype="float32")
    best_offset = None
    best_error = None
    for offset in range(1000, 1250):
        window = decoded[offset : offset + 4000, 0]
        error = float(np.mean(np.abs(window - original[:4000, 0])))
        if best_error is None or error < best_error:
            best_offset, best_error = offset, error
    assert best_offset is not None
    assert abs(best_offset - (result.encoder_delay + 529)) <= 2
    assert best_error is not None and best_error < 0.05


# ----------------------------------------------------------------------
# Vorbis comments
# ----------------------------------------------------------------------
def test_vorbis_comments_round_trip_and_picture_block(tmp_path):
    target = str(tmp_path / "tagged.flac")
    sf.write(target, _stereo_signal(4096), SAMPLE_RATE, subtype="PCM_16")
    audio_before, _ = sf.read(target, always_2d=True, dtype="float32")

    write_vorbis_comments(
        target, {"TITLE": "Titel", "ARTIST": "Künstler", "ALBUM": "Album"}
    )

    blocks = _flac_blocks(target)
    assert blocks[0][0] == 0, "STREAMINFO must stay the first block"
    comment_blocks = [payload for block_type, payload in blocks if block_type == 4]
    assert len(comment_blocks) == 1
    comments = _vorbis_comments(comment_blocks[0])
    assert comments["TITLE"] == "Titel"
    assert comments["ARTIST"] == "Künstler"
    assert comments["ALBUM"] == "Album"

    # The audio survives the rewrite.
    audio_after, sample_rate = sf.read(target, always_2d=True, dtype="float32")
    assert sample_rate == SAMPLE_RATE
    assert np.array_equal(audio_after, audio_before)

    # A picture block is added when artwork is supplied.
    cover = str(tmp_path / "cover.png")
    png = _make_png(8, 6)
    with open(cover, "wb") as handle:
        handle.write(png)

    write_vorbis_comments(target, {"TITLE": "Titel"}, artwork_path=cover)

    blocks = _flac_blocks(target)
    pictures = [payload for block_type, payload in blocks if block_type == 6]
    assert len(pictures) == 1
    picture = _picture(pictures[0])
    assert picture["type"] == 3, "front cover"
    assert picture["mime"] == "image/png"
    assert picture["width"] == 8 and picture["height"] == 6
    assert picture["depth"] == 24
    assert picture["data"] == png

    # An invalid key is rejected.
    with pytest.raises(ValueError):
        write_vorbis_comments(target, {"BAD=KEY": "x"})


# ----------------------------------------------------------------------
# ID3v2
# ----------------------------------------------------------------------
def test_id3v2_frames_and_audio_starts_after_the_tag(source, tmp_path):
    target = str(tmp_path / "tagged.mp3")
    write_mp3(target, [(source, 0, 20000)])

    write_id3v2(
        target,
        {"TITLE": "Titel", "ARTIST": "Künstler", "PRODUCTION": "Prod", "VENUE": "Club"},
    )

    with open(target, "rb") as handle:
        raw = handle.read()

    frames, size = _id3v2_frames(raw)
    assert frames["TIT2"][0][0] == 0x03, "UTF-8 encoding byte"
    assert frames["TIT2"][0][1:].decode("utf-8") == "Titel"
    assert frames["TPE1"][0][1:].decode("utf-8") == "Künstler"

    # Every custom field gets its own TXXX frame.
    descriptions = {}
    for payload in frames["TXXX"]:
        assert payload[0] == 0x03
        description, value = payload[1:].split(b"\x00", 1)
        descriptions[description.decode("utf-8")] = value.decode("utf-8")
    assert descriptions == {"PRODUCTION": "Prod", "VENUE": "Club"}

    # The MPEG frames start right behind the tag.
    assert raw[10 + size] == 0xFF
    assert (raw[10 + size + 1] & 0xE0) == 0xE0

    # The audio is still decodable.
    data, sample_rate = sf.read(target, always_2d=True)
    assert sample_rate == SAMPLE_RATE
    assert len(data) > 0


def test_map_tag_key_to_frame():
    assert map_tag_key_to_frame("TITLE") == "TIT2"
    assert map_tag_key_to_frame("artist") == "TPE1"
    assert map_tag_key_to_frame("ALBUMARTIST") == "TPE2"
    assert map_tag_key_to_frame("PRODUCTION") == "TXXX:PRODUCTION"
    assert map_tag_key_to_frame("REPLAYGAIN_TRACK_GAIN") == "TXXX:REPLAYGAIN_TRACK_GAIN"
    assert map_tag_key_to_frame("NOT_A_REAL_KEY") is None


# ----------------------------------------------------------------------
# CUE
# ----------------------------------------------------------------------
def test_cue_sheet_contents(tmp_path):
    target = str(tmp_path / "album.cue")
    tracks = [
        CueTrack("One", 0, SAMPLE_RATE, artist="Band"),
        CueTrack("Two", SAMPLE_RATE, 2 * SAMPLE_RATE),
        CueTrack("Three", 2 * SAMPLE_RATE, 3 * SAMPLE_RATE),
    ]

    path = write_cue(
        target,
        "album.flac",
        tracks,
        SAMPLE_RATE,
        album="Album",
        artist="Band",
        genre="Rock",
        date="2026-01-02",
    )

    assert path == os.path.abspath(target)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    assert 'FILE "album.flac" WAVE' in text
    assert "REM GENRE Rock" in text
    assert "REM DATE 2026-01-02" in text
    for number, title in enumerate(("One", "Two", "Three"), start=1):
        assert f"TRACK {number:02d} AUDIO" in text
        assert f'TITLE "{title}"' in text
    assert f"INDEX 01 {frames_to_tc(0, SAMPLE_RATE)}" in text
    assert f"INDEX 01 {frames_to_tc(SAMPLE_RATE, SAMPLE_RATE)}" in text
    assert f"INDEX 01 {frames_to_tc(2 * SAMPLE_RATE, SAMPLE_RATE)}" in text
    assert f"INDEX 00 {frames_to_tc(SAMPLE_RATE, SAMPLE_RATE)}" in text


# ----------------------------------------------------------------------
# Atomicity and safety
# ----------------------------------------------------------------------
def test_no_partial_file_is_left_behind(source, source_file, tmp_path):
    flac = str(tmp_path / "a.flac")
    wav = str(tmp_path / "a.wav")
    mp3 = str(tmp_path / "a.mp3")
    cue = str(tmp_path / "a.cue")

    write_flac(flac, [(source, 0, 1000)])
    write_wav(wav, [(source, 0, 1000)])
    write_mp3(mp3, [(source, 0, 1000)])
    write_cue(cue, "a.flac", [CueTrack("t", 0, 1000)], SAMPLE_RATE)
    write_vorbis_comments(flac, {"TITLE": "x"})
    write_id3v2(mp3, {"TITLE": "x"})
    write_riff_info(wav, {"TITLE": "x"})

    assert list(tmp_path.glob("*.partial")) == []


def test_failed_write_removes_the_partial_and_keeps_no_target(
    source, tmp_path, monkeypatch
):
    target = str(tmp_path / "broken.flac")

    def explode(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", explode)

    with pytest.raises(OSError):
        write_flac(target, [(source, 0, 1000)])

    assert not os.path.exists(target)
    assert not os.path.exists(target + ".partial")


def test_export_refuses_to_overwrite_a_source(source, source_file):
    with pytest.raises(ValueError):
        write_flac(source_file, [(source, 0, 100)])
    with pytest.raises(ValueError):
        write_wav(source_file, [(source, 0, 100)])
    with pytest.raises(ValueError):
        write_mp3(source_file, [(source, 0, 100)])


def test_gain_is_applied_to_every_channel_equally(source, tmp_path):
    target = str(tmp_path / "gain.wav")
    write_wav(target, [(source, 0, 2000)], gain=0.5)

    data, _ = sf.read(target, always_2d=True, dtype="float32")
    original, _ = sf.read(
        source.path, start=0, frames=2000, always_2d=True, dtype="float32"
    )
    # The tolerance covers the 16 bit quantisation of the written file.
    assert np.allclose(data, original * 0.5, atol=1e-4)
    # The same gain on every channel: the channel difference scales as well.
    assert np.allclose(
        data[:, 0] - data[:, 1],
        (original[:, 0] - original[:, 1]) * 0.5,
        atol=1e-4,
    )
