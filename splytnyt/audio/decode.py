"""Streaming audio decoding.

Everything that touches full-length audio goes through :class:`AudioSource`,
which yields blocks and never materialises a whole file. A six-hour 96 kHz
stereo recording is processed in 65 536-frame blocks, so peak memory is a few
megabytes regardless of the file's length.

Decoding uses libsndfile (via :mod:`soundfile`), which handles FLAC, WAV,
AIFF, OGG and MP3. Files are opened read-only; nothing in this module can
write to a source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import soundfile as sf

#: Default block size for streaming reads (65536 frames ~ 1.4 s at 48 kHz).
DEFAULT_BLOCK_FRAMES = 1 << 16

#: Extensions we advertise in file dialogs.
SUPPORTED_EXTENSIONS = (
    ".flac", ".wav", ".wave", ".aiff", ".aif", ".aifc",
    ".ogg", ".oga", ".opus", ".mp3", ".wma", ".mp2", ".caf",
)

SUPPORTED_FILTER = (
    "Audio files (*.flac *.wav *.wave *.aiff *.aif *.aifc *.ogg *.oga *.opus "
    "*.mp3 *.wma *.mp2 *.caf);;All files (*)"
)


class AudioError(RuntimeError):
    """Raised for unreadable, corrupt or unsupported audio."""


@dataclass
class AudioInfo:
    """Static description of an audio file."""

    path: str
    sample_rate: int
    channels: int
    frames: int
    format: str
    subtype: str

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frames / float(self.sample_rate)

    @property
    def is_lossless_container(self) -> bool:
        return self.format.upper() in ("FLAC", "WAV", "AIFF", "WAVEX", "CAF")


def probe(path: str) -> AudioInfo:
    """Read the header of ``path`` without decoding the audio."""
    if not os.path.exists(path):
        raise AudioError(f"File not found: {path}")
    try:
        info = sf.info(path)
    except Exception as exc:  # soundfile raises RuntimeError / sf.LibsndfileError
        raise AudioError(f"Cannot read audio file {os.path.basename(path)}: {exc}") from exc
    return AudioInfo(
        path=os.path.abspath(path),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        format=str(info.format),
        subtype=str(info.subtype),
    )


class AudioSource:
    """A read-only, block-wise view of one audio file.

    Usage::

        with AudioSource(path) as src:
            for block in src.blocks():
                ...

    The underlying file handle is kept open between calls so that seeking to
    a boundary for playback is cheap.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        self._info = probe(path)
        self._handle = None

    # ------------------------------------------------------------------
    @property
    def info(self) -> AudioInfo:
        return self._info

    @property
    def sample_rate(self) -> int:
        return self._info.sample_rate

    @property
    def channels(self) -> int:
        return self._info.channels

    @property
    def frames(self) -> int:
        return self._info.frames

    @property
    def duration_seconds(self) -> float:
        return self._info.duration_seconds

    # ------------------------------------------------------------------
    def _ensure_open(self):
        if self._handle is None:
            try:
                self._handle = sf.SoundFile(self.path, "r")
            except Exception as exc:
                raise AudioError(
                    f"Cannot open {os.path.basename(self.path)}: {exc}"
                ) from exc
        return self._handle

    def read(self, start_frame: int = 0, frames: Optional[int] = None) -> np.ndarray:
        """Read ``frames`` frames starting at ``start_frame``.

        Returns a float32 array shaped ``(n, channels)``. Fewer frames are
        returned at the end of the file.
        """
        handle = self._ensure_open()
        start_frame = max(0, int(start_frame))
        if frames is None:
            frames = max(0, self._info.frames - start_frame)
        frames = int(frames)
        if frames <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        try:
            handle.seek(start_frame)
            data = handle.read(frames=frames, dtype="float32", always_2d=True)
        except Exception as exc:
            raise AudioError(
                f"Decode error in {os.path.basename(self.path)} at frame "
                f"{start_frame}: {exc}"
            ) from exc
        return np.asarray(data, dtype=np.float32)

    def blocks(
        self,
        start_frame: int = 0,
        frames: Optional[int] = None,
        block_frames: int = DEFAULT_BLOCK_FRAMES,
    ) -> Iterator[np.ndarray]:
        """Yield successive ``(n, channels)`` float32 blocks."""
        self._ensure_open()
        total = self._info.frames if frames is None else min(frames, self._info.frames)
        start_frame = max(0, int(start_frame))
        position = start_frame
        end = start_frame + total
        while position < end:
            count = int(min(block_frames, end - position))
            block = self.read(position, count)
            if block.shape[0] == 0:
                break
            position += block.shape[0]
            yield block

    def read_mono(self, start_frame: int = 0, frames: Optional[int] = None) -> np.ndarray:
        """Read as a single mono channel (mean of all channels)."""
        data = self.read(start_frame, frames)
        if data.size == 0:
            return np.zeros(0, dtype=np.float32)
        return data.mean(axis=1).astype(np.float32, copy=False)

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None

    def __enter__(self) -> "AudioSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"AudioSource({os.path.basename(self.path)!r}, "
            f"{self.sample_rate} Hz, {self.channels} ch, {self.frames} frames)"
        )


def open_sources(paths: list[str]) -> list[AudioSource]:
    """Open several files, raising :class:`AudioError` for the first failure."""
    sources: list[AudioSource] = []
    try:
        for path in paths:
            sources.append(AudioSource(path))
    except Exception:
        for source in sources:
            source.close()
        raise
    return sources


class MemorySource:
    """An in-memory audio source with the same interface as :class:`AudioSource`.

    Used to represent digital silence when a track spans a gap between two
    source files: the gap must be preserved, not silently dropped, so the
    exporter writes real silence there instead of pretending the gap never
    existed.
    """

    def __init__(self, frames: int, sample_rate: int, channels: int = 2) -> None:
        self._frames = int(frames)
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)

    @property
    def path(self) -> str:
        return "<silence>"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def duration_seconds(self) -> float:
        return self._frames / float(self._sample_rate) if self._sample_rate else 0.0

    def read(self, start_frame: int = 0, frames: Optional[int] = None) -> np.ndarray:
        start_frame = max(0, int(start_frame))
        if frames is None:
            frames = max(0, self._frames - start_frame)
        frames = max(0, int(frames))
        return np.zeros((frames, self._channels), dtype=np.float32)

    def blocks(self, start_frame: int = 0, frames: Optional[int] = None,
               block_frames: int = DEFAULT_BLOCK_FRAMES):
        total = self._frames - start_frame if frames is None else frames
        position = start_frame
        remaining = max(0, total)
        while remaining > 0:
            count = int(min(block_frames, remaining))
            yield self.read(position, count)
            position += count
            remaining -= count

    def close(self) -> None:
        return None

    def __enter__(self) -> "MemorySource":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

