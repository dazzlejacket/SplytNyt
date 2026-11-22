"""Multi-resolution waveform peak pyramids.

The waveform view needs min/max pairs for an arbitrary pixel span at any zoom
level. Recomputing them from the file on every repaint is far too slow, so we
build a pyramid once (streaming, block by block) and cache it next to the
project file.

Level 0 holds the finest resolution (``base_block`` frames per bucket); each
further level halves the resolution by merging pairs of buckets. Peak values
are stored per channel so that a stereo file keeps its channel separation in
the display.

The cache file is plain ``.npz`` — safe to delete, regenerated on demand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .decode import AudioSource

DEFAULT_BASE_BLOCK = 256
DEFAULT_LEVELS = 9

ProgressCallback = Optional[Callable[[float, str], None]]
CancelCallback = Optional[Callable[[], bool]]


class CancelledError(RuntimeError):
    """Raised when a cancelled job stops early."""


@dataclass
class Level:
    """One resolution level of the pyramid."""

    block_frames: int
    mins: np.ndarray  # (buckets, channels) float32
    maxs: np.ndarray  # (buckets, channels) float32

    @property
    def buckets(self) -> int:
        return int(self.mins.shape[0])

    def bucket_for_frame(self, frame: int) -> int:
        return int(frame // self.block_frames)


class PeakPyramid:
    """A set of nested min/max peak levels for one source."""

    def __init__(self, frames: int, channels: int, base_block: int = DEFAULT_BASE_BLOCK) -> None:
        self.frames = int(frames)
        self.channels = int(channels)
        self.base_block = int(base_block)
        self.levels: list[Level] = []

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        source: AudioSource,
        base_block: int = DEFAULT_BASE_BLOCK,
        progress: ProgressCallback = None,
        cancel: CancelCallback = None,
    ) -> "PeakPyramid":
        """Build a pyramid by streaming the file once."""
        pyramid = cls(source.frames, source.channels, base_block)
        channels = source.channels
        bucket_count = max(1, (source.frames + base_block - 1) // base_block)

        mins = np.full((bucket_count, channels), np.inf, dtype=np.float32)
        maxs = np.full((bucket_count, channels), -np.inf, dtype=np.float32)

        processed = 0
        bucket_index = 0
        remainder = np.zeros((0, channels), dtype=np.float32)

        for block in source.blocks(block_frames=base_block * 256):
            if cancel is not None and cancel():
                raise CancelledError("Peak analysis cancelled")
            count = block.shape[0]
            if count == 0:
                continue

            # Accumulate into whole buckets of `base_block` frames.
            buffer = block if remainder.size == 0 else np.concatenate((remainder, block), axis=0)
            whole = (buffer.shape[0] // base_block) * base_block
            if whole:
                reshaped = buffer[:whole].reshape(-1, base_block, channels)
                chunk_mins = reshaped.min(axis=1)
                chunk_maxs = reshaped.max(axis=1)
                end = min(bucket_count, bucket_index + chunk_mins.shape[0])
                if end > bucket_index:
                    mins[bucket_index:end] = chunk_mins[: end - bucket_index]
                    maxs[bucket_index:end] = chunk_maxs[: end - bucket_index]
                bucket_index = end
            remainder = buffer[whole:]

            processed += count
            if progress is not None:
                progress(min(1.0, processed / max(1, source.frames)), "Building waveform peaks")

        # The final partial bucket is padded with silence.
        if remainder.size and bucket_index < bucket_count:
            padded = np.zeros((base_block, channels), dtype=np.float32)
            padded[: remainder.shape[0]] = remainder
            chunk = padded.reshape(1, base_block, channels)
            mins[bucket_index] = chunk.min(axis=1)
            maxs[bucket_index] = chunk.max(axis=1)
            bucket_index += 1

        # Empty buckets (file shorter than one bucket) become silence.
        mins[~np.isfinite(mins)] = 0.0
        maxs[~np.isfinite(maxs)] = 0.0

        pyramid.levels.append(Level(base_block, mins, maxs))
        current = pyramid.levels[0]
        while current.buckets > 1:
            count = current.buckets // 2
            next_mins = np.minimum(current.mins[: count * 2:2], current.mins[1: count * 2:2])
            next_maxs = np.maximum(current.maxs[: count * 2:2], current.maxs[1: count * 2:2])
            current = Level(current.block_frames * 2, next_mins, next_maxs)
            pyramid.levels.append(current)

        return pyramid

    # ------------------------------------------------------------------
    def level_for_resolution(self, frames_per_pixel: float) -> Level:
        """Pick the coarsest level that still resolves ``frames_per_pixel``."""
        if not self.levels:
            raise ValueError("Empty pyramid")
        best = self.levels[0]
        for level in self.levels:
            if level.block_frames <= frames_per_pixel:
                best = level
            else:
                break
        return best

    def peaks_for_span(self, start_frame: int, end_frame: int, buckets: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mins, maxs)`` shaped ``(buckets, channels)`` for a span."""
        if buckets <= 0:
            return np.zeros((0, self.channels), dtype=np.float32), np.zeros(
                (0, self.channels), dtype=np.float32
            )
        span = max(1, end_frame - start_frame)
        frames_per_pixel = span / float(buckets)
        level = self.level_for_resolution(frames_per_pixel)
        level = self._coarsen(level, frames_per_pixel)

        first = max(0, int(start_frame // level.block_frames))
        last = min(level.buckets, int(np.ceil(end_frame / level.block_frames)))
        if last <= first:
            return np.zeros((buckets, self.channels), dtype=np.float32), np.zeros(
                (buckets, self.channels), dtype=np.float32
            )

        selected_mins = level.mins[first:last]
        selected_maxs = level.maxs[first:last]
        out_mins = np.zeros((buckets, self.channels), dtype=np.float32)
        out_maxs = np.zeros((buckets, self.channels), dtype=np.float32)

        edges = np.linspace(start_frame, end_frame, buckets + 1)
        for index in range(buckets):
            lo = int(np.floor(edges[index] / level.block_frames))
            hi = int(np.ceil(edges[index + 1] / level.block_frames))
            lo = max(first, min(lo, last - 1))
            hi = max(lo + 1, min(hi, last))
            out_mins[index] = selected_mins[lo - first: hi - first].min(axis=0)
            out_maxs[index] = selected_maxs[lo - first: hi - first].max(axis=0)
        return out_mins, out_maxs

    def _coarsen(self, level: Level, frames_per_pixel: float) -> Level:
        """Merge levels further until one bucket is finer than a pixel."""
        current = level
        while current.block_frames * 2 <= frames_per_pixel * 2 and len(self.levels):
            index = self.levels.index(current)
            if index + 1 >= len(self.levels):
                break
            current = self.levels[index + 1]
        return current

    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """Write the pyramid to an ``.npz`` file."""
        payload: dict[str, np.ndarray] = {
            "__meta__": np.array(
                [self.frames, self.channels, self.base_block, len(self.levels)], dtype=np.int64
            )
        }
        for index, level in enumerate(self.levels):
            payload[f"mins{index}"] = level.mins
            payload[f"maxs{index}"] = level.maxs
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        temp = path + ".partial"
        np.savez_compressed(temp, **payload)
        # numpy appends .npz when the filename lacks it.
        if not temp.endswith(".npz"):
            os.replace(temp + ".npz", path)
        else:
            os.replace(temp, path)
        return path

    @classmethod
    def load(cls, path: str) -> Optional["PeakPyramid"]:
        """Load a cached pyramid, or ``None`` when it is missing/unreadable."""
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path, allow_pickle=False)
            meta = data["__meta__"]
            frames, channels, base_block, level_count = (int(v) for v in meta)
            pyramid = cls(frames, channels, base_block)
            for index in range(level_count):
                mins = data[f"mins{index}"]
                maxs = data[f"maxs{index}"]
                pyramid.levels.append(Level(base_block * (2 ** index), mins, maxs))
            return pyramid
        except Exception:
            return None


def empty_pyramid(frames: int, channels: int) -> PeakPyramid:
    """A zero-filled pyramid, used when a source has no cached peaks yet."""
    pyramid = PeakPyramid(frames, channels)
    if frames <= 0 or channels <= 0:
        return pyramid
    buckets = max(1, (frames + pyramid.base_block - 1) // pyramid.base_block)
    mins = np.zeros((buckets, channels), dtype=np.float32)
    maxs = np.zeros((buckets, channels), dtype=np.float32)
    pyramid.levels.append(Level(pyramid.base_block, mins, maxs))
    current = pyramid.levels[0]
    while current.buckets > 1:
        count = current.buckets // 2
        next_mins = np.minimum(current.mins[: count * 2:2], current.mins[1: count * 2:2])
        next_maxs = np.maximum(current.maxs[: count * 2:2], current.maxs[1: count * 2:2])
        current = Level(current.block_frames * 2, next_mins, next_maxs)
        pyramid.levels.append(current)
    return pyramid
