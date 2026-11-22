"""Timeline reader.

The project timeline is the concatenation of its sources in order, with
explicit gaps (or overlaps) between them. :class:`TimelineReader` reads a
global timeline range and returns samples, handling:

* a source boundary in the middle of a range (samples continue seamlessly),
* a **gap** between sources (the gap is returned as digital silence, and is
  flagged so the export never drops it silently),
* an **overlap** between sources (the earlier source wins for the overlapping
  frames; the overlap is reported so the UI and the info file can show it).

The reader never mutates anything and never holds a whole file in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.model import Source
from .decode import AudioSource


@dataclass
class TimelineSegment:
    """Where one source sits inside the global timeline."""

    source_id: str
    source_index: int
    timeline_start: int
    frames: int
    source_start: int


class TimelineError(RuntimeError):
    """Raised when the sources cannot form a single coherent timeline."""


class TimelineReader:
    """Random access to the concatenated timeline."""

    def __init__(self, entries: list[tuple[Source, AudioSource]]) -> None:
        if not entries:
            raise TimelineError("The project has no sources.")
        self.entries = entries
        self.sample_rate = entries[0][1].sample_rate
        self.channels = entries[0][1].channels
        for model_source, source in entries:
            if source.sample_rate != self.sample_rate:
                raise TimelineError(
                    "All sources must share a sample rate "
                    f"({self.sample_rate} Hz vs {source.sample_rate} Hz in "
                    f"{model_source.path})."
                )
            if source.channels != self.channels:
                raise TimelineError(
                    "All sources must share a channel layout "
                    f"({self.channels} vs {source.channels} in {model_source.path})."
                )

        self.segments: list[TimelineSegment] = []
        cursor = 0
        for index, (model_source, source) in enumerate(entries):
            if index > 0:
                cursor += model_source.gap_frames
            self.segments.append(
                TimelineSegment(
                    source_id=model_source.source_id,
                    source_index=index,
                    timeline_start=cursor,
                    frames=source.frames,
                    source_start=0,
                )
            )
            cursor += source.frames
        self.total_frames = cursor

    # ------------------------------------------------------------------
    def read(self, start_frame: int, frames: int) -> np.ndarray:
        """Read ``frames`` frames from the global timeline.

        Returns a float32 array shaped ``(n, channels)``. Frames inside a gap
        are digital silence.
        """
        start_frame = max(0, int(start_frame))
        frames = max(0, int(frames))
        if frames == 0:
            return np.zeros((0, self.channels), dtype=np.float32)

        out = np.zeros((frames, self.channels), dtype=np.float32)
        filled = np.zeros(frames, dtype=bool)
        end_frame = start_frame + frames

        for segment in self.segments:
            seg_start = segment.timeline_start
            seg_end = seg_start + segment.frames
            if seg_end <= start_frame or seg_start >= end_frame:
                continue
            read_start = max(start_frame, seg_start)
            read_end = min(end_frame, seg_end)
            if read_end <= read_start:
                continue
            source = self.entries[segment.source_index][1]
            offset = segment.source_start + (read_start - seg_start)
            data = source.read(offset, read_end - read_start)
            if data.size == 0:
                continue
            target_start = read_start - start_frame
            take = min(data.shape[0], read_end - read_start)
            # Overlap semantics: the earlier source keeps the overlapping frames.
            available = ~filled[target_start: target_start + take]
            target = out[target_start: target_start + take]
            target[available] = data[:take][available]
            filled[target_start: target_start + take] = True

        return out

    def blocks(self, start_frame: int = 0, frames: Optional[int] = None,
               block_frames: int = 1 << 16):
        """Yield successive blocks of the timeline."""
        total = self.total_frames - start_frame if frames is None else frames
        total = max(0, total)
        position = start_frame
        remaining = total
        while remaining > 0:
            count = int(min(block_frames, remaining))
            yield self.read(position, count)
            position += count
            remaining -= count

    def segment_at(self, frame: int) -> Optional[TimelineSegment]:
        for segment in self.segments:
            if segment.timeline_start <= frame < segment.timeline_start + segment.frames:
                return segment
        return None

    def close(self) -> None:
        for _, source in self.entries:
            source.close()


def build_reader(project) -> TimelineReader:
    """Open every source of ``project`` and build a :class:`TimelineReader`."""
    from .decode import AudioSource

    entries: list[tuple[Source, AudioSource]] = []
    try:
        for source in project.sources:
            entries.append((source, AudioSource(source.path)))
    except Exception:
        for _, source in entries:
            source.close()
        raise
    return TimelineReader(entries)
