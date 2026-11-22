"""Export encoders.

Three writers turn selected regions of one or more sources into standalone
files: FLAC and WAV through libsndfile, MP3 through LAME, plus a CUE sheet
writer for the continuous-audio-plus-CUE workflow.

Rules every writer obeys:

* **Streaming.** Blocks of at most ``BLOCK_FRAMES`` frames are read from the
  sources and handed to the encoder; a multi-hour recording never sits in
  memory.
* **Atomic.** Output goes to ``<path>.partial`` first and is renamed onto the
  final path only after it is complete, so an interrupted export never leaves
  a plausible-looking truncated file behind.
* **No processing.** No fades, crossfades, de-noise, de-click, compression,
  limiting or resampling happens here. The only modification is the single
  linear gain the caller asks for.
* **Format preserving.** The source sample rate and channel layout are kept;
  the channel count is never changed and the gain is applied identically to
  every channel, so stereo balance survives.

The shared helpers live in this module so that the submodules have a single
place to import from. They are defined *before* the submodule imports at the
bottom of the file, because the submodules import them from this package.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import BinaryIO, Iterator, Optional, Sequence

import numpy as np

from ..decode import AudioSource

#: Number of frames read per streaming block (65536 frames ~ 1.5 s at 44.1 kHz).
BLOCK_FRAMES = 1 << 16

#: One part of an exported track: ``(source, start_frame, end_frame)``.
AudioPart = tuple[AudioSource, int, int]


@dataclass
class WriteResult:
    """What one export produced.

    ``frames_written`` counts the *source* frames written, excluding any
    encoder delay or padding the container adds. ``notes`` carries the
    human-readable remarks the UI shows (subtype conversions, encoder delay,
    padding residuals).
    """

    path: str
    frames_written: int
    sample_rate: int
    channels: int
    encoder_delay: int = 0
    encoder_padding: int = 0
    bytes_written: int = 0
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def normalise_path(path: str) -> str:
    """Absolute, case-normalised path used for comparisons."""
    return os.path.normcase(os.path.abspath(path))


def ensure_not_a_source(path: str, parts: Sequence[AudioPart]) -> None:
    """Refuse to write over one of the source files.

    The export layer checks this too; the writers check it again because the
    consequence of getting it wrong is destroying the user's recording.
    """
    target = normalise_path(path)
    for source, _start, _end in parts:
        if normalise_path(source.path) == target:
            raise ValueError(f"Refusing to overwrite the source file: {path}")
        try:
            if os.path.exists(path) and os.path.samefile(path, source.path):
                raise ValueError(f"Refusing to overwrite the source file: {path}")
        except OSError:
            # The target does not exist yet, or the file system cannot answer.
            pass


def describe_parts(parts: Sequence[AudioPart]) -> tuple[int, int]:
    """Return ``(sample_rate, channels)`` and check that all parts agree."""
    if not parts:
        raise ValueError("No source parts to export")
    first = parts[0][0]
    sample_rate = int(first.sample_rate)
    channels = int(first.channels)
    for source, _start, _end in parts:
        if int(source.sample_rate) != sample_rate or int(source.channels) != channels:
            raise ValueError(
                "Every part of an export must share the sample rate and the "
                f"channel count (found {source.sample_rate} Hz/"
                f"{source.channels} ch and {sample_rate} Hz/{channels} ch)"
            )
    return sample_rate, channels


def source_subtype(parts: Sequence[AudioPart]) -> str:
    """The sample subtype of the first part that carries real audio.

    A track that spans a gap between two source files starts with a run of
    digital silence, which has no file format of its own. The subtype is
    therefore taken from the first part that has one, so the export keeps
    the source's bit depth instead of degrading to a default.
    """
    for source, _start, _end in parts:
        info = getattr(source, "info", None)
        subtype = getattr(info, "subtype", None)
        if subtype:
            return str(subtype).upper()
    return "PCM_16"


def _remove_quietly(path: str) -> None:
    """Delete ``path`` if it exists, ignoring failures."""
    try:
        os.remove(path)
    except OSError:
        pass


@contextmanager
def atomic_output(
    path: str, *, with_handle: bool = True
) -> Iterator[tuple[str, Optional[BinaryIO]]]:
    """Write to ``<path>.partial`` and rename it onto ``path`` on success.

    Yields ``(temp_path, handle)``. ``handle`` is an open binary file object
    when ``with_handle`` is true, otherwise ``None`` (for writers that open
    the temporary file themselves, such as the libsndfile based ones).

    On any failure the partial file is removed and the exception is
    re-raised; I/O errors get a clearer message because "no space left on
    device" is the most common reason an export dies halfway through.
    """
    target = os.path.abspath(path)
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp = target + ".partial"
    handle: Optional[BinaryIO] = None
    try:
        if with_handle:
            handle = open(temp, "wb")
        yield temp, handle
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None
        os.replace(temp, target)
    except BaseException as exc:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        _remove_quietly(temp)
        if isinstance(exc, OSError):
            raise OSError(
                f"Could not write {target}: {exc}. The partial file was removed "
                "and the source files were not touched."
            ) from exc
        raise


def file_size(path: str) -> int:
    """Size of ``path`` in bytes, or 0 when it cannot be read."""
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def iter_part_blocks(
    parts: Sequence[AudioPart], block_frames: int = BLOCK_FRAMES
) -> Iterator[np.ndarray]:
    """Yield float32 blocks for every part, in order.

    The parts are joined with no gap and no overlap: the block that ends one
    part is immediately followed by the first block of the next one.
    """
    for source, start_frame, end_frame in parts:
        start = max(0, int(start_frame))
        end = min(int(end_frame), int(source.frames))
        if end <= start:
            continue
        yield from source.blocks(start, end - start, block_frames)


class GainApplier:
    """Applies one linear gain to every channel equally.

    Stereo balance is preserved by construction: the same multiplier is used
    for every channel and no channel specific processing exists. Nothing is
    clipped here - the caller is expected to have chosen a gain that fits
    under the ceiling; when it does not, the encoder's own conversion clips
    and the fact is reported in :meth:`notes`.

    An optional ``processor`` is applied after the gain, block by block, and
    may carry state across blocks. This is how the explicit limiter is
    hooked in: it is the only processing that is not a plain gain, and it
    only ever runs when the user asked for it.
    """

    def __init__(self, gain: float = 1.0, processor=None) -> None:
        self.gain = float(gain)
        self.processor = processor
        self.peak = 0.0

    def apply(self, block: np.ndarray) -> np.ndarray:
        """Return ``block`` with the gain applied (float32 in, float32 out)."""
        if self.gain != 1.0:
            block = block * np.float32(self.gain)
        if self.processor is not None:
            block = self.processor(block)
        if block.size:
            peak = float(np.abs(block).max())
            if peak > self.peak:
                self.peak = peak
        return block

    def notes(self) -> list[str]:
        """Remarks worth showing in the export log."""
        if self.gain != 1.0 and self.peak > 1.0:
            return [
                f"Gain {self.gain:g} pushed the signal to {self.peak:.4f}, beyond "
                "full scale; the encoder clipped it."
            ]
        return []


from .cue import CueTrack, write_cue, write_split_cue  # noqa: E402
from .flac import write_flac  # noqa: E402
from .mp3 import write_mp3  # noqa: E402
from .wav import write_wav  # noqa: E402

__all__ = [
    "AudioPart",
    "BLOCK_FRAMES",
    "CueTrack",
    "GainApplier",
    "WriteResult",
    "atomic_output",
    "describe_parts",
    "ensure_not_a_source",
    "file_size",
    "iter_part_blocks",
    "normalise_path",
    "source_subtype",
    "write_cue",
    "write_flac",
    "write_mp3",
    "write_split_cue",
    "write_wav",
]
