"""Silence and noise-floor analysis.

Stage 1 of the analysis pipeline, part one.

This module answers one question: *where is there genuinely nothing?* It
does not claim to find tracks. A silence is evidence of a possible boundary,
not proof of one, and the confidence it produces is deliberately modest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Frame length used for the energy analysis (about 43 ms at 44.1 kHz).
DEFAULT_FRAME_FRAMES = 2048
#: Hop between analysis frames.
DEFAULT_HOP_FRAMES = 512

#: How far above the noise floor a frame must sit to count as "not silence".
THRESHOLD_ABOVE_FLOOR_DB = 12.0
#: A gap must be at least this far below the programme level. A passage that
#: is only a few decibels quieter than the music is a quiet moment in the
#: music, not a break in it.
GAP_DEPTH_BELOW_PROGRAMME_DB = 18.0
#: A gap must also be at least this far below the material immediately
#: around it. This is what protects a sustained quiet passage: its own
#: surroundings are quiet too, so it is not a boundary.
LOCAL_CONTRAST_DB = 12.0
#: How much material around a candidate is used as the local reference.
LOCAL_CONTEXT_SECONDS = 5.0


@dataclass
class SilenceRegion:
    """A run of frames whose level stays below the detection threshold."""

    start_frame: int
    end_frame: int
    level_dbfs: float

    def frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)


@dataclass
class NoiseProfile:
    """The adaptive levels used for silence detection."""

    noise_floor_dbfs: float
    threshold_dbfs: float
    method: str
    sample_rate: int = 44100


def frame_levels(
    signal: np.ndarray,
    frame_frames: int = DEFAULT_FRAME_FRAMES,
    hop_frames: int = DEFAULT_HOP_FRAMES,
) -> np.ndarray:
    """RMS level in dBFS for each analysis frame.

    Returns an empty array for an empty signal.
    """
    if signal.size == 0:
        return np.zeros(0, dtype=np.float32)
    frame_frames = max(16, int(frame_frames))
    hop_frames = max(1, int(hop_frames))
    count = 1 + max(0, (signal.size - frame_frames) // hop_frames)
    if signal.size < frame_frames:
        # Short material: measure it as a single frame.
        rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
        db = 20.0 * np.log10(max(rms, 1e-6))
        return np.array([db], dtype=np.float32)

    # Strided view without copying the whole signal.
    usable = (count - 1) * hop_frames + frame_frames
    frames = np.lib.stride_tricks.as_strided(
        signal[:usable],
        shape=(count, frame_frames),
        strides=(signal.strides[0] * hop_frames, signal.strides[0]),
        writeable=False,
    )
    powers = np.mean(np.square(frames, dtype=np.float64), axis=1)
    rms = np.sqrt(np.maximum(powers, 0.0))
    # -120 dBFS is the floor of what we report, so digital silence reads as
    # -120 dBFS rather than an absurd negative number.
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    return db.astype(np.float32)


def analyse_noise_floor(
    signal: np.ndarray,
    sample_rate: int,
    frame_frames: int = DEFAULT_FRAME_FRAMES,
    hop_frames: int = DEFAULT_HOP_FRAMES,
) -> NoiseProfile:
    """Estimate the noise floor and the detection threshold.

    The programme level is a high percentile (90th). The noise floor is
    estimated from the *local minimum* of the level curve: the quietest
    moment within a few seconds of every point, and then a low percentile
    of those.

    A plain low percentile is not good enough. Quiet music - a soft intro,
    a breakdown - is quieter than the music around it, so it drags a
    global percentile up, and then the quiet music looks like it is not
    there at all. The local minimum reaches past a passage of quiet music
    to the room tone on either side of it, which is what the noise floor
    actually is.
    """
    levels = frame_levels(signal, frame_frames, hop_frames)
    if levels.size == 0:
        return NoiseProfile(-70.0, -60.0, "empty signal")

    floor = _local_noise_floor(levels, hop_frames, sample_rate)
    programme = float(np.percentile(levels, 90))
    if programme < floor:
        programme = floor

    # A gap has to be clearly quieter than the programme material. A
    # recording with no quiet parts at all (a constant tone, a heavily
    # compressed master) has nothing to find, and the threshold simply sits
    # below everything.
    threshold = programme - GAP_DEPTH_BELOW_PROGRAMME_DB
    threshold = min(threshold, programme - 1.0)

    return NoiseProfile(
        noise_floor_dbfs=floor,
        threshold_dbfs=threshold,
        method=(
            f"adaptive: the quietest moment within five seconds of each point, "
            f"as the noise floor ({floor:.1f} dBFS); programme level "
            f"{programme:.1f} dBFS, threshold {threshold:.1f} dBFS "
            f"({GAP_DEPTH_BELOW_PROGRAMME_DB:.0f} dB below the programme)"
        ),
        sample_rate=int(sample_rate),
    )


def _local_noise_floor(
    levels: np.ndarray,
    hop_frames: int,
    sample_rate: int,
    seconds: float = 5.0,
) -> float:
    """The noise floor, estimated from local minima of the level curve."""
    from scipy.ndimage import minimum_filter1d

    window = max(3, int(seconds * sample_rate / max(1, hop_frames)))
    if levels.size < window:
        return float(np.percentile(levels, 10.0))
    local_minimum = minimum_filter1d(levels, size=window, mode="nearest")
    return float(np.percentile(local_minimum, 10.0))


def find_silence_regions(
    signal: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float,
    min_silence_seconds: float = 0.35,
    min_silence_frames: Optional[int] = None,
    frame_frames: int = DEFAULT_FRAME_FRAMES,
    hop_frames: int = DEFAULT_HOP_FRAMES,
) -> list[SilenceRegion]:
    """Find runs of frames below ``threshold_dbfs``.

    A region is reported when it lasts at least ``min_silence_seconds`` (or
    ``min_silence_frames`` when given). Region positions are indices into
    ``signal``.
    """
    if signal.size == 0:
        return []
    levels = frame_levels(signal, frame_frames, hop_frames)
    if levels.size == 0:
        return []

    if min_silence_frames is None:
        min_silence_frames = int(round(min_silence_seconds * sample_rate))

    quiet = levels < threshold_dbfs
    regions: list[SilenceRegion] = []
    context_frames = max(1, int(LOCAL_CONTEXT_SECONDS * sample_rate / max(1, hop_frames)))
    start: Optional[int] = None
    for index, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = index
        elif not is_quiet and start is not None:
            region = _make_region(start, index, levels, hop_frames, frame_frames, signal.size)
            if (
                region is not None
                and region.frames() >= min_silence_frames
                and _has_local_contrast(levels, start, index, context_frames)
            ):
                regions.append(region)
            start = None
    if start is not None:
        region = _make_region(start, len(quiet), levels, hop_frames, frame_frames, signal.size)
        if (
            region is not None
            and region.frames() >= min_silence_frames
            and _has_local_contrast(levels, start, len(quiet), context_frames)
        ):
            regions.append(region)
    return regions


def _has_local_contrast(
    levels: np.ndarray,
    start_index: int,
    end_index: int,
    context_frames: int,
    min_contrast_db: float = LOCAL_CONTRAST_DB,
) -> bool:
    """True when a quiet region is clearly quieter than its surroundings.

    This is what stops a sustained quiet passage from being mistaken for a
    gap: the material immediately around a quiet song is quiet as well, so
    the contrast is small and no boundary is proposed. A real gap between
    two songs sits next to loud material and passes easily.

    A burst of applause inside a song also fails this test: it is loud
    enough to be close to the music around it.
    """
    if end_index <= start_index:
        return False
    before = levels[max(0, start_index - context_frames): start_index]
    after = levels[end_index: end_index + context_frames]
    reference_parts = [part for part in (before, after) if part.size]
    if not reference_parts:
        # Nothing to compare against (the region is the whole recording).
        return True
    reference = float(np.median(np.concatenate(reference_parts)))
    region_level = float(np.median(levels[start_index:end_index]))
    return (reference - region_level) >= min_contrast_db


def _make_region(
    start_index: int,
    end_index: int,
    levels: np.ndarray,
    hop_frames: int,
    frame_frames: int,
    total_frames: int,
) -> Optional[SilenceRegion]:
    """Turn a run of quiet analysis frames into a region.

    The analysis frames that straddle a transition are partly loud, so the
    measured run is shorter than the real silence by up to one frame at each
    end. The region is therefore widened by half a frame on each side, which
    both restores the true duration and puts the suggested boundary in the
    middle of the silence rather than at its edge.
    """
    half = frame_frames // 2
    start_frame = max(0, start_index * hop_frames - half)
    end_frame = min(total_frames, end_index * hop_frames + half)
    if end_frame <= start_frame:
        return None
    level = float(np.min(levels[start_index:end_index])) if end_index > start_index else -120.0
    return SilenceRegion(start_frame=start_frame, end_frame=end_frame, level_dbfs=level)
