"""Onset detection and "is this music?" scoring.

This module answers two questions that the old detector had merged into one,
and keeping them apart is what fixes the bad track starts:

* **Where does something begin?** - the per-hop level and spectral analysis
  says which moments carry a change.
* **Is what follows music?** - a separate test says whether the material
  after a candidate is tonal, sustained music rather than applause, speech
  or room noise.

The old detector cut in the middle of a quiet gap because it only ever asked
"where is it quiet?". This module lets the pipeline ask "where does the next
song actually begin?", which is a different question with a different answer.

The music test is deliberately conservative and *level-independent*: a quiet
piano intro is music, a loud applause burst is not. That is what lets a soft
intro be preserved while an audience clap is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .silence import frame_levels

#: Analysis hop. 512 frames is 11.6 ms at 44.1 kHz, which is finer than the
#: margin a cut is placed with.
DEFAULT_HOP_FRAMES = 512
#: Analysis window. 2048 frames is long enough to resolve a pitch.
DEFAULT_N_FFT = 2048

#: The window used to find the *attack* itself, once a candidate has been
#: found. It is short on purpose: a long window smears the start of a note
#: over its whole length, which is what made the reported onset tens of
#: milliseconds early.
ATTACK_WINDOW_FRAMES = 256
#: The step used when following the attack.
ATTACK_HOP_FRAMES = 64
#: How far back the level is compared when deciding whether the material
#: actually *rose* into the onset. A decaying tail does not rise; a song
#: starting does.
ATTACK_LOOKBACK_SECONDS = 0.150
#: How much the level has to rise over that look-back to count as an onset.
ATTACK_RISE_DB = 6.0
#: A level that was below the presence threshold this recently counts as a
#: rise as well, which is what catches a song that fades in slowly.
ATTACK_RECENT_SECONDS = 0.060

#: Spectral flatness at or above this is noise, not a pitched sound.
FLATNESS_NOISE = 0.30
#: A hop counts as "tonal" when its flatness is at or below this fraction of
#: :data:`FLATNESS_NOISE`.
TONALITY_THRESHOLD = 0.45
#: How far above the noise floor a hop must sit to count as "present" at all.
PRESENT_ABOVE_FLOOR_DB = 6.0
#: How much of the window after an onset has to be music for the onset to be
#: believed. A single whistle in a long silence is not a song.
SUSTAIN_SECONDS = 1.25
#: Required fraction of musical hops in the sustain window.
SUSTAIN_DENSITY = 0.45
#: How long the search for the next musical onset may run.
SEARCH_LIMIT_SECONDS = 30.0


@dataclass
class OnsetSearch:
    """The outcome of looking for the next musical onset."""

    #: Frame of the onset, or -1 when none was found.
    frame: int = -1
    #: How much the material after the onset looks like music, 0..1.
    score: float = 0.0
    #: True when the answer is decisive enough to act on without a human.
    confident: bool = False
    #: Plain-language explanation of what was found.
    reason: str = ""
    #: Technical numbers behind the decision, for the details panel.
    details: dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.frame >= 0


def music_hop_profile(
    signal: np.ndarray,
    sample_rate: int,
    hop_frames: int = DEFAULT_HOP_FRAMES,
    n_fft: int = DEFAULT_N_FFT,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-hop level and tonality of a signal.

    Returns ``(levels_dbfs, tonality)``. ``tonality`` is 1 for a pure tone and
    0 for white noise; it is derived from the spectral flatness of each hop.

    The level is the RMS of the analysis window and the tonality is the
    spectral shape of the same window, so the two arrays are index-aligned and
    describe the same instants.
    """
    if signal.size < n_fft:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty

    hop = max(1, int(hop_frames))
    n_fft = int(n_fft)
    count = 1 + (signal.size - n_fft) // hop
    usable = (count - 1) * hop + n_fft
    frames = np.lib.stride_tricks.as_strided(
        signal[:usable],
        shape=(count, n_fft),
        strides=(signal.strides[0] * hop, signal.strides[0]),
        writeable=False,
    )
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1))
    power = np.square(spectrum)

    with np.errstate(divide="ignore", invalid="ignore"):
        geometric = np.exp(np.mean(np.log(power + 1e-20), axis=1))
        arithmetic = np.mean(power, axis=1)
    flatness = np.where(arithmetic > 0, geometric / np.maximum(arithmetic, 1e-20), 1.0)
    flatness = np.clip(np.nan_to_num(flatness, nan=1.0), 0.0, 1.0)

    tonality = 1.0 - np.clip(flatness / FLATNESS_NOISE, 0.0, 1.0)
    levels = frame_levels(signal, frame_frames=n_fft, hop_frames=hop).astype(np.float64)
    # frame_levels and the FFT above use the same count; trim defensively.
    length = min(levels.size, tonality.size)
    return levels[:length], tonality[:length]


def _rolling_density(values: np.ndarray, window: int) -> np.ndarray:
    """Fraction of ``True`` values in every window of ``window`` hops."""
    if values.size == 0:
        return values.astype(np.float64)
    window = max(1, int(window))
    cumulative = np.concatenate(([0.0], np.cumsum(values.astype(np.float64))))
    # density[i] covers values[i : i + window]; hops past the end count as
    # "not musical", so a passage that stops half way through the window is
    # not rewarded for it.
    counts = cumulative[window:] - cumulative[:-window] if values.size >= window else None
    if counts is None:
        return np.zeros(values.size, dtype=np.float64)
    density = np.zeros(values.size, dtype=np.float64)
    density[: counts.size] = counts / float(window)
    return density


def short_window_levels(
    signal: np.ndarray,
    sample_rate: int,
    window_frames: int = ATTACK_WINDOW_FRAMES,
    hop_frames: int = ATTACK_HOP_FRAMES,
) -> np.ndarray:
    """The level of a signal in short windows.

    A short window follows the start of a note closely: a long one smears
    the attack over its whole length and reports a start that is tens of
    milliseconds early.
    """
    if signal.size < window_frames:
        return np.zeros(0, dtype=np.float64)
    window = max(1, int(window_frames))
    hop = max(1, int(hop_frames))
    count = 1 + (signal.size - window) // hop
    usable = (count - 1) * hop + window
    frames = np.lib.stride_tricks.as_strided(
        signal[:usable],
        shape=(count, window),
        strides=(signal.strides[0] * hop, signal.strides[0]),
        writeable=False,
    )
    power = np.mean(np.square(frames.astype(np.float64)), axis=1)
    with np.errstate(divide="ignore"):
        levels = 10.0 * np.log10(np.maximum(power, 1e-20))
    return levels


def find_next_music_onset(
    signal: np.ndarray,
    sample_rate: int,
    search_start: int = 0,
    search_end: Optional[int] = None,
    floor_dbfs: Optional[float] = None,
    sustain_seconds: float = SUSTAIN_SECONDS,
    density_threshold: float = SUSTAIN_DENSITY,
) -> OnsetSearch:
    """Find the first credible musical onset at or after ``search_start``.

    "Credible" means: the material after the candidate is present (audible
    above the noise floor), tonal rather than noisy, and sustained long enough
    to be a piece of music rather than a stray sound.

    The search happens in two steps, because one step cannot do both jobs:

    1. **Coarse.** A long analysis window finds the music. It is robust, but
       it smears the start of a note over 46 ms, so the frame it reports can
       be that much too early.
    2. **Fine.** A short window follows the material around the coarse
       candidate and reports the frame where the music actually *begins*.

    The fine step also insists that the material **rose** into the onset. The
    decay of the previous song is often pitched and audible, and without this
    rule the tail of the song before the gap would be reported as the start
    of the next one.

    ``floor_dbfs`` is the noise floor of the recording. When it is not given
    the floor is estimated from the searched region, which is usually the
    quiet gap itself.

    The returned frame is the *start* of the musical material, so a caller
    that wants a cut can place it a short pre-roll before it.
    """
    end = signal.size if search_end is None else int(min(search_end, signal.size))
    start = max(0, int(search_start))
    if end - start <= 0:
        return OnsetSearch(reason="There is no audio to search.")

    region = signal[start:end]
    levels, tonality = music_hop_profile(region, sample_rate)
    if levels.size == 0:
        return OnsetSearch(reason="There is no audio to search.")

    if floor_dbfs is None:
        floor_dbfs = float(np.percentile(levels, 10.0))

    present = levels > (floor_dbfs + PRESENT_ABOVE_FLOOR_DB)
    musical = present & (tonality >= TONALITY_THRESHOLD)

    hop = DEFAULT_HOP_FRAMES
    sustain_hops = max(1, int(sustain_seconds * sample_rate / hop))
    density = _rolling_density(musical, sustain_hops)

    first = -1
    for index in range(musical.size):
        if musical[index] and density[index] >= density_threshold:
            first = index
            break

    if first < 0:
        # Report how far the search got, so the interface can say *why* it
        # could not place the cut rather than pretending it did.
        best = float(np.max(density)) if density.size else 0.0
        return OnsetSearch(
            frame=-1,
            score=best,
            confident=False,
            reason=(
                "No clear musical start was found after the gap. The audio there "
                "does not look like the beginning of a song."
            ),
            details={
                "noise_floor_dbfs": round(float(floor_dbfs), 1),
                "best_music_density": round(best, 2),
                "searched_from": start,
                "searched_to": end,
            },
        )

    coarse_frame = start + first * hop
    fine = _find_attack(
        region,
        sample_rate,
        start=start,
        coarse_frame=coarse_frame,
        floor_dbfs=floor_dbfs,
        tonality=tonality,
        density=density,
        density_threshold=density_threshold,
    )

    if fine is None:
        # The coarse candidate is all there is. It is reported, but not as a
        # confident answer, because its position may be tens of milliseconds
        # off.
        return OnsetSearch(
            frame=coarse_frame,
            score=round(float(density[first]), 3),
            confident=False,
            reason=(
                "This is the earliest point that could be the start of the next "
                "song, but the detector could not find the exact moment the music "
                "begins. Listen to it before you accept it."
            ),
            details={
                "noise_floor_dbfs": round(float(floor_dbfs), 1),
                "coarse_frame": coarse_frame,
                "music_density": round(float(density[first]), 2),
                "attack_found": False,
            },
        )

    frame, rise_db, level_dbfs = fine
    window_density = float(density[min(len(density) - 1, (frame - start) // hop)])
    mean_tonality = float(np.mean(tonality[first: first + sustain_hops]))
    confident = window_density >= 0.7 and mean_tonality >= 0.55

    if confident:
        reason = (
            "The next song starts here: the audio after this point is pitched, "
            "steady and clearly above the noise floor."
        )
    else:
        reason = (
            "This is the earliest point that could be the start of the next song, "
            "but the detector is not certain: applause and speech can look like "
            "music to it. Listen to it before you accept it."
        )

    return OnsetSearch(
        frame=frame,
        score=round(window_density, 3),
        confident=confident,
        reason=reason,
        details={
            "noise_floor_dbfs": round(float(floor_dbfs), 1),
            "coarse_frame": coarse_frame,
            "attack_found": True,
            "rise_db": round(rise_db, 1),
            "attack_level_dbfs": round(level_dbfs, 1),
            "music_density": round(window_density, 2),
            "tonality": round(mean_tonality, 2),
        },
    )


def _find_attack(
    region: np.ndarray,
    sample_rate: int,
    start: int,
    coarse_frame: int,
    floor_dbfs: float,
    tonality: np.ndarray,
    density: np.ndarray,
    density_threshold: float,
):
    """Find the frame at which the music actually begins.

    The whole transition region is searched, not just a window around the
    coarse candidate: when the coarse candidate lands in the tail of the
    previous song, the real attack can be seconds away, and a search that
    only looked around the candidate would never find it.

    Returns ``(frame, rise_db, level_dbfs)`` or ``None`` when the material
    does not rise into a credible onset.
    """
    short = short_window_levels(region, sample_rate)
    if short.size == 0:
        return None

    hop = ATTACK_HOP_FRAMES
    threshold = floor_dbfs + PRESENT_ABOVE_FLOOR_DB
    lookback = max(1, int(ATTACK_LOOKBACK_SECONDS * sample_rate / hop))
    recent = max(1, int(ATTACK_RECENT_SECONDS * sample_rate / hop))

    last_below = -1
    for index in range(short.size):
        level = short[index]
        if level <= threshold:
            last_below = index
            continue

        frame = start + index * hop
        hop_index = (frame - start) // DEFAULT_HOP_FRAMES
        if hop_index >= tonality.size or hop_index >= density.size:
            continue
        # The material has to be music: pitched, and sustained.
        if tonality[hop_index] < TONALITY_THRESHOLD:
            continue
        if density[hop_index] < density_threshold:
            continue

        back = short[max(0, index - lookback)]
        rise = level - back
        recently_below = last_below >= 0 and (index - last_below) <= recent
        if rise < ATTACK_RISE_DB and not recently_below:
            # The material was already there and is not rising: this is the
            # tail of the previous song, not the start of the next one.
            continue

        return frame, rise, level
    return None


def describe_music_at(
    signal: np.ndarray,
    sample_rate: int,
    frame: int,
    window_seconds: float = 1.25,
) -> tuple[float, dict]:
    """Score how much the material around ``frame`` looks like music.

    Used by the review panel to explain *why* a cut was placed where it was,
    and by the tests to check the reasoning rather than only the outcome.
    """
    window_frames = max(1, int(window_seconds * sample_rate))
    start = max(0, int(frame) - window_frames // 2)
    end = min(signal.size, start + window_frames)
    chunk = signal[start:end]
    if chunk.size == 0:
        return 0.0, {"level_dbfs": -120.0, "tonality": 0.0, "flatness": 1.0}

    levels, tonality = music_hop_profile(chunk, sample_rate)
    if levels.size == 0:
        return 0.0, {"level_dbfs": -120.0, "tonality": 0.0, "flatness": 1.0}

    flatness = float(np.mean(np.clip(1.0 - tonality, 0.0, 1.0)))
    mean_tonality = float(np.mean(tonality))
    mean_level = float(np.mean(levels))
    score = float(np.clip(0.6 * mean_tonality + 0.4 * min(1.0, mean_level / -20.0), 0.0, 1.0))
    return score, {
        "level_dbfs": round(mean_level, 1),
        "tonality": round(mean_tonality, 2),
        "flatness": round(flatness, 3),
    }
