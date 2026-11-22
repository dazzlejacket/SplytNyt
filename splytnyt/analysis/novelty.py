"""Musical-change detection (spectral novelty).

Stage 1 of the analysis pipeline, part two.

Silence detection finds gaps. This finds *changes*: the points where the
timbre or the harmony changes abruptly. That includes the boundary between
two songs that run straight into each other (a segue or a medley) where no
silence exists at all.

A novelty peak is **not** proof of a track boundary. It is evidence that
*something* changed. The pipeline reports it as an uncertain transition.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter1d

DEFAULT_HOP_FRAMES = 512
DEFAULT_N_FFT = 1024

#: Width of the frequency smoothing applied to the spectral shape.
SMOOTHING_BINS = 9

#: Minimum frames between two reported peaks, before the sensitivity
#: multiplier is applied.
_BASE_MIN_SEPARATION_SECONDS = 1.5

#: How far before and after a peak the timbre is compared to decide whether
#: the change persists. A peak this close to the start or the end of the
#: recording cannot be validated, so it is not proposed at all.
PERSISTENCE_WINDOW_SECONDS = 2.0


def persistence_window_seconds() -> float:
    """The distance used to validate that a musical change persists."""
    return PERSISTENCE_WINDOW_SECONDS


def _hann(size: int) -> np.ndarray:
    return np.hanning(size).astype(np.float64)


def novelty_curve(
    signal: np.ndarray,
    sample_rate: int,
    hop_frames: int = DEFAULT_HOP_FRAMES,
    n_fft: int = DEFAULT_N_FFT,
) -> np.ndarray:
    """Spectral-flux novelty curve, one value per hop.

    The curve is non-negative. Its index ``i`` corresponds to the frame
    position ``i * hop_frames``.

    The flux is measured on the energy-normalised spectral shape, so the
    curve reacts to a change of *timbre* rather than to a change of volume.
    A quiet passage inside a song therefore produces almost no flux, which
    is what stops the detector from cutting quiet music apart, while a key
    change or a new instrument still stands out.
    """
    shapes = spectral_shapes(signal, sample_rate, hop_frames, n_fft)
    return novelty_curve_from_shapes(shapes)


def spectral_shapes(
    signal: np.ndarray,
    sample_rate: int,
    hop_frames: int = DEFAULT_HOP_FRAMES,
    n_fft: int = DEFAULT_N_FFT,
) -> np.ndarray:
    """The energy-normalised spectral shape of every analysis frame.

    Each frame is normalised to unit total energy, so the shape describes
    the *timbre* and not the volume. A square root compresses the shape
    without the numerical explosion a logarithm causes in the near-empty
    high-frequency bins.
    """
    if signal.size < n_fft:
        return np.zeros((0, 0), dtype=np.float64)

    hop_frames = max(1, int(hop_frames))
    n_fft = int(n_fft)
    window = _hann(n_fft)

    frame_count = 1 + (signal.size - n_fft) // hop_frames
    usable = (frame_count - 1) * hop_frames + n_fft
    frames = np.lib.stride_tricks.as_strided(
        signal[:usable],
        shape=(frame_count, n_fft),
        strides=(signal.strides[0] * hop_frames, signal.strides[0]),
        writeable=False,
    )
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    spectrum = spectrum[:, 1:]                      # drop the DC bin
    energy = np.sum(spectrum, axis=1, keepdims=True)
    shape = spectrum / np.maximum(energy, 1e-12)
    compressed = np.sqrt(shape)

    # Smooth the shape along the frequency axis. Without this, the position
    # of a narrow peak inside its analysis bin makes the shape flicker as
    # the phase drifts, which shows up as a stream of spurious "musical
    # changes" in material that is really a steady tone.
    if compressed.shape[1] > SMOOTHING_BINS * 2:
        compressed = uniform_filter1d(
            compressed, size=SMOOTHING_BINS, axis=1, mode="nearest"
        )
    return compressed


def novelty_curve_from_shapes(shapes: np.ndarray) -> np.ndarray:
    """The flux curve of a pre-computed shape matrix."""
    if shapes.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    difference = np.diff(shapes, axis=0)
    flux = np.sum(np.maximum(difference, 0.0), axis=1)
    return flux.astype(np.float64)


def normalise_curve(curve: np.ndarray) -> np.ndarray:
    """Scale a novelty curve to roughly 0..1 using its own statistics."""
    if curve.size == 0:
        return curve
    median = float(np.median(curve))
    deviation = float(np.percentile(curve, 95) - median)
    if deviation <= 0:
        return np.zeros_like(curve)
    return np.clip((curve - median) / deviation, 0.0, 1.0)


def find_novelty_peaks(
    curve: np.ndarray,
    hop_frames: int,
    sample_rate: int,
    sensitivity: str = "balanced",
    shapes: Optional[np.ndarray] = None,
) -> list[tuple[int, float]]:
    """Pick the peaks of a novelty curve.

    Returns ``[(frame_position, strength), ...]`` where ``strength`` is in
    0..1. Peaks are separated by at least a sensitivity-dependent distance so
    a single musical event is not reported several times.

    A peak has to stand out from its own surroundings twice over. It has to
    be the strongest point of the curve around it, and it has to *rise* above
    the local baseline by a margin that scales with how noisy the material
    is. Without that second test a steady passage produces a stream of
    "musical changes" from nothing more than the analysis window drifting
    across a bin boundary.

    When ``shapes`` is supplied, a peak is only accepted when the timbre
    *after* it is genuinely different from the timbre *before* it. A quiet
    passage inside a song produces a burst of flux at its edges but the
    music resumes unchanged afterwards, so it is rejected; a real track
    boundary changes the music that follows.
    """
    if curve.size < 3:
        return []

    threshold, separation_seconds = _sensitivity_parameters(sensitivity)
    strength_curve = normalise_curve(curve)

    separation = max(1, int(separation_seconds * sample_rate / max(1, hop_frames)))
    baseline = _local_baseline(curve, hop_frames, sample_rate)
    min_deviation = _minimum_deviation(sensitivity, curve)
    candidates: list[tuple[int, float]] = []
    deviations: list[float] = []

    last_accepted = -10 ** 9
    for index in range(1, strength_curve.size - 1):
        value = strength_curve[index]
        if value < threshold:
            continue
        if value < strength_curve[index - 1] or value < strength_curve[index + 1]:
            continue
        deviation = float(curve[index] - baseline[index])
        if deviation < min_deviation:
            continue
        if shapes is not None and not _timbre_persists(shapes, index, hop_frames, sample_rate):
            continue
        if index - last_accepted < separation:
            # Keep the stronger of two nearby peaks. The comparison uses the
            # rise above the local baseline, not the normalised strength:
            # the normalised value is clipped, so two very different events
            # can both read as 1.0.
            if candidates and deviation > deviations[-1]:
                candidates[-1] = (index, float(value))
                deviations[-1] = deviation
                last_accepted = index
            continue
        candidates.append((index, float(value)))
        deviations.append(deviation)
        last_accepted = index

    return [(index * hop_frames, strength) for index, strength in candidates]


def _local_baseline(
    curve: np.ndarray,
    hop_frames: int,
    sample_rate: int,
    seconds: float = 0.5,
) -> np.ndarray:
    """The average flux around each point of the curve."""
    window = max(3, int(seconds * sample_rate / max(1, hop_frames)))
    if window % 2 == 0:
        window += 1
    return uniform_filter1d(curve.astype(np.float64), size=window, mode="nearest")


def _minimum_deviation(sensitivity: str, curve: np.ndarray) -> float:
    """How far a peak has to rise above its surroundings to count.

    The margin scales with how noisy the material is, so a quiet studio
    recording can use a small margin and a noisy tape needs a large one.
    """
    median = float(np.median(curve))
    deviation = float(np.median(np.abs(curve - median))) * 1.4826
    multiplier = {
        "conservative": 6.0,
        "sensitive": 2.5,
    }.get((sensitivity or "balanced").lower(), 4.0)
    return multiplier * max(deviation, 1e-9)


def _timbre_persists(
    shapes: np.ndarray,
    index: int,
    hop_frames: int,
    sample_rate: int,
    min_distance: float = 0.10,
    compare_seconds: float = PERSISTENCE_WINDOW_SECONDS,
) -> bool:
    """True when the music after a peak is genuinely different from before.

    The comparison deliberately looks *past* the transient: the timbre from
    two seconds before the peak is compared with the timbre two seconds
    after it. A real track boundary changes the music that follows, so the
    two differ. A burst of applause, a cough or a drum hit inside a song
    does not, so the music before and after it is the same and the peak is
    rejected.

    A long transient - a minute of applause over the band, say - outlasts
    the first comparison window, so a second look further ahead decides it:
    if the music *returns* to what it was, the peak was a transient and not
    a change of song.
    """
    if shapes.shape[0] == 0:
        return True
    window = max(2, int(compare_seconds * sample_rate / max(1, hop_frames)))
    half = max(1, window // 2)

    before = shapes[max(0, index - window): max(0, index - half)]
    after = shapes[min(shapes.shape[0], index + half): min(shapes.shape[0], index + window)]
    if before.shape[0] < 2 or after.shape[0] < 2:
        # Not enough material on one side: accept rather than hide a real
        # boundary near the start or the end of the recording.
        return True
    distance = _shape_distance(before, after)
    if distance < min_distance:
        return False

    # The music changed. Did it change for good, or did it come back?
    # The comparison for that is made well away from the peak on both
    # sides: a burst of applause is loud enough to be the loudest thing in
    # the comparison window right next to it, and only the material before
    # the burst and after it shows that the music returned.
    far_before = shapes[
        max(0, index - 2 * window): max(0, index - window - half)
    ]
    later = shapes[
        min(shapes.shape[0], index + window + half): min(shapes.shape[0], index + 2 * window)
    ]
    if far_before.shape[0] >= 2 and later.shape[0] >= 2:
        if _shape_distance(far_before, later) < min_distance:
            # The music returned to what it was: this was a transient.
            return False
    return True


def _shape_distance(before: np.ndarray, after: np.ndarray) -> float:
    """How different two stretches of spectral shape are, 0 to 1."""
    before_mean = before.mean(axis=0)
    after_mean = after.mean(axis=0)
    norm_before = float(np.linalg.norm(before_mean))
    norm_after = float(np.linalg.norm(after_mean))
    if norm_before <= 1e-12 or norm_after <= 1e-12:
        return 1.0
    similarity = float(np.dot(before_mean, after_mean) / (norm_before * norm_after))
    return max(0.0, 1.0 - similarity)


def _sensitivity_parameters(sensitivity: str) -> tuple[float, float]:
    """Return ``(peak_threshold, minimum_separation_seconds)``.

    The threshold is deliberately high: a musical change has to stand out
    from the normal flux of the material. Reporting a cut in the middle of
    a steady passage is worse than missing one, because a missed boundary
    can be added by hand in seconds while a wrong cut has to be found and
    removed.
    """
    name = (sensitivity or "balanced").lower()
    if name == "conservative":
        return 0.70, _BASE_MIN_SEPARATION_SECONDS * 2.0
    if name == "sensitive":
        return 0.35, _BASE_MIN_SEPARATION_SECONDS * 0.5
    return 0.55, _BASE_MIN_SEPARATION_SECONDS
