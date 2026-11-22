"""Frame-level activity measurement.

This module turns audio into a stream of per-frame measurements and then into
**multi-label activity regions**. It depends on nothing beyond NumPy and
SciPy, so it always works, on every machine, with no download.

Design notes that matter:

* **Activities overlap.** A stretch of dialogue played over an underscoring
  holds a ``speech`` activity and an ``instrumental`` activity at the same
  time. The detector never has to choose between "speech" and "music"; it
  reports both, each with its own confidence and its own evidence.
* **Nothing here knows about any particular production.** The thresholds are
  properties of speech, music and applause in general, not of any show.
* **Every region carries its evidence.** The numbers behind a decision are
  kept so the interface can show them and so a person can disagree.

The features are deliberately cheap and explainable:

``rms_db``
    The frame level. Everything else is read relative to it.
``flatness``
    Geometric over arithmetic mean of the power spectrum. High means noisy
    (applause, room tone), low means pitched (music, speech).
``flux``
    Positive change of the magnitude spectrum. Finds attacks.
``harmonic_ratio``
    Share of the energy that survives a median filter along time, which
    separates sustained pitched content (the harmonic part) from transient
    noise (the percussive part).
``f0`` / ``voicing``
    The autocorrelation pitch estimate and how periodic the frame is.
``modulation``
    How fast the level moves. Speech moves at the syllable rate, roughly
    3 to 8 Hz; a sustained chord does not move at all.

The last two together are what separate *sung* from *spoken*: both are
voiced and both are pitched, but a singer holds one note for a long time
while a speaker changes pitch every syllable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # SciPy gives float32 FFTs, which halve the memory of the big blocks.
    from scipy import fft as _sp_fft

    def _rfft(x: np.ndarray, n: int) -> np.ndarray:
        return _sp_fft.rfft(x, n=n, axis=-1)

    def _irfft(x: np.ndarray, n: int) -> np.ndarray:
        return _sp_fft.irfft(x, n=n, axis=-1)

except Exception:  # pragma: no cover - SciPy is a hard dependency in practice

    def _rfft(x: np.ndarray, n: int) -> np.ndarray:
        return np.fft.rfft(x, n=n, axis=-1)

    def _irfft(x: np.ndarray, n: int) -> np.ndarray:
        return np.fft.irfft(x, n=n, axis=-1)


#: Analysis window in samples. 2048 samples is about 43 ms at 44.1 kHz and
#: 46 ms at 48 kHz, long enough to resolve a low male singing note.
DEFAULT_WINDOW = 2048

#: Hop in samples. 512 samples is about 11 ms, which is finer than any
#: boundary a listener would accept and keeps onsets sharp.
DEFAULT_HOP = 512

#: Block length in seconds. The whole signal is never materialised: the
#: features are computed in blocks with an overlap so that a block's edge
#: frames see their neighbours.
BLOCK_SECONDS = 120.0

#: Pitch search range in Hz. Below 70 Hz a voice is usually a rumble; above
#: 1200 Hz the autocorrelation peak becomes unreliable.
F0_MIN_HZ = 70.0
F0_MAX_HZ = 1200.0

#: Two consecutive frames count as the same note when their pitch differs by
#: less than this many semitones. Singing holds notes; speech glides.
NOTE_TOLERANCE_SEMITONES = 0.35


@dataclass
class FrameFeatures:
    """Per-frame measurements. Every array has the same length."""

    sample_rate: int
    hop: int
    window: int
    rms_db: np.ndarray
    flatness: np.ndarray
    centroid_hz: np.ndarray
    flux: np.ndarray
    band_low: np.ndarray
    band_mid: np.ndarray
    band_high: np.ndarray
    harmonic_ratio: np.ndarray
    f0_hz: np.ndarray
    voicing: np.ndarray
    modulation: np.ndarray
    n_frames: int = 0

    def time_of(self, frame: int) -> float:
        """Seconds from the start of the signal to the centre of a frame."""
        return (frame * self.hop + self.window / 2.0) / float(self.sample_rate)


@dataclass
class ActivityScores:
    """Soft scores per frame, one array per activity label."""

    speech: np.ndarray
    singing: np.ndarray
    instrumental: np.ndarray
    applause: np.ndarray
    silence: np.ndarray
    sample_rate: int
    hop: int
    window: int

    def n_frames(self) -> int:
        return int(self.speech.size)

    def as_dict(self) -> dict:
        return {
            "speech": self.speech,
            "singing": self.singing,
            "instrumental": self.instrumental,
            "applause": self.applause,
            "silence": self.silence,
        }


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def _frame_view(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    """A strided view of overlapping frames without copying the signal."""
    if x.size < window:
        return np.zeros((0, window), dtype=x.dtype)
    count = 1 + (x.size - window) // hop
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=(count, window), strides=strides)


def _median_filter_time(values: np.ndarray, size: int) -> np.ndarray:
    """Median filter along the first axis, with edge replication."""
    if values.shape[0] == 0:
        return values
    try:
        from scipy.ndimage import median_filter

        return median_filter(values, size=(size, 1), mode="nearest")
    except Exception:  # pragma: no cover
        pad = size // 2
        padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(values)
        for index in range(values.shape[0]):
            out[index] = np.median(padded[index: index + size], axis=0)
        return out


def _block_features(
    mono: np.ndarray,
    sample_rate: int,
    hop: int,
    window: int,
    need_pitch: bool = True,
    need_harmonic: bool = True,
) -> dict:
    """Compute the raw per-frame measurements for one block."""
    frames = _frame_view(mono, window, hop)
    n = frames.shape[0]
    if n == 0:
        empty = np.zeros(0, dtype=np.float32)
        return {
            "rms_db": empty, "flatness": empty, "centroid_hz": empty,
            "flux": empty, "band_low": empty, "band_mid": empty,
            "band_high": empty, "harmonic_ratio": empty, "f0_hz": empty,
            "voicing": empty,
        }

    windowed = frames * np.hanning(window).astype(np.float32)
    power = np.square(windowed, dtype=np.float32)
    mean_power = power.mean(axis=1)

    rms = np.sqrt(np.maximum(mean_power, 1e-20))
    rms_db = (20.0 * np.log10(rms)).astype(np.float32)

    spectrum = _rfft(windowed, window)
    magnitude = np.abs(spectrum).astype(np.float32)
    spec_power = np.square(magnitude, dtype=np.float32)

    total_power = spec_power.sum(axis=1)
    safe_total = np.maximum(total_power, 1e-20)

    freqs = np.fft.rfftfreq(window, 1.0 / sample_rate)
    low = spec_power[:, freqs < 200.0].sum(axis=1) / safe_total
    mid = spec_power[:, (freqs >= 200.0) & (freqs < 2000.0)].sum(axis=1) / safe_total
    high = spec_power[:, freqs >= 2000.0].sum(axis=1) / safe_total

    centroid = (spec_power * freqs).sum(axis=1) / safe_total

    positive = np.maximum(spec_power, 1e-20)
    log_mean = np.log(positive).mean(axis=1)
    arith_mean = positive.mean(axis=1)
    flatness = np.exp(log_mean) / np.maximum(arith_mean, 1e-20)
    flatness = np.clip(flatness, 0.0, 1.0)

    # Harmonic / percussive split. The median of the spectrum over time is
    # the steady part; what is left over is the transient part. This is the
    # most expensive measurement here, so it can be switched off.
    if need_harmonic:
        median_size = max(3, int(round(0.35 * sample_rate / hop)) | 1)
        steady = _median_filter_time(spec_power, median_size)
        harmonic = np.maximum(spec_power - steady, 0.0).sum(axis=1)
        harmonic_ratio = harmonic / safe_total
    else:
        harmonic_ratio = np.zeros(n, dtype=np.float32)

    flux = np.zeros(n, dtype=np.float32)
    if n > 1:
        difference = np.maximum(magnitude[1:] - magnitude[:-1], 0.0)
        flux[1:] = difference.sum(axis=1) / np.maximum(
            magnitude[:-1].sum(axis=1), 1e-9
        )

    if need_pitch:
        f0_hz, voicing = _estimate_pitch(windowed, sample_rate, window)
    else:
        f0_hz = np.zeros(n, dtype=np.float32)
        voicing = np.zeros(n, dtype=np.float32)

    return {
        "rms_db": rms_db,
        "flatness": flatness.astype(np.float32),
        "centroid_hz": centroid.astype(np.float32),
        "flux": flux.astype(np.float32),
        "band_low": low.astype(np.float32),
        "band_mid": mid.astype(np.float32),
        "band_high": high.astype(np.float32),
        "harmonic_ratio": harmonic_ratio.astype(np.float32),
        "f0_hz": f0_hz.astype(np.float32),
        "voicing": voicing.astype(np.float32),
    }


def _estimate_pitch(
    windowed: np.ndarray, sample_rate: int, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation pitch estimate and periodicity for each frame.

    Returns ``(f0_hz, voicing)``. ``f0_hz`` is zero where nothing periodic
    was found; ``voicing`` is the normalised autocorrelation peak, so it says
    how *confidently* a pitch was found, not how loud the frame is.
    """
    n = windowed.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    spectrum = _rfft(windowed, window)
    autocorr = _irfft(np.square(np.abs(spectrum)), window)
    autocorr = autocorr.astype(np.float32)

    zero_lag = autocorr[:, :1]
    normalised = autocorr / np.maximum(zero_lag, 1e-20)

    lag_min = max(2, int(sample_rate / F0_MAX_HZ))
    lag_max = min(window - 2, int(sample_rate / F0_MIN_HZ))
    if lag_max <= lag_min:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    windowed_ac = normalised[:, lag_min: lag_max + 1]
    best_index = np.argmax(windowed_ac, axis=1)
    best_value = windowed_ac[np.arange(n), best_index]
    lag = (best_index + lag_min).astype(np.float32)

    # Parabolic refinement so the pitch is continuous enough to compare
    # neighbouring frames.
    left = np.clip(lag - 1, 0, normalised.shape[1] - 1).astype(np.int64)
    centre = np.clip(lag, 0, normalised.shape[1] - 1).astype(np.int64)
    right = np.clip(lag + 1, 0, normalised.shape[1] - 1).astype(np.int64)
    y0 = normalised[np.arange(n), left]
    y1 = normalised[np.arange(n), centre]
    y2 = normalised[np.arange(n), right]
    denominator = (y0 - 2.0 * y1 + y2)
    shift = np.where(
        np.abs(denominator) > 1e-9, 0.5 * (y0 - y2) / np.where(np.abs(denominator) > 1e-9, denominator, 1.0), 0.0
    )
    shift = np.clip(shift, -1.0, 1.0)
    refined_lag = lag + shift.astype(np.float32)
    refined_lag = np.where(refined_lag < 1.0, 1.0, refined_lag)

    f0 = sample_rate / refined_lag
    voicing = np.clip(best_value, 0.0, 1.0)
    f0 = np.where(voicing > 0.25, f0, 0.0).astype(np.float32)
    return f0.astype(np.float32), voicing.astype(np.float32)


def _modulation_profile(rms_db: np.ndarray, sample_rate: int, hop: int) -> np.ndarray:
    """How fast the level moves, per frame.

    The level envelope is measured over a sliding window of about 1.5
    seconds and the share of its variation in the 3-8 Hz band is reported.
    Speech syllables land in that band; a sustained chord does not move at
    all. The value is 0 for a still signal and approaches 1 for speech.

    The window is wide and the recording is long, so the work is done in
    blocks: the whole envelope would otherwise be materialised as a
    matrix several hundred megabytes wide.
    """
    n = rms_db.size
    if n < 8:
        return np.zeros(n, dtype=np.float32)

    frame_rate = sample_rate / float(hop)
    window = max(8, int(round(1.5 * frame_rate)))
    if window > n:
        window = max(4, n // 2)

    out = np.zeros(n, dtype=np.float32)
    block = max(1, int(30.0 * frame_rate))
    for begin in range(0, n, block):
        end = min(n, begin + block)
        padded = np.pad(
            rms_db[begin:end],
            (window // 2, window // 2),
            mode="edge",
        )
        view = np.lib.stride_tricks.as_strided(
            padded, shape=(end - begin, window), strides=(padded.strides[0], padded.strides[0])
        )
        centred = view - view.mean(axis=1, keepdims=True)
        spectrum = np.abs(_rfft(centred, window)) ** 2
        freqs = np.fft.rfftfreq(window, 1.0 / frame_rate)
        speech_band = spectrum[:, (freqs >= 3.0) & (freqs <= 8.0)].sum(axis=1)
        slow_band = spectrum[:, (freqs >= 0.2) & (freqs < 3.0)].sum(axis=1)
        fast_band = spectrum[:, freqs > 8.0].sum(axis=1)
        # The share of the movement that happens at the syllable rate.
        ratio = speech_band / (speech_band + slow_band + fast_band + 1e-12)
        out[begin:end] = np.clip(ratio, 0.0, 1.0).astype(np.float32)
    return out


def compute_frame_features(
    mono: "np.ndarray | None" = None,
    sample_rate: int = 44100,
    hop: int = DEFAULT_HOP,
    window: int = DEFAULT_WINDOW,
    cancel=None,
    need_pitch: bool = True,
    need_harmonic: bool = True,
    source=None,
    total_samples: Optional[int] = None,
) -> FrameFeatures:
    """Compute every per-frame measurement for a mono signal.

    The signal is processed in blocks so that memory use depends on the
    block size, not on the length of the recording. ``cancel`` is called
    between blocks; returning ``True`` stops the work early.

    ``need_pitch`` and ``need_harmonic`` switch off the two expensive
    measurements. The pitch estimate costs a second transform per frame
    and the harmonic/percussive split costs a median filter over the whole
    spectrum, and neither is needed when a trained classifier is doing the
    music-and-speech decision.

    ``source`` may be given instead of ``mono``: it is called as
    ``source(start, count)`` and must return ``count`` mono samples
    starting at that position. With a source the recording is never held
    in memory all at once, which matters for a three-hour musical.
    """
    if source is not None:
        if total_samples is None:
            raise ValueError(
                "compute_frame_features needs the total number of samples "
                "when it reads from a source."
            )
        total = int(total_samples)
        fetch = source
    else:
        if mono is None:
            raise ValueError("compute_frame_features needs either a signal or a source.")
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        mono = np.ascontiguousarray(mono, dtype=np.float32)
        total = mono.size
        fetch = lambda start, count: mono[start:start + count]

    blocks: list[dict] = []
    hop_frames = int(hop)
    window_frames = int(window)
    block_frames = max(1, int(BLOCK_SECONDS * sample_rate))
    # The overlap covers the median filter and the flux difference.
    overlap = window_frames * 3

    position = 0
    while position < total:
        if cancel is not None and cancel():
            break
        start = max(0, position - overlap)
        end = min(total, position + block_frames + overlap)
        chunk = np.ascontiguousarray(fetch(start, end - start), dtype=np.float32)
        if chunk.size < window_frames:
            break
        piece = _block_features(
            chunk,
            sample_rate,
            hop_frames,
            window_frames,
            need_pitch=need_pitch,
            need_harmonic=need_harmonic,
        )
        # Drop the frames that belong to the overlap of the previous block.
        skip = 0 if position == 0 else min(piece["rms_db"].size, overlap // hop_frames)
        blocks.append({key: value[skip:] for key, value in piece.items()})
        position += block_frames

    if not blocks:
        empty = np.zeros(0, dtype=np.float32)
        return FrameFeatures(
            sample_rate=sample_rate, hop=hop_frames, window=window_frames,
            rms_db=empty, flatness=empty, centroid_hz=empty, flux=empty,
            band_low=empty, band_mid=empty, band_high=empty,
            harmonic_ratio=empty, f0_hz=empty, voicing=empty, modulation=empty,
            n_frames=0,
        )

    merged = {
        key: np.concatenate([block[key] for block in blocks])
        for key in blocks[0]
    }
    modulation = _modulation_profile(merged["rms_db"], sample_rate, hop_frames)

    return FrameFeatures(
        sample_rate=sample_rate,
        hop=hop_frames,
        window=window_frames,
        rms_db=merged["rms_db"],
        flatness=merged["flatness"],
        centroid_hz=merged["centroid_hz"],
        flux=merged["flux"],
        band_low=merged["band_low"],
        band_mid=merged["band_mid"],
        band_high=merged["band_high"],
        harmonic_ratio=merged["harmonic_ratio"],
        f0_hz=merged["f0_hz"],
        voicing=merged["voicing"],
        modulation=modulation,
        n_frames=int(merged["rms_db"].size),
    )


# ----------------------------------------------------------------------
# Pitch stability
# ----------------------------------------------------------------------
def note_runs(f0_hz: np.ndarray, voicing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group voiced frames into runs that hold the same note.

    Returns ``(run_length_frames, run_index)``. A run breaks when the pitch
    moves by more than :data:`NOTE_TOLERANCE_SEMITONES` or when the frame
    stops being voiced. Speaking produces many short runs; singing produces
    long ones.
    """
    n = f0_hz.size
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    voiced = (f0_hz > 0.0) & (voicing > 0.3)
    run_length = np.zeros(n, dtype=np.int32)
    run_index = np.zeros(n, dtype=np.int32)
    current_run = -1
    length = 0
    previous_f0 = 0.0

    for index in range(n):
        if not voiced[index]:
            length = 0
            previous_f0 = 0.0
            run_index[index] = -1
            continue
        f0 = float(f0_hz[index])
        if previous_f0 > 0.0:
            semitones = abs(12.0 * np.log2(f0 / previous_f0))
            if semitones > NOTE_TOLERANCE_SEMITONES:
                length = 0
        length += 1
        current_run = current_run + 1 if length > 1 else current_run
        run_length[index] = length
        run_index[index] = current_run
        previous_f0 = f0
    return run_length, run_index
