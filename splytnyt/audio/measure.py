"""Loudness and level measurement.

Implements the measurements the exporter and the level dialog need:

* **Integrated loudness** per ITU-R BS.1770-4: K-weighting, 400 ms blocks with
  75 % overlap, absolute gating at -70 LUFS and relative gating 10 LU below
  the ungated mean.
* **True peak** measured on a 4x oversampled signal, reported separately from
  the sample peak.
* **RMS**, per channel and overall.
* **ReplayGain** 1.0 (track gain) and 2.0 (album gain).

Everything streams: a six-hour recording is measured in blocks and the file is
never held in memory. The K-weighting filter state is carried across blocks,
so the result does not depend on the block size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import lfilter, resample_poly

#: Loudness block length in seconds (BS.1770-4).
BLOCK_SECONDS = 0.400
#: Hop between loudness blocks (75 % overlap).
HOP_SECONDS = 0.100
#: Absolute gate: blocks quieter than this are ignored.
ABSOLUTE_GATE_LUFS = -70.0
#: Relative gate: blocks this far below the ungated mean are ignored.
RELATIVE_GATE_LU = -10.0
#: Channel weights for the stereo sum (BS.1770-4).
CHANNEL_WEIGHTS = (1.0, 1.0)
#: Offset in the loudness formula.
LOUDNESS_OFFSET = -0.691

#: ReplayGain reference offset (89 dB SPL).
REPLAYGAIN_OFFSET = 64.82

#: Oversampling factor for the true peak.
TRUE_PEAK_OVERSAMPLE = 4

#: K-weighting filter design constants (BS.1770-4, high shelf + RLB high-pass).
_SHELF_F0 = 1681.974450955533
_SHELF_GAIN_DB = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_HIGHPASS_F0 = 38.13547087602444
_HIGHPASS_Q = 0.5003270373238773


@dataclass
class LoudnessResult:
    """The measurements of one range or programme."""

    integrated_lufs: float = -120.0
    loudness_range: float = 0.0
    true_peak_dbtp: float = -120.0
    sample_peak_dbfs: float = -120.0
    rms_dbfs: float = -120.0
    duration_frames: int = 0

    def is_silent(self) -> bool:
        """True when there is no measurable loudness."""
        return self.integrated_lufs <= -119.0


@dataclass
class Part:
    """One contiguous chunk of audio to be concatenated."""

    source: object
    start_frame: int
    end_frame: int

    def frames(self) -> int:
        return max(0, int(self.end_frame) - int(self.start_frame))


# ----------------------------------------------------------------------
# K-weighting filters
# ----------------------------------------------------------------------
def _high_shelf(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """The BS.1770-4 pre-filter (high shelf) for any sample rate."""
    k = math.tan(math.pi * _SHELF_F0 / sample_rate)
    vh = 10.0 ** (_SHELF_GAIN_DB / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / _SHELF_Q + k * k
    b = np.array([
        (vh + vb * k / _SHELF_Q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / _SHELF_Q + k * k) / a0,
    ])
    a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / _SHELF_Q + k * k) / a0])
    return b, a


def _high_pass(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """The BS.1770-4 RLB high-pass for any sample rate."""
    k = math.tan(math.pi * _HIGHPASS_F0 / sample_rate)
    a0 = 1.0 + k / _HIGHPASS_Q + k * k
    b = np.array([1.0, -2.0, 1.0])
    a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / _HIGHPASS_Q + k * k) / a0])
    return b, a


class _KWeighting:
    """Streaming K-weighting filter with carried state."""

    def __init__(self, channels: int, sample_rate: int) -> None:
        shelf_b, shelf_a = _high_shelf(sample_rate)
        pass_b, pass_a = _high_pass(sample_rate)
        self.shelf_b = shelf_b
        self.shelf_a = shelf_a
        self.pass_b = pass_b
        self.pass_a = pass_a
        self.shelf_state = [np.zeros(2) for _ in range(channels)]
        self.pass_state = [np.zeros(2) for _ in range(channels)]
        self.channels = channels

    def process(self, block: np.ndarray) -> np.ndarray:
        """Filter one block, carrying the filter state to the next call."""
        out = np.empty_like(block, dtype=np.float64)
        for channel in range(self.channels):
            data = block[:, channel].astype(np.float64, copy=False)
            filtered, self.shelf_state[channel] = lfilter(
                self.shelf_b, self.shelf_a, data, zi=self.shelf_state[channel]
            )
            filtered, self.pass_state[channel] = lfilter(
                self.pass_b, self.pass_a, filtered, zi=self.pass_state[channel]
            )
            out[:, channel] = filtered
        return out


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------
def _db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


class _Meter:
    """Accumulates the measurements over a stream of blocks."""

    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_frames = max(1, int(round(BLOCK_SECONDS * sample_rate)))
        self.hop_frames = max(1, int(round(HOP_SECONDS * sample_rate)))
        self.weighting = _KWeighting(channels, sample_rate)
        self._buffer = np.zeros((0, channels), dtype=np.float64)
        self._block_energies: list[float] = []
        self.sum_squares = 0.0
        self.sample_peak = 0.0
        self.true_peak = 0.0
        self.frames = 0
        self.channel_squares = np.zeros(channels, dtype=np.float64)
        self.channel_frames = 0

    # -- loudness ------------------------------------------------------
    def add(self, block: np.ndarray) -> None:
        if block.size == 0:
            return
        data = np.asarray(block, dtype=np.float64)
        self.sum_squares += float(np.sum(np.square(data)))
        self.sample_peak = max(self.sample_peak, float(np.max(np.abs(data))))
        self.channel_squares += np.sum(np.square(data), axis=0)
        self.channel_frames += data.shape[0]
        self.frames += data.shape[0]

        self.true_peak = max(self.true_peak, _true_peak_of(data))

        weighted = self.weighting.process(data)
        self._buffer = (
            weighted if self._buffer.size == 0
            else np.concatenate((self._buffer, weighted), axis=0)
        )
        while self._buffer.shape[0] >= self.block_frames:
            block_view = self._buffer[: self.block_frames]
            energy = float(np.mean(np.square(block_view)))
            weight_sum = energy * sum(CHANNEL_WEIGHTS[: self.channels])
            self._block_energies.append(weight_sum)
            self._buffer = self._buffer[self.hop_frames:]

    # -- results -------------------------------------------------------
    def result(self) -> LoudnessResult:
        if self.frames == 0:
            return LoudnessResult()

        rms = math.sqrt(self.sum_squares / (self.frames * self.channels))
        sample_peak = self.sample_peak

        integrated = self._integrated_loudness()
        return LoudnessResult(
            integrated_lufs=integrated,
            loudness_range=self._loudness_range(),
            true_peak_dbtp=_db(self.true_peak),
            sample_peak_dbfs=_db(sample_peak),
            rms_dbfs=_db(rms),
            duration_frames=self.frames,
        )

    def _integrated_loudness(self) -> float:
        if not self._block_energies:
            return -120.0
        energies = np.asarray(self._block_energies, dtype=np.float64)
        with np.errstate(divide="ignore"):
            loudness = LOUDNESS_OFFSET + 10.0 * np.log10(np.maximum(energies, 1e-30))

        # Absolute gate.
        keep = loudness > ABSOLUTE_GATE_LUFS
        if not np.any(keep):
            return -120.0
        gated = loudness[keep]
        energies = energies[keep]

        # Relative gate: -10 LU below the mean of the absolutely gated blocks.
        relative_threshold = (
            LOUDNESS_OFFSET + 10.0 * math.log10(max(float(np.mean(energies)), 1e-30))
            + RELATIVE_GATE_LU
        )
        keep = gated > relative_threshold
        if not np.any(keep):
            return -120.0
        final_energy = float(np.mean(energies[keep]))
        if final_energy <= 0:
            return -120.0
        return LOUDNESS_OFFSET + 10.0 * math.log10(final_energy)

    def _loudness_range(self) -> float:
        """Loudness range (EBU Tech 3342), reported as 0 when unavailable."""
        if not self._block_energies:
            return 0.0
        energies = np.asarray(self._block_energies, dtype=np.float64)
        with np.errstate(divide="ignore"):
            loudness = LOUDNESS_OFFSET + 10.0 * np.log10(np.maximum(energies, 1e-30))
        keep = loudness > (ABSOLUTE_GATE_LUFS - 20.0)
        if np.count_nonzero(keep) < 2:
            return 0.0
        values = loudness[keep]
        # The 10th and 95th percentiles of the gated distribution.
        low = float(np.percentile(values, 10.0))
        high = float(np.percentile(values, 95.0))
        return max(0.0, high - low)

    # -- ReplayGain ----------------------------------------------------
    def replaygain(self) -> tuple[float, float]:
        """ReplayGain 1.0 (track) and 2.0 (album) in dB.

        Reference: 89 dB SPL with the 64.82 dB offset used by the ReplayGain
        specification. The 1.0 value uses the RMS across all channels; the
        2.0 value uses the mean of the per-channel RMS values.
        """
        if self.channel_frames == 0:
            return 0.0, 0.0
        per_channel_rms = np.sqrt(
            np.maximum(self.channel_squares / max(1, self.channel_frames), 0.0)
        )
        combined = math.sqrt(
            float(np.sum(self.channel_squares))
            / max(1, self.channel_frames * self.channels)
        )
        rg1 = REPLAYGAIN_OFFSET - _db(combined)
        rg2 = REPLAYGAIN_OFFSET - _db(float(np.mean(per_channel_rms)))
        return rg1, rg2


def _true_peak_of(block: np.ndarray) -> float:
    """True peak of one block, measured on a 4x oversampled signal.

    The block is extended by *reflecting* its edges rather than padding
    with zeros. A zero pad would introduce a step discontinuity whose
    pre-ringing overshoots the real signal and inflates the reported peak
    by several tenths of a decibel; a reflected extension is continuous, so
    the interpolation filter sees no step.
    """
    if block.size == 0:
        return 0.0
    data = np.asarray(block, dtype=np.float64)
    rows = data.shape[0]
    if rows < 64:
        return float(np.max(np.abs(data)))
    pad = min(256, rows // 2)
    extended = np.concatenate(
        (data[:pad][::-1], data, data[-pad:][::-1]), axis=0
    )
    oversampled = resample_poly(extended, TRUE_PEAK_OVERSAMPLE, 1, axis=0)
    start = pad * TRUE_PEAK_OVERSAMPLE
    stop = start + rows * TRUE_PEAK_OVERSAMPLE
    usable = oversampled[start:stop]
    if usable.shape[0] == 0:
        return float(np.max(np.abs(data)))
    return float(np.max(np.abs(usable)))


def measure_parts(parts: Iterable[Part]) -> LoudnessResult:
    """Measure a programme made of several concatenated parts."""
    parts = list(parts)
    if not parts:
        return LoudnessResult()

    sample_rate = int(parts[0].source.sample_rate)
    channels = int(parts[0].source.channels)
    meter = _Meter(sample_rate, channels)

    block_frames = 1 << 16
    for part in parts:
        position = int(part.start_frame)
        end = int(part.end_frame)
        while position < end:
            count = int(min(block_frames, end - position))
            data = part.source.read(position, count)
            if data.size == 0:
                break
            meter.add(data)
            position += data.shape[0]
    return meter.result()


def measure_range(source, start_frame: int, end_frame: int) -> LoudnessResult:
    """Measure one contiguous range of a source."""
    return measure_parts([Part(source, start_frame, end_frame)])


def replaygain(source, start_frame: int, end_frame: int) -> tuple[float, float]:
    """ReplayGain 1.0 and 2.0 for one range of a source."""
    part = Part(source, start_frame, end_frame)
    sample_rate = int(part.source.sample_rate)
    channels = int(part.source.channels)
    meter = _Meter(sample_rate, channels)
    position = int(start_frame)
    end = int(end_frame)
    while position < end:
        count = int(min(1 << 16, end - position))
        data = part.source.read(position, count)
        if data.size == 0:
            break
        meter.add(data)
        position += data.shape[0]
    return meter.replaygain()


def measure_with_gain(
    measured: LoudnessResult, gain_db: float
) -> LoudnessResult:
    """Return the measurements that would result from applying a gain."""
    result = LoudnessResult(
        integrated_lufs=(measured.integrated_lufs + gain_db)
        if measured.integrated_lufs > -119.0 else -120.0,
        loudness_range=measured.loudness_range,
        true_peak_dbtp=measured.true_peak_dbtp + gain_db,
        sample_peak_dbfs=measured.sample_peak_dbfs + gain_db,
        rms_dbfs=measured.rms_dbfs + gain_db,
        duration_frames=measured.duration_frames,
    )
    return result
