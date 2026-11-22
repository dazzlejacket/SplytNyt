"""Tests for the loudness measurement and the gain planner.

The loudness numbers are checked against the BS.1770-4 definition computed
independently in the test, so the test fails if the implementation drifts.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splytnyt.audio.decode import AudioSource  # noqa: E402
from splytnyt.audio.gain import plan_per_track_gain, plan_shared_gain  # noqa: E402
from splytnyt.audio.measure import Part, measure_parts, measure_range, replaygain  # noqa: E402

SAMPLE_RATE = 48000


def write_signal(path: str, data: np.ndarray, subtype: str = "PCM_16") -> str:
    sf.write(path, data, SAMPLE_RATE, subtype=subtype)
    return path


def sine(amplitude: float, seconds: float, frequency: float = 1000.0,
         channels: int = 1) -> np.ndarray:
    frames = int(seconds * SAMPLE_RATE)
    time_axis = np.arange(frames) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * frequency * time_axis) * amplitude
    if channels == 1:
        return wave.astype(np.float32).reshape(-1, 1)
    return np.stack([wave] * channels, axis=1).astype(np.float32)


def db_to_amplitude(db: float) -> float:
    return 10.0 ** (db / 20.0)


def test_mono_sine_loudness_matches_the_bs1770_definition(tmp_path):
    """A 1 kHz sine measures what the standard says it must measure.

    BS.1770-4's calibration point: the K-weighting curve has +0.691 dB of
    gain at 1 kHz, which cancels the -0.691 offset in the formula, so a
    1 kHz sine at -23 dBFS RMS reads -23.0 LUFS in a mono file.
    """
    amplitude = db_to_amplitude(-20.0)          # -20 dBFS peak
    data = sine(amplitude, 6.0, channels=1)
    path = write_signal(str(tmp_path / "tone.flac"), data)

    with AudioSource(path) as source:
        measured = measure_range(source, 0, source.frames)

    rms = amplitude / math.sqrt(2.0)
    assert measured.rms_dbfs == pytest.approx(20 * math.log10(rms), abs=0.1)
    assert measured.integrated_lufs == pytest.approx(-23.0, abs=0.2)


def test_dual_mono_is_three_decibels_above_mono(tmp_path):
    """A stereo file with the same signal in both channels reads 3 LU higher."""
    amplitude = db_to_amplitude(-20.0)
    mono_path = write_signal(str(tmp_path / "mono.flac"), sine(amplitude, 5.0, channels=1))
    stereo_path = write_signal(str(tmp_path / "stereo.flac"), sine(amplitude, 5.0, channels=2))

    with AudioSource(mono_path) as source:
        mono_measured = measure_range(source, 0, source.frames)
    with AudioSource(stereo_path) as source:
        stereo_measured = measure_range(source, 0, source.frames)

    assert stereo_measured.integrated_lufs - mono_measured.integrated_lufs == pytest.approx(
        3.01, abs=0.3
    )


def test_true_peak_and_sample_peak_are_reported_separately(tmp_path):
    """The true peak is measured on the oversampled signal."""
    amplitude = db_to_amplitude(-6.0)
    data = sine(amplitude, 3.0)
    path = write_signal(str(tmp_path / "tone.flac"), data)

    with AudioSource(path) as source:
        measured = measure_range(source, 0, source.frames)

    assert measured.sample_peak_dbfs == pytest.approx(-6.0, abs=0.2)
    # A 1 kHz sine reconstructs to slightly above its sample peak.
    assert measured.true_peak_dbtp == pytest.approx(-5.8, abs=0.35)
    assert measured.true_peak_dbtp >= measured.sample_peak_dbfs - 0.05


def test_full_scale_sine_peaks_at_zero_dbtsp(tmp_path):
    data = sine(db_to_amplitude(-0.1), 2.0)
    path = write_signal(str(tmp_path / "tone.flac"), data)
    with AudioSource(path) as source:
        measured = measure_range(source, 0, source.frames)
    assert measured.true_peak_dbtp == pytest.approx(0.0, abs=0.3)


def test_silence_measures_as_silent(tmp_path):
    data = np.zeros((SAMPLE_RATE * 2, 1), dtype=np.float32)
    path = write_signal(str(tmp_path / "silence.flac"), data)
    with AudioSource(path) as source:
        measured = measure_range(source, 0, source.frames)

    assert measured.integrated_lufs <= -119.0
    assert measured.is_silent() is True
    assert measured.true_peak_dbtp <= -119.0


def test_measure_parts_matches_the_whole_and_the_sum(tmp_path):
    amplitude = db_to_amplitude(-12.0)
    data = sine(amplitude, 8.0, channels=2)
    path = write_signal(str(tmp_path / "tone.flac"), data)

    with AudioSource(path) as source:
        total = source.frames
        whole = measure_range(source, 0, total)
        parts = measure_parts([
            Part(source, 0, total // 2),
            Part(source, total // 2, total),
        ])

    assert parts.integrated_lufs == pytest.approx(whole.integrated_lufs, abs=0.3)
    assert parts.duration_frames == total


def test_measurement_streams_in_blocks(tmp_path):
    """A long signal is measured without loading it whole (block-size result)."""
    data = sine(db_to_amplitude(-18.0), 12.0, channels=2)
    path = write_signal(str(tmp_path / "long.flac"), data)
    with AudioSource(path) as source:
        measured = measure_range(source, 0, source.frames)
    assert measured.duration_frames == data.shape[0]
    assert measured.integrated_lufs > -30.0


def test_replaygain_is_reported(tmp_path):
    data = sine(db_to_amplitude(-20.0), 4.0, channels=2)
    path = write_signal(str(tmp_path / "tone.flac"), data)
    with AudioSource(path) as source:
        rg1, rg2 = replaygain(source, 0, source.frames)
    # A -20 dBFS peak sine has an RMS of -23.0 dBFS, so the gain that brings
    # it to the 89 dB reference is 64.82 + 23.0.
    assert rg1 == pytest.approx(64.82 + 23.0, abs=1.0)
    assert rg2 == pytest.approx(rg1, abs=0.5)


# ----------------------------------------------------------------------
# Gain planning
# ----------------------------------------------------------------------
def make_result(lufs: float, peak: float) -> "object":
    from splytnyt.audio.measure import LoudnessResult

    return LoudnessResult(
        integrated_lufs=lufs,
        loudness_range=6.0,
        true_peak_dbtp=peak,
        sample_peak_dbfs=peak - 0.2,
        rms_dbfs=lufs - 0.7,
        duration_frames=100,
    )


def test_gain_plan_that_fits_the_ceiling_is_feasible():
    measured = make_result(-20.0, -10.0)
    plan = plan_shared_gain(measured, -14.0, -1.0)

    assert plan.feasible is True
    assert plan.required_limiting is False
    assert plan.gain_db == pytest.approx(6.0)
    assert plan.peak_after_dbtp == pytest.approx(-4.0)
    assert plan.linear() == pytest.approx(10 ** (6.0 / 20.0))


def test_gain_plan_that_breaches_the_ceiling_is_not_feasible():
    measured = make_result(-20.0, -0.8)
    plan = plan_shared_gain(measured, -14.0, -1.0)

    assert plan.feasible is False
    assert plan.required_limiting is True
    assert plan.gain_db == pytest.approx(6.0)
    assert plan.peak_after_dbtp == pytest.approx(5.2)
    # The safe gain must keep the peak under the ceiling.
    assert plan.safe_gain_db == pytest.approx(-0.2, abs=0.05)
    assert measured.true_peak_dbtp + plan.safe_gain_db <= -1.0 + 1e-6
    assert "ceiling" in plan.reason.lower()


def test_per_track_plan_uses_the_same_rules():
    measured = make_result(-26.0, -12.0)
    plan = plan_per_track_gain(measured, -16.0, -1.0)
    assert plan.mode == "per_track"
    assert plan.feasible is True
    assert plan.gain_db == pytest.approx(10.0)


def test_silent_audio_gets_no_gain():
    from splytnyt.audio.measure import LoudnessResult

    measured = LoudnessResult(integrated_lufs=-120.0, true_peak_dbtp=-120.0)
    plan = plan_shared_gain(measured, -14.0, -1.0)
    assert plan.gain_db == 0.0
    assert plan.feasible is True
    assert "silent" in plan.reason.lower()
