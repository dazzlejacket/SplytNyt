"""Tests for the section classifier.

The heuristic used to call *everything* music, including clear dialogue:
its spectral-flatness thresholds were unreachable on real material. These
tests pin the behaviour down with synthetic signals, so the fix cannot
quietly come back.
"""
from __future__ import annotations

import numpy as np
import pytest

from splytnyt.analysis.aec import EventScores
from splytnyt.analysis.classify import classify_section, classify_with_model


def _chord(sample_rate: int = 44100, seconds: float = 3.0) -> np.ndarray:
    """A sustained chord: stable, tonal, barely modulated."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    signal = sum(
        (0.4 / k) * np.sin(2 * np.pi * f * t) for k, f in enumerate((196, 247, 294), start=1)
    )
    return signal.astype(np.float32)


def _speech_like(sample_rate: int = 44100, seconds: float = 3.0) -> np.ndarray:
    """A signal that moves its level several times a second."""
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    carrier = (
        0.4 * np.sin(2 * np.pi * 140.0 * t)
        + 0.2 * np.sin(2 * np.pi * 700.0 * t)
        + 0.1 * np.sin(2 * np.pi * 1800.0 * t)
    )
    envelope = 0.25 + 0.75 * np.abs(np.sin(2 * np.pi * 4.5 * t))
    return (carrier * envelope).astype(np.float32)


def _events(music: float, speech: float, applause: float, singing: float = 0.0):
    frames = 20
    return EventScores(
        music=np.full(frames, music, dtype=np.float32),
        speech=np.full(frames, speech, dtype=np.float32),
        singing=np.full(frames, singing, dtype=np.float32),
        applause=np.full(frames, applause, dtype=np.float32),
        silence=np.zeros(frames, dtype=np.float32),
        frame_seconds=0.48,
        from_model=True,
        source="test",
    )


class TestHeuristic:
    def test_a_sustained_chord_is_music(self):
        guess = classify_section(_chord(), 44100)
        assert guess.section_type == "music"
        assert guess.reasons

    def test_a_modulated_signal_is_not_called_music(self):
        """The bug: this used to be labelled "music" with high confidence."""
        guess = classify_section(_speech_like(), 44100)
        assert guess.section_type != "music"
        assert guess.section_type == "speech"

    def test_the_reasons_say_it_is_a_heuristic(self):
        guess = classify_section(_chord(), 44100)
        assert any("heuristic" in reason for reason in guess.reasons)

    def test_silence_is_recognised(self):
        guess = classify_section(np.zeros(44100, dtype=np.float32), 44100)
        assert guess.section_type == "silence"

    def test_an_empty_stretch_is_silence(self):
        guess = classify_section(np.zeros(0, dtype=np.float32), 44100)
        assert guess.section_type == "silence"


class TestTrainedClassifier:
    def test_music_scores_give_music(self):
        guess = classify_with_model(_events(0.8, 0.05, 0.0))
        assert guess.section_type == "music"
        assert any("classifier" in reason for reason in guess.reasons)

    def test_speech_scores_give_speech(self):
        guess = classify_with_model(_events(0.05, 0.7, 0.0))
        assert guess.section_type == "speech"

    def test_applause_scores_give_applause(self):
        guess = classify_with_model(_events(0.1, 0.0, 0.9))
        assert guess.section_type == "applause"

    def test_no_scores_give_no_claim(self):
        guess = classify_with_model(None)
        assert guess.section_type == "unknown"
        assert guess.confidence < 0.5

    def test_the_model_path_is_used_when_scores_are_given(self):
        """The classifier decides, not the heuristic."""
        guess = classify_section(
            _speech_like(), 44100, events=_events(0.8, 0.02, 0.0)
        )
        assert guess.section_type == "music"
        assert any("classifier" in reason for reason in guess.reasons)
