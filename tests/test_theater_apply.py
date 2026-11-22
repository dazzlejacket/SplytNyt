"""Tests for applying a theatrical analysis to a project.

The section types have to reach the tracks. They did not for a long time:
the boundary sits a pre-roll before the section it opens, so the two do not
start on the same frame, and the lookup that matched them exactly found
nothing. Every track came out as "unknown", and with it the songs-only
export and the naming both failed.
"""
from __future__ import annotations

import numpy as np
import pytest

from splytnyt.app.jobs import TheaterOutcome
from splytnyt.app.session import Session
from splytnyt.core.model import Source

SAMPLE_RATE = 48000


def _session(total_seconds: float = 60.0) -> Session:
    session = Session()
    session.new_project("theatre test")
    project = session.project
    project.sources = [
        Source(
            source_id="src-test",
            path="test.mp3",
            frames=int(total_seconds * SAMPLE_RATE),
            sample_rate=SAMPLE_RATE,
            channels=2,
        )
    ]
    return session


def _outcome() -> TheaterOutcome:
    """Three sections, with the boundaries a pre-roll before them."""
    from splytnyt.analysis.segment import Suggestion

    pre_roll = int(0.030 * SAMPLE_RATE)
    starts = [0, 20 * SAMPLE_RATE, 40 * SAMPLE_RATE]
    ends = [20 * SAMPLE_RATE, 40 * SAMPLE_RATE, 60 * SAMPLE_RATE]
    types = ["musical_number", "dialogue", "applause"]

    outcome = TheaterOutcome()
    for index in range(1, len(starts)):
        outcome.suggestions.append(
            Suggestion(
                frame=starts[index] - pre_roll,
                confidence=0.6,
                reason="the audio changes character here",
                kind="theater",
            )
        )
    for start, end, section_type in zip(starts, ends, types):
        outcome.classifications.append(
            (start, end, section_type, 0.7, [f"it is {section_type}"])
        )
    return outcome


class TestApplyTheatricalAnalysis:
    def test_the_sections_become_tracks(self):
        session = _session()
        count = session.apply_theater_analysis(_outcome())
        assert count == 3

    def test_every_track_gets_its_section_type(self):
        """The bug: every track came out as "unknown"."""
        session = _session()
        session.apply_theater_analysis(_outcome())
        types = [t.section_type for t in session.project.sorted_tracks()]
        assert types == ["musical_number", "dialogue", "applause"]

    def test_the_reasons_travel_with_the_track(self):
        session = _session()
        session.apply_theater_analysis(_outcome())
        track = session.project.sorted_tracks()[0]
        assert any("it is musical_number" in reason for reason in track.reasons)

    def test_the_tracks_tile_the_whole_recording(self):
        session = _session()
        session.apply_theater_analysis(_outcome())
        tracks = session.project.sorted_tracks()
        assert tracks[0].start_frame == 0
        assert tracks[-1].end_frame == session.project.total_frames()
        for first, second in zip(tracks, tracks[1:]):
            assert first.end_frame == second.start_frame

    def test_it_is_one_undoable_step(self):
        session = _session()
        session.apply_theater_analysis(_outcome())
        assert session.undo_action()
        assert not session.project.tracks
