"""Regression tests for where the cuts go, and what happens when they move.

These are the checks the second round of work asked for. Each one runs the
**real** path - a file is written, decoded, analysed, the boundaries are
applied - and then measures the result against the *annotated song starts*,
not against the middle of a gap. The synthetic corpus in :mod:`materials` says
where every song actually begins, so the question "is the cut near the next
song?" has an answer.

The scenarios covered are the ones that used to go wrong:

* a long gap between two songs (the cut used to land in the middle of it),
* a soft intro before loud music (the intro must not be swallowed),
* applause before the next song (applause is not silence and not the song),
* applause over running music (no boundary at all),
* a quiet passage inside a song (no boundary at all),
* a reverb tail, tape hiss and a seamless transition.

Plus the editing rules that have to hold afterwards: moving a boundary moves
the previous track's end, locked tracks are never moved, refinement is one
undoable step, a saved project reopens unchanged, and an unprocessed lossless
export reconstructs the original PCM exactly.

All of it runs on synthetic material. That proves the rules behave as
specified; it does **not** prove the detector is right on real recordings.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import materials as M  # noqa: E402
from splytnyt.app.exporting import export_project  # noqa: E402
from splytnyt.app.jobs import AnalysisJob, RefineJob  # noqa: E402
from splytnyt.app.session import Session  # noqa: E402
from splytnyt.analysis.placement import (  # noqa: E402
    DEFAULT_PRE_ROLL_SECONDS,
    LEAD_IN_PRE_ROLL_SECONDS,
    PLACEMENT_GAP_MIDDLE,
    PLACEMENT_MANUAL,
    PLACEMENT_NEXT_START,
    PLACEMENT_PREVIOUS_END,
    TIGHT_PRE_ROLL_SECONDS,
    PlacementChoice,
    place_cut,
)
from splytnyt.analysis.segment import find_transition_regions  # noqa: E402
from splytnyt.core.model import (  # noqa: E402
    BOUNDARY_BASIS_MANUAL,
    DEFAULT_PRE_ROLL_SECONDS as CORE_DEFAULT_PRE_ROLL_SECONDS,
    ExportSettings,
    Project,
    Track,
)

RATE = M.RATE

#: How close a cut has to be to the annotated start of a song.
#:
#: The default margin is 30 ms and the onset detector finds the attack a
#: few milliseconds early, so a cut about 35 ms in front of the music is the
#: expected result. A tenth of a second is the limit: beyond that the cut
#: has started to drag the tail of the previous song into the new track.
CUT_TOLERANCE_SECONDS = 0.10

#: How close the detected onset has to be to the annotated start. The onset
#: is where the music begins; the cut is a short margin in front of it.
ONSET_TOLERANCE_SECONDS = 0.030

#: The widest gap in the corpus, used to prove a cut is not in the middle.
LONG_GAP_SECONDS = 6.0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def write_material(path: str, material: M.Material) -> str:
    """Write a synthetic recording as a stereo FLAC file."""
    data = np.stack([material.signal] * 2, axis=1).astype(np.float32)
    sf.write(path, data, material.sample_rate, subtype="PCM_16")
    return path


def analyse(tmp_path, material: M.Material, placement: PlacementChoice | None = None,
            name: str = "recording.flac") -> tuple[Session, object]:
    """Run the real analysis job over a real file and apply the result.

    The placement choice is stored on the project first, exactly as the
    interface does it, and the job is then run with the project's own
    choice. The job itself never writes to the project.
    """
    path = write_material(str(tmp_path / name), material)
    session = Session()
    session.import_files([path])
    if placement is not None:
        session.set_placement(placement.mode, placement.pre_roll_seconds)
    job = AnalysisJob(
        session.project, sensitivity="balanced", placement=session.placement_choice()
    )
    outcome = job.run()
    session.apply_analysis(outcome.suggestions, outcome.classifications)
    return session, outcome


def cut_for_second_song(session: Session) -> int:
    """The boundary between the first and the second track."""
    tracks = session.project.sorted_tracks()
    assert len(tracks) >= 2, "the analysis did not find a second track"
    return tracks[1].start_frame


def seconds(frames: int) -> float:
    return frames / float(RATE)


# ----------------------------------------------------------------------
# 1. A long gap: the cut belongs at the next song, not in the middle
# ----------------------------------------------------------------------
def test_long_gap_cut_lands_at_the_next_onset_not_in_the_middle_of_the_gap(tmp_path):
    """The classic bad cut: six seconds of quiet, and the cut in the middle."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 2, f"expected two tracks, got {len(tracks)}"

    song_start = material.song_starts[1]
    cut = tracks[1].start_frame
    assert abs(cut - song_start) / RATE < CUT_TOLERANCE_SECONDS, (
        f"the cut is {abs(cut - song_start) / RATE:.3f} s from the annotated song "
        f"start, which is too far away"
    )
    # The onset itself has to be where the music begins, not merely somewhere
    # in the quiet in front of it.
    onset = tracks[1].onset_frame
    assert onset > 0, "the onset behind the cut was not recorded"
    assert abs(onset - song_start) / RATE < ONSET_TOLERANCE_SECONDS, (
        f"the detected onset is {abs(onset - song_start) / RATE:.3f} s from the "
        f"annotated song start"
    )

    # The middle of the gap is three seconds before the song starts. A cut
    # there is what this application used to produce.
    gap_middle = (material.gaps[0][0] + material.gaps[0][1]) // 2
    assert abs(cut - gap_middle) / RATE > 1.0, (
        "the cut is in the middle of the gap, which is exactly the bug this "
        "change was meant to fix"
    )


def test_the_quiet_before_the_next_song_stays_with_the_previous_track(tmp_path):
    """The intervening audio belongs to the previous track, by default."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert tracks[0].end_frame == tracks[1].start_frame, (
        "the two tracks do not meet at one boundary"
    )
    gap_middle = (material.gaps[0][0] + material.gaps[0][1]) // 2
    assert tracks[0].start_frame <= gap_middle <= tracks[0].end_frame, (
        "the quiet between the songs is not inside the previous track"
    )
    # No audio is lost and none is invented.
    assert sum(track.frames for track in tracks) == session.project.total_frames()


def test_the_cut_carries_the_onset_it_was_derived_from(tmp_path):
    """The track remembers where the music starts, so it can be refined."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    session, _outcome = analyse(tmp_path, material)

    track = session.project.sorted_tracks()[1]
    assert track.boundary_placement == PLACEMENT_NEXT_START
    assert track.onset_frame > 0, "the onset behind the cut was not recorded"
    assert track.transition_end > track.transition_start >= 0, (
        "the transition region was not recorded, so the boundary cannot be refined"
    )
    assert abs(track.onset_frame - material.song_starts[1]) / RATE < 0.2


# ----------------------------------------------------------------------
# 2. A soft intro belongs to the song that follows it
# ----------------------------------------------------------------------
def test_a_soft_intro_is_kept_with_the_next_song(tmp_path):
    """The quiet opening of a song must not be cut away from the song."""
    material = M.soft_intro_before_loud_music()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 2

    song_start = material.song_starts[1]
    cut = tracks[1].start_frame
    assert abs(cut - song_start) / RATE < CUT_TOLERANCE_SECONDS, (
        f"the cut is {abs(cut - song_start) / RATE:.3f} s from the start of the "
        f"soft intro"
    )
    # The cut must be *before* the intro, never after it: a cut later would
    # drop the intro into the previous track.
    assert cut <= song_start, "the soft intro was cut off from its song"
    assert tracks[0].end_frame == cut
    # The onset itself sits on the first note of the intro, not on the first
    # loud passage further into it.
    onset = tracks[1].onset_frame
    assert abs(onset - song_start) / RATE < ONSET_TOLERANCE_SECONDS, (
        f"the detected onset is {abs(onset - song_start) / RATE:.3f} s from the "
        f"start of the soft intro"
    )


def test_a_soft_intro_is_not_mistaken_for_the_gap(tmp_path):
    """The quiet intro must not be treated as a gap to cut in the middle of."""
    material = M.soft_intro_before_loud_music()
    session, _outcome = analyse(tmp_path, material)

    cut = session.project.sorted_tracks()[1].start_frame
    intro_middle = (material.song_starts[1] + material.song_starts[1] + 2.5 * RATE) // 2
    assert abs(cut - intro_middle) / RATE > 0.5, (
        "the cut landed inside the soft intro instead of in front of it"
    )


# ----------------------------------------------------------------------
# 3. Applause is neither silence nor music
# ----------------------------------------------------------------------
def test_applause_before_the_next_song_is_not_taken_for_the_song(tmp_path):
    """The applause must not become the start of the next track."""
    material = M.applause_before_the_next_song()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 2

    song_start = material.song_starts[1]
    cut = tracks[1].start_frame
    assert abs(cut - song_start) / RATE < 0.45, (
        f"the cut is {abs(cut - song_start) / RATE:.3f} s from the real song start"
    )
    # The applause spans the whole gap. A cut inside the applause would give
    # the next track a few seconds of clapping at its start.
    assert cut >= song_start - 0.5 * RATE, "the cut is inside the applause"


def test_applause_over_running_music_creates_no_boundary(tmp_path):
    """Applause on top of music that never stops is not a track boundary."""
    material = M.applause_over_running_music()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 1, (
        f"applause over running music produced {len(tracks)} tracks: "
        f"{[seconds(t.start_frame) for t in tracks]}"
    )
    assert tracks[0].start_frame == 0
    assert tracks[0].end_frame == session.project.total_frames()


# ----------------------------------------------------------------------
# 4. A quiet passage inside a song is not a boundary
# ----------------------------------------------------------------------
def test_a_quiet_passage_inside_a_song_creates_no_boundary(tmp_path):
    """The same music either side of the quiet means one song, not two."""
    material = M.quiet_passage_inside_a_song()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 1, (
        f"a quiet passage inside a song produced {len(tracks)} tracks: "
        f"{[seconds(t.start_frame) for t in tracks]}"
    )
    assert tracks[0].frames == session.project.total_frames()


# ----------------------------------------------------------------------
# 5. Reverb tail, tape hiss, seamless transition
# ----------------------------------------------------------------------
def test_a_reverb_tail_stays_with_the_previous_song(tmp_path):
    """The hall ringing after the last chord belongs to the song that made it."""
    material = M.reverb_tail_between_songs()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 2

    song_start = material.song_starts[1]
    cut = tracks[1].start_frame
    assert abs(cut - song_start) / RATE < CUT_TOLERANCE_SECONDS, (
        f"the cut is {abs(cut - song_start) / RATE:.3f} s from the start of the "
        f"second song"
    )
    onset = tracks[1].onset_frame
    assert abs(onset - song_start) / RATE < ONSET_TOLERANCE_SECONDS, (
        f"the detected onset is {abs(onset - song_start) / RATE:.3f} s from the "
        f"start of the second song"
    )
    # The tail starts before the cut, so it is inside the first track.
    assert tracks[0].end_frame >= material.gaps[0][0], (
        "the reverb tail was taken away from the song that produced it"
    )


def test_tape_hiss_does_not_hide_the_gap_and_the_onset_is_still_found(tmp_path):
    """A noisy tape: the gap sits at the noise floor, not at digital silence."""
    material = M.noisy_tape_gap()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 2, (
        f"the gap in the noisy recording was not found ({len(tracks)} tracks)"
    )
    song_start = material.song_starts[1]
    cut = tracks[1].start_frame
    assert abs(cut - song_start) / RATE < 0.45, (
        f"the cut is {abs(cut - song_start) / RATE:.3f} s from the song start in "
        f"the noisy recording"
    )


def test_a_seamless_transition_is_flagged_for_review(tmp_path):
    """A medley has no break to find, so the cut is offered, not asserted."""
    material = M.seamless_transition()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 2, "a seamless transition produced no boundary at all"

    cut = tracks[1].start_frame
    assert abs(cut - material.song_starts[1]) / RATE < 0.35, (
        "the proposed cut is nowhere near the change of song"
    )
    assert tracks[1].boundary_uncertain is True, (
        "a transition without a break must be marked as needing a human decision"
    )
    assert tracks[1].boundary_assessment() == "Needs review"


def test_three_songs_are_cut_at_their_own_onsets(tmp_path):
    """A recording with several gaps: every cut lands on its own song."""
    material = M.three_songs()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 3
    for track, song_start in zip(tracks[1:], material.song_starts[1:]):
        assert abs(track.start_frame - song_start) / RATE < CUT_TOLERANCE_SECONDS, (
            f"a cut is {abs(track.start_frame - song_start) / RATE:.3f} s from "
            f"its song start"
        )
        assert abs(track.onset_frame - song_start) / RATE < ONSET_TOLERANCE_SECONDS, (
            f"an onset is {abs(track.onset_frame - song_start) / RATE:.3f} s "
            f"from its song start"
        )


# ----------------------------------------------------------------------
# 6. The placement choices
# ----------------------------------------------------------------------
def _long_gap_region(material: M.Material):
    """The transition region of the long-gap material."""
    regions = find_transition_regions(material.signal, RATE, "balanced")
    assert regions, "the long gap was not found as a transition region"
    gap_start, gap_end = material.gaps[0]
    for region in regions:
        if region.start_frame <= gap_start and region.end_frame >= gap_end - RATE:
            return region
    # Fall back to the region that overlaps the gap the most.
    return max(regions, key=lambda r: min(r.end_frame, gap_end) - max(r.start_frame, gap_start))


def test_the_four_placement_modes_put_the_cut_where_they_say(tmp_path):
    """Each placement choice has to move the cut in the promised direction."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    region = _long_gap_region(material)
    gap_start, gap_end = material.gaps[0]
    song_start = material.song_starts[1]

    next_start = place_cut(
        material.signal, RATE, region, choice=PlacementChoice(mode=PLACEMENT_NEXT_START)
    )
    assert abs(next_start.frame - song_start) / RATE < CUT_TOLERANCE_SECONDS
    assert next_start.details["placement"] == PLACEMENT_NEXT_START
    # The cut sits in front of the music by exactly the requested margin.
    assert next_start.details["applied_pre_roll_ms"] == 30
    assert next_start.details["requested_pre_roll_ms"] == 30
    assert next_start.frame < song_start, "the cut is not in front of the song"

    previous_end = place_cut(
        material.signal, RATE, region, choice=PlacementChoice(mode=PLACEMENT_PREVIOUS_END)
    )
    # The cut goes to the start of the transition region, which is where the
    # previous song was last heard - a moment before the annotated gap
    # begins, because the last chord fades out.
    assert previous_end.frame == region.start_frame
    assert abs(previous_end.frame - gap_start) / RATE < 0.25
    assert previous_end.details["placement"] == PLACEMENT_PREVIOUS_END

    gap_middle = place_cut(
        material.signal, RATE, region, choice=PlacementChoice(mode=PLACEMENT_GAP_MIDDLE)
    )
    assert gap_middle.frame == (region.start_frame + region.end_frame) // 2
    assert gap_middle.details["placement"] == PLACEMENT_GAP_MIDDLE

    manual_frame = gap_start + 1234
    manual = place_cut(
        material.signal, RATE, region,
        choice=PlacementChoice(mode=PLACEMENT_MANUAL), manual_frame=manual_frame,
    )
    assert manual.frame == manual_frame
    assert manual.details["placement"] == PLACEMENT_MANUAL


def test_the_pre_roll_is_configurable(tmp_path):
    """A longer pre-roll moves the cut further in front of the song."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    region = _long_gap_region(material)

    short = place_cut(
        material.signal, RATE, region,
        choice=PlacementChoice(mode=PLACEMENT_NEXT_START, pre_roll_seconds=0.1),
    )
    long = place_cut(
        material.signal, RATE, region,
        choice=PlacementChoice(mode=PLACEMENT_NEXT_START, pre_roll_seconds=0.6),
    )
    assert long.frame == short.frame - int(0.5 * RATE), (
        "a longer pre-roll did not move the cut further in front of the song"
    )
    # The cut never moves in front of the transition itself.
    assert long.frame >= region.start_frame


def test_the_default_pre_roll_is_a_small_tight_margin():
    """The default margin is a safety margin, not a lump of silence.

    A tenth of a second in front of every song is long enough to hear as a
    click or a breath of the previous room. The default therefore sits in the
    20-50 ms band: enough to keep the attack intact, short enough not to drag
    the tail of the previous song into the new track.
    """
    assert 0.020 <= DEFAULT_PRE_ROLL_SECONDS <= 0.050, (
        f"the default pre-roll is {DEFAULT_PRE_ROLL_SECONDS * 1000:.0f} ms, "
        f"which is outside the 20-50 ms band"
    )
    # One single source of truth: the project, the analysis and the refinement
    # must not be able to disagree about the default.
    assert DEFAULT_PRE_ROLL_SECONDS == pytest.approx(CORE_DEFAULT_PRE_ROLL_SECONDS)
    assert Project().placement_pre_roll_seconds == pytest.approx(
        CORE_DEFAULT_PRE_ROLL_SECONDS
    )


def test_the_tight_and_lead_in_presets_sit_where_they_are_promised():
    """The two presets are a short margin and a deliberately longer one."""
    assert 0.015 <= TIGHT_PRE_ROLL_SECONDS <= 0.030
    assert 0.100 <= LEAD_IN_PRE_ROLL_SECONDS <= 0.250
    assert TIGHT_PRE_ROLL_SECONDS < DEFAULT_PRE_ROLL_SECONDS < LEAD_IN_PRE_ROLL_SECONDS


def test_the_project_and_the_analysis_cannot_disagree_about_the_default():
    """A default that drifts between modules is a real, measured bug."""
    project = Project()
    session = Session()
    session.set_placement(PLACEMENT_NEXT_START)
    assert session.placement_choice().pre_roll_seconds == pytest.approx(
        project.placement_pre_roll_seconds
    )
    assert session.placement_choice().pre_roll_seconds == pytest.approx(
        DEFAULT_PRE_ROLL_SECONDS
    )


def test_the_project_remembers_the_placement_choice(tmp_path):
    """The choice is part of the project, not a setting that gets lost."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    session, _outcome = analyse(
        tmp_path, material, placement=PlacementChoice(mode=PLACEMENT_PREVIOUS_END)
    )
    assert session.project.placement_mode == PLACEMENT_PREVIOUS_END

    session.set_placement(PLACEMENT_NEXT_START, pre_roll_seconds=0.25)
    assert session.placement_choice().mode == PLACEMENT_NEXT_START
    assert session.placement_choice().pre_roll_seconds == pytest.approx(0.25)


def test_a_previous_end_placement_puts_the_quiet_at_the_end_of_the_first_track(tmp_path):
    """The other placement choice really does move the cut to the gap start."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    session, _outcome = analyse(
        tmp_path, material, placement=PlacementChoice(mode=PLACEMENT_PREVIOUS_END)
    )
    tracks = session.project.sorted_tracks()
    gap_start = material.gaps[0][0]
    song_start = material.song_starts[1]

    assert tracks[1].boundary_placement == PLACEMENT_PREVIOUS_END
    assert abs(tracks[1].start_frame - gap_start) / RATE < 0.25, (
        "the cut is not where the previous song stopped"
    )
    # The whole quiet stretch now belongs to the first track.
    assert tracks[0].end_frame == tracks[1].start_frame
    assert tracks[1].start_frame <= song_start - 1.0 * RATE, (
        "the quiet is not with the previous track"
    )


# ----------------------------------------------------------------------
# 7. Moving a boundary
# ----------------------------------------------------------------------
def test_moving_a_boundary_moves_the_previous_track_end(tmp_path):
    """The two tracks must always meet at the same frame."""
    material = M.three_songs()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    old = tracks[1].start_frame
    new = old - int(2.0 * RATE)

    assert session.move_boundary(old, new) is True

    tracks = session.project.sorted_tracks()
    assert tracks[0].end_frame == new, "the previous track did not follow the boundary"
    assert tracks[1].start_frame == new
    assert sum(track.frames for track in tracks) == session.project.total_frames(), (
        "moving the boundary changed the amount of audio"
    )


def test_a_moved_boundary_is_marked_as_a_human_decision(tmp_path):
    """A boundary a human moved is not a guess any more."""
    material = M.three_songs()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    session.move_boundary(tracks[1].start_frame, tracks[1].start_frame - 1000)

    tracks = session.project.sorted_tracks()
    assert tracks[0].boundary_basis == BOUNDARY_BASIS_MANUAL
    assert tracks[1].boundary_basis == BOUNDARY_BASIS_MANUAL
    assert tracks[1].boundary_assessment() == "Manual"


def test_a_boundary_cannot_be_moved_outside_the_two_tracks_it_separates(tmp_path):
    """A move that would swallow another track is refused."""
    material = M.three_songs()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    old = tracks[1].start_frame
    assert session.move_boundary(old, tracks[2].end_frame + 10) is False
    assert session.project.sorted_tracks()[1].start_frame == old


# ----------------------------------------------------------------------
# 8. Refinement, locks, undo
# ----------------------------------------------------------------------
def test_a_locked_track_is_never_moved_by_a_refinement(tmp_path):
    """Locks protect a decision the user already made."""
    material = M.three_songs()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    locked_id = tracks[1].track_id
    locked_start = tracks[1].start_frame
    unlocked_start = tracks[2].start_frame
    session.toggle_lock(locked_id)

    # A placement that would move every boundary to the start of its gap.
    job = RefineJob(session.project, placement=PlacementChoice(mode=PLACEMENT_PREVIOUS_END))
    proposals = job.run()

    assert all(proposal.track_id != locked_id for proposal in proposals), (
        "a refinement proposed moving a locked track"
    )

    session.apply_refinement(proposals)
    tracks = session.project.sorted_tracks()
    assert tracks[1].start_frame == locked_start, "a locked boundary was moved"
    assert tracks[1].locked is True
    # The unlocked boundary did move, so the refinement really did run.
    assert tracks[2].start_frame != unlocked_start, (
        "the refinement did not move the unlocked boundary either, so the test "
        "would pass even if nothing ran"
    )


def test_a_refinement_is_one_undoable_step(tmp_path):
    """One undo puts every boundary back where it was."""
    material = M.three_songs()
    session, _outcome = analyse(tmp_path, material)

    before = [track.start_frame for track in session.project.sorted_tracks()]
    job = RefineJob(session.project, placement=PlacementChoice(mode=PLACEMENT_PREVIOUS_END))
    proposals = job.run()
    assert proposals, "the refinement proposed nothing to do"

    applied = session.apply_refinement(proposals)
    assert applied == len(proposals)
    after = [track.start_frame for track in session.project.sorted_tracks()]
    assert after != before, "the refinement changed nothing"

    assert session.undo_action() is True
    assert [track.start_frame for track in session.project.sorted_tracks()] == before


def test_a_refinement_with_the_same_placement_changes_nothing(tmp_path):
    """Re-running the same rule must not shuffle the cuts about."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    session, _outcome = analyse(tmp_path, material)

    before = [track.start_frame for track in session.project.sorted_tracks()]
    job = RefineJob(session.project)
    proposals = job.run()
    assert not proposals, "a refinement with the same placement proposed a move"

    session.apply_refinement(proposals)
    assert [track.start_frame for track in session.project.sorted_tracks()] == before


def test_undo_and_redo_restore_a_moved_boundary(tmp_path):
    """Undo and redo still work after the boundary work."""
    material = M.three_songs()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    old = tracks[1].start_frame
    new = old - 5000
    session.move_boundary(old, new)
    assert session.project.sorted_tracks()[1].start_frame == new

    assert session.undo_action() is True
    assert session.project.sorted_tracks()[1].start_frame == old
    assert session.project.sorted_tracks()[0].end_frame == old

    assert session.redo_action() is True
    assert session.project.sorted_tracks()[1].start_frame == new
    assert session.project.sorted_tracks()[0].end_frame == new


def test_re_analysis_never_overwrites_a_locked_track(tmp_path):
    """A locked track keeps its boundaries through a new analysis."""
    material = M.three_songs()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    locked_id = tracks[1].track_id
    locked_start = tracks[1].start_frame
    session.toggle_lock(locked_id)

    # A second analysis with a different placement.
    job = AnalysisJob(
        session.project, sensitivity="balanced",
        placement=PlacementChoice(mode=PLACEMENT_PREVIOUS_END),
    )
    outcome = job.run()
    session.apply_analysis(outcome.suggestions, outcome.classifications)

    tracks = session.project.sorted_tracks()
    locked = next(t for t in tracks if t.track_id == locked_id)
    assert locked.start_frame == locked_start, "a re-analysis moved a locked track"
    assert locked.locked is True


# ----------------------------------------------------------------------
# 9. Saving and reopening
# ----------------------------------------------------------------------
def test_a_saved_project_reopens_with_the_same_boundaries(tmp_path):
    """Everything about a boundary survives the round trip."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    session, _outcome = analyse(tmp_path, material)

    session.set_placement(PLACEMENT_NEXT_START, pre_roll_seconds=0.2)
    path = str(tmp_path / "project.splytnyt")
    session.save_project(path)

    reopened = Session()
    reopened.open_project(path)

    original = session.project.sorted_tracks()
    copy = reopened.project.sorted_tracks()
    assert len(copy) == len(original)
    for before, after in zip(original, copy):
        assert after.start_frame == before.start_frame
        assert after.end_frame == before.end_frame
        assert after.boundary_placement == before.boundary_placement
        assert after.onset_frame == before.onset_frame
        assert after.transition_start == before.transition_start
        assert after.transition_end == before.transition_end
        assert after.boundary_uncertain == before.boundary_uncertain
        assert after.boundary_basis == before.boundary_basis
    assert reopened.project.placement_mode == PLACEMENT_NEXT_START
    assert reopened.project.placement_pre_roll_seconds == pytest.approx(0.2)


def test_opening_a_project_does_not_change_it(tmp_path):
    """Opening a saved project must never move a boundary behind the user's back."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    session, _outcome = analyse(tmp_path, material)

    # A boundary a human placed by hand, at an odd place.
    tracks = session.project.sorted_tracks()
    session.move_boundary(tracks[1].start_frame, tracks[1].start_frame - 731)

    path = str(tmp_path / "project.splytnyt")
    session.save_project(path)
    before = [
        (t.track_id, t.start_frame, t.end_frame, t.boundary_basis)
        for t in session.project.sorted_tracks()
    ]

    reopened = Session()
    reopened.open_project(path)
    after = [
        (t.track_id, t.start_frame, t.end_frame, t.boundary_basis)
        for t in reopened.project.sorted_tracks()
    ]
    assert after == before, "opening the project changed a boundary"
    assert reopened.dirty is False, "a freshly opened project is marked as changed"


def test_a_locked_track_stays_locked_after_reopening(tmp_path):
    """A lock is part of the decision, so it is saved with it."""
    material = M.three_songs()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    session.toggle_lock(tracks[1].track_id)

    path = str(tmp_path / "project.splytnyt")
    session.save_project(path)

    reopened = Session()
    reopened.open_project(path)
    assert any(track.locked for track in reopened.project.sorted_tracks())


# ----------------------------------------------------------------------
# 10. The audio is never lost
# ----------------------------------------------------------------------
def test_unprocessed_tracks_reconstruct_the_original_pcm_exactly(tmp_path):
    """Exporting the analysed tracks reproduces the source bit for bit.

    This covers the three things the product promises at once: nothing is
    deleted, no silence is inserted and no fades are added.
    """
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    session, _outcome = analyse(tmp_path, material)

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.mode = "separate"
    settings.gain_mode = "untouched"
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    report = export_project(session.project, settings)
    assert len(report.files) == len(session.project.sorted_tracks())

    exported = []
    for path in report.files:
        data, rate = sf.read(path, always_2d=True, dtype="int16")
        assert rate == RATE
        exported.append(data)

    concatenated = np.concatenate(exported, axis=0)
    expected, _rate = sf.read(session.project.sources[0].path, always_2d=True, dtype="int16")
    assert concatenated.shape == expected.shape, (
        f"the export has {concatenated.shape[0]} frames, the source has "
        f"{expected.shape[0]}"
    )
    assert np.array_equal(concatenated, expected), (
        "the exported audio differs from the source: something was deleted, "
        "inserted or faded"
    )


def test_the_export_never_overwrites_the_source_recording(tmp_path):
    """The source file is still exactly the same after an export."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    session, _outcome = analyse(tmp_path, material)
    source_path = session.project.sources[0].path
    before = os.stat(source_path)

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.gain_mode = "untouched"
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    export_project(session.project, settings)

    after = os.stat(source_path)
    assert before.st_size == after.st_size
    assert before.st_mtime_ns == after.st_mtime_ns


def test_a_track_that_covers_the_whole_recording_is_not_claimed_as_confident(tmp_path):
    """The untouched recording must not look like an analysed result."""
    material = M.applause_over_running_music()
    path = write_material(str(tmp_path / "recording.flac"), material)
    session = Session()
    session.import_files([path])

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 1
    assert tracks[0].boundary_assessment() == "Not assessed"
    assert tracks[0].identity_assessment() == "Not assessed"


def test_a_manual_boundary_is_not_reported_as_a_confidence_percentage(tmp_path):
    """The interface shows a word, never a number it cannot stand behind."""
    material = M.two_songs_with_a_long_gap(gap_seconds=LONG_GAP_SECONDS)
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    session.move_boundary(tracks[1].start_frame, tracks[1].start_frame - 100)

    assessments = {t.boundary_assessment() for t in session.project.sorted_tracks()}
    for word in assessments:
        assert "%" not in word, "a percentage is being shown instead of a word"
        assert word in {"Manual", "Not assessed", "Needs review", "High", "Fair", "Low"}
