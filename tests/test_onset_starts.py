"""Where a track actually starts, and whether the whole application agrees.

The complaint this file answers is precise: *the cut was in front of the
visible attack, and it dragged the tail of the previous song in with it*.

So every test here measures the two positions separately:

* the **onset** - the frame the music itself begins at, which has to sit on
  the annotated song start,
* the **cut** - the frame the track starts at, which has to be the onset minus
  a small margin.

The distinction matters. An onset that is accurate but a cut that is a tenth
of a second in front of it still gives the new track a breath of the previous
room; an accurate cut with a wrong onset clips the first note.

The materials are synthetic, so this proves the *rules* hold and nothing
more: it does not prove the detector is right on a real concert recording.
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
    PLACEMENT_NEXT_START,
    PLACEMENT_PREVIOUS_END,
    TIGHT_PRE_ROLL_SECONDS,
    PlacementChoice,
)
from splytnyt.core.model import (  # noqa: E402
    BOUNDARY_BASIS_ACCEPTED,
    ExportSettings,
)

RATE = M.RATE

#: The cut has to be within this of the annotated song start. The default
#: margin is 30 ms and the onset finder lands a few milliseconds early, so
#: roughly 35 ms in front of the music is the expected result.
CUT_TOLERANCE_SECONDS = 0.10

#: The onset itself has to be within this of the annotated song start.
ONSET_TOLERANCE_SECONDS = 0.030


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def analyse(tmp_path, material: M.Material, placement=None, name="recording.flac"):
    """Run the real analysis job over a real file and apply the result."""
    path = str(tmp_path / name)
    data = np.stack([material.signal] * 2, axis=1).astype(np.float32)
    sf.write(path, data, material.sample_rate, subtype="PCM_16")

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


def export(session: Session, tmp_path) -> list[str]:
    """Export the project and return the written files, in track order."""
    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.mode = "separate"
    settings.gain_mode = "untouched"
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False
    report = export_project(session.project, settings)
    return list(report.files)


# ----------------------------------------------------------------------
# 1. A slow attack: the cut belongs at the beginning of the fade
# ----------------------------------------------------------------------
def test_a_slow_attack_is_cut_at_the_beginning_of_the_fade(tmp_path):
    """A note that fades in over a quarter of a second starts at the fade.

    The music begins at the *start* of the fade. A cut placed where the fade
    reaches full volume throws the first 250 ms of the note away.
    """
    material = M.slow_attack_before_a_song(attack_seconds=0.25)
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 2

    song_start = material.song_starts[1]
    cut = tracks[1].start_frame
    onset = tracks[1].onset_frame

    assert abs(onset - song_start) / RATE < ONSET_TOLERANCE_SECONDS, (
        f"the onset is {abs(onset - song_start) / RATE:.3f} s from the start of "
        f"the fade"
    )
    assert abs(cut - song_start) / RATE < CUT_TOLERANCE_SECONDS, (
        f"the cut is {abs(cut - song_start) / RATE:.3f} s from the start of the "
        f"fade"
    )
    assert cut <= song_start, "the cut is after the music starts"


def test_a_slow_attack_is_not_snapped_to_the_loudest_point(tmp_path):
    """The onset is the first note, not the loudest moment of the fade."""
    material = M.slow_attack_before_a_song(attack_seconds=0.25)
    session, _outcome = analyse(tmp_path, material)

    track = session.project.sorted_tracks()[1]
    loudest = material.song_starts[1] + int(0.25 * RATE)

    assert abs(track.onset_frame - loudest) / RATE > 0.15, (
        "the onset was snapped to the point where the fade reaches full "
        "volume, so the start of the note is missing from the track"
    )


def test_the_soft_attack_is_kept_with_the_song_it_belongs_to(tmp_path):
    """The fade-in belongs to the new song, not to the end of the old one."""
    material = M.slow_attack_before_a_song(attack_seconds=0.25)
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert tracks[0].end_frame == tracks[1].start_frame
    assert tracks[1].start_frame <= material.song_starts[1], (
        "the fade-in was left at the end of the previous track"
    )


# ----------------------------------------------------------------------
# 2. Crowd noise in the gap
# ----------------------------------------------------------------------
def test_crowd_noise_does_not_pull_the_cut_into_the_murmur(tmp_path):
    """A gap full of crowd noise is still a gap; the cut is still the song."""
    material = M.crowd_noise_before_the_next_song()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 2, f"expected two tracks, got {len(tracks)}"

    song_start = material.song_starts[1]
    cut = tracks[1].start_frame

    assert abs(cut - song_start) / RATE < CUT_TOLERANCE_SECONDS, (
        f"the cut is {abs(cut - song_start) / RATE:.3f} s from the real song "
        f"start, so the murmur of the crowd ended up inside the new track"
    )
    onset = tracks[1].onset_frame
    assert abs(onset - song_start) / RATE < ONSET_TOLERANCE_SECONDS, (
        f"the onset is {abs(onset - song_start) / RATE:.3f} s from the real "
        f"song start"
    )

    # The middle of the murmur is 1.5 s before the song. A cut there is what
    # this application used to produce.
    murmur_middle = (material.gaps[0][0] + material.gaps[0][1]) // 2
    assert abs(cut - murmur_middle) / RATE > 1.0, (
        "the cut is in the middle of the crowd noise"
    )


def test_the_crowd_noise_stays_at_the_end_of_the_previous_track(tmp_path):
    """Nothing is deleted, and the noise is not handed to the new song."""
    material = M.crowd_noise_before_the_next_song()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert tracks[0].end_frame == tracks[1].start_frame
    assert sum(track.frames for track in tracks) == session.project.total_frames()


# ----------------------------------------------------------------------
# 3. The tail of the previous song
# ----------------------------------------------------------------------
def test_the_previous_songs_tail_does_not_win_the_onset(tmp_path):
    """A ringing tail is not the start of the next song.

    This is the case that used to be badly wrong: the detector found the
    decaying tail of the previous song as the first sound after the gap and
    put the cut there, almost three seconds before the real song.
    """
    material = M.previous_song_tail_before_the_gap()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 2

    song_start = material.song_starts[1]
    cut = tracks[1].start_frame
    onset = tracks[1].onset_frame

    assert abs(cut - song_start) / RATE < CUT_TOLERANCE_SECONDS, (
        f"the cut is {abs(cut - song_start) / RATE:.3f} s from the real song "
        f"start - the previous song's tail was mistaken for the next song"
    )
    assert abs(onset - song_start) / RATE < ONSET_TOLERANCE_SECONDS, (
        f"the onset is {abs(onset - song_start) / RATE:.3f} s from the real "
        f"song start - the tail was mistaken for the attack"
    )


def test_the_tail_stays_with_the_song_that_produced_it(tmp_path):
    """The whole tail belongs to the previous track."""
    material = M.previous_song_tail_before_the_gap()
    session, _outcome = analyse(tmp_path, material)

    tracks = session.project.sorted_tracks()
    tail_end = material.song_ends[0]
    assert tracks[0].end_frame >= tail_end, (
        "the tail of the previous song was taken away from it"
    )
    assert tracks[1].start_frame >= tail_end, (
        "the new track starts inside the previous song's tail"
    )


# ----------------------------------------------------------------------
# 4. The cut and the margin
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "builder",
    [
        lambda: M.two_songs_with_a_long_gap(gap_seconds=6.0),
        M.soft_intro_before_loud_music,
        M.slow_attack_before_a_song,
        M.crowd_noise_before_the_next_song,
        M.previous_song_tail_before_the_gap,
    ],
    ids=["long-gap", "soft-intro", "slow-attack", "crowd", "tail"],
)
def test_the_cut_is_the_onset_minus_the_applied_margin(tmp_path, builder):
    """The margin between the cut and the music is the margin that was asked for.

    This is the arithmetic the interface shows: the cut is the onset minus the
    applied margin, and the applied margin is recorded on the track.
    """
    material = builder()
    session, _outcome = analyse(tmp_path, material)

    track = session.project.sorted_tracks()[1]
    assert track.onset_frame > 0, "no onset was recorded for the cut"
    assert track.boundary_pre_roll_seconds >= 0.0, (
        "the applied margin was not recorded on the track"
    )

    expected = track.onset_frame - int(round(track.boundary_pre_roll_seconds * RATE))
    assert abs(track.start_frame - expected) <= 1, (
        f"the cut is at {track.start_frame} but the onset minus the applied "
        f"margin is {expected}"
    )

    # The applied margin is never larger than the requested one: the cut is
    # never placed in front of the transition itself.
    assert track.boundary_pre_roll_seconds <= DEFAULT_PRE_ROLL_SECONDS + 1e-6


def test_the_applied_margin_is_never_larger_than_the_requested_one(tmp_path):
    """A margin that does not fit in the gap is cut back, and the truth is reported.

    The cut may never be placed in front of the transition itself: that would
    cut into the music of the previous song. So a 900 ms margin in a 600 ms
    gap becomes the largest margin the gap allows, and the applied margin -
    not the requested one - is what the interface reports.
    """
    material = M.two_songs_with_a_long_gap(gap_seconds=0.6)
    session, _outcome = analyse(
        tmp_path,
        material,
        placement=PlacementChoice(mode=PLACEMENT_NEXT_START, pre_roll_seconds=0.9),
    )

    tracks = session.project.sorted_tracks()
    assert len(tracks) == 2, (
        f"the short gap was not found at all ({len(tracks)} tracks)"
    )
    track = tracks[1]
    assert track.boundary_pre_roll_seconds <= 0.9 + 1e-6, (
        f"the applied margin is {track.boundary_pre_roll_seconds:.3f} s, "
        f"which is more than was asked for"
    )
    assert track.boundary_pre_roll_seconds < 0.9, (
        "the margin should have been cut back to fit the gap"
    )
    # The cut sits at the start of the transition, not in front of it.
    assert track.start_frame >= track.transition_start, (
        "the cut was placed in front of the transition, inside the previous song"
    )
    assert track.start_frame == track.transition_start, (
        "the margin should have been cut back to exactly the transition start"
    )


def test_the_presets_move_the_cut_by_the_amount_they_promise(tmp_path):
    """Tight start is tighter; keep more lead-in is longer."""
    material = M.two_songs_with_a_long_gap(gap_seconds=6.0)

    def cut_with(pre_roll: float) -> int:
        session, _outcome = analyse(
            tmp_path,
            material,
            placement=PlacementChoice(
                mode=PLACEMENT_NEXT_START, pre_roll_seconds=pre_roll
            ),
            name=f"recording-{int(pre_roll * 1000)}.flac",
        )
        return session.project.sorted_tracks()[1].start_frame

    tight = cut_with(TIGHT_PRE_ROLL_SECONDS)
    default = cut_with(DEFAULT_PRE_ROLL_SECONDS)
    lead_in = cut_with(LEAD_IN_PRE_ROLL_SECONDS)

    assert tight > default > lead_in, (
        "a smaller margin has to put the cut closer to the music"
    )
    # The cut is the onset minus the margin, so a bigger margin means a cut
    # that is further in front of the music.
    assert abs((tight - default) + (TIGHT_PRE_ROLL_SECONDS - DEFAULT_PRE_ROLL_SECONDS) * RATE) <= 2
    assert abs((lead_in - default) + (LEAD_IN_PRE_ROLL_SECONDS - DEFAULT_PRE_ROLL_SECONDS) * RATE) <= 2


def test_changing_the_margin_of_one_cut_leaves_the_others_alone(tmp_path):
    """A margin change has a scope, and the scope is the cut that was selected."""
    material = M.three_songs()
    session, _outcome = analyse(tmp_path, material)

    before = [track.start_frame for track in session.project.sorted_tracks()]
    target = session.project.sorted_tracks()[1]

    new_frame = session.reposition_boundary(
        target.start_frame, pre_roll_seconds=LEAD_IN_PRE_ROLL_SECONDS
    )
    assert new_frame is not None, "the margin change did not move the cut"

    after = session.project.sorted_tracks()
    assert after[1].start_frame == new_frame
    assert after[0].end_frame == new_frame, (
        "the previous track did not follow the cut"
    )
    assert after[2].start_frame == before[2], (
        "a different boundary was moved as well"
    )
    assert after[1].boundary_pre_roll_seconds == pytest.approx(
        LEAD_IN_PRE_ROLL_SECONDS, abs=0.002
    )


def test_a_margin_change_is_one_undoable_step(tmp_path):
    """One undo puts the cut back where it was."""
    material = M.three_songs()
    session, _outcome = analyse(tmp_path, material)

    before = [track.start_frame for track in session.project.sorted_tracks()]
    target = session.project.sorted_tracks()[1]
    session.reposition_boundary(target.start_frame, pre_roll_seconds=0.4)
    assert [track.start_frame for track in session.project.sorted_tracks()] != before

    assert session.undo_action() is True
    assert [track.start_frame for track in session.project.sorted_tracks()] == before


# ----------------------------------------------------------------------
# 5. The whole application has to agree on where the track starts
# ----------------------------------------------------------------------
def test_the_recorded_margin_follows_a_manual_move(tmp_path):
    """A cut moved by hand reports the margin it actually has.

    The panel promises to show how much margin was applied. If a cut is
    nudged ten milliseconds, the margin is ten milliseconds different, and
    the number on screen has to say so.
    """
    material = M.two_songs_with_a_long_gap(gap_seconds=6.0)
    session, _outcome = analyse(tmp_path, material)

    track = session.project.sorted_tracks()[1]
    onset = track.onset_frame
    before = track.start_frame
    assert session.move_boundary(before, before + int(0.1 * RATE)) is True

    moved = session.project.sorted_tracks()[1]
    assert moved.start_frame == before + int(0.1 * RATE)
    assert moved.boundary_pre_roll_seconds == pytest.approx(
        (onset - moved.start_frame) / RATE, abs=1e-6
    ), (
        f"the recorded margin is {moved.boundary_pre_roll_seconds:.3f} s, but "
        f"the cut is {(onset - moved.start_frame) / RATE:.3f} s in front of the "
        f"onset"
    )


def test_a_cut_without_an_onset_has_no_margin_to_report(tmp_path):
    """A cut that was never derived from an onset has no margin."""
    material = M.two_songs_with_a_long_gap(gap_seconds=6.0)
    session, _outcome = analyse(tmp_path, material)

    track = session.project.sorted_tracks()[1]
    track.onset_frame = -1
    session.move_boundary(track.start_frame, track.start_frame - 100)
    moved = session.project.sorted_tracks()[1]
    assert moved.boundary_pre_roll_seconds < 0, (
        "a margin was invented for a cut that has no onset"
    )


def test_a_project_without_the_margin_loads_with_the_model_default(tmp_path):
    """An older project must not bring a stale default back with it.

    The project file used to carry a hard-coded 100 ms margin for projects
    that did not have the field. That is exactly the drift this round
    removed, so the loader has to take the value from the model.
    """
    from splytnyt.core import serialize

    material = M.two_songs_with_a_long_gap(gap_seconds=6.0)
    session, _outcome = analyse(tmp_path, material)

    data = serialize.project_to_dict(session.project)
    data.pop("placement_pre_roll_seconds", None)

    restored = serialize.project_from_dict(data)
    assert restored.placement_pre_roll_seconds == pytest.approx(
        DEFAULT_PRE_ROLL_SECONDS
    ), (
        f"the project came back with a {restored.placement_pre_roll_seconds} s "
        f"margin instead of the model's {DEFAULT_PRE_ROLL_SECONDS} s"
    )


def test_the_exported_file_starts_exactly_at_the_track_start(tmp_path):
    """What is played, what is shown and what is exported are one position."""
    material = M.two_songs_with_a_long_gap(gap_seconds=6.0)
    session, _outcome = analyse(tmp_path, material)

    files = export(session, tmp_path)
    tracks = session.project.sorted_tracks()
    assert len(files) == len(tracks)

    source, _rate = sf.read(
        session.project.sources[0].path, always_2d=True, dtype="int16"
    )
    for track, path in zip(tracks, files):
        data, rate = sf.read(path, always_2d=True, dtype="int16")
        assert rate == RATE
        expected = source[track.start_frame: track.end_frame]
        assert data.shape == expected.shape, (
            f"the exported track has {data.shape[0]} frames, the track covers "
            f"{expected.shape[0]}"
        )
        assert np.array_equal(data, expected), (
            "the exported audio does not start at the track's start frame"
        )


def test_a_refined_boundary_is_what_the_export_follows(tmp_path):
    """Refinement has to move the boundary *and* the exported audio."""
    material = M.two_songs_with_a_long_gap(gap_seconds=6.0)
    session, _outcome = analyse(tmp_path, material)

    original = session.project.sorted_tracks()[1].start_frame

    job = RefineJob(
        session.project, placement=PlacementChoice(mode=PLACEMENT_PREVIOUS_END)
    )
    proposals = job.run()
    assert proposals, "the refinement proposed nothing"
    applied = session.apply_refinement(proposals)
    assert applied == len(proposals)

    refined = session.project.sorted_tracks()[1].start_frame
    assert refined != original, "the refinement did not move the boundary"

    files = export(session, tmp_path)
    source, _rate = sf.read(
        session.project.sources[0].path, always_2d=True, dtype="int16"
    )
    data, _rate = sf.read(files[1], always_2d=True, dtype="int16")
    expected = source[refined: refined + data.shape[0]]
    assert np.array_equal(data, expected), (
        "the export does not follow the refined boundary"
    )

    # And the previous track grew to meet it, without losing a single frame.
    tracks = session.project.sorted_tracks()
    assert tracks[0].end_frame == refined
    assert sum(track.frames for track in tracks) == session.project.total_frames()


def test_a_refined_boundary_is_saved_and_reopened_unchanged(tmp_path):
    """The refined position survives the round trip to disk."""
    material = M.two_songs_with_a_long_gap(gap_seconds=6.0)
    session, _outcome = analyse(tmp_path, material)

    job = RefineJob(
        session.project, placement=PlacementChoice(mode=PLACEMENT_PREVIOUS_END)
    )
    session.apply_refinement(job.run())

    path = str(tmp_path / "project.splytnyt")
    session.save_project(path)

    reopened = Session()
    reopened.open_project(path)
    original = session.project.sorted_tracks()
    copy = reopened.project.sorted_tracks()
    assert [t.start_frame for t in copy] == [t.start_frame for t in original]
    assert [t.onset_frame for t in copy] == [t.onset_frame for t in original]
    assert [t.boundary_pre_roll_seconds for t in copy] == [
        t.boundary_pre_roll_seconds for t in original
    ]


# ----------------------------------------------------------------------
# 6. Accepting a cut
# ----------------------------------------------------------------------
def test_accepting_a_cut_records_a_human_decision(tmp_path):
    """An accepted cut is no longer an automatic assessment."""
    material = M.two_songs_with_a_long_gap(gap_seconds=6.0)
    session, _outcome = analyse(tmp_path, material)

    track = session.project.sorted_tracks()[1]
    assert session.accept_boundary(track.start_frame) is True

    accepted = session.project.sorted_tracks()[1]
    assert accepted.boundary_basis == BOUNDARY_BASIS_ACCEPTED
    assert accepted.boundary_assessment() == "Accepted"
    assert accepted.boundary_uncertain is False


def test_an_accepted_cut_does_not_move_its_own_boundary(tmp_path):
    """Accepting is a decision, not a move."""
    material = M.two_songs_with_a_long_gap(gap_seconds=6.0)
    session, _outcome = analyse(tmp_path, material)

    track = session.project.sorted_tracks()[1]
    before = track.start_frame
    session.accept_boundary(before)

    after = session.project.sorted_tracks()[1]
    assert after.start_frame == before
    assert after.onset_frame == track.onset_frame
