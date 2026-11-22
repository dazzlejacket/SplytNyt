"""Integration tests for the Phase 1 deliverable.

These are the objective checks the product asks for:

* concatenating every exported track reproduces the original decoded PCM
  exactly (lossless, unprocessed),
* the exported tracks cover the timeline without gaps or overlaps,
* channel layout and timing are preserved,
* tags survive the round trip,
* the export refuses to overwrite a source recording,
* a cancelled export leaves nothing that looks finished,
* a project saved and reloaded is identical.

They use synthetic recordings, so they are fast and deterministic.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splytnyt.app.exporting import export_project  # noqa: E402
from splytnyt.app.session import Session  # noqa: E402
from splytnyt.core import serialize  # noqa: E402
from splytnyt.core.model import ExportSettings, Track  # noqa: E402

SAMPLE_RATE = 44100


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def write_tone(path: str, seconds: float, frequency: float = 440.0,
               channels: int = 2, amplitude: float = 0.4, subtype: str = "PCM_16") -> None:
    frames = int(seconds * SAMPLE_RATE)
    time_axis = np.arange(frames) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * frequency * time_axis) * amplitude
    data = np.stack([wave] * channels, axis=1)
    sf.write(path, data, SAMPLE_RATE, subtype=subtype)


def write_recording(path: str, songs: int = 3, song_seconds: float = 2.0,
                    gap_seconds: float = 1.0, channels: int = 2) -> np.ndarray:
    """A recording of ``songs`` tones separated by silence."""
    pieces = []
    for index in range(songs):
        frames = int(song_seconds * SAMPLE_RATE)
        time_axis = np.arange(frames) / SAMPLE_RATE
        wave = np.sin(2 * np.pi * (330 + 110 * index) * time_axis) * 0.4
        pieces.append(np.stack([wave] * channels, axis=1))
        if index < songs - 1:
            gap_frames = int(gap_seconds * SAMPLE_RATE)
            pieces.append(np.zeros((gap_frames, channels), dtype=np.float32))
    data = np.concatenate(pieces, axis=0).astype(np.float32)
    sf.write(path, data, SAMPLE_RATE, subtype="PCM_16")
    return data


@pytest.fixture()
def recording(tmp_path):
    path = str(tmp_path / "recording.flac")
    data = write_recording(path)
    return path, data


@pytest.fixture()
def session_with_recording(recording, tmp_path):
    path, data = recording
    session = Session()
    session.import_files([path])
    return session, data, str(tmp_path / "out")


def make_session(path: str) -> Session:
    session = Session()
    session.import_files([path])
    return session


def split_into_tracks(session: Session, count: int) -> None:
    """Replace the tracks with ``count`` equal segments covering the timeline."""
    total = session.project.total_frames()
    step = total // count
    session.project.tracks.clear()
    for index in range(count):
        start = index * step
        end = total if index == count - 1 else start + step
        session.project.tracks.append(
            Track(start_frame=start, end_frame=end, title=f"Part {index + 1}",
                  boundary_confidence=1.0)
        )


# ----------------------------------------------------------------------
# PCM reconstruction
# ----------------------------------------------------------------------
def test_export_reconstructs_the_original_pcm_exactly(tmp_path, recording):
    """Concatenating the exported tracks must reproduce the source bit-exactly."""
    source_path, source_data = recording
    session = make_session(source_path)
    split_into_tracks(session, 3)

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.mode = "separate"
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    report = export_project(session.project, settings)
    assert len(report.files) == 3

    exported = []
    for path in sorted(report.files, key=lambda item: item):
        data, rate = sf.read(path, always_2d=True, dtype="int16")
        assert rate == SAMPLE_RATE
        exported.append(data)

    concatenated = np.concatenate(exported, axis=0)
    assert concatenated.shape == source_data.shape, (
        f"reconstruction has {concatenated.shape[0]} frames, "
        f"the source has {source_data.shape[0]}"
    )
    # The reference is the source decoded as integers: the export must be
    # bit-identical to it.
    expected, _rate = sf.read(source_path, always_2d=True, dtype="int16")
    assert np.array_equal(concatenated, expected), "the exported PCM differs from the source"


def test_export_preserves_channel_layout_and_timing(tmp_path, recording):
    source_path, _data = recording
    session = make_session(source_path)
    split_into_tracks(session, 4)

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    report = export_project(session.project, settings)
    total = 0
    for path, track in zip(report.files, session.project.sorted_tracks()):
        info = sf.info(path)
        assert info.channels == 2
        assert info.samplerate == SAMPLE_RATE
        assert info.frames == track.frames, "the exported file is not exactly the track length"
        total += info.frames
    assert total == session.project.total_frames(), "the exported tracks do not cover the timeline"


def test_recorder_file_split_mid_song_reconstructs_the_song(tmp_path):
    """A track that spans two source files must concatenate seamlessly."""
    first_path = str(tmp_path / "part1.flac")
    second_path = str(tmp_path / "part2.flac")
    rate = 44100
    tone_a = np.stack(
        [np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate) * 0.4] * 2, axis=1
    ).astype(np.float32)
    tone_b = np.stack(
        [np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate) * 0.4] * 2, axis=1
    ).astype(np.float32)
    sf.write(first_path, tone_a, rate, subtype="PCM_16")
    sf.write(second_path, tone_b, rate, subtype="PCM_16")

    session = Session()
    session.import_files([first_path, second_path])
    session.project.tracks.clear()
    # One track covering the whole of both files.
    session.project.tracks.append(
        Track(start_frame=0, end_frame=session.project.total_frames(),
              title="One song across two files", boundary_confidence=1.0)
    )

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    report = export_project(session.project, settings)
    assert len(report.files) == 1
    exported, _rate = sf.read(report.files[0], always_2d=True, dtype="int16")
    assert exported.shape[0] == rate * 4, "the two files were not joined seamlessly"

    source_a, _ = sf.read(first_path, always_2d=True, dtype="int16")
    source_b, _ = sf.read(second_path, always_2d=True, dtype="int16")
    assert np.array_equal(exported[: source_a.shape[0]], source_a)
    assert np.array_equal(exported[source_a.shape[0]:], source_b)


def test_gap_between_sources_is_preserved_as_silence(tmp_path):
    """A gap is never dropped silently: it is written as silence and reported."""
    first_path = str(tmp_path / "part1.flac")
    second_path = str(tmp_path / "part2.flac")
    rate = 44100
    tone = np.stack(
        [np.sin(2 * np.pi * 330 * np.arange(rate) / rate) * 0.4] * 2, axis=1
    ).astype(np.float32)
    sf.write(first_path, tone, rate, subtype="PCM_16")
    sf.write(second_path, tone, rate, subtype="PCM_16")

    session = Session()
    session.import_files([first_path, second_path])
    # Insert half a second of silence between the two files.
    session.set_source_gap(session.project.sources[1].source_id, rate // 2)
    session.project.tracks.clear()
    session.project.tracks.append(
        Track(start_frame=0, end_frame=session.project.total_frames(),
              title="Across the gap", boundary_confidence=1.0)
    )

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    report = export_project(session.project, settings)
    exported, _rate = sf.read(report.files[0], always_2d=True, dtype="float32")
    assert exported.shape[0] == rate * 2 + rate // 2
    middle = exported[rate: rate + rate // 2]
    assert np.max(np.abs(middle)) == 0.0, "the gap was not written as silence"
    assert any("gap" in note.lower() for note in report.notes), \
        "the gap was not reported to the user"


def test_per_track_gain_is_applied_to_each_track(tmp_path):
    """Per-track normalisation gives each track its own gain."""
    quiet_path = str(tmp_path / "quiet.flac")
    loud_path = str(tmp_path / "loud.flac")
    rate = 44100
    sf.write(quiet_path,
             np.stack([np.sin(2 * np.pi * 440 * np.arange(rate) / rate) * 0.02] * 2, axis=1),
             rate, subtype="PCM_16")
    sf.write(loud_path,
             np.stack([np.sin(2 * np.pi * 440 * np.arange(rate) / rate) * 0.4] * 2, axis=1),
             rate, subtype="PCM_16")

    session = Session()
    session.import_files([quiet_path, loud_path])
    session.project.tracks.clear()
    total = session.project.total_frames()
    session.project.tracks.append(Track(start_frame=0, end_frame=total // 2,
                                        title="Quiet", boundary_confidence=1.0))
    session.project.tracks.append(Track(start_frame=total // 2, end_frame=total,
                                        title="Loud", boundary_confidence=1.0))

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.gain_mode = "per_track"
    settings.loudness_target_lufs = -20.0
    settings.true_peak_ceiling_dbtp = -1.0
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    report = export_project(session.project, settings)
    assert len(report.files) == 2

    # Per-track normalisation equalises the two tracks to the target, which
    # is the opposite of what they looked like in the source (a 26 dB
    # difference).
    from splytnyt.audio.decode import AudioSource
    from splytnyt.audio.measure import measure_range

    loudness = []
    for path in report.files:
        with AudioSource(path) as source:
            measured = measure_range(source, 0, source.frames)
        loudness.append(measured.integrated_lufs)

    assert abs(loudness[0] - loudness[1]) < 1.0, (
        f"the tracks were not equalised: {loudness}"
    )
    for value in loudness:
        assert abs(value - settings.loudness_target_lufs) < 1.5, (
            f"a track landed at {value:.1f} LUFS, not at the "
            f"{settings.loudness_target_lufs:.1f} LUFS target"
        )
    assert "per-track gain" in report.gain_description


def _quiet_recording_with_a_peak(path: str, rate: int = 44100) -> str:
    """A quiet recording with one loud transient.

    The loudness is far below the peak, so reaching a loudness target would
    push the transient past the ceiling. This is the situation the gain
    planner has to report honestly.
    """
    frames = rate * 3
    time_axis = np.arange(frames) / rate
    wave = np.sin(2 * np.pi * 440 * time_axis) * 0.05
    wave[rate: rate + 200] = 0.99                     # a loud transient
    data = np.stack([wave, wave], axis=1).astype(np.float32)
    sf.write(path, data, rate, subtype="PCM_16")
    return path


def test_shared_gain_that_would_clip_uses_less_gain_without_a_limiter(tmp_path):
    """Without an explicit limiter, less gain is used rather than clipping."""
    path = _quiet_recording_with_a_peak(str(tmp_path / "loud.flac"))
    session = make_session(path)
    split_into_tracks(session, 1)

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.gain_mode = "loudness"
    settings.loudness_target_lufs = -14.0
    settings.true_peak_ceiling_dbtp = -1.0
    settings.allow_limiter = False
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    report = export_project(session.project, settings)
    assert report.gain_feasible is False
    assert any("ceiling" in warning for warning in report.warnings)

    data, _rate = sf.read(report.files[0], always_2d=True, dtype="float32")
    assert float(np.max(np.abs(data))) <= 1.0, "the output was clipped"


def test_the_explicit_limiter_keeps_the_output_under_the_ceiling(tmp_path):
    """With the limiter enabled, the peaks are brought under the ceiling."""
    path = _quiet_recording_with_a_peak(str(tmp_path / "loud.flac"))
    session = make_session(path)
    split_into_tracks(session, 1)

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.gain_mode = "loudness"
    settings.loudness_target_lufs = -14.0
    settings.true_peak_ceiling_dbtp = -1.0
    settings.allow_limiter = True
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    report = export_project(session.project, settings)
    assert any("limiter" in note.lower() for note in report.notes)

    data, _rate = sf.read(report.files[0], always_2d=True, dtype="float32")
    peak = float(np.max(np.abs(data)))
    ceiling = 10.0 ** (settings.true_peak_ceiling_dbtp / 20.0)
    assert peak <= ceiling * 1.05, f"the peak {peak} is above the ceiling {ceiling}"


def test_mono_source_stays_mono(tmp_path):
    path = str(tmp_path / "mono.flac")
    write_tone(path, 1.0, channels=1)
    session = make_session(path)
    split_into_tracks(session, 2)

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "wav"
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    report = export_project(session.project, settings)
    for exported in report.files:
        assert sf.info(exported).channels == 1


# ----------------------------------------------------------------------
# Source protection
# ----------------------------------------------------------------------
def test_export_refuses_to_overwrite_a_source(tmp_path, recording):
    source_path, _data = recording
    session = make_session(source_path)
    split_into_tracks(session, 2)

    settings = ExportSettings()
    settings.output_dir = os.path.dirname(source_path)
    # The template resolves to exactly the source file name.
    settings.filename_template = "{title}"
    settings.container = "flac"
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    stem = os.path.splitext(os.path.basename(source_path))[0]
    session.project.tracks[0].title = stem
    session.project.tracks[1].title = "other"

    with pytest.raises(PermissionError):
        export_project(session.project, settings)

    assert os.path.exists(source_path), "the source recording was damaged"


def test_export_never_touches_the_source_file(tmp_path, recording):
    source_path, _data = recording
    before = os.stat(source_path)
    session = make_session(source_path)
    split_into_tracks(session, 2)

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    export_project(session.project, settings)
    after = os.stat(source_path)
    assert before.st_size == after.st_size
    assert before.st_mtime_ns == after.st_mtime_ns


# ----------------------------------------------------------------------
# Cancellation
# ----------------------------------------------------------------------
def test_cancelled_export_leaves_no_finished_looking_file(tmp_path, recording):
    source_path, _data = recording
    session = make_session(source_path)
    split_into_tracks(session, 4)

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.write_tags = False
    settings.include_info = False
    settings.include_checksums = False

    state = {"calls": 0}

    def cancel() -> bool:
        state["calls"] += 1
        return state["calls"] > 2

    report = export_project(session.project, settings, cancel=cancel)
    assert report.cancelled is True
    assert len(report.files) < 4
    leftovers = [name for name in os.listdir(settings.output_dir) if name.endswith(".partial")]
    assert not leftovers, f"partial files were left behind: {leftovers}"


# ----------------------------------------------------------------------
# Tags
# ----------------------------------------------------------------------
def test_tags_are_written_and_read_back(tmp_path, recording):
    source_path, _data = recording
    session = make_session(source_path)
    split_into_tracks(session, 2)
    session.project.metadata.artist = "The Example Ensemble"
    session.project.metadata.album = "A Recording"
    session.project.tracks[0].title = "First Piece"

    settings = ExportSettings()
    settings.output_dir = str(tmp_path / "export")
    settings.container = "flac"
    settings.write_tags = True
    settings.include_info = False
    settings.include_checksums = False

    report = export_project(session.project, settings)
    assert report.files

    # Read the Vorbis comments back by parsing the FLAC metadata chain.
    data = open(report.files[0], "rb").read()
    assert data[:4] == b"fLaC"
    offset = 4
    comments = {}
    while offset < len(data):
        header = data[offset]
        last = bool(header & 0x80)
        block_type = header & 0x7F
        length = int.from_bytes(data[offset + 1: offset + 4], "big")
        body = data[offset + 4: offset + 4 + length]
        if block_type == 4:
            vendor_length = int.from_bytes(body[:4], "little")
            position = 4 + vendor_length
            count = int.from_bytes(body[position:position + 4], "little")
            position += 4
            for _ in range(count):
                entry_length = int.from_bytes(body[position:position + 4], "little")
                position += 4
                entry = body[position:position + entry_length].decode("utf-8")
                position += entry_length
                key, _, value = entry.partition("=")
                comments[key.upper()] = value
        offset += 4 + length
        if last:
            break

    assert comments.get("ARTIST") == "The Example Ensemble"
    assert comments.get("ALBUM") == "A Recording"
    assert comments.get("TITLE") == "First Piece"


# ----------------------------------------------------------------------
# Project persistence
# ----------------------------------------------------------------------
def test_project_round_trip_is_identical(tmp_path, recording):
    source_path, _data = recording
    session = make_session(source_path)
    split_into_tracks(session, 3)
    session.project.tracks[1].title = "Middle"
    session.project.metadata.venue = "The Old Hall"

    path = str(tmp_path / "project.splytnyt")
    session.save_project(path)
    reloaded = Session()
    reloaded.open_project(path)

    assert len(reloaded.project.tracks) == len(session.project.tracks)
    for original, copy in zip(session.project.sorted_tracks(), reloaded.project.sorted_tracks()):
        assert original.track_id == copy.track_id
        assert original.start_frame == copy.start_frame
        assert original.end_frame == copy.end_frame
        assert original.title == copy.title
    assert reloaded.project.metadata.venue == "The Old Hall"


def test_undo_and_redo_restore_the_previous_state(tmp_path, recording):
    source_path, _data = recording
    session = make_session(source_path)
    split_into_tracks(session, 2)
    before = len(session.project.tracks)

    session.split_at(session.project.tracks[0].start_frame + 1000)
    assert len(session.project.tracks) == before + 1

    assert session.undo_action() is True
    assert len(session.project.tracks) == before
    assert session.redo_action() is True
    assert len(session.project.tracks) == before + 1


def test_removing_a_track_keeps_the_audio(tmp_path, recording):
    source_path, _data = recording
    session = make_session(source_path)
    split_into_tracks(session, 3)
    total_before = session.project.total_frames()

    session.remove_track(session.project.tracks[1].track_id)

    covered = sum(track.frames for track in session.project.tracks)
    assert covered == total_before, "removing a track lost audio"


# ----------------------------------------------------------------------
# Markers
# ----------------------------------------------------------------------
def test_marker_exports_round_trip(tmp_path, recording):
    source_path, _data = recording
    session = make_session(source_path)
    split_into_tracks(session, 3)

    payload = serialize.markers_to_json(session.project)
    segments = serialize.markers_from_json(payload)
    assert len(segments) == 3

    csv_text = serialize.markers_to_csv(session.project)
    assert len(serialize.markers_from_csv(csv_text, SAMPLE_RATE)) == 3

    audacity = serialize.markers_to_audacity(session.project)
    assert len(serialize.markers_from_audacity(audacity, SAMPLE_RATE)) == 3


def test_tracklist_parsing_handles_common_shapes():
    items = serialize.parse_tracklist(
        "1. First Song\n2) Second Song (3:45)\nThird Song — with a note\n"
    )
    assert [item.title for item in items] == ["First Song", "Second Song", "Third Song"]
    assert items[1].duration_hint == pytest.approx(225.0)
    assert items[2].notes == "with a note"


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def test_analysis_finds_silence_boundaries(tmp_path):
    from splytnyt.analysis.segment import suggest_boundaries

    path = str(tmp_path / "songs.flac")
    write_recording(path, songs=3, song_seconds=2.0, gap_seconds=1.0)
    data, _rate = sf.read(path, always_2d=True, dtype="float32")
    mono = data.mean(axis=1)

    suggestions = suggest_boundaries(mono, SAMPLE_RATE, "balanced")
    assert len(suggestions) >= 2
    assert all(suggestion.kind == "silence" for suggestion in suggestions)
    assert all(suggestion.confidence < 0.95 for suggestion in suggestions), \
        "silence evidence must never claim certainty"

    # The suggestions must land inside the silent gaps.
    gap_starts = [int(3.0 * SAMPLE_RATE), int(6.0 * SAMPLE_RATE)]
    for suggestion in suggestions:
        assert any(abs(suggestion.frame - start) < 0.6 * SAMPLE_RATE for start in gap_starts)


def test_continuous_music_change_is_uncertain():
    from splytnyt.analysis.segment import suggest_boundaries

    rate = SAMPLE_RATE
    first = np.sin(2 * np.pi * 220 * np.arange(rate * 2) / rate) * 0.4
    second = np.sin(2 * np.pi * 660 * np.arange(rate * 2) / rate) * 0.4
    mono = np.concatenate([first, second]).astype(np.float32)

    suggestions = suggest_boundaries(mono, rate, "balanced")
    assert suggestions, "a clear timbral change should be reported"
    assert all(suggestion.uncertain for suggestion in suggestions), \
        "a musical change without silence must be marked uncertain"
    assert all(suggestion.confidence <= 0.6 for suggestion in suggestions)


def test_noisy_tape_gap_is_still_detected(tmp_path):
    """A noisy tape: the gap sits at the noise floor, not at digital silence."""
    from splytnyt.analysis.segment import suggest_boundaries

    rate = SAMPLE_RATE
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.006, size=int(6 * rate)).astype(np.float32)  # ~-44 dBFS
    song = (np.sin(2 * np.pi * 440 * np.arange(int(2 * rate)) / rate) * 0.25).astype(np.float32)
    quiet = noise[: int(2 * rate)]
    mono = np.concatenate([song + noise[: int(2 * rate)], quiet, song + noise[-int(2 * rate):]])

    suggestions = suggest_boundaries(mono, rate, "balanced")
    silences = [s for s in suggestions if s.kind == "silence"]
    assert silences, "the gap in a noisy recording was not detected"
    # The boundary must land inside the quiet passage.
    assert 2.0 * rate < silences[0].frame < 4.0 * rate


def test_applause_inside_a_song_is_not_cut(tmp_path):
    """Applause during a song must not become a track boundary."""
    from splytnyt.analysis.segment import suggest_boundaries

    rate = SAMPLE_RATE
    rng = np.random.default_rng(11)
    music = (np.sin(2 * np.pi * 330 * np.arange(int(8 * rate)) / rate) * 0.3).astype(np.float32)
    burst = rng.normal(0.0, 0.12, size=int(1.0 * rate)).astype(np.float32)
    music[3 * rate: 4 * rate] = burst           # applause in the middle of the song
    mono = music

    suggestions = suggest_boundaries(mono, rate, "balanced")
    inside = [
        s for s in suggestions
        if 2.5 * rate < s.frame < 4.5 * rate
    ]
    assert not inside, (
        "a burst of applause inside a song was proposed as a boundary: "
        f"{[(s.frame / rate, s.kind) for s in inside]}"
    )


def test_worklist_alignment_reports_unmatched_entries():
    from splytnyt.analysis.align import align_worklist
    from splytnyt.core.model import WorklistItem

    items = [
        WorklistItem(title="One", duration_hint=100.0),
        WorklistItem(title="Two", duration_hint=100.0),
        WorklistItem(title="Three", duration_hint=100.0),
        WorklistItem(title="Four", duration_hint=100.0),
    ]
    segments = [(0, rate_ := 100 * SAMPLE_RATE), (rate_, 2 * rate_), (2 * rate_, 3 * rate_)]
    result = align_worklist(items, segments, SAMPLE_RATE)

    assert len(result.assignments) == 3
    assert len(result.unmatched_items) == 1
    assert len(result.unmatched_segments) == 0
