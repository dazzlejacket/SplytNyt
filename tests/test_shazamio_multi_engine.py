"""Tests for optional ShazamIO (Experimental), multi-engine recognition, and UI."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from splytnyt.analysis.metadata_split import (
    CoarseFeatures,
    PlannedTrack,
    MetadataSplitOutcome,
    split_dj_mix_with_metadata,
)
from splytnyt.app.session import Session
from splytnyt.core.model import (
    BOUNDARY_BASIS_ASSESSED,
    BOUNDARY_BASIS_MANUAL,
    FLAG_RECOGNITION_CONFLICT,
    Project,
    Track,
)
from splytnyt.core.serialize import project_from_dict, project_to_dict
from splytnyt.metadata.acoustid import ACOUSTID
from splytnyt.metadata.identify import KIND_DJ_MIX, IdentificationResult
from splytnyt.metadata.multi_engine import (
    STATUS_CONFLICT,
    STATUS_CONFIRMED,
    STATUS_LIKELY,
    STATUS_UNCERTAIN,
    STATUS_UNIDENTIFIED,
    ClipRecognition,
    aggregate_clip_recognitions,
    choose_sample_timestamps,
    is_cast_recording,
    same_song,
)
from splytnyt.metadata import shazamio_engine as se
from splytnyt.metadata.shazamio_engine import SHAZAMIO, ShazamMatch, _parse_shazam_response


# ----------------------------------------------------------------------
# 1. Opt-in settings & graceful unavailability
# ----------------------------------------------------------------------

def test_shazamio_disabled_by_default_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLYTNYT_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("SPLYTNYT_OPTIONAL_ENGINES_DIR", str(tmp_path / "engine"))

    assert se.is_enabled() is False
    assert se.is_ready() is False

    se.set_enabled(True)
    assert se.is_enabled() is True
    # Still not ready if package is not installed in isolated dir (unless in global env)
    se.set_enabled(False)
    assert se.is_enabled() is False


def test_shazam_response_parser_and_offset():
    payload = {
        "matches": [
            {"id": "123", "offset": 48.5, "timeskew": 0.025, "frequencyskew": 0.024}
        ],
        "track": {
            "key": "20147984",
            "title": "Breathe (In the Air)",
            "subtitle": "Pink Floyd",
            "isrc": "GBN9Y1100077",
            "sections": [
                {
                    "metadata": [
                        {"title": "Album", "text": "The Dark Side of the Moon"},
                        {"title": "Label", "text": "Columbia/Legacy"},
                        {"title": "Released", "text": "1973"},
                    ]
                }
            ],
        },
    }
    match = _parse_shazam_response(payload)
    assert match is not None
    assert match.title == "Breathe (In the Air)"
    assert match.artist == "Pink Floyd"
    assert match.album == "The Dark Side of the Moon"
    assert match.isrc == "GBN9Y1100077"
    assert abs(match.offset_seconds - 48.5) < 1e-6
    assert abs(match.timeskew - 0.025) < 1e-6

    assert _parse_shazam_response({"matches": []}) is None
    assert _parse_shazam_response({}) is None


# ----------------------------------------------------------------------
# 2. Multi-engine consensus, conflicts, song changes, Broadway guard
# ----------------------------------------------------------------------

def test_same_song_normalization():
    assert same_song("Willy Crook", "Them (feat. Funky Torinos)", "Willy Crook", "Them")
    assert same_song("Pink Floyd", "Breathe (In the Air) - 2011 Remastered", "Pink Floyd", "Breathe (In the Air)")
    assert same_song("Soda Stereo", "Música Ligera", "Soda Stereo", "Musica Ligera")
    assert not same_song("Willy Crook", "Them", "NAFTA", "Andate")


def test_aggregate_confirmed_likely_conflict_unidentified():
    # Unidentified
    r_none = aggregate_clip_recognitions(0.0, 200.0, [])
    assert r_none.status == STATUS_UNIDENTIFIED
    assert r_none.title == ""
    assert r_none.artist == ""

    # Single strong hit -> Likely
    c1 = ClipRecognition(SHAZAMIO, 40.0, 50.0, "Andate", "NAFTA", score=0.85)
    r_likely = aggregate_clip_recognitions(0.0, 200.0, [c1])
    assert r_likely.status == STATUS_LIKELY
    assert r_likely.artist == "NAFTA"
    assert r_likely.title == "Andate"

    # Two consistent clips from same engine -> Confirmed
    c2 = ClipRecognition(SHAZAMIO, 120.0, 130.0, "Andate", "NAFTA", score=0.85)
    r_conf_clips = aggregate_clip_recognitions(0.0, 200.0, [c1, c2])
    assert r_conf_clips.status == STATUS_CONFIRMED
    assert r_conf_clips.confidence >= 0.84

    # Both AcoustID and ShazamIO agree -> Confirmed
    c_ac = ClipRecognition(ACOUSTID, 0.0, 120.0, "Andate", "NAFTA", score=0.90)
    r_conf_both = aggregate_clip_recognitions(0.0, 200.0, [c_ac, c1])
    assert r_conf_both.status == STATUS_CONFIRMED
    assert set(r_conf_both.engines) == {ACOUSTID, SHAZAMIO}

    # AcoustID and ShazamIO disagree -> Conflict!
    c_other = ClipRecognition(ACOUSTID, 0.0, 120.0, "Different Song", "Other Artist", score=0.88)
    r_conflict = aggregate_clip_recognitions(0.0, 200.0, [c_other, c1])
    assert r_conflict.status == STATUS_CONFLICT
    assert r_conflict.has_conflict
    assert "AcoustID" in r_conflict.conflict_detail and "ShazamIO" in r_conflict.conflict_detail


def test_aggregate_detects_song_change_with_shazam_offset():
    c_a = ClipRecognition(SHAZAMIO, 50.0, 60.0, "Them", "Willy Crook", score=0.85, offset_seconds=47.0)
    c_b = ClipRecognition(SHAZAMIO, 300.0, 310.0, "Andate", "NAFTA", score=0.85, offset_seconds=39.0)
    res = aggregate_clip_recognitions(0.0, 400.0, [c_a, c_b])
    assert len(res.song_changes) == 1
    ch = res.song_changes[0]
    assert ch.prev_title == "Them"
    assert ch.next_title == "Andate"
    # Estimated boundary = 300 - 39 = 261s
    assert ch.estimated_boundary_seconds is not None
    assert abs(ch.estimated_boundary_seconds - 261.0) < 1.0


def test_choose_sample_timestamps_avoids_silence_and_applause():
    # 120s segment where [20..50s] is silent (-80 dB) and [60..105s] is loud music (-12 dB)
    rms = np.full(240, -12.0, dtype=np.float32)
    rms[40:100] = -80.0  # 20s..50s silent
    feats = CoarseFeatures(
        hop_seconds=0.5,
        sample_rate=44100,
        total_frames=120 * 44100,
        rms_db=rms,
        bands=np.zeros((240, 16), dtype=np.float32),
        chroma=np.zeros((240, 12), dtype=np.float32),
    )
    times = choose_sample_timestamps(0.0, 120.0, clip_seconds=10.0, n_clips=2, coarse_features=feats)
    assert len(times) == 2
    # Neither chosen clip should land squarely inside the silent middle [25..40s]
    for t in times:
        idx0 = feats.index_of(t)
        idx1 = feats.index_of(t + 10.0)
        assert float(np.mean(rms[idx0:idx1])) > -55.0


def test_broadway_jukebox_guard():
    assert is_cast_recording("Original Broadway Cast of Moulin Rouge! The Musical", "Moulin Rouge! The Musical", "Moulin Rouge")
    assert not is_cast_recording("Lady Gaga", "The Fame Monster", "Moulin Rouge")
    assert not is_cast_recording("Elton John", "Elton John", "Moulin Rouge")


# ----------------------------------------------------------------------
# 3. Serialization & Track identity assessment honesty
# ----------------------------------------------------------------------

def test_track_recognition_fields_roundtrip_and_honest_labels():
    t_aligned = Track(
        start_frame=0,
        end_frame=44100 * 180,
        title="Time",
        identity_confidence=0.82,
        identity_basis=BOUNDARY_BASIS_ASSESSED,
    )
    # MusicBrainz / setlist alignment alone must NEVER be called Confirmed!
    assert t_aligned.recognition_status == ""
    assert t_aligned.identity_assessment() != "Confirmed"

    t_confirmed = Track(
        start_frame=44100 * 180,
        end_frame=44100 * 360,
        title="Andate",
        recognized_artist="NAFTA",
        recognized_title="Andate",
        recognition_status=STATUS_CONFIRMED,
        recognition_engines=[ACOUSTID, SHAZAMIO],
        identity_confidence=0.92,
        identity_basis=BOUNDARY_BASIS_ASSESSED,
    )
    assert t_confirmed.identity_assessment() == "Confirmed"

    t_conflict = Track(
        start_frame=44100 * 360,
        end_frame=44100 * 540,
        title="Song X",
        recognition_status=STATUS_CONFLICT,
        flags=[FLAG_RECOGNITION_CONFLICT],
        identity_basis=BOUNDARY_BASIS_ASSESSED,
    )
    assert t_conflict.identity_assessment() == "Conflict"

    proj = Project(tracks=[t_aligned, t_confirmed, t_conflict])
    reloaded = project_from_dict(project_to_dict(proj))
    assert reloaded.tracks[1].recognized_artist == "NAFTA"
    assert reloaded.tracks[1].recognized_title == "Andate"
    assert reloaded.tracks[1].recognition_status == STATUS_CONFIRMED
    assert reloaded.tracks[1].recognition_engines == [ACOUSTID, SHAZAMIO]
    assert FLAG_RECOGNITION_CONFLICT in reloaded.tracks[2].flags


# ----------------------------------------------------------------------
# 4. Qt UI: TrackTableModel identity text + MusicServicesDialog ShazamIO section
# ----------------------------------------------------------------------

def test_ui_track_table_and_music_services_dialog(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from splytnyt.ui.settings_dialog import MusicServicesDialog
    from splytnyt.ui.track_table import COLUMN_IDENTITY, COLUMN_TITLE, TrackTableModel

    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SPLYTNYT_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("SPLYTNYT_OPTIONAL_ENGINES_DIR", str(tmp_path / "engine"))

    model = TrackTableModel()
    t_mb = Track(title="Money", identity_basis=BOUNDARY_BASIS_ASSESSED, identity_confidence=0.85)
    t_conf = Track(
        title="Andate",
        recognized_artist="NAFTA",
        recognized_title="Andate",
        recognition_status=STATUS_CONFIRMED,
        identity_basis=BOUNDARY_BASIS_ASSESSED,
    )
    t_like = Track(title="Them", recognized_artist="Willy Crook", recognition_status=STATUS_LIKELY, identity_basis=BOUNDARY_BASIS_ASSESSED)
    t_conf_err = Track(title="Conflict Track", recognition_status=STATUS_CONFLICT, identity_basis=BOUNDARY_BASIS_ASSESSED)
    t_unid = Track(title="Track 5", recognition_status=STATUS_UNIDENTIFIED)
    t_typed = Track(title="My Manual Title", identity_basis=BOUNDARY_BASIS_MANUAL)
    model.set_tracks([t_mb, t_conf, t_like, t_conf_err, t_unid, t_typed], 44100)

    assert model.data(model.index(0, COLUMN_IDENTITY)) == "Aligned"  # NOT Confirmed!
    assert model.data(model.index(1, COLUMN_IDENTITY)) == "Confirmed"
    assert "NAFTA" in model.data(model.index(1, COLUMN_TITLE))
    assert model.data(model.index(2, COLUMN_IDENTITY)) == "Likely"
    assert model.data(model.index(3, COLUMN_IDENTITY)) == "Conflict"
    assert model.data(model.index(4, COLUMN_IDENTITY)) == "Unidentified"
    assert model.data(model.index(5, COLUMN_IDENTITY)) == "Typed"

    dlg = MusicServicesDialog()
    assert hasattr(dlg, "shazamio_enable_check")
    assert hasattr(dlg, "shazamio_install_btn")
    assert dlg.shazamio_enable_check.isChecked() is False
    dlg.close()
    app.processEvents()


# ----------------------------------------------------------------------
# 5. Regression tests for final verification bug fixes
# ----------------------------------------------------------------------

def test_shazamio_uninstall_handles_pending_cleanup(tmp_path, monkeypatch):
    """Uninstall removes installed.json, disables ShazamIO, and handles locked DLL marker."""
    from splytnyt.metadata import shazamio_engine

    opt_dir = tmp_path / "opt_shazam"
    site_dir = opt_dir / "site-packages"
    site_dir.mkdir(parents=True)
    (opt_dir / "installed.json").write_text('{"packages":{"shazamio":"0.9.0"}}', encoding="utf-8")
    monkeypatch.setenv("SPLYTNYT_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("SPLYTNYT_OPTIONAL_ENGINES_DIR", str(opt_dir))

    shazamio_engine.set_enabled(True)
    assert shazamio_engine.is_enabled() is True

    ok, msg = shazamio_engine.uninstall_shazamio()
    assert ok is True
    assert shazamio_engine.is_enabled() is False
    assert not (opt_dir / "installed.json").exists()

    # Simulate Windows locked-DLL pending cleanup marker
    opt_dir.mkdir(parents=True, exist_ok=True)
    (opt_dir / ".pending_cleanup").write_text("pending\n", encoding="utf-8")
    assert shazamio_engine.is_installed() is False
    shazamio_engine.cleanup_pending_uninstall()
    assert not opt_dir.exists()


def test_theater_merges_adjacent_dialogue_sections():
    """Continuous dialogue must never be falsely split into back-to-back dialogue tracks."""
    from splytnyt.analysis.theater import TheaterSection, assemble_tracks

    sr = 48000
    s1 = TheaterSection(
        start_frame=0,
        end_frame=60 * sr,
        section_type="dialogue",
        confidence=0.6,
        evidence={"mean_music_score": 0.05},
    )
    s2 = TheaterSection(
        start_frame=60 * sr,
        end_frame=130 * sr,
        section_type="dialogue",
        confidence=0.6,
        evidence={"mean_music_score": 0.05},
    )
    merged = assemble_tracks([s1, s2], sr, separate_applause=False)
    assert len(merged) == 1
    assert merged[0].section_type == "dialogue"
    assert merged[0].start_frame == 0 and merged[0].end_frame == 130 * sr


def test_multi_artist_export_uses_effective_track_artist(tmp_path):
    """Multi-artist projects (e.g. DJ mixes) include per-track artist in filenames and CUE."""
    import soundfile as sf
    from splytnyt.app.exporting import export_project
    from splytnyt.core.model import ExportSettings, Metadata, Source

    sr = 44100
    audio_path = str(tmp_path / "mix.wav")
    samples = np.zeros((sr * 4, 2), dtype=np.float32)
    sf.write(audio_path, samples, sr, subtype="PCM_16")

    proj = Project(
        name="Vinyl Mix",
        sources=[Source(path=audio_path, original_format="WAV", sample_rate=sr, channels=2, frames=sr * 4)],
        tracks=[
            Track(start_frame=0, end_frame=sr * 2, title="Song One", metadata=Metadata(artist="Artist A")),
            Track(start_frame=sr * 2, end_frame=sr * 4, title="Song Two", metadata=Metadata(artist="Artist B")),
        ],
    )
    out_dir = str(tmp_path / "out")
    settings = ExportSettings(
        output_dir=out_dir,
        container="wav",
        mode="separate",
        include_cue=True,
        include_info=False,
        include_checksums=False,
    )
    report = export_project(proj, settings)
    names = sorted(os.path.basename(p) for p in report.files)
    assert any("Artist A" in n and "Song One" in n for n in names), names
    assert any("Artist B" in n and "Song Two" in n for n in names), names
    cue_text = open(report.cue_path, "r", encoding="utf-8").read()
    assert 'PERFORMER "Artist A"' in cue_text
    assert 'PERFORMER "Artist B"' in cue_text
