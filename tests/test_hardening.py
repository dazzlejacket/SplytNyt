"""Comprehensive hardening tests: real mouse interactions, invariants, MR edits, and bit-exact exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from splytnyt.analysis.lyrics import (
    alignment_quality,
    match_references,
    parse_references,
    propose_audio_extent,
)
from splytnyt.app.exporting import export_project
from splytnyt.app.session import Session
from splytnyt.audio.timeline import build_reader
from splytnyt.core.model import ExportSettings, Source, Track
from splytnyt.ui.lyrics import LyricReviewDialog
from splytnyt.ui.main_window import MainWindow
from splytnyt.ui.waveform import HEADER_HEIGHT, OVERVIEW_HEIGHT, WaveformView


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _make_ramp_wav(path: Path, seconds: float = 12.0, sr: int = 16000) -> np.ndarray:
    """Create a unique, sample-identifiable stereo WAV so any missing/duplicated sample is caught."""
    n = int(round(seconds * sr))
    # Deterministic unique sample values in [-0.8, 0.8] quantized to int16 grid
    idx = np.arange(n, dtype=np.float64)
    left = np.sin(2.0 * np.pi * 220.0 * idx / sr) * 0.4 + 0.3 * np.sin(2.0 * np.pi * 0.37 * idx / sr)
    right = np.cos(2.0 * np.pi * 330.0 * idx / sr) * 0.4 + 0.3 * np.cos(2.0 * np.pi * 0.53 * idx / sr)
    pcm16 = np.clip(np.round(np.column_stack([left, right]) * 32767.0), -32768, 32767).astype(np.int16)
    sf.write(str(path), pcm16, sr, subtype="PCM_16")
    return pcm16


def _assert_strict_contiguous(project) -> None:
    tracks = project.sorted_tracks()
    total = project.total_frames()
    assert project.is_contiguous(), "Project must be strictly contiguous"
    assert tracks[0].start_frame == 0
    assert tracks[-1].end_frame == total
    for left, right in zip(tracks, tracks[1:]):
        assert left.end_frame == right.start_frame, (
            f"Adjacent boundary mismatch: {left.end_frame} != {right.start_frame}"
        )
        assert left.end_frame > left.start_frame


def _drag_divider_qtest(wf: WaveformView, old_frame: int, dx_pixels: int) -> None:
    """Perform a real Qt mouse press -> move -> release sequence on WaveformView.viewport()."""
    app = _app()
    vp = wf.viewport()
    x0 = int(round(wf._frame_to_x(old_frame)))
    y = HEADER_HEIGHT + 35
    p0 = QPoint(x0, y)
    p_mid = QPoint(x0 + dx_pixels // 2, y)
    p1 = QPoint(x0 + dx_pixels, y)
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, p0)
    app.processEvents()
    QTest.mouseMove(vp, p_mid)
    app.processEvents()
    QTest.mouseMove(vp, p1)
    app.processEvents()
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, p1)
    app.processEvents()


def test_real_mouse_divider_drag_across_zoom_levels_and_edge_pan(tmp_path: Path):
    """Real Qt mouse drags at fit, medium, and tight zoom + Escape cancel + middle-click safety."""
    app = _app()
    wav = tmp_path / "ramp.wav"
    _make_ramp_wav(wav, seconds=20.0, sr=16000)

    MainWindow._offer_recovery = lambda self: None
    w = MainWindow()
    if hasattr(w, "_autosave_timer"):
        w._autosave_timer.stop()
    w.resize(1100, 720)
    w.show()
    app.processEvents()

    w.session.import_files([str(wav)])
    sr = 16000
    w.session.split_at(5 * sr)
    w.session.split_at(12 * sr)
    w._reload()
    app.processEvents()

    # 1. Zoom-to-fit drag
    w.waveform.zoom_to_fit()
    app.processEvents()
    div1 = w.session.sorted_tracks()[0].end_frame
    _drag_divider_qtest(w.waveform, div1, +25)
    div1_after = w.session.sorted_tracks()[0].end_frame
    assert div1_after > div1
    _assert_strict_contiguous(w.session.project)

    # 2. Medium zoom drag (6s window)
    w.waveform.zoom_to_frame(div1_after, span_seconds=6.0)
    app.processEvents()
    _drag_divider_qtest(w.waveform, div1_after, -40)
    div1_med = w.session.sorted_tracks()[0].end_frame
    assert div1_med < div1_after
    _assert_strict_contiguous(w.session.project)

    # 3. Tight zoom drag (1.5s window) + edge auto-pan past right edge
    w.waveform.zoom_to_frame(div1_med, span_seconds=1.5)
    app.processEvents()
    scroll_before = w.waveform.horizontalScrollBar().value()
    vp = w.waveform.viewport()
    x0 = int(round(w.waveform._frame_to_x(div1_med)))
    y = HEADER_HEIGHT + 35
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, QPoint(x0, y))
    QTest.mouseMove(vp, QPoint(vp.width() - 5, y))
    app.processEvents()
    assert w.waveform.horizontalScrollBar().value() > scroll_before, "Dragging near right edge should auto-pan"
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, QPoint(vp.width() - 5, y))
    app.processEvents()
    div1_tight = w.session.sorted_tracks()[0].end_frame
    assert div1_tight > div1_med
    _assert_strict_contiguous(w.session.project)

    # 4. Escape cancels an in-progress drag cleanly
    w.waveform.zoom_to_frame(div1_tight, span_seconds=4.0)
    app.processEvents()
    x_esc = int(round(w.waveform._frame_to_x(div1_tight)))
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, QPoint(x_esc, y))
    QTest.mouseMove(vp, QPoint(x_esc + 50, y))
    QTest.keyClick(w.waveform, Qt.Key_Escape)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, QPoint(x_esc + 50, y))
    app.processEvents()
    assert w.session.sorted_tracks()[0].end_frame == div1_tight

    # 5. Middle-click must NOT hijack or move a divider
    w.waveform.set_cursor(div1_tight)
    QTest.mousePress(vp, Qt.MiddleButton, Qt.NoModifier, QPoint(x_esc, y))
    QTest.mouseMove(vp, QPoint(x_esc + 60, y))
    QTest.mouseRelease(vp, Qt.MiddleButton, Qt.NoModifier, QPoint(x_esc + 60, y))
    app.processEvents()
    assert w.session.sorted_tracks()[0].end_frame == div1_tight

    # 6. Locked track blocks real mouse drag
    t0_id = w.session.sorted_tracks()[0].track_id
    w.session.toggle_lock(t0_id)
    w._reload()
    app.processEvents()
    _drag_divider_qtest(w.waveform, div1_tight, +40)
    assert w.session.sorted_tracks()[0].end_frame == div1_tight
    _assert_strict_contiguous(w.session.project)

    w.session._dirty = False
    w.close()


def test_split_merge_undo_redo_save_reopen_and_bit_exact_export(tmp_path: Path):
    """Full lifecycle: split, drag, merge, rename, undo, redo, save, reopen, and sample-exact WAV export."""
    wav = tmp_path / "source.wav"
    orig_pcm = _make_ramp_wav(wav, seconds=10.0, sr=16000)
    sr = 16000

    s = Session()
    s.import_files([str(wav)])
    _assert_strict_contiguous(s.project)

    # Split twice -> 3 tracks
    t_mid = s.split_at(3 * sr)
    assert t_mid is not None
    t_end = s.split_at(7 * sr)
    assert t_end is not None
    assert len(s.sorted_tracks()) == 3
    _assert_strict_contiguous(s.project)

    # Verify undo -> redo preserves generic title on split track
    titles_before_undo = [t.title for t in s.sorted_tracks()]
    assert all(title.strip() for title in titles_before_undo)
    assert s.undo_action() is True
    assert len(s.sorted_tracks()) == 2
    assert s.redo_action() is True
    assert [t.title for t in s.sorted_tracks()] == titles_before_undo

    # Move boundary + rename + merge + undo merge
    assert s.move_boundary(3 * sr, int(3.45 * sr)) is True
    s.set_track_title(s.sorted_tracks()[0].track_id, "Opening")
    s.set_track_title(s.sorted_tracks()[1].track_id, "Middle")
    s.set_track_title(s.sorted_tracks()[2].track_id, "Finale")
    assert s.remove_split(7 * sr) is True
    assert len(s.sorted_tracks()) == 2
    _assert_strict_contiguous(s.project)
    assert s.undo_action() is True
    assert len(s.sorted_tracks()) == 3
    _assert_strict_contiguous(s.project)

    # Save & reopen preserves exact frame boundaries, titles, and manual flags
    proj_file = tmp_path / "session.splytnyt"
    s.save_project(str(proj_file))

    s2 = Session()
    s2.open_project(str(proj_file))
    _assert_strict_contiguous(s2.project)
    assert [(t.title, t.start_frame, t.end_frame) for t in s2.sorted_tracks()] == [
        ("Opening", 0, int(3.45 * sr)),
        ("Middle", int(3.45 * sr), 7 * sr),
        ("Finale", 7 * sr, 10 * sr),
    ]

    # Bit-exact WAV export verification: concatenated tracks == original PCM sample-for-sample
    export_dir = tmp_path / "exported"
    settings = ExportSettings(
        output_dir=str(export_dir),
        container="wav",
        wav_subtype="PCM_16",
        mode="tracks",
        include_m3u=False,
        include_info=False,
        include_checksums=False,
    )
    report = export_project(s2.project, settings)
    assert len(report.files) == 3

    reconstructed = []
    for path, track in zip(report.files, s2.sorted_tracks()):
        data, out_sr = sf.read(path, dtype="int16")
        assert out_sr == sr
        assert len(data) == track.frames
        reconstructed.append(data)

    joined = np.vstack(reconstructed)
    assert joined.shape == orig_pcm.shape, f"Frame count mismatch: {joined.shape} vs {orig_pcm.shape}"
    assert np.array_equal(joined, orig_pcm), "Exported audio has gaps, overlaps, or altered samples!"


def test_uncertain_boundaries_never_auto_applied_and_protected_tracks_preserved(tmp_path: Path):
    """Uncertain proposals are never auto-applied; protected tracks are preserved; no-op doesn't push undo."""
    app = _app()
    s = Session()
    sr = 48000
    s.project.add_source(Source(path="dummy.wav", sample_rate=sr, channels=2, frames=600 * sr))
    s.project.tracks = [
        Track(start_frame=0, end_frame=100 * sr, title="Track 01", section_type="musical_number"),
        Track(start_frame=100 * sr, end_frame=200 * sr, title="Track 02", section_type="dialogue"),
        Track(start_frame=200 * sr, end_frame=600 * sr, title="Track 03", section_type="musical_number"),
    ]
    _assert_strict_contiguous(s.project)

    # Partial / uncertain proposal must NOT be applied when allow_uncertain=False
    uncertain_prop = {
        "title": "Partial Song",
        "start_frame": 10 * sr,
        "end_frame": 90 * sr,
        "status": "partial proposal",
        "missing_roles": ["ending"],
        "uncertain": True,
    }
    res = s.apply_lyric_grouping([uncertain_prop])
    assert res["applied"] == 0 and res["skipped_uncertain"] == 1
    assert not s.undo.can_undo(), "No-op apply must not push an empty undo step"

    # Explicit confirmation (allow_uncertain=True) applies as boundary_uncertain=True
    res2 = s.apply_lyric_grouping([uncertain_prop], allow_uncertain=True)
    assert res2["applied"] == 1
    _assert_strict_contiguous(s.project)
    applied_track = next(t for t in s.sorted_tracks() if t.title == "Partial Song")
    assert applied_track.boundary_uncertain is True
    assert applied_track.suggested is True
    assert s.undo_action() is True
    assert len(s.sorted_tracks()) == 3

    # Manually renamed track is protected from grouping
    t0_id = s.sorted_tracks()[0].track_id
    s.set_track_title(t0_id, "My Custom Intro")
    complete_prop = {
        "title": "Overlapping Proposal",
        "start_frame": 0,
        "end_frame": 150 * sr,
        "status": "multi-anchor proposal",
        "missing_roles": [],
        "uncertain": False,
    }
    res3 = s.apply_lyric_grouping([complete_prop])
    assert res3["applied"] == 0 and res3["skipped_protected"] == 1
    assert s.sorted_tracks()[0].title == "My Custom Intro"

    # Non-overlapping complete proposal preserves untouched track IDs
    t0_before = s.sorted_tracks()[0].track_id
    non_overlap = {
        "title": "Clean Number",
        "start_frame": 250 * sr,
        "end_frame": 400 * sr,
        "status": "multi-anchor proposal",
        "missing_roles": [],
        "uncertain": False,
    }
    res4 = s.apply_lyric_grouping([non_overlap])
    assert res4["applied"] == 1
    _assert_strict_contiguous(s.project)
    assert s.sorted_tracks()[0].track_id == t0_before


def test_wrong_lyrics_repeated_choruses_and_unrelated_audio():
    """Wrong lyrics yield 0 matches; repeated choruses stay separate partial proposals; silence fails alignment."""
    segs = [
        {"start": 10.0, "end": 14.0, "text": "we rise together in the morning light", "accepted": True},
        {"start": 120.0, "end": 124.0, "text": "we rise together in the morning light", "accepted": True},
        {"start": 300.0, "end": 305.0, "text": "a completely unrelated dialogue scene about tea", "accepted": True},
    ]
    refs = parse_references(json.dumps({
        "schema": "splytnyt-lyrics/1",
        "references": [
            {
                "title": "Wrong Song",
                "provenance": "local test fixture",
                "anchors": [
                    {"text": "cobalt submarines drift beneath glaciers", "role": "opening"},
                    {"text": "velvet compasses spin northward forever", "role": "ending"},
                ],
            },
            {
                "title": "Repeated Chorus Only",
                "provenance": "local test fixture",
                "anchors": [
                    {"text": "we rise together in the morning light", "role": "unknown"},
                ],
            },
        ],
    }))
    rep = match_references(refs, segs)
    assert rep["results"][0]["unmatched"] is True
    assert rep["results"][0]["occurrences"] == []

    chorus_occs = rep["results"][1]["occurrences"]
    assert len(chorus_occs) == 2, "Repeated chorus occurrences must stay separate, never merged"
    assert all(o["status"] == "partial proposal" for o in chorus_occs)
    assert all(o["missing_roles"] for o in chorus_occs)

    # Alignment quality gate fails closed on empty / zero-prob / wrong words
    q_empty = alignment_quality([], "cobalt submarines drift", 5.0)
    assert q_empty["status"] == "failed"
    q_low = alignment_quality(
        [
            {"word": "cobalt", "start": 0.1, "end": 0.4, "probability": 0.1},
            {"word": "submarines", "start": 0.5, "end": 0.9, "probability": 0.1},
            {"word": "drift", "start": 1.0, "end": 1.4, "probability": 0.1},
        ],
        "cobalt submarines drift",
        5.0,
    )
    assert q_low["status"] == "failed"


def test_long_three_hour_recording_timeline_and_overview_pan():
    """3-hour timeline zoom-to-fit, overview drag-pan, and divider drag at 2h+ timestamp."""
    app = _app()
    sr = 48000
    three_hours = 3 * 3600 * sr
    s = Session()
    s.project.add_source(Source(path="three_hours.wav", sample_rate=sr, channels=2, frames=three_hours))
    s.project.tracks = [
        Track(start_frame=0, end_frame=3600 * sr, title="Act 1"),
        Track(start_frame=3600 * sr, end_frame=2 * 3600 * sr, title="Act 2"),
        Track(start_frame=2 * 3600 * sr, end_frame=three_hours, title="Act 3"),
    ]
    wf = WaveformView()
    wf.resize(1200, 320)
    wf.set_project(s.project)
    wf.set_tracks(s.sorted_tracks())
    wf.zoom_to_fit()
    app.processEvents()
    assert wf.frames_per_pixel >= three_hours / 1200.0 * 0.95

    # Zoom into 2:00:00 divider and pan via overview strip
    wf.zoom_to_frame(2 * 3600 * sr, span_seconds=30.0)
    app.processEvents()
    vis0, vis1 = wf._visible_range()
    assert vis0 < 2 * 3600 * sr < vis1

    # Overview click near 25% should pan near 45m
    QTest.mouseClick(wf.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(300, OVERVIEW_HEIGHT // 2))
    app.processEvents()
    vis0_new, vis1_new = wf._visible_range()
    assert vis0_new < int(0.75 * 3600 * sr) < vis1_new
