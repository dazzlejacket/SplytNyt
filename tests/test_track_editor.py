"""Tests for the simplified drag-and-drop track editor and shared-boundary invariants."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from splytnyt.app.session import Session
from splytnyt.core.model import Source, Track
from splytnyt.ui.main_window import MainWindow
from splytnyt.ui.waveform import HEADER_HEIGHT, OVERVIEW_HEIGHT, WaveformView


def _make_session(total_seconds: int = 300, sr: int = 48000) -> Session:
    s = Session()
    total_frames = total_seconds * sr
    s.project.sources = [
        Source(path="test.wav", sample_rate=sr, channels=2, frames=total_frames)
    ]
    s.project.tracks = [
        Track(start_frame=0, end_frame=total_frames, title="Full recording")
    ]
    return s


def test_shared_boundary_invariant_across_split_drag_remove_and_undo():
    sr = 48000
    s = _make_session(300, sr)
    assert s.project.is_contiguous()

    # Split into 3 tracks
    t2 = s.split_at(100 * sr)
    t3 = s.split_at(200 * sr)
    assert t2 is not None and t3 is not None
    tracks = s.sorted_tracks()
    assert len(tracks) == 3
    assert s.project.is_contiguous()
    assert tracks[0].end_frame == tracks[1].start_frame == 100 * sr
    assert tracks[1].end_frame == tracks[2].start_frame == 200 * sr

    # Move divider between Track 1 and Track 2 -> both update atomically
    assert s.move_boundary(100 * sr, 125 * sr) is True
    tracks = s.sorted_tracks()
    assert tracks[0].end_frame == tracks[1].start_frame == 125 * sr
    assert s.project.is_contiguous()

    # Cannot drag past adjacent track's end
    assert s.move_boundary(125 * sr, 200 * sr) is False
    assert s.move_boundary(125 * sr, 0) is False
    assert s.project.is_contiguous()

    # Remove split merges without losing any audio
    assert s.remove_split(125 * sr) is True
    tracks = s.sorted_tracks()
    assert len(tracks) == 2
    assert tracks[0].start_frame == 0 and tracks[0].end_frame == 200 * sr
    assert tracks[1].start_frame == 200 * sr and tracks[1].end_frame == 300 * sr
    assert s.project.is_contiguous()

    # Undo restores exact prior divider
    assert s.undo_action() is True
    tracks = s.sorted_tracks()
    assert len(tracks) == 3
    assert tracks[0].end_frame == tracks[1].start_frame == 125 * sr
    assert s.project.is_contiguous()


def test_reorder_tracks_preserves_chronological_audio_order_and_respects_locks():
    sr = 48000
    s = _make_session(300, sr)
    s.split_at(100 * sr)
    s.split_at(200 * sr)
    tracks = s.sorted_tracks()
    s.set_track_title(tracks[0].track_id, "Opening")
    s.set_track_title(tracks[1].track_id, "Middle")
    s.set_track_title(tracks[2].track_id, "Finale")

    spans_before = [(t.start_frame, t.end_frame) for t in s.sorted_tracks()]

    # Move "Finale" label up to index 0 -> audio spans MUST stay identical
    assert s.reorder_track(s.sorted_tracks()[2].track_id, 0) is True
    after = s.sorted_tracks()
    assert [(t.start_frame, t.end_frame) for t in after] == spans_before
    assert [t.title for t in after] == ["Finale", "Opening", "Middle"]
    assert s.project.is_contiguous()

    # Undo restores titles cleanly
    assert s.undo_action() is True
    assert [t.title for t in s.sorted_tracks()] == ["Opening", "Middle", "Finale"]
    assert [(t.start_frame, t.end_frame) for t in s.sorted_tracks()] == spans_before

    # Lock middle track -> cannot reorder across it, split it, or move its dividers
    mid = s.sorted_tracks()[1]
    s.toggle_lock(mid.track_id)
    assert s.move_track_up(s.sorted_tracks()[2].track_id) is False
    assert s.split_at(150 * sr) is None
    assert s.move_boundary(100 * sr, 110 * sr) is False
    assert s.remove_split(100 * sr) is False
    assert s.project.is_contiguous()


def test_waveform_multihour_zoom_and_live_divider_drag():
    app = QApplication.instance() or QApplication([])
    sr = 48000
    # 3-hour recording (518.4 million frames)
    total_frames = 3 * 3600 * sr
    w = MainWindow()
    if hasattr(w, "_autosave_timer"):
        w._autosave_timer.stop()
    w.session.project.sources = [
        Source(path="three_hours.wav", sample_rate=sr, channels=2, frames=total_frames)
    ]
    w.session.project.tracks = [
        Track(start_frame=0, end_frame=total_frames // 3, title="Act 1"),
        Track(start_frame=total_frames // 3, end_frame=2 * total_frames // 3, title="Act 2"),
        Track(start_frame=2 * total_frames // 3, end_frame=total_frames, title="Act 3"),
    ]
    w._reload()
    w.resize(1200, 700)
    w.show()
    app.processEvents()

    w.waveform.zoom_to_fit()
    fit_fpp = w.waveform.frames_per_pixel
    assert fit_fpp > 200000.0  # 3 hours across ~800-1100 px

    # Zooming in by 0.8 from zoom_to_fit must NOT snap down to 2000 fpp!
    w.waveform.zoom_by(0.8)
    assert abs(w.waveform.frames_per_pixel - fit_fpp * 0.8) < 10.0

    # Dragging the first divider updates both Track 1 end and Track 2 start
    w.waveform.zoom_to_fit()
    app.processEvents()
    old_div = total_frames // 3
    div_x = w.waveform._frame_to_x(old_div)
    target_x = div_x + 40.0
    y = float(HEADER_HEIGHT + 30)

    press = QMouseEvent(
        QMouseEvent.MouseButtonPress,
        QPointF(div_x, y),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    move = QMouseEvent(
        QMouseEvent.MouseMove,
        QPointF(target_x, y),
        Qt.NoButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    release = QMouseEvent(
        QMouseEvent.MouseButtonRelease,
        QPointF(target_x, y),
        Qt.LeftButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    w.waveform.mousePressEvent(press)
    assert w.waveform._drag_origin == old_div
    w.waveform.mouseMoveEvent(move)
    assert w.waveform._drag_frame > old_div
    w.waveform.mouseReleaseEvent(release)
    app.processEvents()

    tracks = w.session.sorted_tracks()
    assert tracks[0].end_frame == tracks[1].start_frame
    assert tracks[0].end_frame > old_div
    assert w.session.project.is_contiguous()

    # Single-click track selects without seeking; double-click seeks
    w.waveform.set_playhead(0)
    click_x = w.waveform._frame_to_x(tracks[2].start_frame + sr * 600)
    w.waveform.mousePressEvent(
        QMouseEvent(QMouseEvent.MouseButtonPress, QPointF(click_x, y), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    )
    assert w._selected_track_id == tracks[2].track_id
    assert w.waveform._playhead == 0  # single-click did not seek

    w.waveform.mouseDoubleClickEvent(
        QMouseEvent(QMouseEvent.MouseButtonDblClick, QPointF(click_x, y), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    )
    assert w.waveform.cursor_frame > tracks[2].start_frame

    w.session._dirty = False
    w.close()
    app.processEvents()
