"""Tests for online metadata identification and metadata-assisted splitting."""

from __future__ import annotations

import wave
from dataclasses import dataclass

import numpy as np

from splytnyt.analysis.align import align_theater_worklist
from splytnyt.analysis.metadata_split import (
    split_album_with_metadata,
    split_dj_mix_with_metadata,
)
from splytnyt.app.session import Session
from splytnyt.core.model import Worklist, WorklistItem
from splytnyt.metadata.acoustid import (
    AcoustIdProvider,
    FingerprintMatch,
    FingerprintWindow,
)
from splytnyt.metadata.identify import (
    KIND_ALBUM,
    KIND_BROADWAY,
    KIND_CONCERT,
    KIND_DJ_MIX,
    IdentificationResult,
    SelectableCandidate,
    parse_recording_clues,
)
from splytnyt.metadata.musicbrainz import (
    MusicBrainzProvider,
    _track_side,
    parse_event_setlist,
)


def _write_wav(path: str, samples: np.ndarray, sample_rate: int = 22050) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class _ArrayReader:
    """Minimal TimelineReader stand-in backed by a numpy mono array."""

    def __init__(self, mono: np.ndarray) -> None:
        self._stereo = np.stack([mono, mono], axis=1).astype(np.float32)

    def read(self, start: int, count: int) -> np.ndarray:
        s = max(0, int(start))
        e = min(self._stereo.shape[0], s + max(0, int(count)))
        return self._stereo[s:e]

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------
# 1. Clue parsing
# ----------------------------------------------------------------------

def test_parse_clues_all_recording_kinds():
    c_album = parse_recording_clues([
        "Pink Floyd - Dark side of the moon - HD - High Audio Quality (Full Vinyl Album).mp3"
    ])
    assert c_album.recording_kind == KIND_ALBUM
    assert c_album.artist == "Pink Floyd"
    assert "dark side of the moon" in c_album.title.lower()
    assert c_album.format_hint == "vinyl"

    c_concert = parse_recording_clues([
        "Muse - Japan Live 2025  FULL SHOW 4K & Audio ReMaster.mp3"
    ])
    assert c_concert.recording_kind == KIND_CONCERT
    assert c_concert.artist == "Muse"
    assert c_concert.year == "2025"
    assert "japan" in c_concert.location.lower()

    c_dj = parse_recording_clues([
        "1 HORA de PURA MÚSICA ARGENTINA 100% en VINYL ft. @jimenagonik 🇦🇷 SNTS VINYL SESSIONS.mp3"
    ])
    assert c_dj.recording_kind == KIND_DJ_MIX

    c_act1 = parse_recording_clues(["Act1.mp3"])
    assert c_act1.recording_kind == KIND_BROADWAY
    assert c_act1.act_number == 1

    c_act2 = parse_recording_clues(["Act2.mp3"])
    assert c_act2.recording_kind == KIND_BROADWAY
    assert c_act2.act_number == 2


# ----------------------------------------------------------------------
# 2. MusicBrainz helpers
# ----------------------------------------------------------------------

def test_musicbrainz_side_extraction_and_setlist_parser():
    assert _track_side("A1", True, 0, 5) == "A"
    assert _track_side("B4", True, 8, 5) == "B"
    assert _track_side("1", True, 0, 5) == "A"
    assert _track_side("6", True, 5, 5) == "B"
    assert _track_side("1", False, 0, 5) == ""

    raw = """
    @ [mbid1|Muse]
    * [mbid2|Uprising]
    * [mbid3|Plug In Baby] (extended outro)
    # Encore
    * [mbid4|Knights of Cydonia]
    """
    tracks = parse_event_setlist(raw)
    assert [t.title for t in tracks] == ["Uprising", "Plug In Baby", "Knights of Cydonia"]
    assert tracks[2].is_encore is True
    assert tracks[2].set_number == "Encore"


def test_musicbrainz_release_ranking_prefers_vinyl_when_hinted():
    from splytnyt.metadata import ReleaseCandidate

    mb = MusicBrainzProvider(cache_dir="")
    rels = [
        ReleaseCandidate("MusicBrainz", "cd1", "Album", "Artist", "2011, CD, US", "2011", "CD", "US", "", 1, 10),
        ReleaseCandidate("MusicBrainz", "lp1", "Album", "Artist", "1973, 12\" Vinyl, GB", "1973", "12\" Vinyl", "GB", "", 1, 10),
    ]
    ranked = mb.rank_releases(rels, format_hint="vinyl")
    assert ranked[0].provider_id == "lp1"


# ----------------------------------------------------------------------
# 3. Known album splitting: speed drift + rip offset + vinyl side break
# ----------------------------------------------------------------------

def test_album_split_refines_with_audio_for_speed_offset_and_side_break():
    sr = 8000
    rng = np.random.default_rng(42)

    # Published MusicBrainz durations: 4 tracks (Side A: 30s, 30s; Side B: 30s, 30s)
    # Actual vinyl rip:
    #  - 4.0s leading silence (rip offset)
    #  - Side A runs 4% fast (28.8s + 28.8s) with a GAPLESS spectral change between T1 and T2
    #  - 6.0s vinyl side-flip silence gap between Side A and Side B
    #  - Side B runs 2% slow (30.6s + 30.6s) with a 1.5s quiet dip between T3 and T4
    def _tone(dur_s: float, freq: float) -> np.ndarray:
        t = np.arange(int(round(dur_s * sr)), dtype=np.float32) / sr
        return (0.25 * np.sin(2 * np.pi * freq * t) + 0.04 * rng.standard_normal(t.size)).astype(np.float32)

    lead = np.zeros(int(4.0 * sr), dtype=np.float32)
    t1 = _tone(28.8, 220.0)           # 4.0 .. 32.8
    t2 = _tone(28.8, 880.0)           # 32.8 .. 61.6 (gapless spectral shift!)
    flip = np.zeros(int(6.0 * sr), dtype=np.float32)  # 61.6 .. 67.6
    t3 = _tone(30.6, 330.0)           # 67.6 .. 98.2
    dip = np.zeros(int(1.5 * sr), dtype=np.float32)   # 98.2 .. 99.7
    t4 = _tone(30.6, 1320.0)          # 99.7 .. 130.3
    tail = np.zeros(int(3.0 * sr), dtype=np.float32)

    mono = np.concatenate([lead, t1, t2, flip, t3, dip, t4, tail])
    reader = _ArrayReader(mono)

    wl = Worklist(
        name="Test Vinyl",
        mode="album",
        provider="MusicBrainz",
        items=[
            WorklistItem(title="Track One", duration_hint=30.0, position=1, set_number="Side A"),
            WorklistItem(title="Track Two", duration_hint=30.0, position=2, set_number="Side A"),
            WorklistItem(title="Track Three", duration_hint=30.0, position=3, set_number="Side B"),
            WorklistItem(title="Track Four", duration_hint=30.0, position=4, set_number="Side B"),
        ],
    )
    ident = IdentificationResult(
        recording_kind=KIND_ALBUM,
        artist="Test Artist",
        title="Test Vinyl",
        provider="MusicBrainz",
        confidence=0.85,
        worklist=wl,
    )

    outcome = split_album_with_metadata(reader, sr, mono.size, ident)
    assert len(outcome.tracks) == 4
    assert [t.title for t in outcome.tracks] == ["Track One", "Track Two", "Track Three", "Track Four"]

    # True boundaries are at ~32.8s (gapless), ~67.6s (side B start), ~99.7s (dip)
    # Blind published timestamps would be 30.0s, 60.0s, 90.0s (all wrong by 2.8-9.7s!).
    cuts_s = [t.start_frame / sr for t in outcome.tracks[1:]]
    assert abs(cuts_s[0] - 32.8) < 2.0, f"Gapless cut off: {cuts_s[0]:.2f}s vs 32.8s"
    assert abs(cuts_s[1] - 67.6) < 2.5, f"Vinyl side break cut off: {cuts_s[1]:.2f}s vs 67.6s"
    assert abs(cuts_s[2] - 99.7) < 2.5, f"Dip cut off: {cuts_s[2]:.2f}s vs 99.7s"


# ----------------------------------------------------------------------
# 4. DJ Mix: crossfades & never inventing song identities
# ----------------------------------------------------------------------

def test_dj_mix_never_invents_identities_without_acoustid():
    sr = 8000
    rng = np.random.default_rng(7)
    # 2 continuous tracks (150s each) with a 20s crossfade around 150s
    t_a = np.arange(160 * sr, dtype=np.float32) / sr
    t_b = np.arange(160 * sr, dtype=np.float32) / sr
    sig_a = (0.25 * np.sin(2 * np.pi * 180 * t_a) + 0.1 * np.sin(2 * np.pi * 360 * t_a)).astype(np.float32)
    sig_b = (0.25 * np.sin(2 * np.pi * 620 * t_b) + 0.1 * np.sin(2 * np.pi * 1240 * t_b)).astype(np.float32)

    xf = 20 * sr
    fade_out = np.linspace(1.0, 0.0, xf, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    mono = np.concatenate([
        sig_a[:-xf],
        sig_a[-xf:] * fade_out + sig_b[:xf] * fade_in,
        sig_b[xf:],
    ]) + 0.02 * rng.standard_normal(300 * sr).astype(np.float32)

    ident = IdentificationResult(recording_kind=KIND_DJ_MIX, title="Vinyl Session")
    outcome = split_dj_mix_with_metadata(_ArrayReader(mono), sr, mono.size, ident)

    assert len(outcome.tracks) >= 2
    # Must NEVER invent artist/song titles when AcoustID had no matches
    for tr in outcome.tracks:
        assert tr.identity_confidence == 0.0
        assert tr.title.startswith("Track ")
        assert tr.boundary_uncertain is True

    # With AcoustID fingerprint windows, verified names ARE filled in
    ident_fp = IdentificationResult(
        recording_kind=KIND_DJ_MIX,
        title="Vinyl Session",
        fingerprint_windows=[
            FingerprintWindow(30.0, 55.0, FingerprintMatch(0.92, "Song Alpha", "Artist A")),
            FingerprintWindow(90.0, 115.0, FingerprintMatch(0.89, "Song Alpha", "Artist A")),
            FingerprintWindow(180.0, 205.0, FingerprintMatch(0.91, "Song Beta", "Artist B")),
            FingerprintWindow(240.0, 265.0, FingerprintMatch(0.90, "Song Beta", "Artist B")),
        ],
    )
    outcome_fp = split_dj_mix_with_metadata(_ArrayReader(mono), sr, mono.size, ident_fp)
    assert [(t.artist, t.title) for t in outcome_fp.tracks] == [("Artist A", "Song Alpha"), ("Artist B", "Song Beta")]
    cut_s = outcome_fp.tracks[1].start_frame / sr
    assert 130.0 <= cut_s <= 170.0


def test_acoustid_builtin_key_and_unconfigured_fallback(monkeypatch):
    from splytnyt.metadata.acoustid import DEFAULT_APP_KEY, api_key

    monkeypatch.delenv("ACOUSTID_API_KEY", raising=False)
    monkeypatch.setattr("splytnyt.metadata.acoustid.get_credential", lambda _name: "")
    assert api_key() == DEFAULT_APP_KEY
    assert len(DEFAULT_APP_KEY) > 4

    # Explicit empty key must never invent matches
    provider = AcoustIdProvider(cache_dir="", client_key="")
    assert provider.configured is False
    assert provider.scan_recording("nonexistent.mp3", 300.0) == []


def test_windows_credential_roundtrip_and_masking():
    from splytnyt.core.credentials import (
        delete_credential,
        get_credential,
        mask_key,
        set_credential,
    )

    test_key_name = "_splytnyt_unit_test_secret"
    secret = "test_secret_value_123456"
    try:
        assert set_credential(test_key_name, secret) is True
        assert get_credential(test_key_name) == secret
        masked = mask_key(secret)
        assert secret not in masked
        assert masked.startswith("tes") and masked.endswith("456")
    finally:
        assert delete_credential(test_key_name) is True
        assert get_credential(test_key_name) == ""


def test_setlistfm_v1_json_parsing_and_country_mapping():
    from splytnyt.metadata.setlistfm import (
        _parse_performance,
        _parse_tracks,
        country_name_to_code,
    )

    assert country_name_to_code("Japan") == "JP"
    assert country_name_to_code("United Kingdom") == "GB"

    sample_v1 = {
        "id": "1b4475c4",
        "eventDate": "25-09-2025",
        "artist": {"name": "Muse", "mbid": "9c9f1380-2516-4fc9-a3e6-f9f61941d090"},
        "venue": {
            "name": "K-Arena Yokohama",
            "city": {"name": "Yokohama", "country": {"code": "JP", "name": "Japan"}},
        },
        "sets": {
            "set": [
                {
                    "song": [
                        {"name": "Unravelling"},
                        {"name": "Simulation Theory Theme / [JFK]", "tape": True},
                    ]
                },
                {
                    "encore": 1,
                    "song": [
                        {"name": "Starlight"},
                    ],
                },
            ]
        },
    }
    perf = _parse_performance(sample_v1)
    assert perf is not None
    assert perf.artist == "Muse"
    assert perf.venue == "K-Arena Yokohama"
    assert perf.city == "Yokohama"
    assert perf.song_count == 3

    tracks = _parse_tracks(sample_v1)
    assert [t.title for t in tracks] == [
        "Unravelling",
        "Simulation Theory Theme / [JFK]",
        "Starlight",
    ]
    assert "Tape" in tracks[1].notes
    assert tracks[2].is_encore is True


def test_music_services_dialog_instantiates_safely():
    from PySide6.QtWidgets import QApplication
    from splytnyt.ui.settings_dialog import MusicServicesDialog

    app = QApplication.instance() or QApplication([])
    dlg = MusicServicesDialog()
    assert "AcoustID" in dlg.acoustid_status.text() or "Built-in" in dlg.acoustid_status.text() or "Custom" in dlg.acoustid_status.text()
    dlg.close()


# ----------------------------------------------------------------------
# 5. Broadway reference lyrics enhancement in align_theater_worklist
# ----------------------------------------------------------------------

@dataclass
class _DummySection:
    start_frame: int
    end_frame: int
    section_type: str = "musical_number"


def test_theater_alignment_uses_reference_lyrics():
    items = [
        WorklistItem(title="The Sparkling Diamond", duration_hint=240.0),
        WorklistItem(title="Come What May", duration_hint=250.0),
    ]
    sections = [
        _DummySection(0, 260 * 1000),
        _DummySection(300 * 1000, 580 * 1000),
    ]
    # Transcript contains chorus lyrics rather than the exact title "The Sparkling Diamond"
    transcripts = [
        (20.0, 30.0, "diamonds are a girls best friend"),
        (45.0, 55.0, "a kiss on the hand may be quite continental"),
        (340.0, 350.0, "never knew i could feel like this"),
    ]
    lyrics = {
        "The Sparkling Diamond": "A kiss on the hand may be quite continental\nDiamonds are a girl's best friend",
        "Come What May": "Never knew I could feel like this\nLike I've never seen the sky before",
    }

    # Without reference lyrics, title words don't appear in transcript
    al_plain = align_theater_worklist(items, sections, 1000, transcripts=transcripts)
    assert len(al_plain.assignments) == 0

    # With LRCLIB reference lyrics, both numbers match cleanly!
    al_lyrics = align_theater_worklist(
        items, sections, 1000, transcripts=transcripts, reference_lyrics=lyrics
    )
    assert al_lyrics.assignments == [(0, 0), (1, 1)]
    assert "lyrics" in al_lyrics.evidence[(0, 0)]


# ----------------------------------------------------------------------
# 6. Session reversibility, locking, and shared dividers
# ----------------------------------------------------------------------

def test_session_apply_metadata_split_is_undoable_and_contiguous(tmp_path):
    sr = 8000
    wav_path = str(tmp_path / "album.wav")
    _write_wav(wav_path, np.zeros(60 * sr, dtype=np.float32), sample_rate=sr)

    sess = Session()
    sess.import_files([wav_path])
    assert len(sess.project.tracks) == 1

    from splytnyt.analysis.metadata_split import MetadataSplitOutcome, PlannedTrack

    ident = IdentificationResult(
        recording_kind=KIND_ALBUM,
        artist="Pink Floyd",
        title="The Dark Side of the Moon",
        provider="MusicBrainz",
        confidence=0.85,
        candidates=[
            SelectableCandidate("1973 Vinyl", "MusicBrainz", "rel1", "mb_release"),
        ],
    )
    outcome = MetadataSplitOutcome(
        identification=ident,
        tracks=[
            PlannedTrack(0, 20 * sr, title="Speak to Me", boundary_confidence=0.8, identity_confidence=0.8),
            PlannedTrack(20 * sr, 40 * sr, title="Breathe", boundary_confidence=0.75, identity_confidence=0.8),
            PlannedTrack(40 * sr, 60 * sr, title="On the Run", boundary_confidence=0.75, identity_confidence=0.8),
        ],
    )
    sess.apply_metadata_split(outcome)
    assert [t.title for t in sess.project.sorted_tracks()] == ["Speak to Me", "Breathe", "On the Run"]
    assert sess.project.is_contiguous()
    assert len(sess.project.shared_boundaries()) == 2

    # Editable title
    t1 = sess.project.sorted_tracks()[0]
    sess.set_track_title(t1.track_id, "Speak to Me (Intro)")
    assert sess.project.sorted_tracks()[0].title == "Speak to Me (Intro)"

    # Reversible via Undo
    sess.undo_action()  # undo rename
    sess.undo_action()  # undo metadata split
    assert len(sess.project.tracks) == 1
    assert sess.project.tracks[0].title == "Full recording"


def test_main_window_drag_drop_and_smart_banner(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from splytnyt.analysis.metadata_split import MetadataSplitOutcome, PlannedTrack
    from splytnyt.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        assert win.acceptDrops() is True
        assert win.smart_banner.isHidden()

        sr = 8000
        wav_path = str(tmp_path / "dsotm.wav")
        _write_wav(wav_path, np.zeros(40 * sr, dtype=np.float32), sample_rate=sr)
        win.session.import_files([wav_path])

        ident = IdentificationResult(
            recording_kind=KIND_ALBUM,
            artist="Pink Floyd",
            title="The Dark Side of the Moon",
            provider="MusicBrainz",
            confidence=0.85,
            candidates=[
                SelectableCandidate("1973 Vinyl GB", "MusicBrainz", "r1", "mb_release"),
                SelectableCandidate("1994 CD US", "MusicBrainz", "r2", "mb_release"),
            ],
            selected_index=0,
        )
        outcome = MetadataSplitOutcome(
            identification=ident,
            tracks=[
                PlannedTrack(0, 20 * sr, title="Speak to Me", boundary_confidence=0.8, identity_confidence=0.8),
                PlannedTrack(20 * sr, 40 * sr, title="Breathe", boundary_confidence=0.8, identity_confidence=0.8),
            ],
        )
        win._on_smart_split_done(outcome)

        assert not win.smart_banner.isHidden()
        assert win.release_combo.count() == 2
        assert "Pink Floyd" in win.smart_banner_label.text()
        assert [t.title for t in win.session.project.sorted_tracks()] == ["Speak to Me", "Breathe"]
        assert win.session.project.is_contiguous()
    finally:
        win.session._dirty = False
        win.close()
        app.processEvents()
