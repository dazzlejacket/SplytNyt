"""Regression tests for theatrical reliability, caching, and honest naming."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from splytnyt.analysis import aec
from splytnyt.analysis.align import (
    align_theater_worklist,
    lexical_item_evidence,
)
from splytnyt.analysis.theater import (
    TheaterSection,
    _cohere_musical_numbers,
)
from splytnyt.analysis.transcribe import Transcriber
from splytnyt.app.jobs import (
    TheaterJob,
    _decode_events_cache,
    _encode_events_cache,
    aec_settings_fingerprint,
    transcript_settings_fingerprint,
)
from splytnyt.app.session import Session
from splytnyt.core.model import (
    ActivityRegion,
    Project,
    Source,
    WorklistItem,
)
from splytnyt.core.undo import UndoStack


# ----------------------------------------------------------------------
# YAMNet grid alignment (zero cumulative drift)
# ----------------------------------------------------------------------
def test_aec_block_frame_slice_keeps_exact_grid():
    assert aec.BLOCK_FRAMES * aec.FRAME_SECONDS == pytest.approx(60.0)
    assert aec.BLOCK_PADDING_FRAMES * aec.FRAME_SECONDS == pytest.approx(1.92)
    assert aec._block_frame_slice(132, left_frames=4, max_frames=125) == (4, 129)
    assert aec._block_frame_slice(129, left_frames=0, max_frames=125) == (0, 125)
    assert aec._block_frame_slice(3, left_frames=4, max_frames=125) == (0, 0)


def test_classify_events_and_reader_have_zero_drift(monkeypatch):
    """Multi-block AEC must produce exact step-aligned frame count and zero lag."""
    sr_16k = aec.MODEL_SAMPLE_RATE
    step = int(round(aec.FRAME_SECONDS * sr_16k))
    window = int(round(aec.FRAME_WINDOW_SECONDS * sr_16k))

    # Build a synthetic 200 s 16 kHz signal where only t in [120, 150) has high amplitude.
    total_sec = 200
    n_samples = total_sec * sr_16k
    signal = np.zeros(n_samples, dtype=np.float32) + 0.01
    signal[120 * sr_16k : 150 * sr_16k] = 0.9

    class _FakeSession:
        def run(self, _outputs, inputs):
            wav = inputs["waveform"]
            frames = []
            pos = 0
            while pos + window <= wav.size:
                amp = float(np.mean(np.abs(wav[pos : pos + window])))
                row = np.zeros(521, dtype=np.float32)
                row[0] = min(1.0, amp)  # Speech/Music index 0 or class mapped below
                row[1] = 1.0 - row[0]
                frames.append(row)
                pos += step
            return [np.stack(frames, axis=0) if frames else np.zeros((0, 521), dtype=np.float32)]

    fake_map = {"Music": 0, "Speech": 1, "Singing": 2, "Applause": 3, "Silence": 4}
    monkeypatch.setattr(aec, "load_session", lambda root=None: (_FakeSession(), fake_map))

    events = aec.classify_events(signal)
    expected_frames = (n_samples - window) // step + 1
    assert events.n_frames() == expected_frames

    # Peak music score must fall right inside [120/0.48 .. 150/0.48), zero drift.
    idx_125 = int(round(125.0 / aec.FRAME_SECONDS))
    idx_30 = int(round(30.0 / aec.FRAME_SECONDS))
    assert events.music[idx_125] > 0.7
    assert events.music[idx_30] < 0.05


# ----------------------------------------------------------------------
# UndoStack restores activities
# ----------------------------------------------------------------------
def test_undo_restores_project_activities():
    project = Project()
    stack = UndoStack(project)

    def add_act(p: Project) -> None:
        p.activities = [
            ActivityRegion(start_frame=0, end_frame=48000, label="singing", confidence=0.8)
        ]

    stack.apply(add_act, "Add activity")
    assert len(project.activities) == 1
    assert stack.undo()
    assert project.activities == []
    assert stack.redo()
    assert len(project.activities) == 1


# ----------------------------------------------------------------------
# Deterministic transcriber settings & spread language detection
# ----------------------------------------------------------------------
def test_transcriber_uses_deterministic_temperature_and_spread_language():
    tr = Transcriber(model_name="small", cpu_threads=4)
    assert tr.temperature == 0.0
    assert tr.beam_size == 1
    fp = tr.settings_fingerprint(language=None)
    assert fp["temperature"] == 0.0
    assert fp["beam_size"] == 1
    assert fp["language"] == "auto"

    recorded_kwargs = {}

    class _FakeModel:
        def detect_language(self, chunk):
            # Opening chunk looks ambiguous, mid-show chunks strongly "en".
            if float(np.mean(np.abs(chunk))) < 0.2:
                return "haw", 0.4, [("haw", 0.4), ("en", 0.2)]
            return "en", 0.8, [("en", 0.8), ("haw", 0.1)]

        def transcribe(self, audio, **kwargs):
            recorded_kwargs.update(kwargs)
            seg = SimpleNamespace(
                start=1.0,
                end=3.0,
                text="Hello world",
                avg_logprob=-0.2,
                no_speech_prob=0.1,
                compression_ratio=1.1,
                words=[SimpleNamespace(start=1.0, end=2.0, word="Hello", probability=0.9)],
            )
            info = SimpleNamespace(language=kwargs.get("language") or "en", language_probability=0.9, duration=90.0)
            return iter([seg]), info

    tr._model = _FakeModel()
    sr = 16000
    audio = np.zeros(90 * sr, dtype=np.float32)
    audio[: 10 * sr] = 0.05
    audio[10 * sr :] = 0.5
    res = tr.transcribe(audio, sr, language=None)
    assert res.available
    assert recorded_kwargs["temperature"] == 0.0
    assert recorded_kwargs["beam_size"] == 1
    assert recorded_kwargs["language"] == "en"


# ----------------------------------------------------------------------
# Fingerprinted caching & save/reopen reuse
# ----------------------------------------------------------------------
def test_theater_cache_reused_and_invalidated_on_settings_or_source_change(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLYTNYT_CACHE_DIR", str(tmp_path / "cache"))
    sr = 48000
    project = Project(name="cache test")
    project.sources = [
        Source(
            source_id="s1",
            path=str(tmp_path / "audio.wav"),
            frames=60 * sr,
            sample_rate=sr,
            channels=2,
            size_bytes=1000,
            head_hash="abc",
            tail_hash="def",
        )
    ]

    fp1, settings1 = aec_settings_fingerprint(project, aec)
    ev = aec.EventScores(
        music=np.full(125, 0.8, dtype=np.float32),
        speech=np.zeros(125, dtype=np.float32),
        singing=np.full(125, 0.7, dtype=np.float32),
        applause=np.zeros(125, dtype=np.float32),
        silence=np.zeros(125, dtype=np.float32),
    )
    entry = _encode_events_cache(fp1, settings1, ev)
    decoded = _decode_events_cache(entry, fp1, aec)
    assert decoded is not None
    assert np.allclose(decoded.music, ev.music)

    # Changing source hash invalidates AEC cache.
    project.sources[0].head_hash = "changed"
    fp2, _ = aec_settings_fingerprint(project, aec)
    assert fp2 != fp1
    assert _decode_events_cache(entry, fp2, aec) is None

    # Changing Whisper model or language invalidates transcript fingerprint.
    tfp_small_auto, _ = transcript_settings_fingerprint(project, "small", None)
    tfp_small_en, _ = transcript_settings_fingerprint(project, "small", "en")
    tfp_base_auto, _ = transcript_settings_fingerprint(project, "base", None)
    assert len({tfp_small_auto, tfp_small_en, tfp_base_auto}) == 3


def test_save_and_reopen_restores_cached_transcript(tmp_path):
    session = Session()
    session.new_project("reopen cache")
    sr = 48000
    session.project.sources = [
        Source(source_id="s1", path="missing.wav", frames=60 * sr, sample_rate=sr, channels=2)
    ]
    tr_fp, tr_settings = transcript_settings_fingerprint(session.project, "small", None)
    session.project.cache["theater_transcript"] = {
        "fingerprint": tr_fp,
        "settings": tr_settings,
        "language": "en",
        "language_probability": 0.9,
        "segments": [
            {"start": 5.0, "end": 8.0, "text": "The hills are alive with the sound of music", "accepted": True, "reason": ""}
        ],
    }
    proj_path = str(tmp_path / "show.splytnyt")
    session.save_project(proj_path)

    reopened = Session()
    reopened.open_project(proj_path)
    assert len(reopened._theater_transcript) == 1
    assert "sound of music" in reopened._theater_transcript[0]["text"]


def test_theater_job_reuses_cache_unless_reanalyze(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLYTNYT_CACHE_DIR", str(tmp_path / "cache"))
    sr = 48000
    project = Project(name="job cache")
    project.sources = [
        Source(source_id="s1", path="dummy.wav", frames=60 * sr, sample_rate=sr, channels=1)
    ]
    # Pre-populate project cache with matching fingerprints so no audio decode occurs for AEC/ASR.
    aec_fp, aec_settings = aec_settings_fingerprint(project, aec)
    ev = aec.EventScores(
        music=np.full(125, 0.8, dtype=np.float32),
        speech=np.zeros(125, dtype=np.float32),
        singing=np.full(125, 0.7, dtype=np.float32),
        applause=np.zeros(125, dtype=np.float32),
        silence=np.zeros(125, dtype=np.float32),
    )
    project.cache["theater_aec"] = _encode_events_cache(aec_fp, aec_settings, ev)

    tr_model = Transcriber("small")
    tr_fp, tr_settings = transcript_settings_fingerprint(project, "small", None, tr_model)
    project.cache["theater_transcript"] = {
        "fingerprint": tr_fp,
        "settings": tr_settings,
        "language": "en",
        "language_probability": 0.9,
        "segments": [
            {"start": 10.0, "end": 14.0, "text": "Raise your glass tonight", "accepted": True, "reason": ""}
        ],
    }

    job = TheaterJob(project, use_classifier=True, use_transcriber=True, reanalyze=False)
    monkeypatch.setattr(job, "_classify", lambda *a, **k: (ev, ["Reused cached audio-event classification."]))
    # Ensure _transcribe hits cache cleanly without reading audio or loading faster-whisper.
    monkeypatch.setattr(Transcriber, "load", lambda self: (_ for _ in ()).throw(AssertionError("should not load model")))
    res_segs, notes = job._transcribe(None, sr, 60 * sr)
    assert job.outcome.transcript_from_cache is True
    assert len(res_segs) == 1
    assert "Reused cached transcript." in notes

    # With reanalyze=True, cache MUST be bypassed and Transcriber.load called.
    job_re = TheaterJob(project, use_classifier=True, use_transcriber=True, reanalyze=True)
    called = {"load": False}
    monkeypatch.setattr(Transcriber, "load", lambda self: called.__setitem__("load", True) or False)
    job_re._transcribe(None, sr, 60 * sr)
    assert called["load"] is True
    assert job_re.outcome.transcript_from_cache is False


# ----------------------------------------------------------------------
# Segmentation coherence & honest lexical alignment
# ----------------------------------------------------------------------
def _sec(start_s: float, end_s: float, typ: str, mus: float = 0.6, sg: float = 0.8, sp: float = 0.0, ap: float = 0.0, sr: int = 48000):
    return TheaterSection(
        start_frame=int(start_s * sr),
        end_frame=int(end_s * sr),
        section_type=typ,
        confidence=0.6,
        activities=["singing", "instrumental"] if mus > 0.3 else ["speech"],
        reasons=[typ],
        evidence={
            "mean_music_score": mus,
            "duration_seconds": end_s - start_s,
            "activity_coverage": {"speech": sp, "singing": sg, "instrumental": mus, "applause": ap, "silence": 0.0},
        },
    )


def test_cohere_merges_brief_internal_dialogue_not_applause_or_long_scenes():
    sr = 48000
    sections = [
        _sec(0, 80, "musical_number", mus=0.65, sg=0.8),
        _sec(80, 90, "dialogue", mus=0.1, sg=0.0, sp=1.0),  # 10s internal line -> merge
        _sec(90, 120, "underscoring", mus=0.6, sg=0.8),
        _sec(120, 126, "applause", mus=0.0, sg=0.0, ap=1.0),  # applause -> DO NOT merge
        _sec(126, 200, "musical_number", mus=0.7, sg=0.8),
        _sec(200, 250, "dialogue", mus=0.05, sg=0.0, sp=1.0),  # 50s scene -> DO NOT merge
        _sec(250, 290, "musical_number", mus=0.6, sg=0.8),  # 40s sung -> stays separate
    ]
    coherent = _cohere_musical_numbers(sections, sr)
    types_spans = [(round(s.start_frame / sr), round(s.end_frame / sr), s.section_type) for s in coherent]
    assert types_spans == [
        (0, 120, "musical_number"),
        (120, 126, "applause"),
        (126, 200, "musical_number"),
        (200, 250, "dialogue"),
        (250, 290, "musical_number"),
    ]


def test_honest_alignment_never_forces_without_evidence_or_reshuffles():
    sr = 48000
    sections = [
        _sec(10, 190, "musical_number"),
        _sec(220, 400, "musical_number"),
        _sec(440, 620, "musical_number"),
    ]
    items = [
        WorklistItem(title="Firework", duration_hint=163.0),
        WorklistItem(title="Your Song", duration_hint=189.0),
        WorklistItem(title="Nature Boy", duration_hint=159.0),
    ]

    # 1. Without transcript, NOTHING may be named even when counts & durations match.
    no_tr = align_theater_worklist(items, sections, sr, transcripts=None)
    assert no_tr.assignments == []
    assert no_tr.unmatched_items == [0, 1, 2]

    # 2. Only middle segment has independent lyrics ("This is your song" x2);
    # Firework and Nature Boy stay unconfirmed instead of forcing onto wrong stretches.
    transcripts = [
        (230.0, 234.0, "My gift is my song, this is your song"),
        (260.0, 264.0, "You can tell everybody this is your song"),
    ]
    partial = align_theater_worklist(items, sections, sr, transcripts=transcripts)
    assert partial.assignments == [(1, 1)]
    assert partial.unmatched_items == [0, 2]

    # 3. Adding an extra unmatched section or changing transcript in section 0
    # must NEVER reshuffle Your Song off section 1 (now section 2).
    extra_sections = [_sec(0, 8, "musical_number")] + sections
    transcripts_changed = transcripts + [(5.0, 7.0, "Just random words here")]
    shifted = align_theater_worklist(items, extra_sections, sr, transcripts=transcripts_changed)
    assert shifted.assignments == [(1, 2)]


def test_common_phrases_and_reprise_order_disambiguation():
    sr = 48000
    # "while you're in the world" must NOT match "Only Girl (In the World)".
    only_girl = WorklistItem(
        title="Only Girl in a Material World",
        duration_hint=180.0,
        contains=["Only Girl (In the World)"],
    )
    lex, _ = lexical_item_evidence(only_girl, [(10.0, 14.0, "How wonderful life is while you're in the world")])
    assert lex == 0.0

    # "Love is a Battlefield" must NOT match just "love is a ..."
    medley = WorklistItem(
        title="Elephant Love Medley",
        duration_hint=325.0,
        contains=["Love is a Battlefield"],
    )
    lex_love, _ = lexical_item_evidence(medley, [(10.0, 14.0, "Because love is a strange game")])
    assert lex_love == 0.0

    # Duplicate/reprise songs with matching phrase disambiguate via programme order & duration.
    cwm = WorklistItem(title="Come What May", duration_hint=196.0)
    finale = WorklistItem(title="Finale (Come What May)", duration_hint=163.0, contains=["Come What May"])
    secs = [_sec(100, 290, "musical_number"), _sec(900, 1062, "musical_number")]
    tr = [
        (150.0, 155.0, "Come what may I will love you"),
        (950.0, 955.0, "Come what may until my dying day"),
    ]
    res = align_theater_worklist([cwm, finale], secs, sr, transcripts=tr)
    assert res.assignments == [(0, 0), (1, 1)]
