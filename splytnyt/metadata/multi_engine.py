"""Multi-engine song identification combining AcoustID and optional ShazamIO.

Design rules:
* **Never invent song identities.** Unrecognized audio stays unidentified.
* **Sample multiple interior points**, avoiding applause, silence, and dialogue
  when acoustic features or YAMNet audio events are available.
* **Compare across clips and across engines**:
  - ``confirmed``: 2+ consistent acoustic recognition hits (across clips or
    across AcoustID + ShazamIO).
  - ``likely``: 1 strong acoustic match (score >= 0.70) without a second
    corroborating hit.
  - ``conflict``: AcoustID and ShazamIO disagreed on the song identity.
  - ``uncertain``: weak or inconsistent acoustic evidence.
  - ``unidentified``: no acoustic match.
* **Never call a track confirmed merely because MusicBrainz or a setlist
  suggests it.** Worklist alignment stays separate from acoustic recognition.
* **Detect intra-segment song changes** when multiple clips in one segment
  identify different songs, returning candidate transition intervals so the
  splitter can refine boundaries against audio transitions (never blindly
  cutting at the clip timestamp).
* **Broadway / Jukebox guard**: distinguish original pop recordings from cast
  recordings so an original pop song is never mistaken for a Broadway
  arrangement.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from . import ProviderError
from .acoustid import ACOUSTID, AcoustIdProvider, FingerprintMatch, chromaprint_backend
from .musicbrainz import MusicBrainzProvider
from .shazamio_engine import SHAZAMIO, ShazamIoProvider, ShazamMatch

STATUS_CONFIRMED = "confirmed"
STATUS_LIKELY = "likely"
STATUS_UNCERTAIN = "uncertain"
STATUS_CONFLICT = "conflict"
STATUS_UNIDENTIFIED = "unidentified"

STATUS_LABELS = {
    STATUS_CONFIRMED: "Confirmed",
    STATUS_LIKELY: "Likely",
    STATUS_UNCERTAIN: "Uncertain",
    STATUS_CONFLICT: "Conflict",
    STATUS_UNIDENTIFIED: "Unidentified",
}


# ----------------------------------------------------------------------
# Text normalization & song identity comparison
# ----------------------------------------------------------------------

_PAREN_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_FEAT_RE = re.compile(r"\b(?:feat\.?|ft\.?|featuring)\b.*$", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    """Fold accents, strip parenthetical/featuring tags, and normalize whitespace."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKD", text)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _PAREN_RE.sub(" ", s)
    s = _FEAT_RE.sub(" ", s)
    s = _NON_ALNUM_RE.sub(" ", s.lower()).strip()
    return s


def same_song(
    artist_a: str,
    title_a: str,
    artist_b: str,
    title_b: str,
) -> bool:
    """True when two (artist, title) pairs plausibly refer to the same song."""
    ta = normalize_text(title_a)
    tb = normalize_text(title_b)
    if not ta or not tb:
        return False

    title_match = (
        ta == tb
        or (len(ta) >= 4 and ta in tb)
        or (len(tb) >= 4 and tb in ta)
    )
    if not title_match:
        toks_a = set(ta.split())
        toks_b = set(tb.split())
        if toks_a and toks_b:
            overlap = len(toks_a & toks_b) / float(max(len(toks_a), len(toks_b)))
            title_match = overlap >= 0.70
    if not title_match:
        return False

    aa = normalize_text(artist_a)
    ab = normalize_text(artist_b)
    if not aa or not ab:
        return True
    if aa == ab or aa in ab or ab in aa:
        return True
    toks_aa = set(aa.split())
    toks_ab = set(ab.split())
    if toks_aa and toks_ab:
        if len(toks_aa & toks_ab) / float(min(len(toks_aa), len(toks_ab))) >= 0.50:
            return True
    return False


def format_artist_title(artist: str, title: str) -> str:
    a = (artist or "").strip()
    t = (title or "").strip()
    if a and t:
        return f"{a} - {t}"
    return t or a


def is_cast_recording(artist: str, album: str, show_title: str = "") -> bool:
    """True when metadata indicates a theatrical / Broadway cast recording."""
    combined = f"{artist} {album}".lower()
    cast_markers = (
        "broadway",
        "cast",
        "musical",
        "west end",
        "ensemble",
        "soundtrack",
        "motion picture",
    )
    if any(m in combined for m in cast_markers):
        return True
    if show_title:
        norm_show = normalize_text(show_title)
        if norm_show and norm_show in normalize_text(combined):
            return True
    return False


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class ClipRecognition:
    """One acoustic recognition hit at a specific time inside a recording."""

    engine: str  # "AcoustID" | "ShazamIO"
    start_seconds: float
    end_seconds: float
    title: str
    artist: str = ""
    album: str = ""
    score: float = 0.0
    recording_id: str = ""
    isrc: str = ""
    offset_seconds: float = 0.0
    timeskew: float = 0.0

    @property
    def label(self) -> str:
        return format_artist_title(self.artist, self.title)


@dataclass
class SongChange:
    """Detected change from one recognized song to another inside a segment."""

    prev_clip_time: float
    next_clip_time: float
    prev_artist: str
    prev_title: str
    next_artist: str
    next_title: str
    #: Optional estimated start time of the next song (e.g. clip_time - shazam offset).
    estimated_boundary_seconds: Optional[float] = None
    engine: str = ""


@dataclass
class SegmentRecognition:
    """Aggregated multi-clip, multi-engine identification for one segment."""

    start_seconds: float
    end_seconds: float
    status: str = STATUS_UNIDENTIFIED
    artist: str = ""
    title: str = ""
    album: str = ""
    confidence: float = 0.0
    engines: list[str] = field(default_factory=list)
    clips: list[ClipRecognition] = field(default_factory=list)
    conflict_detail: str = ""
    song_changes: list[SongChange] = field(default_factory=list)
    mb_recording_id: str = ""
    mb_duration_seconds: float = 0.0
    reasons: list[str] = field(default_factory=list)
    #: For Broadway/jukebox recordings: recognized original pop songs kept separate
    #: from the cast number title so they are never confused with the Broadway arrangement.
    constituent_songs: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return format_artist_title(self.artist, self.title)

    @property
    def has_conflict(self) -> bool:
        return self.status == STATUS_CONFLICT


# ----------------------------------------------------------------------
# Smart clip position selection (avoid silence, applause, dialogue)
# ----------------------------------------------------------------------

def choose_sample_timestamps(
    start_seconds: float,
    end_seconds: float,
    clip_seconds: float = 10.0,
    n_clips: Optional[int] = None,
    coarse_features=None,
    audio_events=None,
) -> list[float]:
    """Choose interior clip start timestamps inside ``[start_seconds, end_seconds]``.

    Avoids leading/trailing transition margins and prefers high-music,
    low-silence, low-applause, low-speech stretches when ``coarse_features``
    or YAMNet ``audio_events`` are provided.
    """
    seg_dur = float(end_seconds) - float(start_seconds)
    if seg_dur < max(12.0, clip_seconds + 2.0):
        return []

    if n_clips is None:
        if seg_dur >= 360.0:
            n_clips = 4
        elif seg_dur >= 150.0:
            n_clips = 3
        elif seg_dur >= 55.0:
            n_clips = 2
        else:
            n_clips = 1

    margin = min(14.0, max(4.0, seg_dur * 0.10))
    usable_start = start_seconds + margin
    usable_end = end_seconds - margin - clip_seconds
    if usable_end <= usable_start:
        return [max(start_seconds, start_seconds + 0.5 * (seg_dur - clip_seconds))]

    if n_clips == 1:
        fracs = [0.45]
    elif n_clips == 2:
        fracs = [0.28, 0.70]
    elif n_clips == 3:
        fracs = [0.20, 0.50, 0.78]
    else:
        fracs = [0.15 + 0.70 * (i / float(n_clips - 1)) for i in range(n_clips)]

    chosen: list[float] = []
    for frac in fracs:
        target = usable_start + frac * (usable_end - usable_start)
        best_t = _refine_clip_time(
            target,
            usable_start,
            usable_end,
            clip_seconds,
            coarse_features=coarse_features,
            audio_events=audio_events,
        )
        # Ensure clips do not heavily overlap each other
        if chosen and abs(best_t - chosen[-1]) < clip_seconds * 0.9:
            continue
        chosen.append(round(best_t, 2))
    return chosen


def _refine_clip_time(
    target_s: float,
    lo_s: float,
    hi_s: float,
    clip_seconds: float,
    coarse_features=None,
    audio_events=None,
) -> float:
    """Search ±12 s around ``target_s`` for the cleanest music window."""
    if coarse_features is None and audio_events is None:
        return float(np.clip(target_s, lo_s, hi_s))

    search_lo = max(lo_s, target_s - 12.0)
    search_hi = min(hi_s, target_s + 12.0)
    if search_hi <= search_lo + 1.0:
        return float(np.clip(target_s, lo_s, hi_s))

    candidates = np.linspace(search_lo, search_hi, num=9)
    best_t = float(target_s)
    best_score = -1e9

    for t in candidates:
        score = 0.0
        # Gentle preference to stay near the slot center
        score -= 0.015 * abs(float(t) - target_s)

        if audio_events is not None and audio_events.n_frames() > 0:
            dt = float(audio_events.frame_seconds)
            i0 = max(0, int(round(t / dt)))
            i1 = min(audio_events.n_frames(), max(i0 + 1, int(round((t + clip_seconds) / dt))))
            m = float(np.mean(audio_events.music[i0:i1]))
            a = float(np.mean(audio_events.applause[i0:i1]))
            sp = float(np.mean(audio_events.speech[i0:i1]))
            score += 3.0 * m - 2.5 * a - 1.8 * sp

        if coarse_features is not None and coarse_features.n_frames > 0:
            i0 = coarse_features.index_of(float(t))
            i1 = max(i0 + 1, coarse_features.index_of(float(t) + clip_seconds))
            mean_rms = float(np.mean(coarse_features.rms_db[i0:i1]))
            # Penalize quiet/silent stretches (< -42 dBFS)
            if mean_rms < -42.0:
                score -= (-42.0 - mean_rms) * 0.25
            else:
                score += min(0.0, mean_rms + 18.0) * 0.02

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t


# ----------------------------------------------------------------------
# Consensus & conflict aggregation
# ----------------------------------------------------------------------

def aggregate_clip_recognitions(
    start_seconds: float,
    end_seconds: float,
    clips: Sequence[ClipRecognition],
) -> SegmentRecognition:
    """Combine multiple clip hits from AcoustID and/or ShazamIO into one verdict."""
    seg = SegmentRecognition(
        start_seconds=float(start_seconds),
        end_seconds=float(end_seconds),
        clips=list(clips),
    )
    if not clips:
        seg.status = STATUS_UNIDENTIFIED
        seg.confidence = 0.0
        return seg

    # Cluster clips that refer to the same song
    clusters: list[list[ClipRecognition]] = []
    for clip in clips:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            if same_song(rep.artist, rep.title, clip.artist, clip.title):
                cluster.append(clip)
                placed = True
                break
        if not placed:
            clusters.append([clip])

    # Detect intra-segment song transitions from time-ordered clips
    ordered = sorted(clips, key=lambda c: c.start_seconds)
    for i in range(len(ordered) - 1):
        a = ordered[i]
        b = ordered[i + 1]
        if b.start_seconds - a.start_seconds < 12.0:
            continue
        if not same_song(a.artist, a.title, b.artist, b.title):
            est_bound: Optional[float] = None
            if b.offset_seconds > 0.0:
                cand = b.start_seconds - b.offset_seconds
                if a.start_seconds + 3.0 <= cand <= b.start_seconds + 2.0:
                    est_bound = cand
            seg.song_changes.append(
                SongChange(
                    prev_clip_time=a.start_seconds,
                    next_clip_time=b.start_seconds,
                    prev_artist=a.artist,
                    prev_title=a.title,
                    next_artist=b.artist,
                    next_title=b.title,
                    estimated_boundary_seconds=est_bound,
                    engine=b.engine,
                )
            )

    # Check for cross-engine disagreement (AcoustID vs ShazamIO)
    ac_clips = [c for c in clips if c.engine == ACOUSTID]
    sh_clips = [c for c in clips if c.engine == SHAZAMIO]

    def _top_of(group: list[ClipRecognition]) -> Optional[ClipRecognition]:
        if not group:
            return None
        # Prefer the song with the most hits, breaking ties by score
        best_c: Optional[ClipRecognition] = None
        best_key = (-1, -1.0)
        for c in group:
            count = sum(1 for o in group if same_song(c.artist, c.title, o.artist, o.title))
            key = (count, c.score)
            if key > best_key:
                best_key = key
                best_c = c
        return best_c

    top_ac = _top_of(ac_clips)
    top_sh = _top_of(sh_clips)
    has_cross_engine_conflict = (
        top_ac is not None
        and top_sh is not None
        and not same_song(top_ac.artist, top_ac.title, top_sh.artist, top_sh.title)
    )

    # Pick winning cluster (most supporting clips, then multi-engine bonus, then score)
    def _cluster_rank(cl: list[ClipRecognition]) -> tuple[int, int, float]:
        engines_in = {c.engine for c in cl}
        return (len(engines_in), len(cl), max(c.score for c in cl))

    clusters.sort(key=_cluster_rank, reverse=True)
    winner = clusters[0]

    # Prefer ShazamIO or highest-score representative for clean artist/title display
    rep = max(winner, key=lambda c: (1 if c.artist else 0, c.score))
    seg.artist = rep.artist
    seg.title = rep.title
    seg.album = next((c.album for c in winner if c.album), "")
    seg.mb_recording_id = next((c.recording_id for c in winner if c.recording_id), "")
    seg.engines = sorted({c.engine for c in clips})

    winner_engines = sorted({c.engine for c in winner})
    max_score = max(c.score for c in winner)

    if has_cross_engine_conflict and top_ac is not None and top_sh is not None:
        seg.status = STATUS_CONFLICT
        seg.confidence = 0.45
        seg.conflict_detail = (
            f"Recognition conflict: AcoustID suggested '{top_ac.label}' ({top_ac.score:.2f}) "
            f"while ShazamIO suggested '{top_sh.label}' ({top_sh.score:.2f})."
        )
        seg.reasons.append(seg.conflict_detail)
    elif len(winner_engines) >= 2:
        seg.status = STATUS_CONFIRMED
        seg.confidence = round(min(0.96, max(0.88, max_score)), 3)
        seg.reasons.append(
            f"Confirmed by both {' + '.join(winner_engines)} ({len(winner)} consistent clip(s)): '{seg.label}'."
        )
    elif len(winner) >= 2:
        seg.status = STATUS_CONFIRMED
        seg.confidence = round(min(0.93, max(0.84, max_score)), 3)
        seg.reasons.append(
            f"Confirmed by {len(winner)} consistent {winner_engines[0]} clips: '{seg.label}'."
        )
    elif max_score >= 0.70:
        seg.status = STATUS_LIKELY
        seg.confidence = round(min(0.82, max_score), 3)
        seg.reasons.append(
            f"Likely match from {winner_engines[0]} (1 clip, score {max_score:.2f}): '{seg.label}'."
        )
    else:
        seg.status = STATUS_UNCERTAIN
        seg.confidence = round(max_score, 3)
        seg.reasons.append(
            f"Uncertain match from {winner_engines[0]} (score {max_score:.2f}): '{seg.label}'."
        )

    if seg.song_changes:
        for ch in seg.song_changes:
            seg.reasons.append(
                f"Song changed inside segment between {ch.prev_clip_time:.0f}s "
                f"('{format_artist_title(ch.prev_artist, ch.prev_title)}') and "
                f"{ch.next_clip_time:.0f}s ('{format_artist_title(ch.next_artist, ch.next_title)}')."
            )

    return seg


# ----------------------------------------------------------------------
# Multi-engine orchestrator
# ----------------------------------------------------------------------

class MultiEngineRecognizer:
    """Coordinates AcoustID, optional ShazamIO, and MusicBrainz enrichment."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_acoustid: bool = True,
        use_shazamio: Optional[bool] = None,
        enrich_musicbrainz: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.use_acoustid = bool(use_acoustid)
        self.enrich_musicbrainz = bool(enrich_musicbrainz)

        self.acoustid: Optional[AcoustIdProvider] = None
        if self.use_acoustid and chromaprint_backend():
            ac = AcoustIdProvider(cache_dir=cache_dir)
            if ac.configured:
                self.acoustid = ac

        self.shazamio: Optional[ShazamIoProvider] = None
        if use_shazamio is None:
            # Respect user's explicit opt-in toggle!
            sh = ShazamIoProvider(cache_dir=cache_dir, require_enabled=True)
            if sh.available:
                self.shazamio = sh
        elif use_shazamio:
            sh = ShazamIoProvider(cache_dir=cache_dir, require_enabled=False)
            if sh.available:
                self.shazamio = sh

        self._mb: Optional[MusicBrainzProvider] = None
        self._shazam_rate_limited = False

    @property
    def active_engines(self) -> list[str]:
        out: list[str] = []
        if self.acoustid is not None:
            out.append(ACOUSTID)
        if self.shazamio is not None and not self._shazam_rate_limited:
            out.append(SHAZAMIO)
        return out

    def identify_segment(
        self,
        audio_path: str,
        start_seconds: float,
        end_seconds: float,
        reader=None,
        sample_rate: int = 44100,
        coarse_features=None,
        audio_events=None,
        n_clips: Optional[int] = None,
        acoustid_offsets: Sequence[float] = (0.0, 8.0),
        cancel: Optional[Callable[[], bool]] = None,
    ) -> SegmentRecognition:
        """Identify one proposed track segment using all active recognition engines."""
        clips: list[ClipRecognition] = []

        # 1. AcoustID (whole-segment / start-offset Chromaprint fingerprinting)
        if self.acoustid is not None and audio_path and not (cancel and cancel()):
            try:
                ac_match: Optional[FingerprintMatch] = self.acoustid.identify_segment(
                    audio_path,
                    start_seconds,
                    end_seconds,
                    offsets=acoustid_offsets,
                )
            except ProviderError:
                ac_match = None
            if ac_match is not None:
                clips.append(
                    ClipRecognition(
                        engine=ACOUSTID,
                        start_seconds=start_seconds,
                        end_seconds=min(end_seconds, start_seconds + 120.0),
                        title=ac_match.title,
                        artist=ac_match.artist,
                        album=ac_match.release_title,
                        score=float(ac_match.score),
                        recording_id=ac_match.recording_id,
                    )
                )

        # 2. Optional ShazamIO (multi-point interior clips avoiding silence/applause/speech)
        if self.shazamio is not None and not self._shazam_rate_limited and not (cancel and cancel()):
            sample_times = choose_sample_timestamps(
                start_seconds,
                end_seconds,
                clip_seconds=10.0,
                n_clips=n_clips,
                coarse_features=coarse_features,
                audio_events=audio_events,
            )
            for t_s in sample_times:
                if cancel and cancel():
                    break
                try:
                    if reader is not None:
                        sh_match: Optional[ShazamMatch] = self.shazamio.recognize_reader_slice(
                            reader, sample_rate, t_s, duration_seconds=10.0
                        )
                    else:
                        sh_match = self.shazamio.recognize_file_slice(
                            audio_path, t_s, duration_seconds=10.0
                        )
                except ProviderError as exc:
                    if "429" in str(exc) or "rate" in str(exc).lower():
                        self._shazam_rate_limited = True
                    break

                if sh_match is not None:
                    clips.append(
                        ClipRecognition(
                            engine=SHAZAMIO,
                            start_seconds=t_s,
                            end_seconds=t_s + 10.0,
                            title=sh_match.title,
                            artist=sh_match.artist,
                            album=sh_match.album,
                            score=float(sh_match.score),
                            isrc=sh_match.isrc,
                            offset_seconds=float(sh_match.offset_seconds),
                            timeskew=float(sh_match.timeskew),
                        )
                    )

        seg = aggregate_clip_recognitions(start_seconds, end_seconds, clips)

        # 3. Optional MusicBrainz metadata enrichment (cached, non-fatal)
        if (
            self.enrich_musicbrainz
            and seg.status in (STATUS_CONFIRMED, STATUS_LIKELY)
            and seg.artist
            and seg.title
            and not seg.mb_recording_id
            and not (cancel and cancel())
        ):
            self._enrich_segment_musicbrainz(seg)

        return seg

    def identify_segments(
        self,
        audio_path: str,
        segments_seconds: Sequence[tuple[float, float]],
        reader=None,
        sample_rate: int = 44100,
        coarse_features=None,
        audio_events=None,
        n_clips: Optional[int] = None,
        progress: Optional[Callable[[float, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> list[SegmentRecognition]:
        """Identify a sequence of segments using all active engines."""
        results: list[SegmentRecognition] = []
        total = max(1, len(segments_seconds))
        eng_label = " + ".join(self.active_engines) or "audio"
        for idx, (s_s, e_s) in enumerate(segments_seconds):
            if cancel and cancel():
                break
            if progress:
                progress(idx / float(total), f"Identifying track {idx + 1}/{total} ({eng_label})...")
            res = self.identify_segment(
                audio_path,
                s_s,
                e_s,
                reader=reader,
                sample_rate=sample_rate,
                coarse_features=coarse_features,
                audio_events=audio_events,
                n_clips=n_clips,
                cancel=cancel,
            )
            results.append(res)
        if progress:
            progress(1.0, f"Song identification complete ({eng_label}).")
        return results

    def _enrich_segment_musicbrainz(self, seg: SegmentRecognition) -> None:
        """Query MusicBrainz for canonical recording duration/album (cached, survivable)."""
        try:
            if self._mb is None:
                self._mb = MusicBrainzProvider(cache_dir=self.cache_dir)
            query = f'recording:"{seg.title}" AND artist:"{seg.artist}"'
            data = self._mb.client.get_json(
                "https://musicbrainz.org/ws/2/recording",
                {"query": query, "fmt": "json", "limit": "3"},
            )
            recs = data.get("recordings") or []
            if not recs:
                return
            top = recs[0]
            seg.mb_recording_id = str(top.get("id") or "")
            length_ms = top.get("length")
            if length_ms:
                seg.mb_duration_seconds = float(length_ms) / 1000.0
            if not seg.album:
                releases = top.get("releases") or []
                if releases:
                    seg.album = str(releases[0].get("title") or "")
        except Exception:
            pass
