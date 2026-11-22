"""Automatic recording identification from filename, tags, and online metadata.

When a user drags a long recording into SplytNyt, this module figures out what
kind of recording it probably is (known album, live concert, DJ mix, or
Broadway/theatre), queries appropriate metadata sources (MusicBrainz,
AcoustID/Chromaprint, setlist.fm, LRCLIB, bundled theatrical references), and
returns an editable, reversible :class:`IdentificationResult` with selectable
release candidates.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..core.model import Worklist, WorklistItem, now_iso
from . import (
    MUSICBRAINZ,
    SETLISTFM,
    ProviderError,
    ReleaseCandidate,
    TrackCandidate,
)
from .acoustid import ACOUSTID, AcoustIdProvider, FingerprintWindow, chromaprint_backend
from .lrclib import LRCLIB, LrclibProvider, LyricEntry
from .musicbrainz import MusicBrainzProvider
from .references import TheaterReference, build_worklist, list_references
from .setlistfm import (
    SetlistFmProvider,
    country_name_to_code,
    has_api_key as has_setlistfm_key,
)

KIND_ALBUM = "album"
KIND_CONCERT = "concert"
KIND_DJ_MIX = "dj_mix"
KIND_BROADWAY = "broadway"
KIND_UNKNOWN = "unknown"

KIND_LABELS = {
    KIND_ALBUM: "Known album",
    KIND_CONCERT: "Live concert",
    KIND_DJ_MIX: "DJ mix / vinyl session",
    KIND_BROADWAY: "Broadway / theatre",
    KIND_UNKNOWN: "Recording",
}


@dataclass
class RecordingClues:
    """Clues extracted locally from file paths and embedded tags."""

    recording_kind: str = KIND_UNKNOWN
    artist: str = ""
    title: str = ""
    year: str = ""
    location: str = ""
    format_hint: str = ""
    act_number: Optional[int] = None
    raw_name: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class SelectableCandidate:
    """One release, concert, or theatrical reference the user can choose."""

    label: str
    provider: str
    candidate_id: str
    kind: str  # "mb_release" | "mb_event" | "setlistfm" | "theater_ref"
    artist: str = ""
    title: str = ""
    detail: str = ""
    year: str = ""
    format: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class IdentificationResult:
    """What SplytNyt thinks the recording is, with selectable alternatives."""

    recording_kind: str = KIND_UNKNOWN
    artist: str = ""
    title: str = ""
    subtitle: str = ""
    provider: str = ""
    confidence: float = 0.0
    clues: RecordingClues = field(default_factory=RecordingClues)
    candidates: list[SelectableCandidate] = field(default_factory=list)
    selected_index: int = 0
    worklist: Optional[Worklist] = None
    fingerprint_windows: list[FingerprintWindow] = field(default_factory=list)
    lyrics_by_title: dict[str, LyricEntry] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def selected_candidate(self) -> Optional[SelectableCandidate]:
        if 0 <= self.selected_index < len(self.candidates):
            return self.candidates[self.selected_index]
        return None


# ----------------------------------------------------------------------
# Local clue extraction (filename + embedded tags)
# ----------------------------------------------------------------------

_NOISE_TAGS_RE = re.compile(
    r"\b("
    r"hd|4k|8k|uhd|hq|hi[\s-]?res|high\s+audio\s+quality|audio\s+remaster(?:ed)?|"
    r"remaster(?:ed)?|official\s+audio|full\s+vinyl\s+album|full\s+album|"
    r"vinyl\s+album|full\s+show|full\s+concert|full\s+set|complete\s+show|"
    r"320\s*kbps|flac|lossless|vinyl\s+rip|lp\s+rip"
    r")\b",
    re.IGNORECASE,
)

_DJ_CUES_RE = re.compile(
    r"\b("
    r"dj\s*set|dj\s*mix|vinyl\s+sessions?|mix\s*tape|megamix|continuous\s+mix|"
    r"boiler\s+room|ft\.|feat\.|b2b|1\s+hora|2\s+horas?|pura\s+m[uú]sica|"
    r"radio\s+show|podcast|club\s+mix|house\s+mix|techno\s+mix"
    r")\b",
    re.IGNORECASE,
)

_CONCERT_CUES_RE = re.compile(
    r"\b("
    r"live|full\s+show|concert|in\s+concert|tour|festival|soundboard|bootleg|"
    r"unplugged|at\s+the\s+|stadium|arena|budokan|wembley|tokyo|japan"
    r")\b",
    re.IGNORECASE,
)

_ALBUM_CUES_RE = re.compile(
    r"\b(full\s+(?:vinyl\s+)?album|vinyl\s+album|full\s+lp|studio\s+album)\b",
    re.IGNORECASE,
)

_ACT_RE = re.compile(r"\bact[\s_-]*([12])\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")


def _read_embedded_tags(path: str) -> dict[str, str]:
    """Read basic artist/album/title/date tags with mutagen if available."""
    out: dict[str, str] = {}
    if not path or not os.path.exists(path):
        return out
    try:
        import mutagen

        audio = mutagen.File(path, easy=True)
        if audio and audio.tags:
            for key in ("artist", "albumartist", "album", "title", "date"):
                val = audio.tags.get(key)
                if val:
                    text = str(val[0] if isinstance(val, list) else val).strip()
                    if text:
                        out[key] = text
    except Exception:
        pass
    return out


def _clean_segment(text: str) -> str:
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
    text = _NOISE_TAGS_RE.sub(" ", text)
    text = re.sub(r"[&|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -_.,:")
    return text


def parse_recording_clues(
    paths: Sequence[str],
    total_duration_seconds: float = 0.0,
) -> RecordingClues:
    """Infer recording kind, artist, title, format, and act from file paths & tags."""
    clues = RecordingClues()
    if not paths:
        return clues

    first = paths[0]
    stem = os.path.splitext(os.path.basename(first))[0]
    parent = os.path.basename(os.path.dirname(os.path.abspath(first)))
    clues.raw_name = stem

    tags = _read_embedded_tags(first)
    tag_artist = tags.get("albumartist") or tags.get("artist") or ""
    tag_album = tags.get("album") or ""
    tag_title = tags.get("title") or ""
    tag_date = tags.get("date") or ""

    # Format hints
    lower = stem.lower()
    if any(w in lower for w in ("vinyl", "lp", "12\"")):
        clues.format_hint = "vinyl"
    elif "cd" in re.findall(r"[a-z0-9]+", lower):
        clues.format_hint = "cd"

    # Year
    m_year = _YEAR_RE.search(stem) or _YEAR_RE.search(tag_date)
    if m_year:
        clues.year = m_year.group(1)

    # Act number (Broadway / musical theatre)
    m_act = _ACT_RE.search(stem)
    if m_act:
        clues.act_number = int(m_act.group(1))

    # Classify recording kind
    is_album_explicit = bool(_ALBUM_CUES_RE.search(stem))
    is_dj = bool(_DJ_CUES_RE.search(stem)) and not is_album_explicit
    is_concert = bool(_CONCERT_CUES_RE.search(stem)) and not is_album_explicit and not is_dj
    is_broadway = (
        clues.act_number is not None
        or any(w in lower for w in ("broadway", "musical", "cast recording", "moulin rouge"))
        or any(w in parent.lower() for w in ("broadway", "musical", "theatre", "theater"))
    )

    if is_broadway:
        clues.recording_kind = KIND_BROADWAY
    elif is_dj:
        clues.recording_kind = KIND_DJ_MIX
    elif is_concert:
        clues.recording_kind = KIND_CONCERT
    elif is_album_explicit:
        clues.recording_kind = KIND_ALBUM

    # Parse "Artist - Title - ..." structure from filename
    parts = [p.strip() for p in re.split(r"\s+-\s+", stem) if p.strip()]
    if tag_artist:
        clues.artist = tag_artist
    elif len(parts) >= 2 and not is_dj and not is_broadway:
        clues.artist = _clean_segment(parts[0])

    if tag_album:
        clues.title = tag_album
    elif tag_title and not is_broadway:
        clues.title = tag_title
    elif len(parts) >= 2 and not is_dj:
        clues.title = _clean_segment(parts[1])
    elif not is_broadway:
        clues.title = _clean_segment(stem)

    if clues.recording_kind == KIND_CONCERT:
        # Extract location/venue words from the non-artist part (e.g. "Japan Live 2025")
        rest = " ".join(parts[1:]) if len(parts) >= 2 else stem
        loc = _NOISE_TAGS_RE.sub(" ", rest)
        loc = re.sub(r"\b(live|19\d\d|20\d\d|full|show|concert|audio|remaster)\b", " ", loc, flags=re.IGNORECASE)
        loc = re.sub(r"[&|()\[\]]+", " ", loc)
        loc = re.sub(r"\s+", " ", loc).strip(" -_.,:")
        clues.location = loc
        if not clues.title:
            clues.title = f"{clues.artist} {loc} {clues.year}".strip()

    if clues.recording_kind == KIND_UNKNOWN and clues.artist and clues.title:
        clues.recording_kind = KIND_ALBUM

    return clues


# ----------------------------------------------------------------------
# Main identification orchestrator
# ----------------------------------------------------------------------

def identify_recording(
    paths: Sequence[str],
    total_duration_seconds: float = 0.0,
    allow_network: bool = True,
    allow_fingerprint: bool = True,
    fetch_lyrics: bool = True,
    cache_dir: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> IdentificationResult:
    """Identify a recording and fetch candidate tracklists / metadata."""
    clues = parse_recording_clues(paths, total_duration_seconds)
    result = IdentificationResult(
        recording_kind=clues.recording_kind,
        artist=clues.artist,
        title=clues.title or clues.raw_name,
        clues=clues,
    )

    if clues.recording_kind == KIND_ALBUM:
        _identify_album(result, total_duration_seconds, allow_network, cache_dir, progress)
    elif clues.recording_kind == KIND_CONCERT:
        _identify_concert(result, total_duration_seconds, allow_network, fetch_lyrics, cache_dir, progress)
    elif clues.recording_kind == KIND_DJ_MIX:
        _identify_dj_mix(result, paths, total_duration_seconds, allow_network and allow_fingerprint, cache_dir, progress, cancel)
    elif clues.recording_kind == KIND_BROADWAY:
        _identify_broadway(result, paths, allow_network, fetch_lyrics, cache_dir, progress)
    else:
        # Unknown recording: try AcoustID on a sample if configured, else MusicBrainz if clues exist.
        if allow_network and allow_fingerprint and paths:
            _try_acoustid_fallback(result, paths[0], total_duration_seconds, cache_dir)
        if result.worklist is None and allow_network and clues.artist:
            _identify_album(result, total_duration_seconds, allow_network, cache_dir, progress)

    return result


def load_candidate_worklist(
    candidate: SelectableCandidate,
    cache_dir: Optional[str] = None,
) -> Optional[Worklist]:
    """Fetch and build a Worklist for a user-selected candidate release/reference."""
    if candidate.kind == "mb_release":
        mb = MusicBrainzProvider(cache_dir=cache_dir)
        tracks = mb.release_tracks(candidate.candidate_id)
        return _worklist_from_tracks(
            name=f"{candidate.artist} - {candidate.title}" if candidate.artist else candidate.title,
            tracks=tracks,
            provider=MUSICBRAINZ,
            provider_id=candidate.candidate_id,
            mode="album",
            notes=f"MusicBrainz release: {candidate.detail}",
        )
    if candidate.kind == "mb_event":
        mb = MusicBrainzProvider(cache_dir=cache_dir)
        tracks = mb.event_tracks(candidate.candidate_id)
        return _worklist_from_tracks(
            name=candidate.title or candidate.artist,
            tracks=tracks,
            provider=MUSICBRAINZ,
            provider_id=candidate.candidate_id,
            mode="concert",
            notes=f"MusicBrainz event setlist: {candidate.detail}",
        )
    if candidate.kind == "setlistfm":
        provider = SetlistFmProvider(cache_dir=cache_dir)
        tracks = provider.concert_tracks(candidate.candidate_id)
        return _worklist_from_tracks(
            name=candidate.label,
            tracks=tracks,
            provider=SETLISTFM,
            provider_id=candidate.candidate_id,
            mode="concert",
            notes=f"setlist.fm concert: {candidate.detail}",
        )
    if candidate.kind == "theater_ref":
        ref_id = candidate.extra.get("reference_id", "")
        version_id = candidate.extra.get("version_id", "")
        act = candidate.extra.get("act")
        for ref in list_references():
            if ref.reference_id == ref_id:
                return build_worklist(ref, version_id=version_id, act=act)
    return None


# ----------------------------------------------------------------------
# Per-kind identification helpers
# ----------------------------------------------------------------------

def _identify_album(
    result: IdentificationResult,
    total_duration_seconds: float,
    allow_network: bool,
    cache_dir: Optional[str],
    progress: Optional[Callable[[float, str], None]],
) -> None:
    if not allow_network:
        result.notes.append("Network disabled; skipped MusicBrainz album lookup.")
        return
    if not (result.clues.artist or result.clues.title):
        result.notes.append("No artist or album title found in filename or tags.")
        return

    mb = MusicBrainzProvider(cache_dir=cache_dir)
    if progress:
        progress(0.15, f"Searching MusicBrainz for {result.clues.artist} - {result.clues.title}...")
    try:
        groups = mb.search_release_groups_ranked(
            result.clues.artist,
            result.clues.title,
            format_hint=result.clues.format_hint,
            limit=10,
        )
    except ProviderError as exc:
        result.notes.append(f"MusicBrainz search failed: {exc}")
        return

    if not groups:
        result.notes.append("No matching album found on MusicBrainz.")
        return

    top_group = groups[0]
    result.artist = top_group.artist or result.clues.artist
    result.title = top_group.title or result.clues.title

    if progress:
        progress(0.40, f"Loading editions of {result.title}...")
    try:
        releases = mb.releases_of(top_group.provider_id)
    except ProviderError as exc:
        result.notes.append(f"MusicBrainz editions lookup failed: {exc}")
        return

    ranked = mb.rank_releases(
        releases,
        format_hint=result.clues.format_hint,
        year_hint=result.clues.year,
    )
    if not ranked:
        result.notes.append("Release group had no editions on MusicBrainz.")
        return

    # Deduplicate editions that look identical by (year, format, country, track_count)
    # while keeping up to 10 distinct choices for the user.
    seen_keys: set[tuple] = set()
    distinct: list[ReleaseCandidate] = []
    for rel in ranked:
        key = (rel.year, rel.format, rel.country, rel.track_count, rel.label)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        distinct.append(rel)
        if len(distinct) >= 10:
            break

    # Fetch tracklist for the top 1-3 editions to pick the one whose total
    # duration best matches the recording (handling vinyl vs CD bonus tracks).
    best_idx = 0
    best_tracks: list[TrackCandidate] = []
    best_dur_err = float("inf")
    probe_count = min(3, len(distinct))
    for idx in range(probe_count):
        rel = distinct[idx]
        if progress:
            progress(0.55 + 0.30 * (idx / max(1, probe_count)), f"Checking edition {idx + 1}/{probe_count}...")
        try:
            tracks = mb.release_tracks(rel.provider_id)
        except ProviderError:
            continue
        if not tracks:
            continue
        total_mb = sum(t.duration_seconds for t in tracks)
        if total_duration_seconds > 0 and total_mb > 0:
            rel_err = abs(total_mb - total_duration_seconds) / total_duration_seconds
        else:
            rel_err = 0.0
        # Slight preference for higher-ranked format match when durations are close.
        weighted_err = rel_err + 0.015 * idx
        if weighted_err < best_dur_err:
            best_dur_err = weighted_err
            best_idx = idx
            best_tracks = tracks

    for rel in distinct:
        detail = rel.detail or f"{rel.year} {rel.format} ({rel.track_count} tracks)".strip()
        result.candidates.append(
            SelectableCandidate(
                label=f"{rel.title or result.title} ({detail}, {rel.track_count} tracks)",
                provider=MUSICBRAINZ,
                candidate_id=rel.provider_id,
                kind="mb_release",
                artist=rel.artist or result.artist,
                title=rel.title or result.title,
                detail=detail,
                year=rel.year,
                format=rel.format,
            )
        )

    result.selected_index = best_idx
    chosen = result.candidates[best_idx]
    result.provider = MUSICBRAINZ
    result.subtitle = chosen.detail
    result.confidence = 0.85 if best_dur_err < 0.08 else 0.65
    if best_tracks:
        result.worklist = _worklist_from_tracks(
            name=f"{result.artist} - {result.title}",
            tracks=best_tracks,
            provider=MUSICBRAINZ,
            provider_id=chosen.candidate_id,
            mode="album",
            notes=f"MusicBrainz edition: {chosen.detail}",
        )
        total_mb = sum(t.duration_seconds for t in best_tracks)
        result.notes.append(
            f"Matched MusicBrainz edition '{chosen.detail}' ({len(best_tracks)} tracks, "
            f"published total {total_mb / 60:.1f} min vs recording {total_duration_seconds / 60:.1f} min)."
        )


def _identify_concert(
    result: IdentificationResult,
    total_duration_seconds: float,
    allow_network: bool,
    fetch_lyrics: bool,
    cache_dir: Optional[str],
    progress: Optional[Callable[[float, str], None]],
) -> None:
    if not allow_network:
        result.notes.append("Network disabled; skipped concert setlist lookup.")
        return

    artist = result.clues.artist
    year = result.clues.year
    location = result.clues.location
    if not artist:
        result.notes.append("No artist name found in filename or tags for concert lookup.")
        return

    # 1. Try setlist.fm if API key is configured.
    if has_setlistfm_key():
        if progress:
            progress(0.25, f"Searching setlist.fm for {artist}...")
        try:
            sfm = SetlistFmProvider(cache_dir=cache_dir)
            cc = country_name_to_code(location)
            city_arg = "" if cc else location
            concerts = sfm.search_concerts(
                artist=artist,
                city=city_arg,
                country_code=cc,
                year=year,
                limit=20,
            )
            if not concerts and (year or cc or city_arg):
                # Retry without year/location constraint if exact filter returned nothing
                concerts = sfm.search_concerts(artist=artist, country_code=cc, limit=15)

            # Rank concerts: prefer exact artist match and full concerts with real song lists
            # (avoids empty stubs or short festival/cover acts sharing the artist name).
            want_artist = artist.strip().lower()
            def _concert_rank(c):
                exact_art = 1 if c.artist.strip().lower() == want_artist else 0
                has_full_set = 1 if c.song_count >= 10 else 0
                return (exact_art, has_full_set, c.song_count)

            concerts = [c for c in concerts if c.song_count > 0]
            concerts.sort(key=_concert_rank, reverse=True)

            for c in concerts[:8]:
                result.candidates.append(
                    SelectableCandidate(
                        label=f"{c.artist} - {c.venue} ({c.date}, {c.city}) [{c.song_count} songs]",
                        provider=SETLISTFM,
                        candidate_id=c.provider_id,
                        kind="setlistfm",
                        artist=c.artist,
                        title=c.venue,
                        detail=c.detail,
                        year=c.date[-4:] if len(c.date) >= 4 else year,
                    )
                )
            if concerts:
                tracks = sfm.concert_tracks(concerts[0].provider_id)
                if tracks:
                    result.provider = SETLISTFM
                    result.subtitle = concerts[0].detail
                    result.confidence = 0.85 if concerts[0].song_count >= 10 else 0.65
                    result.worklist = _worklist_from_tracks(
                        name=f"{artist} - {concerts[0].venue} ({concerts[0].date})",
                        tracks=tracks,
                        provider=SETLISTFM,
                        provider_id=concerts[0].provider_id,
                        mode="concert",
                        notes=f"setlist.fm setlist: {concerts[0].detail}",
                    )
                    result.notes.append(
                        f"Loaded setlist from setlist.fm: {concerts[0].venue} ({concerts[0].date}, {len(tracks)} songs)."
                    )
        except ProviderError as exc:
            result.notes.append(f"setlist.fm lookup skipped: {exc}")
    else:
        result.notes.append("setlist.fm key is not configured; checking MusicBrainz events and artist catalog.")

    # 2. Try MusicBrainz events (no API key needed).
    mb = MusicBrainzProvider(cache_dir=cache_dir)
    if result.worklist is None:
        if progress:
            progress(0.45, f"Searching MusicBrainz events for {artist}...")
        try:
            events = mb.search_events(artist=artist, year=year, location=location, limit=10)
            if not events and (year or location):
                events = mb.search_events(artist=artist, year="", location=location, limit=10)
            for ev in events[:8]:
                if ev.song_count <= 0:
                    continue
                result.candidates.append(
                    SelectableCandidate(
                        label=f"{ev.title} ({ev.detail})",
                        provider=MUSICBRAINZ,
                        candidate_id=ev.provider_id,
                        kind="mb_event",
                        artist=artist,
                        title=ev.title,
                        detail=ev.detail,
                        year=ev.date[:4],
                    )
                )
            if result.candidates and result.worklist is None:
                best = result.candidates[0]
                tracks = mb.event_tracks(best.candidate_id)
                if tracks:
                    result.provider = MUSICBRAINZ
                    result.subtitle = best.detail
                    result.confidence = 0.72
                    result.worklist = _worklist_from_tracks(
                        name=f"{artist} - {best.title}",
                        tracks=tracks,
                        provider=MUSICBRAINZ,
                        provider_id=best.candidate_id,
                        mode="concert",
                        notes=f"MusicBrainz event setlist: {best.detail}",
                    )
                    result.notes.append(f"Loaded concert setlist from MusicBrainz ({len(tracks)} songs).")
        except ProviderError as exc:
            result.notes.append(f"MusicBrainz event search unavailable: {exc}")

    # 3. Supporting catalog: fetch artist's official recordings from MusicBrainz
    # so even without an exact concert setlist, we know typical song durations & titles.
    if result.worklist is None:
        if progress:
            progress(0.65, f"Fetching {artist} recording catalog from MusicBrainz...")
        try:
            catalog = mb.search_artist_recordings(artist, limit=40)
            if catalog:
                result.provider = MUSICBRAINZ
                result.subtitle = f"{artist} recording catalog (no exact setlist found)"
                result.confidence = 0.35
                result.worklist = _worklist_from_tracks(
                    name=f"{artist} (catalog reference)",
                    tracks=catalog,
                    provider=MUSICBRAINZ,
                    provider_id=f"catalog:{artist}",
                    mode="concert_catalog",
                    notes="Supporting artist catalog only (not an exact concert setlist).",
                )
                result.notes.append(
                    f"No exact setlist found for {artist} {location} {year}; "
                    f"fetched {len(catalog)} artist recordings from MusicBrainz as supporting catalog."
                )
        except ProviderError as exc:
            result.notes.append(f"MusicBrainz catalog lookup failed: {exc}")

    # Optionally fetch lyrics (and typical track durations!) for exact setlists.
    if fetch_lyrics and result.worklist and result.worklist.mode == "concert" and len(result.worklist.items) <= 30:
        if progress:
            progress(0.85, "Fetching reference lyrics & song durations from LRCLIB...")
        lrclib = LrclibProvider(cache_dir=cache_dir)
        titles = [item.title for item in result.worklist.items]
        result.lyrics_by_title = lrclib.fetch_for_titles(titles, artist=artist, max_items=30)
        # Enrich missing setlist.fm track durations from LRCLIB so alignment has real length priors.
        for item in result.worklist.items:
            if item.duration_hint <= 0.0:
                entry = result.lyrics_by_title.get(item.title)
                if entry is not None and entry.duration_seconds > 0.0:
                    item.duration_hint = float(entry.duration_seconds)
                elif "tape" in (item.notes or "").lower():
                    item.duration_hint = 45.0


def _identify_dj_mix(
    result: IdentificationResult,
    paths: Sequence[str],
    total_duration_seconds: float,
    allow_fingerprint: bool,
    cache_dir: Optional[str],
    progress: Optional[Callable[[float, str], None]],
    cancel: Optional[Callable[[], bool]],
) -> None:
    result.subtitle = "Continuous DJ / vinyl mix"
    backend = chromaprint_backend()
    if not allow_fingerprint:
        result.notes.append("Fingerprinting disabled; detecting crossfades from audio only.")
        return
    from .shazamio_engine import SHAZAMIO, is_ready as shazamio_ready

    sh_ready = shazamio_ready()
    engines_ready: list[str] = []
    if backend and AcoustIdProvider(cache_dir=cache_dir).configured:
        engines_ready.append(ACOUSTID)
    if sh_ready:
        engines_ready.append(SHAZAMIO)

    if not engines_ready:
        result.notes.append(
            "No acoustic recognition engine is ready; crossfade boundaries will be "
            "detected from audio without inventing song names."
        )
        return

    # Multi-engine recognition runs inside split_dj_mix_with_metadata after
    # crossfade transitions are localized so each lookup samples clean interior clips.
    result.provider = " + ".join(engines_ready)
    result.notes.append(
        f"Recognition ready ({result.provider}): will sample detected segments after crossfade analysis."
    )


def _identify_broadway(
    result: IdentificationResult,
    paths: Sequence[str],
    allow_network: bool,
    fetch_lyrics: bool,
    cache_dir: Optional[str],
    progress: Optional[Callable[[float, str], None]],
) -> None:
    act = result.clues.act_number
    refs = list_references()
    raw_context = " ".join(
        [result.clues.raw_name, os.path.basename(os.path.dirname(os.path.abspath(paths[0]))) if paths else ""]
    ).lower()

    best_ref: Optional[TheaterReference] = None
    best_score = -1
    for ref in refs:
        score = 0
        title_words = [w for w in re.findall(r"[a-z]+", ref.title.lower()) if len(w) >= 3]
        initials = "".join(w[0] for w in title_words[:2])
        if any(w in raw_context for w in title_words):
            score += 3
        if initials and re.search(rf"\b{re.escape(initials)}\b", raw_context):
            score += 2
        if score > best_score:
            best_score = score
            best_ref = ref

        # Add all versions as selectable candidates
        for version in ref.versions or [None]:
            ver_id = version.version_id if version else ""
            ver_label = version.label if version else "Default"
            act_suffix = f" - Act {act}" if act else ""
            result.candidates.append(
                SelectableCandidate(
                    label=f"{ref.title} — {ver_label}{act_suffix}",
                    provider="theatre-reference",
                    candidate_id=f"{ref.reference_id}:{ver_id}:{act or 'all'}",
                    kind="theater_ref",
                    artist=ref.producer or "Original Broadway Cast",
                    title=ref.title,
                    detail=f"{ver_label}{act_suffix}",
                    extra={"reference_id": ref.reference_id, "version_id": ver_id, "act": act},
                )
            )

    if best_ref is not None:
        version_id = best_ref.versions[0].version_id if best_ref.versions else ""
        result.title = best_ref.title
        result.artist = best_ref.producer or "Broadway Cast"
        result.subtitle = f"Act {act}" if act else "Full show"
        result.provider = "theatre-reference"
        result.confidence = 0.80 if best_score >= 2 else 0.50
        result.worklist = build_worklist(best_ref, version_id=version_id, act=act)
        result.notes.append(
            f"Loaded cast reference '{result.worklist.name}' ({len(result.worklist.items)} numbers). "
            "Cast album durations are never treated as live stage boundaries."
        )

        if allow_network and fetch_lyrics and result.worklist.items:
            if progress:
                progress(0.65, f"Fetching reference lyrics for {best_ref.title} from LRCLIB...")
            lrclib = LrclibProvider(cache_dir=cache_dir)
            titles = [item.title for item in result.worklist.items]
            result.lyrics_by_title = lrclib.fetch_for_titles(
                titles,
                album=best_ref.title,
                max_items=25,
            )
            if result.lyrics_by_title:
                result.notes.append(
                    f"Fetched reference lyrics for {len(result.lyrics_by_title)}/{len(titles)} "
                    f"numbers from {LRCLIB}."
                )


def _try_acoustid_fallback(
    result: IdentificationResult,
    audio_path: str,
    total_duration_seconds: float,
    cache_dir: Optional[str],
) -> None:
    provider = AcoustIdProvider(cache_dir=cache_dir)
    if not provider.configured or not chromaprint_backend():
        return
    windows = provider.scan_recording(
        audio_path,
        total_duration_seconds,
        window_seconds=25.0,
        step_seconds=max(120.0, total_duration_seconds / 6.0),
        max_windows=6,
    )
    result.fingerprint_windows = windows
    hits = [w.match for w in windows if w.match is not None]
    if hits:
        result.artist = hits[0].artist
        result.title = hits[0].release_title or hits[0].title
        result.provider = ACOUSTID
        result.notes.append(f"AcoustID fingerprint suggested: {result.artist} - {result.title}.")


def _worklist_from_tracks(
    name: str,
    tracks: Sequence[TrackCandidate],
    provider: str,
    provider_id: str,
    mode: str,
    notes: str = "",
) -> Worklist:
    items: list[WorklistItem] = []
    for idx, t in enumerate(tracks, start=1):
        note_parts = []
        if t.side:
            note_parts.append(f"Side {t.side}")
        if t.is_encore:
            note_parts.append("Encore")
        if t.notes:
            note_parts.append(t.notes)
        items.append(
            WorklistItem(
                title=t.title,
                duration_hint=float(t.duration_seconds or 0.0),
                notes=" - ".join(note_parts),
                position=t.position or idx,
                set_number=t.set_number or (f"Side {t.side}" if t.side else ""),
                provider=provider,
                provider_id=provider_id,
            )
        )
    return Worklist(
        name=name,
        items=items,
        provider=provider,
        provider_id=provider_id,
        fetched_at=now_iso(),
        mode=mode,
        notes=notes,
    )
