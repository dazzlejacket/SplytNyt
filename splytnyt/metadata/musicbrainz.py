"""MusicBrainz: album lookups.

MusicBrainz describes releases: a release group (the work) with one or more
releases (the editions - a 1974 LP, a 1994 CD, a 2005 remaster). The user
picks the edition they actually hold, because the track order and the
lengths differ between them.

The service asks for a real User-Agent and for no more than one request a
second; both are handled by the shared client.

MusicBrainz's own documentation asks that applications identify themselves
and give a way to reach the people behind them. Set ``SPLYTNYT_CONTACT`` in
the environment to add your own contact details to the User-Agent.
"""

from __future__ import annotations

import re
from typing import Optional

from . import (
    MUSICBRAINZ,
    PerformanceCandidate,
    ProviderNotFound,
    ReleaseCandidate,
    TrackCandidate,
)
from .base import HttpClient, default_cache_dir

API_ROOT = "https://musicbrainz.org/ws/2"

#: MusicBrainz asks for at most one request per second.
MINIMUM_INTERVAL = 1.0


class MusicBrainzProvider:
    """Looks up albums and their track lists."""

    name = MUSICBRAINZ

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.client = HttpClient(
            MUSICBRAINZ,
            cache_dir=cache_dir if cache_dir is not None else default_cache_dir(),
            min_interval=MINIMUM_INTERVAL,
        )

    def test_connection(self) -> tuple[bool, str]:
        """Verify that MusicBrainz responds cleanly (no API key required)."""
        try:
            data = self.client.get_json(
                f"{API_ROOT}/artist/",
                params={"query": 'artist:"Pink Floyd"', "fmt": "json", "limit": 1},
                use_cache=False,
            )
        except Exception as exc:
            return False, str(exc)
        if isinstance(data, dict) and "artists" in data:
            return True, "Connected to MusicBrainz (free, no key needed)."
        return False, "MusicBrainz returned an unexpected response."

    # ------------------------------------------------------------------
    def search_albums(self, artist: str, title: str = "", limit: int = 25) -> list[ReleaseCandidate]:
        """Find release groups by artist and title."""
        query_parts = []
        if artist.strip():
            query_parts.append(f'artist:"{artist.strip()}"')
        if title.strip():
            query_parts.append(f'releasegroup:"{title.strip()}"')
        if not query_parts:
            raise ProviderNotFound("Type an artist or an album title to search for.")
        query = " AND ".join(query_parts)

        data = self.client.get_json(
            f"{API_ROOT}/release-group/",
            params={"query": query, "fmt": "json", "limit": limit},
        )
        groups = data.get("release-groups") or []
        results: list[ReleaseCandidate] = []
        for group in groups:
            artist_credit = _artist_credit(group.get("artist-credit"))
            results.append(
                ReleaseCandidate(
                    provider=MUSICBRAINZ,
                    provider_id=group.get("id", ""),
                    title=group.get("title", ""),
                    artist=artist_credit,
                    detail=_first_release_detail(group),
                    year=str(group.get("first-release-date", "") or "")[:4],
                    track_count=int(group.get("release-group", {}).get("primary-type-count", 0) or 0),
                )
            )
        return results

    # ------------------------------------------------------------------
    def releases_of(self, release_group_id: str) -> list[ReleaseCandidate]:
        """The editions of a release group, so the user can pick the right one."""
        data = self.client.get_json(
            f"{API_ROOT}/release-group/{release_group_id}",
            params={"inc": "releases+media", "fmt": "json"},
        )
        results: list[ReleaseCandidate] = []
        for release in data.get("releases") or []:
            media = release.get("media") or []
            discs = len(media)
            tracks = sum(int(item.get("track-count", 0) or 0) for item in media)
            formats = _format_names(release)
            format_text = ", ".join(formats)
            results.append(
                ReleaseCandidate(
                    provider=MUSICBRAINZ,
                    provider_id=release.get("id", ""),
                    title=release.get("title", ""),
                    artist=_artist_credit(release.get("artist-credit")),
                    detail=", ".join(
                        part
                        for part in (
                            str(release.get("date", "") or "")[:4],
                            format_text,
                            release.get("country", "") or "",
                            _label_name(release),
                        )
                        if part
                    ),
                    year=str(release.get("date", "") or "")[:4],
                    format=format_text,
                    country=release.get("country", "") or "",
                    label=_label_name(release),
                    discs=discs,
                    track_count=tracks,
                )
            )
        return results

    # ------------------------------------------------------------------
    def release_tracks(self, release_id: str) -> list[TrackCandidate]:
        """The track list of one exact release."""
        data = self.client.get_json(
            f"{API_ROOT}/release/{release_id}",
            params={"inc": "recordings+media+artist-credits", "fmt": "json"},
        )
        tracks: list[TrackCandidate] = []
        position = 0
        for disc_number, medium in enumerate(data.get("media") or [], start=1):
            medium_tracks = medium.get("tracks") or []
            is_vinyl = "vinyl" in str(medium.get("format", "")).lower()
            split = len(medium_tracks) // 2
            for index, track in enumerate(medium_tracks):
                position += 1
                length = track.get("length") or (track.get("recording") or {}).get("length") or 0
                side = _track_side(track.get("number", ""), is_vinyl, index, split)
                tracks.append(
                    TrackCandidate(
                        title=track.get("title", ""),
                        duration_seconds=float(length or 0) / 1000.0,
                        position=position,
                        disc=disc_number,
                        side=side,
                        notes=_track_notes(track),
                    )
                )
        return tracks

    # ------------------------------------------------------------------
    def search_release_groups_ranked(
        self,
        artist: str,
        title: str = "",
        format_hint: str = "",
        limit: int = 15,
    ) -> list[ReleaseCandidate]:
        """Search release groups and rank canonical studio/official editions first."""
        query_parts = []
        if artist.strip():
            query_parts.append(f'artist:"{artist.strip()}"')
        if title.strip():
            query_parts.append(f'releasegroup:"{title.strip()}"')
        if not query_parts:
            raise ProviderNotFound("Type an artist or an album title to search for.")
        query = " AND ".join(query_parts)

        data = self.client.get_json(
            f"{API_ROOT}/release-group/",
            params={"query": query, "fmt": "json", "limit": limit},
        )
        groups = data.get("release-groups") or []
        norm_title = _norm_text(title)
        norm_artist = _norm_text(artist)

        scored: list[tuple[float, ReleaseCandidate]] = []
        for group in groups:
            artist_credit = _artist_credit(group.get("artist-credit"))
            g_title = group.get("title", "") or ""
            releases = group.get("releases") or []
            edition_count = int(group.get("count") or len(releases) or 0)
            primary_type = (group.get("primary-type") or "").lower()
            secondary_types = [s.lower() for s in (group.get("secondary-types") or [])]

            score = float(group.get("score", 50) or 50) / 100.0
            if primary_type == "album":
                score += 0.35
            if any(t in ("bootleg", "demo", "remix", "interview") for t in secondary_types):
                score -= 0.45
            lower_title = g_title.lower()
            if any(tag in lower_title for tag in ("[restored]", "(demo", "demos)", "karaoke", "tribute")):
                score -= 0.55
            # Strongly favor canonical release groups with many editions.
            score += min(0.85, 0.15 * (edition_count ** 0.5))

            nt = _norm_text(g_title)
            if norm_title:
                if nt == norm_title or nt.removeprefix("the ") == norm_title.removeprefix("the "):
                    score += 0.50
                elif norm_title in nt or nt in norm_title:
                    score += 0.25
            if norm_artist and norm_artist in _norm_text(artist_credit):
                score += 0.20

            candidate = ReleaseCandidate(
                provider=MUSICBRAINZ,
                provider_id=group.get("id", ""),
                title=g_title,
                artist=artist_credit,
                detail=_first_release_detail(group),
                year=str(group.get("first-release-date", "") or "")[:4],
                track_count=edition_count,
            )
            scored.append((score, candidate))

        scored.sort(key=lambda pair: -pair[0])
        return [candidate for _, candidate in scored]

    # ------------------------------------------------------------------
    def rank_releases(
        self,
        releases: list[ReleaseCandidate],
        format_hint: str = "",
        year_hint: str = "",
    ) -> list[ReleaseCandidate]:
        """Order editions of a release group to put the most likely edition first."""
        hint = (format_hint or "").lower()
        want_vinyl = any(w in hint for w in ("vinyl", "lp", "12\"", "record"))
        want_cd = "cd" in hint

        def _score(rel: ReleaseCandidate) -> tuple[float, int]:
            score = 0.0
            fmt = (rel.format or "").lower()
            if want_vinyl:
                if "vinyl" in fmt or "12\"" in fmt:
                    score += 2.0
                elif "cd" in fmt or "digital" in fmt:
                    score -= 0.5
            elif want_cd:
                if "cd" in fmt:
                    score += 1.5
            if rel.discs == 1:
                score += 0.4
            if rel.track_count >= 4:
                score += 0.3
            if year_hint and rel.year == year_hint:
                score += 0.6
            # Earlier original pressings are usually cleaner canonical tracklists.
            try:
                yr = int(rel.year) if rel.year else 9999
            except ValueError:
                yr = 9999
            return (score, -yr)

        return sorted(releases, key=_score, reverse=True)

    # ------------------------------------------------------------------
    def search_events(
        self,
        artist: str = "",
        year: str = "",
        location: str = "",
        limit: int = 15,
    ) -> list[PerformanceCandidate]:
        """Search MusicBrainz events (concerts) that may carry setlists."""
        terms = []
        if artist.strip():
            terms.append(f'artist:"{artist.strip()}"')
        if location.strip():
            terms.append(f'place:"{location.strip()}" OR area:"{location.strip()}" OR name:"{location.strip()}"')
        if year.strip():
            terms.append(f'begin:{year.strip()}*')
        if not terms:
            return []
        query = " AND ".join(f"({t})" for t in terms)
        data = self.client.get_json(
            f"{API_ROOT}/event/",
            params={"query": query, "fmt": "json", "limit": limit},
        )
        results: list[PerformanceCandidate] = []
        for ev in data.get("events") or []:
            setlist_raw = ev.get("setlist") or ""
            parsed = parse_event_setlist(setlist_raw)
            span = ev.get("life-span") or {}
            date = str(span.get("begin") or "")
            results.append(
                PerformanceCandidate(
                    provider=MUSICBRAINZ,
                    provider_id=ev.get("id", ""),
                    artist=artist.strip(),
                    title=ev.get("name", "") or "",
                    date=date,
                    venue=ev.get("name", "") or "",
                    detail=", ".join(part for part in (date, ev.get("type", ""), f"{len(parsed)} songs" if parsed else "") if part),
                    song_count=len(parsed),
                )
            )
        results.sort(key=lambda item: (item.song_count > 0, item.song_count, item.date), reverse=True)
        return results

    def event_tracks(self, event_id: str) -> list[TrackCandidate]:
        """Fetch and parse the setlist of a MusicBrainz event."""
        data = self.client.get_json(
            f"{API_ROOT}/event/{event_id}",
            params={"fmt": "json"},
        )
        return parse_event_setlist(data.get("setlist") or "")

    def search_artist_recordings(self, artist: str, limit: int = 50) -> list[TrackCandidate]:
        """Fetch canonical recordings and durations for an artist."""
        if not artist.strip():
            return []
        data = self.client.get_json(
            f"{API_ROOT}/recording/",
            params={"query": f'artist:"{artist.strip()}" AND status:official', "fmt": "json", "limit": limit},
        )
        seen: set[str] = set()
        tracks: list[TrackCandidate] = []
        for rec in data.get("recordings") or []:
            title = (rec.get("title") or "").strip()
            length = float(rec.get("length") or 0) / 1000.0
            key = _norm_text(title)
            if not title or not key or key in seen or length < 30.0:
                continue
            seen.add(key)
            tracks.append(
                TrackCandidate(
                    title=title,
                    duration_seconds=length,
                    position=len(tracks) + 1,
                    notes=rec.get("disambiguation") or "",
                )
            )
        return tracks


# ----------------------------------------------------------------------
def _artist_credit(credit) -> str:
    """The artist name from a MusicBrainz credit block."""
    if not credit:
        return ""
    if isinstance(credit, str):
        return credit
    parts = []
    for entry in credit:
        if isinstance(entry, dict):
            name = entry.get("name", "")
            if name:
                parts.append(name)
        elif isinstance(entry, str):
            parts.append(entry)
    return "".join(parts)


def _first_release_detail(group: dict) -> str:
    """What tells the editions of a release group apart, in one line.

    The plain search answer carries only the identity of each edition, not
    its year, format or label, so the number of editions is used to fill the
    gap: it is what lets somebody recognise "the 1974 LP" from a list.
    """
    releases = group.get("releases") or []
    if releases:
        release = releases[0]
        parts = [str(release.get("date", "") or "")[:4]]
        if release.get("country"):
            parts.append(release["country"])
        formats = _format_names(release)
        if formats:
            parts.append(", ".join(formats))
        detail = ", ".join(part for part in parts if part)
        if detail:
            return detail
    count = len(releases)
    if count:
        return f"{count} edition{'s' if count != 1 else ''}"
    return ""


def _format_names(release: dict) -> list[str]:
    """The formats of a release: "CD", "12\" Vinyl", and so on."""
    names = []
    for item in release.get("media") or []:
        single = item.get("format")
        if isinstance(single, str) and single and single not in names:
            names.append(single)
        for entry in item.get("formats") or []:
            name = entry.get("name")
            if name and name not in names:
                names.append(name)
    return names


def _label_name(release: dict) -> str:
    for entry in release.get("label-info") or []:
        label = entry.get("label") or {}
        name = label.get("name")
        if name:
            catalog = entry.get("catalog-number")
            return f"{name} {catalog}".strip() if catalog else name
    return ""


def _track_notes(track: dict) -> str:
    """Anything the provider says about this track."""
    parts = []
    recording = track.get("recording") or {}
    length = recording.get("length")
    if track.get("length") and length and track["length"] != length:
        parts.append("the recording is a different length from the track")
    if recording.get("disambiguation"):
        parts.append(recording["disambiguation"])
    return "; ".join(parts)


def _track_side(track_number: str, is_vinyl: bool, index: int, split: int) -> str:
    """Extract vinyl side ('A' or 'B' per disc) from track number or fallback."""
    num = str(track_number or "").strip().upper()
    if num and num[0].isalpha() and num[0] in "ABCDEFGH":
        return "A" if ((ord(num[0]) - ord("A")) % 2 == 0) else "B"
    if is_vinyl:
        return "A" if index < split else "B"
    return ""


def _norm_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def parse_event_setlist(setlist_text: str) -> list[TrackCandidate]:
    """Parse MusicBrainz wiki-style setlist syntax into TrackCandidates.

    MusicBrainz setlist format:
        @ [mbid|Artist Name]
        * [mbid|Song Title]
        # Encore / set comment
    """
    tracks: list[TrackCandidate] = []
    if not setlist_text:
        return tracks
    set_number = "1"
    is_encore = False
    for raw_line in setlist_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip().lower()
            if "encore" in heading:
                is_encore = True
                set_number = "Encore"
            elif "set 2" in heading or "act 2" in heading:
                set_number = "2"
                is_encore = False
            continue
        if line.startswith("*"):
            body = line[1:].strip()
            # Unwrap [mbid|Title] wiki links
            body = re.sub(r"\[[^\]|]+\|([^\]]+)\]", r"\1", body)
            body = re.sub(r"\[([^\]]+)\]", r"\1", body)
            # Separate parenthetical note if present
            note = ""
            m_note = re.search(r"\(([^)]+)\)\s*$", body)
            if m_note and any(k in m_note.group(1).lower() for k in ("cover", "encore", "medley", "intro", "outro", "tape", "snippet")):
                note = m_note.group(1).strip()
                body = body[:m_note.start()].strip()
            if body:
                tracks.append(
                    TrackCandidate(
                        title=body,
                        duration_seconds=0.0,
                        position=len(tracks) + 1,
                        notes=note or ("Encore" if is_encore else ""),
                        set_number=set_number,
                        is_encore=is_encore,
                        is_medley="medley" in (note.lower() + " " + body.lower()),
                    )
                )
    return tracks


def api_key_or_none() -> Optional[str]:
    """MusicBrainz needs no key; this exists so both providers look alike."""
    return None
