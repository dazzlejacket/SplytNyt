"""setlist.fm: concert lookups.

setlist.fm knows what a band played at a particular concert, and it knows
the shape of the evening: the sets, the encores, the tape intros, and the
notes next to a song.

The service needs an API key, which the user can enter in Settings > Music
Services (stored securely in Windows Credential Manager) or supply via the
environment variable ``SETLISTFM_API_KEY``. Without it the provider says so
plainly and the rest of SplytNyt carries on working.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from ..core.credentials import KEY_SETLISTFM, get_credential
from . import (
    SETLISTFM,
    PerformanceCandidate,
    ProviderConfigurationError,
    ProviderError,
    ProviderNotFound,
    TrackCandidate,
)
from .base import HttpClient, default_cache_dir

API_ROOT = "https://api.setlist.fm/rest/1.0"

#: The environment variable that holds the API key (overrides Credential Manager).
API_KEY_ENVIRONMENT = "SETLISTFM_API_KEY"

#: The service asks for polite rate limiting; 1 req/s is well within limits.
MINIMUM_INTERVAL = 1.0

_COUNTRY_CODES = {
    "japan": "JP",
    "jp": "JP",
    "united states": "US",
    "usa": "US",
    "us": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "france": "FR",
    "italy": "IT",
    "spain": "ES",
    "argentina": "AR",
    "brazil": "BR",
    "mexico": "MX",
    "netherlands": "NL",
    "belgium": "BE",
    "switzerland": "CH",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "finland": "FI",
    "ireland": "IE",
    "poland": "PL",
    "portugal": "PT",
    "austria": "AT",
    "south korea": "KR",
    "korea": "KR",
    "china": "CN",
    "new zealand": "NZ",
    "chile": "CL",
}


def country_name_to_code(text: str) -> str:
    """Map a country name (or 2-letter code) found in a filename to ISO 3166-1 alpha-2."""
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return ""
    if cleaned in _COUNTRY_CODES:
        return _COUNTRY_CODES[cleaned]
    for name, code in _COUNTRY_CODES.items():
        if len(name) >= 4 and re.search(rf"\b{re.escape(name)}\b", cleaned):
            return code
    return ""


def _format_setlistfm_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to dd-MM-yyyy expected by setlist.fm API."""
    s = (date_str or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return s


def api_key() -> str:
    """Return the setlist.fm API key (env var -> Windows Credential Manager), or ``""``."""
    env_val = (os.environ.get(API_KEY_ENVIRONMENT, "") or "").strip()
    if env_val:
        return env_val
    return get_credential(KEY_SETLISTFM)


def has_api_key() -> bool:
    return bool(api_key())


class SetlistFmProvider:
    """Looks up concerts and the songs that were played."""

    name = SETLISTFM

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        api_key_override: Optional[str] = None,
    ) -> None:
        self.cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
        key = (api_key_override if api_key_override is not None else api_key()).strip()
        self._key = key
        headers = {"x-api-key": key} if key else {}
        self.client = HttpClient(
            SETLISTFM,
            cache_dir=self.cache_dir,
            min_interval=MINIMUM_INTERVAL,
            headers=headers,
        )
        self._inline_tracks: dict[str, list[TrackCandidate]] = {}

    # ------------------------------------------------------------------
    def _require_key(self) -> None:
        if not self._key:
            raise ProviderConfigurationError(
                "setlist.fm needs an API key. Add your key in Music Services Settings "
                f"(or set {API_KEY_ENVIRONMENT}) and try again. "
                "A pasted tracklist works without any of this."
            )

    def test_connection(self) -> tuple[bool, str]:
        """Verify that the configured setlist.fm key connects cleanly."""
        if not self._key:
            return False, "No setlist.fm API key configured."
        try:
            data = self.client.get_json(
                f"{API_ROOT}/search/artists",
                params={"artistName": "Muse", "p": 1},
                use_cache=False,
            )
        except ProviderError as exc:
            return False, str(exc)
        if isinstance(data, dict) and ("artist" in data or "total" in data):
            return True, "Connected to setlist.fm."
        return False, "setlist.fm returned an unexpected response."

    # ------------------------------------------------------------------
    def search_concerts(
        self,
        artist: str = "",
        date: str = "",
        venue: str = "",
        city: str = "",
        country_code: str = "",
        year: str = "",
        limit: int = 25,
    ) -> list[PerformanceCandidate]:
        """Find concerts by artist, date, year, country, city, or venue."""
        self._require_key()
        if not any(
            part.strip()
            for part in (artist, date, venue, city, country_code, year)
        ):
            raise ProviderNotFound(
                "Type an artist, a date or a venue to search for."
            )

        # If city looks like a country name (e.g. "Japan"), route it to countryCode.
        cc = (country_code or "").strip().upper()
        city_clean = (city or "").strip()
        if not cc and city_clean:
            inferred_cc = country_name_to_code(city_clean)
            if inferred_cc:
                cc = inferred_cc
                city_clean = ""

        params = {
            "artistName": artist.strip() or None,
            "date": _format_setlistfm_date(date) or None,
            "year": (year or "").strip() or None,
            "venueName": venue.strip() or None,
            "cityName": city_clean or None,
            "countryCode": cc or None,
            "p": 1,
        }
        data = self.client.get_json(f"{API_ROOT}/search/setlists", params=params)
        raw_items = data.get("setlist")
        if raw_items is None:
            raw_items = data.get("setlists") or []

        results: list[PerformanceCandidate] = []
        for item in raw_items:
            cand = _parse_performance(item)
            if cand is not None:
                results.append(cand)
                # Cache inline tracks from the search response so concert_tracks()
                # does not need a second HTTP request (avoids 429 rate limits).
                parsed = _parse_tracks(item)
                if parsed and cand.provider_id:
                    self._inline_tracks[cand.provider_id] = parsed
                    self.client._write_cache(f"{API_ROOT}/setlist/{cand.provider_id}", item)
        return results[:limit]

    # ------------------------------------------------------------------
    def concert_tracks(self, setlist_id: str) -> list[TrackCandidate]:
        """The songs of one concert, with their sets and encores."""
        self._require_key()
        if setlist_id in self._inline_tracks:
            return list(self._inline_tracks[setlist_id])
        data = self.client.get_json(f"{API_ROOT}/setlist/{setlist_id}")
        return _parse_tracks(data)


# ----------------------------------------------------------------------
# JSON parsers (real setlist.fm REST v1.0 + legacy flat test fixtures)
# ----------------------------------------------------------------------

def _parse_performance(item: dict) -> Optional[PerformanceCandidate]:
    if not isinstance(item, dict):
        return None
    artist_obj = item.get("artist")
    if isinstance(artist_obj, dict):
        artist_name = (artist_obj.get("name") or "").strip()
    else:
        artist_name = (item.get("artistName") or "").strip()

    venue_obj = item.get("venue")
    if isinstance(venue_obj, dict):
        venue_name = (venue_obj.get("name") or "").strip()
        city_obj = venue_obj.get("city") or {}
        city_name = (city_obj.get("name") or "").strip()
        country_obj = city_obj.get("country") or {}
        country_name = (country_obj.get("name") or country_obj.get("code") or "").strip()
    else:
        venue_name = (item.get("venueName") or "").strip()
        city_name = (item.get("venueCity") or "").strip()
        country_name = (item.get("venueCountry") or "").strip()

    event_date = (item.get("eventDate") or "").strip()
    detail_parts = [p for p in (event_date, venue_name, city_name) if p]
    return PerformanceCandidate(
        provider=SETLISTFM,
        provider_id=str(item.get("id") or ""),
        artist=artist_name,
        title=venue_name,
        date=event_date,
        venue=venue_name,
        city=city_name,
        country=country_name,
        detail=", ".join(detail_parts),
        song_count=_song_count(item),
    )


def _iter_set_blocks(item: dict) -> list[dict]:
    sets_obj = item.get("sets")
    if isinstance(sets_obj, dict):
        blocks = sets_obj.get("set") or []
        return blocks if isinstance(blocks, list) else []
    if isinstance(sets_obj, list):
        return sets_obj
    return []


def _parse_tracks(data: dict) -> list[TrackCandidate]:
    tracks: list[TrackCandidate] = []
    position = 0
    for set_number, set_block in enumerate(_iter_set_blocks(data), start=1):
        block_encore = set_block.get("encore")
        songs = set_block.get("song")
        if songs is None:
            songs = set_block.get("songs") or []
        for song in songs:
            name = (song.get("name") or "").strip()
            if not name:
                continue
            position += 1
            song_encore = song.get("encore") or block_encore or ""
            is_enc = is_encour(str(song_encore))
            notes = _song_notes(song, is_encore=is_enc)
            tracks.append(
                TrackCandidate(
                    title=name,
                    duration_seconds=0.0,
                    position=position,
                    notes=notes,
                    set_number=str(set_number),
                    is_encore=is_enc,
                    is_medley="medley" in notes.lower(),
                )
            )
    return tracks


def is_encour(value: str) -> bool:
    """True when the provider says a song was an encore."""
    return bool(value) and str(value).strip().lower() not in ("", "0", "false", "none")


def _song_count(item: dict) -> int:
    total = 0
    for set_block in _iter_set_blocks(item):
        songs = set_block.get("song")
        if songs is None:
            songs = set_block.get("songs") or []
        total += len(songs)
    return total


def _song_notes(song: dict, is_encore: bool = False) -> str:
    """Everything the provider says about a song."""
    parts = []
    if is_encore or song.get("encore"):
        parts.append("Encore")
    if song.get("tape"):
        parts.append("Tape")
    cover = song.get("cover")
    if isinstance(cover, dict) and cover.get("name"):
        parts.append(f"Cover ({cover['name']})")
    elif song.get("isCover"):
        parts.append("Cover")
    info = (song.get("info") or song.get("note") or "").strip()
    if info:
        parts.append(info)
    return ", ".join(parts)
