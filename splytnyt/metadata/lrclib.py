"""LRCLIB: open reference lyrics lookup (no API key required).

Used as supporting evidence for Broadway cast numbers and live concert song
verification when combined with local speech/lyric transcription.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from . import ProviderError
from .base import HttpClient, default_cache_dir

LRCLIB = "LRCLIB"
API_ROOT = "https://lrclib.net/api"
MINIMUM_INTERVAL = 0.25


@dataclass
class LyricEntry:
    """Reference lyrics fetched from LRCLIB."""

    title: str
    artist: str = ""
    album: str = ""
    duration_seconds: float = 0.0
    plain_lyrics: str = ""
    synced_lyrics: str = ""
    provenance: str = ""


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


class LrclibProvider:
    """Fetches reference lyrics from lrclib.net with local disk caching."""

    name = LRCLIB

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.client = HttpClient(
            LRCLIB,
            cache_dir=cache_dir if cache_dir is not None else default_cache_dir(),
            min_interval=MINIMUM_INTERVAL,
            timeout=12.0,
            max_retries=2,
        )

    def test_connection(self) -> tuple[bool, str]:
        """Verify that LRCLIB responds cleanly (no API key required)."""
        try:
            data = self.client.get_json(
                f"{API_ROOT}/search",
                params={"track_name": "Time", "artist_name": "Pink Floyd"},
                use_cache=False,
            )
        except ProviderError as exc:
            return False, str(exc)
        if isinstance(data, list):
            return True, "Connected to LRCLIB (free, no key needed)."
        return False, "LRCLIB returned an unexpected response."

    def fetch_lyrics(
        self,
        title: str,
        artist: str = "",
        album: str = "",
    ) -> Optional[LyricEntry]:
        """Return the best matching lyric entry for a song, or ``None``."""
        clean_title = (title or "").strip()
        if not clean_title:
            return None
        params: dict[str, str] = {"track_name": clean_title}
        if artist.strip():
            params["artist_name"] = artist.strip()
        elif album.strip():
            params["album_name"] = album.strip()

        try:
            data = self.client.get_json(f"{API_ROOT}/search", params=params)
        except ProviderError:
            return None

        if not isinstance(data, list) or not data:
            # Fallback to free-text q search when strict field search misses.
            q = " ".join(part for part in (clean_title, artist.strip() or album.strip()) if part)
            try:
                data = self.client.get_json(f"{API_ROOT}/search", params={"q": q})
            except ProviderError:
                return None

        if not isinstance(data, list) or not data:
            return None

        want_title = _norm(clean_title)
        want_artist = _norm(artist)
        want_album = _norm(album)

        best_score = -1.0
        best_item = None
        for item in data:
            if not isinstance(item, dict):
                continue
            plain = (item.get("plainLyrics") or "").strip()
            synced = (item.get("syncedLyrics") or "").strip()
            if not plain and not synced:
                continue
            it_title = _norm(item.get("trackName") or "")
            it_artist = _norm(item.get("artistName") or "")
            it_album = _norm(item.get("albumName") or "")

            score = 0.0
            if it_title == want_title:
                score += 2.0
            elif want_title and (want_title in it_title or it_title in want_title):
                score += 1.0
            if want_artist and want_artist in it_artist:
                score += 1.5
            if want_album and want_album in it_album:
                score += 1.5
            if plain:
                score += 0.3
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None or best_score < 0.8:
            return None

        plain = (best_item.get("plainLyrics") or "").strip()
        synced = (best_item.get("syncedLyrics") or "").strip()
        if not plain and synced:
            plain = re.sub(r"\[[0-9:.]+\]\s*", "", synced).strip()

        rec_id = best_item.get("id", "")
        return LyricEntry(
            title=best_item.get("trackName") or clean_title,
            artist=best_item.get("artistName") or artist,
            album=best_item.get("albumName") or album,
            duration_seconds=float(best_item.get("duration") or 0.0),
            plain_lyrics=plain,
            synced_lyrics=synced,
            provenance=f"LRCLIB #{rec_id} ({best_item.get('albumName') or best_item.get('artistName') or 'search'})",
        )

    def fetch_for_titles(
        self,
        titles: Sequence[str],
        artist: str = "",
        album: str = "",
        max_items: int = 25,
    ) -> dict[str, LyricEntry]:
        """Fetch lyrics for a sequence of track titles, keyed by input title."""
        found: dict[str, LyricEntry] = {}
        for title in list(titles)[:max_items]:
            entry = self.fetch_lyrics(title, artist=artist, album=album)
            if entry is not None:
                found[title] = entry
        return found
