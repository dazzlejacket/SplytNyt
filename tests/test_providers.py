"""Tests for the optional metadata providers.

The suite never touches the network. It replays **recorded** responses
through the real HTTP client, so the caching, the User-Agent, the rate limit
and the error mapping are all exercised without needing a connection.

The rules these tests defend are the ones the product promises:

* a provider is optional and a pasted tracklist works without one,
* a provider never invents a match: an empty answer stays an empty answer,
* a failure is reported in plain language and never breaks the project,
* setlist.fm needs an API key and says so when there is none.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splytnyt.app.session import Session  # noqa: E402
from splytnyt.metadata import (  # noqa: E402
    ProviderConfigurationError,
    ProviderError,
    ProviderNotFound,
    ProviderRateLimited,
    ProviderUnavailable,
)
from splytnyt.metadata import base as metadata_base  # noqa: E402
from splytnyt.metadata.musicbrainz import MusicBrainzProvider  # noqa: E402
from splytnyt.metadata.setlistfm import SetlistFmProvider  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


# ----------------------------------------------------------------------
# A fake transport that replays recorded responses
# ----------------------------------------------------------------------
class FakeResponse:
    """The little bit of a HTTP response the client actually uses."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> bool:
        return False


def fixture_bytes(name: str) -> bytes:
    with open(os.path.join(FIXTURES, name), "rb") as handle:
        return handle.read()


def install_transport(monkeypatch, routes: dict, calls: list | None = None):
    """Make the HTTP client fetch from ``routes`` instead of the network.

    ``routes`` maps a fragment of the request URL to either the raw bytes to
    return or an exception instance to raise.
    """

    def urlopen(request, timeout=None):
        url = request.full_url
        if calls is not None:
            calls.append(url)
        for fragment, answer in routes.items():
            if fragment in url:
                if isinstance(answer, Exception):
                    raise answer
                return FakeResponse(answer)
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(metadata_base.urllib.request, "urlopen", urlopen)


def http_error(code: int, message: str = "Error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid/", code, message, None, None)


# ----------------------------------------------------------------------
# MusicBrainz
# ----------------------------------------------------------------------
def test_the_release_group_search_finds_the_album(monkeypatch, tmp_path):
    install_transport(
        monkeypatch,
        {"/release-group/": fixture_bytes("musicbrainz_release_group_search.json")},
    )
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    groups = provider.search_albums("Radiohead", "Kid A")

    assert groups, "the recorded answer produced no release group"
    first = groups[0]
    assert first.title == "Kid A"
    assert first.artist == "Radiohead"
    assert first.provider_id
    assert first.detail, "the edition that distinguishes the release is missing"


def test_the_editions_of_a_release_group_are_listed_for_the_user_to_choose(monkeypatch, tmp_path):
    """The user has to be able to pick the exact edition they hold."""
    install_transport(
        monkeypatch,
        {"/release-group/": fixture_bytes("musicbrainz_release_group.json")},
    )
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    editions = provider.releases_of("e75c0549-ad55-39e3-8025-c72c5d4a3c5d")

    assert editions, "no editions were listed"
    for edition in editions:
        assert edition.provider_id
        assert edition.title
        assert edition.track_count > 0, "an edition was listed without a track count"
    # The editions are told apart by their year, their format and their label.
    assert any(edition.format for edition in editions), (
        "the format is missing, so the user cannot tell a CD from a record"
    )


def test_the_exact_release_gives_a_track_list_with_durations_and_sides(monkeypatch, tmp_path):
    install_transport(
        monkeypatch,
        {"/release/": fixture_bytes("musicbrainz_release.json")},
    )
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    tracks = provider.release_tracks("677cf459-923a-49ca-b144-90083728d2dc")

    assert len(tracks) == 11, f"expected the recorded 11 tracks, got {len(tracks)}"
    assert [track.position for track in tracks] == list(range(1, 12)), (
        "the positions are not in order"
    )
    assert tracks[0].title == "Everything in Its Right Place"
    assert tracks[0].duration_seconds == pytest.approx(251.426, abs=0.01), (
        "the duration of the first track was not read"
    )
    # A two-disc record: the discs are numbered, and the sides are worked out.
    assert {track.disc for track in tracks} == {1, 2}
    assert {track.side for track in tracks} == {"A", "B"}
    assert tracks[0].side == "A"
    assert tracks[4].side == "B"


def test_a_provider_never_invents_a_match(monkeypatch, tmp_path):
    """An empty answer stays an empty answer."""
    install_transport(monkeypatch, {"/release-group/": b'{"release-groups": []}'})
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    assert provider.search_albums("Nobody At All", "Nothing") == []


def test_a_release_with_no_artist_gets_no_side_and_no_crash(monkeypatch, tmp_path):
    """Odd data must not produce an exception."""
    payload = json.dumps(
        {
            "media": [
                {
                    "format": "CD",
                    "tracks": [
                        {"position": 1, "title": "One", "length": 1000},
                        {"position": 2, "title": "Two", "length": 2000},
                    ],
                }
            ]
        }
    ).encode("utf-8")
    install_transport(monkeypatch, {"/release/": payload})
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    tracks = provider.release_tracks("anything")

    assert [track.title for track in tracks] == ["One", "Two"]
    assert tracks[0].duration_seconds == pytest.approx(1.0)
    assert tracks[0].side == "", "a CD track was given a side of a record"


def test_an_empty_search_is_refused_before_any_request(monkeypatch, tmp_path):
    """Asking for nothing is answered with a message, not a request."""
    calls: list[str] = []
    install_transport(monkeypatch, {}, calls)
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    with pytest.raises(ProviderNotFound) as info:
        provider.search_albums("   ", "")
    assert "artist" in str(info.value).lower()
    assert calls == [], "a request was made even though there was nothing to ask for"


# ----------------------------------------------------------------------
# setlist.fm
# ----------------------------------------------------------------------
def test_concerts_are_listed(monkeypatch, tmp_path):
    monkeypatch.setenv("SETLISTFM_API_KEY", "test-key")
    install_transport(
        monkeypatch,
        {"/search/setlists": fixture_bytes("setlistfm_search.json")},
    )
    provider = SetlistFmProvider(cache_dir=str(tmp_path / "cache"))

    concerts = provider.search_concerts(artist="The Example Ensemble")

    assert len(concerts) == 2
    first = concerts[0]
    assert first.artist == "The Example Ensemble"
    assert first.venue == "The Old Hall"
    assert first.city == "Reykjavik"
    assert first.date == "13-06-2005"
    assert first.song_count == 2


def test_the_songs_of_a_concert_keep_their_sets_encores_and_medleys(monkeypatch, tmp_path):
    monkeypatch.setenv("SETLISTFM_API_KEY", "test-key")
    install_transport(
        monkeypatch,
        {"/setlist/": fixture_bytes("setlistfm_setlist.json")},
    )
    provider = SetlistFmProvider(cache_dir=str(tmp_path / "cache"))

    songs = provider.concert_tracks("8bfcc3f9-9d2f-4f0e-9a3d-1a2b3c4d5e6f")

    assert len(songs) == 5
    assert songs[0].title == "Opening Piece"
    assert songs[0].set_number == "1"
    assert songs[0].is_encore is False
    # The encore is in the second set and is marked as one.
    assert songs[3].title == "Encore Piece"
    assert songs[3].is_encore is True
    assert songs[3].set_number == "2"
    # A cover is said to be a cover.
    assert "Cover" in songs[2].notes
    # A medley is recognised and kept as it is.
    assert songs[4].is_medley is True


def test_setlist_fm_without_an_api_key_says_so_and_offers_the_alternative(monkeypatch, tmp_path):
    """No key is a configuration problem, and it must be explained."""
    monkeypatch.delenv("SETLISTFM_API_KEY", raising=False)
    monkeypatch.setattr("splytnyt.metadata.setlistfm.get_credential", lambda _name: "")
    provider = SetlistFmProvider(cache_dir=str(tmp_path / "cache"))

    with pytest.raises(ProviderConfigurationError) as info:
        provider.search_concerts(artist="Anyone")
    message = str(info.value)
    assert "SETLISTFM_API_KEY" in message
    assert "pasted tracklist" in message.lower(), (
        "the message does not mention the offline alternative"
    )


def test_setlist_fm_needs_something_to_search_for(monkeypatch, tmp_path):
    monkeypatch.setenv("SETLISTFM_API_KEY", "test-key")
    provider = SetlistFmProvider(cache_dir=str(tmp_path / "cache"))

    with pytest.raises(ProviderNotFound):
        provider.search_concerts(artist="", date="", venue="", city="")


def test_the_api_key_is_sent_as_a_header(monkeypatch, tmp_path):
    """The key has to reach the service, or the request is refused."""
    monkeypatch.setenv("SETLISTFM_API_KEY", "test-key")
    seen: list[dict] = []

    def urlopen(request, timeout=None):
        seen.append(dict(request.header_items()))
        return FakeResponse(fixture_bytes("setlistfm_search.json"))

    monkeypatch.setattr(metadata_base.urllib.request, "urlopen", urlopen)
    provider = SetlistFmProvider(cache_dir=str(tmp_path / "cache"))
    provider.search_concerts(artist="The Example Ensemble")

    headers = {key.lower(): value for key, value in seen[0].items()}
    assert headers.get("x-api-key") == "test-key"
    assert "splytnyt" in headers.get("user-agent", "").lower()


# ----------------------------------------------------------------------
# Failures are survivable
# ----------------------------------------------------------------------
def test_a_refused_request_becomes_a_readable_message(monkeypatch, tmp_path):
    """A 403 must not be a traceback."""
    monkeypatch.setenv("SETLISTFM_API_KEY", "test-key")
    install_transport(monkeypatch, {"/search/setlists": http_error(403, "Forbidden")})
    provider = SetlistFmProvider(cache_dir=str(tmp_path / "cache"))

    with pytest.raises(ProviderUnavailable) as info:
        provider.search_concerts(artist="Anyone")
    message = str(info.value)
    assert "403" in message
    assert "api key" in message.lower()


def test_a_rate_limit_says_so(monkeypatch, tmp_path):
    install_transport(monkeypatch, {"/release-group/": http_error(429, "Too Many Requests")})
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    with pytest.raises(ProviderRateLimited) as info:
        provider.search_albums("Radiohead")
    assert "slower" in str(info.value).lower()


def test_a_missing_answer_is_reported(monkeypatch, tmp_path):
    install_transport(monkeypatch, {"/release-group/": http_error(404, "Not Found")})
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    with pytest.raises(ProviderError):
        provider.search_albums("Radiohead")


def test_an_unreachable_service_does_not_break_anything(monkeypatch, tmp_path):
    """No network at all must leave the application working."""
    install_transport(monkeypatch, {"/release-group/": urllib.error.URLError("no route")})
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    with pytest.raises(ProviderUnavailable) as info:
        provider.search_albums("Radiohead")
    assert "offline" in str(info.value).lower()


def test_a_response_that_is_not_json_is_reported(monkeypatch, tmp_path):
    install_transport(monkeypatch, {"/release-group/": b"<html>not json</html>"})
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    with pytest.raises(ProviderError) as info:
        provider.search_albums("Radiohead")
    assert "json" in str(info.value).lower()


def test_a_provider_failure_leaves_the_project_alone(monkeypatch, tmp_path):
    """The project survives a provider that is not working."""
    install_transport(monkeypatch, {"/release-group/": urllib.error.URLError("no route")})
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    session = Session()
    session.project.name = "A concert"
    before = len(session.project.worklists)

    with pytest.raises(ProviderError):
        provider.search_albums("Radiohead")

    assert len(session.project.worklists) == before, (
        "a failed lookup changed the project"
    )
    assert session.project.name == "A concert"


# ----------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------
def test_a_recorded_answer_is_cached_and_the_network_is_not_used_twice(monkeypatch, tmp_path):
    """A second lookup of the same thing comes from the cache."""
    calls: list[str] = []
    install_transport(
        monkeypatch,
        {"/release-group/": fixture_bytes("musicbrainz_release_group_search.json")},
        calls,
    )
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))

    first = provider.search_albums("Radiohead", "Kid A")
    assert len(calls) == 1
    second = provider.search_albums("Radiohead", "Kid A")

    assert len(calls) == 1, "the answer was fetched twice instead of being cached"
    assert [item.provider_id for item in first] == [item.provider_id for item in second]


def test_the_user_agent_identifies_the_application(monkeypatch, tmp_path):
    """MusicBrainz asks for a real User-Agent."""
    seen: list[dict] = []

    def urlopen(request, timeout=None):
        seen.append(dict(request.header_items()))
        return FakeResponse(fixture_bytes("musicbrainz_release_group_search.json"))

    monkeypatch.setattr(metadata_base.urllib.request, "urlopen", urlopen)
    provider = MusicBrainzProvider(cache_dir=str(tmp_path / "cache"))
    provider.search_albums("Radiohead", "Kid A")

    headers = {key.lower(): value for key, value in seen[0].items()}
    assert "splytnyt" in headers.get("user-agent", "").lower()


# ----------------------------------------------------------------------
# A pasted tracklist needs no provider at all
# ----------------------------------------------------------------------
def test_a_pasted_tracklist_works_offline(tmp_path):
    """The offline route has to be a first-class way to work."""
    session = Session()
    worklist = session.paste_tracklist(
        "1. Opening Piece\n2. Second Piece\n3. Third Piece\n"
    )

    assert len(worklist.items) == 3
    assert [item.title for item in worklist.items] == [
        "Opening Piece", "Second Piece", "Third Piece",
    ]
    assert worklist.provider == "manual"


def test_a_pasted_tracklist_can_be_aligned_with_the_detected_tracks(tmp_path):
    """The lengths in a tracklist are evidence, and they line the songs up."""
    import numpy as np
    import soundfile as sf

    from splytnyt.core.model import Track

    rate = 44100
    path = str(tmp_path / "recording.flac")
    tone = np.stack(
        [np.sin(2 * np.pi * 330 * np.arange(rate * 6) / rate) * 0.4] * 2, axis=1
    ).astype(np.float32)
    sf.write(path, tone, rate, subtype="PCM_16")

    session = Session()
    session.import_files([path])
    session.project.tracks.clear()
    session.project.tracks.append(Track(start_frame=0, end_frame=rate * 2, title=""))
    session.project.tracks.append(Track(start_frame=rate * 2, end_frame=rate * 4, title=""))
    session.project.tracks.append(Track(start_frame=rate * 4, end_frame=rate * 6, title=""))

    session.paste_tracklist("1. First\n2. Second\n3. Third\n")
    result = session.align_with_worklist()

    assert result["matched"] == 3
    assert result["unmatched_items"] == 0
    titles = [track.title for track in session.project.sorted_tracks()]
    assert titles == ["First", "Second", "Third"]
    # An aligned title is not an acoustically verified identity.
    for track in session.project.sorted_tracks():
        assert track.identity_assessment() in {"Fair", "High", "Low"}
