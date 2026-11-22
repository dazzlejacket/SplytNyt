"""Metadata providers.

A provider is an online source of tracklists: an album's track list with its
lengths, or the songs a band played at a concert.

Everything in this package follows the same rules:

* **The provider is optional.** SplytNyt works completely offline; a pasted
  tracklist is a first-class way to work and never needs a provider.
* **Metadata guides; audio refines.** Online metadata (MusicBrainz tracklists
  and durations, AcoustID/Chromaprint fingerprints, setlist.fm / MusicBrainz
  concert setlists, and LRCLIB reference lyrics) suggests expected songs and
  approximate boundary regions, which are always refined against the actual
  audio — never blindly cut at published timestamps.
* **Nothing is forced.** Unidentified tracks stay unnamed rather than invented,
  and every metadata suggestion is editable and reversible.
* **Failures are typed and survivable.** No network, no API key, a rate
  limit or a malformed response produces a clear message and never a crash
  and never an empty result that looks like "nothing matched".
* **Requests are polite.** MusicBrainz requires a real User-Agent and asks
  for one request per second; setlist.fm asks for an API key and limits the
  request rate. Both are handled here, not left to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The name shown in the interface for each provider.
MUSICBRAINZ = "MusicBrainz"
SETLISTFM = "setlist.fm"

PROVIDER_NAMES = (MUSICBRAINZ, SETLISTFM)


class ProviderError(RuntimeError):
    """A provider could not do what was asked."""


class ProviderUnavailable(ProviderError):
    """The provider cannot be reached, or is not configured."""


class ProviderNotFound(ProviderError):
    """The provider has no answer for this query."""


class ProviderRateLimited(ProviderError):
    """The provider asked us to slow down."""


class ProviderConfigurationError(ProviderUnavailable):
    """The provider needs something - an API key - that is not set."""


@dataclass
class TrackCandidate:
    """One song a provider knows about."""

    title: str
    #: Approximate length in seconds. Evidence, never a cutting instruction.
    duration_seconds: float = 0.0
    #: Position within the release or the set, starting at 1.
    position: int = 0
    #: Disc number for an album.
    disc: int = 1
    #: Side of a vinyl record ("A" or "B"), when the provider says so.
    side: str = ""
    #: Free text: "Encore", "Medley", a note from the provider.
    notes: str = ""
    #: The set a song belongs to, for a concert.
    set_number: str = ""
    is_encore: bool = False
    is_medley: bool = False


@dataclass
class ReleaseCandidate:
    """One edition of an album, as the provider lists it."""

    provider: str
    provider_id: str
    title: str
    artist: str = ""
    #: Everything that distinguishes this edition from the others.
    detail: str = ""
    year: str = ""
    format: str = ""
    country: str = ""
    label: str = ""
    discs: int = 1
    track_count: int = 0


@dataclass
class PerformanceCandidate:
    """One concert, as the provider lists it."""

    provider: str
    provider_id: str
    artist: str = ""
    title: str = ""
    date: str = ""
    venue: str = ""
    city: str = ""
    country: str = ""
    detail: str = ""
    song_count: int = 0


@dataclass
class ProviderResult:
    """What a provider returned."""

    provider: str
    #: The exact thing the user picked.
    title: str = ""
    artist: str = ""
    subtitle: str = ""
    provider_id: str = ""
    tracks: list[TrackCandidate] = field(default_factory=list)
    notes: str = ""
