"""The SplytNyt project model.

Pure data. No Qt, no numpy, no audio I/O. Everything here is importable in a
bare Python interpreter, which keeps the project file format portable and the
model unit-testable without native dependencies.

Design rules enforced by this module:

* Every position is an ``int`` frame index in one global timeline space.
* Sources are never mutated; the model only ever *references* them.
* ``boundary_confidence`` and ``identity_confidence`` are distinct fields and
  are never merged.
* Locks protect a track's boundaries and metadata from re-analysis.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Iterator, Optional

SCHEMA_VERSION = 2

# Section types. "uncertain" is a first-class state: the product requires that
# a boundary we are not sure about says so rather than guessing confidently.
SECTION_TYPES = (
    "unknown",
    "music",
    "applause",
    "speech",
    "silence",
    "dialogue",
    "uncertain",
    # --- theatrical section types -------------------------------------
    # A musical number is a song, not merely "music": it is a self-contained
    # stretch of singing (with or without accompaniment) that a listener would
    # name. "Underscoring" is instrumental music that carries the action
    # between lines. "Mixed" is dialogue that happens *over* music, which is
    # common in theatre and which a single label cannot express.
    "musical_number",
    "underscoring",
    "mixed",
)

#: Section types that describe theatrical material rather than a plain
#: recording. Kept separate so the ordinary modes are untouched.
THEATER_SECTION_TYPES = (
    "musical_number",
    "dialogue",
    "underscoring",
    "applause",
    "mixed",
    "silence",
    "uncertain",
    "unknown",
)

#: Activity labels. These are deliberately **multi-label**: the same stretch
#: of audio can carry several at once. Dialogue played over an underscoring
#: holds a "speech" activity and an "instrumental" activity at the same
#: time, and forcing a choice between "speech" and "music" would throw away
#: exactly the information a theatre review needs.
ACTIVITY_TYPES = (
    "speech",       # spoken words
    "singing",      # sung voice
    "instrumental", # music without a sung line
    "applause",     # the audience
    "silence",      # nothing audible
)

#: How an activity's evidence was obtained.
ACTIVITY_BASIS_ACOUSTIC = "acoustic"
ACTIVITY_BASIS_ASR = "asr"
ACTIVITY_BASIS_MIXED = "acoustic+asr"
ACTIVITY_BASIS_MANUAL = "manual"

#: Export profiles.
#:
#: "complete" keeps every stretch of the recording: the tracks tile the
#: timeline, so the exported audio is the whole performance with nothing
#: lost. "songs" exports the musical numbers only, and reports every
#: exclusion so the user can see what is being left out.
EXPORT_PROFILE_COMPLETE = "complete"
EXPORT_PROFILE_SONGS = "songs"
EXPORT_PROFILES = (EXPORT_PROFILE_COMPLETE, EXPORT_PROFILE_SONGS)

#: How a confidence came about. "none" means nobody has assessed it, which
#: the interface shows as "Not assessed" rather than as a meaningless 0 %.
BOUNDARY_BASIS_NONE = "none"
BOUNDARY_BASIS_ASSESSED = "assessed"
BOUNDARY_BASIS_MANUAL = "manual"
#: A person looked at a proposed cut and accepted it. It is not a manual
#: placement, and it is not an automatic assessment either.
BOUNDARY_BASIS_ACCEPTED = "accepted"

BASIS_LABELS = {
    BOUNDARY_BASIS_NONE: "Not assessed",
    BOUNDARY_BASIS_ASSESSED: "Assessed",
    BOUNDARY_BASIS_MANUAL: "Manual",
    BOUNDARY_BASIS_ACCEPTED: "Accepted",
}

#: The default margin between a cut and the musical onset it was derived
#: from. It lives here, in the data model, so that the project, the analysis
#: and the refinement all use the same number: when they disagreed, the
#: analysis placed a cut and the refinement immediately moved it.
DEFAULT_PRE_ROLL_SECONDS = 0.030

# Flag identifiers used by the review UI and the export report.
FLAG_TRACK_COUNT_MISMATCH = "track_count_mismatch"
FLAG_SUSPICIOUSLY_SHORT = "suspiciously_short"
FLAG_SUSPICIOUSLY_LONG = "suspiciously_long"
FLAG_CROSSES_SOURCE = "crosses_source"
FLAG_SPANS_GAP = "spans_gap"
FLAG_UNIDENTIFIED = "unidentified"
FLAG_LOCKED = "locked"
FLAG_GAP_BETWEEN_SOURCES = "gap_between_sources"
FLAG_OVERLAP_BETWEEN_SOURCES = "overlap_between_sources"
#: The section holds more than one kind of activity at once (dialogue over
#: music, or a number with applause inside it).
FLAG_MIXED_ACTIVITY = "mixed_activity"
#: The section contains more than one song (a medley or a composed sequence).
FLAG_MEDLEY = "medley"
#: The section type could not be decided and needs a human.
FLAG_UNCERTAIN_SECTION = "uncertain_section"
#: Two acoustic recognition engines (e.g. AcoustID and ShazamIO) disagreed.
FLAG_RECOGNITION_CONFLICT = "recognition_conflict"

FLAG_LABELS = {
    FLAG_TRACK_COUNT_MISMATCH: "Track count differs from worklist",
    FLAG_SUSPICIOUSLY_SHORT: "Suspiciously short",
    FLAG_SUSPICIOUSLY_LONG: "Suspiciously long",
    FLAG_CROSSES_SOURCE: "Spans a source file boundary",
    FLAG_SPANS_GAP: "Spans a gap between source files",
    FLAG_UNIDENTIFIED: "Unidentified",
    FLAG_LOCKED: "Locked",
    FLAG_GAP_BETWEEN_SOURCES: "Gap between source files",
    FLAG_OVERLAP_BETWEEN_SOURCES: "Overlap between source files",
    FLAG_MIXED_ACTIVITY: "More than one kind of activity",
    FLAG_MEDLEY: "Contains more than one song",
    FLAG_UNCERTAIN_SECTION: "Section type needs a human decision",
    FLAG_RECOGNITION_CONFLICT: "Recognition engines disagreed",
}


def new_id(prefix: str) -> str:
    """Generate a short, collision-resistant identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (no timezone objects involved)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Metadata:
    """Editable tag payload.

    The same class is used for project-level and track-level metadata. Track
    values override project values field-by-field when both are set, which
    keeps a single global tag sheet editable while still allowing a single
    track to override, say, the performer.
    """

    title: str = ""
    artist: str = ""
    album: str = ""
    album_artist: str = ""
    composer: str = ""
    genre: str = ""
    date: str = ""
    comments: str = ""
    track_number: int = 0
    disc_number: int = 0
    lyrics: str = ""
    publisher: str = ""
    conductor: str = ""
    performer: str = ""
    production: str = ""
    act: str = ""
    song: str = ""
    # Live / performance fields.
    performance_date: str = ""
    venue: str = ""
    city: str = ""
    country: str = ""
    tour: str = ""
    source_notes: str = ""
    # Provenance / archival fields.
    equipment: str = ""
    lineage: str = ""
    identifier: str = ""
    artwork_path: str = ""

    _TAGGED = (
        "title", "artist", "album", "album_artist", "composer", "genre", "date",
        "comments", "track_number", "disc_number", "lyrics", "publisher",
        "conductor", "performer", "production", "act", "song",
        "performance_date", "venue", "city", "country", "tour", "source_notes",
        "equipment", "lineage", "identifier",
    )

    def is_empty(self) -> bool:
        """True when no field carries a value."""
        for name in self._TAGGED:
            if getattr(self, name):
                return False
        return not self.artwork_path

    def merged_with(self, override: "Metadata") -> "Metadata":
        """Return a copy of ``self`` with non-empty fields of ``override`` applied."""
        result = replace(self)
        for name in self._TAGGED:
            value = getattr(override, name)
            if value not in ("", 0):
                setattr(result, name, value)
        if override.artwork_path:
            result.artwork_path = override.artwork_path
        return result


@dataclass
class Source:
    """A reference to an imported audio file. The file itself is never written to."""

    source_id: str = field(default_factory=lambda: new_id("src"))
    path: str = ""
    original_format: str = ""
    sample_rate: int = 0
    channels: int = 0
    frames: int = 0
    size_bytes: int = 0
    mtime_ns: int = 0
    #: Frames of silence inserted *before* this source relative to the previous
    #: one. Negative values mean the sources overlap. Never silently altered.
    gap_frames: int = 0
    #: Content fingerprint of the head/tail of the decoded audio, used to
    #: detect that a file changed underneath us.
    head_hash: str = ""
    tail_hash: str = ""
    #: True when the file was missing the last time the project was opened.
    missing: bool = False
    label: str = ""

    def fingerprint_matches(self, other: "Source") -> bool:
        """True when ``other`` describes the same bytes as this source."""
        return (
            self.path == other.path
            and self.size_bytes == other.size_bytes
            and self.head_hash == other.head_hash
            and self.tail_hash == other.tail_hash
        )


@dataclass
class WorklistItem:
    """One entry of a candidate song list (pasted, or fetched in Phase 2)."""

    title: str = ""
    #: Optional approximate duration in seconds, from a provider. It is
    #: evidence, never a cutting instruction.
    duration_hint: float = 0.0
    notes: str = ""
    position: int = 0
    set_number: str = ""
    provider: str = ""
    provider_id: str = ""
    #: The songs that can live *inside* this number, when the list says it
    #: holds several. A medley's inner songs are separate things from the
    #: number itself, so they are kept apart from the title rather than
    #: being merged into it.
    contains: list[str] = field(default_factory=list)


@dataclass
class Worklist:
    """A candidate song list a project can be aligned against."""

    worklist_id: str = field(default_factory=lambda: new_id("wl"))
    name: str = "Tracklist"
    items: list[WorklistItem] = field(default_factory=list)
    provider: str = "manual"
    provider_id: str = ""
    fetched_at: str = ""
    #: One of: concert, album, theater, unknown.
    mode: str = "unknown"
    notes: str = ""

    def total_hint_seconds(self) -> float:
        return sum(item.duration_hint for item in self.items)


@dataclass
class ActivityRegion:
    """One span of one activity in the timeline.

    Activities overlap freely. A stretch of dialogue played over an
    underscoring holds a ``speech`` region and an ``instrumental`` region at
    the same time; they are two records, not one record with two labels, so
    that each can carry its own confidence and its own evidence.

    The frame numbers are global timeline frames, like every other position
    in this model.
    """

    activity_id: str = field(default_factory=lambda: new_id("act"))
    start_frame: int = 0
    end_frame: int = 0
    #: One of :data:`ACTIVITY_TYPES`.
    label: str = "speech"
    confidence: float = 0.0
    #: One of ``acoustic``, ``asr``, ``acoustic+asr``, ``manual``.
    basis: str = ACTIVITY_BASIS_ACOUSTIC
    reason: str = ""
    #: The numbers behind the decision, for the details panel.
    evidence: dict = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    def duration_seconds(self, sample_rate: int) -> float:
        return self.frames / float(sample_rate) if sample_rate else 0.0


@dataclass
class ComponentSong:
    """A named song found *inside* a larger number.

    A medley or a composed sequence holds several songs, and the number that
    contains them has its own title. The component keeps the inner song's
    name, its own boundaries and its own confidence, so the two never get
    confused: the outer track is the number, the component is the song
    inside it.
    """

    component_id: str = field(default_factory=lambda: new_id("cmp"))
    title: str = ""
    start_frame: int = 0
    end_frame: int = 0
    confidence: float = 0.0
    #: One of ``reference``, ``asr``, ``manual``.
    basis: str = ""
    reason: str = ""

    @property
    def frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)


@dataclass
class Track:
    """A segment of the timeline with its identity and evidence."""

    track_id: str = field(default_factory=lambda: new_id("trk"))
    start_frame: int = 0
    end_frame: int = 0
    title: str = ""
    section_type: str = "unknown"
    #: How sure we are that *this cut* is in the right place.
    boundary_confidence: float = 0.0
    #: How sure we are *what this is*.
    identity_confidence: float = 0.0
    #: How sure we are of the section-type label.
    classification_confidence: float = 0.0
    #: Human-readable evidence for the proposed cut.
    reasons: list[str] = field(default_factory=list)
    locked: bool = False
    flags: list[str] = field(default_factory=list)
    metadata: Metadata = field(default_factory=Metadata)
    worklist_id: str = ""
    worklist_index: int = -1
    notes: str = ""
    #: Set when the track was produced by a suggestion, so the UI can offer
    #: "accept / reject" affordances.
    suggested: bool = False
    #: How the boundary confidence came about: none, assessed or manual.
    boundary_basis: str = BOUNDARY_BASIS_NONE
    #: How the identity confidence came about: none, assessed or manual.
    identity_basis: str = BOUNDARY_BASIS_NONE
    #: Which placement rule produced the cut at the start of this track.
    boundary_placement: str = ""
    #: The musical onset the cut was derived from, or -1.
    onset_frame: int = -1
    #: The margin that was actually applied between the cut and that onset,
    #: in seconds. Negative when the cut was not derived from an onset.
    boundary_pre_roll_seconds: float = -1.0
    #: The transition region the cut sits in, so it can be refined later.
    transition_start: int = -1
    transition_end: int = -1
    #: True when the detector was not sure about this cut and a human has
    #: to look at it.
    boundary_uncertain: bool = False
    #: Which activities occur inside this track. Multi-label on purpose:
    #: a track holding dialogue over an underscoring carries both.
    activities: list[str] = field(default_factory=list)
    #: Named songs found inside this track (a medley, a composed sequence).
    #: Kept separate from the track's own title.
    components: list[ComponentSong] = field(default_factory=list)
    #: A manual grouping/type/tag edit that reanalysis must preserve.
    manual_override: bool = False
    #: Acoustic recognition results (kept distinct from worklist/setlist alignment).
    recognized_artist: str = ""
    recognized_title: str = ""
    #: One of "", "confirmed", "likely", "uncertain", "conflict", "unidentified".
    recognition_status: str = ""
    recognition_engines: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Human-readable assessments
    # ------------------------------------------------------------------
    def boundary_assessment(self) -> str:
        """A word, not a fake percentage.

        A number such as "63 %" claims a precision the detector does not
        have, so the interface shows what the situation actually is.
        """
        if self.boundary_basis == BOUNDARY_BASIS_MANUAL:
            return "Manual"
        if self.boundary_basis == BOUNDARY_BASIS_ACCEPTED:
            return "Accepted"
        if self.boundary_basis == BOUNDARY_BASIS_NONE:
            return "Not assessed"
        if self.boundary_uncertain:
            return "Needs review"
        value = self.boundary_confidence
        if value >= 0.70:
            return "High"
        if value >= 0.45:
            return "Fair"
        return "Low"

    def identity_assessment(self) -> str:
        if self.identity_basis == BOUNDARY_BASIS_MANUAL:
            return "Manual"
        if self.recognition_status:
            return {
                "confirmed": "Confirmed",
                "likely": "Likely",
                "uncertain": "Uncertain",
                "conflict": "Conflict",
                "unidentified": "Unidentified",
            }.get(self.recognition_status, "Uncertain")
        if self.identity_basis == BOUNDARY_BASIS_NONE:
            return "Not assessed"
        value = self.identity_confidence
        if value >= 0.70:
            return "High"
        if value >= 0.45:
            return "Fair"
        return "Low"

    @property
    def frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    def duration_seconds(self, sample_rate: int) -> float:
        return self.frames / float(sample_rate) if sample_rate else 0.0

    def add_reason(self, reason: str) -> None:
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    def has_flag(self, flag: str) -> bool:
        return flag in self.flags


@dataclass
class Marker:
    """A named annotation attached to a timeline position."""

    marker_id: str = field(default_factory=lambda: new_id("mk"))
    frame: int = 0
    label: str = ""
    notes: str = ""
    color: str = "#e0b040"


@dataclass
class ExportSettings:
    """Everything the exporter needs. Kept in the project so exports repeat."""

    output_dir: str = ""
    filename_template: str = "{n:02d} {artist} - {title}"
    container: str = "flac"  # flac | wav | mp3
    mode: str = "separate"  # separate | continuous
    include_cue: bool = True
    include_m3u: bool = False
    include_info: bool = True
    include_checksums: bool = True
    write_tags: bool = True
    artwork_path: str = ""
    # Level handling.
    gain_mode: str = "untouched"  # untouched | shared | loudness | per_track
    shared_gain_db: float = 0.0
    loudness_target_lufs: float = -14.0
    true_peak_ceiling_dbtp: float = -1.0
    allow_limiter: bool = False
    replaygain: bool = False
    # Encoding.
    mp3_bitrate_kbps: int = 192
    mp3_vbr: bool = False
    flac_compression: int = 5
    wav_subtype: str = "PCM_16"
    sample_rate_override: int = 0
    channels_override: int = 0
    #: Which export profile to use. "complete" keeps every stretch of the
    #: recording (the tracks tile the timeline, so no audio is lost);
    #: "songs" exports the musical numbers only and reports every exclusion.
    export_profile: str = EXPORT_PROFILE_COMPLETE
    #: Write named submarkers for songs found inside a number (medleys).
    export_submarkers: bool = False
    #: Include the section type and the detected activities in the written
    #: tags, so an exported file says what it contains.
    export_activity_tags: bool = True


@dataclass
class Project:
    """The whole document: sources, tracks, worklists, metadata, settings."""

    project_id: str = field(default_factory=lambda: new_id("prj"))
    name: str = "Untitled project"
    schema_version: int = SCHEMA_VERSION
    created: str = field(default_factory=now_iso)
    modified: str = field(default_factory=now_iso)
    path: str = ""

    sources: list[Source] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    #: Timeline-level activity annotations. Overlapping by design.
    activities: list[ActivityRegion] = field(default_factory=list)
    worklists: list[Worklist] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    metadata: Metadata = field(default_factory=Metadata)
    export_settings: ExportSettings = field(default_factory=ExportSettings)
    active_worklist_id: str = ""
    notes: str = ""
    #: Free-form key/value store for caches that must survive a reload.
    cache: dict = field(default_factory=dict)
    #: How new boundaries are placed: near the next song, near the previous
    #: one, in the middle of the gap, or by hand.
    placement_mode: str = "next_start"
    #: How far before a musical onset a cut is placed.
    placement_pre_roll_seconds: float = DEFAULT_PRE_ROLL_SECONDS

    # ------------------------------------------------------------------
    # Timeline geometry
    # ------------------------------------------------------------------
    def source_start(self, source: Source) -> int:
        """Global timeline frame where ``source`` begins."""
        start = 0
        for candidate in self.sources:
            if candidate.source_id == source.source_id:
                return start
            start += candidate.frames + candidate.gap_frames
        return start

    def source_starts(self) -> dict[str, int]:
        starts: dict[str, int] = {}
        cursor = 0
        for source in self.sources:
            starts[source.source_id] = cursor
            cursor += source.frames + source.gap_frames
        return starts

    def total_frames(self) -> int:
        """Length of the timeline in frames."""
        if not self.sources:
            return 0
        return sum(s.frames for s in self.sources) + sum(s.gap_frames for s in self.sources[1:])

    def sample_rate(self) -> int:
        """Sample rate of the first source (the timeline's reference rate)."""
        for source in self.sources:
            if source.sample_rate:
                return source.sample_rate
        return 0

    def channels(self) -> int:
        for source in self.sources:
            if source.channels:
                return source.channels
        return 0

    def source_at(self, frame: int) -> Optional[Source]:
        """The source containing ``frame``, or ``None`` in a gap."""
        for source in self.sources:
            start = self.source_start(source)
            if start <= frame < start + source.frames:
                return source
        return None

    def gap_regions(self) -> list[tuple[int, int]]:
        """Gaps between sources as ``(start, end)`` timeline frame pairs."""
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for index, source in enumerate(self.sources):
            if index > 0:
                if source.gap_frames > 0:
                    gaps.append((cursor, cursor + source.gap_frames))
                elif source.gap_frames < 0:
                    gaps.append((cursor + source.gap_frames, cursor))
            cursor += source.frames + source.gap_frames
        return gaps

    def overlap_regions(self) -> list[tuple[int, int]]:
        """Overlaps between sources as ``(start, end)`` timeline frame pairs."""
        overlaps: list[tuple[int, int]] = []
        cursor = 0
        for index, source in enumerate(self.sources):
            if index > 0 and source.gap_frames < 0:
                overlaps.append((cursor + source.gap_frames, cursor))
            cursor += source.frames + source.gap_frames
        return overlaps

    # ------------------------------------------------------------------
    # Track helpers
    # ------------------------------------------------------------------
    def sorted_tracks(self) -> list[Track]:
        return sorted(self.tracks, key=lambda t: (t.start_frame, t.end_frame))

    def track_at(self, frame: int) -> Optional[Track]:
        for track in self.sorted_tracks():
            if track.start_frame <= frame < track.end_frame:
                return track
        return None

    def boundary_positions(self) -> list[int]:
        """Every interior cut position, sorted."""
        positions: set[int] = set()
        for track in self.tracks:
            positions.add(track.start_frame)
            positions.add(track.end_frame)
        if self.total_frames():
            positions.add(0)
            positions.add(self.total_frames())
        return sorted(positions)

    def shared_boundaries(self) -> list[int]:
        """Every interior shared divider between adjacent tracks, sorted."""
        tracks = self.sorted_tracks()
        boundaries: list[int] = []
        for left, right in zip(tracks, tracks[1:]):
            if left.end_frame == right.start_frame:
                boundaries.append(left.end_frame)
        return boundaries

    def is_contiguous(self) -> bool:
        """True when tracks tile [0, total_frames) with zero gaps or overlaps.

        Every adjacent pair satisfies ``tracks[i].end_frame == tracks[i+1].start_frame``.
        """
        tracks = self.sorted_tracks()
        total = self.total_frames()
        if not tracks:
            return total == 0
        if total > 0 and (tracks[0].start_frame != 0 or tracks[-1].end_frame != total):
            return False
        for left, right in zip(tracks, tracks[1:]):
            if left.end_frame <= left.start_frame:
                return False
            if left.end_frame != right.start_frame:
                return False
        return tracks[-1].end_frame > tracks[-1].start_frame

    def enforce_shared_boundaries(self) -> None:
        """Normalize tracks so they are strictly chronological and contiguous.

        Guarantees ``tracks[i].end_frame == tracks[i+1].start_frame`` with no
        gaps or overlaps and full coverage of ``[0, total_frames)``.
        """
        total = self.total_frames()
        tracks = self.sorted_tracks()
        if not tracks:
            return
        cleaned: list[Track] = []
        for track in tracks:
            if track.end_frame > track.start_frame:
                cleaned.append(track)
        if not cleaned:
            cleaned = [tracks[0]]
        if total > 0:
            cleaned[0].start_frame = 0
            cleaned[-1].end_frame = total
        for index in range(len(cleaned) - 1):
            left = cleaned[index]
            right = cleaned[index + 1]
            # Shared divider: snap right.start_frame to left.end_frame
            # clamped strictly inside (left.start_frame, right.end_frame).
            divider = max(left.start_frame + 1, min(left.end_frame, right.end_frame - 1))
            left.end_frame = divider
            right.start_frame = divider
        self.tracks = cleaned

    def active_worklist(self) -> Optional[Worklist]:
        for worklist in self.worklists:
            if worklist.worklist_id == self.active_worklist_id:
                return worklist
        return self.worklists[0] if self.worklists else None

    # ------------------------------------------------------------------
    # Activities
    # ------------------------------------------------------------------
    def activities_in(self, start_frame: int, end_frame: int) -> list[ActivityRegion]:
        """Activity regions overlapping ``[start_frame, end_frame)``."""
        return [
            activity
            for activity in self.activities
            if activity.start_frame < end_frame and start_frame < activity.end_frame
        ]

    def activity_labels_for(self, track: Track) -> list[str]:
        """The distinct activity labels occurring inside ``track``."""
        labels: list[str] = []
        for activity in self.activities_in(track.start_frame, track.end_frame):
            if activity.label not in labels:
                labels.append(activity.label)
        return labels

    def refresh_track_activities(self) -> None:
        """Recompute each track's activity summary from the activity regions."""
        for track in self.tracks:
            track.activities = self.activity_labels_for(track)

    def effective_metadata(self, track: Optional[Track] = None) -> Metadata:
        """Project metadata with the track's overrides applied."""
        if track is None:
            return self.metadata
        return self.metadata.merged_with(track.metadata)

    # ------------------------------------------------------------------
    # Mutations (all called through the undo stack by the app layer)
    # ------------------------------------------------------------------
    def add_source(self, source: Source) -> Source:
        self.sources.append(source)
        self.touch()
        return source

    def add_track(self, track: Track) -> Track:
        self.tracks.append(track)
        self.touch()
        return track

    def touch(self) -> None:
        self.modified = now_iso()

    def clone(self) -> "Project":
        """Deep copy suitable for an undo snapshot."""
        import copy

        return copy.deepcopy(self)


def fingerprint_file(path: str, head_bytes: int = 1 << 20) -> tuple[int, int, str, str]:
    """Return ``(size, mtime_ns, head_sha1, tail_sha1)`` for a file."""
    try:
        stat = os.stat(path)
    except OSError:
        return 0, 0, "", ""
    size = stat.st_size
    mtime = stat.st_mtime_ns
    digest = hashlib.sha1()
    try:
        with open(path, "rb") as handle:
            head = handle.read(head_bytes)
            digest.update(head)
            if size > head_bytes:
                handle.seek(max(0, size - head_bytes))
                digest.update(handle.read(head_bytes))
    except OSError:
        return size, mtime, "", ""
    return size, mtime, digest.hexdigest(), ""


def iter_source_paths(project: Project) -> Iterator[str]:
    for source in project.sources:
        yield source.path
