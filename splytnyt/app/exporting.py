"""Export orchestration.

Turns a project into files on disk:

* separate tracks, or one continuous file plus a CUE sheet,
* FLAC, WAV or MP3, preserving the source sample rate and channel layout,
* tags written by the format-specific tag writers,
* optional M3U playlist, archival info file and SHA-256 checksums.

Safety rules enforced here:

* the output directory is checked against every source path before a byte is
  written, and each individual output path is checked again,
* writes go through the format writers, which write to a ``.partial`` file and
  rename on success,
* gaps between source files are preserved as digital silence and reported,
  never dropped silently.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..core import paths as path_tools
from ..core.model import Project, Source, Track
from ..core.units import frames_to_timestamp

#: How non-music sections (applause, speech) are grouped during export.
SECTION_POLICIES = ("keep_as_own", "attach_previous", "attach_next", "omit")

#: Sections that the policy applies to.
MANAGEABLE_SECTIONS = ("applause", "speech", "dialogue")

#: The section types that count as a song for the songs-focused profile.
#: A track that holds named component songs counts too, even when the
#: track itself was typed as something else, because the song is in there.
SONG_SECTION_TYPES = ("musical_number",)


def write_submarkers(project: Project, path: str) -> str:
    """Write the songs found *inside* the numbers as submarkers.

    A medley is one number that holds several songs. The number keeps its
    own title and its own boundaries; the songs inside it are written
    here, with their own boundaries and their own confidence, so the two
    are never confused.
    """
    import json

    sample_rate = project.sample_rate() or 44100
    payload = {
        "format": "splytnyt-submarkers",
        "version": 1,
        "project": project.name,
        "sample_rate": sample_rate,
        "submarkers": [],
    }
    for track in project.sorted_tracks():
        for component in getattr(track, "components", []):
            payload["submarkers"].append(
                {
                    "parent_track_id": track.track_id,
                    "parent_title": track.title,
                    "title": component.title,
                    "start_frame": component.start_frame,
                    "end_frame": component.end_frame,
                    "start_seconds": round(component.start_frame / sample_rate, 6),
                    "end_seconds": round(component.end_frame / sample_rate, 6),
                    "duration_seconds": round(
                        max(0, component.end_frame - component.start_frame) / sample_rate, 6
                    ),
                    "confidence": component.confidence,
                    "basis": component.basis,
                    "reason": component.reason,
                }
            )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def is_song(track: Track) -> bool:
    if track.section_type in SONG_SECTION_TYPES:
        return True
    return bool(getattr(track, "components", []))


def preview_profile(project: Project, profile: str = "complete") -> dict:
    """What an export profile would keep and what it would leave out.

    The songs-focused profile is destructive by nature - it throws away
    everything that is not a number - so the exclusions are worked out and
    shown **before** anything is written.
    """
    tracks = project.sorted_tracks()
    included: list[Track] = []
    excluded: list[tuple[Track, str]] = []
    for track in tracks:
        if profile == "songs" and not is_song(track):
            reason = (
                f"it was detected as '{track.section_type}', not as a musical number"
                if track.section_type != "unknown"
                else "no section type was decided for it"
            )
            excluded.append((track, reason))
        else:
            included.append(track)
    return {
        "profile": profile,
        "included": included,
        "excluded": excluded,
        "included_seconds": sum(
            track.duration_seconds(project.sample_rate() or 44100) for track in included
        ),
        "excluded_seconds": sum(
            track.duration_seconds(project.sample_rate() or 44100)
            for track, _reason in excluded
        ),
    }


@dataclass
class ExportItem:
    """One file to be written."""

    track: Track
    path: str
    parts: list = field(default_factory=list)   # list[(source_like, start, end)]
    frames: int = 0
    title: str = ""
    track_number: int = 0


@dataclass
class ExportReport:
    """Everything the UI needs to tell the user what happened."""

    output_dir: str = ""
    container: str = ""
    files: list[str] = field(default_factory=list)
    #: The total number of source frames written across every file.
    total_frames: int = 0
    #: The total number of seconds of audio written.
    total_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    omissions: list[tuple[str, str]] = field(default_factory=list)
    info_path: str = ""
    cue_path: str = ""
    m3u_path: str = ""
    checksum_path: str = ""
    submarker_path: str = ""
    gain_description: str = ""
    gain_feasible: bool = True
    loudness_lufs: float = 0.0
    true_peak_dbtp: float = 0.0
    verified: bool = False
    verification_note: str = ""
    elapsed_seconds: float = 0.0
    cancelled: bool = False

    def summary_text(self) -> str:
        lines = [f"Exported {len(self.files)} file(s) to {self.output_dir}"]
        if self.total_seconds:
            lines.append(
                f"Total audio: {self.total_seconds:.1f}s "
                f"({int(self.total_seconds // 60)}m {self.total_seconds % 60:.1f}s)"
            )
        if self.gain_description:
            lines.append(f"Level: {self.gain_description}")
        if self.omissions:
            lines.append(f"Omitted {len(self.omissions)} section(s):")
            for title, reason in self.omissions:
                lines.append(f"  - {title or 'untitled'}: {reason}")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        for note in self.notes:
            lines.append(note)
        if self.info_path:
            lines.append(f"Info file: {self.info_path}")
        if self.cue_path:
            lines.append(f"CUE sheet: {self.cue_path}")
        if self.submarker_path:
            lines.append(f"Submarkers: {self.submarker_path}")
        if self.m3u_path:
            lines.append(f"M3U playlist: {self.m3u_path}")
        if self.checksum_path:
            lines.append(f"Checksums: {self.checksum_path}")
        if self.verified:
            lines.append(f"Verified: {self.verification_note}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Timeline segmentation
# ----------------------------------------------------------------------
@dataclass
class _Segment:
    kind: str                # "source" | "gap"
    start: int
    end: int
    source: Optional[Source] = None


def timeline_segments(project: Project) -> list[_Segment]:
    """Split the timeline into source regions and gap regions."""
    segments: list[_Segment] = []
    cursor = 0
    for index, source in enumerate(project.sources):
        if index > 0:
            gap = source.gap_frames
            if gap > 0:
                segments.append(_Segment("gap", cursor, cursor + gap))
                cursor += gap
            elif gap < 0:
                cursor += gap
        segments.append(_Segment("source", cursor, cursor + source.frames, source))
        cursor += source.frames
    return segments


def _subtract(interval: tuple[int, int], covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove already-covered parts from ``interval``."""
    start, end = interval
    pieces = [(start, end)]
    for cover_start, cover_end in covered:
        remaining: list[tuple[int, int]] = []
        for piece_start, piece_end in pieces:
            if cover_end <= piece_start or cover_start >= piece_end:
                remaining.append((piece_start, piece_end))
                continue
            if piece_start < cover_start:
                remaining.append((piece_start, cover_start))
            if cover_end < piece_end:
                remaining.append((cover_end, piece_end))
        pieces = remaining
    return pieces


class PartBuilder:
    """Builds the list of source chunks that make up a timeline range."""

    def __init__(self, project: Project) -> None:
        from ..audio.decode import AudioSource

        self.project = project
        self.sample_rate = project.sample_rate() or 44100
        self.channels = project.channels() or 2
        self.segments = timeline_segments(project)
        self.open_sources: dict[str, object] = {}
        self._source_objects: dict[str, object] = {}
        for source in project.sources:
            self._source_objects[source.source_id] = AudioSource(source.path)

    def close(self) -> None:
        for source in self._source_objects.values():
            try:
                source.close()
            except Exception:
                pass

    def parts_for(self, start: int, end: int) -> list[tuple[object, int, int]]:
        """Chunks covering ``[start, end)``, in order, with gaps as silence."""
        if end <= start:
            return []
        parts: list[tuple[object, int, int]] = []
        covered: list[tuple[int, int]] = []
        for segment in self.segments:
            if segment.end <= start:
                covered.append((segment.start, segment.end))
                continue
            if segment.start >= end:
                break
            for piece_start, piece_end in _subtract(
                (max(segment.start, start), min(segment.end, end)), covered
            ):
                if piece_end <= piece_start:
                    continue
                if segment.kind == "gap":
                    from ..audio.decode import MemorySource

                    silence = MemorySource(piece_end - piece_start, self.sample_rate, self.channels)
                    parts.append((silence, 0, piece_end - piece_start))
                else:
                    source = self._source_objects[segment.source.source_id]
                    offset = piece_start - segment.start
                    parts.append((source, offset, offset + (piece_end - piece_start)))
                covered.append((piece_start, piece_end))
        return parts

    def span_has_gap(self, start: int, end: int) -> bool:
        for segment in self.segments:
            if segment.kind == "gap" and segment.start < end and start < segment.end:
                return True
        return False


# ----------------------------------------------------------------------
# Tag helpers
# ----------------------------------------------------------------------
_METADATA_TO_TAG = {
    "title": "TITLE",
    "artist": "ARTIST",
    "album": "ALBUM",
    "album_artist": "ALBUMARTIST",
    "composer": "COMPOSER",
    "genre": "GENRE",
    "date": "DATE",
    "comments": "COMMENT",
    "track_number": "TRACKNUMBER",
    "disc_number": "DISCNUMBER",
    "lyrics": "LYRICS",
    "publisher": "PUBLISHER",
    "conductor": "CONDUCTOR",
    "performer": "PERFORMER",
    "production": "PRODUCTION",
    "act": "ACT",
    "song": "SONG",
    "performance_date": "PERFORMANCE_DATE",
    "venue": "VENUE",
    "city": "CITY",
    "country": "COUNTRY",
    "tour": "TOUR",
    "source_notes": "SOURCE_NOTES",
    "equipment": "EQUIPMENT",
    "lineage": "LINEAGE",
    "identifier": "IDENTIFIER",
}


def build_tags(project: Project, track: Optional[Track], extra: Optional[dict] = None) -> dict:
    """Canonical tag dictionary for a track (project metadata + overrides)."""
    effective = project.effective_metadata(track)
    tags: dict[str, str] = {}
    for field_name, tag_name in _METADATA_TO_TAG.items():
        value = getattr(effective, field_name, "")
        if value not in ("", 0, None):
            tags[tag_name] = str(value)
    if track is not None:
        tags.setdefault("TITLE", track.title or "")
    if extra:
        tags.update({key: str(value) for key, value in extra.items() if value not in ("", None)})
    return {key: value for key, value in tags.items() if value != ""}


# ----------------------------------------------------------------------
# Main export
# ----------------------------------------------------------------------
def export_project(
    project: Project,
    settings,
    track_ids: Optional[list[str]] = None,
    section_policy: str = "keep_as_own",
    progress: Optional[Callable[[float, str], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> ExportReport:
    """Run a full export. Returns a report describing exactly what happened."""
    started = time.time()
    report = ExportReport()

    def report_progress(fraction: float, message: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, fraction)), message)

    def cancelled() -> bool:
        return bool(cancel and cancel())

    if not project.sources:
        raise ValueError("The project has no audio to export.")
    if not project.tracks:
        raise ValueError("The project has no tracks to export.")

    source_paths = [source.path for source in project.sources]
    for source in project.sources:
        if not os.path.exists(source.path):
            raise FileNotFoundError(
                f"The source file is missing, so nothing was exported: {source.path}"
            )

    output_dir = path_tools.ensure_writable_directory(settings.output_dir)
    report.output_dir = output_dir
    report.container = settings.container

    # ---- select tracks -------------------------------------------------
    tracks = project.sorted_tracks()
    if track_ids is not None:
        wanted = set(track_ids)
        tracks = [track for track in tracks if track.track_id in wanted]
    if not tracks:
        raise ValueError("No tracks were selected for export.")

    # Section policy: group or drop non-music sections.
    # Default "keep_as_own" preserves everything as separate files.
    effective_tracks: list[Track] = []
    deferred: list[Track] = []

    # The export profile decides what is written at all. "complete" keeps
    # every stretch of the recording; "songs" keeps the musical numbers
    # and reports every exclusion so nothing disappears silently.
    profile = getattr(settings, "export_profile", "complete") or "complete"
    if profile == "songs":
        preview = preview_profile(project, profile)
        for track, reason in preview["excluded"]:
            report.omissions.append(
                (
                    track.title or f"{track.section_type or 'untitled'} section",
                    reason,
                )
            )
        report.notes.append(
            f"Songs-focused export: {len(preview['included'])} number(s) kept, "
            f"{len(preview['excluded'])} section(s) left out. "
            f"About {preview['excluded_seconds'] / 60:.1f} min of audio is not in this export."
        )
        allowed = {track.track_id for track in preview["included"]}
        tracks = [track for track in tracks if track.track_id in allowed]
        if not tracks:
            raise ValueError(
                "No musical numbers were detected, so the songs-focused export "
                "would have produced nothing. Use the complete export instead."
            )

    for track in tracks:
        manageable = track.section_type in MANAGEABLE_SECTIONS
        if manageable and section_policy in ("attach_previous", "attach_next", "omit"):
            if section_policy == "omit":
                report.omissions.append(
                    (track.title or f"{track.section_type} section", "excluded from this export")
                )
                continue
            if section_policy == "attach_previous":
                if effective_tracks:
                    effective_tracks[-1].end_frame = max(
                        effective_tracks[-1].end_frame, track.end_frame
                    )
                    effective_tracks[-1].add_reason(
                        f"Attached the following {track.section_type} section"
                    )
                    continue
                # No preceding track to attach to: keep it as its own file.
            elif section_policy == "attach_next":
                deferred.append(track)
                continue
        if deferred:
            section_start = min(item.start_frame for item in deferred)
            track.start_frame = min(track.start_frame, section_start)
            kinds = "/".join(sorted({item.section_type for item in deferred}))
            track.add_reason(f"Attached the preceding {kinds} section")
            deferred = []
        effective_tracks.append(track)

    if deferred and effective_tracks:
        effective_tracks[-1].end_frame = max(
            effective_tracks[-1].end_frame, max(item.end_frame for item in deferred)
        )
        effective_tracks[-1].add_reason("Attached a trailing section")
    elif deferred:
        for item in deferred:
            report.omissions.append(
                (item.title or f"{item.section_type} section",
                 "no following track to attach to, so it was exported on its own")
            )
            effective_tracks.append(item)

    if not effective_tracks:
        raise ValueError("Every track was excluded, so nothing was exported.")

    # ---- build parts ---------------------------------------------------
    builder = PartBuilder(project)
    try:
        items: list[ExportItem] = []
        taken_paths: list[str] = []
        for number, track in enumerate(effective_tracks, start=1):
            parts = builder.parts_for(track.start_frame, track.end_frame)
            frames = sum(end - start for _, start, end in parts)
            effective_meta = project.effective_metadata(track)
            context = {
                "n": number,
                "track": number,
                "title": track.title or f"Track {number:02d}",
                "artist": effective_meta.artist,
                "album": effective_meta.album or project.name,
                "albumartist": effective_meta.album_artist,
                "disc": effective_meta.disc_number,
                "date": effective_meta.date,
                "genre": effective_meta.genre,
                "uniq": number,
            }
            filename = path_tools.format_filename(settings.filename_template, context)
            filename = _sanitise_filename(filename)
            extension = {"flac": ".flac", "wav": ".wav", "mp3": ".mp3"}.get(settings.container, ".flac")
            candidate = os.path.join(output_dir, filename + extension)
            # Refuse before collision handling: renaming the output would
            # quietly write the user's file somewhere they did not ask for,
            # and the requested name is a source recording.
            if path_tools.path_conflicts_with_sources(candidate, source_paths):
                raise PermissionError(
                    f"Refusing to write {candidate}: it would overwrite a source recording."
                )
            candidate = path_tools.unique_path(candidate, taken_paths)
            if path_tools.path_conflicts_with_sources(candidate, source_paths):
                raise PermissionError(
                    f"Refusing to write {candidate}: it would overwrite a source recording."
                )
            taken_paths.append(candidate)
            items.append(
                ExportItem(
                    track=track,
                    path=candidate,
                    parts=parts,
                    frames=frames,
                    title=context["title"],
                    track_number=number,
                )
            )
            if builder.span_has_gap(track.start_frame, track.end_frame):
                report.notes.append(
                    f"Track {number} ({context['title']}) spans a gap between source files; "
                    "the gap is preserved as silence in the exported audio."
                )

        # How much audio this export actually contains. The complete profile
        # tiles the whole timeline, so this is the length of the recording;
        # the songs profile holds only the musical numbers, and the
        # difference is the audio that was left out on purpose.
        report.total_frames = sum(item.frames for item in items)
        report.total_seconds = report.total_frames / float(project.sample_rate() or 48000)

        # ---- gain planning ---------------------------------------------
        gain = 1.0
        shared_gain_db = 0.0
        per_track_gain_db: dict[str, float] = {}
        limiter_needed = False

        if settings.gain_mode in ("shared", "loudness"):
            from ..audio.measure import Part, measure_parts

            measured_parts = []
            for item in items:
                for source_like, start, end in item.parts:
                    measured_parts.append(Part(source_like, start, end))
            if measured_parts:
                measured = measure_parts(measured_parts)
                report.loudness_lufs = measured.integrated_lufs
                report.true_peak_dbtp = measured.true_peak_dbtp
                from ..audio.gain import plan_shared_gain

                plan = plan_shared_gain(
                    measured,
                    settings.loudness_target_lufs,
                    settings.true_peak_ceiling_dbtp,
                )
                report.gain_description = (
                    f"shared gain {plan.gain_db:+.1f} dB "
                    f"(program {measured.integrated_lufs:.1f} LUFS, "
                    f"peak {measured.true_peak_dbtp:.1f} dBTP)"
                )
                report.gain_feasible = plan.feasible
                if not plan.feasible:
                    report.warnings.append(plan.reason)
                    if settings.allow_limiter:
                        # The limiter does the work; the gain is applied and
                        # the peaks are then brought back under the ceiling.
                        shared_gain_db = plan.gain_db
                        limiter_needed = True
                    else:
                        report.warnings.append(
                            "No limiter was applied. Less gain was used instead."
                        )
                        shared_gain_db = plan.safe_gain_db
                else:
                    shared_gain_db = plan.gain_db
                gain = _db_to_linear(shared_gain_db)

        elif settings.gain_mode == "per_track":
            from ..audio.gain import plan_per_track_gain
            from ..audio.measure import Part, measure_parts

            descriptions = []
            all_feasible = True
            for item in items:
                parts = [Part(source_like, start, end)
                         for source_like, start, end in item.parts]
                if not parts:
                    continue
                measured = measure_parts(parts)
                plan = plan_per_track_gain(
                    measured,
                    settings.loudness_target_lufs,
                    settings.true_peak_ceiling_dbtp,
                )
                if not plan.feasible:
                    all_feasible = False
                    report.warnings.append(
                        f"{item.title or 'track'}: {plan.reason}"
                    )
                    if settings.allow_limiter:
                        per_track_gain_db[item.path] = plan.gain_db
                        limiter_needed = True
                    else:
                        per_track_gain_db[item.path] = plan.safe_gain_db
                else:
                    per_track_gain_db[item.path] = plan.gain_db
                descriptions.append(
                    f"{item.title or 'track'}: {per_track_gain_db[item.path]:+.1f} dB "
                    f"({measured.integrated_lufs:.1f} LUFS)"
                )
            report.gain_description = "per-track gain: " + "; ".join(descriptions)
            report.gain_feasible = all_feasible

        report_progress(0.05, "Preparing export")

        # ---- write the files -------------------------------------------
        from ..audio.encode import write_cue, write_flac, write_mp3, write_wav

        writers = {"flac": write_flac, "wav": write_wav, "mp3": write_mp3}
        writer = writers.get(settings.container)
        if writer is None:
            raise ValueError(f"Unknown container: {settings.container}")

        total_items = len(items)
        for index, item in enumerate(items):
            if cancelled():
                report.cancelled = True
                report.warnings.append("The export was cancelled; no complete file was produced for the remaining tracks.")
                break
            report_progress(
                0.1 + 0.7 * (index / max(1, total_items)),
                f"Writing {os.path.basename(item.path)}",
            )
            item_gain = gain
            if per_track_gain_db:
                item_gain = _db_to_linear(per_track_gain_db.get(item.path, 0.0))
            processor = None
            if limiter_needed:
                from ..audio.limit import PeakLimiter

                processor = PeakLimiter(
                    settings.true_peak_ceiling_dbtp,
                    project.sample_rate() or 44100,
                )
            if settings.container == "flac":
                result = writer(
                    item.path, item.parts, gain=item_gain,
                    compression=settings.flac_compression, processor=processor,
                )
            elif settings.container == "wav":
                result = writer(
                    item.path, item.parts, gain=item_gain,
                    subtype=settings.wav_subtype, processor=processor,
                )
            else:
                result = writer(
                    item.path,
                    item.parts,
                    gain=item_gain,
                    bitrate_kbps=settings.mp3_bitrate_kbps,
                    vbr=settings.mp3_vbr,
                    processor=processor,
                )
            if processor is not None:
                report.notes.extend(processor.notes())
            report.files.append(result.path)
            for note in getattr(result, "notes", []):
                report.notes.append(f"{os.path.basename(item.path)}: {note}")

            if settings.write_tags:
                _write_tags(settings.container, result.path, project, item.track, item.track_number)

        # ---- continuous + CUE -------------------------------------------
        if settings.mode == "continuous" and not report.cancelled:
            report_progress(0.85, "Writing continuous audio and CUE sheet")
            all_parts: list = []
            for item in items:
                all_parts.extend(item.parts)
            processor = None
            if limiter_needed:
                from ..audio.limit import PeakLimiter

                processor = PeakLimiter(
                    settings.true_peak_ceiling_dbtp,
                    project.sample_rate() or 44100,
                )
            if settings.container == "flac":
                continuous_path = os.path.join(output_dir, "program.flac")
                write_flac(continuous_path, all_parts, gain=gain,
                           compression=settings.flac_compression, processor=processor)
            elif settings.container == "wav":
                continuous_path = os.path.join(output_dir, "program.wav")
                write_wav(continuous_path, all_parts, gain=gain,
                          subtype=settings.wav_subtype, processor=processor)
            else:
                continuous_path = os.path.join(output_dir, "program.mp3")
                write_mp3(continuous_path, all_parts, gain=gain,
                          bitrate_kbps=settings.mp3_bitrate_kbps,
                          vbr=settings.mp3_vbr, processor=processor)
            if processor is not None:
                report.notes.extend(processor.notes())
            report.files.append(continuous_path)

            if settings.include_cue:
                from ..audio.encode import CueTrack

                cue_tracks = []
                for item in items:
                    eff = project.effective_metadata(item.track)
                    cue_tracks.append(
                        CueTrack(
                            title=item.title,
                            start_frame=item.track.start_frame,
                            end_frame=item.track.end_frame,
                            artist=eff.artist,
                            album=eff.album or project.name,
                            track_number=item.track_number,
                        )
                    )
                cue_path = os.path.join(output_dir, "program.cue")
                write_cue(
                    cue_path,
                    os.path.basename(continuous_path),
                    cue_tracks,
                    project.sample_rate() or 44100,
                    album=project.metadata.album or project.name,
                    artist=project.metadata.artist,
                    genre=project.metadata.genre,
                    date=project.metadata.date,
                    comment=project.metadata.comments,
                )
                report.cue_path = cue_path
                report.notes.append(
                    "CUE timecodes are quantised to 75 frames per second; the residual "
                    "error is at most 13.3 ms and is not corrected in the sheet."
                )

        # ---- M3U --------------------------------------------------------
        if settings.include_m3u and report.files:
            m3u_path = os.path.join(output_dir, "playlist.m3u")
            with open(m3u_path, "w", encoding="utf-8") as handle:
                handle.write("#EXTM3U\n")
                for path in report.files:
                    handle.write(f"#EXTINF:-1,{os.path.splitext(os.path.basename(path))[0]}\n")
                    handle.write(os.path.basename(path) + "\n")
            report.m3u_path = m3u_path

        # ---- CUE for the separate files --------------------------------
        if settings.mode == "separate" and settings.include_cue and report.files:
            from ..audio.encode import write_split_cue

            cue_path = os.path.join(output_dir, "tracks.cue")
            write_split_cue(
                cue_path,
                [
                    (
                        os.path.basename(item.path),
                        item.title,
                        project.effective_metadata(item.track).artist,
                    )
                    for item in items
                ],
                album=project.metadata.album or project.name,
                artist=project.metadata.artist,
                genre=project.metadata.genre,
                date=project.metadata.date,
                comment=project.metadata.comments,
            )
            report.cue_path = cue_path

        # ---- submarkers for songs inside a number -----------------------
        if getattr(settings, "export_submarkers", False):
            components = [
                component
                for track in project.sorted_tracks()
                for component in getattr(track, "components", [])
            ]
            if components:
                report_progress(0.96, "Writing the submarkers")
                submarker_path = os.path.join(output_dir, "submarkers.json")
                write_submarkers(project, submarker_path)
                report.submarker_path = submarker_path
                report.notes.append(
                    f"{len(components)} song(s) found inside the numbers were written "
                    "as submarkers. A number keeps its own title; the songs inside it "
                    "are listed separately."
                )

        # ---- info file --------------------------------------------------
        if settings.include_info:
            report_progress(0.95, "Writing the info file")
            info_path = os.path.join(output_dir, "spleynt_info.txt")
            write_info_file(
                info_path,
                project,
                items,
                report,
                section_policy=section_policy,
                gain_description=report.gain_description,
            )
            report.info_path = info_path

        # ---- checksums --------------------------------------------------
        if settings.include_checksums and report.files:
            report_progress(0.98, "Calculating checksums")
            checksum_path = os.path.join(output_dir, "checksums.sha256")
            with open(checksum_path, "w", encoding="utf-8") as handle:
                for path in report.files:
                    handle.write(f"{sha256_file(path)}  {os.path.basename(path)}\n")
            report.checksum_path = checksum_path

        # ---- verify the lossless output --------------------------------
        if settings.container == "flac" and report.files:
            report_progress(0.99, "Verifying the exported files")
            failures = []
            for path in report.files:
                ok, explanation = verify_lossless_export(path)
                if not ok:
                    failures.append(explanation)
                else:
                    report.notes.append(explanation)
            report.verified = not failures
            if failures:
                report.verified = False
                report.verification_note = (
                    f"{len(failures)} of {len(report.files)} file(s) failed verification"
                )
                report.warnings.extend(failures)
            else:
                report.verification_note = (
                    f"all {len(report.files)} exported file(s) decode cleanly"
                )

        report_progress(1.0, "Export complete")
    finally:
        builder.close()

    report.elapsed_seconds = time.time() - started
    return report


def _db_to_linear(db: float) -> float:
    if db <= -120.0:
        return 0.0
    return 10.0 ** (db / 20.0)


def _sanitise_filename(name: str) -> str:
    from ..core.units import safe_filename

    return safe_filename(name, fallback="track")


def _write_tags(container: str, path: str, project: Project, track: Track, number: int) -> None:
    extra = {"TRACKNUMBER": str(number)}
    if track.title:
        extra.setdefault("TITLE", track.title)
    tags = build_tags(project, track, extra)
    if not tags:
        return
    try:
        if container == "flac":
            from ..audio.tags.flac import write_vorbis_comments

            write_vorbis_comments(path, tags)
        elif container == "mp3":
            from ..audio.tags.id3v2 import write_id3v2

            write_id3v2(path, tags)
        else:
            from ..audio.tags.riff import write_riff_info

            write_riff_info(path, tags)
    except Exception as exc:
        raise RuntimeError(f"Could not write tags to {os.path.basename(path)}: {exc}") from exc


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# Archival info file
# ----------------------------------------------------------------------
def write_info_file(
    path: str,
    project: Project,
    items: list[ExportItem],
    report: ExportReport,
    section_policy: str = "keep_as_own",
    gain_description: str = "",
) -> str:
    """Write the plain-text archival / sharing info file."""
    sample_rate = project.sample_rate() or 44100
    meta = project.metadata
    lines: list[str] = []
    add = lines.append

    add("SplytNyt export information")
    add("=" * 60)
    add("")
    add(f"Project: {project.name}")
    add(f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    add("")

    add("Performer / production")
    add("-" * 60)
    performer = meta.performer or meta.artist or meta.production
    add(f"  {performer or '(not specified)'}")
    if meta.production:
        add(f"  Production: {meta.production}")
    if meta.act:
        add(f"  Act: {meta.act}")
    if meta.album:
        add(f"  Album: {meta.album}")
    add("")

    add("Event details")
    add("-" * 60)
    add(f"  Date: {meta.performance_date or meta.date or '(not specified)'}")
    add(f"  Venue: {meta.venue or '(not specified)'}")
    location = ", ".join(part for part in (meta.city, meta.country) if part)
    add(f"  Location: {location or '(not specified)'}")
    add(f"  Tour: {meta.tour or '(not specified)'}")
    if meta.source_notes:
        add(f"  Source notes: {meta.source_notes}")
    add("")

    add("Tracklist")
    add("-" * 60)
    total_seconds = 0.0
    for index, item in enumerate(items, start=1):
        track = item.track
        seconds = track.frames / sample_rate if sample_rate else 0.0
        total_seconds += seconds
        label = item.title or f"Track {index:02d}"
        add(
            f"  {index:02d}. {label}"
            f"   [{frames_to_timestamp(track.start_frame, sample_rate)} - "
            f"{frames_to_timestamp(track.end_frame, sample_rate)}]"
        )
        add(f"      duration: {seconds:.2f} s ({int(track.frames)} frames)")
        details = []
        if track.section_type:
            details.append(f"section: {track.section_type}")
        if track.boundary_confidence:
            details.append(f"boundary confidence: {track.boundary_confidence:.0%}")
        if track.identity_confidence:
            details.append(f"identity confidence: {track.identity_confidence:.0%}")
        if track.worklist_index >= 0:
            details.append(f"matched tracklist entry #{track.worklist_index + 1}")
        if details:
            add("      " + "; ".join(details))
        if track.notes:
            add(f"      note: {track.notes}")
        if track.reasons:
            add(f"      boundary evidence: {'; '.join(track.reasons)}")
    add("")
    add(f"  Total runtime: {total_seconds:.2f} s ({int(total_seconds // 60)} min {total_seconds % 60:.0f} s)")
    add(f"  Track count: {len(items)}")
    add("")

    add("Source material")
    add("-" * 60)
    for index, source in enumerate(project.sources, start=1):
        add(f"  {index}. {os.path.basename(source.path)}")
        add(f"     format: {source.original_format} @ {source.sample_rate} Hz, {source.channels} ch")
        add(f"     frames: {source.frames}")
        if index > 1 and source.gap_frames:
            seconds = source.gap_frames / sample_rate
            if seconds > 0:
                add(f"     gap before this file: {seconds:.3f} s (preserved as silence)")
            else:
                add(f"     overlap before this file: {abs(seconds):.3f} s (preserved as-is)")
    add("")

    add("Equipment and lineage")
    add("-" * 60)
    add(f"  {meta.equipment or '(not supplied by the user)'}")
    add(f"  {meta.lineage or '(not supplied by the user)'}")
    add("")

    add("Processing actually applied")
    add("-" * 60)
    add("  Boundary detection: SplytNyt analysis (silence/noise-floor and musical-change cues)")
    add("  Non-destructive: the source recordings were read only and never modified")
    add(f"  Level handling: {gain_description or 'untouched'}")
    if gain_description:
        add("    (no limiter, compressor, denoiser or resampler was applied)")
    container = getattr(report, "container", "") or ""
    add(f"  Export format: {container}")
    add("  Sample rate and channel layout were preserved from the source")
    if report.notes:
        add("  Notes:")
        for note in report.notes:
            add(f"    - {note}")
    add("")

    add("Uncertainty, gaps and omissions")
    add("-" * 60)
    if report.omissions:
        for title, reason in report.omissions:
            add(f"  Omitted: {title or 'untitled'} - {reason}")
    else:
        add("  Nothing was omitted from this export.")
    uncertain = [
        item for item in items
        if item.track.boundary_confidence and item.track.boundary_confidence < 0.6
    ]
    if uncertain:
        add("  Low-confidence boundaries (check these by ear):")
        for item in uncertain:
            add(
                f"    - {item.title or 'untitled'} "
                f"({item.track.boundary_confidence:.0%} boundary confidence)"
            )
    unidentified = [item for item in items if not item.track.title.strip()]
    if unidentified:
        add(f"  {len(unidentified)} track(s) have no confirmed title.")
    add(f"  Section handling during export: {section_policy}")
    add("")

    add("Sharing")
    add("-" * 60)
    add("  This package was produced by SplytNyt. It makes no claim that it satisfies")
    add("  the rules of any particular sharing site. Check the site's terms yourself.")
    add("")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------
def verify_lossless_export(path: str, expected_frames: Optional[int] = None) -> tuple[bool, str]:
    """Decode an exported lossless file and confirm it is intact.

    Returns ``(ok, explanation)``. This checks that the file decodes cleanly
    and has the expected length; it does not claim the content is musically
    correct.
    """
    from ..audio.decode import AudioSource

    try:
        with AudioSource(path) as source:
            frames = source.frames
            total = 0
            for block in source.blocks():
                total += block.shape[0]
                if not np.isfinite(block).all():
                    return False, "the decoded audio contains non-finite samples"
            if expected_frames is not None and frames != expected_frames:
                return False, (
                    f"expected {expected_frames} frames, the file has {frames}"
                )
            if total != frames:
                return False, f"decoded {total} frames but the header says {frames}"
            return True, (
                f"{os.path.basename(path)} decodes cleanly: {frames} frames, "
                f"{source.channels} channel(s), {source.sample_rate} Hz"
            )
    except Exception as exc:
        return False, f"{os.path.basename(path)} could not be decoded: {exc}"


def verify_flac_files(paths: list[str]) -> list[tuple[str, bool, str]]:
    """Verify a list of FLAC exports."""
    return [(path, *verify_lossless_export(path)) for path in paths]
