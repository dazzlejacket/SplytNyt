"""Review flags and issue detection.

These are the objective, checkable things the review UI surfaces:

* the track count disagrees with the active worklist,
* a track is suspiciously short or long,
* a track crosses a source-file boundary (usually a recorder file split
  mid-song, which is legitimate but must be visible),
* a track spans a gap between source files (the export will omit that gap, so
  it must never be silent),
* a source file is missing or has changed since it was imported.

"Short" and "long" thresholds are heuristics informed by the active worklist
when there is one (a track far shorter than the list's shortest entry is
probably a mis-cut), and fall back to absolute durations otherwise.
"""

from __future__ import annotations

from .model import (  # noqa: F401
    FLAG_CROSSES_SOURCE,
    FLAG_OVERLAP_BETWEEN_SOURCES,
    FLAG_SPANS_GAP,
    FLAG_SUSPICIOUSLY_LONG,
    FLAG_SUSPICIOUSLY_SHORT,
    FLAG_TRACK_COUNT_MISMATCH,
    FLAG_UNIDENTIFIED,
    Project,
    Track,
)

#: Absolute thresholds used when no worklist hints are available (seconds).
DEFAULT_SHORT_SECONDS = 10.0
DEFAULT_LONG_SECONDS = 900.0


def _worklist_durations(project: Project) -> list[float]:
    worklist = project.active_worklist()
    if not worklist:
        return []
    return [item.duration_hint for item in worklist.items if item.duration_hint > 0]


def compute_flags(project: Project) -> dict[str, list[str]]:
    """Recompute every flag and return ``{track_id: [flag, ...]}``."""
    sample_rate = project.sample_rate() or 44100
    flags: dict[str, list[str]] = {}
    hints = _worklist_durations(project)

    worklist = project.active_worklist()
    if worklist and worklist.items:
        if len(worklist.items) != len(project.tracks):
            for track in project.tracks:
                flags.setdefault(track.track_id, []).append(FLAG_TRACK_COUNT_MISMATCH)

    short_limit = DEFAULT_SHORT_SECONDS
    long_limit = DEFAULT_LONG_SECONDS
    if hints:
        shortest = min(hints)
        longest = max(hints)
        # A cut is suspicious when it is dramatically shorter/longer than
        # anything the list suggests.
        short_limit = max(DEFAULT_SHORT_SECONDS, shortest * 0.35)
        long_limit = max(DEFAULT_LONG_SECONDS, longest * 2.0)

    gap_regions = project.gap_regions()
    source_starts = project.source_starts()

    for track in project.tracks:
        track_flags: list[str] = []
        duration = track.duration_seconds(sample_rate)

        if duration < short_limit:
            track_flags.append(FLAG_SUSPICIOUSLY_SHORT)
        if duration > long_limit:
            track_flags.append(FLAG_SUSPICIOUSLY_LONG)

        # Does the track cross a source boundary?
        boundaries = [start for start in source_starts.values() if start > 0]
        for boundary in boundaries:
            if track.start_frame < boundary < track.end_frame:
                track_flags.append(FLAG_CROSSES_SOURCE)
                break

        # Does the track span a gap between source files?
        for gap_start, gap_end in gap_regions:
            if track.start_frame < gap_end and gap_start < track.end_frame:
                track_flags.append(FLAG_SPANS_GAP)
                break

        if not track.title.strip() or track.identity_confidence <= 0.0:
            track_flags.append(FLAG_UNIDENTIFIED)

        flags[track.track_id] = track_flags

    return flags


def apply_flags(project: Project) -> None:
    """Recompute flags and store them on the tracks (used by the app layer)."""
    computed = compute_flags(project)
    for track in project.tracks:
        track.flags = list(computed.get(track.track_id, []))


def source_issues(project: Project) -> list[str]:
    """Human-readable problems with the imported sources."""
    issues: list[str] = []
    for source in project.sources:
        name = source.path or "(unnamed)"
        if source.missing:
            issues.append(f"Source missing: {name}")
        else:
            import os

            if not os.path.exists(source.path):
                issues.append(f"Source file not found: {name}")
    for index, source in enumerate(project.sources):
        if index > 0 and source.gap_frames > 0:
            rate = project.sample_rate() or 44100
            issues.append(
                f"Gap of {source.gap_frames / rate:.3f} s before source {index + 1}: "
                f"{os.path.basename(source.path) if source.path else 'unnamed'}"
            )
        if index > 0 and source.gap_frames < 0:
            rate = project.sample_rate() or 44100
            issues.append(
                f"Overlap of {abs(source.gap_frames) / rate:.3f} s before source {index + 1}: "
                f"{os.path.basename(source.path) if source.path else 'unnamed'}"
            )
    return issues


def describe_flag(flag: str) -> str:
    from .model import FLAG_LABELS

    return FLAG_LABELS.get(flag, flag.replace("_", " ").capitalize())


def track_summary(project: Project) -> dict:
    """Small summary used by the status bar and the export report."""
    sample_rate = project.sample_rate() or 44100
    total = project.total_frames()
    durations = [track.duration_seconds(sample_rate) for track in project.tracks]
    identified = sum(1 for track in project.tracks if track.title.strip())
    return {
        "source_count": len(project.sources),
        "track_count": len(project.tracks),
        "total_seconds": total / sample_rate if sample_rate else 0.0,
        "longest_seconds": max(durations) if durations else 0.0,
        "shortest_seconds": min(durations) if durations else 0.0,
        "identified_tracks": identified,
        "unidentified_tracks": len(project.tracks) - identified,
    }


def sorted_tracks_by_start(project: Project) -> list[Track]:
    return sorted(project.tracks, key=lambda t: (t.start_frame, t.end_frame))
