"""Project persistence and marker interchange.

The project file is a ZIP container (``.splytnyt``) holding:

    project.json      the versioned model document
    peaks/            cached waveform peak pyramids (optional, regenerable)
    README.txt        a one-paragraph plain-text note about the format

A plain ``.json`` file with the same ``project.json`` payload is also accepted
on open, which makes the format easy to inspect and to script against.
"""

from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from typing import Optional

from .model import (
    DEFAULT_PRE_ROLL_SECONDS,
    SCHEMA_VERSION,
    ActivityRegion,
    ComponentSong,
    ExportSettings,
    Marker,
    Metadata,
    Project,
    Source,
    Track,
    Worklist,
    WorklistItem,
    now_iso,
)

PROJECT_SUFFIX = ".splytnyt"
MARKER_JSON_SUFFIX = ".splytnyt-markers.json"


# ----------------------------------------------------------------------
# Model <-> dict
# ----------------------------------------------------------------------
def _metadata_to_dict(meta: Metadata) -> dict:
    return {name: getattr(meta, name) for name in Metadata._TAGGED + ("artwork_path",)}


def _metadata_from_dict(data: dict) -> Metadata:
    meta = Metadata()
    for name in Metadata._TAGGED + ("artwork_path",):
        if name in data and data[name] is not None:
            setattr(meta, name, data[name])
    return meta


def project_to_dict(project: Project) -> dict:
    """Serialise the model to a JSON-compatible dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "app": "SplytNyt",
        "project_id": project.project_id,
        "name": project.name,
        "created": project.created,
        "modified": project.modified,
        "notes": project.notes,
        "active_worklist_id": project.active_worklist_id,
        "placement_mode": project.placement_mode,
        "placement_pre_roll_seconds": project.placement_pre_roll_seconds,
        "metadata": _metadata_to_dict(project.metadata),
        "export_settings": {
            key: getattr(project.export_settings, key)
            for key in vars(project.export_settings)
        },
        "sources": [
            {
                "source_id": s.source_id,
                "path": s.path,
                "original_format": s.original_format,
                "sample_rate": s.sample_rate,
                "channels": s.channels,
                "frames": s.frames,
                "size_bytes": s.size_bytes,
                "mtime_ns": s.mtime_ns,
                "gap_frames": s.gap_frames,
                "head_hash": s.head_hash,
                "tail_hash": s.tail_hash,
                "missing": s.missing,
                "label": s.label,
            }
            for s in project.sources
        ],
        "tracks": [
            {
                "track_id": t.track_id,
                "start_frame": t.start_frame,
                "end_frame": t.end_frame,
                "title": t.title,
                "section_type": t.section_type,
                "boundary_confidence": t.boundary_confidence,
                "identity_confidence": t.identity_confidence,
                "classification_confidence": t.classification_confidence,
                "reasons": list(t.reasons),
                "locked": t.locked,
                "flags": list(t.flags),
                "metadata": _metadata_to_dict(t.metadata),
                "worklist_id": t.worklist_id,
                "worklist_index": t.worklist_index,
                "notes": t.notes,
                "suggested": t.suggested,
                "manual_override": t.manual_override,
                "recognized_artist": t.recognized_artist,
                "recognized_title": t.recognized_title,
                "recognition_status": t.recognition_status,
                "recognition_engines": list(t.recognition_engines),
                "boundary_basis": t.boundary_basis,
                "identity_basis": t.identity_basis,
                "boundary_placement": t.boundary_placement,
                "onset_frame": t.onset_frame,
                "boundary_pre_roll_seconds": t.boundary_pre_roll_seconds,
                "transition_start": t.transition_start,
                "transition_end": t.transition_end,
                "boundary_uncertain": t.boundary_uncertain,
                "activities": list(t.activities),
                "components": [
                    {
                        "component_id": c.component_id,
                        "title": c.title,
                        "start_frame": c.start_frame,
                        "end_frame": c.end_frame,
                        "confidence": c.confidence,
                        "basis": c.basis,
                        "reason": c.reason,
                    }
                    for c in t.components
                ],
            }
            for t in project.tracks
        ],
        "activities": [
            {
                "activity_id": a.activity_id,
                "start_frame": a.start_frame,
                "end_frame": a.end_frame,
                "label": a.label,
                "confidence": a.confidence,
                "basis": a.basis,
                "reason": a.reason,
                "evidence": dict(a.evidence),
            }
            for a in project.activities
        ],
        "worklists": [
            {
                "worklist_id": w.worklist_id,
                "name": w.name,
                "provider": w.provider,
                "provider_id": w.provider_id,
                "fetched_at": w.fetched_at,
                "mode": w.mode,
                "notes": w.notes,
                "items": [
                    {
                        "title": i.title,
                        "duration_hint": i.duration_hint,
                        "notes": i.notes,
                        "position": i.position,
                        "set_number": i.set_number,
                        "provider": i.provider,
                        "provider_id": i.provider_id,
                        "contains": list(i.contains),
                    }
                    for i in w.items
                ],
            }
            for w in project.worklists
        ],
        "markers": [
            {
                "marker_id": m.marker_id,
                "frame": m.frame,
                "label": m.label,
                "notes": m.notes,
                "color": m.color,
            }
            for m in project.markers
        ],
        "cache": project.cache,
    }


def project_from_dict(data: dict) -> Project:
    """Rebuild a project from :func:`project_to_dict` output."""
    if int(data.get("schema_version", 0)) > SCHEMA_VERSION:
        raise ValueError(
            "This project was written by a newer version of SplytNyt "
            f"(schema {data.get('schema_version')} > {SCHEMA_VERSION})."
        )
    project = Project()
    project.project_id = data.get("project_id", project.project_id)
    project.name = data.get("name", project.name)
    project.created = data.get("created", project.created)
    project.modified = data.get("modified", project.modified)
    project.notes = data.get("notes", "")
    project.active_worklist_id = data.get("active_worklist_id", "")
    project.placement_mode = data.get("placement_mode", "next_start")
    project.placement_pre_roll_seconds = float(
        data.get("placement_pre_roll_seconds", DEFAULT_PRE_ROLL_SECONDS)
    )
    project.metadata = _metadata_from_dict(data.get("metadata", {}))

    settings = ExportSettings()
    for key, value in (data.get("export_settings") or {}).items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    project.export_settings = settings

    project.sources = [
        Source(
            source_id=s.get("source_id", Source().source_id),
            path=s.get("path", ""),
            original_format=s.get("original_format", ""),
            sample_rate=int(s.get("sample_rate", 0)),
            channels=int(s.get("channels", 0)),
            frames=int(s.get("frames", 0)),
            size_bytes=int(s.get("size_bytes", 0)),
            mtime_ns=int(s.get("mtime_ns", 0)),
            gap_frames=int(s.get("gap_frames", 0)),
            head_hash=s.get("head_hash", ""),
            tail_hash=s.get("tail_hash", ""),
            missing=bool(s.get("missing", False)),
            label=s.get("label", ""),
        )
        for s in data.get("sources", [])
    ]

    project.tracks = [
        Track(
            track_id=t.get("track_id", Track().track_id),
            start_frame=int(t.get("start_frame", 0)),
            end_frame=int(t.get("end_frame", 0)),
            title=t.get("title", ""),
            section_type=t.get("section_type", "unknown"),
            boundary_confidence=float(t.get("boundary_confidence", 0.0)),
            identity_confidence=float(t.get("identity_confidence", 0.0)),
            classification_confidence=float(t.get("classification_confidence", 0.0)),
            reasons=list(t.get("reasons", [])),
            locked=bool(t.get("locked", False)),
            flags=list(t.get("flags", [])),
            metadata=_metadata_from_dict(t.get("metadata", {})),
            worklist_id=t.get("worklist_id", ""),
            worklist_index=int(t.get("worklist_index", -1)),
            notes=t.get("notes", ""),
            suggested=bool(t.get("suggested", False)),
            manual_override=bool(t.get("manual_override", False)),
            recognized_artist=str(t.get("recognized_artist", "") or ""),
            recognized_title=str(t.get("recognized_title", "") or ""),
            recognition_status=str(t.get("recognition_status", "") or ""),
            recognition_engines=list(t.get("recognition_engines", []) or []),
            boundary_basis=t.get("boundary_basis", "none"),
            identity_basis=t.get("identity_basis", "none"),
            boundary_placement=t.get("boundary_placement", ""),
            onset_frame=int(t.get("onset_frame", -1)),
            boundary_pre_roll_seconds=float(t.get("boundary_pre_roll_seconds", -1.0)),
            transition_start=int(t.get("transition_start", -1)),
            transition_end=int(t.get("transition_end", -1)),
            boundary_uncertain=bool(t.get("boundary_uncertain", False)),
            activities=list(t.get("activities", [])),
            components=[
                ComponentSong(
                    component_id=c.get("component_id", ComponentSong().component_id),
                    title=c.get("title", ""),
                    start_frame=int(c.get("start_frame", 0)),
                    end_frame=int(c.get("end_frame", 0)),
                    confidence=float(c.get("confidence", 0.0)),
                    basis=c.get("basis", ""),
                    reason=c.get("reason", ""),
                )
                for c in t.get("components", [])
            ],
        )
        for t in data.get("tracks", [])
    ]

    project.activities = [
        ActivityRegion(
            activity_id=a.get("activity_id", ActivityRegion().activity_id),
            start_frame=int(a.get("start_frame", 0)),
            end_frame=int(a.get("end_frame", 0)),
            label=a.get("label", "speech"),
            confidence=float(a.get("confidence", 0.0)),
            basis=a.get("basis", "acoustic"),
            reason=a.get("reason", ""),
            evidence=dict(a.get("evidence", {}) or {}),
        )
        for a in data.get("activities", [])
    ]

    project.worklists = [
        Worklist(
            worklist_id=w.get("worklist_id", Worklist().worklist_id),
            name=w.get("name", "Tracklist"),
            provider=w.get("provider", "manual"),
            provider_id=w.get("provider_id", ""),
            fetched_at=w.get("fetched_at", ""),
            mode=w.get("mode", "unknown"),
            notes=w.get("notes", ""),
            items=[
                WorklistItem(
                    title=i.get("title", ""),
                    duration_hint=float(i.get("duration_hint", 0.0)),
                    notes=i.get("notes", ""),
                    position=int(i.get("position", index)),
                    set_number=i.get("set_number", ""),
                    provider=i.get("provider", ""),
                    provider_id=i.get("provider_id", ""),
                    contains=list(i.get("contains", [])),
                )
                for index, i in enumerate(w.get("items", []))
            ],
        )
        for w in data.get("worklists", [])
    ]

    project.markers = [
        Marker(
            marker_id=m.get("marker_id", Marker().marker_id),
            frame=int(m.get("frame", 0)),
            label=m.get("label", ""),
            notes=m.get("notes", ""),
            color=m.get("color", "#e0b040"),
        )
        for m in data.get("markers", [])
    ]
    project.cache = data.get("cache", {}) or {}
    return project


# ----------------------------------------------------------------------
# Project container
# ----------------------------------------------------------------------
PROJECT_README = """SplytNyt project file
=====================

This is a ZIP container. It contains:

  project.json   the versioned project document (JSON, UTF-8)
  peaks/         cached waveform peak pyramids (safe to delete; regenerated)

The project references your original audio files by absolute path. The audio
itself is never modified or embedded unless you explicitly chose to store
copies inside the container.

Move a project together with its source files, or re-link the sources when
opening it on another machine.
"""


def save_project(project: Project, path: str, include_peaks: bool = True) -> str:
    """Write the project to ``path`` atomically.

    Writes to a temporary file in the same directory and renames on success,
    so an interrupted save never destroys the previous project file.
    """
    project.modified = now_iso()
    payload = json.dumps(project_to_dict(project), indent=2, ensure_ascii=False)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = path + ".partial"

    with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", payload.encode("utf-8"))
        archive.writestr("README.txt", PROJECT_README.encode("utf-8"))
        if include_peaks:
            for source in project.sources:
                cache_path = peaks_cache_path(project, source)
                if cache_path and os.path.exists(cache_path):
                    archive.write(cache_path, f"peaks/{os.path.basename(cache_path)}")

    os.replace(temp_path, path)
    project.path = os.path.abspath(path)
    return path


def load_project(path: str) -> Project:
    """Load a project from a ``.splytnyt`` container or a bare JSON file."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if "project.json" not in names:
                raise ValueError(f"{path} is not a SplytNyt project (no project.json).")
            with archive.open("project.json") as handle:
                payload = json.loads(handle.read().decode("utf-8"))
            project = project_from_dict(payload)
            # Restore the peak caches next to the project file.
            for name in names:
                if name.startswith("peaks/") and not name.endswith("/"):
                    target = os.path.join(os.path.dirname(os.path.abspath(path)), "peaks",
                                          os.path.basename(name))
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with archive.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
    else:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        project = project_from_dict(payload)

    project.path = os.path.abspath(path)
    project.modified = now_iso()
    return project


def peaks_cache_path(project: Project, source: Source) -> Optional[str]:
    """Where the peak cache for ``source`` lives (inside the project folder)."""
    if not project.path:
        return None
    base = os.path.join(os.path.dirname(os.path.abspath(project.path)), "peaks")
    return os.path.join(base, f"{source.source_id}.peaks")


# ----------------------------------------------------------------------
# Marker interchange
# ----------------------------------------------------------------------
def markers_to_json(project: Project) -> str:
    """Export markers/segments in the documented JSON format."""
    sample_rate = project.sample_rate() or 44100
    starts = project.source_starts()
    payload = {
        "format": "splytnyt-markers",
        "version": 1,
        "project": project.name,
        "sample_rate": sample_rate,
        "total_frames": project.total_frames(),
        "sources": [
            {
                "source_id": s.source_id,
                "path": s.path,
                "timeline_start_frame": starts.get(s.source_id, 0),
                "frames": s.frames,
                "sample_rate": s.sample_rate,
            }
            for s in project.sources
        ],
        "segments": [
            {
                "track_id": t.track_id,
                "start_frame": t.start_frame,
                "end_frame": t.end_frame,
                "start_seconds": round(t.start_frame / sample_rate, 6) if sample_rate else 0.0,
                "end_seconds": round(t.end_frame / sample_rate, 6) if sample_rate else 0.0,
                "duration_seconds": round(t.frames / sample_rate, 6) if sample_rate else 0.0,
                "title": t.title,
                "section_type": t.section_type,
                "boundary_confidence": t.boundary_confidence,
                "identity_confidence": t.identity_confidence,
            }
            for t in project.sorted_tracks()
        ],
        "markers": [
            {
                "frame": m.frame,
                "seconds": round(m.frame / sample_rate, 6) if sample_rate else 0.0,
                "label": m.label,
                "notes": m.notes,
            }
            for m in sorted(project.markers, key=lambda m: m.frame)
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def markers_to_csv(project: Project) -> str:
    """Export segments as CSV (auditable in a spreadsheet)."""
    sample_rate = project.sample_rate() or 44100
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "index", "title", "start_frame", "end_frame",
            "start_seconds", "end_seconds", "duration_seconds",
            "section_type", "boundary_confidence", "identity_confidence", "notes",
        ]
    )
    for index, track in enumerate(project.sorted_tracks(), start=1):
        writer.writerow(
            [
                index,
                track.title,
                track.start_frame,
                track.end_frame,
                f"{track.start_frame / sample_rate:.6f}" if sample_rate else "0",
                f"{track.end_frame / sample_rate:.6f}" if sample_rate else "0",
                f"{track.frames / sample_rate:.6f}" if sample_rate else "0",
                track.section_type,
                f"{track.boundary_confidence:.3f}",
                f"{track.identity_confidence:.3f}",
                track.notes,
            ]
        )
    return buffer.getvalue()


def markers_to_audacity(project: Project) -> str:
    """Export segments in Audacity's label format (``start\\tend\\tlabel``)."""
    sample_rate = project.sample_rate() or 44100
    lines = []
    for track in project.sorted_tracks():
        start = track.start_frame / sample_rate if sample_rate else 0.0
        end = track.end_frame / sample_rate if sample_rate else 0.0
        label = track.title or track.section_type or "Track"
        lines.append(f"{start:.6f}\t{end:.6f}\t{label}")
    return "\n".join(lines) + ("\n" if lines else "")


def markers_from_audacity(text: str, sample_rate: int) -> list[tuple[int, int, str]]:
    """Parse Audacity labels into ``(start_frame, end_frame, label)`` tuples."""
    from .units import parse_timestamp

    results: list[tuple[int, int, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            start = parse_timestamp(parts[0])
            end = parse_timestamp(parts[1])
        except ValueError:
            continue
        label = parts[2] if len(parts) > 2 else ""
        results.append(
            (
                int(round(start * sample_rate)),
                int(round(end * sample_rate)),
                label.strip(),
            )
        )
    return results


def markers_from_json(text: str) -> list[tuple[int, int, str]]:
    """Parse the documented marker JSON format into segment tuples."""
    data = json.loads(text)
    sample_rate = int(data.get("sample_rate", 44100))
    results: list[tuple[int, int, str]] = []
    for segment in data.get("segments", []):
        start = int(segment.get("start_frame", 0))
        end = int(segment.get("end_frame", 0))
        if "start_frame" not in segment and "start_seconds" in segment:
            start = int(round(float(segment["start_seconds"]) * sample_rate))
            end = int(round(float(segment.get("end_seconds", 0)) * sample_rate))
        results.append((start, end, str(segment.get("title", ""))))
    return results


def markers_from_csv(text: str, sample_rate: int) -> list[tuple[int, int, str]]:
    """Parse exported CSV back into segment tuples."""
    results: list[tuple[int, int, str]] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        if not row.get("start_frame"):
            continue
        try:
            start = int(row["start_frame"])
            end = int(row["end_frame"])
        except (TypeError, ValueError):
            continue
        results.append((start, end, (row.get("title") or "").strip()))
    return results


# ----------------------------------------------------------------------
# Tracklist pasting
# ----------------------------------------------------------------------
def parse_tracklist(text: str) -> list[WorklistItem]:
    """Parse a pasted tracklist into worklist items.

    Accepts one track per line. A leading number, ``1.``, ``1)`` or ``01 -``
    is treated as a position prefix and removed. A trailing ``(4:32)`` or
    ``[4:32]`` is read as a duration hint in minutes:seconds. Everything
    else is kept as the title, including artist prefixes.
    """
    from .units import parse_timestamp

    items: list[WorklistItem] = []
    for index, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        notes = ""
        if " - " in line and line.split(" - ", 1)[0].strip().isdigit():
            line = line.split(" - ", 1)[1].strip()
        # Strip a leading index.
        for prefix in ("#",):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].rstrip(".)").isdigit():
            line = parts[1].strip()
        elif parts and parts[0].rstrip(".)").isdigit() and len(parts) == 1:
            line = parts[0].rstrip(".)").strip()
        # Trailing duration hint.
        duration = 0.0
        for opener, closer in (("(", ")"), ("[", "]")):
            if line.endswith(closer) and opener in line:
                head, _, tail = line.rpartition(opener)
                candidate = tail.rstrip(closer).strip()
                try:
                    duration = parse_timestamp(candidate)
                except ValueError:
                    duration = 0.0
                if duration:
                    line = head.strip()
                break
        # Trailing free-text note after an em dash.
        if "—" in line:
            line, _, notes = line.partition("—")
            line = line.strip()
            notes = notes.strip()
        if not line:
            continue
        items.append(
            WorklistItem(
                title=line,
                duration_hint=duration,
                notes=notes,
                position=len(items),
                provider="manual",
            )
        )
    return items
