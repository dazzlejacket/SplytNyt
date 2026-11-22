"""Bundled theatrical reference lists.

A *reference* is a documented list of the musical numbers of a production:
the titles, their order, which act they belong to, and - where a source
published them - approximate durations. It is **data**, not code, and it
lives in a JSON file beside this module so that the engine itself knows
nothing about any particular show.

The rules the reference obeys, and that this module enforces:

* **A reference suggests names and an order. It never establishes a
  timestamp and never proves that a number is in the recording.** The
  durations are approximate evidence, never cutting instructions.
* **The provenance is kept.** Every reference records where it came from,
  which production or version it describes, and when it was read. That
  provenance travels with the worklist into the project, so a reviewer can
  always see what a title was based on.
* **The user confirms and edits.** The worklist that comes out of here is
  an ordinary editable worklist. Nothing downstream treats it as truth.

The file format is described by ``schema`` in each file:

``schema``
    ``splytnyt-theater-reference/1``.
``reference_id``, ``title``, ``producer``
    Identification.
``versions``
    The production or revision the list describes. Each has an ``id``, a
    ``label`` and free-text ``notes``.
``sources``
    Where the information came from. Each has an ``id``, a ``label``, a
    ``url``, a ``type`` (``official``/``commercial``/``unofficial``) and the
    date it was read.
``act_break``
    Which positions sit either side of the interval, plus a note.
``items``
    The numbers. Each has ``position``, ``act``, ``title``, an optional
    ``duration_hint_seconds``, an optional ``album_track``, the ``versions``
    it belongs to, an optional ``notes`` and an optional ``contains`` list:
    the songs that live *inside* a medley.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from ..core.model import Worklist, WorklistItem, now_iso

#: The directory holding the bundled reference files.
REFERENCE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references")

#: The schema this module understands.
REFERENCE_SCHEMA = "splytnyt-theater-reference/1"

#: The provider name recorded on worklists built from a reference.
PROVIDER_REFERENCE = "theatre-reference"


@dataclass
class ReferenceSource:
    """One place a reference's information came from."""

    source_id: str
    label: str
    url: str = ""
    type: str = "unofficial"
    accessed: str = ""


@dataclass
class ReferenceVersion:
    """One production or revision a reference describes."""

    version_id: str
    label: str
    notes: str = ""


@dataclass
class ReferenceItem:
    """One number in a reference list."""

    position: int
    act: int
    title: str
    #: Approximate duration in seconds. Evidence, never a cutting instruction.
    duration_hint_seconds: float = 0.0
    album_track: int = 0
    versions: list[str] = field(default_factory=list)
    notes: str = ""
    #: Songs that live inside this number (a medley).
    contains: list[str] = field(default_factory=list)


@dataclass
class TheaterReference:
    """A complete reference document."""

    reference_id: str
    title: str
    producer: str = ""
    path: str = ""
    versions: list[ReferenceVersion] = field(default_factory=list)
    sources: list[ReferenceSource] = field(default_factory=list)
    items: list[ReferenceItem] = field(default_factory=list)
    act_break: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def source(self, source_id: str) -> Optional[ReferenceSource]:
        for candidate in self.sources:
            if candidate.source_id == source_id:
                return candidate
        return None

    def version(self, version_id: str) -> Optional[ReferenceVersion]:
        for candidate in self.versions:
            if candidate.version_id == version_id:
                return candidate
        return None

    def items_for_version(self, version_id: str) -> list[ReferenceItem]:
        """The numbers that belong to one version, in order."""
        chosen = [
            item
            for item in self.items
            if not item.versions or version_id in item.versions
        ]
        return sorted(chosen, key=lambda item: item.position)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def reference_files(directory: Optional[str] = None) -> list[str]:
    """Every bundled reference file, sorted by name."""
    directory = directory or REFERENCE_DIRECTORY
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".json")
    )


def load_reference(path: str) -> TheaterReference:
    """Read one reference file."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    schema = data.get("schema", "")
    if schema != REFERENCE_SCHEMA:
        raise ValueError(
            f"{os.path.basename(path)} has schema '{schema}', "
            f"but this version of SplytNyt understands '{REFERENCE_SCHEMA}'."
        )

    act_break = dict(data.get("act_break") or {})
    return TheaterReference(
        reference_id=data.get("reference_id", ""),
        title=data.get("title", ""),
        producer=data.get("producer", ""),
        path=os.path.abspath(path),
        versions=[
            ReferenceVersion(
                version_id=entry.get("id", ""),
                label=entry.get("label", ""),
                notes=entry.get("notes", ""),
            )
            for entry in data.get("versions", [])
        ],
        sources=[
            ReferenceSource(
                source_id=entry.get("id", ""),
                label=entry.get("label", ""),
                url=entry.get("url", ""),
                type=entry.get("type", "unofficial"),
                accessed=entry.get("accessed", ""),
            )
            for entry in data.get("sources", [])
        ],
        items=[
            ReferenceItem(
                position=int(entry.get("position", index + 1)),
                act=int(entry.get("act", 1)),
                title=entry.get("title", ""),
                duration_hint_seconds=float(entry.get("duration_hint_seconds", 0.0)),
                album_track=int(entry.get("album_track", 0)),
                versions=list(entry.get("versions", [])),
                notes=entry.get("notes", ""),
                contains=list(entry.get("contains", [])),
            )
            for index, entry in enumerate(data.get("items", []))
        ],
        act_break=act_break,
        notes=list(data.get("notes", [])),
    )


def list_references(directory: Optional[str] = None) -> list[TheaterReference]:
    """Every bundled reference that can be read."""
    references: list[TheaterReference] = []
    for path in reference_files(directory):
        try:
            references.append(load_reference(path))
        except (OSError, ValueError, json.JSONDecodeError):
            # A broken reference file must never stop the application.
            continue
    return references


# ----------------------------------------------------------------------
# Turning a reference into a worklist
# ----------------------------------------------------------------------
def provenance_text(reference: TheaterReference, version_id: str = "") -> str:
    """A readable provenance block for the worklist."""
    lines = [
        f"Reference: {reference.title}"
        + (f" ({reference.producer})" if reference.producer else ""),
    ]
    if version_id:
        version = reference.version(version_id)
        if version:
            lines.append(f"Version: {version.label}")
            if version.notes:
                lines.append(version.notes)
    lines.append("")
    lines.append(
        "This list suggests names and an order. It does not establish any "
        "timestamp and it does not prove that a number is in this recording. "
        "The durations are approximate and are never cutting instructions."
    )
    if reference.act_break.get("note"):
        lines.append("")
        lines.append(f"Act break: {reference.act_break['note']}")
    lines.append("")
    lines.append("Sources:")
    for source in reference.sources:
        accessed = f", read {source.accessed}" if source.accessed else ""
        lines.append(f"  - [{source.type}] {source.label}{accessed}")
        if source.url:
            lines.append(f"    {source.url}")
    if reference.notes:
        lines.append("")
        lines.extend(reference.notes)
    return "\n".join(lines)


def build_worklist(
    reference: TheaterReference,
    version_id: str = "",
    act: Optional[int] = None,
) -> Worklist:
    """Turn a reference into an ordinary, editable worklist.

    ``act`` selects one act; ``None`` keeps every act, with the act number
    written into each entry's notes so the order stays unambiguous.
    """
    items = reference.items_for_version(version_id) if version_id else list(reference.items)
    worklist_items: list[WorklistItem] = []
    for item in items:
        if act is not None and item.act != act:
            continue
        notes = []
        if act is None:
            notes.append(f"Act {item.act}")
        if item.album_track:
            notes.append(f"album track {item.album_track}")
        if item.notes:
            notes.append(item.notes)
        if item.contains:
            notes.append("contains: " + ", ".join(item.contains))
        worklist_items.append(
            WorklistItem(
                title=item.title,
                duration_hint=item.duration_hint_seconds,
                notes=" - ".join(notes),
                position=item.position,
                set_number=f"Act {item.act}",
                provider=PROVIDER_REFERENCE,
                provider_id=f"{reference.reference_id}:{version_id or 'all'}",
                contains=list(item.contains),
            )
        )

    name = reference.title
    if version_id:
        version = reference.version(version_id)
        if version:
            name = f"{reference.title} - {version.label}"
    if act is not None:
        name = f"{name} - Act {act}"

    return Worklist(
        name=name,
        items=worklist_items,
        provider=PROVIDER_REFERENCE,
        provider_id=f"{reference.reference_id}:{version_id or 'all'}",
        fetched_at=now_iso(),
        mode="theater",
        notes=provenance_text(reference, version_id),
    )


def component_titles(reference: TheaterReference, item_title: str) -> list[str]:
    """The songs a reference says live inside one number."""
    for item in reference.items:
        if item.title == item_title:
            return list(item.contains)
    return []
