"""Application service layer.

:class:`Session` owns the project, the undo stack and every mutation the user
can perform. It knows nothing about Qt; the UI subscribes to plain callbacks
and calls these methods.

Two invariants are enforced here rather than in the UI:

* A mutation that removes a track does not remove audio. ``remove_track``
  merges into the neighbouring track so the source stays fully covered.
* Re-analysis never overwrites a locked track.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, Optional

from ..core import flags as flag_tools
from ..core import serialize
from ..core.model import (
    BOUNDARY_BASIS_ACCEPTED,
    BOUNDARY_BASIS_ASSESSED,
    BOUNDARY_BASIS_MANUAL,
    BOUNDARY_BASIS_NONE,
    Metadata,
    Project,
    Source,
    Track,
    Worklist,
    fingerprint_file,
    new_id,
    now_iso,
)
from ..core.undo import UndoStack
from .events import (
    BUSY,
    PROJECT_CHANGED,
    STATUS,
    TRACKS_CHANGED,
    Emitter,
)


class SessionError(RuntimeError):
    """Raised for operations the session refuses to perform."""


class Session:
    """Central controller for one open project."""

    def __init__(self) -> None:
        self.project = Project()
        self.undo = UndoStack(self.project)
        self.events = Emitter()
        self._recovery_path: str = ""
        self._dirty = False
        #: The transcript from the last theatrical analysis, kept for the
        #: naming step. It is evidence, not an answer, and it is not written
        #: into the project.
        self._theater_transcript: list[dict] = []
        self.undo.set_on_change(self._on_undo_change)

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------
    def on(self, event: str, handler: Callable[..., None]) -> Callable[..., None]:
        return self.events.on(event, handler)

    def status(self, message: str) -> None:
        self.events.emit(STATUS, message=message)

    def busy(self, active: bool, message: str = "") -> None:
        self.events.emit(BUSY, active=active, message=message)

    def _on_undo_change(self) -> None:
        self._dirty = True
        self.events.emit(TRACKS_CHANGED)
        self.events.emit(PROJECT_CHANGED)

    def _mutate(self, label: str, func: Callable[[Project], None]) -> None:
        """Run a project mutation as a single undoable step."""
        self.undo.apply(func, label)
        self._dirty = True

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------
    def _restore_theater_transcript_from_cache(self) -> None:
        cached = (self.project.cache or {}).get("theater_transcript")
        if isinstance(cached, dict) and isinstance(cached.get("segments"), list):
            self._theater_transcript = [
                dict(segment)
                for segment in cached["segments"]
                if segment.get("accepted", True)
            ]
        else:
            self._theater_transcript = []

    def new_project(self, name: str = "Untitled project") -> None:
        self.project = Project(name=name)
        self.undo = UndoStack(self.project)
        self.undo.set_on_change(self._on_undo_change)
        self._theater_transcript = []
        self._dirty = False
        self.events.emit(PROJECT_CHANGED)
        self.events.emit(TRACKS_CHANGED)
        self.status(f"New project: {name}")

    def open_project(self, path: str) -> None:
        project = serialize.load_project(path)
        self.project = project
        self.undo = UndoStack(self.project)
        self.undo.set_on_change(self._on_undo_change)
        self._mark_missing_sources()
        self._restore_theater_transcript_from_cache()
        self._dirty = False
        self.events.emit(PROJECT_CHANGED)
        self.events.emit(TRACKS_CHANGED)
        self.status(f"Opened {os.path.basename(path)}")

    def save_project(self, path: Optional[str] = None) -> str:
        target = path or self.project.path
        if not target:
            raise SessionError("No project path is set.")
        serialize.save_project(self.project, target)
        self._dirty = False
        self.events.emit(PROJECT_CHANGED)
        self.status(f"Saved {os.path.basename(target)}")
        return target

    def mark_dirty(self) -> None:
        self._dirty = True

    @property
    def dirty(self) -> bool:
        return self._dirty

    def _mark_missing_sources(self) -> None:
        for source in self.project.sources:
            source.missing = not os.path.exists(source.path)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------
    def import_files(self, paths: list[str]) -> list[Source]:
        """Import audio files into the timeline.

        The first import sets the timeline's sample rate and channel layout;
        later imports must match it (mixing formats would force a resample,
        which this application never does silently).
        """
        from ..audio.decode import AudioError, probe

        if not paths:
            return []

        infos = []
        for path in paths:
            try:
                infos.append(probe(path))
            except AudioError as exc:
                raise SessionError(str(exc)) from exc

        reference = self.project.sample_rate()
        reference_channels = self.project.channels()
        if reference:
            for info in infos:
                if info.sample_rate != reference:
                    raise SessionError(
                        f"{os.path.basename(info.path)} is {info.sample_rate} Hz but the "
                        f"project is {reference} Hz. SplytNyt never resamples "
                        "automatically - convert the file first."
                    )
                if info.channels != reference_channels:
                    raise SessionError(
                        f"{os.path.basename(info.path)} has {info.channels} channels but the "
                        f"project has {reference_channels}. SplytNyt never changes the "
                        "channel layout automatically."
                    )

        created: list[Source] = []
        for info in infos:
            size, mtime, head_hash, _ = fingerprint_file(info.path)
            source = Source(
                source_id=new_id("src"),
                path=info.path,
                original_format=info.format,
                sample_rate=info.sample_rate,
                channels=info.channels,
                frames=info.frames,
                size_bytes=size,
                mtime_ns=mtime,
                head_hash=head_hash,
                gap_frames=0,
                label=os.path.splitext(os.path.basename(info.path))[0],
            )

            def add(project: Project, source: Source = source) -> None:
                project.sources.append(source)
                project.touch()

            self._mutate(f"Import {os.path.basename(info.path)}", add)
            created.append(source)

        self._ensure_full_coverage()
        self.events.emit(TRACKS_CHANGED)
        self.events.emit(PROJECT_CHANGED)
        self.status(
            f"Imported {len(created)} file(s): "
            + ", ".join(os.path.basename(s.path) for s in created)
        )
        return created

    def _ensure_full_coverage(self) -> None:
        """Guarantee the tracks cover the whole timeline (no audio is ever lost)."""
        total = self.project.total_frames()
        if total <= 0:
            return
        tracks = self.project.sorted_tracks()
        if not tracks:
            self.project.tracks.append(
                Track(
                    start_frame=0,
                    end_frame=total,
                    title="Full recording",
                    section_type="unknown",
                    boundary_confidence=0.0,
                    boundary_basis=BOUNDARY_BASIS_NONE,
                )
            )
            return
        first = tracks[0]
        if first.start_frame > 0:
            first.start_frame = 0
        last = tracks[-1]
        if last.end_frame < total:
            last.end_frame = total

    def remove_source(self, source_id: str) -> None:
        def remove(project: Project) -> None:
            project.sources = [s for s in project.sources if s.source_id != source_id]
            project.touch()

        self._mutate("Remove source", remove)
        self.events.emit(TRACKS_CHANGED)

    def set_source_gap(self, source_id: str, gap_frames: int) -> None:
        """Set the gap (positive) or overlap (negative) before a source."""

        def change(project: Project) -> None:
            for source in project.sources:
                if source.source_id == source_id:
                    source.gap_frames = int(gap_frames)
                    break
            project.touch()

        self._mutate("Change source spacing", change)
        self.events.emit(TRACKS_CHANGED)

    # ------------------------------------------------------------------
    # Boundary placement
    # ------------------------------------------------------------------
    def placement_choice(self):
        """The project's placement preference as a placement choice."""
        from ..analysis.placement import PlacementChoice

        return PlacementChoice(
            mode=self.project.placement_mode,
            pre_roll_seconds=self.project.placement_pre_roll_seconds,
        )

    def set_placement(self, mode: str, pre_roll_seconds: Optional[float] = None) -> None:
        """Remember how new boundaries should be placed."""
        from ..analysis.placement import PlacementChoice

        choice = PlacementChoice(
            mode=mode,
            pre_roll_seconds=(
                self.project.placement_pre_roll_seconds
                if pre_roll_seconds is None
                else pre_roll_seconds
            ),
        ).normalised()

        def change(project: Project) -> None:
            project.placement_mode = choice.mode
            project.placement_pre_roll_seconds = choice.pre_roll_seconds
            project.touch()

        self._mutate("Change boundary placement", change)
        self.events.emit(PROJECT_CHANGED)

    def apply_refinement(self, proposals: Iterable) -> int:
        """Move the boundaries a refinement proposes.

        Locked tracks are never moved. The whole set is one undoable step,
        so a single undo puts every boundary back where it was.
        """
        proposals = list(proposals)
        applied = 0

        def change(project: Project) -> None:
            nonlocal applied
            for proposal in proposals:
                for index, track in enumerate(project.tracks):
                    if track.track_id != proposal.track_id:
                        continue
                    if track.locked or index == 0:
                        continue
                    left = project.tracks[index - 1]
                    if not (left.start_frame < proposal.new_frame < track.end_frame):
                        continue
                    left.end_frame = proposal.new_frame
                    track.start_frame = proposal.new_frame
                    track.add_reason("Boundary refined")
                    track.boundary_uncertain = bool(proposal.uncertain)
                    track.onset_frame = int(proposal.onset_frame)
                    track.boundary_pre_roll_seconds = float(
                        proposal.pre_roll_seconds
                    )
                    if not proposal.uncertain:
                        track.boundary_basis = BOUNDARY_BASIS_ASSESSED
                    applied += 1
                    break
            project.touch()

        self._mutate("Refine boundaries", change)
        self.events.emit(TRACKS_CHANGED)
        return applied

    def reposition_boundary(
        self,
        frame: int,
        pre_roll_seconds: float,
        onset_frame: Optional[int] = None,
    ) -> Optional[int]:
        """Move one cut to a new margin in front of its onset.

        This is what the margin control in the review panel uses when it is
        set to "this cut": the cut moves, the previous track's end moves with
        it, and the whole thing is one undoable step.

        Returns the new frame, or ``None`` when the cut could not move.
        """
        tracks = self.project.sorted_tracks()
        target = None
        index = -1
        for position, track in enumerate(tracks):
            if track.start_frame == frame and position > 0:
                target = track
                index = position
                break
        if target is None:
            return None

        onset = target.onset_frame if onset_frame is None else int(onset_frame)
        if onset < 0:
            return None

        rate = self.project.sample_rate() or 44100
        new_frame = onset - int(round(float(pre_roll_seconds) * rate))
        left = tracks[index - 1]
        new_frame = max(new_frame, left.start_frame + 1)
        new_frame = min(new_frame, target.end_frame - 1)
        if new_frame == frame:
            return None

        def move(project: Project) -> None:
            left.end_frame = new_frame
            target.start_frame = new_frame
            target.boundary_pre_roll_seconds = (
                onset - new_frame
            ) / float(rate)
            target.add_reason("The margin before the music was changed by hand")
            project.touch()

        self._mutate("Change the margin before the music", move)
        self.events.emit(TRACKS_CHANGED)
        return new_frame

    def accept_boundary(self, frame: int) -> bool:
        """Accept a proposed cut as it is.

        Accepting is a decision by a person, so the boundary stops being an
        automatic assessment and is recorded as accepted. The confidence
        number it had is kept: it is the evidence behind the cut, and it is
        still shown in the technical details.
        """
        for track in self.project.tracks:
            if track.start_frame == frame:
                def accept(project: Project) -> None:
                    for item in project.tracks:
                        if item.track_id == track.track_id:
                            item.suggested = False
                            item.boundary_uncertain = False
                            item.boundary_basis = BOUNDARY_BASIS_ACCEPTED
                            item.add_reason("Accepted by hand")
                            break
                    project.touch()

                self._mutate("Accept the cut", accept)
                self.events.emit(TRACKS_CHANGED)
                return True
        return False

    # ------------------------------------------------------------------
    # Track editing
    # ------------------------------------------------------------------
    def track_for_frame(self, frame: int) -> Optional[Track]:
        return self.project.track_at(frame)

    def sorted_tracks(self) -> list[Track]:
        return self.project.sorted_tracks()

    def _generic_title(self, index: int) -> str:
        return f"Track {index:02d}"

    def split_at(self, frame: int, reason: str = "manual split") -> Optional[Track]:
        """Split the track containing ``frame`` into two.

        Returns the new (right-hand) track, or ``None`` when the frame is not
        strictly inside an unlocked track.
        """
        track = self.project.track_at(frame)
        if track is None or frame <= track.start_frame or frame >= track.end_frame:
            return None
        if track.locked:
            self.status("Unlock the track before splitting it.")
            return None
        created: dict[str, Track] = {}

        def split(project: Project) -> None:
            new_track = Track(
                start_frame=frame,
                end_frame=track.end_frame,
                title="",
                section_type=track.section_type,
                boundary_confidence=0.0,
                boundary_basis=BOUNDARY_BASIS_MANUAL,
                identity_confidence=0.0,
                identity_basis=BOUNDARY_BASIS_NONE,
                classification_confidence=track.classification_confidence,
                reasons=[reason],
                metadata=Metadata(),
                notes="",
            )
            track.end_frame = frame
            track.boundary_confidence = 0.0
            track.boundary_basis = BOUNDARY_BASIS_MANUAL
            track.add_reason(reason)
            project.tracks.append(new_track)
            project.enforce_shared_boundaries()
            for idx, item in enumerate(project.sorted_tracks(), start=1):
                if not item.title.strip():
                    item.title = f"Track {idx:02d}"
            project.touch()
            created["track"] = new_track

        self._mutate("Split", split)
        self.events.emit(TRACKS_CHANGED)
        return created.get("track")

    def remove_split(self, boundary_frame: int) -> bool:
        """Remove the shared divider at ``boundary_frame`` without deleting audio.

        The right-hand track is merged into the left-hand track. Locked tracks
        are protected.
        """
        tracks = self.project.sorted_tracks()
        target = None
        for left, right in zip(tracks, tracks[1:]):
            if left.end_frame == boundary_frame and right.start_frame == boundary_frame:
                if left.locked or right.locked:
                    self.status("Unlock adjacent tracks before removing this split.")
                    return False
                target = right
                break
        if target is None:
            return False
        return self.merge_with_previous(target.track_id)

    def merge_with_previous(self, track_id: str) -> bool:
        """Merge a track into the one before it, keeping all of the audio."""
        tracks = self.project.sorted_tracks()
        index = next((i for i, t in enumerate(tracks) if t.track_id == track_id), -1)
        if index <= 0:
            return False
        previous, current = tracks[index - 1], tracks[index]
        if previous.locked or current.locked:
            self.status("Unlock both tracks before merging them.")
            return False

        def merge(project: Project) -> None:
            previous.end_frame = max(previous.end_frame, current.end_frame)
            previous.manual_override = True
            previous.identity_confidence = min(
                previous.identity_confidence, current.identity_confidence
            )
            previous.add_reason("Merged with the following segment")
            project.tracks = [t for t in project.tracks if t.track_id != track_id]
            project.enforce_shared_boundaries()
            project.touch()

        self._mutate("Merge", merge)
        self.events.emit(TRACKS_CHANGED)
        return True

    def remove_track(self, track_id: str) -> bool:
        """Remove a track *without* removing audio: it is merged into a neighbour."""
        tracks = self.project.sorted_tracks()
        if len(tracks) <= 1:
            return False
        target = next((t for t in tracks if t.track_id == track_id), None)
        if target is None:
            return False
        if target.locked:
            self.status("Unlock the track before removing it.")
            return False
        index = tracks.index(target)
        if index > 0:
            return self.merge_with_previous(track_id)
        # First track: merge it into the next one.
        following = tracks[1]
        if following.locked:
            self.status("Unlock the following track before removing this track.")
            return False

        def merge_forward(project: Project) -> None:
            following.start_frame = min(following.start_frame, target.start_frame)
            following.manual_override = True
            following.identity_confidence = min(
                following.identity_confidence, target.identity_confidence
            )
            project.tracks = [t for t in project.tracks if t.track_id != track_id]
            project.enforce_shared_boundaries()
            project.touch()

        self._mutate("Remove track", merge_forward)
        self.events.emit(TRACKS_CHANGED)
        return True

    def move_boundary(self, old_frame: int, new_frame: int) -> bool:
        """Move the shared boundary at ``old_frame`` to ``new_frame``.

        The boundary must stay strictly inside the two tracks it separates, so
        ``left.end_frame == right.start_frame`` always holds and a move can
        never cross into a different track. Locked tracks are protected.
        """
        tracks = self.project.sorted_tracks()
        left = None
        right = None
        for index in range(len(tracks) - 1):
            if tracks[index].end_frame == old_frame and tracks[index + 1].start_frame == old_frame:
                left, right = tracks[index], tracks[index + 1]
                break
        if left is None:
            return False
        if left.locked or right.locked:
            self.status("Unlock adjacent tracks before moving this boundary.")
            return False
        if new_frame <= left.start_frame or new_frame >= right.end_frame:
            return False

        def move(project: Project) -> None:
            left.end_frame = new_frame
            right.start_frame = new_frame
            left.add_reason("Boundary adjusted by hand")
            right.add_reason("Boundary adjusted by hand")
            # A boundary a human moved is a decision, not a guess.
            left.boundary_basis = BOUNDARY_BASIS_MANUAL
            left.boundary_confidence = 0.0
            left.boundary_uncertain = False
            right.boundary_basis = BOUNDARY_BASIS_MANUAL
            right.boundary_confidence = 0.0
            right.boundary_uncertain = False
            # The margin that was applied is now a different one, so it is
            # measured again. Otherwise the interface would keep reporting
            # the margin from before the move. A cut that was never derived
            # from an onset has no margin at all.
            rate = self.project.sample_rate() or 44100
            right.boundary_pre_roll_seconds = (
                (right.onset_frame - new_frame) / float(rate)
                if right.onset_frame >= 0
                else -1.0
            )
            project.enforce_shared_boundaries()
            project.touch()

        self._mutate("Move boundary", move)
        self.events.emit(TRACKS_CHANGED)
        return True

    def reorder_track(self, track_id: str, new_index: int) -> bool:
        """Move a track's label/metadata to ``new_index`` while preserving
        strict chronological audio order.

        Audio boundaries (``start_frame``, ``end_frame``) NEVER move or
        rearrange: only the track labels and metadata shift across the
        chronological timeline slots. Locked tracks cannot be shifted.
        """
        tracks = self.project.sorted_tracks()
        old_index = next((i for i, t in enumerate(tracks) if t.track_id == track_id), -1)
        if old_index < 0:
            return False
        new_index = max(0, min(int(new_index), len(tracks) - 1))
        if new_index == old_index:
            return False
        lo, hi = min(old_index, new_index), max(old_index, new_index)
        if any(tracks[i].locked for i in range(lo, hi + 1)):
            self.status("Cannot reorder across a locked track.")
            return False

        _IDENTITY_FIELDS = (
            "title",
            "section_type",
            "identity_confidence",
            "identity_basis",
            "classification_confidence",
            "metadata",
            "worklist_id",
            "worklist_index",
            "notes",
            "manual_override",
        )

        def reorder(project: Project) -> None:
            import copy

            ordered = project.sorted_tracks()
            payloads = [
                {field: copy.deepcopy(getattr(t, field)) for field in _IDENTITY_FIELDS}
                for t in ordered
            ]
            moved = payloads.pop(old_index)
            payloads.insert(new_index, moved)
            for track, payload in zip(ordered, payloads):
                for field, value in payload.items():
                    setattr(track, field, value)
                track.manual_override = True
            project.enforce_shared_boundaries()
            project.touch()

        self._mutate("Reorder tracks", reorder)
        self.events.emit(TRACKS_CHANGED)
        return True

    def move_track_up(self, track_id: str) -> bool:
        tracks = self.project.sorted_tracks()
        index = next((i for i, t in enumerate(tracks) if t.track_id == track_id), -1)
        if index <= 0:
            return False
        return self.reorder_track(track_id, index - 1)

    def move_track_down(self, track_id: str) -> bool:
        tracks = self.project.sorted_tracks()
        index = next((i for i, t in enumerate(tracks) if t.track_id == track_id), -1)
        if index < 0 or index >= len(tracks) - 1:
            return False
        return self.reorder_track(track_id, index + 1)

    def nudge_boundary(self, frame: int, delta_frames: int) -> bool:
        """Move a boundary by a small amount (keyboard nudging)."""
        return self.move_boundary(frame, frame + int(delta_frames))

    def set_track_title(self, track_id: str, title: str) -> None:
        def change(project: Project) -> None:
            for track in project.tracks:
                if track.track_id == track_id:
                    track.title = title
                    track.manual_override = True
                    if title.strip():
                        # A human typed it: the identity is no longer a guess.
                        track.identity_confidence = 1.0
                        track.identity_basis = BOUNDARY_BASIS_MANUAL
                        track.suggested = False
                    project.touch()
                    break

        self._mutate("Rename track", change)
        self.events.emit(TRACKS_CHANGED)

    def set_track_section_type(self, track_id: str, section_type: str) -> None:
        def change(project: Project) -> None:
            for track in project.tracks:
                if track.track_id == track_id:
                    track.section_type = section_type
                    track.manual_override = True
                    project.touch()
                    break

        self._mutate("Change section type", change)
        self.events.emit(TRACKS_CHANGED)

    def set_track_metadata(self, track_id: str, field: str, value) -> None:
        def change(project: Project) -> None:
            for track in project.tracks:
                if track.track_id == track_id:
                    if hasattr(track.metadata, field):
                        setattr(track.metadata, field, value)
                        track.manual_override = True
                    project.touch()
                    break

        self._mutate("Edit metadata", change)
        self.events.emit(TRACKS_CHANGED)

    def set_project_metadata(self, field: str, value) -> None:
        def change(project: Project) -> None:
            if hasattr(project.metadata, field):
                setattr(project.metadata, field, value)
            project.touch()

        self._mutate("Edit metadata", change)
        self.events.emit(PROJECT_CHANGED)

    def toggle_lock(self, track_id: str) -> bool:
        locked = False

        def change(project: Project) -> None:
            nonlocal locked
            for track in project.tracks:
                if track.track_id == track_id:
                    track.locked = not track.locked
                    locked = track.locked
                    project.touch()
                    break

        self._mutate("Toggle lock", change)
        self.events.emit(TRACKS_CHANGED)
        return locked

    def lock_all(self, locked: bool = True) -> None:
        def change(project: Project) -> None:
            for track in project.tracks:
                track.locked = locked
            project.touch()

        self._mutate("Lock tracks", change)
        self.events.emit(TRACKS_CHANGED)

    def _renumber_generic_titles(self) -> None:
        """Give untitled tracks stable generic names (Track 01, Track 02...)."""
        for index, track in enumerate(self.project.sorted_tracks(), start=1):
            if not track.title.strip():
                track.title = f"Track {index:02d}"

    # ------------------------------------------------------------------
    # Suggestions and worklists
    # ------------------------------------------------------------------
    def apply_suggestions(self, suggestions: Iterable) -> None:
        """Replace the track list with the suggested boundaries.

        Locked tracks keep their exact boundaries, their titles and their
        metadata: a re-analysis never overwrites an approved decision.
        """
        self.apply_analysis(suggestions)

    def apply_analysis(self, suggestions: Iterable, classifications: Iterable = ()) -> None:
        """Apply boundary suggestions and section classifications together.

        This is one undoable step, applied on the main thread: the analysis
        job only ever reads the project.
        """
        suggestions = list(suggestions)
        classifications = list(classifications)
        frames = sorted(
            {int(s.frame) for s in suggestions if 0 < int(s.frame) < self.project.total_frames()}
        )
        by_frame = {int(s.frame): s for s in suggestions}
        confidence = {int(s.frame): float(s.confidence) for s in suggestions}
        reasons = {int(s.frame): s.reason for s in suggestions}
        uncertain = {int(s.frame): bool(getattr(s, "uncertain", False)) for s in suggestions}

        by_start = {
            int(start): (section_type, conf, list(notes))
            for start, _end, section_type, conf, notes in classifications
        }
        pre_roll = {
            int(s.frame): float(s.details.get("applied_pre_roll_ms", -1)) / 1000.0
            for s in suggestions
            if s.details.get("applied_pre_roll_ms") is not None
        }

        def apply(project: Project) -> None:
            locked = [t for t in project.tracks if t.locked]
            protected: list[tuple[int, int]] = [(t.start_frame, t.end_frame) for t in locked]

            boundaries = set(frames)
            for start, end in protected:
                boundaries = {f for f in boundaries if not (start < f < end)}

            edges = sorted({0, project.total_frames()} | boundaries)
            new_tracks: list[Track] = []
            for index in range(len(edges) - 1):
                start, end = edges[index], edges[index + 1]
                if end <= start:
                    continue
                if any(start < p_end and p_start < end for p_start, p_end in protected):
                    continue
                new_track = Track(
                    start_frame=start,
                    end_frame=end,
                    title="",
                    section_type="unknown",
                    boundary_confidence=confidence.get(start, 0.0),
                    identity_confidence=0.0,
                    reasons=[reasons.get(start, "Suggested boundary")],
                    suggested=True,
                    boundary_basis=BOUNDARY_BASIS_ASSESSED,
                    identity_basis=BOUNDARY_BASIS_NONE,
                    boundary_placement=getattr(by_frame.get(start), "placement", "") or "",
                    onset_frame=int(getattr(by_frame.get(start), "onset_frame", -1)),
                    boundary_pre_roll_seconds=pre_roll.get(start, -1.0),
                    transition_start=int(getattr(by_frame.get(start), "transition_start", -1)),
                    transition_end=int(getattr(by_frame.get(start), "transition_end", -1)),
                    boundary_uncertain=uncertain.get(start, False),
                )
                if uncertain.get(start):
                    new_track.add_reason(
                        "This cut was not certain - listen to it before you accept it."
                    )
                classification = by_start.get(start)
                if classification:
                    section_type, section_confidence, notes = classification
                    new_track.section_type = section_type
                    new_track.classification_confidence = section_confidence
                    for note in notes:
                        new_track.add_reason(note)
                new_tracks.append(new_track)

            project.tracks = new_tracks + locked
            project.tracks.sort(key=lambda t: (t.start_frame, t.end_frame))
            project.touch()

        self._mutate("Apply analysis", apply)
        self._renumber_generic_titles()
        self.events.emit(TRACKS_CHANGED)

    def apply_theater_analysis(self, outcome) -> int:
        """Apply a theatrical analysis as a single undoable step.

        The sections become the tracks, the overlapping activity regions
        are stored on the project, and each track is told which activities
        it holds. Locked tracks keep their boundaries, their titles and
        their metadata, exactly as in the ordinary analysis.
        """
        suggestions = list(getattr(outcome, "suggestions", []))
        classifications = list(getattr(outcome, "classifications", []))
        activities = list(getattr(outcome, "activities", []))

        def apply(project: Project) -> None:
            locked = [t for t in project.tracks if t.locked or t.manual_override
                      or t.boundary_basis in ("manual", "accepted")
                      or t.identity_basis == "manual"
                      or any(c.basis == "manual" for c in t.components)]
            protected = [(t.start_frame, t.end_frame) for t in locked]

            boundaries = set()
            for suggestion in suggestions:
                frame = int(suggestion.frame)
                if 0 < frame < project.total_frames():
                    boundaries.add(frame)
            for start, end in protected:
                boundaries = {f for f in boundaries if not (start < f < end)}

            edges = sorted({0, project.total_frames()} | boundaries |
                           {f for span in protected for f in span})
            ordered_classifications = sorted(
                classifications, key=lambda item: int(item[0])
            )

            def classification_for(start: int, end: int):
                """The classification of the section this stretch belongs to.

                The boundary sits a pre-roll before the section it opens, so
                the two do not start on the same frame and an exact match
                finds nothing. The section that overlaps the stretch the
                most is the one that describes it.
                """
                best = None
                best_overlap = 0
                for item in ordered_classifications:
                    overlap = min(end, int(item[1])) - max(start, int(item[0]))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best = item
                if best is None:
                    return None
                return (best[2], float(best[3]), list(best[4]))

            confidence = {
                int(s.frame): float(s.confidence) for s in suggestions
            }
            reasons = {int(s.frame): s.reason for s in suggestions}
            uncertain = {
                int(s.frame): bool(getattr(s, "uncertain", False)) for s in suggestions
            }
            details = {int(s.frame): dict(getattr(s, "details", {}) or {}) for s in suggestions}

            new_tracks: list[Track] = []
            for index in range(len(edges) - 1):
                start, end = edges[index], edges[index + 1]
                if end <= start:
                    continue
                if any(start < p_end and p_start < end for p_start, p_end in protected):
                    continue
                track = Track(
                    start_frame=start,
                    end_frame=end,
                    title="",
                    section_type="unknown",
                    boundary_confidence=confidence.get(start, 0.0),
                    identity_confidence=0.0,
                    reasons=[reasons.get(start, "Suggested boundary")],
                    suggested=True,
                    boundary_basis=BOUNDARY_BASIS_ASSESSED,
                    identity_basis=BOUNDARY_BASIS_NONE,
                    boundary_placement="theater",
                    boundary_pre_roll_seconds=0.030,
                    boundary_uncertain=uncertain.get(start, False),
                )
                if uncertain.get(start):
                    track.add_reason(
                        "This cut was not certain - listen to it before you accept it."
                    )
                classification = classification_for(start, end)
                if classification:
                    section_type, section_confidence, notes = classification
                    track.section_type = section_type
                    track.classification_confidence = section_confidence
                    for note in notes:
                        track.add_reason(note)
                extra = details.get(start, {})
                if extra.get("activities"):
                    track.add_reason(
                        "Activities found here: " + ", ".join(extra["activities"])
                    )
                new_tracks.append(track)

            project.tracks = new_tracks + locked
            project.tracks.sort(key=lambda t: (t.start_frame, t.end_frame))
            project.activities = activities
            project.cache["musical_grouping"] = {
                "version": 1,
                "decisions": list(getattr(outcome, "grouping_decisions", [])),
                "reference_matches": list(getattr(outcome, "reference_matches", [])),
                "groups": [{"start_frame": s.start_frame, "end_frame": s.end_frame,
                            "type": s.section_type, "evidence": s.evidence}
                           for s in getattr(outcome, "sections", [])],
            }
            project.cache["theater_review_regions"] = [
                region for section in getattr(outcome, "sections", [])
                for region in section.evidence.get("review_regions", [])
            ]
            if getattr(outcome, "aec_cache_entry", None) is not None:
                project.cache["theater_aec"] = outcome.aec_cache_entry
            if getattr(outcome, "transcript_cache_entry", None) is not None:
                project.cache["theater_transcript"] = outcome.transcript_cache_entry
            project.refresh_track_activities()
            project.touch()

        self._mutate("Apply theatrical analysis", apply)
        self._renumber_generic_titles()
        raw_transcript = getattr(outcome, "transcript", None) or []
        if raw_transcript:
            self._theater_transcript = [
                dict(segment)
                for segment in raw_transcript
                if segment.get("accepted", True)
            ]
        else:
            self._restore_theater_transcript_from_cache()
        self.events.emit(TRACKS_CHANGED)
        return len(self.project.tracks)

    def paste_tracklist(self, text: str, name: str = "Pasted tracklist") -> Worklist:
        items = serialize.parse_tracklist(text)
        worklist = Worklist(name=name, items=items, provider="manual", fetched_at=now_iso())

        def add(project: Project) -> None:
            project.worklists.append(worklist)
            project.active_worklist_id = worklist.worklist_id
            project.touch()

        self._mutate("Paste tracklist", add)
        self.events.emit(PROJECT_CHANGED)
        self.status(f"Imported {len(items)} tracklist entries")
        return worklist

    def add_worklist(self, worklist: Worklist) -> Worklist:
        """Add a tracklist that came from a provider."""
        def add(project: Project) -> None:
            project.worklists.append(worklist)
            project.active_worklist_id = worklist.worklist_id
            project.touch()

        self._mutate(f"Add the {worklist.name} tracklist", add)
        self.events.emit(PROJECT_CHANGED)
        return worklist

    def set_active_worklist(self, worklist_id: str) -> None:
        def change(project: Project) -> None:
            project.active_worklist_id = worklist_id
            project.touch()

        self._mutate("Select tracklist", change)
        self.events.emit(PROJECT_CHANGED)

    def align_with_worklist(self) -> dict:
        """Match the active worklist to the detected segments.

        Only *identity* changes as a result: titles and identity confidence
        are updated, boundary confidence is untouched.
        """
        from ..analysis.align import align_worklist

        worklist = self.project.active_worklist()
        if not worklist or not worklist.items:
            raise SessionError("There is no tracklist to align with.")
        segments = [(t.start_frame, t.end_frame) for t in self.project.sorted_tracks()]
        result = align_worklist(worklist.items, segments, self.project.sample_rate() or 44100)

        assignments = {
            segment: (item, confidence)
            for (item, segment), confidence in zip(result.assignments, result.identity_confidence)
        }

        def apply(project: Project) -> None:
            for index, track in enumerate(project.sorted_tracks()):
                match = assignments.get(index)
                if match is None:
                    continue
                item_index, confidence = match
                item = worklist.items[item_index]
                track.title = item.title
                track.identity_confidence = confidence
                track.identity_basis = BOUNDARY_BASIS_ASSESSED
                track.worklist_id = worklist.worklist_id
                track.worklist_index = item_index
                track.suggested = False
                if item.notes:
                    track.notes = item.notes
            project.touch()

        self._mutate("Align tracklist", apply)
        self.events.emit(TRACKS_CHANGED)
        return {
            "matched": len(result.assignments),
            "unmatched_items": len(result.unmatched_items),
            "unmatched_segments": len(result.unmatched_segments),
        }

    def propose_theater_names(self) -> dict:
        """Propose names for the detected numbers from the reference list.

        The reference list suggests names and an order. It does not
        establish a timestamp and it does not prove that a number is in
        the recording, so nothing here is forced: a number that is not in
        the recording stays unmatched, a stretch with no number stays
        unmatched, and both are reported.

        Only the *identity* changes: the titles and the identity
        confidence. The boundaries are untouched.
        """
        from ..analysis.align import align_theater_worklist

        worklist = self.project.active_worklist()
        if not worklist or not worklist.items:
            raise SessionError("There is no reference list to align with.")

        candidates = [
            track
            for track in self.project.sorted_tracks()
            if track.section_type == "musical_number" or track.components
        ]
        if not candidates:
            raise SessionError(
                "No musical numbers were detected, so there is nothing to name."
            )

        transcript = [
            (
                float(segment.get("start", 0.0)),
                float(segment.get("end", 0.0)),
                str(segment.get("text", "") or ""),
            )
            for segment in getattr(self, "_theater_transcript", []) or []
            if segment.get("accepted", True)
        ]

        result = align_theater_worklist(
            worklist.items,
            candidates,
            self.project.sample_rate() or 44100,
            transcripts=transcript,
        )
        assignments = {
            segment: (item, confidence)
            for (item, segment), confidence in zip(result.assignments, result.identity_confidence)
        }

        def apply(project: Project) -> None:
            for index, track in enumerate(candidates):
                match = assignments.get(index)
                if match is None:
                    if not track.locked and track.worklist_id == worklist.worklist_id:
                        track.title = ""
                        track.identity_confidence = 0.0
                        track.identity_basis = BOUNDARY_BASIS_NONE
                        track.worklist_id = ""
                        track.worklist_index = -1
                    if not track.title:
                        track.add_reason(
                            "Left unnamed: transcript did not provide sufficient "
                            "independent evidence to confirm a reference title."
                        )
                    continue
                item_index, confidence = match
                item = worklist.items[item_index]
                track.title = item.title
                track.identity_confidence = confidence
                track.identity_basis = BOUNDARY_BASIS_ASSESSED
                track.worklist_id = worklist.worklist_id
                track.worklist_index = item_index
                track.suggested = False
                ev_text = result.evidence.get((item_index, index), "")
                if ev_text:
                    track.add_reason(ev_text)
                track.add_reason(
                    "The title comes from the reference list with transcript "
                    "support, which is evidence and not a confirmed identification. "
                    "Listen before you trust it."
                )
                if item.notes:
                    track.notes = item.notes
            project.touch()

        self._mutate("Propose names from the reference list", apply)
        self._renumber_generic_titles()
        components = self._name_medley_components(worklist)
        self.events.emit(TRACKS_CHANGED)
        return {
            "matched": len(result.assignments),
            "unmatched_items": len(result.unmatched_items),
            "unmatched_segments": len(result.unmatched_segments),
            "candidates": len(candidates),
            "components": components,
        }

    # ------------------------------------------------------------------
    # The songs inside a medley
    # ------------------------------------------------------------------
    def _name_medley_components(self, worklist: Worklist) -> dict:
        """Find the songs inside the numbers that hold several.

        A medley is one number that holds several songs. The number keeps
        its own title and its own boundaries; the songs inside it are
        separate things, with their own names and their own boundaries, and
        the two are never merged.

        Two independent pieces of evidence are used, and neither may invent
        anything:

        * **What the words say.** The reference list says which songs can
          live inside this number. A name is proposed only where the words
          the transcriber recovered actually support it, across a run of
          lines rather than a single one.
        * **What the recording does not say.** Where two reference titles
          support the same stretch, the evidence cannot tell them apart, so
          neither is proposed, and the number says so.

        The number's own title is not touched.
        """
        from ..analysis.medley import segment_components
        from ..core.model import ComponentSong

        sample_rate = self.project.sample_rate() or 44100
        sources = getattr(self.project, "sources", []) or []
        transcript = [
            (
                float(segment.get("start", 0.0)),
                float(segment.get("end", 0.0)),
                str(segment.get("text", "") or ""),
            )
            for segment in getattr(self, "_theater_transcript", []) or []
        ]

        # Which numbers can hold other songs? The reference says so, and the
        # reference is data: nothing here knows about any particular show.
        medleys = []
        for track in self.project.sorted_tracks():
            if track.worklist_id != worklist.worklist_id:
                continue
            if track.worklist_index < 0:
                continue
            item = worklist.items[track.worklist_index]
            if item.contains:
                medleys.append((track, item))

        if not medleys or not sources:
            return {"numbers": 0, "songs": 0, "named": 0, "unlocated": 0}

        found = []
        unlocated_total = 0
        for track, item in medleys:
            start = track.start_frame
            end = track.end_frame
            if end <= start:
                continue
            start_seconds = start / float(sample_rate)
            end_seconds = end / float(sample_rate)
            inside = [
                # The stretches are made relative to the number, because
                # that is the frame the detector adds its base to.
                (t_start - start_seconds, t_end - start_seconds, text)
                for (t_start, t_end, text) in transcript
                if t_end > start_seconds and t_start < end_seconds
            ]
            matches, unlocated = segment_components(
                inside, list(item.contains), start, sample_rate
            )
            if not matches and not unlocated:
                continue
            components = [
                ComponentSong(
                    title=match.title,
                    start_frame=match.start_frame,
                    end_frame=match.end_frame,
                    confidence=match.confidence,
                    basis=match.basis,
                    reason=match.reason,
                )
                for match in matches
            ]
            unlocated_total += len(unlocated)
            found.append((track.track_id, components, unlocated))

        if not found:
            return {"numbers": 0, "songs": 0, "named": 0, "unlocated": 0}

        def apply(project: Project) -> None:
            by_id = {track.track_id: track for track in project.tracks}
            for track_id, components, unlocated in found:
                track = by_id.get(track_id)
                if track is None:
                    continue
                track.components = components
                named = sum(1 for c in components if c.title)
                track.add_reason(
                    f"This number holds {named} song(s) that the reference "
                    "and the recovered words agree on. They are listed "
                    "separately from the number's own title."
                )
                if unlocated:
                    track.add_reason(
                        f"The reference also lists {len(unlocated)} other "
                        "song(s) that can live inside this number "
                        "(" + ", ".join(unlocated) + "). The recording did "
                        "not give enough evidence to say where they are or "
                        "that they are here, so they are not listed as "
                        "separate tracks. Listen if you want them."
                    )
            project.touch()

        self._mutate("Find the songs inside the medleys", apply)
        return {
            "numbers": len(found),
            "songs": sum(len(components) for _, components, _ in found),
            "named": sum(
                1 for _, components, _ in found
                for component in components if component.title
            ),
            "unlocated": unlocated_total,
        }

    def apply_lyric_grouping(
        self,
        proposals: Iterable[dict],
        *,
        allow_uncertain: bool = False,
    ) -> dict:
        """Apply proposed lyric-assisted musical-number groupings in one undoable step.

        Each proposal must supply ``start_frame``, ``end_frame``, and ``title``.
        Locked and manually edited tracks are never overwritten. Uncertain or
        partial proposals (missing opening/interior/ending coverage, unresolved
        audio extent, or explicitly flagged ``uncertain``) are skipped unless
        ``allow_uncertain=True`` after explicit user confirmation. Existing
        boundaries outside the applied proposals are preserved, interior
        boundaries inside each applied number are grouped into one track, and
        the shared-boundary contiguous timeline invariant is strictly enforced.
        """
        import copy

        total = self.project.total_frames()
        counts = {
            "applied": 0,
            "skipped_protected": 0,
            "skipped_overlap": 0,
            "skipped_uncertain": 0,
        }
        if total <= 0:
            return counts

        cleaned: list[dict] = []
        for item in proposals:
            start = max(0, min(int(item.get("start_frame", 0)), total))
            end = max(0, min(int(item.get("end_frame", 0)), total))
            if end <= start:
                continue
            status = str(item.get("status", "proposal") or "proposal")
            missing_roles = list(item.get("missing_roles") or [])
            intro_unresolved = bool(item.get("intro_unresolved", False))
            tail_unresolved = bool(item.get("tail_unresolved", False))
            is_uncertain = (
                bool(item.get("uncertain", False))
                or bool(missing_roles)
                or intro_unresolved
                or tail_unresolved
                or status not in ("multi-anchor proposal", "proposal")
            )
            cleaned.append({
                "start_frame": start,
                "end_frame": end,
                "title": str(item.get("title", "") or "").strip(),
                "status": status,
                "provenance": str(item.get("provenance", "") or ""),
                "uncertain": is_uncertain,
                "missing_roles": missing_roles,
            })
        cleaned.sort(key=lambda p: (p["start_frame"], p["end_frame"]))

        # Pre-filter before _mutate so a no-op never pushes an undo snapshot or marks dirty.
        prior_preview = self.project.sorted_tracks()
        protected_preview = [
            t for t in prior_preview
            if t.locked
            or t.manual_override
            or t.boundary_basis in (BOUNDARY_BASIS_MANUAL, BOUNDARY_BASIS_ACCEPTED)
            or t.identity_basis == BOUNDARY_BASIS_MANUAL
        ]
        protected_preview_spans = [(t.start_frame, t.end_frame) for t in protected_preview]

        accepted_preview: list[dict] = []
        for prop in cleaned:
            ps, pe = prop["start_frame"], prop["end_frame"]
            if prop["uncertain"] and not allow_uncertain:
                counts["skipped_uncertain"] += 1
                continue
            if any(ps < end and start < pe for start, end in protected_preview_spans):
                counts["skipped_protected"] += 1
                continue
            if any(ps < a["end_frame"] and a["start_frame"] < pe for a in accepted_preview):
                counts["skipped_overlap"] += 1
                continue
            accepted_preview.append(prop)

        if not accepted_preview:
            return counts

        counts["applied"] = len(accepted_preview)

        def apply(project: Project) -> None:
            prior_tracks = project.sorted_tracks()
            protected = [
                t for t in prior_tracks
                if t.locked
                or t.manual_override
                or t.boundary_basis in (BOUNDARY_BASIS_MANUAL, BOUNDARY_BASIS_ACCEPTED)
                or t.identity_basis == BOUNDARY_BASIS_MANUAL
            ]
            protected_spans = [(t.start_frame, t.end_frame) for t in protected]
            prior_by_span = {(t.start_frame, t.end_frame): t for t in prior_tracks}

            accepted_spans = [(a["start_frame"], a["end_frame"]) for a in accepted_preview]
            by_span = {(a["start_frame"], a["end_frame"]): a for a in accepted_preview}

            # Keep existing boundaries that are outside every accepted proposal
            kept_edges: set[int] = {0, total}
            for s, e in protected_spans:
                kept_edges.add(s)
                kept_edges.add(e)
            for s, e in accepted_spans:
                kept_edges.add(s)
                kept_edges.add(e)
            for t in prior_tracks:
                for edge in (t.start_frame, t.end_frame):
                    if 0 < edge < total:
                        if not any(ps < edge < pe for ps, pe in accepted_spans):
                            kept_edges.add(edge)

            edges = sorted(kept_edges)

            def _best_prior(s: int, e: int) -> Optional[Track]:
                best = None
                best_ov = 0
                for t in prior_tracks:
                    ov = min(e, t.end_frame) - max(s, t.start_frame)
                    if ov > best_ov:
                        best_ov = ov
                        best = t
                return best

            new_tracks: list[Track] = []
            for start, end in zip(edges, edges[1:]):
                if end <= start:
                    continue
                if any(start < pe and ps < end for ps, pe in protected_spans):
                    continue
                prop = by_span.get((start, end))
                if prop is not None:
                    reason = f"Grouped from lyric proposal ({prop['status']})"
                    if prop["provenance"]:
                        reason += f" — {prop['provenance']}"
                    is_unc = prop["uncertain"]
                    track = Track(
                        start_frame=start,
                        end_frame=end,
                        title=prop["title"],
                        section_type="musical_number",
                        boundary_confidence=0.35 if is_unc else 0.7,
                        identity_confidence=0.5 if (prop["title"] and is_unc) else (0.7 if prop["title"] else 0.0),
                        reasons=[reason],
                        suggested=is_unc,
                        boundary_uncertain=is_unc,
                        boundary_basis=BOUNDARY_BASIS_ASSESSED,
                        identity_basis=BOUNDARY_BASIS_ASSESSED if prop["title"] else BOUNDARY_BASIS_NONE,
                        boundary_placement="lyric_grouping",
                    )
                    new_tracks.append(track)
                else:
                    exact = prior_by_span.get((start, end))
                    if exact is not None:
                        new_tracks.append(exact)
                    else:
                        prior = _best_prior(start, end)
                        track = Track(
                            start_frame=start,
                            end_frame=end,
                            title=prior.title if prior else "",
                            section_type=prior.section_type if prior else "unknown",
                            boundary_confidence=prior.boundary_confidence if prior else 0.0,
                            identity_confidence=prior.identity_confidence if prior else 0.0,
                            classification_confidence=prior.classification_confidence if prior else 0.0,
                            reasons=list(prior.reasons) if prior else [],
                            boundary_basis=prior.boundary_basis if prior else BOUNDARY_BASIS_NONE,
                            identity_basis=prior.identity_basis if prior else BOUNDARY_BASIS_NONE,
                            boundary_placement=prior.boundary_placement if prior else "",
                            boundary_uncertain=prior.boundary_uncertain if prior else False,
                            suggested=prior.suggested if prior else False,
                            metadata=copy.deepcopy(prior.metadata) if prior else Metadata(),
                            notes=prior.notes if prior else "",
                        )
                        new_tracks.append(track)

            project.tracks = new_tracks + protected
            project.enforce_shared_boundaries()
            for idx, item in enumerate(project.sorted_tracks(), start=1):
                if not item.title.strip():
                    item.title = f"Track {idx:02d}"
            project.refresh_track_activities()
            project.touch()

        self._mutate("Apply proposed grouping", apply)
        self.events.emit(TRACKS_CHANGED)
        return counts

    def apply_metadata_split(self, outcome) -> int:
        """Apply a metadata-assisted split as a single undoable step.

        * Locked tracks keep their boundaries and metadata.
        * Adjacent tracks share one contiguous divider (no gaps or overlaps).
        * Metadata titles, worklist, and release candidates remain editable and
          fully reversible via Undo (Ctrl+Z) or Revert to audio-only.
        """
        ident = outcome.identification
        planned = list(getattr(outcome, "tracks", []) or [])
        worklist = getattr(ident, "worklist", None)

        def apply(project: Project) -> None:
            locked = [t for t in project.tracks if t.locked]
            protected = [(t.start_frame, t.end_frame) for t in locked]

            if worklist is not None:
                # Replace any previous auto-worklist with the same provider_id, else append
                existing = next(
                    (w for w in project.worklists if w.provider_id and w.provider_id == worklist.provider_id),
                    None,
                )
                if existing is None:
                    project.worklists.append(worklist)
                    project.active_worklist_id = worklist.worklist_id
                else:
                    existing.items = list(worklist.items)
                    existing.name = worklist.name
                    existing.notes = worklist.notes
                    project.active_worklist_id = existing.worklist_id

            if ident.artist and not project.metadata.artist:
                project.metadata.artist = ident.artist
            if ident.title and not project.metadata.album and ident.recording_kind in ("album", "broadway"):
                project.metadata.album = ident.title

            new_tracks: list[Track] = []
            wl_id = project.active_worklist_id if worklist is not None else ""
            for idx, item in enumerate(planned):
                start = max(0, int(item.start_frame))
                end = min(project.total_frames(), int(item.end_frame))
                if end <= start:
                    continue
                if any(start < p_end and p_start < end for p_start, p_end in protected):
                    continue
                has_id = bool(item.title and item.identity_confidence > 0.0)
                track = Track(
                    start_frame=start,
                    end_frame=end,
                    title=item.title or f"Track {idx + 1:02d}",
                    section_type=item.section_type or "music",
                    boundary_confidence=float(item.boundary_confidence),
                    identity_confidence=float(item.identity_confidence),
                    reasons=list(item.reasons),
                    flags=list(getattr(item, "flags", []) or []),
                    suggested=True,
                    boundary_basis=BOUNDARY_BASIS_ASSESSED,
                    identity_basis=BOUNDARY_BASIS_ASSESSED if has_id else BOUNDARY_BASIS_NONE,
                    boundary_placement=item.boundary_placement or "metadata_refined",
                    onset_frame=int(item.onset_frame),
                    boundary_pre_roll_seconds=float(item.boundary_pre_roll_seconds),
                    transition_start=int(item.transition_start),
                    transition_end=int(item.transition_end),
                    boundary_uncertain=bool(item.boundary_uncertain),
                    worklist_id=wl_id if item.worklist_index >= 0 else "",
                    worklist_index=int(item.worklist_index),
                    notes=item.notes or "",
                    recognized_artist=str(getattr(item, "recognized_artist", "") or ""),
                    recognized_title=str(getattr(item, "recognized_title", "") or ""),
                    recognition_status=str(getattr(item, "recognition_status", "") or ""),
                    recognition_engines=list(getattr(item, "recognition_engines", []) or []),
                )
                art = str(getattr(item, "artist", "") or getattr(item, "recognized_artist", "") or "").strip()
                if art:
                    track.metadata.artist = art
                new_tracks.append(track)

            if new_tracks or locked:
                project.tracks = new_tracks + locked
                project.enforce_shared_boundaries()
            if getattr(outcome, "activities", None):
                project.activities = list(outcome.activities)
                project.refresh_track_activities()

            project.cache["smart_identification"] = {
                "recording_kind": ident.recording_kind,
                "artist": ident.artist,
                "title": ident.title,
                "subtitle": ident.subtitle,
                "provider": ident.provider,
                "confidence": ident.confidence,
                "selected_index": ident.selected_index,
                "notes": list(ident.notes) + list(outcome.notes),
                "diagnostics": dict(outcome.diagnostics or {}),
            }
            project.touch()

        self._mutate("Apply smart metadata split", apply)
        self.events.emit(TRACKS_CHANGED)
        self.events.emit(PROJECT_CHANGED)
        return len(self.project.tracks)

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------
    def undo_action(self) -> bool:
        return self.undo.undo()

    def redo_action(self) -> bool:
        return self.undo.redo()

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------
    def refresh_flags(self) -> None:
        flag_tools.apply_flags(self.project)

    def source_issues(self) -> list[str]:
        return flag_tools.source_issues(self.project)

    def summary(self) -> dict:
        return flag_tools.track_summary(self.project)
