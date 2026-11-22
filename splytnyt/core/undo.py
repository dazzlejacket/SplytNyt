"""Undo / redo.

Snapshot-based: the project model is small, pure data, so a deep copy is both
simpler and far less bug-prone than inverting a long chain of commands.
Analysis caches and decoded audio live outside the model and are therefore
never captured by an undo snapshot — undoing a boundary change must not throw
away a multi-minute peak pyramid.

The stack is bounded so a long session cannot grow without limit.
"""

from __future__ import annotations

from typing import Callable, Optional

from .model import Project


class UndoStack:
    """Bounded snapshot undo/redo for a :class:`Project`."""

    def __init__(self, project: Project, limit: int = 100) -> None:
        self._project = project
        self._limit = limit
        self._undo: list[Project] = []
        self._redo: list[Project] = []
        self._pending: Optional[Project] = None
        self._on_change: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------
    def set_on_change(self, callback: Optional[Callable[[], None]]) -> None:
        """Register a callback fired whenever the stack changes."""
        self._on_change = callback

    def _notify(self) -> None:
        if self._on_change:
            self._on_change()

    # ------------------------------------------------------------------
    def begin(self) -> None:
        """Capture the state *before* a mutation.

        Call before changing the project, then :meth:`commit` (or
        :meth:`rollback`) afterwards.
        """
        if self._pending is None:
            self._pending = self._project.clone()

    def commit(self, label: str = "") -> None:
        """Finish an edit, pushing the captured snapshot onto the undo stack."""
        if self._pending is None:
            return
        snapshot = self._pending
        self._pending = None
        self._undo.append(snapshot)
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()
        self._notify()

    def rollback(self) -> None:
        """Discard the captured snapshot without pushing it."""
        self._pending = None

    # ------------------------------------------------------------------
    def apply(self, mutate: Callable[[Project], None], label: str = "") -> None:
        """Run ``mutate`` as one undoable step."""
        self.begin()
        try:
            mutate(self._project)
        except Exception:
            self._pending = None
            raise
        self.commit(label)

    # ------------------------------------------------------------------
    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        """Restore the previous state. Returns ``False`` when nothing to undo."""
        if not self._undo:
            return False
        current = self._project.clone()
        previous = self._undo.pop()
        self._redo.append(current)
        self._restore(previous)
        self._notify()
        return True

    def redo(self) -> bool:
        """Re-apply the last undone change."""
        if not self._redo:
            return False
        current = self._project.clone()
        nxt = self._redo.pop()
        self._undo.append(current)
        self._restore(nxt)
        self._notify()
        return True

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._pending = None
        self._notify()

    def _restore(self, snapshot: Project) -> None:
        target = self._project
        target.sources = snapshot.sources
        target.tracks = snapshot.tracks
        target.activities = snapshot.activities
        target.worklists = snapshot.worklists
        target.markers = snapshot.markers
        target.metadata = snapshot.metadata
        target.export_settings = snapshot.export_settings
        target.active_worklist_id = snapshot.active_worklist_id
        target.placement_mode = snapshot.placement_mode
        target.placement_pre_roll_seconds = snapshot.placement_pre_roll_seconds
        target.notes = snapshot.notes
        target.cache = snapshot.cache
        target.name = snapshot.name
        target.modified = snapshot.modified
