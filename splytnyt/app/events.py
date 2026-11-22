"""A tiny synchronous event emitter.

The service layer stays free of Qt so it can be unit tested without a display
server; the UI subscribes with plain callbacks and marshals to the GUI thread
itself.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

# Event names used by :class:`splytnyt.app.session.Session`.
PROJECT_CHANGED = "project_changed"
TRACKS_CHANGED = "tracks_changed"
STATUS = "status"
BUSY = "busy"
PLAYBACK = "playback"
SELECTION_CHANGED = "selection_changed"


class Emitter:
    """Minimal event bus."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable[..., Any]]] = {}

    def on(self, event: str, handler: Callable[..., Any]) -> Callable[..., Any]:
        self._handlers.setdefault(event, []).append(handler)
        return handler

    def off(self, event: str, handler: Callable[..., Any]) -> None:
        handlers = self._handlers.get(event)
        if handlers and handler in handlers:
            handlers.remove(handler)

    def emit(self, event: str, **payload: Any) -> None:
        for handler in list(self._handlers.get(event, [])):
            try:
                handler(**payload)
            except Exception:
                # A broken listener must never take the application down.
                import traceback

                traceback.print_exc()

    def clear(self) -> None:
        self._handlers.clear()
