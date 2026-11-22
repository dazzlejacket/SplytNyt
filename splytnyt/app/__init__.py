"""Application services: session, jobs, export orchestration.

The service layer is free of Qt so it can be exercised by tests without a
display server. The UI calls into it and subscribes to plain callbacks.
"""

from .events import (  # noqa: F401
    BUSY,
    PLAYBACK,
    PROJECT_CHANGED,
    SELECTION_CHANGED,
    STATUS,
    TRACKS_CHANGED,
    Emitter,
)
from .session import Session, SessionError  # noqa: F401
