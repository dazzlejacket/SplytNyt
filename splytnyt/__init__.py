"""SplytNyt — local-first audio splitting and tagging.

Layers (see docs/ARCHITECTURE.md):

    core/       pure data model, no Qt, no audio, no numpy
    audio/      decode / measure / gain / encode / tag
    analysis/   boundary detection, section classification, identification
    metadata/   provider adapters and alignment
    app/        application services (orchestration, undo, export jobs)
    ui/         PySide6 widgets

Import order inside the package is strictly layered: ``core`` imports nothing
from the layers above it.
"""

__version__ = "0.1.0-beta"
APP_NAME = "SplytNyt"
