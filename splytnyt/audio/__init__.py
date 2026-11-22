"""Audio engine: decode, peaks, measurement, gain, encoding, tagging.

This package never imports from :mod:`splytnyt.ui`, :mod:`splytnyt.app`,
:mod:`splytnyt.analysis` or :mod:`splytnyt.metadata`. It is pure audio I/O
and numeric work, usable from a console script or a test without Qt.
"""

from .decode import (  # noqa: F401
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FILTER,
    AudioError,
    AudioInfo,
    AudioSource,
    open_sources,
    probe,
)
from .peaks import CancelledError, PeakPyramid, empty_pyramid  # noqa: F401
