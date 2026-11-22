"""Analysis pipeline.

Three separate stages with three separate confidences:

    stage 1  boundary detection   -> boundary_confidence
    stage 2  section classification -> classification_confidence
    stage 3  track identification -> identity_confidence

They are never merged. Silence detection alone is never presented as
intelligent track detection.
"""

from .align import Alignment, align_worklist  # noqa: F401
from .classify import SectionGuess, classify_section  # noqa: F401
from .novelty import find_novelty_peaks, novelty_curve  # noqa: F401
from .segment import (  # noqa: F401
    SensitivityPreset,
    Suggestion,
    get_preset,
    suggest_boundaries,
)
from .silence import (  # noqa: F401
    NoiseProfile,
    SilenceRegion,
    analyse_noise_floor,
    find_silence_regions,
    frame_levels,
)
