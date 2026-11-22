"""Boundary suggestion.

Stage 1 of the analysis pipeline, part three: it merges the silence regions
and the musical-change peaks into a single list of candidate boundaries.

The rules encode what the product insists on:

* A silence is evidence, not proof. Silence-derived boundaries get a modest
  confidence, capped below certainty.
* A musical change inside continuous audio is reported as an **uncertain
  transition**, never as a confident cut.
* Quiet music, a solo, a short pause or applause inside a song must not be
  fragmented: short silences are ignored unless the preset asks for them,
  and the minimum separation between candidates keeps a single event from
  being cut several times.

Finding a transition and placing the cut are two separate jobs. This module
finds the transition; :mod:`splytnyt.analysis.placement` decides where inside
it the cut belongs, so the next song starts at its music and not in the
middle of the quiet in front of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .novelty import (
    find_novelty_peaks,
    novelty_curve_from_shapes,
    persistence_window_seconds,
    spectral_shapes,
)
from .placement import (
    PLACEMENT_NEXT_START,
    PlacementChoice,
    PlacementResult,
    place_cut,
)
from .silence import analyse_noise_floor, find_silence_regions


@dataclass
class Suggestion:
    """A proposed boundary."""

    frame: int
    confidence: float
    reason: str
    kind: str = "silence"          # "silence" | "novelty" | "manual"
    uncertain: bool = False
    #: Which placement rule produced this cut.
    placement: str = PLACEMENT_NEXT_START
    #: The musical onset the cut was derived from, or -1.
    onset_frame: int = -1
    #: The transition region the cut was placed inside.
    transition_start: int = -1
    transition_end: int = -1
    #: Technical numbers behind the decision, for the details panel.
    details: dict = field(default_factory=dict)


@dataclass
class SensitivityPreset:
    """Concrete parameters behind a sensitivity name."""

    name: str
    min_silence_seconds: float
    novelty_threshold_scale: float
    min_novelty_separation_seconds: float
    silence_confidence_scale: float
    novelty_confidence_scale: float


_PRESETS = {
    "conservative": SensitivityPreset(
        name="conservative",
        min_silence_seconds=0.8,
        novelty_threshold_scale=1.8,
        min_novelty_separation_seconds=6.0,
        silence_confidence_scale=1.0,
        novelty_confidence_scale=0.6,
    ),
    "balanced": SensitivityPreset(
        name="balanced",
        min_silence_seconds=0.45,
        novelty_threshold_scale=1.0,
        min_novelty_separation_seconds=3.0,
        silence_confidence_scale=1.0,
        novelty_confidence_scale=1.0,
    ),
    "sensitive": SensitivityPreset(
        name="sensitive",
        min_silence_seconds=0.2,
        novelty_threshold_scale=0.6,
        min_novelty_separation_seconds=1.5,
        silence_confidence_scale=1.0,
        novelty_confidence_scale=1.4,
    ),
}


def get_preset(name: str) -> SensitivityPreset:
    """Return the preset for a sensitivity name (unknown names fall back)."""
    return _PRESETS.get((name or "balanced").lower(), _PRESETS["balanced"])


def find_transition_regions(
    signal: np.ndarray,
    sample_rate: int,
    sensitivity: str = "balanced",
    threshold_dbfs: Optional[float] = None,
) -> list:
    """Find the stretches of quiet that may contain a boundary.

    These are *transition regions*: the detector does not yet know where
    inside them the cut belongs.
    """
    if signal.size == 0 or sample_rate <= 0:
        return []
    preset = get_preset(sensitivity)
    if threshold_dbfs is None:
        profile = analyse_noise_floor(signal, sample_rate)
        threshold_dbfs = profile.threshold_dbfs
    return find_silence_regions(
        signal,
        sample_rate,
        threshold_dbfs=threshold_dbfs,
        min_silence_seconds=preset.min_silence_seconds,
    )


def suggest_boundaries(
    signal: np.ndarray,
    sample_rate: int,
    sensitivity: str = "balanced",
    placement: Optional[PlacementChoice] = None,
    floor_dbfs: Optional[float] = None,
) -> list[Suggestion]:
    """Propose candidate boundaries in a mono signal."""
    if signal.size == 0 or sample_rate <= 0:
        return []
    preset = get_preset(sensitivity)

    profile = analyse_noise_floor(signal, sample_rate)
    floor = profile.noise_floor_dbfs if floor_dbfs is None else floor_dbfs
    regions = find_silence_regions(
        signal,
        sample_rate,
        threshold_dbfs=profile.threshold_dbfs,
        min_silence_seconds=preset.min_silence_seconds,
    )

    suggestions: list[Suggestion] = []
    for region in regions:
        result = place_cut(
            signal,
            sample_rate,
            region,
            choice=placement,
            floor_dbfs=floor,
        )
        if result.rejected:
            # The music simply continues across the quiet stretch: this is a
            # pause inside a song, not a boundary.
            continue
        suggestions.append(silence_suggestion(region, profile, preset, result))

    shapes = spectral_shapes(signal, sample_rate)
    curve = novelty_curve_from_shapes(shapes)
    peaks = find_novelty_peaks(curve, 512, sample_rate, sensitivity, shapes=shapes)
    # A peak this close to the start or the end of the recording cannot be
    # checked for persistence, so it is not proposed at all. On a short
    # recording the margin shrinks, otherwise nothing would ever be found.
    duration = signal.size / float(sample_rate)
    edge = max(1, int(min(persistence_window_seconds(), duration * 0.1) * sample_rate))
    novelty_items = []
    margin = max(1, int(0.5 * sample_rate))
    for frame, strength in peaks:
        if frame <= edge or frame >= signal.size - edge:
            continue
        if any(
            region.start_frame - margin <= frame <= region.end_frame + margin
            for region in regions
        ):
            continue
        novelty_items.append(novelty_suggestion(frame, strength, preset))

    novelty_items = filter_redundant_novelties(novelty_items, regions, sample_rate)
    suggestions.extend(novelty_items)

    return merge_suggestions(suggestions, sample_rate, preset)


def filter_redundant_novelties(
    suggestions: Iterable[Suggestion],
    regions: Iterable,
    sample_rate: int,
) -> list[Suggestion]:
    """Drop musical-change peaks that a quiet stretch already accounts for.

    A song that stops produces a change in the sound: the release of its
    last chord. That is not a second boundary, it is the same event as the
    quiet stretch that follows it, and the quiet stretch is the better
    evidence because it says where the music actually ended.

    The same is true in the other direction: the attack of the next song is
    not a separate boundary from the gap in front of it.
    """
    regions = list(regions)
    margin = max(1, int(0.5 * sample_rate))
    before_margin = max(1, int(1.5 * sample_rate))
    # How far after a musical change a quiet stretch may start before the
    # change is understood as the music stopping rather than as a new song.
    gap_follow_seconds = 5.0

    kept: list[Suggestion] = []
    for suggestion in suggestions:
        frame = suggestion.frame
        if any(
            region.start_frame - before_margin <= frame <= region.end_frame + margin
            for region in regions
        ):
            continue
        if any(
            frame < region.start_frame <= frame + gap_follow_seconds * sample_rate
            for region in regions
        ):
            continue
        kept.append(suggestion)
    return kept


def silence_suggestion(
    region,
    profile,
    preset: SensitivityPreset,
    placement: Optional[PlacementResult] = None,
) -> Suggestion:
    """Build the suggestion for one silence region."""
    sample_rate = getattr(profile, "sample_rate", 44100) or 44100
    duration = region.frames() / float(sample_rate)
    depth = max(0.0, profile.threshold_dbfs - region.level_dbfs)
    confidence = (
        0.30
        + 0.08 * min(5.0, duration)
        + 0.20 * min(1.0, depth / 18.0)
    ) * preset.silence_confidence_scale

    details = {
        "silence_seconds": round(duration, 2),
        "silence_level_dbfs": round(float(region.level_dbfs), 1),
        "noise_floor_dbfs": round(float(profile.noise_floor_dbfs), 1),
        "detection_threshold_dbfs": round(float(profile.threshold_dbfs), 1),
    }

    if placement is None:
        frame = (region.start_frame + region.end_frame) // 2
        reason = (
            f"{duration:.1f} s of quiet here. The cut is in the middle of it."
        )
        uncertain = False
        onset_frame = -1
        transition_start = region.start_frame
        transition_end = region.end_frame
        placement_mode = PLACEMENT_NEXT_START
    else:
        frame = placement.frame
        reason = placement.reason
        uncertain = placement.uncertain
        onset_frame = placement.onset_frame
        transition_start = placement.transition_start
        transition_end = placement.transition_end
        placement_mode = placement.details.get("placement", PLACEMENT_NEXT_START)
        confidence *= placement.confidence_scale
        details.update(placement.details)

    # Silence is evidence, not proof: never claim certainty.
    confidence = min(0.85, confidence)

    return Suggestion(
        frame=frame,
        confidence=confidence,
        reason=reason,
        kind="silence",
        uncertain=uncertain,
        placement=placement_mode,
        onset_frame=onset_frame,
        transition_start=transition_start,
        transition_end=transition_end,
        details=details,
    )


def novelty_suggestion(frame: int, strength: float, preset: SensitivityPreset) -> Suggestion:
    """Build the suggestion for one musical-change peak."""
    confidence = min(0.55, 0.15 + 0.4 * strength) * preset.novelty_confidence_scale
    return Suggestion(
        frame=int(frame),
        confidence=confidence,
        reason=(
            "The music changed without a break in the sound, so this is an "
            "uncertain transition. Listen to it and decide where the cut goes."
        ),
        kind="novelty",
        uncertain=True,
        details={"novelty_strength": round(float(strength), 2)},
    )


def merge_suggestions(
    suggestions: Iterable[Suggestion],
    sample_rate: int,
    preset: SensitivityPreset,
) -> list[Suggestion]:
    """Merge candidates that describe the same event.

    Two different rules apply, because the two kinds of evidence are
    different things:

    * two silence regions very close together are one quiet passage,
    * two musical-change peaks close together are one musical event.

    The separation window for musical changes must never be applied to
    silences: two real gaps 3 seconds apart are two gaps, not one.
    """
    ordered = sorted(suggestions, key=lambda s: s.frame)
    silence_separation = max(1, int(0.5 * sample_rate))
    novelty_separation = max(1, int(preset.min_novelty_separation_seconds * sample_rate))

    silences: list[Suggestion] = []
    novelties: list[Suggestion] = []
    for suggestion in ordered:
        if suggestion.kind == "novelty":
            novelties.append(suggestion)
        else:
            silences.append(suggestion)

    silences = _collapse(silences, silence_separation)
    novelties = _collapse(novelties, novelty_separation)

    # A musical-change peak next to a silence is the same event seen twice.
    kept_novelties: list[Suggestion] = []
    for novelty in novelties:
        if any(abs(novelty.frame - silence.frame) <= silence_separation
               for silence in silences):
            continue
        kept_novelties.append(novelty)

    return sorted(silences + kept_novelties, key=lambda s: s.frame)


def _collapse(items: list[Suggestion], separation: int) -> list[Suggestion]:
    """Collapse candidates that are closer than ``separation`` frames."""
    collapsed: list[Suggestion] = []
    for item in items:
        if collapsed and item.frame - collapsed[-1].frame < separation:
            previous = collapsed[-1]
            # Silence evidence beats a musical change; otherwise keep the
            # more confident one.
            if previous.kind == "silence" and item.kind == "novelty":
                continue
            if item.kind == "silence" and previous.kind == "novelty":
                collapsed[-1] = item
                continue
            if item.confidence > previous.confidence:
                collapsed[-1] = item
            continue
        collapsed.append(item)
    return collapsed
