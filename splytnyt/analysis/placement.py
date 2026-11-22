"""Choosing where the cut goes.

The detector finds **transition regions**: places where the recording changes
character, such as a stretch of quiet between two songs. That is a different
question from *where inside that region the cut belongs*, and the old code
answered it wrongly - it always cut in the middle of the quiet, which leaves
the next song starting with a tail of silence.

Four placements are offered:

``next_start`` (default)
    The cut goes just before the next credible musical onset, with a small
    configurable margin (30 ms by default). Everything between the
    previous song and that onset - silence, applause, room noise - stays at
    the end of the previous track. A soft intro, a pickup note or a breath
    that belongs to the next song is preserved, because the onset is found
    from the music itself rather than from the loudest sound in the gap.

``previous_end``
    The cut goes where the previous song stopped. Useful when the silence
    before the next song should belong to the previous track.

``gap_middle``
    The cut goes in the middle of the quiet gap. This is the historical
    behaviour; it is offered only as an explicit choice.

``manual``
    The user places the cut by hand.

Two things the detector refuses to do:

* It does not cut a **quiet passage inside a song**. The music before the
  quiet stretch and the music after it are compared; when it is the same
  music, the quiet stretch is not a boundary.
* It does not pretend that applause or speech is music. When the material
  after the gap cannot be told apart from applause, the boundary is marked
  **uncertain** and the interface asks for a human decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .onset import (
    SEARCH_LIMIT_SECONDS,
    TONALITY_THRESHOLD,
    OnsetSearch,
    find_next_music_onset,
)

#: The default margin is defined in the data model, so that the project, the
#: analysis and the refinement cannot drift apart.
from ..core.model import DEFAULT_PRE_ROLL_SECONDS

#: Cut just before the next musical onset.
PLACEMENT_NEXT_START = "next_start"
#: Cut where the previous song stopped.
PLACEMENT_PREVIOUS_END = "previous_end"
#: Cut in the middle of the quiet gap.
PLACEMENT_GAP_MIDDLE = "gap_middle"
#: The user places the cut.
PLACEMENT_MANUAL = "manual"

PLACEMENT_MODES = (
    PLACEMENT_NEXT_START,
    PLACEMENT_PREVIOUS_END,
    PLACEMENT_GAP_MIDDLE,
    PLACEMENT_MANUAL,
)

PLACEMENT_LABELS = {
    PLACEMENT_NEXT_START: "Near next song start",
    PLACEMENT_PREVIOUS_END: "Near previous song end",
    PLACEMENT_GAP_MIDDLE: "Middle of gap",
    PLACEMENT_MANUAL: "Manual",
}

PLACEMENT_HINTS = {
    PLACEMENT_NEXT_START: (
        "The cut goes just before the next musical onset. The quiet and the "
        "applause stay at the end of the previous track."
    ),
    PLACEMENT_PREVIOUS_END: (
        "The cut goes where the previous song stopped. The quiet before the "
        "next song belongs to the previous track."
    ),
    PLACEMENT_GAP_MIDDLE: (
        "The cut goes in the middle of the quiet gap. The next track starts "
        "with the tail of the gap."
    ),
    PLACEMENT_MANUAL: "You place the cut yourself.",
}

#: How far before the musical onset the cut is placed by default.
#:
#: The default is deliberately small: the onset is the moment the music
#: begins, so a cut that is far in front of it drags the tail of the previous
#: song, or the applause, into the start of the next track. A short margin
#: keeps the first note from being clipped without dragging anything in.
#: (The value itself comes from the data model, so that the project, the
#: analysis and the refinement always agree.)

#: The margin used by the "tight start" preset.
TIGHT_PRE_ROLL_SECONDS = 0.020
#: The margin used by the "keep more lead-in" preset.
LEAD_IN_PRE_ROLL_SECONDS = 0.150

#: How much audio is compared on each side of a gap when deciding whether
#: the music simply continues across it.
CONTEXT_SECONDS = 2.0
#: Below this timbre distance the music before and after the gap is the same
#: music, so the gap is a quiet passage inside a song and not a boundary.
SAME_MUSIC_DISTANCE = 0.06
#: Below this (but above the first threshold) the boundary is proposed but
#: marked uncertain: the two songs may be similar.
UNCERTAIN_DISTANCE = 0.15
#: Window used for the timbre comparison. A long window resolves the
#: difference between two chords; a short one cannot tell them apart.
TIMBRE_N_FFT = 4096
TIMBRE_HOP = 1024


@dataclass
class PlacementChoice:
    """The user's boundary placement preference."""

    mode: str = PLACEMENT_NEXT_START
    pre_roll_seconds: float = DEFAULT_PRE_ROLL_SECONDS

    @property
    def label(self) -> str:
        return PLACEMENT_LABELS.get(self.mode, PLACEMENT_LABELS[PLACEMENT_NEXT_START])

    @property
    def hint(self) -> str:
        return PLACEMENT_HINTS.get(self.mode, "")

    def normalised(self) -> "PlacementChoice":
        mode = self.mode if self.mode in PLACEMENT_MODES else PLACEMENT_NEXT_START
        pre_roll = max(0.0, min(2.0, float(self.pre_roll_seconds)))
        return PlacementChoice(mode=mode, pre_roll_seconds=pre_roll)


@dataclass
class PlacementResult:
    """Where a cut was placed, and why."""

    frame: int
    reason: str
    uncertain: bool = False
    confidence_scale: float = 1.0
    #: The musical onset the cut was derived from, or -1.
    onset_frame: int = -1
    #: The transition region the placement was made inside.
    transition_start: int = -1
    transition_end: int = -1
    #: Set when the placement is *not* a boundary at all.
    rejected: bool = False
    #: Technical numbers behind the decision.
    details: dict = field(default_factory=dict)


def place_cut(
    signal: np.ndarray,
    sample_rate: int,
    region,
    choice: Optional[PlacementChoice] = None,
    floor_dbfs: Optional[float] = None,
    search_end: Optional[int] = None,
    manual_frame: Optional[int] = None,
    base_frame: int = 0,
) -> PlacementResult:
    """Decide where the cut goes inside a transition region.

    ``region`` is a silence region: a stretch of quiet that marks a possible
    boundary. ``choice`` says how the user wants the cut placed inside it.

    ``signal`` may be a window of the recording rather than the whole thing;
    ``base_frame`` is the absolute timeline frame the window starts at. All
    frames in the result are absolute.
    """
    choice = (choice or PlacementChoice()).normalised()
    start = int(region.start_frame)
    end = int(region.end_frame)
    local_start = max(0, start - int(base_frame))
    local_end = max(local_start, end - int(base_frame))

    if choice.mode == PLACEMENT_MANUAL:
        frame = int(manual_frame) if manual_frame is not None else start
        frame = max(0, frame)
        return PlacementResult(
            frame=frame,
            reason="You placed this cut by hand.",
            uncertain=False,
            confidence_scale=1.0,
            transition_start=start,
            transition_end=end,
            details={"placement": PLACEMENT_MANUAL},
        )

    if choice.mode == PLACEMENT_PREVIOUS_END:
        return PlacementResult(
            frame=start,
            reason=(
                "The cut is where the previous song stopped, so the quiet "
                "before the next song belongs to the previous track."
            ),
            uncertain=False,
            confidence_scale=0.9,
            transition_start=start,
            transition_end=end,
            details={"placement": PLACEMENT_PREVIOUS_END},
        )

    if choice.mode == PLACEMENT_GAP_MIDDLE:
        return PlacementResult(
            frame=(start + end) // 2,
            reason=(
                "The cut is in the middle of the quiet gap, so the next track "
                "starts with the tail of the gap."
            ),
            uncertain=False,
            confidence_scale=0.7,
            transition_start=start,
            transition_end=end,
            details={"placement": PLACEMENT_GAP_MIDDLE},
        )

    # --- the default: near the start of the next song ---------------------
    limit = int(signal.size)
    if search_end is not None:
        limit = min(limit, int(search_end) - int(base_frame))
    else:
        limit = min(limit, local_end + int(SEARCH_LIMIT_SECONDS * sample_rate))

    onset = find_next_music_onset(
        signal,
        sample_rate,
        search_start=local_start,
        search_end=limit,
        floor_dbfs=floor_dbfs,
    )
    if onset.found:
        # The search worked in the coordinates of the window it was given;
        # the result is reported in the timeline's coordinates.
        onset.frame += int(base_frame)
    result = _placement_from_onset(onset, region, choice, sample_rate)

    if result.rejected or onset.frame < 0:
        return result

    # Is the music simply continuing across the gap? If the timbre before
    # and after the gap is the same, this quiet stretch is inside a song and
    # not a boundary between two of them.
    #
    # The comparison uses the material *after the quiet stretch*, not the
    # material right after the onset: a quiet passage inside a song is
    # followed by the same song again, and that is what has to be recognised.
    after_start = max(local_end, onset.frame - int(base_frame))
    distance = timbre_distance_across(
        signal,
        sample_rate,
        before_end=local_start,
        after_start=after_start,
    )
    result.details["timbre_distance"] = round(distance, 3)
    if distance < SAME_MUSIC_DISTANCE:
        # The music either side of the gap is the same. That is either a
        # quiet passage inside one song, or two songs that happen to sound
        # alike. What settles it is whether the quiet stretch itself still
        # contains music: a passage holds quiet music, a break holds only
        # room noise.
        gap_tonality = _gap_tonality(signal, sample_rate, local_start, local_end)
        result.details["gap_tonality"] = round(gap_tonality, 2)
        if gap_tonality >= TONALITY_THRESHOLD:
            result.rejected = True
            result.reason = (
                "The music before and after this quiet stretch is the same music, "
                "and the stretch still holds music of its own, so this is a quiet "
                "passage inside a song and not a boundary."
            )
        else:
            result.uncertain = True
            result.confidence_scale = min(result.confidence_scale, 0.6)
            result.reason += (
                " The music either side of the gap is very similar, so this "
                "boundary needs a human decision."
            )
    elif distance < UNCERTAIN_DISTANCE:
        result.uncertain = True
        result.confidence_scale = min(result.confidence_scale, 0.6)
        result.reason += (
            " The music either side of the gap is very similar, so this "
            "boundary needs a human decision."
        )
    return result


def _gap_tonality(
    signal: np.ndarray,
    sample_rate: int,
    start: int,
    end: int,
) -> float:
    """How musical the material inside a quiet stretch is.

    The middle of the stretch is used: the edges always contain a little of
    the music on either side.
    """
    from .onset import music_hop_profile

    span = end - start
    if span <= 0:
        return 0.0
    middle_start = start + span // 4
    middle_end = end - span // 4
    if middle_end - middle_start < sample_rate // 8:
        middle_start, middle_end = start, end
    chunk = signal[middle_start:middle_end]
    if chunk.size == 0:
        return 0.0
    _levels, tonality = music_hop_profile(chunk.astype(np.float32, copy=False), sample_rate)
    if tonality.size == 0:
        return 0.0
    return float(np.median(tonality))


def _placement_from_onset(
    onset: OnsetSearch,
    region,
    choice: PlacementChoice,
    sample_rate: int,
) -> PlacementResult:
    """Turn an onset search into a cut position."""
    start = int(region.start_frame)
    end = int(region.end_frame)
    pre_roll_frames = int(round(choice.pre_roll_seconds * sample_rate))

    if not onset.found:
        # Nothing credible was found. Put the cut where the previous song
        # stopped: that keeps every scrap of audio, invents nothing, and
        # leaves the decision to a human.
        return PlacementResult(
            frame=start,
            reason=(
                "No clear musical start was found after the gap, so the cut is "
                "at the end of the previous song. This boundary needs a human "
                "decision."
            ),
            uncertain=True,
            confidence_scale=0.5,
            transition_start=start,
            transition_end=end,
            details=dict(onset.details, placement=PLACEMENT_NEXT_START),
        )

    frame = int(onset.frame) - pre_roll_frames
    # Never cut before the transition begins: the previous song's own music
    # has to stay with the previous song.
    frame = max(start, frame)
    frame = max(0, frame)

    pre_roll_ms = int(round(choice.pre_roll_seconds * 1000))
    # How far the cut actually ended up in front of the onset. It is not
    # always the requested margin: the cut is never placed before the
    # transition begins, so a small margin in a short gap can end up as no
    # margin at all.
    applied_ms = int(round((int(onset.frame) - frame) / sample_rate * 1000))
    if onset.confident:
        reason = (
            f"The next song starts {applied_ms} ms after this cut, so the "
            "first note is not clipped. The quiet and the applause stay at the "
            "end of the previous track."
        )
    else:
        reason = (
            f"This cut sits {applied_ms} ms before the first point that could "
            "be the start of the next song. The detector could not tell music "
            "apart from applause or speech here, so please listen before you "
            "accept it."
        )

    return PlacementResult(
        frame=frame,
        reason=reason,
        uncertain=not onset.confident,
        confidence_scale=1.0 if onset.confident else 0.7,
        onset_frame=int(onset.frame),
        transition_start=start,
        transition_end=end,
        details=dict(
            onset.details,
            placement=PLACEMENT_NEXT_START,
            requested_pre_roll_ms=pre_roll_ms,
            applied_pre_roll_ms=applied_ms,
        ),
    )


def timbre_distance_across(
    signal: np.ndarray,
    sample_rate: int,
    before_end: int,
    after_start: int,
    context_seconds: float = CONTEXT_SECONDS,
) -> float:
    """How different the music before a gap is from the music after it.

    Returns 0 for the same music and 1 for completely different music. When
    there is not enough audio on one side, the maximum distance is returned,
    because "cannot tell" must never mean "same music".
    """
    from .novelty import spectral_shapes

    context = int(context_seconds * sample_rate)
    before_slice = signal[max(0, before_end - context): before_end]
    after_slice = signal[after_start: after_start + context]

    if before_slice.size < sample_rate // 2 or after_slice.size < sample_rate // 2:
        return 1.0

    before_shapes = spectral_shapes(
        before_slice.astype(np.float32, copy=False),
        sample_rate,
        hop_frames=TIMBRE_HOP,
        n_fft=TIMBRE_N_FFT,
    )
    after_shapes = spectral_shapes(
        after_slice.astype(np.float32, copy=False),
        sample_rate,
        hop_frames=TIMBRE_HOP,
        n_fft=TIMBRE_N_FFT,
    )
    if before_shapes.shape[0] < 2 or after_shapes.shape[0] < 2:
        return 1.0

    before_mean = np.median(before_shapes, axis=0)
    after_mean = np.median(after_shapes, axis=0)
    norm_before = float(np.linalg.norm(before_mean))
    norm_after = float(np.linalg.norm(after_mean))
    if norm_before <= 1e-12 or norm_after <= 1e-12:
        return 1.0
    similarity = float(np.dot(before_mean, after_mean) / (norm_before * norm_after))
    return max(0.0, 1.0 - similarity)


def refine_cut(
    signal: np.ndarray,
    sample_rate: int,
    region,
    choice: Optional[PlacementChoice] = None,
    floor_dbfs: Optional[float] = None,
    search_end: Optional[int] = None,
    base_frame: int = 0,
) -> PlacementResult:
    """Re-derive a cut inside a transition region that is already known.

    This is what "refine existing boundaries" uses: the transition region is
    remembered from the first analysis, and the cut inside it is worked out
    again with the current placement preference.
    """
    return place_cut(
        signal,
        sample_rate,
        region,
        choice=choice,
        floor_dbfs=floor_dbfs,
        search_end=search_end,
        base_frame=base_frame,
    )
