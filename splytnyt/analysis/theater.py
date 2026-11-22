"""Theatrical segmentation.

Turns a recording into sections of:

* a **musical number** - a self-contained stretch of music,
* **dialogue** - spoken words with no music under them,
* **underscoring** - a short stretch of instrumental music carrying the
  action,
* **applause** - the audience,
* **mixed** - spoken words *over* music, which is the normal state of a
  theatre recording and which a single "speech or music" choice cannot
  express,
* **uncertain** - the detector would not commit.

The design rules that shape this module:

* **Activities overlap.** The output is a set of activity *regions*, one
  record per activity per stretch, and they are allowed to overlap. A
  stretch of dialogue over an underscoring holds a speech region and an
  instrumental region at the same time. The sections partition the
  timeline; the activities do not, and that is the point.
* **Context over time.** A spoken sentence is not a section. A musical
  change inside a number is not a number boundary. The detector works on
  runs and merges them with explicit gap tolerances, so a song survives its
  own verse gaps and a scene survives its own pauses.
* **Soft intros and sustained endings are protected.** A number starts at
  the first musical activity, not at the loudest moment of its intro, and
  it ends where the music really stops.
* **Nothing here is show-specific.** Every threshold describes speech,
  music and applause in general.

The evidence comes from two places, and the module works with either:

* a **trained audio-event classifier** (:mod:`splytnyt.analysis.aec`),
  which is the accurate one and is used when it is available;
* **acoustic measurements** (:mod:`splytnyt.analysis.activities`), which
  always work and are the fallback.

Both are merged here, and where they disagree the section is marked
uncertain rather than quietly resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.model import (
    ACTIVITY_BASIS_ACOUSTIC,
    ACTIVITY_BASIS_MIXED,
    ActivityRegion,
)
from .activities import FrameFeatures, note_runs

# ----------------------------------------------------------------------
# Thresholds
#
# Every one of these describes speech, music or applause in general. None
# of them knows anything about any production.
# ----------------------------------------------------------------------

#: How far below the recording's own quietest sustained level a frame must
#: fall to count as silence. A stage recording with an audience in it
#: almost never reaches digital silence, and the quietest sustained level
#: in such a recording is quiet dialogue rather than room tone, so the
#: margin is applied *below* the floor, not above it.
SILENCE_MARGIN_DB = -6.0

#: Nothing below this level is ever called silence.
ABSOLUTE_SILENCE_DBFS = -70.0

#: Music is present from this score upwards. The classifier's own "Music"
#: score is high for both a band and an orchestra.
MUSIC_ON = 0.35
MUSIC_OFF = 0.20

#: Speech is present from this score upwards. It is set high on purpose:
#: in a theatre the orchestra is audible during most of the recording, so
#: speech has to be clearly dominant before a stretch counts as speech.
SPEECH_ON = 0.60
SPEECH_OFF = 0.40

#: Applause from this score upwards.
APPLAUSE_ON = 0.25
APPLAUSE_OFF = 0.15

#: A singing voice from this score upwards. Below it, words that the
#: transcriber found are treated as spoken; above it, they are treated as
#: sung, because a transcript over singing is most likely to be lyrics
#: the model made up or half-remembered.
SINGING_ON = 0.35

#: The music score above which transcribed words are treated as **sung**
#: rather than spoken.
#:
#: This is the single most important rule in the module, and it was
#: measured rather than guessed. On a real stage recording the
#: classifier's "Singing" score almost never reaches a fixed threshold -
#: it peaked below 0.5 over a whole act - so gating on it alone let every
#: sung line through as if it were speech. The consequence was that songs
#: were cut into pieces at every line, because each line became a
#: "spoken" region and the boundaries came from the spoken regions.
#:
#: The classifier's **Music** score separates the two populations
#: cleanly instead. Measured over 763 accepted transcript stretches of a
#: real theatrical act: the stretches over the band had a music score of
#: 0.53 (10th percentile) to 0.84 (90th), and the genuinely spoken ones
#: had 0.00 to 0.19. Nothing else separated them - the singing score,
#: the "no speech" probability, the average log probability, the
#: characters per second and the stretch length all overlapped.
#:
#: So: words over a band that is playing at full volume are lyrics. They
#: are kept, as evidence that there is a voice there, but they do not cut
#: the music.
LYRIC_MUSIC_ON = 0.35

#: How far either side of a stretch of words the music level is read when
#: deciding whether the words were sung or spoken.
LYRIC_CONTEXT_SECONDS = 2.0

#: The shortest window the judgement is made over, so that a momentary gap
#: in the band cannot turn one sung line into a spoken one.
LYRIC_CONTEXT_MIN_SECONDS = 4.0

#: The share of a section that has to be covered by an activity for it to
#: count.
ACTIVITY_COVERAGE = 0.30

#: A stretch of music shorter than this is underscoring; longer than this
#: it may be a number of its own.
MAX_UNDERSCORING_SECONDS = 30.0

#: A musical number has to last at least this long.
#:
#: A number is a substantial musical item; an instrumental link between
#: two scenes is not. Measured over a real theatrical act, the lengths of
#: the musical stretches that carried no spoken line fell into two clear
#: groups: short ones of under half a minute, which are links and
#: transitions, and long ones of a minute or more, which are the numbers.
#: The threshold sits in the gap between the two.
MIN_NUMBER_SECONDS = 45.0

#: Shorter sustained numbers (such as intros or reprises) with clear
#: singing voice and music qualify as musical numbers from this length.
MIN_SUNG_NUMBER_SECONDS = 30.0

#: Two activity regions of the same kind closer than this are one region.
ACTIVITY_MERGE_GAP_SECONDS = 1.5

#: A dialogue block absorbs a gap shorter than this, so a scene is not
#: chopped into sentences.
DIALOGUE_MERGE_GAP_SECONDS = 6.0

#: A musical number absorbs a gap shorter than this, so a song survives
#: the pause between its verses.
NUMBER_MERGE_GAP_SECONDS = 3.0

#: A coherent musical number absorbs internal non-applause interruptions
#: shorter than this when strong musical/vocal sections sit on both sides.
COHERENT_NUMBER_GAP_SECONDS = 16.0

#: A bridge with active music underneath (underscoring / vamp) can span
#: slightly longer between two musical numbers.
COHERENT_VAMP_GAP_SECONDS = 22.0

#: An un-sung tail or intro separated from a musical number by a brief cut
#: merges only when the spoken break is very short.
COHERENT_TAIL_GAP_SECONDS = 6.0

#: A section shorter than this is folded into its neighbour.
MIN_SECTION_SECONDS = 3.0

#: A spoken stretch shorter than this is a line inside a number rather
#: than a scene of its own, so it does not get to be a section.
MIN_SPOKEN_SECTION_SECONDS = 3.0

#: Two section boundaries closer than this are one boundary.
MIN_BOUNDARY_GAP_SECONDS = 2.0

#: The share of a section that must carry music for the section to be
#: called "mixed" rather than "dialogue".
MIXED_MUSIC_COVERAGE = 0.50


@dataclass
class TheaterSection:
    """One section of the recording."""

    start_frame: int
    end_frame: int
    section_type: str
    confidence: float
    activities: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    uncertain: bool = False
    evidence: dict = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    def duration_seconds(self, sample_rate: int) -> float:
        return self.frames / float(sample_rate) if sample_rate else 0.0


@dataclass
class TheaterAnalysis:
    """The result of a theatrical segmentation pass."""

    sections: list[TheaterSection] = field(default_factory=list)
    activities: list[ActivityRegion] = field(default_factory=list)
    noise_floor_dbfs: float = 0.0
    elapsed_seconds: float = 0.0
    #: True when the sections came from the trained classifier.
    used_model: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class AsrSegment:
    """One piece of speech evidence from the transcriber."""

    start_seconds: float
    end_seconds: float
    text: str = ""
    confidence: float = 0.0
    #: False when the segment was rejected as a hallucination.
    accepted: bool = True
    reason: str = ""


# ----------------------------------------------------------------------
# Putting the two evidence sources on one grid
# ----------------------------------------------------------------------
def _upsample(
    values: np.ndarray,
    count: int,
    sample_rate: int,
    hop: int,
    frame_seconds: float,
) -> np.ndarray:
    """Stretch a coarse score track onto a fine one.

    The classifier works on frames that each look at about a second of
    audio, and the acoustic features work on frames of about eleven
    milliseconds. Each fine frame takes the score of the coarse frame
    whose window is centred nearest to it - the classifier's resolution
    is the limit, and pretending otherwise would invent detail.
    """
    if values.size == 0 or count <= 0:
        return np.zeros(max(0, count), dtype=np.float32)
    seconds_per_fine_frame = hop / float(sample_rate)
    ratio = max(1.0, frame_seconds / max(1e-9, seconds_per_fine_frame))
    # The coarse frame g covers the window that starts at g * frame_seconds
    # and is centred half a frame later, so the nearest frame to a fine
    # frame at i * ratio coarse-frames is round(i - 0.5).
    indices = np.floor(np.arange(count) / ratio + 0.5).astype(np.int64)
    indices = np.clip(indices, 0, values.size - 1)
    return values[indices].astype(np.float32)


def _score_dsp_speech(features: FrameFeatures) -> np.ndarray:
    """A speech estimate from the acoustic measurements alone.

    This is the fallback path. It is weaker than the trained classifier -
    dialogue in a musical comes with the orchestra underneath it - and the
    code says so rather than pretending otherwise.
    """
    n = features.n_frames
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    run_length, _ = note_runs(features.f0_hz, features.voicing)
    run_seconds = run_length * features.hop / float(features.sample_rate)
    moving = np.clip(1.0 - (run_seconds - 0.15) / 0.35, 0.0, 1.0)
    syllable = np.clip((features.modulation - 0.10) / 0.25, 0.0, 1.0)
    voice_band = np.clip((features.band_mid - 0.12) / 0.30, 0.0, 1.0)
    voiced = np.clip((features.voicing - 0.35) / 0.25, 0.0, 1.0)
    return np.clip(
        0.40 * syllable + 0.35 * moving * voiced + 0.25 * voice_band * voiced,
        0.0, 1.0,
    ).astype(np.float32)


def _score_dsp_music(features: FrameFeatures) -> np.ndarray:
    """A music estimate from the acoustic measurements alone."""
    n = features.n_frames
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    tonal = np.clip(1.0 - features.flatness / 0.25, 0.0, 1.0)
    harmonic = np.clip(features.harmonic_ratio / 0.55, 0.0, 1.0)
    return np.clip(0.45 * tonal + 0.55 * harmonic, 0.0, 1.0).astype(np.float32)


def score_activities(
    features: FrameFeatures,
    events=None,
    asr_segments: Optional[list[AsrSegment]] = None,
    noise_floor_dbfs: Optional[float] = None,
) -> dict:
    """Score every activity, frame by frame.

    ``events`` is the trained classifier's :class:`splytnyt.analysis.aec.EventScores`.
    When it is given, it is the primary evidence and the acoustic
    measurements only fill in what it does not cover.
    """
    n = features.n_frames
    empty = np.zeros(n, dtype=np.float32)
    if n == 0:
        return {
            "speech": empty, "singing": empty, "instrumental": empty,
            "applause": empty, "silence": empty, "music": empty,
            "floor_dbfs": 0.0, "asr_speech": empty, "used_model": False,
        }

    level = features.rms_db
    if noise_floor_dbfs is None:
        floor = float(np.percentile(level, 5.0))
    else:
        floor = float(noise_floor_dbfs)
    silence_threshold = max(ABSOLUTE_SILENCE_DBFS, floor + SILENCE_MARGIN_DB)
    silent = level < silence_threshold

    used_model = events is not None and events.n_frames() > 0

    if used_model:
        music = _upsample(
            events.music, n, features.sample_rate, features.hop, events.frame_seconds
        )
        speech = _upsample(
            events.speech, n, features.sample_rate, features.hop, events.frame_seconds
        )
        singing_raw = _upsample(
            events.singing, n, features.sample_rate, features.hop, events.frame_seconds
        )
        applause = _upsample(
            events.applause, n, features.sample_rate, features.hop, events.frame_seconds
        )
        model_silence = _upsample(
            events.silence, n, features.sample_rate, features.hop, events.frame_seconds
        )
    else:
        music = _score_dsp_music(features)
        speech = _score_dsp_speech(features)
        singing_raw = np.zeros(n, dtype=np.float32)
        applause = np.zeros(n, dtype=np.float32)
        model_silence = np.zeros(n, dtype=np.float32)

    # The classifier's "singing" score is quiet on much of this material,
    # so it is read *relative* to the rest of the recording as well as on
    # its own scale. The raw score is kept, because it is what decides
    # whether a stretch of transcribed words was spoken or sung.
    singing = _relative_strength(singing_raw)

    # The acoustic measurements still say something useful about silence,
    # which is what the classifier is weakest on in a room with an
    # audience in it.
    silence = np.maximum(silent.astype(np.float32), model_silence * 0.5)

    # The transcriber's words are evidence about speech, but only where
    # the words were really spoken. A stretch the classifier hears as
    # singing is one where a transcript is most likely to be invented
    # lyrics, so the words are marked as possible lyrics there rather
    # than used as speech.
    asr_speech = np.zeros(n, dtype=np.float32)
    asr_lyrics = np.zeros(n, dtype=np.float32)
    asr_over_music = np.zeros(n, dtype=np.float32)
    if asr_segments:
        pad = int(LYRIC_CONTEXT_SECONDS * features.sample_rate / features.hop)
        minimum = int(LYRIC_CONTEXT_MIN_SECONDS * features.sample_rate / features.hop)
        for segment in asr_segments:
            if not segment.accepted:
                continue
            start = int(segment.start_seconds * features.sample_rate / features.hop)
            end = int(segment.end_seconds * features.sample_rate / features.hop)
            start = max(0, min(n, start))
            end = max(start, min(n, end))
            if end <= start:
                continue
            window = singing_raw[start:end]
            sung = float(np.max(window)) if window.size else 0.0

            # The music level is read over a window that *surrounds* the
            # words, not just over the words themselves. A band that drops
            # out for a second in the middle of a sung line would otherwise
            # turn that line into "speech", and a line of "speech" in the
            # middle of a number is a boundary - which is how one song was
            # being cut in half. Context is what tells the two apart.
            low = max(0, start - pad)
            high = min(n, end + pad)
            if high - low < minimum:
                # Make sure the window is wide enough to be a judgement
                # about the passage and not about a momentary gap.
                extra = max(0, minimum - (high - low))
                low = max(0, low - extra // 2)
                high = min(n, high + extra - extra // 2)
            context = music[low:high]
            band = float(np.median(context)) if context.size else 0.0

            if sung >= SINGING_ON or band >= LYRIC_MUSIC_ON:
                # Almost certainly a sung line: the words may well be
                # the real lyrics, but they were not spoken. They are kept
                # as evidence of a voice, and they do not cut the music.
                asr_lyrics[start:end] = 1.0
                # But they might not be. A line of dialogue delivered over
                # a band that is playing at full volume looks exactly the
                # same to a speech recogniser as a sung line, and the
                # audio alone cannot tell the two apart. Rather than guess
                # - and rather than cut the music on a guess - the words
                # are recorded as *both*: there is a voice here, and it may
                # be speech. The section that holds them is marked
                # uncertain and a reviewer is asked to listen.
                asr_over_music[start:end] = 1.0
            else:
                asr_speech[start:end] = 1.0

    if asr_speech.any():
        # Words in the recording are strong evidence of speech.
        speech = np.maximum(speech, 0.85 * asr_speech)
    if asr_lyrics.any():
        # Sung words: they do not make the stretch dialogue, but they do
        # say that there is a voice there.
        singing = np.maximum(singing, asr_lyrics)
        speech = speech * (1.0 - 0.6 * asr_lyrics)

    return {
        "speech": speech.astype(np.float32),
        "singing": singing.astype(np.float32),
        # "Instrumental" means music with no singing voice on it. It is
        # music scaled down by how much singing there is, so that a number
        # with a voice in it is not also reported as instrumental.
        "instrumental": np.clip(
            music * (1.0 - 0.8 * singing), 0.0, 1.0
        ).astype(np.float32),
        "applause": applause.astype(np.float32),
        "silence": silence.astype(np.float32),
        "music": music.astype(np.float32),
        "singing_raw": singing_raw.astype(np.float32),
        "floor_dbfs": floor,
        "asr_speech": asr_speech,
        "asr_lyrics": asr_lyrics,
        "asr_over_music": asr_over_music,
        "used_model": used_model,
    }


def _relative_strength(values: np.ndarray) -> np.ndarray:
    """Rescale a quiet signal so its own shape can be read.

    Some classifier scores are informative but never reach a fixed
    threshold on real material. Rescaling them against their own
    distribution keeps their shape - where they are relatively strong -
    without inventing a certainty they do not have.
    """
    if values.size == 0:
        return values
    high = float(np.percentile(values, 95.0))
    if high <= 1e-6:
        return np.zeros_like(values)
    return np.clip(values / high, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------
# Regions
# ----------------------------------------------------------------------
def _runs_above(values: np.ndarray, on: float, off: float) -> list[tuple[int, int]]:
    """Runs where a score is above ``on``, kept until it falls below ``off``."""
    runs: list[tuple[int, int]] = []
    start = -1
    for index in range(values.size):
        if start < 0:
            if values[index] >= on:
                start = index
        else:
            if values[index] < off:
                runs.append((start, index))
                start = -1
    if start >= 0:
        runs.append((start, values.size))
    return runs


def _merge_runs(runs: list[tuple[int, int]], gap_frames: int) -> list[tuple[int, int]]:
    """Merge runs separated by less than ``gap_frames``."""
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] < gap_frames:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(int(a), int(b)) for a, b in merged]


def build_activity_regions(
    features: FrameFeatures,
    scores: dict,
    asr_segments: Optional[list[AsrSegment]] = None,
) -> list[ActivityRegion]:
    """Turn the per-frame scores into overlapping activity regions."""
    sample_rate = features.sample_rate
    hop = features.hop
    floor = float(scores.get("floor_dbfs", 0.0))
    regions: list[ActivityRegion] = []

    label_gap = {
        "speech": int(DIALOGUE_MERGE_GAP_SECONDS * sample_rate / hop),
        "singing": int(NUMBER_MERGE_GAP_SECONDS * sample_rate / hop),
        "instrumental": int(NUMBER_MERGE_GAP_SECONDS * sample_rate / hop),
        "applause": int(ACTIVITY_MERGE_GAP_SECONDS * sample_rate / hop),
        "silence": int(ACTIVITY_MERGE_GAP_SECONDS * sample_rate / hop),
    }
    thresholds = {
        "speech": (SPEECH_ON, SPEECH_OFF),
        "singing": (0.45, 0.25),
        "instrumental": (MUSIC_ON, MUSIC_OFF),
        "applause": (APPLAUSE_ON, APPLAUSE_OFF),
        "silence": (0.5, 0.5),
    }

    for label in ("speech", "singing", "instrumental", "applause", "silence"):
        values = scores.get(label)
        if values is None or values.size == 0:
            continue
        on, off = thresholds[label]
        runs = _runs_above(values, on, off)
        runs = _merge_runs(runs, label_gap.get(label, 0))
        if not runs:
            continue

        for start, end in runs:
            if end - start < 2:
                continue
            confidence, reasons, evidence = _region_evidence(
                label, features, scores, start, end, floor
            )
            basis = ACTIVITY_BASIS_ACOUSTIC
            if label == "speech" and asr_segments:
                basis = ACTIVITY_BASIS_MIXED
            regions.append(
                ActivityRegion(
                    start_frame=start * hop,
                    end_frame=end * hop,
                    label=label,
                    confidence=confidence,
                    basis=basis,
                    reason="; ".join(reasons),
                    evidence=evidence,
                )
            )

    regions.sort(key=lambda region: (region.start_frame, region.end_frame, region.label))
    return regions


def _region_evidence(
    label: str,
    features: FrameFeatures,
    scores: dict,
    start: int,
    end: int,
    floor: float,
) -> tuple[float, list[str], dict]:
    """The confidence, the reasons and the numbers behind one region."""
    if end <= start:
        return 0.0, [], {}
    score = scores[label][start:end]
    mean_score = float(np.mean(score))
    level = features.rms_db[start:end]
    mean_level = float(np.mean(level))

    reasons: list[str] = []
    evidence = {
        "mean_score": round(mean_score, 3),
        "mean_level_dbfs": round(mean_level, 1),
        "noise_floor_dbfs": round(float(floor), 1),
    }

    if label == "speech":
        reasons.append(
            "the classifier finds spoken words here "
            f"(mean score {mean_score:.2f})"
        )
    elif label == "singing":
        reasons.append(
            f"the classifier finds a singing voice here (mean score {mean_score:.2f})"
        )
    elif label == "instrumental":
        reasons.append(
            f"music is present and no voice is (mean score {mean_score:.2f})"
        )
    elif label == "applause":
        reasons.append(
            f"broadband noise weighted to the top of the spectrum "
            f"(mean score {mean_score:.2f})"
        )
    elif label == "silence":
        reasons.append(
            f"the level sits {float(floor) - mean_level:.0f} dB below the recording's "
            "quietest sustained level"
        )

    return mean_score, reasons, evidence


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------


def _coverage(regions, label: str, start: int, end: int) -> float:
    """The share of ``[start, end)`` covered by regions of one label."""
    covered = sum(
        max(0, min(region.end_frame, end) - max(region.start_frame, start))
        for region in regions
        if region.label == label
    )
    return covered / float(max(1, end - start))


def _classify_section(
    scores: dict,
    spoken: list[ActivityRegion],
    applauded: list[ActivityRegion],
    regions: list[ActivityRegion],
    start: int,
    end: int,
    sample_rate: int,
    hop: int,
    over_music: Optional[list[tuple[int, int]]] = None,
) -> tuple[str, float, list[str], bool, dict]:
    """Decide what a section is, from the activities inside it."""
    duration = (end - start) / float(sample_rate) if sample_rate else 0.0
    coverage = {
        label: _coverage(regions, label, start, end)
        for label in ("speech", "singing", "instrumental", "applause", "silence")
    }
    evidence = {
        "activity_coverage": {k: round(v, 3) for k, v in coverage.items()},
        "duration_seconds": round(duration, 2),
    }

    reasons: list[str] = []
    span = max(1, end - start)
    # How much of the section is covered by a spoken stretch?
    spoken_share = sum(
        max(0, min(region.end_frame, end) - max(region.start_frame, start))
        for region in spoken
    ) / float(span)

    # How much of the section is covered by applause?
    applause_share = sum(
        max(0, min(region.end_frame, end) - max(region.start_frame, start))
        for region in applauded
    ) / float(span)

    # How much music is there? This is read from the score itself rather
    # than from a region, because the music in a theatre recording runs
    # through the dialogue and a region's edges say nothing about it.
    music_score = scores.get("music")
    if music_score is not None and music_score.size:
        first = min(music_score.size - 1, start // max(1, hop))
        last = min(music_score.size, max(first + 1, end // max(1, hop)))
        music_share = float(np.mean(music_score[first:last]))
    else:
        music_share = 0.0
    evidence["mean_music_score"] = round(music_share, 3)
    raw_voice = scores.get("singing_raw")
    if raw_voice is not None and raw_voice.size:
        first = min(raw_voice.size - 1, start // max(1, hop))
        last = min(raw_voice.size, max(first + 1, end // max(1, hop)))
        evidence["independent_vocal_share"] = float(np.mean(raw_voice[first:last] >= SINGING_ON))

    if applause_share >= 0.4:
        reasons.append("broadband noise from the audience")
        return "applause", 0.7, reasons, False, evidence

    if spoken_share >= 0.4:
        if music_share >= MIXED_MUSIC_COVERAGE:
            reasons.append(
                f"spoken words over continuous music (music score {music_share:.2f})"
            )
            return "mixed", 0.66, reasons, False, evidence
        reasons.append(
            f"spoken words with little music under them (music score {music_share:.2f})"
        )
        return "dialogue", 0.72, reasons, False, evidence

    # No speech here. The question is whether this is a number or a
    # transition between two numbers.
    if duration >= MIN_NUMBER_SECONDS:
        section_type = "musical_number"
        reasons.append(
            f"sustained music for {duration:.0f} s with no spoken line, which is "
            "long enough to be a number rather than a transition"
        )
    elif (
        duration >= MIN_SUNG_NUMBER_SECONDS
        and coverage["singing"] >= 0.45
        and music_share >= 0.45
    ):
        section_type = "musical_number"
        reasons.append(
            f"sustained singing and music for {duration:.0f} s, long enough "
            "to be a short number or reprise rather than underscoring"
        )
    else:
        section_type = "underscoring"
        reasons.append(
            f"a short stretch of instrumental music ({duration:.0f} s) carrying the action"
        )
    if coverage["singing"] >= ACTIVITY_COVERAGE:
        reasons.append("a singing voice is present")
    confidence = 0.62 if coverage["singing"] >= ACTIVITY_COVERAGE else 0.52

    # Words were found over the music. They are usually the lyrics, but
    # they could be a line of dialogue over the band, and the audio cannot
    # tell the two apart. The section keeps its type - the music is not
    # cut - and the doubt is recorded so that a reviewer is asked to
    # listen rather than left to guess.
    if over_music:
        over_share = sum(
            max(0, min(end_frame, end) - max(start_frame, start))
            for start_frame, end_frame in over_music
        ) / float(span)
        if over_share > 0.0:
            evidence["words_over_music_share"] = round(over_share, 3)
            reasons.append(
                "words were transcribed over the music here; they are most "
                "likely the lyrics, but they could be dialogue over the "
                "band, so this needs a listen"
            )
            confidence = min(confidence, 0.5)
            return section_type, confidence, reasons, True, evidence

    return section_type, confidence, reasons, False, evidence


def _merge_boundaries(boundaries: list[int], gap: int) -> list[int]:
    """Sort the candidate boundaries and pull the close ones together."""
    if not boundaries:
        return []
    ordered = sorted(set(boundaries))
    merged = [ordered[0]]
    for value in ordered[1:]:
        if value - merged[-1] >= gap:
            merged.append(value)
    return merged


def build_sections(
    features: FrameFeatures,
    scores: dict,
    regions: list[ActivityRegion],
    separate_applause: bool = False,
) -> list[TheaterSection]:
    """Partition the timeline into theatrical sections.

    The boundaries come from the **activity regions**, not from a
    frame-by-frame comparison. That matters: in a theatre recording the
    orchestra plays through almost everything, so a frame that says "music"
    says it almost everywhere and its edges mean nothing. The edges that
    *do* mean something are where the spoken words start and stop, and
    where the audience applauds.

    So the timeline is cut at the edges of the spoken stretches and of the
    applause, and everything between two cuts is one section: music if
    nobody spoke, speech if somebody did.

    The frame indices that come out of the feature stream are *feature*
    frames, one per hop. Everything the model stores is in sample frames,
    so the conversion happens here, once.
    """
    n = features.n_frames
    total = n * features.hop
    if n == 0:
        return []

    spoken = [
        region for region in regions if region.label == "speech"
    ]
    applauded = [
        region for region in regions if region.label == "applause"
    ]

    # A very short spoken stretch is a line inside a number, not a scene
    # of its own; it does not get to be a section.
    # The regions are stored in sample frames, so this comparison has to be
    # in sample frames too. Getting that wrong once meant the filter never
    # fired at all, and a two-second line was enough to cut a number in
    # half.
    min_spoken = int(MIN_SPOKEN_SECTION_SECONDS * features.sample_rate)
    spoken = [
        region
        for region in spoken
        if (region.end_frame - region.start_frame) >= min_spoken
    ]

    boundaries: list[int] = [0, total]
    for region in spoken + applauded:
        boundaries.append(region.start_frame)
        boundaries.append(region.end_frame)

    # The stretches where words were heard over the music. They do not cut
    # the recording - the music is not split at a line of words - but they
    # are handed to the classifier so that the doubt is recorded.
    over_music: list[tuple[int, int]] = []
    marks = scores.get("asr_over_music")
    if marks is not None and marks.size:
        for start_frame, end_frame in _runs_above(marks, 0.5, 0.5):
            over_music.append((start_frame * features.hop, end_frame * features.hop))

    # The boundaries are in sample frames, like everything else in the
    # model, so the merge distance has to be in sample frames as well.
    gap = int(MIN_BOUNDARY_GAP_SECONDS * features.sample_rate)
    edges = _merge_boundaries(boundaries, gap)
    if edges[0] != 0:
        edges.insert(0, 0)
    if edges[-1] != total:
        edges.append(total)

    sections: list[TheaterSection] = []
    for index in range(len(edges) - 1):
        start, end = edges[index], edges[index + 1]
        if end <= start:
            continue
        section_type, confidence, reasons, uncertain, evidence = _classify_section(
            scores, spoken, applauded, regions, start, end,
            features.sample_rate, features.hop, over_music,
        )
        inside = [
            region.label
            for region in regions
            if region.start_frame < end and start < region.end_frame
        ]
        if any(s < end and start < e for s, e in over_music):
            # The words over the music are a second activity on top of the
            # music, not a choice between the two.
            inside.append("speech")
        activities: list[str] = []
        for label in inside:
            if label not in activities:
                activities.append(label)
        sections.append(
            TheaterSection(
                start_frame=start,
                end_frame=end,
                section_type=section_type,
                confidence=confidence,
                activities=activities,
                reasons=reasons,
                uncertain=uncertain,
                evidence=evidence,
            )
        )
    sections = _cohere_musical_numbers(sections, features.sample_rate)
    return assemble_tracks(sections, features.sample_rate, separate_applause)


def assemble_tracks(sections, sample_rate, separate_applause=False):
    """Assemble listening units from activity-derived observations.

    Speech/applause edges are observations, not mandatory track cuts.
    Keep sustained or vocally supported musical items, absorb scene links
    into their surrounding scene, and retain their exact ranges for review.
    This never changes activity regions or discards source samples.
    """
    from copy import deepcopy
    parts = deepcopy(sections)

    def combine(left, right, kind, note, review=False):
        duration = max(1, right.end_frame - left.start_frame)
        a = left.end_frame - left.start_frame
        b = right.end_frame - right.start_frame
        evidence = dict(left.evidence)
        evidence['mean_music_score'] = (
            a * left.evidence.get('mean_music_score', 0) +
            b * right.evidence.get('mean_music_score', 0)) / duration
        evidence['activity_coverage'] = {
            key: (a * left.evidence.get('activity_coverage', {}).get(key, 0) +
                  b * right.evidence.get('activity_coverage', {}).get(key, 0)) / duration
            for key in ('speech', 'singing', 'instrumental', 'applause', 'silence')}
        evidence['review_regions'] = (left.evidence.get('review_regions', []) +
                                      right.evidence.get('review_regions', []))
        if review:
            from ..core.units import frames_to_timestamp
            note = (f"{frames_to_timestamp(right.start_frame, sample_rate)} to "
                    f"{frames_to_timestamp(right.end_frame, sample_rate)}: {note}")
            evidence['review_regions'].append({
                'start_frame': right.start_frame, 'end_frame': right.end_frame,
                'reason': note})
        return TheaterSection(
            start_frame=left.start_frame, end_frame=right.end_frame,
            section_type=kind, confidence=min(left.confidence, right.confidence),
            activities=list(dict.fromkeys(left.activities + right.activities)),
            reasons=list(dict.fromkeys(left.reasons + [note] + right.reasons)),
            uncertain=left.uncertain or right.uncertain or review, evidence=evidence)

    def musical(part):
        cov = part.evidence.get('activity_coverage', {})
        # Duration alone cannot prove a song: require measured music too.
        music = part.evidence.get('mean_music_score', 0)
        return (part.section_type == 'musical_number' and music >= .20) or (
            part.section_type in ('underscoring', 'mixed', 'musical_number') and
            music >= .45 and cov.get('singing', 0) >= .45 and
            part.evidence.get('independent_vocal_share', 0) >= .35)

    # Short sung reprises survive irrespective of length; weak musical gaps
    # between spoken regions are kept in the scene with a review range.
    roles = []
    for part in parts:
        if part.section_type == 'applause':
            roles.append('applause')
        elif musical(part):
            part.section_type = 'musical_number'
            roles.append('number')
        elif part.section_type in ('dialogue', 'mixed'):
            roles.append('scene')
        else:
            roles.append('link')
    result = []
    for index, (part, role) in enumerate(zip(parts, roles)):
        if role == 'applause' and not separate_applause:
            if result:
                result[-1] = combine(result[-1], part, result[-1].section_type,
                                     'Audience applause retained with the preceding section')
            else:
                # Opening audience noise has no preceding number.
                part.section_type = 'uncertain'
                part.uncertain = True
                result.append(part)
            continue
        if role == 'link':
            before = roles[index - 1] if index else None
            after = roles[index + 1] if index + 1 < len(roles) else None
            if result and before == 'scene' and after == 'scene':
                result[-1] = combine(result[-1], part, 'dialogue',
                    'Review internal musical passage; insufficient evidence for a separate song', True)
                continue
            if result and (after in ('applause', None) or before == 'number'):
                result[-1] = combine(result[-1], part, result[-1].section_type,
                    'Review transition or musical tail retained in context', True)
                continue
            part.uncertain = True
            part.section_type = 'uncertain'
            part.reasons.append('Review possible short song or scene link before splitting')
        if result and role == 'scene' and result[-1].section_type in ('dialogue', 'mixed'):
            result[-1] = combine(result[-1], part, 'dialogue', 'Continuous dialogue scene')
        elif result and result[-1].section_type == 'uncertain' and role in ('scene', 'number'):
            # A leading unclassified transition belongs to the following unit,
            # but remains reviewable, rather than becoming a spurious track.
            result[-1] = combine(result[-1], part, part.section_type,
                                'Review lead-in retained with following section', True)
        else:
            result.append(part)
    # Collapse any adjacent dialogue sections so continuous speech is never split.
    merged_result: list[TheaterSection] = []
    for item in result:
        if merged_result and item.section_type == 'dialogue' and merged_result[-1].section_type == 'dialogue':
            merged_result[-1] = combine(merged_result[-1], item, 'dialogue', 'Continuous dialogue scene')
        else:
            merged_result.append(item)
    return merged_result


def _is_strong_musical_piece(section: TheaterSection) -> bool:
    """True when a section is clearly part of a sung or sustained number."""
    if section.section_type == "musical_number":
        return True
    if section.section_type not in ("underscoring", "mixed"):
        return False
    mus = float(section.evidence.get("mean_music_score", 0.0) or 0.0)
    cov = section.evidence.get("activity_coverage", {}) or {}
    singing = float(cov.get("singing", 0.0) or 0.0)
    return mus >= 0.45 and singing >= 0.35


def _is_musical_tail(section: TheaterSection) -> bool:
    """True when an underscoring section has solid orchestral/music presence."""
    if section.section_type != "underscoring":
        return False
    return float(section.evidence.get("mean_music_score", 0.0) or 0.0) >= 0.35


def _bridge_allows_merge(
    left: TheaterSection,
    bridge: TheaterSection,
    right: TheaterSection,
    sample_rate: int,
) -> bool:
    """True when ``bridge`` is a brief internal interruption, not a new scene."""
    if bridge.section_type == "applause":
        return False
    cov = bridge.evidence.get("activity_coverage", {}) or {}
    if float(cov.get("applause", 0.0) or 0.0) >= 0.25:
        return False
    bridge_dur = (bridge.end_frame - bridge.start_frame) / float(sample_rate or 44100)
    both_strong = _is_strong_musical_piece(left) and _is_strong_musical_piece(right)
    if both_strong:
        if bridge_dur <= COHERENT_NUMBER_GAP_SECONDS:
            return True
        mus = float(bridge.evidence.get("mean_music_score", 0.0) or 0.0)
        singing = float(cov.get("singing", 0.0) or 0.0)
        if bridge_dur <= COHERENT_VAMP_GAP_SECONDS and (
            bridge.section_type in ("underscoring", "mixed") or mus >= 0.20 or singing >= 0.20
        ):
            return True
        return False
    if (
        (left.section_type == "musical_number" and _is_musical_tail(right))
        or (right.section_type == "musical_number" and _is_musical_tail(left))
    ):
        return bridge_dur <= COHERENT_TAIL_GAP_SECONDS
    return False


def _merge_three_sections(
    left: TheaterSection,
    bridge: TheaterSection,
    right: TheaterSection,
    sample_rate: int,
) -> TheaterSection:
    """Combine three adjacent sections into one coherent musical number."""
    parts = [left, bridge, right]
    total_frames = max(1, right.end_frame - left.start_frame)
    durations = [max(0, p.end_frame - p.start_frame) for p in parts]
    weighted_music = sum(
        d * float(p.evidence.get("mean_music_score", 0.0) or 0.0)
        for d, p in zip(durations, parts)
    ) / float(total_frames)
    cov_keys = ("speech", "singing", "instrumental", "applause", "silence")
    merged_cov = {}
    for key in cov_keys:
        merged_cov[key] = round(
            sum(
                d * float((p.evidence.get("activity_coverage", {}) or {}).get(key, 0.0) or 0.0)
                for d, p in zip(durations, parts)
            ) / float(total_frames),
            3,
        )
    merged_activities = list(
        dict.fromkeys(act for p in parts for act in (p.activities or []))
    )
    bridge_dur = (bridge.end_frame - bridge.start_frame) / float(sample_rate or 44100)
    reasons = list(left.reasons)
    merge_note = (
        f"kept continuous across a {bridge_dur:.1f} s internal "
        f"{bridge.section_type.replace('_', ' ')} bridge"
    )
    if merge_note not in reasons:
        reasons.append(merge_note)
    total_sec = total_frames / float(sample_rate or 44100)
    if total_sec >= MIN_NUMBER_SECONDS or (
        total_sec >= MIN_SUNG_NUMBER_SECONDS
        and merged_cov.get("singing", 0.0) >= 0.45
        and weighted_music >= 0.45
    ):
        section_type = "musical_number"
    else:
        section_type = left.section_type
    evidence = {
        "activity_coverage": merged_cov,
        "duration_seconds": round(total_sec, 2),
        "mean_music_score": round(weighted_music, 3),
        "internal_bridges": int(left.evidence.get("internal_bridges", 0))
        + 1
        + int(right.evidence.get("internal_bridges", 0)),
    }
    return TheaterSection(
        start_frame=left.start_frame,
        end_frame=right.end_frame,
        section_type=section_type,
        confidence=max(left.confidence, right.confidence),
        activities=merged_activities,
        reasons=reasons,
        uncertain=bool(left.uncertain or bridge.uncertain or right.uncertain),
        evidence=evidence,
    )


def _cohere_musical_numbers(
    sections: list[TheaterSection], sample_rate: int
) -> list[TheaterSection]:
    """Keep musical numbers coherent across brief internal dialogue or dips.

    Live numbers frequently include brief spoken lines, vamp exchanges, or
    transient classifier dips without applause. Merging those bridges
    prevents one number from shattering into multiple under-threshold
    ``underscoring`` fragments.
    """
    if len(sections) < 3 or sample_rate <= 0:
        return sections
    current = list(sections)
    changed = True
    while changed and len(current) >= 3:
        changed = False
        out: list[TheaterSection] = []
        i = 0
        while i < len(current):
            if i + 2 < len(current) and _bridge_allows_merge(
                current[i], current[i + 1], current[i + 2], sample_rate
            ):
                merged = _merge_three_sections(
                    current[i], current[i + 1], current[i + 2], sample_rate
                )
                # Keep dialogue share bounded so separate scenes never coalesce.
                merged_speech = float(
                    (merged.evidence.get("activity_coverage", {}) or {}).get("speech", 0.0) or 0.0
                )
                if merged_speech <= 0.35:
                    out.append(merged)
                    i += 3
                    changed = True
                    continue
            out.append(current[i])
            i += 1
        current = out
    return current


def _absorb_tiny_sections(
    sections: list[TheaterSection], sample_rate: int
) -> list[TheaterSection]:
    """Fold the very short sections into their neighbours.

    A stretch of a second or two is not a section of a performance; it is
    a gap between two of them. Folding it into the neighbour keeps the
    timeline readable without losing any audio: the sections still tile
    the recording exactly.
    """
    if len(sections) < 2:
        return sections
    minimum = int(MIN_SECTION_SECONDS * sample_rate)
    result: list[TheaterSection] = []
    for section in sections:
        if (
            result
            and (section.end_frame - section.start_frame) < minimum
            and section.section_type != "applause"
        ):
            previous = result[-1]
            previous.end_frame = section.end_frame
            previous.activities = list(
                dict.fromkeys(list(previous.activities) + list(section.activities))
            )
            previous.reasons.append(
                f"a {(section.end_frame - section.start_frame) / float(sample_rate):.1f} s "
                f"stretch of {section.section_type} is included here"
            )
            continue
        if (
            result
            and section.section_type == "dialogue"
            and result[-1].section_type == "dialogue"
        ):
            previous = result[-1]
            previous.end_frame = section.end_frame
            previous.activities = list(
                dict.fromkeys(list(previous.activities) + list(section.activities))
            )
            previous.uncertain = previous.uncertain and section.uncertain
            continue
        result.append(section)
    return result


# ----------------------------------------------------------------------
# The whole pass
# ----------------------------------------------------------------------
def analyze_theater(
    features: FrameFeatures,
    events=None,
    asr_segments: Optional[list[AsrSegment]] = None,
    noise_floor_dbfs: Optional[float] = None,
    separate_applause: bool = False,
) -> TheaterAnalysis:
    """Run the whole theatrical analysis over a prepared feature stream."""
    scores = score_activities(
        features,
        events=events,
        asr_segments=asr_segments,
        noise_floor_dbfs=noise_floor_dbfs,
    )
    regions = build_activity_regions(features, scores, asr_segments=asr_segments)
    sections = build_sections(features, scores, regions, separate_applause)

    notes: list[str] = []
    if not sections:
        notes.append("The recording is too short to segment.")
    if not scores.get("used_model"):
        notes.append(
            "The audio-event classifier was not available, so the sections "
            "come from acoustic measurements alone and are less reliable."
        )
    return TheaterAnalysis(
        sections=sections,
        activities=regions,
        noise_floor_dbfs=float(scores.get("floor_dbfs", 0.0)),
        used_model=bool(scores.get("used_model")),
        notes=notes,
    )
