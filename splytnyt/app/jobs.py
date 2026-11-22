"""Background jobs.

Long operations (importing and analysing a multi-hour recording, exporting)
must not freeze the interface. Each job runs on a worker thread, reports
progress through a callback and can be cancelled between blocks.

The analysis job is **chunked**: a six-hour recording is analysed in 30
second windows with an overlap, and the per-chunk results are merged. The
full mono signal is never materialised, so memory use is bounded by the
chunk size, not by the length of the recording.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

#: Analysis chunk length in seconds.
CHUNK_SECONDS = 30.0
#: Overlap between chunks, in seconds, so a boundary near a chunk edge is
#: still detected correctly.
CHUNK_OVERLAP_SECONDS = 1.0

ProgressCallback = Optional[Callable[[float, str], None]]


class Job:
    """Base class for a cancellable background job."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._cancel = threading.Event()
        self.error: str = ""

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled(self.name)

    def run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class JobCancelled(RuntimeError):
    """Raised inside a job when the user cancels it."""


# ----------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------
class ImportJob(Job):
    """Builds the waveform peak pyramids for the imported sources."""

    def __init__(self, project, cache_dir: Optional[str] = None,
                 progress: ProgressCallback = None) -> None:
        super().__init__("Building waveform")
        self.project = project
        self.cache_dir = cache_dir
        self.progress = progress
        self.pyramids: dict[str, object] = {}

    def run(self) -> None:
        from ..audio.decode import AudioSource
        from ..audio.peaks import PeakPyramid

        sources = self.project.sources
        for index, source in enumerate(sources):
            self.check_cancelled()
            base = 0.1 + 0.85 * (index / max(1, len(sources)))
            if self.progress:
                self.progress(base, f"Analysing {os.path.basename(source.path)}")

            cache_path = None
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_path = os.path.join(self.cache_dir, f"{source.source_id}.peaks")

            pyramid = None
            if cache_path:
                pyramid = PeakPyramid.load(cache_path)
            if pyramid is None:
                with AudioSource(source.path) as audio:
                    pyramid = PeakPyramid.build(
                        audio,
                        progress=None,
                        cancel=lambda: self.cancelled,
                    )
                if cache_path:
                    try:
                        pyramid.save(cache_path)
                    except OSError:
                        pass
            self.pyramids[source.source_id] = pyramid
        if self.progress:
            self.progress(1.0, "Waveform ready")
        # The job hands itself back, so the caller gets the pyramids it
        # built. Without this the interface would finish with an empty
        # waveform and no hint that anything had gone wrong.
        return self


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
@dataclass
class AnalysisOutcome:
    """What the analysis job produced."""

    suggestions: list = field(default_factory=list)
    #: (start_frame, end_frame, section_type, confidence, reasons)
    classifications: list = field(default_factory=list)
    classified: int = 0
    noise_floor_dbfs: float = 0.0
    threshold_dbfs: float = 0.0
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


class AnalysisJob(Job):
    """Detects candidate boundaries and classifies the resulting sections.

    The work happens in two passes:

    1. **Find the transitions.** The recording is walked in 30 second
       windows with an overlap, and the quiet stretches and the musical
       changes are found. Nothing is decided here: a quiet stretch is only
       a place where a boundary *might* be.
    2. **Place the cuts.** Each transition region is then read again and
       the cut inside it is worked out, so the next song starts at its
       music rather than in the middle of the quiet in front of it.

    The job never mutates the project: it returns the suggestions and the
    classifications so the caller can apply them together as a single
    undoable step on the main thread.
    """

    def __init__(self, project, sensitivity: str = "balanced",
                 classify: bool = True, progress: ProgressCallback = None,
                 placement=None) -> None:
        super().__init__("Analysing recording")
        self.project = project
        self.sensitivity = sensitivity
        self.classify = classify
        self.progress = progress
        self.placement = placement
        self.outcome = AnalysisOutcome()

    # ------------------------------------------------------------------
    def run(self) -> AnalysisOutcome:
        from ..analysis.classify import classify_section
        from ..analysis.novelty import (
            find_novelty_peaks,
            novelty_curve_from_shapes,
            persistence_window_seconds,
            spectral_shapes,
        )
        from ..analysis.placement import place_cut
        from ..analysis.segment import (
            filter_redundant_novelties,
            get_preset,
            merge_suggestions,
            novelty_suggestion,
        )
        from ..analysis.silence import analyse_noise_floor, find_silence_regions
        from ..audio.timeline import build_reader

        started = time.time()
        sample_rate = self.project.sample_rate() or 44100
        preset = get_preset(self.sensitivity)

        reader = build_reader(self.project)
        try:
            hop = 512
            chunk_frames = int(CHUNK_SECONDS * sample_rate)
            overlap_frames = int(CHUNK_OVERLAP_SECONDS * sample_rate)

            total = self.project.total_frames()
            regions = []
            novelty_items = []
            noise_floor = -70.0
            threshold = -55.0
            position = 0
            first_chunk = True

            while position < total:
                self.check_cancelled()
                end = min(total, position + chunk_frames)
                signal = reader.read(position, end - position)
                if signal.size:
                    mono = signal.mean(axis=1).astype(np.float32, copy=False)
                else:
                    mono = np.zeros(0, dtype=np.float32)

                if first_chunk:
                    profile = analyse_noise_floor(mono, sample_rate)
                    noise_floor = profile.noise_floor_dbfs
                    threshold = profile.threshold_dbfs
                    first_chunk = False
                else:
                    # Re-estimate from this chunk as well. The threshold
                    # keeps the stricter (higher) value, so quiet passages
                    # in a noisy source are not over-detected. The noise
                    # floor keeps the *lower* value: it is the quietest
                    # material in the recording, and a chunk that happens
                    # to contain nothing but loud music says nothing about
                    # it.
                    profile = analyse_noise_floor(mono, sample_rate)
                    noise_floor = min(noise_floor, profile.noise_floor_dbfs)
                    threshold = max(threshold, profile.threshold_dbfs)

                local = find_silence_regions(
                    mono,
                    sample_rate,
                    threshold_dbfs=threshold,
                    min_silence_seconds=preset.min_silence_seconds,
                )
                for region in local:
                    regions.append(
                        type(region)(
                            start_frame=region.start_frame + position,
                            end_frame=region.end_frame + position,
                            level_dbfs=region.level_dbfs,
                        )
                    )

                # Musical-change detection on this chunk (with overlap so a
                # change on the chunk edge is not lost).
                read_start = max(0, position - overlap_frames)
                read_end = min(total, end + overlap_frames)
                context = reader.read(read_start, read_end - read_start)
                if context.size:
                    context_mono = context.mean(axis=1).astype(np.float32, copy=False)
                    shapes = spectral_shapes(context_mono, sample_rate)
                    curve = novelty_curve_from_shapes(shapes)
                    peaks = find_novelty_peaks(
                        curve, hop, sample_rate, self.sensitivity, shapes=shapes
                    )
                    context_start = read_start
                    duration = total / float(sample_rate)
                    edge = max(
                        1,
                        int(
                            min(persistence_window_seconds(), duration * 0.1)
                            * sample_rate
                        ),
                    )
                    for frame, strength in peaks:
                        absolute = context_start + frame
                        if absolute <= position or absolute >= end:
                            continue
                        # The start and the end of the recording are not cuts.
                        if absolute <= edge or absolute >= total - edge:
                            continue
                        # A change peak at the edge of a detected silence is
                        # the same event; the silence wins.
                        margin = max(1, int(0.5 * sample_rate))
                        if any(
                            region.start_frame + position - margin
                            <= absolute
                            <= region.end_frame + position + margin
                            for region in local
                        ):
                            continue
                        novelty_items.append(novelty_suggestion(absolute, strength, preset))

                position = end
                if self.progress:
                    self.progress(
                        0.05 + 0.5 * (position / max(1, total)),
                        f"Analysing {position / max(1, sample_rate) / 60:.1f} min",
                    )

            regions.sort(key=lambda region: region.start_frame)
            suggestions = self._place_boundaries(
                reader, regions, sample_rate, noise_floor, place_cut, preset, total
            )
            # A musical change that a quiet stretch already accounts for is
            # the same event twice.
            novelty_items = filter_redundant_novelties(novelty_items, regions, sample_rate)
            suggestions.extend(novelty_items)

            merged = merge_suggestions(suggestions, sample_rate, preset)
            self.outcome.suggestions = merged
            self.outcome.noise_floor_dbfs = noise_floor
            self.outcome.threshold_dbfs = threshold
            self.outcome.elapsed_seconds = time.time() - started

            if self.classify:
                events = self._classify_events(reader, sample_rate, total)
                self._classify_segments(
                    reader, classify_section, sample_rate, merged, events
                )
        finally:
            reader.close()

        if self.progress:
            self.progress(1.0, "Analysis complete")
        return self.outcome

    # ------------------------------------------------------------------
    def _place_boundaries(
        self, reader, regions, sample_rate, noise_floor, place_cut, preset, total
    ) -> list:
        """Second pass: work out where the cut goes inside each region."""
        from ..analysis.onset import SEARCH_LIMIT_SECONDS
        from ..analysis.placement import PlacementChoice
        from ..analysis.segment import silence_suggestion

        choice = self.placement or PlacementChoice()
        context_before = int(3.0 * sample_rate)
        search_after = int(SEARCH_LIMIT_SECONDS * sample_rate)
        suggestions = []

        for index, region in enumerate(regions):
            self.check_cancelled()
            window_start = max(0, region.start_frame - context_before)
            window_end = min(total, region.end_frame + search_after)
            if window_end <= window_start:
                continue
            data = reader.read(window_start, window_end - window_start)
            if not data.size:
                continue
            mono = data.mean(axis=1).astype(np.float32, copy=False)

            next_start = None
            if index + 1 < len(regions):
                next_start = regions[index + 1].start_frame

            result = place_cut(
                mono,
                sample_rate,
                region,
                choice=choice,
                floor_dbfs=noise_floor,
                search_end=next_start,
                base_frame=window_start,
            )
            if result.rejected:
                continue
            suggestions.append(silence_suggestion(region, self._profile(noise_floor), preset, result))
            if self.progress:
                self.progress(
                    0.55 + 0.2 * ((index + 1) / max(1, len(regions))),
                    f"Placing cut {index + 1}/{len(regions)}",
                )
        return suggestions

    @staticmethod
    def _profile(noise_floor: float):
        """A minimal stand-in for the noise profile used by the suggestions."""
        from ..analysis.silence import NoiseProfile

        return NoiseProfile(
            noise_floor_dbfs=noise_floor,
            threshold_dbfs=noise_floor + 18.0,
            method="adaptive",
            sample_rate=44100,
        )

    # ------------------------------------------------------------------
    def _classify_events(self, reader, sample_rate: int, total: int):
        """Run the audio-event classifier over the whole recording.

        This is the same optional classifier the theatre mode uses. When it
        is installed it is much better than the heuristic at telling music
        from speech from applause - measured on a real stage recording, the
        heuristic's spectral test called *everything* music, while the
        trained model separated the three cleanly - and it costs about two
        seconds per hour of audio. When it is missing the heuristic is used
        and the reasons say so.
        """
        from ..analysis import aec

        if not aec.onnxruntime_available() or not aec.model_installed():
            return None
        try:
            def _on_progress(fraction: float, _pos: int) -> None:
                if self.progress:
                    self.progress(
                        0.6 + 0.15 * fraction,
                        "Classifying the audio events",
                    )

            def _check_cancel() -> bool:
                return self.cancelled

            events = aec.classify_reader(
                reader,
                sample_rate,
                total,
                progress=_on_progress,
                cancel=_check_cancel,
            )
            self.check_cancelled()
            return events if events.n_frames() > 0 else None
        except JobCancelled:
            raise
        except Exception:
            # A classifier that will not run must not stop the analysis.
            return None

    # ------------------------------------------------------------------
    def _classify_segments(
        self, reader, classify_section, sample_rate: int, suggestions, events=None
    ) -> None:
        """Classify the segments the suggestions imply.

        A segment often contains more than one kind of audio (a song plus
        the applause at its end), so several windows spread across it are
        classified and the most confident verdict wins. The job never
        mutates the project: it returns the classification so the caller
        can apply suggestions and classifications together as a single
        undoable step on the main thread.
        """
        total = self.project.total_frames()
        edges = [0] + [int(item.frame) for item in suggestions] + [total]
        edges = sorted(set(edge for edge in edges if 0 <= edge <= total))
        window_frames = int(8 * sample_rate)

        for index in range(len(edges) - 1):
            self.check_cancelled()
            start, end = edges[index], edges[index + 1]
            if end - start <= 0:
                continue

            span = end - start
            length = min(span, window_frames)
            if length <= 0:
                continue

            # Up to three windows spread across the segment.
            if span <= window_frames:
                positions = [start + max(0, (span - length) // 2)]
            else:
                positions = [
                    start + int(span * fraction) - length // 2
                    for fraction in (0.2, 0.5, 0.8)
                ]
                positions = [max(start, min(end - length, position)) for position in positions]

            best = None
            for position in positions:
                data = reader.read(position, length)
                if data.size == 0:
                    continue
                mono = data.mean(axis=1).astype(np.float32, copy=False)
                guess = classify_section(
                    mono,
                    sample_rate,
                    events=self._slice_events(events, position, length, sample_rate),
                )
                if best is None or self._better(guess, best):
                    best = guess

            if best is None:
                continue
            self.outcome.classifications.append(
                (start, end, best.section_type, best.confidence, list(best.reasons))
            )
            self.outcome.classified += 1
            if self.progress:
                self.progress(
                    0.75 + 0.2 * ((index + 1) / max(1, len(edges) - 1)),
                    f"Classifying section {index + 1}/{len(edges) - 1}",
                )

    @staticmethod
    def _slice_events(events, position: int, length: int, sample_rate: int):
        """The classifier's scores for one window of the recording."""
        if events is None or events.n_frames() == 0:
            return None
        from ..analysis import aec

        seconds_per_frame = events.frame_seconds
        first = int(position / float(sample_rate) / seconds_per_frame)
        last = int((position + length) / float(sample_rate) / seconds_per_frame)
        first = max(0, min(events.n_frames(), first))
        last = max(first, min(events.n_frames(), last))
        if last <= first:
            return None
        return aec.EventScores(
            music=events.music[first:last],
            speech=events.speech[first:last],
            singing=events.singing[first:last],
            applause=events.applause[first:last],
            silence=events.silence[first:last],
            frame_seconds=events.frame_seconds,
            from_model=True,
            source=events.source,
        )

    @staticmethod
    def _better(candidate, current) -> bool:
        """Prefer a decisive verdict over an inconclusive one."""
        if current is None:
            return True
        if candidate.section_type == "unknown":
            return False
        if current.section_type == "unknown":
            return True
        return candidate.confidence > current.confidence


def _describe(item: dict, sample_rate: int) -> str:
    """Human-readable evidence for a candidate boundary (legacy helper)."""
    if item.get("kind") == "silence":
        return f"silence ({item.get('duration', 0.0):.1f} s below the noise floor)"
    return "musical change without silence - uncertain"


# ----------------------------------------------------------------------
# Theatrical analysis
# ----------------------------------------------------------------------
def _stable_hash(payload: dict) -> str:
    """Return a deterministic SHA-256 hex digest for JSON-compatible metadata."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sources_fingerprint(project) -> str:
    """Fingerprint the ordered timeline sources and their geometry."""
    items = []
    for src in getattr(project, "sources", []) or []:
        path = getattr(src, "path", "") or ""
        norm_path = os.path.normcase(os.path.abspath(path)) if path else ""
        items.append(
            {
                "path": norm_path,
                "frames": int(getattr(src, "frames", 0) or 0),
                "sample_rate": int(getattr(src, "sample_rate", 0) or 0),
                "channels": int(getattr(src, "channels", 0) or 0),
                "size_bytes": int(getattr(src, "size_bytes", 0) or 0),
                "mtime_ns": int(getattr(src, "mtime_ns", 0) or 0),
                "gap_frames": int(getattr(src, "gap_frames", 0) or 0),
                "head_hash": getattr(src, "head_hash", "") or "",
                "tail_hash": getattr(src, "tail_hash", "") or "",
            }
        )
    return _stable_hash({"sources": items})


def aec_settings_fingerprint(project, aec_module) -> tuple[str, dict]:
    """Build the fingerprint and descriptor for cached audio-event scores."""
    model_size = 0
    try:
        model_path = os.path.join(aec_module.model_folder(), aec_module.MODEL_FILE)
        if os.path.exists(model_path):
            model_size = int(os.path.getsize(model_path))
    except OSError:
        model_size = 0
    settings = {
        "sources": sources_fingerprint(project),
        "engine": "yamnet",
        "frame_seconds": float(getattr(aec_module, "FRAME_SECONDS", 0.48)),
        "block_frames": int(getattr(aec_module, "BLOCK_FRAMES", 125)),
        "padding_frames": int(getattr(aec_module, "BLOCK_PADDING_FRAMES", 4)),
        "model_size": model_size,
    }
    return _stable_hash(settings), settings


def transcript_settings_fingerprint(
    project, model: str, language: Optional[str], transcriber=None
) -> tuple[str, dict]:
    """Build the fingerprint and descriptor for cached transcripts."""
    if transcriber is not None and hasattr(transcriber, "settings_fingerprint"):
        tr_settings = dict(transcriber.settings_fingerprint(language))
    else:
        tr_settings = {
            "model": model,
            "language": language or "auto",
            "beam_size": 1,
            "temperature": 0.0,
        }
    settings = {
        "sources": sources_fingerprint(project),
        "transcriber": tr_settings,
    }
    return _stable_hash(settings), settings


def _theater_cache_dir() -> str:
    override = os.environ.get("SPLYTNYT_CACHE_DIR", "").strip()
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".splytnyt", "cache", "theater")


def _read_disk_cache(name: str) -> Optional[dict]:
    path = os.path.join(_theater_cache_dir(), name)
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_disk_cache(name: str, payload: dict) -> None:
    folder = _theater_cache_dir()
    try:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass


def _encode_events_cache(fingerprint: str, settings: dict, events) -> dict:
    stacked = np.stack(
        [
            np.asarray(events.music, dtype=np.float32),
            np.asarray(events.speech, dtype=np.float32),
            np.asarray(events.singing, dtype=np.float32),
            np.asarray(events.applause, dtype=np.float32),
            np.asarray(events.silence, dtype=np.float32),
        ],
        axis=0,
    )
    raw = zlib.compress(stacked.tobytes(), level=6)
    return {
        "fingerprint": fingerprint,
        "settings": settings,
        "n_frames": int(events.n_frames()),
        "frame_seconds": float(events.frame_seconds),
        "from_model": bool(events.from_model),
        "source": str(events.source or "yamnet"),
        "arrays_zlib_b64": base64.b64encode(raw).decode("ascii"),
    }


def _decode_events_cache(payload: object, expected_fp: str, aec_module):
    if not isinstance(payload, dict):
        return None
    if payload.get("fingerprint") != expected_fp:
        return None
    blob = payload.get("arrays_zlib_b64")
    n_frames = int(payload.get("n_frames", 0) or 0)
    if not blob or n_frames <= 0:
        return None
    try:
        raw = zlib.decompress(base64.b64decode(blob.encode("ascii")))
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.size != 5 * n_frames:
            return None
        stacked = arr.reshape((5, n_frames)).copy()
        return aec_module.EventScores(
            music=stacked[0],
            speech=stacked[1],
            singing=stacked[2],
            applause=stacked[3],
            silence=stacked[4],
            frame_seconds=float(payload.get("frame_seconds", aec_module.FRAME_SECONDS)),
            from_model=bool(payload.get("from_model", True)),
            source=str(payload.get("source", "yamnet")),
        )
    except Exception:
        return None


@dataclass
class TheaterOutcome:
    """What the theatrical analysis produced."""

    #: The sections, in order.
    sections: list = field(default_factory=list)
    #: The overlapping activity regions.
    activities: list = field(default_factory=list)
    #: Boundary proposals, in the same shape the ordinary analysis uses.
    suggestions: list = field(default_factory=list)
    #: (start_frame, end_frame, section_type, confidence, reasons)
    classifications: list = field(default_factory=list)
    #: What the transcriber found, for the review panel.
    transcript: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    #: True when the trained audio-event classifier was used.
    used_model: bool = False
    #: True when transcription ran and produced something.
    transcribed: bool = False
    transcription_error: str = ""
    #: How many stretches the transcriber produced and how many survived.
    transcript_kept: int = 0
    transcript_dropped: int = 0
    language: str = ""
    grouping_decisions: list = field(default_factory=list)
    reference_matches: list = field(default_factory=list)
    aec_from_cache: bool = False
    transcript_from_cache: bool = False
    aec_cache_entry: Optional[dict] = None
    transcript_cache_entry: Optional[dict] = None


class TheaterJob(Job):
    """Segments a recording into theatrical sections.

    The work happens in three passes, and each pass can be switched off:

    1. **Classify.** The recording is read in blocks and handed to the
       audio-event classifier, which says where there is music, where
       there is speech, where there is applause and where there is
       silence. The blocks are read one at a time, so the memory use
       depends on the block and not on the length of the recording.
    2. **Transcribe** (optional). The whole recording is transcribed
       locally. The result is *evidence*: it tells the classifier where
       the spoken words are and it is shown to the reviewer, but a line
       the model made up is thrown away rather than believed.
    3. **Assemble.** The scores become activity regions and then into
       sections: a musical number, dialogue, underscoring, applause,
       mixed speech and music, or an uncertain stretch.

    The job never mutates the project: it returns everything so the
    caller can apply it as a single undoable step on the main thread.
    """

    #: How much audio is read at a time, in seconds.
    BLOCK_SECONDS = 60.0

    #: The overlap between blocks, in seconds. It keeps the classifier
    #: from losing an event that straddles a block edge.
    BLOCK_OVERLAP_SECONDS = 2.0

    def __init__(
        self,
        project,
        use_classifier: bool = True,
        use_transcriber: bool = False,
        transcript_model: str = "small",
        transcript_language: Optional[str] = None,
        transcript_threads: int = 0,
        reanalyze: bool = False,
        separate_applause: bool = False,
        progress: ProgressCallback = None,
    ) -> None:
        super().__init__("Analysing the recording")
        self.project = project
        self.use_classifier = use_classifier
        self.use_transcriber = use_transcriber
        self.transcript_model = transcript_model
        self.transcript_language = transcript_language
        #: How many threads the transcriber may use. Zero means "as many as
        #: the machine reports", which is usually right but not always.
        self.transcript_threads = transcript_threads
        self.reanalyze = bool(reanalyze)
        self.separate_applause = separate_applause
        self.progress = progress
        self.outcome = TheaterOutcome()

    # ------------------------------------------------------------------
    def run(self) -> TheaterOutcome:
        import time

        from ..analysis import aec
        from ..analysis.activities import compute_frame_features
        from ..analysis.theater import analyze_theater
        from ..audio.timeline import build_reader

        started = time.time()
        sample_rate = self.project.sample_rate() or 44100
        total = self.project.total_frames()

        reader = build_reader(self.project)
        try:
            events = None
            notes: list[str] = []

            if self.use_classifier:
                events, notes = self._classify(reader, sample_rate, total, aec)
                self.outcome.used_model = events is not None

            # The acoustic measurements are always made: they are cheap
            # when the classifier is doing the work, and they are the only
            # evidence when it is not there.
            need_pitch = events is None
            need_harmonic = events is None
            if self.progress:
                self.progress(0.35, "Measuring the audio")

            # The recording is read in blocks, so the whole act is never
            # held in memory. The noise floor is measured on a sample
            # spread across the recording rather than on all of it.
            profile = self._noise_profile(reader, sample_rate, total)

            def fetch(start: int, count: int) -> np.ndarray:
                data = reader.read(start, count)
                if not data.size:
                    return np.zeros(0, dtype=np.float32)
                return data.mean(axis=1).astype(np.float32)

            features = compute_frame_features(
                sample_rate=sample_rate,
                source=fetch,
                total_samples=total,
                cancel=lambda: self.cancelled,
                need_pitch=need_pitch,
                need_harmonic=need_harmonic,
            )

            asr_segments = None
            if self.use_transcriber:
                asr_segments, transcript_notes = self._transcribe(
                    reader, sample_rate, total
                )
                notes.extend(transcript_notes)

            if self.progress:
                self.progress(0.75, "Assembling the sections")

            analysis = analyze_theater(
                features,
                events=events,
                asr_segments=asr_segments,
                noise_floor_dbfs=profile.noise_floor_dbfs if profile else None,
                separate_applause=self.separate_applause,
            )

            from ..analysis.musical import measure_context, group_numbers
            from ..analysis.reference_audio import match_references
            if self.progress:
                self.progress(.82, "Comparing musical phrases across spoken passages")
            context = measure_context(reader, lambda: self.cancelled)
            entries = self.project.cache.get("reference_audio_inputs", [])
            anchors, reports = match_references(entries, context, sample_rate,
                                                 lambda: self.cancelled)
            self.outcome.reference_matches = reports
            analysis.sections, self.outcome.grouping_decisions = group_numbers(
                analysis.sections, context, sample_rate, self.outcome.transcript, anchors, events)
            self.outcome.sections = analysis.sections
            self.outcome.activities = analysis.activities
            self.outcome.notes.extend(notes)
            self.outcome.notes.extend(analysis.notes)

            self._build_suggestions(analysis.sections, sample_rate, total)
            self.outcome.elapsed_seconds = time.time() - started
        finally:
            reader.close()

        if self.progress:
            self.progress(1.0, "Theatrical analysis complete")
        return self.outcome

    # ------------------------------------------------------------------
    def _noise_profile(self, reader, sample_rate: int, total: int):
        """Estimate the noise floor from a sample spread across the recording.

        Measuring the floor over the whole recording would mean holding the
        whole recording in memory. The room tone does not change over the
        course of a show, so a few minutes taken from the start, the middle
        and the end describe it just as well.
        """
        from ..analysis.silence import analyse_noise_floor

        if total <= 0:
            return None
        span = min(int(180.0 * sample_rate), total)
        if span <= 0:
            return None
        offsets = [0, max(0, total // 2 - span // 2), max(0, total - span)]
        pieces = []
        for offset in offsets:
            data = reader.read(offset, span)
            if data.size:
                pieces.append(data.mean(axis=1).astype(np.float32))
        if not pieces:
            return None
        sample = np.concatenate(pieces)
        return analyse_noise_floor(sample, sample_rate)

    # ------------------------------------------------------------------
    def _classify(self, reader, sample_rate: int, total: int, aec):
        """Run or reuse the cached audio-event classifier over the recording."""
        notes: list[str] = []
        aec_fp, aec_settings = aec_settings_fingerprint(self.project, aec)
        cache_name = f"aec_{aec_fp}.json.gz"
        if not self.reanalyze:
            for candidate in (
                (getattr(self.project, "cache", {}) or {}).get("theater_aec"),
                _read_disk_cache(cache_name),
            ):
                cached = _decode_events_cache(candidate, aec_fp, aec)
                if cached is not None:
                    self.outcome.aec_from_cache = True
                    self.outcome.aec_cache_entry = candidate
                    notes.append("Reused cached audio-event classification.")
                    return cached, notes

        if not aec.onnxruntime_available():
            notes.append(
                "The audio-event classifier needs the optional 'onnxruntime' "
                "package, so the sections come from acoustic measurements alone. "
                + aec.install_hint()
            )
            return None, notes
        if not aec.model_installed():
            notes.append(
                "The audio-event classifier has not been downloaded, so the "
                "sections come from acoustic measurements alone."
            )
            return None, notes

        def _on_progress(fraction: float, src_end: int) -> None:
            if self.progress:
                self.progress(
                    0.05 + 0.30 * fraction,
                    f"Classifying audio ({src_end / max(1, sample_rate) / 60:.1f} min)",
                )

        def _check_cancel() -> bool:
            return self.cancelled

        try:
            events = aec.classify_reader(
                reader,
                sample_rate,
                total,
                progress=_on_progress,
                cancel=_check_cancel,
            )
        except aec.ModelDownloadError as exc:
            notes.append(str(exc))
            return None, notes
        self.check_cancelled()
        if events.n_frames() == 0:
            return None, notes
        entry = _encode_events_cache(aec_fp, aec_settings, events)
        self.outcome.aec_cache_entry = entry
        _write_disk_cache(cache_name, entry)
        return events, notes

    # ------------------------------------------------------------------
    def _transcribe(self, reader, sample_rate: int, total: int):
        """Transcribe the recording locally (or reuse cached transcript)."""
        from ..analysis.theater import AsrSegment
        from ..analysis.transcribe import (
            WHISPER_SAMPLE_RATE,
            Transcriber,
            resample_stream_to_whisper_rate,
        )

        notes: list[str] = []
        transcriber = Transcriber(self.transcript_model, cpu_threads=self.transcript_threads)
        tr_fp, tr_settings = transcript_settings_fingerprint(
            self.project, self.transcript_model, self.transcript_language, transcriber
        )
        cache_name = f"tr_{tr_fp}.json.gz"
        if not self.reanalyze:
            for candidate in (
                (getattr(self.project, "cache", {}) or {}).get("theater_transcript"),
                _read_disk_cache(cache_name),
            ):
                if (
                    isinstance(candidate, dict)
                    and candidate.get("fingerprint") == tr_fp
                    and isinstance(candidate.get("segments"), list)
                ):
                    raw_segs = candidate["segments"]
                    self.outcome.transcript = [dict(s) for s in raw_segs]
                    self.outcome.language = str(candidate.get("language", "") or "")
                    self.outcome.transcribed = bool(raw_segs)
                    self.outcome.transcript_kept = sum(
                        1 for s in raw_segs if s.get("accepted", True)
                    )
                    self.outcome.transcript_dropped = (
                        len(raw_segs) - self.outcome.transcript_kept
                    )
                    self.outcome.transcript_from_cache = True
                    self.outcome.transcript_cache_entry = candidate
                    notes.append("Reused cached transcript.")
                    segments = [
                        AsrSegment(
                            start_seconds=float(s.get("start", 0.0)),
                            end_seconds=float(s.get("end", 0.0)),
                            text=str(s.get("text", "")),
                            accepted=bool(s.get("accepted", True)),
                            reason=str(s.get("reason", "")),
                        )
                        for s in raw_segs
                    ]
                    return segments, notes

        if not transcriber.available or not transcriber.load():
            self.outcome.transcription_error = transcriber.error
            notes.append(
                "Transcription was not available, so the sections come from the "
                "audio-event classifier alone. " + transcriber.error
            )
            return None, notes

        # The recording is read in blocks and resampled on the way, so the
        # full-rate signal is never in memory. Only the 16 kHz signal that
        # the model actually needs is kept.
        def fetch(start: int, count: int) -> np.ndarray:
            data = reader.read(start, count)
            if not data.size:
                return np.zeros(0, dtype=np.float32)
            return data.mean(axis=1).astype(np.float32)

        prepared = resample_stream_to_whisper_rate(fetch, total, sample_rate)
        if not prepared.size:
            notes.append("The recording could not be read for transcription.")
            return None, notes

        def report(fraction: float, message: str) -> None:
            if self.progress:
                self.progress(0.40 + 0.33 * fraction, message)

        result = transcriber.transcribe(
            prepared,
            WHISPER_SAMPLE_RATE,
            language=self.transcript_language,
            progress=report,
            cancel=lambda: self.cancelled,
        )
        del prepared

        self.outcome.transcribed = result.available and bool(result.segments)
        self.outcome.transcription_error = result.error
        self.outcome.transcript_kept = len(result.accepted)
        self.outcome.transcript_dropped = len(result.rejected)
        self.outcome.language = result.language
        self.outcome.transcript = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "accepted": segment.accepted,
                "reason": segment.reject_reason,
            }
            for segment in result.segments
        ]
        if result.available and not result.error and not self.cancelled:
            entry = {
                "fingerprint": tr_fp,
                "settings": tr_settings,
                "language": result.language,
                "language_probability": float(result.language_probability),
                "segments": list(self.outcome.transcript),
            }
            self.outcome.transcript_cache_entry = entry
            _write_disk_cache(cache_name, entry)
        notes.extend(result.notes)
        if result.error:
            notes.append(f"Transcription did not finish: {result.error}")

        segments = [
            AsrSegment(
                start_seconds=segment.start,
                end_seconds=segment.end,
                text=segment.text,
                accepted=segment.accepted,
                reason=segment.reject_reason,
            )
            for segment in result.segments
        ]
        return segments, notes

    # ------------------------------------------------------------------
    def _build_suggestions(self, sections, sample_rate: int, total: int) -> None:
        """Turn the section edges into boundary proposals."""
        from ..analysis.segment import Suggestion

        suggestions = []
        classifications = []
        for index, section in enumerate(sections):
            classifications.append(
                (
                    section.start_frame,
                    section.end_frame,
                    section.section_type,
                    section.confidence,
                    list(section.reasons),
                )
            )
            if index == 0:
                continue
            frame = section.start_frame
            # The cut sits just inside the new section, so the first
            # moment of the new material is not clipped.
            pre_roll = int(0.030 * sample_rate)
            cut = max(0, frame - pre_roll)
            uncertain = section.uncertain
            reason = (
                f"the audio changes character here: {section.section_type}"
            )
            suggestions.append(
                Suggestion(
                    frame=cut,
                    confidence=section.confidence,
                    reason=reason,
                    kind="theater",
                    uncertain=uncertain,
                    details={
                        "section_type": section.section_type,
                        "activities": list(section.activities),
                        "evidence": dict(section.evidence),
                        "applied_pre_roll_ms": 30,
                    },
                )
            )
        self.outcome.suggestions = suggestions
        self.outcome.classifications = classifications


# ----------------------------------------------------------------------
# Refinement
# ----------------------------------------------------------------------
@dataclass
class RefinementProposal:
    """Where one existing boundary would move to."""

    track_id: str
    old_frame: int
    new_frame: int
    reason: str
    uncertain: bool = False
    onset_frame: int = -1
    #: The margin that would be applied between the new cut and the onset.
    pre_roll_seconds: float = -1.0


class RefineJob(Job):
    """Works out where the existing boundaries would move.

    Nothing is changed here: the job returns proposals so the interface can
    show them before the user accepts them. A locked track is never
    included, and a project that is merely opened is never touched.
    """

    def __init__(self, project, placement=None, progress: ProgressCallback = None) -> None:
        super().__init__("Refining boundaries")
        self.project = project
        self.placement = placement
        self.progress = progress
        self.proposals: list[RefinementProposal] = []

    def run(self) -> list[RefinementProposal]:
        from ..analysis.onset import SEARCH_LIMIT_SECONDS
        from ..analysis.placement import PlacementChoice, place_cut
        from ..analysis.silence import frame_levels
        from ..audio.timeline import build_reader

        choice = self.placement or PlacementChoice()
        sample_rate = self.project.sample_rate() or 44100
        total = self.project.total_frames()
        context_before = int(3.0 * sample_rate)
        search_after = int(SEARCH_LIMIT_SECONDS * sample_rate)

        reader = build_reader(self.project)
        try:
            candidates = [
                track for track in self.project.sorted_tracks()
                if track.transition_start >= 0 and track.transition_end > track.transition_start
            ]
            for index, track in enumerate(candidates):
                self.check_cancelled()
                if track.locked:
                    continue

                window_start = max(0, track.transition_start - context_before)
                window_end = min(total, track.transition_end + search_after)
                data = reader.read(window_start, window_end - window_start)
                if not data.size:
                    continue
                mono = data.mean(axis=1).astype(np.float32, copy=False)

                region = type("Region", (), {})()
                region.start_frame = track.transition_start
                region.end_frame = track.transition_end
                region.level_dbfs = float(
                    np.min(frame_levels(mono, frame_frames=2048, hop_frames=512))
                )

                result = place_cut(
                    mono,
                    sample_rate,
                    region,
                    choice=choice,
                    search_end=None,
                    base_frame=window_start,
                )
                if result.rejected or result.frame == track.start_frame:
                    continue
                self.proposals.append(
                    RefinementProposal(
                        track_id=track.track_id,
                        old_frame=track.start_frame,
                        new_frame=result.frame,
                        reason=result.reason,
                        uncertain=result.uncertain,
                        onset_frame=result.onset_frame,
                        pre_roll_seconds=float(
                            result.details.get("applied_pre_roll_ms", -1)
                        ) / 1000.0,
                    )
                )
                if self.progress:
                    self.progress(
                        (index + 1) / max(1, len(candidates)),
                        f"Refining boundary {index + 1}/{len(candidates)}",
                    )
        finally:
            reader.close()
        return self.proposals


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------
class ExportJob(Job):
    """Runs the export and reports progress."""

    def __init__(self, project, settings, track_ids: Optional[list[str]] = None,
                 section_policy: str = "keep_as_own",
                 progress: ProgressCallback = None) -> None:
        super().__init__("Exporting")
        self.project = project
        self.settings = settings
        self.track_ids = track_ids
        self.section_policy = section_policy
        self.progress = progress
        self.report = None

    def run(self):
        from .exporting import export_project

        self.report = export_project(
            self.project,
            self.settings,
            track_ids=self.track_ids,
            section_policy=self.section_policy,
            progress=self.progress,
            cancel=lambda: self.cancelled,
        )
        return self.report


class LyricMatchJob(Job):
    """Read-only matching on a snapshot; analysis never edits tracks."""
    def __init__(self, project, references):
        super().__init__('Matching reference lyrics')
        self.project=project.clone(); self.references=references; self.progress=None
    def run(self):
        from ..analysis.lyrics import match_references, propose_audio_extent, evidence_signature
        from ..audio.timeline import build_reader
        segments=self.project.cache.get('theater_transcript',{}).get('segments',[])
        if not segments:raise ValueError('Run theatre transcription first. No local transcript is cached.')
        if self.progress:self.progress(.1,'Matching independent lyric anchors')
        report=match_references(self.references,segments,cancel=self.check_cancelled)
        report['evidence_signature']=evidence_signature(self.project)
        reader=build_reader(self.project)
        try:
            for result in report['results']:
                for occurrence in result['occurrences']:
                    self.check_cancelled()
                    occurrence['boundaries']=propose_audio_extent(reader,occurrence['start'],occurrence['end'])
        finally:reader.close()
        if self.progress:self.progress(1.,'Lyric proposals ready for listening')
        return report


# ----------------------------------------------------------------------
# Smart Metadata-Assisted Split
# ----------------------------------------------------------------------
class SmartSplitJob(Job):
    """Identify the recording online/locally and run metadata-assisted splitting."""

    def __init__(
        self,
        project,
        identification=None,
        allow_network: bool = True,
        allow_fingerprint: bool = True,
        transcribe_broadway: bool = False,
        cache_dir: Optional[str] = None,
        placement=None,
        progress: ProgressCallback = None,
    ) -> None:
        super().__init__("Smart split")
        self.project = project.clone()
        self.identification = identification
        self.allow_network = allow_network
        self.allow_fingerprint = allow_fingerprint
        self.transcribe_broadway = transcribe_broadway
        self.cache_dir = cache_dir
        self.placement = placement
        self.progress = progress

    def run(self):
        from ..analysis.metadata_split import (
            MetadataSplitOutcome,
            PlannedTrack,
            split_album_with_metadata,
            split_broadway_with_metadata,
            split_concert_with_metadata,
            split_dj_mix_with_metadata,
        )
        from ..audio.timeline import build_reader
        from ..metadata.identify import (
            KIND_ALBUM,
            KIND_BROADWAY,
            KIND_CONCERT,
            KIND_DJ_MIX,
            identify_recording,
        )

        sample_rate = self.project.sample_rate() or 44100
        total_frames = self.project.total_frames()
        total_seconds = total_frames / float(sample_rate)
        paths = [src.path for src in self.project.sources if src.path]

        ident = self.identification
        if ident is None:
            if self.progress:
                self.progress(0.05, "Identifying recording...")
            ident = identify_recording(
                paths,
                total_duration_seconds=total_seconds,
                allow_network=self.allow_network,
                allow_fingerprint=self.allow_fingerprint,
                cache_dir=self.cache_dir,
                progress=lambda f, m: self.progress(0.05 + 0.25 * f, m) if self.progress else None,
                cancel=lambda: self.cancelled,
            )
        self.check_cancelled()

        def _stage_progress(f: float, msg: str) -> None:
            if self.progress:
                self.progress(0.30 + 0.68 * f, msg)

        if ident.recording_kind == KIND_BROADWAY:
            outcome = split_broadway_with_metadata(
                self.project,
                ident,
                use_transcriber=self.transcribe_broadway,
                allow_fingerprint=self.allow_network and self.allow_fingerprint,
                cache_dir=self.cache_dir,
                progress=_stage_progress,
                cancel=lambda: self.cancelled,
            )
            if self.progress:
                self.progress(1.0, "Smart split complete")
            return outcome

        reader = build_reader(self.project)
        try:
            if ident.recording_kind == KIND_ALBUM and ident.worklist and len(ident.worklist.items) >= 2:
                outcome = split_album_with_metadata(
                    reader,
                    sample_rate,
                    total_frames,
                    ident,
                    placement=self.placement,
                    audio_path=paths[0] if paths else "",
                    allow_fingerprint=self.allow_network and self.allow_fingerprint,
                    cache_dir=self.cache_dir,
                    progress=_stage_progress,
                    cancel=lambda: self.cancelled,
                )
            elif ident.recording_kind == KIND_CONCERT:
                outcome = split_concert_with_metadata(
                    reader,
                    sample_rate,
                    total_frames,
                    ident,
                    placement=self.placement,
                    audio_path=paths[0] if paths else "",
                    allow_fingerprint=self.allow_network and self.allow_fingerprint,
                    cache_dir=self.cache_dir,
                    progress=_stage_progress,
                    cancel=lambda: self.cancelled,
                )
            elif ident.recording_kind == KIND_DJ_MIX:
                outcome = split_dj_mix_with_metadata(
                    reader,
                    sample_rate,
                    total_frames,
                    ident,
                    audio_path=paths[0] if paths else "",
                    allow_fingerprint=self.allow_network and self.allow_fingerprint,
                    cache_dir=self.cache_dir,
                    progress=_stage_progress,
                    cancel=lambda: self.cancelled,
                )
            else:
                # Unknown recording with no metadata match: fall back to standard audio analysis.
                analysis = AnalysisJob(
                    self.project,
                    sensitivity="balanced",
                    classify=True,
                    placement=self.placement,
                    progress=_stage_progress,
                ).run()
                outcome = MetadataSplitOutcome(
                    identification=ident,
                    suggestions=analysis.suggestions,
                    notes=["No online metadata matched; used audio-only boundary detection."],
                    elapsed_seconds=analysis.elapsed_seconds,
                )
                edges = sorted(
                    {0, total_frames}
                    | {int(s.frame) for s in analysis.suggestions if 0 < int(s.frame) < total_frames}
                )
                sugg_map = {int(s.frame): s for s in analysis.suggestions}
                for idx in range(len(edges) - 1):
                    s_f, e_f = edges[idx], edges[idx + 1]
                    sugg = sugg_map.get(s_f)
                    outcome.tracks.append(
                        PlannedTrack(
                            start_frame=s_f,
                            end_frame=e_f,
                            title=f"Track {idx + 1:02d}",
                            section_type="unknown",
                            boundary_confidence=float(sugg.confidence) if sugg else 0.0,
                            boundary_uncertain=bool(sugg.uncertain) if sugg else False,
                            boundary_placement="audio_fallback",
                            reasons=[sugg.reason] if sugg else ["Start of recording"],
                        )
                    )
        finally:
            reader.close()

        if self.progress:
            self.progress(1.0, "Smart split complete")
        return outcome
