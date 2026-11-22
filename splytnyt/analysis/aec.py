"""Audio event classification with a small trained model.

The heuristics in :mod:`splytnyt.analysis.classify` answer "is this music or
is this speech?" by looking at the spectrum. On real theatre audio that
question turns out to be much harder than it looks, because dialogue in a
musical comes *with* the orchestra underneath it and because a singer's
voice and a violin's voice look alike to a spectrum.

This module therefore uses a small trained classifier when one is available.
The model is YAMNet (a MobileNet-v1 trained on AudioSet), which is:

* **15.3 MB**, downloaded once and then used offline;
* **fast**: about two seconds for a whole act, so it is never the
  bottleneck;
* **licensed permissively** (Apache-2.0 for the code and weights, CC BY 4.0
  for the AudioSet label names).

The model is **optional**. Without it the application falls back to the
dependency-free measurements in :mod:`splytnyt.analysis.activities`, and
says so. The download shows its size before it starts, reports progress,
can be cancelled, and a failure never stops the analysis.

What the classifier is asked for is deliberately modest:

``music``
    Music of any kind: singing, an orchestra, a band.
``speech``
    People talking.
``singing``
    A voice singing. On some material this class is quiet, and the code
    does not pretend otherwise.
``applause``
    The audience.
``silence``
    Nothing audible.

The first two overlap on purpose: dialogue played over music raises both
at once, and that overlap is exactly what a "speech *or* music" choice
would throw away.
"""

from __future__ import annotations

import csv
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

#: Where the model and its label list come from.
MODEL_BASE_URL = "https://huggingface.co/audiomagic/yamnet-onnx/resolve/main"
MODEL_FILE = "yamnet.onnx"
CLASS_MAP_FILE = "yamnet_class_map.csv"

#: The exact sizes, measured from the files themselves.
MODEL_SIZE_BYTES = 16_093_355
CLASS_MAP_SIZE_BYTES = 14_096

#: The sample rate the model works at.
MODEL_SAMPLE_RATE = 16000

#: How far apart the classifier's frames are, in seconds.
FRAME_SECONDS = 0.48

#: How long each frame looks at, in seconds. The published formula for
#: the frame count does not describe this particular export of the model,
#: so the frame-to-time mapping was measured instead of assumed: a frame
#: covers the window that starts at its own time and runs for this long.
FRAME_WINDOW_SECONDS = 0.96

#: How much audio is handed to the model at a time, in frames and seconds.
#: 125 * 0.48 s = 60.0 s, an exact integer number of YAMNet frames.
BLOCK_FRAMES = 125
BLOCK_SECONDS = BLOCK_FRAMES * FRAME_SECONDS

#: How many extra frames (and seconds) are included around each block so
#: that an event on a block edge is seen whole without shifting the grid.
#: 4 * 0.48 s = 1.92 s (>= FRAME_WINDOW_SECONDS).
BLOCK_PADDING_FRAMES = 4
BLOCK_PADDING_SECONDS = BLOCK_PADDING_FRAMES * FRAME_SECONDS

#: The class groups that make up each activity. AudioSet has 521 labels;
#: these are the ones that matter for a musical-theatre recording.
MUSIC_CLASSES = (
    "Music",
    "Musical instrument",
    "Vocal music",
    "Background music",
    "Theme music",
    "A Cappella",
    "Symphony",
    "Classical music",
    "Orchestra",
    "Dance music",
    "Soundtrack music",
)

SPEECH_CLASSES = (
    "Speech",
    "Conversation",
    "Narration, monologue",
    "Speech synthesizer",
)

SINGING_CLASSES = (
    "Singing",
    "Choir",
    "Child singing",
    "Synthetic singing",
    "A Cappella",
    "Hum",
    "Yodeling",
)

APPLAUSE_CLASSES = (
    "Applause",
    "Whoop, whoop",
    "Cheering",
)

SILENCE_CLASSES = (
    "Silence",
)

#: The User-Agent used for the download.
_AGENT = "SplytNyt/2.0 (local-first audio splitter)"


@dataclass
class EventScores:
    """Per-frame event scores, on the classifier's own time grid."""

    music: np.ndarray = field(default=None)
    speech: np.ndarray = field(default=None)
    singing: np.ndarray = field(default=None)
    applause: np.ndarray = field(default=None)
    silence: np.ndarray = field(default=None)
    #: Seconds per frame.
    frame_seconds: float = 0.48
    #: True when the scores came from a trained model.
    from_model: bool = False
    source: str = ""

    def n_frames(self) -> int:
        return int(self.music.size) if self.music is not None else 0

    def as_dict(self) -> dict:
        return {
            "music": self.music,
            "speech": self.speech,
            "singing": self.singing,
            "applause": self.applause,
            "silence": self.silence,
        }


def default_model_root() -> str:
    """Where the classifier's files live."""
    base = os.environ.get("SPLYTNYT_MODEL_DIR", "").strip()
    if base:
        return os.path.abspath(base)
    return os.path.join(os.path.expanduser("~"), ".splytnyt", "models")


def onnxruntime_available() -> bool:
    """True when the optional runtime can be imported."""
    try:
        import onnxruntime
    except Exception:
        return False
    return onnxruntime is not None


def install_hint() -> str:
    """What to tell someone who wants the classifier but does not have it."""
    return (
        "Audio event classification needs the optional 'onnxruntime' package. "
        "Install it with:  pip install onnxruntime"
    )


def model_folder(root: Optional[str] = None) -> str:
    return os.path.join(root or default_model_root(), "yamnet")


def model_installed(root: Optional[str] = None) -> bool:
    """True when the classifier's files are already on disk."""
    folder = model_folder(root)
    return os.path.exists(os.path.join(folder, MODEL_FILE)) and os.path.exists(
        os.path.join(folder, CLASS_MAP_FILE)
    )


def describe_model(root: Optional[str] = None) -> dict:
    """What the interface needs to show before a download starts."""
    total = MODEL_SIZE_BYTES + CLASS_MAP_SIZE_BYTES
    return {
        "name": "YAMNet",
        "size_bytes": total,
        "size_text": human_size(total),
        "description": "A small trained audio-event classifier",
        "note": (
            "It tells music from speech, applause from silence, at about two "
            "seconds per hour of audio. Without it the application falls back "
            "to acoustic measurements alone."
        ),
        "installed": model_installed(root),
        "runtime_available": onnxruntime_available(),
    }


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


class ModelDownloadError(RuntimeError):
    """The classifier could not be fetched."""


class ModelDownloadCancelled(ModelDownloadError):
    """The user cancelled the download."""


def download_model(
    root: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> str:
    """Fetch the classifier and its label list.

    The size is known exactly before the download starts. The download
    reports its progress and can be cancelled; a failure raises, and the
    caller falls back to the acoustic measurements.
    """
    folder = model_folder(root)
    os.makedirs(folder, exist_ok=True)

    if cancel is not None and cancel():
        raise ModelDownloadCancelled("The download was cancelled before it started.")

    total = MODEL_SIZE_BYTES + CLASS_MAP_SIZE_BYTES
    done = 0
    for filename, expected in ((MODEL_FILE, MODEL_SIZE_BYTES), (CLASS_MAP_FILE, CLASS_MAP_SIZE_BYTES)):
        target = os.path.join(folder, filename)
        if os.path.exists(target) and os.path.getsize(target) >= expected:
            done += os.path.getsize(target)
            continue
        if cancel is not None and cancel():
            raise ModelDownloadCancelled("The download was cancelled.")
        url = f"{MODEL_BASE_URL}/{filename}"
        temporary = target + ".partial"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                received = 0
                with open(temporary, "wb") as handle:
                    while True:
                        if cancel is not None and cancel():
                            handle.close()
                            try:
                                os.remove(temporary)
                            except OSError:
                                pass
                            raise ModelDownloadCancelled("The download was cancelled.")
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        received += len(chunk)
                        if progress:
                            progress(
                                min(0.99, (done + received) / float(total)),
                                f"Downloading the audio-event classifier "
                                f"({human_size(done + received)} of {human_size(total)})",
                            )
            os.replace(temporary, target)
            done += os.path.getsize(target)
        except ModelDownloadCancelled:
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            try:
                os.remove(temporary)
            except OSError:
                pass
            raise ModelDownloadError(
                f"The audio-event classifier could not be downloaded: {exc}. "
                "The analysis will continue with acoustic measurements only."
            ) from exc

    if progress:
        progress(1.0, "The audio-event classifier is ready")
    return folder


def load_class_map(root: Optional[str] = None) -> dict[str, int]:
    """Map each AudioSet label to its column in the model's output."""
    path = os.path.join(model_folder(root), CLASS_MAP_FILE)
    mapping: dict[str, int] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                mapping[row["display_name"]] = int(row["index"])
            except (KeyError, TypeError, ValueError):
                continue
    return mapping


def _group_scores(scores, class_map: dict[str, int], names) -> np.ndarray:
    """The highest score among a group of labels."""

    total = np.zeros(scores.shape[0], dtype=np.float32)
    for name in names:
        index = class_map.get(name)
        if index is None:
            continue
        total = np.maximum(total, scores[:, index])
    return total


def smooth_scores(values: np.ndarray, size: int = 5) -> np.ndarray:
    """Smooth a score track in time.

    The classifier works on half-second frames and its scores jump around
    from one frame to the next. A median filter over about two and a half
    seconds removes the flicker without smearing a real change: a
    boundary between a song and dialogue lasts many frames, while the
    noise around a threshold lasts one or two.
    """

    if values.size == 0 or size < 2:
        return values
    try:
        from scipy.ndimage import median_filter

        return median_filter(values, size=size, mode="nearest").astype(np.float32)
    except Exception:  # pragma: no cover
        out = values.astype(np.float32).copy()
        half = size // 2
        for index in range(values.size):
            low = max(0, index - half)
            high = min(values.size, index + half + 1)
            out[index] = float(np.median(values[low:high]))
        return out


def load_session(root: Optional[str] = None):
    """Load the YAMNet ONNX session and class map once for repeated blocks."""
    if not onnxruntime_available():
        raise ModelDownloadError(install_hint())
    if not model_installed(root):
        raise ModelDownloadError(
            "The audio-event classifier is not downloaded yet."
        )

    import onnxruntime as ort

    folder = model_folder(root)
    class_map = load_class_map(root)
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, os.cpu_count() or 1)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        os.path.join(folder, MODEL_FILE),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session, class_map


def build_event_scores(scores, class_map: dict[str, int]) -> EventScores:
    """Convert raw stacked per-frame model scores into smoothed EventScores."""
    if scores.size == 0 or scores.shape[0] == 0:
        empty = np.zeros(0, dtype=np.float32)
        return EventScores(
            music=empty, speech=empty, singing=empty, applause=empty,
            silence=empty, from_model=True, source="yamnet",
        )
    # A self-check: a working classifier always says *something* somewhere.
    if float(np.max(scores)) <= 0.05:
        raise ModelDownloadError(
            "The audio-event classifier produced no usable output; "
            "the analysis will continue with acoustic measurements only."
        )
    return EventScores(
        music=smooth_scores(_group_scores(scores, class_map, MUSIC_CLASSES)),
        speech=smooth_scores(_group_scores(scores, class_map, SPEECH_CLASSES)),
        singing=smooth_scores(_group_scores(scores, class_map, SINGING_CLASSES)),
        applause=smooth_scores(_group_scores(scores, class_map, APPLAUSE_CLASSES)),
        silence=smooth_scores(_group_scores(scores, class_map, SILENCE_CLASSES)),
        frame_seconds=FRAME_SECONDS,
        from_model=True,
        source="yamnet",
    )


def _block_frame_slice(
    n_frames: int, left_frames: int, max_frames: int
) -> tuple[int, int]:
    """Slice exact grid-aligned frames inside a padded block."""
    if n_frames <= left_frames:
        return 0, 0
    end = min(n_frames, left_frames + max_frames)
    return left_frames, end


def classify_events(
    audio,
    root: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
    session=None,
    class_map: Optional[dict[str, int]] = None,
) -> EventScores:
    """Run the classifier over a signal and return per-frame scores.

    The signal must already be at 16 kHz; the caller resamples.

    Each block is padded symmetrically on an integer multiple of
    ``FRAME_SECONDS`` samples so frames align with ``index * FRAME_SECONDS``
    with zero cumulative drift across block boundaries.
    """
    if session is None or class_map is None:
        session, class_map = load_session(root)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return build_event_scores(np.zeros((0, 0), dtype=np.float32), class_map)

    step = int(round(FRAME_SECONDS * MODEL_SAMPLE_RATE))
    block = BLOCK_FRAMES * step
    pad_frames = BLOCK_PADDING_FRAMES
    padding = pad_frames * step

    results: list = []
    position = 0
    while position < audio.size:
        if cancel is not None and cancel():
            break
        left_frames = pad_frames if position > 0 else 0
        start = position - left_frames * step
        end = min(audio.size, position + block)
        end_pad = min(audio.size, end + padding)
        chunk = audio[start:end_pad]
        scores = session.run(None, {"waveform": chunk})[0]
        first, last_exc = _block_frame_slice(
            scores.shape[0], left_frames, BLOCK_FRAMES
        )
        if last_exc > first:
            results.append(scores[first:last_exc])
        position = end
        if progress:
            progress(
                min(1.0, position / float(audio.size)),
                f"Classifying audio events ({position / 60:.0f} s)",
            )

    if not results:
        return build_event_scores(np.zeros((0, 0), dtype=np.float32), class_map)

    scores = np.concatenate(results, axis=0)
    return build_event_scores(scores, class_map)


def classify_reader(
    reader,
    sample_rate: int,
    total_frames: int,
    root: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> EventScores:
    """Stream a recording through a single YAMNet session with zero drift.

    Reads ``BLOCK_SECONDS`` blocks with symmetric ``FRAME_SECONDS``-aligned
    left and right context so every emitted frame ``k`` begins at
    ``k * FRAME_SECONDS`` seconds and scores are smoothed once across the
    entire recording.
    """
    session, class_map = load_session(root)
    if total_frames <= 0 or sample_rate <= 0:
        return build_event_scores(np.zeros((0, 0), dtype=np.float32), class_map)

    from .transcribe import resample_to_whisper_rate

    pad_sec = BLOCK_PADDING_FRAMES * FRAME_SECONDS
    raw_chunks: list = []
    block_idx = 0
    while True:
        if cancel is not None and cancel():
            break
        t_pos = block_idx * BLOCK_SECONDS
        src_pos = int(round(t_pos * sample_rate))
        if src_pos >= total_frames:
            break
        left_frames = BLOCK_PADDING_FRAMES if block_idx > 0 else 0
        src_left_pad = (
            int(round(left_frames * FRAME_SECONDS * sample_rate))
            if left_frames > 0
            else 0
        )
        read_start = max(0, src_pos - src_left_pad)
        t_end = (block_idx + 1) * BLOCK_SECONDS
        src_end = min(total_frames, int(round(t_end * sample_rate)))
        src_right_pad = int(round(pad_sec * sample_rate))
        read_end = min(total_frames, src_end + src_right_pad)

        data = reader.read(read_start, read_end - read_start)
        if not data.size:
            break
        mono = data.mean(axis=1).astype(np.float32)
        prepared = resample_to_whisper_rate(mono, sample_rate)
        scores = session.run(None, {"waveform": prepared})[0]
        if left_frames > 0 and sample_rate > 0:
            actual_left_frames = int(
                round((src_pos - read_start) / sample_rate / FRAME_SECONDS)
            )
        else:
            actual_left_frames = 0
        first, last_exc = _block_frame_slice(
            scores.shape[0], actual_left_frames, BLOCK_FRAMES
        )
        if last_exc > first:
            raw_chunks.append(scores[first:last_exc])

        block_idx += 1
        if progress:
            progress(min(1.0, src_end / float(total_frames)), src_end)

    if not raw_chunks:
        return build_event_scores(np.zeros((0, 0), dtype=np.float32), class_map)
    return build_event_scores(np.concatenate(raw_chunks, axis=0), class_map)


def resample_to_model_rate(audio, sample_rate: int):
    """Resample to the 16 kHz the classifier expects."""
    from .transcribe import resample_to_whisper_rate

    return resample_to_whisper_rate(audio, sample_rate)
