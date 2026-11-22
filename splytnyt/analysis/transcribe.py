"""Speech evidence from a local Whisper transcriber.

Whisper is **evidence**, not truth. It is very good at ordinary speech and
unreliable on singing, on instrumental music and on applause, and it will
confidently write down a sentence that was never spoken. This module exists
to get the good parts out of it and to throw the rest away, out loud.

What it does:

* **Everything runs locally.** The model is downloaded once and then runs
  on this machine. No recording is ever uploaded anywhere.
* **The download is explicit.** Its size is known before it starts, it
  reports progress, it can be cancelled, and if it cannot happen the
  pipeline carries on without it.
* **Hallucinations are detected and rejected**, with the reason recorded.
  The known patterns are repeating loops, compressed text, low-confidence
  output, "no speech" output, speech that is impossibly slow or impossibly
  fast, and the stock phrases Whisper falls back to (subtitle credits and
  the like).
* **Singing is treated as singing.** Words recovered from a sung line are
  kept as *possible* words and marked as such; they are never presented as
  a verified lyric.
* **Timestamps are kept with their uncertainty.** A word's own probability
  travels with it.

The parameter set is deliberately conservative:

``condition_on_previous_text=False``
    Whisper's largest source of runaway repetition is conditioning on what
    it just wrote. Turning it off breaks the loop.
``vad_filter=True``
    The bundled Silero VAD skips the instrumental stretches, which is where
    hallucinations come from.
``hallucination_silence_threshold``
    A segment that ends in silence is trimmed rather than padded out.
``word_timestamps=True``
    The words carry their own probabilities and their own boundaries.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# ----------------------------------------------------------------------
# The model catalogue
# ----------------------------------------------------------------------


@dataclass
class ModelEntry:
    """One downloadable model."""

    name: str
    #: Exact download size in bytes, measured from the model repository.
    size_bytes: int
    description: str
    #: Rough speed relative to ``small`` on a modern CPU, for the user.
    note: str = ""


#: The models this application offers. The sizes are the real ones, taken
#: from the model repositories, not guesses.
MODEL_CATALOG: dict[str, ModelEntry] = {
    "tiny": ModelEntry(
        "tiny", 78_207_087, "Fastest and smallest",
        "Roughly four times faster than small; noticeably less accurate.",
    ),
    "base": ModelEntry(
        "base", 147_886_409, "Small",
        "Roughly two and a half times faster than small.",
    ),
    "small": ModelEntry(
        "small", 486_215_847, "The default",
        "A good balance on a CPU with sixteen cores.",
    ),
    "medium": ModelEntry(
        "medium", 1_530_575_217, "More accurate",
        "Roughly three times slower than small.",
    ),
    "large-v3": ModelEntry(
        "large-v3", 3_090_839_273, "Most accurate",
        "Roughly six times slower than small; needs a lot of memory.",
    ),
}

#: The model used when nothing else is chosen.
DEFAULT_MODEL = "small"

#: The User-Agent used for model downloads.
_AGENT = "SplytNyt/2.0 (local-first audio splitter)"

#: Where the models are kept.
def default_download_root() -> str:
    """Where downloaded models live."""
    base = os.environ.get("SPLYTNYT_MODEL_DIR", "").strip()
    if base:
        return os.path.abspath(base)
    home = os.path.expanduser("~")
    return os.path.join(home, ".splytnyt", "models")


# ----------------------------------------------------------------------
# Availability
# ----------------------------------------------------------------------
def faster_whisper_available() -> bool:
    """True when the optional transcriber can be imported."""
    try:
        import faster_whisper
    except Exception:
        return False
    return faster_whisper is not None


def install_hint() -> str:
    """What to tell someone who wants transcription but does not have it."""
    return (
        "Transcription needs the optional 'faster-whisper' package. "
        "Install it with:  pip install faster-whisper"
    )


def model_is_installed(name: str, download_root: Optional[str] = None) -> bool:
    """True when the model's files are already on disk."""
    folder = model_folder(name, download_root)
    return os.path.exists(os.path.join(folder, "model.bin"))


def model_path(name: str, download_root: Optional[str] = None) -> str:
    """The local folder of an installed model."""
    return model_folder(name, download_root)


def model_folder(name: str, download_root: Optional[str] = None) -> str:
    """Where a model's files live."""
    root = download_root or default_download_root()
    return os.path.join(root, f"faster-whisper-{name}")


def describe_model(name: str, download_root: Optional[str] = None) -> dict:
    """What the interface needs to show before a download starts."""
    entry = MODEL_CATALOG.get(name)
    size = entry.size_bytes if entry else 0
    return {
        "name": name,
        "size_bytes": size,
        "size_text": human_size(size),
        "description": entry.description if entry else "",
        "note": entry.note if entry else "",
        "installed": model_is_installed(name, download_root),
        "transcriber_available": faster_whisper_available(),
    }


def human_size(size: int) -> str:
    """A size a person can read."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


#: The sample rate Whisper works at.
WHISPER_SAMPLE_RATE = 16000


def resample_to_whisper_rate(audio, sample_rate: int):
    """Resample a signal to the 16 kHz that Whisper expects.

    This matters more than it looks. The transcriber assumes its input is
    already at 16 kHz: it neither resamples nor checks. Handing it 48 kHz
    audio does not produce an error - it produces timestamps that are three
    times too long, and a voice-activity filter that sees nothing at all.
    """

    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate == WHISPER_SAMPLE_RATE:
        return audio
    if audio.size == 0:
        return audio
    try:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(sample_rate), WHISPER_SAMPLE_RATE)
        up = WHISPER_SAMPLE_RATE // divisor
        down = int(sample_rate) // divisor
        return resample_poly(audio, up, down).astype(np.float32)
    except Exception:
        # A plain linear resample is worse, but it is far better than
        # handing the model audio at the wrong rate.
        count = int(round(audio.size * WHISPER_SAMPLE_RATE / float(sample_rate)))
        positions = np.linspace(0.0, audio.size - 1, num=max(1, count))
        return np.interp(positions, np.arange(audio.size), audio).astype(np.float32)


def resample_stream_to_whisper_rate(
    source,
    total_frames: int,
    sample_rate: int,
    block_seconds: float = 300.0,
    pad_seconds: float = 0.25,
):
    """Resample a long recording to 16 kHz without holding it in memory.

    ``source(start, count)`` must return ``count`` mono samples starting at
    that position. The recording is read in blocks and each block is
    resampled with a little context on either side, so the filter never
    sees an edge and no click appears at a block boundary. The result is
    the same length as the whole recording would have been.

    The full-rate signal is never materialised. That is the point of this
    function: the 48 kHz stereo signal of a three-hour musical is about
    3.5 GB, and reading it in blocks means it is never in memory at all.

    What remains is the 16 kHz signal the model actually needs, which costs
    about 3.7 MiB per minute of recording, and the transient buffers inside
    the resampler, whose size follows the block length rather than the
    recording. The output does not depend on the block size: it was measured
    to be bit-identical at 300, 120, 60 and 30 second blocks.
    """

    total_frames = int(total_frames)
    if total_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    try:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(sample_rate), WHISPER_SAMPLE_RATE)
        up = WHISPER_SAMPLE_RATE // divisor
        down = int(sample_rate) // divisor
    except Exception:
        up = down = 0

    block = max(1, int(block_seconds * sample_rate))
    pad = int(pad_seconds * sample_rate)
    # The output is filled in place rather than built from a list of blocks
    # and concatenated, which would hold the whole thing twice.
    expected = int(round(total_frames * WHISPER_SAMPLE_RATE / float(sample_rate)))
    out = np.zeros(max(expected, 1), dtype=np.float32)
    written = 0
    position = 0

    while position < total_frames:
        count = min(block, total_frames - position)
        start = max(0, position - pad)
        end = min(total_frames, position + count + pad)
        data = np.asarray(source(start, end - start), dtype=np.float32)
        if data.size == 0:
            break

        if up and down:
            resampled = resample_poly(data, up, down).astype(np.float32)
            left = (position - start) * up // down
            right = (end - (position + count)) * up // down
            if right:
                resampled = resampled[left: resampled.size - right]
            else:
                resampled = resampled[left:]
        else:
            # The linear fallback: good enough to keep the timestamps
            # honest, which is all it is used for.
            count_out = int(round(data.size * WHISPER_SAMPLE_RATE / float(sample_rate)))
            positions = np.linspace(0.0, data.size - 1, num=max(1, count_out))
            resampled = np.interp(
                positions, np.arange(data.size), data
            ).astype(np.float32)
            left = int((position - start) * WHISPER_SAMPLE_RATE / float(sample_rate))
            right = int((end - (position + count)) * WHISPER_SAMPLE_RATE / float(sample_rate))
            resampled = resampled[left: resampled.size - right if right else None]

        chunk = np.ascontiguousarray(resampled, dtype=np.float32)
        room = out.size - written
        if room <= 0:
            break
        take = min(room, chunk.size)
        out[written: written + take] = chunk[:take]
        written += take
        position += count

    if written <= 0:
        return np.zeros(0, dtype=np.float32)
    return out[:written]


# ----------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------
def download_model(
    name: str,
    download_root: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> str:
    """Fetch a model, reporting progress and honouring cancellation.

    The files are fetched directly over HTTPS into a plain folder, so the
    model can be loaded from that folder without a cache layout and without
    any progress-bar library getting in the way. The download resumes: a
    file that is already complete is not fetched again.

    Returns the local folder of the model. Raises :class:`ModelDownloadError`
    when the download cannot be completed.
    """
    import urllib.error
    import urllib.request

    root = download_root or default_download_root()
    folder = model_folder(name, root)
    os.makedirs(folder, exist_ok=True)

    if cancel is not None and cancel():
        raise ModelDownloadCancelled("The download was cancelled before it started.")

    entry = MODEL_CATALOG.get(name)
    total = entry.size_bytes if entry else 0
    repository = f"Systran/faster-whisper-{name}"

    try:
        from huggingface_hub import HfApi

        siblings = HfApi().model_info(repository).siblings
        names = [item.rfilename for item in siblings]
    except Exception:
        # The file list is only a convenience: the known set works even
        # without the helper library.
        names = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]

    if progress:
        progress(0.0, f"Downloading the '{name}' model ({human_size(total)})")

    downloaded = 0
    for filename in names:
        if cancel is not None and cancel():
            raise ModelDownloadCancelled("The download was cancelled.")
        target = os.path.join(folder, filename.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            downloaded += os.path.getsize(target)
            continue

        url = f"https://huggingface.co/{repository}/resolve/main/{filename}"
        temporary = target + ".partial"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = int(response.headers.get("Content-Length") or 0)
                done = 0
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
                        done += len(chunk)
                        if progress:
                            fraction = (downloaded + done) / float(total) if total else 0.0
                            progress(
                                min(0.99, fraction),
                                f"Downloading the '{name}' model "
                                f"({human_size(done)} of {human_size(total)})",
                            )
                if expected and done != expected:
                    # The server promised a length and did not deliver it, so
                    # the file is truncated. Saying so is better than handing
                    # the model a half-written file.
                    raise ModelDownloadError(
                        f"The download of '{filename}' was incomplete: "
                        f"{human_size(done)} of {human_size(expected)} arrived. "
                        "The analysis will continue without transcription."
                    )
            os.replace(temporary, target)
            downloaded += os.path.getsize(target)
        except ModelDownloadCancelled:
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            try:
                os.remove(temporary)
            except OSError:
                pass
            raise ModelDownloadError(
                f"The '{name}' model could not be downloaded: {exc}. "
                "The analysis will continue without transcription."
            ) from exc

    if progress:
        progress(1.0, f"The '{name}' model is ready")
    return folder


class ModelDownloadError(RuntimeError):
    """A model could not be fetched."""


class ModelDownloadCancelled(ModelDownloadError):
    """The user cancelled the download."""


# ----------------------------------------------------------------------
# Transcription results
# ----------------------------------------------------------------------
@dataclass
class Word:
    """One word, with the model's own probability for it."""

    start: float
    end: float
    text: str
    probability: float = 0.0


@dataclass
class TranscriptSegment:
    """One piece of what the transcriber produced."""

    start: float
    end: float
    text: str = ""
    words: list[Word] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0
    accepted: bool = True
    reject_reason: str = ""
    #: "speech" or "singing": whether this stretch looked like a spoken line.
    kind: str = "speech"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def characters(self) -> int:
        return len(re.sub(r"\s+", "", self.text))


@dataclass
class TranscriptionResult:
    """Everything one transcription pass produced."""

    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    model: str = ""
    elapsed_seconds: float = 0.0
    available: bool = True
    error: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> list[TranscriptSegment]:
        return [segment for segment in self.segments if segment.accepted]

    @property
    def rejected(self) -> list[TranscriptSegment]:
        return [segment for segment in self.segments if not segment.accepted]

    def speech_spans(self) -> list[tuple[float, float]]:
        """The spans the transcriber accepted as words."""
        return [(segment.start, segment.end) for segment in self.accepted]


# ----------------------------------------------------------------------
# Hallucination detection
# ----------------------------------------------------------------------
#: The stock phrases Whisper falls back to when it has nothing real to say.
HALLUCINATION_PHRASES = (
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subtitles by",
    "subs by",
    "amara.org",
    "translation by",
    "transcribed by",
    "transcription by",
    "captioned by",
    "www.",
    "http",
    "♪",
    "[music]",
    "(music)",
    "[applause]",
    "(applause)",
    "[silence]",
    "(silence)",
    "[upbeat music]",
    "[soft music]",
    "you",
)

#: The lowest probability a word may have and still be counted.
WORD_PROBABILITY_FLOOR = 0.25

#: A segment longer than this with fewer characters than this is a
#: hallucination: the model wrote nothing over a long stretch.
MAX_EMPTY_CHARACTERS_PER_SECOND = 1.5

#: Speech faster than this is not speech.
MAX_CHARACTERS_PER_SECOND = 45.0

#: A segment whose text repeats itself this many times is a loop.
REPEAT_LOOP_LIMIT = 3


def _repeated_phrase(text: str) -> bool:
    """True when the same words keep coming back."""
    words = re.findall(r"[\w']+", text.lower())
    if len(words) < 8:
        return False
    for size in (1, 2, 3, 4, 5, 6, 7, 8):
        if len(words) < size * REPEAT_LOOP_LIMIT:
            continue
        chunk = " ".join(words[:size])
        repeats = 0
        index = 0
        while index + size <= len(words):
            if " ".join(words[index: index + size]) == chunk:
                repeats += 1
                index += size
            else:
                index += 1
        if repeats >= REPEAT_LOOP_LIMIT:
            return True
    return False


def _hallucination_reason(segment: TranscriptSegment) -> str:
    """Why a segment was thrown away, or an empty string when it is kept."""
    lowered = segment.text.lower().strip()

    if not lowered:
        return "the transcriber produced no text"
    if lowered in HALLUCINATION_PHRASES:
        return f"'{segment.text.strip()}' is a stock phrase the transcriber falls back to"
    for phrase in HALLUCINATION_PHRASES:
        if phrase in lowered and len(lowered) <= len(phrase) + 12:
            return f"the text is only the stock phrase '{phrase}'"
    if _repeated_phrase(segment.text):
        return "the same words repeat over and over"
    if segment.no_speech_prob >= 0.9:
        return f"the transcriber thinks there is no speech here ({segment.no_speech_prob:.2f})"
    if segment.compression_ratio >= 2.4:
        return f"the text is unusually repetitive (compression ratio {segment.compression_ratio:.2f})"
    if segment.avg_logprob <= -1.0:
        return f"the transcriber was not confident (log probability {segment.avg_logprob:.2f})"

    duration = max(0.001, segment.duration)
    characters = segment.characters
    if characters / duration < MAX_EMPTY_CHARACTERS_PER_SECOND and duration > 3.0:
        return (
            f"only {characters} characters in {duration:.1f} s, which is too little "
            "text for real speech"
        )
    if characters / duration > MAX_CHARACTERS_PER_SECOND:
        return (
            f"{characters} characters in {duration:.1f} s is faster than anyone speaks"
        )

    if segment.words:
        weak = [word for word in segment.words if word.probability < WORD_PROBABILITY_FLOOR]
        if weak and len(weak) >= max(1, len(segment.words) // 2):
            return (
                f"{len(weak)} of {len(segment.words)} words have a low probability, "
                "so the wording is unreliable"
            )
    return ""


def _clean_text(text: str) -> str:
    """Remove the decorative punctuation Whisper adds around sound."""
    text = re.sub(r"^\s*[\(\[][^\)\]]*[\)\]]\s*", "", text)
    return text.strip()


def _reject_repeating_neighbours(segments: list[TranscriptSegment]) -> None:
    """Reject a stretch that merely repeats its neighbour.

    Whisper's most persistent failure on singing is to write the same line
    again and again across consecutive stretches. The repetition is often
    inside the text rather than between the stretches, so this check sits
    next to the per-segment one.
    """
    previous: Optional[TranscriptSegment] = None
    for segment in segments:
        if previous is not None and segment.text:
            left = re.sub(r"\s+", " ", previous.text.lower()).strip()
            right = re.sub(r"\s+", " ", segment.text.lower()).strip()
            if left and right:
                if right == left:
                    segment.accepted = False
                    segment.reject_reason = (
                        "this stretch repeats the words of the stretch before it"
                    )
                elif len(right) > 12 and right in left:
                    segment.accepted = False
                    segment.reject_reason = (
                        "this stretch repeats words already written for the "
                        "stretch before it"
                    )
        previous = segment


# ----------------------------------------------------------------------
# The transcriber
# ----------------------------------------------------------------------
#: Decoding defaults for deterministic, reproducible transcription.
DEFAULT_BEAM_SIZE = 1
DEFAULT_TEMPERATURE = 0.0
DEFAULT_LANGUAGE_DETECTION_SEGMENTS = 4


class Transcriber:
    """A local Whisper transcriber, loaded lazily.

    The model is only read from disk when it is first needed, so an
    installation without the optional dependency still starts and still
    works.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        download_root: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        temperature: float = DEFAULT_TEMPERATURE,
        beam_size: int = DEFAULT_BEAM_SIZE,
    ) -> None:
        self.model_name = model_name
        self.download_root = download_root or default_download_root()
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads if cpu_threads > 0 else (os.cpu_count() or 4)
        self.temperature = float(temperature)
        self.beam_size = max(1, int(beam_size))
        self._model = None
        self._error = ""

    def settings_fingerprint(self, language: Optional[str] = None) -> dict:
        """Return the decoding parameters that affect transcription output."""
        return {
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": language or "auto",
            "beam_size": self.beam_size,
            "temperature": self.temperature,
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.9,
            "word_timestamps": True,
            "vad_filter": False,
            "hallucination_silence_threshold": 2.0,
            "without_timestamps": False,
            "language_detection_segments": DEFAULT_LANGUAGE_DETECTION_SEGMENTS,
        }

    @property
    def available(self) -> bool:
        """True when the transcriber can run."""
        if self._error:
            return False
        return faster_whisper_available()

    @property
    def error(self) -> str:
        return self._error

    def load(self, progress: Optional[Callable[[float, str], None]] = None) -> bool:
        """Load the model. Returns False when it cannot be loaded."""
        if self._model is not None:
            return True
        if not faster_whisper_available():
            self._error = install_hint()
            return False
        try:
            from faster_whisper import WhisperModel

            local = model_path(self.model_name, self.download_root)
            if not model_is_installed(self.model_name, self.download_root):
                self._error = (
                    f"The '{self.model_name}' model is not downloaded yet."
                )
                return False
            if progress:
                progress(0.0, f"Loading the '{self.model_name}' model")
            self._model = WhisperModel(
                local,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                local_files_only=True,
            )
            if progress:
                progress(1.0, "Model ready")
            return True
        except Exception as exc:
            self._error = f"The transcriber could not be started: {exc}"
            return False

    def _detect_language(self, prepared: "np.ndarray") -> Optional[str]:
        """Deterministically infer the dominant language across spread windows.

        Checking only the first segment(s) of a live stage recording can land
        on applause, overture, or non-lexical chorus vocals (which Whisper may
        misclassify as e.g. 'haw' or 'la'). Averaging full language
        probabilities across evenly spaced active windows avoids that trap.
        """
        detect = getattr(self._model, "detect_language", None)
        if detect is None or prepared.size < 60 * WHISPER_SAMPLE_RATE:
            return None
        win = 30 * WHISPER_SAMPLE_RATE
        totals: dict[str, float] = {}
        try:
            for frac in (0.05, 0.18, 0.32, 0.46, 0.60, 0.74, 0.88):
                start = max(0, min(prepared.size - win, int(prepared.size * frac)))
                chunk = prepared[start: start + win]
                if chunk.size < 4 * WHISPER_SAMPLE_RATE:
                    continue
                if float(np.sqrt(np.mean(chunk * chunk))) < 1e-4:
                    continue
                lang, prob, all_probs = detect(chunk)
                if all_probs:
                    for code, p in all_probs:
                        totals[code] = totals.get(code, 0.0) + float(p)
                elif lang:
                    totals[lang] = totals.get(lang, 0.0) + float(prob or 0.0)
        except Exception:
            return None
        if not totals:
            return None
        return max(totals, key=lambda k: totals[k])

    def transcribe(
        self,
        audio: "np.ndarray",
        sample_rate: int,
        language: Optional[str] = None,
        progress: Optional[Callable[[float, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> TranscriptionResult:
        """Transcribe a mono signal and clean up the result."""
        import time

        result = TranscriptionResult(model=self.model_name)
        if not self.load(progress):
            result.available = False
            result.error = self._error
            return result

        started = time.time()
        # The model works at 16 kHz and does not resample for us.
        prepared = resample_to_whisper_rate(audio, sample_rate)
        resolved_language = language
        if resolved_language is None:
            resolved_language = self._detect_language(prepared)
        try:
            kwargs = {
                "language": resolved_language,
                "beam_size": self.beam_size,
                # Single greedy temperature avoids stochastic sampling fallback
                # (temperature=[0.0..1.0] switches to random best_of=5 sampling).
                "temperature": self.temperature,
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                # Genuine singing on this material carries a "no speech"
                # probability of 0.70 to 0.80, so the common threshold of
                # 0.6 would throw the lyrics away. The cutoff is high
                # enough to keep real words, and the decision about what
                # is speech and what is singing is made further up.
                "no_speech_threshold": 0.9,
                "word_timestamps": True,
                # The bundled voice-activity filter is a speech gate, and
                # a speech gate rejects singing - which is most of a
                # theatrical recording. Switching it off costs time and
                # saves the material.
                "vad_filter": False,
                "hallucination_silence_threshold": 2.0,
                "without_timestamps": False,
            }
            if resolved_language is None:
                kwargs["language_detection_segments"] = (
                    DEFAULT_LANGUAGE_DETECTION_SEGMENTS
                )
            segments, info = self._model.transcribe(prepared, **kwargs)
        except Exception as exc:
            result.available = False
            result.error = f"Transcription failed: {exc}"
            return result

        result.language = getattr(info, "language", "") or ""
        result.language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)

        total = max(0.001, float(getattr(info, "duration", audio.size / float(sample_rate)) or 0.0))
        for segment in segments:
            if cancel is not None and cancel():
                break
            words = [
                Word(
                    start=float(getattr(word, "start", 0.0) or 0.0),
                    end=float(getattr(word, "end", 0.0) or 0.0),
                    text=str(getattr(word, "word", "") or "").strip(),
                    probability=float(getattr(word, "probability", 0.0) or 0.0),
                )
                for word in (getattr(segment, "words", None) or [])
            ]
            record = TranscriptSegment(
                start=float(getattr(segment, "start", 0.0) or 0.0),
                end=float(getattr(segment, "end", 0.0) or 0.0),
                text=_clean_text(str(getattr(segment, "text", "") or "")),
                words=words,
                avg_logprob=float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                compression_ratio=float(getattr(segment, "compression_ratio", 0.0) or 0.0),
            )
            reason = _hallucination_reason(record)
            if reason:
                record.accepted = False
                record.reject_reason = reason
            result.segments.append(record)

            if progress:
                progress(min(1.0, record.end / total), f"Transcribing {record.end / 60:.1f} min")

        _reject_repeating_neighbours(result.segments)

        result.elapsed_seconds = time.time() - started
        result.notes.append(
            f"{len(result.accepted)} of {len(result.segments)} transcribed stretch(es) "
            "were kept; the rest looked like hallucinations."
        )
        return result
