"""Section classification.

Two ways to reach the same verdict, behind one call signature.

**With the trained classifier.** When the optional audio-event model is
installed, the recording is classified with it and its scores decide. This
is the better path, and it is the one the theatre mode uses.

**Without it.** The fallback is an honest **heuristic**: it uses energy,
spectral flatness, band ratios and modulation to decide whether a section
looks like music, applause, speech or silence, and the reasons it returns
always say that it is a heuristic, so nobody mistakes it for a measurement.

The fallback is weak on real material, and the reasons why were measured
rather than guessed. On a real stage recording the geometric-mean spectral
flatness of *music*, of *speech* and of *applause* alike was between 1e-8
and 1e-5 - three to four orders of magnitude below the absolute thresholds
this module used to carry - so the "music" branch fired on everything and
the "applause" branch could never fire. Clipping the power spectrum before
the log, the normalised spectral entropy and the harmonic share were all
measured as alternatives, and none of them separated the three either. The
conclusion is that a section-local spectral measurement cannot tell music
from speech from applause in a full-band recording, so the honest response
is to use the trained model when it is there and to say plainly when it is
not.

The interface is deliberately narrow so that the trained classifier can
replace the heuristic without touching the rest of the application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: Levels below this are treated as digital silence.
SILENCE_DBFS = -70.0

#: Spectral flatness above this means "noisy, no clear pitch".
FLATNESS_NOISE = 0.18
FLATNESS_TONAL = 0.06

#: The trained classifier's thresholds. They are the ones the theatre mode
#: uses as well, so both modes answer consistently.
MODEL_MUSIC_ON = 0.35
MODEL_SPEECH_ON = 0.35
MODEL_APPLAUSE_ON = 0.25


@dataclass
class SectionGuess:
    """The heuristic's opinion about a section."""

    section_type: str
    confidence: float
    reasons: list[str] = field(default_factory=list)


def classify_with_model(events) -> SectionGuess:
    """Classify a stretch from the trained classifier's scores.

    ``events`` is an :class:`splytnyt.analysis.aec.EventScores` covering the
    stretch being classified.
    """
    if events is None or events.n_frames() == 0:
        return SectionGuess(
            "unknown", 0.2, ["there are no classifier scores for this stretch"]
        )

    music = float(np.mean(events.music))
    speech = float(np.mean(events.speech))
    singing = float(np.mean(events.singing))
    applause = float(np.mean(events.applause))
    silence = float(np.mean(events.silence))

    reasons = [
        f"the audio-event classifier found music {music:.2f}, "
        f"speech {speech:.2f}, singing {singing:.2f} and applause {applause:.2f}"
    ]

    if applause >= MODEL_APPLAUSE_ON and applause >= music:
        return SectionGuess("applause", min(0.8, 0.4 + applause * 0.4), reasons)
    if silence >= 0.5 and music < MODEL_MUSIC_ON and speech < MODEL_SPEECH_ON:
        reasons.append("the stretch is quiet")
        return SectionGuess("silence", 0.7, reasons)
    if music >= MODEL_MUSIC_ON and music >= speech:
        if singing > 0.0:
            reasons.append("there is a singing voice in it")
        return SectionGuess("music", min(0.85, 0.45 + music * 0.4), reasons)
    if speech >= MODEL_SPEECH_ON:
        return SectionGuess("speech", min(0.8, 0.4 + speech * 0.4), reasons)
    reasons.append("no decisive pattern")
    return SectionGuess("unknown", 0.3, reasons)


def _spectrum_features(signal: np.ndarray, sample_rate: int) -> dict:
    """Energy distribution and spectral flatness of a signal.

    Several windows spread across the stretch are measured and the median
    of each measurement is taken. Looking at only the first fraction of a
    second used to describe the whole section, which is not enough of it
    to say anything about.
    """
    empty = {
        "rms_dbfs": -120.0,
        "flatness": 0.0,
        "low": 0.0,
        "mid": 0.0,
        "high": 0.0,
        "centroid": 0.0,
        "zcr": 0.0,
        "crest": 0.0,
    }

    # Up to five windows spread across the stretch. The median of each
    # measurement is taken, so one loud or one quiet moment does not
    # describe the whole section.
    windows: list[dict] = []
    if signal.size >= 1024:
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            offset = int((signal.size - 1) * fraction)
            measured = _window_features(signal, sample_rate, offset)
            if measured is not None:
                windows.append(measured)
    if not windows:
        return empty

    return {
        key: float(np.median([w[key] for w in windows]))
        for key in empty
    }


def _window_features(signal: np.ndarray, sample_rate: int, offset: int) -> Optional[dict]:
    """The measurements for one window of the stretch."""
    size = 8192
    if signal.size < offset + 512:
        return None
    window = signal[offset: offset + size]
    if window.size < 512:
        window = signal[-512:]

    rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
    rms_db = 20.0 * np.log10(max(rms, 1e-12))
    peak = float(np.max(np.abs(window)))
    crest = peak / max(rms, 1e-12)

    spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size)))
    power = np.square(spectrum)
    total = float(np.sum(power))
    if total <= 0:
        return {
            "rms_dbfs": rms_db, "flatness": 1.0, "low": 0.0, "mid": 0.0,
            "high": 0.0, "centroid": 0.0, "zcr": 0.0, "crest": crest,
        }

    freqs = np.fft.rfftfreq(window.size, 1.0 / sample_rate)
    low = float(np.sum(power[freqs < 200.0])) / total
    mid = float(np.sum(power[(freqs >= 200.0) & (freqs < 2000.0)])) / total
    high = float(np.sum(power[freqs >= 2000.0])) / total
    centroid = float(np.sum(freqs * power) / total)

    positive = power[power > 0]
    if positive.size:
        geometric = float(np.exp(np.mean(np.log(positive))))
        arithmetic = float(np.mean(positive))
        flatness = geometric / arithmetic if arithmetic > 0 else 0.0
    else:
        flatness = 0.0

    crossings = float(np.mean(np.abs(np.diff(np.signbit(window)))))
    return {
        "rms_dbfs": rms_db,
        "flatness": flatness,
        "low": low,
        "mid": mid,
        "high": high,
        "centroid": centroid,
        "zcr": crossings,
        "crest": crest,
    }


def _modulation(signal: np.ndarray, sample_rate: int) -> float:
    """How strongly the level varies over time.

    The spread is measured with the interquartile range of the *linear*
    frame levels rather than the standard deviation of the decibel levels.
    A single quiet passage inside a song moves the standard deviation a long
    way but barely moves the interquartile range, while speech modulates
    continuously and moves both.

    The measurement is made over short stretches and the median of those is
    taken. Music swells: a chorus that is twice as loud as the verse is
    normal, and over a long window that looks like heavy modulation. Over a
    third of a second it is not. Speech, on the other hand, modulates
    several times a second, so it stays high in every short stretch.
    """
    from .silence import frame_levels

    levels = frame_levels(signal, frame_frames=1024, hop_frames=512)
    if levels.size < 8:
        return 0.0
    linear = np.power(10.0, levels / 20.0)

    window = max(4, int(0.3 * sample_rate / 512))
    if levels.size < window * 2:
        return _spread(linear)

    spreads = []
    for start in range(0, levels.size - window + 1, window):
        spreads.append(_spread(linear[start: start + window]))
    if not spreads:
        return _spread(linear)
    return float(np.median(spreads))


def _spread(linear: np.ndarray) -> float:
    """The interquartile range of the linear levels, relative to the median."""
    if linear.size < 4:
        return 0.0
    q1, q3 = np.percentile(linear, [25.0, 75.0])
    median = float(np.median(linear))
    if median <= 1e-9:
        return 0.0
    return float(max(0.0, q3 - q1) / median)


def _decay_profile(signal: np.ndarray, sample_rate: int) -> float:
    """How much the level falls from the start to the end (applause decays).

    Measured in decibels over the first and last quarter of the section.
    Tape hiss and room tone do not decay; applause does.
    """
    from .silence import frame_levels

    levels = frame_levels(signal, frame_frames=2048, hop_frames=1024)
    if levels.size < 8:
        return 0.0
    quarter = max(1, levels.size // 4)
    head = float(np.mean(levels[:quarter]))
    tail = float(np.mean(levels[-quarter:]))
    return head - tail


def classify_section(
    signal: np.ndarray,
    sample_rate: int,
    events=None,
) -> SectionGuess:
    """Guess what kind of audio a section contains.

    When ``events`` is given - the trained audio-event classifier's scores
    for this stretch - they decide, and the reasons say so. Without them
    the heuristic decides, and the reasons say *that*, so nobody mistakes
    one for the other.

    The result is an estimate with a confidence and the evidence behind it.
    """
    if signal.size == 0:
        return SectionGuess("silence", 0.9, ["heuristic: no audio in this section"])

    if events is not None:
        return classify_with_model(events)

    features = _spectrum_features(signal, sample_rate)
    rms_db = features["rms_dbfs"]
    flatness = features["flatness"]
    reasons: list[str] = []

    if rms_db <= SILENCE_DBFS:
        return SectionGuess(
            "silence",
            0.95,
            [f"heuristic: level {rms_db:.1f} dBFS is at or below the silence threshold"],
        )

    modulation = _modulation(signal, sample_rate)
    decay = _decay_profile(signal, sample_rate)

    # --- applause: broadband, noisy, no pitch, and it decays ---------
    # Stationary hiss (a noisy tape) is broadband too, but it does not
    # decay, so the decay test keeps the two apart.
    #
    # This branch rarely fires on a stage recording, and the reason was
    # measured: in a theatre the applause happens *over* the orchestra's
    # last chord, so the spectrum is tonal and the flatness is low. The
    # test is kept because it is right when the audience is away from the
    # band - a concert recording, for instance.
    if flatness >= 0.22 and decay > 4.0 and features["high"] > 0.30:
        reasons.append(
            f"heuristic: broadband noise (flatness {flatness:.2f}) with "
            f"{features['high'] * 100:.0f}% of the energy above 2 kHz that decays by "
            f"{decay:.0f} dB - typical of applause"
        )
        confidence = min(0.75, 0.35 + flatness * 0.5 + min(0.25, decay / 60.0))
        return SectionGuess("applause", confidence, reasons)

    # --- speech: modulated, and weighted to the voice band -----------
    # Checked *before* music, which is the opposite of the order this
    # module used to use, and the reason is measured. On a real stage
    # recording the music, the speech and the applause all have a
    # geometric-mean flatness between 1e-8 and 1e-5, so a "tonal spectrum"
    # test passes for all three of them and the music branch swallowed
    # every section - including clear dialogue.
    #
    # The level modulation is what does separate them: measured over the
    # same material, spoken stretches modulated 0.31 to 0.43 and musical
    # ones 0.27 to 0.29. Speech moves the level several times a second;
    # music swells slowly. So the speech test comes first, and what is
    # left over is music.
    if modulation >= 0.30 and features["mid"] > 0.20:
        reasons.append(
            f"heuristic: continuously modulated level (variation {modulation:.2f}) with "
            f"{features['mid'] * 100:.0f}% of the energy in the 200-2000 Hz voice band"
        )
        confidence = min(0.7, 0.3 + modulation * 0.6)
        return SectionGuess("speech", confidence, reasons)

    # --- music: a tonal, stable spectrum -----------------------------
    if flatness <= 0.12 and modulation < 0.30 and features["zcr"] < 0.25:
        reasons.append(
            f"heuristic: tonal spectrum (flatness {flatness:.2f}) with "
            f"{(features['low'] + features['mid']) * 100:.0f}% of the energy below 2 kHz "
            f"and a stable level (modulation {modulation:.2f})"
        )
        confidence = min(0.8, 0.45 + (0.12 - flatness) * 2.5)
        return SectionGuess("music", confidence, reasons)

    reasons.append(
        f"heuristic: no decisive pattern (flatness {flatness:.2f}, "
        f"level {rms_db:.1f} dBFS, modulation {modulation:.2f})"
    )
    return SectionGuess("unknown", 0.25, reasons)
