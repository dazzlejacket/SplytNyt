"""Synthetic recordings that stand in for real ones.

These are *not* real recordings. They are built to contain the situations the
product has to survive - a long gap between songs, a soft intro, applause
before a song, applause over running music, a quiet passage inside a song, a
reverb tail, tape hiss and a seamless transition - so that the detector's
behaviour can be checked against known ground truth.

Every builder returns the audio together with the annotated positions, so a
test can measure the distance between a proposed cut and the real song start
instead of the distance from the middle of the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RATE = 44100


@dataclass
class Material:
    """A synthetic recording and what is known about it."""

    signal: np.ndarray
    sample_rate: int = RATE
    #: The frame at which each song's music actually starts.
    song_starts: list[int] = field(default_factory=list)
    #: The frame at which each song's music actually stops.
    song_ends: list[int] = field(default_factory=list)
    #: The frame at which each song's title text should appear.
    song_titles: list[str] = field(default_factory=list)
    #: The quiet stretches between the songs, as ``(start, end)`` frames.
    gaps: list[tuple[int, int]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.signal.size / float(self.sample_rate)


def chord(frequency: float, seconds: float, rate: int = RATE, amplitude: float = 0.35,
          harmonics: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)) -> np.ndarray:
    """A sustained chord with an attack and a release."""
    frames = max(1, int(seconds * rate))
    time_axis = np.arange(frames) / rate
    wave = np.zeros(frames, dtype=np.float64)
    for index, multiple in enumerate(harmonics):
        weight = 1.0 / (index + 1.0)
        wave += weight * np.sin(2 * np.pi * frequency * multiple * time_axis)
    wave /= sum(1.0 / (index + 1.0) for index in range(len(harmonics)))
    # A short attack and release so the start of the note is a real onset.
    attack = max(1, int(0.006 * rate))
    release = max(1, int(0.05 * rate))
    envelope = np.ones(frames)
    envelope[:attack] = np.linspace(0.0, 1.0, attack)
    if release < frames:
        envelope[-release:] = np.linspace(1.0, 0.0, release)
    return (wave * envelope * amplitude).astype(np.float32)


def applause(seconds: float, rate: int = RATE, amplitude: float = 0.09,
             decay: bool = True, seed: int = 5) -> np.ndarray:
    """A burst of audience noise."""
    rng = np.random.default_rng(seed)
    frames = max(1, int(seconds * rate))
    noise = rng.normal(0.0, amplitude, size=frames)
    if decay:
        noise *= np.linspace(1.0, 0.02, frames)
    return noise.astype(np.float32)


def tape_hiss(seconds: float, rate: int = RATE, level: float = 0.0012,
              seed: int = 9) -> np.ndarray:
    """A constant noise floor, as a tape transfer has."""
    rng = np.random.default_rng(seed)
    frames = max(1, int(seconds * rate))
    return rng.normal(0.0, level, size=frames).astype(np.float32)


def reverb_tail(seconds: float, rate: int = RATE, decay_db: float = 24.0,
                seed: int = 13) -> np.ndarray:
    """A decaying noise tail, as a hall leaves behind a last chord."""
    rng = np.random.default_rng(seed)
    frames = max(1, int(seconds * rate))
    noise = rng.normal(0.0, 0.06, size=frames)
    curve = np.linspace(0.0, -decay_db, frames)
    noise *= 10.0 ** (curve / 20.0)
    return noise.astype(np.float32)


def _join(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)


def two_songs_with_a_long_gap(
    gap_seconds: float = 6.0,
    song_seconds: float = 5.0,
    rate: int = RATE,
) -> Material:
    """Song, a long quiet stretch, song. The classic bad-cut case."""
    first = chord(196.0, song_seconds, rate)
    gap = np.zeros(int(gap_seconds * rate), dtype=np.float32)
    second = chord(246.94, song_seconds, rate)

    signal = _join([first, gap, second])
    gap_start = first.size
    gap_end = gap_start + gap.size
    return Material(
        signal=signal,
        sample_rate=rate,
        song_starts=[0, gap_end],
        song_ends=[first.size, signal.size],
        gaps=[(gap_start, gap_end)],
    )


def soft_intro_before_loud_music(rate: int = RATE) -> Material:
    """A song that starts quietly and only later comes up to full volume."""
    intro = chord(220.0, 2.5, rate, amplitude=0.028)
    loud = chord(220.0, 4.0, rate, amplitude=0.35)
    first = chord(146.83, 4.0, rate)

    signal = _join([first, np.zeros(int(1.2 * rate), dtype=np.float32), intro, loud])
    intro_start = first.size + int(1.2 * rate)
    return Material(
        signal=signal,
        sample_rate=rate,
        song_starts=[0, intro_start],
        song_ends=[first.size, signal.size],
        gaps=[(first.size, intro_start)],
    )


def applause_before_the_next_song(rate: int = RATE) -> Material:
    """Applause fills the gap, then the next song starts."""
    first = chord(196.0, 5.0, rate)
    noise = applause(4.0, rate)
    second = chord(246.94, 5.0, rate)

    signal = _join([first, noise, second])
    gap_start = first.size
    song_start = first.size + noise.size
    return Material(
        signal=signal,
        sample_rate=rate,
        song_starts=[0, song_start],
        song_ends=[first.size, signal.size],
        gaps=[(gap_start, song_start)],
    )


def applause_over_running_music(rate: int = RATE) -> Material:
    """Applause on top of music that never stops."""
    music = chord(196.0, 10.0, rate)
    noise = applause(2.0, rate, amplitude=0.16, decay=False)
    start = int(4.0 * rate)
    music[start: start + noise.size] += noise
    return Material(
        signal=music.astype(np.float32),
        sample_rate=rate,
        song_starts=[0],
        song_ends=[music.size],
    )


def quiet_passage_inside_a_song(rate: int = RATE) -> Material:
    """A song with a long quiet passage in the middle of it."""
    first = chord(196.0, 4.0, rate)
    quiet = chord(196.0, 1.6, rate, amplitude=0.004)
    second = chord(196.0, 4.0, rate)
    signal = _join([first, quiet, second])
    return Material(
        signal=signal,
        sample_rate=rate,
        song_starts=[0],
        song_ends=[signal.size],
    )


def reverb_tail_between_songs(rate: int = RATE) -> Material:
    """The hall keeps ringing after the last chord of a song."""
    first = chord(196.0, 5.0, rate)
    tail = reverb_tail(1.2, rate)
    second = chord(246.94, 5.0, rate)
    signal = _join([first, tail, second])
    return Material(
        signal=signal,
        sample_rate=rate,
        song_starts=[0, first.size + tail.size],
        song_ends=[first.size, signal.size],
        gaps=[(first.size, first.size + tail.size)],
    )


def noisy_tape_gap(rate: int = RATE) -> Material:
    """A tape transfer: a constant hiss, and a gap that is only hiss."""
    hiss = tape_hiss(14.0, rate)
    first = chord(196.0, 5.0, rate) + hiss[: int(5.0 * rate)]
    quiet = hiss[int(5.0 * rate): int(9.0 * rate)]
    second = chord(246.94, 5.0, rate) + hiss[int(9.0 * rate):]
    signal = _join([first, quiet, second])
    return Material(
        signal=signal,
        sample_rate=rate,
        song_starts=[0, first.size + quiet.size],
        song_ends=[first.size, signal.size],
        gaps=[(first.size, first.size + quiet.size)],
    )


def seamless_transition(rate: int = RATE) -> Material:
    """Two songs with no gap at all: a medley."""
    first = chord(196.0, 5.0, rate)
    second = chord(329.63, 5.0, rate)
    signal = _join([first, second])
    return Material(
        signal=signal,
        sample_rate=rate,
        song_starts=[0, first.size],
        song_ends=[first.size, signal.size],
    )


def three_songs(rate: int = RATE) -> Material:
    """Three songs with ordinary gaps, for the export and alignment tests."""
    first = chord(196.0, 4.0, rate)
    gap_one = np.zeros(int(1.5 * rate), dtype=np.float32)
    second = chord(246.94, 4.0, rate)
    gap_two = np.zeros(int(1.5 * rate), dtype=np.float32)
    third = chord(293.66, 4.0, rate)
    signal = _join([first, gap_one, second, gap_two, third])
    second_start = first.size + gap_one.size
    third_start = second_start + second.size + gap_two.size
    return Material(
        signal=signal,
        sample_rate=rate,
        song_starts=[0, second_start, third_start],
        song_ends=[first.size, second_start + second.size, signal.size],
        gaps=[(first.size, second_start), (second_start + second.size, third_start)],
    )


def slow_attack_before_a_song(
    attack_seconds: float = 0.25, rate: int = RATE
) -> Material:
    """A song that fades in over a quarter of a second.

    The music starts at the *beginning* of the fade, not where it reaches
    full volume, so the cut has to be in front of the fade: a cut later
    would throw the start of the note away.
    """
    first = chord(196.0, 5.0, rate)
    gap = np.zeros(int(2.0 * rate), dtype=np.float32)

    body = chord(246.94, 5.0, rate)
    attack = int(attack_seconds * rate)
    ramp = np.ones(body.size, dtype=np.float32)
    ramp[:attack] = np.linspace(0.0, 1.0, attack)
    second = (body * ramp).astype(np.float32)

    signal = _join([first, gap, second])
    start = first.size + gap.size
    return Material(
        signal=signal,
        sample_rate=rate,
        song_starts=[0, start],
        song_ends=[first.size, signal.size],
        gaps=[(first.size, start)],
    )


def crowd_noise_before_the_next_song(rate: int = RATE) -> Material:
    """A gap filled with the murmur of a crowd, not with silence.

    The crowd dies away as the applause dies away, but it never reaches
    digital silence, so the quiet stretch is only a *quieter* stretch. The
    cut still belongs at the start of the next song, not in the murmur.
    """
    first = chord(196.0, 5.0, rate)
    murmur = crowd(3.0, rate)
    second = chord(246.94, 5.0, rate)

    signal = _join([first, murmur, second])
    start = first.size + murmur.size
    return Material(
        signal=signal,
        sample_rate=rate,
        song_starts=[0, start],
        song_ends=[first.size, signal.size],
        gaps=[(first.size, start)],
    )


def previous_song_tail_before_the_gap(rate: int = RATE) -> Material:
    """The song before the gap ends with a long, audible decay.

    The tail rings on into the quiet stretch, so a detector that simply
    looks for the first audible sound after the gap finds the tail instead
    of the next song. The cut still belongs at the next song.
    """
    first = chord(196.0, 4.0, rate)
    tail = decaying_tone(2.0, rate, start_level=0.09)
    gap = np.zeros(int(1.5 * rate), dtype=np.float32)
    second = chord(246.94, 5.0, rate)

    signal = _join([first, tail, gap, second])
    start = first.size + tail.size + gap.size
    return Material(
        signal=signal,
        sample_rate=rate,
        song_starts=[0, start],
        song_ends=[first.size + tail.size, signal.size],
        gaps=[(first.size, first.size + tail.size + gap.size)],
    )


def crowd(seconds: float, rate: int = RATE, amplitude: float = 0.02,
          seed: int = 17) -> np.ndarray:
    """The murmur of a crowd: noise that never quite stops."""
    rng = np.random.default_rng(seed)
    frames = max(1, int(seconds * rate))
    noise = rng.normal(0.0, amplitude, size=frames)
    # The crowd thins out over the stretch.
    noise *= np.linspace(1.0, 0.25, frames)
    return noise.astype(np.float32)


def decaying_tone(seconds: float, rate: int = RATE, frequency: float = 196.0,
                  start_level: float = 0.09) -> np.ndarray:
    """A tone that fades away: the tail of a cymbal or a ringing chord."""
    frames = max(1, int(seconds * rate))
    time_axis = np.arange(frames) / rate
    wave = np.sin(2 * np.pi * frequency * time_axis)
    wave += 0.4 * np.sin(2 * np.pi * frequency * 1.5 * time_axis)
    envelope = 10.0 ** (np.linspace(0.0, -30.0, frames) / 20.0)
    return (wave * envelope * start_level).astype(np.float32)
