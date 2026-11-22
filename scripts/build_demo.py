"""Build a demo recording for the screenshots.

The recording is synthetic: four songs separated by applause, a quiet
passage inside the second song, and a soft intro before the third. It is
made to look like a concert recording so the interface can be reviewed
without a real one being needed.

It is a *development* helper. It is never loaded into a user's project.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import materials as M  # noqa: E402

RATE = M.RATE
HERE = os.path.dirname(os.path.abspath(__file__))


def build_recording(path: str) -> None:
    """Four songs with applause between them and a soft intro."""
    rng = np.random.default_rng(11)

    def song(frequency: float, seconds: float, amplitude: float, swell: float = 0.25) -> np.ndarray:
        """A chord that breathes a little, the way a played instrument does."""
        wave = M.chord(frequency, seconds, RATE, amplitude=amplitude)
        time_axis = np.arange(wave.size) / RATE
        envelope = 1.0 - swell + swell * np.sin(2 * np.pi * 0.11 * time_axis + 1.0)
        return (wave * envelope).astype(np.float32)

    # Song one: a full, loud chord.
    first = song(196.0, 16.0, 0.32)

    # Applause, then song two, with a quiet passage inside it.
    applause_one = M.applause(3.5, RATE, amplitude=0.07, seed=3)
    second = song(246.94, 7.0, 0.32)
    quiet = M.chord(246.94, 0.8, RATE, amplitude=0.006)
    second_tail = song(246.94, 9.0, 0.32)

    # Applause, then a song that starts quietly and comes up.
    applause_two = M.applause(3.5, RATE, amplitude=0.07, seed=5)
    intro = M.chord(164.81, 3.0, RATE, amplitude=0.03)
    third = song(164.81, 13.0, 0.32)

    # Applause, then the last song.
    applause_three = M.applause(2.5, RATE, amplitude=0.06, seed=8)
    fourth = song(293.66, 15.0, 0.32)

    parts = [
        first, applause_one, second, quiet, second_tail,
        applause_two, intro, third, applause_three, fourth,
    ]
    signal = np.concatenate(parts).astype(np.float32)

    # A little room noise everywhere.
    noise = rng.normal(0.0, 0.0012, size=signal.size).astype(np.float32)
    signal = signal + noise

    stereo = np.stack([signal, signal * 0.97], axis=1)
    sf.write(path, stereo, RATE, subtype="PCM_16")
    print(f"wrote {path}: {stereo.shape[0]} frames, {stereo.shape[0] / RATE:.1f} s")


def main() -> int:
    os.makedirs(os.path.join(HERE, "demo"), exist_ok=True)
    recording = os.path.join(HERE, "demo", "demo_concert.flac")
    build_recording(recording)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
