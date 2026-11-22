"""Build a synthetic verification project for the packaged release check.

Creates a synthetic recording (no copyrighted material), imports it through the
real import path, splits it at known boundaries and saves a .splytnyt project
that the frozen application can open.

Usage:
    python make_verify_project.py <output-dir>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import soundfile as sf

RATE = 44100


def chord(frequency: float, seconds: float, amplitude: float = 0.32) -> np.ndarray:
    frames = max(1, int(seconds * RATE))
    time_axis = np.arange(frames) / RATE
    wave = np.zeros(frames, dtype=np.float64)
    for index, multiple in enumerate((1.0, 1.5, 2.0, 3.0)):
        wave += (1.0 / (index + 1.0)) * np.sin(2 * np.pi * frequency * multiple * time_axis)
    wave /= sum(1.0 / (i + 1.0) for i in range(4))
    attack = int(0.006 * RATE)
    release = int(0.05 * RATE)
    envelope = np.ones(frames)
    envelope[:attack] = np.linspace(0.0, 1.0, attack)
    envelope[-release:] = np.linspace(1.0, 0.0, release)
    return (wave * envelope * amplitude).astype(np.float32)


def main(argv: list[str]) -> int:
    out_dir = os.path.abspath(argv[1] if len(argv) > 1 else ".")
    os.makedirs(out_dir, exist_ok=True)

    # Five songs with ordinary gaps between them.
    notes = [196.00, 246.94, 293.66, 329.63, 392.00]
    gap_seconds = 1.5
    song_seconds = 4.0

    pieces = []
    boundaries = []
    cursor = 0
    for index, frequency in enumerate(notes):
        song = chord(frequency, song_seconds)
        pieces.append(song)
        cursor += song.size
        if index < len(notes) - 1:
            gap = np.zeros(int(gap_seconds * RATE), dtype=np.float32)
            pieces.append(gap)
            cursor += gap.size
            boundaries.append(cursor)  # the next song starts here

    signal = np.concatenate(pieces)
    stereo = np.stack([signal, signal * 0.9], axis=1)

    audio_path = os.path.join(out_dir, "verification-recording.wav")
    sf.write(audio_path, stereo, RATE, subtype="PCM_16")
    print(f"recording: {audio_path}  ({len(signal) / RATE:.1f}s, {len(signal)} frames)")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from splytnyt.app.session import Session

    session = Session()
    session.new_project("Release verification")
    session.import_files([audio_path])
    rate = session.project.sample_rate()

    for frame in boundaries:
        session.split_at(int(frame * rate / RATE))

    tracks = session.sorted_tracks()
    print(f"tracks after splitting: {len(tracks)}")
    for index, track in enumerate(tracks, start=1):
        session.set_track_title(track.track_id, f"Synthetic Song {index}")

    project_path = os.path.join(out_dir, "verification-project.splytnyt")
    saved = session.save_project(project_path)
    print(f"project: {saved}")

    tracks = session.sorted_tracks()
    print(f"final track count: {len(tracks)}")
    for track in tracks:
        print(f"  {track.title:20s} {track.start_frame:>9d} -> {track.end_frame:>9d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
