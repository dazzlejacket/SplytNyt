"""CUE sheet writer.

The CUE sheet describes the tracks inside one continuous audio file. It
quantises the frame positions to ``MM:SS:FF`` at 75 frames per second, which
is what the format can express; the quantisation error is at most one CUE
frame (1/75 s) and is reported by the export layer, never hidden here.

Every track gets a sequential track number, its title and performer (when
known) and an ``INDEX 01`` pointing at its start. ``INDEX 00`` points at the
end of the previous track, so the gap between two tracks stays visible in the
sheet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ...core.units import frames_to_tc
from . import atomic_output


@dataclass
class CueTrack:
    """One track inside a CUE sheet.

    ``start_frame`` and ``end_frame`` are frame indices in the continuous
    audio file. ``track_number`` is advisory: the writer always numbers the
    tracks sequentially so the sheet stays valid.
    """

    title: str
    start_frame: int
    end_frame: int
    artist: str = ""
    album: str = ""
    track_number: int = 0


def write_cue(
    path: str,
    audio_filename: str,
    tracks: list[CueTrack],
    sample_rate: int,
    album: str = "",
    artist: str = "",
    genre: str = "",
    date: str = "",
    comment: str = "",
) -> str:
    """Write a CUE sheet and return the path that was written.

    Args:
        path: Target ``.cue`` file, written atomically.
        audio_filename: Name of the audio file the sheet refers to; it is
            written on the same line as the ``FILE`` directive.
        tracks: The tracks in order.
        sample_rate: Sample rate of the audio file, used for the timecodes.
        album: Album title (written as the sheet's ``TITLE``).
        artist: Album artist (written as the sheet's ``PERFORMER``).
        genre: Genre (written as ``REM GENRE``).
        date: Date (written as ``REM DATE``).
        comment: Free text comment (written as ``REM COMMENT``).
    """
    lines: list[str] = []
    if genre:
        lines.append(f"REM GENRE {_clean(genre)}")
    if date:
        lines.append(f"REM DATE {_clean(date)}")
    if comment:
        lines.append(f"REM COMMENT {_clean(comment)}")
    if artist:
        lines.append(f'PERFORMER "{_clean(artist)}"')
    if album:
        lines.append(f'TITLE "{_clean(album)}"')
    lines.append(f'FILE "{_clean(audio_filename)}" WAVE')

    previous_end = 0
    for number, track in enumerate(tracks, start=1):
        lines.append(f"  TRACK {number:02d} AUDIO")
        if track.title:
            lines.append(f'    TITLE "{_clean(track.title)}"')
        performer = track.artist or artist
        if performer:
            lines.append(f'    PERFORMER "{_clean(performer)}"')
        lines.append(f"    INDEX 00 {frames_to_tc(previous_end, sample_rate)}")
        lines.append(f"    INDEX 01 {frames_to_tc(track.start_frame, sample_rate)}")
        previous_end = max(previous_end, int(track.end_frame))

    text = "\n".join(lines) + "\n"
    with atomic_output(path) as (_temp, handle):
        if handle is None:  # pragma: no cover - atomic_output always opens one
            raise RuntimeError("No output handle for the CUE writer")
        handle.write(text.encode("utf-8"))
    return os.path.abspath(path)


def write_split_cue(
    path: str,
    files: list[tuple[str, str]],
    album: str = "",
    artist: str = "",
    genre: str = "",
    date: str = "",
    comment: str = "",
) -> str:
    """Write a CUE sheet for separate track files.

    ``files`` is a list of ``(filename, title)`` pairs. Every file gets its
    own ``FILE`` directive and one ``TRACK`` entry whose ``INDEX 01`` points
    at the start of that file, which is the standard "split cue" layout.
    """
    lines: list[str] = []
    if genre:
        lines.append(f"REM GENRE {_clean(genre)}")
    if date:
        lines.append(f"REM DATE {_clean(date)}")
    if comment:
        lines.append(f"REM COMMENT {_clean(comment)}")
    if artist:
        lines.append(f'PERFORMER "{_clean(artist)}"')
    if album:
        lines.append(f'TITLE "{_clean(album)}"')

    for number, entry in enumerate(files, start=1):
        filename = entry[0]
        title = entry[1]
        track_artist = entry[2] if len(entry) >= 3 and entry[2] else artist
        lines.append(f'FILE "{_clean(filename)}" WAVE')
        lines.append(f"  TRACK {number:02d} AUDIO")
        if title:
            lines.append(f'    TITLE "{_clean(title)}"')
        if track_artist:
            lines.append(f'    PERFORMER "{_clean(track_artist)}"')
        lines.append("    INDEX 01 00:00:00")

    text = "\n".join(lines) + "\n"
    with atomic_output(path) as (_temp, handle):
        if handle is None:  # pragma: no cover - atomic_output always opens one
            raise RuntimeError("No output handle for the CUE writer")
        handle.write(text.encode("utf-8"))
    return os.path.abspath(path)


def _clean(text: str) -> str:
    """Make a value safe for a quoted CUE field."""
    return str(text).replace('"', "'").replace("\r", " ").replace("\n", " ").strip()
