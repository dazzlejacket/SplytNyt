"""Unit conversion helpers.

The engine stores every position as an ``int`` frame index. Seconds exist only
as a derived view, produced at the edges (UI, reports, CUE sheets).
"""

from __future__ import annotations


def frames_to_seconds(frames: int, sample_rate: int) -> float:
    """Convert a frame count to seconds."""
    if sample_rate <= 0:
        return 0.0
    return frames / float(sample_rate)


def seconds_to_frames(seconds: float, sample_rate: int) -> int:
    """Convert seconds to a frame count, rounding to nearest."""
    if sample_rate <= 0:
        return 0
    return int(round(seconds * sample_rate))


def frames_to_tc(frames: int, sample_rate: int, fps: int = 75) -> str:
    """Format a frame count as ``MM:SS:FF`` timecode at ``fps`` frames/sec.

    Used by the CUE writer. The frame component is truncated toward zero, so
    the result never overshoots the true position; the residual error is
    reported separately by the exporter.
    """
    if sample_rate <= 0 or fps <= 0:
        return "00:00:00"
    total_seconds = frames / float(sample_rate)
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    remainder = total_seconds - minutes * 60 - seconds
    frame_part = int(remainder * fps)
    if frame_part >= fps:
        frame_part = fps - 1
    return f"{minutes:02d}:{seconds:02d}:{frame_part:02d}"


def frames_to_timestamp(frames: int, sample_rate: int) -> str:
    """Human readable ``H:MM:SS.mmm`` / ``M:SS.mmm`` timestamp."""
    if sample_rate <= 0:
        return "0:00.000"
    total = frames / float(sample_rate)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    seconds = total % 60
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:06.3f}"
    return f"{minutes}:{seconds:06.3f}"


def parse_timestamp(text: str) -> float:
    """Parse ``h:mm:ss.mmm``, ``mm:ss.mmm`` or ``ss.mmm`` into seconds.

    Accepts ``.`` or ``,`` as the decimal separator and tolerates surrounding
    whitespace. Raises :class:`ValueError` for anything else.
    """
    raw = text.strip()
    if not raw:
        raise ValueError("empty timestamp")
    raw = raw.replace(",", ".")
    negative = raw.startswith("-")
    if negative:
        raw = raw[1:]
    parts = raw.split(":")
    if len(parts) > 3:
        raise ValueError(f"bad timestamp: {text!r}")
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"bad timestamp: {text!r}") from exc
    seconds = 0.0
    for value in values:
        seconds = seconds * 60.0 + value
    return -seconds if negative else seconds


def parse_db(text: str) -> float:
    """Parse a decibel string such as ``"-3.5 dB"`` or ``"+2"``."""
    raw = text.strip().lower().replace("db", "").strip()
    if raw in ("", "+"):
        raw = "0"
    return float(raw)


def db_to_linear(db: float) -> float:
    """Convert decibels to a linear amplitude multiplier."""
    if db <= -120.0:
        return 0.0
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert a linear amplitude multiplier to decibels."""
    if linear <= 0.0:
        return -120.0
    return 20.0 * float(_log10(linear))


def _log10(value: float) -> float:
    import math

    return math.log10(value)


def format_db(db: float) -> str:
    """Format decibels for display."""
    if db <= -120.0:
        return "-inf dB"
    return f"{db:+.1f} dB"


def format_confidence(value: float) -> str:
    """Format a 0..1 confidence for display."""
    return f"{max(0.0, min(1.0, value)) * 100:.0f}%"


def safe_filename(text: str, fallback: str = "untitled", max_length: int = 120) -> str:
    """Sanitise a string for use as a filename on Windows, macOS and Linux.

    Removes characters that are illegal on any of the three platforms plus
    control characters, collapses whitespace, trims trailing dots and spaces
    (illegal on Windows), and truncates on a path-length-safe boundary.
    """
    cleaned = []
    for char in text:
        if char in '<>:"/\\|?*':
            cleaned.append("_")
        elif ord(char) < 32 or ord(char) == 127:
            cleaned.append("_")
        else:
            cleaned.append(char)
    result = "".join(cleaned).strip()
    result = result.rstrip(". ")
    result = " ".join(result.split())
    if not result:
        result = fallback
    if len(result) > max_length:
        result = result[:max_length].rstrip(". ")
    return result or fallback
