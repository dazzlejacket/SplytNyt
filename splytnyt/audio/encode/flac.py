"""FLAC export.

FLAC is written through libsndfile, which produces the STREAMINFO block and
the audio frames. Tags are added afterwards by
:func:`splytnyt.audio.tags.flac.write_vorbis_comments`; this module only
concerns itself with the audio.

The source bit depth is preserved whenever libsndfile's FLAC encoder can
express it. Its FLAC subtypes are 8, 16 and 24 bit integer only, so a 32 bit
or floating point source is written as 24 bit and the conversion is reported
in :attr:`WriteResult.notes` for the UI to show.
"""

from __future__ import annotations

import os

import soundfile as sf

from . import (
    AudioPart,
    GainApplier,
    WriteResult,
    atomic_output,
    describe_parts,
    source_subtype as source_subtype_of,
    ensure_not_a_source,
    file_size,
    iter_part_blocks,
)

#: Source subtype -> (FLAC subtype, note). The note is empty when the source
#: subtype can be preserved exactly.
_FLAC_SUBTYPES: dict[str, tuple[str, str]] = {
    "PCM_S8": ("PCM_S8", ""),
    "PCM_16": ("PCM_16", ""),
    "PCM_24": ("PCM_24", ""),
    "PCM_32": (
        "PCM_24",
        "Source is 32-bit PCM; written as 24-bit PCM (libsndfile has no "
        "32-bit FLAC subtype).",
    ),
    "FLOAT": (
        "PCM_24",
        "Source is 32-bit float; written as 24-bit PCM (libsndfile's FLAC "
        "encoder has no float subtype).",
    ),
    "DOUBLE": (
        "PCM_24",
        "Source is 64-bit float; written as 24-bit PCM (libsndfile's FLAC "
        "encoder has no float subtype).",
    ),
}


def write_flac(
    path: str,
    parts: list[AudioPart],
    gain: float = 1.0,
    compression: int = 5,
    processor=None,
) -> WriteResult:
    """Write the given parts of the sources to a FLAC file.

    Args:
        path: Target file. Written atomically via ``<path>.partial``.
        parts: ``(source, start_frame, end_frame)`` chunks, concatenated in
            order without gaps or overlaps.
        gain: One linear gain applied identically to every channel.
        compression: Requested FLAC compression level (0-8).
        processor: Optional callable applied to every block after the gain.
            It may carry state across blocks.

    Returns:
        A :class:`WriteResult` describing what was written.
    """
    parts = list(parts)
    ensure_not_a_source(path, parts)
    sample_rate, channels = describe_parts(parts)

    source_subtype = source_subtype_of(parts)
    subtype, note = _FLAC_SUBTYPES.get(
        source_subtype,
        (
            "PCM_24",
            f"Source subtype {source_subtype} has no FLAC equivalent; written "
            "as 24-bit PCM.",
        ),
    )

    notes: list[str] = []
    if note:
        notes.append(note)
    notes.append(
        f"FLAC compression level {int(compression)} was requested; libsndfile's "
        "default compression level was used (python-soundfile cannot set "
        "SFC_SET_COMPRESSION_LEVEL)."
    )

    applier = GainApplier(gain, processor)
    written = 0
    with atomic_output(path, with_handle=False) as (temp, _handle):
        with sf.SoundFile(
            temp,
            "w",
            samplerate=sample_rate,
            channels=channels,
            subtype=subtype,
            format="FLAC",
        ) as out:
            for block in iter_part_blocks(parts):
                out.write(applier.apply(block))
                written += block.shape[0]
    notes.extend(applier.notes())

    return WriteResult(
        path=os.path.abspath(path),
        frames_written=written,
        sample_rate=sample_rate,
        channels=channels,
        bytes_written=file_size(path),
        notes=notes,
    )
