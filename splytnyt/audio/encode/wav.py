"""WAV export.

WAV is written through libsndfile. The RIFF INFO and ``id3 `` chunks are
added afterwards by :func:`splytnyt.audio.tags.riff.write_riff_info`; this
module only concerns itself with the audio.

The sample rate and the channel layout of the source are preserved. The
subtype is chosen by the caller; when it is less precise than the source,
the conversion is reported in :attr:`WriteResult.notes`.
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

#: Subtypes libsndfile can write into a WAV container.
_WAV_SUBTYPES = frozenset(sf.available_subtypes("WAV"))

#: Rough precision ranking, used to report a lossy subtype choice.
_PRECISION = {
    "PCM_S8": 1,
    "PCM_U8": 1,
    "PCM_16": 2,
    "PCM_24": 3,
    "PCM_32": 4,
    "FLOAT": 5,
    "DOUBLE": 6,
}


def write_wav(
    path: str,
    parts: list[AudioPart],
    gain: float = 1.0,
    subtype: str = "PCM_16",
    processor=None,
) -> WriteResult:
    """Write the given parts of the sources to a WAV file.

    Args:
        path: Target file. Written atomically via ``<path>.partial``.
        parts: ``(source, start_frame, end_frame)`` chunks, concatenated in
            order without gaps or overlaps.
        gain: One linear gain applied identically to every channel.
        subtype: libsndfile WAV subtype, e.g. ``PCM_16`` or ``PCM_24``.

    Returns:
        A :class:`WriteResult` describing what was written.
    """
    parts = list(parts)
    ensure_not_a_source(path, parts)
    sample_rate, channels = describe_parts(parts)

    subtype = str(subtype).upper()
    if subtype not in _WAV_SUBTYPES:
        raise ValueError(
            f"Unsupported WAV subtype {subtype!r}; libsndfile offers "
            f"{', '.join(sorted(_WAV_SUBTYPES))}."
        )

    notes: list[str] = []
    source_subtype = source_subtype_of(parts)
    source_precision = _PRECISION.get(source_subtype)
    target_precision = _PRECISION.get(subtype)
    if (
        source_precision is not None
        and target_precision is not None
        and target_precision < source_precision
    ):
        notes.append(
            f"Source is {source_subtype}; written as {subtype} (lower "
            "precision, the extra bits are quantised away)."
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
            format="WAV",
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
