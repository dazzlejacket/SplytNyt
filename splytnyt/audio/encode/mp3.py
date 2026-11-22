"""MP3 export through LAME.

MP3 is re-encoded, never stream-copied, so a cut is never sample accurate.
The writer therefore reports what LAME actually did:

* the **encoder delay** (LAME's documented 576 samples for MPEG Layer III -
  ``lameenc`` exposes no value, so the constant is used),
* the **padding residual**, i.e. ``encoded_frames * samples_per_frame -
  source_frames``, which is how much longer the decoded stream is than the
  material that was fed in,
* the fact that the file carries no LAME gapless tag, so players may trim
  the 1105-sample offset (576 encoder delay + 529 decoder delay) differently.

None of this is hidden from the user: the numbers end up in
:attr:`WriteResult.notes` and in the export log.
"""

from __future__ import annotations

import os

import lameenc
import numpy as np

from . import (
    AudioPart,
    GainApplier,
    WriteResult,
    atomic_output,
    describe_parts,
    ensure_not_a_source,
    file_size,
    iter_part_blocks,
)

#: Frames handed to LAME per call (1152 samples per MPEG-1 frame).
_MP3_BLOCK_FRAMES = 1152 * 64

#: LAME's encoder delay for MPEG Layer III, in samples. lameenc 1.9 does not
#: expose the value, so the documented constant is used and reported.
LAME_ENCODER_DELAY = 576

#: Delay introduced by the MP3 decoder itself, in samples.
MP3_DECODER_DELAY = 529

#: lameenc's VBR mode: LAME's default VBR ("vbr_mtrh"), the mode behind
#: ``lame -V``.
_LAME_VBR_MODE = 4

# Bit rate tables (kbps) indexed by the bit rate index of a frame header.
_BITRATES_MPEG1 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_BITRATES_MPEG2 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)

# Sample rates indexed by the MPEG version id, then by the rate index.
_SAMPLE_RATES = {
    3: (44100, 48000, 32000),  # MPEG 1
    2: (22050, 24000, 16000),  # MPEG 2
    0: (11025, 12000, 8000),  # MPEG 2.5
}


def write_mp3(
    path: str,
    parts: list[AudioPart],
    gain: float = 1.0,
    bitrate_kbps: int = 192,
    vbr: bool = False,
    processor=None,
    quality: int = 2,
) -> WriteResult:
    """Write the given parts of the sources to an MP3 file.

    Args:
        path: Target file. Written atomically via ``<path>.partial``.
        parts: ``(source, start_frame, end_frame)`` chunks, concatenated in
            order without gaps or overlaps.
        gain: One linear gain applied identically to every channel.
        bitrate_kbps: Constant bit rate for CBR mode.
        vbr: Use variable bit rate instead of CBR.
        quality: LAME quality (0 = best, 9 = worst); for VBR this is LAME's
            VBR quality scale.

    Returns:
        A :class:`WriteResult` describing what was written.
    """
    parts = list(parts)
    ensure_not_a_source(path, parts)
    sample_rate, channels = describe_parts(parts)

    encoder = lameenc.Encoder()
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(channels)
    if vbr:
        encoder.set_vbr(_LAME_VBR_MODE)
        encoder.set_vbr_quality(int(quality))
    else:
        encoder.set_bit_rate(int(bitrate_kbps))
        encoder.set_quality(int(quality))

    applier = GainApplier(gain, processor)
    written = 0
    with atomic_output(path) as (_temp, handle):
        if handle is None:  # pragma: no cover - atomic_output always opens one
            raise RuntimeError("No output handle for the MP3 writer")
        for block in iter_part_blocks(parts, block_frames=_MP3_BLOCK_FRAMES):
            encoded = encoder.encode(_to_int16(applier.apply(block)))
            if encoded:
                handle.write(encoded)
            written += block.shape[0]
        tail = encoder.flush()
        if tail:
            handle.write(tail)

    frames, samples_per_frame, _encoded_rate = _scan_mpeg_frames(path)
    encoded_samples = frames * samples_per_frame
    residual = encoded_samples - written

    notes: list[str] = []
    if vbr:
        notes.append(
            f"MP3 written with VBR (LAME VBR quality {int(quality)}); the "
            "average bit rate is whatever the material needed."
        )
    else:
        notes.append(
            f"MP3 written with CBR {int(bitrate_kbps)} kbps (LAME quality "
            f"{int(quality)})."
        )
    notes.append(
        f"MP3 encoder delay: {LAME_ENCODER_DELAY} samples (LAME's documented "
        "MPEG Layer III delay; lameenc 1.9 does not expose the value)."
    )
    notes.append(
        f"MP3 padding residual: {residual} samples ({frames} encoded frames x "
        f"{samples_per_frame} samples - {written} source frames); the decoded "
        f"stream is {residual} samples "
        f"({residual / float(sample_rate) * 1000.0:.1f} ms) longer than the "
        "source."
    )
    notes.append(
        "MP3 cuts are not sample-accurate: playback adds a "
        f"{LAME_ENCODER_DELAY + MP3_DECODER_DELAY}-sample gapless offset "
        f"({LAME_ENCODER_DELAY} encoder delay + {MP3_DECODER_DELAY} decoder "
        "delay) and the file carries no LAME gapless tag, so players may trim "
        "the start differently."
    )
    notes.extend(applier.notes())

    return WriteResult(
        path=os.path.abspath(path),
        frames_written=written,
        sample_rate=sample_rate,
        channels=channels,
        encoder_delay=LAME_ENCODER_DELAY,
        encoder_padding=residual,
        bytes_written=file_size(path),
        notes=notes,
    )


def _to_int16(block: np.ndarray) -> bytes:
    """Convert float32 samples to interleaved little-endian 16-bit PCM.

    The scale factor matches libsndfile's own float to integer conversion so
    that the lossy and the lossless paths agree about what a sample value
    means. Values outside the integer range are clipped here because the
    format itself cannot hold them.
    """
    scaled = np.round(block * np.float32(32768.0))
    np.clip(scaled, -32768.0, 32767.0, out=scaled)
    return scaled.astype("<i2").tobytes()


def _scan_mpeg_frames(path: str) -> tuple[int, int, int]:
    """Return ``(frame_count, samples_per_frame, sample_rate)`` of an MP3.

    The file is walked frame by frame from its first header, so no more than
    a handful of bytes are held in memory at a time.
    """
    size = file_size(path)
    frames = 0
    samples_per_frame = 1152
    sample_rate = 0
    with open(path, "rb") as handle:
        position = 0
        while position + 4 <= size:
            handle.seek(position)
            header = handle.read(4)
            if len(header) < 4:
                break
            if header[0] != 0xFF or (header[1] & 0xE0) != 0xE0:
                break
            version = (header[1] >> 3) & 0x03
            layer = (header[1] >> 1) & 0x03
            bitrate_index = (header[2] >> 4) & 0x0F
            rate_index = (header[2] >> 2) & 0x03
            padding = (header[2] >> 1) & 0x01
            if layer != 1 or bitrate_index in (0, 15) or rate_index == 3:
                break
            if version not in _SAMPLE_RATES:
                break
            table = _BITRATES_MPEG1 if version == 3 else _BITRATES_MPEG2
            bitrate = table[bitrate_index] * 1000
            sample_rate = _SAMPLE_RATES[version][rate_index]
            if version == 3:
                length = (144 * bitrate) // sample_rate + padding
                samples_per_frame = 1152
            else:
                length = (72 * bitrate) // sample_rate + padding
                samples_per_frame = 576
            if length <= 4:
                break
            frames += 1
            position += length
    return frames, samples_per_frame, sample_rate
