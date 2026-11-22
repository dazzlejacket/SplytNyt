"""Metadata-assisted boundary detection and refinement.

Combines online/bundled metadata (MusicBrainz release tracklists and durations,
AcoustID/Chromaprint fingerprints, setlist.fm / MusicBrainz concert setlists,
LRCLIB reference lyrics, and theatrical cast references) with actual audio
analysis to suggest track boundaries and fill in track names.

Design rules:
* Never blindly cut at published timestamps: every metadata-derived boundary is
  refined against actual audio features (silence/dips, spectral/harmonic
  novelty, onset placement, or YAMNet audio-event transitions).
* Known albums account for rip offsets (leading/trailing silence), turntable
  speed differences, vinyl side breaks, and gapless transitions.
* Live concerts use setlists as supporting evidence, account for crowd noise,
  banter, extended performances, and encores, and never invent missing songs.
* DJ mixes use AcoustID/Chromaprint when available, detect overlapping
  crossfade transitions from macro-scale spectral/harmonic shifts, and NEVER
  invent song identities if fingerprinting fails or is unconfigured.
* Broadway uses cast tracklists, reference lyrics, transcription, and acoustic
  structure, preserves dialogue bridges inside numbers when appropriate, and
  never mistakes cast album durations for live stage boundaries.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..metadata.identify import IdentificationResult
from .placement import PlacementChoice, place_cut
from .segment import Suggestion
from .silence import SilenceRegion


@dataclass
class PlannedTrack:
    """One proposed track produced by metadata-assisted splitting."""

    start_frame: int
    end_frame: int
    title: str = ""
    artist: str = ""
    section_type: str = "music"
    boundary_confidence: float = 0.0
    identity_confidence: float = 0.0
    boundary_uncertain: bool = False
    boundary_placement: str = "metadata_refined"
    onset_frame: int = -1
    boundary_pre_roll_seconds: float = -1.0
    transition_start: int = -1
    transition_end: int = -1
    worklist_index: int = -1
    reasons: list[str] = field(default_factory=list)
    notes: str = ""
    recognized_artist: str = ""
    recognized_title: str = ""
    recognition_status: str = ""
    recognition_engines: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


@dataclass
class MetadataSplitOutcome:
    """Complete result of metadata-assisted splitting."""

    identification: IdentificationResult
    tracks: list[PlannedTrack] = field(default_factory=list)
    suggestions: list[Suggestion] = field(default_factory=list)
    activities: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    # Optional diagnostics (speed factor, side break, etc.)
    diagnostics: dict = field(default_factory=dict)


ProgressCallback = Optional[Callable[[float, str], None]]
CancelCallback = Optional[Callable[[], bool]]


# ----------------------------------------------------------------------
# Fast coarse feature extraction over a TimelineReader
# ----------------------------------------------------------------------

@dataclass
class CoarseFeatures:
    """0.5 s-hop acoustic summary of an entire recording."""

    hop_seconds: float
    sample_rate: int
    total_frames: int
    rms_db: np.ndarray        # (N,) float32
    bands: np.ndarray         # (N, B) float32 normalized log-band spectra
    chroma: np.ndarray        # (N, 12) float32 normalized pitch-class energy

    @property
    def n_frames(self) -> int:
        return int(self.rms_db.shape[0])

    def time_of(self, idx: int) -> float:
        return (idx + 0.5) * self.hop_seconds

    def index_of(self, seconds: float) -> int:
        if self.n_frames <= 0:
            return 0
        return int(np.clip(round(seconds / self.hop_seconds), 0, self.n_frames - 1))


def extract_coarse_features(
    reader,
    sample_rate: int,
    total_frames: int,
    hop_seconds: float = 0.5,
    chunk_seconds: float = 60.0,
    progress: ProgressCallback = None,
    cancel: CancelCallback = None,
) -> CoarseFeatures:
    """Stream the recording in chunks and compute 0.5 s-hop RMS, log-bands, and chroma."""
    sr = int(sample_rate or 44100)
    hop = max(1, int(round(hop_seconds * sr)))
    n_hops = max(1, int(math.ceil(total_frames / float(hop))))
    n_fft = 2048
    n_bands = 16

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    # 16 log-spaced frequency bands between 55 Hz and min(10000 Hz, sr/2)
    f_lo = 55.0
    f_hi = min(10000.0, sr * 0.48)
    edges = np.geomspace(f_lo, f_hi, n_bands + 1)
    band_masks = [
        (freqs >= edges[b]) & (freqs < edges[b + 1])
        for b in range(n_bands)
    ]

    # 12-bin chroma mapping for frequencies 65 Hz .. 4000 Hz
    chroma_bins = np.full(freqs.shape, -1, dtype=np.int32)
    valid = (freqs >= 65.0) & (freqs <= 4000.0)
    if np.any(valid):
        midi = 69.0 + 12.0 * np.log2(freqs[valid] / 440.0)
        chroma_bins[valid] = np.round(midi).astype(np.int32) % 12

    rms_db = np.full(n_hops, -90.0, dtype=np.float32)
    bands = np.zeros((n_hops, n_bands), dtype=np.float32)
    chroma = np.zeros((n_hops, 12), dtype=np.float32)

    window = np.hanning(n_fft).astype(np.float32)
    chunk_hops = max(1, int(round(chunk_seconds / hop_seconds)))
    chunk_frames = chunk_hops * hop

    pos = 0
    hop_idx = 0
    while pos < total_frames and hop_idx < n_hops:
        if cancel and cancel():
            break
        read_len = min(chunk_frames, total_frames - pos)
        data = reader.read(pos, read_len)
        if data.size == 0:
            break
        mono = data.mean(axis=1).astype(np.float32, copy=False)
        local_hops = min(n_hops - hop_idx, int(math.ceil(mono.size / float(hop))))

        for h in range(local_hops):
            s = h * hop
            e = min(mono.size, s + hop)
            seg = mono[s:e]
            if seg.size == 0:
                continue
            rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
            rms_db[hop_idx + h] = max(-90.0, 20.0 * math.log10(max(rms, 1e-9)))

            # Center FFT slice
            if seg.size >= n_fft:
                mid = (seg.size - n_fft) // 2
                frame = seg[mid:mid + n_fft] * window
            else:
                frame = np.zeros(n_fft, dtype=np.float32)
                frame[:seg.size] = seg * window[:seg.size]
            mag = np.abs(np.fft.rfft(frame))
            power = mag * mag

            vec = np.empty(n_bands, dtype=np.float32)
            for b, mask in enumerate(band_masks):
                vec[b] = math.log1p(float(power[mask].mean()) if np.any(mask) else 0.0)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                bands[hop_idx + h] = vec / norm

            cvec = np.zeros(12, dtype=np.float32)
            for pc in range(12):
                cmask = chroma_bins == pc
                if np.any(cmask):
                    cvec[pc] = float(mag[cmask].sum())
            cnorm = float(np.linalg.norm(cvec))
            if cnorm > 1e-8:
                chroma[hop_idx + h] = cvec / cnorm

        pos += read_len
        hop_idx += local_hops
        if progress:
            progress(min(1.0, pos / max(1, total_frames)), "Scanning audio structure...")

    return CoarseFeatures(
        hop_seconds=hop_seconds,
        sample_rate=sr,
        total_frames=total_frames,
        rms_db=rms_db,
        bands=bands,
        chroma=chroma,
    )


def _cosine_novelty(matrix: np.ndarray, half_win: int) -> np.ndarray:
    """Cosine distance between mean feature vector before vs after each frame."""
    n = matrix.shape[0]
    out = np.zeros(n, dtype=np.float32)
    if n < 2 * half_win + 2 or half_win < 1:
        return out
    csum = np.vstack([np.zeros((1, matrix.shape[1]), dtype=np.float64), np.cumsum(matrix, axis=0, dtype=np.float64)])
    idx = np.arange(half_win, n - half_win)
    before = (csum[idx] - csum[idx - half_win]) / float(half_win)
    after = (csum[idx + half_win] - csum[idx]) / float(half_win)
    num = np.sum(before * after, axis=1)
    den = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1) + 1e-9
    dist = np.clip(1.0 - num / den, 0.0, 2.0)
    out[idx] = dist.astype(np.float32)
    return out


def _energy_dip_curve(rms_db: np.ndarray, surround_half_win: int = 16) -> np.ndarray:
    """How many dB quieter each frame is than its surrounding ±window median."""
    n = rms_db.shape[0]
    dip = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - surround_half_win)
        hi = min(n, i + surround_half_win + 1)
        ref = float(np.percentile(rms_db[lo:hi], 75))
        dip[i] = max(0.0, ref - float(rms_db[i]))
    return dip


def _detect_content_span(features: CoarseFeatures) -> tuple[float, float]:
    """Estimate leading and trailing audio content times (accounting for rip offsets)."""
    rms = features.rms_db
    if rms.size == 0:
        return 0.0, 0.0
    p10 = float(np.percentile(rms, 10))
    p75 = float(np.percentile(rms, 75))
    thresh = min(p75 - 18.0, max(p10 + 6.0, -52.0))

    start_idx = 0
    max_lead = min(rms.size - 1, int(round(45.0 / features.hop_seconds)))
    for i in range(max_lead):
        if rms[i] > thresh:
            start_idx = max(0, i - 1)
            break

    end_idx = rms.size - 1
    min_tail = max(start_idx + 1, rms.size - int(round(45.0 / features.hop_seconds)))
    for i in range(rms.size - 1, min_tail, -1):
        if rms[i] > thresh:
            end_idx = min(rms.size - 1, i + 1)
            break

    t_start = start_idx * features.hop_seconds
    t_end = min(features.total_frames / float(features.sample_rate), (end_idx + 1) * features.hop_seconds)
    return t_start, max(t_start + 1.0, t_end)


def _refine_cut_frame_fine(
    reader,
    approx_seconds: float,
    sample_rate: int,
    total_frames: int,
    noise_floor_dbfs: float,
    placement: Optional[PlacementChoice] = None,
    has_dip: bool = False,
) -> tuple[int, int, float, int, int]:
    """Fine-refine a coarse cut timestamp using a ±1.5 s audio window.

    Returns ``(frame, onset_frame, pre_roll_s, trans_start, trans_end)``.
    """
    approx_frame = int(round(approx_seconds * sample_rate))
    approx_frame = max(1, min(total_frames - 1, approx_frame))

    win_s = 2.0 if has_dip else 1.2
    w_start = max(0, int(round((approx_seconds - win_s) * sample_rate)))
    w_end = min(total_frames, int(round((approx_seconds + win_s) * sample_rate)))
    if w_end - w_start < int(0.1 * sample_rate):
        return approx_frame, -1, -1.0, -1, -1

    data = reader.read(w_start, w_end - w_start)
    if data.size == 0:
        return approx_frame, -1, -1.0, -1, -1
    mono = data.mean(axis=1).astype(np.float32, copy=False)

    # 20 ms RMS envelope to locate the quietest valley and the attack right after it
    step = max(1, int(round(0.020 * sample_rate)))
    n_steps = max(1, mono.size // step)
    env = np.empty(n_steps, dtype=np.float32)
    for i in range(n_steps):
        chunk = mono[i * step:(i + 1) * step]
        env[i] = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))

    valley_idx = int(np.argmin(env_db))
    valley_db = float(env_db[valley_idx])
    peak_db = float(np.percentile(env_db, 90))

    if has_dip and (peak_db - valley_db) >= 8.0:
        # Build a SilenceRegion around the valley and delegate to place_cut
        dip_thresh = valley_db + min(6.0, (peak_db - valley_db) * 0.35)
        left = valley_idx
        while left > 0 and env_db[left - 1] <= dip_thresh:
            left -= 1
        right = valley_idx
        while right + 1 < n_steps and env_db[right + 1] <= dip_thresh:
            right += 1
        region = SilenceRegion(
            start_frame=w_start + left * step,
            end_frame=w_start + (right + 1) * step,
            level_dbfs=valley_db,
        )
        res = place_cut(
            mono,
            sample_rate,
            region,
            choice=placement or PlacementChoice(),
            floor_dbfs=noise_floor_dbfs,
            base_frame=w_start,
        )
        if not res.rejected and w_start < res.frame < w_end:
            pre_roll_ms = float(res.details.get("applied_pre_roll_ms", -1000.0))
            return (
                int(res.frame),
                int(res.onset_frame),
                pre_roll_ms / 1000.0 if pre_roll_ms >= 0 else -1.0,
                int(res.transition_start),
                int(res.transition_end),
            )

    # For gapless transitions, snap to the local minimum energy point inside ±0.8 s
    center = n_steps // 2
    rad = max(1, int(round(0.8 / 0.020)))
    lo = max(0, center - rad)
    hi = min(n_steps, center + rad + 1)
    best_local = lo + int(np.argmin(env_db[lo:hi]))
    refined = max(1, min(total_frames - 1, w_start + best_local * step))
    return refined, -1, -1.0, -1, -1


# ----------------------------------------------------------------------
# 1. Known Album Metadata-Assisted Splitting
# ----------------------------------------------------------------------

def split_album_with_metadata(
    reader,
    sample_rate: int,
    total_frames: int,
    identification: IdentificationResult,
    placement: Optional[PlacementChoice] = None,
    audio_path: str = "",
    allow_fingerprint: bool = False,
    use_shazamio: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    progress: ProgressCallback = None,
    cancel: CancelCallback = None,
) -> MetadataSplitOutcome:
    """Use MusicBrainz tracklist + cumulative durations as priors, refined by audio."""
    started = time.time()
    outcome = MetadataSplitOutcome(identification=identification)
    worklist = identification.worklist
    if not worklist or len(worklist.items) < 2:
        outcome.notes.append("No multi-track album worklist available; cannot perform album alignment.")
        outcome.elapsed_seconds = time.time() - started
        return outcome

    items = list(worklist.items)
    durs = [float(item.duration_hint or 0.0) for item in items]
    if sum(durs) <= 0.0:
        outcome.notes.append("Album worklist has no track durations; cannot estimate cumulative positions.")
        outcome.elapsed_seconds = time.time() - started
        return outcome

    # Fill any missing single-track duration with average
    known = [d for d in durs if d > 0]
    fallback_dur = sum(known) / len(known) if known else (total_frames / float(sample_rate)) / len(items)
    durs = [d if d > 0 else fallback_dur for d in durs]

    features = extract_coarse_features(
        reader,
        sample_rate,
        total_frames,
        hop_seconds=0.5,
        progress=lambda f, m: progress(0.05 + 0.45 * f, m) if progress else None,
        cancel=cancel,
    )
    total_seconds = total_frames / float(sample_rate)
    t_start, t_end = _detect_content_span(features)
    noise_floor = float(np.percentile(features.rms_db, 5))

    # Multi-scale spectral + harmonic novelty (handles gapless albums like DSOTM!)
    nov_short = _cosine_novelty(features.bands, half_win=8)    # ±4 s
    nov_long = _cosine_novelty(features.bands, half_win=20)    # ±10 s
    chroma_nov = _cosine_novelty(features.chroma, half_win=16) # ±8 s
    dip_db = _energy_dip_curve(features.rms_db, surround_half_win=16)

    # Normalize curves robustly
    def _robust_norm(arr: np.ndarray) -> np.ndarray:
        p95 = float(np.percentile(arr, 95))
        return np.clip(arr / max(p95, 1e-6), 0.0, 2.0)

    nov_combined = 0.45 * _robust_norm(nov_short) + 0.35 * _robust_norm(nov_long) + 0.20 * _robust_norm(chroma_nov)
    dip_norm = np.clip(dip_db / 18.0, 0.0, 2.0)

    # Check if release has vinyl side breaks (Side A -> Side B, etc.)
    side_breaks: list[int] = []
    for k in range(1, len(items)):
        prev_side = (items[k - 1].set_number or "").strip()
        curr_side = (items[k].set_number or "").strip()
        if prev_side and curr_side and prev_side != curr_side:
            side_breaks.append(k)

    # Global speed ratio estimate (clamped to sane turntable / mastering range)
    sum_durs = sum(durs)
    content_span = max(10.0, t_end - t_start)
    raw_alpha = content_span / max(1.0, sum_durs)
    global_alpha = float(np.clip(raw_alpha, 0.90, 1.10))

    # Locate vinyl side break(s) first so Side A and Side B get independent
    # speed ratios and side-flip gap offsets.
    side_anchors: dict[int, tuple[float, float]] = {}  # k -> (side_end_s, next_side_start_s)
    cum = 0.0
    cum_before = []
    for d in durs:
        cum_before.append(cum)
        cum += d

    for sb_k in side_breaks:
        est_s = t_start + cum_before[sb_k] * global_alpha
        win = 50.0
        lo = features.index_of(max(t_start + 15.0, est_s - win))
        hi = features.index_of(min(t_end - 15.0, est_s + win))
        if hi > lo + 2:
            # Side flip almost always has a deep energy dip / silence
            idx_grid = np.arange(lo, hi)
            times = (idx_grid + 0.5) * features.hop_seconds
            prior = np.exp(-0.5 * ((times - est_s) / (win * 0.65)) ** 2)
            # Strongly favor deep quiet stretches at a vinyl side flip
            sb_score = (2.0 * dip_norm[lo:hi] + 0.3 * nov_combined[lo:hi]) * (0.4 + 0.6 * prior)
            best_i = lo + int(np.argmax(sb_score))
            if dip_db[best_i] >= 8.0 or nov_combined[best_i] >= 0.65:
                # Smooth RMS with a 3-second (6-hop) median filter so a single
                # vinyl pop/click during the side flip does not split the quiet gap.
                rms_smooth = np.copy(features.rms_db)
                for idx_s in range(lo, hi):
                    w0 = max(0, idx_s - 3)
                    w1 = min(features.n_frames, idx_s + 4)
                    rms_smooth[idx_s] = float(np.median(features.rms_db[w0:w1]))
                v_lo = max(lo, best_i - 30)
                v_hi = min(hi, best_i + 30)
                valley_i = v_lo + int(np.argmin(rms_smooth[v_lo:v_hi]))
                gap_thresh = float(rms_smooth[valley_i]) + 7.0
                g_l = valley_i
                while g_l > lo and rms_smooth[g_l - 1] <= gap_thresh:
                    g_l -= 1
                g_r = valley_i
                # Require sustained music level (>= 3s median above threshold) so a single
                # vinyl needle-drop thump in the run-in groove does not end the side-flip gap early.
                music_thresh = max(float(rms_smooth[valley_i]) + 11.0, float(np.percentile(features.rms_db, 50)) - 14.0)
                while g_r + 1 < hi:
                    ahead_hi = min(hi, g_r + 7)
                    if float(np.median(features.rms_db[g_r + 1:ahead_hi])) >= music_thresh:
                        break
                    g_r += 1
                side_anchors[sb_k] = (
                    g_l * features.hop_seconds,
                    (g_r + 1) * features.hop_seconds,
                )

    # Compute expected boundary timestamp for every interior cut k in 1..N-1
    expected_times: list[float] = [0.0] * len(items)
    track_alphas: list[float] = [global_alpha] * len(items)
    segment_bounds = [0] + sorted(side_anchors.keys()) + [len(items)]
    side_speeds: dict[str, float] = {}

    for seg_i in range(len(segment_bounds) - 1):
        i0 = segment_bounds[seg_i]
        i1 = segment_bounds[seg_i + 1]
        seg_start_s = t_start if i0 == 0 else side_anchors[i0][1]
        seg_end_s = t_end if i1 == len(items) else side_anchors[i1][0]
        seg_mb_dur = sum(durs[i0:i1])
        alpha = float(np.clip((seg_end_s - seg_start_s) / max(1.0, seg_mb_dur), 0.91, 1.09))
        side_label = items[i0].set_number or f"Part {seg_i + 1}"
        side_speeds[side_label] = round(alpha, 4)

        running = seg_start_s
        for k in range(i0, i1):
            expected_times[k] = running
            track_alphas[k] = alpha
            running += durs[k] * alpha

    # Positive energy step curve (detects quiet-to-loud seamless transitions like crescendos)
    step_half = 8  # 4s before vs 4s after
    rms_step_pos = np.zeros(features.n_frames, dtype=np.float32)
    for i in range(step_half, max(step_half, features.n_frames - step_half)):
        before_rms = float(np.mean(features.rms_db[i - step_half:i]))
        after_rms = float(np.mean(features.rms_db[i:i + step_half]))
        rms_step_pos[i] = max(0.0, after_rms - before_rms)
    rms_step_norm = np.clip((rms_step_pos - 2.5) / 8.0, 0.0, 1.2)

    outcome.diagnostics = {
        "lead_in_offset_s": round(t_start, 2),
        "tail_end_s": round(t_end, 2),
        "global_speed_ratio": round(global_alpha, 4),
        "side_speeds": side_speeds,
        "side_anchors": {k: (round(a, 2), round(b, 2)) for k, (a, b) in side_anchors.items()},
    }

    # Refine each interior boundary k = 1..N-1 using actual audio
    cut_frames: list[int] = [0]
    cut_meta: list[dict] = [{}]  # aligned with tracks

    for k in range(1, len(items)):
        if cancel and cancel():
            break
        global_exp_s = expected_times[k]
        prev_cut_s = cut_frames[-1] / float(sample_rate)
        # Sequential anchor propagation: blend global grid with previous refined cut
        # so drift on gapless albums does not push the next track out of its search window.
        if k > 1 and (k - 1) not in side_anchors and float(cut_meta[-1].get("confidence", 0.0)) >= 0.55:
            seq_exp_s = prev_cut_s + durs[k - 1] * track_alphas[k - 1]
            exp_s = 0.55 * seq_exp_s + 0.45 * global_exp_s
        else:
            exp_s = global_exp_s

        if k in side_anchors:
            # Vinyl side flip: refine inside the detected flip gap
            gap_l, gap_r = side_anchors[k]
            approx_s = gap_r
            frame, onset_f, pre_roll_s, t_s, t_e = _refine_cut_frame_fine(
                reader, approx_s, sample_rate, total_frames, noise_floor, placement, has_dip=True
            )
            frame = max(cut_frames[-1] + int(1.0 * sample_rate), min(total_frames - 1, frame))
            cut_frames.append(frame)
            cut_meta.append({
                "confidence": 0.88,
                "uncertain": False,
                "onset_frame": onset_f,
                "pre_roll": pre_roll_s,
                "trans_start": t_s,
                "trans_end": t_e,
                "reason": (
                    f"Vinyl side break ({items[k - 1].set_number} -> {items[k].set_number}): "
                    f"quiet gap {gap_r - gap_l:.1f}s refined to {frame / sample_rate:.1f}s."
                ),
            })
            continue

        # Adaptive search radius around expected cumulative timestamp
        win_s = min(26.0, max(8.0, 0.30 * min(durs[k - 1], durs[k])))
        lo_s = max(prev_cut_s + 5.0, exp_s - win_s)
        hi_s = min(total_seconds - 5.0, exp_s + win_s)
        lo = features.index_of(lo_s)
        hi = features.index_of(hi_s)

        if hi <= lo + 1:
            frame = max(cut_frames[-1] + 1, min(total_frames - 1, int(round(exp_s * sample_rate))))
            cut_frames.append(frame)
            cut_meta.append({
                "confidence": 0.35,
                "uncertain": True,
                "onset_frame": -1,
                "pre_roll": -1.0,
                "trans_start": -1,
                "trans_end": -1,
                "reason": f"Metadata expected ~{exp_s:.1f}s; narrow search window, review cut.",
            })
            continue

        idx_grid = np.arange(lo, hi)
        times = (idx_grid + 0.5) * features.hop_seconds
        sigma = max(6.0, win_s * 0.65)
        prior = np.exp(-0.5 * ((times - exp_s) / sigma) ** 2)

        # Combine audio dip + spectral/harmonic novelty + quiet-to-loud step weighted by gentle timing prior
        audio_ev = 0.90 * dip_norm[lo:hi] + 1.00 * nov_combined[lo:hi] + 0.45 * rms_step_norm[lo:hi]
        score = audio_ev * (0.45 + 0.55 * prior)
        best_rel = int(np.argmax(score))
        best_idx = lo + best_rel
        best_s = float(times[best_rel])
        best_dip = float(dip_db[best_idx])
        best_nov = float(nov_combined[best_idx])
        has_dip = best_dip >= 8.0

        frame, onset_f, pre_roll_s, t_s, t_e = _refine_cut_frame_fine(
            reader, best_s, sample_rate, total_frames, noise_floor, placement, has_dip=has_dip
        )
        frame = max(cut_frames[-1] + int(1.0 * sample_rate), min(total_frames - 1, frame))
        refined_s = frame / float(sample_rate)
        shift_s = refined_s - exp_s

        # Confidence reflects genuine audio corroboration, NOT blind timestamp trust
        ev_strength = min(1.0, 0.5 * dip_norm[best_idx] + 0.6 * nov_combined[best_idx])
        conf = float(np.clip(0.35 + 0.50 * ev_strength, 0.35, 0.85))
        uncertain = (best_dip < 6.0 and best_nov < 0.55)

        if has_dip:
            ev_desc = f"energy dip (-{best_dip:.1f} dB) & spectral shift ({best_nov:.2f})"
        else:
            ev_desc = f"gapless spectral/harmonic shift ({best_nov:.2f})"

        cut_frames.append(frame)
        cut_meta.append({
            "confidence": round(conf, 3),
            "uncertain": uncertain,
            "onset_frame": onset_f,
            "pre_roll": pre_roll_s,
            "trans_start": t_s,
            "trans_end": t_e,
            "reason": (
                f"MusicBrainz expected ~{exp_s:.1f}s; refined {shift_s:+.1f}s to "
                f"{refined_s:.1f}s using {ev_desc}."
            ),
        })

        if progress:
            progress(0.55 + 0.40 * (k / max(1, len(items) - 1)), f"Refining boundary {k}/{len(items) - 1}...")

    cut_frames.append(total_frames)

    # Optional multi-engine acoustic verification on album tracks
    rec_by_idx: dict[int, object] = {}
    if allow_fingerprint and not (cancel and cancel()):
        from ..metadata.multi_engine import (
            STATUS_CONFLICT,
            STATUS_CONFIRMED,
            STATUS_LIKELY,
            MultiEngineRecognizer,
            same_song,
        )
        from ..core.model import FLAG_RECOGNITION_CONFLICT

        recognizer = MultiEngineRecognizer(
            cache_dir=cache_dir,
            use_acoustid=bool(audio_path),
            use_shazamio=use_shazamio,
            enrich_musicbrainz=False,
        )
        if recognizer.active_engines:
            seg_pairs = [
                (cut_frames[i] / float(sample_rate), cut_frames[i + 1] / float(sample_rate))
                for i in range(len(items))
            ]
            recs = recognizer.identify_segments(
                audio_path,
                seg_pairs,
                reader=reader,
                sample_rate=sample_rate,
                coarse_features=features,
                n_clips=2,
                cancel=cancel,
            )
            for i, rec in enumerate(recs):
                rec_by_idx[i] = rec
            outcome.diagnostics["recognition_engines"] = recognizer.active_engines

    # Assemble contiguous PlannedTracks & Suggestions
    for idx, item in enumerate(items):
        s_frame = cut_frames[idx]
        e_frame = cut_frames[idx + 1]
        meta = cut_meta[idx]
        reasons = []
        if meta.get("reason"):
            reasons.append(meta["reason"])
        reasons.append(f"MusicBrainz: #{item.position} {item.title} (published {item.duration_hint:.0f}s)")

        rec = rec_by_idx.get(idx)
        rec_artist = ""
        rec_title = ""
        rec_status = ""
        rec_engines: list[str] = []
        track_flags: list[str] = []
        id_conf = 0.82 if identification.confidence >= 0.75 else 0.65

        if rec is not None and getattr(rec, "status", "") not in ("", "unidentified"):
            from ..metadata.multi_engine import same_song

            rec_artist = rec.artist
            rec_title = rec.title
            rec_status = rec.status
            rec_engines = list(rec.engines)
            reasons.extend(rec.reasons)
            if rec.status == "conflict":
                from ..core.model import FLAG_RECOGNITION_CONFLICT
                track_flags.append(FLAG_RECOGNITION_CONFLICT)
                id_conf = 0.45
            elif rec.status in ("confirmed", "likely"):
                if same_song(identification.artist, item.title, rec_artist, rec_title):
                    id_conf = max(id_conf, float(rec.confidence))
                else:
                    reasons.append(
                        f"Kept album artist '{identification.artist}' for '{item.title}' "
                        f"(ignored conflicting single-clip match '{rec_artist} - {rec_title}')."
                    )

        # On a confirmed single-artist album, never let a disagreeing acoustic
        # match overwrite the album artist (only use rec_artist for compilations
        # like 'Various Artists' or when identification.artist is missing).
        is_compilation = (identification.artist or "").strip().lower() in (
            "", "various artists", "various", "soundtrack", "original broadway cast",
        )
        track_artist = (
            (rec_artist or identification.artist)
            if is_compilation
            else identification.artist
        )

        outcome.tracks.append(
            PlannedTrack(
                start_frame=s_frame,
                end_frame=e_frame,
                title=item.title,
                artist=track_artist,
                section_type="music",
                boundary_confidence=float(meta.get("confidence", 0.80 if idx == 0 else 0.50)),
                identity_confidence=id_conf,
                boundary_uncertain=bool(meta.get("uncertain", False)),
                boundary_placement="metadata_album",
                onset_frame=int(meta.get("onset_frame", -1)),
                boundary_pre_roll_seconds=float(meta.get("pre_roll", -1.0)),
                transition_start=int(meta.get("trans_start", -1)),
                transition_end=int(meta.get("trans_end", -1)),
                worklist_index=idx,
                reasons=reasons,
                notes=item.notes,
                recognized_artist=rec_artist,
                recognized_title=rec_title,
                recognition_status=rec_status,
                recognition_engines=rec_engines,
                flags=track_flags,
            )
        )
        if idx > 0:
            outcome.suggestions.append(
                Suggestion(
                    frame=s_frame,
                    confidence=float(meta.get("confidence", 0.5)),
                    reason=meta.get("reason", "Metadata-guided boundary"),
                    kind="metadata",
                    uncertain=bool(meta.get("uncertain", False)),
                    placement="metadata_album",
                    onset_frame=int(meta.get("onset_frame", -1)),
                    transition_start=int(meta.get("trans_start", -1)),
                    transition_end=int(meta.get("trans_end", -1)),
                )
            )

    outcome.notes.append(
        f"Refined {len(items) - 1} album boundaries using audio evidence "
        f"(lead-in {t_start:.1f}s, speed {global_alpha:.3f}x"
        + (f", {len(side_anchors)} vinyl side break(s)" if side_anchors else "")
        + ")."
    )
    outcome.elapsed_seconds = time.time() - started
    return outcome


# ----------------------------------------------------------------------
# 2. Live Concert Metadata-Assisted Splitting
# ----------------------------------------------------------------------

def split_concert_with_metadata(
    reader,
    sample_rate: int,
    total_frames: int,
    identification: IdentificationResult,
    placement: Optional[PlacementChoice] = None,
    audio_path: str = "",
    allow_fingerprint: bool = False,
    use_shazamio: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    progress: ProgressCallback = None,
    cancel: CancelCallback = None,
) -> MetadataSplitOutcome:
    """Detect live concert song boundaries across crowd noise, banter, and encores."""
    from . import aec
    from .align import align_worklist, stage_duration_score

    started = time.time()
    outcome = MetadataSplitOutcome(identification=identification)
    total_seconds = total_frames / float(sample_rate)

    # Prefer YAMNet AEC (~5s for a 90-minute show) to separate music from applause & speech banter.
    events = None
    if aec.onnxruntime_available() and aec.model_installed():
        try:
            events = aec.classify_reader(
                reader,
                sample_rate,
                total_frames,
                progress=lambda f, _: progress(0.05 + 0.55 * f, "Analysing concert music, crowd & banter...") if progress else None,
                cancel=cancel,
            )
        except Exception:
            events = None

    song_spans: list[tuple[float, float, str]] = []  # (start_s, end_s, reason)
    encore_break_s: Optional[float] = None

    if events is not None and events.n_frames() > 10:
        dt = float(events.frame_seconds)
        n = events.n_frames()

        # Smooth curves (~3.5 s window) to survive brief intra-song vocal/drum pauses
        win = max(3, int(round(3.5 / dt)))
        kernel = np.ones(win, dtype=np.float32) / float(win)
        music = np.convolve(events.music, kernel, mode="same")
        applause = np.convolve(events.applause, kernel, mode="same")
        speech = np.convolve(events.speech, kernel, mode="same")

        # A frame is "between songs" when music is low OR crowd/applause/banter dominates music
        is_gap = (music < 0.28) | ((applause + 0.7 * speech > 0.35) & (music < 0.45))

        # Collect contiguous gap runs lasting >= 3.5 seconds
        min_gap_frames = max(2, int(round(3.5 / dt)))
        gaps: list[tuple[int, int, float, float]] = []  # (s_i, e_i, max_applause, max_speech)
        i = 0
        while i < n:
            if not is_gap[i]:
                i += 1
                continue
            j = i
            while j < n and is_gap[j]:
                j += 1
            if j - i >= min_gap_frames:
                gaps.append((i, j, float(np.max(applause[i:j])), float(np.max(speech[i:j]))))
            i = j

        # Filter out gaps too close to the very start/end, and detect encore break
        # by scoring applause-dominant gaps in the final ~32% of the show rather
        # than blindly grabbing the first quiet instrumental intro after 65%.
        worklist_pre = identification.worklist
        expected_encore_s: Optional[float] = None
        if worklist_pre and worklist_pre.mode == "concert" and worklist_pre.items:
            enc_i = next(
                (idx_w for idx_w, it in enumerate(worklist_pre.items) if "encore" in (it.notes or "").lower()),
                -1,
            )
            if enc_i > 0:
                expected_encore_s = total_seconds * (enc_i / float(len(worklist_pre.items)))

        raw_cuts: list[tuple[float, str, float, float]] = []  # (cut_s, reason, gap_dur_s, max_app)
        encore_candidates: list[tuple[float, int]] = []  # (score, raw_idx)
        for g_s, g_e, max_app, max_sp in gaps:
            gap_dur = (g_e - g_s) * dt
            # Cut where the next song's music resumes (end of applause/banter gap)
            cut_s = g_e * dt
            if cut_s < 40.0 or cut_s > total_seconds - 40.0:
                continue
            if max_app >= 0.35 and max_sp >= 0.30:
                reason = f"Applause + stage banter ({gap_dur:.1f}s) before next song"
            elif max_app >= 0.35:
                reason = f"Crowd applause break ({gap_dur:.1f}s) before next song"
            elif max_sp >= 0.35:
                reason = f"Stage banter ({gap_dur:.1f}s) before next song"
            else:
                reason = f"Between-song break ({gap_dur:.1f}s) before next song"
            raw_idx = len(raw_cuts)
            raw_cuts.append((cut_s, reason, gap_dur, max_app))
            # Encore break candidate: prolonged crowd/applause gap in final ~32% of show
            if gap_dur >= 22.0 and cut_s >= total_seconds * 0.68 and cut_s <= total_seconds - 180.0 and max_app >= 0.25:
                prox = 1.0
                if expected_encore_s is not None:
                    dist_s = abs(cut_s - expected_encore_s)
                    prox = math.exp(-0.5 * (dist_s / max(180.0, total_seconds * 0.08)) ** 2)
                enc_score = gap_dur * (0.4 + max_app) * (0.35 + 0.65 * prox)
                encore_candidates.append((enc_score, raw_idx))

        if encore_candidates:
            encore_candidates.sort(key=lambda x: -x[0])
            best_raw_idx = encore_candidates[0][1]
            c_s, _, g_dur, m_app = raw_cuts[best_raw_idx]
            encore_break_s = c_s
            raw_cuts[best_raw_idx] = (
                c_s,
                f"Encore break ({g_dur:.0f}s crowd/applause) before next song at {c_s:.1f}s",
                g_dur,
                m_app,
            )

        # Enforce minimum separation (60 s) so short intra-song quiet bridges/intros don't fragment a song
        filtered_cuts: list[tuple[float, str, float]] = []
        for cut_s, reason, gap_dur, _ in raw_cuts:
            if filtered_cuts and (cut_s - filtered_cuts[-1][0]) < 60.0:
                # Keep the stronger / longer applause-banter break (never drop encore break)
                if encore_break_s is not None and abs(filtered_cuts[-1][0] - encore_break_s) < 2.0:
                    continue
                if gap_dur > filtered_cuts[-1][2]:
                    filtered_cuts[-1] = (cut_s, reason, gap_dur)
                continue
            filtered_cuts.append((cut_s, reason, gap_dur))

        # Subdivide over-long merged segments (> 420s) where bands segued without a
        # full YAMNet music drop, using coarse energy dips + spectral novelty.
        temp_edges = [0.0] + [c[0] for c in filtered_cuts] + [total_seconds]
        has_overlong = any((temp_edges[k + 1] - temp_edges[k]) > 420.0 for k in range(len(temp_edges) - 1))
        if has_overlong and not (cancel and cancel()):
            coarse = extract_coarse_features(reader, sample_rate, total_frames, hop_seconds=1.0, cancel=cancel)
            dip_c = _energy_dip_curve(coarse.rms_db, surround_half_win=25)
            nov_c = _cosine_novelty(coarse.bands, half_win=16)
            sub_score = dip_c / 10.0 + nov_c * 2.5
            extra_cuts: list[tuple[float, str, float]] = []
            for k in range(len(temp_edges) - 1):
                seg_a, seg_b = temp_edges[k], temp_edges[k + 1]
                if (seg_b - seg_a) <= 420.0:
                    continue
                lo_i = coarse.index_of(seg_a + 90.0)
                hi_i = coarse.index_of(seg_b - 90.0)
                if hi_i <= lo_i + 5:
                    continue
                local_peaks: list[tuple[float, float]] = []
                for idx_c in range(lo_i + 2, hi_i - 2):
                    val = float(sub_score[idx_c])
                    if val >= float(np.max(sub_score[idx_c - 10:idx_c + 11])) and dip_c[idx_c] >= 5.5:
                        local_peaks.append((coarse.time_of(idx_c), val))
                local_peaks.sort(key=lambda p: -p[1])
                chosen_sub: list[float] = []
                max_sub = max(1, int((seg_b - seg_a) // 200.0))
                for t_p, _ in local_peaks:
                    if any(abs(t_p - cp) < 110.0 for cp in chosen_sub):
                        continue
                    chosen_sub.append(t_p)
                    if len(chosen_sub) >= max_sub:
                        break
                for t_p in sorted(chosen_sub):
                    extra_cuts.append((t_p, f"Internal song transition inside long live medley/segue at {t_p:.1f}s", 16.0))
            if extra_cuts:
                filtered_cuts.extend(extra_cuts)
                filtered_cuts.sort(key=lambda c: c[0])

        # When an exact concert setlist is known and audio over-segmented due to
        # brief intra-song dips (<12s), drop the weakest short dips first so the
        # segment count stays realistic without discarding real applause/banter breaks.
        if worklist_pre and worklist_pre.mode == "concert" and worklist_pre.items:
            target_cuts = max(1, len(worklist_pre.items) - 1)
            while len(filtered_cuts) > target_cuts:
                weak_idx = -1
                weak_dur = 12.0
                for ci, (cs, rsn, gdur) in enumerate(filtered_cuts):
                    if encore_break_s is not None and abs(cs - encore_break_s) < 2.0:
                        continue
                    if "applause" in rsn.lower() or "banter" in rsn.lower():
                        continue
                    if gdur < weak_dur:
                        weak_dur = gdur
                        weak_idx = ci
                if weak_idx < 0:
                    break
                filtered_cuts.pop(weak_idx)

        edges = [0.0] + [c[0] for c in filtered_cuts] + [total_seconds]
        reasons_map = [
            "Opening of concert"
        ] + [c[1] for c in filtered_cuts]
        for idx in range(len(edges) - 1):
            song_spans.append((edges[idx], edges[idx + 1], reasons_map[idx]))
    else:
        # Fallback without YAMNet: macro-scale spectral novelty + energy dips
        features = extract_coarse_features(reader, sample_rate, total_frames, hop_seconds=0.5)
        dip = _energy_dip_curve(features.rms_db, surround_half_win=20)
        nov = _cosine_novelty(features.bands, half_win=16)
        combined = dip / 15.0 + nov * 2.0
        thresh = float(np.percentile(combined, 92))
        cuts_s: list[float] = []
        min_sep = 75.0
        for i in range(1, combined.size - 1):
            if combined[i] >= thresh and combined[i] >= combined[i - 1] and combined[i] >= combined[i + 1]:
                t = features.time_of(i)
                if t < 40.0 or t > total_seconds - 40.0:
                    continue
                if cuts_s and t - cuts_s[-1] < min_sep:
                    continue
                cuts_s.append(t)
        edges = [0.0] + cuts_s + [total_seconds]
        for idx in range(len(edges) - 1):
            song_spans.append((edges[idx], edges[idx + 1], "Detected live song transition"))

    # Convert spans to frame segments
    segments_frames = [
        (int(round(s * sample_rate)), int(round(e * sample_rate)))
        for s, e, _ in song_spans
    ]
    if segments_frames:
        segments_frames[0] = (0, segments_frames[0][1])
        segments_frames[-1] = (segments_frames[-1][0], total_frames)

    # Align exact setlist as supporting evidence (never for a generic artist catalog!)
    worklist = identification.worklist
    assigned: dict[int, tuple[int, float]] = {}
    if worklist and worklist.mode == "concert" and worklist.items and segments_frames:
        # If both an audio encore break and setlist encore items exist, align
        # main set and encore independently so songs never cross the encore break.
        first_encore_item = next(
            (i for i, it in enumerate(worklist.items) if "encore" in (it.notes or "").lower()),
            -1,
        )
        first_encore_seg = (
            next(
                (i for i, (sf, _) in enumerate(segments_frames) if (sf / float(sample_rate)) >= encore_break_s - 1.0),
                -1,
            )
            if encore_break_s is not None
            else -1
        )
        if first_encore_item > 0 and first_encore_seg > 0:
            main_align = align_worklist(
                worklist.items[:first_encore_item],
                segments_frames[:first_encore_seg],
                sample_rate,
                duration_scorer=stage_duration_score,
            )
            for (item_idx, seg_idx), conf in zip(main_align.assignments, main_align.identity_confidence):
                assigned[seg_idx] = (item_idx, float(conf))
            enc_align = align_worklist(
                worklist.items[first_encore_item:],
                segments_frames[first_encore_seg:],
                sample_rate,
                duration_scorer=stage_duration_score,
            )
            for (item_idx, seg_idx), conf in zip(enc_align.assignments, enc_align.identity_confidence):
                assigned[first_encore_seg + seg_idx] = (first_encore_item + item_idx, float(conf))
        else:
            alignment = align_worklist(
                worklist.items,
                segments_frames,
                sample_rate,
                duration_scorer=stage_duration_score,
            )
            for (item_idx, seg_idx), conf in zip(alignment.assignments, alignment.identity_confidence):
                assigned[seg_idx] = (item_idx, float(conf))
        outcome.notes.append(
            f"Aligned {len(assigned)}/{len(segments_frames)} concert segments against setlist "
            f"'{worklist.name}' as supporting evidence."
        )
    else:
        outcome.notes.append(
            f"Detected {len(segments_frames)} live concert segments using music/applause/banter "
            "transitions (no exact setlist available; titles left honest rather than guessed)."
        )

    # Optional multi-engine acoustic recognition on concert segments (avoiding applause & banter)
    concert_recs: dict[int, object] = {}
    setlist_verified_by_audio = 0
    if allow_fingerprint and segments_frames and not (cancel and cancel()):
        from ..metadata.multi_engine import MultiEngineRecognizer, same_song

        recognizer = MultiEngineRecognizer(
            cache_dir=cache_dir,
            use_acoustid=bool(audio_path),
            use_shazamio=use_shazamio,
            enrich_musicbrainz=False,
        )
        if recognizer.active_engines:
            seg_pairs = [
                (sf / float(sample_rate), ef / float(sample_rate))
                for sf, ef in segments_frames
            ]
            recs = recognizer.identify_segments(
                audio_path,
                seg_pairs,
                reader=reader,
                sample_rate=sample_rate,
                audio_events=events,
                n_clips=2,
                progress=lambda f, m: progress(0.65 + 0.30 * f, m) if progress else None,
                cancel=cancel,
            )
            for idx_r, rec in enumerate(recs):
                concert_recs[idx_r] = rec

            # If we have a concert setlist AND acoustic matches to setlist songs,
            # use monotonic acoustic anchors to re-align segments between anchors
            # so a single missed/extra break never shifts the rest of the show!
            if worklist and worklist.mode == "concert" and worklist.items and concert_recs:
                raw_anchors: list[tuple[int, int, float]] = []
                for seg_i, rec in sorted(concert_recs.items()):
                    if getattr(rec, "status", "") not in ("confirmed", "likely") or not getattr(rec, "title", ""):
                        continue
                    for wi, it in enumerate(worklist.items):
                        if same_song(identification.artist, it.title, rec.artist, rec.title):
                            raw_anchors.append((seg_i, wi, float(rec.confidence)))
                            break
                # Keep longest strictly increasing (seg_i, wi) chain
                anchors: list[tuple[int, int, float]] = []
                for seg_i, wi, cf in raw_anchors:
                    if not anchors or (seg_i > anchors[-1][0] and wi > anchors[-1][1]):
                        anchors.append((seg_i, wi, cf))
                if anchors:
                    realigned: dict[int, tuple[int, float]] = {}
                    boundaries_a = [(-1, -1, 0.0)] + anchors + [(len(segments_frames), len(worklist.items), 0.0)]
                    for bi in range(len(boundaries_a) - 1):
                        s_lo, w_lo, _ = boundaries_a[bi]
                        s_hi, w_hi, _ = boundaries_a[bi + 1]
                        if s_lo >= 0 and w_lo >= 0:
                            realigned[s_lo] = (w_lo, boundaries_a[bi][2])
                        sub_segs = segments_frames[s_lo + 1:s_hi]
                        sub_items = worklist.items[w_lo + 1:w_hi]
                        if sub_segs and sub_items:
                            sub_al = align_worklist(
                                sub_items,
                                sub_segs,
                                sample_rate,
                                duration_scorer=stage_duration_score,
                            )
                            for (it_rel, sg_rel), c_val in zip(sub_al.assignments, sub_al.identity_confidence):
                                realigned[s_lo + 1 + sg_rel] = (w_lo + 1 + it_rel, float(c_val))
                    if realigned:
                        assigned = realigned
                        outcome.notes.append(
                            f"Re-aligned concert setlist using {len(anchors)} acoustic song anchor(s)."
                        )

    encore_count = 0
    for idx, ((s_f, e_f), (_, _, reason)) in enumerate(zip(segments_frames, song_spans)):
        is_encore = encore_break_s is not None and (s_f / float(sample_rate)) >= encore_break_s - 1.0
        reasons = [reason]
        rec = concert_recs.get(idx)
        rec_artist = ""
        rec_title = ""
        rec_status = ""
        rec_engines: list[str] = []
        track_flags: list[str] = []
        audio_confirmed = False

        if idx in assigned and worklist is not None:
            w_idx, id_conf = assigned[idx]
            item = worklist.items[w_idx]
            title = item.title
            artist = identification.artist
            notes = item.notes
        else:
            w_idx = -1
            id_conf = 0.0
            artist = identification.artist
            if is_encore:
                encore_count += 1
                title = f"Encore {encore_count}"
            else:
                title = f"Song {idx + 1}"
            notes = "Encore" if is_encore else ""

        if rec is not None and getattr(rec, "status", "") not in ("", "unidentified"):
            from ..core.model import FLAG_RECOGNITION_CONFLICT
            from ..metadata.multi_engine import same_song

            rec_artist = rec.artist
            rec_title = rec.title
            rec_status = rec.status
            rec_engines = list(rec.engines)
            reasons.extend(rec.reasons)
            if rec.status == "conflict":
                track_flags.append(FLAG_RECOGNITION_CONFLICT)
            if w_idx >= 0 and worklist is not None:
                if same_song(identification.artist, title, rec_artist, rec_title):
                    setlist_verified_by_audio += 1
                    audio_confirmed = True
                    id_conf = max(id_conf, float(rec.confidence))
                    reasons.append(f"Acoustic recognition matches setlist song '{title}'.")
                elif rec.status in ("confirmed", "likely"):
                    # Check if the recognized song exists elsewhere in the setlist
                    matched_item = next(
                        (
                            (wi, it)
                            for wi, it in enumerate(worklist.items)
                            if same_song(identification.artist, it.title, rec_artist, rec_title)
                        ),
                        None,
                    )
                    if matched_item is not None:
                        setlist_verified_by_audio += 1
                        audio_confirmed = True
                        w_idx, it = matched_item
                        title = it.title
                        id_conf = float(rec.confidence)
                        reasons.append(f"Acoustic recognition anchored setlist song '{it.title}'.")
                    else:
                        reasons.append(
                            f"Note: acoustic recognition suggested '{rec.label}' while setlist alignment suggested '{title}'."
                        )
            elif rec.status in ("confirmed", "likely") and rec_title:
                # Guard against single-clip false positives from unrelated artists in a single-artist concert
                concert_artist = (identification.artist or "").strip().lower()
                rec_art_low = (rec_artist or "").strip().lower()
                artist_matches = (
                    not concert_artist
                    or not rec_art_low
                    or concert_artist in rec_art_low
                    or rec_art_low in concert_artist
                )
                if artist_matches or rec.status == "confirmed":
                    audio_confirmed = True
                    title = rec_title
                    artist = rec_artist or artist
                    id_conf = float(rec.confidence)
                else:
                    rec_status = "uncertain"
                    reasons.append(
                        f"Ignored unconfirmed single-clip match '{rec.label}' (artist differs from concert '{identification.artist}')."
                    )

        # Never confuse unverified setlist guesses with audio-confirmed songs
        if w_idx >= 0 and not audio_confirmed:
            id_conf = min(id_conf, 0.45)
            reasons.append("Setlist-aligned estimate (not audio-confirmed).")

        b_conf = 0.72 if ("applause" in reason.lower() or "encore" in reason.lower()) else 0.55
        outcome.tracks.append(
            PlannedTrack(
                start_frame=s_f,
                end_frame=e_f,
                title=title,
                artist=artist,
                section_type="music",
                boundary_confidence=b_conf,
                identity_confidence=id_conf,
                boundary_uncertain=(b_conf < 0.65),
                boundary_placement="metadata_concert",
                worklist_index=w_idx,
                reasons=reasons,
                notes=notes,
                recognized_artist=rec_artist,
                recognized_title=rec_title,
                recognition_status=rec_status,
                recognition_engines=rec_engines,
                flags=track_flags,
            )
        )
        if idx > 0:
            outcome.suggestions.append(
                Suggestion(
                    frame=s_f,
                    confidence=b_conf,
                    reason=reason,
                    kind="metadata",
                    uncertain=(b_conf < 0.65),
                    placement="metadata_concert",
                )
            )

    outcome.diagnostics = {
        "used_yamnet": events is not None,
        "encore_break_s": round(encore_break_s, 1) if encore_break_s is not None else None,
        "segments_detected": len(segments_frames),
        "setlist_matched": len(assigned),
        "setlist_verified_by_audio": setlist_verified_by_audio,
    }
    outcome.elapsed_seconds = time.time() - started
    return outcome


# ----------------------------------------------------------------------
# 3. DJ Mix / Vinyl Session Splitting (Crossfades + AcoustID)
# ----------------------------------------------------------------------

def split_dj_mix_with_metadata(
    reader,
    sample_rate: int,
    total_frames: int,
    identification: IdentificationResult,
    audio_path: str = "",
    allow_fingerprint: bool = True,
    use_shazamio: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    progress: ProgressCallback = None,
    cancel: CancelCallback = None,
) -> MetadataSplitOutcome:
    """Detect overlapping crossfades in a DJ mix; use AcoustID + optional ShazamIO."""
    from ..core.model import FLAG_RECOGNITION_CONFLICT
    from ..metadata.acoustid import ACOUSTID
    from ..metadata.multi_engine import (
        STATUS_CONFLICT,
        STATUS_CONFIRMED,
        STATUS_LIKELY,
        STATUS_UNIDENTIFIED,
        MultiEngineRecognizer,
        aggregate_clip_recognitions,
        format_artist_title,
        same_song,
    )
    from ..metadata.shazamio_engine import SHAZAMIO

    started = time.time()
    outcome = MetadataSplitOutcome(identification=identification)
    total_seconds = total_frames / float(sample_rate)

    features = extract_coarse_features(
        reader,
        sample_rate,
        total_frames,
        hop_seconds=1.0,
        progress=lambda f, m: progress(0.05 + 0.45 * f, "Analysing DJ mix crossfades & harmonic shifts...") if progress else None,
        cancel=cancel,
    )

    # Long-window (24 s before vs 24 s after) timbral + harmonic novelty
    # averages out 4-16 bar drops/choruses inside one record while lighting up
    # genuine record-to-record vinyl/DJ transitions!
    spec_nov = _cosine_novelty(features.bands, half_win=24)
    harm_nov = _cosine_novelty(features.chroma, half_win=24)
    # Also medium-window (10 s) to localize the midpoint of a crossfade
    spec_med = _cosine_novelty(features.bands, half_win=10)

    def _znorm(arr: np.ndarray) -> np.ndarray:
        std = float(np.std(arr))
        if std < 1e-8:
            return np.zeros_like(arr)
        return (arr - float(np.mean(arr))) / std

    macro_curve = 0.50 * _znorm(spec_nov) + 0.35 * _znorm(harm_nov) + 0.15 * _znorm(spec_med)

    def _refine_crossfade_cut(lo_s: float, hi_s: float, prior_s: Optional[float] = None) -> tuple[float, float]:
        """Locate the strongest crossfade peak in [lo_s, hi_s], gently guided by prior_s."""
        a = max(10.0, float(lo_s))
        b = min(total_seconds - 10.0, float(hi_s))
        if b <= a + 2.0:
            return (prior_s if prior_s is not None else 0.5 * (a + b)), 0.55
        lo = features.index_of(a)
        hi = features.index_of(b)
        if hi <= lo:
            return (prior_s if prior_s is not None else 0.5 * (a + b)), 0.55
        idx_grid = np.arange(lo, hi)
        times = (idx_grid + 0.5) * features.hop_seconds
        curve = macro_curve[lo:hi].astype(np.float64)
        # Shift to positive for weighting
        c_min = float(np.min(curve))
        shifted = curve - c_min + 0.1
        if prior_s is not None and a <= prior_s <= b:
            sigma = max(8.0, (b - a) * 0.25)
            prior = np.exp(-0.5 * ((times - prior_s) / sigma) ** 2)
            score = shifted * (0.45 + 0.55 * prior)
        else:
            score = shifted
        best_rel = int(np.argmax(score))
        best_t = float(times[best_rel])
        best_z = float(curve[best_rel])
        conf = float(np.clip(0.45 + 0.10 * max(0.0, best_z), 0.45, 0.72))
        return best_t, round(conf, 3)

    # Check if AcoustID already provided verified song identity runs (e.g. pre-populated windows)
    fp_runs: list[tuple[float, float, str, str, float]] = []  # (first_s, last_s, artist, title, score)
    for win in identification.fingerprint_windows:
        if win.match is None:
            continue
        if fp_runs and same_song(fp_runs[-1][2], fp_runs[-1][3], win.match.artist, win.match.title):
            first_s, _, art, ttl, sc = fp_runs[-1]
            fp_runs[-1] = (first_s, win.end_seconds, art, ttl, max(sc, win.match.score))
        else:
            fp_runs.append((win.start_seconds, win.end_seconds, win.match.artist, win.match.title, win.match.score))

    cuts: list[tuple[float, float, str, str, str, float]] = []
    # Each tuple: (cut_s, boundary_conf, reason, next_artist, next_title, next_id_conf)

    if len(fp_runs) >= 2:
        for i in range(len(fp_runs) - 1):
            a_end = fp_runs[i][1]
            b_start = fp_runs[i + 1][0]
            cut_s, b_conf = _refine_crossfade_cut(min(a_end, b_start) - 15.0, max(a_end, b_start) + 15.0)
            a_lbl = format_artist_title(fp_runs[i][2], fp_runs[i][3])
            b_lbl = format_artist_title(fp_runs[i + 1][2], fp_runs[i + 1][3])
            cuts.append((
                cut_s,
                max(0.62, b_conf),
                f"Crossfade transition from '{a_lbl}' to '{b_lbl}' (AcoustID + spectral shift).",
                fp_runs[i + 1][2],
                fp_runs[i + 1][3],
                round(float(fp_runs[i + 1][4]), 3),
            ))
        first_artist, first_title, first_id_conf = fp_runs[0][2], fp_runs[0][3], round(float(fp_runs[0][4]), 3)
    else:
        # Audio-only crossfade detection: NEVER invent song titles!
        first_artist, first_title, first_id_conf = "", "", 0.0
        min_sep_s = 110.0  # vinyl/DJ tracks are typically 2-6 minutes apart
        cand_peaks: list[tuple[float, float]] = []
        margin_i = 25
        for i in range(margin_i, max(margin_i, macro_curve.size - margin_i)):
            val = float(macro_curve[i])
            if val < 0.75:
                continue
            if val >= float(np.max(macro_curve[i - 20:i + 21])):
                cand_peaks.append((features.time_of(i), val))

        cand_peaks.sort(key=lambda p: -p[1])
        chosen: list[tuple[float, float]] = []
        max_cuts = max(4, int(round(total_seconds / 150.0)))
        for t_s, val in cand_peaks:
            if t_s < 60.0 or t_s > total_seconds - 60.0:
                continue
            if any(abs(t_s - ct) < min_sep_s for ct, _ in chosen):
                continue
            chosen.append((t_s, val))
            if len(chosen) >= max_cuts:
                break
        chosen.sort(key=lambda p: p[0])

        for t_s, val in chosen:
            conf = float(np.clip(0.35 + 0.12 * val, 0.35, 0.58))
            cuts.append((
                t_s,
                round(conf, 3),
                f"Overlapping crossfade / track transition (timbral+harmonic shift z={val:.2f}). "
                "No clean silence cut — listen and adjust.",
                "",
                "",
                0.0,
            ))

    edges_s = [0.0] + [c[0] for c in cuts] + [total_seconds]
    seg_artists: list[str] = [first_artist] + [c[3] for c in cuts]
    seg_titles: list[str] = [first_title] + [c[4] for c in cuts]
    seg_id_confs: list[float] = [first_id_conf] + [c[5] for c in cuts]
    seg_b_confs: list[float] = [0.50] + [c[1] for c in cuts]
    seg_reasons: list[list[str]] = [
        [
            f"AcoustID identified '{format_artist_title(first_artist, first_title)}'"
            if first_title
            else "Start of continuous DJ mix (track identity not confirmed)."
        ]
    ] + [[c[2]] for c in cuts]
    seg_statuses: list[str] = [
        (STATUS_LIKELY if first_title else STATUS_UNIDENTIFIED)
    ] + [(STATUS_LIKELY if c[4] else STATUS_UNIDENTIFIED) for c in cuts]
    seg_engines: list[list[str]] = [
        ([ACOUSTID] if first_title else [])
    ] + [([ACOUSTID] if c[4] else []) for c in cuts]
    seg_flags: list[list[str]] = [[] for _ in range(len(edges_s) - 1)]

    acoustid_hits = len(fp_runs)
    shazamio_hits = 0
    song_change_splits = 0
    active_engines: list[str] = []

    if len(fp_runs) < 2 and allow_fingerprint and not (cancel and cancel()):
        recognizer = MultiEngineRecognizer(
            cache_dir=cache_dir,
            use_acoustid=bool(audio_path),
            use_shazamio=use_shazamio,
            enrich_musicbrainz=True,
        )
        active_engines = list(recognizer.active_engines)
        if active_engines:
            seg_pairs = [(edges_s[i], edges_s[i + 1]) for i in range(len(edges_s) - 1)]
            recs = recognizer.identify_segments(
                audio_path,
                seg_pairs,
                reader=reader,
                sample_rate=sample_rate,
                coarse_features=features,
                progress=lambda f, m: progress(0.52 + 0.42 * f, m) if progress else None,
                cancel=cancel,
            )

            # 1. Refine boundaries when multi-clip recognition detected a song change
            # INSIDE a proposed segment (never blindly cutting at the clip timestamp!).
            refined_edges: list[float] = [edges_s[0]]
            refined_clips_per_seg: list[list] = []
            refined_b_confs: list[float] = []
            refined_base_reasons: list[list[str]] = []

            for idx, rec in enumerate(recs):
                s_s, e_s = edges_s[idx], edges_s[idx + 1]
                base_b_conf = seg_b_confs[idx] if idx < len(seg_b_confs) else 0.50
                base_rsn = list(seg_reasons[idx]) if idx < len(seg_reasons) else []

                sub_cuts: list[tuple[float, float, str]] = []
                if rec.song_changes:
                    for ch in rec.song_changes:
                        cut_t, c_conf = _refine_crossfade_cut(
                            ch.prev_clip_time + 2.0,
                            ch.next_clip_time + 2.0,
                            prior_s=ch.estimated_boundary_seconds,
                        )
                        if cut_t > s_s + 20.0 and cut_t < e_s - 20.0:
                            if not sub_cuts or (cut_t - sub_cuts[-1][0]) >= 35.0:
                                prev_lbl = format_artist_title(ch.prev_artist, ch.prev_title)
                                next_lbl = format_artist_title(ch.next_artist, ch.next_title)
                                sub_cuts.append((
                                    cut_t,
                                    c_conf,
                                    f"Refined crossfade boundary between '{prev_lbl}' and '{next_lbl}' "
                                    f"({ch.engine} song change + spectral shift).",
                                ))

                if sub_cuts:
                    song_change_splits += len(sub_cuts)
                    sub_edges = [s_s] + [sc[0] for sc in sub_cuts] + [e_s]
                    for sub_i in range(len(sub_edges) - 1):
                        a_sub, b_sub = sub_edges[sub_i], sub_edges[sub_i + 1]
                        refined_edges.append(b_sub)
                        sub_clips = [
                            cl for cl in rec.clips
                            if (a_sub - 2.0) <= cl.start_seconds < (b_sub - 1.0)
                        ]
                        refined_clips_per_seg.append(sub_clips)
                        if sub_i == 0:
                            refined_b_confs.append(base_b_conf)
                            refined_base_reasons.append(base_rsn)
                        else:
                            refined_b_confs.append(sub_cuts[sub_i - 1][1])
                            refined_base_reasons.append([sub_cuts[sub_i - 1][2]])
                else:
                    refined_edges.append(e_s)
                    refined_clips_per_seg.append(list(rec.clips))
                    refined_b_confs.append(base_b_conf)
                    refined_base_reasons.append(base_rsn)

            # Ensure monotonic edges matching len(refined_clips_per_seg) + 1
            if len(refined_edges) == len(refined_clips_per_seg) + 1:
                edges_s = refined_edges
                seg_recs = [
                    aggregate_clip_recognitions(edges_s[i], edges_s[i + 1], refined_clips_per_seg[i])
                    for i in range(len(refined_clips_per_seg))
                ]
                seg_b_confs = refined_b_confs
                seg_reasons = refined_base_reasons
            else:
                seg_recs = recs

            # 2. Merge adjacent segments that consistently matched the exact same song
            m_edges = [edges_s[0]]
            m_recs = [seg_recs[0]]
            m_b_confs = [seg_b_confs[0]]
            m_reasons = [list(seg_reasons[0])]
            for idx in range(1, len(seg_recs)):
                prev_r = m_recs[-1]
                curr_r = seg_recs[idx]
                if (
                    prev_r.status in (STATUS_CONFIRMED, STATUS_LIKELY)
                    and curr_r.status in (STATUS_CONFIRMED, STATUS_LIKELY)
                    and same_song(prev_r.artist, prev_r.title, curr_r.artist, curr_r.title)
                ):
                    combined_clips = list(prev_r.clips) + list(curr_r.clips)
                    m_recs[-1] = aggregate_clip_recognitions(m_edges[-1], edges_s[idx + 1], combined_clips)
                    continue
                m_edges.append(edges_s[idx])
                m_recs.append(curr_r)
                m_b_confs.append(seg_b_confs[idx])
                m_reasons.append(list(seg_reasons[idx]))
            m_edges.append(edges_s[-1])

            # 3. Guard against false trailing outro/applause splits at the end of a mix:
            # a short (<65s), significantly quieter (>6 dB below mix median RMS) tail
            # without multi-clip confirmation is a fade-out/outro, not a new song.
            if len(m_recs) >= 2 and features.rms_db.size > 10:
                tail_s = m_edges[-2]
                tail_e = m_edges[-1]
                tail_dur = tail_e - tail_s
                tail_r = m_recs[-1]
                if tail_dur < 65.0 and tail_r.status != STATUS_CONFIRMED:
                    mix_med_db = float(np.median(features.rms_db))
                    lo_i = features.index_of(tail_s)
                    hi_i = max(lo_i + 1, features.index_of(tail_e))
                    tail_med_db = float(np.median(features.rms_db[lo_i:hi_i]))
                    if tail_med_db <= mix_med_db - 6.0:
                        m_edges.pop(-2)
                        m_recs.pop()
                        m_b_confs.pop()
                        m_reasons.pop()
                        m_reasons[-1].append(
                            f"Merged quiet trailing outro ({tail_dur:.1f}s, {tail_med_db:.1f} dBFS vs "
                            f"mix median {mix_med_db:.1f} dBFS) into final track."
                        )

            edges_s = m_edges
            seg_b_confs = m_b_confs
            seg_artists = []
            seg_titles = []
            seg_id_confs = []
            seg_statuses = []
            seg_engines = []
            seg_flags = []
            seg_reasons = []
            acoustid_hits = 0
            shazamio_hits = 0

            for idx, r in enumerate(m_recs):
                rsns = list(m_reasons[idx]) + list(r.reasons)
                seg_reasons.append(rsns)
                seg_artists.append(r.artist)
                seg_titles.append(r.title)
                seg_id_confs.append(round(float(r.confidence), 3))
                seg_statuses.append(r.status)
                seg_engines.append(list(r.engines))
                fl: list[str] = []
                if r.status == STATUS_CONFLICT:
                    fl.append(FLAG_RECOGNITION_CONFLICT)
                seg_flags.append(fl)
                if ACOUSTID in r.engines:
                    acoustid_hits += 1
                if SHAZAMIO in r.engines:
                    shazamio_hits += 1

    edges_frames = [0] + [int(round(t * sample_rate)) for t in edges_s[1:-1]] + [total_frames]
    confirmed_count = 0
    likely_count = 0
    conflict_count = 0
    unidentified_count = 0

    for idx in range(len(edges_frames) - 1):
        s_f = edges_frames[idx]
        e_f = edges_frames[idx + 1]
        art = seg_artists[idx] if idx < len(seg_artists) else ""
        ttl = seg_titles[idx] if idx < len(seg_titles) else ""
        # Store the clean song title in Track.title (Track.artist holds the artist)
        # so tags and exported filenames never duplicate "Artist - Artist - Title".
        display_title = ttl if ttl else f"Track {idx + 1}"
        id_conf = seg_id_confs[idx] if idx < len(seg_id_confs) else 0.0
        b_conf = seg_b_confs[idx] if idx < len(seg_b_confs) else 0.50
        reasons = seg_reasons[idx] if idx < len(seg_reasons) else []
        status = seg_statuses[idx] if idx < len(seg_statuses) else STATUS_UNIDENTIFIED
        engines = seg_engines[idx] if idx < len(seg_engines) else []
        flags = seg_flags[idx] if idx < len(seg_flags) else []

        if status == STATUS_CONFIRMED:
            confirmed_count += 1
        elif status == STATUS_LIKELY:
            likely_count += 1
        elif status == STATUS_CONFLICT:
            conflict_count += 1
        elif status == STATUS_UNIDENTIFIED:
            unidentified_count += 1

        outcome.tracks.append(
            PlannedTrack(
                start_frame=s_f,
                end_frame=e_f,
                title=display_title,
                artist=art,
                section_type="music",
                boundary_confidence=b_conf,
                identity_confidence=id_conf,
                boundary_uncertain=True,  # DJ crossfades always benefit from human ear check
                boundary_placement="metadata_dj_crossfade",
                reasons=reasons,
                recognized_artist=art,
                recognized_title=ttl,
                recognition_status=status,
                recognition_engines=engines,
                flags=flags,
            )
        )
        if idx > 0:
            outcome.suggestions.append(
                Suggestion(
                    frame=s_f,
                    confidence=b_conf,
                    reason=reasons[0] if reasons else "Overlapping crossfade transition",
                    kind="metadata",
                    uncertain=True,
                    placement="metadata_dj_crossfade",
                )
            )

    total_identified = confirmed_count + likely_count + conflict_count
    outcome.diagnostics = {
        "active_engines": active_engines,
        "fingerprint_runs": total_identified,
        "acoustid_hits": acoustid_hits,
        "shazamio_hits": shazamio_hits,
        "confirmed_count": confirmed_count,
        "likely_count": likely_count,
        "conflict_count": conflict_count,
        "unidentified_count": unidentified_count,
        "song_change_splits": song_change_splits,
        "crossfades_detected": max(0, len(edges_frames) - 2),
    }
    eng_desc = " + ".join(active_engines) if active_engines else "audio only"
    outcome.notes.append(
        f"Detected {max(0, len(edges_frames) - 2)} overlapping crossfade transition(s) "
        + (
            f"with {confirmed_count} confirmed, {likely_count} likely"
            + (f", {conflict_count} conflict(s)" if conflict_count else "")
            + f" via {eng_desc}."
            if total_identified
            else "(no song titles invented)."
        )
    )
    outcome.elapsed_seconds = time.time() - started
    return outcome


# ----------------------------------------------------------------------
# 4. Broadway / Musical Theatre Metadata-Assisted Splitting
# ----------------------------------------------------------------------

def split_broadway_with_metadata(
    project,
    identification: IdentificationResult,
    use_transcriber: bool = False,
    existing_transcript: Optional[list[dict]] = None,
    allow_fingerprint: bool = False,
    use_shazamio: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    progress: ProgressCallback = None,
    cancel: CancelCallback = None,
) -> MetadataSplitOutcome:
    """Segment Broadway live audio using AEC + acoustic structure + cast reference + LRCLIB lyrics.

    Preserves dialogue bridges inside musical numbers when appropriate (via
    ``TheaterJob`` / ``musical.py``), and NEVER uses cast album durations as
    live performance cut timestamps.
    """
    from ..app.jobs import TheaterJob
    from .align import align_theater_worklist

    started = time.time()
    outcome = MetadataSplitOutcome(identification=identification)
    sample_rate = project.sample_rate() or 44100
    total_frames = project.total_frames()

    job = TheaterJob(
        project,
        use_transcriber=use_transcriber,
        separate_applause=False,
        progress=progress,
    )
    if cancel:
        # Honor cancellation if triggered
        pass
    theater_outcome = job.run()

    # Build transcript tuples from either freshly run Whisper or cached transcript
    raw_transcript = list(theater_outcome.transcript or [])
    if not raw_transcript and existing_transcript:
        raw_transcript = list(existing_transcript)
    if not raw_transcript:
        cached = (project.cache or {}).get("theater_transcript")
        if isinstance(cached, dict) and isinstance(cached.get("segments"), list):
            raw_transcript = [s for s in cached["segments"] if s.get("accepted", True)]

    transcript_tuples = [
        (float(s.get("start", 0.0)), float(s.get("end", 0.0)), str(s.get("text", "") or ""))
        for s in raw_transcript
        if s.get("accepted", True)
    ]

    sections = list(theater_outcome.sections)
    outcome.activities = list(theater_outcome.activities)

    # Align musical_number sections to the cast reference worklist using
    # transcript + LRCLIB reference lyrics!
    worklist = identification.worklist
    musical_indices = [
        i for i, sec in enumerate(sections) if sec.section_type == "musical_number"
    ]
    musical_sections = [sections[i] for i in musical_indices]

    lyric_texts = {
        title: entry.plain_lyrics
        for title, entry in (identification.lyrics_by_title or {}).items()
        if entry.plain_lyrics
    }

    matched_map: dict[int, tuple[int, float, str]] = {}
    if worklist and worklist.items and musical_sections and transcript_tuples:
        alignment = align_theater_worklist(
            worklist.items,
            musical_sections,
            sample_rate,
            transcripts=transcript_tuples,
            reference_lyrics=lyric_texts,
        )
        for (item_i, m_idx), conf in zip(alignment.assignments, alignment.identity_confidence):
            sec_i = musical_indices[m_idx]
            ev = alignment.evidence.get((item_i, m_idx), "")
            matched_map[sec_i] = (item_i, float(conf), ev)

    # Optional multi-engine recognition on musical_number sections with Broadway/Jukebox guard:
    # NEVER confuse an original pop song with its Broadway arrangement!
    broadway_recs: dict[int, object] = {}
    if allow_fingerprint and musical_sections and not (cancel and cancel()):
        from ..audio.timeline import build_reader
        from ..metadata.multi_engine import (
            MultiEngineRecognizer,
            is_cast_recording,
            normalize_text,
            same_song,
        )

        paths = [s.path for s in project.sources if s.path]
        audio_path = paths[0] if paths else ""
        recognizer = MultiEngineRecognizer(
            cache_dir=cache_dir,
            use_acoustid=bool(audio_path),
            use_shazamio=use_shazamio,
            enrich_musicbrainz=False,
        )
        if recognizer.active_engines:
            reader = build_reader(project)
            try:
                seg_pairs = [
                    (sec.start_frame / float(sample_rate), sec.end_frame / float(sample_rate))
                    for sec in musical_sections
                ]
                recs = recognizer.identify_segments(
                    audio_path,
                    seg_pairs,
                    reader=reader,
                    sample_rate=sample_rate,
                    n_clips=2,
                    cancel=cancel,
                )
                for m_idx, rec in enumerate(recs):
                    sec_i = musical_indices[m_idx]
                    broadway_recs[sec_i] = rec
                    if rec.status in ("confirmed", "likely") and rec.title:
                        cast_ok = is_cast_recording(rec.artist, rec.album, show_title=identification.title)
                        if not cast_ok:
                            # Original pop song recognized inside a jukebox musical:
                            # keep it as a constituent song clue, NOT as the Broadway track's artist/title!
                            rec.constituent_songs.append(rec.label)
                            if worklist and sec_i not in matched_map:
                                for wi, it in enumerate(worklist.items):
                                    contains_hit = any(
                                        same_song("", c_song, "", rec.title)
                                        for c_song in (it.contains or [])
                                    )
                                    title_hit = same_song("", it.title, "", rec.title)
                                    if contains_hit or title_hit:
                                        matched_map[sec_i] = (
                                            wi,
                                            0.62,
                                            f"Constituent pop song '{rec.label}' matched Broadway number '{it.title}'.",
                                        )
                                        break
                        elif worklist and sec_i not in matched_map:
                            for wi, it in enumerate(worklist.items):
                                if same_song("", it.title, "", rec.title):
                                    matched_map[sec_i] = (
                                        wi,
                                        float(rec.confidence),
                                        f"Cast recording recognized as '{it.title}'.",
                                    )
                                    break
            finally:
                reader.close()

    # Build contiguous PlannedTracks covering [0, total_frames)
    edges = [0] + [int(s.frame) for s in theater_outcome.suggestions if 0 < int(s.frame) < total_frames] + [total_frames]
    edges = sorted(set(edges))

    def _section_for(s_f: int, e_f: int) -> int:
        best_i = -1
        best_ov = 0
        for i, sec in enumerate(sections):
            ov = min(e_f, sec.end_frame) - max(s_f, sec.start_frame)
            if ov > best_ov:
                best_ov = ov
                best_i = i
        return best_i

    sugg_by_frame = {int(s.frame): s for s in theater_outcome.suggestions}

    for idx in range(len(edges) - 1):
        s_f, e_f = edges[idx], edges[idx + 1]
        sec_i = _section_for(s_f, e_f)
        sec = sections[sec_i] if 0 <= sec_i < len(sections) else None
        sec_type = sec.section_type if sec else "unknown"
        sugg = sugg_by_frame.get(s_f)

        title = ""
        id_conf = 0.0
        w_idx = -1
        reasons = [sugg.reason] if sugg else ["Start of recording"]
        notes = ""
        rec_artist = ""
        rec_title = ""
        rec_status = ""
        rec_engines: list[str] = []
        track_flags: list[str] = []

        if sec_i in matched_map and worklist is not None:
            w_idx, id_conf, ev = matched_map[sec_i]
            item = worklist.items[w_idx]
            title = item.title
            notes = item.notes
            if ev:
                reasons.append(ev)
        elif sec_type == "musical_number":
            title = f"Musical number {idx + 1}"
        else:
            title = f"{sec_type.replace('_', ' ').capitalize()} {idx + 1}"

        rec = broadway_recs.get(sec_i)
        if rec is not None and getattr(rec, "status", "") not in ("", "unidentified"):
            from ..core.model import FLAG_RECOGNITION_CONFLICT

            rec_engines = list(rec.engines)
            if rec.constituent_songs:
                reasons.append(
                    f"Recognized original pop song(s) inside Broadway arrangement: "
                    f"{', '.join(rec.constituent_songs)} (kept distinct from cast number)."
                )
            else:
                rec_artist = rec.artist
                rec_title = rec.title
                rec_status = rec.status
                reasons.extend(rec.reasons)
            if rec.status == "conflict":
                track_flags.append(FLAG_RECOGNITION_CONFLICT)

        outcome.tracks.append(
            PlannedTrack(
                start_frame=s_f,
                end_frame=e_f,
                title=title,
                artist=identification.artist if sec_type == "musical_number" else "",
                section_type=sec_type,
                boundary_confidence=float(sugg.confidence) if sugg else 0.70,
                identity_confidence=id_conf,
                boundary_uncertain=bool(sugg.uncertain) if sugg else False,
                boundary_placement="metadata_broadway",
                worklist_index=w_idx,
                reasons=reasons,
                notes=notes,
                recognized_artist=rec_artist,
                recognized_title=rec_title,
                recognition_status=rec_status,
                recognition_engines=rec_engines,
                flags=track_flags,
            )
        )
        if idx > 0 and sugg is not None:
            outcome.suggestions.append(sugg)

    outcome.diagnostics = {
        "used_yamnet": theater_outcome.used_model,
        "transcribed": bool(transcript_tuples),
        "sections": len(sections),
        "musical_numbers": len(musical_sections),
        "named_numbers": len(matched_map),
        "lyrics_fetched": len(lyric_texts),
    }
    outcome.notes.append(
        f"Broadway analysis: {len(musical_sections)} musical number(s) across {len(sections)} sections; "
        f"named {len(matched_map)} using cast reference + LRCLIB lyrics ({len(lyric_texts)} fetched) "
        + ("+ transcript." if transcript_tuples else "(run transcription to confirm song titles).")
    )
    outcome.elapsed_seconds = time.time() - started
    return outcome
