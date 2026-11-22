"""AcoustID / Chromaprint audio fingerprinting.

Generates Chromaprint fingerprints locally using ``ffmpeg`` (with the built-in
``chromaprint`` muxer) or ``fpcalc`` and queries the AcoustID lookup API using
SplytNyt's built-in official application key (or an optional user override).

Rules:
* SplytNyt ships with its official AcoustID application key built in so users
  do not need to register for their own key.
* Fingerprinting is optional and survivable: missing ``ffmpeg``/``fpcalc`` or
  network failure never crashes SplytNyt.
* Never invent song identities when fingerprinting fails or returns a low-score
  match.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..core.credentials import KEY_ACOUSTID, get_credential
from . import ProviderConfigurationError, ProviderError
from .base import HttpClient, default_cache_dir

ACOUSTID = "AcoustID"
API_ROOT = "https://api.acoustid.org/v2/lookup"
API_KEY_ENVIRONMENT = "ACOUSTID_API_KEY"
#: Official SplytNyt AcoustID application key (not a personal user key).
DEFAULT_APP_KEY = "ztpTKhkIHy"
MINIMUM_INTERVAL = 0.4  # AcoustID allows 3 requests / second
MIN_MATCH_SCORE = 0.55


@dataclass
class FingerprintMatch:
    """One song identification returned by AcoustID."""

    score: float
    title: str
    artist: str = ""
    recording_id: str = ""
    release_group_id: str = ""
    release_title: str = ""
    duration_seconds: float = 0.0


@dataclass
class FingerprintWindow:
    """Result of fingerprinting one time slice of a long recording."""

    start_seconds: float
    end_seconds: float
    match: Optional[FingerprintMatch] = None
    fingerprint: str = ""
    error: str = ""


def api_key() -> str:
    """Return the active AcoustID key: env var -> stored override -> built-in app key."""
    env_val = (os.environ.get(API_KEY_ENVIRONMENT, "") or "").strip()
    if env_val:
        return env_val
    stored = get_credential(KEY_ACOUSTID)
    if stored:
        return stored
    return DEFAULT_APP_KEY


def has_custom_api_key() -> bool:
    """True if the user configured an explicit override key."""
    return bool(
        (os.environ.get(API_KEY_ENVIRONMENT, "") or "").strip()
        or get_credential(KEY_ACOUSTID)
    )


def has_api_key() -> bool:
    return bool(api_key())


def chromaprint_backend() -> Optional[str]:
    """Return ``'ffmpeg'``, ``'fpcalc'``, or ``None`` if neither is available."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    if shutil.which("fpcalc"):
        return "fpcalc"
    return None


def fingerprint_slice(
    audio_path: str,
    start_seconds: float = 0.0,
    duration_seconds: float = 120.0,
) -> tuple[str, int]:
    """Compute a Chromaprint base64 fingerprint for ``[start_seconds, start+dur]``.

    Returns ``(fingerprint_b64, actual_duration_int)`` or ``("", 0)`` on failure.
    """
    if not audio_path or not os.path.exists(audio_path):
        return "", 0
    dur = max(10.0, min(120.0, float(duration_seconds)))
    start = max(0.0, float(start_seconds))

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-v", "error",
            "-ss", f"{start:.3f}",
            "-t", f"{dur:.3f}",
            "-i", audio_path,
            "-f", "chromaprint",
            "-fp_format", "base64",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if proc.returncode == 0:
                fp = proc.stdout.decode("ascii", errors="ignore").strip()
                if fp:
                    return fp, int(round(dur))
        except (OSError, subprocess.TimeoutExpired):
            pass

    fpcalc = shutil.which("fpcalc")
    if fpcalc and start == 0.0:
        try:
            proc = subprocess.run(
                [fpcalc, "-length", str(int(round(dur))), audio_path],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode == 0:
                fp = ""
                out_dur = int(round(dur))
                for line in proc.stdout.splitlines():
                    if line.startswith("FINGERPRINT="):
                        fp = line.split("=", 1)[1].strip()
                    elif line.startswith("DURATION="):
                        try:
                            out_dur = int(round(float(line.split("=", 1)[1].strip())))
                        except ValueError:
                            pass
                if fp:
                    return fp, out_dur
        except (OSError, subprocess.TimeoutExpired):
            pass

    return "", 0


class AcoustIdProvider:
    """Looks up Chromaprint fingerprints against the AcoustID web service."""

    name = ACOUSTID

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        client_key: Optional[str] = None,
    ) -> None:
        self._key = (client_key if client_key is not None else api_key()).strip()
        self.client = HttpClient(
            ACOUSTID,
            cache_dir=cache_dir if cache_dir is not None else default_cache_dir(),
            min_interval=MINIMUM_INTERVAL,
            timeout=15.0,
        )

    @property
    def configured(self) -> bool:
        return bool(self._key)

    def test_connection(self) -> tuple[bool, str]:
        """Verify that the configured AcoustID key is accepted by api.acoustid.org."""
        if not self._key:
            return False, "No AcoustID application key configured."
        # AcoustID validates the client key before checking fingerprint format,
        # OR we can send a valid tiny lookup by trackid/meta to verify key + connectivity.
        try:
            data = self.client.post_form_json(
                API_ROOT,
                {
                    "client": self._key,
                    "trackid": "9e7ada7d-b887-4e13-b6a4-7e1db0d98b2f",
                    "meta": "recordings",
                },
                use_cache=False,
            )
        except ProviderError as exc:
            return False, str(exc)
        if isinstance(data, dict) and data.get("status") == "ok":
            backend = chromaprint_backend()
            if backend:
                return True, f"Connected to AcoustID (Chromaprint via {backend})."
            return True, "Connected to AcoustID (install ffmpeg or fpcalc for local fingerprinting)."
        err = (data or {}).get("error", {}).get("message", "Unexpected response")
        return False, f"AcoustID rejected the request: {err}"

    def lookup_fingerprint(
        self,
        fingerprint: str,
        duration_seconds: int,
        min_score: float = MIN_MATCH_SCORE,
    ) -> list[FingerprintMatch]:
        """Query AcoustID for one Chromaprint fingerprint."""
        if not self._key:
            raise ProviderConfigurationError(
                f"AcoustID needs an API key in {API_KEY_ENVIRONMENT} for online audio "
                "fingerprinting. SplytNyt will continue without inventing track names."
            )
        if not fingerprint or duration_seconds <= 0:
            return []

        data = self.client.post_form_json(
            API_ROOT,
            {
                "client": self._key,
                "meta": "recordings releasegroups",
                "duration": str(int(duration_seconds)),
                "fingerprint": fingerprint,
            },
        )
        if not isinstance(data, dict) or data.get("status") != "ok":
            err = (data or {}).get("error", {}).get("message", "Unknown AcoustID error")
            raise ProviderError(f"AcoustID lookup failed: {err}")

        matches: list[FingerprintMatch] = []
        for result in data.get("results") or []:
            score = float(result.get("score") or 0.0)
            if score < min_score:
                continue
            for rec in result.get("recordings") or []:
                title = (rec.get("title") or "").strip()
                if not title:
                    continue
                artists = ", ".join(
                    a.get("name", "") for a in (rec.get("artists") or []) if a.get("name")
                )
                rg_id = ""
                rg_title = ""
                rgs = rec.get("releasegroups") or []
                if rgs:
                    rg_id = rgs[0].get("id", "") or ""
                    rg_title = rgs[0].get("title", "") or ""
                matches.append(
                    FingerprintMatch(
                        score=score,
                        title=title,
                        artist=artists,
                        recording_id=rec.get("id", "") or "",
                        release_group_id=rg_id,
                        release_title=rg_title,
                        duration_seconds=float(rec.get("duration") or 0.0),
                    )
                )
        matches.sort(key=lambda m: -m.score)
        return matches

    def identify_segment(
        self,
        audio_path: str,
        start_seconds: float,
        end_seconds: float,
        offsets: Sequence[float] = (0.0, 5.0),
        min_score: float = MIN_MATCH_SCORE,
    ) -> Optional[FingerprintMatch]:
        """Identify one candidate track segment ``[start_seconds, end_seconds]``.

        AcoustID matches Chromaprint fingerprints from the beginning of a track
        against the track's full duration. Probing a small set of start offsets
        handles leading vinyl silence or DJ crossfade overlaps without inventing
        matches.
        """
        if not self.configured or not chromaprint_backend():
            return None
        seg_dur = float(end_seconds) - float(start_seconds)
        if seg_dur < 25.0:
            return None

        best_match: Optional[FingerprintMatch] = None
        fp_len = min(120.0, max(20.0, seg_dur - 2.0))
        q_dur = int(round(seg_dur))

        for off in offsets:
            probe_start = max(0.0, float(start_seconds) + float(off))
            if probe_start + 15.0 >= end_seconds:
                continue
            fp, _ = fingerprint_slice(audio_path, probe_start, min(fp_len, end_seconds - probe_start))
            if not fp:
                continue
            try:
                hits = self.lookup_fingerprint(fp, q_dur, min_score=min_score)
            except ProviderError:
                return best_match
            if hits:
                cand = hits[0]
                if best_match is None or cand.score > best_match.score:
                    best_match = cand
                if best_match.score >= 0.80:
                    return best_match
        return best_match

    def identify_segments(
        self,
        audio_path: str,
        segments_seconds: Sequence[tuple[float, float]],
        offsets: Sequence[float] = (0.0, 6.0),
        min_score: float = MIN_MATCH_SCORE,
        progress: Optional[Callable[[float, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> list[FingerprintWindow]:
        """Identify a sequence of detected track segments via AcoustID."""
        if not self.configured or not chromaprint_backend():
            return []
        windows: list[FingerprintWindow] = []
        total = max(1, len(segments_seconds))
        for idx, (start_s, end_s) in enumerate(segments_seconds):
            if cancel and cancel():
                break
            match = self.identify_segment(
                audio_path,
                start_s,
                end_s,
                offsets=offsets,
                min_score=min_score,
            )
            windows.append(FingerprintWindow(start_s, end_s, match=match))
            if progress:
                progress((idx + 1) / float(total), f"Fingerprinting track {idx + 1}/{total}...")
        return windows

    def scan_recording(
        self,
        audio_path: str,
        total_duration_seconds: float,
        window_seconds: float = 90.0,
        step_seconds: float = 180.0,
        max_windows: int = 20,
        assumed_track_duration: int = 240,
        progress: Optional[Callable[[float, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> list[FingerprintWindow]:
        """Fingerprint sliding windows across a long recording.

        Returns a list of ``FingerprintWindow`` entries. Never invents matches.
        """
        if not self.configured or not chromaprint_backend():
            return []
        if total_duration_seconds <= 15.0:
            return []

        step = max(30.0, float(step_seconds))
        if total_duration_seconds / step > max_windows:
            step = total_duration_seconds / float(max_windows)

        starts: list[float] = []
        pos = 2.0
        while pos + 15.0 < total_duration_seconds:
            starts.append(pos)
            pos += step

        windows: list[FingerprintWindow] = []
        for idx, start_s in enumerate(starts):
            if cancel and cancel():
                break
            dur_s = min(window_seconds, total_duration_seconds - start_s)
            fp, actual_dur = fingerprint_slice(audio_path, start_s, dur_s)
            if not fp:
                windows.append(FingerprintWindow(start_s, start_s + dur_s, error="fingerprint failed"))
                continue
            q_dur = int(min(total_duration_seconds, max(actual_dur, assumed_track_duration)))
            try:
                hits = self.lookup_fingerprint(fp, q_dur)
                best = hits[0] if hits else None
                windows.append(FingerprintWindow(start_s, start_s + dur_s, match=best, fingerprint=fp))
            except ProviderError as exc:
                windows.append(FingerprintWindow(start_s, start_s + dur_s, fingerprint=fp, error=str(exc)))
                break
            if progress:
                progress((idx + 1) / max(1, len(starts)), f"Fingerprinting {idx + 1}/{len(starts)}")
        return windows
