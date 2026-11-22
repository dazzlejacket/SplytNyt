"""Optional ShazamIO (Experimental) song recognition engine.

Design & compliance rules:
* **Strictly optional & opt-in**: Disabled by default. Never enabled
  automatically without explicit user consent.
* **Not bundled in public EXEs**: Because ShazamIO talks to Shazam's unofficial
  online endpoints (no API key required, subject to Shazam's Terms of Service,
  rate limits, and endpoint changes), ``shazamio`` and ``shazamio-core`` are
  excluded from PyInstaller builds.
* **No Python or CLI required for end users**: Users can install or uninstall
  the optional dependency from inside *Settings -> Music Services*. SplytNyt
  downloads the official MIT-licensed PyPI binary wheels directly, verifies
  their SHA-256 digests, and extracts them into the user's local app-data
  directory (``~/.splytnyt/optional_engines/shazamio/site-packages``).
* **Survivable & polite**: Missing dependency, disabled toggle, rate limits
  (HTTP 429), or network/endpoint failures never crash SplytNyt — everything
  else keeps working. Requests are rate-limited and disk-cached by audio
  content hash.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from . import ProviderError, ProviderRateLimited, ProviderUnavailable
from .base import CACHE_TTL_SECONDS, RateLimiter, default_cache_dir, user_agent

SHAZAMIO = "ShazamIO"
MINIMUM_INTERVAL = 1.6  # Polite spacing to avoid Shazam 429 rate limits
CLIP_SAMPLE_RATE = 16000
DEFAULT_CLIP_SECONDS = 10.0

SERVICE_NOTICE = (
    "ShazamIO (Experimental) uses Shazam's unofficial online recognition service "
    "(no API key required). It is not bundled with SplytNyt; installing it "
    "downloads open-source packages from PyPI into your local user folder. "
    "Because the online service is unofficial, it may rate-limit or stop working "
    "without notice. SplytNyt works normally if ShazamIO is disabled or unavailable."
)

# Direct + transitive packages required for shazamio on Windows/desktop Python.
# Pure-Python wheel installer resolves exact compatible binary wheels from PyPI.
_SHAZAMIO_PACKAGES: tuple[str, ...] = (
    "shazamio",
    "shazamio-core",
    "aiohttp",
    "aiohttp-retry",
    "aiohappyeyeballs",
    "aiosignal",
    "attrs",
    "frozenlist",
    "multidict",
    "propcache",
    "yarl",
    "idna",
    "pydantic",
    "pydantic-core",
    "annotated-types",
    "typing-extensions",
    "typing-inspection",
)


# ----------------------------------------------------------------------
# Persistent user settings (opt-in toggle, works headless and in GUI)
# ----------------------------------------------------------------------

def settings_file_path() -> str:
    override = (os.environ.get("SPLYTNYT_SETTINGS_PATH") or "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(os.path.expanduser("~"), ".splytnyt", "settings.json")


def _load_settings() -> dict[str, Any]:
    path = settings_file_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_settings(data: dict[str, Any]) -> None:
    path = settings_file_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def is_enabled() -> bool:
    """True only when the user explicitly opted in to ShazamIO (Experimental)."""
    return bool(_load_settings().get("shazamio_enabled", False))


def set_enabled(enabled: bool) -> None:
    """Persist the user's explicit Enable/Disable choice for ShazamIO."""
    data = _load_settings()
    data["shazamio_enabled"] = bool(enabled)
    _save_settings(data)


# ----------------------------------------------------------------------
# Optional engine location & import management
# ----------------------------------------------------------------------

def optional_engine_root() -> str:
    override = (os.environ.get("SPLYTNYT_OPTIONAL_ENGINES_DIR") or "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(os.path.expanduser("~"), ".splytnyt", "optional_engines", "shazamio")


def optional_site_packages_dir() -> str:
    return os.path.join(optional_engine_root(), "site-packages")


def _ensure_optional_path_registered() -> bool:
    site = optional_site_packages_dir()
    if not os.path.isdir(site):
        return False
    if site not in sys.path:
        sys.path.insert(0, site)
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        for sub in (site, os.path.join(site, "shazamio_core")):
            if os.path.isdir(sub):
                try:
                    os.add_dll_directory(sub)
                except OSError:
                    pass
    return True


def _pending_cleanup_path() -> str:
    return os.path.join(optional_engine_root(), ".pending_cleanup")


def cleanup_pending_uninstall() -> None:
    """Finish deleting any optional-engine files that were locked on Windows before restart."""
    root = optional_engine_root()
    marker = _pending_cleanup_path()
    if os.path.isfile(marker):
        shutil.rmtree(root, ignore_errors=True)


def _evict_shazamio_modules() -> None:
    for mod_name in list(sys.modules):
        if mod_name == "shazamio" or mod_name.startswith("shazamio.") or mod_name == "shazamio_core" or mod_name.startswith("shazamio_core."):
            sys.modules.pop(mod_name, None)


def is_installed() -> bool:
    """Return True if ``shazamio`` is installed and importable."""
    # If a previous uninstall left locked Windows .pyd DLLs awaiting restart,
    # never report the engine as installed.
    if os.path.isfile(_pending_cleanup_path()):
        return False
    site = optional_site_packages_dir()
    manifest = os.path.join(optional_engine_root(), "installed.json")
    # If the optional engine folder exists on disk without installed.json,
    # treat it as uninstalled/incomplete unless shazamio is in the base env.
    if os.path.isdir(site) and os.path.isfile(manifest):
        _ensure_optional_path_registered()
    elif site in sys.path:
        try:
            sys.path.remove(site)
        except ValueError:
            pass
    importlib.invalidate_caches()
    try:
        import shazamio  # noqa: F401
        import shazamio_core  # noqa: F401
        return True
    except Exception:
        return False


def installed_version() -> str:
    """Return installed shazamio version string, or ``""`` if not installed."""
    if not is_installed():
        return ""
    manifest = os.path.join(optional_engine_root(), "installed.json")
    if os.path.isfile(manifest):
        try:
            with open(manifest, "r", encoding="utf-8") as fh:
                info = json.load(fh)
            ver = info.get("packages", {}).get("shazamio", "")
            if ver:
                return str(ver)
        except Exception:
            pass
    try:
        from importlib import metadata as il_meta
        return str(il_meta.version("shazamio"))
    except Exception:
        return "installed"


def is_ready() -> bool:
    """True when ShazamIO is BOTH explicitly enabled by the user AND installed."""
    return is_enabled() and is_installed()


def engine_status() -> dict[str, Any]:
    installed = is_installed()
    enabled = is_enabled()
    return {
        "installed": installed,
        "enabled": enabled,
        "ready": installed and enabled,
        "version": installed_version() if installed else "",
        "install_dir": optional_site_packages_dir(),
        "notice": SERVICE_NOTICE,
    }


# ----------------------------------------------------------------------
# Pure-Python PyPI wheel installer (no Python / pip / CLI required!)
# ----------------------------------------------------------------------

def _platform_tags() -> list[str]:
    if sys.platform == "win32":
        arch = platform.machine().lower()
        if "64" in arch or sys.maxsize > 2**32:
            return ["win_amd64", "any"]
        return ["win32", "any"]
    if sys.platform == "darwin":
        return ["macosx", "any"]
    return ["manylinux", "linux", "any"]


def _wheel_compatibility_score(filename: str) -> int:
    """Score a PyPI wheel filename for the running Python interpreter.

    Returns 0 if incompatible, or a positive priority score (higher is better).
    """
    if not filename.endswith(".whl"):
        return 0
    stem = filename[:-4]
    parts = stem.split("-")
    if len(parts) < 5:
        return 0
    py_tag, abi_tag, plat_tag = parts[-3], parts[-2], parts[-1]

    # Platform check
    plat_tags = _platform_tags()
    plat_score = 0
    for idx, p in enumerate(plat_tags):
        if p == "any" and plat_tag == "any":
            plat_score = 10
            break
        if p != "any" and p in plat_tag:
            plat_score = 30 - idx * 5
            break
    if plat_score == 0:
        return 0

    major = sys.version_info.major
    minor = sys.version_info.minor
    exact_cp = f"cp{major}{minor}"

    # ABI & Python tag check
    if abi_tag == "none":
        if "py3" in py_tag or f"cp{major}" in py_tag:
            return plat_score + 20
        return 0

    if abi_tag == exact_cp:
        return plat_score + 100

    if abi_tag == "abi3":
        # Compatible if wheel's minimum cp3X <= current cp3Y
        for sub in py_tag.split("."):
            if sub.startswith(f"cp{major}"):
                try:
                    req_minor = int(sub[3:])
                    if req_minor <= minor:
                        return plat_score + 80
                except ValueError:
                    pass
        return 0

    return 0


def _pick_wheel_from_pypi(package_name: str, timeout: float = 20.0) -> tuple[str, str, str, str]:
    """Query PyPI JSON API and return ``(version, filename, url, sha256)``."""
    api_url = f"https://pypi.org/pypi/{package_name}/json"
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": user_agent(), "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    version = str(data.get("info", {}).get("version") or "")
    urls = data.get("urls") or []
    best: Optional[tuple[int, dict]] = None
    for entry in urls:
        if entry.get("packagetype") != "bdist_wheel":
            continue
        fname = str(entry.get("filename") or "")
        sc = _wheel_compatibility_score(fname)
        if sc > 0 and (best is None or sc > best[0]):
            best = (sc, entry)

    if best is None:
        raise ProviderUnavailable(
            f"No compatible binary wheel found on PyPI for '{package_name}' "
            f"(Python {sys.version_info.major}.{sys.version_info.minor}, {sys.platform})."
        )

    chosen = best[1]
    sha256 = str((chosen.get("digests") or {}).get("sha256") or "")
    return version, str(chosen["filename"]), str(chosen["url"]), sha256


def install_shazamio(
    progress: Optional[Callable[[float, str], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> tuple[bool, str]:
    """Download and install ShazamIO + dependencies into SplytNyt's user folder.

    Works identically from source and from the packaged Windows EXE without
    requiring the user to install Python or run command-line tools.
    """
    root = optional_engine_root()
    cleanup_pending_uninstall()
    pending = _pending_cleanup_path()
    if os.path.isfile(pending):
        try:
            os.remove(pending)
        except OSError:
            pass
    target_site = optional_site_packages_dir()
    staging = os.path.join(root, f"staging_{int(time.time())}")
    os.makedirs(staging, exist_ok=True)

    installed_versions: dict[str, str] = {}
    total = len(_SHAZAMIO_PACKAGES)

    try:
        for idx, pkg in enumerate(_SHAZAMIO_PACKAGES):
            if cancel and cancel():
                shutil.rmtree(staging, ignore_errors=True)
                return False, "Installation cancelled."

            if progress:
                progress(idx / float(total + 1), f"Downloading {pkg} ({idx + 1}/{total})...")

            version, fname, url, expected_sha = _pick_wheel_from_pypi(pkg)
            req = urllib.request.Request(url, headers={"User-Agent": user_agent()})
            with urllib.request.urlopen(req, timeout=45.0) as resp:
                wheel_bytes = resp.read()

            if expected_sha:
                actual_sha = hashlib.sha256(wheel_bytes).hexdigest()
                if actual_sha.lower() != expected_sha.lower():
                    raise ProviderError(f"Checksum mismatch while downloading {fname}.")

            with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as zf:
                for member in zf.infolist():
                    # Guard against path traversal
                    norm = os.path.normpath(member.filename)
                    if norm.startswith("..") or os.path.isabs(norm):
                        continue
                    zf.extract(member, staging)

            installed_versions[pkg] = version

        if progress:
            progress(total / float(total + 1), "Finalizing ShazamIO installation...")

        backup = target_site + ".old"
        if os.path.exists(backup):
            shutil.rmtree(backup, ignore_errors=True)
        if os.path.exists(target_site):
            os.replace(target_site, backup)
        os.replace(staging, target_site)
        if os.path.exists(backup):
            shutil.rmtree(backup, ignore_errors=True)

        manifest_path = os.path.join(root, "installed.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "packages": installed_versions,
                },
                fh,
                indent=2,
            )

        _ensure_optional_path_registered()
        importlib.invalidate_caches()
        # Verify actual import
        import shazamio  # noqa: F401
        import shazamio_core  # noqa: F401

        if progress:
            progress(1.0, "ShazamIO installed.")
        ver = installed_versions.get("shazamio", "")
        return True, f"Installed ShazamIO {ver} (optional user engine)."
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        return False, f"Could not install ShazamIO: {exc}"


def uninstall_shazamio() -> tuple[bool, str]:
    """Remove the locally installed optional ShazamIO engine and disable it."""
    set_enabled(False)
    root = optional_engine_root()
    site = optional_site_packages_dir()
    manifest = os.path.join(root, "installed.json")
    if os.path.isfile(manifest):
        try:
            os.remove(manifest)
        except OSError:
            pass
    if site in sys.path:
        try:
            sys.path.remove(site)
        except ValueError:
            pass
    _evict_shazamio_modules()
    importlib.invalidate_caches()
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
        # On Windows, a loaded Rust .pyd DLL may remain locked until process exit.
        # Mark it for automatic cleanup on next launch and report honestly.
        if os.path.isdir(root):
            try:
                with open(_pending_cleanup_path(), "w", encoding="utf-8") as fh:
                    fh.write("pending\n")
            except OSError:
                pass
            return (
                True,
                "Disabled and removed ShazamIO (locked native library will finish cleaning up on restart).",
            )
    return True, "Removed optional ShazamIO engine and disabled it."


# ----------------------------------------------------------------------
# Match data model & audio clip encoding
# ----------------------------------------------------------------------

@dataclass
class ShazamMatch:
    """One song identification returned by ShazamIO."""

    title: str
    artist: str = ""
    album: str = ""
    label: str = ""
    released: str = ""
    isrc: str = ""
    shazam_key: str = ""
    #: Offset (seconds) inside the studio track where our sampled clip began.
    offset_seconds: float = 0.0
    timeskew: float = 0.0
    frequencyskew: float = 0.0
    match_count: int = 1
    score: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "label": self.label,
            "released": self.released,
            "isrc": self.isrc,
            "shazam_key": self.shazam_key,
            "offset_seconds": self.offset_seconds,
            "timeskew": self.timeskew,
            "frequencyskew": self.frequencyskew,
            "match_count": self.match_count,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShazamMatch":
        return cls(
            title=str(data.get("title") or ""),
            artist=str(data.get("artist") or ""),
            album=str(data.get("album") or ""),
            label=str(data.get("label") or ""),
            released=str(data.get("released") or ""),
            isrc=str(data.get("isrc") or ""),
            shazam_key=str(data.get("shazam_key") or ""),
            offset_seconds=float(data.get("offset_seconds") or 0.0),
            timeskew=float(data.get("timeskew") or 0.0),
            frequencyskew=float(data.get("frequencyskew") or 0.0),
            match_count=int(data.get("match_count") or 1),
            score=float(data.get("score") or 0.85),
        )


def _resample_mono_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if arr.size == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float32)
    if sample_rate == CLIP_SAMPLE_RATE:
        return arr
    try:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(int(sample_rate), CLIP_SAMPLE_RATE)
        return resample_poly(arr, CLIP_SAMPLE_RATE // g, int(sample_rate) // g).astype(np.float32)
    except Exception:
        count = max(1, int(round(arr.size * CLIP_SAMPLE_RATE / float(sample_rate))))
        pos = np.linspace(0.0, max(0, arr.size - 1), num=count)
        return np.interp(pos, np.arange(arr.size), arr).astype(np.float32)


def encode_pcm_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Convert mono/stereo float32 audio into compact 16 kHz mono 16-bit WAV bytes."""
    mono = _resample_mono_16k(samples, sample_rate)
    if mono.size == 0:
        return b""
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(CLIP_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _parse_shazam_response(payload: Any) -> Optional[ShazamMatch]:
    if not isinstance(payload, dict):
        return None
    track = payload.get("track")
    if not isinstance(track, dict):
        return None
    title = str(track.get("title") or "").strip()
    if not title:
        return None
    artist = str(track.get("subtitle") or "").strip()
    isrc = str(track.get("isrc") or "").strip()
    key = str(track.get("key") or "").strip()

    album = ""
    label = ""
    released = ""
    for sec in track.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        for item in sec.get("metadata") or []:
            if not isinstance(item, dict):
                continue
            m_title = str(item.get("title") or "").strip().lower()
            m_text = str(item.get("text") or "").strip()
            if m_title == "album" and not album:
                album = m_text
            elif m_title == "label" and not label:
                label = m_text
            elif m_title == "released" and not released:
                released = m_text

    matches = payload.get("matches") or []
    offset_s = 0.0
    timeskew = 0.0
    freqskew = 0.0
    if matches and isinstance(matches[0], dict):
        offset_s = float(matches[0].get("offset") or 0.0)
        timeskew = float(matches[0].get("timeskew") or 0.0)
        freqskew = float(matches[0].get("frequencyskew") or 0.0)

    # Shazam only returns a `track` when its fingerprint signature matches.
    # Base score is high (0.85), slightly boosted for multiple signature hits
    # and gently reduced if time/frequency skew is unusually extreme (>5%).
    base_score = 0.85
    if len(matches) >= 2:
        base_score += 0.05
    if abs(timeskew) > 0.05 or abs(freqskew) > 0.05:
        base_score -= 0.08
    score = float(np.clip(base_score, 0.65, 0.95))

    return ShazamMatch(
        title=title,
        artist=artist,
        album=album,
        label=label,
        released=released,
        isrc=isrc,
        shazam_key=key,
        offset_seconds=offset_s,
        timeskew=timeskew,
        frequencyskew=freqskew,
        match_count=max(1, len(matches)),
        score=score,
    )


# ----------------------------------------------------------------------
# Provider class
# ----------------------------------------------------------------------

class ShazamIoProvider:
    """Rate-limited, disk-cached wrapper around the optional ``shazamio`` library."""

    name = SHAZAMIO

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        min_interval: float = MINIMUM_INTERVAL,
        require_enabled: bool = True,
        max_retries: int = 2,
    ) -> None:
        self.cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
        self.limiter = RateLimiter(min_interval)
        self.require_enabled = require_enabled
        self.max_retries = max(0, int(max_retries))

    @property
    def available(self) -> bool:
        if self.require_enabled and not is_enabled():
            return False
        return is_installed()

    # ------------------------------------------------------------------
    # Disk cache keyed by audio content SHA-256
    # ------------------------------------------------------------------
    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"ShazamIO-{key[:24]}.json")

    def _read_cache(self, key: str) -> tuple[bool, Optional[ShazamMatch]]:
        """Return ``(hit, match_or_none)``."""
        path = self._cache_path(key)
        if not path or not os.path.isfile(path):
            return False, None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return False, None
        stamp = float(payload.get("cached_at", 0.0))
        if time.time() - stamp > CACHE_TTL_SECONDS:
            return False, None
        raw = payload.get("match")
        if raw is None:
            return True, None
        return True, ShazamMatch.from_dict(raw)

    def _write_cache(self, key: str, match: Optional[ShazamMatch]) -> None:
        path = self._cache_path(key)
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "cached_at": time.time(),
                        "match": match.to_dict() if match is not None else None,
                    },
                    fh,
                )
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Async execution helper (safe from worker threads & Qt event loops)
    # ------------------------------------------------------------------
    @staticmethod
    async def _recognize_wav_bytes_async(wav_bytes: bytes) -> dict:
        _ensure_optional_path_registered()
        from shazamio import Shazam

        shazam = Shazam()
        try:
            try:
                return await asyncio.wait_for(shazam.recognize(wav_bytes), timeout=15.0)
            except TypeError:
                # Older shazamio versions expect a file path rather than raw bytes
                tmp_path = ""
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tf.write(wav_bytes)
                        tmp_path = tf.name
                    return await asyncio.wait_for(shazam.recognize(tmp_path), timeout=15.0)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
        finally:
            # Close aiohttp RetryClient/ClientSession cleanly so no resource warnings leak
            http_client = getattr(shazam, "http_client", None)
            if http_client is not None:
                close_direct = getattr(http_client, "close", None)
                if callable(close_direct):
                    try:
                        res = close_direct()
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception:
                        pass
                else:
                    for attr in ("_retry_client", "_client", "client", "_session", "session"):
                        inner = getattr(http_client, attr, None)
                        close_fn = getattr(inner, "close", None)
                        if callable(close_fn):
                            try:
                                res = close_fn()
                                if asyncio.iscoroutine(res):
                                    await res
                            except Exception:
                                pass

    def _run_async(self, coro) -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.run_until_complete(asyncio.sleep(0.05))
            except Exception:
                pass
            loop.close()

    # ------------------------------------------------------------------
    # Public recognition API
    # ------------------------------------------------------------------
    def recognize_wav_bytes(
        self,
        wav_bytes: bytes,
        use_cache: bool = True,
    ) -> Optional[ShazamMatch]:
        """Recognize one short WAV clip via ShazamIO (rate-limited & cached)."""
        if not wav_bytes:
            return None
        if self.require_enabled and not is_enabled():
            return None
        if not is_installed():
            return None

        cache_key = hashlib.sha256(wav_bytes).hexdigest()
        if use_cache:
            hit, cached = self._read_cache(cache_key)
            if hit:
                return cached

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                raw = self._run_async(self._recognize_wav_bytes_async(wav_bytes))
                match = _parse_shazam_response(raw)
                if use_cache:
                    self._write_cache(cache_key, match)
                return match
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                is_rate = "429" in msg or "rate" in msg or "too many" in msg
                if is_rate and attempt < self.max_retries:
                    time.sleep(min(8.0, 2.0 * (attempt + 1)))
                    continue
                if is_rate:
                    raise ProviderRateLimited(
                        "ShazamIO rate-limited the request (429). Slowing down or skipping remaining clips."
                    ) from exc
                if attempt < self.max_retries and ("timeout" in msg or "503" in msg or "connection" in msg):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise ProviderError(f"ShazamIO recognition failed: {exc}") from exc

        if last_exc is not None:
            raise ProviderError(f"ShazamIO recognition failed: {last_exc}") from last_exc
        return None

    def recognize_pcm(
        self,
        samples: np.ndarray,
        sample_rate: int,
        use_cache: bool = True,
    ) -> Optional[ShazamMatch]:
        """Recognize an in-memory float32 audio slice."""
        wav_bytes = encode_pcm_wav_bytes(samples, sample_rate)
        return self.recognize_wav_bytes(wav_bytes, use_cache=use_cache)

    def recognize_reader_slice(
        self,
        reader,
        sample_rate: int,
        start_seconds: float,
        duration_seconds: float = DEFAULT_CLIP_SECONDS,
        use_cache: bool = True,
    ) -> Optional[ShazamMatch]:
        """Read ``[start_seconds, start_seconds + duration_seconds]`` from a TimelineReader and recognize."""
        if reader is None or sample_rate <= 0:
            return None
        s_frame = max(0, int(round(start_seconds * sample_rate)))
        count = max(1, int(round(max(4.0, min(14.0, duration_seconds)) * sample_rate)))
        data = reader.read(s_frame, count)
        if data is None or getattr(data, "size", 0) == 0:
            return None
        return self.recognize_pcm(data, sample_rate, use_cache=use_cache)

    def recognize_file_slice(
        self,
        audio_path: str,
        start_seconds: float,
        duration_seconds: float = DEFAULT_CLIP_SECONDS,
        use_cache: bool = True,
    ) -> Optional[ShazamMatch]:
        """Decode a short slice directly from an audio file and recognize via ShazamIO."""
        if not audio_path or not os.path.isfile(audio_path):
            return None
        from ..audio.decode import AudioSource

        with AudioSource(audio_path) as src:
            sr = src.sample_rate or 44100
            s_frame = max(0, int(round(start_seconds * sr)))
            count = max(1, int(round(max(4.0, min(14.0, duration_seconds)) * sr)))
            data = src.read(s_frame, count)
        if data is None or data.size == 0:
            return None
        return self.recognize_pcm(data, sr, use_cache=use_cache)

    def test_connection(self) -> tuple[bool, str]:
        """Verify ShazamIO is installed and can reach the unofficial Shazam endpoint."""
        if not is_installed():
            return False, "ShazamIO is not installed yet. Click 'Install Optional Engine' first."
        # Send a 5-second quiet sine tone; Shazam responds with an empty match dict (HTTP 200)
        # which proves both local Rust fingerprinting (shazamio-core) and online reachability work.
        t = np.linspace(0.0, 5.0, CLIP_SAMPLE_RATE * 5, endpoint=False, dtype=np.float32)
        probe = 0.05 * np.sin(2.0 * np.pi * 440.0 * t)
        wav_bytes = encode_pcm_wav_bytes(probe, CLIP_SAMPLE_RATE)
        try:
            self._run_async(self._recognize_wav_bytes_async(wav_bytes))
        except Exception as exc:
            return False, f"ShazamIO endpoint check failed: {exc}"
        ver = installed_version()
        state = "enabled" if is_enabled() else "installed (currently disabled)"
        return True, f"Connected to ShazamIO {ver} ({state}; unofficial service)."
