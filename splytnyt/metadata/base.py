"""The shared machinery every provider needs: HTTP, rate limiting, caching.

Keeping this in one place means the politeness rules - a real User-Agent, a
minimum interval between requests, an on-disk cache - are the same for every
provider, and they are enforced rather than left to the user.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from . import ProviderError, ProviderRateLimited, ProviderUnavailable

#: How long a cached answer stays fresh, in seconds.
CACHE_TTL_SECONDS = 7 * 24 * 3600

#: The User-Agent sent with every request. MusicBrainz asks for an agent
#: that identifies the application and gives a way to get in touch; set
#: ``SPLYTNYT_CONTACT`` to add your own contact details.
DEFAULT_USER_AGENT = "SplytNyt/2.0 (local-first audio splitter)"

#: Where the cache lives, relative to the application data directory.
CACHE_DIRECTORY = "metadata-cache"


def user_agent() -> str:
    """The User-Agent to send."""
    contact = os.environ.get("SPLYTNYT_CONTACT", "").strip()
    agent = DEFAULT_USER_AGENT
    if contact:
        agent = f"{agent} (contact: {contact})"
    return agent


class RateLimiter:
    """Keeps at least ``interval`` seconds between two requests."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = max(0.0, float(interval))
        self._last = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        now = time.monotonic()
        remaining = self._last + self.interval - now
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()


class HttpClient:
    """A small JSON client with a cache and a rate limit."""

    def __init__(
        self,
        provider: str,
        cache_dir: Optional[str] = None,
        min_interval: float = 1.0,
        timeout: float = 20.0,
        headers: Optional[dict[str, str]] = None,
        max_retries: int = 3,
    ) -> None:
        self.provider = provider
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.limiter = RateLimiter(min_interval)
        self.extra_headers = dict(headers or {})
        self.max_retries = max(0, int(max_retries))

    # ------------------------------------------------------------------
    def _send_json_request(
        self,
        cache_key: str,
        request: urllib.request.Request,
        use_cache: bool = True,
    ) -> Any:
        cached = self._read_cache(cache_key) if use_cache else None
        if cached is not None:
            return cached

        payload = b""
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 503) and attempt < self.max_retries:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        delay = float(retry_after) if retry_after else 1.5 * (2 ** attempt)
                    except ValueError:
                        delay = 1.5 * (2 ** attempt)
                    time.sleep(min(8.0, max(1.0, delay)))
                    continue
                if exc.code in (429, 503):
                    raise ProviderRateLimited(
                        f"{self.provider} asked for a slower request rate. Try again in a moment."
                    ) from exc
                if exc.code == 403:
                    raise ProviderUnavailable(
                        f"{self.provider} refused the request (403). "
                        + (
                            "This provider needs an API key; set it in the environment "
                            "and try again."
                            if self.extra_headers
                            else "The request was rejected."
                        )
                    ) from exc
                if exc.code == 404:
                    raise ProviderError(f"{self.provider} has no answer for that request.") from exc
                raise ProviderError(
                    f"{self.provider} returned an error ({exc.code})."
                ) from exc
            except urllib.error.URLError as exc:
                raise ProviderUnavailable(
                    f"{self.provider} could not be reached: {exc.reason}. "
                    "SplytNyt keeps working offline - a pasted tracklist needs no network."
                ) from exc
            except TimeoutError as exc:
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise ProviderUnavailable(
                    f"{self.provider} did not answer in time."
                ) from exc

        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"{self.provider} sent something that is not JSON."
            ) from exc

        self._write_cache(cache_key, data)
        return data

    # ------------------------------------------------------------------
    def get_json(
        self,
        url: str,
        params: Optional[dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Any:
        """Fetch a JSON document, with a cache, retry/backoff, and typed failures."""
        full_url = url
        if params:
            query = urllib.parse.urlencode(
                {key: value for key, value in params.items() if value not in (None, "")}
            )
            separator = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{separator}{query}"

        request = urllib.request.Request(
            full_url,
            headers={
                "User-Agent": user_agent(),
                "Accept": "application/json",
                **self.extra_headers,
            },
            method="GET",
        )
        return self._send_json_request(full_url, request, use_cache=use_cache)

    def post_form_json(
        self,
        url: str,
        form_data: dict[str, Any],
        use_cache: bool = True,
    ) -> Any:
        """POST form-encoded data and parse a JSON response."""
        encoded = urllib.parse.urlencode(
            {key: value for key, value in form_data.items() if value not in (None, "")}
        ).encode("utf-8")
        cache_key = f"POST:{url}?{encoded.decode('utf-8', 'replace')}"
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={
                "User-Agent": user_agent(),
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                **self.extra_headers,
            },
            method="POST",
        )
        return self._send_json_request(cache_key, request, use_cache=use_cache)

    # ------------------------------------------------------------------
    def _cache_path(self, url: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{self.provider}-{digest[:24]}.json")

    def _read_cache(self, url: str) -> Optional[dict]:
        path = self._cache_path(url)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        stamp = payload.get("cached_at", 0)
        if time.time() - float(stamp) > CACHE_TTL_SECONDS:
            return None
        return payload.get("data")

    def _write_cache(self, url: str, data: dict) -> None:
        path = self._cache_path(url)
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"url": url, "cached_at": time.time(), "data": data}, handle)
        except OSError:
            # A cache that cannot be written is not a reason to fail.
            pass


def default_cache_dir() -> Optional[str]:
    """Where the provider cache lives."""
    from PySide6.QtCore import QStandardPaths

    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".splytnyt")
    return os.path.join(base, CACHE_DIRECTORY)
