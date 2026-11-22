"""Record real provider responses as test fixtures.

Run once by hand; the recorded files are committed and the automated suite
never touches the network. MusicBrainz needs no key, so its responses are
genuine recordings. setlist.fm needs an API key, so its fixture is a
hand-written sample of the documented response shape.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splytnyt.metadata.base import HttpClient  # noqa: E402

FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures"
)
ROOT = "https://musicbrainz.org/ws/2"


def save(name: str, data) -> None:
    os.makedirs(FIXTURES, exist_ok=True)
    path = os.path.join(FIXTURES, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


def fetch(client: HttpClient, url: str, params: dict) -> dict:
    """Fetch with a few retries: MusicBrainz answers 503 when it is busy."""
    from splytnyt.metadata import ProviderError

    last = None
    for attempt in range(6):
        try:
            return client.get_json(url, params=params)
        except ProviderError as exc:
            last = exc
            wait = 3.0 * (attempt + 1)
            print(f"  attempt {attempt + 1} failed ({exc}); waiting {wait:.0f} s")
            time.sleep(wait)
    raise last


def main() -> int:
    client = HttpClient("musicbrainz", cache_dir=None, min_interval=1.1)

    search = fetch(
        client,
        f"{ROOT}/release-group/",
        {"query": 'artist:"Radiohead" AND releasegroup:"Kid A"', "fmt": "json", "limit": 25},
    )
    groups = search.get("release-groups", [])
    trimmed = dict(search)
    trimmed["release-groups"] = groups[:2]
    trimmed["count"] = len(groups)
    save("musicbrainz_release_group_search.json", trimmed)

    if not groups:
        print("no release groups found")
        return 1

    group_id = groups[0]["id"]
    group = fetch(
        client, f"{ROOT}/release-group/{group_id}", {"inc": "releases+media", "fmt": "json"}
    )
    releases = group.get("releases", [])
    group["releases"] = releases[:3]
    save("musicbrainz_release_group.json", group)

    if not releases:
        print("no releases found")
        return 1

    release_id = releases[0]["id"]
    release = fetch(
        client,
        f"{ROOT}/release/{release_id}",
        {"inc": "recordings+media+artist-credits", "fmt": "json"},
    )
    save("musicbrainz_release.json", release)

    print(f"release group: {group_id}")
    print(f"release: {release_id}")
    time.sleep(1.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
