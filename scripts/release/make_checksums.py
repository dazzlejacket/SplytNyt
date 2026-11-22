"""Write SHA-256 checksums for the release artefacts.

Usage:
    python scripts/release/make_checksums.py release/

It writes ``checksums.sha256`` next to the artefacts, in the format used by
``sha256sum -c``:

    4f2a...  SplytNyt-0.1.0-beta-setup.exe
"""
from __future__ import annotations

import hashlib
import os
import sys

CHUNK = 1024 * 1024


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    folder = os.path.abspath(argv[1])
    if not os.path.isdir(folder):
        print(f"not a directory: {folder}")
        return 1

    artefacts = [
        name
        for name in sorted(os.listdir(folder))
        if name.lower().endswith((".exe", ".msi", ".zip"))
        and not name.lower().endswith(".sha256")
    ]
    if not artefacts:
        print(f"no release artefacts found in {folder}")
        return 1

    lines = []
    for name in artefacts:
        path = os.path.join(folder, name)
        digest = sha256_of(path)
        size = os.path.getsize(path)
        lines.append(f"{digest}  {name}")
        print(f"{digest}  {name}  ({size:,} bytes)")

    out = os.path.join(folder, "checksums.sha256")
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
