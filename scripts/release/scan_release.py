"""Security, licence and copyright scan for the public release tree.

Checks what would actually be published - the git-tracked files - and the full
git history for:

  * credentials and API keys
  * copyrighted media (audio / video) and lyric transcripts
  * personal filesystem paths and personal identifiers
  * build junk, virtual environments and caches

Exit code 0 means clean. Anything else means the release must not be published.

Usage:
    python scripts/release/scan_release.py [repo_root]
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

#: The project's own AcoustID *application* key. It is deliberately public:
#: it is registered to the project, not to a person, and the application has
#: to work without asking every user to register for one.
ALLOWED_PUBLIC_KEYS = {"ztpTKhkIHy"}

#: Binary/text file extensions that must never appear in the repository.
MEDIA_EXTENSIONS = {
    ".mp3", ".flac", ".wav", ".aiff", ".aif", ".aifc", ".m4a", ".mp4", ".aac",
    ".ogg", ".oga", ".opus", ".ape", ".wma", ".mka", ".wv", ".mpc", ".dsf",
    ".dff", ".m3u", ".m3u8", ".pls", ".cue", ".vob", ".mkv", ".avi", ".mov",
    ".webm", ".flv", ".wmv", ".m4v", ".srt", ".ass", ".vtt",
}

#: File extensions that are scanned as text.
TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".toml", ".yml", ".yaml", ".cfg", ".ini",
    ".spec", ".iss", ".html", ".css", ".js", ".ts",
}

#: Vendored third-party licence texts. They are scanned for real credentials,
#: but not for generic hash patterns: they legitimately contain upstream
#: commit hashes and checksum digests.
VENDORED_LICENCE_DIR = "licenses/"

SECRET_PATTERNS = [
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),
    ("github pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    ("openai key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("stripe key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{20,}\b")),
    ("npm token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer token", re.compile(r"(?i)\bauthorization\s*[:=]\s*[\"']?(?:bearer|token)\s+[A-Za-z0-9_\-\.=]{20,}")),
    ("generic secret assignment",
     re.compile(r"""(?i)\b(?:api[_-]?key|apikey|secret|password|passwd|auth[_-]?token|access[_-]?token|client[_-]?secret)\b\s*[:=]\s*["']([A-Za-z0-9_\-\.\+/=]{12,})["']""")),
]

#: A bare 40-hex string only matters when it looks like it is being used as a
#: credential. The same shape also appears as a git SHA and as a content
#: checksum, so the neighbourhood has to agree.
BARE_HEX_KEY = re.compile(r"(?<![0-9a-fA-F])([0-9a-f]{40})(?![0-9a-fA-F])")
CREDENTIAL_NEIGHBOUR = re.compile(
    r"(?i)\b(?:key|token|secret|setlist|acoustid|api|auth|bearer|credential|"
    r"apikey|api_key|client)\b"
)

#: Personal identifiers and absolute paths that must not leak.
PERSONAL_PATTERNS = [
    ("windows user path", re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\\s\"'<>:;,)\]]+")),
    ("home path", re.compile(r"(?i)(?:/home/|/Users/)[A-Za-z0-9_.\-]+")),
    ("private key file", re.compile(r"(?i)\b\S+\.(?:pem|ppk|p12|pfx|keystore)\b")),
]

#: Project-specific directories that hold private material and must never be
#: referenced from a public file - not even by a relative path.
PRIVATE_PATH_PATTERNS = [
    ("private deliverables path", re.compile(r"(?i)(?:^|[\s\"'(/])deliverables/[\w\-./ ]+")),
    ("private sample directory", re.compile(r"(?i)(?:^|[\s\"'(/])sample MR/[\w\-./ ]+")),
    ("private project artefact", re.compile(r"(?i)lyrics-selection-\d{4}-\d{2}-\d{2}")),
    ("private recovery artefact", re.compile(r"(?i)recovery-\d{4}-\d{2}-\d{2}")),
]

#: Directories that must not be tracked.
JUNK_NAMES = {
    ".venv", "venv", "env", "__pycache__", ".pytest_cache", "node_modules",
    "dist", "build", ".build_work", ".build_recovery", "dist-selfcheck",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode", ".eggs",
}

JUNK_SUFFIXES = (".pyc", ".pyo", ".pyd", ".npz", ".log", ".egg-info")

#: Obvious non-secrets: synthetic test values and documentation placeholders.
BENIGN_MARKERS = (
    "your_", "example", "placeholder", "changeme", "xxxxx", "notarealkey",
    "dummy", "redacted", "todo", "test_secret_value", "unit_test_secret",
    "<", "sample", "fake", "notreal",
)


def is_probably_text(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTENSIONS:
        return True
    if ext in {".gitignore", ".gitattributes"} or os.path.basename(path).startswith("."):
        return True
    return False


def is_benign(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in BENIGN_MARKERS)


def scan_text(text: str, where: str) -> list[tuple[str, str, str]]:
    findings: list[tuple[str, str, str]] = []
    vendored = where.replace("\\", "/").startswith(VENDORED_LICENCE_DIR)

    for label, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1) if (label == "generic secret assignment" and match.groups()) else match.group(0)
            if value in ALLOWED_PUBLIC_KEYS or is_benign(value):
                continue
            findings.append(("SECRET", where, f"{label}: {value[:24]}..."))

    if not vendored:
        for match in BARE_HEX_KEY.finditer(text):
            value = match.group(1)
            if value in ALLOWED_PUBLIC_KEYS or is_benign(value):
                continue
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            if not CREDENTIAL_NEIGHBOUR.search(text[start:end]):
                continue
            findings.append(("SECRET", where, f"bare 40-hex key near credential wording: {value[:12]}..."))

    for label, pattern in PERSONAL_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            if is_benign(value) or any(token in value.lower() for token in (
                "yourname", "your-name", "my own recordings", "my cast",
                "my music", "username", "music/artist", "path/to",
            )):
                continue
            findings.append(("PERSONAL", where, f"{label}: {value[:60]}"))

    for label, pattern in PRIVATE_PATH_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(("PRIVATE-PATH", where, f"{label}: {match.group(0).strip()[:70]}"))

    return findings


def git_tracked(root: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", root, "ls-files", "-z"],
        capture_output=True, timeout=120,
    )
    return [name for name in out.stdout.decode("utf-8", "replace").split("\0") if name]


def scan_tree(root: str) -> list[tuple[str, str, str]]:
    """Scan every file git would publish."""
    findings: list[tuple[str, str, str]] = []

    for rel in git_tracked(root):
        path = os.path.join(root, rel.replace("/", os.sep))
        ext = os.path.splitext(rel)[1].lower()
        if ext in MEDIA_EXTENSIONS:
            findings.append(("MEDIA", rel, f"media file with extension {ext!r}"))
        if rel.endswith(JUNK_SUFFIXES) or "/__pycache__/" in f"/{rel}":
            findings.append(("JUNK", rel, "build/cache artefact is tracked"))
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 40 * 1024 * 1024:
                findings.append(("SIZE", rel, f"file is {size:,} bytes"))
        if not is_probably_text(rel):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        findings.extend(scan_text(text, rel))

    # Untracked junk: worth knowing about, but it is not published.
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        rel_base = os.path.relpath(base, root)
        for d in dirs:
            if d in JUNK_NAMES:
                findings.append(("UNTRACKED-JUNK", os.path.join(rel_base, d).replace("\\", "/"),
                                 f"directory {d!r} exists in the working tree (not tracked)"))
    return findings


def scan_history(root: str) -> list[tuple[str, str, str]]:
    """Scan every blob in the full git history."""
    findings: list[tuple[str, str, str]] = []
    try:
        objects = subprocess.run(
            ["git", "-C", root, "rev-list", "--objects", "--all"],
            capture_output=True, text=True, timeout=300,
        ).stdout.splitlines()
    except Exception as exc:
        return [("ERROR", "git", f"could not list history: {exc}")]

    seen = set()
    for line in objects:
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        sha, name = parts
        if sha in seen:
            continue
        seen.add(sha)

        kind = subprocess.run(
            ["git", "-C", root, "cat-file", "-t", sha],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        if kind != "blob":
            continue  # trees only contain object ids, not content

        if os.path.splitext(name)[1].lower() in MEDIA_EXTENSIONS:
            findings.append(("MEDIA", f"history:{name}", "media file in git history"))
        if not is_probably_text(name):
            continue
        blob = subprocess.run(
            ["git", "-C", root, "cat-file", "-p", sha],
            capture_output=True, timeout=120,
        )
        if blob.returncode != 0:
            continue
        try:
            text = blob.stdout.decode("utf-8")
        except UnicodeDecodeError:
            continue
        findings.extend(scan_text(text, f"history:{name}"))
    return findings


def main(argv: list[str]) -> int:
    root = os.path.abspath(argv[1] if len(argv) > 1 else os.getcwd())

    print(f"scanning {root}\n")
    tree_findings = scan_tree(root)
    history_findings = scan_history(root)
    # Untracked junk is informational: it is not published.
    blocking = [f for f in tree_findings + history_findings if f[0] != "UNTRACKED-JUNK"]
    informational = [f for f in tree_findings + history_findings if f[0] == "UNTRACKED-JUNK"]

    print(f"tracked files : {len([f for f in tree_findings if f[0] != 'UNTRACKED-JUNK'])} blocking finding(s)")
    print(f"git history   : {len(history_findings)} finding(s)")

    if informational:
        print("\ninformational (not published):")
        for severity, where, message in informational:
            print(f"  {where}: {message}")

    if not blocking:
        print("\nCLEAN: no secrets, media, personal paths or tracked build junk found.")
        return 0

    print("\nBLOCKING FINDINGS")
    print("=" * 72)
    for severity, where, message in blocking:
        print(f"[{severity:8s}] {where}\n           {message}")
    print("=" * 72)
    print(f"\n{len(blocking)} blocking finding(s) - the release must not be published yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
