"""Filesystem safety helpers.

The application must never write over a source recording, not even when the
user picks the source's own folder or an equivalent path (a symlink, a
different case, a drive letter, a UNC path). These helpers compare paths the
conservative way: normalised text first, then file identity where the target
exists.
"""

from __future__ import annotations

import os
import re
from typing import Iterable

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def normalise(path: str) -> str:
    """Absolute, normalised, case-folded path text (for comparison)."""
    try:
        absolute = os.path.abspath(path)
    except Exception:
        absolute = path
    return os.path.normcase(os.path.normpath(absolute))


def same_path(first: str, second: str) -> bool:
    """True when two paths refer to the same location."""
    if not first or not second:
        return False
    if normalise(first) == normalise(second):
        return True
    try:
        if os.path.exists(first) and os.path.exists(second):
            return os.path.samefile(first, second)
    except OSError:
        pass
    return False


def same_file_identity(first: str, second: str) -> bool:
    """True when both paths exist and point at the same file (hard link, symlink)."""
    try:
        if not (os.path.exists(first) and os.path.exists(second)):
            return False
        stat_a = os.stat(first)
        stat_b = os.stat(second)
        return (stat_a.st_dev, stat_a.st_ino) == (stat_b.st_dev, stat_b.st_ino)
    except OSError:
        return False


def is_inside_directory(path: str, directory: str) -> bool:
    """True when ``path`` lies inside ``directory``."""
    try:
        path_abs = os.path.abspath(path)
        dir_abs = os.path.abspath(directory)
    except Exception:
        return False
    if normalise(path_abs) == normalise(dir_abs):
        return True
    try:
        return os.path.commonpath([path_abs, dir_abs]) == dir_abs
    except ValueError:
        return False


def path_conflicts_with_sources(path: str, source_paths: Iterable[str]) -> bool:
    """True when writing to ``path`` would overwrite one of the sources."""
    for source in source_paths:
        if not source:
            continue
        if same_path(path, source):
            return True
        if same_file_identity(path, source):
            return True
    return False


def is_reserved_name(name: str) -> bool:
    """True for Windows reserved device names (``CON``, ``LPT1``, ...)."""
    stem = name.split(".")[0].upper()
    return stem in _WINDOWS_RESERVED


def unique_path(path: str, taken: Iterable[str] = ()) -> str:
    """Return ``path`` or a non-colliding variant (``name (2).ext``)."""
    taken_normalised = {normalise(item) for item in taken}
    if normalise(path) not in taken_normalised and not os.path.exists(path):
        return path
    directory = os.path.dirname(path)
    stem, extension = os.path.splitext(os.path.basename(path))
    index = 2
    while True:
        candidate = os.path.join(directory, f"{stem} ({index}){extension}")
        if normalise(candidate) not in taken_normalised and not os.path.exists(candidate):
            return candidate
        index += 1


def ensure_writable_directory(path: str) -> str:
    """Create ``path`` if needed and confirm it is a writable directory."""
    if not path:
        raise ValueError("No output directory was given.")
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory.")
    probe = os.path.join(path, ".splytnyt-write-test")
    try:
        with open(probe, "wb") as handle:
            handle.write(b"")
        os.remove(probe)
    except OSError as exc:
        raise PermissionError(f"{path} is not writable: {exc}") from exc
    return os.path.abspath(path)


def free_space_bytes(path: str) -> int:
    """Free space on the volume holding ``path`` (0 when unknown)."""
    try:
        usage = os.statvfs(path)  # type: ignore[attr-defined]
        return usage.f_bavail * usage.f_frsize
    except Exception:
        return 0


_TEMPLATE_FIELD = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?::([^}]*))?\}")


def format_filename(template: str, context: dict) -> str:
    """Expand a filename template.

    Supported fields: ``n`` (track number), ``track``, ``title``, ``artist``,
    ``album``, ``albumartist``, ``disc``, ``date``, ``year``, ``genre``,
    ``uniq`` (a per-export counter used for collision protection).
    Unknown fields are left in place verbatim so a typo is visible instead of
    silently producing an empty name.
    """

    def replace(match: re.Match) -> str:
        name = match.group(1)
        spec = match.group(2)
        if name not in context:
            return match.group(0)
        value = context[name]
        if spec:
            try:
                return format(value, spec)
            except (TypeError, ValueError):
                return str(value)
        return str(value)

    return _TEMPLATE_FIELD.sub(replace, template)
