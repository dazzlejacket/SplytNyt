"""Build the Windows application.

Usage:
    python scripts/build_windows.py            # build
    python scripts/build_windows.py --clean    # build from scratch

The build produces ``dist/SplytNyt/SplytNyt.exe`` and everything it needs
next to it. The optional analysis models are *not* bundled: they are
downloaded on demand into the user's profile, and the application says so
before it downloads anything.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def directory_size(path: str) -> int:
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def human(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--console", action="store_true",
                        help="build a console build, which is easier to debug")
    args = parser.parse_args()

    dist = os.path.join(ROOT, "dist")
    build = os.path.join(ROOT, "build")
    work = os.path.join(ROOT, ".build_work")

    if args.clean:
        for path in (dist, build, work):
            if os.path.exists(path):
                print(f"removing {path}")
                shutil.rmtree(path, ignore_errors=True)

    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--distpath", dist,
        "--workpath", work,
        os.path.join("scripts", "splytnyt.spec"),
    ]
    if args.console:
        command.append("--console")

    started = time.time()
    print("running:", " ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        print("the build failed")
        return result.returncode

    target = os.path.join(dist, "SplytNyt")
    exe = os.path.join(target, "SplytNyt.exe")
    print(f"built in {time.time() - started:.1f}s")
    if os.path.exists(exe):
        print(f"executable: {exe}")
        print(f"folder size: {human(directory_size(target))}")
    else:
        print("the executable was not produced")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
