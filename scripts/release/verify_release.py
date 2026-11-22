"""Verify the built Windows release end to end.

  * install the installer silently into a fresh directory
  * check the installed layout is complete
  * run the installed application with a PATH that contains no Python
  * run the application's own opt-in self-test (real clicks, real audio out,
    real export) against a synthetic project
  * uninstall and check the install is gone

Usage:
    python verify_release.py <installer.exe> <work-dir>
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

results = []


def step(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    return ok


def directory_size(path: str) -> int:
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2

    installer = os.path.abspath(argv[1])
    work = os.path.abspath(argv[2])
    install_dir = os.path.join(work, "install")
    project_dir = os.path.join(work, "project")
    os.makedirs(work, exist_ok=True)

    if not os.path.isfile(installer):
        step("installer exists", False, installer)
        return 1
    step("installer exists", True, f"{os.path.getsize(installer):,} bytes")

    # ---- 1. install ------------------------------------------------------
    log = os.path.join(work, "install.log")
    started = time.time()
    proc = subprocess.run(
        [
            installer, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
            "/CURRENTUSER", f"/DIR={install_dir}", f"/LOG={log}",
            "/TASKS=\"\"",
        ],
        capture_output=True, timeout=900,
    )
    elapsed = time.time() - started
    step("silent install", proc.returncode == 0,
         f"exit {proc.returncode} in {elapsed:.0f}s")

    exe = os.path.join(install_dir, "SplytNyt.exe")
    step("installed executable present", os.path.isfile(exe), exe)
    if not os.path.isfile(exe):
        return 1

    internal = os.path.join(install_dir, "_internal")
    step("dependencies folder present", os.path.isdir(internal), "_internal")
    size = directory_size(install_dir)
    step("installed size", size > 50 * 1024 * 1024, f"{size / 1024 / 1024:.0f} MB")

    licenses = os.path.join(install_dir, "licenses")
    step("licence texts installed", os.path.isdir(licenses)
         and os.path.isfile(os.path.join(licenses, "LGPL-3.0.txt")),
         licenses)
    step("third-party notices installed",
         os.path.isfile(os.path.join(install_dir, "THIRD_PARTY_NOTICES.md")))

    # ---- 2. the installed app must not need Python -----------------------
    clean_path = os.pathsep.join([
        r"C:\Windows\System32", r"C:\Windows", os.path.join(work, "empty-bin"),
    ])
    os.makedirs(os.path.join(work, "empty-bin"), exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = clean_path
    env["PYTHONHOME"] = ""
    env["PYTHONPATH"] = ""

    version = subprocess.run([exe, "--version"], capture_output=True, timeout=180,
                             env=env, creationflags=0x08000000)  # CREATE_NO_WINDOW
    step("launch with no Python on PATH", version.returncode == 0,
         f"exit {version.returncode}")

    # ---- 3. the application's own self-test ------------------------------
    report_path = os.path.join(work, "verify-selection.json")
    if os.path.exists(report_path):
        os.remove(report_path)
    project = os.path.join(project_dir, "verification-project.splytnyt")
    if not os.path.isfile(project):
        step("verification project present", False, project)
        return 1

    started = time.time()
    proc = subprocess.run(
        [exe, project, "--verify-selection", report_path],
        capture_output=True, timeout=600, env=env, creationflags=0x08000000,
    )
    elapsed = time.time() - started
    step("self-test ran", os.path.isfile(report_path),
         f"exit {proc.returncode} in {elapsed:.0f}s")

    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        step("self-test passed", bool(report.get("passed")),
             str(report.get("error", "")))
        step("running as a frozen application", bool(report.get("frozen")))
        checks = report.get("checks") or []
        step("real audio output", len(checks) > 0
             and all(c.get("diagnostics", {}).get("callback_peak", 0) > 0 for c in checks),
             f"{len(checks)} playback cases")
        export = report.get("export") or {}
        step("FLAC export bit-exact", bool(export.get("flac", {}).get("bit_exact")),
             str(export.get("flac", {})))
        step("WAV export bit-exact", bool(export.get("wav", {}).get("bit_exact")),
             str(export.get("wav", {})))
        step("MP3 export produced", bool(export.get("mp3", {}).get("decoded")),
             str(export.get("mp3", {})))
        step("CUE sheet written", bool(export.get("continuous", {}).get("cue_written")),
             str(export.get("continuous", {})))
        imported = report.get("imported") or {}
        step("file import worked", bool(imported.get("frames")), str(imported))
    else:
        step("self-test report written", False)

    # ---- 4. uninstall -----------------------------------------------------
    uninstaller = os.path.join(install_dir, "unins000.exe")
    if os.path.isfile(uninstaller):
        proc = subprocess.run(
            [uninstaller, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
            capture_output=True, timeout=600,
        )
        step("uninstaller ran", proc.returncode == 0, f"exit {proc.returncode}")
        time.sleep(3)
        step("installation removed", not os.path.isfile(exe))
    else:
        step("uninstaller present", False, uninstaller)

    # ---- summary ----------------------------------------------------------
    failed = [name for name, ok, _ in results if not ok]
    print("\n" + "=" * 64)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
