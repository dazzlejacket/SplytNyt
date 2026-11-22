"""Entry point for the packaged Windows application.

PyInstaller needs a real script file. This is it: it does nothing except
hand over to the package's own entry point, so the frozen application and
the source tree behave identically.
"""
import multiprocessing
import sys

if __name__ == "__main__":
    # Not used here, but it keeps a frozen build from re-launching itself
    # in a loop on Windows.
    multiprocessing.freeze_support()
    from splytnyt.main import main

    raise SystemExit(main())
