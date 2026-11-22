"""Application entry point."""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="splytnyt",
        description="Split long audio recordings into named, tagged tracks.",
    )
    parser.add_argument("project", nargs="?", help="a .splytnyt project file to open")
    parser.add_argument('--verify-selection', metavar='REPORT_JSON',
                        help='Run opt-in Qt selection and real audio-output regression, then exit')
    parser.add_argument(
        "--version", action="store_true", help="print the version and exit"
    )
    args = parser.parse_args(argv)

    if args.version:
        from . import APP_NAME, __version__

        print(f"{APP_NAME} {__version__}")
        return 0

    # Qt must be imported after the argument parsing so --version works on a
    # machine without a display.
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("SplytNyt")
    app.setOrganizationName("SplytNyt")

    from .ui import theme
    from .metadata.shazamio_engine import cleanup_pending_uninstall

    cleanup_pending_uninstall()
    theme.apply_theme(app, theme.stored_mode())

    from .ui.main_window import MainWindow

    window = MainWindow()
    if args.project and os.path.exists(args.project):
        try:
            window.session.open_project(args.project)
            window._reload()
            window._rebuild_peaks()
            window.waveform.zoom_to_fit()
        except Exception as exc:
            window.session.status(f"Could not open {os.path.basename(args.project)}: {exc}")
    window.show()
    if args.verify_selection:
        from PySide6.QtCore import QTimer
        from .selection_check import verify
        QTimer.singleShot(1000, lambda: verify(window, args.verify_selection, app))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
