"""Capture screenshots of the running application, in both themes.

This is a development helper. It builds a demo recording, loads it into the
real interface, runs the real analysis, and saves a PNG of each theme so
the result can be reviewed without a human driving it.

    python scripts/screenshot.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402

from splytnyt.app.jobs import AnalysisJob, ImportJob  # noqa: E402
from splytnyt.ui import theme  # noqa: E402

RATE = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main() -> int:
    os.makedirs(os.path.join(HERE, "demo"), exist_ok=True)
    recording = os.path.join(HERE, "demo", "demo_concert.flac")
    if not os.path.exists(recording):
        from build_demo import build_recording

        build_recording(recording)

    app = QApplication(sys.argv[:1])

    from splytnyt.ui.main_window import MainWindow

    window = MainWindow()
    window.resize(1600, 1000)
    window.show()
    app.processEvents()

    window.session.import_files([recording])
    window._reload()

    peaks = ImportJob(window.session.project, cache_dir=os.path.join(HERE, "demo", "peaks"))
    peaks.run()
    window._pyramids = peaks.pyramids
    window.waveform.set_pyramids(window._pyramids)

    analysis = AnalysisJob(
        window.session.project,
        sensitivity="balanced",
        placement=window.session.placement_choice(),
    )
    outcome = analysis.run()
    print(f"{len(outcome.suggestions)} suggested boundaries")
    window.session.apply_analysis(outcome.suggestions, outcome.classifications)

    window.session.paste_tracklist(
        "Opening Song\nSecond Piece (5:10)\nThird Piece\nFinal Song - with applause"
    )
    window.session.align_with_worklist()

    window._reload()
    window.waveform.zoom_to_fit()
    app.processEvents()

    tracks = window.session.project.sorted_tracks()
    if tracks:
        window.table.select_track(tracks[1].track_id)
        window.waveform.select_track(tracks[1].track_id)
        window.waveform.set_cursor(tracks[1].start_frame)
        window._select_boundary(tracks[1].start_frame)
        window.waveform.zoom_to_frame(tracks[1].start_frame, span_seconds=14.0)
    app.processEvents()

    for mode in ("dark", "light"):
        window.set_theme(mode, remember=False)
        app.processEvents()
        app.processEvents()
        output = os.path.join(ROOT, "docs", f"screenshot-{mode}.png")
        window.grab().save(output)
        print(f"saved {output}")

    # The dark one is the main screenshot in the README.
    window.set_theme("dark", remember=False)
    app.processEvents()
    window.grab().save(os.path.join(ROOT, "docs", "screenshot.png"))
    print("saved docs/screenshot.png")

    for track in tracks:
        print(
            f"  {track.title or '(untitled)':<28} "
            f"{track.start_frame / RATE:7.2f}s -> {track.end_frame / RATE:7.2f}s  "
            f"section={track.section_type:<9} "
            f"boundary={track.boundary_assessment():<13} "
            f"placement={track.boundary_placement}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
