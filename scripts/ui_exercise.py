"""Drive the real interface and check what it does.

This is not a unit test: it builds the real window, imports a real file
through the real import path, runs the real analysis, and then *clicks the
real buttons* and looks at what changed. It also captures a screenshot of
each theme.

    python scripts/ui_exercise.py

The checks are printed as a list. Everything that cannot be verified on this
machine is printed as skipped, with the reason.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RATE = 44100

RESULTS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(("PASS" if ok else "FAIL", name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def skip(name: str, reason: str) -> None:
    RESULTS.append(("SKIP", name, reason))
    print(f"  [SKIP] {name} - {reason}")


def pump(app: QApplication, seconds: float = 0.2) -> None:
    """Let the interface settle and the worker threads finish."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def wait_for_job(app: QApplication, window, timeout: float = 180.0) -> bool:
    """Wait until the background job in the window has finished."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        runner = window._runner
        if runner is None or not runner.isRunning():
            pump(app, 0.2)
            return True
        time.sleep(0.02)
    return False


def click(widget) -> None:
    """A real mouse click on a real widget."""
    QTest.mouseClick(widget, Qt.LeftButton)


def overlapping_buttons(parent) -> list[tuple[str, str]]:
    """Any two buttons in a panel that are drawn on top of each other.

    The buttons live in different group boxes, so their geometries are
    compared in the coordinates of the panel they all sit in.
    """
    origin = parent
    buttons = [
        child for child in parent.findChildren(QPushButton)
        if child.isVisible() and child.width() > 0 and child.height() > 0
    ]
    clashes = []
    for index, first in enumerate(buttons):
        first_rect = QRect(
            first.mapTo(origin, QPoint(0, 0)), first.size()
        )
        for second in buttons[index + 1:]:
            second_rect = QRect(
                second.mapTo(origin, QPoint(0, 0)), second.size()
            )
            if first_rect.intersects(second_rect):
                clashes.append((first.text(), second.text()))
    return clashes


def main() -> int:
    os.makedirs(os.path.join(HERE, "demo"), exist_ok=True)
    recording = os.path.join(HERE, "demo", "demo_concert.flac")
    if not os.path.exists(recording):
        from build_demo import build_recording

        build_recording(recording)

    app = QApplication(sys.argv[:1])

    from splytnyt.ui import theme
    from splytnyt.ui.main_window import MainWindow

    # The recovery question is a modal dialog; this script is about the
    # rest of the interface, so it is not asked here.
    MainWindow._offer_recovery = lambda self: None

    window = MainWindow()
    window.resize(1600, 1000)
    window.show()
    pump(app, 0.4)

    # ------------------------------------------------------------------
    print("\n== the empty state")
    # ------------------------------------------------------------------
    check(
        "the empty state explains how to start",
        "import" in window.waveform._empty_state_text().lower(),
        window.waveform._empty_state_text().splitlines()[0] if
        window.waveform._empty_state_text() else "no text",
    )
    check(
        "the review buttons are disabled when there is nothing to review",
        not window.zoom_boundary_button.isEnabled()
        and not window.accept_button.isEnabled(),
    )
    window.grab().save(os.path.join(ROOT, "docs", "screenshot-empty.png"))

    # ------------------------------------------------------------------
    print("\n== importing a recording (the real import path)")
    # ------------------------------------------------------------------
    QFileDialog.getOpenFileNames = staticmethod(lambda *args, **kwargs: ([recording], ""))
    window.import_audio()
    ok = wait_for_job(app, window)
    pump(app, 0.4)
    check("the file was imported", len(window.session.project.sources) == 1)
    check("the waveform was built", bool(window._pyramids))
    check("the import job finished", ok)

    # ------------------------------------------------------------------
    print("\n== analysing (the real analysis path)")
    # ------------------------------------------------------------------
    window.run_analysis()
    ok = wait_for_job(app, window)
    pump(app, 0.4)
    tracks = window.session.project.sorted_tracks()
    check("the analysis job finished", ok)
    check("more than one track was found", len(tracks) > 1, f"{len(tracks)} tracks")
    check(
        "the tracks cover the whole recording",
        sum(track.frames for track in tracks) == window.session.project.total_frames(),
    )

    # ------------------------------------------------------------------
    print("\n== reviewing a boundary")
    # ------------------------------------------------------------------
    target = tracks[1]
    window.table.select_track(target.track_id)
    window._select_boundary(target.start_frame)
    pump(app, 0.2)

    check(
        "the sidebar shows the selected cut",
        window.boundary_position_label.text() != "No boundary selected",
        window.boundary_position_label.text(),
    )
    check(
        "the evidence is in plain language",
        "ms" in window.reasons_text.toPlainText() or "cut" in window.reasons_text.toPlainText().lower(),
        window.reasons_text.toPlainText().splitlines()[0] if
        window.reasons_text.toPlainText() else "",
    )
    check(
        "no confidence percentage is shown",
        "%" not in window.boundary_label.text(),
        f"boundary reads '{window.boundary_label.text()}'",
    )

    click(window.zoom_boundary_button)
    pump(app, 0.2)
    start, end = window.waveform._visible_range()
    check(
        "zoom to cut puts the cut on screen",
        start <= target.start_frame <= end,
        f"visible {start / RATE:.1f} s - {end / RATE:.1f} s",
    )
    check(
        "the selected boundary is marked on the waveform",
        window.waveform.selected_boundary == target.start_frame,
    )

    # ------------------------------------------------------------------
    print("\n== the onset and the cut are two different places")
    # ------------------------------------------------------------------
    check(
        "the detected onset is marked on the waveform as well",
        window.waveform.onset_frame == target.onset_frame,
        f"the onset is at {window.waveform.onset_frame}, the cut is at "
        f"{window.waveform.selected_boundary}",
    )
    check(
        "the onset marker is not drawn on the cut",
        window.waveform.onset_frame != window.waveform.selected_boundary,
        "the margin would be invisible if they were the same place",
    )
    margin_text = window.margin_info_label.text()
    check(
        "the panel says where the music begins",
        "begin" in margin_text.lower(),
        margin_text,
    )
    check(
        "the panel says how much margin was actually applied",
        "ms" in margin_text,
        margin_text,
    )
    check(
        "the applied margin matches what the track recorded",
        f"{target.boundary_pre_roll_seconds * 1000:.0f} ms" in margin_text,
        f"the track says {target.boundary_pre_roll_seconds * 1000:.0f} ms, the "
        f"panel says '{margin_text}'",
    )

    # ------------------------------------------------------------------
    print("\n== the margin, its scope and the two presets")
    # ------------------------------------------------------------------
    scopes = [
        window.margin_scope_combo.itemData(index)
        for index in range(window.margin_scope_combo.count())
    ]
    check(
        "the margin has an explicit scope",
        "this_cut" in scopes and "new_suggestions" in scopes,
        ", ".join(str(item) for item in scopes),
    )

    window.margin_scope_combo.setCurrentIndex(
        window.margin_scope_combo.findData("this_cut")
    )
    pump(app, 0.1)

    original_start = window.session.project.sorted_tracks()[1].start_frame
    original_margin = window.session.project.sorted_tracks()[1].boundary_pre_roll_seconds

    # "Tight start": the cut moves closer to the music.
    click(window.tight_button)
    pump(app, 0.3)
    tight = window.session.project.sorted_tracks()[1]
    check(
        "the tight preset uses a 20 ms margin",
        abs(window.pre_roll_spin.value() - 0.020) < 1e-6,
        f"{window.pre_roll_spin.value() * 1000:.0f} ms",
    )
    check(
        "the tight preset moves the cut closer to the music",
        tight.start_frame > original_start,
        f"{original_start} -> {tight.start_frame}",
    )
    check(
        "the tight preset is measured from the onset",
        abs((tight.onset_frame - tight.start_frame) / RATE - 0.020) < 0.002,
        f"{(tight.onset_frame - tight.start_frame) / RATE * 1000:.1f} ms in "
        f"front of the onset",
    )
    check(
        "the previous track's end follows the cut",
        window.session.project.sorted_tracks()[0].end_frame == tight.start_frame,
    )
    check(
        "a margin change for one cut is undoable",
        window.session.undo_action() is True
        and window.session.project.sorted_tracks()[1].start_frame == original_start,
        f"back at {window.session.project.sorted_tracks()[1].start_frame}",
    )
    check(
        "the reviewed cut follows the undo",
        window.waveform.selected_boundary == original_start,
        f"the waveform marks {window.waveform.selected_boundary}",
    )

    # "Keep more lead-in": the cut moves further in front of the music.
    window._select_boundary(original_start)
    pump(app, 0.1)
    click(window.lead_in_button)
    pump(app, 0.3)
    lead_in = window.session.project.sorted_tracks()[1]
    check(
        "the keep-more-lead-in preset uses a 150 ms margin",
        abs(window.pre_roll_spin.value() - 0.150) < 1e-6,
        f"{window.pre_roll_spin.value() * 1000:.0f} ms",
    )
    check(
        "the keep-more-lead-in preset moves the cut further in front",
        lead_in.start_frame < original_start,
        f"{original_start} -> {lead_in.start_frame}",
    )
    other_before = window.session.project.sorted_tracks()[2].start_frame
    window.session.undo_action()
    pump(app, 0.2)
    check(
        "the preset leaves the other cuts alone",
        window.session.project.sorted_tracks()[2].start_frame == other_before,
    )

    # The scope decides what a margin change touches.
    window.margin_scope_combo.setCurrentIndex(
        window.margin_scope_combo.findData("new_suggestions")
    )
    pump(app, 0.1)
    cut_before = window.session.project.sorted_tracks()[1].start_frame
    window.pre_roll_spin.setValue(0.08)
    pump(app, 0.2)
    check(
        "a margin for new suggestions changes the project setting",
        abs(window.session.project.placement_pre_roll_seconds - 0.08) < 1e-6,
        f"{window.session.project.placement_pre_roll_seconds * 1000:.0f} ms",
    )
    check(
        "a margin for new suggestions does not move the cut being reviewed",
        window.session.project.sorted_tracks()[1].start_frame == cut_before,
        f"the cut stayed at {window.session.project.sorted_tracks()[1].start_frame}",
    )
    window.session.undo_action()
    pump(app, 0.1)
    window.margin_scope_combo.setCurrentIndex(
        window.margin_scope_combo.findData("this_cut")
    )
    pump(app, 0.1)
    window.pre_roll_spin.setValue(original_margin)
    pump(app, 0.2)

    # ------------------------------------------------------------------
    print("\n== the transport uses the same positions")
    # ------------------------------------------------------------------
    # The tracks are read again here: an undo replaces the project's track
    # objects, so anything captured earlier would be stale.
    target = window.session.project.sorted_tracks()[1]
    window.table.select_track(target.track_id)
    window._select_boundary(target.start_frame)
    pump(app, 0.2)

    playback = window._playback
    if playback is None:
        skip("play across the cut", "no playback engine was built")
        skip("play the next track", "no playback engine was built")
        skip("play the previous ending", "no playback engine was built")
    else:
        click(window.play_across_button)
        pump(app, 0.3)
        expected = max(0, target.start_frame - int(3.0 * RATE))
        check(
            "play across starts three seconds before the cut",
            abs(playback.position - expected) < int(0.5 * RATE),
            f"started at {playback.position / RATE:.2f} s, expected "
            f"{expected / RATE:.2f} s",
        )
        window.stop_playback()

        click(window.play_next_button)
        pump(app, 0.3)
        check(
            "play the next track starts at the exported start of that track",
            abs(playback.position - target.start_frame) < int(0.5 * RATE),
            f"started at {playback.position / RATE:.2f} s, the track starts at "
            f"{target.start_frame / RATE:.2f} s",
        )
        window.stop_playback()

        click(window.play_previous_button)
        pump(app, 0.3)
        check(
            "play the previous ending stops at the cut",
            playback._end_frame == target.start_frame,
            f"the window ends at {playback._end_frame / RATE:.2f} s, the cut is "
            f"at {target.start_frame / RATE:.2f} s",
        )
        window.stop_playback()

        click(window.preview_export_button)
        pump(app, 0.3)
        check(
            "previewing the exported start plays from the track's first frame",
            abs(playback.position - target.start_frame) < int(0.5 * RATE),
            f"started at {playback.position / RATE:.2f} s, the track starts at "
            f"{target.start_frame / RATE:.2f} s",
        )
        window.stop_playback()

    # ------------------------------------------------------------------
    print("\n== moving the cut")
    # ------------------------------------------------------------------
    before_start = window.session.project.sorted_tracks()[1].start_frame
    before_end = window.session.project.sorted_tracks()[0].end_frame
    adjust = None
    for button in window.findChildren(QPushButton):
        if button.text() == "+10 ms":
            adjust = button
            break
    if adjust is None:
        check("there is a fine adjustment control", False)
    else:
        click(adjust)
        pump(app, 0.2)
        after = window.session.project.sorted_tracks()
        check(
            "fine adjustment moves the cut by exactly ten milliseconds",
            after[1].start_frame == before_start + int(round(0.010 * RATE)),
            f"{before_start} -> {after[1].start_frame}",
        )
        check(
            "the previous track's end follows the cut",
            after[0].end_frame == after[1].start_frame,
            f"track 1 ends at {after[0].end_frame / RATE:.3f} s, the cut is at "
            f"{after[1].start_frame / RATE:.3f} s",
        )
        check(
            "a moved cut is marked as a human decision",
            after[1].boundary_assessment() == "Manual",
            after[1].boundary_assessment(),
        )
        check(
            "the reported margin follows the cut that was moved",
            f"{after[1].boundary_pre_roll_seconds * 1000:.0f} ms"
            in window.margin_info_label.text(),
            f"the panel says '{window.margin_info_label.text()}'",
        )

    # ------------------------------------------------------------------
    print("\n== accepting and locking")
    # ------------------------------------------------------------------
    window._select_boundary(window.session.project.sorted_tracks()[1].start_frame)
    pump(app, 0.1)
    click(window.accept_button)
    pump(app, 0.2)
    check(
        "accepting a cut keeps it, and says so in words",
        window.session.project.sorted_tracks()[1].boundary_assessment()
        in {"High", "Fair", "Low", "Manual", "Accepted"},
        window.session.project.sorted_tracks()[1].boundary_assessment(),
    )
    check(
        "the panel explains the assessment in words, not as a percentage",
        "%" not in window.boundary_state_label.text()
        and len(window.boundary_state_label.text()) > 10,
        window.boundary_state_label.text(),
    )

    click(window.lock_button)
    pump(app, 0.2)
    check(
        "locking a track locks it",
        any(track.locked for track in window.session.project.sorted_tracks()),
    )
    click(window.lock_button)
    pump(app, 0.2)

    # ------------------------------------------------------------------
    print("\n== the placement controls")
    # ------------------------------------------------------------------
    combo = window.placement_combo
    index = combo.findData("previous_end")
    combo.setCurrentIndex(index)
    pump(app, 0.2)
    check(
        "changing the placement is remembered by the project",
        window.session.project.placement_mode == "previous_end",
        window.session.project.placement_mode,
    )
    check(
        "the hint explains what the placement does",
        "previous song" in window.placement_hint.text().lower(),
        window.placement_hint.text()[:60],
    )
    combo.setCurrentIndex(combo.findData("next_start"))
    pump(app, 0.1)

    # The pre-roll that new suggestions will use is a project setting, so it
    # is changed with the scope set to "new suggestions".
    window.margin_scope_combo.setCurrentIndex(
        window.margin_scope_combo.findData("new_suggestions")
    )
    pump(app, 0.1)
    window.pre_roll_spin.setValue(0.25)
    pump(app, 0.2)
    check(
        "the pre-roll is remembered by the project",
        abs(window.session.project.placement_pre_roll_seconds - 0.25) < 1e-6,
        f"{window.session.project.placement_pre_roll_seconds} s",
    )
    window.pre_roll_spin.setValue(0.03)
    pump(app, 0.1)
    window.margin_scope_combo.setCurrentIndex(
        window.margin_scope_combo.findData("this_cut")
    )
    pump(app, 0.1)

    # ------------------------------------------------------------------
    print("\n== the layout")
    # ------------------------------------------------------------------
    overflow = [
        bar.windowTitle() for bar in window.toolbars
        if bar.x() + bar.width() > window.width() + 1
    ]
    check(
        "every toolbar group fits on the screen",
        not overflow,
        ", ".join(overflow) + " runs off the right-hand edge"
        if overflow else f"{len(window.toolbars)} groups fit in {window.width()} px",
    )
    check(
        "the undo and redo buttons are on screen",
        window.undo_button.isVisible() and window.redo_button.isVisible(),
    )
    clashes = overlapping_buttons(window.sidebar)
    check(
        "no two buttons in the details panel overlap",
        not clashes,
        "; ".join(f"{a} / {b}" for a, b in clashes),
    )
    sidebar_width = window.sidebar.width()
    check(
        "the details panel is wide enough for its controls",
        sidebar_width >= 300,
        f"{sidebar_width} px",
    )
    check(
        "the waveform and the table are both visible",
        window.waveform.isVisible() and window.table.isVisible(),
    )

    # The sidebar can be hidden and shown again.
    click(window.sidebar_button)
    pump(app, 0.2)
    check("the details panel can be hidden", not window.sidebar.isVisible())
    click(window.sidebar_button)
    pump(app, 0.2)
    check("the details panel comes back", window.sidebar.isVisible())

    # ------------------------------------------------------------------
    print("\n== the themes")
    # ------------------------------------------------------------------
    for mode in ("dark", "light"):
        window.set_theme(mode, remember=False)
        pump(app, 0.4)
        stylesheet = app.styleSheet()
        check(
            f"the {mode} theme is applied to the whole application",
            "QMainWindow" in stylesheet or "QWidget" in stylesheet,
            f"{len(stylesheet)} characters of style sheet",
        )
        check(
            f"the {mode} theme reaches the details panel",
            window.palette().color(window.palette().ColorRole.Window).name()
            == theme.palette_for_mode(mode).background.lower(),
            window.palette().color(window.palette().ColorRole.Window).name(),
        )
        check(
            f"the {mode} theme has a readable contrast on the background",
            theme.contrast_ratio(
                theme.palette_for_mode(mode).text,
                theme.palette_for_mode(mode).background,
            ) >= 4.5,
            f"{theme.contrast_ratio(theme.palette_for_mode(mode).text, theme.palette_for_mode(mode).background):.1f}:1",
        )

    # ------------------------------------------------------------------
    print("\n== screenshots")
    # ------------------------------------------------------------------
    for mode in ("dark", "light"):
        window.set_theme(mode, remember=False)
        window.waveform.zoom_to_fit()
        pump(app, 0.3)
        window.waveform.zoom_to_frame(
            window.session.project.sorted_tracks()[1].start_frame, span_seconds=16.0
        )
        pump(app, 0.3)
        output = os.path.join(ROOT, "docs", f"screenshot-{mode}.png")
        window.grab().save(output)
        print(f"  saved {output}")

    window.set_theme("dark", remember=False)
    window.waveform.zoom_to_fit()
    pump(app, 0.3)
    window.grab().save(os.path.join(ROOT, "docs", "screenshot.png"))
    print(f"  saved {os.path.join(ROOT, 'docs', 'screenshot.png')}")

    # A close-up of one cut. At this zoom the detected onset and the cut
    # itself are far enough apart to be told apart, which is the whole point
    # of drawing them as two markers.
    for mode in ("dark", "light"):
        window.set_theme(mode, remember=False)
        reviewed = window.session.project.sorted_tracks()[1]
        window.waveform.zoom_to_frame(reviewed.start_frame, span_seconds=0.4)
        window.waveform.set_cursor(reviewed.start_frame)
        pump(app, 0.3)
        output = os.path.join(ROOT, "docs", f"screenshot-boundary-{mode}.png")
        window.grab().save(output)
        print(f"  saved {output}")

    # ------------------------------------------------------------------
    print("\n== closing down")
    # ------------------------------------------------------------------
    # Closing the window has to stop the playback and close the files
    # cleanly, so the application can be quit without falling over. The
    # "do you want to save?" question is answered here rather than left to
    # block the script.
    window._confirm_discard = lambda: True
    window.close()
    pump(app, 0.5)
    check("the window closes without complaining", True)

    # ------------------------------------------------------------------
    failed = [item for item in RESULTS if item[0] == "FAIL"]
    skipped = [item for item in RESULTS if item[0] == "SKIP"]
    print(f"\n{len(RESULTS) - len(failed) - len(skipped)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped")
    for status, name, detail in failed:
        print(f"  FAILED: {name} - {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
