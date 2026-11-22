"""Dialogs.

Each dialog is a thin shell over the service layer: it collects choices and
hands them back, and never performs audio work itself.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..core.model import ExportSettings
from ..core.serialize import parse_tracklist
from ..core.units import frames_to_timestamp


class TracklistDialog(QDialog):
    """Paste a tracklist. Works without any network service."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Paste tracklist")
        self.setMinimumSize(620, 480)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Paste one track per line. A leading number is ignored; a trailing "
            "(4:32) is read as a duration hint."
        )
        intro.setProperty("hint", True)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.name_edit = QLineEdit("Pasted tracklist")
        layout.addWidget(self.name_edit)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Unknown / smart split", "unknown")
        self.mode_combo.addItem("Concert", "concert")
        self.mode_combo.addItem("Vinyl / cassette", "album")
        self.mode_combo.addItem("Musical theatre", "theater")
        layout.addWidget(self.mode_combo)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            "1. The Song of the Storm\n2. Another Song (4:32)\n3. - Encore -\n"
        )
        self.text_edit.setTabChangesFocus(True)
        layout.addWidget(self.text_edit)

        self.preview_label = QLabel("")
        self.preview_label.setProperty("hint", True)
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        self.text_edit.textChanged.connect(self._update_preview)
        self._update_preview()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_preview(self) -> None:
        items = parse_tracklist(self.text_edit.toPlainText())
        hinted = sum(1 for item in items if item.duration_hint > 0)
        self.preview_label.setText(
            f"{len(items)} entr{'y' if len(items) == 1 else 'ies'} recognised"
            + (f", {hinted} with a duration hint" if hinted else "")
        )

    def values(self) -> tuple[str, str, list]:
        return (
            self.name_edit.text().strip() or "Tracklist",
            self.mode_combo.currentData(),
            parse_tracklist(self.text_edit.toPlainText()),
        )


class ReferenceDialog(QDialog):
    """Choose one of the bundled theatrical reference lists.

    A reference list suggests names and an order. It does not establish a
    timestamp and it does not prove that a number is in the recording, so
    the dialog says so and shows where the list came from.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load a reference list")
        self.setMinimumSize(660, 560)

        from ..metadata import references as reference_tools

        self._references = reference_tools.list_references()
        self._worklist = None

        layout = QVBoxLayout(self)

        intro = QLabel(
            "A reference list suggests the names and the order of the musical "
            "numbers of a production. It does not establish any timestamp and "
            "it does not prove that a number is in your recording. You can edit "
            "the list before you use it, and nothing is ever forced to match."
        )
        intro.setProperty("hint", True)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.reference_combo = QComboBox()
        for reference in self._references:
            self.reference_combo.addItem(reference.title, reference.reference_id)
        layout.addWidget(self.reference_combo)

        self.version_combo = QComboBox()
        layout.addWidget(self.version_combo)

        self.act_combo = QComboBox()
        self.act_combo.addItem("Every act", None)
        self.act_combo.addItem("Act 1 only", 1)
        self.act_combo.addItem("Act 2 only", 2)
        layout.addWidget(self.act_combo)

        self.sources_view = QTextBrowser()
        self.sources_view.setReadOnly(True)
        layout.addWidget(self.sources_view)

        self.reference_combo.currentIndexChanged.connect(self._refresh)
        self.act_combo.currentIndexChanged.connect(self._refresh)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh()

    # ------------------------------------------------------------------
    def _current(self):
        index = self.reference_combo.currentIndex()
        if index < 0 or index >= len(self._references):
            return None
        return self._references[index]

    def _refresh(self) -> None:
        reference = self._current()
        self.version_combo.blockSignals(True)
        self.version_combo.clear()
        if reference:
            for version in reference.versions:
                self.version_combo.addItem(version.label, version.version_id)
        self.version_combo.blockSignals(False)

        if reference:
            from ..metadata import references as reference_tools

            text = reference_tools.provenance_text(
                reference, self.version_combo.currentData()
            )
            self.sources_view.setPlainText(text)
        else:
            self.sources_view.setPlainText(
                "No reference lists are installed. A pasted tracklist works "
                "without one."
            )

    # ------------------------------------------------------------------
    def worklist(self):
        """The worklist the user asked for, or ``None``."""
        from ..metadata import references as reference_tools

        reference = self._current()
        if reference is None:
            return None
        return reference_tools.build_worklist(
            reference,
            self.version_combo.currentData(),
            self.act_combo.currentData(),
        )


class ExportDialog(QDialog):
    """Export options.

    The "Measure levels" button runs a background measurement and fills in
    the measured loudness, the measured peak and the proposed gain, including
    the explicit trade-off when a target cannot be met under the ceiling.
    """

    def __init__(self, settings: ExportSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # ---- files ----------------------------------------------------
        files = QWidget()
        files_layout = QFormLayout(files)

        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit(settings.output_dir)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._choose_directory)
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(browse)
        files_layout.addRow("Output folder", dir_row)

        self.template_edit = QLineEdit(settings.filename_template)
        self.template_edit.setToolTip(
            "Available fields: {n} {track} {title} {artist} {album} "
            "{albumartist} {disc} {date} {genre}"
        )
        files_layout.addRow("Filename template", self.template_edit)

        self.container_combo = QComboBox()
        self.container_combo.addItem("FLAC (lossless)", "flac")
        self.container_combo.addItem("WAV (uncompressed)", "wav")
        self.container_combo.addItem("MP3 (lossy)", "mp3")
        self.container_combo.setCurrentIndex(max(0, self.container_combo.findData(settings.container)))
        files_layout.addRow("Format", self.container_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Separate track files", "separate")
        self.mode_combo.addItem("One continuous file + CUE sheet", "continuous")
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(settings.mode)))
        files_layout.addRow("Layout", self.mode_combo)

        self.profile_combo = QComboBox()
        self.profile_combo.addItem(
            "Complete performance (every section, no audio lost)", "complete"
        )
        self.profile_combo.addItem(
            "Songs only (the musical numbers; everything else is reported)", "songs"
        )
        self.profile_combo.setToolTip(
            "The complete export keeps every stretch of the recording. The songs "
            "export writes only the musical numbers and reports every exclusion, "
            "so you can see what is being left out."
        )
        self.profile_combo.setCurrentIndex(
            max(0, self.profile_combo.findData(getattr(settings, "export_profile", "complete")))
        )
        files_layout.addRow("What to export", self.profile_combo)

        self.submarker_check = QCheckBox(
            "Write the songs found inside a number as submarkers"
        )
        self.submarker_check.setChecked(bool(getattr(settings, "export_submarkers", False)))
        self.submarker_check.setToolTip(
            "A medley is one number that holds several songs. The number keeps "
            "its own title; the songs inside it are written separately."
        )
        files_layout.addRow("", self.submarker_check)

        self.section_combo = QComboBox()
        self.section_combo.addItem("Keep every section as it is (default)", "keep_as_own")
        self.section_combo.addItem("Attach applause / talk to the previous track", "attach_previous")
        self.section_combo.addItem("Attach applause / talk to the next track", "attach_next")
        self.section_combo.addItem("Omit applause / talk from this export", "omit")
        files_layout.addRow("Applause / banter", self.section_combo)

        self.selected_only_check = QCheckBox("Export only the track selected in the table")
        self.selected_only_check.setToolTip(
            "Leave unticked to export the whole project."
        )
        files_layout.addRow("", self.selected_only_check)

        self.mp3_bitrate = QComboBox()
        for rate in (128, 160, 192, 256, 320):
            self.mp3_bitrate.addItem(f"{rate} kbit/s", rate)
        self.mp3_bitrate.setCurrentIndex(max(0, self.mp3_bitrate.findData(settings.mp3_bitrate_kbps)))
        files_layout.addRow("MP3 quality", self.mp3_bitrate)

        tabs.addTab(files, "Files")

        # ---- levels ---------------------------------------------------
        levels = QWidget()
        levels_layout = QFormLayout(levels)

        self.gain_combo = QComboBox()
        self.gain_combo.addItem("Untouched (no gain applied)", "untouched")
        self.gain_combo.addItem("Shared gain for the whole recording", "shared")
        self.gain_combo.addItem("Loudness normalisation (whole recording)", "loudness")
        self.gain_combo.addItem("Normalise each track separately", "per_track")
        self.gain_combo.setCurrentIndex(max(0, self.gain_combo.findData(settings.gain_mode)))
        self.gain_combo.currentIndexChanged.connect(self._update_level_state)
        levels_layout.addRow("Level handling", self.gain_combo)

        self.target_spin = QDoubleSpinBox()
        self.target_spin.setRange(-40.0, -5.0)
        self.target_spin.setSingleStep(0.5)
        self.target_spin.setSuffix(" LUFS")
        self.target_spin.setValue(settings.loudness_target_lufs)
        levels_layout.addRow("Target loudness", self.target_spin)

        self.ceiling_spin = QDoubleSpinBox()
        self.ceiling_spin.setRange(-20.0, 0.0)
        self.ceiling_spin.setSingleStep(0.5)
        self.ceiling_spin.setSuffix(" dBTP")
        self.ceiling_spin.setValue(settings.true_peak_ceiling_dbtp)
        levels_layout.addRow("True-peak ceiling", self.ceiling_spin)

        self.limiter_check = QCheckBox("Allow an explicit limiter to reach the target")
        self.limiter_check.setChecked(settings.allow_limiter)
        self.limiter_check.setToolTip(
            "Off by default. Without this, SplytNyt uses less gain rather than "
            "altering the signal."
        )
        levels_layout.addRow("", self.limiter_check)

        self.replaygain_check = QCheckBox("Write ReplayGain metadata where supported")
        self.replaygain_check.setChecked(settings.replaygain)
        levels_layout.addRow("", self.replaygain_check)

        warning = QLabel(
            "SplytNyt never compresses, limits, denoises or resamples on its own. "
            "Stereo balance is preserved."
        )
        warning.setProperty("hint", True)
        warning.setWordWrap(True)
        levels_layout.addRow("", warning)

        self.measure_button = QPushButton("Measure levels")
        self.measure_button.clicked.connect(self._request_measurement)
        levels_layout.addRow("", self.measure_button)

        self.measure_label = QLabel("Not measured yet.")
        self.measure_label.setWordWrap(True)
        self.measure_label.setTextFormat(Qt.RichText)
        levels_layout.addRow("", self.measure_label)

        tabs.addTab(levels, "Levels")

        # ---- extras ---------------------------------------------------
        extras = QWidget()
        extras_layout = QFormLayout(extras)

        self.tags_check = QCheckBox("Write tags (title, artist, album, ...)")
        self.tags_check.setChecked(settings.write_tags)
        extras_layout.addRow("", self.tags_check)

        self.cue_check = QCheckBox("Write a CUE sheet with the continuous file")
        self.cue_check.setChecked(settings.include_cue)
        extras_layout.addRow("", self.cue_check)

        self.m3u_check = QCheckBox("Write an M3U playlist")
        self.m3u_check.setChecked(settings.include_m3u)
        extras_layout.addRow("", self.m3u_check)

        self.info_check = QCheckBox("Write the plain-text info file (archival package)")
        self.info_check.setChecked(settings.include_info)
        extras_layout.addRow("", self.info_check)

        self.checksum_check = QCheckBox("Write SHA-256 checksums")
        self.checksum_check.setChecked(settings.include_checksums)
        extras_layout.addRow("", self.checksum_check)

        tabs.addTab(extras, "Extras")

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.button(QDialogButtonBox.Ok).setText("Export")
        buttons.button(QDialogButtonBox.Ok).setProperty("primary", True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_level_state()

    # ------------------------------------------------------------------
    def _choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose the output folder", self.dir_edit.text()
        )
        if directory:
            self.dir_edit.setText(directory)

    def _update_level_state(self) -> None:
        mode = self.gain_combo.currentData()
        enabled = mode in ("shared", "loudness", "per_track")
        self.target_spin.setEnabled(enabled)
        self.ceiling_spin.setEnabled(enabled)
        self.limiter_check.setEnabled(enabled)

    def _request_measurement(self) -> None:
        parent = self.parent()
        if parent is not None and hasattr(parent, "measure_for_dialog"):
            parent.measure_for_dialog(self)
        else:
            self.measure_label.setText("Measurement is not available in this context.")

    def set_measurement(self, text: str) -> None:
        self.measure_label.setText(text)

    def values(self) -> ExportSettings:
        settings = ExportSettings()
        settings.output_dir = self.dir_edit.text().strip()
        settings.filename_template = self.template_edit.text().strip() or "{n:02d} {title}"
        settings.container = self.container_combo.currentData()
        settings.mode = self.mode_combo.currentData()
        settings.gain_mode = self.gain_combo.currentData()
        settings.loudness_target_lufs = self.target_spin.value()
        settings.true_peak_ceiling_dbtp = self.ceiling_spin.value()
        settings.allow_limiter = self.limiter_check.isChecked()
        settings.replaygain = self.replaygain_check.isChecked()
        settings.mp3_bitrate_kbps = self.mp3_bitrate.currentData()
        settings.write_tags = self.tags_check.isChecked()
        settings.include_cue = self.cue_check.isChecked()
        settings.include_m3u = self.m3u_check.isChecked()
        settings.include_info = self.info_check.isChecked()
        settings.include_checksums = self.checksum_check.isChecked()
        settings.export_profile = self.profile_combo.currentData()
        settings.export_submarkers = self.submarker_check.isChecked()
        return settings

    def section_policy(self) -> str:
        return self.section_combo.currentData()


class MetadataDialog(QDialog):
    """Project-level metadata and the live performance fields."""

    FIELDS = [
        ("title", "Title", "album", "Album"),
        ("artist", "Artist", "album_artist", "Album artist"),
        ("album", "Album", "composer", "Composer"),
        ("genre", "Genre", "publisher", "Publisher"),
        ("date", "Date", "conductor", "Conductor"),
        ("comments", "Comments", "performer", "Performer"),
        ("performance_date", "Performance date", "venue", "Venue"),
        ("city", "City", "country", "Country"),
        ("tour", "Tour", "source_notes", "Source notes"),
        ("equipment", "Equipment", "lineage", "Lineage"),
        ("production", "Production", "act", "Act"),
        ("song", "Song", "identifier", "Identifier"),
    ]

    def __init__(self, metadata, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project metadata")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.edits: dict[str, QLineEdit] = {}
        for left, left_label, right, right_label in self.FIELDS:
            left_edit = QLineEdit(str(getattr(metadata, left, "") or ""))
            right_edit = QLineEdit(str(getattr(metadata, right, "") or ""))
            self.edits[left] = left_edit
            self.edits[right] = right_edit
            row = QHBoxLayout()
            row.addWidget(left_edit)
            row.addWidget(right_edit)
            form.addRow(f"{left_label} / {right_label}", row)
        layout.addWidget(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        return {key: edit.text().strip() for key, edit in self.edits.items()}


class ShortcutsDialog(QDialog):
    """The keyboard reference."""

    SHORTCUTS = [
        ("Space", "Play / pause"),
        ("Left / Right", "Previous / next boundary"),
        ("Enter", "Preview the boundary at the cursor"),
        ("S", "Split at the cursor"),
        ("M", "Merge with the previous track"),
        ("Backspace", "Remove the track at the cursor (audio is kept)"),
        ("[ / ]", "Nudge the boundary by 10 ms"),
        ("Shift+[ / Shift+]", "Nudge the boundary by 1 second"),
        ("L", "Lock or unlock the selected track"),
        ("A", "Accept all suggestions (locked tracks are untouched)"),
        ("Ctrl+Z", "Undo"),
        ("Ctrl+Shift+Z", "Redo"),
        ("Ctrl+wheel", "Zoom in / out at the pointer"),
        ("0", "Zoom to fit"),
        ("F", "Zoom to the selected track"),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        rows = "".join(
            f"<tr><td><b>{key}</b></td><td>{description}</td></tr>"
            for key, description in self.SHORTCUTS
        )
        browser.setHtml(
            f"<table cellpadding='4'>{rows}</table>"
            "<p><i>Arrow keys move to the previous / next boundary.</i></p>"
        )
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, Qt.Horizontal, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class AboutDialog(QDialog):
    """About box with the third-party notices."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About SplytNyt")
        self.setMinimumSize(560, 420)
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setHtml(
            """
            <h2>SplytNyt</h2>
            <p>Local-first splitting and tagging of long audio recordings.<br>
            The working title is provisional.</p>
            <h3>What this build does</h3>
            <ul>
              <li>Import audio into one ordered timeline</li>
              <li>Zoomable waveform with draggable boundaries</li>
              <li>Sample-accurate playback and boundary preview</li>
              <li>Adaptive boundary suggestions with stated evidence</li>
              <li>Tracklist alignment</li>
              <li>Non-destructive export to FLAC, WAV and MP3</li>
            </ul>
            <h3>Privacy</h3>
            <p>Processing is local. Nothing is uploaded. No account is required.</p>
            <h3>Third-party components</h3>
            <ul>
              <li>Qt for Python (PySide6) - LGPL v3</li>
              <li>libsndfile via python-soundfile - LGPL v2.1</li>
              <li>LAME via lameenc - LGPL v2.1</li>
              <li>NumPy and SciPy - BSD 3-Clause</li>
              <li>PortAudio via sounddevice - MIT</li>
            </ul>
            <p><i>Model weights, if optional models are enabled later, are downloaded
            on request and their size is shown before the download starts.</i></p>
            """
        )
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, Qt.Horizontal, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class RefinePreviewDialog(QDialog):
    """Show where the existing boundaries would move before they move.

    Refining is never silent: the proposal is shown first, every move can be
    unticked, and the whole set is one undoable step.
    """

    def __init__(self, proposals, sample_rate: int = 44100,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Refine boundaries")
        self.setMinimumSize(640, 420)
        self._proposals = list(proposals)
        self._sample_rate = sample_rate or 44100

        layout = QVBoxLayout(self)

        heading = QLabel("These cuts would move")
        heading.setProperty("heading", True)
        layout.addWidget(heading)

        note = QLabel(
            "Nothing has changed yet. Untick a move to leave that cut where it "
            "is. Locked tracks are never moved."
        )
        note.setProperty("hint", True)
        note.setWordWrap(True)
        layout.addWidget(note)

        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

        self.table = QTableWidget(len(self._proposals), 4)
        self.table.setHorizontalHeaderLabels(["Move", "From", "To", "Why"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, self.table.horizontalHeader().Fixed)
        header.setSectionResizeMode(1, self.table.horizontalHeader().Fixed)
        header.setSectionResizeMode(2, self.table.horizontalHeader().Fixed)
        header.setSectionResizeMode(3, self.table.horizontalHeader().Stretch)
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 100)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        self._checks: list[QCheckBox] = []
        for row, proposal in enumerate(self._proposals):
            self.table.setRowCount(row + 1)
            check = QCheckBox("")
            check.setChecked(True)
            self._checks.append(check)
            self.table.setCellWidget(row, 0, check)
            self.table.setItem(row, 1, QTableWidgetItem(
                frames_to_timestamp(proposal.old_frame, self._sample_rate)))
            self.table.setItem(row, 2, QTableWidgetItem(
                frames_to_timestamp(proposal.new_frame, self._sample_rate)))
            self.table.setItem(row, 3, QTableWidgetItem(proposal.reason))
        layout.addWidget(self.table, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.button(QDialogButtonBox.Ok).setText("Move these cuts")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_proposals(self) -> list:
        """The proposals the user kept."""
        return [
            proposal
            for proposal, check in zip(self._proposals, self._checks)
            if check.isChecked()
        ]


class SourcesDialog(QDialog):
    """List the imported sources and let the user set the gap between them."""

    def __init__(self, project, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Source files")
        self.setMinimumSize(620, 340)
        layout = QVBoxLayout(self)

        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

        self.table = QTableWidget(len(project.sources), 5)
        self.table.setHorizontalHeaderLabels(
            ["File", "Format", "Sample rate", "Channels", "Gap before (s)"]
        )
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, self.table.horizontalHeader().Stretch)
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        self.table.verticalHeader().setVisible(False)

        self._gaps: list[QDoubleSpinBox] = []
        for row, source in enumerate(project.sources):
            self.table.setItem(row, 0, QTableWidgetItem(os.path.basename(source.path)))
            self.table.setItem(row, 1, QTableWidgetItem(source.original_format))
            self.table.setItem(row, 2, QTableWidgetItem(f"{source.sample_rate} Hz"))
            self.table.setItem(row, 3, QTableWidgetItem(str(source.channels)))
            if row == 0:
                self.table.setItem(row, 4, QTableWidgetItem("-"))
                continue
            spin = QDoubleSpinBox()
            spin.setRange(-600.0, 600.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setSuffix(" s")
            spin.setValue(source.gap_frames / (project.sample_rate() or 44100))
            spin.setToolTip(
                "Positive values insert silence between the files; negative values "
                "make them overlap. Both are preserved exactly."
            )
            self._gaps.append(spin)
            self.table.setCellWidget(row, 4, spin)
        layout.addWidget(self.table)

        note = QLabel(
            "Gaps and overlaps between source files are preserved. They are never "
            "removed silently."
        )
        note.setProperty("hint", True)
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def gap_values(self) -> list[float]:
        return [spin.value() for spin in self._gaps]
