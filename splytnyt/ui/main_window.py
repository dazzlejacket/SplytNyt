"""The main window.

Assembles the waveform, the track table and the details panel, owns the
playback engine, and runs long jobs (import, analysis, export, measurement)
on worker threads so the interface never blocks.

The window is laid out in three bands:

* a **toolbar** grouped into Import, Analyse, Review and Export,
* a **transport** with play, stop, the time and the track being played,
* the **work area**: the waveform above the track table, with a collapsible
  details panel on the right.

Everything a reviewer needs to judge one cut is in the details panel: the
evidence in plain language, the technical numbers behind it, and the
controls for moving the cut.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from PySide6.QtCore import QObject, QSettings, QStandardPaths, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ..app.events import BUSY, PROJECT_CHANGED, STATUS, TRACKS_CHANGED
from ..app.jobs import (
    AnalysisJob,
    ExportJob,
    ImportJob,
    Job,
    JobCancelled,
    RefineJob,
    SmartSplitJob,
    TheaterJob,
)
from ..app.session import Session, SessionError
from ..audio.decode import SUPPORTED_FILTER
from ..audio.playback import PlaybackEngine
from ..core import serialize
from ..core.model import SECTION_TYPES
from ..core.units import frames_to_timestamp
from . import theme
from .dialogs import (
    AboutDialog,
    ExportDialog,
    MetadataDialog,
    ReferenceDialog,
    RefinePreviewDialog,
    ShortcutsDialog,
    SourcesDialog,
    TracklistDialog,
)
from .providers import ProviderLookupDialog
from .settings_dialog import MusicServicesDialog
from .track_table import TrackTable
from .waveform import WaveformView

AUTOSAVE_SECONDS = 30


class JobRunner(QThread):
    """Runs a :class:`~splytnyt.app.jobs.Job` off the GUI thread."""

    progress = Signal(float, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, job: Job, parent=None) -> None:
        super().__init__(parent)
        self.job = job
        # The job's own progress callback goes into this thread's signal, so
        # the update is delivered to the interface on the main thread.
        # Calling a widget from the worker thread is not allowed in Qt, and
        # it crashes the application.
        job.progress = self.progress.emit

    def run(self) -> None:
        try:
            result = self.job.run()
            self.succeeded.emit(result)
        except JobCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class PlaybackBridge(QObject):
    """Carries the playback engine's callbacks onto the interface thread.

    The audio engine knows nothing about Qt and reports from its own worker
    thread. A signal emitted from that thread and connected to a receiver
    in the interface thread is delivered in the interface thread, which is
    the only place a widget or a timer may be touched.
    """

    position = Signal(int)
    state = Signal(bool)
    error = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)


class MainWindow(QMainWindow):
    """SplytNyt main window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SplytNyt")
        self.resize(1500, 960)
        self.setMinimumSize(1024, 640)

        self.session = Session()
        self._runner: Optional[JobRunner] = None
        self._playback: Optional[PlaybackEngine] = None
        self._reader = None
        self._paused_end = None
        self._output_device = None
        self._pyramids: dict = {}
        self._preview_before = 3.0
        self._preview_after = 2.0
        self._last_autosave = 0.0
        self._selected_boundary: Optional[int] = None
        self._selected_track_id = ""
        self._sidebar_visible = True
        self._last_identification = None
        self._auto_smart_split_after_peaks = False

        self.setAcceptDrops(True)
        self.settings = QSettings("SplytNyt", "SplytNyt")

        self._build_ui()
        self._build_menus()
        self._connect_session()
        self._apply_stored_theme()

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(AUTOSAVE_SECONDS * 1000)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()

        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(40)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        self._reload()
        # Show the main window and open a command-line project before asking
        # about recovery. A modal during construction hides the review UI.
        QTimer.singleShot(0, self._offer_recovery)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.waveform = WaveformView(self)
        self.waveform.cursorMoved.connect(self._on_cursor_moved)
        self.waveform.boundaryDragged.connect(self._on_boundary_dragged)
        self.waveform.boundaryClicked.connect(self._on_boundary_clicked)
        self.waveform.trackSelected.connect(self._on_track_selected)
        self.waveform.zoomChanged.connect(self._on_zoom_changed)
        self.waveform.seekRequested.connect(self._on_waveform_seek)
        self.waveform.splitRequested.connect(lambda frame: self.session.split_at(frame))
        self.waveform.removeSplitRequested.connect(self._on_remove_split)
        self.waveform.renameTrackRequested.connect(self._rename_track_dialog)
        self.waveform.playTrackRequested.connect(self._play_track_by_id)
        self.waveform.zoomTrackRequested.connect(self._zoom_track_by_id)
        self.waveform.toggleLockRequested.connect(self.session.toggle_lock)

        self.table = TrackTable(self)
        self.table.set_on_edit(self._on_table_edit)
        self.table.selectionModel().selectionChanged.connect(self._on_table_selection)
        self.table.trackActivated.connect(self._on_track_activated)
        self.table.moveUpRequested.connect(self._move_track_up_by_id)
        self.table.moveDownRequested.connect(self._move_track_down_by_id)
        self.table.mergePreviousRequested.connect(self.session.merge_with_previous)
        self.table.renameRequested.connect(self._rename_track_dialog)
        self.table.toggleLockRequested.connect(self.session.toggle_lock)
        self.table.playRequested.connect(self._play_track_by_id)

        table_pane = QWidget(self)
        table_layout = QVBoxLayout(table_pane)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(2)
        table_layout.addWidget(self._build_track_edit_bar())
        table_layout.addWidget(self.table, 1)

        work = QSplitter(Qt.Vertical, self)
        work.addWidget(self.waveform)
        work.addWidget(table_pane)
        # The waveform is the working view and gets most of the room; the
        # table is there to be read and scrolled, not to be stared at.
        work.setStretchFactor(0, 4)
        work.setStretchFactor(1, 1)
        work.setSizes([520, 170])
        work.setChildrenCollapsible(False)
        self.work_splitter = work

        self.sidebar = self._build_sidebar()
        self.sidebar.setMinimumWidth(300)
        self.sidebar.setMaximumWidth(480)

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.addWidget(work)
        self.splitter.addWidget(self.sidebar)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([1140, 360])
        self.splitter.setChildrenCollapsible(False)

        self._build_transport()
        self._build_smart_banner()

        holder = QWidget(self)
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(0)
        # The transport is a fixed-height strip; the work area takes
        # everything else. Without this the spare height is shared out
        # evenly and the transport ends up floating in the middle of the
        # window.
        self.transport_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.transport_bar.setFixedHeight(self.transport_bar.sizeHint().height())
        holder_layout.addWidget(self.transport_bar, 0)
        holder_layout.addWidget(self.smart_banner, 0)
        holder_layout.addWidget(self.splitter, 1)
        self.setCentralWidget(holder)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("Ready")
        self.summary_label = QLabel("")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.summary_label)
        self.statusBar().addPermanentWidget(self.progress_bar)

    # ------------------------------------------------------------------
    def _build_transport(self) -> None:
        """The transport row: play, stop, the time, the selected track."""
        bar = QFrame(self)
        bar.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self.play_button = QPushButton("Play")
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setToolTip("Play from the playhead, or pause (Space). Selecting a track does not move the playhead.")
        self.play_button.setMinimumWidth(92)
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setToolTip("Stop (Esc)")
        self.stop_button.clicked.connect(self.stop_playback)
        layout.addWidget(self.stop_button)

        self.time_label = QLabel("0:00.000")
        self.time_label.setFont(theme.mono_font())
        self.time_label.setMinimumWidth(110)
        self.time_label.setToolTip("The position of the playhead")
        layout.addWidget(self.time_label)

        separator = QFrame(self)
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        self.play_track_button = QPushButton("Play selected track")
        self.play_track_button.setToolTip("Play the selected track from its start")
        self.play_track_button.clicked.connect(self.play_selected_track)
        layout.addWidget(self.play_track_button)

        self.position_label = QLabel("")
        self.position_label.setProperty("dim", True)
        layout.addWidget(self.position_label)

        layout.addStretch(1)

        # Undo and redo sit next to the playback controls: they are editing
        # actions, and the toolbar has no room for them.
        self.undo_button = QToolButton(self)
        self.undo_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        self.undo_button.setToolTip("Undo (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo)
        layout.addWidget(self.undo_button)

        self.redo_button = QToolButton(self)
        self.redo_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
        self.redo_button.setToolTip("Redo (Ctrl+Shift+Z)")
        self.redo_button.clicked.connect(self.redo)
        layout.addWidget(self.redo_button)

        self.sidebar_button = QToolButton(self)
        self.sidebar_button.setText("Hide details")
        self.sidebar_button.setCheckable(True)
        self.sidebar_button.setChecked(True)
        self.sidebar_button.setToolTip("Show or hide the details panel")
        self.sidebar_button.toggled.connect(self._toggle_sidebar)
        layout.addWidget(self.sidebar_button)

        self.transport_bar = bar
        return bar

    def _build_smart_banner(self) -> QFrame:
        """Compact, reversible inline bar showing identified release and edition picker."""
        bar = QFrame(self)
        bar.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)

        self.smart_banner_label = QLabel("")
        self.smart_banner_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout.addWidget(self.smart_banner_label)

        self.release_combo = QComboBox(self)
        self.release_combo.setMinimumWidth(260)
        self.release_combo.setMaximumWidth(480)
        self.release_combo.setToolTip("Choose the matching release or edition")
        self.release_combo.currentIndexChanged.connect(self._on_release_candidate_changed)
        layout.addWidget(self.release_combo, 1)

        self.revert_metadata_button = QPushButton("Audio-only", self)
        self.revert_metadata_button.setToolTip(
            "Switch back to audio-only boundary detection (Ctrl+Z also undoes smart split)"
        )
        self.revert_metadata_button.clicked.connect(self._revert_to_audio_only)
        layout.addWidget(self.revert_metadata_button)

        self.services_button = QPushButton("Services...", self)
        self.services_button.setToolTip("Configure Music Services (AcoustID, setlist.fm, MusicBrainz, LRCLIB)")
        self.services_button.clicked.connect(self.open_music_services)
        layout.addWidget(self.services_button)

        close_btn = QToolButton(self)
        close_btn.setText("×")
        close_btn.setToolTip("Hide this bar")
        close_btn.clicked.connect(lambda: self.smart_banner.setVisible(False))
        layout.addWidget(close_btn)

        bar.setVisible(False)
        self.smart_banner = bar
        return bar

    def _build_track_edit_bar(self) -> QFrame:
        """Compact bar for direct track splitting, renaming, reordering, and zoom."""
        bar = QFrame(self)
        bar.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)

        self.split_button = QPushButton("Split here")
        self.split_button.setToolTip("Add a shared track divider at the cursor (S). Audio is never deleted.")
        self.split_button.clicked.connect(self.split_at_cursor)
        layout.addWidget(self.split_button)

        self.remove_split_button = QPushButton("Remove split")
        self.remove_split_button.setToolTip("Remove the selected divider and merge adjacent tracks (M). Audio is kept.")
        self.remove_split_button.clicked.connect(self._remove_selected_split)
        layout.addWidget(self.remove_split_button)

        self.rename_track_button = QPushButton("Rename")
        self.rename_track_button.setToolTip("Rename the selected track (F2)")
        self.rename_track_button.clicked.connect(lambda: self._rename_track_dialog(self._selected_track_id))
        layout.addWidget(self.rename_track_button)

        self.move_up_button = QPushButton("Move label up")
        self.move_up_button.setToolTip("Shift this track label up one slot while preserving chronological audio order")
        self.move_up_button.clicked.connect(self.move_selected_track_up)
        layout.addWidget(self.move_up_button)

        self.move_down_button = QPushButton("Move label down")
        self.move_down_button.setToolTip("Shift this track label down one slot while preserving chronological audio order")
        self.move_down_button.clicked.connect(self.move_selected_track_down)
        layout.addWidget(self.move_down_button)

        layout.addStretch(1)

        self.zoom_out_button = QToolButton(self)
        self.zoom_out_button.setText("−")
        self.zoom_out_button.setToolTip("Zoom out (-)")
        self.zoom_out_button.clicked.connect(lambda: self.waveform.zoom_by(1.4))
        layout.addWidget(self.zoom_out_button)

        self.zoom_in_button = QToolButton(self)
        self.zoom_in_button.setText("+")
        self.zoom_in_button.setToolTip("Zoom in (+)")
        self.zoom_in_button.clicked.connect(lambda: self.waveform.zoom_by(0.7))
        layout.addWidget(self.zoom_in_button)

        self.zoom_fit_button = QToolButton(self)
        self.zoom_fit_button.setText("Fit")
        self.zoom_fit_button.setToolTip("Zoom to fit the whole recording (0)")
        self.zoom_fit_button.clicked.connect(self.waveform.zoom_to_fit)
        layout.addWidget(self.zoom_fit_button)

        self.track_edit_bar = bar
        return bar

    # ------------------------------------------------------------------
    def _build_sidebar(self) -> QWidget:
        """The details panel: the boundary review, the track, the project.

        It scrolls, because the review controls need more room than a
        short window can give them.
        """
        from PySide6.QtWidgets import QScrollArea

        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(4, 4, 8, 4)
        outer.setSpacing(10)

        outer.addWidget(self._build_review_group())
        outer.addWidget(self._build_track_group())
        outer.addWidget(self._build_project_group())
        outer.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        return scroll

    def _build_review_group(self) -> QGroupBox:
        box = QGroupBox("Boundary review")
        layout = QVBoxLayout(box)
        layout.setSpacing(7)

        self.boundary_position_label = QLabel("No boundary selected")
        self.boundary_position_label.setFont(theme.mono_font())
        layout.addWidget(self.boundary_position_label)

        self.boundary_state_label = QLabel("")
        self.boundary_state_label.setWordWrap(True)
        layout.addWidget(self.boundary_state_label)

        self.margin_info_label = QLabel("")
        self.margin_info_label.setProperty("hint", True)
        self.margin_info_label.setWordWrap(True)
        layout.addWidget(self.margin_info_label)

        grid = QGridLayout()
        grid.setSpacing(4)
        self.zoom_boundary_button = QPushButton("Zoom to cut")
        self.zoom_boundary_button.clicked.connect(self.zoom_to_boundary)
        self.play_across_button = QPushButton("Play across")
        self.play_across_button.clicked.connect(self.play_across_boundary)
        grid.addWidget(self.zoom_boundary_button, 0, 0)
        grid.addWidget(self.play_across_button, 0, 1)
        layout.addLayout(grid)

        fine_label = QLabel("Precise divider nudge")
        fine_label.setProperty("dim", True)
        layout.addWidget(fine_label)

        fine = QGridLayout()
        fine.setSpacing(4)
        adjustments = [
            ("-1 s", -1.0), ("-100 ms", -0.1), ("-10 ms", -0.010),
            ("+10 ms", 0.010), ("+100 ms", 0.1), ("+1 s", 1.0),
        ]
        for index, (label, amount) in enumerate(adjustments):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, a=amount: self.fine_adjust(a))
            fine.addWidget(button, index // 3, index % 3)
        layout.addLayout(fine)

        accept_row = QHBoxLayout()
        self.accept_button = QPushButton("Accept cut")
        self.accept_button.setProperty("primary", True)
        self.accept_button.setToolTip("Accept this cut as it is")
        self.accept_button.clicked.connect(self.accept_boundary)
        self.lock_button = QPushButton("Lock")
        self.lock_button.setToolTip("Lock this track so re-analysis will not move it")
        self.lock_button.clicked.connect(self.lock_boundary)
        accept_row.addWidget(self.accept_button)
        accept_row.addWidget(self.lock_button)
        layout.addLayout(accept_row)

        # Advanced boundary controls: kept available, collapsed by default so
        # basic drag-and-drop track splitting stays uncluttered.
        self.advanced_boundary_toggle = QToolButton(self)
        self.advanced_boundary_toggle.setText("Advanced boundary controls")
        self.advanced_boundary_toggle.setCheckable(True)
        self.advanced_boundary_toggle.setChecked(False)
        self.advanced_boundary_toggle.setArrowType(Qt.RightArrow)
        self.advanced_boundary_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_boundary_toggle.toggled.connect(self._on_advanced_boundary_toggled)
        layout.addWidget(self.advanced_boundary_toggle)

        self.advanced_boundary_panel = QWidget(self)
        adv = QVBoxLayout(self.advanced_boundary_panel)
        adv.setContentsMargins(0, 4, 0, 0)
        adv.setSpacing(6)

        adv_play = QGridLayout()
        adv_play.setSpacing(4)
        self.play_next_button = QPushButton("Play next track")
        self.play_next_button.clicked.connect(self.play_next_track)
        self.play_previous_button = QPushButton("Play previous ending")
        self.play_previous_button.clicked.connect(self.play_previous_ending)
        self.preview_export_button = QPushButton("Preview exported start")
        self.preview_export_button.setToolTip(
            "Play from the exact frame the next track is exported from, so you "
            "hear what the file will start with."
        )
        self.preview_export_button.clicked.connect(self.preview_exported_start)
        adv_play.addWidget(self.play_next_button, 0, 0)
        adv_play.addWidget(self.play_previous_button, 0, 1)
        adv_play.addWidget(self.preview_export_button, 1, 0, 1, 2)
        adv.addLayout(adv_play)

        uncertain_row = QHBoxLayout()
        self.prev_uncertain_button = QPushButton("< Uncertain")
        self.prev_uncertain_button.setToolTip("Go to the previous boundary that needs review")
        self.prev_uncertain_button.clicked.connect(lambda: self.step_uncertain(-1))
        self.next_uncertain_button = QPushButton("Uncertain >")
        self.next_uncertain_button.setToolTip("Go to the next boundary that needs review")
        self.next_uncertain_button.clicked.connect(lambda: self.step_uncertain(1))
        uncertain_row.addWidget(self.prev_uncertain_button)
        uncertain_row.addWidget(self.next_uncertain_button)
        adv.addLayout(uncertain_row)

        adv.addWidget(self._separator())

        placement_label = QLabel("Where to place new cuts")
        placement_label.setProperty("dim", True)
        adv.addWidget(placement_label)

        self.placement_combo = QComboBox()
        from ..analysis.placement import PLACEMENT_LABELS, PLACEMENT_MODES

        for mode in PLACEMENT_MODES:
            self.placement_combo.addItem(PLACEMENT_LABELS[mode], mode)
        self.placement_combo.setToolTip(
            "Near next song start puts the cut just before the music begins, so the "
            "next track does not start with a tail of silence."
        )
        self.placement_combo.currentIndexChanged.connect(self._on_placement_changed)
        adv.addWidget(self.placement_combo)

        self.placement_hint = QLabel("")
        self.placement_hint.setProperty("hint", True)
        self.placement_hint.setWordWrap(True)
        adv.addWidget(self.placement_hint)

        adv.addWidget(self._separator())

        margin_label = QLabel("Margin before the music")
        margin_label.setProperty("dim", True)
        margin_label.setToolTip(
            "How far in front of the first note the cut is placed. A small "
            "margin keeps the note from being clipped; a larger one keeps "
            "more of the run-up."
        )
        adv.addWidget(margin_label)

        margin_row = QHBoxLayout()
        self.pre_roll_spin = QDoubleSpinBox()
        self.pre_roll_spin.setRange(0.0, 2.0)
        self.pre_roll_spin.setSingleStep(0.005)
        self.pre_roll_spin.setDecimals(3)
        self.pre_roll_spin.setSuffix(" s")
        self.pre_roll_spin.setValue(self.session.project.placement_pre_roll_seconds)
        self.pre_roll_spin.setToolTip(margin_label.toolTip())
        self.pre_roll_spin.valueChanged.connect(self._on_pre_roll_changed)
        margin_row.addWidget(self.pre_roll_spin, 1)

        self.margin_scope_combo = QComboBox()
        self.margin_scope_combo.addItem("This cut", "this_cut")
        self.margin_scope_combo.addItem("New suggestions", "new_suggestions")
        self.margin_scope_combo.setToolTip(
            "Whether the margin applies to the cut you are looking at, or to "
            "the cuts that are proposed from now on."
        )
        self.margin_scope_combo.currentIndexChanged.connect(self._on_margin_scope_changed)
        margin_row.addWidget(self.margin_scope_combo)
        adv.addLayout(margin_row)

        presets = QHBoxLayout()
        self.tight_button = QPushButton("Tight start")
        self.tight_button.setToolTip(
            "A 20 ms margin: the track starts as close to the first note as "
            "it can without clipping it."
        )
        self.tight_button.clicked.connect(lambda: self._apply_preset("tight"))
        self.lead_in_button = QPushButton("Keep more lead-in")
        self.lead_in_button.setToolTip(
            "A 150 ms margin: more of the run-up stays with the new track."
        )
        self.lead_in_button.clicked.connect(lambda: self._apply_preset("lead_in"))
        presets.addWidget(self.tight_button)
        presets.addWidget(self.lead_in_button)
        adv.addLayout(presets)

        self.refine_button = QPushButton("Refine existing boundaries...")
        self.refine_button.setToolTip(
            "Look at the existing cuts again and propose better ones. "
            "Nothing changes until you accept the proposal."
        )
        self.refine_button.clicked.connect(self.refine_boundaries)
        adv.addWidget(self.refine_button)

        self.advanced_boundary_panel.setVisible(False)
        layout.addWidget(self.advanced_boundary_panel)

        return box

    def _build_track_group(self) -> QGroupBox:
        box = QGroupBox("Selected track")
        form = QFormLayout(box)
        form.setSpacing(7)

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Untitled")
        self.title_edit.editingFinished.connect(self._on_title_edited)
        form.addRow("Title", self.title_edit)

        self.artist_edit = QLineEdit()
        self.artist_edit.setPlaceholderText("Recognized or typed artist")
        self.artist_edit.editingFinished.connect(self._on_artist_edited)
        form.addRow("Artist", self.artist_edit)

        self.section_combo = QComboBox()
        for section in SECTION_TYPES:
            self.section_combo.addItem(section.replace("_", " ").capitalize(), section)
        self.section_combo.currentIndexChanged.connect(self._on_section_changed)
        form.addRow("Section", self.section_combo)

        self.boundary_label = QLabel("-")
        self.identity_label = QLabel("-")
        form.addRow("Boundary", self.boundary_label)
        form.addRow("Identity", self.identity_label)

        self.lock_check = QCheckBox("Lock this track")
        self.lock_check.setToolTip("A locked track keeps its boundaries and its metadata")
        self.lock_check.toggled.connect(self._on_lock_toggled)
        form.addRow("", self.lock_check)

        self.reasons_text = QTextEdit()
        self.reasons_text.setReadOnly(True)
        self.reasons_text.setMinimumHeight(96)
        self.reasons_text.setToolTip("Why this cut was proposed, in plain language")
        form.addRow("Evidence", self.reasons_text)

        self.details_toggle = QToolButton(self)
        self.details_toggle.setText("Technical details")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setChecked(False)
        self.details_toggle.setArrowType(Qt.RightArrow)
        self.details_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.details_toggle.toggled.connect(self._on_details_toggled)
        form.addRow("", self.details_toggle)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setVisible(False)
        self.details_text.setFont(theme.mono_font())
        self.details_text.setMinimumHeight(90)
        form.addRow("", self.details_text)

        return box

    def _build_project_group(self) -> QGroupBox:
        box = QGroupBox("Project")
        form = QFormLayout(box)
        form.setSpacing(7)

        self.artist_edit = QLineEdit()
        self.artist_edit.setPlaceholderText("Not set")
        self.artist_edit.editingFinished.connect(
            lambda: self.session.set_project_metadata("artist", self.artist_edit.text())
        )
        self.album_edit = QLineEdit()
        self.album_edit.setPlaceholderText("Not set")
        self.album_edit.editingFinished.connect(
            lambda: self.session.set_project_metadata("album", self.album_edit.text())
        )
        self.date_edit = QLineEdit()
        self.date_edit.setPlaceholderText("YYYY-MM-DD")
        self.date_edit.editingFinished.connect(
            lambda: self.session.set_project_metadata("date", self.date_edit.text())
        )
        self.venue_edit = QLineEdit()
        self.venue_edit.setPlaceholderText("Not set")
        self.venue_edit.editingFinished.connect(
            lambda: self.session.set_project_metadata("venue", self.venue_edit.text())
        )
        form.addRow("Artist", self.artist_edit)
        form.addRow("Album", self.album_edit)
        form.addRow("Date", self.date_edit)
        form.addRow("Venue", self.venue_edit)

        more = QPushButton("All metadata...")
        more.clicked.connect(self._edit_metadata)
        form.addRow("", more)

        return box

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    # ------------------------------------------------------------------
    def _build_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        self._action(file_menu, "&New project", "Ctrl+N", self.new_project)
        self._action(file_menu, "&Open project...", "Ctrl+O", self.open_project)
        self._action(file_menu, "&Save project", "Ctrl+S", self.save_project)
        self._action(file_menu, "Save project &as...", "Ctrl+Shift+S", self.save_project_as)
        file_menu.addSeparator()
        self._action(file_menu, "&Import audio...", "Ctrl+I", self.import_audio)
        self._action(file_menu, "Source files and &spacing...", None, self.edit_sources)
        file_menu.addSeparator()
        self._action(file_menu, "&Export...", "Ctrl+E", self.export_project)
        self._action(file_menu, "Export &markers...", None, self.export_markers)
        self._action(file_menu, "Import markers...", None, self.import_markers)
        file_menu.addSeparator()
        self._action(file_menu, "&Music Services...", "Ctrl+,", self.open_music_services)
        file_menu.addSeparator()
        self._action(file_menu, "E&xit", "Ctrl+Q", self.close)

        edit_menu = menubar.addMenu("&Edit")
        self._action(edit_menu, "&Undo", "Ctrl+Z", self.undo)
        self._action(edit_menu, "&Redo", "Ctrl+Shift+Z", self.redo)
        edit_menu.addSeparator()
        self._action(edit_menu, "&Split at cursor", "S", self.split_at_cursor)
        self._action(edit_menu, "&Merge with previous track", "M", self.merge_with_previous)
        self._action(edit_menu, "Remove selected &split", None, self._remove_selected_split)
        self._action(edit_menu, "Remo&ve track (audio is kept)", "Backspace", self.remove_track)
        edit_menu.addSeparator()
        self._action(edit_menu, "Move track label &up (keeps audio order)", "Alt+Up", self.move_selected_track_up)
        self._action(edit_menu, "Move track label &down (keeps audio order)", "Alt+Down", self.move_selected_track_down)
        edit_menu.addSeparator()
        self._action(edit_menu, "Nudge boundary &-10 ms", "[", lambda: self._nudge(-0.010))
        self._action(edit_menu, "Nudge boundary &+10 ms", "]", lambda: self._nudge(0.010))
        self._action(edit_menu, "Nudge boundary -1 s", "Shift+[", lambda: self._nudge(-1.0))
        self._action(edit_menu, "Nudge boundary +1 s", "Shift+]", lambda: self._nudge(1.0))
        edit_menu.addSeparator()
        self._action(edit_menu, "&Lock / unlock track", "L", self.toggle_lock)
        self._action(edit_menu, "&Accept all suggestions", "A", self.accept_suggestions)

        analyse_menu = menubar.addMenu("&Analyse")
        self._action(analyse_menu, "Smart &identify && split", "Ctrl+Shift+I", self.run_smart_split)
        self._action(analyse_menu, "Suggest &boundaries", "Ctrl+R", self.run_analysis)
        sensitivity = analyse_menu.addMenu("Sensitivity")
        for name in ("conservative", "balanced", "sensitive"):
            action = QAction(name.capitalize(), self)
            action.triggered.connect(lambda _checked=False, n=name: self.run_analysis(n))
            sensitivity.addAction(action)
        analyse_menu.addSeparator()
        self._action(analyse_menu, "&Theatre segmentation...", "Ctrl+Shift+R", self.run_theater_analysis)
        theatre = analyse_menu.addMenu("Theatre")
        self._action(theatre, "Segment the recording", None, self.run_theater_analysis)
        self._action(theatre, "Segment it and transcribe it too", None,
                    lambda: self.run_theater_analysis(transcribe=True))
        self._action(theatre, "Re-analyse from scratch (ignore cache)", None,
                    lambda: self.run_theater_analysis(transcribe=True, reanalyze=True))
        self.separate_applause_action = theatre.addAction("Separate applause tracks (explicit opt-in)")
        self.separate_applause_action.setCheckable(True)
        self._action(theatre, "Propose names from the reference list", None,
                    self.propose_theater_names)
        self._action(theatre, "Load a reference list...", None, self.load_reference_list)
        self._action(theatre, "Load local reference audio manifest...", None,
                     self.load_reference_audio_manifest)
        self._action(theatre, "Match reference lyrics…", None, self.match_reference_lyrics)
        self._action(theatre, "Review lyric anchors and gaps…", None, self.review_lyric_matches)
        analyse_menu.addSeparator()
        self._action(analyse_menu, "Paste &tracklist...", "Ctrl+T", self.paste_tracklist)
        self._action(analyse_menu, "&Align with the tracklist", "Ctrl+L", self.align_with_tracklist)
        analyse_menu.addSeparator()
        self._action(analyse_menu, "&Look up an album (MusicBrainz)...", None, self.lookup_album)
        self._action(analyse_menu, "Look up a &concert (setlist.fm)...", None, self.lookup_concert)

        review_menu = menubar.addMenu("&Review")
        self._action(review_menu, "&Zoom to the selected cut", "Z", self.zoom_to_boundary)
        self._action(review_menu, "Play &across the cut", "Return", self.play_across_boundary)
        self._action(review_menu, "Play the &next track", None, self.play_next_track)
        self._action(review_menu, "Play the &previous track's ending", None, self.play_previous_ending)
        review_menu.addSeparator()
        self._action(review_menu, "&Previous uncertain boundary", "Ctrl+[", lambda: self.step_uncertain(-1))
        self._action(review_menu, "&Next uncertain boundary", "Ctrl+]", lambda: self.step_uncertain(1))
        review_menu.addSeparator()
        self._action(review_menu, "&Refine existing boundaries...", None, self.refine_boundaries)

        playback_menu = menubar.addMenu("&Playback")
        self._action(playback_menu, "&Play / pause", "Space", self.toggle_playback)
        self._action(playback_menu, "S&top", "Esc", self.stop_playback)
        self._action(playback_menu, "Play the &selected track", None, self.play_selected_track)
        self._action(playback_menu, "Playback diagnostics...", None, self._show_playback_diagnostics)
        playback_menu.addSeparator()
        devices_menu = playback_menu.addMenu("Output device")
        devices_menu.aboutToShow.connect(lambda: self._populate_output_devices(devices_menu))
        before = QDoubleSpinBox()
        before.setRange(0.0, 30.0)
        before.setValue(self._preview_before)
        before.setSuffix(" s")
        before.setToolTip("How much audio to play before the boundary")
        before.valueChanged.connect(lambda value: setattr(self, "_preview_before", value))
        before_action = QWidgetAction(playback_menu)
        before_action.setDefaultWidget(before)
        playback_menu.addAction(before_action)

        after = QDoubleSpinBox()
        after.setRange(0.0, 30.0)
        after.setValue(self._preview_after)
        after.setSuffix(" s")
        after.setToolTip("How much audio to play after the boundary")
        after.valueChanged.connect(lambda value: setattr(self, "_preview_after", value))
        after_action = QWidgetAction(playback_menu)
        after_action.setDefaultWidget(after)
        playback_menu.addAction(after_action)

        view_menu = menubar.addMenu("&View")
        self._action(view_menu, "Zoom &in", "Ctrl++", lambda: self.waveform.zoom_by(0.7))
        self._action(view_menu, "Zoom &out", "Ctrl+-", lambda: self.waveform.zoom_by(1.4))
        self._action(view_menu, "Zoom to &fit", "0", self.waveform.zoom_to_fit)
        self._action(view_menu, "Zoom to &selection", "F", self.zoom_to_selection)
        view_menu.addSeparator()

        theme_menu = view_menu.addMenu("&Theme")
        self._theme_group = QActionGroup(self)
        for mode in (theme.THEME_LIGHT, theme.THEME_DARK, theme.THEME_SYSTEM):
            action = QAction(theme.THEME_LABELS[mode], self, checkable=True)
            action.setData(mode)
            action.triggered.connect(lambda _checked=False, m=mode: self.set_theme(m))
            self._theme_group.addAction(action)
            theme_menu.addAction(action)
        view_menu.addSeparator()
        self._action(view_menu, "&Project metadata...", None, self._edit_metadata)
        self._action(view_menu, "&Keyboard shortcuts...", "F1", self.show_shortcuts)

        help_menu = menubar.addMenu("&Help")
        self._action(help_menu, "&About SplytNyt", None, self.show_about)

        self._build_toolbars()

    def _build_toolbars(self) -> None:
        """Four grouped toolbars: Import, Analyse, Review, Export.

        The labels are short on purpose: a row of toolbars that does not
        fit on a normal screen puts its last buttons out of reach.
        """
        self.toolbars: list[QToolBar] = []

        import_bar = self._group_toolbar("Import")
        self._action(import_bar, "Import audio", "Ctrl+I", self.import_audio,
                    icon=QStyle.SP_DialogOpenButton)
        self._action(import_bar, "Open", "Ctrl+O", self.open_project,
                    icon=QStyle.SP_DialogOpenButton)
        self._action(import_bar, "Save", "Ctrl+S", self.save_project,
                    icon=QStyle.SP_DialogSaveButton)

        analyse_bar = self._group_toolbar("Analyse")
        self._action(analyse_bar, "Smart split", "Ctrl+Shift+I", self.run_smart_split,
                    icon=QStyle.SP_DialogApplyButton)
        self._action(analyse_bar, "Suggest boundaries", "Ctrl+R", self.run_analysis,
                    icon=QStyle.SP_BrowserReload)
        self._action(analyse_bar, "Look up", None, self.lookup_album)

        review_bar = self._group_toolbar("Review")
        self._action(review_bar, "Zoom to cut", "Z", self.zoom_to_boundary)
        self._action(review_bar, "Play across", "Return", self.play_across_boundary)
        self._action(review_bar, "Refine", None, self.refine_boundaries)

        export_bar = self._group_toolbar("Export")
        self._action(export_bar, "Export tracks", "Ctrl+E", self.export_project,
                    icon=QStyle.SP_DialogSaveButton)
        self._action(export_bar, "Markers", None, self.export_markers)

    def _group_toolbar(self, title: str) -> QToolBar:
        bar = QToolBar(title, self)
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        bar.setIconSize(bar.iconSize())
        label = QLabel(f"  {title}  ")
        label.setProperty("dim", True)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        action = QWidgetAction(bar)
        action.setDefaultWidget(label)
        bar.addAction(action)
        self.addToolBar(bar)
        self.toolbars.append(bar)
        return bar

    def _action(
        self, menu, text: str, shortcut: Optional[str], handler, icon=None
    ) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if icon is not None:
            action.setIcon(self.style().standardIcon(icon))
        action.triggered.connect(lambda _checked=False: handler())
        menu.addAction(action)
        return action

    # ------------------------------------------------------------------
    def _connect_session(self) -> None:
        self.session.on(TRACKS_CHANGED, self._reload)
        self.session.on(PROJECT_CHANGED, self._reload)
        self.session.on(STATUS, lambda message: self.status_label.setText(message))
        self.session.on(BUSY, self._on_busy)

    # ------------------------------------------------------------------
    # Reloading the views
    # ------------------------------------------------------------------
    def _reload(self) -> None:
        project = self.session.project
        # The table is rebuilt from scratch, which clears the selection. The
        # track that was being reviewed is selected again, so that "accept"
        # and "lock" still act on it after an edit.
        selected_id = self._selected_track_id
        self.session.refresh_flags()
        self.waveform.set_project(project, self._pyramids)
        self.waveform.set_tracks(project.tracks)
        self.table.set_tracks(project.tracks, project.sample_rate() or 44100)
        if selected_id:
            self.table.select_track(selected_id)
        self._on_table_selection()
        self._update_project_fields()
        self._update_placement_controls()
        self._update_details()
        self._update_summary()
        self._rebuild_reader()

    def _update_project_fields(self) -> None:
        """Show the project's metadata in the quick-edit fields."""
        meta = self.session.project.metadata
        for widget, value in (
            (self.artist_edit, meta.artist),
            (self.album_edit, meta.album),
            (self.date_edit, meta.date),
            (self.venue_edit, meta.venue),
        ):
            widget.blockSignals(True)
            widget.setText(value or "")
            widget.blockSignals(False)

    def _update_placement_controls(self) -> None:
        project = self.session.project
        self.placement_combo.blockSignals(True)
        index = self.placement_combo.findData(project.placement_mode)
        self.placement_combo.setCurrentIndex(max(0, index))
        self.placement_combo.blockSignals(False)
        self._update_placement_hint()
        self._sync_margin_spinbox()

    def _update_placement_hint(self) -> None:
        from ..analysis.placement import PLACEMENT_HINTS

        mode = self.placement_combo.currentData()
        self.placement_hint.setText(PLACEMENT_HINTS.get(mode, ""))

    def _on_busy(self, active: bool, message: str = "") -> None:
        """Called when the service layer enters or leaves a busy state."""
        if active:
            self.progress_bar.setVisible(True)
            if message:
                self.status_label.setText(message)
        else:
            self.progress_bar.setVisible(False)

    def _update_summary(self) -> None:
        summary = self.session.summary()
        rate = self.session.project.sample_rate() or 44100
        total = self.session.project.total_frames() / rate if rate else 0.0
        parts = [
            f"{summary['track_count']} tracks",
            f"{int(total // 60)}:{int(total % 60):02d} total",
            f"{summary['source_count']} source(s)",
        ]
        if summary["unidentified_tracks"]:
            parts.append(f"{summary['unidentified_tracks']} unnamed")
        uncertain = sum(1 for track in self.session.project.tracks if track.boundary_uncertain)
        if uncertain:
            parts.append(f"{uncertain} need review")
        issues = self.session.source_issues()
        if issues:
            parts.append(f"{len(issues)} issue(s)")
        self.summary_label.setText("   |   ".join(parts))
        self.summary_label.setToolTip("\n".join(issues) if issues else "")

    def _update_details(self) -> None:
        track = next((t for t in self.session.project.tracks
                      if t.track_id == self._selected_track_id), None)
        if track is None:
            self.title_edit.setText("")
            self.artist_edit.setText("")
            self.section_combo.setCurrentIndex(0)
            self.boundary_label.setText("Not assessed")
            self.identity_label.setText("Not assessed")
            self.lock_check.setChecked(False)
            self.reasons_text.setPlainText("")
            self.details_text.setPlainText("")
        else:
            self.title_edit.blockSignals(True)
            self.title_edit.setText(track.title)
            self.title_edit.blockSignals(False)
            self.artist_edit.blockSignals(True)
            self.artist_edit.setText(track.metadata.artist or track.recognized_artist or "")
            self.artist_edit.blockSignals(False)
            self.section_combo.blockSignals(True)
            index = self.section_combo.findData(track.section_type)
            self.section_combo.setCurrentIndex(max(0, index))
            self.section_combo.blockSignals(False)
            self.boundary_label.setText(track.boundary_assessment())
            from .track_table import TrackTableModel

            self.identity_label.setText(TrackTableModel._identity_text(track))
            self.lock_check.blockSignals(True)
            self.lock_check.setChecked(track.locked)
            self.lock_check.blockSignals(False)
            lines = list(track.reasons)
            from ..core.flags import describe_flag

            lines.extend(describe_flag(flag) for flag in track.flags)
            self.reasons_text.setPlainText("\n".join(lines) if lines else "No evidence recorded.")
            self.details_text.setPlainText(self._technical_details(track))
        self._update_boundary_panel()

    def _technical_details(self, track) -> str:
        """The numbers behind the decision, for the expandable panel."""
        rate = self.session.project.sample_rate() or 44100
        lines = [
            f"start            {frames_to_timestamp(track.start_frame, rate)}"
            f"  ({track.start_frame} frames)",
            f"end              {frames_to_timestamp(track.end_frame, rate)}"
            f"  ({track.end_frame} frames)",
            f"length           {track.frames} frames"
            f"  ({track.frames / rate:.3f} s)",
            f"boundary score   {track.boundary_confidence:.3f}",
            f"identity score   {track.identity_confidence:.3f}",
            f"placement        {track.boundary_placement or 'not set'}",
        ]
        if track.onset_frame >= 0:
            lines.append(
                f"musical onset    {frames_to_timestamp(track.onset_frame, rate)}"
                f"  ({track.onset_frame} frames)"
            )
        if track.transition_start >= 0:
            lines.append(
                f"transition       {frames_to_timestamp(track.transition_start, rate)}"
                f" - {frames_to_timestamp(track.transition_end, rate)}"
            )
        grouping = self.session.project.cache.get("musical_grouping", {})
        for group in grouping.get("groups", []):
            if group["start_frame"] < track.end_frame and group["end_frame"] > track.start_frame:
                for part in group.get("evidence", {}).get("internal_sections", []):
                    lines.append(f"internal activity {frames_to_timestamp(part['start_frame'], rate)} - "
                                 f"{frames_to_timestamp(part['end_frame'], rate)}: {part['section_type']}")
        for decision in grouping.get("decisions", []):
            if track.start_frame <= decision["start_frame"] < track.end_frame:
                lines.append(f"grouping {'joined' if decision['merged'] else 'kept separate'}: {decision['reason']}")
                lines.append(f"grouping evidence {decision['evidence']}")
        from ..analysis.lyrics import evidence_signature
        lyric_report = self.session.project.cache.get("lyric_review", {})
        if lyric_report.get('evidence_signature') != evidence_signature(self.session.project):
            lyric_report = {}
        for result in lyric_report.get("results", []):
            for occurrence in result["occurrences"]:
                if occurrence["start"] * rate < track.end_frame and occurrence["end"] * rate > track.start_frame:
                    lines.append(f"lyric proposal   {result['title']}: {occurrence['status']} (not verified)")
                    lines.append(f"lyric source     {result['provenance']}")
                    for anchor in occurrence["anchors"]:
                        lines.append(f"{anchor['role']} anchor {anchor['start']:.2f}–{anchor['end']:.2f}s: {anchor['text']} (similarity {anchor['score']})")
                    for gap in occurrence["gaps"]:
                        lines.append(f"uncertain gap    {gap['start']:.2f}–{gap['end']:.2f}s: {gap['reason']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Boundary review
    # ------------------------------------------------------------------
    def _current_boundary(self) -> Optional[int]:
        """The cut being reviewed.

        The frame is remembered so that the panel keeps its place while the
        project is reloaded. A cut can move, though: an undo, a refinement or
        a margin change all move it. So the remembered frame is checked
        against the project, and the selected track is used when the frame no
        longer belongs to a boundary - otherwise the panel would talk about a
        cut that is not there any more.
        """
        frame = self._selected_boundary
        if frame is not None:
            total = self.session.project.total_frames()
            if self._track_starting_at(frame) is not None or frame >= total:
                return frame
            self._selected_boundary = None
            self.waveform.set_selected_boundary(None)
        track = self.table.selected_track()
        if track is not None:
            self._selected_boundary = track.start_frame
            self.waveform.set_selected_boundary(track.start_frame)
            return track.start_frame
        return None

    def _update_boundary_panel(self) -> None:
        frame = self._current_boundary()
        rate = self.session.project.sample_rate() or 44100
        has_boundary = frame is not None and 0 < frame < self.session.project.total_frames()
        uncertain_count = sum(
            1 for track in self.session.project.tracks if track.boundary_uncertain
        )

        for button in (
            self.zoom_boundary_button,
            self.play_across_button,
            self.play_next_button,
            self.play_previous_button,
            self.preview_export_button,
            self.accept_button,
            self.lock_button,
        ):
            button.setEnabled(has_boundary)

        # There is nothing to step to when no boundary needs a decision.
        self.prev_uncertain_button.setEnabled(has_boundary and uncertain_count > 0)
        self.next_uncertain_button.setEnabled(has_boundary and uncertain_count > 0)

        if not has_boundary:
            self.boundary_position_label.setText("No boundary selected")
            self.boundary_state_label.setText(
                "Click a cut in the waveform, or pick a track in the table."
            )
            self.margin_info_label.setText("")
            self.waveform.set_onset_marker(None)
            return

        self.boundary_position_label.setText(frames_to_timestamp(frame, rate))
        track = self._track_starting_at(frame)
        if track is None:
            self.boundary_state_label.setText("This cut is at the end of the recording.")
            self.margin_info_label.setText("")
            self.waveform.set_onset_marker(None)
            return

        self.boundary_state_label.setText(self._describe_boundary(track))
        self.margin_info_label.setText(self._describe_margin(track, rate))
        self.waveform.set_onset_marker(
            track.onset_frame if track.onset_frame > 0 else None
        )
        self.lock_button.setText("Unlock" if track.locked else "Lock")

    def _describe_boundary(self, track) -> str:
        """Say what the assessment means, in words.

        A word on its own ("Low") tells the reader nothing. This says what
        the situation actually is, so nobody has to guess why a cut is
        marked the way it is.
        """
        if track.boundary_basis == "manual":
            return "You placed this cut by hand."
        if track.boundary_basis == "accepted":
            return "You accepted this cut. It is kept as it is."
        if track.boundary_basis == "none":
            return "This cut has not been assessed."

        reasons = list(track.reasons)
        evidence = reasons[0] if reasons else ""
        if track.boundary_uncertain:
            return "Needs review - " + (evidence or "the detector was not sure.")
        word = track.boundary_assessment()
        if word == "High":
            return "High - " + (evidence or "the evidence behind this cut is strong.")
        if word == "Fair":
            return "Fair - " + (
                evidence or "there is some evidence here, but not much."
            )
        return "Low - " + (
            evidence or "there is little evidence behind this cut; listen to it."
        )

    def _describe_margin(self, track, rate: int) -> str:
        """Show where the music begins and how much margin was applied."""
        if track.onset_frame is None or track.onset_frame <= 0:
            return "No musical onset was found for this cut."
        onset = frames_to_timestamp(track.onset_frame, rate)
        margin = track.boundary_pre_roll_seconds
        if margin is None or margin < 0:
            return f"The music was detected to begin at {onset}."
        return (
            f"The music was detected to begin at {onset}. "
            f"The cut is {margin * 1000:.0f} ms in front of it."
        )

    def _track_starting_at(self, frame: int):
        for track in self.session.project.tracks:
            if track.start_frame == frame:
                return track
        return None

    def _track_ending_at(self, frame: int):
        for track in self.session.project.tracks:
            if track.end_frame == frame:
                return track
        return None

    def zoom_to_boundary(self) -> None:
        frame = self._current_boundary()
        if frame is None:
            self.session.status("Select a boundary first.")
            return
        self.waveform.zoom_to_frame(frame, span_seconds=12.0)
        self.waveform.set_cursor(frame)
        self.session.status(f"Zoomed to the cut at {frames_to_timestamp(frame, self.session.project.sample_rate() or 44100)}")

    def play_across_boundary(self) -> None:
        """Play a window around the cut so it can be judged in context."""
        frame = self._current_boundary()
        if frame is None:
            self.session.status("Select a boundary first.")
            return
        if self._playback is None:
            self.session.status("No audio is loaded.")
            return
        rate = self.session.project.sample_rate() or 44100
        before = int(self._preview_before * rate)
        after = int(self._preview_after * rate)
        start = max(0, frame - before)
        end = min(self.session.project.total_frames(), frame + after)
        try:
            self._playback.play(start, end)
            self.session.status(
                f"Playing across the cut at {frames_to_timestamp(frame, rate)}"
            )
        except Exception as exc:
            QMessageBox.warning(self, "Playback", str(exc))

    def play_next_track(self) -> None:
        """Play the track that starts at the cut, from its exported start."""
        frame = self._current_boundary()
        if frame is None:
            self.session.status("Select a boundary first.")
            return
        track = self._track_starting_at(frame)
        if track is None:
            self.session.status("There is no track after this cut.")
            return
        self._play_range(track.start_frame, track.end_frame,
                         f"Playing {track.title or 'the next track'} from its start")

    def play_previous_ending(self) -> None:
        """Play the end of the track before the cut."""
        frame = self._current_boundary()
        if frame is None:
            self.session.status("Select a boundary first.")
            return
        track = self._track_ending_at(frame)
        if track is None:
            self.session.status("There is no track before this cut.")
            return
        rate = self.session.project.sample_rate() or 44100
        window = int(min(5.0, track.frames / rate) * rate)
        start = max(0, track.end_frame - window)
        self._play_range(start, track.end_frame,
                         f"Playing the end of {track.title or 'the previous track'}")

    def preview_exported_start(self) -> None:
        """Play the first moments of the file the next track is exported as.

        The export writes exactly the frames of the track, so the file
        starts at the track's first frame. Hearing those first moments is
        the only way to know whether the track starts on the note or with
        the tail of the song before it.
        """
        frame = self._current_boundary()
        if frame is None:
            self.session.status("Select a boundary first.")
            return
        track = self._track_starting_at(frame)
        if track is None:
            self.session.status("There is no track after this cut.")
            return
        rate = self.session.project.sample_rate() or 44100
        length = int(min(3.0, max(0.2, track.frames / rate)) * rate)
        self._play_range(
            track.start_frame,
            track.start_frame + length,
            f"Playing the exported start of {track.title or 'the next track'}",
        )

    def play_selected_track(self) -> None:
        track = next((t for t in self.session.project.tracks
                      if t.track_id == self._selected_track_id), None)
        if track is None:
            self.session.status("Select a track first.")
            return
        self._play_range(track.start_frame, track.end_frame,
                         f"Playing {track.title or 'the selected track'}")

    def _play_range(self, start: int, end: int, message: str) -> None:
        if self._playback is None:
            self.session.status("No audio is loaded.")
            return
        try:
            self._paused_end = None
            self._playback.play(start, end)
            self.session.status(message)
        except Exception as exc:
            QMessageBox.warning(self, "Playback", str(exc))

    def step_uncertain(self, direction: int) -> None:
        """Jump to the next boundary that still needs a human decision."""
        tracks = self.session.project.sorted_tracks()
        uncertain = [track.start_frame for track in tracks if track.boundary_uncertain]
        if not uncertain:
            self.session.status("No boundary needs review.")
            return
        current = self._current_boundary()
        if current is None:
            target = uncertain[0]
        elif direction > 0:
            later = [value for value in uncertain if value > current]
            target = later[0] if later else uncertain[0]
        else:
            earlier = [value for value in uncertain if value < current]
            target = earlier[-1] if earlier else uncertain[-1]
        self._select_boundary(target)
        self.zoom_to_boundary()

    def _select_boundary(self, frame: int) -> None:
        self._selected_boundary = frame
        self.waveform.set_selected_boundary(frame)
        self.waveform.set_cursor(frame)
        track = self._track_starting_at(frame)
        if track is not None:
            self.table.select_track(track.track_id)
            self.waveform.select_track(track.track_id)
        self._update_details()

    def fine_adjust(self, seconds: float) -> None:
        """Move the reviewed cut by a small amount."""
        frame = self._current_boundary()
        if frame is None:
            self.session.status("Select a boundary first.")
            return
        rate = self.session.project.sample_rate() or 44100
        delta = int(round(seconds * rate))
        if self.session.nudge_boundary(frame, delta):
            new_frame = frame + delta
            self._selected_boundary = new_frame
            self.waveform.set_selected_boundary(new_frame)
            self.waveform.set_cursor(new_frame)
            self.session.status(f"The cut moved by {seconds * 1000:+.0f} ms")
        else:
            self.session.status("That cut cannot move there.")

    def accept_boundary(self) -> None:
        """Accept the cut as it is."""
        frame = self._current_boundary()
        if frame is None:
            self.session.status("Select a boundary first.")
            return
        if not self.session.accept_boundary(frame):
            self.session.status("Select a boundary first.")
            return
        self.session.status(
            f"Accepted the cut at "
            f"{frames_to_timestamp(frame, self.session.project.sample_rate() or 44100)}"
        )
        self._reload()

    def lock_boundary(self) -> None:
        track = self.table.selected_track()
        if track is None:
            self.session.status("Select a track first.")
            return
        locked = self.session.toggle_lock(track.track_id)
        self.session.status("Track locked." if locked else "Track unlocked.")
        self._update_boundary_panel()

    def _on_details_toggled(self, checked: bool) -> None:
        self.details_text.setVisible(checked)
        self.details_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _on_placement_changed(self, index: int) -> None:
        mode = self.placement_combo.itemData(index)
        self._update_placement_hint()
        self.session.set_placement(mode)

    # ------------------------------------------------------------------
    # The margin before the music
    # ------------------------------------------------------------------
    def _on_margin_scope_changed(self, index: int) -> None:
        """Show the margin that belongs to the scope that was chosen."""
        self._sync_margin_spinbox()

    def _sync_margin_spinbox(self) -> None:
        """Put the right number in the margin box, without applying it."""
        scope = self.margin_scope_combo.currentData()
        value = self.session.project.placement_pre_roll_seconds
        if scope == "this_cut":
            frame = self._current_boundary()
            track = self._track_starting_at(frame) if frame is not None else None
            if track is not None and track.boundary_pre_roll_seconds >= 0:
                value = track.boundary_pre_roll_seconds
        self.pre_roll_spin.blockSignals(True)
        self.pre_roll_spin.setValue(value)
        self.pre_roll_spin.blockSignals(False)

    def _on_pre_roll_changed(self, value: float) -> None:
        """The margin changed: apply it to the scope that was chosen."""
        scope = self.margin_scope_combo.currentData()
        if scope == "this_cut":
            frame = self._current_boundary()
            if frame is None:
                self.session.status("Select a cut first.")
                return
            track = self._track_starting_at(frame)
            if track is None or track.onset_frame <= 0:
                self.session.status(
                    "This cut was not derived from a musical onset, so there is "
                    "no margin to change. Move it by hand instead."
                )
                self._sync_margin_spinbox()
                return
            new_frame = self.session.reposition_boundary(frame, value)
            if new_frame is None or new_frame < 0:
                self.session.status("That cut cannot move there.")
                self._sync_margin_spinbox()
                return
            self._selected_boundary = new_frame
            self.waveform.set_selected_boundary(new_frame)
            self.waveform.set_cursor(new_frame)
            self.session.status(
                f"The cut now sits {value * 1000:.0f} ms in front of the music."
            )
            self._reload()
        else:
            self.session.set_placement(self.session.project.placement_mode, value)
            self.session.status(
                f"New suggestions will be placed {value * 1000:.0f} ms in front "
                f"of the music."
            )

    def _apply_preset(self, name: str) -> None:
        """The "tight start" and "keep more lead-in" presets."""
        from ..analysis.placement import LEAD_IN_PRE_ROLL_SECONDS, TIGHT_PRE_ROLL_SECONDS

        value = TIGHT_PRE_ROLL_SECONDS if name == "tight" else LEAD_IN_PRE_ROLL_SECONDS
        self.pre_roll_spin.setValue(value)
        if self.margin_scope_combo.currentData() != "this_cut":
            self.session.status(
                "The preset applies to new suggestions. Choose \"This cut\" to "
                "move the cut you are looking at."
            )

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def _populate_output_devices(self, menu) -> None:
        from ..audio.playback import list_output_devices
        menu.clear()
        action = self._action(menu, "Windows default (WASAPI)", None, lambda: self._set_output_device(None))
        action.setCheckable(True)
        action.setChecked(self._output_device is None)
        try:
            for device in list_output_devices():
                index = device["index"]
                action = self._action(menu, f"{index}: {device['name']} ({device['hostapi']}; {device['channels']} ch; {device['default_samplerate']:g} Hz)", None,
                                      lambda checked=False, index=index: self._set_output_device(index))
                action.setCheckable(True)
                action.setChecked(self._output_device == index)
        except Exception as exc:
            QMessageBox.warning(self, "Audio output", str(exc))

    def _set_output_device(self, device) -> None:
        self.stop_playback()
        self._output_device = device
        if self._playback:
            self._playback.device = device
        self.session.status(f"Output selected: {device if device is not None else 'Windows default (WASAPI)'}; no automatic fallback")

    def _show_playback_diagnostics(self) -> None:
        import json
        if self._playback is None:
            QMessageBox.information(self, "Playback diagnostics", "Load audio first.")
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Playback diagnostics")
        dialog.setText(f"Output: {self._playback.device_name or 'not opened yet'}\n"
                       f"Trace: {self._playback.trace_path}\n"
                       "Callback samples prove submission only. Endpoint capture and listening verify output.")
        dialog.setDetailedText(json.dumps(self._playback.diagnostics, indent=2))
        dialog.exec()

    def _rebuild_reader(self) -> None:
        if self._playback is not None:
            self._playback.stop()
            self._playback = None
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        if self._playback is not None:
            self._playback.stop()
            self._playback = None
        project = self.session.project
        if not project.sources:
            return
        try:
            from ..audio.timeline import build_reader

            self._reader = build_reader(project)
            self._playback_bridge = PlaybackBridge(self)
            self._playback_bridge.position.connect(self._on_playback_position)
            self._playback_bridge.state.connect(self._on_playback_state)
            self._playback_bridge.error.connect(self._on_playback_error)
            self._playback = PlaybackEngine(
                self._reader,
                on_position=self._playback_bridge.position.emit,
                on_state=self._playback_bridge.state.emit,
                on_error=self._playback_bridge.error.emit,
                device=self._output_device,
            )
        except Exception as exc:
            self.session.status(f"Playback unavailable: {exc}")

    def _on_playback_error(self, message: str) -> None:
        self.session.status(message)
        QMessageBox.warning(self, "Audio output failed", message)

    def _on_playback_position(self, frame: int) -> None:
        self.waveform.set_playhead(frame)
        self._update_time_label(frame)

    def _on_playback_state(self, playing: bool) -> None:
        # A stopped worker's queued Qt signal can arrive after a replacement
        # stream starts (seek, preview, selected track, or resume). Render the
        # current engine state instead of letting that stale event stop the UI.
        if self._playback is not None:
            playing = self._playback.is_playing
        if playing:
            self._playback_timer.start()
        else:
            self._playback_timer.stop()
        self.play_button.setText("Pause" if playing else "Play")
        self.play_button.setIcon(
            self.style().standardIcon(
                QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay
            )
        )

    def _on_playback_tick(self) -> None:
        if self._playback is not None:
            self.waveform.set_playhead(self._playback.position)
            self._update_time_label(self._playback.position)

    def _update_time_label(self, frame: int) -> None:
        rate = self.session.project.sample_rate() or 44100
        total = self.session.project.total_frames()
        self.time_label.setText(
            f"{frames_to_timestamp(frame, rate)} / {frames_to_timestamp(total, rate)}"
        )

    def toggle_playback(self) -> None:
        if self._playback is None:
            self.session.status("No audio is loaded.")
            return
        if self._playback.is_playing:
            self._paused_end = self._playback._end_frame
            self._playback.stop()
            self.waveform.set_cursor(self._playback.position)
            self.session.status("Paused")
        else:
            frame = self.waveform.cursor_frame
            try:
                self._playback.play(frame, self._paused_end)
                self._paused_end = None
                self.session.status(
                    f"Playing from {frames_to_timestamp(frame, self.session.project.sample_rate() or 44100)}"
                )
            except Exception as exc:
                QMessageBox.warning(self, "Playback", str(exc))

    def stop_playback(self) -> None:
        if self.waveform.cancel_drag():
            return
        self._paused_end = None
        if self._playback is not None:
            self._playback.stop()
        if self._playback:
            self._playback.set_position(self.waveform.cursor_frame)
        self.waveform.set_playhead(self.waveform.cursor_frame)
        self._update_time_label(self.waveform.cursor_frame)

    def preview_boundary(self) -> None:
        """Play a window around the nearest boundary so the cut can be judged."""
        self.play_across_boundary()

    def _nearest_boundary_frame(self, frame: int) -> int:
        best = frame
        best_distance = None
        for track in self.session.project.tracks:
            for candidate in (track.start_frame, track.end_frame):
                distance = abs(candidate - frame)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best = candidate
        return best

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def _on_cursor_moved(self, frame: int) -> None:
        self._paused_end = None
        self.waveform.set_cursor(frame)
        if self._playback:
            try:
                if self._playback.is_playing:
                    self._playback.play(frame)
                else:
                    self._playback.set_position(frame)
                    self.waveform.set_playhead(frame)
                    self._update_time_label(frame)
            except Exception as exc:
                self._on_playback_error(str(exc))

    def _on_boundary_dragged(self, old_frame: int, new_frame: int) -> None:
        if not self.session.move_boundary(old_frame, new_frame):
            self.session.status("The cut cannot move there.")
            return
        self._selected_boundary = new_frame
        self.waveform.set_selected_boundary(new_frame)

    def _on_waveform_seek(self, frame: int) -> None:
        self._paused_end = None
        if self._playback and self._playback.is_playing:
            self._playback.stop()
        self._on_cursor_moved(frame)

    def _on_remove_split(self, boundary_frame: int) -> None:
        if self.session.remove_split(boundary_frame):
            if self._selected_boundary == boundary_frame:
                self._selected_boundary = None
                self.waveform.set_selected_boundary(None)

    def _remove_selected_split(self) -> None:
        if self._selected_boundary is not None:
            if self.session.remove_split(self._selected_boundary):
                self._selected_boundary = None
                self.waveform.set_selected_boundary(None)
                return
        if self._selected_track_id:
            self.session.merge_with_previous(self._selected_track_id)

    def _rename_track_dialog(self, track_id: str) -> None:
        track = next((t for t in self.session.project.tracks if t.track_id == track_id), None)
        if track is None:
            return
        text, ok = QInputDialog.getText(
            self,
            "Rename track",
            "Track title:",
            QLineEdit.Normal,
            track.title,
        )
        if ok:
            self.session.set_track_title(track_id, text.strip())

    def _play_track_by_id(self, track_id: str) -> None:
        self._on_track_selected(track_id)
        self.play_selected_track()

    def _zoom_track_by_id(self, track_id: str) -> None:
        self._on_track_selected(track_id)
        self.zoom_to_selection()

    def _move_track_up_by_id(self, track_id: str) -> None:
        tracks = self.session.sorted_tracks()
        idx = next((i for i, t in enumerate(tracks) if t.track_id == track_id), -1)
        if idx > 0 and self.session.move_track_up(track_id):
            new_tracks = self.session.sorted_tracks()
            if 0 <= idx - 1 < len(new_tracks):
                self._on_track_selected(new_tracks[idx - 1].track_id)

    def _move_track_down_by_id(self, track_id: str) -> None:
        tracks = self.session.sorted_tracks()
        idx = next((i for i, t in enumerate(tracks) if t.track_id == track_id), -1)
        if 0 <= idx < len(tracks) - 1 and self.session.move_track_down(track_id):
            new_tracks = self.session.sorted_tracks()
            if 0 <= idx + 1 < len(new_tracks):
                self._on_track_selected(new_tracks[idx + 1].track_id)

    def move_selected_track_up(self) -> None:
        if self._selected_track_id:
            self._move_track_up_by_id(self._selected_track_id)

    def move_selected_track_down(self) -> None:
        if self._selected_track_id:
            self._move_track_down_by_id(self._selected_track_id)

    def _on_advanced_boundary_toggled(self, expanded: bool) -> None:
        self.advanced_boundary_panel.setVisible(expanded)
        self.advanced_boundary_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _on_boundary_clicked(self, frame: int) -> None:
        self.waveform.set_cursor(frame)
        self._selected_boundary = frame
        self.waveform.set_selected_boundary(frame)
        self._select_boundary(frame)

    def _select_boundary_track(self, frame: int) -> None:
        for track in self.session.project.tracks:
            if track.start_frame == frame:
                self.table.select_track(track.track_id)
                self.waveform.select_track(track.track_id)
                return

    def _on_track_selected(self, track_id: str) -> None:
        self.table.select_track(track_id)
        self._on_table_selection()

    def _on_track_activated(self, track_id: str) -> None:
        track = next((t for t in self.session.project.tracks if t.track_id == track_id), None)
        if track is None:
            return
        self._on_track_selected(track_id)
        self._paused_end = None
        # Seeking by double-click never starts or redirects running audio.
        if self._playback and self._playback.is_playing:
            self._playback.stop()
        self._on_cursor_moved(track.start_frame)
        self.session.status(
            f"{track.title or 'Untitled'} - {frames_to_timestamp(track.start_frame, self.session.project.sample_rate() or 44100)}"
        )

    def _on_table_selection(self, *args) -> None:
        track = self.table.selected_track()
        self._selected_track_id = track.track_id if track else ""
        self.waveform.select_track(self._selected_track_id)
        self.play_track_button.setEnabled(track is not None)
        self.play_track_button.setToolTip(
            f"Play {track.title or 'Untitled'} from its start to its end. Selection does not autoplay."
            if track else "Select a track in the table or waveform first.")
        if track is not None:
            self._selected_boundary = track.start_frame
            self.waveform.set_selected_boundary(track.start_frame)
        else:
            self._selected_boundary = None
            self.waveform.set_selected_boundary(None)
        self._update_details()

    def _on_table_edit(self, track_id: str, field: str, value) -> None:
        if field == "title":
            self.session.set_track_title(track_id, str(value))
        elif field == "section_type":
            self.session.set_track_section_type(track_id, str(value))

    def _on_title_edited(self) -> None:
        track = self.table.selected_track()
        if track is None:
            return
        if self.title_edit.text() != track.title:
            self.session.set_track_title(track.track_id, self.title_edit.text())

    def _on_artist_edited(self) -> None:
        track = self.table.selected_track()
        if track is None:
            return
        new_artist = self.artist_edit.text().strip()
        cur_artist = (track.metadata.artist or "").strip()
        if new_artist != cur_artist:
            self.session.set_track_metadata(track.track_id, "artist", new_artist)

    def _on_section_changed(self, index: int) -> None:
        track = self.table.selected_track()
        if track is None:
            return
        value = self.section_combo.itemData(index)
        if value != track.section_type:
            self.session.set_track_section_type(track.track_id, value)

    def _on_lock_toggled(self, checked: bool) -> None:
        track = self.table.selected_track()
        if track is None:
            return
        if track.locked != checked:
            self.session.toggle_lock(track.track_id)

    def split_at_cursor(self) -> None:
        frame = self.waveform.cursor_frame
        track = self.session.track_for_frame(frame)
        if track is None:
            self.session.status("The cursor is not inside a track.")
            return
        new_track = self.session.split_at(frame)
        if new_track is None:
            self.session.status("The cursor is already on a boundary.")
            return
        self.table.select_track(new_track.track_id)
        self.session.status(f"Split at {frames_to_timestamp(frame, self.session.project.sample_rate() or 44100)}")

    def merge_with_previous(self) -> None:
        track = self.table.selected_track()
        if track is None:
            self.session.status("Select a track first.")
            return
        if not self.session.merge_with_previous(track.track_id):
            self.session.status("There is no previous track to merge with.")

    def remove_track(self) -> None:
        track = self.table.selected_track()
        if track is None:
            self.session.status("Select a track first.")
            return
        if not self.session.remove_track(track.track_id):
            self.session.status("A project needs at least one track.")
        else:
            self.session.status(
                "Track removed. Its audio was merged into a neighbour, so nothing was lost."
            )

    def _nudge(self, seconds: float) -> None:
        rate = self.session.project.sample_rate() or 44100
        delta = int(round(seconds * rate))
        frame = self._current_boundary()
        if frame is None:
            frame = self._nearest_boundary_frame(self.waveform.cursor_frame)
        if self.session.nudge_boundary(frame, delta):
            self._selected_boundary = frame + delta
            self.waveform.set_selected_boundary(frame + delta)
            self.waveform.set_cursor(frame + delta)
            self.session.status(f"The cut moved by {seconds * 1000:+.0f} ms")
        else:
            self.session.status("That cut cannot move there.")

    def toggle_lock(self) -> None:
        track = self.table.selected_track()
        if track is None:
            self.session.status("Select a track first.")
            return
        locked = self.session.toggle_lock(track.track_id)
        self.session.status("Track locked." if locked else "Track unlocked.")

    def undo(self) -> None:
        if self.session.undo_action():
            self.session.status("Undone")
        else:
            self.session.status("Nothing to undo")

    def redo(self) -> None:
        if self.session.redo_action():
            self.session.status("Redone")
        else:
            self.session.status("Nothing to redo")

    def zoom_to_selection(self) -> None:
        track = self.table.selected_track()
        if track is None:
            self.session.status("Select a track first.")
            return
        self.waveform.zoom_to_frame(track.start_frame, span_seconds=track.duration_seconds(
            self.session.project.sample_rate() or 44100) or 12.0)

    def _on_zoom_changed(self, value: float) -> None:
        rate = self.session.project.sample_rate() or 44100
        self.position_label.setText(f"{value / max(1, rate) * 1000:.1f} ms per pixel")

    def _toggle_sidebar(self, visible: bool) -> None:
        self._sidebar_visible = visible
        self.sidebar.setVisible(visible)
        self.sidebar_button.setText("Hide details" if visible else "Show details")
        if visible:
            self.splitter.setSizes([1140, 360])

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------
    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self.session.new_project()
        self._pyramids.clear()
        self._selected_boundary = None
        self._reload()
        self.waveform.zoom_to_fit()

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", "SplytNyt project (*.splytnyt *.json);;All files (*)"
        )
        if not path:
            return
        try:
            self.session.open_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open project", str(exc))
            return
        self._pyramids.clear()
        self._selected_boundary = None
        self._reload()
        self._rebuild_peaks()
        self.waveform.zoom_to_fit()

    def save_project(self) -> None:
        if not self.session.project.path:
            self.save_project_as()
            return
        try:
            self.session.save_project()
            self._clear_recovery()
        except Exception as exc:
            QMessageBox.critical(self, "Save project", str(exc))

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save project",
            self.session.project.path or f"{self.session.project.name}.splytnyt",
            "SplytNyt project (*.splytnyt)",
        )
        if not path:
            return
        if not path.lower().endswith(".splytnyt"):
            path += ".splytnyt"
        try:
            self.session.save_project(path)
            self._clear_recovery()
        except Exception as exc:
            QMessageBox.critical(self, "Save project", str(exc))

    def _confirm_discard(self) -> bool:
        if not self.session.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "The project has unsaved changes. Discard them?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            self.save_project()
            return True
        return True

    def import_audio(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import audio", "", SUPPORTED_FILTER
        )
        if not paths:
            return
        try:
            self.session.import_files(paths)
        except SessionError as exc:
            QMessageBox.warning(self, "Import", str(exc))
            return
        self._reload()
        self._rebuild_peaks()
        self.waveform.zoom_to_fit()

    def edit_sources(self) -> None:
        if not self.session.project.sources:
            self.session.status("No sources have been imported yet.")
            return
        dialog = SourcesDialog(self.session.project, self)
        if dialog.exec() != SourcesDialog.Accepted:
            return
        rate = self.session.project.sample_rate() or 44100
        gaps = dialog.gap_values()
        for index, gap in enumerate(gaps):
            source = self.session.project.sources[index + 1]
            self.session.set_source_gap(source.source_id, int(round(gap * rate)))
        self._reload()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    def _run_job(self, job: Job, on_success) -> None:
        if self._runner is not None and self._runner.isRunning():
            QMessageBox.information(self, "Busy", "Another job is still running.")
            return
        runner = JobRunner(job, self)
        self._runner = runner
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        # The job reports through the runner's signal, so the interface is
        # only ever touched on the main thread.
        runner.progress.connect(self._on_job_progress)
        runner.succeeded.connect(lambda result: self._on_job_success(job, result, on_success))
        runner.failed.connect(self._on_job_failed)
        runner.cancelled.connect(self._on_job_cancelled)
        runner.start()

    def _on_job_progress(self, fraction: float, message: str) -> None:
        self.progress_bar.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        self.status_label.setText(message)

    def _on_job_success(self, job, result, on_success) -> None:
        self.progress_bar.setVisible(False)
        self._runner = None
        if on_success is not None:
            on_success(result)

    def _on_job_failed(self, message: str) -> None:
        self.progress_bar.setVisible(False)
        self._runner = None
        QMessageBox.critical(self, "Job failed", message)

    def _on_job_cancelled(self) -> None:
        self.progress_bar.setVisible(False)
        self._runner = None
        self.session.status("Cancelled.")

    def _rebuild_peaks(self) -> None:
        if not self.session.project.sources:
            return
        cache_dir = os.path.join(self._app_data_dir(), "peaks")
        job = ImportJob(self.session.project, cache_dir=cache_dir)
        self._run_job(job, self._on_peaks_ready)

    def _on_peaks_ready(self, result) -> None:
        self._pyramids = result.pyramids
        self.waveform.set_pyramids(self._pyramids)
        self.waveform.viewport().update()
        if self._auto_smart_split_after_peaks:
            self._auto_smart_split_after_peaks = False
            self.run_smart_split()

    # ------------------------------------------------------------------
    # Drag-and-drop audio import + smart split
    # ------------------------------------------------------------------
    _DROP_AUDIO_EXTENSIONS = {
        ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aif", ".aiff", ".wma",
    }

    def dragEnterEvent(self, event) -> None:
        mime = event.mimeData()
        if mime and mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    ext = os.path.splitext(url.toLocalFile())[1].lower()
                    if ext in self._DROP_AUDIO_EXTENSIONS:
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event) -> None:
        mime = event.mimeData()
        paths: list[str] = []
        if mime and mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    local = url.toLocalFile()
                    if os.path.splitext(local)[1].lower() in self._DROP_AUDIO_EXTENSIONS:
                        paths.append(local)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self.import_dropped_files(paths)

    def import_dropped_files(self, paths: list[str], auto_smart_split: bool = True) -> None:
        """Import dragged audio files, build waveform peaks, and run Smart Split."""
        try:
            self.session.import_files(paths)
        except SessionError as exc:
            QMessageBox.warning(self, "Import", str(exc))
            return
        self._auto_smart_split_after_peaks = bool(auto_smart_split)
        self._reload()
        self._rebuild_peaks()
        self.waveform.zoom_to_fit()

    # ------------------------------------------------------------------
    # Smart metadata-assisted split
    # ------------------------------------------------------------------
    def run_smart_split(self, identification=None) -> None:
        if not self.session.project.sources:
            self.session.status("Import audio before running Smart split.")
            return
        job = SmartSplitJob(
            self.session.project,
            identification=identification,
            placement=self.session.placement_choice(),
        )
        self._run_job(job, self._on_smart_split_done)

    def _on_smart_split_done(self, outcome) -> None:
        ident = outcome.identification
        self._last_identification = ident
        count = self.session.apply_metadata_split(outcome)
        self.waveform.zoom_to_fit()
        self._update_smart_banner(ident)
        summary_bits = [f"{count} track(s) suggested"]
        if ident.artist and ident.title:
            summary_bits.append(f"{ident.artist} — {ident.title}")
        elif ident.title:
            summary_bits.append(ident.title)
        if outcome.notes:
            summary_bits.append(outcome.notes[-1])
        self.session.status(" | ".join(summary_bits) + "  (Editable; Ctrl+Z to undo)")

    def _update_smart_banner(self, ident) -> None:
        from ..metadata.identify import KIND_LABELS

        kind_label = KIND_LABELS.get(ident.recording_kind, "Recording")
        if ident.artist and ident.title:
            header = f"{kind_label}: {ident.artist} — {ident.title}"
        elif ident.title:
            header = f"{kind_label}: {ident.title}"
        else:
            header = kind_label
        self.smart_banner_label.setText(header)

        self.release_combo.blockSignals(True)
        self.release_combo.clear()
        for cand in ident.candidates:
            self.release_combo.addItem(cand.label, cand)
        has_choices = len(ident.candidates) > 0
        self.release_combo.setVisible(has_choices)
        if has_choices and 0 <= ident.selected_index < len(ident.candidates):
            self.release_combo.setCurrentIndex(ident.selected_index)
        self.release_combo.blockSignals(False)
        self.smart_banner.setVisible(True)

    def _on_release_candidate_changed(self, index: int) -> None:
        ident = self._last_identification
        if ident is None or index < 0 or index >= len(ident.candidates):
            return
        if index == ident.selected_index:
            return
        from ..metadata.identify import load_candidate_worklist

        cand = ident.candidates[index]
        ident.selected_index = index
        ident.subtitle = cand.detail
        self.session.status(f"Loading release '{cand.label}'...")
        try:
            wl = load_candidate_worklist(cand)
        except Exception as exc:
            QMessageBox.warning(self, "Release lookup", str(exc))
            return
        if wl is not None:
            ident.worklist = wl
            self.run_smart_split(identification=ident)

    def _revert_to_audio_only(self) -> None:
        """Revert metadata-assisted split to plain audio-only boundary detection."""
        self.smart_banner.setVisible(False)
        self.run_analysis()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def run_analysis(self, sensitivity: str = "balanced") -> None:
        if not self.session.project.sources:
            self.session.status("Import audio before analysing.")
            return
        job = AnalysisJob(
            self.session.project,
            sensitivity=sensitivity,
            placement=self.session.placement_choice(),
        )
        self._run_job(job, self._on_analysis_done)

    def _on_analysis_done(self, outcome) -> None:
        suggestions = outcome.suggestions
        self.session.apply_analysis(suggestions, outcome.classifications)
        self.waveform.zoom_to_fit()
        uncertain = sum(1 for item in suggestions if item.uncertain)
        message = (
            f"{len(suggestions)} boundary suggestion(s) "
            f"(noise floor {outcome.noise_floor_dbfs:.1f} dBFS). "
        )
        if uncertain:
            message += f"{uncertain} of them need a human decision. "
        message += "Review them - they are proposals, not decisions."
        self.session.status(message)

    # ------------------------------------------------------------------
    # Theatrical segmentation
    # ------------------------------------------------------------------
    def run_theater_analysis(
        self, transcribe: bool = False, reanalyze: bool = False
    ) -> None:
        """Segment the recording into theatrical sections."""
        if not self.session.project.sources:
            self.session.status("Import audio before analysing.")
            return
        job = TheaterJob(
            self.session.project,
            use_transcriber=transcribe,
            reanalyze=reanalyze,
            separate_applause=self.separate_applause_action.isChecked(),
        )
        self._run_job(job, self._on_theater_done)

    def _on_theater_done(self, outcome) -> None:
        self.session.apply_theater_analysis(outcome)
        self.waveform.zoom_to_fit()
        counts: dict[str, int] = {}
        for section in outcome.sections:
            counts[section.section_type] = counts.get(section.section_type, 0) + 1
        summary = ", ".join(
            f"{count} {name.replace('_', ' ')}"
            for name, count in sorted(counts.items(), key=lambda pair: -pair[1])
        )
        message = (
            f"{len(outcome.sections)} section(s): {summary}. "
        )
        if outcome.used_model:
            message += "The audio-event classifier was used. "
        else:
            message += (
                "The audio-event classifier was not available, so the sections "
                "come from acoustic measurements alone. "
            )
        if outcome.transcribed:
            message += (
                f"The transcriber kept {outcome.transcript_kept} stretch(es) and "
                f"rejected {outcome.transcript_dropped}. "
            )
        message += "Review them - they are proposals, not decisions."
        self.session.status(message)

    def propose_theater_names(self) -> None:
        """Name the detected numbers from the active reference list."""
        try:
            result = self.session.propose_theater_names()
        except SessionError as exc:
            QMessageBox.information(self, "Reference list", str(exc))
            return
        self.session.status(
            f"{result['matched']} number(s) named from the reference list; "
            f"{result['unmatched_items']} list entry/entries and "
            f"{result['unmatched_segments']} detected number(s) were left unmatched "
            "(nothing was forced). The names are evidence, not confirmed identifications."
        )

    def match_reference_lyrics(self) -> None:
        from .lyrics import LyricInputDialog
        from ..app.jobs import LyricMatchJob
        dialog = LyricInputDialog(self.session.project.cache.get('lyric_references', []), self)
        if dialog.exec() != dialog.Accepted:
            return
        references = dialog.references
        project_id = self.session.project.project_id
        def done(report):
            from ..analysis.lyrics import evidence_signature
            if (self.session.project.project_id != project_id or
                    report.get('evidence_signature') != evidence_signature(self.session.project)):
                self.session.status('Project changed; lyric proposals discarded.')
                return
            def apply(project):
                project.cache['lyric_references'] = references
                project.cache['lyric_review'] = report
                project.touch()
            self.session._mutate('Match reference lyrics (review only)', apply)
            self.review_lyric_matches()
        self._run_job(LyricMatchJob(self.session.project, references), done)

    def review_lyric_matches(self) -> None:
        from .lyrics import LyricReviewDialog
        report = self.session.project.cache.get('lyric_review')
        if not report:
            self.session.status('Use Theatre > Match reference lyrics first.')
            return
        from ..analysis.lyrics import evidence_signature
        if report.get('evidence_signature') != evidence_signature(self.session.project):
            self.session.status('Audio sources or transcript changed. Match reference lyrics again before reviewing these proposals.')
            return
        LyricReviewDialog(report, self).exec()

    def apply_lyric_grouping(self, proposals: list[dict], *, allow_uncertain: bool = False) -> dict:
        result = self.session.apply_lyric_grouping(proposals, allow_uncertain=allow_uncertain)
        applied = result.get("applied", 0)
        skipped = result.get("skipped_protected", 0)
        skipped_unc = result.get("skipped_uncertain", 0)
        if applied:
            msg = f"Applied {applied} proposed musical-number grouping(s). Ctrl+Z to undo."
            notes = []
            if skipped:
                notes.append(f"{skipped} skipped to protect locked/edited tracks")
            if skipped_unc:
                notes.append(f"{skipped_unc} uncertain skipped")
            if notes:
                msg += " (" + "; ".join(notes) + ")"
            self.session.status(msg)
        else:
            reasons = []
            if skipped:
                reasons.append("protected by locked/edited tracks")
            if skipped_unc:
                reasons.append("uncertain boundaries not auto-applied")
            suffix = f" ({'; '.join(reasons)})" if reasons else ""
            self.session.status(f"No proposed groupings applied{suffix}.")
        return result

    def load_reference_audio_manifest(self) -> None:
        from ..analysis.reference_audio import load_manifest
        path, _ = QFileDialog.getOpenFileName(self, "Local reference audio manifest", "",
                                              "JSON manifest (*.json)")
        if not path:
            return
        try:
            entries = load_manifest(path)
        except (ValueError, OSError, KeyError) as exc:
            QMessageBox.warning(self, "Reference audio", str(exc))
            return
        def apply(project):
            project.cache["reference_audio_inputs"] = entries
            project.touch()
        self.session._mutate("Load local reference audio", apply)
        self.session.status(f"Loaded {len(entries)} local audio references. Run theatre segmentation to compare. Titles remain unconfirmed.")

    def load_reference_list(self) -> None:
        """Load one of the bundled reference lists."""
        dialog = ReferenceDialog(self)
        if dialog.exec() != ReferenceDialog.Accepted:
            return
        worklist = dialog.worklist()
        if worklist is None:
            return
        self.session.add_worklist(worklist)
        self.session.status(
            f"Loaded '{worklist.name}' with {len(worklist.items)} entry/entries. "
            "Edit it, then use 'Propose names from the reference list'."
        )

    def accept_suggestions(self) -> None:
        answer = QMessageBox.question(
            self,
            "Accept suggestions",
            "Replace the current boundaries with the last analysis result?\n\n"
            "Locked tracks keep their boundaries and metadata.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.run_analysis()

    def refine_boundaries(self) -> None:
        """Look at the existing cuts again and propose better ones."""
        if not self.session.project.tracks:
            self.session.status("There are no boundaries to refine yet.")
            return
        job = RefineJob(self.session.project, placement=self.session.placement_choice())
        self._run_job(job, self._on_refinement_done)

    def _on_refinement_done(self, proposals) -> None:
        if not proposals:
            self.session.status(
                "No boundary would move. The existing cuts are already where the "
                "detector would put them."
            )
            return
        dialog = RefinePreviewDialog(proposals, self.session.project.sample_rate() or 44100, self)
        if dialog.exec() != RefinePreviewDialog.Accepted:
            self.session.status("No boundary was changed.")
            return
        chosen = dialog.selected_proposals()
        applied = self.session.apply_refinement(chosen)
        self.session.status(f"Moved {applied} boundary/boundaries. One undo puts them back.")

    def paste_tracklist(self) -> None:
        dialog = TracklistDialog(self)
        if dialog.exec() != TracklistDialog.Accepted:
            return
        name, mode, items = dialog.values()
        if not items:
            QMessageBox.information(self, "Tracklist", "No track entries were recognised.")
            return
        self.session.paste_tracklist(
            "\n".join(item.title for item in items), name
        )
        self.session.status(
            f"{len(items)} entries added. Use 'Align with the tracklist' to match them "
            "to the detected segments."
        )

    def align_with_tracklist(self) -> None:
        try:
            result = self.session.align_with_worklist()
        except SessionError as exc:
            QMessageBox.information(self, "Align", str(exc))
            return
        self.session.status(
            f"Matched {result['matched']} entries; "
            f"{result['unmatched_items']} tracklist entries and "
            f"{result['unmatched_segments']} detected segments were left unmatched "
            "(nothing was forced)."
        )

    # ------------------------------------------------------------------
    # Metadata providers
    # ------------------------------------------------------------------
    def lookup_album(self) -> None:
        dialog = ProviderLookupDialog("album", self)
        if dialog.exec() != ProviderLookupDialog.Accepted:
            return
        worklist = dialog.worklist()
        if worklist is None:
            return
        self.session.add_worklist(worklist)
        self.session.status(
            f"Added {len(worklist.items)} track(s) from {worklist.provider}. "
            "Use 'Align with the tracklist' to match them to the audio."
        )

    def lookup_concert(self) -> None:
        dialog = ProviderLookupDialog("concert", self)
        if dialog.exec() != ProviderLookupDialog.Accepted:
            return
        worklist = dialog.worklist()
        if worklist is None:
            return
        self.session.add_worklist(worklist)
        self.session.status(
            f"Added {len(worklist.items)} track(s) from {worklist.provider}. "
            "Use 'Align with the tracklist' to match them to the audio."
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_project(self) -> None:
        if not self.session.project.tracks:
            QMessageBox.information(self, "Export", "There is nothing to export yet.")
            return
        dialog = ExportDialog(self.session.project.export_settings, self)
        if dialog.exec() != ExportDialog.Accepted:
            return
        settings = dialog.values()
        if not settings.output_dir:
            QMessageBox.warning(self, "Export", "Choose an output folder first.")
            return
        self.session.project.export_settings = settings
        track_ids = None
        if dialog.selected_only_check.isChecked():
            selected = self.table.selected_track()
            if selected is None:
                QMessageBox.information(
                    self, "Export", "Select a track in the table first, or untick "
                    "'Export only the track selected in the table'."
                )
                return
            track_ids = [selected.track_id]
        job = ExportJob(
            self.session.project,
            settings,
            track_ids=track_ids,
            section_policy=dialog.section_policy(),
        )
        self._run_job(job, self._on_export_done)

    def _on_export_done(self, report) -> None:
        message = report.summary_text()
        self.session.status(f"Export finished in {report.elapsed_seconds:.1f} s")
        box = QMessageBox(self)
        box.setWindowTitle("Export complete")
        box.setText("The export finished.")
        box.setInformativeText(message)
        box.setDetailedText(message)
        box.exec()

    def measure_for_dialog(self, dialog: ExportDialog) -> None:
        """Measure the programme and fill the export dialog with the numbers."""
        if not self.session.project.tracks:
            dialog.set_measurement("There is nothing to measure yet.")
            return
        dialog.set_measurement("Measuring...")

        class _Measure(Job):
            def run(self):
                from ..audio.measure import Part, measure_parts
                from ..app.exporting import PartBuilder

                builder = PartBuilder(self.project)
                try:
                    parts = []
                    for track in self.project.sorted_tracks():
                        for source_like, start, end in builder.parts_for(
                            track.start_frame, track.end_frame
                        ):
                            parts.append(Part(source_like, start, end))
                    measured = measure_parts(parts)
                finally:
                    builder.close()
                return measured

        job = _Measure("Measuring")
        self._run_job(job, lambda measured: self._on_measured(dialog, measured))

    def _on_measured(self, dialog: ExportDialog, measured) -> None:
        from ..audio.gain import plan_shared_gain

        target = dialog.target_spin.value()
        ceiling = dialog.ceiling_spin.value()
        plan = plan_shared_gain(measured, target, ceiling)
        if plan.feasible:
            text = (
                f"<b>Measured:</b> {measured.integrated_lufs:.1f} LUFS integrated, "
                f"peak {measured.true_peak_dbtp:.1f} dBTP, "
                f"sample peak {measured.sample_peak_dbfs:.1f} dBFS<br>"
                f"<b>Proposed gain:</b> {plan.gain_db:+.1f} dB "
                f"(resulting peak {plan.peak_after_dbtp:.1f} dBTP, "
                f"ceiling {ceiling:.1f} dBTP)"
            )
        else:
            text = (
                f"<b>Measured:</b> {measured.integrated_lufs:.1f} LUFS integrated, "
                f"peak {measured.true_peak_dbtp:.1f} dBTP<br>"
                f"<b>{plan.reason}</b><br>"
                f"<i>Nothing is applied automatically: use less gain, or explicitly "
                "allow a limiter.</i>"
            )
        dialog.set_measurement(text)

    # ------------------------------------------------------------------
    # Markers
    # ------------------------------------------------------------------
    def export_markers(self) -> None:
        if not self.session.project.tracks:
            QMessageBox.information(self, "Markers", "There are no segments to export.")
            return
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Export markers",
            "markers.splytnyt-markers.json",
            "Marker JSON (*.json);;CSV (*.csv);;Audacity labels (*.txt);;All files (*)",
        )
        if not path:
            return
        project = self.session.project
        try:
            if path.lower().endswith(".csv"):
                text = serialize.markers_to_csv(project)
            elif path.lower().endswith(".txt"):
                text = serialize.markers_to_audacity(project)
            else:
                text = serialize.markers_to_json(project)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        except Exception as exc:
            QMessageBox.critical(self, "Export markers", str(exc))
            return
        self.session.status(f"Markers exported to {os.path.basename(path)}")

    def import_markers(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import markers",
            "",
            "Marker files (*.json *.csv *.txt);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            if path.lower().endswith(".csv"):
                segments = serialize.markers_from_csv(text, self.session.project.sample_rate() or 44100)
            elif path.lower().endswith(".txt"):
                segments = serialize.markers_from_audacity(text, self.session.project.sample_rate() or 44100)
            else:
                segments = serialize.markers_from_json(text)
        except Exception as exc:
            QMessageBox.critical(self, "Import markers", str(exc))
            return

        def apply(project):
            from ..core.model import Track

            project.tracks = [
                Track(start_frame=start, end_frame=end, title=title,
                      boundary_basis="manual", boundary_placement="manual")
                for start, end, title in segments
            ]
            project.touch()

        self.session._mutate("Import markers", apply)
        self.session.status(f"Imported {len(segments)} segments from {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _edit_metadata(self) -> None:
        dialog = MetadataDialog(self.session.project.metadata, self)
        if dialog.exec() != MetadataDialog.Accepted:
            return
        for field, value in dialog.values().items():
            self.session.set_project_metadata(field, value)

    def show_shortcuts(self) -> None:
        ShortcutsDialog(self).exec()

    def show_about(self) -> None:
        AboutDialog(self).exec()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def _apply_stored_theme(self) -> None:
        mode = theme.stored_mode()
        self.set_theme(mode, remember=False)

    def set_theme(self, mode: str, remember: bool = True) -> None:
        """Switch the theme immediately and remember the choice."""
        from PySide6.QtWidgets import QApplication

        theme.apply_theme(QApplication.instance(), mode)
        if remember:
            theme.store_mode(mode)
        for action in self._theme_group.actions():
            action.setChecked(action.data() == mode)
        self.waveform.viewport().update()
        self.table.viewport().update()
        self._update_details()

    # ------------------------------------------------------------------
    # Autosave and recovery
    # ------------------------------------------------------------------
    def _app_data_dir(self) -> str:
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not base:
            base = os.path.join(os.path.expanduser("~"), ".splytnyt")
        os.makedirs(base, exist_ok=True)
        return base

    def _recovery_path(self) -> str:
        return os.path.join(self._app_data_dir(), "recovery.splytnyt")

    def _autosave(self) -> None:
        if not self.session.project.sources:
            return
        if not self.session.dirty:
            # Nothing has changed since the last save, so there is nothing
            # to recover. Writing a recovery copy anyway would make the
            # next start ask the user to restore work they had already
            # saved.
            return
        try:
            from copy import deepcopy
            serialize.save_project(deepcopy(self.session.project), self._recovery_path(), include_peaks=False)
            self._last_autosave = time.time()
        except Exception:
            pass

    def _clear_recovery(self) -> None:
        """Forget the recovery copy: the work has been saved properly."""
        try:
            path = self._recovery_path()
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    def _offer_recovery(self) -> None:
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        recovery = self._recovery_path()
        if not os.path.exists(recovery):
            return
        try:
            recovery_time = os.path.getmtime(recovery)
        except OSError:
            return
        project_path = self.session.project.path
        if project_path and os.path.exists(project_path):
            try:
                if os.path.getmtime(project_path) >= recovery_time:
                    return
            except OSError:
                pass
        answer = QMessageBox.question(
            self,
            "Recover unsaved work",
            "An autosaved version of this project is newer than the last saved file.\n\n"
            "Restore the autosaved version?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            try:
                self.session.open_project(recovery)
                self.session.project.path = ""
                self.session.mark_dirty()
                self._reload()
                self.session.status("Restored the autosaved project.")
            except Exception as exc:
                QMessageBox.warning(self, "Recovery", str(exc))

    # ------------------------------------------------------------------
    def open_music_services(self) -> None:
        """Open the simple Music Services settings dialog."""
        dialog = MusicServicesDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        self._music_services_dialog = dialog
        dialog.open()

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        if self._runner is not None and self._runner.isRunning():
            self._runner.job.cancel()
            self._runner.wait(2000)
        if self._playback is not None:
            self._playback.stop()
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
        if not self._confirm_discard():
            event.ignore()
            return
        self._autosave()
        event.accept()
