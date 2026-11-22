"""The track table.

Shows every segment of the timeline with the information a reviewer needs:
title, start, end, duration, section type, what is known about the boundary
and about the identity of the track, and any flags.

The two confidences the product keeps deliberately separate are shown as
words, not as percentages. A number such as "63 %" claims a precision the
detector does not have, and "Not assessed" is more honest than "0 %".

Title and section type are editable in place; edits are routed back through
the session so they become undoable steps.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMenu,
    QTableView,
    QWidget,
)

from ..core.flags import describe_flag
from ..core.model import Track
from ..core.units import frames_to_timestamp
from . import theme

COLUMN_INDEX = 0
COLUMN_TITLE = 1
COLUMN_START = 2
COLUMN_END = 3
COLUMN_DURATION = 4
COLUMN_SECTION = 5
COLUMN_BOUNDARY = 6
COLUMN_IDENTITY = 7
COLUMN_FLAGS = 8

HEADERS = [
    "#", "Title", "Start", "End", "Length", "Section",
    "Boundary", "Identity", "Notes",
]


class TrackTableModel(QAbstractTableModel):
    """Table model over ``project.tracks``."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tracks: list[Track] = []
        self._sample_rate = 44100
        self._on_edit: Optional[Callable[[str, str, object], None]] = None
        self._on_activate: Optional[Callable[[str], None]] = None

    # ------------------------------------------------------------------
    def set_tracks(self, tracks: list[Track], sample_rate: int) -> None:
        self.beginResetModel()
        self._tracks = sorted(tracks, key=lambda t: (t.start_frame, t.end_frame))
        self._sample_rate = sample_rate or 44100
        self.endResetModel()

    def track_at_row(self, row: int) -> Optional[Track]:
        if 0 <= row < len(self._tracks):
            return self._tracks[row]
        return None

    def row_of_track(self, track_id: str) -> int:
        for index, track in enumerate(self._tracks):
            if track.track_id == track_id:
                return index
        return -1

    def set_on_edit(self, callback: Callable[[str, str, object], None]) -> None:
        self._on_edit = callback

    def set_on_activate(self, callback: Callable[[str], None]) -> None:
        self._on_activate = callback

    # ------------------------------------------------------------------
    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._tracks)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return HEADERS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        track = self._tracks[index.row()]
        column = index.column()

        if role == Qt.DisplayRole:
            return self._display(track, column)
        if role == Qt.EditRole and column == COLUMN_TITLE:
            return track.title
        if role == Qt.EditRole and column == COLUMN_SECTION:
            return track.section_type
        if role == Qt.ToolTipRole:
            return self._tooltip(track)
        if role == Qt.ForegroundRole:
            if column == COLUMN_BOUNDARY:
                return QBrush(theme.assessment_color(track.boundary_assessment()))
            if column == COLUMN_IDENTITY:
                return QBrush(theme.assessment_color(self._identity_text(track)))
            if column == COLUMN_TITLE:
                if not track.title.strip():
                    return QBrush(QColor(theme.active().text_faint))
                return QBrush(QColor(theme.active().text))
            if column == COLUMN_SECTION:
                return QBrush(theme.color_for_section(track.section_type))
            if column == COLUMN_FLAGS and track.flags:
                return QBrush(QColor(theme.active().uncertain))
            if column == COLUMN_INDEX and track.locked:
                return QBrush(QColor(theme.active().locked))
        if role == Qt.FontRole:
            if column == COLUMN_INDEX:
                font = QFont()
                font.setPointSize(8)
                return font
            if column == COLUMN_BOUNDARY or column == COLUMN_IDENTITY:
                font = QFont()
                font.setBold(True)
                return font
        return None

    def _display(self, track: Track, column: int) -> str:
        rate = self._sample_rate
        if column == COLUMN_INDEX:
            number = self._tracks.index(track) + 1
            return f"{number}" if not track.locked else f"{number} *"
        if column == COLUMN_TITLE:
            title = track.title or "(untitled)"
            artist = (track.metadata.artist or track.recognized_artist or "").strip()
            if artist and title != "(untitled)" and artist.lower() not in title.lower():
                return f"{artist} - {title}"
            return title
        if column == COLUMN_START:
            return frames_to_timestamp(track.start_frame, rate)
        if column == COLUMN_END:
            return frames_to_timestamp(track.end_frame, rate)
        if column == COLUMN_DURATION:
            seconds = track.frames / rate if rate else 0.0
            if seconds >= 60:
                return f"{int(seconds // 60)}:{seconds % 60:04.1f}"
            return f"{seconds:.2f} s"
        if column == COLUMN_SECTION:
            return track.section_type.replace("_", " ")
        if column == COLUMN_BOUNDARY:
            return track.boundary_assessment()
        if column == COLUMN_IDENTITY:
            return self._identity_text(track)
        if column == COLUMN_FLAGS:
            if not track.flags:
                return ""
            return ", ".join(describe_flag(flag) for flag in track.flags)
        return ""

    @staticmethod
    def _identity_text(track: Track) -> str:
        """What is known about what this track *is*.

        A title that came from a pasted tracklist, MusicBrainz, or a setlist is
        an alignment — NEVER called "Confirmed" without multiple consistent
        acoustic recognition results.
        """
        if track.identity_basis == "manual":
            return "Typed"
        if track.recognition_status:
            return {
                "confirmed": "Confirmed",
                "likely": "Likely",
                "uncertain": "Uncertain",
                "conflict": "Conflict",
                "unidentified": "Unidentified",
            }.get(track.recognition_status, "Uncertain")
        if track.identity_basis == "assessed":
            return "Aligned"
        return "Not assessed"

    def _tooltip(self, track: Track) -> str:
        lines = [f"<b>{track.title or 'Untitled'}</b>"]
        artist = (track.metadata.artist or track.recognized_artist or "").strip()
        if artist:
            lines.append(f"Artist: {artist}")
        lines.append(f"Section: {track.section_type}")
        lines.append(f"Boundary: {track.boundary_assessment()}")
        lines.append(f"Identity: {self._identity_text(track)}")
        if track.recognition_engines:
            lines.append(f"Engines: {', '.join(track.recognition_engines)}")
        if track.boundary_basis == "assessed":
            lines.append(
                f"Boundary confidence: {track.boundary_confidence:.2f} "
                "(a guide, not a measurement)"
            )
        if track.identity_basis == "assessed" and not track.recognition_status:
            lines.append(
                "Identity comes from matching a tracklist/setlist by duration. "
                "The recording was not confirmed acoustically."
            )
        if track.reasons:
            lines.append("<br/>Evidence:")
            lines.extend(f"- {reason}" for reason in track.reasons)
        if track.locked:
            lines.append("<br/><i>Locked: re-analysis will not change this track.</i>")
        return "<br/>".join(lines)

    # ------------------------------------------------------------------
    def flags(self, index: QModelIndex) -> int:
        if not index.isValid():
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() in (COLUMN_TITLE, COLUMN_SECTION):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:
        if role != Qt.EditRole or not index.isValid():
            return False
        track = self._tracks[index.row()]
        column = index.column()
        if column == COLUMN_TITLE:
            if self._on_edit:
                self._on_edit(track.track_id, "title", str(value))
            return True
        if column == COLUMN_SECTION:
            if self._on_edit:
                self._on_edit(track.track_id, "section_type", str(value))
            return True
        return False


class TrackTable(QTableView):
    """The review table."""

    trackActivated = Signal(str)
    trackSelected = Signal(str)
    moveUpRequested = Signal(str)
    moveDownRequested = Signal(str)
    mergePreviousRequested = Signal(str)
    renameRequested = Signal(str)
    toggleLockRequested = Signal(str)
    playRequested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = TrackTableModel(self)
        self.setModel(self._model)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(False)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(26)
        self.setWordWrap(False)
        self.setShowGrid(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setToolTip('One click or arrow keys selects a track without seeking. Double-click a non-editable cell to seek and stop; double-click Title or Section to edit. F2 also edits.')
        self.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )

        header = self.horizontalHeader()
        header.setSectionResizeMode(COLUMN_INDEX, QHeaderView.Fixed)
        header.setSectionResizeMode(COLUMN_TITLE, QHeaderView.Stretch)
        header.setSectionResizeMode(COLUMN_START, QHeaderView.Fixed)
        header.setSectionResizeMode(COLUMN_END, QHeaderView.Fixed)
        header.setSectionResizeMode(COLUMN_DURATION, QHeaderView.Fixed)
        header.setSectionResizeMode(COLUMN_SECTION, QHeaderView.Fixed)
        header.setSectionResizeMode(COLUMN_BOUNDARY, QHeaderView.Fixed)
        header.setSectionResizeMode(COLUMN_IDENTITY, QHeaderView.Fixed)
        header.setSectionResizeMode(COLUMN_FLAGS, QHeaderView.Interactive)
        header.setHighlightSections(False)
        self.setColumnWidth(COLUMN_INDEX, 40)
        self.setColumnWidth(COLUMN_START, 84)
        self.setColumnWidth(COLUMN_END, 84)
        self.setColumnWidth(COLUMN_DURATION, 76)
        self.setColumnWidth(COLUMN_SECTION, 84)
        self.setColumnWidth(COLUMN_BOUNDARY, 96)
        self.setColumnWidth(COLUMN_IDENTITY, 96)
        self.setColumnWidth(COLUMN_FLAGS, 190)

        self.doubleClicked.connect(self._on_double_click)

    # ------------------------------------------------------------------
    def set_tracks(self, tracks: list[Track], sample_rate: int) -> None:
        self._model.set_tracks(tracks, sample_rate)

    def set_on_edit(self, callback) -> None:
        self._model.set_on_edit(callback)

    def set_on_activate(self, callback) -> None:
        self._model.set_on_activate(callback)

    def track_at_row(self, row: int) -> Optional[Track]:
        return self._model.track_at_row(row)

    def selected_track(self) -> Optional[Track]:
        indexes = self.selectionModel().selectedRows()
        if not indexes:
            return None
        return self._model.track_at_row(indexes[0].row())

    def select_track(self, track_id: str) -> None:
        row = self._model.row_of_track(track_id)
        if row < 0:
            return
        index = self._model.index(row, COLUMN_TITLE)
        self.selectionModel().setCurrentIndex(
            index,
            self.selectionModel().SelectionFlag.ClearAndSelect
            | self.selectionModel().SelectionFlag.Rows,
        )
        self.scrollTo(index)

    def _on_double_click(self, index: QModelIndex) -> None:
        # Editable cells open their editor. Other cells seek; never autoplay.
        if index.column() in (COLUMN_TITLE, COLUMN_SECTION):
            return
        track = self._model.track_at_row(index.row())
        if track is not None:
            self.trackActivated.emit(track.track_id)

    def contextMenuEvent(self, event) -> None:
        index = self.indexAt(event.pos())
        if not index.isValid():
            return
        track = self._model.track_at_row(index.row())
        if track is None:
            return
        self.select_track(track.track_id)
        menu = QMenu(self)
        play_act = menu.addAction("Play this track")
        play_act.triggered.connect(lambda: self.playRequested.emit(track.track_id))
        rename_act = menu.addAction("Rename track…")
        rename_act.triggered.connect(lambda: self.renameRequested.emit(track.track_id))
        menu.addSeparator()
        if index.row() > 0:
            up_act = menu.addAction("Move label up (keeps audio order)")
            up_act.triggered.connect(lambda: self.moveUpRequested.emit(track.track_id))
            merge_act = menu.addAction("Remove split (merge with previous)")
            merge_act.triggered.connect(lambda: self.mergePreviousRequested.emit(track.track_id))
        if index.row() < self._model.rowCount() - 1:
            down_act = menu.addAction("Move label down (keeps audio order)")
            down_act.triggered.connect(lambda: self.moveDownRequested.emit(track.track_id))
        menu.addSeparator()
        lock_act = menu.addAction("Unlock track" if track.locked else "Lock track")
        lock_act.triggered.connect(lambda: self.toggleLockRequested.emit(track.track_id))
        menu.exec(event.globalPos())
