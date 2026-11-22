"""Looking up a tracklist online.

The dialog has three steps: search, choose, edit. Nothing is applied to the
project until the user has confirmed the exact release or performance and
edited the tracklist to suit the recording - a release may have a different
running order, a concert may have an extra song or miss one, and a medley
may have to be entered by hand.

The lookup runs on a worker thread, so the interface never freezes while a
provider is being asked.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.model import Worklist, WorklistItem, now_iso
from ..metadata import (
    MUSICBRAINZ,
    SETLISTFM,
    PerformanceCandidate,
    ProviderError,
    ReleaseCandidate,
    TrackCandidate,
)

ALBUM = "album"
CONCERT = "concert"


class _Worker(QThread):
    """Runs a provider call off the interface thread."""

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, function, parent=None) -> None:
        super().__init__(parent)
        self._function = function

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._function())
        except ProviderError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(f"The lookup failed: {exc}")


class ProviderLookupDialog(QDialog):
    """Search for an album or a concert, then edit the tracklist."""

    def __init__(self, mode: str = ALBUM, parent=None) -> None:
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle(
            "Look up an album on MusicBrainz" if mode == ALBUM
            else "Look up a concert on setlist.fm"
        )
        self.resize(760, 620)

        self._groups: list[ReleaseCandidate] = []
        self._releases: list[ReleaseCandidate] = []
        self._performances: list[PerformanceCandidate] = []
        self._selected_release: Optional[ReleaseCandidate] = None
        self._selected_performance: Optional[PerformanceCandidate] = None
        self._tracks: list[TrackCandidate] = []
        self._worker: Optional[_Worker] = None
        self._pending_step = "matches"

        self._build_ui()
        self._go_to_search()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.stack = QStackedWidget(self)
        layout.addWidget(self.stack, 1)

        self.stack.addWidget(self._build_search_page())
        self.stack.addWidget(self._build_matches_page())
        self.stack.addWidget(self._build_editions_page())
        self.stack.addWidget(self._build_tracklist_page())

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self._go_back)
        buttons.addWidget(self.back_button)
        buttons.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        self.use_button = QPushButton("Use this tracklist")
        self.use_button.setProperty("primary", True)
        self.use_button.setToolTip(
            "Add this tracklist to the project. It is only evidence: it never "
            "decides where a cut goes."
        )
        self.use_button.clicked.connect(self.accept)
        buttons.addWidget(self.use_button)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------
    def _build_search_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        heading = QLabel(
            "Find the release or the performance you actually hold."
            if self.mode == ALBUM
            else "Find the concert you recorded."
        )
        heading.setProperty("heading", True)
        layout.addWidget(heading)

        help_text = QLabel(
            "SplytNyt asks the provider for a list of songs and their lengths. "
            "The lengths are only evidence for lining the list up against the "
            "audio - they never decide where a cut goes."
            if self.mode == ALBUM
            else "SplytNyt asks the provider for the songs of the concert, "
            "with the sets and the encores as they were recorded. The list is "
            "only evidence for lining it up against the audio."
        )
        help_text.setWordWrap(True)
        help_text.setProperty("hint", True)
        layout.addWidget(help_text)

        form = QFormLayout()
        if self.mode == ALBUM:
            self.artist_edit = QLineEdit()
            self.artist_edit.setPlaceholderText("Artist")
            self.title_edit = QLineEdit()
            self.title_edit.setPlaceholderText("Album title (optional)")
            form.addRow("Artist", self.artist_edit)
            form.addRow("Album", self.title_edit)
        else:
            self.artist_edit = QLineEdit()
            self.artist_edit.setPlaceholderText("Artist")
            self.date_edit = QLineEdit()
            self.date_edit.setPlaceholderText("YYYY-MM-DD")
            self.venue_edit = QLineEdit()
            self.venue_edit.setPlaceholderText("Venue")
            form.addRow("Artist", self.artist_edit)
            form.addRow("Date", self.date_edit)
            form.addRow("Venue", self.venue_edit)
        layout.addLayout(form)

        row = QHBoxLayout()
        self.search_button = QPushButton("Search")
        self.search_button.setProperty("primary", True)
        self.search_button.clicked.connect(self._start_search)
        row.addWidget(self.search_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.offline_note = QLabel(
            "A pasted tracklist works without any network at all: "
            "Analyse > Paste tracklist."
        )
        self.offline_note.setWordWrap(True)
        self.offline_note.setProperty("hint", True)
        layout.addWidget(self.offline_note)
        layout.addStretch(1)
        return page

    def _build_matches_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.matches_heading = QLabel("Matches")
        self.matches_heading.setProperty("heading", True)
        layout.addWidget(self.matches_heading)

        self.matches_list = QListWidget()
        self.matches_list.itemDoubleClicked.connect(self._choose_match)
        layout.addWidget(self.matches_list, 1)

        row = QHBoxLayout()
        self.choose_button = QPushButton("Choose this")
        self.choose_button.setProperty("primary", True)
        self.choose_button.clicked.connect(self._choose_match)
        row.addWidget(self.choose_button)
        row.addStretch(1)
        layout.addLayout(row)
        return page

    def _build_editions_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel("Which edition do you have?")
        heading.setProperty("heading", True)
        layout.addWidget(heading)

        note = QLabel(
            "Different editions have different track orders and different "
            "lengths. Pick the one that matches your recording."
        )
        note.setWordWrap(True)
        note.setProperty("hint", True)
        layout.addWidget(note)

        self.editions_list = QListWidget()
        self.editions_list.itemDoubleClicked.connect(self._choose_edition)
        layout.addWidget(self.editions_list, 1)

        row = QHBoxLayout()
        self.edition_button = QPushButton("Choose this edition")
        self.edition_button.setProperty("primary", True)
        self.edition_button.clicked.connect(self._choose_edition)
        row.addWidget(self.edition_button)
        row.addStretch(1)
        layout.addLayout(row)
        return page

    def _build_tracklist_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.tracklist_heading = QLabel("The tracklist")
        self.tracklist_heading.setProperty("heading", True)
        layout.addWidget(self.tracklist_heading)

        self.tracklist_note = QLabel(
            "Edit the list to match the recording: remove a song that is not "
            "there, add one that is, and put the medleys and the encores where "
            "they belong."
        )
        self.tracklist_note.setWordWrap(True)
        self.tracklist_note.setProperty("hint", True)
        layout.addWidget(self.tracklist_note)

        self.track_table = QTableWidget(0, 4)
        self.track_table.setHorizontalHeaderLabels(["#", "Title", "Length", "Notes"])
        self.track_table.verticalHeader().setVisible(False)
        self.track_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.track_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.track_table.setColumnWidth(0, 40)
        self.track_table.setColumnWidth(2, 90)
        layout.addWidget(self.track_table, 1)

        row = QHBoxLayout()
        add = QPushButton("Add a row")
        add.clicked.connect(self._add_row)
        remove = QPushButton("Remove the selected row")
        remove.clicked.connect(self._remove_row)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        layout.addLayout(row)
        return page

    # ------------------------------------------------------------------
    # The steps
    # ------------------------------------------------------------------
    def _go_to_search(self) -> None:
        self.stack.setCurrentIndex(0)
        self.back_button.setEnabled(False)
        self.use_button.setEnabled(False)

    def _go_back(self) -> None:
        index = self.stack.currentIndex()
        if index == 1:
            self._go_to_search()
        elif index == 2:
            self.stack.setCurrentIndex(1)
            self.back_button.setEnabled(True)
        elif index == 3:
            self.stack.setCurrentIndex(1 if self.mode == ALBUM else 0)
            self.back_button.setEnabled(True)

    def _start_search(self) -> None:
        self.status_label.setText("Asking the provider...")
        self.search_button.setEnabled(False)
        try:
            if self.mode == ALBUM:
                from ..metadata.musicbrainz import MusicBrainzProvider

                provider = MusicBrainzProvider()
                artist = self.artist_edit.text()
                title = self.title_edit.text()
                self._start_worker(lambda: provider.search_albums(artist, title))
            else:
                from ..metadata.setlistfm import SetlistFmProvider

                provider = SetlistFmProvider()
                artist = self.artist_edit.text()
                date = self.date_edit.text()
                venue = self.venue_edit.text()
                self._start_worker(
                    lambda: provider.search_concerts(artist=artist, date=date, venue=venue)
                )
        except ProviderError as exc:
            self.status_label.setText(str(exc))
            self.search_button.setEnabled(True)

    def _start_worker(self, function, step: str) -> None:
        self._pending_step = step
        worker = _Worker(function, self)
        worker.finished_ok.connect(self._on_search_done)
        worker.failed.connect(self._on_search_failed)
        self._worker = worker
        worker.start()

    def _on_search_failed(self, message: str) -> None:
        self.search_button.setEnabled(True)
        self.status_label.setText(message)

    def _on_search_done(self, results) -> None:
        self.search_button.setEnabled(True)
        self.status_label.setText("")
        if self._pending_step == "editions":
            self._on_editions_ready(results)
        elif self._pending_step == "tracks":
            self._on_tracks_ready(results)
        else:
            self._show_matches(results)

    def _show_matches(self, results) -> None:
        """Fill the list of matches and move on to it."""
        if self.mode == ALBUM:
            self._groups = list(results)
            self.matches_list.clear()
            for group in self._groups:
                detail = " - ".join(part for part in (group.artist, group.year, group.detail) if part)
                item = QListWidgetItem(f"{group.title}   ({detail})" if detail else group.title)
                item.setData(Qt.UserRole, group.provider_id)
                self.matches_list.addItem(item)
            self.matches_heading.setText("Albums")
            if not self._groups:
                self.status_label.setText(
                    "The provider has no album for that. Try a different spelling "
                    "of the artist or the title."
                )
                return
        else:
            self._performances = list(results)
            self.matches_list.clear()
            for performance in self._performances:
                detail = " - ".join(
                    part
                    for part in (performance.date, performance.venue, performance.city)
                    if part
                )
                item = QListWidgetItem(
                    f"{performance.artist}   ({detail})" if detail else performance.artist
                )
                item.setData(Qt.UserRole, performance.provider_id)
                self.matches_list.addItem(item)
            self.matches_heading.setText("Concerts")
            if not self._performances:
                self.status_label.setText(
                    "The provider has no concert for that. Try a different "
                    "spelling, or a date in the form YYYY-MM-DD."
                )
                return
        self.stack.setCurrentIndex(1)
        self.back_button.setEnabled(True)

    def _choose_match(self) -> None:
        item = self.matches_list.currentItem()
        if item is None:
            self.status_label.setText("Choose one of the matches first.")
            return
        identifier = item.data(Qt.UserRole)
        if self.mode == ALBUM:
            self.status_label.setText("Asking for the editions...")
            from ..metadata.musicbrainz import MusicBrainzProvider

            provider = MusicBrainzProvider()
            self._start_worker(lambda: provider.releases_of(identifier), "editions")
        else:
            self.status_label.setText("Asking for the songs...")
            from ..metadata.setlistfm import SetlistFmProvider

            provider = SetlistFmProvider()
            self._selected_performance = next(
                (
                    performance
                    for performance in self._performances
                    if performance.provider_id == identifier
                ),
                None,
            )
            self._start_worker(lambda: provider.concert_tracks(identifier), "tracks")

    def _on_editions_ready(self, releases) -> None:
        self._releases = list(releases)
        self.editions_list.clear()
        for release in self._releases:
            detail = " - ".join(
                part for part in (release.year, release.format, release.country, release.label) if part
            )
            text = f"{release.title}   ({detail})" if detail else release.title
            entry = QListWidgetItem(text)
            entry.setData(Qt.UserRole, release.provider_id)
            self.editions_list.addItem(entry)
        if not self._releases:
            self.status_label.setText("That album has no editions listed.")
            return
        self.stack.setCurrentIndex(2)
        self.back_button.setEnabled(True)

    def _choose_edition(self) -> None:
        item = self.editions_list.currentItem()
        if item is None:
            self.status_label.setText("Choose one of the editions first.")
            return
        identifier = item.data(Qt.UserRole)
        self._selected_release = next(
            (release for release in self._releases if release.provider_id == identifier), None
        )
        self.status_label.setText("Asking for the track list...")
        from ..metadata.musicbrainz import MusicBrainzProvider

        provider = MusicBrainzProvider()
        self._start_worker(lambda: provider.release_tracks(identifier), "tracks")

    def _on_tracks_ready(self, tracks) -> None:
        self._tracks = list(tracks)
        self.track_table.setRowCount(0)
        for track in self._tracks:
            self._append_row(track.title, track.duration_seconds, track.notes)
        if self.mode == ALBUM and self._selected_release is not None:
            self.tracklist_heading.setText(
                f"The track list of {self._selected_release.title}"
            )
        elif self._selected_performance is not None:
            self.tracklist_heading.setText(
                f"The songs played by {self._selected_performance.artist}"
            )
        self.stack.setCurrentIndex(3)
        self.back_button.setEnabled(True)
        self.use_button.setEnabled(True)

    # ------------------------------------------------------------------
    def _append_row(self, title: str, duration: float, notes: str) -> None:
        row = self.track_table.rowCount()
        self.track_table.insertRow(row)
        number = QTableWidgetItem(str(row + 1))
        number.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.track_table.setItem(row, 0, number)
        self.track_table.setItem(row, 1, QTableWidgetItem(title))
        length = QTableWidgetItem(
            f"{int(duration // 60)}:{int(duration % 60):02d}" if duration > 0 else ""
        )
        self.track_table.setItem(row, 2, length)
        self.track_table.setItem(row, 3, QTableWidgetItem(notes))

    def _add_row(self) -> None:
        self._append_row("", 0.0, "")

    def _remove_row(self) -> None:
        rows = sorted({index.row() for index in self.track_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.track_table.removeRow(row)
        for row in range(self.track_table.rowCount()):
            item = self.track_table.item(row, 0)
            if item is not None:
                item.setText(str(row + 1))

    # ------------------------------------------------------------------
    def worklist(self) -> Optional[Worklist]:
        """The edited tracklist as a worklist for the project."""
        items: list[WorklistItem] = []
        for row in range(self.track_table.rowCount()):
            title_item = self.track_table.item(row, 1)
            title = (title_item.text() if title_item else "").strip()
            if not title:
                continue
            length_item = self.track_table.item(row, 2)
            duration = 0.0
            if length_item is not None and length_item.text().strip():
                try:
                    minutes, _, seconds = length_item.text().strip().partition(":")
                    duration = int(minutes) * 60 + float(seconds)
                except ValueError:
                    duration = 0.0
            notes_item = self.track_table.item(row, 3)
            items.append(
                WorklistItem(
                    title=title,
                    duration_hint=duration,
                    notes=(notes_item.text() if notes_item else "").strip(),
                    position=len(items) + 1,
                )
            )
        if not items:
            return None

        provider = MUSICBRAINZ if self.mode == ALBUM else SETLISTFM
        name = "Album" if self.mode == ALBUM else "Concert"
        provider_id = ""
        if self._selected_release is not None:
            name = self._selected_release.title
            provider_id = self._selected_release.provider_id
        elif self._selected_performance is not None:
            name = f"{self._selected_performance.artist} {self._selected_performance.date}".strip()
            provider_id = self._selected_performance.provider_id

        return Worklist(
            name=name,
            items=items,
            provider=provider,
            provider_id=provider_id,
            fetched_at=now_iso(),
            mode="album" if self.mode == ALBUM else "concert",
            notes="Fetched from "
            + ("MusicBrainz" if self.mode == ALBUM else "setlist.fm")
            + ". The lengths are evidence, not cutting instructions.",
        )
