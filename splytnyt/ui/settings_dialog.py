"""Simple Music Services settings dialog.

Shows connection status for AcoustID, setlist.fm, MusicBrainz, and LRCLIB,
lets the user enter or change API keys (stored securely in Windows Credential
Manager), and provides one-click Test Connection buttons.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ..core.credentials import (
    KEY_ACOUSTID,
    KEY_SETLISTFM,
    backend_name,
    delete_credential,
    get_credential,
    mask_key,
    set_credential,
)
from ..metadata.acoustid import (
    AcoustIdProvider,
    chromaprint_backend,
    has_custom_api_key as has_custom_acoustid_key,
)
from ..metadata.lrclib import LrclibProvider
from ..metadata.musicbrainz import MusicBrainzProvider
from ..metadata.setlistfm import SetlistFmProvider, api_key as setlistfm_api_key
from ..metadata.shazamio_engine import (
    SERVICE_NOTICE as SHAZAMIO_NOTICE,
    ShazamIoProvider,
    install_shazamio,
    installed_version as shazamio_version,
    is_enabled as shazamio_enabled,
    is_installed as shazamio_installed,
    set_enabled as set_shazamio_enabled,
    uninstall_shazamio,
)


class _InstallShazamWorker(QThread):
    progress_text = Signal(str)
    finished_install = Signal(bool, str)

    def __init__(self, uninstall: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._uninstall = uninstall

    def run(self) -> None:
        if self._uninstall:
            ok, msg = uninstall_shazamio()
            self.finished_install.emit(ok, msg)
            return
        ok, msg = install_shazamio(
            progress=lambda _f, text: self.progress_text.emit(text),
        )
        self.finished_install.emit(ok, msg)


class _ConnectionTestWorker(QThread):
    finished_result = Signal(str, bool, str)

    def __init__(self, service: str, fn, parent=None) -> None:
        super().__init__(parent)
        self._service = service
        self._fn = fn

    def run(self) -> None:
        try:
            ok, msg = self._fn()
        except Exception as exc:
            ok, msg = False, str(exc)
        self.finished_result.emit(self._service, ok, msg)


class MusicServicesDialog(QDialog):
    """Simple Settings page for SplytNyt's music metadata services."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Music Services")
        self.resize(650, 600)
        self._workers: list[QThread] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        intro = QLabel(
            "SplytNyt uses online music services to identify recordings, fetch tracklists "
            "and concert setlists, and improve song boundaries. Personal keys are stored "
            f"locally in {backend_name()} — never in project files."
        )
        intro.setWordWrap(True)
        intro.setProperty("hint", True)
        layout.addWidget(intro)

        # 1. AcoustID
        ac_box = QGroupBox("AcoustID (Audio Fingerprinting)")
        ac_form = QFormLayout(ac_box)
        self.acoustid_status = QLabel("")
        self.acoustid_status.setWordWrap(True)
        ac_form.addRow("Status", self.acoustid_status)

        ac_row = QHBoxLayout()
        self.acoustid_edit = QLineEdit()
        self.acoustid_edit.setEchoMode(QLineEdit.Password)
        self.acoustid_edit.setPlaceholderText("Using built-in SplytNyt app key (optional override)")
        ac_row.addWidget(self.acoustid_edit, 1)

        self.acoustid_save_btn = QPushButton("Save")
        self.acoustid_save_btn.clicked.connect(self._save_acoustid_key)
        ac_row.addWidget(self.acoustid_save_btn)

        self.acoustid_reset_btn = QPushButton("Use built-in")
        self.acoustid_reset_btn.clicked.connect(self._reset_acoustid_key)
        ac_row.addWidget(self.acoustid_reset_btn)

        self.acoustid_test_btn = QPushButton("Test Connection")
        self.acoustid_test_btn.clicked.connect(self._test_acoustid)
        ac_row.addWidget(self.acoustid_test_btn)
        ac_form.addRow("App key", ac_row)
        layout.addWidget(ac_box)

        # 2. ShazamIO (Experimental) — optional, opt-in, not bundled in public EXE
        sh_box = QGroupBox("ShazamIO (Experimental — Optional Song Recognition)")
        sh_form = QFormLayout(sh_box)
        sh_notice = QLabel(SHAZAMIO_NOTICE)
        sh_notice.setWordWrap(True)
        sh_notice.setProperty("hint", True)
        sh_form.addRow(sh_notice)

        self.shazamio_status = QLabel("")
        self.shazamio_status.setWordWrap(True)
        sh_form.addRow("Status", self.shazamio_status)

        sh_row = QHBoxLayout()
        self.shazamio_enable_check = QCheckBox("Enable ShazamIO (Experimental)")
        self.shazamio_enable_check.toggled.connect(self._toggle_shazamio)
        sh_row.addWidget(self.shazamio_enable_check)
        sh_row.addStretch(1)

        self.shazamio_install_btn = QPushButton("Install Optional Engine")
        self.shazamio_install_btn.clicked.connect(self._install_or_uninstall_shazamio)
        sh_row.addWidget(self.shazamio_install_btn)

        self.shazamio_test_btn = QPushButton("Test Connection")
        self.shazamio_test_btn.clicked.connect(self._test_shazamio)
        sh_row.addWidget(self.shazamio_test_btn)
        sh_form.addRow("Engine", sh_row)
        layout.addWidget(sh_box)

        # 3. setlist.fm
        sfm_box = QGroupBox("setlist.fm (Live Concert Setlists)")
        sfm_form = QFormLayout(sfm_box)
        self.setlistfm_status = QLabel("")
        self.setlistfm_status.setWordWrap(True)
        sfm_form.addRow("Status", self.setlistfm_status)

        sfm_row = QHBoxLayout()
        self.setlistfm_edit = QLineEdit()
        self.setlistfm_edit.setEchoMode(QLineEdit.Password)
        self.setlistfm_edit.setPlaceholderText("Optional: paste your setlist.fm API key")
        sfm_row.addWidget(self.setlistfm_edit, 1)

        self.setlistfm_save_btn = QPushButton("Save")
        self.setlistfm_save_btn.clicked.connect(self._save_setlistfm_key)
        sfm_row.addWidget(self.setlistfm_save_btn)

        self.setlistfm_clear_btn = QPushButton("Clear")
        self.setlistfm_clear_btn.clicked.connect(self._clear_setlistfm_key)
        sfm_row.addWidget(self.setlistfm_clear_btn)

        self.setlistfm_test_btn = QPushButton("Test Connection")
        self.setlistfm_test_btn.clicked.connect(self._test_setlistfm)
        sfm_row.addWidget(self.setlistfm_test_btn)
        sfm_form.addRow("API key", sfm_row)
        layout.addWidget(sfm_box)

        # 3. Free Services (MusicBrainz & LRCLIB)
        free_box = QGroupBox("Free Services (No API Key Needed)")
        free_form = QFormLayout(free_box)

        mb_row = QHBoxLayout()
        self.mb_status = QLabel("Ready — free, automatic")
        self.mb_status.setWordWrap(True)
        mb_row.addWidget(self.mb_status, 1)
        self.mb_test_btn = QPushButton("Test Connection")
        self.mb_test_btn.clicked.connect(self._test_musicbrainz)
        mb_row.addWidget(self.mb_test_btn)
        free_form.addRow("MusicBrainz", mb_row)

        lrc_row = QHBoxLayout()
        self.lrclib_status = QLabel("Ready — free, automatic")
        self.lrclib_status.setWordWrap(True)
        lrc_row.addWidget(self.lrclib_status, 1)
        self.lrclib_test_btn = QPushButton("Test Connection")
        self.lrclib_test_btn.clicked.connect(self._test_lrclib)
        lrc_row.addWidget(self.lrclib_test_btn)
        free_form.addRow("LRCLIB", lrc_row)
        layout.addWidget(free_box)

        # Footer
        footer = QHBoxLayout()
        self.test_all_btn = QPushButton("Test All Connections")
        self.test_all_btn.clicked.connect(self._test_all)
        footer.addWidget(self.test_all_btn)
        footer.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setProperty("primary", True)
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        layout.addLayout(footer)

        self.refresh_status()

    # ------------------------------------------------------------------
    def refresh_status(self) -> None:
        backend = chromaprint_backend()
        fp_info = f"Chromaprint via {backend}" if backend else "ffmpeg/fpcalc not found"
        if has_custom_acoustid_key():
            stored = get_credential(KEY_ACOUSTID)
            masked = mask_key(stored) if stored else "environment variable"
            self.acoustid_status.setText(f"Custom key ({masked}) • {fp_info}")
            self.acoustid_edit.setPlaceholderText(f"Saved override ({masked}) — type to replace")
        else:
            self.acoustid_status.setText(f"Built-in SplytNyt application key (ready) • {fp_info}")
            self.acoustid_edit.setPlaceholderText("Using built-in SplytNyt app key (optional override)")

        sfm_key = setlistfm_api_key()
        if sfm_key:
            masked = mask_key(sfm_key)
            self.setlistfm_status.setText(f"Configured ({masked} in {backend_name()})")
            self.setlistfm_edit.setPlaceholderText(f"Saved ({masked}) — type to replace")
        else:
            self.setlistfm_status.setText("Not configured (optional — add key for live concert setlists)")
            self.setlistfm_edit.setPlaceholderText("Optional: paste your setlist.fm API key")

        installed = shazamio_installed()
        enabled = shazamio_enabled()
        self.shazamio_enable_check.blockSignals(True)
        self.shazamio_enable_check.setChecked(enabled and installed)
        self.shazamio_enable_check.setEnabled(installed)
        self.shazamio_enable_check.blockSignals(False)
        if installed:
            ver = shazamio_version()
            state_lbl = "Enabled (experimental)" if enabled else "Installed — disabled"
            self.shazamio_status.setText(f"{state_lbl} • ShazamIO {ver} (unofficial online service)")
            self.shazamio_install_btn.setText("Uninstall")
            self.shazamio_test_btn.setEnabled(True)
        else:
            self.shazamio_status.setText(
                "Not installed (optional — click 'Install Optional Engine' to download from PyPI; no API key required)"
            )
            self.shazamio_install_btn.setText("Install Optional Engine")
            self.shazamio_test_btn.setEnabled(False)

    # ------------------------------------------------------------------
    def _save_acoustid_key(self) -> None:
        val = self.acoustid_edit.text().strip()
        if not val:
            return
        if set_credential(KEY_ACOUSTID, val):
            self.acoustid_edit.clear()
            self.refresh_status()
            self.acoustid_status.setText("Saved custom AcoustID key.")
        else:
            self.acoustid_status.setText("Could not save key to credential store.")

    def _reset_acoustid_key(self) -> None:
        delete_credential(KEY_ACOUSTID)
        self.acoustid_edit.clear()
        self.refresh_status()

    def _save_setlistfm_key(self) -> None:
        val = self.setlistfm_edit.text().strip()
        if not val:
            return
        if set_credential(KEY_SETLISTFM, val):
            self.setlistfm_edit.clear()
            self.refresh_status()
        else:
            self.setlistfm_status.setText("Could not save key to credential store.")

    def _clear_setlistfm_key(self) -> None:
        delete_credential(KEY_SETLISTFM)
        self.setlistfm_edit.clear()
        self.refresh_status()

    # ------------------------------------------------------------------
    # ShazamIO (Experimental) optional install & opt-in toggle
    # ------------------------------------------------------------------
    def _toggle_shazamio(self, checked: bool) -> None:
        if checked and not shazamio_installed():
            self.shazamio_enable_check.blockSignals(True)
            self.shazamio_enable_check.setChecked(False)
            self.shazamio_enable_check.blockSignals(False)
            self.shazamio_status.setText("Install the optional ShazamIO engine first before enabling it.")
            return
        set_shazamio_enabled(bool(checked))
        self.refresh_status()

    def _install_or_uninstall_shazamio(self) -> None:
        uninstall = shazamio_installed()
        self.shazamio_install_btn.setEnabled(False)
        self.shazamio_enable_check.setEnabled(False)
        self.shazamio_test_btn.setEnabled(False)
        self.shazamio_status.setText("Removing optional ShazamIO engine..." if uninstall else "Installing ShazamIO from PyPI...")
        worker = _InstallShazamWorker(uninstall=uninstall, parent=self)
        worker.progress_text.connect(self.shazamio_status.setText)
        worker.finished_install.connect(self._on_shazamio_install_done)
        self._workers.append(worker)
        worker.start()

    def _on_shazamio_install_done(self, ok: bool, message: str) -> None:
        self.shazamio_install_btn.setEnabled(True)
        self.refresh_status()
        self.shazamio_status.setText(message)

    # ------------------------------------------------------------------
    def _run_test(self, service: str, label: QLabel, btn: QPushButton, fn) -> None:
        label.setText("Testing connection...")
        btn.setEnabled(False)
        worker = _ConnectionTestWorker(service, fn, self)
        worker.finished_result.connect(self._on_test_done)
        self._workers.append(worker)
        worker.start()

    def _on_test_done(self, service: str, ok: bool, message: str) -> None:
        prefix = "Connected: " if ok else "Failed: "
        text = message if message.lower().startswith(("connected", "failed")) else f"{prefix}{message}"
        mapping = {
            "acoustid": (self.acoustid_status, self.acoustid_test_btn),
            "shazamio": (self.shazamio_status, self.shazamio_test_btn),
            "setlistfm": (self.setlistfm_status, self.setlistfm_test_btn),
            "musicbrainz": (self.mb_status, self.mb_test_btn),
            "lrclib": (self.lrclib_status, self.lrclib_test_btn),
        }
        if service in mapping:
            lbl, btn = mapping[service]
            lbl.setText(text)
            btn.setEnabled(True)

    def _test_acoustid(self) -> None:
        override = self.acoustid_edit.text().strip() or None
        self._run_test(
            "acoustid",
            self.acoustid_status,
            self.acoustid_test_btn,
            lambda: AcoustIdProvider(cache_dir=None, client_key=override).test_connection(),
        )

    def _test_setlistfm(self) -> None:
        override = self.setlistfm_edit.text().strip() or None
        self._run_test(
            "setlistfm",
            self.setlistfm_status,
            self.setlistfm_test_btn,
            lambda: SetlistFmProvider(cache_dir=None, api_key_override=override).test_connection(),
        )

    def _test_musicbrainz(self) -> None:
        self._run_test(
            "musicbrainz",
            self.mb_status,
            self.mb_test_btn,
            lambda: MusicBrainzProvider(cache_dir=None).test_connection(),
        )

    def _test_lrclib(self) -> None:
        self._run_test(
            "lrclib",
            self.lrclib_status,
            self.lrclib_test_btn,
            lambda: LrclibProvider(cache_dir=None).test_connection(),
        )

    def _test_shazamio(self) -> None:
        self._run_test(
            "shazamio",
            self.shazamio_status,
            self.shazamio_test_btn,
            lambda: ShazamIoProvider(cache_dir=None, require_enabled=False).test_connection(),
        )

    def _test_all(self) -> None:
        self._test_acoustid()
        if shazamio_installed():
            self._test_shazamio()
        self._test_setlistfm()
        self._test_musicbrainz()
        self._test_lrclib()

    def closeEvent(self, event) -> None:
        for w in self._workers:
            if w.isRunning():
                w.wait(1500)
        super().closeEvent(event)
