# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the Windows build of SplytNyt.

The build is a *one folder* build on purpose: a musical-theatre session
may want to keep the optional analysis models next to the application, and
a single file that unpacks itself on every launch is both slower and
harder to inspect.

The optional stacks - the audio-event classifier (onnxruntime) and the
transcriber (faster-whisper / ctranslate2 / tokenizers) - are included,
because the theatrical analysis needs them and a build that cannot do the
job it exists for is not worth shipping. The *models* are not included:
they are downloaded on demand into the user's profile, and their size
(about 16 MB for the classifier, about 486 MB for the transcriber) makes
them a poor thing to hide inside an executable.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs

HERE = os.path.dirname(os.path.abspath(SPEC))
ROOT = os.path.dirname(HERE)

block_cipher = None

# Native libraries that must travel with the application.
binaries = []
for package in ("soundfile", "onnxruntime", "ctranslate2", "tokenizers",
                "huggingface_hub", "sounddevice", "lameenc"):
    try:
        binaries += collect_dynamic_libs(package)
    except Exception:
        # A missing optional stack must not break the build: the
        # application degrades to the dependency-free path without it.
        pass

hiddenimports = [
    "splytnyt",
    "splytnyt.core.credentials",
    "splytnyt.ui.main_window",
    "splytnyt.ui.dialogs",
    "splytnyt.ui.settings_dialog",
    "splytnyt.ui.theme",
    "splytnyt.app.jobs",
    "splytnyt.app.session",
    "splytnyt.app.exporting",
    "splytnyt.analysis.aec",
    "splytnyt.analysis.activities",
    "splytnyt.analysis.theater",
    "splytnyt.analysis.transcribe",
    "splytnyt.analysis.medley",
    "splytnyt.analysis.align",
    "splytnyt.analysis.metadata_split",
    "splytnyt.metadata.references",
    "splytnyt.metadata.musicbrainz",
    "splytnyt.metadata.setlistfm",
    "splytnyt.metadata.acoustid",
    "splytnyt.metadata.lrclib",
    "splytnyt.metadata.identify",
    "splytnyt.metadata.shazamio_engine",
    "splytnyt.metadata.multi_engine",
    # Stdlib modules needed when the user installs the optional ShazamIO wheels at runtime
    "asyncio",
    "http.cookies",
    "decimal",
    "fractions",
    "uuid",
    "wave",
    "zipfile",
    "zoneinfo",
    # NOTE: Mutagen is deliberately NOT listed here. It is GPL-2.0-or-later
    # and bundling it would impose that licence on this MIT-licensed
    # application. SplytNyt writes its own tag formats instead, and reading
    # tags embedded in a source file is an optional extra that a source
    # install can enable by installing Mutagen itself.
    "soundfile",
    "lameenc",
    "sounddevice",
    "scipy.signal",
    "scipy.fft",
    "numpy",
]

# The optional analysis stacks. Missing ones are simply not collected.
for optional in (
    "onnxruntime",
    "onnxruntime.capi",
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "huggingface_hub",
):
    try:
        __import__(optional)
    except Exception:
        continue
    hiddenimports.append(optional)

excludes = [
    # Optional unofficial recognition engine: NEVER bundled in public EXE;
    # users can opt in and install it into their local profile from Settings.
    "shazamio",
    "shazamio_core",
    # GPL-2.0-or-later: must not be bundled into an MIT-licensed release.
    "mutagen",
    "tkinter",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "torch",
    "tensorflow",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.Qt3DCore",
    "PySide6.QtMultimedia",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtPdf",
]

a = Analysis(
    [os.path.join(ROOT, "scripts", "splytnyt_entry.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=[
        (os.path.join(ROOT, "splytnyt", "metadata", "references"),
         os.path.join("splytnyt", "metadata", "references")),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

# Pin the C/C++ runtime to the versions shipped with Qt/Python. Otherwise
# PyInstaller can pick incompatible DLLs from unrelated tools on PATH.
import PySide6
qt_dir = os.path.dirname(PySide6.__file__)
runtime_sources = {
    "msvcp140.dll": os.path.join(qt_dir, "MSVCP140.dll"),
    "vcruntime140.dll": os.path.join(sys.base_prefix, "vcruntime140.dll"),
    "vcruntime140_1.dll": os.path.join(sys.base_prefix, "vcruntime140_1.dll"),
    "ucrtbase.dll": os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "ucrtbase.dll"),
}
# ICU from Poppler/Conda has versioned exports and cannot satisfy Qt's
# imports from the Windows ICU API. Let Windows resolve its system ICU.
a.binaries = [(name, source, kind) for name, source, kind in a.binaries
              if name.lower() not in ("icuuc.dll", "icudt78.dll")]
a.binaries = [
    (name, runtime_sources.get(name.lower(), source), kind)
    if name.lower() in runtime_sources and os.path.isfile(runtime_sources[name.lower()])
    else (name, source, kind)
    for name, source, kind in a.binaries
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SplytNyt",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SplytNyt",
)
