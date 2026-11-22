# Contributing to SplytNyt

Thanks for taking the time. This is a beta, so bug reports and honest
feedback about where the automatic detection goes wrong are the most useful
things you can send.

## Before you file an issue

* **Do not attach copyrighted recordings.** A `.splytnyt` project file is small
  and contains no audio — attach that instead, plus the relevant part of the
  log. Please never upload concert recordings, vinyl rips or anything else you
  do not own.
* Say what you expected and what happened, and mention your Windows version and
  the SplytNyt version (shown in *Help → About*).
* If a cut was wrong, it helps enormously to say **which track** and **where**.
  The review panel's *Technical details* panel has the numbers the detector
  used.

## Setting up

```bash
git clone https://github.com/dazzlejacket/SplytNyt.git
cd SplytNyt
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
python -m pytest
```

The test suite is offline. It uses recorded provider responses as fixtures and
synthesises its own audio, so it needs no network access, no API key and none
of your own recordings.

## Running the application

```bash
python -m splytnyt.main
```

To drive the real interface — building the window, importing a file through the
real import path, running the real analysis and clicking the real buttons:

```bash
python scripts/build_demo.py     # synthetic demo recording
python scripts/ui_exercise.py    # writes the screenshots in docs/
```

## Building the Windows release

```bash
python scripts/build_windows.py --clean
```

This produces `dist/SplytNyt/SplytNyt.exe` and everything it needs beside it.

To build the installer, install [Inno Setup 6](https://jrsoftware.org/isinfo.php)
and run:

```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" scripts\installer\SplytNyt.iss
```

The installer script reads `dist/SplytNyt/` and `licenses/`, so run the
PyInstaller build first.

## Code layout

```
splytnyt/
  core/        pure data model - no Qt, no audio, no numpy
  audio/       decode, peaks, measurement, gain, encoders, tags, playback
  analysis/    onset detection, boundary placement, classification, alignment
  metadata/    the optional providers
  app/         session, background jobs, export orchestration
  ui/          PySide6 widgets
```

The layering is strict: `core` imports nothing from the layers above it,
`audio` and `analysis` never import `ui`, and `ui` calls `app` for the
providers and the encoders. Please keep it that way — it is what lets the
service layer be tested without a display.

## Things that are not negotiable

These are the rules the project has held to from the start. A change that
breaks one of them will not be merged.

* **No copyrighted material in the repository.** No recordings, no sample
  audio, no lyrics, no transcripts. Test material is synthesised.
* **No personal credentials in the source.** The setlist.fm key belongs to each
  user and is stored in the Windows Credential Manager. The AcoustID key in the
  source is the project's own *application* key, not a personal one.
* **ShazamIO stays optional, unofficial and off by default**, and is never
  bundled into a release build.
* **Mutagen stays out of the release build.** It is GPL-2.0-or-later; this
  project is MIT-licensed and writes its own tag writers on purpose.
* **No silent audio processing.** SplytNyt never limits, compresses, denoises,
  declicks or resamples without saying so.
* **The exporter never writes over a source recording**, and an unprocessed
  lossless export must reproduce the source PCM bit for bit.

## Style

* Follow the surrounding code. The existing style is plain, typed Python with
  docstrings that explain *why* something is done, not just what.
* Keep the evidence visible. A detector that is not sure should say so rather
  than present a guess as a fact.
* Add a test for a behaviour change. The suite is fast and does not need a
  network.

## Pull requests

* One topic per pull request.
* Say what changed and why, and what you verified.
* If you change the release packaging or the licensing situation, say so
  explicitly — those get extra scrutiny.

## Licence

By contributing you agree that your contribution is licensed under the MIT
licence, the same as the rest of the project.
