# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-beta] — 2026-11-21

First public beta.

### Added

- Multi-file import into one ordered timeline, preserving explicit gaps and
  overlaps between files.
- Zoomable multi-resolution waveform with draggable, shareable track boundaries;
  place, drag, nudge (10 ms / 100 ms / 1 s), split and merge.
- Undo/redo, autosave, crash recovery and portable `.splytnyt` project files.
- Automatic split suggestions: adaptive silence detection plus musical-change
  detection, with conservative / balanced / sensitive presets.
- Cut placement that defaults to the start of the next song, with a
  configurable pre-roll margin and an explicit scope (this cut only, or all new
  suggestions).
- A review panel that shows the evidence behind every proposed cut, with the
  numbers behind it, zoom-to-cut, play-across, preview-exported-start and
  accept / lock actions.
- Metadata-assisted splitting via MusicBrainz, AcoustID (Chromaprint) and
  LRCLIB, plus optional setlist.fm concert lookup.
- Theatre segmentation mode: distinguishes musical numbers, dialogue,
  instrumental transitions, applause and speech over music.
- Reference lists for a production, shipped as data with provenance, that
  suggest names without ever forcing a match.
- Tracklist pasting and alignment, where durations are approximate evidence
  and unmatched entries are reported rather than forced.
- Optional, opt-in **ShazamIO (Experimental)** recognition engine, installed
  and uninstalled from inside the app with no Python or command line required.
  Disabled by default and never bundled in the release build.
- Export to FLAC, WAV and MP3 with tags, plus a CUE sheet for a single
  continuous file, marker export/import (JSON, CSV, Audacity labels), optional
  M3U, a plain-text info file, SHA-256 checksums and FLAC verification.
- Two export profiles: *complete performance* (every stretch is written, so no
  audio is lost) and *songs only* (musical numbers only, with every exclusion
  reported with its reason).
- Light, dark and follow-the-system themes.

### Security and packaging

- The SplytNyt AcoustID application key ships in the application so that
  fingerprint lookup works without user registration. It is an application
  key, not a personal key.
- No personal API key is included. setlist.fm requires the user to supply
  their own key, which is stored in the Windows Credential Manager.
- The release build excludes Mutagen (GPL-2.0-or-later) so that the
  distributed application stays MIT-licensed.
- No copyrighted recordings, sample audio, lyrics or transcripts are included
  in the source repository or in any release artefact.

### Known limitations

- Automatic detection is not perfect. Live concert recordings in particular
  usually need manual correction.
- ShazamIO talks to an unofficial online service and may rate-limit or stop
  working without notice.
