# SplytNyt — Architecture & Milestone Plan

Working title: **SplytNyt** (provisional). A local-first, cross-platform desktop
application that splits long audio recordings into named, tagged tracks.

---

## 1. Scope of this document

This document fixes the technical shape of the application *before* code is
written: the stack, the module boundaries, the data model, the safety
invariants, and the phase plan. Phase 1 is implemented against it; Phases 2
and 3 are specified so the architecture is known to accommodate them.

**This is not a DAW.** It does not mix, edit, or master. It *segments* a
recording into a timeline of tracks, *labels* them, and *exports* them.

---

## 2. Technology stack and why

| Concern | Choice | Licence | Rationale |
|---|---|---|---|
| Language / runtime | CPython 3.12+ (3.14 validated) | PSF | Mature DSP + audio ecosystem; identical source on all 3 OSes. |
| UI toolkit | PySide6 (Qt 6) | LGPL-3.0 (or commercial) | Only Python toolkit with a first-class desktop story on Win/mac/Linux, native widgets, QThread workers, and packaged wheels for all three platforms. |
| Numeric DSP | NumPy (+ SciPy) | BSD-3 | Vectorised analysis over float32 blocks; SciPy for resampling/filtering used by loudness measurement. |
| Decode / lossless encode | libsndfile via `soundfile` | LGPL-2.1 | Decodes FLAC, WAV, AIFF, OGG, and (≥0.13) MP3. Encodes FLAC and WAV. Reads/writes are streaming and block-based, so multi-hour files never sit in RAM. |
| MP3 encode | LAME via `lameenc` | LGPL-2.1 | Mature, small, well-understood encoder; exposes encoder delay/padding so we can report timing truthfully. |
| Tag writing | **In-house tag writer** (`splytnyt/tags/`) | MIT (ours) | Deliberately *not* Mutagen: Mutagen is GPL-2.0+, and linking it would force the whole app's licence. We write Vorbis Comments (FLAC), ID3v2.4 (MP3), and RIFF INFO/ID3 (WAV) directly — a few hundred lines, full control, no copyleft. |
| Playback | `sounddevice` (PortAudio) | MIT / MIT | Sample-accurate pull-style callback from the same decoder used for export, so what you hear is what gets written. |
| Build / package | PyInstaller + OS-specific CI | GPL-2.0 (with bootloader exception) | Produces a per-platform binary; `--onedir` keeps the LGPL linking obligations straightforward. |

**Rejected alternatives and why**

- **Electron / Tauri + Rust + WebAudio.** Rust would give us the best audio
  runtime, but the FFmpeg/whisper/ML ecosystem is Python-shaped, and rewriting
  the analysis layer in Rust would put Phase 3 far out of reach. Tauri needs a
  system webview, which varies by Linux distro — the worst cross-platform
  bet of the three.
- **Qt for Python + libmp3lame via ffmpeg subprocess.** Shelling out to an
  ffmpeg binary makes distribution fragile (licence differs per build) and
  adds process-management surface to every export.
- **JUCE / a C++ app.** Best audio performance, worst iteration speed for the
  UI-heavy review workflow we actually need, and the smallest ecosystem for
  metadata providers.

**Acceleration.** CPU-only by default. Optional acceleration is a *later*
concern: the design isolates the hot loops behind `splytnyt.audio` and
`splytnyt.analysis` so a GPU/ONNX path can be dropped in behind the same
interface in Phase 3.

---

## 3. Layering

```
┌──────────────────────────────────────────────────────────────┐
│ ui/          PySide6 widgets, QThread workers, no DSP logic   │
├──────────────────────────────────────────────────────────────┤
│ app/         Application services: orchestration, undo,       │
│               export jobs, provider calls, autosave           │
├───────────────────────┬──────────────────────────────────────┤
│ analysis/              │ metadata/                            │
│ silence, novelty,      │ provider adapters (base/manual/      │
│ classification,         │ setlistfm/musicbrainz/acoustid),     │
│ segmentation, pipeline │ alignment, caching, rate limits      │
├───────────────────────┴──────────────────────────────────────┤
│ core/        Pure data model. No Qt, no I/O, no numpy.        │
│              Project, Source, Track, Metadata, undo stacks,   │
│              serialisation, unit conversion, quality flags.   │
├──────────────────────────────────────────────────────────────┤
│ audio/       Engine. Decoding, peaks, measurement, gain,      │
│              encoders, tagging. Pure Python + numpy/soundfile.│
└──────────────────────────────────────────────────────────────┘
```

**Hard rules the layering enforces**

1. `core` imports nothing from `audio`, `analysis`, `metadata`, or `ui`. The
   project model is portable data and is unit-testable with zero native deps.
2. `audio` and `analysis` never import `ui`. They emit results and signals;
   the UI renders them.
3. `ui` never calls a provider or an encoder directly. It calls `app/`.
4. Nothing in the codebase opens a file in a mode that can mutate it.

---

## 4. Core data model (`core/`)

### 4.1 Integer sample positions everywhere

Every boundary, every track edge, every duration is an `int64` **frame
index** in a single global timeline coordinate space. Seconds are a *derived
view* (`frames / sample_rate`) computed only at the UI boundary. Nothing in
the engine stores floats for time.

The timeline is the concatenation of the project's sources in order:

```
source i occupies timeline frames [start_i, start_i + frames_i)
start_0 = 0
start_i = start_{i-1} + frames_{i-1} + gap_i     gap_i may be 0 (default)
```

`gap_i` is an explicit, user-visible property of the project, never inferred
and never silently collapsed. A negative gap means the sources **overlap**;
both regions are kept and reported.

### 4.2 Entities

```
Project
 ├─ id, name, created, modified, schema_version
 ├─ sources: [Source]                 ordered, immutable file references
 ├─ tracks: [Track]                   ordered, non-overlapping, contiguous-or-gapped
 ├─ worklists: [Worklist]             pasted/imported tracklists (mode-independent)
 ├─ markers: [Marker]                 named boundary annotations
 ├─ export_settings: ExportSettings
 ├─ metadata: Metadata                album-level fields
 └─ analysis_cache: {key: cache_entry}

Source
 ├─ source_id, path, original_format
 ├─ sample_rate, channels, frames    (as decoded)
 ├─ fingerprint: size, mtime_ns, head_hash, tail_hash
 ├─ gap_frames                        (to previous source)
 └─ missing: bool                     (file absent since last open)

Track
 ├─ track_id
 ├─ start_frame, end_frame            (global timeline, int64)
 ├─ title, metadata                   (editable, per-track)
 ├─ section_type                      music|applause|speech|silence|unknown|uncertain
 ├─ boundary_confidence               0..1  (how sure we are of THIS cut)
 ├─ identity_confidence               0..1  (how sure we are of WHAT it is)
 ├─ classification_confidence         0..1  (how sure of the section label)
 ├─ reasons: [str]                    human-readable evidence for the cut
 ├─ locked                            (boundary + metadata pinned)
 ├─ flags: [Flag]                     short/long/mismatch/discontinuity/overlap
 └─ worklist_ref                      link into a Worklist for identity

Worklist          # a candidate song list (pasted, or from a provider in Phase 2)
 ├─ items: [WorklistItem]  { title, duration_hint, position, notes }
 └─ provider, provider_id, fetched_at, cached

Marker            # named boundary annotation / comment
```

**Two confidences, never merged.** `boundary_confidence` answers *"is this
cut in the right place?"*; `identity_confidence` answers *"is this the song I
think it is?"*. They are stored, displayed, filtered, and exported separately.
The product rule "keep song identification confidence separate from boundary
confidence" is therefore a schema property, not a UI convention.

### 4.3 Non-destructive guarantee

Source files are opened **read-only, always**. Exports go to a user-chosen
output directory that is checked against the source paths before any byte is
written. See §8.

---

## 5. Audio engine (`audio/`)

| Module | Responsibility |
|---|---|
| `decode.py` | `AudioSource` — streaming reader over libsndfile. Yields float32 blocks of shape `(n, channels)`. Exposes `frames`, `sample_rate`, `channels`. Never materialises the whole file. |
| `peaks.py` | Streaming min/max **peak pyramid** (multi-resolution), persisted to a sidecar cache file so reopening a project is instant. Powers the waveform at every zoom level. |
| `measure.py` | True-peak (4× oversampled), RMS, EBU R128 integrated loudness (BS.1770-4, with gating), ReplayGain 1.0/2.0, per-track and whole-program measurements. |
| `gain.py` | Gain application policy: untouched / shared program gain / loudness normalise with true-peak ceiling / per-track normalise. Computes and returns the **trade-off report** when a target cannot be met without limiting. |
| `encode/flac.py` | Lossless FLAC via libsndfile, then Vorbis-comment insertion by `tags/flac.py`. |
| `encode/wav.py` | WAV via libsndfile + RIFF INFO / ID3 chunk insertion. |
| `encode/mp3.py` | MP3 via LAME, CBR/VBR, with encoder delay/padding accounting. |
| `encode/cue.py` | CUE sheet writer for the continuous-audio-plus-CUE workflow. |
| `tags/` | Vorbis Comments, ID3v2.4, RIFF INFO writers (ours; see §2). |

### 5.1 Streaming discipline

Everything that touches full-length audio is a block loop:

```
for block in source.blocks(frames=1 << 16):   # float32, interleaved->planar
    ...
```

Peak building, loudness measurement, export, and analysis all use this. A
6-hour 96 kHz/24-bit stereo file is never loaded into memory.

### 5.2 Measurement honesty

- **True peak** is measured on a 4× oversampled signal, not the sample peak.
- **Integrated loudness** implements ITU-R BS.1770-4 with absolute + relative
  gating, so silence between tracks does not drag the number down.
- If a requested loudness target cannot be reached without exceeding the
  true-peak ceiling, `gain.py` returns `GainPlan(required_db, peak_after_db,
  feasible=False, reason=...)`. The UI must present the trade-off and require
  an explicit choice: apply a **limiter** (opt-in, explicit) or use **less
  gain**. SplytNyt never silently limits.
- ReplayGain values are written as Vorbis `REPLAYGAIN_*` and ID3
  `TXXX` frames where the format supports them; they are *metadata*, not
  applied processing.

### 5.3 Lossy sources

Export records the source's origin. A track exported to FLAC from an MP3
source is lossless **re-encoding of a lossy source**; the UI and the info file
both say so explicitly ("Lossless container, lossy origin").

---

## 6. Analysis pipeline (`analysis/`)

Three **separate** stages, exactly as the product requires. They never run
as one fused "smart detector":

```
  ┌─ Stage 1 ──────────┐   ┌─ Stage 2 ───────────┐   ┌─ Stage 3 ────────────┐
  │ Boundary detection │   │ Section             │   │ Track identification  │
  │                    │   │ classification      │   │                      │
  │ silence/noise floor│→  │ music/applause/    │→  │ align a worklist to  │
  │ spectral novelty   │   │ speech/silence     │   │ detected segments    │
  │ energy change      │   │ (heuristic in P1)  │   │ (manual now,        │
  │                    │   │                    │   │ providers in P2)     │
  └────────────────────┘   └────────────────────┘   └──────────────────────┘
        confidence              confidence                confidence
        (boundary)            (classification)            (identity)
```

- **Stage 1 — Boundary detection.** Adaptive noise-floor estimation (robust
  low percentile of frame RMS), silence-run detection with a configurable
  minimum silence length, and spectral-flux novelty peaks. Candidates are
  merged and each keeps a `reason` string ("0.9 s silence", "spectral change
  at 12:41", "uncertain transition"). A *spectral change inside continuous
  music* is reported as **uncertain**, never as a confident cut.
- **Stage 2 — Section classification.** In Phase 1 this is an honest
  **heuristic** classifier (energy + spectral flatness + band ratios) labelled
  in the UI as `heuristic`. The interface is designed so a Phase 3 neural
  classifier (YAMNet/other, optional download, size disclosed) can replace it
  behind the same call signature.
- **Stage 3 — Track identification.** Phase 1 supports a pasted/manual
  tracklist, aligned to detected segments by a duration-aware dynamic
  program. Phase 2 adds setlist.fm / MusicBrainz providers behind the same
  `Provider` interface.

**Sensitivity presets** (conservative / balanced / sensitive) map to concrete
parameter sets, not to vague adjectives. Conservative demands long silences
and strong novelty peaks; sensitive accepts short silences and weak peaks.

**Never force a match.** Alignment may leave segments unmatched or worklist
items unmatched; both sets are reported rather than forced.

---

## 7. Metadata providers (`metadata/`)

```python
class Provider(Protocol):
    name: str
    def search(self, query) -> list[SearchResult]: ...
    def fetch(self, ref) -> Worklist: ...
    def supports_caching(self) -> bool: ...
```

Every adapter must implement:
- **caching** (on-disk cache keyed by request, with TTL),
- **rate limiting** (token bucket per provider),
- **graceful failure** (network error, 404, malformed response → a typed
  error surfaced in the UI, never a crash, never a silent empty result),
- **credential hygiene** (keys read from the environment or the OS keychain,
  never from source, never written to logs).

Provider **terms of service must be verified before an integration ships**;
the setlist.fm and MusicBrainz adapters are gated on that check. MusicBrainz
requires a proper `User-Agent` with contact info and rate limiting to 1 req/s
— this is implemented in the adapter, not left to the user.

---

## 8. Export (`app/exporting.py`, `audio/encode/`)

- **Two shapes.** (a) *Separate tracks*: one file per track. (b) *Continuous
  + CUE*: the program audio as one file plus a CUE sheet referencing it.
- **Sample-accurate internal cuts.** Boundaries are int64 frame indices; the
  CUE writer quantises to `MM:SS:FF` (75 fps) and the quantisation error is
  reported in the export log and the info file — never hidden.
- **MP3 honesty.** We do not promise sample-accurate MP3 stream-copy cuts. We
  *re-encode* the selected region, and we report encoder delay/padding
  (LAME's `encoder_delay`/`encoder_padding`) plus the residual offset in the
  export log.
- **Preserve source sample rate and channel layout** by default. The export
  dialog shows both and offers explicit overrides; a change is recorded in the
  info file.
- **Stereo balance** is preserved; shared gain and per-track gain are applied
  identically to all channels. No channel-specific processing exists in the
  codebase.
- **No automatic fades or crossfades**, no automatic de-noise, de-click,
  compression, limiting, or resampling.
- **Safe writes.** Export writes to `<name>.partial` in the destination,
  flushes, verifies, then atomically renames. A crash leaves the partial file
  marked as such, never a plausible-looking corrupt final file.
- **Never overwrite inputs.** Before writing, every candidate output path is
  compared against every source path using (a) normalised absolute path,
  (b) `os.path.samefile` where the target exists, and (c) on Windows the
  volume serial + file index. A match aborts the export with a clear error.
- **Collision protection** on the filename template: `{n}`, `{title}`,
  `{artist}`, `{album}`, `{disc}`, `{date}` plus an auto-incrementing
  `{uniq}` slot, with illegal-character sanitisation per OS.

### 8.1 Marker interchange

A documented JSON format (`.splytnyt-markers.json`) plus CSV and Audacity
label formats are importable/exportable, so boundaries survive a round trip
through other tools.

### 8.2 Archival / sharing package

A plain-text `.info` file containing artist/production, date, venue, city,
country, tour, source notes, a numbered tracklist with durations and total
runtime, user-supplied equipment and lineage, source formats, the **actual**
processing/export steps taken, and notes about gaps, incomplete songs, and
uncertainty. Optional SHA-256 checksums of the exported files, and optional
`flac -t`-equivalent verification (implemented in-house via libsndfile
re-decode comparison). The package makes **no claim** about satisfying any
particular sharing site's rules, and contains no upload client.

---

## 9. UI design (`ui/`)

- **Waveform view.** Custom `QAbstractScrollArea`. Multi-resolution peak
  pyramid; zoom with `Ctrl+wheel`, pan with drag/space; click to place a
  cursor; drag a boundary marker to nudge it; markers show their confidence
  as a colour band. Draws source boundaries and gaps distinctly.
- **Track table.** `QTableView` over a model: title, start, end, duration,
  section type, confidence columns, flags, lock toggle. Editable in place.
- **Review helpers.** Play a configurable window around a boundary (default
  3 s before / 2 s after) so the cut can be judged in context.
- **Keyboard shortcuts.** `←/→` previous/next boundary, `Space` play/pause,
  `Enter` accept suggestion, `S` split at cursor, `M` merge with previous,
  `[` / `]` nudge boundary by 10 ms, `Shift+[` / `Shift+]` by 1 s,
  `Ctrl+Z` / `Ctrl+Shift+Z` undo/redo, `L` lock/unlock, `A` accept all
  suggestions (with confirmation).
- **Honest explanations.** Every proposed cut carries a reason chip; the
  table and the boundary popup both show it.
- **Flags.** Track-count mismatch vs. the active worklist, suspiciously
  short/long tracks, source-boundary crossings, gaps/overlaps between
  sources, and unknown-source files.
- **Responsiveness.** Analysis and export run in `QThread` workers with
  progress + cancellation. The UI never blocks on audio work.

### 9.1 Project persistence

A single portable `.splytnyt` file: a ZIP container holding a versioned JSON
document, the peak-pyramid cache, and optionally a copy of the sources
(reference mode by default — sources stay where they are).

- **Autosave** to a recovery file every N seconds and on every structural
  edit.
- **Crash recovery**: on launch, if a recovery file is newer than the last
  saved project, offer to restore it.
- **Undo/redo**: snapshot-based (the model is small, pure data). Analysis
  caches and audio data live *outside* the undoable state.
- **Locks**: a locked track's boundaries *and* metadata are preserved across
  re-analysis. Re-analysis only proposes; it never overwrites a locked track.

---

## 10. Safety, failure handling, and concurrency

| Situation | Behaviour |
|---|---|
| Disk full mid-export | Exception surfaces, partial file deleted or left clearly marked `.partial`, source untouched. |
| Corrupt input | Decode error is reported with the file and block offset; the rest of the project stays usable. |
| Missing source | Project still opens; the source is flagged `missing`; export refuses to proceed without it. |
| Interrupted export | Partial files are recognisable and never presented as complete. |
| Cancelled analysis | Worker observes the cancel token between blocks; partial cache is not committed. |
| Concurrent edits | All project mutations go through the main thread; workers only *propose* changes via signals. |
| Clock/timezone | Dates are stored as ISO-8601 strings, never as datetimes. |

---

## 11. Phase plan

### Phase 1 — Working editor (this implementation)

Delivered and verified in this repository:

- Import one or more audio files into a single ordered timeline (with explicit
  inter-source gaps/overlaps preserved).
- Zoomable waveform with draggable boundary markers and a multi-resolution
  peak pyramid cached to disk.
- Sample-accurate playback via the same decoder used for export, with
  playhead, region loop, and boundary preview (configurable before/after).
- Manual boundaries: place, drag, nudge, split, merge.
- Track table with title, start, end, duration, section type, confidence,
  flags; editable in place.
- Undo/redo, autosave, crash recovery, portable project files.
- Pasted/manual tracklists with duration-aware alignment to detected segments.
- Basic **adaptive** boundary suggestions (noise floor + silence runs +
  spectral novelty) with per-cut reasons and honest confidence; conservative,
  balanced, and sensitive presets.
- Shared program gain and per-track gain (no limiting/compression without
  explicit opt-in; loudness normalisation with a true-peak ceiling and an
  explicit trade-off report).
- Verified export to FLAC, WAV, and MP3 with tagging; separate-track and
  continuous + CUE; filename templates with collision protection; marker
  export/import (JSON, CSV, Audacity); optional M3U; archival info file;
  optional SHA-256 checksums.

### Phase 2 — Metadata providers

**Implemented in this build:**

- setlist.fm adapter (concert mode): search by artist/date/venue, let the user
  confirm the performance and edit the list, preserve sets/encores/covers/
  medleys/notes, never force entries onto the audio. Needs an API key in
  `SETLISTFM_API_KEY`; without one the provider says so and the application
  carries on working.
- MusicBrainz adapter (vinyl/cassette mode): exact release/edition selection,
  format, year, track order, disc/side positions, durations as *approximate
  evidence*. Needs no key, but does need a real User-Agent; set
  `SPLYTNYT_CONTACT` to add your contact details to it.
- Both providers cache their answers on disk for seven days, keep to one
  request a second, and fail with typed, readable errors. A provider failure
  never touches the project.
- Metadata alignment: a pasted or fetched tracklist is lined up against the
  detected segments, tolerating missing and extra songs, and reporting both
  ends of every mismatch. An aligned title is labelled *Aligned*, never
  *verified*.

**Still to do in this phase:**

- AcoustID identification as an optional aid.
- Improved segmentation informed by the aligned worklist (still
  non-destructive; suggestions remain editable).

### Phase 2 — Boundary placement

Also implemented in this build, and the reason a proposed cut can be measured
against song starts rather than against gap centres:

- The detection of a **transition region** and the **placement of the cut**
  inside it are separate questions, in separate modules.
- Four placements: near the next song start (default, with a configurable
  margin), near the previous song's end, the middle of the gap, and manual.
- The **onset** and the **cut** are separate numbers. The onset is found with
  a coarse long-window search and then a fine short-window search that
  requires a *rise*, so a decaying tail cannot win over a real attack. The
  cut is the onset minus a small margin, and the applied margin is recorded
  on the track.
- The default margin lives in `core/model.py`, so the project, the analysis
  and the refinement cannot drift apart.
- A quiet passage inside a song is rejected by comparing the timbre on the two
  sides of the gap, and applause or speech is never presented as music.
- Refinement is explicit: the job proposes, the preview shows, the user
  accepts, and a locked track is never moved.

### Phase 3 — Models, theatre, transcription

- Optional audio-event models (music/applause/speech) with **disclosed
  download sizes** and graceful fallback to the heuristic classifier.
  **Implemented.** YAMNet (ONNX, 16.1 MB) is the classifier; the download shows
  its size, reports progress, can be cancelled and falls back to the
  dependency-free DSP path when it is missing.
- Musical-theater mode: acts, overtures, songs, reprises, dialogue,
  underscoring, applause, intermissions, curtain calls; overlapping labels
  (dialogue-over-music is a valid state, not "speech or music").
  **Implemented**, with one deliberate limit: the section types are the ones
  above minus *overture*, *intermission* and *curtain call*, which are not
  detected on their own. See §6.1.
- Optional local Whisper transcription to assist review and text matching —
  never as the sole boundary authority, never inventing song titles.
  **Implemented.** faster-whisper *small*, int8, CPU, run locally. The
  transcript is evidence: it places the spoken-word regions, and every line it
  made up is rejected rather than believed.
- Optional alignment against user-supplied scripts/lyrics. **Partly
  implemented**: a reference list of a production's numbers, with provenance,
  aligns to the detected numbers and proposes names. The alignment never
  forces a match and the names are labelled as evidence.
- Complete-performance vs songs-only export profiles; songs-only previews all
  omissions and leaves sources untouched. **Implemented.**
- More advanced unknown-recording segmentation. **Not implemented.**

### 6.1 The theatre pipeline

The theatre mode is a second analysis pipeline beside the ordinary one. It
never replaces it, and the ordinary modes are unaffected by it.

```
  ┌─ Classify ────────┐   ┌─ Transcribe ──┐   ┌─ Assemble ─────────────┐
  │ YAMNet (ONNX),    │   │ Whisper,      │   │ overlapping activity   │
  │ read in blocks,   │→  │ local, int8,  │→  │ regions → sections     │
  │ drift-free frame  │   │ hallucination │   │ (number / dialogue /   │
  │ selection         │   │ rejection     │   │  underscoring /        │
  └───────────────────┘   └───────────────┘   │  applause / mixed)     │
                                                └───────────────────────┘
```

Two decisions are worth stating, because they were made from measurement
rather than from the literature:

- **The boundaries come from the spoken words and the applause.** In a stage
  recording the orchestra plays through almost everything, so the edges of
  the *music* carry no information. The edges of the *speech* and of the
  *applause* do.
- **Words over a band playing at full volume are lyrics, not dialogue.** This
  was measured over 763 accepted transcript stretches of a real act: the
  stretches over the band had a music score of 0.53–0.84 and the spoken ones
  0.00–0.19, while the singing score, the no-speech probability, the average
  log probability, the characters per second and the stretch length all
  overlapped. The music score is therefore the gate, and it is read over a
  window that surrounds the words, so that a momentary gap in the band
  cannot cut a number in half.

A line of *dialogue* over a loud band is acoustically indistinguishable from
a sung line. The engine does not guess: such a section keeps its type, is
marked uncertain, and is listed for a human to check.

A **medley** is one number that holds several songs. The number keeps its
own title and its own boundaries; the songs inside it are separate objects
with their own names, their own boundaries and their own confidence, and
the exporter writes them as submarkers rather than merging them into the
number. Finding them does *not* use the musical-change detector, which was
measured to find nothing inside a real medley — the band plays straight
through the songs in one — but the lyrics, which do change from one song
to the next. The rules are strict on purpose, because a title's words turn
up in the lyric of other songs: a name is proposed only where the words of
a title agree across a **run** of at least four consecutive transcript
stretches (of at least six words each), where the agreement score reaches
**0.8**, and where no other reference title agrees with the same words as
well. Two titles that agree equally cancel each other out, and the confidence
of a located song is **capped at 0.75**, because a proposal is not an
identification. A title that cannot be located is not guessed at: it is
listed by name in the number's own notes as one the recording did not give
enough evidence for.

**Rule for every phase:** a phase ships as a working application. Advanced
features stay out of the UI until they are implemented.

---

## 12. Validation strategy

**Objective (machine-checkable):**
- **PCM reconstruction.** For an unprocessed lossless export where every
  section is included, concatenating the exported tracks must reproduce the
  original decoded PCM **bit-exactly**. Automated test, all supported source
  formats, both mono and stereo.
- Boundary coverage: no gap and no overlap in the emitted timeline, except
  where the user explicitly created one.
- Channel preservation: exported channel count and inter-channel phase
  relationship match the source.
- Timing: each exported file's length equals `end - start` in frames exactly.
- Tagging: written tags read back correctly.
- Source protection: exporting to a directory containing the source, or to
  the source path itself, aborts with an error and leaves the source
  unchanged.
- Cancellation: cancelling an export leaves no file that looks complete.
- Reopen round-trip: a project saved and reloaded produces an identical
  model.

**Subjective (human judgement, reported honestly):**
- Detection quality is evaluated against manually marked recordings.
- **Objective timing errors** (detected boundary vs. manual boundary, in ms)
  are reported separately from **subjective editorial choices** (where a
  human deliberately moved a cut for musical reasons). The two are never
  averaged into a single misleading number.

Test corpus covers: quiet songs, noisy tape, applause during music,
uninterrupted medleys, dialogue over underscoring, reprises, incomplete
recordings, incorrect tracklists, and recorder files split mid-song.
