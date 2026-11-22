# Musical-number grouping

The theatre pipeline now has a separate musical grouping pass **after** activity
classification and initial scene assembly. Export tracks follow proposed numbers
and scenes; speech, singing, instrumental material and applause remain timed,
overlapping project activities. Group membership also preserves the original
sections and their evidence as `cache.musical_grouping.groups[].evidence.internal_sections`.
All grouping decisions and reference comparisons are saved with the project.
The review panel's Technical details shows the internal ranges and join evidence.

## Continuity evidence

The pass reads the actual source PCM in blocks, downsamples only its analysis
copy to 8 kHz, and measures locally whitened, octave-pooled chroma every half
second. It compares evolving six-second phrases within 90 seconds of musical
material on either side of a spoken passage, across transpositions and local
phrase stretching. Ordered similarity must exceed an unordered pitch-profile
baseline and the pitch content must change. A shared key or static chord does
not establish continuity. This measures accompaniment/harmonic recurrence,
not a separately transcribed melody or an acoustically verified song title.

Transcript discourse contributes independent evidence: staged introductions
can belong to a medley; narrative or next-performance resets keep a boundary.
A demonstrated public presentation supplies context for later internal exchanges.
Speech dominance relative to weak accompaniment and audience closure help retain
independent scenes even when motifs return. Scores and lexical cues are heuristics,
not calibrated probabilities. They are English-language cues and can fail on ASR
errors, unrecognized discourse, or other languages. Automatic joins remain reviewable.
An absence of trusted evidence preserves separate units rather than forcing an
album-sized number. Existing duration heuristics still seed activity-derived
sections; this change does not increase minimum track lengths.

## Optional local reference audio

In Analyse > Theatre, choose **Load local reference audio manifest...**, then run
theatre segmentation. Only existing local files are read. Loading the manifest
is undoable and stored in the project. An empty reference list removes inputs.
The feature matching path never downloads recordings or treats tracklist data
as audio. Example manifest (edit the file path to your own authorized local audio):

```json
{
  "schema": "splytnyt-reference-audio/1",
  "references": [
    {
      "reference_id": "cast-number-01",
      "title": "Possible number title",
      "path": "C:/Music/My cast recording/01.flac",
      "query_ranges": [[10, 22], [50, 62]]
    }
  ]
}
```

Paths may be absolute or relative to the manifest. Each optional query range
must be 4–30 seconds; absent ranges select four sparse twelve-second fragments.
These are **query positions in the local reference**, never cutting positions
in the live recording. Octave pooling reduces singer/instrument differences;
transposition search and subsequence DTW steps (1,1), (1,2), (2,1) tolerate
local tempo differences. Independent fragments can anchor either side of
inserted dialogue without forcing the dialogue to align. Up to four occurrences
per fragment support reprises. Several reference IDs may anchor an already coherent medley;
missing references remain unmatched, and conflicting identities keep a boundary.
All reference results and titles remain unconfirmed. Large arrangement changes
may fail to match. Related chord progressions can produce false matches.

The cast-audio path is evaluated with synthesized audio and feature sequences,
not with an actual cast-to-live comparison. Metadata suggests an opening title;
it cannot establish its audible content, identity, or extent.

## Research

- [AudioLabs: Audio Matching](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_AudioMatching.html)
  describes chroma/CENS sequences and subsequence DTW, including limitations
  across differing instruments and arrangements. This implementation uses
  whitened chroma, not the notebook's full CENS implementation.
- [AudioLabs: Subsequence DTW](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html)
  explains local alignment and bounded step alternatives.
- [Official cast CD tracklist](https://store.moulinrougemusical.com/products/mrb-original-broadway-cast-cd)
  lists Welcome to the Moulin Rouge! first. This is metadata evidence only.
- [Official music page](https://moulinrougemusical.com/music/)
  describes the production's many constituent songs. An album item can be a
  medley and its duration is not an instruction for cutting a live production.

## Validation and limits

`tests/test_musical_grouping.py` separately checks number grouping, scene
preservation, reference alignment and editing protection. Its generated audio
includes changed tempo, key and overtone mix, inserted noise/dialogue space,
and a repeated fragment. It does not establish accuracy on real cast recordings.

Real-act activity accuracy and musical grouping accuracy are separately
reported as **not measured**: there is no independently annotated full-act
reference. Number counts, activity counts, cached model consistency and source
coverage are reported separately from accuracy. The opening extent is a
recording-derived review proposal, not a listening-certified timestamp.

Locked, manually placed and accepted tracks retain their complete boundaries
and metadata on reanalysis, including typed titles and manual grouping/type/tag
overrides persisted in the track model. The application applies grouping in one undoable
step; save/reopen retains the proposals and annotations. Playback and export
still use the same unchanged frame timeline. Complete export retains all audio.
