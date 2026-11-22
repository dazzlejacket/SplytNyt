# Third-party notices

SplytNyt is MIT-licensed (see [`LICENSE`](LICENSE)). It bundles or depends on
the components below. Full licence texts are in [`licenses/`](licenses/).

This file is also shipped, together with the `licenses/` folder, alongside the
installed Windows application.

---

## Runtime dependencies

These are imported by the application and are bundled into the Windows release
build.

| Component | Version at release | Licence | Why it is here | Text |
|---|---|---|---|---|
| [PySide6](https://pypi.org/project/PySide6/) (Qt 6 for Python) | 6.12.0 | LGPL-3.0-only | The user interface. | [`licenses/LGPL-3.0.txt`](licenses/LGPL-3.0.txt), [`licenses/GPL-3.0.txt`](licenses/GPL-3.0.txt) |
| [numpy](https://numpy.org/) | 2.5.5 | BSD-3-Clause | Vectorised DSP. | [`licenses/BSD-3-Clause-numpy.txt`](licenses/BSD-3-Clause-numpy.txt) |
| [scipy](https://scipy.org/) | 1.18.1 | BSD-3-Clause | Resampling and filtering. | [`licenses/BSD-3-Clause-scipy.txt`](licenses/BSD-3-Clause-scipy.txt) |
| [soundfile](https://pypi.org/project/soundfile/) | 0.14.0 | BSD-3-Clause | Python bindings for libsndfile. | [`licenses/BSD-3-Clause-soundfile.txt`](licenses/BSD-3-Clause-soundfile.txt) |
| [libsndfile](https://github.com/libsndfile/libsndfile) (via soundfile) | 1.2.x | LGPL-2.1-only | Decodes and encodes WAV, FLAC, AIFF, OGG and MP3. | [`licenses/LGPL-2.1.txt`](licenses/LGPL-2.1.txt) |
| [lameenc](https://pypi.org/project/lameenc/) | 1.9.0 | LGPL-2.1-only | Python bindings for the LAME encoder. | [`licenses/LGPL-2.1-LAME-lameenc.txt`](licenses/LGPL-2.1-LAME-lameenc.txt) |
| [sounddevice](https://pypi.org/project/sounddevice/) | 0.5.6 | BSD-2-Clause | Python bindings for PortAudio; used for playback. | [`licenses/BSD-2-Clause-sounddevice.txt`](licenses/BSD-2-Clause-sounddevice.txt) |
| [PortAudio](https://github.com/PortAudio/portaudio) (via sounddevice) | 19.x | MIT | The playback backend. | [`licenses/MIT.txt`](licenses/MIT.txt) |
| [CPython](https://www.python.org/) | 3.14 | PSF-2.0 | The runtime the application is built on. | [`licenses/Python-2.0.txt`](licenses/Python-2.0.txt) |

## Optional analysis dependencies

These are bundled in the Windows release build **when they are present in the
build environment**. They power the optional theatre-mode features. The
application degrades gracefully when they are absent.

| Component | Version at release | Licence | Why it is here | Text |
|---|---|---|---|---|
| [onnxruntime](https://github.com/microsoft/onnxruntime) | 1.32.0 | MIT | Runs the YAMNet audio-event classifier. | [`licenses/MIT-onnxruntime.txt`](licenses/MIT-onnxruntime.txt), [`licenses/MIT-onnxruntime-ThirdPartyNotices.txt`](licenses/MIT-onnxruntime-ThirdPartyNotices.txt) |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | 4.9.0 | MIT | Inference backend for the local speech recogniser. | [`licenses/MIT.txt`](licenses/MIT.txt) |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 1.2.1 | MIT | Local speech recognition. | [`licenses/MIT-faster-whisper.txt`](licenses/MIT-faster-whisper.txt) |
| [tokenizers](https://github.com/huggingface/tokenizers) | 0.23.x | Apache-2.0 | Tokenisation for the speech recogniser. | [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| [huggingface-hub](https://github.com/huggingface/huggingface_hub) | 2.4.0 | Apache-2.0 | Downloads the optional models on demand. | [`licenses/Apache-2.0-huggingface-hub.txt`](licenses/Apache-2.0-huggingface-hub.txt) |

### Models downloaded on demand

These are **not** bundled. They are downloaded into `~/.splytnyt/models` at your
request, with the size shown before the download starts.

| Model | Licence | Used for |
|---|---|---|
| [YAMNet](https://github.com/tensorflow/models) audio-event classifier (ONNX) + its class labels | Apache-2.0 (code) / CC BY 4.0 (label list) | Classifying music, speech, applause and silence |
| [Whisper *small*](https://github.com/openai/whisper) (int8, CPU) | MIT | Local transcription for theatre mode |

## Deliberately excluded

| Component | Licence | Why it is not here |
|---|---|---|
| [Mutagen](https://github.com/quodlibet/mutagen) | GPL-2.0-or-later | Linking it would impose GPL on this MIT-licensed application. The tag writers here (Vorbis Comments, ID3v2.4, RIFF INFO) are written from scratch instead, and Mutagen is excluded from the release build. |
| [ShazamIO](https://github.com/marrcel/shazamio) | MIT | Deliberately **not bundled**. It talks to an unofficial online service, so it is installed by the user, on request, from *Settings → Music Services*. See the README. |

## Build-time only

| Component | Licence | Used for |
|---|---|---|
| [PyInstaller](https://pyinstaller.org/) | GPL-2.0-or-later **with a special exception** | Freezing the application into an executable. The bundling exception expressly permits shipping a proprietary or differently-licensed application built with PyInstaller. |
| [pytest](https://pytest.org/) | MIT | The test suite. |
| [Inno Setup](https://jrsoftware.org/isinfo.php) | BSD-style | Building the Windows installer. The installer is a build artefact and is not redistributed as part of the source. |

---

## LGPL compliance note

Qt 6 (PySide6), libsndfile and LAME are used under the LGPL. They are consumed
as **separate shared libraries**, not statically linked into SplytNyt.

* You may replace those shared libraries with your own builds. Nothing in
  SplytNyt prevents it, and the application does not verify their identity.
* The corresponding licence texts are in the `licenses/` folder, which is
  installed next to the application and is also in this repository.
* Nothing here prevents reverse engineering for debugging modifications to the
  LGPL components, to the extent the LGPL permits it.

## Data sent to third parties

SplytNyt itself uploads nothing: your recordings never leave your machine.

The optional metadata providers are contacted only when you ask for a lookup:

| Service | What is sent | Terms |
|---|---|---|
| [MusicBrainz](https://musicbrainz.org/doc/MusicBrainz_API) | Search terms you type (artist, release, recording). No audio. | Free, no key. [Usage policy](https://musicbrainz.org/doc/MusicBrainz_API#Rate_Limiting) asks for no more than one request per second and requires a descriptive User-Agent. |
| [AcoustID](https://acoustid.org/api) | A Chromaprint fingerprint and a duration. No audio. | Free. The application key shipped with SplytNyt is registered to the project; using your own key is possible in *Settings → Music Services*. |
| [LRCLIB](https://lrclib.net/docs) | Track and artist names. No audio. | Free, no key. |
| [setlist.fm](https://www.setlist.fm/api) | Search terms you type, with **your** API key. No audio. | Free with a personal key. Every user needs their own key; none is shipped. |
| [ShazamIO](https://github.com/marrcel/shazamio) | A short audio clip, to an **unofficial** Shazam endpoint. | **Unofficial.** No API key, no agreement, no guarantee. Off by default and not bundled. |

Please respect these services' terms, and use them only for material you are
entitled to work with.

## Copyrighted material

No copyrighted recordings, sample audio, lyrics, transcripts or private
transcripts are included in this repository or in any release artefact. The
test suite generates its own synthetic material. The theatrical reference list
contains factual metadata only (number titles, order, approximate durations)
together with the sources it was drawn from.
