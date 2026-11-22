# Example configuration

These files are **examples**. They contain no credentials and no personal paths.

## `settings.example.json`

SplytNyt keeps its settings in:

| Platform | Path |
|---|---|
| Windows | `%USERPROFILE%\.splytnyt\settings.json` |
| macOS / Linux | `~/.splytnyt/settings.json` |

Copy `settings.example.json` there and edit it, or simply change the settings
in the interface and let SplytNyt write the file for you.

The only setting in it is `shazamio_enabled`, which is the opt-in toggle for
the optional, unofficial **ShazamIO (Experimental)** recognition engine. It is
`false` in the example, and that is the default: SplytNyt never turns it on by
itself.

### API keys are not stored here

Your **setlist.fm** key, and any **AcoustID** override you choose to use, are
stored in the **Windows Credential Manager** — not in this file, not in a
project file and not in the source code. Enter them under
*Settings → Music Services*.

You can also supply them through environment variables, which take precedence:

| Variable | Service |
|---|---|
| `SETLISTFM_API_KEY` | setlist.fm (your own personal key) |
| `ACOUSTID_API_KEY` | AcoustID (optional; overrides the built-in application key) |

**Never put a key in a file that you commit.** Never commit one.

## `reference-audio.example.json`

An example manifest for the optional local reference audio feature. Use it with
*Analyse → Theatre → Load local reference audio manifest…*

Every `path` must point at a file **you own and are allowed to use**. SplytNyt
only reads local files: it never downloads a recording, and it never treats
tracklist metadata as audio.

`query_ranges` are **query positions inside the local reference**, never cutting
positions in the recording you are analysing. Each range must be 4–30 seconds.
Leave them out and four sparse twelve-second fragments are chosen for you.
