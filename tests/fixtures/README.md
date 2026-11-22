# Recorded provider fixtures

These files are the responses the provider adapters are tested against. The
automated test suite never touches the network: it replays these files
through the real HTTP client, so the caching, the User-Agent, the rate limit
and the error handling are all exercised.

| File | Provenance |
| --- | --- |
| `musicbrainz_release_group_search.json` | **Recorded** from musicbrainz.org (`/ws/2/release-group/?query=…`), trimmed to the first two results. |
| `musicbrainz_release_group.json` | **Recorded** from musicbrainz.org (`/ws/2/release-group/{mbid}?inc=releases+media`), trimmed to the first three editions. |
| `musicbrainz_release.json` | **Recorded** from musicbrainz.org (`/ws/2/release/{mbid}?inc=recordings+media+artist-credits`) - a two-disc 10" vinyl edition of *Kid A*. |
| `setlistfm_search.json` | **Hand-written** sample of the documented setlist.fm response shape. setlist.fm needs an API key for every request, so no live response is recorded here. |
| `setlistfm_setlist.json` | **Hand-written** sample of the documented setlist.fm response shape, with a second set, an encore and a medley. |

The MusicBrainz files are real recordings. The setlist.fm files are not: they
follow the shape setlist.fm documents, and they were written by hand to
exercise the adapter. The live setlist.fm path is therefore covered by a
fixture of the documented shape rather than by a recorded response.

Re-record the MusicBrainz files with:

```
python scripts/record_fixtures.py
```
