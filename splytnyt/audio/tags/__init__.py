"""In-house tag writers.

We write the tag formats ourselves instead of using Mutagen: Mutagen is
GPL-2.0+, and linking it would force this application's licence (see
``docs/ARCHITECTURE.md`` section 2).

Three formats are covered:

* :mod:`splytnyt.audio.tags.flac` - Vorbis comments and RFC 9639 pictures,
* :mod:`splytnyt.audio.tags.id3v2` - ID3v2.4 frames,
* :mod:`splytnyt.audio.tags.riff` - RIFF INFO lists and ``id3 `` chunks.

Every writer rewrites its file atomically through ``<path>.partial`` and
never touches a source recording.
"""

from .flac import write_vorbis_comments
from .id3v2 import build_id3v2, map_tag_key_to_frame, write_id3v2
from .riff import write_riff_info

__all__ = [
    "build_id3v2",
    "map_tag_key_to_frame",
    "write_id3v2",
    "write_riff_info",
    "write_vorbis_comments",
]
