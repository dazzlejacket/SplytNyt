"""Finding the songs inside a number.

A medley is one number that holds several songs. The number keeps its own
title and its own boundaries; the songs inside it are separate things, with
their own names and their own boundaries, and the two must never be
confused.

Where the inner songs are, and what they are called, come from **the words
the transcriber recovered inside the number**, compared with the songs a
reference list says can live inside it.

The obvious alternative — finding the places where the music changes — was
measured on a real medley and does not work there. The band plays
continuously across the songs of a medley, so the musical-change detector
found **no** internal boundary in a 287-second medley: one peak, at the very
end. The lyrics, on the other hand, change from one song to the next, and
that is the signal this module uses.

A name is only proposed where the words actually support it, and even then
it is a proposal and not an identification. The measurement behind that
caution is in the module's own tests: a title's words turn up in the lyric
of a *different* song often enough that a single agreeing line means
nothing. Two guards follow from it:

* the words of a title must be found across a **run** of consecutive
  stretches, not in one line;
* two titles that support the same stretch cancel each other out, because
  the evidence cannot tell them apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

#: Words that carry no meaning for a title match.
STOP_WORDS = frozenset(
    """a an and the of in on at to for with from is are was were be been
    it its this that these those my your his her our their you i we they
    he she them him us me do does did done not no so if then than as by
    into out up down over under again more most some such only own same
    too very can will just dont don't should would could""".split()
)

#: A title must have at least this many meaningful words to be matched.
MIN_TITLE_WORDS = 2

#: The share of a title's words that must appear before a name is proposed.
MIN_MATCH_RATIO = 0.6

#: The share that a proposal has to reach before it is offered at all. This
#: is stricter than the ratio above, because a title's words turn up in the
#: lyric of other songs: in a real medley, "What's Love Got to Do with It"
#: agreed with two words of a line from a different song.
MIN_MATCH_SCORE = 0.8

#: No name found this way is ever certain. The alignment has the same cap,
#: for the same reason: a duration or a word can support an identity but
#: never prove one.
MAX_IDENTITY_CONFIDENCE = 0.75

#: The shortest stretch of a medley that can hold a song, in seconds.
MIN_COMPONENT_SECONDS = 8.0

#: How many consecutive transcript stretches are considered together when a
#: title is matched. A song's words recur across its lines, so the words of a
#: title are looked for in a run of stretches rather than in a single line.
MATCH_RUN_STRETCHES = 4

#: A run must carry at least this many words before a name is proposed from
#: it. One short line that happens to contain the words of a title is not
#: evidence: the same phrase appears in other songs.
MIN_MATCH_WORDS = 6


@dataclass
class ComponentMatch:
    """A song found inside a number."""

    title: str
    start_frame: int
    end_frame: int
    confidence: float = 0.0
    basis: str = "reference"
    reason: str = ""


def significant_words(text: str) -> set[str]:
    """The words of a title that carry its meaning."""
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {word for word in words if word not in STOP_WORDS and len(word) > 1}


def match_title(text: str, title: str) -> float:
    """How well a stretch of words supports a title.

    The score is the share of the title's meaningful words that appear in
    the text. It is never high on its own: the transcriber is unreliable
    on singing, so this is evidence, not proof.
    """
    title_words = significant_words(title)
    if len(title_words) < MIN_TITLE_WORDS:
        return 0.0
    text_words = significant_words(text)
    if not text_words:
        return 0.0
    shared = title_words & text_words
    return len(shared) / float(len(title_words))


def match_in_runs(
    transcripts: Sequence[tuple[float, float, str]],
    title: str,
    max_run: int = MATCH_RUN_STRETCHES,
) -> tuple[float, float, float, str]:
    """Find where a title's words are, looking at runs of consecutive stretches.

    A song's words recur across its lines, so the words of a title are looked
    for in a run of consecutive stretches rather than in a single line. This
    matters: a single line can contain the words of a title by accident —
    "everlasting love" appears in the lyric of a song that is not called
    "Everlasting Love" — and a proposal built on one line is not evidence.

    The run must carry enough words for the agreement to mean something.
    A title is only proposed where a run of lines actually supports it.

    Returns ``(score, start, end, text)`` for the best run, with a score of
    zero when nothing supports the title.
    """
    best = (0.0, 0.0, 0.0, "")
    count = len(transcripts)
    for index in range(count):
        for length in range(1, max_run + 1):
            if index + length > count:
                break
            window = transcripts[index:index + length]
            if window[-1][1] - window[0][0] > 24.0:
                break
            text = " ".join(item[2] for item in window)
            words = significant_words(text)
            if len(words) < MIN_MATCH_WORDS and length < 2:
                continue
            raw_tokens = re.findall(r"[a-z0-9']+", text.lower())
            if length >= 2 and len(raw_tokens) < MIN_MATCH_WORDS:
                continue
            score = match_title(text, title)
            if score > best[0]:
                # Tighten boundaries to the sub-span containing matched words.
                title_sig = significant_words(title)
                hit_stretches = [
                    item for item in window
                    if significant_words(item[2]) & title_sig
                ]
                if hit_stretches:
                    best = (score, hit_stretches[0][0], hit_stretches[-1][1], text)
                else:
                    best = (score, window[0][0], window[-1][1], text)
    return best


def segment_components(
    transcripts: Sequence[tuple[float, float, str]],
    titles: Sequence[str],
    base_frame: int,
    sample_rate: int,
    min_score: float = MIN_MATCH_SCORE,
) -> tuple[list[ComponentMatch], list[str]]:
    """The songs inside a number that the recovered words can support.

    ``transcripts`` are the stretches of words the transcriber recovered
    inside the number, ``titles`` are the songs a reference says can live
    inside it, and ``base_frame`` is where the number starts in the
    timeline.

    Returns the located songs and the titles that could not be located.

    Three rules keep this from inventing anything:

    * a title needs a **run** of consecutive stretches that agree with it,
      not one agreeing line;
    * two titles that support the same stretch cancel each other out,
      because the evidence cannot tell them apart;
    * the confidence is the share of the title's words that were found, and
      it is capped well below certainty, because a title's words turn up in
      the lyric of other songs often enough to be a warning rather than a
      proof.
    """
    located: list[ComponentMatch] = []
    unlocated: list[str] = []
    if not transcripts or not titles:
        return located, list(titles)

    # Score every title first, and keep the raw score: the clash test below
    # has to compare how well the words agreed, not the capped confidence,
    # because two strong agreements are capped to the same number.
    scored = []
    for title in titles:
        score, start, end, text = match_in_runs(transcripts, title)
        scored.append((title, score, start, end, text))

    for title, score, start, end, text in scored:
        if score < min_score:
            unlocated.append(title)

    # Two titles that support the same stretch cannot both be right, and
    # the evidence cannot say which one is. The one the words agree with
    # better wins; a tie means neither can be preferred, so neither is
    # proposed.
    kept: list[ComponentMatch] = []
    for index, (title, score, start, end, text) in enumerate(scored):
        if score < min_score:
            continue
        clash = []
        for other_index, entry in enumerate(scored):
            other_title, other_score, other_start, other_end, _ = entry
            if other_index == index:
                continue
            if other_score < min_score:
                continue
            if other_start < end and start < other_end:
                clash.append(entry)
        if clash:
            if any(other[1] > score for other in clash):
                # Something else agrees with these words better, so this one
                # is not proposed.
                if title not in unlocated:
                    unlocated.append(title)
                continue
            if any(other[1] == score for other in clash):
                # A tie: neither can be preferred, so neither is proposed.
                if title not in unlocated:
                    unlocated.append(title)
                continue
        kept.append(ComponentMatch(
            title=title,
            start_frame=base_frame + int(start * sample_rate),
            end_frame=base_frame + int(end * sample_rate),
            confidence=min(score, MAX_IDENTITY_CONFIDENCE),
            basis="reference",
            reason=(
                f"{score * 100:.0f}% of the words in the title '{title}' "
                "appear in the words the transcriber recovered here. The "
                "transcriber is unreliable on singing, and a title's words "
                "turn up in other songs' lyrics as well, so this is a "
                "proposal and not an identification. Listen before you "
                "trust it."
            ),
        ))

    kept.sort(key=lambda m: m.start_frame)
    return kept, unlocated
