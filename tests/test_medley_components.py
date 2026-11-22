"""Tests for the songs inside a medley.

A medley is one number that holds several songs. The number keeps its own
title; the songs inside it are separate things, with their own names and
their own boundaries.

The measurement behind these tests is in the module's own docstring: the
musical-change detector finds *nothing* inside a real 287-second medley,
because the band plays straight through the songs in it. The signal that
does work is the lyrics, and even that has to be handled carefully — a
title's words turn up in the lyric of other songs often enough that one
agreeing line means nothing.
"""
from __future__ import annotations

from splytnyt.analysis.medley import (
    MAX_IDENTITY_CONFIDENCE,
    MIN_MATCH_SCORE,
    match_in_runs,
    match_title,
    segment_components,
)

SAMPLE_RATE = 48000


def _stretches(*lines):
    """Transcript stretches, relative to the start of the number."""
    out = []
    start = 0.0
    for text in lines:
        end = start + 2.0
        out.append((start, end, text))
        start = end
    return out


def test_a_title_is_matched_on_its_meaningful_words():
    score = match_title("a love song is here", "Love Song")
    assert score > 0.0
    assert match_title("", "Anything") == 0.0
    # Words that carry no meaning do not count for or against a title.
    assert match_title("a the of", "Love") == 0.0
    # A title needs at least two meaningful words before it is matched at all.
    assert match_title("lovers forever", "Lovers") == 0.0


def test_one_agreeing_line_is_not_enough():
    """A single line that contains the words of a title means nothing.

    "everlasting love" appears in the lyric of a song that is not called
    "Everlasting Love", so a proposal needs a run of lines.
    """
    one_line = _stretches("here I'll send you by everlasting love")
    score, start, end, text = match_in_runs(one_line, "Everlasting Love")
    assert score <= 0.0 or len(text.split()) < 6

    several = _stretches(
        "here I'll send you by everlasting love",
        "and I'll send you by everlasting love",
        "we are everlasting love",
    )
    score, start, end, text = match_in_runs(several, "Everlasting Love")
    assert score > 0.0
    assert end > start


def test_a_song_is_located_where_its_words_are():
    stretches = _stretches(
        "some other song entirely here",
        "we can be lovers, we can be lovers",
        "up where we belong, out of the darkness",
        "and another line that is not the one",
    )
    matches, unlocated = segment_components(
        stretches, ["Up Where We Belong"], 0, SAMPLE_RATE
    )
    assert len(matches) == 1
    match = matches[0]
    assert match.title == "Up Where We Belong"
    assert match.start_frame >= 0
    assert match.end_frame > match.start_frame
    # The position is the words that support the title, not the whole number.
    assert match.end_frame < 20 * SAMPLE_RATE
    assert not unlocated


def test_two_titles_that_agree_equally_cancel_each_other_out():
    """The evidence cannot tell them apart, so neither is proposed."""
    stretches = _stretches(
        "alpha beta gamma",
        "alpha beta delta",
        "alpha beta epsilon",
    )
    matches, unlocated = segment_components(
        stretches,
        ["Alpha Beta", "Beta Alpha"],
        0, SAMPLE_RATE,
    )
    assert matches == []
    assert set(unlocated) == {"Alpha Beta", "Beta Alpha"}


def test_the_better_supported_title_wins():
    """When one title agrees with the words more than the other, it wins."""
    stretches = _stretches(
        "I will always love you",
        "and I will always love you",
        "I will always love you for ever more",
    )
    matches, unlocated = segment_components(
        stretches,
        ["I Will Always Love You", "I Love You Always Forever"],
        0, SAMPLE_RATE,
    )
    assert [match.title for match in matches] == ["I Will Always Love You"]
    assert "I Love You Always Forever" in unlocated


def test_a_title_the_words_do_not_support_is_not_located():
    stretches = _stretches(
        "the words of something else entirely",
        "and more of the same thing here",
        "with nothing that matches the title",
    )
    matches, unlocated = segment_components(
        stretches, ["Take On Me"], 0, SAMPLE_RATE
    )
    assert matches == []
    assert unlocated == ["Take On Me"]


def test_a_name_is_never_certain():
    stretches = _stretches(
        "up where we belong, out of the darkness",
        "we are free and hot, up where we belong",
    )
    matches, _ = segment_components(
        stretches, ["Up Where We Belong"], 0, SAMPLE_RATE
    )
    assert matches
    assert matches[0].confidence <= MAX_IDENTITY_CONFIDENCE
    assert "not an identification" in matches[0].reason


def test_the_threshold_is_stricter_than_the_agreement_ratio():
    """A weak agreement is not proposed."""
    stretches = _stretches(
        "what is it all about to do",
        "what is it all about to send you hate emotion",
        "what is left to do, what is left to do",
    )
    # The title's words are "love" and "what"; only "what" is here.
    matches, unlocated = segment_components(
        stretches, ["What's Love Got to Do with It"], 0, SAMPLE_RATE
    )
    assert matches == []
    assert MIN_MATCH_SCORE > 0.6
