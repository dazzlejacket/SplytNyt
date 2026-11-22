"""Tracklist alignment (stage 3).

Stage 3 of the analysis pipeline: matching a candidate song list to the
segments that were detected in the audio.

Two rules shape this module:

* Durations are **approximate evidence**, not cutting instructions. A live
  cover lasting 4:50 against a 4:12 studio original is still a match; it is
  simply less certain.
* Nothing is forced. A song that is not in the recording stays unmatched, and
  a segment with no song stays unmatched. Both are reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

#: Score for pairing an item with a segment when neither duration is known.
UNKNOWN_DURATION_SCORE = 0.10
#: Penalty for skipping an item (the song is not in the recording).
SKIP_ITEM_PENALTY = -0.65
#: Penalty for skipping a segment (extra audio that is not on the list).
SKIP_SEGMENT_PENALTY = -0.65


@dataclass
class Alignment:
    """The result of aligning a worklist to detected segments."""

    #: ``(worklist_index, segment_index)`` pairs.
    assignments: list[tuple[int, int]] = field(default_factory=list)
    unmatched_items: list[int] = field(default_factory=list)
    unmatched_segments: list[int] = field(default_factory=list)
    #: Identity confidence per assignment, in the same order as ``assignments``.
    identity_confidence: list[float] = field(default_factory=list)
    #: Human-readable evidence summary keyed by ``(worklist_index, segment_index)``.
    evidence: dict[tuple[int, int], str] = field(default_factory=dict)


def duration_score(item_seconds: float, segment_seconds: float) -> float:
    """How well two durations agree, in 0..1."""
    if item_seconds <= 0 or segment_seconds <= 0:
        return UNKNOWN_DURATION_SCORE
    difference = abs(item_seconds - segment_seconds)
    scale = max(item_seconds, segment_seconds, 1.0)
    relative = difference / scale
    # 10 % difference still scores well; 50 % difference scores poorly.
    score = max(0.0, 1.0 - (relative / 0.5))
    return score


def align_worklist(
    items: Sequence,
    segments: Sequence[tuple[int, int]],
    sample_rate: int,
    duration_scorer=None,
    segment_prior: Optional[Sequence[float]] = None,
    skip_item_penalty: float = SKIP_ITEM_PENALTY,
    skip_segment_penalty: float = SKIP_SEGMENT_PENALTY,
) -> Alignment:
    """Align ``items`` to ``segments`` using a global dynamic program.

    ``items`` are objects with ``duration_hint`` (seconds) attributes, or
    plain numbers. ``segments`` are ``(start_frame, end_frame)`` pairs.

    ``duration_scorer`` and ``segment_prior`` exist so that a different
    kind of recording can be aligned without changing the algorithm: the
    theatrical alignment uses a duration comparison that understands that
    a number runs longer on stage than on a record, and it uses the
    detected section type as a prior. With the defaults the behaviour is
    exactly what it always was.
    """
    scorer = duration_scorer or duration_score
    item_count = len(items)
    segment_count = len(segments)
    alignment = Alignment()
    if item_count == 0 or segment_count == 0:
        alignment.unmatched_items = list(range(item_count))
        alignment.unmatched_segments = list(range(segment_count))
        return alignment

    durations: list[float] = []
    for item in items:
        value = getattr(item, "duration_hint", item)
        try:
            durations.append(float(value or 0.0))
        except (TypeError, ValueError):
            durations.append(0.0)

    segment_durations = [
        max(0, end - start) / float(sample_rate or 44100) for start, end in segments
    ]

    priors = list(segment_prior) if segment_prior else [0.0] * segment_count

    # dp[i][j] = best score for items[:i] against segments[:j]
    dp = [[0.0] * (segment_count + 1) for _ in range(item_count + 1)]
    back = [[None] * (segment_count + 1) for _ in range(item_count + 1)]
    for i in range(1, item_count + 1):
        dp[i][0] = dp[i - 1][0] + skip_item_penalty
        back[i][0] = "item"
    for j in range(1, segment_count + 1):
        dp[0][j] = dp[0][j - 1] + skip_segment_penalty
        back[0][j] = "segment"

    for i in range(1, item_count + 1):
        for j in range(1, segment_count + 1):
            match = (
                dp[i - 1][j - 1]
                + scorer(durations[i - 1], segment_durations[j - 1])
                + priors[j - 1]
            )
            skip_item = dp[i - 1][j] + skip_item_penalty
            skip_segment = dp[i][j - 1] + skip_segment_penalty
            best = max(match, skip_item, skip_segment)
            dp[i][j] = best
            if best == match:
                back[i][j] = "match"
            elif best == skip_item:
                back[i][j] = "item"
            else:
                back[i][j] = "segment"

    # Backtrack.
    assignments: list[tuple[int, int]] = []
    confidences: list[float] = []
    matched_items: set[int] = set()
    matched_segments: set[int] = set()

    i, j = item_count, segment_count
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "match":
            score = scorer(durations[i - 1], segment_durations[j - 1])
            assignments.append((i - 1, j - 1))
            confidences.append(
                _confidence(score, durations[i - 1], segment_durations[j - 1])
            )
            matched_items.add(i - 1)
            matched_segments.add(j - 1)
            i -= 1
            j -= 1
        elif move == "item":
            i -= 1
        elif move == "segment":
            j -= 1
        else:  # pragma: no cover - defensive
            break

    assignments.reverse()
    confidences.reverse()

    alignment.assignments = assignments
    alignment.identity_confidence = confidences
    alignment.unmatched_items = [index for index in range(item_count) if index not in matched_items]
    alignment.unmatched_segments = [
        index for index in range(segment_count) if index not in matched_segments
    ]
    return alignment


# ----------------------------------------------------------------------
# Theatrical alignment
# ----------------------------------------------------------------------

#: How much the detected section type moves a pairing up or down.
SECTION_PRIORS = {
    "musical_number": 0.55,
    "mixed": 0.20,
    "music": 0.15,
    "dialogue": -0.05,
    "underscoring": -0.15,
    "applause": -0.45,
    "silence": -0.35,
    "unknown": 0.0,
    "uncertain": -0.20,
}

#: How much longer than the reference a number may run on stage before the
#: duration stops agreeing at all. A number is staged slower than it is
#: recorded, so a stage rendering is normally longer than the studio
#: original and that is not a disagreement.
STAGE_OVERRUN_TOLERANCE = 1.5

#: How much shorter than the reference a number may be.
STAGE_UNDERRUN_TOLERANCE = 0.5


def stage_duration_score(item_seconds: float, segment_seconds: float) -> float:
    """How well two durations agree when the recording is a stage show.

    A number runs longer on stage than it does on a record, because the
    staging, the applause and the pacing all take time. A stage rendering
    that is longer than the reference is therefore *not* a disagreement,
    while one that is much shorter is.
    """
    if item_seconds <= 0 or segment_seconds <= 0:
        return UNKNOWN_DURATION_SCORE
    if segment_seconds >= item_seconds:
        overrun = segment_seconds / item_seconds - 1.0
        return max(0.0, 1.0 - overrun / STAGE_OVERRUN_TOLERANCE)
    shortfall = item_seconds / segment_seconds - 1.0
    return max(0.0, 1.0 - shortfall / STAGE_UNDERRUN_TOLERANCE)


#: Minimum independent lexical evidence required before proposing a title.
#: Order and duration alone (score 0.0) never reach this floor.
MIN_THEATER_LEXICAL_EVIDENCE = 0.35

#: Extra common words that should never carry a lexical match on their own.
_ALIGN_STOP_WORDS = frozenset(
    """a an and the of in on at to for with from is are was were be been
    it its this that these those my your his her our their you i we they
    he she them him us me do does did done not no so if then than as by
    into out up down over under again more most some such only own same
    too very can will just dont cant wont thats youre im ive youve weve
    what when where who why how which shall should would could want need
    like know got get make made take give come go going gonna wanna let
    one two now ever never thing things way see say said right here there
    oh yeah well hey""".split()
)

_BOILERPLATE_SUBTITLES = frozenset(
    {"reprise", "encore", "finale", "intro", "outro", "overture", "part 1", "part 2"}
)


def _norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower().replace("'", ""))


def _expand_title_variants(title: str) -> list[str]:
    """Return searchable title strings, including parenthetical subtitles."""
    clean = (title or "").strip()
    if not clean:
        return []
    variants = [clean]
    no_parens = re.sub(r"\([^)]*\)", "", clean).strip()
    if no_parens and no_parens != clean:
        variants.append(no_parens)
    for inside in re.findall(r"\(([^)]*)\)", clean):
        part = inside.strip()
        if part and part.lower() not in _BOILERPLATE_SUBTITLES:
            part_non_stop = [w for w in _norm_tokens(part) if w not in _ALIGN_STOP_WORDS]
            if len(part_non_stop) >= 2:
                variants.append(part)
    seen: set[str] = set()
    unique: list[str] = []
    for item in variants:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _count_phrase_hits(
    seg_token_lists: Sequence[list[str]],
    phrase_tokens: list[str],
    allow_single_word: bool,
) -> tuple[int, str]:
    """Count distinct transcript stretches containing a non-trivial phrase match."""
    if not phrase_tokens:
        return 0, ""
    if len(phrase_tokens) == 1:
        word = phrase_tokens[0]
        if (
            not allow_single_word
            or len(word) < 6
            or word in _ALIGN_STOP_WORDS
        ):
            return 0, ""
        hits = sum(1 for toks in seg_token_lists if word in toks)
        if hits >= 2 or (hits >= 1 and len(word) >= 8):
            return hits, word
        return 0, ""

    non_stop = [w for w in phrase_tokens if w not in _ALIGN_STOP_WORDS]
    if not non_stop:
        return 0, ""

    # Match full phrase consecutively (plus article-stripped variant and
    # distinctive >=2-word content slices for multi-word titles).
    slice_len = len(phrase_tokens)
    patterns: list[list[str]] = [phrase_tokens]
    stripped = [
        w for i, w in enumerate(phrase_tokens)
        if not (i == 0 and w in ("the", "a", "an"))
    ]
    if len(stripped) >= 2 and stripped != phrase_tokens:
        patterns.append(stripped)
    if slice_len >= 4:
        for width in range(slice_len - 1, 1, -1):
            for offset in range(slice_len - width + 1):
                sub = phrase_tokens[offset: offset + width]
                sub_non_stop = [w for w in sub if w not in _ALIGN_STOP_WORDS]
                if width >= 4 and len(sub_non_stop) >= 1:
                    patterns.append(sub)
                elif len(sub_non_stop) >= 2:
                    patterns.append(sub)
    elif slice_len == 3:
        for offset in range(2):
            sub = phrase_tokens[offset: offset + 2]
            if all(w not in _ALIGN_STOP_WORDS and len(w) >= 4 for w in sub):
                patterns.append(sub)

    matched_indices: set[int] = set()
    matched_text = ""
    for pat in patterns:
        plen = len(pat)
        pat_non_stop = [w for w in pat if w not in _ALIGN_STOP_WORDS]
        # Skip partial multi-word patterns supported by only a single generic word.
        if plen < 4 and len(pat_non_stop) < 2 and pat != phrase_tokens:
            continue
        pat_hits = []
        for idx, toks in enumerate(seg_token_lists):
            if len(toks) < plen:
                continue
            for pos in range(len(toks) - plen + 1):
                if toks[pos: pos + plen] == pat:
                    pat_hits.append(idx)
                    break
        # For 2-token phrases where only one token is non-stop (e.g. "your song"),
        # require at least 2 occurrences so stray lines do not trigger.
        if plen == 2 and len(pat_non_stop) < 2 and len(pat_hits) < 2:
            continue
        # Require >=2 non-stop words on partial slices, or a full 3-word
        # verbatim title where every word is at least 3 letters.
        if plen == 3 and len(pat_non_stop) < 2:
            is_full_substantial = (pat == phrase_tokens and all(len(w) >= 3 for w in pat))
            if not is_full_substantial and not any(len(w) >= 5 for w in pat_non_stop) and len(pat_hits) < 2:
                continue
        if pat_hits:
            matched_indices.update(pat_hits)
            if not matched_text:
                matched_text = " ".join(pat)
            if plen == slice_len:
                break
    return len(matched_indices), matched_text


def _lyric_phrases(lyrics_text: str, max_phrases: int = 50) -> list[list[str]]:
    """Extract distinctive 3-4 word content phrases from reference lyrics."""
    if not lyrics_text:
        return []
    seen: set[tuple[str, ...]] = set()
    phrases: list[list[str]] = []
    for raw_line in lyrics_text.splitlines():
        toks = _norm_tokens(raw_line)
        if len(toks) < 3:
            continue
        for width in (3, 4):
            if len(toks) < width:
                continue
            for pos in range(len(toks) - width + 1):
                win = toks[pos: pos + width]
                non_stop = [w for w in win if w not in _ALIGN_STOP_WORDS and len(w) >= 3]
                if len(non_stop) < 2:
                    continue
                key = tuple(win)
                if key in seen:
                    continue
                seen.add(key)
                phrases.append(win)
                if len(phrases) >= max_phrases:
                    return phrases
    return phrases


def lexical_item_evidence(
    item,
    section_transcripts: Sequence[tuple[float, float, str]],
    reference_lyrics: Optional[str] = None,
) -> tuple[float, list[str]]:
    """Measure independent transcript evidence supporting ``item`` in one section.

    Returns ``(score in 0..1, hit descriptions)``. Only concrete title,
    contained-song, or reference-lyric phrase matches qualify; repeated weak
    stop-words or single short nouns do not.
    """
    if not section_transcripts:
        return 0.0, []

    seg_token_lists = [_norm_tokens(text) for _, _, text in section_transcripts]
    seg_token_lists = [toks for toks in seg_token_lists if toks]
    if not seg_token_lists:
        return 0.0, []

    best_primary = 0.0
    hits: list[str] = []
    title = getattr(item, "title", "") or ""
    for variant in _expand_title_variants(title):
        vtoks = _norm_tokens(variant)
        count, matched = _count_phrase_hits(
            seg_token_lists, vtoks, allow_single_word=True
        )
        if count <= 0:
            continue
        if len(vtoks) >= 3:
            score = 0.68 if count >= 2 else 0.58
        elif len(vtoks) == 2:
            distinctive = len([w for w in vtoks if w not in _ALIGN_STOP_WORDS])
            score = 0.62 if distinctive == 2 else 0.48
        else:
            score = 0.50 if count >= 2 else 0.42
        if score > best_primary:
            best_primary = score
            hits = [f"title phrase '{matched}' ({count}x)"]

    contained_hits: list[str] = []
    for sub in getattr(item, "contains", []) or []:
        best_sub = 0
        best_text = ""
        for variant in _expand_title_variants(sub):
            vtoks = _norm_tokens(variant)
            # Only distinctive single words (>= 6 chars, >= 2 occurrences) count for contained songs.
            allow_single = len(vtoks) == 1 and len(vtoks[0]) >= 6
            count, matched = _count_phrase_hits(
                seg_token_lists, vtoks, allow_single_word=allow_single
            )
            if allow_single and count < 2:
                continue
            if count > best_sub:
                best_sub = count
                best_text = matched
        if best_sub > 0:
            contained_hits.append(f"'{sub}' ('{best_text}')")

    contained_score = 0.0
    if len(contained_hits) >= 3:
        contained_score = 0.78
    elif len(contained_hits) == 2:
        contained_score = 0.66
    elif len(contained_hits) == 1:
        contained_score = 0.50

    # Optional reference-lyric phrase matching (e.g. from LRCLIB)
    lyric_hits: list[str] = []
    lyric_score = 0.0
    if reference_lyrics:
        # Join section tokens for fast 3- and 4-gram lookup across short Whisper splits
        flat_toks = [w for toks in seg_token_lists for w in toks]
        flat_ngrams = {
            tuple(flat_toks[p: p + width])
            for width in (3, 4)
            for p in range(max(0, len(flat_toks) - width + 1))
        }
        for phrase in _lyric_phrases(reference_lyrics):
            if tuple(phrase) in flat_ngrams:
                lyric_hits.append("'" + " ".join(phrase) + "'")
                if len(lyric_hits) >= 4:
                    break
        if len(lyric_hits) >= 3:
            lyric_score = 0.76
        elif len(lyric_hits) == 2:
            lyric_score = 0.62
        elif len(lyric_hits) == 1:
            lyric_score = 0.44

    combined = max(best_primary, contained_score, lyric_score)
    sources_agreeing = sum(1 for s in (best_primary, contained_score, lyric_score) if s > 0.0)
    if sources_agreeing >= 2:
        combined = min(0.88, combined + 0.08)
    all_hits = hits + contained_hits[:3] + ([f"lyrics {', '.join(lyric_hits[:3])}"] if lyric_hits else [])
    return combined, all_hits


def align_theater_worklist(
    items: Sequence,
    sections: Sequence,
    sample_rate: int,
    transcripts: Optional[Sequence[tuple[float, float, str]]] = None,
    reference_lyrics: Optional[dict[str, str]] = None,
) -> Alignment:
    """Align a reference list of numbers to detected sections.

    Programme order and durations are supporting evidence only: a pairing
    requires independent transcript evidence (title, contained-song, or
    reference-lyric hits inside the section) meeting
    ``MIN_THEATER_LEXICAL_EVIDENCE``. Without transcript evidence, sections
    remain unconfirmed rather than guessed.
    """
    item_count = len(items)
    segment_count = len(sections)
    alignment = Alignment()
    if item_count == 0 or segment_count == 0:
        alignment.unmatched_items = list(range(item_count))
        alignment.unmatched_segments = list(range(segment_count))
        return alignment

    sr = float(sample_rate or 44100)
    transcripts = list(transcripts or [])
    lyrics_map = reference_lyrics or {}
    seg_durations = [
        max(0, section.end_frame - section.start_frame) / sr for section in sections
    ]

    # Precompute transcript stretches overlapping each candidate section.
    per_section_text: list[list[tuple[float, float, str]]] = []
    for section in sections:
        st = section.start_frame / sr
        en = section.end_frame / sr
        per_section_text.append(
            [seg for seg in transcripts if seg[1] > st and seg[0] < en]
        )

    # Precompute independent lexical evidence matrix; unrelated segments stay
    # strictly ineligible so a small transcript edit cannot reshuffle the show.
    pair_score = [[-1e6] * segment_count for _ in range(item_count)]
    pair_lex = [[0.0] * segment_count for _ in range(item_count)]
    pair_hits: dict[tuple[int, int], list[str]] = {}
    for i, item in enumerate(items):
        hint = float(getattr(item, "duration_hint", 0.0) or 0.0)
        item_lyrics = lyrics_map.get(getattr(item, "title", "") or "", None)
        for j, section in enumerate(sections):
            lex, hits = lexical_item_evidence(item, per_section_text[j], reference_lyrics=item_lyrics)
            if lex < MIN_THEATER_LEXICAL_EVIDENCE:
                continue
            dur_fit = stage_duration_score(hint, seg_durations[j]) if hint > 0 else 0.0
            prior = max(0.0, SECTION_PRIORS.get(getattr(section, "section_type", "unknown"), 0.0))
            pair_lex[i][j] = lex
            pair_score[i][j] = lex + 0.15 * dur_fit + 0.05 * prior
            pair_hits[(i, j)] = hits

    # Global monotonic DP with zero skip penalties so skipping unconfirmed
    # items or unmatched sections never forces unrelated pairings.
    dp = [[0.0] * (segment_count + 1) for _ in range(item_count + 1)]
    back: list[list[Optional[str]]] = [[None] * (segment_count + 1) for _ in range(item_count + 1)]
    for i in range(1, item_count + 1):
        back[i][0] = "item"
    for j in range(1, segment_count + 1):
        back[0][j] = "segment"

    for i in range(1, item_count + 1):
        for j in range(1, segment_count + 1):
            skip_item = dp[i - 1][j]
            skip_segment = dp[i][j - 1]
            cand = pair_score[i - 1][j - 1]
            if cand > 0.0:
                match = dp[i - 1][j - 1] + cand
                if match >= skip_item and match >= skip_segment:
                    dp[i][j] = match
                    back[i][j] = "match"
                    continue
            if skip_item >= skip_segment:
                dp[i][j] = skip_item
                back[i][j] = "item"
            else:
                dp[i][j] = skip_segment
                back[i][j] = "segment"

    assignments: list[tuple[int, int]] = []
    confidences: list[float] = []
    matched_items: set[int] = set()
    matched_segments: set[int] = set()

    i, j = item_count, segment_count
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "match":
            ii, jj = i - 1, j - 1
            lex = pair_lex[ii][jj]
            hint = float(getattr(items[ii], "duration_hint", 0.0) or 0.0)
            dur_fit = stage_duration_score(hint, seg_durations[jj]) if hint > 0 else 0.0
            conf = min(0.75, 0.30 + 0.40 * lex + 0.15 * dur_fit)
            if hint > 0 and seg_durations[jj] < 0.55 * hint:
                conf = min(conf, 0.50)
            assignments.append((ii, jj))
            confidences.append(round(conf, 3))
            matched_items.add(ii)
            matched_segments.add(jj)
            alignment.evidence[(ii, jj)] = (
                "Transcript evidence: " + ", ".join(pair_hits.get((ii, jj), []))
            )
            i -= 1
            j -= 1
        elif move == "item":
            i -= 1
        elif move == "segment":
            j -= 1
        else:  # pragma: no cover - defensive
            break

    assignments.reverse()
    confidences.reverse()

    alignment.assignments = assignments
    alignment.identity_confidence = confidences
    alignment.unmatched_items = [k for k in range(item_count) if k not in matched_items]
    alignment.unmatched_segments = [k for k in range(segment_count) if k not in matched_segments]
    return alignment


def _confidence(score: float, item_seconds: float, segment_seconds: float) -> float:
    """Turn a duration agreement into an identity confidence.

    Without a duration hint the alignment is a guess about ordering only, so
    its confidence is capped low. A good duration agreement is capped below
    certainty because durations never prove an identity.
    """
    if item_seconds <= 0 or segment_seconds <= 0:
        return min(0.45, 0.25 + 0.3 * score)
    return min(0.9, 0.35 + 0.55 * score)
