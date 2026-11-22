"""Lyric anchors locate candidates; neither ASR nor alignment verifies identity.

No downloads, title matching, duration matching, or timeline mutations here.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np


_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "is", "am", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "so", "if", "as", "do", "did", "all", "just", "no", "not", "up", "out",
    "from", "into", "get", "got", "oh", "ah", "yeah",
})


def tokens(text):
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold().replace('’', "'"), re.UNICODE)


def _token_match_score(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if len(a) >= 4 and len(b) >= 4:
        # Handle minor ASR inflection / phonetic drift (e.g. burning/burn, birds/burdened)
        if a.startswith(b) or b.startswith(a):
            if min(len(a), len(b)) / max(len(a), len(b)) >= 0.55:
                return 0.85
        ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
        if ratio >= 0.72:
            return ratio
    return 0.0


def _window_similarity(query: list[str], candidate: list[str]) -> tuple[float, int, int]:
    """Order-preserving fuzzy token similarity with content-word support.

    Returns ``(score, matched_tokens, matched_content_words)``.
    """
    if not query or not candidate:
        return 0.0, 0, 0
    # Exact token SequenceMatcher baseline
    exact_score = SequenceMatcher(None, query, candidate, autojunk=False).ratio()
    # Greedy order-preserving fuzzy alignment for ASR drift tolerance
    c_idx = 0
    fuzzy_sum = 0.0
    matched_tokens = 0
    matched_content: set[str] = set()
    q_content = {t for t in query if t not in _STOPWORDS and len(t) >= 3}
    for q in query:
        best_j = -1
        best_s = 0.0
        # Allow small local skip in candidate window
        for j in range(c_idx, min(len(candidate), c_idx + 4)):
            s = _token_match_score(q, candidate[j])
            if s > best_s:
                best_s = s
                best_j = j
                if s >= 1.0:
                    break
        if best_j >= 0 and best_s > 0.0:
            fuzzy_sum += best_s
            matched_tokens += 1
            if q in q_content:
                matched_content.add(q)
            c_idx = best_j + 1
    fuzzy_score = (2.0 * fuzzy_sum) / (len(query) + len(candidate))
    return max(exact_score, fuzzy_score), matched_tokens, len(matched_content)


def parse_references(text):
    """JSON manifest or pasted blocks: # Title / Source: ... / Version: ... ."""
    if text.lstrip().startswith('{'):
        data = json.loads(text)
        if data.get('schema') != 'splytnyt-lyrics/1':
            raise ValueError('Expected schema splytnyt-lyrics/1')
        refs = data.get('references', [])
    else:
        refs, current = [], None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('# '):
                current = dict(title=line[2:].strip(), provenance='', version='', lyrics=[])
                refs.append(current)
            elif current is not None and line.lower().startswith('source:'):
                current['provenance'] = line[7:].strip()
            elif current is not None and line.lower().startswith('version:'):
                current['version'] = line[8:].strip()
            elif line and current is not None and not line.startswith('['):
                current['lyrics'].append(line)
        for ref in refs:
            ref['lyrics'] = '\n'.join(ref['lyrics'])
    if not isinstance(refs, list) or not refs or len(refs) > 200:
        raise ValueError('Supply 1–200 references with titles and provenance.')
    for ref in refs:
        if not isinstance(ref, dict) or not ref.get('title', '').strip() or not ref.get('provenance', '').strip():
            raise ValueError('Every reference needs a title and provenance (source URL or local source description).')
        if not ref.get('lyrics') and not ref.get('anchors'):
            raise ValueError('Every reference needs lyrics or short anchors.')
        if ref.get('lyrics') and not isinstance(ref['lyrics'], str):
            raise ValueError('Lyrics must be plain text.')
        if ref.get('anchors') and (not isinstance(ref['anchors'], list) or
                any(not isinstance(a,dict) or not isinstance(a.get('text'),str)
                    or a.get('role','unknown') not in ('opening','interior','ending','unknown') for a in ref['anchors'])):
            raise ValueError('Anchors need text and an opening, interior, ending or unknown role.')
        ref.pop('reference_id', None)
        ref['reference_id'] = hashlib.sha256(json.dumps(ref, sort_keys=True).encode()).hexdigest()[:16]
    return refs


def transcript_words(segments):
    result = []
    for seg_index, segment in enumerate(segments):
        if not segment.get('accepted', True):
            continue
        seg_text = str(segment.get('text', '') or '')
        seg_start = float(segment['start'])
        seg_end = float(segment['end'])
        words = segment.get('words') or []
        if words:
            for word in words:
                for token in tokens(word.get('word', word.get('text', ''))):
                    result.append(dict(
                        token=token, start=float(word['start']), end=float(word['end']),
                        timing='word', seg_index=seg_index, seg_text=seg_text,
                        seg_start=seg_start, seg_end=seg_end,
                    ))
        else:
            # Segment precision is retained. Never invent word timestamps.
            for token in tokens(seg_text):
                result.append(dict(
                    token=token, start=seg_start, end=seg_end,
                    timing='segment', seg_index=seg_index, seg_text=seg_text,
                    seg_start=seg_start, seg_end=seg_end,
                ))
    return sorted(result, key=lambda w: (w['start'], w['end']))


def reference_anchors(ref):
    if ref.get('anchors'):
        return [dict(text=a['text'], role=a.get('role', 'unknown'), order=i)
                for i, a in enumerate(ref['anchors']) if len(tokens(a['text'])) >= 3]
    lines = [line.strip() for line in ref['lyrics'].splitlines() if len(tokens(line)) >= 4 and not line.lstrip().startswith('[')]
    result = []
    for i, line in enumerate(lines):
        role = 'opening' if i < max(1, len(lines)/3) else 'ending' if i >= len(lines)*2/3 else 'interior'
        # Long lines become local phrases, so omissions do not poison a whole verse.
        ts = tokens(line)
        for offset in range(0, len(ts), 8):
            chunk = ts[offset:offset+8]
            if len(chunk) >= 4:
                result.append(dict(text=' '.join(chunk), role=role, order=len(result)))
    return result


def match_references(refs, segments, *, max_gap=240., max_span=1200., cancel=lambda: None):
    words = transcript_words(segments)
    corpus = [w['token'] for w in words]
    index = {}
    for i, token in enumerate(corpus):
        index.setdefault(token, []).append(i)
    all_anchors = [reference_anchors(ref) for ref in refs]
    counts = Counter(tuple(tokens(a['text'])) for anchors in all_anchors for a in anchors)

    # First pass: collect candidate anchor hits per reference so we can guard
    # against chaining across another song's distinctive anchors.
    ref_hits = []
    for ref, anchors in zip(refs, all_anchors):
        cancel()
        hits = []
        for anchor in anchors:
            query = tokens(anchor['text'])
            if len(set(query)) < 3:
                continue
            q_content = [t for t in set(query) if t not in _STOPWORDS and len(t) >= 3]
            min_content = 2 if len(q_content) >= 2 else len(q_content)
            # Rare query words seed windows (preferring content words).
            seed_pool = sorted(
                set(query),
                key=lambda t: (t in _STOPWORDS, len(index.get(t, []))),
            )
            seeds = seed_pool[:min(4, len(seed_pool))]
            starts = set()
            for seed in seeds:
                for pos in index.get(seed, []):
                    for offset, token in enumerate(query):
                        if token == seed:
                            starts.update(range(max(0, pos - offset - 2), min(len(words), pos - offset + 3)))
            local = []
            for start in sorted(starts):
                for length in range(max(3, len(query) - 2), len(query) + 3):
                    end = start + length
                    if end > len(words) or words[end - 1]['end'] - words[start]['start'] > 35:
                        continue
                    # Do not make a phrase out of words separated by a scene break.
                    if any(words[j + 1]['start'] - words[j]['end'] > 5 for j in range(start, end - 1)):
                        continue
                    candidate = corpus[start:end]
                    score, matched_n, matched_content = _window_similarity(query, candidate)
                    if score < 0.74 or matched_n < 3 or matched_content < min_content:
                        continue
                    # Gather local segment context so forced alignment has surrounding words.
                    seen_segs = []
                    seg_starts = []
                    seg_ends = []
                    for w in words[start:end]:
                        if w['seg_index'] not in seen_segs:
                            seen_segs.append(w['seg_index'])
                            seg_starts.append(w['seg_start'])
                            seg_ends.append(w['seg_end'])
                    context_parts = []
                    for si in seen_segs:
                        if 0 <= si < len(segments):
                            context_parts.append(str(segments[si].get('text', '') or '').strip())
                    context_text = ' '.join(p for p in context_parts if p)
                    local.append(dict(
                        **anchor,
                        start=words[start]['start'],
                        end=words[end - 1]['end'],
                        score=round(score, 3),
                        transcript=' '.join(candidate),
                        context_text=context_text,
                        context_start=min(seg_starts) if seg_starts else words[start]['start'],
                        context_end=max(seg_ends) if seg_ends else words[end - 1]['end'],
                        distinctive=counts[tuple(query)] == 1,
                        timing='word' if all(w['timing'] == 'word' for w in words[start:end]) else 'segment',
                    ))
            # One hit per phrase occurrence, preserving later choruses/reprises.
            chosen = []
            for hit in sorted(local, key=lambda h: -h['score']):
                if not any(hit['start'] < h['end'] and hit['end'] > h['start'] for h in chosen):
                    chosen.append(hit)
            hits.extend(chosen[:40])
        ref_hits.append(hits)

    # Distinctive intervals belonging to each reference (for cross-song separation).
    distinctive_by_ref = [
        [(h['start'], h['end']) for h in hits if h['distinctive'] and h['score'] >= 0.78]
        for hits in ref_hits
    ]

    results = []
    for ref_idx, (ref, hits) in enumerate(zip(refs, ref_hits)):
        cancel()
        other_intervals = [
            span
            for other_idx, spans in enumerate(distinctive_by_ref)
            if other_idx != ref_idx
            for span in spans
        ]

        def _blocked_by_other_song(a_end: float, b_start: float) -> bool:
            if b_start <= a_end:
                return False
            for o_start, o_end in other_intervals:
                if o_start >= a_end and o_end <= b_start:
                    return True
            return False

        # Beam of ordered chains. Reference order cannot run backward across choruses
        # and cannot chain across another song's distinctive anchors.
        chains = []
        for hit in sorted(hits, key=lambda h: (h['start'], h['order'])):
            additions = [[hit]]
            for chain in chains:
                prev = chain[-1]
                if (
                    hit['order'] > prev['order']
                    and hit['start'] >= prev['end']
                    and hit['start'] - prev['end'] <= max_gap
                    and hit['end'] - chain[0]['start'] <= max_span
                    and not _blocked_by_other_song(prev['end'], hit['start'])
                ):
                    additions.append(chain + [hit])
            chains.extend(additions)
            chains = sorted(
                chains,
                key=lambda c: (len({h['order'] for h in c}), sum(h['score'] for h in c)),
                reverse=True,
            )[:300]
        occurrences = []
        for chain in sorted(chains, key=lambda c: (len(c), sum(h['score'] for h in c)), reverse=True):
            start, end = chain[0]['start'], chain[-1]['end']
            if any(start < c['end'] and end > c['start'] for c in occurrences):
                continue
            roles = {h['role'] for h in chain if h['distinctive']}
            sufficient = {'opening', 'interior', 'ending'} <= roles and len(chain) >= 3
            gaps = [
                dict(
                    start=a['end'],
                    end=b['start'],
                    reason='Dialogue, pause, instrumental bridge, omitted lyrics or ASR failure inside proposed number',
                )
                for a, b in zip(chain, chain[1:])
                if b['start'] - a['end'] > 10
            ]
            occurrences.append(dict(
                start=start,
                end=end,
                anchors=chain,
                gaps=gaps,
                status='multi-anchor proposal' if sufficient else 'partial proposal',
                verified=False,
                coverage=sorted(roles),
                missing_roles=sorted({'opening', 'interior', 'ending'} - roles),
                alignment=dict(status='not run'),
                boundaries=None,
            ))
        results.append(dict(
            reference_id=ref['reference_id'],
            title=ref['title'],
            provenance=ref['provenance'],
            version=ref.get('version', 'unknown'),
            scope=ref.get('scope', 'User supplied lyrics'),
            occurrences=sorted(occurrences, key=lambda c: c['start']),
            unmatched=not occurrences,
        ))
    return dict(
        schema='splytnyt-lyric-review/1',
        results=results,
        limits=dict(max_gap_seconds=max_gap, max_span_seconds=max_span),
        verified_matches=0,
        note='Text similarity is evidence only; all identities and boundaries require listening.',
    )


def evidence_signature(project):
    data = [(s.source_id,s.path,s.frames,s.sample_rate,s.size_bytes,s.mtime_ns,s.head_hash,s.gap_frames)
            for s in project.sources]
    data.append(project.cache.get('theater_transcript',{}).get('fingerprint',''))
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def propose_audio_extent(reader, start, end):
    """Bounded energy envelope, not a classifier of final chords or scene identity."""
    rate = reader.sample_rate
    lo, hi = max(0,int((start-20)*rate)), min(reader.total_frames,int((end+30)*rate))
    if hi <= lo or not (0 <= start < end <= reader.total_frames/rate):
        return dict(status='unavailable',verified=False,start_frame=0,end_frame=0,
                    intro_unresolved=True,tail_unresolved=True,
                    reason='Candidate timing falls outside the source audio; no extent proposed.')
    block = max(1, round(rate*.1))
    levels = []
    for pos in range(lo,hi,block):
        x = reader.read(pos,min(block,hi-pos))
        levels.append(float(np.sqrt(np.mean(x.astype(np.float64)**2))) if len(x) else 0)
    levels = np.asarray(levels)
    floor = max(1e-5, float(np.percentile(levels,10))*2)
    first = max(0,min(len(levels)-1,int((start*rate-lo)/block)))
    last = max(first,min(len(levels)-1,int((end*rate-lo)/block)))
    # Require sustained quiet; a brief beat/rest does not cut an intro or decay.
    quiet = [i for i in range(max(0,len(levels)-4)) if np.all(levels[i:i+5] < floor)]
    before = [i+5 for i in quiet if i+5 <= first]
    after = [i for i in quiet if i >= last]
    left = max(before) if before else 0
    right = min(after) if after else len(levels)
    return dict(start_frame=min(math.floor(start*rate),max(0,lo+left*block)),
                end_frame=max(math.ceil(end*rate),min(hi,lo+right*block)),
                status='audio-envelope proposal',verified=False, threshold_rms=floor,
                intro_unresolved=not bool(before),tail_unresolved=not bool(after),
                reason='Local PCM energy includes possible intro/chord/tail; speech and applause can also carry energy. Listen before applying.')


def _extract_anchor_words(raw_words: list[dict], query_text: str) -> list[dict]:
    """Locate the contiguous aligned-word span best matching ``query_text``."""
    expected = tokens(query_text)
    if not raw_words or not expected or len(raw_words) <= len(expected) + 1:
        return raw_words
    word_tokens = []
    for w in raw_words:
        ts = tokens(str(w.get('word', w.get('text', '')) or ''))
        word_tokens.append(ts[0] if ts else '')
    best_score = -1.0
    best_slice = raw_words
    n = len(expected)
    for start in range(len(raw_words)):
        for length in range(max(1, n - 1), min(len(raw_words) - start, n + 2) + 1):
            cand = word_tokens[start:start + length]
            score, _, _ = _window_similarity(expected, cand)
            # Prefer higher mean probability as tie-breaker
            probs = [float(w.get('score', w.get('probability', 0)) or 0) for w in raw_words[start:start + length]]
            combined = score + 0.05 * (sum(probs) / len(probs) if probs else 0.0)
            if combined > best_score:
                best_score = combined
                best_slice = raw_words[start:start + length]
    return best_slice


def alignment_quality(raw_words, query, duration):
    """Fail closed: fabricated or fallback timings alone cannot pass this gate.

    Calibrated for live sung theatre audio: short function words or phrase
    onsets can carry lower decoder probability even when content words align
    strongly, while wrong-text and silence controls stay near zero.
    """
    expected = tokens(query)
    finite = all(
        isinstance(w.get(k), (int, float)) and math.isfinite(w[k])
        for w in raw_words
        for k in ('start', 'end')
    )
    bounded = [
        w for w in raw_words
        if isinstance(w.get('start'), (int, float))
        and isinstance(w.get('end'), (int, float))
        and math.isfinite(w['start'])
        and math.isfinite(w['end'])
        and 0 <= w['start'] < w['end'] <= duration
        and (w['end'] - w['start']) <= 4.0
    ]
    scored = [
        w for w in bounded
        if float(w.get('score', w.get('probability', 0)) or 0) >= 0.25
    ]
    strong = [
        w for w in bounded
        if float(w.get('score', w.get('probability', 0)) or 0) >= 0.50
    ]
    actual = [t for w in scored for t in tokens(w.get('word', w.get('text', '')))]
    ordered = all(a['end'] <= b['start'] + 0.02 for a, b in zip(scored, scored[1:]))
    coverage = SequenceMatcher(None, expected, actual, autojunk=False).ratio()
    probs = [float(w.get('score', w.get('probability', 0)) or 0) for w in scored]
    mean_prob = (sum(probs) / len(probs)) if probs else 0.0
    min_scored = max(2, int(math.ceil(len(expected) * 0.65))) if expected else 3
    min_strong = max(2, int(math.floor(len(expected) * 0.45))) if expected else 2
    ok = (
        finite
        and ordered
        and coverage >= 0.70
        and len(scored) >= min_scored
        and len(strong) >= min_strong
        and mean_prob >= 0.50
    )
    failures = []
    if not finite:
        failures.append('nonfinite or missing timestamps')
    if not ordered:
        failures.append('overlapping or out-of-order word timings')
    if coverage < 0.70:
        failures.append('insufficient text coverage with acoustic support')
    if len(scored) < min_scored or len(strong) < min_strong:
        failures.append('too few scored, nonzero, bounded word spans')
    if scored and mean_prob < 0.50:
        failures.append('mean acoustic probability below 0.50')
    return dict(
        status='alignment support' if ok else 'failed',
        verified=False,
        coverage=round(coverage, 3),
        mean_probability=round(mean_prob, 3),
        strong_words=len(strong),
        words=scored,
        raw_words=raw_words,
        reason=(('; '.join(failures) + '. ') if failures else '')
        + 'Alignment support requires independent text anchors and listening; timings do not prove a lyric match.',
    )


def align_candidate(audio16, text, backend, model, *, context_text=None):
    """Explicit preloaded local model only. No automatic model downloads.

    When ``context_text`` is supplied (the full local transcript segment
    containing ``text``), the aligner aligns ``context_text`` to the segment
    audio and then extracts the aligned words matching ``text``. This prevents
    Whisper forced alignment from failing when a short anchor sits in the
    middle or end of a longer sung phrase.
    """
    align_text = (context_text or '').strip() or text
    duration = len(audio16) / 16000.0
    try:
        if backend == 'stable-ts':
            result = model.align(audio16, align_text, language='en', failure_threshold=0.25, verbose=None)
            raw = [] if result is None else [w.to_dict() for s in result.segments for w in (s.words or [])]
        elif backend == 'WhisperX':
            import whisperx
            acoustic, metadata = model
            result = whisperx.align(
                [dict(start=0., end=duration, text=align_text)],
                acoustic,
                metadata,
                audio16,
                'cpu',
                interpolate_method='ignore',
                return_char_alignments=False,
            )
            raw = result.get('word_segments', [])
        else:
            return dict(status='unavailable', verified=False, reason='No local aligner selected')
        target_words = _extract_anchor_words(raw, text) if context_text and align_text != text else raw
        report = alignment_quality(target_words, text, duration)
        if context_text and align_text != text:
            report['context_raw_words'] = raw
        return report
    except Exception as exc:
        return dict(status='failed', verified=False, reason=f'{type(exc).__name__}: {exc}')
