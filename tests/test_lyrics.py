"""Authored nonsense fixtures; no real lyrics or ASR-derived references."""
import json
import numpy as np
import pytest
from splytnyt.analysis.lyrics import *

def ref():
    return parse_references('# Imaginary lantern number\nSource: authored test fixture\nVersion: synthetic\nSilver lanterns carry winter home\nAmber rivers turn the copper wheel\nViolet mountains close the paper sky')

def seg(text,start,end,**kw):return dict(text=text,start=start,end=end,**kw)

def test_dialogue_and_instrumental_gaps_keep_ordered_number_with_omission_and_error():
    segments=[seg('Silver lanterns carry winter home',10,15),seg('The mayor is here',25,32),
              seg('Amber rivers turn a copper wheel',70,77),seg('Violet mountains close the paper sky',140,148)]
    result=match_references(ref(),segments)['results'][0]['occurrences'][0]
    assert len(result['anchors'])==3 and len(result['gaps'])==2
    assert result['status']=='multi-anchor proposal' and not result['verified']
    assert (result['start'],result['end'])==(10,148)

def test_distant_reprise_is_separate_and_chorus_alone_does_not_establish_identity():
    lyrics=ref()[0]['lyrics'].splitlines()
    segments=[seg(line,10+i*20,15+i*20) for i,line in enumerate(lyrics)]
    segments += [seg(line,1010+i*20,1015+i*20) for i,line in enumerate(lyrics)]
    occurrences=match_references(ref(),segments)['results'][0]['occurrences']
    assert len(occurrences)==2
    assert all(o['end']-o['start']<100 for o in occurrences)
    partial=match_references(ref(),[segments[1],segments[4]])['results'][0]['occurrences']
    assert all(o['status']=='partial proposal' for o in partial)

def test_rejected_asr_title_and_duration_never_make_match():
    result=match_references(ref(),[seg('Imaginary lantern number',0,200),seg(ref()[0]['lyrics'],250,270,accepted=False)])
    assert result['results'][0]['unmatched']

def test_missing_provenance_is_error():
    with pytest.raises(ValueError):parse_references('# Test\nSilver lanterns carry winter home')

def test_alignment_can_fail_even_when_all_words_have_timestamps():
    text='Silver lanterns carry winter home'
    raw=[dict(word=t,start=i*.2,end=(i+1)*.2) for i,t in enumerate(tokens(text))]
    assert alignment_quality(raw,text,10)['status']=='failed' # No acoustic scores.
    for word in raw:word['score']=.9
    assert alignment_quality(raw,text,10)['status']=='alignment support'
    raw[-1]['start']=float('nan')
    assert alignment_quality(raw,text,10)['status']=='failed'

def test_alignment_backend_exception_and_empty_output_fail_closed():
    class Fail:
        def align(self,*a,**k):raise RuntimeError('could not align')
    assert align_candidate(np.zeros(16000),'test words here','stable-ts',Fail())['status']=='failed'
    assert alignment_quality([], 'test words here', 1)['status']=='failed'

def test_audio_extent_includes_intro_and_decay_and_reports_unresolved_continuity():
    class Reader:
        sample_rate=100;total_frames=10000
        def read(self,start,count):
            t=np.arange(start,start+count)/100
            return np.where((t>=8)&(t<27),.2,0).astype('float32')[:,None]
    extent=propose_audio_extent(Reader(),10,24)
    assert extent['start_frame']<=800 and extent['end_frame']>=2700
    assert not extent['intro_unresolved'] and not extent['tail_unresolved']
    assert not extent['verified']

def test_review_cache_is_undoable_and_preserves_locked_manual_tracks(tmp_path):
    from splytnyt.app.session import Session
    from splytnyt.core.model import Track
    s=Session();s.project.tracks=[Track(start_frame=0,end_frame=100,locked=True,title='Edited title')]
    before=s.project.clone().tracks
    report=match_references(ref(),[seg('Silver lanterns carry winter home',1,3)])
    s._mutate('Lyric review',lambda p:p.cache.update(lyric_review=report,lyric_references=ref()))
    assert s.project.tracks==before
    s.undo_action();assert 'lyric_review' not in s.project.cache and s.project.tracks==before
    s.redo_action();assert s.project.cache['lyric_review']==report
    path=tmp_path/'test.splytnyt';s.save_project(str(path));s.open_project(str(path))
    assert s.project.cache['lyric_review']==report and s.project.tracks==before

def test_reference_ids_are_stable_across_import_and_source_changes_invalidate_evidence():
    from splytnyt.core.model import Project,Source
    references=ref()
    assert parse_references(json.dumps(dict(schema='splytnyt-lyrics/1',references=references)))==references
    p=Project();p.sources=[Source(path='one.wav',frames=100,sample_rate=100)]
    before=evidence_signature(p);p.sources[0].gap_frames=25
    assert evidence_signature(p)!=before
    before=evidence_signature(p);p.cache['theater_transcript']={'fingerprint':'new ASR'}
    assert evidence_signature(p)!=before

def test_audio_extent_never_quantizes_inside_the_last_sung_word():
    class Reader:
        sample_rate=100;total_frames=10000
        def read(self,start,count):return np.zeros((count,1),dtype='float32')
    extent=propose_audio_extent(Reader(),10.03,24.07)
    assert extent['start_frame']<=1003 and extent['end_frame']>=2407


def test_cross_song_guard_prevents_merging_across_another_song():
    refs = parse_references(
        '# Song One\nSource: synthetic\nVersion: test\n'
        'Silver lanterns carry winter home\nAmber rivers turn the copper wheel\nViolet mountains close the paper sky\n'
        '# Song Two\nSource: synthetic\nVersion: test\n'
        'Golden trumpets wake the marble hall\nCrimson banners cross the harbor bridge\n'
    )
    segments = [
        seg('Silver lanterns carry winter home', 10, 15),
        seg('Golden trumpets wake the marble hall', 60, 66),
        seg('Violet mountains close the paper sky', 110, 116),
    ]
    res = match_references(refs, segments)
    song_one_occs = res['results'][0]['occurrences']
    # Song One's opening (10-15) and ending (110-116) must NOT merge across Song Two (60-66)
    assert len(song_one_occs) == 2
    assert song_one_occs[0]['end'] <= 15 and song_one_occs[1]['start'] >= 110


def test_stopword_only_overlap_does_not_create_false_positive():
    refs = parse_references(
        '# Street Number\nSource: synthetic\nVersion: test\n'
        'stuff on the street\n'
    )
    # "out on the street" shares only stopwords + 1 content word ("street")
    segments = [
        seg('she walked out on the street alone', 10, 15),
        seg('strutting her stuff on the street tonight', 50, 56),
    ]
    occs = match_references(refs, segments)['results'][0]['occurrences']
    assert len(occs) == 1
    assert occs[0]['start'] == 50


def test_apply_proposed_grouping_is_undoable_contiguous_and_preserves_locks():
    from splytnyt.app.session import Session
    from splytnyt.core.model import Source, Track

    sr = 100
    s = Session()
    s.project.sources = [Source(path='show.wav', sample_rate=sr, channels=2, frames=100000)]
    s.project.tracks = [
        Track(start_frame=0, end_frame=20000, title='Intro'),
        Track(start_frame=20000, end_frame=40000, title='Bridge'),
        Track(start_frame=40000, end_frame=60000, title='Ending'),
        Track(start_frame=60000, end_frame=80000, title='Locked Scene', locked=True),
        Track(start_frame=80000, end_frame=100000, title='Next Song'),
    ]
    before_spans = [(t.start_frame, t.end_frame, t.title) for t in s.sorted_tracks()]

    res = s.apply_lyric_grouping([
        {'title': 'Big Opening Number', 'start_frame': 5000, 'end_frame': 55000, 'status': 'multi-anchor proposal'},
        {'title': 'Should Be Skipped', 'start_frame': 65000, 'end_frame': 90000, 'status': 'multi-anchor proposal'},
    ])
    assert res['applied'] == 1
    assert res['skipped_protected'] == 1
    assert s.project.is_contiguous()

    grouped = [t for t in s.sorted_tracks() if t.title == 'Big Opening Number']
    assert len(grouped) == 1
    assert (grouped[0].start_frame, grouped[0].end_frame) == (5000, 55000)
    assert grouped[0].section_type == 'musical_number'

    locked = [t for t in s.sorted_tracks() if t.locked]
    assert len(locked) == 1 and locked[0].title == 'Locked Scene'
    assert (locked[0].start_frame, locked[0].end_frame) == (60000, 80000)

    # Single Ctrl+Z restores exact prior timeline
    assert s.undo_action() is True
    assert [(t.start_frame, t.end_frame, t.title) for t in s.sorted_tracks()] == before_spans
    assert s.project.is_contiguous()


