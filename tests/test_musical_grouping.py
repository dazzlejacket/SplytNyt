"""Musical membership is evaluated independently of activity labels."""
from types import SimpleNamespace
import numpy as np
from splytnyt.analysis.musical import MusicalContext, group_numbers, recurrence
from splytnyt.analysis.theater import TheaterSection
from splytnyt.analysis.reference_audio import subsequence_dtw, load_manifest
import pytest


def fixture(bridge_seconds=75, speech=.3, music=.1, shift=0):
    rng=np.random.default_rng(91)
    phrase=rng.normal(size=(24,12));phrase/=np.linalg.norm(phrase,axis=1,keepdims=True)
    left=np.tile(phrase,(4,1));bridge=rng.normal(size=(bridge_seconds*2,12))
    bridge/=np.linalg.norm(bridge,axis=1,keepdims=True)
    right=np.roll(left,shift,axis=1)
    c=MusicalContext(np.concatenate([left,bridge,right]))
    edges=[0,len(left)//2,len(left)//2+bridge_seconds,len(c.chroma)//2]
    parts=[TheaterSection(a*100,b*100,k,.7,activities=[k],evidence={'original':i})
           for i,(a,b,k) in enumerate(zip(edges,edges[1:],['musical_number','dialogue','musical_number']))]
    n=len(c.chroma)
    ev=SimpleNamespace(music=np.full(n,music),speech=np.full(n,speech),applause=np.zeros(n),frame_seconds=.5,n_frames=lambda:n)
    return parts,c,ev


def test_number_survives_long_spoken_passage_and_modulation_without_minimum_length_change():
    parts,c,ev=fixture(75,shift=4)
    groups,decisions=group_numbers(parts,c,100,events=ev)
    assert len(groups)==1 and decisions[0]['merged']
    assert groups[0].start_frame==0 and groups[0].end_frame==parts[-1].end_frame
    assert len(groups[0].evidence['internal_sections'])==3
    assert parts[1].section_type=='dialogue' and parts[0].evidence=={'original':0}


def test_independent_dialogue_scene_survives_a_related_accompaniment_and_same_motif_return():
    parts,c,ev=fixture(75,speech=.8)
    groups,decisions=group_numbers(parts,c,100,events=ev)
    assert len(groups)==3 and not decisions[0]['merged']
    assert decisions[0]['evidence']['scene_dominant']


def test_narrative_reset_preserves_following_scene_despite_returning_music():
    parts,c,ev=fixture(25)
    transcript=[dict(start=48,end=65,text='This is a story about our travels. We arrived years ago.')]
    groups,d=group_numbers(parts,c,100,transcript,events=ev)
    assert len(groups)==3 and d[0]['evidence']['narrative_reset']


def test_same_key_or_static_chord_does_not_establish_number_membership():
    c=np.tile(np.eye(12)[0],(120,1))
    score=recurrence(c,c)
    assert score['phrase']==0 and score['contrast']==0
    assert subsequence_dtw(c[:24],c) is None


def test_conflicting_reference_identities_keep_separate_numbers():
    parts,c,ev=fixture(10)
    anchors=[dict(reference_id='a',start_frame=0,end_frame=1000),
             dict(reference_id='b',start_frame=parts[-1].start_frame,end_frame=parts[-1].end_frame)]
    groups,d=group_numbers(parts,c,100,anchors=anchors,events=ev)
    assert len(groups)==3 and d[0]['evidence']['reference_conflict']


def test_dtw_tolerates_tempo_transposition_and_different_spectral_timbre_features():
    rng=np.random.default_rng(4);q=rng.normal(size=(24,12));q/=np.linalg.norm(q,axis=1,keepdims=True)
    # Pitch pooling has removed octave/instrument identity; add performance noise.
    changed=np.repeat(np.roll(q,3,axis=1),2,axis=0)+rng.normal(0,.02,(48,12))
    changed/=np.linalg.norm(changed,axis=1,keepdims=True)
    d=np.concatenate([rng.normal(size=(15,12)),changed,rng.normal(size=(15,12))])
    d/=np.linalg.norm(d,axis=1,keepdims=True)
    match=subsequence_dtw(q,d)
    assert match['cost']<.05 and match['shift']==3
    assert abs(match['start']-15)<=2 and abs(match['end']-63)<=2


def test_manifest_cannot_substitute_tracklist_metadata_for_audio(tmp_path):
    path=tmp_path/'reference.json'
    path.write_text('{"schema":"splytnyt-reference-audio/1","references":[{"reference_id":"a","path":"missing.flac"}]}')
    with pytest.raises(ValueError,match='missing'):load_manifest(path)


def test_local_audio_fragments_anchor_arrangement_change_inserted_dialogue_and_reprise(tmp_path):
    import soundfile as sf
    from splytnyt.analysis.musical import measure_context
    from splytnyt.analysis.reference_audio import match_references
    sr=8000
    notes=[60,64,67,62,65,69,59,63,66,61,68,65]
    def music(notes,seconds,timbre=0,shift=0):
        pieces=[]
        for note in notes:
            t=np.arange(round(sr*seconds))/sr;f=440*2**((note+shift-69)/12)
            x=np.sin(2*np.pi*f*t)+(.15 if timbre==0 else .5)*np.sin(2*np.pi*2*f*t)
            pieces.append(x*.2)
        return np.concatenate(pieces)
    reference=music(notes,1.)
    # Different tempo, key and overtone mix, with a spoken/noisy insertion.
    rng=np.random.default_rng(5)
    performance=np.concatenate([music(notes[:6],1.5,1,3),rng.normal(0,.02,sr*5),
                                music(notes[6:],1.5,1,3),rng.normal(0,.02,sr*3),
                                music(notes[:6],1.5,1,3)])
    refpath=tmp_path/'reference.wav';sf.write(refpath,reference,sr)
    class Reader:
        sample_rate=sr;total_frames=len(performance)
        def read(self,start,count):return performance[start:start+count,None]
    context=measure_context(Reader())
    entries=[dict(reference_id='synthetic',title='Unconfirmed test',path=str(refpath),
                  query_ranges=[[0,6],[6,12]])]
    anchors,reports=match_references(entries,context,sr)
    assert len(anchors)>=3
    assert all(not a['confirmed'] for a in anchors)
    assert any(a['start_frame']>sr*24 for a in anchors) # reprise
    assert any(sr*12<a['start_frame']<sr*17 for a in anchors) # after insertion


def test_apply_grouping_preserves_manual_override_locked_track_and_undo():
    from splytnyt.app.session import Session
    from splytnyt.app.jobs import TheaterOutcome
    from splytnyt.core.model import Source,Track
    session=Session();p=session.project
    p.sources=[Source(path='fixture.wav',frames=10000,sample_rate=100,channels=1)]
    manual=Track(start_frame=2000,end_frame=4000,title='My scene',boundary_basis='manual')
    locked=Track(start_frame=6000,end_frame=8000,title='Locked',locked=True)
    p.tracks=[Track(start_frame=0,end_frame=2000),manual,Track(start_frame=4000,end_frame=6000),locked,Track(start_frame=8000,end_frame=10000)]
    before=[(t.start_frame,t.end_frame,t.title,t.locked) for t in p.sorted_tracks()]
    outcome=TheaterOutcome(classifications=[(0,10000,'musical_number',.7,[])])
    session.apply_theater_analysis(outcome)
    tracks=p.sorted_tracks()
    assert next(t for t in tracks if t.title=='My scene').start_frame==2000
    assert next(t for t in tracks if t.title=='Locked').end_frame==8000
    assert tracks[0].start_frame==0 and tracks[-1].end_frame==10000
    assert all(a.end_frame==b.start_frame for a,b in zip(tracks,tracks[1:]))
    session.undo_action()
    assert [(t.start_frame,t.end_frame,t.title,t.locked) for t in session.project.sorted_tracks()]==before


@pytest.mark.parametrize('edit',['title','type','tag','merge'])
def test_manual_edits_survive_save_reopen_and_reanalysis(tmp_path,edit):
    from splytnyt.app.session import Session
    from splytnyt.app.jobs import TheaterOutcome
    from splytnyt.core.model import Source,Track
    session=Session();p=session.project
    p.sources=[Source(path='fixture.wav',frames=10000,sample_rate=100,channels=1)]
    a=Track(start_frame=0,end_frame=3000,title='Track 01',boundary_basis='assessed')
    b=Track(start_frame=3000,end_frame=6000,title='Track 02',boundary_basis='assessed')
    c=Track(start_frame=6000,end_frame=10000,title='Track 03',boundary_basis='assessed')
    p.tracks=[a,b,c]
    if edit=='title':session.set_track_title(b.track_id,'My number')
    elif edit=='type':session.set_track_section_type(b.track_id,'dialogue')
    elif edit=='tag':session.set_track_metadata(b.track_id,'artist','My cast')
    else:session.merge_with_previous(b.track_id)
    target=a if edit=='merge' else b
    expected=(target.start_frame,target.end_frame,target.title,target.section_type,target.metadata.artist)
    session.save_project(str(tmp_path/'manual.splytnyt'))
    session.open_project(str(tmp_path/'manual.splytnyt'))
    session.apply_theater_analysis(TheaterOutcome(classifications=[(0,10000,'musical_number',.7,[])]))
    preserved=next(t for t in session.project.tracks if t.track_id==target.track_id)
    assert (preserved.start_frame,preserved.end_frame,preserved.title,preserved.section_type,preserved.metadata.artist)==expected
