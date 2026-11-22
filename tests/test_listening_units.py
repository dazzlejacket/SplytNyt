"""Regression cases for listening-unit assembly, independent of show timings."""
from splytnyt.analysis.theater import TheaterSection, assemble_tracks


def piece(start, end, kind, music=0., singing=0.):
    return TheaterSection(start_frame=start*100,end_frame=end*100,section_type=kind,
        confidence=.7, activities=[kind], evidence={'mean_music_score':music,
        'independent_vocal_share':singing, 'activity_coverage':{'singing':singing}})


def test_scene_music_link_does_not_become_a_track_and_remains_reviewable():
    parts=[piece(0,50,'dialogue'),piece(50,56,'underscoring',.7),piece(56,120,'dialogue')]
    result=assemble_tracks(parts,100)
    assert len(result)==1 and result[0].end_frame==12000
    assert result[0].uncertain and result[0].evidence['review_regions']
    assert len(parts)==3 and parts[0].end_frame==5000


def test_short_vocal_reprise_is_preserved_between_scenes():
    result=assemble_tracks([piece(0,50,'dialogue'),piece(50,58,'underscoring',.8,.8),
                            piece(58,120,'dialogue')],100)
    assert len(result)==3 and result[1].section_type=='musical_number'


def test_applause_attaches_without_merging_separate_numbers():
    parts=[piece(0,80,'musical_number',.8),piece(80,88,'applause'),
           piece(88,150,'musical_number',.7)]
    result=assemble_tracks(parts,100)
    assert [(p.start_frame,p.end_frame) for p in result]==[(0,8800),(8800,15000)]
    assert len(assemble_tracks(parts,100,separate_applause=True))==3


def test_long_spoken_scene_is_not_absorbed_into_adjacent_songs():
    result=assemble_tracks([piece(0,80,'musical_number',.8),piece(80,160,'dialogue'),
                            piece(160,230,'musical_number',.7)],100)
    assert len(result)==3


def test_transcribed_words_over_music_are_not_independent_proof_of_a_short_song():
    link=piece(50,56,'underscoring',.8,.9)
    link.evidence['independent_vocal_share']=0
    result=assemble_tracks([piece(0,50,'dialogue'),link,piece(56,120,'dialogue')],100)
    assert len(result)==1 and result[0].uncertain
