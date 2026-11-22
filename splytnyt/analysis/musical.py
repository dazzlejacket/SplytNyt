"""Musical context above activity labels. No production names or cut times."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import resample_poly
from scipy.ndimage import uniform_filter1d

@dataclass
class MusicalContext:
    chroma: np.ndarray
    hop_seconds: float = .5

    def excerpt(self, start, end, sample_rate):
        return self.chroma[max(0,int(start/sample_rate/self.hop_seconds)):max(0,int(end/sample_rate/self.hop_seconds))]

def measure_context(reader, cancel=lambda: False):
    """Streaming, octave pooled, locally whitened pitch-class energy at 2 Hz."""
    from math import gcd
    sr=reader.sample_rate; target=8000; g=gcd(sr,target)
    freq=np.fft.rfftfreq(4096,1/target)
    valid=(freq>=65)&(freq<=3000)
    pitch=np.round(69+12*np.log2(np.maximum(freq,1)/440)).astype(int)%12
    rows=[]
    for start in range(0,reader.total_frames,30*sr):
        if cancel(): raise InterruptedError('Musical context cancelled')
        # A half-second of real context spans block edges. Only the actual
        # recording endpoints are zero padded.
        margin=sr//2
        lo=max(0,start-margin);hi=min(reader.total_frames,start+30*sr+margin)
        x=reader.read(lo,hi-lo).mean(axis=1)
        x=resample_poly(x,target//g,sr//g)
        left=round((start-lo)*target/sr)
        x=np.pad(x,(2048,2048))
        count=int(np.ceil(min(30*sr,reader.total_frames-start)/sr*2))
        for index in range(count):
            pos=2048+left+index*4000
            spec=np.abs(np.fft.rfft(x[pos-2048:pos+2048]*np.hanning(4096)))
            spec=np.maximum(0,spec-uniform_filter1d(spec,31))
            c=np.array([np.sum(spec[valid&(pitch==k)]) for k in range(12)])
            c=np.sqrt(c/max(c.sum(),1e-9));c-=c.mean()
            rows.append(c/max(np.linalg.norm(c),1e-9))
    return MusicalContext(np.asarray(rows,dtype=np.float32))

def recurrence(left,right):
    """Ordered phrase similarity minus unordered pitch-profile similarity.

    Require changing harmony: a static chord/key must never prove continuity.
    Compares 6-second phrases at several tempi and transpositions.
    """
    if min(len(left),len(right))<12:return {'phrase':0.,'contrast':0.,'shift':0}
    best=(0.,0.,0)
    for stretch in (.75,1.,1.25):
        n=max(12,int(12*stretch)); offsets=np.linspace(0,n-1,12).astype(int)
        if len(right)<n:continue
        a=np.lib.stride_tricks.sliding_window_view(left,12,axis=0).transpose(0,2,1)[::4]
        b=np.lib.stride_tricks.sliding_window_view(right,n,axis=0).transpose(0,2,1)[::4][:,offsets]
        variation=np.minimum(np.mean(np.std(a,axis=1),axis=1)[:,None],np.mean(np.std(b,axis=1),axis=1)[None,:])
        for shift in range(12):
            br=np.roll(b,shift,axis=2)
            scores=np.einsum('itk,jtk->ij',a,br)/12
            unordered=np.einsum('ik,jk->ij',a.mean(axis=1),br.mean(axis=1))
            contrast=scores-unordered
            objective=np.where(variation>.12,scores+contrast,-1)
            i,j=np.unravel_index(np.argmax(objective),objective.shape)
            if objective[i,j]>=0 and scores[i,j]+contrast[i,j]>best[0]+best[1]:best=(float(scores[i,j]),float(contrast[i,j]),shift)
    return dict(zip(('phrase','contrast','shift'),best))


def group_numbers(sections, context, sample_rate, transcript=(), anchors=(), events=None):
    """Propose number membership; retain speech/singing as timed observations.

    Returning motifs are supporting evidence, never proof of identity. Scene
    discourse vetoes a merge even when an accompaniment returns. All automatic
    joins remain reviewable; the activity classifier is never rewritten.
    """
    from copy import deepcopy
    import re
    from .theater import TheaterSection
    parts=deepcopy(sections)
    decisions=[]
    for p in parts:
        p.evidence['internal_sections']=[dict(start_frame=p.start_frame,
            end_frame=p.end_frame,section_type=p.section_type,
            activities=list(p.activities),evidence=deepcopy(p.evidence))]
    def words(p):
        return ' '.join(str(s.get('text','')) for s in transcript
            if s.get('accepted',True) and s.get('start',0)*sample_rate<p.end_frame
            and s.get('end',0)*sample_rate>p.start_frame).lower()
    def join(items,reason,ev):
        a,b=items[0],items[-1]
        annotations=[v for p in items for v in p.evidence['internal_sections']]
        return TheaterSection(a.start_frame,b.end_frame,'musical_number',
            min(p.confidence for p in items),
            list(dict.fromkeys(v for p in items for v in p.activities)),
            list(dict.fromkeys([r for p in items for r in p.reasons]+[reason])),
            True,dict(internal_sections=annotations,grouping_evidence=ev,
                review_regions=[v for p in items for v in p.evidence.get('review_regions',[])]+[dict(start_frame=items[1].start_frame,
                    end_frame=items[-1].start_frame,reason=reason)]))
    out=[];i=0
    while i<len(parts):
        left=parts[i];i+=1
        while i+1<len(parts) and left.section_type=='musical_number':
            bridge,right=parts[i:i+2]
            if bridge.section_type not in ('dialogue','mixed','underscoring','uncertain') or right.section_type!='musical_number':break
            # Compare evolving pitch sequences, not a mean chroma/key vector.
            a=context.excerpt(max(left.start_frame,left.end_frame-90*sample_rate),left.end_frame,sample_rate)
            b=context.excerpt(right.start_frame,min(right.end_frame,right.start_frame+90*sample_rate),sample_rate)
            ev=recurrence(a,b)
            text=words(bridge)
            staging=bool(re.search(r'ladies and gentlemen|may i (?:introduce|present)|welcome (?:here|to)|our own beloved|these are|this is where',text))
            narrative=bool(re.search(r'(?:this is a |tell (?:you |us )?(?:a |the )?)story|(?:i|we) arrived|hours before|years ago|the next (?:day|morning)|turns out they|they wanted me',text))
            # Two separated reference anchors to the same reference item may
            # support a return; a conflicting identity always keeps a boundary.
            la=[v for v in anchors if v['start_frame']<left.end_frame and v['end_frame']>left.start_frame]
            ra=[v for v in anchors if v['start_frame']<right.end_frame and v['end_frame']>right.start_frame]
            shared={v['reference_id'] for v in la}&{v['reference_id'] for v in ra}
            conflict=bool(la and ra and not shared)
            music=speech=closure=0.
            if events is not None and events.n_frames():
                lo=max(0,int(bridge.start_frame/sample_rate/events.frame_seconds))
                hi=min(events.n_frames(),max(lo+1,int(bridge.end_frame/sample_rate/events.frame_seconds)))
                music=float(np.mean(events.music[lo:hi]))
                speech=float(np.mean(events.speech[lo:hi]))
                closure=float(np.max(events.applause[lo:hi]))
            # Public address inside a demonstrated medley supplies discourse
            # context for later exchanges until a programme/narrative reset.
            presentation_context=bool(left.evidence.get('grouping_evidence', {}).get('presentation_context',False))
            presentation_context=presentation_context or staging
            scene_dominant=(events is not None and speech>.32 and music<.35
                            and speech>1.25*music and not presentation_context)
            new_presentation=bool(re.search(r'and now.*(?:time has come|meet our|present)',text))
            scene_reset=narrative or scene_dominant or new_presentation
            # Applause plus independent spoken exchange is closure evidence;
            # applause during a staged introduction may remain inside a medley.
            closed_scene=closure>=.55 and not staging
            resume=ev['phrase']>=.56 and ev['contrast']>=.14
            medley=staging and ev['phrase']>=.40 and ev['contrast']>=.14
            merged=bool((resume or medley or shared) and not scene_reset and not closed_scene and not conflict)
            reason=('Recurring changing accompaniment supports a resumed number' if resume else
                    'Staged introduction and changing musical phrases support an internal medley passage' if medley else
                    'Local reference anchors support a returning number' if shared else
                    'Insufficient independent evidence that the number resumes')
            if narrative:reason='Narrative scene reset in transcript; recurring music alone cannot merge this scene'
            if scene_dominant:reason='Sustained independent speech with weak music; motif return is insufficient to absorb a scene'
            if new_presentation:reason='Transcript introduces the next featured performance; preserve the scene boundary'
            if closed_scene:reason='Audience closure followed by a spoken scene; preserve the boundary'
            if conflict:reason='Different local reference identities; preserve separate numbers'
            ev.update(staged_introduction=staging,narrative_reset=narrative,
                bridge_music=music,bridge_speech=speech,bridge_applause_peak=closure,
                scene_dominant=scene_dominant,new_presentation=new_presentation,
                presentation_context=presentation_context,
                shared_reference_ids=sorted(shared),reference_conflict=conflict)
            decisions.append(dict(start_frame=bridge.start_frame,end_frame=right.start_frame,
                merged=merged,reason=reason,evidence=ev))
            if not merged:break
            left=join([left,bridge,right],reason,ev);i+=2
        out.append(left)
    return out,decisions
