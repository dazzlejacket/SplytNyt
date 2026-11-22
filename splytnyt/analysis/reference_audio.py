"""Optional local audio anchors. Never fetches commercial recordings."""
import json
from pathlib import Path
import numpy as np
from .musical import measure_context


def load_manifest(path):
    root=Path(path).resolve()
    payload=json.loads(root.read_text(encoding='utf-8'))
    if payload.get('schema')!='splytnyt-reference-audio/1':
        raise ValueError('Expected splytnyt-reference-audio/1 manifest')
    entries=[]
    for item in payload.get('references',[]):
        file=(root.parent/item['path']).resolve()
        if not file.is_file():raise ValueError(f'Reference audio missing: {file}')
        ranges=item.get('query_ranges',[])
        if any(len(r)!=2 or not 0<=float(r[0])<float(r[1]) or not 4<=float(r[1])-float(r[0])<=30 for r in ranges):
            raise ValueError('Each query range must span 4 to 30 seconds at nonnegative times')
        entries.append(dict(reference_id=str(item['reference_id']),title=str(item.get('title','')),
            path=str(file),query_ranges=ranges))
    return entries


def subsequence_dtw(query, document):
    """Cosine DTW with bounded local tempo steps (1,1), (1,2), (2,1).

    A fragment can occur anywhere. Inserted dialogue is handled by independent
    fragments, not by stretching an album's full duration across a scene.
    """
    q=np.asarray(query);d=np.asarray(document)
    if min(len(q),len(d))<4:return None
    if np.mean(np.std(q,axis=0))<.12:return None
    best=None
    for shift in range(12):
        cost=1-np.clip(np.roll(q,shift,axis=1)@d.T,-1,1)
        n,m=cost.shape
        acc=np.full((n+2,m+2),np.inf);acc[1,1:]=0
        origins=np.zeros((n+2,m+2),dtype=np.int32)
        origins[1,:]=np.arange(m+2)
        columns=np.arange(m)
        for i in range(n):
            # These steps depend only on earlier query rows, so all document
            # positions can be evaluated together (no Python sample loop).
            options=np.stack((acc[i+1,1:m+1]+cost[i],
                acc[i+1,:m]+cost[i] if i>0 else np.full(m,np.inf),
                acc[i,1:m+1]+cost[i-1]+cost[i] if i>=2 else np.full(m,np.inf)))
            k=np.argmin(options,axis=0)
            acc[i+2,2:]=options[k,columns]
            previous=np.stack((origins[i+1,1:m+1],origins[i+1,:m],origins[i,1:m+1]))
            origins[i+2,2:]=previous[k,columns]
        j=int(np.argmin(acc[-1,2:]))+2
        start=max(0,int(origins[-1,j])-1);end=j-1
        score=float(acc[-1,j]/n)
        if end-start<len(q)*.4 or end-start>len(q)*2.1:continue
        if best is None or score<best['cost']:
            best=dict(start=start,end=end,cost=score,shift=shift)
    return best


def match_references(entries, context, sample_rate, cancel=lambda:False):
    """Multiple independent anchors allow medleys, reprises and missing items.

    Titles remain unconfirmed. A weak match is reported, not forced. No
    reference length becomes a proposed recording boundary.
    """
    import soundfile as sf
    class Reader:
        def __init__(self,path):
            self.file=sf.SoundFile(path);self.sample_rate=self.file.samplerate
            self.total_frames=len(self.file)
        def read(self,start,count):
            self.file.seek(start);return self.file.read(count,dtype='float32',always_2d=True)
    anchors=[];reports=[]
    for entry in entries:
        if cancel():raise InterruptedError('Reference matching cancelled')
        reader=Reader(entry['path'])
        try:ref=measure_context(reader,cancel)
        finally:reader.file.close()
        # Sparse short fragments survive inserted dialogue and arrangement edits.
        ranges=entry['query_ranges'] or [[s,s+12] for s in np.linspace(0,max(0,len(ref.chroma)*.5-12),4)]
        for start,end in ranges:
            if cancel():raise InterruptedError('Reference matching cancelled')
            query=ref.chroma[int(start/.5):int(end/.5)]
            search=context.chroma.copy()
            for occurrence in range(4):
                if cancel():raise InterruptedError('Reference matching cancelled')
                match=subsequence_dtw(query,search)
                record=dict(reference_id=entry['reference_id'],title=entry['title'],
                    query_start=float(start),query_end=float(end),match=match,confirmed=False)
                reports.append(record)
                if not match or match['cost']>.32:break
                anchors.append(dict(reference_id=entry['reference_id'],title=entry['title'],
                    start_frame=round(match['start']*.5*sample_rate),
                    end_frame=round(match['end']*.5*sample_rate),
                    cost=match['cost'],confirmed=False))
                search[match['start']:match['end']]=0
    return anchors,reports
