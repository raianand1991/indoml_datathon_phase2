# Extracted unchanged from the Phase-2 submission notebook
# (notebooks/00_phase2_submission_dictionary_all.ipynb, cell 5).
"""Track-2 direct residual reconstruction. Inference never accepts a clean reference."""
from __future__ import annotations
import copy, hashlib, io, json, math, os, re, shutil, time, unicodedata, wave, zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import soundfile as sf

RECON_VERSION = 'track2_v6_direct_residual_20260918_r1'

@dataclass(frozen=True)
class ReconConfig:
    sample_rate: int = 16000
    max_components: int = 16
    duplicate_iou: float = 0.65
    alignment_radius: int = 4
    min_abs_gain: float = 0.01
    max_abs_gain: float = 8.0
    max_condition: float = 1e7
    max_gain_disagreement: float = 0.50
    max_removal_energy_ratio: float = 1.15
    max_output_energy_ratio: float = 1.05
    max_natural_modified_fraction: float = 0.05
    ridge_relative: float = 1e-9
    whitening_floor_fraction: float = 0.02
    gls_iterations: int = 3
    natural_damage_db: float = -24.0


def jsonable(x):
    if isinstance(x, dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, np.generic): return jsonable(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def put_json(path, obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp')
    with open(t,'w',encoding='utf-8') as f:
        json.dump(jsonable(obj),f,indent=2,ensure_ascii=False,allow_nan=False)
    os.replace(t,p)


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()


def sha_json(obj):
    return hashlib.sha256(json.dumps(jsonable(obj),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def canonical_pcm(y):
    a=np.asarray(y,np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Wave must be finite, nonempty, and mono.')
    # Deterministic rounding, not peak normalization. Used for metrics AND export.
    return np.clip(np.rint(a*32768.0),-32768,32767).astype('<i2')


def pcm_roundtrip(y): return canonical_pcm(y).astype(np.float64)/32768.0


def wav_bytes(y, sr=16000):
    buf=io.BytesIO()
    with wave.open(buf,'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(canonical_pcm(y).tobytes())
    return buf.getvalue()


def write_pcm(path,y,sr=16000):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    data=wav_bytes(y,sr); tmp=p.with_name(p.name+'.tmp')
    with open(tmp,'wb') as f: f.write(data)
    os.replace(tmp,p)
    return hashlib.sha256(data).hexdigest()


def read_wave_strict(path_or_bytes,sr=16000):
    source=io.BytesIO(path_or_bytes) if isinstance(path_or_bytes,(bytes,bytearray)) else path_or_bytes
    y,fs=sf.read(source,dtype='float32',always_2d=True)
    if fs!=sr or y.shape[1]!=1:
        raise ValueError(f'Expected native {sr}-Hz mono audio, got {fs} Hz / {y.shape[1]} channels. No unvalidated resampling.')
    a=y[:,0]
    if not len(a) or not np.isfinite(a).all(): raise ValueError('Empty or invalid audio.')
    return a


def wav_audit(path_or_bytes,expected_frames,sr=16000):
    data=path_or_bytes if isinstance(path_or_bytes,bytes) else Path(path_or_bytes).read_bytes()
    with wave.open(io.BytesIO(data),'rb') as w:
        got=(w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes(),w.getcomptype())
        wanted=(sr,1,2,int(expected_frames),'NONE')
        if got!=wanted: raise ValueError(f'WAV compliance error: {got} != {wanted}')
        pcm=w.readframes(expected_frames)
        if len(pcm)!=expected_frames*2: raise ValueError('Truncated PCM payload.')
    return hashlib.sha256(data).hexdigest()


class StrictWaveSource:
    """One explicit directory/ZIP, exact IDs; no duplicate stems or inferred paths."""
    def __init__(self,path,required_ids):
        self.path=str(path); self.archive=None; self.index={}
        required_ids=set(map(str,required_ids))
        p=Path(path)
        if p.is_file() and zipfile.is_zipfile(p):
            self.archive=zipfile.ZipFile(p)
            names=[z.filename for z in self.archive.infolist() if not z.is_dir() and z.filename.lower().endswith('.wav') and '__MACOSX' not in z.filename]
        elif p.is_dir(): names=[str(q) for q in p.rglob('*') if q.is_file() and q.suffix.lower()=='.wav']
        else: raise FileNotFoundError(p)
        for name in names:
            cid=Path(name).stem
            if cid not in required_ids: continue
            if cid in self.index: raise ValueError(f'Duplicate waveform ID: {cid} in {p}')
            self.index[cid]=name
        missing=required_ids-set(self.index)
        if missing: raise ValueError(f'{p}: {len(missing)} missing IDs; examples {sorted(missing)[:3]}')
    def bytes(self,cid):
        loc=self.index[cid]
        return self.archive.read(loc) if self.archive is not None else Path(loc).read_bytes()
    def load(self,cid): return read_wave_strict(self.bytes(cid))
    def close(self):
        if self.archive is not None: self.archive.close()


def si_sdr(estimate,reference):
    """Zero-mean SI-SDR, no alignment/resampling; 120-dB numerical ceiling only."""
    x=np.asarray(estimate,np.float64); s=np.asarray(reference,np.float64)
    if x.shape!=s.shape or x.ndim!=1: raise ValueError('SI-SDR length/shape mismatch.')
    if not np.isfinite(x).all() or not np.isfinite(s).all(): raise ValueError('Nonfinite SI-SDR input.')
    x=x-x.mean(); s=s-s.mean(); ss=float(s@s)
    if ss<=1e-18: raise ValueError('Silent reference: SI-SDR undefined, not silently excluded.')
    target=(float(x@s)/ss)*s; e=x-target
    num=float(target@target); den=float(e@e)
    value=10*math.log10(max(num,ss*1e-16)/max(den,ss*1e-12))
    return float(min(120.,value))


def residual_change_db(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    return float(10*np.log10(max(float((a-b)@(a-b)),1e-24)/max(float(b@b),1e-24)))


def fit_centered(A,y,weights=None,ridge=1e-9):
    A=np.asarray(A,np.float64); y=np.asarray(y,np.float64)
    if A.ndim==1: A=A[:,None]
    w=np.ones(len(y),np.float64) if weights is None else np.asarray(weights,np.float64)
    w=np.clip(w,0,None); w=w/max(float(w.mean()),1e-30)
    am=(A*w[:,None]).sum(0)/max(w.sum(),1e-30); ym=float((y*w).sum()/max(w.sum(),1e-30))
    B=A-am; v=y-ym
    gram=B.T@(w[:,None]*B); rhs=B.T@(w*v)
    diag=np.diag(gram); scale=np.sqrt(np.maximum(diag,1e-30))
    G=gram/scale[:,None]/scale[None,:]; r=rhs/scale
    cond=float(np.linalg.cond(G))
    g=np.linalg.solve(G+ridge*np.eye(G.shape[0]),r)/scale
    return g,cond


def emphasis(A,y):
    return A[1:]-.90*A[:-1],y[1:]-.90*y[:-1]


def residual_psd(residual,nfft,sr=16000):
    nwin=min(512,len(residual))
    if nwin<16: return np.ones(nfft//2+1)
    hop=max(1,nwin//2); win=np.hanning(nwin)
    starts=range(0,len(residual)-nwin+1,hop)
    # Median-Welch PSD limits unusually loud speech frames without target labels.
    frames=np.stack([residual[a:a+nwin]*win for a in starts])
    p=np.median(np.abs(np.fft.rfft(frames,n=512,axis=1))**2,axis=0)
    p=np.convolve(np.pad(p,(3,3),mode='edge'),np.ones(7)/7,mode='valid')
    return np.interp(np.linspace(0,sr/2,nfft//2+1),np.linspace(0,sr/2,len(p)),p)


def fit_spectral(A,y,cfg):
    g,_=fit_centered(*emphasis(A,y),ridge=cfg.ridge_relative)
    nfft=1<<(len(y)-1).bit_length()
    AF=np.fft.rfft(A,n=nfft,axis=0); YF=np.fft.rfft(y,n=nfft)
    freq=np.linspace(0,cfg.sample_rate/2,len(YF))
    cond=1.
    for _ in range(cfg.gls_iterations):
        psd=residual_psd(y-A@g,nfft,cfg.sample_rate)
        positive=psd[psd>0]
        floor=max(float(np.median(positive))*cfg.whitening_floor_fraction if len(positive) else 0.,1e-20)
        w=1./np.maximum(psd,floor); w/=max(float(np.median(w)),1e-30)
        w=np.clip(w,1e-3,1e3); w[freq<70.]=0.
        # One-sided real FFT Parseval weights.
        w[1:-1]*=2.
        gram=(AF.conj().T@(w[:,None]*AF)).real
        rhs=(AF.conj().T@(w*YF)).real
        scale=np.sqrt(np.maximum(np.diag(gram),1e-30))
        G=gram/scale[:,None]/scale[None,:]
        cond=float(np.linalg.cond(G))
        g=np.linalg.solve(G+cfg.ridge_relative*np.eye(len(g)),rhs/scale)/scale
    return g,cond


def estimate_gains(A,y,polarity,method,cfg):
    if method=='stored_gain': return np.asarray(polarity,np.float64),1.
    if method=='mixture_ols': return fit_centered(A,y,ridge=cfg.ridge_relative)
    P,v=emphasis(A,y)
    if method=='preemphasis_ols': return fit_centered(P,v,ridge=cfg.ridge_relative)
    if method=='spectral_gls': return fit_spectral(A,y,cfg)
    if method=='block_irls':
        g,cond=fit_centered(P,v,ridge=cfg.ridge_relative)
        block=max(64,int(.030*cfg.sample_rate))
        idx=np.arange(len(v))//block
        for _ in range(cfg.gls_iterations):
            residual=v-P@g
            counts=np.bincount(idx); power=np.bincount(idx,weights=residual**2)/counts
            floor=max(float(np.median(power))*.05,1e-15)
            bw=1./np.maximum(power,floor); bw/=max(float(np.median(bw)),1e-30)
            weights=np.clip(bw,0.02,50.)[idx]
            g,cond=fit_centered(P,v,weights,cfg.ridge_relative)
        return g,cond
    raise ValueError(f'Unknown deployable gain method: {method}')


def gain_disagreement(A,y,g,cfg):
    # Two interleaved block fits: agreement diagnostic, not a probability of correctness.
    P,v=emphasis(A,y); block=max(64,int(.020*cfg.sample_rate))
    parity=(np.arange(len(v))//block)%2
    folds=[]
    for f in (0,1):
        mask=parity==f
        if mask.sum()<4*A.shape[1] or np.any((P[mask]**2).sum(0)<1e-14): return math.inf
        try: gg,cc=fit_centered(P[mask],v[mask],ridge=cfg.ridge_relative)
        except np.linalg.LinAlgError: return math.inf
        if cc>cfg.max_condition: return math.inf
        folds.append(gg)
    return float(np.max(np.abs(folds[0]-folds[1])/np.maximum(np.abs(g),.025)))


def refine_placement(y,t,start,radius=4):
    """Integer-sample correlation refinement. The raw template is never faded/stretched."""
    t=np.asarray(t,np.float64); dt=t[1:]-.9*t[:-1]; dt=dt-dt.mean()
    ten=float(dt@dt)
    if ten<1e-18: return start
    best=(-math.inf,start)
    for a in range(max(0,start-radius),min(len(y)-len(t),start+radius)+1):
        seg=np.asarray(y[a:a+len(t)],np.float64); ds=seg[1:]-.9*seg[:-1]; ds-=ds.mean()
        q=abs(float(ds@dt))/math.sqrt(max(ten*float(ds@ds),1e-30))
        # Stable tie-breaking favours the verified V5 start.
        score=q-1e-12*abs(a-start)
        if score>best[0]: best=(score,a)
    return best[1]


def interval_iou(a,b):
    inter=max(0,min(a[1],b[1])-max(a[0],b[0])); union=max(a[1],b[1])-min(a[0],b[0])
    return inter/union if union>0 else 0.


def prepared_matches(y,proposals,bank_by_id,cfg):
    """Prepare ALL verified alternatives once; threshold/NMS are applied afterwards."""
    out=[]
    for p in proposals:
        tid=p['template_id']
        if tid not in bank_by_id: raise ValueError(f'Unknown cached template {tid}; wrong bank.')
        t=np.asarray(bank_by_id[tid]['wave'],np.float64)
        start=int(round(float(p['onset'])*cfg.sample_rate))
        if start<0 or start+len(t)>len(y): continue
        if abs((float(p['offset'])-float(p['onset']))*cfg.sample_rate-len(t))>2:
            raise ValueError('Match duration is incompatible with frozen template length.')
        a=refine_placement(y,t,start,cfg.alignment_radius)
        out.append(dict(start=int(a),end=int(a+len(t)),quality=float(p['quality']),
                        polarity=int(p['polarity']),template_id=tid,wave=t))
    return out


def select_evidence(prepared,threshold,cfg):
    keep=[]
    for r in sorted(prepared,key=lambda q:q['quality'],reverse=True):
        if r['quality']<threshold: continue
        if any(interval_iou((r['start'],r['end']),(v['start'],v['end']))>=cfg.duplicate_iou for v in keep): continue
        keep.append(r)
        if len(keep)>=cfg.max_components: break
    return sorted(keep,key=lambda q:q['start'])


def overlap_components(rows):
    groups=[]
    for r in sorted(rows,key=lambda q:q['start']):
        if groups and r['start']<max(x['end'] for x in groups[-1]): groups[-1].append(r)
        else: groups.append([r])
    return groups


def component_matrix(y,rows):
    lo=max(0,min(r['start'] for r in rows)-32)
    hi=min(len(y),max(r['end'] for r in rows)+32)
    A=np.zeros((hi-lo,len(rows)),np.float64)
    for j,r in enumerate(rows): A[r['start']-lo:r['end']-lo,j]=r['wave']
    return lo,hi,A,np.asarray(y[lo:hi],np.float64)


def reconstruct(y,prepared,policy,cfg):
    """Deployable path: input mixture, template matches, frozen policy. NO references."""
    y=np.asarray(y,np.float64)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all(): raise ValueError('Invalid noisy waveform.')
    selected=select_evidence(prepared,float(policy['threshold']),cfg)
    z=y.copy(); log=[]; accepted=[]
    for rows in overlap_components(selected):
        lo,hi,A,v=component_matrix(y,rows)
        why=None; cond=None; disagreement=None; gains=None
        try:
            gains,cond=estimate_gains(A,v,[r['polarity'] for r in rows],policy['method'],cfg)
            if not np.isfinite(gains).all() or cond>cfg.max_condition: why='ill_conditioned_fit'
            if why is None and (np.any(np.abs(gains)<cfg.min_abs_gain) or np.any(np.abs(gains)>cfg.max_abs_gain)):
                why='gain_out_of_bounds'
            if why is None and any(float(g)*r['polarity']<=0 for g,r in zip(gains,rows)): why='gain_polarity_mismatch'
            if why is None:
                disagreement=gain_disagreement(A,v,gains,cfg)
                if disagreement>cfg.max_gain_disagreement: why='inconsistent_gain'
            noise=A@gains if gains is not None else np.zeros(len(v))
            removed_ratio=float(noise@noise)/max(float(v@v),1e-20)
            candidate=v-noise
            if why is None and removed_ratio>cfg.max_removal_energy_ratio: why='excessive_removal_energy'
            if why is None and float(candidate@candidate)>cfg.max_output_energy_ratio*max(float(v@v),1e-20): why='output_energy_increase'
            if why is None and (candidate.min()<-1. or candidate.max()>32767./32768.+1e-8): why='would_clip_pcm16'
        except (np.linalg.LinAlgError,FloatingPointError,ValueError) as e:
            why=f'numerical_fit_rejected:{type(e).__name__}'
        if why is None:
            # Reconstruct from ORIGINAL mixture, once. No generic mask or second fade.
            z[lo:hi]-=noise
            accepted.extend(dict(template_id=r['template_id'],start=r['start'],end=r['end'],gain=float(g),quality=r['quality'])
                            for g,r in zip(gains,rows))
        log.append(dict(template_ids=[r['template_id'] for r in rows],accepted=why is None,reason=why,
                        condition=cond,gain_disagreement=disagreement,
                        gains=None if gains is None else gains.tolist()))
    q=pcm_roundtrip(z); base=pcm_roundtrip(y)
    return q,dict(method=policy['method'],threshold=float(policy['threshold']),
                  proposed=len(prepared),selected=len(selected),accepted_templates=len(accepted),
                  changed=bool(np.any(q!=base)),change_db=residual_change_db(q,base),components=log,matches=accepted)


def reference_assisted_diagnostic(y,clean,prepared,policy,cfg):
    """DIAGNOSTIC ONLY. Same noisy-only matches, but clean is a nuisance regressor for ideal gains.
    Never exported as audio. Does NOT search using clean/GT and is NOT a deployable score.
    """
    y=np.asarray(y,np.float64); clean=np.asarray(clean,np.float64)
    z=y.copy(); nfit=0
    for rows in overlap_components(select_evidence(prepared,policy['threshold'],cfg)):
        lo,hi,A,v=component_matrix(y,rows)
        B=np.column_stack((A,clean[lo:hi]))
        try:
            g,_=fit_centered(B,v,ridge=1e-12)
            if np.isfinite(g).all(): z[lo:hi]-=A@g[:-1]; nfit+=len(rows)
        except np.linalg.LinAlgError: pass
    return dict(reference_assisted_sisdr_float=si_sdr(z,clean),
                reference_assisted_sisdr_pcm16=si_sdr(pcm_roundtrip(z),clean),fitted_templates=nfit)


TAG_RE=re.compile(r'</?[^<>]*>|\[[^\[\]]*\]')
def normalize_text(text):
    s=TAG_RE.sub(' ',str(text))
    s=''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s)
    return ' '.join(s.lower().split())


def word_errors(reference,hypothesis):
    a=normalize_text(reference).split(); b=normalize_text(hypothesis).split()
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1): nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1],len(a)


def pooled_wer(ids,refs,transcripts):
    errors=words=0
    for cid in ids:
        e,n=word_errors(refs[cid],transcripts[cid]['text']); errors+=e; words+=n
    if words==0: raise ValueError('No reference words: real WER cannot be evaluated.')
    return dict(wer=errors/words,errors=errors,words=words,clips=len(ids))


def hyp_text(h):
    if isinstance(h,str): return h.strip()
    if isinstance(h,dict) and isinstance(h.get('text'),str): return h['text'].strip()
    if hasattr(h,'text') and isinstance(h.text,str): return h.text.strip()
    if isinstance(h,(list,tuple)) and len(h)==1: return hyp_text(h[0])
    raise TypeError(f'Unrecognized ASR hypothesis type: {type(h)}. Refusing an invented empty transcript.')


class HashASR:
    """Model-revision + exact WAV byte hash cache. Never reuses an ID-only transcript."""
    def __init__(self,model,model_signature,cache_dir,batch_size=8):
        self.model=model; self.model_signature=model_signature
        self.cache_dir=Path(cache_dir); self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.batch_size=batch_size
    def _call(self,paths):
        try:
            hs=self.model.transcribe([str(p) for p in paths],return_hypotheses=True)
            if isinstance(hs,(list,tuple)) and len(hs)==len(paths): return [hyp_text(h) for h in hs]
            if len(paths)==1: return [hyp_text(hs)]
        except Exception:
            if len(paths)==1: raise
        # A cardinality mismatch never silently truncates zip(...).
        return [self._call([p])[0] for p in paths]
    def transcribe(self,paths,progress=lambda x,**k:x):
        out={}; todo={}
        for cid,path in paths.items():
            sh=sha_file(path); key=sha_json(dict(model=self.model_signature,wav=sh))
            dest=self.cache_dir/(key+'.json')
            rec=None
            if dest.is_file():
                with open(dest,encoding='utf-8') as f: r=json.load(f)
                if r.get('model_signature')==self.model_signature and r.get('wav_sha256')==sh and isinstance(r.get('text'),str): rec=r
            if rec is not None: out[cid]=rec
            else:
                if key not in todo: todo[key]=dict(path=str(path),sha=sh,ids=[],dest=dest)
                todo[key]['ids'].append(cid)
        jobs=list(todo.values())
        print(f'ASR exact-hash hits: {len(out)} | unique WAVs to decode: {len(jobs)}')
        for a in progress(range(0,len(jobs),self.batch_size),desc='SraVaani final-waveform ASR'):
            ch=jobs[a:a+self.batch_size]; texts=self._call([j['path'] for j in ch])
            if len(texts)!=len(ch): raise RuntimeError('ASR cardinality mismatch.')
            for job,text in zip(ch,texts):
                # Empty output is allowed only when the real recognizer returned it.
                rec=dict(text=text,wav_sha256=job['sha'],model_signature=self.model_signature)
                put_json(job['dest'],rec)
                for cid in job['ids']: out[cid]=rec
        if set(out)!=set(paths): raise RuntimeError('Incomplete ASR coverage.')
        return out


def pack_submission(paths,asr,expected_frames,output_zip,sr=16000):
    """Only final audio and matching ASR enter the ZIP; diagnostics/clean refs never do."""
    ids=sorted(paths)
    if set(ids)!=set(asr) or set(ids)!=set(expected_frames): raise ValueError('Submission ID mismatch.')
    for cid in ids:
        if not cid or Path(cid).name!=cid or any(c in cid for c in ('/','\\','\n','\r')): raise ValueError('Unsafe clip ID.')
        h=wav_audit(paths[cid],expected_frames[cid],sr)
        if h!=asr[cid]['wav_sha256']: raise ValueError(f'ASR/audio hash mismatch: {cid}')
        if not isinstance(asr[cid]['text'],str): raise ValueError('ASR text must be a string.')
    p=Path(output_zip); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+'.tmp')
    tr=''.join(json.dumps(dict(clip_id=c,text=asr[c]['text']),ensure_ascii=False,allow_nan=False)+'\n' for c in ids).encode('utf-8')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=9,allowZip64=True) as z:
        z.writestr('transcripts.jsonl',tr)
        for cid in ids: z.write(paths[cid],cid+'.wav')
    with zipfile.ZipFile(tmp) as z:
        wanted={'transcripts.jsonl'}|{c+'.wav' for c in ids}
        if len(z.namelist())!=len(wanted) or set(z.namelist())!=wanted: raise ValueError('Archive contents mismatch.')
        if z.testzip() is not None: raise ValueError('Archive CRC failure.')
        for cid in ids:
            data=z.read(cid+'.wav')
            if wav_audit(data,expected_frames[cid],sr)!=asr[cid]['wav_sha256']: raise ValueError('Packaged audio mismatch.')
        rows=[json.loads(x) for x in z.read('transcripts.jsonl').decode().splitlines()]
        if len(rows)!=len(ids) or {r['clip_id'] for r in rows}!=set(ids): raise ValueError('Packaged transcript mismatch.')
    os.replace(tmp,p)
    return dict(zip_path=str(p),zip_sha256=sha_file(p),clips=len(ids),transcripts_sha256=hashlib.sha256(tr).hexdigest())
