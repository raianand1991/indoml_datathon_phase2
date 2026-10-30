# Extracted unchanged from the Phase-2 submission notebook
# (notebooks/00_phase2_submission_dictionary_all.ipynb, cell 4).
"""Full-clip residual-template detector. No query labels/clean audio in inference."""
from __future__ import annotations
import os, io, re, json, math, time, hashlib, pickle, zipfile, copy
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import OrderedDict, Counter
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

VERSION = 'residual_evidence_v5_20260917_r1'

@dataclass(frozen=True)
class MatcherConfig:
    sample_rate: int = 16000
    coarse_rate: int = 4000
    coarse_floor: float = 0.25
    template_batch: int = 32
    peaks_per_template: int = 3
    verify_topk: int = 128
    raw_radius: int = 12
    min_template_sec: float = 0.08
    max_template_sec: float = 6.0
    max_templates: int = 2000
    chunk_sec: float = 24.0
    fft_cache_mb: int = 384
    merge_iou: float = 0.65
    seed: int = 517


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def wave_sha(y):
    return hashlib.sha256(np.asarray(y, '<f4').tobytes()).hexdigest()


def atomic_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, p)


def atomic_pickle(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)


def event_list(v):
    if isinstance(v, dict):
        if 'events' not in v:
            raise ValueError('Prediction row has no events field.')
        v = v['events']
    out = []
    for e in v:
        if isinstance(e, dict):
            a = float(e['onset'] if 'onset' in e else e['start'])
            b = float(e['offset'] if 'offset' in e else e['end'])
        else:
            a, b = float(e[0]), float(e[1])
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b):
            raise ValueError(f'Invalid event: {e}')
        out.append((a, b))
    return out


def read_jsonl(path, predictions=True):
    out = {}
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip(): continue
            r = json.loads(line)
            cid = str(r['clip_id'])
            if cid in out: raise ValueError(f'Duplicate clip_id on line {n}: {cid}')
            if predictions: event_list(r)
            out[cid] = r
    if not out: raise ValueError(f'Empty JSONL: {path}')
    return out


def source_group(cid):
    # Best-effort recording prefix, NOT a claim of verified speaker identity.
    # Example: ...844425030213767_010_LowerDibangvalley-SPECIFIC_...
    m = re.search(r'_\d+_(?:GENERIC|[^_]+-SPECIFIC)_', str(cid))
    return str(cid)[:m.start()] if m else str(cid).rsplit('_', 3)[0]


def split_bucket(group):
    return int(hashlib.sha256(('v5_bank_source_' + group).encode()).hexdigest()[:12], 16) % 10


def bandlimited_resample(y, src, dst):
    """Windowed-sinc resampling, no SciPy/librosa dependency; common grid for pairs."""
    y = np.asarray(y, np.float32)
    if src == dst: return y.copy()
    n = int(round(len(y) * dst / src))
    if n < 1: raise ValueError('Audio became empty during resampling.')
    cutoff = min(1.0, float(dst) / src) * 0.94
    radius = int(math.ceil(20 / cutoff))
    offsets = np.arange(-radius, radius + 1)
    out = np.empty(n, np.float32)
    for a in range(0, n, 2048):
        b = min(n, a + 2048)
        pos = np.arange(a, b, dtype=np.float64) * src / dst
        ix = np.floor(pos).astype(np.int64)[:, None] + offsets[None]
        distance = pos[:, None] - ix
        w = cutoff * np.sinc(cutoff * distance)
        w *= np.where(np.abs(distance) < radius, 0.5 + 0.5*np.cos(np.pi*distance/radius), 0)
        valid = (ix >= 0) & (ix < len(y))
        w *= valid
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        out[a:b] = np.sum(y[np.clip(ix, 0, len(y)-1)] * w, axis=1)
    return out


def load_audio(path_or_bytes, sr=16000):
    obj = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    y, fs = sf.read(obj, dtype='float32', always_2d=True)
    y = np.mean(y, axis=1, dtype=np.float32)
    if not len(y) or not np.isfinite(y).all(): raise ValueError(f'Invalid audio: {str(path_or_bytes)[:100]}')
    return bandlimited_resample(y, int(fs), sr)


def condition_wave(y):
    """Mild pre-emphasis, identically applied to templates and query recordings."""
    y = np.asarray(y, np.float32)
    return np.concatenate([y[:1], y[1:] - 0.85*y[:-1]]).astype(np.float32)


def coarse_wave(y, cfg):
    ratio = cfg.sample_rate // cfg.coarse_rate
    if ratio * cfg.coarse_rate != cfg.sample_rate:
        raise ValueError('coarse_rate must divide sample_rate.')
    y = condition_wave(y)
    t = np.arange(-32, 33, dtype=np.float64)
    cutoff = 0.90 / ratio
    kernel = cutoff * np.sinc(cutoff*t) * np.hanning(len(t))
    kernel /= kernel.sum()
    # Odd, symmetric FIR; zero padding preserves the time origin.
    padded = np.pad(y, (32, 32))
    return np.convolve(padded, kernel, mode='valid')[::ratio].astype(np.float32)


def iou(a, b):
    inter = max(0., min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.


def closest_first_pairs(pred, ref):
    # Same explicitly named local scorer used in the user's notebooks.
    edges = []
    for i, (a,b) in enumerate(pred):
        for j, (s,e) in enumerate(ref):
            tol = max(0.20 * (e-s), 0.05)
            if abs(a-s) <= tol and abs(b-e) <= tol:
                edges.append((abs(a-s) + abs(b-e), i, j))
    used_p, used_r, pairs = set(), set(), []
    for _, i, j in sorted(edges):
        if i not in used_p and j not in used_r:
            used_p.add(i); used_r.add(j); pairs.append((i,j))
    return pairs


def score_events(pred, refs, durs, ids):
    tp = fp = fn = 0; dice = []; npred = ngt = 0
    for cid in ids:
        p, g = event_list(pred[cid]), event_list(refs[cid])
        nmatch = len(closest_first_pairs(p, g))
        tp += nmatch; fp += len(p)-nmatch; fn += len(g)-nmatch
        npred += len(p); ngt += len(g)
        n = max(1, int(round(durs[cid]/0.01)))
        pm = np.zeros(n, bool); gm = np.zeros(n, bool)
        for events, mask in ((p,pm),(g,gm)):
            for a,b in events:
                lo,hi = max(0,int(round(a/0.01))), min(n,int(round(b/0.01)))
                if hi > lo: mask[lo:hi] = True
        den = int(pm.sum()+gm.sum())
        dice.append(1. if den == 0 else 2.*np.logical_and(pm,gm).sum()/den)
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    f1 = 2.*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    d = float(np.mean(dice)) if dice else 0.
    return dict(f1=f1, precision=precision, recall=recall, dice=d, combined=f1+d,
                tp=tp, fp=fp, fn=fn, clips=len(ids), predicted_events=npred, gt_events=ngt)


def wilson_lower(success, total, z=1.96):
    if total == 0: return 0.
    p = success/total; den = 1.+z*z/total
    return (p+z*z/(2*total)-z*math.sqrt(p*(1-p)/total+z*z/(4*total*total)))/den


def build_bank(bank_ids, refs, audio_paths, clean_paths, groups, cfg, progress=lambda x, **k: x):
    records, audits, source_hashes = [], [], []
    sr = cfg.sample_rate
    for cid in progress(bank_ids, desc='BANK pairs only: noisy minus clean'):
        x = load_audio(audio_paths[cid], sr)
        s = load_audio(clean_paths[cid], sr)
        if len(x) != len(s):
            raise ValueError(f'Pair length mismatch: {cid}; no silent truncation allowed.')
        mask = np.zeros(len(x), bool)
        for a,b in refs[cid]:
            mask[max(0,int(round(a*sr))):min(len(x),int(round(b*sr)))] = True
        outside = ~mask
        # Gain alignment is estimated from BANK-only non-event samples.
        scale = float(np.dot(x[outside],s[outside])/(np.dot(s[outside],s[outside])+1e-12)) if outside.sum()>sr//10 else 1.
        scale = float(np.clip(scale,0.8,1.2))
        residual = x - scale*s
        inside_rms = float(np.sqrt(np.mean(residual[mask]**2)+1e-20)) if mask.any() else 0.
        outside_rms = float(np.sqrt(np.mean(residual[outside]**2)+1e-20)) if outside.any() else 0.
        ratio = outside_rms/max(inside_rms,1e-10)
        good = bool(inside_rms>1e-5 and ratio<=0.15)
        audits.append(dict(cid=cid, scale=scale, inside_rms=inside_rms, outside_rms=outside_rms,
                           outside_inside_ratio=ratio, used=good))
        source_hashes.append(wave_sha(x))
        if not good: continue
        for j,(a,b) in enumerate(refs[cid]):
            # Mixed overlapping events are not single-template examples.
            if any(k!=j and min(b,e)-max(a,s0)>0.005 for k,(s0,e) in enumerate(refs[cid])): continue
            if not cfg.min_template_sec <= b-a <= cfg.max_template_sec: continue
            lo,hi = int(round(a*sr)), int(round(b*sr))
            if lo < 0 or hi > len(x) + int(.02*sr): raise ValueError(f'BANK label out of audio range: {cid}')
            hi = min(hi, len(x))
            w = residual[lo:hi].astype(np.float32)
            if len(w)<16 or float(np.std(w))<1e-5: continue
            coarse = coarse_wave(w,cfg)
            coarse -= coarse.mean()
            norm = float(np.linalg.norm(coarse))
            if norm < 1e-7: continue
            raw = condition_wave(w)
            # Reject templates with effectively just one energetic quarter.
            chunks = np.array([np.sum(v*v) for v in np.array_split(raw,4)])
            if np.sum(chunks > .02*max(float(chunks.sum()),1e-10)) < 2: continue
            records.append(dict(tid=f'{cid}::{j}', source_cid=cid, group=groups[cid],
                                wave=w, raw=raw, coarse=(coarse/norm).astype(np.float32),
                                annotation_duration=b-a))
    # Exact waveform deduplication only; no fitting on SELECT/HOLD.
    unique = {}; dedup = 0
    for r in records:
        w = r['wave']/max(float(np.max(np.abs(r['wave']))),1e-9)
        key = hashlib.sha256(np.round(w*32700).astype('<i2').tobytes()).hexdigest()
        if key in unique: dedup += 1
        else: unique[key] = r
    records = sorted(unique.values(), key=lambda r: hashlib.sha256(r['tid'].encode()).hexdigest())
    capped = max(0,len(records)-cfg.max_templates)
    records = records[:cfg.max_templates]
    if not records: raise RuntimeError('No valid additive-noise templates. Stop: the waveform-template hypothesis is unsupported for these pairs.')
    return dict(version=VERSION, cfg=asdict(cfg), records=records, audits=audits,
                bank_ids=list(bank_ids), bank_groups=sorted({groups[c] for c in bank_ids}),
                source_hashes=source_hashes, deduplicated=dedup, capped=capped)


class ResidualMatcher:
    def __init__(self, bank, cfg, device=None):
        self.cfg, self.bank = cfg, bank
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.records = bank['records']
        self.source_hashes = set(bank.get('source_hashes',[]))
        self.cache = OrderedDict(); self.cache_bytes = 0
        self.signature = digest(dict(version=VERSION,cfg=asdict(cfg),
            templates=[(r['tid'], wave_sha(r['wave'])) for r in self.records]))

    def _spectrum(self, ix, nfft):
        key = (tuple(ix), nfft)
        if key in self.cache:
            z = self.cache.pop(key); self.cache[key] = z
            return z
        t = torch.zeros((len(ix),nfft),dtype=torch.float32,device=self.device)
        for j,i in enumerate(ix):
            x = self.records[i]['coarse']; t[j,:len(x)] = torch.as_tensor(x,device=self.device)
        z = torch.fft.rfft(t,n=nfft,dim=-1)
        cost = z.numel()*z.element_size(); max_bytes = self.cfg.fft_cache_mb*(1<<20)
        while self.cache and self.cache_bytes+cost > max_bytes:
            _,old = self.cache.popitem(last=False); self.cache_bytes-=old.numel()*old.element_size()
        if cost<=max_bytes: self.cache[key] = z; self.cache_bytes+=cost
        return z

    @torch.inference_mode()
    def _coarse_scan(self,y,exclude_group=None):
        x = coarse_wave(y,self.cfg); n = len(x)
        eligible = [i for i,r in enumerate(self.records)
                    if len(r['coarse'])<=n and (exclude_group is None or r['group']!=exclude_group)]
        if not eligible: return []
        mmax = max(len(self.records[i]['coarse']) for i in eligible)
        nfft = 1<<(n+mmax-2).bit_length()
        xt = torch.as_tensor(x,dtype=torch.float32,device=self.device)
        xf = torch.fft.rfft(xt,n=nfft)
        # Use float64 prefix sums to avoid variance cancellation near silence.
        xx = xt.double()
        pref = torch.cat([torch.zeros(1,dtype=torch.float64,device=self.device),xx.cumsum(0)])
        pref2 = torch.cat([torch.zeros(1,dtype=torch.float64,device=self.device),(xx*xx).cumsum(0)])
        starts = torch.arange(n,device=self.device)[None]
        rows = []
        for a in range(0,len(eligible),self.cfg.template_batch):
            ix = eligible[a:a+self.cfg.template_batch]
            lens = torch.tensor([len(self.records[i]['coarse']) for i in ix],device=self.device)[:,None]
            tf = self._spectrum(ix,nfft)
            dot = torch.fft.irfft(xf[None]*tf.conj(),n=nfft,dim=-1)[:,:n]
            ends = (starts+lens).clamp(max=n)
            total = pref[ends]-pref[starts]; total2 = pref2[ends]-pref2[starts]
            energy = (total2-total*total/lens).clamp(min=1e-16).float()
            q = (dot.abs()/torch.sqrt(energy)).clamp(max=1.)
            q = q.masked_fill((starts+lens>n)|(energy<1e-12),-1.)
            for _ in range(self.cfg.peaks_per_template):
                vals, pos = q.max(dim=1)
                for j,(v,p) in enumerate(zip(vals.cpu().tolist(),pos.cpu().tolist())):
                    if v>=self.cfg.coarse_floor:
                        rows.append(dict(template_index=ix[j], sample4k=int(p), coarse=float(v)))
                rad = torch.maximum(lens//2,torch.full_like(lens,int(.06*self.cfg.coarse_rate)))
                q = q.masked_fill((starts-pos[:,None]).abs()<=rad,-1.)
        return sorted(rows,key=lambda r:r['coarse'],reverse=True)[:self.cfg.verify_topk]

    def _verify(self,raw_query,proposal):
        r = self.records[proposal['template_index']]; t = r['raw']; m = len(t)
        center = proposal['sample4k']*(self.cfg.sample_rate//self.cfg.coarse_rate)
        lo,hi = max(0,center-self.cfg.raw_radius), min(len(raw_query)-m,center+self.cfg.raw_radius)
        if hi<lo: return None
        t0=t.astype(np.float64)-np.mean(t,dtype=np.float64); ten=float(t0@t0)
        if ten<1e-12: return None
        seg=raw_query[lo:hi+m].astype(np.float64)
        dots=np.correlate(seg,t0,mode='valid')
        cs=np.r_[0.,np.cumsum(seg)]; cs2=np.r_[0.,np.cumsum(seg*seg)]
        sums=cs[m:]-cs[:-m]; energies=np.maximum(cs2[m:]-cs2[:-m]-sums*sums/m,1e-15)
        corr=dots/np.sqrt(energies*ten)
        j=int(np.argmax(np.abs(corr))); start=lo+j; sign=1. if corr[j]>=0 else -1.
        qseg=raw_query[start:start+m]
        chunks=[]
        for inds in np.array_split(np.arange(m),4):
            tt=t[inds].astype(np.float64); yy=qseg[inds].astype(np.float64)
            tt-=tt.mean(); yy-=yy.mean(); et=float(tt@tt)
            if et < .015*ten: continue
            chunks.append(float(sign*np.dot(tt,yy)/math.sqrt(max(et*float(yy@yy),1e-20))))
        if len(chunks)<2: return None
        raw_corr=float(abs(corr[j])); chunk=float(np.median(chunks))
        quality=float(np.clip(min(raw_corr,chunk),0.,1.))
        return dict(onset=start/self.cfg.sample_rate,offset=(start+m)/self.cfg.sample_rate,
                    quality=quality,corr16=raw_corr,corr4k=proposal['coarse'],chunk_median=chunk,
                    template_id=r['tid'],template_source=r['source_cid'],polarity=int(sign))

    def scan(self,y,exclude_group=None):
        """Only noisy waveform and a training bank are inputs. No event/GT arguments."""
        y=np.asarray(y,np.float32)
        if not np.isfinite(y).all(): raise ValueError('Nonfinite query audio.')
        if wave_sha(y) in self.source_hashes:
            raise RuntimeError('Query waveform exactly duplicates a BANK source. Refusing self-retrieval.')
        chunk=int(self.cfg.chunk_sec*self.cfg.sample_rate)
        overlap=int((self.cfg.max_template_sec+.05)*self.cfg.sample_rate)
        if chunk<=overlap: raise ValueError('chunk_sec must exceed max_template_sec.')
        starts=list(range(0,max(1,len(y)-overlap),chunk-overlap))
        found=[]
        for a in starts:
            yy=y[a:min(len(y),a+chunk)]
            if len(yy)<int(self.cfg.min_template_sec*self.cfg.sample_rate): continue
            raw=condition_wave(yy)
            for p in self._coarse_scan(yy,exclude_group):
                v=self._verify(raw,p)
                if v is not None:
                    v['onset']+=a/self.cfg.sample_rate; v['offset']+=a/self.cfg.sample_rate
                    found.append(v)
        # Keep geometric alternatives; final threshold/NMS is applied later.
        best={}
        for r in found:
            k=(round(r['onset'],4),round(r['offset'],4),r['template_id'])
            if k not in best or r['quality']>best[k]['quality']: best[k]=r
        return sorted(best.values(),key=lambda r:r['quality'],reverse=True)


def select_matches(rows, threshold, duplicate_iou=.65):
    chosen=[]
    for r in sorted(rows,key=lambda z:z['quality'],reverse=True):
        if r['quality']<threshold: continue
        ev=(r['onset'],r['offset'])
        if any(iou(ev,(x['onset'],x['offset']))>=duplicate_iou or
               (abs(ev[0]-x['onset'])<=.02 and abs(ev[1]-x['offset'])<=.02) for x in chosen): continue
        chosen.append(r)
    return sorted(chosen,key=lambda z:(z['onset'],z['offset']))


def patch_events(base, matches, max_move=.25, minimum_iou=.50, allow_add=False, add_threshold=.85):
    """Conservative edit: only mutual one-to-one overlaps; unmatched base retained."""
    base=event_list(base); proposal=[(m['onset'],m['offset']) for m in matches]
    edges=[]
    p2b={j:[] for j in range(len(proposal))}; b2p={i:[] for i in range(len(base))}
    for j,p in enumerate(proposal):
        for i,b in enumerate(base):
            ov=iou(p,b)
            if ov>=minimum_iou and max(abs(p[0]-b[0]),abs(p[1]-b[1]))<=max_move:
                p2b[j].append(i); b2p[i].append(j); edges.append((i,j))
    out=list(base); used=set(); changed=0
    for i,j in edges:
        if len(p2b[j])==len(b2p[i])==1:
            out[i]=proposal[j]; used.add(j); changed+=int(out[i]!=base[i])
    added=0
    if allow_add:
        for j,p in enumerate(proposal):
            if j in used or matches[j]['quality']<add_threshold: continue
            # No addition overlapping or touching an existing interval.
            if any(min(p[1],b[1]) - max(p[0],b[0]) > -.04 for b in out): continue
            out.append(p); added+=1
    return sorted(out),dict(changed=changed,added=added)


class WaveSource:
    """Read WAVs from one explicit folder/ZIP without extracting or guessing IDs."""
    def __init__(self,path,required_ids):
        self.path=str(path); self.archive=None; self.index={}
        if Path(path).is_file() and zipfile.is_zipfile(path):
            self.archive=zipfile.ZipFile(path)
            names=[i.filename for i in self.archive.infolist() if not i.is_dir() and i.filename.lower().endswith('.wav') and '__MACOSX' not in i.filename]
        elif Path(path).is_dir():
            names=[str(p) for p in Path(path).rglob('*.wav')]
        else: raise FileNotFoundError(path)
        for name in names:
            cid=Path(name).stem
            if cid not in required_ids: continue
            if cid in self.index: raise ValueError(f'Duplicate waveform stem in {path}: {cid}')
            self.index[cid]=name
        missing=set(required_ids)-set(self.index)
        if missing: raise ValueError(f'{path}: missing {len(missing)} required WAVs; examples {sorted(missing)[:3]}')
    def load(self,cid,sr=16000):
        loc=self.index[cid]
        return load_audio(self.archive.read(loc) if self.archive is not None else loc,sr)
    def duration(self,cid):
        loc=self.index[cid]
        x=io.BytesIO(self.archive.read(loc)) if self.archive is not None else loc
        info=sf.info(x)
        return info.frames/info.samplerate


def persist_query(matcher,y,path,exclude_group=None):
    wh=wave_sha(y); path=Path(path)
    signature=digest(dict(matcher=matcher.signature,wave=wh,exclude_group=exclude_group))
    if path.is_file():
        with open(path) as f: old=json.load(f)
        if old.get('signature')==signature: return old['proposals']
    rows=matcher.scan(y,exclude_group)
    atomic_json(path,dict(signature=signature,query_wave_sha256=wh,proposals=rows))
    return rows


def unit_tests():
    cfg=MatcherConfig(coarse_floor=.15,template_batch=2,peaks_per_template=2,verify_topk=20,fft_cache_mb=16)
    rng=np.random.default_rng(601)
    sr=cfg.sample_rate
    t=(rng.normal(0,.15,int(.36*sr))*np.hanning(int(.36*sr))).astype(np.float32)
    source=rng.normal(0,.02,int(1.8*sr)).astype(np.float32)
    a=1923; source[a:a+len(t)]+=t
    coarse=coarse_wave(t,cfg); coarse-=coarse.mean(); coarse/=np.linalg.norm(coarse)
    bank=dict(records=[dict(tid='test_template',source_cid='bank',group='g0',wave=t,raw=condition_wave(t),coarse=coarse)],source_hashes=[])
    m=ResidualMatcher(bank,cfg,device='cpu')
    got=select_matches(m.scan(source),.85)
    assert got and abs(got[0]['onset']-a/sr)<=1/sr, got
    assert abs(got[0]['offset']-(a+len(t))/sr)<=1/sr
    neg=m.scan(rng.normal(0,.02,len(source)).astype(np.float32))
    assert not select_matches(neg,.85)
    # Confirm no self-retrieval.
    m.source_hashes.add(wave_sha(source))
    try: m.scan(source)
    except RuntimeError: pass
    else: raise AssertionError('Self-retrieval guard failed.')
    refs={'x':[(1.,1.5)]}; durs={'x':2.}
    assert score_events(refs,refs,durs,['x'])['f1']==1.
    assert score_events({'x':[]},refs,durs,['x'])['fn']==1
    row=dict(onset=1.01,offset=1.51,quality=.99)
    patched,diag=patch_events([(1.,1.5)],[row])
    assert len(patched)==1 and diag['changed']==1
    patched,diag=patch_events([(1.,1.5)],[dict(onset=0.,offset=.3,quality=.99)])
    assert patched==[(1.,1.5)] and diag['added']==0
    print('PASS: FFT timing, raw sample alignment, negative audio, self-retrieval guard, scorer, conservative edits.')
    return True
