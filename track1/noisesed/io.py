"""Loading posterior caches and prediction files."""
import hashlib
import json
import pickle
import zipfile

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[str(r["clip_id"])] = r
    return rows


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for cid in sorted(rows):
            f.write(json.dumps(rows[cid], ensure_ascii=False) + "\n")


def write_submission_zip(jsonl_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl_path, arcname="predictions.jsonl")


def json_events(row):
    out = []
    for e in row.get("events", []):
        a, b = float(e["onset"]), float(e["offset"])
        if b > a:
            out.append((a, b))
    return sorted(out)


def _clip_record(r, frame):
    p = np.asarray(r.get("frame", r.get("prob", [])), np.float32).reshape(-1)
    on = np.asarray(r.get("onset", r.get("onset_dist", [])), np.float32).reshape(-1)
    off = np.asarray(r.get("offset", r.get("offset_dist", [])), np.float32).reshape(-1)
    t = min(len(p), len(on), len(off))
    dur = float(r.get("dur", r.get("duration", t * frame)))
    return dict(frame=p[:t], onset=on[:t], offset=off[:t], dur=dur)


def load_posteriors(path, frame=0.010):
    """Detector output cache -> {clip_id: dict(frame, onset, offset, dur)}.

    Accepts either a list of records with a clip_id field or a dict keyed by
    clip id. Validation records may also carry 'gt' and 'synthetic'.
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)
    items = raw.items() if isinstance(raw, dict) else ((r["clip_id"], r) for r in raw)
    cache, refs, is_syn = {}, {}, {}
    for cid, r in items:
        cid = str(cid)
        rec = _clip_record(r, frame)
        cache[cid] = rec
        if "gt" in r:
            dur = rec["dur"]
            gt = []
            for ev in r["gt"]:
                a = min(max(float(ev[0]), 0.0), dur)
                b = min(max(float(ev[1]), 0.0), dur)
                if b > a:
                    gt.append((a, b))
            refs[cid] = sorted(gt)
        is_syn[cid] = bool(r.get("synthetic", False))
    return cache, refs, is_syn
