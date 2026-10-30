"""Re-implementation of the Track 1 scorer (event F1 + segment Dice).

Matching follows the competition rule: a prediction hits a reference event when
both onset and offset are within max(0.2 * ref_duration, 0.05 s). Pairs are
assigned greedily in order of total boundary error, one-to-one.
"""
import numpy as np

from . import FRAME


def matches(pred, ref):
    tol = max(0.20 * (ref[1] - ref[0]), 0.05)
    return abs(pred[0] - ref[0]) <= tol and abs(pred[1] - ref[1]) <= tol


def boundary_error(pred, ref):
    return abs(pred[0] - ref[0]) + abs(pred[1] - ref[1])


def match_events(pred, ref):
    """Greedy one-to-one matching. Returns list of (pred_idx, ref_idx)."""
    cands = [
        (boundary_error(p, r), i, j)
        for i, p in enumerate(pred)
        for j, r in enumerate(ref)
        if matches(p, r)
    ]
    cands.sort()
    used_p, used_r, pairs = set(), set(), []
    for _, i, j in cands:
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        pairs.append((i, j))
    return pairs


def event_counts(preds, refs, ids):
    tp = fp = fn = 0
    for cid in ids:
        p, r = preds.get(cid, []), refs[cid]
        hit = len(match_events(p, r))
        tp += hit
        fp += len(p) - hit
        fn += len(r) - hit
    return tp, fp, fn


def _mask(events, n):
    m = np.zeros(n, dtype=bool)
    for on, off in events:
        a = max(0, int(round(on / FRAME)))
        b = min(n, int(round(off / FRAME)))
        if b > a:
            m[a:b] = True
    return m


def segment_dice(preds, refs, durs, ids):
    vals = []
    for cid in ids:
        n = max(1, int(round(durs[cid] / FRAME)))
        p = _mask(preds.get(cid, []), n)
        g = _mask(refs[cid], n)
        den = p.sum() + g.sum()
        vals.append(1.0 if den == 0 else 2 * np.logical_and(p, g).sum() / den)
    return float(np.mean(vals)) if vals else 0.0


def score(preds, refs, durs, ids):
    tp, fp, fn = event_counts(preds, refs, ids)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    dice = segment_dice(preds, refs, durs, ids)
    return dict(f1=f1, precision=prec, recall=rec, tp=tp, fp=fp, fn=fn,
                dice=dice, combined=f1 + dice)


def interval_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0
