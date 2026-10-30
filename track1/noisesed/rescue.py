"""Add-only rescue of events the baseline decoder missed.

A candidate is a positive example only if it matches a reference event that
the baseline did NOT already match. The decoder can add events but never drops
or moves a baseline event.

Bug history: the competition run (notebooks/04_rescue_v34_1.ipynb) left
`matched_missed_count` in the feature matrix. That column is computed from the
labels, so every split showed AUC 1.0, and at test time it was filled with
zeros, which made the model reject every candidate. LABEL_COLUMNS below is the
guard against that; predict() also refuses to fill missing columns.
"""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from . import FRAME
from .decode import THRESHOLD, regions
from .metrics import interval_iou, match_events, matches
from .proposals import SOURCE_NAMES, is_pair_source, near_same

LABEL_COLUMNS = {"label", "matched_missed_count", "base_sample_weight"}
ID_COLUMNS = {"cid", "cand_idx", "split", "onset", "offset"}
BASELINE_DUP_SEC = 0.03


def _q(x, q):
    return float(np.quantile(x, q)) if len(x) else 0.0


def _baseline_relation(ev, base):
    if not base:
        return dict(base_max_iou=0.0, base_overlap_count=0, near_base_on=99.0,
                    near_base_off=99.0, near_base_both=99.0, inside_base=0,
                    contains_base=0, duration_ratio_near=0.0)
    ious = [interval_iou(ev, b) for b in base]
    d_on = [abs(ev[0] - b[0]) for b in base]
    d_off = [abs(ev[1] - b[1]) for b in base]
    both = [x + y for x, y in zip(d_on, d_off)]
    k = int(np.argmin(both))
    return dict(
        base_max_iou=max(ious),
        base_overlap_count=sum(x > 0 for x in ious),
        near_base_on=min(d_on),
        near_base_off=min(d_off),
        near_base_both=min(both),
        inside_base=int(any(ev[0] >= b[0] and ev[1] <= b[1] for b in base)),
        contains_base=int(any(ev[0] <= b[0] and ev[1] >= b[1] for b in base)),
        duration_ratio_near=(ev[1] - ev[0]) / max(base[k][1] - base[k][0], 1e-4),
    )


def features(rec, cand, base, n_cands):
    p, on, off, dur = rec["frame"], rec["onset"], rec["offset"], rec["dur"]
    t = len(p)
    s, e = float(cand["onset"]), float(cand["offset"])
    a = max(0, min(t - 1, int(round(s / FRAME))))
    b = max(a + 1, min(t, int(round(e / FRAME))))
    b_idx = min(t - 1, b)

    seg = p[a:b]
    bg = np.r_[p[max(0, a - 25):a], p[b:min(t, b + 25)]]
    if not len(bg):
        bg = p
    w = max(1, int(round(0.15 / FRAME)))
    on_win = on[max(0, a - w):min(t, a + w + 1)]
    off_win = off[max(0, b_idx - w):min(t, b_idx + w + 1)]
    src = set(cand["source"].split("|"))

    row = dict(
        duration=e - s,
        duration_frac=(e - s) / max(dur, 1e-6),
        onset_frac=s / max(dur, 1e-6),
        offset_frac=e / max(dur, 1e-6),
        p_max=float(seg.max()) if len(seg) else 0.0,
        p_mean=float(seg.mean()) if len(seg) else 0.0,
        p_min=float(seg.min()) if len(seg) else 0.0,
        p_q10=_q(seg, 0.10), p_q25=_q(seg, 0.25), p_q50=_q(seg, 0.50),
        p_q75=_q(seg, 0.75), p_q90=_q(seg, 0.90),
        bg_mean=float(bg.mean()),
        bg_q90=_q(bg, 0.90), bg_q95=_q(bg, 0.95),
        contrast_mean=float(seg.mean() - bg.mean()) if len(seg) else 0.0,
        contrast_q90=_q(seg, 0.90) - _q(bg, 0.95) if len(seg) else 0.0,
        occupancy_above_0325=float(np.mean(seg >= 0.325)) if len(seg) else 0.0,
        occupancy_above_050=float(np.mean(seg >= 0.50)) if len(seg) else 0.0,
        strong_islands=len(regions(seg >= THRESHOLD)) if len(seg) else 0,
        onset_value=float(on[a]),
        offset_value=float(off[b_idx]),
        onset_local_min=float(on_win.min()) if len(on_win) else float(on[a]),
        offset_local_min=float(off_win.min()) if len(off_win) else float(off[b_idx]),
        onset_global_pct=float(np.mean(on <= on[a])),
        offset_global_pct=float(np.mean(off <= off[b_idx])),
        seed_score=float(cand.get("seed_score", 0.0)),
        source_count=len(src),
        is_pair=int(any(x.startswith("pair_") for x in src)),
        is_global_pair=int("global_pair" in src),
        is_region=int(any(x.startswith("region_") for x in src)),
        candidate_count=n_cands,
        baseline_count=len(base),
        **_baseline_relation((s, e), base),
    )
    for name in SOURCE_NAMES:
        row["src_" + name] = int(name in src)
    return row


def rescue_rows(rec, cands, base):
    """Non-baseline candidates for one clip, with their index into `cands`."""
    out = []
    for j, z in enumerate(cands):
        if "baseline" in z["source"].split("|"):
            continue
        ev = (z["onset"], z["offset"])
        if any(near_same(ev, b, BASELINE_DUP_SEC) for b in base):
            continue
        out.append((j, z))
    return out


def build_table(cache, cands_by_clip, base_pred, ids, refs=None, split_of=None):
    """Feature table. With refs, also attaches labels and sample weights."""
    rows = []
    for cid in ids:
        base = base_pred[cid]
        props = rescue_rows(cache[cid], cands_by_clip[cid], base)
        if refs is not None:
            hit = {j for _, j in match_events(base, refs[cid])}
            missed = [g for g in range(len(refs[cid])) if g not in hit]
            matched = {
                j: [g for g in missed if matches((z["onset"], z["offset"]), refs[cid][g])]
                for j, z in props
            }
            per_gt = Counter(g for ms in matched.values() for g in ms)

        for j, z in props:
            r = features(cache[cid], z, base, len(props))
            r.update(cid=cid, cand_idx=j, onset=z["onset"], offset=z["offset"])
            if split_of is not None:
                r["split"] = split_of[cid]
            if refs is not None:
                ms = matched[j]
                r["label"] = int(bool(ms))
                r["matched_missed_count"] = len(ms)
                # one GT with many near-duplicate candidates should count once
                r["base_sample_weight"] = max(1.0 / per_gt[g] for g in ms) if ms else 1.0
            rows.append(r)
    return pd.DataFrame(rows)


def feature_columns(df):
    cols = [c for c in df.columns if c not in LABEL_COLUMNS | ID_COLUMNS]
    assert not LABEL_COLUMNS & set(cols)
    return cols


def _matrix(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"feature columns missing at predict time: {missing}")
    x = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return x.to_numpy(np.float32)


class RescueModel:
    def __init__(self, seed=42):
        self.seed = seed
        self.cols = None
        self.hgb = HistGradientBoostingClassifier(
            learning_rate=0.045, max_iter=260, max_leaf_nodes=31,
            min_samples_leaf=24, l2_regularization=0.8, random_state=seed)
        self.et = ExtraTreesClassifier(
            n_estimators=500, max_depth=18, min_samples_leaf=3, max_features=0.75,
            class_weight="balanced", n_jobs=-1, random_state=seed)

    def fit(self, df):
        self.cols = feature_columns(df)
        x, y = _matrix(df, self.cols), df["label"].to_numpy(np.int64)
        w = df["base_sample_weight"].to_numpy(np.float64)
        pos, neg = max(1, y.sum()), max(1, len(y) - y.sum())
        boost = min(25.0, neg / pos)
        self.hgb.fit(x, y, sample_weight=np.where(y == 1, w * boost, w))
        self.et.fit(x, y, sample_weight=w)
        return self

    def predict(self, df):
        x = _matrix(df, self.cols)
        return 0.6 * self.hgb.predict_proba(x)[:, 1] + 0.4 * self.et.predict_proba(x)[:, 1]


def scored_candidates(model, df, cands_by_clip, ids):
    out = {cid: [] for cid in ids}
    if not len(df):
        return out
    probs = model.predict(df)
    for cid, j, prob in zip(df["cid"], df["cand_idx"], probs):
        z = cands_by_clip[cid][int(j)]
        out[cid].append(dict(event=(z["onset"], z["offset"]), prob=float(prob),
                             is_pair=is_pair_source(z["source"].split("|"))))
    return out


def add_only_decode(base, scored, policy):
    """Add up to policy['max_add'] candidates above threshold. Baseline untouched."""
    added = []
    for z in sorted(scored, key=lambda z: z["prob"], reverse=True):
        if z["prob"] < policy["threshold"]:
            break
        if policy["source_mode"] == "pairs_only" and not z["is_pair"]:
            continue
        ev = z["event"]
        if any(near_same(ev, x, policy["dup_sec"]) for x in list(base) + added):
            continue
        added.append(ev)
        if len(added) >= policy["max_add"]:
            break
    return sorted(list(base) + added)
