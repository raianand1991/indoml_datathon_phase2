"""Fit the rescue classifier on TRAIN and pick the add-only policy on TUNE.

SELECT and HOLD are only reported. They never feed back into the policy.

    python scripts/train_rescue.py --val-cache cache/val_posteriors.pkl --out models/
"""
import argparse
import itertools
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noisesed.metrics import score  # noqa: E402
from noisesed.rescue import (RescueModel, add_only_decode, build_table,  # noqa: E402
                             scored_candidates)
from noisesed.validation import load_natural_val  # noqa: E402

SOURCE_MODES = ["pairs_only", "all_nonbaseline"]
MAX_ADDS = [1, 2, 3, 4, 6]
DUP_SECS = [0.03, 0.05, 0.08]
FIXED_THR = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
             0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]


def decode_all(base, scored, ids, policy):
    return {c: add_only_decode(base[c], scored[c], policy) for c in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = load_natural_val(args.val_cache)
    table = build_table(d.cache, d.cands, d.base, d.ids, refs=d.refs, split_of=d.split_of)
    print(table.groupby("split")["label"].agg(["count", "sum", "mean"]))

    model = RescueModel(seed=args.seed).fit(table[table.split == "TRAIN"])
    print(f"{len(model.cols)} features")

    for split in d.splits:
        g = table[table.split == split]
        y, p = g["label"].to_numpy(), model.predict(g)
        auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
        ap_ = average_precision_score(y, p) if len(set(y)) > 1 else float("nan")
        print(f"{split:6s} AUC {auc:.4f}  PR-AUC {ap_:.4f}  prevalence {y.mean():.4f}")
        if split != "TRAIN" and auc > 0.995:
            raise RuntimeError(f"AUC {auc:.4f} on {split}: check for label leakage")

    scored = scored_candidates(model, table, d.cands, d.ids)
    tune = d.splits["TUNE"]
    tune_probs = np.array([z["prob"] for c in tune for z in scored[c]])
    thresholds = np.unique(np.clip(
        np.r_[FIXED_THR, np.quantile(tune_probs, np.linspace(0.50, 0.995, 36))],
        0.01, 0.995))

    grid = []
    for mode, mx, dup, thr in itertools.product(SOURCE_MODES, MAX_ADDS, DUP_SECS, thresholds):
        pol = dict(source_mode=mode, max_add=mx, dup_sec=dup, threshold=float(thr))
        grid.append({**pol, **score(decode_all(d.base, scored, tune, pol), d.refs, d.durs, tune)})
    grid = pd.DataFrame(grid).sort_values(["f1", "combined", "precision"], ascending=False)
    grid.to_csv(Path(args.out) / "tune_policy_grid.csv", index=False)

    best = grid.iloc[0]
    policy = dict(source_mode=str(best.source_mode), max_add=int(best.max_add),
                  dup_sec=float(best.dup_sec), threshold=float(best.threshold))
    print("policy (TUNE only):", policy)

    report = []
    for split in ["TUNE", "SELECT", "HOLD"]:
        ids = d.splits[split]
        b = score(d.base, d.refs, d.durs, ids)
        v = score(decode_all(d.base, scored, ids, policy), d.refs, d.durs, ids)
        report += [{"split": split, "system": "baseline", **b},
                   {"split": split, "system": "rescue", **v}]
    report = pd.DataFrame(report)
    print(report.round(4).to_string(index=False))
    report.to_csv(Path(args.out) / "val_rescue.csv", index=False)

    with open(Path(args.out) / "rescue_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(Path(args.out) / "policy.json", "w") as f:
        json.dump(policy, f, indent=2)


if __name__ == "__main__":
    main()
