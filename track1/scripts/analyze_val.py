"""Baseline, proposal coverage and ADD-only oracle on the validation Natural clips.

    python scripts/analyze_val.py --val-cache cache/val_posteriors.pkl --out results/
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noisesed.coverage import add_only_oracle, coverage  # noqa: E402
from noisesed.metrics import score  # noqa: E402
from noisesed.validation import load_natural_val  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = load_natural_val(args.val_cache)
    oracle = add_only_oracle(d.base, d.cands, d.refs, d.ids)

    rows = []
    for split, ids in d.splits.items():
        b = score(d.base, d.refs, d.durs, ids)
        o = score(oracle, d.refs, d.durs, ids)
        pairs = coverage(d.cands, d.refs, ids, "pairs")
        union = coverage(d.cands, d.refs, ids, "union")
        rows.append(dict(
            split=split, clips=len(ids), gt_events=union["gt_events"],
            base_recall=b["recall"], base_f1=b["f1"], base_combined=b["combined"],
            pairs_max_matching=pairs["max_matching"],
            union_existence=union["existence"], union_max_matching=union["max_matching"],
            oracle_f1=o["f1"], oracle_combined=o["combined"],
        ))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    table.to_csv(Path(args.out) / "val_natural_analysis.csv", index=False)


if __name__ == "__main__":
    main()
