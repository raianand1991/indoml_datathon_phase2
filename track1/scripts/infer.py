"""Build the Track 1 submission.

Synthetic-routed clips are copied from --base unchanged. Natural-routed clips
keep every event in --base, plus whatever the rescue model adds (only if
--model and --policy are given).

    # reproduce the leaderboard file (no rescue)
    python scripts/infer.py --base preds/base_predictions.jsonl \
        --router preds/router_psyn.jsonl --out submission/

    # with the rescue stage
    python scripts/infer.py --base preds/base_predictions.jsonl \
        --router preds/router_psyn.jsonl --test-cache cache/test_posteriors.pkl \
        --model models/rescue_model.pkl --policy models/policy.json --out submission/
"""
import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noisesed.io import (json_events, load_posteriors, read_jsonl,  # noqa: E402
                         sha256_file, write_jsonl, write_submission_zip)
from noisesed.proposals import generate  # noqa: E402
from noisesed.rescue import add_only_decode, build_table, scored_candidates  # noqa: E402

GATE = 0.425


def as_event_list(events):
    return [{"onset": round(float(a), 4), "offset": round(float(b), 4)} for a, b in events]


def interval_set(row):
    return {(round(float(e["onset"]), 4), round(float(e["offset"]), 4))
            for e in row.get("events", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="predictions JSONL with both branches applied")
    ap.add_argument("--router", required=True, help="JSONL with clip_id and p_syn")
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--test-cache", help="detector posteriors for the test clips")
    ap.add_argument("--model")
    ap.add_argument("--policy")
    ap.add_argument("--expect-sha", help="refuse to run if --base hash differs")
    ap.add_argument("--out", default="submission")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base_sha = sha256_file(args.base)
    if args.expect_sha and base_sha != args.expect_sha:
        sys.exit(f"base predictions hash {base_sha} != expected {args.expect_sha}")

    base = read_jsonl(args.base)
    router = read_jsonl(args.router)
    assert set(base) == set(router), "base and router cover different clips"

    natural = sorted(c for c in base if float(router[c]["p_syn"]) < args.gate)
    synthetic = sorted(set(base) - set(natural))
    print(f"{len(base)} clips: {len(natural)} natural, {len(synthetic)} synthetic")

    final = copy.deepcopy(base)
    n_added = 0

    use_rescue = bool(args.model and args.policy)
    if use_rescue:
        if not args.test_cache:
            sys.exit("--test-cache is needed with --model")
        cache, _, _ = load_posteriors(args.test_cache)
        missing = [c for c in natural if c not in cache]
        assert not missing, f"{len(missing)} natural clips missing from test cache"
        with open(args.model, "rb") as f:
            model = pickle.load(f)
        with open(args.policy) as f:
            policy = json.load(f)

        nat_base = {c: json_events(base[c]) for c in natural}
        cands = {}
        for i, c in enumerate(natural, 1):
            cands[c] = generate(cache[c], nat_base[c])
            if i % 500 == 0:
                print(f"  proposals {i}/{len(natural)}")
        table = build_table(cache, cands, nat_base, natural)
        scored = scored_candidates(model, table, cands, natural)
        for c in natural:
            events = add_only_decode(nat_base[c], scored[c], policy)
            n_added += len(events) - len(nat_base[c])
            final[c]["events"] = as_event_list(events)
        print(f"rescue added {n_added} events to "
              f"{sum(len(final[c]['events']) > len(base[c]['events']) for c in natural)} clips")

    changed = [c for c in synthetic if final[c].get("events") != base[c].get("events")]
    assert not changed, f"{len(changed)} synthetic rows changed"
    lost = [c for c in natural if not interval_set(base[c]) <= interval_set(final[c])]
    assert not lost, f"{len(lost)} natural rows lost an original event"

    out = Path(args.out)
    write_jsonl(final, out / "predictions.jsonl")
    write_submission_zip(out / "predictions.jsonl", out / "submission.zip")
    manifest = dict(
        base=str(args.base), base_sha256=base_sha,
        clips=len(final), natural=len(natural), synthetic=len(synthetic), gate=args.gate,
        rescue=use_rescue, rescue_added=n_added,
        predictions_sha256=sha256_file(out / "predictions.jsonl"),
    )
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
