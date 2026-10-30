# IndoML 2026 Datathon, Track 1: noise event detection (team speechie)

Code for our Track 1 entry. Hidden-test score: Combined 1.4979 (Natural F1
0.4354, Synthetic F1 0.9258).

The pipeline in short:

1. Frozen WavLM-base-plus and BEATs features, plus log-mel, go into a CNN +
   BiGRU with three per-frame outputs: noise posterior, distance to nearest
   onset, distance to nearest offset. Trained only on the verified-timestamp
   clips (`notebooks/01`, `noisesed/model.py`).
2. Decode: threshold the posterior at 0.325, snap each edge to the minimum of
   its distance curve within 200 ms, drop events under 60 ms
   (`noisesed/decode.py`).
3. A WavLM + IndicWav2Vec2 classifier gives `p_syn` per clip. Clips with
   `p_syn >= 0.425` are treated as synthetic (`notebooks/02`).
4. Synthetic clips: the organisers' synthetic data reuses a finite set of
   noise recordings. We rebuild that set from the validation noisy/clean pairs
   and find each template in the test clip by normalised cross-correlation,
   which gives near sample-exact boundaries (`notebooks/03`).
5. Natural clips: keep the detector output. An add-only rescue stage was
   built for these (`noisesed/rescue.py`), but it did not change the
   submitted file. See "Rescue stage" below.

The report covers the reasoning and the validation analysis.

## Layout

```
noisesed/
  model.py        detector definition (mel CNN + WavLM + BEATs -> BiGRU, three heads)
  metrics.py      scorer (event F1 with the official matching rule, segment Dice)
  decode.py       threshold + snap decoder
  proposals.py    high-recall candidate intervals from the posteriors
  coverage.py     proposal coverage via Hopcroft-Karp, ADD-only oracle
  rescue.py       features, labels, classifier and add-only decoder
  splits.py       speaker-grouped TRAIN/TUNE/SELECT/HOLD buckets
  io.py           caches, JSONL, zip
scripts/
  extract_posteriors.py  detector over a folder of WAVs -> posterior cache
  analyze_val.py  baseline / coverage / oracle table on validation Natural clips
  train_rescue.py fit on TRAIN, pick policy on TUNE, report SELECT and HOLD
  infer.py        builds predictions.jsonl and submission.zip
notebooks/        Colab notebooks for the detector, router and template branch
```

## Setup

```
pip install -r requirements.txt
```

The scripts need only numpy, pandas, scipy and scikit-learn. torch and
transformers are needed for `noisesed/model.py` and the notebooks. BEATs comes
from the microsoft/unilm source tree with the `BEATs_iter3_plus_AS2M.pt`
checkpoint.

## Reproducing the submission

The notebooks write the files the scripts start from:

| file | from | contents |
| --- | --- | --- |
| `cache/val_posteriors.pkl` | 01 (`official_val_posteriors_epoch_2.pkl`) | list of dicts: clip_id, prob, onset_dist, offset_dist, duration, gt, synthetic |
| `cache/test_posteriors.pkl` | `scripts/extract_posteriors.py` | same, without gt / synthetic |
| `preds/router_psyn.jsonl` | 02 | one line per test clip: clip_id, p_syn |
| `preds/base_predictions.jsonl` | 03 | full submission with both branches applied |

Test posteriors from the trained checkpoint:

```
python scripts/extract_posteriors.py --wav-dir data/test \
    --ckpt ckpt/boundary_wavlm_beats_verified_stable_best.pt \
    --beats-ckpt pretrained/BEATs_iter3_plus_AS2M.pt \
    --beats-src unilm/beats --out cache/test_posteriors.pkl
```

Then:

```
python scripts/infer.py \
    --base preds/base_predictions.jsonl \
    --router preds/router_psyn.jsonl \
    --expect-sha bee50cc35a306ca46482e092bd842328ae8b39040e004e6a9baee0a2a47a0cfa \
    --out submission/
```

This is the leaderboard file. `--expect-sha` makes the script stop if the base
predictions are not the exact file we submitted. The output
`predictions.jsonl` should have the same hash.

`infer.py` always checks two things before writing the zip: no synthetic row
differs from `--base`, and every natural row still contains all of its
original events.

## Validation analysis

```
python scripts/analyze_val.py --val-cache cache/val_posteriors.pkl --out results/
```

On the 1,806 natural validation clips, the decoder recalls about 32% of
reference events. The candidate proposals can cover about 63% one-to-one, and
adding the best-matching candidate for each missed event (no new false
positives) takes HOLD F1 from 0.350 to 0.586. Detection, not boundary
placement, is what holds natural F1 down.

Coverage numbers for "baseline only" or "regions only" are not reported: the
deduplication step in `proposals.py` prefers pair candidates, so baseline-tagged
candidates get absorbed and a per-source count would be misleading.

## Rescue stage

```
python scripts/train_rescue.py --val-cache cache/val_posteriors.pkl --out models/
python scripts/infer.py --base preds/base_predictions.jsonl --router preds/router_psyn.jsonl \
    --test-cache cache/test_posteriors.pkl \
    --model models/rescue_model.pkl --policy models/policy.json --out submission_rescue/
```

The version we ran during the competition (`notebooks/04_rescue_v34_1.ipynb`)
had a label leak: `matched_missed_count` was left in the feature list. It
reported AUC 1.0 everywhere, and at test time the missing column was filled
with zeros, so no events were added and the submitted file was identical to
the base. The code here drops that column, asserts that no label column is
used as a feature, refuses to zero-fill missing features, and stops if
held-out AUC is above 0.995.

We have not rerun the fixed version on the full validation set, so we make no
claim about its effect.

One known mismatch: during training the "baseline" a candidate is compared
against is our local decoder output, but at test time it is the submitted
natural events. The baseline-relation features may shift between the two.

## Splits

Validation natural clips are bucketed by MD5 of the sixth `_`-separated field
of the clip id (the speaker/group field), mod 10: buckets 0-3 TRAIN, 4-5 TUNE,
6-7 SELECT, 8-9 HOLD. Everything tunable in the rescue stage is chosen on TUNE.

## Team

speechie: Anand Kumar Rai (IIT Kharagpur) [add other members]

## License

MIT, see ../LICENSE
