# IndoML 2026 Datathon, Track 2: noise event removal (team speechie)

Phase-2 leaderboard entry: submission 994952, rank 12.

| SI-SDR (dB) | ΔWER (%) | ΔWER synthetic | ΔWER raw | Combined |
| --- | --- | --- | --- | --- |
| 9.5599 | 5.2785 | 15.5706 | 0.4275 | 14.8384 |

## What the scored submission does

994952 is residual-template subtraction ("dictionary-all"):

1. Bank. For each of the 695 released synthetic validation pairs, the noise
   is recovered as noisy - g * clean, with g fitted on samples outside the
   labelled events and clipped to [0.8, 1.2]. Residuals that leak outside
   their events (outside/inside RMS > 0.15) are dropped. 2,000 templates of
   0.08-6 s are kept (cap from Phase 1; 207 omitted).
2. Search. Every test clip is cross-correlated with every template at 4 kHz
   (FFT), the best 3 peaks per template are kept, and the top 128 are
   re-scored at 16 kHz within +/-12 samples.
3. Reconstruction. Matches with quality >= 0.55 survive non-maximum
   suppression (IoU 0.65, at most 16). Overlapping matches get their gains
   fitted jointly by block-reweighted least squares. A component is rejected
   if the fit is ill-conditioned, a gain is outside [0.01, 8] or has the wrong
   sign, two interleaved half-fits disagree by more than 0.5, or subtraction
   would remove more than 1.15x the local energy or raise output energy by
   more than 5%. Accepted templates are subtracted; every other sample is
   left bit-exact.

Method and threshold were frozen in Phase 1 on a validation split and were
not changed for Phase 2. No Phase-2 audio contributes a template.

On the 3,178 Phase-2 clips: 400,808 proposals, 8,321 accepted templates,
1,716 clips changed, 1,462 left unchanged.

**This method does not meet the organisers' Phase-2 guidance** (no fixed
noise dictionary, no unchanged clips). We report it because it is what
produced the score. Our general neural enhancer is described below.

## Always-on neural enhancer (not the scored entry)

`denoise/` holds a Demucs dns64 with a clip-level noise conditioner
(FiLM on the LSTM, bounded removal strength), trained in five stages and
applied identically to every clip with a sign ensemble. Its own Phase-2
submission scored SI-SDR 9.39 dB, Combined 8.89. It is the method we would
submit under the guidance. Training used the Phase-1 dictionary outputs on
Phase-1 test audio as offline targets, and the first stage (V32)
synthesised speech from SraVaani's own transcripts of our earlier Phase-2
submission audio. No reference text or organiser labels were used; see the
report.

## Layout

```
dictionary/
  matcher.py          bank building and FFT template search (from the submission notebook)
  reconstruct.py      gain fitting, safety checks, WAV I/O, ASR cache, packaging
denoise/              neural enhancer (model, inference, losses, data)
scripts/
  infer_dictionary.py   reproduces 994952
  infer_neural.py       neural enhancer over a folder of WAVs
  build_teacher_pairs.py, train.py   last neural training stage
notebooks/            submission notebook, Phase-1 system, neural training stages
```

## Reproducing 994952

```
pip install -r requirements.txt
python scripts/infer_dictionary.py \
    --val-root data/validation --val-cache cache/official_val_posteriors_epoch_2.pkl \
    --frozen-bank artifacts/v5/frozen_bank.pkl \
    --frozen-policy artifacts/v6/frozen_before_hold.json \
    --input data/phase2_input_data.zip --out outputs/dict_all \
    --asr-revision <SraVaani commit> --zip submission.zip
```

`frozen_bank.pkl` supplies the matcher settings and `frozen_before_hold.json`
the reconstruction settings, method (block_irls) and threshold (0.55), both
from Phase 1.

## Team

speechie: Anand Kumar Rai (IIT Kharagpur)

## License

MIT, see ../LICENSE
