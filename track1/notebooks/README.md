Colab notebooks, in the order they were run. Paths at the top of each point at
our Drive layout (`/content/drive/MyDrive/Datathon/work`); change `WORK` if you
run them elsewhere.

| notebook | produces |
| --- | --- |
| 01_detector_wavlm_beats.ipynb | trains the detector on the 11,101 verified-timestamp clips (cell "STABLE TRAINING CELL", the second one; the first hit NaN losses), saves `ckpt/boundary_wavlm_beats_verified_stable_best.pt` (best dev loss at epoch 2), writes `official_val_posteriors_epoch_2.pkl` and tunes the decoder on 1,500 validation clips. Its last cell writes `submission_WavLM_BEATs_tuned.zip` but does not save test posteriors; use `scripts/extract_posteriors.py` for those. |
| 02_router_wavlm_indicw2v.ipynb | natural/synthetic classifier; writes `p_syn` per test clip |
| 03_synthetic_templates.ipynb | template bank from validation noisy-minus-clean pairs, cross-correlation search on synthetic-routed test clips; writes the base predictions JSONL |
| 04_rescue_v34_1.ipynb | the rescue run as submitted, outputs kept. This is the version with the `matched_missed_count` leak (see the cell 8 output: AUC 1.0 on every split). The fixed code is in `noisesed/rescue.py`. |
