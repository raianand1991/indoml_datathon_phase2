"""Run the detector over a folder of WAVs and write a posterior cache.

The output is what scripts/infer.py reads with --test-cache (one record per
clip: clip_id, duration, prob, onset_dist, offset_dist). The validation cache
with ground truth is written by notebooks/01 (cell "AUTO-TUNE DECODER").

    python scripts/extract_posteriors.py --wav-dir data/test \
        --ckpt ckpt/boundary_wavlm_beats_verified_stable_best.pt \
        --beats-ckpt pretrained/BEATs_iter3_plus_AS2M.pt \
        --beats-src unilm/beats --out cache/test_posteriors.pkl
"""
import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noisesed.model import SR, load_stable, model_inputs  # noqa: E402


def read_wav(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR).astype(np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--beats-ckpt", required=True)
    ap.add_argument("--beats-src", required=True, help="directory containing BEATs.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_stable(args.ckpt, args.beats_ckpt, args.beats_src, device)
    wavs = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))
    print(f"{len(wavs)} clips on {device}")

    records, t0 = [], time.time()
    with torch.inference_mode():
        for i, path in enumerate(wavs, 1):
            cid = os.path.splitext(os.path.basename(path))[0]
            y = read_wav(path)
            if not len(y):
                print("empty audio:", cid)
                continue
            mel, yy, n_valid = model_inputs(y)
            t = mel.shape[1]
            xt = torch.from_numpy(mel)[None, None].to(device)
            wt = torch.from_numpy(yy)[None].to(device)
            wl = torch.tensor([n_valid], dtype=torch.long, device=device)
            noise, on, off = model(xt, wt, wl)
            records.append(dict(
                clip_id=cid,
                duration=len(y) / SR,
                prob=torch.sigmoid(noise[0, :t]).float().cpu().numpy(),
                onset_dist=on[0, :t].float().cpu().numpy(),
                offset_dist=off[0, :t].float().cpu().numpy(),
            ))
            if i % 500 == 0:
                print(f"  {i}/{len(wavs)}  {(time.time() - t0) / 60:.1f} min")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(records, f)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
