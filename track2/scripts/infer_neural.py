"""Enhance every WAV in a folder and write the Codabench submission.

    python scripts/infer_neural.py INPUT_DIR OUTPUT_DIR \
        --checkpoint ckpt/teacher_epoch_01.pt --alpha 0.95 \
        --asr-revision <40-char SraVaani commit> [--zip submission.zip]

Every clip goes through the same model. There is no lookup table, no
natural/synthetic decision and no bypass; the script stops if any output is
bit-identical to its input.
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from denoise.audio import read_wav, to_pcm16, write_wav  # noqa: E402
from denoise.enhance import enhance  # noqa: E402
from denoise.model import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--no-sign-ensemble", action="store_true")
    ap.add_argument("--asr-revision", help="SraVaani commit; omit to skip transcripts")
    ap.add_argument("--zip", help="also write a Codabench zip here")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wavs = sorted(Path(args.input_dir).glob("*.wav"))
    if not wavs:
        sys.exit(f"no WAVs in {args.input_dir}")

    model = build(args.checkpoint, device)
    print(f"{len(wavs)} clips | alpha {args.alpha} | sign ensemble {not args.no_sign_ensemble} | {device}")

    hashes, unchanged = {}, []
    for i, p in enumerate(wavs, 1):
        x = read_wav(p)
        z = enhance(model, x, alpha=args.alpha, sign_ensemble=not args.no_sign_ensemble, device=device)
        if np.array_equal(to_pcm16(x), to_pcm16(z)):
            unchanged.append(p.stem)
        hashes[p.stem] = write_wav(out / p.name, z)
        if i % 250 == 0:
            print(f"  enhanced {i}/{len(wavs)}")
    if unchanged:
        sys.exit(f"{len(unchanged)} outputs identical to input, e.g. {unchanged[:3]}")

    if not args.asr_revision:
        print("skipping transcripts (no --asr-revision)")
        return

    from denoise.asr import SraVaani
    model.cpu()
    torch.cuda.empty_cache()
    asr = SraVaani(args.asr_revision, device, token=os.environ.get("HF_TOKEN"))
    rows = []
    for i, p in enumerate(wavs, 1):
        rows.append(dict(clip_id=p.stem, text=asr(out / p.name)))
        if i % 250 == 0:
            print(f"  transcribed {i}/{len(wavs)}")
    tr = out / "transcripts.jsonl"
    with open(tr, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(tr, "transcripts.jsonl")
            for p in wavs:
                z.write(out / p.name, p.name)
        with zipfile.ZipFile(args.zip) as z:
            assert z.testzip() is None
            assert set(z.namelist()) == {"transcripts.jsonl"} | {p.name for p in wavs}
        print("zip:", args.zip, hashlib.sha256(Path(args.zip).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
