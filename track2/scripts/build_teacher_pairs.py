"""Recover (noisy, target) pairs from a teacher submission zip.

    python scripts/build_teacher_pairs.py --inputs data/phase1_test \
        --teacher-zip submission_Track2_V23_RECOVER_BALANCED_a8dfcafd.zip \
        --out pairs/teacher

Writes noisy/ and clean/ WAVs plus manifest.jsonl, and prints how many clips
were kept or rejected and why.
"""
import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from denoise.audio import read_wav, write_wav  # noqa: E402
from denoise.data import recover_teacher_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="folder of the original noisy WAVs")
    ap.add_argument("--teacher-zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    inputs = {p.stem: p for p in Path(args.inputs).glob("*.wav")}
    counts, rows = Counter(), []
    with zipfile.ZipFile(args.teacher_zip) as z:
        names = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".wav")}
        common = sorted(set(names) & set(inputs))
        counts["common"] = len(common)
        for cid in common:
            x = read_wav(inputs[cid])
            target, why = recover_teacher_pairs(x, read_wav(z.read(names[cid])))
            counts[why] += 1
            if target is None:
                continue
            npath, cpath = out / "noisy" / f"{cid}.wav", out / "clean" / f"{cid}.wav"
            write_wav(npath, x)
            write_wav(cpath, target)
            rows.append(dict(pair_id="p1_" + cid, noisy_path=str(npath), clean_path=str(cpath)))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(dict(counts))


if __name__ == "__main__":
    main()
