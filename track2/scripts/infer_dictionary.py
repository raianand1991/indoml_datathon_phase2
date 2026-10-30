"""Reproduce leaderboard submission 994952 (dictionary-all, Phase 2).

    python scripts/infer_dictionary.py \
        --val-root data/validation --val-cache cache/official_val_posteriors_epoch_2.pkl \
        --frozen-bank artifacts/v5/frozen_bank.pkl \
        --frozen-policy artifacts/v6/frozen_before_hold.json \
        --input data/phase2_input_data.zip --out outputs/dict_all \
        --asr-revision <SraVaani commit> --zip submission.zip

The bank is rebuilt from the 695 released synthetic validation pairs only.
Every test clip is searched; clips with no accepted match are written back
unchanged. The method and threshold come from --frozen-policy (chosen in
Phase 1) and are not retuned here.
"""
import argparse
import os
import pickle
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dictionary.matcher import (MatcherConfig, ResidualMatcher, build_bank,  # noqa: E402
                                event_list, persist_query, source_group, wave_sha)
from dictionary.reconstruct import (HashASR, ReconConfig, StrictWaveSource,  # noqa: E402
                                    pack_submission, prepared_matches, reconstruct,
                                    sha_json, write_pcm)


def wav_index(root, ids):
    idx = {p.stem: str(p) for p in Path(root).rglob("*.wav") if p.stem in ids}
    missing = set(ids) - set(idx)
    if missing:
        sys.exit(f"{root}: {len(missing)} files missing, e.g. {sorted(missing)[:3]}")
    return idx


def test_ids(source):
    import zipfile
    p = Path(source)
    if p.is_file():
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".wav") and "__MACOSX" not in n]
    else:
        names = [str(q) for q in p.rglob("*.wav")]
    return sorted(Path(n).stem for n in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-root", required=True, help="has syntheticNoiseAudio/ and syntheticCleanRefAudio/")
    ap.add_argument("--val-cache", required=True, help="validation records with gt and synthetic flags")
    ap.add_argument("--frozen-bank", required=True, help="Phase-1 bank pickle; only its matcher config is used")
    ap.add_argument("--frozen-policy", required=True, help="Phase-1 frozen_before_hold.json")
    ap.add_argument("--input", required=True, help="Phase-2 input zip or folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr-revision")
    ap.add_argument("--zip")
    args = ap.parse_args()
    import json

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.frozen_bank, "rb") as f:
        mcfg = MatcherConfig(**pickle.load(f)["cfg"])
    with open(args.frozen_policy, encoding="utf-8") as f:
        frozen = json.load(f)
    rcfg = ReconConfig(**frozen["recon_config"])
    policy = dict(method=str(frozen["method"]), threshold=float(frozen["threshold"]))
    print("policy", policy)

    with open(args.val_cache, "rb") as f:
        raw = pickle.load(f)
    records = raw if isinstance(raw, list) else [dict(v, clip_id=k) for k, v in raw.items()]
    refs = {str(r["clip_id"]): event_list(r["gt"]) for r in records}
    syn = sorted(str(r["clip_id"]) for r in records if bool(r["synthetic"]))
    noisy = wav_index(Path(args.val_root) / "syntheticNoiseAudio", syn)
    clean = wav_index(Path(args.val_root) / "syntheticCleanRefAudio", syn)

    bank = build_bank(syn, refs, noisy, clean, {c: source_group(c) for c in syn}, mcfg)
    matcher = ResidualMatcher(bank, mcfg)
    by_id = {r["tid"]: r for r in bank["records"]}
    print(f"bank: {len(syn)} pairs -> {len(by_id)} templates")

    ids = test_ids(args.input)
    if set(ids) & set(refs):
        sys.exit("test IDs overlap the validation set")
    src = StrictWaveSource(args.input, ids)
    paths, frames, changed = {}, {}, 0
    for i, cid in enumerate(ids, 1):
        y = src.load(cid)
        if wave_sha(y) in matcher.source_hashes:
            sys.exit(f"{cid} duplicates a bank source waveform")
        props = persist_query(matcher, y, out / "queries" / f"{cid}.json")
        z, info = reconstruct(y, prepared_matches(y, props, by_id, rcfg), policy, rcfg)
        changed += bool(info.get("changed"))
        p = out / "wavs" / f"{cid}.wav"
        write_pcm(p, z)
        paths[cid], frames[cid] = str(p), len(y)
        if i % 100 == 0:
            print(f"  {i}/{len(ids)} changed {changed}")
    src.close()
    print(f"{len(ids)} clips, {changed} changed, {len(ids) - changed} unchanged")

    if not args.asr_revision:
        return
    import torch
    from transformers import AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("ARTPARK-IISc/SraVaani-1.0", revision=args.asr_revision,
                                      code_revision=args.asr_revision, trust_remote_code=True,
                                      token=os.environ.get("HF_TOKEN")).to(device).eval()
    asr = HashASR(model, sha_json(dict(repo="ARTPARK-IISc/SraVaani-1.0", revision=args.asr_revision)),
                  out / "asr_cache")
    texts = asr.transcribe(paths)
    if args.zip:
        info = pack_submission(paths, texts, frames, args.zip)
        print("zip", args.zip, info.get("zip_sha256"))


if __name__ == "__main__":
    main()
