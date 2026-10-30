"""Final fine-tuning stage (V36): distil teacher pairs into the adaptive Demucs.

    python scripts/train.py --init ckpt/v35_asrft_epoch_03.pt \
        --gold pairs/gold_train.jsonl --teacher pairs/teacher/manifest.jsonl \
        --out ckpt/v36

Epochs 1-2 update the conditioner, FiLM, strength head and Demucs decoder.
From epoch 3 the encoder and LSTM are unfrozen at a very low learning rate.
One checkpoint per epoch; selection happens separately on the dev split.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from denoise.data import RemixDataset  # noqa: E402
from denoise.losses import distillation_loss  # noqa: E402
from denoise.model import build  # noqa: E402

LR_ADAPTER, LR_DECODER, LR_FULL = 2e-5, 1e-6, 2e-7


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def optimizer_for(model, epoch):
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = [model.conditioner, model.film, model.strength]
    for m in adapters + [model.base.decoder]:
        for p in m.parameters():
            p.requires_grad_(True)
    groups = [dict(params=[p for m in adapters for p in m.parameters()], lr=LR_ADAPTER),
              dict(params=list(model.base.decoder.parameters()), lr=LR_DECODER)]
    if epoch >= 3:
        rest = list(model.base.encoder.parameters()) + list(model.base.lstm.parameters())
        for p in rest:
            p.requires_grad_(True)
        groups.append(dict(params=rest, lr=LR_FULL))
    return torch.optim.AdamW(groups, weight_decay=1e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=517)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build(args.init, device)
    ds = RemixDataset(read_manifest(args.gold), read_manifest(args.teacher), args.draws, args.seed)
    print(f"gold {len(ds.gold)} | teacher {len(ds.teacher)} | draws/epoch {args.draws}")
    dl = DataLoader(ds, batch_size=args.batch, num_workers=2, pin_memory=True, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        ds.set_epoch(epoch)
        opt = optimizer_for(model, epoch)
        model.train()
        for m in model.modules():
            if isinstance(m, nn.RNNBase):
                m.flatten_parameters()
        losses = []
        for x, s, n, mask in dl:
            x, s, n, mask = (t.to(device, non_blocking=True) for t in (x, s, n, mask))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss, _ = distillation_loss(model, x, s, n, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 3.0)
            opt.step()
            losses.append(float(loss.detach()))
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   out / f"teacher_epoch_{epoch:02d}.pt")
        print(f"epoch {epoch} loss {np.mean(losses):.4f}")


if __name__ == "__main__":
    main()
