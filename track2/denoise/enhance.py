"""Chunked, sign-ensembled inference. The same path runs on every clip."""
import numpy as np
import torch

CHUNK = 51200   # 3.2 s
HOP = 25600     # 50% overlap


@torch.inference_mode()
def enhance(model, y, alpha=0.95, sign_ensemble=True, device="cpu"):
    """Return y - alpha * predicted_noise, overlap-added with a Hann window.

    With sign_ensemble the model also sees -y and the two noise estimates are
    averaged as 0.5 * (n(y) - n(-y)); this cancels any polarity bias the
    network picked up in training.
    """
    y = np.asarray(y, np.float32)
    out = np.zeros(len(y), np.float64)
    den = np.zeros(len(y), np.float64)
    weight = np.maximum(np.hanning(CHUNK), 1e-3)

    for start in range(0, len(y), HOP):
        count = min(CHUNK, len(y) - start)
        seg = np.pad(y[start:start + count], (0, CHUNK - count))
        xt = torch.from_numpy(seg)[None, None].to(device)
        _, n1, _, _ = model(xt, return_aux=True)
        if sign_ensemble:
            _, n2, _, _ = model(-xt, return_aux=True)
            noise = 0.5 * (n1 - n2)
        else:
            noise = n1
        z = (xt - float(alpha) * noise)[0, 0].float().cpu().numpy()
        if not np.isfinite(z).all():
            raise FloatingPointError("non-finite enhancement output")
        out[start:start + count] += z[:count] * weight[:count]
        den[start:start + count] += weight[:count]
        if start + CHUNK >= len(y):
            break

    if (den <= 0).any():
        raise RuntimeError("some samples were not covered by any chunk")
    return (out / den).astype(np.float32)
