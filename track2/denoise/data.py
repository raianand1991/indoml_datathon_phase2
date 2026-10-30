"""Training pairs and the dynamic remixing dataset.

Two pair pools feed training:

gold     released validation synthetic pairs (noisy, clean), train split only
teacher  (Phase-1 noisy input, Phase-1 dictionary-system output) for clips the
         teacher actually changed; see recover_teacher_pairs()

Each draw is one of: identity (clean in, clean out), a direct pair crop
centred on noise activity, or a fresh mixture of one speech target with 1-3
residuals (noisy - clean) taken from random pairs at a random SNR.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import read_wav, to_pcm16

SEGMENT = 51200
IDENTITY_P = 0.05
DIRECT_P = 0.45
GOLD_SHARE_OF_DIRECT = 0.35
SNR_RANGE = (-8.0, 18.0)


def recover_teacher_pairs(noisy, teacher):
    """Turn one (input, teacher output) clip into a training pair, or None.

    Rejects clips the teacher left unchanged, gain mismatches outside
    [0.4, 2.5], near-silent targets, and residuals that correlate with the
    target (|corr| > 0.75), which would mean the teacher removed speech.
    """
    if len(noisy) != len(teacher):
        return None, "length_mismatch"
    if np.array_equal(to_pcm16(noisy), to_pcm16(teacher)):
        return None, "unchanged"
    x, y = noisy.astype(np.float64), teacher.astype(np.float64)
    gain = float(x @ y / (y @ y + 1e-12))
    if not 0.40 <= gain <= 2.50:
        return None, "rejected_gain"
    target = gain * y
    resid = x - target
    if np.sqrt(np.mean(resid ** 2)) < 2e-5:
        return None, "gain_only"
    if np.sqrt(np.mean(target ** 2)) < 1e-4:
        return None, "silent_target"
    corr = abs(resid @ target) / (np.linalg.norm(resid) * np.linalg.norm(target) + 1e-12)
    if corr > 0.75:
        return None, "rejected_correlation"
    return target.astype(np.float32), "accepted"


def _crop(y, start, length=SEGMENT):
    z = y[start:start + length]
    return np.pad(z, (0, max(0, length - len(z))))[:length].astype(np.float32)


def _random_crop(y, rng, length=SEGMENT):
    start = int(rng.integers(0, len(y) - length + 1)) if len(y) > length else 0
    return _crop(y, start, length)


def _event_crop(x, s, rng):
    """Crop a window that contains noise activity, so direct pairs are not
    mostly speech-only."""
    if len(x) <= SEGMENT:
        return _crop(x, 0), _crop(s, 0)
    energy = np.convolve((x - s).astype(np.float64) ** 2, np.ones(400) / 400, mode="same")
    active = np.flatnonzero(energy > np.quantile(energy, 0.75))
    anchor = int(rng.choice(active)) if len(active) else int(rng.integers(0, len(x)))
    start = max(0, min(len(x) - SEGMENT, anchor - int(rng.integers(0, SEGMENT + 1))))
    return _crop(x, start), _crop(s, start)


class RemixDataset(Dataset):
    def __init__(self, gold, teacher, draws=4000, seed=517):
        self.gold, self.teacher = list(gold), list(teacher)
        self.records = self.gold + self.teacher
        self.draws, self.seed, self.epoch = draws, seed, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.draws

    @staticmethod
    def _load(r):
        return read_wav(r["noisy_path"]), read_wav(r["clean_path"])

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + i * 9176)
        u = rng.random()
        if u < IDENTITY_P:
            _, s = self._load(self.records[rng.integers(len(self.records))])
            s = _random_crop(s, rng)
            x, n = s.copy(), np.zeros_like(s)
        elif u < IDENTITY_P + DIRECT_P:
            pool = self.gold if rng.random() < GOLD_SHARE_OF_DIRECT else self.teacher
            xx, ss = self._load(pool[rng.integers(len(pool))])
            x, s = _event_crop(xx, ss, rng)
            n = x - s
        else:
            _, speech = self._load(self.records[rng.integers(len(self.records))])
            s = _random_crop(speech, rng)
            n = np.zeros_like(s)
            k = int(rng.choice([1, 2, 3], p=[0.55, 0.30, 0.15]))
            s_rms = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)) + 1e-6)
            for _ in range(k):
                nx, ns = self._load(self.records[rng.integers(len(self.records))])
                piece = np.roll(_random_crop(nx - ns, rng), int(rng.integers(0, SEGMENT)))
                if rng.random() < 0.5:
                    piece = -piece
                p_rms = float(np.sqrt(np.mean(piece.astype(np.float64) ** 2)) + 1e-7)
                snr = rng.uniform(*SNR_RANGE)
                n += piece * np.float32(s_rms / (p_rms * 10 ** (snr / 20)) / k ** 0.5)
            x = s + n
            peak = float(max(np.abs(x).max(), np.abs(s).max(), 1e-8))
            if peak > 0.98:
                x, s, n = x * 0.98 / peak, s * 0.98 / peak, n * 0.98 / peak

        smooth = np.convolve(n.astype(np.float64) ** 2, np.ones(400) / 400, mode="same")
        mask = (smooth > max(1e-8, float(np.quantile(smooth, 0.60)) * 0.20)).astype(np.float32)
        return tuple(torch.from_numpy(z.astype(np.float32)) for z in (x, s, n, mask))
