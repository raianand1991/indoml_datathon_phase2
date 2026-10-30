"""Strict 16 kHz mono WAV I/O and the competition metrics."""
import hashlib
import io
import math
import re
import unicodedata
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from . import SR


def read_wav(path_or_bytes, sr=SR):
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    y, fs = sf.read(src, dtype="float32", always_2d=True)
    if fs != sr or y.shape[1] != 1:
        raise ValueError(f"expected {sr} Hz mono, got {fs} Hz / {y.shape[1]} ch")
    y = y[:, 0]
    if not len(y) or not np.isfinite(y).all():
        raise ValueError("empty or non-finite audio")
    return y


def to_pcm16(y):
    a = np.asarray(y, np.float64)
    return np.clip(np.rint(a * 32768.0), -32768, 32767).astype("<i2")


def write_wav(path, y, sr=SR):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(to_pcm16(y).tobytes())
    data = buf.getvalue()
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def si_sdr(est, ref):
    x = np.asarray(est, np.float64)
    s = np.asarray(ref, np.float64)
    if x.shape != s.shape:
        raise ValueError("length mismatch")
    x, s = x - x.mean(), s - s.mean()
    ss = float(s @ s)
    if ss <= 1e-18:
        raise ValueError("silent reference")
    target = (float(x @ s) / ss) * s
    err = x - target
    val = 10 * math.log10(max(float(target @ target), ss * 1e-16) / max(float(err @ err), ss * 1e-12))
    return min(120.0, val)


def tokens(text):
    text = re.sub(r"<[^>]*>|\[[^\]]*\]", " ", str(text or ""))
    text = unicodedata.normalize("NFC", text).lower()
    return "".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split()


def word_errors(ref, hyp):
    """(errors, reference words) by Levenshtein over tokens."""
    a, b = tokens(ref), tokens(hyp)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j - 1] + (a[i - 1] != b[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1], len(a)
