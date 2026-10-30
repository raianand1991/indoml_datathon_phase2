"""High-recall candidate intervals from detector posteriors.

Three sources, tagged on each candidate:
  region_<thr>  connected regions at several posterior thresholds, edges snapped
  pair_<thr>    for regions at a few thresholds, every combination of the 4 best
                onset minima near the region start and 4 best offset minima near
                its end
  global_pair   onset x offset combinations over the strongest local minima of
                the whole clip; this is what recovers split and overlapping events
"""
import numpy as np
from scipy.signal import find_peaks

from . import FRAME
from .decode import SNAP_SEC, regions, snap

REGION_THRESHOLDS = [0.08, 0.12, 0.16, 0.20, 0.25, 0.325, 0.40, 0.50, 0.60]
PAIR_THRESHOLDS = [0.08, 0.12, 0.20, 0.325]
PAIR_WIN_SEC = 0.40
PEAK_MIN_DIST_SEC = 0.04
MAX_GLOBAL_MINIMA = 18
MIN_DUR, MAX_DUR = 0.050, 6.0
MAX_PER_CLIP = 180
PAIR_MIN_MAX_POST = 0.20
PAIR_MIN_MEAN_POST = 0.05

SOURCE_NAMES = (
    [f"region_{t:.3f}" for t in REGION_THRESHOLDS]
    + [f"pair_{t:.3f}" for t in PAIR_THRESHOLDS]
    + ["global_pair"]
)


def is_pair_source(sources):
    return any(s.startswith("pair_") or s == "global_pair" for s in sources)


def near_same(a, b, sec=0.02):
    return abs(a[0] - b[0]) <= sec and abs(a[1] - b[1]) <= sec


def _local_minima(x, max_n=MAX_GLOBAL_MINIMA):
    x = np.asarray(x, float)
    if not len(x):
        return []
    dist = max(1, int(round(PEAK_MIN_DIST_SEC / FRAME)))
    peaks, _ = find_peaks(-x, distance=dist)
    if not len(peaks):
        peaks = np.arange(len(x))
    ranked = sorted(peaks, key=lambda i: float(x[i])) + [int(np.argmin(x)), 0, len(x) - 1]
    out = []
    for i in ranked:
        if int(i) not in out:
            out.append(int(i))
        if len(out) >= max_n:
            break
    return sorted(out)


def _nearest_minima(x, center, win_sec, k=4):
    win = max(1, int(round(win_sec / FRAME)))
    lo, hi = max(0, center - win), min(len(x), center + win + 1)
    if hi <= lo:
        return [max(0, min(len(x) - 1, center))]
    order = np.argsort(x[lo:hi])[:k]
    return sorted({lo + int(o) for o in order})


def interval_stats(rec, s, e):
    p = rec["frame"]
    a = max(0, min(len(p) - 1, int(round(s / FRAME))))
    b = max(a + 1, min(len(p), int(round(e / FRAME))))
    seg = p[a:b]
    if not len(seg):
        return 0.0, 0.0
    return float(seg.max()), float(seg.mean())


class _Store:
    def __init__(self, rec):
        self.rec = rec
        self.items = {}

    def add(self, s, e, source, require_support=False):
        dur = self.rec["dur"]
        s, e = min(max(float(s), 0.0), dur), min(max(float(e), 0.0), dur)
        if not MIN_DUR <= e - s <= MAX_DUR:
            return
        mx, mean = interval_stats(self.rec, s, e)
        if require_support and (mx < PAIR_MIN_MAX_POST or mean < PAIR_MIN_MEAN_POST):
            return
        key = (round(s, 3), round(e, 3))
        if key in self.items:
            z = self.items[key]
            z["source"] = "|".join(sorted(set(z["source"].split("|")) | {source}))
            z["seed_score"] = max(z["seed_score"], mx + mean)
        else:
            self.items[key] = dict(onset=s, offset=e, source=source, seed_score=mx + mean)


def _dedupe(cands, max_n=MAX_PER_CLIP):
    # Pair proposals win under cap pressure. Note that a baseline event that
    # sits within 20 ms of a pair proposal is dropped here, so per-source
    # coverage numbers for 'baseline' are not meaningful after this step.
    def priority(z):
        src = z["source"].split("|")
        return (-int(is_pair_source(src)), -z["seed_score"], -len(src), z["onset"])

    kept = []
    for z in sorted(cands, key=priority):
        ev = (z["onset"], z["offset"])
        if any(near_same(ev, (k["onset"], k["offset"])) for k in kept):
            continue
        kept.append(z)
        if len(kept) >= max_n:
            break
    return sorted(kept, key=lambda z: (z["onset"], z["offset"]))


def generate(rec, baseline_events):
    p, on, off, dur = rec["frame"], rec["onset"], rec["offset"], rec["dur"]
    store = _Store(rec)
    snap_win = max(1, int(round(SNAP_SEC / FRAME)))

    for s, e in baseline_events:
        store.add(s, e, "baseline")

    for thr in REGION_THRESHOLDS:
        for a, b in regions(p >= thr):
            s = snap(on, a, snap_win) * FRAME
            e = min(dur, snap(off, b, snap_win) * FRAME)
            store.add(s, e, f"region_{thr:.3f}")
            if thr in PAIR_THRESHOLDS:
                for x in _nearest_minima(on, a, PAIR_WIN_SEC):
                    for y in _nearest_minima(off, max(a + 1, b - 1), PAIR_WIN_SEC):
                        if y > x:
                            store.add(x * FRAME, min(dur, y * FRAME), f"pair_{thr:.3f}",
                                      require_support=True)

    ons, offs = _local_minima(on), _local_minima(off)
    for x in ons:
        for y in offs:
            if y > x:
                store.add(x * FRAME, min(dur, y * FRAME), "global_pair", require_support=True)

    return _dedupe(list(store.items.values()))
