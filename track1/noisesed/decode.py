"""Threshold-and-snap decoder for the three-head detector."""
import numpy as np

from . import FRAME

THRESHOLD = 0.325
MIN_DUR = 0.060
SNAP_SEC = 0.20


def regions(mask):
    """Runs of True in a boolean array as half-open (start, end) frame pairs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def snap(curve, idx, win):
    """Index of the minimum of `curve` within +/- win frames of idx."""
    lo, hi = max(0, idx - win), min(len(curve), idx + win + 1)
    if hi <= lo:
        return idx
    return lo + int(np.argmin(curve[lo:hi]))


def decode(rec, thr=THRESHOLD, min_dur=MIN_DUR, snap_sec=SNAP_SEC):
    """Posterior regions above thr, edges moved to the distance-curve minima."""
    p, on, off, dur = rec["frame"], rec["onset"], rec["offset"], rec["dur"]
    win = max(1, int(round(snap_sec / FRAME)))
    events = []
    for a, b in regions(p >= thr):
        s = max(0.0, snap(on, a, win) * FRAME)
        e = min(dur, snap(off, b, win) * FRAME)
        if e - s >= min_dur:
            events.append((s, e))
    return sorted(events)
