"""Speaker-grouped splits of the released validation set.

Clip ids look like <...>_<...>_<...>_<...>_<...>_<group>_<...>. Hashing the
group field keeps all clips of a speaker in one bucket.
"""
import hashlib

SPLIT_BUCKETS = {
    "TRAIN": {0, 1, 2, 3},
    "TUNE": {4, 5},
    "SELECT": {6, 7},
    "HOLD": {8, 9},
}


def group_key(cid):
    parts = str(cid).split("_")
    return parts[5] if len(parts) > 6 else str(cid)


def bucket(cid):
    return int(hashlib.md5(group_key(cid).encode()).hexdigest(), 16) % 10


def make_splits(ids):
    out = {name: [] for name in SPLIT_BUCKETS}
    for cid in sorted(ids):
        b = bucket(cid)
        for name, members in SPLIT_BUCKETS.items():
            if b in members:
                out[name].append(cid)
    return out
