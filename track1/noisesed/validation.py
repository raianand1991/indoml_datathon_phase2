"""Shared setup for scripts that work on the released validation set."""
from dataclasses import dataclass, field

from .decode import decode
from .io import load_posteriors
from .proposals import generate
from .splits import make_splits


@dataclass
class ValData:
    cache: dict
    refs: dict
    durs: dict
    ids: list
    splits: dict
    base: dict = field(default_factory=dict)
    cands: dict = field(default_factory=dict)

    @property
    def split_of(self):
        return {c: name for name, ids in self.splits.items() for c in ids}


def load_natural_val(path, with_candidates=True, log=print):
    cache, refs, is_syn = load_posteriors(path)
    ids = sorted(c for c in cache if not is_syn[c] and c in refs)
    d = ValData(cache=cache, refs=refs, durs={c: cache[c]["dur"] for c in cache},
                ids=ids, splits=make_splits(ids))
    log(f"natural clips {len(ids)}, events {sum(len(refs[c]) for c in ids)}")
    log("split sizes " + ", ".join(f"{k}={len(v)}" for k, v in d.splits.items()))
    d.base = {c: decode(cache[c]) for c in ids}
    if with_candidates:
        for i, c in enumerate(ids, 1):
            d.cands[c] = generate(cache[c], d.base[c])
            if i % 500 == 0:
                log(f"  proposals {i}/{len(ids)}")
    return d
