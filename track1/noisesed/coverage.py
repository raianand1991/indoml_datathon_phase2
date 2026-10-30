"""How many reference events could a perfect selector recover from the proposals?

existence:     each reference event asks whether any candidate matches it
max matching:  maximum one-to-one assignment (Hopcroft-Karp). This is the real
               ceiling, since one candidate cannot be credited for two events.
"""
from collections import deque

from .metrics import boundary_error, match_events, matches
from .proposals import is_pair_source


def hopcroft_karp(adj, n_left, n_right):
    nil = -1
    pair_u, pair_v = [nil] * n_left, [nil] * n_right
    dist = [0] * n_left

    def bfs():
        q, found = deque(), False
        for u in range(n_left):
            if pair_u[u] == nil:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == nil:
                    found = True
                elif dist[pu] < 0:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == nil or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u], pair_v[v] = v, u
                return True
        dist[u] = -1
        return False

    size = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == nil and dfs(u):
                size += 1
    return size


def _events(cands, mode):
    out = []
    for z in cands:
        if mode == "pairs" and not is_pair_source(z["source"].split("|")):
            continue
        out.append((z["onset"], z["offset"]))
    return out


def coverage(cands_by_clip, refs, ids, mode="union"):
    """mode: 'pairs' or 'union'."""
    total = exist = matched = 0
    for cid in ids:
        cand, gt = _events(cands_by_clip[cid], mode), refs[cid]
        total += len(gt)
        exist += sum(any(matches(c, g) for c in cand) for g in gt)
        adj = [[j for j, g in enumerate(gt) if matches(c, g)] for c in cand]
        matched += hopcroft_karp(adj, len(cand), len(gt))
    mm = matched / total if total else 0.0
    return dict(gt_events=total,
                existence=exist / total if total else 0.0,
                max_matching=mm,
                f1_ceiling=2 * mm / (1 + mm) if mm else 0.0)


def add_only_oracle(base_pred, cands_by_clip, refs, ids):
    """Keep every baseline event; for each missed reference add the closest
    non-baseline candidate that matches it."""
    out = {}
    for cid in ids:
        base, gt = list(base_pred[cid]), refs[cid]
        hit = {j for _, j in match_events(base, gt)}
        used, pred = set(), list(base)
        for j, g in enumerate(gt):
            if j in hit:
                continue
            elig = [
                (boundary_error((z["onset"], z["offset"]), g), k)
                for k, z in enumerate(cands_by_clip[cid])
                if "baseline" not in z["source"].split("|")
                and matches((z["onset"], z["offset"]), g)
            ]
            if elig:
                _, k = min(elig)
                if k not in used:
                    used.add(k)
                    z = cands_by_clip[cid][k]
                    pred.append((z["onset"], z["offset"]))
        out[cid] = sorted(pred)
    return out
