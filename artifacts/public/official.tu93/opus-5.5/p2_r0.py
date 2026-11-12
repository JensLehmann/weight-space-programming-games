import numpy as np
import time
from collections import deque

D = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPP = [1, 0, 3, 2]
ACTS = ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4']
SUBDIR = {(0, 1): 0, (2, 1): 1, (1, 0): 2, (1, 2): 3}


def parse(obs):
    o = obs[:63]
    ys, xs = np.where((o == 9) | (o == 4))
    if len(ys) == 0:
        return None
    y0, x0 = int(ys.min()), int(xs.min())
    h = int(ys.max()) - y0 + 1
    w = int(xs.max()) - x0 + 1
    s = max(1, int(round(max(h, w) / 3.0)))
    c = 3 * s
    P = 2 * c
    pf = 0
    y4, x4 = np.where(o == 4)
    if len(y4):
        pf = SUBDIR.get(((int(y4.min()) - y0) // s, (int(x4.min()) - x0) // s), 0)

    def blk(y, x):
        ya, yb = max(0, y), min(63, y + c)
        xa, xb = max(0, x), min(64, x + c)
        if ya >= yb or xa >= xb:
            return None, ya, xa
        return o[ya:yb, xa:xb], ya, xa

    nodes = {}
    enemies = []
    exitp = None
    for i in range(-(y0 // P) - 1, (63 - y0) // P + 2):
        for j in range(-(x0 // P) - 1, (64 - x0) // P + 2):
            y = y0 + i * P
            x = x0 + j * P
            b, ya, xa = blk(y, x)
            if b is None:
                continue
            vals = set(np.unique(b).tolist())
            kind = None
            if 9 in vals or 4 in vals:
                kind = 'P'
            elif 14 in vals:
                kind = 'X'
            elif 8 in vals:
                kind = 'R'
            elif 12 in vals:
                kind = 'B'
            elif 13 in vals:
                kind = 'M'
            elif 0 in vals:
                kind = 'N'
            if kind is None:
                continue
            nodes[(i, j)] = kind
            if kind == 'X':
                exitp = (i, j)
            if kind in 'RBM':
                act = 11 in vals
                mk = np.where((b == 15) | (b == 11))
                d = 0
                if len(mk[0]):
                    yy = int(mk[0][0]) + ya - y
                    xx = int(mk[1][0]) + xa - x
                    d = SUBDIR.get((yy // s, xx // s), 0)
                enemies.append((kind, i, j, d, act, pf if (kind == 'M' and act) else d))
    edges = {}
    for (i, j) in nodes:
        es = set()
        for k, (di, dj) in enumerate(D):
            if (i + di, j + dj) not in nodes:
                continue
            b, _, _ = blk(y0 + i * P + di * c, x0 + j * P + dj * c)
            if b is not None and (b == 2).any():
                es.add(k)
        edges[(i, j)] = es
    for p in list(edges):
        for k in list(edges[p]):
            q = (p[0] + D[k][0], p[1] + D[k][1])
            edges.setdefault(q, set()).add(OPP[k])
    return (0, 0), tuple(enemies), edges, exitp


def sim(state, d, edges, exitp):
    p, ens = state
    if d not in edges.get(p, ()):
        return None, None
    q = (p[0] + D[d][0], p[1] + D[d][1])
    kept = [e for e in ens if (e[1], e[2]) != q]
    for e in kept:
        if e[0] == 'R' and (e[1] + D[e[3]][0], e[2] + D[e[3]][1]) == q:
            return 'dead', None
    new = []
    for e in kept:
        t, i, j, dr, act, nx = e
        if t == 'B':
            i += D[dr][0]; j += D[dr][1]
            if (i, j) == q:
                return 'dead', None
            if dr not in edges.get((i, j), ()):
                dr = OPP[dr]
        elif t == 'M':
            if act:
                i += D[dr][0]; j += D[dr][1]
                if (i, j) == q:
                    return 'dead', None
                dr = nx
                nx = d
            else:
                if (i + 2 * D[dr][0], j + 2 * D[dr][1]) == q:
                    act = True
                    nx = dr
        new.append((t, i, j, dr, act, nx))
    ns = (q, tuple(new))
    if q == exitp:
        return 'win', ns
    return 'ok', ns


def key(state):
    p, ens = state
    return (p, tuple(sorted(tuple(e[:5]) for e in ens)))


def norm(st):
    q, es = st
    return ((0, 0), tuple((e[0], e[1] - q[0], e[2] - q[1]) + tuple(e[3:]) for e in es))


def bfs(start, edges, exitp, tlim=6.0):
    t0 = time.time()
    if exitp is None:
        return None
    par = {key(start): None}
    dq = deque([start])
    while dq:
        if time.time() - t0 > tlim:
            return None
        st = dq.popleft()
        for d in range(4):
            r, ns = sim(st, d, edges, exitp)
            if r is None or r == 'dead':
                continue
            k = key(ns)
            if k in par:
                continue
            par[k] = (st, d, ns)
            if r == 'win':
                acts, states = [], []
                cur = k
                while par[cur] is not None:
                    pst, dd, nst = par[cur]
                    acts.append(dd)
                    states.append(nst)
                    cur = key(pst)
                return acts[::-1], states[::-1]
            dq.append(ns)
    return None


def merge(obs_state, pred):
    p, oens = obs_state
    oens = list(oens)
    pens = pred[1]
    out = []
    for o in oens:
        m = None
        for e in pens:
            if tuple(e[:5]) == tuple(o[:5]):
                m = e
                break
        out.append(tuple(m) if m is not None else tuple(o))
    occ = set((o[1], o[2]) for o in oens)
    for e in pens:
        if not any(tuple(o[:4]) == tuple(e[:4]) for o in oens) and (e[1], e[2]) in occ:
            out.append(tuple(e))
    return (p, tuple(out))


def act(obs, available, memory):
    obs = np.asarray(obs)
    parsed = parse(obs)
    if parsed is None:
        return ('ACTION1', None, None)
    p, ens, edges, exitp = parsed
    sobs = (p, ens)
    pred = memory.get('pred')
    state = sobs
    match = False
    if pred is not None:
        merged = merge(sobs, pred)
        if key(merged) == key(pred):
            state = pred
            match = True
        else:
            state = merged
    plan = memory.get('plan')
    preds = memory.get('preds')
    if match and plan and preds:
        memory['plan'] = plan[1:]
        memory['pred'] = preds[0]
        memory['preds'] = preds[1:]
        return (ACTS[plan[0]], None, None)
    res = bfs(state, edges, exitp)
    if res:
        acts, states = res
        rel = [norm(st) for st in states]
        memory['plan'] = acts[1:]
        memory['pred'] = rel[0]
        memory['preds'] = rel[1:]
        return (ACTS[acts[0]], None, None)
    memory['plan'] = None
    memory['preds'] = None
    best, bd, bns = None, None, None
    for d in range(4):
        r, ns = sim(state, d, edges, exitp)
        if r is None:
            continue
        if r == 'win':
            memory['pred'] = None
            return (ACTS[d], None, None)
        score = 1000 if r == 'dead' else 0
        q = (p[0] + D[d][0], p[1] + D[d][1])
        if exitp is not None:
            score += abs(q[0] - exitp[0]) + abs(q[1] - exitp[1])
        if bd is None or score < bd:
            bd, best, bns = score, d, ns
    if best is None:
        memory['pred'] = None
        return ('ACTION1', None, None)
    memory['pred'] = norm(bns) if bns is not None else None
    return (ACTS[best], None, None)
