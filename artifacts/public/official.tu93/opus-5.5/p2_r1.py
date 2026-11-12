import time
from collections import deque
import numpy as np

DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
ACTS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
MARK = {(0, 1): 0, (2, 1): 1, (1, 0): 2, (1, 2): 3}


def parse(obs):
    ys, xs = (obs[:63] == 9).nonzero()
    if len(ys) == 0:
        return None
    py, px = int(ys.min()), int(xs.min())
    pdir = 0
    blk = obs[py:py + 3, px:px + 3]
    for (r, c), d in MARK.items():
        if r < blk.shape[0] and c < blk.shape[1] and blk[r, c] == 4:
            pdir = d
    ox, oy = px % 6, py % 6
    exits, reds, blues, chasers = set(), [], [], []
    edges = set()
    for y in range(oy, 61, 6):
        for x in range(ox, 62, 6):
            b = obs[y:y + 3, x:x + 3]
            if b.shape != (3, 3):
                continue
            vals = set(int(v) for v in b.flatten())
            d = None
            act = False
            for (r, c), dd in MARK.items():
                if b[r, c] in (15, 11):
                    d = dd
                    act = b[r, c] == 11
            if 14 in vals:
                exits.add((x, y))
            elif 8 in vals and d is not None:
                reds.append((x, y, d))
            elif 12 in vals and d is not None:
                blues.append((x, y, d))
            elif 13 in vals and d is not None:
                chasers.append([x, y, d, act])
            for k, (dx, dy) in enumerate(DIRS):
                r, c = y + 3 * dy, x + 3 * dx
                if 0 <= r < 63 and 0 <= c < 64 and obs[r, c] == 2:
                    edges.add((x, y, k))
    ch = []
    for x, y, d, act in chasers:
        if act:
            dx, dy = DIRS[d]
            if (px, py) == (x + 12 * dx, y + 12 * dy):
                q = (d,)
            else:
                q = (pdir,)
            ch.append((x, y, d, True, q))
        else:
            ch.append((x, y, d, False, ()))
    return (px, py), reds, blues, ch, edges, exits


def sim(state, a, edges, exits):
    (px, py), reds, blues, chs = state
    if (px, py, a) not in edges:
        return None, 'inv'
    dx, dy = DIRS[a]
    P = (px + 6 * dx, py + 6 * dy)
    chs = [(x, y, d, ac, (q + (a,)) if ac else q) for (x, y, d, ac, q) in chs]
    reds = [r for r in reds if (r[0], r[1]) != P]
    blues = [b for b in blues if (b[0], b[1]) != P]
    chs = [c for c in chs if (c[0], c[1]) != P]
    for (x, y, d) in reds:
        ex, ey = DIRS[d]
        if (x + 6 * ex, y + 6 * ey) == P:
            return None, 'dead'
    nb = []
    for (x, y, d) in blues:
        ex, ey = DIRS[d]
        nb.append((x + 6 * ex, y + 6 * ey, d))
    nc = []
    for (x, y, d, ac, q) in chs:
        if ac:
            ex, ey = DIRS[d]
            x, y = x + 6 * ex, y + 6 * ey
        nc.append((x, y, d, ac, q))
    for b in nb:
        if (b[0], b[1]) == P:
            return None, 'dead'
    for c in nc:
        if c[3] and (c[0], c[1]) == P:
            return None, 'dead'
    nb2 = []
    for (x, y, d) in nb:
        if (x, y, d) not in edges:
            d = d ^ 1
        nb2.append((x, y, d))
    nc2 = []
    for (x, y, d, ac, q) in nc:
        if not ac:
            ex, ey = DIRS[d]
            if (x + 12 * ex, y + 12 * ey) == P:
                ac = True
                q = (d, d)
        if ac and q:
            d = q[0]
            q = q[1:]
        nc2.append((x, y, d, ac, q))
    ns = (P, tuple(sorted(reds)), tuple(sorted(nb2)), tuple(sorted(nc2)))
    if P in exits:
        return ns, 'win'
    return ns, 'ok'


def act(obs, available, memory):
    t0 = time.time()
    obs = np.asarray(obs)
    p = parse(obs)
    if p is None:
        return ("ACTION1", None, None)
    P, reds, blues, chs, edges, exits = p
    pred = memory.get('pred')
    if pred is not None:
        _, preds_r, preds_b, preds_c = pred
        rpos = set((r[0], r[1]) for r in reds)
        bpos = set((b[0], b[1]) for b in blues)
        cpos = set((c[0], c[1]) for c in chs)
        # hidden reds (under blues/chasers)
        for r in preds_r:
            pos = (r[0], r[1])
            if pos not in rpos and pos != P and (pos in bpos or pos in cpos):
                reds.append(r)
        # hidden blues (under reds/chasers)
        for b in preds_b:
            pos = (b[0], b[1])
            if pos not in bpos and pos != P and (pos in rpos or pos in cpos):
                blues.append(b)
        # chasers: refine queue or restore hidden
        newch = []
        for c in chs:
            m = [pc for pc in preds_c if pc[:4] == c[:4]]
            newch.append(m[0] if m else c)
        chs = newch
        for c in preds_c:
            pos = (c[0], c[1])
            if pos not in cpos and pos != P and (pos in rpos or pos in bpos):
                chs.append(c)
    start = (P, tuple(sorted(set(reds))), tuple(sorted(set(blues))), tuple(sorted(set(chs))))

    q = deque([start])
    first = {start: None}
    found = None
    n = 0
    while q and found is None:
        s = q.popleft()
        n += 1
        if n > 300000 or (n % 2000 == 0 and time.time() - t0 > 7.0):
            break
        for a in range(4):
            ns, st = sim(s, a, edges, exits)
            if st in ('inv', 'dead'):
                continue
            fa = a if first[s] is None else first[s]
            if st == 'win':
                found = fa
                break
            if ns not in first:
                first[ns] = fa
                q.append(ns)
    best = found
    if best is None:
        bd = None
        for a in range(4):
            ns, st = sim(start, a, edges, exits)
            if st in ('inv', 'dead'):
                continue
            PP = ns[0]
            d = min([abs(PP[0] - e[0]) + abs(PP[1] - e[1]) for e in exits] or [0])
            if bd is None or d < bd:
                best, bd = a, d
    if best is None:
        for a in range(4):
            if (P[0], P[1], a) in edges:
                best = a
                break
    if best is None:
        best = 0
    ns, st = sim(start, best, edges, exits)
    if st == 'ok':
        memory['pred'] = ns
    elif st == 'inv':
        memory['pred'] = start
    else:
        memory['pred'] = None
    return (ACTS[best], None, None)
