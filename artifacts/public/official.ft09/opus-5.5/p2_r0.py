import numpy as np
import random
from math import gcd

BG = 4

def _classify(b, bg):
    if (b == b[0, 0]).all():
        v = int(b[0, 0])
        if v == bg:
            return None
        return ('H', v, None)
    if (b == 6).any() and b[1, 1] != 6:
        oth = b[b != 6]
        if (oth == oth[0]).all():
            offs = {(0, 0)}
            for j in range(3):
                for i in range(3):
                    if b[j, i] == 6:
                        offs.add((i - 1, j - 1))
            return ('N', int(b[1, 1]), offs)
    c = int(b[1, 1])
    if c == bg or c in (0, 2, 3):
        return None
    border = []
    for j in range(3):
        for i in range(3):
            if (i, j) == (1, 1):
                continue
            v = int(b[j, i])
            if v not in (0, 1, 2, 3):
                return None
            border.append((i - 1, j - 1, v == 0))
    return ('C', c, border)

def _parse_bg(g, bg):
    H, W = g.shape
    best = {}
    for oy in range(4):
        for ox in range(4):
            cells = {}
            for gy in range(oy, H - 2, 4):
                for gx in range(ox, W - 2, 4):
                    ok = True
                    for yy in range(gy - 1, gy + 4):
                        for xx in range(gx - 1, gx + 4):
                            if gy <= yy <= gy + 2 and gx <= xx <= gx + 2:
                                continue
                            if 0 <= yy < H and 0 <= xx < W and g[yy, xx] != bg:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    r = _classify(g[gy:gy + 3, gx:gx + 3], bg)
                    if r:
                        cells[(gx, gy)] = r
            if len(cells) > len(best):
                best = cells
    return best

def parse(obs):
    g = np.asarray(obs)[::2, ::2]
    cells = _parse_bg(g, BG)
    if not cells:
        vals, cnt = np.unique(g, return_counts=True)
        cells = _parse_bg(g, int(vals[np.argmax(cnt)]))
    return cells

def color_order(cells, nxt):
    C = set()
    for p, (k, c, e) in cells.items():
        C.add(c)
    for a, b in nxt.items():
        C.add(a); C.add(b)
    vals = set(nxt.values())
    starts = sorted(c for c in C if c not in vals) + sorted(c for c in C if c in vals)
    order, used = [], set()
    for c in starts:
        while c is not None and c not in used:
            order.append(c); used.add(c)
            c = nxt.get(c)
    if len(order) < 2:
        order.append(-99)
    return order

def rref(M, n, k):
    r = 0
    piv = []
    for c in range(n):
        pr = None
        for i in range(r, len(M)):
            v = M[i][c] % k
            if v and gcd(v, k) == 1:
                pr = i
                break
        if pr is None:
            continue
        M[r], M[pr] = M[pr], M[r]
        inv = pow(M[r][c] % k, -1, k)
        M[r] = [(x * inv) % k for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] % k:
                f = M[i][c]
                Mr = M[r]
                M[i] = [(a - f * b) % k for a, b in zip(M[i], Mr)]
        piv.append(c)
        r += 1
    for i in range(r, len(M)):
        if M[i][n] % k:
            return None
    return M, piv

def plan(cells, elp_on, nxt, rng):
    order = color_order(cells, nxt)
    k = len(order)
    idx = {c: i for i, c in enumerate(order)}
    tiles = sorted(p for p, v in cells.items() if v[0] in 'HN')
    tset = set(tiles)
    allowed = {}
    for p, (kind, c, border) in cells.items():
        if kind != 'C':
            continue
        ci = idx[c]
        for dx, dy, eq in border:
            q = (p[0] + 4 * dx, p[1] + 4 * dy)
            if q not in tset:
                continue
            S = {ci} if eq else set(range(k)) - {ci}
            allowed[q] = allowed.get(q, set(range(k))) & S
    if not tiles:
        return None, allowed, idx, k
    cons = sorted(allowed)
    for q in cons:
        if not allowed[q]:
            return None, allowed, idx, k
    n = len(tiles)
    col = {p: i for i, p in enumerate(tiles)}
    A = {q: [0] * n for q in cons}
    for p in tiles:
        kind, c, e = cells[p]
        offs = e if kind == 'N' else elp_on
        for dx, dy in offs:
            q = (p[0] + 4 * dx, p[1] + 4 * dy)
            if q in A:
                A[q][col[p]] = 1
    multi = any(len(allowed[q]) > 1 for q in cons)
    tries = 40 if (multi or k == 2) else 1
    best, bestcost = None, None
    for t in range(tries):
        M = []
        for q in cons:
            s = idx[cells[q][1]]
            S = sorted(allowed[q])
            if t == 0:
                v = min(S, key=lambda v: (v - s) % k)
            else:
                v = rng.choice(S)
            M.append(A[q][:] + [(v - s) % k])
        R = rref(M, n, k)
        if R is None:
            continue
        M, piv = R
        free = [j for j in range(n) if j not in set(piv)]
        x = [0] * n
        for j in free:
            x[j] = 0 if t == 0 else rng.randrange(k) if rng.random() < 0.3 else 0
        for i, c in enumerate(piv):
            x[c] = (M[i][n] - sum(M[i][j] * x[j] for j in free)) % k
        cost = sum(x)
        if bestcost is None or cost < bestcost:
            best, bestcost = x, cost
    if best is None:
        return None, allowed, idx, k
    return {tiles[i]: best[i] for i in range(n) if best[i]}, allowed, idx, k

def act(obs, available, memory):
    cells = parse(obs)
    nxt = memory.setdefault('nxt', {})
    elp_on = memory.setdefault('elp_on', {(0, 0)})
    prev = memory.get('prev')
    last = memory.get('last')
    if prev is not None and last is not None:
        D = set()
        for p, (k0, c0, e0) in prev.items():
            if k0 in 'HN' and p in cells and cells[p][0] in 'HN' and cells[p][1] != c0:
                D.add(p)
                nxt[c0] = cells[p][1]
        if prev.get(last, ('X',))[0] == 'H' and D:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (last[0] + 4 * dx, last[1] + 4 * dy)
                    if q in prev and prev[q][0] in 'HN' and q in cells:
                        if q in D:
                            elp_on.add((dx, dy))
                        else:
                            elp_on.discard((dx, dy))
    step = memory.get('step', 0)
    memory['step'] = step + 1
    rng = random.Random(1234 + step)
    tiles = sorted(p for p, v in cells.items() if v[0] in 'HN')
    target = None
    if tiles:
        sol, allowed, idx, k = plan(cells, elp_on, nxt, rng)
        if sol:
            target = sorted(sol, key=lambda p: (-sol[p], p[1], p[0]))[0]
        else:
            bad = [q for q in sorted(allowed)
                   if idx.get(cells[q][1], -1) not in allowed[q]]
            if bad:
                target = bad[step % len(bad)]
            else:
                target = tiles[step % len(tiles)]
    memory['prev'] = cells
    memory['last'] = target
    if target is None:
        return ('ACTION6', 32, 32)
    gx, gy = target
    row = 2 * (gy + 1) + 1
    col = 2 * (gx + 1) + 1
    return ('ACTION6', int(row), int(col))
