import numpy as np
from collections import deque

FLOOR = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14]
MVS = [('ACTION1', 0, -1), ('ACTION2', 0, 1), ('ACTION3', -1, 0), ('ACTION4', 1, 0)]


def _comps(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                q = [(r, c)]; seen[r, c] = True; pts = []
                while q:
                    a, b = q.pop(); pts.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < H and 0 <= nb < W and mask[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True; q.append((na, nb))
                out.append(pts)
    return out


def _checker(g):
    pad = np.full((g.shape[0] + 2, g.shape[1] + 2), -1); pad[1:-1, 1:-1] = g
    C = g
    orth = (pad[:-2, 1:-1] != C) & (pad[2:, 1:-1] != C) & (pad[1:-1, :-2] != C) & (pad[1:-1, 2:] != C)
    diag = (pad[:-2, :-2] == C) | (pad[:-2, 2:] == C) | (pad[2:, :-2] == C) | (pad[2:, 2:] == C)
    return orth & diag & ~np.isin(C, [0, 3, 4, 5, 6, 7])


def act(obs, available, memory):
    g = np.array(obs, dtype=int)
    rows = [r for r in range(63) if not np.all(g[r] == 3)]
    if not rows:
        return ('ACTION1', None, None)
    y0, y1 = rows[0], rows[-1] + 1
    px = 64
    for c in range(8, 63):
        if np.all(np.isin(g[y0:y1, c], [0, 5])):
            px = c; break
    play = g[y0:y1, :px]
    ys, xs = np.where(play == 14)
    if len(xs) == 0:
        return ('ACTION1', None, None)
    s = max(1, int(xs.max() - xs.min() + 1))
    P = (int(xs.min()), int(ys.min()) + y0)
    gy, gx = np.where(play == 11)
    G = (int(gx.min()), int(gy.min()) + y0) if len(gx) else None
    bar = int(np.sum(g[63] == 3))
    pan = g[y0:y1, px:]
    btns = []
    for comp in _comps(~np.isin(pan, [0, 5, 3])):
        if len(comp) < 4:
            continue
        cols = [int(pan[a, b]) for a, b in comp]
        col = max(set(cols), key=cols.count)
        my = sum(a for a, b in comp) / len(comp); mx = sum(b for a, b in comp) / len(comp)
        rep = min(comp, key=lambda p: (p[0] - my) ** 2 + (p[1] - mx) ** 2)
        btns.append(((col, int(my + y0) // 4, int(mx + px) // 4), col, rep[0] + y0, rep[1] + px))
    btns.sort()
    par = memory.setdefault('par', {})
    patch = memory.setdefault('patch', {})
    unsafe = memory.setdefault('unsafe', set())
    bad = memory.setdefault('bad', set())
    vis = memory.setdefault('vis', set())
    la = memory.get('last')
    prev = memory.get('prev')
    if la is not None and prev is not None:
        if la[0] == 'ACTION6':
            bid, ppos = la[2], la[1]
            if bar - memory.get('bar', bar) >= 4:
                unsafe.add((bid, ppos, la[3]))
            else:
                p0 = par.get(bid, 0)
                if bid not in patch:
                    M = (prev[:63] != g[:63]) & (prev[:63] != 14) & (g[:63] != 14)
                    M[:, px:] = False
                    full = np.zeros_like(g, dtype=bool); full[:63] = M
                    a, b = prev[full], g[full]
                    patch[bid] = (full, a, b) if p0 == 0 else (full, b, a)
                par[bid] = p0 ^ 1
        elif la[1] == P:
            bad.add(la[2])
    vis.add(P)
    memory['prev'] = g.copy(); memory['bar'] = bar
    known = [b for b in btns if b[0] in patch]
    unknown = [b for b in btns if b[0] not in patch]
    curpt = tuple(par.get(b[0], 0) for b in known)
    cache = {}

    def info(pt):
        if pt in cache:
            return cache[pt]
        f = g.copy()
        for i, b in enumerate(known):
            M, V0, V1 = patch[b[0]]
            if pt[i] != curpt[i]:
                f[M] = V1 if pt[i] else V0
        chk = _checker(f); chk[:, px:] = False
        cb = np.zeros_like(chk)
        for dy in range(s):
            for dx in range(s):
                cb[:64 - dy, :64 - dx] |= chk[dy:, dx:]
        W = np.isin(f, FLOOR) & ~cb
        W[:y0] = False; W[y1 - s + 1:] = False; W[:, px - s + 1:] = False
        tele = []
        for (x, y) in zip(*np.where(np.isin(f[:, :px], [6, 7]))[::-1]):
            blk = f[y:y + 2, x:x + 2]
            if blk.shape == (2, 2) and np.all(np.isin(blk, [6, 7])) and len(set(blk.flatten())) == 2 \
                    and (x, y) not in tele and (x - 1, y) not in tele and (x, y - 1) not in tele:
                tele.append((int(x), int(y)))
        cache[pt] = (f, W, tele)
        return cache[pt]

    def walk(pos, pt):
        x, y = pos
        if not (0 <= x < 64 and 0 <= y < 64):
            return False
        return bool(info(pt)[1][y, x])

    def bfs(pred, limit=40000):
        st = (P, curpt)
        parent = {st: None}
        dq = deque([st])
        while dq:
            cur = dq.popleft()
            if pred(cur):
                c = cur; a = None
                while parent[c] is not None:
                    c, a = parent[c]
                return a, cur
            if len(parent) > limit:
                break
            pos, pt = cur
            for name, dx, dy in MVS:
                if (pos, name, pt) in bad:
                    continue
                n = ((pos[0] + dx * s, pos[1] + dy * s), pt)
                if n not in parent and walk(*n):
                    parent[n] = (cur, (name, pos, pt)); dq.append(n)
            for i, b in enumerate(known):
                if (b[0], pos, pt) in unsafe:
                    continue
                npt = pt[:i] + (pt[i] ^ 1,) + pt[i + 1:]
                npos = pos
                if b[1] == 6:
                    tl = info(pt)[2]
                    if pos in tl and len(tl) >= 2:
                        npos = [t for t in tl if t != pos][0]
                n = (npos, npt)
                if n not in parent and walk(npos, npt):
                    parent[n] = (cur, ('CLICK', b, pt)); dq.append(n)
        return None, None

    def emit(a):
        if a[0] == 'CLICK':
            b = a[1]
            memory['last'] = ('ACTION6', P, b[0], curpt)
            return ('ACTION6', int(b[2]), int(b[3]))
        memory['last'] = (a[0], P, (P, a[0], curpt))
        return (a[0], None, None)

    if G is not None:
        a, _ = bfs(lambda st: st[0] == G)
        if a is not None:
            return emit(a)
    if unknown:
        def safe(st):
            pos, pt = st
            f = info(pt)[0]
            if pos != P:
                return bool(np.all(f[pos[1]:pos[1] + s, pos[0]:pos[0] + s] == 2))
            return any(int(g[yy, pos[0]]) == 2 for yy in (pos[1] - 1, pos[1] + s) if 0 <= yy < 64)
        a, end = bfs(safe)
        if a is None or end == (P, curpt):
            b = unknown[0]
            memory['last'] = ('ACTION6', P, b[0], curpt)
            return ('ACTION6', int(b[2]), int(b[3]))
        return emit(a)
    chk0 = _checker(g)
    m15 = (g == 15) & ~chk0
    m15[:, px:] = False; m15[:y0] = False; m15[y1:] = False
    pick = set()
    for comp in _comps(m15):
        if len(comp) <= 6:
            for (r, c) in comp:
                pick.add((c, r))
    if pick:
        def pp(st):
            pos = st[0]
            if pos in vis:
                return False
            return any((pos[0] + dx, pos[1] + dy) in pick for dx in range(s) for dy in range(s))
        a, _ = bfs(pp)
        if a is not None:
            return emit(a)
    a, _ = bfs(lambda st: st[0] not in vis)
    if a is not None:
        return emit(a)
    k = memory.get('k', 0); memory['k'] = k + 1
    name = MVS[k % 4][0]
    memory['last'] = (name, P, (P, name, curpt))
    return (name, None, None)
