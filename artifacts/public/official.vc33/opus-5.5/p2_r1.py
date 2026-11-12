import numpy as np
from collections import deque

def _label(mask):
    H, W = mask.shape
    lab = -np.ones((H, W), int); comps = []
    rs, cs = np.nonzero(mask)
    for r0, c0 in zip(rs, cs):
        if lab[r0, c0] >= 0: continue
        k = len(comps); lab[r0, c0] = k; q = [(r0, c0)]; comp = []
        while q:
            r, c = q.pop(); comp.append((r, c))
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and lab[rr, cc] < 0:
                    lab[rr, cc] = k; q.append((rr, cc))
        comps.append(np.array(comp))
    return lab, comps

WALL = (5, 1, 12)
WALLC = (5, 1, 12, 11, 14)

def _center(P):
    m = P.mean(axis=0)
    i = int(np.argmin(((P - m) ** 2).sum(axis=1)))
    return (int(P[i, 0]), int(P[i, 1]))

def _orient(g):
    lab4, _ = _label(g == 4)
    _, cs = _label((g == 11) | (g == 14))
    for P in cs:
        for r, c in P:
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < 64 and 0 <= cc < 64 and lab4[rr, cc] >= 0:
                    return (-dr, -dc)
    return None

def _tf(g, o):
    if o == (1, 0): return g.copy()
    if o == (-1, 0): return g[::-1].copy()
    if o == (0, 1): return g.T.copy()
    return g.T[::-1].copy()

def _back(r, c, o):
    if o == (1, 0): return r, c
    if o == (-1, 0): return 63 - r, c
    if o == (0, 1): return c, r
    return c, 63 - r

def _solve(obs):
    g = np.array(obs, dtype=int).copy()
    g[0, :] = g[1, :]
    s = 2 if (np.all(g[2::2] == g[3::2]) and np.all(g[:, ::2] == g[:, 1::2])) else 1
    o = _orient(g)
    if o is None: return None
    horiz = o[1] != 0; neg = (o[0] + o[1]) < 0
    C = _tf(g, o)
    step = (2 if s == 2 else 3) * s
    off = (6 if (horiz and neg and step == 3) else 4) * s
    _, bcs = _label(C == 9)
    tanks = []
    def find(a, c0, c1):
        for i, t in enumerate(tanks):
            if t[0] == a and t[1] <= c1 and c0 <= t[2]: return i
        return -1
    binfo = []
    for P in bcs:
        br = P[:, 0].max(); bt = P[:, 0].min(); b0 = P[:, 1].min(); b1 = P[:, 1].max()
        binfo.append((b0, b1, bt, _center(P)))
        c0 = b0
        while c0 - 1 >= 0 and C[br, c0-1] not in WALL: c0 -= 1
        c1 = b1
        while c1 + 1 < 64 and C[br, c1+1] not in WALL: c1 += 1
        run = C[br, c0:c1+1]
        bounded = (c0 - 1 >= 0 and C[br, c0-1] in WALL) and (c1 + 1 < 64 and C[br, c1+1] in WALL)
        if not (np.any(run == 0) or bounded): continue
        if br + 1 < 64 and np.any(C[br+1, c0:c1+1] == 0): continue
        if find(br, c0, c1) < 0: tanks.append([br, c0, c1])
    _, kcs = _label(C == 0)
    for P in kcs:
        a = P[:, 0].max(); c0 = P[:, 1].min(); c1 = P[:, 1].max()
        if c1 - c0 < 2: continue
        if find(a, c0, c1) < 0: tanks.append([a, c0, c1])
    tops = []; cols = []
    for a, c0, c1 in tanks:
        cands = sorted(range(c0, c1+1), key=lambda x: abs(x - (c0+c1)/2))
        col = next((x for x in cands if C[a, x] != 9), cands[0]); cols.append(col)
        if C[a, col] not in (0, 9): tops.append(a + 1); continue
        r = a
        while r - 1 >= 0 and C[r-1, col] in (0, 9): r -= 1
        tops.append(r)
    wl = []; fl = []
    for i, (a, c0, c1) in enumerate(tanks):
        lim = None
        for x in (c0 - 1, c1 + 1):
            if 0 <= x < 64 and C[a, x] in WALLC:
                r = a
                while r - 1 >= 0 and C[r-1, x] in WALLC: r -= 1
                lim = r if lim is None else max(lim, r)
        wl.append(lim if lim is not None else 0)
        f = None; r = tops[i] - 1
        while r >= 0:
            if C[r, cols[i]] == 5: f = r + 1; break
            r -= 1
        fl.append(f)
    lab4, _ = _label(C == 4)
    pieces = []; targets = []
    for colr in (11, 14):
        _, cs = _label(C == colr)
        for P in cs:
            att = False
            for r, c in P:
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < 64 and 0 <= cc < 64 and lab4[rr, cc] >= 0: att = True
            pb = P[:, 0].max(); p0 = P[:, 1].min(); p1 = P[:, 1].max()
            if att:
                best = None
                for i, (a, c0, c1) in enumerate(tanks):
                    if c0 <= p0 and p1 <= c1:
                        d = abs(tops[i] - (pb + 1))
                        if best is None or d < best[0]: best = (d, i)
                if best is not None: pieces.append((colr, best[1]))
            else:
                targets.append((colr, pb, p0, p1))
    _, qcs = _label((C == 1) | (C == 12))
    pipes = []
    for P in qcs:
        q0 = P[:, 1].min(); q1 = P[:, 1].max(); r1 = P[:, 0].max()
        L = [i for i, t in enumerate(tanks) if t[2] == q0 - 1 and t[0] >= r1]
        R = [i for i, t in enumerate(tanks) if t[1] == q1 + 1 and t[0] >= r1]
        if L and R:
            l = min(L, key=lambda i: tanks[i][0]); rr = min(R, key=lambda i: tanks[i][0])
            pipes.append((l, rr, r1 + 1, _center(P)))
    acts = []
    for (b0, b1, bt, cen) in binfo:
        bw = b1 - b0 + 1
        ok = [j for j, t in enumerate(tanks) if t[0] >= bt - 1]
        own = [j for j in ok if tanks[j][1] == b0]
        if own:
            src = [j for j in ok if tanks[j][2] == b0 - bw - 1]
        else:
            own = [j for j in ok if tanks[j][1] < b0 <= tanks[j][2]]
            src = [j for j in ok if tanks[j][1] == b0 + 2 * bw]
        if own and src:
            B = min(own, key=lambda j: tanks[j][0]); A = min(src, key=lambda j: tanks[j][0])
            if A != B: acts.append(('b', A, B, cen))
    if not pieces: return None
    pcols = [p[0] for p in pieces]
    def goal(st):
        tp, pt = st
        for k, ti in enumerate(pt):
            a, c0, c1 = tanks[ti]; ok = False
            for (tc, tb, t0, t1) in targets:
                if tc == pcols[k] and (t1 == c0 - 1 or t0 == c1 + 1) and tb == tp[ti] - 1:
                    ok = True; break
            if not ok: return False
        return True
    def succ(st):
        tp, pt = st
        for ac in acts:
            _, A, B, cen = ac
            if tp[A] > tanks[A][0]: continue
            if fl[B] is not None:
                lim = fl[B] + (off if B in pt else 0)
            else:
                lim = wl[B]
            if not tp[B] > lim: continue
            nt = list(tp); nt[A] += step; nt[B] -= step
            yield cen, (tuple(nt), pt)
        for (l, r, lv, cen) in pipes:
            if tp[l] == lv and tp[r] == lv and (l in pt or r in pt):
                npt = tuple(r if x == l else (l if x == r else x) for x in pt)
                yield cen, (tp, npt)
    st0 = (tuple(tops), tuple(p[1] for p in pieces))
    if goal(st0): return None
    prev = {st0: None}; dq = deque([st0]); cnt = 0
    while dq and cnt < 150000:
        st = dq.popleft(); cnt += 1
        for cen, ns in succ(st):
            if ns in prev: continue
            prev[ns] = (st, cen)
            if goal(ns):
                cur = ns
                while prev[cur][0] != st0:
                    cur = prev[cur][0]
                first = prev[cur][1]
                return _back(first[0], first[1], o)
            dq.append(ns)
    return None

def act(obs, available, memory):
    try:
        res = _solve(obs)
    except Exception:
        res = None
    if res is not None:
        return ("ACTION6", int(res[0]), int(res[1]))
    g = np.array(obs, dtype=int)
    _, bcs = _label(g == 9)
    if bcs:
        k = memory.get('k', 0); memory['k'] = k + 1
        r, c = _center(bcs[k % len(bcs)])
        return ("ACTION6", r, c)
    return ("ACTION6", 32, 32)
