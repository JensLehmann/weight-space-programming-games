import numpy as np
from collections import deque

def _parse(obs):
    a0 = np.array(obs, dtype=int)
    m = a0 != 3
    m[:, 0] = False; m[1, :] = False
    rows = np.where(m.any(1))[0]; cols = np.where(m.any(0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
    cands = []
    for s in range(1, 9):
        for W in range(1, 64 // s + 1):
            ox = (64 - W * s) // 2
            if ox + W * s - 1 != x1 or not (ox == x0 or (x0 == 1 and ox == 0)):
                continue
            for H in range(1, 64 // s + 1):
                if min(64 // W, 64 // H) != s:
                    continue
                oy = (64 - H * s) // 2
                if oy + H * s - 1 != y1 or not (oy == y0 or (y0 == 2 and oy <= 1)):
                    continue
                cands.append((s, W, H, ox, oy))
    best = None
    for (s, W, H, ox, oy) in sorted(cands, reverse=True):
        a = a0.copy()
        if s >= 2:
            if oy <= 1:
                if (1 - oy) % s != 0: a[1, :] = a[0, :]
                else: a[1, :] = a[2, :]
            if ox == 0: a[:, 0] = a[:, 1]
        sub = a[oy:oy + H * s, ox:ox + W * s]
        g = sub[::s, ::s]
        if s == 1 or np.array_equal(np.repeat(np.repeat(g, s, 0), s, 1), sub):
            best = (s, W, H, ox, oy, g); break
    if best is None:
        if not cands: return None
        s, W, H, ox, oy = cands[-1]
        best = (s, W, H, ox, oy, a0[oy:oy + H * s:s, ox:ox + W * s:s])
    s, W, H, ox, oy, G = best
    slots = {}
    for j in range(0, (H + 1) // 3 + 1):
        for i in range(0, (W + 1) // 3 + 1):
            x, y = 3 * i, 3 * j
            if x + 1 >= W or y + 1 >= H: continue
            c = int(G[y, x])
            if c in (3, 4, 8, 14) or not (G[y, x + 1] == c and G[y + 1, x] == c and G[y + 1, x + 1] == c):
                c = None
            slots[(i, j)] = c
    targets = []
    for py in range(2, H - 3, 3):
        for px in range(2, W - 3, 3):
            c = int(G[py, px])
            if c in (11, 12) and G[py, px + 3] == c and G[py + 3, px] == c and G[py + 3, px + 3] == c:
                targets.append((((px + 1) // 3, (py + 1) // 3), c))
    buttons = {}
    seen = np.zeros(G.shape, bool)
    for y in range(H):
        for x in range(W):
            c = int(G[y, x])
            if c in (8, 14) and not seen[y, x]:
                st = [(y, x)]; seen[y, x] = True; comp = []
                while st:
                    cy, cx = st.pop(); comp.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and G[ny, nx] == c:
                            seen[ny, nx] = True; st.append((ny, nx))
                ys = [p[0] for p in comp]; xs = [p[1] for p in comp]
                gx = (min(xs) + max(xs)) // 2; gy = (min(ys) + max(ys)) // 2
                if G[gy, gx] != c:
                    gy, gx = comp[0]
                bid = (c, min(xs), min(ys))
                buttons[bid] = (oy + gy * s + s // 2, ox + gx * s + s // 2, (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    return (W, H), slots, targets, buttons

def _hung(a):
    n = len(a); INF = float('inf')
    u = [0] * (n + 1); v = [0] * (n + 1); p = [0] * (n + 1); way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0; minv = [INF] * (n + 1); used = [False] * (n + 1)
        while True:
            used[j0] = True; i0 = p[j0]; delta = INF; j1 = 0
            row = a[i0 - 1]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]: minv[j] = cur; way[j] = j0
                    if minv[j] < delta: delta = minv[j]; j1 = j
            for j in range(n + 1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    res = [0] * n
    for j in range(1, n + 1): res[p[j] - 1] = j - 1
    return res

def _dist(s, d):
    return ((s[0] - d[0]) ** 2 + (s[1] - d[1]) ** 2) ** 0.5

def _changed(obsl):
    U = set()
    for pr, cu in obsl:
        for k in cu:
            if pr.get(k) != cu.get(k): U.add(k)
    return U

def _learn_hung(obsl, U):
    cost = []
    for s in U:
        row = []
        for d in U:
            bad = sum(1 for pr, cu in obsl if cu.get(d) != pr.get(s))
            row.append(1000 * bad + _dist(s, d))
        cost.append(row)
    r = _hung(cost)
    return {U[i]: U[r[i]] for i in range(len(U)) if U[i] != U[r[i]]}

def _learn(obsl):
    U = sorted(_changed(obsl))
    if not U: return {}
    n = len(U)
    cand = {}
    for s in U:
        cs = [(_dist(s, d), d) for d in U if d != s and all(cu.get(d) == pr.get(s) for pr, cu in obsl)]
        if not cs:
            return _learn_hung(obsl, U)
        cs.sort(); cand[s] = cs
    minc = {s: cand[s][0][0] for s in U}
    PEN = 10.0
    best = [float('inf'), None]
    succ = {}; tgt = set(); cnt = [0]
    rem = [sum(minc.values())]

    def rec(cur, start, cost):
        cnt[0] += 1
        if cnt[0] > 60000: return
        rem[0] -= minc[cur]
        for c, d in cand[cur]:
            if d in tgt: continue
            if cost + c + rem[0] + PEN >= best[0]: break
            succ[cur] = d; tgt.add(d)
            if d == start:
                nc = cost + c + PEN
                if len(succ) == n:
                    if nc < best[0]: best[0] = nc; best[1] = dict(succ)
                else:
                    ns = next(x for x in U if x not in succ)
                    rec(ns, ns, nc)
            else:
                rec(d, start, cost + c)
            del succ[cur]; tgt.discard(d)
        rem[0] += minc[cur]
    rec(U[0], U[0], 0.0)
    if best[1] is None:
        return _learn_hung(obsl, U)
    return {k: v for k, v in best[1].items() if k != v}

def act(obs, available, memory):
    P = _parse(obs)
    if P is None:
        return ("ACTION6", 32, 32)
    key, slots, targets, buttons = P
    if memory.get('key') != key:
        memory.clear(); memory['key'] = key; memory['obs'] = {}
    if 'last' in memory:
        bid, prev = memory.pop('last')
        memory['obs'].setdefault(bid, []).append((prev, dict(slots)))
    if not buttons:
        return ("ACTION6", 32, 32)
    O = memory['obs']
    # pairing
    Us = {b: _changed(O[b]) for b in O if b in buttons}
    pairs = {}
    cl = []
    for bl in Us:
        if bl[0] != 8: continue
        for br in Us:
            if br[0] != 14: continue
            A, B = Us[bl], Us[br]
            if not A or not B: continue
            j = len(A & B) / float(len(A | B))
            if j > 0.5: cl.append((j, bl, br))
    cl.sort(reverse=True)
    for j, bl, br in cl:
        if bl in pairs or br in pairs: continue
        pairs[bl] = br; pairs[br] = bl
    perms = {}
    for b in Us:
        if b in perms: continue
        if b in pairs:
            o = pairs[b]
            data = list(O[b]) + [(cu, pr) for pr, cu in O[o]]
            pm = _learn(data)
            perms[b] = pm; perms[o] = {v: k for k, v in pm.items()}
        else:
            perms[b] = _learn(O[b])
    observed = set(perms)
    for b in buttons:
        if b in perms: continue
        best = None
        for o in buttons:
            if o[0] != b[0] and o in observed and perms[o] and o not in pairs:
                d = (buttons[o][2] - buttons[b][2]) ** 2 + (buttons[o][3] - buttons[b][3]) ** 2
                if d <= 64 and (best is None or d < best[0]): best = (d, o)
        if best:
            perms[b] = {v: k for k, v in perms[best[1]].items()}
    gcols = sorted(set(c for _, c in targets))
    start = tuple(tuple(sorted(k for k, v in slots.items() if v == c)) for c in gcols)
    def ok(st):
        for t, c in targets:
            if t not in st[gcols.index(c)]: return False
        return True
    plan = None
    if targets and not ok(start):
        usable = [(b, p) for b, p in perms.items() if p and b in buttons]
        par = {start: None}; q = deque([start]); found = None
        while q and len(par) < 120000:
            st = q.popleft()
            for b, p in usable:
                ns = tuple(tuple(sorted(p.get(x, x) for x in grp)) for grp in st)
                if ns not in par:
                    par[ns] = (st, b)
                    if ok(ns): found = ns; break
                    q.append(ns)
            if found: break
        if found:
            cur = found; first = None
            while par[cur] is not None:
                cur, first = par[cur][0], par[cur][1]
            plan = first
    if plan is None:
        bl = sorted(buttons)
        cand = [b for b in bl if b not in observed and b not in perms]
        if not cand: cand = [b for b in bl if b not in observed]
        if not cand:
            lb = memory.get('lastb')
            avoid = pairs.get(lb)
            c2 = [b for b in bl if b != avoid]
            if not c2: c2 = bl
            cand = sorted(c2, key=lambda b: (len(O.get(b, [])) + (len(O.get(pairs[b], [])) if b in pairs else 0), b != lb))
        plan = cand[0]
    memory['last'] = (plan, dict(slots))
    memory['lastb'] = plan
    r, c = buttons[plan][0], buttons[plan][1]
    return ("ACTION6", int(r), int(c))
