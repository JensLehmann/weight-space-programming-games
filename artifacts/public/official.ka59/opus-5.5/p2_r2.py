import numpy as np, time, heapq, random

DIRS = {"ACTION1": (-3, 0), "ACTION2": (3, 0), "ACTION3": (0, -3), "ACTION4": (0, 3)}
P = 16

def _comps(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask, bool); out = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                q = [(r, c)]; seen[r, c] = True; pts = []
                while q:
                    y, x = q.pop(); pts.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                out.append(pts)
    return out

PATS = []
for (h, w) in [(6,6),(6,3),(3,6),(3,3)]:
    if (h, w) == (3,3): cs, fs = [(1,1)], []
    elif (h, w) == (6,3): cs, fs = [(2,1),(3,1)], [(1,1),(4,1)]
    elif (h, w) == (3,6): cs, fs = [(1,2),(1,3)], [(1,1),(1,4)]
    else:
        cs = [(2,2),(2,3),(3,2),(3,3)]
        fs = [(a,b) for a in range(1,5) for b in range(1,5) if (a,b) not in cs]
    PATS.append((h, w, cs, fs))

def parse(obs, memory):
    g = np.array(obs).astype(int); g[63, :] = 2
    isb = np.isin(g, [14, 0, 5]); g4 = g == 4
    ok = isb | g4
    pd = np.pad(ok, 1, constant_values=False)
    nb = pd[:-2,1:-1] & pd[2:,1:-1] & pd[1:-1,:-2] & pd[1:-1,2:]
    c4 = g4 & nb
    M = isb | c4
    goals = memory.setdefault("goals", set())
    for pts in _comps(g4 & ~c4):
        ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
        t, l, b, r = min(ys), min(xs), max(ys), max(xs); h, w = b-t+1, r-l+1
        border = all(y in (t, b) or x in (l, r) for y, x in pts)
        if border and h >= 5 and w >= 5:
            okb = True
            for x in range(l, r+1):
                for y in (t, b):
                    if not (g4[y, x] or M[y, x]): okb = False
            for y in range(t, b+1):
                for x in (l, r):
                    if not (g4[y, x] or M[y, x]): okb = False
            if okb: goals.add(("R", t, l, h, w))
        elif h == 11 and w == 11 and len(pts) >= 30:
            goals.add(("C", t, l, 11, 11))
    objs = []; pos = []; sel = None; ticks = []
    used = np.zeros((64, 64), bool)
    for y in range(64):
        for x in range(64):
            if M[y, x] and not used[y, x]:
                found = False
                for (h, w, cs, fs) in PATS:
                    if y+h > 64 or x+w > 64: continue
                    if not M[y:y+h, x:x+w].all() or used[y:y+h, x:x+w].any(): continue
                    cv = [g[y+a, x+b] for a, b in cs]
                    if not all(v in (0, 4, 5) for v in cv) or len(set(cv)) != 1: continue
                    if not all(g[y+a, x+b] == 14 for a, b in fs): continue
                    if cv[0] == 0: sel = len(objs)
                    objs.append(("B", np.ones((h, w), bool), True, 0)); pos.append((y, x)); ticks.append(0)
                    used[y:y+h, x:x+w] = True; found = True; break
                if not found: used[y, x] = True
    for pts in _comps(g == 11):
        ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
        t, l, b, r = min(ys), min(xs), max(ys), max(xs)
        m = g[t:b+1, l:r+1] == 11
        objs.append(("C", m, bool(m.all()), 0)); pos.append((t, l)); ticks.append(0)
    oldb = memory.get("bombs", [])
    newb = []
    for pts in _comps(np.isin(g, [13, 12])):
        ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
        t, l, b, r = min(ys), min(xs), max(ys), max(xs)
        if b-t+1 == 6 and r-l+1 == 6:
            sub = g[t:t+6, l:l+6] == 12
            k = int(sub.sum()) // 6
            rot = None
            if 0 < k < 6:
                if sub[0].all() and not sub[5].all(): rot = 0
                elif sub[5].all() and not sub[0].all(): rot = 180
                elif sub[:, 5].all() and not sub[:, 0].all(): rot = 90
                elif sub[:, 0].all() and not sub[:, 5].all(): rot = 270
            if rot is None:
                best = None
                for (oy, ox, orot) in oldb:
                    d = abs(oy-t) + abs(ox-l)
                    if orot is not None and (best is None or d < best[0]): best = (d, orot)
                if best is not None and best[0] <= 30: rot = best[1]
            newb.append((t, l, rot))
            objs.append(("X", np.ones((6, 6), bool), True, rot if rot is not None else 0))
            pos.append((t, l)); ticks.append(k)
    memory["bombs"] = newb
    W2 = np.pad(g == 2, P, constant_values=True)
    W15 = np.pad(g == 15, P, constant_values=False)
    if sel is None:
        sel = next((i for i, o in enumerate(objs) if o[0] == "B"), 0)
    rk = tuple(sorted(str(b[2]) for b in newb))
    return objs, tuple(pos), sel, tuple(ticks), W2, W15, rk

def act(obs, available, memory):
    obs = np.array(obs)
    objs, pos0, sel0, tk0, W2, W15, rk = parse(obs, memory)
    if not objs:
        return (random.choice(list(DIRS)), None, None)
    masks = [o[1] for o in objs]; full = [o[2] for o in objs]
    shp = [m.shape for m in masks]
    n = len(objs)
    bombs = [i for i in range(n) if objs[i][0] == "X"]
    def canonf(p, sl, tk):
        return (tuple(sorted((objs[i][0],) + shp[i] + p[i] + (tk[i],) for i in range(n))), p[sl])
    canon = canonf(pos0, sel0, tk0)

    def hits(W, i, y, x):
        h, w = shp[i]
        y0, x0 = y+P, x+P
        if y0 < 0 or x0 < 0 or y0+h > W.shape[0] or x0+w > W.shape[1]:
            return W is W2
        return bool((W[y0:y0+h, x0:x0+w] & masks[i]).any())

    def ov(i, yi, xi, j, yj, xj):
        hi, wi = shp[i]; hj, wj = shp[j]
        y0 = max(yi, yj); y1 = min(yi+hi, yj+hj)
        x0 = max(xi, xj); x1 = min(xi+wi, xj+wj)
        if y0 >= y1 or x0 >= x1: return False
        if full[i] and full[j]: return True
        return bool((masks[i][y0-yi:y1-yi, x0-xi:x1-xi] & masks[j][y0-yj:y1-yj, x0-xj:x1-xj]).any())

    def rov(i, yi, xi, ry, rx, rh, rw):
        hi, wi = shp[i]
        y0 = max(yi, ry); y1 = min(yi+hi, ry+rh)
        x0 = max(xi, rx); x1 = min(xi+wi, rx+rw)
        if y0 >= y1 or x0 >= x1: return False
        if full[i]: return True
        return bool(masks[i][y0-yi:y1-yi, x0-xi:x1-xi].any())

    def push(p, j, dy, dx, depth=0):
        oy, ox = p[j]; ny, nx = oy+dy, ox+dx
        p[j] = (ny, nx)
        if hits(W2, j, ny, nx) or depth > 20:
            p[j] = (oy, ox); return True
        cl = [i for i in range(n) if i != j and ov(j, ny, nx, i, p[i][0], p[i][1])]
        for i in cl:
            if push(p, i, dy, dx, depth+1):
                p[j] = (oy, ox); return True
        return False

    def move(pos, sel, tk, dy, dx):
        y, x = pos[sel]; ny, nx = y+dy, x+dx
        p = list(pos)
        if not (hits(W2, sel, ny, nx) or hits(W15, sel, ny, nx)):
            cl = [j for j in range(n) if j != sel and ov(sel, ny, nx, j, pos[j][0], pos[j][1])]
            if not cl:
                p[sel] = (ny, nx)
            else:
                k = 0
                for _ in range(60):
                    res = []
                    for j in cl:
                        if k >= 5 and not hits(W15, j, p[j][0], p[j][1]): res.append(True)
                        else: res.append(push(p, j, dy, dx))
                    if all(res): break
                    k += 1
        if not bombs:
            p = tuple(p)
            return None if p == pos else (p, tk)
        t = list(tk); ex = []
        for b in bombs:
            t[b] += 1
            if t[b] >= 6: ex.append(b); t[b] = 0
        if ex:
            dirs = {}
            for idx in (1, 2, 3):
                L = 3*idx
                for b in ex:
                    by, bx = p[b]; rot = objs[b][3]
                    if rot == 0: r = (by+6, bx, L, 6); d = (3, 0)
                    elif rot == 180: r = (by-L, bx, L, 6); d = (-3, 0)
                    elif rot == 90: r = (by, bx-L, 6, L); d = (0, -3)
                    else: r = (by, bx+6, 6, L); d = (0, 3)
                    for j in range(n):
                        if j != b and rov(j, p[j][0], p[j][1], *r):
                            push(p, j, d[0], d[1]); dirs[j] = d
            for _ in range(40):
                mv = False
                for j, d in dirs.items():
                    if hits(W15, j, p[j][0], p[j][1]):
                        if not push(p, j, d[0], d[1]): mv = True
                if not mv: break
        return (tuple(p), tuple(t))

    goals = list(memory.get("goals", set()))
    gl = []
    for (kd, t, l, h, w) in goals:
        if kd == "R":
            c = [i for i in range(n) if objs[i][0] == "B" and shp[i] == (h-2, w-2)]
        else:
            c = [i for i in range(n) if objs[i][0] == "C"]
        if c: gl.append((t+1, l+1, c))

    def H(pos):
        tot = 0
        for (ty, tx, c) in gl:
            tot += min(abs(pos[i][0]-ty) + abs(pos[i][1]-tx) for i in c)
        return tot / 3.0

    last = memory.get("last")
    stuck = (last is not None and memory.get("lastact") in DIRS and np.array_equal(last, obs[:63])
             and not bombs)
    memory["last"] = obs[:63].copy()
    rkchg = memory.get("rk") != rk
    memory["rk"] = rk

    plan = memory.get("plan")
    if plan and memory.get("expect") == canon and not stuck and not rkchg:
        a = plan.pop(0)
        memory["expect"] = memory["expects"].pop(0)
        memory["lastact"] = a[0]
        return a
    if stuck or not gl:
        memory["plan"] = None
        a = random.choice(list(DIRS)); memory["lastact"] = a
        return (a, None, None)

    blocks = [i for i in range(n) if objs[i][0] == "B"]
    start = (pos0, sel0, tk0)
    par = {start: None}
    h0 = H(pos0)
    heap = [(h0, 0, 0, start)]; cnt = 1
    best = (h0, 0, start); goal = None
    t0 = time.time(); exp = 0
    while heap:
        f, _, gc, st = heapq.heappop(heap)
        pos, sel, tk = st
        hv = H(pos)
        if hv == 0: goal = st; break
        if (hv, gc) < best[:2]: best = (hv, gc, st)
        exp += 1
        if exp % 20 == 0 and time.time() - t0 > 7.0: break
        succ = []
        for a, (dy, dx) in DIRS.items():
            r = move(pos, sel, tk, dy, dx)
            if r is not None: succ.append(((r[0], sel, r[1]), (a,)))
        for j in blocks:
            if j != sel: succ.append(((pos, j, tk), ("S", j)))
        for ns, a in succ:
            if ns in par: continue
            par[ns] = (st, a)
            heapq.heappush(heap, (gc+1 + 1.5*H(ns[0]), cnt, gc+1, ns)); cnt += 1
    tgt = goal if goal is not None else best[2]
    if tgt == start:
        a = random.choice(list(DIRS)); memory["lastact"] = a; memory["plan"] = None
        return (a, None, None)
    path = []; s = tgt
    while par[s] is not None:
        ps, a = par[s]; path.append((ps, a, s)); s = ps
    path.reverse()
    acts = []; exps = []
    for ps, a, s in path:
        if a[0] == "S":
            j = a[1]; y, x = ps[0][j]; h, w = shp[j]
            acts.append(("ACTION6", int(y + h//2), int(x + w//2)))
        else:
            acts.append((a[0], None, None))
        exps.append(canonf(s[0], s[1], s[2]))
    a = acts.pop(0); memory["expect"] = exps.pop(0)
    memory["plan"] = acts; memory["expects"] = exps
    memory["lastact"] = a[0]
    return a
