import numpy as np, heapq, time

MV = [("ACTION1", 0, -3), ("ACTION2", 0, 3), ("ACTION3", -3, 0), ("ACTION4", 3, 0)]
OFF = 16; TS = 64 + OFF

def comps(mask):
    H, W = mask.shape; seen = np.zeros(mask.shape, bool); out = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]: continue
        st = [(r, c)]; seen[r, c] = True; cl = []
        while st:
            y, x = st.pop(); cl.append((y, x))
            for yy, xx in ((y+1,x),(y-1,x),(y,x+1),(y,x-1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; st.append((yy, xx))
        out.append(cl)
    return out

def bb(cl):
    ys = [p[0] for p in cl]; xs = [p[1] for p in cl]
    return min(xs), min(ys), max(xs)-min(xs)+1, max(ys)-min(ys)+1

def ctr(w, h):
    return (1 if w == 3 else 2), (1 if h == 3 else 2)

def shape_info(w, h):
    if w == 3 and h == 3: return [(1, 1)], []
    if w == 6 and h == 3: return [(1, 2), (1, 3)], [(1, 1), (1, 4)]
    if w == 3 and h == 6: return [(2, 1), (3, 1)], [(1, 1), (4, 1)]
    mk = [(2, 2), (2, 3), (3, 2), (3, 3)]
    ring = [(dy, dx) for dy in range(1, 5) for dx in range(1, 5) if (dy, dx) not in mk]
    return mk, ring

def parse_blocks(obs):
    m = np.isin(obs, [14, 0, 5]); m[63, :] = False
    res = []
    for cl in comps(m):
        S = set((int(a), int(b)) for a, b in cl); cov = set()
        for (r, c) in sorted(S):
            if (r, c) in cov: continue
            got = None
            for (w, h) in ((6, 6), (6, 3), (3, 6), (3, 3)):
                if r + h > 63 or c + w > 64: continue
                mk, ring = shape_info(w, h); ok = True
                mks = set(mk); rgs = set(ring)
                for dy in range(h):
                    for dx in range(w):
                        p = (r+dy, c+dx); v = obs[p]
                        if (dy, dx) in mks:
                            if v not in (5, 0, 4) or (p in cov) or (p not in S and v != 4): ok = False
                        elif (dy, dx) in rgs:
                            if v != 14 or p not in S or p in cov: ok = False
                        elif dy == 0 or dx == 0 or dy == h-1 or dx == w-1:
                            if v not in (14, 0) or p not in S or p in cov: ok = False
                        else:
                            if p not in S or p in cov: ok = False
                        if not ok: break
                    if not ok: break
                if ok: got = (w, h); break
            if got is None:
                cov.add((r, c)); continue
            w, h = got
            for dy in range(h):
                for dx in range(w): cov.add((r+dy, c+dx))
            mk, _ = shape_info(w, h)
            sel = all(obs[r+dy, c+dx] == 0 for (dy, dx) in mk)
            res.append((c, r, w, h, sel))
    return res

def act(obs, available, memory):
    obs = np.array(obs)
    objs = []; sel = None
    for (x, y, w, h, s) in parse_blocks(obs):
        if s: sel = len(objs)
        objs.append(dict(k='b', x=x, y=y, w=w, h=h, m=np.ones((h, w), bool), d=None))
    if sel is None: sel = 0
    mm = obs == 11; mm[63, :] = False
    for cl in comps(mm):
        x, y, w, h = bb(cl); msk = np.zeros((h, w), bool)
        for (yy, xx) in cl: msk[yy-y, xx-x] = True
        objs.append(dict(k='c', x=x, y=y, w=w, h=h, m=msk, d=None))
    mm = np.isin(obs, [12, 13]); mm[63, :] = False
    bd = memory.setdefault("bd", {})
    cnt = 0
    for cl in comps(mm):
        x, y, w, h = bb(cl)
        sub = obs[y:y+h, x:x+w]
        c12 = int((sub == 12).sum()); k = c12 // max(w, 1)
        d = None
        if 0 < c12 < w * h:
            if (sub[0, :] == 12).all(): d = (0, 3)
            elif (sub[-1, :] == 12).all(): d = (0, -3)
            elif (sub[:, 0] == 12).all(): d = (3, 0)
            elif (sub[:, -1] == 12).all(): d = (-3, 0)
        if d is not None: bd[(x, y)] = d
        else: d = bd.get((x, y), (0, 3))
        cnt = max(cnt, k)
        objs.append(dict(k='x', x=x, y=y, w=w, h=h, m=np.ones((h, w), bool), d=d))
    n = len(objs)
    if n == 0 or objs[sel]['k'] != 'b':
        return ("ACTION1", None, None)
    if "targets" not in memory:
        m4 = obs == 4; m4[63, :] = False
        for o in objs:
            if o['k'] == 'b': m4[o['y']:o['y']+o['h'], o['x']:o['x']+o['w']] = False
        T = []
        for cl in comps(m4):
            x, y, w, h = bb(cl)
            if w >= 5 and h >= 5:
                T.append((x, y, w, h, 'b' if obs[y, x] == 4 else 'c'))
        memory["targets"] = T
    targets = memory["targets"]
    wall = obs == 15; pad = obs == 2; pad[63, :] = True
    for o in objs:
        reg = (slice(o['y'], o['y']+o['h']), slice(o['x'], o['x']+o['w']))
        wall[reg] &= ~o['m']; pad[reg] &= ~o['m']
    def table(arr, oobv):
        E = OFF; Pe = np.full((TS + 16, TS + 16), oobv, bool)
        Pe[E:E+64, E:E+64] = arr
        res = []
        for o in objs:
            t = np.zeros((TS, TS), bool)
            for my, mx in zip(*np.nonzero(o['m'])):
                t |= Pe[my:my+TS, mx:mx+TS]
            res.append(t)
        return res
    TP = table(pad, True); TW = table(wall, False)
    full = [bool(o['m'].all()) for o in objs]
    W_ = [o['w'] for o in objs]; H_ = [o['h'] for o in objs]
    bombs = [i for i in range(n) if objs[i]['k'] == 'x']
    def hP(i, x, y):
        tx, ty = x + OFF, y + OFF
        if tx < 0 or ty < 0 or tx >= TS or ty >= TS: return True
        return TP[i][ty, tx]
    def hW(i, x, y):
        tx, ty = x + OFF, y + OFF
        if tx < 0 or ty < 0 or tx >= TS or ty >= TS: return False
        return TW[i][ty, tx]
    def ov(i, xi, yi, j, xj, yj):
        if xi >= xj+W_[j] or xj >= xi+W_[i] or yi >= yj+H_[j] or yj >= yi+H_[i]: return False
        if full[i] and full[j]: return True
        x0, x1 = max(xi, xj), min(xi+W_[i], xj+W_[j]); y0, y1 = max(yi, yj), min(yi+H_[i], yj+H_[j])
        return bool((objs[i]['m'][y0-yi:y1-yi, x0-xi:x1-xi] & objs[j]['m'][y0-yj:y1-yj, x0-xj:x1-xj]).any())
    def rov(j, pos, rx0, ry0, rx1, ry1):
        xj, yj = pos[j]
        x0, x1 = max(rx0, xj), min(rx1, xj+W_[j]); y0, y1 = max(ry0, yj), min(ry1, yj+H_[j])
        if x0 >= x1 or y0 >= y1: return False
        if full[j]: return True
        return bool(objs[j]['m'][y0-yj:y1-yj, x0-xj:x1-xj].any())
    def push(pos, i, dx, dy, d=0):
        if d > 12: return True
        ox, oy = pos[i]; nx, ny = ox+dx, oy+dy
        if hP(i, nx, ny): return True
        pos[i] = (nx, ny)
        for j in range(n):
            if j != i and ov(i, nx, ny, j, *pos[j]):
                if push(pos, j, dx, dy, d+1):
                    pos[i] = (ox, oy); return True
        return False
    def explode(pos):
        moved = {}
        for k in (1, 2, 3):
            L = 3 * k
            for b in bombs:
                dx, dy = objs[b]['d']; bx, by = pos[b]; w, h = W_[b], H_[b]
                if dy > 0: r = (bx, by+h, bx+w, by+h+L)
                elif dy < 0: r = (bx, by-L, bx+w, by)
                elif dx < 0: r = (bx-L, by, bx, by+h)
                else: r = (bx+w, by, bx+w+L, by+h)
                for j in range(n):
                    if j != b and rov(j, pos, *r):
                        push(pos, j, dx, dy); moved[j] = (dx, dy)
        for _ in range(40):
            prog = False
            for j, (dx, dy) in moved.items():
                if hW(j, *pos[j]) and not push(pos, j, dx, dy): prog = True
            if not prog: break
    def step(pos, s, c, dx, dy):
        pos = list(pos); x, y = pos[s][0]+dx, pos[s][1]+dy
        if not (hW(s, x, y) or hP(s, x, y)):
            L = [j for j in range(n) if j != s and ov(s, x, y, j, *pos[j])]
            if not L: pos[s] = (x, y)
            else:
                k = 0
                for _ in range(40):
                    done = []
                    for i in L:
                        if k >= 5 and not hW(i, *pos[i]): done.append(True)
                        else: done.append(push(pos, i, dx, dy))
                    if all(done): break
                    k += 1
        if bombs:
            c += 1
            if c >= 6:
                explode(pos); c = 0
        return tuple(pos), c
    def hval(pos):
        t = 0
        for (tx, ty, tw, th, tk) in targets:
            best = None
            for i in range(n):
                if objs[i]['k'] == tk and W_[i] == tw-2 and H_[i] == th-2:
                    v = (abs(pos[i][0]-tx-1) + abs(pos[i][1]-ty-1)) / 3.0
                    if best is None or v < best: best = v
            if best is not None: t += best
        return t
    def sig(pos, s, c):
        return (tuple(sorted((objs[i]['k'], W_[i], H_[i]) + pos[i] for i in range(n))), pos[s], c)
    pos0 = tuple((o['x'], o['y']) for o in objs)
    c0 = cnt if bombs else 0
    cur = sig(pos0, sel, c0)
    plan = memory.get("plan")
    if plan and memory.get("exp") == cur:
        a, e = plan.pop(0); memory["exp"] = e
        return a
    sels = [i for i in range(n) if objs[i]['k'] == 'b']
    t0 = time.time()
    start = (pos0, sel, c0)
    par = {start: None}; g = {start: 0}
    h0 = hval(pos0)
    pq = [(h0, 0, start)]; ct = 1
    bestn, besth = start, h0; goal = None
    while pq and time.time() - t0 < 8.0 and len(par) < 300000:
        f, _, st = heapq.heappop(pq)
        pos, s, c = st; hv = hval(pos)
        if hv < besth: besth, bestn = hv, st
        if hv == 0 and targets: goal = st; break
        gc = g[st]
        succ = []
        for (nm, dx, dy) in MV:
            np_, nc = step(pos, s, c, dx, dy)
            succ.append(((np_, s, nc), (nm, None, None)))
        for j in sels:
            if j != s:
                cx, cy = ctr(W_[j], H_[j])
                succ.append(((pos, j, c), ("ACTION6", pos[j][1]+cy, pos[j][0]+cx)))
        for ns, a in succ:
            if ns not in g or g[ns] > gc + 1:
                g[ns] = gc + 1; par[ns] = (st, a)
                heapq.heappush(pq, (gc + 1 + 2.0 * hval(ns[0]), ct, ns)); ct += 1
    end = goal if goal is not None else bestn
    seq = []; st = end
    while par[st] is not None:
        p, a = par[st]; seq.append((a, sig(*st))); st = p
    seq.reverse()
    if not seq:
        memory["plan"] = None
        return (MV[np.random.randint(4)][0], None, None)
    a, e = seq.pop(0)
    memory["plan"] = seq; memory["exp"] = e
    return a
