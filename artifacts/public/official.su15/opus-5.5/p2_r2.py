import numpy as np, time

FC = {10:(0,1),6:(1,2),15:(2,3),11:(3,4),12:(4,5),8:(5,7),7:(7,9),14:(8,10)}
SZ = {0:1,1:2,2:3,3:4,4:5,5:7,6:8,7:9,8:10}
S2L = {v:k for k,v in SZ.items()}
EC = {7:1,14:2,13:3}

def comps(m, eight=False):
    seen = np.zeros(m.shape, bool); out = []
    H, W = m.shape
    nb = [(1,0),(-1,0),(0,1),(0,-1)] + ([(1,1),(1,-1),(-1,1),(-1,-1)] if eight else [])
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        if seen[y, x]: continue
        st = [(y, x)]; seen[y, x] = True; pts = []
        while st:
            a, b = st.pop(); pts.append((a, b))
            for da, db in nb:
                c, d = a+da, b+db
                if 0 <= c < H and 0 <= d < W and m[c, d] and not seen[c, d]:
                    seen[c, d] = True; st.append((c, d))
        ya = [p[0] for p in pts]; xa = [p[1] for p in pts]
        out.append((min(xa), min(ya), max(xa), max(ya), len(pts), set(pts)))
    return out

def elevel(c, x0, y0, pts):
    if c in EC: return EC[c]
    if (y0, x0) in pts: return 3
    if (y0+2, x0+2) in pts: return 2
    return 1

def parse(obs):
    fr = []; en = []
    for col, (lv, s) in FC.items():
        m = obs == col; m[:10, :] = False
        for x0, y0, x1, y1, n, _ in comps(m):
            w = x1-x0+1; h = y1-y0+1
            if n < max(1, s*s//2): continue
            if col in (7, 14) and n < 0.8*w*h: continue
            for i in range(max(1, round(w/s))):
                for j in range(max(1, round(h/s))):
                    fr.append((lv, min(x0+i*s, 64-s), max(10, min(y0+j*s, 64-s))))
    for col in (7, 13, 14):
        m = obs == col; m[:10, :] = False
        for f in fr:
            s = SZ[f[0]]; m[f[2]:f[2]+s, f[1]:f[1]+s] = False
        for x0, y0, x1, y1, n, pts in comps(m, True):
            if n < 4: continue
            for i in range(max(1, round((x1-x0+1)/5))):
                en.append((EC[col], min(59, x0+5*i), max(10, min(60, y0))))
    return fr, en

def setup(obs, mem):
    goals = []
    m = (obs == 9) | (obs == 2); m[:10, :] = False
    for x0, y0, x1, y1, n, _ in comps(m):
        if n >= 40 and x1-x0 == 8 and y1-y0 == 8: goals.append((x0, y0))
    cnt = {}; ecnt = {}
    bar = obs.copy(); bar[10:, :] = -1; bar[:, :24] = -1
    for col in set(FC) | {2, 13}:
        for x0, y0, x1, y1, n, pts in comps(bar == col, True):
            w = x1-x0+1; h = y1-y0+1
            if w == h and n == w*w and w in S2L:
                cnt[S2L[w]] = cnt.get(S2L[w], 0) + 1
            elif w == 5 and h == 4 and n < 20 and col in (2, 7, 13, 14):
                e = elevel(col, x0, y0, pts); ecnt[e] = ecnt.get(e, 0) + 1
    if cnt: L = max(cnt); mem['L'], mem['N'] = L, cnt[L]
    else: mem['L'], mem['N'] = 3, 1
    if ecnt: E = max(ecnt); mem['E'], mem['M'] = E, ecnt[E]
    else: mem['E'], mem['M'] = 1, 0
    mem['goals'] = goals

def near(cx, cy, x, y, w, h):
    wx = min(max(cx, x), x+w-1); wy = min(max(cy, y), y+h-1)
    return (cx-wx)**2 + (cy-wy)**2 <= 64

def ov(ax, ay, aw, ah, bx, by, bw, bh):
    return not (ax+aw <= bx or bx+bw <= ax or ay+ah <= by or by+bh <= ay)

def cl(v, lo, hi): return max(lo, min(hi, v))

def merge(items, idx, w_of, h_of, maxlv, ymax):
    par = {i: i for i in idx}
    def f(i):
        while par[i] != i: i = par[i]
        return i
    for a in range(len(idx)):
        i = idx[a]
        for b in range(a+1, len(idx)):
            j = idx[b]; I = items[i]; J = items[j]
            if ov(I[1], I[2], w_of(I[0]), h_of(I[0]), J[1], J[2], w_of(J[0]), h_of(J[0])):
                if I[0] != J[0]: return None
                par[f(j)] = f(i)
    g = {}
    for i in idx: g.setdefault(f(i), []).append(i)
    out = []
    for grp in g.values():
        I = items[grp[0]]; lv = I[0]
        if len(grp) == 1: out.append(tuple(I)); continue
        if lv >= maxlv: continue
        ax = sum(items[i][1] + w_of(lv)//2 for i in grp)//len(grp)
        ay = sum(items[i][2] + h_of(lv)//2 for i in grp)//len(grp)
        nw, nh = w_of(lv+1), h_of(lv+1)
        out.append((lv+1, cl(ax-nw//2, 0, 64-nw), cl(ay-nh//2, 10, min(63, 64-nh))))
    return out

def sim(fr, en, cx, cy):
    sF = [i for i, f in enumerate(fr) if near(cx, cy, f[1], f[2], SZ[f[0]], SZ[f[0]])]
    sE = [j for j, e in enumerate(en) if near(cx, cy, e[1], e[2], 5, 4)]
    if not sF and not sE: return None
    F = [list(f) for f in fr]; E = [list(e) for e in en]
    ed = {}
    for j in sE:
        dx = cx - (E[j][1]+2); dy = cy - (E[j][2]+2); d = (dx*dx+dy*dy)**.5
        ed[j] = [dx/d if d else 0, dy/d if d else 0, float(E[j][1]), float(E[j][2])]
    kn = {}; cool = set(sE)
    for t in range(4):
        for i in sF:
            if i in kn: continue
            s = SZ[F[i][0]]; p = F[i]
            p[1] = cl(p[1] + cl(cx-(p[1]+s//2), -4, 4), 0, 64-s)
            p[2] = cl(p[2] + cl(cy-(p[2]+s//2), -4, 4), 10, min(63, 64-s))
        for j in sE:
            d = ed[j]; d[2] = cl(round(d[2]+d[0]*3.4), 0, 59); d[3] = cl(round(d[3]+d[1]*3.4), 10, 60)
            E[j][1], E[j][2] = d[2], d[3]
        for j in range(len(E)):
            if j in cool: continue
            ex, ey = E[j][1]+2, E[j][2]+2; best = None
            for i, f in enumerate(F):
                if i in kn: continue
                s = SZ[f[0]]; dd = (f[1]+s//2-ex)**2 + (f[2]+s//2-ey)**2
                if best is None or dd < best[0]: best = (dd, f[1]+s//2, f[2]+s//2)
            if best is None: break
            k = 2 if E[j][0] == 3 else 1
            E[j][1] = cl(E[j][1] + (k if best[1] > ex else -k if best[1] < ex else 0), 0, 59)
            E[j][2] = cl(E[j][2] + (k if best[2] > ey else -k if best[2] < ey else 0), 10, 60)
            for i, f in enumerate(F):
                if i in kn: continue
                s = SZ[f[0]]
                if ov(E[j][1], E[j][2], 5, 4, f[1], f[2], s, s):
                    dx = f[1]+s//2-(E[j][1]+2); dy = f[2]+s//2-(E[j][2]+2); d = (dx*dx+dy*dy)**.5
                    kn[i] = (dx/d, dy/d) if d else (0, -1); cool.add(j); break
    dead = set()
    for i, (ux, uy) in kn.items():
        if F[i][0] == 0: dead.add(i); continue
        F[i][0] -= 1; s = SZ[F[i][0]]
        F[i][1] = cl(round(F[i][1]+ux*10), 0, 64-s); F[i][2] = cl(round(F[i][2]+uy*10), 10, 64-s)
    ne = merge(E, list(range(len(E))), lambda l: 5, lambda l: 4, 3, 60)
    if ne is None: return None
    mi = [i for i in sF if i not in kn]
    nf = merge(F, mi, lambda l: SZ[l], lambda l: SZ[l], 8, 63)
    if nf is None: return None
    nf += [tuple(F[i]) for i in range(len(F)) if i not in mi and i not in dead]
    return nf, ne

def ing(cx, cy, goals): return any(gx <= cx < gx+9 and gy <= cy < gy+9 for gx, gy in goals)

def dg(x, y, goals):
    return min([((x-gx-4.5)**2 + (y-gy-4.5)**2)**.5 for gx, gy in goals] or [0])

def heur(fr, en, mem):
    L, N, E, M, goals = mem['L'], mem['N'], mem['E'], mem['M'], mem['goals']
    fc = [(f[0], f[1]+SZ[f[0]]//2, f[2]+SZ[f[0]]//2) for f in fr]
    ec = [(e[0], e[1]+2, e[2]+2) for e in en]
    Lin = sum(1 for l, x, y in fc if l == L and ing(x, y, goals))
    Ein = sum(1 for l, x, y in ec if l == E and ing(x, y, goals))
    if Lin == N and (M == 0 or Ein == M): return -1000.0
    cost = 0.0
    if sum(2**l for l, _, _ in fc) < N*2**L: cost += 500
    if M and sum(2**l for l, _, _ in ec) < M*2**E: cost += 500
    lc = []
    for k, (lv, x, y) in enumerate(fc):
        if lv < L:
            cost += 2*(L-lv)
            ds = [((x-a)**2+(y-b)**2)**.5 for j, (l2, a, b) in enumerate(fc) if j != k and l2 == lv]
            cost += min(ds)/16 if ds else 1.0
        elif lv > L:
            ds = [((x-a)**2+(y-b)**2)**.5 for _, a, b in ec]
            cost += 2*(lv-L) + (min(ds)/16 if ds else 50)
        else:
            lc.append(0.0 if ing(x, y, goals) else 1 + dg(x, y, goals)/16)
    lc.sort(); cost += sum(lc[:N]) + 1.5*max(0, N-len(lc))
    if Lin > N: cost += 3*(Lin-N)
    if M:
        ecs = []
        for k, (lv, x, y) in enumerate(ec):
            if lv < E:
                cost += 2*(E-lv)
                ds = [((x-a)**2+(y-b)**2)**.5 for j, (l2, a, b) in enumerate(ec) if j != k and l2 == lv]
                cost += min(ds)/16 if ds else 1.0
            elif lv == E:
                ecs.append(0.0 if ing(x, y, goals) else 1 + dg(x, y, goals)/16)
        ecs.sort(); cost += sum(ecs[:M]) + 1.5*max(0, M-len(ecs))
        if Ein > M: cost += 3*(Ein-M)
    return cost

def cands(fr, en):
    pts = set()
    for lv, x, y in fr:
        s = SZ[lv]
        for cx in range(x-8, x+s+8, 2):
            for cy in range(y-8, y+s+8, 2):
                if 0 <= cx < 64 and 10 <= cy <= 62: pts.add((cx, cy))
    for lv, x, y in en:
        for cx in range(x-8, x+13, 2):
            for cy in range(y-8, y+12, 2):
                if 0 <= cx < 64 and 10 <= cy <= 62: pts.add((cx, cy))
    return pts

def sig(fr, en): return (tuple(sorted(fr)), tuple(sorted(en)))

def act(obs, available, memory):
    t0 = time.time(); obs = np.asarray(obs)
    if 'goals' not in memory: setup(obs, memory)
    seen = memory.setdefault('seen', {})
    bad = memory.setdefault('bad', set()); noeff = memory.setdefault('noeff', set())
    badc = memory.setdefault('badc', {})
    fr, en = parse(obs)
    cur = sig(fr, en); seen[cur] = seen.get(cur, 0) + 1
    last = memory.get('last')
    if last and last[0] == cur:
        noeff.add((cur, last[1])); bad.add(last[2])
    h0 = heur(fr, en, memory)
    if h0 >= 400 and 'ACTION7' in available and memory.get('undos', 0) < 12:
        memory['undos'] = memory.get('undos', 0) + 1
        if last:
            bad.add(last[2]); badc.setdefault(last[0], []).append(last[1])
        memory['last'] = None
        return ("ACTION7", None, None)
    memory['last'] = None
    if not fr: return ("ACTION6", 32, 32)
    bl = badc.get(cur, [])
    first = {}
    for c in cands(fr, en):
        if (cur, c) in noeff: continue
        if any(abs(c[0]-b[0]) <= 3 and abs(c[1]-b[1]) <= 3 for b in bl): continue
        r = sim(fr, en, c[0], c[1])
        if r is None: continue
        sg = sig(*r)
        if sg == cur or sg in first or sg in bad: continue
        h = heur(r[0], r[1], memory)
        if h <= -1000:
            memory['last'] = (cur, c, sg)
            return ("ACTION6", c[1], c[0])
        first[sg] = (h + 0.7*seen.get(sg, 0), c, r, sg)
    if not first: return ("ACTION6", 32, 32)
    items = sorted(first.values(), key=lambda v: v[0])
    best, bc, bs = items[0][0], items[0][1], items[0][3]
    for h1, c1, r1, s1 in items[:8]:
        if time.time() - t0 > 5: break
        vis = set()
        for c in cands(*r1):
            r2 = sim(r1[0], r1[1], c[0], c[1])
            if r2 is None: continue
            sg = sig(*r2)
            if sg in vis or sg in bad: continue
            vis.add(sg)
            h2 = heur(r2[0], r2[1], memory) + 0.3 + 0.7*seen.get(sg, 0)
            if h2 < best: best, bc, bs = h2, c1, s1
    memory['last'] = (cur, bc, bs)
    return ("ACTION6", bc[1], bc[0])
