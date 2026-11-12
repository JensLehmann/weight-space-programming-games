import numpy as np
import math

S = [1, 2, 3, 4, 5, 7, 8, 9, 10]
FC = {10: 0, 6: 1, 15: 2, 11: 3, 12: 4, 8: 5}
EC = {7: 1, 14: 2, 13: 3}

def comps(obs, r0, r1, c0, c1, skip):
    seen = np.zeros((64, 64), bool); out = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if seen[r, c]: continue
            col = int(obs[r, c])
            if col in skip:
                seen[r, c] = True; continue
            eight = col in (7, 13, 14)
            st = [(r, c)]; seen[r, c] = True; ys = []; xs = []
            while st:
                y, x = st.pop(); ys.append(y); xs.append(x)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if (dy == 0 and dx == 0) or (not eight and dy and dx): continue
                        ny, nx = y + dy, x + dx
                        if r0 <= ny < r1 and c0 <= nx < c1 and not seen[ny, nx] and obs[ny, nx] == col:
                            seen[ny, nx] = True; st.append((ny, nx))
            out.append((col, min(ys), min(xs), max(ys), max(xs), len(ys)))
    return out

def parse(obs):
    F = []; E = []; T = []
    for col, y0, x0, y1, x1, cnt in comps(obs, 10, 63, 0, 64, {5, 3, 0}):
        w = x1 - x0 + 1; h = y1 - y0 + 1
        if col == 9 and cnt == w * h and w <= 8 and h <= 8:
            F.append([6, x0, y0])
        elif col in (9, 2):
            if w >= 7 and h >= 7: T.append((x0, y0))
        elif col in FC:
            l = FC[col]; s = S[l]
            k = max(1, int(round(cnt / float(s * s))))
            F.append([l, x0, y0])
            if k >= 2: F.append([l, x1 - s + 1, y1 - s + 1])
        elif col == 7 and w >= 6 and h >= 6:
            F.append([7, x0, y0])
        elif col == 14 and w >= 6 and h >= 6:
            F.append([8, x0, y0])
        elif col in EC:
            E.append([EC[col], x0, y0])
    return F, E, T

def parse_goal(obs):
    G = {}; EG = {}
    for col, y0, x0, y1, x1, cnt in comps(obs, 0, 10, 24, 64, {4, 5}):
        if col in FC: G[FC[col]] = G.get(FC[col], 0) + 1
        elif col in EC: EG[EC[col]] = EG.get(EC[col], 0) + 1
    return list(G.items()), list(EG.items())

def fdim(f): s = S[f[0]]; return s, s
def edim(e): return 5, 4
def cen(o, d): w, h = d(o); return o[1] + w // 2, o[2] + h // 2
def clampv(v, a): return max(-a, min(a, v))
def inrad(px, py, x, y, w, h):
    cx = min(max(px, x), x + w - 1); cy = min(max(py, y), y + h - 1)
    return (px - cx) ** 2 + (py - cy) ** 2 <= 64
def ovl(a, da, b, db):
    wa, ha = da(a); wb, hb = db(b)
    return not (a[1] + wa <= b[1] or b[1] + wb <= a[1] or a[2] + ha <= b[2] or b[2] + hb <= a[2])
def place(o, d, cx, cy):
    w, h = d(o)
    o[1] = max(0, min(64 - w, cx - w // 2)); o[2] = max(10, min(64 - h, 63, cy - h // 2))

def merge_groups(objs, d, idx):
    par = list(range(len(objs)))
    def fd(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if ovl(objs[i], d, objs[j], d):
                if objs[i][0] != objs[j][0]: return None
                par[fd(j)] = fd(i)
    g = {}
    for i in idx: g.setdefault(fd(i), []).append(i)
    return [v for v in g.values() if len(v) > 1]

def sim(F, E, px, py):
    F = [list(f) for f in F]; E = [list(e) for e in E]
    inF = [i for i, f in enumerate(F) if inrad(px, py, f[1], f[2], *fdim(f))]
    inE = [i for i, e in enumerate(E) if inrad(px, py, e[1], e[2], 5, 4)]
    for i in inF:
        f = F[i]; cx, cy = cen(f, fdim)
        place(f, fdim, cx + clampv(px - cx, 16), cy + clampv(py - cy, 16))
    for i in inE:
        e = E[i]; cx, cy = cen(e, edim)
        if e[0] == 1:
            dx, dy = px - cx, py - cy; r = math.hypot(dx, dy)
            if r > 0:
                e[1] = max(0, min(59, int(round(e[1] + dx / r * 13.6))))
                e[2] = max(10, min(60, int(round(e[2] + dy / r * 13.6))))
        else:
            place(e, edim, cx + clampv(px - cx, 16), cy + clampv(py - cy, 16))
    hit = set()
    for i, e in enumerate(E):
        if i in inE or not F: continue
        cx, cy = cen(e, edim)
        best = min(range(len(F)), key=lambda j: (cen(F[j], fdim)[0] - cx) ** 2 + (cen(F[j], fdim)[1] - cy) ** 2)
        tx, ty = cen(F[best], fdim); k = 8 if e[0] == 3 else 4
        place(e, edim, cx + clampv(tx - cx, k), cy + clampv(ty - cy, k))
        for j, f in enumerate(F):
            if j in hit: continue
            if ovl(e, edim, f, fdim):
                hit.add(j); fx, fy = cen(f, fdim)
                dx, dy = fx - cx, fy - cy; r = math.hypot(dx, dy) or 1.0
                f[0] -= 1
                if f[0] >= 0: place(f, fdim, int(fx + dx / r * 10), int(fy + dy / r * 10))
    gr = merge_groups(F, fdim, [i for i in inF if F[i][0] >= 0])
    if gr is None: return None
    rem = set(); new = []
    for g in gr:
        l = F[g[0]][0]; rem.update(g)
        if l >= 8: continue
        cx = sum(cen(F[i], fdim)[0] for i in g) // len(g); cy = sum(cen(F[i], fdim)[1] for i in g) // len(g)
        nf = [l + 1, 0, 0]; place(nf, fdim, cx, cy); new.append(nf)
    F = [f for i, f in enumerate(F) if i not in rem and f[0] >= 0] + new
    gr = merge_groups(E, edim, list(range(len(E))))
    if gr is None: return None
    rem = set(); new = []
    for g in gr:
        t = E[g[0]][0]; rem.update(g)
        if t >= 3: continue
        cx = sum(cen(E[i], edim)[0] for i in g) // len(g); cy = sum(cen(E[i], edim)[1] for i in g) // len(g)
        ne = [t + 1, 0, 0]; place(ne, edim, cx, cy); new.append(ne)
    E = [e for i, e in enumerate(E) if i not in rem] + new
    return F, E

def inT(cx, cy, T): return T[0] <= cx < T[0] + 9 and T[1] <= cy < T[1] + 9

def value(F, E, G, EG, Ts):
    gl = set(l for l, n in G); ge = set(t for t, n in EG)
    occE = set(i for i, T in enumerate(Ts) for e in E if e[0] in ge and inT(*cen(e, edim), T))
    occF = set(i for i, T in enumerate(Ts) for f in F if f[0] in gl and inT(*cen(f, fdim), T))
    def info(o, d):
        occ = occE if d is fdim else occF
        tl = [T for i, T in enumerate(Ts) if i not in occ] or Ts
        cx, cy = cen(o, d); bd = 1e9; ins = False
        for T in Ts:
            if inT(cx, cy, T): ins = True
        for T in tl:
            bd = min(bd, math.hypot(cx - T[0] - 4, cy - T[1] - 4))
        return bd, ins
    v = 0.0; maxL = max([l for l, n in G] + [0])
    need = sum(n * 2 ** L for L, n in G)
    if G and sum(2 ** f[0] for f in F) < need: v -= 5000
    for objs, d, goals in ((F, fdim, G), (E, edim, EG)):
        for L, n in goals:
            at = [o for o in objs if o[0] == L]
            ds = sorted(info(o, d) for o in at)
            ins = sum(1 for x in ds if x[1])
            for dd, ii in ds[:n]: v -= dd + (0 if ii else 15) - (40 if ii else 0)
            if ins > n: v -= 100 * (ins - n)
            low = [o for o in objs if o[0] < L]
            short = n - len(at)
            if short > 0:
                v -= 60 * short + 25 * len(low)
                for o in low:
                    cx, cy = cen(o, d); bd = None
                    for q in low:
                        if q is o: continue
                        qx, qy = cen(q, d); dd = math.hypot(qx - cx, qy - cy)
                        if q[0] != o[0]: dd += 20
                        if bd is None or dd < bd: bd = dd
                    v -= 0.6 * (bd or 0) + 0.1 * info(o, d)[0]
    for f in F:
        fx, fy = cen(f, fdim)
        if f[0] > maxL and G:
            v -= 150 * (f[0] - maxL)
            if E:
                v -= min(math.hypot(fx - cen(e, edim)[0], fy - cen(e, edim)[1]) for e in E)
            continue
        for e in E:
            ex, ey = cen(e, edim); dd = math.hypot(fx - ex, fy - ey)
            if dd < 10: v -= (10 - dd) * 2
    return v

def act(obs, available, memory):
    obs = np.asarray(obs)
    F, E, T = parse(obs)
    G, EG = parse_goal(obs)
    if (G or EG) and 'G' not in memory: memory['G'] = (G, EG)
    G, EG = memory.get('G', ([(3, 1)], []))
    if T and 'T' not in memory: memory['T'] = T
    Ts = memory.get('T', [(27, 32)])
    cur = value(F, E, G, EG, Ts)
    nclicks = memory.get('clicks', 0)
    if cur <= -4000 and nclicks > 0 and memory.get('undos', 0) < 6 and "ACTION7" in available:
        memory['undos'] = memory.get('undos', 0) + 1
        memory['clicks'] = nclicks - 1
        return ("ACTION7", None, None)
    cands = set()
    for x in range(0, 64, 2):
        for y in range(10, 63, 2): cands.add((x, y))
    for f in F: cands.add(cen(f, fdim))
    for e in E: cands.add(cen(e, edim))
    scored = []
    for (x, y) in cands:
        if not (10 <= y <= 62 and 0 <= x <= 63): continue
        r = sim(F, E, x, y)
        if r is None: continue
        scored.append((value(r[0], r[1], G, EG, Ts), -abs(x - 32) * 0.001, x, y))
    if not scored:
        return ("ACTION6", 36, 32)
    scored.sort(reverse=True)
    h = hash(obs[:63].tobytes())
    seen = memory.setdefault('seen', {})
    k = seen.get(h, 0); seen[h] = k + 1
    v, _, x, y = scored[min(k, len(scored) - 1)]
    memory['clicks'] = nclicks + 1
    return ("ACTION6", int(y), int(x))
