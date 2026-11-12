import numpy as np, time, heapq
from collections import deque

DIRS = {'ACTION1': (0, -1), 'ACTION2': (0, 1), 'ACTION3': (-1, 0), 'ACTION4': (1, 0)}

def find_heads(g):
    res = []
    for y in range(59):
        for x in range(59):
            c = g[y, x]
            if c != 6 and c != 15: continue
            if g[y+2, x+2] != c or g[y+3, x+3] != c: continue
            b = g[y:y+6, x:x+6]
            ring = [b[1, 1], b[1, 4], b[4, 1], b[4, 4]]
            if len(set(ring)) != 1 or ring[0] not in (0, 4): continue
            sides = {(0, -1): b[0, :], (0, 1): b[5, :], (-1, 0): b[:, 0], (1, 0): b[:, 5]}
            full = [k for k, s in sides.items() if (s == c).all()]
            if len(full) != 3: continue
            d = [k for k in sides if k not in full][0]
            s = sides[d]
            if not (s[0] == c and s[5] == c and s[1] == c and s[4] == c): continue
            res.append((x, y, int(c), d, ring[0] == 0))
    return res

def seg_at(g, x, y, hz):
    if x < 0 or y < 0 or x + 5 > 63 or y + 5 > 63: return False
    if hz: pts = [(y+2, x), (y+3, x), (y+2, x+5), (y+3, x+5)]
    else: pts = [(y, x+2), (y, x+3), (y+5, x+2), (y+5, x+3)]
    v = [int(g[p]) for p in pts]
    return all(a in (1, 2, 3) for a in v) and v[0] != v[1] and v[2] != v[3]

class Ctx: pass

def parse(g):
    heads = find_heads(g)
    play = [h for h in heads if h[1] < 53]
    bottom = [h for h in heads if h[1] >= 53]
    ctrl = []
    for p in sorted(play, key=lambda h: (h[2], h[0], h[1])):
        bm = [b for b in bottom if b[2] == p[2]]
        if bm: ctrl.append((p, bm[0]))
    if not ctrl: raise ValueError
    mask = (g[:53] == 4).copy()
    for h in play: mask[h[1]:h[1]+6, h[0]:h[0]+6] = False
    ys, xs = np.nonzero(mask)
    ctx = Ctx()
    ctx.g = g
    ctx.x0, ctx.x1, ctx.y0, ctx.y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    ctx.boxes = [(h[0], h[1]) for h in play]
    ctx.walls = set(); blocks = []
    for y in range(ctx.y0, ctx.y1 - 5, 6):
        for x in range(ctx.x0, ctx.x1 - 5, 6):
            if (g[y:y+6, x:x+6] == 5).all(): ctx.walls.add((x, y)); continue
            c = int(g[y+1, x+1])
            if c not in (4, 5) and g[y+1, x+4] == c and g[y+4, x+1] == c and g[y+4, x+4] == c:
                blocks.append((x, y, c))
    ctx.dirs = []; ctx.targets = []; hs = []; sel = 0; owned = set()
    for i, (p, b) in enumerate(ctrl):
        dx, dy = p[3]; hz = dx != 0
        ctx.dirs.append((dx, dy))
        L = 1
        owned.add((p[0], p[1], hz))
        while seg_at(g, p[0] + L*dx*6, p[1] + L*dy*6, hz):
            owned.add((p[0] + L*dx*6, p[1] + L*dy*6, hz)); L += 1
        hs.append((p[0], p[1], L))
        if p[4]: sel = i
        bx, by, _, (bdx, bdy), _ = b
        t = []; k = 1
        while True:
            x = bx + k*bdx*6; y = by + k*bdy*6
            if x < 0 or y < 0 or x + 5 > 63 or y + 5 > 63: break
            c = int(g[y+1, x+1])
            if c not in (4, 5) and g[y+1, x+4] == c and g[y+4, x+1] == c and g[y+4, x+4] == c:
                t.append(c); k += 1
            else: break
        ctx.targets.append(t)
    ox = ctx.x0 % 6; oy = ctx.y0 % 6
    ctx.sseg = {}
    for y in range(oy, 48, 6):
        for x in range(ox, 59, 6):
            for hz in (True, False):
                if (x, y, hz) not in owned and seg_at(g, x, y, hz):
                    ctx.sseg.setdefault((x, y), []).append((hz, ('s', x, y, hz)))
    return ctx, (sel, tuple(hs), tuple(sorted(blocks)))

def inb(ctx, x, y):
    return x >= ctx.x0 and x + 6 <= ctx.x1 and y >= ctx.y0 and y + 6 <= ctx.y1 and (x, y) not in ctx.walls

def track_ok(ctx, hx, hy, mx, my):
    px = hx + 2 + 3*mx; py = hy + 2 + 3*my
    for bx, by in ctx.boxes:
        if bx <= px < bx + 6 and by <= py < by + 6:
            px += mx; py += my; break
    if not (0 <= px < 63 and 0 <= py < 53): return False
    g = ctx.g; v = g[py, px]
    if v not in (2, 3): return False
    w = g[py, px+1] if my != 0 else g[py+1, px]
    return w == v

def sim(ctx, st, a):
    sel, heads, blocks = st
    if isinstance(a, tuple):
        return (a[1], heads, blocks)
    mx, my = DIRS[a]
    hx, hy, L = heads[sel]; rx, ry = ctx.dirs[sel]
    bd = {(x, y): c for x, y, c in blocks}
    segpos = {k: list(v) for k, v in ctx.sseg.items()}
    for i, (x, y, l) in enumerate(heads):
        dx, dy = ctx.dirs[i]; hz = dx != 0
        for k in range(l):
            segpos.setdefault((x + k*dx*6, y + k*dy*6), []).append((hz, (i, k)))
    D = (mx*6, my*6); hm = mx != 0
    opq = set()

    def pb(pos, hst):
        if pos in opq: return True
        n = (pos[0] + D[0], pos[1] + D[1])
        if not inb(ctx, n[0], n[1]): return False
        for q in (pos, n):
            for hz, sid in segpos.get(q, ()):
                if sid != hst and hz != hm: return False
        if n in bd and not pb(n, None): return False
        opq.add(pos); return True

    def ps(k):
        sx = hx + k*rx*6; sy = hy + k*ry*6; hz = rx != 0
        if not inb(ctx, sx + D[0], sy + D[1]):
            if not ((rx, ry) == (-mx, -my) or not inb(ctx, sx, sy)): return False
        for q in ((sx, sy), (sx + D[0], sy + D[1])):
            if q in bd:
                if not pb(q, (sel, k)) and hz != hm: return False
        return True

    nh = list(heads)
    if (mx, my) == (rx, ry):
        lx = hx + (L-1)*rx*6 + D[0]; ly = hy + (L-1)*ry*6 + D[1]
        if not inb(ctx, lx, ly): return None
        for k in range(L): ps(k)
        nh[sel] = (hx, hy, L + 1)
    elif (mx, my) == (-rx, -ry):
        if L == 1: return None
        for k in range(1, L): ps(k)
        nh[sel] = (hx, hy, L - 1)
    else:
        if not track_ok(ctx, hx, hy, mx, my): return None
        for k in range(L):
            if not ps(k): return None
        nh[sel] = (hx + D[0], hy + D[1], L)
    nb = []
    for (x, y), c in bd.items():
        if (x, y) in opq: nb.append((x + D[0], y + D[1], c))
        else: nb.append((x, y, c))
    ns = (sel, tuple(nh), tuple(sorted(nb)))
    if ns == st: return None
    return ns

def heur(ctx, st):
    sel, heads, blocks = st
    bd = {(x, y): c for x, y, c in blocks}
    m = 0; ds = 0
    for i, (x, y, l) in enumerate(heads):
        dx, dy = ctx.dirs[i]; t = ctx.targets[i]
        f = []
        for k in range(l):
            p = (x + k*dx*6, y + k*dy*6)
            if p in bd: f.append(bd[p])
        for j in range(len(t)):
            if j < len(f) and f[j] == t[j]: continue
            m += 1
            best = 99
            for (bx, by, c) in blocks:
                if c != t[j]: continue
                if dx != 0:
                    d = abs(by - y) // 6
                    if (bx - x) * dx < 0: d += abs(bx - x) // 6
                else:
                    d = abs(bx - x) // 6
                    if (by - y) * dy < 0: d += abs(by - y) // 6
                if d < best: best = d
            ds += best if best < 99 else 10
    return 3 * m + ds, m == 0

def path_to(prev, s):
    path = []
    while prev[s] is not None:
        p, a = prev[s]; path.append(a); s = p
    return path[::-1]

def solve(ctx, st0, tl):
    t0 = time.time()
    h0, ok = heur(ctx, st0)
    if ok: return [], True
    prev = {st0: None}; gc = {st0: 0}
    pq = [(2 * h0, 0, 0, st0)]; cnt = 1
    best = None; bh = h0
    n_ctrl = len(st0[1])
    while pq:
        if time.time() - t0 > tl: break
        _, _, gs, s = heapq.heappop(pq)
        if gs > gc.get(s, 1e9): continue
        acts = list(DIRS.keys()) + [('C', i) for i in range(n_ctrl) if i != s[0]]
        for a in acts:
            n = sim(ctx, s, a)
            if n is None: continue
            ng = gs + 1
            if ng >= gc.get(n, 1e9): continue
            gc[n] = ng; prev[n] = (s, a)
            h, ok = heur(ctx, n)
            if ok: return path_to(prev, n), True
            if h < bh: bh = h; best = n
            heapq.heappush(pq, (ng + 2 * h, cnt, ng, n)); cnt += 1
    if best is None: return None, False
    return path_to(prev, best), False

def act(obs, available, memory):
    g = np.array(obs, dtype=int)
    try:
        ctx, st = parse(g)
    except Exception:
        return ('ACTION4', None, None)
    vis = memory.setdefault('vis', {})
    vis[st] = vis.get(st, 0) + 1
    plan = memory.get('plan')
    if not plan or memory.get('exp') != st:
        plan, ok = solve(ctx, st, 6.5)
        memory['ctx'] = ctx
        if not plan:
            cands = []
            n_ctrl = len(st[1])
            for a in list(DIRS.keys()) + [('C', i) for i in range(n_ctrl) if i != st[0]]:
                try: n = sim(ctx, st, a)
                except Exception: n = None
                if n is not None: cands.append((vis.get(n, 0), isinstance(a, tuple), len(cands), a))
            if not cands:
                memory['plan'] = None
                return ('ACTION4', None, None)
            cands.sort()
            plan = [cands[0][3]]
    else:
        ctx = memory.get('ctx', ctx)
    a = plan[0]
    memory['plan'] = plan[1:]
    try:
        memory['exp'] = sim(ctx, st, a)
    except Exception:
        memory['exp'] = None
    if isinstance(a, tuple):
        hx, hy, _ = st[1][a[1]]
        return ('ACTION6', hy + 2, hx + 2)
    return (a, None, None)
