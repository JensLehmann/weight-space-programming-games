import numpy as np
import time, random
from collections import deque

MAPK = {
    1: {1: 4, 2: 3, 3: 1, 4: 2},
    2: {1: 2, 2: 1, 3: 4, 4: 3},
    3: {1: 3, 2: 4, 3: 2, 4: 1},
}

def detect_k(obs):
    ok = lambda a: np.all((a == 0) | (a == 14))
    if ok(obs[0, :]): return 0
    if ok(obs[:, 0]): return 1
    if ok(obs[63, :]): return 2
    if ok(obs[:, 63]): return 3
    return 0

def parse(obs):
    k = detect_k(obs)
    C = np.rot90(obs, -k).copy()
    C[0, :] = C[1, :]
    for n in range(10, 33):
        s = 64 // n; off = (64 - n * s) // 2; e = off + n * s
        if off > 0 or e < 64:
            if not (np.all(C[:off, :] == 1) and np.all(C[:, :off] == 1)
                    and np.all(C[e:, :] == 1) and np.all(C[:, e:] == 1)):
                continue
        G = C[off:e, off:e].reshape(n, s, n, s)
        rep = G[:, s // 2, :, s // 2]
        if np.all(G == rep[:, None, :, None]):
            return k, n, s, off, rep
    return None

def comps(g, n, colors):
    seen = set(); out = []
    for y in range(n):
        for x in range(n):
            if int(g[y, x]) in colors and (x, y) not in seen:
                col = int(g[y, x]); q = [(x, y)]; seen.add((x, y)); cells = []
                while q:
                    cx, cy = q.pop(); cells.append((cx, cy))
                    for ax, ay in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + ax, cy + ay
                        if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in seen and int(g[ny, nx]) in colors:
                            if colors != {11, 13, 0} and int(g[ny, nx]) != col:
                                continue
                            seen.add((nx, ny)); q.append((nx, ny))
                out.append((col, cells))
    return out

def simulate(n, static, drops0, sources, pieces, pos, ncups):
    occ = dict(static)
    for i, p in enumerate(pieces):
        px, py = pos[i]
        for cx, cy in p['rel']:
            occ.setdefault((px + cx, py + cy), (p['t'], i))
    def get(x, y):
        if x < 0 or y < 0 or x >= n or y >= n: return ('B',)
        return occ.get((x, y))
    active = [(x, y, 0, 1) for (x, y) in drops0]
    for (x, y) in sources:
        if get(x, y + 1) is None:
            occ[(x, y + 1)] = ('R',); active.append((x, y + 1, 0, 1))
    filled = set(); hits = 0; it = 0; cap = False
    while active:
        it += 1
        if it > 400 or len(active) > 3000:
            cap = True; break
        new = []
        for (x, y, dx, dy) in active:
            tx, ty = x + dx, y + dy
            t = get(tx, ty)
            o = ((-1, 0), (1, 0)) if dy != 0 else ((0, -1), (0, 1))
            if t is None:
                occ[(tx, ty)] = ('R',); new.append((tx, ty, dx, dy)); continue
            kk = t[0]
            if kk == 'R':
                new.append((tx, ty, dx, dy)); continue
            spread = False
            if kk == 'P':
                spread = True
            elif kk == 'C':
                a = get(x + o[0][0], y + o[0][1]); b = get(x + o[1][0], y + o[1][1])
                if a == t and b == t: filled.add(t[1])
                else: spread = True
            elif kk == 'D':
                a = get(x + o[0][0], y + o[0][1]); b = get(x + o[1][0], y + o[1][1])
                if a == t and b is None:
                    nx, ny = x + dy, y - dx
                    if get(nx, ny) is None: occ[(nx, ny)] = ('R',)
                    new.append((nx, ny, dy, -dx))
                if b == t and a is None:
                    nx, ny = x - dy, y + dx
                    if get(nx, ny) is None: occ[(nx, ny)] = ('R',)
                    new.append((nx, ny, -dy, dx))
                else:
                    spread = True
            elif kk == 'W':
                hits += 1
            if spread:
                for ox, oy in o:
                    cx, cy = x + ox, y + oy
                    if get(cx, cy) is None:
                        occ[(cx, cy)] = ('R',); new.append((cx, cy, dx, dy))
        active = new
    f = len(filled)
    win = (not cap) and hits == 0 and f == ncups
    return 10 * f - 3 * hits - (50 if cap else 0), win

def plan(obs, seed):
    t0 = time.time()
    rng = random.Random(seed)
    P = parse(obs)
    if P is None: return None
    k, n, s, off, g = P
    static = {}; drops0 = []; sources = []; blocked = set()
    for y in range(n):
        for x in range(n):
            c = int(g[y, x])
            if c == 4: static[(x, y)] = ('S',); sources.append((x, y)); blocked.add((x, y))
            elif c == 6: static[(x, y)] = ('R',); drops0.append((x, y)); blocked.add((x, y))
            elif c in (1, 14): static[(x, y)] = ('W',); blocked.add((x, y))
    cups = []
    for ci, (col, cells) in enumerate(comps(g, n, {11, 13, 0})):
        for c in cells: static[c] = ('C', ci); blocked.add(c)
        xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
        cups.append((min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
    pcs = []
    for col, cells in comps(g, n, {8, 9, 15}):
        xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
        x0, y0 = min(xs), min(ys); w = max(xs) - x0 + 1; h = max(ys) - y0 + 1
        t = 'P' if (w == 1 or h == 1) and col != 15 else 'D'
        pcs.append({'t': t, 'rel': [(cx - x0, cy - y0) for cx, cy in cells], 'pos': (x0, y0),
                    'w': w, 'h': h, 'sel': col == 9})
    pcs.sort(key=lambda p: 0 if p['t'] == 'P' else 1)
    ncups = len(cups); NP = len(pcs)

    def valid(p, x, y):
        if y < 3: return False
        for cx, cy in p['rel']:
            ax, ay = x + cx, y + cy
            if ax < 0 or ay < 0 or ax >= n or ay >= n or (ax, ay) in blocked: return False
        for rx, ry, rw, rh in cups:
            if x < rx + rw + 1 and x + p['w'] > rx - 1 and y < ry + rh + 1 and y + p['h'] > ry - 1:
                return False
        return True

    reach = []
    for p in pcs:
        st = p['pos']; dist = {st: 0}; par = {st: None}; q = deque([st])
        while q:
            cx, cy = q.popleft()
            for a, (dx, dy) in ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in dist and valid(p, nx, ny):
                    dist[(nx, ny)] = dist[(cx, cy)] + 1; par[(nx, ny)] = ((cx, cy), a); q.append((nx, ny))
        reach.append((dist, par))
    cands = [list(r[0].keys()) for r in reach]

    def cost(pos):
        c = 0
        for i, q in enumerate(pos):
            if q != pcs[i]['pos']:
                c += reach[i][0][q] + (0 if pcs[i]['sel'] else 1)
        return c
    cache = {}
    def ev(pos):
        t = tuple(pos); r = cache.get(t)
        if r is None:
            r = simulate(n, static, drops0, sources, pcs, pos, ncups); cache[t] = r
        return r

    deadline = t0 + 7.4
    cur = [p['pos'] for p in pcs]
    sc, win = ev(cur)
    best = (win, sc, -cost(cur)); bestpos = list(cur)
    stall = 0
    while not best[0] and NP and time.time() < deadline:
        order = list(range(NP)); rng.shuffle(order)
        moved = False
        for i in order:
            if time.time() > deadline: break
            ci = cands[i]
            sample = ci if len(ci) <= 90 else rng.sample(ci, 90)
            bi = None
            for q in sample:
                if q == cur[i]: continue
                pos2 = list(cur); pos2[i] = q
                s2, w2 = ev(pos2)
                if w2: bi = (s2, q, True); break
                if bi is None or s2 > bi[0] or (s2 == bi[0] and rng.random() < 0.3):
                    bi = (s2, q, False)
            if bi is None: continue
            if bi[2] or bi[0] > sc or (bi[0] == sc and rng.random() < 0.5):
                if bi[2] or bi[0] > sc: moved = True
                cur[i] = bi[1]; sc = bi[0]; win = bi[2]
                key = (win, sc, -cost(cur))
                if key > best: best = key; bestpos = list(cur)
                if win: break
        if best[0]: break
        if not moved:
            stall += 1
            if stall >= 2:
                stall = 0
                if rng.random() < 0.5: cur = list(bestpos)
                for _ in range(rng.randint(1, 2)):
                    i = rng.randrange(NP); cur[i] = rng.choice(cands[i])
                sc, win = ev(cur)
                key = (win, sc, -cost(cur))
                if key > best: best = key; bestpos = list(cur)
        else:
            stall = 0
    cur = list(bestpos)
    if best[0]:
        for i in range(NP):
            if time.time() > t0 + 8.8: break
            if cur[i] == pcs[i]['pos']: continue
            dcur = reach[i][0][cur[i]]
            for q in sorted(cands[i], key=lambda z: reach[i][0][z]):
                if reach[i][0][q] >= dcur or time.time() > t0 + 8.8: break
                pos2 = list(cur); pos2[i] = q
                if ev(pos2)[1]: cur = pos2; break

    inv = {v: kk for kk, v in MAPK.get(k, {}).items()} if k else None
    def mapa(a): return a if not k else inv[a]
    def disp(gx, gy):
        px = off + gx * s + s // 2; py = off + gy * s + s // 2
        if k == 0: return px, py
        if k == 1: return py, 63 - px
        if k == 2: return 63 - px, 63 - py
        return 63 - py, px
    queue = []
    order = sorted(range(NP), key=lambda i: 0 if pcs[i]['sel'] else 1)
    now = [p['pos'] for p in pcs]
    selected = next((i for i in range(NP) if pcs[i]['sel']), None)
    for i in order:
        if cur[i] == pcs[i]['pos']: continue
        if selected != i:
            px, py = now[i]; choice = None
            for cx, cy in pcs[i]['rel']:
                ax, ay = px + cx, py + cy; bad = False
                for j in range(NP):
                    if j == i: continue
                    qx, qy = now[j]
                    if qx <= ax < qx + pcs[j]['w'] and qy <= ay < qy + pcs[j]['h']: bad = True
                if not bad: choice = (ax, ay); break
            if choice is None: choice = (px + pcs[i]['rel'][0][0], py + pcs[i]['rel'][0][1])
            hx, hy = disp(*choice)
            queue.append(("ACTION6", hy, hx)); selected = i
        path = []; node = cur[i]; par = reach[i][1]
        while par[node] is not None:
            prev, a = par[node]; path.append(a); node = prev
        for a in reversed(path):
            queue.append(("ACTION%d" % mapa(a), None, None))
        now[i] = cur[i]
    queue.append(("ACTION5", None, None))
    return queue

def act(obs, available, memory):
    obs = np.asarray(obs)
    q = memory.get('queue')
    if not q:
        cnt = memory.get('plans', 0); memory['plans'] = cnt + 1
        try:
            q = plan(obs, 12345 + 7919 * cnt)
        except Exception:
            q = None
        if not q:
            q = [("ACTION5", None, None)]
    a = q.pop(0)
    memory['queue'] = q
    return a
