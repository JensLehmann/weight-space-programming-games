import numpy as np
import time, random
from collections import deque

MFK = {1: {1: 4, 2: 3, 3: 1, 4: 2}, 2: {1: 2, 2: 1, 3: 4, 4: 3}, 3: {1: 3, 2: 4, 3: 2, 4: 1}}
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def comps(C, colors, W):
    seen = set(); out = []
    for y in range(W):
        for x in range(W):
            if C[y][x] in colors and (x, y) not in seen:
                q = [(x, y)]; seen.add((x, y)); cs = []
                while q:
                    a, b = q.pop(); cs.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < W and 0 <= nb < W and (na, nb) not in seen and C[nb][na] in colors:
                            seen.add((na, nb)); q.append((na, nb))
                out.append(cs)
    return out


def parse(obs):
    obs = np.array(obs, dtype=int)
    k = 0
    edges = [obs[0, :], obs[:, 0], obs[63, :], obs[:, 63]]
    for kk, e in enumerate(edges):
        if np.all((e == 0) | (e == 14)):
            k = kk; break
    orig = np.rot90(obs, -k)
    for W in range(6, 33):
        s = 64 // W; off = (64 - W * s) // 2
        C = orig[off + s // 2: off + W * s: s, off + s // 2: off + W * s: s][:W, :W]
        exp = np.full((64, 64), 1)
        exp[off:off + W * s, off:off + W * s] = np.kron(C, np.ones((s, s), dtype=int))
        if np.array_equal(exp[1:], orig[1:]):
            return k, W, s, off, C.tolist()
    W = 16
    s = 4; off = 0
    C = orig[2::4, 2::4].tolist()
    return k, W, s, off, C


def plan(obs, t0):
    k, W, s, off, C = parse(obs)
    pieces = []
    for cs in comps(C, (8, 15, 9), W):
        # split by color
        pass
    allp = comps(C, (8,), W) + comps(C, (15,), W) + comps(C, (9,), W)
    sel_idx = None
    for cs in allp:
        xs = [a for a, b in cs]; ys = [b for a, b in cs]
        x0, y0 = min(xs), min(ys); w = max(xs) - x0 + 1; h = max(ys) - y0 + 1
        col = C[cs[0][1]][cs[0][0]]
        isL = (col == 15) or (col == 9 and len(cs) == 3 and w == 2 and h == 2)
        pieces.append(dict(cells=[(a - x0, b - y0) for a, b in cs], pos=(x0, y0), w=w, h=h, L=isL))
        if col == 9:
            sel_idx = len(pieces) - 1
    base = [0] * (W * W)
    cups = []
    for cs in comps(C, (11, 13, 0), W):
        cid = len(cups)
        xs = [a for a, b in cs]; ys = [b for a, b in cs]
        cups.append((min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
        for a, b in cs:
            base[b * W + a] = 10 + cid
    liquids = []; sources = []
    for y in range(W):
        for x in range(W):
            c = C[y][x]
            if c in (1, 14): base[y * W + x] = 1
            elif c == 6: base[y * W + x] = 2; liquids.append((x, y))
            elif c == 4: base[y * W + x] = 3; sources.append((x, y))
            elif c in (12, 8, 9, 15, 11, 13, 0): pass
            else: base[y * W + x] = 4
    n = len(pieces)

    def legal(i, x, y):
        p = pieces[i]
        if y < 3: return False
        for a, b in p['cells']:
            X, Y = x + a, y + b
            if not (0 <= X < W and 0 <= Y < W) or base[Y * W + X] != 0: return False
        for rx, ry, cw, ch in cups:
            if x < rx + cw + 1 and x + p['w'] > rx - 1 and y < ry + ch + 1 and y + p['h'] > ry - 1:
                return False
        return True

    reach = []
    for i in range(n):
        st = pieces[i]['pos']; par = {st: None}; dist = {st: 0}; q = deque([st])
        while q:
            cx, cy = q.popleft()
            for aid, (dx, dy) in DIRS.items():
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in par and legal(i, nx, ny):
                    par[(nx, ny)] = ((cx, cy), aid); dist[(nx, ny)] = dist[(cx, cy)] + 1; q.append((nx, ny))
        reach.append((par, dist))

    def sim(cfg):
        g = base[:]
        for i, (x, y) in enumerate(cfg):
            code = 100 + i if pieces[i]['L'] else 5
            for a, b in pieces[i]['cells']:
                g[(y + b) * W + x + a] = code
        parts = [(x, y, 0, 1) for x, y in liquids]
        for sx, sy in sources:
            if sy + 1 < W and g[(sy + 1) * W + sx] == 0:
                g[(sy + 1) * W + sx] = 2; parts.append((sx, sy + 1, 0, 1))
        filled = set(); wall = False; offg = False; it = 0

        def at(x, y):
            if 0 <= x < W and 0 <= y < W: return g[y * W + x]
            return -1
        while parts and it < 300:
            it += 1; new = []
            for x, y, dx, dy in parts:
                tx, ty = x + dx, y + dy
                if not (0 <= tx < W and 0 <= ty < W):
                    offg = True; continue
                c = g[ty * W + tx]
                if c == 0:
                    g[ty * W + tx] = 2; new.append((tx, ty, dx, dy)); continue
                if c == 2:
                    new.append((tx, ty, dx, dy)); continue
                sd = ((-1, 0), (1, 0)) if dy != 0 else ((0, -1), (0, 1))
                def spread():
                    for a, b in sd:
                        if at(x + a, y + b) == 0:
                            g[(y + b) * W + x + a] = 2; new.append((x + a, y + b, dx, dy))
                if c == 5:
                    spread(); continue
                if 10 <= c < 100:
                    if at(x + sd[0][0], y + sd[0][1]) == c and at(x + sd[1][0], y + sd[1][1]) == c:
                        filled.add(c)
                    else:
                        spread()
                    continue
                if c >= 100:
                    o = at(x + sd[0][0], y + sd[0][1]); f = at(x + sd[1][0], y + sd[1][1])
                    if o == c and f == 0:
                        nx, ny = x + dy, y - dx
                        if 0 <= nx < W and 0 <= ny < W: g[ny * W + nx] = 2
                        new.append((nx, ny, dy, -dx))
                    if f == c and o == 0:
                        nx, ny = x - dy, y + dx
                        if 0 <= nx < W and 0 <= ny < W: g[ny * W + nx] = 2
                        new.append((nx, ny, -dy, dx))
                    else:
                        spread()
                    continue
                if c == 1:
                    wall = True
            parts = new
        return filled, wall, offg

    def overlap(cfg):
        seen = set()
        for i, (x, y) in enumerate(cfg):
            for a, b in pieces[i]['cells']:
                if (x + a, y + b) in seen: return True
                seen.add((x + a, y + b))
        return False

    def cost(cfg):
        c = 0
        for i in range(n):
            if cfg[i] != pieces[i]['pos']:
                c += reach[i][1][cfg[i]] + (0 if i == sel_idx else 1)
        return c

    nc = len(cups)

    def evalc(cfg):
        f, w, o = sim(cfg)
        nf = len(f)
        return nf * 4 - 2 * w - o, (nf == nc and not w and not o), (nf == nc and not w)

    cur = [p['pos'] for p in pieces]
    best_strict = None; best_len = None
    curv, st, le = evalc(cur)
    if st: best_strict = (0, cur[:])
    elif le: best_len = (0, cur[:])
    rnd = random.Random(1)
    while best_strict is None and n > 0 and nc > 0 and time.time() - t0 < 6.5:
        bv = None; bc = None
        for i in range(n):
            for pos in reach[i][1]:
                if pos == cur[i]: continue
                cfg = cur[:]; cfg[i] = pos
                if overlap(cfg): continue
                v, st, le = evalc(cfg)
                cc = cost(cfg)
                if st and (best_strict is None or cc < best_strict[0]): best_strict = (cc, cfg)
                if le and (best_len is None or cc < best_len[0]): best_len = (cc, cfg)
                if bv is None or v > bv[0] or (v == bv[0] and cc < bv[1]):
                    bv = (v, cc); bc = cfg
            if time.time() - t0 > 7.5: break
        if best_strict: break
        if bv is not None and bv[0] > curv:
            cur = bc; curv = bv[0]
        else:
            for _ in range(20):
                i = rnd.randrange(n); pos = rnd.choice(list(reach[i][1].keys()))
                cfg = cur[:]; cfg[i] = pos
                if not overlap(cfg): cur = cfg; break
            curv = evalc(cur)[0]
    if best_strict: target = best_strict[1]
    elif best_len: target = best_len[1]
    else: target = cur

    # build action plan
    inv = {}
    if k in MFK:
        for p_, a_ in MFK[k].items(): inv[a_] = p_
    else:
        inv = {1: 1, 2: 2, 3: 3, 4: 4}
    idxarr = np.rot90(np.arange(4096).reshape(64, 64), k)
    actions = []
    positions = [p['pos'] for p in pieces]
    order = [i for i in range(n) if target[i] != positions[i]]
    order.sort(key=lambda i: 0 if i == sel_idx else 1)
    for i in order:
        if i != sel_idx:
            x, y = positions[i]
            cell = None
            for a, b in pieces[i]['cells']:
                X, Y = x + a, y + b; ok = True
                for j in range(n):
                    if j == i: continue
                    jx, jy = positions[j]
                    if jx <= X < jx + pieces[j]['w'] and jy <= Y < jy + pieces[j]['h']: ok = False
                if ok: cell = (X, Y); break
            if cell is None: cell = (x + pieces[i]['cells'][0][0], y + pieces[i]['cells'][0][1])
            pr = off + cell[1] * s + s // 2; pc = off + cell[0] * s + s // 2
            loc = np.argwhere(idxarr == pr * 64 + pc)[0]
            actions.append(('ACTION6', int(loc[0]), int(loc[1])))
        path = []; node = target[i]; par = reach[i][0]
        while par[node] is not None:
            prev, aid = par[node]; path.append(aid); node = prev
        path.reverse()
        for aid in path:
            actions.append(('ACTION%d' % inv[aid], None, None))
        positions[i] = target[i]
    actions.append(('ACTION5', None, None))
    return actions


def act(obs, available, memory):
    t0 = time.time()
    pl = memory.get('plan')
    if not pl:
        try:
            pl = plan(obs, t0)
        except Exception:
            pl = [('ACTION5', None, None)]
        memory['plan'] = pl
    a = pl.pop(0)
    return a
