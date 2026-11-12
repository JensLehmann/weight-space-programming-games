import numpy as np
import time
from collections import deque

PASS = ('.', '2')
SOLIDF = ('o', 'g', '1', 'x', 'y')

def _cls(b):
    c = np.bincount(b.ravel().astype(np.int64) & 15, minlength=16)
    if c[9] >= 2: return 'P'
    if c[7] >= 2: return 'G'
    if c[8] >= 4: return 'g'
    if c[14] >= 4: return 'x'
    if c[12] >= 8: return '1'
    if c[12] >= 2: return '2'
    if c[15] >= 4 or c[11] >= 2:
        if c[11] >= 1 or c[0] >= 1: return 'v'
        if c[3] >= 2: return 'y'
        return 'm'
    if c[10] >= 18: return '.'
    return 'o'

def _win(y, G):
    return range(y - 6, y + 4) if G < 0 else range(y - 4, y + 6)

def act(obs, available, memory):
    t0 = time.time()
    obs = np.asarray(obs)
    ys, xs = np.where(obs[:63] == 9)
    if len(ys) == 0:
        return ("ACTION4" if "ACTION4" in available else available[0], None, None)
    ptop = int(ys.min()) - 1
    pxc = int(xs.min()) // 6
    G = 1 if ptop < 31 else -1
    tiles = []
    for k in range(-7, 8):
        t = ptop + 6 * k
        if t < 0 or t + 6 > 63: continue
        for c in range(11):
            ch = _cls(obs[t:t + 6, c * 6:min(64, c * 6 + 6)])
            if ch == 'P': ch = '.'
            if k == 0 and c == pxc: ch = '.'
            tiles.append((k, c, ch))
    mp = memory.setdefault('map', {})
    if 'py' not in memory:
        py = 0
    else:
        prev = memory['py']; pred = memory.get('pred')
        best = None
        for cand in range(prev - 35, prev + 36):
            s = 0
            for k, c, ch in tiles:
                if ch == 'm': continue
                v = mp.get((cand + k, c))
                if v is None or v == 'm': continue
                s += 1 if v == ch else -1
            sc = s - (0.3 * abs(cand - pred) if pred is not None else 0.05 * abs(cand - prev))
            if best is None or sc > best[0]:
                best = (sc, cand)
        py = best[1]
    for k, c, ch in tiles:
        mp[(py + k, c)] = ch
    memory['py'] = py
    vis = memory.setdefault('vis', {})
    vis[(py, pxc, G)] = vis.get((py, pxc, G), 0) + 1
    gpos = None
    for kk, v in mp.items():
        if v == 'G': gpos = kk; break

    def mkget(m):
        def get(y, x):
            if x < 0 or x > 10: return 'o'
            if (y, x) in m: return m[(y, x)]
            return mp.get((y, x))
        return get

    def fall(get, x, y, g):
        cur = y
        for _ in range(80):
            n = get(cur + g, x)
            if n is None: return ('u', cur)
            if n in PASS: cur += g; continue
            if n == 'G': return ('w', cur)
            if n in ('v', 'm'): return ('l', cur)
            return ('k', cur)
        return ('u', cur)

    def expand(st):
        x, y, g, mods = st
        base = dict(mods)
        get = mkget(base)
        out = []
        for dx, nm in ((-1, 'ACTION3'), (1, 'ACTION4')):
            c = get(y, x + dx)
            if c is None: out.append((nm, 'u', (x + dx, y, g, mods))); continue
            if c == 'G': out.append((nm, 'w', None)); continue
            if c == 'm': continue
            if c in PASS:
                o, e = fall(get, x + dx, y, g)
                out.append((nm, o, (x + dx, e, g, mods)))
        for cy in _win(y, g):
            for cx in range(11):
                ch = get(cy, cx)
                if ch not in ('x', '1', '2', 'g', 'y'): continue
                m = dict(base); adj = (cx == x and cy == y + g)
                nm = ('C', cx, cy)
                if ch == '2':
                    if (cx == x and cy == y) or adj: continue
                    m[(cy, cx)] = '1'
                    out.append((nm, 'k', (x, y, g, frozenset(m.items())))); continue
                if ch in ('x', '1', 'y'):
                    m[(cy, cx)] = '.' if ch != '1' else '2'
                    if ch == 'y':
                        for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nb = (cy + ddy, cx + ddx)
                            if nb != (y, x) and 0 <= nb[1] <= 10 and get(nb[0], nb[1]) == '.':
                                m[nb] = 'y'
                    fm = frozenset(m.items())
                    if adj:
                        o, e = fall(mkget(m), x, cy, g)
                        out.append((nm, o, (x, e, g, fm)))
                    else:
                        out.append((nm, 'k', (x, y, g, fm)))
                    continue
                m[(cy, cx)] = '.'
                g2 = -g
                fm = frozenset(m.items())
                ey = y + g2
                ec = 'g' if (ey == cy and x == cx) else get(ey, x)
                if ec is None: out.append((nm, 'u', (x, ey, g2, fm))); continue
                if ec in SOLIDF: out.append((nm, 'k', (x, y, g2, fm))); continue
                if ec == 'G': out.append((nm, 'w', None)); continue
                if ec in ('v', 'm'): continue
                o, e = fall(mkget(m), x, ey, g2)
                out.append((nm, o, (x, e, g2, fm)))
        return out

    ncache = {}
    def nov(y, g):
        kk = (y, g)
        if kk not in ncache:
            ncache[kk] = sum(1 for r in _win(y, g) for c in range(11) if (r, c) not in mp)
        return ncache[kk]

    def score(st, cost, unk):
        x, y, g, _ = st
        n = nov(y, g) + (6 if unk else 0)
        s = -0.3 * cost - 1.5 * vis.get((y, x, g), 0)
        if gpos is not None:
            s += 0.4 * n - (abs(gpos[0] - y) + 0.5 * abs(gpos[1] - x))
        else:
            s += n
        return s

    start = (pxc, py, G, frozenset())
    root_out = {}
    q = deque([(start, None, 0)]); seen = {start}
    bestleaf = None; win_a = None; nodes = 0
    while q and nodes < 20000 and time.time() - t0 < 6.0:
        st, first, cost = q.popleft(); nodes += 1
        for a, o, nst in expand(st):
            fa = first if first is not None else a
            if first is None: root_out[a] = (o, nst)
            if o == 'w': win_a = fa; break
            if o == 'l': continue
            if o == 'u':
                sc = score(nst, cost + 1, True)
                if bestleaf is None or sc > bestleaf[0]: bestleaf = (sc, fa)
                continue
            if nst in seen: continue
            seen.add(nst)
            sc = score(nst, cost + 1, False)
            if bestleaf is None or sc > bestleaf[0]: bestleaf = (sc, fa)
            q.append((nst, fa, cost + 1))
        if win_a is not None: break
    a = win_a if win_a is not None else (bestleaf[1] if bestleaf else None)
    if a is None:
        a = 'ACTION4' if mkget({})(py, pxc + 1) in PASS else 'ACTION3'
    ro = root_out.get(a)
    memory['pred'] = ro[1][1] if (ro and ro[0] == 'k' and ro[1] is not None) else None
    if isinstance(a, tuple):
        _, cx, cy = a
        row = ptop + (cy - py) * 6 + 3; col = cx * 6 + 3
        if "ACTION6" in available and 0 <= row < 63 and 0 <= col < 64:
            return ("ACTION6", int(row), int(col))
        a = 'ACTION4'
    if a not in available:
        a = available[0]
    return (a, None, None)
