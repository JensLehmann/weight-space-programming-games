import numpy as np
from collections import deque

EX = {3, 4, 5, 8, 14}

def _geo(f):
    minrun = 99
    for y in range(2, 64):
        row = f[y]; x = 1
        while x < 64:
            c = row[x]; x2 = x
            while x2 < 64 and row[x2] == c:
                x2 += 1
            if c not in EX and x > 1 and x2 < 64:
                minrun = min(minrun, x2 - x)
            x = x2
    s = minrun if minrun < 99 else 1
    s = max(1, min(4, s))
    P = 3 * s
    cx = {}; cy = {}
    for y in range(0, 64 - 2 * s + 1):
        for x in range(1, 64 - 2 * s + 1):
            c = f[y, x]
            if c in EX:
                continue
            if x > 1 and f[y, x - 1] == c:
                continue
            if y > 0 and f[y - 1, x] == c:
                continue
            blk = f[y:y + 2 * s, x:x + 2 * s]
            if np.all(blk == c):
                cx[x % P] = cx.get(x % P, 0) + 1
                cy[y % P] = cy.get(y % P, 0) + 1
    ox = max(cx, key=cx.get) if cx else 0
    oy = max(cy, key=cy.get) if cy else 0
    return s, ox, oy

def _parse(f, s, ox, oy):
    P = 3 * s
    cells = {}; markers = []; U = []
    for py in range(oy, 64 - 2 * s + 1, P):
        for px in range(ox, 64 - 2 * s + 1, P):
            i = (px - ox) // P; j = (py - oy) // P
            U.append((i, j))
            blk = f[py:py + 2 * s, max(px, 1):px + 2 * s]
            if blk.size:
                v = int(blk.flat[0])
                if v not in EX and np.all(blk == v):
                    cells[(i, j)] = v
            cs = [(py - s, px - s), (py - s, px + 2 * s), (py + 2 * s, px - s), (py + 2 * s, px + 2 * s)]
            if all(0 <= a < 64 and 1 <= b < 64 for a, b in cs):
                vals = [int(f[a, b]) for a, b in cs]
                if len(set(vals)) == 1 and vals[0] not in EX:
                    markers.append(((i, j), vals[0]))
    return cells, markers, U

def _buttons(f):
    seen = np.zeros((64, 64), bool); out = []
    for y in range(2, 64):
        for x in range(1, 64):
            c = f[y, x]
            if c not in (8, 14) or seen[y, x]:
                continue
            q = [(y, x)]; seen[y, x] = True; comp = []
            while q:
                a, b = q.pop(); comp.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 2 <= na < 64 and 1 <= nb < 64 and not seen[na, nb] and f[na, nb] == c:
                        seen[na, nb] = True; q.append((na, nb))
            if len(comp) < 3:
                continue
            my = sum(p[0] for p in comp) / len(comp); mx = sum(p[1] for p in comp) / len(comp)
            best = min(comp, key=lambda p: (p[0] - my) ** 2 + (p[1] - mx) ** 2)
            out.append((int(c), best[0], best[1]))
    return out

def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def act(obs, available, memory):
    f = np.array(obs, dtype=int)
    if 'geo' not in memory:
        memory['geo'] = _geo(f)
        bt = _buttons(f)
        memory['btns'] = [(y, x) for c, y, x in bt]
        Ls = [k for k, b in enumerate(bt) if b[0] == 8]
        Rs = [k for k, b in enumerate(bt) if b[0] == 14]
        pairs = sorted(((bt[a][1] - bt[b][1]) ** 2 + (bt[a][2] - bt[b][2]) ** 2, a, b) for a in Ls for b in Rs)
        part = {}
        for d, a, b in pairs:
            if a not in part and b not in part:
                part[a] = b; part[b] = a
        memory['part'] = part
        memory['cand'] = {}; memory['chg'] = {}
        memory['n'] = 0
    s, ox, oy = memory['geo']
    btns = memory['btns']; part = memory['part']
    cand = memory['cand']; chg = memory['chg']
    if not btns:
        return ('ACTION6', 32, 32)
    cells, markers, U = _parse(f, s, ox, oy)
    gcs = sorted(set(c for _, c in markers))
    G = tuple(frozenset(c for c, v in cells.items() if v == gc) for gc in gcs)

    def upd(b, X, Y):
        byc = {}
        for d in U:
            byc.setdefault(Y.get(d), set()).add(d)
        cb = cand.setdefault(b, {}); ch = chg.setdefault(b, set())
        for c in U:
            xc = X.get(c)
            sset = byc.get(xc, set())
            if c in cb:
                nset = cb[c] & sset
                cb[c] = nset if nset else set(sset)
            else:
                cb[c] = set(sset)
            if xc != Y.get(c):
                ch.add(c)

    last = memory.get('last')
    if last is not None:
        b, B = last
        upd(b, B, cells)
        if b in part:
            upd(part[b], cells, B)

    models = {}

    def model(b):
        if b in models:
            return models[b]
        if b not in cand:
            models[b] = None
            return None
        cb = cand[b]; ch = chg.get(b, set())
        fixed = {}; taken = set()
        for _ in range(5):
            prog = False
            for c in ch:
                if c in fixed:
                    continue
                st = cb.get(c, {c}) - taken
                st.discard(c)
                if len(st) == 1:
                    d = next(iter(st)); fixed[c] = d; taken.add(d); prog = True
            if not prog:
                break
        models[b] = (cb, ch, fixed, taken, {})
        return models[b]

    def get(b, c):
        m = model(b)
        if m is None:
            return c, True
        cb, ch, fixed, taken, memo = m
        if c in memo:
            return memo[c]
        if c in fixed:
            r = (fixed[c], False)
        else:
            st = cb.get(c, {c})
            if c not in ch and c in st:
                r = (c, False)
            else:
                s2 = st - taken - {c}
                if not s2:
                    s2 = st - {c}
                if not s2:
                    s2 = st if st else {c}
                d = min(s2, key=lambda z: (_cheb(c, z), abs(c[0] - z[0]) + abs(c[1] - z[1]), z))
                r = (d, len(s2) > 1)
        memo[c] = r
        return r

    def pred(state, b):
        out = []; unc = False
        for g in state:
            ns = set()
            for c in g:
                d, u = get(b, c)
                unc = unc or u
                ns.add(d)
            if len(ns) != len(g):
                return 'X', unc
            out.append(frozenset(ns))
        return tuple(out), unc

    def win(state):
        for (cell, c) in markers:
            if cell not in state[gcs.index(c)]:
                return False
        return True

    nb = len(btns)
    parent = {G: None}
    dq = deque([G]); unk = None; choice = None; exp = 0
    if markers:
        while dq and exp < 20000:
            st = dq.popleft(); exp += 1
            for b in range(nb):
                nx, u = pred(st, b)
                if u and unk is None:
                    unk = (st, b)
                if nx == 'X' or nx in parent:
                    continue
                parent[nx] = (st, b)
                if win(nx):
                    choice = nx; break
                dq.append(nx)
            if choice is not None:
                break

    def first(st, fallback):
        act_b = fallback
        while parent[st] is not None:
            p, bb = parent[st]
            act_b = bb; st = p
        return act_b

    if choice is not None:
        b = first(choice, 0)
    elif unk is not None:
        b = first(unk[0], unk[1])
    else:
        memory['n'] += 1
        b = (memory['n'] * 7 + 3) % nb
    memory['last'] = (b, cells)
    y, x = btns[b]
    return ('ACTION6', int(y), int(x))
