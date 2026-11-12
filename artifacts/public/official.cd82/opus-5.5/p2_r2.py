import numpy as np
from collections import deque

def _tm(rows):
    m = {'.': -1, '2': 2, 'f': 15, '0': 0}
    return np.array([[m[c] for c in r] for r in rows], dtype=int)

_H = _tm(["2............2"] + ["2ffffffffffff2"] * 7 + ["22222222222222"])
_D = _tm(["......2..........", ".....222.........", "....22f22........",
          "...22fff22.......", "..22fffff22......", ".22fffffff22.....",
          "22fffffffff22....", "..ffffffffff22...", "...ffffffffff22..",
          "....ffffffffff22.", ".....ffffffffff22", "......ffffffff22.",
          ".......ffffff22..", "........ffff22...", ".........ff22....",
          "..........22.....", "..........2......"])
_N = _tm(["2ffff2", "2ffff2", "2ffff2", "220022", ".2222."])
_BASK = [(_H, 25, 24, 180), (_D, 33, 21, 0), (_H, 38, 32, 270), (_D, 33, 40, 90),
         (_H, 25, 45, 0), (_D, 14, 40, 180), (_H, 17, 32, 90), (_D, 14, 21, 270)]
_NOZ = {0: (29, 18, 180), 2: (48, 36, 270), 4: (29, 55, 0), 6: (11, 36, 90)}
_CO = {0: (0, 1), 1: (0, 2), 2: (1, 2), 3: (2, 2), 4: (2, 1), 5: (2, 0), 6: (1, 0), 7: (0, 0)}

def _rend(t, rot):
    k = ((-rot) % 360) // 90
    return np.rot90(t, k) if k else t

def _score(obs, t, x, y):
    h, w = t.shape
    sub = obs[y:y + h, x:x + w]
    if sub.shape != t.shape:
        return 0.0
    m = t == 2
    return float((sub[m] == 2).mean())

def _regions():
    regs = []
    for p in range(8):
        m = np.zeros((10, 10), bool)
        if p == 0: m[0:5, :] = True
        elif p == 4: m[5:10, :] = True
        elif p == 6: m[:, 0:5] = True
        elif p == 2: m[:, 5:10] = True
        else:
            for i in range(10):
                if p == 1: m[i, i:10] = True
                elif p == 3: m[i, 9 - i:10] = True
                elif p == 5: m[i, 0:i + 1] = True
                elif p == 7: m[i, 0:10 - i] = True
        regs.append((p, 'B', m))
    for p, sl in [(0, (slice(0, 3), slice(3, 7))), (4, (slice(7, 10), slice(3, 7))),
                  (6, (slice(3, 7), slice(0, 3))), (2, (slice(3, 7), slice(7, 10)))]:
        m = np.zeros((10, 10), bool); m[sl] = True
        regs.append((p, 'N', m))
    return regs

_REGS = _regions()

def _dist(a, b):
    d = abs(a - b)
    return min(d, 8 - d)

def _move(bp, p):
    d = (p - bp) % 8
    stp = 1 if d <= 4 else -1
    nb = (bp + stp) % 8
    (r1, c1), (r2, c2) = _CO[bp], _CO[nb]
    if r2 < r1: return ("ACTION1", None, None)
    if r2 > r1: return ("ACTION2", None, None)
    if c2 < c1: return ("ACTION3", None, None)
    return ("ACTION4", None, None)

def _plan(target, canvas, sw, bp, curcol, hasnoz):
    rel = 0; diff = 0
    tc = [int(target[k // 10, k % 10]) for k in range(100)]
    for i in range(10):
        for j in range(10):
            if j != i and j != 9 - i:
                rel |= 1 << (i * 10 + j)
                if canvas[i, j] != target[i, j]:
                    diff |= 1 << (i * 10 + j)
    if rel & diff == 0:
        return 0, None
    ops = []
    for p, kind, m in _REGS:
        if kind == 'N' and not hasnoz:
            continue
        cells = [k for k in range(100) if m[k // 10, k % 10] and (rel >> k) & 1]
        bits = 0
        for k in cells: bits |= 1 << k
        ops.append((p, kind, cells, bits))
    tmemo = {}
    def trans(N):
        if N in tmemo: return tmemo[N]
        res = []
        for oi, (p, kind, cells, bits) in enumerate(ops):
            if not (bits & N): continue
            cs = {tc[k] for k in cells if (N >> k) & 1}
            if len(cs) != 1: continue
            c = cs.pop()
            if c not in sw: continue
            res.append((oi, c, N & ~bits))
        tmemo[N] = res
        return res
    D = None
    dist = {rel: 0}; q = deque([rel])
    while q and D is None:
        N = q.popleft()
        for oi, c, N2 in trans(N):
            if N2 not in dist:
                dist[N2] = dist[N] + 1
                if N2 & diff == 0:
                    D = dist[N2]; break
                q.append(N2)
    if D is None:
        return None, None
    seqs = []; dead = set()
    def dfs(N, rem, seq):
        if len(seqs) >= 3000: return True
        if rem == 0:
            if N & diff == 0:
                seqs.append(list(seq)); return True
            return False
        if (N, rem) in dead: return False
        ok = False
        for oi, c, N2 in trans(N):
            seq.append((oi, c))
            if dfs(N2, rem - 1, seq): ok = True
            seq.pop()
            if len(seqs) >= 3000: return True
        if not ok: dead.add((N, rem))
        return ok
    dfs(rel, D, [])
    bestc, bests = None, None
    for s in seqs:
        f = s[::-1]
        cost = 0; p = bp; col = curcol
        for oi, c in f:
            if c != col: cost += 1; col = c
            cost += _dist(p, ops[oi][0]) + 1; p = ops[oi][0]
        if bestc is None or cost < bestc:
            bestc, bests = cost, f
    if not bests:
        return None, None
    oi, c = bests[0]
    return D, (ops[oi][0], ops[oi][1], c)

def act(obs, available, memory):
    obs = np.array(obs, dtype=int)
    target = obs[3:13, 3:13]
    canvas = obs[34:44, 27:37]
    sw = {}
    for x in range(16, 60):
        if (obs[2, x:x + 5] == 4).all() and (obs[6, x:x + 5] == 4).all() and \
           (obs[3:6, x] == 4).all() and (obs[3:6, x + 4] == 4).all():
            c = int(obs[4, x + 2])
            if (obs[3:6, x + 1:x + 4] == c).all() and c != 4 and c not in sw:
                sw[c] = (4, x + 2)
    best, bp = -1, 0
    for p, (t, x, y, r) in enumerate(_BASK):
        s = _score(obs, _rend(t, r), x, y)
        if s > best:
            best, bp = s, p
    t, x, y, r = _BASK[bp]
    rt = _rend(t, r)
    vals = obs[y:y + rt.shape[0], x:x + rt.shape[1]][rt == 15]
    vals = vals[vals >= 0]
    curcol = int(np.bincount(vals).argmax()) if vals.size else -1
    nozclick = None
    if bp in _NOZ:
        nx, ny, nr = _NOZ[bp]
        nt = _rend(_N, nr)
        if _score(obs, nt, nx, ny) >= 0.95:
            memory['noz'] = True
            h, w = nt.shape
            cy, cx = h // 2, w // 2
            if nt[cy, cx] < 0:
                idx = np.argwhere(nt >= 0)[0]; cy, cx = int(idx[0]), int(idx[1])
            nozclick = (ny + cy, nx + cx)
        elif 'noz' not in memory:
            memory['noz'] = False
    known = memory.get('noz', None)

    if known is None:
        D, step = _plan(target, canvas, sw, bp, curcol, False)
        if D is None:
            D, step = _plan(target, canvas, sw, bp, curcol, True)
    else:
        D, step = _plan(target, canvas, sw, bp, curcol, known)

    if D == 0:
        return ("ACTION5", None, None)
    if D is None or step is None:
        # explore: move basket (lets nozzle detection happen / changes state)
        return _move(bp, (bp + 1) % 8)
    p, kind, c = step
    if c != curcol and c in sw:
        r_, c_ = sw[c]
        return ("ACTION6", r_, c_)
    if p != bp:
        return _move(bp, p)
    if kind == 'B':
        return ("ACTION5", None, None)
    if nozclick is not None:
        return ("ACTION6", int(nozclick[0]), int(nozclick[1]))
    # nozzle assumed but not present here: mark absent and move on
    memory['noz'] = False
    return _move(bp, (bp + 1) % 8)
