import numpy as np
import time
from collections import deque

E,W,B,G,T,P,S,X,GOAL,U = range(10)
SOL = (W,B,G,T,X)

def _cls(obs, top, left):
    cell = obs[top:top+6, left:min(left+6,64)]
    c = int(obs[top+3, min(left+3,63)])
    if (cell == 7).any(): return GOAL
    if c == 9 or c == 10: return E
    if c == 14: return B
    if c == 8: return G
    if c == 12:
        return T if int(obs[top+1, left+2]) == 12 else P
    if c == 15:
        if (cell == 0).any() or (cell == 11).any(): return S
        if int(obs[top+1, left+1]) == 3: return X
        return S
    return W

def _parse(obs):
    ys, xs = np.where(obs[:63] == 9)
    if len(ys) == 0: return None
    ptop = int(ys.min()) - 1
    px = int(xs.mean()) // 6
    g = -1 if ptop >= 31 else 1
    cells = {}
    for j in range(-10, 11):
        top = ptop + 6*j
        if top < 0 or top + 5 > 62: continue
        for cx in range(11):
            cells[(cx, j)] = _cls(obs, top, cx*6)
    cells[(px, 0)] = E
    return ptop, px, g, cells

def _plan(M, vis, px, pr, g, t0):
    def get(x, y, ch):
        if x < 0 or x > 10: return W
        k = (x, y)
        if k in ch: return ch[k]
        return M.get(k, U)
    def fall(x, y, gg, ch):
        n = 0
        while n < 200:
            n += 1
            t = get(x, y+gg, ch)
            if t in (E, P): y += gg; continue
            if t == GOAL: return 'win'
            if t == S: return 'dead'
            if t == U: return 'exit'
            return (x, y)
        return 'exit'
    Gs = [k for k, v in M.items() if v == G]
    rowk = {}
    def unk(y):
        s = 0
        for yy in range(y-6, y+7):
            if yy not in rowk:
                rowk[yy] = 11 - sum(1 for cx in range(11) if (cx, yy) in M)
            s += rowk[yy]
        return s
    start = (px, pr, g, pr, frozenset())
    nodes = [(start, -1, None)]
    seen = {start}
    q = deque([(0, 0)])
    best = (-1e9, None, None)
    def path(i):
        out = []
        while i > 0:
            st, par, a = nodes[i]
            out.append((a, st[:3]))
            i = par
        return out[::-1]
    while q:
        if len(nodes) > 15000 or time.time() - t0 > 6: break
        i, d = q.popleft()
        x, y, gg, cam, chs = nodes[i][0]
        if d >= 28: continue
        ch = dict(chs)
        lo, hi = (-6, 3) if gg == -1 else (-4, 5)
        succ = []
        for dx in (-1, 1):
            t = get(x+dx, y, ch)
            if t == GOAL: succ.append(('win', ('M', dx))); continue
            if t in (E, P):
                rr = fall(x+dx, y, gg, ch)
                if isinstance(rr, tuple):
                    nc = cam if rr[1] == y else rr[1]
                    succ.append(((rr[0], rr[1], gg, nc, chs), ('M', dx)))
                else:
                    succ.append((rr, ('M', dx)))
        cand = {(x-1,y),(x+1,y),(x,y+gg),(x-1,y+gg),(x+1,y+gg),(x,y-gg)}
        cand.update(Gs)
        for k in cand:
            cx, cy = k
            if cx < 0 or cx > 10 or not (lo <= cy - cam <= hi): continue
            t = get(cx, cy, ch)
            below = (cx == x and cy == y + gg)
            nch = dict(ch); res = None
            if t in (B, T, X):
                if t == B: nch[k] = E
                elif t == T: nch[k] = P
                else:
                    nch[k] = E
                    for ddx, ddy in ((-1,0),(1,0),(0,-1),(0,1)):
                        kk = (cx+ddx, cy+ddy)
                        if kk != (x, y) and get(kk[0], kk[1], ch) == E:
                            nch[kk] = X
                if below:
                    res = fall(x, y, gg, nch)
                    if isinstance(res, tuple): res = (res[0], res[1], gg, res[1])
                else:
                    res = (x, y, gg, cam)
            elif t == P:
                if k not in ((x-1, y+gg), (x+1, y+gg)): continue
                nch[k] = T; res = (x, y, gg, cam)
            elif t == G:
                ng = -gg
                orig = get(x, y+ng, ch)
                nch[k] = E
                if orig in SOL: res = (x, y, ng, y)
                elif orig in (S, GOAL): res = 'dead'
                elif orig == U: res = 'exit'
                else:
                    res = fall(x, y, ng, nch)
                    if isinstance(res, tuple): res = (res[0], res[1], ng, res[1])
            else:
                continue
            if isinstance(res, tuple):
                res = res + (frozenset(nch.items()),)
            succ.append((res, ('C', k)))
        for res, a in succ:
            if res == 'win':
                return path(i) + [(a, None)]
            if res == 'dead': continue
            if res == 'exit':
                v = 8 - d
                if v > best[0]: best = (v, i, a)
                continue
            if res in seen: continue
            seen.add(res)
            nodes.append((res, i, a))
            j = len(nodes) - 1
            v = 0.7*unk(res[1]) - (d+1) - 15*vis.get(res[:3], 0) + 0.3*(res[1]-pr)*res[2]
            if v > best[0]: best = (v, j, None)
            q.append((j, d+1))
    if best[1] is None: return None
    if best[2] is not None:
        return path(best[1]) + [(best[2], None)]
    p = path(best[1])
    return p if p else None

def _take(memory, ptop, pr):
    a, ps = memory["plan"].pop(0)
    memory["pred"] = ps
    if a[0] == 'M':
        return ("ACTION4" if a[1] > 0 else "ACTION3", None, None)
    cx, cy = a[1]
    row = ptop + 6*(cy - pr) + 3
    return ("ACTION6", max(0, min(62, row)), min(cx*6+3, 63))

def act(obs, available, memory):
    t0 = time.time()
    obs = np.asarray(obs)
    r = _parse(obs)
    if r is None:
        return ("ACTION4", None, None)
    ptop, px, g, cells = r
    M = memory.setdefault("M", {})
    vis = memory.setdefault("vis", {})
    pred = memory.get("pred")
    if not M:
        pr = 0
    else:
        b0 = pred[1] if pred else memory.get("pr", 0)
        best = None
        for d in range(-16, 17):
            c = b0 + d; s = 0
            for (cx, j), v in cells.items():
                m = M.get((cx, c+j))
                if m is not None:
                    s += 1 if m == v else -1
            key = (s, -abs(d))
            if best is None or key > best[0]: best = (key, c)
        pr = best[1]
    new = 0
    for (cx, j), v in cells.items():
        k = (cx, pr+j)
        if k not in M: new += 1
        M[k] = v
    memory["pr"] = pr
    vk = (px, pr, g)
    vis[vk] = vis.get(vk, 0) + 1
    plan = memory.get("plan")
    if plan and pred == (px, pr, g) and new == 0:
        return _take(memory, ptop, pr)
    p = _plan(M, vis, px, pr, g, t0)
    if p:
        memory["plan"] = p
        return _take(memory, ptop, pr)
    memory["plan"] = None
    memory["pred"] = None
    dr = memory.get("dir", 1)
    memory["dir"] = -dr
    return ("ACTION4" if dr > 0 else "ACTION3", None, None)
