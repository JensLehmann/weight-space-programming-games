import numpy as np
from collections import deque

def _comps(m):
    H, W = m.shape
    seen = np.zeros_like(m, dtype=bool)
    out = []
    for y0, x0 in zip(*np.nonzero(m)):
        if seen[y0, x0]:
            continue
        q = [(y0, x0)]; seen[y0, x0] = True; c = []
        while q:
            y, x = q.pop(); c.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                a, b = y+dy, x+dx
                if 0 <= a < H and 0 <= b < W and m[a, b] and not seen[a, b]:
                    seen[a, b] = True; q.append((a, b))
        ys = [p[0] for p in c]; xs = [p[1] for p in c]
        out.append((min(ys), min(xs), max(ys)-min(ys)+1, max(xs)-min(xs)+1))
    return out

def act(obs, available, memory):
    g = np.array(obs)[:, :62]
    q = memory.setdefault("queue", [])
    if q:
        r, c = q.pop(0)
        return ("ACTION6", r, c)
    m = (g == 9) | (g == 10)
    m[46:, :] = False
    if "exit" not in memory:
        cs = _comps(m)
        ex = [c for c in cs if (c[2], c[3]) in ((5, 6), (6, 5), (6, 6))]
        if not ex and cs:
            ex = [max(cs, key=lambda c: c[2]*c[3])]
        if ex:
            memory["exit"] = ex[0]
    if "exit" not in memory:
        return ("ACTION4", None, None)
    ey, exx, eh, ew = memory["exit"]
    pm = m.copy()
    pm[ey:ey+eh, exx:exx+ew] = False
    cs = _comps(pm)
    if not cs:
        return ("ACTION4", None, None)
    py, px, ph, pw = min(cs, key=lambda c: abs(c[2]-4)+abs(c[3]-4))
    s = 4 if max(ph, pw) >= 3 else 2
    walk = (g == 2) | m
    def free(y, x):
        return 0 <= y and 0 <= x and y+s <= 64 and x+s <= 62 and walk[y:y+s, x:x+s].all()
    def hits(y, x):
        return y < ey+eh and y+s > ey and x < exx+ew and x+s > exx
    moves = [(-1,0,"ACTION1"),(1,0,"ACTION2"),(0,-1,"ACTION3"),(0,1,"ACTION4")]
    start = (py, px)
    first = {start: None}
    dq = deque([start]); found = None
    while dq and found is None:
        y, x = dq.popleft()
        for dy, dx, n in moves:
            steps = [s] if s == 2 else [4, 2]
            for st in steps:
                a, b = y+dy*st, x+dx*st
                if hits(a, b):
                    found = first[(y, x)] or n
                    break
                if free(a, b):
                    if (a, b) not in first:
                        first[(a, b)] = first[(y, x)] or n
                        dq.append((a, b))
                    break
            if found:
                break
    if found:
        return (found, None, None)
    if not memory.get("cast"):
        memory["cast"] = True
        q.append((40, 58))
        for r, c in [(0,1),(1,0),(1,2),(2,1)]:
            q.append((50+5*r, 25+5*c))
        r, c = q.pop(0)
        return ("ACTION6", r, c)
    k = memory.get("k", 0); memory["k"] = k + 1
    cy, cx = ey+eh//2, exx+ew//2
    if k % 2 == 0 and cx != px:
        return ("ACTION4" if cx > px else "ACTION3", None, None)
    return ("ACTION2" if cy > py else "ACTION1", None, None)
