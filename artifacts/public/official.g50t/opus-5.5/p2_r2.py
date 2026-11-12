import numpy as np
from collections import deque

DIRS = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}

def _find(o, cc, ring):
    ys, xs = np.where(o == cc)
    for y, x in zip(ys, xs):
        if 2 <= y < 61 and 2 <= x < 62:
            if (o[y-2:y+3, x-2:x+3] == ring).sum() == 24:
                return (int(x), int(y))
    return None

def _cell(o, x, y):
    if not (2 <= x < 62 and 2 <= y < 61):
        return None
    c = o[y, x]
    if c == 0:
        return None
    if c in (5, 9, 2, 15):
        return 'ok'
    b = o[y-2:y+3, x-2:x+3]
    if (b == c).sum() >= 15:
        return 'door'
    if (o[y-1:y+2, x-1:x+2] == c).all():
        return 'plate'
    return 'ok'

def _bfs(o, p, target_fn, bad, doors_ok):
    prev = {p: None}
    q = deque([p])
    while q:
        c = q.popleft()
        if c != p and target_fn(c):
            while prev[c][0] != p:
                c = prev[c][0]
            return prev[c][1]
        for a, (dx, dy) in DIRS.items():
            n = (c[0]+6*dx, c[1]+6*dy)
            if n in prev or (c, a) in bad:
                continue
            t = _cell(o, n[0], n[1])
            if t is None or (t == 'door' and not doors_ok):
                continue
            prev[n] = (c, a)
            q.append(n)
    return None

def act(obs, available, memory):
    o = np.array(obs)
    o[63, :] = 0
    bad = memory.setdefault("bad", set())
    t = memory["t"] = memory.get("t", 0) + 1
    if t % 20 == 0:
        bad.clear()
    p = _find(o, 5, 9)
    lp, la = memory.get("lp"), memory.get("la")
    if p is not None and lp == p and la in DIRS:
        bad.add((p, la))
    memory["lp"] = p
    if p is None:
        memory["la"] = None
        return ("ACTION1", None, None)
    plates = memory.setdefault("plates", set())
    for yy in range(p[1] % 6, 61, 6):
        for xx in range(p[0] % 6, 62, 6):
            if _cell(o, xx, yy) == 'plate':
                plates.add((xx, yy))
    g = _find(o, 9, 5)
    used = memory.setdefault("used", set())
    near = lambda c: g is not None and abs(c[0]-g[0]) <= 1 and abs(c[1]-g[1]) <= 1
    if p in plates and p not in used and memory.get("moved"):
        used.add(p)
        bad.clear()
        memory["moved"] = False
        memory["la"] = "ACTION5"
        return ("ACTION5", None, None)
    a = None
    if g is not None:
        a = _bfs(o, p, near, bad, False)
    if a is None:
        a = _bfs(o, p, lambda c: c in plates and c not in used, bad, False)
    if a is None and g is not None:
        a = _bfs(o, p, near, bad, True)
        if a is not None:
            dx, dy = DIRS[a]
            if _cell(o, p[0]+6*dx, p[1]+6*dy) == 'door':
                a = None
    if a is None:
        order = list(DIRS.keys())
        k = t % 4
        for b in order[k:] + order[:k]:
            ddx, ddy = DIRS[b]
            tt = _cell(o, p[0]+6*ddx, p[1]+6*ddy)
            if tt in ('ok', 'plate') and (p, b) not in bad:
                a = b
                break
        if a is None:
            a = order[k]
    memory["moved"] = True
    memory["la"] = a
    return (a, None, None)
