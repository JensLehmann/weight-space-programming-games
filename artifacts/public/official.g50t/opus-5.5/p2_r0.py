import numpy as np
import heapq

DIRS = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}

def _player(o):
    for r in range(8, 62):
        for c in range(1, 63):
            if o[r, c] == 5 and (o[r-1:r+2, c-1:c+2] == 9).sum() == 8:
                return (r, c)
    return None

def _goal(o):
    for r in range(8, 62):
        for c in range(1, 63):
            if o[r, c] == 9 and (o[r-1:r+2, c-1:c+2] == 5).sum() == 8:
                return (r, c)
    return None

def _c8(o, r, c):
    return int((o[max(0, r-2):r+3, max(0, c-2):c+3] == 8).sum())

def _door(o, r, c):
    return _c8(o, r, c) >= 16

def _cell(o, r, c):
    if not (9 <= r <= 59 and 2 <= c <= 61):
        return False
    return bool((o[r-2:r+3, c-2:c+3] != 0).all())

def _buttons(o, p):
    res = set()
    for r in range(9, 60):
        for c in range(2, 62):
            if (r - p[0]) % 6 or (c - p[1]) % 6:
                continue
            if (o[r-1:r+2, c-1:c+2] == 8).all() and not _door(o, r, c):
                res.add((r, c))
    return res

def _step_ok(o, a, b):
    (r1, c1), (r2, c2) = a, b
    if not _cell(o, r2, c2):
        return False
    dr = (r2 > r1) - (r2 < r1); dc = (c2 > c1) - (c2 < c1)
    for k in range(1, 7):
        if o[r1 + dr * k, c1 + dc * k] == 0:
            return False
    return True

def _dij(o, start, blocked):
    dist = {start: 0}; prev = {start: None}
    pq = [(0, start)]
    while pq:
        d, cur = heapq.heappop(pq)
        if d > dist[cur]:
            continue
        for a, (dr, dc) in DIRS.items():
            if (cur, a) in blocked:
                continue
            nx = (cur[0] + 6 * dr, cur[1] + 6 * dc)
            if not _step_ok(o, cur, nx):
                continue
            nd = d + 1 + (100 if _door(o, nx[0], nx[1]) else 0)
            if nx not in dist or nd < dist[nx]:
                dist[nx] = nd; prev[nx] = (cur, a)
                heapq.heappush(pq, (nd, nx))
    return dist, prev

def _first(prev, start, t):
    n = t
    while prev[n][0] != start:
        n = prev[n][0]
    return prev[n][1], n

def act(obs, available, memory):
    o = np.array(obs)
    t = memory.get("t", 0) + 1
    memory["t"] = t
    p = _player(o)
    bl = memory.setdefault("bl", {})
    used = memory.setdefault("used", set())
    btn = memory.setdefault("btn", set())
    last = memory.get("last")
    if last in DIRS and p is not None and memory.get("lastp") == p:
        bl[(p, last)] = t + 3
    for k in [k for k, v in bl.items() if v < t]:
        del bl[k]
    blocked = set(bl)

    def ret(a):
        memory["last"] = a; memory["lastp"] = p
        return (a, None, None)

    if p is None:
        return ret("ACTION1")
    btn |= _buttons(o, p)
    g = _goal(o)
    dist, prev = _dij(o, p, blocked)
    if g is not None and g in dist and dist[g] < 100 and g != p:
        memory["wait"] = 0
        return ret(_first(prev, p, g)[0])
    unused = btn - used
    if p in unused:
        used.add(p)
        memory["wait"] = 0
        return ret("ACTION5")
    zero = [b for b in unused if b in dist and dist[b] < 100]
    if zero:
        b = min(zero, key=lambda x: dist[x])
        return ret(_first(prev, p, b)[0])
    cand = [b for b in unused if b in dist]
    target = None
    if cand:
        target = min(cand, key=lambda x: dist[x])
    elif g is not None and g in dist and g != p:
        target = g
    w = memory.get("wait", 0) + 1
    memory["wait"] = w
    if w > 45 and "RESET" in available:
        keep = memory.get("btn", set())
        memory.clear()
        memory["t"] = t; memory["btn"] = keep
        memory["last"] = "RESET"
        return ("RESET", None, None)
    if target is not None:
        a, nx = _first(prev, p, target)
        if not _door(o, nx[0], nx[1]):
            return ret(a)
    # wait: make a legal move not into a door
    for a, (dr, dc) in DIRS.items():
        if (p, a) in blocked:
            continue
        nx = (p[0] + 6 * dr, p[1] + 6 * dc)
        if _step_ok(o, p, nx) and not _door(o, nx[0], nx[1]):
            return ret(a)
    return ret("ACTION5")
