import numpy as np
from collections import deque

DIRS = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}

def _scan(o, f):
    H, W = o.shape
    return [(r, c) for r in range(3, H - 3) for c in range(3, W - 3) if f(o, r, c)]

def _player(o, r, c):
    return o[r, c] == 5 and (o[r-2:r+3, c-2:c+3] == 9).sum() == 24

def _goal(o, r, c):
    return o[r, c] == 9 and (o[r-2:r+3, c-2:c+3] == 5).sum() == 24

def _plate(o, r, c):
    v = o[r, c]
    if v not in (8, 11, 15):
        return False
    b = o[r-2:r+3, c-2:c+3]
    if not (b[1:4, 1:4] == v).all():
        return False
    ring = np.concatenate([b[0], b[4], b[1:4, 0], b[1:4, 4]])
    return (ring == 5).sum() >= 12

def _is_gate(o, r, c):
    b = o[max(0, r-2):r+3, max(0, c-2):c+3]
    return ((b == 8).sum() >= 15) or ((b == 11).sum() >= 15)

def _bfs(o, p, targets, allow_gate, blk):
    H, W = o.shape
    prev = {p: None}
    q = deque([p])
    found = None
    while q:
        cur = q.popleft()
        if cur in targets and cur != p:
            found = cur
            break
        for name, (dr, dc) in DIRS.items():
            if cur == p and name in blk:
                continue
            nx = (cur[0] + 6 * dr, cur[1] + 6 * dc)
            if nx in prev or not (0 <= nx[0] < H - 2 and 0 <= nx[1] < W):
                continue
            if o[nx] == 0:
                continue
            if not allow_gate and _is_gate(o, nx[0], nx[1]):
                continue
            prev[nx] = (cur, name)
            q.append(nx)
    if found is None:
        return None
    cur = found
    while prev[cur][0] != p:
        cur = prev[cur][0]
    return prev[cur][1]

def act(obs, available, memory):
    o = np.array(obs)
    t = memory.get("t", 0)
    memory["t"] = t + 1
    ps = _scan(o, _player)
    p = ps[0] if ps else None
    bl = memory.setdefault("bl", {})
    last = memory.get("last")
    if last in DIRS and memory.get("prev") is not None and np.array_equal(memory["prev"], o):
        bl[(memory.get("ppos"), last)] = t
    memory["prev"] = o.copy()
    memory["ppos"] = p
    if p is None:
        memory["last"] = None
        return ("ACTION1", None, None)
    blk = set(n for (pp, n), s in bl.items() if pp == p and t - s <= 1)
    phase = memory.get("phase", 0)
    goals = set(_scan(o, _goal))
    a = None
    if goals:
        a = _bfs(o, p, goals, False, blk)
        if a is None and phase == 1:
            a = _bfs(o, p, goals, True, blk)
    if a is None and phase == 0:
        plates = set(_scan(o, _plate))
        if p in plates and "ACTION5" in available:
            memory["phase"] = 1
            memory["last"] = "ACTION5"
            bl.clear()
            return ("ACTION5", None, None)
        if plates:
            a = _bfs(o, p, plates, False, blk)
    if a is None:
        names = [n for n in DIRS if n not in blk] or list(DIRS)
        a = names[t % len(names)]
    memory["last"] = a
    return (a, None, None)
