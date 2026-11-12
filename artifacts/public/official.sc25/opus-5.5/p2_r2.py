import numpy as np
from collections import deque

PASS = {2, 9, 10, 11, 15, 14, 7}
MOVES = [("ACTION1", 0, -1), ("ACTION2", 0, 1), ("ACTION3", -1, 0), ("ACTION4", 1, 0)]
PATS = {"size": [[0,1,0],[1,0,1],[0,1,0]],
        "fire": [[0,1,0],[0,1,0],[0,1,0]],
        "tp": [[1,1,0],[0,1,0],[0,0,0]]}

def _comps(m):
    H, W = m.shape
    seen = np.zeros_like(m, dtype=bool); out = []
    for r in range(H):
        for c in range(W):
            if m[r, c] and not seen[r, c]:
                q = [(r, c)]; seen[r, c] = True; pts = []
                while q:
                    y, x = q.pop(); pts.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
                out.append((len(pts), min(ys), min(xs), max(ys), max(xs)))
    return out

def _search(ok, start, s, st, g=None):
    def free(y, x):
        if y < 0 or x < 0 or y + s > 46 or x + s > 62:
            return False
        return bool(ok[y:y+s, x:x+s].all())
    def hit(y, x):
        if g is None: return False
        gy0, gx0, gy1, gx1 = g
        return not (x > gx1 or x + s - 1 < gx0 or y > gy1 or y + s - 1 < gy0)
    prev = {start: None}; dq = deque([start]); order = [start]
    while dq:
        cur = dq.popleft()
        for i, (a, dx, dy) in enumerate(MOVES):
            ny, nx = cur[0] + dy*st, cur[1] + dx*st
            if hit(ny, nx):
                c, j = cur, i
                while prev[c] is not None:
                    c, j = prev[c]
                return j, prev, order
            if (ny, nx) not in prev and free(ny, nx):
                prev[(ny, nx)] = (cur, i); dq.append((ny, nx)); order.append((ny, nx))
    return None, prev, order

def _first(prev, pos):
    j = None
    while prev[pos] is not None:
        pos, j = prev[pos]
    return j

def _clicks(obs, pat):
    out = []
    for r in range(3):
        for c in range(3):
            on = obs[50+5*r, 25+5*c] == 14
            if bool(on) != bool(pat[r][c]):
                out.append(("ACTION6", 50+5*r, 25+5*c))
    return out

def _ray(obs, pos, s, d, tgs):
    y, x = pos
    _, dx, dy = MOVES[d]
    cy = y + (s-1 if dy > 0 else 0); cx = x + (s-1 if dx > 0 else 0)
    for i in range(1, 64):
        ry, rx = cy + dy*i, cx + dx*i
        if not (0 <= ry < 46 and 0 <= rx < 62): return False
        for (ty, tx) in tgs:
            if ty <= ry < ty+4 and tx <= rx < tx+4: return True
        if obs[ry, rx] in (5, 13, 12): return False
    return False

def _spells(obs):
    av = set()
    for r in range(0, 55):
        for c in range(0, 54):
            if obs[r, c] == 3 and obs[r+9, c] == 3 and obs[r, c+9] == 3 and obs[r+9, c+9] == 3:
                inn = obs[r+1:r+9, c+1:c+9]
                if (inn == 3).any() or (inn == 2).sum() < 30: continue
                for col, nm in ((15, "size"), (6, "fire"), (11, "tp")):
                    if (inn == col).any(): av.add(nm)
    return av

def act(obs, available, memory):
    obs = np.asarray(obs)
    t = memory.get("t", 0); memory["t"] = t + 1
    pend = memory.get("pend")
    if pend:
        cl = _clicks(obs, PATS[pend])
        if cl:
            if len(cl) == 1: memory["pend"] = None
            return cl[0]
        memory["pend"] = None
    if "av" not in memory:
        memory["av"] = _spells(obs)
    av = memory["av"]

    def cast(nm):
        cl = _clicks(obs, PATS[nm])
        if not cl: return None
        memory["pend"] = nm if len(cl) > 1 else None
        return cl[0]

    cs = _comps(np.isin(obs[:46, :62], [9, 10]))
    if len(cs) < 2:
        return (MOVES[t % 4][0], None, None)
    cs.sort(key=lambda z: -z[0])
    g = cs[0][1:]; p = cs[1]
    big = max(p[3]-p[1]+1, p[4]-p[2]+1) >= 3
    s, st = (4, 4) if big else (2, 2)
    ok = np.isin(obs, list(PASS))
    start = (p[1], p[2])

    def mv(i):
        memory["dir"] = i
        return (MOVES[i][0], None, None)

    j, prev, order = _search(ok, start, s, st, g)
    if j is not None:
        return mv(j)
    s2 = 2 if big else 4
    if "size" in av and memory.get("nsz", 0) < 6:
        j2, _, _ = _search(ok, start, s2, s2, g)
        if j2 is not None:
            a = cast("size")
            if a:
                memory["nsz"] = memory.get("nsz", 0) + 1
                return a
    if "fire" in av:
        tgs = []
        for r in range(1, 45):
            for c in range(1, 61):
                if obs[r, c] in (13, 12) and obs[r-1, c] == 6 and obs[r, c-1] == 6 and obs[r-1, c-1] == 6:
                    tgs.append((r-1, c-1))
        if tgs:
            best = None
            for pos in order:
                for d in range(4):
                    if _ray(obs, pos, s, d, tgs):
                        best = (pos, d); break
                if best: break
            if best:
                pos, d = best
                if pos == start:
                    if memory.get("dir") == d:
                        a = cast("fire")
                        if a: return a
                    return mv(d)
                jj = _first(prev, pos)
                if jj is not None:
                    return mv(jj)
    opts = []
    if "tp" in av and memory.get("ntp", 0) < 6: opts.append("tp")
    if "size" in av and memory.get("nsz", 0) < 6: opts.append("size")
    if opts:
        last = memory.get("last")
        nm = opts[0]
        if len(opts) > 1 and last == opts[0]:
            nm = opts[1]
        a = cast(nm)
        if a:
            memory["last"] = nm
            k = "ntp" if nm == "tp" else "nsz"
            memory[k] = memory.get(k, 0) + 1
            return a
    dx = (g[1] + g[3]) / 2 - p[2]
    dy = (g[0] + g[2]) / 2 - p[1]
    if abs(dx) > abs(dy):
        i = 3 if dx > 0 else 2
    else:
        i = 1 if dy > 0 else 0
    if t % 5 == 4:
        i = (t // 5) % 4
    return mv(i)
