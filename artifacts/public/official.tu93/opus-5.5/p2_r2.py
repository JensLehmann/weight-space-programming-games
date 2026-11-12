import numpy as np
from collections import deque

DV = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACT = {0: "ACTION1", 1: "ACTION4", 2: "ACTION2", 3: "ACTION3"}
MARK = [(1, 0), (2, 1), (1, 2), (0, 1)]


def parse(obs):
    f = obs[:63]
    ys, xs = (f == 9).nonzero()
    if len(xs) == 0:
        return None
    x0, x1, y0 = int(xs.min()), int(xs.max()), int(ys.min())
    s = max(1, (x1 - x0 + 1) // 3)
    L = 6 * s
    H = 3 * s
    ox, oy = x0 % L, y0 % L

    def col(px, py):
        if 0 <= px < 64 and 0 <= py < 63:
            return int(f[py, px])
        return 5

    edges = set()
    enemies = []
    exitp = None
    hx = 0
    while ox + H * hx < 64:
        hy = 0
        while oy + H * hy < 63:
            px, py = ox + H * hx, oy + H * hy
            c = col(px, py)
            if hx % 2 == 0 and hy % 2 == 0:
                if c == 14:
                    exitp = (hx // 2, hy // 2)
                elif c in (8, 12, 13):
                    r, m = 0, 0
                    for d, (cx, cy) in enumerate(MARK):
                        v = col(px + cx * s, py + cy * s)
                        if v in (15, 11):
                            r, m = d, (1 if v == 11 else 0)
                            break
                    q = (r,) if (c == 13 and m) else ()
                    enemies.append((c, hx // 2, hy // 2, r, m, q))
            elif (hx + hy) % 2 == 1:
                if c == 2:
                    edges.add((hx, hy))
            hy += 1
        hx += 1
    player = ((x0 - ox) // L, (y0 - oy) // L)
    return (player[0], player[1], tuple(sorted(enemies))), edges, exitp


def sim(state, d, edges, exitp):
    px, py, ens = state
    dx, dy = DV[d]
    if (2 * px + dx, 2 * py + dy) not in edges:
        return None, False, False
    nx, ny = px + dx, py + dy
    lst = []
    for (t, x, y, r, m, q) in ens:
        if t == 13 and m:
            q = q + (d,)
        if (x, y) == (nx, ny):
            continue
        lst.append([t, x, y, r, m, q])
    for e in lst:
        t, x, y, r, m, q = e
        ex, ey = DV[r]
        if t == 8:
            if (x + ex, y + ey) == (nx, ny):
                e[4] = 1
                e[1], e[2] = x + ex, y + ey
        elif t == 12 or (t == 13 and m):
            e[1], e[2] = x + ex, y + ey
    for e in lst:
        t, x, y, r, m, q = e
        if (x, y) == (nx, ny) and (t == 12 or m):
            return None, True, False
    for e in lst:
        t, x, y, r, m, q = e
        if t == 12:
            ex, ey = DV[r]
            if (2 * x + ex, 2 * y + ey) not in edges:
                e[3] = (r + 2) % 4
    for e in lst:
        t, x, y, r, m, q = e
        if t == 13:
            ex, ey = DV[r]
            if not m and (x + 2 * ex, y + 2 * ey) == (nx, ny):
                e[4] = 1
                q = (r, r)
            if e[4] and q:
                e[3] = q[0]
                q = q[1:]
            e[5] = q
    ns = (nx, ny, tuple(sorted(tuple(e) for e in lst)))
    return ns, False, (exitp is not None and (nx, ny) == exitp)


def key(st):
    return (st[0], st[1], tuple(sorted(e[:5] for e in st[2])))


def bfs(start, edges, exitp):
    if exitp is None:
        return None
    par = {start: None}
    dq = deque([start])
    n = 0
    while dq:
        cur = dq.popleft()
        n += 1
        if n > 80000:
            break
        for d in range(4):
            ns, dead, won = sim(cur, d, edges, exitp)
            if ns is None or ns in par:
                continue
            par[ns] = (cur, d)
            if won:
                plan = []
                k = ns
                while par[k] is not None:
                    p, dd = par[k]
                    plan.append((dd, k))
                    k = p
                plan.reverse()
                return plan
            dq.append(ns)
    return None


def act(obs, available, memory):
    obs = np.asarray(obs)
    res = parse(obs)
    if res is None:
        return ("ACTION1", None, None)
    st, edges, exitp = res
    if exitp is not None:
        memory["exit"] = exitp
    exitp = memory.get("exit", exitp)
    pred = memory.get("pred")
    matched = pred is not None and key(pred) == key(st)
    if matched:
        st = pred
    plan = memory.get("plan") if matched else None
    if not plan:
        plan = bfs(st, edges, exitp)
    if plan:
        d, ns = plan[0]
        memory["plan"] = plan[1:]
        memory["pred"] = ns
        return (ACT[d], None, None)
    memory["plan"] = None
    for d in range(4):
        ns, dead, won = sim(st, d, edges, exitp)
        if ns is not None:
            memory["pred"] = ns
            return (ACT[d], None, None)
    memory["pred"] = None
    return ("ACTION1", None, None)
