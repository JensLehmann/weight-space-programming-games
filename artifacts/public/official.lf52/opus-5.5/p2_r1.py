import time, heapq

D4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ARR = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}

def _parse(obs):
    H, W = obs.shape
    oy = ox = None
    for r in range(2, H):
        for c in range(1, W):
            v = obs[r, c]
            if (v == 1 and obs[r-1, c] == 0 and obs[r, c-1] == 0) or \
               (v == 12 and obs[r-1, c] == 11 and obs[r, c-1] == 11):
                oy, ox = (r - 1) % 6, (c - 1) % 6
                break
        if oy is not None:
            break
    if oy is None:
        return None
    pieces, normal, orange, track = {}, set(), set(), set()
    for ty in range(-1, (H - oy) // 6 + 1):
        for tx in range(-1, (W - ox) // 6 + 1):
            r0, c0 = oy + ty * 6, ox + tx * 6
            if r0 + 3 >= H or c0 + 3 >= W or r0 + 1 < 1 or c0 < 0:
                continue
            a, b = int(obs[r0+1, c0]), int(obs[r0+2, c0])
            if a == b and a in (0, 11):
                if a == 0:
                    normal.add((tx, ty))
                else:
                    orange.add((tx, ty))
            v = int(obs[r0+3, c0+3])
            if v == 2:
                v = 14
            if v in (14, 8, 9, 15):
                pieces[(tx, ty)] = v
            elif v == 5 and obs[r0+2, c0+2] == 5:
                track.add((tx, ty))
    track |= orange
    return oy, ox, pieces, normal, frozenset(orange), track

def _jumps(p, normal, orange):
    res = []
    for a, col in p.items():
        if col not in (14, 8, 9):
            continue
        for dx, dy in D4:
            b = (a[0]+dx, a[1]+dy)
            c = (a[0]+2*dx, a[1]+2*dy)
            if b in p and c not in p and (c in normal or c in orange):
                cap = (p[b] == col and col != 9)
                res.append((cap, a, b, c))
    res.sort(key=lambda m: not m[0])
    return res

def _apply_jump(p, m):
    cap, a, b, c = m
    q = dict(p)
    col = q.pop(a)
    q[c] = col
    if cap:
        del q[b]
    return q

def _arrow(p, orange, track, d):
    dx, dy = d
    ol = sorted(orange, key=(lambda o: o[0]) if dx else (lambda o: o[1]), reverse=(dx > 0 or dy > 0))
    cur = set(orange)
    q = dict(p)
    moved = False
    for o in ol:
        nb = (o[0]+dx, o[1]+dy)
        if nb in cur:
            continue
        if nb in track:
            cur.discard(o); cur.add(nb)
            if o in q:
                q[nb] = q.pop(o)
            moved = True
    if not moved:
        return None
    return q, frozenset(cur)

def _cnt(p):
    return sum(1 for v in p.values() if v in (14, 8))

def _key(p, o):
    return (frozenset(p.items()), o)

def _norm(p, o):
    pts = list(p.keys()) + list(o)
    if not pts:
        return (frozenset(), frozenset()), (0, 0)
    mx = min(x for x, y in pts); my = min(y for x, y in pts)
    return (frozenset(((x-mx, y-my), v) for (x, y), v in p.items()),
            frozenset((x-mx, y-my) for x, y in o)), (mx, my)

def _succ(p, o, normal, track):
    out = []
    for m in _jumps(p, normal, o):
        out.append((("J", m), _apply_jump(p, m), o))
    for name, d in ARR.items():
        r = _arrow(p, o, track, d)
        if r:
            out.append((("A", name), r[0], r[1]))
    return out

def _search(p, o, normal, track, goal, tl):
    t0 = time.time()
    k0 = _key(p, o)
    par = {k0: None}
    c0 = _cnt(p)
    cnt = 0
    heap = [(2 * (c0 - goal), 0, cnt, p, o)]
    best = (c0, 0, k0)
    found = None
    while heap and time.time() - t0 < tl and len(par) < 400000:
        f, d, _, pp, oo = heapq.heappop(heap)
        k = _key(pp, oo)
        for mv, q2, o2 in _succ(pp, oo, normal, track):
            k2 = _key(q2, o2)
            if k2 in par:
                continue
            par[k2] = (k, mv, q2, o2)
            c2 = _cnt(q2)
            if c2 <= goal:
                found = k2
                break
            if (c2, d + 1) < best[:2]:
                best = (c2, d + 1, k2)
            cnt += 1
            heapq.heappush(heap, (d + 1 + 2 * (c2 - goal), d + 1, cnt, q2, o2))
        if found:
            break
    tgt = found if found else best[2]
    path = []
    k = tgt
    while par[k] is not None:
        pk, mv, q2, o2 = par[k]
        path.append((mv, q2, o2))
        k = pk
    path.reverse()
    return path, found is not None

def _shift_mv(mv, sx, sy):
    if mv[0] == "A":
        return mv
    cap, a, b, c = mv[1]
    f = lambda t: (t[0] + sx, t[1] + sy)
    return ("J", (cap, f(a), f(b), f(c)))

def _emit(mv, oy, ox, memory):
    if mv[0] == "A":
        return (mv[1], None, None)
    cap, a, b, c = mv[1]
    memory['pending'] = (oy + c[1]*6 + 3, ox + c[0]*6 + 3)
    return ("ACTION6", oy + a[1]*6 + 3, ox + a[0]*6 + 3)

def act(obs, available, memory):
    obs = obs.astype(int)
    if memory.get('pending'):
        r, c = memory.pop('pending')
        return ("ACTION6", r, c)
    fh = hash(obs[1:].tobytes())
    fv = memory.setdefault('fv', {})
    fv[fh] = fv.get(fh, 0) + 1
    if fv[fh] >= 8 and memory.get('resets', 0) < 2:
        memory['resets'] = memory.get('resets', 0) + 1
        memory['fv'] = {}; memory['vis'] = {}; memory['plan'] = None
        return ("RESET", None, None)
    P = _parse(obs)
    if P is None:
        return ("ACTION7" if "ACTION7" in available else "RESET", None, None)
    oy, ox, p, normal, orange, track = P
    if 'goal' not in memory:
        cols = set(v for v in p.values() if v in (14, 8))
        memory['goal'] = max(1, len(cols))
    goal = memory['goal']
    nk, (mx, my) = _norm(p, orange)
    vis = memory.setdefault('vis', {})
    vis[nk] = vis.get(nk, 0) + 1

    plan = memory.get('plan')
    if plan and memory.get('expect') is not None and fv[fh] <= 2:
        ek, (emx, emy) = memory['expect']
        if ek == nk and plan:
            mv, q2, o2 = plan.pop(0)
            sx, sy = mx - emx, my - emy
            mv = _shift_mv(mv, sx, sy)
            nk2, mn2 = _norm(q2, o2)
            memory['expect'] = (nk2, mn2)
            memory['plan'] = plan
            memory['expect'] = (nk2, (mn2[0], mn2[1]))
            # store shift-adjusted expectation: predicted min + accumulated shift
            memory['expect'] = (nk2, (mn2[0] + sx, mn2[1] + sy))
            memory['plan'] = [(_shift_mv(m, sx, sy), {(k[0]+sx, k[1]+sy): v for k, v in q.items()},
                               frozenset((x+sx, y+sy) for x, y in o)) for m, q, o in plan]
            return _emit(mv, oy, ox, memory)
    memory['plan'] = None
    memory['expect'] = None

    succ = _succ(p, orange, normal, track)
    if not succ:
        return ("ACTION7" if "ACTION7" in available else "RESET", None, None)

    path = []
    solved = False
    if fv[fh] <= 2:
        path, solved = _search(p, orange, normal, track, goal, 4.0)
    if path and (solved or _cnt(path[-1][1]) < _cnt(p)):
        mv, q2, o2 = path[0]
        memory['plan'] = path[1:]
        memory['expect'] = _norm(q2, o2)
        return _emit(mv, oy, ox, memory)

    def sc(s):
        m, q2, o2 = s
        return (vis.get(_norm(q2, o2)[0], 0),
                0 if (m[0] == "J" and m[1][0]) else 1,
                0 if m[0] == "J" else 1)
    ordered = sorted(succ, key=sc)
    if fv[fh] > 2:
        mv = ordered[(fv[fh] - 3) % len(ordered)][0]
    else:
        mv = ordered[0][0]
    return _emit(mv, oy, ox, memory)
