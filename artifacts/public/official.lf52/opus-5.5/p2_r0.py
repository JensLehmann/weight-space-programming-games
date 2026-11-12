import numpy as np
import time, heapq

DIRS = [(0, -1, "ACTION1"), (0, 1, "ACTION2"), (-1, 0, "ACTION3"), (1, 0, "ACTION4")]

def _parse(obs):
    best = None
    H, W = obs.shape
    for oy in range(6):
        for ox in range(6):
            cells = {}; score = 0
            for y in range(oy, H - 5, 6):
                if y < 1:
                    continue
                for x in range(ox, W - 5, 6):
                    b = obs[y:y+6, x:x+6]
                    if (b[:, 0] == 11).all() or (b[:, 5] == 11).all():
                        k = 'Y'; score += 1
                    elif (b[:, 0] == 0).all() and (b[:, 5] == 0).all() and (b[0, 1:5] != 10).all():
                        k = 'E'; score += 1
                    elif (b[2:4, 2:4] == 5).all():
                        cells[(x, y)] = ('T', None); continue
                    else:
                        continue
                    c = b[2:4, 2:4]; o = None
                    if (b == 15).any(): o = 'p'
                    elif (c == 14).any(): o = 'g'
                    elif (c == 8).any(): o = 'r'
                    elif (c == 9).any(): o = 'b'
                    elif (c == 2).any(): o = 'x'
                    cells[(x, y)] = (k, o)
            if best is None or score > best[0]:
                best = (score, ox, oy, cells)
    return best

def _key(Y, occ):
    return (frozenset(Y), frozenset(occ.items()))

def _cnt(occ):
    return sum(1 for v in occ.values() if v in 'gr')

def _goal(occ):
    g = sum(1 for v in occ.values() if v == 'g')
    r = sum(1 for v in occ.values() if v == 'r')
    return g + r == 1 or (g == 1 and r == 1)

def _jumps(base, Y, occ, badland):
    res = []
    for p, k in occ.items():
        if k not in 'grb':
            continue
        for dx, dy, _ in DIRS:
            m = (p[0]+dx, p[1]+dy); t = (p[0]+2*dx, p[1]+2*dy)
            if m in occ and occ[m] != 'x' and t not in occ and (base.get(t) == 'E' or t in Y):
                no = dict(occ); del no[p]; no[t] = k
                if occ[m] == k and k in 'gr':
                    del no[m]
                c = _cnt(no)
                if any(t == bt and c <= bc for bt, bc in badland):
                    continue
                res.append((('J', p, t), Y, no))
    return res

def _plat(base, Y, occ, dx, dy):
    ys = sorted(Y, key=lambda c: (c[0]*dx + c[1]*dy), reverse=True)
    Yc = set(Y); no = dict(occ); moved = False
    for c in ys:
        n = (c[0]+dx, c[1]+dy)
        if n in Yc or base.get(n) != 'T':
            continue
        Yc.discard(c); Yc.add(n); moved = True
        if c in no:
            no[n] = no.pop(c)
    return (Yc, no) if moved else None

def _succ(base, Y, occ, badland):
    res = _jumps(base, Y, occ, badland)
    if Y:
        for dx, dy, a in DIRS:
            r = _plat(base, Y, occ, dx, dy)
            if r:
                res.append((('M', a), r[0], r[1]))
    return res

def _search(base, Y, occ, badland, limit=6.0):
    t0 = time.time()
    start = _key(Y, occ)
    parent = {start: None}
    heap = [(_cnt(occ), 0, 0, start)]
    tie = 1
    while heap and time.time() - t0 < limit:
        _, d, _, s = heapq.heappop(heap)
        Ys = set(s[0]); oc = dict(s[1])
        for op, Y2, o2 in _succ(base, Ys, oc, badland):
            k2 = _key(Y2, o2)
            if k2 in parent:
                continue
            parent[k2] = (s, op)
            if op[0] == 'J' and _goal(o2):
                path = []; cur = k2
                while parent[cur] is not None:
                    ps, pop = parent[cur]
                    path.append((pop, cur)); cur = ps
                return path[::-1]
            heapq.heappush(heap, (_cnt(o2), d + 1, tie, k2)); tie += 1
    return None

def _build(memory, cells, O):
    Ox, Oy = O
    base = dict(memory.get('base', {})); Y = set(memory.get('Y', ())); occ = dict(memory.get('occ', {}))
    for c in list(base):
        px, py = Ox + 6*c[0], Oy + 6*c[1]
        if 0 <= px <= 58 and 1 <= py <= 58:
            base.pop(c); Y.discard(c); occ.pop(c, None)
    for (px, py), (k, o) in cells.items():
        c = ((px - Ox) // 6, (py - Oy) // 6)
        if k == 'Y':
            base[c] = 'T'; Y.add(c)
        else:
            base[c] = k
        if o:
            occ[c] = o
    return base, Y, occ

def _clear(memory):
    for k in ('origin', 'base', 'Y', 'occ', 'plan', 'expect', 'pending', 'lastj'):
        memory.pop(k, None)

def act(obs, available, memory):
    obs = np.asarray(obs)
    if memory.get("pending"):
        r, c = memory.pop("pending")
        return ("ACTION6", int(r), int(c))
    P = _parse(obs)
    badland = memory.setdefault("badland", [])
    if P is None or not P[3]:
        _clear(memory)
        return ("RESET", None, None)
    _, ox, oy, cells = P
    if any(o == 'x' for k, o in cells.values()):
        lj = memory.get("lastj")
        if lj is not None and lj not in badland:
            badland.append(lj)
        _clear(memory)
        return ("RESET", None, None)
    if 'origin' not in memory:
        O = (min(p[0] for p in cells), min(p[1] for p in cells))
        base, Y, occ = _build({}, cells, O)
    else:
        Ox, Oy = memory['origin']
        cands = []
        for dx in range(-66, 67):
            if (Ox + dx - ox) % 6:
                continue
            for dy in range(-36, 37):
                if (Oy + dy - oy) % 6:
                    continue
                cands.append((abs(dx) + abs(dy), dx, dy))
        cands.sort()
        exp = memory.get('expect')
        choice = None
        for _, dx, dy in cands:
            O2 = (Ox + dx, Oy + dy)
            st = _build(memory, cells, O2)
            if choice is None:
                choice = (O2, st)
            if exp is not None and _key(st[1], st[2]) == exp:
                choice = (O2, st)
                break
        O, (base, Y, occ) = choice
    memory['origin'] = O; memory['base'] = base; memory['Y'] = Y; memory['occ'] = occ
    Ox, Oy = O
    k = _key(Y, occ)
    plan = memory.get("plan")
    if not (plan and memory.get("expect") == k):
        plan = None
        if memory.get("fails", 0) < 6:
            plan = _search(base, Y, occ, badland)
            if plan is None:
                memory["fails"] = memory.get("fails", 0) + 1

    def emit(op, Y2, o2, ek):
        memory["expect"] = ek
        if op[0] == 'J':
            p, t = op[1], op[2]
            memory["lastj"] = (t, _cnt(o2) if o2 is not None else 99)
            memory["pending"] = (Oy + 6*t[1] + 3, Ox + 6*t[0] + 3)
            return ("ACTION6", Oy + 6*p[1] + 3, Ox + 6*p[0] + 3)
        memory["lastj"] = None
        return (op[1], None, None)

    if plan:
        op, ek = plan[0]
        memory["plan"] = plan[1:]
        o2 = dict(ek[1])
        return emit(op, None, o2, ek)
    memory.pop("plan", None)
    js = _jumps(base, Y, occ, badland)
    if js:
        cnt = memory.get("g", 0); memory["g"] = cnt + 1
        j = js[cnt % len(js)]
        return emit(j[0], j[1], j[2], _key(j[1], j[2]))
    if Y:
        cnt = memory.get("m", 0); memory["m"] = cnt + 1
        opts = []
        for dx, dy, a in DIRS:
            r = _plat(base, Y, occ, dx, dy)
            if r:
                opts.append((a, r))
        if opts:
            a, r = opts[cnt % len(opts)]
            return emit(('M', a), r[0], r[1], _key(r[0], r[1]))
    _clear(memory)
    return ("RESET", None, None)
