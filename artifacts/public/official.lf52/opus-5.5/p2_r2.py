import numpy as np, heapq, time

def _parse(obs):
    H, W = obs.shape
    oy = ox = None
    for val in (1, 12):
        for r in range(1, H - 4):
            for c in range(0, W - 4):
                if obs[r, c] == val and (obs[r:r+4, c:c+4] == val).all():
                    oy, ox = (r - 1) % 6, (c - 1) % 6
                    break
            if oy is not None: break
        if oy is not None: break
    if oy is None:
        return None
    cells = {}
    fail = False
    for ty in range(oy, H - 5, 6):
        if ty < 1: continue
        for tx in range(ox, W - 5, 6):
            t = obs[ty:ty+6, tx:tx+6]
            inner = t[1:5, 1:5]; cen = t[2:4, 2:4]
            key = ((tx - ox) // 6, (ty - oy) // 6)
            plat = bool((t == 11).any() or (inner == 12).any())
            piece = None
            if (cen == 14).all(): piece = 'G'
            elif (cen == 8).all(): piece = 'R'
            elif (cen == 9).all(): piece = 'B'
            elif (inner == 15).any(): piece = 'P'
            elif (cen == 2).all() and inner[0, 0] in (1, 12) and inner[1, 0] == 2:
                fail = True
            if plat:
                kind = 'plat'
            elif piece is not None or (inner == 1).any():
                kind = 'board'
            elif (cen == 5).all():
                kind = 'track'
            else:
                continue
            cells[key] = (kind, piece, ty, tx)
    return cells, fail, ox, oy

def _shift(a, b, ddx, ddy):
    best = None; bv = None
    a = a[1:]; b = b[1:]
    H, W = a.shape
    for dy in range(-36, 37):
        if (dy - ddy) % 6: continue
        for dx in range(-60, 61):
            if (dx - ddx) % 6: continue
            if W - abs(dx) < 12 or H - abs(dy) < 12: continue
            bs = b[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)]
            as_ = a[max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx)]
            m = (as_ != 10) | (bs != 10)
            eq = as_ == bs
            s = int((eq & m).sum()) - 2 * int((~eq & m).sum())
            v = (s, -(abs(dx) + abs(dy)))
            if bv is None or v > bv:
                bv = v; best = (dx, dy)
    return best if best else (0, 0)

def _arrow(dx, dy, pieces, plats, track):
    if dx != 0:
        order = sorted(plats, key=lambda p: p[0], reverse=dx > 0)
    else:
        order = sorted(plats, key=lambda p: p[1], reverse=dy > 0)
    npl = set(plats); npc = dict(pieces); ch = False
    for p in order:
        t = (p[0] + dx, p[1] + dy)
        if t in npl or t not in track: continue
        npl.discard(p); npl.add(t); ch = True
        c = None
        if p in npc:
            c = npc.pop(p); npc[t] = c
        if dx < 0 and c == 'G':
            break
    return (npc, npl) if ch else None

def _key(pc, pl):
    return (frozenset(pc.items()), frozenset(pl))

def _solve(world, seen, usegoal, onscreen):
    pieces = {k: v[1] for k, v in world.items() if v[1] is not None}
    plats = set(k for k, v in world.items() if v[0] == 'plat')
    board = set(k for k, v in world.items() if v[0] == 'board')
    track = set(k for k, v in world.items() if v[0] in ('track', 'plat'))
    def cnt(pc):
        g = sum(1 for c in pc.values() if c == 'G'); r = sum(1 for c in pc.values() if c == 'R')
        return g, r
    def ex(pc, pl):
        xs = [p[0] for p, c in pc.items() if p in pl and c in ('G', 'R')]
        return max(xs) if xs else -999
    arrows = [('ACTION1', 0, -1), ('ACTION2', 0, 1), ('ACTION3', -1, 0), ('ACTION4', 1, 0)] if plats else []
    start = _key(pieces, plats)
    g0, r0 = cnt(pieces)
    if usegoal and g0 == 1 and r0 <= 1:
        return None
    parent = {start: None}
    heap = [(g0 + r0, -ex(pieces, plats), 0, 0, start, pieces, plats)]
    n = 1; best = None; bestv = None; t0 = time.time(); pops = 0; goal = None
    while heap:
        sc, e, d, _, k, pc, pl = heapq.heappop(heap)
        pops += 1
        g, r = cnt(pc)
        if k != start:
            if usegoal and g == 1 and r <= 1:
                goal = k; break
            if k not in seen:
                v = (sc, e, d)
                if bestv is None or v < bestv:
                    bestv = v; best = k
        if pops > 25000 or time.time() - t0 > 5:
            break
        succ = []
        for p, c in pc.items():
            if c == 'P': continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                m = (p[0] + dx, p[1] + dy); t = (p[0] + 2*dx, p[1] + 2*dy)
                if m not in pc or t in pc: continue
                if not (t in board or t in pl): continue
                if d == 0 and not (onscreen(p) and onscreen(t)): continue
                nn = dict(pc); del nn[p]; nn[t] = c
                if pc[m] == c and c not in ('B', 'P'):
                    del nn[m]
                succ.append((('J', p, t), nn, pl))
        for name, dx, dy in arrows:
            res = _arrow(dx, dy, pc, pl, track)
            if res: succ.append((('A', name), res[0], res[1]))
        for mv, nn, npl in succ:
            nk = _key(nn, npl)
            if nk in parent: continue
            parent[nk] = (k, mv)
            gg, rr = cnt(nn)
            n += 1
            heapq.heappush(heap, (gg + rr, -ex(nn, npl), d + 1, n, nk, nn, npl))
    k = goal if goal is not None else best
    if k is None:
        return []
    path = []
    while parent[k] is not None:
        k, mv = parent[k]; path.append(mv)
    path.reverse()
    return path

def act(obs, available, memory):
    obs = np.asarray(obs)
    if memory.get('land') is not None:
        r, c = memory.pop('land')
        return ('ACTION6', r, c)
    res = _parse(obs)
    if res is None:
        return ('ACTION4', None, None) if 'ACTION4' in available else (available[0], None, None)
    cells, fail, ox, oy = res
    has_green = any(v[1] == 'G' for v in cells.values())
    if fail and not has_green:
        memory.clear()
        if 'RESET' in available: return ('RESET', None, None)
        return ('ACTION7', None, None)
    # camera tracking
    if 'prev' not in memory:
        memory['cx'] = 0; memory['cy'] = 0; memory['wox'] = ox; memory['woy'] = oy
        memory['world'] = {}
    else:
        dx, dy = _shift(memory['prev'], obs, ox - memory['pox'], oy - memory['poy'])
        memory['cx'] += dx; memory['cy'] += dy
    memory['prev'] = obs.copy(); memory['pox'] = ox; memory['poy'] = oy
    cx, cy, wox, woy = memory['cx'], memory['cy'], memory['wox'], memory['woy']
    world = memory['world']
    for k, v in cells.items():
        kw = ((v[3] - cx - wox) // 6, (v[2] - cy - woy) // 6)
        world[kw] = (v[0], v[1])
    def scr(k):
        return (woy + 6 * k[1] + cy + 3, wox + 6 * k[0] + cx + 3)
    def onscreen(k):
        r, c = scr(k)
        return 3 <= r <= 60 and 2 <= c <= 61
    seen = memory.setdefault('seen', set())
    pieces = {k: v[1] for k, v in world.items() if v[1] is not None}
    plats = set(k for k, v in world.items() if v[0] == 'plat')
    seen.add(_key(pieces, plats))
    usegoal = not memory.get('nogoal', False)
    path = _solve(world, seen, usegoal, onscreen)
    if path is None:
        memory['nogoal'] = True
        path = _solve(world, seen, False, onscreen)
    if path:
        mv = path[0]
        if mv[0] == 'J':
            p, t = mv[1], mv[2]
            pr, pc = scr(p); tr, tc = scr(t)
            memory['land'] = (int(min(63, max(0, tr))), int(min(63, max(0, tc))))
            return ('ACTION6', int(pr), int(pc))
        if mv[1] in available:
            return (mv[1], None, None)
    k = memory.get('fails', 0); memory['fails'] = k + 1
    moves = [a for a in ['ACTION4', 'ACTION2', 'ACTION3', 'ACTION1'] if a in available]
    if not moves:
        return (available[0], None, None)
    return (moves[(k // 3) % len(moves)], None, None)
