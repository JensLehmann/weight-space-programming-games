import numpy as np
from collections import deque

DIRS = {'ACTION1': (0, -1), 'ACTION2': (0, 1), 'ACTION3': (-1, 0), 'ACTION4': (1, 0)}
D2A = {v: k for k, v in DIRS.items()}
ROT2D = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}


def inb(c):
    return 0 <= c[0] < 16 and 0 <= c[1] < 16


def parse(obs):
    walls = set(); bnz = set(); boxes = {}; helpers = set(); advs = set()
    player = None; rot = 0; border9 = set()
    for cy in range(16):
        for cx in range(16):
            raw = obs[cy*4:cy*4+4, cx*4:cx*4+4]
            b = raw.copy()
            if cy == 15:
                b[3, :] = b[0, :]
            c = (cx, cy)
            if (raw[:3] == 14).any():
                player = c
                if (raw[0] == 0).all(): rot = 0
                elif (raw[0:3, 3] == 0).all(): rot = 90
                elif (raw[0:3, 0] == 0).all(): rot = 270
                else: rot = 180
            elif (b[1:3, 1:3] == 9).all() and b[0, 0] != 9:
                boxes[c] = int(b[0, 0])
            elif (b == 12).all():
                helpers.add(c)
            elif (b[1:3, 1:3] == 15).all():
                advs.add(c)
            elif (b == 5).all():
                walls.add(c)
            elif set(np.unique(b).tolist()) == {1, 2}:
                bnz.add(c)
            elif (b == 9).any():
                border9.add(c)
    comps = []; seen = set()
    for c in border9:
        if c in seen: continue
        q = [c]; seen.add(c); xs = []; ys = []
        while q:
            u = q.pop(); xs.append(u[0]); ys.append(u[1])
            for dx, dy in DIRS.values():
                n = (u[0]+dx, u[1]+dy)
                if n in border9 and n not in seen:
                    seen.add(n); q.append(n)
        comps.append([min(xs), max(xs), min(ys), max(ys)])
    merged = True
    while merged:
        merged = False
        for i in range(len(comps)):
            for j in range(i+1, len(comps)):
                a, bb = comps[i], comps[j]
                ov_x = a[0] <= bb[1]+1 and bb[0] <= a[1]+1
                ov_y = a[2] <= bb[3]+1 and bb[2] <= a[3]+1
                strict = a[0] <= bb[1] and bb[0] <= a[1] and a[2] <= bb[3] and bb[2] <= a[3]
                same = (a[2] == bb[2] and a[3] == bb[3]) or (a[0] == bb[0] and a[1] == bb[1])
                if strict or (ov_x and ov_y and same):
                    comps[i] = [min(a[0], bb[0]), max(a[1], bb[1]), min(a[2], bb[2]), max(a[3], bb[3])]
                    comps.pop(j); merged = True; break
            if merged: break
    targets = set()
    for x0, x1, y0, y1 in comps:
        for x in range(x0, x1+1):
            for y in range(y0, y1+1):
                targets.add((x, y))
    return walls, bnz, boxes, helpers, advs, player, rot, targets


def bfs(start, block):
    dist = {start: 0}; first = {start: None}; q = deque([start])
    while q:
        c = q.popleft()
        for a, (dx, dy) in DIRS.items():
            n = (c[0]+dx, c[1]+dy)
            if inb(n) and n not in block and n not in dist:
                dist[n] = dist[c]+1; first[n] = first[c] or a; q.append(n)
    return dist, first


def carry(p, d, blockP, blockB, goal):
    if (p[0]+d[0], p[1]+d[1]) in goal:
        return (0, None)
    seen = {p: None}; q = deque([(p, 0, None)])
    while q:
        c, k, f = q.popleft()
        for a, (dx, dy) in DIRS.items():
            n = (c[0]+dx, c[1]+dy); nb = (n[0]+d[0], n[1]+d[1])
            if n in seen or not inb(n) or n in blockP or not inb(nb) or nb in blockB:
                continue
            ff = f or a
            if nb in goal:
                return (k+1, ff)
            seen[n] = ff; q.append((n, k+1, ff))
    return None


def joint(P, B, h, blockP, blockB, goal, face, helpers, cap=80000):
    start = (P, B, h); seen = {start: None}; q = deque([start])
    while q:
        s = q.popleft(); P, B, h = s; f = seen[s]
        if B in goal:
            if h:
                return 'ACTION5' if f is None else f
            if f is not None:
                return f
        succ = []
        if h:
            adjh = any((B[0]+dx, B[1]+dy) in helpers for dx, dy in DIRS.values())
            if not adjh:
                succ.append(((P, B, 0), 'ACTION5'))
            for a, (dx, dy) in DIRS.items():
                nP = (P[0]+dx, P[1]+dy); nB = (B[0]+dx, B[1]+dy)
                if inb(nP) and inb(nB) and nP not in blockP and nB not in blockB:
                    succ.append(((nP, nB, 1), a))
        else:
            for a, (dx, dy) in DIRS.items():
                nP = (P[0]+dx, P[1]+dy)
                if inb(nP) and nP not in blockP and nP != B:
                    succ.append(((nP, B, 0), a))
            d = (B[0]-P[0], B[1]-P[1])
            if abs(d[0])+abs(d[1]) == 1:
                succ.append(((P, B, 1), 'ACTION5' if face == d else D2A[d]))
        for ns, a in succ:
            if ns not in seen:
                seen[ns] = f if f is not None else a
                q.append(ns)
        if len(seen) > cap:
            break
    return None


def handoff(walls, bnz, boxset, helpers, advs, targets, freeT, exclude):
    if not helpers or not freeT:
        return set()
    bs = boxset - exclude
    blkP = walls | bnz | bs | advs
    blkB = walls | bs | advs
    R = set()
    for h in helpers:
        dd, _ = bfs(h, blkP - {h})
        R |= set(dd)
    allok = set(); anyok = set()
    for x in range(16):
        for y in range(16):
            c = (x, y)
            if c in targets or c in walls or c in bs or c in helpers or c in advs:
                continue
            A = [(x+dx, y+dy) for dx, dy in DIRS.values() if (x+dx, y+dy) in R]
            if not A:
                continue
            good = 0
            for h in A:
                d = (c[0]-h[0], c[1]-h[1])
                if carry(h, d, blkP | {c}, blkB, freeT) is not None:
                    good += 1
            if good == len(A):
                allok.add(c)
            elif good:
                anyok.add(c)
    return allok or anyok


def act(obs, available, memory):
    walls, bnz, boxes, helpers, advs, p, rot, targets = parse(obs)
    W = memory.setdefault('W', set()); W |= walls; walls = set(W)
    Bm = memory.setdefault('B', set()); Bm |= bnz; bnz = set(Bm)
    T = memory.setdefault('T', set())
    T |= targets
    targets = set(T) - walls - bnz
    if p is None:
        return ('ACTION5', None, None)
    agents_all = helpers | advs
    prevc = memory.get('prevc', {})
    aheld = set()
    for b, col in boxes.items():
        if col == 5:
            aheld.add(b)
        elif col == 3:
            adj = any((b[0]+dx, b[1]+dy) in agents_all for dx, dy in DIRS.values())
            if prevc.get(b) == 5 or (b not in prevc and adj):
                aheld.add(b)
    memory['prevc'] = dict(boxes)
    pst = memory.get('still', {})
    still = {b: pst.get(b, 0)+1 for b in aheld}
    memory['still'] = still
    aheld = {b for b in aheld if still[b] < 4}
    memory['rng'] = (memory.get('rng', 12345) * 1103515245 + 12345) % (2**31)
    moves = ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4']
    key = (p, rot, tuple(sorted(boxes)))
    sc = memory.setdefault('seen', {})
    sc[key] = sc.get(key, 0) + 1
    if sc[key] > 2 and sc[key] % 2 == 1:
        return (moves[(memory['rng'] >> 12) % 4], None, None)
    face = ROT2D[rot]
    boxset = set(boxes)
    freeT = targets - boxset
    Hc = {}

    def getH(excl):
        k = tuple(sorted(excl))
        if k not in Hc:
            Hc[k] = handoff(walls, bnz, boxset, helpers, advs, targets, freeT, excl)
        return Hc[k]

    held = [b for b, col in boxes.items() if col == 0 and abs(b[0]-p[0])+abs(b[1]-p[1]) == 1]
    for ignore in (False, True):
        agents = set() if ignore else agents_all
        blockP = walls | bnz | boxset | agents
        blockB = walls | boxset | agents
        if held:
            hb = held[0]
            if hb in targets:
                return ('ACTION5', None, None)
            bp = (blockP - {hb}) | bnz
            bb = blockB - {hb}
            for goal in (freeT, None):
                if goal is None:
                    goal = getH({hb})
                if not goal:
                    continue
                a = joint(p, hb, 1, bp, bb, goal, face, helpers)
                if a:
                    return (a, None, None)
            continue
        dist, first = bfs(p, blockP)

        def go(c, s):
            if c == p:
                if face == s:
                    return ('ACTION5', None, None)
                return (D2A[s], None, None)
            return (first[c], None, None)

        if advs:
            best = None
            for A in advs:
                for s in DIRS.values():
                    c = (A[0]-s[0], A[1]-s[1])
                    if c in dist:
                        cost = dist[c]
                        if best is None or cost < best[0]:
                            best = (cost, c, s)
            if best:
                return go(best[1], best[2])
        avail = [B for B in boxes if B not in targets and B not in aheld]
        for phase in (0, 1):
            cands = []
            for B in avail:
                if phase == 0:
                    goal = freeT
                else:
                    goal = getH({B})
                    if B in goal:
                        continue
                if not goal:
                    continue
                for s in DIRS.values():
                    c = (B[0]-s[0], B[1]-s[1])
                    if c not in dist:
                        continue
                    r = carry(c, s, (blockP - {B}) | bnz, blockB - {B}, goal)
                    if r is None:
                        continue
                    cands.append((dist[c] + 2 + r[0], c, s, B))
            if cands:
                cands.sort(key=lambda t: t[0])
                best = cands[0]
                cm = memory.get('commit')
                if cm is not None:
                    for cd in cands:
                        if (cd[3], cd[2]) == cm and cd[0] <= best[0] + 3:
                            best = cd
                            break
                memory['commit'] = (best[3], best[2])
                return go(best[1], best[2])
        # joint search fallback (reposition boxes via release/regrab)
        avail.sort(key=lambda B: abs(B[0]-p[0])+abs(B[1]-p[1]))
        for B in avail[:6]:
            bp = (blockP - {B}) | bnz
            bb = blockB - {B}
            for goal in (freeT, None):
                if goal is None:
                    goal = getH({B})
                if not goal or B in goal:
                    continue
                a = joint(p, B, 0, bp, bb, goal, face, helpers)
                if a:
                    return (a, None, None)
    if held:
        return ('ACTION5', None, None)
    if any((p[0]+dx, p[1]+dy) in helpers for dx, dy in DIRS.values()):
        blk = walls | bnz | boxset | agents_all
        bestm = None
        for a, (dx, dy) in DIRS.items():
            n = (p[0]+dx, p[1]+dy)
            if inb(n) and n not in blk:
                md = min(abs(n[0]-h[0])+abs(n[1]-h[1]) for h in helpers)
                if bestm is None or md > bestm[0]:
                    bestm = (md, a)
        if bestm and bestm[0] > 1:
            return (bestm[1], None, None)
    front = (p[0]+face[0], p[1]+face[1])
    if front not in boxset and front not in advs:
        return ('ACTION5', None, None)
    return (moves[(memory['rng'] >> 12) % 4], None, None)
