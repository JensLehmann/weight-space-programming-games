import numpy as np
from collections import deque

DIRS = {'ACTION1': (-1, 0), 'ACTION2': (1, 0), 'ACTION3': (0, -1), 'ACTION4': (0, 1)}
PM = np.array([[1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1]], dtype=bool)


def inb(p):
    return 0 <= p[0] < 16 and 0 <= p[1] < 16


def nb(p):
    return [(p[0] + dr, p[1] + dc) for dr, dc in DIRS.values()]


def parse(obs):
    st = dict(player=None, facing='ACTION2', boxes={}, helpers=set(), enemies=set(),
              walls=set(), pmarg=set(), pattern={}, all2=set())
    for r in range(16):
        for c in range(16):
            b = obs[4 * r:4 * r + 4, 4 * c:4 * c + 4]
            if r == 15:
                b = b[:3]
            h = len(b)
            cen = b[1:3, 1:3]
            if (b == 14).any():
                st['player'] = (r, c)
                if (b[0] == 0).all():
                    st['facing'] = 'ACTION1'
                elif h == 4 and (b[3] == 0).all():
                    st['facing'] = 'ACTION2'
                elif (b[:, 0] == 0).all():
                    st['facing'] = 'ACTION3'
                elif (b[:, 3] == 0).all():
                    st['facing'] = 'ACTION4'
                else:
                    st['facing'] = 'ACTION2'
            elif (b == 12).all():
                st['helpers'].add((r, c))
            elif (cen == 15).all():
                st['enemies'].add((r, c))
            elif (cen == 9).all() and b[0, 0] in (0, 3, 4, 5) and (b[0] == b[0, 0]).all():
                st['boxes'][(r, c)] = int(b[0, 0])
            elif (b == 5).all():
                st['walls'].add((r, c))
            elif ((b == 2) == PM[:h]).all():
                st['pmarg'].add((r, c))
            elif (b == 2).all():
                st['all2'].add((r, c))
            elif np.isin(b, (2, 9)).all() and (cen == 2).all() and (b == 9).any():
                top = bool((b[0] == 9).all())
                bot = bool((b[3] == 9).all()) if h == 4 else True
                lef = bool((b[:, 0] == 9).all())
                rig = bool((b[:, 3] == 9).all())
                st['pattern'][(r, c)] = (top, bot, lef, rig)
    return st


def find_targets(st):
    occl = set(st['boxes']) | st['helpers'] | st['enemies']
    if st['player']:
        occl.add(st['player'])
    targets = set(st['pattern'])
    q = deque((p, e) for p, e in st['pattern'].items())
    while q:
        p, e = q.popleft()
        for i, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            if e[i]:
                continue
            n = (p[0] + dr, p[1] + dc)
            if n in targets or not inb(n):
                continue
            if n in st['all2']:
                targets.add(n)
                q.append((n, (False,) * 4))
            elif n in occl:
                targets.add(n)
    return targets


def walk_bfs(srcs, block):
    dist, first, q = {}, {}, deque()
    for s in srcs:
        dist[s] = 0
        first[s] = None
        q.append(s)
    while q:
        u = q.popleft()
        for a, (dr, dc) in DIRS.items():
            n = (u[0] + dr, u[1] + dc)
            if inb(n) and n not in block and n not in dist:
                dist[n] = dist[u] + 1
                first[n] = first[u] or a
                q.append(n)
    return dist, first


def carry_bfs(P, off, obst, pmarg, goals):
    def isgoal(p):
        return (p[0] + off[0], p[1] + off[1]) in goals

    if isgoal(P):
        return []
    prev = {P: None}
    q = deque([P])
    while q:
        u = q.popleft()
        for a, (dr, dc) in DIRS.items():
            n = (u[0] + dr, u[1] + dc)
            bn = (n[0] + off[0], n[1] + off[1])
            if n in prev or not inb(n) or not inb(bn):
                continue
            if n in obst or n in pmarg or bn in obst:
                continue
            prev[n] = (u, a)
            if isgoal(n):
                path = []
                x = n
                while prev[x] is not None:
                    path.append(prev[x][1])
                    x = prev[x][0]
                return path[::-1]
            q.append(n)
    return None


def act(obs, available, memory):
    obs = np.asarray(obs)
    st = parse(obs)
    P = st['player']
    if P is None:
        return ('ACTION5', None, None)
    tg = memory.get('targets', set()) | find_targets(st)
    memory['targets'] = tg
    pm = memory.get('pmarg', set()) | st['pmarg']
    memory['pmarg'] = pm
    boxes = st['boxes']
    boxset = set(boxes)
    helpers = st['helpers']
    walls = st['walls']
    enem = st['enemies']
    base = walls | helpers | enem
    facing = st['facing']
    fd = DIRS[facing]
    front = (P[0] + fd[0], P[1] + fd[1])
    goals = tg - boxset

    hheld = set()
    holder = {}
    busy_helpers = set()
    for b, col in boxes.items():
        adjh = [h for h in nb(b) if h in helpers]
        adje = [e for e in nb(b) if e in enem]
        if col == 5 or (col == 3 and (adjh or adje)):
            hheld.add(b)
            busy_helpers.update(adjh)
            if adjh and not adje:
                holder[b] = adjh[0]

    # helper delivery paths / stuck detection
    stuck = set()
    hroute = set()
    for b, h in holder.items():
        if b in tg:
            continue
        off = (b[0] - h[0], b[1] - h[1])
        path = carry_bfs(h, off, walls | enem | (boxset - {b}), pm, goals)
        if path is None:
            stuck.add(b)
        else:
            p = h
            hroute.update([h, b])
            for a in path:
                d = DIRS[a]
                p = (p[0] + d[0], p[1] + d[1])
                hroute.add(p)
                hroute.add((p[0] + off[0], p[1] + off[1]))

    hdist, _ = walk_bfs(list(helpers), walls | boxset | pm | enem)
    Gc = [None]

    def getG():
        if Gc[0] is None:
            hobst = walls | enem | (boxset & tg) | (hheld - stuck)
            cand = set()
            for c in hdist:
                for n in nb(c):
                    if inb(n) and n not in tg and n not in walls and n not in helpers:
                        cand.add(n)
            res = set()
            for X in cand:
                Hs = [H for H in nb(X) if H in hdist]
                ok = bool(Hs)
                for H in Hs:
                    if carry_bfs(H, (X[0] - H[0], X[1] - H[1]), hobst - {X}, pm, goals) is None:
                        ok = False
                        break
                if ok:
                    res.add(X)
            Gc[0] = res
        return Gc[0]

    # 1. holding a box
    held = None
    for b, col in boxes.items():
        if col == 0 and abs(b[0] - P[0]) + abs(b[1] - P[1]) == 1:
            held = b
            break
    if held is not None:
        if held in tg:
            return ('ACTION5', None, None)
        off = (held[0] - P[0], held[1] - P[1])
        obst = base | (boxset - {held})
        path = carry_bfs(P, off, obst, pm, goals)
        if path:
            return (path[0], None, None)
        if helpers:
            G = getG()
            if held in G:
                return ('ACTION5', None, None)
            path = carry_bfs(P, off, obst, pm, G - boxset)
            if path:
                return (path[0], None, None)
        return ('ACTION5', None, None)

    block = base | boxset | pm
    dist, first = walk_bfs([P], block)

    def approach(T):
        best = None
        for a, (dr, dc) in DIRS.items():
            A = (T[0] - dr, T[1] - dc)
            if A in dist:
                c = dist[A] + (0 if (A == P and facing == a) else 1)
                if best is None or c < best[0]:
                    best = (c, A, a)
        return best

    def do(A, a):
        if A == P:
            return ('ACTION5', None, None) if facing == a else (a, None, None)
        return (first[A], None, None)

    # 2. kill enemies
    bestE = None
    for E in enem:
        r = approach(E)
        if r and (bestE is None or r[0] < bestE[0]):
            bestE = r
    if bestE:
        return do(bestE[1], bestE[2])

    free = [B for B in boxes if B not in tg and (B not in hheld or B in stuck)]

    claimed = set()
    hblock = walls | boxset | pm | enem
    for h in helpers:
        if h in busy_helpers:
            continue
        d, _ = walk_bfs([h], hblock)
        bb = None
        for B in free:
            if B in stuck:
                continue
            for n in nb(B):
                if n in d and (bb is None or d[n] < bb[0]):
                    bb = (d[n], B)
        if bb:
            claimed.add(bb[1])

    def plan(cands, gset):
        best = None
        for B in cands:
            obst = base | (boxset - {B})
            for a, (dr, dc) in DIRS.items():
                A = (B[0] - dr, B[1] - dc)
                if A not in dist:
                    continue
                path = carry_bfs(A, (dr, dc), obst, pm, gset)
                if path is None:
                    continue
                c = dist[A] + (0 if (A == P and facing == a) else 1) + 1 + len(path) + 1
                if best is None or c < best[0]:
                    best = (c, A, a, B)
        return best

    best = None
    com = memory.get('commit')
    if com is not None and com in free:
        best = plan([com], goals)
    if best is None:
        best = plan([B for B in free if B not in claimed], goals)
    if best is None:
        best = plan(free, goals)
    if best:
        memory['commit'] = best[3]
        return do(best[1], best[2])
    memory['commit'] = None
    if helpers:
        G = getG()
        best = plan([B for B in free if B not in G or B in stuck], G - boxset)
        if best:
            return do(best[1], best[2])

    # 4. wait out of the way
    avoid = set(hroute)
    for x in helpers | hheld:
        avoid.add(x)
        avoid.update(nb(x))
    ft = goals
    ftn = set()
    for t in ft:
        ftn.update(nb(t))
    tiers = (avoid | ft | ftn, avoid | ft, avoid)
    w = memory.get('wait')
    target = None
    if w is not None and w in dist and w not in tiers[1]:
        target = w
    else:
        for av in tiers:
            cand = [c for c in dist if c not in av]
            if cand:
                target = min(cand, key=lambda c: dist[c])
                break
    if target is not None:
        memory['wait'] = target
        if target != P:
            return (first[target], None, None)
    if front not in boxset and front not in enem:
        return ('ACTION5', None, None)
    for a, (dr, dc) in DIRS.items():
        n = (P[0] + dr, P[1] + dc)
        if (not inb(n) or n in walls) and n not in boxset:
            return (a, None, None)
    for a, (dr, dc) in DIRS.items():
        n = (P[0] + dr, P[1] + dc)
        if n in dist and n != P:
            return (a, None, None)
    return ('ACTION5', None, None)
