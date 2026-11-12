import numpy as np
from collections import deque

D = {'ACTION1': (0, -1), 'ACTION2': (0, 1), 'ACTION3': (-1, 0), 'ACTION4': (1, 0)}
NAMES = list(D)
DIRS = list(D.values())


def inb(c):
    return 0 <= c[0] < 16 and 0 <= c[1] < 16


def parse(obs):
    boxes = {}; walls = set(); blockers = set(); helpers = set(); opps = set()
    player = None; face = (0, -1); tish = {}
    for cy in range(16):
        for cx in range(16):
            b = obs[4 * cy:4 * cy + 4, 4 * cx:4 * cx + 4]
            r = 3 if cy == 15 else 4
            bb = b[:r]
            if b[1, 1] == 9 and b[1, 2] == 9 and b[2, 1] == 9 and b[2, 2] == 9 and b[0, 0] in (0, 3, 4, 5):
                boxes[(cx, cy)] = int(b[0, 0])
            elif (bb == 14).any():
                player = (cx, cy)
                if b[0, 1] == 0 and b[0, 2] == 0:
                    face = (0, -1)
                elif b[1, 3] == 0 and b[2, 3] == 0:
                    face = (1, 0)
                elif b[1, 0] == 0 and b[2, 0] == 0:
                    face = (-1, 0)
                else:
                    face = (0, 1)
            elif b[1, 1] == 12 and b[2, 2] == 12:
                helpers.add((cx, cy))
            elif b[1, 1] == 15 and b[2, 2] == 15:
                opps.add((cx, cy))
            elif (bb == 5).all():
                walls.add((cx, cy))
            else:
                vals = set(np.unique(bb).tolist())
                if vals <= {2, 9}:
                    tish[(cx, cy)] = b
                elif vals == {1, 2}:
                    blockers.add((cx, cy))
    return boxes, walls, blockers, helpers, opps, player, face, tish


def find_targets(tish):
    seen = set(); res = set()
    for c in tish:
        if c in seen:
            continue
        comp = []; q = [c]; seen.add(c)
        while q:
            u = q.pop(); comp.append(u)
            for d in DIRS:
                v = (u[0] + d[0], u[1] + d[1])
                if v in tish and v not in seen:
                    seen.add(v); q.append(v)
        if not any((tish[u] == 9).any() for u in comp):
            continue
        x0 = min(u[0] for u in comp); x1 = max(u[0] for u in comp)
        y0 = min(u[1] for u in comp); y1 = max(u[1] for u in comp)
        for _ in range(6):
            ch = False
            for u in comp:
                b = tish[u]
                if u[0] == x1 and x1 < 15 and b[1, 3] == 2 and b[2, 3] == 2:
                    x1 += 1; ch = True; break
            for u in comp:
                b = tish[u]
                if u[0] == x0 and x0 > 0 and b[1, 0] == 2 and b[2, 0] == 2:
                    x0 -= 1; ch = True; break
            for u in comp:
                b = tish[u]
                if u[1] == y0 and y0 > 0 and b[0, 1] == 2 and b[0, 2] == 2:
                    y0 -= 1; ch = True; break
            for u in comp:
                b = tish[u]
                if u[1] == y1 and y1 < 15 and b[3, 1] == 2 and b[3, 2] == 2:
                    y1 += 1; ch = True; break
            if not ch:
                break
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                res.add((x, y))
    return res


def walk_bfs(start, face, blocked):
    s0 = (start, face)
    dist = {s0: 0}; first = {s0: None}; q = deque([s0])
    while q:
        s = q.popleft(); pos = s[0]
        for nm, d in D.items():
            n = (pos[0] + d[0], pos[1] + d[1])
            ns = (n, d) if (inb(n) and n not in blocked) else (pos, d)
            if ns not in dist:
                dist[ns] = dist[s] + 1
                first[ns] = first[s] or nm
                q.append(ns)
    return dist, first


def pair_bfs(P, o, hard, blk, goal):
    B = (P[0] + o[0], P[1] + o[1])
    if goal(B):
        return 0, None
    seen = {P}; q = deque([(P, 0, None)])
    while q:
        p, dd, f = q.popleft()
        for nm, (dx, dy) in D.items():
            p2 = (p[0] + dx, p[1] + dy); b2 = (p2[0] + o[0], p2[1] + o[1])
            if p2 in seen or not inb(p2) or not inb(b2) or p2 in hard or p2 in blk or b2 in hard:
                continue
            ff = f or nm
            if goal(b2):
                return dd + 1, ff
            seen.add(p2); q.append((p2, dd + 1, ff))
    return None


def act(obs, available, memory):
    obs = np.asarray(obs)
    boxes, walls, blk_now, helpers, opps, player, face, tish = parse(obs)
    T = memory.setdefault('T', set())
    T |= find_targets(tish)
    BL = memory.setdefault('BL', set())
    BL |= blk_now
    blockers = BL - T
    if player is None:
        return ('ACTION5', None, None)

    c5 = memory.get('c5', {}); nc5 = {}
    for b, c in boxes.items():
        if c == 5:
            nc5[b] = c5.get(b, 0) + 1
    memory['c5'] = nc5

    key = hash(obs.tobytes())
    cnt = memory.setdefault('cnt', {})
    cnt[key] = cnt.get(key, 0) + 1
    if cnt[key] > 4:
        memory.pop('commit', None)
        return (NAMES[cnt[key] % 4], None, None)

    hist = memory.setdefault('hist', [])
    hist.append(player)
    if len(hist) >= 7:
        h = hist[-7:]
        if h[0] != h[1] and all(h[i] == h[i % 2] for i in range(7)):
            memory.pop('commit', None)
            hist.clear()
            for nm, d in D.items():
                n = (player[0] + d[0], player[1] + d[1])
                if n != h[1] and inb(n) and n not in walls and n not in boxes and n not in blockers:
                    return (nm, None, None)

    hard = walls | set(boxes) | helpers | opps | {player}
    walk_block = (hard - {player}) | blockers

    def nbrs(c):
        return [(c[0] + d[0], c[1] + d[1]) for d in DIRS]

    carried = None
    for d in DIRS:
        c = (player[0] + d[0], player[1] + d[1])
        if boxes.get(c) == 0:
            carried = c

    H = set(helpers); q = deque(helpers)
    while q:
        u = q.popleft()
        for d in DIRS:
            v = (u[0] + d[0], u[1] + d[1])
            if inb(v) and v not in H and v not in hard and v not in blockers:
                H.add(v); q.append(v)

    goal_T = lambda B: B in T
    hh = hard - {player} - ({carried} if carried else set())
    hcache = {}

    def goal_H(B):
        if B in T or not helpers:
            return False
        r = hcache.get(B)
        if r is not None:
            return r
        Ps = [n for n in nbrs(B) if n in H]
        ok = bool(Ps)
        for P in Ps:
            o = (B[0] - P[0], B[1] - P[1])
            if pair_bfs(P, o, hh - {P, B}, blockers, goal_T) is None:
                ok = False; break
        hcache[B] = ok
        return ok

    gall = lambda B: B in T or (bool(helpers) and goal_H(B))

    def regrip(P0, o, h2, cap=70):
        seen = {P0: (0, None)}; qq = deque([P0]); best = None; n = 0
        while qq and n < cap:
            p = qq.popleft(); n += 1
            d, f = seen[p]
            B = (p[0] + o[0], p[1] + o[1])
            if not any(x in helpers for x in nbrs(B)):
                for s2 in DIRS:
                    if s2 == o:
                        continue
                    P2 = (B[0] - s2[0], B[1] - s2[1])
                    if not inb(P2) or P2 in h2 or P2 in blockers:
                        continue
                    r = pair_bfs(P2, s2, h2, blockers, gall)
                    if r is None or r[0] == 0:
                        continue
                    c = d + 3 + abs(P2[0] - p[0]) + abs(P2[1] - p[1]) + r[0]
                    if best is None or c < best[0]:
                        best = (c, f)
            for nm, (dx, dy) in D.items():
                p2 = (p[0] + dx, p[1] + dy); b2 = (p2[0] + o[0], p2[1] + o[1])
                if p2 in seen or not inb(p2) or not inb(b2) or p2 in h2 or p2 in blockers or b2 in h2:
                    continue
                seen[p2] = (d + 1, f or nm); qq.append(p2)
        return best

    if carried is not None:
        memory.pop('commit', None)
        o = (carried[0] - player[0], carried[1] - player[1])
        hard2 = hard - {player, carried}
        for g in ([goal_T, goal_H] if helpers else [goal_T]):
            r = pair_bfs(player, o, hard2, blockers, g)
            if r is not None:
                if r[0] == 0:
                    return ('ACTION5', None, None)
                return (r[1], None, None)
        rg = regrip(player, o, hard2)
        if rg is not None and rg[1] is not None:
            return (rg[1], None, None)
        return ('ACTION5', None, None)

    dist, first = walk_bfs(player, face, walk_block)

    def go(st):
        if dist[st] == 0:
            return ('ACTION5', None, None)
        return (first[st], None, None)

    best = None
    for o in opps:
        for s in DIRS:
            st = ((o[0] - s[0], o[1] - s[1]), s)
            if st in dist and (best is None or dist[st] < best[0]):
                best = (dist[st], st)
    if best is not None:
        return go(best[1])

    busy = set(x for x in helpers if any(boxes.get(n) == 5 for n in nbrs(x)))

    def free_box(b):
        c = boxes.get(b)
        if b in T:
            return False
        if c == 5:
            return nc5.get(b, 0) >= 3
        if c not in (3, 4):
            return False
        if any(n in helpers and n not in busy for n in nbrs(b)):
            return False
        return True

    cands = [b for b in boxes if free_box(b)]

    def evalc(b, s, goal):
        P = (b[0] - s[0], b[1] - s[1]); st = (P, s)
        if st not in dist:
            return None
        r = pair_bfs(P, s, hard - {player, b}, blockers, goal)
        if r is None:
            return None
        return dist[st] + 2 + r[0], st

    com = memory.get('commit')
    if com is not None:
        mode, b, s = com
        ok = b in boxes and free_box(b)
        if ok and mode == 'H' and goal_H(b):
            ok = False
        if ok:
            e = evalc(b, s, goal_T if mode == 'T' else goal_H)
            if e is not None:
                return go(e[1])
        memory.pop('commit', None)

    def plan(goal, cl):
        bst = None
        for b in cl:
            for s in DIRS:
                e = evalc(b, s, goal)
                if e is not None and (bst is None or e[0] < bst[0]):
                    bst = (e[0], e[1], b, s)
        return bst

    bst = plan(goal_T, cands); mode = 'T'
    if bst is None and helpers:
        hc = [b for b in cands if boxes.get(b) == 5 or not goal_H(b)]
        bst = plan(goal_H, hc); mode = 'H'
    if bst is not None:
        memory['commit'] = (mode, bst[2], bst[3])
        return go(bst[1])

    # regrip fallback
    opts = []
    for b in cands:
        for s in DIRS:
            st = ((b[0] - s[0], b[1] - s[1]), s)
            if st in dist:
                opts.append((dist[st], b, s, st))
    opts.sort()
    rbest = None
    for dd, b, s, st in opts[:8]:
        rg = regrip(st[0], s, hard - {player, b})
        if rg is not None and (rbest is None or dd + rg[0] < rbest[0]):
            rbest = (dd + rg[0], st)
    if rbest is not None:
        return go(rbest[1])

    def good(c):
        if c in T:
            return False
        if any(abs(c[0] - t[0]) + abs(c[1] - t[1]) < 2 for t in T):
            return False
        if any(n in boxes for n in nbrs(c)):
            return False
        return True

    if not good(player):
        bb = None
        for st, dd in dist.items():
            if st[0] != player and good(st[0]) and (bb is None or dd < bb[0]):
                bb = (dd, st)
        if bb is not None:
            return (first[bb[1]], None, None)

    front = (player[0] + face[0], player[1] + face[1])
    if front not in boxes:
        return ('ACTION5', None, None)
    for nm, d in D.items():
        n = (player[0] + d[0], player[1] + d[1])
        if (not inb(n) or n in walk_block) and n not in boxes:
            return (nm, None, None)
    for nm, d in D.items():
        n = (player[0] + d[0], player[1] + d[1])
        if n not in boxes:
            return (nm, None, None)
    return ('ACTION1', None, None)
