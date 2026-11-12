import numpy as np
import time

G = 20

def rot(s):
    h, w, b, cn = s
    f = lambda p: (p[1], h - 1 - p[0])
    return (w, h, frozenset(f(p) for p in b), frozenset(f(p) for p in cn))

def rots(s):
    out = []
    t = s
    for _ in range(4):
        if t not in out:
            out.append(t)
        t = rot(t)
    return out

def observe(g, bg, grey):
    H, W = g.shape
    body = set(); conn = set()
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if grey:
                if v not in (bg, 4, 3, 8):
                    body.add((r, c))
            else:
                if v == 0 and bg != 0:
                    body.add((r, c))
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v in (8, 3) and v != bg and (r, c) not in body:
                if any((r + dr, c + dc) in body for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    conn.add((r, c))
    if not body:
        return None
    allc = body | conn
    r0 = min(p[0] for p in allc); c0 = min(p[1] for p in allc)
    h = max(p[0] for p in allc) - r0 + 1; w = max(p[1] for p in allc) - c0 + 1
    shape = (h, w, frozenset((r - r0, c - c0) for r, c in body), frozenset((r - r0, c - c0) for r, c in conn))
    return shape, (c0, r0), frozenset(body)

def norm(cells):
    r0 = min(p[0] for p in cells); c0 = min(p[1] for p in cells)
    return frozenset((r - r0, c - c0) for r, c in cells), r0, c0

def abs_cells(sp):
    h, w, b, cn = sp['cur']; x, y = sp['pos']
    return {(y + r, x + c) for r, c in (b | cn)}

def identify(sps, key, sel, g):
    nk, kr0, kc0 = norm(key)
    best = None
    for j, sp in enumerate(sps):
        for o in sp['opts']:
            if not o[2]:
                continue
            no, or0, oc0 = norm(o[2])
            if no != nk:
                continue
            px, py = kc0 - oc0, kr0 - or0
            tot = len(o[3]); hit = 0
            for r, c in o[3]:
                rr, cc = py + r, px + c
                if 0 <= rr < G and 0 <= cc < G and int(g[rr, cc]) in (8, 3):
                    hit += 1
            sc = (hit == tot, hit - tot, j == sel, (px, py) == sp['pos'])
            if best is None or sc > best[0]:
                best = (sc, (j, o, (px, py)))
    return best[1] if best else None

def rcost(sp, t):
    if sp['cur'] == t:
        return 0
    if sp['kind'] == 'rot':
        s = sp['cur']
        for k in range(1, 4):
            s = rot(s)
            if s == t:
                return k
        return 3
    try:
        return 1 + abs(sp['opts'].index(t) - sp['opts'].index(sp['cur']))
    except ValueError:
        return 3

def solve_all(sps, deadline):
    P = [i for i, sp in enumerate(sps) if any(o[3] for o in sp['opts'])]
    sols = []
    if not P:
        return sols
    seen = set()
    p0 = P[0]

    def span_ok(placed):
        mr = mc = 10 ** 9; Mr = Mc = -10 ** 9
        for j, (o, x, y) in placed.items():
            mr = min(mr, y); mc = min(mc, x); Mr = max(Mr, y + o[0]); Mc = max(Mc, x + o[1])
        return Mr - mr <= G and Mc - mc <= G

    def dfs(placed, cnt):
        if len(sols) >= 300 or time.time() > deadline:
            return
        un = [c for c, v in cnt.items() if v == 1]
        if not un:
            if len(placed) == len(P):
                o0, x0, y0 = placed[p0]
                k = tuple(sorted(((j, o, x - x0, y - y0) for j, (o, x, y) in placed.items()), key=lambda t: t[0]))
                if k not in seen:
                    seen.add(k); sols.append((k, dict(placed)))
            return
        cr, cc = min(un)
        for j in P:
            if j in placed:
                continue
            for o in sps[j]['opts']:
                for (r, c) in o[3]:
                    y = cr - r; x = cc - c
                    cells = [(y + rr, x + c2) for rr, c2 in o[3]]
                    if any(cnt.get(p, 0) >= 2 for p in cells):
                        continue
                    placed[j] = (o, x, y)
                    if span_ok(placed):
                        for p in cells:
                            cnt[p] = cnt.get(p, 0) + 1
                        dfs(placed, cnt)
                        for p in cells:
                            cnt[p] -= 1
                            if cnt[p] == 0:
                                del cnt[p]
                    del placed[j]

    for o in sps[p0]['opts']:
        cnt = {}
        for p in o[3]:
            cnt[p] = cnt.get(p, 0) + 1
        dfs({p0: (o, 0, 0)}, cnt)
    return sols

def rank(sps, sols, tried):
    best = None
    for k, pl in sols:
        if k in tried:
            continue
        items = list(pl.items())
        mnx = min(x for j, (o, x, y) in items); mxx = max(x + o[1] for j, (o, x, y) in items)
        mny = min(y for j, (o, x, y) in items); mxy = max(y + o[0] for j, (o, x, y) in items)
        if mxx - mnx > G or mxy - mny > G:
            continue
        bx = min(range(-mnx, G - mxx + 1), key=lambda t: sum(abs(x + t - sps[j]['pos'][0]) for j, (o, x, y) in items))
        by = min(range(-mny, G - mxy + 1), key=lambda t: sum(abs(y + t - sps[j]['pos'][1]) for j, (o, x, y) in items))
        cost = 0; plan = {}
        for j, (o, x, y) in items:
            X, Y = x + bx, y + by
            sp = sps[j]
            d = rcost(sp, o) + abs(X - sp['pos'][0]) + abs(Y - sp['pos'][1])
            if d:
                d += 1
            cost += d
            plan[j] = (o, X, Y)
        if best is None or cost < best[0]:
            best = (cost, k, plan)
    return best

def click(cell):
    r, c = cell
    return ('ACTION6', 3 * r + 3, 3 * c + 3)

def act(obs, available, memory):
    t0 = time.time()
    obs = np.asarray(obs)
    g = obs[3::3, 3::3][:G, :G]
    vals, cnts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(cnts)])
    grey = memory.get('grey', False) or bool(((g == 4) & (g != bg)).sum() > 0)
    memory['grey'] = grey
    sps = memory.setdefault('sps', [])
    memory.setdefault('phase', 'discover')
    memory.setdefault('ignored', set())
    memory.setdefault('tried', set())
    ob = observe(g, bg, grey)
    pr = memory.get('probe')
    expect = memory.pop('expect', None)
    prev_state = memory.get('lastsel')

    if pr is not None:
        i = pr['i']; sp = sps[i]
        done = False
        if ob:
            s, pos, key = ob
            if sp['kind'] is None:
                if s == rot(sp['cur']):
                    sp['kind'] = 'rot'; sp['opts'] = rots(s); done = True
                else:
                    sp['kind'] = 'stack'
                    if s != sp['cur']:
                        sp['opts'] = [sp['cur'], s]; pr['since'] = 0
                    else:
                        pr['since'] = 1
            else:
                if s in sp['opts']:
                    pr['since'] += 1
                else:
                    sp['opts'].append(s); pr['since'] = 0
            sp['cur'] = s; sp['pos'] = pos; memory['sel'] = i
            if sp['kind'] == 'stack' and (pr['since'] >= len(sp['opts']) or pr.get('n', 0) > 14):
                done = True
            pr['n'] = pr.get('n', 0) + 1
        else:
            if sp['kind'] is None:
                sp['kind'] = 'rot'; sp['opts'] = rots(sp['cur'])
            done = True
        if not done:
            return ('ACTION5', None, None)
        memory['probe'] = None
    else:
        sel = memory.get('sel')
        if ob:
            s, pos, key = ob
            m = identify(sps, key, sel, g)
            if m:
                j, o, p = m
                sps[j]['cur'] = o; sps[j]['pos'] = p; memory['sel'] = j
                if expect is not None:
                    memory['ignored'].add(expect)
            elif memory['phase'] == 'discover' and len(sps) < 10:
                sps.append({'kind': None, 'opts': [s], 'cur': s, 'pos': pos})
                memory['sel'] = len(sps) - 1
            else:
                memory['sel'] = None
                if expect is not None:
                    memory['ignored'].add(expect)
        else:
            memory['sel'] = None
            if expect is not None:
                memory['ignored'].add(expect)

    sel = memory.get('sel')
    if memory['phase'] == 'discover':
        if sel is not None and sps[sel]['kind'] is None:
            memory['probe'] = {'i': sel, 'since': 0, 'n': 0}
            return ('ACTION5', None, None)
        known = set()
        for sp in sps:
            known |= abs_cells(sp)
        cand = [(r, c) for r in range(G) for c in range(G)
                if g[r, c] != bg and (r, c) not in known and (r, c) not in memory['ignored']]
        if cand and len(sps) < 10:
            memory['expect'] = cand[0]
            return click(cand[0])
        memory['phase'] = 'solve'

    state = tuple((sp['cur'], sp['pos']) for sp in sps) + (sel,)
    if state == prev_state:
        memory['stall'] = memory.get('stall', 0) + 1
    else:
        memory['stall'] = 0
    memory['lastsel'] = state
    if 'sols' not in memory:
        memory['sols'] = solve_all(sps, t0 + 6.0)
    sols = memory['sols']
    for attempt in range(6):
        plan = memory.get('plan')
        if plan is None:
            best = rank(sps, sols, memory['tried'])
            if best is None:
                memory['tried'] = set()
                best = rank(sps, sols, memory['tried'])
            if best is None:
                mv = ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION5']
                n = memory.get('rnd', 0)
                memory['rnd'] = n + 1
                return (mv[n % 5], None, None)
            memory['plan'] = best[2]; memory['plankey'] = best[1]
            plan = best[2]
        if memory.get('stall', 0) > 4:
            memory['tried'].add(memory['plankey']); memory['plan'] = None; memory['stall'] = 0
            continue
        target = None
        for j in sorted(plan):
            o, X, Y = plan[j]
            sp = sps[j]
            if sp['cur'] != o or sp['pos'] != (X, Y):
                target = j
                break
        if target is None:
            memory['tried'].add(memory['plankey']); memory['plan'] = None
            continue
        o, X, Y = plan[target]
        sp = sps[target]
        if sel != target:
            mine = abs_cells(sp)
            others = set()
            for k, s2 in enumerate(sps):
                if k != target:
                    others |= abs_cells(s2)
            h, w, b, cn = sp['cur']; x, y = sp['pos']
            bodyabs = [(y + r, x + c) for r, c in sorted(b)]
            pref = [p for p in bodyabs if p not in others] or [p for p in sorted(mine) if p not in others] or sorted(mine)
            return click(pref[0])
        if sp['cur'] != o:
            return ('ACTION5', None, None)
        x, y = sp['pos']
        if X < x:
            return ('ACTION3', None, None)
        if X > x:
            return ('ACTION4', None, None)
        if Y < y:
            return ('ACTION1', None, None)
        return ('ACTION2', None, None)
    return ('ACTION5', None, None)
