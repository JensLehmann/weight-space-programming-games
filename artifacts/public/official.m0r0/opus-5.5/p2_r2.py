import time, itertools, random
from collections import deque
import numpy as np

TYPES = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
ACTS = {'ACTION1': (0, -1), 'ACTION2': (0, 1), 'ACTION3': (-1, 0), 'ACTION4': (1, 0)}


def parse(obs):
    pm = (obs == 10) | (obs == 1)
    if not pm.any():
        return None
    runs = []
    for r in range(1, 63):
        row = pm[r]; c = 0
        while c < 64:
            if row[c]:
                c2 = c
                while c2 < 64 and row[c2]:
                    c2 += 1
                runs.append((c2 - c, r, c)); c = c2
            else:
                c += 1
    vr = []
    for c in range(64):
        col = pm[:, c]; r = 1
        while r < 63:
            if col[r]:
                r2 = r
                while r2 < 63 and col[r2]:
                    r2 += 1
                vr.append(r2 - r); r = r2
            else:
                r += 1
    s = max(1, min(min(x[0] for x in runs), min(vr) if vr else 64))
    rr = min(x[1] for x in runs)
    cc = min(x[2] for x in runs if x[1] == rr)
    ox, oy = cc % s, rr % s
    corners = [obs[1, 0], obs[1, 63], obs[62, 0], obs[62, 63]]

    def expected(r, c):
        return corners[(0 if r < 32 else 2) + (0 if c < 32 else 1)]

    nonwall = set(); players = []; blocks = []; haz = set(); colored = {}
    selected = False; selblock = False
    cy = 0
    while oy + cy * s + s <= 64:
        cx = 0
        while ox + cx * s + s <= 64:
            r0, c0 = oy + cy * s, ox + cx * s
            r, c = r0 + s // 2, c0 + s // 2
            if r <= 0 or r >= 63:
                cx += 1; continue
            col = int(obs[r, c]); cell = (cx, cy)
            blk = obs[r0:r0 + s, c0:c0 + s]
            ischeck = False
            if s >= 2:
                par = (np.add.outer(np.arange(r0, r0 + s), np.arange(c0, c0 + s)) % 2) == 1
                if (blk[par] == 5).all() and (blk[~par] != 5).all() and len(set(blk[~par].tolist())) == 1:
                    ischeck = True
            if ischeck:
                haz.add(cell); nonwall.add(cell)
            elif s >= 3 and obs[r0, c0] == 5 and col != 5:
                blocks.append(cell); nonwall.add(cell)
                if col == 11:
                    selblock = True
            elif col in (10, 1) and col != expected(r, c):
                players.append(cell); nonwall.add(cell)
                if col == 1:
                    selected = True
            elif col == expected(r, c):
                pass
            elif col == 5:
                nonwall.add(cell)
            else:
                colored.setdefault(col, set()).add(cell); nonwall.add(cell)
            cx += 1
        cy += 1
    gates = {}; plates = {}
    for k, cs in colored.items():
        seen = set()
        for c0 in cs:
            if c0 in seen:
                continue
            comp = [c0]; seen.add(c0); q = [c0]
            while q:
                x, y = q.pop()
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (x + d[0], y + d[1])
                    if n in cs and n not in seen:
                        seen.add(n); comp.append(n); q.append(n)
            (gates if len(comp) >= 2 else plates).setdefault(k, set()).update(comp)
    return dict(s=s, ox=ox, oy=oy, nonwall=nonwall, players=players, blocks=blocks,
                haz=haz, gates=gates, plates=plates, selected=selected or selblock)


def sim(pl, d, env):
    nonwall, blocks, gates, plates, haz = env
    openk = set(k for k, cs in plates.items() if any((not m) and (x, y) in cs for x, y, t, m in pl))
    closed = set()
    for k, cs in gates.items():
        if k not in openk:
            closed |= cs
    new = list(pl); prev = [(p[0], p[1]) for p in pl]
    for i, (x, y, t, m) in enumerate(pl):
        if m:
            continue
        sx, sy = TYPES[t]
        c = (x + sx * d[0], y + sy * d[1])
        if c not in nonwall or c in blocks or c in closed:
            continue
        new[i] = (c[0], c[1], t, m)
    act = [i for i in range(len(pl)) if not pl[i][3]]
    for i in act:
        if (new[i][0], new[i][1]) in haz:
            return None
    pos = {i: (new[i][0], new[i][1]) for i in act}
    for a, i in enumerate(act):
        for j in act[a + 1:]:
            if abs(prev[i][0] - prev[j][0]) == 1 and prev[i][1] == prev[j][1]:
                if pos[i] == prev[j] or pos[j] == prev[i]:
                    mid = ((pos[i][0] + pos[j][0]) // 2, (pos[i][1] + pos[j][1]) // 2)
                    pos[i] = mid; pos[j] = mid
    grp = {}
    for i in act:
        grp.setdefault(pos[i], []).append(i)
    merged = set()
    for p, lst in grp.items():
        if len(lst) >= 2:
            merged.update(lst[:2])
            for i in lst[2:]:
                pos[i] = prev[i]
    out = list(pl)
    for i in act:
        out[i] = (pos[i][0], pos[i][1], pl[i][2], i in merged)
    return tuple(out)


def bfs(start, env, deadline, limit=250000):
    if all(p[3] for p in start):
        return []
    par = {start: None}; q = deque([start])
    while q:
        if len(par) > limit or time.time() > deadline:
            return None
        st = q.popleft()
        for a, d in ACTS.items():
            ns = sim(st, d, env)
            if ns is None or ns in par:
                continue
            par[ns] = (st, a)
            if all(p[3] for p in ns):
                path = []; cur = ns
                while par[cur] is not None:
                    cur, aa = par[cur]; path.append(aa)
                return path[::-1]
            q.append(ns)
    return None


def replay(start, env, path):
    sts = [start]; st = start
    for a in path:
        st = sim(st, ACTS[a], env)
        if st is None:
            return None
        sts.append(st)
    if not all(p[3] for p in st):
        return None
    return sts


def base_obst(state, P, gates, plates):
    openk = set(k for k, cs in plates.items() if any((not m) and (x, y) in cs for x, y, t, m in state))
    obst = set(P['haz'])
    for k, cs in gates.items():
        if k not in openk:
            obst |= cs
    for cs in plates.values():
        obst |= cs
    for x, y, t, m in state:
        if not m:
            obst.add((x, y))
    return obst


def relocate_plan(state, P, gates, plates, deadline):
    nonwall, haz = P['nonwall'], P['haz']
    env0 = (nonwall, set(), gates, plates, haz)
    p = bfs(state, env0, deadline)
    if p is None:
        return None
    sts = replay(state, env0, p)
    if sts is None:
        return None
    S = set((x, y) for st in sts for (x, y, t, m) in st)
    need = [b for b in P['blocks'] if b in S]
    if not need:
        return None
    obst = base_obst(state, P, gates, plates)
    best = None
    for order in itertools.permutations(need):
        if time.time() > deadline:
            break
        cur = set(P['blocks']); acts = []; cost = 0; ok = True
        for b in order:
            others = cur - {b}
            par = {b: None}; q = deque([b]); dest = None
            while q:
                c = q.popleft()
                if c not in S:
                    dest = c; break
                for a, d in ACTS.items():
                    n = (c[0] + d[0], c[1] + d[1])
                    if n in par or n not in nonwall or n in others or n in obst:
                        continue
                    par[n] = (c, a); q.append(n)
            if dest is None:
                ok = False; break
            path = []; c = dest
            while par[c] is not None:
                c, a = par[c]; path.append(a)
            path = path[::-1]
            acts += [('CLICK', b)] + [('B', a) for a in path]
            cost += 1 + len(path)
            cur = others | {dest}
        if not ok:
            continue
        env2 = (nonwall, cur, gates, plates, haz)
        pp = p
        if replay(state, env2, p) is None:
            pp = bfs(state, env2, min(deadline, time.time() + 1.0), 80000)
            if pp is None:
                continue
        total = cost + 1 + len(pp)
        if best is None or total < best[0]:
            best = (total, acts + [('DESEL', None)] + [('M', a) for a in pp])
    return best


def make_plan(state, P, gates, plates, t0):
    env = (P['nonwall'], set(P['blocks']), gates, plates, P['haz'])
    best = None
    p = bfs(state, env, t0 + 3.0)
    if p is not None:
        best = (len(p), [('M', a) for a in p])
    if P['blocks'] and time.time() < t0 + 7.0:
        r = relocate_plan(state, P, gates, plates, t0 + 7.0)
        if r is not None and (best is None or r[0] < best[0]):
            best = r
    return best[1] if best else None


def init_hyps(cells):
    n = min(len(cells), 4)
    perms = list(itertools.permutations(range(4), n))
    perms.sort(key=lambda pr: (0 if set(pr) <= {0, 1} or set(pr) == {0, 1, 2, 3} else 1))
    hyps = []
    for i, pr in enumerate(perms):
        st = tuple(sorted([(cells[k][0], cells[k][1], pr[k], False) for k in range(n)], key=lambda z: z[2]))
        hyps.append((i, st))
    return hyps


def act(obs, available, memory):
    t0 = time.time()
    obs = np.asarray(obs)
    P = parse(obs)
    if P is None:
        return ('ACTION1', None, None)
    s, ox, oy = P['s'], P['ox'], P['oy']

    def center(c):
        return (oy + c[1] * s + s // 2, ox + c[0] * s + s // 2)

    gates = memory.setdefault('gates', {}); plates = memory.setdefault('plates', {})
    for k, cs in P['gates'].items():
        gates.setdefault(k, set()).update(cs)
    for k, cs in P['plates'].items():
        plates.setdefault(k, set()).update(cs)
    for k in list(plates.keys()):
        if k in gates:
            plates[k] -= gates[k]
    obsset = set(P['players'])
    env = (P['nonwall'], set(P['blocks']), gates, plates, P['haz'])
    last = memory.get('last')
    if 'hyps' not in memory:
        memory['hyps'] = init_hyps(P['players']); memory['plan'] = []
    elif last is not None and last[0] == 'M':
        nh = []
        for pid, st in memory['hyps']:
            ns = sim(st, ACTS[last[1]], env)
            if ns is not None and set((p[0], p[1]) for p in ns) == obsset:
                nh.append((pid, ns))
        if not nh:
            nh = init_hyps(P['players']); memory['plan'] = []
        memory['hyps'] = nh
        if memory.get('pid') not in [h[0] for h in nh]:
            memory['plan'] = []
    elif last is not None and last[0] == 'B':
        if memory.get('expb') is not None and memory['expb'] not in set(P['blocks']):
            memory['plan'] = []
    plan = memory.get('plan', [])
    if not plan:
        if P['selected']:
            plan = [('DESEL', None)]
        else:
            pid, st = memory['hyps'][0]
            pl = make_plan(st, P, gates, plates, t0)
            if pl is None:
                pl = [('M', random.choice(list(ACTS.keys())))]
            plan = pl; memory['pid'] = pid
    step = plan.pop(0); memory['plan'] = plan
    memory['last'] = step
    if step[0] == 'M':
        return (step[1], None, None)
    if step[0] == 'B':
        cur = memory.get('selb')
        if cur is not None:
            d = ACTS[step[1]]
            memory['expb'] = (cur[0] + d[0], cur[1] + d[1]); memory['selb'] = memory['expb']
        return (step[1], None, None)
    if step[0] == 'CLICK':
        memory['selb'] = step[1]; memory['expb'] = None
        r, c = center(step[1])
        return ('ACTION6', int(r), int(c))
    memory['selb'] = None; memory['expb'] = None
    tgt = P['players'][0]
    r, c = center(tgt)
    return ('ACTION6', int(r), int(c))
