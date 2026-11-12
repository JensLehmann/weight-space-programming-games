import numpy as np
import heapq, time
from itertools import permutations
from collections import deque

TYPES = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
DIRS = {(0, -1): 'ACTION1', (0, 1): 'ACTION2', (-1, 0): 'ACTION3', (1, 0): 'ACTION4'}
NB = ((1, 0), (-1, 0), (0, 1), (0, -1))

def parse(obs):
    m = (obs == 10) | (obs == 1)
    m[0, :] = False; m[63, :] = False
    pts = np.argwhere(m)
    if len(pts) == 0:
        return None
    r, c = int(pts[0][0]), int(pts[0][1])
    hr = 0
    while c + hr < 64 and m[r, c + hr]: hr += 1
    vr = 0
    while r + vr < 64 and m[r + vr, c]: vr += 1
    s = max(1, min(hr, vr))
    oy, ox = r % s, c % s
    selmode = bool((obs == 1).any())
    wc = {}
    for qi, (ys, xs) in enumerate([(slice(1, 32), slice(0, 32)), (slice(1, 32), slice(32, 64)),
                                   (slice(32, 63), slice(0, 32)), (slice(32, 63), slice(32, 64))]):
        v = obs[ys, xs].ravel()
        v = v[~np.isin(v, [5, 9, 10, 1, 11])]
        wc[qi] = int(np.bincount(v.astype(np.int64), minlength=16).argmax()) if len(v) else 0
    free = set(); players = []; blocks = set(); sel = None; haz = set(); cand = {}
    ncol = (64 - ox) // s; nrow = (64 - oy) // s
    for cy in range(nrow):
        y0 = oy + cy * s
        for cx in range(ncol):
            x0 = ox + cx * s
            yc, xc = y0 + s // 2, x0 + s // 2
            cc = int(obs[yc, xc]); cell = (cx, cy)
            sub = obs[max(y0, 1):min(y0 + s, 63), x0:x0 + s]
            if cc in (10, 1):
                players.append(cell); free.add(cell); continue
            if s >= 3:
                it = obs[y0 + 1:y0 + s - 1, x0 + 1:x0 + s - 1]
                if (it == 5).any() and (it != 5).any():
                    haz.add(cell); continue
            if cc == 5:
                if sub.size and (sub != 5).any() and s >= 2: haz.add(cell)
                else: free.add(cell)
                continue
            if cc == 9 or (cc == 11 and selmode):
                blocks.add(cell); free.add(cell)
                if cc == 11: sel = cell
                continue
            q = (0 if yc < 32 else 2) + (0 if xc < 32 else 1)
            if cc == wc[q]: continue
            if cc in (12, 14, 15):
                cand[cell] = cc; free.add(cell)
    sw = {}; gates = {}; seen = set()
    for cell, col in cand.items():
        if cell in seen: continue
        comp = [cell]; seen.add(cell); st = [cell]
        while st:
            a = st.pop()
            for d in NB:
                b = (a[0] + d[0], a[1] + d[1])
                if b not in seen and cand.get(b) == col:
                    seen.add(b); comp.append(b); st.append(b)
        if len(comp) == 1: sw[cell] = col
        else:
            for x in comp: gates[x] = col
    return dict(s=s, ox=ox, oy=oy, free=free, players=players, blocks=frozenset(blocks),
                sel=sel, haz=haz, sw=sw, gates=gates)

class Env:
    def __init__(self, free, sw, gates, haz, T):
        self.free = set(free); self.sw = dict(sw); self.haz = set(haz); self.T = T
        self.ok = self.free - self.haz
        self.gbc = {}; self.sbc = {}
        for c, col in gates.items(): self.gbc.setdefault(col, set()).add(c)
        for c, col in sw.items(): self.sbc.setdefault(col, set()).add(c)
        self.cache = {}; self.dist = {}
        for c in self.ok:
            d = {c: 0}; q = deque([c])
            while q:
                a = q.popleft()
                for dd in NB:
                    b = (a[0] + dd[0], a[1] + dd[1])
                    if b in self.ok and b not in d:
                        d[b] = d[a] + 1; q.append(b)
            self.dist[c] = d

    def pd(self, a, b):
        d = self.dist.get(a)
        return 99 if d is None else d.get(b, 99)

    def ag(self, P, D):
        if not self.gbc: return frozenset()
        k = (P, D)
        if k in self.cache: return self.cache[k]
        res = set()
        for col, cells in self.gbc.items():
            pressed = col in self.sbc and any((not D[i]) and P[i] in self.sbc[col] for i in range(len(P)))
            if not pressed: res |= cells
        self.cache[k] = res
        return res

    def bmove_ok(self, P, D, B, nc, ag):
        if nc not in self.ok or nc in B or nc in self.sw or nc in ag: return False
        return not any(P[i] == nc and not D[i] for i in range(len(P)))

    def sim(self, st, a):
        P, D, B, sel = st
        if a[0] == 'c':
            return (P, D, B, a[1])
        dx, dy = a
        ag = self.ag(P, D)
        if sel is not None:
            nc = (sel[0] + dx, sel[1] + dy)
            if not self.bmove_ok(P, D, B, nc, ag): return st
            return (P, D, (B - {sel}) | {nc}, nc)
        nP = list(P)
        act = [i for i in range(len(P)) if not D[i]]
        for i in act:
            mx, my = self.T[i]
            nc = (P[i][0] + mx * dx, P[i][1] + my * dy)
            if nc in self.ok and nc not in B and nc not in ag:
                nP[i] = nc
        for ii, i in enumerate(act):
            for j in act[ii + 1:]:
                pi, pj = P[i], P[j]
                if abs(pi[0] - pj[0]) == 1 and pi[1] == pj[1]:
                    if nP[i] == pj or nP[j] == pi:
                        mm = ((nP[i][0] + nP[j][0]) // 2, (nP[i][1] + nP[j][1]) // 2)
                        nP[i] = mm; nP[j] = mm
        nD = list(D); groups = {}
        for i in act: groups.setdefault(nP[i], []).append(i)
        for cell, g in groups.items():
            if len(g) >= 2:
                nD[g[0]] = True; nD[g[1]] = True
                for k in g[2:]: nP[k] = P[k]
        return (tuple(nP), tuple(nD), B, None)

def heur(st, env):
    P, D, B, sel = st
    act = [P[i] for i in range(len(P)) if not D[i]]
    if len(act) < 2: return 0
    def h2(a, b): return (env.pd(a, b) + 1) // 2
    if len(act) == 4:
        a, b, c, d = act
        h = min(max(h2(a, b), h2(c, d)), max(h2(a, c), h2(b, d)), max(h2(a, d), h2(b, c)))
    else:
        h = min(h2(act[i], act[j]) for i in range(len(act)) for j in range(i + 1, len(act)))
    return h + (1 if sel is not None else 0)

def solve(start, env, deadline, blocks_ok=True, wt=1.5):
    h0 = heur(start, env); cnt = 0
    heap = [(wt * h0, 0, 0, start)]
    par = {start: None}; g = {start: 0}
    best = (h0, start); goal = None; it = 0
    while heap:
        it += 1
        if (it & 63) == 0 and time.time() > deadline: break
        f, gg, _, st = heapq.heappop(heap)
        if gg > g.get(st, 1e9): continue
        if all(st[1]):
            goal = st; break
        P, D, B, sel = st
        acts = list(DIRS.keys())
        if blocks_ok:
            for b in B:
                if b != sel: acts.append(('c', b))
            if sel is not None: acts.append(('c', None))
        for a in acts:
            ns = env.sim(st, a)
            if ns == st: continue
            ng = gg + 1
            if ng < g.get(ns, 1e9):
                g[ns] = ng; par[ns] = (st, a)
                h = heur(ns, env)
                if h < best[0]: best = (h, ns)
                cnt += 1
                heapq.heappush(heap, (ng + wt * h, ng, cnt, ns))
    tgt = goal if goal is not None else best[1]
    path = []
    while par.get(tgt):
        p, a = par[tgt]
        path.append((a, tgt)); tgt = p
    path.reverse()
    return path, goal is not None

def connected(env, P, D, B):
    pas = env.ok - B
    act = [P[i] for i in range(len(P)) if not D[i]]
    comp = {}; cid = 0
    for a in act:
        if a in comp: continue
        cid += 1; comp[a] = cid; q = deque([a])
        while q:
            x = q.popleft()
            for d in NB:
                y = (x[0] + d[0], x[1] + d[1])
                if y in pas and y not in comp:
                    comp[y] = cid; q.append(y)
    cnt = {}
    for a in act: cnt[comp[a]] = cnt.get(comp[a], 0) + 1
    return all(cnt[comp[a]] >= 2 for a in act)

def two_phase(state, env, deadline):
    P, D, B0, sel0 = state
    ag = env.ag(P, D)
    start = (B0, sel0)
    par = {start: None}; q = deque([start]); tried = set()
    while q and time.time() < deadline:
        B, sel = q.popleft()
        if B not in tried:
            tried.add(B)
            if connected(env, P, D, B):
                pp, ok = solve((P, D, B, None), env, min(deadline, time.time() + 1.5), False)
                if ok:
                    bp = []; k = (B, sel)
                    while par[k]:
                        pk, a = par[k]
                        bp.append((a, (P, D, k[0], k[1]))); k = pk
                    bp.reverse()
                    if sel is not None:
                        bp.append((('c', None), (P, D, B, None)))
                    return bp + pp
        nxt = []
        for b in B:
            if b != sel: nxt.append((('c', b), (B, b)))
        if sel is not None:
            for d in DIRS:
                nc = (sel[0] + d[0], sel[1] + d[1])
                if env.bmove_ok(P, D, B, nc, ag):
                    nxt.append((d, ((B - {sel}) | {nc}, nc)))
        for a, ns in nxt:
            if ns not in par:
                par[ns] = ((B, sel), a); q.append(ns)
    return None

def act(obs, available, memory):
    t0 = time.time()
    obs = np.asarray(obs).astype(np.int64)
    e = parse(obs)
    if e is None:
        return ('ACTION1', None, None)
    sw = memory.setdefault('sw', {}); gates = memory.setdefault('gate', {}); free = memory.setdefault('free', set())
    for c, v in e['sw'].items():
        if c not in gates: sw[c] = v
    for c, v in e['gates'].items():
        if c not in sw: gates[c] = v
    free |= e['free']
    free -= e['haz']
    obsP = set(e['players'])
    state = None
    if 'prev' in memory and 'T' in memory:
        T = memory['T']
        envp = Env(free, sw, gates, e['haz'], T)
        P, D, B, sel = memory['prev']
        for perm in permutations(range(len(P))):
            Pp = tuple(P[k] for k in perm); Dp = tuple(D[k] for k in perm)
            ns = envp.sim((Pp, Dp, B, sel), memory['last'])
            if set(ns[0]) == obsP and ns[2] == e['blocks'] and ns[3] == e['sel']:
                state = ns; break
    if state is None:
        pl = list(e['players'])
        if len(pl) == 4: pl.sort(key=lambda c: (c[1], c[0]))
        else: pl.sort()
        pl = pl[:4]
        memory['T'] = TYPES[:len(pl)]
        state = (tuple(pl), tuple(False for _ in pl), e['blocks'], e['sel'])
        memory['plan'] = []
    T = memory['T']
    plan = memory.get('plan') or []
    s, ox, oy = e['s'], e['ox'], e['oy']
    if not (plan and memory.get('expect') == state):
        env = Env(free, sw, gates, e['haz'], T)
        plan = two_phase(state, env, t0 + 4.5)
        if not plan:
            plan, ok = solve(state, env, t0 + 7.5, True, 2.0)
    if not plan:
        memory['plan'] = []
        memory.pop('prev', None)
        return ('ACTION1', None, None)
    a, ns = plan[0]
    memory['plan'] = plan[1:]
    memory['expect'] = ns
    memory['prev'] = state
    memory['last'] = a
    if a[0] == 'c':
        cell = a[1]
        if cell is None:
            P, D = state[0], state[1]
            cand = [P[i] for i in range(len(P)) if not D[i]] or list(P)
            cell = cand[0]
        return ('ACTION6', int(oy + cell[1] * s + s // 2), int(ox + cell[0] * s + s // 2))
    return (DIRS[a], None, None)
