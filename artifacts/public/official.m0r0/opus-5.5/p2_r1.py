import numpy as np, heapq, time, random, itertools
from collections import deque

DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

def parse(obs):
    o = np.array(obs, dtype=int)
    c1, c2, c3 = int(o[1, 0]), int(o[1, 63]), int(o[62, 0])
    pat = np.empty((64, 64), int)
    quad = (c3 == c2 and c1 != c2)
    if quad:
        pat[:32, :32] = c1; pat[:32, 32:] = c2; pat[32:, :32] = c2; pat[32:, 32:] = c1
    else:
        pat[:, :32] = c1; pat[:, 32:] = c2
    W = o == pat
    m = np.where(W, -2, o)
    rok = np.ones(64, bool); rok[0] = rok[63] = False
    cand = []
    for N in range(5, 33):
        s = 64 // N
        if s < 2: continue
        off = (64 - N * s) // 2; end = off + N * s
        out = np.ones((64, 64), bool); out[off:end, off:end] = False
        out[0, :] = False; out[63, :] = False
        if not W[out].all(): continue
        u = 0; t = 0
        for cy in range(N):
            y0 = off + cy * s
            rr = rok[y0:y0 + s]
            if not rr.any(): continue
            for cx in range(N):
                x0 = off + cx * s
                sub = m[y0:y0 + s, x0:x0 + s][rr]
                t += 1
                if (sub == sub.flat[0]).all(): u += 1
        if t: cand.append((N, u / t))
    if not cand: return None
    fmax = max(f for _, f in cand)
    N = min(n for n, f in cand if f >= fmax - 0.05)
    s = 64 // N; off = (64 - N * s) // 2
    cells = {}
    for cy in range(N):
        for cx in range(N):
            y0 = off + cy * s; x0 = off + cx * s
            sub = o[y0:y0 + s, x0:x0 + s]
            wsub = W[y0:y0 + s, x0:x0 + s]
            msk = rok[y0:y0 + s]
            k = None
            if s >= 3 and msk.all():
                bd = np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]])
                inn = sub[1:-1, 1:-1]
                if (bd == 5).all() and (inn == inn.flat[0]).all() and inn.flat[0] in (9, 11):
                    k = ('B', int(inn.flat[0]))
            if k is None and msk.all() and s >= 2:
                yy, xx = np.mgrid[y0:y0 + s, x0:x0 + s]
                par = (yy + xx) % 2 == 1
                if (sub[par] == 5).all() and (sub[~par] != 5).all() and not wsub[~par].all():
                    k = ('H', 0)
            if k is None:
                vals = sub[msk].ravel(); wv = wsub[msk].ravel()
                if vals.size == 0: vals = sub.ravel(); wv = wsub.ravel()
                nw = int(wv.sum())
                if nw * 2 > vals.size: k = ('W', 0)
                else:
                    cnt = np.bincount(vals[~wv], minlength=16)
                    mc = int(cnt.argmax())
                    if mc == 5: k = ('F', 0)
                    elif mc in (10, 1): k = ('P', 0)
                    elif mc in (12, 14, 15): k = ('C', mc)
                    else: k = ('W', 0)
            cells[(cx, cy)] = k
    return N, s, off, cells, quad

def comps(cc):
    seen = set(); res = []
    for c in cc:
        if c in seen: continue
        col = cc[c]; st = [c]; seen.add(c); grp = []
        while st:
            p = st.pop(); grp.append(p)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (p[0] + d[0], p[1] + d[1])
                if q in cc and q not in seen and cc[q] == col:
                    seen.add(q); st.append(q)
        res.append((col, grp))
    return res

MULTS2 = [(1, 1), (-1, 1)]
MULTS4 = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

class Env:
    def __init__(s, N, walls, haz, gates, sw, mults):
        s.N = N; s.walls = walls; s.haz = haz; s.gates = gates; s.sw = sw; s.mults = mults
    def inb(s, p): return 0 <= p[0] < s.N and 0 <= p[1] < s.N
    def on(s, P):
        return set(s.sw[p] for p in P if p is not None and p in s.sw)
    def mp(s, P, dx, dy, bl, forbid):
        on = s.on(P); new = list(P)
        for i, p in enumerate(P):
            if p is None: continue
            mx, my = s.mults[i]
            ddx, ddy = dx * mx, dy * my
            if ddx == 0 and ddy == 0: continue
            t = (p[0] + ddx, p[1] + ddy)
            if not s.inb(t) or t in s.walls or t in bl: continue
            if t in s.gates and s.gates[t] not in on: continue
            new[i] = t
        if forbid and any(q is not None and q in s.haz for q in new): return None, []
        act = [i for i in range(len(P)) if P[i] is not None]
        for ii, a in enumerate(act):
            for b in act[ii + 1:]:
                if abs(P[a][0] - P[b][0]) == 1 and P[a][1] == P[b][1]:
                    if new[a] == P[b] or new[b] == P[a]:
                        md = ((new[a][0] + new[b][0]) // 2, (new[a][1] + new[b][1]) // 2)
                        new[a] = md; new[b] = md
        grp = {}
        for i in act: grp.setdefault(new[i], []).append(i)
        merged = []
        for pos, ids in grp.items():
            if len(ids) >= 2:
                merged.append(pos)
                for i in ids[:2]: new[i] = None
                for i in ids[2:]: new[i] = P[i]
        return tuple(new), merged
    def bok(s, t, P, others):
        if not s.inb(t) or t in s.walls or t in others or t in P or t in s.sw or t in s.haz:
            return False
        if t in s.gates and s.gates[t] not in s.on(P): return False
        return True
    def step(s, st, a, forbid=True):
        P, bl, sel = st
        if a[0] == 'C': return (P, bl, a[1])
        if a[0] == 'D': return (P, bl, None)
        dx, dy = DIRS[a[1]]
        if sel is not None:
            t = (sel[0] + dx, sel[1] + dy)
            if not s.bok(t, P, bl): return st
            nb = tuple(sorted([t if b == sel else b for b in bl]))
            return (P, nb, t)
        nP, _ = s.mp(P, dx, dy, bl, forbid)
        if nP is None: return None
        if all(q is None for q in nP): return 'WIN'
        return (nP, bl, sel)

def hfun(P):
    act = [p for p in P if p is not None]
    def ph(a, b): return max((abs(a[0] - b[0]) + 1) // 2, abs(a[1] - b[1]))
    if len(act) == 2: return ph(act[0], act[1])
    if len(act) == 4:
        a, b, c, d = act
        return min(max(ph(a, b), ph(c, d)), max(ph(a, c), ph(b, d)), max(ph(a, d), ph(b, c)))
    return 0

def pbfs(env, P, bl, deadline):
    st0 = (P, bl, None)
    par = {st0: None}; dq = deque([st0])
    while dq:
        if time.time() > deadline: return None
        st = dq.popleft()
        for k in (1, 2, 3, 4):
            ns = env.step(st, ('M', k))
            if ns is None or ns == st: continue
            if ns == 'WIN':
                path = [('M', k)]; cur = st
                while par[cur] is not None:
                    pa, aa = par[cur]; path.append(aa); cur = pa
                return path[::-1]
            if ns not in par:
                par[ns] = (st, ('M', k)); dq.append(ns)
    return None

def bfs_block(env, b, others, P, imp):
    par = {b: None}; dq = deque([b])
    while dq:
        c = dq.popleft()
        if c != b and c not in imp:
            mv = []; cur = c
            while par[cur] is not None:
                pc, k = par[cur]; mv.append(k); cur = pc
            return mv[::-1], c
        for k, (dx, dy) in DIRS.items():
            t = (c[0] + dx, c[1] + dy)
            if t in par or not env.bok(t, P, others): continue
            par[t] = (c, k); dq.append(t)
    return None

def relocate(env, st, deadline):
    P, bl, sel = st
    pp = pbfs(env, P, (), deadline)
    if pp is None: return None
    imp = set(q for q in P if q is not None)
    cur = P
    for a in pp:
        dx, dy = DIRS[a[1]]
        nP, merged = env.mp(cur, dx, dy, (), False)
        for q in nP:
            if q is not None: imp.add(q)
        for q in merged: imp.add(q)
        cur = nP
    need = [b for b in bl if b in imp]
    if not need:
        return ([('D',)] if sel is not None else []) + pp
    for cnt, order in enumerate(itertools.permutations(need)):
        if cnt > 120 or time.time() > deadline: break
        cb = set(bl); acts = []; ok = True
        for b in order:
            r = bfs_block(env, b, cb - {b}, P, imp)
            if r is None: ok = False; break
            mv, fin = r
            acts.append(('C', b)); acts += [('M', k) for k in mv]
            cb.discard(b); cb.add(fin)
        if ok:
            return acts + [('D',)] + pp
    return None

def blocksearch(env, st0, deadline):
    P, bl0, sel0 = st0
    start = (P, bl0, sel0)
    par = {start: None}; dq = deque([start]); tested = {bl0}
    while dq:
        if time.time() > deadline: return None
        st = dq.popleft()
        acts = [('C', b) for b in st[1] if b != st[2]]
        if st[2] is not None: acts += [('M', k) for k in (1, 2, 3, 4)]
        for a in acts:
            ns = env.step(st, a)
            if ns is None or ns == st or ns in par: continue
            par[ns] = (st, a); dq.append(ns)
            if ns[1] not in tested:
                tested.add(ns[1])
                pp = pbfs(env, P, ns[1], deadline)
                if pp is not None:
                    path = []; cur = ns
                    while par[cur] is not None:
                        pa, aa = par[cur]; path.append(aa); cur = pa
                    return path[::-1] + [('D',)] + pp
    return None

def astar(env, st0, deadline):
    cnt = 0
    pq = [(hfun(st0[0]), 0, cnt, st0)]
    par = {st0: None}; g = {st0: 0}
    while pq:
        if time.time() > deadline: return None
        f, gc, _, st = heapq.heappop(pq)
        if gc > g.get(st, 1e9): continue
        acts = [('M', k) for k in (1, 2, 3, 4)] + [('C', b) for b in st[1] if b != st[2]]
        if st[2] is not None: acts.append(('D',))
        for a in acts:
            ns = env.step(st, a)
            if ns is None or ns == st: continue
            if ns == 'WIN':
                path = [a]; cur = st
                while par[cur] is not None:
                    pa, aa = par[cur]; path.append(aa); cur = pa
                return path[::-1]
            ng = gc + 1
            if ng < g.get(ns, 1e9):
                g[ns] = ng; par[ns] = (st, a); cnt += 1
                heapq.heappush(pq, (ng + 1.5 * hfun(ns[0]), ng, cnt, ns))
    return None

def plan(env, st, t0):
    p = pbfs(env, st[0], st[1], t0 + 2.0)
    if p is not None:
        return ([('D',)] if st[2] is not None else []) + p
    if st[1]:
        p = relocate(env, st, t0 + 4.0)
        if p is not None: return p
        p = blocksearch(env, st, t0 + 6.0)
        if p is not None: return p
    return astar(env, st, t0 + 8.3)

def act(obs, available, memory):
    t0 = time.time()
    p = parse(obs)
    if p is None:
        return ('ACTION' + str(random.randint(1, 4)), None, None)
    N, s, off, cells, quad = p
    walls = set(c for c, k in cells.items() if k[0] == 'W')
    haz = set(c for c, k in cells.items() if k[0] == 'H')
    blocks = tuple(sorted(c for c, k in cells.items() if k[0] == 'B'))
    sel = None
    for c, k in cells.items():
        if k[0] == 'B' and k[1] == 11: sel = c
    obsP = set(c for c, k in cells.items() if k[0] == 'P')
    colc = {c: k[1] for c, k in cells.items() if k[0] == 'C'}
    gates = memory.setdefault('gates', {}); sw = memory.setdefault('sw', {})
    for col, grp in comps(colc):
        for c in grp:
            if len(grp) >= 2: gates[c] = col
            elif c not in gates: sw[c] = col
    for c in list(walls):
        if c in gates or c in sw: walls.discard(c)
    dead = memory.setdefault('dead', set())
    if 'k' not in memory:
        memory['k'] = 4 if (len(obsP) >= 4 or quad) else 2
    k = memory['k']
    env = Env(N, walls, haz, gates, sw, MULTS4 if k == 4 else MULTS2)

    def click(c):
        return ('ACTION6', off + c[1] * s + s // 2, off + c[0] * s + s // 2)
    def wallclick():
        w = min(walls) if walls else (0, 0)
        return click(w)

    actP = obsP - dead
    pred = memory.get('pred')
    replan = False
    P = None
    if pred is not None:
        pp = set(q for q in pred[0] if q is not None)
        if pp == actP or (pp <= obsP and len(actP) < len(pp)):
            P = pred[0]
            if pred[1] != blocks or pred[2] != sel: replan = True
        else:
            replan = True
            prev = memory.get('prev'); la = memory.get('last')
            if prev is not None and la is not None and la[0] == 'M':
                for perm in itertools.permutations(range(len(prev[0]))):
                    PP = tuple(prev[0][i] for i in perm)
                    nP, _ = env.mp(PP, DIRS[la[1]][0], DIRS[la[1]][1], prev[1], False)
                    if set(q for q in nP if q is not None) == actP:
                        P = nP; break
            if P is None:
                lst = list(actP); newP = []
                for q in pred[0]:
                    if q is None or not lst: newP.append(None); continue
                    b = min(lst, key=lambda c: abs(c[0] - q[0]) + abs(c[1] - q[1]))
                    lst.remove(b); newP.append(b)
                P = tuple(newP)
    if P is None:
        if len(actP) != k:
            if sel is not None: return wallclick()
            return ('ACTION' + str(random.randint(1, 4)), None, None)
        if k == 2: P = tuple(sorted(actP))
        else: P = tuple(sorted(actP, key=lambda c: (c[1], c[0])))
        replan = True
    st = (P, blocks, sel)
    pl = memory.get('plan')
    if replan or not pl:
        pl = plan(env, st, t0)
        memory['plan'] = pl
    if not pl:
        if sel is not None:
            a = ('D',)
        else:
            opts = [kk for kk in (1, 2, 3, 4) if env.step(st, ('M', kk)) not in (None, st)]
            a = ('M', random.choice(opts) if opts else random.randint(1, 4))
    else:
        a = pl.pop(0); memory['plan'] = pl
    memory['prev'] = st; memory['last'] = a
    if a[0] == 'M' and sel is None:
        nP, merged = env.mp(P, DIRS[a[1]][0], DIRS[a[1]][1], blocks, False)
        for c in merged: dead.add(c)
        memory['pred'] = (nP, blocks, sel)
    else:
        ns = env.step(st, a, forbid=False)
        memory['pred'] = ns if isinstance(ns, tuple) else None
    if a[0] == 'M':
        return ('ACTION' + str(a[1]), None, None)
    if a[0] == 'C':
        return click(a[1])
    return wallclick()
