import numpy as np
import time, itertools, random

def _parse(obs):
    N = 21
    for n in range(21, 15, -1):
        off = (64 - 3 * n) // 2
        ok = True
        for r in range(64):
            for c in range(63):
                if (r < off or r >= off + 3 * n or c < off or c >= off + 3 * n) and obs[r, c] != 5:
                    ok = False; break
            if not ok: break
        if ok:
            N = n; break
    off = (64 - 3 * N) // 2
    m = -(-(64 - 3 * N) // 2) - off
    C = np.zeros((N, N), int); K = np.zeros((N, N), int)
    for y in range(N):
        for x in range(N):
            C[y, x] = obs[off + 3 * y + 2, off + 3 * x + 2]
            K[y, x] = obs[off + 3 * y + m, off + 3 * x + m]
    targets = [(x, y) for y in range(N) for x in range(N) if K[y, x] == 11 or C[y, x] == 11]
    vx = [x for x in range(N) if (C[:, x] == 10).sum() >= 6]
    hy = [y for y in range(N) if (C[y, :] == 10).sum() >= 6]
    mirrors = []
    for kind, lst in (('v', vx), ('h', hy)):
        for p in lst:
            cells = [(p, y) for y in range(N)] if kind == 'v' else [(x, p) for x in range(N)]
            cells = [(x, y) for x, y in cells if C[y, x] == 10 and not (kind == 'v' and y in hy) and not (kind == 'h' and x in vx)]
            fixed = any(K[y, x] == 10 for x, y in cells)
            sel = any(K[y, x] == 0 for x, y in cells)
            mirrors.append({'k': kind, 'p': p, 'fixed': fixed, 'sel': sel, 'cells': cells})
    seen = set(); pieces = []
    for y in range(N):
        for x in range(N):
            if C[y, x] == 5 and (x, y) not in seen:
                comp = []; st = [(x, y)]; seen.add((x, y))
                while st:
                    a, b = st.pop(); comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        q = (a + da, b + db)
                        if 0 <= q[0] < N and 0 <= q[1] < N and q not in seen and C[q[1], q[0]] == 5:
                            seen.add(q); st.append(q)
                x0 = min(a for a, b in comp); y0 = min(b for a, b in comp)
                w = max(a for a, b in comp) - x0 + 1; h = max(b for a, b in comp) - y0 + 1
                pieces.append({'rel': [(a - x0, b - y0) for a, b in comp], 'x': x0, 'y': y0, 'w': w, 'h': h,
                               'fixed': any(K[b, a] == 5 for a, b in comp),
                               'sel': any(K[b, a] == 0 for a, b in comp), 'cells': comp})
    return N, off, targets, mirrors, pieces

def _orbit(x, y, ml):
    seen = {(x, y)}; fr = [(x, y)]; d = 0
    while fr and d < 13:
        nw = []
        for (a, b) in fr:
            for k, p in ml:
                q = (2 * p - a, b) if k == 'v' else (a, 2 * p - b)
                if q not in seen:
                    seen.add(q); nw.append(q)
        fr = nw; d += 1
    return seen

def _pc(v): return bin(v).count('1')

def _solve(N, targets, mirrors, pieces, t0):
    tid = {t: i for i, t in enumerate(targets)}
    T = len(targets); full = (1 << T) - 1
    mm = [i for i, mr in enumerate(mirrors) if not mr['fixed']]
    if len(mm) > 2:
        ranges = [range(max(0, mirrors[i]['p'] - 3), min(N, mirrors[i]['p'] + 4)) for i in mm]
    else:
        ranges = [range(N) for i in mm]
    cfgs = []
    for cfg in itertools.product(*ranges):
        c = 0
        for i, p in zip(mm, cfg):
            d = abs(p - mirrors[i]['p'])
            c += d + (1 if d and not mirrors[i]['sel'] else 0)
        cfgs.append((c, cfg))
    cfgs.sort()
    mp = [i for i, pc in enumerate(pieces) if not pc['fixed']]
    fp = [i for i, pc in enumerate(pieces) if pc['fixed']]
    best = [10 ** 9, None]
    for ccost, cfg in cfgs:
        if ccost >= best[0] or time.time() - t0 > 7.5: break
        pos = {i: mr['p'] for i, mr in enumerate(mirrors)}
        for i, p in zip(mm, cfg): pos[i] = p
        ml = [(mirrors[i]['k'], pos[i]) for i in range(len(mirrors))]
        cm = [0] * (N * N)
        for y in range(N):
            for x in range(N):
                v = 0
                for q in _orbit(x, y, ml):
                    j = tid.get(q)
                    if j is not None: v |= 1 << j
                cm[y * N + x] = v
        cov0 = 0
        for i in fp:
            pc = pieces[i]
            for a, b in pc['rel']: cov0 |= cm[(pc['y'] + b) * N + pc['x'] + a]
        opts = []
        for i in mp:
            pc = pieces[i]; d = {}
            for py in range(N - pc['h'] + 1):
                for px in range(N - pc['w'] + 1):
                    v = 0
                    for a, b in pc['rel']: v |= cm[(py + b) * N + px + a]
                    dist = abs(px - pc['x']) + abs(py - pc['y'])
                    c = dist + (1 if dist and not pc['sel'] else 0)
                    if v not in d or d[v][0] > c: d[v] = (c, (px, py))
            opts.append(sorted(((c, v, p) for v, (c, p) in d.items()), key=lambda z: z[0]))
        n = len(opts)
        rmax = [0] * (n + 1); rmin = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            rmax[i] = rmax[i + 1] + max(_pc(v) for c, v, p in opts[i])
            rmin[i] = rmin[i + 1] + min(c for c, v, p in opts[i])
        choice = [None] * n
        def dfs(i, cov, cost):
            if time.time() - t0 > 8.5: return
            if cov == full:
                if cost < best[0]:
                    best[0] = cost
                    best[1] = (dict(pos), {mp[k]: (choice[k] if k < i else (pieces[mp[k]]['x'], pieces[mp[k]]['y'])) for k in range(n)})
                return
            if i == n: return
            if _pc(cov) + rmax[i] < T: return
            for c, v, p in opts[i]:
                if cost + c + rmin[i + 1] >= best[0]: break
                nc = cov | v
                if _pc(nc) + rmax[i + 1] < T: continue
                choice[i] = p
                dfs(i + 1, nc, cost + c)
        dfs(0, cov0, ccost)
    return best[1]

def _plan(obs):
    t0 = time.time()
    N, off, targets, mirrors, pieces = _parse(obs)
    if not targets: return []
    sol = _solve(N, targets, mirrors, pieces, t0)
    if sol is None: return []
    mpos, ppos = sol
    acts = []
    sel = None
    for i, mr in enumerate(mirrors):
        if mr['sel']: sel = ('m', i)
    for i, pc in enumerate(pieces):
        if pc['sel']: sel = ('p', i)
    def click(x, y): acts.append(('ACTION6', off + 3 * y + 1, off + 3 * x + 1))
    hlines = set(); vlines = set()
    for i, mr in enumerate(mirrors):
        (hlines if mr['k'] == 'h' else vlines).update([mr['p'], mpos[i]])
    for i, mr in enumerate(mirrors):
        d = mpos[i] - mr['p']
        if d == 0: continue
        if sel != ('m', i):
            cells = [c for c in mr['cells'] if (mr['k'] == 'v' and c[1] not in hlines) or (mr['k'] == 'h' and c[0] not in vlines)] or mr['cells']
            if not cells: continue
            click(*cells[0]); sel = ('m', i)
        if mr['k'] == 'v': acts += [('ACTION4' if d > 0 else 'ACTION3', None, None)] * abs(d)
        else: acts += [('ACTION2' if d > 0 else 'ACTION1', None, None)] * abs(d)
    order = sorted(ppos.keys(), key=lambda i: 0 if sel == ('p', i) else 1)
    occupied_final = set()
    cur = {i: set(pc['cells']) for i, pc in enumerate(pieces)}
    for i in order:
        pc = pieces[i]; px, py = ppos[i]
        dx = px - pc['x']; dy = py - pc['y']
        if dx == 0 and dy == 0:
            continue
        if sel != ('p', i):
            others = set(occupied_final)
            for j in cur:
                if j != i: others |= cur[j]
            cand = [c for c in pc['cells'] if c not in others] or pc['cells']
            click(*cand[0]); sel = ('p', i)
        acts += [('ACTION4' if dx > 0 else 'ACTION3', None, None)] * abs(dx)
        acts += [('ACTION2' if dy > 0 else 'ACTION1', None, None)] * abs(dy)
        cur[i] = set((px + a, py + b) for a, b in pc['rel'])
    return acts

def act(obs, available, memory):
    obs = np.array(obs)
    plan = memory.get('plan')
    if not plan:
        try:
            plan = _plan(obs)
        except Exception:
            plan = []
        memory['plan'] = plan
        memory['fails'] = memory.get('fails', 0) + (0 if plan else 1)
    if plan:
        a = plan.pop(0)
        if a[0] in available or not available:
            return a
    return (random.choice(['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4']), None, None)
