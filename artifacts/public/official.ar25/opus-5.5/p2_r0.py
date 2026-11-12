import time, math
import numpy as np

def _detect(obs):
    best = None
    for N in range(32, 7, -1):
        s = 64 // N; used = N * s; o = (64 - used) // 2
        mask = np.ones((64, 63), bool)
        mask[o:o+used, o:min(o+used, 63)] = False
        cnt = int(mask.sum())
        if cnt == 0:
            if best is None:
                best = (0, N, s, o)
            continue
        if (obs[:, :63][mask] == 5).all():
            if best is None or cnt > best[0]:
                best = (cnt, N, s, o)
    if best is None:
        return 21, 3, 0
    return best[1], best[2], best[3]

def _parse(obs):
    N, s, o = _detect(obs)
    vg = math.ceil((64 - N * s) / 2) + 1
    corner = {}; center = {}
    for y in range(N):
        for x in range(N):
            c = int(obs[o + y*s, o + x*s])
            corner[(x, y)] = c
            cx = x*s + vg + 1; cy = y*s + vg + 1
            if s == 3: cx -= 2; cy -= 2
            if 0 <= cx < 64 and 0 <= cy < 64 and s in (3, 4, 5):
                center[(x, y)] = int(obs[cy, cx])
            else:
                center[(x, y)] = c
    targets = [p for p in corner if corner[p] == 11 or center[p] == 11]
    thr = max(4, N // 3)
    mov = []; fixed = []
    for x in range(N):
        cells = [(x, y) for y in range(N) if corner[(x, y)] == 10]
        if len(cells) >= thr:
            fx = any(center[p] == 10 for p in cells)
            (fixed if fx else mov).append(('V', x))
    for y in range(N):
        cells = [(x, y) for x in range(N) if corner[(x, y)] == 10]
        if len(cells) >= thr:
            fx = any(center[p] == 10 for p in cells)
            (fixed if fx else mov).append(('H', y))
    pc = set(p for p in corner if corner[p] not in (9, 10, 11, 4))
    pieces = []; seen = set()
    for p in sorted(pc, key=lambda q: (q[1], q[0])):
        if p in seen: continue
        comp = []; st = [p]; seen.add(p)
        while st:
            q = st.pop(); comp.append(q)
            for d in ((1,0),(-1,0),(0,1),(0,-1)):
                r = (q[0]+d[0], q[1]+d[1])
                if r in pc and r not in seen:
                    seen.add(r); st.append(r)
        pieces.append(comp)
    return N, s, o, targets, mov, fixed, pieces

def _orbit(c, mirrors, N):
    seen = {c}; fr = [c]
    for d in range(13):
        nf = []
        for (x, y) in fr:
            for k, m in mirrors:
                q = (2*m - x, y) if k == 'V' else (x, 2*m - y)
                if q not in seen:
                    seen.add(q); nf.append(q)
        fr = nf
        if not fr: break
    return [p for p in seen if 0 <= p[0] < N and 0 <= p[1] < N]

def _search(N, targets, mov, fixed, pieces, t0):
    tidx = {p: i for i, p in enumerate(targets)}
    full = (1 << len(targets)) - 1
    if full == 0: return None
    import itertools
    ranges = [list(range(N)) for _ in mov]
    cfgs = []
    for combo in itertools.product(*ranges):
        c = sum(abs(v - m[1]) + (1 if v != m[1] else 0) for v, m in zip(combo, mov))
        cfgs.append((c, combo))
    cfgs.sort(key=lambda t: t[0])
    shapes = []
    for pcs in pieces:
        shapes.append(pcs)
    best = None
    for mcost, combo in cfgs:
        if best is not None and mcost >= best[0]: break
        if time.time() - t0 > 7.0: break
        mirrors = fixed + [(m[0], v) for m, v in zip(mov, combo)]
        cm = {}
        for y in range(N):
            for x in range(N):
                mk = 0
                for q in _orbit((x, y), mirrors, N):
                    i = tidx.get(q)
                    if i is not None: mk |= 1 << i
                cm[(x, y)] = mk
        states = {0: (0, ())}
        for pcs in shapes:
            xs = [p[0] for p in pcs]; ys = [p[1] for p in pcs]
            opts = {}
            for dx in range(-min(xs), N - max(xs)):
                for dy in range(-min(ys), N - max(ys)):
                    mk = 0
                    for (x, y) in pcs:
                        mk |= cm[(x+dx, y+dy)]
                    c = abs(dx) + abs(dy) + (1 if (dx or dy) else 0)
                    if mk not in opts or opts[mk][0] > c:
                        opts[mk] = (c, (dx, dy))
            kept = []
            for mk, (c, d) in sorted(opts.items(), key=lambda t: t[1][0]):
                if any((mk | k) == k for k, _, _ in kept): continue
                kept.append((mk, c, d))
            new = {}
            for sm, (sc, ch) in states.items():
                for mk, c, d in kept:
                    nm = sm | mk; nc = sc + c
                    if best is not None and mcost + nc >= best[0]: continue
                    if nm not in new or new[nm][0] > nc:
                        new[nm] = (nc, ch + (d,))
            if len(new) > 4000:
                it = sorted(new.items(), key=lambda t: (-bin(t[0]).count('1'), t[1][0]))[:4000]
                new = dict(it)
            states = new
            if not states: break
        if full in states:
            tot = mcost + states[full][0]
            if best is None or tot < best[0]:
                best = (tot, combo, states[full][1])
    return best

def _plan(obs):
    t0 = time.time()
    N, s, o, targets, mov, fixed, pieces = _parse(obs)
    res = _search(N, targets, mov, fixed, pieces, t0)
    if res is None: return []
    _, combo, moves = res
    acts = []
    cur_m = [list(m) for m in mov]
    cur_p = [list(p) for p in pieces]
    px = lambda g: o + g * s + s // 2
    def occ_pieces(excl=-1):
        st = set()
        for i, p in enumerate(cur_p):
            if i != excl: st.update(p)
        return st
    for i, v in enumerate(combo):
        k, c = cur_m[i]
        if v == c: continue
        occ = occ_pieces()
        others = [m for j, m in enumerate(cur_m) if j != i] + [list(f) for f in fixed]
        cell = None
        for t in range(N):
            q = (c, t) if k == 'V' else (t, c)
            if q in occ: continue
            if any((mm[0] == 'V' and q[0] == mm[1]) or (mm[0] == 'H' and q[1] == mm[1]) for mm in others):
                continue
            cell = q; break
        if cell is None: cell = (c, 0) if k == 'V' else (0, c)
        acts.append(('ACTION6', px(cell[1]), px(cell[0])))
        d = v - c
        if k == 'V':
            acts += [('ACTION4' if d > 0 else 'ACTION3', None, None)] * abs(d)
        else:
            acts += [('ACTION2' if d > 0 else 'ACTION1', None, None)] * abs(d)
        cur_m[i][1] = v
    for i, (dx, dy) in enumerate(moves):
        if dx == 0 and dy == 0: continue
        occ = occ_pieces(i)
        cand = [q for q in cur_p[i] if q not in occ]
        cell = cand[0] if cand else cur_p[i][0]
        acts.append(('ACTION6', px(cell[1]), px(cell[0])))
        acts += [('ACTION4' if dx > 0 else 'ACTION3', None, None)] * abs(dx)
        acts += [('ACTION2' if dy > 0 else 'ACTION1', None, None)] * abs(dy)
        cur_p[i] = [(x+dx, y+dy) for (x, y) in cur_p[i]]
    return acts

def act(obs, available, memory):
    obs = np.asarray(obs)
    plan = memory.get('plan')
    if not plan:
        n = memory.get('replans', 0)
        memory['replans'] = n + 1
        plan = []
        if n < 6:
            try:
                plan = _plan(obs)
            except Exception:
                plan = []
        memory['plan'] = plan
    if plan:
        a = plan.pop(0)
        if a[0] in available or not available:
            return a
    if 'ACTION5' in available:
        return ('ACTION5', None, None)
    return (available[0] if available else 'ACTION5', None, None)
