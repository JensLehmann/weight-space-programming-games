import time

G = 20


def _parse(obs):
    bg = int(obs[63, 63])
    g = [[int(obs[3 + 3 * y, 3 + 3 * x]) for x in range(G)] for y in range(G)]
    seen = set()
    comps = []
    for y in range(G):
        for x in range(G):
            if g[y][x] != bg and (x, y) not in seen:
                q = [(x, y)]
                seen.add((x, y))
                cells = []
                while q:
                    cx, cy = q.pop()
                    cells.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < G and 0 <= ny < G and (nx, ny) not in seen and g[ny][nx] != bg:
                            seen.add((nx, ny))
                            q.append((nx, ny))
                comps.append(cells)
    return g, bg, comps


def _rots(w, h, cells, conn):
    lst = []
    c = frozenset(cells)
    cn = frozenset(conn)
    for _ in range(4):
        lst.append((w, h, c, cn))
        w, h, c, cn = (h, w,
                       frozenset((h - 1 - y, x) for x, y in c),
                       frozenset((h - 1 - y, x) for x, y in cn))
    return lst


def _solve(R, idx, anchor, aopts, limit=60, tl=3.5):
    t0 = time.time()
    nconn = {i: max(len(o[3]) for o in R[i]) for i in idx}
    cnt = {}
    placed = {i: None for i in idx}
    sols = []
    keys = set()

    def place(i, k, px, py, d):
        for (cx, cy) in R[i][k][3]:
            c = (px + cx, py + cy)
            v = cnt.get(c, 0) + d
            if v == 0:
                cnt.pop(c, None)
            else:
                cnt[c] = v

    def bbox_ok():
        mnx = mny = 10 ** 9
        mxx = mxy = -10 ** 9
        for i, p in placed.items():
            if p is None:
                continue
            k, px, py = p
            w, h = R[i][k][0], R[i][k][1]
            mnx = min(mnx, px); mny = min(mny, py)
            mxx = max(mxx, px + w); mxy = max(mxy, py + h)
        return mxx - mnx <= G and mxy - mny <= G

    def rec(rem):
        if len(sols) >= limit or time.time() - t0 > tl:
            return
        un = [c for c, v in cnt.items() if v == 1]
        if not un:
            if rem == 0:
                key = frozenset(
                    (i, frozenset((p[1] + x, p[2] + y) for x, y in R[i][p[0]][2]))
                    for i, p in placed.items())
                if key not in keys:
                    keys.add(key)
                    sols.append(dict(placed))
            return
        if len(un) > sum(nconn[i] for i in idx if placed[i] is None):
            return
        c = min(un)
        for i in idx:
            if placed[i] is not None:
                continue
            for k in range(len(R[i])):
                w, h, cells, conn = R[i][k]
                for (cx, cy) in conn:
                    px, py = c[0] - cx, c[1] - cy
                    if not all(cnt.get((px + x, py + y), 0) <= 1 for x, y in conn):
                        continue
                    placed[i] = (k, px, py)
                    if bbox_ok():
                        place(i, k, px, py, 1)
                        rec(rem - 1)
                        place(i, k, px, py, -1)
                    placed[i] = None

    for k0 in aopts:
        if not R[anchor][k0][3]:
            continue
        placed[anchor] = (k0, 0, 0)
        place(anchor, k0, 0, 0, 1)
        rec(len(idx) - 1)
        place(anchor, k0, 0, 0, -1)
        placed[anchor] = None
    return sols


def _bounce(i, d, n):
    if d:
        j = i + 1
        if j >= n:
            d = False; j = i - 1
            if j < 0:
                d = True; j = i
    else:
        j = i - 1
        if j < 0:
            d = True; j = i + 1
            if j >= n:
                d = False; j = i
    return j, d


def _click_cell(R, pos, rot, i):
    occ = {}
    for j in range(len(R)):
        for (a, b) in R[j][rot[j]][2]:
            c = (pos[j][0] + a, pos[j][1] + b)
            occ[c] = occ.get(c, 0) + 1
    cells = sorted(R[i][rot[i]][2])
    for (a, b) in cells:
        c = (pos[i][0] + a, pos[i][1] + b)
        if occ.get(c, 0) == 1:
            return c
    a, b = cells[0]
    return (pos[i][0] + a, pos[i][1] + b)


def _plan(m, sol):
    R = m['R']
    kind = m['kind']
    pos = [list(p) for p in m['pos']]
    rot = list(m['rot'])
    sdir = list(m['sdir'])
    sel = m['sel']
    press = {}
    ppos = {}
    ndir = {}
    for i, s in sol.items():
        k = s[0]
        x, y = pos[i]
        r = rot[i]
        d = sdir[i]
        p = 0
        n = len(R[i])
        while r != k:
            if p > 3 * n + 2:
                return None
            if kind[i] == 'stack':
                r, d = _bounce(r, d, n)
            else:
                r = (r + 1) % 4
            p += 1
            w, h = R[i][r][0], R[i][r][1]
            x = max(0, min(x, G - w))
            y = max(0, min(y, G - h))
        press[i] = p
        ppos[i] = (x, y)
        ndir[i] = d

    def best_t(ax):
        lo = max(-sol[i][1 + ax] for i in sol)
        hi = min(G - R[i][sol[i][0]][ax] - sol[i][1 + ax] for i in sol)
        bt, bc = None, None
        for t in range(lo, hi + 1):
            c = sum(abs(t + sol[i][1 + ax] - ppos[i][ax]) for i in sol)
            if bc is None or c < bc:
                bt, bc = t, c
        return bt

    tx = best_t(0)
    ty = best_t(1)
    if tx is None or ty is None:
        return None
    order = sorted(sol.keys(), key=lambda i: 0 if i == sel else 1)
    acts = []
    for i in order:
        tgt = (tx + sol[i][1], ty + sol[i][2])
        if press[i] == 0 and tuple(pos[i]) == tgt:
            continue
        if i != sel:
            cx, cy = _click_cell(R, pos, rot, i)
            acts.append(('ACTION6', 3 + 3 * cy, 3 + 3 * cx))
            sel = i
        for _ in range(press[i]):
            acts.append(('ACTION5', None, None))
        rot[i] = sol[i][0]
        sdir[i] = ndir[i]
        x, y = ppos[i]
        while x < tgt[0]:
            acts.append(('ACTION4', None, None)); x += 1
        while x > tgt[0]:
            acts.append(('ACTION3', None, None)); x -= 1
        while y < tgt[1]:
            acts.append(('ACTION2', None, None)); y += 1
        while y > tgt[1]:
            acts.append(('ACTION1', None, None)); y -= 1
        pos[i] = [x, y]
    return acts, pos, rot, sel, sdir


def _init(m, obs):
    g, bg, comps = _parse(obs)
    base = []
    grey = False
    sel = None
    for cells in comps:
        if set(g[y][x] for x, y in cells) <= {3, 4}:
            grey = True
    for ci, cells in enumerate(comps):
        x0 = min(c[0] for c in cells)
        y0 = min(c[1] for c in cells)
        w = max(c[0] for c in cells) - x0 + 1
        h = max(c[1] for c in cells) - y0 + 1
        rel = [(x - x0, y - y0) for x, y in cells]
        cols = set(g[y][x] for x, y in cells)
        conn = [(x - x0, y - y0) for x, y in cells if g[y][x] in (8, 3)]
        if grey:
            if not cols <= {3, 4}:
                sel = ci
            else:
                conn = []
        elif 0 in cols:
            sel = ci
        base.append([w, h, rel, conn, x0, y0])
    m['grey'] = grey
    m['base'] = base
    m['sel'] = sel
    n = len(base)
    m['kind'] = ['rot'] * n
    m['opts'] = [None] * n
    m['rot'] = [0] * n
    m['sdir'] = [True] * n
    m['disc'] = [i for i in range(n) if grey and i != sel]
    m['sq'] = []
    if grey and sel is not None:
        cols = set(g[y][x] for x, y in comps[sel])
        if 0 in cols:
            m['sq'].append(sel)
    m['sd'] = None
    m['pend'] = None
    m['phase'] = 'disc'
    m['queue'] = []
    m['sols'] = None
    m['tried'] = set()


def _sel_shape(obs):
    g, bg, comps = _parse(obs)
    for cells in comps:
        cols = set(g[y][x] for x, y in cells)
        if not cols <= {3, 4}:
            cs = [(x, y) for x, y in cells if g[y][x] != 4]
            cn = [(x, y) for x, y in cs if g[y][x] in (8, 3)]
            return frozenset(cs), frozenset(cn)
    return None, None


def _stack_step(m, obs):
    sd = m['sd']
    cs, cn = _sel_shape(obs)
    if cs is None:
        m['sd'] = None
        return None
    sd['opts'][cs] = cn
    sd['seq'].append(cs)
    seq = sd['seq']
    b = 0
    for t in range(1, len(seq) - 1):
        if seq[t + 1] == seq[t - 1]:
            b += 1
    if b < 2 and len(seq) < 16:
        return ('ACTION5', None, None)
    adj = {}
    for a, c in zip(seq, seq[1:]):
        if a != c:
            adj.setdefault(a, set()).add(c)
            adj.setdefault(c, set()).add(a)
    keys = list(sd['opts'].keys())
    i = sd['i']
    if len(keys) < 2 or any(len(v) > 2 for v in adj.values()):
        m['sd'] = None
        return None
    start = [k for k in keys if len(adj.get(k, ())) <= 1][0]
    order = [start]
    while len(order) < len(keys):
        nxt = [k for k in adj.get(order[-1], ()) if k not in order]
        if not nxt:
            break
        order.append(nxt[0])
    if len(order) < len(keys):
        m['sd'] = None
        return None
    ox = min(x for k in keys for x, y in k)
    oy = min(y for k in keys for x, y in k)
    opts = []
    for k in order:
        cells = frozenset((x - ox, y - oy) for x, y in k)
        conn = frozenset((x - ox, y - oy) for x, y in sd['opts'][k])
        w = max(x for x, y in cells) + 1
        h = max(y for x, y in cells) + 1
        opts.append((w, h, cells, conn))
    cur = order.index(seq[-1])
    prv = order.index(seq[-2])
    m['opts'][i] = opts
    m['kind'][i] = 'stack'
    m['rot'][i] = cur
    m['sdir'][i] = cur > prv
    bb = m['base'][i]
    bb[4], bb[5] = ox, oy
    bb[3] = list(opts[cur][3])
    bb[2] = list(opts[cur][2])
    m['sel'] = i
    m['sd'] = None
    return None


def _fallback(m):
    m['fb'] = m.get('fb', 0) + 1
    fb = m['fb']
    n = len(m.get('base', []))
    m['phase'] = 'init'
    if n == 0:
        return ('ACTION5', None, None)
    if m.get('sel') is None or fb % 3 == 0:
        i = fb % n
        if m.get('sel') == i and n > 1:
            i = (i + 1) % n
        b = m['base'][i]
        cx, cy = b[4] + b[2][0][0], b[5] + b[2][0][1]
        return ('ACTION6', 3 + 3 * cy, 3 + 3 * cx)
    return ('ACTION5', None, None)


def _disc(m, obs):
    if m['sd'] is not None:
        r = _stack_step(m, obs)
        if r is not None:
            return r
    if m['pend'] is not None:
        j = m['pend']
        b = m['base'][j]
        conn = []
        has0 = False
        for (a, c) in b[2]:
            x, y = b[4] + a, b[5] + c
            v = int(obs[3 + 3 * y, 3 + 3 * x])
            if v in (8, 3):
                conn.append((a, c))
            if v == 0:
                has0 = True
        b[3] = conn
        m['sel'] = j
        m['pend'] = None
        if has0:
            m['sq'].append(j)
    if m['sq']:
        i = m['sq'].pop(0)
        cs, cn = _sel_shape(obs)
        if cs is not None:
            m['sd'] = {'i': i, 'seq': [cs], 'opts': {cs: cn}}
            return ('ACTION5', None, None)
    if m['disc']:
        j = m['disc'].pop(0)
        m['pend'] = j
        b = m['base'][j]
        cx, cy = b[4] + b[2][0][0], b[5] + b[2][0][1]
        return ('ACTION6', 3 + 3 * cy, 3 + 3 * cx)
    return None


def act(obs, available, memory):
    m = memory
    try:
        if m.get('phase', 'init') == 'init':
            _init(m, obs)
        if m['phase'] == 'disc':
            r = _disc(m, obs)
            if r is not None:
                return r
            base = m['base']
            n = len(base)
            R = []
            for i in range(n):
                if m['kind'][i] == 'stack':
                    R.append(m['opts'][i])
                else:
                    b = base[i]
                    R.append(_rots(b[0], b[1], b[2], b[3]))
            m['R'] = R
            m['pos'] = [[b[4], b[5]] for b in base]
            idx = [i for i in range(n) if any(len(o[3]) > 0 for o in R[i])]
            sols = []
            if idx:
                stacks = [i for i in idx if m['kind'][i] == 'stack']
                if stacks:
                    a = stacks[0]
                    sols = _solve(R, idx, a, list(range(len(R[a]))))
                else:
                    a = max(idx, key=lambda i: len(R[i][0][3]))
                    sols = _solve(R, idx, a, [0])
            if not sols:
                return _fallback(m)
            m['sols'] = sols
            m['phase'] = 'exec'
        if m['phase'] == 'exec':
            if not m['queue']:
                best = None
                for si, s in enumerate(m['sols']):
                    if si in m['tried']:
                        continue
                    p = _plan(m, s)
                    if p is None:
                        continue
                    if best is None or len(p[0]) < len(best[1][0]):
                        best = (si, p)
                if best is None:
                    return _fallback(m)
                si, (acts, pos, rot, sel, sdir) = best
                m['tried'].add(si)
                m['queue'] = list(acts)
                m['pos'], m['rot'], m['sel'], m['sdir'] = pos, rot, sel, sdir
                if not m['queue']:
                    return _fallback(m)
            return m['queue'].pop(0)
    except Exception:
        m['phase'] = 'init'
        return ('ACTION5', None, None)
    return ('ACTION5', None, None)
