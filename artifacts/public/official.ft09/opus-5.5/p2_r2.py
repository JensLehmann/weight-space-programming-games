import numpy as np
import random


def _classify(b):
    c = int(b[1, 1])
    border = [(r, cc, int(b[r, cc])) for r in range(3) for cc in range(3) if not (r == 1 and cc == 1)]
    vals = [v for _, _, v in border]
    if all(v == c for v in vals):
        return ('H', c, None)
    if all(v in (6, c) for v in vals) and 6 in vals:
        pat = [((cc - 1) * 4, (r - 1) * 4) for r, cc, v in border if v == 6]
        return ('N', c, pat)
    pat = [((cc - 1) * 4, (r - 1) * 4, v == 0) for r, cc, v in border]
    return ('B', c, pat)


def _parse(g):
    H, W = g.shape
    best = {}
    for oy in range(4):
        for ox in range(4):
            tiles = {}
            for y in range(oy, H - 2, 4):
                for x in range(ox, W - 2, 4):
                    b = g[y:y + 3, x:x + 3]
                    if (b == 4).any() or (b == 5).any():
                        continue
                    y0, y1 = max(0, y - 1), min(H, y + 4)
                    x0, x1 = max(0, x - 1), min(W, x + 4)
                    win = g[y0:y1, x0:x1].copy()
                    win[y - y0:y - y0 + 3, x - x0:x - x0 + 3] = 4
                    if (win == 5).any():
                        continue
                    tiles[(x, y)] = _classify(b)
            score = len(tiles) + (5 if any(t[0] == 'B' for t in tiles.values()) else 0)
            bscore = len(best) + (5 if any(t[0] == 'B' for t in best.values()) else 0)
            if score > bscore:
                best = tiles
    return best


def _solve(A, bvec, K):
    m = len(A)
    if m == 0:
        return []
    n = len(A[0])
    M = [list(row) + [bv] for row, bv in zip(A, bvec)]
    piv = []
    r = 0
    for c in range(n):
        pr = None
        inv = None
        for i in range(r, m):
            a = M[i][c] % K
            if a:
                try:
                    inv = pow(a, -1, K)
                    pr = i
                    break
                except ValueError:
                    continue
        if pr is None:
            continue
        M[r], M[pr] = M[pr], M[r]
        M[r] = [(v * inv) % K for v in M[r]]
        for i in range(m):
            if i != r and M[i][c] % K:
                f = M[i][c]
                M[i] = [(vi - f * vr) % K for vi, vr in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == m:
            break
    for i in range(r, m):
        if M[i][n] % K:
            return None
    x = [0] * n
    for i, c in enumerate(piv):
        x[c] = M[i][n] % K
    return x


def act(obs, available, memory):
    obs = np.asarray(obs).astype(int)
    tiles = {}
    s_used = 2
    for s in (2, 4):
        g = obs[::s, ::s]
        t = _parse(g)
        if len(t) > len(tiles) and any(v[0] == 'B' for v in t.values()):
            tiles, s_used = t, s
    if not tiles:
        return ('ACTION6', 32, 32)
    g = obs[::s_used, ::s_used]
    clickable = sorted(p for p, t in tiles.items() if t[0] in ('H', 'N'))
    cset = set(clickable)
    colors = {p: tiles[p][1] for p in clickable}

    hk = memory.setdefault('hk', [(0, 0)])
    eff_learn = memory.setdefault('eff', {})
    nxt = memory.setdefault('next', {})

    # learning from last click
    last = memory.get('last')
    if last is not None:
        cp, before = last
        if set(before.keys()) == cset and cp in cset:
            changed = [p for p in clickable if before[p] != colors[p]]
            if changed:
                offs = sorted({(p[0] - cp[0], p[1] - cp[1]) for p in changed})
                eff_learn[cp] = offs
                if tiles[cp][0] == 'H' and offs != sorted(hk):
                    memory['hk'] = hk = offs
                for p in changed:
                    nxt[before[p]] = colors[p]
        memory['last'] = None

    # color order
    relevant = set(colors.values()) | {t[1] for t in tiles.values() if t[0] == 'B'}
    mask = np.ones(g.shape, bool)
    for (x, y) in tiles:
        mask[y:y + 3, x:x + 3] = False
    legend = []
    H, W = g.shape
    for y in range(H):
        for x in range(W):
            if mask[y, x]:
                v = int(g[y, x])
                if v in relevant and v not in legend:
                    legend.append(v)
    if clickable:
        vals = [colors[p] for p in clickable]
        maj = max(set(vals), key=vals.count)
    else:
        maj = None
    for v in sorted(relevant):
        if v not in legend:
            legend.append(v)
    start = legend[0] if legend else maj
    order = [start]
    while len(order) < len(legend):
        cur = order[-1]
        cand = nxt.get(cur)
        if cand is None or cand in order:
            cand = next(v for v in legend if v not in order)
        order.append(cand)
    for v, w in nxt.items():
        for z in (v, w):
            if z not in order:
                order.append(z)
    K = max(2, len(order))
    while len(order) < K:
        order.append(-100 - len(order))
    idx = {c: i for i, c in enumerate(order)}

    # constraints
    allowed = {}
    for p, t in tiles.items():
        if t[0] != 'B':
            continue
        c = t[1]
        for dx, dy, eq in t[2]:
            q = (p[0] + dx, p[1] + dy)
            if q not in cset:
                continue
            S = allowed.setdefault(q, set(range(K)))
            if c in idx:
                ci = idx[c]
                if eq:
                    S &= {ci}
                else:
                    S.discard(ci)
            elif eq:
                S.clear()

    def effects(p):
        if p in eff_learn:
            offs = eff_learn[p]
        elif tiles[p][0] == 'H':
            offs = hk
        else:
            offs = [(0, 0)] + list(tiles[p][2])
        return [(p[0] + dx, p[1] + dy) for dx, dy in offs if (p[0] + dx, p[1] + dy) in cset]

    cells = [q for q in sorted(allowed) if allowed[q]]
    vidx = {p: i for i, p in enumerate(clickable)}
    n = len(clickable)
    A = [[0] * n for _ in cells]
    rowof = {q: i for i, q in enumerate(cells)}
    for p in clickable:
        for q in effects(p):
            if q in rowof:
                A[rowof[q]][vidx[p]] = (A[rowof[q]][vidx[p]] + 1) % K
    curi = {q: idx.get(colors[q], 0) for q in cells}

    sol = None
    for attempt in range(60):
        rng = random.Random(attempt)
        bvec = []
        for q in cells:
            S = sorted(allowed[q])
            if attempt == 0:
                tgt = curi[q] if curi[q] in S else S[0]
            else:
                tgt = rng.choice(S)
            bvec.append((tgt - curi[q]) % K)
        sol = _solve(A, bvec, K)
        if sol is not None:
            break

    target = None
    if sol is not None:
        for i, p in enumerate(clickable):
            if sol[i] % K:
                target = p
                break
    if target is None:
        # fallback: click a cell violating constraints
        for q in cells:
            if curi[q] not in allowed[q]:
                target = q
                break
    if target is None:
        target = clickable[0] if clickable else None
    if target is None:
        return ('ACTION6', 32, 32)
    memory['last'] = (target, dict(colors))
    row = (target[1] + 1) * s_used
    col = (target[0] + 1) * s_used
    return ('ACTION6', int(row), int(col))
