import numpy as np
import itertools, random

BG = 4

def _parse(obs):
    g = np.asarray(obs)[::2, ::2].astype(int)
    H, W = g.shape
    blocks = []
    used = np.zeros((H, W), bool)
    for y in range(H - 2):
        for x in range(W - 2):
            b = g[y:y+3, x:x+3]
            if (b == BG).any():
                continue
            ok = True
            for yy in range(y-1, y+4):
                for xx in range(x-1, x+4):
                    if y <= yy <= y+2 and x <= xx <= x+2:
                        continue
                    if 0 <= yy < H and 0 <= xx < W and g[yy, xx] != BG:
                        ok = False; break
                if not ok: break
            if ok:
                blocks.append((x, y, b.copy()))
                used[y:y+3, x:x+3] = True
    tiles, cons = [], []
    for x, y, b in blocks:
        c = int(b[1, 1])
        if (b == c).all():
            tiles.append({'pos': (x, y), 'kind': 'H', 'color': c, 'pat': None})
        elif c != 6 and (b == 6).any() and (b[b != 6] == c).all():
            pat = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if b[1+dy, 1+dx] == 6]
            if (0, 0) not in pat:
                pat.append((0, 0))
            tiles.append({'pos': (x, y), 'kind': 'N', 'color': c, 'pat': pat})
        else:
            cons.append(((x, y), c, b))
    # legend
    legend = None
    tcols = set(t['color'] for t in tiles)
    best = None
    for r in range(H - 1):
        for c in range(W - 1):
            if used[r, c] or used[r, c+1]:
                continue
            if r > 0 and not (g[r-1, c:c+2] == BG).all():
                continue
            cols = []
            rr = r
            while rr + 1 < H:
                blk = g[rr:rr+2, c:c+2]
                v = blk[0, 0]
                if v == BG or not (blk == v).all() or used[rr:rr+2, c:c+2].any():
                    break
                if cols and cols[-1] == v:
                    break
                lok = (c - 1 < 0 or (g[rr:rr+2, c-1] == BG).all()) and \
                      (c + 2 >= W or (g[rr:rr+2, c+2] == BG).all())
                if not lok:
                    break
                cols.append(int(v)); rr += 2
            if len(cols) >= 2 and len(set(cols)) == len(cols) and tcols <= set(cols):
                if best is None or len(cols) > len(best):
                    best = cols
    legend = best
    return tiles, cons, legend

def _inv(a, p):
    return pow(a, p - 2, p)

def _solve(tiles, cons, k, idx, hpat):
    n = len(tiles)
    pos2i = {t['pos']: i for i, t in enumerate(tiles)}
    coef = [[0]*n for _ in range(n)]
    for s, t in enumerate(tiles):
        x, y = t['pos']
        pat = t['pat'] if t['kind'] == 'N' else [o for o, v in hpat.items() if v]
        for dx, dy in pat:
            j = pos2i.get((x + 4*dx, y + 4*dy))
            if j is not None:
                coef[j][s] = (coef[j][s] + 1) % k
    eqs, neqs = [], []
    for (cx, cy), tgc, b in cons:
        tg = idx.get(tgc)
        if tg is None:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                j = pos2i.get((cx + 4*dx, cy + 4*dy))
                if j is None:
                    continue
                c0 = idx.get(tiles[j]['color'])
                if c0 is None:
                    continue
                rhs = (tg - c0) % k
                weq = b[1+dy, 1+dx] == 0
                if weq:
                    eqs.append(coef[j][:] + [rhs])
                elif k == 2:
                    eqs.append(coef[j][:] + [(rhs + 1) % 2])
                else:
                    neqs.append((coef[j][:], rhs))
    rows = [r[:] for r in eqs]
    piv = []
    r = 0
    for col in range(n):
        pr = None
        for i in range(r, len(rows)):
            if rows[i][col] % k:
                pr = i; break
        if pr is None:
            continue
        rows[r], rows[pr] = rows[pr], rows[r]
        iv = _inv(rows[r][col] % k, k)
        rows[r] = [(v * iv) % k for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col] % k:
                f = rows[i][col]
                rows[i] = [(a - f*bb) % k for a, bb in zip(rows[i], rows[r])]
        piv.append(col); r += 1
    for i in range(r, len(rows)):
        if rows[i][n] % k:
            return None
    rows = rows[:r]
    pset = set(piv)
    rel = set()
    for row in rows:
        for c in range(n):
            if c not in pset and row[c]:
                rel.add(c)
    for cf, _ in neqs:
        for c in range(n):
            if c not in pset and cf[c]:
                rel.add(c)
    rel = sorted(rel)
    f = len(rel)
    if k ** f <= 4096:
        cands = itertools.product(range(k), repeat=f)
    else:
        rnd = random.Random(1)
        cands = [tuple([0]*f)] + [tuple(rnd.randrange(k) for _ in range(f)) for _ in range(2000)]
    best, bcost = None, None
    for asg in cands:
        x = [0]*n
        for c, v in zip(rel, asg):
            x[c] = v
        for row, pc in zip(rows, piv):
            s = row[n]
            for c in rel:
                if row[c]:
                    s -= row[c] * x[c]
            x[pc] = s % k
        ok = True
        for cf, rhs in neqs:
            if sum(a*b for a, b in zip(cf, x)) % k == rhs:
                ok = False; break
        if not ok:
            continue
        cost = sum(x)
        if bcost is None or cost < bcost:
            best, bcost = x, cost
    return best, coef

def act(obs, available, memory):
    tiles, cons, legend = _parse(obs)
    if not tiles:
        return ('ACTION6', 32, 32)
    if 'hpat' not in memory:
        memory['hpat'] = {(0, 0): 1}
        memory['trans'] = {}
        memory['seen'] = []
    hpat = memory['hpat']
    if legend:
        memory['legend'] = legend
    seen = memory['seen']
    for c in [t['color'] for t in tiles] + [c[1] for c in cons]:
        if c not in seen:
            seen.append(c)
    if memory.get('legend'):
        gqb = memory['legend']
    else:
        tr = memory['trans']
        start = tiles[0]['color']
        gqb = [start]
        while gqb[-1] in tr and tr[gqb[-1]] not in gqb:
            gqb.append(tr[gqb[-1]])
        for c in seen:
            if c not in gqb:
                gqb.append(c)
    k = max(2, len(gqb))
    idx = {c: i for i, c in enumerate(gqb)}
    cur = {t['pos']: t['color'] for t in tiles}
    last = memory.get('last')
    if last:
        prev = last['colors']
        for p, c in cur.items():
            if p in prev and prev[p] != c:
                memory['trans'][prev[p]] = c
        if last['kind'] == 'H':
            lx, ly = last['pos']
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    p = (lx + 4*dx, ly + 4*dy)
                    if p in prev and p in cur:
                        hpat[(dx, dy)] = 1 if prev[p] != cur[p] else 0
    res = _solve(tiles, cons, k, idx, hpat)
    choice = None
    if res is not None and res[0] is not None:
        x, coef = res
        for i, v in enumerate(x):
            if v > 0:
                choice = i; break
        if choice is None:
            # already satisfied: click a tile not affecting constrained tiles
            pos2i = {t['pos']: i for i, t in enumerate(tiles)}
            constrained = set()
            for (cx, cy), _, _ in cons:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        j = pos2i.get((cx + 4*dx, cy + 4*dy))
                        if j is not None:
                            constrained.add(j)
            for s in range(len(tiles)):
                if all(coef[j][s] == 0 for j in constrained):
                    choice = s; break
            if choice is None:
                choice = 0
    else:
        tried = memory.setdefault('tried', [])
        for i, t in enumerate(tiles):
            if t['kind'] == 'H' and t['pos'] not in tried:
                choice = i; break
        if choice is None:
            choice = random.randrange(len(tiles))
        tried.append(tiles[choice]['pos'])
    t = tiles[choice]
    memory['last'] = {'pos': t['pos'], 'kind': t['kind'], 'colors': cur}
    x, y = t['pos']
    return ('ACTION6', 2 * (y + 1), 2 * (x + 1))
