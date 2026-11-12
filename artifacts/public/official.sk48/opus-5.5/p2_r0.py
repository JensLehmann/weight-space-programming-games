import time
import heapq
import numpy as np

DIR = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}
MV = {'ACTION1': (0, -1), 'ACTION2': (0, 1), 'ACTION3': (-1, 0), 'ACTION4': (1, 0)}
MOVES = ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4']


def chainlen(g, x, y, r):
    dx, dy = DIR[r]
    hz = r in (0, 180)
    pts = ((2, 0), (3, 0), (2, 5), (3, 5)) if hz else ((0, 2), (0, 3), (5, 2), (5, 3))
    L = 1
    while True:
        cx, cy = x + 6 * L * dx, y + 6 * L * dy
        if not (0 <= cx <= 58 and 0 <= cy <= 58):
            break
        if all(g[cy + a][cx + b] in (1, 2, 3) for a, b in pts):
            L += 1
        else:
            break
    return L


def parse(obs, memory):
    g = obs.tolist()
    heads = []
    for y in range(59):
        row = g[y]; row5 = g[y + 5]
        for x in range(59):
            B = row[x]
            if B not in (5, 6, 15) or row[x + 5] != B or row5[x] != B or row5[x + 5] != B:
                continue
            if not (row[x + 1] == B and row[x + 4] == B and g[y + 1][x] == B and g[y + 4][x] == B
                    and g[y + 1][x + 5] == B and g[y + 4][x + 5] == B and row5[x + 1] == B and row5[x + 4] == B):
                continue
            C = g[y + 2][x + 2]
            if not (g[y + 2][x + 3] == C == g[y + 3][x + 2] == g[y + 3][x + 3]):
                continue
            R = g[y + 1][x + 1]
            if not (g[y + 1][x + 4] == R == g[y + 4][x + 1] == g[y + 4][x + 4]):
                continue
            if not ((B in (6, 15) and C == B and R in (0, 4)) or (B == 5 and C == 11 and R == 5)):
                continue
            rot = 0
            if B == 5:
                for rr in (0, 90, 180, 270):
                    if chainlen(g, x, y, rr) > 1:
                        rot = rr
                        break
            else:
                if g[y + 2][x + 5] != B and g[y + 3][x + 5] != B: rot = 0
                elif g[y + 5][x + 2] != B and g[y + 5][x + 3] != B: rot = 90
                elif g[y + 2][x] != B and g[y + 3][x] != B: rot = 180
                elif g[y][x + 2] != B and g[y][x + 3] != B: rot = 270
            heads.append((x, y, B, R, rot, chainlen(g, x, y, rot)))
    real = [h for h in heads if h[2] in (6, 15)]
    if not real:
        return None
    ox, oy = real[0][0] % 6, real[0][1] % 6
    top = [h for h in heads if h[1] + 5 < 53]
    bot = [h for h in heads if h[1] >= 53]
    headcells = set((h[0], h[1]) for h in heads)
    xs = []; ys = []
    for y in range(oy, 48, 6):
        for x in range(ox, 59, 6):
            if g[y][x] == 4 and g[y + 5][x + 5] == 4 and g[y][x + 5] == 4 and y + 5 < 53:
                xs.append(x); ys.append(y)
    if xs:
        area = (min(xs), min(ys), max(xs) + 6, max(ys) + 6)
    else:
        area = (0, 0, 64, 53)
    AX, AY, AX2, AY2 = area
    obst = set()
    for y in range(oy, 48, 6):
        for x in range(ox, 59, 6):
            if AX <= x and x + 6 <= AX2 and AY <= y and y + 6 <= AY2:
                if g[y][x] == 5 and g[y + 2][x + 2] == 5 and g[y + 5][x + 5] == 5 and g[y + 1][x + 1] == 5:
                    obst.add((x, y))
    blocks = []
    for y in range(oy, 48, 6):
        if y + 5 >= 53:
            continue
        for x in range(ox, 59, 6):
            if (x, y) in headcells:
                continue
            r = g[y + 1][x + 1]
            if r in (4, 5):
                continue
            if g[y + 1][x + 4] == r and g[y + 4][x + 1] == r and g[y + 4][x + 4] == r and g[y + 1][x + 2] == r:
                blocks.append((x, y, r))
    movable = sorted([h for h in top if h[2] in (6, 15)], key=lambda h: (h[2], h[4], h[1], h[0]))
    statics = [(h[0], h[1], h[4], h[5]) for h in top if h[2] not in (6, 15)]
    targets = {}
    for j, h in enumerate(movable):
        for b in bot:
            if b[2] == h[2]:
                dx, dy = DIR[b[4]]
                targets[j] = [g[b[1] + 6 * i * dy + 1][b[0] + 6 * i * dx + 1] for i in range(1, b[5])]
                break
    sel = 0
    for j, h in enumerate(movable):
        if h[3] == 0:
            sel = j
    m = (obs == 2) | (obs == 3)
    m[53:, :] = False
    m[AY:AY2, AX:AX2] = False
    for h in top:
        dx, dy = DIR[h[4]]
        for i in range(h[5]):
            cx, cy = h[0] + 6 * i * dx, h[1] + 6 * i * dy
            m[max(cy, 0):max(cy + 6, 0), max(cx, 0):max(cx + 6, 0)] = False
    yy, xx = np.nonzero(m)
    rails = memory.setdefault('rails', set())
    rails.update(zip(xx.tolist(), yy.tolist()))
    ctx = {'rots': [h[4] for h in movable], 'statics': statics, 'area': area, 'obst': obst,
           'targets': targets, 'rails': rails}
    st = (sel, tuple((h[0], h[1], h[5]) for h in movable), tuple(sorted(blocks)))
    return ctx, st


def sim(ctx, st, a):
    sel, hs, bl = st
    if a[0] == 'S':
        return (a[1], hs, bl)
    mx, my = MV[a]
    rots = ctx['rots']; AX, AY, AX2, AY2 = ctx['area']; obst = ctx['obst']
    segkey = {}

    def mk(x, y, r, L):
        dx, dy = DIR[r]; hz = r in (0, 180); ch = []
        for i in range(L):
            o = [x + 6 * i * dx, y + 6 * i * dy, r, 1, 0]
            ch.append(o)
            segkey.setdefault((o[0], o[1], hz), []).append(o)
        return ch

    chains = [mk(x, y, rots[j], L) for j, (x, y, L) in enumerate(hs)]
    for (x, y, r, L) in ctx['statics']:
        mk(x, y, r, L)
    bmap = {}
    for (x, y, c) in bl:
        bmap[(x, y)] = [x, y, 0, 0, c]

    def oob(s, ax, ay):
        x = s[0] + 6 * ax; y = s[1] + 6 * ay
        return x < AX or x + 6 > AX2 or y < AY or y + 6 > AY2 or (x, y) in obst

    def segat(x, y, hz, ex):
        for o in segkey.get((x, y, hz), ()):
            if o is not ex:
                return o
        return None

    moved = set()
    DX, DY = 6 * mx, 6 * my

    def push(s, ex=None):
        if id(s) in moved:
            return True
        if oob(s, mx, my):
            if not (s[3] == 1 and (DIR[s[2]] == (-mx, -my) or oob(s, 0, 0))):
                return False
        if s[3] == 1:
            for ox, oy in ((0, 0), (DX, DY)):
                b = bmap.get((s[0] + ox, s[1] + oy))
                if b:
                    if push(b, s):
                        moved.add(id(b))
                    elif (DIR[s[2]][0] == 0) != (mx == 0):
                        return False
        else:
            hz = mx != 0
            for ox, oy in ((0, 0), (DX, DY)):
                k = segat(s[0] + ox, s[1] + oy, not hz, ex)
                if k and (DIR[k[2]][0] == 0) != (mx == 0):
                    return False
            w = bmap.get((s[0] + DX, s[1] + DY))
            if w and not push(w):
                return False
        moved.add(id(s))
        return True

    hx, hy, L = hs[sel]; r = rots[sel]; bx, by = DIR[r]; C = chains[sel]
    if (mx, my) == (bx, by):
        if oob(C[-1], mx, my):
            return None
        for k in C:
            push(k)
        nh = (hx, hy, L + 1)
    elif (mx, my) == (-bx, -by):
        if L == 1:
            return None
        C.pop(0)
        for k in C:
            push(k)
        nh = (hx, hy, L - 1)
    else:
        rails = ctx['rails']
        if my:
            a0 = min(hy, hy + DY)
            ok = any((hx + 2, a0 + kk) in rails for kk in (5, 6, 7))
        else:
            a0 = min(hx, hx + DX)
            ok = any((a0 + kk, hy + 2) in rails for kk in (5, 6, 7))
        if not ok:
            return None
        for k in C:
            if not push(k):
                return None
        nh = (hx + DX, hy + DY, L)
    nb = []
    for o in bmap.values():
        if id(o) in moved:
            o[0] += DX; o[1] += DY
        nb.append((o[0], o[1], o[4]))
    hs2 = list(hs); hs2[sel] = nh
    ns = (sel, tuple(hs2), tuple(sorted(nb)))
    return None if ns == st else ns


def heur(ctx, st):
    sel, hs, bl = st
    bm = {(x, y): c for x, y, c in bl}
    h = 0; ok = True
    for j, t in ctx['targets'].items():
        x, y, L = hs[j]; dx, dy = DIR[ctx['rots'][j]]
        cols = []
        for i in range(L):
            p = (x + 6 * i * dx, y + 6 * i * dy)
            if p in bm:
                cols.append(bm[p])
        first = None
        for i in range(len(t)):
            if not (i < len(cols) and cols[i] == t[i]):
                h += 3; ok = False
                if first is None:
                    first = i
        if first is not None:
            c = t[first]; best = 0
            ds = [(abs(b - y) if dy == 0 else abs(a - x)) // 6 for a, b, cc in bl if cc == c]
            if ds:
                best = min(ds)
            h += best
    return h, ok


def search(ctx, st, deadline, bad, W, hc):
    gc = {st: 0}; par = {st: None}
    if st not in hc:
        hc[st] = heur(ctx, st)
    cnt = 0
    heap = [(W * hc[st][0], hc[st][0], cnt, 0, st)]
    tgt = list(ctx['targets'].keys())
    while heap:
        if time.time() > deadline:
            break
        f, hh, _, g, s = heapq.heappop(heap)
        if g > gc[s]:
            continue
        acts = MOVES + [('S', j) for j in tgt if j != s[0]]
        for a in acts:
            key = (s, a if isinstance(a, str) else a[1])
            if key in bad:
                continue
            n = sim(ctx, s, a)
            if n is None:
                continue
            ng = g + 1
            if n in gc and gc[n] <= ng:
                continue
            gc[n] = ng; par[n] = (s, a)
            if n not in hc:
                hc[n] = heur(ctx, n)
            hn, okn = hc[n]
            if okn:
                return n, gc, par
            cnt += 1
            heapq.heappush(heap, (ng + W * hn, hn, cnt, ng, n))
    return None, gc, par


def path_to(par, end):
    path = []
    while par.get(end) is not None:
        s, a = par[end]
        path.append((a, end))
        end = s
    path.reverse()
    return path


def solve(ctx, st, t0, bad, visits):
    if heur(ctx, st)[1]:
        return []
    hc = {}
    best = None
    for W, dl in ((6, 2.5), (3, 5.0), (1.5, 7.5)):
        if time.time() - t0 > dl:
            continue
        goal, gc, par = search(ctx, st, t0 + dl, bad, W, hc)
        if goal is not None:
            return path_to(par, goal)
        for n in gc:
            if n == st:
                continue
            v = hc[n][0] + 3 * visits.get(n, 0) + 0.01 * gc[n]
            if best is None or v < best[0]:
                best = (v, n, par)
    if best is None:
        return []
    return path_to(best[2], best[1])


def act(obs, available, memory):
    t0 = time.time()
    obs = np.asarray(obs)
    try:
        P = parse(obs, memory)
    except Exception:
        P = None
    if P is None:
        return ('ACTION4', None, None)
    ctx, st = P
    bad = memory.setdefault('bad', set())
    visits = memory.setdefault('visits', {})
    visits[st] = visits.get(st, 0) + 1
    plan = memory.get('plan')
    exp = memory.get('exp')
    if memory.get('prev') == st and memory.get('lasta') is not None:
        bad.add((st, memory['lasta']))
        plan = None
    if exp is not None and exp != st:
        plan = None
    if not plan:
        try:
            plan = solve(ctx, st, t0, bad, visits)
        except Exception:
            plan = []
    if not plan:
        best = None
        for a in MOVES:
            if (st, a) in bad:
                continue
            try:
                n = sim(ctx, st, a)
            except Exception:
                n = None
            if n is None:
                continue
            v = visits.get(n, 0)
            if best is None or v < best[0]:
                best = (v, a, n)
        if best is None:
            k = memory.get('fb', 0); memory['fb'] = k + 1
            a = MOVES[k % 4]; n = None
        else:
            a, n = best[1], best[2]
        memory['plan'] = []; memory['exp'] = n; memory['prev'] = st; memory['lasta'] = a
        return (a, None, None)
    a, ns = plan.pop(0)
    memory['plan'] = plan
    memory['exp'] = ns
    memory['prev'] = st
    if isinstance(a, tuple):
        memory['lasta'] = a[1]
        x, y, _ = st[1][a[1]]
        return ('ACTION6', int(y + 2), int(x + 2))
    memory['lasta'] = a
    return (a, None, None)
