import numpy as np, heapq, time, random

DIR = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}
MV = {'ACTION1': (0, -1), 'ACTION2': (0, 1), 'ACTION3': (-1, 0), 'ACTION4': (1, 0)}
S = (1, 2, 3)
MAXTHINK = 10

def parse(o):
    L = o.tolist()
    m4 = (o[:53] == 4)
    ys = np.where(m4.sum(1) >= 10)[0]; xs = np.where(m4.sum(0) >= 10)[0]
    if not len(ys) or not len(xs): return None
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    heads = []
    for y in range(59):
        for x in range(59):
            c = L[y][x]
            if c in (1, 2, 3, 4) or L[y][x+5] != c or L[y+5][x] != c or L[y+5][x+5] != c: continue
            if any(L[a][b] != c for a, b in ((y, x+1), (y, x+4), (y+1, x), (y+4, x), (y+1, x+5), (y+4, x+5), (y+5, x+1), (y+5, x+4))): continue
            ic = L[y+1][x+1]
            if L[y+1][x+4] != ic or L[y+4][x+1] != ic or L[y+4][x+4] != ic: continue
            cc = L[y+2][x+2]
            if L[y+2][x+3] != cc or L[y+3][x+2] != cc or L[y+3][x+3] != cc or cc == ic: continue
            if cc in (1, 2, 3, 4, 5): continue
            sides = {0: [(y+2, x+5), (y+3, x+5)], 90: [(y+5, x+2), (y+5, x+3)],
                     180: [(y+2, x), (y+3, x)], 270: [(y, x+2), (y, x+3)]}
            rot = None
            for r, pts in sides.items():
                if any(L[a][b] != c for a, b in pts): rot = r; break
            if rot is None: continue
            if y < 53 and ((x - bx0) % 6 or (y - by0) % 6): continue
            heads.append((cc, c, x, y, rot, ic))

    def stripe(x, y, rot):
        if not (0 <= x <= 58 and 0 <= y <= 58): return False
        if rot in (0, 180): return L[y+2][x] in S and L[y+3][x] in S
        return L[y][x+2] in S and L[y][x+3] in S

    def walk(x, y, rot):
        n = 1; d = DIR[rot]
        while n < 12 and stripe(x + n*6*d[0], y + n*6*d[1], rot): n += 1
        return n

    def blk(x, y):
        if not (0 <= x <= 58 and 0 <= y <= 58): return None
        c = L[y+1][x+1]
        if c in (4, 5) or L[y+1][x+4] != c or L[y+4][x+1] != c or L[y+4][x+4] != c: return None
        return c

    top = sorted([h for h in heads if h[3] < 53], key=lambda h: (h[0], h[1], h[4]))
    bot = [h for h in heads if h[3] >= 53]
    if not top: return None
    hpos = set((h[2], h[3]) for h in top)
    rots = [h[4] for h in top]
    hs = tuple((h[2], h[3], walk(h[2], h[3], h[4])) for h in top)
    pairs = []
    for i, h in enumerate(top):
        for b in bot:
            if b[0] == h[0]:
                n = walk(b[2], b[3], b[4]); d = DIR[b[4]]
                T = []
                for k in range(1, n):
                    c = blk(b[2] + k*6*d[0], b[3] + k*6*d[1])
                    if c is not None: T.append(c)
                pairs.append((i, T)); break
    blocks = []; walls = set()
    for y in range(by0, by1 - 5, 6):
        for x in range(bx0, bx1 - 5, 6):
            if (x, y) in hpos: continue
            if L[y][x] == 5 and L[y+1][x+1] == 5 and L[y+2][x+2] == 5: walls.add((x, y)); continue
            c = blk(x, y)
            if c is not None: blocks.append((x, y, c))
    R = np.isin(o, [2, 3]); R[53:] = False; R[by0:by1, bx0:bx1] = False
    for h in top: R[h[3]:h[3]+6, h[2]:h[2]+6] = False
    R0 = R.copy()
    for h in top:
        x, y = h[2], h[3]
        for py in range(y, y+6):
            for px in range(x, x+6):
                up = y-1 >= 0 and R0[y-1, px]; dn = y+6 < 64 and R0[y+6, px]
                lf = x-1 >= 0 and R0[py, x-1]; rt = x+6 < 64 and R0[py, x+6]
                if up or dn or lf or rt: R[py, px] = True
    clickable = [p[0] for p in pairs]
    act_i = None
    for i, h in enumerate(top):
        if h[5] == 0 and i in clickable: act_i = i
    if act_i is None:
        if not clickable: return None
        act_i = clickable[0]
    G = dict(rots=rots, pairs=pairs, walls=walls, R=R, b=(bx0, bx1, by0, by1), click=clickable)
    return G, (act_i, hs, tuple(sorted(blocks)))

def sim(G, s, a):
    if (s, a) in G['bad']: return None
    ai, hs, bl = s
    if a[0] == 'C': return (a[1], hs, bl)
    mx, my = MV[a]; dx, dy = mx*6, my*6
    rots = G['rots']; walls = G['walls']; bx0, bx1, by0, by1 = G['b']
    def oob(x, y): return x < bx0 or x+6 > bx1 or y < by0 or y+6 > by1 or (x, y) in walls
    def sp(h, i):
        x, y, n = hs[h]; d = DIR[rots[h]]; return (x + i*6*d[0], y + i*6*d[1])
    blockat = {(b[0], b[1]): j for j, b in enumerate(bl)}
    segat = {}
    for h, (x, y, n) in enumerate(hs):
        hz = rots[h] in (0, 180)
        for i in range(n): segat.setdefault(sp(h, i), []).append((h, i, hz))
    moved = set()
    def push(obj, exc):
        if obj in moved: return True
        if obj[0] == 's':
            _, hh, i = obj; px, py = sp(hh, i); r = rots[hh]
            if oob(px+dx, py+dy):
                if not (DIR[r] == (-mx, -my) or oob(px, py)): return False
            for ox, oy in ((0, 0), (dx, dy)):
                j = blockat.get((px+ox, py+oy))
                if j is not None:
                    if push(('b', j), obj): moved.add(('b', j))
                    elif (DIR[r][0] == 0) != (mx == 0): return False
        else:
            j = obj[1]; px, py = bl[j][0], bl[j][1]
            if oob(px+dx, py+dy): return False
            hz = mx != 0
            for ox, oy in ((0, 0), (dx, dy)):
                for sg in segat.get((px+ox, py+oy), ()):
                    if exc is not None and sg[0] == exc[1] and sg[1] == exc[2]: continue
                    if sg[2] == (not hz): return False
            w = blockat.get((px+dx, py+dy))
            if w is not None and not push(('b', w), None): return False
        moved.add(obj); return True
    h = ai; x, y, n = hs[h]; bd = DIR[rots[h]]; nh = list(hs)
    if (mx, my) == bd:
        tx, ty = sp(h, n-1)
        if oob(tx+dx, ty+dy): return None
        for i in range(n): push(('s', h, i), None)
        nh[h] = (x, y, n+1)
    elif (mx, my) == (-bd[0], -bd[1]):
        if n == 1: return None
        for i in range(1, n): push(('s', h, i), None)
        nh[h] = (x, y, n-1)
    else:
        px, py = x+2+3*mx, y+2+3*my
        if not (0 <= px < 64 and 0 <= py < 64 and G['R'][py, px]): return None
        for i in range(n):
            if not push(('s', h, i), None): return None
        nh[h] = (x+dx, y+dy, n)
    nb = [(b[0]+dx, b[1]+dy, b[2]) if ('b', j) in moved else b for j, b in enumerate(bl)]
    return (ai, tuple(nh), tuple(sorted(nb)))

def hval(G, s):
    ai, hs, bl = s
    col = {(b[0], b[1]): b[2] for b in bl}; tot = 0; dd = 0
    for h, T in G['pairs']:
        x, y, n = hs[h]; d = DIR[G['rots'][h]]
        Lc = [col[(x+i*6*d[0], y+i*6*d[1])] for i in range(n) if (x+i*6*d[0], y+i*6*d[1]) in col]
        for i, t in enumerate(T):
            if i >= len(Lc) or Lc[i] != t:
                tot += 1; bst = 20
                for b in bl:
                    if b[2] == t:
                        dist = abs(b[1]-y)//6 if d[1] == 0 else abs(b[0]-x)//6
                        if dist < bst: bst = dist
                dd += bst
    return tot, dd

def search(G, st, t0, mem):
    S0 = mem.get('S')
    if S0 is None or S0['root'] != st or S0['nb'] != len(G['bad']):
        h0, d0 = hval(G, st)
        S0 = {'root': st, 'nb': len(G['bad']), 'par': {st: None}, 'heap': [(2*h0, h0, d0, 0, 0, st)],
              'cnt': 0, 'best': (h0, 0, st), 'goal': (st if h0 == 0 else None)}
        mem['S'] = S0
    par = S0['par']; heap = S0['heap']
    acts = ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4']
    while heap and S0['goal'] is None and time.time() - t0 < 7.5:
        f, hh, dd, g, _, s = heapq.heappop(heap)
        al = acts + [('C', j) for j in G['click'] if j != s[0]]
        for a in al:
            ns = sim(G, s, a)
            if ns is None or ns in par: continue
            par[ns] = (s, a)
            nh, nd = hval(G, ns); S0['cnt'] += 1
            if nh == 0: S0['goal'] = ns; break
            if (nh, g+1) < S0['best'][:2]: S0['best'] = (nh, g+1, ns)
            heapq.heappush(heap, (g+1+2*nh, nh, nd, g+1, S0['cnt'], ns))
    return S0

def getpath(par, tgt):
    path = []
    while par.get(tgt) is not None:
        p, a = par[tgt]; path.append((a, tgt)); tgt = p
    return path[::-1]

def out(a, s):
    if isinstance(a, tuple):
        x, y, n = s[1][a[1]]; return ('ACTION6', y+2, x+2)
    return (a, None, None)

def act(obs, available, memory):
    t0 = time.time()
    o = np.array(obs, dtype=int)
    try:
        P = parse(o)
    except Exception:
        P = None
    if P is None:
        return ('ACTION4', None, None)
    G, st = P
    bad = memory.setdefault('bad', set())
    prev = memory.get('prev')
    if prev is not None and prev[0] == st:
        bad.add((st, prev[1]))
    memory['prev'] = None
    G['bad'] = bad
    plan = memory.get('plan')
    if plan and memory.get('exp') == st:
        a, ns = plan.pop(0); memory['exp'] = ns
        memory['prev'] = (st, a)
        return out(a, st)
    path = []
    try:
        S0 = search(G, st, t0, memory)
        if S0['goal'] is not None:
            path = getpath(S0['par'], S0['goal'])
        else:
            th = memory.get('think', 0)
            if th < MAXTHINK and S0['heap']:
                memory['think'] = th + 1
                memory['plan'] = None
                return ('ACTION6', 0, 0)
            path = getpath(S0['par'], S0['best'][2])
    except Exception:
        path = []
    memory['S'] = None
    if not path:
        memory['plan'] = None
        opts = [a for a in ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'] if (st, a) not in bad]
        a = random.choice(opts or ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'])
        memory['prev'] = (st, a); memory['exp'] = None
        return (a, None, None)
    a, ns = path[0]
    memory['plan'] = path[1:]; memory['exp'] = ns
    memory['prev'] = (st, a)
    return out(a, st)
