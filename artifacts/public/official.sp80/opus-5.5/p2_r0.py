import numpy as np, time, random, math
from collections import deque

UMAP = {1:{1:4,2:3,3:1,4:2},2:{1:2,2:1,3:4,4:3},3:{1:3,2:4,3:2,4:1}}
DIRA = {(0,-1):1,(0,1):2,(-1,0):3,(1,0):4}
NONE, BORD, WAT = 0, 1, 2

def detect_k(obs):
    edges = {0: obs[0,:], 1: obs[:,0], 2: obs[63,:], 3: obs[:,63]}
    best, bs = 0, -1
    for k, e in edges.items():
        if np.all((e == 0) | (e == 14)):
            sc = int(np.sum(e == 14))
            if sc > bs: best, bs = k, sc
    return best

def detect_grid(o):
    for N in range(6, 65):
        s = 64 // N
        off = (64 - N*s) // 2
        mask = np.ones((64,64), bool); mask[off:off+N*s, off:off+N*s] = False; mask[0,:] = False
        if np.any(o[mask] != 1): continue
        reg = o[off:off+N*s, off:off+N*s]
        g = reg[s-1::s, s//2::s]
        rep = np.repeat(np.repeat(g, s, 0), s, 1)
        d = rep != reg
        if off == 0: d[0,:] = False
        if not np.any(d): return N, s, off, g.astype(int)
    return None

def comps(g, N, colors):
    seen = set(); out = []
    for y in range(N):
        for x in range(N):
            if g[y][x] in colors and (x,y) not in seen:
                c = g[y][x]; q = [(x,y)]; seen.add((x,y)); cells = []
                while q:
                    cx, cy = q.pop(); cells.append((cx,cy))
                    for ddx, ddy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = cx+ddx, cy+ddy
                        if 0<=nx<N and 0<=ny<N and (nx,ny) not in seen and g[ny][nx]==c:
                            seen.add((nx,ny)); q.append((nx,ny))
                out.append((c, cells))
    return out

def bbox(cells):
    xs=[c[0] for c in cells]; ys=[c[1] for c in cells]
    return min(xs), min(ys), max(xs)-min(xs)+1, max(ys)-min(ys)+1

def plan(obs):
    k = detect_k(obs)
    o = np.rot90(obs, -k)
    r = detect_grid(o)
    if r is None: return [('ACTION5', None, None)]
    N, s, off, g = r
    g = g.tolist()
    static = [[0]*N for _ in range(N)]
    types = {BORD:'o', WAT:'w'}
    nid = [3]
    def newid(t):
        i = nid[0]; nid[0] += 1; types[i] = t; return i
    cups = []; pieces = []; srcs = []; taps = []
    for c, cells in comps(g, N, set(range(16)) - {12}):
        if c in (8, 9, 15):
            xs = set(p[0] for p in cells); ys = set(p[1] for p in cells)
            kind = 'b' if (len(xs) == 1 or len(ys) == 1) and c != 15 else 'd'
            pieces.append({'cells': cells, 'sel': c == 9, 'kind': kind})
            continue
        if c == 11:
            i = newid('c'); cups.append((i, bbox(cells)))
        elif c == 1: i = newid('f')
        elif c == 6:
            i = WAT; srcs.extend(cells)
        elif c == 4:
            i = BORD; taps.extend(cells)
        else: i = BORD
        for (x, y) in cells: static[y][x] = i
    if not pieces: return [('ACTION5', None, None)]
    cupids = [c[0] for c in cups]
    pids = [newid(p['kind']) for p in pieces]

    def legal(p, ox, oy):
        cells = p['cells']
        for (x, y) in cells:
            nx, ny = x+ox, y+oy
            if not (0 <= nx < N and 0 <= ny < N) or static[ny][nx] != 0: return False
        bx, by, bw, bh = p['bb']; x = bx+ox; y = by+oy
        if y < 3: return False
        for _, (rx, ry, rw, rh) in cups:
            if x < rx+rw+1 and x+bw > rx-1 and y < ry+rh+1 and y+bh > ry-1: return False
        return True
    for p in pieces:
        p['bb'] = bbox(p['cells'])
        par = {(0,0): None}; dist = {(0,0): 0}; q = deque([(0,0)])
        while q:
            cur = q.popleft()
            for d in ((0,-1),(0,1),(-1,0),(1,0)):
                n = (cur[0]+d[0], cur[1]+d[1])
                if n not in dist and legal(p, n[0], n[1]):
                    dist[n] = dist[cur]+1; par[n] = (cur, d); q.append(n)
        p['dist'] = dist; p['par'] = par; p['keys'] = list(dist.keys())

    def sim(pos):
        gg = [row[:] for row in static]
        for i, p in enumerate(pieces):
            ox, oy = pos[i]
            for (x, y) in p['cells']: gg[y+oy][x+ox] = pids[i]
        def get(x, y):
            if 0 <= x < N and 0 <= y < N: return gg[y][x]
            return BORD
        def put(x, y):
            if 0 <= x < N and 0 <= y < N and gg[y][x] == 0: gg[y][x] = WAT
        parts = [(x, y, 0, 1) for (x, y) in srcs]
        for (tx, ty) in taps:
            if get(tx, ty+1) == 0:
                put(tx, ty+1); parts.append((tx, ty+1, 0, 1))
        filled = set(); fails = 0; it = 0
        while parts:
            it += 1
            if it > 400: return filled, fails+5
            new = []
            for (x, y, dx, dy) in parts:
                if dy != 0: o1 = (-1, 0); o2 = (1, 0)
                else: o1 = (0, -1); o2 = (0, 1)
                t = get(x+dx, y+dy)
                if t == 0:
                    put(x+dx, y+dy); new.append((x+dx, y+dy, dx, dy)); continue
                T = types[t]
                if T == 'w':
                    new.append((x+dx, y+dy, dx, dy)); continue
                if T == 'b':
                    for (a, b) in (o1, o2):
                        if get(x+a, y+b) == 0:
                            put(x+a, y+b); new.append((x+a, y+b, dx, dy))
                    continue
                if T == 'c':
                    if get(x+o1[0], y+o1[1]) == t and get(x+o2[0], y+o2[1]) == t:
                        filled.add(t); continue
                    for (a, b) in (o1, o2):
                        if get(x+a, y+b) == 0:
                            put(x+a, y+b); new.append((x+a, y+b, dx, dy))
                    continue
                if T == 'd':
                    A = get(x+o1[0], y+o1[1]); B = get(x+o2[0], y+o2[1])
                    if A == t and B == 0:
                        put(x+dy, y-dx); new.append((x+dy, y-dx, dy, -dx))
                    if B == t and A == 0:
                        put(x-dy, y+dx); new.append((x-dy, y+dx, -dy, dx))
                    else:
                        for (a, b) in (o1, o2):
                            if get(x+a, y+b) == 0:
                                put(x+a, y+b); new.append((x+a, y+b, dx, dy))
                    continue
                if T == 'f':
                    fails += 1; continue
            parts = new
        return filled, fails

    sel = next((i for i, p in enumerate(pieces) if p['sel']), -1)
    def cost(pos):
        moved = [i for i in range(len(pieces)) if pos[i] != (0, 0)]
        c = sum(pieces[i]['dist'][pos[i]] for i in moved) + len(moved) + 1
        if sel in moved: c -= 1
        return c
    def ev(pos):
        f, fl = sim(pos)
        return 10*sum(1 for c in cupids if c not in f) + fl
    t0 = time.time(); TL = 6.0
    P = len(pieces)
    cur = [(0, 0)]*P; cb = ev(cur)
    best = (cb, cost(cur), list(cur))
    def upd(pos, b):
        nonlocal best
        c = cost(pos)
        if (b, c) < best[:2]: best = (b, c, list(pos))
    if cb > 0:
        for i in range(P):
            for key in pieces[i]['keys']:
                if time.time()-t0 > TL*0.4: break
                pos = [(0, 0)]*P; pos[i] = key
                upd(pos, ev(pos))
        cur = list(best[2]); cb = best[0]; T = 3.0; found_t = None
        if cb == 0: found_t = time.time()
        rnd = random.Random(1)
        while time.time()-t0 < TL:
            if found_t and time.time()-found_t > 1.0: break
            i = rnd.randrange(P); p = pieces[i]; new = list(cur)
            u = rnd.random()
            if u < 0.15: new[i] = (0, 0)
            elif u < 0.55: new[i] = rnd.choice(p['keys'])
            else:
                c0 = cur[i]
                cand = (c0[0]+rnd.randint(-2, 2), c0[1]+rnd.randint(-2, 2))
                if cand not in p['dist']: continue
                new[i] = cand
            nb = ev(new)
            delta = (nb + 0.05*cost(new)) - (cb + 0.05*cost(cur))
            if delta <= 0 or rnd.random() < math.exp(-delta/max(T, 0.05)):
                cur, cb = new, nb
                upd(cur, cb)
                if cb == 0 and found_t is None: found_t = time.time()
            T = max(0.1, T*0.999)
            if rnd.random() < 0.002: cur = list(best[2]); cb = best[0]
    pos = best[2]

    def uact(n):
        if k == 0: return 'ACTION%d' % n
        inv = {v: u for u, v in UMAP[k].items()}
        return 'ACTION%d' % inv[n]
    def click(gx, gy):
        ox = off + gx*s + s//2; oy = off + gy*s + s//2
        if k == 0: ux, uy = ox, oy
        elif k == 1: ux, uy = oy, 63-ox
        elif k == 2: ux, uy = 63-ox, 63-oy
        else: ux, uy = 63-oy, ox
        return ('ACTION6', int(uy), int(ux))
    acts = []
    order = [i for i in range(P) if pos[i] != (0, 0)]
    order.sort(key=lambda i: 0 if i == sel else 1)
    curpos = [(0, 0)]*P
    for i in order:
        p = pieces[i]
        if i != sel or acts:
            boxes = []
            for j in range(P):
                if j == i: continue
                bx, by, bw, bh = pieces[j]['bb']
                boxes.append((bx+curpos[j][0], by+curpos[j][1], bw, bh))
            ch = p['cells'][0]
            for (x, y) in p['cells']:
                if not any(bx <= x < bx+bw and by <= y < by+bh for (bx, by, bw, bh) in boxes):
                    ch = (x, y); break
            acts.append(click(ch[0], ch[1]))
        path = []; n = pos[i]
        while p['par'][n] is not None:
            prev, d = p['par'][n]; path.append(d); n = prev
        for d in reversed(path): acts.append((uact(DIRA[d]), None, None))
        curpos[i] = pos[i]
    acts.append(('ACTION5', None, None))
    return acts

def act(obs, available, memory):
    q = memory.get('plan')
    if not q:
        try:
            q = plan(np.array(obs, dtype=int))
        except Exception:
            q = [('ACTION5', None, None)]
        memory['plan'] = q
    a = q.pop(0)
    return a
