import numpy as np
import heapq

POS = [("h",25,24,180),("d",33,21,0),("h",38,32,270),("d",33,40,90),
       ("h",25,45,0),("d",14,40,180),("h",17,32,90),("d",14,21,270)]
COORD = {0:(0,1),1:(0,2),2:(1,2),3:(2,2),4:(2,1),5:(2,0),6:(1,0),7:(0,0)}
SMALL = {0:((4,-6),180),2:((10,4),270),4:((4,10),0),6:((-6,4),90)}

def _rot(T, rot):
    k = ((-rot) % 360) // 90
    return np.rot90(T, k) if k else T

def _htpl():
    H = np.full((9,14),15); H[0,1:13] = -1; H[:,0] = 2; H[:,13] = 2; H[8,:] = 2
    return H

def _dtpl():
    D = np.full((17,17),-1)
    D[0,6] = 2; D[16,10] = 2
    for r in range(1,7):
        D[r,6-r] = D[r,7-r] = 2; D[r,8-r:5+r] = 15; D[r,5+r] = D[r,6+r] = 2
    for r in range(7,11):
        D[r,r-5:r+5] = 15; D[r,r+5] = D[r,r+6] = 2
    for r in range(11,16):
        D[r,r-5:25-r] = 15; D[r,25-r] = D[r,26-r] = 2
    return D

def _stpl():
    return np.array([[2,15,15,15,15,2]]*3+[[2,2,0,0,2,2],[-1,2,2,2,2,-1]])

def detect_basket(obs):
    best = (-1, 0, 15)
    H, D = _htpl(), _dtpl()
    for i,(k,x,y,r) in enumerate(POS):
        T = _rot(H if k == "h" else D, r)
        h, w = T.shape
        reg = obs[y:y+h, x:x+w]
        if reg.shape != T.shape: continue
        m2 = T == 2
        sc = np.mean(reg[m2] == 2)
        if sc > best[0]:
            vals = reg[T == 15].astype(int)
            col = int(np.bincount(vals, minlength=16).argmax()) if len(vals) else 15
            best = (sc, i, col)
    return best[1], best[2]

def small_info(p):
    (ox,oy), r = SMALL[p]
    _, bx, by, _ = POS[p]
    T = _rot(_stpl(), r)
    return bx+ox, by+oy, T

def small_present(obs, p):
    x, y, T = small_info(p)
    h, w = T.shape
    reg = obs[y:y+h, x:x+w]
    if reg.shape != T.shape: return False
    return np.mean(reg[T == 2] == 2) > 0.9

def find_palette(obs):
    pal = {}
    for y in range(0, 12):
        for x in range(18, 60):
            b = obs[y:y+5, x:x+5]
            if b.shape != (5,5): continue
            if np.all(b[0,:] == 4) and np.all(b[4,:] == 4) and np.all(b[:,0] == 4) and np.all(b[:,4] == 4):
                inner = b[1:4,1:4]
                c = int(inner[1,1])
                if c != 4 and np.all(inner == c) and c not in pal:
                    pal[c] = (y+2, x+2)
    return pal

def build_ops(use_small):
    ops = []
    def mk(m):
        v = 0
        for i in range(10):
            for j in range(10):
                if m[i,j]: v |= 1 << (i*10+j)
        return v
    for p in range(8):
        m = np.zeros((10,10), bool)
        if p == 0: m[0:5,:] = True
        elif p == 4: m[5:10,:] = True
        elif p == 6: m[:,0:5] = True
        elif p == 2: m[:,5:10] = True
        elif p == 5:
            for i in range(10): m[i,0:i+1] = True
        elif p == 3:
            for i in range(10): m[i,9-i:10] = True
        elif p == 1:
            for i in range(10): m[i,i:10] = True
        elif p == 7:
            for i in range(10): m[i,0:10-i] = True
        ops.append((p, "big", mk(m)))
    if use_small:
        for p,(a,b,c,d) in {0:(0,3,3,7),4:(7,10,3,7),6:(3,7,0,3),2:(3,7,7,10)}.items():
            m = np.zeros((10,10), bool); m[a:b,c:d] = True
            ops.append((p, "small", mk(m)))
    return ops

def rdist(a, b):
    d = abs(a-b); return min(d, 8-d)

def plan(tgt, cur, pos0, col0, palette, use_small):
    ops = build_ops(use_small)
    nd = 0; mis = 0; tc = {}
    for i in range(10):
        for j in range(10):
            if i == j or i+j == 9: continue
            b = 1 << (i*10+j)
            nd |= b
            c = int(tgt[i,j]); tc[c] = tc.get(c,0) | b
            if cur[i,j] != tgt[i,j]: mis |= b
    if mis == 0: return None
    pal = set(palette) if palette else set(tc.keys())
    covc = {0: 0}
    def cov(S):
        if S in covc: return covc[S]
        v = 0
        for k in range(len(ops)):
            if S >> k & 1: v |= ops[k][2]
        covc[S] = v; return v
    def valid(o, cv):
        U = ops[o][2] & nd & ~cv
        if U == 0: return None
        for c, m in tc.items():
            if U & ~m == 0:
                return c if c in pal else None
        return None
    heap = []; cnt = 0; dist = {}; parent = {}
    for o in range(len(ops)):
        c = valid(o, 0)
        if c is not None:
            key = (1 << o, o, c)
            dist[key] = 1; parent[key] = None
            heapq.heappush(heap, (1, cnt, key, False)); cnt += 1
    while heap:
        d, _, key, end = heapq.heappop(heap)
        if end:
            return ops[key[1]], key[2]
        if dist.get(key, 1e9) < d: continue
        S, o, c = key
        cv = cov(S)
        if (nd & ~cv) & mis == 0:
            t = d + rdist(pos0, ops[o][0]) + (1 if c != col0 else 0)
            heapq.heappush(heap, (t, cnt, key, True)); cnt += 1
        for o2 in range(len(ops)):
            if S >> o2 & 1: continue
            c2 = valid(o2, cv)
            if c2 is None: continue
            nk = (S | (1 << o2), o2, c2)
            nd2 = d + 1 + rdist(ops[o2][0], ops[o][0]) + (1 if c2 != c else 0)
            if nd2 < dist.get(nk, 1e9):
                dist[nk] = nd2; parent[nk] = key
                heapq.heappush(heap, (nd2, cnt, nk, False)); cnt += 1
    return None

def act(obs, available, memory):
    obs = np.asarray(obs)
    tgt = obs[3:13, 3:13]
    cur = obs[34:44, 27:37]
    pos, col = detect_basket(obs)
    palette = find_palette(obs)
    if pos % 2 == 0:
        memory['small'] = bool(small_present(obs, pos))
    small = memory.get('small', None)
    res = plan(tgt, cur, pos, col, palette, bool(small))
    if res is None and small is None:
        res = plan(tgt, cur, pos, col, palette, True)
    if res is None:
        return ("ACTION5", None, None)
    (p, kind, _), c = res
    if pos != p:
        fwd = (p - pos) % 8
        nxt = (pos + 1) % 8 if fwd <= 4 else (pos - 1) % 8
        r0, c0 = COORD[pos]; r1, c1 = COORD[nxt]
        if r1 < r0: return ("ACTION1", None, None)
        if r1 > r0: return ("ACTION2", None, None)
        if c1 < c0: return ("ACTION3", None, None)
        return ("ACTION4", None, None)
    if col != c and c in palette:
        yy, xx = palette[c]
        return ("ACTION6", int(yy), int(xx))
    if kind == "big":
        return ("ACTION5", None, None)
    x, y, T = small_info(p)
    h, w = T.shape
    return ("ACTION6", int(y + h // 2), int(x + w // 2))
