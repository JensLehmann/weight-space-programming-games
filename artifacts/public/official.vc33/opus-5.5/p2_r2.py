import numpy as np
from collections import deque

def _comps(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask, bool); out = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                st = [(r, c)]; seen[r, c] = True; px = []
                while st:
                    y, x = st.pop(); px.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; st.append((ny, nx))
                out.append(px)
    return out

def _bbox(px):
    rs = [p[0] for p in px]; cs = [p[1] for p in px]
    return min(rs), max(rs), min(cs), max(cs)

def _center(px):
    rs = [p[0] for p in px]; cs = [p[1] for p in px]
    mr, mc = sum(rs)/len(rs), sum(cs)/len(cs)
    return min(px, key=lambda p: (p[0]-mr)**2 + (p[1]-mc)**2)

def act(obs, available, memory):
    F = np.array(obs, dtype=int).copy()
    F[0] = F[1]
    if 's' not in memory:
        memory['s'] = 2 if ((F[2:, 0::2] == F[2:, 1::2]).all() and (F[2::2] == F[3::2]).all()) else 1
    s = memory['s']
    if 'k' not in memory:
        cnt = {}
        ys, xs = np.where(F == 4)
        for y, x in zip(ys, xs):
            for d in ((1,0),(-1,0),(0,1),(0,-1)):
                ny, nx = y+d[0], x+d[1]
                if 0 <= ny < 64 and 0 <= nx < 64 and F[ny, nx] in (11, 14):
                    cnt[d] = cnt.get(d, 0) + 1
        g = max(cnt, key=cnt.get) if cnt else (1, 0)
        memory['k'] = {(1,0):0, (-1,0):2, (0,1):3, (0,-1):1}[g]
    k = memory['k']
    A = np.rot90(F, k)
    RI = np.rot90(np.indices((64, 64))[0], k)
    CI = np.rot90(np.indices((64, 64))[1], k)
    def orig(p):
        return int(RI[p[0], p[1]]), int(CI[p[0], p[1]])

    if 'cols' not in memory:
        cl = []
        for px in _comps((A == 0) | (A == 9)):
            r0, r1, c0, c1 = _bbox(px)
            if c1 - c0 + 1 <= s or len(px) < 4 * s * s: continue
            if not any(A[p] == 0 for p in px): continue
            cl.append((c0, c1, r1))
        # drained columns: buttons not covered by any column
        for px in _comps(A == 9):
            br0, br1, bc0, bc1 = _bbox(px)
            if any(c[0] <= bc0 and bc1 <= c[1] and c[2] >= br0 for c in cl): continue
            r = br1; a = bc0; b = bc1
            while a - 1 >= 0 and A[r, a-1] in (0, 3, 9): a -= 1
            while b + 1 < 64 and A[r, b+1] in (0, 3, 9): b += 1
            lw = a - 1 >= 0 and A[r, a-1] == 5
            rw = b + 1 < 64 and A[r, b+1] == 5
            if (lw or rw) and b - a > bc1 - bc0:
                cl.append((a, b, br1))
        memory['cols'] = cl
    cols = memory['cols']
    nc = len(cols)
    step = memory.get('step', 4 if s == 2 else 3)

    def top_of(c0, c1, r1):
        w = c1 - c0 + 1; r = r1; t = r1 + 1
        while r >= 1:
            row = A[r, c0:c1+1]
            if np.count_nonzero((row == 0) | (row == 9)) * 2 >= w:
                t = r; r -= 1
            else:
                break
        return t
    T0 = tuple(top_of(*c) for c in cols)
    r1s = [c[2] for c in cols]

    lim = []; fb = []
    for i, (c0, c1, r1) in enumerate(cols):
        w = c1 - c0 + 1; f = None
        for r in range(T0[i] - 1, 0, -1):
            if np.count_nonzero(A[r, c0:c1+1] == 5) * 2 >= w:
                f = r; break
        fb.append(f)
        tops = []
        for cc in (c0-1, c1+1):
            if 0 <= cc < 64:
                ww = np.where(A[:r1+1, cc] == 5)[0]
                if len(ww): tops.append(int(ww.min()))
        lim.append(max(tops) if tops else -1)

    def pick(cands):
        return min(cands)[-1] if cands else None

    actions = []
    for px in _comps(A == 9):
        br0, br1, bc0, bc1 = _bbox(px)
        o = pick([(c[2], i) for i, c in enumerate(cols)
                  if c[0] <= bc0 and bc1 <= c[1] and c[2] >= br0])
        if o is None: continue
        c0, c1, _ = cols[o]
        if bc0 - c0 <= c1 - bc1:
            cand = [(c0 - c[1], c[2], i) for i, c in enumerate(cols) if i != o and c[1] < c0 and c[2] >= br0]
        else:
            cand = [(c[0] - c1, c[2], i) for i, c in enumerate(cols) if i != o and c[0] > c1 and c[2] >= br0]
        n = pick(cand)
        if n is None: continue
        actions.append(('b', o, n, orig(_center(px))))
    for px in _comps((A == 1) | (A == 12)):
        gr0, gr1, gc0, gc1 = _bbox(px)
        L = pick([(c[2], i) for i, c in enumerate(cols) if 0 <= gc0 - c[1] - 1 <= 1 and c[2] > gr1])
        R = pick([(c[2], i) for i, c in enumerate(cols) if 0 <= c[0] - gc1 - 1 <= 1 and c[2] > gr1])
        if L is not None and R is not None:
            actions.append(('g', L, R, orig(_center(px)), gr1))
    objs = []; tgts = []
    for px in _comps((A == 4) | (A == 11) | (A == 14)):
        strip = [p for p in px if A[p] in (11, 14)]
        if not strip: continue
        color = int(A[strip[0]])
        rb = max(p[0] for p in strip)
        r0, r1, c0, c1 = _bbox(px)
        if any(A[p] == 4 for p in px):
            best = None
            for i, c in enumerate(cols):
                if c[0] <= c1 and c0 <= c[1]:
                    sc = abs(T0[i] - rb - 1)
                    if best is None or sc < best[0]: best = (sc, i)
            objs.append((color, best[1] if best else -1))
        else:
            al = set(i for i, c in enumerate(cols)
                     if 0 <= c0 - c[1] - 1 <= 1 or 0 <= c[0] - c1 - 1 <= 1)
            tgts.append((color, rb, al))
    O0 = tuple(o[1] for o in objs)
    key = (T0, O0)

    blocked = memory.setdefault('blocked', set())
    if 'pend' in memory:
        pkey, pclick, po, ptop = memory.pop('pend')
        if pkey == key:
            blocked.add((pkey, pclick))
        elif po is not None and po < nc:
            d = ptop - T0[po]
            if 0 < d <= 12: memory['step'] = d; step = d
    colors = [o[0] for o in objs]

    def win(T, O):
        for ci, col in zip(colors, O):
            if col < 0: return False
            if not any(tc == ci and col in al and T[col] - 1 == trb for (tc, trb, al) in tgts):
                return False
        return True

    def can_rise(T, O, o):
        if fb[o] is not None:
            m = 2 * step if o in O else 0
            return T[o] - (fb[o] + 1) > m
        return T[o] > lim[o]

    def apply(T, O, a):
        if a[0] == 'b':
            _, o, n, _c = a
            if T[n] <= r1s[n] and can_rise(T, O, o):
                T2 = list(T); T2[o] -= step; T2[n] = min(T2[n] + step, r1s[n] + 1)
                return tuple(T2), O
            return None
        _, L, R, _c, gb = a
        if T[L] == gb + 1 and T[R] == gb + 1:
            O2 = tuple(R if x == L else (L if x == R else x) for x in O)
            if O2 != O: return T, O2
        return None

    def go(a):
        memory['pend'] = (key, a[3], a[1] if a[0] == 'b' else None, T0[a[1]] if a[0] == 'b' else None)
        return ("ACTION6", int(a[3][0]), int(a[3][1]))

    if not actions:
        return ("ACTION6", 32, 32)
    first = None
    if objs and not win(T0, O0):
        start = (T0, O0)
        par = {start: None}
        dq = deque([start]); found = None
        while dq and len(par) < 150000:
            st = dq.popleft()
            for ai, a in enumerate(actions):
                if st == start and (key, a[3]) in blocked: continue
                nx = apply(st[0], st[1], a)
                if nx is None or nx in par: continue
                par[nx] = (st, ai)
                if win(*nx):
                    found = nx; break
                dq.append(nx)
            if found: break
        if found:
            cur = found
            while par[cur][0] != start:
                cur = par[cur][0]
            first = par[cur][1]
    if first is None:
        cand = [a for a in actions if (key, a[3]) not in blocked and apply(T0, O0, a) is not None]
        if not cand:
            cand = [a for a in actions if (key, a[3]) not in blocked] or actions
        i = memory.get('cyc', 0); memory['cyc'] = i + 1
        return go(cand[i % len(cand)])
    return go(actions[first])
