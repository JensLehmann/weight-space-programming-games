import numpy as np, random

def _buttons(g):
    m = np.zeros((64, 64), bool); m[:62] = g[:62] == 2
    seen = np.zeros_like(m); C = []
    for r, c in np.argwhere(m):
        if seen[r, c]: continue
        st = [(r, c)]; seen[r, c] = True; ys = []; xs = []
        while st:
            y, x = st.pop(); ys.append(y); xs.append(x)
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < 64 and 0 <= xx < 64 and m[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; st.append((yy, xx))
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        h = y1-y0+1; w = x1-x0+1
        if h < 3 or w < 3: continue
        cy = (y0+y1)//2; cx = (x0+x1)//2
        if h == w:
            C.append([cy, cx, 'rot', -1])
        else:
            if w > h: ext = (cy, x1-2); shr = (cy, x0+2)
            else: ext = (y1-2, cx); shr = (y0+2, cx)
            i = len(C)
            C.append([ext[0], ext[1], 'ext', i+1])
            C.append([shr[0], shr[1], 'ext', i])
    return C

def _metric(g):
    m = np.zeros((64, 64), bool); m[:62] = g[:62] == 13
    p = np.pad(m, 1)
    up = p[:-2,1:-1]; dn = p[2:,1:-1]; lf = p[1:-1,:-2]; rt = p[1:-1,2:]
    cen = up & dn & lf & rt
    n8 = up | dn | lf | rt | p[:-2,:-2] | p[:-2,2:] | p[2:,:-2] | p[2:,2:]
    dots = np.argwhere(m & ~n8)
    tot = 0
    for r, c in np.argwhere(cen):
        if m[r, c]: continue
        if len(dots) == 0: tot += 100; continue
        tot += int(np.min(np.abs(dots[:,0]-r) + np.abs(dots[:,1]-c)))
    return tot

def _click(b):
    return ('ACTION6', int(b[0]), int(b[1]))

def act(obs, available, memory):
    g = np.array(obs)
    if 'cands' not in memory:
        memory['cands'] = _buttons(g); memory['idx'] = 0; memory['trial'] = None
    C = memory['cands']
    if not C:
        return ('ACTION6', 32, 32)
    m = _metric(g)
    t = memory['trial']
    if t is not None:
        a, m0, f0, rc = t
        memory['trial'] = None
        changed = not np.array_equal(f0, g[:62])
        if changed and m < m0:
            memory['idx'] = 0
        elif changed:
            if C[a][2] == 'rot':
                if rc < 3:
                    memory['trial'] = (a, m0, f0, rc+1)
                    return _click(C[a])
                memory['idx'] += 1
                return _click(C[a])
            memory['idx'] += 1
            return _click(C[C[a][3]])
        else:
            memory['idx'] += 1
    i = memory['idx']
    if i >= len(C):
        memory['idx'] = 0
        return _click(C[random.randrange(len(C))])
    memory['trial'] = (i, m, g[:62].copy(), 1)
    return _click(C[i])
