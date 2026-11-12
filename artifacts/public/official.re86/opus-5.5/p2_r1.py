import numpy as np
from collections import deque

H, W = 63, 64
MOVES = [("ACTION1", 0, -3), ("ACTION2", 0, 3), ("ACTION3", -3, 0), ("ACTION4", 3, 0)]

def comps(mask, conn8=False):
    seen = np.zeros(mask.shape, bool); out = []
    nb = [(-1,0),(1,0),(0,-1),(0,1)] + ([(-1,-1),(-1,1),(1,-1),(1,1)] if conn8 else [])
    for r, c in np.argwhere(mask):
        if seen[r, c]: continue
        seen[r, c] = True; q = deque([(int(r), int(c))]); cs = []
        while q:
            a, b = q.popleft(); cs.append((a, b))
            for dr, dc in nb:
                x, y = a+dr, b+dc
                if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y] and not seen[x, y]:
                    seen[x, y] = True; q.append((x, y))
        out.append(cs)
    return out

def intact(g, p):
    r, c = p
    return (g[r-1:r+2, c-1:c+2] == 4).sum() == 8

def act(obs, available, memory):
    g = np.array(obs)[:H]
    T = memory.setdefault("T", {})
    for r in range(1, H-1):
        for c in range(1, W-1):
            if g[r, c] != 4 and (r, c) not in T and (g[r-1:r+2, c-1:c+2] == 4).sum() == 8:
                T[(r, c)] = int(g[r, c])
    boxes = []; boxpix = set()
    for cs in comps(g == 2):
        rs = [p[0] for p in cs]; cl = [p[1] for p in cs]
        r0, r1, c0, c1 = min(rs), max(rs), min(cl), max(cl)
        if r1-r0 >= 3 and c1-c0 >= 3:
            pix = set((a, b) for a in range(r0, r1+1) for b in range(c0, c1+1))
            boxes.append((int(g[r0+1, c0+1]), pix)); boxpix |= pix
    zs = np.argwhere(g == 0)
    if len(zs) == 0:
        return ("ACTION5", None, None)
    zr, zc = int(zs[0][0]), int(zs[0][1])
    col = None; S = set()
    for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)):
        r, c = zr+dr, zc+dc
        if 0 <= r < H and 0 <= c < W and g[r, c] not in (0, 5, 4, 2) and (r, c) not in boxpix:
            col = int(g[r, c]); break
    if col is not None:
        for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)):
            r, c = zr+dr, zc+dc
            while 0 <= r < H and 0 <= c < W and g[r, c] == col:
                S.add((r, c)); r += dr; c += dc
    else:
        best = None
        for r, c in np.argwhere((g != 5) & (g != 4) & (g != 0) & (g != 2)):
            r, c = int(r), int(c)
            if (r, c) in boxpix or (r, c) in T: continue
            d = abs(r-zr)+abs(c-zc)
            if best is None or d < best[0]: best = (d, r, c)
        if best is None: return ("ACTION5", None, None)
        col = int(g[best[1], best[2]])
        for cs in comps(g == col, True):
            if (best[1], best[2]) in cs: S = set(cs); break
    Sall = S | {(zr, zc)}
    forb = set()
    for k, pix in boxes:
        if k != col: forb |= pix
    for r, c in np.argwhere(g == 1):
        if (int(r), int(c)) not in Sall: forb.add((int(r), int(c)))
    start = (0, 0); dist = {start: 0}; par = {}; q = deque([start])
    while q:
        ox, oy = q.popleft()
        for name, dx, dy in MOVES:
            nx, ny = ox+dx, oy+dy
            if (nx, ny) in dist: continue
            if not (0 <= zr+ny < 63 and 0 <= zc+nx < 64): continue
            if forb and any((r+ny, c+nx) in forb for r, c in Sall): continue
            dist[(nx, ny)] = dist[(ox, oy)]+1; par[(nx, ny)] = ((ox, oy), name); q.append((nx, ny))
    def uncovered(p):
        return p in Sall or intact(g, p)
    unc = [p for p, k in T.items() if k == col and uncovered(p)]
    goal = None
    if unc:
        bk = None
        for (ox, oy), d in dist.items():
            cnt = sum(1 for r, c in unc if (r-oy, c-ox) in S)
            if cnt == 0: continue
            key = (-cnt, d)
            if bk is None or key < bk: bk = key; goal = (ox, oy)
    if goal is None:
        need = set(k for p, k in T.items() if uncovered(p)) - {col}
        bk = None
        for k, pix in boxes:
            if k not in need: continue
            for (ox, oy), d in dist.items():
                if d == 0: continue
                if any((r+oy, c+ox) in pix for r, c in Sall):
                    if bk is None or d < bk: bk = d; goal = (ox, oy)
    sig = (zr, zc, col, len(S))
    cnts = memory.setdefault("cnt", {})
    cnts[sig] = cnts.get(sig, 0)+1
    if goal is None or goal == (0, 0) or cnts[sig] > 6:
        if cnts[sig] > 6: memory["cnt"] = {}
        return ("ACTION5", None, None)
    node = goal; name = None
    while node != start:
        node, name = par[node]
    return (name, None, None)
