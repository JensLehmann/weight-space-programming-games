import numpy as np
from collections import deque

def _blobs(obs):
    m = (obs == 9) | (obs == 10)
    m[:, 62:] = False; m[47:, :] = False
    seen = np.zeros_like(m); out = []
    for r in range(64):
        for c in range(62):
            if m[r, c] and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = 1; pts = []
                while q:
                    a, b = q.popleft(); pts.append((a, b))
                    for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                        x, y = a+da, b+db
                        if 0 <= x < 64 and 0 <= y < 62 and m[x, y] and not seen[x, y]:
                            seen[x, y] = 1; q.append((x, y))
                out.append(pts)
    return out

CROSS = {(0,1),(1,0),(1,2),(2,1)}
COL = {(0,1),(1,1),(2,1)}
MOVES = {"ACTION1": (-1,0), "ACTION2": (1,0), "ACTION3": (0,-1), "ACTION4": (0,1)}

def _plan(obs, pat, memory):
    diffs = []
    for r in range(3):
        for c in range(3):
            on = obs[50+5*r, 25+5*c] not in (2, 0)
            if on != ((r, c) in pat):
                diffs.append((r, c))
    if not diffs:
        return None
    memory["pend"] = diffs[1:]
    memory["blocked"] = set()
    r, c = diffs[0]
    return ("ACTION6", 50+5*r, 25+5*c)

def _search(free, sz, pos, gb, blocked):
    I = np.zeros((65, 65), int); I[1:,1:] = np.cumsum(np.cumsum(free, 0), 1)
    def ok(r, c):
        if r < 0 or c < 0 or r+sz > 64 or c+sz > 62: return False
        return I[r+sz, c+sz]-I[r, c+sz]-I[r+sz, c]+I[r, c] == sz*sz
    def ov(r, c):
        return r <= gb[2] and r+sz-1 >= gb[0] and c <= gb[3] and c+sz-1 >= gb[1]
    steps = [2] if sz <= 2 else [4, 2]
    par = {pos: None}; dq = deque([pos])
    while dq:
        p = dq.popleft()
        for a, (dr, dc) in MOVES.items():
            if p == pos and (pos, a) in blocked: continue
            for s in steps:
                n = (p[0]+s*dr, p[1]+s*dc)
                if ok(*n):
                    if ov(*n):
                        q = (p, a)
                        while par[q[0]] is not None:
                            q = par[q[0]]
                        return q[1]
                    if n not in par:
                        par[n] = (p, a); dq.append(n)
                    break
    return None

def act(obs, available, memory):
    obs = np.asarray(obs)
    if not memory.get("init"):
        memory["init"] = 1
        bl = _blobs(obs); bl.sort(key=len)
        if bl:
            G = np.array(bl[-1])
            memory["gbox"] = (G[:,0].min(), G[:,1].min(), G[:,0].max(), G[:,1].max())
        return ("ACTION6", 5, 45)
    pend = memory.get("pend")
    if pend:
        r, c = pend.pop(0)
        return ("ACTION6", 50+5*r, 25+5*c)
    gb = memory.get("gbox")
    if gb is None:
        return ("ACTION6", 5, 45)
    m = (obs == 9) | (obs == 10)
    m[:, 62:] = False; m[47:, :] = False
    m[gb[0]:gb[2]+1, gb[1]:gb[3]+1] = False
    P = np.argwhere(m)
    if len(P) == 0:
        return ("ACTION6", 5, 45)
    pr, pc = int(P[:,0].min()), int(P[:,1].min())
    sz = int(max(P[:,0].max()-pr+1, P[:,1].max()-pc+1))
    sz = 2 if sz <= 3 else 4
    pos = (pr, pc)
    free = ~np.isin(obs, [5, 4, 6, 12, 13, 11, 15])
    free[:, 62:] = False; free[47:, :] = False
    free[gb[0]:gb[2]+1, gb[1]:gb[3]+1] = True
    for a, b in P: free[a, b] = True
    blocked = memory.setdefault("blocked", set())
    last = memory.get("last")
    if last and memory.get("pos") == pos:
        blocked.add((pos, last))
    memory["pos"] = pos
    a = _search(free, sz, pos, gb, blocked)
    if a:
        memory["last"] = a
        return (a, None, None)
    memory["last"] = None
    alt = 4 if sz == 2 else 2
    if memory.get("tc", 0) < 2 and _search(free, alt, pos, gb, set()):
        memory["tc"] = memory.get("tc", 0) + 1
        r = _plan(obs, CROSS, memory)
        if r: return r
    tg = None
    for r in range(1, 46):
        for c in range(1, 61):
            if obs[r, c] in (13, 12) and obs[r-1, c] == 6 and obs[r, c-1] == 6:
                tg = (r, c); break
        if tg: break
    if tg is None:
        r = _plan(obs, COL, memory)
        return r if r else ("ACTION1", None, None)
    tr, tc = tg
    if pr <= tr and pr + sz > tr:
        a = "ACTION4" if tc > pc else "ACTION3"
    elif pc <= tc and pc + sz > tc:
        a = "ACTION2" if tr > pr else "ACTION1"
    else:
        a = None
    if a:
        if memory.get("faced") == a:
            memory["faced"] = None
            r = _plan(obs, COL, memory)
            if r: return r
        memory["faced"] = a
        return (a, None, None)
    memory["faced"] = None
    step = 2 if sz <= 2 else 4
    cand = []
    for a, (dr, dc) in MOVES.items():
        if (pos, a) in blocked: continue
        nr, nc = pr+step*dr, pc+step*dc
        cand.append((min(abs(nr-tr), abs(nc-tc)), a))
    if cand:
        cand.sort(); memory["last"] = cand[0][1]
        return (cand[0][1], None, None)
    blocked.clear()
    return ("ACTION1", None, None)
