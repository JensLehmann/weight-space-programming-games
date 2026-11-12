import numpy as np

def _handles(o):
    res = []
    for r in range(1, 63):
        for c in range(1, 63):
            v = o[r, c]
            if v in (0, 3, 5):
                continue
            if all(o[r+dr, c+dc] in (0, 3) for dr, dc in ((1,0),(-1,0),(0,1),(0,-1))):
                res.append((r, c, int(v)))
    return res

def _targets(o):
    res = []
    for r in range(3, 61):
        for c in range(3, 61):
            a = o[r-1, c-3]; b = o[r-1, c+3]
            if a == 5 or b == 5:
                continue
            if o[r+1, c-3] == a and o[r+1, c+3] == b and o[r-3, c-1] == a and o[r-3, c+1] == b and o[r, c] != a:
                res.append((r, c, {int(a), int(b)}))
    return res

def _safe(o, r, c, bad):
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if abs(dr) == 2 and abs(dc) == 2:
                continue
            y, x = r+dr, c+dc
            if not (0 <= y < 64 and 1 <= x < 64):
                return False
            if o[y, x] in bad:
                return False
    return True

def _far(p, q, d=3):
    return abs(p[0]-q[0]) >= d or abs(p[1]-q[1]) >= d

def _layout(o, hl, others, T, n, relax):
    offs = [(a, b) for a in range(-12, 13) for b in range(-12, 13)]
    if n > 1:
        offs = [d for d in offs if d != (0, 0)]
    offs.sort(key=lambda d: abs(d[0])+abs(d[1]))
    for d in offs:
        dests = []
        for i in range(n):
            k = i - (n-1)/2.0
            dests.append((int(round(T[0]+d[0]*k)), int(round(T[1]+d[1]*k))))
        dests[-1] = (n*T[0]-sum(p[0] for p in dests[:-1]), n*T[1]-sum(p[1] for p in dests[:-1]))
        ok = all(_far(dests[i], dests[j]) for i in range(n) for j in range(i+1, n))
        if not ok:
            continue
        if any(not (0 <= p[0] < 64 and 1 <= p[1] < 64) for p in dests):
            continue
        if relax < 2:
            if any(not _safe(o, p[0], p[1], (2,)) for p in dests):
                continue
            if any(not _far(p, q2, 5) for p in dests for q2 in others):
                continue
        if not all(_far(dests[j], hl[i]) for j in range(n) for i in range(j+1, n)):
            continue
        if relax < 2:
            pos = list(hl)
            for i in range(n):
                pos[i] = dests[i]
                if relax == 1 and i < n-1:
                    continue
                mr = sum(p[0] for p in pos)//n; mc = sum(p[1] for p in pos)//n
                if not _safe(o, mr, mc, (10,)):
                    ok = False; break
            if not ok:
                continue
        return dests
    return None

def _plan(o, targets):
    hs = _handles(o)
    q = []
    groups = {}
    for h in hs:
        groups.setdefault(h[2], []).append((h[0], h[1]))
    cur = [(h[0], h[1]) for h in hs]
    for v, hl in groups.items():
        cand = [t for t in targets if v in t[2]]
        if not cand:
            continue
        n = len(hl)
        br = sum(p[0] for p in hl)//n; bc = sum(p[1] for p in hl)//n
        t = min(cand, key=lambda t: abs(t[0]-br)+abs(t[1]-bc))
        T = (t[0], t[1]+1)
        if abs(br-T[0]) + abs(bc-T[1]) <= 1:
            continue
        others = [p for p in cur if p not in hl]
        best = None
        for relax in (0, 1, 2):
            best = _layout(o, hl, others, T, n, relax)
            if best:
                break
        if best is None:
            best = [(T[0], T[1] + 4*(i - (n-1)//2)) for i in range(n)]
        for i in range(n):
            q.append(hl[i]); q.append(best[i])
        cur = others + best
    return q

def act(obs, available, memory):
    o = np.array(obs)
    if "targets" not in memory:
        memory["targets"] = _targets(o)
        memory["q"] = []
    q = memory["q"]
    if not q:
        q.extend(_plan(o, memory["targets"]))
        if not q:
            hs = _handles(o)
            if hs:
                r, c, _ = hs[memory.get("fb", 0) % len(hs)]
                memory["fb"] = memory.get("fb", 0) + 1
                for dr, dc in ((0, 6), (6, 0), (0, -6), (-6, 0)):
                    if _safe(o, r+dr, c+dc, (2, 10)):
                        q.append((r+dr, c+dc)); break
                return ("ACTION6", int(r), int(c))
            return ("ACTION6", 32, 32)
    r, c = q.pop(0)
    return ("ACTION6", int(max(0, min(63, r))), int(max(1, min(63, c))))
