import numpy as np

def _legs(obs):
    H, W = obs.shape
    out = []
    F = (0, 3)
    for r in range(2, H - 2):
        for c in range(2, W - 2):
            v = int(obs[r, c])
            if v in (0, 3, 5, 2, 10, 1):
                continue
            if all(obs[r + dr, c + dc] in F for dr, dc in ((-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2))):
                out.append((r, c, v))
    return out

def _goal(obs, col):
    H, W = obs.shape
    for r in range(3, H - 3):
        for c in range(3, W - 3):
            if (obs[r-1, c-3] == col and obs[r+1, c-3] == col and obs[r-1, c+3] == col
                    and obs[r+1, c+3] == col and obs[r-3, c-1] == col and obs[r-3, c+1] == col
                    and obs[r, c] != col):
                return (r, c)
    return None

OFFS = [(0, -4), (0, 4), (-3, 0), (3, 0), (-3, -3), (3, 3), (-3, 3), (3, -3)]

def act(obs, available, memory):
    q = memory.get('q', [])
    if not q:
        legs = _legs(obs)
        groups = {}
        for r, c, v in legs:
            groups.setdefault(v, []).append((r, c))
        for v, ls in groups.items():
            g = _goal(obs, v)
            if g is None:
                continue
            br = sum(p[0] for p in ls) / len(ls)
            bc = sum(p[1] for p in ls) / len(ls)
            if abs(g[0] - br) <= 2 and abs(g[1] - bc) <= 2:
                continue
            ls = sorted(ls, key=lambda p: (p[0]-g[0])**2 + (p[1]-g[1])**2)
            for i, (r, c) in enumerate(ls):
                o = OFFS[i % len(OFFS)]
                nr = min(63, max(0, g[0] + o[0])); nc = min(63, max(1, g[1] + o[1]))
                q.append((r, c)); q.append((nr, nc))
            break
        if not q:
            if legs:
                q.append((legs[0][0], legs[0][1]))
            else:
                q.append((32, 32))
    r, c = q.pop(0)
    memory['q'] = q
    return ('ACTION6', int(r), int(c))
