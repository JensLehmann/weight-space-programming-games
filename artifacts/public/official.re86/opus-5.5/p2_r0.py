import numpy as np
from collections import Counter

def _targets(obs):
    T = []
    for r in range(1, 62):
        for c in range(1, 63):
            v = obs[r, c]
            if v in (4, 5):
                continue
            blk = obs[r-1:r+2, c-1:c+2].copy()
            blk[1, 1] = 4
            if np.all(blk == 4):
                T.append((r, c, int(v)))
    return T

def _pads(obs):
    P = []
    for r in range(0, 60):
        for c in range(0, 60):
            if obs[r, c] == 2 and obs[r, c+1] == 2 and obs[r+1, c] == 2:
                v = int(obs[r+1, c+1])
                if v in (2, 5, 4, 0):
                    continue
                if r > 0 and obs[r-1, c] == 2: continue
                if c > 0 and obs[r, c-1] == 2: continue
                w = 0
                while c+w < 64 and obs[r, c+w] == 2: w += 1
                P.append((r + w//2, c + w//2, v))
    return P

def act(obs, available, memory):
    obs = np.array(obs)
    if 'T' not in memory:
        memory['T'] = {}
        memory['bad'] = set()
        memory['last'] = None
        memory['n5'] = 0
        memory['vis'] = {}
    for r, c, v in _targets(obs):
        memory['T'][(r, c)] = v
    T = memory['T']
    zs = np.argwhere(obs[:63] == 0)
    if len(zs) == 0:
        memory['n5'] += 1
        memory['last'] = None
        return ("ACTION5", None, None)
    r0, c0 = map(int, zs[0])
    nb = obs[max(0, r0-1):r0+2, max(0, c0-1):c0+2].flatten()
    cnt = Counter(int(v) for v in nb if v not in (0, 4, 5))
    if not cnt:
        memory['n5'] += 1
        memory['last'] = None
        return ("ACTION5", None, None)
    col = cnt.most_common(1)[0][0]
    seen = {(r0, c0)}
    st = [(r0, c0)]
    while st:
        r, c = st.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                a, b = r+dr, c+dc
                if 0 <= a < 63 and 0 <= b < 64 and (a, b) not in seen and obs[a, b] == col:
                    seen.add((a, b)); st.append((a, b))
    mask = seen
    uncovered = [(r, c, v) for (r, c), v in T.items() if obs[r, c] != v or obs[r-1, c] == 4]
    tg = [(r, c) for r, c, v in uncovered if v == col]
    pos = (r0, c0)
    key_s = (pos, col, len(mask))
    memory['vis'][key_s] = memory['vis'].get(key_s, 0) + 1
    last = memory['last']
    if last is not None:
        if last[0] == pos:
            memory['bad'].add((pos, last[1]))
        elif memory['vis'][key_s] >= 3:
            memory['bad'].add((last[0], last[1]))
    best = None
    for dy in range(-63, 64, 3):
        if not (0 <= r0+dy < 63): continue
        for dx in range(-63, 64, 3):
            if not (0 <= c0+dx < 64): continue
            sc = sum(1 for (r, c) in tg if (r-dy, c-dx) in mask)
            key = (-sc, abs(dy)+abs(dx))
            if best is None or key < best[0]:
                best = (key, dy, dx)
    (nsc, _), dy, dx = best
    nshapes = max(2, len(set(v for _, _, v in uncovered)) + 2)
    if nsc == 0 or (dy == 0 and dx == 0 and memory['n5'] >= nshapes):
        want = set(v for _, _, v in uncovered) - {col}
        pads = [p for p in _pads(obs) if p[2] in want]
        if pads and (nsc == 0 and (memory['n5'] >= nshapes or not tg)):
            pr, pc, _ = min(pads, key=lambda p: abs(p[0]-r0)+abs(p[1]-c0))
            dy, dx = pr - r0, pc - c0
        elif nsc == 0:
            dy = dx = 0
    opts = []
    if abs(dy) >= abs(dx):
        if dy: opts.append("ACTION1" if dy < 0 else "ACTION2")
        if dx: opts.append("ACTION3" if dx < 0 else "ACTION4")
    else:
        if dx: opts.append("ACTION3" if dx < 0 else "ACTION4")
        if dy: opts.append("ACTION1" if dy < 0 else "ACTION2")
    if opts and memory['vis'][key_s] >= 3:
        for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if a not in opts:
                opts.append(a)
    for a in opts:
        if (pos, a) not in memory['bad']:
            memory['last'] = (pos, a)
            memory['n5'] = 0
            return (a, None, None)
    memory['last'] = None
    memory['n5'] += 1
    if memory['n5'] > 4 * nshapes:
        memory['n5'] = 0
        memory['bad'] = set()
        memory['vis'] = {}
        return ("RESET", None, None)
    return ("ACTION5", None, None)
