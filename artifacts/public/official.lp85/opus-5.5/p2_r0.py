import numpy as np
import time

def _grid(obs, s, gw, gh, px0, py0):
    G = np.full((gh, gw), -1, dtype=int)
    for gy in range(gh):
        r = py0 + gy * s + (s - 1)
        for gx in range(gw):
            c = px0 + gx * s + (s - 1)
            if 0 <= r < 64 and 0 <= c < 64:
                G[gy, gx] = obs[r, c]
    tiles = {}
    for y in range(0, gh - 1, 3):
        for x in range(0, gw - 1, 3):
            c = G[y, x]
            if c in (-1, 3, 4, 8, 14):
                continue
            if G[y, x + 1] == c and G[y + 1, x] == c and G[y + 1, x + 1] == c:
                tiles[(y, x)] = int(c)
    frames = []
    for y in range(2, gh - 3, 3):
        for x in range(2, gw - 3, 3):
            c = G[y, x]
            if c in (11, 12) and G[y, x + 3] == c and G[y + 3, x] == c and G[y + 3, x + 3] == c:
                frames.append(((y + 1, x + 1), int(c)))
    return G, tiles, frames

def _parse(obs):
    obs = np.asarray(obs)
    m = (obs == 4).copy(); m[0:2, :] = False; m[:, 0] = False
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    minx, maxx, miny, maxy = xs.min(), xs.max(), ys.min(), ys.max()
    cands = []; bestc = None
    for gw in range(1, 65):
        for gh in range(1, 65):
            s = min(64 // gw, 64 // gh)
            if s < 1:
                continue
            px0 = (64 - gw * s) // 2; py0 = (64 - gh * s) // 2
            px1 = px0 + gw * s - 1; py1 = py0 + gh * s - 1
            c = abs(max(px0, 1) - minx) + abs(px1 - maxx) + abs(max(py0, 2) - miny) + abs(py1 - maxy)
            if bestc is None or c < bestc:
                bestc = c; cands = [(s, gw, gh, px0, py0)]
            elif c == bestc:
                cands.append((s, gw, gh, px0, py0))
    best = None
    for cd in cands[:12]:
        G, tiles, frames = _grid(obs, *cd)
        sc = (len(tiles) + 3 * len(frames), cd[0])
        if best is None or sc > best[0]:
            best = (sc, cd, G, tiles, frames)
    _, (s, gw, gh, px0, py0), G, tiles, frames = best
    seen = np.zeros_like(G, dtype=bool)
    buttons = {}
    for y in range(gh):
        for x in range(gw):
            c = G[y, x]
            if c in (8, 14) and not seen[y, x]:
                stack = [(y, x)]; seen[y, x] = True; cells = []
                while stack:
                    cy, cx = stack.pop(); cells.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < gh and 0 <= nx < gw and not seen[ny, nx] and G[ny, nx] == c:
                            seen[ny, nx] = True; stack.append((ny, nx))
                my = sum(p[0] for p in cells) / len(cells); mx = sum(p[1] for p in cells) / len(cells)
                cy, cx = min(cells, key=lambda p: (p[0] - my) ** 2 + (p[1] - mx) ** 2)
                key = (int(c), min(p[0] for p in cells), min(p[1] for p in cells))
                buttons[key] = (py0 + cy * s + s // 2, px0 + cx * s + s // 2, my, mx)
    return tiles, frames, buttons

def _hungarian(a):
    n = len(a); INF = float('inf')
    u = [0] * (n + 1); v = [0] * (n + 1); p = [0] * (n + 1); way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0; minv = [INF] * (n + 1); used = [False] * (n + 1)
        while True:
            used[j0] = True; i0 = p[j0]; delta = INF; j1 = 0
            row = a[i0 - 1]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    ans = [0] * n
    for j in range(1, n + 1):
        ans[p[j] - 1] = j - 1
    return ans

def _solve(pos, olist):
    n = len(pos)
    if n == 0:
        return None
    B = np.array([o[0] for o in olist]); A = np.array([o[1] for o in olist])
    allowed = (A[:, None, :] == B[:, :, None]).all(0)
    P = np.array(pos)
    d = np.maximum(np.abs(P[:, None, 0] - P[None, :, 0]), np.abs(P[:, None, 1] - P[None, :, 1])) // 3
    cost = np.where(d == 0, 0, np.where(d == 1, 1, 2 + d))
    cost = np.where(allowed, cost, 10 ** 6)
    perm = _hungarian(cost.tolist())
    for i, j in enumerate(perm):
        if not allowed[i, j]:
            return None
    inv = [0] * n
    for i, j in enumerate(perm):
        inv[j] = i
    return np.array(inv)

def _pairs(buttons):
    Ls = [k for k in buttons if k[0] == 8]; Rs = [k for k in buttons if k[0] == 14]
    cand = []
    for l in Ls:
        for r in Rs:
            ly, lx = buttons[l][2], buttons[l][3]; ry, rx = buttons[r][2], buttons[r][3]
            if abs(ly - ry) <= 1 or abs(lx - rx) <= 1:
                cand.append((abs(ly - ry) + abs(lx - rx), l, r))
    cand.sort(); pr = {}
    for _, l, r in cand:
        if l not in pr and r not in pr:
            pr[l] = r; pr[r] = l
    return pr

def _bfs(start, models, gidx, gcol, t0):
    st = np.array(start, dtype=np.uint8)
    gidx = np.array(gidx); gcol = np.array(gcol, dtype=np.uint8)
    if (st[gidx] == gcol).all():
        return None
    parents = {st.tobytes(): None}
    frontier = st[None, :]; fb = [st.tobytes()]
    while len(frontier) and len(parents) < 400000 and time.time() - t0 < 7:
        nl = []; nb = []
        for key, inv in models:
            N = frontier[:, inv]
            ok = (N[:, gidx] == gcol).all(1)
            for r in range(len(N)):
                kb = N[r].tobytes()
                if kb in parents:
                    continue
                parents[kb] = (fb[r], key)
                if ok[r]:
                    first = key; cur = kb
                    while parents[cur] is not None:
                        first = parents[cur][1]; cur = parents[cur][0]
                    return first
                nl.append(N[r]); nb.append(kb)
        if not nl:
            break
        frontier = np.array(nl); fb = nb
    return None

def act(obs, available, memory):
    t0 = time.time()
    pr = _parse(obs)
    if pr is None or not pr[2]:
        return ('ACTION6', 32, 32)
    tiles, frames, buttons = pr
    sig = (tuple(sorted(buttons.keys())), tuple(frames))
    if memory.get('sig') != sig:
        memory.clear(); memory['sig'] = sig; memory['obs'] = {}
        memory['U'] = set(); memory['cnt'] = {}; memory['nochg'] = {}
    U = memory['U']
    U.update(tiles.keys())
    for p, c in frames:
        U.add(p)
    pend = memory.pop('pending', None)
    if pend is not None and pend[0] in buttons:
        before = dict(pend[1])
        memory['obs'].setdefault(pend[0], []).append((before, dict(tiles)))
        if before == tiles:
            memory['nochg'][pend[0]] = memory['nochg'].get(pend[0], 0) + 1
        else:
            memory['nochg'][pend[0]] = 0
    pos = sorted(U)
    def vec(d):
        return [d.get(p, 0) for p in pos]
    cols = vec(tiles)
    allobs = memory['obs']
    pairs = _pairs(buttons)
    models = []; known = set()
    for key in sorted(buttons.keys(), key=lambda k: (-k[0], k[1], k[2])):
        own = [(vec(b), vec(a)) for (b, a) in allobs.get(key, [])]
        po = [(vec(a), vec(b)) for (b, a) in allobs.get(pairs.get(key), [])]
        inv = None
        if own or po:
            inv = _solve(pos, own + po)
            if inv is None and own and po:
                inv = _solve(pos, own)
        if inv is not None:
            models.append((key, inv)); known.add(key)
    pidx = {p: i for i, p in enumerate(pos)}
    gidx = [pidx[p] for p, c in frames]; gcol = [c for p, c in frames]
    gset = set(gcol)
    red = [c if c in gset else 0 for c in cols]
    choice = None
    if models and gidx:
        choice = _bfs(red, models, gidx, gcol, t0)
    cnt = memory['cnt']
    if choice is not None and memory['nochg'].get(choice, 0) >= 3:
        choice = None
    if choice is None:
        unk = [k for k in buttons if k not in known]
        if unk:
            choice = min(unk, key=lambda k: (cnt.get(k, 0), -k[0], k[1], k[2]))
        else:
            choice = min(buttons.keys(), key=lambda k: (memory['nochg'].get(k, 0) >= 3, len(allobs.get(k, [])), cnt.get(k, 0), k))
    cnt[choice] = cnt.get(choice, 0) + 1
    memory['pending'] = (choice, list(tiles.items()))
    r, c = buttons[choice][0], buttons[choice][1]
    return ('ACTION6', int(r), int(c))
