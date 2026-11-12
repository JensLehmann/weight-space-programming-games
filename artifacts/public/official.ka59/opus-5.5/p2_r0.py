import numpy as np, time, heapq
from collections import deque

CROSS = np.zeros((9, 9), bool); CROSS[3:6, :] = True; CROSS[:, 3:6] = True

def _comps(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; ys = []; xs = []
                while q:
                    y, x = q.popleft(); ys.append(y); xs.append(x)
                    for yy, xx in ((y+1, x), (y-1, x), (y, x+1), (y, x-1)):
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True; q.append((yy, xx))
                out.append((min(ys), min(xs), max(ys)-min(ys)+1, max(xs)-min(xs)+1))
    return out

def _detect(o):
    ok = (o == 14) | (o == 0)
    ok[63, :] = False
    used = np.zeros((64, 64), bool)
    boxes = []
    specs = [(6, 6, 2, 2, 2, 2), (6, 3, 2, 1, 2, 1), (3, 6, 1, 2, 1, 2), (3, 3, 1, 1, 1, 1)]
    for h, w, cr, cc, ch, cw in specs:
        for r in range(0, 63 - h + 1):
            for c in range(0, 64 - w + 1):
                if not ok[r, c] or used[r, c]:
                    continue
                if used[r:r+h, c:c+w].any():
                    continue
                cen = o[r+cr:r+cr+ch, c+cc:c+cc+cw]
                v = int(cen[0, 0])
                if v not in (0, 4, 5) or not (cen == v).all():
                    continue
                ring = ok[r:r+h, c:c+w].copy(); ring[cr:cr+ch, cc:cc+cw] = True
                if not ring.all():
                    continue
                if v != 0 and not (o[r:r+h, c:c+w] == 14).any():
                    continue
                used[r:r+h, c:c+w] = True
                boxes.append((r, c, h, w, v == 0))
    crosses = [(r, c) for r, c, h, w in _comps(o[:63] == 11) if h == 9 and w == 9]
    blocks = []
    for r, c, h, w in _comps((o[:63] == 13) | (o[:63] == 12)):
        if h != 6 or w != 6:
            continue
        sub = o[r:r+6, c:c+6] == 12
        rows = sub.all(1); cols = sub.all(0)
        rot = None; cnt = 0
        if sub.any() and not sub.all():
            if rows[0] and not rows[5]: rot, cnt = 0, int(rows.sum())
            elif rows[5] and not rows[0]: rot, cnt = 180, int(rows.sum())
            elif cols[5] and not cols[0]: rot, cnt = 90, int(cols.sum())
            elif cols[0] and not cols[5]: rot, cnt = 270, int(cols.sum())
        blocks.append((r, c, rot, cnt))
    rt = []; xt = []
    for r, c, h, w in _comps(o[:63] == 4):
        if h >= 5 and w >= 5:
            if o[r, c] == 4:
                rt.append((r+1, c+1, h-2, w-2))
            else:
                xt.append((r+1, c+1))
    return boxes, crosses, blocks, rt, xt

def _sig(kinds, dims, pos, sel, cnt):
    items = tuple(sorted((kinds[i], pos[i][0], pos[i][1], dims[i][0], dims[i][1]) for i in range(len(kinds))))
    return (items, (pos[sel][0], pos[sel][1]) if sel is not None else None, cnt)

def _search(kinds, dims, masks, pos0, sel0, cnt0, rots, rtargs, xtargs, pad, wall, deadline):
    n = len(kinds)
    boxidx = [i for i in range(n) if kinds[i] == 'B']
    crossidx = [i for i in range(n) if kinds[i] == 'X']
    blockidx = [i for i in range(n) if kinds[i] == 'R']
    cache = {}
    def shit(i, r, c, which):
        key = (i, r, c, which)
        v = cache.get(key)
        if v is None:
            h, w = dims[i]
            if r < 0 or c < 0 or r + h > 64 or c + w > 64:
                v = (which == 0)
            else:
                arr = pad if which == 0 else wall
                v = bool((arr[r:r+h, c:c+w] & masks[i]).any())
            cache[key] = v
        return v
    def ov(i, p, j, q):
        h1, w1 = dims[i]; h2, w2 = dims[j]
        if p[0] >= q[0] + h2 or q[0] >= p[0] + h1 or p[1] >= q[1] + w2 or q[1] >= p[1] + w1:
            return False
        if kinds[i] != 'X' and kinds[j] != 'X':
            return True
        r0 = max(p[0], q[0]); r1 = min(p[0]+h1, q[0]+h2); c0 = max(p[1], q[1]); c1 = min(p[1]+w1, q[1]+w2)
        a = masks[i][r0-p[0]:r1-p[0], c0-p[1]:c1-p[1]]; b = masks[j][r0-q[0]:r1-q[0], c0-q[1]:c1-q[1]]
        return bool((a & b).any())
    def zone_ov(z, j, q):
        zr, zc, zh, zw = z
        h2, w2 = dims[j]
        if zr >= q[0] + h2 or q[0] >= zr + zh or zc >= q[1] + w2 or q[1] >= zc + zw:
            return False
        if kinds[j] != 'X':
            return True
        r0 = max(zr, q[0]); r1 = min(zr+zh, q[0]+h2); c0 = max(zc, q[1]); c1 = min(zc+zw, q[1]+w2)
        return bool(masks[j][r0-q[0]:r1-q[0], c0-q[1]:c1-q[1]].any())
    def push(j, dr, dc, pos, depth=0):
        r, c = pos[j]; nr, nc = r + dr, c + dc
        pos[j] = (nr, nc)
        if shit(j, nr, nc, 0):
            pos[j] = (r, c); return True
        if depth > 12:
            pos[j] = (r, c); return True
        coll = [k for k in range(n) if k != j and ov(j, pos[j], k, pos[k])]
        for k in coll:
            if push(k, dr, dc, pos, depth + 1):
                pos[j] = (r, c); return True
        return False
    def explode(pos):
        last = {}
        for i in (1, 2, 3):
            for b in blockidx:
                r, c = pos[b]; rot = rots.get(b, 0)
                if rot == 0: z = (r+6, c, 3*i, 6); d = (3, 0)
                elif rot == 180: z = (r-3*i, c, 3*i, 6); d = (-3, 0)
                elif rot == 90: z = (r, c-3*i, 6, 3*i); d = (0, -3)
                else: z = (r, c+6, 6, 3*i); d = (0, 3)
                for j in range(n):
                    if j != b and zone_ov(z, j, pos[j]):
                        push(j, d[0], d[1], pos); last[j] = d
        for _ in range(40):
            moved = False
            for j, d in last.items():
                if shit(j, pos[j][0], pos[j][1], 1):
                    if not push(j, d[0], d[1], pos):
                        moved = True
            if not moved:
                break
    def move(st, dr, dc):
        pos, sel, cnt = st
        pos = list(pos)
        r, c = pos[sel]; nr, nc = r + dr, c + dc
        if not (shit(sel, nr, nc, 0) or shit(sel, nr, nc, 1)):
            hits = [j for j in range(n) if j != sel and ov(sel, (nr, nc), j, pos[j])]
            if not hits:
                pos[sel] = (nr, nc)
            else:
                k = 0
                for _ in range(80):
                    res = []
                    for j in hits:
                        if k >= 5 and not shit(j, pos[j][0], pos[j][1], 1):
                            res.append(True)
                        else:
                            res.append(push(j, dr, dc, pos))
                    if all(res):
                        break
                    k += 1
        if cnt is not None:
            cnt += 1
            if cnt >= 6:
                explode(pos); cnt = 0
        return (tuple(pos), sel, cnt)
    def win(pos):
        for (tr, tc, th, tw) in rtargs:
            if not any(dims[j] == (th, tw) and pos[j] == (tr, tc) for j in boxidx):
                return False
        for (tr, tc) in xtargs:
            if not any(pos[j] == (tr, tc) for j in crossidx):
                return False
        return True
    def hfun(pos):
        s = 0.0
        for (tr, tc, th, tw) in rtargs:
            best = None
            for j in boxidx:
                if dims[j] == (th, tw) and (pos[j][0]-tr) % 3 == 0 and (pos[j][1]-tc) % 3 == 0:
                    d = (abs(pos[j][0]-tr) + abs(pos[j][1]-tc)) / 3.0
                    if best is None or d < best: best = d
            s += best or 0
        for (tr, tc) in xtargs:
            best = None
            for j in crossidx:
                d = (abs(pos[j][0]-tr) + abs(pos[j][1]-tc)) / 6.0
                if best is None or d < best: best = d
            s += best or 0
        return s
    DIRS = [("ACTION1", -3, 0), ("ACTION2", 3, 0), ("ACTION3", 0, -3), ("ACTION4", 0, 3)]
    start = (tuple(pos0), sel0, cnt0)
    parent = {start: None}
    h0 = hfun(start[0])
    heap = [(2 * h0, 0, 0, start)]
    best = (h0, 0, start)
    cntr = 0
    goal = None
    pops = 0
    while heap:
        pops += 1
        if pops % 100 == 0 and time.time() > deadline:
            break
        f, g, _, st = heapq.heappop(heap)
        if win(st[0]):
            goal = st; break
        succ = []
        for a, dr, dc in DIRS:
            ns = move(st, dr, dc)
            if ns != st:
                succ.append((ns, (a, None)))
        for j in boxidx:
            if j != st[1]:
                succ.append(((st[0], j, st[2]), ("ACTION6", j)))
        for ns, a in succ:
            if ns in parent:
                continue
            parent[ns] = (st, a)
            hh = hfun(ns[0])
            cntr += 1
            heapq.heappush(heap, (g + 1 + 2 * hh, g + 1, cntr, ns))
            if (hh, g + 1) < (best[0], best[1]):
                best = (hh, g + 1, ns)
    tgt = goal if goal is not None else best[2]
    seq = []
    st = tgt
    while parent[st] is not None:
        pst, a = parent[st]
        seq.append((pst, a, st))
        st = pst
    seq.reverse()
    plan = []
    for pst, (a, j), nst in seq:
        if a == "ACTION6":
            r, c = pst[0][j]; h, w = dims[j]
            act = ("ACTION6", int(r + h // 2), int(c + w // 2))
        else:
            act = (a, None, None)
        plan.append((act, _sig(kinds, dims, nst[0], nst[1], nst[2])))
    return plan, goal is not None

def act(obs, available, memory):
    t0 = time.time()
    o = np.array(obs, dtype=int)
    if 'pad' not in memory:
        memory['pad'] = np.zeros((64, 64), bool); memory['wall'] = np.zeros((64, 64), bool)
        memory['rt'] = set(); memory['xt'] = set(); memory['rots'] = {}
    pad = memory['pad'] | (o == 2); pad[63, :] = True; memory['pad'] = pad
    wall = memory['wall'] | (o == 15); memory['wall'] = wall
    try:
        boxes, crosses, blocks, rt, xt = _detect(o)
    except Exception:
        boxes, crosses, blocks, rt, xt = [], [], [], [], []
    memory['rt'] |= set(rt); memory['xt'] |= set(xt)
    for (r, c, rot, cn) in blocks:
        if rot is not None:
            memory['rots'][(r, c)] = rot
    kinds = []; dims = []; masks = []; pos = []; rots = {}
    for (r, c) in crosses:
        kinds.append('X'); dims.append((9, 9)); masks.append(CROSS); pos.append((r, c))
    for (r, c, rot, cn) in blocks:
        rots[len(kinds)] = memory['rots'].get((r, c), 0)
        kinds.append('R'); dims.append((6, 6)); masks.append(np.ones((6, 6), bool)); pos.append((r, c))
    sel = None
    for (r, c, h, w, s) in boxes:
        if s: sel = len(kinds)
        kinds.append('B'); dims.append((h, w)); masks.append(np.ones((h, w), bool)); pos.append((r, c))
    boxidx = [i for i in range(len(kinds)) if kinds[i] == 'B']
    if sel is None and boxidx:
        sr = memory.get('selrect')
        sel = boxidx[0]
        for i in boxidx:
            if pos[i] == sr: sel = i
    cnt = max([b[3] for b in blocks]) if blocks else None
    rotsig = tuple(sorted(rots.items()))
    sig = _sig(kinds, dims, pos, sel, cnt) if sel is not None else None
    plan = memory.get('plan')
    if plan and sig is not None and memory.get('expect') == sig and memory.get('rotsig') == rotsig:
        a, exp = plan.pop(0)
        memory['expect'] = exp; memory['selrect'] = exp[1]
        return a
    if sel is not None:
        try:
            newplan, ok = _search(kinds, dims, masks, pos, sel, cnt, rots,
                                  sorted(memory['rt']), sorted(memory['xt']), pad, wall, t0 + 7.5)
        except Exception:
            newplan = []
        if newplan:
            a, exp = newplan.pop(0)
            memory['plan'] = newplan; memory['expect'] = exp; memory['rotsig'] = rotsig
            memory['selrect'] = exp[1]
            return a
    memory['plan'] = None
    f = memory.get('fail', 0) + 1; memory['fail'] = f
    others = [i for i in boxidx if i != sel]
    if f % 4 == 0 and others:
        i = others[(f // 4) % len(others)]
        r, c = pos[i]; h, w = dims[i]
        return ("ACTION6", int(r + h // 2), int(c + w // 2))
    return (["ACTION1", "ACTION4", "ACTION2", "ACTION3"][f % 4], None, None)
