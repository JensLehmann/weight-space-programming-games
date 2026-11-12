import numpy as np
from itertools import product

_T = {
"A1":["..#..","..#..",".###.","..#..","#####"],
"A2":["....#","..#.#","#####","#.#..","#...."],
"A3":["#####","#.#..","#....","#.#..","#####"],
"A4":["..#..","#####","#.#.#","#.#.#","..#.."],
"A5":["#...#","#####","..#..","#####","#...#"],
"A6":["#####","..#..","###..","#....","#####"],
"A7":["#...#","#####","#...#","#...#","##.##"],
"B1":["#....","####.","#..#.","#..#.","#####"],
"B2":["#####","#...#","###.#","#.#.#","#####"],
"B3":["..###","..#.#","#####","#.#..","###.."],
"B4":[".###.",".#.#.","#####","#...#","#####"],
"B5":["#####","#...#","#...#","#####","..#.."],
"B6":["####.","#..##","#...#","##..#",".####"],
"B7":["..#..","#####","#.#.#","#####","..#.."],
"C1":["#.#.#","#....","#.#.#","#....","#####"],
"C2":["###.#","#....",".....","....#","#.###"],
"C3":["#.###","..#..","..#..","..#..","###.#"],
"C4":["..#..","#.#.#","..#..","#.#.#","..#.."],
"C5":["##.##","#...#","#.#.#","#...#","##.##"],
"C6":["#.###","#.#.#","###.#",".....","#.#.#"],
"C7":["#...#",".....","#...#","##.##",".#.#."]}
_G = {}

def _canon(sub):
    best = None
    for k in range(4):
        b = np.packbits(np.rot90(sub, k).flatten()).tobytes()
        if best is None or b < best:
            best = b
    return best

def _init():
    if _G:
        return
    nxt = {}
    for fam in "ABC":
        cs = []
        for i in range(1, 8):
            rows = _T[fam + str(i)]
            arr = np.array([[c == '#' for c in r] for r in rows])
            assert arr.shape == (5, 5)
            cs.append(_canon(arr))
        for i in range(7):
            nxt.setdefault(cs[i], cs[(i + 1) % 7])
    _G['next'] = nxt

def _comps(obs):
    mask = ~np.isin(obs, [0, 1, 2, 3, 4]); mask[63:, :] = False
    seen = np.zeros_like(mask); out = []
    H, W = mask.shape
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not seen[y, x]:
                st = [(y, x)]; seen[y, x] = True
                y0, y1, x0, x1 = y, y, x, x
                while st:
                    a, b = st.pop()
                    y0 = min(y0, a); y1 = max(y1, a); x0 = min(x0, b); x1 = max(x1, b)
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        u, v = a + da, b + db
                        if 0 <= u < H and 0 <= v < W and mask[u, v] and not seen[u, v]:
                            seen[u, v] = True; st.append((u, v))
                h = y1 - y0 + 1; w = x1 - x0 + 1
                if h == 7 and w % 7 == 0:
                    tiles = []
                    for i in range(w // 7):
                        sy, sx = y0 + 1, x0 + 7 * i + 1
                        tiles.append((sy, sx, _canon(obs[sy:sy + 5, sx:sx + 5] == 5)))
                    out.append((y0, x0, x1, tiles))
    return out

def _shift(nxt, c, d):
    for _ in range(d):
        c = nxt.get(c)
        if c is None:
            return None
    return c

def _dist(nxt, a, b):
    if a == b:
        return 0
    cur = a
    for s in range(1, 7):
        cur = nxt.get(cur)
        if cur is None:
            return 'U'
        if cur == b:
            return s
    return -1

def _parse(top, lefts):
    pos = 0; segs = []
    while pos < len(top):
        for r, L in enumerate(lefts):
            if L is not None and tuple(top[pos:pos + len(L)]) == L:
                segs.append(r); pos += len(L); break
        else:
            return None
    return segs

def act(obs, available, memory):
    _init()
    obs = np.array(obs)
    if 'next' not in memory:
        memory['next'] = dict(_G['next'])
    nxt = memory['next']
    memory['t'] = memory.get('t', 0) + 1
    comps = _comps(obs)
    allpos = {}
    for c in comps:
        for (sy, sx, cn) in c[3]:
            allpos[(sy, sx)] = cn
    last = memory.get('last')
    if last and last[0] in ('ACTION1', 'ACTION2'):
        for p, old in last[1].items():
            new = allpos.get(p)
            if new is not None and new != old:
                if last[0] == 'ACTION2':
                    nxt[old] = new
                else:
                    nxt[new] = old

    def ret(a):
        memory['last'] = (a, allpos)
        return (a, None, None)

    bnd = 34
    for r in range(64):
        if np.all(obs[r] == 3):
            bnd = r; break
    topc = [c for c in comps if c[0] < bnd]
    botc = [c for c in comps if c[0] >= bnd]
    if not botc:
        return ret('ACTION2')
    ymin = min(c[0] for c in botc)
    src = []; dst = []
    for c in sorted(botc, key=lambda c: (c[0], c[1])):
        (src if c[0] == ymin else dst).extend(c[3])
    srcC = [t[2] for t in src]; dstC = [t[2] for t in dst]
    rules = []
    for c in topc:
        y0, x0, x1, tl = c
        if x1 + 1 < 64 and obs[y0 + 3, x1 + 1] == 3:
            cand = [d for d in topc if d[0] == y0 and d[1] > x1]
            if cand:
                rc = min(cand, key=lambda d: d[1])
                rules.append(((y0, x1), tl, rc[3]))
    rules.sort(key=lambda r: r[0])
    zr = np.argwhere(obs[:63] == 0)
    if len(zr) == 0:
        return ret('ACTION2')
    rmin = zr[:, 0].min()
    cx = zr[zr[:, 0] == rmin][:, 1].min()
    cy = rmin + 4
    alter = cy < bnd

    if not alter:
        lefts = [tuple(t[2] for t in r[1]) for r in rules]
        segs = _parse(srcC, lefts)
        target = []
        if segs is not None:
            for r in segs:
                target += [t[2] for t in rules[r][2]]
        n = len(dstC)
        if n == 0:
            return ret('ACTION2')
        cur = 0
        for j, t in enumerate(dst):
            if t[0] == cy and t[1] == cx:
                cur = j
        need = []
        for j in range(n):
            d = _dist(nxt, dstC[j], target[j]) if j < len(target) else 0
            if d == -1:
                d = 0
            need.append(d)
    else:
        sets = []
        for r in rules:
            sets.append(r[1]); sets.append(r[2])
        n = len(sets)
        if n == 0:
            return ret('ACTION2')
        cur = 0
        for k, s in enumerate(sets):
            if s[0][0] == cy and s[0][1] == cx:
                cur = k
        curC = [tuple(t[2] for t in s) for s in sets]
        tgt = memory.get('atgt')
        ok = tgt is not None and len(tgt) == n
        if ok:
            need = []
            for k in range(n):
                d = _dist(nxt, curC[k][0], tgt[k][0])
                if d == 'U' or d == -1:
                    ok = False; break
                need.append(d)
        if not ok:
            R = len(rules)
            sl = []; rm = []
            for r in range(R):
                L = curC[2 * r]; Rt = curC[2 * r + 1]
                opts = []
                for d in range(7):
                    s = tuple(_shift(nxt, c, d) for c in L)
                    opts.append(None if None in s else s)
                sl.append(opts)
                m = {}
                for d in range(7):
                    s = tuple(_shift(nxt, c, d) for c in Rt)
                    if None in s:
                        continue
                    cst = min(d, 7 - d)
                    if s not in m or m[s][1] > cst:
                        m[s] = (d, cst)
                rm.append(m)
            best = None; bestsol = None
            for combo in product(range(7), repeat=R):
                if any(sl[r][combo[r]] is None for r in range(R)):
                    continue
                cl = sum(min(d, 7 - d) for d in combo)
                if best is not None and cl >= best:
                    continue
                lefts = [sl[r][combo[r]] for r in range(R)]
                segs = _parse(srcC, lefts)
                if segs is None:
                    continue
                pb = 0; asg = {}; fail = False
                for r in segs:
                    ln = len(curC[2 * r + 1])
                    if pb + ln > len(dstC):
                        fail = True; break
                    v = rm[r].get(tuple(dstC[pb:pb + ln]))
                    if v is None or (r in asg and asg[r] != v):
                        fail = True; break
                    asg[r] = v; pb += ln
                if fail:
                    continue
                tot = cl + sum(v[1] for v in asg.values())
                if best is None or tot < best:
                    best = tot; bestsol = (combo, asg)
            if bestsol is None:
                return ret('ACTION2' if memory['t'] % 2 else 'ACTION4')
            combo, asg = bestsol
            tgt = []; need = []
            for r in range(R):
                tgt.append(sl[r][combo[r]]); need.append(combo[r])
                d = asg[r][0] if r in asg else 0
                tgt.append(tuple(_shift(nxt, c, d) for c in curC[2 * r + 1])); need.append(d)
            memory['atgt'] = tgt

    d = need[cur] if cur < len(need) else 0
    if d == 'U':
        return ret('ACTION2')
    if d != 0:
        return ret('ACTION2' if d <= 3 else 'ACTION1')
    cands = [j for j in range(len(need)) if need[j] != 0]
    if not cands:
        return ret('ACTION2')
    bd = None; ba = None
    for j in cands:
        rr = (j - cur) % n; ll = (cur - j) % n
        if bd is None or min(rr, ll) < bd:
            bd = min(rr, ll); ba = 'ACTION4' if rr <= ll else 'ACTION3'
    return ret(ba)
