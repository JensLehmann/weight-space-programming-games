import numpy as np
from collections import deque

_T = {
"A1":"..#../..#../.###./..#../#####","A2":"....#/..#.#/#####/#.#../#....",
"A3":"#####/#.#../#..../#.#../#####","A4":"..#../#####/#.#.#/#.#.#/..#..",
"A5":"#...#/#####/..#../#####/#...#","A6":"#####/..#../###../#..../#####",
"A7":"#...#/#####/#...#/#...#/##.##","B1":"#..../####./#..#./#..#./#####",
"B2":"#####/#...#/###.#/#.#.#/#####","B3":"..###/..#.#/#####/#.#../###..",
"B4":".###./.#.#./#####/#...#/#####","B5":"#####/#...#/#...#/#####/..#..",
"B6":"####./#..##/#...#/##..#/.####","B7":"..#../#####/#.#.#/#####/..#..",
"C1":"#.#.#/#..../#.#.#/#..../#####","C2":"###.#/#..../...../....#/#.###",
"C3":"#.###/..#../..#../..#../###.#","C4":"..#../#.#.#/..#../#.#.#/..#..",
"C5":"##.##/#...#/#.#.#/#...#/##.##","C6":"#.###/#.#.#/###.#/...../#.#.#",
"C7":"#...#/...../#...#/##.##/.#.#."}

def _canon(a):
    return min(''.join('1' if v else '0' for v in np.rot90(a, k).flatten()) for k in range(4))

_KEY = {}
for _n, _s in _T.items():
    _a = np.array([[c == '#' for c in r] for r in _s.split('/')])
    _KEY.setdefault(_canon(_a), _n)

def _ident(pat):
    c = _canon(pat)
    return _KEY.get(c, "U" + c)

def _parse(obs):
    bgy = 34
    for r in range(64):
        if (obs[r] == 3).all():
            bgy = r; break
    mask = ~np.isin(obs, [0, 1, 2, 3, 4]); mask[63, :] = False
    seen = np.zeros((64, 64), bool); comps = []
    for y in range(64):
        for x in range(64):
            if mask[y, x] and not seen[y, x]:
                q = deque([(y, x)]); seen[y, x] = True; ys = [y]; xs = [x]
                while q:
                    a, b = q.popleft()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < 64 and 0 <= nb < 64 and mask[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True; q.append((na, nb)); ys.append(na); xs.append(nb)
                y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
                if y1 - y0 < 4: continue
                n = max(1, int(round((x1 - x0 + 1) / 7.0)))
                cells = []
                for i in range(n):
                    gx, gy = x0 + 7 * i + 1, y0 + 1
                    pat = obs[gy:gy + 5, gx:gx + 5] == 5
                    if pat.shape != (5, 5): continue
                    cells.append((_ident(pat), gx, gy))
                comps.append((y0, x0, x1, cells))
    top = sorted([c for c in comps if c[0] < bgy])
    bot = sorted([c for c in comps if c[0] > bgy])
    rules = []
    for c in top:
        y0, x0, x1, cells = c
        r = y0 + 3; x = x1 + 1
        if r < 64 and x < 64 and obs[r, x] == 3:
            while x < 64 and obs[r, x] == 3: x += 1
            for d in top:
                if d[0] == y0 and abs(d[1] - x) <= 1:
                    rules.append((cells, d[3])); break
    rows = {}
    for c in bot: rows.setdefault(c[0], []).extend(c[3])
    ks = sorted(rows)
    src = sorted(rows[ks[0]], key=lambda t: t[1]) if len(ks) > 0 else []
    tgt = sorted(rows[ks[1]], key=lambda t: t[1]) if len(ks) > 1 else []
    z = np.argwhere(obs[:63] == 0)
    cur = None
    if len(z):
        cy = int(z[:, 0].min()); cx = int(z[z[:, 0] == cy][:, 1].min())
        cur = (cx, cy)
    return bgy, rules, src, tgt, cur

def _need(a, b):
    if a == b: return 0
    if a[0] != 'U' and b[0] != 'U' and a[0] == b[0]:
        return (int(b[1]) - int(a[1])) % 7
    return None

def _sh(x, s):
    if s == 0: return x
    if x[0] == 'U': return None
    return x[0] + str((int(x[1]) - 1 + s) % 7 + 1)

def _match(names, s, seq, pos):
    if pos + len(names) > len(seq): return False
    return all(_sh(n, s) == seq[pos + k] for k, n in enumerate(names))

def _needset(names, seq, pos):
    if pos + len(names) > len(seq): return None
    s = _need(names[0], seq[pos])
    if s is None or not _match(names, s, seq, pos): return None
    return s

def _solve(S, T, sets):
    R = len(sets) // 2
    best = [None, 1e9]; cnt = [0]
    def cost(s): return min(s, 7 - s)
    def finish(assign, forbid):
        full = []; c = 0
        for k in range(len(sets)):
            if k in assign: s = assign[k]
            else:
                f = forbid.get(k, frozenset())
                s = min((v for v in range(7) if v not in f), key=cost)
            full.append(s); c += cost(s)
        if c < best[1]: best[0], best[1] = full, c
    def tryrule(j, p, q, assign, forbid):
        rn = sets[2 * j + 1]; ri = 2 * j + 1
        if ri in assign:
            if _match(rn, assign[ri], T, q):
                dfs(p + len(sets[2 * j]), q + len(rn), assign, forbid)
        else:
            s = _needset(rn, T, q)
            if s is None or s in forbid.get(ri, ()): return
            a2 = dict(assign); a2[ri] = s
            dfs(p + len(sets[2 * j]), q + len(rn), a2, forbid)
    def dfs(p, q, assign, forbid):
        cnt[0] += 1
        if cnt[0] > 30000: return
        if p == len(S):
            finish(assign, forbid); return
        for j in range(R):
            li = 2 * j; L = sets[li]
            if li in assign:
                if _match(L, assign[li], S, p):
                    tryrule(j, p, q, assign, forbid); return
                continue
            s = _needset(L, S, p)
            if s is None: continue
            if s in forbid.get(li, ()): continue
            a2 = dict(assign); a2[li] = s
            tryrule(j, p, q, a2, forbid)
            f2 = dict(forbid); f2[li] = forbid.get(li, frozenset()) | {s}; forbid = f2
    dfs(0, 0, {}, {})
    return best[0]

def _fallback(memory):
    c = memory.get("fb", 0); memory["fb"] = c + 1
    return ("ACTION4" if c % 8 == 7 else "ACTION2", None, None)

def _nav(ci, ti, n, s):
    fwd = (ti - ci) % n
    if fwd == 0:
        return ("ACTION2" if s <= 3 else "ACTION1", None, None)
    return ("ACTION4" if fwd <= n - fwd else "ACTION3", None, None)

def act(obs, available, memory):
    obs = np.asarray(obs)
    try:
        bgy, rules, src, tgt, cur = _parse(obs)
    except Exception:
        return _fallback(memory)
    if cur is None or not src:
        return _fallback(memory)
    cx, cy = cur; gy = cy + 4
    S = [c[0] for c in src]; T = [c[0] for c in tgt]
    if gy < bgy:
        sets = []; pos = []
        for L, Rr in rules:
            sets.append([c[0] for c in L]); pos.append((L[0][1], L[0][2]))
            sets.append([c[0] for c in Rr]); pos.append((Rr[0][1], Rr[0][2]))
        if not sets: return _fallback(memory)
        ci = min(range(len(pos)), key=lambda k: abs(pos[k][0] - cx) + abs(pos[k][1] - gy))
        sol = _solve(S, T, sets)
        if sol is None: return _fallback(memory)
        n = len(sets)
        cand = [k for k in range(n) if sol[k] != 0]
        if not cand: return _fallback(memory)
        k = min(cand, key=lambda k: min((k - ci) % n, (ci - k) % n))
        return _nav(ci, k, n, sol[k])
    # normal mode
    if not T: return _fallback(memory)
    D = []; p = 0; ok = True
    rl = [([c[0] for c in L], [c[0] for c in Rr]) for L, Rr in rules]
    while p < len(S):
        for L, Rr in rl:
            if S[p:p + len(L)] == L:
                D += Rr; p += len(L); break
        else:
            ok = False; break
    if not ok or len(D) > len(T) or not D:
        return _fallback(memory)
    n = len(T)
    ci = min(range(n), key=lambda k: abs(tgt[k][1] - cx))
    bad = [i for i in range(len(D)) if D[i] != T[i]]
    if not bad: return _fallback(memory)
    i = min(bad, key=lambda k: min((k - ci) % n, (ci - k) % n))
    s = _need(T[i], D[i])
    if s is None: s = 1
    return _nav(ci, i, n, s)
