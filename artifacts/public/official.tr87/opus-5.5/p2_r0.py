import numpy as np
import itertools

_T = {
'A1':"..#..|..#..|.###.|..#..|#####",'A2':"....#|..#.#|#####|#.#..|#....",
'A3':"#####|#.#..|#....|#.#..|#####",'A4':"..#..|#####|#.#.#|#.#.#|..#..",
'A5':"#...#|#####|..#..|#####|#...#",'A6':"#####|..#..|###..|#....|#####",
'A7':"#...#|#####|#...#|#...#|##.##",
'B1':"#....|####.|#..#.|#..#.|#####",'B2':"#####|#...#|###.#|#.#.#|#####",
'B3':"..###|..#.#|#####|#.#..|###..",'B4':".###.|.#.#.|#####|#...#|#####",
'B5':"#####|#...#|#...#|#####|..#..",'B6':"####.|#..##|#...#|##..#|.####",
'B7':"..#..|#####|#.#.#|#####|..#..",
'C1':"#.#.#|#....|#.#.#|#....|#####",'C2':"###.#|#....|.....|....#|#.###",
'C3':"#.###|..#..|..#..|..#..|###.#",'C4':"..#..|#.#.#|..#..|#.#.#|..#..",
'C5':"##.##|#...#|#.#.#|#...#|##.##",'C6':"#.###|#.#.#|###.#|.....|#.#.#",
'C7':"#...#|.....|#...#|##.##|.#.#.",
}
BGL = {10: 'A', 7: 'B', 11: 'C'}

def _canon(p):
    return min(np.rot90(p, k).astype(np.uint8).tobytes() for k in range(4))

TEMPL = {}
for _n, _s in _T.items():
    _p = np.array([[ch == '#' for ch in r] for r in _s.split('|')])
    TEMPL.setdefault(_canon(_p), []).append((_n[0], int(_n[1])))

def ident(pat, bg):
    c = _canon(pat)
    cand = TEMPL.get(c)
    if cand:
        for l, d in cand:
            if l == BGL.get(bg):
                return (l, d)
        return cand[0]
    return ('?' + str(bg), c)

def shk(k, s):
    if not isinstance(k[1], int):
        return None
    return (k[0], (k[1] - 1 + s) % 7 + 1)

def parse(obs):
    boundary = 64
    for y in range(64):
        if obs[y, 0] == 3 and obs[y, 63] == 3:
            boundary = y
            break
    blocks = []
    for y in range(0, 57):
        x = 0
        while x < 64:
            c = int(obs[y, x]); x1 = x
            while x1 < 64 and obs[y, x1] == c:
                x1 += 1
            L = x1 - x
            if c not in (0, 1, 2, 3, 4, 5) and L >= 7 and L % 7 == 0 and (y == 0 or obs[y-1, x] != c) \
               and np.all(obs[y+6, x:x1] == c) and np.all(obs[y:y+7, x] == c) and np.all(obs[y:y+7, x1-1] == c):
                cells = []
                for k in range(L // 7):
                    cx = x + 7 * k
                    pat = obs[y+1:y+6, cx+1:cx+6] == 5
                    cells.append(((cx, y), ident(pat, c)))
                blocks.append({'y': y, 'x0': x, 'x1': x1, 'cells': cells})
            x = x1
    top = [b for b in blocks if b['y'] < boundary]
    bot = [b for b in blocks if b['y'] >= boundary]
    top.sort(key=lambda b: (b['y'], b['x0']))
    rules = []
    i = 0
    while i < len(top) - 1:
        b1, b2 = top[i], top[i+1]
        if b1['y'] == b2['y'] and b2['x0'] > b1['x1'] and np.all(obs[b1['y']+3, b1['x1']:b2['x0']] == 3):
            rules.append((b1['cells'], b2['cells']))
            i += 2
        else:
            i += 1
    S, T = [], []
    if bot:
        my = min(b['y'] for b in bot)
        for b in sorted(bot, key=lambda b: (b['y'], b['x0'])):
            (S if b['y'] == my else T).extend(b['cells'])
    return boundary, rules, S, T

def cursor(obs):
    pts = np.argwhere(obs[:63] == 0)
    if len(pts) == 0:
        return None
    y0 = pts[:, 0].min()
    x0 = pts[pts[:, 0] == y0][:, 1].min()
    return (int(x0) - 1, int(y0) + 3)

def gen(rules, S, tree):
    out = []; i = 0
    while i < len(S):
        for lk, rk in rules:
            if S[i:i+len(lk)] == lk:
                if tree:
                    o2 = []; ok = True
                    for e in rk:
                        for l2, r2 in rules:
                            if l2 and l2[0] == e:
                                o2 += r2; break
                        else:
                            ok = False; break
                    if not ok:
                        continue
                    out += o2
                else:
                    out += rk
                i += len(lk)
                break
        else:
            return None
    return out

def nearest(pos, posl):
    best = 0; bd = 1e9
    for i, p in enumerate(posl):
        d = abs(p[0]-pos[0]) + abs(p[1]-pos[1])
        if d < bd:
            bd = d; best = i
    return best

def choose(c, M, K, cur, des):
    if c in M:
        if isinstance(cur[1], int) and des is not None and isinstance(des[1], int):
            d = (des[1] - cur[1]) % 7
            return 'ACTION2' if d <= 3 else 'ACTION1'
        return 'ACTION2'
    best = min(M, key=lambda m: min((m-c) % K, (c-m) % K))
    f = (best - c) % K
    return 'ACTION4' if f <= K - f else 'ACTION3'

def solve_alter(rules, S, T):
    n = len(rules)
    Lk = [[k for _, k in r[0]] for r in rules]
    Rk = [[k for _, k in r[1]] for r in rules]
    def req(keys, seq, i):
        if i + len(keys) > len(seq):
            return None
        s = None
        for j, k in enumerate(keys):
            t = seq[i+j]
            if k[0] != t[0] or not isinstance(k[1], int) or not isinstance(t[1], int):
                return None
            v = (t[1] - k[1]) % 7
            if s is None: s = v
            elif s != v: return None
        return s
    reqL = [[req(Lk[r], S, i) for i in range(len(S))] for r in range(n)]
    reqR = {}
    best = None; bc = 1e9
    if n > 6:
        return None
    for sh in itertools.product(range(7), repeat=n):
        i = 0; em = []; ok = True
        while i < len(S):
            for r in range(n):
                if reqL[r][i] == sh[r]:
                    em.append(r); i += len(Lk[r]); break
            else:
                ok = False; break
        if not ok:
            continue
        p = 0; rs = {}
        for r in em:
            key = (r, p)
            if key not in reqR:
                reqR[key] = req(Rk[r], T, p)
            v = reqR[key]
            if v is None or rs.get(r, v) != v:
                ok = False; break
            rs[r] = v; p += len(Rk[r])
        if not ok or p != len(T):
            continue
        cost = sum(min(s, 7-s) for s in sh) + sum(min(v, 7-v) for v in rs.values())
        if cost < bc:
            bc = cost; best = (sh, rs)
    if best is None:
        return None
    sh, rs = best
    plan = []
    for r in range(n):
        plan.append([shk(k, sh[r]) for k in Lk[r]])
        plan.append([shk(k, rs.get(r, 0)) for k in Rk[r]])
    return plan

def act(obs, available, memory):
    obs = np.asarray(obs)
    try:
        boundary, rules, S, T = parse(obs)
        cur = cursor(obs)
        if cur is None:
            return ('ACTION4', None, None)
        if cur[1] < boundary:
            memory['alter'] = True
        if memory.get('alter'):
            sets = []
            for l, r in rules:
                sets.append(l); sets.append(r)
            K = len(sets)
            if K == 0:
                return ('ACTION4', None, None)
            if 'plan' not in memory:
                memory['plan'] = solve_alter(rules, [k for _, k in S], [k for _, k in T])
            plan = memory['plan']
            c = nearest(cur, [s[0][0] for s in sets])
            if plan is None or len(plan) != K:
                return ('ACTION2', None, None)
            M = [i for i in range(K) if [k for _, k in sets[i]] != plan[i]]
            if not M:
                return ('ACTION4', None, None)
            return (choose(c, M, K, sets[c][0][1], plan[c][0]), None, None)
        else:
            krules = [([k for _, k in l], [k for _, k in r]) for l, r in rules]
            Sk = [k for _, k in S]; Tk = [k for _, k in T]
            if not Tk:
                return ('ACTION4', None, None)
            d0 = gen(krules, Sk, False)
            d1 = gen(krules, Sk, True)
            D = None
            for d in (d0, d1):
                if d is not None and len(d) == len(Tk):
                    D = d; break
            if D is None:
                for d in (d0, d1):
                    if d is not None and len(d) <= len(Tk):
                        D = d; break
            if D is None:
                D = d0 or d1
            if not D:
                return ('ACTION4', None, None)
            D = D[:len(Tk)]
            K = len(Tk)
            c = nearest(cur, [p for p, _ in T])
            M = [i for i in range(len(D)) if Tk[i] != D[i]]
            if not M:
                return ('ACTION4', None, None)
            des = D[c] if c < len(D) else Tk[c]
            return (choose(c, M, K, Tk[c], des), None, None)
    except Exception:
        return ('ACTION2', None, None)
