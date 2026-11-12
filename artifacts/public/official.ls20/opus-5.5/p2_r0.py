import numpy as np
from collections import deque
import random

SH = [[[1,1,0],[0,1,1],[1,0,1]],
      [[0,1,0],[0,1,0],[1,1,1]],
      [[1,0,1],[1,0,1],[1,1,1]],
      [[0,1,1],[1,0,1],[0,1,0]],
      [[0,1,0],[1,1,0],[0,1,1]],
      [[1,1,1],[0,0,1],[1,0,1]]]
COLS = [12, 9, 14, 8]
ACTS = [("ACTION1", -1, 0), ("ACTION2", 1, 0), ("ACTION3", 0, -1), ("ACTION4", 0, 1)]
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
NR, NC = 11, 12

def _rmask(s, r):
    m = np.array(SH[s], dtype=bool)
    k = ((-r) % 360) // 90
    return np.rot90(m, k) if k else m

def _match(mask):
    out = []
    for s in range(6):
        for ri in range(4):
            if np.array_equal(_rmask(s, ri * 90), mask):
                out.append((s, ri))
    return out

def act(obs, available, memory):
    obs = np.asarray(obs).astype(int)
    mp = memory.setdefault('map', {})
    player = None
    goals = {}
    chg = {'S': set(), 'K': set(), 'R': set()}
    for r in range(NR):
        for c in range(NC):
            y, x = 5 * r, 4 + 5 * c
            p = obs[y:y + 5, x:x + 5]
            if (p[:2] == 12).all() and (p[2:] == 9).all():
                player = (r, c)
                if mp.get((r, c)) in ('#', 'G'):
                    mp[(r, c)] = '.'
                continue
            if (p == 5).all():
                continue
            u = set(np.unique(p).tolist())
            ring = np.concatenate([p[0], p[4], p[:, 0], p[:, 4]])
            if (p == 4).sum() >= 18:
                t = '#'
            elif (ring == 5).all() and len(u) > 1:
                t = 'G'
                g = p[1:4, 1:4]
                gm = g != 5
                cs = [v for v in g[gm].tolist() if v in COLS]
                if cs:
                    goals[(r, c)] = (set(_match(gm)), COLS.index(cs[0]))
            elif 11 in u:
                t = 'F'
            elif 1 in u and 0 in u:
                t = 'R'
            elif 14 in u and 0 in u:
                t = 'K'
            elif 0 in u:
                t = 'S'
            else:
                t = '.'
            mp[(r, c)] = t
            if t in chg:
                chg[t].add((r, c))

    def typ(cell):
        r, c = cell
        if r < 0 or r >= NR or c < 0 or c >= NC:
            return '#'
        return mp.get(cell, '.')

    # ---- mover tracking ----
    vis = memory.setdefault('vis', {'S': set(), 'K': set(), 'R': set()})
    dead = memory.setdefault('dead', {'S': set(), 'K': set(), 'R': set()})
    prev = memory.get('chg', {'S': set(), 'K': set(), 'R': set()})
    mv = memory.get('mv', [])

    def adj(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def ddir(a, b):
        return DIRS.index((b[0] - a[0], b[1] - a[1]))

    newmv = []
    taken = set()
    for tp, pos, d, pred in mv:
        cur = chg[tp]
        if pos in cur and pos not in taken:
            newmv.append([tp, pos, d, None]); taken.add(pos); continue
        cand = [n for n in cur if adj(n, pos) and n not in prev[tp] and n not in taken]
        if cand:
            n = cand[0]
            if pred is not None and pred != n:
                dead[tp].add(pred)
            vis[tp].add(n); vis[tp].add(pos)
            newmv.append([tp, n, ddir(pos, n), None]); taken.add(n)
        elif player is not None and (pos == player or pred == player):
            newmv.append([tp, player if pred == player else pos, d, None])
    for tp in chg:
        for a in chg[tp] - prev[tp]:
            if a in taken:
                continue
            for dd in prev[tp] - chg[tp]:
                if adj(a, dd):
                    vis[tp].add(a); vis[tp].add(dd)
                    newmv.append([tp, a, ddir(dd, a), None]); taken.add(a)
                    break
    memory['chg'] = chg
    for m_ in newmv:
        if mp.get(m_[1]) == m_[0]:
            mp[m_[1]] = '.'

    def mstep(tp, pos, d):
        for nd in (d, (d - 1) % 4, (d + 1) % 4, (d + 2) % 4):
            dr, dc = DIRS[nd]
            n = (pos[0] + dr, pos[1] + dc)
            if n in vis[tp] or (nd == d and n not in dead[tp] and typ(n) not in ('#', 'G')):
                return n, nd
        return pos, d

    TM = 300
    seq = [tuple((m_[1], m_[2]) for m_ in newmv)]
    for t in range(TM):
        seq.append(tuple(mstep(newmv[i][0], s_[0], s_[1]) for i, s_ in enumerate(seq[-1])))
    for i, m_ in enumerate(newmv):
        m_[3] = seq[1][i][0]
    memory['mv'] = newmv
    P = None
    if newmv:
        for k in range(1, TM + 1):
            if seq[k] == seq[0]:
                P = k
                break
    else:
        P = 1
    dyn = [{s_[0]: newmv[i][0] for i, s_ in enumerate(st)} for st in seq]

    def dget(t):
        return dyn[t % P] if P else dyn[min(t, TM)]

    # pushers
    push = {}
    for y in range(55):
        for c in range(NC):
            x = 4 + 5 * c
            if (obs[y, x:x + 5] == 1).all():
                if y % 5 == 4:
                    push[((y + 1) // 5, c)] = (1, 0)
                elif y % 5 == 0:
                    push[(y // 5 - 1, c)] = (-1, 0)
    for x in range(4, 64):
        for r in range(NR):
            y = 5 * r
            if (obs[y:y + 5, x] == 1).all():
                if (x - 4) % 5 == 4:
                    push[(r, (x - 3) // 5)] = (0, 1)
                elif (x - 4) % 5 == 0:
                    push[(r, (x - 4) // 5 - 1)] = (0, -1)
    pdist = {}
    for T, (dr, dc) in push.items():
        k = 0
        for i in range(1, 12):
            cc = (T[0] + dr * i, T[1] + dc * i)
            if typ(cc) in ('#', 'G'):
                k = i - 1
                break
        if k > 0:
            pdist[T] = (T[0] + dr * k, T[1] + dc * k)

    row = obs[61, 13:55]
    cur = int((row == 11).sum())
    mx = int(((row == 11) | (row == 3)).sum()) or 42
    last = memory.get('last')
    if last is not None and last[0] - cur == 2:
        memory['dec'] = 2
    dec = memory.get('dec', 1)

    smp = obs[55:61:2, 3:9:2]
    km = (smp != 5) & (smp != 4)
    cs = [v for v in smp[km].tolist() if v in COLS]
    mt = _match(km)
    if mt and cs:
        key = (mt[0][0], COLS.index(cs[0]), mt[0][1])
        memory['key'] = key
    else:
        key = memory.get('key', (0, 0, 0))

    def ret(a):
        memory['last'] = (cur, a)
        return (a, None, None)

    if player is None:
        return ret(random.choice(["ACTION1", "ACTION2", "ACTION3", "ACTION4"]))

    refills = [cl for cl, t in mp.items() if t == 'F']
    ridx = {cl: i for i, cl in enumerate(refills)}

    def eff(cell, s, co, ro, m, b, full, dm):
        t = dm.get(cell) or typ(cell)
        if t == 'S':
            s = (s + 1) % 6
        elif t == 'K':
            co = (co + 1) % 4
        elif t == 'R':
            ro = (ro + 1) % 4
        elif t == 'F' and not (m >> ridx[cell]) & 1:
            m |= 1 << ridx[cell]
            b = full
        return s, co, ro, m, b

    def search(limit):
        full = mx // dec if limit else 10 ** 6
        b0 = cur // dec if limit else 10 ** 6
        st = (player, key[0], key[1], key[2], 0)
        best = {(st, 0): b0}
        q = deque([(st, b0, None, 0)])
        pops = 0
        while q:
            (pc, s, co, ro, m), b, first, t = q.popleft()
            pops += 1
            if pops > 200000 or (not P and t >= TM):
                break
            dm = dget(t + 1)
            for a, dr, dc in ACTS:
                N = (pc[0] + dr, pc[1] + dc)
                tt = typ(N)
                if tt == '#':
                    continue
                fa = first or a
                if N in goals:
                    req, gc = goals[N]
                    if co == gc and (s, ro) in req:
                        return fa
                    continue
                if tt == 'G':
                    continue
                isref = tt == 'F' and not (m >> ridx[N]) & 1
                if not isref and b - 1 < 0:
                    continue
                s2, c2, r2, m2, nb = eff(N, s, co, ro, m, b - 1, full, dm)
                Pp = N
                if N in pdist:
                    Pp = pdist[N]
                    s2, c2, r2, m2, nb = eff(Pp, s2, c2, r2, m2, nb, full, dm)
                ns = (Pp, s2, c2, r2, m2)
                kk = (ns, (t + 1) % P if P else t + 1)
                if best.get(kk, -1) < nb:
                    best[kk] = nb
                    q.append((ns, nb, fa, t + 1))
        return None

    a = search(True)
    if a is None:
        a = search(False)
    if a is None:
        opts = [n for n, dr, dc in ACTS
                if typ((player[0] + dr, player[1] + dc)) not in ('#', 'G')]
        a = random.choice(opts) if opts else "ACTION1"
    return ret(a)
