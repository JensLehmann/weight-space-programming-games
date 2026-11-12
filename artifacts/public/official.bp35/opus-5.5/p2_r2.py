import numpy as np
from collections import deque

def act(obs, available, memory):
    o = np.asarray(obs)
    ys, xs = np.where(o[:63] == 9)
    if len(ys) == 0:
        return ("ACTION4", None, None)
    r0 = int(ys.min()) - 1
    base = r0 % 6
    pk = (r0 - base) // 6
    pc = int(xs.mean()) // 6

    def px(r, c):
        if 0 <= r < 63 and 0 <= c < 64:
            return int(o[r, c])
        return -1

    cells = {}
    k = 0
    while base + 6 * k + 3 < 63:
        top = base + 6 * k
        for c in range(11):
            l = 6 * c
            ctr = px(top + 3, l + 3)
            a11 = px(top + 1, l + 1)
            a12 = px(top + 1, l + 2)
            if ctr < 0:
                continue
            if ctr == 14: t = 'B'
            elif ctr == 8: t = 'G'
            elif ctr == 7: t = 'W'
            elif ctr == 12: t = 'T' if a12 == 12 else 'P'
            elif ctr == 15: t = 'Y' if a11 == 3 else 'S'
            elif ctr in (10, 9, 11): t = 'E'
            else: t = 'X'
            cells[(k, c)] = t
        k += 1
    cells[(pk, pc)] = 'E'

    def mk_get(mods):
        def get(kk, cc):
            if cc < 0 or cc > 10:
                return 'X'
            if (kk, cc) in mods:
                return mods[(kk, cc)]
            return cells.get((kk, cc), 'U')
        return get

    def fall(kk, cc, g, get):
        for _ in range(60):
            t = get(kk + g, cc)
            if t in 'EP':
                kk += g
                continue
            if t == 'U': return ('U', kk, cc)
            if t == 'W': return ('W', kk, cc)
            if t == 'S': return ('D', kk, cc)
            return ('L', kk, cc)
        return ('U', kk, cc)

    g0 = memory.get('g', -1)

    def succ(state):
        sk, sc, g, mods = state
        md = dict(mods)
        get = mk_get(md)
        out = []
        for d, an in ((-1, "ACTION3"), (1, "ACTION4")):
            if an not in available:
                continue
            nc = sc + d
            t = get(sk, nc)
            if t == 'W':
                out.append(((an, None, None), 'W', None)); continue
            if t == 'U':
                out.append(((an, None, None), 'U', None)); continue
            if t in 'EP':
                r, kk, cc = fall(sk, nc, g, get)
                out.append(((an, None, None), r, (kk, cc, g, mods)))
        if "ACTION6" in available:
            items = sorted(cells.items(), key=lambda it: (abs(it[0][1] - sc), abs(it[0][0] - sk)))
            for (ck, cc), t0 in items:
                t = md.get((ck, cc), t0)
                if t not in 'BGTPY':
                    continue
                if abs(cc - sc) > 1:
                    continue
                a = ("ACTION6", base + 6 * ck + 3, 6 * cc + 3)
                nm = dict(md)
                ng = g
                adj = (ck == sk + g and cc == sc)
                if t in 'BY':
                    nm[(ck, cc)] = 'E'
                    if t == 'Y':
                        for dk, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                            q = (ck + dk, cc + dc)
                            if q != (sk, sc) and mk_get(nm)(*q) == 'E' and q in cells:
                                nm[q] = 'Y'
                elif t == 'T':
                    nm[(ck, cc)] = 'P'
                elif t == 'P':
                    nm[(ck, cc)] = 'T'
                    adj = False
                elif t == 'G':
                    nm[(ck, cc)] = 'E'
                    ng = -g
                    adj = True
                nmods = frozenset(nm.items())
                if adj:
                    gg = mk_get(nm)
                    tt = gg(sk + ng, sc)
                    if tt in 'EPUWS':
                        r, kk, c2 = fall(sk, sc, ng, gg)
                        out.append((a, r, (kk, c2, ng, nmods)))
                    else:
                        out.append((a, 'L', (sk, sc, ng, nmods)))
                else:
                    out.append((a, 'L', (sk, sc, ng, nmods)))
        return out

    start = (pk, pc, g0, frozenset())
    seen = {start}
    q = deque([(start, None)])
    explore = None
    firsts = []
    nodes = 0
    win = None
    while q and nodes < 8000:
        st, fa = q.popleft()
        nodes += 1
        for a, r, ns in succ(st):
            f = fa if fa is not None else a
            if r == 'W':
                win = (f, ns); break
            if r == 'D':
                continue
            if r == 'U':
                if explore is None:
                    explore = (f, ns)
                continue
            if fa is None:
                firsts.append((a, ns))
            if ns not in seen:
                seen.add(ns)
                q.append((ns, f))
        if win:
            break

    def commit(a, ns):
        if a[0] == "ACTION6":
            t = cells.get(((a[1] - base - 3) // 6, (a[2] - 3) // 6))
            if t == 'G':
                memory['g'] = -g0
        return a

    if win:
        return commit(*win)
    if explore:
        return commit(*explore)
    moves = [f for f in firsts if f[1] is not None and f[1] != start]
    if moves:
        i = memory.get('i', 0)
        memory['i'] = i + 1
        return commit(*moves[i % len(moves)])
    n = memory.get('n', 0)
    memory['n'] = n + 1
    return ("ACTION4" if (n // 3) % 2 == 0 else "ACTION3", None, None)
