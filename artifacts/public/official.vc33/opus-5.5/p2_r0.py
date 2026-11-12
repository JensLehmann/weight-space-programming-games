import numpy as np
from collections import deque

EXCL = {0, 1, 3, 4, 5, 9, 12}
NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _comps(mask):
    H, W = mask.shape
    seen = np.zeros(mask.shape, bool)
    out = []
    rs, cs = np.nonzero(mask)
    for r0, c0 in zip(rs.tolist(), cs.tolist()):
        if seen[r0, c0]:
            continue
        st = [(r0, c0)]; seen[r0, c0] = True; pts = []
        while st:
            r, c = st.pop(); pts.append((r, c))
            for dr, dc in NB:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True; st.append((rr, cc))
        out.append(pts)
    return out


def _bbox(pts):
    rr = [p[0] for p in pts]; cc = [p[1] for p in pts]
    return min(rr), max(rr), min(cc), max(cc)


def _cpt(pts):
    rr = np.array([p[0] for p in pts]); cc = np.array([p[1] for p in pts])
    i = int(np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2))
    return int(rr[i]), int(cc[i])


def _split(pts):
    r0, r1, c0, c1 = _bbox(pts)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if len(pts) != h * w:
        return [pts]
    if w >= 2 * h:
        return [[p for p in pts if s <= p[1] < s + h] for s in range(c0, c1 + 1, h)]
    if h >= 2 * w:
        return [[p for p in pts if s <= p[0] < s + w] for s in range(r0, r1 + 1, w)]
    return [pts]


def _colored(g):
    H, W = g.shape
    is4 = g == 4
    res = []
    for col in np.unique(g).tolist():
        if col in EXCL:
            continue
        for pts in _comps(g == col):
            adj = any(0 <= r + dr < H and 0 <= c + dc < W and is4[r + dr, c + dc]
                      for r, c in pts for dr, dc in NB)
            res.append((col, pts, adj))
    return res


def act(obs, available, memory):
    obs = np.array(obs, dtype=int)
    if "ACTION6" not in available:
        return ("RESET" if "RESET" in available else available[0], None, None)
    g = obs.copy(); g[0, :] = g[1, :]
    if "k" not in memory:
        k = 0
        cl = _colored(g)
        bp = [p for col, pts, adj in cl if adj for p in pts]
        f4 = np.argwhere(g == 4)
        if bp and len(f4):
            b = np.array(bp).mean(0); f = f4.mean(0); d = b - f
            if abs(d[0]) >= abs(d[1]):
                k = 0 if d[0] > 0 else 2
            else:
                k = 3 if d[1] > 0 else 1
        memory["k"] = k
    k = memory["k"]
    C = np.rot90(g, k)
    RI, CI = np.indices((64, 64))
    RR = np.rot90(RI, k); CC = np.rot90(CI, k)

    def click(r, c):
        return ("ACTION6", int(RR[r, c]), int(CC[r, c]))

    gates = []
    for pts in _comps((C == 1) | (C == 12)):
        r0, r1, c0, c1 = _bbox(pts)
        rr, cc = _cpt(pts)
        gates.append((("g", (r0 + r1) // 2, (c0 + c1) // 2), c0, c1, r1 + 1, rr, cc))
    conts = memory.setdefault("conts", [])
    cur = {}
    for pts in _comps((C == 0) | (C == 9)):
        zr = [p[0] for p in pts if C[p[0], p[1]] == 0]
        if not zr:
            continue
        r0, r1, c0, c1 = _bbox(pts)
        if c1 - c0 <= 1 or r1 - r0 <= 1:
            continue
        bot = r1 + 1
        idx = None
        for i, (s0, s1, sb) in enumerate(conts):
            if abs(sb - bot) <= 1 and min(s1, c1) - max(s0, c0) >= 0:
                idx = i; break
        if idx is None:
            conts.append((c0, c1, bot)); idx = len(conts) - 1
        else:
            s0, s1, sb = conts[idx]
            conts[idx] = (min(s0, c0), max(s1, c1), sb)
        cur[idx] = min(cur.get(idx, 99), min(zr))
    n = len(conts)
    surf = tuple(cur.get(i, conts[i][2]) for i in range(n))
    boats, targets = [], []
    for col, pts, adj in _colored(C):
        r0, r1, c0, c1 = _bbox(pts)
        if adj:
            mid = (c0 + c1) / 2.0; best = None
            for i, (s0, s1, sb) in enumerate(conts):
                if s0 - 1 <= mid <= s1 + 1:
                    d = abs(surf[i] - (r1 + 1))
                    if best is None or d < best[0]:
                        best = (d, i)
            boats.append((col, best[1] if best else -1))
        else:
            targets.append((col, r1, c0, c1))
    btup = tuple(b[1] for b in boats)
    bcols = [b[0] for b in boats]
    buttons = []
    for pts in _comps(C == 9):
        for ch in _split(pts):
            r0, r1, c0, c1 = _bbox(ch)
            r, c = _cpt(ch)
            buttons.append((("b", (r0 + r1) // 2, (c0 + c1) // 2), r, c))
    state = (surf, btup)
    model = memory.setdefault("model", {})
    blocked = memory.setdefault("blocked", set())
    tries = memory.setdefault("tries", {})
    prev = memory.get("prev")
    if prev is not None:
        ps, key = prev
        p0 = tuple(ps[0]) + tuple(conts[i][2] for i in range(len(ps[0]), n))
        if len(p0) == n:
            if key[0] == "b":
                dl = tuple(a - b for a, b in zip(surf, p0))
                if any(dl):
                    model[key] = dl
                else:
                    blocked.add((ps, key))
            elif ps[1] == btup:
                blocked.add((ps, key))
    memory["prev"] = None

    def goal(s, bt):
        if not bt:
            return False
        for col, ci in zip(bcols, bt):
            if ci < 0 or ci >= len(s):
                return False
            c0, c1, _ = conts[ci]
            ok = False
            for tc, tr1, t0, t1 in targets:
                if tc == col and s[ci] == tr1 + 1 and (abs(c1 + 1 - t0) <= 1 or abs(t1 + 1 - c0) <= 1):
                    ok = True; break
            if not ok:
                return False
        return True

    gate_info = {gk: (g0, g1, gb, r, c) for gk, g0, g1, gb, r, c in gates}
    mdl = {}
    for key, dl in model.items():
        dl = tuple(dl)[:n]
        mdl[key] = dl + (0,) * (n - len(dl))

    def succ(st):
        s, bt = st
        for key, dl in mdl.items():
            if (st, key) in blocked:
                continue
            ns = tuple(a + b for a, b in zip(s, dl))
            if any(ns[i] > conts[i][2] or ns[i] < 1 for i in range(n)):
                continue
            yield key, (ns, bt)
        for gk, (g0, g1, gb, _, _) in gate_info.items():
            if (st, gk) in blocked:
                continue
            L = [i for i in range(n) if s[i] == gb and abs(conts[i][1] + 1 - g0) <= 1]
            R = [i for i in range(n) if s[i] == gb and abs(conts[i][0] - (g1 + 1)) <= 1]
            if not L or not R:
                continue
            li, ri = L[0], R[0]
            nb = tuple(ri if b == li else (li if b == ri else b) for b in bt)
            if nb != bt:
                yield gk, (s, nb)

    first = None
    try:
        if len(mdl) or gates:
            par = {state: None}
            dq = deque([state])
            found = None
            cnt = 0
            while dq and cnt < 40000:
                st = dq.popleft(); cnt += 1
                if st != state and goal(*st):
                    found = st; break
                for key, ns in succ(st):
                    if ns not in par:
                        par[ns] = (st, key); dq.append(ns)
            if found is not None:
                st = found
                while par[st][0] != state:
                    st = par[st][0]
                first = par[st][1]
    except Exception:
        first = None
    bmap = {key: (r, c) for key, r, c in buttons}
    r = c = None
    if first is not None:
        if first[0] == "g" and first in gate_info:
            r, c = gate_info[first][3], gate_info[first][4]
        elif first in bmap:
            r, c = bmap[first]
        else:
            first = None
    if first is None:
        cand = [key for key, _, _ in buttons if key not in model]
        if not cand:
            cand = [key for key, _, _ in buttons]
        if not cand:
            return ("ACTION6", 32, 32)
        first = min(cand, key=lambda kk: (tries.get((state, kk), 0), tries.get(kk, 0)))
        r, c = bmap[first]
    tries[(state, first)] = tries.get((state, first), 0) + 1
    tries[first] = tries.get(first, 0) + 1
    memory["prev"] = (state, first)
    return click(r, c)
