import numpy as np
import random
from collections import deque

WALK = [1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15]
DIRS = [("ACTION1", -1, 0), ("ACTION2", 1, 0), ("ACTION3", 0, -1), ("ACTION4", 0, 1)]


def find_buttons(area, px):
    buttons = []
    if px >= 62:
        return buttons
    reg = area[:, px + 2:]
    bm = ~np.isin(reg, [0, 3, 4, 5])
    seen = np.zeros(bm.shape, bool)
    hh, ww = bm.shape
    for r in range(hh):
        for c in range(ww):
            if bm[r, c] and not seen[r, c]:
                comp = []
                st = [(r, c)]
                seen[r, c] = True
                while st:
                    a, b = st.pop()
                    comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < hh and 0 <= nb < ww and bm[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True
                            st.append((na, nb))
                if len(comp) < 4:
                    continue
                my = sum(p[0] for p in comp) / len(comp)
                mx = sum(p[1] for p in comp) / len(comp)
                best = min(comp, key=lambda p: (p[0] - my) ** 2 + (p[1] - mx) ** 2)
                buttons.append((int(best[0]), int(best[1] + px + 2)))
    return buttons


def reset_model(M):
    M.update(eff={}, bits={}, mask=0, bad=set(), blocked=set())


def act(obs, available, memory):
    obs = np.asarray(obs).astype(int)
    M = memory
    if 'rng' not in M:
        M.update(rng=random.Random(1), tele={}, prev=None, ttried=set(), fails={},
                 dyn=set(), pairs=set(), vis=set())
        reset_model(M)
    rng = M['rng']
    area = obs[:63]
    nonpad = area != 3
    cnt5 = ((area == 5) & nonpad).sum(0)
    tot = nonpad.sum(0)
    px = 64
    for c in range(64):
        if tot[c] > 0 and cnt5[c] > 0.4 * tot[c]:
            px = c
            break
    play = area[:, :px].copy()
    ys, xs = np.where(play == 14)
    if len(ys) == 0:
        M['prev'] = None
        return (rng.choice(["ACTION1", "ACTION2", "ACTION3", "ACTION4"]), None, None)
    pr, pc = int(ys.min()), int(xs.min())
    s = max(1, int(ys.max()) - pr + 1)
    pos = (pr, pc)
    buttons = find_buttons(area, px)
    M['vis'].add(pos)

    pv = M['prev']
    if pv is not None and pv[3].shape == play.shape:
        kind, ppos, arg, pplay, pmask = pv
        if kind == 'move' and pos == ppos:
            M['blocked'].add((ppos, arg, M['mask']))
        if kind == 'click':
            b = arg
            ch = pplay != play
            for (a, c) in (ppos, pos):
                ch[a:a + s, c:c + s] = False
            if abs(pos[0] - ppos[0]) + abs(pos[1] - ppos[1]) > s:
                M['tele'][(ppos, b)] = pos
            M['ttried'].add((ppos, b))
            if b in M['dyn']:
                pass
            elif ch.any():
                if b not in M['eff']:
                    idx = np.where(ch)
                    M['eff'][b] = (idx, pplay[idx].copy(), play[idx].copy())
                    M['bits'][b] = 1 << len(M['bits'])
                    M['mask'] ^= M['bits'][b]
                else:
                    idx, v0, v1 = M['eff'][b]
                    pred = pplay.copy()
                    pred[idx] = v0 if (M['mask'] & M['bits'][b]) else v1
                    d = pred != play
                    for (a, c) in (ppos, pos):
                        d[a:a + s, c:c + s] = False
                    if d.any():
                        M['dyn'].add(b)
                        reset_model(M)
                    else:
                        M['mask'] ^= M['bits'][b]
            else:
                M['bad'].add((ppos, b, M['mask']))
                M['fails'][b] = M['fails'].get(b, 0) + 1
    cur_mask = M['mask']
    eff, bits = M['eff'], M['bits']
    gcache = {}

    def grid(m):
        if m in gcache:
            return gcache[m]
        g = play.copy()
        diff = m ^ cur_mask
        for b, bt in bits.items():
            if diff & bt:
                idx, v0, v1 = eff[b]
                g[idx] = v1 if (m & bt) else v0
        W = np.isin(g, WALK)
        H, Wd = W.shape
        cw = np.zeros((H, Wd), bool)
        if H >= s and Wd >= s:
            acc = np.ones((H - s + 1, Wd - s + 1), bool)
            for i in range(s):
                for j in range(s):
                    acc &= W[i:H - s + 1 + i, j:Wd - s + 1 + j]
            cw[:H - s + 1, :Wd - s + 1] = acc
        gcache[m] = (g, cw)
        return gcache[m]

    def ok(m, p):
        r, c = p
        cw = grid(m)[1]
        return 0 <= r < cw.shape[0] and 0 <= c < cw.shape[1] and cw[r, c]

    gy, gx = np.where(play == 11)
    goal = (int(gy.min()), int(gx.min())) if len(gy) else None
    unlearned = [b for b in buttons if b not in eff and b not in M['dyn'] and M['fails'].get(b, 0) < 3]
    learned = [b for b in buttons if b in eff]

    start = (pos, cur_mask)
    par = {start: None}
    dq = deque([start])
    found = explore = tele_ex = None
    n = 0
    while dq and n < 40000:
        n += 1
        st = dq.popleft()
        p, m = st
        if goal is not None and p == goal and st != start:
            found = (st, None)
            break
        g = grid(m)[0]
        if explore is None:
            for b in unlearned:
                if (p, b, m) not in M['bad']:
                    explore = (st, ('click', b))
                    break
            if explore is None and st != start and s == 2:
                blk = g[p[0]:p[0] + 2, p[1]:p[1] + 2]
                if (blk == 15).sum() == 3:
                    explore = (st, None)
        if tele_ex is None:
            blk = g[p[0]:p[0] + s, p[1]:p[1] + s]
            if (blk == 6).any() and (blk == 7).any():
                for b in learned:
                    if (p, b) not in M['ttried'] and (p, b, m) not in M['bad']:
                        tele_ex = (st, ('click', b))
                        break
        for name, dr, dc in DIRS:
            if (p, name, m) in M['blocked']:
                continue
            np2 = (p[0] + dr * s, p[1] + dc * s)
            ns = (np2, m)
            if ns in par or not ok(m, np2):
                continue
            par[ns] = (st, ('move', name))
            dq.append(ns)
        for b in learned:
            if (p, b, m) in M['bad']:
                continue
            m2 = m ^ bits[b]
            p2 = M['tele'].get((p, b), p)
            ns = (p2, m2)
            if ns in par or not ok(m2, p2):
                continue
            par[ns] = (st, ('click', b))
            dq.append(ns)

    target = found or explore or tele_ex
    action = None
    if target is not None:
        st, final = target
        if st == start:
            action = final
        else:
            a = None
            while par[st] is not None:
                st, a = par[st]
            action = a
    if action is None:
        h = hash(play.tobytes())
        order = [b for b in buttons if b in M['dyn']] + [b for b in buttons if b not in M['dyn']]
        for b in order:
            if (h, b) not in M['pairs']:
                M['pairs'].add((h, b))
                action = ('click', b)
                break
    if action is None:
        for st2 in par:
            if st2[1] == cur_mask and st2[0] not in M['vis']:
                a = None
                while par[st2] is not None:
                    st2, a = par[st2]
                action = a
                break
    if action is None:
        if buttons and rng.random() < 0.6:
            action = ('click', rng.choice(buttons))
        else:
            action = ('move', rng.choice(["ACTION1", "ACTION2", "ACTION3", "ACTION4"]))
    if action[0] == 'click':
        b = action[1]
        M['prev'] = ('click', pos, b, play, cur_mask)
        return ("ACTION6", int(b[0]), int(b[1]))
    M['prev'] = ('move', pos, action[1], play, cur_mask)
    return (action[1], None, None)
