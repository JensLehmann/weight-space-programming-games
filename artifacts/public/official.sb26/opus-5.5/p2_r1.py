import time
import numpy as np

def _parse(obs):
    H, W = obs.shape
    targets = []
    for y in range(0, 47):
        x = 0
        while x <= W - 6:
            c = obs[y, x]
            if c not in (4, 5) and np.all(obs[y, x:x+6] == c) and np.all(obs[y+5, x:x+6] == c) \
               and np.all(obs[y:y+6, x] == c) and np.all(obs[y:y+6, x+5] == c) and obs[y+2, x+2] == 5:
                targets.append((y, x, int(c))); x += 6
            else:
                x += 1
    targets.sort()
    frames = []
    for y in range(0, 44):
        x = 0
        while x < W:
            c = obs[y, x]
            if c in (4, 5, 0):
                x += 1; continue
            e = x
            while e < W and obs[y, e] == c:
                e += 1
            L = e - x
            if L >= 16 and (L - 4) % 6 == 0 and y + 9 < 53 and np.all(obs[y+9, x:e] == c) \
               and np.all(obs[y:y+10, x] == c) and np.all(obs[y:y+10, e-1] == c) and obs[y+1, x+1] == 4:
                frames.append((y, x, (L - 4) // 6, int(c)))
            x = e
    frames.sort()
    def classify(px, py):
        a = int(obs[py+1, px+1]); ctr = int(obs[py+2, px+2])
        if a == 4 or a == 0:
            return None
        return ('C', a) if ctr == a else ('P', a)
    slots = {}
    for fi, (fy, fx, n, c) in enumerate(frames):
        for i in range(n):
            px, py = fx + 2 + 6 * i, fy + 2
            slots[(fi, i)] = ((px, py), classify(px, py))
    inv = {}
    for yy in range(54, 59):
        if yy + 4 >= H:
            break
        for s in range(1, W - 4):
            c = obs[yy+1, s]
            if c in (4, 0):
                continue
            if np.all(obs[yy+1, s:s+4] == c) and obs[yy+1, s-1] in (4, 0) and (s+4 >= W or obs[yy+1, s+4] in (4, 0)) \
               and np.all(obs[yy+4, s:s+4] == c) and obs[yy, s] in (4, 0):
                inv[(s-1, yy)] = classify(s-1, yy)
    return [t[2] for t in targets], frames, slots, inv

def _solve(T, frames, slots, inv, banned):
    nf = [f[2] for f in frames]
    first = {}
    for i, f in enumerate(frames):
        first.setdefault(f[3], i)
    cur = {k: v[1] for k, v in slots.items()}
    rem = {}
    for v in list(cur.values()) + list(inv.values()):
        if v is not None:
            rem[v] = rem.get(v, 0) + 1
    def sim(assign):
        stack = [(0, 0)]; t = 0; steps = 0
        while True:
            steps += 1
            if steps > 400:
                return None
            f, i = stack[-1]
            if (f, i) not in assign:
                return ('need', (f, i), T[t])
            tok = assign[(f, i)]
            if tok is None:
                return None
            if tok[0] == 'C':
                if tok[1] != T[t]:
                    return None
                t += 1
                if t == len(T):
                    return ('win',)
                while True:
                    f, i = stack[-1]; i += 1
                    if i < nf[f]:
                        stack[-1] = (f, i); break
                    stack.pop()
                    if not stack:
                        return None
            else:
                g = first.get(tok[1])
                if g is None:
                    return None
                if i == 0 and (f, 0) in stack[:-1] and stack[-2][1] == 0:
                    return None
                stack.append((g, 0))
    best = [None, 10**9]
    t0 = time.time()
    def cost(a):
        return sum(1 for k, v in a.items() if cur.get(k) != v)
    def dfs(assign):
        if time.time() - t0 > 3:
            return
        c = cost(assign)
        if c >= best[1]:
            return
        r = sim(assign)
        if r is None:
            return
        if r[0] == 'win':
            key = tuple(sorted(assign.items()))
            if key not in banned:
                best[0] = dict(assign); best[1] = c
            return
        slot, tc = r[1], r[2]
        cands = [k for k, v in rem.items() if v > 0 and (k == ('C', tc) or k[0] == 'P')]
        cands.sort(key=lambda k: (k != cur.get(slot), k[0] == 'P', k))
        for tok in cands:
            assign[slot] = tok; rem[tok] -= 1
            dfs(assign)
            rem[tok] += 1; del assign[slot]
    if T and frames:
        dfs({})
    return best[0]

def act(obs, available, memory):
    obs = np.asarray(obs)
    T, frames, slots, inv = _parse(obs)
    positions = {}
    for k, (pos, tok) in slots.items():
        positions[pos] = tok
    for pos, tok in inv.items():
        positions[pos] = tok
    sel = None
    for (px, py) in positions:
        if py + 5 < 64 and px + 5 < 64 and obs[py, px] == 0 and obs[py+5, px+5] == 0:
            sel = (px, py); break
    pend = memory.get('pending')
    if sel is not None:
        if pend and pend[0] == sel:
            memory['pending'] = None
            dx, dy = pend[1]
            return ("ACTION6", dy + 2, dx + 2)
        memory['pending'] = None
        return ("ACTION6", sel[1] + 2, sel[0] + 2)
    memory['pending'] = None
    banned = memory.setdefault('banned', set())
    plan = memory.get('plan')
    if plan is not None and any(k not in slots for k in plan):
        plan = None
    if plan is None:
        plan = _solve(T, frames, slots, inv, banned)
        memory['plan'] = plan
    if plan is None:
        return ("ACTION5", None, None)
    cur = {k: v[1] for k, v in slots.items()}
    wrong = sorted(k for k, v in plan.items() if cur.get(k) != v)
    if not wrong:
        key = tuple(sorted(plan.items()))
        if memory.get('evaluated') == key:
            banned.add(key)
            memory['plan'] = None
            memory['evaluated'] = None
            plan = _solve(T, frames, slots, inv, banned)
            memory['plan'] = plan
            if plan is None:
                return ("ACTION5", None, None)
            wrong = sorted(k for k, v in plan.items() if cur.get(k) != v)
            if not wrong:
                memory['evaluated'] = tuple(sorted(plan.items()))
                return ("ACTION5", None, None)
        else:
            memory['evaluated'] = key
            return ("ACTION5", None, None)
    s = wrong[0]
    tok = plan[s]
    dest = slots[s][0]
    src = None
    for pos, t in inv.items():
        if t == tok:
            src = pos; break
    if src is None:
        for k, (pos, t) in sorted(slots.items()):
            if k == s or t != tok:
                continue
            if k in plan and plan[k] == t:
                continue
            src = pos; break
    if src is None:
        memory['plan'] = None
        return ("ACTION5", None, None)
    memory['pending'] = (src, dest)
    return ("ACTION6", src[1] + 2, src[0] + 2)
