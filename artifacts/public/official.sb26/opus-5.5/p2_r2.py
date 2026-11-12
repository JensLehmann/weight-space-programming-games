import numpy as np

def _parse(obs):
    H, W = obs.shape
    targets = []
    for y in range(0, 21):
        for x in range(0, 59):
            c = obs[y, x]
            if c in (4, 5):
                continue
            if not (np.all(obs[y, x:x+6] == c) and np.all(obs[y+5, x:x+6] == c)
                    and np.all(obs[y:y+6, x] == c) and np.all(obs[y:y+6, x+5] == c)):
                continue
            if np.all(obs[y+1:y+5, x+1:x+5] == 5):
                targets.append((y, x, int(c)))
    targets.sort()
    T = [t[2] for t in targets]
    frames = []
    for y in range(8, 44):
        for x in range(0, 55):
            c = obs[y, x]
            if c in (4, 0, 5):
                continue
            if x > 0 and obs[y, x-1] == c:
                continue
            if obs[y-1, x] == c:
                continue
            if not np.all(obs[y:y+10, x] == c):
                continue
            x2 = x
            while x2 + 1 < W and obs[y, x2+1] == c:
                x2 += 1
            w = x2 - x + 1
            if w < 10 or (w - 4) % 6 != 0:
                continue
            if not (np.all(obs[y:y+10, x2] == c) and np.all(obs[y+9, x:x2+1] == c)):
                continue
            if obs[y+1, x+1] != 4:
                continue
            n = (w - 4) // 6
            slots = []
            for i in range(n):
                sx, sy = x + 2 + 6 * i, y + 2
                if obs[sy+2, sx+2] == 2 and obs[sy+1, sx+1] == 4:
                    slots.append(('empty', -1, sx, sy))
                else:
                    cc = int(obs[sy+1, sx+1])
                    kind = 'item' if obs[sy+2, sx+2] == cc else 'call'
                    slots.append((kind, cc, sx, sy))
            frames.append({'y': y, 'x': x, 'color': int(c), 'slots': slots})
    frames.sort(key=lambda f: (f['y'], f['x']))
    inv = []
    x = 0
    while x < 61:
        c = obs[57, x]
        if c != 4 and c != 2 and np.all(obs[57, x:x+4] == c) and np.all(obs[60, x:x+4] == c) \
                and np.all(obs[57:61, x] == c) and np.all(obs[57:61, x+3] == c):
            kind = 'item' if obs[58, x+1] == c else 'call'
            inv.append((kind, int(c), x - 1, 56))
            x += 4
        else:
            x += 1
    pert = False
    for x in range(0, 63):
        if obs[58, x] == 2 and obs[58, x+1] == 2 and obs[59, x] == 2 and obs[57, x] == 4:
            pert = True
    if np.any(obs[56, :] == 0) or np.any(obs[61, :] == 0):
        pert = True
    for f in frames:
        for (k, c, sx, sy) in f['slots']:
            if np.all(obs[sy, sx:sx+6] == 0):
                pert = True
    return T, frames, inv, pert

def _simulate(T, frames, inv, assign):
    if not T or not frames:
        return ('fail',)
    def content(f, i):
        if (f, i) in assign:
            k, c, _, _ = inv[assign[(f, i)]]
            return k, c
        s = frames[f]['slots'][i]
        return s[0], s[1]
    stack = [[0, 0]]
    p = 0
    filled = False
    ret = False
    for _ in range(2000):
        f, i = stack[-1]
        kind, c = content(f, i)
        if not filled and kind == 'item':
            if c == T[p]:
                filled = True
                continue
            return ('fail',)
        if p == len(T) - 1 and filled:
            return ('win',)
        if ret or kind == 'item':
            ret = False
            i += 1
            if i < len(frames[f]['slots']):
                stack[-1] = [f, i]
                p += 1
                filled = False
            elif len(stack) > 1:
                stack.pop()
                ret = True
            else:
                return ('fail',)
        elif kind == 'call':
            if i == 0 and len(stack) > 1 and [f, 0] in stack[:-1] and stack[-2][1] == 0:
                return ('fail',)
            g = None
            for j, fr in enumerate(frames):
                if fr['color'] == c:
                    g = j
                    break
            if g is None:
                return ('fail',)
            stack.append([g, 0])
        else:
            return ('need', (f, i), p)
    return ('fail',)

def _solve(T, frames, inv):
    cnt = [0]
    def dfs(assign, used):
        cnt[0] += 1
        if cnt[0] > 20000:
            return None
        r = _simulate(T, frames, inv, assign)
        if r[0] == 'win':
            return dict(assign)
        if r[0] == 'fail':
            return None
        slot, p = r[1], r[2]
        seen = set()
        for idx, (k, c, _, _) in enumerate(inv):
            if idx in used or (k, c) in seen:
                continue
            seen.add((k, c))
            if k == 'item' and c != T[p]:
                continue
            assign[slot] = idx
            used.add(idx)
            res = dfs(assign, used)
            if res is not None:
                return res
            del assign[slot]
            used.discard(idx)
        return None
    return dfs({}, set())

def act(obs, available, memory):
    obs = np.asarray(obs)
    plan = memory.get('plan')
    if plan:
        a = plan.pop(0)
        return a
    T, frames, inv, pert = _parse(obs)
    resets = memory.get('resets', 0)
    if (pert or memory.get('executed')) and resets < 2 and 'RESET' in available:
        memory['resets'] = resets + 1
        memory['executed'] = False
        return ('RESET', None, None)
    sol = None
    try:
        sol = _solve(T, frames, inv)
    except Exception:
        sol = None
    plan = []
    if sol is not None:
        for (f, i), idx in sol.items():
            k, c, ix, iy = inv[idx]
            s = frames[f]['slots'][i]
            plan.append(('ACTION6', iy + 2, ix + 2))
            plan.append(('ACTION6', s[3] + 2, s[2] + 2))
    plan.append(('ACTION5', None, None))
    memory['executed'] = True
    a = plan.pop(0)
    memory['plan'] = plan
    return a
