import numpy as np

VALID_L = {6 * k + 4 for k in range(1, 10)}


def parse(obs):
    H, W = obs.shape
    tg = []
    for y in range(0, 24):
        for x in range(0, W - 5):
            c = obs[y, x]
            if c in (4, 5):
                continue
            if (np.all(obs[y, x:x+6] == c) and np.all(obs[y+5, x:x+6] == c)
                    and np.all(obs[y:y+6, x] == c) and np.all(obs[y:y+6, x+5] == c)
                    and np.all(obs[y+1:y+5, x+1:x+5] == 5)):
                tg.append((y, x, int(c)))
    tg.sort()
    T = [c for _, _, c in tg]
    frames = []
    for y in range(8, 44):
        x = 0
        while x < W:
            c = obs[y, x]
            if c != 4 and c != 0 and (x == 0 or obs[y, x-1] != c):
                L = 0
                while x + L < W and obs[y, x+L] == c:
                    L += 1
                if (L in VALID_L and y + 9 < 53 and np.all(obs[y+9, x:x+L] == c)
                        and np.all(obs[y:y+10, x] == c) and np.all(obs[y:y+10, x+L-1] == c)
                        and obs[y+1, x+1] == 4):
                    frames.append((y, x, int(c), (L - 4) // 6))
                    x += L
                    continue
            x += 1
    frames.sort()
    fixed = {}
    empty = []
    for fi, (fy, fx, fc, k) in enumerate(frames):
        for i in range(k):
            col = fx + 3 + 6 * i
            a = obs[fy+3, col]
            cen = obs[fy+4, col+1]
            if a == 4:
                if cen == 2:
                    empty.append((fi, i))
            else:
                fixed[(fi, i)] = ('S' if cen == a else 'H', int(a))
    pal = []
    x = 0
    while x < W - 3:
        c = obs[57, x]
        if c != 4 and (x == 0 or obs[57, x-1] == 4) and np.all(obs[57, x:x+4] == c) and obs[60, x] == c:
            typ = 'S' if obs[58, x+1] == c else 'H'
            pal.append((typ, int(c), 58, x + 1))
            x += 4
            continue
        x += 1
    return T, frames, fixed, empty, pal


def solutions(T, frames, fixed, empty, pal):
    N = len(T)
    K = [f[3] for f in frames]
    fcol = {}
    for i, f in enumerate(frames):
        fcol.setdefault(f[2], i)
    emptyset = set(empty)
    counts = {}
    for typ, c, _, _ in pal:
        counts[(typ, c)] = counts.get((typ, c), 0) + 1
    budget = [40000]

    def sim(assign):
        stack = [[0, 0]]
        t = 0
        steps = 0
        while True:
            steps += 1
            if steps > 3000:
                return ('fail',)
            f, s = stack[-1]
            if s >= K[f]:
                if len(stack) > 1:
                    stack.pop()
                    stack[-1][1] += 1
                    continue
                return ('fail',)
            key = (f, s)
            it = fixed.get(key) or assign.get(key)
            if it is None:
                if key in emptyset:
                    return ('need', key, t)
                return ('fail',)
            typ, c = it
            if typ == 'S':
                if c == T[t]:
                    t += 1
                    if t == N:
                        return ('win',)
                    stack[-1][1] += 1
                else:
                    return ('fail',)
            else:
                g = fcol.get(c)
                if g is None:
                    return ('fail',)
                if s == 0 and len(stack) > 1 and any(e[0] == f and e[1] == 0 for e in stack[:-1]) and stack[-2][1] == 0:
                    return ('fail',)
                if len(stack) > 40:
                    return ('fail',)
                stack.append([g, 0])

    def dfs(assign):
        budget[0] -= 1
        if budget[0] < 0:
            return
        r = sim(assign)
        if r[0] == 'win':
            yield dict(assign)
            return
        if r[0] != 'need':
            return
        key, t = r[1], r[2]
        opts = []
        if counts.get(('S', T[t]), 0) > 0:
            opts.append(('S', T[t]))
        for (typ, c), n in counts.items():
            if typ == 'H' and n > 0 and c in fcol:
                opts.append((typ, c))
        for o in opts:
            counts[o] -= 1
            assign[key] = o
            for sol in dfs(assign):
                yield sol
            del assign[key]
            counts[o] += 1

    if N == 0 or not frames:
        return
    for sol in dfs({}):
        yield sol


def plan(obs, attempt):
    T, frames, fixed, empty, pal = parse(obs)
    sol = None
    idx = 0
    try:
        for s in solutions(T, frames, fixed, empty, pal):
            sol = s
            if idx >= attempt:
                break
            idx += 1
    except Exception:
        sol = None
    acts = []
    if sol is None:
        return [('ACTION5', None, None)]
    avail = list(pal)
    for (fi, si), (typ, c) in sorted(sol.items()):
        pick = None
        for j, p in enumerate(avail):
            if p[0] == typ and p[1] == c:
                pick = j
                break
        if pick is None:
            continue
        p = avail.pop(pick)
        fy, fx, fc, k = frames[fi]
        acts.append(('ACTION6', p[2], p[3]))
        acts.append(('ACTION6', fy + 4, fx + 4 + 6 * si))
    acts.append(('ACTION5', None, None))
    return acts


def act(obs, available, memory):
    obs = np.asarray(obs)
    phase = memory.get('phase', 'undo')
    if phase == 'undo':
        prev = memory.get('prev')
        cnt = memory.get('ucnt', 0)
        if prev is not None and (np.array_equal(prev, obs) or cnt > 40):
            memory['phase'] = 'run'
            memory['queue'] = plan(obs, memory.get('attempt', 0))
            memory['prev'] = None
        else:
            memory['prev'] = obs.copy()
            memory['ucnt'] = cnt + 1
            return ('ACTION7', None, None)
    q = memory.get('queue', [])
    if not q:
        memory['phase'] = 'undo'
        memory['ucnt'] = 0
        memory['prev'] = obs.copy()
        memory['attempt'] = memory.get('attempt', 0) + 1
        return ('ACTION7', None, None)
    a = q.pop(0)
    if a[0] == 'ACTION5' and not q:
        memory['phase'] = 'undo'
        memory['ucnt'] = 0
        memory['prev'] = None
        memory['attempt'] = memory.get('attempt', 0) + 1
    if a[0] == 'ACTION6':
        return ('ACTION6', int(a[1]), int(a[2]))
    return (a[0], None, None)
