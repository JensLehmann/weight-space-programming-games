import numpy as np

def _parse(obs):
    H, W = obs.shape
    seen = np.zeros((H, W), bool); acts = []
    for y in range(H - 1):
        for x in range(W):
            if obs[y, x] == 2 and not seen[y, x]:
                st = [(y, x)]; seen[y, x] = True; ys = []; xs = []
                while st:
                    a, b = st.pop(); ys.append(a); xs.append(b)
                    for c, d in ((a+1, b), (a-1, b), (a, b+1), (a, b-1)):
                        if 0 <= c < H-1 and 0 <= d < W and not seen[c, d] and obs[c, d] == 2:
                            seen[c, d] = True; st.append((c, d))
                y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
                w = x1-x0+1; h = y1-y0+1
                if w < 5 or h < 5: continue
                cy = (y0+y1)//2; cx = (x0+x1)//2; i = len(acts)
                if w == h: acts.append((cy, cx, 'rot', i))
                elif w > h:
                    acts.append((cy, x0+2, 'sh', i+1)); acts.append((cy, x1-2, 'ex', i))
                else:
                    acts.append((y0+2, cx, 'sh', i+1)); acts.append((y1-2, cx, 'ex', i))
    return acts

def _score(obs):
    d = (obs[:63] == 13); H, W = d.shape; t = []; dia = set()
    for y in range(1, H-1):
        for x in range(1, W-1):
            if d[y-1, x] and d[y+1, x] and d[y, x-1] and d[y, x+1]:
                t.append((y, x)); dia.update([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
    m = [(y, x) for y, x in zip(*np.nonzero(d)) if (y, x) not in dia]
    s = 0
    for ty, tx in t:
        if d[ty, tx]: continue
        s += min(abs(ty-a)+abs(tx-b) for a, b in m) if m else 100
    return s

def act(obs, available, memory):
    obs = np.asarray(obs)
    if 'acts' not in memory:
        memory.update(acts=_parse(obs), res={}, vis={}, pend=[], prev=None)
    acts = memory['acts']
    if not acts: return ("ACTION6", 32, 32)
    h = hash(obs[:63].tobytes()); s = _score(obs)
    memory['vis'][h] = memory['vis'].get(h, 0) + 1
    res = memory['res'].setdefault(h, {})
    p = memory['prev']; memory['prev'] = None
    if p is not None:
        ph, ps, ai = p
        memory['res'].setdefault(ph, {})[ai] = (h, s)
        if s > ps and h != ph:
            k = acts[ai]
            if k[2] != 'rot':
                memory['pend'] = [k[3]]
    if memory['pend']:
        ai = memory['pend'].pop(0)
        return ("ACTION6", int(acts[ai][0]), int(acts[ai][1]))
    vis = memory['vis']; best = None; bv = None
    for ai in range(len(acts)):
        if ai in res:
            nh, ns = res[ai]
            if nh == h:
                v = 1e6
            else:
                v = ns * 10 + vis.get(nh, 0) * 60 + (500 if ns > s else 0)
        else:
            v = s * 10 - 8
        v += np.random.rand() * 3
        if bv is None or v < bv: bv = v; best = ai
    memory['prev'] = (h, s, best)
    return ("ACTION6", int(acts[best][0]), int(acts[best][1]))
