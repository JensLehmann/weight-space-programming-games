import numpy as np
import random

def _score(obs):
    d = (obs[:63] == 13)
    p = np.pad(d, 1)
    up = p[:-2, 1:-1]; dn = p[2:, 1:-1]; lf = p[1:-1, :-2]; rt = p[1:-1, 2:]
    cen = up & dn & lf & rt
    n8 = (up.astype(int) + dn + lf + rt + p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:])
    dots = d & (n8 == 0)
    tc = np.argwhere(cen & ~d)
    dp = np.argwhere(dots)
    if len(tc) == 0:
        return 0
    if len(dp) == 0:
        return 1000 * len(tc)
    return sum(int(np.min(np.abs(dp - t).sum(axis=1))) for t in tc)

def _buttons(obs):
    m = (obs[:63] == 2)
    seen = np.zeros_like(m)
    acts = []
    H, W = m.shape
    for y in range(H):
        for x in range(W):
            if m[y, x] and not seen[y, x]:
                st = [(y, x)]; seen[y, x] = True
                y0 = y1 = y; x0 = x1 = x
                while st:
                    cy, cx = st.pop()
                    y0 = min(y0, cy); y1 = max(y1, cy); x0 = min(x0, cx); x1 = max(x1, cx)
                    for ny, nx in ((cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)):
                        if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; st.append((ny, nx))
                w = x1 - x0 + 1; h = y1 - y0 + 1
                if w < 5 or h < 5:
                    continue
                if w == h:
                    acts.append(['T', (y0 + h // 2, x0 + w // 2), -1])
                else:
                    if w > h:
                        cy = (y0 + y1) // 2
                        ext = (cy, x1 - 1); shr = (cy, x0 + 1)
                    else:
                        cx = (x0 + x1) // 2
                        ext = (y1 - 1, cx); shr = (y0 + 1, cx)
                    i = len(acts)
                    acts.append(['E', ext, i + 1])
                    acts.append(['S', shr, i])
    return acts

def act(obs, available, memory):
    obs = np.asarray(obs)
    if 'acts' not in memory:
        memory['acts'] = _buttons(obs)
        memory['tried'] = {}
        memory['eff'] = {}
        memory['prev'] = None
        memory['rng'] = random.Random(0)
    acts = memory['acts']
    rng = memory['rng']
    if not acts:
        return ('ACTION6', 32, 32)
    key = hash(obs[:63].tobytes())
    s = _score(obs)
    tried = memory['tried']
    eff = memory['eff']
    prev = memory['prev']
    if prev is not None:
        pk, ps, pa, was_undo = prev
        tried.setdefault(pk, set()).add(pa)
        if key != pk:
            eff.setdefault(pk, set()).add(pa)
        if key != pk and not was_undo and s > ps and acts[pa][0] in 'ES':
            inv = acts[pa][2]
            tried.setdefault(key, set()).add(inv)
            memory['prev'] = (key, s, inv, True)
            y, x = acts[inv][1]
            return ('ACTION6', int(y), int(x))
    t = tried.setdefault(key, set())
    cand = [i for i in range(len(acts)) if i not in t]
    if cand:
        rng.shuffle(cand)
        cand.sort(key=lambda i: {'E': 0, 'T': 1, 'S': 2}[acts[i][0]])
        a = cand[0]
    else:
        e = list(eff.get(key, set()))
        if e and rng.random() < 0.8:
            a = rng.choice(e)
        else:
            a = rng.randrange(len(acts))
    memory['prev'] = (key, s, a, False)
    y, x = acts[a][1]
    return ('ACTION6', int(y), int(x))
