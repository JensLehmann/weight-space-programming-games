import math
import time
import numpy as np

_SIZES = (1, 2, 3, 4, 5, 7, 8, 9, 10)
_COLORS = (10, 6, 15, 11, 12, 8, 9, 7, 14)
_ECOLORS = (0, 7, 14, 13)
_ESHAPES = {
    1: ((2, 0), (1, 1), (3, 1), (0, 2), (4, 2),
        (1, 3), (2, 3), (3, 3)),
    2: ((2, 0), (1, 1), (3, 1), (0, 2), (2, 2),
        (4, 2), (1, 3), (3, 3)),
    3: ((0, 0), (2, 0), (4, 0), (1, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
        (1, 3), (2, 3), (3, 3)),
}
_PAD = tuple(
    (x, y) for y in range(9) for x in range(9)
    if (2 <= x <= 6 if y in (0, 8)
        else 1 <= x <= 7 if y in (1, 7) else True)
)


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    H, W = mask.shape
    result = []
    for yy, xx in np.argwhere(mask):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(x, y)]
        comp = []
        while stack:
            a, b = stack.pop()
            comp.append((a, b))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    u, v = a + dx, b + dy
                    if (0 <= u < W and 0 <= v < H
                            and mask[v, u] and not seen[v, u]):
                        seen[v, u] = True
                        stack.append((u, v))
        result.append(comp)
    return result


def _counts(mask, s):
    a = np.pad(mask.astype(np.int16), ((1, 0), (1, 0)))
    a = a.cumsum(0).cumsum(1)
    return a[s:, s:] - a[:-s, s:] - a[s:, :-s] + a[:-s, :-s]


def _read_goals(obs):
    mask = (obs[:10] != 4) & (obs[:10] != 5)
    mask[:, :24] = False
    nf, ne = {}, {}
    for comp in _components(mask):
        xs, ys = zip(*comp)
        x, y = min(xs), min(ys)
        w, h = max(xs)-x+1, max(ys)-y+1
        if w == h and w in _SIZES and len(comp) >= w*h*0.8:
            k = _SIZES.index(w)
            nf[k] = nf.get(k, 0)+1
        elif w == 5 and h == 4:
            points = {(a-x, b-y) for a, b in comp}
            k = min(_ESHAPES,
                    key=lambda t: len(points ^ set(_ESHAPES[t])))
            ne[k] = ne.get(k, 0)+1
    return nf, ne


def _read_pads(obs, fruits):
    evidence = ((obs == 9) | (obs == 2)).copy()
    evidence[:10] = False
    evidence[63] = False
    for k, x, y in fruits:
        if k == 6:
            evidence[y:y+8, x:x+8] = False
    pads = []
    for comp in _components(evidence):
        if len(comp) < 10:
            continue
        xs, ys = zip(*comp)
        lx, hx, ly, hy = min(xs), max(xs), min(ys), max(ys)
        if hx-lx > 12 or hy-ly > 12:
            continue
        best = None
        for y in range(max(10, hy-8), min(55, ly)+1):
            for x in range(max(0, hx-8), min(55, lx)+1):
                shape = {(x+dx, y+dy) for dx, dy in _PAD}
                hit = sum((a, b) in shape for a, b in comp)
                missing = sum(obs[y+dy, x+dx] == 5 for dx, dy in _PAD)
                score = 3*hit - 5*(len(comp)-hit) - missing
                if best is None or score > best[0]:
                    best = (score, x, y)
        if best and best[0] >= 18:
            p = (best[1], best[2])
            if not any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 3
                       for q in pads):
                pads.append(p)
    return tuple(pads)


def _read_scene(obs, known_pads=()):
    fruits = []
    for k, (s, color) in enumerate(zip(_SIZES, _COLORS)):
        mask = (obs == color).copy()
        mask[:10] = False
        if s > 1:
            mask[63] |= obs[63] == 0
        else:
            mask[63] = False
        counts = _counts(mask, s)
        occupied = np.zeros((64, 64), dtype=bool)
        for yy, xx in np.argwhere(counts == s*s):
            x, y = int(xx), int(yy)
            if y < 10 or y > 62 or occupied[y:y+s, x:x+s].any():
                continue
            fruits.append((k, x, y))
            occupied[y:y+s, x:x+s] = True

    pads = list(known_pads)
    for p in _read_pads(obs, fruits):
        if not any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 3 for q in pads):
            pads.append(p)

    pad_mask = np.zeros((64, 64), dtype=bool)
    for px, py in pads:
        for dx, dy in _PAD:
            if 0 <= py+dy < 64 and 0 <= px+dx < 64:
                pad_mask[py+dy, px+dx] = True

    opaque = np.zeros((64, 64), dtype=bool)
    for k, x, y in fruits:
        s = _SIZES[k]
        opaque[y:y+s, x:x+s] = True

    # Recover square pieces for which only a strip is visible.
    for iteration in range(2):
        added = False
        for k, (s, color) in enumerate(zip(_SIZES, _COLORS)):
            if s == 1:
                continue
            visible = (obs == color) & ~opaque
            visible[:10] = False
            visible[63] = False
            if k == 6:
                visible &= ~pad_mask
            if not visible.any():
                continue
            allowed = (obs == color) | opaque
            allowed[:10] = False
            allowed[63] |= obs[63] == 0
            full = _counts(allowed, s)
            evidence = _counts(visible, s)
            minimum = s if k in (7, 8) else 1
            ys, xs = np.nonzero((full == s*s) & (evidence >= minimum))
            candidates = sorted(
                ((-int(evidence[y, x]), int(y), int(x))
                 for y, x in zip(ys, xs))
            )
            for _, y, x in candidates:
                if y < 10 or y > 62:
                    continue
                new = visible[y:y+s, x:x+s] & ~opaque[y:y+s, x:x+s]
                if int(new.sum()) < minimum:
                    continue
                fruits.append((k, x, y))
                opaque[y:y+s, x:x+s] = True
                added = True
        if not added:
            break

    # Enemy pixels are below the solid pieces. Match visible portions of
    # their templates while allowing the rest to be hidden by a square.
    enemies = []
    covered = opaque.copy()
    covered[63] |= obs[63] == 0
    for k in (1, 2, 3):
        color = _ECOLORS[k]
        good = np.ones((51, 60), dtype=bool)
        evidence = np.zeros((51, 60), dtype=np.int16)
        shape = set(_ESHAPES[k])
        for dx, dy in shape:
            a = obs[10+dy:61+dy, dx:60+dx]
            c = covered[10+dy:61+dy, dx:60+dx]
            match = (a == color) & ~c
            good &= (a == color) | c
            evidence += match
        for dy in range(4):
            for dx in range(5):
                if (dx, dy) in shape:
                    continue
                a = obs[10+dy:61+dy, dx:60+dx]
                c = covered[10+dy:61+dy, dx:60+dx]
                good &= (a != color) | c
        ys, xs = np.nonzero(good & (evidence > 0))
        claimed = set()
        candidates = sorted(
            ((-int(evidence[y, x]), int(y)+10, int(x))
             for y, x in zip(ys, xs))
        )
        for _, y, x in candidates:
            points = {
                (x+dx, y+dy) for dx, dy in shape
                if not covered[y+dy, x+dx] and obs[y+dy, x+dx] == color
            }
            if not points - claimed:
                continue
            enemies.append((k, x, y))
            claimed.update(points)
    return (tuple(fruits), tuple(enemies)), tuple(pads)


def _compatible(obs, state, pads):
    img = np.full((64, 64), 5, dtype=np.int8)
    for x, y in pads:
        for dx, dy in _PAD:
            if 0 <= y+dy < 64 and 0 <= x+dx < 64:
                img[y+dy, x+dx] = 9
    f, e = state
    for k, x, y in e:
        for dx, dy in _ESHAPES[k]:
            if 0 <= x+dx < 64 and 0 <= y+dy < 64:
                img[y+dy, x+dx] = _ECOLORS[k]
    order = sorted(
        f, key=lambda a: 10 if a[0] == 0 else
        9 if a[0] == 1 else 2 if a[0] == 8 else 0
    )
    for k, x, y in order:
        s = _SIZES[k]
        img[max(y, 0):min(y+s, 64), max(x, 0):min(x+s, 64)] = _COLORS[k]

    bad = (obs != img) & (obs != 0) & (obs != 3)
    bad[:10] = False
    bad[63] = False
    bad &= ~((obs == 2) & (img == 9))
    # Downgrading preserves a sprite's old layer; allow either square
    # to be visible where two predicted squares overlap.
    for k, x, y in f:
        s = _SIZES[k]
        region = bad[y:y+s, x:x+s]
        region[obs[y:y+s, x:x+s] == _COLORS[k]] = False
    return not bad.any()


def _key(state):
    return tuple(sorted(state[0])), tuple(sorted(state[1]))


def _coarse(state):
    return (
        tuple(sorted((k, (x+1)//3, (y+1)//3) for k, x, y in state[0])),
        tuple(sorted((k, (x+1)//3, (y+1)//3) for k, x, y in state[1]))
    )


def _center(a, enemy=False):
    if enemy:
        return a[1]+2, a[2]+2
    s = _SIZES[a[0]]
    return a[1]+s//2, a[2]+s//2


def _overlap(a, b, ea=False, eb=False):
    wa, ha = (5, 4) if ea else (_SIZES[a[0]],)*2
    wb, hb = (5, 4) if eb else (_SIZES[b[0]],)*2
    return (a[1] < b[1]+wb and b[1] < a[1]+wa
            and a[2] < b[2]+hb and b[2] < a[2]+ha)


def _selected(a, x, y, enemy=False):
    w, h = (5, 4) if enemy else (_SIZES[a[0]],)*2
    dx = max(a[1]-x, 0, x-(a[1]+w-1))
    dy = max(a[2]-y, 0, y-(a[2]+h-1))
    return dx*dx+dy*dy <= 64


def _clamp(a, x, y, enemy=False):
    w, h = (5, 4) if enemy else (_SIZES[a[0]],)*2
    a[1] = max(0, min(64-w, int(x)))
    a[2] = max(10, min(64-h, 63, int(y)))


def _inpad(x, y, pads):
    return any(px <= x < px+9 and py <= y < py+9 for px, py in pads)


def _won(f, e, pads, nf, ne):
    cf, ce = {}, {}
    for a in f:
        if a is not None:
            x, y = _center(a)
            if _inpad(x, y, pads):
                cf[a[0]] = cf.get(a[0], 0)+1
    for a in e:
        if a is not None:
            x, y = _center(a, True)
            if _inpad(x, y, pads):
                ce[a[0]] = ce.get(a[0], 0)+1
    return (
        bool(nf or ne)
        and all(cf.get(k, 0) == n for k, n in nf.items())
        and all(ce.get(k, 0) == n for k, n in ne.items())
    )


def _can_make(pieces, goals, downgrade):
    counts = [0]*9
    for a in pieces:
        counts[a[0]] += 1
    for rank in sorted(goals, reverse=True):
        for unused in range(goals[rank]):
            if counts[rank]:
                counts[rank] -= 1
                continue
            higher = None
            if downgrade:
                for k in range(rank+1, 9):
                    if counts[k]:
                        higher = k
                        break
            if higher is not None:
                counts[higher] -= 1
                continue
            need = 1 << rank
            for k in range(rank-1, -1, -1):
                take = min(counts[k], need >> k)
                counts[k] -= take
                need -= take << k
            if need:
                return False
    return True


def _possible(state, nf, ne):
    f, e = state
    return (bool(f)
            and _can_make(f, nf, bool(e))
            and _can_make(e, ne, False))


def _groups(arr, indices, enemy):
    indices = [i for i in indices if i < len(arr) and arr[i] is not None]
    parent = {i: i for i in indices}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for p, i in enumerate(indices):
        for j in indices[p+1:]:
            if _overlap(arr[i], arr[j], enemy, enemy):
                if arr[i][0] != arr[j][0]:
                    return None
                parent[root(j)] = root(i)
    groups = {}
    for i in indices:
        groups.setdefault(root(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def _simulate(state, action, pads, nf, ne, mass_need):
    x, y = action
    f = [list(a) for a in state[0]]
    e = [list(a) for a in state[1]]
    sf = {i for i, a in enumerate(f) if _selected(a, x, y)}
    se = {i for i, a in enumerate(e) if _selected(a, x, y, True)}
    velocity = {}
    for i in se:
        if e[i][0] == 1:
            cx, cy = _center(e[i], True)
            dx, dy = x-cx, y-cy
            d = math.hypot(dx, dy)
            velocity[i] = (
                (round(3.4*dx/d), round(3.4*dy/d)) if d else (0, 0)
            )

    recoil = {}
    cooldown = {}
    merged = False

    def merge_enemies():
        groups = _groups(e, range(len(e)), True)
        if groups is None:
            return False
        for group in groups:
            k = e[group[0]][0]
            cx = sum(_center(e[i], True)[0] for i in group)//len(group)
            cy = sum(_center(e[i], True)[1] for i in group)//len(group)
            if k < 3:
                a = [k+1, 0, 0]
                _clamp(a, cx-2, cy-2, True)
                cooldown[len(e)] = 4
                e.append(a)
            for i in group:
                e[i] = None
                se.discard(i)
                cooldown.pop(i, None)
        return True

    for tick in range(24):
        freeze = any(4 <= r[0] < 8 and r[5] > 0 for r in recoil.values())

        for i in list(sf):
            if f[i] is None or i in recoil:
                continue
            a = f[i]
            cx, cy = _center(a)
            _clamp(a, a[1]+max(-4, min(4, x-cx)),
                   a[2]+max(-4, min(4, y-cy)))

        if not freeze:
            for i in list(se):
                if e[i] is None:
                    continue
                a = e[i]
                if i in velocity:
                    dx, dy = velocity[i]
                else:
                    cx, cy = _center(a, True)
                    dx = max(-4, min(4, x-cx))
                    dy = max(-4, min(4, y-cy))
                _clamp(a, a[1]+dx, a[2]+dy, True)

        for i in list(recoil):
            if f[i] is None:
                del recoil[i]
                continue
            r = recoil[i]
            stage, ox, oy, vx, vy, old_k, fx, fy = r
            if stage < 4:
                nx, ny = ox, oy+(0, -1, 0, 1)[stage]
            else:
                if stage == 4:
                    f[i][0] = old_k-1
                fx += vx*2.5
                fy += vy*2.5
                nx, ny = round(fx), round(fy)
                r[6], r[7] = fx, fy
            _clamp(f[i], nx, ny)
            r[0] += 1
            if r[0] >= (4 if old_k == 0 else 8):
                if old_k == 0:
                    f[i] = None
                    sf.discard(i)
                del recoil[i]

        for i in list(cooldown):
            cooldown[i] -= 1
            if cooldown[i] <= 0:
                del cooldown[i]

        if tick < 4 and not freeze and any(a is not None for a in f):
            for i, a in enumerate(e):
                if a is None or i in se or i in cooldown:
                    continue
                cx, cy = _center(a, True)
                best = None
                best_d = 10**9
                for j, b in enumerate(f):
                    if b is None or j in recoil:
                        continue
                    bx, by = _center(b)
                    d = (bx-cx)**2+(by-cy)**2
                    if d < best_d:
                        best_d, best = d, (bx, by)
                if best is not None:
                    step = 2 if a[0] == 3 else 1
                    dx = step*((best[0] > cx)-(best[0] < cx))
                    dy = step*((best[1] > cy)-(best[1] < cy))
                    _clamp(a, a[1]+dx, a[2]+dy, True)

            inactive = set(cooldown)
            for i, a in enumerate(e):
                if a is None or i in se or i in inactive:
                    continue
                for j, b in enumerate(f):
                    if b is None or j in recoil:
                        continue
                    if not _overlap(a, b, True, False):
                        continue
                    bx, by = _center(b)
                    ax, ay = _center(a, True)
                    dx, dy = bx-ax, by-ay
                    d = math.hypot(dx, dy)
                    vx, vy = (dx/d, dy/d) if d else (0.0, -1.0)
                    recoil[j] = [
                        0, b[1], b[2], vx, vy, b[0],
                        float(b[1]), float(b[2])
                    ]
                    cooldown[i] = 9
            if not merge_enemies():
                return None, False

        if tick >= 3 and not merged:
            merged = True
            if not merge_enemies():
                return None, False
            groups = _groups(f, sf, False)
            if groups is None:
                return None, False
            for group in groups:
                k = f[group[0]][0]
                cx = sum(_center(f[i])[0] for i in group)//len(group)
                cy = sum(_center(f[i])[1] for i in group)//len(group)
                if k < 8:
                    a = [k+1, 0, 0]
                    s = _SIZES[k+1]
                    _clamp(a, cx-s//2, cy-s//2)
                    f.append(a)
                for i in group:
                    f[i] = None
                    sf.discard(i)
                    recoil.pop(i, None)

        if _won(f, e, pads, nf, ne):
            result = (tuple(tuple(a) for a in f if a is not None),
                      tuple(tuple(a) for a in e if a is not None))
            return result, True
        if tick >= 3 and not recoil:
            break

    result = (tuple(tuple(a) for a in f if a is not None),
              tuple(tuple(a) for a in e if a is not None))
    if (not result[0]
            or sum(1 << a[0] for a in result[0]) < mass_need
            or not _possible(result, nf, ne)):
        return None, False
    return result, False


def _pad_distance(cx, cy, pads):
    if not pads:
        return 40.0
    return min(max(px-cx, cx-px-8, py-cy, cy-py-8, 0)
               for px, py in pads)


def _pool(pieces, enemy, max_rank):
    nodes = []
    buckets = {}
    originals = []
    for i, a in enumerate(pieces):
        cx, cy = _center(a, enemy)
        node = (a[0], float(cx), float(cy), 1 << i, 0.0)
        nodes.append(node)
        originals.append(node)
        buckets.setdefault(a[0], []).append(node)
    low = 1 if enemy else 0
    for rank in range(low, max_rank):
        bucket = buckets.get(rank, [])[:]
        while len(bucket) >= 2:
            best = None
            for i in range(len(bucket)):
                for j in range(i+1, len(bucket)):
                    a, b = bucket[i], bucket[j]
                    d = max(abs(a[1]-b[1]), abs(a[2]-b[2]))
                    score = d+0.4*(a[4]+b[4])
                    if best is None or score < best[0]:
                        best = (score, i, j, d)
            _, i, j, d = best
            a, b = bucket[i], bucket[j]
            cx, cy = (a[1]+b[1])/2, (a[2]+b[2])/2
            bits = a[3] | b[3]
            obstruction = 0.0
            if not enemy and d <= 28 and a[4] == 0 and b[4] == 0:
                for k, ox, oy, obits, unused in originals:
                    if k == rank or obits & bits:
                        continue
                    clearance = max(abs(ox-cx), abs(oy-cy))
                    obstruction += max(
                        0.0, 7.0+_SIZES[k]//2-clearance
                    )/5.0
            cost = (a[4]+b[4]+0.85
                    + d/(15 if enemy else 17)+obstruction)
            node = (rank+1, cx, cy, bits, cost)
            nodes.append(node)
            buckets.setdefault(rank+1, []).append(node)
            bucket.pop(j)
            bucket.pop(i)
    return nodes


def _assignment(options):
    if not options:
        return 0.0
    options.sort(key=len)
    best = [float("inf")]

    def rec(i, mask, cost):
        if cost >= best[0]:
            return
        if i == len(options):
            best[0] = cost
            return
        count = 0
        for val, bits in options[i]:
            if not bits & mask:
                rec(i+1, mask | bits, cost+val)
                count += 1
                if count >= 7:
                    break

    rec(0, 0, 0.0)
    if best[0] == float("inf"):
        return 35.0+sum(o[0][0] if o else 12.0 for o in options)
    return best[0]


def _heuristic(state, pads, nf, ne, mass_need):
    f, e = state
    fc = [_center(a) for a in f]
    ec = [_center(a, True) for a in e]
    pool = _pool(f, False, max(nf, default=0))
    opts = []
    for rank, count in nf.items():
        choices = []
        for k, cx, cy, bits, assembly in pool:
            if k < rank or (k > rank and not e):
                continue
            cost = 1.55*assembly
            if k > rank:
                de = min(max(abs(cx-ex), abs(cy-ey)) for ex, ey in ec)
                cost += 4.2*(k-rank)+max(0, de-7)/4.0
            cost += _pad_distance(cx, cy, pads)/8.5
            if k == rank and assembly == 0 and e:
                de = min(max(abs(cx-ex), abs(cy-ey)) for ex, ey in ec)
                cost += max(0, 12-de)*0.11
            choices.append((cost, bits))
        choices.sort()
        opts.extend([choices]*count)
    numeric = _assignment(opts)

    epool = _pool(e, True, max(ne, default=1))
    opts = []
    ready = all(sum(a[0] == k for a in f) >= n for k, n in nf.items())
    for rank, count in ne.items():
        choices = []
        for k, cx, cy, bits, assembly in epool:
            if k != rank:
                continue
            delivery = _pad_distance(cx, cy, pads)/10.5
            if ready and pads:
                values = []
                for px, py in pads:
                    d = max(px-cx, cx-px-8, py-cy, cy-py-8, 0)/10.5
                    danger = 0.0
                    for a, (fx, fy) in zip(f, fc):
                        if a[0] in nf:
                            near = max(abs(fx-(px+4)), abs(fy-(py+4)))
                            danger += max(0, 12-near)*0.055
                    values.append(d+danger)
                delivery = min(values)
            choices.append((
                1.2*assembly+(0.95 if ready else 0.65)*delivery, bits
            ))
        choices.sort()
        opts.extend([choices]*count)
    enemy_cost = _assignment(opts)

    penalty = 0.0
    mass = sum(1 << a[0] for a in f)
    if mass <= mass_need and e:
        for fx, fy in fc:
            d = min(max(abs(fx-ex), abs(fy-ey)) for ex, ey in ec)
            penalty += max(0, 11-d)*0.09
    for pieces, goals, is_enemy in ((f, nf, False), (e, ne, True)):
        inside = {}
        for a in pieces:
            cx, cy = _center(a, is_enemy)
            if _inpad(cx, cy, pads):
                inside[a[0]] = inside.get(a[0], 0)+1
        for k, n in goals.items():
            penalty += 4.0*max(0, inside.get(k, 0)-n)
    return numeric+enemy_cost+penalty


def _actions(state, pads):
    f, e = state
    actions, seen = [], set()

    def add(x, y):
        p = (max(0, min(63, int(round(x)))),
             max(10, min(62, int(round(y)))))
        if p not in seen:
            seen.add(p)
            actions.append(p)

    def toward(a, tx, ty, enemy=False):
        cx, cy = _center(a, enemy)
        dx, dy = tx-cx, ty-cy
        d = math.hypot(dx, dy)
        if d == 0:
            add(cx, cy)
            return
        radius = (7.0 if enemy and a[0] == 1 else
                  8.0+(2 if enemy else _SIZES[a[0]]//2))
        length = min(d, radius)
        for unused in range(5):
            xx = round(cx+dx*length/d)
            yy = round(cy+dy*length/d)
            if _selected(a, xx, yy, enemy):
                add(xx, yy)
                return
            length -= 0.6
        add(cx, cy)

    for pieces, enemy in ((f, False), (e, True)):
        for a in pieces:
            cx, cy = _center(a, enemy)
            add(cx, cy)
            for px, py in pads:
                nx = max(px, min(px+8, cx))
                ny = max(py, min(py+8, cy))
                toward(a, nx, ny, enemy)
                toward(a, px+4, py+4, enemy)
                if _selected(a, px+4, py+4, enemy):
                    add(px+4, py+4)
        for i, a in enumerate(pieces):
            ax, ay = _center(a, enemy)
            peers = sorted(
                (b for j, b in enumerate(pieces)
                 if j != i and b[0] == a[0]),
                key=lambda b: max(abs(_center(b, enemy)[0]-ax),
                                  abs(_center(b, enemy)[1]-ay))
            )[:3]
            for b in peers:
                bx, by = _center(b, enemy)
                toward(a, bx, by, enemy)
                if max(abs(ax-bx), abs(ay-by)) <= 29:
                    mx, my = (ax+bx)/2, (ay+by)/2
                    add(mx, my)
                    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                        add(mx+dx, my+dy)

    for p in ((0, 10), (63, 10), (0, 62), (63, 62), (32, 10)):
        add(*p)

    for a in f:
        ax, ay = _center(a)
        for b in e:
            bx, by = _center(b, True)
            if max(abs(ax-bx), abs(ay-by)) <= 27:
                add((ax+bx)/2, (ay+by)/2)
                toward(a, bx, by)
                toward(b, ax, ay, True)

    for pieces, enemy in ((e, True), (f, False)):
        for a in pieces:
            cx, cy = _center(a, enemy)
            n = 20 if enemy and a[0] == 1 else 8
            for j in range(n):
                angle = 2*math.pi*j/n
                toward(a, cx+30*math.cos(angle),
                       cy+30*math.sin(angle), enemy)
    return actions[:300]


def _search(state, pads, nf, ne, visits, banned, deadline):
    mass_need = sum((1 << k)*n for k, n in nf.items())
    h0 = _heuristic(state, pads, nf, ne, mass_need)
    beam = [(state, (), h0)]
    best_path = ()
    best_value = h0
    seen = {_key(state): 0}
    heuristic_cache = {}
    width = 22 if state[1] else 18
    fallback = None

    for depth in range(1, 17):
        candidates = []
        for s, path, old_h in beam:
            local = []
            forbidden = banned.get(_key(s), ())
            for action in _actions(s, pads):
                if time.perf_counter() >= deadline:
                    break
                if action in forbidden:
                    continue
                ns, win = _simulate(s, action, pads, nf, ne, mass_need)
                if ns is None:
                    continue
                npth = path+(action,)
                if win:
                    return list(npth), True
                key = _key(ns)
                if key in seen and seen[key] <= depth:
                    continue
                seen[key] = depth
                h = heuristic_cache.get(key)
                if h is None:
                    h = _heuristic(ns, pads, nf, ne, mass_need)
                    heuristic_cache[key] = h
                score = h+min(4.0, 0.8*visits.get(key, 0))
                value = score+0.045*depth
                if fallback is None or value < fallback[0]:
                    fallback = (value, npth)
                if value < best_value-0.015:
                    best_value, best_path = value, npth
                local.append((score, ns, npth, h))
            local.sort(key=lambda v: v[0])
            used = set()
            kept = 0
            for item in local:
                ck = _coarse(item[1])
                if ck in used:
                    continue
                used.add(ck)
                candidates.append(item)
                kept += 1
                if kept >= 10:
                    break
            if time.perf_counter() >= deadline:
                break

        if not candidates or time.perf_counter() >= deadline:
            break
        candidates.sort(key=lambda v: v[0])
        beam = []
        used = set()
        for score, s, p, h in candidates:
            ck = _coarse(s)
            if ck in used:
                continue
            used.add(ck)
            beam.append((s, p, h))
            if len(beam) >= width:
                break
    if best_path:
        return list(best_path), False
    if fallback:
        return list(fallback[1]), False
    return [], False


def _ban_last(memory):
    if memory.get("last_kind") != 6:
        return
    state = memory.get("last_state")
    action = memory.get("last_action")
    if state is not None and action is not None:
        memory.setdefault("banned", {}).setdefault(_key(state), set()).add(action)


def _reset(memory):
    resets = memory.get("resets", 0)+1
    memory.clear()
    memory["resets"] = resets
    memory["last_kind"] = 0
    return "RESET", None, None


def _undo(memory, available):
    _ban_last(memory)
    previous = (memory.get("last_state")
                if memory.get("last_kind") == 6 else None)
    memory["predicted"] = previous
    memory["plan"] = []
    memory["solved_plan"] = False
    memory["last_kind"] = 7
    if "ACTION7" in available:
        return "ACTION7", None, None
    return _reset(memory)


def act(obs, available, memory):
    start = time.perf_counter()
    obs = np.asarray(obs)
    available = set(available)
    if "ACTION6" not in available:
        if "ACTION7" in available:
            return "ACTION7", None, None
        return "RESET", None, None

    nf, ne = _read_goals(obs)
    if nf or ne:
        memory["goals"] = (nf, ne)
    else:
        nf, ne = memory.get("goals", ({}, {}))

    body = obs[10:63].tobytes()
    bar = 0
    while bar < 64 and obs[63, bar] == 0:
        bar += 1

    same = (memory.get("last_kind") == 6
            and memory.get("last_body") == body)
    if same:
        _ban_last(memory)
        memory["failed_clicks"] = memory.get("failed_clicks", 0)+1
        memory["plan"] = []
        memory["solved_plan"] = False
    elif memory.get("last_kind") == 6:
        memory["failed_clicks"] = 0

    parsed, pads = _read_scene(obs, memory.get("pads", ()))
    memory["pads"] = pads
    predicted = memory.get("predicted")
    prediction_ok = (predicted is not None
                     and _compatible(obs, predicted, pads))
    if prediction_ok:
        state = predicted
        memory["model_misses"] = 0
    else:
        state = parsed
        memory["plan"] = []
        memory["solved_plan"] = False
        if predicted is not None:
            memory["model_misses"] = memory.get("model_misses", 0)+1

    if memory.get("failed_clicks", 0) >= 2:
        return _reset(memory)
    if same and bar < 20:
        return _reset(memory)

    if not pads or not (nf or ne):
        memory["recoveries"] = memory.get("recoveries", 0)+1
        if memory["recoveries"] <= 2:
            return _undo(memory, available)
        return _reset(memory)

    if not _possible(state, nf, ne):
        memory["recoveries"] = memory.get("recoveries", 0)+1
        if memory["recoveries"] <= 5 and bar >= 16:
            return _undo(memory, available)
        return _reset(memory)
    memory["recoveries"] = 0

    visits = memory.setdefault("visits", {})
    banned = memory.setdefault("banned", {})
    key = _key(state)
    visits[key] = visits.get(key, 0)+1

    mass_need = sum((1 << k)*n for k, n in nf.items())
    h = _heuristic(state, pads, nf, ne, mass_need)
    if h < memory.get("best_h", float("inf"))-0.15:
        memory["best_h"] = h
        memory["stalled"] = 0
    else:
        memory["stalled"] = memory.get("stalled", 0)+1

    plan = memory.get("plan", []) if prediction_ok and not same else []
    solved = bool(plan and memory.get("solved_plan"))
    clicks = memory.get("clicks", 0)

    if not solved:
        if visits[key] >= 4:
            return _reset(memory)
        if clicks >= 12 and memory.get("stalled", 0) >= 7:
            return _reset(memory)
        if bar < 5:
            return _reset(memory)
        if memory.get("model_misses", 0) >= 4:
            return _reset(memory)

    if plan and plan[0] in banned.get(key, ()):
        plan = []
        solved = False

    if not plan:
        plan, solved = _search(
            state, pads, nf, ne, visits, banned, start+7.6
        )
        if not solved:
            plan = plan[:3]

    if not plan:
        best = None
        for action in _actions(state, pads):
            if time.perf_counter() >= start+8.6:
                break
            if action in banned.get(key, ()):
                continue
            ns, win = _simulate(state, action, pads, nf, ne, mass_need)
            if ns is None or (not win and _key(ns) == key):
                continue
            value = (-1000 if win
                     else _heuristic(ns, pads, nf, ne, mass_need))
            value += visits.get(_key(ns), 0)
            if best is None or value < best[0]:
                best = (value, action, ns)
        if best is None:
            return _reset(memory)
        action, predicted = best[1], best[2]
        plan = []
        solved = False
    else:
        action = plan.pop(0)
        predicted, unused = _simulate(
            state, action, pads, nf, ne, mass_need
        )
        if predicted is None:
            banned.setdefault(key, set()).add(action)
            return _reset(memory)

    memory["plan"] = plan
    memory["solved_plan"] = solved
    memory["predicted"] = predicted
    memory["last_state"] = state
    memory["last_action"] = action
    memory["last_kind"] = 6
    memory["last_body"] = body
    memory["clicks"] = clicks+1
    return "ACTION6", int(action[1]), int(action[0])
