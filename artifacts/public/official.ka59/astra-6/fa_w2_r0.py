import numpy as np
import math


def build():
    P, G = {}, []
    S = {
        "x0": (16, 64, 64), "coords": (3, 64, 64),
        "mem": (11,), "aprev": (8,), "avail": (8,)
    }
    serial, scope = [0], [0]
    current = [G]
    pcache, ecache, shcache, cfgcache = {}, {}, {}, {}

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def par(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in pcache:
            k = "p" + str(len(P))
            pcache[key] = k
            P[k] = a
        return pcache[key]

    def emit(op, xs, ident=None, **kw):
        xs = list(xs)
        key = (scope[0], op, tuple(xs), tuple(sorted(kw.items())))
        if ident is None and key in ecache:
            return ecache[key]
        ident = ident or fresh()
        nd = {"id": ident, "op": op, "in": xs}
        nd.update(kw)
        current[0].append(nd)
        ss = [S[x] for x in xs]
        if op == "conv":
            shape = (P[kw["w"]].shape[0], 64, 64)
        elif op == "dense":
            shape = (P[kw["w"]].shape[0],)
        elif op in ("slice", "vslice"):
            shape = (kw["stop"] - kw["start"],) + ss[0][1:]
        elif op in ("concat", "vconcat"):
            shape = (sum(s[0] for s in ss),) + ss[0][1:]
        elif op in ("gmax", "gmean"):
            shape = (ss[0][0],)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = (max(s[0] for s in ss), 64, 64)
        elif op in ("vadd", "vmul"):
            shape = (max(s[0] for s in ss),)
        else:
            shape = ss[0]
        S[ident] = shape
        ecache[key] = ident
        return ident

    def conv(x, w, b=None, ident=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return emit("conv", [x], ident=ident, **kw)

    def dense(x, w, b=None, ident=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return emit("dense", [x], ident=ident, **kw)

    def cat(xs, vector=False):
        xs = list(xs)
        if len(xs) == 1:
            return xs[0]
        return emit("vconcat" if vector else "concat", xs)

    def sl(x, a, b):
        if a == 0 and b == S[x][0]:
            return x
        return emit(
            "slice" if len(S[x]) == 3 else "vslice",
            [x], start=a, stop=b
        )

    def aff(x, a=1., b=0.):
        c = S[x][0]
        aa = np.broadcast_to(np.asarray(a, np.float32), (c,))
        bb = np.broadcast_to(np.asarray(b, np.float32), (c,))
        if np.all(aa == 1) and np.all(bb == 0):
            return x
        return conv(
            x, np.diag(aa)[:, :, None, None],
            None if np.all(bb == 0) else bb
        )

    def va(x, a=1., b=0.):
        c = S[x][0]
        aa = np.broadcast_to(np.asarray(a, np.float32), (c,))
        bb = np.broadcast_to(np.asarray(b, np.float32), (c,))
        return dense(
            x, np.diag(aa),
            None if np.all(bb == 0) else bb
        )

    def const(values):
        a = np.asarray(values, np.float32).reshape(-1)
        return dense("avail", np.zeros((len(a), 8), np.float32), a)

    def add(a, b):
        return emit("add", [a, b])

    def sub(a, b):
        return emit("sub", [a, b])

    def mul(a, b):
        return emit("mul", [a, b])

    def mx(a, b):
        return emit("max2", [a, b])

    def mn(a, b):
        return emit("min2", [a, b])

    def relu(x):
        return emit("relu", [x])

    def maximum(xs):
        xs = list(xs)
        z = xs[0]
        for x in xs[1:]:
            z = mx(z, x)
        return z

    def total(x):
        c = S[x][0]
        return x if c == 1 else conv(
            x, np.ones((1, c, 1, 1), np.float32)
        )

    def pick(x, selector):
        return total(mul(x, selector))

    def shift(x, dy, dx):
        key = (scope[0], x, int(dy), int(dx))
        if key in shcache:
            return shcache[key]
        yy, xx, z = int(dy), int(dx), x
        while yy or xx:
            sy = max(-3, min(3, yy))
            sx = max(-3, min(3, xx))
            r = max(abs(sy), abs(sx))
            c = S[z][0]
            w = np.zeros((c, c, 2*r+1, 2*r+1), np.float32)
            w[np.arange(c), np.arange(c), r+sy, r+sx] = 1
            z = conv(z, w)
            yy, xx = yy-sy, xx-sx
        shcache[key] = z
        return z

    directions = [(-3, 0), (3, 0), (0, -3), (0, 3)]

    def four(x, bias=None):
        c = S[x][0]
        w = np.zeros((4*c, c, 7, 7), np.float32)
        for d, (dy, dx) in enumerate(directions):
            for j in range(c):
                w[d*c+j, j, 3+dy, 3+dx] = 1
        b = None if bias is None else np.full(4*c, bias, np.float32)
        return conv(x, w, b)

    directional_kernels = []
    for radius in (1, 2):
        w = np.zeros((4, 4, 2*radius+1, 2*radius+1), np.float32)
        for d, (dy, dx) in enumerate(directions):
            w[d, d, radius+radius*dy//3, radius+radius*dx//3] = 1
        directional_kernels.append(w)

    def dshift(x):
        for w in directional_kernels:
            x = conv(x, w)
        return x

    def reduce4(x):
        return maximum(sl(x, d, d+1) for d in range(4))

    def transport(x, iterations):
        parent, oldscope = current[0], scope[0]
        body = []
        current[0], scope[0] = body, serial[0] + 300000
        state = fresh("transport_state")
        S[state] = (4, 64, 64)
        nxt = dshift(state)
        current[0], scope[0] = parent, oldscope
        lid = fresh("transport")
        parent.append({
            "id": lid, "op": "loop", "T": iterations,
            "state": [{"id": state, "init": x}],
            "body": body, "next": {state: nxt}
        })
        out = lid + "." + state
        S[out] = (4, 64, 64)
        return out

    def stencil(x, specs, bias=None):
        groups = {}
        for o, spec in enumerate(specs):
            for (ic, dy, dx), value in spec.items():
                if not value:
                    continue
                by = -3 if dy < -3 else 3 if dy > 3 else 0
                bx = -3 if dx < -3 else 3 if dx > 3 else 0
                groups.setdefault((by, bx), []).append(
                    (o, ic, dy-by, dx-bx, value)
                )
        parts = []
        for j, ((by, bx), entries) in enumerate(sorted(groups.items())):
            r = max(max(abs(e[2]), abs(e[3])) for e in entries)
            w = np.zeros(
                (len(specs), S[x][0], 2*r+1, 2*r+1), np.float32
            )
            for o, ic, dy, dx, value in entries:
                w[o, ic, r+dy, r+dx] += value
            parts.append(conv(
                shift(x, by, bx), w, bias if j == 0 else None
            ))
        z = parts[0]
        for p in parts[1:]:
            z = add(z, p)
        return z

    one = sl("coords", 2, 3)
    zero = aff(one, 0)
    zero4 = conv(one, np.zeros((4, 1, 1, 1), np.float32))
    onev, zerov, cost6 = const([1]), const([0]), const([6])
    row = aff(sl("coords", 0, 1), 63)
    col = aff(sl("coords", 1, 2), 63)
    lex = conv(
        "coords",
        np.array([[-4032, -63, 0]], np.float32)[:, :, None, None],
        np.array([4096], np.float32)
    )

    def clip(x, nonnegative=False):
        return mn(x if nonnegative else relu(x), one)

    def lift(x):
        return add(zero, x)

    def vc(x, nonnegative=False):
        return emit("gmax", [clip(lift(x), nonnegative)])

    def comp(x):
        return va(x, -1, 1)

    def vadd(a, b):
        return emit("vadd", [a, b])

    def vmul(a, b):
        return emit("vmul", [a, b])

    def vmix(a, b, gate):
        return vadd(vmul(a, gate), vmul(b, comp(gate)))

    def mix(a, b, gate):
        return add(mul(a, gate), mul(b, comp(gate)))

    def sample(field, marker):
        return emit("gmax", [mul(field, marker)])

    def anypos(x):
        return vc(
            dense(x, np.full((1, S[x][0]), 8, np.float32)), True
        )

    def first(x, order):
        w = np.eye(S[x][0], dtype=np.float32)
        earlier = []
        for i in order:
            w[i, earlier] = -1
            earlier.append(i)
        return emit("vrelu", [dense(x, w)])

    def best(mask, value):
        rank = mul(add(aff(value, 8192), lex), mask)
        high = emit("gmax", [rank])
        return mul(relu(aff(sub(rank, high), 2, 1)), mask)

    def at_position(r, c):
        distance = add(
            emit("abs", [sub(row, r)]),
            emit("abs", [sub(col, c)])
        )
        return clip(aff(distance, -4, 2))

    def configs(x, cross=False, reverse=False):
        key = (scope[0], x, cross, reverse)
        if key in cfgcache:
            return cfgcache[key]
        patch = conv(x, np.ones((1, 1, 3, 3), np.float32))
        d = -3 if reverse else 3
        down, right = shift(patch, d, 0), shift(patch, 0, d)
        fields = [patch, down, right, shift(down, 0, d)]
        w = np.array([
            [1, 0, 0, 0], [1, 1, 0, 0],
            [1, 0, 1, 0], [1, 1, 1, 1]
        ], np.float32)
        if cross:
            fields += [shift(patch, -d, 0), shift(patch, 0, -d)]
            w = np.pad(w, ((0, 1), (0, 2)))
            w[4] = [1, 1, 1, 0, 1, 1]
        z = conv(cat(fields), w[:, :, None, None])
        cfgcache[key] = z
        return z

    def stamp(marker, selector, cross=False):
        return clip(pick(configs(marker, cross, True), selector), True)

    colors = {
        i: sl("x0", i, i+1)
        for i in [0, 2, 4, 5, 11, 12, 13, 14, 15]
    }
    c0, c2, c4, c5 = [colors[i] for i in [0, 2, 4, 5]]
    c11, c12, c13, c14, c15 = [
        colors[i] for i in [11, 12, 13, 14, 15]
    ]

    # Four rectangle shapes, including the selected piece's black edges.
    dimensions = [(3, 3), (3, 6), (6, 3), (6, 6)]
    edge_specs = [dict() for _ in range(32)]
    edge_bias = np.zeros(32, np.float32)
    core_specs, core_bias = [], []
    for t, (w, h) in enumerate(dimensions):
        edges = [
            [(-1, x) for x in range(-1, w-1)],
            [(h-2, x) for x in range(-1, w-1)],
            [(y, -1) for y in range(-1, h-1)],
            [(y, w-2) for y in range(-1, h-1)]
        ]
        for d, positions in enumerate(edges):
            j = 4*t+d
            for y, x in positions[1:-1]:
                edge_specs[j][(0, y, x)] = 1
            for y, x in positions:
                edge_specs[16+j][(1, y, x)] = 1
            edge_bias[j] = 1-len(positions[1:-1])
            edge_bias[16+j] = 1-len(positions)
        ys = [0] if h == 3 else [1, 2]
        xs = [0] if w == 3 else [1, 2]
        spec = {(0, y, x): 1 for y in ys for x in xs}
        core_specs.append(spec)
        core_bias.append(1-len(spec))
    edge = relu(stencil(cat([c14, c0]), edge_specs, edge_bias))
    edge = mx(sl(edge, 0, 16), sl(edge, 16, 32))
    core = relu(stencil(
        add(add(c0, c4), c5), core_specs, np.array(core_bias)
    ))
    w = np.zeros((4, 20, 1, 1), np.float32)
    for t in range(4):
        w[t, 4*t:4*t+4, 0, 0] = 1
        w[t, 16+t, 0, 0] = 1
    actors = relu(conv(cat([edge, core]), w, np.full(4, -4)))
    selected = mul(
        actors, relu(stencil(c0, core_specs, np.array(core_bias)))
    )
    cross = relu(aff(sl(configs(c11, True), 4, 5), 1, -44))
    objects = cat([actors, cross])
    bombs = relu(aff(sl(configs(add(c12, c13)), 3, 4), 1, -35))

    # Terrain configuration spaces and movable-object occupancy.
    valid_row = clip(aff(row, -1, 63))
    hard_pixels = mul(sub(one, c2), valid_row)
    land_pixels = sub(hard_pixels, c15)
    areas = np.array([9, 18, 18, 36, 45], np.float32)
    hard5 = relu(aff(configs(hard_pixels, True), 1, 1-areas))
    land5 = relu(aff(configs(land_pixels, True), 1, 1-areas))
    hard4, land4 = sl(hard5, 0, 4), sl(land5, 0, 4)
    land_next = [shift(land4, dy, dx) for dy, dx in directions]
    w = np.zeros((1, 16, 1, 1), np.float32)
    w[0, [0, 5, 11, 12, 13, 14], 0, 0] = 1
    occupancy = configs(mul(conv("x0", w), valid_row))
    w = np.zeros((1, 16, 1, 1), np.float32)
    w[0, [11, 12, 13], 0, 0] = 1
    nonrect = configs(conv("x0", w))
    nonrect_hit = clip(nonrect, True)

    # Outline goals.
    goal_specs, goal_bias = [], []
    for (w, h), tolerance in zip(dimensions, [3, 6, 6, 7]):
        border = {
            (y, x) for y in range(-2, h) for x in range(-2, w)
            if y in (-2, h-1) or x in (-2, w-1)
        }
        goal_specs.append({(0, y, x): 1 for y, x in border})
        goal_bias.append(1-len(border)+tolerance)
    outline = set()
    for y in range(11):
        if y in (0, 10):
            xs = range(3, 8)
        elif y in (1, 2, 8, 9):
            xs = (3, 7)
        elif y in (3, 7):
            xs = list(range(4)) + list(range(7, 11))
        else:
            xs = (0, 10)
        outline.update((y-5, x-5) for x in xs)
    goal_specs.append({(0, y, x): 1 for y, x in outline})
    goal_bias.append(1-len(outline)+7)
    goal_scores = stencil(
        c4, goal_specs, np.array(goal_bias, np.float32)
    )
    detected = clip(goal_scores)
    cross_goal = best(sl(detected, 4, 5), relu(sl(goal_scores, 4, 5)))

    # Memory: initialization, cross goal, committed goal type and position.
    initialized = sl("mem", 0, 1)
    remembered = [
        vmix(sl("mem", i+1, i+2), z, initialized)
        for i, z in enumerate([
            sample(row, cross_goal), sample(col, cross_goal),
            emit("gmax", [cross_goal])
        ])
    ]
    rr, cc, pp = remembered
    goals = cat([
        sl(detected, 0, 4), mul(at_position(rr, cc), pp)
    ])
    unsatisfied = relu(sub(goals, objects))
    old_type = sl("mem", 4, 9)
    old_marker = at_position(sl("mem", 9, 10), sl("mem", 10, 11))
    old_done = sample(pick(objects, old_type), old_marker)
    hold = vmul(anypos(old_type), comp(old_done))
    new_type = first(emit("gmax", [unsatisfied]), [4, 1, 3, 2, 0])
    new_marker = best(pick(unsatisfied, new_type), zero)
    goal_type = vmix(old_type, new_type, hold)
    goal_marker = mix(old_marker, new_marker, hold)
    type4 = sl(goal_type, 0, 4)
    emit("vconcat", [
        onev, rr, cc, pp, goal_type,
        sample(row, goal_marker), sample(col, goal_marker)
    ], ident="mem_next")

    walk_primary = dense(
        goal_type, np.array([[1, 1, 1, 1, 0]], np.float32)
    )
    primary_cost = dense(
        goal_type, np.array([[6, 6, 6, 6, 3]], np.float32)
    )
    actor_count = dense(
        emit("gmean", [actors]), np.full((1, 4), 4096, np.float32)
    )
    multiple = vc(va(actor_count, 1, -1))
    solo = vmul(walk_primary, comp(multiple))
    primary_seed = aff(goal_marker, 200)
    primary_candidates = pick(relu(sub(objects, goals)), goal_type)

    # Bomb orientation and launch bands.
    charge_specs = []
    for d in range(4):
        if d == 0:
            back = [(4, x) for x in range(-1, 5)]
            front = [(-1, x) for x in range(-1, 5)]
        elif d == 1:
            back = [(-1, x) for x in range(-1, 5)]
            front = [(4, x) for x in range(-1, 5)]
        elif d == 2:
            back = [(y, 4) for y in range(-1, 5)]
            front = [(y, -1) for y in range(-1, 5)]
        else:
            back = [(y, -1) for y in range(-1, 5)]
            front = [(y, 4) for y in range(-1, 5)]
        spec = {(0, y, x): 1 for y, x in back}
        spec.update({(1, y, x): 1 for y, x in front})
        charge_specs.append(spec)
    charged = relu(stencil(
        cat([c12, c13]), charge_specs, np.full(4, -11)
    ))
    uncharged = sub(one, reduce4(charged))
    wall_above = clip(stencil(c2, [
        {(0, -2, x): 1 for x in range(-1, 5)}
    ]), True)
    bomb_dirs = []
    for d in range(4):
        fallback = sub(one, wall_above) if d == 0 else (
            wall_above if d == 1 else zero
        )
        bomb_dirs.append(mul(
            bombs, add(sl(charged, d, d+1), mul(uncharged, fallback))
        ))
    offsets = [
        [(3, 0), (3, -3)], [(-6, 0), (-6, -3)],
        [(0, 3), (-3, 3)], [(0, -6), (-3, -6)]
    ]
    bands, unions = [], []
    for d, (dy, dx) in enumerate(directions):
        z = clip(stencil(bomb_dirs[d], [
            {(0, y, x): 1 for y, x in offsets[d]}
        ]), True)
        z1 = shift(z, -dy, -dx)
        z2 = shift(z1, -dy, -dx)
        bands.append([z, z1, z2])
        unions.append(maximum([z, z1, z2]))
    packed_bands = [
        cat([bands[d][j] for d in range(4)]) for j in range(3)
    ]
    launch_union = cat(unions)

    # A scalar resting-state value and four directional slide values.
    def motion(seed, hard, land, walking, cost, ordinary, blast):
        hard_cap, land_cap = aff(hard, 200), aff(land, 200)
        gulch_cap = aff(relu(sub(hard, land)), 200)
        walk_cap = mul(land_cap, walking)
        blocked = sub(one, four(hard))
        next_land = aff(four(land), 200)
        seed4 = cat([seed, seed, seed, seed])
        parent, oldscope = current[0], scope[0]
        body = []
        current[0], scope[0] = body, serial[0]+100000
        v, s, q = fresh("value"), fresh("slides"), fresh("kicks")
        S[v], S[s], S[q] = (1, 64, 64), (4, 64, 64), (4, 64, 64)
        slide = mx(v, mn(dshift(s), gulch_cap))
        bounce = mul(v, blocked)
        f, stages = slide, []
        for _ in range(5):
            f = mn(mx(dshift(f), bounce), hard_cap)
            stages.append(f)
        blast_value = maximum([
            mul(stages[2], packed_bands[0]),
            mul(stages[1], packed_bands[1]),
            mul(stages[0], packed_bands[2])
        ])
        nq = mx(mul(stages[4], ordinary), mul(blast_value, blast))
        walk_value = mn(
            sub(reduce4(mn(stages[0], next_land)), onev), walk_cap
        )
        nv = relu(mn(maximum([
            seed, v, sub(reduce4(nq), cost), walk_value
        ]), land_cap))
        current[0], scope[0] = parent, oldscope
        lid = fresh("plan")
        parent.append({
            "id": lid, "op": "loop", "T": 28,
            "state": [
                {"id": v, "init": seed},
                {"id": s, "init": seed4},
                {"id": q, "zeros": [4, 64, 64]}
            ],
            "body": body, "next": {v: nv, s: slide, q: nq}
        })
        for z in (v, s, q):
            S[lid+"."+z] = S[z]
        return lid+"."+v, lid+"."+q, lid+"."+s

    def walking_fields(seed, free, iterations=28):
        cap = aff(free, 200)
        parent, oldscope = current[0], scope[0]
        body = []
        current[0], scope[0] = body, serial[0]+200000
        state = fresh("walking_state")
        S[state] = (4, 64, 64)
        neighbor = four(state, -1)
        nxt = relu(mx(seed, mn(
            maximum(sl(neighbor, 4*d, 4*d+4) for d in range(4)),
            cap
        )))
        current[0], scope[0] = parent, oldscope
        lid = fresh("walking")
        parent.append({
            "id": lid, "op": "loop", "T": iterations,
            "state": [{"id": state, "init": seed}],
            "body": body, "next": {state: nxt}
        })
        out = lid+"."+state
        S[out] = (4, 64, 64)
        return out

    pblock = pick(cat([nonrect_hit, zero]), goal_type)
    phard = mul(pick(hard5, goal_type), sub(one, pblock))
    pland = mul(pick(land5, goal_type), sub(one, pblock))
    pv, pq, ps = motion(
        primary_seed, phard, pland, walk_primary, primary_cost,
        comp(solo), sl(goal_type, 0, 1)
    )
    primary = best(primary_candidates, pv)
    primary_counts = configs(stamp(primary, goal_type, True))

    def route(value, marker, own_counts, typ, land):
        collision = clip(sub(occupancy, own_counts))
        free = mul(land, sub(one, pick(collision, typ)))
        valid_map = four(free)
        values = sample(mul(four(value), valid_map), marker)
        valid = sample(valid_map, marker)
        base = sample(value, marker)
        scores = dense(
            cat([values, valid], True),
            np.concatenate([np.eye(4), 500*np.eye(4)], axis=1),
            np.full(4, -500)
        )
        improvement = emit("vrelu", [dense(
            cat([values, base], True),
            np.concatenate([np.eye(4), -np.ones((4, 1))], axis=1),
            np.full(4, -.25)
        )])
        return scores, anypos(improvement), base

    def rewards(q, marker, base, cost):
        inp = cat([sample(q, marker), base, cost], True)
        wg = np.zeros((4, 6), np.float32)
        wr = np.zeros((4, 6), np.float32)
        for d in range(4):
            wg[d, d], wg[d, 4] = 4, -4
            wr[d, d], wr[d, 5] = 1, -1
        gate = vc(dense(inp, wg, np.full(4, -1)))
        return vmul(emit("vrelu", [dense(inp, wr)]), gate)

    def contacts(counts, reward):
        collision = clip(counts, True)
        w = np.concatenate(
            [np.eye(4), -np.eye(4), np.eye(4), np.eye(4)], axis=1
        ).astype(np.float32)[:, :, None, None]
        fields = []
        for d, (dy, dx) in enumerate(directions):
            legal = relu(conv(cat([
                shift(collision, dy, dx), collision, land4, land_next[d]
            ]), w, np.full(4, -2)))
            fields.append(mul(legal, sl(reward, d, d+1)))
        return fields, maximum(fields)

    def goal_ready(value, base):
        z = dense(
            cat([value, base], True),
            np.array([[10, -10]], np.float32), np.array([1])
        )
        return emit("gmax", [clip(mn(lift(z), value))])

    reverse = np.eye(4, dtype=np.float32)[[1, 0, 3, 2]]

    def waiting(marker, reward):
        scores = vmul(sample(launch_union, marker), reward)
        return dense(scores, reverse), anypos(scores)

    pwalk, pready, pbase = route(
        pv, primary, primary_counts, type4, pland
    )
    pready = vmul(pready, walk_primary)
    preward = rewards(pq, primary, pbase, primary_cost)
    pwait, pwait_ready = waiting(primary, preward)
    pwait_ready = vmul(pwait_ready, walk_primary)
    pscore = vmix(pwalk, pwait, pready)
    pready = vc(vadd(pready, pwait_ready), True)
    pcontacts, pseed = contacts(primary_counts, preward)

    # Select and route a helper.
    eligible = relu(sub(actors, mul(primary, type4)))
    fallback_type = first(emit("gmax", [eligible]), [0, 1, 2, 3])
    helper_clear = sub(one, mx(nonrect_hit, clip(primary_counts, True)))
    helper_land = mul(land4, helper_clear)
    reach = walking_fields(pseed, helper_land, 22)
    reach = sub(reach, aff(mul(actors, sl(goals, 0, 4)), 12))
    type_scores = sample(reach, eligible)
    largest = emit("gmax", [reduce4(lift(type_scores))])
    tied = emit("gmax", [
        clip(aff(sub(lift(type_scores), largest), 8, 1))
    ])
    helper_type = first(
        vmul(tied, emit("gmax", [eligible])), [0, 1, 2, 3]
    )
    helper_count = dense(
        emit("gmean", [eligible]), np.full((1, 4), 4096, np.float32)
    )
    helper_ordinary = vc(va(helper_count, 1, -1))
    sseed = pick(pseed, helper_type)
    shard = pick(mul(hard4, helper_clear), helper_type)
    sland_simple = pick(helper_land, helper_type)
    own_type_image = stamp(pick(actors, helper_type), helper_type)
    other_hit = clip(sub(occupancy, configs(own_type_image)))
    sland = mul(
        sland_simple, sub(one, pick(other_hit, helper_type))
    )
    sv, sq, ss = motion(
        sseed, shard, sland, onev, cost6,
        helper_ordinary, sl(helper_type, 0, 1)
    )
    secondary = best(pick(eligible, helper_type), sv)
    secondary_counts = configs(stamp(secondary, helper_type))
    swalk, swalk_ready, sbase = route(
        sv, secondary, secondary_counts, helper_type, sland
    )
    sreward = rewards(sq, secondary, sbase, cost6)
    scontact = sample(
        cat([pick(z, helper_type) for z in pcontacts]), secondary
    )
    scontact_ready = goal_ready(sample(sseed, secondary), sbase)
    swait, swait_ready = waiting(secondary, sreward)
    sscore = vmix(
        scontact, vmix(swalk, swait, swalk_ready), scontact_ready
    )
    sready = vc(vadd(vadd(scontact_ready, swalk_ready), swait_ready), True)

    # Third-piece push goals.
    plain_contacts, _ = contacts(
        secondary_counts, vmul(sreward, helper_ordinary)
    )

    # Optional launcher relocation, with long shifts compiled as loops.
    chosen_bomb = best(bombs, row)
    orientation = sample(cat(bomb_dirs), chosen_bomb)
    objective_slide = mix(ps, ss, solo)
    far4 = transport(objective_slide, 4)
    far5 = dshift(far4)
    w = np.zeros((4, 8, 1, 1), np.float32)
    for d in range(4):
        w[d, d if d in (0, 2) else d+4, 0, 0] = 1
    launch_value = conv(cat([far4, far5]), w)
    w = np.zeros((4, 4, 7, 7), np.float32)
    for d in range(4):
        dy, dx = (0, 3) if d < 2 else (3, 0)
        w[d, d, 3+dy, 3+dx] = 1
    launch_value = mx(launch_value, conv(launch_value, w))
    bomb_use = vc(vadd(
        sl(helper_type, 0, 1), vmul(solo, sl(goal_type, 0, 1))
    ), True)
    bomb_value = mul(pick(launch_value, orientation), bomb_use)
    bomb_destination = mul(bomb_value, sl(hard4, 3, 4))
    bomb_successors = transport(
        cat([bomb_destination]*4), 5
    )
    bomb_reward = rewards(
        bomb_successors, chosen_bomb,
        sample(bomb_value, chosen_bomb), const([3])
    )
    bomb_counts = configs(stamp(
        chosen_bomb, const([0, 0, 0, 1])
    ))
    bomb_contacts, _ = contacts(bomb_counts, bomb_reward)
    tcontacts = [
        mx(a, b) for a, b in zip(plain_contacts, bomb_contacts)
    ]
    tseed4 = maximum(tcontacts)

    teligible = relu(sub(
        eligible, mul(mul(secondary, helper_type), helper_ordinary)
    ))
    teligible = mx(teligible, mul(actors, solo))
    ttype = first(emit("gmax", [teligible]), [0, 1, 2, 3])
    tblock = clip(add(
        nonrect, mul(secondary_counts, helper_ordinary)
    ), True)
    tclear = sub(one, tblock)
    tfree = mul(land4, tclear)
    tseed = pick(tseed4, ttype)
    thard = pick(mul(hard4, tclear), ttype)
    tland = pick(tfree, ttype)
    tv, tq, _ = motion(
        tseed, thard, tland, onev, cost6, zerov, sl(ttype, 0, 1)
    )
    twalk_alternative = pick(
        walking_fields(tseed4, tfree, 40), ttype
    )

    tertiary = best(pick(teligible, ttype), tv)
    tertiary_counts = configs(stamp(tertiary, ttype))
    twalk, twalk_ready, tbase = route(
        tv, tertiary, tertiary_counts, ttype, tland
    )
    tcontact = sample(
        cat([pick(z, ttype) for z in tcontacts]), tertiary
    )
    tcontact_ready = goal_ready(sample(tseed, tertiary), tbase)
    twait, _ = waiting(
        tertiary, rewards(tq, tertiary, tbase, cost6)
    )
    tscore = vmix(
        tcontact, vmix(twalk, twait, twalk_ready), tcontact_ready
    )

    spresent = emit("gmax", [secondary])
    tpresent = emit("gmax", [tertiary])
    fallback_marker = mix(secondary, primary, spresent)
    fallback_score = vmix(swalk, pscore, spresent)
    tail_marker = mix(tertiary, fallback_marker, tpresent)
    tail_score = vmix(tscore, fallback_score, tpresent)
    middle_marker = mix(secondary, tail_marker, sready)
    middle_score = vmix(sscore, tail_score, sready)
    marker = mix(primary, middle_marker, pready)
    score = vmix(pscore, middle_score, pready)

    conv(marker, np.ones((1, 1, 1, 1), np.float32), ident="click")
    active = sample(total(selected), marker)
    out = cat([score, active, "aprev"], True)
    wo = np.zeros((8, 13), np.float32)
    bo = np.full(8, -10000, np.float32)
    for d in range(4):
        wo[d+1, d], bo[d+1] = 1, 0
    wo[6, 4], bo[6] = -2000, 1000
    for action, opposite in [(1, 2), (2, 1), (3, 4), (4, 3)]:
        wo[action, 5+opposite] = -.125
    dense(out, wo, bo, ident="logits")

    # Compilation below examines graph structure only.
    def dependencies(n):
        if n["op"] != "loop":
            return set(n["in"])
        local = {s["id"] for s in n["state"]}
        local.update(b["id"] for b in n["body"])
        result = {s["init"] for s in n["state"] if "init" in s}
        for b in n["body"]:
            result.update(b["in"])
        return result-local

    def prune_body(n):
        producers = {b["id"]: b for b in n["body"]}
        used = set()

        def visit(z):
            if z in used or z not in producers:
                return
            used.add(z)
            for x in producers[z]["in"]:
                visit(x)

        for x in n["next"].values():
            visit(x)
        n["body"] = [b for b in n["body"] if b["id"] in used]

    def prune(nodes):
        producers = {n["id"]: n for n in nodes}
        used = set()

        def visit(z):
            root = z.split(".", 1)[0]
            if root in used or root not in producers:
                return
            used.add(root)
            for x in dependencies(producers[root]):
                visit(x)

        for x in ("logits", "click", "mem_next"):
            visit(x)
        return [n for n in nodes if n["id"] in used]

    def redirect(nodes, mapping):
        def replace(x):
            while x in mapping:
                x = mapping[x]
            return x

        for n in nodes:
            if n["op"] == "loop":
                for s in n["state"]:
                    if "init" in s:
                        s["init"] = replace(s["init"])
                for b in n["body"]:
                    b["in"] = [replace(x) for x in b["in"]]
                n["next"] = {
                    k: replace(v) for k, v in n["next"].items()
                }
            else:
                n["in"] = [replace(x) for x in n["in"]]

    def fold(nodes):
        producers = {n["id"]: n for n in nodes}
        uses = {}
        for n in nodes:
            for x in dependencies(n):
                uses[x] = uses.get(x, 0)+1
        for n in nodes:
            if n["op"] == "loop":
                n["body"] = fold(n["body"])
                continue
            if n["op"] not in ("conv", "dense"):
                continue
            parent = producers.get(n["in"][0])
            if (parent is None or uses.get(parent["id"], 0) != 1
                    or parent["op"] != n["op"]):
                continue
            a, b = P[n["w"]], P[parent["w"]]
            if n["op"] == "conv":
                if a.shape[2:] != (1, 1):
                    continue
                entries = a.shape[0]*b.shape[1]*b.shape[2]*b.shape[3]
                if entries > a.size+b.size:
                    continue
                weight = np.einsum("oi,ichw->ochw", a[:, :, 0, 0], b)
                matrix = a[:, :, 0, 0]
            else:
                if a.shape[0]*b.shape[1] > a.size+b.size:
                    continue
                weight, matrix = a @ b, a
            bias = (
                P[n["b"]].copy() if "b" in n
                else np.zeros(a.shape[0], np.float32)
            )
            if "b" in parent:
                bias += matrix @ P[parent["b"]]
            n["in"], n["w"] = parent["in"][:], par(weight)
            if np.any(bias != 0):
                n["b"] = par(bias)
            else:
                n.pop("b", None)
        return nodes

    def compile_live():
        live = prune(G)
        for _ in range(2):
            live = fold(live)
            for n in live:
                if n["op"] == "loop":
                    prune_body(n)
            live = prune(live)
        return live

    def parameter_names(nodes):
        used = set()
        for n in nodes:
            for b in n["body"] if n["op"] == "loop" else [n]:
                for key in ("w", "b"):
                    if key in b:
                        used.add(b[key])
        return used

    def static_size(nodes):
        # Conservatively count loop states and their exported values too.
        count, channels = len(nodes), 19
        for n in nodes:
            if n["op"] == "loop":
                count += len(n["body"]) + len(n["state"])
                for s in n["state"]:
                    shape = (
                        S[s["init"]] if "init" in s
                        else tuple(s["zeros"])
                    )
                    if len(shape) == 3:
                        channels += 2*shape[0]
                for b in n["body"]:
                    if len(S[b["id"]]) == 3:
                        channels += S[b["id"]][0]
            elif len(S[n["id"]]) == 3:
                channels += S[n["id"]][0]
        entries = sum(P[k].size for k in parameter_names(nodes))
        return count, channels, entries

    def fits(nodes):
        n, c, p = static_size(nodes)
        return n <= 990 and c <= 3990 and p <= 149000

    # Every fallback has an earlier, fully defined replacement.
    # Goal commitment and binary arbitration are never removed.
    fallbacks = [
        {helper_type: fallback_type},
        {sland: sland_simple},
        {tv: twalk_alternative, tq: zero4},
        {a: b for a, b in zip(tcontacts, plain_contacts)},
        {tail_marker: fallback_marker, tail_score: fallback_score},
        {middle_marker: primary, middle_score: pscore},
    ]
    live = compile_live()
    for mapping in fallbacks:
        if fits(live):
            break
        redirect(G, mapping)
        live = compile_live()

    # Exact MAC and node-evaluation accounting.
    def mac(n):
        op = n["op"]
        if op == "conv":
            o, c, k, _ = P[n["w"]].shape
            return int(o*c*k*k*4096)
        if op == "dense":
            m, k = P[n["w"]].shape
            return int(m*k)
        if op in ("add", "sub", "mul", "max2", "min2"):
            return int(np.prod(S[n["id"]]))
        if op in ("gmax", "gmean", "rowmax", "colmax", "rowsum", "colsum"):
            return int(np.prod(S[n["in"][0]]))
        if op in ("vadd", "vmul"):
            return S[n["id"]][0]
        return 0

    loops = [n for n in live if n["op"] == "loop"]
    # Transport loops have exact geometric lengths and are not shortened.
    fixed_loops = [n for n in loops if n["id"].startswith("transport")]
    variable_loops = [n for n in loops if n not in fixed_loops]
    fixed_cost = sum(mac(n) for n in live if n["op"] != "loop")
    fixed_cost += sum(
        n["T"]*sum(mac(b) for b in n["body"]) for n in fixed_loops
    )
    fixed_evals = len(live) + sum(
        n["T"]*len(n["body"]) for n in fixed_loops
    )
    variable_cost = sum(
        n["T"]*sum(mac(b) for b in n["body"]) for n in variable_loops
    )
    variable_evals = sum(
        n["T"]*len(n["body"]) for n in variable_loops
    )
    factor = min(
        1.,
        max(0., (1_480_000_000-fixed_cost)/max(1, variable_cost)),
        max(0., (9800-fixed_evals)/max(1, variable_evals))
    )
    for n in variable_loops:
        n["T"] = max(1, int(math.floor(n["T"]*factor)))

    used = parameter_names(live)
    params = {k: v for k, v in P.items() if k in used}
    return {"graph": live, "memory_size": 11, "params": params}
