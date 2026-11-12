import numpy as np
import math


def build():
    P = {}
    graph = []
    current = graph
    shapes = {
        "x0": (16, True), "coords": (3, True),
        "mem": (33, False), "aprev": (8, False), "avail": (8, False)
    }
    serial = [0]
    pcache = {}
    caches = {}
    linear_defs = {}

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def param(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in pcache:
            name = "p" + str(len(P))
            P[name] = a
            pcache[key] = name
        return pcache[key]

    def node(op, ins, name=None, **kw):
        key = (op, tuple(ins), tuple(sorted(kw.items())))
        cache = caches.setdefault(id(current), {})
        if name is None and key in cache:
            return cache[key]
        name = name or fresh()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(kw)
        current.append(nd)
        c, spatial = shapes[ins[0]]
        if op in ("conv", "dense"):
            c = P[kw["w"]].shape[0]
            spatial = op == "conv"
        elif op in ("concat", "vconcat"):
            c = sum(shapes[x][0] for x in ins)
        elif op in ("slice", "vslice"):
            c = kw["stop"] - kw["start"]
        elif op in ("gmax", "gmean"):
            spatial = False
        elif op in ("add", "sub", "mul", "max2", "min2", "vadd", "vmul"):
            c = max(shapes[x][0] for x in ins)
            spatial = any(shapes[x][1] for x in ins)
        shapes[name] = (c, spatial)
        cache[key] = name
        return name

    def conv(x, w, b=None, name=None):
        kw = {"w": param(w)}
        if b is not None:
            kw["b"] = param(b)
        return node("conv", [x], name=name, **kw)

    def lin(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        spatial = shapes[x][1]
        bias = np.zeros(w.shape[0], np.float32) if b is None else \
            np.broadcast_to(np.asarray(b, np.float32), (w.shape[0],)).copy()
        kw = {"w": param(w[:, :, None, None] if spatial else w)}
        if b is not None:
            kw["b"] = param(bias)
        out = node("conv" if spatial else "dense", [x], name=name, **kw)
        linear_defs[out] = (x, w, bias)
        return out

    def affine(x, a=1.0, b=0.0, name=None):
        if name is None and a == 1 and np.all(np.asarray(b) == 0):
            return x
        if x in linear_defs:
            source, w, old_b = linear_defs[x]
            return lin(source, w * a, old_b * a + b, name)
        return lin(x, np.eye(shapes[x][0], dtype=np.float32) * a, b, name)

    def sl(x, a, b):
        return node("slice" if shapes[x][1] else "vslice",
                    [x], start=a, stop=b)

    def cat(xs):
        return node("concat" if shapes[xs[0]][1] else "vconcat", xs)

    def add(a, b):
        return node("add" if shapes[a][1] or shapes[b][1] else "vadd", [a, b])

    def sub(a, b):
        if shapes[a][1] or shapes[b][1]:
            return node("sub", [a, b])
        return add(a, affine(b, -1))

    def mul(a, b):
        return node("mul" if shapes[a][1] or shapes[b][1] else "vmul", [a, b])

    def relu(x):
        return node("relu" if shapes[x][1] else "vrelu", [x])

    def inv(x):
        return affine(x, -1, 1)

    def mx(a, b):
        return node("max2", [a, b])

    def mn(a, b):
        return node("min2", [a, b])

    def gm(x):
        return node("gmax", [x])

    def total(x):
        return lin(x, np.ones((1, shapes[x][0]), np.float32))

    ones = sl("coords", 2, 3)
    zero = affine(ones, 0)
    onev = gm(ones)
    xy = sl("coords", 0, 2)
    rank = lin("coords", [[-126, -1.96875, 0]])

    def cap(x):
        out = mn(x, ones)
        return out if shapes[x][1] else gm(out)

    def const(values, spatial=False):
        values = np.asarray(values, np.float32).reshape(-1)
        return lin(zero if spatial else onev,
                   np.zeros((len(values), 1), np.float32), values)

    def shift(x, dy, dx, bias=None):
        c = shapes[x][0]
        k = 1 if dy == 0 and dx == 0 else 7
        w = np.zeros((c, c, k, k), np.float32)
        for i in range(c):
            w[i, i, k // 2 + dy, k // 2 + dx] = 1
        return conv(x, w, None if bias is None else np.full(c, bias, np.float32))

    directions = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    unit_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dmat = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]], np.float32)
    merge_dirs = np.concatenate([np.eye(4), np.eye(4)], axis=1).astype(np.float32)
    neighbor_w = np.zeros((4, 1, 7, 7), np.float32)
    for i, (dy, dx) in enumerate(directions):
        neighbor_w[i, 0, 3 + dy, 3 + dx] = 1

    def neighbors(x, bias=0):
        return conv(x, neighbor_w, np.full(4, bias, np.float32))

    def dshift(x):
        return cat([shift(sl(x, d, d + 1), dy, dx)
                    for d, (dy, dx) in enumerate(directions)])

    def banks6(x):
        right = shift(shift(x, 0, 3), 0, 3)
        down = shift(shift(x, 3, 0), 3, 0)
        diagonal = shift(shift(right, 3, 0), 3, 0)
        return cat([x, right, down, diagonal])

    def cmax(x):
        out = sl(x, 0, 1)
        for i in range(1, shapes[x][0]):
            out = mx(out, sl(x, i, i + 1))
        return out

    def winner(v, old=None, gate=None):
        n = shapes[v][0]
        v = affine(v, 1, -np.arange(n, dtype=np.float32) / 64)
        if old is not None:
            bonus = affine(old, 512)
            if gate is not None:
                bonus = mul(bonus, gate)
            v = add(v, bonus)
        w = np.zeros((n * n, n), np.float32)
        r = np.zeros((n, n * n), np.float32)
        for i in range(n):
            for j in range(n):
                w[i * n + j, j] += 1
                w[i * n + j, i] -= 1
                r[i, i * n + j] = 1
        loss = lin(relu(lin(v, w)), r)
        return relu(affine(loss, -1024, 1))

    def distance(ref):
        return lin(node("abs", [sub(xy, ref)]), [[63, 63]])

    def position(mask):
        return gm(mul(xy, mask))

    def bbox_center(mask):
        high = gm(mul(xy, mask))
        low = inv(gm(mul(inv(xy), mask)))
        return affine(add(high, low), .5)

    def choose(det, base, ref=None, pred=None, typ=None, gate=None, extra=None):
        score = add(const(base, True), rank)
        if ref is not None:
            score = add(score, affine(distance(ref), -10))
        if pred is not None:
            continuity = affine(distance(pred), -100, 10000)
            type_gate = typ if gate is None else mul(typ, gate)
            score = add(score, mul(continuity, type_gate))
        if extra is not None:
            score = add(score, extra)
        score = mul(relu(score), det)
        scalar = total(score)
        best = gm(scalar)
        win = relu(affine(sub(best, scalar), -1024, 1))
        chosen = mul(det, win)
        mask = total(chosen)
        return chosen, mask, gm(chosen), position(mask)

    def dot(a, b):
        return total(mul(a, b))

    def same_pos(a, b):
        delta = add(zero, sub(a, b))
        length = lin(node("abs", [delta]), [[63, 63]])
        return gm(relu(affine(length, -4, 1)))

    dims = [(3, 3), (6, 3), (3, 6), (6, 6)]
    cells = [
        [(0, 0)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    ]

    # Controller templates, anchored at top-left + (1,1).
    w = np.zeros((3, 16), np.float32)
    w[0, [14, 0]] = 1
    w[1, [0, 4, 5]] = 1
    w[2, 14] = 1
    controller_colours = lin("x0", w)
    w = np.zeros((4, 3, 7, 7), np.float32)
    b = []
    for t, (h, wid) in enumerate(dims):
        cy, cx = (h - 1) // 2, (wid - 1) // 2
        ys = [1] if h == 3 else [2, 3]
        xs = [1] if wid == 3 else [2, 3]
        for y in range(h):
            for x in range(wid):
                if y in ys and x in xs:
                    ch = 1
                elif y in (0, h - 1) or x in (0, wid - 1):
                    ch = 0
                else:
                    ch = 2
                w[t, ch, 3 + y - cy, 3 + x - cx] = 1
        b.append(1 - h * wid)
    raw = relu(conv(controller_colours, w, b))
    detected = [
        shift(sl(raw, t, t + 1), (h - 1) // 2 - 1, (wid - 1) // 2 - 1)
        for t, (h, wid) in enumerate(dims)
    ]

    # Black adjacency borders can imitate centres. Suppress such nested matches.
    inside_big = cap(shift(conv(
        detected[3], np.ones((1, 1, 7, 7), np.float32)
    ), -1, -1))
    medium_v = mul(detected[1], inv(inside_big))
    medium_h = mul(detected[2], inv(inside_big))
    kv = np.zeros((1, 1, 7, 7), np.float32)
    kh = np.zeros((1, 1, 7, 7), np.float32)
    kv[0, 0, :, 2:5] = 1
    kh[0, 0, 2:5, :] = 1
    inside_medium = add(
        shift(conv(medium_v, kv), -1, 0),
        shift(conv(medium_h, kh), 0, -1)
    )
    small = mul(detected[0], inv(cap(add(inside_big, inside_medium))))
    controllers = cat([small, medium_v, medium_h, detected[3]])

    w = np.zeros((4, 1, 7, 7), np.float32)
    for t, (h, wid) in enumerate(dims):
        ys = [1] if h == 3 else [2, 3]
        xs = [1] if wid == 3 else [2, 3]
        for y in ys:
            for x in xs:
                w[t, 0, y + 2, x + 2] = 1 / (len(ys) * len(xs))
    active = mul(controllers, conv(sl("x0", 0, 1), w))
    active_mask = total(active)
    active_pos = position(active_mask)
    count = affine(node("gmean", [total(controllers)]), 4096)
    only_one = cap(relu(affine(count, -1, 2)))

    k3 = np.ones((1, 1, 3, 3), np.float32)
    plus = np.zeros((1, 1, 7, 7), np.float32)
    for y, x in [(0, 0), (-3, 0), (3, 0), (0, -3), (0, 3)]:
        plus[0, 0, y + 3, x + 3] = 1
    cross_unit = relu(conv(sl("x0", 11, 12), k3, [-8]))
    cross = shift(relu(conv(cross_unit, plus, [-4])), 3, 3)
    objects = cat([controllers, cross])

    # Goal outlines.
    w = np.zeros((2, 16), np.float32)
    w[0, [0, 4, 5, 11, 12, 13, 14]] = 1
    w[1, 4] = 1
    goal_bank = banks6(lin("x0", w))
    outlines = []
    for h, wid in dims:
        outlines.append([
            (y - 2, x - 2)
            for y in range(h + 2) for x in range(wid + 2)
            if y in (0, h + 1) or x in (0, wid + 1)
        ])
    co = []
    for y in range(11):
        if y in (0, 10):
            xs = range(3, 8)
        elif y in (1, 2, 8, 9):
            xs = (3, 7)
        elif y in (3, 7):
            xs = list(range(4)) + list(range(7, 11))
        else:
            xs = (0, 10)
        co.extend((y - 2, x - 2) for x in xs)
    outlines.append(co)
    w = np.zeros((10, 8, 7, 7), np.float32)
    for t, points in enumerate(outlines):
        for y, x in points:
            by, bx = (6 if y > 3 else 0), (6 if x > 3 else 0)
            bank = (2 if by else 0) + (1 if bx else 0)
            w[t, 2 * bank, 3 + y - by, 3 + x - bx] += 1
            w[t + 5, 2 * bank + 1, 3 + y - by, 3 + x - bx] += 1
    outline_counts = conv(goal_bank, w)
    lengths = np.array([len(p) for p in outlines], np.float32)
    complete_outline = relu(affine(sl(outline_counts, 0, 5), 1, 1 - lengths))
    visible_outline = cap(relu(affine(
        sl(outline_counts, 5, 10), 1, 1 - np.ceil(lengths * .5)
    )))
    goals = mul(complete_outline, visible_outline)
    unfinished = mul(objects, inv(goals))
    free_goals = mul(goals, inv(objects))

    w = np.zeros((1, 16), np.float32)
    w[0, [12, 13]] = 1
    bomb_pixels = lin("x0", w)
    w = np.zeros((1, 1, 7, 7), np.float32)
    w[0, 0, 1:7, 1:7] = 1
    bombs = shift(relu(conv(bomb_pixels, w, [-35])), 1, 1)
    bomb_exists = gm(bombs)

    charging_colours = cat([sl("x0", 13, 14), sl("x0", 12, 13)])
    w = np.zeros((4, 2, 7, 7), np.float32)
    edges = [
        ((3, 0), (-2, 0)), ((-2, 0), (3, 0)),
        ((0, 3), (0, -2)), ((0, -2), (0, 3))
    ]
    for d, (red, purple) in enumerate(edges):
        w[d, 1, red[0] + 3, red[1] + 3] = 1
        w[d, 0, purple[0] + 3, purple[1] + 3] = 1
    bomb_orientation = shift(relu(conv(charging_colours, w, [-1] * 4)), 1, 1)

    sum_w = np.zeros((4, 1, 7, 7), np.float32)
    paint_w = np.zeros((1, 4, 7, 7), np.float32)
    for t in range(4):
        for y, x in cells[t]:
            sum_w[t, 0, 3 + 3 * y, 3 + 3 * x] = 1
            paint_w[0, t, 3 - 3 * y, 3 - 3 * x] = 1

    def shape_sum(unit):
        return cat([conv(unit, sum_w), shift(conv(unit, plus), 3, 3)])

    def paint(selected):
        rectangles = conv(sl(selected, 0, 4), paint_w)
        cross_part = shift(conv(sl(selected, 4, 5), plus), -3, -3)
        return cap(conv(add(rectangles, cross_part), k3))

    def collision(field):
        return cap(shape_sum(cap(conv(field, k3))))

    bottom = relu(lin("coords", [[63, 0, 0]], [-62]))
    walls = mx(sl("x0", 2, 3), bottom)
    outside = affine(conv(ones, k3), -1, 9)
    wall_unit = cap(add(conv(walls, k3), outside))
    gulch_unit = cap(conv(sl("x0", 15, 16), k3))
    gulch_shapes = cap(shape_sum(gulch_unit))
    geometric_free = inv(cap(shape_sum(wall_unit)))
    all_body = cap(add(paint(objects), bomb_pixels))
    cross_body = paint(cat([affine(controllers, 0), cross]))

    def free_shapes(extra):
        return inv(cap(shape_sum(cap(add(wall_unit, conv(extra, k3))))))

    walk_extra = cap(add(add(bomb_pixels, cross_body), sl("x0", 15, 16)))
    walk_free = sl(free_shapes(walk_extra), 0, 4)

    def select_shape(field, typ):
        return total(mul(field, typ))

    stance_free = inv(cap(add(
        add(wall_unit, gulch_unit), conv(bomb_pixels, k3)
    )))
    sb = banks6(stance_free)
    w = np.zeros((20, 4, 7, 7), np.float32)
    for t, cs in enumerate(cells):
        occupied = set(cs)
        for d, (dy, dx) in enumerate(unit_dirs):
            for y in range(-1, 4):
                for x in range(-1, 4):
                    if (y, x) not in occupied and (y + dy, x + dx) in occupied:
                        yy, xx = y * 3, x * 3
                        by, bx = (6 if yy > 3 else 0), (6 if xx > 3 else 0)
                        bank = (2 if by else 0) + (1 if bx else 0)
                        w[4 * t + d, bank, 3 + yy - by, 3 + xx - bx] = 1
    abstract_stances = cap(conv(sb, w))
    expand_type = np.zeros((20, 5), np.float32)
    reduce_stances = np.zeros((4, 20), np.float32)
    for t in range(5):
        for d in range(4):
            expand_type[4 * t + d, t] = 1
            reduce_stances[d, 4 * t + d] = 1

    def stances(typ, helper):
        return mul(lin(mul(abstract_stances, lin(typ, expand_type)),
                       reduce_stances), helper)

    expand_dir = np.zeros((16, 4), np.float32)
    reduce_contact = np.zeros((4, 16), np.float32)
    for d in range(4):
        for t in range(4):
            expand_dir[4 * d + t, d] = 1
            reduce_contact[t, 4 * d + t] = 1

    def contact_goals(body, direction):
        overlap = sl(collision(body), 0, 4)
        ahead = cat([shift(overlap, dy, dx) for dy, dx in directions])
        contact = lin(mul(ahead, lin(direction, expand_dir)), reduce_contact)
        return mul(mul(contact, inv(overlap)), walk_free), overlap

    def scalar_route(free, goal, iterations):
        nonlocal current
        outer = current
        free_cap = affine(free, 200)
        seed = affine(mul(goal, free), 200)
        s = fresh("s")
        shapes[s] = (1, True)
        body = []
        current = body
        best = relu(cmax(neighbors(s, -1)))
        nxt = mx(seed, mn(best, free_cap))
        current = outer
        lid = fresh("L")
        current.append({
            "id": lid, "op": "loop", "T": iterations,
            "state": [{"id": s, "zeros": [1, 64, 64]}],
            "body": body, "next": {s: nxt}
        })
        shapes[lid + "." + s] = (1, True)
        return lid + "." + s

    def macro_plan(free, gulch, goal, stance, direct, iterations):
        nonlocal current
        outer = current
        free_cap = affine(free, 200)
        gulch_cap = affine(gulch, 200)
        dry = inv(gulch)
        next_free = neighbors(free)
        stop = inv(next_free)
        next_cap = affine(next_free, 200)
        walk_cap = affine(mul(mul(next_free, neighbors(dry)), direct), 200)
        stance_cap = affine(stance, 200)
        seed = affine(mul(goal, free), 200)
        states = []

        def state(c):
            name = fresh("s")
            shapes[name] = (c, True)
            states.append({"id": name, "zeros": [c, 64, 64]})
            return name

        value = state(1)
        momentum = [state(4) for _ in range(5)]
        walk_output = state(4)
        push_output = state(4)
        body = []
        current = body
        walk_q = mn(relu(neighbors(value, -1)), walk_cap)
        stopped = mul(value, stop)
        stationary = mul(value, dry)
        forwarded = [add(mn(dshift(m), next_cap), stopped) for m in momentum]
        nxt = {
            momentum[0]: add(mn(forwarded[0], gulch_cap), stationary),
            walk_output: walk_q
        }
        for j in range(1, 5):
            nxt[momentum[j]] = forwarded[j - 1]
        push_q = mn(relu(affine(forwarded[4], 1, -6)), stance_cap)
        nxt[push_output] = push_q
        nxt[value] = mn(mx(seed, cmax(mx(walk_q, push_q))), free_cap)
        current = outer
        lid = fresh("L")
        current.append({
            "id": lid, "op": "loop", "T": iterations,
            "state": states, "body": body, "next": nxt
        })
        for s in states:
            shapes[lid + "." + s["id"]] = shapes[s["id"]]
        return lid + "." + value, cat([
            lid + "." + walk_output, lid + "." + push_output
        ])

    def route_four(free, goal, iterations):
        nonlocal current
        outer = current
        fc = [affine(sl(free, t, t + 1), 200) for t in range(4)]
        seeds = [affine(sl(goal, t, t + 1), 200) for t in range(4)]
        states = []
        for _ in range(4):
            s = fresh("s")
            shapes[s] = (1, True)
            states.append({"id": s, "zeros": [1, 64, 64]})
        body = []
        current = body
        nxt = {}
        for t, s in enumerate(states):
            ns = [shift(s["id"], dy, dx, -1) for dy, dx in directions]
            best = ns[0]
            for v in ns[1:]:
                best = mx(best, v)
            nxt[s["id"]] = mx(seeds[t], mn(relu(best), fc[t]))
        current = outer
        lid = fresh("L")
        current.append({
            "id": lid, "op": "loop", "T": iterations,
            "state": states, "body": body, "next": nxt
        })
        for s in states:
            shapes[lid + "." + s["id"]] = (1, True)
        return cat([lid + "." + s["id"] for s in states])

    mp = sl("mem", 0, 2)
    mt = sl("mem", 2, 7)
    mc = sl("mem", 7, 15)
    msp = sl("mem", 15, 17)
    mst = sl("mem", 17, 21)
    moldp = sl("mem", 21, 23)
    msc = sl("mem", 23, 31)
    molds = sl("mem", 31, 33)

    primary, pmask, ptype, ppos = choose(
        unfinished, [12000, 12200, 12400, 12600, 30000],
        pred=mp, typ=mt
    )
    pbody = paint(primary)
    pfree = select_shape(free_shapes(relu(sub(all_body, pbody))), ptype)
    pgulch = select_shape(gulch_shapes, ptype)
    pgoal = select_shape(free_goals, ptype)
    pcross = sl(ptype, 4, 5)
    pcontroller = total(sl(ptype, 0, 4))
    helpers = cap(relu(sub(count, pcontroller)))
    pv, pq = macro_plan(
        pfree, pgulch, pgoal, stances(ptype, helpers), pcontroller, 48
    )
    pvalues = gm(mul(pq, pmask))
    ppath = cap(gm(mul(pv, pmask)))
    force_push = cap(add(pcross, mul(mul(inv(ppath), only_one), bomb_exists)))
    penalty = lin(force_push, [[-1000]] * 4 + [[0]] * 4)
    pcommit = mul(same_pos(ppos, moldp), dot(ptype, mt))
    legal_moves = gm(mul(
        neighbors(select_shape(geometric_free, ptype)), pmask
    ))
    pcommit = mul(pcommit, dot(legal_moves, lin(mc, merge_dirs)))
    pcommit = mul(pcommit, total(sl(mc, 4, 8)))
    pchoice = winner(add(pvalues, penalty), mc, pcommit)
    pdir = lin(pchoice, merge_dirs)
    pdirect = total(sl(pchoice, 0, 4))
    ppush = inv(pdirect)

    sgoals, _ = contact_goals(pbody, pdir)
    target_offset = lin(ptype, [
        [0, 1.5, 0, 1.5, 3], [0, 0, 1.5, 1.5, 3]
    ])
    stance_ref = add(ppos, affine(add(
        target_offset, lin(pdir, -6 * dmat)
    ), 1 / 63))
    spin = mul(pcommit, dot(pchoice, mc))
    secondary, smask, stype4, spos = choose(
        mul(controllers, inv(pmask)), [12000] * 4,
        ref=stance_ref, pred=msp, typ=mst, gate=spin
    )
    stype = lin(stype4, np.vstack([np.eye(4), np.zeros((1, 4))]))
    sbody = paint(cat([secondary, affine(cross, 0)]))
    sfree = select_shape(free_shapes(relu(sub(all_body, sbody))), stype)
    sgulch = select_shape(gulch_shapes, stype)
    sgoal = select_shape(sgoals, stype4)
    sv, sq = macro_plan(
        sfree, sgulch, sgoal,
        stances(stype, cap(relu(affine(count, 1, -1)))), onev, 40
    )

    # Prefer an actual walk to the stance over an abstract additional push.
    swalk = scalar_route(mul(sfree, inv(sgulch)), sgoal, 48)
    walk_path = cap(gm(mul(swalk, smask)))
    walk_values = gm(mul(neighbors(swalk), smask))
    macro_values = gm(mul(sq, smask))
    svalues = add(
        mul(macro_values, inv(walk_path)),
        mul(cat([walk_values, const([0] * 4)]), walk_path)
    )
    scommit = mul(mul(spin, same_pos(spos, molds)), inv(walk_path))
    scommit = mul(scommit, total(sl(msc, 4, 8)))
    schoice = winner(svalues, msc, scommit)
    sdir = lin(schoice, merge_dirs)
    sdirect = total(sl(schoice, 0, 4))
    satgoal = gm(total(mul(secondary, sgoals)))
    spath = cap(gm(mul(sv, smask)))
    ordinary_third_goals, scol = contact_goals(sbody, sdir)

    # Reachability in the controller's present dry region.
    active_free = select_shape(walk_free, gm(active))
    active_reach = scalar_route(active_free, active_mask, 64)
    reach_mask = cap(active_reach)
    near_reach = cap(conv(reach_mask, np.ones((1, 1, 7, 7), np.float32)))
    accessible_bombs = mul(
        bombs, cap(add(near_reach, shift(near_reach, 3, 3)))
    )

    fallback = mul(mul(mul(ppush, inv(spath)), only_one), bomb_exists)
    desired_ref = add(
        mul(ppos, pcross), mul(bbox_center(pgoal), inv(pcross))
    )
    _, bmask, _, bpos = choose(accessible_bombs, [40000], ref=active_pos)
    bdelta = sub(desired_ref, bpos)
    orientation = gm(mul(bomb_orientation, bmask))
    default_w = dmat.T.copy() * 63
    default_w[:2] *= 100
    blast = winner(add(affine(orientation, 1000000), lin(bdelta, default_w)))
    opposite = lin(blast, [
        [0, 1, 0, 0], [1, 0, 0, 0],
        [0, 0, 0, 1], [0, 0, 1, 0]
    ])
    empty_bomb = affine(bmask, 0)
    selected_bomb = paint(cat([
        empty_bomb, empty_bomb, empty_bomb, bmask, empty_bomb
    ]))
    front_goals, _ = contact_goals(selected_bomb, opposite)

    # Ray circuit accumulates possible landings 9--21 pixels ahead.
    flight_seed = add(mul(sv, pcross), mul(pv, inv(pcross)))
    ray_init = neighbors(flight_seed)
    outer = current
    ray_s, best_s, tick_s = fresh("s"), fresh("s"), fresh("s")
    shapes[ray_s] = (4, True)
    shapes[best_s] = (4, True)
    shapes[tick_s] = (1, True)
    body = []
    current = body
    next_ray = dshift(ray_s)
    allowed = mn(tick_s, ones)
    next_best = mx(best_s, mul(next_ray, allowed))
    next_tick = affine(tick_s, 1, 1)
    current = outer
    ray_id = fresh("L")
    current.append({
        "id": ray_id, "op": "loop", "T": 6,
        "state": [
            {"id": ray_s, "init": ray_init},
            {"id": best_s, "zeros": [4, 64, 64]},
            {"id": tick_s, "zeros": [1, 64, 64]}
        ],
        "body": body,
        "next": {ray_s: next_ray, best_s: next_best, tick_s: next_tick}
    })
    for s in (ray_s, best_s, tick_s):
        shapes[ray_id + "." + s] = shapes[s]
    useful_lane = cap(total(mul(ray_id + "." + best_s, blast)))
    has_lane = cap(gm(total(mul(front_goals, useful_lane))))
    perpendicular = lin(blast, [
        [0, 0, 1, 1], [0, 0, 1, 1],
        [1, 1, 0, 0], [1, 1, 0, 0]
    ])
    bomb_moves = gm(mul(neighbors(sl(geometric_free, 3, 4)), bmask))
    fixed_lateral = inv(cap(dot(bomb_moves, perpendicular)))
    aligned = cap(add(has_lane, fixed_lateral))
    side_push = winner(add(
        lin(bdelta, dmat.T * 63),
        affine(perpendicular, 10000, -10000)
    ))
    lateral_goals, _ = contact_goals(selected_bomb, side_push)
    firing_goals = mul(front_goals, mx(useful_lane, inv(has_lane)))
    bomb_goals = add(mul(firing_goals, aligned),
                     mul(lateral_goals, inv(aligned)))
    tactical_direction = add(mul(blast, aligned), mul(side_push, inv(aligned)))

    third_goals = add(mul(ordinary_third_goals, inv(fallback)),
                      mul(bomb_goals, fallback))
    third_free = mul(walk_free, inv(mul(scol, inv(fallback))))
    third_goals = mul(third_goals, third_free)
    tv = route_four(third_free, third_goals, 56)
    third_candidates = mul(controllers, inv(mul(smask, inv(fallback))))
    third_ref = add(mul(stance_ref, inv(fallback)), mul(bpos, fallback))
    third, tmask, ttype, tpos = choose(
        third_candidates, [12000] * 4, ref=third_ref,
        extra=add(affine(tv, 50), affine(active, 2))
    )
    tatgoal = gm(total(mul(third, third_goals)))
    tq = [gm(total(mul(third, shift(tv, dy, dx))))
          for dy, dx in directions]
    tnav = winner(cat(tq))
    ttrigger = add(mul(sdir, inv(fallback)), mul(tactical_direction, fallback))
    tdir = add(mul(ttrigger, tatgoal), mul(tnav, inv(tatgoal)))

    use_primary = mul(pdirect, inv(fallback))
    use_secondary = mul(
        mul(ppush, add(satgoal, mul(inv(satgoal), sdirect))),
        inv(fallback)
    )
    use_third = inv(add(use_primary, use_secondary))
    secondary_direction = add(mul(pdir, satgoal), mul(sdir, inv(satgoal)))
    desired_dir = add(add(
        mul(pdir, use_primary), mul(secondary_direction, use_secondary)
    ), mul(tdir, use_third))

    selected = add(add(
        mul(sl(primary, 0, 4), use_primary), mul(secondary, use_secondary)
    ), mul(third, use_third))
    selected_active = gm(total(mul(selected, active)))
    affine(total(selected), name="click")
    output_w = np.zeros((8, 5), np.float32)
    output_w[1:5, :4] = np.eye(4) * 10
    output_w[6, 4] = 20
    lin(cat([desired_dir, inv(selected_active)]), output_w,
        [-100, 0, 0, 0, 0, -100, 0, -100], name="logits")

    third_is_primary = gm(total(mul(third, sl(primary, 0, 4))))
    pdistance = add(affine(use_primary, 3),
                    affine(mul(use_secondary, satgoal), 15))
    pdelta1 = mul(lin(pdir, dmat), pdistance)
    pdelta2 = mul(lin(tnav, dmat), affine(mul(
        mul(use_third, inv(tatgoal)), third_is_primary
    ), 3))
    predicted_primary = add(ppos, affine(
        mul(add(pdelta1, pdelta2), selected_active), 1 / 63
    ))
    sdistance = add(
        affine(mul(use_secondary, inv(satgoal)), 3),
        affine(mul(mul(use_third, tatgoal), inv(fallback)), 15)
    )
    predicted_secondary = add(spos, affine(
        mul(mul(lin(sdir, dmat), sdistance), selected_active), 1 / 63
    ))
    memory_output = cat([
        predicted_primary, ptype, pchoice,
        predicted_secondary, stype4, ppos, schoice, spos
    ])
    affine(memory_output, name="mem_next")

    # Keep only the output dependency graph.
    top = {nd["id"]: nd for nd in graph}
    exports = {}
    for nd in graph:
        if nd["op"] == "loop":
            for s in nd["state"]:
                exports[nd["id"] + "." + s["id"]] = nd["id"]
    needed = set()

    def mark(name):
        name = exports.get(name, name)
        if name not in top or name in needed:
            return
        needed.add(name)
        nd = top[name]
        if nd["op"] != "loop":
            for source in nd["in"]:
                mark(source)
            return
        local = {s["id"] for s in nd["state"]}
        local.update(child["id"] for child in nd["body"])
        for s in nd["state"]:
            if "init" in s:
                mark(s["init"])
        for child in nd["body"]:
            for source in child.get("in", []):
                if source not in local:
                    mark(source)

    for out in ("logits", "click", "mem_next"):
        mark(out)
    graph = [nd for nd in graph if nd["id"] in needed]

    used = set()

    def collect(nodes):
        for nd in nodes:
            for key in ("w", "b"):
                if key in nd:
                    used.add(nd[key])
            if nd["op"] == "loop":
                collect(nd["body"])

    collect(graph)
    P = {k: v for k, v in P.items() if k in used}

    def mac(nd):
        op = nd["op"]
        c = shapes[nd["id"]][0]
        if op == "conv":
            return int(np.prod(P[nd["w"]].shape)) * 4096
        if op == "dense":
            return int(np.prod(P[nd["w"]].shape))
        if op in ("add", "sub", "mul", "max2", "min2"):
            return c * 4096
        if op in ("vadd", "vmul"):
            return c
        if op in ("rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"):
            return shapes[nd["in"][0]][0] * 4096
        return 0

    long_loops = []
    fixed_evals = 0
    fixed_macs = 0
    for nd in graph:
        if nd["op"] != "loop":
            fixed_evals += 1
            fixed_macs += mac(nd)
        elif nd["T"] < 20:
            fixed_evals += 1 + nd["T"] * len(nd["body"])
            fixed_macs += nd["T"] * sum(mac(x) for x in nd["body"])
        else:
            fixed_evals += 1
            long_loops.append(nd)

    requested_evals = sum(nd["T"] * len(nd["body"]) for nd in long_loops)
    requested_macs = sum(
        nd["T"] * sum(mac(x) for x in nd["body"]) for nd in long_loops
    )
    ratio = min(
        1.0,
        (9750 - fixed_evals) / max(1, requested_evals),
        (1_450_000_000 - fixed_macs) / max(1, requested_macs)
    )
    for nd in long_loops:
        nd["T"] = max(1, int(math.floor(nd["T"] * ratio)))

    return {"graph": graph, "memory_size": 33, "params": P}
