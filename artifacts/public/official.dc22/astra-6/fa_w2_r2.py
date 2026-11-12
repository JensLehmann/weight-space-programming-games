import numpy as np


def build():
    P, G = {}, []
    g = G
    serial = 0
    pcache, ncache = {}, {}
    affine = {}
    constants = {}

    def uid():
        nonlocal serial
        serial += 1
        return "n%d" % serial

    def param(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in pcache:
            name = "p%d" % len(P)
            pcache[key] = name
            P[name] = a
        return pcache[key]

    def node(op, ins=None, name=None, **kw):
        key = (id(g), op, tuple(ins or []), repr(sorted(kw.items())))
        if name is None and op != "loop" and key in ncache:
            return ncache[key]
        name = name or uid()
        nd = {"id": name, "op": op}
        if op != "loop":
            nd["in"] = list(ins or [])
        nd.update(kw)
        g.append(nd)
        if op != "loop":
            ncache[key] = name
        return name

    def conv(x, w, b=None):
        kw = {"w": param(w)}
        if b is not None:
            kw["b"] = param(np.asarray(b, np.float32).reshape(-1))
        return node("conv", [x], **kw)

    def dense(x, w, b=None):
        kw = {"w": param(w)}
        if b is not None:
            kw["b"] = param(np.asarray(b, np.float32).reshape(-1))
        return node("dense", [x], **kw)

    def sl(x, i, j=None):
        return node("slice", [x], start=i, stop=i + 1 if j is None else j)

    def vs(x, i, j=None):
        return node("vslice", [x], start=i, stop=i + 1 if j is None else j)

    def cat(xs):
        return xs[0] if len(xs) == 1 else node("concat", xs)

    one = sl("coords", 2)
    constants[one] = 1.0

    def aff(x, a=1.0, b=0.0):
        a, b = float(a), float(b)
        if x in constants:
            b = a * constants[x] + b
            a, x = 0.0, one
            if b == 1.0:
                return one
        elif x in affine:
            root, aa, bb = affine[x]
            x, a, b = root, a * aa, a * bb + b
        if a == 1.0 and b == 0.0:
            return x
        y = conv(x, np.array([[[[a]]]], np.float32),
                 None if b == 0.0 else [b])
        affine[y] = (x, a, b)
        if a == 0.0:
            constants[y] = b
        return y

    def add(a, b):
        if b in constants:
            return aff(a, 1, constants[b])
        if a in constants:
            return aff(b, 1, constants[a])
        return node("add", [a, b])

    def sub(a, b):
        if b in constants:
            return aff(a, 1, -constants[b])
        if a in constants:
            return aff(b, -1, constants[a])
        return node("sub", [a, b])

    def mul(a, b):
        if b in constants:
            return aff(a, constants[b])
        if a in constants:
            return aff(b, constants[a])
        return node("mul", [a, b])

    def mx(a, b):
        return a if a == b else node("max2", [a, b])

    def mn(a, b):
        return a if a == b else node("min2", [a, b])

    zero = aff(one, 0)
    X = aff(sl("coords", 1), 63)
    Y = aff(sl("coords", 0), 63)

    def const(v):
        return aff(one, 0, v)

    def inv(x):
        return aff(x, -1, 1)

    def clip(x):
        return mn(node("relu", [x]), one)

    def gt(x, t=0.5):
        return clip(aff(x, 4, -4 * t))

    def ge(x, n):
        return gt(x, n - 0.5)

    def gte(a, b):
        return gt(sub(a, b), -0.5)

    def union(xs):
        z = xs[0]
        for x in xs[1:]:
            z = mx(z, x)
        return z

    def prod(xs):
        z = xs[0]
        for x in xs[1:]:
            z = mul(z, x)
        return z

    def lift(v):
        # Deliberately not constant-folded: v may be a vector.
        return node("mul", [one, v])

    def glob(x, op="gmax"):
        return lift(node(op, [x]))

    def total(x):
        return aff(glob(x, "gmean"), 4096)

    def sample(x, dy=0, dx=0):
        while dy or dx:
            sy = max(-3, min(3, dy))
            sx = max(-3, min(3, dx))
            r = max(abs(sy), abs(sx))
            w = np.zeros((1, 1, 2 * r + 1, 2 * r + 1), np.float32)
            w[0, 0, r + sy, r + sx] = 1
            x = conv(x, w)
            dy -= sy
            dx -= sx
        return x

    def move_image(x, dy=0, dx=0):
        return sample(x, -dy, -dx)

    def boxsum(x, h=2, w=2):
        z = np.zeros((1, 1, 7, 7), np.float32)
        z[0, 0, 3:3 + h, 3:3 + w] = 1
        return conv(x, z)

    def fit(x, h=2, w=2):
        z = np.zeros((1, 1, 7, 7), np.float32)
        z[0, 0, 3:3 + h, 3:3 + w] = 1
        return node("relu", [conv(x, z, [1 - h * w])])

    def paint(x, h, w):
        z = np.zeros((1, 1, 7, 7), np.float32)
        z[0, 0, 4 - h:4, 4 - w:4] = 1
        return clip(conv(x, z))

    def dilate(x, ry=0, rx=0):
        r = max(ry, rx)
        w = np.zeros((1, 1, 2 * r + 1, 2 * r + 1), np.float32)
        w[0, 0, r - ry:r + ry + 1, r - rx:r + rx + 1] = 1
        return clip(conv(x, w))

    def bounds(mask, coord):
        hi = glob(mul(mask, coord))
        lo = aff(glob(mul(mask, aff(coord, -1, 64))), -1, 64)
        return lo, hi

    def between(z, lo, hi):
        return mul(gte(z, lo), gte(hi, z))

    def bands(z, center, intervals):
        """Parallel closed-interval tests, with half-pixel margins."""
        intervals = np.asarray(intervals, np.float32)
        n = len(intervals)
        mid = intervals.mean(axis=1)
        half = (intervals[:, 1] - intervals[:, 0]) * 0.5
        delta = sub(z, center)
        q = conv(delta, np.ones((n, 1, 1, 1), np.float32), -mid)
        q = node("abs", [q])
        q = conv(q, (-4 * np.eye(n, dtype=np.float32)).reshape(n, n, 1, 1),
                 4 * (half + 0.5))
        q = mn(node("relu", [q]), one)
        return [sl(q, i) for i in range(n)]

    raw = [sl("x0", i) for i in range(16)]
    ui_column = ge(node("colsum", [raw[5]]), 20)
    board = mul(inv(ui_column), ge(aff(Y, -1, 62), 0))
    c = [mul(q, board) for q in raw]
    u = [mul(q, ui_column) for q in raw]

    # Large switches and small crane controls.
    buttons = {}
    cap_kernel = np.ones((1, 1, 5, 5), np.float32)
    for key, color in [("a", 8), ("b", 9), ("c", 6),
                       ("d", 15), ("e", 2)]:
        buttons[key] = node("relu", [conv(u[color], cap_kernel, [-24])])
    buttons["f"] = u[11]

    square_kernel = np.zeros((1, 1, 7, 7), np.float32)
    square_kernel[0, 0, 2:6, 2:6] = 1
    square_kernel[0, 0, 1, 2:6] = -2
    square_kernel[0, 0, 6, 2:6] = -2
    square_kernel[0, 0, 2:6, 1] = -2
    square_kernel[0, 0, 2:6, 6] = -2
    tiny10 = node("relu", [conv(u[10], square_kernel, [-15])])
    tiny9 = node("relu", [conv(u[9], square_kernel, [-15])])
    for mask, left_name, right_name in [
        (tiny10, "left", "right"), (tiny9, "up", "down")
    ]:
        lo, hi = bounds(mask, X)
        midpoint = aff(add(lo, hi), 0.5)
        left_half = gt(sub(midpoint, X), 0.25)
        buttons[left_name] = mul(mask, left_half)
        buttons[right_name] = mul(mask, inv(left_half))

    grab_kernel = np.zeros((1, 1, 5, 5), np.float32)
    grab_kernel[0, 0, 1:4, 1:4] = 1
    grab_kernel[0, 0, 0, 2] = -10
    grab_kernel[0, 0, 4, 2] = -10
    buttons["grab"] = node("relu", [conv(u[8], grab_kernel, [-8])])
    have = {k: glob(buttons[k]) for k in ("a", "b", "c", "d")}

    # Reconstruct bridge arms and their quarter-turned alternatives.
    pivot = fit(add(c[13], c[14]), 4, 4)
    arm_pixels = add(c[8], c[14])
    sensors = cat([
        sample(arm_pixels, 1, 0), sample(arm_pixels, 1, 3),
        sample(arm_pixels, 0, 1), sample(arm_pixels, 3, 1)
    ])
    piv4 = cat([pivot] * 4)
    ss, cc, ll = uid(), uid(), uid()
    outer = g
    g = []
    sn = cat([
        sample(sl(ss, i), dy, dx)
        for i, (dy, dx) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)])
    ])
    cn = mn(cc, sn)
    ln = add(ll, cn)
    body = g
    g = outer
    measure = node(
        "loop", T=16,
        state=[{"id": ss, "init": sensors},
               {"id": cc, "init": piv4},
               {"id": ll, "zeros": [4, 64, 64]}],
        body=body, next={ss: sn, cc: cn, ll: ln}
    )
    lengths = [sl(measure + "." + ll, i) for i in range(4)]
    sources = cat(lengths + [lengths[2], lengths[3],
                             lengths[0], lengths[1]])
    ww = uid()
    outer = g
    g = []
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] * 2
    transported = cat([
        move_image(sl(ww, i), dy, dx)
        for i, (dy, dx) in enumerate(directions)
    ])
    decay_w = np.eye(8, dtype=np.float32).reshape(8, 8, 1, 1)
    decayed = node("relu", [conv(transported, decay_w, [-1] * 8)])
    wn = mx(sources, decayed)
    body = g
    g = outer
    draw = node(
        "loop", T=16, state=[{"id": ww, "init": sources}],
        body=body, next={ww: wn}
    )
    drawn = []
    for i in range(8):
        z = sl(draw + "." + ww, i)
        kind = i % 4
        if kind == 0:
            z = paint(move_image(z, 0, -1), 4, 1)
        elif kind == 1:
            z = paint(move_image(z, 0, 4), 4, 1)
        elif kind == 2:
            z = paint(move_image(z, -1, 0), 1, 4)
        else:
            z = paint(move_image(z, 4, 0), 1, 4)
        drawn.append(z)
    pivot_floor = paint(pivot, 4, 4)
    bridge_now = union([pivot_floor] + drawn[:4])
    bridge_other = union([pivot_floor] + drawn[4:])

    player = fit(c[14])
    goal = fit(c[11])
    visible_portals = mul(fit(add(c[6], c[7])), ge(boxsum(c[7]), 1))
    portal_count = total(visible_portals)
    odd_terms = []
    for k in (1, 3, 5, 7):
        delta = node("abs", [aff(portal_count, 1, -k)])
        odd_terms.append(node("relu", [aff(delta, -1, 1)]))
    portals = union([visible_portals, mul(player, union(odd_terms))])

    # Pickup memory survives temporary visual occlusion.
    pickup_kernel = np.zeros((1, 2, 7, 7), np.float32)
    for dy, dx in [(0, 1), (0, 2),
                   (1, 0), (1, 1), (1, 2), (1, 3)]:
        pickup_kernel[0, 0, 3 + dy, 3 + dx] = 1
    for dx in range(4):
        pickup_kernel[0, 1, 4, 3 + dx] = 1
    pickup_kernel[0, 0, 3, 3] = -4
    pickup_kernel[0, 0, 3, 6] = -4
    pickup_seen = node(
        "relu", [conv(cat([c[15], c[8]]), pickup_kernel, [-5])]
    )
    seen = glob(pickup_seen)
    unseen = inv(seen)
    pickup_x = add(glob(mul(pickup_seen, X)),
                   mul(lift(vs("mem", 1)), unseen))
    pickup_y = add(glob(mul(pickup_seen, Y)),
                   mul(lift(vs("mem", 2)), unseen))
    pickup_exists = mul(mx(seen, lift(vs("mem", 3))), inv(have["d"]))
    pickup = prod([
        pickup_exists,
        inv(gt(node("abs", [sub(X, pickup_x)]), 0.25)),
        inv(gt(node("abs", [sub(Y, pickup_y)]), 0.25))
    ])
    pickup_floor = paint(pickup, 2, 4)

    neighbor_kernel = np.zeros((1, 1, 5, 5), np.float32)
    for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        neighbor_kernel[0, 0, 2 + dy, 2 + dx] = 1

    # Solid/checkerboard switch states, including player occlusion.
    dynamic, covered_dynamic = [], []
    for color in (9, 15):
        restored = paint(fit(add(c[color], c[14]), 4, 4), 4, 4)
        solid = fit(mx(c[color], restored))
        checks = []
        for parity in (0, 1):
            w = np.zeros((1, 1, 3, 3), np.float32)
            for dy in range(2):
                for dx in range(2):
                    w[0, 0, 1 + dy, 1 + dx] = (
                        1 if (dy + dx) % 2 == parity else -2
                    )
            checks.append(node("relu", [conv(c[color], w, [-1])]))
        dynamic.append((solid, add(checks[0], checks[1])))
        covered_dynamic.append(mul(player, solid))

    # Thin cyclic platforms use bounded geometric envelopes.
    line_raw = union([c[1], c[12], c[5], c[14]])
    thin = prod([
        line_raw, sample(line_raw, 1, 0),
        inv(sample(line_raw, -1, 0)), inv(sample(line_raw, 2, 0))
    ])
    line_count = node("rowsum", [thin])
    e_row = ge(line_count, 10)
    f_row = mul(ge(line_count, 4), inv(e_row))
    e_top = mul(thin, e_row)
    f_top = mul(thin, f_row)
    e_union = paint(e_top, 2, 1)
    thin_floor = paint(thin, 2, 1)

    land = union([c[2], c[9], c[15], c[13], c[8], goal, portals])
    near_land = dilate(dilate(land, 3, 0), 1, 0)
    land_hi = node("rowmax", [mul(near_land, X)])
    land_lo = aff(
        node("rowmax", [mul(near_land, aff(X, -1, 64))]), -1, 64
    )
    interval = between(X, aff(land_lo, 1, 4), land_hi)
    f_extended = f_top
    for r in (3, 3, 3, 1):
        f_extended = dilate(f_extended, 0, r)
    f_union = paint(union([f_top, mul(f_extended, interval)]), 2, 1)

    # Locate the claw and infer its track from visible geometry.
    claw_kernel = np.zeros((1, 1, 7, 7), np.float32)
    positives = []
    for dy in (-1, 2):
        positives += [(dy, dx) for dx in range(4)]
    for dy in (0, 1):
        positives += [(dy, dx) for dx in range(-1, 3)]
    for dy, dx in positives:
        claw_kernel[0, 0, 3 + dy, 3 + dx] = 1
    for dy, dx in [(-1, -1), (2, -1), (0, 3), (1, 3)]:
        claw_kernel[0, 0, 3 + dy, 3 + dx] = -4
    claw = node("relu", [conv(c[8], claw_kernel, [-15])])
    has_claw = glob(claw)
    claw_x = aff(glob(mul(claw, X)), 1, 8)
    claw_y = aff(glob(mul(claw, Y)), 1, 1)
    attached = glob(mul(claw, inv(sample(c[8], -3, 2))))

    obj = mul(c[12], inv(thin_floor))
    has_crane = mul(has_claw, glob(obj))
    ox0, ox1 = bounds(obj, X)
    oy0, oy1 = bounds(obj, Y)
    object_x = add(mul(attached, claw_x),
                   mul(inv(attached), aff(add(ox0, ox1), 0.5, 0.5)))
    object_y = add(mul(attached, claw_y),
                   mul(inv(attached), aff(add(oy0, oy1), 0.5, 0.5)))
    tx0, unused_tx1 = bounds(c[0], X)
    ty0, ty1 = bounds(c[0], Y)
    base_x = aff(tx0, 1, 8)
    bottom_y = aff(ty1, 1, -1)

    poses = [(3, 0), (2, 0), (1, 0), (0, 0), (0, 1),
             (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]
    x_regions = bands(X, base_x, [(4 * k - 1, 4 * k + 2) for k in range(4)])
    y_regions = bands(Y, bottom_y,
                      [(-4 * k - 6, -4 * k + 5) for k in range(4)])
    x_match = bands(object_x, base_x, [(4 * k, 4 * k) for k in range(4)])
    y_match = bands(object_y, bottom_y, [(-4 * k, -4 * k) for k in range(4)])
    pose_regions = [mul(x_regions[x], y_regions[y]) for x, y in poses]
    pose_matches = cat([mul(x_match[x], y_match[y]) for x, y in poses])
    pose_indicator = node("gmax", [mul(pose_matches, has_crane)])
    current_object = conv(
        node("mul", [cat(pose_regions), pose_indicator]),
        np.ones((1, 10, 1, 1), np.float32)
    )

    red_floor = mx(c[2], paint(fit(add(c[2], c[14]), 4, 4), 4, 4))
    neutral_player = mul(
        player, inv(union(covered_dynamic + [bridge_now, current_object]))
    )
    fixed12 = mul(c[12], mx(thin_floor, inv(has_crane)))
    static_base = union([
        red_floor, c[1], fixed12, neutral_player,
        goal, portals, pickup_floor
    ])
    optimistic_base = union([static_base, e_union, f_union])
    physical = union([
        static_base, current_object, bridge_now,
        dynamic[0][0], dynamic[1][0]
    ])
    claw_pixels = mul(c[8], inv(bridge_now))
    claw_collision = ge(boxsum(claw_pixels), 1)
    physical = mul(physical, inv(claw_collision))

    # Eight switch layers plus ten crane poses with four switch layers each.
    supports = []
    for mode in range(8):
        supports.append(union([
            optimistic_base, current_object,
            bridge_other if mode & 1 else bridge_now,
            dynamic[0][1 if mode & 2 else 0],
            dynamic[1][1 if mode & 4 else 0]
        ]))
    crane_bases = []
    for mode in range(4):
        crane_bases.append(union([
            optimistic_base, bridge_now,
            dynamic[0][1 if mode & 1 else 0],
            dynamic[1][1 if mode & 2 else 0]
        ]))
    crane_base_stack = cat(crane_bases)
    support_stack = cat(
        supports + [mx(crane_base_stack, region) for region in pose_regions]
    )
    objective = add(mul(goal, inv(pickup_exists)), pickup)
    seeds = mn(objective, support_stack)

    # Batched wavefront propagation. Accumulated reachability encodes distance.
    reach_id, score_id = uid(), uid()
    outer = g
    g = []
    rc = [sl(reach_id, i) for i in range(48)]
    flood_kernel = neighbor_kernel.copy()
    flood_kernel[0, 0, 2, 2] = 1
    grouped_kernel = np.zeros((3, 3, 5, 5), np.float32)
    for i in range(3):
        grouped_kernel[i, i] = flood_kernel[0, 0]
    walking = cat([
        conv(sl(reach_id, i, i + 3), grouped_kernel)
        for i in range(0, 48, 3)
    ])
    edges = [walking]
    for key, normal_bit, crane_bit in [("a", 1, 0), ("b", 2, 1), ("d", 4, 2)]:
        switched = [rc[i ^ normal_bit] for i in range(8)]
        switched += [
            rc[8 + 4 * p + (m ^ crane_bit)]
            for p in range(10) for m in range(4)
        ]
        edges.append(mul(cat(switched), have[key]))
    for direction in (-1, 1):
        shifted = [zero] * 8
        for p in range(10):
            q = p + direction
            shifted.extend(
                [rc[8 + 4 * q + m] for m in range(4)]
                if 0 <= q < 10 else [zero] * 4
            )
        edges.append(cat(shifted))
    portal_global = node("gmax", [mul(reach_id, portals)])
    portal_broadcast = node("mul", [portals, portal_global])
    edges.append(mul(portal_broadcast, have["c"]))
    joined = edges[0]
    for edge in edges[1:]:
        joined = add(joined, edge)
    reach_next = mn(joined, support_stack)
    score_next = add(score_id, reach_next)
    body = g
    g = outer
    flood = node(
        "loop", T=72,
        state=[{"id": reach_id, "init": seeds},
               {"id": score_id, "init": seeds}],
        body=body, next={reach_id: reach_next, score_id: score_next}
    )
    values = flood + "." + score_id

    indicator = node("vconcat", [
        node("gmax", [inv(has_crane)]), pose_indicator
    ])
    selector_w = np.zeros((48, 11), np.float32)
    selector_w[:8, 0] = 1
    for p in range(10):
        selector_w[8 + 4 * p:12 + 4 * p, 1 + p] = 1
    selector = dense(indicator, selector_w)
    selected_values = node("mul", [values, selector])
    value_w = np.zeros((1, 48, 1, 1), np.float32)
    value_w[0, 0, 0, 0] = 1
    for p in range(10):
        value_w[0, 8 + 4 * p, 0, 0] = 1
    value0 = conv(selected_values, value_w)

    pooled_values = node("gmax", [mul(values, player)])
    selected_pool = node("vmul", [pooled_values, selector])
    switch_w = np.zeros((3, 48), np.float32)
    switch_w[0, 1] = 1
    switch_w[1, 2] = 1
    switch_w[2, 4] = 1
    for p in range(10):
        switch_w[1, 8 + 4 * p + 1] = 1
        switch_w[2, 8 + 4 * p + 2] = 1
    switch_values = dense(selected_pool, switch_w)

    def at_player(x):
        return glob(mul(player, x))

    no_path = inv(ge(at_player(value0), 1))
    clock = lift(vs("mem", 0))
    advanced = aff(clock, 1, 1)
    next_clock = sub(advanced, aff(ge(advanced, 16), 16))
    node("vconcat", [
        node("gmax", [next_clock]),
        node("gmax", [pickup_x]),
        node("gmax", [pickup_y]),
        node("gmax", [pickup_exists])
    ], name="mem_next")

    phase = conv(clock, np.ones((15, 1, 1, 1), np.float32),
                 -np.arange(15, dtype=np.float32))
    phase = node("abs", [phase])
    phase = conv(
        phase, (-0.2 * np.eye(15, dtype=np.float32)).reshape(15, 15, 1, 1),
        np.full(15, 0.2, np.float32)
    )
    phase = node("relu", [phase])
    fallback_stack = add(mul(phase, no_path), aff(no_path, 0.1))
    fallback = [sl(fallback_stack, i) for i in range(15)]
    keys = ["a", "b", "c", "d", "e", "f",
            "left", "up", "right", "down", "grab"]
    fallback_for = dict(zip(keys, fallback))

    def penalize(value, valid):
        return add(value, aff(valid, 1000, -1000))

    click_scores = {}
    for i, key in enumerate(("a", "b", "d")):
        value = lift(vs(switch_values, i))
        click_scores[key] = penalize(
            add(value, fallback_for[key]), ge(value, 1)
        )
    other_portals = mul(portals, inv(player))
    click_scores["c"] = penalize(
        add(glob(mul(value0, other_portals)), fallback_for["c"]),
        at_player(portals)
    )

    frontier = conv(player, neighbor_kernel)
    missing = mul(frontier, inv(physical))
    for key, region in [("e", e_union), ("f", f_union)]:
        urgency = glob(mul(mul(missing, region), value0))
        click_scores[key] = penalize(
            add(urgency, fallback_for[key]), mx(ge(urgency, 1), no_path)
        )

    # Destination-layer values reject crane moves that remove support.
    crane_step_values = {}
    for key in ("left", "right", "up", "down"):
        w = np.zeros((48, 11), np.float32)
        for p, (x, y) in enumerate(poses):
            for q in (p - 1, p + 1):
                if not 0 <= q < 10:
                    continue
                xx, yy = poses[q]
                direction = (
                    "left" if xx < x else "right" if xx > x
                    else "up" if yy > y else "down"
                )
                if direction == key:
                    w[8 + 4 * q, 1 + p] = 1
        target_selector = dense(indicator, w)
        weighted = node("vmul", [pooled_values, target_selector])
        crane_step_values[key] = lift(
            dense(weighted, np.ones((1, 48), np.float32))
        )
    best_change = union(list(crane_step_values.values()))

    # Fetch the object before issuing carrying moves.
    vertical_difference = sub(claw_y, object_y)
    need_vertical = gt(node("abs", [vertical_difference]), 0.5)
    off_left = gt(sub(claw_x, base_x), 0.5)
    fetch = {}
    fetch["left"] = union([
        mul(need_vertical, off_left),
        mul(inv(need_vertical), gt(sub(claw_x, object_x), 0.5))
    ])
    fetch["right"] = mul(
        inv(need_vertical), gt(sub(object_x, claw_x), 0.5)
    )
    fetch["up"] = prod([
        need_vertical, inv(off_left), gt(vertical_difference, 0.5)
    ])
    fetch["down"] = prod([
        need_vertical, inv(off_left), gt(aff(vertical_difference, -1), 0.5)
    ])
    fetch["grab"] = mul(
        inv(gt(node("abs", [sub(claw_x, object_x)]), 0.5)),
        inv(gt(node("abs", [sub(claw_y, object_y)]), 0.5))
    )
    for key in ("left", "right", "up", "down", "grab"):
        empty_value = prod([inv(attached), fetch[key], best_change])
        value = empty_value if key == "grab" else add(
            empty_value, mul(attached, crane_step_values[key])
        )
        click_scores[key] = penalize(value, ge(value, 1))

    score_stack = cat([click_scores[key] for key in keys])
    score_stack = conv(
        score_stack, np.eye(11, dtype=np.float32).reshape(11, 11, 1, 1),
        np.full(11, 1000.125, np.float32)
    )
    candidates = mul(cat([buttons[key] for key in keys]), score_stack)
    click_map = aff(union([sl(candidates, i) for i in range(11)]), 1, -1000)
    node("add", [click_map, zero], name="click")

    movement_scores = []
    for i, (dy, dx) in enumerate([(-2, 0), (2, 0), (0, -2), (0, 2)]):
        value = add(at_player(sample(value0, dy, dx)), fallback[11 + i])
        valid = at_player(sample(physical, dy, dx))
        movement_scores.append(penalize(value, valid))

    negative = const(-10000)
    output_maps = cat([
        negative, movement_scores[0], movement_scores[1],
        movement_scores[2], movement_scores[3],
        negative, click_map, negative
    ])
    node("gmax", [output_maps], name="logits")

    # Static bookkeeping: retain only nodes and arrays feeding the outputs.
    by_id = {nd["id"]: nd for nd in G}
    live = set()

    def external_dependencies(nd):
        if nd["op"] != "loop":
            return nd.get("in", [])
        local_ids = {s["id"] for s in nd["state"]}
        local_ids.update(q["id"] for q in nd["body"])
        deps = [s["init"] for s in nd["state"] if "init" in s]
        for q in nd["body"]:
            deps.extend(x for x in q.get("in", []) if x not in local_ids)
        return deps

    def mark(ref):
        base = ref.split(".", 1)[0]
        if base not in by_id or base in live:
            return
        live.add(base)
        for dep in external_dependencies(by_id[base]):
            mark(dep)

    for output in ("logits", "click", "mem_next"):
        mark(output)
    G = [nd for nd in G if nd["id"] in live]

    used_params = set()

    def collect(nodes):
        for nd in nodes:
            for field in ("w", "b"):
                if field in nd:
                    used_params.add(nd[field])
            if nd["op"] == "loop":
                collect(nd["body"])

    collect(G)
    P = {name: value for name, value in P.items() if name in used_params}
    return {"graph": G, "memory_size": 4, "params": P}
