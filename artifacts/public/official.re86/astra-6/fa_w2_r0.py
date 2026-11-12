import numpy as np


def build():
    G = []
    root = G
    P = {}
    S = {
        "x0": (1, 16), "coords": (1, 3), "mem": (0, 97),
        "aprev": (0, 8), "avail": (0, 8)
    }
    serial = 0
    cache = {}
    const_cache = {}
    const_values = {}
    affine_cache = {}

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def param(a):
        p = fresh("p")
        P[p] = np.asarray(a, np.float32)
        return p

    def emit(op, ins, shape, name=None, **fields):
        key = (id(G), op, tuple(ins), tuple(sorted(fields.items())))
        if name is None and key in cache:
            return cache[key]
        n = name or fresh()
        d = {"id": n, "op": op, "in": list(ins)}
        d.update(fields)
        G.append(d)
        S[n] = shape
        if name is None:
            cache[key] = n
        return n

    def const(a):
        a = np.asarray(a, np.float32).reshape(-1)
        key = tuple(float(v) for v in a)
        if key in const_cache:
            return const_cache[key]
        n = fresh()
        root.append({
            "id": n, "op": "dense", "in": ["avail"],
            "w": param(np.zeros((len(a), 8), np.float32)),
            "b": param(a)
        })
        S[n] = (0, len(a))
        const_cache[key] = n
        const_values[n] = a
        return n

    def scalar_is(x, value):
        return (
            x in const_values and len(const_values[x]) == 1
            and float(const_values[x][0]) == value
        )

    def lin(x, w, b=None, name=None):
        spatial, ci = S[x]
        w = np.asarray(w, np.float32).reshape(-1, ci)
        co = len(w)
        if spatial:
            w = w[:, :, None, None]
        fields = {"w": param(w)}
        if b is not None:
            b = np.asarray(b, np.float32)
            if b.ndim == 0:
                b = np.full(co, float(b), np.float32)
            fields["b"] = param(b.reshape(co))
        return emit("conv" if spatial else "dense", [x],
                    (spatial, co), name=name, **fields)

    def aff(x, a=1., b=0.):
        if a == 1 and b == 0:
            return x
        key = (id(G), x, float(a), float(b))
        if key not in affine_cache:
            c = S[x][1]
            affine_cache[key] = lin(
                x, np.eye(c, dtype=np.float32) * a,
                None if b == 0 else np.full(c, b, np.float32)
            )
        return affine_cache[key]

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(np.asarray(b, np.float32).reshape(len(w)))
        return emit("conv", [x], (1, len(w)), name=name, **fields)

    def sl(x, a, b):
        k, c = S[x]
        if a == 0 and b == c:
            return x
        return emit("slice" if k else "vslice", [x], (k, b-a),
                    start=int(a), stop=int(b))

    def cat(*xs, name=None):
        k = S[xs[0]][0]
        assert all(S[x][0] == k for x in xs)
        return emit("concat" if k else "vconcat", xs,
                    (k, sum(S[x][1] for x in xs)), name=name)

    def bs(a, b):
        ka, ca = S[a]
        kb, cb = S[b]
        assert ca == cb or ca == 1 or cb == 1
        return max(ka, kb), max(ca, cb)

    def add(a, b):
        if scalar_is(a, 0):
            return b
        if scalar_is(b, 0):
            return a
        sh = bs(a, b)
        return emit("add" if sh[0] else "vadd", [a, b], sh)

    def sub(a, b):
        if scalar_is(b, 0):
            return a
        sh = bs(a, b)
        return emit("sub", [a, b], sh) if sh[0] else add(a, aff(b, -1))

    def mul(a, b):
        if scalar_is(a, 1):
            return b
        if scalar_is(b, 1):
            return a
        sh = bs(a, b)
        return emit("mul" if sh[0] else "vmul", [a, b], sh)

    def relu(x):
        return emit("relu" if S[x][0] else "vrelu", [x], S[x])

    def absolute(x):
        if S[x][0]:
            return emit("abs", [x], S[x])
        return add(relu(x), relu(aff(x, -1)))

    def maximum(a, b):
        if a == b:
            return a
        sh = bs(a, b)
        if sh[0]:
            return emit("max2", [a, b], sh)
        return add(relu(sub(a, b)), b)

    def mx(xs):
        xs = list(xs)
        while len(xs) > 1:
            xs = [
                maximum(xs[i], xs[i+1]) if i+1 < len(xs) else xs[i]
                for i in range(0, len(xs), 2)
            ]
        return xs[0]

    def clip(x):
        return relu(aff(relu(aff(x, -1, 1)), -1, 1))

    def inv(x):
        return aff(x, -1, 1)

    def le(a, b):
        return relu(aff(relu(aff(sub(a, b), 4, -1)), -1, 1))

    def eq(a, b, tol=.10, slope=16.):
        d = absolute(sub(a, b))
        if tol:
            d = relu(aff(d, 1, -tol))
        return relu(aff(d, -slope, 1))

    def pool(x, op="gmax"):
        return emit(op, [x], (0, S[x][1]))

    def total(x):
        return aff(pool(x, "gmean"), 4096)

    def channelsum(x):
        if S[x][1] == 1:
            return x
        return lin(x, np.ones((1, S[x][1]), np.float32))

    def rep(x, n):
        return lin(x, np.ones((n, 1), np.float32))

    def wt(x, mask):
        return total(mul(x, mask))

    def blend(g, yes, no):
        return add(mul(g, yes), mul(inv(g), no))

    def pick(x):
        # Competitive one-hot maximum with deterministic tie preference.
        n = S[x][1]
        w = np.zeros((n*n, n), np.float32)
        b = np.zeros(n*n, np.float32)
        summation = np.zeros((n, n*n), np.float32)
        for i in range(n):
            for j in range(n):
                k = i*n+j
                w[k, j] += 1
                w[k, i] -= 1
                b[k] = .125 * (i-j)
                summation[i, k] = 1
        defeats = relu(lin(x, w, b))
        excess = relu(lin(defeats, summation, -.01))
        return relu(aff(excess, -1000, 1))

    def state(shape):
        s = fresh("s")
        S[s] = shape
        return s

    def loop(body, states, nexts, count):
        lid = fresh("L")
        G.append({
            "id": lid, "op": "loop", "T": int(count),
            "state": states, "body": body, "next": nexts
        })
        out = {}
        for entry in states:
            s = entry["id"]
            out[s] = lid + "." + s
            S[out[s]] = S[s]
        return out

    zero = const([0])
    one = const([1])
    half = const([.5])
    colour_ids = const(np.arange(6, 15))

    yy = lin("coords", [[63, 0, 0]])
    xx = lin("coords", [[0, 63, 0]])
    xy = cat(yy, xx)
    fresh_episode = inv(sl("mem", 0, 1))
    previous_switch = sl("aprev", 5, 6)

    # Track the centre through an occluding, higher-layer object.
    visible_anchor = sl("x0", 0, 1)
    visible_anchor_present = pool(visible_anchor)
    movement = np.zeros((2, 8), np.float32)
    movement[0, 1] = -3
    movement[0, 2] = 3
    movement[1, 3] = -3
    movement[1, 4] = 3
    predicted_xy = add(sl("mem", 95, 97), lin("aprev", movement))
    moved = lin("aprev", [[0, 1, 1, 1, 1, 0, 0, 0]])
    fallback_gate = mul(
        mul(inv(visible_anchor_present), moved),
        sl("mem", 0, 1)
    )
    predicted_anchor = mul(
        eq(yy, sl(predicted_xy, 0, 1)),
        eq(xx, sl(predicted_xy, 1, 2))
    )
    anchor = add(visible_anchor, mul(predicted_anchor, fallback_gate))
    anchor_present = pool(anchor)
    anchor_xy = wt(xy, anchor)
    cy, cx = sl(anchor_xy, 0, 1), sl(anchor_xy, 1, 2)
    dy, dx = sub(yy, cy), sub(xx, cx)
    ay, ax = absolute(dy), absolute(dx)
    raw = sl("x0", 6, 15)
    raw_sum = channelsum(raw)
    four = sl("x0", 4, 5)

    # Small colour-4 rings mark target locations.
    tk = np.zeros((2, 1, 3, 3), np.float32)
    for r, c in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        tk[0, 0, r, c] = 1
    for r, c in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        tk[1, 0, r, c] = 1
    tests = clip(aff(conv(four, tk), 1, -2))
    targets = mul(maximum(sl(tests, 0, 1), sl(tests, 1, 2)), inv(four))

    # Paint-station interiors are solid 3x3 colour patches.
    pk = np.zeros((9, 9, 3, 3), np.float32)
    for c in range(9):
        pk[c, c] = 1
    paint_centres = clip(aff(conv(raw, pk), 1, -8))
    paint_full = clip(conv(
        paint_centres, np.ones((1, 9, 5, 5), np.float32)
    ))
    foreground = mul(mul(raw, inv(targets)), inv(paint_full))
    ak = pk.copy()
    for c in range(9):
        ak[c, c, 1, 1] = 0
    adjacent = wt(conv(foreground, ak), anchor)

    vertical = eq(dx, zero)
    horizontal = eq(dy, zero)
    rays = [
        mul(vertical, le(one, aff(dy, -1))),
        mul(vertical, le(one, dy)),
        mul(horizontal, le(one, aff(dx, -1))),
        mul(horizontal, le(one, dx)),
    ]
    nd, nv = [], []
    for ray, dist in zip(rays, [ay, ay, ax, ax]):
        score = pool(mul(foreground, mul(ray, aff(dist, -1, 64))))
        nd.append(aff(score, -1, 64))
        nv.append(clip(score))

    symmetry = add(
        absolute(sub(nd[0], nd[1])),
        absolute(sub(nd[2], nd[3]))
    )
    distance_sum = add(add(nd[0], nd[1]), add(nd[2], nd[3]))
    seen = add(add(nv[0], nv[1]), add(nv[2], nv[3]))
    init_colour = pick(add(
        add(aff(adjacent, 1000), aff(seen, 100)),
        add(aff(symmetry, -1), aff(distance_sum, -.03))
    ))

    def selected(v):
        return channelsum(mul(v, init_colour))

    def pair_radius(i, j):
        # A crossing object can introduce a nearer point on one side.
        # The farther of the two opposing outline tips is more robust.
        return maximum(
            selected(mul(nd[i], nv[i])),
            selected(mul(nd[j], nv[j]))
        )

    hollow_h = pair_radius(0, 1)
    hollow_w = pair_radius(2, 3)

    # Trace straight arms; simultaneously construct the movement lattice.
    rk = np.zeros((8, 8, 3, 3), np.float32)
    dirs = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    for i, (dr, dc) in enumerate(dirs):
        rk[i, i, 1-dr, 1-dc] = 1
    lk = np.zeros((1, 1, 7, 7), np.float32)
    for dr, dc in [(0, 0), (-3, 0), (3, 0), (0, -3), (0, 3)]:
        lk[0, 0, 3+dr, 3+dc] = 1

    allowed_ray = mul(clip(add(raw_sum, anchor)), inv(paint_full))
    ray_init = rep(anchor, 8)
    sr = state((1, 8))
    lattice_state = state((1, 1))
    outer = G
    body = []
    G = body
    ray_next = maximum(sr, mul(conv(sr, rk), allowed_ray))
    lattice_next = clip(conv(lattice_state, lk))
    G = outer
    traced = loop(
        body,
        [
            {"id": sr, "init": ray_init},
            {"id": lattice_state, "init": anchor}
        ],
        {sr: ray_next, lattice_state: lattice_next}, 42
    )

    lengths = pool(mul(
        traced[sr], cat(ay, ay, ax, ax, ax, ax, ax, ax)
    ))
    card_h = maximum(sl(lengths, 0, 1), sl(lengths, 1, 2))
    card_w = maximum(sl(lengths, 2, 3), sl(lengths, 3, 4))
    diagonal_r = mx(sl(lengths, i, i+1) for i in range(4, 8))
    plus = clip(maximum(card_h, card_w))
    cross = mul(inv(plus), clip(diagonal_r))
    hollow = inv(add(plus, cross))

    nk = np.zeros((2, 1, 3, 3), np.float32)
    nk[0, 0, 1, 0] = nk[0, 0, 1, 2] = 1
    nk[1, 0, 0, 1] = nk[1, 0, 2, 1] = 1
    tip_neighbours = conv(channelsum(mul(raw, init_colour)), nk)
    vertical_tips = mul(vertical, eq(ay, hollow_h))
    horizontal_tips = mul(horizontal, eq(ax, hollow_w))
    straight_tip = maximum(
        pool(mul(vertical_tips, sl(tip_neighbours, 0, 1))),
        pool(mul(horizontal_tips, sl(tip_neighbours, 1, 2)))
    )
    box_test = clip(aff(straight_tip, 1, -1))
    box = mul(hollow, box_test)
    diamond = mul(hollow, inv(box_test))
    inferred_w = add(
        add(mul(plus, card_w), mul(cross, diagonal_r)),
        mul(hollow, hollow_w)
    )
    inferred_h = add(
        add(mul(plus, card_h), mul(cross, diagonal_r)),
        mul(hollow, hollow_h)
    )
    inferred_geometry = cat(
        plus, cross, box, diamond, inferred_w, inferred_h
    )

    # First survey until a hollow outline is selected, or a full selection
    # cycle has established that no hollow outline is available.
    survey_start = blend(fresh_episode, anchor_xy, sl("mem", 93, 95))
    returned = mul(
        eq(cy, sl(survey_start, 0, 1)),
        eq(cx, sl(survey_start, 1, 2))
    )
    survey_done = clip(add(
        sl("mem", 92, 93),
        mul(anchor_present, add(hollow, mul(returned, previous_switch)))
    ))

    new_plan = clip(add(
        fresh_episode, add(previous_switch, sl("aprev", 0, 1))
    ))
    geometry = blend(new_plan, inferred_geometry, sl("mem", 12, 18))
    gp, gx, gb, gd, rw, rh = [
        sl(geometry, i, i+1) for i in range(6)
    ]
    phase0 = add(sl("mem", 19, 20), previous_switch)
    phase = sub(phase0, aff(le(const([12]), phase0), 12))
    exploratory = le(const([6]), phase)
    age = mul(inv(new_plan), aff(sl("mem", 18, 19), 1, 1))

    def hit(delta_y, delta_x, half_w, half_h, exclude_centre=False):
        abs_x, abs_y = absolute(delta_x), absolute(delta_y)
        bx = le(abs_x, add(rw, half_w))
        by = le(abs_y, add(rh, half_h))
        bounds = mul(bx, by)

        p = maximum(
            mul(le(abs_x, half_w), by),
            mul(le(abs_y, half_h), bx)
        )

        dt = add(half_w, half_h)
        d1 = le(absolute(sub(delta_x, delta_y)), dt)
        d2 = le(absolute(add(delta_x, delta_y)), dt)
        x = mul(bounds, maximum(d1, d2))

        dmin = add(
            relu(sub(abs_x, half_w)),
            relu(sub(abs_y, half_h))
        )
        dmax = add(add(abs_x, abs_y), dt)
        d = mul(le(dmin, rw), le(rw, dmax))

        ex = le(sub(rw, half_w), abs_x)
        ey = le(sub(rh, half_h), abs_y)
        b = mul(bounds, maximum(ex, ey))

        ans = add(
            add(mul(gp, p), mul(gx, x)),
            add(mul(gb, b), mul(gd, d))
        )
        if exclude_centre:
            ans = mul(ans, le(half, add(abs_x, abs_y)))
        return ans

    current_shape = hit(dy, dx, zero, zero, True)
    current_colour = pick(add(
        wt(raw, current_shape), aff(init_colour, .2)
    ))

    # Station boxes and shaping obstacles are used for safe routing.
    barrier_pixels = mul(sl("x0", 1, 2), le(yy, const([61])))
    station_points = cat(paint_centres, barrier_pixels)
    station_present = pool(station_points)
    min_y = aff(
        pool(mul(station_points, aff(yy, -1, 64))), -1, 64
    )
    min_x = aff(
        pool(mul(station_points, aff(xx, -1, 64))), -1, 64
    )
    max_y = pool(mul(station_points, yy))
    max_x = pool(mul(station_points, xx))
    padding = const([2] * 9 + [0])
    station_y = aff(add(min_y, max_y), .5)
    station_x = aff(add(min_x, max_x), .5)
    station_h = relu(add(aff(sub(max_y, min_y), .5), padding))
    station_w = relu(add(aff(sub(max_x, min_x), .5), padding))
    station_hits = mul(
        hit(
            sub(yy, station_y), sub(xx, station_x),
            station_w, station_h
        ),
        station_present
    )
    paint_hits = sl(station_hits, 0, 9)
    barrier_hit = sl(station_hits, 9, 10)
    paint_present = sl(station_present, 0, 9)

    near_kernel = np.ones((1, 1, 3, 3), np.float32)
    near_kernel[0, 0, 1, 1] = 0
    overlay = clip(conv(raw_sum, near_kernel))
    priority = lin("coords", [[-4032, -63, 0]], [4096])

    # Infer initially hidden target colours from nearby visible markers.
    unprocessed = state((1, 1))
    nearest_field = state((1, 9))
    outer = G
    body = []
    G = body
    high = pool(mul(unprocessed, priority))
    point = mul(unprocessed, eq(priority, high))
    pos = wt(xy, point)
    py, px = sl(pos, 0, 1), sl(pos, 1, 2)
    pc = wt(raw, point)
    uncovered = inv(wt(overlay, point))
    proximity = relu(aff(
        add(absolute(sub(yy, py)), absolute(sub(xx, px))),
        -1, 256
    ))
    proposal = mul(proximity, mul(pc, uncovered))
    nf_next = maximum(nearest_field, proposal)
    up_next = relu(sub(unprocessed, point))
    G = outer
    nearest_result = loop(
        body,
        [
            {"id": unprocessed, "init": targets},
            {"id": nearest_field, "zeros": [9, 64, 64]}
        ],
        {unprocessed: up_next, nearest_field: nf_next}, 24
    )

    # Store 24 persistent (row, column, colour) triples. Marker identity
    # therefore survives complete visual occlusion by several outlines.
    shift = np.zeros((72, 72), np.float32)
    for i in range(69):
        shift[i, i+3] = 1
    append = np.zeros((72, 3), np.float32)
    append[-3:, :] = np.eye(3, dtype=np.float32)

    remaining = state((1, 1))
    accumulator = state((1, 9))
    old_bank = state((0, 72))
    new_bank = state((0, 72))
    bank_init = sl("mem", 20, 92)

    outer = G
    body = []
    G = body

    best_priority = pool(mul(remaining, priority))
    detected_point = mul(remaining, eq(priority, best_priority))
    detected_xy = wt(xy, detected_point)
    detected_present = pool(detected_point)

    remembered_xy = sl(old_bank, 0, 2)
    remembered_id = sl(old_bank, 2, 3)
    point_xy = blend(fresh_episode, detected_xy, remembered_xy)
    point_present = blend(
        fresh_episode, detected_present, clip(remembered_id)
    )
    point_y = sl(point_xy, 0, 1)
    point_x = sl(point_xy, 1, 2)
    point = mul(
        mul(eq(yy, point_y), eq(xx, point_x)),
        point_present
    )

    point_raw = add(
        wt(raw, point), mul(wt(anchor, point), current_colour)
    )
    point_overlay = wt(overlay, point)
    inferred_point = pick(wt(nearest_result[nearest_field], point))
    initial_desired = blend(point_overlay, inferred_point, point_raw)
    remembered_colour = eq(remembered_id, colour_ids)
    desired = mul(
        blend(fresh_episode, initial_desired, remembered_colour),
        point_present
    )

    desired_id = channelsum(mul(desired, colour_ids))
    stored_triple = cat(point_xy, desired_id)
    next_old_bank = lin(old_bank, shift)
    next_new_bank = add(
        lin(new_bank, shift), lin(stored_triple, append)
    )

    correct = clip(channelsum(mul(desired, point_raw)))
    satisfied = mul(point_overlay, correct)
    own_coverage = wt(current_shape, point)
    retention = aff(exploratory, -600, 800)

    # Peripheral markers tend to have fewer compatible placements.
    # This small bonus breaks equal-coverage assignments, not cardinality.
    edge_bonus = add(
        absolute(aff(point_y, 1, -31.5)),
        absolute(aff(point_x, 1, -31.5))
    )
    reward = add(
        mul(inv(satisfied), aff(edge_bonus, 1, 1000)),
        mul(mul(satisfied, own_coverage), add(retention, edge_bonus))
    )
    known = channelsum(desired)
    signed_reward = sub(
        mul(desired, aff(reward, 1, 1200)),
        aff(known, 1200)
    )
    placement_hit = hit(
        sub(yy, point_y), sub(xx, point_x), zero, zero, True
    )
    next_acc = add(accumulator, mul(placement_hit, signed_reward))
    next_remaining = relu(sub(remaining, detected_point))
    G = outer
    matched = loop(
        body,
        [
            {"id": remaining, "init": targets},
            {"id": accumulator, "zeros": [9, 64, 64]},
            {"id": old_bank, "init": bank_init},
            {"id": new_bank, "zeros": [72]}
        ],
        {
            remaining: next_remaining,
            accumulator: next_acc,
            old_bank: next_old_bank,
            new_bank: next_new_bank
        },
        24
    )

    # Score all reachable centres and desired colours.
    total_paint = channelsum(paint_hits)
    wrong_goal_paint = clip(add(
        sub(total_paint, paint_hits), barrier_hit
    ))
    permitted_colours = clip(add(current_colour, paint_present))
    placement_valid = mul(
        mul(traced[lattice_state], inv(wrong_goal_paint)),
        permitted_colours
    )
    travel_cost = add(ax, ay)
    utility = add(
        sub(matched[accumulator], travel_cost),
        aff(current_colour, 60)
    )
    utility = add(utility, aff(inv(placement_valid), -1000000))
    proposed_colour = pick(pool(utility))
    chosen_utility = channelsum(mul(utility, proposed_colour))
    best_utility = pool(chosen_utility)
    best_cells = eq(chosen_utility, best_utility, tol=.02, slope=100)
    proposed_y = pool(mul(best_cells, yy))
    best_row = mul(best_cells, eq(yy, proposed_y))
    proposed_x = pool(mul(best_row, xx))
    proposed_xy = cat(proposed_y, proposed_x)

    destination_xy = blend(
        new_plan, proposed_xy, sl("mem", 1, 3)
    )
    destination_colour = blend(
        new_plan, proposed_colour, sl("mem", 3, 12)
    )
    dest_y = sl(destination_xy, 0, 1)
    dest_x = sl(destination_xy, 1, 2)

    agree = channelsum(mul(current_colour, destination_colour))
    need_paint = clip(inv(agree))
    destination_seed = mul(eq(yy, dest_y), eq(xx, dest_x))
    paint_seed = clip(channelsum(mul(paint_hits, destination_colour)))
    seed = blend(need_paint, paint_seed, destination_seed)

    safe_colours = clip(add(current_colour, destination_colour))
    forbidden_paint = channelsum(mul(paint_hits, inv(safe_colours)))
    navigable = inv(clip(add(forbidden_paint, barrier_hit)))
    seed_value = aff(mul(seed, navigable), 100)

    move_kernel = np.zeros((4, 1, 7, 7), np.float32)
    for i, (dr, dc) in enumerate([(-3, 0), (3, 0), (0, -3), (0, 3)]):
        move_kernel[i, 0, 3+dr, 3+dc] = 1

    value_state = state((1, 1))
    outer = G
    body = []
    G = body
    neighbours = conv(value_state, move_kernel)
    best_neighbour = mx(sl(neighbours, i, i+1) for i in range(4))
    propagated = relu(aff(best_neighbour, 1, -1))
    value_next = mul(maximum(seed_value, propagated), navigable)
    G = outer
    nav = loop(
        body,
        [{"id": value_state, "init": seed_value}],
        {value_state: value_next}, 64
    )

    value = nav[value_state]
    movement_values = wt(conv(value, move_kernel), anchor)
    centre_value = wt(value, anchor)
    switch_value = sub(
        aff(centre_value, 1, .05), aff(need_paint, 200)
    )
    timeout = clip(aff(age, 1, -65))
    unreachable = inv(clip(centre_value))
    force_switch = clip(add(
        add(add(inv(anchor_present), timeout), unreachable),
        inv(survey_done)
    ))
    action_features = cat(movement_values, switch_value, force_switch)

    ow = np.zeros((8, 6), np.float32)
    for i in range(4):
        ow[i+1, i] = 1
    ow[5, 4] = 1
    ow[5, 5] = 1000
    ob = np.full(8, -10000, np.float32)
    ob[1:6] = 0
    lin(action_features, ow, ob, name="logits")
    conv(anchor, np.zeros((1, 1, 1, 1), np.float32), name="click")
    cat(
        one, destination_xy, destination_colour, geometry,
        age, phase, matched[new_bank], survey_done,
        survey_start, anchor_xy, name="mem_next"
    )

    # Algebraically combine consecutive affine maps.
    compiled = {}

    def fuse(seq):
        for nd in seq:
            if nd["op"] == "loop":
                fuse(nd["body"])
                compiled[nd["id"]] = nd
                continue
            if nd["op"] in ("conv", "dense"):
                while True:
                    parent = compiled.get(nd["in"][0])
                    if parent is None or parent["op"] != nd["op"]:
                        break
                    a = P[nd["w"]]
                    b = P[parent["w"]]
                    spatial = nd["op"] == "conv"
                    if spatial:
                        if a.shape[2:] != (1, 1) or b.shape[2:] != (1, 1):
                            break
                        aa, bb = a[:, :, 0, 0], b[:, :, 0, 0]
                    else:
                        aa, bb = a, b
                    bias_a = P[nd["b"]] if "b" in nd else np.zeros(
                        aa.shape[0], np.float32
                    )
                    bias_b = P[parent["b"]] if "b" in parent else np.zeros(
                        bb.shape[0], np.float32
                    )
                    new_w = aa @ bb
                    new_b = aa @ bias_b + bias_a
                    if spatial:
                        new_w = new_w[:, :, None, None]
                    nd["w"] = param(new_w)
                    nd["b"] = param(new_b)
                    nd["in"] = list(parent["in"])
            compiled[nd["id"]] = nd

    fuse(root)

    # Prune unused calculations and parameters after affine composition.
    nodes = {}
    owners = {}
    state_owner = {}

    def index_nodes(seq, owner=None):
        for nd in seq:
            nodes[nd["id"]] = nd
            owners[nd["id"]] = owner
            if nd["op"] == "loop":
                for st in nd["state"]:
                    state_owner[st["id"]] = nd["id"]
                    state_owner[nd["id"] + "." + st["id"]] = nd["id"]
                index_nodes(nd["body"], nd["id"])

    index_nodes(root)
    live = set()
    pending = ["logits", "click", "mem_next"]

    def require_loop(lid):
        if lid in live:
            return
        live.add(lid)
        nd = nodes[lid]
        for st in nd["state"]:
            if "init" in st:
                pending.append(st["init"])
            pending.append(nd["next"][st["id"]])

    while pending:
        nid = pending.pop()
        if nid in state_owner:
            require_loop(state_owner[nid])
            continue
        if nid in live or nid not in nodes:
            continue
        nd = nodes[nid]
        if nd["op"] == "loop":
            require_loop(nid)
            continue
        live.add(nid)
        if owners[nid] is not None:
            require_loop(owners[nid])
        pending.extend(nd.get("in", []))

    def retained(seq):
        result = []
        for nd in seq:
            if nd["id"] not in live:
                continue
            if nd["op"] == "loop":
                nd = dict(nd)
                nd["body"] = retained(nd["body"])
            result.append(nd)
        return result

    graph = retained(root)
    used_params = set()

    def collect(seq):
        for nd in seq:
            for key in ("w", "b"):
                if key in nd:
                    used_params.add(nd[key])
            if nd["op"] == "loop":
                collect(nd["body"])

    collect(graph)
    params = {k: v for k, v in P.items() if k in used_params}
    return {"graph": graph, "memory_size": 97, "params": params}
