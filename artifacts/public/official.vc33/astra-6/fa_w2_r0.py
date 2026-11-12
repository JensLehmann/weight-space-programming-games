import numpy as np


def build():
    graph = []
    params = {}
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "avail": ("v", 8),
        "aprev": ("v", 8),
    }
    counter = [0]
    current = [graph]
    cache = {}

    def fresh(prefix="n"):
        counter[0] += 1
        return prefix + str(counter[0])

    def parameter(a):
        name = fresh("p")
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, ins, name=None, **kw):
        name = name or fresh()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(kw)
        current[0].append(nd)
        if op == "conv":
            shapes[name] = ("s", params[kw["w"]].shape[0])
        elif op == "dense":
            shapes[name] = ("v", params[kw["w"]].shape[0])
        elif op in ("slice", "vslice"):
            shapes[name] = (shapes[ins[0]][0], kw["stop"] - kw["start"])
        elif op in ("concat", "vconcat"):
            shapes[name] = (
                shapes[ins[0]][0],
                sum(shapes[z][1] for z in ins),
            )
        elif op in ("gmax", "gmean"):
            shapes[name] = ("v", shapes[ins[0]][1])
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shapes[name] = ("s", max(shapes[z][1] for z in ins))
        elif op in ("vadd", "vmul"):
            shapes[name] = ("v", max(shapes[z][1] for z in ins))
        else:
            shapes[name] = shapes[ins[0]]
        return name

    def conv(x, w, b=None, name=None):
        wn = w if isinstance(w, str) else parameter(w)
        kw = {"w": wn}
        if b is not None:
            kw["b"] = parameter(b)
        return node("conv", [x], name=name, **kw)

    def dense(x, w, b=None, name=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("dense", [x], name=name, **kw)

    def sl(x, a, b=None):
        b = a + 1 if b is None else b
        op = "slice" if shapes[x][0] == "s" else "vslice"
        return node(op, [x], start=a, stop=b)

    def cat(*xs):
        op = "concat" if shapes[xs[0]][0] == "s" else "vconcat"
        return node(op, xs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def mn(a, b):
        return node("min2", [a, b])

    def gm(x):
        return node("gmax", [x])

    def affine(x, a=1.0, b=0.0):
        c = shapes[x][1]
        if shapes[x][0] == "v":
            return dense(x, np.eye(c) * a, np.full(c, b))
        return conv(
            x, (np.eye(c) * a)[:, :, None, None], np.full(c, b)
        )

    def channel_sum(x):
        c = shapes[x][1]
        return conv(x, np.ones((1, c, 1, 1), np.float32))

    one = sl("coords", 2)
    row = affine(sl("coords", 0), 63.0)
    col = affine(sl("coords", 1), 63.0)

    def clip(x):
        return mn(node("relu", [x]), one)

    def inv(x):
        return affine(x, -1.0, 1.0)

    def gt(x):
        # Exact Boolean on integer-valued quantities, with a rounding margin.
        return clip(affine(x, 2.0, -1.0))

    def ge(x):
        return clip(affine(x, 2.0, 1.0))

    def eq(x):
        return clip(affine(node("abs", [x]), -2.0, 1.0))

    def exact_max_mask(x, eligible):
        d = node("abs", [sub(x, gm(x))])
        return mul(clip(affine(d, -20000.0, 1.0)), eligible)

    def product(*xs):
        out = xs[0]
        for x in xs[1:]:
            out = mul(out, x)
        return out

    ui = clip(inv(row))
    domain = inv(ui)
    yellow = product(sl("x0", 4), domain)
    buttons = sl("x0", 9)
    stripes = cat(sl("x0", 11), sl("x0", 14))
    gate = add(sl("x0", 1), sl("x0", 12))
    grey = sl("x0", 5)

    # Direction order: down, up, right, left.
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    yw = np.zeros((4, 1, 7, 7), np.float32)
    for d, (dy, dx) in enumerate(dirs):
        for k in (1, 2, 3):
            yw[d, 0, 3 - k * dy, 3 - k * dx] = 1
    yellow_behind = clip(conv(yellow, yw))
    orientation = gm(mul(yellow_behind, channel_sum(stripes)))

    vertical = dense(orientation, [[1, 1, 0, 0]])
    horizontal = dense(orientation, [[0, 0, 1, 1]])
    sy = dense(orientation, [[1, -1, 0, 0]])
    sx = dense(orientation, [[0, 0, 1, -1]])
    q = add(mul(row, sy), mul(col, sx))
    hcoord = affine(q, -1.0, 128.0)
    target_h_grid = affine(q, -1.0, 127.0)

    pawn = mul(stripes, clip(channel_sum(yellow_behind)))
    goals = sub(stripes, pawn)
    goals_any = channel_sum(goals)

    # Buttons are either 3x3 or 4x4 in display coordinates.
    sw = np.zeros((1, 1, 5, 5), np.float32)
    for k in (-1, 0, 1, 2):
        sw[0, 0, 2 + k, 2] = 1
    small = gm(clip(affine(conv(buttons, sw), 2.0, -7.0)))

    def shift_bank(x):
        c = shapes[x][1]
        key = ("shift_bank", c)
        if key not in cache:
            w = np.zeros((4 * c, c, 3, 3), np.float32)
            for d, (dy, dx) in enumerate(dirs):
                for j in range(c):
                    w[d * c + j, j, 1 + dy, 1 + dx] = 1
            cache[key] = parameter(w)
        z = conv(x, cache[key])
        return [sl(z, d * c, (d + 1) * c) for d in range(4)]

    def forward(x):
        ss = shift_bank(x)
        out = mul(ss[0], sl(orientation, 0))
        for d in range(1, 4):
            out = add(out, mul(ss[d], sl(orientation, d)))
        return out

    def edge(x):
        return mul(x, inv(forward(x)))

    def along_pool(x):
        a = node("colmax", [x])
        b = node("rowmax", [x])
        return add(mul(a, vertical), mul(b, horizontal))

    def across_pool(x):
        a = node("rowmax", [x])
        b = node("colmax", [x])
        return add(mul(a, vertical), mul(b, horizontal))

    def perpendicular_expand(x):
        # Binary dilation by four pixels, only across gravity.
        c = shapes[x][1]
        z = x
        for radius in (3, 1):
            k = 2 * radius + 1
            key = ("perp", c, radius)
            if key not in cache:
                w = np.zeros((2 * c, c, k, k), np.float32)
                for j in range(c):
                    w[j, j, radius, :] = 1
                    w[c + j, j, :, radius] = 1
                cache[key] = parameter(w)
            t = conv(z, cache[key])
            z = clip(add(
                mul(sl(t, 0, c), vertical),
                mul(sl(t, c, 2 * c), horizontal),
            ))
        return z

    def perpendicular_neighbor_max(x):
        ss = shift_bank(x)
        return add(
            mul(mx(ss[2], ss[3]), vertical),
            mul(mx(ss[0], ss[1]), horizontal),
        )

    def unit_shift(x, d):
        # A displacement of 3 + small pixels, without resampling the image.
        c = shapes[x][1]
        dy, dx = dirs[d]
        key3 = ("unit3", c, d)
        key1 = ("unit1", c, d)
        if key3 not in cache:
            w = np.zeros((c, c, 7, 7), np.float32)
            for j in range(c):
                w[j, j, 3 + 3 * dy, 3 + 3 * dx] = 1
            cache[key3] = parameter(w)
        if key1 not in cache:
            w = np.zeros((c, c, 3, 3), np.float32)
            for j in range(c):
                w[j, j, 1 + dy, 1 + dx] = 1
            cache[key1] = parameter(w)
        a = conv(x, cache[key3])
        b = conv(a, cache[key1])
        return add(a, mul(sub(b, a), small))

    def diffuse(seed, mask, T, cost=None, binary=False):
        c = shapes[seed][1]
        lid = fresh("loop")
        sid = fresh("state")
        shapes[sid] = shapes[seed]
        body = []
        old_graph = current[0]
        current[0] = body

        if binary:
            key = ("binary_flood", c)
            if key not in cache:
                w = np.zeros((c, c, 3, 3), np.float32)
                for j in range(c):
                    for dy, dx in dirs + [(0, 0)]:
                        w[j, j, 1 + dy, 1 + dx] = 1
                cache[key] = parameter(w)
            z = conv(sid, cache[key])
            z = mul(z, mask)
            z = mn(z, one)
        else:
            ss = shift_bank(sid)
            z = mx(mx(ss[0], ss[1]), mx(ss[2], ss[3]))
            if cost is None:
                z = mx(z, sid)
            else:
                z = mx(sub(z, cost), seed)
            z = mul(z, mask)

        current[0] = old_graph
        current[0].append({
            "id": lid,
            "op": "loop",
            "T": T,
            "state": [{"id": sid, "init": seed}],
            "body": body,
            "next": {sid: z},
        })
        out = lid + "." + sid
        shapes[out] = shapes[seed]
        return out

    # Longitudinal walls partition the board into platform lanes.
    actual_wall = clip(add(add(grey, gate), goals_any))
    lw = np.zeros((2, 1, 7, 7), np.float32)
    lw[0, 0, :, 3] = 1
    lw[1, 0, 3, :] = 1
    long_lines = clip(affine(conv(actual_wall, lw), 2.0, -11.0))
    vwall = node("colmax", [sl(long_lines, 0)])
    hwall = node("rowmax", [sl(long_lines, 1)])
    barrier = clip(add(mul(vwall, vertical), mul(hwall, horizontal)))
    free = inv(mx(mx(barrier, grey), ui))

    wall_h = along_pool(product(hcoord, actual_wall, barrier))
    wall_cost = mul(affine(wall_h, -1.0, 256.0), barrier)
    cap_seed = mul(perpendicular_neighbor_max(wall_cost), free)

    # Suppress the thin black decorations beside activated gates.
    bw = np.zeros((2, 1, 3, 3), np.float32)
    bw[0, 0, 1, :] = 1
    bw[1, 0, :, 1] = 1
    bb = conv(sl("x0", 0), bw)
    black_run = add(
        mul(sl(bb, 0), vertical),
        mul(sl(bb, 1), horizontal),
    )
    clean_black = product(ge(affine(black_run, 1.0, -3.0)), free)

    seed_a = cat(
        mul(hcoord, clean_black),
        mul(affine(q, 1.0, 128.0), free),
        mul(hcoord, free),
    )
    seed_b = cat(cap_seed, mul(pawn, free))
    geom_a = diffuse(seed_a, free, 128)
    geom_b = diffuse(seed_b, free, 128)

    empty_h = affine(sl(geom_a, 1), -1.0, 255.0)
    current_h = mx(sl(geom_a, 0), empty_h)
    floor_cap = sl(geom_a, 2)
    wall_cap = affine(sl(geom_b, 0), -1.0, 256.0)
    occupied = sl(geom_b, 1, 3)
    occupied_any = clip(channel_sum(occupied))
    action_cap = mn(
        wall_cap, sub(floor_cap, affine(occupied_any, 6.0))
    )
    future_cap = mn(wall_cap, affine(floor_cap, 1.0, -6.0))

    # Buttons connect hydraulic regions, whereas gates connect pawn routes.
    bridge = mul(barrier, perpendicular_expand(buttons))
    hydraulic_free = mx(free, bridge)

    # Each rectangular region contributes its length once, at a corner.
    fs = shift_bank(free)
    positive_perp = add(mul(fs[2], vertical), mul(fs[0], horizontal))
    corner = product(free, inv(forward(free)), inv(positive_perp))
    volume = sub(current_h, empty_h)
    tie = affine(add(affine(row, 64.0), col), 0.00013)

    unused = corner
    component_seeds = []
    component_volumes = []
    for _ in range(8):
        ranking = mul(add(affine(volume, 1.0, 1.0), tie), unused)
        pick = exact_max_mask(ranking, unused)
        component_seeds.append(pick)
        component_volumes.append(gm(mul(volume, pick)))
        unused = mul(unused, inv(pick))

    reach = diffuse(cat(*component_seeds), hydraulic_free, 192, binary=True)
    volumes_vector = cat(*component_volumes)
    total_volume = channel_sum(mul(reach, volumes_vector))
    component_code = conv(
        reach,
        (2.0 ** np.arange(8, dtype=np.float32)).reshape(1, 8, 1, 1),
    )

    # Goals are seeded only on sides with sufficient volume and headroom.
    goal_edge = edge(goals)
    pawn_edge = edge(pawn)
    required_length = sub(target_h_grid, empty_h)
    goal_feasible = product(
        ge(required_length),
        ge(sub(total_volume, required_length)),
        ge(sub(future_cap, target_h_grid)),
        free,
    )
    goal_seed = mul(perpendicular_expand(goal_edge), goal_feasible)
    goal_here = product(
        perpendicular_expand(mul(goal_seed, occupied)),
        goal_edge,
    )

    solved = gm(mul(goal_here, across_pool(pawn_edge)))
    presence = gm(pawn)
    active = node("vmul", [presence, affine(solved, -1.0, 1.0)])
    active_0 = sl(active, 0)
    active_1 = node("vmul", [
        sl(active, 1), affine(active_0, -1.0, 1.0)
    ])
    active = cat(active_0, active_1)
    selected_occupancy = channel_sum(mul(occupied, active))

    # Evaluate both sides of each gate, including their joint volume demand.
    gate_fields = cat(empty_h, future_cap, total_volume, component_code, free)
    gf = [unit_shift(gate_fields, d) for d in range(4)]
    left_fields = add(mul(gf[3], vertical), mul(gf[1], horizontal))
    right_fields = add(mul(gf[2], vertical), mul(gf[0], horizontal))

    el, er = sl(left_fields, 0), sl(right_fields, 0)
    cl, cr = sl(left_fields, 1), sl(right_fields, 1)
    vl, vr = sl(left_fields, 2), sl(right_fields, 2)
    gl, gr = sl(left_fields, 3), sl(right_fields, 3)
    fl, fr = sl(left_fields, 4), sl(right_fields, 4)
    need_l, need_r = sub(target_h_grid, el), sub(target_h_grid, er)
    same_network = eq(sub(gl, gr))
    joint_ok = ge(sub(sub(vl, need_l), need_r))
    gate_feasible = product(
        fl, fr,
        ge(need_l), ge(need_r),
        ge(sub(cl, target_h_grid)), ge(sub(cr, target_h_grid)),
        ge(sub(vl, need_l)), ge(sub(vr, need_r)),
        add(inv(same_network), mul(same_network, joint_ok)),
    )

    route = diffuse(
        affine(goal_seed, 1024.0),
        mx(free, gate),
        224,
        cost=affine(gate, 8.0, 0.02),
    )
    gate_edge = edge(gate)
    valid_gate = product(
        gate_edge,
        gate_feasible,
        perpendicular_expand(occupied),
        gt(route),
    )

    candidate = add(
        mul(valid_gate, affine(route, 1.0, 1000.0)),
        affine(goal_here, 4000.0),
    )
    candidate = channel_sum(mul(candidate, active))
    candidate_valid = gt(channel_sum(mul(add(valid_gate, goal_here), active)))
    candidate_tie = affine(add(affine(row, 64.0), col), 0.00131)
    candidate = add(candidate, mul(candidate_valid, candidate_tie))
    selected = exact_max_mask(candidate, candidate_valid)

    selected_gate = gm(mul(selected, gate))
    desired_h = gm(mul(selected, target_h_grid))
    target_seed = product(
        perpendicular_expand(selected),
        free,
        clip(add(selected_occupancy, selected_gate)),
    )
    target_regions = diffuse(target_seed, free, 128, binary=True)

    deficit = mul(gt(sub(desired_h, current_h)), target_regions)
    surplus = mul(gt(sub(current_h, desired_h)), target_regions)
    minimum_h = add(
        empty_h, mul(target_regions, sub(desired_h, empty_h))
    )
    can_donate = gt(sub(current_h, minimum_h))
    has_room = gt(sub(action_cap, current_h))
    slack_sink = mul(inv(target_regions), has_room)

    flow = diffuse(
        affine(cat(deficit, surplus, slack_sink), 1024.0),
        hydraulic_free,
        192,
        cost=affine(bridge, 8.0, 0.02),
    )
    pull, push, sink = sl(flow, 0), sl(flow, 1), sl(flow, 2)
    grow_mode = gm(deficit)
    shrink_mode = affine(grow_mode, -1.0, 1.0)

    # Compare a button with its paired button across the intervening wall.
    button_fields = cat(buttons, can_donate, pull, push, sink, barrier)
    directional_scores = []
    for d in range(4):
        halfway = unit_shift(button_fields, d)
        opposite = unit_shift(halfway, d)
        other_button = sl(opposite, 0)
        other_donor = sl(opposite, 1)
        other_pull = sl(opposite, 2)
        other_push = sl(opposite, 3)
        other_sink = sl(opposite, 4)
        middle_wall = sl(halfway, 5)

        pull_gradient = gt(affine(sub(pull, other_pull), 1.0, -4.0))
        push_gradient = product(
            gt(affine(sub(other_push, push), 1.0, -4.0)),
            gt(affine(sub(sink, other_sink), 1.0, -4.0)),
        )
        allowed_axis = horizontal if d < 2 else vertical
        legal = product(
            buttons, has_room, other_button, other_donor,
            middle_wall, allowed_axis,
        )
        grow_score = product(
            pull_gradient, grow_mode, affine(pull, 1.0, 2048.0)
        )
        shrink_score = product(
            push_gradient, shrink_mode, affine(other_push, 1.0, 2048.0)
        )
        directional_scores.append(mul(legal, add(grow_score, shrink_score)))

    button_score = directional_scores[0]
    for z in directional_scores[1:]:
        button_score = mx(button_score, z)

    ready = node("vmul", [
        selected_gate, affine(gm(deficit), -1.0, 1.0)
    ])
    ready = node("vmul", [
        ready, affine(gm(surplus), -1.0, 1.0)
    ])
    gate_score = affine(mul(selected, ready), 10000.0)
    fallback = add(
        affine(goals_any, 0.0001),
        affine(buttons, 0.00001),
    )
    node("add", [mx(button_score, gate_score), fallback], name="click")

    logit_bias = np.zeros(8, np.float32)
    logit_bias[6] = 10.0
    dense(
        "avail", np.zeros((8, 8), np.float32),
        logit_bias, name="logits",
    )

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }
