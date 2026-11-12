import numpy as np


def build():
    graph = []
    params = {}
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "aprev": (8,),
        "avail": (8,),
        "mem": (1,),
    }
    costs = {}
    cache = {}
    serial = [0]

    def uid():
        serial[0] += 1
        return "n%d" % serial[0]

    def parameter(key, array):
        if key in cache:
            return cache[key]
        name = "p%d" % len(params)
        params[name] = np.asarray(array, dtype=np.float32)
        cache[key] = name
        return name

    def emit(op, ins, shape, name=None, **fields):
        name = name or uid()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        graph.append(nd)
        shapes[name] = tuple(shape)
        if op in ("conv", "dense"):
            cost = int(np.prod(params[fields["w"]].shape))
            if op == "conv":
                cost *= 4096
        elif op in (
            "add", "sub", "mul", "max2", "min2", "vadd", "vmul"
        ):
            cost = int(np.prod(shape))
        elif op in (
            "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"
        ):
            cost = int(np.prod(shapes[ins[0]]))
        else:
            cost = 0
        costs[name] = cost
        return name

    def conv(x, w, b=None, key=None, name=None):
        w = np.asarray(w, np.float32)
        key = key if key is not None else ("unique", uid())
        fields = {"w": parameter(("w", key), w)}
        if b is not None:
            fields["b"] = parameter(("b", key), b)
        return emit(
            "conv", [x], (w.shape[0], 64, 64), name=name, **fields
        )

    def dense(x, w, b=None, key=None, name=None):
        w = np.asarray(w, np.float32)
        key = key if key is not None else ("unique_dense", uid())
        fields = {"w": parameter(("dw", key), w)}
        if b is not None:
            fields["b"] = parameter(("db", key), b)
        return emit("dense", [x], (w.shape[0],), name=name, **fields)

    def unary(op, x, name=None):
        return emit(op, [x], shapes[x], name=name)

    def binary(op, x, y, name=None):
        sx, sy = shapes[x], shapes[y]
        c = max(sx[0], sy[0])
        shape = (c, 64, 64) if len(sx) == 3 or len(sy) == 3 else (c,)
        return emit(op, [x, y], shape, name=name)

    def cat(xs, vector=False):
        c = sum(shapes[x][0] for x in xs)
        return emit(
            "vconcat" if vector else "concat",
            xs, (c,) if vector else (c, 64, 64),
        )

    def channel(x, i):
        return emit(
            "slice", [x], (1, 64, 64), start=i, stop=i + 1
        )

    def pool(x):
        return emit("gmax", [x], (shapes[x][0],))

    def scale(x, a=1.0, b=0.0, name=None):
        c = shapes[x][0]
        w = np.eye(c, dtype=np.float32) * np.float32(a)
        bias = np.full(c, b, np.float32) if b != 0 else None
        key = ("scale", len(shapes[x]), c, float(a), float(b))
        if len(shapes[x]) == 3:
            return conv(
                x, w[:, :, None, None], bias, key=key, name=name
            )
        return dense(x, w, bias, key=key, name=name)

    one = channel("coords", 2)
    zero = scale(one, 0.0)
    constants = {0.0: zero, 1.0: one}

    def constant(v):
        v = float(v)
        if v not in constants:
            constants[v] = scale(one, 0.0, v)
        return constants[v]

    def add(x, y):
        return binary("add", x, y)

    def sub(x, y):
        return binary("sub", x, y)

    def mul(x, y):
        return binary("mul", x, y)

    def maximum(x, y):
        return binary("max2", x, y)

    def relu(x):
        return unary("relu", x)

    def clip(x):
        return binary("min2", relu(x), one)

    def total(xs):
        result = xs[0]
        for x in xs[1:]:
            result = add(result, x)
        return result

    def duplicate(x):
        return cat([x, x])

    def shift6(x, dy=0, dx=0):
        c = shapes[x][0]
        w = np.zeros((c, c, 7, 7), np.float32)
        for i in range(c):
            yy = (-3 if i == 0 else 3) if dy == "gravity" else 3 * dy
            w[i, i, 3 + yy, 3 + 3 * dx] = 1.0
        key = ("shift", c, dy, dx)
        return conv(conv(x, w, key=key), w, key=key)

    def gravity_shift(x):
        return shift6(x, dy="gravity")

    def choose(q1, l1, q2, l2, score1=None, score2=None):
        s1 = q1 if score1 is None else score1
        s2 = q2 if score2 is None else score2
        difference = scale(sub(s2, s1), 1.0, -0.125)
        gate = clip(sub(one, scale(relu(difference), 8.0)))
        label = add(l2, mul(gate, sub(l1, l2)))
        return maximum(q1, q2), label

    def invalidate(q, mask):
        return sub(mul(q, mask), sub(one, mask))

    def begin_body():
        nonlocal graph
        old = graph
        graph = []
        return old

    def end_loop(old, states, next_map, iterations):
        nonlocal graph
        body = graph
        graph = old
        lid = uid()
        graph.append({
            "id": lid,
            "op": "loop",
            "T": int(iterations),
            "state": states,
            "body": body,
            "next": next_map,
        })
        for state in states:
            shapes[lid + "." + state["id"]] = shapes[state["id"]]
        return lid

    # Retain the last horizontal direction across intervening clicks.
    mw = np.zeros((2, 8), np.float32)
    mw[0, 3] = -1.0
    mw[0, 4] = 1.0
    mw[1, 3] = mw[1, 4] = 1.0
    previous_features = dense("aprev", mw, key="previous_direction")
    previous_sign = emit(
        "vslice", [previous_features], (1,), start=0, stop=1
    )
    previous_horizontal = emit(
        "vslice", [previous_features], (1,), start=1, stop=2
    )
    keep_memory = scale(previous_horizontal, -1.0, 1.0)
    retained_memory = binary("vmul", keep_memory, "mem")
    mem_next = binary(
        "vadd", previous_sign, retained_memory, name="mem_next"
    )
    prefer_left = clip(scale(mul(one, mem_next), -1.0))
    prefer_right = sub(one, prefer_left)
    left_tie_bonus = scale(prefer_left, 0.25)
    right_tie_bonus = scale(prefer_right, 0.25)

    # Player templates are centred at local pixel (3, 3).
    player_templates = [
        [
            "  ... ",
            " ..x..",
            " .xxr.",
            " .xxr.",
            " ..x..",
            "  ... ",
            "       ",
        ],
        [
            "  ... ",
            " ..x..",
            " .rxx.",
            " .rxx.",
            " ..x..",
            "  ... ",
            "       ",
        ],
    ]
    palette = {".": 5, "x": 9, "r": 11}
    pw = np.zeros((2, 16, 7, 7), np.float32)
    pb = np.zeros(2, np.float32)
    for o, template in enumerate(player_templates):
        count = 0
        for r, row in enumerate(template):
            for c, ch in enumerate(row):
                if ch in palette:
                    pw[o, palette[ch], r, c] = 1.0
                    count += 1
        pb[o] = 1.0 - count
    matches = relu(conv("x0", pw, pb, key="player_templates"))
    player = clip(conv(
        matches, np.ones((1, 2, 1, 1), np.float32),
        key="player_union",
    ))

    # Recover the vertical six-pixel lattice phase from the player.
    row_seed = emit("rowmax", [player], (1, 64, 64))
    rs = uid()
    shapes[rs] = (1, 64, 64)
    old = begin_body()
    rwa = np.zeros((2, 1, 7, 7), np.float32)
    rwa[0, 0, 0, 3] = 1.0
    rwa[1, 0, 6, 3] = 1.0
    rwb = np.zeros((1, 2, 7, 7), np.float32)
    rwb[0, 0, 0, 3] = 1.0
    rwb[0, 1, 6, 3] = 1.0
    row_shift = conv(
        conv(rs, rwa, key="row_phase_a"),
        rwb, key="row_phase_b",
    )
    rn = clip(add(rs, row_shift))
    rl = end_loop(
        old, [{"id": rs, "init": row_seed}], {rs: rn}, 12
    )
    row_lattice = rl + "." + rs

    # Horizontal lattice centres: 3, 9, ..., 63.
    xw = np.zeros((11, 3, 1, 1), np.float32)
    xb = np.zeros(11, np.float32)
    for i in range(11):
        xw[i, 1, 0, 0] = 63.0
        xb[i] = -(3 + 6 * i)
    xd = unary("abs", conv("coords", xw, xb, key="x_centres"))
    xr = clip(scale(relu(scale(xd, -1.0, 1.0)), 100.0, -98.0))
    x_lattice = conv(
        xr, np.ones((1, 11, 1, 1), np.float32),
        key="x_union",
    )
    pixel_row = scale(channel("coords", 0), 63.0)
    valid_low = clip(scale(pixel_row, 4.0, -6.0))
    valid_high = clip(scale(pixel_row, -4.0, 242.0))
    lattice = mul(
        mul(row_lattice, x_lattice),
        mul(valid_low, valid_high),
    )

    # Tile channels:
    # wall, removable, solid-toggle, ghost-toggle, expanding,
    # spike, gravity-switch, goal.
    tw = np.zeros((8, 16, 5, 5), np.float32)
    tb = np.zeros(8, np.float32)
    tw[0, 5, 2, 2] = 1.0
    tw[1, 14, 2, 2] = 1.0
    tw[2, 12, 2, 2] = tw[2, 12, 2, 1] = 1.0
    tb[2] = -1.0
    tw[3, 12, 2, 2] = 1.0
    tw[3, 12, 2, 1] = -1.0
    tw[4, 15, 2, 2] = tw[4, 15, 2, 0] = 1.0
    tb[4] = -1.0
    tw[5, 15, 2, 2] = 1.0
    tw[5, 15, 2, 0] = -1.0
    tw[6, 8, 2, 2] = 1.0
    tw[7, 7, 2, 2] = 1.0
    tiles = mul(
        relu(conv("x0", tw, tb, key="tile_detectors")),
        lattice,
    )
    wall, block, solid, ghost, spread, spike, switch, goal = [
        channel(tiles, i) for i in range(8)
    ]
    player_cell = mul(player, lattice)

    removable = total([block, solid, spread])
    blocking = total([wall, removable, switch])
    walkable = relu(sub(lattice, total([wall, spike, switch])))
    fall_free = relu(sub(lattice, add(blocking, spike)))
    blank = relu(sub(
        lattice,
        total([
            wall, block, solid, ghost, spread,
            spike, switch, goal, player_cell,
        ]),
    ))

    # Click labels encode 8 + row*64 + column.
    cw = np.zeros((1, 3, 1, 1), np.float32)
    cw[0, 0, 0, 0] = 4032.0
    cw[0, 1, 0, 0] = 63.0
    cw[0, 2, 0, 0] = 8.0
    code_map = conv("coords", cw, key="position_code")
    has_switch = pool(switch)
    switch_code = pool(mul(switch, code_map))

    # Construction distance and first prerequisite click.
    construction_allowed = add(blank, spread)
    construction_seed = scale(spread, 16.0)
    construction_label_seed = mul(spread, code_map)
    cs, cl = uid(), uid()
    shapes[cs] = shapes[cl] = (1, 64, 64)
    old = begin_body()
    best_s, best_l = cs, cl
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        candidate_s = relu(sub(shift6(cs, dy, dx), one))
        candidate_l = shift6(cl, dy, dx)
        best_s, best_l = choose(
            best_s, best_l, candidate_s, candidate_l
        )
    csn = mul(best_s, construction_allowed)
    cln = mul(best_l, construction_allowed)
    construction_loop = end_loop(
        old,
        [
            {"id": cs, "init": construction_seed},
            {"id": cl, "init": construction_label_seed},
        ],
        {cs: csn, cl: cln},
        16,
    )
    construction_strength = construction_loop + "." + cs
    construction_label = construction_loop + "." + cl

    reachable_blank = mul(blank, clip(construction_strength))
    virtual_floor = add(ghost, reachable_blank)
    spread_cost = mul(
        reachable_blank,
        scale(sub(constant(16.0), construction_strength), 3.0),
    )
    construction_cost = add(ghost, spread_cost)
    construction_click = add(
        mul(ghost, code_map),
        mul(reachable_blank, construction_label),
    )

    # Channel 0: decreasing-row gravity, the initial gravity.
    # Channel 1: increasing-row gravity.
    floor_blocking = gravity_shift(duplicate(blocking))
    floor_free = gravity_shift(duplicate(fall_free))
    floor_removable = gravity_shift(duplicate(removable))
    floor_blank = gravity_shift(duplicate(blank))
    floor_virtual = gravity_shift(duplicate(virtual_floor))
    floor_construct_cost = gravity_shift(duplicate(construction_cost))
    floor_construct_click = gravity_shift(duplicate(construction_click))
    walk_left = shift6(duplicate(walkable), dx=-1)
    walk_right = shift6(duplicate(walkable), dx=1)

    upper_exit = mul(lattice, clip(scale(pixel_row, -2.0, 15.0)))
    lower_exit = mul(lattice, clip(scale(pixel_row, 2.0, -111.0)))
    exit_modes = cat([upper_exit, lower_exit])
    terminal = add(
        scale(duplicate(goal), 1000.0),
        scale(exit_modes, 80.0),
    )

    removal_cost = total([block, solid, scale(spread, 2.0)])
    own_removal_click = mul(removable, code_map)
    not_removable = sub(one, removable)

    # The camera places the player near row 39 or 29 according to gravity.
    observed_player_row = pool(mul(player, pixel_row))
    dr = dense(
        observed_player_row,
        np.array([[0.5]], np.float32),
        np.array([-16.5], np.float32),
        key="gravity_threshold",
    )
    dp = unary("vrelu", dr)
    de = unary("vrelu", scale(dp, 1.0, -1.0))
    direction_up = binary("vadd", dp, scale(de, -1.0))
    direction_modes = dense(
        direction_up,
        np.array([[1.0], [-1.0]], np.float32),
        np.array([0.0, 1.0], np.float32),
        key="gravity_modes",
    )

    # Only the actual-gravity channel may consume a switch.
    # The opposite channel models its continuation without another reversal.
    switch_available_modes = binary(
        "vmul", direction_modes, has_switch
    )

    # Return reversals discount exploratory value, not the large goal value.
    # The discount is zero for the initial-to-reversed transition.
    return_discount = dense(
        direction_modes,
        np.array([[0.0, 0.0], [0.0, 0.75]], np.float32),
        key="return_exploration_discount",
    )
    exploration_cap = constant(200.0)

    # Reward physical landings that advance the camera.
    forward_up = sub(mul(one, observed_player_row), pixel_row)
    forward_down = scale(forward_up, -1.0)
    forward = cat([forward_up, forward_down])
    advanced = clip(scale(forward, 4.0, -18.0))
    landing_reward = mul(
        mul(advanced, scale(relu(forward), 1.0, 120.0)),
        walkable,
    )

    # V: standing value.
    # F: entering-and-settling value.
    # P: prerequisite click before entry.
    # A: first action for a standing player.
    sv, sf, sp, sa = uid(), uid(), uid(), uid()
    for s in [sv, sf, sp, sa]:
        shapes[s] = (2, 64, 64)

    old = begin_body()
    f_left = shift6(sf, dx=-1)
    f_right = shift6(sf, dx=1)
    p_left = shift6(sp, dx=-1)
    p_right = shift6(sp, dx=1)
    f_below = gravity_shift(sf)
    p_below = gravity_shift(sp)

    left_has_prep = clip(p_left)
    right_has_prep = clip(p_right)
    left_label = add(
        mul(left_has_prep, p_left),
        scale(sub(one, left_has_prep), 3.0),
    )
    right_label = add(
        mul(right_has_prep, p_right),
        scale(sub(one, right_has_prep), 4.0),
    )
    q_left = invalidate(relu(sub(f_left, one)), walk_left)
    q_right = invalidate(relu(sub(f_right, one)), walk_right)

    # Construct a deeper landing before removing the current support.
    q_support = invalidate(f_below, floor_removable)
    support_label = p_below

    swap_w = np.zeros((2, 2, 1, 1), np.float32)
    swap_w[0, 1, 0, 0] = 1.0
    swap_w[1, 0, 0, 0] = 1.0
    opposite_f = conv(sf, swap_w, key="swap_gravity")
    opposite_p = conv(sp, swap_w, key="swap_gravity")

    low_switch_value = binary(
        "min2", opposite_f, exploration_cap
    )
    reversal_value = sub(
        opposite_f, mul(low_switch_value, return_discount)
    )
    q_switch = invalidate(
        relu(sub(reversal_value, one)), switch_available_modes
    )
    switch_has_prep = clip(opposite_p)
    switch_label = add(
        mul(switch_has_prep, opposite_p),
        mul(sub(one, switch_has_prep), switch_code),
    )

    # Directional memory affects tie-breaking, not propagated values.
    v_best, a_best = choose(
        q_left, left_label, q_right, right_label,
        add(q_left, left_tie_bonus),
        add(q_right, right_tie_bonus),
    )
    v_best, a_best = choose(
        v_best, a_best, q_support, support_label
    )
    v_best, a_best = choose(
        v_best, a_best, q_switch, switch_label
    )
    v_next = mul(
        maximum(maximum(relu(v_best), terminal), landing_reward),
        walkable,
    )
    a_next = mul(a_best, walkable)

    # Stop above a solid floor, or continue through the next free cell.
    f_standing = mul(sv, floor_blocking)
    f_flow = mul(f_below, floor_free)
    p_flow = mul(p_below, floor_free)
    f_base, p_base = choose(
        f_standing, zero, f_flow, p_flow
    )

    # Expanding-block removal may directly create an adjacent support.
    spread_landing = mul(mul(sv, spread), floor_blank)
    f_base, p_base = choose(
        f_base, p_base, spread_landing, zero
    )

    # Virtual floors must be built before committing to the fall.
    f_construct = mul(
        relu(sub(sv, floor_construct_cost)), floor_virtual
    )
    f_base, p_base = choose(
        f_base, p_base, f_construct, floor_construct_click
    )
    f_base, p_base = choose(
        terminal, zero, f_base, p_base
    )

    f_next = mul(relu(sub(f_base, removal_cost)), walkable)
    deeper_prep = clip(p_base)
    removable_label = add(
        mul(deeper_prep, p_base),
        mul(sub(one, deeper_prep), own_removal_click),
    )
    p_next = mul(
        add(
            mul(removable, removable_label),
            mul(not_removable, p_base),
        ),
        walkable,
    )

    main_loop = end_loop(
        old,
        [
            {"id": sv, "zeros": [2, 64, 64]},
            {"id": sf, "zeros": [2, 64, 64]},
            {"id": sp, "zeros": [2, 64, 64]},
            {"id": sa, "zeros": [2, 64, 64]},
        ],
        {sv: v_next, sf: f_next, sp: p_next, sa: a_next},
        1,
    )
    action_field = main_loop + "." + sa
    value_field = main_loop + "." + sv

    selected_modes = pool(
        mul(mul(action_field, player), direction_modes)
    )
    selected_code = dense(
        selected_modes, np.ones((1, 2), np.float32),
        key="selected_code",
    )

    player_values = pool(
        mul(mul(value_field, player), direction_modes)
    )
    player_value = dense(
        player_values, np.ones((1, 2), np.float32),
        key="player_value",
    )
    no_continuation = clip(sub(one, player_value))

    left_indicator = clip(relu(sub(
        one, unary("abs", sub(constant(3.0), selected_code))
    )))
    right_indicator = clip(relu(sub(
        one, unary("abs", sub(constant(4.0), selected_code))
    )))
    click_indicator = clip(sub(
        mul(one, selected_code), constant(7.0)
    ))
    action_features = cat([
        pool(left_indicator),
        pool(right_indicator),
        pool(click_indicator),
        pool(no_continuation),
    ], vector=True)

    lw = np.zeros((8, 4), np.float32)
    lb = np.full(8, -100.0, np.float32)
    lb[0] = -10.0
    lb[3] = lb[4] = lb[6] = 0.0
    lw[0, 3] = 40.0
    lw[3, 0] = lw[4, 1] = lw[6, 2] = 20.0
    dense(
        action_features, lw, lb,
        key="action_logits", name="logits",
    )
    click_distance = unary("abs", sub(code_map, selected_code))
    scale(click_distance, -1.0, name="click")

    # Architecture accounting only: choose a fixed iteration horizon.
    def accounting(nodes):
        macs = evaluations = 0
        for nd in nodes:
            if nd["op"] == "loop":
                m, e = accounting(nd["body"])
                macs += nd["T"] * m
                evaluations += 1 + nd["T"] * e
            else:
                macs += costs[nd["id"]]
                evaluations += 1
        return macs, evaluations

    main_node = next(nd for nd in graph if nd["id"] == main_loop)
    body_macs, body_evals = accounting(main_node["body"])
    total_macs, total_evals = accounting(graph)
    fixed_macs = total_macs - body_macs
    fixed_evals = total_evals - body_evals
    by_macs = (1_480_000_000 - fixed_macs) // max(body_macs, 1)
    by_evals = (9_900 - fixed_evals) // max(body_evals, 1)
    main_node["T"] = int(max(1, min(80, by_macs, by_evals)))

    return {
        "graph": graph,
        "memory_size": 1,
        "params": params,
    }
