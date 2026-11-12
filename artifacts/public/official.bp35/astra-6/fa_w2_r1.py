import numpy as np


def build():
    params = {}
    graph = []
    active = graph
    serial = 0
    shift_weights = {}

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return "%s_%d" % (prefix, serial)

    def parameter(a):
        key = "p%d" % len(params)
        params[key] = np.asarray(a, dtype=np.float32)
        return key

    def emit(op, inputs=None, name=None, **fields):
        node = {"id": name or fresh(), "op": op}
        if inputs is not None:
            node["in"] = list(inputs)
        node.update(fields)
        active.append(node)
        return node["id"]

    def conv(x, w, b=None, name=None):
        fields = {"w": w if isinstance(w, str) else parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("dense", [x], name=name, **fields)

    def cat(xs):
        return xs[0] if len(xs) == 1 else emit("concat", xs)

    def lin(xs, weights, bias=0.0, name=None):
        w = np.asarray(weights, np.float32).reshape(1, len(xs), 1, 1)
        return conv(cat(xs), w, [bias], name)

    def scale(x, a=1.0, b=0.0):
        return lin([x], [a], b)

    def vscale(x, n, a=1.0, b=0.0):
        return dense(
            x, np.eye(n, dtype=np.float32) * a,
            np.full(n, b, np.float32)
        )

    def relu(x):
        return emit("relu", [x])

    def mul(a, b):
        return emit("mul", [a, b])

    def add(a, b):
        return emit("add", [a, b])

    def sub(a, b):
        return emit("sub", [a, b])

    def maximum(xs):
        z = xs[0]
        for x in xs[1:]:
            z = emit("max2", [z, x])
        return z

    def sample(x, dx=0, dy=0):
        key = (dx, dy)
        if key not in shift_weights:
            w = np.zeros((1, 1, 7, 7), np.float32)
            w[0, 0, 3 + dy // 2, 3 + dx // 2] = 1
            shift_weights[key] = parameter(w)
        w = shift_weights[key]
        return conv(conv(x, w), w)

    def pooled(x):
        return emit("gmax", [x])

    def vslice(x, start, stop):
        return emit("vslice", [x], start=start, stop=stop)

    one = emit("slice", ["coords"], start=2, stop=3)
    row = conv(
        "coords",
        np.array([63, 0, 0], np.float32).reshape(1, 3, 1, 1)
    )
    col = conv(
        "coords",
        np.array([0, 63, 0], np.float32).reshape(1, 3, 1, 1)
    )

    def clip01(x):
        return emit("min2", [relu(x), one])

    def invert(x):
        return scale(x, -1, 1)

    def mask_score(value, mask, cost=0.0):
        return lin([value, mask], [1, 1000], -1000 - cost)

    def trusted(value):
        return lin([value, clip01(value)], [1, 1000], -1000)

    visible_rows = clip01(scale(row, -2, 125))
    obs = mul("x0", visible_rows)

    def colour(c):
        return emit("slice", [obs], start=c, stop=c + 1)

    empty = colour(10)
    yellow = colour(9)
    gem = colour(7)
    magenta = colour(14)
    red = colour(8)
    green_colour = colour(12)
    pale_colour = colour(15)

    # Tile-centre anchor for either player facing direction.
    w = np.zeros((1, 16, 7, 7), np.float32)
    for dy, c in [
        (-3, 5), (-2, 9), (-1, 9), (0, 9), (1, 9), (2, 5)
    ]:
        w[0, c, 3 + dy, 3] = 1
    player = relu(conv(obs, w, [-5]))
    py = pooled(mul(player, row))
    px = pooled(mul(player, col))

    # Settled camera framing identifies gravity direction.
    up_raw = emit("vrelu", [vscale(py, 1, 1, -34)])
    up_excess = emit("vrelu", [vscale(up_raw, 1, 1, -1)])
    up = emit("vadd", [up_raw, vscale(up_excess, 1, -1)])
    down = vscale(up, 1, -1, 1)
    up_map = mul(one, up)

    def choose_direction(up_value, down_value):
        return add(mul(up_value, up), mul(down_value, down))

    # Distinguish a solid green block from its transparent cross.
    w = np.zeros((1, 16, 3, 3), np.float32)
    w[0, 12, 1, 1] = 1
    w[0, 12, 0, 1] = 1
    solid_green = relu(conv(obs, w, [-1]))
    ghost_green = sub(green_colour, solid_green)

    # White block centre: wide white interior and grey corner.
    w = np.zeros((1, 16, 5, 5), np.float32)
    w[0, 15, 2, 2] = 1
    w[0, 15, 2, 0] = 1
    w[0, 3, 0, 0] = 1
    white = relu(conv(obs, w, [-2]))
    spike = sub(pale_colour, white)

    free = lin([empty, yellow, gem, ghost_green], [1, 1, 1, 1])
    breakable = lin([magenta, solid_green, white], [1, 1, 1])
    ordinary_breakable = lin([magenta, solid_green], [1, 1])
    solid_support = mul(
        lin([free, spike], [-1, -1], 1), visible_rows
    )

    under = {
        -6: sample(player, 0, 6),
        6: sample(player, 0, -6),
    }
    current_under = choose_direction(under[-6], under[6])

    # Underfoot white expansion is a movement, not ordinary construction.
    protected_white = mul(current_under, white)
    construction_white = mul(white, invert(protected_white))
    growth_terrain = add(empty, construction_white)

    def lattice_axis(axis, offsets, subtract_vector=None):
        offsets = np.asarray(offsets, np.float32)
        n = len(offsets)
        w = np.zeros((n, 3, 1, 1), np.float32)
        w[:, axis, 0, 0] = 63
        d = conv("coords", w, -offsets)
        if subtract_vector is not None:
            d = sub(d, subtract_vector)
        d = emit("abs", [d])
        w = -np.eye(n, dtype=np.float32).reshape(n, n, 1, 1)
        triangles = relu(conv(d, w, np.ones(n, np.float32)))
        return conv(triangles, np.ones((1, n, 1, 1), np.float32))

    lattice_x = lattice_axis(1, 3 + 6 * np.arange(11))
    lattice_y = lattice_axis(0, 6 * np.arange(-10, 11), py)
    lattice = mul(mul(lattice_x, lattice_y), visible_rows)

    red_centres = mul(red, lattice)
    red_present = pooled(clip01(scale(red_centres, 2)))
    red_count = vscale(emit("gmean", [red_centres]), 1, 4096)
    two_red_map = clip01(scale(mul(one, red_count), 1, -1))

    # At a hypothetical settled player position, determine whether a
    # known red block would remain inside that gravity's camera window.
    red_rows = emit("rowmax", [red_centres])
    red_offsets = {0: red_rows}
    for k in range(1, 7):
        red_offsets[-k] = sample(red_offsets[-k + 1], 0, -6)
    for k in range(1, 6):
        red_offsets[k] = sample(red_offsets[k - 1], 0, 6)

    red_visible = {
        -6: maximum([red_offsets[k] for k in range(-6, 4)]),
        6: maximum([red_offsets[k] for k in range(-4, 6)]),
    }
    two_red_visible = {
        dy: mul(red_visible[dy], two_red_map)
        for dy in (-6, 6)
    }

    goal_value = scale(gem, 200)
    top_edge = clip01(scale(row, -1, 6))
    bottom_edge = clip01(scale(row, 1, -56))
    top_landing = clip01(scale(row, -1.0 / 6.0, 3))
    bottom_landing = clip01(scale(row, 1.0 / 6.0, -7.5))
    top_reward = scale(up_map, 10, 70)
    bottom_reward = scale(up_map, -10, 80)

    landing_reward = {
        -6: mul(top_landing, top_reward),
        6: mul(bottom_landing, bottom_reward),
    }
    frontier = {
        -6: mul(top_edge, scale(top_reward, 1, -18)),
        6: mul(bottom_edge, scale(bottom_reward, 1, -18)),
    }

    py_map = mul(one, py)
    local_seed = {
        -6: relu(scale(sub(py_map, row), 1.0 / 6.0, 4)),
        6: relu(scale(sub(row, py_map), 1.0 / 6.0, 4)),
    }

    def diffusion(seed, terrain, iterations=20):
        nonlocal active
        saved = active
        body = []
        active = body
        state = fresh("diff_state")
        neighbours = [
            sample(state, -6, 0),
            sample(state, 6, 0),
            sample(state, 0, -6),
            sample(state, 0, 6),
        ]
        spread = mul(relu(scale(maximum(neighbours), 1, -1)), terrain)
        nxt = maximum([seed, spread])
        active = saved
        loop = emit(
            "loop", T=iterations,
            state=[{"id": state, "init": seed}],
            body=body, next={state: nxt},
        )
        return loop + "." + state

    white_strength = diffusion(
        scale(construction_white, 24), growth_terrain
    )
    growable = mul(empty, clip01(white_strength))
    grow_cost = mul(scale(white_strength, -1, 24), growable)
    support_cost = lin(
        [grow_cost, ghost_green, solid_support, growable],
        [1, -999, -1000, -1000], 1000,
    )

    # Shift support availability rather than cost: padding is unknown,
    # not a fictitious zero-cost platform outside the image.
    support_saving = scale(support_cost, -1, 1000)

    horizontal_cost = lin([free, breakable], [-1000, -999], 1001)
    horizontal_left = sample(horizontal_cost, -6, 0)
    horizontal_right = sample(horizontal_cost, 6, 0)

    stop_cost = {}
    pass_cost = {}
    drop_cost = {}
    safe_next = {}
    white_stop_at_tile = {}
    white_stop_at_player = {}
    white_stop_inverse = {}

    for dy in (-6, 6):
        stop_cost[dy] = scale(
            sample(support_saving, 0, dy), -1, 1000
        )
        pass_cost[dy] = scale(
            sample(free, 0, dy), -1000, 1000.04
        )
        drop_cost[dy] = scale(
            sample(breakable, 0, dy), -1000, 1001
        )
        safe_next[dy] = sample(solid_support, 0, dy)
        white_stop_at_tile[dy] = mul(white, sample(empty, 0, dy))
        white_stop_at_player[dy] = sample(
            white_stop_at_tile[dy], 0, dy
        )
        white_stop_inverse[dy] = invert(white_stop_at_player[dy])

    def value_iteration(flip_options=()):
        nonlocal active
        flip_offers = {}
        for dy in (-6, 6):
            flip_offers[dy] = [
                mask_score(falls[-dy], availability[dy], 2)
                for falls, availability in flip_options
            ]

        saved = active
        body = []
        active = body
        states = {
            dy: {"S": fresh("standing"), "F": fresh("falling")}
            for dy in (-6, 6)
        }
        updates = {}

        for dy in (-6, 6):
            s = states[dy]["S"]
            f = states[dy]["F"]
            left_value = sub(sample(f, -6, 0), horizontal_left)
            right_value = sub(sample(f, 6, 0), horizontal_right)
            below_f = sample(f, 0, dy)
            below_s = sample(s, 0, dy)
            after_drop = add(
                mul(below_f, white_stop_inverse[dy]),
                mul(below_s, white_stop_at_player[dy]),
            )
            drop_value = sub(after_drop, drop_cost[dy])

            offers = [
                left_value, right_value, drop_value,
                local_seed[dy], landing_reward[dy], goal_value,
            ] + flip_offers[dy]
            new_s = relu(maximum(offers))
            new_f = relu(maximum([
                sub(new_s, stop_cost[dy]),
                sub(below_f, pass_cost[dy]),
                goal_value, frontier[dy],
            ]))
            updates[s] = new_s
            updates[f] = new_f

        active = saved
        loop = emit(
            "loop", T=36,
            state=[
                {"id": states[dy][kind], "zeros": [1, 64, 64]}
                for dy in (-6, 6) for kind in ("S", "F")
            ],
            body=body, next=updates,
        )
        return {
            dy: {
                kind: loop + "." + states[dy][kind]
                for kind in ("S", "F")
            }
            for dy in (-6, 6)
        }

    def actual_falls(values):
        nonlocal active
        standing_stops = {
            dy: mul(values[dy]["S"], safe_next[dy])
            for dy in (-6, 6)
        }
        saved = active
        body = []
        active = body
        states = {dy: fresh("actual_fall") for dy in (-6, 6)}
        updates = {}
        for dy in (-6, 6):
            passed = sub(sample(states[dy], 0, dy), pass_cost[dy])
            updates[states[dy]] = relu(maximum([
                standing_stops[dy], passed, goal_value, frontier[dy]
            ]))
        active = saved
        loop = emit(
            "loop", T=12,
            state=[
                {"id": states[dy], "zeros": [1, 64, 64]}
                for dy in (-6, 6)
            ],
            body=body, next=updates,
        )
        return {dy: loop + "." + states[dy] for dy in (-6, 6)}

    no_flip = value_iteration()
    plain_actual = actual_falls(no_flip)
    one_flip = value_iteration([(plain_actual, red_visible)])
    one_flip_actual = actual_falls(one_flip)
    main_values = value_iteration([
        (plain_actual, red_visible),
        (one_flip_actual, two_red_visible),
    ])
    main_actual = actual_falls(main_values)

    current_actual = choose_direction(main_actual[-6], main_actual[6])
    current_prepared = choose_direction(
        main_values[-6]["F"], main_values[6]["F"]
    )

    left_position = sample(player, 6, 0)
    right_position = sample(player, -6, 0)
    side_positions = clip01(add(left_position, right_position))
    free_side_positions = mul(side_positions, free)

    # Possible immediate falling rays identify useful construction sites.
    ray_seeds = {}
    for dy, direction in ((-6, up), (6, down)):
        normal_seed = clip01(add(
            free_side_positions,
            mul(under[dy], ordinary_breakable),
        ))
        ray_seeds[("normal", dy)] = mul(normal_seed, direction)
        opposite_direction = down if dy == -6 else up
        ray_seeds[("flip", dy)] = mul(
            mul(player, opposite_direction), red_present
        )

    saved = active
    ray_body = []
    active = ray_body
    ray_states = {key: fresh("ray") for key in ray_seeds}
    ray_updates = {}
    for key, seed in ray_seeds.items():
        _, dy = key
        propagated = mul(sample(ray_states[key], 0, -dy), free)
        ray_updates[ray_states[key]] = maximum([seed, propagated])
    active = saved
    ray_loop = emit(
        "loop", T=10,
        state=[
            {"id": ray_states[key], "init": ray_seeds[key]}
            for key in ray_seeds
        ],
        body=ray_body, next=ray_updates,
    )

    construction_offers = []
    for key in ray_seeds:
        label, dy = key
        ray = ray_loop + "." + ray_states[key]
        values = main_values if label == "normal" else no_flip
        action_cost = 1 if label == "normal" else 2
        resting_offer = mask_score(values[dy]["S"], ray, action_cost)
        construction_offers.append(sample(resting_offer, 0, -dy))

    construction_value = maximum(construction_offers)
    expansion_seed = mul(relu(construction_value), empty)
    expansion_value = diffusion(expansion_seed, growth_terrain)

    clear_score = mask_score(
        trusted(current_prepared),
        mul(side_positions, breakable), 2,
    )

    after_white_drop = {}
    for dy in (-6, 6):
        special = white_stop_at_tile[dy]
        after_white_drop[dy] = add(
            mul(main_actual[dy], invert(special)),
            mul(main_values[dy]["S"], special),
        )
    drop_outcome = choose_direction(
        after_white_drop[-6], after_white_drop[6]
    )
    drop_score = mask_score(
        trusted(drop_outcome), mul(current_under, breakable), 1
    )

    # Controlled white stepping reveals terrain without committing to
    # a potentially lethal off-screen fall. Disable this preference
    # whenever the planner already has a goal-reaching value.
    current_value = pooled(mul(current_actual, player))
    exploration = clip01(scale(mul(one, current_value), -1, 150))
    controlled_step = mul(protected_white, exploration)
    drop_score = add(drop_score, scale(controlled_step, 3))

    green_score = mask_score(construction_value, ghost_green, 1)
    white_score = mask_score(
        expansion_value, construction_white, 0.15
    )

    remaining_flip = {}
    for dy in (-6, 6):
        remaining_flip[dy] = add(
            mul(one_flip_actual[dy], two_red_map),
            mul(plain_actual[dy], invert(two_red_map)),
        )
    opposite_actual = add(
        mul(remaining_flip[-6], down),
        mul(remaining_flip[6], up),
    )
    flip_outcome = pooled(mul(opposite_actual, player))
    red_score = mask_score(
        trusted(mul(one, flip_outcome)), red, 2
    )

    # After undoing a bad reversal, first seek a different launch point.
    previous_undo = vslice("aprev", 7, 8)
    red_score = sub(
        red_score, mul(one, vscale(previous_undo, 1, 30))
    )

    click_candidates = maximum([
        clear_score, drop_score, green_score, white_score, red_score
    ])
    click_candidates = mask_score(click_candidates, lattice)

    # Local history suppresses repeated white expansion cycles.
    penalties = []
    for i, weight in enumerate((5.0, 4.0, 3.0, 2.0)):
        base = 50 + 3 * i
        old_y = vslice("mem", base, base + 1)
        old_x = vslice("mem", base + 1, base + 2)
        old_valid = vslice("mem", base + 2, base + 3)
        distance = add(
            emit("abs", [sub(row, old_y)]),
            emit("abs", [sub(col, old_x)]),
        )
        same_cell = relu(scale(distance, -1, 1))
        penalties.append(scale(mul(same_cell, old_valid), weight))
    repeat_penalty = penalties[0]
    for p in penalties[1:]:
        repeat_penalty = add(repeat_penalty, p)
    click_candidates = sub(
        click_candidates, mul(white, repeat_penalty)
    )

    distance = add(
        emit("abs", [sub(row, py)]),
        emit("abs", [sub(col, px)]),
    )
    tie_break = lin(
        [distance, row, col], [-0.0002, -0.00005, -0.000003]
    )
    click = emit("add", [click_candidates, tie_break], name="click")
    click_logit = pooled(click)

    safe_actual = trusted(current_actual)
    left_logit = pooled(mask_score(
        safe_actual, mul(left_position, free), 1
    ))
    right_logit = pooled(mask_score(
        safe_actual, mul(right_position, free), 1
    ))

    # Visual fingerprint excludes the action counter.
    def sums(x):
        return vscale(emit("gmean", [x]), 16, 4096)

    counts = sums(obs)
    row_moment = sums(mul(obs, scale(row, 1.0 / 64.0)))
    col_moment = sums(mul(obs, scale(col, 1.0 / 64.0)))
    features = emit("vconcat", [counts, py, px, row_moment, col_moment])

    previous_features = vslice("mem", 0, 50)
    difference = emit("vadd", [
        features, vscale(previous_features, 50, -1)
    ])
    positive = emit("vrelu", [difference])
    negative = emit("vrelu", [vscale(difference, 50, -1)])
    absolute = emit("vadd", [positive, negative])
    unchanged = emit("vrelu", [
        dense(absolute, -2 * np.ones((1, 50), np.float32), [1])
    ])
    ineffective_previous = emit("vmul", ["aprev", unchanged])

    action_features = emit("vconcat", [
        left_logit, right_logit, click_logit,
        "aprev", ineffective_previous,
    ])
    w = np.zeros((8, 19), np.float32)
    b = np.full(8, -10000, np.float32)
    b[7] = -500
    for action, feature in ((3, 0), (4, 1), (6, 2)):
        w[action, feature] = 1
        b[action] = 0
        w[action, 11 + action] = -60
    w[3, 3 + 4] = -0.08
    w[4, 3 + 3] = -0.08
    logits = dense(action_features, w, b, name="logits")

    # Recover the uniquely tie-broken click maximum for local memory.
    best_mask = relu(scale(sub(click, click_logit), 1000000, 1))
    chosen_y = pooled(mul(best_mask, row))
    chosen_x = pooled(mul(best_mask, col))
    chosen_white = pooled(mul(best_mask, white))

    left_final = vslice(logits, 3, 4)
    right_final = vslice(logits, 4, 5)
    click_final = vslice(logits, 6, 7)
    difference = emit("vadd", [
        left_final, vscale(right_final, 1, -1)
    ])
    horizontal_max = emit("vadd", [
        emit("vrelu", [difference]), right_final
    ])
    click_margin = emit("vadd", [
        click_final, vscale(horizontal_max, 1, -1)
    ])
    gate_raw = emit("vrelu", [vscale(click_margin, 1, 1024)])
    gate_excess = emit("vrelu", [vscale(gate_raw, 1, 1, -1)])
    click_gate = emit("vadd", [
        gate_raw, vscale(gate_excess, 1, -1)
    ])
    history_valid = emit("vmul", [click_gate, chosen_white])
    previous_history = vslice("mem", 50, 59)
    emit(
        "vconcat",
        [features, chosen_y, chosen_x, history_valid, previous_history],
        name="mem_next"
    )

    return {
        "graph": graph,
        "memory_size": 62,
        "params": params,
    }
