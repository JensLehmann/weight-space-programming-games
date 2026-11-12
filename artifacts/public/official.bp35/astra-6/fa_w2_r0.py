import numpy as np


def build():
    params = {}
    graph = []
    stack = [graph]
    channels = {
        "x0": 16, "coords": 3, "aprev": 8, "avail": 8, "mem": 2
    }
    serial = [0]
    kernel_cache = {}

    def fresh(prefix="n"):
        serial[0] += 1
        return "%s%d" % (prefix, serial[0])

    def parameter(a):
        name = "P%d" % len(params)
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def emit(op, inputs, count, name=None, **fields):
        name = name or fresh()
        node = {"id": name, "op": op, "in": list(inputs)}
        node.update(fields)
        stack[-1].append(node)
        channels[name] = count
        return name

    def unary(op, a):
        return emit(op, [a], channels[a])

    def binary(op, a, b, name=None):
        return emit(
            op, [a, b], max(channels[a], channels[b]), name=name
        )

    def concat(items, vector=False, name=None):
        return emit(
            "vconcat" if vector else "concat",
            items, sum(channels[s] for s in items), name=name
        )

    def take(a, start, stop, vector=False):
        return emit(
            "vslice" if vector else "slice",
            [a], stop - start, start=start, stop=stop
        )

    def linear(a, matrix, bias=None, vector=False, name=None):
        w = np.asarray(matrix, dtype=np.float32)
        if w.ndim == 1:
            w = w[None, :]
        assert w.ndim == 2 and w.shape[1] == channels[a]
        out = w.shape[0]
        fields = {
            "w": parameter(w if vector else w[:, :, None, None])
        }
        if bias is not None:
            b = np.asarray(bias, dtype=np.float32)
            if b.ndim == 0:
                b = np.full(out, b, dtype=np.float32)
            assert b.shape == (out,)
            fields["b"] = parameter(b)
        return emit(
            "dense" if vector else "conv",
            [a], out, name=name, **fields
        )

    def vl(a, matrix, bias=None, name=None):
        return linear(a, matrix, bias, vector=True, name=name)

    def affine(a, scale, bias=0.0, vector=False):
        return linear(
            a,
            np.eye(channels[a], dtype=np.float32) * scale,
            bias, vector=vector
        )

    def va(a, scale, bias=0.0):
        return affine(a, scale, bias, vector=True)

    def relu(a):
        return unary("relu", a)

    def clip(a, vector=False):
        if vector:
            c = channels[a]
            eye = np.eye(c, dtype=np.float32)
            z = vl(
                a, np.concatenate([eye, eye], axis=0),
                np.concatenate([np.zeros(c), -np.ones(c)])
            )
            z = unary("vrelu", z)
            return vl(z, np.concatenate([eye, -eye], axis=1))
        r = relu(a)
        s = relu(affine(a, 1.0, -1.0))
        return binary("sub", r, s)

    def vsub(a, b):
        assert channels[a] == channels[b]
        c = channels[a]
        eye = np.eye(c, dtype=np.float32)
        return vl(
            concat([a, b], True),
            np.concatenate([eye, -eye], axis=1)
        )

    def vmix(a, b, gate):
        return binary(
            "vadd", a, binary("vmul", vsub(b, a), gate)
        )

    def pool(a):
        return emit("gmax", [a], channels[a])

    def duplicate(a):
        assert channels[a] == 1
        return linear(a, [[1.0], [1.0]])

    def shift(a, direction):
        # Two three-pixel shifts implement one six-pixel tile step.
        # Splitting channels into pairs reduces the dense-convolution MACs.
        c = channels[a]
        if c > 2:
            assert c % 2 == 0
            return concat([
                shift(take(a, j, j + 2), direction)
                for j in range(0, c, 2)
            ])

        key = (c, direction)
        if key not in kernel_cache:
            w = np.zeros((c, c, 7, 7), np.float32)
            for j in range(c):
                dy = dx = 0
                if direction == "L":
                    dx = -3
                elif direction == "R":
                    dx = 3
                elif direction == "U":
                    dy = -3
                elif direction == "D":
                    dy = 3
                elif direction == "G":
                    assert c == 2
                    dy = -3 if j == 0 else 3
                else:
                    raise ValueError(direction)
                w[j, j, 3 + dy, 3 + dx] = 1.0
            kernel_cache[key] = parameter(w)
        wname = kernel_cache[key]
        a = emit("conv", [a], c, w=wname)
        return emit("conv", [a], c, w=wname)

    def loop(iterations, initial, body_function):
        lid = fresh("loop")
        sid = fresh("state")
        channels[sid] = channels[initial]
        body = []
        stack.append(body)
        final = body_function(sid)
        stack.pop()
        assert channels[final] == channels[initial]
        stack[-1].append({
            "id": lid,
            "op": "loop",
            "T": iterations,
            "state": [{"id": sid, "init": initial}],
            "body": body,
            "next": {sid: final},
        })
        result = lid + "." + sid
        channels[result] = channels[initial]
        return result

    def wave(seed, allowed, iterations=20):
        def body(z):
            u = shift(z, "U")
            d = shift(z, "D")
            l = shift(z, "L")
            r = shift(z, "R")
            q = binary(
                "max2", binary("max2", u, d), binary("max2", l, r)
            )
            q = relu(affine(q, 1.0, -1.0))
            return binary(
                "mul", binary("max2", q, seed), allowed
            )
        return loop(iterations, seed, body)

    # Locate the central column of the player sprite.
    wp = np.zeros((1, 16, 7, 7), np.float32)
    for dy in (-2, -1, 0, 1):
        wp[0, 9, 3 + dy, 3] = 1.0
    player = emit(
        "conv", ["x0"], 1,
        w=parameter(wp), b=parameter([-3.0])
    )
    player = relu(player)
    pcoords = pool(binary("mul", "coords", player))

    row = take("coords", 0, 1)
    col = take("coords", 1, 2)
    row_pixels = affine(row, 63.0)
    col_pixels = affine(col, 63.0)
    player_row_pixels = vl(pcoords, [63.0, 0.0, 0.0])

    # Camera placement encodes the current gravity direction.
    up = clip(
        vl(pcoords, [31.5, 0.0, 0.0], -16.5), vector=True
    )
    directions = vl(up, [[1.0], [-1.0]], [0.0, 1.0])
    opposite_directions = vl(up, [[-1.0], [1.0]], [1.0, 0.0])

    old_down = take("mem", 0, 1, True)
    old_complex = take("mem", 1, 2, True)
    down = va(up, -1.0, 1.0)
    remembered_down = clip(
        binary("vadd", old_down, down), vector=True
    )
    preferred_directions = vl(
        remembered_down, [[-1.0], [1.0]], [1.0, 0.0]
    )

    def comb(coordinate, centers):
        centers = np.asarray(centers, np.float32)
        z = linear(
            coordinate, np.ones((len(centers), 1), np.float32),
            -centers
        )
        z = unary("abs", z)
        z = relu(affine(z, -4.0, 1.0))
        z = linear(z, np.ones((1, len(centers)), np.float32))
        return clip(affine(z, 2.0, -0.5))

    lattice_x = comb(col_pixels, list(range(3, 64, 6)))
    relative_rows = binary(
        "sub", row_pixels, player_row_pixels
    )
    lattice_y = comb(
        relative_rows, [6 * j for j in range(-10, 11)]
    )
    lattice = binary("mul", lattice_x, lattice_y)
    interior_x = clip(affine(col_pixels, -1.0, 62.0))
    interior_y = clip(affine(row_pixels, -1.0, 62.0))
    lattice = binary(
        "mul", lattice, binary("mul", interior_x, interior_y)
    )

    # Channels: open, pink, red, solid cyan, ghost cyan,
    # growing white, wall, goal, genuinely empty.
    wt = np.zeros((9, 16, 5, 5), np.float32)
    bt = np.zeros(9, np.float32)
    for color in (10, 9, 7):
        wt[0, color, 2, 2] = 1.0
    wt[1, 14, 2, 2] = 1.0
    wt[2, 8, 2, 2] = 1.0

    wt[3, 12, 2, 2] = 1.0
    wt[3, 12, 2, 3] = 1.0
    bt[3] = -1.0

    wt[4, 12, 2, 2] = 1.0
    wt[4, 12, 2, 3] = -1.0

    # White blocks have a gray corner and a five-pixel-wide interior.
    wt[5, 15, 2, 2] = 1.0
    wt[5, 15, 2, 4] = 1.0
    wt[5, 3, 0, 0] = 1.0
    bt[5] = -2.0

    wt[6, 5, 2, 2] = 1.0
    wt[7, 7, 2, 2] = 1.0
    wt[8, 10, 2, 2] = 1.0

    tiles = emit(
        "conv", ["x0"], 9, w=parameter(wt), b=parameter(bt)
    )
    tiles = binary("mul", relu(tiles), lattice)

    def tile_sum(indices):
        w = np.zeros((1, 9), np.float32)
        w[0, indices] = 1.0
        return linear(tiles, w)

    opened = tile_sum([0, 4])
    mineable = tile_sum([1, 3, 5])
    red = take(tiles, 2, 3)
    ghost = take(tiles, 4, 5)
    white = take(tiles, 5, 6)
    solid = tile_sum([1, 2, 3, 5, 6])
    gem = take(tiles, 7, 8)
    empty = take(tiles, 8, 9)
    horizontal_allowed = binary("add", opened, mineable)
    occupiable = binary("add", horizontal_allowed, red)
    red_exists = pool(red)
    white_exists = pool(white)

    complex_now = pool(tile_sum([3, 4, 5]))
    remembered_complex = clip(
        binary("vadd", old_complex, complex_now), vector=True
    )
    concat(
        [remembered_down, remembered_complex],
        vector=True, name="mem_next"
    )

    # Hysteresis for simple reversible routes; local gravity-following
    # progress for routes requiring platform manipulation.
    progress_directions = vmix(
        preferred_directions, directions, remembered_complex
    )

    # Resource layers: zero, one, or two remaining gravity reversals.
    red_mean = emit("gmean", [red], 1)
    red_count = va(red_mean, 4096.0)
    has_one = clip(red_count, vector=True)
    has_two = clip(va(red_count, 1.0, -1.0), vector=True)
    budget_onehot = vl(
        concat([has_one, has_two], True),
        [[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]],
        [1.0, 0.0, 0.0]
    )
    wb = np.zeros((6, 3), np.float32)
    for j in range(3):
        wb[2 * j:2 * j + 2, j] = 1.0
    budget_channels = vl(budget_onehot, wb)
    selected_channels = binary(
        "vmul", budget_channels,
        concat([directions, directions, directions], True)
    )

    def repeat_budgets(a):
        assert channels[a] == 2
        return concat([a, a, a])

    def all_directions(a):
        return repeat_budgets(duplicate(a))

    def ahead(a):
        return shift(duplicate(a), "G")

    # A supporting white block may be clicked safely if the next tile is
    # empty (a new white block is created), solid, or the goal. A spike or
    # an unverified ghost shaft beyond it is not a safe growth source.
    support_pair = shift(duplicate(player), "G")
    support_map = linear(
        binary("mul", support_pair, opposite_directions),
        [[1.0, 1.0]]
    )
    safe_after_source = tile_sum([1, 2, 3, 5, 6, 7, 8])
    safe_after_source = linear(
        binary("mul", ahead(safe_after_source), directions),
        [[1.0, 1.0]]
    )
    unsafe_source = binary(
        "mul", support_map,
        affine(safe_after_source, -1.0, 1.0)
    )
    grow_white = binary(
        "mul", white, affine(unsafe_source, -1.0, 1.0)
    )

    # White-growth reachability and approximate click distance.
    growth_allowed = binary("add", empty, grow_white)
    white_wave = wave(affine(grow_white, 40.0), growth_allowed)
    growth_reachable = binary("mul", clip(white_wave), empty)
    growth_cost = binary(
        "mul", affine(white_wave, -2.0, 80.0),
        growth_reachable
    )

    g_open = repeat_budgets(ahead(opened))
    g_mine = repeat_budgets(ahead(mineable))
    g_solid = repeat_budgets(ahead(solid))
    g_ghost = repeat_budgets(ahead(ghost))
    g_growth = repeat_budgets(ahead(growth_reachable))
    g_growth_cost = repeat_budgets(ahead(growth_cost))
    g_optional = binary("add", g_ghost, g_growth)
    preparation_cost = binary(
        "add", affine(g_ghost, 2.0), g_growth_cost
    )
    vertical_allowed = binary("add", g_open, g_mine)
    entry_cost = affine(all_directions(mineable), 2.0)
    falling_cost = affine(g_mine, 2.0)
    horizontal_allowed_all = all_directions(horizontal_allowed)
    occupiable_all = all_directions(occupiable)

    progress = concat([
        affine(row_pixels, -1.0, 63.0), row_pixels
    ])
    progress = binary("mul", progress, progress_directions)
    progress = linear(progress, [[1.0, 1.0]])
    safe_reward = affine(progress, 2.0, 100.0)
    safe_reward_all = all_directions(safe_reward)
    safe_terminals = binary(
        "mul", safe_reward_all, g_solid
    )

    # Prefer verified intermediate landings over blind falls when the
    # observation supplies resources with which to construct a landing.
    landing_resources = pool(binary("max2", ghost, white))
    ordinary_frontier = vl(
        preferred_directions,
        30.0 * np.eye(2, dtype=np.float32),
        [480.0, 480.0]
    )
    cautious_frontier = vl(
        preferred_directions,
        5.0 * np.eye(2, dtype=np.float32),
        [180.0, 180.0]
    )
    cautious_frontier = binary(
        "vadd", cautious_frontier, va(white_exists, -60.0)
    )
    frontier_rewards = vmix(
        ordinary_frontier, cautious_frontier, landing_resources
    )

    upper_edge = clip(affine(row_pixels, -1.0, 6.0))
    lower_edge = clip(affine(row_pixels, 1.0, -56.0))
    frontiers = binary(
        "mul", concat([upper_edge, lower_edge]), duplicate(opened)
    )
    frontier_terminals = repeat_budgets(
        binary("mul", frontiers, frontier_rewards)
    )
    terminals = binary(
        "max2", safe_terminals, frontier_terminals
    )
    terminals = binary(
        "max2", terminals, affine(all_directions(gem), 2000.0)
    )

    reverse_matrix = np.zeros((6, 6), np.float32)
    for budget in (1, 2):
        for direction in (0, 1):
            reverse_matrix[
                2 * budget + direction,
                2 * (budget - 1) + (direction ^ 1)
            ] = 1.0

    def value_parts(v):
        enter = binary(
            "mul",
            relu(binary("sub", v, entry_cost)),
            horizontal_allowed_all
        )
        left = relu(affine(shift(enter, "L"), 1.0, -2.0))
        right = relu(affine(shift(enter, "R"), 1.0, -2.0))
        reverse = relu(affine(
            linear(v, reverse_matrix), 1.0, -6.0
        ))
        stable = binary(
            "max2", binary("max2", left, right), reverse
        )
        falling = binary(
            "mul",
            relu(binary("sub", shift(v, "G"), falling_cost)),
            vertical_allowed
        )
        prepared_continuation = binary(
            "max2", stable, safe_reward_all
        )
        prepared = binary(
            "mul",
            relu(binary(
                "sub", prepared_continuation, preparation_cost
            )),
            g_optional
        )
        return left, right, reverse, stable, falling, prepared

    def value_body(v):
        left, right, reverse, stable, falling, prepared = value_parts(v)
        grounded = binary("mul", stable, g_solid)
        q = binary("max2", falling, grounded)
        q = binary("max2", q, prepared)
        q = binary("max2", q, terminals)
        return binary("mul", q, occupiable_all)

    values = loop(64, terminals, value_body)
    left_value, right_value, reverse_value, stable_value, fall_value, prep_value = (
        value_parts(values)
    )

    # Propagate the first required preparatory click backward through
    # each shaft. Payload: six x values, six y values, six click kinds.
    prep_difference = binary(
        "sub", prep_value, binary("max2", fall_value, terminals)
    )
    prep_chosen = clip(affine(prep_difference, 4.0, -0.25))
    prep_chosen = binary("mul", prep_chosen, occupiable_all)
    target_x = all_directions(col)
    target_y = linear(
        all_directions(row),
        np.eye(6, dtype=np.float32),
        [-6.0 / 63.0, 6.0 / 63.0] * 3
    )
    target_kind = binary("add", g_ghost, affine(g_growth, 2.0))
    local_payload = concat([
        binary("mul", target_x, prep_chosen),
        binary("mul", target_y, prep_chosen),
        binary("mul", target_kind, prep_chosen),
    ])
    continue_fall = binary(
        "mul", g_open, affine(prep_chosen, -1.0, 1.0)
    )
    continue_fall = binary(
        "mul", continue_fall,
        affine(clip(terminals), -1.0, 1.0)
    )
    continue_fall = binary("mul", continue_fall, occupiable_all)
    continue_payload = concat([continue_fall] * 3)

    def preparation_body(z):
        return binary(
            "add", local_payload,
            binary("mul", shift(z, "G"), continue_payload)
        )

    preparations = loop(11, local_payload, preparation_body)
    payload_selection = concat([selected_channels] * 3, True)

    def at_one(a):
        assert channels[a] == 1
        return pool(binary("mul", a, player))

    def at_direction(a):
        assert channels[a] == 6
        a = binary("mul", a, selected_channels)
        a = linear(a, np.ones((1, 6), np.float32))
        return at_one(a)

    def at_payload(a):
        assert channels[a] == 18
        a = binary("mul", a, payload_selection)
        w = np.zeros((3, 18), np.float32)
        for j in range(3):
            w[j, 6 * j:6 * j + 6] = 1.0
        return pool(binary("mul", linear(a, w), player))

    def payload_flag(command):
        return clip(take(command, 2, 3, True), vector=True)

    def horizontal_target(dx):
        return vl(
            pcoords,
            [[0.0, 1.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
            [dx / 63.0, 0.0, 1.0]
        )

    # Candidate fields: score, left, right, click, x, y, kind.
    def movement_candidate(value, direction, action_index):
        shifted_mine = at_one(shift(mineable, direction))
        command = at_payload(shift(preparations, direction))
        direct = horizontal_target(-6 if direction == "L" else 6)
        command = vmix(command, direct, shifted_mine)
        flag = payload_flag(command)

        valid = at_one(shift(horizontal_allowed, direction))
        features = concat(
            [at_direction(value), valid, "aprev"], True
        )
        w = np.zeros((1, 10), np.float32)
        w[0, 0] = 100.0
        w[0, 1] = 2.0
        inverse_action = 4 if action_index == 3 else 3
        w[0, 2 + inverse_action] = -1.0
        score = vl(features, w)

        features = concat([score, command, flag], True)
        w = np.zeros((7, 5), np.float32)
        b = np.zeros(7, np.float32)
        w[0, 0] = 1.0
        movement_slot = 1 if action_index == 3 else 2
        b[movement_slot] = 1.0
        w[movement_slot, 4] = -1.0
        w[3, 4] = 1.0
        w[4, 1] = 1.0
        w[5, 2] = 1.0
        w[6, 3] = 1.0
        return vl(features, w, b)

    left_candidate = movement_candidate(left_value, "L", 3)
    right_candidate = movement_candidate(right_value, "R", 4)

    def click_candidate(score, command):
        f = concat([score, command], True)
        w = np.zeros((7, 4), np.float32)
        b = np.zeros(7, np.float32)
        w[0, 0] = 1.0
        b[3] = 1.0
        w[4, 1] = 1.0
        w[5, 2] = 1.0
        w[6, 3] = 1.0
        return vl(f, w, b)

    mine_strength = at_direction(binary("mul", fall_value, g_mine))
    mine_present = at_direction(g_mine)
    mine_score = vl(
        concat([mine_strength, mine_present], True),
        [100.0, 3.0]
    )
    direct_mine = vl(
        concat([pcoords, directions], True),
        [[0.0, 1.0, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, -6.0 / 63.0, 6.0 / 63.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]],
        [0.0, 0.0, 1.0]
    )
    future_mine = at_payload(shift(preparations, "G"))
    mine_command = vmix(
        direct_mine, future_mine, payload_flag(future_mine)
    )
    mine_candidate = click_candidate(mine_score, mine_command)

    reverse_score = vl(
        concat([at_direction(reverse_value), red_exists], True),
        [100.0, 5.0]
    )
    direct_reverse = vl(
        pcoords,
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]],
        [0.0, 0.0, 3.0]
    )
    payload_reverse_matrix = np.zeros((18, 18), np.float32)
    for j in range(3):
        payload_reverse_matrix[
            6 * j:6 * j + 6, 6 * j:6 * j + 6
        ] = reverse_matrix
    future_reverse = at_payload(
        linear(preparations, payload_reverse_matrix)
    )
    reverse_command = vmix(
        direct_reverse, future_reverse,
        payload_flag(future_reverse)
    )
    reverse_candidate = click_candidate(reverse_score, reverse_command)

    def choose(a, b):
        difference = vsub(b, a)
        difference_score = take(difference, 0, 1, True)
        gate = clip(va(difference_score, 64.0, -0.5), vector=True)
        return binary(
            "vadd", a, binary("vmul", difference, gate)
        )

    winner = choose(left_candidate, right_candidate)
    winner = choose(winner, mine_candidate)
    winner = choose(winner, reverse_candidate)

    wo = np.zeros((8, 7), np.float32)
    bo = np.full(8, -10.0, np.float32)
    wo[3, 1] = 20.0
    wo[4, 2] = 20.0
    wo[6, 3] = 20.0
    vl(winner, wo, bo, name="logits")

    # Click kinds: direct tile, white-growth preparation, gravity switch.
    tx = take(winner, 4, 5, True)
    ty = take(winner, 5, 6, True)
    kind = take(winner, 6, 7, True)

    distance = binary(
        "add",
        unary("abs", binary("sub", col, tx)),
        unary("abs", binary("sub", row, ty))
    )
    distance = affine(distance, 63.0)
    generic_click = affine(distance, -1.0)

    point = clip(
        affine(relu(affine(distance, -4.0, 1.0)), 2.0)
    )
    point = binary("mul", point, lattice)
    trace = wave(affine(point, 40.0), growth_allowed)
    growth_click = binary(
        "add",
        binary(
            "add", affine(grow_white, 2000.0), affine(trace, 10.0)
        ),
        generic_click
    )

    player_x = take(pcoords, 1, 2, True)
    player_y = take(pcoords, 0, 1, True)
    red_distance = binary(
        "add",
        unary("abs", binary("sub", col, player_x)),
        unary("abs", binary("sub", row, player_y))
    )
    red_click = binary(
        "add", affine(red, 2000.0), affine(red_distance, -63.0)
    )

    non_generic = clip(va(kind, 1.0, -1.0), vector=True)
    red_flag = clip(va(kind, 1.0, -2.0), vector=True)
    growth_flag = vsub(non_generic, red_flag)
    generic_flag = va(non_generic, -1.0, 1.0)

    click_a = binary("mul", generic_click, generic_flag)
    click_b = binary("mul", growth_click, growth_flag)
    click_c = binary("mul", red_click, red_flag)
    binary(
        "add", binary("add", click_a, click_b), click_c,
        name="click"
    )

    return {
        "graph": graph,
        "memory_size": 2,
        "params": params,
    }
