import numpy as np


def build():
    P = {}
    cache = {}
    graph = []
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (96,),
        "aprev": (8,),
        "avail": (8,),
    }
    serial = [0]
    target = [graph]

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p%d" % len(P)
            cache[key] = name
            P[name] = a.copy()
        return cache[key]

    def node(op, ins=None, name=None, shape=None, **kw):
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        d = {"id": name, "op": op}
        if ins is not None:
            d["in"] = list(ins)
        d.update(kw)
        target[0].append(d)
        if shape is not None:
            shapes[name] = shape
        return name

    def conv(x, w, b=None, name=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("conv", [x], name, (w.shape[0], 64, 64), **kw)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("dense", [x], name, (w.shape[0],), **kw)

    def unary(op, x, name=None):
        return node(op, [x], name, shapes[x])

    def binary(op, a, b, name=None):
        ca, cb = shapes[a][0], shapes[b][0]
        spatial = len(shapes[a]) == 3 or len(shapes[b]) == 3
        shape = (max(ca, cb), 64, 64) if spatial else (max(ca, cb),)
        return node(op, [a, b], name, shape)

    def add(a, b, name=None):
        return binary("add", a, b, name)

    def sub(a, b, name=None):
        return binary("sub", a, b, name)

    def mul(a, b, name=None):
        return binary("mul", a, b, name)

    def vmax(a, b, name=None):
        return binary("max2", a, b, name)

    def relu(x):
        return unary("relu", x)

    def vadd(a, b, name=None):
        return binary("vadd", a, b, name)

    def vmul(a, b, name=None):
        return binary("vmul", a, b, name)

    def cat(xs, name=None):
        return node(
            "concat", xs, name,
            (sum(shapes[x][0] for x in xs), 64, 64)
        )

    def vcat(xs, name=None):
        return node(
            "vconcat", xs, name,
            (sum(shapes[x][0] for x in xs),)
        )

    def sl(x, a, b=None):
        if b is None:
            b = a + 1
        return node("slice", [x], shape=(b - a, 64, 64),
                    start=a, stop=b)

    def vs(x, a, b=None):
        if b is None:
            b = a + 1
        return node("vslice", [x], shape=(b - a,),
                    start=a, stop=b)

    def affine(x, a=1.0, b=None, name=None):
        c = shapes[x][0]
        w = (np.eye(c, dtype=np.float32) * a).reshape(c, c, 1, 1)
        bias = None if b is None else np.full(c, b, np.float32)
        return conv(x, w, bias, name)

    def va(x, a=1.0, b=None, name=None):
        c = shapes[x][0]
        w = np.eye(c, dtype=np.float32) * a
        bias = None if b is None else np.full(c, b, np.float32)
        return dense(x, w, bias, name)

    def maximum(xs):
        z = xs[0]
        for x in xs[1:]:
            z = vmax(z, x)
        return z

    def total(xs):
        z = xs[0]
        for x in xs[1:]:
            z = add(z, x)
        return z

    def vtotal(xs):
        z = xs[0]
        for x in xs[1:]:
            z = vadd(z, x)
        return z

    def gmax(x):
        return node("gmax", [x], shape=(shapes[x][0],))

    def gsum(x):
        z = node("gmean", [x], shape=(shapes[x][0],))
        return va(z, 4096.0)

    def reduce_channels(x, weights=None):
        c = shapes[x][0]
        if weights is None:
            weights = np.ones(c, np.float32)
        w = np.asarray(weights, np.float32).reshape(1, c, 1, 1)
        return conv(x, w)

    def channel_max(x):
        return maximum([sl(x, j) for j in range(shapes[x][0])])

    # Pull a feature from a neighbouring cell. Cell pitch is six pixels.
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def pull(x, d, distance):
        dy, dx = directions[d]
        c = shapes[x][0]
        z = x
        remaining = distance
        while remaining:
            step = min(3, remaining)
            k = 2 * step + 1
            w = np.zeros((c, c, k, k), np.float32)
            for j in range(c):
                w[j, j, step + dy * step, step + dx * step] = 1
            z = conv(z, w)
            remaining -= step
        return z

    def push(x, d, distance):
        return pull(x, (d + 2) % 4, distance)

    row = sl("coords", 0)
    col = sl("coords", 1)
    one = sl("coords", 2)
    zero = affine(one, 0)
    one_v = gmax(one)

    def const_v(value):
        return va(one_v, 0, value)

    def lift(v):
        return add(zero, v)

    def clamp01(x):
        return binary("min2", relu(x), one)

    def valid_score(value, valid):
        # Invalid alternatives receive -100.
        return affine(mul(add(value, affine(one, 100)), valid), 1, -100)

    # ------------------------------------------------------------------
    # Canonical 6x6 sprite detectors, anchored at local pixel (2,2).
    # Outputs: green, red, blue, failed-grey, destination ring, selection,
    # purple block, empty white tile, empty gold tile, gold tile, rail ink.
    # ------------------------------------------------------------------
    W = np.zeros((11, 16, 7, 7), np.float32)
    B = np.zeros(11, np.float32)

    def put(o, color, y, x, value=1):
        W[o, color, y + 3, x + 3] += value

    pellet = [
        (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1), (0, 2),
        (1, -1), (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
    ]
    corners = [(-1, -1), (-1, 2), (2, -1), (2, 2)]
    for o, color in enumerate([14, 8, 9, 2]):
        for y, x in pellet:
            put(o, color, y, x)
        for y, x in corners:
            put(o, color, y, x, -12)
        B[o] = -11

    ring_points = [
        (-1, 0), (-1, 1), (0, -1), (0, 2),
        (1, -1), (1, 2), (2, 0), (2, 1),
    ]
    for y, x in ring_points:
        put(4, 2, y, x)
    for y in [0, 1]:
        for x in [0, 1]:
            put(4, 2, y, x, -8)
    B[4] = -7

    selection_pattern = [
        " xxxx ",
        "xx  xx",
        "x    x",
        "x    x",
        "xx  xx",
        " xxxx ",
    ]
    n = 0
    for y, line in enumerate(selection_pattern):
        for x, ch in enumerate(line):
            if ch == "x":
                put(5, 3, y - 2, x - 2)
                n += 1
    B[5] = 1 - n

    # Purple square perimeter.
    n = 0
    for y in range(4):
        for x in range(1, 5):
            if y in (0, 3) or x in (1, 4):
                put(6, 15, y - 2, x - 2)
                n += 1
    B[6] = 1 - n

    for o, border, inside in [(7, 0, 1), (8, 11, 12)]:
        for y in range(6):
            for x in range(6):
                if y in (0, 5) or x in (0, 5):
                    put(o, border, y - 2, x - 2)
                else:
                    put(o, inside, y - 2, x - 2)
                    put(o, 2, y - 2, x - 2)
        B[o] = -35

    for y, x in [(-2, -2), (-2, 3), (3, -2), (3, 3)]:
        put(9, 11, y, x)
    B[9] = -3

    for y in (0, 1):
        for x in (0, 1):
            put(10, 5, y, x)
    B[10] = -3

    detected = relu(conv("x0", W, B))
    G, R, BLUE, GREY, RING, SELECT, BLOCK, EMPTY, EGOLD, GOLD, INK = [
        sl(detected, j) for j in range(11)
    ]
    colors = cat([G, R])
    occupied = total([G, R, BLUE, BLOCK])
    on_g = mul(GOLD, G)
    on_r = mul(GOLD, R)
    on_block = mul(GOLD, BLOCK)

    # ------------------------------------------------------------------
    # Recover the visible six-pixel lattice from recognised sprites.
    # The two +/-3 convolutions propagate by +/-6 without changing phase.
    # ------------------------------------------------------------------
    seed = clamp01(total([G, R, BLUE, GREY, BLOCK, EMPTY, EGOLD, GOLD]))
    row_seed = node("rowmax", [seed], shape=shapes[seed])
    col_seed = node("colmax", [seed], shape=shapes[seed])

    lattice_body = []
    target[0] = lattice_body
    shapes["lat_rows"] = (1, 64, 64)
    shapes["lat_cols"] = (1, 64, 64)

    def spread_axis(x, vertical):
        w = np.zeros((1, 1, 7, 7), np.float32)
        if vertical:
            w[0, 0, 0, 3] = 1
            w[0, 0, 6, 3] = 1
        else:
            w[0, 0, 3, 0] = 1
            w[0, 0, 3, 6] = 1
        return clamp01(add(x, conv(conv(x, w), w)))

    next_rows = spread_axis("lat_rows", True)
    next_cols = spread_axis("lat_cols", False)
    target[0] = graph
    node(
        "loop", name="lattice_loop", T=11,
        state=[
            {"id": "lat_rows", "init": row_seed},
            {"id": "lat_cols", "init": col_seed},
        ],
        body=lattice_body,
        next={"lat_rows": next_rows, "lat_cols": next_cols},
    )
    shapes["lattice_loop.lat_rows"] = (1, 64, 64)
    shapes["lattice_loop.lat_cols"] = (1, 64, 64)
    lattice = mul("lattice_loop.lat_rows", "lattice_loop.lat_cols")
    track = clamp01(add(mul(lattice, INK), GOLD))
    bare = relu(sub(track, GOLD))
    empty_actual = clamp01(add(EMPTY, EGOLD))

    # ------------------------------------------------------------------
    # Recurrent tabu memory: twelve records, stored in eight field-major
    # groups: four state moments, click row/column, action id, valid flag.
    # ------------------------------------------------------------------
    H = 12
    history = [vs("mem", j * H, (j + 1) * H) for j in range(8)]
    fingerprint = reduce_channels(
        cat([G, R, BLUE, BLOCK, GOLD]), [1, 3, 7, 11, 17]
    )
    moment2 = total([mul(row, row), mul(col, col), mul(row, col)])
    hashes = [
        gsum(fingerprint),
        gsum(mul(fingerprint, row)),
        gsum(mul(fingerprint, col)),
        gsum(mul(fingerprint, moment2)),
    ]
    differences = []
    for j in range(4):
        broadcast = dense(hashes[j], np.ones((H, 1), np.float32))
        difference = vadd(broadcast, va(history[j], -1))
        differences.append(vmul(difference, difference))
    state_match = unary(
        "vrelu", va(vtotal(differences), -10000, 1)
    )
    state_match = vmul(state_match, history[7])

    def action_equal(a):
        d = va(history[6], 1, -a)
        absolute = vadd(unary("vrelu", d),
                        unary("vrelu", va(d, -1)))
        return unary("vrelu", va(absolute, -2, 1))

    action_matches = {a: action_equal(a) for a in [1, 2, 3, 4, 6]}
    click_history = vmul(action_matches[6], history[7])
    tabu_click_history = vmul(action_matches[6], state_match)

    dr = unary("abs", sub(row, history[4]))
    dc = unary("abs", sub(col, history[5]))
    same_pixel = relu(affine(add(dr, dc), -1000, 1))
    tabu_click = reduce_channels(mul(same_pixel, tabu_click_history))
    recent_click = reduce_channels(mul(same_pixel, click_history))
    click_penalty = add(
        affine(tabu_click, 20),
        affine(recent_click, 0.075),
    )
    arrow_tabu = {}
    for a in [1, 2, 3, 4]:
        arrow_tabu[a] = dense(
            vmul(action_matches[a], state_match),
            np.ones((1, H), np.float32) * 20
        )

    # ------------------------------------------------------------------
    # Static jump features. Blue pieces and purple blocks are reusable
    # stepping stones. Only matching green/red pieces yield capture reward.
    # Bare rail can be a potential landing if an empty platform exists.
    # ------------------------------------------------------------------
    has_empty = gmax(EGOLD)
    has_block = gmax(on_block)
    potential_landing = clamp01(add(empty_actual, mul(bare, has_empty)))
    virtual_block = mul(bare, has_block)
    stationary_colors = cat([
        relu(sub(G, on_g)),
        relu(sub(R, on_r)),
    ])
    plan_obstacles = relu(sub(sub(occupied, on_g), on_r))
    potential_middle = clamp01(add(plan_obstacles, virtual_block))

    actual_middle = []
    actual_capture = []
    actual_destination = []
    potential_destination = []
    plan_capture = []
    plan_valid = []
    virtual_middle_at_source = []
    bare_destination = []
    sacrifice_parts = []

    for d in range(4):
        am = pull(occupied, d, 6)
        ac = pull(colors, d, 6)
        ad = pull(empty_actual, d, 12)
        pd = pull(potential_landing, d, 12)
        pc = pull(stationary_colors, d, 6)
        pm = pull(potential_middle, d, 6)
        vm = pull(virtual_block, d, 6)
        bd = pull(bare, d, 12)
        opposite_empty = pull(potential_landing, (d + 2) % 4, 6)

        actual_middle.append(am)
        actual_capture.append(ac)
        actual_destination.append(ad)
        potential_destination.append(pd)
        plan_capture.append(pc)
        plan_valid.append(mul(pm, pd))
        virtual_middle_at_source.append(vm)
        bare_destination.append(bd)
        sacrifice_parts.append(mul(pc, opposite_empty))

    # Small exploration value for reaching ordinary floor, biased gently
    # towards the right side of the visible board.
    floor_value = mul(
        EMPTY,
        total([affine(one, 0.40), affine(col, 0.30), affine(row, 0.025)])
    )
    terminal = []
    for d in range(4):
        incoming = maximum([
            sacrifice_parts[j] for j in range(4)
            if j != (d + 2) % 4
        ])
        terminal.append(vmax(floor_value, affine(incoming, 0.90)))
    sacrifice_all = affine(maximum(sacrifice_parts), 0.90)

    right_edge = clamp01(affine(col, 63, -57))
    frontier = affine(mul(track, right_edge), 0.40)
    passenger_rail = relu(sub(sub(track, BLOCK), BLUE))

    # ------------------------------------------------------------------
    # Non-backtracking jump-value iteration, coupled to rail-value
    # propagation. Each Q state has green and red channels.
    # ------------------------------------------------------------------
    value_body = []
    target[0] = value_body
    qstates = ["jump_q%d" % d for d in range(4)]
    for s in qstates + ["passenger_v"]:
        shapes[s] = (2, 64, 64)

    def future_value(qs, rv, d):
        return maximum(
            [qs[j] for j in range(4) if j != (d + 2) % 4]
            + [rv, terminal[d]]
        )

    qnext = []
    for d in range(4):
        future = pull(future_value(qstates, "passenger_v", d), d, 12)
        cost = total([
            affine(one, 0.045),
            affine(virtual_middle_at_source[d], 0.08),
            affine(bare_destination[d], 0.045),
        ])
        value = sub(
            add(plan_capture[d], affine(future, 0.92)),
            cost,
        )
        qnext.append(mul(relu(value), plan_valid[d]))

    rail_options = [maximum(qstates), sacrifice_all, frontier]
    for d in range(4):
        neighbour = pull("passenger_v", d, 6)
        rail_options.append(relu(affine(neighbour, 0.995, -0.005)))
    rnext = mul(maximum(rail_options), passenger_rail)

    target[0] = graph
    node(
        "loop", name="value_loop", T=24,
        state=[
            {"id": s, "zeros": [2, 64, 64]}
            for s in qstates + ["passenger_v"]
        ],
        body=value_body,
        next={
            **{qstates[d]: qnext[d] for d in range(4)},
            "passenger_v": rnext,
        },
    )
    qfinal = ["value_loop." + s for s in qstates]
    rfinal = "value_loop.passenger_v"
    for s in qfinal + [rfinal]:
        shapes[s] = (2, 64, 64)

    # ------------------------------------------------------------------
    # Score immediately executable jumps and outstanding platform tasks.
    # Source and destination clicks are combined in the same click map:
    # a selected source emits its score at its destination instead.
    # ------------------------------------------------------------------
    source_options = []
    landing_options = []
    empty_goals = []
    block_goals = []

    for d in range(4):
        future = pull(future_value(qfinal, rfinal, d), d, 12)
        raw_value = affine(
            add(actual_capture[d], affine(future, 0.92)), 1, -0.045
        )

        penalty = pull(click_penalty, d, 12)
        scored = sub(raw_value, penalty)
        valid = mul(
            colors,
            mul(actual_middle[d], actual_destination[d])
        )
        jump_score = channel_max(valid_score(scored, valid))

        source_options.append(sub(jump_score, affine(SELECT, 100)))
        selected_score = affine(
            add(jump_score, affine(SELECT, 100)), 1, -100
        )
        landed = push(selected_score, d, 12)
        landing_options.append(valid_score(landed, RING))

        # Empty-platform task: provide a landing beyond an actual obstacle.
        empty_valid = mul(
            colors,
            mul(actual_middle[d], bare_destination[d])
        )
        empty_value = channel_max(
            mul(relu(raw_value), empty_valid)
        )
        empty_goals.append(push(empty_value, d, 12))

        # Purple-platform task: provide the missing middle of a jump.
        mid_bare = pull(bare, d, 6)
        block_valid = mul(
            colors,
            mul(mid_bare, potential_destination[d])
        )
        block_value = relu(affine(future, 0.92, -0.085))
        block_value = channel_max(mul(block_value, block_valid))
        block_goals.append(push(block_value, d, 6))

    source_click = maximum(source_options)
    destination_click = maximum(landing_options)
    click = vmax(source_click, destination_click, name="click")
    click_score = gmax(click)

    empty_goal = mul(maximum(empty_goals), has_empty)
    block_goal = mul(maximum(block_goals), has_block)
    task_seed = cat([empty_goal, block_goal])

    # ------------------------------------------------------------------
    # Route empty and purple-block platforms to their requested cells.
    # ------------------------------------------------------------------
    task_body = []
    target[0] = task_body
    shapes["task_v"] = (2, 64, 64)
    options = [task_seed]
    for d in range(4):
        options.append(
            relu(affine(pull("task_v", d, 6), 0.995, -0.005))
        )
    task_next = mul(maximum(options), track)
    target[0] = graph
    node(
        "loop", name="task_loop", T=32,
        state=[{"id": "task_v", "init": task_seed}],
        body=task_body,
        next={"task_v": task_next},
    )
    shapes["task_loop.task_v"] = (2, 64, 64)

    flow = cat([
        sl("task_loop.task_v", 0),
        sl("task_loop.task_v", 1),
        sl(rfinal, 0),
        sl(rfinal, 1),
    ])
    carriers = cat([EGOLD, on_block, on_g, on_r])
    arrow_scores_by_direction = []

    # Directions use up, right, down, left here.
    action_for_direction = [1, 4, 2, 3]
    reverse_action = [2, 3, 1, 4]
    exploratory_bonus = [0.000, 0.010, 0.004, 0.001]

    for d in range(4):
        neighbour = pull(flow, d, 6)
        improvement = clamp01(
            affine(sub(neighbour, flow), 1000, -0.10)
        )
        destination_free = pull(bare, d, 6)
        movable = mul(carriers, destination_free)
        route_valid = mul(movable, improvement)
        route_score = gmax(channel_max(valid_score(
            affine(neighbour, 0.995, -0.005), route_valid
        )))

        # Low-priority rail exploration when no computed task is improving.
        fallback_valid = mul(GOLD, destination_free)
        fallback_value = affine(
            one, 0, 0.034 + exploratory_bonus[d]
        )
        fallback_score = gmax(valid_score(fallback_value, fallback_valid))
        arrow_score = gmax(vmax(lift(route_score), lift(fallback_score)))

        action_id = action_for_direction[d]
        arrow_score = vadd(arrow_score, va(arrow_tabu[action_id], -1))
        reverse_w = np.zeros((1, 8), np.float32)
        reverse_w[0, reverse_action[d]] = -0.025
        arrow_score = vadd(arrow_score, dense("aprev", reverse_w))
        arrow_scores_by_direction.append(arrow_score)

    grey_present = gmax(GREY)
    undo_score = va(grey_present, 100, 0.025)

    scores = vcat([
        const_v(-1000),
        arrow_scores_by_direction[0],
        arrow_scores_by_direction[2],
        arrow_scores_by_direction[3],
        arrow_scores_by_direction[1],
        const_v(-1000),
        click_score,
        undo_score,
    ])
    scores = dense(
        scores,
        np.eye(8, dtype=np.float32),
        -np.arange(8, dtype=np.float32) * 1e-5,
    )
    availability = dense(
        "avail",
        np.eye(8, dtype=np.float32) * 1000,
        np.full(8, -1000, np.float32),
    )
    logits = vadd(scores, availability, name="logits")

    # ------------------------------------------------------------------
    # Recover the decoder's chosen action and row-major click maximum
    # using only permitted max reductions and piecewise-linear operations.
    # This is used solely to write the next tabu-memory record.
    # ------------------------------------------------------------------
    scalar_logits = [lift(vs(logits, j)) for j in range(8)]
    max_logit = gmax(maximum(scalar_logits))
    action_delta = vadd(logits, va(max_logit, -1))
    chosen_action = unary("vrelu", va(action_delta, 1e6, 1))

    click_max = gmax(click)
    peaks = relu(affine(sub(click, click_max), 1e9, 1))

    row_rank = sub(affine(peaks, 1000), affine(row, 63))
    row_best = gmax(row_rank)
    row_peaks = relu(affine(sub(row_rank, row_best), 2, 1))

    col_rank = sub(affine(row_peaks, 1000), affine(col, 63))
    col_best = gmax(col_rank)
    winner = relu(affine(sub(col_rank, col_best), 2, 1))

    chosen_row = gmax(mul(winner, row))
    chosen_col = gmax(mul(winner, col))
    clicked_landing = gmax(mul(winner, RING))

    arrow_flag = dense(
        chosen_action,
        np.asarray([[0, 1, 1, 1, 1, 0, 0, 0]], np.float32),
    )
    click_flag = vs(chosen_action, 6)
    record = vadd(arrow_flag, vmul(click_flag, clicked_landing))

    action_code = dense(
        chosen_action,
        np.arange(8, dtype=np.float32).reshape(1, 8),
    )
    new_fields = hashes + [
        chosen_row, chosen_col, action_code, one_v
    ]
    shifted_groups = [
        vcat([new_fields[j], vs(history[j], 0, H - 1)])
        for j in range(8)
    ]
    proposed_memory = vcat(shifted_groups)
    retained = vmul("mem", va(record, -1, 1))
    inserted = vmul(proposed_memory, record)
    vadd(retained, inserted, name="mem_next")

    return {
        "graph": graph,
        "memory_size": 96,
        "params": P,
    }
