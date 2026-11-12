import numpy as np


def build():
    graph, params = [], {}
    H = 24
    HISTORY_END = 5 + 6 * H
    WALL = HISTORY_END + 11
    M = WALL + 11

    channels = {
        "x0": 16, "coords": 3, "aprev": 8, "avail": 8, "mem": M
    }
    spatial = {
        "x0": True, "coords": True, "aprev": False,
        "avail": False, "mem": False
    }
    serial = [0]
    destination = [graph]

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def emit(op, inputs, c, sp, name=None, **fields):
        name = name or fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        destination[0].append(nd)
        channels[name], spatial[name] = c, sp
        return name

    def cat(xs):
        xs = list(xs)
        if len(xs) == 1:
            return xs[0]
        sp = spatial[xs[0]]
        return emit(
            "concat" if sp else "vconcat", xs,
            sum(channels[x] for x in xs), sp
        )

    def linear(xs, w, b=None, name=None):
        if isinstance(xs, str):
            xs = [xs]
        x = cat(xs)
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        sp = spatial[x]
        wn = fresh("p")
        params[wn] = w[:, :, None, None] if sp else w
        fields = {"w": wn}
        if b is not None:
            bn = fresh("p")
            params[bn] = np.broadcast_to(
                np.asarray(b, np.float32), (w.shape[0],)
            ).copy()
            fields["b"] = bn
        return emit(
            "conv" if sp else "dense", [x], w.shape[0],
            sp, name=name, **fields
        )

    constants = {}

    def const(value):
        a = np.asarray(value, np.float32).reshape(-1)
        key = tuple(float(v) for v in a)
        if key not in constants:
            constants[key] = linear(
                "aprev", np.zeros((len(a), 8), np.float32), a
            )
        return constants[key]

    def aff(x, a=1.0, b=0.0):
        if a == 1 and b == 0:
            return x
        c = channels[x]
        if not spatial[x] and c > 8:
            z = x
            if a != 1:
                z = emit("vmul", [z, const(a)], c, False)
            if b != 0:
                z = emit("vadd", [z, const(b)], c, False)
            return z
        return linear(x, np.eye(c, dtype=np.float32) * a, b)

    def add(a, b):
        sp = spatial[a] or spatial[b]
        return emit(
            "add" if sp else "vadd", [a, b],
            max(channels[a], channels[b]), sp
        )

    def mul(a, b):
        sp = spatial[a] or spatial[b]
        return emit(
            "mul" if sp else "vmul", [a, b],
            max(channels[a], channels[b]), sp
        )

    def sub(a, b):
        if spatial[a] or spatial[b]:
            return emit(
                "sub", [a, b], max(channels[a], channels[b]), True
            )
        return add(a, aff(b, -1))

    def relu(x):
        return emit(
            "relu" if spatial[x] else "vrelu",
            [x], channels[x], spatial[x]
        )

    def absolute(x):
        if spatial[x]:
            return emit("abs", [x], channels[x], True)
        eye = np.eye(channels[x], dtype=np.float32)
        h = relu(linear(x, np.concatenate([eye, -eye], axis=0)))
        return linear(h, np.concatenate([eye, eye], axis=1))

    def clip(x):
        if spatial[x]:
            return emit("min2", [relu(x), const(1)], channels[x], True)
        n = channels[x]
        eye = np.eye(n, dtype=np.float32)
        h = relu(linear(
            x, np.concatenate([eye, eye], axis=0),
            np.concatenate([
                np.zeros(n, np.float32), -np.ones(n, np.float32)
            ])
        ))
        return linear(h, np.concatenate([eye, -eye], axis=1))

    def inv(x):
        return aff(x, -1, 1)

    def eq0(x):
        if spatial[x]:
            return clip(aff(absolute(x), -2, 1.5))
        n = channels[x]
        eye = np.eye(n, dtype=np.float32)
        w = np.concatenate([eye, eye, eye, eye], axis=0)
        b = np.repeat(
            np.asarray([0.75, 0.25, -0.25, -0.75], np.float32), n
        )
        h = relu(linear(x, w, b))
        return linear(
            h, np.concatenate([2*eye, -2*eye, -2*eye, 2*eye], axis=1)
        )

    def pool(x, mean=False):
        return emit(
            "gmean" if mean else "gmax", [x], channels[x], False
        )

    def sl(x, start, stop):
        return emit(
            "slice" if spatial[x] else "vslice",
            [x], stop - start, spatial[x], start=start, stop=stop
        )

    def total(x):
        return linear(x, np.ones((1, channels[x]), np.float32))

    def comb(xs, weights, bias=0):
        return linear(xs, np.asarray(weights, np.float32), bias)

    def kernel(x, w, b=None):
        w = np.asarray(w, np.float32)
        wn = fresh("p")
        params[wn] = w
        fields = {"w": wn}
        if b is not None:
            bn = fresh("p")
            params[bn] = np.broadcast_to(
                np.asarray(b, np.float32), (w.shape[0],)
            ).copy()
            fields["b"] = bn
        return emit("conv", [x], w.shape[0], True, **fields)

    def coord_at(mask, coordinate):
        return pool(mul(mask, coordinate))

    def product(*xs):
        z = xs[0]
        for x in xs[1:]:
            z = mul(z, x)
        return z

    def sum_nodes(*xs):
        z = xs[0]
        for x in xs[1:]:
            z = add(z, x)
        return z

    reverse_matrix = np.asarray([
        [0, 1, 0, 0], [1, 0, 0, 0],
        [0, 0, 0, 1], [0, 0, 1, 0]
    ], np.float32)

    # Cooperative-controller memory.
    phase = sl("mem", HISTORY_END, HISTORY_END + 4)
    releasing, assisting, departing, returning = [
        sl(phase, i, i + 1) for i in range(4)
    ]
    normal = inv(clip(total(phase)))
    saved_dirs = sl("mem", HISTORY_END + 4, HISTORY_END + 8)
    saved_colour = sl("mem", HISTORY_END + 8, HISTORY_END + 9)
    saved_xy = sl("mem", HISTORY_END + 9, HISTORY_END + 11)

    # Obstacle-controller memory.
    wall_phase = sl("mem", WALL, WALL + 8)
    (wall_seek, wall_transport, wall_release, wall_push,
     wall_grow, wall_retract, wall_park, wall_final) = [
        sl(wall_phase, i, i + 1) for i in range(8)
    ]
    wall_on = clip(total(wall_phase))
    wall_idle = inv(wall_on)
    probe_colour = sl("mem", WALL + 8, WALL + 9)
    approach_coordinate = sl("mem", WALL + 9, WALL + 10)
    wall_sign = sl("mem", WALL + 10, WALL + 11)

    # Pixel coordinates.
    xy = linear("coords", [[0, 63, 0], [63, 0, 0]])
    X, Y = sl(xy, 0, 1), sl(xy, 1, 2)
    top = clip(aff(Y, -1, 53))
    bottom = clip(aff(Y, 1, -54))

    # Tile perimeter detector; the centre may contain a target mark.
    colours = list(range(6, 16))
    perimeter = [
        (dy, dx)
        for dy in range(-1, 3)
        for dx in range(-1, 3)
        if dy in (-1, 2) or dx in (-1, 2)
    ]
    w = np.zeros((10, 16, 5, 5), np.float32)
    for k, colour in enumerate(colours):
        for dy, dx in perimeter:
            w[k, colour, dy + 2, dx + 2] = 1
    tiles = relu(kernel("x0", w, -(len(perimeter) - 1)))
    occupancy = total(tiles)
    number = linear(tiles, [colours])
    world_occ = mul(occupancy, top)
    world_num = mul(number, top)

    probe_tiles = mul(world_occ, eq0(sub(number, probe_colour)))
    probe_count = aff(pool(probe_tiles, mean=True), 4096)

    # Selected and unselected movable handles.
    w = np.zeros((2, 16, 5, 5), np.float32)
    centre_colours = [c for c in range(16) if c not in (0, 4, 5)]
    for out, ring in enumerate((0, 4)):
        for dy, dx in ((-1, 0), (2, 0), (0, -1), (0, 2)):
            w[out, ring, dy + 2, dx + 2] += 1
        for colour in centre_colours:
            w[out, colour, 2, 2] += 1
            w[out, colour, 3, 3] += 1
    handles = relu(kernel("x0", w, -5))
    selected_all, other_all = sl(handles, 0, 1), sl(handles, 1, 2)
    active = mul(selected_all, top)
    legend = mul(selected_all, bottom)
    inactive = mul(other_all, top)
    multiple = pool(inactive)
    bx, by = coord_at(active, X), coord_at(active, Y)
    lx = coord_at(legend, X)
    base_xy = cat([bx, by])

    # Extension direction: up, down, left, right.
    w = np.zeros((4, 16, 7, 7), np.float32)
    for k, (dy, dx) in enumerate(((-2, 0), (3, 0), (0, -2), (0, 3))):
        for colour in (1, 2, 3):
            w[k, colour, dy + 3, dx + 3] = 1
    dirs = pool(mul(kernel("x0", w), active))
    reverse_dirs = linear(dirs, reverse_matrix)
    horizontal = linear(dirs, [[0, 0, 1, 1]])
    vertical = inv(horizontal)
    dxdy = linear(dirs, [[0, 0, -1, 1], [-1, 1, 0, 0]])
    perp_coeff = cat([vertical, horizontal])
    relative_xy = sub(xy, base_xy)
    u = total(mul(relative_xy, dxdy))
    v = total(mul(relative_xy, perp_coeff))
    base_v = total(mul(base_xy, perp_coeff))
    perp_neg_dirs = linear(
        horizontal, [[1], [0], [-1], [0]], [0, 0, 1, 0]
    )
    perp_pos_dirs = linear(
        horizontal, [[0], [1], [0], [-1]], [0, 0, 0, 1]
    )

    # Board bounds from interior background patches.
    w = np.zeros((1, 16, 3, 3), np.float32)
    w[0, 4] = 1
    floor = mul(relu(kernel("x0", w, -8)), top)
    min_x = aff(pool(mul(floor, aff(X, -1, 63))), -1, 62)
    min_y = aff(pool(mul(floor, aff(Y, -1, 63))), -1, 62)
    max_x = aff(pool(mul(floor, X)), 1, 1)
    max_y = aff(pool(mul(floor, Y)), 1, 1)
    min_v = aff(total(mul(cat([min_x, min_y]), perp_coeff)), 1, 2)
    max_v = aff(total(mul(cat([max_x, max_y]), perp_coeff)), 1, -3)

    # Read the selected target sequence.
    offsets = np.arange(1, 9, dtype=np.float32) * 6
    gx = linear(X, np.ones((8, 1), np.float32), -offsets)
    goal_positions = eq0(sub(gx, lx))
    goal_raw = pool(product(goal_positions, number, bottom))
    goal_present = clip(goal_raw)
    goal_valid = relu(linear(
        goal_present, np.tril(np.ones((8, 8), np.float32)),
        -np.arange(8, dtype=np.float32)
    ))
    base_goal = mul(goal_raw, goal_valid)
    base_goal_count = total(goal_valid)

    # Cooperative extra-colour subgoal.
    append_dist = linear(
        base_goal_count, np.ones((8, 1), np.float32),
        -np.arange(8, dtype=np.float32)
    )
    append_slot = mul(eq0(append_dist), assisting)
    goal = add(base_goal, mul(append_slot, saved_colour))
    goal_count = add(base_goal_count, assisting)

    # The obstacle maneuver gathers a same-colour probe group.
    probe_valid = clip(linear(
        probe_count, np.ones((8, 1), np.float32),
        -np.arange(8, dtype=np.float32)
    ))
    probe_goal = mul(probe_valid, probe_colour)
    goal = add(mul(goal, inv(wall_seek)), mul(probe_goal, wall_seek))
    goal_count = add(
        mul(goal_count, inv(wall_seek)), mul(probe_count, wall_seek)
    )

    # Conveyor extent and occupied cells.
    active_ink = mul(sl("x0", 1, 2), top)
    tip = pool(mul(active_ink, u))
    tip_offset = linear(dirs, [[2, 3, 2, 3]])
    extent = sub(tip, tip_offset)
    on_line = eq0(v)
    in_front = clip(aff(u, 1, -2))
    before_tip = clip(aff(sub(extent, u), 1, 1))
    carried = product(world_occ, on_line, in_front, before_tip)

    # Directional accumulation ranks carried blocks.
    drive = linear(carried, np.ones((4, 1), np.float32))
    body = []
    loop_name, state_name = fresh("rank"), fresh("rs")
    channels[state_name], spatial[state_name] = 4, True
    destination[0] = body
    w = np.zeros((4, 4, 3, 3), np.float32)
    for k, (dy, dx) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        w[k, k, dy + 1, dx + 1] = 1
    accumulated = add(kernel(state_name, w), drive)
    destination[0] = graph
    graph.append({
        "id": loop_name, "op": "loop", "T": 64,
        "state": [{"id": state_name, "zeros": [4, 64, 64]}],
        "body": body, "next": {state_name: accumulated}
    })
    scan = loop_name + "." + state_name
    channels[scan], spatial[scan] = 4, True
    rank = total(mul(scan, dirs))
    carried_count = pool(mul(rank, carried))
    rank_distances = linear(
        rank, np.ones((8, 1), np.float32),
        -np.arange(1, 9, dtype=np.float32)
    )
    expected = total(mul(eq0(rank_distances), goal))
    bad = mul(clip(absolute(sub(number, expected))), carried)
    first_bad_score = pool(mul(aff(rank, -1, 9), bad))
    prefix_limit = aff(first_bad_score, -1, 8)
    prefix = sub(carried_count, relu(sub(carried_count, prefix_limit)))
    complete = mul(
        clip(aff(sub(prefix, goal_count), 2, 1)), clip(goal_count)
    )
    next_dist = linear(
        prefix, np.ones((8, 1), np.float32),
        1 - np.arange(1, 9, dtype=np.float32)
    )
    next_colour = total(mul(eq0(next_dist), goal))

    # Transverse inactive conveyors.
    w = np.zeros((2, 16, 5, 5), np.float32)
    for out, positions in enumerate((
        ((0, -2), (1, -2)), ((-2, 0), (-2, 1))
    )):
        for dy, dx in positions:
            for colour in (2, 3):
                w[out, colour, dy + 2, dx + 2] = 1
    old_belts = relu(kernel("x0", w, -1))
    crossing = total(mul(old_belts, perp_coeff))
    pinned = pool(mul(crossing, carried))

    # Boundary and obstacle legality.
    valid_neg = clip(aff(sub(aff(base_v, 1, -6), min_v), 4, 1))
    valid_pos = clip(aff(sub(aff(max_v, 1, -6), base_v), 4, 1))
    max_u = total(mul(cat([
        aff(sub(by, min_y), 1, -2),
        aff(sub(max_y, by), 1, -3),
        aff(sub(bx, min_x), 1, -2),
        aff(sub(max_x, bx), 1, -3),
    ]), dirs))
    valid_ext = clip(aff(sub(aff(max_u, 1, -6), extent), 4, 1))
    valid_ret = clip(aff(extent, 1.0 / 6))

    rectangle = product(
        clip(aff(sub(X, min_x), 1, 1)),
        clip(aff(sub(max_x, X), 1, 1)),
        clip(aff(sub(Y, min_y), 1, 1)),
        clip(aff(sub(max_y, Y), 1, 1))
    )
    walls = mul(sl("x0", 5, 6), rectangle)
    belt_cover = mul(
        clip(aff(u, 1, -3)), clip(aff(sub(extent, u), 1, 4))
    )
    neg_wall = pool(product(walls, belt_cover, eq0(aff(v, 1, 6))))
    pos_wall = pool(product(walls, belt_cover, eq0(aff(v, 1, -6))))
    valid_neg = mul(valid_neg, inv(neg_wall))
    valid_pos = mul(valid_pos, inv(pos_wall))
    next_cover = mul(
        clip(aff(sub(u, extent), 1, -3)),
        clip(aff(sub(extent, u), 1, 10))
    )
    valid_ext = mul(
        valid_ext, inv(pool(product(walls, next_cover, on_line)))
    )

    # Shadow of each obstacle along the extension axis.
    wall_strength = product(walls, in_front, aff(u, -1, 64))
    row_wall = emit("rowmax", [wall_strength], 1, True)
    col_wall = emit("colmax", [wall_strength], 1, True)
    near_wall_score = add(
        mul(row_wall, horizontal), mul(col_wall, vertical)
    )
    shadow = clip(aff(sub(u, aff(near_wall_score, -1, 64)), 1, 1))

    # Push-chain immovability.
    shift_w = np.zeros((4, 4, 7, 7), np.float32)
    for k, (dy, dx) in enumerate(((-3, 0), (3, 0), (0, -3), (0, 3))):
        shift_w[k, k, dy + 3, dx + 3] = 1
    shift_param = fresh("p")
    params[shift_param] = shift_w

    def shift_six(x):
        a = emit("conv", [x], 4, True, w=shift_param)
        return emit("conv", [a], 4, True, w=shift_param)

    out_bounds = cat([
        clip(aff(sub(min_y, Y), 1, 8)),
        clip(aff(sub(Y, max_y), 1, 9)),
        clip(aff(sub(min_x, X), 1, 8)),
        clip(aff(sub(X, max_x), 1, 9)),
    ])
    wall_four = linear(walls, np.ones((4, 1), np.float32))
    fixed_block = clip(add(out_bounds, shift_six(wall_four)))
    block_init = mul(fixed_block, world_occ)
    body = []
    block_loop, block_state = fresh("push"), fresh("ps")
    channels[block_state], spatial[block_state] = 4, True
    destination[0] = body
    block_next = mul(
        clip(add(fixed_block, shift_six(block_state))), world_occ
    )
    destination[0] = graph
    graph.append({
        "id": block_loop, "op": "loop", "T": 8,
        "state": [{"id": block_state, "init": block_init}],
        "body": body, "next": {block_state: block_next}
    })
    blocked = block_loop + "." + block_state
    channels[blocked], spatial[blocked] = 4, True
    encountered = clip(add(blocked, shift_six(blocked)))
    belt_centres = product(
        on_line, in_front, clip(aff(sub(extent, u), 1, 1))
    )
    physically_valid = inv(clip(pool(mul(encountered, belt_centres))))
    valid_neg = mul(
        valid_neg, total(mul(physically_valid, perp_neg_dirs))
    )
    valid_pos = mul(
        valid_pos, total(mul(physically_valid, perp_pos_dirs))
    )

    # Next-colour candidates, preferring accessible blocks.
    unheld = sub(world_occ, carried)
    desired = mul(eq0(sub(number, next_colour)), unheld)
    too_close = clip(aff(sub(aff(prefix, 6), u), 1, 1))
    request_distance = total(absolute(sub(xy, saved_xy)))
    candidate_value = comb(
        [absolute(v), u, too_close, X, Y,
         mul(request_distance, assisting), shadow],
        [-1, -0.015, -35, -0.001, -0.00002, -0.5, -90], 200
    )
    candidate_score = mul(relu(candidate_value), desired)
    score_same = pool(mul(candidate_score, on_line))
    score_neg = aff(
        pool(mul(candidate_score, clip(aff(v, -1)))), 1, 0.2
    )
    score_pos = aff(pool(mul(candidate_score, clip(v))), 1, 0.1)
    best_candidate = pool(candidate_score)
    chosen = mul(
        clip(aff(absolute(sub(candidate_score, best_candidate)),
                 -32768, 1)),
        desired
    )
    target_exists = clip(pool(chosen))
    target_u = aff(pool(mul(chosen, aff(u, 1, 64))), 1, -64)
    target_v = aff(pool(mul(chosen, aff(v, 1, 64))), 1, -64)
    target_xy = pool(mul(chosen, xy))
    target_neg = mul(clip(aff(target_v, -1)), target_exists)
    target_pos = mul(clip(target_v), target_exists)
    target_offline = clip(add(target_neg, target_pos))

    # Prefix compaction.
    initialized = sl("mem", 0, 1)
    old_mode = sl("mem", 1, 2)
    old_prefix = sl("mem", 2, 3)
    previous_click = sl("aprev", 6, 7)
    grew = clip(aff(sub(prefix, old_prefix), 2, -0.5))
    wrong = clip(aff(sub(carried_count, prefix), 2, -0.5))
    initial_or_switch = clip(add(inv(initialized), previous_click))
    initial_loaded = mul(initial_or_switch, clip(carried_count))
    compact_trigger = clip(sum_nodes(grew, wrong, initial_loaded))
    wants_compact = clip(add(old_mode, compact_trigger))
    needs_retraction = clip(aff(sub(extent, aff(prefix, 6)), 2, -0.5))
    completion_allows = clip(sum_nodes(inv(complete), assisting, wall_seek))
    useful_compaction = clip(sum_nodes(inv(multiple), wrong, assisting))
    mode = product(
        wants_compact, needs_retraction, completion_allows,
        inv(mul(pinned, multiple)), useful_compaction
    )
    can_perp = clip(add(valid_neg, valid_pos))
    preparation = product(
        wrong, inv(old_mode), inv(pinned), mode, can_perp, inv(wall_seek)
    )
    prep_neg = valid_neg
    prep_pos = mul(inv(valid_neg), valid_pos)
    escape_pos = valid_pos
    escape_neg = mul(inv(valid_pos), valid_neg)
    compact_neg = add(
        mul(preparation, prep_neg),
        product(inv(preparation), pinned, escape_neg)
    )
    compact_pos = add(
        mul(preparation, prep_pos),
        product(inv(preparation), pinned, escape_pos)
    )
    retract_choice = product(mode, inv(pinned), inv(preparation))
    gather = inv(mode)
    neg_score = add(
        mul(gather, score_neg), aff(mul(mode, compact_neg), 400)
    )
    pos_score = add(
        mul(gather, score_pos), aff(mul(mode, compact_pos), 400)
    )
    ret_score = aff(retract_choice, 400)
    ext_score = mul(gather, score_same)

    # Approach-length adjustment for multiple movable conveyors.
    minimum_extent = aff(prefix, 6)
    requested_extent = add(
        minimum_extent, relu(sub(aff(target_u, 1, -6), minimum_extent))
    )
    approaching = product(multiple, gather, target_offline)
    approach_extend = product(
        approaching, clip(aff(sub(requested_extent, extent), 1/6)),
        valid_ext
    )
    approach_retract = product(
        approaching, clip(aff(sub(extent, requested_extent), 1/6)),
        valid_ret, inv(pinned)
    )
    approach_change = clip(add(approach_extend, approach_retract))
    neg_score = mul(neg_score, inv(approach_change))
    pos_score = mul(pos_score, inv(approach_change))
    ext_score = add(
        mul(ext_score, inv(approach_change)), aff(approach_extend, 450)
    )
    ret_score = add(
        mul(ret_score, inv(approach_change)), aff(approach_retract, 450)
    )

    # Cooperative relocation request.
    blocked_target_side = clip(add(
        mul(target_neg, inv(valid_neg)), mul(target_pos, inv(valid_pos))
    ))
    request = product(
        normal, wall_idle, multiple, pinned, clip(prefix),
        blocked_target_side, inv(complete), valid_ret
    )
    release_more = mul(releasing, pinned)
    switch_helper = mul(releasing, inv(pinned))
    helper_ready = product(assisting, complete, inv(needs_retraction))

    raw_scores = sum_nodes(
        mul(dirs, ext_score), mul(reverse_dirs, ret_score),
        mul(perp_neg_dirs, neg_score), mul(perp_pos_dirs, pos_score)
    )
    valid_dirs = sum_nodes(
        mul(dirs, valid_ext), mul(reverse_dirs, valid_ret),
        mul(perp_neg_dirs, valid_neg), mul(perp_pos_dirs, valid_pos)
    )

    # Accessible, non-target-colour probe group.
    probe_candidates = product(
        world_occ, inv(eq0(sub(number, next_colour))), inv(shadow)
    )
    probe_quality = comb(
        [absolute(u), absolute(v), X, Y],
        [-0.5, -0.5, -0.001, -0.00002], 200
    )
    probe_scores = mul(relu(probe_quality), probe_candidates)
    probe_best = pool(probe_scores)
    probe_choice = mul(
        clip(aff(absolute(sub(probe_scores, probe_best)), -32768, 1)),
        probe_candidates
    )
    new_probe_colour = pool(mul(probe_choice, number))
    target_shadow = pool(mul(chosen, shadow))
    wall_begin = product(
        normal, wall_idle, inv(multiple), target_shadow,
        clip(new_probe_colour), inv(complete)
    )

    target_abs_v = add(base_v, target_v)
    room_before = clip(aff(sub(target_abs_v, min_v), 1, -11))
    new_wall_sign = aff(room_before, 2, -1)
    new_approach = sub(target_abs_v, aff(new_wall_sign, 6))

    # Obstacle-controller transitions.
    source_delta = sub(approach_coordinate, base_v)
    at_source = eq0(source_delta)
    probe_ready = product(wall_seek, complete, inv(needs_retraction))
    transport_active = clip(add(probe_ready, wall_transport))
    transport_neg = product(
        transport_active, clip(aff(source_delta, -1/6))
    )
    transport_pos = product(
        transport_active, clip(aff(source_delta, 1/6))
    )
    ready_at_source = mul(transport_active, at_source)

    last_retract = clip(aff(extent, -2, 13))
    release_active = clip(add(ready_at_source, wall_release))
    release_finished = mul(release_active, last_retract)

    row_probe_count = aff(pool(mul(probe_tiles, on_line), mean=True), 4096)
    group_tail_offset = aff(relu(aff(row_probe_count, 1, -2)), 6, 6)
    push_limit = sub(max_u, group_tail_offset)
    push_at_limit = clip(aff(sub(extent, push_limit), 2, 1))
    push_finished = mul(wall_push, push_at_limit)
    push_extend = mul(wall_push, inv(push_at_limit))

    toward_positive = aff(wall_sign, 0.5, 0.5)
    toward_wall_dirs = add(
        mul(perp_pos_dirs, toward_positive),
        mul(perp_neg_dirs, inv(toward_positive))
    )
    away_wall_dirs = linear(toward_wall_dirs, reverse_matrix)

    full_extent = clip(aff(sub(extent, max_u), 2, 1))
    shove = mul(wall_grow, full_extent)
    grow_extend = mul(wall_grow, inv(full_extent))
    parking_start = product(
        wall_retract, last_retract, clip(carried_count)
    )
    park_at_limit = clip(aff(sub(extent, aff(max_u, 1, -6)), 2, 1))
    park_finished = mul(wall_park, park_at_limit)
    park_extend = mul(wall_park, inv(park_at_limit))

    wall_force_ret = clip(sum_nodes(
        release_active, wall_retract, park_finished, wall_final
    ))
    wall_force_ext = clip(sum_nodes(push_extend, grow_extend, park_extend))
    wall_direction_override = sum_nodes(
        mul(reverse_dirs, wall_force_ret),
        mul(dirs, wall_force_ext),
        mul(perp_neg_dirs, transport_neg),
        mul(perp_pos_dirs, transport_pos),
        mul(away_wall_dirs, push_finished),
        mul(toward_wall_dirs, shove)
    )
    next_wall_phase = cat([
        clip(add(wall_begin, mul(wall_seek, inv(probe_ready)))),
        mul(transport_active, inv(at_source)),
        mul(release_active, inv(last_retract)),
        clip(add(release_finished, push_extend)),
        clip(add(push_finished, grow_extend)),
        clip(add(shove, mul(wall_retract, inv(last_retract)))),
        clip(add(parking_start, park_extend)),
        clip(add(park_finished, mul(wall_final, inv(last_retract)))),
    ])

    # Recent state-action fingerprints.
    hash_field_1 = comb([X, Y], [1, 67], 1)
    hash_1_world = aff(
        pool(mul(world_num, hash_field_1), mean=True), 4096/1000
    )
    hash_1 = comb(
        [hash_1_world, bx, by, extent, horizontal],
        [1, 0.013, 0.117, 1.731, 17.13]
    )
    hash_field_2 = comb(
        [mul(X, X), mul(Y, Y), mul(X, Y)], [1, 5, 7], 11
    )
    hash_2_world = aff(
        pool(mul(world_num, hash_field_2), mean=True), 4096/10000
    )
    hash_2 = comb(
        [hash_2_world, bx, by, extent], [1, 0.071, 0.037, 0.193]
    )
    hist_1 = sl("mem", 5, 5 + H)
    hist_2 = sl("mem", 5 + H, 5 + 2 * H)
    hist_actions = sl("mem", 5 + 2 * H, HISTORY_END)
    compare_w = np.zeros((H, H + 1), np.float32)
    compare_w[:, 0] = 1
    compare_w[np.arange(H), np.arange(H) + 1] = -1
    difference_1 = linear([hash_1, hist_1], compare_w)
    difference_2 = linear([hash_2, hist_2], compare_w)
    same_history = mul(
        clip(aff(absolute(difference_1), -1000, 1)),
        clip(aff(absolute(difference_2), -1000, 1))
    )
    aligned_actions = cat([
        sl("aprev", 1, 5), sl(hist_actions, 0, 4 * (H - 1))
    ])
    repeat_w = np.zeros((4 * H, H), np.float32)
    for i in range(H):
        repeat_w[4*i:4*i+4, i] = 1
    matched_actions = mul(linear(same_history, repeat_w), aligned_actions)
    count_w = np.zeros((4, 4 * H), np.float32)
    for i in range(H):
        count_w[:, 4*i:4*i+4] = np.eye(4, dtype=np.float32)
    visit_counts = linear(matched_actions, count_w)
    direction_scores = sum_nodes(
        raw_scores, aff(valid_dirs, 5000, -5000),
        aff(visit_counts, -500)
    )

    # Cooperative and obstacle action overrides.
    forced_retract = clip(sum_nodes(request, release_more, helper_ready))
    opposite_saved = linear(saved_dirs, reverse_matrix)
    departure_ok = clip(total(mul(opposite_saved, valid_dirs)))
    departure_dirs = add(
        mul(opposite_saved, departure_ok),
        mul(saved_dirs, inv(departure_ok))
    )
    direction_scores = add(
        direction_scores,
        aff(add(
            mul(reverse_dirs, forced_retract),
            mul(departure_dirs, departing)
        ), 12000)
    )
    direction_scores = add(
        direction_scores, aff(wall_direction_override, 25000)
    )

    ordinary_click = product(complete, multiple, normal, wall_idle)
    forced_click = clip(add(switch_helper, returning))
    click_score = add(
        aff(ordinary_click, 3000, -1000), aff(forced_click, 15000)
    )
    output_w = np.zeros((8, 5), np.float32)
    output_w[1:5, :4] = np.eye(4, dtype=np.float32)
    output_w[6, 4] = 1
    output_b = np.asarray(
        [-20000, 0, 0, 0, 0, -20000, 0, -15000], np.float32
    )
    linear(
        [direction_scores, click_score],
        output_w, output_b, name="logits"
    )
    emit("slice", [inactive], 1, True, name="click", start=0, stop=1)

    next_phase = cat([
        clip(add(request, release_more)),
        clip(add(switch_helper, mul(assisting, inv(helper_ready)))),
        helper_ready, departing,
    ])
    next_saved_dirs = add(
        mul(saved_dirs, inv(request)), mul(dirs, request)
    )
    next_saved_colour = add(
        mul(saved_colour, inv(request)), mul(next_colour, request)
    )
    next_saved_xy = add(
        mul(saved_xy, inv(request)), mul(target_xy, request)
    )
    clear_mode = clip(sum_nodes(
        forced_retract, forced_click, departing, wall_begin,
        sub(wall_on, wall_seek), ready_at_source
    ))
    next_mode = mul(mode, inv(clear_mode))
    next_probe_colour = add(
        mul(probe_colour, inv(wall_begin)),
        mul(new_probe_colour, wall_begin)
    )
    next_approach = add(
        mul(approach_coordinate, inv(wall_begin)),
        mul(new_approach, wall_begin)
    )
    next_wall_sign = add(
        mul(wall_sign, inv(wall_begin)), mul(new_wall_sign, wall_begin)
    )
    emit("vconcat", [
        const(1), next_mode, prefix, extent, base_v,
        cat([hash_1, sl(hist_1, 0, H - 1)]),
        cat([hash_2, sl(hist_2, 0, H - 1)]),
        aligned_actions,
        next_phase, next_saved_dirs, next_saved_colour, next_saved_xy,
        next_wall_phase, next_probe_colour, next_approach, next_wall_sign
    ], M, False, name="mem_next")

    # Data-independent expression sharing and dead-code elimination.
    canonical_params, parameter_alias, seen_arrays = {}, {}, {}
    for name, value in params.items():
        value = np.ascontiguousarray(value, dtype=np.float32)
        key = (value.shape, value.tobytes())
        if key in seen_arrays:
            parameter_alias[name] = seen_arrays[key]
        else:
            seen_arrays[key] = name
            parameter_alias[name] = name
            canonical_params[name] = value

    protected = {"logits", "click", "mem_next"}

    def simplify(nodes, aliases=None, expressions=None):
        aliases = {} if aliases is None else dict(aliases)
        expressions = {} if expressions is None else dict(expressions)
        result = []
        for old in nodes:
            nd = dict(old)
            if nd["op"] == "loop":
                states = []
                for state in nd["state"]:
                    state = dict(state)
                    if "init" in state:
                        state["init"] = aliases.get(
                            state["init"], state["init"]
                        )
                    states.append(state)
                nd["state"] = states
                nd["body"], local_aliases = simplify(
                    nd["body"], aliases, expressions
                )
                nd["next"] = {
                    k: local_aliases.get(v, v)
                    for k, v in nd["next"].items()
                }
                result.append(nd)
                continue
            nd["in"] = [aliases.get(x, x) for x in nd["in"]]
            for field in ("w", "b"):
                if field in nd:
                    nd[field] = parameter_alias[nd[field]]
            key = (
                nd["op"], tuple(nd["in"]),
                tuple(sorted(
                    (k, v) for k, v in nd.items()
                    if k not in ("id", "op", "in", "comment")
                ))
            )
            if key in expressions and nd["id"] not in protected:
                aliases[nd["id"]] = expressions[key]
            else:
                expressions[key] = nd["id"]
                result.append(nd)
        return result, aliases

    graph, _ = simplify(graph)
    node_by_id = {nd["id"]: nd for nd in graph}
    required = set()

    def mark(reference):
        root = reference.split(".", 1)[0]
        if root not in node_by_id or root in required:
            return
        required.add(root)
        nd = node_by_id[root]
        if nd["op"] == "loop":
            for state in nd["state"]:
                if "init" in state:
                    mark(state["init"])
            for inner in nd["body"]:
                for inp in inner.get("in", []):
                    mark(inp)
            for inp in nd["next"].values():
                mark(inp)
        else:
            for inp in nd["in"]:
                mark(inp)

    for output in protected:
        mark(output)
    graph = [nd for nd in graph if nd["id"] in required]

    used_parameters = set()

    def collect_parameters(nodes):
        for nd in nodes:
            for field in ("w", "b"):
                if field in nd:
                    used_parameters.add(nd[field])
            if nd["op"] == "loop":
                collect_parameters(nd["body"])

    collect_parameters(graph)
    final_params = {
        name: value for name, value in canonical_params.items()
        if name in used_parameters
    }
    return {"graph": graph, "memory_size": M, "params": final_params}
