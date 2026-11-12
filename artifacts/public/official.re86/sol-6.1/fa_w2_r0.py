import numpy as np


def build():
    graph, params = [], {}
    shapes = {
        "x0": (16, True), "coords": (3, True),
        "mem": (43, False), "aprev": (8, False), "avail": (8, False)
    }
    active = graph
    serial = 0
    parameter_cache, node_caches = {}, {}

    def fresh():
        nonlocal serial
        serial += 1
        return "n%d" % serial

    def parameter(array):
        array = np.asarray(array, dtype=np.float32)
        key = (array.shape, array.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = array
        return parameter_cache[key]

    def node(op, inputs, shape, name=None, **fields):
        cache = node_caches.setdefault(id(active), {})
        key = (op, tuple(inputs), tuple(sorted(fields.items())))
        if name is None and key in cache:
            return cache[key]
        ident = name or fresh()
        item = {"id": ident, "op": op, "in": list(inputs)}
        item.update(fields)
        active.append(item)
        shapes[ident] = shape
        if name is None:
            cache[key] = ident
        return ident

    def linear(x, weights, bias=None, name=None):
        weights = np.asarray(weights, dtype=np.float32)
        spatial = shapes[x][1]
        if spatial and weights.ndim == 2:
            weights = weights[:, :, None, None]
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node(
            "conv" if spatial else "dense", [x],
            (weights.shape[0], spatial), name=name, **fields
        )

    def affine(x, scale=1, bias=0, name=None):
        c = shapes[x][0]
        return linear(
            x, np.eye(c, dtype=np.float32) * scale,
            np.full(c, bias, dtype=np.float32), name=name
        )

    def unary(op, x, name=None):
        if op == "relu" and not shapes[x][1]:
            op = "vrelu"
        return node(op, [x], shapes[x], name=name)

    def binary(op, a, b, name=None):
        ca, sa = shapes[a]
        cb, sb = shapes[b]
        assert ca == cb or ca == 1 or cb == 1
        spatial = sa or sb
        if not spatial:
            assert op in ("add", "mul")
            op = "vadd" if op == "add" else "vmul"
        return node(op, [a, b], (max(ca, cb), spatial), name=name)

    def add(a, b):
        return binary("add", a, b)

    def mul(a, b):
        return binary("mul", a, b)

    def sub(a, b):
        if shapes[a][1] or shapes[b][1]:
            return binary("sub", a, b)
        return add(a, affine(b, -1))

    def cat(xs, name=None):
        spatial = shapes[xs[0]][1]
        assert all(shapes[x][1] == spatial for x in xs)
        return node(
            "concat" if spatial else "vconcat", xs,
            (sum(shapes[x][0] for x in xs), spatial), name=name
        )

    def cut(x, start, stop):
        return node(
            "slice" if shapes[x][1] else "vslice", [x],
            (stop - start, shapes[x][1]), start=start, stop=stop
        )

    def pool(op, x):
        return node(op, [x], (shapes[x][0], False))

    def reduce_spatial(op, x):
        return node(op, [x], shapes[x])

    def total(x):
        return linear(x, np.ones((1, shapes[x][0]), dtype=np.float32))

    def constant(value, size=1):
        return linear(
            "aprev", np.zeros((size, 8), dtype=np.float32),
            np.full(size, value, dtype=np.float32)
        )

    one, zero = constant(1), constant(0)
    canvas_one = cut("coords", 2, 3)

    def lift(x):
        return mul(canvas_one, x)

    def clip(x):
        if shapes[x][1]:
            return binary("min2", unary("relu", x), one)
        r = unary("relu", x)
        return affine(unary("relu", affine(r, -1, 1)), -1, 1)

    def absolute(x):
        if shapes[x][1]:
            return unary("abs", x)
        return add(unary("relu", x), unary("relu", affine(x, -1)))

    def bit(x, threshold=1):
        return clip(affine(x, 16, 8 - 16 * threshold))

    def equal(a, b):
        return clip(affine(absolute(sub(a, b)), -16, 8))

    def maximum(a, b):
        if shapes[a][1] or shapes[b][1]:
            return binary("max2", a, b)
        return add(b, unary("relu", sub(a, b)))

    def vector_max(x):
        field = lift(x)
        result = cut(field, 0, 1)
        for i in range(1, shapes[x][0]):
            result = maximum(result, cut(field, i, i + 1))
        return pool("gmax", result)

    def dot(a, b):
        return total(mul(a, b))

    def blend(new, old, gate):
        return add(mul(new, gate), mul(old, affine(gate, -1, 1)))

    def shift_kernel(c, dy, dx, k=3):
        weights = np.zeros((c, c, k, k), dtype=np.float32)
        weights[
            np.arange(c), np.arange(c),
            k // 2 - dy, k // 2 - dx
        ] = 1
        return weights

    def finish_loop(T, states, body, next_values):
        ident = fresh()
        graph.append({
            "id": ident, "op": "loop", "T": T,
            "state": states, "body": body, "next": next_values
        })
        outputs = {}
        for state in states:
            sid = state["id"]
            outputs[sid] = ident + "." + sid
            shapes[outputs[sid]] = shapes[sid]
        return outputs

    # Coordinates and selected-object centre.
    xy = affine(cut("coords", 0, 2), 63)
    yy, xx = cut(xy, 0, 1), cut(xy, 1, 2)
    black = cut("x0", 0, 1)
    found = pool("gmax", black)
    observed_centre = affine(pool("gmean", mul(xy, black)), 4096)
    motion = np.zeros((2, 8), dtype=np.float32)
    motion[0, 1], motion[0, 2] = -3, 3
    motion[1, 3], motion[1, 4] = -3, 3
    forecast = add(cut("mem", 22, 24), linear("aprev", motion))
    centre = blend(observed_centre, forecast, found)
    cy, cx = cut(centre, 0, 1), cut(centre, 1, 2)
    relative = sub(xy, centre)
    dy, dx = cut(relative, 0, 1), cut(relative, 1, 2)
    ay, ax = absolute(dy), absolute(dx)
    axis_y, axis_x = equal(dy, zero), equal(dx, zero)
    marker = mul(axis_y, axis_x)
    left, right = bit(affine(dx, -1)), bit(dx)
    up, down = bit(affine(dy, -1)), bit(dy)

    # Stable target slots, including frames partly overwritten off-centre.
    four = cut("x0", 4, 5)
    ring_weights = np.zeros((3, 1, 3, 3), dtype=np.float32)
    for y, x in ((0, 0), (0, 2), (2, 0), (2, 2)):
        ring_weights[0, 0, y, x] = 1
    for y, x in ((0, 1), (1, 0), (1, 2), (2, 1)):
        ring_weights[1, 0, y, x] = 1
    ring_weights[2, 0] = 1
    ring_weights[2, 0, 1, 1] = 0
    rings = linear(four, ring_weights)
    corners, edges, ring = (
        cut(rings, 0, 1), cut(rings, 1, 2), cut(rings, 2, 3)
    )
    targets = maximum(
        maximum(bit(corners, 4), bit(edges, 4)), bit(ring, 5)
    )
    non_frame_centre = affine(
        add(four, cut("x0", 5, 6)), -1, 1
    )
    targets = mul(targets, non_frame_centre)
    visible = bit(ring, 8)

    row_total = reduce_spatial("rowsum", targets)
    hs, vs = fresh(), fresh()
    shapes[hs] = shapes[vs] = (1, True)
    body = []
    active = body
    hn = add(linear(hs, shift_kernel(1, 0, 1)), targets)
    vn = add(linear(vs, shift_kernel(1, 1, 0)), row_total)
    active = graph
    prefix = finish_loop(
        64,
        [{"id": hs, "zeros": [1, 64, 64]},
         {"id": vs, "zeros": [1, 64, 64]}],
        body, {hs: hn, vs: vn}
    )
    rank = sub(add(prefix[hs], prefix[vs]), row_total)
    N = 16
    rank_difference = linear(
        rank, np.ones((N, 1), dtype=np.float32),
        -np.arange(1, N + 1, dtype=np.float32)
    )
    slots = mul(
        targets, unary("relu", affine(absolute(rank_difference), -1, 1))
    )
    palette_value = linear(
        "x0", np.arange(16, dtype=np.float32)[None, :]
    )
    presence = affine(pool("gmean", slots), 4096)
    visible_slots = mul(slots, visible)
    visible_mass = affine(pool("gmean", visible_slots), 4096)
    revealed = affine(
        pool("gmean", mul(visible_slots, palette_value)), 4096
    )
    old_goal_codes = cut("mem", 0, N)
    goal_codes = blend(revealed, old_goal_codes, visible_mass)
    known = bit(goal_codes)
    newly_known = bit(total(mul(
        visible_mass, affine(bit(old_goal_codes), -1, 1)
    )))
    target_y = affine(pool("gmean", mul(slots, yy)), 4096)
    target_x = affine(pool("gmean", mul(slots, xx)), 4096)
    observed_codes = affine(
        pool("gmean", mul(slots, palette_value)), 4096
    )
    covered = mul(
        mul(equal(observed_codes, goal_codes), affine(visible_mass, -1, 1)),
        known
    )

    # Pads are filled colour blocks, not arbitrary nearby coloured strokes.
    border = cut("x0", 2, 3)
    pad_near = clip(linear(
        border, np.ones((1, 1, 7, 7), dtype=np.float32)
    ))
    colours = cut("x0", 6, 15)
    block_weights = np.zeros((9, 9, 3, 3), dtype=np.float32)
    for c in range(9):
        block_weights[c, c] = 1
    filled_blocks = bit(linear(colours, block_weights), 4)
    pad_pixels = mul(mul(colours, filled_blocks), pad_near)
    pad_mask = clip(total(pad_pixels))
    excluded = maximum(mul(targets, visible), pad_mask)
    pixels = mul(colours, affine(excluded, -1, 1))

    # Local directional stroke evidence.
    local_weights = np.zeros((27, 9, 3, 3), dtype=np.float32)
    for c in range(9):
        local_weights[c, c, 1, 0] = local_weights[c, c, 1, 2] = 1
        local_weights[9 + c, c, 0, 1] = local_weights[9 + c, c, 2, 1] = 1
        for y, x in ((0, 0), (0, 2), (2, 0), (2, 2)):
            local_weights[18 + c, c, y, x] = 1
    local = affine(
        pool("gmean", mul(linear(pixels, local_weights), marker)), 4096
    )
    local_h, local_v, local_d = (
        cut(local, 0, 9), cut(local, 9, 18), cut(local, 18, 27)
    )

    side_masks = [
        mul(axis_y, left), mul(axis_y, right),
        mul(axis_x, up), mul(axis_x, down)
    ]
    nearest, evidence = [], []
    for side, distance in zip(side_masks, (ax, ax, ay, ay)):
        q = mul(pixels, side)
        evidence.append(pool("gmax", q))
        nearest.append(affine(
            pool("gmax", mul(q, affine(distance, -1, 64))), -1, 64
        ))
    colour_score = affine(add(add(local_h, local_v), local_d), 100)
    for e, d in zip(evidence, nearest):
        colour_score = add(
            colour_score, sub(affine(e, 8), affine(d, 0.04))
        )
    colour_score = linear(
        colour_score, np.eye(9, dtype=np.float32),
        np.arange(9, 0, -1, dtype=np.float32) * 0.005
    )
    colour_best = vector_max(colour_score)
    selected_colour = clip(affine(
        absolute(sub(colour_score, colour_best)), -400, 1.5
    ))
    selected_pixels = total(mul(pixels, selected_colour))

    # Contiguous cardinal and diagonal arms.
    directions = [
        (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    ray_weights = np.zeros((8, 8, 3, 3), dtype=np.float32)
    for c, (sy, sx) in enumerate(directions):
        ray_weights[c, c, 1 - sy, 1 - sx] = 1
    seed = linear(marker, np.ones((8, 1), dtype=np.float32))
    allowed = maximum(selected_pixels, marker)
    ray_state = fresh()
    shapes[ray_state] = (8, True)
    body = []
    active = body
    ray_next = maximum(
        ray_state, mul(linear(ray_state, ray_weights), allowed)
    )
    active = graph
    rays = finish_loop(
        32, [{"id": ray_state, "init": seed}], body,
        {ray_state: ray_next}
    )[ray_state]
    ray_extent = pool("gmax", mul(rays, maximum(ax, ay)))
    arms = [cut(ray_extent, i, i + 1) for i in range(4)]
    near = [dot(d, selected_colour) for d in nearest]

    # Cross geometry may have an obstacle-displaced intersection.
    long_rows = bit(reduce_spatial("rowsum", pixels), 8)
    long_cols = bit(reduce_spatial("colsum", pixels), 8)
    number_rows = dot(
        affine(pool("gmean", long_rows), 64), selected_colour
    )
    number_cols = dot(
        affine(pool("gmean", long_cols), 64), selected_colour
    )
    bar_y = dot(
        affine(pool("gmean", mul(long_rows, yy)), 64), selected_colour
    )
    bar_x = dot(
        affine(pool("gmean", mul(long_cols, xx)), 64), selected_colour
    )
    lh, lv, ld = (
        dot(local_h, selected_colour),
        dot(local_v, selected_colour),
        dot(local_d, selected_colour)
    )
    local_cross = mul(bit(lh, 2), bit(lv, 2))
    x_hint = bit(ld, 3)
    displaced_cross = mul(
        mul(equal(number_rows, one), equal(number_cols, one)),
        affine(x_hint, -1, 1)
    )
    is_cross = clip(add(local_cross, displaced_cross))
    not_cross = affine(is_cross, -1, 1)
    is_x = mul(x_hint, not_cross)
    remaining = mul(not_cross, affine(is_x, -1, 1))
    is_h = mul(bit(lh, 2), remaining)
    is_v = mul(mul(bit(lv, 2), remaining), affine(is_h, -1, 1))
    is_hollow = affine(
        add(add(is_cross, is_x), add(is_h, is_v)), -1, 1
    )
    displaced = mul(is_cross, affine(local_cross, -1, 1))
    off_y = mul(sub(bar_y, cy), displaced)
    off_x = mul(sub(bar_x, cx), displaced)

    horizontal_pixels = mul(
        selected_pixels, total(mul(long_rows, selected_colour))
    )
    vertical_pixels = mul(
        selected_pixels, total(mul(long_cols, selected_colour))
    )
    far_l = pool("gmax", mul(mul(horizontal_pixels, left), ax))
    far_r = pool("gmax", mul(mul(horizontal_pixels, right), ax))
    far_u = pool("gmax", mul(mul(vertical_pixels, up), ay))
    far_d = pool("gmax", mul(mul(vertical_pixels, down), ay))
    measured_x = blend(
        maximum(far_l, far_r), maximum(arms[0], arms[1]), displaced
    )
    measured_y = blend(
        maximum(far_u, far_d), maximum(arms[2], arms[3]), displaced
    )
    measured_diagonal = vector_max(cut(ray_extent, 4, 8))
    # A nearer same-colour shape may obscure one cardinal edge of a diamond.
    measured_diamond = vector_max(cat(near))
    measured_radii = cat([
        measured_x, measured_y, measured_diagonal, measured_diamond
    ])

    old_plan_colour = cut("mem", 28, 37)
    plan_present = bit(total(old_plan_colour))
    fresh_selection = clip(add(
        cut("aprev", 5, 6), affine(plan_present, -1, 1)
    ))
    radii = blend(
        measured_radii,
        pool("gmax", maximum(lift(measured_radii), lift(cut("mem", 24, 28)))),
        fresh_selection
    )
    radius_x, radius_y = cut(radii, 0, 1), cut(radii, 1, 2)
    x_radius, diamond_radius = cut(radii, 2, 3), cut(radii, 3, 4)

    def within(distance, radius):
        return clip(affine(sub(lift(radius), distance), 16, 8))

    def rectangle(y, x):
        l, r, u, d = near
        inside_x = mul(
            clip(affine(add(x, l), 16, 8)),
            clip(affine(sub(lift(r), x), 16, 8))
        )
        inside_y = mul(
            clip(affine(add(y, u), 16, 8)),
            clip(affine(sub(lift(d), y), 16, 8))
        )
        vertical_edges = maximum(equal(x, affine(l, -1)), equal(x, r))
        horizontal_edges = maximum(equal(y, affine(u, -1)), equal(y, d))
        return maximum(
            mul(vertical_edges, inside_y), mul(horizontal_edges, inside_x)
        )

    actual_rectangle = rectangle(dy, dx)
    actual_diamond = equal(add(ay, ax), diamond_radius)
    rectangle_support = pool("gmean", mul(actual_rectangle, selected_pixels))
    diamond_support = pool("gmean", mul(actual_diamond, selected_pixels))
    is_rectangle = bit(
        affine(sub(rectangle_support, diamond_support), 4096), 1
    )

    # Shape membership at every target relative to every candidate centre.
    target_dy = sub(lift(target_y), yy)
    target_dx = sub(lift(target_x), xx)
    target_ay, target_ax = absolute(target_dy), absolute(target_dx)
    horizontal_limit = within(target_ax, radius_x)
    vertical_limit = within(target_ay, radius_y)
    cross = maximum(
        mul(equal(target_dy, off_y), horizontal_limit),
        mul(equal(target_dx, off_x), vertical_limit)
    )
    horizontal = mul(equal(target_dy, zero), horizontal_limit)
    vertical = mul(equal(target_dx, zero), vertical_limit)
    diagonal = mul(equal(target_ay, target_ax), within(target_ax, x_radius))
    diagonal = mul(
        diagonal,
        affine(mul(equal(target_dy, zero), equal(target_dx, zero)), -1, 1)
    )
    diamond = equal(add(target_ay, target_ax), diamond_radius)
    hollow = blend(rectangle(target_dy, target_dx), diamond, is_rectangle)
    candidate_shapes = add(
        add(mul(cross, is_cross), mul(diagonal, is_x)),
        add(
            add(mul(horizontal, is_h), mul(vertical, is_v)),
            mul(hollow, is_hollow)
        )
    )
    own_cover = affine(pool("gmean", mul(candidate_shapes, marker)), 4096)
    target_weights = mul(
        mul(known, presence),
        add(
            affine(covered, -3, 3),
            mul(covered, own_cover)
        )
    )

    # Reachable centres share the selected object's modulo-three phase.
    row_seed = reduce_spatial("rowmax", marker)
    col_seed = reduce_spatial("colmax", marker)
    row_state, col_state = fresh(), fresh()
    shapes[row_state] = shapes[col_state] = (1, True)
    body = []
    active = body
    row_kernel = (
        shift_kernel(1, -3, 0, 7) + shift_kernel(1, 3, 0, 7)
        + shift_kernel(1, 0, 0, 7)
    )
    col_kernel = (
        shift_kernel(1, 0, -3, 7) + shift_kernel(1, 0, 3, 7)
        + shift_kernel(1, 0, 0, 7)
    )
    row_next, col_next = (
        clip(linear(row_state, row_kernel)),
        clip(linear(col_state, col_kernel))
    )
    active = graph
    reach = finish_loop(
        21,
        [{"id": row_state, "init": row_seed},
         {"id": col_state, "init": col_seed}],
        body, {row_state: row_next, col_state: col_next}
    )
    reachable = mul(reach[row_state], reach[col_state])

    def mask_centre(mask, coordinate):
        high = pool("gmax", mul(mask, coordinate))
        reverse = pool("gmax", mul(mask, affine(coordinate, -1, 63)))
        return affine(sub(high, reverse), 0.5, 31.5)

    pad_available = pool("gmax", pad_pixels)
    pad_y, pad_x = mask_centre(pad_pixels, yy), mask_centre(pad_pixels, xx)
    pad_distance = add(absolute(sub(pad_y, cy)), absolute(sub(pad_x, cx)))
    permitted = clip(add(pad_available, selected_colour))

    # Score placement and colour jointly.
    repeat_targets = np.tile(np.eye(N, dtype=np.float32), (9, 1))
    expanded_codes = linear(
        goal_codes, repeat_targets,
        -np.repeat(np.arange(6, 15, dtype=np.float32), N)
    )
    wanted = clip(affine(absolute(expanded_codes), -16, 8))
    weights_by_colour = mul(wanted, linear(target_weights, repeat_targets))
    weighted_shapes = mul(
        linear(candidate_shapes, repeat_targets), weights_by_colour
    )
    reduce_colours = np.zeros((9, 9 * N), dtype=np.float32)
    for c in range(9):
        reduce_colours[c, c * N:(c + 1) * N] = 1
    coverage = linear(weighted_shapes, reduce_colours)

    travel = add(ay, ax)
    ordering = add(affine(yy, 64), xx)
    base_cost = add(affine(travel, -0.001987), affine(ordering, -0.000002))
    base_cost = add(base_cost, affine(reachable, 100, -100))
    colour_cost = add(
        affine(selected_colour, 0.15, -0.15),
        affine(pad_distance, -0.0005)
    )
    colour_cost = add(colour_cost, affine(permitted, 100, -100))
    colour_cost = linear(
        colour_cost, np.eye(9, dtype=np.float32),
        np.arange(9, dtype=np.float32) * 0.00013
    )
    scores = add(add(coverage, base_cost), colour_cost)
    best_by_colour = pool("gmax", scores)
    best_value = vector_max(best_by_colour)
    proposed_colour = clip(affine(
        absolute(sub(best_by_colour, best_value)), -100000, 1.5
    ))
    winning_cells = clip(affine(
        absolute(sub(scores, best_value)), -1000000, 1.5
    ))
    winning_cells = total(mul(winning_cells, proposed_colour))
    proposed_y = pool("gmax", mul(winning_cells, yy))
    proposed_x = pool("gmax", mul(winning_cells, xx))
    proposed_position = cat([proposed_y, proposed_x])

    # Obstacle recovery and shape-change detection.
    above_ui = clip(affine(yy, -16, 16 * 62 - 8))
    obstacle = mul(cut("x0", 1, 2), above_ui)
    obstacle_exists = pool("gmax", obstacle)
    obstacle_y, obstacle_x = mask_centre(obstacle, yy), mask_centre(obstacle, xx)
    old_stuck, old_timer = cut("mem", 16, 17), cut("mem", 17, 18)
    old_phase = cut("mem", 18, 22)
    previous_move = total(cut("aprev", 1, 5))
    unchanged = equal(
        total(absolute(sub(centre, cut("mem", 22, 24)))), zero
    )
    unproductive = clip(add(
        cut("aprev", 5, 6), mul(previous_move, unchanged)
    ))
    stuck = mul(add(old_stuck, one), unproductive)
    remaining_targets = total(mul(
        mul(known, presence), affine(covered, -1, 1)
    ))
    trigger = mul(
        mul(bit(stuck, 5), affine(bit(old_timer), -1, 1)),
        mul(bit(remaining_targets), obstacle_exists)
    )
    stuck_next = mul(stuck, affine(trigger, -1, 1))
    timer_next = add(
        unary("relu", affine(old_timer, 1, -1)), affine(trigger, 20)
    )
    phase_rotation = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        phase_rotation[(i + 1) % 4, i] = 1
    initial_phase = linear(
        unary("relu", affine(total(old_phase), -1, 1)),
        np.array([[1], [0], [0], [0]], dtype=np.float32)
    )
    phase_next = blend(
        add(linear(old_phase, phase_rotation), initial_phase),
        old_phase, trigger
    )
    explore = bit(timer_next)
    near_vector = cat(near)
    hollow_change = mul(
        mul(is_hollow, obstacle_exists),
        bit(total(absolute(sub(near_vector, cut("mem", 39, 43)))), 2)
    )
    replan = clip(add(
        add(fresh_selection, newly_known),
        add(bit(old_timer), hollow_change)
    ))
    desired_colour = blend(proposed_colour, old_plan_colour, replan)
    goal_position = blend(
        proposed_position, cut("mem", 37, 39), replan
    )
    goal_y, goal_x = cut(goal_position, 0, 1), cut(goal_position, 1, 2)

    # Retain the chosen colour until both recolouring and placement finish.
    recolour = affine(dot(desired_colour, selected_colour), -1, 1)
    waypoint_y = blend(dot(pad_y, desired_colour), goal_y, recolour)
    waypoint_x = blend(dot(pad_x, desired_colour), goal_x, recolour)
    explore_y = add(
        obstacle_y,
        linear(phase_next, np.array([[0, 0, 18, -18]], dtype=np.float32))
    )
    explore_x = add(
        obstacle_x,
        linear(phase_next, np.array([[18, -18, 0, 0]], dtype=np.float32))
    )
    waypoint_y = blend(explore_y, waypoint_y, explore)
    waypoint_x = blend(explore_x, waypoint_x, explore)

    # Reveal unknown target centres before committing to their colours.
    reveal = bit(total(mul(
        mul(affine(known, -1, 1), presence), own_cover
    )))
    previous_vertical = total(cut("aprev", 1, 3))
    reveal_horizontal = clip(add(
        is_v, mul(affine(add(is_h, is_v), -1, 1), previous_vertical)
    ))
    sign_y, sign_x = affine(bit(cy, 32), -2, 1), affine(bit(cx, 32), -2, 1)
    reveal_y = add(
        cy, mul(affine(sign_y, 6), affine(reveal_horizontal, -1, 1))
    )
    reveal_x = add(cx, mul(affine(sign_x, 6), reveal_horizontal))
    waypoint_y = blend(reveal_y, waypoint_y, reveal)
    waypoint_x = blend(reveal_x, waypoint_x, reveal)

    # Directional logits; ACTION5 wins when the goal is reached.
    nav_y, nav_x = sub(waypoint_y, cy), sub(waypoint_x, cx)
    navigation = cat([nav_x, nav_y, absolute(nav_x), absolute(nav_y)])
    action_weights = np.zeros((8, 4), dtype=np.float32)
    action_weights[1, 1], action_weights[2, 1] = -1, 1
    action_weights[3, 0], action_weights[4, 0] = -1, 1
    action_weights[5, 2:] = -1
    linear(
        navigation, action_weights,
        np.array([-1000, 0, 0, 0, 0, 1, -1000, -1000], dtype=np.float32),
        name="logits"
    )
    linear(
        "coords", np.zeros((1, 3), dtype=np.float32),
        np.zeros(1, dtype=np.float32), name="click"
    )
    cat([
        goal_codes, stuck_next, timer_next, phase_next, centre,
        radii, desired_colour, goal_position, near_vector
    ], name="mem_next")

    # Structural dead-code removal only: no fitting or gameplay simulation.
    definitions, exports = {}, {}

    def register(nodes):
        for nd in nodes:
            definitions[nd["id"]] = nd
            if nd["op"] == "loop":
                for state in nd["state"]:
                    exports[nd["id"] + "." + state["id"]] = nd["id"]
                register(nd["body"])

    register(graph)
    needed = set()

    def visit(ident):
        ident = exports.get(ident, ident)
        if ident in needed or ident not in definitions:
            return
        needed.add(ident)
        nd = definitions[ident]
        if nd["op"] == "loop":
            for state in nd["state"]:
                if "init" in state:
                    visit(state["init"])
            for source in nd["next"].values():
                visit(source)
        else:
            for source in nd["in"]:
                visit(source)

    for output in ("logits", "click", "mem_next"):
        visit(output)

    def prune(nodes):
        result = []
        for nd in nodes:
            if nd["id"] in needed:
                if nd["op"] == "loop":
                    nd = dict(nd)
                    nd["body"] = prune(nd["body"])
                result.append(nd)
        return result

    graph = prune(graph)
    used_parameters = set()

    def collect(nodes):
        for nd in nodes:
            for field in ("w", "b"):
                if field in nd:
                    used_parameters.add(nd[field])
            if nd["op"] == "loop":
                collect(nd["body"])

    collect(graph)
    params = {
        name: array for name, array in params.items()
        if name in used_parameters
    }
    return {"graph": graph, "memory_size": 43, "params": params}
