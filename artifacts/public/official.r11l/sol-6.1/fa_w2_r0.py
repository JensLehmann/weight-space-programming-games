import numpy as np


def build():
    graph = []
    params = {}
    serial = 0

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def parameter(value):
        key = fresh("p")
        params[key] = np.asarray(value, dtype=np.float32)
        return key

    def node(kind, inputs, **fields):
        key = fresh()
        graph.append({
            "id": key, "op": kind, "in": list(inputs), **fields
        })
        return key

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def constant(values):
        values = np.asarray(values, np.float32).reshape(-1)
        return dense(
            "avail", np.zeros((len(values), 8), np.float32), values
        )

    one = constant([1])
    zero = constant([0])

    def affine(x, scale=1.0, bias=0.0, channels=1):
        w = np.eye(channels, dtype=np.float32)[:, :, None, None]
        return conv(
            x, w * scale, np.full(channels, bias, np.float32)
        )

    def va(x, scale=1.0, bias=0.0, channels=1):
        return dense(
            x, np.eye(channels, dtype=np.float32) * scale,
            np.full(channels, bias, np.float32)
        )

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def vadd(a, b):
        return node("vadd", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def clip(x):
        return node("min2", [node("relu", [x]), one])

    def threshold(x, cutoff=0.0, gain=1.0, channels=1):
        return clip(affine(x, gain, -cutoff * gain, channels))

    def inverse(x):
        return sub(one, x)

    def vi(x):
        return va(x, -1, 1)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vs(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def cat(*inputs):
        return node("concat", inputs)

    def vcat(*inputs):
        return node("vconcat", inputs)

    def maximum(x):
        return node("gmax", [x])

    def total(x, channels=1):
        return va(node("gmean", [x]), 4096, channels=channels)

    def read(x, mask, channels=1):
        return total(mul(x, mask), channels)

    def channel_sum(x, channels):
        return conv(x, np.ones((1, channels, 1, 1), np.float32))

    def mix(old, new, keep):
        return vadd(vmul(old, keep), vmul(new, vi(keep)))

    def scalar_gate(x, cutoff, gain=100.0):
        return maximum(threshold(
            mul(spatial_one, x), cutoff, gain
        ))

    def color(c):
        return sl("x0", c, c + 1)

    row = conv(
        "coords", np.array([[[[63]], [[0]], [[0]]]], np.float32)
    )
    col = conv(
        "coords", np.array([[[[0]], [[63]], [[0]]]], np.float32)
    )
    spatial_one = sl("coords", 2, 3)
    priority = conv(
        "coords",
        np.array([[[[-4032]], [[-63]], [[0]]]], np.float32),
        [4096]
    )

    def distance(y, x):
        return add(
            node("abs", [sub(row, y)]),
            node("abs", [sub(col, x)])
        )

    def point(mask):
        return read(row, mask), read(col, mask)

    def penalize(score, mask, amount=10000.0):
        return sub(score, affine(inverse(mask), amount))

    def winner(score):
        equal = threshold(
            sub(score, maximum(score)), -0.00001, 100000
        )
        ordered = mul(equal, priority)
        return threshold(
            sub(ordered, maximum(ordered)), -0.00001, 100000
        )

    def dw(channels, kernel):
        k = kernel.shape[0]
        w = np.zeros((channels, channels, k, k), np.float32)
        for c in range(channels):
            w[c, c] = kernel
        return w

    def dilate(x, channels, sizes):
        for k in sizes:
            kernel = np.ones((k, k), np.float32)
            x = clip(conv(x, dw(channels, kernel)))
        return x

    def palette_sum(included):
        w = np.zeros((1, 16, 1, 1), np.float32)
        for c in included:
            w[0, c, 0, 0] = 1
        return conv("x0", w)

    C = 16
    diamond = np.zeros((5, 5), np.float32)
    disc = np.zeros((5, 5), np.float32)
    for y in range(-2, 3):
        for x in range(-2, 3):
            diamond[y + 2, x + 2] = float(abs(y) + abs(x) <= 2)
            disc[y + 2, x + 2] = float(
                not (abs(y) == 2 and abs(x) == 2)
            )
    diamond_edge = diamond.copy()
    diamond_edge[2, 2] = 0
    disc_edge = disc.copy()
    disc_edge[2, 2] = 0

    # Diamond centres and the currently selected diamond.
    handle_boundary = add(color(0), color(3))
    handles = threshold(
        conv(
            handle_boundary,
            (diamond_edge - (disc - diamond))[None, None]
        ),
        11.5, 2
    )
    selected = mul(
        handles,
        threshold(
            conv(color(0), diamond_edge[None, None]), 11.5, 2
        )
    )
    selected_any = maximum(selected)
    hy, hx = point(selected)
    handle_colors = mul("x0", handles)

    # Filled circular sprites. Background and connector pixels cannot
    # satisfy the twenty required perimeter pixels.
    solid_colors = [
        c for c in range(16) if c not in (1, 2, 3, 5, 10)
    ]
    solid = palette_sum(solid_colors)
    discs = threshold(
        conv(solid, disc_edge[None, None]), 19.5, 2
    )
    colored_balls = mul(discs, color(6))
    plain_black_balls = mul(
        discs,
        threshold(
            conv(color(0), disc_edge[None, None]), 19.5, 2
        )
    )
    exterior = np.ones((7, 7), np.float32)
    exterior[1:6, 1:6] -= disc
    near_line = threshold(
        conv(color(1), exterior[None, None]), 0.5, 2
    )
    balls = mx(
        colored_balls,
        mx(plain_black_balls, mul(discs, near_line))
    )

    # Half-colours at the left and right edges of each filled circle.
    wl = np.zeros((C, C, 5, 5), np.float32)
    wr = np.zeros_like(wl)
    for c in range(C):
        wl[c, c, 2, 0] = 1
        wr[c, c, 2, 4] = 1
    ball_left = conv("x0", wl)
    ball_right = conv("x0", wr)

    left_ring = np.zeros((7, 7), np.float32)
    right_ring = np.zeros((7, 7), np.float32)
    ring_offsets = [
        (-3, -1), (-2, -2), (-1, -3),
        (1, -3), (2, -2), (3, -1)
    ]
    for y, x in ring_offsets:
        left_ring[y + 3, x + 3] = 1
        right_ring[y + 3, -x + 3] = 1
    ring = left_ring + right_ring

    # Explicit ring presence, independent of colour matching.
    ring_ink = palette_sum([6, 7, 8, 9, 11, 12, 13, 14, 15])
    ring_present = threshold(
        conv(ring_ink, ring[None, None]), 11.5, 2
    )
    uniform_left = threshold(
        conv("x0", dw(C, left_ring)), 5.5, 2, C
    )
    uniform_right = threshold(
        conv("x0", dw(C, right_ring)), 5.5, 2, C
    )
    uniform_valid = mul(
        clip(channel_sum(uniform_left, C)),
        clip(channel_sum(uniform_right, C))
    )
    uniform_goals = mul(ring_present, uniform_valid)
    have_uniform = maximum(uniform_goals)
    goals = mx(
        uniform_goals,
        mul(ring_present, vi(have_uniform))
    )

    # Sample the two top ring pixels for colour identities.
    glw = np.zeros((C, C, 7, 7), np.float32)
    grw = np.zeros_like(glw)
    for c in range(C):
        glw[c, c, 0, 2] = 1
        grw[c, c, 0, 4] = 1
    goal_left = mul(conv("x0", glw), goals)
    goal_right = mul(conv("x0", grw), goals)
    same_halves = clip(
        channel_sum(mul(goal_left, goal_right), C)
    )
    mixed_goals = mul(goals, inverse(same_halves))
    have_mixed = maximum(mixed_goals)

    # Completion outlines remain detectable even when a ball occludes
    # some of the ring pixels.
    completed_ring = threshold(
        conv(color(0), ring[None, None]), 6.5, 2
    )
    inside_view = mul(
        threshold(col, 2.9, 20),
        threshold(affine(col, -1, 61.1), 0, 20)
    )
    completed_ring = mul(completed_ring, inside_view)
    completed_near = dilate(completed_ring, 1, (7, 5))

    # Remaining target colours prevent completed balls from being
    # repeatedly selected after their coloured goal disappears.
    needed_left = maximum(goal_left)
    needed_right = maximum(goal_right)
    possible_left = clip(add(
        channel_sum(mul(ball_left, needed_left), C),
        sl(ball_left, 0, 1)
    ))
    possible_right = clip(add(
        channel_sum(mul(ball_right, needed_right), C),
        sl(ball_right, 0, 1)
    ))
    remaining_compatible = mul(possible_left, possible_right)
    fresh_candidates = mul(
        mul(balls, remaining_compatible),
        inverse(completed_near)
    )
    # Fallback is used only if all compatibility candidates disappear.
    have_candidate = maximum(fresh_candidates)
    fresh_candidates = mx(
        fresh_candidates,
        mul(mul(balls, inverse(completed_near)), vi(have_candidate))
    )

    # Memory: active, predicted ball y/x, goal y/x, bad handle y/x,
    # then the two target-colour one-hots.
    M = 7 + 2 * C
    old_active = vs("mem", 0, 1)
    old_by = vs("mem", 1, 2)
    old_bx = vs("mem", 2, 3)
    old_gy = vs("mem", 3, 4)
    old_gx = vs("mem", 4, 5)
    old_bad_y = vs("mem", 5, 6)
    old_bad_x = vs("mem", 6, 7)
    old_left = vs("mem", 7, 7 + C)
    old_right = vs("mem", 7 + C, M)

    tracked = winner(
        penalize(affine(distance(old_by, old_bx), -1), discs)
    )
    tracked_y, tracked_x = point(tracked)
    tracked_left = read(ball_left, tracked, C)
    tracked_right = read(ball_right, tracked, C)
    remembered_left_ok = dense(
        vmul(tracked_left, old_left), np.ones((1, C), np.float32)
    )
    remembered_right_ok = dense(
        vmul(tracked_right, old_right), np.ones((1, C), np.float32)
    )
    remembered_distance = read(
        distance(old_gy, old_gx), tracked
    )
    remembered_reached = vmul(
        vmul(remembered_left_ok, remembered_right_ok),
        vi(scalar_gate(remembered_distance, 2.5))
    )
    tracked_done = scalar_gate(
        vadd(read(completed_near, tracked), remembered_reached), 0.5
    )
    continuing = vmul(
        vmul(old_active, vi(tracked_done)), maximum(discs)
    )

    fresh_ball = winner(
        penalize(affine(distance(hy, hx), -1), fresh_candidates)
    )
    chosen_ball = add(
        mul(tracked, continuing),
        mul(fresh_ball, vi(continuing))
    )
    by, bx = point(chosen_ball)
    is_colored = read(color(6), chosen_ball)
    bl = read(ball_left, chosen_ball, C)
    br = read(ball_right, chosen_ball, C)
    left_blank = vs(bl, 0, 1)
    right_blank = vs(br, 0, 1)

    color_goal_match = mul(
        clip(channel_sum(mul(goal_left, bl), C)),
        clip(channel_sum(mul(goal_right, bl), C))
    )
    black_left_match = clip(add(
        channel_sum(mul(goal_left, bl), C), left_blank
    ))
    black_right_match = clip(add(
        channel_sum(mul(goal_right, br), C), right_blank
    ))
    black_goal_match = mul(black_left_match, black_right_match)
    black_base = mx(mixed_goals, mul(goals, vi(have_mixed)))
    goal_match = mul(goals, add(
        mul(color_goal_match, is_colored),
        mul(mul(black_goal_match, black_base), vi(is_colored))
    ))
    have_match = maximum(goal_match)
    goal_mask = mx(goal_match, mul(goals, vi(have_match)))

    fresh_goal = winner(
        penalize(affine(distance(by, bx), -1), goal_mask)
    )
    new_gy, new_gx = point(fresh_goal)
    gy = mix(old_gy, new_gy, continuing)
    gx = mix(old_gx, new_gx, continuing)
    target_left = mix(
        old_left, read(goal_left, fresh_goal, C), continuing
    )
    target_right = mix(
        old_right, read(goal_right, fresh_goal, C), continuing
    )

    left_ok = dense(
        vmul(bl, target_left), np.ones((1, C), np.float32)
    )
    right_ok = dense(
        vmul(br, target_right), np.ones((1, C), np.float32)
    )
    need_left = vmul(vi(is_colored), vi(left_ok))
    need_right = vmul(vi(is_colored), vi(right_ok))
    suitable_pickups = clip(add(
        mul(
            channel_sum(mul(ball_left, target_left), C),
            need_left
        ),
        mul(
            channel_sum(mul(ball_right, target_right), C),
            need_right
        )
    ))
    black_half = clip(add(
        sl(ball_left, 0, 1), sl(ball_right, 0, 1)
    ))
    pickups = mul(
        mul(mul(discs, black_half), inverse(balls)),
        suitable_pickups
    )
    have_pickup = maximum(pickups)
    pickup = winner(
        penalize(affine(distance(by, bx), -1), pickups)
    )
    py, px = point(pickup)
    ty = mix(gy, py, have_pickup)
    tx = mix(gx, px, have_pickup)

    def flood(seed, allowed, iterations=128):
        loop_id, state_id = fresh("F"), fresh("s")
        kernel = parameter(np.ones((1, 1, 3, 3), np.float32))
        body = []

        def bn(kind, inputs, **fields):
            key = fresh("b")
            body.append({
                "id": key, "op": kind, "in": list(inputs), **fields
            })
            return key

        spread = bn("conv", [state_id], w=kernel)
        positive = bn("relu", [spread])
        capped = bn("min2", [positive, one])
        united = bn("max2", [capped, seed])
        result = bn("mul", [united, allowed])
        graph.append({
            "id": loop_id, "op": "loop", "T": iterations,
            "state": [{"id": state_id, "init": seed}],
            "body": body, "next": {state_id: result}
        })
        return loop_id + "." + state_id

    # Connector-component membership for black balls.
    endpoints = dilate(mx(handles, discs), 1, (5,))
    connector_world = clip(add(color(1), endpoints))
    connected = flood(chosen_ball, connector_world)
    black_members = mul(
        mul(handles, color(4)), connected
    )
    colored_members = mul(
        handles, clip(channel_sum(mul(handle_colors, bl), C))
    )
    members = add(
        mul(colored_members, is_colored),
        mul(black_members, vi(is_colored))
    )
    have_members = maximum(members)
    nearest_handle = winner(
        penalize(affine(distance(by, bx), -1), handles)
    )
    members = mx(members, mul(nearest_handle, vi(have_members)))
    selected_belongs = read(members, selected)

    # Handle-count reciprocal, exact at integer counts 1 through 16.
    count = vadd(
        node("vrelu", [va(total(members), 1, -1)]), one
    )
    hinges = dense(
        count, np.ones((16, 1), np.float32),
        -np.arange(1, 17, dtype=np.float32)
    )
    hinges = node("vrelu", [hinges])
    slopes = [1.0 / (i + 1) - 1.0 / i for i in range(1, 16)]
    reciprocal_w = np.zeros((1, 16), np.float32)
    reciprocal_w[0, 0] = slopes[0]
    for i in range(1, 15):
        reciprocal_w[0, i] = slopes[i] - slopes[i - 1]
    reciprocal_w[0, 15] = -slopes[-1]
    reciprocal = dense(hinges, reciprocal_w, [1])

    # Use the actual handle centroid, retaining its fractional part.
    mean_y = vmul(read(row, members), reciprocal)
    mean_x = vmul(read(col, members), reciprocal)
    predicted_y = add(mul(sub(row, hy), reciprocal), mean_y)
    predicted_x = add(mul(sub(col, hx), reciprocal), mean_x)

    bounded_y = mul(
        threshold(row, 1.9, 20),
        threshold(affine(row, -1, 61.1), 0, 20)
    )
    bounded_x = mul(
        threshold(col, 1.9, 20),
        threshold(affine(col, -1, 61.1), 0, 20)
    )
    bounds = mul(bounded_y, bounded_x)

    # Only orange is a ball-loss hazard. Walls constrain the handles.
    hazard_overlap = clip(
        conv(color(10), disc[None, None])
    )
    ball_safe = mul(inverse(hazard_overlap), bounds)
    handle_blocked = clip(
        conv(color(2), diamond[None, None])
    )
    handle_click_boxes = dilate(handles, 1, (5,))
    handle_safe = mul(
        mul(inverse(handle_blocked), inverse(handle_click_boxes)),
        bounds
    )

    def propagate(seed, allowed, iterations=128):
        w = np.zeros((4, 1, 3, 3), np.float32)
        for c, (y, x) in enumerate(
            [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ):
            w[c, 0, y + 1, x + 1] = 1
        neighbor_w = parameter(w)
        decrement_w = parameter(np.array([[[[1]]]], np.float32))
        decrement_b = parameter([-1])
        loop_id, state_id = fresh("D"), fresh("s")
        body = []

        def bn(kind, inputs, **fields):
            key = fresh("b")
            body.append({
                "id": key, "op": kind, "in": list(inputs), **fields
            })
            return key

        neighbors = bn("conv", [state_id], w=neighbor_w)
        parts = [
            bn("slice", [neighbors], start=i, stop=i + 1)
            for i in range(4)
        ]
        a = bn("max2", parts[:2])
        b = bn("max2", parts[2:])
        c = bn("max2", [a, b])
        d = bn("conv", [c], w=decrement_w, b=decrement_b)
        e = bn("relu", [d])
        f = bn("max2", [e, seed])
        result = bn("mul", [f, allowed])
        graph.append({
            "id": loop_id, "op": "loop", "T": iterations,
            "state": [{"id": state_id, "init": seed}],
            "body": body, "next": {state_id: result}
        })
        return loop_id + "." + state_id

    target_distance = distance(ty, tx)
    target_seed = threshold(
        affine(target_distance, -1, 2.1), 0, 20
    )
    seed = mul(affine(target_seed, 256), ball_safe)
    field = propagate(seed, ball_safe)
    source = cat(field, ball_safe)

    def sample_axis(source_tensor, coordinate, predicted, reduction):
        loop_id = fresh("W")
        accumulator, counter = fresh("a"), fresh("t")
        body = []
        triangle_w = parameter(np.array([[[[-1]]]], np.float32))
        triangle_b = parameter([1])

        def bn(kind, inputs, **fields):
            key = fresh("b")
            body.append({
                "id": key, "op": kind, "in": list(inputs), **fields
            })
            return key

        d = bn("sub", [coordinate, counter])
        d = bn("abs", [d])
        d = bn("conv", [d], w=triangle_w, b=triangle_b)
        source_mask = bn("relu", [d])
        isolated = bn("mul", [source_tensor, source_mask])
        profile = bn(reduction, [isolated])
        q = bn("sub", [predicted, counter])
        q = bn("abs", [q])
        q = bn("conv", [q], w=triangle_w, b=triangle_b)
        weight = bn("relu", [q])
        contribution = bn("mul", [profile, weight])
        accumulated = bn("add", [accumulator, contribution])
        advanced = bn("vadd", [counter, one])
        graph.append({
            "id": loop_id, "op": "loop", "T": 64,
            "state": [
                {"id": accumulator, "zeros": [2, 64, 64]},
                {"id": counter, "init": zero}
            ],
            "body": body,
            "next": {
                accumulator: accumulated,
                counter: advanced
            }
        })
        return loop_id + "." + accumulator

    sampled_y = sample_axis(source, row, predicted_y, "colsum")
    sampled = sample_axis(sampled_y, col, predicted_x, "rowsum")
    candidate_field = sl(sampled, 0, 1)
    candidate_safety = threshold(sl(sampled, 1, 2), 0.90, 20)
    legal = mul(handle_safe, candidate_safety)

    predicted_distance = add(
        node("abs", [sub(predicted_y, ty)]),
        node("abs", [sub(predicted_x, tx)])
    )
    quality = add(
        affine(predicted_distance, -1),
        affine(candidate_field, 0.10)
    )
    move_score = penalize(quality, legal)
    move_winner = winner(move_score)
    click_y, click_x = point(move_winner)
    best_score = read(move_score, move_winner)

    # Compare with the current centroid using the same sampling rule.
    baseline = read(quality, selected)
    improvement = vadd(best_score, va(baseline, -1))
    progress = scalar_gate(improvement, 0.02, 100)
    moving = vmul(
        vmul(progress, selected_belongs), selected_any
    )

    failed_current = vmul(
        vmul(selected_belongs, selected_any), vi(progress)
    )
    previous_bad = threshold(
        affine(distance(old_bad_y, old_bad_x), -1, 0.1), 0, 20
    )
    alternative_score = sub(
        target_distance,
        affine(mul(selected, failed_current), 10000)
    )
    alternative_score = sub(
        alternative_score,
        affine(mul(previous_bad, continuing), 1000)
    )
    alternative_score = penalize(alternative_score, members)
    handle_winner = winner(alternative_score)

    next_by = mix(
        by, read(predicted_y, move_winner), vi(moving)
    )
    next_bx = mix(
        bx, read(predicted_x, move_winner), vi(moving)
    )
    bad_y = mix(old_bad_y, zero, continuing)
    bad_x = mix(old_bad_x, zero, continuing)
    next_bad_y = mix(bad_y, hy, vi(failed_current))
    next_bad_x = mix(bad_x, hx, vi(failed_current))
    active = maximum(chosen_ball)

    memory_output = vcat(
        active, next_by, next_bx, gy, gx,
        next_bad_y, next_bad_x, target_left, target_right
    )
    graph.append({
        "id": "mem_next", "op": "vadd",
        "in": [memory_output, zero]
    })

    action_bias = np.zeros(8, np.float32)
    action_bias[6] = 1
    graph.append({
        "id": "logits", "op": "dense", "in": ["avail"],
        "w": parameter(np.zeros((8, 8), np.float32)),
        "b": parameter(action_bias)
    })

    final_click = add(
        mul(move_winner, moving),
        mul(handle_winner, vi(moving))
    )
    graph.append({
        "id": "click", "op": "slice",
        "in": [final_click], "start": 0, "stop": 1
    })

    return {
        "graph": graph,
        "memory_size": M,
        "params": params
    }
