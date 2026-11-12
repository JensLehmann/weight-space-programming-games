import numpy as np


def build():
    graph = []
    params = {}
    where = [graph]
    serial = [0]
    pserial = [0]

    def fresh():
        name = "n%d" % serial[0]
        serial[0] += 1
        return name

    def parameter(a):
        name = "p%d" % pserial[0]
        pserial[0] += 1
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def emit(op, ins=None, out=None, **kw):
        name = fresh() if out is None else out
        nd = {"id": name, "op": op}
        if ins is not None:
            nd["in"] = list(ins)
        nd.update(kw)
        where[0].append(nd)
        return name

    def conv(x, w, b=None, out=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return emit("conv", [x], out=out, **kw)

    def linear(x, w, b=None, out=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b, out)

    def dense(x, w, b=None, out=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return emit("dense", [x], out=out, **kw)

    def sl(x, a, b):
        return emit("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return emit("vslice", [x], start=a, stop=b)

    def cat(*xs):
        return emit("concat", xs)

    def vcat(*xs):
        return emit("vconcat", xs)

    one_v = dense("avail", np.zeros((1, 8)), [1.0])
    constants = {(1.0,): one_v}

    def constant(a):
        a = np.asarray(a, np.float32).reshape(-1)
        key = tuple(float(v) for v in a)
        if key not in constants:
            saved = where[0]
            where[0] = graph
            constants[key] = dense(one_v, a[:, None])
            where[0] = saved
        return constants[key]

    one_s = sl("coords", 2, 3)

    def add(a, b):
        return emit("add", [a, b])

    def sub(a, b):
        return emit("sub", [a, b])

    def mul(a, b):
        return emit("mul", [a, b])

    def scale(a, k):
        return mul(a, constant(k))

    def bias(a, k):
        return add(a, constant(k))

    def relu(a):
        return emit("relu", [a])

    def absolute(a):
        return emit("abs", [a])

    def clip(a):
        return emit("min2", [relu(a), one_s])

    def threshold(a, integer_threshold):
        return clip(bias(a, 1.0 - integer_threshold))

    def complement(a):
        return sub(one_s, a)

    def gmax(a):
        return emit("gmax", [a])

    def gmean(a):
        return emit("gmean", [a])

    def reduce_channels(a, channels):
        return linear(a, np.ones((1, channels), np.float32))

    def va(a, b):
        return emit("vadd", [a, b])

    def vm(a, b):
        return emit("vmul", [a, b])

    def vk(a, k):
        return vm(a, constant(k))

    def vb(a, k):
        return va(a, constant(k))

    def vr(a):
        return emit("vrelu", [a])

    def vminus(a, b):
        return va(a, vk(b, -1.0))

    def vclip(a):
        return vminus(vr(a), vr(vb(a, -1.0)))

    def vsum(a, length):
        return dense(a, np.ones((1, length), np.float32))

    pixel = linear("coords", [[63, 0, 0], [0, 63, 0]])
    py = sl(pixel, 0, 1)
    px = sl(pixel, 1, 2)
    rank = linear(pixel, [[-64, -1]])

    def distance_to(v):
        return reduce_channels(absolute(sub(pixel, v)), 2)

    def gather(mask, features):
        return gmax(mul(mask, features))

    def choose(score, mask):
        masked = add(score, scale(sub(mask, one_s), 1.0e7))
        best = gmax(masked)
        peak = relu(bias(scale(sub(masked, best), 1.0e4), 1.0))
        ordered = add(scale(peak, 8192.0), rank)
        top = gmax(ordered)
        win = clip(bias(scale(sub(ordered, top), 32.0), 1.0))
        return mul(win, mask)

    def first(mask):
        ordered = add(rank, scale(mask, 8192.0))
        top = gmax(ordered)
        win = relu(bias(scale(sub(ordered, top), 16.0), 1.0))
        return mul(win, mask)

    diamond = [
        (y, x) for y in range(-2, 3) for x in range(-2, 3)
        if abs(y) + abs(x) <= 2
    ]
    anchor_border = [p for p in diamond if p != (0, 0)]
    round_shape = [
        (y, x) for y in range(-2, 3) for x in range(-2, 3)
        if not (abs(y) == 2 and abs(x) == 2)
    ]
    object_colors = [0, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15]
    center_colors = [4, 6, 7, 8, 9, 11, 12, 13, 14, 15]

    # Selected and unselected anchor silhouettes.
    wa = np.zeros((2, 16, 5, 5), np.float32)
    for y, x in anchor_border:
        wa[0, 0, y + 2, x + 2] = 1
        wa[1, 3, y + 2, x + 2] = 1
    anchor_exterior = relu(bias(conv("x0", wa), -11.0))
    wc = np.zeros((1, 16), np.float32)
    wc[0, center_colors] = 1
    center_ok = linear("x0", wc)
    anchor_selected = mul(sl(anchor_exterior, 0, 1), center_ok)
    anchor_other = mul(sl(anchor_exterior, 1, 2), center_ok)
    anchors = clip(add(anchor_selected, anchor_other))

    # Round objects and their color counts.
    wr = np.zeros((1, 16, 5, 5), np.float32)
    wp = np.zeros((16, 16, 5, 5), np.float32)
    for y, x in round_shape:
        wr[0, object_colors, y + 2, x + 2] = 1
        for c in range(16):
            wp[c, c, y + 2, x + 2] = 1
    rounds = relu(bias(conv("x0", wr), -20.0))
    pigment_counts = conv("x0", wp)
    color_filter = np.ones(16, np.float32)
    color_filter[[0, 6]] = 0
    pigment = mul(clip(pigment_counts), constant(color_filter))
    dominant_pigment = mul(
        threshold(pigment_counts, 11), constant(color_filter)
    )
    six_plane = sl("x0", 6, 7)

    # Dashed goal rings.
    left_offsets = [
        (-3, -1), (-2, -2), (-1, -3),
        (1, -3), (2, -2), (3, -1)
    ]
    right_offsets = [(y, -x) for y, x in left_offsets]
    wg = np.zeros((32, 16, 7, 7), np.float32)
    for c in range(16):
        for y, x in left_offsets:
            wg[c, c, y + 3, x + 3] = 1
        for y, x in right_offsets:
            wg[16 + c, c, y + 3, x + 3] = 1
    ring_counts = conv("x0", wg)
    ring_exact = relu(bias(ring_counts, -5.0))
    nonblack = np.ones(16, np.float32)
    nonblack[0] = 0
    goal_left = mul(sl(ring_exact, 0, 16), constant(nonblack))
    goal_right = mul(sl(ring_exact, 16, 32), constant(nonblack))
    goal_valid = mul(
        reduce_channels(goal_left, 16),
        reduce_channels(goal_right, 16)
    )
    goal_colors = emit("max2", [goal_left, goal_right])
    goal_size = reduce_channels(goal_colors, 16)

    black_count = add(sl(ring_counts, 0, 1), sl(ring_counts, 16, 17))
    black_ring = threshold(black_count, 8)
    completed_area = conv(black_ring, np.ones((1, 1, 7, 7)))
    completed_area = clip(
        conv(completed_area, np.ones((1, 1, 5, 5)))
    )

    # Extract paintable-ball anchors once. These all have center color 4.
    black_anchors = mul(anchors, sl("x0", 4, 5))
    remaining = black_anchors
    slot_masks = []
    ys, xs, exists = [], [], []
    for _ in range(8):
        slot = first(remaining)
        slot_masks.append(slot)
        position = gather(slot, pixel)
        ys.append(vsl(position, 0, 1))
        xs.append(vsl(position, 1, 2))
        exists.append(gmax(slot))
        remaining = relu(sub(remaining, slot))
    slots_y = vcat(*ys)
    slots_x = vcat(*xs)
    slots_exist = vcat(*exists)
    anchor_slots = cat(*slot_masks)

    bits = (
        (np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1
    ).astype(np.float32)
    counts = bits.sum(axis=1)
    count_vector = constant(counts)
    slack = constant(counts - 1.0)
    sum_y = dense(slots_y, bits)
    sum_x = dense(slots_x, bits)
    sum_exists = dense(slots_exist, bits)
    missing_members = vminus(count_vector, sum_exists)
    allowed_subset = constant((counts >= 2).astype(np.float32))

    def centroid_error(position):
        cy = vsl(position, 0, 1)
        cx = vsl(position, 1, 2)
        ncy = vm(count_vector, cy)
        ncx = vm(count_vector, cx)
        ey0 = vr(vminus(ncy, sum_y))
        ey1 = vr(vminus(vminus(sum_y, ncy), slack))
        ex0 = vr(vminus(ncx, sum_x))
        ex1 = vr(vminus(vminus(sum_x, ncx), slack))
        return va(va(ey0, ey1), va(ex0, ex1))

    # Validate round objects against actual anchor centroids. This separates
    # paint pickups from paintable balls without relying on nearby line pixels.
    perception_id = fresh()
    candidates_state = fresh()
    real_balls_state = fresh()
    perception_body = []
    circle_features = cat(pixel, six_plane)

    where[0] = perception_body
    candidate = first(candidates_state)
    candidate_data = gather(candidate, circle_features)
    candidate_position = vsl(candidate_data, 0, 2)
    candidate_regular = vsl(candidate_data, 2, 3)
    candidate_error = centroid_error(candidate_position)
    candidate_validity = vclip(
        vb(
            va(vk(candidate_error, -4.0),
               vk(missing_members, -4.0)),
            2.0
        )
    )
    candidate_validity = vm(candidate_validity, allowed_subset)
    candidate_black = vclip(vsum(candidate_validity, 256))
    candidate_is_ball = vclip(va(candidate_black, candidate_regular))
    found_ball = mul(candidate, candidate_is_ball)
    found_balls = emit("max2", [real_balls_state, found_ball])
    remaining_candidates = relu(sub(candidates_state, candidate))
    where[0] = graph
    graph.append({
        "id": perception_id,
        "op": "loop",
        "T": 16,
        "state": [
            {"id": candidates_state, "init": rounds},
            {"id": real_balls_state, "zeros": [1, 64, 64]},
        ],
        "body": perception_body,
        "next": {
            candidates_state: remaining_candidates,
            real_balls_state: found_balls,
        },
    })
    balls = perception_id + "." + real_balls_state
    unsolved_balls = mul(balls, complement(completed_area))

    active_position = gather(anchor_selected, pixel)
    old_position = vsl("mem", 0, 2)
    old_valid = vsl("mem", 2, 3)
    reference_position = va(
        vm(old_position, old_valid),
        vm(active_position, vminus(one_v, old_valid))
    )
    focus = choose(
        scale(distance_to(reference_position), -1.0), unsolved_balls
    )
    has_focus = gmax(focus)
    ball_position = gather(focus, pixel)
    regular = gather(focus, six_plane)
    black_mode = vminus(one_v, regular)

    # Majority color protects ordinary-ball identity near another object.
    ball_colors = va(
        vm(gather(focus, dominant_pigment), regular),
        vm(gather(focus, pigment), black_mode)
    )
    ball_color_count = vsum(ball_colors, 16)

    # Select the paintable ball's anchor subset using a fixed tournament.
    error = centroid_error(ball_position)
    tie = np.arange(256, dtype=np.float32) * 0.001
    tie[counts < 2] -= 1.0e6
    subset_score = va(
        va(vk(error, -10000.0),
           vk(missing_members, -1000000.0)),
        constant(tie - counts)
    )

    tournament_parameters = {}

    def vector_max(x, length):
        while length > 1:
            if length not in tournament_parameters:
                half = length // 2
                wd = np.zeros((half, length), np.float32)
                wb = np.zeros((half, length), np.float32)
                r = np.arange(half)
                wd[r, 2 * r] = 1
                wd[r, 2 * r + 1] = -1
                wb[r, 2 * r + 1] = 1
                tournament_parameters[length] = (
                    parameter(wd), parameter(wb)
                )
            pd, pb = tournament_parameters[length]
            difference = emit("dense", [x], w=pd)
            base = emit("dense", [x], w=pb)
            x = va(vr(difference), base)
            length //= 2
        return x

    top_subset_score = vector_max(subset_score, 256)
    subset_peaks = vr(
        vb(vk(vminus(subset_score, top_subset_score), 10000.0), 1.0)
    )
    subset_ordered = va(
        vk(subset_peaks, 1024.0),
        constant(np.arange(256, dtype=np.float32))
    )
    subset_top = vector_max(subset_ordered, 256)
    subset_choice = vr(
        vb(vk(vminus(subset_ordered, subset_top), 32.0), 1.0)
    )
    membership = dense(subset_choice, bits.T)
    paint_group = reduce_channels(mul(anchor_slots, membership), 8)

    tag_match = reduce_channels(mul("x0", ball_colors), 16)
    regular_group = mul(anchors, tag_match)
    group_anchors = add(
        mul(paint_group, black_mode),
        mul(regular_group, regular)
    )
    group_sum = vk(gmean(mul(group_anchors, pixel)), 4096.0)
    group_count = vk(gmean(group_anchors), 4096.0)

    # Exact reciprocal values at integral counts, represented by a ReLU spline.
    knots = np.arange(16, dtype=np.float32)
    recip_features = vr(
        dense(group_count, np.ones((16, 1)), -knots)
    )
    values = np.array(
        [1.0] + [1.0 / k for k in range(1, 17)], np.float32
    )
    slopes = np.diff(values)
    coefficients = np.empty(16, np.float32)
    coefficients[0] = slopes[0]
    coefficients[1:] = slopes[1:] - slopes[:-1]
    inverse_n = dense(recip_features, coefficients[None, :], [1.0])

    # Only validated non-ball circles are pigment pickups.
    pickup_pool = mul(rounds, complement(balls))
    pickup_pool = mul(pickup_pool, complement(completed_area))
    pickup_colors_available = gather(pickup_pool, pigment)
    available_colors = vclip(va(ball_colors, pickup_colors_available))

    overlap = reduce_channels(mul(goal_colors, ball_colors), 16)
    available_overlap = reduce_channels(
        mul(goal_colors, available_colors), 16
    )
    mismatch = sub(ball_color_count, overlap)
    unavailable_count = sub(goal_size, available_overlap)
    double_goals = mul(goal_valid, threshold(goal_size, 2))
    has_double_goal = gmax(double_goals)
    desired_goal_size = va(
        vm(regular, ball_color_count),
        vm(black_mode, va(one_v, has_double_goal))
    )
    right_size = clip(
        sub(one_s, absolute(sub(goal_size, desired_goal_size)))
    )
    candidate_goals = mul(goal_valid, right_size)
    goal_score = add(
        scale(distance_to(ball_position), -0.01),
        add(scale(mismatch, -1000.0),
            scale(unavailable_count, -1000.0))
    )
    goal = choose(goal_score, candidate_goals)
    goal_position = gather(goal, pixel)
    chosen_goal_colors = gather(goal, goal_colors)
    missing_colors = vr(vminus(chosen_goal_colors, ball_colors))
    needs_paint = vclip(vsum(missing_colors, 16))

    useful_pickup = threshold(
        reduce_channels(mul(pigment, missing_colors), 16), 1
    )
    useful_pickup = mul(useful_pickup, pickup_pool)
    pickup = choose(
        scale(distance_to(ball_position), -1.0), useful_pickup
    )
    pickup_position = gather(pickup, pixel)
    target = va(
        vm(pickup_position, needs_paint),
        vm(goal_position, vminus(one_v, needs_paint))
    )

    # Terrain and object-separation constraints.
    wall = sl("x0", 2, 3)
    hazard = sl("x0", 10, 11)
    obstacle = add(wall, hazard)
    body_kernel = np.zeros((1, 1, 5, 5), np.float32)
    for y, x in round_shape:
        body_kernel[0, 0, y + 2, x + 2] = 1
    terrain_bad = clip(conv(obstacle, body_kernel))
    outside = add(
        add(relu(bias(scale(py, -1), 2)), relu(bias(py, -61))),
        add(relu(bias(scale(px, -1), 2)), relu(bias(px, -61)))
    )

    other_balls = relu(sub(balls, focus))
    protected_objects = add(other_balls, anchors)
    separation_bad = conv(
        protected_objects, np.ones((1, 1, 7, 7))
    )
    separation_bad = clip(
        conv(separation_bad, np.ones((1, 1, 5, 5)))
    )
    ball_bad = clip(add(add(terrain_bad, outside), separation_bad))
    ball_safe = complement(ball_bad)

    target_distance = distance_to(target)
    target_seed = scale(relu(sub(one_s, target_distance)), 100.0)

    pool_params = {}

    def pool_axis(x, axis):
        if axis not in pool_params:
            w = np.zeros((7, 1, 7, 7), np.float32)
            for i in range(7):
                if axis == 0:
                    w[i, 0, i, 3] = 1
                else:
                    w[i, 0, 3, i] = 1
            pool_params[axis] = parameter(w)
        shifted = emit("conv", [x], w=pool_params[axis])
        parts = [sl(shifted, i, i + 1) for i in range(7)]
        z = parts[0]
        for p in parts[1:]:
            z = emit("max2", [z, p])
        return z

    potential_id = fresh()
    potential_state = fresh()
    potential_body = []
    where[0] = potential_body
    propagated = potential_state
    for axis in (0, 0, 0, 0, 1, 1, 1, 1):
        propagated = pool_axis(propagated, axis)
    propagated = mul(relu(bias(propagated, -1.0)), ball_safe)
    propagated = emit("max2", [propagated, target_seed])
    where[0] = graph
    graph.append({
        "id": potential_id,
        "op": "loop",
        "T": 20,
        "state": [{"id": potential_state, "init": target_seed}],
        "body": potential_body,
        "next": {potential_state: propagated},
    })
    potential = potential_id + "." + potential_state

    from_ball = absolute(sub(pixel, ball_position))
    within_y = clip(bias(scale(sl(from_ball, 0, 1), -1.0), 13.0))
    within_x = clip(bias(scale(sl(from_ball, 1, 2), -1.0), 13.0))
    guide_mask = mul(ball_safe, mul(within_y, within_x))
    guide_score = add(
        scale(potential, 1000.0),
        add(
            scale(target_distance, -1.0),
            scale(reduce_channels(from_ball, 2), -0.1)
        )
    )
    guide = choose(guide_score, guide_mask)
    guide_position = gather(guide, pixel)

    backwards = vminus(ball_position, guide_position)
    anchor_score = reduce_channels(mul(pixel, backwards), 2)
    anchor_score = add(anchor_score, scale(anchor_selected, 0.25))
    chosen_anchor = choose(anchor_score, group_anchors)
    chosen_anchor_position = gather(chosen_anchor, pixel)
    already_selected = gmax(mul(chosen_anchor, anchor_selected))
    other_sum = vminus(group_sum, chosen_anchor_position)

    raw_predicted = add(
        mul(pixel, inverse_n), vm(other_sum, inverse_n)
    )

    # Floor the predicted centroid. Small plateaus absorb float32 coordinate
    # roundoff while preserving all integer arithmetic cases.
    floor_w = np.zeros((126, 2, 1, 1), np.float32)
    floor_b = np.empty(126, np.float32)
    floor_reduce = np.zeros((2, 126), np.float32)
    for axis in range(2):
        for k in range(1, 64):
            i = axis * 63 + k - 1
            floor_w[i, axis, 0, 0] = 10000.0
            floor_b[i] = -10000.0 * k + 2.0
            floor_reduce[axis, i] = 1.0
    floor_steps = clip(conv(raw_predicted, floor_w, floor_b))
    predicted = linear(floor_steps, floor_reduce)
    predicted_y = sl(predicted, 0, 1)
    predicted_x = sl(predicted, 1, 2)

    offsets = -np.arange(64, dtype=np.float32)

    def tent_bank(coordinate):
        d = conv(coordinate, np.ones((64, 1, 1, 1)), offsets)
        return relu(sub(one_s, absolute(d)))

    row_basis = tent_bank(py)
    col_basis = tent_bank(px)
    row_weights = tent_bank(predicted_y)
    col_weights = tent_bank(predicted_x)

    def warp(source):
        rows = mul(source, row_basis)
        rows = emit("colsum", [rows])
        rows = mul(rows, row_weights)
        intermediate = reduce_channels(rows, 64)
        columns = mul(intermediate, col_basis)
        columns = emit("rowsum", [columns])
        columns = mul(columns, col_weights)
        return reduce_channels(columns, 64)

    future_potential = warp(potential)
    future_bad = warp(ball_bad)

    anchor_kernel = np.zeros((1, 1, 5, 5), np.float32)
    for y, x in diamond:
        anchor_kernel[0, 0, y + 2, x + 2] = 1
    anchor_bad = clip(conv(wall, anchor_kernel))
    anchor_bad = clip(add(anchor_bad, outside))
    near_anchors = clip(conv(anchors, np.ones((1, 1, 7, 7))))
    near_rounds = conv(rounds, np.ones((1, 1, 7, 7)))
    near_rounds = clip(conv(near_rounds, np.ones((1, 1, 3, 3))))
    legal_click = mul(complement(anchor_bad), complement(near_anchors))
    legal_click = mul(legal_click, complement(near_rounds))

    # The moved anchor must remain disjoint from its own new ball.
    separation = absolute(sub(predicted, pixel))
    overlap_y = clip(bias(scale(sl(separation, 0, 1), -1.0), 5.0))
    overlap_x = clip(bias(scale(sl(separation, 1, 2), -1.0), 5.0))
    overlap_penalty = mul(overlap_y, overlap_x)

    future_guide_distance = reduce_channels(
        absolute(sub(predicted, guide_position)), 2
    )
    future_target_distance = reduce_channels(
        absolute(sub(predicted, target)), 2
    )
    destination_score = add(
        scale(future_potential, 1000.0),
        add(
            scale(add(future_bad, overlap_penalty), -100000.0),
            add(
                scale(future_guide_distance, -1.0),
                scale(future_target_distance, -0.1)
            )
        )
    )
    destination = choose(destination_score, legal_click)

    selection_click = mul(
        chosen_anchor, vminus(one_v, already_selected)
    )
    movement_click = mul(destination, already_selected)
    final_click = add(selection_click, movement_click)
    emit("add", [final_click, constant(0.0)], out="click")

    expected_position = gather(destination, predicted)
    next_position = va(
        vm(expected_position, already_selected),
        vm(ball_position, vminus(one_v, already_selected))
    )
    emit("vconcat", [next_position, has_focus], out="mem_next")

    # If no controllable ball can be recovered, prefer a level reset when
    # RESET is available rather than repeatedly clicking an empty location.
    logit_input = vcat(has_focus, one_v)
    logit_w = np.zeros((8, 2), np.float32)
    logit_w[0] = [-150.0, 50.0]
    logit_w[6, 1] = 10.0
    dense(logit_input, logit_w, out="logits")

    assert sum(a.size for a in params.values()) <= 150000
    assert all(np.isfinite(a).all() for a in params.values())

    return {
        "graph": graph,
        "memory_size": 3,
        "params": params,
    }
