import numpy as np


def build():
    graph = []
    params = {}
    seq = [0]

    def node(op, inputs, **fields):
        name = "n%d" % seq[0]
        seq[0] += 1
        d = {"id": name, "op": op, "in": inputs}
        d.update(fields)
        graph.append(d)
        return name

    def param(value):
        name = "p%d" % len(params)
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def conv(x, w, b=None):
        kw = {"w": param(w)}
        if b is not None:
            kw["b"] = param(b)
        return node("conv", [x], **kw)

    def affine(x, scale, bias=0.0, channels=1):
        w = np.zeros((channels, channels, 1, 1), dtype=np.float32)
        for i in range(channels):
            w[i, i, 0, 0] = scale
        return conv(x, w, np.full(channels, bias, dtype=np.float32))

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def mn(a, b):
        return node("min2", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def absolute(a):
        return node("abs", [a])

    def relu(a):
        return node("relu", [a])

    one = sl("coords", 2, 3)
    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)

    def clip(x):
        return mn(relu(x), one)

    def threshold(x, t=0.5, gain=1000.0, channels=1):
        return clip(affine(x, gain, -gain * t, channels))

    def inverse(x):
        return sub(one, x)

    def constant(x):
        return affine(one, 0.0, x)

    def gmax_map(x):
        return mul(one, node("gmax", [x]))

    def sum_vector(x):
        v = node("gmean", [x])
        return node(
            "dense", [v],
            w=param(np.array([[4096.0]], dtype=np.float32))
        )

    def gsum_map(x):
        return mul(one, sum_vector(x))

    def present(x):
        return threshold(gmax_map(x), 0.5, 2.0)

    def coord(mask, axis):
        return gsum_map(mul(mask, axis))

    def dilate(x, k):
        return clip(conv(
            x, np.ones((1, 1, k, k), dtype=np.float32)
        ))

    def choose(score, mask):
        masked = sub(score, affine(inverse(mask), 10.0))
        top = gmax_map(masked)
        equal = clip(affine(sub(masked, top), 1000000.0, 1.0))
        return mul(equal, mask)

    def distance(r1, c1, r2, c2):
        return add(absolute(sub(r1, r2)), absolute(sub(c1, c2)))

    def near(r, c, radius):
        d = distance(row, col, r, c)
        return inverse(threshold(
            d, (radius + 0.25) / 63.0, 10000.0
        ))

    def centroid(mask):
        features = node("concat", [
            mask, mul(mask, row), mul(mask, col)
        ])
        mean = node("gmean", [features])

        count_w = np.zeros((8, 3), dtype=np.float32)
        count_w[:, 0] = 4096.0
        count_b = -np.arange(8, dtype=np.float32)
        hinges = node(
            "dense", [mean], w=param(count_w), b=param(count_b)
        )
        hinges = node("vrelu", [hinges])

        slopes = [1.0, -0.5]
        for n in range(2, 8):
            slopes.append(1.0 / (n + 1) - 1.0 / n)
        coeff = [slopes[0]]
        for i in range(1, 8):
            coeff.append(slopes[i] - slopes[i - 1])
        reciprocal = node(
            "dense", [hinges],
            w=param(np.array([coeff], dtype=np.float32))
        )
        sums = node(
            "dense", [mean],
            w=param(np.array([
                [0.0, 4096.0, 0.0],
                [0.0, 0.0, 4096.0],
            ], dtype=np.float32))
        )
        average = node("vmul", [sums, reciprocal])
        avg_r = mul(one, vsl(average, 0, 1))
        avg_c = mul(one, vsl(average, 1, 2))
        allow_r = threshold(sub(avg_r, row), -0.0001, 20000.0)
        allow_c = threshold(sub(avg_c, col), -0.0001, 20000.0)
        center_r = gmax_map(mul(row, allow_r))
        center_c = gmax_map(mul(col, allow_c))
        count = node(
            "dense", [mean],
            w=param(np.array([[4096.0, 0.0, 0.0]], dtype=np.float32))
        )
        return mul(one, count), center_r, center_c

    # Controller templates.
    diamond = [
        (0, 2),
        (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3),
        (4, 2),
    ]
    outline_w = np.zeros((2, 16, 5, 5), dtype=np.float32)
    for y, x in diamond:
        outline_w[0, 0, y, x] = 1.0
        outline_w[1, 3, y, x] = 1.0
    outlines = conv("x0", outline_w)
    selected_outline = threshold(sl(outlines, 0, 1), 8.5, 2.0)
    unselected_outline = threshold(sl(outlines, 1, 2), 8.5, 2.0)
    any_outline = clip(add(selected_outline, unselected_outline))

    c14 = sl("x0", 14, 15)
    c15 = sl("x0", 15, 16)
    c4 = sl("x0", 4, 5)
    active14 = mul(selected_outline, c14)
    active15 = mul(selected_outline, c15)
    active4 = mul(selected_outline, c4)
    ctl14 = mul(any_outline, c14)
    ctl15 = mul(any_outline, c15)
    ctl4 = mul(any_outline, c4)
    active = clip(add(add(active14, active15), active4))
    all_ctl = clip(add(add(ctl14, ctl15), ctl4))
    is14 = present(active14)
    is15 = present(active15)
    is4 = present(active4)
    active_r = coord(active, row)
    active_c = coord(active, col)

    # Body and collectible templates.
    disk = [
        (y, x)
        for y in range(5)
        for x in range(5)
        if not (y in (0, 4) and x in (0, 4))
    ]
    body_w = np.zeros((3, 16, 5, 5), dtype=np.float32)
    for y, x in disk:
        if (y, x) == (2, 2):
            body_w[0, 6, y, x] = 1.0
            body_w[1, 6, y, x] = 1.0
        else:
            body_w[0, 14, y, x] = 1.0
            body_w[1, 15, y, x] = 1.0
        for color in (0, 8, 9, 11, 14):
            body_w[2, color, y, x] = 1.0
    body_raw = conv("x0", body_w)
    body14 = threshold(sl(body_raw, 0, 1), 20.5, 2.0)
    body15 = threshold(sl(body_raw, 1, 2), 20.5, 2.0)
    body4 = threshold(sl(body_raw, 2, 3), 20.5, 2.0)

    patterns = [
        [
            [-1, 9, 0, 0, -1],
            [9, 9, 0, 0, 0],
            [9, 9, 0, 0, 0],
            [9, 9, 9, 0, 0],
            [-1, 9, 9, 0, -1],
        ],
        [
            [-1, 0, 8, 8, -1],
            [0, 0, 8, 8, 8],
            [0, 0, 8, 8, 8],
            [0, 0, 0, 8, 8],
            [-1, 0, 0, 8, -1],
        ],
        [
            [-1, 14, 0, 0, -1],
            [14, 14, 0, 0, 0],
            [14, 14, 0, 0, 0],
            [14, 14, 14, 0, 0],
            [-1, 14, 14, 0, -1],
        ],
        [
            [-1, 0, 11, 11, -1],
            [0, 0, 11, 11, 11],
            [0, 0, 11, 11, 11],
            [0, 0, 0, 11, 11],
            [-1, 0, 0, 11, -1],
        ],
    ]
    item_w = np.zeros((4, 16, 5, 5), dtype=np.float32)
    for i, pattern in enumerate(patterns):
        for y in range(5):
            for x in range(5):
                c = pattern[y][x]
                if c >= 0:
                    item_w[i, c, y, x] = 1.0
    item_raw = conv("x0", item_w)
    items = [
        threshold(sl(item_raw, i, i + 1), 20.5, 2.0)
        for i in range(4)
    ]

    ring = [
        (0, 2), (0, 4),
        (1, 1), (1, 5),
        (2, 0), (2, 6),
        (4, 0), (4, 6),
        (5, 1), (5, 5),
        (6, 2), (6, 4),
    ]
    goal_w = np.zeros((4, 16, 7, 7), dtype=np.float32)
    for y, x in ring:
        goal_w[0, 14, y, x] = 1.0
        goal_w[1, 15, y, x] = 1.0
        goal_w[2, 9 if x < 3 else 8, y, x] = 1.0
        goal_w[3, 14 if x < 3 else 11, y, x] = 1.0
    goal_raw = conv("x0", goal_w)
    goals = [
        threshold(sl(goal_raw, i, i + 1), 7.5, 2.0)
        for i in range(4)
    ]

    # Connector flood is a fallback when centroid matching is obscured.
    line = sl("x0", 1, 2)
    line_near = dilate(line, 7)
    black_near_line = mul(body4, line_near)
    all_bodies = clip(add(add(body14, body15), black_near_line))
    passable = clip(add(
        add(line, dilate(all_ctl, 7)),
        dilate(all_bodies, 7),
    ))
    flood_w = param(np.ones((1, 1, 3, 3), dtype=np.float32))
    graph.append({
        "id": "Flood",
        "op": "loop",
        "T": 128,
        "state": [{"id": "fstate", "init": active}],
        "body": [
            {"id": "fconv", "op": "conv",
             "in": ["fstate"], "w": flood_w},
            {"id": "fclip", "op": "min2", "in": ["fconv", one]},
            {"id": "fnext", "op": "min2",
             "in": ["fclip", passable]},
        ],
        "next": {"fstate": "fnext"},
    })
    reached = "Flood.fstate"

    # Enumerate the other visible black controllers in a deterministic
    # order. Candidate groups contain the active controller plus one
    # or two of these controllers.
    remaining = mul(ctl4, inverse(active4))
    order_score = sub(affine(row, -1.0), affine(col, 0.01))
    handles = []
    handle_r = []
    handle_c = []
    handle_present = []
    for _ in range(4):
        h = choose(order_score, remaining)
        handles.append(h)
        handle_r.append(sum_vector(mul(h, row)))
        handle_c.append(sum_vector(mul(h, col)))
        handle_present.append(node("gmax", [h]))
        remaining = mul(remaining, inverse(h))

    active_r_vec = node("gmax", [active_r])
    active_c_vec = node("gmax", [active_c])
    positions = node("vconcat", [
        active_r_vec, active_c_vec,
        handle_r[0], handle_c[0],
        handle_r[1], handle_c[1],
        handle_r[2], handle_c[2],
        handle_r[3], handle_c[3],
    ])
    presences = node("vconcat", handle_present)

    candidates = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3),
        (0,), (1,), (2,), (3,),
    ]
    ncan = len(candidates)
    wr = np.zeros((ncan, 10), dtype=np.float32)
    wc = np.zeros((ncan, 10), dtype=np.float32)
    valid_w = np.zeros((ncan, 4), dtype=np.float32)
    valid_b = np.zeros(ncan, dtype=np.float32)
    membership = np.zeros((4, ncan), dtype=np.float32)
    for k, subset in enumerate(candidates):
        count = 1 + len(subset)
        wr[k, 0] = 1.0 / count
        wc[k, 1] = 1.0 / count
        for j in subset:
            wr[k, 2 + 2 * j] = 1.0 / count
            wc[k, 3 + 2 * j] = 1.0 / count
            valid_w[k, j] = 2.0
            membership[j, k] = 1.0
        valid_b[k] = 1.0 - 2.0 * len(subset)

    candidate_r = node("dense", [positions], w=param(wr))
    candidate_c = node("dense", [positions], w=param(wc))
    valid = node(
        "dense", [presences], w=param(valid_w), b=param(valid_b)
    )
    valid = node("vrelu", [valid])
    candidate_r_map = mul(one, candidate_r)
    candidate_c_map = mul(one, candidate_c)

    # A body's position is the componentwise floor of its centroid.
    rdiff = sub(candidate_r_map, row)
    cdiff = sub(candidate_c_map, col)
    rlo = threshold(rdiff, -0.0001, 20000.0, ncan)
    clo = threshold(cdiff, -0.0001, 20000.0, ncan)
    rhi = threshold(
        affine(rdiff, -1.0, channels=ncan),
        -1.0 / 63.0 + 0.0001, 20000.0, ncan
    )
    chi = threshold(
        affine(cdiff, -1.0, channels=ncan),
        -1.0 / 63.0 + 0.0001, 20000.0, ncan
    )
    candidate_cells = mul(mul(rlo, rhi), mul(clo, chi))
    candidate_hits = node("gmax", [mul(candidate_cells, body4)])
    candidate_hits = node("vmul", [candidate_hits, valid])

    # Select the first matching subset.
    prefix_w = np.zeros((ncan, ncan), dtype=np.float32)
    for i in range(ncan):
        for j in range(i):
            prefix_w[i, j] = -1.0
    first_gate = node(
        "dense", [candidate_hits],
        w=param(prefix_w), b=param(np.ones(ncan, dtype=np.float32))
    )
    first_gate = node("vrelu", [first_gate])
    selected_candidate = node("vmul", [candidate_hits, first_gate])
    member_vec = node(
        "dense", [selected_candidate], w=param(membership)
    )
    matched_vec = node(
        "dense", [selected_candidate],
        w=param(np.ones((1, ncan), dtype=np.float32))
    )
    matched = mul(one, matched_vec)

    matched_group = active4
    for i in range(4):
        matched_group = add(
            matched_group, mul(handles[i], vsl(member_vec, i, i + 1))
        )
    matched_group = clip(matched_group)
    flood_group = clip(add(active4, mul(ctl4, reached)))
    black_group = clip(add(
        mul(matched_group, matched),
        mul(flood_group, inverse(matched)),
    ))

    group = clip(add(
        add(mul(ctl14, is14), mul(ctl15, is15)),
        mul(black_group, is4),
    ))
    group_count, body_r, body_c = centroid(group)
    group_exists = present(group)

    _, green_r, green_c = centroid(ctl14)
    _, purple_r, purple_c = centroid(ctl15)
    green_done = present(mul(goals[0], near(green_r, green_c, 3)))
    purple_done = present(mul(goals[1], near(purple_r, purple_c, 3)))
    green_needed = mul(present(goals[0]), inverse(green_done))
    purple_needed = mul(present(goals[1]), inverse(purple_done))

    # Inspect collectible colors at the selected body's centroid.
    center = inverse(threshold(
        distance(row, col, body_r, body_c), 0.001, 2000.0
    ))
    disk_w = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for y, x in disk:
        disk_w[0, 0, y, x] = 1.0
    flags = []
    for color in (9, 8, 14, 11):
        plane = sl("x0", color, color + 1)
        local = conv(plane, disk_w)
        flags.append(present(mul(center, threshold(local, 0.5, 2.0))))
    f9, f8, f14, f11 = flags
    empty = inverse(clip(add(add(f9, f8), add(f14, f11))))

    want9 = clip(add(
        empty,
        mul(mul(f8, inverse(f9)), mul(inverse(f14), inverse(f11))),
    ))
    want8 = clip(add(
        empty,
        mul(mul(f9, inverse(f8)), mul(inverse(f14), inverse(f11))),
    ))
    want14 = clip(add(
        empty,
        mul(mul(f11, inverse(f14)), mul(inverse(f9), inverse(f8))),
    ))
    want11 = clip(add(
        empty,
        mul(mul(f14, inverse(f11)), mul(inverse(f9), inverse(f8))),
    ))
    pair98 = mul(mul(f9, f8), mul(inverse(f14), inverse(f11)))
    pair1411 = mul(mul(f14, f11), mul(inverse(f9), inverse(f8)))

    item_targets = clip(add(
        add(mul(items[0], want9), mul(items[1], want8)),
        add(mul(items[2], want14), mul(items[3], want11)),
    ))
    item_targets = mul(item_targets, inverse(near(body_r, body_c, 2)))

    black_goals = clip(add(
        mul(goals[2], pair98),
        mul(goals[3], pair1411),
    ))
    colored_goals = clip(add(
        mul(goals[0], is14),
        mul(goals[1], is15),
    ))
    matching_goals = clip(add(colored_goals, mul(black_goals, is4)))
    at_goal = present(mul(matching_goals, near(body_r, body_c, 3)))
    matching_goals = mul(matching_goals, inverse(at_goal))
    targets = clip(add(matching_goals, mul(item_targets, is4)))
    target_exists = mul(present(targets), group_exists)

    target_score = affine(distance(row, col, body_r, body_c), -1.0)
    target_score = sub(target_score, affine(row, 0.0002))
    target_score = sub(target_score, affine(col, 0.0001))
    target = choose(target_score, targets)
    target_r = coord(target, row)
    target_c = coord(target, col)

    # Select the controller with the most useful movement range.
    dr = mul(group_count, sub(target_r, body_r))
    dc = mul(group_count, sub(target_c, body_c))
    low = constant(2.0 / 63.0)
    high = constant(61.0 / 63.0)
    ideal_r = add(row, dr)
    ideal_c = add(col, dc)
    clamped_r = mn(mx(ideal_r, low), high)
    clamped_c = mn(mx(ideal_c, low), high)
    remainder = add(
        absolute(sub(ideal_r, clamped_r)),
        absolute(sub(ideal_c, clamped_c)),
    )
    controller_score = affine(remainder, -1.0)
    controller_score = add(controller_score, affine(active, 0.001))
    controller_score = sub(controller_score, affine(row, 0.00002))
    controller_score = sub(controller_score, affine(col, 0.00001))

    old_r = vsl("mem", 0, 1)
    old_c = vsl("mem", 1, 2)
    pending = vsl("mem", 2, 3)
    old_failed = vsl("mem", 3, 4)
    same_active = inverse(threshold(
        add(absolute(sub(active_r, old_r)),
            absolute(sub(active_c, old_c))),
        0.001, 2000.0,
    ))
    failed_now = mul(same_active, pending)
    succeeded_now = mul(inverse(same_active), pending)
    failed_memory = mul(
        clip(add(mul(one, old_failed), failed_now)),
        inverse(succeeded_now),
    )
    failed_location = inverse(threshold(
        add(absolute(sub(row, old_r)), absolute(sub(col, old_c))),
        0.001, 2000.0,
    ))
    multiple = threshold(group_count, 1.5, 2.0)
    controller_score = sub(
        controller_score,
        affine(mul(mul(failed_location, failed_memory), multiple), 5.0),
    )
    best_controller = choose(controller_score, group)

    other_controllers = mul(all_ctl, inverse(group))
    other_score = add(
        add(mul(ctl14, green_needed), mul(ctl15, purple_needed)),
        affine(ctl4, 0.8),
    )
    other_score = sub(other_score, affine(row, 0.0002))
    other_score = sub(other_score, affine(col, 0.0001))
    best_other = choose(other_score, other_controllers)

    active_is_best = present(mul(best_controller, active))
    move_flag = mul(target_exists, active_is_best)
    selection = add(
        mul(best_controller, target_exists),
        mul(best_other, inverse(target_exists)),
    )

    # Choose a reachable body waypoint with a margin around hazards.
    waypoint_controller_r = add(
        active_r, mul(group_count, sub(row, body_r))
    )
    waypoint_controller_c = add(
        active_c, mul(group_count, sub(col, body_c))
    )

    def in_bounds(x):
        lo = threshold(x, 2.0 / 63.0 - 0.0001, 10000.0)
        hi = threshold(
            affine(x, -1.0), -61.0 / 63.0 - 0.0001, 10000.0
        )
        return mul(lo, hi)

    reachable = mul(
        in_bounds(waypoint_controller_r),
        in_bounds(waypoint_controller_c),
    )

    hazard = sl("x0", 10, 11)
    hazard_near = conv(
        hazard, np.ones((1, 1, 3, 3), dtype=np.float32)
    )
    hidden_hazard = mul(line, threshold(hazard_near, 2.5, 2.0))
    hazard = clip(add(hazard, hidden_hazard))
    expanded_hazard = dilate(hazard, 3)
    body_hazard = conv(expanded_hazard, disk_w)
    safe = inverse(threshold(body_hazard, 0.5, 2.0))

    # Do not move a body over a stationary controller.
    stationary = mul(all_ctl, inverse(active))
    stationary_exclusion = dilate(dilate(stationary, 7), 5)
    safe = mul(safe, inverse(stationary_exclusion))
    waypoint_mask = mul(reachable, safe)

    waypoint_score = affine(
        distance(row, col, target_r, target_c), -1.0
    )
    waypoint_score = sub(waypoint_score, affine(row, 0.0002))
    waypoint_score = sub(waypoint_score, affine(col, 0.0001))
    waypoint = choose(waypoint_score, waypoint_mask)
    waypoint_r = coord(waypoint, row)
    waypoint_c = coord(waypoint, col)
    desired_r = add(active_r, mul(group_count, sub(waypoint_r, body_r)))
    desired_c = add(active_c, mul(group_count, sub(waypoint_c, body_c)))

    # Legal controller placement: avoid walls and keep all controller
    # sprites separated so that they remain individually detectable.
    wall = sl("x0", 2, 3)
    diamond_w = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for y, x in diamond:
        diamond_w[0, 0, y, x] = 1.0
    diamond_w[0, 0, 2, 2] = 1.0
    wall_hit = conv(wall, diamond_w)
    wall_free = inverse(threshold(wall_hit, 0.5, 2.0))
    controller_exclusion = dilate(dilate(stationary, 7), 5)
    active_box = dilate(active, 5)
    legal = mul(wall_free, inverse(controller_exclusion))
    legal = mul(legal, inverse(active_box))
    legal = mul(legal, in_bounds(row))
    legal = mul(legal, in_bounds(col))

    destination_score = affine(
        distance(row, col, desired_r, desired_c), -1.0
    )
    destination_score = sub(destination_score, affine(row, 0.0002))
    destination_score = sub(destination_score, affine(col, 0.0001))
    destination = choose(destination_score, legal)

    click_value = add(
        mul(destination, move_flag),
        mul(selection, inverse(move_flag)),
    )
    graph.append({"id": "click", "op": "relu", "in": [click_value]})

    # Remember the last attempted controller until a move succeeds.
    next_r_map = add(mul(active_r, move_flag),
                     mul(mul(one, old_r), inverse(move_flag)))
    next_c_map = add(mul(active_c, move_flag),
                     mul(mul(one, old_c), inverse(move_flag)))
    next_fail_map = mul(failed_memory, inverse(move_flag))
    next_r = node("gmax", [next_r_map])
    next_c = node("gmax", [next_c_map])
    next_pending = node("gmax", [move_flag])
    next_fail = node("gmax", [next_fail_map])
    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [next_r, next_c, next_pending, next_fail],
    })

    logits_w = param(np.zeros((8, 8), dtype=np.float32))
    logits_b = param(np.array(
        [0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32
    ))
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": ["aprev"],
        "w": logits_w,
        "b": logits_b,
    })

    return {"graph": graph, "memory_size": 4, "params": params}
