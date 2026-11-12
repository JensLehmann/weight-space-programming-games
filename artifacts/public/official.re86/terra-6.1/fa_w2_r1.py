import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]

    def fresh(prefix="n"):
        counter[0] += 1
        return "%s%d" % (prefix, counter[0])

    def param(value):
        name = fresh("p")
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(kind, inputs=None, dest=None, body=None, **fields):
        name = dest or fresh("n")
        nd = {"id": name, "op": kind}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        (graph if body is None else body).append(nd)
        return name

    def conv(x, w, bias=None, dest=None, body=None):
        fields = {"w": param(w)}
        if bias is not None:
            fields["b"] = param(bias)
        return node("conv", [x], dest=dest, body=body, **fields)

    def dense(x, w, bias=None, dest=None, body=None):
        fields = {"w": param(w)}
        if bias is not None:
            fields["b"] = param(bias)
        return node("dense", [x], dest=dest, body=body, **fields)

    def shared_conv(x, w, body=None):
        return node("conv", [x], body=body, w=w)

    def shared_dense(x, w, bias=None, body=None):
        fields = {"w": w}
        if bias is not None:
            fields["b"] = bias
        return node("dense", [x], body=body, **fields)

    def u(kind, x, body=None):
        return node(kind, [x], body=body)

    def b(kind, x, y, body=None):
        return node(kind, [x, y], body=body)

    def sa(x, scale, bias=0.0, body=None):
        return conv(
            x,
            np.array([[[[scale]]]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
            body=body,
        )

    def va(x, length, scale, bias=0.0, body=None):
        return dense(
            x,
            np.eye(length, dtype=np.float32) * scale,
            np.full(length, bias, dtype=np.float32),
            body=body,
        )

    def vn(x, length, body=None):
        return va(x, length, -1.0, body=body)

    def vs(x, y, length, body=None):
        return b("vadd", x, vn(y, length, body), body)

    def sc(x, body=None):
        positive = u("relu", x, body)
        above_one = u("relu", sa(x, 1.0, -1.0, body), body)
        return b("sub", positive, above_one, body)

    def vc(x, length, body=None):
        positive = u("vrelu", x, body)
        above_one = u("vrelu", va(x, length, 1.0, -1.0, body), body)
        return vs(positive, above_one, length, body)

    def sslice(x, start, stop, body=None):
        return node("slice", [x], body=body, start=start, stop=stop)

    def vslice(x, start, stop, body=None):
        return node("vslice", [x], body=body, start=start, stop=stop)

    def channel_sum(x, n, body=None):
        return conv(
            x, np.ones((1, n, 1, 1), dtype=np.float32), body=body
        )

    def constant(values):
        values = np.asarray(values, dtype=np.float32)
        return dense(
            "avail",
            np.zeros((len(values), 8), dtype=np.float32),
            values,
        )

    def shift_parameter(dy, dx, size):
        w = np.zeros((1, 1, size, size), dtype=np.float32)
        center = size // 2
        w[0, 0, center - dy, center - dx] = 1.0
        return param(w)

    def loop(states, body, next_map, iterations):
        name = fresh("loop")
        graph.append({
            "id": name,
            "op": "loop",
            "T": int(iterations),
            "state": states,
            "body": body,
            "next": next_map,
        })
        return name

    def scalar_present(x):
        return vc(va(x, 1, 2.0, -1.0), 1)

    def choose_colour(scores):
        present_scores = b("mul", "x0", scores)
        score_map = channel_sum(present_scores, 16)
        best = u("gmax", score_map)
        delta = b("vadd", scores, vn(best, 1))
        delta = va(delta, 16, 1.0, 0.0002)
        return vc(va(delta, 16, 10000.0), 16)

    one = constant([1.0])
    zero = constant([0.0])
    rows = sslice("coords", 0, 1)
    cols = sslice("coords", 1, 2)
    selection = sslice("x0", 0, 1)
    selection_present = u("gmax", selection)

    allowed = np.ones(16, dtype=np.float32)
    allowed[[0, 1, 2, 4, 5, 15]] = 0.0
    allowed_v = constant(allowed)
    allowed_map = channel_sum(b("mul", "x0", allowed_v), 16)

    # Find target marker centres. A complete ring makes its colour reliable.
    four = sslice("x0", 4, 5)
    five = sslice("x0", 5, 6)
    ring_w = np.ones((1, 1, 3, 3), dtype=np.float32)
    ring_w[0, 0, 1, 1] = 0.0
    ring_count = conv(four, ring_w)
    marker_ring = sc(sa(ring_count, 2.0, -9.0))
    non_background = sc(sa(b("add", four, five), -1.0, 1.0))
    marker_map = b("mul", marker_ring, non_background)
    reliable = b(
        "mul", sc(sa(ring_count, 2.0, -15.0)), allowed_map
    )

    priority_w = np.zeros((1, 3, 1, 1), dtype=np.float32)
    priority_w[0, 0, 0, 0] = -1.0
    priority_w[0, 1, 0, 0] = -1.0 / 64.0
    priority = conv(
        "coords", priority_w, np.array([3.0], dtype=np.float32)
    )
    colour_map = conv(
        "x0", np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
    )

    # Extract up to 32 (row, column, colour) triples.
    body = []
    candidates = fresh("candidate")
    slots_state = fresh("slots")
    ranked = b("mul", candidates, priority, body)
    best = u("gmax", ranked, body)
    chosen = sc(sa(b("sub", ranked, best, body), 40000.0, 2.0, body), body)
    chosen = b("mul", chosen, candidates, body)

    cy = u("gmax", b("mul", chosen, rows, body), body)
    cx = u("gmax", b("mul", chosen, cols, body), body)
    reliable_colour = b("mul", colour_map, reliable, body)
    cc = u("gmax", b("mul", chosen, reliable_colour, body), body)
    triple = node("vconcat", [cy, cx, cc], body=body)

    shift_slots = np.zeros((96, 96), dtype=np.float32)
    shift_slots[3:, :-3] = np.eye(93, dtype=np.float32)
    insert_slots = np.zeros((96, 3), dtype=np.float32)
    insert_slots[:3] = np.eye(3, dtype=np.float32)
    shifted = dense(slots_state, shift_slots, body=body)
    inserted = dense(triple, insert_slots, body=body)
    slots_next = b("vadd", shifted, inserted, body)
    candidates_next = u("relu", b("sub", candidates, chosen, body), body)

    extraction = loop(
        [
            {"id": candidates, "init": marker_map},
            {"id": slots_state, "zeros": [96]},
        ],
        body,
        {candidates: candidates_next, slots_state: slots_next},
        32,
    )

    initialized = vslice("mem", 0, 1)
    uninitialized = vs(one, initialized, 1)
    previous_slots = vslice("mem", 1, 97)
    initial_slots = extraction + "." + slots_state
    slots = b(
        "vadd",
        b("vmul", previous_slots, initialized),
        b("vmul", initial_slots, uninitialized),
    )

    project_y = np.zeros((32, 96), dtype=np.float32)
    project_x = np.zeros((32, 96), dtype=np.float32)
    project_c = np.zeros((32, 96), dtype=np.float32)
    for i in range(32):
        project_y[i, 3 * i] = 1.0
        project_x[i, 3 * i + 1] = 1.0
        project_c[i, 3 * i + 2] = 1.0

    goal_y = dense(slots, project_y)
    goal_x = dense(slots, project_x)
    old_goal_c = dense(slots, project_c)

    diagonal32 = np.eye(32, dtype=np.float32).reshape(32, 32, 1, 1)
    row_delta = u("abs", b("sub", rows, goal_y))
    col_delta = u("abs", b("sub", cols, goal_x))
    row_match = u(
        "relu",
        conv(row_delta, -1000.0 * diagonal32, np.ones(32, dtype=np.float32)),
    )
    col_match = u(
        "relu",
        conv(col_delta, -1000.0 * diagonal32, np.ones(32, dtype=np.float32)),
    )
    positions = b("mul", row_match, col_match)

    # Update a stored target only when its complete ring is visible.
    visible = u("gmax", b("mul", positions, reliable))
    observed_c = u("gmax", b("mul", positions, colour_map))
    goal_c = b(
        "vadd",
        b("vmul", old_goal_c, vs(one, visible, 32)),
        b("vmul", observed_c, visible),
    )

    pack_w = np.zeros((96, 96), dtype=np.float32)
    for i in range(32):
        pack_w[3 * i, i] = 1.0
        pack_w[3 * i + 1, 32 + i] = 1.0
        pack_w[3 * i + 2, 64 + i] = 1.0
    packed = dense(node("vconcat", [goal_y, goal_x, goal_c]), pack_w)

    # Count remembered targets by colour with exact integer equality tests.
    identity = np.eye(32, dtype=np.float32)
    count_input_w = param(
        np.concatenate([identity, identity, identity], axis=0)
    )
    count_output_w = param(
        np.concatenate([
            np.ones(32, dtype=np.float32),
            -2.0 * np.ones(32, dtype=np.float32),
            np.ones(32, dtype=np.float32),
        ]).reshape(1, 96)
    )
    counts = []
    for colour in range(16):
        bias = np.concatenate([
            np.full(32, 1.0 - colour, dtype=np.float32),
            np.full(32, -float(colour), dtype=np.float32),
            np.full(32, -1.0 - colour, dtype=np.float32),
        ])
        z = shared_dense(goal_c, count_input_w, param(bias))
        counts.append(shared_dense(u("vrelu", z), count_output_w))
    target_counts = node("vconcat", counts)

    # Identify the selected colour using symmetric rays from the centre.
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (1, 1), (-1, 1), (1, -1),
    ]
    ray_weights = [shift_parameter(dy, dx, 3) for dy, dx in directions]

    body = []
    ray_states = [fresh("ray") for _ in directions]
    evidence_state = fresh("evidence")
    moved_rays = []
    hits = []
    for state_id, weights in zip(ray_states, ray_weights):
        moved = shared_conv(state_id, weights, body)
        moved_rays.append(moved)
        hits.append(u("gmax", b("mul", moved, "x0", body), body))

    pair0 = b("vmul", hits[0], hits[1], body)
    pair1 = b("vmul", hits[2], hits[3], body)
    pair2 = b("vmul", hits[4], hits[5], body)
    pair3 = b("vmul", hits[6], hits[7], body)
    fours = b(
        "vadd",
        b("vmul", pair0, pair1, body),
        b("vmul", pair2, pair3, body),
        body,
    )
    pairs = b(
        "vadd",
        b("vadd", pair0, pair1, body),
        b("vadd", pair2, pair3, body),
        body,
    )
    evidence_next = b(
        "vadd",
        evidence_state,
        b("vadd", va(fours, 16, 100.0, body=body), pairs, body),
        body,
    )
    ray_next_map = dict(zip(ray_states, moved_rays))
    ray_next_map[evidence_state] = evidence_next
    ray_loop = loop(
        [{"id": s, "init": selection} for s in ray_states]
        + [{"id": evidence_state, "zeros": [16]}],
        body,
        ray_next_map,
        30,
    )

    colour_bias = np.arange(16, dtype=np.float32) * 0.001
    colour_bias[[0, 1, 2, 4, 5, 15]] = -10000.0
    evidence = dense(
        ray_loop + "." + evidence_state,
        np.eye(16, dtype=np.float32),
        colour_bias,
    )
    active_colour = choose_colour(evidence)
    active_pixels = sc(channel_sum(b("mul", "x0", active_colour), 16))

    sy = u("gmax", b("mul", selection, rows))
    sx = u("gmax", b("mul", selection, cols))
    dy_map = b("sub", rows, sy)
    dx_map = b("sub", cols, sx)
    abs_dy = u("abs", dy_map)
    abs_dx = u("abs", dx_map)
    distance = b("add", abs_dy, abs_dx)

    # Classify cross, X, line, and diamond geometries.
    immediate = []
    for weights in ray_weights:
        shifted = shared_conv(selection, weights)
        immediate.append(u("gmax", b("mul", shifted, active_pixels)))

    vertical_pair = b("vmul", immediate[0], immediate[1])
    horizontal_pair = b("vmul", immediate[2], immediate[3])
    diag_pair1 = b("vmul", immediate[4], immediate[5])
    diag_pair2 = b("vmul", immediate[6], immediate[7])
    cross_flag = b("vmul", vertical_pair, horizontal_pair)
    x_flag = b("vmul", diag_pair1, diag_pair2)
    not_cross = vs(one, cross_flag, 1)
    horizontal_flag = b("vmul", horizontal_pair, not_cross)
    vertical_flag = b("vmul", vertical_pair, not_cross)
    known = b(
        "vadd",
        b("vadd", cross_flag, x_flag),
        b("vadd", horizontal_flag, vertical_flag),
    )
    diamond_flag = vs(one, vc(known, 1), 1)

    # Memory for plateau recovery and detecting a blocked move.
    previous_cycles = vslice("mem", 97, 98)
    previous_mode = vslice("mem", 98, 99)
    previous_y = vslice("mem", 99, 100)
    previous_x = vslice("mem", 100, 101)
    previous_position_valid = vslice("mem", 101, 102)
    previous_cycle_action = vslice("aprev", 5, 6)

    # Count consecutive ACTION5s, saturating at four.
    cycles_plus_one = va(previous_cycles, 1, 1.0, 1.0)
    cycles_over_four = u("vrelu", va(cycles_plus_one, 1, 1.0, -4.0))
    capped_cycles = vs(cycles_plus_one, cycles_over_four, 1)
    next_cycles = b("vmul", capped_cycles, previous_cycle_action)
    plateau = vc(va(next_cycles, 1, 2.0, -5.0), 1)

    # Once cycling stalls, try assigning an X to different targets.
    # Keep this mode while that X is moving; clear it after the next cycle.
    keep_mode = b("vmul", previous_mode, vs(one, previous_cycle_action, 1))
    start_mode = b(
        "vmul",
        b("vmul", plateau, x_flag),
        vs(one, previous_mode, 1),
    )
    next_mode = vc(b("vadd", keep_mode, start_mode), 1)
    aggressive = b("vmul", next_mode, x_flag)

    node(
        "vconcat",
        [one, packed, next_cycles, next_mode, sy, sx, selection_present],
        dest="mem_next",
    )

    # Find the radius of a diamond from its cardinal boundary.
    body = []
    diamond_rays = [fresh("dray") for _ in range(4)]
    radius_state = fresh("radius")
    seen_state = fresh("seen")
    moved = []
    diamond_hits = []
    for state_id, weights in zip(diamond_rays, ray_weights[:4]):
        ray = shared_conv(state_id, weights, body)
        moved.append(ray)
        diamond_hits.append(
            u("gmax", b("mul", ray, active_pixels, body), body)
        )

    four_hit = b(
        "vmul",
        b("vmul", diamond_hits[0], diamond_hits[1], body),
        b("vmul", diamond_hits[2], diamond_hits[3], body),
        body,
    )
    take = b("vmul", four_hit, vs(one, seen_state, 1, body), body)
    south_y = u("gmax", b("mul", moved[1], rows, body), body)
    candidate_radius = vs(south_y, sy, 1, body)
    radius_next = b(
        "vadd",
        radius_state,
        b("vmul", take, candidate_radius, body),
        body,
    )
    seen_next = vc(b("vadd", seen_state, take, body), 1, body)
    diamond_next = dict(zip(diamond_rays, moved))
    diamond_next[radius_state] = radius_next
    diamond_next[seen_state] = seen_next
    diamond_loop = loop(
        [{"id": s, "init": selection} for s in diamond_rays]
        + [
            {"id": radius_state, "zeros": [1]},
            {"id": seen_state, "zeros": [1]},
        ],
        body,
        diamond_next,
        30,
    )
    radius = diamond_loop + "." + radius_state
    radius_seen = diamond_loop + "." + seen_state

    same_row = u("relu", sa(abs_dy, -500.0, 1.0))
    same_col = u("relu", sa(abs_dx, -500.0, 1.0))
    diagonal_delta = u("abs", b("sub", abs_dy, abs_dx))
    diagonal_mask = u("relu", sa(diagonal_delta, -500.0, 1.0))
    radius_delta = u("abs", b("sub", distance, radius))
    diamond_mask = u("relu", sa(radius_delta, -500.0, 1.0))
    diamond_mask = b("mul", diamond_mask, radius_seen)
    cross_mask = sc(b("add", same_row, same_col))
    geometry = b(
        "add",
        b(
            "add",
            b("mul", cross_mask, cross_flag),
            b("mul", diagonal_mask, x_flag),
        ),
        b(
            "add",
            b("mul", same_row, horizontal_flag),
            b("mul", same_col, vertical_flag),
        ),
    )
    geometry = sc(
        b("add", geometry, b("mul", diamond_mask, diamond_flag))
    )

    plausible = b("mul", active_pixels, geometry)
    local_count = conv(
        plausible, np.ones((1, 1, 3, 3), dtype=np.float32)
    )
    eligible = b("mul", plausible, sc(sa(local_count, 2.0, -3.0)))
    near_score = b("mul", sa(distance, -63.0, 128.0), eligible)
    near_best = u("gmax", near_score)
    seed = sc(sa(b("sub", near_score, near_best), 8.0, 2.0))
    seed = b("mul", seed, eligible)

    traversable = sc(b("add", plausible, selection))
    flood_weight = param(np.ones((1, 1, 3, 3), dtype=np.float32))
    body = []
    flood_state = fresh("flood")
    expanded = sc(shared_conv(flood_state, flood_weight, body), body)
    flood_next = b("mul", expanded, traversable, body)
    flood_loop = loop(
        [{"id": flood_state, "init": seed}],
        body,
        {flood_state: flood_next},
        64,
    )
    component = flood_loop + "." + flood_state
    component = u(
        "relu",
        b("sub", component, b("mul", selection, diamond_flag)),
    )

    # Form the active-colour target map from remembered positions.
    active_number = dense(
        active_colour,
        np.arange(16, dtype=np.float32).reshape(1, 16),
    )
    colour_delta = b("vadd", goal_c, vn(active_number, 1))
    colour_abs = b(
        "vadd",
        u("vrelu", colour_delta),
        u("vrelu", vn(colour_delta, 32)),
    )
    colour_match = u("vrelu", va(colour_abs, 32, -1.0, 1.0))
    all_goals = channel_sum(b("mul", positions, colour_match), 32)

    # Normally preserve a target covered by another same-colour piece.
    neighbourhood = conv(
        active_pixels, np.ones((1, 1, 3, 3), dtype=np.float32)
    )
    has_neighbour = sc(sa(neighbourhood, 2.0, -3.0))
    covered = b("mul", active_pixels, has_neighbour)
    available_goal = sc(b("add", sa(covered, -1.0, 1.0), component))
    normal_goals = b("mul", all_goals, available_goal)

    # In reassignment mode, seek targets not already on this component,
    # including targets currently covered by another piece.
    alternative_goals = b("mul", all_goals, sa(component, -1.0, 1.0))
    goal_map = b(
        "add",
        b("mul", normal_goals, vs(one, aggressive, 1)),
        b("mul", alternative_goals, aggressive),
    )

    active_count = dense(
        b("vmul", target_counts, active_colour),
        np.ones((1, 16), dtype=np.float32),
    )
    has_active_goals = scalar_present(active_count)
    no_active_goals = vs(one, has_active_goals, 1)

    # Correlate piece pixels with targets at translations by multiples
    # of three. Nine channels retain all residues within each 3x3 cell.
    offsets = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)]
    neighbourhood_w = np.zeros((9, 1, 3, 3), dtype=np.float32)
    finish_w = np.zeros((1, 9, 3, 3), dtype=np.float32)
    for i, (dy, dx) in enumerate(offsets):
        neighbourhood_w[i, 0, 1 + dy, 1 + dx] = 1.0
        finish_w[0, i, 1 + dy, 1 + dx] = 1.0
    component_neighbours = conv(component, neighbourhood_w)

    shifts = {}
    for dy, dx in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
        shifts[(dy, dx)] = shift_parameter(dy, dx, 7)

    accumulator = b("mul", component_neighbours, zero)

    for sign_y in (1, -1):
        for sign_x in (1, -1):
            row_seed = selection
            row_goal = goal_map
            if sign_y < 0:
                row_seed = shared_conv(row_seed, shifts[(-3, 0)])
                row_goal = shared_conv(row_goal, shifts[(3, 0)])

            nrows = 11 if sign_y > 0 else 10
            ncols = 11 if sign_x > 0 else 10

            for row_index in range(nrows):
                col_seed = row_seed
                col_goal = row_goal
                if sign_x < 0:
                    col_seed = shared_conv(col_seed, shifts[(0, -3)])
                    col_goal = shared_conv(col_goal, shifts[(0, 3)])

                body = []
                scan_seed = fresh("scan")
                scan_goal = fresh("scan")
                scan_acc = fresh("scan")
                present = u(
                    "gmax",
                    b("mul", component_neighbours, scan_seed, body),
                    body,
                )
                contribution = b("mul", scan_goal, present, body)
                acc_next = b("add", scan_acc, contribution, body)
                seed_next = shared_conv(
                    scan_seed, shifts[(0, 3 * sign_x)], body
                )
                goal_next = shared_conv(
                    scan_goal, shifts[(0, -3 * sign_x)], body
                )
                scan_loop = loop(
                    [
                        {"id": scan_seed, "init": col_seed},
                        {"id": scan_goal, "init": col_goal},
                        {"id": scan_acc, "init": accumulator},
                    ],
                    body,
                    {
                        scan_seed: seed_next,
                        scan_goal: goal_next,
                        scan_acc: acc_next,
                    },
                    ncols,
                )
                accumulator = scan_loop + "." + scan_acc

                if row_index + 1 < nrows:
                    row_seed = shared_conv(
                        row_seed, shifts[(3 * sign_y, 0)]
                    )
                    row_goal = shared_conv(
                        row_goal, shifts[(-3 * sign_y, 0)]
                    )

    correlation = conv(accumulator, finish_w)

    lattice_w = np.zeros((1, 1, 7, 7), dtype=np.float32)
    for iy, ix in [(3, 3), (0, 3), (6, 3), (3, 0), (3, 6)]:
        lattice_w[0, 0, iy, ix] = 1.0
    lattice_weight = param(lattice_w)
    body = []
    lattice_state = fresh("lattice")
    lattice_next = sc(shared_conv(lattice_state, lattice_weight, body), body)
    lattice_loop = loop(
        [{"id": lattice_state, "init": selection}],
        body,
        {lattice_state: lattice_next},
        42,
    )
    lattice = lattice_loop + "." + lattice_state

    goal_score = sa(correlation, 100.0, 100.0)
    goal_score = b("sub", goal_score, sa(distance, 10.0))
    goal_score = b("mul", goal_score, lattice)

    # If the selected colour has no targets, head toward a needed dye.
    solid_w = np.zeros((16, 16, 3, 3), dtype=np.float32)
    for colour in range(16):
        solid_w[colour, colour, :, :] = 1.0
    solid_count = conv("x0", solid_w)
    solid_diag = np.eye(16, dtype=np.float32).reshape(16, 16, 1, 1)
    dye_planes = u(
        "relu",
        conv(
            solid_count,
            2.0 * solid_diag,
            np.full(16, -17.0, dtype=np.float32),
        ),
    )
    dye_planes = b("mul", dye_planes, allowed_v)
    dye_present = u("gmax", dye_planes)
    dye_scores = b("vmul", target_counts, dye_present)
    dye_eligible = vc(va(dye_scores, 16, 2.0), 16)
    dye_tie = constant(np.arange(16, dtype=np.float32) * 0.001)
    dye_scores = b(
        "vadd", dye_scores, b("vmul", dye_eligible, dye_tie)
    )
    dye_colour = b("vmul", choose_colour(dye_scores), dye_eligible)
    dye_map = channel_sum(b("mul", dye_planes, dye_colour), 16)
    has_dye = scalar_present(u("gmax", dye_map))
    dye_score = b("mul", sa(distance, -63.0, 200.0), dye_map)

    combined = b(
        "add",
        b("mul", goal_score, has_active_goals),
        b("mul", dye_score, no_active_goals),
    )
    masks = [
        sc(sa(dy_map, -63.0)),
        sc(sa(dy_map, 63.0)),
        sc(sa(dx_map, -63.0)),
        sc(sa(dx_map, 63.0)),
    ]
    direction_scores = [
        u("gmax", b("mul", combined, mask)) for mask in masks
    ]
    stay = va(u("gmax", b("mul", combined, selection)), 1, 1.0, 0.01)
    missing_selection = vs(one, selection_present, 1)
    missing_dye = vs(one, has_dye, 1)
    cannot_paint = b("vmul", no_active_goals, missing_dye)
    force_cycle = b("vadd", missing_selection, cannot_paint)
    stay = b("vadd", stay, va(force_cycle, 1, 1000.0))

    # Penalize the immediately preceding direction if it did not move
    # the selected centre. This allows a perpendicular detour.
    delta_y = vs(sy, previous_y, 1)
    delta_x = vs(sx, previous_x, 1)
    abs_delta_y = b("vadd", u("vrelu", delta_y), u("vrelu", vn(delta_y, 1)))
    abs_delta_x = b("vadd", u("vrelu", delta_x), u("vrelu", vn(delta_x, 1)))
    moved_amount = b("vadd", abs_delta_y, abs_delta_x)
    moved_flag = vc(va(moved_amount, 1, 1000.0), 1)
    blocked = b(
        "vmul",
        b("vmul", previous_position_valid, selection_present),
        vs(one, moved_flag, 1),
    )
    previous_directions = vslice("aprev", 1, 5)
    blocked_directions = b("vmul", previous_directions, blocked)
    direction_vector = node("vconcat", direction_scores)
    direction_vector = b(
        "vadd", direction_vector, va(blocked_directions, 4, -1000.0)
    )

    choices = node("vconcat", [direction_vector, stay])
    output_w = np.zeros((8, 5), dtype=np.float32)
    for i in range(5):
        output_w[i + 1, i] = 1.0
    output_b = np.zeros(8, dtype=np.float32)
    output_b[[0, 6, 7]] = -10000.0
    dense(choices, output_w, output_b, dest="logits")

    conv(
        selection,
        np.zeros((1, 1, 1, 1), dtype=np.float32),
        dest="click",
    )

    return {"graph": graph, "memory_size": 102, "params": params}
