import numpy as np


def build():
    G, P, cache, shapes = [], {}, {}, {}
    serial = 0
    N = 16
    A = 5 * N
    M = A + 25
    shapes.update({
        "x0": ("s", 16), "coords": ("s", 3),
        "mem": ("v", M), "aprev": ("v", 8), "avail": ("v", 8),
    })

    def fresh():
        nonlocal serial
        serial += 1
        return "n%d" % serial

    def parameter(value):
        value = np.asarray(value, np.float32)
        key = (value.shape, value.tobytes())
        if key not in cache:
            name = "p%d" % len(P)
            cache[key] = name
            P[name] = value
        return cache[key]

    def node(op, ins, **fields):
        name = fresh()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        G.append(nd)
        if op in ("conv", "dense"):
            shapes[name] = (
                "s" if op == "conv" else "v",
                P[fields["w"]].shape[0],
            )
        elif op in ("slice", "vslice"):
            shapes[name] = (
                shapes[ins[0]][0], fields["stop"] - fields["start"]
            )
        elif op in ("concat", "vconcat"):
            shapes[name] = (
                shapes[ins[0]][0], sum(shapes[x][1] for x in ins)
            )
        elif op in ("gmax", "gmean"):
            shapes[name] = ("v", shapes[ins[0]][1])
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shapes[name] = ("s", max(shapes[x][1] for x in ins))
        elif op in ("vadd", "vmul"):
            shapes[name] = ("v", max(shapes[x][1] for x in ins))
        else:
            shapes[name] = shapes[ins[0]]
        return name

    def C(x, w, b=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def V(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def sl(x, start, stop):
        return node(
            "slice" if shapes[x][0] == "s" else "vslice",
            [x], start=start, stop=stop,
        )

    def cat(*xs):
        return node(
            "concat" if shapes[xs[0]][0] == "s" else "vconcat", xs
        )

    def add(x, y):
        spatial = shapes[x][0] == "s" or shapes[y][0] == "s"
        return node("add" if spatial else "vadd", [x, y])

    def mul(x, y):
        spatial = shapes[x][0] == "s" or shapes[y][0] == "s"
        return node("mul" if spatial else "vmul", [x, y])

    def affine(x, scale=1.0, bias=0.0):
        n = shapes[x][1]
        w = np.eye(n, dtype=np.float32) * scale
        b = np.full(n, bias, np.float32)
        return C(x, w, b) if shapes[x][0] == "s" else V(x, w, b)

    def sub(x, y):
        if shapes[x][0] == "s" or shapes[y][0] == "s":
            return node("sub", [x, y])
        return add(x, affine(y, -1))

    def relu(x):
        return node("relu" if shapes[x][0] == "s" else "vrelu", [x])

    def absolute(x):
        if shapes[x][0] == "s":
            return node("abs", [x])
        return add(relu(x), relu(affine(x, -1)))

    one = V("aprev", np.zeros((1, 8)), [1])

    def clip(x):
        if shapes[x][0] == "s":
            return node("min2", [relu(x), one])
        return sub(relu(x), relu(affine(x, 1, -1)))

    def eq(x, sharpness=8.0):
        z = relu(affine(absolute(x), -sharpness, 2))
        if shapes[x][0] == "s":
            return node("min2", [z, one])
        return sub(z, relu(affine(z, 1, -1)))

    def gm(x):
        return node("gmax", [x])

    def channel_sum(x):
        return C(x, np.ones((1, shapes[x][1])))

    def channel_max(x):
        out = sl(x, 0, 1)
        for j in range(1, shapes[x][1]):
            out = node("max2", [out, sl(x, j, j + 1)])
        return out

    def state(init):
        name = fresh()
        shapes[name] = shapes[init]
        return {"id": name, "init": init}

    def zeros(kind, channels):
        name = fresh()
        shapes[name] = (kind, channels)
        return {
            "id": name,
            "zeros": (
                [channels, 64, 64] if kind == "s" else [channels]
            ),
        }

    def finish_loop(T, states, body, updates):
        name = fresh()
        G.append({
            "id": name, "op": "loop", "T": T,
            "state": states, "body": body, "next": updates,
        })
        outputs = []
        for s in states:
            out = name + "." + s["id"]
            shapes[out] = shapes[s["id"]]
            outputs.append(out)
        return outputs

    def shift(channels, dy, dx):
        k = 7 if max(abs(dy), abs(dx)) > 1 else 3
        w = np.zeros((channels, channels, k, k), np.float32)
        for c in range(channels):
            w[c, c, k // 2 + dy, k // 2 + dx] = 1
        return w

    # Selected centre and integer-coordinate fields.
    row = C(sl("coords", 0, 1), [[63]])
    col = C(sl("coords", 1, 2), [[63]])
    zero = sl("x0", 0, 1)
    colours = sl("x0", 6, 15)
    ids = np.arange(6, 15, dtype=np.float32)
    cr = gm(mul(row, zero))
    cc = gm(mul(col, zero))
    dy = sub(row, cr)
    dx = sub(col, cc)
    ady, adx = absolute(dy), absolute(dx)
    distance = add(ady, adx)

    # A marker centre has four axial or four diagonal colour-4 neighbours.
    marker_w = np.zeros((2, 16, 3, 3), np.float32)
    for r, c in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        marker_w[0, 4, r, c] = 1
    for r, c in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        marker_w[1, 4, r, c] = 1
    marker_parts = C("x0", marker_w)
    corner_marker = clip(affine(sl(marker_parts, 0, 1), 1, -3))
    axial_marker = clip(affine(sl(marker_parts, 1, 2), 1, -3))
    marker = node("max2", [corner_marker, axial_marker])
    marker = mul(marker, affine(sl("x0", 4, 5), -1, 1))
    ring_count = channel_sum(marker_parts)
    intact = clip(affine(ring_count, 1, -7))
    known = mul(mul(colours, marker), intact)

    # Estimate the colour of an initially covered target from colour centroids.
    counts = affine(node("gmean", [known]), 4096)
    rw = np.zeros((153, 9), np.float32)
    rb = np.zeros(153, np.float32)
    for c in range(9):
        for k in range(17):
            rw[17 * c + k, c] = 1
            rb[17 * c + k] = -k
    hinges = relu(V(counts, rw, rb))
    values = [0.0] + [1.0 / k for k in range(1, 17)]
    slopes = [
        values[k + 1] - values[k] for k in range(16)
    ] + [0.0]
    inverse_w = np.zeros((9, 153), np.float32)
    for c in range(9):
        inverse_w[c, 17 * c] = slopes[0]
        for k in range(1, 17):
            inverse_w[c, 17 * c + k] = slopes[k] - slopes[k - 1]
    inverse = V(hinges, inverse_w)
    mean_rows = mul(
        affine(node("gmean", [mul(known, row)]), 4096), inverse
    )
    mean_cols = mul(
        affine(node("gmean", [mul(known, col)]), 4096), inverse
    )
    absent = clip(affine(counts, -1, 1))
    hue_tie = V(
        "aprev", np.zeros((9, 8)),
        np.arange(9, dtype=np.float32) * 0.01,
    )
    hue_distance = add(
        absolute(sub(C(row, np.ones((9, 1))), mean_rows)),
        absolute(sub(C(col, np.ones((9, 1))), mean_cols)),
    )
    hue_distance = add(
        add(hue_distance, affine(absent, 2048)), hue_tie
    )
    negative_distance = affine(hue_distance, -1)
    nearest_hue = eq(
        sub(negative_distance, channel_max(negative_distance)), 10000
    )
    repaired_targets = add(
        known, mul(mul(nearest_hue, marker), affine(intact, -1, 1))
    )

    # Remove target centres and wells from long-range colour measurements.
    well_border = sl("x0", 2, 3)
    well_area = clip(C(well_border, np.ones((1, 1, 7, 7))))
    clean_colours = mul(
        colours, mul(affine(marker, -1, 1), affine(well_area, -1, 1))
    )

    vertical = eq(dx)
    horizontal = eq(dy)
    diagonal_a = eq(sub(dx, dy))
    diagonal_b = eq(add(dx, dy))
    pos_y, neg_y = clip(dy), clip(affine(dy, -1))
    pos_x, neg_x = clip(dx), clip(affine(dx, -1))
    proximity = relu(affine(distance, -1, 32))
    rays = [
        mul(vertical, pos_y), mul(vertical, neg_y),
        mul(horizontal, pos_x), mul(horizontal, neg_x),
        mul(diagonal_a, pos_y), mul(diagonal_a, neg_y),
        mul(diagonal_b, pos_y), mul(diagonal_b, neg_y),
    ]
    ray_colours = [
        gm(mul(clean_colours, mul(mask, proximity))) for mask in rays
    ]
    hue_score = mul(ray_colours[0], ray_colours[1])
    for j in range(1, 4):
        hue_score = add(
            hue_score, mul(ray_colours[2 * j], ray_colours[2 * j + 1])
        )

    # Immediate coloured neighbours take precedence over distant intersections.
    local_hue_w = np.zeros((9, 16, 3, 3), np.float32)
    for c in range(9):
        local_hue_w[c, c + 6] = 1
        local_hue_w[c, c + 6, 1, 1] = 0
    local_hue = gm(mul(C("x0", local_hue_w), zero))
    hue_score = add(
        add(hue_score, affine(local_hue, 100000)), hue_tie
    )

    compare_w = np.zeros((81, 9), np.float32)
    for i in range(9):
        for j in range(9):
            compare_w[9 * i + j, j] += 1
            compare_w[9 * i + j, i] -= 1
    differences = relu(V(hue_score, compare_w))
    winner_w = np.zeros((9, 81), np.float32)
    for i in range(9):
        winner_w[i, 9 * i:9 * i + 9] = -10000
    hue = relu(V(differences, winner_w, np.ones(9)))
    hue_id = V(hue, ids[None])
    occupancy = channel_sum(mul(colours, hue))
    clean_occupancy = channel_sum(mul(clean_colours, hue))

    # Identify solid strokes at the selected centre.
    pair_w = np.zeros((4, 1, 3, 3), np.float32)
    pairs = [
        [(0, 1), (2, 1)], [(1, 0), (1, 2)],
        [(0, 0), (2, 2)], [(0, 2), (2, 0)],
    ]
    for q, points in enumerate(pairs):
        for r, c in points:
            pair_w[q, 0, r, c] = 1
    pair_counts = gm(mul(C(occupancy, pair_w), zero))
    stroke_types = clip(affine(pair_counts, 2, -3))
    solid = clip(V(stroke_types, np.ones((1, 4))))
    hollow = affine(solid, -1, 1)
    axes = [vertical, horizontal, diagonal_a, diagonal_b]
    solid_geometry = mul(axes[0], sl(stroke_types, 0, 1))
    for j in range(1, 4):
        solid_geometry = add(
            solid_geometry, mul(axes[j], sl(stroke_types, j, j + 1))
        )
    solid_geometry = clip(solid_geometry)

    # Hollow rectangle and diamond geometry.
    edge_distances = []
    for j in range(4):
        near = gm(mul(clean_occupancy, mul(rays[j], proximity)))
        edge_distances.append(affine(near, -1, 32))
    bottom, top, right, left = edge_distances
    left_edge = eq(add(dx, left))
    right_edge = eq(sub(dx, right))
    top_edge = eq(add(dy, top))
    bottom_edge = eq(sub(dy, bottom))
    vertical_edges = clip(add(left_edge, right_edge))
    horizontal_edges = clip(add(top_edge, bottom_edge))
    within_x = mul(
        clip(affine(add(dx, left), 1, 1)),
        clip(affine(sub(right, dx), 1, 1)),
    )
    within_y = mul(
        clip(affine(add(dy, top), 1, 1)),
        clip(affine(sub(bottom, dy), 1, 1)),
    )
    rectangle = clip(add(
        mul(vertical_edges, within_y),
        mul(horizontal_edges, within_x),
    ))
    corners = mul(vertical_edges, horizontal_edges)
    rectangle_type = clip(gm(mul(occupancy, corners)))
    diamond = eq(sub(distance, left))
    hollow_geometry = add(
        mul(rectangle, rectangle_type),
        mul(diamond, affine(rectangle_type, -1, 1)),
    )
    geometry = clip(add(solid_geometry, mul(hollow_geometry, hollow)))
    footprint = add(mul(occupancy, geometry), mul(zero, solid))

    # Archive sixteen target records: exists, row, column, colour, unclaimed.
    priority = C(cat(row, col), [[-64, -1]], [4096])
    remaining_state = state(marker)
    archive_state = zeros("v", A)
    rotate = np.zeros((A, A), np.float32)
    rotate[:A - 5, 5:] = np.eye(A - 5)
    insert = np.zeros((A, 5), np.float32)
    insert[-5:] = np.eye(5)

    outer = G
    G = []
    rank = mul(priority, remaining_state["id"])
    pick = mul(
        eq(sub(rank, gm(rank))), remaining_state["id"]
    )
    exists_pick = gm(pick)
    row_pick = gm(mul(row, pick))
    col_pick = gm(mul(col, pick))
    id_pick = V(gm(mul(repaired_targets, pick)), ids[None])
    record = cat(exists_pick, row_pick, col_pick, id_pick, exists_pick)
    next_archive = add(
        V(archive_state["id"], rotate), V(record, insert)
    )
    next_remaining = sub(remaining_state["id"], pick)
    body = G
    G = outer
    _, initial_archive = finish_loop(
        N, [remaining_state, archive_state], body,
        {
            remaining_state["id"]: next_remaining,
            archive_state["id"]: next_archive,
        },
    )
    initialized = sl("mem", A, A + 1)
    archive = add(
        mul(sl("mem", 0, A), initialized),
        mul(initial_archive, affine(initialized, -1, 1)),
    )
    locks = sl("mem", A + 1, M)

    fields = []
    for f in range(5):
        w = np.zeros((N, A), np.float32)
        for j in range(N):
            w[j, 5 * j + f] = 1
        fields.append(V(archive, w))
    exists, target_rows, target_cols, target_ids, unclaimed = fields
    slot_rows = eq(sub(C(row, np.ones((N, 1))), target_rows))
    slot_cols = eq(sub(C(col, np.ones((N, 1))), target_cols))
    slots = mul(mul(slot_rows, slot_cols), exists)

    pending_map = channel_sum(mul(slots, unclaimed))
    count_map = channel_sum(slots)
    colour_map = channel_sum(mul(slots, target_ids))
    square_map = channel_sum(mul(slots, mul(target_ids, target_ids)))
    statistics = cat(pending_map, count_map, colour_map, square_map)

    # Correlate every integer-offset footprint pixel with target statistics.
    offsets = np.arange(-21, 22, dtype=np.float32)
    column_bands = eq(C(dx, np.ones((43, 1)), -offsets))
    shifted = statistics
    for delta in [(-3, 0), (0, -3)]:
        s = state(shifted)
        outer = G
        G = []
        nxt = C(s["id"], shift(4, *delta))
        body = G
        G = outer
        shifted, = finish_loop(7, [s], body, {s["id"]: nxt})

    map_state = state(shifted)
    score_state = zeros("s", 4)
    offset_state = state(V("aprev", np.zeros((1, 8)), [-21]))
    outer = G
    G = []
    row_band = eq(sub(dy, offset_state["id"]))
    samples = gm(mul(column_bands, mul(footprint, row_band)))
    score = score_state["id"]
    moving = map_state["id"]
    for j in range(43):
        present = sl(samples, j, j + 1)
        score = add(score, mul(moving, present))
        if j != 42:
            moving = C(moving, shift(4, 0, 1))
    next_row = C(map_state["id"], shift(4, 1, 0))
    next_offset = affine(offset_state["id"], 1, 1)
    body = G
    G = outer
    _, correlations, _ = finish_loop(
        43, [map_state, score_state, offset_state], body,
        {
            map_state["id"]: next_row,
            score_state["id"]: score,
            offset_state["id"]: next_offset,
        },
    )

    pending_count = sl(correlations, 0, 1)
    total_count = sl(correlations, 1, 2)
    summed_ids = sl(correlations, 2, 3)
    summed_squares = sl(correlations, 3, 4)
    variance = relu(sub(
        mul(total_count, summed_squares),
        mul(summed_ids, summed_ids),
    ))
    current_match = eq(sub(summed_ids, mul(total_count, hue_id)))

    # Reachable centres share the selected centre's three-pixel lattice.
    lattice_offsets = np.arange(-63, 64, 3, dtype=np.float32)
    lattice_rows = channel_sum(
        eq(C(dy, np.ones((43, 1)), -lattice_offsets))
    )
    lattice_cols = channel_sum(
        eq(C(dx, np.ones((43, 1)), -lattice_offsets))
    )
    lattice = mul(lattice_rows, lattice_cols)

    placement = sub(affine(pending_count, 100), affine(variance, 20))
    placement = add(
        placement, affine(mul(current_match, pending_count), 2)
    )
    placement = sub(placement, affine(distance, 0.01))
    placement = add(placement, affine(lattice, 10000, -10000))
    best_score = gm(placement)
    winners = eq(sub(placement, best_score), 100000)
    goal_row = gm(mul(row, winners))
    winning_row = mul(winners, eq(sub(row, goal_row)))
    goal_col = gm(mul(col, winning_row))
    goal_cell = mul(winning_row, eq(sub(col, goal_col)))

    goal_count = gm(mul(total_count, goal_cell))
    goal_colour_sum = gm(mul(summed_ids, goal_cell))
    desired_hue = eq(sub(
        V(goal_colour_sum, np.ones((9, 1))),
        V(goal_count, ids[:, None]),
    ))
    same_hue = clip(V(mul(hue, desired_hue), np.ones((1, 9))))
    has_target = clip(affine(best_score, 1, -0.5))

    # Find a well of the desired colour.
    adjacent_w = np.ones((1, 1, 3, 3), np.float32)
    adjacent_w[0, 0, 1, 1] = 0
    adjacent = clip(C(well_border, adjacent_w))
    desired_well = mul(
        adjacent, channel_sum(mul(colours, desired_hue))
    )
    well_score = add(
        affine(distance, -0.01), affine(desired_well, 10000, -10000)
    )
    well_winners = eq(sub(well_score, gm(well_score)), 100000)
    well_row = gm(mul(row, well_winners))
    well_row_cells = mul(well_winners, eq(sub(row, well_row)))
    well_col = gm(mul(col, well_row_cells))
    needs_colour = affine(same_hue, -1, 1)
    waypoint_row = add(
        mul(goal_row, same_hue), mul(well_row, needs_colour)
    )
    waypoint_col = add(
        mul(goal_col, same_hue), mul(well_col, needs_colour)
    )
    error_row = sub(waypoint_row, cr)
    error_col = sub(waypoint_col, cc)

    # Skip previously completed poses.
    lock_fields = []
    for f in range(3):
        w = np.zeros((8, 24), np.float32)
        for j in range(8):
            w[j, 3 * j + f] = 1
        lock_fields.append(V(locks, w))
    lock_error = add(
        add(absolute(sub(lock_fields[0], cr)),
            absolute(sub(lock_fields[1], cc))),
        absolute(sub(lock_fields[2], hue_id)),
    )
    locked = clip(V(eq(lock_error), np.ones((1, 8))))
    at_goal = mul(eq(sub(goal_row, cr)), eq(sub(goal_col, cc)))
    complete = mul(
        mul(mul(at_goal, same_hue), has_target),
        affine(locked, -1, 1),
    )
    skip = clip(add(
        add(locked, complete), affine(has_target, -1, 1)
    ))

    # A selected solid centre returns to its sprite colour on switching.
    observed_ids = C(colours, ids[None])
    observed_ids = add(
        observed_ids, mul(zero, mul(hue_id, solid))
    )
    observed_slots = gm(mul(slots, observed_ids))
    correct_slots = eq(sub(observed_slots, target_ids))
    new_claims = mul(
        unclaimed, affine(mul(correct_slots, complete), -1, 1)
    )
    keep = np.eye(A, dtype=np.float32)
    claim_w = np.zeros((A, N), np.float32)
    for j in range(N):
        keep[5 * j + 4] = 0
        claim_w[5 * j + 4, j] = 1
    updated_archive = add(V(archive, keep), V(new_claims, claim_w))

    lock_rotate = np.zeros((24, 24), np.float32)
    lock_rotate[3:, :21] = np.eye(21)
    lock_insert = np.zeros((24, 3), np.float32)
    lock_insert[:3] = np.eye(3)
    pose = cat(cr, cc, hue_id)
    proposed_locks = add(
        V(locks, lock_rotate), V(pose, lock_insert)
    )
    updated_locks = add(
        mul(locks, affine(complete, -1, 1)),
        mul(proposed_locks, complete),
    )
    mem_output = cat(updated_archive, one, updated_locks)
    G[-1]["id"] = "mem_next"
    shapes["mem_next"] = shapes.pop(mem_output)

    controls = cat(
        affine(error_row, -1), error_row,
        affine(error_col, -1), error_col, skip,
    )
    action_w = np.zeros((8, 5), np.float32)
    for j in range(4):
        action_w[j + 1, j] = 1
    action_w[5, 4] = 1000
    action_b = np.full(8, -10000, np.float32)
    action_b[1:5] = 0
    action_b[5] = 0.5
    output = V(controls, action_w, action_b)
    G[-1]["id"] = "logits"
    shapes["logits"] = shapes.pop(output)

    output = C(sl("coords", 2, 3), [[0]])
    G[-1]["id"] = "click"
    shapes["click"] = shapes.pop(output)

    return {"graph": G, "memory_size": M, "params": P}
