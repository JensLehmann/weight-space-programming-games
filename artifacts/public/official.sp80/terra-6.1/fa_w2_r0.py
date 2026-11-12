import numpy as np


def build():
    graph = []
    params = {}
    serial = 0
    shift_cache = {}

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return f"{prefix}_{serial}"

    def put(value, prefix="p"):
        name = fresh(prefix)
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, **fields):
        name = fresh()
        graph.append({"id": name, "op": op, "in": inputs, **fields})
        return name

    def unary(op, x):
        return node(op, [x])

    def binary(op, x, y):
        return node(op, [x, y])

    def cat(*xs):
        return node("concat", list(xs))

    def vcat(*xs):
        return node("vconcat", list(xs))

    def spatial_slice(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vector_slice(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def conv(x, weights, bias=None):
        fields = {"w": put(weights)}
        if bias is not None:
            fields["b"] = put(bias)
        return node("conv", [x], **fields)

    def dense(x, weights, bias=None):
        fields = {"w": put(weights)}
        if bias is not None:
            fields["b"] = put(bias)
        return node("dense", [x], **fields)

    def add_all(xs):
        out = xs[0]
        for x in xs[1:]:
            out = binary("add", out, x)
        return out

    def max_all(xs):
        out = xs[0]
        for x in xs[1:]:
            out = binary("max2", out, x)
        return out

    def vector_max(a, b):
        # max(a,b) = b + relu(a-b)
        neg_b = dense(b, [[-1.0]])
        difference = binary("vadd", a, neg_b)
        return binary("vadd", b, unary("vrelu", difference))

    def vector_clip01(x):
        positive = unary("vrelu", x)
        one_minus = dense(positive, [[-1.0]], [1.0])
        return dense(unary("vrelu", one_minus), [[-1.0]], [1.0])

    def shift(x, channels, dy, dx, distance):
        # Output at (r,c) reads input at (r+dy*d,c+dx*d).
        # Four pixels are shifted as two successive two-pixel shifts.
        step = 2 if distance == 4 else distance
        key = (channels, dy, dx, step)
        if key not in shift_cache:
            k = 2 * step + 1
            w = np.zeros((channels, channels, k, k), np.float32)
            for c in range(channels):
                w[c, c, step + dy * step, step + dx * step] = 1.0
            shift_cache[key] = put(w, "shift")
        wname = shift_cache[key]
        out = node("conv", [x], w=wname)
        if distance == 4:
            out = node("conv", [out], w=wname)
        return out

    def shift_cell(x, channels, dy, dx, scale3, scale4):
        a = shift(x, channels, dy, dx, 3)
        b = shift(x, channels, dy, dx, 4)
        return binary(
            "add",
            binary("mul", a, scale3),
            binary("mul", b, scale4),
        )

    def scale_spatial(x, factor, bias=0.0):
        return conv(x, [[[[factor]]]], [bias])

    # Input planes.
    background = spatial_slice("x0", 12, 13)
    source = spatial_slice("x0", 6, 7)
    source_head = spatial_slice("x0", 4, 5)
    goal = spatial_slice("x0", 11, 12)
    selected = spatial_slice("x0", 9, 10)
    blue = spatial_slice("x0", 8, 9)
    corner = spatial_slice("x0", 15, 16)
    wall = spatial_slice("x0", 1, 2)
    row = spatial_slice("coords", 0, 1)
    col = spatial_slice("coords", 1, 2)
    one = spatial_slice("coords", 2, 3)

    # Infer the display scale from the width of a source block.
    source_four = binary("min2", source, shift(source, 1, 0, 1, 3))
    scale4 = unary("gmax", source_four)
    scale3 = dense(scale4, [[-1.0]], [1.0])

    # Flow directions in display coordinates: down, up, left, right.
    directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    direction_flags = []
    for dy, dx in directions:
        head_upstream = shift(source_head, 1, -dy, -dx, 1)
        direction_flags.append(
            unary("gmax", binary("min2", source, head_upstream))
        )
    vertical = binary("vadd", direction_flags[0], direction_flags[1])
    horizontal = binary("vadd", direction_flags[2], direction_flags[3])

    # Detect target openings and short approach regions.
    mouths_by_scale = []
    near_by_scale = []
    for scale in (3, 4):
        mouths = []
        approaches = []
        for i, (dy, dx) in enumerate(directions):
            ty, tx = dx, dy
            a = shift(goal, 1, ty, tx, scale)
            b = shift(goal, 1, -ty, -tx, scale)
            back = shift(goal, 1, dy, dx, scale)
            m = binary("min2", background, a)
            m = binary("min2", m, b)
            m = binary("min2", m, back)
            m = binary("mul", m, direction_flags[i])
            mouths.append(m)

            approach = m
            q = m
            for _ in range(3):
                q = shift(q, 1, dy, dx, scale)
                approach = binary("max2", approach, q)
            approaches.append(approach)

        mouths_by_scale.append(max_all(mouths))
        near_by_scale.append(max_all(approaches))

    mouth = binary(
        "add",
        binary("mul", mouths_by_scale[0], scale3),
        binary("mul", mouths_by_scale[1], scale4),
    )
    near_goal = binary(
        "add",
        binary("mul", near_by_scale[0], scale3),
        binary("mul", near_by_scale[1], scale4),
    )

    # Four endpoint-potential channels. They provide a weak gradient even
    # when moving one cell does not yet change the simulated flow.
    vertical_mouth = binary("mul", mouth, vertical)
    horizontal_mouth = binary("mul", mouth, horizontal)
    vertical_projection = unary("colmax", vertical_mouth)
    horizontal_projection = unary("rowmax", horizontal_mouth)

    seeds = cat(
        shift_cell(vertical_projection, 1, 0, -1, scale3, scale4),
        shift_cell(vertical_projection, 1, 0, 1, scale3, scale4),
        shift_cell(horizontal_projection, 1, -1, 0, scale3, scale4),
        shift_cell(horizontal_projection, 1, 1, 0, scale3, scale4),
    )

    def potential_shift_weights(sign):
        w = np.zeros((4, 4, 3, 3), np.float32)
        for c in range(4):
            dy = sign if c >= 2 else 0
            dx = sign if c < 2 else 0
            w[c, c, 1 + dy, 1 + dx] = 1.0
        return put(w, "potential_shift")

    pot_neg_w = potential_shift_weights(-1)
    pot_pos_w = potential_shift_weights(1)
    pot_decay_w = put(np.eye(4, dtype=np.float32).reshape(4, 4, 1, 1))
    pot_decay_b = put([-0.0125] * 4)

    outer = graph
    graph = []
    neg = node("conv", ["potential_state"], w=pot_neg_w)
    pos = node("conv", ["potential_state"], w=pot_pos_w)
    propagated = binary("max2", neg, pos)
    propagated = node("conv", [propagated], w=pot_decay_w, b=pot_decay_b)
    propagated = unary("relu", propagated)
    pot_next = binary("max2", "potential_state", propagated)
    potential_body = graph
    graph = outer
    graph.append({
        "id": "potential_loop",
        "op": "loop",
        "T": 64,
        "state": [{"id": "potential_state", "init": seeds}],
        "body": potential_body,
        "next": {"potential_state": pot_next},
    })
    potential = "potential_loop.potential_state"

    ends = []
    for dy, dx in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        neighbor = shift_cell(selected, 1, dy, dx, scale3, scale4)
        ends.append(unary("relu", binary("sub", selected, neighbor)))

    # Candidate 0 is the current placement; 1-4 are screen directions.
    candidate_directions = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    candidate_selected = [selected]
    for dy, dx in candidate_directions[1:]:
        candidate_selected.append(
            shift_cell(selected, 1, -dy, -dx, scale3, scale4)
        )
    candidates = cat(*candidate_selected)

    alignment_values = []
    for dy, dx in candidate_directions:
        if dy == 0 and dx == 0:
            moved_ends = ends
        else:
            moved_ends = [
                shift_cell(e, 1, -dy, -dx, scale3, scale4)
                for e in ends
            ]
        endpoint_tensor = cat(*moved_ends)
        endpoint_scores = unary(
            "gmax", binary("mul", endpoint_tensor, potential)
        )
        alignment_values.append(dense(endpoint_scores, [[1, 1, 1, 1]]))
    alignment = vcat(*alignment_values)

    # Approximate the game's forbidden area around targets.
    goal_buffer_parts = [goal]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            q = goal
            if dy:
                q = shift_cell(q, 1, dy, 0, scale3, scale4)
            if dx:
                q = shift_cell(q, 1, 0, dx, scale3, scale4)
            goal_buffer_parts.append(q)
    goal_buffer = max_all(goal_buffer_parts)
    forbidden = max_all([goal_buffer, wall, source, source_head])
    invalid = unary("gmax", binary("min2", candidates, forbidden))

    # Static obstacles for each of the five candidate placements.
    other_objects = binary("max2", blue, corner)
    candidate_bars = binary("max2", candidates, other_objects)
    obstacles = max_all([candidate_bars, goal, wall, source_head])
    free = unary("relu", binary("sub", one, obstacles))

    # For flow splitting, only bars, corners, and non-catching goal pixels
    # act as branching surfaces.
    branching_surface = binary("max2", candidate_bars, goal)
    ahead_parts = []
    for i, (dy, dx) in enumerate(directions):
        q = shift(branching_surface, 5, dy, dx, 1)
        ahead_parts.append(binary("mul", q, direction_flags[i]))
    surface_ahead = max_all(ahead_parts)
    branch_here = unary("relu", binary("sub", surface_ahead, mouth))

    # Direction-aware wall contact indicators.
    wall_ahead_parts = []
    wall_side_parts = []
    for i, (dy, dx) in enumerate(directions):
        q = shift(wall, 1, dy, dx, 1)
        wall_ahead_parts.append(binary("mul", q, direction_flags[i]))
        ty, tx = dx, dy
        left_wall = shift(wall, 1, ty, tx, 1)
        right_wall = shift(wall, 1, -ty, -tx, 1)
        side = binary("max2", left_wall, right_wall)
        wall_side_parts.append(binary("mul", side, direction_flags[i]))
    wall_ahead = max_all(wall_ahead_parts)
    wall_side = max_all(wall_side_parts)

    # Flow approximation for all five placements in parallel.
    # Water advances one display pixel per iteration; on encountering a
    # branching surface it can spread transversely.
    initial_flow = cat(source, source, source, source, source)

    outer = graph
    graph = []
    forward_parts = []
    for i, (dy, dx) in enumerate(directions):
        q = shift("flow_state", 5, -dy, -dx, 1)
        forward_parts.append(binary("mul", q, direction_flags[i]))
    forward = max_all(forward_parts)

    blocked_flow = binary("min2", "flow_state", branch_here)
    side_up = shift(blocked_flow, 5, -1, 0, 1)
    side_down = shift(blocked_flow, 5, 1, 0, 1)
    side_left = shift(blocked_flow, 5, 0, -1, 1)
    side_right = shift(blocked_flow, 5, 0, 1, 1)
    side_vertical = binary(
        "mul", binary("max2", side_up, side_down), horizontal
    )
    side_horizontal = binary(
        "mul", binary("max2", side_left, side_right), vertical
    )
    sideways = binary("max2", side_vertical, side_horizontal)
    incoming = binary("max2", forward, sideways)
    incoming = binary("min2", incoming, free)
    flow_next = binary("max2", "flow_state", incoming)
    flow_body = graph
    graph = outer
    graph.append({
        "id": "flow_loop",
        "op": "loop",
        "T": 96,
        "state": [{"id": "flow_state", "init": initial_flow}],
        "body": flow_body,
        "next": {"flow_state": flow_next},
    })
    flow = "flow_loop.flow_state"

    goal_measure = unary("gmean", binary("mul", flow, mouth))
    approach_measure = unary("gmean", binary("mul", flow, near_goal))
    direct_wall = binary("mul", flow, wall_ahead)
    side_wall = binary(
        "mul", binary("min2", flow, branch_here), wall_side
    )
    leak_measure = unary(
        "gmean", binary("max2", direct_wall, side_wall)
    )

    # Scores are in approximately display-pixel units.
    score_inputs = vcat(
        goal_measure, approach_measure, leak_measure, alignment, invalid
    )
    score_w = np.zeros((5, 25), np.float32)
    for i in range(5):
        score_w[i, i] = 4.0 * 4096.0
        score_w[i, 5 + i] = 0.25 * 4096.0
        score_w[i, 10 + i] = -20.0 * 4096.0
        score_w[i, 15 + i] = 1.0
        if i:
            score_w[i, 20 + i] = -100000.0
    scores = dense(score_inputs, score_w)

    # Detect a movement that left the selected object's bounds unchanged.
    selected_row = unary("gmax", binary("mul", selected, row))
    selected_col = unary("gmax", binary("mul", selected, col))
    position = vcat(selected_row, selected_col)
    previous_position = vector_slice("mem", 1, 3)
    negative_previous = dense(
        previous_position, [[-1.0, 0.0], [0.0, -1.0]]
    )
    position_delta = binary("vadd", position, negative_previous)
    negative_delta = dense(
        position_delta, [[-1.0, 0.0], [0.0, -1.0]]
    )
    absolute_delta = binary(
        "vadd", unary("vrelu", position_delta), unary("vrelu", negative_delta)
    )
    changed_amount = dense(absolute_delta, [[100.0, 100.0]])
    changed = vector_clip01(changed_amount)
    unchanged = dense(changed, [[-1.0]], [1.0])
    previous_was_move = dense(
        "aprev", [[0, 1, 1, 1, 1, 0, 0, 0]]
    )
    stalled = binary("vmul", unchanged, previous_was_move)

    current_score = vector_slice(scores, 0, 1)
    move_scores = vector_slice(scores, 1, 5)
    stall_penalty = dense(stalled, [[-100000.0]])
    move_scores = binary("vadd", move_scores, stall_penalty)

    best_move = vector_slice(move_scores, 0, 1)
    for i in range(1, 4):
        best_move = vector_max(best_move, vector_slice(move_scores, i, i + 1))
    baseline = dense(current_score, [[1.0]], [0.01])
    negative_baseline = dense(baseline, [[-1.0]])
    advantage = binary("vadd", best_move, negative_baseline)
    positive_advantage = unary("vrelu", advantage)
    move_gate = vector_clip01(dense(positive_advantage, [[100.0]]))
    no_move = dense(move_gate, [[-1.0]], [1.0])

    # Cursor-based selection of the remaining visible movable objects.
    # A failed spill starts another selection pass.
    cursor = vector_slice("mem", 0, 1)
    previous_spill = vector_slice("aprev", 5, 6)
    keep_cursor = dense(previous_spill, [[-1.0]], [1.0])
    cursor = binary("vmul", cursor, keep_cursor)

    key_w = np.zeros((1, 3, 1, 1), np.float32)
    key_w[0, 0, 0, 0] = 1.0
    key_w[0, 1, 0, 0] = 1.0 / 64.0
    key = conv("coords", key_w)
    selectable = binary("max2", blue, corner)
    after_cursor = binary("sub", key, cursor)
    after_cursor = unary("relu", scale_spatial(after_cursor, 4096.0))
    after_cursor = binary("min2", after_cursor, one)
    eligible = binary("mul", selectable, after_cursor)
    has_eligible = unary("gmax", eligible)

    click_score = binary("sub", scale_spatial(eligible, 10.0), key)
    graph.append({"id": "click", "op": "relu", "in": [click_score]})

    # Flood the selected click component to advance the cursor beyond it.
    best_click = unary("gmax", "click")
    click_difference = binary("sub", "click", best_click)
    flood_seed = unary(
        "relu", scale_spatial(click_difference, 100000.0, 1.0)
    )
    flood_seed = binary("min2", flood_seed, one)
    flood_seed = binary("mul", flood_seed, eligible)

    flood_w = np.zeros((1, 1, 3, 3), np.float32)
    flood_w[0, 0, 1, 1] = 1
    flood_w[0, 0, 0, 1] = 1
    flood_w[0, 0, 2, 1] = 1
    flood_w[0, 0, 1, 0] = 1
    flood_w[0, 0, 1, 2] = 1
    flood_param = put(flood_w, "flood")

    outer = graph
    graph = []
    spread = node("conv", ["flood_state"], w=flood_param)
    spread = binary("min2", spread, one)
    spread = binary("min2", spread, selectable)
    flood_body = graph
    graph = outer
    graph.append({
        "id": "flood_loop",
        "op": "loop",
        "T": 64,
        "state": [{"id": "flood_state", "init": flood_seed}],
        "body": flood_body,
        "next": {"flood_state": spread},
    })

    clicked_end = unary(
        "gmax", binary("mul", "flood_loop.flood_state", key)
    )
    do_click = binary("vmul", no_move, has_eligible)
    cursor_delta = binary(
        "vadd", clicked_end, dense(cursor, [[-1.0]])
    )
    cursor_delta = binary("vmul", cursor_delta, do_click)
    next_cursor = binary("vadd", cursor, cursor_delta)
    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [next_cursor, position],
    })

    # Movement logits compete with the baseline. If no move improves it,
    # select another object when possible; otherwise spill.
    controls = vcat(move_scores, baseline, has_eligible)
    output_w = np.zeros((8, 6), np.float32)
    for i in range(4):
        output_w[1 + i, i] = 1.0
    output_w[5, 4] = 1.0
    output_w[5, 5] = -100000.0
    output_w[6, 4] = 1.0
    output_w[6, 5] = 100000.0
    output_b = np.zeros(8, np.float32)
    output_b[0] = -1000000.0
    output_b[6] = -100000.0
    output_b[7] = -1000000.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [controls],
        "w": put(output_w, "output_w"),
        "b": put(output_b, "output_b"),
    })

    return {"graph": graph, "memory_size": 3, "params": params}
