import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]

    def param(value):
        name = "p%d" % counter[0]
        counter[0] += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(name, op, inputs, **fields):
        graph.append({"id": name, "op": op, "in": list(inputs), **fields})
        return name

    def conv(name, source, weights, bias=None):
        fields = {"w": param(weights)}
        if bias is not None:
            fields["b"] = param(bias)
        return node(name, "conv", [source], **fields)

    def dense(name, source, weights, bias=None):
        fields = {"w": param(weights)}
        if bias is not None:
            fields["b"] = param(bias)
        return node(name, "dense", [source], **fields)

    def section(name, source, start, stop):
        return node(name, "slice", [source], start=start, stop=stop)

    shapes = [(3, 3), (3, 6), (6, 3), (6, 6), (9, 9)]
    anchors = [(1, 1), (1, 2), (2, 1), (2, 2), (4, 4)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Detect rectangular selected and unselected pieces.
    template_w = np.zeros((8, 16, 7, 7), dtype=np.float32)
    template_b = np.zeros(8, dtype=np.float32)
    edge_w = np.zeros((4, 16, 7, 7), dtype=np.float32)

    for s, (h, w) in enumerate(shapes[:4]):
        ay, ax = anchors[s]
        for y in range(h):
            for x in range(w):
                ky, kx = y - ay + 3, x - ax + 3
                border = y == 0 or y == h - 1 or x == 0 or x == w - 1
                if border:
                    template_w[s, 14, ky, kx] = 2
                    template_w[s, 0, ky, kx] = 2
                    template_w[4 + s, 14, ky, kx] = 2
                    edge_w[s, 14, ky, kx] = 2
                else:
                    template_w[s, 0, ky, kx] = 2
                    template_w[4 + s, 4, ky, kx] = 2
                    template_w[4 + s, 5, ky, kx] = 2
        template_b[s] = -(2 * h * w - 1)
        template_b[4 + s] = -(2 * h * w - 1)

    conv("template_scores", "x0", template_w, template_b)
    node("template_hits", "relu", ["template_scores"])
    section("active_raw", "template_hits", 0, 4)
    section("inactive_rect", "template_hits", 4, 8)
    conv("edge_scores", "x0", edge_w, np.full(4, -7, dtype=np.float32))
    node("edge_hits", "relu", ["edge_scores"])
    node("active_checked", "min2", ["active_raw", "edge_hits"])
    node("active_types", "gmax", ["active_checked"])

    type_w = np.array([
        [1, -1, -1, -1],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    dense("type_scores", "active_types", type_w)
    node("type_mask", "vrelu", ["type_scores"])
    node("active_rect", "mul", ["active_checked", "type_mask"])

    # Detect the cross piece.
    cross_w = np.zeros((1, 16, 7, 7), dtype=np.float32)
    cross_count = 0
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if abs(dy) <= 1 or abs(dx) <= 1:
                cross_w[0, 11, dy + 3, dx + 3] = 2
                cross_count += 1
            else:
                cross_w[0, 11, dy + 3, dx + 3] = -2
    conv("cross_score", "x0", cross_w,
         np.array([-(2 * cross_count - 1)], dtype=np.float32))
    node("cross_piece", "relu", ["cross_score"])
    conv("zero_cross", "cross_piece",
         np.zeros((1, 1, 1, 1), dtype=np.float32))
    node("active", "concat", ["active_rect", "zero_cross"])
    node("inactive", "concat", ["inactive_rect", "cross_piece"])
    node("pieces", "max2", ["active", "inactive"])

    # Detect rectangular outlines, allowing visible pieces to cover samples.
    occluders = [4, 14, 0, 5, 11, 12, 13]
    goal_shift_w = np.zeros((8, 16, 3, 3), dtype=np.float32)
    for c in range(4):
        dy = 1 if c & 2 else 0
        dx = 1 if c & 1 else 0
        for color in occluders:
            goal_shift_w[c, color, 1 + dy, 1 + dx] = 1
        goal_shift_w[4 + c, 4, 1 + dy, 1 + dx] = 1
    conv("goal_shift", "x0", goal_shift_w)

    goal_w = np.zeros((8, 8, 7, 7), dtype=np.float32)
    goal_b = np.zeros(8, dtype=np.float32)
    for s, (h, w) in enumerate(shapes[:4]):
        top = -2 if h == 3 else -3
        bottom = 2 if h == 3 else 4
        left = -2 if w == 3 else -3
        right = 2 if w == 3 else 4
        samples = [
            (top, left), (top, right),
            (bottom, left), (bottom, right),
            (top, 0), (bottom, 0), (0, left), (0, right),
        ]
        for dy0, dx0 in samples:
            dy, dx = dy0, dx0
            channel = 0
            if dy == 4:
                channel |= 2
                dy = 3
            if dx == 4:
                channel |= 1
                dx = 3
            goal_w[s, channel, dy + 3, dx + 3] += 2
            goal_w[4 + s, 4 + channel, dy + 3, dx + 3] += 1
        goal_b[s] = -15
        goal_b[4 + s] = -2

    conv("goal_scores", "goal_shift", goal_w, goal_b)
    node("goal_hits", "relu", ["goal_scores"])
    section("goal_possible", "goal_hits", 0, 4)
    section("goal_visible", "goal_hits", 4, 8)
    node("rect_goals", "min2", ["goal_possible", "goal_visible"])

    # Detect cross outline.
    cross_shift_w = np.zeros((8, 16, 5, 5), dtype=np.float32)
    shifts = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    for c, (dy, dx) in enumerate(shifts):
        for color in occluders:
            cross_shift_w[c, color, dy + 2, dx + 2] = 1
        cross_shift_w[4 + c, 4, dy + 2, dx + 2] = 1
    conv("cross_goal_shift", "x0", cross_shift_w)

    cross_goal_w = np.zeros((2, 8, 7, 7), dtype=np.float32)
    for t in (-2, 0, 2):
        for c, dy, dx in [
            (0, -3, t), (1, 3, t),
            (2, t, -3), (3, t, 3),
        ]:
            cross_goal_w[0, c, dy + 3, dx + 3] += 2
            cross_goal_w[1, 4 + c, dy + 3, dx + 3] += 1
    conv("cross_goal_scores", "cross_goal_shift", cross_goal_w,
         np.array([-23, -2], dtype=np.float32))
    node("cross_goal_hits", "relu", ["cross_goal_scores"])
    section("cross_goal_possible", "cross_goal_hits", 0, 1)
    section("cross_goal_visible", "cross_goal_hits", 1, 2)
    node("cross_goal", "min2", ["cross_goal_possible", "cross_goal_visible"])
    node("goals", "concat", ["rect_goals", "cross_goal"])

    # Selected position and whether it is currently on a matching goal.
    conv("active_sum", "active", np.ones((1, 5, 1, 1), dtype=np.float32))
    section("position_coords", "coords", 0, 2)
    node("position_map", "mul", ["active_sum", "position_coords"])
    node("position", "gmax", ["position_map"])
    node("active_goal_map", "mul", ["active", "goals"])
    node("active_goal_types", "gmax", ["active_goal_map"])
    dense("active_on_goal", "active_goal_types",
          np.ones((1, 5), dtype=np.float32))

    # Reserve the selected piece's goal until another piece is selected.
    node("old_reserved_position", "vslice", ["mem"], start=6, stop=8)
    node("old_reserved_flag", "vslice", ["mem"], start=8, stop=9)
    node("previous_click", "vslice", ["aprev"], start=6, stop=7)
    dense("not_previous_click", "previous_click",
          np.array([[-1]], dtype=np.float32),
          np.array([1], dtype=np.float32))
    dense("not_on_goal", "active_on_goal",
          np.array([[-1]], dtype=np.float32),
          np.array([1], dtype=np.float32))
    node("old_reservation_live", "vmul",
         ["old_reserved_flag", "not_previous_click"])
    node("keep_old_reservation", "vmul",
         ["old_reservation_live", "not_on_goal"])
    node("reserved_flag", "vadd", ["active_on_goal", "keep_old_reservation"])
    node("new_reserved_position", "vmul", ["position", "active_on_goal"])
    node("kept_reserved_position", "vmul",
         ["old_reserved_position", "keep_old_reservation"])
    node("reserved_position", "vadd",
         ["new_reserved_position", "kept_reserved_position"])

    node("reserved_difference", "sub", ["position_coords", "reserved_position"])
    node("reserved_absolute", "abs", ["reserved_difference"])
    conv("reserved_match_score", "reserved_absolute",
         np.array([[[[-126]], [[-126]]]], dtype=np.float32),
         np.array([1], dtype=np.float32))
    node("reserved_match", "relu", ["reserved_match_score"])
    node("reserved_map", "mul", ["reserved_match", "reserved_flag"])
    node("unoccupied_difference", "sub", ["goals", "pieces"])
    node("unoccupied_goals", "relu", ["unoccupied_difference"])
    node("open_difference", "sub", ["unoccupied_goals", "reserved_map"])
    node("open_goals", "relu", ["open_difference"])

    # Terrain and movable footprints.
    wall_w = np.zeros((1, 16, 1, 1), dtype=np.float32)
    wall_w[0, 2, 0, 0] = 1
    conv("hard_walls", "x0", wall_w)
    gap_w = np.zeros((1, 16, 1, 1), dtype=np.float32)
    gap_w[0, 15, 0, 0] = 1
    conv("gaps", "x0", gap_w)
    node("terrain", "max2", ["hard_walls", "gaps"])

    footprint_w = np.zeros((1, 5, 7, 7), dtype=np.float32)
    for s, (h, w) in enumerate(shapes[:4]):
        ay, ax = anchors[s]
        for y in range(h):
            for x in range(w):
                footprint_w[0, s, 3 - (y - ay), 3 - (x - ax)] = 1
    conv("inactive_footprints", "inactive", footprint_w)
    section("ones", "coords", 2, 3)
    node("inactive_occupied", "min2", ["inactive_footprints", "ones"])

    obstacle_w = np.zeros((1, 16, 1, 1), dtype=np.float32)
    for color in (11, 12, 13):
        obstacle_w[0, color, 0, 0] = 1
    conv("other_obstacles", "x0", obstacle_w)
    node("terrain_and_pieces", "max2", ["terrain", "inactive_occupied"])
    node("solid", "max2", ["terrain_and_pieces", "other_obstacles"])

    wall_shift_w = np.zeros((5, 1, 3, 3), dtype=np.float32)
    wall_shift_w[0, 0, 1, 1] = 1
    wall_shift_w[1, 0, 0, 1] = 1
    wall_shift_w[2, 0, 2, 1] = 1
    wall_shift_w[3, 0, 1, 0] = 1
    wall_shift_w[4, 0, 1, 2] = 1
    wall_shift_param = param(wall_shift_w)
    node("solid_shift", "conv", ["solid"], w=wall_shift_param)
    node("hard_shift", "conv", ["hard_walls"], w=wall_shift_param)

    clear_w = np.zeros((5, 5, 7, 7), dtype=np.float32)
    for s, (h, w) in enumerate(shapes[:4]):
        ay, ax = anchors[s]
        for y in range(h):
            for x in range(w):
                clear_w[s, 0, y - ay + 3, x - ax + 3] = -1
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if abs(dy) <= 1 or abs(dx) <= 1:
                clear_w[4, 0, dy + 3, dx + 3] = -1
    for t in (-1, 0, 1):
        clear_w[4, 1, 0, t + 3] = -1
        clear_w[4, 2, 6, t + 3] = -1
        clear_w[4, 3, t + 3, 0] = -1
        clear_w[4, 4, t + 3, 6] = -1

    clear_param = param(clear_w)
    clear_bias = param(np.ones(5, dtype=np.float32))
    node("safe_clearance_scores", "conv", ["solid_shift"],
         w=clear_param, b=clear_bias)
    node("virtual_clearance_scores", "conv", ["hard_shift"],
         w=clear_param, b=clear_bias)
    node("safe_valid", "relu", ["safe_clearance_scores"])
    node("virtual_valid", "relu", ["virtual_clearance_scores"])
    node("safe_goals", "mul", ["open_goals", "safe_valid"])
    node("virtual_goals", "mul", ["open_goals", "virtual_valid"])

    seed_w = np.zeros((5, 5, 1, 1), dtype=np.float32)
    for s in range(5):
        seed_w[s, s, 0, 0] = 96
    seed_param = param(seed_w)
    node("safe_seed", "conv", ["safe_goals"], w=seed_param)
    node("virtual_seed", "conv", ["virtual_goals"], w=seed_param)

    # Shared one-pixel channelwise shifts.
    shift_params = []
    for dy, dx in directions:
        w = np.zeros((5, 5, 3, 3), dtype=np.float32)
        for s in range(5):
            w[s, s, 1 + dy, 1 + dx] = 1
        shift_params.append(param(w))

    cost_w = np.zeros((5, 5, 1, 1), dtype=np.float32)
    for s in range(5):
        cost_w[s, s, 0, 0] = 1
    cost_param = param(cost_w)
    cost_bias = param(-np.ones(5, dtype=np.float32))

    def add_flow(prefix, seed, valid):
        state = prefix + "_state"
        body = []
        neighbor_ids = []
        for d in range(4):
            previous = state
            for k in range(3):
                name = "%s_d%d_s%d" % (prefix, d, k)
                body.append({
                    "id": name, "op": "conv", "in": [previous],
                    "w": shift_params[d],
                })
                previous = name
            neighbor_ids.append(previous)

        body.extend([
            {"id": prefix + "_ud", "op": "max2",
             "in": [neighbor_ids[0], neighbor_ids[1]]},
            {"id": prefix + "_lr", "op": "max2",
             "in": [neighbor_ids[2], neighbor_ids[3]]},
            {"id": prefix + "_best", "op": "max2",
             "in": [prefix + "_ud", prefix + "_lr"]},
            {"id": prefix + "_cost", "op": "conv",
             "in": [prefix + "_best"], "w": cost_param, "b": cost_bias},
            {"id": prefix + "_positive", "op": "relu",
             "in": [prefix + "_cost"]},
            {"id": prefix + "_allowed", "op": "mul",
             "in": [prefix + "_positive", valid]},
            {"id": prefix + "_next", "op": "max2",
             "in": [prefix + "_allowed", seed]},
        ])
        graph.append({
            "id": prefix,
            "op": "loop",
            "T": 35,
            "state": [{"id": state, "init": seed}],
            "body": body,
            "next": {state: prefix + "_next"},
        })
        return prefix + "." + state

    def shift_by(prefix, source, direction, count):
        previous = source
        for k in range(count):
            name = "%s_%d" % (prefix, k)
            node(name, "conv", [previous], w=shift_params[direction])
            previous = name
        return previous

    def neighbors(prefix, source, count):
        return [
            shift_by("%s_d%d" % (prefix, d), source, d, count)
            for d in range(4)
        ]

    safe_wave = add_flow("safe_flow", "safe_seed", "safe_valid")
    virtual_wave = add_flow("virtual_flow", "virtual_seed", "virtual_valid")
    safe_neighbors = neighbors("safe_neighbor", safe_wave, 3)

    node("inactive_difference", "sub", ["inactive", "goals"])
    node("inactive_unplaced", "relu", ["inactive_difference"])

    # Evaluate other selectable pieces from neighboring reachable anchors.
    near_maps = []
    for d in range(4):
        farther = shift_by("candidate_far_d%d" % d, safe_neighbors[d], d, 3)
        node("candidate_near_d%d" % d, "max2", [safe_neighbors[d], farther])
        near_maps.append("candidate_near_d%d" % d)
    node("candidate_ud", "max2", [near_maps[0], near_maps[1]])
    node("candidate_lr", "max2", [near_maps[2], near_maps[3]])
    node("candidate_reach", "max2", ["candidate_ud", "candidate_lr"])

    selectable_w = np.zeros((5, 5, 1, 1), dtype=np.float32)
    for s in range(4):
        selectable_w[s, s, 0, 0] = 1
    conv("selectable_unplaced", "inactive_unplaced", selectable_w)
    node("click_by_type", "mul", ["selectable_unplaced", "candidate_reach"])
    conv("click", "click_by_type", np.ones((1, 5, 1, 1), dtype=np.float32))
    node("click_max", "gmax", ["click"])
    dense("no_candidate_raw", "click_max",
          np.array([[-1]], dtype=np.float32),
          np.array([1], dtype=np.float32))
    node("no_candidate", "vrelu", ["no_candidate_raw"])
    dense("has_candidate", "no_candidate",
          np.array([[-1]], dtype=np.float32),
          np.array([1], dtype=np.float32))

    # Estimate the effect of a five-move push and find pusher stances.
    stance_maps = []
    for d, (dy, dx) in enumerate(directions):
        virtual_far = shift_by("virtual_far_d%d" % d, virtual_wave, d, 15)
        safe_far = shift_by("safe_far_d%d" % d, safe_wave, d, 15)
        node("gain_raw_%d" % d, "sub", [virtual_far, virtual_wave])
        node("gain_%d" % d, "relu", ["gain_raw_%d" % d])
        node("gain_indicator_%d" % d, "min2", ["gain_%d" % d, "ones"])
        node("target_indicator_%d" % d, "min2",
             ["gain_indicator_%d" % d, "inactive_unplaced"])

        priority_w = np.zeros((5, 5, 1, 1), dtype=np.float32)
        for s in range(5):
            priority_w[s, s, 0, 0] = 1
        conv("priority_base_%d" % d, safe_far, priority_w,
             np.full(5, 80, dtype=np.float32))
        node("priority_with_gain_%d" % d, "add",
             ["priority_base_%d" % d, "gain_%d" % d])
        node("target_score_%d" % d, "mul",
             ["priority_with_gain_%d" % d, "target_indicator_%d" % d])

        previous = shift_by("stance_shift_d%d" % d,
                            "target_score_%d" % d, d, 6)
        stance_w = np.zeros((5, 5, 7, 7), dtype=np.float32)

        for p in range(4):
            ph, pw = shapes[p]
            pay, pax = anchors[p]
            for t in range(5):
                th, tw = shapes[t]
                tay, tax = anchors[t]
                if dy == 1:
                    by, bx = ph + tay - pay, tax - pax
                    offsets = [(by, bx + 3 * q)
                               for q in range(-(pw // 3) + 1, tw // 3)]
                elif dy == -1:
                    by, bx = -(th + pay - tay), tax - pax
                    offsets = [(by, bx + 3 * q)
                               for q in range(-(pw // 3) + 1, tw // 3)]
                elif dx == 1:
                    by, bx = tay - pay, pw + tax - pax
                    offsets = [(by + 3 * q, bx)
                               for q in range(-(ph // 3) + 1, th // 3)]
                else:
                    by, bx = tay - pay, -(tw + pax - tax)
                    offsets = [(by + 3 * q, bx)
                               for q in range(-(ph // 3) + 1, th // 3)]

                for oy, ox in offsets:
                    ry, rx = oy - 6 * dy, ox - 6 * dx
                    if -3 <= ry <= 3 and -3 <= rx <= 3:
                        stance_w[p, t, ry + 3, rx + 3] = 1

        conv("stance_raw_%d" % d, previous, stance_w)
        node("stance_%d" % d, "mul",
             ["stance_raw_%d" % d, "safe_valid"])
        stance_maps.append("stance_%d" % d)

    node("stance_ud", "max2", [stance_maps[0], stance_maps[1]])
    node("stance_lr", "max2", [stance_maps[2], stance_maps[3]])
    node("push_seed", "max2", ["stance_ud", "stance_lr"])
    push_wave = add_flow("push_flow", "push_seed", "safe_valid")
    push_neighbors = neighbors("push_neighbor", push_wave, 3)

    def values_at_active(prefix, maps):
        scalars = []
        for d, source in enumerate(maps):
            node("%s_map_%d" % (prefix, d), "mul", [source, "active"])
            node("%s_max_%d" % (prefix, d), "gmax",
                 ["%s_map_%d" % (prefix, d)])
            dense("%s_scalar_%d" % (prefix, d), "%s_max_%d" % (prefix, d),
                  np.ones((1, 5), dtype=np.float32))
            scalars.append("%s_scalar_%d" % (prefix, d))
        node(prefix, "vconcat", scalars)

    values_at_active("safe_values", safe_neighbors)
    values_at_active("push_values", push_neighbors)
    values_at_active("at_stance", stance_maps)

    node("active_value_map", "mul", [safe_wave, "active"])
    node("active_value_types", "gmax", ["active_value_map"])
    dense("active_value", "active_value_types",
          np.ones((1, 5), dtype=np.float32))
    dense("unreachable_raw", "active_value",
          np.array([[-1]], dtype=np.float32),
          np.array([1], dtype=np.float32))
    node("unreachable", "vrelu", ["unreachable_raw"])
    node("need_change_raw", "vadd", ["unreachable", "active_on_goal"])
    dense("normal_gate_raw", "need_change_raw",
          np.array([[-1]], dtype=np.float32),
          np.array([1], dtype=np.float32))
    node("normal_gate", "vrelu", ["normal_gate_raw"])
    dense("change_gate", "normal_gate",
          np.array([[-1]], dtype=np.float32),
          np.array([1], dtype=np.float32))
    node("click_gate", "vmul", ["change_gate", "has_candidate"])
    node("push_gate", "vmul", ["change_gate", "no_candidate"])
    node("normal_scores", "vmul", ["safe_values", "normal_gate"])
    node("push_navigation", "vmul", ["push_values", "push_gate"])
    node("push_now", "vmul", ["at_stance", "push_gate"])

    # Penalize directions that did not move the selected piece.
    node("old_position", "vslice", ["mem"], start=0, stop=2)
    dense("negative_old_position", "old_position",
          -np.eye(2, dtype=np.float32))
    node("position_difference", "vadd", ["position", "negative_old_position"])
    node("positive_difference", "vrelu", ["position_difference"])
    dense("negative_difference_raw", "position_difference",
          -np.eye(2, dtype=np.float32))
    node("negative_difference", "vrelu", ["negative_difference_raw"])
    node("absolute_difference", "vadd",
         ["positive_difference", "negative_difference"])
    dense("same_position_raw", "absolute_difference",
          np.array([[-100, -100]], dtype=np.float32),
          np.array([1], dtype=np.float32))
    node("same_position", "vrelu", ["same_position_raw"])
    node("old_penalties", "vslice", ["mem"], start=2, stop=6)
    node("previous_direction", "vslice", ["aprev"], start=1, stop=5)
    node("accumulated_penalties", "vadd",
         ["old_penalties", "previous_direction"])
    node("penalties", "vmul", ["accumulated_penalties", "same_position"])
    node("mem_next", "vconcat",
         ["position", "penalties", "reserved_position", "reserved_flag"])

    node("features", "vconcat",
         ["normal_scores", "push_navigation", "push_now",
          "penalties", "click_gate"])
    out_w = np.zeros((8, 17), dtype=np.float32)
    out_b = np.full(8, -1000000, dtype=np.float32)
    for d in range(4):
        out_w[1 + d, d] = 1
        out_w[1 + d, 4 + d] = 1
        out_w[1 + d, 8 + d] = 3
        out_w[1 + d, 12 + d] = -200
        out_b[1 + d] = 0
    out_w[6, 16] = 2000
    out_b[6] = -1000
    dense("logits", "features", out_w, out_b)

    return {"graph": graph, "memory_size": 9, "params": params}
