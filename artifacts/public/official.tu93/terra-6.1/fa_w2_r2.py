import numpy as np


def build():
    graph = []
    params = {}

    def weight(name, value):
        params[name] = np.asarray(value, dtype=np.float32)

    def node(name, op, inputs, **fields):
        item = {"id": name, "op": op, "in": inputs}
        item.update(fields)
        graph.append(item)
        return name

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    scales = (1, 2)

    node("track", "slice", ["x0"], start=2, stop=3)
    node("row_coord", "slice", ["coords"], start=0, stop=1)
    node("col_coord", "slice", ["coords"], start=1, stop=2)
    node("coord_pair", "slice", ["coords"], start=0, stop=2)
    node("ones", "slice", ["coords"], start=2, stop=3)
    node("click", "slice", ["x0"], start=5, stop=6)

    # Sprite detectors, track edges, and shift kernels at each camera scale.
    for scale in scales:
        p = "s%d_" % scale
        size = 3 * scale
        k = 3 if scale == 1 else 7
        anchor = (size - 1) // 2
        start = k // 2 - anchor

        templates = []
        for d in range(4):
            templates.append((9, 4, d, "player"))
        for color in (8, 12):
            for marker in (15, 11):
                for d in range(4):
                    templates.append((
                        color, marker, d,
                        "blue" if color == 8 else "patrol"
                    ))
        for d in range(4):
            templates.append((13, 11, d, "pursuer"))

        detector = np.zeros((len(templates), 16, k, k), np.float32)
        for t, (color, marker, d, _) in enumerate(templates):
            dy, dx = directions[d]
            mr, mc = 1 + dy, 1 + dx
            for r in range(size):
                for c in range(size):
                    channel = marker if (
                        r // scale == mr and c // scale == mc
                    ) else color
                    detector[t, channel, start + r, start + c] = 2.0

        weight(p + "det_w", detector)
        weight(p + "det_b",
               np.full(len(templates), -(2 * size * size - 1), np.float32))
        node(p + "det_raw", "conv", ["x0"], w=p + "det_w", b=p + "det_b")
        node(p + "det", "relu", [p + "det_raw"])

        player_w = np.zeros((1, len(templates), 1, 1), np.float32)
        player_w[0, :4, 0, 0] = 1.0
        weight(p + "player_w", player_w)
        node(p + "player_raw", "conv", [p + "det"], w=p + "player_w")
        node(p + "blue_static", "slice", [p + "det"], start=4, stop=8)
        node(p + "blue_active", "slice", [p + "det"], start=8, stop=12)

        enemy_w = np.zeros((4, len(templates), 1, 1), np.float32)
        pursuer_w = np.zeros_like(enemy_w)
        lane_w = np.zeros_like(enemy_w)
        for t, (_, _, d, kind) in enumerate(templates):
            if kind != "player":
                enemy_w[d, t, 0, 0] = 1.0
            if kind == "pursuer":
                pursuer_w[d, t, 0, 0] = 1.0
            if kind == "patrol":
                if d in (0, 1):
                    lane_w[0, t, 0, 0] = 1.0
                    lane_w[1, t, 0, 0] = 1.0
                else:
                    lane_w[2, t, 0, 0] = 1.0
                    lane_w[3, t, 0, 0] = 1.0

        weight(p + "enemy_w", enemy_w)
        weight(p + "pursuer_w", pursuer_w)
        weight(p + "lane_w", lane_w)
        node(p + "enemy_raw", "conv", [p + "det"], w=p + "enemy_w")
        node(p + "pursuer_dirs", "conv", [p + "det"],
             w=p + "pursuer_w")
        node(p + "lane_seed_raw", "conv", [p + "det"], w=p + "lane_w")
        node(p + "lane_seed", "min2", [p + "lane_seed_raw", "ones"])

        forward_start = np.zeros((4, 1, 7, 7), np.float32)
        forward_more = np.zeros((4, 4, 7, 7), np.float32)
        reverse_more = np.zeros((4, 4, 7, 7), np.float32)
        for d, (dy, dx) in enumerate(directions):
            forward_start[d, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
            forward_more[d, d, 3 + 3 * dy, 3 + 3 * dx] = 1.0
            reverse_more[d, d, 3 - 3 * dy, 3 - 3 * dx] = 1.0

            individual = np.zeros((1, 1, 7, 7), np.float32)
            individual[0, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
            weight(p + "individual%d" % d, individual)

        weight(p + "forward_start", forward_start)
        weight(p + "forward_more", forward_more)
        weight(p + "reverse_more", reverse_more)
        weight(p + "sum4", np.ones((1, 4, 1, 1), np.float32))

        edge = node(p + "edge0", "conv", ["track"],
                    w=p + "forward_start")
        for i in range(scale - 1):
            edge = node(p + "edge%d" % (i + 1), "conv", [edge],
                        w=p + "forward_more")

        goal_w = np.zeros((1, 16, k, k), np.float32)
        goal_w[0, 14, start:start + size, start:start + size] = 200.0
        weight(p + "goal_w", goal_w)
        weight(p + "goal_b",
               np.array([-(2 * size * size - 1) * 100.0], np.float32))
        node(p + "goal_raw", "conv", ["x0"],
             w=p + "goal_w", b=p + "goal_b")
        node(p + "goal", "relu", [p + "goal_raw"])

    # Select the scale using the complete player template.
    node("large_player_present", "gmax", ["s2_player_raw"])
    weight("small_gate_w", np.array([[-1.0]], np.float32))
    weight("small_gate_b", np.array([1.0], np.float32))
    node("small_gate", "dense", ["large_player_present"],
         w="small_gate_w", b="small_gate_b")
    node("s1_player", "mul", ["s1_player_raw", "small_gate"])
    node("s2_player", "min2", ["s2_player_raw", "ones"])
    node("player_map", "max2", ["s1_player", "s2_player"])

    # Last twelve player positions, stored as normalized coordinates.
    node("player_coords_map", "mul", ["coord_pair", "player_map"])
    node("player_coords", "gmax", ["player_coords_map"])
    node("old_history", "vslice", ["mem"], start=0, stop=22)
    node("history_next", "vconcat", ["player_coords", "old_history"])

    weight("history_match_w",
           np.array([[[[-63.0]], [[-63.0]]]], np.float32))
    weight("history_match_b", np.array([1.0], np.float32))
    history_maps = []
    for i in range(12):
        node("history_pos%d" % i, "vslice", ["mem"],
             start=2 * i, stop=2 * i + 2)
        node("history_diff%d" % i, "sub",
             ["coord_pair", "history_pos%d" % i])
        node("history_abs%d" % i, "abs", ["history_diff%d" % i])
        node("history_raw%d" % i, "conv", ["history_abs%d" % i],
             w="history_match_w", b="history_match_b")
        node("history_map%d" % i, "relu", ["history_raw%d" % i])
        history_maps.append("history_map%d" % i)

    history = history_maps[0]
    for i in range(1, len(history_maps)):
        history = node("history_sum%d" % i, "add",
                       [history, history_maps[i]])

    weight("history_path_w", np.array([[[[0.5]]]], np.float32))
    weight("history_move_w", np.array([[[[2.0]]]], np.float32))
    node("history_path_cost", "conv", [history], w="history_path_w")
    node("history_move_cost", "conv", [history], w="history_move_w")

    # Remember one inactive blue enemy in each 8x8 image region. A slot
    # contains its row, column, and heading. A purple sprite at the stored
    # position preserves the slot while the blue enemy is occluded.
    node("small_blue", "mul", ["s1_blue_static", "small_gate"])
    node("large_blue", "mul", ["s2_blue_static", "large_player_present"])
    node("static_blue", "max2", ["small_blue", "large_blue"])

    weight("blue_sum_w", np.ones((1, 4, 1, 1), np.float32))
    weight("blue_dir_w", np.array([[[[1.0]], [[2.0]],
                                     [[3.0]], [[4.0]]]], np.float32))
    node("blue_presence_raw", "conv", ["static_blue"], w="blue_sum_w")
    node("blue_presence", "min2", ["blue_presence_raw", "ones"])
    node("blue_direction", "conv", ["static_blue"], w="blue_dir_w")

    region_weights = {}
    region_biases = {}
    for label in ("rl", "rh", "cl", "ch"):
        region_weights[label] = np.zeros((64, 3, 1, 1), np.float32)
        region_biases[label] = np.zeros(64, np.float32)

    for br in range(8):
        for bc in range(8):
            i = 8 * br + bc
            region_weights["rl"][i, 0, 0, 0] = 126.0
            region_biases["rl"][i] = 1.0 - 16.0 * br
            region_weights["rh"][i, 0, 0, 0] = -126.0
            region_biases["rh"][i] = 16.0 * br + 15.0
            region_weights["cl"][i, 1, 0, 0] = 126.0
            region_biases["cl"][i] = 1.0 - 16.0 * bc
            region_weights["ch"][i, 1, 0, 0] = -126.0
            region_biases["ch"][i] = 16.0 * bc + 15.0

    for label in ("rl", "rh", "cl", "ch"):
        weight("region_" + label + "_w", region_weights[label])
        weight("region_" + label + "_b", region_biases[label])
        node("region_" + label, "conv", ["coords"],
             w="region_" + label + "_w",
             b="region_" + label + "_b")

    node("region_r_min", "min2", ["region_rl", "region_rh"])
    node("region_c_min", "min2", ["region_cl", "region_ch"])
    node("region_r_pos", "relu", ["region_r_min"])
    node("region_c_pos", "relu", ["region_c_min"])
    node("region_r", "min2", ["region_r_pos", "ones"])
    node("region_c", "min2", ["region_c_pos", "ones"])
    node("region_mask", "mul", ["region_r", "region_c"])
    node("blue_regions", "mul", ["region_mask", "blue_presence"])

    # Select the first blue position in each region.
    weight("rank_w",
           np.array([[[[-4032.0]], [[-63.0]], [[0.0]]]], np.float32))
    weight("rank_b", np.array([4096.0], np.float32))
    node("rank", "conv", ["coords"], w="rank_w", b="rank_b")
    node("region_rank", "mul", ["blue_regions", "rank"])
    node("best_rank", "gmax", ["region_rank"])
    node("rank_difference", "sub", ["rank", "best_rank"])
    weight("rank_threshold_w",
           2.0 * np.eye(64, dtype=np.float32).reshape(64, 64, 1, 1))
    weight("rank_threshold_b", np.full(64, 1.5, np.float32))
    node("rank_threshold", "conv", ["rank_difference"],
         w="rank_threshold_w", b="rank_threshold_b")
    node("rank_positive", "relu", ["rank_threshold"])
    node("rank_selected", "min2", ["rank_positive", "ones"])
    node("selected_blue", "mul", ["rank_selected", "blue_regions"])

    node("selected_row_map", "mul", ["selected_blue", "row_coord"])
    node("selected_col_map", "mul", ["selected_blue", "col_coord"])
    node("selected_dir_map", "mul", ["selected_blue", "blue_direction"])
    node("selected_row", "gmax", ["selected_row_map"])
    node("selected_col", "gmax", ["selected_col_map"])
    node("selected_dir", "gmax", ["selected_dir_map"])

    node("old_blue_row", "vslice", ["mem"], start=24, stop=88)
    node("old_blue_col", "vslice", ["mem"], start=88, stop=152)
    node("old_blue_dir", "vslice", ["mem"], start=152, stop=216)
    node("old_row_diff", "sub", ["row_coord", "old_blue_row"])
    node("old_col_diff", "sub", ["col_coord", "old_blue_col"])
    node("old_row_abs", "abs", ["old_row_diff"])
    node("old_col_abs", "abs", ["old_col_diff"])
    node("old_distance", "add", ["old_row_abs", "old_col_abs"])

    weight("old_match_w",
           -63.0 * np.eye(64, dtype=np.float32).reshape(64, 64, 1, 1))
    weight("old_match_b", np.ones(64, np.float32))
    node("old_match_raw", "conv", ["old_distance"],
         w="old_match_w", b="old_match_b")
    node("old_match", "relu", ["old_match_raw"])

    node("purple12", "slice", ["x0"], start=12, stop=13)
    node("purple13", "slice", ["x0"], start=13, stop=14)
    node("purple", "max2", ["purple12", "purple13"])
    node("hidden_blue", "mul", ["old_match", "purple"])
    node("hidden_flags", "gmax", ["hidden_blue"])

    weight("not_hidden_w", -np.eye(64, dtype=np.float32))
    weight("not_hidden_b", np.ones(64, np.float32))
    node("not_hidden", "dense", ["hidden_flags"],
         w="not_hidden_w", b="not_hidden_b")

    for field in ("row", "col", "dir"):
        node("keep_old_" + field, "vmul",
             ["old_blue_" + field, "hidden_flags"])
        node("keep_new_" + field, "vmul",
             ["selected_" + field, "not_hidden"])
        node("next_blue_" + field, "vadd",
             ["keep_old_" + field, "keep_new_" + field])

    node("mem_next", "vconcat",
         ["history_next", "next_blue_row", "next_blue_col", "next_blue_dir"])

    weight("sum64_w", np.ones((1, 64, 1, 1), np.float32))
    node("hidden_sum_raw", "conv", ["hidden_blue"], w="sum64_w")
    node("hidden_sum", "min2", ["hidden_sum_raw", "ones"])
    node("hidden_direction_channels", "mul", ["hidden_blue", "old_blue_dir"])
    node("hidden_direction_sum", "conv", ["hidden_direction_channels"],
         w="sum64_w")
    weight("ghost_direction_w", np.ones((4, 1, 1, 1), np.float32))
    weight("ghost_direction_b", -np.arange(1, 5, dtype=np.float32))
    node("ghost_direction_diff", "conv", ["hidden_direction_sum"],
         w="ghost_direction_w", b="ghost_direction_b")
    node("ghost_direction_abs", "abs", ["ghost_direction_diff"])
    node("ghost_direction_raw", "sub", ["ones", "ghost_direction_abs"])
    node("ghost_direction_pos", "relu", ["ghost_direction_raw"])
    node("ghost_directions", "mul", ["ghost_direction_pos", "hidden_sum"])

    node("s1_ghost", "mul", ["ghost_directions", "small_gate"])
    node("s2_ghost", "mul", ["ghost_directions", "large_player_present"])
    for scale in scales:
        p = "s%d_" % scale
        node(p + "enemy_dirs", "max2", [p + "enemy_raw", p + "ghost"])
        node(p + "visible_blue_dirs", "max2",
             [p + "blue_static", p + "blue_active"])
        node(p + "blue_dirs", "max2",
             [p + "visible_blue_dirs", p + "ghost"])

    score_vectors = []

    for scale in scales:
        p = "s%d_" % scale
        edge = p + "edge%d" % (scale - 1)

        def shift_forward(name, source):
            value = node(name + "_0", "conv", [source],
                         w=p + "forward_start")
            for j in range(2 * scale - 1):
                value = node(name + "_%d" % (j + 1), "conv", [value],
                             w=p + "forward_more")
            return value

        def shift_reverse(name, source, steps):
            value = source
            for j in range(steps):
                value = node(name + "_%d" % j, "conv", [value],
                             w=p + "reverse_more")
            return value

        enemy_next = shift_reverse(
            p + "enemy_next", p + "enemy_dirs", 2 * scale
        )
        node(p + "threat1_sum", "conv", [enemy_next], w=p + "sum4")
        node(p + "threat1", "min2", [p + "threat1_sum", "ones"])

        blue_next = shift_reverse(
            p + "blue_next", p + "blue_dirs", 2 * scale
        )
        node(p + "blue_threat_sum", "conv", [blue_next], w=p + "sum4")
        node(p + "blue_threat", "min2", [p + "blue_threat_sum", "ones"])

        pursuer_two = shift_reverse(
            p + "pursuer_two", p + "pursuer_dirs", 4 * scale
        )
        node(p + "threat2_sum", "conv", [pursuer_two], w=p + "sum4")
        node(p + "threat2", "min2", [p + "threat2_sum", "ones"])

        # Trace straight patrol corridors.
        lane_state = p + "lane_state"
        lane_body = []

        def lane_node(name, op, inputs, **fields):
            item = {"id": name, "op": op, "in": inputs}
            item.update(fields)
            lane_body.append(item)
            return name

        lane_shift = lane_state
        for j in range(2 * scale):
            lane_shift = lane_node(
                p + "lane_shift%d" % j, "conv", [lane_shift],
                w=p + "forward_more"
            )
        lane_node(p + "lane_connected", "mul", [lane_shift, edge])
        lane_node(p + "lane_next", "max2",
                  [lane_state, p + "lane_connected"])
        graph.append({
            "id": p + "lane_loop",
            "op": "loop",
            "T": 12,
            "state": [{"id": lane_state, "init": p + "lane_seed"}],
            "body": lane_body,
            "next": {lane_state: p + "lane_next"},
        })

        lane_final = p + "lane_loop." + lane_state
        for d in range(4):
            node(p + "lane_dir%d" % d, "slice", [lane_final],
                 start=d, stop=d + 1)
        node(p + "lane_max01", "max2",
             [p + "lane_dir0", p + "lane_dir1"])
        node(p + "lane_max23", "max2",
             [p + "lane_dir2", p + "lane_dir3"])
        node(p + "lane_map", "max2",
             [p + "lane_max01", p + "lane_max23"])

        weight(p + "lane_cost_w", np.array([[[[9.0]]]], np.float32))
        weight(p + "threat_path_w", np.array([[[[3.0]]]], np.float32))
        weight(p + "blue_path_w", np.array([[[[28.0]]]], np.float32))
        weight(p + "threat_move_w", np.array([[[[203.0]]]], np.float32))
        node(p + "lane_cost", "conv", [p + "lane_map"],
             w=p + "lane_cost_w")
        node(p + "threat_path_cost", "conv", [p + "threat1"],
             w=p + "threat_path_w")
        node(p + "blue_path_cost", "conv", [p + "blue_threat"],
             w=p + "blue_path_w")
        node(p + "threat_move_cost", "conv", [p + "threat1"],
             w=p + "threat_move_w")
        node(p + "base_path_cost", "add",
             [p + "lane_cost", "history_path_cost"])
        node(p + "path_cost0", "add",
             [p + "base_path_cost", p + "threat_path_cost"])
        node(p + "path_cost", "add",
             [p + "path_cost0", p + "blue_path_cost"])

        # Penalize first moves that have no safe second move against an
        # already-active pursuer.
        threat2_neighbors = shift_forward(p + "threat2_neighbors",
                                          p + "threat2")
        node(p + "unsafe_edges", "mul", [edge, threat2_neighbors])
        node(p + "safe_edges", "sub", [edge, p + "unsafe_edges"])
        for d in range(4):
            node(p + "safe_dir%d" % d, "slice", [p + "safe_edges"],
                 start=d, stop=d + 1)
        node(p + "safe01", "max2", [p + "safe_dir0", p + "safe_dir1"])
        node(p + "safe23", "max2", [p + "safe_dir2", p + "safe_dir3"])
        node(p + "safe_any", "max2", [p + "safe01", p + "safe23"])
        node(p + "no_escape_raw", "sub", ["ones", p + "safe_any"])

        weight(p + "goal_unit_w", np.array([[[[0.01]]]], np.float32))
        node(p + "goal_unit", "conv", [p + "goal"], w=p + "goal_unit_w")
        node(p + "not_goal", "sub", ["ones", p + "goal_unit"])
        node(p + "no_escape", "mul",
             [p + "no_escape_raw", p + "not_goal"])
        weight(p + "deadend_w", np.array([[[[150.0]]]], np.float32))
        node(p + "deadend_cost", "conv", [p + "no_escape"],
             w=p + "deadend_w")

        # Backward value propagation over the track graph.
        step_w = np.zeros((4, 4, 1, 1), np.float32)
        for d in range(4):
            step_w[d, d, 0, 0] = 1.0
        weight(p + "step_w", step_w)
        weight(p + "step_b", np.full(4, -1.0, np.float32))

        value_state = p + "value_state"
        body = []

        def body_node(name, op, inputs, **fields):
            item = {"id": name, "op": op, "in": inputs}
            item.update(fields)
            body.append(item)
            return name

        body_node(p + "value_less_cost", "sub",
                  [value_state, p + "path_cost"])
        body_node(p + "value_clipped", "relu", [p + "value_less_cost"])
        body_node(p + "prop_first", "conv", [p + "value_clipped"],
                  w=p + "forward_start")

        parts = []
        for d in range(4):
            part = body_node(p + "prop_slice%d" % d, "slice",
                             [p + "prop_first"], start=d, stop=d + 1)
            for j in range(2 * scale - 1):
                part = body_node(
                    p + "prop_dir%d_shift%d" % (d, j), "conv", [part],
                    w=p + "individual%d" % d
                )
            parts.append(part)

        body_node(p + "prop_join", "concat", parts)
        body_node(p + "prop_edges", "mul", [p + "prop_join", edge])
        body_node(p + "prop_cost", "conv", [p + "prop_edges"],
                  w=p + "step_w", b=p + "step_b")
        body_node(p + "prop_positive", "relu", [p + "prop_cost"])
        for d in range(4):
            body_node(p + "value_dir%d" % d, "slice",
                      [p + "prop_positive"], start=d, stop=d + 1)
        body_node(p + "value_max01", "max2",
                  [p + "value_dir0", p + "value_dir1"])
        body_node(p + "value_max23", "max2",
                  [p + "value_dir2", p + "value_dir3"])
        body_node(p + "value_max", "max2",
                  [p + "value_max01", p + "value_max23"])
        body_node(p + "value_next", "max2", [value_state, p + "value_max"])

        graph.append({
            "id": p + "planner",
            "op": "loop",
            "T": 84,
            "state": [{"id": value_state, "init": p + "goal"}],
            "body": body,
            "next": {value_state: p + "value_next"},
        })
        final_value = p + "planner." + value_state

        node(p + "move_cost0", "add",
             [p + "lane_cost", "history_move_cost"])
        node(p + "move_cost1", "add",
             [p + "move_cost0", p + "threat_move_cost"])
        node(p + "move_cost", "add",
             [p + "move_cost1", p + "deadend_cost"])
        node(p + "scored_field", "sub", [final_value, p + "move_cost"])
        candidates = shift_forward(p + "candidate", p + "scored_field")

        edge_penalty_w = np.zeros((4, 4, 1, 1), np.float32)
        for d in range(4):
            edge_penalty_w[d, d, 0, 0] = 1000.0
        weight(p + "edge_penalty_w", edge_penalty_w)
        weight(p + "edge_penalty_b", np.full(4, -1000.0, np.float32))
        node(p + "edge_penalty", "conv", [edge],
             w=p + "edge_penalty_w", b=p + "edge_penalty_b")
        node(p + "candidate_valid", "add",
             [candidates, p + "edge_penalty"])
        node(p + "at_player", "mul",
             [p + "candidate_valid", p + "player"])
        node(p + "scores", "gmean", [p + "at_player"])
        score_vectors.append(p + "scores")

    node("all_scores", "vconcat", score_vectors + ["aprev"])
    output_w = np.zeros((8, 16), np.float32)
    output_b = np.full(8, -10000.0, np.float32)
    reverse_action = [2, 1, 4, 3]
    for d in range(4):
        action = d + 1
        output_b[action] = 0.0
        output_w[action, d] = 4096.0
        output_w[action, 4 + d] = 4096.0
        output_w[action, 8 + reverse_action[d]] = -0.05
    weight("output_w", output_w)
    weight("output_b", output_b)
    node("logits", "dense", ["all_scores"], w="output_w", b="output_b")

    return {"graph": graph, "memory_size": 216, "params": params}
