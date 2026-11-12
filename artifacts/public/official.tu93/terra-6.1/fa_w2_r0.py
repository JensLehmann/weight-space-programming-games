import numpy as np


def build():
    P = {}
    G = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def param(name, value):
        P[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(name, op, inputs, **fields):
        item = {"id": name, "op": op, "in": inputs}
        item.update(fields)
        G.append(item)
        return name

    def conv(name, inp, weight, bias=None):
        fields = {"w": weight}
        if bias is not None:
            fields["b"] = bias
        return node(name, "conv", [inp], **fields)

    def sl(name, inp, start, stop):
        return node(name, "slice", [inp], start=start, stop=stop)

    def vsl(name, inp, start, stop):
        return node(name, "vslice", [inp], start=start, stop=stop)

    param("two", [[[[2.0]]]])
    param("goal_scale", [[[[128.0]]]])
    param("danger_scale", [[[[180.0]]]])
    param("deadend_scale", [[[[24.0]]]])
    param("identity", [[[[1.0]]]])
    param("score_bias", [240.0])

    # Centre detectors for track nodes, the player, and the exit.
    w = np.zeros((1, 16, 3, 3), np.float32)
    for c in (0, 4, 8, 9, 11, 12, 13, 14, 15):
        w[0, c, :, :] = 1.0
    param("node_w", w)
    param("node_b", [-8.5])

    w = np.zeros((1, 16, 3, 3), np.float32)
    w[0, 9, :, :] = 1.0
    param("player_w", w)
    param("player_b", [-7.5])

    w = np.zeros((1, 16, 3, 3), np.float32)
    w[0, 14, :, :] = 1.0
    param("exit_w", w)
    param("exit_b", [-8.5])

    # Directional gathers, three pixels per application.
    w = np.zeros((4, 1, 7, 7), np.float32)
    for i, (dy, dx) in enumerate(dirs):
        w[i, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
    param("gather", w)

    w = np.zeros((4, 4, 7, 7), np.float32)
    for i, (dy, dx) in enumerate(dirs):
        w[i, i, 3 + 3 * dy, 3 + 3 * dx] = 1.0
    param("gather_diag", w)

    # Enemy channels: direction for each of blue, pink, and
    # activated purple.
    w = np.zeros((12, 16, 3, 3), np.float32)
    for kind, body in enumerate((8, 12, 13)):
        for d, (dy, dx) in enumerate(dirs):
            out = 4 * kind + d
            w[out, body, :, :] = 1.0
            markers = (
                (15, 11) if kind == 0
                else (15,) if kind == 1
                else (11,)
            )
            for marker in markers:
                w[out, marker, 1 + dy, 1 + dx] = 1.0
    param("enemy_w", w)
    param("enemy_b", np.full(12, -8.5, np.float32))

    # Advance each directional enemy channel by three pixels.
    w = np.zeros((12, 12, 7, 7), np.float32)
    for kind in range(3):
        for d, (dy, dx) in enumerate(dirs):
            c = 4 * kind + d
            w[c, c, 3 - 3 * dy, 3 - 3 * dx] = 1.0
    param("enemy_forward", w)

    w = np.zeros((3, 12, 1, 1), np.float32)
    for kind in range(3):
        w[kind, 4 * kind:4 * kind + 4, 0, 0] = 2.0
    param("enemy_sum", w)

    # Pink enemies seed vertical and horizontal patrol segments.
    w = np.zeros((2, 12, 1, 1), np.float32)
    w[0, 4:6, 0, 0] = 2.0
    w[1, 6:8, 0, 0] = 2.0
    param("patrol_seed_w", w)

    w = np.zeros((4, 2, 7, 7), np.float32)
    for d, (dy, dx) in enumerate(dirs):
        axis = 0 if d < 2 else 1
        w[d, axis, 3 + 3 * dy, 3 + 3 * dx] = 1.0
    param("patrol_gather", w)

    # Cost excluding the separately weighted pink patrol lane.
    w = np.zeros((1, 2, 1, 1), np.float32)
    w[0, :, 0, 0] = [3.0, 0.35]
    param("base_cost_w", w)
    param("base_cost_b", [1.0])

    # Match a normalized remembered coordinate to a cell.
    w = np.zeros((1, 3, 1, 1), np.float32)
    w[0, 0, 0, 0] = -63.0
    w[0, 1, 0, 0] = -63.0
    param("coord_w", w)
    param("coord_b", [0.5])

    param("one_minus_w", [[-1.0]])
    param("one_minus_b", [1.0])
    param("negative_w", [[-1.0]])
    param("minus_one_w", [[1.0]])
    param("minus_one_b", [-1.0])

    w = np.full((1, 4, 1, 1), -1.0, np.float32)
    param("deadend_w", w)
    param("deadend_b", [1.5])

    # Maps of nodes, player, exit, and edges.
    x = conv("node_raw", "x0", "node_w", "node_b")
    x = node("node_positive", "relu", [x])
    node_map = conv("node_map", x, "two")

    x = conv("player_raw", "x0", "player_w", "player_b")
    x = node("player_positive", "relu", [x])
    player_map = conv("player_map", x, "two")

    x = conv("exit_raw", "x0", "exit_w", "exit_b")
    x = node("exit_positive", "relu", [x])
    exit_map = conv("exit_map", x, "two")
    edge_map = sl("edge_map", "x0", 2, 3)

    player_coords = node("player_coords", "mul", ["coords", player_map])
    player_pos = node("player_pos", "gmax", [player_coords])
    exit_coords = node("exit_coords", "mul", ["coords", exit_map])
    exit_pos = node("exit_pos", "gmax", [exit_coords])

    # Remember the exit when it is temporarily hidden by another sprite.
    old_exit = vsl("old_exit", "mem", 0, 3)
    exit_valid = vsl("exit_valid", exit_pos, 2, 3)
    exit_hidden = node(
        "exit_hidden", "dense", [exit_valid],
        w="one_minus_w", b="one_minus_b"
    )
    hidden_exit = node("hidden_exit", "vmul", [old_exit, exit_hidden])
    remembered_exit = node("remembered_exit", "vadd", [exit_pos, hidden_exit])

    # Memory: exit coordinate, ten recent player coordinates, reset flag.
    history = vsl("old_history", "mem", 3, 30)
    old_reset_flag = vsl("old_reset_flag", "mem", 33, 34)
    previous_reset = vsl("previous_reset", "aprev", 0, 1)
    reset_flag = node("reset_flag", "vadd", [old_reset_flag, previous_reset])
    node(
        "mem_next", "vconcat",
        [remembered_exit, player_pos, history, reset_flag]
    )

    delta = node("exit_delta", "sub", ["coords", remembered_exit])
    delta = node("exit_abs", "abs", [delta])
    goal = conv("goal_match_raw", delta, "coord_w", "coord_b")
    goal = node("goal_match_positive", "relu", [goal])
    goal_valid = vsl("goal_valid", remembered_exit, 2, 3)
    goal = node("goal_match_valid", "mul", [goal, goal_valid])
    goal_map = conv("goal_map", goal, "two")
    goal_seed = conv("goal_seed", goal_map, "goal_scale")

    enemies = conv("enemy_raw", "x0", "enemy_w", "enemy_b")
    enemies = node("enemy_positive", "relu", [enemies])
    patrol_seed = conv("patrol_seed", enemies, "patrol_seed_w")

    # Count pink patrols and construct a clipped multi-patrol indicator.
    patrol_mean = node("patrol_mean", "gmean", [patrol_seed])
    param("patrol_count_w", [[4096.0, 4096.0]])
    param("patrol_count_b", [-1.0])
    extra = node(
        "extra_patrol_raw", "dense", [patrol_mean],
        w="patrol_count_w", b="patrol_count_b"
    )
    extra = node("extra_patrol", "vrelu", [extra])
    excess = node(
        "extra_above_one_raw", "dense", [extra],
        w="minus_one_w", b="minus_one_b"
    )
    excess = node("extra_above_one", "vrelu", [excess])
    negative_excess = node(
        "negative_excess", "dense", [excess], w="negative_w"
    )
    multi = node("multi_patrol", "vadd", [extra, negative_excess])

    param("patrol_weight_w", [[-17.0]])
    param("patrol_weight_b", [25.0])
    patrol_weight = node(
        "patrol_weight", "dense", [multi],
        w="patrol_weight_w", b="patrol_weight_b"
    )
    param("history_weight_w", [[3.5]])
    param("history_weight_b", [0.5])
    history_weight = node(
        "history_weight", "dense", [multi],
        w="history_weight_w", b="history_weight_b"
    )

    # Expand each pink patrol's straight track segment.
    patrol_body = [
        {"id": "patrol_first", "op": "conv", "in": ["patrol_state"], "w": "patrol_gather"},
        {"id": "patrol_middle", "op": "mul", "in": ["patrol_first", edge_map]},
        {"id": "patrol_second", "op": "conv", "in": ["patrol_middle"], "w": "gather_diag"},
        {"id": "patrol_u", "op": "slice", "in": ["patrol_second"], "start": 0, "stop": 1},
        {"id": "patrol_d", "op": "slice", "in": ["patrol_second"], "start": 1, "stop": 2},
        {"id": "patrol_l", "op": "slice", "in": ["patrol_second"], "start": 2, "stop": 3},
        {"id": "patrol_r", "op": "slice", "in": ["patrol_second"], "start": 3, "stop": 4},
        {"id": "patrol_vertical", "op": "max2", "in": ["patrol_u", "patrol_d"]},
        {"id": "patrol_horizontal", "op": "max2", "in": ["patrol_l", "patrol_r"]},
        {"id": "patrol_axes", "op": "concat", "in": ["patrol_vertical", "patrol_horizontal"]},
        {"id": "patrol_nodes", "op": "mul", "in": ["patrol_axes", node_map]},
        {"id": "patrol_next", "op": "max2", "in": ["patrol_state", "patrol_nodes"]},
    ]
    G.append({
        "id": "patrol",
        "op": "loop",
        "T": 32,
        "state": [{"id": "patrol_state", "init": patrol_seed}],
        "body": patrol_body,
        "next": {"patrol_state": "patrol_next"},
    })
    pv = sl("patrol_vertical_final", "patrol.patrol_state", 0, 1)
    ph = sl("patrol_horizontal_final", "patrol.patrol_state", 1, 2)
    patrol_lane = node("patrol_lane", "max2", [pv, ph])

    # Immediate enemy destinations.
    advanced = conv("enemy_advance_half", enemies, "enemy_forward")
    advanced = conv("enemy_advance_full", advanced, "enemy_forward")
    kinds = conv("enemy_kinds", advanced, "enemy_sum")
    blue = sl("blue_threat", kinds, 0, 1)
    pink = sl("pink_threat", kinds, 1, 2)
    purple = sl("purple_threat", kinds, 2, 3)
    danger = node("danger_bp", "max2", [blue, pink])
    danger = node("danger_all", "max2", [danger, purple])

    base_inputs = node("base_inputs", "concat", [blue, purple])
    base_cost = conv("base_cost", base_inputs, "base_cost_w", "base_cost_b")
    patrol_cost = node("patrol_cost", "mul", [patrol_lane, patrol_weight])
    path_cost = node("path_cost", "add", [base_cost, patrol_cost])

    # Build a value-propagation loop. A track move is six pixels:
    # three to an edge midpoint, then three to the next node.
    def add_walk(loop_id, prefix, cost):
        def s(suffix):
            return prefix + suffix

        body = [
            {"id": s("taxed"), "op": "sub", "in": [s("state"), cost]},
            {"id": s("positive"), "op": "relu", "in": [s("taxed")]},
            {"id": s("sources"), "op": "mul", "in": [s("positive"), node_map]},
            {"id": s("first"), "op": "conv", "in": [s("sources")], "w": "gather"},
        ]
        for i in range(4):
            body.append({
                "id": s("f%d" % i), "op": "slice", "in": [s("first")],
                "start": i, "stop": i + 1
            })
        body.extend([
            {"id": s("f01"), "op": "max2", "in": [s("f0"), s("f1")]},
            {"id": s("f23"), "op": "max2", "in": [s("f2"), s("f3")]},
            {"id": s("fmax"), "op": "max2", "in": [s("f01"), s("f23")]},
            {"id": s("middle"), "op": "mul", "in": [s("fmax"), edge_map]},
            {"id": s("second"), "op": "conv", "in": [s("middle")], "w": "gather"},
        ])
        for i in range(4):
            body.append({
                "id": s("q%d" % i), "op": "slice", "in": [s("second")],
                "start": i, "stop": i + 1
            })
        body.extend([
            {"id": s("q01"), "op": "max2", "in": [s("q0"), s("q1")]},
            {"id": s("q23"), "op": "max2", "in": [s("q2"), s("q3")]},
            {"id": s("qmax"), "op": "max2", "in": [s("q01"), s("q23")]},
            {"id": s("nodes"), "op": "mul", "in": [s("qmax"), node_map]},
            {"id": s("next"), "op": "max2", "in": [s("nodes"), goal_seed]},
        ])
        G.append({
            "id": loop_id,
            "op": "loop",
            "T": 128,
            "state": [{"id": s("state"), "init": goal_seed}],
            "body": body,
            "next": {s("state"): s("next")},
        })
        return loop_id + "." + s("state")

    weighted_value = add_walk("weighted_walk", "ww_", path_cost)

    # Unweighted distance map for the low-step reset decision.
    param("unit_cost_w", [[0.0]])
    param("unit_cost_b", [1.0])
    unit_cost = node(
        "unit_cost", "dense", [goal_valid],
        w="unit_cost_w", b="unit_cost_b"
    )
    distance_value = add_walk("distance_walk", "dw_", unit_cost)

    # Visit penalties.
    visit_maps = []
    for i in range(10):
        pos = vsl("history_pos_%d" % i, "mem", 3 + 3 * i, 6 + 3 * i)
        valid = vsl("history_valid_%d" % i, pos, 2, 3)
        d = node("history_delta_%d" % i, "sub", ["coords", pos])
        d = node("history_abs_%d" % i, "abs", [d])
        match = conv("history_raw_%d" % i, d, "coord_w", "coord_b")
        match = node("history_positive_%d" % i, "relu", [match])
        match = node("history_valid_map_%d" % i, "mul", [match, valid])
        match = node(
            "history_cost_%d" % i, "mul", [match, history_weight]
        )
        visit_maps.append(match)

    visits = visit_maps[0]
    for i in range(1, 10):
        visits = node("history_sum_%d" % i, "add", [visits, visit_maps[i]])

    # Non-goal dead ends are less attractive.
    adjacent = conv("adjacent_edges", edge_map, "gather")
    deadend = conv("deadend_raw", adjacent, "deadend_w", "deadend_b")
    deadend = node("deadend_positive", "relu", [deadend])
    deadend = conv("deadend_full", deadend, "two")
    deadend = node("deadend_nodes", "mul", [deadend, node_map])
    deadend = node("deadend_without_goal", "sub", [deadend, goal_map])
    deadend = node("deadend_final", "relu", [deadend])
    deadend_cost = conv("deadend_cost", deadend, "deadend_scale")

    score = conv("score_base", weighted_value, "identity", "score_bias")
    danger_cost = conv("danger_cost", danger, "danger_scale")
    score = node("score_safe", "sub", [score, danger_cost])
    score = node("score_history", "sub", [score, visits])
    score = node("score_deadend", "sub", [score, deadend_cost])
    score = node("score_nodes", "mul", [score, node_map])

    candidates = conv("candidate_half", score, "gather")
    candidates = conv("candidate_full", candidates, "gather_diag")
    candidate_edges = conv("candidate_edges", edge_map, "gather")
    candidates = node("candidate_legal", "mul", [candidates, candidate_edges])
    candidates = node("candidate_at_player", "mul", [candidates, player_map])
    candidate_values = node("candidate_values", "gmax", [candidates])

    # Request at most one extra reset if the step bar is almost empty
    # and the exit is still at least three track moves away.
    bar = sl("bar_plane", "x0", 6, 7)
    bar_mean = node("bar_mean", "gmean", [bar])
    param("bar_w", [[-4096.0]])
    param("bar_b", [6.5])
    bar_short = node("bar_short_raw", "dense", [bar_mean], w="bar_w", b="bar_b")
    bar_short = node("bar_short", "vrelu", [bar_short])

    distance_at_player = node(
        "distance_at_player", "mul", [distance_value, player_map]
    )
    distance_at_player = node(
        "distance_player_value", "gmax", [distance_at_player]
    )
    param("distance_need_w", [[-1.0]])
    param("distance_need_b", [125.5])
    distance_need = node(
        "distance_need_raw", "dense", [distance_at_player],
        w="distance_need_w", b="distance_need_b"
    )
    distance_need = node("distance_need", "vrelu", [distance_need])

    can_reset = node(
        "can_reset_raw", "dense", [reset_flag],
        w="one_minus_w", b="one_minus_b"
    )
    can_reset = node("can_reset", "vrelu", [can_reset])
    reset_signal = node("reset_bar_distance", "vmul", [bar_short, distance_need])
    reset_signal = node("reset_signal", "vmul", [reset_signal, can_reset])

    action_inputs = node(
        "action_inputs", "vconcat", [candidate_values, reset_signal]
    )
    w = np.zeros((8, 5), np.float32)
    for i in range(4):
        w[i + 1, i] = 1.0
    w[0, 4] = 1000.0
    param("action_w", w)
    param("action_b", [200.0, 0.0, 0.0, 0.0, 0.0, -500.0, -500.0, -500.0])
    node("logits", "dense", [action_inputs], w="action_w", b="action_b")

    # This game does not use click actions.
    sl("click", "x0", 0, 1)

    return {"graph": G, "memory_size": 34, "params": P}
