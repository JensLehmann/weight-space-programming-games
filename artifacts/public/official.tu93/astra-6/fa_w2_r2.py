import numpy as np


def build():
    params = {}
    graph = []
    target = graph
    counter = 0

    def parameter(name, value):
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, *inputs, ident=None, **fields):
        nonlocal counter
        if ident is None:
            ident = "n%d" % counter
            counter += 1
        nd = {"id": ident, "op": op, "in": list(inputs)}
        nd.update(fields)
        target.append(nd)
        return ident

    def conv(x, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return node("conv", x, **fields)

    def sl(x, start, stop):
        return node("slice", x, start=start, stop=stop)

    # Direction channels: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    read_w = np.zeros((4, 4, 7, 7), np.float32)
    move_w = np.zeros_like(read_w)
    read_scalar_w = np.zeros((4, 1, 7, 7), np.float32)
    for a, (dy, dx) in enumerate(directions):
        read_w[a, a, 3 + 3 * dy, 3 + 3 * dx] = 1
        move_w[a, a, 3 - 3 * dy, 3 - 3 * dx] = 1
        read_scalar_w[a, 0, 3 + 3 * dy, 3 + 3 * dx] = 1

    parameter("read3", read_w)
    parameter("move3", move_w)
    parameter("read_scalar3", read_scalar_w)
    parameter("step_bias", -np.ones(4, np.float32))
    parameter("sum4", np.ones((1, 4, 1, 1), np.float32))
    parameter("repeat4", np.ones((4, 1, 1, 1), np.float32))

    flip_w = np.zeros((4, 4, 1, 1), np.float32)
    orth_a_w = np.zeros_like(flip_w)
    orth_b_w = np.zeros_like(flip_w)
    for a in range(4):
        flip_w[a, opposite[a], 0, 0] = 1
        orth_a_w[a, [2, 2, 0, 0][a], 0, 0] = 1
        orth_b_w[a, [3, 3, 1, 1][a], 0, 0] = 1
    parameter("flip", flip_w)
    parameter("orth_a", orth_a_w)
    parameter("orth_b", orth_b_w)

    def move6(x):
        return conv(conv(x, "move3"), "move3")

    def read6(x, charge=False):
        return conv(
            conv(x, "read3"),
            "read3",
            "step_bias" if charge else None,
        )

    def sum4(x):
        return conv(x, "sum4")

    one = sl("coords", 2, 3)
    zero = node("sub", one, one)

    # Detect 3x3 token and unoccupied-node centres by their four corners.
    center_colors = [9, 14, 8, 12, 13, 0]
    center_w = np.zeros((6, 16, 3, 3), np.float32)
    for out, color in enumerate(center_colors):
        for y in (0, 2):
            for x in (0, 2):
                center_w[out, color, y, x] = 1
    parameter("centers_w", center_w)
    parameter("centers_b", -3 * np.ones(6, np.float32))
    centers = node("relu", conv("x0", "centers_w", "centers_b"))
    player, goal, visible_red, blue, purple = [
        sl(centers, i, i + 1) for i in range(5)
    ]
    parameter("node_centers_w", np.ones((1, 6, 1, 1), np.float32))
    node_centers = conv(centers, "node_centers_w")

    # Four track connectors around every potential node.
    edge_w = np.zeros((4, 16, 7, 7), np.float32)
    for a, (dy, dx) in enumerate(directions):
        edge_w[a, 2, 3 + 3 * dy, 3 + 3 * dx] = 1
    parameter("edges_w", edge_w)
    edges = conv("x0", "edges_w")

    marker_w = np.zeros((4, 16, 3, 3), np.float32)
    for a, (dy, dx) in enumerate(directions):
        marker_w[a, 15, 1 + dy, 1 + dx] = 1
        marker_w[a, 11, 1 + dy, 1 + dx] = 1
    parameter("marker_w", marker_w)
    markers = conv("x0", "marker_w")
    visible_red_dirs = node("mul", markers, visible_red)
    blue_dirs = node("mul", markers, blue)

    active_w = np.zeros((1, 16, 3, 3), np.float32)
    for dy, dx in directions:
        active_w[0, 11, 1 + dy, 1 + dx] = 1
    parameter("active_w", active_w)
    purple_active = node("mul", conv("x0", "active_w"), purple)
    visible_idle = node("sub", purple, purple_active)
    active_dirs = node("mul", markers, purple_active)
    visible_idle_dirs = node("mul", markers, visible_idle)

    # Fixed 6x6 spatial bins. Because graph nodes are six pixels apart,
    # each bin contains at most one node on the current lattice.
    memory_size = 121
    row_w = np.zeros((memory_size, 3, 1, 1), np.float32)
    col_w = np.zeros_like(row_w)
    row_b = np.zeros(memory_size, np.float32)
    col_b = np.zeros(memory_size, np.float32)
    for r in range(11):
        for c in range(11):
            k = 11 * r + c
            row_w[k, 0, 0, 0] = 63
            col_w[k, 1, 0, 0] = 63
            row_b[k] = -(6 * r + 2.5)
            col_b[k] = -(6 * c + 2.5)

    parameter("bin_rows_w", row_w)
    parameter("bin_cols_w", col_w)
    parameter("bin_rows_b", row_b)
    parameter("bin_cols_b", col_b)

    # Shared scalar broadcasts avoid a large diagonal convolution.
    parameter("minus_ten_w", [[-10]])
    parameter("thirty_b", [30])
    pooled_one = node("gmax", one)
    minus_ten = node("dense", pooled_one, w="minus_ten_w")
    parameter("constant_zero_w", [[0]])
    thirty = node(
        "dense", pooled_one, w="constant_zero_w", b="thirty_b"
    )

    row_dist = node(
        "abs", conv("coords", "bin_rows_w", "bin_rows_b")
    )
    col_dist = node(
        "abs", conv("coords", "bin_cols_w", "bin_cols_b")
    )
    row_inside = node(
        "relu", node("add", node("mul", row_dist, minus_ten), thirty)
    )
    col_inside = node(
        "relu", node("add", node("mul", col_dist, minus_ten), thirty)
    )
    row_inside = node("min2", row_inside, one)
    col_inside = node("min2", col_inside, one)
    bins = node("min2", row_inside, col_inside)

    # Codes 1..4: stationary red orientations.
    # Codes 5..8: inactive purple orientations.
    parameter(
        "red_code_w",
        np.arange(1, 5, dtype=np.float32).reshape(1, 4, 1, 1),
    )
    parameter(
        "idle_code_w",
        np.arange(5, 9, dtype=np.float32).reshape(1, 4, 1, 1),
    )
    visible_code = node(
        "add",
        conv(visible_red_dirs, "red_code_w"),
        conv(visible_idle_dirs, "idle_code_w"),
    )

    remembered_bins = node("mul", bins, "mem")
    parameter(
        "memory_sum_w", np.ones((1, memory_size, 1, 1), np.float32)
    )
    remembered_code = conv(remembered_bins, "memory_sum_w")
    remembered_code = node("mul", remembered_code, node_centers)
    stationary_code = node("max2", remembered_code, visible_code)

    # Player occupancy means capture. An activated purple is no longer
    # a stationary enemy, even though it has not moved away yet.
    clear_locations = node("max2", player, purple_active)
    retain = node("sub", one, clear_locations)
    stationary_code = node("mul", stationary_code, retain)
    encoded_memory = node("mul", stationary_code, bins)
    node("gmax", encoded_memory, ident="mem_next")

    parameter("decode_w", np.ones((8, 1, 1, 1), np.float32))
    parameter("decode_b", -np.arange(1, 9, dtype=np.float32))
    code_distance = node(
        "abs", conv(stationary_code, "decode_w", "decode_b")
    )
    decoded = node("relu", node("sub", one, code_distance))
    red_dirs = sl(decoded, 0, 4)
    idle_dirs = sl(decoded, 4, 8)
    red = sum4(red_dirs)
    purple_idle = sum4(idle_dirs)

    red_front = sum4(move6(red_dirs))
    idle_front_dirs = move6(idle_dirs)
    idle_front = sum4(idle_front_dirs)
    idle_trigger = sum4(move6(idle_front_dirs))
    active_next = sum4(move6(active_dirs))
    static_danger = node("add", red_front, idle_front)

    # Active purple tokens follow with a two-move delay.
    active_flag = node("gmax", purple_active)
    parameter("turn_penalty_w", [[2000]])
    turn_penalty = node("dense", active_flag, w="turn_penalty_w")

    subgoal = node("max2", red, purple_idle)
    subgoal = node("max2", subgoal, idle_trigger)
    parameter(
        "subgoal_w",
        np.asarray([500], np.float32).reshape(1, 1, 1, 1),
    )
    parameter(
        "goal_w",
        np.asarray([1000], np.float32).reshape(1, 1, 1, 1),
    )
    seed = node(
        "max2", conv(subgoal, "subgoal_w"), conv(goal, "goal_w")
    )
    seed4 = conv(seed, "repeat4")

    def action_values(value, safety):
        masked = node("mul", value, safety)
        q = node("relu", read6(masked, charge=True))
        return node("mul", q, edges)

    def reduce_turns(q):
        perpendicular = node(
            "max2", conv(q, "orth_a"), conv(q, "orth_b")
        )
        non_reversing = node("max2", q, perpendicular)
        reversing = node("sub", conv(q, "flip"), turn_penalty)
        return node("max2", non_reversing, reversing)

    def bellman(value, safety):
        return node(
            "max2", reduce_turns(action_values(value, safety)), seed
        )

    horizon = 96

    # Forward prediction, with newest safety map first in a shift register.
    forward_body = []
    target = forward_body

    arrived = move6("patrol_state")
    continuing = node("mul", arrived, edges)
    reversing = node("sub", arrived, continuing)
    patrol_next = node("add", continuing, conv(reversing, "flip"))

    blue_occupancy = sum4(arrived)
    danger = node("add", static_danger, blue_occupancy)
    safety = node("relu", node("sub", one, danger))
    old_history = sl("prediction_history", 0, horizon - 1)
    history_next = node("concat", safety, old_history)

    target = graph
    graph.append({
        "id": "predict",
        "op": "loop",
        "T": horizon,
        "state": [
            {"id": "patrol_state", "init": blue_dirs},
            {
                "id": "prediction_history",
                "zeros": [horizon, 64, 64],
            },
        ],
        "body": forward_body,
        "next": {
            "patrol_state": patrol_next,
            "prediction_history": history_next,
        },
    })

    # Backward finite-horizon value propagation.
    backward_body = []
    target = backward_body

    current_safety = sl("backup_history", 0, 1)
    value_next = bellman("backup_value", current_safety)
    remaining_history = sl("backup_history", 1, horizon)
    backup_history_next = node("concat", remaining_history, zero)

    target = graph
    graph.append({
        "id": "backup",
        "op": "loop",
        "T": horizon - 1,
        "state": [
            {"id": "backup_value", "init": seed4},
            {
                "id": "backup_history",
                "init": "predict.prediction_history",
            },
        ],
        "body": backward_body,
        "next": {
            "backup_value": value_next,
            "backup_history": backup_history_next,
        },
    })

    first_safety = sl("backup.backup_history", 0, 1)
    root_safe = node("relu", node("sub", first_safety, active_next))
    root_q = action_values("backup.backup_value", root_safe)
    q_at_player = node("gmax", node("mul", root_q, player))

    safe_actions = conv(root_safe, "read_scalar3")
    safe_actions = conv(safe_actions, "read3")
    safe_actions = node("mul", safe_actions, edges)
    safe_at_player = node("gmax", node("mul", safe_actions, player))
    edges_at_player = node("gmax", node("mul", edges, player))

    readout = node(
        "vconcat", q_at_player, safe_at_player, edges_at_player
    )
    output_w = np.zeros((8, 12), np.float32)
    output_b = np.full(8, -1000000, np.float32)
    for a in range(4):
        output_w[a + 1, a] = 1
        output_w[a + 1, 4 + a] = 20
        output_w[a + 1, 8 + a] = 1
        output_b[a + 1] = 0
    parameter("output_w", output_w)
    parameter("output_b", output_b)
    node(
        "dense", readout, ident="logits",
        w="output_w", b="output_b"
    )

    parameter("click_w", np.zeros((1, 1, 1, 1), np.float32))
    node("conv", goal, ident="click", w="click_w")

    return {
        "graph": graph,
        "memory_size": memory_size,
        "params": params,
    }
