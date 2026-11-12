import numpy as np


def build():
    P = {}
    graph = []
    serial = 0

    def param(name, value):
        P[name] = np.asarray(value, dtype=np.float32)
        return name

    def emit(op, inputs, target=None, ident=None, **fields):
        nonlocal serial
        if target is None:
            target = graph
        if ident is None:
            ident = "n%d" % serial
            serial += 1
        node = {"id": ident, "op": op, "in": list(inputs)}
        node.update(fields)
        target.append(node)
        return ident

    def conv(x, w, b=None, target=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return emit("conv", [x], target=target, **fields)

    def dense(x, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return emit("dense", [x], **fields)

    def channel(x, i, target=None):
        return emit(
            "slice", [x], target=target, start=i, stop=i + 1
        )

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    B = 121

    # Player, blue, red, purple, exit, and generic node centres.
    wc = np.zeros((6, 16, 3, 3), np.float32)
    bc = np.array([-7, -7, -7, -7, -8, -8], np.float32)
    for out, colour in enumerate([9, 8, 12, 13, 14]):
        wc[out, colour, :, :] = 1
    for colour in [0, 4, 8, 9, 11, 12, 13, 14, 15]:
        wc[5, colour, :, :] = 1
    param("centres_w", wc)
    param("centres_b", bc)

    wm = np.zeros((4, 1, 3, 3), np.float32)
    for d, (dy, dx) in enumerate(directions):
        wm[d, 0, 1 + dy, 1 + dx] = 1
    param("marker_dir_w", wm)

    marker_merge = np.zeros((1, 16, 1, 1), np.float32)
    marker_merge[0, 11, 0, 0] = 1
    marker_merge[0, 15, 0, 0] = 1
    param("marker_merge_w", marker_merge)

    wg = np.zeros((4, 1, 7, 7), np.float32)
    wgd = np.zeros((4, 4, 7, 7), np.float32)
    wmove = np.zeros((4, 4, 7, 7), np.float32)
    for d, (dy, dx) in enumerate(directions):
        wg[d, 0, 3 + 3 * dy, 3 + 3 * dx] = 1
        wgd[d, d, 3 + 3 * dy, 3 + 3 * dx] = 1
        wmove[d, d, 3 - 3 * dy, 3 - 3 * dx] = 1
    param("gather3_w", wg)
    param("gather3_diag_w", wgd)
    param("move3_w", wmove)

    reverse = np.zeros((4, 4, 1, 1), np.float32)
    for d, opposite in enumerate([1, 0, 3, 2]):
        reverse[d, opposite, 0, 0] = 1
    param("reverse_w", reverse)
    param("sum_dirs_w", np.ones((1, 4, 1, 1), np.float32))
    param("repeat4_w", np.ones((4, 1, 1, 1), np.float32))

    param("two_w", [[2]])
    param("seed_w", [[256]])
    param("hazard_w", [[512]])
    param("missing_w", [[-1]])
    param("missing_b", [1])
    param("discount_w", [[-128]])
    param("discount_b", [128])
    param("inactive_w", [[-1]])
    param("inactive_b", [1])

    # Direction codes 1..4, with zero meaning empty.
    param(
        "encode_w",
        np.arange(1, 5, dtype=np.float32).reshape(1, 4, 1, 1)
    )
    param("decode_w", np.ones((4, 1, 1, 1), np.float32))
    param("decode_b", -np.arange(1, 5, dtype=np.float32))
    param("bin_sum_w", np.ones((1, B, 1, 1), np.float32))

    # Six-pixel bins uniquely index nodes on any six-pixel lattice.
    # The half-pixel margins make masks exactly binary on pixel centres.
    sides_w = np.zeros((44, 3, 1, 1), np.float32)
    sides_b = np.zeros(44, np.float32)
    for axis in range(2):
        for k in range(11):
            base = axis * 22 + 2 * k
            low, high = 6 * k, 6 * k + 5
            sides_w[base, axis, 0, 0] = 252
            sides_b[base] = 4 * (0.5 - low)
            sides_w[base + 1, axis, 0, 0] = -252
            sides_b[base + 1] = 4 * (high + 0.5)
    param("bin_sides_w", sides_w)
    param("bin_sides_b", sides_b)

    bin_w = np.zeros((B, 44, 1, 1), np.float32)
    for row in range(11):
        for col in range(11):
            out = row * 11 + col
            for inp in [2 * row, 2 * row + 1,
                        22 + 2 * col, 23 + 2 * col]:
                bin_w[out, inp, 0, 0] = 1
    param("bin_w", bin_w)
    param("bin_b", np.full(B, -3, np.float32))
    param("retain_w", -np.eye(B, dtype=np.float32))
    param("retain_b", np.ones(B, np.float32))

    reset_w = np.zeros((1, 8), np.float32)
    reset_w[0, 0] = -1
    param("reset_keep_w", reset_w)
    param("reset_keep_b", [1])

    param("pixel_coords_w", 63 * np.eye(2, dtype=np.float32))
    delta_w = np.array([
        [1, 0, -1, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
        [0, -1, 0, 1],
    ], np.float32)
    param("movement_delta_w", delta_w)
    param("movement_sum_w", np.full((1, 4), 1 / 6, np.float32))

    one_sp = emit(
        "slice", ["coords"], ident="click", start=2, stop=3
    )
    one = emit("gmax", [one_sp])
    two = dense(one, "two_w")
    seed_value = dense(one, "seed_w")
    hazard_value = dense(one, "hazard_w")

    centres = emit(
        "relu", [conv("x0", "centres_w", "centres_b")]
    )
    player, blue, red, purple, goal, nodes = [
        channel(centres, i) for i in range(6)
    ]
    bridge = channel("x0", 2)
    edges = conv(bridge, "gather3_w")

    marker = conv("x0", "marker_merge_w")
    marker_dirs = conv(marker, "marker_dir_w")
    active_marker = channel("x0", 11)
    active_dirs = conv(active_marker, "marker_dir_w")
    visible_blue_dirs = emit("mul", [blue, marker_dirs])
    visible_red_dirs = emit("mul", [red, marker_dirs])
    purple_dirs = emit("mul", [purple, active_dirs])

    sides = conv("coords", "bin_sides_w", "bin_sides_b")
    sides = emit("relu", [sides])
    sides = emit("min2", [sides, one_sp])
    bins = emit("relu", [conv(sides, "bin_w", "bin_b")])

    keep_memory = dense(
        "aprev", "reset_keep_w", "reset_keep_b"
    )
    blue_memory = emit("vslice", ["mem"], start=0, stop=B)
    red_memory = emit("vslice", ["mem"], start=B, stop=2 * B)
    old_position = emit(
        "vslice", ["mem"], start=2 * B, stop=2 * B + 2
    )
    blue_memory = emit("vmul", [blue_memory, keep_memory])
    red_memory = emit("vmul", [red_memory, keep_memory])

    # Remember the player's location to distinguish valid moves
    # from invalid actions, which do not advance patrols.
    xy = emit("slice", ["coords"], start=0, stop=2)
    player_xy = emit("mul", [xy, player])
    position = dense(emit("gmax", [player_xy]), "pixel_coords_w")
    position_pair = emit("vconcat", [position, old_position])
    delta = emit(
        "vrelu", [dense(position_pair, "movement_delta_w")]
    )
    moved_vector = dense(delta, "movement_sum_w")
    moved_plane = emit("min2", [one_sp, moved_vector])
    stayed_plane = emit("sub", [one_sp, moved_plane])
    not_player = emit("sub", [one_sp, player])

    def decode(code_map):
        distance = emit("abs", [
            conv(code_map, "decode_w", "decode_b")
        ])
        matches = emit("sub", [one_sp, distance])
        matches = emit("relu", [matches])
        return emit("mul", [matches, nodes])

    def move_six(x):
        return conv(conv(x, "move3_w"), "move3_w")

    def advance_red(x):
        moved = move_six(x)
        continuing = emit("mul", [moved, edges])
        turning = emit("sub", [moved, continuing])
        turned = conv(turning, "reverse_w")
        return emit("add", [continuing, turned]), moved

    # Clear cached blue enemies when the player reaches their tile.
    player_bins = emit("mul", [bins, player])
    occupied_bin = emit("gmax", [player_bins])
    retention = dense(occupied_bin, "retain_w", "retain_b")
    retained_blue = emit("vmul", [blue_memory, retention])
    old_blue_bins = emit("mul", [bins, retained_blue])
    live_blue_code = conv(visible_blue_dirs, "encode_w")
    live_blue_bins = emit("mul", [bins, live_blue_code])
    merged_blue_bins = emit(
        "max2", [old_blue_bins, live_blue_bins]
    )
    next_blue_memory = emit("gmax", [merged_blue_bins])
    blue_code_map = conv(merged_blue_bins, "bin_sum_w")
    blue_dirs = decode(blue_code_map)

    # Red memory stores the previous observation's patrol state.
    # Intercepted patrols are erased before advancing it.
    old_red_bins = emit("mul", [bins, red_memory])
    old_red_code = conv(old_red_bins, "bin_sum_w")
    old_red_dirs = decode(old_red_code)
    old_red_dirs = emit("mul", [old_red_dirs, not_player])
    advanced_red, _ = advance_red(old_red_dirs)
    moved_red = emit("mul", [advanced_red, moved_plane])
    stationary_red = emit("mul", [old_red_dirs, stayed_plane])
    inferred_red = emit("add", [moved_red, stationary_red])
    red_dirs = emit("max2", [visible_red_dirs, inferred_red])
    red_code = conv(red_dirs, "encode_w")
    red_bins = emit("mul", [bins, red_code])
    next_red_memory = emit("gmax", [red_bins])

    emit(
        "vconcat",
        [next_blue_memory, next_red_memory, position],
        ident="mem_next"
    )

    purple_active_map = conv(purple_dirs, "sum_dirs_w")
    purple_active = emit("gmax", [purple_active_map])
    purple_inactive = dense(
        purple_active, "inactive_w", "inactive_b"
    )

    blue_front = conv(move_six(blue_dirs), "sum_dirs_w")
    blue_front = emit("min2", [blue_front, one_sp])
    not_blue_front = emit("sub", [one_sp, blue_front])
    safe_nodes = emit("mul", [nodes, not_blue_front])
    path = emit("add", [nodes, bridge])
    safe_path = emit("mul", [path, not_blue_front])

    goal_seed = emit("mul", [goal, safe_nodes])
    goal_seed = emit("mul", [goal_seed, seed_value])
    blue_map = conv(blue_dirs, "sum_dirs_w")
    blue_seed = emit("mul", [blue_map, safe_nodes])
    blue_seed = emit("mul", [blue_seed, seed_value])

    # Static graph-distance field.
    body = []
    shifted = conv("goal_state", "gather3_w", target=body)
    cs = [channel(shifted, d, target=body) for d in range(4)]
    ud = emit("max2", cs[:2], target=body)
    lr = emit("max2", cs[2:], target=body)
    best = emit("max2", [ud, lr], target=body)
    cost = emit("sub", [best, one], target=body)
    positive = emit("relu", [cost], target=body)
    seeded = emit("max2", [positive, goal_seed], target=body)
    nxt = emit("mul", [seeded, safe_path], target=body)
    graph.append({
        "id": "GoalWalk",
        "op": "loop",
        "T": 256,
        "state": [{"id": "goal_state", "init": goal_seed}],
        "body": body,
        "next": {"goal_state": nxt},
    })
    goal_field = "GoalWalk.goal_state"

    goal_at_player = emit(
        "gmax", [emit("mul", [goal_field, player])]
    )
    missing = emit(
        "vrelu",
        [dense(goal_at_player, "missing_w", "missing_b")]
    )
    blue_discount = dense(missing, "discount_w", "discount_b")
    discounted_blue_seed = emit("sub", [blue_seed, blue_discount])
    discounted_blue_seed = emit("relu", [discounted_blue_seed])
    stopping_seed = emit(
        "max2", [goal_seed, discounted_blue_seed]
    )

    purple_next = conv(move_six(purple_dirs), "sum_dirs_w")
    horizon = 26
    red_hazards = []
    red_state = red_dirs
    for _ in range(horizon):
        red_state, moved = advance_red(red_state)
        red_hazards.append(conv(moved, "sum_dirs_w"))

    immediate_hazard = emit(
        "add", [red_hazards[0], purple_next]
    )

    def action_values(future, hazard):
        penalty = emit("mul", [hazard, hazard_value])
        masked = emit("sub", [future, penalty])
        masked = emit("relu", [masked])
        near = conv(masked, "gather3_diag_w")
        far = conv(near, "gather3_diag_w")
        return emit("mul", [far, edges])

    # Four value channels conditioned on incoming direction.
    value = conv(stopping_seed, "repeat4_w")
    for t in range(horizon - 1, 0, -1):
        q = action_values(value, red_hazards[t])
        u, d, l, r = [channel(q, k) for k in range(4)]
        ud = emit("max2", [u, d])
        lr = emit("max2", [l, r])
        alternatives = emit("concat", [lr, lr, ud, ud])
        no_reverse = emit("max2", [q, alternatives])
        reverse_q = conv(q, "reverse_w")
        allowed_reverse = emit(
            "mul", [reverse_q, purple_inactive]
        )
        best = emit("max2", [no_reverse, allowed_reverse])
        charged = emit("sub", [best, two])
        charged = emit("relu", [charged])
        seeded = emit("max2", [charged, stopping_seed])
        value = emit("mul", [seeded, safe_nodes])

    first_q = action_values(value, immediate_hazard)
    q_vector = emit(
        "gmax", [emit("mul", [first_q, player])]
    )

    hazard_binary = emit("min2", [immediate_hazard, one_sp])
    immediate_free = emit("sub", [one_sp, hazard_binary])
    immediate_safe_nodes = emit("mul", [safe_nodes, immediate_free])
    safe_near = conv(immediate_safe_nodes, "gather3_w")
    safe_far = conv(safe_near, "gather3_diag_w")
    safe_moves = emit("mul", [safe_far, edges])
    safe_vector = emit(
        "gmax", [emit("mul", [safe_moves, player])]
    )

    static_near = conv(goal_field, "gather3_w")
    static_far = conv(static_near, "gather3_diag_w")
    static_safe = emit("mul", [static_far, safe_moves])
    static_vector = emit(
        "gmax", [emit("mul", [static_safe, player])]
    )

    head_input = emit(
        "vconcat", [q_vector, safe_vector, static_vector]
    )
    head_w = np.zeros((8, 12), np.float32)
    head_b = np.full(8, -1000, np.float32)
    head_b[0] = 0.5
    for d in range(4):
        head_w[d + 1, d] = 1
        head_w[d + 1, d + 4] = 1
        head_w[d + 1, d + 8] = 0.001
        head_b[d + 1] = (d + 1) * 0.0001
    param("head_w", head_w)
    param("head_b", head_b)
    emit(
        "dense", [head_input], ident="logits",
        w="head_w", b="head_b"
    )

    def node_count(ns):
        return sum(
            1 + (
                node_count(n["body"]) if n["op"] == "loop" else 0
            )
            for n in ns
        )

    assert node_count(graph) <= 1000
    assert sum(a.size for a in P.values()) <= 150000

    return {
        "graph": graph,
        "memory_size": 244,
        "params": P,
    }
