import numpy as np


def build():
    graph = []
    params = {}

    def node(name, op, inputs=None, **fields):
        nd = {"id": name, "op": op}
        if inputs is not None:
            nd["in"] = inputs
        nd.update(fields)
        graph.append(nd)
        return name

    def param(name, value):
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def conv1(name, source, weights, bias=None):
        w = "p_" + name + "_w"
        param(w, np.asarray(weights, dtype=np.float32)[:, :, None, None])
        fields = {"w": w}
        if bias is not None:
            b = "p_" + name + "_b"
            param(b, bias)
            fields["b"] = b
        return node(name, "conv", [source], **fields)

    def dense(name, source, weights, bias=None):
        w = "p_" + name + "_w"
        param(w, weights)
        fields = {"w": w}
        if bias is not None:
            b = "p_" + name + "_b"
            param(b, bias)
            fields["b"] = b
        return node(name, "dense", [source], **fields)

    def binary(name, op, a, b):
        return node(name, op, [a, b])

    def relu(name, source):
        return node(name, "relu", [source])

    for name, dy, dx in [
        ("p_up", -3, 0),
        ("p_down", 3, 0),
        ("p_left", 0, -3),
        ("p_right", 0, 3),
    ]:
        w = np.zeros((1, 1, 7, 7), dtype=np.float32)
        w[0, 0, 3 + dy, 3 + dx] = 1.0
        param(name, w)

    def shift6(prefix, source, weight, dest=None):
        out = graph if dest is None else dest
        a = prefix + "_first"
        b = prefix + "_second"
        out.append({"id": a, "op": "conv", "in": [source], "w": weight})
        out.append({"id": b, "op": "conv", "in": [a], "w": weight})
        return b

    # Detect the four-pixel vertical stroke common to both player sprites.
    w = np.zeros((1, 16, 5, 5), dtype=np.float32)
    for dy in (-2, -1, 0, 1):
        w[0, 9, 2 + dy, 2] = 1.0
    param("p_player", w)
    param("p_player_bias", [-3.5])
    node("player_raw", "conv", ["x0"], w="p_player", b="p_player_bias")
    relu("player_half", "player_raw")
    conv1("player", "player_half", [[2.0]])

    # Tile-centre coordinate tents.
    centres = (
        [(1, x) for x in range(3, 64, 6)]
        + [(0, y) for y in range(3, 63, 6)]
        + [(0, y) for y in range(5, 63, 6)]
    )
    n = len(centres)
    cw = np.zeros((n, 3), dtype=np.float32)
    cb = np.zeros(n, dtype=np.float32)
    for i, (axis, position) in enumerate(centres):
        cw[i, axis] = 63.0
        cb[i] = -float(position)
    conv1("coord_offsets", "coords", cw, cb)
    node("coord_distances", "abs", ["coord_offsets"])
    conv1(
        "coord_tents_raw",
        "coord_distances",
        -np.eye(n, dtype=np.float32),
        np.full(n, 0.5, dtype=np.float32),
    )
    relu("coord_tents", "coord_tents_raw")
    groups = np.zeros((3, n), dtype=np.float32)
    groups[0, :11] = 2.0
    groups[1, 11:21] = 2.0
    groups[2, 21:31] = 2.0
    conv1("lattice_groups", "coord_tents", groups)
    node("x_lattice", "slice", ["lattice_groups"], start=0, stop=1)
    node("y_normal", "slice", ["lattice_groups"], start=1, stop=2)
    node("y_reverse", "slice", ["lattice_groups"], start=2, stop=3)
    binary("normal_player", "mul", "player", "y_normal")
    binary("reverse_player", "mul", "player", "y_reverse")
    node("normal_phase", "gmax", ["normal_player"])
    node("reverse_phase", "gmax", ["reverse_player"])
    binary("normal_rows", "mul", "y_normal", "normal_phase")
    binary("reverse_rows", "mul", "y_reverse", "reverse_phase")
    binary("active_rows", "add", "normal_rows", "reverse_rows")
    binary("grid", "mul", "x_lattice", "active_rows")

    # Remember gravity and use a cooldown after a reversal.
    node("old_normal", "vslice", ["mem"], start=0, stop=1)
    node("old_reverse", "vslice", ["mem"], start=1, stop=2)
    node("old_cooldown", "vslice", ["mem"], start=2, stop=3)
    node("old_visits", "vslice", ["mem"], start=3, stop=14)
    node("changed_normal", "vmul", ["normal_phase", "old_reverse"])
    node("changed_reverse", "vmul", ["reverse_phase", "old_normal"])
    node("phase_changed", "vadd", ["changed_normal", "changed_reverse"])
    dense("cooldown_dec_raw", "old_cooldown", [[1.0]], [-1.0])
    node("cooldown_dec", "vrelu", ["cooldown_dec_raw"])
    dense("cooldown_new", "phase_changed", [[12.0]])
    node("cooldown", "vadd", ["cooldown_dec", "cooldown_new"])
    dense("switch_allowed_raw", "cooldown", [[-1.0]], [1.0])
    node("switch_allowed", "vrelu", ["switch_allowed_raw"])

    # A small decaying visitation record helps break horizontal oscillations.
    node("x_tents_half", "slice", ["coord_tents"], start=0, stop=11)
    conv1("x_tents", "x_tents_half", 2.0 * np.eye(11, dtype=np.float32))
    binary("player_column_maps", "mul", "x_tents", "player")
    node("player_column", "gmax", ["player_column_maps"])
    dense("old_visits_decay", "old_visits", 0.75 * np.eye(11, dtype=np.float32))
    dense("keep_visits", "phase_changed", [[-1.0]], [1.0])
    node("visits_kept", "vmul", ["old_visits_decay", "keep_visits"])
    dense("visit_increment", "player_column", 0.6 * np.eye(11, dtype=np.float32))
    node("visits_raw", "vadd", ["visits_kept", "visit_increment"])
    dense(
        "visit_excess_raw",
        "visits_raw",
        np.eye(11, dtype=np.float32),
        -np.ones(11, dtype=np.float32),
    )
    node("visit_excess", "vrelu", ["visit_excess_raw"])
    dense("negative_excess", "visit_excess", -np.eye(11, dtype=np.float32))
    node("visits_next", "vadd", ["visits_raw", "negative_excess"])
    node(
        "mem_next", "vconcat",
        ["normal_phase", "reverse_phase", "cooldown", "visits_next"]
    )
    binary("visit_column_channels", "mul", "x_tents", "old_visits")
    conv1(
        "visit_column_map", "visit_column_channels",
        np.ones((1, 11), dtype=np.float32)
    )

    for colour, name in [
        (7, "colour_goal"),
        (8, "colour_green"),
        (10, "colour_empty"),
        (12, "colour_purple"),
        (14, "colour_rock"),
        (15, "colour_yellow"),
    ]:
        node(name, "slice", ["x0"], start=colour, stop=colour + 1)

    # Solid purple has a purple pixel immediately right of its centre.
    w = np.zeros((1, 1, 3, 3), dtype=np.float32)
    w[0, 0, 1, 2] = 1.0
    param("p_right_one", w)
    node("purple_right", "conv", ["colour_purple"], w="p_right_one")
    binary("purple_solid_raw", "mul", "colour_purple", "purple_right")
    binary("purple_ghost_raw", "sub", "colour_purple", "purple_solid_raw")

    # Yellow blocks are wider at their centre than yellow spikes.
    w = np.zeros((1, 1, 5, 5), dtype=np.float32)
    w[0, 0, 2, 4] = 1.0
    param("p_right_two", w)
    node("yellow_right", "conv", ["colour_yellow"], w="p_right_two")
    binary("gold_raw", "mul", "colour_yellow", "yellow_right")
    binary("spike_raw", "sub", "colour_yellow", "gold_raw")

    for name, source in [
        ("empty_tile", "colour_empty"),
        ("goal_tile", "colour_goal"),
        ("green_tile", "colour_green"),
        ("rock_tile", "colour_rock"),
        ("gold_tile", "gold_raw"),
        ("spike_tile", "spike_raw"),
        ("purple_solid", "purple_solid_raw"),
        ("purple_ghost", "purple_ghost_raw"),
        ("player_tile", "player"),
    ]:
        binary(name, "mul", source, "grid")

    binary("open_a", "max2", "empty_tile", "purple_ghost")
    binary("open_without_goal", "max2", "open_a", "player_tile")
    binary("open_tile", "max2", "open_without_goal", "goal_tile")
    binary("break_a", "max2", "rock_tile", "purple_solid")
    binary("break_tile", "max2", "break_a", "gold_tile")

    # Propagate spike danger against gravity through open tiles.
    def make_danger(loop_name, state_name, prefix, opposite=False):
        body = []
        up = shift6(prefix + "_up", state_name, "p_up", body)
        down = shift6(prefix + "_down", state_name, "p_down", body)
        normal_source, reverse_source = (down, up) if opposite else (up, down)
        body.extend([
            {
                "id": prefix + "_normal",
                "op": "mul",
                "in": [normal_source, "normal_phase"],
            },
            {
                "id": prefix + "_reverse",
                "op": "mul",
                "in": [reverse_source, "reverse_phase"],
            },
            {
                "id": prefix + "_incoming",
                "op": "add",
                "in": [prefix + "_normal", prefix + "_reverse"],
            },
            {
                "id": prefix + "_through",
                "op": "mul",
                "in": [prefix + "_incoming", "open_without_goal"],
            },
            {
                "id": prefix + "_update",
                "op": "max2",
                "in": ["spike_tile", prefix + "_through"],
            },
        ])
        graph.append({
            "id": loop_name,
            "op": "loop",
            "T": 12,
            "state": [{"id": state_name, "init": "spike_tile"}],
            "body": body,
            "next": {state_name: prefix + "_update"},
        })
        return loop_name + "." + state_name

    danger = make_danger("danger_loop", "danger_state", "danger")
    opposite_danger = make_danger(
        "opposite_loop", "opposite_state", "opposite_danger", True
    )
    conv1("safe_open_raw", danger, [[-1.0]], [1.0])
    relu("safe_open", "safe_open_raw")
    danger_up = shift6("danger_after_up", danger, "p_up")
    danger_down = shift6("danger_after_down", danger, "p_down")
    binary("danger_after_normal", "mul", danger_up, "normal_phase")
    binary("danger_after_reverse", "mul", danger_down, "reverse_phase")
    binary("danger_after", "add", "danger_after_normal", "danger_after_reverse")
    conv1("safe_break_raw", "danger_after", [[-1.0]], [1.0])
    relu("safe_break", "safe_break_raw")
    binary("safe_open_tile", "mul", "open_tile", "safe_open")
    binary("safe_break_tile", "mul", "break_tile", "safe_break")
    binary("safe_pass_tile", "max2", "safe_open_tile", "safe_break_tile")

    open_up = shift6("open_above", "open_tile", "p_up")
    open_down = shift6("open_below", "open_tile", "p_down")
    binary("open_gravity_normal", "mul", open_up, "normal_phase")
    binary("open_gravity_reverse", "mul", open_down, "reverse_phase")
    binary("open_gravity", "add", "open_gravity_normal", "open_gravity_reverse")
    conv1("support_raw", "open_gravity", [[-1.0]], [1.0])
    relu("support", "support_raw")
    binary("standing", "max2", "support", "player_tile")

    # Seed a value field with gravity-directed progress and visible goals.
    conv1("normal_progress", "coords", [[-70.0, 0.0, 0.0]], [75.0])
    conv1("reverse_progress", "coords", [[70.0, 0.0, 0.0]], [5.0])
    binary("normal_progress_active", "mul", "normal_progress", "normal_phase")
    binary("reverse_progress_active", "mul", "reverse_progress", "reverse_phase")
    binary("progress", "add", "normal_progress_active", "reverse_progress_active")
    conv1("goal_bonus", "goal_tile", [[200.0]])
    binary("seed_unmasked", "add", "progress", "goal_bonus")
    binary("seed", "mul", "seed_unmasked", "safe_pass_tile")
    conv1("break_cost", "break_tile", [[1.5]])
    conv1("gold_extra_cost", "gold_tile", [[2.0]])
    binary("extra_cost", "add", "break_cost", "gold_extra_cost")
    conv1("step_cost", "extra_cost", [[1.0]], [0.5])

    body = []
    vl = shift6("value_left", "value_state", "p_left", body)
    vr = shift6("value_right", "value_state", "p_right", body)
    vu = shift6("value_up", "value_state", "p_up", body)
    vd = shift6("value_down", "value_state", "p_down", body)
    body.extend([
        {"id": "value_normal", "op": "mul", "in": [vu, "normal_phase"]},
        {"id": "value_reverse", "op": "mul", "in": [vd, "reverse_phase"]},
        {
            "id": "value_gravity",
            "op": "add",
            "in": ["value_normal", "value_reverse"],
        },
        {"id": "value_horizontal_raw", "op": "max2", "in": [vl, vr]},
        {
            "id": "value_horizontal",
            "op": "mul",
            "in": ["value_horizontal_raw", "standing"],
        },
        {
            "id": "value_neighbour",
            "op": "max2",
            "in": ["value_horizontal", "value_gravity"],
        },
        {"id": "value_less_cost", "op": "sub", "in": ["value_neighbour", "step_cost"]},
        {"id": "value_clipped", "op": "relu", "in": ["value_less_cost"]},
        {"id": "value_candidate", "op": "max2", "in": ["seed", "value_clipped"]},
        {"id": "value_update", "op": "mul", "in": ["value_candidate", "safe_pass_tile"]},
    ])
    graph.append({
        "id": "value_loop",
        "op": "loop",
        "T": 80,
        "state": [{"id": "value_state", "init": "seed"}],
        "body": body,
        "next": {"value_state": "value_update"},
    })

    binary("value_player_map", "mul", "value_loop.value_state", "player_tile")
    node("value_player", "gmax", ["value_player_map"])
    binary("improvement_raw", "sub", "value_loop.value_state", "value_player")
    conv1("improvement_offset", "improvement_raw", [[1.0]], [-0.1])
    relu("improvement", "improvement_offset")

    left = shift6("player_to_left", "player_tile", "p_right")
    right = shift6("player_to_right", "player_tile", "p_left")
    up = shift6("player_to_up", "player_tile", "p_down")
    down = shift6("player_to_down", "player_tile", "p_up")
    binary("gravity_target_normal", "mul", up, "normal_phase")
    binary("gravity_target_reverse", "mul", down, "reverse_phase")
    binary("gravity_target", "add", "gravity_target_normal", "gravity_target_reverse")

    binary("left_legal_map", "mul", left, "safe_open_tile")
    binary("right_legal_map", "mul", right, "safe_open_tile")
    node("left_legal", "gmax", ["left_legal_map"])
    node("right_legal", "gmax", ["right_legal_map"])
    binary("left_score_map", "mul", "left_legal_map", "improvement")
    binary("right_score_map", "mul", "right_legal_map", "improvement")
    node("left_score", "gmax", ["left_score_map"])
    node("right_score", "gmax", ["right_score_map"])
    binary("left_visit_map", "mul", left, "visit_column_map")
    binary("right_visit_map", "mul", right, "visit_column_map")
    node("left_visit", "gmax", ["left_visit_map"])
    node("right_visit", "gmax", ["right_visit_map"])
    node("left_legal_visit", "vmul", ["left_visit", "left_legal"])
    node("right_legal_visit", "vmul", ["right_visit", "right_legal"])

    binary("side_targets", "max2", left, right)
    binary("side_break_targets", "mul", "side_targets", "break_tile")
    binary("gravity_break_targets", "mul", "gravity_target", "safe_break_tile")
    binary("break_targets", "max2", "side_break_targets", "gravity_break_targets")
    binary("break_score_map", "mul", "break_targets", "improvement")

    # Safe gravity switches are a low-priority fallback.
    binary("opposite_danger_player_map", "mul", opposite_danger, "player_tile")
    node("opposite_danger_player", "gmax", ["opposite_danger_player_map"])
    dense("reversal_safety_raw", "opposite_danger_player", [[-1.0]], [1.0])
    node("reversal_safety", "vrelu", ["reversal_safety_raw"])
    node("switch_safety", "vmul", ["reversal_safety", "switch_allowed"])
    binary("safe_green", "mul", "green_tile", "switch_safety")
    conv1("green_fallback", "safe_green", [[0.05]])

    # Mark dangerous adjacent columns.
    binary("left_danger_map", "mul", left, danger)
    binary("right_danger_map", "mul", right, danger)
    node("left_danger", "gmax", ["left_danger_map"])
    node("right_danger", "gmax", ["right_danger_map"])
    node("danger_nearby", "vadd", ["left_danger", "right_danger"])
    binary("adjacent_danger_map", "max2", "left_danger_map", "right_danger_map")
    node("adjacent_danger_columns", "colmax", ["adjacent_danger_map"])

    node("coord_row", "slice", ["coords"], start=0, stop=1)
    node("coord_col", "slice", ["coords"], start=1, stop=2)
    binary("player_row_map", "mul", "coord_row", "player_tile")
    binary("player_col_map", "mul", "coord_col", "player_tile")
    node("player_row", "gmax", ["player_row_map"])
    node("player_col", "gmax", ["player_col_map"])

    # A ghost-purple tile in a dangerous adjacent fall column can be
    # made solid to create support. Prefer the closest such tile.
    binary("row_above_player", "sub", "player_row", "coord_row")
    binary("row_below_player", "sub", "coord_row", "player_row")
    binary("gravity_row_normal", "mul", "row_above_player", "normal_phase")
    binary("gravity_row_reverse", "mul", "row_below_player", "reverse_phase")
    binary("gravity_row_distance", "add", "gravity_row_normal", "gravity_row_reverse")
    conv1("ahead_raw", "gravity_row_distance", [[63.0]], [-3.0])
    relu("ahead_positive", "ahead_raw")
    conv1("ahead_excess_raw", "ahead_positive", [[1.0]], [-1.0])
    relu("ahead_excess", "ahead_excess_raw")
    binary("ahead_mask", "sub", "ahead_positive", "ahead_excess")
    binary("ghost_danger", "mul", "purple_ghost", danger)
    binary("ghost_column", "mul", "ghost_danger", "adjacent_danger_columns")
    binary("ghost_ahead", "mul", "ghost_column", "ahead_mask")
    node("row_distance_abs", "abs", ["row_above_player"])
    conv1("ghost_proximity", "row_distance_abs", [[-0.02]], [0.10])
    binary("ghost_fallback", "mul", "ghost_ahead", "ghost_proximity")

    # Yellow-block fallback for building a support near a dangerous gap.
    node(
        "gold_target_features", "vconcat",
        [
            "player_row", "player_col", "left_danger", "right_danger",
            "normal_phase", "reverse_phase",
        ],
    )
    target_w = np.zeros((2, 6), dtype=np.float32)
    target_w[0, 0] = 1.0
    target_w[0, 4] = -12.0 / 63.0
    target_w[0, 5] = 12.0 / 63.0
    target_w[1, 1] = 1.0
    target_w[1, 2] = -6.0 / 63.0
    target_w[1, 3] = 6.0 / 63.0
    dense("gold_target", "gold_target_features", target_w)
    node("gold_target_row", "vslice", ["gold_target"], start=0, stop=1)
    node("gold_target_col", "vslice", ["gold_target"], start=1, stop=2)
    binary("gold_row_delta", "sub", "coord_row", "gold_target_row")
    binary("gold_col_delta", "sub", "coord_col", "gold_target_col")
    node("gold_row_distance", "abs", ["gold_row_delta"])
    node("gold_col_distance", "abs", ["gold_col_delta"])
    binary("gold_distance", "add", "gold_row_distance", "gold_col_distance")
    conv1("gold_proximity_raw", "gold_distance", [[-1.0]], [1.0])
    relu("gold_proximity", "gold_proximity_raw")
    binary("gold_near_tile", "mul", "gold_proximity", "gold_tile")
    binary("gold_gap_candidate", "mul", "gold_near_tile", "danger_nearby")
    conv1("gold_gap_fallback", "gold_gap_candidate", [[0.08]])

    # If boxed in by spreading blocks, opening a side block is preferable
    # to repeatedly walking into it.
    binary("side_gold", "mul", "side_targets", "gold_tile")
    conv1("side_gold_fallback", "side_gold", [[0.055]])

    binary("click_a", "max2", "break_score_map", "green_fallback")
    binary("click_b", "max2", "click_a", "ghost_fallback")
    binary("click_c", "max2", "click_b", "gold_gap_fallback")
    binary("click", "max2", "click_c", "side_gold_fallback")
    node("click_score", "gmax", ["click"])

    node("legal_pair", "vconcat", ["left_legal", "right_legal"])
    previous_w = np.zeros((2, 8), dtype=np.float32)
    previous_w[0, 3] = 1.0
    previous_w[1, 4] = 1.0
    dense("previous_direction", "aprev", previous_w)
    node("legal_previous", "vmul", ["legal_pair", "previous_direction"])
    node(
        "action_features",
        "vconcat",
        [
            "left_score", "right_score", "click_score", "legal_pair",
            "legal_previous", "left_legal_visit", "right_legal_visit",
        ],
    )
    out_w = np.zeros((8, 9), dtype=np.float32)
    out_w[3, 0] = 1.0
    out_w[4, 1] = 1.0
    out_w[6, 2] = 1.0
    out_w[3, 3] = 0.02
    out_w[4, 4] = 0.02
    out_w[3, 5] = 0.005
    out_w[4, 6] = 0.005
    out_w[3, 7] = -0.012
    out_w[4, 8] = -0.012
    out_b = np.full(8, -10.0, dtype=np.float32)
    out_b[3] = out_b[4] = out_b[6] = 0.0
    dense("logits", "action_features", out_w, out_b)

    return {"graph": graph, "memory_size": 14, "params": params}
