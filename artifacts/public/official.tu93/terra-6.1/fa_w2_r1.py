import numpy as np


def build():
    graph = []
    params = {}

    def n(name, op, inputs, **fields):
        graph.append({"id": name, "op": op, "in": inputs, **fields})
        return name

    def sl(name, src, start, stop):
        return n(name, "slice", [src], start=start, stop=stop)

    def vsl(name, src, start, stop):
        return n(name, "vslice", [src], start=start, stop=stop)

    def conv(name, src, weight, bias=None):
        fields = {"w": weight}
        if bias is not None:
            fields["b"] = bias
        return n(name, "conv", [src], **fields)

    # Track blocks and visible sprite centers.
    extract_w = np.zeros((6, 16, 3, 3), np.float32)
    for c in (0, 2, 4, 8, 9, 11, 12, 13, 14, 15):
        extract_w[0, c, :, :] = 1
    for out, c in enumerate((9, 14, 8, 12, 13), 1):
        extract_w[out, c, :, :] = 1
    params["extract_w"] = extract_w
    params["extract_b"] = np.array([-8, -7, -8, -7, -7, -7], np.float32)

    conv("extract_raw", "x0", "extract_w", "extract_b")
    n("extract", "relu", ["extract_raw"])
    sl("track", "extract", 0, 1)
    sl("player", "extract", 1, 2)
    sl("visible_goal", "extract", 2, 3)
    sl("blue", "extract", 3, 4)
    sl("purple", "extract", 4, 5)
    sl("orange", "extract", 5, 6)
    sl("ones", "coords", 2, 3)
    sl("coord_rc", "coords", 0, 2)

    # Memory begins with goal row, goal column, and a seen flag.
    vsl("old_goal_memory", "mem", 0, 3)
    n("goal_coord_pixels", "mul", ["visible_goal", "coords"])
    n("goal_features", "gmax", ["goal_coord_pixels"])
    vsl("goal_seen_now", "goal_features", 2, 3)
    params["keep_w"] = np.array([[-1]], np.float32)
    params["keep_b"] = np.array([1], np.float32)
    n("keep_old", "dense", ["goal_seen_now"], w="keep_w", b="keep_b")
    n("old_goal_kept", "vmul", ["old_goal_memory", "keep_old"])
    n("goal_memory", "vadd", ["old_goal_kept", "goal_features"])

    params["recall_w"] = np.array([[[[-63.0]], [[-63.0]]]], np.float32)
    params["recall_b"] = np.array([1.0], np.float32)
    vsl("saved_goal_rc", "goal_memory", 0, 2)
    vsl("saved_goal_seen", "goal_memory", 2, 3)
    n("goal_delta", "sub", ["coord_rc", "saved_goal_rc"])
    n("goal_abs_delta", "abs", ["goal_delta"])
    conv("recalled_goal_raw", "goal_abs_delta", "recall_w", "recall_b")
    n("recalled_goal_positive", "relu", ["recalled_goal_raw"])
    n("recalled_goal", "mul", ["recalled_goal_positive", "saved_goal_seen"])
    n("goal", "max2", ["visible_goal", "recalled_goal"])

    # Direction channels: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    marker_w = np.zeros((8, 16, 3, 3), np.float32)
    for d, (dy, dx) in enumerate(directions):
        marker_w[d, 15, 1 + dy, 1 + dx] = 1
        marker_w[d, 11, 1 + dy, 1 + dx] = 1
        marker_w[4 + d, 11, 1 + dy, 1 + dx] = 1
    params["marker_w"] = marker_w
    conv("markers", "x0", "marker_w")
    sl("all_markers", "markers", 0, 4)
    sl("active_markers", "markers", 4, 8)
    n("blue_visible_dirs", "mul", ["blue", "all_markers"])
    n("purple_dirs", "mul", ["purple", "all_markers"])
    n("orange_active_dirs", "mul", ["orange", "active_markers"])

    params["sum4_w"] = np.ones((1, 4, 1, 1), np.float32)
    conv("orange_active_pos", "orange_active_dirs", "sum4_w")
    n("orange_inactive_raw", "sub", ["orange", "orange_active_pos"])
    n("orange_inactive_pos", "relu", ["orange_inactive_raw"])

    # The remaining memory contains eight blue-enemy slots:
    # row, column, and four direction indicators.
    # Restore a remembered blue only where a purple or orange sprite
    # currently covers its last observed center.
    n("blue_occluder", "max2", ["purple", "orange"])
    blue_combined = "blue_visible_dirs"
    slots = 8

    for i in range(slots):
        p = "ghost%d_" % i
        start = 3 + 6 * i
        vsl(p + "rc", "mem", start, start + 2)
        vsl(p + "dir", "mem", start + 2, start + 6)
        n(p + "delta", "sub", ["coord_rc", p + "rc"])
        n(p + "abs", "abs", [p + "delta"])
        conv(p + "raw", p + "abs", "recall_w", "recall_b")
        n(p + "center", "relu", [p + "raw"])
        n(p + "covered", "mul", [p + "center", "blue_occluder"])
        n(p + "dirs", "mul", [p + "covered", p + "dir"])
        n(p + "combined", "max2", [blue_combined, p + "dirs"])
        blue_combined = p + "combined"

    # Pack the current blue maps into the eight slots. The coordinate
    # score orders centers uniquely; each pass extracts the largest.
    params["coord_id_w"] = np.array([[[[64.0]], [[1.0]], [[0.0]]]], np.float32)
    params["coord_id_b"] = np.array([1.0], np.float32)
    params["rank_mask_w"] = np.array([[[[-256.0]]]], np.float32)
    params["rank_mask_b"] = np.array([2.0], np.float32)
    conv("coord_id", "coords", "coord_id_w", "coord_id_b")

    remaining_blue = blue_combined
    blue_memory_parts = []
    for i in range(slots):
        p = "pack%d_" % i
        conv(p + "count", remaining_blue, "sum4_w")
        n(p + "occupied", "min2", [p + "count", "ones"])
        n(p + "score_pixels", "mul", [p + "occupied", "coord_id"])
        n(p + "score", "gmax", [p + "score_pixels"])
        n(p + "score_delta", "sub", ["coord_id", p + "score"])
        n(p + "score_abs", "abs", [p + "score_delta"])
        conv(p + "mask_raw", p + "score_abs", "rank_mask_w", "rank_mask_b")
        n(p + "mask_positive", "relu", [p + "mask_raw"])
        n(p + "mask", "min2", [p + "mask_positive", "ones"])
        n(p + "selected", "mul", [p + "mask", p + "occupied"])
        n(p + "rc_pixels", "mul", [p + "selected", "coord_rc"])
        n(p + "rc", "gmax", [p + "rc_pixels"])
        n(p + "dir_pixels", "mul", [p + "mask", remaining_blue])
        n(p + "dir", "gmax", [p + "dir_pixels"])
        blue_memory_parts.extend([p + "rc", p + "dir"])
        n(p + "not_selected", "sub", ["ones", p + "mask"])
        n(p + "remaining", "mul", [remaining_blue, p + "not_selected"])
        remaining_blue = p + "remaining"

    n("mem_next", "vconcat", ["goal_memory"] + blue_memory_parts)

    # Shifts by three pixels; two shifts correspond to one game move.
    gather_w = np.zeros((4, 1, 7, 7), np.float32)
    gather_diag_w = np.zeros((4, 4, 7, 7), np.float32)
    scatter_w = np.zeros((4, 4, 7, 7), np.float32)
    for d, (dy, dx) in enumerate(directions):
        gather_w[d, 0, 3 + 3 * dy, 3 + 3 * dx] = 1
        gather_diag_w[d, d, 3 + 3 * dy, 3 + 3 * dx] = 1
        scatter_w[d, d, 3 - 3 * dy, 3 - 3 * dx] = 1
    params["gather_w"] = gather_w
    params["gather_diag_w"] = gather_diag_w
    params["scatter_w"] = scatter_w

    reverse_w = np.zeros((4, 4, 1, 1), np.float32)
    for d, opposite in enumerate((1, 0, 3, 2)):
        reverse_w[opposite, d, 0, 0] = 1
    params["reverse_w"] = reverse_w

    sl("edge_color", "x0", 2, 3)
    conv("edge_probes", "edge_color", "gather_w")

    # Blue enemies attack the adjacent square they face.
    conv("blue_half", blue_combined, "scatter_w")
    conv("blue_forward", "blue_half", "scatter_w")
    conv("blue_threat", "blue_forward", "sum4_w")
    conv("blue_count", blue_combined, "sum4_w")
    n("blue_positions", "min2", ["blue_count", "ones"])

    # Static discounted goal field.
    params["discount3_w"] = np.array([[[[0.99]]]], np.float32)
    conv("track_discount", "track", "discount3_w")
    n("goal_seed", "mul", ["goal", "track"])

    graph.append({
        "id": "base_field",
        "op": "loop",
        "T": 192,
        "state": [{"id": "base_state", "init": "goal_seed"}],
        "body": [
            {
                "id": "base_neighbors",
                "op": "conv",
                "in": ["base_state"],
                "w": "gather_w",
            },
            {
                "id": "base_up",
                "op": "slice",
                "in": ["base_neighbors"],
                "start": 0,
                "stop": 1,
            },
            {
                "id": "base_down",
                "op": "slice",
                "in": ["base_neighbors"],
                "start": 1,
                "stop": 2,
            },
            {
                "id": "base_left",
                "op": "slice",
                "in": ["base_neighbors"],
                "start": 2,
                "stop": 3,
            },
            {
                "id": "base_right",
                "op": "slice",
                "in": ["base_neighbors"],
                "start": 3,
                "stop": 4,
            },
            {"id": "base_ud", "op": "max2", "in": ["base_up", "base_down"]},
            {"id": "base_lr", "op": "max2", "in": ["base_left", "base_right"]},
            {"id": "base_best", "op": "max2", "in": ["base_ud", "base_lr"]},
            {
                "id": "base_step",
                "op": "mul",
                "in": ["base_best", "track_discount"],
            },
            {"id": "base_next", "op": "max2", "in": ["base_step", "goal_seed"]},
        ],
        "next": {"base_state": "base_next"},
    })

    # Forecast purple patrols and short-term active orange motion.
    horizon = 20
    purple_state = "purple_dirs"
    orange_state = "orange_active_dirs"
    safety = []
    kill_sources = []

    for t in range(1, horizon + 1):
        p = "t%d_" % t

        conv(p + "purple_source", purple_state, "sum4_w")
        conv(p + "purple_half", purple_state, "scatter_w")
        conv(p + "purple_forward", p + "purple_half", "scatter_w")
        conv(p + "purple_threat", p + "purple_forward", "sum4_w")

        n(p + "purple_open", "mul", [p + "purple_forward", "edge_probes"])
        n(p + "purple_closed", "sub", [p + "purple_forward", p + "purple_open"])
        conv(p + "purple_reverse", p + "purple_closed", "reverse_w")
        n(p + "purple_next", "add", [p + "purple_open", p + "purple_reverse"])
        purple_state = p + "purple_next"

        n(p + "threat0", "max2", [p + "purple_threat", "blue_threat"])
        n(p + "source0", "max2", [p + "purple_source", "blue_positions"])
        n(p + "source1", "max2", [p + "source0", "orange_inactive_pos"])
        threat = p + "threat0"
        source = p + "source1"

        if t <= 2:
            conv(p + "orange_source", orange_state, "sum4_w")
            conv(p + "orange_half", orange_state, "scatter_w")
            conv(p + "orange_forward", p + "orange_half", "scatter_w")
            conv(p + "orange_threat", p + "orange_forward", "sum4_w")
            n(p + "threat1", "max2", [threat, p + "orange_threat"])
            n(p + "source2", "max2", [source, p + "orange_source"])
            threat = p + "threat1"
            source = p + "source2"
            orange_state = p + "orange_forward"

        n(p + "safe_raw", "sub", ["ones", threat])
        n(p + "safe", "relu", [p + "safe_raw"])
        n(p + "source", "min2", [source, "ones"])
        safety.append(p + "safe")
        kill_sources.append(p + "source")

    # Backward time-expanded value propagation.
    params["discount6_w"] = np.array([[[[0.9801]]]], np.float32)
    future = "base_field.base_state"
    for t in range(horizon, 0, -1):
        p = "v%d_" % t
        conv(p + "half", future, "gather_w")
        conv(p + "dest", p + "half", "gather_diag_w")
        n(p + "legal_dest", "mul", [p + "dest", "edge_probes"])
        sl(p + "up", p + "legal_dest", 0, 1)
        sl(p + "down", p + "legal_dest", 1, 2)
        sl(p + "left", p + "legal_dest", 2, 3)
        sl(p + "right", p + "legal_dest", 3, 4)
        n(p + "ud", "max2", [p + "up", p + "down"])
        n(p + "lr", "max2", [p + "left", p + "right"])
        n(p + "best", "max2", [p + "ud", p + "lr"])
        conv(p + "discounted", p + "best", "discount6_w")
        n(p + "with_goal", "max2", [p + "discounted", "goal_seed"])
        n(p + "kill_value", "mul", ["base_field.base_state", kill_sources[t - 1]])
        n(p + "with_kill", "max2", [p + "with_goal", p + "kill_value"])
        n(p + "safe_value", "mul", [p + "with_kill", safety[t - 1]])
        future = p + "safe_value"

    # Read the four destinations at the player center.
    conv("choice_half", future, "gather_w")
    conv("choice_dest", "choice_half", "gather_diag_w")
    n("choice_at_player", "mul", ["choice_dest", "player"])
    n("choice_values", "gmax", ["choice_at_player"])

    conv("safe_half", safety[0], "gather_w")
    conv("safe_dest", "safe_half", "gather_diag_w")
    n("safe_at_player", "mul", ["safe_dest", "player"])
    n("choice_safety", "gmax", ["safe_at_player"])

    n("legal_at_player", "mul", ["edge_probes", "player"])
    n("choice_legal", "gmax", ["legal_at_player"])
    n("features", "vconcat", [
        "choice_values", "choice_safety", "choice_legal", "aprev"
    ])

    out_w = np.zeros((8, 20), np.float32)
    out_b = np.full(8, -100, np.float32)
    for d in range(4):
        action = d + 1
        out_w[action, d] = 1
        out_w[action, 4 + d] = 2
        out_w[action, 8 + d] = 10
        reverse_action = (2, 1, 4, 3)[d]
        out_w[action, 12 + reverse_action] = -0.004
        out_b[action] = -10
    params["out_w"] = out_w
    params["out_b"] = out_b
    n("logits", "dense", ["features"], w="out_w", b="out_b")

    sl("click", "x0", 0, 1)
    return {"graph": graph, "memory_size": 3 + 6 * slots, "params": params}
