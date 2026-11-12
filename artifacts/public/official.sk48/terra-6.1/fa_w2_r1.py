import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]

    def fresh(prefix):
        serial[0] += 1
        return "%s_%d" % (prefix, serial[0])

    def node(op, inputs, **fields):
        ident = fresh(op)
        graph.append({"id": ident, "op": op, "in": inputs, **fields})
        return ident

    def conv(x, weights, bias=None):
        wn = fresh("w")
        params[wn] = np.asarray(weights, dtype=np.float32)
        fields = {"w": wn}
        if bias is not None:
            bn = fresh("b")
            params[bn] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bn
        return node("conv", [x], **fields)

    def dense(x, weights, bias=None):
        wn = fresh("w")
        params[wn] = np.asarray(weights, dtype=np.float32)
        fields = {"w": wn}
        if bias is not None:
            bn = fresh("b")
            params[bn] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bn
        return node("dense", [x], **fields)

    def unary(op, x):
        return node(op, [x])

    def binary(op, x, y):
        return node(op, [x, y])

    def spatial_affine(x, scale, bias=0.0):
        return conv(x, np.array([[[[scale]]]], dtype=np.float32), [bias])

    def vector_affine(x, scale, bias=0.0):
        return dense(x, [[scale]], [bias])

    def shifted_color(color, dy, dx):
        w = np.zeros((1, 16, 7, 7), dtype=np.float32)
        w[0, color, 3 + dy, 3 + dx] = 1.0
        return conv("x0", w)

    def coordinate_of(mask, coord):
        return unary("gmax", binary("mul", mask, coord))

    def vector_saturate(x):
        excess = unary("vrelu", vector_affine(x, 1.0, -1.0))
        return binary("vadd", x, vector_affine(excess, -1.0))

    def vector_sum(*items):
        result = items[0]
        for item in items[1:]:
            result = binary("vadd", result, item)
        return result

    def vector_abs(x):
        positive = unary("vrelu", x)
        negative = unary("vrelu", vector_affine(x, -1.0))
        return binary("vadd", positive, negative)

    def part(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    row = node("slice", ["coords"], start=0, stop=1)
    col = node("slice", ["coords"], start=1, stop=2)
    ones = node("slice", ["coords"], start=2, stop=3)

    board = unary("relu", spatial_affine(row, -63.0, 53.0))
    board = binary("min2", board, ones)
    display = unary("relu", spatial_affine(row, 63.0, -53.0))
    display = binary("min2", display, ones)

    # Selected arm head.
    w = np.zeros((1, 16, 7, 7), dtype=np.float32)
    for dy, dx in [(-1, -1), (-1, 2), (2, -1), (2, 2)]:
        w[0, 0, 3 + dy, 3 + dx] = 2.0
    w[0, 6, 3, 3] = 2.0
    w[0, 15, 3, 3] = 2.0
    head = unary("relu", conv("x0", w, [-9.0]))
    head = binary("mul", head, board)

    hy = coordinate_of(head, row)
    hx = coordinate_of(head, col)

    probe_right = shifted_color(1, 0, 3)
    probe_down = shifted_color(1, 3, 0)
    probe_left = shifted_color(1, 0, -2)
    probe_up = shifted_color(1, -2, 1)

    def orientation(probe):
        return unary("gmax", binary("mul", head, probe))

    ori_r = orientation(probe_right)
    ori_d = orientation(probe_down)
    ori_l = orientation(probe_left)
    ori_u = orientation(probe_up)
    horizontal = binary("vadd", ori_r, ori_l)
    vertical = binary("vadd", ori_d, ori_u)

    # Tile masks, one channel per possible non-background colour.
    colors = list(range(6, 16))
    w = np.zeros((len(colors), 16, 7, 7), dtype=np.float32)
    for i, color in enumerate(colors):
        for dy, dx in [(-1, -1), (-1, 2), (2, -1), (2, 2)]:
            w[i, color, 3 + dy, 3 + dx] = 2.0
    tiles = unary("relu", conv("x0", w, [-7.0] * len(colors)))

    # Select the first unmatched target in the lower display.
    black = node("slice", ["x0"], start=0, stop=1)
    not_black = binary("sub", ones, black)
    targets = binary("mul", tiles, display)
    targets = binary("mul", targets, probe_right)
    targets = binary("mul", targets, not_black)
    target_priority = spatial_affine(col, -63.0, 64.0)
    target_scores = binary("mul", targets, target_priority)
    per_color_score = unary("gmax", target_scores)
    score_sum = conv(
        target_scores,
        np.ones((1, len(colors), 1, 1), dtype=np.float32),
    )
    best_target = unary("gmax", score_sum)
    difference = binary(
        "vadd", per_color_score, vector_affine(best_target, -1.0)
    )
    chosen_color = dense(
        difference,
        np.eye(len(colors), dtype=np.float32),
        np.ones(len(colors), dtype=np.float32),
    )
    chosen_color = unary("vrelu", chosen_color)

    # Separate attached and unattached tiles.
    active_edge = binary("add", probe_right, probe_down)
    active_edge = binary("add", active_edge, probe_left)
    active_edge = binary("add", active_edge, probe_up)
    unattached = unary("relu", binary("sub", ones, active_edge))

    world_tiles = binary("mul", tiles, board)
    attached = binary("mul", world_tiles, active_edge)
    attached = conv(
        attached,
        np.ones((1, len(colors), 1, 1), dtype=np.float32),
    )
    attached_count = vector_affine(unary("gmean", attached), 4096.0)

    # Crossing-beam evidence at attached tiles.
    cross_v1 = shifted_color(2, -2, 0)
    cross_v2 = shifted_color(2, 3, 0)
    cross_h1 = shifted_color(2, 0, -2)
    cross_h2 = shifted_color(2, 0, 3)
    cross_v = binary("max2", cross_v1, cross_v2)
    cross_h = binary("max2", cross_h1, cross_h2)
    crossing_horizontal = unary("gmax", binary("mul", attached, cross_v))
    crossing_vertical = unary("gmax", binary("mul", attached, cross_h))
    crossing_horizontal = vector_saturate(crossing_horizontal)
    crossing_vertical = vector_saturate(crossing_vertical)
    free_horizontal = vector_affine(crossing_horizontal, -1.0, 1.0)
    free_vertical = vector_affine(crossing_vertical, -1.0, 1.0)

    candidates = binary("mul", world_tiles, chosen_color)
    candidates = conv(
        candidates,
        np.ones((1, len(colors), 1, 1), dtype=np.float32),
    )
    candidates = binary("mul", candidates, unattached)

    # Nearest candidate, with a deterministic tie-break.
    dist_y = unary("abs", binary("sub", row, hy))
    dist_x = unary("abs", binary("sub", col, hx))
    distance = binary("add", dist_y, dist_x)
    closeness = spatial_affine(distance, -6300.0, 16000.0)
    closeness = binary("sub", closeness, spatial_affine(row, 10.0))
    closeness = binary("sub", closeness, col)
    candidate_score = binary("mul", candidates, closeness)
    best_candidate = unary("gmax", candidate_score)
    candidate_gap = binary("sub", candidate_score, best_candidate)
    selected = unary("relu", spatial_affine(candidate_gap, 1.0, 0.02))
    selected = binary("mul", selected, candidates)
    selected = spatial_affine(selected, 50.0)

    gy = coordinate_of(selected, row)
    gx = coordinate_of(selected, col)
    dy = binary("vadd", gy, vector_affine(hy, -1.0))
    dx = binary("vadd", gx, vector_affine(hx, -1.0))

    pos_y = unary("vrelu", vector_affine(dy, 63.0))
    neg_y = unary("vrelu", vector_affine(dy, -63.0))
    pos_x = unary("vrelu", vector_affine(dx, 63.0))
    neg_x = unary("vrelu", vector_affine(dx, -63.0))
    abs_y = binary("vadd", pos_y, neg_y)
    abs_x = binary("vadd", pos_x, neg_x)
    different_y = vector_saturate(abs_y)
    different_x = vector_saturate(abs_x)
    same_y = vector_affine(different_y, -1.0, 1.0)
    same_x = vector_affine(different_x, -1.0, 1.0)

    # Estimate beam length from selected colour-1 pixels on the head axis.
    color1 = node("slice", ["x0"], start=1, stop=2)
    head_row = unary("rowmax", head)
    head_col = unary("colmax", head)
    horizontal_line = binary("mul", color1, head_row)
    horizontal_line = binary("mul", horizontal_line, board)
    vertical_line = binary("mul", color1, head_col)
    vertical_line = binary("mul", vertical_line, board)

    inverse_col = spatial_affine(col, -1.0, 1.0)
    inverse_row = spatial_affine(row, -1.0, 1.0)
    tail_r = coordinate_of(horizontal_line, col)
    tail_l_inv = coordinate_of(horizontal_line, inverse_col)
    tail_d = coordinate_of(vertical_line, row)
    tail_u_inv = coordinate_of(vertical_line, inverse_row)

    count_distance = vector_affine(attached_count, -6.0)
    excess_r = vector_sum(
        vector_affine(tail_r, 63.0),
        vector_affine(hx, -63.0),
        count_distance,
    )
    excess_r = unary("vrelu", vector_affine(excess_r, 1.0, -3.0))

    excess_l = vector_sum(
        vector_affine(tail_l_inv, 63.0, -63.0),
        vector_affine(hx, 63.0),
        count_distance,
    )
    excess_l = unary("vrelu", vector_affine(excess_l, 1.0, -2.0))

    excess_d = vector_sum(
        vector_affine(tail_d, 63.0),
        vector_affine(hy, -63.0),
        count_distance,
    )
    excess_d = unary("vrelu", vector_affine(excess_d, 1.0, -3.0))

    excess_u = vector_sum(
        vector_affine(tail_u_inv, 63.0, -63.0),
        vector_affine(hy, 63.0),
        count_distance,
    )
    excess_u = unary("vrelu", vector_affine(excess_u, 1.0, -2.0))

    ex_r = vector_saturate(excess_r)
    ex_l = vector_saturate(excess_l)
    ex_d = vector_saturate(excess_d)
    ex_u = vector_saturate(excess_u)

    retract_r = binary("vmul", binary("vmul", ori_r, ex_r), different_y)
    retract_l = binary("vmul", binary("vmul", ori_l, ex_l), different_y)
    retract_d = binary("vmul", binary("vmul", ori_d, ex_d), different_x)
    retract_u = binary("vmul", binary("vmul", ori_u, ex_u), different_x)
    retract_r = binary("vmul", retract_r, free_horizontal)
    retract_l = binary("vmul", retract_l, free_horizontal)
    retract_d = binary("vmul", retract_d, free_vertical)
    retract_u = binary("vmul", retract_u, free_vertical)

    retract_any = vector_saturate(
        vector_sum(retract_r, retract_l, retract_d, retract_u)
    )
    not_retracting = vector_affine(retract_any, -1.0, 1.0)

    up = vector_sum(
        binary("vmul", horizontal, neg_y),
        binary("vmul", ori_u, same_x),
    )
    down = vector_sum(
        binary("vmul", horizontal, pos_y),
        binary("vmul", ori_d, same_x),
    )
    left = vector_sum(
        binary("vmul", vertical, neg_x),
        binary("vmul", ori_l, same_y),
    )
    right = vector_sum(
        binary("vmul", vertical, pos_x),
        binary("vmul", ori_r, same_y),
    )

    up = binary("vmul", up, not_retracting)
    down = binary("vmul", down, not_retracting)
    left = binary("vmul", left, not_retracting)
    right = binary("vmul", right, not_retracting)
    up = binary("vadd", up, vector_affine(retract_d, 100.0))
    down = binary("vadd", down, vector_affine(retract_u, 100.0))
    left = binary("vadd", left, vector_affine(retract_r, 100.0))
    right = binary("vadd", right, vector_affine(retract_l, 100.0))

    # Memory: previous head position and four retreat-direction flags.
    old_hx = part("mem", 0, 1)
    old_hy = part("mem", 1, 2)
    old_up = part("mem", 2, 3)
    old_down = part("mem", 3, 4)
    old_left = part("mem", 4, 5)
    old_right = part("mem", 5, 6)

    prev_up = part("aprev", 1, 2)
    prev_down = part("aprev", 2, 3)
    prev_left = part("aprev", 3, 4)
    prev_right = part("aprev", 4, 5)

    moved_x = vector_abs(binary("vadd", hx, vector_affine(old_hx, -1.0)))
    moved_y = vector_abs(binary("vadd", hy, vector_affine(old_hy, -1.0)))
    moved = vector_sum(moved_x, moved_y)
    unchanged = unary("vrelu", vector_affine(moved, -63.0, 1.0))

    old_active = vector_saturate(
        vector_sum(old_up, old_down, old_left, old_right)
    )
    no_old_mode = vector_affine(old_active, -1.0, 1.0)
    trigger_base = binary("vmul", unchanged, no_old_mode)

    cross_for_h = binary("vmul", horizontal, crossing_horizontal)
    cross_for_v = binary("vmul", vertical, crossing_vertical)
    start_up = binary(
        "vmul", binary("vmul", trigger_base, prev_down), cross_for_h
    )
    start_down = binary(
        "vmul", binary("vmul", trigger_base, prev_up), cross_for_h
    )
    start_left = binary(
        "vmul", binary("vmul", trigger_base, prev_right), cross_for_v
    )
    start_right = binary(
        "vmul", binary("vmul", trigger_base, prev_left), cross_for_v
    )

    cross_now = vector_saturate(vector_sum(cross_for_h, cross_for_v))
    excess_now = vector_saturate(vector_sum(
        binary("vmul", ori_r, ex_r),
        binary("vmul", ori_l, ex_l),
        binary("vmul", ori_d, ex_d),
        binary("vmul", ori_u, ex_u),
    ))
    keep_mode = vector_saturate(vector_sum(cross_now, excess_now))

    mode_up = binary("vmul", vector_sum(old_up, start_up), keep_mode)
    mode_down = binary("vmul", vector_sum(old_down, start_down), keep_mode)
    mode_left = binary("vmul", vector_sum(old_left, start_left), keep_mode)
    mode_right = binary("vmul", vector_sum(old_right, start_right), keep_mode)
    mode_active = vector_saturate(
        vector_sum(mode_up, mode_down, mode_left, mode_right)
    )
    normal_mode = vector_affine(mode_active, -1.0, 1.0)

    # In retreat mode, move off the crossing first, then compact the beam.
    no_cross = vector_affine(cross_now, -1.0, 1.0)
    compact = binary("vmul", mode_active, no_cross)

    special_up = vector_sum(
        binary("vmul", mode_up, cross_now),
        binary("vmul", compact, binary("vmul", ori_d, ex_d)),
    )
    special_down = vector_sum(
        binary("vmul", mode_down, cross_now),
        binary("vmul", compact, binary("vmul", ori_u, ex_u)),
    )
    special_left = vector_sum(
        binary("vmul", mode_left, cross_now),
        binary("vmul", compact, binary("vmul", ori_r, ex_r)),
    )
    special_right = vector_sum(
        binary("vmul", mode_right, cross_now),
        binary("vmul", compact, binary("vmul", ori_l, ex_l)),
    )

    up = binary(
        "vadd", binary("vmul", up, normal_mode),
        vector_affine(special_up, 1000.0)
    )
    down = binary(
        "vadd", binary("vmul", down, normal_mode),
        vector_affine(special_down, 1000.0)
    )
    left = binary(
        "vadd", binary("vmul", left, normal_mode),
        vector_affine(special_left, 1000.0)
    )
    right = binary(
        "vadd", binary("vmul", right, normal_mode),
        vector_affine(special_right, 1000.0)
    )

    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [hx, hy, mode_up, mode_down, mode_left, mode_right],
    })

    # Click an inactive display head when this arm's targets are complete.
    w = np.zeros((1, 16, 7, 7), dtype=np.float32)
    for ddy, ddx in [(-1, -1), (-1, 2), (2, -1), (2, 2)]:
        w[0, 4, 3 + ddy, 3 + ddx] = 2.0
    w[0, 6, 3, 3] = 2.0
    w[0, 15, 3, 3] = 2.0
    inactive_head = unary("relu", conv("x0", w, [-9.0]))
    inactive_head = binary("mul", inactive_head, display)
    inactive_present = unary("gmax", inactive_head)
    no_unmatched = unary("vrelu", vector_affine(best_target, -1.0, 1.0))
    click_logit = binary("vmul", inactive_present, no_unmatched)
    click_logit = vector_affine(click_logit, 200.0, -1.0)

    disabled = vector_affine(hx, 0.0, -100.0)
    graph.append({
        "id": "logits",
        "op": "vconcat",
        "in": [
            disabled, up, down, left, right, disabled, click_logit, disabled
        ],
    })

    click_weights = fresh("w")
    params[click_weights] = np.ones((1, 1, 1, 1), dtype=np.float32)
    graph.append({
        "id": "click",
        "op": "conv",
        "in": [inactive_head],
        "w": click_weights,
    })

    return {"graph": graph, "memory_size": 6, "params": params}
