import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]

    def fresh():
        serial[0] += 1
        return "n%d" % serial[0]

    def emit(op, inputs, out=None, dst=None, **fields):
        if out is None:
            out = fresh()
        nd = {"id": out, "op": op, "in": list(inputs)}
        nd.update(fields)
        (graph if dst is None else dst).append(nd)
        return out

    def weighted(op, inp, w, b=None, out=None, dst=None):
        if out is None:
            out = fresh()
        wn = "W_" + out
        params[wn] = np.asarray(w, dtype=np.float32)
        fields = {"w": wn}
        if b is not None:
            bn = "b_" + out
            params[bn] = np.asarray(b, dtype=np.float32)
            fields["b"] = bn
        return emit(op, [inp], out=out, dst=dst, **fields)

    def conv(inp, w, b=None, out=None, dst=None):
        return weighted("conv", inp, w, b, out, dst)

    def dense(inp, w, b=None, out=None, dst=None):
        return weighted("dense", inp, w, b, out, dst)

    def sl(inp, start, stop, vector=False, dst=None):
        return emit(
            "vslice" if vector else "slice",
            [inp], dst=dst, start=start, stop=stop
        )

    def meet(items):
        result = items[0]
        for item in items[1:]:
            result = emit("min2", [result, item])
        return result

    one = sl("coords", 2, 3)

    # Coordinate predicates in the target's local 10x10 drawing area.
    # Each predicate is (a * row + b * col >= threshold).
    # Saturation margins make their values exactly zero or one.
    predicates = [
        (1, 0, 3),       # 0: row >= 3
        (-1, 0, -12),    # 1: row <= 12
        (0, 1, 3),       # 2: col >= 3
        (0, -1, -12),    # 3: col <= 12
        (-1, 0, -7),     # 4: top half
        (1, 0, 8),       # 5: bottom half
        (0, -1, -7),     # 6: left half
        (0, 1, 8),       # 7: right half
        (-1, 0, -5),     # 8: top stamp rows
        (1, 0, 10),      # 9: bottom stamp rows
        (0, -1, -5),     # 10: left stamp columns
        (0, 1, 10),      # 11: right stamp columns
        (1, 0, 6),       # 12: middle rows lower bound
        (-1, 0, -9),     # 13: middle rows upper bound
        (0, 1, 6),       # 14: middle columns lower bound
        (0, -1, -9),     # 15: middle columns upper bound
        (1, -1, 1),      # 16: below main diagonal
        (-1, 1, 1),      # 17: above main diagonal
        (1, 1, 16),      # 18: below anti-diagonal
        (-1, -1, -14),   # 19: above anti-diagonal
    ]
    wp = np.zeros((20, 3, 1, 1), np.float32)
    bp = np.zeros(20, np.float32)
    for i, (a, b, threshold) in enumerate(predicates):
        wp[i, 0, 0, 0] = 252 * a
        wp[i, 1, 0, 0] = 252 * b
        bp[i] = 2 - 4 * threshold

    p_raw = conv("coords", wp, bp)
    p_pos = emit("relu", [p_raw])
    p_binary = emit("min2", [p_pos, one])
    p = [sl(p_binary, i, i + 1) for i in range(20)]

    not_main = emit("max2", [p[16], p[17]])
    not_anti = emit("max2", [p[18], p[19]])
    roi = meet([p[0], p[1], p[2], p[3], not_main, not_anti])

    # Stroke order: eight basket orientations, followed by four stamps.
    masks = [
        meet([roi, p[4]]),
        meet([roi, p[17]]),
        meet([roi, p[7]]),
        meet([roi, p[18]]),
        meet([roi, p[5]]),
        meet([roi, p[16]]),
        meet([roi, p[6]]),
        meet([roi, p[19]]),
        meet([roi, p[8], p[14], p[15]]),
        meet([roi, p[11], p[12], p[13]]),
        meet([roi, p[9], p[14], p[15]]),
        meet([roi, p[10], p[12], p[13]]),
    ]
    mask_stack = emit("concat", masks)

    # Expand each stroke mask over all sixteen colour channels.
    expand_masks = np.zeros((192, 12, 1, 1), np.float32)
    for i in range(192):
        expand_masks[i, i // 16, 0, 0] = 1
    mask_colours = conv(mask_stack, expand_masks)

    # Translate the current canvas into target coordinates.
    colour_image = conv(
        "x0", np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
    )
    shift_w = np.zeros((1, 1, 7, 7), np.float32)
    shift_w[0, 0, 6, 5] = 1  # Read from row+3, col+2.
    shift_body = []
    shifted_iteration = conv("shift_state", shift_w, dst=shift_body)
    graph.append({
        "id": "transport",
        "op": "loop",
        "T": 10,
        "state": [{"id": "shift_state", "init": colour_image}],
        "body": shift_body,
        "next": {"shift_state": shifted_iteration},
    })

    shift_tail = np.zeros((1, 1, 7, 7), np.float32)
    shift_tail[0, 0, 4, 6] = 1
    translated = conv("transport.shift_state", shift_tail)
    shift_last = np.zeros((1, 1, 3, 3), np.float32)
    shift_last[0, 0, 1, 2] = 1
    translated = conv(translated, shift_last)

    difference = emit("sub", [translated, colour_image])
    difference = emit("abs", [difference])
    mismatch = emit("min2", [difference, one])

    # Reverse-layer planner. A footprint is peelable precisely when its
    # uncovered target cells contain exactly one distinct colour.
    count_colours = np.zeros((12, 192), np.float32)
    repeat_selection = np.zeros((192, 12), np.float32)
    sum_selected_colours = np.zeros((16, 192), np.float32)
    for s in range(12):
        count_colours[s, 16 * s:16 * (s + 1)] = 1
        repeat_selection[16 * s:16 * (s + 1), s] = 1
        sum_selected_colours[:, 16 * s:16 * (s + 1)] = np.eye(
            16, dtype=np.float32
        )

    eye12 = np.eye(12, dtype=np.float32)
    validity_w1 = np.concatenate([-eye12, -eye12], axis=0)
    validity_b1 = np.concatenate([
        np.full(12, 2, np.float32),
        np.ones(12, np.float32),
    ])
    validity_w2 = np.concatenate([eye12, -2 * eye12], axis=1)
    priority_w = -np.tril(np.ones((12, 12), np.float32), k=-1)

    body = []
    remaining_colours = emit(
        "mul", ["x0", "remaining"], dst=body
    )
    repeated_colours = emit(
        "concat", [remaining_colours] * 12, dst=body
    )
    footprint_colours = emit(
        "mul", [repeated_colours, mask_colours], dst=body
    )
    presence = emit("gmax", [footprint_colours], dst=body)
    counts = dense(presence, count_colours, dst=body)

    # For integral counts:
    # relu(2-count) - 2*relu(1-count) == [count == 1].
    validity_parts = dense(
        counts, validity_w1, validity_b1, dst=body
    )
    validity_parts = emit("vrelu", [validity_parts], dst=body)
    valid = dense(validity_parts, validity_w2, dst=body)

    first_gate = dense(
        valid, priority_w, np.ones(12, np.float32), dst=body
    )
    first_gate = emit("vrelu", [first_gate], dst=body)
    selection = emit("vmul", [valid, first_gate], dst=body)

    selected_channels = dense(
        selection, repeat_selection, dst=body
    )
    selected_presence = emit(
        "vmul", [presence, selected_channels], dst=body
    )
    selected_colour = dense(
        selected_presence, sum_selected_colours, dst=body
    )
    candidate = emit(
        "vconcat", [selection, selected_colour], dst=body
    )

    uncovered_errors = emit(
        "mul", [mismatch, "remaining"], dst=body
    )
    need = emit("gmax", [uncovered_errors], dst=body)
    no_need = dense(
        need, np.array([[-1]], np.float32),
        np.array([1], np.float32), dst=body
    )
    new_part = emit("vmul", [candidate, need], dst=body)
    old_part = emit("vmul", ["saved_plan", no_need], dst=body)
    plan_next = emit("vadd", [new_part, old_part], dst=body)

    selected_masks = emit(
        "mul", [mask_stack, selection], dst=body
    )
    peeled_mask = conv(
        selected_masks, np.ones((1, 12, 1, 1), np.float32),
        dst=body
    )
    remaining_next = emit(
        "sub", ["remaining", peeled_mask], dst=body
    )
    remaining_next = emit("relu", [remaining_next], dst=body)

    graph.append({
        "id": "peel",
        "op": "loop",
        "T": 12,
        "state": [
            {"id": "remaining", "init": roi},
            {"id": "saved_plan", "zeros": [28]},
        ],
        "body": body,
        "next": {
            "remaining": remaining_next,
            "saved_plan": plan_next,
        },
    })

    desired_shape = sl("peel.saved_plan", 0, 12, vector=True)
    desired_colour = sl("peel.saved_plan", 12, 28, vector=True)
    desired_stamp = sl(desired_shape, 8, 12, vector=True)

    # Point detectors: eight unique red basket-border anchors, then the
    # four small-stamp click centres. These follow the simulator geometry.
    points = [
        (24, 25), (21, 39), (32, 46), (46, 49),
        (53, 25), (56, 24), (32, 17), (31, 14),
        (20, 32), (39, 50), (57, 32), (39, 13),
    ]
    point_w1 = np.zeros((24, 3, 1, 1), np.float32)
    point_b1 = np.zeros(24, np.float32)
    point_w2 = np.zeros((12, 24, 1, 1), np.float32)
    for i, (row, col) in enumerate(points):
        point_w1[2 * i, 0, 0, 0] = 63
        point_w1[2 * i + 1, 1, 0, 0] = 63
        point_b1[2 * i] = -row
        point_b1[2 * i + 1] = -col
        point_w2[i, 2 * i, 0, 0] = -4
        point_w2[i, 2 * i + 1, 0, 0] = -4

    point_dist = conv("coords", point_w1, point_b1)
    point_dist = emit("abs", [point_dist])
    point_scores = conv(
        point_dist, point_w2, np.full(12, 2, np.float32)
    )
    point_scores = emit("relu", [point_scores])
    point_masks = emit("min2", [point_scores, one])

    basket_anchors = sl(point_masks, 0, 8)
    stamp_centres = sl(point_masks, 8, 12)
    red = sl("x0", 2, 3)
    orientation_map = emit("mul", [basket_anchors, red])
    current_orientation = emit("gmax", [orientation_map])

    # Locate palette buttons by their 5x5 border, and the selected button
    # by the five black pixels immediately below it.
    padding_colour = sl("x0", 4, 5)
    black = sl("x0", 0, 1)
    border_w = np.zeros((1, 1, 5, 5), np.float32)
    border_w[0, 0, 0, :] = 1
    border_w[0, 0, 4, :] = 1
    border_w[0, 0, :, 0] = 1
    border_w[0, 0, :, 4] = 1
    buttons = conv(
        padding_colour, border_w, np.array([-15], np.float32)
    )
    buttons = emit("relu", [buttons])

    underline_w = np.zeros((1, 1, 7, 7), np.float32)
    underline_w[0, 0, 6, 1:6] = 1
    underline = conv(
        black, underline_w, np.array([-4], np.float32)
    )
    underline = emit("relu", [underline])
    active_button = emit("mul", [buttons, underline])
    active_colour_map = emit("mul", ["x0", active_button])
    current_colour = emit("gmax", [active_colour_map])

    # Desired orientation, including the cardinal orientations of stamps.
    shape_orientations = list(range(8)) + [0, 2, 4, 6]
    orientation_w = np.zeros((8, 12), np.float32)
    for s, orientation in enumerate(shape_orientations):
        orientation_w[orientation, s] = 1
    desired_orientation = dense(desired_shape, orientation_w)

    # Form all current/desired orientation pairs and route around the ring.
    current_repeat = np.zeros((64, 8), np.float32)
    desired_repeat = np.zeros((64, 8), np.float32)
    routing = np.zeros((4, 64), np.float32)
    equal_orientation = np.zeros((1, 64), np.float32)
    increasing_action = [4, 2, 2, 3, 3, 1, 1, 4]
    decreasing_action = [3, 3, 1, 1, 4, 4, 2, 2]

    for current in range(8):
        for desired in range(8):
            k = 8 * current + desired
            current_repeat[k, current] = 1
            desired_repeat[k, desired] = 1
            distance = (desired - current) % 8
            if distance == 0:
                equal_orientation[0, k] = 1
            else:
                action = (
                    increasing_action[current]
                    if distance <= 4
                    else decreasing_action[current]
                )
                routing[action - 1, k] = 1

    current_pairs = dense(current_orientation, current_repeat)
    desired_pairs = dense(desired_orientation, desired_repeat)
    pairs = emit("vmul", [current_pairs, desired_pairs])
    movement = dense(pairs, routing)
    orientation_matches = dense(pairs, equal_orientation)

    colour_overlap = emit(
        "vmul", [current_colour, desired_colour]
    )
    colour_matches = dense(
        colour_overlap, np.ones((1, 16), np.float32)
    )
    colour_differs = dense(
        colour_matches, np.array([[-1]], np.float32),
        np.array([1], np.float32)
    )
    is_stamp = dense(
        desired_stamp, np.ones((1, 4), np.float32)
    )
    is_large = dense(
        is_stamp, np.array([[-1]], np.float32),
        np.array([1], np.float32)
    )

    ready = emit("vmul", [colour_matches, orientation_matches])
    do_large = emit("vmul", [ready, is_large])
    do_stamp = emit("vmul", [ready, is_stamp])
    do_click = emit("vadd", [colour_differs, do_stamp])
    do_move = emit("vmul", [movement, colour_matches])
    controls = emit("vconcat", [do_move, do_large, do_click])

    output_w = np.zeros((8, 6), np.float32)
    for i in range(4):
        output_w[i + 1, i] = 20
    output_w[5, 4] = 20
    output_w[6, 5] = 20
    output_b = np.full(8, -1, np.float32)
    output_b[0] = -100
    output_b[7] = -100
    dense(controls, output_w, output_b, out="logits")

    # Click either the desired-colour button or the desired small stamp.
    desired_colour_map = emit("mul", ["x0", desired_colour])
    desired_colour_map = conv(
        desired_colour_map, np.ones((1, 16, 1, 1), np.float32)
    )
    palette_click = emit("mul", [desired_colour_map, buttons])
    palette_click = emit("mul", [palette_click, colour_differs])

    stamp_click = emit("mul", [stamp_centres, desired_stamp])
    stamp_click = conv(
        stamp_click, np.ones((1, 4, 1, 1), np.float32)
    )
    stamp_click = emit("mul", [stamp_click, colour_matches])
    emit("add", [palette_click, stamp_click], out="click")

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }
