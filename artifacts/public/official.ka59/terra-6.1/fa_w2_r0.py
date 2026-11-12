import numpy as np


def build():
    graph = []
    params = {}

    def node(name, op, inputs=None, **fields):
        n = {"id": name, "op": op}
        if inputs is not None:
            n["in"] = list(inputs)
        n.update(fields)
        graph.append(n)
        return name

    def conv(name, source, weight, bias=None):
        wn = name + "_w"
        params[wn] = np.asarray(weight, dtype=np.float32)
        fields = {"w": wn}
        if bias is not None:
            bn = name + "_b"
            params[bn] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bn
        return node(name, "conv", [source], **fields)

    def dense(name, source, weight, bias=None):
        wn = name + "_w"
        params[wn] = np.asarray(weight, dtype=np.float32)
        fields = {"w": wn}
        if bias is not None:
            bn = name + "_b"
            params[bn] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bn
        return node(name, "dense", [source], **fields)

    def sl(name, source, a, b):
        return node(name, "slice", [source], start=a, stop=b)

    def vsl(name, source, a, b):
        return node(name, "vslice", [source], start=a, stop=b)

    def combine(name, kind, values):
        current = values[0]
        for i, value in enumerate(values[1:]):
            current = node(
                name + "_" + str(i), kind, [current, value]
            )
        return current

    def scale(name, source, amount):
        return conv(
            name,
            source,
            np.array([[[[amount]]]], dtype=np.float32),
        )

    colour = {}
    for k in (0, 2, 4, 5, 11, 12, 13, 14, 15):
        colour[k] = sl("colour_" + str(k), "x0", k, k + 1)

    one = sl("one", "coords", 2, 3)
    row = sl("row", "coords", 0, 1)
    col = sl("col", "coords", 1, 2)
    walls = node("walls", "add", [colour[2], colour[15]])
    occluder = combine(
        "occluder", "add",
        [colour[k] for k in (0, 5, 11, 12, 13, 14)]
    )
    piece_body = node(
        "piece_body", "add", [colour[0], colour[14]]
    )
    other_marker = node(
        "other_marker", "add", [colour[4], colour[5]]
    )
    piece_input = node(
        "piece_input", "concat",
        [colour[0], other_marker, piece_body]
    )

    shifts = [(y, x) for y in (-2, 0, 2) for x in (-2, 0, 2)]
    shift_index = {s: i for i, s in enumerate(shifts)}
    shift_w = np.zeros((9, 1, 5, 5), dtype=np.float32)
    for i, (dy, dx) in enumerate(shifts):
        shift_w[i, 0, dy + 2, dx + 2] = 1.0

    goal_shift = conv("goal_shift", colour[4], shift_w)
    occ_shift = conv("occ_shift", occluder, shift_w)
    goal_input = node(
        "goal_input", "concat", [goal_shift, occ_shift]
    )
    cross_shift = conv("cross_shift", colour[11], shift_w)
    hard_shift = conv("hard_shift", colour[2], shift_w)

    def template_slot(dy, dx):
        sy = -2 if dy < -3 else (2 if dy > 3 else 0)
        sx = -2 if dx < -3 else (2 if dx > 3 else 0)
        return shift_index[(sy, sx)], dy - sy + 3, dx - sx + 3

    def detect_goal(name, offsets, minimum_visible):
        w = np.zeros((2, 18, 7, 7), dtype=np.float32)
        for dy, dx in offsets:
            ch, ky, kx = template_slot(dy, dx)
            w[0, ch, ky, kx] += 1.0
            w[0, ch + 9, ky, kx] += 1.0
            w[1, ch, ky, kx] += 1.0
        raw = conv(
            name + "_raw", goal_input, w,
            [1.0 - len(offsets), 1.0 - minimum_visible]
        )
        pos = node(name + "_pos", "relu", [raw])
        complete = sl(name + "_complete", pos, 0, 1)
        visible = sl(name + "_visible", pos, 1, 2)
        return node(name, "min2", [complete, visible])

    def rectangle_outline(h, w):
        ay, ax = (h - 1) // 2, (w - 1) // 2
        return [
            (r - ay, q - ax)
            for r in range(-1, h + 1)
            for q in range(-1, w + 1)
            if r in (-1, h) or q in (-1, w)
        ]

    def piece_kernel(h, w):
        ay, ax = (h - 1) // 2, (w - 1) // 2
        marks = {
            (r, q)
            for r in range(ay, ay + (2 if h == 6 else 1))
            for q in range(ax, ax + (2 if w == 6 else 1))
        }
        weight = np.zeros((2, 3, 7, 7), dtype=np.float32)
        for r in range(h):
            for q in range(w):
                y, x = r - ay + 3, q - ax + 3
                if (r, q) in marks:
                    weight[0, 0, y, x] = 2.0
                    weight[1, 1, y, x] = 2.0
                else:
                    weight[0, 2, y, x] = 2.0
                    weight[1, 2, y, x] = 2.0
        return weight

    def valid_kernel(h, w):
        ay, ax = (h - 1) // 2, (w - 1) // 2
        weight = np.zeros((1, 1, 7, 7), dtype=np.float32)
        for r in range(h):
            for q in range(w):
                weight[0, 0, r - ay + 3, q - ax + 3] = -1.0
        return weight

    directions = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    neighbour_w = np.zeros((4, 1, 7, 7), dtype=np.float32)
    diagonal_w = np.zeros((4, 4, 7, 7), dtype=np.float32)
    for i, (dy, dx) in enumerate(directions):
        neighbour_w[i, 0, dy + 3, dx + 3] = 1.0
        diagonal_w[i, i, dy + 3, dx + 3] = 1.0

    params["neighbour_w"] = neighbour_w
    params["diagonal_w"] = diagonal_w
    params["decay_w"] = np.ones((1, 1, 1, 1), dtype=np.float32)
    params["decay_b"] = np.array([-1.0], dtype=np.float32)

    def neighbours(name, source):
        return node(name, "conv", [source], w="neighbour_w")

    def ray(name, source):
        current = neighbours(name + "_0", source)
        stages = [current]
        for i in range(1, 5):
            current = node(
                name + "_" + str(i),
                "conv", [current], w="diagonal_w"
            )
            stages.append(current)
        return stages

    def wave(name, seed, valid):
        state = name + "_state"
        body = [{
            "id": name + "_neighbours",
            "op": "conv",
            "in": [state],
            "w": "neighbour_w",
        }]
        for i in range(4):
            body.append({
                "id": name + "_n" + str(i),
                "op": "slice",
                "in": [name + "_neighbours"],
                "start": i,
                "stop": i + 1,
            })
        body.extend([
            {
                "id": name + "_m01",
                "op": "max2",
                "in": [name + "_n0", name + "_n1"],
            },
            {
                "id": name + "_m23",
                "op": "max2",
                "in": [name + "_n2", name + "_n3"],
            },
            {
                "id": name + "_m",
                "op": "max2",
                "in": [name + "_m01", name + "_m23"],
            },
            {
                "id": name + "_decay",
                "op": "conv",
                "in": [name + "_m"],
                "w": "decay_w",
                "b": "decay_b",
            },
            {
                "id": name + "_positive",
                "op": "relu",
                "in": [name + "_decay"],
            },
            {
                "id": name + "_retain",
                "op": "max2",
                "in": [state, name + "_positive"],
            },
            {
                "id": name + "_next",
                "op": "mul",
                "in": [name + "_retain", valid],
            },
        ])
        graph.append({
            "id": name,
            "op": "loop",
            "T": 60,
            "state": [{"id": state, "init": seed}],
            "body": body,
            "next": {state: name + "_next"},
        })
        return name + "." + state

    rectangles = []
    ordinary_clicks = []
    ordinary_directions = []

    for i, (h, w) in enumerate(((3, 3), (6, 3), (3, 6), (6, 6))):
        p = "rect_" + str(i)
        goal = detect_goal(
            p + "_goal",
            rectangle_outline(h, w),
            8 if (h, w) == (3, 3) else 12,
        )
        raw = conv(
            p + "_raw",
            piece_input,
            piece_kernel(h, w),
            [1.0 - 2.0 * h * w, 1.0 - 2.0 * h * w],
        )
        detected = node(p + "_detected", "relu", [raw])
        selected = sl(p + "_selected", detected, 0, 1)
        other = sl(p + "_other", detected, 1, 2)
        pieces = node(p + "_pieces", "add", [selected, other])

        empty_raw = node(p + "_empty_raw", "sub", [goal, pieces])
        empty = node(p + "_empty", "relu", [empty_raw])
        selected_raw = node(
            p + "_selected_raw", "sub", [selected, goal]
        )
        selected_active = node(
            p + "_selected_active", "relu", [selected_raw]
        )
        other_raw = node(p + "_other_raw", "sub", [other, goal])
        other_active = node(
            p + "_other_active", "relu", [other_raw]
        )

        valid_raw = conv(
            p + "_valid_raw", walls, valid_kernel(h, w), [1.0]
        )
        valid = node(p + "_valid", "relu", [valid_raw])
        hard_raw = conv(
            p + "_hard_raw", colour[2], valid_kernel(h, w), [1.0]
        )
        hard_valid = node(p + "_hard_valid", "relu", [hard_raw])

        direct_mask = node(
            p + "_direct_mask", "mul", [empty, valid]
        )
        direct_seed = scale(p + "_direct_seed", direct_mask, 100.0)
        direct_field = wave(p + "_direct", direct_seed, valid)

        relaxed_mask = node(
            p + "_relaxed_mask", "mul", [empty, hard_valid]
        )
        relaxed_seed = scale(
            p + "_relaxed_seed", relaxed_mask, 100.0
        )
        relaxed_field = wave(
            p + "_relaxed", relaxed_seed, hard_valid
        )

        ordinary_clicks.append(
            node(
                p + "_click_score", "mul",
                [other_active, direct_field]
            )
        )
        around = neighbours(p + "_around", direct_field)
        selected_around = node(
            p + "_selected_around", "mul",
            [around, selected_active]
        )
        ordinary_directions.append(
            node(p + "_directions", "gmax", [selected_around])
        )
        rectangles.append({
            "h": h,
            "w": w,
            "selected": selected_active,
            "other": other_active,
            "valid": valid,
            "relaxed": relaxed_field,
        })

    outline = [
        "...44444...",
        "...4...4...",
        "...4...4...",
        "4444...4444",
        "4.........4",
        "4.........4",
        "4.........4",
        "4444...4444",
        "...4...4...",
        "...4...4...",
        "...44444...",
    ]
    cross_goal_offsets = [
        (r - 5, q - 5)
        for r, line in enumerate(outline)
        for q, value in enumerate(line)
        if value == "4"
    ]
    cross_goal = detect_goal(
        "cross_goal", cross_goal_offsets, 16
    )

    cross_offsets = [
        (r - 4, q - 4)
        for r in range(9)
        for q in range(9)
        if 3 <= r <= 5 or 3 <= q <= 5
    ]
    cross_kernel = np.zeros((1, 9, 7, 7), dtype=np.float32)
    cross_wall_kernel = np.zeros((1, 9, 7, 7), dtype=np.float32)
    for dy, dx in cross_offsets:
        ch, ky, kx = template_slot(dy, dx)
        cross_kernel[0, ch, ky, kx] += 1.0
        cross_wall_kernel[0, ch, ky, kx] -= 1.0

    cross_raw = conv(
        "cross_raw", cross_shift, cross_kernel,
        [1.0 - len(cross_offsets)]
    )
    cross_piece = node("cross_piece", "relu", [cross_raw])
    cross_target_raw = node(
        "cross_target_raw", "sub", [cross_piece, cross_goal]
    )
    cross_target = node("cross_target", "relu", [cross_target_raw])
    cross_empty_raw = node(
        "cross_empty_raw", "sub", [cross_goal, cross_piece]
    )
    cross_empty = node("cross_empty", "relu", [cross_empty_raw])
    cross_valid_raw = conv(
        "cross_valid_raw", hard_shift, cross_wall_kernel, [1.0]
    )
    cross_valid = node("cross_valid", "relu", [cross_valid_raw])
    cross_mask = node(
        "cross_mask", "mul", [cross_empty, cross_valid]
    )
    cross_seed = scale("cross_seed", cross_mask, 100.0)
    cross_field = wave("cross_field", cross_seed, cross_valid)

    gray_ray = combine(
        "gray_ray_max", "max2", ray("gray_ray", colour[15])
    )

    def sample_offset(name, source, dy, dx):
        y1 = max(-3, min(3, dy))
        x1 = max(-3, min(3, dx))
        y2, x2 = dy - y1, dx - x1
        k1 = np.zeros((1, 1, 7, 7), dtype=np.float32)
        k1[0, 0, y1 + 3, x1 + 3] = 1.0
        current = conv(name + "_a", source, k1)
        if y2 or x2:
            k2 = np.zeros((1, 1, 7, 7), dtype=np.float32)
            k2[0, 0, y2 + 3, x2 + 3] = 1.0
            current = conv(name + "_b", current, k2)
        return current

    def contact_offsets(h, w, direction):
        ay, ax = (h - 1) // 2, (w - 1) // 2
        ys = [1 - ay + 3 * k for k in range((h + 2) // 3)]
        xs = [1 - ax + 3 * k for k in range((w + 2) // 3)]
        if direction == 0:
            return [(-(h - ay + 1), -x) for x in xs]
        if direction == 1:
            return [(ay + 2, -x) for x in xs]
        if direction == 2:
            return [(-y, -(w - ax + 1)) for y in ys]
        return [(-y, ax + 2) for y in ys]

    cross_contact_offsets = [
        [(-6, 0), (-3, -3), (-3, 3)],
        [(6, 0), (3, -3), (3, 3)],
        [(0, -6), (-3, -3), (3, -3)],
        [(0, 6), (-3, 3), (3, 3)],
    ]

    other_seed_parts = [[] for _ in range(4)]
    selected_seed_parts = [[] for _ in range(4)]

    for t in range(5):
        if t < 4:
            r = rectangles[t]
            target_other = r["other"]
            target_selected = r["selected"]
            potential = r["relaxed"]
            h, w = r["h"], r["w"]
        else:
            target_other = cross_target
            target_selected = None
            potential = cross_field
            h, w = 9, 9

        p = "push_target_" + str(t)
        future = combine(
            p + "_future_max", "max2", ray(p + "_future", potential)
        )

        for d in range(4):
            destination = sl(p + "_dest_" + str(d), future, d, d + 1)
            improvement = node(
                p + "_improve_raw_" + str(d),
                "sub", [destination, potential]
            )
            improvement = node(
                p + "_improve_pos_" + str(d), "relu", [improvement]
            )
            improvement = node(
                p + "_improve_binary_" + str(d),
                "min2", [improvement, one]
            )

            if t < 4:
                gray = sl(p + "_gray_" + str(d), gray_ray, d, d + 1)
                gray = node(
                    p + "_gray_binary_" + str(d), "min2", [gray, one]
                )
                improvement = node(
                    p + "_gray_improve_" + str(d),
                    "mul", [improvement, gray]
                )

            mark_other = node(
                p + "_other_mark_" + str(d),
                "mul", [target_other, improvement]
            )
            if target_selected is not None:
                mark_selected = node(
                    p + "_selected_mark_" + str(d),
                    "mul", [target_selected, improvement]
                )

            offsets = (
                cross_contact_offsets[d]
                if t == 4 else contact_offsets(h, w, d)
            )
            for j, (dy, dx) in enumerate(offsets):
                other_seed_parts[d].append(
                    sample_offset(
                        p + "_other_position_%d_%d" % (d, j),
                        mark_other, dy, dx
                    )
                )
                if target_selected is not None:
                    selected_seed_parts[d].append(
                        sample_offset(
                            p + "_selected_position_%d_%d" % (d, j),
                            mark_selected, dy, dx
                        )
                    )

    other_seeds = [
        combine("other_seed_" + str(d), "max2", other_seed_parts[d])
        for d in range(4)
    ]
    selected_seeds = [
        combine(
            "selected_seed_" + str(d),
            "max2", selected_seed_parts[d]
        )
        for d in range(4)
    ]

    other_seed_any = combine("other_seed_any", "max2", other_seeds)
    other_seed_valid = node(
        "other_seed_valid", "mul",
        [other_seed_any, rectangles[0]["valid"]]
    )
    other_seed = scale("other_seed", other_seed_valid, 150.0)
    other_push_field = wave(
        "other_push_field", other_seed, rectangles[0]["valid"]
    )

    selected_seed_any = combine(
        "selected_seed_any", "max2", selected_seeds
    )
    selected_seed_valid = node(
        "selected_seed_valid", "mul",
        [selected_seed_any, rectangles[0]["valid"]]
    )
    selected_seed = scale(
        "selected_seed", selected_seed_valid, 150.0
    )
    selected_push_field = wave(
        "selected_push_field", selected_seed, rectangles[0]["valid"]
    )

    # All detected 3x3 pieces can serve as pushers, including pieces
    # already on goals.
    selected_pusher = sl(
        "selected_pusher", "rect_0_detected", 0, 1
    )
    other_pusher = sl("other_pusher", "rect_0_detected", 1, 2)

    push_click_other = node(
        "push_click_other", "mul", [other_pusher, other_push_field]
    )
    push_click_selected = node(
        "push_click_selected", "mul",
        [other_pusher, selected_push_field]
    )
    push_click = node(
        "push_click", "max2", [push_click_other, push_click_selected]
    )

    push_around = neighbours("push_around", other_push_field)
    push_selected = node(
        "push_selected", "mul", [push_around, selected_pusher]
    )
    push_direction_base = node(
        "push_direction_base", "gmax", [push_selected]
    )

    contacts = []
    for d in range(4):
        contact = node(
            "contact_" + str(d), "mul",
            [other_seeds[d], selected_pusher]
        )
        contacts.append(node("contact_max_" + str(d), "gmax", [contact]))
    contact_vector = node("contact_vector", "vconcat", contacts)
    direct_push = dense(
        "direct_push", contact_vector,
        155.0 * np.eye(4, dtype=np.float32)
    )

    negative_push_base = dense(
        "negative_push_base", push_direction_base,
        -np.eye(4, dtype=np.float32)
    )
    push_difference = node(
        "push_difference", "vadd", [direct_push, negative_push_base]
    )
    push_difference_positive = node(
        "push_difference_positive", "vrelu", [push_difference]
    )
    push_directions = node(
        "push_directions", "vadd",
        [push_direction_base, push_difference_positive]
    )

    ordinary_click = combine(
        "ordinary_click", "max2", ordinary_clicks
    )
    click = node("click", "max2", [ordinary_click, push_click])
    best_click = node("best_click", "gmax", [click])

    d01 = node(
        "ordinary_d01", "vadd",
        [ordinary_directions[0], ordinary_directions[1]]
    )
    d23 = node(
        "ordinary_d23", "vadd",
        [ordinary_directions[2], ordinary_directions[3]]
    )
    direct_directions = node(
        "direct_directions", "vadd", [d01, d23]
    )
    negative_direct = dense(
        "negative_direct", direct_directions,
        -np.eye(4, dtype=np.float32)
    )
    direction_difference = node(
        "direction_difference", "vadd",
        [push_directions, negative_direct]
    )
    direction_difference_positive = node(
        "direction_difference_positive", "vrelu",
        [direction_difference]
    )
    chosen_directions = node(
        "chosen_directions", "vadd",
        [direct_directions, direction_difference_positive]
    )

    # Observation signature for remembering ineffective directions.
    valid_row_raw = conv(
        "valid_row_raw", row,
        np.array([[[[-63.0]]]], dtype=np.float32), [63.0]
    )
    valid_row_pos = node("valid_row_pos", "relu", [valid_row_raw])
    valid_row = node("valid_row", "min2", [valid_row_pos, one])
    tracked = node(
        "tracked", "concat",
        [colour[k] for k in (0, 4, 5, 11, 13, 14)]
    )
    tracked_valid = node("tracked_valid", "mul", [tracked, valid_row])
    tracked_row = node("tracked_row", "mul", [tracked_valid, row])
    tracked_col = node("tracked_col", "mul", [tracked_valid, col])
    mass = node("mass", "gmean", [tracked_valid])
    row_moment = node("row_moment", "gmean", [tracked_row])
    col_moment = node("col_moment", "gmean", [tracked_col])
    signature_mean = node(
        "signature_mean", "vconcat", [mass, row_moment, col_moment]
    )
    signature = dense(
        "signature", signature_mean,
        4096.0 * np.eye(18, dtype=np.float32)
    )

    old_bans = vsl("old_bans", "mem", 0, 4)
    old_signature = vsl("old_signature", "mem", 4, 22)
    negative_old = dense(
        "negative_old", old_signature,
        -np.eye(18, dtype=np.float32)
    )
    delta = node("delta", "vadd", [signature, negative_old])
    positive_delta = node("positive_delta", "vrelu", [delta])
    negative_delta = dense(
        "negative_delta", delta,
        -np.eye(18, dtype=np.float32)
    )
    positive_negative_delta = node(
        "positive_negative_delta", "vrelu", [negative_delta]
    )
    absolute_delta = node(
        "absolute_delta", "vadd",
        [positive_delta, positive_negative_delta]
    )
    same_raw = dense(
        "same_raw", absolute_delta,
        np.full((1, 18), -10.0, dtype=np.float32), [1.0]
    )
    same = node("same", "vrelu", [same_raw])
    previous_direction = vsl("previous_direction", "aprev", 1, 5)
    proposed = node(
        "proposed", "vadd", [old_bans, previous_direction]
    )
    carried = node("carried", "vmul", [proposed, same])
    complement = dense(
        "complement", carried, -np.eye(4, dtype=np.float32),
        np.ones(4, dtype=np.float32)
    )
    complement_positive = node(
        "complement_positive", "vrelu", [complement]
    )
    bans = dense(
        "bans", complement_positive,
        -np.eye(4, dtype=np.float32),
        np.ones(4, dtype=np.float32)
    )
    node("mem_next", "vconcat", [bans, signature])

    penalty = dense(
        "penalty", bans, -200.0 * np.eye(4, dtype=np.float32)
    )
    safe_directions = node(
        "safe_directions", "vadd", [chosen_directions, penalty]
    )
    decisions = node(
        "decisions", "vconcat", [safe_directions, best_click]
    )
    output_w = np.zeros((8, 5), dtype=np.float32)
    for i in range(4):
        output_w[i + 1, i] = 1.0
    output_w[6, 4] = 1.0
    dense(
        "logits", decisions, output_w,
        [-99.0, 0.0, 0.0, 0.0, 0.0, -99.0, -0.5, -99.0]
    )

    return {"graph": graph, "memory_size": 22, "params": params}
