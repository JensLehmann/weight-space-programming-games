import numpy as np


def build():
    graph = []
    params = {}
    serial = 0

    def name():
        nonlocal serial
        serial += 1
        return "n%d" % serial

    def param(value):
        p = name()
        params[p] = np.asarray(value, dtype=np.float32)
        return p

    def node(op, inputs, **fields):
        i = name()
        graph.append({"id": i, "op": op, "in": inputs, **fields})
        return i

    def conv(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], **fields)

    def sa(x, weight, bias=0.0):
        return conv(
            x,
            np.array([[[[weight]]]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
        )

    def va(xs, weights, bias=0.0):
        z = node("vconcat", xs)
        return dense(
            z,
            np.asarray([weights], dtype=np.float32),
            np.asarray([bias], dtype=np.float32),
        )

    def ssum(xs, weights):
        z = node("concat", xs)
        w = np.asarray(weights, dtype=np.float32).reshape(
            1, len(xs), 1, 1
        )
        return conv(z, w)

    def vclip(x):
        a = node("vrelu", [x])
        b = node("vrelu", [va([x], [1.0], -1.0)])
        return va([a, b], [1.0, -1.0])

    def sclip(x):
        a = node("relu", [x])
        b = node("relu", [sa(x, 1.0, -1.0)])
        return node("sub", [a, b])

    def vslice(x, i):
        return node("vslice", [x], start=i, stop=i + 1)

    def sslice(x, i):
        return node("slice", [x], start=i, stop=i + 1)

    blue = node("slice", ["x0"], start=11, stop=12)
    row = sa(node("slice", ["coords"], start=0, stop=1), 63.0)
    col = sa(node("slice", ["coords"], start=1, stop=2), 63.0)

    def shape_detector(templates, size, center):
        w = np.zeros((len(templates), 1, 7, 7), dtype=np.float32)
        b = np.zeros(len(templates), dtype=np.float32)
        for k, template in enumerate(templates):
            occupied = 0
            for y in range(size):
                for x in range(size):
                    value = bool(template[y, x])
                    occupied += int(value)
                    w[k, 0, y - center + 3, x - center + 3] = (
                        1.0 if value else -1.0
                    )
            b[k] = -occupied + 0.5
        raw = conv(blue, w, b)
        positive = node("relu", [raw])
        scale = np.eye(len(templates), dtype=np.float32).reshape(
            len(templates), len(templates), 1, 1
        ) * 2.0
        return conv(positive, scale)

    block = np.ones((4, 4), dtype=np.float32)
    block[3, 1:3] = 0.0
    blocks = [np.rot90(block, -r).copy() for r in range(4)]
    blocks2 = [
        np.repeat(np.repeat(t, 2, axis=0), 2, axis=1)[:7, :7]
        for t in blocks
    ]
    obj1 = shape_detector(blocks, 4, 1)
    obj2 = shape_detector(blocks2, 7, 3)

    scale2_by_rotation = node("gmax", [obj2])
    scale2_sum = dense(
        scale2_by_rotation, np.ones((1, 4), dtype=np.float32)
    )
    scale2 = vclip(scale2_sum)
    not_scale2 = va([scale2], [-1.0], 1.0)
    obj1_only = node("mul", [obj1, not_scale2])
    objects = node("add", [obj1_only, obj2])

    goal = np.zeros((6, 6), dtype=np.float32)
    goal[1:5, 0] = 1.0
    goal[1:5, 5] = 1.0
    goal[4, 2:4] = 1.0
    goal[5, :] = 1.0
    goals = [np.rot90(goal, -r).copy() for r in range(4)]
    goal_maps = shape_detector(goals, 6, 2)

    sum4 = np.ones((1, 4, 1, 1), dtype=np.float32)
    object_map = conv(objects, sum4)
    goal_map = conv(goal_maps, sum4)

    obj_row = node("gmax", [node("mul", [object_map, row])])
    obj_col = node("gmax", [node("mul", [object_map, col])])
    goal_row = node("gmax", [node("mul", [goal_map, row])])
    goal_col = node("gmax", [node("mul", [goal_map, col])])

    dx = va([goal_col, obj_col, scale2], [1.0, -1.0, 2.0])
    dy = va([goal_row, obj_row, scale2], [1.0, -1.0, 2.0])
    dx_neg = node("vrelu", [va([dx], [-1.0])])
    dx_pos = node("vrelu", [dx])
    dy_neg = node("vrelu", [va([dy], [-1.0])])
    dy_pos = node("vrelu", [dy])

    left = vclip(dx_neg)
    right = vclip(dx_pos)
    up = vclip(dy_neg)
    down = vclip(dy_pos)
    horizontal_steps = va([dx_neg, dx_pos], [0.25, 0.25])
    vertical_steps = va([dy_neg, dy_pos], [0.25, 0.25])

    obj_rot = node("gmax", [objects])
    goal_rot = node("gmax", [goal_maps])
    obj_r = [vslice(obj_rot, i) for i in range(4)]
    goal_r = [vslice(goal_rot, i) for i in range(4)]
    rotation_delta = []
    for d in (1, 2, 3):
        terms = [
            node("vmul", [obj_r[i], goal_r[(i + d) % 4]])
            for i in range(4)
        ]
        rotation_delta.append(vclip(va(terms, [1.0] * 4)))
    rot1, rot2, rot3 = rotation_delta
    rotate_needed = vclip(va(rotation_delta, [1.0, 1.0, 1.0]))

    def stroke_detector(vertical):
        w = np.zeros((1, 16, 7, 7), dtype=np.float32)
        for d in (-1, 0, 1):
            yy, xx = (3 + d, 3) if vertical else (3, 3 + d)
            w[0, 1, yy, xx] = 1.0
            w[0, 5, yy, xx] = 1.0
        for d in (-2, 2):
            yy, xx = (3 + d, 3) if vertical else (3, 3 + d)
            w[0, 0, yy, xx] = 1.0
        s = node("relu", [conv("x0", w, [-4.5])])
        return sa(s, 2.0)

    horizontal = stroke_detector(False)
    vertical = stroke_detector(True)
    after_31 = sclip(sa(row, 1.0, -31.0))
    before_51 = sclip(sa(row, -1.0, 51.0))
    roi = node("mul", [after_31, before_51])
    horizontal = node("mul", [horizontal, roi])
    vertical = node("mul", [vertical, roi])

    col_reverse = sa(col, -1.0, 64.0)
    row_reverse = sa(row, -1.0, 64.0)
    first_col_reverse = node(
        "gmax", [node("mul", [horizontal, col_reverse])]
    )
    first_row_reverse = node(
        "gmax", [node("mul", [horizontal, row_reverse])]
    )
    first_col = va([first_col_reverse], [-1.0], 64.0)
    first_row = va([first_row_reverse], [-1.0], 64.0)
    slot = sa(node("sub", [col, first_col]), 0.2)

    # Direct-program branch.
    end0 = scale2
    end1 = va([end0, horizontal_steps], [1.0, 1.0])
    end2 = va([end1, vertical_steps], [1.0, 1.0])
    end3 = va([end2, rotate_needed], [1.0, 1.0])

    def before(end):
        return sclip(node("sub", [end, slot]))

    p0, p1, p2, p3 = [before(e) for e in (end0, end1, end2, end3)]
    shrink_slot = p0
    horizontal_slot = node("sub", [p1, p0])
    vertical_slot = node("sub", [p2, p1])
    rotation_slot = node("sub", [p3, p2])

    hl = node("mul", [horizontal_slot, left])
    hr = node("mul", [horizontal_slot, right])
    vu = node("mul", [vertical_slot, up])
    vd = node("mul", [vertical_slot, down])
    rr1 = node("mul", [rotation_slot, rot1])
    rr2 = node("mul", [rotation_slot, rot2])
    rr3 = node("mul", [rotation_slot, rot3])

    bit0 = ssum([hl, vu, vd, shrink_slot, rr1, rr2], [1.0] * 6)
    bit1 = ssum([hr, vd, rr2, rr3], [1.0] * 4)
    bit2 = rotation_slot
    bit3 = shrink_slot
    zero_spatial = sa(slot, 0.0)
    direct_bits = [bit0, bit1, bit2, bit3, zero_spatial, vu]

    relative_row = node("sub", [row, first_row])
    row_masks = []
    for i in range(6):
        offset = sa(relative_row, 1.0, -3.0 * i)
        distance = node("abs", [offset])
        row_masks.append(node("relu", [sa(distance, -1.0, 1.0)]))
    row_stack = node("concat", row_masks)
    direct_terms = [
        node("mul", [direct_bits[i], row_masks[i]]) for i in range(6)
    ]
    direct_desired = ssum(direct_terms, [1.0] * 6)

    # Visible checkpoint detector.
    checker0 = np.fromfunction(
        lambda y, x: ((y + x) % 2 == 0), (4, 4), dtype=int
    ).astype(np.float32)
    checker1 = 1.0 - checker0
    checkers = shape_detector([checker0, checker1], 4, 1)
    checker = node("max2", [sslice(checkers, 0), sslice(checkers, 1)])
    checker_present = vclip(node("gmax", [checker]))

    # Free 4x4 object positions for the split-board layout.
    wall = node("slice", ["x0"], start=6, stop=7)
    wall_kernel = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for yy in range(-1, 3):
        for xx in range(-1, 3):
            wall_kernel[0, 0, yy + 2, xx + 2] = 1.0
    wall_count = conv(wall, wall_kernel)
    no_wall = node("relu", [sa(wall_count, -1.0, 1.0)])

    xmin = sclip(sa(col, 1.0, -33.0))
    xmax = sclip(sa(col, -1.0, 59.0))
    ymin = sclip(sa(row, 1.0, -4.0))
    ymax = sclip(sa(row, -1.0, 30.0))
    bounds = node("mul", [xmin, xmax])
    bounds = node("mul", [bounds, ymin])
    bounds = node("mul", [bounds, ymax])
    free = node("mul", [bounds, no_wall])
    checker = node("mul", [checker, free])
    goal_anchor = node("mul", [goal_map, free])
    start_anchor = node("mul", [object_map, free])

    # Four shifts of four pixels, implemented as two 5x5 convolutions.
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    shift1 = np.zeros((4, 1, 5, 5), dtype=np.float32)
    shift2 = np.zeros((4, 4, 5, 5), dtype=np.float32)
    for i, (dx_, dy_) in enumerate(directions):
        yy = 2 - 2 * dy_
        xx = 2 - 2 * dx_
        shift1[i, 0, yy, xx] = 1.0
        shift2[i, i, yy, xx] = 1.0
    shift1_param = param(shift1)
    shift2_param = param(shift2)
    minus_one_w = param(np.array([[[[1.0]]]], dtype=np.float32))
    minus_one_b = param(np.array([-1.0], dtype=np.float32))

    def distance_wave(source):
        seed = sa(source, 100.0)
        loop_id = name()
        state_id = name()
        body = []

        def bn(op, ins, **fields):
            i = name()
            body.append({"id": i, "op": op, "in": ins, **fields})
            return i

        a = bn("conv", [state_id], w=shift1_param)
        b = bn("conv", [a], w=shift2_param)
        parts = [
            bn("slice", [b], start=i, stop=i + 1) for i in range(4)
        ]
        m = bn("max2", [parts[0], parts[1]])
        m = bn("max2", [m, parts[2]])
        m = bn("max2", [m, parts[3]])
        m = bn("conv", [m], w=minus_one_w, b=minus_one_b)
        m = bn("relu", [m])
        m = bn("max2", [m, state_id])
        m = bn("mul", [m, free])
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 40,
            "state": [{"id": state_id, "init": seed}],
            "body": body,
            "next": {state_id: m},
        })
        return loop_id + "." + state_id

    goal_wave = distance_wave(goal_anchor)
    start_wave = distance_wave(start_anchor)

    start_goal_value = node(
        "gmax", [node("mul", [goal_wave, start_anchor])]
    )
    direct_threshold = va(
        [start_goal_value, rotate_needed, scale2],
        [1.0, -1.0, -1.0],
        -93.0,
    )
    goal_reachable = vclip(direct_threshold)

    reachable_checkpoint = sclip(sa(start_wave, 1.0, -93.0))
    reachable_checkpoint = node(
        "mul", [reachable_checkpoint, checker]
    )
    checkpoint_score_map = node(
        "mul", [reachable_checkpoint, goal_wave]
    )
    checkpoint_score = node("gmax", [checkpoint_score_map])
    improvement = vclip(
        va([checkpoint_score, start_goal_value], [1.0, -1.0])
    )
    not_direct = va([goal_reachable], [-1.0], 1.0)
    take_checkpoint = node("vmul", [not_direct, improvement])
    take_goal = va([take_checkpoint], [-1.0], 1.0)

    score_difference = node("sub", [goal_wave, checkpoint_score])
    score_distance = node("abs", [score_difference])
    best_checkpoint = node(
        "relu", [sa(score_distance, -1.0, 1.0)]
    )
    best_checkpoint = node(
        "mul", [best_checkpoint, reachable_checkpoint]
    )
    target_goal = node("mul", [goal_anchor, take_goal])
    target_checkpoint = node("mul", [best_checkpoint, take_checkpoint])
    target = node("add", [target_goal, target_checkpoint])
    target_wave = distance_wave(target)

    # Iteratively follow the target wave and accumulate six encoded commands.
    zero_vector = dense("avail", np.zeros((1, 8), dtype=np.float32))
    bit_matrix = np.array([
        [1, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.float32)
    bit_matrix_param = param(bit_matrix)
    sum6_param = param(np.ones((1, 6, 1, 1), dtype=np.float32))

    plan_id = name()
    pos_state = name()
    step_state = name()
    rot_state = name()
    output_state = name()
    body = []

    def bn(op, ins, **fields):
        i = name()
        body.append({"id": i, "op": op, "in": ins, **fields})
        return i

    def bva(xs, weights, bias=0.0):
        z = bn("vconcat", xs)
        w = param(np.asarray([weights], dtype=np.float32))
        b = param(np.asarray([bias], dtype=np.float32))
        return bn("dense", [z], w=w, b=b)

    def bsa(x, weight, bias=0.0):
        w = param(np.array([[[[weight]]]], dtype=np.float32))
        b = param(np.array([bias], dtype=np.float32))
        return bn("conv", [x], w=w, b=b)

    def bvclip(x):
        a = bn("vrelu", [x])
        b = bn("vrelu", [bva([x], [1.0], -1.0)])
        return bva([a, b], [1.0, -1.0])

    def bvmax(a, b):
        d = bva([a, b], [1.0, -1.0])
        return bn("vadd", [b, bn("vrelu", [d])])

    at_target = bvclip(
        bn("gmax", [bn("mul", [pos_state, target])])
    )
    moving = bva([at_target], [-1.0], 1.0)
    a = bn("conv", [pos_state], w=shift1_param)
    candidates = bn("conv", [a], w=shift2_param)
    candidate_maps = [
        bn("slice", [candidates], start=i, stop=i + 1)
        for i in range(4)
    ]
    scores = []
    for i in range(4):
        score = bn(
            "gmax", [bn("mul", [candidate_maps[i], target_wave])]
        )
        scores.append(bva([score], [1.0], 0.01 * (4 - i)))

    best = scores[0]
    for score in scores[1:]:
        best = bvmax(best, score)

    choices = []
    for score in scores:
        difference = bva([score, best], [1000.0, -1000.0], 1.0)
        selected = bvclip(difference)
        choices.append(bn("vmul", [selected, moving]))

    not_rotated = bva([rot_state], [-1.0], 1.0)
    rotation_event = bn("vmul", [at_target, take_goal])
    rotation_event = bn("vmul", [rotation_event, rotate_needed])
    rotation_event = bn("vmul", [rotation_event, not_rotated])
    next_rot = bn("vadd", [rot_state, rotation_event])
    rotations = [
        bn("vmul", [rotation_event, r]) for r in (rot1, rot2, rot3)
    ]
    controls = bn("vconcat", choices + rotations)
    bits = bn("dense", [controls], w=bit_matrix_param)

    weighted_rows = bn("mul", [row_stack, bits])
    bit_map = bn("conv", [weighted_rows], w=sum6_param)
    slot_delta = bn("sub", [slot, step_state])
    slot_distance = bn("abs", [slot_delta])
    slot_mask = bn("relu", [bsa(slot_distance, -1.0, 1.0)])
    slot_bits = bn("mul", [slot_mask, bit_map])
    next_output = bn("add", [output_state, slot_bits])

    moved = [
        bn("mul", [candidate_maps[i], choices[i]]) for i in range(4)
    ]
    stay = bn("mul", [pos_state, at_target])
    positions = bn("concat", moved + [stay])
    sum5 = param(np.ones((1, 5, 1, 1), dtype=np.float32))
    next_pos = bn("conv", [positions], w=sum5)
    next_step = bva([step_state], [1.0], 1.0)

    graph.append({
        "id": plan_id,
        "op": "loop",
        "T": 6,
        "state": [
            {"id": pos_state, "init": start_anchor},
            {"id": step_state, "init": zero_vector},
            {"id": rot_state, "init": zero_vector},
            {"id": output_state, "init": zero_spatial},
        ],
        "body": body,
        "next": {
            pos_state: next_pos,
            step_state: next_step,
            rot_state: next_rot,
            output_state: next_output,
        },
    })
    path_desired = plan_id + "." + output_state

    not_checker = va([checker_present], [-1.0], 1.0)
    direct_part = node("mul", [direct_desired, not_checker])
    path_part = node("mul", [path_desired, checker_present])
    desired = node("add", [direct_part, path_part])

    valid_row = ssum(row_masks, [1.0] * 6)
    stroke = node("max2", [horizontal, vertical])
    eligible = node("mul", [stroke, valid_row])
    current_bit = node("slice", ["x0"], start=5, stop=6)
    difference = node("abs", [node("sub", [desired, current_bit])])
    mismatch = node("mul", [eligible, difference])
    mismatch_score = sa(mismatch, 4.0)

    nine = node("slice", ["x0"], start=9, stop=10)
    button_w = np.ones((1, 1, 7, 7), dtype=np.float32)
    button = node("relu", [conv(nine, button_w, [-48.5])])
    button = sa(button, 2.0)
    graph.append({
        "id": "click",
        "op": "add",
        "in": [mismatch_score, button],
    })

    logits_w = np.zeros((8, 8), dtype=np.float32)
    logits_b = np.zeros(8, dtype=np.float32)
    logits_b[6] = 1.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": ["avail"],
        "w": param(logits_w),
        "b": param(logits_b),
    })

    return {"graph": graph, "memory_size": 0, "params": params}
