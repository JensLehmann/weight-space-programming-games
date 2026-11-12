import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]

    def name(prefix="n"):
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    def node(op, inputs, **fields):
        ident = name()
        graph.append({"id": ident, "op": op, "in": inputs, **fields})
        return ident

    def conv(x, w, b=None):
        wn = name("w")
        params[wn] = np.asarray(w, dtype=np.float32)
        fields = {"w": wn}
        if b is not None:
            bn = name("b")
            params[bn] = np.asarray(b, dtype=np.float32)
            fields["b"] = bn
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        wn = name("w")
        params[wn] = np.asarray(w, dtype=np.float32)
        fields = {"w": wn}
        if b is not None:
            bn = name("b")
            params[bn] = np.asarray(b, dtype=np.float32)
            fields["b"] = bn
        return node("dense", [x], **fields)

    def affine(x, scale, bias=0):
        return conv(x, [[[[scale]]]], [bias])

    def vaffine(x, scale, bias=0):
        return dense(x, [[scale]], [bias])

    def vadd(a, b):
        return node("vadd", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def vnot(x):
        return vaffine(x, -1, 1)

    def clip01(x):
        positive = node("relu", [x])
        excess = node("relu", [affine(positive, 1, -1)])
        return node("sub", [positive, excess])

    def vclip01(x):
        positive = node("vrelu", [x])
        excess = node("vrelu", [vaffine(positive, 1, -1)])
        return vadd(positive, vaffine(excess, -1))

    row = node("slice", ["coords"], start=0, stop=1)
    col = node("slice", ["coords"], start=1, stop=2)
    board = clip01(affine(row, -63, 53))
    row58_distance = node("abs", [affine(row, 63, -58)])
    row58 = clip01(affine(row58_distance, -1, 1.1))

    # Template-match selected and unselected crane rings.
    base = np.array([
        [6, 6, 6, 6, 6, 6],
        [6, 4, 4, 4, 4, 6],
        [6, 4, 6, 6, 4, -1],
        [6, 4, 6, 6, 4, -1],
        [6, 4, 4, 4, 4, 6],
        [6, 6, 6, 6, 6, 6],
    ], dtype=np.int32)

    def ring_filters(selected):
        w = np.zeros((8, 16, 7, 7), dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        k = 0
        for color in (6, 15):
            template = base.copy()
            template[template == 6] = color
            if selected:
                template[template == 4] = 0
            for rotation in range(4):
                t = np.rot90(template, -rotation)
                count = 0
                for y in range(6):
                    for x in range(6):
                        c = int(t[y, x])
                        if c >= 0:
                            w[k, c, y + 1, x + 1] = 1
                            count += 1
                b[k] = -(count - 1)
                k += 1
        return w, b

    ws, bs = ring_filters(True)
    wu, bu = ring_filters(False)
    selected = node("relu", [conv("x0", ws, bs)])
    unselected = node("relu", [conv("x0", wu, bu)])

    sum8 = np.ones((1, 8, 1, 1), dtype=np.float32)
    selected_map = conv(selected, sum8)
    unselected_map = conv(unselected, sum8)
    active_board = node("mul", [selected_map, board])
    inactive_board = node("mul", [unselected_map, board])

    selected_board_channels = node("mul", [selected, board])
    orientations8 = node("gmax", [selected_board_channels])
    orientation_weights = np.zeros((4, 8), dtype=np.float32)
    for i in range(8):
        orientation_weights[i % 4, i] = 1
    orientation = dense(orientations8, orientation_weights)

    axis = dense(orientation, [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])
    horizontal = node("vslice", [axis], start=0, stop=1)
    vertical = node("vslice", [axis], start=1, stop=2)
    crane_row = node("gmax", [node("mul", [active_board, row])])
    crane_col = node("gmax", [node("mul", [active_board, col])])

    # Flood rightward from the selected UI ring, stopping at another ring.
    ui_seed = node("mul", [selected_map, row58])
    ui_all = node("mul", [
        node("add", [selected_map, unselected_map]), row58
    ])
    ui_barrier = node("relu", [node("sub", [ui_all, ui_seed])])

    shift_w = np.zeros((1, 1, 3, 3), dtype=np.float32)
    shift_w[0, 0, 1, 0] = 1
    shift_name = name("w")
    params[shift_name] = shift_w
    loop_id = name("loop")
    state_id = name("state")
    shifted_id = name("shifted")
    merged_id = name("merged")
    blocked_id = name("blocked")
    next_id = name("next")
    graph.append({
        "id": loop_id,
        "op": "loop",
        "T": 64,
        "state": [{"id": state_id, "zeros": [1, 64, 64]}],
        "body": [
            {"id": shifted_id, "op": "conv", "in": [state_id],
             "w": shift_name},
            {"id": merged_id, "op": "max2",
             "in": [shifted_id, ui_seed]},
            {"id": blocked_id, "op": "sub",
             "in": [merged_id, ui_barrier]},
            {"id": next_id, "op": "relu", "in": [blocked_id]},
        ],
        "next": {state_id: next_id},
    })
    ui_span = f"{loop_id}.{state_id}"

    # The black interior marker distinguishes matched target tiles.
    colors = list(range(6, 16))
    target_w = np.zeros((len(colors), 16, 3, 3), dtype=np.float32)
    for i, c in enumerate(colors):
        target_w[i, c, 1, 1] = 1
        target_w[i, c, 0, 1] = 1
        target_w[i, c, 1, 2] = 1
        target_w[i, 4, 0, 0] = 1
    target = node("relu", [
        conv("x0", target_w, np.full(len(colors), -3, np.float32))
    ])
    target = node("mul", [target, row58])
    target = node("mul", [target, ui_span])

    priority = affine(col, -1, 2)
    weighted_target = node("mul", [target, priority])
    sum_colors = np.ones((1, len(colors), 1, 1), dtype=np.float32)
    target_peak = node("gmax", [conv(weighted_target, sum_colors)])
    target_distance = node("abs", [node("sub", [priority, target_peak])])
    first_gate = node("relu", [affine(target_distance, -100, 1)])
    target_color = node("gmax", [node("mul", [target, first_gate])])
    target_present = node("gmax", [conv(target, sum_colors)])

    # Detect the top-left interior pixel of a colored 4x4 block.
    block_w = np.zeros((len(colors), 16, 5, 5), dtype=np.float32)
    for i, c in enumerate(colors):
        for dy in range(3):
            for dx in range(3):
                block_w[i, c, 2 + dy, 2 + dx] = 1
        block_w[i, c, 1, 2] = -1
        block_w[i, c, 2, 1] = -1
    blocks = node("relu", [
        conv("x0", block_w, np.full(len(colors), -8, np.float32))
    ])
    blocks = node("mul", [blocks, board])
    matching = conv(node("mul", [blocks, target_color]), sum_colors)

    # Prefer a nearby matching block, with perpendicular distance weighted
    # more heavily than distance along the crane's axis.
    dr = node("abs", [affine(node("sub", [row, crane_row]), 63, 1)])
    dc = node("abs", [affine(node("sub", [col, crane_col]), 63, 1)])
    perpendicular = node("add", [
        node("mul", [dr, horizontal]),
        node("mul", [dc, vertical]),
    ])
    axial = node("add", [
        node("mul", [dc, horizontal]),
        node("mul", [dr, vertical]),
    ])
    score = node("add", [
        affine(perpendicular, -4, 100),
        affine(axial, -0.2, 0),
    ])
    score = node("add", [score, affine(row, -0.01, 0)])
    score = node("add", [score, affine(col, -0.02, 0)])
    best_score = node("gmax", [node("mul", [matching, score])])
    score_distance = node("abs", [node("sub", [score, best_score])])
    best_gate = node("relu", [affine(score_distance, -10000, 1)])
    chosen = node("mul", [matching, best_gate])
    block_row = node("gmax", [node("mul", [chosen, row])])
    block_col = node("gmax", [node("mul", [chosen, col])])

    positions = node("vconcat", [
        block_row, crane_row, block_col, crane_col
    ])
    deltas = dense(positions, [
        [63, -63, 0, 0],
        [0, 0, 63, -63],
    ], [1, 1])
    signed = dense(deltas, [
        [-1, 0], [1, 0], [0, -1], [0, 1]
    ])
    signed_pos = node("vrelu", [signed])
    signed_excess = node("vrelu", [
        dense(signed_pos, np.eye(4, dtype=np.float32),
              np.full(4, -1, dtype=np.float32))
    ])
    movement_flags = vadd(
        signed_pos,
        dense(signed_excess, -np.eye(4, dtype=np.float32))
    )
    axis_flags = dense(axis, [
        [1, 0], [1, 0], [0, 1], [0, 1]
    ])
    perpendicular_moves = vmul(movement_flags, axis_flags)
    aligned = vclip01(
        dense(perpendicular_moves, [[-1, -1, -1, -1]], [1])
    )
    extension = dense(orientation, [
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ])
    moves = vadd(perpendicular_moves, vmul(extension, aligned))

    # Board fingerprint for detecting a previous action with no effect.
    color_weights = np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
    color_map = node("mul", [conv("x0", color_weights), board])
    fingerprint_maps = node("concat", [
        color_map,
        node("mul", [color_map, row]),
        node("mul", [color_map, col]),
    ])
    fingerprint = dense(
        node("gmean", [fingerprint_maps]),
        4096 * np.eye(3, dtype=np.float32)
    )
    old_fingerprint = node("vslice", ["mem"], start=0, stop=3)
    difference = vadd(
        fingerprint, dense(old_fingerprint, -np.eye(3, dtype=np.float32))
    )
    positive_diff = node("vrelu", [difference])
    negative_diff = node("vrelu", [
        dense(difference, -np.eye(3, dtype=np.float32))
    ])
    diff_sum = dense(vadd(positive_diff, negative_diff), [[1, 1, 1]])
    unchanged = vclip01(vaffine(diff_sum, -20, 1))
    previous_move = dense("aprev", [[0, 1, 1, 1, 1, 0, 0, 0]])
    stuck = vmul(unchanged, previous_move)

    retreat = dense(orientation, [
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])
    moves = vadd(
        moves,
        dense(vmul(retreat, stuck), 2.5 * np.eye(4, dtype=np.float32))
    )

    # Detect a 6x6 black obstacle using its four outer rows. The middle
    # rows are omitted because a hook can be drawn over them.
    wall_w = np.zeros((1, 16, 7, 7), dtype=np.float32)
    for y in (1, 2, 5, 6):
        wall_w[0, 5, y, 1:7] = 1
    wall_w[0, 4, 0, 3] = 1
    wall_w[0, 4, 2, 0] = 1
    wall_map = node("relu", [conv("x0", wall_w, [-25])])
    wall_present = vclip01(
        node("gmax", [node("mul", [wall_map, board])])
    )

    previous_directions = node("vslice", ["aprev"], start=1, stop=5)
    previous_extension = dense(
        vmul(previous_directions, extension), [[1, 1, 1, 1]]
    )

    # Memory: fingerprint, four detour phases, saved side, side toggle.
    old_phase = node("vslice", ["mem"], start=3, stop=7)
    old_side = node("vslice", ["mem"], start=7, stop=11)
    toggle = node("vslice", ["mem"], start=11, stop=12)
    phase_total = vclip01(dense(old_phase, [[1, 1, 1, 1]]))
    trigger = vclip01(vmul(
        vmul(unchanged, previous_extension),
        vmul(wall_present, vnot(phase_total))
    ))

    side_coordinate = vadd(
        vmul(crane_row, horizontal),
        vmul(crane_col, vertical)
    )
    high_side = vclip01(vaffine(side_coordinate, 63, -31))
    low_side = vnot(high_side)

    # Alternate the detour side after each attempt.
    go_low = vadd(vmul(low_side, vnot(toggle)),
                  vmul(high_side, toggle))
    go_high = vnot(go_low)
    side = node("vconcat", [
        vmul(go_low, horizontal),
        vmul(go_high, horizontal),
        vmul(go_low, vertical),
        vmul(go_high, vertical),
    ])

    phase_extend = dense(old_phase, [[1, 1, 1, 0]])
    phase_return = node("vslice", [old_phase], start=3, stop=4)
    reverse_side = dense(old_side, [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ])
    special_active = vclip01(vadd(trigger, phase_total))
    moves = vadd(
        vmul(moves, vnot(special_active)),
        vadd(
            vmul(side, trigger),
            vadd(
                vmul(extension, phase_extend),
                vmul(reverse_side, phase_return)
            )
        )
    )

    phase_shift = dense(old_phase, [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])
    next_phase = vadd(phase_shift, dense(trigger, [[1], [0], [0], [0]]))
    next_side = vadd(vmul(side, trigger), vmul(old_side, vnot(trigger)))
    next_toggle = vadd(vmul(toggle, vnot(trigger)),
                       vmul(vnot(toggle), trigger))
    memory_value = node("vconcat", [
        fingerprint, next_phase, next_side, next_toggle
    ])
    mem_w = name("w")
    params[mem_w] = np.eye(12, dtype=np.float32)
    graph.append({
        "id": "mem_next", "op": "dense", "in": [memory_value],
        "w": mem_w
    })

    no_target = vclip01(vaffine(target_present, -1, 1))
    inactive_present = node("gmax", [inactive_board])
    switch = vmul(no_target, inactive_present)

    out = node("vconcat", [moves, switch])
    out_w = np.zeros((8, 5), dtype=np.float32)
    for i in range(4):
        out_w[i + 1, i] = 1
    out_w[6, 4] = 4
    out_b = np.array([-2, 0, 0, 0, 0, -2, 0, -2], dtype=np.float32)
    wn = name("w")
    bn = name("b")
    params[wn] = out_w
    params[bn] = out_b
    graph.append({
        "id": "logits", "op": "dense", "in": [out], "w": wn, "b": bn
    })
    graph.append({
        "id": "click", "op": "relu", "in": [inactive_board]
    })

    return {"graph": graph, "memory_size": 12, "params": params}
