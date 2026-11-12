import numpy as np


def build():
    graph, params = [], {}
    counter = [0]

    def fresh(prefix="n"):
        counter[0] += 1
        return "%s_%d" % (prefix, counter[0])

    def parameter(a):
        name = fresh("p")
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, ins, name=None, **fields):
        name = name or fresh(op)
        d = {"id": name, "op": op, "in": list(ins)}
        d.update(fields)
        graph.append(d)
        return name

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], name, **fields)

    def sa(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        if b is not None:
            b = np.broadcast_to(
                np.asarray(b, np.float32), (w.shape[0],)
            ).copy()
        return conv(x, w[:, :, None, None], b, name)

    def va(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        if b is not None:
            b = np.broadcast_to(
                np.asarray(b, np.float32), (w.shape[0],)
            ).copy()
        return dense(x, w, b, name)

    def sl(x, start, stop, vector=False):
        return node(
            "vslice" if vector else "slice",
            [x], start=start, stop=stop
        )

    def cat(xs, vector=False):
        if len(xs) == 1:
            return xs[0]
        return node("vconcat" if vector else "concat", xs)

    def vl(xs, weights, bias=0):
        return va(cat(xs, True), weights, bias)

    def gm(x):
        return node("gmax", [x])

    def sm(a, b):
        return node("mul", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def sadd(xs):
        y = xs[0]
        for x in xs[1:]:
            y = node("add", [y, x])
        return y

    one = sa("coords", [[0, 0, 0]], [1])
    vone = gm(one)

    def sc(x):
        return node("min2", [node("relu", [x]), one])

    def sb(x, scale=1, bias=0):
        return sc(sa(x, [[scale]], [bias]))

    def vb(x, scale=1, bias=0):
        return gm(sb(sm(one, x), scale, bias))

    def inv(x):
        return va(x, [[-1]], [1])

    # Display coordinates, ordered row then column.
    pix = sa("coords", [[63, 0, 0], [0, 63, 0]])
    prow, pcol = sl(pix, 0, 1), sl(pix, 1, 2)
    reverse_pix = sa(
        pix, -np.eye(2, dtype=np.float32), [63, 63]
    )

    # Canonical cell-centre selectors.
    centres = np.arange(21, dtype=np.float32) * 3 + 1
    cw = np.zeros((42, 3), np.float32)
    cw[:21, 0] = 63
    cw[21:, 1] = 63
    distances = node("abs", [
        sa("coords", cw, -np.concatenate([centres, centres]))
    ])
    selectors = sc(sa(
        distances,
        -8 * np.eye(42, dtype=np.float32),
        np.full(42, 2, np.float32)
    ))
    row_selectors = sl(selectors, 0, 21)
    col_selectors = sl(selectors, 21, 42)
    sum21 = np.ones((1, 21), np.float32)
    lattice = sm(
        sa(row_selectors, sum21),
        sa(col_selectors, sum21)
    )

    colour0 = sl("x0", 0, 1)
    colour5 = sl("x0", 5, 6)
    colour9 = sl("x0", 9, 10)
    colour11 = sl("x0", 11, 12)
    target = sm(lattice, colour11)

    def outer_colour(colour):
        w = np.zeros((1, 16, 3, 3), np.float32)
        w[0, colour] = 1
        w[0, colour, 1, 1] = 0
        return sm(lattice, sc(conv("x0", w, [-7])))

    original = outer_colour(5)
    axis_cells = outer_colour(10)
    movable_original = sm(
        original, sa(colour5, [[-1]], [1])
    )

    vertical_line = sb(node("colsum", [axis_cells]), 1, -3)
    horizontal_line = sb(node("rowsum", [axis_cells]), 1, -3)
    vertical_exists = gm(vertical_line)
    horizontal_exists = gm(horizontal_line)
    vertical_position = gm(sm(vertical_line, pcol))
    horizontal_position = gm(sm(horizontal_line, prow))

    vertical_exclusive = sm(
        vertical_line, sa(horizontal_line, [[-1]], [1])
    )
    horizontal_exclusive = sm(
        horizontal_line, sa(vertical_line, [[-1]], [1])
    )
    movable_axis_cells = sm(
        axis_cells, sadd([colour0, colour9])
    )
    selected_axis_cells = sm(axis_cells, colour0)

    vertical_movable = gm(
        sm(vertical_exclusive, movable_axis_cells)
    )
    horizontal_movable = gm(
        sm(horizontal_exclusive, movable_axis_cells)
    )
    vertical_selected = gm(
        sm(vertical_exclusive, selected_axis_cells)
    )
    horizontal_selected = gm(
        sm(horizontal_exclusive, selected_axis_cells)
    )
    any_axis_selected = vb(vl(
        [vertical_selected, horizontal_selected], [1, 1]
    ))

    selection_seed = sm(
        sm(original, colour0), inv(any_axis_selected)
    )

    def bounds(mask):
        maximum = gm(sm(mask, pix))
        minimum = va(
            gm(sm(mask, reverse_pix)),
            -np.eye(2, dtype=np.float32), [63, 63]
        )
        extent = node("vadd", [
            maximum,
            va(minimum, -np.eye(2, dtype=np.float32))
        ])
        return minimum, maximum, extent

    target_min, target_max, _ = bounds(target)
    target_mid = va(
        cat([target_min, target_max], True),
        [[0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5]]
    )
    desired_horizontal = sl(target_mid, 0, 1, True)
    desired_vertical = sl(target_mid, 1, 2, True)

    def directions(current, desired):
        return (
            vb(vl([current, desired], [1, -1]), 10, -16),
            vb(vl([current, desired], [-1, 1]), 10, -16)
        )

    axis_left, axis_right = directions(
        vertical_position, desired_vertical
    )
    axis_up, axis_down = directions(
        horizontal_position, desired_horizontal
    )
    vertical_need = vm(
        vertical_movable,
        vl([axis_left, axis_right], [1, 1])
    )
    horizontal_need_raw = vm(
        horizontal_movable,
        vl([axis_up, axis_down], [1, 1])
    )
    horizontal_need = vm(
        horizontal_need_raw, inv(vertical_need)
    )
    object_phase = vm(
        inv(vertical_need), inv(horizontal_need_raw)
    )

    # Reflection operators parameterized by observed axis positions.
    def reflection_destinations(coordinate, position):
        base = sa(
            coordinate, np.ones((21, 1), np.float32), centres
        )
        distance = node("abs", [
            node("add", [base, va(position, [[-2]])])
        ])
        return sc(sa(
            distance,
            -8 * np.eye(21, dtype=np.float32),
            np.full(21, 2, np.float32)
        ))

    vertical_dest = reflection_destinations(
        pcol, vertical_position
    )
    horizontal_dest = reflection_destinations(
        prow, horizontal_position
    )

    def reflect(x, vertical):
        if vertical:
            sampled = sm(x, col_selectors)
            expanded = node("rowmax", [sampled])
            mapped = sm(expanded, vertical_dest)
            exists = vertical_exists
        else:
            sampled = sm(x, row_selectors)
            expanded = node("colmax", [sampled])
            mapped = sm(expanded, horizontal_dest)
            exists = horizontal_exists
        return sm(sa(mapped, sum21), exists)

    def orbit(x):
        vx = reflect(x, True)
        hx = reflect(x, False)
        hvx = reflect(vx, False)
        return sadd([x, vx, hx, hvx])

    def point_mask(point):
        distance = node("abs", [node("sub", [pix, point])])
        return sm(
            lattice, sb(sa(distance, [[1, 1]]), -8, 2)
        )

    # Shared flood kernel. Flooding is used only when acquiring geometry.
    flood_w = np.zeros((1, 1, 7, 7), np.float32)
    for r, c in [(3, 3), (0, 3), (6, 3), (3, 0), (3, 6)]:
        flood_w[0, 0, r, c] = 1
    flood_param = parameter(flood_w)

    def flood(seed, allowed):
        loop_id = fresh("flood")
        state = fresh("component")
        spread, bounded = fresh("spread"), fresh("bounded")
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 64,
            "state": [{"id": state, "init": seed}],
            "body": [
                {
                    "id": spread, "op": "conv",
                    "in": [state], "w": flood_param
                },
                {
                    "id": bounded, "op": "min2",
                    "in": [spread, allowed]
                }
            ],
            "next": {state: bounded}
        })
        return loop_id + "." + state

    # Memory: 144 native-shape entries, two anchor coordinates, validity.
    memory_shape = sl("mem", 0, 144, True)
    memory_anchor = sl("mem", 144, 146, True)
    memory_valid = sl("mem", 146, 147, True)

    # A dynamic spatial basis for a 12x12 bitmap at an arbitrary anchor.
    basis_coord_w = np.zeros((24, 3), np.float32)
    basis_coord_w[:12, 0] = 63
    basis_coord_w[12:, 1] = 63
    offsets = np.tile(
        np.arange(12, dtype=np.float32) * 3, 2
    )
    basis_coords = sa("coords", basis_coord_w, -offsets)

    anchor_expand_w = np.zeros((24, 2), np.float32)
    anchor_expand_w[:12, 0] = 1
    anchor_expand_w[12:, 1] = 1

    row_expand_w = np.zeros((144, 12), np.float32)
    col_expand_w = np.zeros((144, 12), np.float32)
    for r in range(12):
        for c in range(12):
            row_expand_w[r * 12 + c, r] = 1
            col_expand_w[r * 12 + c, c] = 1

    anchor_expand_param = parameter(anchor_expand_w)
    basis_clip_param = parameter(
        (-8 * np.eye(24, dtype=np.float32))[:, :, None, None]
    )
    basis_clip_bias = parameter(np.full(24, 2, np.float32))
    row_expand_param = parameter(
        row_expand_w[:, :, None, None]
    )
    col_expand_param = parameter(
        col_expand_w[:, :, None, None]
    )
    bitmap_sum_param = parameter(
        np.ones((1, 144, 1, 1), np.float32)
    )

    def bitmap_basis(anchor):
        expanded_anchor = node(
            "dense", [anchor], w=anchor_expand_param
        )
        distance = node("abs", [
            node("sub", [basis_coords, expanded_anchor])
        ])
        raw = node(
            "conv", [distance],
            w=basis_clip_param, b=basis_clip_bias
        )
        flags = sc(raw)
        rows = sl(flags, 0, 12)
        cols = sl(flags, 12, 24)
        row_planes = node(
            "conv", [rows], w=row_expand_param
        )
        col_planes = node(
            "conv", [cols], w=col_expand_param
        )
        return sm(row_planes, col_planes)

    remembered_basis = bitmap_basis(memory_anchor)
    remembered_piece = node(
        "conv",
        [sm(remembered_basis, memory_shape)],
        w=bitmap_sum_param
    )

    # At episode entry, black centres identify the selected native piece.
    # Unselected background-colour native centres are barriers.
    initial_allowed = sm(
        original,
        sc(sadd([
            colour0, colour11, vertical_line, horizontal_line
        ]))
    )
    initial_piece = flood(selection_seed, initial_allowed)
    usable_memory = vm(memory_valid, inv(any_axis_selected))
    selected = sadd([
        sm(remembered_piece, usable_memory),
        sm(initial_piece, inv(usable_memory))
    ])
    selected = sm(selected, inv(any_axis_selected))
    selected_exists = gm(selected)
    anchor, _, extent = bounds(selected)

    other_original = node("relu", [
        node("sub", [original, selected])
    ])
    other_coverage = sc(orbit(other_original))
    uncovered = sm(
        target, sa(other_coverage, [[-1]], [1])
    )

    # Correct orbit multiplicities at mirror fixed points.
    weighted_uncovered = sm(
        sm(uncovered, sa(vertical_line, [[-0.5]], [1])),
        sa(horizontal_line, [[-0.5]], [1])
    )
    desired_map = orbit(weighted_uncovered)

    # Exact monotone scan of the selected native 12x12 kernel.
    probe = point_mask(anchor)
    phase_bias = np.zeros(12, np.float32)
    phase_bias[0] = 1
    phase_init = va(
        vone, np.zeros((12, 1), np.float32), phase_bias
    )

    phase_w = np.zeros((12, 12), np.float32)
    for i in range(12):
        phase_w[(i + 1) % 12, i] = 1
    gate_w = np.zeros((2, 12), np.float32)
    gate_w[0, :11] = 1
    gate_w[1, 11] = 1
    phase_param = parameter(phase_w)
    gate_param = parameter(gate_w)

    def shift_parameter(r, c):
        w = np.zeros((1, 1, 7, 7), np.float32)
        w[0, 0, r, c] = 1
        return parameter(w)

    probe_right_param = shift_parameter(3, 0)
    probe_down_param = shift_parameter(0, 3)
    target_left_param = shift_parameter(3, 6)
    target_up_param = shift_parameter(6, 3)

    scan_id = fresh("correlation")
    p, t, pb, tb, score, phase = [
        fresh("state") for _ in range(6)
    ]
    body = []

    def bn(op, ins, **fields):
        name = fresh("scan")
        d = {"id": name, "op": op, "in": list(ins)}
        d.update(fields)
        body.append(d)
        return name

    hit = bn("mul", [selected, p])
    coefficient = bn("gmax", [hit])
    contribution = bn("mul", [t, coefficient])
    score_next = bn("add", [score, contribution])

    gates = bn("dense", [phase], w=gate_param)
    regular = bn("vslice", [gates], start=0, stop=1)
    wrap = bn("vslice", [gates], start=1, stop=2)

    probe_right = bn("conv", [p], w=probe_right_param)
    probe_down = bn("conv", [pb], w=probe_down_param)
    target_left = bn("conv", [t], w=target_left_param)
    target_up = bn("conv", [tb], w=target_up_param)

    probe_reset = bn("mul", [probe_down, wrap])
    target_reset = bn("mul", [target_up, wrap])
    p_next = bn("add", [
        bn("mul", [probe_right, regular]), probe_reset
    ])
    t_next = bn("add", [
        bn("mul", [target_left, regular]), target_reset
    ])
    pb_next = bn("add", [
        bn("mul", [pb, regular]), probe_reset
    ])
    tb_next = bn("add", [
        bn("mul", [tb, regular]), target_reset
    ])
    phase_next = bn("dense", [phase], w=phase_param)

    graph.append({
        "id": scan_id,
        "op": "loop",
        "T": 144,
        "state": [
            {"id": p, "init": probe},
            {"id": t, "init": desired_map},
            {"id": pb, "init": probe},
            {"id": tb, "init": desired_map},
            {"id": score, "zeros": [1, 64, 64]},
            {"id": phase, "init": phase_init}
        ],
        "body": body,
        "next": {
            p: p_next,
            t: t_next,
            pb: pb_next,
            tb: tb_next,
            score: score_next,
            phase: phase_next
        }
    })
    scores = scan_id + "." + score

    # Legal native bounding-box placements.
    upper_bound = va(
        extent, -np.eye(2, dtype=np.float32), [61.5, 61.5]
    )
    room = node("sub", [sm(one, upper_bound), pix])
    room_flags = sc(sa(
        room, 4 * np.eye(2, dtype=np.float32)
    ))
    legal = sm(
        sm(sl(room_flags, 0, 1), sl(room_flags, 1, 2)),
        lattice
    )
    scores = node("add", [
        scores, sa(legal, [[10000]], [-10000])
    ])
    best_candidates = sb(
        node("sub", [scores, gm(scores)]), 20, 2
    )

    # Tie-break by shortest translation, then row-major order.
    distance = sa(
        node("abs", [node("sub", [pix, anchor])]),
        [[1, 1]]
    )
    near_metric = sm(
        sa(distance, [[-1]], [256]), best_candidates
    )
    nearest = sb(
        node("sub", [near_metric, gm(near_metric)]), 8, 2
    )
    index = sa(pix, [[64, 1]])
    lex_metric = sm(
        sa(index, [[-1]], [4096]), nearest
    )
    destination_mask = sb(
        node("sub", [lex_metric, gm(lex_metric)]), 8, 2
    )
    destination = gm(sm(destination_mask, pix))

    object_up, object_down = directions(
        sl(anchor, 0, 1, True),
        sl(destination, 0, 1, True)
    )
    object_left, object_right = directions(
        sl(anchor, 1, 2, True),
        sl(destination, 1, 2, True)
    )
    object_enabled = vm(
        vm(object_phase, selected_exists),
        inv(any_axis_selected)
    )
    object_up = vm(object_up, object_enabled)
    object_down = vm(object_down, object_enabled)
    object_vertical_move = vl(
        [object_up, object_down], [1, 1]
    )
    horizontal_enabled = vm(
        object_enabled, inv(object_vertical_move)
    )
    object_left = vm(object_left, horizontal_enabled)
    object_right = vm(object_right, horizontal_enabled)

    vertical_axis_move = vm(
        vertical_need, vertical_selected
    )
    horizontal_axis_move = vm(
        horizontal_need, horizontal_selected
    )

    move_up = vl([
        object_up, vm(horizontal_axis_move, axis_up)
    ], [1, 1])
    move_down = vl([
        object_down, vm(horizontal_axis_move, axis_down)
    ], [1, 1])
    move_left = vl([
        object_left, vm(vertical_axis_move, axis_left)
    ], [1, 1])
    move_right = vl([
        object_right, vm(vertical_axis_move, axis_right)
    ], [1, 1])
    moves = cat(
        [move_up, move_down, move_left, move_right], True
    )

    # Circular row-major selection of native pieces.
    other_clickable = node("relu", [
        node("sub", [movable_original, selected])
    ])
    selected_last = gm(sm(selected, index))
    after_selected = sb(
        node("sub", [index, selected_last]), 2, -1
    )
    object_click_values = sm(
        sm(
            other_clickable,
            sadd([
                sa(index, [[-1]], [8192]),
                sa(after_selected, [[8192]])
            ])
        ),
        object_phase
    )

    axis_click_values = sa(index, [[-1]], [8192])
    vertical_click_values = sm(
        sm(
            sm(axis_cells, vertical_exclusive),
            axis_click_values
        ),
        vm(vertical_need, inv(vertical_selected))
    )
    horizontal_click_values = sm(
        sm(
            sm(axis_cells, horizontal_exclusive),
            axis_click_values
        ),
        vm(horizontal_need, inv(horizontal_selected))
    )
    click_values = sadd([
        object_click_values,
        vertical_click_values,
        horizontal_click_values
    ])
    sa(click_values, [[1]], name="click")

    # If no useful native click remains, cycle selection instead of
    # repeatedly clicking an empty pixel.
    click_exists = vb(gm(click_values), 1, 0)
    any_move = va(moves, [[1, 1, 1, 1]])
    cycle = vm(inv(click_exists), inv(any_move))

    logit_input = cat([moves, cycle], True)
    logit_w = np.zeros((8, 5), np.float32)
    logit_w[1:5, :4] = 5 * np.eye(4, dtype=np.float32)
    logit_w[5, 4] = 12
    dense(
        logit_input, logit_w,
        [-10, 0, 0, 0, 0, -10, 1, -10],
        name="logits"
    )

    # Acquire the clicked geometry before the click is executed.
    # The existing selected piece is a barrier, so touching pieces do
    # not become a single component.
    click_winner = sb(
        node("sub", [click_values, gm(click_values)]), 8, 2
    )
    clicked_seed = sm(click_winner, other_original)
    clicked_piece = flood(clicked_seed, other_original)

    click_action = vm(inv(any_move), inv(cycle))
    object_click = vm(
        vm(click_action, object_phase),
        gm(other_clickable)
    )
    next_piece_unmoved = sadd([
        sm(selected, inv(object_click)),
        sm(clicked_piece, object_click)
    ])
    next_anchor_unmoved, _, _ = bounds(next_piece_unmoved)

    # Encode the selected native geometry independently of its position.
    next_basis = bitmap_basis(next_anchor_unmoved)
    next_shape = gm(sm(next_basis, next_piece_unmoved))

    # Only native-piece movement translates the remembered anchor.
    native_displacement = va(
        cat([
            object_up, object_down, object_left, object_right
        ], True),
        [[-3, 3, 0, 0], [0, 0, -3, 3]]
    )
    next_anchor = node("vadd", [
        next_anchor_unmoved, native_displacement
    ])

    next_valid = gm(next_piece_unmoved)
    axis_click = vm(
        click_action,
        vl([vertical_need, horizontal_need], [1, 1])
    )
    next_valid = vm(
        vm(next_valid, inv(axis_click)), inv(cycle)
    )

    memory_output = cat(
        [next_shape, next_anchor, next_valid], True
    )
    # A vector identity node supplies the required output identifier.
    node(
        "vslice", [memory_output],
        name="mem_next", start=0, stop=147
    )

    return {
        "graph": graph,
        "memory_size": 147,
        "params": params
    }
