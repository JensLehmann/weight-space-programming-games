import numpy as np


def build():
    P = {}
    G = []
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "mem": ("v", 104),
        "aprev": ("v", 8),
        "avail": ("v", 8),
    }
    serial = 0

    def par(value):
        key = "p%d" % len(P)
        P[key] = np.asarray(value, dtype=np.float32)
        return key

    def emit(op, inputs, name=None, **fields):
        nonlocal serial
        if name is None:
            name = "n%d" % serial
            serial += 1
        assert name not in shapes
        assert all(isinstance(i, str) and i in shapes for i in inputs)
        ss = [shapes[i] for i in inputs]

        if op == "conv":
            w = P[fields["w"]]
            assert ss[0][0] == "s" and w.shape[1] == ss[0][1]
            result = ("s", w.shape[0])
        elif op == "dense":
            w = P[fields["w"]]
            assert ss[0][0] == "v" and w.shape[1] == ss[0][1]
            result = ("v", w.shape[0])
        elif op in ("relu", "tanh", "sigmoid", "abs",
                    "rowmax", "colmax", "rowsum", "colsum"):
            assert ss[0][0] == "s"
            result = ss[0]
        elif op in ("vrelu", "vtanh", "vsigmoid"):
            assert ss[0][0] == "v"
            result = ss[0]
        elif op in ("gmax", "gmean"):
            assert ss[0][0] == "s"
            result = ("v", ss[0][1])
        elif op in ("add", "sub", "mul", "max2", "min2"):
            assert any(s[0] == "s" for s in ss)
            assert ss[0][1] == ss[1][1] or 1 in (ss[0][1], ss[1][1])
            result = ("s", max(ss[0][1], ss[1][1]))
        elif op in ("vadd", "vmul"):
            assert all(s[0] == "v" for s in ss)
            assert ss[0][1] == ss[1][1] or 1 in (ss[0][1], ss[1][1])
            result = ("v", max(ss[0][1], ss[1][1]))
        elif op == "concat":
            assert all(s[0] == "s" for s in ss)
            result = ("s", sum(s[1] for s in ss))
        elif op == "vconcat":
            assert all(s[0] == "v" for s in ss)
            result = ("v", sum(s[1] for s in ss))
        elif op in ("slice", "vslice"):
            kind = "s" if op == "slice" else "v"
            assert ss[0][0] == kind
            assert 0 <= fields["start"] < fields["stop"] <= ss[0][1]
            result = (kind, fields["stop"] - fields["start"])
        else:
            raise ValueError(op)

        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        G.append(nd)
        shapes[name] = result
        return name

    def conv(x, w, b=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return emit("conv", [x], **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return emit("dense", [x], name=name, **fields)

    def sl(x, a, b):
        return emit("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return emit("vslice", [x], start=a, stop=b)

    def scale(x, a, b=0.0):
        c = shapes[x][1]
        w = (a * np.eye(c, dtype=np.float32))[:, :, None, None]
        return conv(x, w, np.full(c, b, dtype=np.float32))

    def vscale(x, a, b=0.0):
        c = shapes[x][1]
        return dense(
            x,
            a * np.eye(c, dtype=np.float32),
            np.full(c, b, dtype=np.float32),
        )

    def relu(x):
        return emit("relu", [x])

    def vr(x):
        return emit("vrelu", [x])

    def add(a, b):
        return emit("add", [a, b])

    def sub(a, b):
        return emit("sub", [a, b])

    def mul(a, b):
        return emit("mul", [a, b])

    def va(a, b):
        return emit("vadd", [a, b])

    def vm(a, b):
        return emit("vmul", [a, b])

    def gm(x):
        return emit("gmax", [x])

    def ab(x):
        return emit("abs", [x])

    def clip(x, one):
        return emit("min2", [relu(x), one])

    def vclip(x):
        return va(vr(x), vscale(vr(vscale(x, 1.0, -1.0)), -1.0))

    def vabs(x):
        return va(vr(x), vr(vscale(x, -1.0)))

    def eqzero(x):
        return relu(scale(x, -2.0, 1.0))

    def within(d, radius, one):
        return clip(add(scale(d, -1.0, 1.0), radius), one)

    def vmax_scalar(a, b):
        return va(b, vr(va(a, vscale(b, -1.0))))

    colours = list(range(6, 15))
    nc = len(colours)
    one = sl("coords", 2, 3)
    yy = scale(sl("coords", 0, 1), 63.0)
    xx = scale(sl("coords", 1, 2), 63.0)
    marker = sl("x0", 0, 1)

    # Detect target centres from the colour-4 border and coloured centre.
    tw = np.zeros((nc, 16, 3, 3), dtype=np.float32)
    for i, c in enumerate(colours):
        tw[i, c, 1, 1] = 10.0
        tw[i, 4, :, :] = 1.0
        tw[i, 4, 1, 1] = 0.0
    targets = clip(
        scale(relu(conv("x0", tw, [-14.5] * nc)), 2.0), one
    )
    target_mask = clip(
        conv(targets, np.ones((1, nc, 1, 1), dtype=np.float32)), one
    )
    target_colour = conv(
        targets,
        np.asarray(colours, dtype=np.float32)[None, :, None, None],
    )

    # Collect at most 32 visible targets as (row, column, colour) triples.
    priority = add(add(scale(yy, 64.0), xx), one)
    shapes["goal_state"] = ("s", 1)
    shapes["stack_state"] = ("v", 96)
    outer = G
    G = []

    score = mul("goal_state", priority)
    best = gm(score)
    chosen = mul(
        relu(scale(sub(score, best), 2.0, 1.0)),
        "goal_state",
    )
    token = emit("vconcat", [
        gm(mul(chosen, yy)),
        gm(mul(chosen, xx)),
        gm(mul(chosen, target_colour)),
    ])
    shift = np.zeros((96, 96), dtype=np.float32)
    shift[3:, :-3] = np.eye(93, dtype=np.float32)
    insert = np.zeros((96, 3), dtype=np.float32)
    insert[:3, :] = np.eye(3, dtype=np.float32)
    next_stack = va(dense("stack_state", shift), dense(token, insert))
    next_goals = relu(sub("goal_state", chosen))
    body = G
    G = outer
    G.append({
        "id": "collect",
        "op": "loop",
        "T": 32,
        "state": [
            {"id": "goal_state", "init": target_mask},
            {"id": "stack_state", "zeros": [96]},
        ],
        "body": body,
        "next": {
            "goal_state": next_goals,
            "stack_state": next_stack,
        },
    })
    shapes["collect.goal_state"] = ("s", 1)
    shapes["collect.stack_state"] = ("v", 96)

    old_targets = vsl("mem", 0, 96)
    flag = vsl("mem", 96, 97)
    remembered = va(
        vm(old_targets, flag),
        vm("collect.stack_state", vscale(flag, -1.0, 1.0)),
    )

    sy = np.zeros((32, 96), dtype=np.float32)
    sx = np.zeros((32, 96), dtype=np.float32)
    sc = np.zeros((32, 96), dtype=np.float32)
    for i in range(32):
        sy[i, 3 * i] = 1.0
        sx[i, 3 * i + 1] = 1.0
        sc[i, 3 * i + 2] = 1.0
    goal_y = dense(remembered, sy)
    goal_x = dense(remembered, sx)
    goal_c = dense(remembered, sc)

    centre_y = gm(mul(marker, yy))
    centre_x = gm(mul(marker, xx))
    cy_dist = ab(sub(yy, centre_y))
    cx_dist = ab(sub(xx, centre_x))
    centre_dist = add(cy_dist, cx_dist)

    # Estimate the selected shape's colour from nearby non-target pixels.
    boxes = clip(
        conv(target_mask, np.ones((1, 1, 3, 3), dtype=np.float32)),
        one,
    )
    not_box = relu(scale(boxes, -1.0, 1.0))
    near = relu(scale(centre_dist, -1.0, 64.0))
    eligible = np.zeros(16, dtype=np.float32)
    for c in colours:
        eligible[c] = 1.0 + 0.001 * c
    eligible_planes = conv(
        "x0", np.diag(eligible)[:, :, None, None]
    )
    nearby_colours = mul(mul(eligible_planes, near), not_box)
    colour_scores = gm(nearby_colours)
    colour_map = conv(
        nearby_colours, np.ones((1, 16, 1, 1), dtype=np.float32)
    )
    best_colour = gm(colour_map)
    colour_difference = va(
        colour_scores, vscale(best_colour, -1.0, 0.002)
    )
    colour_choice = vclip(vscale(vr(colour_difference), 1000.0))
    selected_colour = dense(
        colour_choice, np.arange(16, dtype=np.float32)[None, :]
    )
    piece = mul(
        conv(
            mul("x0", colour_choice),
            np.ones((1, 16, 1, 1), dtype=np.float32),
        ),
        not_box,
    )

    # Classify local geometry around the black centre.
    lw = np.zeros((3, 1, 3, 3), dtype=np.float32)
    lw[0, 0, 1, 0] = lw[0, 0, 1, 2] = 1.0
    lw[1, 0, 0, 1] = lw[1, 0, 2, 1] = 1.0
    for i, j in ((0, 0), (0, 2), (2, 0), (2, 2)):
        lw[2, 0, i, j] = 1.0
    local = gm(mul(conv(piece, lw), marker))
    horizontal = vsl(local, 0, 1)
    vertical = vsl(local, 1, 2)
    diagonal = vsl(local, 2, 3)
    x_type = vclip(va(
        vscale(diagonal, 0.5),
        vscale(va(horizontal, vertical), -0.5),
    ))
    hollow_type = vclip(vscale(
        va(va(horizontal, vertical), diagonal), -1.0, 1.0
    ))
    h_type = vclip(va(horizontal, vscale(vertical, -1.0)))
    h_type = vm(
        vm(h_type, vscale(x_type, -1.0, 1.0)),
        vscale(hollow_type, -1.0, 1.0),
    )
    cross_type = vm(
        vm(
            vscale(x_type, -1.0, 1.0),
            vscale(hollow_type, -1.0, 1.0),
        ),
        vscale(h_type, -1.0, 1.0),
    )

    close = clip(
        scale(emit("max2", [cy_dist, cx_dist]), -1.0, 25.0),
        one,
    )
    local_piece = mul(piece, close)
    same_row = eqzero(cy_dist)
    same_col = eqzero(cx_dist)
    same_diag = eqzero(ab(sub(cy_dist, cx_dist)))
    rx = gm(mul(mul(local_piece, same_row), cx_dist))
    ry = gm(mul(mul(local_piece, same_col), cy_dist))
    rd = gm(mul(mul(local_piece, same_diag), cx_dist))
    hollow_radius = vscale(gm(mul(local_piece, near)), -1.0, 64.0)

    corner = mul(
        eqzero(ab(sub(cx_dist, rx))),
        eqzero(ab(sub(cy_dist, ry))),
    )
    square_type = vm(vclip(gm(mul(local_piece, corner))), hollow_type)
    diamond_type = vm(hollow_type, vscale(square_type, -1.0, 1.0))

    # Geometry channels evaluate all remembered targets at candidate centres.
    dy = ab(sub(yy, goal_y))
    dx = ab(sub(xx, goal_x))
    row_match = mul(eqzero(dy), within(dx, rx, one))
    col_match = mul(eqzero(dx), within(dy, ry, one))
    cross_match = emit("max2", [row_match, col_match])
    diag_match = mul(eqzero(ab(sub(dy, dx))), within(dx, rd, one))
    diamond_match = eqzero(ab(sub(add(dy, dx), hollow_radius)))
    square_x = mul(eqzero(ab(sub(dx, rx))), within(dy, ry, one))
    square_y = mul(eqzero(ab(sub(dy, ry))), within(dx, rx, one))
    square_match = emit("max2", [square_x, square_y])
    geometry = add(
        add(mul(row_match, h_type), mul(cross_match, cross_type)),
        add(
            mul(diag_match, x_type),
            add(
                mul(diamond_match, diamond_type),
                mul(square_match, square_type),
            ),
        ),
    )

    # Discount targets covered by other shapes, but retain those covered by
    # the currently selected shape.
    centres = mul(eqzero(dy), eqzero(dx))
    bw = np.ones((1, 1, 3, 3), dtype=np.float32)
    bw[0, 0, 1, 1] = 0.0
    border = conv(sl("x0", 4, 5), bw)
    border_at_target = gm(mul(centres, border))
    uncovered = vclip(vscale(border_at_target, 1.0, -7.0))

    colour_delta = va(
        goal_c,
        dense(selected_colour, -np.ones((32, 1), dtype=np.float32)),
    )
    same_colour = vclip(vscale(vabs(colour_delta), -2.0, 1.0))
    covered_by_selected = vm(gm(mul(geometry, marker)), same_colour)
    useful = vclip(va(uncovered, covered_by_selected))

    # Compute residues modulo three, restricting candidate positions to the
    # selected centre's movement lattice.
    mw = np.zeros((128, 3, 1, 1), dtype=np.float32)
    mb = np.zeros(128, dtype=np.float32)
    for j in range(64):
        mw[j, 0, 0, 0] = 63.0
        mb[j] = -float(j)
        mw[64 + j, 1, 0, 0] = 63.0
        mb[64 + j] = -float(j)
    hinges = relu(conv("coords", mw, mb))
    cw = np.zeros((6, 128, 1, 1), dtype=np.float32)
    cb = np.zeros(6, dtype=np.float32)
    for axis in range(2):
        for r in range(3):
            out = 3 * axis + r
            vals = np.asarray(
                [1.0 if j % 3 == r else 0.0 for j in range(64)],
                dtype=np.float32,
            )
            cb[out] = vals[0]
            cw[out, 64 * axis, 0, 0] = vals[1] - vals[0]
            for j in range(1, 63):
                cw[out, 64 * axis + j, 0, 0] = (
                    vals[j + 1] - 2.0 * vals[j] + vals[j - 1]
                )
    residues = clip(scale(conv(hinges, cw, cb), 10.0, -5.0), one)
    selected_residues = gm(mul(residues, marker))
    residue_maps = mul(residues, selected_residues)
    rw = np.zeros((2, 6, 1, 1), dtype=np.float32)
    rw[0, :3, 0, 0] = 1.0
    rw[1, 3:, 0, 0] = 1.0
    axes = conv(residue_maps, rw)
    reachable = mul(sl(axes, 0, 1), sl(axes, 1, 2))

    # Detect palette interiors.
    pw = np.zeros((nc, 16, 3, 3), dtype=np.float32)
    for i, c in enumerate(colours):
        pw[i, c, :, :] = 1.0
    palettes = clip(
        scale(relu(conv("x0", pw, [-8.5] * nc)), 2.0),
        one,
    )
    palette_exists = gm(palettes)

    # Score placements for each target colour.
    group_masks = []
    for c in colours:
        delta = vscale(goal_c, 1.0, -float(c))
        equal = vclip(vscale(vabs(delta), -2.0, 1.0))
        group_masks.append(vm(equal, useful))
    all_masks = emit("vconcat", group_masks)
    repeated = emit("concat", [geometry] * nc)
    grouped = mul(repeated, all_masks)
    sum_w = np.zeros((nc, 32 * nc, 1, 1), dtype=np.float32)
    for i in range(nc):
        sum_w[i, 32 * i:32 * (i + 1), 0, 0] = 1.0
    group_votes = conv(grouped, sum_w)
    group_scores = gm(mul(group_votes, reachable))

    current_w = np.zeros((nc, 16), dtype=np.float32)
    for i, c in enumerate(colours):
        current_w[i, c] = 1.0
    current_group = dense(colour_choice, current_w)
    valid_group = vclip(va(current_group, palette_exists))
    positive = vclip(vscale(group_scores, 2.0))
    current_bonus = vm(current_group, positive)
    adjusted = va(
        vm(group_scores, valid_group),
        vscale(current_bonus, 0.5),
    )
    tie_bias = np.asarray(
        [0.0001 * (nc - i) for i in range(nc)],
        dtype=np.float32,
    )
    adjusted = dense(adjusted, np.eye(nc, dtype=np.float32), tie_bias)
    maximum = vsl(adjusted, 0, 1)
    for i in range(1, nc):
        maximum = vmax_scalar(maximum, vsl(adjusted, i, i + 1))
    group_difference = va(
        adjusted,
        dense(
            maximum,
            -np.ones((nc, 1), dtype=np.float32),
            0.00005 * np.ones(nc, dtype=np.float32),
        ),
    )
    chosen_group = vclip(vscale(vr(group_difference), 100000.0))
    chosen_current = dense(
        vm(chosen_group, current_group),
        np.ones((1, nc), dtype=np.float32),
    )
    chosen_score = dense(
        vm(chosen_group, group_scores),
        np.ones((1, nc), dtype=np.float32),
    )
    recolour = vm(
        vscale(chosen_current, -1.0, 1.0),
        vclip(vscale(chosen_score, 2.0)),
    )
    votes = conv(
        mul(group_votes, chosen_group),
        np.ones((1, nc, 1, 1), dtype=np.float32),
    )

    # Find the nearest palette cell of the selected target colour.
    chosen_palette = conv(
        mul(palettes, chosen_group),
        np.ones((1, nc, 1, 1), dtype=np.float32),
    )
    palette_nearness = mul(chosen_palette, near)
    palette_best = gm(palette_nearness)
    palette_selected = mul(
        relu(scale(sub(palette_nearness, palette_best), 2.0, 1.0)),
        chosen_palette,
    )
    palette_priority = mul(
        palette_selected,
        add(scale(yy, -64.0, 4097.0), scale(xx, -1.0)),
    )
    palette_priority_best = gm(palette_priority)
    palette_selected = mul(
        relu(scale(
            sub(palette_priority, palette_priority_best), 2.0, 1.0
        )),
        palette_selected,
    )
    palette_y = gm(mul(palette_selected, yy))
    palette_x = gm(mul(palette_selected, xx))
    palette_dist = add(ab(sub(yy, palette_y)), ab(sub(xx, palette_x)))

    proximity = scale(centre_dist, -0.001, 0.128)
    placement_score = add(votes, proximity)

    # For a hollow shape, aim its outline rather than its empty centre at
    # the palette.
    hollow_palette_dist = ab(sub(palette_dist, hollow_radius))
    palette_cost = add(
        mul(palette_dist, vscale(hollow_type, -1.0, 1.0)),
        mul(hollow_palette_dist, hollow_type),
    )
    palette_score = add(scale(palette_cost, -0.01, 2.0), proximity)
    position_score = mul(
        add(
            mul(placement_score, vscale(recolour, -1.0, 1.0)),
            mul(palette_score, recolour),
        ),
        reachable,
    )

    # Decode a deterministic best destination into two scalar coordinates.
    best_position = gm(position_score)
    best_mask = mul(
        clip(scale(sub(position_score, best_position), 1000.0, 1.0), one),
        reachable,
    )
    position_priority = mul(
        best_mask,
        add(scale(yy, -64.0, 4097.0), scale(xx, -1.0)),
    )
    best_priority = gm(position_priority)
    destination_mask = mul(
        relu(scale(sub(position_priority, best_priority), 2.0, 1.0)),
        best_mask,
    )
    new_y = gm(mul(destination_mask, yy))
    new_x = gm(mul(destination_mask, xx))

    # Keep the destination until it is reached or the selected colour changes.
    old_y = vsl("mem", 97, 98)
    old_x = vsl("mem", 98, 99)
    old_active = vsl("mem", 99, 100)
    old_colour = vsl("mem", 100, 101)
    colour_changed = vabs(va(selected_colour, vscale(old_colour, -1.0)))
    colour_unchanged = vclip(vscale(colour_changed, -2.0, 1.0))
    keep = vm(old_active, colour_unchanged)
    replace = vscale(keep, -1.0, 1.0)
    dest_y = va(vm(old_y, keep), vm(new_y, replace))
    dest_x = va(vm(old_x, keep), vm(new_x, replace))

    delta_y = va(dest_y, vscale(centre_y, -1.0))
    delta_x = va(dest_x, vscale(centre_x, -1.0))
    distance = va(vabs(delta_y), vabs(delta_x))
    reached = vclip(vscale(distance, -2.0, 1.0))

    # If two consecutive movement actions did not move the centre, cycle
    # the selection instead of repeatedly pressing a blocked direction.
    prev_y = vsl("mem", 101, 102)
    prev_x = vsl("mem", 102, 103)
    old_stuck = vsl("mem", 103, 104)
    displacement = va(
        vabs(va(centre_y, vscale(prev_y, -1.0))),
        vabs(va(centre_x, vscale(prev_x, -1.0))),
    )
    unchanged = vclip(vscale(displacement, -2.0, 1.0))
    prev_move_w = np.zeros((1, 8), dtype=np.float32)
    prev_move_w[0, 1:5] = 1.0
    previous_was_move = dense("aprev", prev_move_w)
    stuck = vm(
        vm(unchanged, previous_was_move),
        vscale(old_stuck, 1.0, 1.0),
    )
    blocked = vclip(vscale(stuck, 1.0, -1.0))
    done = vclip(va(reached, blocked))
    moving = vscale(done, -1.0, 1.0)

    action_features = emit("vconcat", [
        vm(vclip(vr(vscale(delta_y, -1.0))), moving),
        vm(vclip(vr(delta_y)), moving),
        vm(vclip(vr(vscale(delta_x, -1.0))), moving),
        vm(vclip(vr(delta_x)), moving),
        done,
    ])
    action_w = np.zeros((8, 5), dtype=np.float32)
    for i in range(5):
        action_w[i + 1, i] = 1.0
    action_b = np.full(8, -100.0, dtype=np.float32)
    action_b[1:6] = 0.0
    dense(action_features, action_w, action_b, name="logits")

    emit("vconcat", [
        remembered,
        vscale(flag, 0.0, 1.0),
        dest_y,
        dest_x,
        moving,
        selected_colour,
        centre_y,
        centre_x,
        vm(stuck, moving),
    ], name="mem_next")

    emit("mul", [one, vscale(flag, 0.0)], name="click")
    return {"graph": G, "memory_size": 104, "params": P}
