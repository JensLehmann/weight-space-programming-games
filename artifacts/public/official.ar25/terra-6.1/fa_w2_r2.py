import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]
    kernels = {}

    def uid():
        counter[0] += 1
        return "n%d" % counter[0]

    def param(a):
        name = uid()
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs=None, **fields):
        name = uid()
        d = {"id": name, "op": op}
        if inputs is not None:
            d["in"] = list(inputs)
        d.update(fields)
        graph.append(d)
        return name

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def conv(x, w, b=None):
        d = {"w": param(w)}
        if b is not None:
            d["b"] = param(b)
        return node("conv", [x], **d)

    def dense(x, w, b=None):
        d = {"w": param(w)}
        if b is not None:
            d["b"] = param(b)
        return node("dense", [x], **d)

    def saff(x, weights, bias=0.0):
        return conv(
            x,
            np.asarray(weights, np.float32).reshape(1, -1, 1, 1),
            [bias],
        )

    def vaff(x, weights, bias=0.0):
        return dense(
            x,
            np.asarray(weights, np.float32).reshape(1, -1),
            [bias],
        )

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def vadd(a, b):
        return node("vadd", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def gmax(a):
        return node("gmax", [a])

    def vnot(a):
        return vaff(a, [-1.0], 1.0)

    def vmax2(a, b):
        d = vaff(vcat(a, b), [1.0, -1.0])
        return vadd(b, node("vrelu", [d]))

    def vclip(a):
        r = node("vrelu", [a])
        e = node("vrelu", [vaff(r, [1.0], -1.0)])
        return vaff(vcat(r, e), [1.0, -1.0])

    def sclip(a):
        r = node("relu", [a])
        e = node("relu", [saff(r, [1.0], -1.0)])
        return sub(r, e)

    def shift(x, channels, dx, dy):
        key = (channels, dx, dy)
        if key not in kernels:
            w = np.zeros((channels, channels, 7, 7), np.float32)
            for i in range(channels):
                w[i, i, 3 - 3 * dy, 3 - 3 * dx] = 1.0
            kernels[key] = param(w)
        return node("conv", [x], w=kernels[key])

    def loop(T, states, body_fn):
        nonlocal graph
        outer = graph
        graph = []
        nxt = body_fn()
        body = graph
        graph = outer
        lid = uid()
        spec = []
        for sid, init in states:
            if isinstance(init, str):
                spec.append({"id": sid, "init": init})
            else:
                spec.append({"id": sid, "zeros": init})
        graph.append({
            "id": lid,
            "op": "loop",
            "T": T,
            "state": spec,
            "body": body,
            "next": nxt,
        })
        return {sid: lid + "." + sid for sid, _ in states}

    # Centers of the canonical 21-by-21 board cells.
    w = np.zeros((42, 3, 1, 1), np.float32)
    b = np.zeros(42, np.float32)
    for i in range(21):
        w[i, 0, 0, 0] = 63.0
        b[i] = -(1 + 3 * i)
        w[21 + i, 1, 0, 0] = 63.0
        b[21 + i] = -(1 + 3 * i)
    dist = node("abs", [conv("coords", w, b)])

    w = np.zeros((42, 42, 1, 1), np.float32)
    for i in range(42):
        w[i, i, 0, 0] = -100.0
    near = conv(dist, w, np.full(42, 2.0, np.float32))
    near1 = node("relu", [near])
    w = np.zeros((42, 42, 1, 1), np.float32)
    for i in range(42):
        w[i, i, 0, 0] = 1.0
    near2 = node("relu", [
        conv(near, w, np.full(42, -1.0, np.float32))
    ])
    exact = sub(near1, near2)

    w = np.zeros((2, 42, 1, 1), np.float32)
    w[0, :21, 0, 0] = 1.0
    w[1, 21:, 0, 0] = 1.0
    rc = conv(exact, w)
    board = mul(sl(rc, 0, 1), sl(rc, 1, 2))

    cy = sl("coords", 0, 1)
    cx = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    c0 = sl("x0", 0, 1)
    c5 = sl("x0", 5, 6)
    c9 = sl("x0", 9, 10)
    c10 = sl("x0", 10, 11)
    c11 = sl("x0", 11, 12)

    def corner(color):
        w = np.zeros((1, 1, 3, 3), np.float32)
        w[0, 0, 0, 0] = 1.0
        return conv(color, w)

    black = mul(board, corner(c5))
    goal = mul(board, c11)
    corner10 = corner(c10)

    vaxis = mul(
        board,
        sclip(saff(node("colsum", [c10]), [1.0], -20.0)),
    )
    haxis = mul(
        board,
        sclip(saff(node("rowsum", [c10]), [1.0], -20.0)),
    )
    line_center = add(c0, c9)
    vclick = mul(
        mul(mul(vaxis, sub(one, haxis)), corner10), line_center
    )
    hclick = mul(
        mul(mul(haxis, sub(one, vaxis)), corner10), line_center
    )
    vselected = gmax(mul(vclick, c0))
    hselected = gmax(mul(hclick, c0))
    vmovable = gmax(vclick)
    hmovable = gmax(hclick)

    def coord_max(mask, coord):
        return gmax(mul(mask, coord))

    def coord_min(mask, coord):
        return vnot(gmax(mul(mask, sub(one, coord))))

    target_x = vaff(
        vcat(coord_min(goal, cx), coord_max(goal, cx)), [0.5, 0.5]
    )
    target_y = vaff(
        vcat(coord_min(goal, cy), coord_max(goal, cy)), [0.5, 0.5]
    )
    axis_x = coord_max(vaxis, cx)
    axis_y = coord_max(haxis, cy)

    def axis_needs(target, pos, movable):
        p = node("vrelu", [
            vaff(vcat(target, pos), [63.0, -63.0], -1.51)
        ])
        m = node("vrelu", [
            vaff(vcat(target, pos), [-63.0, 63.0], -1.51)
        ])
        p = vmul(vclip(p), movable)
        m = vmul(vclip(m), movable)
        return p, m, vclip(vadd(p, m))

    vp, vm, vf = axis_needs(target_x, axis_x, vmovable)
    hp, hm, hf0 = axis_needs(target_y, axis_y, hmovable)
    hf = vmul(hf0, vnot(vf))
    shape_mode = vmul(vnot(vf), vnot(hf))

    def reflect(src, axis, horizontal, weighted=False):
        pair = cat(src, axis)
        ps, ms, out = uid(), uid(), uid()

        def body():
            if horizontal:
                pn = shift(ps, 2, 1, 0)
                mn = shift(ms, 2, -1, 0)
            else:
                pn = shift(ps, 2, 0, 1)
                mn = shift(ms, 2, 0, -1)
            pg, pa = sl(pn, 0, 1), sl(pn, 1, 2)
            mg, ma = sl(mn, 0, 1), sl(mn, 1, 2)
            reduction = "rowmax" if horizontal else "colmax"
            pr = mul(node(reduction, [mul(pg, axis)]), pa)
            mr = mul(node(reduction, [mul(mg, axis)]), ma)
            if weighted:
                new_out = add(add(out, pr), mr)
            else:
                new_out = node("max2", [node("max2", [out, pr]), mr])
            return {ps: pn, ms: mn, out: new_out}

        res = loop(20, [(ps, pair), (ms, pair), (out, src)], body)
        return res[out]

    # Capture a newly selected piece.
    axis_selected = vclip(vadd(vselected, hselected))
    seed = mul(mul(black, c0), vnot(axis_selected))
    eligible = mul(black, sclip(add(c0, c11)))
    fs = uid()

    def flood_body():
        a = shift(fs, 1, -1, 0)
        b = shift(fs, 1, 1, 0)
        c = shift(fs, 1, 0, -1)
        d = shift(fs, 1, 0, 1)
        z = node("max2", [a, b])
        z = node("max2", [z, c])
        z = node("max2", [z, d])
        z = node("max2", [z, fs])
        return {fs: mul(z, eligible)}

    raw_piece = loop(40, [(fs, seed)], flood_body)[fs]
    raw_exists = gmax(raw_piece)
    raw_x = coord_min(raw_piece, cx)
    raw_y = coord_min(raw_piece, cy)

    def coordinate_equal(coord, value):
        d = node("abs", [saff(coord, [1.0], -value)])
        return node("relu", [saff(d, [-100.0], 1.0)])

    left_edge = mul(board, coordinate_equal(cx, 1.0 / 63.0))
    top_edge = mul(board, coordinate_equal(cy, 1.0 / 63.0))
    ns = uid()

    def normalize_body():
        at_left = gmax(mul(ns, left_edge))
        x = add(
            mul(ns, at_left),
            mul(shift(ns, 1, -1, 0), vnot(at_left)),
        )
        at_top = gmax(mul(x, top_edge))
        y = add(
            mul(x, at_top),
            mul(shift(x, 1, 0, -1), vnot(at_top)),
        )
        return {ns: y}

    raw_norm = loop(20, [(ns, raw_piece)], normalize_body)[ns]
    fixed_rows = sl(exact, 0, 10)
    fixed_cols = sl(exact, 21, 31)

    def outer_masks(rows, cols):
        wrow = np.zeros((100, 10, 1, 1), np.float32)
        wcol = np.zeros((100, 10, 1, 1), np.float32)
        for y in range(10):
            for x in range(10):
                k = 10 * y + x
                wrow[k, y, 0, 0] = 1.0
                wcol[k, x, 0, 0] = 1.0
        return mul(conv(rows, wrow), conv(cols, wcol))

    fixed_masks = outer_masks(fixed_rows, fixed_cols)
    raw_bits = gmax(mul(fixed_masks, raw_norm))

    old_bits = vsl("mem", 0, 100)
    old_x = vsl("mem", 100, 101)
    old_y = vsl("mem", 101, 102)
    old_dest_x = vsl("mem", 102, 103)
    old_dest_y = vsl("mem", 103, 104)
    old_active = vsl("mem", 104, 105)
    prev_click = vsl("aprev", 6, 7)

    take = vmul(vmax2(prev_click, vnot(old_active)), raw_exists)
    keep = vmul(old_active, vnot(prev_click))
    active = vclip(vadd(take, keep))
    bits = vadd(vmul(raw_bits, take), vmul(old_bits, keep))

    delta_w = np.zeros((2, 8), np.float32)
    delta_w[0, 3] = -3.0 / 63.0
    delta_w[0, 4] = 3.0 / 63.0
    delta_w[1, 1] = -3.0 / 63.0
    delta_w[1, 2] = 3.0 / 63.0
    delta = dense("aprev", delta_w)
    dx_prev = vsl(delta, 0, 1)
    dy_prev = vsl(delta, 1, 2)
    anchor_x = vadd(vmul(raw_x, take), vmul(vadd(old_x, dx_prev), keep))
    anchor_y = vadd(vmul(raw_y, take), vmul(vadd(old_y, dy_prev), keep))

    def dynamic_axis(coord, anchor):
        w = np.full((10, 1, 1, 1), 63.0, np.float32)
        b = np.array([-3.0 * i for i in range(10)], np.float32)
        values = conv(coord, w, b)
        anchor10 = dense(
            anchor, np.full((10, 1), 63.0, np.float32)
        )
        d = node("abs", [sub(values, anchor10)])
        w = np.zeros((10, 10, 1, 1), np.float32)
        for i in range(10):
            w[i, i, 0, 0] = -1.0
        return node("relu", [conv(d, w, np.ones(10, np.float32))])

    dyn_rows = dynamic_axis(cy, anchor_y)
    dyn_cols = dynamic_axis(cx, anchor_x)
    dyn_masks = outer_masks(dyn_rows, dyn_cols)
    sum100 = np.ones((1, 100, 1, 1), np.float32)
    selected = conv(mul(dyn_masks, bits), sum100)
    normalized = conv(mul(fixed_masks, bits), sum100)

    other = node("relu", [sub(black, selected)])
    other_exists = gmax(other)

    # Remove targets already covered by the other source pieces.
    other_images = reflect(other, vaxis, True)
    other_images = reflect(other_images, haxis, False)
    residual_goal = mul(goal, sub(one, sclip(other_images)))

    # Each source location receives the number of remaining target cells
    # in its reflection orbit.
    reflected = reflect(residual_goal, vaxis, True, weighted=True)
    reflected = reflect(reflected, haxis, False, weighted=True)

    norm_max_x = coord_max(normalized, cx)
    norm_max_y = coord_max(normalized, cy)

    pair0 = cat(normalized, reflected)
    origin = mul(left_edge, top_edge)
    ps, bs, accum, count = uid(), uid(), uid(), uid()

    def correlation_body():
        piece = sl(ps, 0, 1)
        targets = sl(ps, 1, 2)
        bit = gmax(mul(piece, origin))
        new_accum = add(accum, mul(targets, bit))

        row_end = node("vrelu", [vaff(count, [1.0], -9.0)])
        not_end = vnot(row_end)
        along = shift(ps, 2, -1, 0)
        next_row = shift(bs, 2, 0, -1)
        new_pair = add(mul(along, not_end), mul(next_row, row_end))
        new_base = add(mul(bs, not_end), mul(next_row, row_end))
        new_count = vaff(vcat(count, row_end), [1.0, -10.0], 1.0)
        return {ps: new_pair, bs: new_base, accum: new_accum, count: new_count}

    correlation = loop(
        100,
        [
            (ps, pair0),
            (bs, pair0),
            (accum, [1, 64, 64]),
            (count, [1]),
        ],
        correlation_body,
    )[accum]

    dx = node("abs", [sub(cx, anchor_x)])
    dy = node("abs", [sub(cy, anchor_y)])
    score = node("relu", [
        saff(cat(correlation, dy, dx), [1000.0, -63.0, -63.0])
    ])
    valid_x = sclip(saff(add(cx, norm_max_x), [-63.0], 63.0))
    valid_y = sclip(saff(add(cy, norm_max_y), [-63.0], 63.0))
    valid = mul(mul(board, valid_x), valid_y)

    # With multiple pieces, avoid counting both sides of a source piece
    # as independent matches to the same mirror orbit.
    left_of_axis = sclip(
        saff(sub(sub(axis_x, cx), norm_max_x), [63.0], 2.0)
    )
    right_of_axis = sclip(saff(sub(cx, axis_x), [63.0], 1.0))
    above_axis = sclip(
        saff(sub(sub(axis_y, cy), norm_max_y), [63.0], 2.0)
    )
    below_axis = sclip(saff(sub(cy, axis_y), [63.0], 1.0))
    x_one_side = sclip(add(left_of_axis, right_of_axis))
    y_one_side = sclip(add(above_axis, below_axis))
    constrain_x = vmul(other_exists, gmax(vaxis))
    constrain_y = vmul(other_exists, gmax(haxis))
    x_allowed = add(mul(x_one_side, constrain_x), vnot(constrain_x))
    y_allowed = add(mul(y_one_side, constrain_y), vnot(constrain_y))
    valid = mul(mul(valid, x_allowed), y_allowed)
    score = mul(score, valid)

    best_score = gmax(score)
    score_gap = node("abs", [sub(score, best_score)])
    winners = mul(
        valid, node("relu", [saff(score_gap, [-1.0], 1.0)])
    )
    proposed_y = coord_min(winners, cy)
    y_gap = node("abs", [sub(cy, proposed_y)])
    winning_row = node("relu", [saff(y_gap, [-100.0], 1.0)])
    proposed_x = coord_min(mul(winners, winning_row), cx)

    dest_x = vadd(vmul(proposed_x, take), vmul(old_dest_x, keep))
    dest_y = vadd(vmul(proposed_y, take), vmul(old_dest_y, keep))

    ux = node("vrelu", [
        vaff(vcat(dest_x, anchor_x), [-63.0, 63.0], -1.0)
    ])
    rx = node("vrelu", [
        vaff(vcat(dest_x, anchor_x), [63.0, -63.0], -1.0)
    ])
    uy = node("vrelu", [
        vaff(vcat(dest_y, anchor_y), [-63.0, 63.0], -1.0)
    ])
    dypos = node("vrelu", [
        vaff(vcat(dest_y, anchor_y), [63.0, -63.0], -1.0)
    ])
    left_need = vclip(ux)
    right_need = vclip(rx)
    up_need = vclip(uy)
    down_need = vclip(dypos)
    moving = vclip(vadd(
        vadd(left_need, right_need), vadd(up_need, down_need)
    ))
    at_destination = vnot(moving)

    need_piece_click = vclip(vadd(
        vnot(active), vmul(at_destination, other_exists)
    ))
    shape_click_flag = vmul(shape_mode, need_piece_click)
    v_click_flag = vmul(vf, vnot(vselected))
    h_click_flag = vmul(hf, vnot(hselected))
    click_flag = vclip(vadd(
        vadd(v_click_flag, h_click_flag), shape_click_flag
    ))

    other_click = add(other, mul(other, c9))
    click_map = add(
        add(mul(vclick, v_click_flag), mul(hclick, h_click_flag)),
        mul(other_click, shape_click_flag),
    )
    graph.append({"id": "click", "op": "relu", "in": [click_map]})

    shape_move = vmul(shape_mode, vnot(need_piece_click))
    vmove = vmul(vf, vselected)
    hmove = vmul(hf, hselected)

    up = vadd(vmul(shape_move, up_need), vmul(hmove, hm))
    down = vadd(vmul(shape_move, down_need), vmul(hmove, hp))
    left = vadd(vmul(shape_move, left_need), vmul(vmove, vm))
    right = vadd(vmul(shape_move, right_need), vmul(vmove, vp))
    features = vcat(up, down, left, right, click_flag)

    w = np.zeros((8, 5), np.float32)
    for i in range(4):
        w[i + 1, i] = 1000.0
    w[6, 4] = 1000.0
    bias = np.array(
        [-1000000.0, 0.0, 0.0, 0.0, 0.0,
         -1000000.0, 0.0, -1000000.0],
        np.float32,
    )
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [features],
        "w": param(w),
        "b": param(bias),
    })

    memory = vcat(bits, anchor_x, anchor_y, dest_x, dest_y, active)
    graph.append({
        "id": "mem_next",
        "op": "vslice",
        "in": [memory],
        "start": 0,
        "stop": 105,
    })

    return {"graph": graph, "memory_size": 105, "params": params}
