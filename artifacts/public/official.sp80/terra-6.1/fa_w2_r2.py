import numpy as np


def build():
    graph = []
    params = {}
    counter = 0
    pcounter = 0

    def emit(target, op, inputs, name=None, **fields):
        nonlocal counter
        if name is None:
            counter += 1
            name = "n%d" % counter
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        target.append(nd)
        return name

    def node(op, inputs, name=None, **fields):
        return emit(graph, op, inputs, name, **fields)

    def param(value):
        nonlocal pcounter
        pcounter += 1
        name = "p%d" % pcounter
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def max2(a, b):
        return node("max2", [a, b])

    def min2(a, b):
        return node("min2", [a, b])

    def relu(x):
        return node("relu", [x])

    def gmax(x):
        return node("gmax", [x])

    def gmean(x):
        return node("gmean", [x])

    def vadd(a, b):
        return node("vadd", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def vrelu(x):
        return node("vrelu", [x])

    spatial_affines = {}
    vector_affines = {}

    def affine(x, scale, bias=0.0):
        key = (float(scale), float(bias))
        if key not in spatial_affines:
            w = param([[[[scale]]]])
            b = param([bias]) if bias != 0.0 else None
            spatial_affines[key] = (w, b)
        w, b = spatial_affines[key]
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return node("conv", [x], **fields)

    def vaffine(x, scale, bias=0.0, size=1):
        key = (int(size), float(scale), float(bias))
        if key not in vector_affines:
            w = param(np.eye(size, dtype=np.float32) * scale)
            b = (
                param(np.full((size,), bias, dtype=np.float32))
                if bias != 0.0 else None
            )
            vector_affines[key] = (w, b)
        w, b = vector_affines[key]
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return node("dense", [x], **fields)

    def vsub(a, b, size=1):
        return vadd(a, vaffine(b, -1.0, size=size))

    def vclip(x, size=1):
        positive = vrelu(x)
        excess = vrelu(vaffine(x, 1.0, -1.0, size=size))
        return vsub(positive, excess, size=size)

    def vmax(a, b):
        return vadd(b, vrelu(vsub(a, b)))

    def max_all(xs):
        result = xs[0]
        for x in xs[1:]:
            result = max2(result, x)
        return result

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    zero = sub(one, one)

    def clip(x):
        return min2(relu(x), one)

    # Shift one spatial channel by a fixed number of display pixels.
    shift_weights = {}
    shift_nodes = {}

    def shift(x, axis, amount):
        if amount == 0:
            return x
        key = (x, axis, int(amount))
        if key in shift_nodes:
            return shift_nodes[key]
        step = min(3, amount) if amount > 0 else max(-3, amount)
        previous = shift(x, axis, amount - step)
        wk = (axis, int(step))
        if wk not in shift_weights:
            w = np.zeros((1, 1, 7, 7), dtype=np.float32)
            if axis == "x":
                w[0, 0, 3, 3 - step] = 1.0
            else:
                w[0, 0, 3 - step, 3] = 1.0
            shift_weights[wk] = param(w)
        result = node("conv", [previous], w=shift_weights[wk])
        shift_nodes[key] = result
        return result

    p1 = sl("x0", 1, 2)
    p4 = sl("x0", 4, 5)
    p6 = sl("x0", 6, 7)
    p8 = sl("x0", 8, 9)
    p9 = sl("x0", 9, 10)
    p11 = sl("x0", 11, 12)
    p15 = sl("x0", 15, 16)

    # Gravity is the displacement from the colour-4 source to colour 6.
    mass4 = gmean(p4)
    mass6 = gmean(p6)
    x4 = gmean(mul(p4, col))
    x6 = gmean(mul(p6, col))
    y4 = gmean(mul(p4, row))
    y6 = gmean(mul(p6, row))
    dx = vsub(vmul(mass4, x6), vmul(mass6, x4))
    dy = vsub(vmul(mass4, y6), vmul(mass6, y4))

    right = clip(affine(mul(one, dx), 1.0e7))
    left = clip(affine(mul(one, dx), -1.0e7))
    down = clip(affine(mul(one, dy), 1.0e7))
    up = clip(affine(mul(one, dy), -1.0e7))
    vertical = max2(up, down)
    horizontal = max2(left, right)

    # The source thickness distinguishes the 3x and 4x camera scales.
    source_rows = node("rowmax", [p6])
    source_cols = node("colmax", [p6])
    ywidth = vaffine(gmean(source_rows), 64.0)
    xwidth = vaffine(gmean(source_cols), 64.0)
    thickness = vadd(
        vmul(ywidth, gmax(vertical)),
        vmul(xwidth, gmax(horizontal)),
    )
    e3 = node("abs", [affine(mul(one, thickness), 1.0, -3.0)])
    e4 = node("abs", [affine(mul(one, thickness), 1.0, -4.0)])
    scale3 = clip(affine(e3, -1.0, 1.0))
    scale4 = clip(affine(e4, -1.0, 1.0))

    # Detect the gap in each target facing the gravity direction.
    goals = []
    for scale, gate in ((3, scale3), (4, scale4)):
        side_x = min2(
            shift(p11, "x", scale), shift(p11, "x", -scale)
        )
        side_y = min2(
            shift(p11, "y", scale), shift(p11, "y", -scale)
        )
        forward_target = max_all([
            mul(shift(p11, "y", -scale), down),
            mul(shift(p11, "y", scale), up),
            mul(shift(p11, "x", -scale), right),
            mul(shift(p11, "x", scale), left),
        ])
        sides = max2(mul(side_x, vertical), mul(side_y, horizontal))
        notch = relu(sub(min2(sides, forward_target), p11))
        goals.append(mul(notch, gate))
    goal = max2(goals[0], goals[1])

    # Approximate the one-cell clearance around targets.
    target_body = max2(p11, goal)
    margins = []
    for scale, gate in ((3, scale3), (4, scale4)):
        xexpanded = max_all([
            target_body,
            shift(target_body, "x", scale),
            shift(target_body, "x", -scale),
        ])
        expanded = max_all([
            xexpanded,
            shift(xexpanded, "y", scale),
            shift(xexpanded, "y", -scale),
        ])
        margins.append(mul(expanded, gate))
    target_margin = max2(margins[0], margins[1])

    # Selected-sprite position and shape.
    sel_x_proj = node("colmax", [p9])
    sel_y_proj = node("rowmax", [p9])
    sel_x_edge = relu(sub(sel_x_proj, shift(sel_x_proj, "x", 1)))
    sel_y_edge = relu(sub(sel_y_proj, shift(sel_y_proj, "y", 1)))
    sel_x = vaffine(gmean(mul(sel_x_edge, col)), 64.0)
    sel_y = vaffine(gmean(mul(sel_y_edge, row)), 64.0)
    sel_x_width = vaffine(gmean(sel_x_proj), 64.0)
    sel_y_width = vaffine(gmean(sel_y_proj), 64.0)
    selected_exists = gmax(p9)

    across_width = vadd(
        vmul(sel_x_width, gmax(vertical)),
        vmul(sel_y_width, gmax(horizontal)),
    )
    along_width = vadd(
        vmul(sel_y_width, gmax(vertical)),
        vmul(sel_x_width, gmax(horizontal)),
    )
    selected_is_bar = vclip(vsub(across_width, along_width))

    # Four remembered sprite rectangles prevent immediate reselection.
    banned = zero
    for i in range(4):
        old_y = vsl("mem", 3 * i, 3 * i + 1)
        old_x = vsl("mem", 3 * i + 1, 3 * i + 2)
        valid = vsl("mem", 3 * i + 2, 3 * i + 3)

        xd = sub(col, old_x)
        yd = sub(row, old_y)
        xs = clip(affine(xd, 63.0, 1.0))
        ys = clip(affine(yd, 63.0, 1.0))
        xlong = mul(xs, clip(affine(xd, -63.0, 28.0)))
        ylong = mul(ys, clip(affine(yd, -63.0, 28.0)))
        xshort = mul(xs, clip(affine(xd, -63.0, 5.0)))
        yshort = mul(ys, clip(affine(yd, -63.0, 5.0)))
        rect = max2(
            mul(mul(xlong, yshort), vertical),
            mul(mul(xshort, ylong), horizontal),
        )
        banned = max2(banned, mul(rect, valid))

    # Five parallel hypotheses: -3, -1, 0, +1, +3 cells.
    displacements = (-3, -1, 0, 1, 3)
    hypotheses = []
    for displacement in displacements:
        if displacement == 0:
            hypotheses.append(p9)
            continue
        shifted = []
        for scale, gate in ((3, scale3), (4, scale4)):
            xm = mul(
                shift(p9, "x", displacement * scale), vertical
            )
            ym = mul(
                shift(p9, "y", displacement * scale), horizontal
            )
            shifted.append(mul(max2(xm, ym), gate))
        hypotheses.append(max2(shifted[0], shifted[1]))
    bars = node("concat", hypotheses)
    n = len(displacements)

    invalid_region = max_all([p1, p4, p6, p15, target_margin])
    collision_invalid = gmax(mul(bars, invalid_region))
    original_mass = gmean(p9)
    hypothesis_mass = gmean(bars)
    lost_mass = vsub(original_mass, hypothesis_mass, size=n)
    clipped_invalid = vclip(
        vaffine(lost_mass, 1.0e6, size=n), size=n
    )
    invalid = vclip(vadd(collision_invalid, clipped_invalid), size=n)

    bounce = max2(bars, max_all([p8, p11, p15]))
    obstacle = max_all([bounce, p1, p4])
    free = sub(one, obstacle)
    non_goal = sub(one, goal)

    # Shared one-pixel shift kernels. Direction gates select which of the
    # channels is shifted; repeating the shift 3 or 4 times moves one cell.
    forward_w = np.zeros((n, 4 * n, 3, 3), dtype=np.float32)
    side_neg_w = np.zeros((n, 2 * n, 3, 3), dtype=np.float32)
    side_pos_w = np.zeros((n, 2 * n, 3, 3), dtype=np.float32)
    for i in range(n):
        # Inputs: up, down, left, right.
        forward_w[i, i, 2, 1] = 1.0
        forward_w[i, n + i, 0, 1] = 1.0
        forward_w[i, 2 * n + i, 1, 2] = 1.0
        forward_w[i, 3 * n + i, 1, 0] = 1.0
        # Inputs: vertical gravity, horizontal gravity.
        side_neg_w[i, i, 1, 2] = 1.0
        side_neg_w[i, n + i, 2, 1] = 1.0
        side_pos_w[i, i, 1, 0] = 1.0
        side_pos_w[i, n + i, 0, 1] = 1.0

    forward_param = param(forward_w)
    side_neg_param = param(side_neg_w)
    side_pos_param = param(side_pos_w)

    def directional_shift(target, x, kind, reverse=False):
        current = x
        third = None
        for step in range(4):
            if kind == "forward":
                gates = (
                    (down, up, right, left)
                    if reverse else
                    (up, down, left, right)
                )
                pieces = [
                    emit(target, "mul", [current, gate])
                    for gate in gates
                ]
                packed = emit(target, "concat", pieces)
                current = emit(
                    target, "conv", [packed], w=forward_param
                )
            else:
                a = emit(target, "mul", [current, vertical])
                b = emit(target, "mul", [current, horizontal])
                packed = emit(target, "concat", [a, b])
                w = side_neg_param if kind == "negative" else side_pos_param
                current = emit(target, "conv", [packed], w=w)
            if step == 2:
                third = current
        a = emit(target, "mul", [third, scale3])
        b = emit(target, "mul", [current, scale4])
        return emit(target, "add", [a, b])

    # At a water cell, this map says whether a bounce obstacle is one
    # game cell ahead.
    bounce_ahead = directional_shift(
        graph, bounce, "forward", reverse=True
    )
    bounce_ahead = min2(bounce_ahead, one)

    # Wall immediately ahead, for estimating leaks.
    bad_maps = []
    for scale, gate in ((3, scale3), (4, scale4)):
        bad = max_all([
            mul(shift(p1, "y", scale), up),
            mul(shift(p1, "y", -scale), down),
            mul(shift(p1, "x", scale), left),
            mul(shift(p1, "x", -scale), right),
        ])
        bad_maps.append(mul(bad, gate))
    bad_ahead = max2(bad_maps[0], bad_maps[1])

    initial_flow = node("concat", [p6] * n)
    body = []
    live = emit(body, "mul", ["wave", non_goal])
    forward = directional_shift(body, live, "forward")
    blocked = emit(body, "mul", [live, bounce_ahead])
    side_negative = directional_shift(body, blocked, "negative")
    side_positive = directional_shift(body, blocked, "positive")
    sides = emit(body, "add", [side_negative, side_positive])
    new_flow = emit(body, "add", [forward, sides])
    new_flow = emit(body, "min2", [new_flow, one])
    new_flow = emit(body, "mul", [new_flow, free])
    next_wave = emit(body, "max2", ["wave", new_flow])

    graph.append({
        "id": "flow",
        "op": "loop",
        "T": 40,
        "state": [{"id": "wave", "init": initial_flow}],
        "body": body,
        "next": {"wave": next_wave},
    })

    reached = gmean(mul("flow.wave", goal))
    escaped = gmean(mul("flow.wave", bad_ahead))
    raw_score = vadd(
        vaffine(reached, 512.0, size=n),
        vaffine(escaped, -384.0, size=n),
    )
    score = vadd(raw_score, vaffine(invalid, -20.0, size=n))
    baseline = vsl(score, 2, 3)

    def advantage(index, cost):
        candidate = vsl(score, index, index + 1)
        return vrelu(vaffine(vsub(candidate, baseline), 1.0, -cost))

    neg = vmax(advantage(0, 0.005), advantage(1, 0.002))
    pos = vmax(advantage(4, 0.005), advantage(3, 0.002))

    # The first step itself must be valid even for a three-cell lookahead.
    neg_valid = vaffine(vsl(invalid, 1, 2), -1.0, 1.0)
    pos_valid = vaffine(vsl(invalid, 3, 4), -1.0, 1.0)
    neg = vmul(vmul(neg, neg_valid), selected_is_bar)
    pos = vmul(vmul(pos, pos_valid), selected_is_bar)

    # A configuration is considered ready when all detected target gaps
    # are reached and no predicted stream reaches a wall.
    total_goal = gmean(goal)
    reached_now = vsl(reached, 2, 3)
    escaped_now = vsl(escaped, 2, 3)
    missing = vrelu(vsub(total_goal, reached_now))
    missing_flag = vclip(vaffine(missing, 1.0e6))
    escaped_flag = vclip(vaffine(escaped_now, 1.0e6))
    goal_exists = vclip(vaffine(total_goal, 1.0e6))
    imperfect = vmax(missing_flag, escaped_flag)
    perfect = vmul(goal_exists, vaffine(imperfect, -1.0, 1.0))
    not_perfect = vaffine(perfect, -1.0, 1.0)

    neg = vmul(neg, not_perfect)
    pos = vmul(pos, not_perfect)

    # After an unsuccessful spill, make a small exploratory adjustment
    # instead of repeating the same spill immediately.
    previous_spill = vsl("aprev", 5, 6)
    exploratory = vmul(
        vmul(vmul(previous_spill, not_perfect), selected_is_bar),
        neg_valid,
    )
    neg = vadd(neg, vmul(exploratory, vaffine(exploratory, 0.0, 1.0)))

    move_strength = vmax(neg, pos)
    moving = vclip(vaffine(move_strength, 1.0e6))
    not_moving = vaffine(moving, -1.0, 1.0)

    click_mask = mul(p8, sub(one, banned))
    node("mul", [click_mask, one], name="click")
    click_exists = gmax(click_mask)
    choose_click = vmul(vmul(not_moving, not_perfect), click_exists)
    no_click = vaffine(click_exists, -1.0, 1.0)
    default_spill = vmul(vmul(not_moving, not_perfect), no_click)
    spill = vadd(perfect, default_spill)

    # Movement starts a fresh pass. Clicking records the previous sprite.
    record = node("vconcat", [
        sel_y, sel_x, selected_exists, vsl("mem", 0, 9)
    ])
    active = vadd(vadd(moving, choose_click), spill)
    hold = vaffine(active, -1.0, 1.0)
    kept = vmul("mem", hold)
    remembered = vmul(record, choose_click)
    node("vadd", [kept, remembered], name="mem_next")

    vertical_v = gmax(vertical)
    horizontal_v = gmax(horizontal)
    low = vaffine(selected_exists, 0.0, -100.0)
    neg_logit = vaffine(neg, 1000.0)
    pos_logit = vaffine(pos, 1000.0)
    node("vconcat", [
        low,
        vmul(neg_logit, horizontal_v),
        vmul(pos_logit, horizontal_v),
        vmul(neg_logit, vertical_v),
        vmul(pos_logit, vertical_v),
        vaffine(spill, 2.0),
        choose_click,
        low,
    ], name="logits")

    return {
        "graph": graph,
        "memory_size": 12,
        "params": params,
    }
