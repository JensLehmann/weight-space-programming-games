import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]
    cache = {}

    def fresh(prefix="n"):
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    def param(a):
        name = fresh("p")
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def cached(key, make):
        if key not in cache:
            cache[key] = param(make())
        return cache[key]

    def eye(n, scale=1.0):
        return cached(("eye", n, scale),
                      lambda: np.eye(n, dtype=np.float32) * scale)

    def ones(n, value=1.0):
        return cached(("bias", n, value),
                      lambda: np.full(n, value, dtype=np.float32))

    def node(dst, op, inputs, **kw):
        name = fresh()
        d = {"id": name, "op": op, "in": inputs}
        d.update(kw)
        dst.append(d)
        return name

    def dense(dst, x, w, b=None):
        kw = {"w": w}
        if b is not None:
            kw["b"] = b
        return node(dst, "dense", [x], **kw)

    def conv(dst, x, w, b=None):
        kw = {"w": w}
        if b is not None:
            kw["b"] = b
        return node(dst, "conv", [x], **kw)

    def sl(dst, x, start, stop):
        return node(dst, "slice", [x], start=start, stop=stop)

    def vsl(dst, x, start, stop):
        return node(dst, "vslice", [x], start=start, stop=stop)

    def one_minus(dst, x, n):
        return dense(dst, x, eye(n, -1.0), ones(n))

    def present(dst, x, n):
        # Exact 0/1 for nonnegative integer-valued x.
        t = dense(dst, x, eye(n, -1.0), ones(n))
        t = node(dst, "vrelu", [t])
        return one_minus(dst, t, n)

    def categorical(dst, scalar, values):
        # One-hot for an integer scalar, with all zeros outside values.
        values = np.asarray(values, dtype=np.float32)
        n = len(values)
        wp = param(np.ones((n, 1), dtype=np.float32))
        wn = param(-np.ones((n, 1), dtype=np.float32))
        bp = param(-values)
        bn = param(values)
        a = dense(dst, scalar, wp, bp)
        b = dense(dst, scalar, wn, bn)
        a = node(dst, "vrelu", [a])
        b = node(dst, "vrelu", [b])
        d = node(dst, "vadd", [a, b])
        e = dense(dst, d, eye(n, -1.0), ones(n))
        return node(dst, "vrelu", [e])

    # A spatial constant one.
    spatial_one = sl(graph, "coords", 2, 3)

    # Build row and column indicators for the target and the canvas.
    # Clamping the peaks to one avoids coordinate-rounding errors at centers.
    def axis_masks(axis, positions):
        n = len(positions)
        w = np.zeros((n, 3, 1, 1), dtype=np.float32)
        w[:, axis, 0, 0] = 63.0
        z = conv(graph, "coords", param(w),
                 param(-np.asarray(positions, dtype=np.float32)))
        z = node(graph, "abs", [z])
        w2 = np.zeros((n, n, 1, 1), dtype=np.float32)
        for i in range(n):
            w2[i, i, 0, 0] = -2.0
        z = conv(graph, z, param(w2), ones(n, 2.0))
        z = node(graph, "relu", [z])
        return node(graph, "min2", [z, spatial_one])

    rows = axis_masks(0, list(range(3, 13)) + list(range(34, 44)))
    cols = axis_masks(1, list(range(3, 13)) + list(range(27, 37)))

    # Encode the observed colour as its integer palette index.
    wc = np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
    colour_image = conv(graph, "x0", param(wc))
    red_image = sl(graph, "x0", 2, 3)

    # Extract the 80 scored pixels, omitting the two ignored diagonals.
    def extract_picture(row_offset, col_offset):
        pieces = []
        cm = sl(graph, cols, col_offset, col_offset + 10)
        for i in range(10):
            rm = sl(graph, rows, row_offset + i, row_offset + i + 1)
            mask = node(graph, "mul", [rm, cm])
            vals = node(graph, "mul", [mask, colour_image])
            vals = node(graph, "gmax", [vals])
            keep = [j for j in range(10) if j != i and j != 9 - i]
            start = keep[0]
            prev = keep[0]
            for j in keep[1:]:
                if j != prev + 1:
                    pieces.append(vsl(graph, vals, start, prev + 1))
                    start = j
                prev = j
            pieces.append(vsl(graph, vals, start, prev + 1))
        return node(graph, "vconcat", pieces)

    target = extract_picture(0, 0)
    canvas = extract_picture(10, 10)

    # Sample distinctive red border pixels for the eight basket positions,
    # followed by one nozzle pixel for each cardinal position.
    points = [
        (24, 25), (21, 39), (38, 46), (40, 43),
        (53, 25), (56, 24), (38, 17), (21, 24),
        (20, 29), (36, 48), (55, 29), (36, 12),
    ]
    npt = len(points)
    wp = np.zeros((2 * npt, 3, 1, 1), dtype=np.float32)
    bp = np.zeros(2 * npt, dtype=np.float32)
    for k, (r, c) in enumerate(points):
        wp[2 * k, 0, 0, 0] = 63.0
        wp[2 * k + 1, 1, 0, 0] = 63.0
        bp[2 * k] = -r
        bp[2 * k + 1] = -c
    point_diff = conv(graph, "coords", param(wp), param(bp))
    point_diff = node(graph, "abs", [point_diff])
    wm = np.zeros((npt, 2 * npt, 1, 1), dtype=np.float32)
    for k in range(npt):
        wm[k, 2 * k, 0, 0] = -2.0
        wm[k, 2 * k + 1, 0, 0] = -2.0
    point_masks = conv(graph, point_diff, param(wm), ones(npt, 2.0))
    point_masks = node(graph, "relu", [point_masks])
    point_masks = node(graph, "min2", [point_masks, spatial_one])
    red_points = node(graph, "mul", [point_masks, red_image])
    red_points = node(graph, "gmax", [red_points])
    # Snap sampled indicators to exact zero or one.
    red_points = dense(graph, red_points, eye(npt, 4.0), ones(npt, -1.0))
    red_points = node(graph, "vrelu", [red_points])
    red_points = dense(graph, red_points, eye(npt, -1.0), ones(npt))
    red_points = node(graph, "vrelu", [red_points])
    red_points = one_minus(graph, red_points, npt)
    position = vsl(graph, red_points, 0, 8)
    nozzle_seen = vsl(graph, red_points, 8, 12)

    # Palette centers are surrounded by a 5x5 colour-4 border.
    border = np.zeros((1, 1, 5, 5), dtype=np.float32)
    border[0, 0, 0, :] = 1
    border[0, 0, 4, :] = 1
    border[0, 0, :, 0] = 1
    border[0, 0, :, 4] = 1
    colour4 = sl(graph, "x0", 4, 5)
    ring = conv(graph, colour4, param(border), ones(1, -15.5))
    ring = node(graph, "relu", [ring])
    ring_scale = param(np.array([[[[2.0]]]], dtype=np.float32))
    ring = conv(graph, ring, ring_scale)
    palette_row = axis_masks(0, [4])
    swatches = node(graph, "mul", [ring, palette_row])

    palette_planes = node(graph, "mul", ["x0", swatches])
    palette = node(graph, "gmax", [palette_planes])
    palette_bits = [vsl(graph, palette, c, c + 1) for c in range(16)]

    # The underline at row 7 identifies the selected palette swatch.
    zero_plane = sl(graph, "x0", 0, 1)
    shift_up = np.zeros((1, 1, 7, 7), dtype=np.float32)
    shift_up[0, 0, 6, 3] = 1.0
    underline = conv(graph, zero_plane, param(shift_up))

    # Detect a permanently absent nozzle when a cardinal basket is visible.
    pick_cardinal = np.zeros((4, 8), dtype=np.float32)
    for k, p in enumerate((0, 2, 4, 6)):
        pick_cardinal[k, p] = 1.0
    cardinal = dense(graph, position, param(pick_cardinal))
    cardinal_sum = dense(graph, cardinal,
                         param(np.ones((1, 4), dtype=np.float32)))
    visible_nozzle = node(graph, "vmul", [cardinal, nozzle_seen])
    visible_nozzle = dense(graph, visible_nozzle,
                           param(np.ones((1, 4), dtype=np.float32)))
    missing = node(graph, "vadd", [
        cardinal_sum,
        dense(graph, visible_nozzle, eye(1, -1.0))
    ])

    old_stack = vsl(graph, "mem", 0, 24)
    old_missing = vsl(graph, "mem", 24, 25)
    old_not_missing = one_minus(graph, old_missing, 1)
    newly_missing = node(graph, "vmul", [missing, old_not_missing])
    missing_difference = node(graph, "vadd", [
        old_missing, dense(graph, missing, eye(1, -1.0))
    ])
    missing_difference = node(graph, "vrelu", [missing_difference])
    no_nozzle = node(graph, "vadd", [missing, missing_difference])

    # Equality vectors for each possible target colour. The tolerance absorbs
    # tiny floating-point errors in the coordinate masks.
    eq = []
    for c in range(16):
        plus = dense(graph, target, eye(80), ones(80, -float(c)))
        minus = dense(graph, target, eye(80, -1.0), ones(80, float(c)))
        plus = node(graph, "vrelu", [plus])
        minus = node(graph, "vrelu", [minus])
        dist = node(graph, "vadd", [plus, minus])
        dist = dense(graph, dist, eye(80, 4.0), ones(80, -2.0))
        dist = node(graph, "vrelu", [dist])
        same = dense(graph, dist, eye(80, -1.0), ones(80))
        eq.append(node(graph, "vrelu", [same]))

    diff1 = node(graph, "vadd", [
        target, dense(graph, canvas, eye(80, -1.0))
    ])
    diff2 = node(graph, "vadd", [
        canvas, dense(graph, target, eye(80, -1.0))
    ])
    diff1 = node(graph, "vrelu", [diff1])
    diff2 = node(graph, "vrelu", [diff2])
    diff = node(graph, "vadd", [diff1, diff2])
    diff = dense(graph, diff, eye(80, 4.0), ones(80, -2.0))
    diff = node(graph, "vrelu", [diff])
    same = dense(graph, diff, eye(80, -1.0), ones(80))
    same = node(graph, "vrelu", [same])
    mismatch = one_minus(graph, same, 80)

    # Paint masks in the same order as the eight basket positions, then the
    # four nozzle positions. Each row describes only scored pixels.
    cells = [(i, j) for i in range(10) for j in range(10)
             if j != i and j != 9 - i]

    def contains(k, i, j):
        if k == 0:
            return i < 5
        if k == 1:
            return j >= i
        if k == 2:
            return j >= 5
        if k == 3:
            return j >= 9 - i
        if k == 4:
            return i >= 5
        if k == 5:
            return j <= i
        if k == 6:
            return j < 5
        if k == 7:
            return j <= 9 - i
        if k == 8:
            return i < 3 and 3 <= j < 7
        if k == 9:
            return 3 <= i < 7 and j >= 7
        if k == 10:
            return i >= 7 and 3 <= j < 7
        return 3 <= i < 7 and j < 3

    masks = np.asarray(
        [[float(contains(k, i, j)) for i, j in cells]
         for k in range(12)], dtype=np.float32
    )
    mask_w = param(masks)
    neg_mask_w = param(-masks)

    # The first eight regions are always enabled; the last four require
    # a nozzle. The flag is exact once a cardinal position is observed.
    allow_w = np.zeros((12, 1), dtype=np.float32)
    allow_w[8:, 0] = -1.0
    allow = dense(graph, no_nozzle, param(allow_w), ones(12))

    # Reverse planning. A candidate is valid when every still-uncovered
    # pixel in its mask has that colour, and it covers a mismatch.
    body = []
    q = "plan_q"
    stack = "plan_stack"
    neg_total = dense(body, q, neg_mask_w)
    remaining_bad = node(body, "vmul", [q, mismatch])
    progress = dense(body, remaining_bad, mask_w)
    progress = present(body, progress, 12)

    candidates = []
    for c in range(16):
        covered_colour = node(body, "vmul", [q, eq[c]])
        matches = dense(body, covered_colour, mask_w)
        good = node(body, "vadd", [matches, neg_total])
        good = dense(body, good, eye(12), ones(12))
        good = node(body, "vrelu", [good])
        good = node(body, "vmul", [good, progress])
        good = node(body, "vmul", [good, allow])
        good = node(body, "vmul", [good, palette_bits[c]])
        candidates.append(good)
    candidates = node(body, "vconcat", candidates)

    # Pick the first valid candidate in region-major, then colour order.
    prefix = np.zeros((192, 192), dtype=np.float32)
    for c in range(16):
        for k in range(12):
            i = 12 * c + k
            rank_i = 16 * k + c
            for d in range(16):
                for h in range(12):
                    j = 12 * d + h
                    if 16 * h + d < rank_i:
                        prefix[i, j] = -1.0
    eligible = dense(body, candidates, param(prefix), ones(192))
    eligible = node(body, "vrelu", [eligible])
    winner = node(body, "vmul", [candidates, eligible])
    any_winner = dense(body, winner,
                       param(np.ones((1, 192), dtype=np.float32)))

    remove = np.zeros((80, 192), dtype=np.float32)
    insert = np.zeros((24, 192), dtype=np.float32)
    for c in range(16):
        for k in range(12):
            j = 12 * c + k
            remove[:, j] = -masks[k]
            insert[0, j] = k + 1
            insert[12, j] = c + 1
    keep = dense(body, winner, param(remove), ones(80))
    next_q = node(body, "vmul", [q, keep])

    shift = np.zeros((24, 24), dtype=np.float32)
    for i in range(1, 12):
        shift[i, i - 1] = 1.0
        shift[12 + i, 12 + i - 1] = 1.0
    shifted = dense(body, stack, param(shift))
    inserted = dense(body, winner, param(insert))
    proposed = node(body, "vadd", [shifted, inserted])
    proposed = node(body, "vmul", [proposed, any_winner])
    no_winner = one_minus(body, any_winner, 1)
    unchanged = node(body, "vmul", [stack, no_winner])
    next_stack = node(body, "vadd", [proposed, unchanged])

    graph.append({
        "id": "planning",
        "op": "loop",
        "T": 12,
        "state": [
            {"id": q, "zeros": [80]},
            {"id": stack, "zeros": [24]},
        ],
        "body": body,
        "next": {q: next_q, stack: next_stack},
    })

    # The reverse planner needs its uncovered set initialized to ones.
    # Replace its zero initializer with a vector derived from the input.
    zero_to_one_w = param(np.zeros((80, 8), dtype=np.float32))
    initial_q = dense(graph, "aprev", zero_to_one_w, ones(80))
    # initial_q must precede the loop.
    initial_node = graph.pop()
    loop_node = graph.pop()
    graph.append(initial_node)
    loop_node["state"][0] = {"id": q, "init": initial_q}
    graph.append(loop_node)

    # Continue an existing plan unless nozzle absence was newly discovered.
    old_top = vsl(graph, old_stack, 0, 1)
    old_active = present(graph, old_top, 1)
    not_newly_missing = one_minus(graph, newly_missing, 1)
    use_old = node(graph, "vmul", [old_active, not_newly_missing])
    use_new = one_minus(graph, use_old, 1)
    kept_old = node(graph, "vmul", [old_stack, use_old])
    planned = node(graph, "vmul", ["planning.plan_stack", use_new])
    active_stack = node(graph, "vadd", [kept_old, planned])

    shape_code = vsl(graph, active_stack, 0, 1)
    colour_code = vsl(graph, active_stack, 12, 13)
    shape = categorical(graph, shape_code, range(1, 13))
    wanted_colour = categorical(graph, colour_code, range(1, 17))
    active = dense(graph, shape, param(np.ones((1, 12), dtype=np.float32)))

    # Locate the desired palette button and determine whether it is selected.
    matching_planes = node(graph, "mul", ["x0", wanted_colour])
    sum16 = param(np.ones((1, 16, 1, 1), dtype=np.float32))
    matching_colour = conv(graph, matching_planes, sum16)
    colour_click_map = node(graph, "mul", [matching_colour, swatches])
    selected_map = node(graph, "mul", [colour_click_map, underline])
    selected = node(graph, "gmax", [selected_map])
    need_colour = one_minus(graph, selected, 1)
    palette_click = node(graph, "vmul", [active, need_colour])

    # Position/shape pairs drive a shortest route around the eight-position
    # ring. Coordinates are the game's 3x3 logical basket positions.
    logical = [
        (0, 1), (0, 2), (1, 2), (2, 2),
        (2, 1), (2, 0), (1, 0), (0, 0),
    ]
    desired_position = list(range(8)) + [0, 2, 4, 6]
    expand_pos = np.zeros((96, 8), dtype=np.float32)
    expand_shape = np.zeros((96, 12), dtype=np.float32)
    directions = np.zeros((4, 96), dtype=np.float32)
    at_target = np.zeros((1, 96), dtype=np.float32)
    for p in range(8):
        for k in range(12):
            j = 12 * p + k
            expand_pos[j, p] = 1.0
            expand_shape[j, k] = 1.0
            goal = desired_position[k]
            if p == goal:
                at_target[0, j] = 1.0
                continue
            clockwise = (goal - p) % 8
            nxt = (p + 1) % 8 if clockwise <= 4 else (p - 1) % 8
            dr = logical[nxt][0] - logical[p][0]
            dc = logical[nxt][1] - logical[p][1]
            action = {( -1, 0): 0, (1, 0): 1,
                      (0, -1): 2, (0, 1): 3}[(dr, dc)]
            directions[action, j] = 1.0
    pp = dense(graph, position, param(expand_pos))
    ss = dense(graph, shape, param(expand_shape))
    pair = node(graph, "vmul", [pp, ss])
    move = dense(graph, pair, param(directions))
    move = node(graph, "vmul", [move, selected])
    at = dense(graph, pair, param(at_target))
    at = node(graph, "vmul", [at, selected])

    basket_kind_w = np.zeros((1, 12), dtype=np.float32)
    basket_kind_w[0, :8] = 1
    nozzle_kind_w = np.zeros((1, 12), dtype=np.float32)
    nozzle_kind_w[0, 8:] = 1
    basket_kind = dense(graph, shape, param(basket_kind_w))
    nozzle_kind = dense(graph, shape, param(nozzle_kind_w))
    paint_basket = node(graph, "vmul", [at, basket_kind])
    paint_nozzle = node(graph, "vmul", [at, nozzle_kind])
    paint = node(graph, "vadd", [paint_basket, paint_nozzle])
    click_action = node(graph, "vadd", [palette_click, paint_nozzle])

    # Click either the palette center or the visible nozzle border pixel.
    nozzle_masks = sl(graph, point_masks, 8, 12)
    nozzle_shape = vsl(graph, shape, 8, 12)
    nozzle_map = node(graph, "mul", [nozzle_masks, nozzle_shape])
    sum4 = param(np.ones((1, 4, 1, 1), dtype=np.float32))
    nozzle_map = conv(graph, nozzle_map, sum4)
    a = node(graph, "mul", [colour_click_map, palette_click])
    b = node(graph, "mul", [nozzle_map, paint_nozzle])
    graph.append({"id": "click", "op": "add", "in": [a, b]})

    outputs = node(graph, "vconcat", [move, paint_basket, click_action])
    logits_w = np.zeros((8, 6), dtype=np.float32)
    for i in range(4):
        logits_w[i + 1, i] = 2.0
    logits_w[5, 4] = 2.0
    logits_w[6, 5] = 2.0
    logits_b = np.zeros(8, dtype=np.float32)
    logits_b[0] = -1.0
    logits_b[5] = 0.01
    logits_b[7] = -1.0
    graph.append({
        "id": "logits", "op": "dense", "in": [outputs],
        "w": param(logits_w), "b": param(logits_b),
    })

    # Pop the plan only after a paint action, not while moving or selecting.
    shifted_active = dense(graph, active_stack, cache.get("unused", "") or
                           # The same shift matrix is deliberately reused.
                           next(name for name, value in params.items()
                                if value.shape == (24, 24)
                                and np.array_equal(value, shift)))
    not_paint = one_minus(graph, paint, 1)
    keep_stack = node(graph, "vmul", [active_stack, not_paint])
    pop_stack = node(graph, "vmul", [shifted_active, paint])
    next_mem_stack = node(graph, "vadd", [keep_stack, pop_stack])
    graph.append({
        "id": "mem_next", "op": "vconcat",
        "in": [next_mem_stack, no_nozzle],
    })

    return {"graph": graph, "memory_size": 25, "params": params}
