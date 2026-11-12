import numpy as np


def build():
    P, C, S = 5, 5, 4
    R = S + 3 * P
    M = 6 * P + 4

    params = {}
    parameter_cache = {}
    root = []
    root_cache = {}
    current = root
    cache = root_cache
    scope = 0
    serial = 0
    shapes = {
        "x0": ("s", 16), "coords": ("s", 3),
        "mem": ("v", M), "aprev": ("v", 8), "avail": ("v", 8),
    }
    owners = {k: 0 for k in shapes}
    producers = {}
    constants = {}

    def uid():
        nonlocal serial
        serial += 1
        return "n%d" % serial

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def emit(op, ins, out=None, **fields):
        local = scope != 0 and any(owners[x] != 0 for x in ins)
        destination = current if local else root
        destination_cache = cache if local else root_cache
        key = (op, tuple(ins), tuple(sorted(fields.items())))
        if out is None and key in destination_cache:
            return destination_cache[key]
        name = out or uid()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        destination.append(nd)
        producers[name] = nd
        owners[name] = scope if local else 0
        if out is None:
            destination_cache[key] = name

        if op == "conv":
            shapes[name] = ("s", params[fields["w"]].shape[0])
        elif op == "dense":
            shapes[name] = ("v", params[fields["w"]].shape[0])
        elif op in ("concat", "vconcat"):
            shapes[name] = (
                shapes[ins[0]][0], sum(shapes[x][1] for x in ins)
            )
        elif op in ("slice", "vslice"):
            shapes[name] = (
                shapes[ins[0]][0], fields["stop"] - fields["start"]
            )
        elif op in ("gmax", "gmean"):
            shapes[name] = ("v", shapes[ins[0]][1])
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shapes[name] = ("s", max(shapes[x][1] for x in ins))
        elif op in ("vadd", "vmul"):
            shapes[name] = ("v", max(shapes[x][1] for x in ins))
        else:
            shapes[name] = shapes[ins[0]]
        return name

    def affine(x, w, b=None, out=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        b = (
            np.zeros(w.shape[0], np.float32)
            if b is None else np.asarray(b, np.float32).reshape(-1)
        )
        if (
            out is None and not np.any(b)
            and w.shape[0] == w.shape[1]
            and np.array_equal(w, np.eye(w.shape[0], dtype=np.float32))
        ):
            return x

        # Exact algebraic fusion of adjacent affine operations.
        while x in producers:
            previous = producers[x]
            if previous["op"] not in ("dense", "conv"):
                break
            old_w = params[previous["w"]]
            if previous["op"] == "conv":
                if old_w.shape[-2:] != (1, 1):
                    break
                old_w = old_w[:, :, 0, 0]
            old_b = (
                params[previous["b"]] if "b" in previous
                else np.zeros(old_w.shape[0], np.float32)
            )
            b = w @ old_b + b
            w = w @ old_w
            x = previous["in"][0]

        spatial_input = shapes[x][0] == "s"
        fields = {
            "w": parameter(w[:, :, None, None] if spatial_input else w)
        }
        if np.any(b != 0):
            fields["b"] = parameter(b)
        return emit(
            "conv" if spatial_input else "dense",
            [x], out=out, **fields
        )

    def constant(values):
        a = np.asarray(values, np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key not in constants:
            constants[key] = affine(
                "avail", np.zeros((len(a), 8), np.float32), a
            )
        return constants[key]

    def scalar(x):
        return constant([x])

    def cat(xs):
        xs = list(xs)
        if len(xs) == 1:
            return xs[0]
        return emit(
            "concat" if shapes[xs[0]][0] == "s" else "vconcat", xs
        )

    def sl(x, a, b):
        if a == 0 and b == shapes[x][1]:
            return x
        return emit(
            "slice" if shapes[x][0] == "s" else "vslice",
            [x], start=a, stop=b
        )

    def add(a, b):
        spatial_input = shapes[a][0] == "s" or shapes[b][0] == "s"
        return emit("add" if spatial_input else "vadd", [a, b])

    def mul(a, b):
        spatial_input = shapes[a][0] == "s" or shapes[b][0] == "s"
        return emit("mul" if spatial_input else "vmul", [a, b])

    def scale(a, k):
        return affine(a, np.eye(shapes[a][1], dtype=np.float32) * k)

    def bias(a, b):
        n = shapes[a][1]
        return affine(a, np.eye(n, dtype=np.float32), np.full(n, b))

    def neg(a):
        return scale(a, -1)

    def sub(a, b):
        if shapes[a][0] == "s" or shapes[b][0] == "s":
            return emit("sub", [a, b])
        na, nb = shapes[a][1], shapes[b][1]
        n = max(na, nb)
        wa = np.ones((n, 1), np.float32) if na == 1 else np.eye(n)
        wb = np.ones((n, 1), np.float32) if nb == 1 else np.eye(n)
        return affine(cat([a, b]), np.concatenate([wa, -wb], axis=1))

    def relu(a):
        return emit("relu" if shapes[a][0] == "s" else "vrelu", [a])

    def tanh(a):
        return emit("tanh" if shapes[a][0] == "s" else "vtanh", [a])

    def absolute(a):
        if shapes[a][0] == "s":
            return emit("abs", [a])
        eye = np.eye(shapes[a][1], dtype=np.float32)
        both = relu(affine(a, np.concatenate([eye, -eye], axis=0)))
        return affine(both, np.concatenate([eye, eye], axis=1))

    def step(a, threshold=0.0, gain=1000.0):
        n = shapes[a][1]
        z = affine(
            a, np.eye(n, dtype=np.float32) * gain,
            np.full(n, -threshold * gain, np.float32)
        )
        return relu(tanh(z))

    def equal(a, b, tolerance=.2):
        return step(neg(absolute(sub(a, b))), -tolerance)

    def total(a):
        return affine(a, np.ones((1, shapes[a][1]), np.float32))

    def gm(a):
        return emit("gmax", [a])

    def choose(flag, yes, no):
        return add(mul(flag, yes), mul(bias(neg(flag), 1), no))

    def max_channels(a):
        while shapes[a][1] > 1:
            n = shapes[a][1]
            half = n // 2
            reduced = emit(
                "max2", [sl(a, 0, half), sl(a, half, 2 * half)]
            )
            if n % 2:
                reduced = cat([reduced, sl(a, n - 1, n)])
            a = reduced
        return a

    zero = scalar(0)
    zsp = affine("coords", [[0, 0, 0]])
    osp = affine("coords", [[0, 0, 1]])

    def spatial(v):
        return add(zsp, v)

    # Canonical orientation is inferred from the step-meter edge.
    screen_y = affine("coords", [[63, 0, 0]])
    screen_x = affine("coords", [[0, 63, 0]])
    meter = add(sl("x0", 14, 15), sl("x0", 0, 1))
    edges = relu(affine(
        "coords",
        [[-63, 0, 0], [0, -63, 0], [63, 0, 0], [0, 63, 0]],
        [1, 1, -62, -62]
    ))
    rotation = step(
        emit("gmean", [mul(meter, edges)]), .005, 10000
    )
    all_x = affine(
        "coords",
        [[0, 63, 0], [-63, 0, 63], [0, -63, 63], [63, 0, 0]]
    )
    all_y = affine(
        "coords",
        [[63, 0, 0], [0, 63, 0], [-63, 0, 63], [0, -63, 63]]
    )
    X = total(mul(all_x, rotation))
    Y = total(mul(all_y, rotation))
    xy = cat([X, Y])
    bbox_features = affine(
        xy, [[1, 0], [0, 1], [-1, 0], [0, -1]],
        [1, 1, 64, 64]
    )
    moment_features = cat([osp, X, Y])
    priority = affine(xy, [[-1, -64]], [5000])
    bbox_matrix = np.array([
        [0, 0, -1, 0], [1, 0, 0, 0],
        [0, 0, 0, -1], [0, 1, 0, 0],
    ], np.float32)
    bbox_bias = [64, -1, 64, -1]

    def component(mask, seed_mask=None):
        nonlocal current, cache, scope
        if seed_mask is None:
            seed_mask = mask
        scores = mul(seed_mask, priority)
        seed = mul(seed_mask, relu(bias(sub(scores, gm(scores)), 1)))
        saved = current, cache, scope
        body = []
        state_id = uid()
        current, cache, scope = body, {}, state_id
        shapes[state_id], owners[state_id] = ("s", 1), scope
        kernel = np.zeros((1, 1, 3, 3), np.float32)
        kernel[0, 0, 1, :] = 1
        kernel[0, 0, :, 1] = 1
        dilation = emit("conv", [state_id], w=parameter(kernel))
        grown = emit("min2", [dilation, mask])
        current, cache, scope = saved
        lid = uid()
        root.append({
            "id": lid, "op": "loop", "T": 40,
            "state": [{"id": state_id, "init": seed}],
            "body": body, "next": {state_id: grown},
        })
        result = lid + "." + state_id
        shapes[result], owners[result] = ("s", 1), 0
        return result

    source_masks = []
    remaining = sl("x0", 6, 7)
    for _ in range(S):
        comp = component(remaining)
        source_masks.append(comp)
        remaining = sub(remaining, comp)

    first_source_box = affine(
        gm(mul(source_masks[0], bbox_features)),
        bbox_matrix, bbox_bias
    )
    cell = affine(first_source_box, [[-1, 1, 0, 0]], [1])
    pad = sub(sl(first_source_box, 2, 3), cell)
    minimum_y = add(sl(first_source_box, 2, 3), scale(cell, 2))
    background_box = affine(
        gm(mul(sl("x0", 12, 13), bbox_features)),
        bbox_matrix, bbox_bias
    )
    minimum_x = sl(background_box, 0, 1)
    maximum_x = sl(background_box, 1, 2)
    maximum_y = sl(background_box, 3, 4)
    side_walls = step(
        sub(sub(scalar(63), pad), add(maximum_x, cell)), -.25
    )

    embedded_sources = mul(
        sl("x0", 4, 5), step(sub(Y, minimum_y), -.25)
    )
    selected_mask = sl("x0", 9, 10)
    platform_masks = [
        component(add(selected_mask, embedded_sources), selected_mask)
    ]
    remaining = sub(
        add(add(sl("x0", 8, 9), sl("x0", 15, 16)), embedded_sources),
        mul(platform_masks[0], embedded_sources)
    )
    for _ in range(P - 1):
        comp = component(remaining)
        platform_masks.append(comp)
        remaining = sub(remaining, comp)

    cup_masks = []
    remaining = add(sl("x0", 11, 12), sl("x0", 13, 14))
    for _ in range(C):
        comp = component(remaining)
        cup_masks.append(comp)
        remaining = sub(remaining, comp)

    # Batch object geometry and asymmetry measurements.
    masks = cat(source_masks + platform_masks + cup_masks)
    N = S + P + C
    existence = gm(masks)
    repeated_bbox_features = affine(
        bbox_features, np.repeat(np.eye(4, dtype=np.float32), N, axis=0)
    )
    extremes = gm(mul(cat([masks] * 4), repeated_bbox_features))
    box_w = np.zeros((4 * N, 4 * N), np.float32)
    box_b = np.zeros(4 * N, np.float32)
    for i in range(N):
        box_w[i, 2 * N + i] = -1
        box_b[i] = 64
        box_w[N + i, i] = 1
        box_b[N + i] = -1
        box_w[2 * N + i, 3 * N + i] = -1
        box_b[2 * N + i] = 64
        box_w[3 * N + i, N + i] = 1
        box_b[3 * N + i] = -1
    boxes = mul(affine(extremes, box_w, box_b), cat([existence] * 4))
    repeated_moment_features = affine(
        moment_features,
        np.repeat(np.eye(3, dtype=np.float32), N, axis=0)
    )
    moments = emit("gmean", [
        mul(cat([masks] * 3), repeated_moment_features)
    ])
    centre_w = np.zeros((2 * N, 4 * N), np.float32)
    for i in range(N):
        centre_w[i, i] = centre_w[i, N + i] = .5
        centre_w[N + i, 2 * N + i] = .5
        centre_w[N + i, 3 * N + i] = .5
    centres = affine(boxes, centre_w)
    masses = cat([sl(moments, 0, N)] * 2)
    signs = tanh(scale(
        sub(sl(moments, N, 3 * N), mul(centres, masses)), 100000
    ))

    def fields(start, count):
        result = [
            sl(boxes, j * N + start, j * N + start + count)
            for j in range(4)
        ]
        result += [
            sl(signs, j * N + start, j * N + start + count)
            for j in range(2)
        ]
        result.append(sl(existence, start, start + count))
        return result

    sx0, sx1, sy0, sy1, _, _, source_exists = fields(0, S)
    source_x = scale(add(sx0, sx1), .5)
    source_y = scale(add(sy0, sy1), .5)
    px0, px1, py0, py1, pmx, pmy, pexists = fields(S, P)
    pw, ph = bias(sub(px1, px0), 1), bias(sub(py1, py0), 1)
    corner = mul(
        step(sub(pw, cell), .25), step(sub(ph, cell), .25)
    )
    hx, hy = mul(neg(pmx), corner), mul(neg(pmy), corner)
    top_corner = relu(neg(hy))
    shx, shy = mul(cell, hx), mul(cell, hy)

    cx0, cx1, cy0, cy1, cmx, cmy, cexists = fields(S + P, C)
    cw, ch = bias(sub(cx1, cx0), 1), bias(sub(cy1, cy0), 1)
    wide_cup = step(sub(cw, ch), .25)
    cmx = mul(cmx, bias(neg(wide_cup), 1))
    cmy = mul(cmy, wide_cup)
    cup_x = sub(
        scale(add(cx0, cx1), .5), scale(mul(cmx, cell), .5)
    )
    cup_y = sub(
        scale(add(cy0, cy1), .5), scale(mul(cmy, cell), .5)
    )
    cup_down, cup_right, cup_left = relu(cmy), relu(cmx), relu(neg(cmx))

    # Pairwise clearances. Flat bars may merge on the same row.
    pairs = [(i, j) for i in range(P) for j in range(i + 1, P)]
    K = len(pairs)
    pick_a = np.zeros((K, P), np.float32)
    pick_b = np.zeros((K, P), np.float32)
    for k, (i, j) in enumerate(pairs):
        pick_a[k, i] = 1
        pick_b[k, j] = 1
    pair_difference = pick_a - pick_b
    pair_half_sum = .5 * (pick_a + pick_b)
    pair_exists = mul(affine(pexists, pick_a), affine(pexists, pick_b))
    horizontal_bar = equal(ph, cell)
    pair_bars = mul(
        affine(horizontal_bar, pick_a), affine(horizontal_bar, pick_b)
    )
    pair_radius_x = bias(add(affine(pw, pair_half_sum), cell), -.5)
    pair_radius_y = bias(add(affine(ph, pair_half_sum), cell), -.5)
    half_pw = scale(bias(pw, -1), .5)
    half_ph = scale(bias(ph, -1), .5)

    # Computational rows represent rays; columns represent trial placements.
    row_difference = affine(
        screen_y, np.ones((R, 1)), -np.arange(R, dtype=np.float32)
    )
    row_masks = step(neg(absolute(row_difference)), -.2)
    object_matrix = np.zeros((P, R), np.float32)
    left_matrix = np.zeros((P, R), np.float32)
    right_matrix = np.zeros((P, R), np.float32)
    turn_matrix = np.zeros((P, R), np.float32)
    source_matrix = np.zeros((S, R), np.float32)
    down_matrix = np.zeros((1, R), np.float32)
    for i in range(S):
        source_matrix[i, i] = 1
        down_matrix[0, i] = 1
    for i in range(P):
        r = S + 3 * i
        object_matrix[i, r:r + 3] = 1
        left_matrix[i, r] = 1
        right_matrix[i, r + 1] = 1
        turn_matrix[i, r + 2] = 1
        down_matrix[0, r:r + 2] = 1

    object_rows = affine(row_masks, object_matrix)
    normal_left_rows = affine(row_masks, left_matrix)
    normal_right_rows = affine(row_masks, right_matrix)
    turn_rows = affine(row_masks, turn_matrix)
    source_rows = affine(row_masks, source_matrix)
    down = affine(row_masks, down_matrix)
    horizontal_signed = total(mul(turn_rows, hx))
    right, left = relu(horizontal_signed), relu(neg(horizontal_signed))
    horizontal = add(right, left)
    source_active = total(mul(source_rows, source_exists))
    normal_parent_rows = add(
        mul(normal_left_rows,
            bias(neg(mul(top_corner, relu(neg(hx)))), 1)),
        mul(normal_right_rows,
            bias(neg(mul(top_corner, relu(hx))), 1))
    )
    parent_rows = cat([normal_parent_rows, turn_rows])
    possible_emission = total(parent_rows)

    offset_input = cat([cell, source_x, source_y, pw, ph, shx, shy])
    n_offset = shapes[offset_input][1]
    wx = np.zeros((R, n_offset), np.float32)
    wy = np.zeros_like(wx)
    bx, by = np.zeros(R, np.float32), np.zeros(R, np.float32)
    ix, iy = 1, 1 + S
    iw, ih = 1 + 2 * S, 1 + 2 * S + P
    ihx, ihy = ih + P, ih + 2 * P
    for i in range(S):
        wx[i, ix + i] = 1
        wy[i, iy + i] = 1
    for i in range(P):
        r = S + 3 * i
        wx[r, 0], bx[r] = -.5, -.5
        wx[r + 1, 0] = .5
        wx[r + 1, iw + i] = 1
        bx[r + 1] = -.5
        wy[r:r + 2, 0] = -.5
        by[r:r + 2] = -.5
        wx[r + 2, iw + i] = .5
        wx[r + 2, ihx + i] = 1.5
        bx[r + 2] = -.5
        wy[r + 2, ih + i] = .5
        wy[r + 2, ihy + i] = .5
        by[r + 2] = -.5

    ray_offset_x = total(mul(row_masks, affine(offset_input, wx, bx)))
    ray_offset_y = total(mul(row_masks, affine(offset_input, wy, by)))
    hole_offset_x = add(half_pw, scale(shx, .5))
    all_width, all_height = cat([pw, cw]), cat([ph, ch])
    target_exists = cat([pexists, cexists])
    cup_x_sp, cup_y_sp = spatial(cx0), spatial(cy0)
    horizontal_trial = step(neg(screen_x), -31.5)
    vertical_trial = bias(neg(horizontal_trial), 1)
    trial_x = add(mul(screen_x, cell), pad)
    trial_y = add(mul(bias(screen_x, -32), cell), pad)

    mgx, mgy = sl("mem", 0, P), sl("mem", P, 2 * P)
    mcx, mcy = sl("mem", 2 * P, 3 * P), sl("mem", 3 * P, 4 * P)
    mtodo, mlast = sl("mem", 4 * P, 5 * P), sl("mem", 5 * P, 6 * P)
    mactive = sl("mem", 6 * P, 6 * P + 1)
    attempts = sl("mem", 6 * P + 1, 6 * P + 2)
    last_x = sl("mem", 6 * P + 2, 6 * P + 3)
    last_y = sl("mem", 6 * P + 3, 6 * P + 4)
    failed_before = step(attempts, .5)

    first_selector = constant([1] + [0] * (P - 1))
    initializers = {
        "qx": px0, "qy": py0,
        "bestx": px0, "besty": py0,
        "bestvalue": scalar(-1000000),
        "selector": first_selector,
        "clock": zero, "randomcol": scale(attempts, 7),
    }

    saved = current, cache, scope
    loop_body = []
    current, cache, scope = loop_body, {}, uid()
    states = {}
    for key, initial in initializers.items():
        sid = uid()
        states[key] = sid
        shapes[sid], owners[sid] = shapes[initial], scope

    qx, qy = states["qx"], states["qy"]
    selector = mul(states["selector"], pexists)
    candidate_x = add(
        spatial(qx),
        mul(selector, mul(horizontal_trial, sub(trial_x, qx)))
    )
    candidate_y = add(
        spatial(qy),
        mul(selector, mul(vertical_trial, sub(trial_y, qy)))
    )
    selected_exists = total(selector)
    test_x = total(mul(candidate_x, states["selector"]))
    test_y = total(mul(candidate_y, states["selector"]))
    test_w = total(mul(pw, states["selector"]))
    test_h = total(mul(ph, states["selector"]))
    test_right = bias(add(test_x, test_w), -1)
    test_bottom = bias(add(test_y, test_h), -1)

    valid = mul(
        step(sub(test_x, minimum_x), -.2),
        step(sub(maximum_x, test_right), -.2)
    )
    valid = mul(valid, step(sub(test_y, minimum_y), -.2))
    valid = mul(valid, step(sub(maximum_y, test_bottom), -.2))
    near_x = mul(
        step(sub(add(cx1, cell), test_x), -.2),
        step(sub(test_right, sub(cx0, cell)), -.2)
    )
    near_y = mul(
        step(sub(add(cy1, cell), test_y), -.2),
        step(sub(test_bottom, sub(cy0, cell)), -.2)
    )
    collision = step(total(mul(mul(near_x, near_y), cexists)), .5)
    valid = mul(valid, bias(neg(collision), 1))
    valid = add(
        mul(valid, selected_exists), bias(neg(selected_exists), 1)
    )

    # Penalize globally obstructed spreading corridors, not just overlaps.
    pair_dx = absolute(affine(add(candidate_x, half_pw), pair_difference))
    pair_dy = absolute(affine(add(candidate_y, half_ph), pair_difference))
    same_row = step(neg(pair_dy), -.2)
    nearby = mul(
        step(sub(pair_radius_x, pair_dx)),
        step(sub(pair_radius_y, pair_dy))
    )
    conflicts = total(mul(
        mul(nearby, pair_exists),
        bias(neg(mul(same_row, pair_bars)), 1)
    ))

    ray_x = add(total(mul(candidate_x, object_rows)), ray_offset_x)
    ray_y = add(total(mul(candidate_y, object_rows)), ray_offset_y)
    in_board = mul(
        step(sub(ray_x, minimum_x), -.2),
        step(sub(maximum_x, ray_x), -.2)
    )
    in_board = mul(in_board, step(sub(maximum_y, ray_y), -.2))
    retained = add(in_board, mul(side_walls, bias(neg(in_board), 1)))
    tx, ty = cat([candidate_x, cup_x_sp]), cat([candidate_y, cup_y_sp])
    tx1, ty1 = bias(add(tx, all_width), -1), bias(add(ty, all_height), -1)
    distance = add(
        mul(down, sub(ty, ray_y)),
        add(mul(right, sub(tx, ray_x)), mul(left, sub(ray_x, tx1)))
    )
    cross_vertical = mul(
        step(sub(ray_x, tx), -.2), step(sub(tx1, ray_x), -.2)
    )
    cross_horizontal = mul(
        step(sub(ray_y, ty), -.2), step(sub(ty1, ray_y), -.2)
    )
    cross = add(mul(down, cross_vertical), mul(horizontal, cross_horizontal))
    intersection = mul(mul(cross, step(distance, .2)), target_exists)
    ray_score = sub(
        neg(distance), scale(bias(neg(intersection), 1), 1024)
    )
    nearest = max_channels(ray_score)
    hit = mul(intersection, step(sub(ray_score, nearest), -.05))
    hit_platform, hit_cup = sl(hit, 0, P), sl(hit, P, P + C)
    turn_good = mul(
        equal(ray_x, add(candidate_x, hole_offset_x), .3), top_corner
    )
    downward_hit = mul(hit_platform, down)
    turn_hit = mul(downward_hit, top_corner)
    normal_hit = mul(downward_hit, bias(neg(turn_good), 1))
    adjacency = cat([normal_hit, turn_hit])

    active = mul(source_active, retained)
    for _ in range(P):
        arrived = emit("colmax", [mul(adjacency, active)])
        emitted = total(mul(arrived, parent_rows))
        active = mul(add(source_active, emitted), retained)

    compatible = add(
        mul(down, cup_down),
        add(mul(right, cup_right), mul(left, cup_left))
    )
    perpendicular = add(
        mul(down, absolute(sub(ray_x, cup_x))),
        mul(horizontal, absolute(sub(ray_y, cup_y)))
    )
    capture = mul(
        hit_cup, mul(compatible, step(neg(perpendicular), -.3))
    )
    filled = emit("colmax", [mul(capture, active)])
    filled_count = total(filled)
    hits_parent = step(total(downward_hit), .5)
    is_terminal = bias(neg(hits_parent), 1)
    terminal = mul(active, is_terminal)
    captured_here = step(total(capture), .5)
    leaking = mul(terminal, bias(neg(captured_here), 1))

    along_cup = add(
        mul(down, sub(cup_y, ray_y)),
        add(mul(right, sub(cup_x, ray_x)), mul(left, sub(ray_x, cup_x)))
    )
    miss_distance = add(
        perpendicular,
        add(
            mul(scale(cell, 6), bias(neg(compatible), 1)),
            relu(neg(along_cup))
        )
    )
    coverage_score = sub(
        neg(miss_distance), scale(bias(neg(terminal), 1), 10000)
    )
    coverage = neg(emit("colmax", [coverage_score]))
    coverage_cost = total(mul(coverage, cexists))
    nearest_cup_score = sub(
        neg(miss_distance), scale(bias(neg(cexists), 1), 10000)
    )
    nearest_cup_distance = neg(max_channels(nearest_cup_score))
    leak_cost = emit("colsum", [
        mul(leaking, add(nearest_cup_distance, scale(cell, 2)))
    ])

    # A weak downstream-alignment term resolves inactive-piece plateaus.
    potential_rays = mul(mul(possible_emission, retained), is_terminal)
    potential_cost = emit("colsum", [
        mul(potential_rays, nearest_cup_distance)
    ])
    movement = total(add(
        absolute(sub(candidate_x, px0)),
        absolute(sub(candidate_y, py0))
    ))
    unchanged_failure = mul(
        failed_before, step(neg(movement), -.2)
    )

    score = sub(
        sub(mul(filled_count, scale(cell, 30)), coverage_cost),
        add(scale(leak_cost, .5), scale(movement, .03))
    )
    score = sub(score, scale(potential_cost, .18))
    score = sub(score, scale(conflicts, 2000))
    score = sub(score, scale(unchanged_failure, 1000))
    score = sub(score, scale(bias(neg(valid), 1), 100000))
    score = sub(score, scale(screen_x, .01))

    best_trial_value = gm(score)
    best_mask = step(sub(score, best_trial_value), -.001, 20000)
    best_trial_x = gm(mul(candidate_x, best_mask))
    best_trial_y = gm(mul(candidate_y, best_mask))
    improved = step(
        sub(best_trial_value, states["bestvalue"]), .0001, 100000
    )
    new_best_x = choose(improved, best_trial_x, states["bestx"])
    new_best_y = choose(improved, best_trial_y, states["besty"])
    new_best_value = choose(
        improved, best_trial_value, states["bestvalue"]
    )

    explore = step(states["clock"], 9.5)
    exploration_score = sub(
        neg(absolute(sub(screen_x, states["randomcol"]))),
        add(scale(bias(neg(valid), 1), 2048), scale(screen_x, .01))
    )
    exploration_mask = step(
        sub(exploration_score, gm(exploration_score)), -.001, 20000
    )
    exploration_x = gm(mul(candidate_x, exploration_mask))
    exploration_y = gm(mul(candidate_y, exploration_mask))
    next_x = choose(explore, exploration_x, best_trial_x)
    next_y = choose(explore, exploration_y, best_trial_y)

    rotation_matrix = np.zeros((P, P), np.float32)
    for i in range(P):
        rotation_matrix[(i + 1) % P, i] = 1
    rotated_selector = mul(
        affine(states["selector"], rotation_matrix), pexists
    )
    next_selector = add(
        rotated_selector,
        mul(bias(neg(total(rotated_selector)), 1), first_selector)
    )
    next_clock = sub(bias(states["clock"], 1), scale(explore, 11))
    random_sum = bias(states["randomcol"], 17)
    next_random = sub(random_sum, scale(step(random_sum, 63.5), 64))
    next_values = {
        "qx": next_x, "qy": next_y,
        "bestx": new_best_x, "besty": new_best_y,
        "bestvalue": new_best_value, "selector": next_selector,
        "clock": next_clock, "randomcol": next_random,
    }

    current, cache, scope = saved
    planner_id = uid()
    planner = {
        "id": planner_id, "op": "loop", "T": 48,
        "state": [
            {"id": states[k], "init": initializers[k]} for k in initializers
        ],
        "body": loop_body,
        "next": {states[k]: next_values[k] for k in states},
    }
    root.append(planner)
    planned = {}
    for key, sid in states.items():
        name = planner_id + "." + sid
        shapes[name], owners[name] = shapes[sid], 0
        planned[key] = name

    # Execute a stored placement plan using click selection and unit moves.
    planned_x, planned_y = planned["bestx"], planned["besty"]
    new_todo = mul(
        step(add(
            absolute(sub(planned_x, px0)),
            absolute(sub(planned_y, py0))
        ), .2),
        pexists
    )
    goal_x = choose(mactive, mgx, planned_x)
    goal_y = choose(mactive, mgy, planned_y)
    click_x = choose(mactive, mcx, px0)
    click_y = choose(mactive, mcy, py0)
    todo = choose(mactive, mtodo, new_todo)

    first_matrix = np.eye(P, dtype=np.float32)
    for i in range(P):
        first_matrix[i, :i] = -1

    def first_pending(t):
        return relu(affine(t, first_matrix))

    selected_x, selected_y = sl(px0, 0, 1), sl(py0, 0, 1)
    selected_exists_now = sl(pexists, 0, 1)
    first = first_pending(todo)
    at_goal = mul(
        equal(selected_x, total(mul(first, goal_x))),
        equal(selected_y, total(mul(first, goal_y)))
    )
    at_goal = mul(at_goal, selected_exists_now)
    todo = sub(todo, mul(first, at_goal))
    stage = first_pending(todo)
    has_work = step(total(todo), .5)

    target_x = total(mul(stage, goal_x))
    target_y = total(mul(stage, goal_y))
    original_x = total(mul(stage, click_x))
    original_y = total(mul(stage, click_y))
    at_original = mul(
        equal(selected_x, original_x), equal(selected_y, original_y)
    )
    continued = mul(total(mul(stage, mlast)), mactive)
    engaged = mul(
        step(add(at_original, continued), .5), selected_exists_now
    )
    dx, dy = sub(target_x, selected_x), sub(target_y, selected_y)
    use_x = step(absolute(dx), .25)
    normal_up = mul(bias(neg(use_x), 1), step(neg(dy), .25))
    normal_down = mul(bias(neg(use_x), 1), step(dy, .25))
    normal_left = mul(use_x, step(neg(dx), .25))
    normal_right = mul(use_x, step(dx, .25))

    same_position = mul(
        equal(selected_x, last_x), equal(selected_y, last_y)
    )
    stalled = mul(
        mul(same_position, total(sl("aprev", 1, 5))),
        mul(mactive, engaged)
    )
    escape_down_flag = step(dy, .25)
    centre_x = scale(add(minimum_x, maximum_x), .5)
    escape_right_flag = step(sub(centre_x, selected_x), -.1)
    escape_up = mul(use_x, bias(neg(escape_down_flag), 1))
    escape_down = mul(use_x, escape_down_flag)
    escape_left = mul(
        bias(neg(use_x), 1), bias(neg(escape_right_flag), 1)
    )
    escape_right = mul(bias(neg(use_x), 1), escape_right_flag)

    normal_actions = cat([
        normal_up, normal_down, normal_left, normal_right
    ])
    escape_actions = cat([
        escape_up, escape_down, escape_left, escape_right
    ])
    canonical_action = mul(
        choose(stalled, escape_actions, normal_actions),
        mul(has_work, engaged)
    )
    permutations = [
        [1, 2, 3, 4], [3, 4, 2, 1],
        [2, 1, 4, 3], [4, 3, 1, 2],
    ]
    screen_actions = []
    for k, permutation in enumerate(permutations):
        w = np.zeros((4, 4), np.float32)
        for j, action in enumerate(permutation):
            w[action - 1, j] = 1
        screen_actions.append(
            mul(affine(canonical_action, w), sl(rotation, k, k + 1))
        )
    direction_logits = screen_actions[0]
    for action in screen_actions[1:]:
        direction_logits = add(direction_logits, action)

    pour = bias(neg(has_work), 1)
    clicking = mul(has_work, bias(neg(engaged), 1))
    action_values = cat([zero, direction_logits, pour, clicking, zero])
    affine(
        action_values, 20 * np.eye(8, dtype=np.float32),
        [-100, 0, 0, 0, 0, 0, 0, -100], out="logits"
    )

    click_centre_x = add(original_x, scale(bias(cell, -1), .5))
    click_centre_y = add(original_y, scale(bias(cell, -1), .5))
    click_distance = add(
        absolute(sub(X, click_centre_x)),
        absolute(sub(Y, click_centre_y))
    )
    affine(click_distance, [[-1]], out="click")
    memory_value = cat([
        goal_x, goal_y, click_x, click_y, todo, stage,
        has_work, add(attempts, pour), selected_x, selected_y,
    ])
    emit("vadd", [memory_value, zero], out="mem_next")

    # Remove unused algebraic intermediates, including within loop bodies.
    def prune(nodes, wanted):
        needed = set(wanted)
        kept = []
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                outputs = {
                    nd["id"] + "." + s["id"] for s in nd["state"]
                }
                if not outputs.intersection(needed):
                    continue
                nd["body"] = prune(nd["body"], set(nd["next"].values()))
                internal = {s["id"] for s in nd["state"]}
                internal.update(n["id"] for n in nd["body"])
                for state in nd["state"]:
                    if "init" in state:
                        needed.add(state["init"])
                for node in nd["body"]:
                    needed.update(
                        x for x in node.get("in", []) if x not in internal
                    )
                needed.update(
                    x for x in nd["next"].values() if x not in internal
                )
                kept.append(nd)
            elif nd["id"] in needed:
                needed.update(nd.get("in", []))
                kept.append(nd)
        return list(reversed(kept))

    graph = prune(root, {"logits", "click", "mem_next"})

    def statistics(nodes):
        macs, evaluations = 0, 0
        for nd in nodes:
            op = nd["op"]
            if op == "loop":
                bm, be = statistics(nd["body"])
                macs += nd["T"] * bm
                evaluations += 1 + nd["T"] * be
                continue
            evaluations += 1
            if op == "conv":
                macs += int(np.prod(params[nd["w"]].shape)) * 4096
            elif op == "dense":
                macs += int(np.prod(params[nd["w"]].shape))
            elif op in ("add", "sub", "mul", "max2", "min2"):
                macs += shapes[nd["id"]][1] * 4096
            elif op in (
                "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"
            ):
                macs += shapes[nd["in"][0]][1] * 4096
            elif op in ("vadd", "vmul"):
                macs += shapes[nd["id"]][1]
        return macs, evaluations

    # The iteration count is determined only by static execution costs.
    body_macs, body_evaluations = statistics(planner["body"])
    all_macs, all_evaluations = statistics(graph)
    fixed_macs = all_macs - planner["T"] * body_macs
    fixed_evaluations = all_evaluations - planner["T"] * body_evaluations
    max_by_macs = (1_490_000_000 - fixed_macs) // max(1, body_macs)
    max_by_evaluations = (
        (9900 - fixed_evaluations) // max(1, body_evaluations)
    )
    planner["T"] = int(max(
        1, min(48, max_by_macs, max_by_evaluations)
    ))

    used_parameters = set()

    def collect(nodes):
        for nd in nodes:
            if nd["op"] == "loop":
                collect(nd["body"])
            else:
                for key in ("w", "b"):
                    if key in nd:
                        used_parameters.add(nd[key])

    collect(graph)
    params = {k: v for k, v in params.items() if k in used_parameters}
    return {"graph": graph, "memory_size": M, "params": params}
