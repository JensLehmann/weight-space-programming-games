import numpy as np


def build():
    P = {}
    cache = {}
    graph = []
    current = [graph]
    serial = [0]
    M = 233
    dims = {
        "x0": (16, True),
        "coords": (3, True),
        "aprev": (8, False),
        "avail": (8, False),
        "mem": (M, False),
    }

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p" + str(len(P))
            cache[key] = name
            P[name] = a
        return cache[key]

    def emit(op, ins, name=None, **fields):
        name = fresh() if name is None else name
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        current[0].append(nd)

        if op in ("conv", "dense"):
            d = (P[fields["w"]].shape[0], op == "conv")
        elif op in ("slice", "vslice"):
            d = (fields["stop"] - fields["start"], op == "slice")
        elif op in ("concat", "vconcat"):
            d = (sum(dims[x][0] for x in ins), op == "concat")
        elif op in ("gmax", "gmean"):
            d = (dims[ins[0]][0], False)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            d = (max(dims[x][0] for x in ins), True)
        elif op in ("vadd", "vmul"):
            d = (max(dims[x][0] for x in ins), False)
        else:
            d = dims[ins[0]]
        dims[name] = d
        return name

    def linear(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        spatial = dims[x][1]
        if spatial and w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            b = np.broadcast_to(
                np.asarray(b, dtype=np.float32), (w.shape[0],)
            ).copy()
            fields["b"] = parameter(b)
        return emit(
            "conv" if spatial else "dense",
            [x], name=name, **fields
        )

    def scale(x, a, b=0.0):
        return linear(x, np.eye(dims[x][0], dtype=np.float32) * a, b)

    def const(a):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        return linear("aprev", np.zeros((len(a), 8), np.float32), a)

    def sl(x, start, stop):
        return emit(
            "slice" if dims[x][1] else "vslice",
            [x], start=start, stop=stop
        )

    def cat(xs, name=None):
        return emit(
            "concat" if dims[xs[0]][1] else "vconcat",
            xs, name=name
        )

    def relu(x):
        return emit("relu" if dims[x][1] else "vrelu", [x])

    def va(x, y):
        return emit("vadd", [x, y])

    def vm(x, y):
        return emit("vmul", [x, y])

    def vs(x, y):
        return va(x, scale(y, -1))

    def sa(x, y):
        return emit("add", [x, y])

    def ss(x, y):
        return emit("sub", [x, y])

    def sm(x, y):
        return emit("mul", [x, y])

    def mx(x, y):
        return emit("max2", [x, y])

    def gm(x):
        return emit("gmax", [x])

    one = const([1])
    zero = const([0])

    def clipv(x):
        return scale(relu(scale(relu(x), -1, 1)), -1, 1)

    def clips(x):
        return emit("min2", [relu(x), one])

    def absv(x):
        return va(relu(x), relu(scale(x, -1)))

    def positive(x, threshold=0.5):
        return clipv(scale(x, 4, -4 * threshold))

    def mix(a, b, gate):
        return sa(sm(a, scale(gate, -1, 1)), sm(b, gate))

    def vmix(a, b, gate):
        return va(vm(a, scale(gate, -1, 1)), vm(b, gate))

    def loop(T, initial, fn):
        outer = current[0]
        body = []
        states = []
        ids = []
        for init in initial:
            sid = fresh("s")
            ids.append(sid)
            if isinstance(init, str):
                states.append({"id": sid, "init": init})
                dims[sid] = dims[init]
            else:
                shape = list(init)
                states.append({"id": sid, "zeros": shape})
                dims[sid] = (shape[0], len(shape) == 3)
        current[0] = body
        nxt = fn(ids)
        current[0] = outer
        lid = fresh("L")
        outer.append({
            "id": lid,
            "op": "loop",
            "T": T,
            "state": states,
            "body": body,
            "next": dict(zip(ids, nxt)),
        })
        result = []
        for sid, out in zip(ids, nxt):
            assert dims[sid] == dims[out]
            ref = lid + "." + sid
            dims[ref] = dims[sid]
            result.append(ref)
        return result

    def shift(x, dx=0, dy=0):
        c = dims[x][0]
        if c > 2:
            return cat([
                shift(sl(x, i, min(i + 2, c)), dx, dy)
                for i in range(0, c, 2)
            ])
        w = np.zeros((c, c, 7, 7), np.float32)
        for i in range(c):
            w[i, i, 3 - dy, 3 - dx] = 1
        return linear(x, w)

    # Memory: normalized 15x15 silhouette, position, route and flags.
    old_x = sl("mem", 225, 226)
    old_y = sl("mem", 226, 227)
    old_valid = sl("mem", 227, 228)
    old_dest_x = sl("mem", 228, 229)
    old_dest_y = sl("mem", 229, 230)
    old_allow_x = sl("mem", 230, 231)
    old_allow_y = sl("mem", 231, 232)
    old_idle = sl("mem", 232, 233)

    prev_up = sl("aprev", 1, 2)
    prev_down = sl("aprev", 2, 3)
    prev_left = sl("aprev", 3, 4)
    prev_right = sl("aprev", 4, 5)
    prev_cycle = sl("aprev", 5, 6)
    prev_move = va(va(prev_up, prev_down), va(prev_left, prev_right))
    carry = vm(old_valid, prev_move)
    idle = vm(prev_cycle, scale(old_idle, 1, 1))
    idle = scale(clipv(scale(idle, 0.125)), 8)
    escape = clipv(scale(idle, 1, -3))

    # Canonical rendering: centers of the 21x21 game cells.
    w = np.zeros((42, 3), np.float32)
    b = np.zeros(42, np.float32)
    for i in range(21):
        w[i, 0] = 63
        w[21 + i, 1] = 63
        b[i] = b[21 + i] = -(1 + 3 * i)
    delta = linear("coords", w, b)
    distance = emit("abs", [delta])
    hats = relu(scale(relu(scale(distance, 8, -1)), -1, 1))
    row_basis = sl(hats, 0, 21)
    col_basis = sl(hats, 21, 42)
    row_grid = linear(row_basis, np.ones((1, 21), np.float32))
    col_grid = linear(col_basis, np.ones((1, 21), np.float32))
    grid = sm(row_grid, col_grid)
    first_row = sl(row_basis, 0, 1)
    first_col = sl(col_basis, 0, 1)

    w = np.zeros((4, 42), np.float32)
    for i in range(21):
        w[0, 21 + i] = i + 1
        w[1, 21 + i] = 21 - i
        w[2, i] = i + 1
        w[3, i] = 21 - i
    codes = linear(hats, w)
    xcode = sl(codes, 0, 1)
    ycode = sl(codes, 2, 3)

    # Expand remembered geometry onto the lattice.
    cols15 = sl(col_basis, 0, 15)
    memory_rows = [sl(row_basis, i, i + 1) for i in range(15)]
    decoded_rows = []
    for i in range(15):
        bits = sl("mem", 15 * i, 15 * (i + 1))
        row = linear(sm(cols15, bits), np.ones((1, 15), np.float32))
        decoded_rows.append(sm(row, memory_rows[i]))
    stored_template = decoded_rows[0]
    for row in decoded_rows[1:]:
        stored_template = sa(stored_template, row)

    # Corners retain body/ghost/axis colors; centers contain UI marks.
    w = np.zeros((3, 16, 3, 3), np.float32)
    for color in (1, 2, 3, 5, 6, 7, 8, 12, 13, 14, 15):
        w[0, color, 0, 0] = 1
    w[1, 4, 0, 0] = 1
    w[2, 10, 0, 0] = 1
    corners = linear("x0", w)
    original = sm(sl(corners, 0, 1), grid)
    ghosts = sm(sl(corners, 1, 2), grid)
    axes = sm(sl(corners, 2, 3), grid)
    occupied = sa(original, ghosts)

    c0 = sl("x0", 0, 1)
    c9 = sl("x0", 9, 10)
    cb = sl("x0", 11, 12)
    ca = sl("x0", 10, 11)
    target = sm(cb, grid)

    vline = clips(scale(emit("colsum", [axes]), 1, -4))
    hline = clips(scale(emit("rowsum", [axes]), 1, -4))
    has_x = gm(vline)
    has_y = gm(hline)
    axis_x = vs(gm(sm(vline, xcode)), has_x)
    axis_y = vs(gm(sm(hline, ycode)), has_y)

    selected_axis_pixels = sm(axes, c0)
    selected_x = clipv(scale(
        gm(emit("colsum", [selected_axis_pixels])), 1, -3
    ))
    selected_y = clipv(scale(
        gm(emit("rowsum", [selected_axis_pixels])), 1, -3
    ))

    # Movable mirrors have selected/unselected center marks (0 or 9).
    # Fixed mirrors retain color 10 and must never be requested.
    selectable_axis_pixels = sm(axes, sa(c0, c9))
    movable_x = clipv(scale(
        gm(emit("colsum", [selectable_axis_pixels])), 1, -3
    ))
    movable_y = clipv(scale(
        gm(emit("rowsum", [selectable_axis_pixels])), 1, -3
    ))

    seed = sm(original, c0)
    traversable = sm(original, clips(sa(sa(c0, cb), ca)))
    flood_w = np.zeros((1, 1, 7, 7), np.float32)
    for dy, dx in ((0, 0), (-3, 0), (3, 0), (0, -3), (0, 3)):
        flood_w[0, 0, 3 + dy, 3 + dx] = 1

    def flood_body(s):
        reach = clips(linear(s[0], flood_w))
        return [sm(reach, traversable)]

    observed_selected = loop(32, [seed], flood_body)[0]

    target_extents = gm(sm(target, codes))
    target_center = linear(
        target_extents,
        np.array([[0.5, -0.5, 0, 0],
                  [0, 0, 0.5, -0.5]], np.float32),
        [10, 10],
    )
    target_x = sl(target_center, 0, 1)
    target_y = sl(target_center, 1, 2)

    # Global axis scheduling, independent of the current selection.
    axis_dx = vs(target_x, axis_x)
    axis_dy = vs(target_y, axis_y)
    need_x = vm(positive(absv(axis_dx), 0.51), movable_x)
    need_y = vm(positive(absv(axis_dy), 0.51), movable_y)
    need_axis = clipv(va(need_x, need_y))
    selected_needs_motion = clipv(va(
        vm(need_x, selected_x), vm(need_y, selected_y)
    ))
    request_click = vm(need_axis, scale(selected_needs_motion, -1, 1))

    click_axis_scores = sa(
        sm(vline, need_x),
        sm(hline, scale(need_y, 2)),
    )
    click_scores = sm(selectable_axis_pixels, click_axis_scores)
    linear(click_scores, np.ones((1, 1), np.float32), name="click")

    def reverse_image(x, horizontal):
        c = dims[x][0]
        boundary = first_col if horizontal else first_row

        def body(s):
            source, accumulator = s
            edge = sm(source, boundary)
            if horizontal:
                source_next = shift(source, dx=-3)
                acc_shifted = shift(accumulator, dx=3)
            else:
                source_next = shift(source, dy=-3)
                acc_shifted = shift(accumulator, dy=3)
            return [source_next, sa(acc_shifted, edge)]

        return loop(21, [x, [c, 64, 64]], body)[1]

    # Canonical orientation makes reflection scans monotone.
    flip_x = clipv(scale(axis_x, 1, -10))
    flip_y = clipv(scale(axis_y, 1, -10))
    sign_x = scale(flip_x, -2, 1)
    sign_y = scale(flip_y, -2, 1)
    canonical_x = va(axis_x, vm(flip_x, scale(axis_x, -2, 20)))
    canonical_y = va(axis_y, vm(flip_y, scale(axis_y, -2, 20)))

    packed = cat([observed_selected, target, occupied, seed, original])
    packed = mix(packed, reverse_image(packed, True), flip_x)
    packed = mix(packed, reverse_image(packed, False), flip_y)
    observed_c = sl(packed, 0, 1)
    target_c = sl(packed, 1, 2)
    occupied_c = sl(packed, 2, 3)
    seed_c = sl(packed, 3, 4)
    original_c = sl(packed, 4, 5)

    # Predict selected geometry from memory and the previous action.
    predicted_x = va(old_x, vm(vs(prev_right, prev_left), sign_x))
    predicted_y = va(old_y, vm(vs(prev_down, prev_up), sign_y))

    def place_memory_body(s):
        image, t = s
        gx = clipv(vs(predicted_x, t))
        gy = clipv(vs(predicted_y, t))
        image = mix(image, shift(image, dx=3), gx)
        image = mix(image, shift(image, dy=3), gy)
        return [image, scale(t, 1, 1)]

    predicted = loop(20, [stored_template, zero], place_memory_body)[0]
    contradicted = gm(relu(ss(seed_c, predicted)))
    keep = vm(carry, scale(contradicted, -1, 1))
    keep = vm(keep, scale(selected_x, -1, 1))
    keep = vm(keep, scale(selected_y, -1, 1))
    selected_c = mix(observed_c, predicted, keep)

    shape_active = clipv(va(gm(seed), keep))
    shape_active = vm(shape_active, scale(selected_x, -1, 1))
    shape_active = vm(shape_active, scale(selected_y, -1, 1))

    ext = gm(sm(selected_c, codes))
    box = linear(
        ext,
        np.array([
            [0, -1, 0, 0],
            [0, 0, 0, -1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], np.float32),
        [21, 21, -21, -21],
    )
    current_x = sl(box, 0, 1)
    current_y = sl(box, 1, 2)
    width = sl(box, 2, 3)
    height = sl(box, 3, 4)

    def normalize_body(s):
        image = s[0]
        at_left = gm(sm(image, first_col))
        image = mix(shift(image, dx=-3), image, at_left)
        at_top = gm(sm(image, first_row))
        image = mix(shift(image, dy=-3), image, at_top)
        return [image]

    normalized = loop(20, [selected_c], normalize_body)[0]
    template = mix(normalized, stored_template, keep)

    def align_reflection(x, amount, horizontal):
        def body(s):
            image, t = s
            gate = clipv(vs(amount, t))
            moved = (
                shift(image, dx=-3) if horizontal
                else shift(image, dy=-3)
            )
            return [mix(image, moved, gate), scale(t, 1, 1)]

        return loop(20, [x, zero], body)[0]

    # Reflect other originals separately, preserving overlap coverage.
    other_original = relu(ss(original_c, selected_c))
    triplet = cat([selected_c, template, other_original])

    rx = reverse_image(triplet, True)
    rx = align_reflection(rx, scale(canonical_x, -2, 20), True)
    rx = sm(rx, has_x)

    joined = cat([triplet, rx])
    ry = reverse_image(joined, False)
    ry = align_reflection(ry, scale(canonical_y, -2, 20), False)
    ry = sm(ry, has_y)

    rx_current = sl(rx, 0, 1)
    rx_template = sl(rx, 1, 2)
    rx_other = sl(rx, 2, 3)
    ry_current = sl(ry, 0, 1)
    ry_template = sl(ry, 1, 2)
    ry_other = sl(ry, 2, 3)
    rxy_current = sl(ry, 3, 4)
    rxy_template = sl(ry, 4, 5)
    rxy_other = sl(ry, 5, 6)

    inferred_x = scale(gm(relu(ss(rx_current, occupied_c))), -1, 1)
    inferred_y = scale(gm(relu(ss(ry_current, occupied_c))), -1, 1)
    allow_x = vmix(inferred_x, old_allow_x, keep)
    allow_y = vmix(inferred_y, old_allow_y, keep)
    allow_xy = vm(allow_x, allow_y)

    projection = mx(selected_c, sm(rx_current, allow_x))
    projection = mx(projection, sm(ry_current, allow_y))
    projection = mx(projection, sm(rxy_current, allow_xy))

    other_coverage = mx(other_original, rx_other)
    other_coverage = mx(other_coverage, ry_other)
    other_coverage = mx(other_coverage, rxy_other)
    other_coverage = emit("min2", [other_coverage, occupied_c])

    residual = sm(target_c, ss(one, other_coverage))
    current_target_coverage = sm(projection, target_c)

    # Parallel horizontal placement candidates and reflected counterparts.
    quad = cat([
        template,
        sm(rx_template, allow_x),
        sm(ry_template, allow_y),
        sm(rxy_template, allow_xy),
    ])
    horizontal_w = np.zeros((4, 4, 7, 7), np.float32)
    for i, dx in enumerate((3, -3, 3, -3)):
        horizontal_w[i, i, 3, 3 - dx] = 1

    originals = []
    reflections = []
    for i in range(21):
        a, b, c, d = [sl(quad, j, j + 1) for j in range(4)]
        originals.append(mx(a, b))
        reflections.append(mx(c, d))
        if i != 20:
            quad = linear(quad, horizontal_w)

    originals = cat(originals)
    reflections = cat(reflections)

    xi = const(np.arange(21, dtype=np.float32))
    x_distance = absv(vs(xi, current_x))
    invalid_x = clipv(scale(va(xi, width), 1, -21))
    x_penalty = va(scale(x_distance, -8), scale(xi, -0.125))
    x_penalty = va(x_penalty, scale(invalid_x, -1000000))

    initial_best = const(np.full(21, -100000, np.float32))
    initial_y = const(np.zeros(21, np.float32))

    def placement_body(s):
        orig, refl, best, best_y, t = s
        candidate = mx(orig, refl)
        counts = emit("gmean", [sm(candidate, residual)])
        score = scale(counts, 4096 * 1024)

        # Escape repeated selection-only plateaus by changing coverage.
        candidate_targets = sm(candidate, target_c)
        changed = emit(
            "abs", [ss(candidate_targets, current_target_coverage)]
        )
        changed = clipv(scale(emit("gmean", [changed]), 4096))
        unchanged = scale(changed, -1, 1)
        score = va(score, scale(vm(unchanged, escape), -500000))

        y_distance = absv(vs(t, current_y))
        invalid_y = clipv(scale(va(t, height), 1, -21))
        y_penalty = va(
            scale(y_distance, -8),
            scale(invalid_y, -1000000),
        )
        score = va(va(score, x_penalty), y_penalty)

        improvement = vs(score, best)
        better = clipv(scale(improvement, 8))
        best_next = va(best, relu(improvement))
        best_y_next = va(best_y, vm(better, vs(t, best_y)))

        orig_next = cat([
            shift(sl(orig, i, i + 1), dy=3) for i in range(21)
        ])
        refl_next = cat([
            shift(sl(refl, i, i + 1), dy=-3) for i in range(21)
        ])
        return [
            orig_next, refl_next, best_next, best_y_next,
            scale(t, 1, 1),
        ]

    result = loop(
        21,
        [originals, reflections, initial_best, initial_y, zero],
        placement_body,
    )
    best, best_y = result[2], result[3]

    shifted_best = scale(best, 1, 100000)
    score_map = linear(
        sm(col_basis, shifted_best), np.ones((1, 21), np.float32)
    )
    maximum = gm(score_map)
    winner = relu(scale(vs(maximum, shifted_best), -8, 1))
    planned_x = linear(
        winner, np.arange(21, dtype=np.float32)[None, :]
    )
    planned_y = linear(
        vm(winner, best_y), np.ones((1, 21), np.float32)
    )
    plan_exists = clipv(maximum)
    planned_x = vmix(current_x, planned_x, plan_exists)
    planned_y = vmix(current_y, planned_y, plan_exists)

    destination_x = vmix(planned_x, old_dest_x, keep)
    destination_y = vmix(planned_y, old_dest_y, keep)
    dx = vm(vs(destination_x, current_x), sign_x)
    dy = vm(vs(destination_y, current_y), sign_y)

    piece_up = vm(positive(scale(dy, -1)), shape_active)
    piece_down = vm(positive(dy), shape_active)
    piece_left = vm(positive(scale(dx, -1)), shape_active)
    piece_right = vm(positive(dx), shape_active)

    axis_up = vm(positive(scale(axis_dy, -1), 0.51), selected_y)
    axis_down = vm(positive(axis_dy, 0.51), selected_y)
    axis_left = vm(positive(scale(axis_dx, -1), 0.51), selected_x)
    axis_right = vm(positive(axis_dx, 0.51), selected_x)

    commands = cat([
        piece_up, piece_down, piece_left, piece_right,
        axis_up, axis_down, axis_left, axis_right,
        request_click,
    ])
    w = np.zeros((8, 9), np.float32)
    for i in range(4):
        w[i + 1, i] = 8
        w[i + 1, i + 4] = 10
    w[6, 8] = 120
    linear(
        commands, w,
        np.array([-100, 0, 0, 0, 0, 1, -100, -100], np.float32),
        name="logits",
    )

    encoded = []
    for row in memory_rows:
        encoded.append(gm(sm(sm(template, row), cols15)))
    fits_memory = vm(
        clipv(scale(width, -1, 16)),
        clipv(scale(height, -1, 16)),
    )
    memory_valid = vm(shape_active, fits_memory)
    cat(
        encoded + [
            current_x, current_y, memory_valid,
            destination_x, destination_y,
            allow_x, allow_y, idle,
        ],
        name="mem_next",
    )

    # Static construction checks only; no game simulation or weight fitting.
    def measures(nodes):
        node_count = 0
        evaluations = 0
        macs = 0
        for nd in nodes:
            node_count += 1
            evaluations += 1
            op = nd["op"]
            if op == "loop":
                n, e, m = measures(nd["body"])
                node_count += n
                evaluations += nd["T"] * e
                macs += nd["T"] * m
                continue
            out_n, _ = dims[nd["id"]]
            if op == "conv":
                macs += int(P[nd["w"]].size) * 4096
            elif op == "dense":
                macs += int(P[nd["w"]].size)
            elif op in ("add", "sub", "mul", "max2", "min2"):
                macs += out_n * 4096
            elif op in (
                "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"
            ):
                macs += dims[nd["in"][0]][0] * 4096
            elif op in ("vadd", "vmul"):
                macs += out_n
        return node_count, evaluations, macs

    nodes, evaluations, macs = measures(graph)
    assert nodes <= 1000
    assert evaluations <= 10000
    assert macs <= 1500000000
    assert sum(a.size for a in P.values()) <= 150000
    assert sum(n for n, spatial in dims.values() if spatial) <= 4096
    assert all(
        n <= (512 if spatial else 4096)
        for n, spatial in dims.values()
    )
    assert dims["mem_next"] == (M, False)
    assert all(np.isfinite(a).all() for a in P.values())

    return {"graph": graph, "memory_size": M, "params": P}
