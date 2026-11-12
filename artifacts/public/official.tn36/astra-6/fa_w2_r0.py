import numpy as np


def build():
    P = {}
    cache = {}
    graph = []
    active = graph
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (37,),
        "aprev": (8,),
        "avail": (8,),
    }
    serial = [0]

    def parameter(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p" + str(len(P))
            cache[key] = name
            P[name] = a
        return cache[key]

    def emit(op, ins, name=None, **kw):
        if name is None:
            name = "n" + str(serial[0])
            serial[0] += 1
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(kw)
        ss = [shapes[i] for i in ins]
        if op == "conv":
            out = (P[kw["w"]].shape[0], 64, 64)
        elif op == "dense":
            out = (P[kw["w"]].shape[0],)
        elif op in ("concat", "vconcat"):
            out = (sum(s[0] for s in ss),) + ss[0][1:]
        elif op in ("slice", "vslice"):
            out = (kw["stop"] - kw["start"],) + ss[0][1:]
        elif op in ("gmax", "gmean"):
            out = (ss[0][0],)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            out = (max(s[0] for s in ss), 64, 64)
        elif op in ("vadd", "vmul"):
            out = (max(s[0] for s in ss),)
        else:
            out = ss[0]
        shapes[name] = out
        active.append(nd)
        return name

    def lin(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        spatial = len(shapes[x]) == 3
        if spatial and w.ndim == 2:
            w = w[:, :, None, None]
        kw = {"w": parameter(w)}
        if b is not None:
            bb = np.asarray(b, dtype=np.float32)
            if bb.ndim == 0:
                bb = np.full(w.shape[0], bb, np.float32)
            kw["b"] = parameter(bb)
        return emit("conv" if spatial else "dense", [x], name=name, **kw)

    def aff(x, gain=1.0, bias=0.0, name=None):
        c = shapes[x][0]
        return lin(x, np.eye(c, dtype=np.float32) * gain,
                   np.full(c, bias, np.float32), name)

    def sl(x, start, stop):
        return emit("slice" if len(shapes[x]) == 3 else "vslice",
                    [x], start=start, stop=stop)

    def cat(xs):
        xs = list(xs)
        if len(xs) == 1:
            return xs[0]
        return emit("concat" if len(shapes[xs[0]]) == 3 else "vconcat", xs)

    def add(a, b):
        vector = len(shapes[a]) == len(shapes[b]) == 1
        return emit("vadd" if vector else "add", [a, b])

    def mul(a, b):
        vector = len(shapes[a]) == len(shapes[b]) == 1
        return emit("vmul" if vector else "mul", [a, b])

    def sub(a, b):
        if len(shapes[a]) == len(shapes[b]) == 1:
            return add(a, aff(b, -1))
        return emit("sub", [a, b])

    def relu(x):
        return emit("relu" if len(shapes[x]) == 3 else "vrelu", [x])

    def absolute(x):
        return emit("abs", [x])

    def pool(x):
        return emit("gmax", [x])

    def total(xs):
        xs = list(xs)
        z = xs[0]
        for x in xs[1:]:
            z = add(z, x)
        return z

    def sum_channels(x):
        return lin(x, np.ones((1, shapes[x][0]), np.float32))

    def constant(v):
        v = np.asarray(v, np.float32).reshape(-1)
        return lin("avail", np.zeros((len(v), 8), np.float32), v)

    one = constant([1.0])

    def snap(x):
        # Exact binary output for inputs close to either zero or one.
        return emit("min2", [relu(aff(x, 4, -2)), one])

    def binary(x, threshold=0.5):
        c = shapes[x][0]
        eye = np.eye(c, dtype=np.float32)
        w = np.concatenate([4 * eye, 4 * eye], axis=0)
        b = np.concatenate([
            np.full(c, -4 * threshold, np.float32),
            np.full(c, -4 * threshold - 1, np.float32),
        ])
        h = relu(lin(x, w, b))
        return lin(h, np.concatenate([eye, -eye], axis=1))

    def equality_vector(scalar, values):
        # Triangular cardinal functions, exact on integer inputs.
        values = np.asarray(values, np.float32)
        n = len(values)
        w = np.ones((3 * n, 1), np.float32)
        b = np.concatenate([1 - values, -values, -1 - values])
        h = relu(lin(scalar, w, b))
        out = np.zeros((n, 3 * n), np.float32)
        for i in range(n):
            out[i, i] = 1
            out[i, n + i] = -2
            out[i, 2 * n + i] = 1
        return lin(h, out)

    def shift(x, dy, dx):
        # Output(r,c) = input(r+dy,c+dx).
        y = x
        dy, dx = int(dy), int(dx)
        while dy or dx:
            m = max(abs(dy), abs(dx))
            cap = 2 if m > 3 else 3
            sy = int(np.sign(dy)) * min(abs(dy), cap)
            sx = int(np.sign(dx)) * min(abs(dx), cap)
            k = 2 * max(abs(sy), abs(sx)) + 1
            c = shapes[y][0]
            w = np.zeros((c, c, k, k), np.float32)
            for j in range(c):
                w[j, j, k // 2 + sy, k // 2 + sx] = 1
            y = lin(y, w)
            dy -= sy
            dx -= sx
        return y

    xy = lin("coords", [[63, 0, 0], [0, 63, 0]])
    black = sl("x0", 5, 6)
    zero_colour = sl("x0", 0, 1)

    def extrema(mask):
        p = aff(mask, 128, -128)
        lo = aff(pool(sub(p, xy)), -1)
        hi = pool(add(p, xy))
        return lo, hi

    def point_mask(position):
        # position begins with (row, column).
        n = shapes[position][0]
        w = np.zeros((2, n), np.float32)
        w[0, 0] = -1
        w[1, 1] = -1
        d = absolute(add(xy, lin(position, w)))
        return snap(relu(lin(d, [[-1, -1]], [1])))

    # ----- Detect the editable binary-program bars. -----
    wb = np.zeros((4, 16, 5, 5), np.float32)
    for orientation in range(2):
        if orientation == 0:
            signal = [(2, 1), (2, 2), (2, 3)]
            surround = [(r, c) for r in (1, 3) for c in range(5)]
            surround += [(2, 0), (2, 4)]
        else:
            signal = [(1, 2), (2, 2), (3, 2)]
            surround = [(r, c) for c in (1, 3) for r in range(5)]
            surround += [(0, 2), (4, 2)]
        for q, colour in enumerate((1, 5)):
            o = 2 * orientation + q
            for r, c in signal:
                wb[o, colour, r, c] = 1
            for r, c in surround:
                wb[o, 0, r, c] = 1
    bars4 = aff(relu(lin("x0", wb, np.full(4, -14.5))), 2)
    bars = sum_channels(bars4)

    # UI bars are in the lower program-panel band.
    band_d = relu(lin(xy, [[-1, 0], [1, 0]], [31, -50]))
    band = snap(relu(lin(band_d, [[-1, -1]], [1])))
    bars = mul(bars, band)
    origin, last = extrema(bars)

    ob = cat([origin, last])
    counts = lin(ob, [
        [-1 / 3, 0, 1 / 3, 0],
        [0, -1 / 5, 0, 1 / 5],
    ], [1, 1])
    bit_count = sl(counts, 0, 1)
    column_count = sl(counts, 1, 2)
    six_bits = binary(bit_count, 4)
    n_onehot = binary(equality_vector(column_count, range(1, 7)))
    n_flags = [sl(n_onehot, i, i + 1) for i in range(6)]

    # Thirty-six possible bar-centre masks: column-major, low bit first.
    w = np.zeros((12, 2), np.float32)
    b = np.zeros(12, np.float32)
    wo = np.zeros((12, 2), np.float32)
    for j in range(6):
        w[j, 0] = 1
        b[j] = -3 * j
        wo[j, 0] = -1
        w[6 + j, 1] = 1
        b[6 + j] = -5 * j
        wo[6 + j, 1] = -1
    distances = absolute(add(lin(xy, w, b), lin(origin, wo)))
    wslot = np.zeros((36, 12), np.float32)
    for i in range(6):
        for j in range(6):
            wslot[6 * i + j, j] = -1
            wslot[6 * i + j, 6 + i] = -1
    slots = snap(relu(lin(distances, wslot, np.ones(36))))
    present = binary(pool(mul(slots, bars)))
    observed_bits = binary(pool(mul(slots, black)))

    # ----- Recover board bounds from the black program-panel frame. -----
    row_diff = add(lin(xy, [[1, 0]], [1]),
                   lin(origin, [[-1, 0]]))
    frame_row = snap(relu(aff(absolute(row_diff), -1, 1)))
    frame = mul(frame_row, zero_colour)
    frame_lo, frame_hi = extrema(frame)
    left = sl(frame_lo, 1, 2)
    right = sl(frame_hi, 1, 2)
    r0 = sl(origin, 0, 1)
    edgevec = cat([left, right, r0])

    # Interior-column and above-UI mask for finding the checkerboard top.
    wc = np.array([[0, -1], [0, 1], [1, 0]], np.float32)
    we = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                  np.float32)
    ctest = relu(add(lin(xy, wc, [1, 1, 2]), lin(edgevec, we)))
    interior = snap(relu(lin(ctest, [[-1, -1, -1]], [1])))
    checker = mul(interior, sl("x0", 4, 5))
    checker_lo, _ = extrema(checker)
    top = sl(checker_lo, 0, 1)
    geometry = cat([left, right, top, r0])

    # Legal top-left positions for 4, 8, 12, and 16 pixel bodies.
    wxy = np.zeros((16, 2), np.float32)
    wg = np.zeros((16, 4), np.float32)
    bb = np.zeros(16, np.float32)
    for s in range(1, 5):
        k = 4 * (s - 1)
        wxy[k, 1], wg[k, 0], bb[k] = -1, 1, 1
        wxy[k + 1, 1], wg[k + 1, 1] = 1, -1
        bb[k + 1] = 4 * s
        wxy[k + 2, 0], wg[k + 2, 2] = -1, 1
        wxy[k + 3, 0], wg[k + 3, 3] = 1, -1
        bb[k + 3] = 4 * s + 1
    violations = relu(add(lin(xy, wxy, bb), lin(geometry, wg)))
    collapse = np.zeros((4, 16), np.float32)
    for s in range(4):
        collapse[s, 4 * s:4 * s + 4] = -1
    bounds = snap(relu(lin(violations, collapse, np.ones(4))))

    # Conservative obstacle plane; expand to each body footprint.
    wall_w = np.zeros((1, 16), np.float32)
    wall_w[0, [2, 6, 7, 10, 12, 13]] = 1
    wall = lin("x0", wall_w)
    expand_w = np.zeros((1, 1, 3, 3), np.float32)
    expand_w[0, 0, 1:3, 1:3] = 1
    expanded = wall
    footprint_walls = []
    for t in range(1, 16):
        expanded = emit("min2", [lin(expanded, expand_w), one])
        if t in (3, 7, 11, 15):
            footprint_walls.append(expanded)
    valid = mul(bounds, aff(cat(footprint_walls), -1, 1))

    # ----- Exact scaled silhouette detectors. -----
    colour_ids = [8, 9, 11, 15]
    colour_w = np.zeros((4, 16), np.float32)
    for i, c in enumerate(colour_ids):
        colour_w[i, c] = 1
    colours = lin("x0", colour_w)
    occupied = sum_channels(colours)

    agent_shape = np.ones((4, 4), np.float32)
    agent_shape[3, 1:3] = 0
    goal_shape = np.zeros((6, 6), np.float32)
    goal_shape[1:5, 0] = 1
    goal_shape[1:5, 5] = 1
    goal_shape[4, 2:4] = 1
    goal_shape[5, :] = 1
    check_shape = np.fromfunction(
        lambda r, c: ((r + c) % 2 == 0), (4, 4)
    ).astype(np.float32)

    prototypes = []
    for r in range(4):
        prototypes.append(np.rot90(agent_shape, -r))
    for r in range(4):
        prototypes.append(np.rot90(goal_shape, -r))
    prototypes += [check_shape, 1 - check_shape]

    template_w = np.zeros((10, 36), np.float32)
    positive_counts = []
    for i, a in enumerate(prototypes):
        h, width = a.shape
        for r in range(h):
            for c in range(width):
                template_w[i, 6 * r + c] = 2 * a[r, c] - 1
        positive_counts.append(float(a.sum()))

    agent_maps, goal_maps, checkpoint_maps = [], [], []
    agent_rotations, goal_rotations = [], []
    for s in range(1, 5):
        k = 2 * (s - 1) + 1
        box_w = np.zeros((1, 1, k, k), np.float32)
        box_w[0, 0, k // 2:k // 2 + s, k // 2:k // 2 + s] = 1
        block = lin(occupied, box_w)

        columns = [block]
        for j in range(5):
            columns.append(shift(columns[-1], 0, s))
        six_columns = cat(columns)
        rows = [six_columns]
        for j in range(5):
            rows.append(shift(rows[-1], s, 0))
        samples = cat(rows)

        bias = -np.asarray(positive_counts, np.float32) * (s * s) + 0.5
        detected = aff(relu(lin(samples, template_w, bias)), 2)
        sbound = sl(bounds, s - 1, s)

        am = mul(sl(detected, 0, 4), sbound)
        gm = mul(shift(sl(detected, 4, 8), -s, -s), sbound)
        cm = mul(sum_channels(sl(detected, 8, 10)), sbound)

        agent_rotations.append(pool(am))
        goal_rotations.append(pool(gm))
        agent_maps.append(sum_channels(am))
        goal_maps.append(sum_channels(gm))
        checkpoint_maps.append(cm)

    A = cat(agent_maps)
    G = cat(goal_maps)
    C = cat(checkpoint_maps)
    a_union = sum_channels(A)
    g_union = sum_channels(G)
    a_rotation = total(agent_rotations)
    g_rotation = total(goal_rotations)
    a_size = lin(pool(A), [[1, 2, 3, 4]])
    g_size = lin(pool(G), [[1, 2, 3, 4]])

    _, a_xy = extrema(a_union)
    _, g_xy = extrema(g_union)
    initial_position = cat([a_xy, a_size])
    a_colour = pool(mul(a_union, colours))

    # Sample a guaranteed solid corner of the goal, given its orientation.
    corner_factors = lin(g_rotation, [
        [4, -1, -1, 4],
        [-1, -1, 4, 4],
    ])
    goal_corner = add(g_xy, mul(corner_factors, g_size))
    g_colour = pool(mul(point_mask(goal_corner), colours))

    # Orientation delta and its binary instruction.
    wa = np.zeros((16, 4), np.float32)
    wgrot = np.zeros((16, 4), np.float32)
    for a in range(4):
        for g in range(4):
            wa[4 * a + g, a] = 1
            wgrot[4 * a + g, g] = 1
    pairs = mul(lin(a_rotation, wa), lin(g_rotation, wgrot))
    rotation_decode = np.zeros((7, 16), np.float32)
    rotate_codes = [0, 5, 7, 6]
    for a in range(4):
        for g in range(4):
            delta = (g - a) % 4
            code = rotate_codes[delta]
            for bit in range(6):
                rotation_decode[bit, 4 * a + g] = (code >> bit) & 1
            rotation_decode[6, 4 * a + g] = float(delta != 0)
    rotation_info = lin(pairs, rotation_decode)
    rotation_bits = sl(rotation_info, 0, 6)
    rotation_needed = sl(rotation_info, 6, 7)

    colour_needed = lin(mul(a_colour, g_colour),
                        -np.ones((1, 4), np.float32), [1])
    colour_decode = np.zeros((6, 4), np.float32)
    for j, code in enumerate([15, 14, 0, 63]):
        for bit in range(6):
            colour_decode[bit, j] = (code >> bit) & 1
    colour_bits = lin(g_colour, colour_decode)
    colour_first = mul(colour_needed, aff(rotation_needed, -1, 1))
    colour_second = mul(colour_needed, rotation_needed)
    free_first = aff(add(rotation_needed, colour_first), -1, 1)
    free_second = aff(colour_second, -1, 1)

    # ----- Spatial Bellman circuits. -----
    advanced_penalty = aff(six_bits, 20000, -20000)
    grow_w = np.zeros((4, 4), np.float32)
    shrink_w = np.zeros((4, 4), np.float32)
    for i in range(4):
        if i < 3:
            grow_w[i, i + 1] = 1
        shrink_w[i, max(0, i - 1)] = 1

    def neighbours(v, cost):
        paid = aff(v, 1, -cost)
        advanced = add(paid, advanced_penalty)
        left4 = shift(paid, 0, -4)
        right4 = shift(paid, 0, 4)
        down4 = shift(paid, 4, 0)
        up4 = shift(advanced, -4, 0)
        right8 = add(shift(right4, 0, 4), advanced_penalty)
        left8 = add(shift(left4, 0, -4), advanced_penalty)
        grow = lin(advanced, grow_w)
        shrink = lin(advanced, shrink_w)
        return [v, left4, right4, down4, up4,
                right8, left8, grow, shrink]

    def bellman(v, cost):
        ns = neighbours(v, cost)
        z = ns[0]
        for q in ns[1:]:
            z = emit("max2", [z, q])
        return mul(z, valid)

    global_seed = mul(aff(G, 1000), valid)
    shapes["distance_state"] = (4, 64, 64)
    body = []
    outer = active
    active = body
    next_distance = bellman("distance_state", 10)
    active = outer
    graph.append({
        "id": "distance",
        "op": "loop",
        "T": 32,
        "state": [{"id": "distance_state", "init": global_seed}],
        "body": body,
        "next": {"distance_state": next_distance},
    })
    shapes["distance.distance_state"] = (4, 64, 64)

    checkpoint_values = mul(C, shapes and "distance.distance_state")
    terminal_values = emit("max2", [
        checkpoint_values,
        aff(G, 10000),
    ])
    terminal_values = mul(terminal_values, valid)
    horizons = [terminal_values]
    for i in range(5):
        horizons.append(bellman(horizons[-1], 1))

    # Winner-take-all for nine vector scores. Ties have fixed small offsets.
    pair_w = np.zeros((81, 9), np.float32)
    for a in range(9):
        for b in range(9):
            pair_w[9 * a + b, b] += 1
            pair_w[9 * a + b, a] -= 1
    win_w = np.zeros((9, 81), np.float32)
    for a in range(9):
        win_w[a, 9 * a:9 * a + 9] = -200

    def winner(scores):
        regret = relu(lin(scores, pair_w))
        return relu(lin(regret, win_w, np.ones(9)))

    codes = [0, 1, 2, 3, 33, 10, 12, 8, 9]
    bit_decode = np.zeros((6, 9), np.float32)
    for a, code in enumerate(codes):
        for bit in range(6):
            bit_decode[bit, a] = (code >> bit) & 1

    delta_w = np.array([
        [0, 0, 0, 4, -4, 0, 0, 0, 0],
        [0, -4, 4, 0, 0, 8, -8, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -1],
    ], np.float32)
    size_repeat = np.zeros((36, 4), np.float32)
    reduce_scores = np.zeros((9, 36), np.float32)
    for a in range(9):
        size_repeat[4 * a:4 * a + 4] = np.eye(4, dtype=np.float32)
        reduce_scores[a, 4 * a:4 * a + 4] = 1

    position = initial_position
    program_columns = []
    for i in range(6):
        selected_horizon = total(
            mul(horizons[j], n_flags[i + j])
            for j in range(6 - i)
        )
        ns = neighbours(selected_horizon, 1)
        here = point_mask(position)
        candidate_values = pool(mul(cat(ns), here))
        size_oh = equality_vector(sl(position, 2, 3), [1, 2, 3, 4])
        candidate_values = mul(candidate_values, lin(size_oh, size_repeat))
        scores = lin(candidate_values, reduce_scores,
                     -0.02 * np.arange(9, dtype=np.float32))
        choice = winner(scores)

        active_column_w = np.zeros((1, 6), np.float32)
        active_column_w[0, i:] = 1
        active_column = lin(n_onehot, active_column_w)
        if i == 0:
            movement_enabled = mul(active_column, free_first)
        elif i == 1:
            movement_enabled = mul(active_column, free_second)
        else:
            movement_enabled = active_column

        instruction = mul(lin(choice, bit_decode), movement_enabled)
        if i == 0:
            instruction = add(instruction, rotation_bits)
            instruction = add(instruction, mul(colour_bits, colour_first))
        elif i == 1:
            instruction = add(instruction, mul(colour_bits, colour_second))
        program_columns.append(instruction)

        if i < 5:
            displacement = mul(lin(choice, delta_w), movement_enabled)
            position = add(position, displacement)

    proposed_program = cat(program_columns)

    # ----- Keep a plan stable while editing its bit representation. -----
    remembered_program = sl("mem", 0, 36)
    holding = sl("mem", 36, 37)
    target = add(
        mul(proposed_program, aff(holding, -1, 1)),
        mul(remembered_program, holding),
    )
    difference = sub(target, observed_bits)
    magnitude = add(relu(difference), relu(aff(difference, -1)))
    changes = mul(binary(magnitude), present)
    done = relu(lin(changes, -np.ones((1, 36), np.float32), [1]))

    edit_priority = lin(changes,
                        np.diag(np.arange(100, 64, -1, dtype=np.float32)))
    edit_map = sum_channels(mul(slots, edit_priority))

    # A filled 5x5 orange patch identifies the interior of the run button.
    run_w = np.zeros((1, 16, 5, 5), np.float32)
    run_w[0, 9, :, :] = 1
    run_interior = aff(relu(lin("x0", run_w, [-24.5])), 2)
    bottom = emit("min2", [
        relu(lin(xy, [[1, 0]], [-50])),
        one,
    ])
    run_map = mul(mul(run_interior, bottom), aff(done, 1000))
    emit("add", [edit_map, run_map], name="click")

    next_holding = aff(done, -1, 1)
    next_memory = cat([target, next_holding])
    aff(next_memory, name="mem_next")

    logits_bias = np.zeros(8, np.float32)
    logits_bias[6] = 10
    lin("avail", np.zeros((8, 8), np.float32),
        logits_bias, name="logits")

    # Structural accounting only; no game execution or weight fitting.
    def costs(nodes):
        macs = 0
        evaluations = 0
        node_count = 0
        channels = 0
        for nd in nodes:
            node_count += 1
            if nd["op"] == "loop":
                bm, be, bn, bc = costs(nd["body"])
                macs += nd["T"] * bm
                evaluations += nd["T"] * be
                node_count += bn
                channels += bc
                for state in nd["state"]:
                    shape = shapes[state["id"]]
                    if len(shape) == 3:
                        channels += shape[0]
                continue
            evaluations += 1
            op = nd["op"]
            out = shapes[nd["id"]]
            if len(out) == 3:
                channels += out[0]
                assert out[0] <= 512
            else:
                assert out[0] <= 4096
            if op == "conv":
                w = P[nd["w"]]
                macs += int(np.prod(w.shape)) * 4096
            elif op == "dense":
                macs += int(np.prod(P[nd["w"]].shape))
            elif op in ("add", "sub", "mul", "max2", "min2"):
                macs += int(np.prod(out))
            elif op in ("vadd", "vmul"):
                macs += out[0]
            elif op in ("gmax", "gmean", "rowmax", "colmax",
                        "rowsum", "colsum"):
                macs += int(np.prod(shapes[nd["in"][0]]))
        return macs, evaluations, node_count, channels

    macs, evaluations, nodes, channels = costs(graph)
    assert sum(a.size for a in P.values()) <= 150000
    assert macs <= 1500000000
    assert evaluations <= 10000
    assert nodes <= 1000
    assert channels + 19 + 4 <= 4096
    assert all(np.isfinite(a).all() for a in P.values())

    return {"graph": graph, "memory_size": 37, "params": P}
