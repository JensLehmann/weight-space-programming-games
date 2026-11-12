import numpy as np


def build():
    params, graph, cache = {}, [], {}
    active = graph
    serial = 0
    shapes = {
        "x0": (16, True), "coords": (3, True),
        "mem": (12, False), "aprev": (8, False), "avail": (8, False),
    }

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def parameter(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p" + str(len(params))
            cache[key] = name
            params[name] = a
        return cache[key]

    def node(op, ins, name=None, **fields):
        name = name or fresh()
        active.append({"id": name, "op": op, "in": list(ins), **fields})
        if op == "conv":
            sh = (params[fields["w"]].shape[0], True)
        elif op == "dense":
            sh = (params[fields["w"]].shape[0], False)
        elif op in ("concat", "vconcat"):
            sh = (sum(shapes[x][0] for x in ins), op == "concat")
        elif op in ("slice", "vslice"):
            sh = (fields["stop"] - fields["start"], op == "slice")
        elif op in ("gmax", "gmean"):
            sh = (shapes[ins[0]][0], False)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            sh = (max(shapes[x][0] for x in ins), True)
        elif op in ("vadd", "vmul"):
            sh = (max(shapes[x][0] for x in ins), False)
        else:
            sh = shapes[ins[0]]
        shapes[name] = sh
        return name

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            b = np.asarray(b, np.float32)
            if b.ndim == 0:
                b = np.full(w.shape[0], b, np.float32)
            fields["b"] = parameter(b)
        return node("conv", [x], name, **fields)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            b = np.asarray(b, np.float32)
            if b.ndim == 0:
                b = np.full(w.shape[0], b, np.float32)
            fields["b"] = parameter(b)
        return node("dense", [x], name, **fields)

    def cut(x, a, b):
        return node("slice" if shapes[x][1] else "vslice", [x],
                    start=a, stop=b)

    def cat(xs, name=None):
        return node("concat" if shapes[xs[0]][1] else "vconcat", xs, name)

    def relu(x):
        return node("relu" if shapes[x][1] else "vrelu", [x])

    def add(a, b):
        return node("add" if shapes[a][1] or shapes[b][1] else "vadd",
                    [a, b])

    def mul(a, b):
        return node("mul" if shapes[a][1] or shapes[b][1] else "vmul",
                    [a, b])

    def scale(x, a, b=0, name=None):
        w = np.eye(shapes[x][0], dtype=np.float32) * a
        if shapes[x][1]:
            return conv(x, w, b, name)
        return dense(x, w, b, name)

    def sub(a, b):
        if shapes[a][1] or shapes[b][1]:
            return node("sub", [a, b])
        return add(a, scale(b, -1))

    def inv(x):
        return scale(x, -1, 1)

    def total(x):
        w = np.ones((1, shapes[x][0]), np.float32)
        if shapes[x][1]:
            return conv(x, w)
        return dense(x, w)

    def gm(x):
        return node("gmax", [x])

    def absolute(x):
        return node("abs", [x])

    def clip(x):
        y = relu(x)
        return sub(y, relu(scale(y, 1, -1)))

    def product(xs):
        y = xs[0]
        for x in xs[1:]:
            y = mul(y, x)
        return y

    def summation(xs):
        y = xs[0]
        for x in xs[1:]:
            y = add(y, x)
        return y

    def winner(x, n):
        # First maximal integer-valued score, with explicit tie margins.
        w = np.zeros((n * n, n), np.float32)
        b = np.zeros(n * n, np.float32)
        for i in range(n):
            for j in range(n):
                k = i * n + j
                w[k, j] += 1
                w[k, i] -= 1
                b[k] = 0.25 if j < i else -0.25
        z = relu(dense(x, w, b))
        w = np.zeros((n, n * n), np.float32)
        for i in range(n):
            w[i, i * n:(i + 1) * n] = -4
        return relu(dense(z, w, 1))

    directions = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    sizes = [(3, 3), (6, 3), (3, 6), (6, 6)]

    def shift(x, dy, dx):
        while abs(dy) > 3 or abs(dx) > 3:
            sy = max(-3, min(3, dy))
            sx = max(-3, min(3, dx))
            x = shift(x, sy, sx)
            dy -= sy
            dx -= sx
        n = shapes[x][0]
        k = 7 if max(abs(dy), abs(dx)) > 1 else 3
        r = k // 2
        w = np.zeros((n, n, k, k), np.float32)
        for i in range(n):
            w[i, i, r - dy, r - dx] = 1
        return conv(x, w)

    def neighbors(x, bias=0):
        w = np.zeros((4, 1, 7, 7), np.float32)
        for d, (dy, dx) in enumerate(directions):
            w[d, 0, 3 + dy, 3 + dx] = 1
        return conv(x, w, bias if bias else None)

    def expanded(x):
        return cat([
            x, shift(x, 2, 0), shift(x, -2, 0),
            shift(x, 0, 2), shift(x, 0, -2),
        ])

    def patterns(x, channels, entries, biases):
        w = np.zeros((len(entries), channels * 5, 7, 7), np.float32)
        for o, terms in enumerate(entries):
            for c, dy, dx, value in terms:
                group = 0
                if dy < -3:
                    group, dy = 1, dy + 2
                elif dy > 3:
                    group, dy = 2, dy - 2
                elif dx < -3:
                    group, dx = 3, dx + 2
                elif dx > 3:
                    group, dx = 4, dx - 2
                w[o, group * channels + c, 3 + dy, 3 + dx] += value
        return conv(x, w, biases)

    def loop(T, initial, body_function):
        nonlocal active
        lid = fresh("L")
        state_ids = [fresh("s") for _ in initial]
        for s, x in zip(state_ids, initial):
            shapes[s] = shapes[x]
        parent = active
        body = []
        active = body
        result = body_function(state_ids)
        active = parent
        parent.append({
            "id": lid, "op": "loop", "T": T,
            "state": [{"id": s, "init": x}
                      for s, x in zip(state_ids, initial)],
            "body": body,
            "next": dict(zip(state_ids, result)),
        })
        outputs = []
        for s, x in zip(state_ids, result):
            name = lid + "." + s
            shapes[name] = shapes[x]
            outputs.append(name)
        return outputs

    one = cut("coords", 2, 3)
    xy = conv("coords", [[63, 0, 0], [0, 63, 0]])
    black = cut("x0", 0, 1)
    yellow = cut("x0", 4, 5)
    wall = cut("x0", 2, 3)
    gulch = cut("x0", 15, 16)
    cross_colour = cut("x0", 11, 12)

    # Four rectangular shape detectors, followed by selected-shape detectors.
    w = np.zeros((8, 16, 7, 7), np.float32)
    b = np.zeros(8, np.float32)
    for t, (h, width) in enumerate(sizes):
        cy, cx = h // 2, width // 2
        markers = 0
        for y in range(h):
            for x in range(width):
                yy, xx = 3 + y - cy, 3 + x - cx
                boundary = y in (0, h - 1) or x in (0, width - 1)
                my = y == 1 if h == 3 else y in (2, 3)
                mx = x == 1 if width == 3 else x in (2, 3)
                if boundary or not (my and mx):
                    for c in (0, 14):
                        w[t, c, yy, xx] = 1
                        w[t + 4, c, yy, xx] = 1
                else:
                    for c in (0, 4, 5):
                        w[t, c, yy, xx] = 1
                        w[t + 4, c, yy, xx] = 1
                    w[t + 4, 0, yy, xx] += 1
                    markers += 1
        b[t] = 1 - h * width
        b[t + 4] = 1 - h * width - markers
    raw = relu(conv("x0", w, b))

    # Contact-edge consistency excludes small false rectangles inside
    # the black marker of a selected large rectangle.
    tests, test_types = [], []
    for t, (h, width) in enumerate(sizes):
        cy, cx = h // 2, width // 2
        edges = [
            ([(0, x) for x in range(1, width - 1)],
             [(0, 0), (0, width - 1)]),
            ([(h - 1, x) for x in range(1, width - 1)],
             [(h - 1, 0), (h - 1, width - 1)]),
            ([(y, 0) for y in range(1, h - 1)],
             [(0, 0), (h - 1, 0)]),
            ([(y, width - 1) for y in range(1, h - 1)],
             [(0, width - 1), (h - 1, width - 1)]),
        ]
        for middle, corners in edges:
            for y, x in middle:
                for ey, ex in corners:
                    k = np.zeros((1, 7, 7), np.float32)
                    k[0, 3 + y - cy, 3 + x - cx] += 1
                    k[0, 3 + ey - cy, 3 + ex - cx] -= 1
                    tests.append(k)
                    test_types.append(t)
    defects = relu(conv(black, np.stack(tests)))
    wd = np.zeros((4, len(tests)), np.float32)
    for i, t in enumerate(test_types):
        wd[t, i] = 1
    defects = conv(defects, wd)
    tiles = relu(sub(cut(raw, 0, 4), defects))
    selected = relu(sub(cut(raw, 4, 8), defects))
    selected_type = gm(selected)
    selected_point = total(selected)
    tile_points = total(tiles)
    selected_xy = gm(mul(xy, selected_point))
    selected_exists = clip(gm(selected_point))

    # Reconstruct rectangular footprints for occlusion-aware goal detection.
    wo = np.zeros((1, 4, 7, 7), np.float32)
    for t, (h, width) in enumerate(sizes):
        cy, cx = h // 2, width // 2
        for y in range(h):
            for x in range(width):
                wo[0, t, 3 - y + cy, 3 - x + cx] = 1
    occupancy = conv(tiles, wo)

    wc = np.zeros((1, 16, 7, 7), np.float32)
    for dy, dx in [(0, 0), (-3, 0), (3, 0), (0, -3), (0, 3)]:
        wc[0, 11, 3 + dy, 3 + dx] = 1
    for dy in (-2, 2):
        for dx in (-2, 2):
            wc[0, 11, 3 + dy, 3 + dx] = -1
    cross_point = relu(conv("x0", wc, -4))

    occluders = clip(add(occupancy, cross_colour))
    border_allowed = node("max2", [yellow, occluders])
    visible_yellow = mul(yellow, inv(occluders))
    goal_source = expanded(cat([border_allowed, visible_yellow]))
    borders = []
    for h, width in sizes:
        cy, cx = h // 2, width // 2
        borders.append([
            (y - cy - 1, x - cx - 1)
            for y in range(h + 2) for x in range(width + 2)
            if y in (0, h + 1) or x in (0, width + 1)
        ])
    cross_border = []
    for y in range(11):
        for x in range(11):
            visible = (
                (y in (0, 10) and 3 <= x <= 7)
                or (y in (1, 2, 8, 9) and x in (3, 7))
                or (y in (3, 7) and (x <= 3 or x >= 7))
                or (y in (4, 5, 6) and x in (0, 10))
            )
            if visible:
                cross_border.append((y - 5, x - 5))
    borders.append(cross_border)
    entries = [
        [(0, y, x, 1) for y, x in border] for border in borders
    ] + [
        [(1, y, x, 1) for y, x in border] for border in borders
    ]
    gd = relu(patterns(
        goal_source, 2, entries,
        [1 - len(x) for x in borders] + [-3] * 5))
    goal_detection = mul(cut(gd, 0, 5), clip(cut(gd, 5, 10)))
    goals = cut(goal_detection, 0, 4)
    cross_goal = cut(goal_detection, 4, 5)

    # Bomb detectors remain valid throughout their countdown.
    wb = np.zeros((2, 16, 7, 7), np.float32)
    for o, size in enumerate((3, 6)):
        c = size // 2
        for y in range(size):
            for x in range(size):
                for colour in (12, 13):
                    wb[o, colour, 3 + y - c, 3 + x - c] = 1
        if size == 3:
            for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                for colour in (12, 13):
                    wb[o, colour, 3 + dy, 3 + dx] = -1
    bombs = relu(conv("x0", wb, [-8, -35]))
    bombs_exist = clip(gm(total(bombs)))

    # Rectangular configuration spaces: ordinary travel and travel
    # relaxed only with respect to gulches.
    hard_w = np.zeros((2, 16), np.float32)
    hard_w[0, [2, 11, 12, 13, 15]] = 1
    hard_w[1, [2, 11, 12, 13]] = 1
    hard = conv("x0", hard_w)

    def rectangular_clear(obstacle):
        source = cat([obstacle, one])
        w = np.zeros((4, 2, 7, 7), np.float32)
        biases = []
        for t, (h, width) in enumerate(sizes):
            cy, cx = h // 2, width // 2
            for y in range(h):
                for x in range(width):
                    yy, xx = 3 + y - cy, 3 + x - cx
                    w[t, 0, yy, xx] = -1
                    w[t, 1, yy, xx] = 1
            biases.append(1 - h * width)
        return relu(conv(source, w, biases))

    legal_true = rectangular_clear(cut(hard, 0, 1))
    legal_relaxed = rectangular_clear(cut(hard, 1, 2))
    f_blocked = inv(rectangular_clear(gulch))

    # Cross configuration space and analytical launch value propagation.
    cross_offsets = [
        (y, x) for y in range(-4, 5) for x in range(-4, 5)
        if abs(y) <= 1 or abs(x) <= 1
    ]
    terms = []
    for y, x in cross_offsets:
        terms.extend([(0, y, x, -1), (1, y, x, 1)])
    cross_legal = relu(patterns(
        expanded(cat([wall, one])), 2,
        [terms], [1 - len(cross_offsets)]))
    f_count = patterns(
        expanded(gulch), 1,
        [[(0, y, x, 1) for y, x in cross_offsets]], [0])
    cross_on_f = inv(relu(scale(f_count, -1, 1)))
    cross_seed = scale(mul(cross_goal, cross_legal), 500)
    step_gates, continuation_gates = [], []
    for dy, dx in directions:
        g = shift(cross_legal, -dy, -dx)
        step_gates.append(g)
        continuation_gates.append(mul(g, cross_on_f))
    zero_map = scale(one, 0)

    def cross_body(states):
        value = states[0]
        qs = []
        for d, (dy, dx) in enumerate(directions):
            gate = continuation_gates[d]
            stationary = mul(value, inv(gate))
            r = value
            for _ in range(8):
                r = add(mul(shift(r, -dy, -dx), gate), stationary)
            gate = step_gates[d]
            stationary = mul(value, inv(gate))
            q = r
            for _ in range(5):
                q = add(mul(shift(q, -dy, -dx), gate), stationary)
            qs.append(q)
        best = scale(qs[0], 1, -1)
        for q in qs[1:]:
            best = node("max2", [best, scale(q, 1, -1)])
        new_value = mul(node("max2", [best, cross_seed]), cross_legal)
        return [new_value] + qs

    cross_values = loop(16, [cross_seed] + [zero_map] * 4, cross_body)
    cross_scores = gm(mul(cat(cross_values[1:]), cross_point))
    cross_direction = winner(cross_scores, 4)
    cross_done = clip(gm(mul(cross_point, cross_goal)))
    cross_needed = product([
        clip(gm(cross_point)), clip(gm(cross_goal)), inv(cross_done)
    ])

    # Memory: launch/blast flags, tracked centre, shape and direction.
    launch_mem = cut("mem", 0, 1)
    blast_mem = cut("mem", 1, 2)
    remembered_xy = cut("mem", 2, 4)
    remembered_type = cut("mem", 4, 8)
    remembered_direction = cut("mem", 8, 12)
    delta = sub(xy, remembered_xy)
    distance_components = absolute(delta)
    memory_point = clip(scale(total(distance_components), -4, 2))
    victim_type_points = total(mul(tiles, remembered_type))
    victim_present = clip(gm(mul(memory_point, victim_type_points)))
    recovery = mul(launch_mem, inv(victim_present))
    signed = conv(delta, [[-1, 0], [1, 0], [0, -1], [0, 1]])
    forward_progress = total(mul(signed, remembered_direction))
    selected_progress = gm(mul(forward_progress, selected_point))
    blast_finished = clip(scale(selected_progress, 1, -5))
    active_launch = mul(launch_mem, victim_present)
    active_blast = mul(blast_mem, inv(blast_finished))
    normal_mode = inv(add(active_launch, active_blast))

    def targets(source, offsets, direction):
        cin = shapes[source][0]
        outputs = []
        for d in range(4):
            axial_y = d < 2
            axial = [
                dy if axial_y else dx
                for row in offsets[d] for positions in row
                for dy, dx in positions
            ]
            maximum = max(abs(v) for v in axial)
            sign = 1 if sum(axial) >= 0 else -1
            prefix = sign * (6 if maximum > 6 else 3 if maximum > 3 else 0)
            sy, sx = (prefix, 0) if axial_y else (0, prefix)
            shifted = shift(source, sy, sx) if prefix else source
            entries = []
            large = False
            for row in offsets[d]:
                terms = []
                for inp, positions in enumerate(row):
                    for dy, dx in positions:
                        ry, rx = dy - sy, dx - sx
                        large |= max(abs(ry), abs(rx)) > 3
                        terms.append((inp, -ry, -rx, 1))
                entries.append(terms)
            if large:
                out = patterns(expanded(shifted), cin, entries, [0] * 4)
            else:
                w = np.zeros((4, cin, 7, 7), np.float32)
                for o, terms in enumerate(entries):
                    for c, dy, dx, v in terms:
                        w[o, c, 3 + dy, 3 + dx] += v
                out = conv(shifted, w)
            outputs.append(mul(out, cut(direction, d, d + 1)))
        return clip(summation(outputs))

    cross_approach_offsets = [[] for _ in range(4)]
    for d in range(4):
        for h, width in sizes:
            cy, cx = h // 2, width // 2
            top, bottom = -(2 + h - 1 - cy), 2 + cy
            left, right = -(2 + width - 1 - cx), 2 + cx
            py = 0 if h == 3 else 2
            px = 0 if width == 3 else 2
            if d == 0:
                positions = [(bottom, left), (bottom, right), (5 + cy, px)]
            elif d == 1:
                positions = [(top, left), (top, right),
                             (-(5 + h - 1 - cy), px)]
            elif d == 2:
                positions = [(top, right), (bottom, right), (py, 5 + cx)]
            else:
                positions = [(top, left), (bottom, left),
                             (py, -(5 + width - 1 - cx))]
            cross_approach_offsets[d].append([positions])
    cross_approaches = targets(
        cross_point, cross_approach_offsets, cross_direction)

    launch_offsets = [[] for _ in range(4)]
    for d in range(4):
        for lh, lw in sizes:
            lcy, lcx = lh // 2, lw // 2
            row = []
            for vh, vw in sizes:
                vcy, vcx = vh // 2, vw // 2
                if d == 0:
                    p = (vh - vcy + lcy, lcx - vcx)
                elif d == 1:
                    p = (-(vcy + lh - lcy), lcx - vcx)
                elif d == 2:
                    p = (lcy - vcy, vw - vcx + lcx)
                else:
                    p = (lcy - vcy, -(vcx + lw - lcx))
                row.append([p])
            launch_offsets[d].append(row)
    victim_marker = mul(memory_point, remembered_type)
    launch_targets = targets(
        victim_marker, launch_offsets, remembered_direction)

    blast_offsets = [[] for _ in range(4)]
    for d in range(4):
        for h, width in sizes:
            cy, cx = h // 2, width // 2
            row = []
            for size in (3, 6):
                bc = size // 2
                py = cy - bc if size == 3 else (1 if h == 3 else 0)
                px = cx - bc if size == 3 else (1 if width == 3 else 0)
                if d == 0:
                    p = (-(bc + h - cy), px)
                elif d == 1:
                    p = (size - bc + cy, px)
                elif d == 2:
                    p = (py, -(bc + width - cx))
                else:
                    p = (py, size - bc + cx)
                row.append([p])
            blast_offsets[d].append(row)
    blast_targets = targets(bombs, blast_offsets, remembered_direction)

    empty_goals = mul(goals, inv(tiles))
    main_targets = add(
        mul(cross_approaches, cross_needed),
        mul(empty_goals, inv(cross_needed)))
    objectives = summation([
        mul(main_targets, normal_mode),
        mul(launch_targets, active_launch),
        mul(blast_targets, active_blast),
    ])
    true_seed = scale(mul(objectives, legal_true), 1000)
    relaxed_seed = scale(mul(objectives, legal_relaxed), 1000)
    seeds = [cut(true_seed, i, i + 1) for i in range(4)]
    seeds += [cut(relaxed_seed, i, i + 1) for i in range(4)]
    legal_maps = [cut(legal_true, i, i + 1) for i in range(4)]
    legal_maps += [cut(legal_relaxed, i, i + 1) for i in range(4)]

    def navigation_body(states):
        outputs = []
        for i, value in enumerate(states):
            q = neighbors(value, -1)
            a = node("max2", [cut(q, 0, 1), cut(q, 1, 2)])
            b = node("max2", [cut(q, 2, 3), cut(q, 3, 4)])
            best = node("max2", [a, b])
            outputs.append(node("max2", [
                mul(best, legal_maps[i]), seeds[i]
            ]))
        return outputs

    nav = loop(72, seeds, navigation_body)
    true_fields = cat(nav[:4])
    relaxed_fields = cat(nav[4:])

    def selected_field(x):
        return total(mul(x, selected_type))

    true_field = selected_field(true_fields)
    relaxed_field = selected_field(relaxed_fields)
    true_here = gm(mul(true_field, selected_point))
    relaxed_here = gm(mul(relaxed_field, selected_point))
    has_true = clip(true_here)
    has_relaxed = clip(relaxed_here)
    has_route = clip(add(has_true, has_relaxed))
    true_scores = gm(mul(neighbors(true_field), selected_point))
    relaxed_scores = gm(mul(neighbors(relaxed_field), selected_point))
    true_direction = winner(true_scores, 4)
    relaxed_direction = winner(relaxed_scores, 4)
    route_direction = add(
        mul(true_direction, has_true),
        mul(relaxed_direction, inv(has_true)))
    at_objective = clip(gm(mul(
        selected_field(objectives), selected_point)))
    selected_done = clip(gm(total(mul(selected, goals))))

    # Face the bomb while waiting. Against a backed-up bomb this advances
    # its countdown without moving the piece out of the blast lane.
    wait_w = np.zeros((4, 4), np.float32)
    wait_w[1, 0] = 1
    wait_w[0, 1] = 1
    wait_w[3, 2] = 1
    wait_w[2, 3] = 1
    blast_wait = dense(remembered_direction, wait_w)
    push_direction = summation([
        product([cross_direction, cross_needed, normal_mode]),
        mul(remembered_direction, active_launch),
        mul(blast_wait, active_blast),
    ])
    movement = add(
        mul(route_direction, inv(at_objective)),
        mul(push_direction, at_objective))

    other_points = relu(sub(tile_points, selected_point))
    other_tiles = relu(sub(tiles, selected))
    others_exist = clip(gm(other_points))

    # Solved pieces may still be used as launchers, but they must not
    # masquerade as unfinished tasks when deciding to switch pieces.
    eligible_types = add(
        mul(inv(goals), inv(cross_needed)),
        mul(scale(goals, 0, 1), cross_needed))
    other_task_tiles = mul(other_tiles, eligible_types)
    other_values = total(mul(true_fields, other_task_tiles))
    other_reachable = clip(gm(other_values))

    prefer_other = product([
        normal_mode, inv(has_true), other_reachable, inv(recovery)
    ])
    f_samples = gm(mul(
        neighbors(selected_field(f_blocked)), selected_point))
    wants_gulch = clip(total(mul(f_samples, relaxed_direction)))
    normal_actionable = product([
        normal_mode, inv(mul(selected_done, inv(cross_needed))),
        inv(recovery), inv(prefer_other),
    ])
    launch_actionable = add(normal_actionable, active_launch)
    handoff = product([
        launch_actionable, inv(has_true), has_relaxed,
        wants_gulch, inv(at_objective),
    ])
    new_launch = mul(handoff, others_exist)
    new_blast = product([handoff, inv(others_exist), bombs_exist])
    store = clip(add(new_launch, new_blast))

    abort_launch = mul(active_launch, inv(has_relaxed))
    keep_launch = product([
        active_launch, inv(abort_launch), inv(store)
    ])

    unfinished_points = total(mul(tiles, inv(goals)))
    distance_to_selected = total(absolute(sub(xy, selected_xy)))
    ordinary_score = add(
        scale(distance_to_selected, -1, 150),
        scale(unfinished_points, 250))
    launcher_candidates = relu(sub(
        other_points, mul(memory_point, active_launch)))
    candidate_exists = clip(gm(launcher_candidates))
    ordinary_candidates = add(
        mul(launcher_candidates, candidate_exists),
        mul(other_points, inv(candidate_exists)))
    ordinary_click = mul(ordinary_candidates, ordinary_score)

    reachable_points = clip(total(other_task_tiles))
    reachable_click = mul(
        reachable_points, add(other_values, ordinary_score))
    ordinary_click = add(
        mul(reachable_click, prefer_other),
        mul(ordinary_click, inv(prefer_other)))

    perpendicular = conv(
        distance_components, [[0, 1], [0, 1], [1, 0], [1, 0]])
    perpendicular = total(mul(perpendicular, remembered_direction))
    ray_clear = clip(scale(perpendicular, -4, 2))
    forward_gate = clip(scale(forward_progress, 1, -2))
    ray_candidates = product([
        victim_type_points, ray_clear, forward_gate
    ])
    ray_exists = clip(gm(ray_candidates))
    ray_score = scale(
        absolute(scale(forward_progress, 1, -15)), -1, 150)
    recovered_click = add(
        mul(mul(ray_candidates, ray_score), ray_exists),
        mul(ordinary_click, inv(ray_exists)))
    click_map = add(
        mul(recovered_click, recovery),
        mul(ordinary_click, inv(recovery)))

    switch_solved = product([
        normal_mode, inv(cross_needed), selected_done, others_exist
    ])
    switch_unreachable = product([
        normal_mode, inv(has_true), inv(has_relaxed), others_exist
    ])
    switch_launcher = mul(abort_launch, others_exist)
    click_flag = clip(summation([
        recovery, new_launch, switch_solved, switch_unreachable,
        switch_launcher, prefer_other,
    ]))

    click_map = add(
        mul(click_map, selected_exists),
        mul(scale(tile_points, 100), inv(selected_exists)))
    click_flag = clip(add(
        click_flag, mul(inv(selected_exists), clip(gm(tile_points)))))

    fallback_w = np.zeros((4, 8), np.float32)
    fallback_w[3, 1] = 1
    fallback_w[2, 2] = 1
    fallback_w[0, 3] = 1
    fallback_w[1, 4] = 1
    fallback = winner(dense("aprev", fallback_w, [0, 0, 0, 1]), 4)
    movement = add(
        mul(movement, has_route), mul(fallback, inv(has_route)))

    features = cat([movement, click_flag])
    output_w = np.zeros((8, 5), np.float32)
    for d in range(4):
        output_w[d + 1, d] = 10
    output_w[6, 4] = 20
    dense(features, output_w,
          [-1000, 0, 0, 0, 0, -1000, 0, -1000], "logits")
    scale(click_map, 1, name="click")

    next_launch = add(keep_launch, new_launch)
    next_blast = add(mul(active_blast, inv(store)), new_blast)
    update_xy = clip(add(store, active_blast))
    next_xy = add(
        mul(remembered_xy, inv(update_xy)),
        mul(selected_xy, update_xy))
    next_type = add(
        mul(remembered_type, inv(store)), mul(selected_type, store))
    next_direction = add(
        mul(remembered_direction, inv(store)),
        mul(relaxed_direction, store))
    cat([next_launch, next_blast, next_xy, next_type, next_direction],
        "mem_next")

    return {"graph": graph, "memory_size": 12, "params": params}
