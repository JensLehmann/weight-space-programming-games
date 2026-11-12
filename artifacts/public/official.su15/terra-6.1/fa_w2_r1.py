import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]

    def fresh():
        s = "n%d" % counter[0]
        counter[0] += 1
        return s

    def op(kind, *inputs, **fields):
        ident = fresh()
        nd = {"id": ident, "op": kind, "in": list(inputs)}
        nd.update(fields)
        graph.append(nd)
        return ident

    def conv(x, w, b=None):
        ident = fresh()
        wn = fresh()
        params[wn] = np.asarray(w, dtype=np.float32)
        nd = {"id": ident, "op": "conv", "in": [x], "w": wn}
        if b is not None:
            bn = fresh()
            params[bn] = np.asarray(b, dtype=np.float32)
            nd["b"] = bn
        graph.append(nd)
        return ident

    def dense(x, w, b=None):
        ident = fresh()
        wn = fresh()
        params[wn] = np.asarray(w, dtype=np.float32)
        nd = {"id": ident, "op": "dense", "in": [x], "w": wn}
        if b is not None:
            bn = fresh()
            params[bn] = np.asarray(b, dtype=np.float32)
            nd["b"] = bn
        graph.append(nd)
        return ident

    def sa(x, channels, scale, bias=0.0):
        w = np.zeros((channels, channels, 1, 1), np.float32)
        for i in range(channels):
            w[i, i, 0, 0] = scale
        return conv(x, w, np.full(channels, bias, np.float32))

    def va(x, channels, scale, bias=0.0):
        return dense(
            x,
            np.eye(channels, dtype=np.float32) * scale,
            np.full(channels, bias, np.float32),
        )

    def ss(x, start, stop):
        return op("slice", x, start=start, stop=stop)

    def vs(x, start, stop):
        return op("vslice", x, start=start, stop=stop)

    def vsub(a, b, channels):
        return op("vadd", a, va(b, channels, -1.0))

    one = ss("coords", 2, 3)
    row = ss("coords", 0, 1)
    col = ss("coords", 1, 2)

    def clip_sp(x):
        return op("min2", op("relu", x), one)

    def clip_vec(x, channels):
        y = op("vrelu", x)
        excess = op("vrelu", va(y, channels, 1.0, -1.0))
        return vsub(y, excess, channels)

    play_lo = clip_sp(sa(row, 1, 63.0, -9.0))
    play_hi = clip_sp(sa(row, 1, -63.0, 63.0))
    play = op("mul", play_lo, play_hi)
    header_y = clip_sp(sa(row, 1, -63.0, 10.0))
    header_x = clip_sp(sa(col, 1, 63.0, -23.0))
    header = op("mul", header_y, header_x)

    anchors3 = (-3, 0, 3)

    def index3(d):
        if d <= -2:
            return 0
        if d >= 2:
            return 2
        return 1

    def anchor3(x):
        w = np.zeros((9, 1, 7, 7), np.float32)
        for iy, ay in enumerate(anchors3):
            for ix, ax in enumerate(anchors3):
                w[iy * 3 + ix, 0, ay + 3, ax + 3] = 1.0
        return conv(x, w)

    def kernel13(x, kernels, biases=None):
        a = anchor3(x)
        w = np.zeros((len(kernels), 9, 7, 7), np.float32)
        for out, kernel in enumerate(kernels):
            for (dy, dx), value in kernel.items():
                iy, ix = index3(dy), index3(dx)
                ry = dy - anchors3[iy]
                rx = dx - anchors3[ix]
                w[out, iy * 3 + ix, ry + 3, rx + 3] += value
        return conv(a, w, biases)

    def square_detector(color, size):
        plane = ss("x0", color, color + 1)
        lo = -(size // 2)
        hi = lo + size - 1
        kernel = {}
        for dy in range(lo - 1, hi + 2):
            for dx in range(lo - 1, hi + 2):
                inside = lo <= dy <= hi and lo <= dx <= hi
                kernel[(dy, dx)] = 1.0 if inside else -1.0
        raw = kernel13(plane, [kernel], [-(size * size - 0.5)])
        return clip_sp(sa(op("relu", raw), 1, 2.0))

    def hazard_detector(color, pattern):
        plane = ss("x0", color, color + 1)
        occupied = set()
        for y, line in enumerate(pattern):
            for x, value in enumerate(line):
                if value:
                    occupied.add((y - 2, x - 2))
        w = np.zeros((1, 1, 7, 7), np.float32)
        for dy in range(-3, 3):
            for dx in range(-3, 4):
                w[0, 0, dy + 3, dx + 3] = (
                    1.0 if (dy, dx) in occupied else -1.0
                )
        raw = conv(plane, w, [-(len(occupied) - 0.5)])
        return clip_sp(sa(op("relu", raw), 1, 2.0))

    # Goal-background template matching.
    circle = set()
    for y, width in enumerate((5, 7, 9, 9, 9, 9, 9, 7, 5)):
        for x in range(-(width // 2), width // 2 + 1):
            circle.add((y - 4, x))
    circle_kernel = {}
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            circle_kernel[(dy, dx)] = (
                1.0 if (dy, dx) in circle else -1.0
            )
    nine = ss("x0", 9, 10)
    circle_raw = kernel13(nine, [circle_kernel])
    circle_raw = op("mul", op("relu", circle_raw), play)
    circle_max = op("gmax", circle_raw)
    peaks = clip_sp(sa(op("sub", circle_raw, circle_max), 1, 10.0, 1.0))
    enough = clip_vec(va(circle_max, 1, 1.0, -48.0), 1)
    peaks = op("mul", peaks, enough)

    # Extract up to four distinct goal coordinates.
    priority_w = np.array(
        [[[[-63.0]], [[-63.0 / 64.0]], [[128.0]]]],
        dtype=np.float32,
    )
    priority = conv("coords", priority_w)
    remaining = peaks
    descriptors = []
    for _ in range(4):
        ranked = op("mul", remaining, priority)
        best = op("gmax", ranked)
        selected = clip_sp(sa(op("sub", ranked, best), 1, 200.0, 1.0))
        selected = op("mul", selected, remaining)
        descriptors.append(op("gmax", op("mul", selected, "coords")))
        remaining = op("relu", op("sub", remaining, sa(selected, 1, 2.0)))
    new_goals = op("vconcat", *descriptors)

    fruit_specs = [
        (10, 1), (6, 2), (15, 3), (11, 4), (12, 5),
        (8, 7), (9, 8), (7, 9), (14, 10),
    ]
    hazard_specs = [
        (7, [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ]),
        (14, [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ]),
        (13, [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ]),
    ]
    detections = [square_detector(c, s) for c, s in fruit_specs]
    detections += [hazard_detector(c, p) for c, p in hazard_specs]
    objects = op("concat", *detections)
    in_play = op("mul", objects, play)
    in_header = op("mul", objects, header)
    new_desired = va(op("gmean", in_header), 12, 4096.0)

    # Refresh memory at episode start and after RESET.
    old_goals = vs("mem", 0, 12)
    old_desired = vs("mem", 12, 24)
    initialized = vs("mem", 24, 25)
    reset_prev = vs("aprev", 0, 1)
    not_reset = va(reset_prev, 1, -1.0, 1.0)
    carry = op("vmul", initialized, not_reset)
    first = va(carry, 1, -1.0, 1.0)
    stored_goals = op(
        "vadd",
        op("vmul", old_goals, carry),
        op("vmul", new_goals, first),
    )
    desired = op(
        "vadd",
        op("vmul", old_desired, carry),
        op("vmul", new_desired, first),
    )
    init_next = va(initialized, 1, 0.0, 1.0)
    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [stored_goals, desired, init_next],
    })

    # Reconstruct saved goal masks.
    centers = None
    for i in range(4):
        gy = vs(stored_goals, 3 * i, 3 * i + 1)
        gx = vs(stored_goals, 3 * i + 1, 3 * i + 2)
        present = vs(stored_goals, 3 * i + 2, 3 * i + 3)
        dy = op("abs", op("sub", row, gy))
        dx = op("abs", op("sub", col, gx))
        ym = op("relu", sa(dy, 1, -63.0, 0.5))
        xm = op("relu", sa(dx, 1, -63.0, 0.5))
        point = sa(op("mul", ym, xm), 1, 4.0)
        point = op("mul", point, present)
        centers = point if centers is None else op("max2", centers, point)
    centers = clip_sp(centers)

    box = clip_sp(conv(centers, np.ones((1, 1, 7, 7), np.float32)))
    goal_box = clip_sp(conv(box, np.ones((1, 1, 3, 3), np.float32)))

    # Eight-neighbor distance field.
    shift_w = np.zeros((8, 1, 3, 3), np.float32)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]
    for i, (dy, dx) in enumerate(directions):
        shift_w[i, 0, dy + 1, dx + 1] = 1.0
    params["field_shift_w"] = shift_w
    params["field_shift_b"] = -np.ones(8, np.float32)

    def distance_field(seed):
        initial = sa(seed, 1, 64.0)
        loop_id = fresh()
        state_id = fresh()
        shift_id = fresh()
        body = [{
            "id": shift_id,
            "op": "conv",
            "in": [state_id],
            "w": "field_shift_w",
            "b": "field_shift_b",
        }]
        running = state_id
        for i in range(8):
            part = fresh()
            body.append({
                "id": part,
                "op": "slice",
                "in": [shift_id],
                "start": i,
                "stop": i + 1,
            })
            merged = fresh()
            body.append({
                "id": merged,
                "op": "max2",
                "in": [running, part],
            })
            running = merged
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 64,
            "state": [{"id": state_id, "init": initial}],
            "body": body,
            "next": {state_id: running},
        })
        return loop_id + "." + state_id

    goal_field = distance_field(centers)

    desired_present = clip_vec(desired, 12)
    parked = op("mul", goal_box, desired_present)
    not_parked = op("relu", op("sub", one, parked))
    active = op("mul", in_play, not_parked)

    count = va(op("gmean", in_play), 12, 4096.0)
    sum_y = va(op("gmean", op("mul", in_play, row)), 12, 4096.0)
    sum_x = va(op("gmean", op("mul", in_play, col)), 12, 4096.0)

    # Per-type local counts and moments using factored 17x17 kernels.
    def kernel17(x, kernels):
        first_stage = anchor3(x)
        w2 = np.zeros((9, 9, 7, 7), np.float32)
        for iy, ay in enumerate(anchors3):
            for ix, ax in enumerate(anchors3):
                k = iy * 3 + ix
                w2[k, k, ay + 3, ax + 3] = 1.0
        second_stage = conv(first_stage, w2)
        w3 = np.zeros((len(kernels), 9, 7, 7), np.float32)
        anchors6 = (-6, 0, 6)

        def idx(d):
            if d <= -3:
                return 0
            if d >= 3:
                return 2
            return 1

        for out, kernel in enumerate(kernels):
            for (dy, dx), value in kernel.items():
                iy, ix = idx(dy), idx(dx)
                ry = dy - anchors6[iy]
                rx = dx - anchors6[ix]
                w3[out, iy * 3 + ix, ry + 3, rx + 3] += value
        return conv(second_stage, w3)

    count_kernel = {}
    moment_y_kernel = {}
    moment_x_kernel = {}
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            if dx * dx + dy * dy <= 64:
                count_kernel[(dy, dx)] = 1.0
                moment_y_kernel[(dy, dx)] = float(-dy)
                moment_x_kernel[(dy, dx)] = float(-dx)

    covers, moments_y, moments_x = [], [], []
    for i in range(12):
        obj = ss(active, i, i + 1)
        features = kernel17(
            obj, [count_kernel, moment_y_kernel, moment_x_kernel]
        )
        covers.append(ss(features, 0, 1))
        moments_y.append(ss(features, 1, 2))
        moments_x.append(ss(features, 2, 3))
    cover = op("concat", *covers)
    my = op("concat", *moments_y)
    mx = op("concat", *moments_x)

    all_cover = conv(cover, np.ones((1, 12, 1, 1), np.float32))
    excess = op("relu", sa(all_cover, 1, 1.0, -1.0))
    single = clip_sp(
        op("relu", op("sub", all_cover, sa(excess, 1, 2.0)))
    )

    # Identify higher-rank fruit that can be downgraded by a hazard.
    deficit = op("vrelu", vsub(desired, count, 12))
    lower_w = np.zeros((12, 12), np.float32)
    for i in range(9):
        for j in range(i):
            lower_w[i, j] = 1.0
    lower_deficit = clip_vec(dense(deficit, lower_w), 12)

    hazard_active = ss(active, 9, 12)
    hazard_seed = clip_sp(
        conv(hazard_active, np.ones((1, 3, 1, 1), np.float32))
    )
    hazard_exists = clip_vec(op("gmax", hazard_seed), 1)
    downgrade = op("vmul", lower_deficit, hazard_exists)
    downgrade_fruit = vs(downgrade, 0, 9)
    downgrade_any = clip_vec(
        dense(downgrade_fruit, np.ones((1, 9), np.float32)), 1
    )

    protected_gate = va(downgrade_fruit, 9, -1.0, 1.0)
    protected_fruit = op("mul", ss(in_play, 0, 9), protected_gate)
    protected_seed = clip_sp(
        conv(protected_fruit, np.ones((1, 9, 1, 1), np.float32))
    )
    danger_field = distance_field(hazard_seed)
    fruit_field = distance_field(protected_seed)

    threat = op("gmax", op("mul", danger_field, protected_seed))
    threat_gate = clip_vec(va(threat, 1, 1.0, -52.0), 1)

    # Expand only hazard centers that are near protected fruit.
    threatened_seed = op(
        "mul",
        hazard_seed,
        clip_sp(sa(fruit_field, 1, 1.0, -52.0)),
    )
    threatened_region = threatened_seed
    for _ in range(3):
        threatened_region = clip_sp(
            conv(threatened_region, np.ones((1, 1, 7, 7), np.float32))
        )

    # Pair-merging gates.
    higher_w = np.zeros((12, 12), np.float32)
    for start, end in ((0, 9), (9, 12)):
        for i in range(start, end):
            for j in range(i + 1, end):
                higher_w[i, j] = 1.0
    higher = clip_vec(dense(desired, higher_w), 12)
    surplus = va(vsub(count, desired, 12), 12, 1.0, -1.0)
    merge_gate = op("vmul", clip_vec(surplus, 12), higher)

    # Suppress hazard transport while a downgrade is needed.
    fruit_desired = vs(desired_present, 0, 9)
    hazard_desired = vs(desired_present, 9, 12)
    no_downgrade = va(downgrade_any, 1, -1.0, 1.0)
    hazard_transport = op("vmul", hazard_desired, no_downgrade)
    transport_gate = op("vconcat", fruit_desired, hazard_transport)

    transport = op("mul", goal_field, cover)
    transport = op("mul", transport, single)
    transport = op("mul", transport, transport_gate)

    at_least_two = op("relu", sa(cover, 12, 1.0, -1.0))
    at_least_three = op("relu", sa(cover, 12, 1.0, -2.0))
    pair = op(
        "relu", op("sub", at_least_two, sa(at_least_three, 12, 2.0))
    )
    others = op("relu", op("sub", all_cover, cover))
    safe = op("relu", sa(others, 12, -1.0, 1.0))
    pair = op("mul", op("mul", pair, safe), merge_gate)
    pair_base = sa(pair, 12, 1000.0)
    pair_danger = op("mul", pair, danger_field)
    pair = op("add", pair_base, sa(pair_danger, 12, 5.0))

    dy = op("sub", sum_y, op("mul", row, count))
    dx = op("sub", sum_x, op("mul", col, count))
    direction = op(
        "relu", op("add", op("mul", dy, my), op("mul", dx, mx))
    )
    direction = op("mul", op("mul", direction, single), merge_gate)
    direction = sa(direction, 12, 50.0)

    # Lure downgrade candidates toward the hazard centroid.
    hazard_count = dense(
        vs(count, 9, 12), np.ones((1, 3), np.float32)
    )
    hazard_y = dense(
        vs(sum_y, 9, 12), np.ones((1, 3), np.float32)
    )
    hazard_x = dense(
        vs(sum_x, 9, 12), np.ones((1, 3), np.float32)
    )
    lure_y = op("sub", hazard_y, op("mul", row, hazard_count))
    lure_x = op("sub", hazard_x, op("mul", col, hazard_count))
    lure = op(
        "relu",
        op("add", op("mul", lure_y, my), op("mul", lure_x, mx)),
    )
    lure = op("mul", op("mul", lure, single), downgrade)
    lure = sa(lure, 12, 300.0)

    per_type = op(
        "add", op("add", transport, pair), op("add", direction, lure)
    )
    score = conv(per_type, np.ones((1, 12, 1, 1), np.float32))

    # Deflect threatened hazards away from protected fruit.
    hazard_cover = clip_sp(
        conv(ss(cover, 9, 12), np.ones((1, 3, 1, 1), np.float32))
    )
    fruit_cover = clip_sp(
        conv(ss(cover, 0, 9), np.ones((1, 9, 1, 1), np.float32))
    )
    safe_hazard_click = op("relu", op("sub", one, fruit_cover))
    defense = op("mul", hazard_cover, safe_hazard_click)
    defense = op("mul", defense, threatened_region)
    defense = op("mul", defense, threat_gate)
    away = sa(fruit_field, 1, -20.0, 4280.0)
    defense = op("mul", defense, away)
    score = op("add", score, defense)

    fallback = sa(danger_field, 1, -0.00001, 0.002)
    score = op("add", score, fallback)
    score = op("mul", score, play)

    params["click_w"] = np.ones((1, 1, 1, 1), np.float32)
    graph.append({
        "id": "click",
        "op": "conv",
        "in": [score],
        "w": "click_w",
    })

    # A red objective indicates insufficient remaining fruit value.
    red = ss("x0", 2, 3)
    red_objective = op("gmax", op("mul", red, header))
    logit_w = np.zeros((8, 1), np.float32)
    logit_w[0, 0] = 2.0
    logit_b = np.zeros(8, np.float32)
    logit_b[6] = 1.0
    params["logit_w"] = logit_w
    params["logit_b"] = logit_b
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [red_objective],
        "w": "logit_w",
        "b": "logit_b",
    })

    return {"graph": graph, "memory_size": 25, "params": params}
