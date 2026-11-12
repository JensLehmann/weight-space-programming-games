import numpy as np


def build():
    params = {}
    graph = []
    sink = graph
    counter = [0]
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "mem": ("v", 91),
        "aprev": ("v", 8),
        "avail": ("v", 8),
    }

    def fresh(prefix="n"):
        counter[0] += 1
        return prefix + str(counter[0])

    def parameter(value):
        name = fresh("p")
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, name=None, **fields):
        name = fresh() if name is None else name
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        sink.append(nd)

        if op == "conv":
            shape = ("s", params[fields["w"]].shape[0])
        elif op == "dense":
            shape = ("v", params[fields["w"]].shape[0])
        elif op in ("concat", "vconcat"):
            shape = (shapes[inputs[0]][0],
                     sum(shapes[x][1] for x in inputs))
        elif op in ("slice", "vslice"):
            shape = (shapes[inputs[0]][0],
                     fields["stop"] - fields["start"])
        elif op in ("gmax", "gmean"):
            shape = ("v", shapes[inputs[0]][1])
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = ("s", max(shapes[x][1] for x in inputs))
        elif op in ("vadd", "vmul"):
            shape = ("v", max(shapes[x][1] for x in inputs))
        else:
            shape = shapes[inputs[0]]
        shapes[name] = shape
        return name

    def linear(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        spatial = shapes[x][0] == "s"
        fields = {
            "w": parameter(w[:, :, None, None] if spatial else w)
        }
        if b is not None:
            bias = np.broadcast_to(
                np.asarray(b, np.float32), (w.shape[0],)
            ).copy()
            if np.any(bias):
                fields["b"] = parameter(bias)
        return node("conv" if spatial else "dense",
                    [x], name=name, **fields)

    def scale(x, a, b=0.0, name=None):
        return linear(
            x, np.eye(shapes[x][1], dtype=np.float32) * a, b, name
        )

    def constant(values):
        values = np.asarray(values, np.float32).reshape(-1)
        return linear(
            "avail", np.zeros((len(values), 8), np.float32), values
        )

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def sl(x, start, stop):
        return node(
            "slice" if shapes[x][0] == "s" else "vslice",
            [x], start=start, stop=stop
        )

    def cat(*xs):
        return node(
            "concat" if shapes[xs[0]][0] == "s" else "vconcat", xs
        )

    def add(a, b):
        spatial = shapes[a][0] == "s" or shapes[b][0] == "s"
        return node("add" if spatial else "vadd", [a, b])

    def sub(a, b):
        if shapes[a][0] == "s" or shapes[b][0] == "s":
            return node("sub", [a, b])
        return add(a, scale(b, -1))

    def mul(a, b):
        spatial = shapes[a][0] == "s" or shapes[b][0] == "s"
        return node("mul" if spatial else "vmul", [a, b])

    def relu(x):
        return node(
            "relu" if shapes[x][0] == "s" else "vrelu", [x]
        )

    def absolute(x):
        if shapes[x][0] == "s":
            return node("abs", [x])
        return add(relu(x), relu(scale(x, -1)))

    def maximum(a, b):
        if shapes[a][0] == "s" or shapes[b][0] == "s":
            return node("max2", [a, b])
        return add(a, relu(sub(b, a)))

    def total(x):
        return linear(x, np.ones((1, shapes[x][1]), np.float32))

    def gmax(x):
        return node("gmax", [x])

    def gmean(x):
        return node("gmean", [x])

    zero = constant([0])
    one = constant([1])

    def clip(x):
        y = relu(x)
        if shapes[x][0] == "s":
            return node("min2", [y, one])
        return scale(relu(scale(y, -1, 1)), -1, 1)

    def positive(x, k=256):
        return clip(scale(x, k))

    def le(x, threshold=0.0, k=256):
        return clip(scale(x, -k, 1 + k * threshold))

    def inv(x):
        return scale(x, -1, 1)

    def blend(flag, yes, no):
        return add(mul(flag, yes), mul(inv(flag), no))

    def dot_channels(field, vector):
        return total(mul(field, vector))

    def gather(field, mask):
        return scale(gmean(mul(field, mask)), 4096)

    xy = linear("coords", [[63, 0, 0], [0, 63, 0]])
    yy, xx = sl(xy, 0, 1), sl(xy, 1, 2)
    key = linear("coords", [[-4032, -63, 4096]])

    def point(position):
        return le(total(absolute(sub(xy, position))), 0.10, 64)

    def first(mask):
        score = mul(mask, key)
        peak = gmax(score)
        equal = relu(scale(absolute(sub(score, peak)), -4, 1))
        return mul(equal, gmax(mask))

    def winner(score, valid=None):
        peak = gmax(score)
        tied = relu(scale(absolute(sub(score, peak)), -1.0e8, 1))
        if valid is not None:
            tied = mul(tied, valid)
        return first(tied)

    def loop(iterations, initial, body):
        nonlocal sink
        lid = fresh("L")
        states = {}
        specs = []
        for logical, init in initial.items():
            sid = fresh("s")
            states[logical] = sid
            if isinstance(init, str):
                specs.append({"id": sid, "init": init})
                shapes[sid] = shapes[init]
            else:
                zshape = list(init)
                specs.append({"id": sid, "zeros": zshape})
                shapes[sid] = (
                    "s" if len(zshape) == 3 else "v", zshape[0]
                )
        outer = sink
        sink = []
        next_values = body(states)
        body_nodes = sink
        sink = outer
        sink.append({
            "id": lid,
            "op": "loop",
            "T": iterations,
            "state": specs,
            "body": body_nodes,
            "next": {
                states[k]: next_values[k] for k in states
            },
        })
        result = {}
        for logical, sid in states.items():
            out = lid + "." + sid
            shapes[out] = shapes[sid]
            result[logical] = out
        return result

    def expand(mask, kernel):
        return clip(conv(
            mask, np.ones((1, 1, kernel, kernel), np.float32)
        ))

    def expand9(mask):
        return expand(expand(mask, 5), 5)

    # Detect complete selected and unselected anchor diamonds.
    wa = np.zeros((2, 16, 5, 5), np.float32)
    for out, color in enumerate((0, 3)):
        for r in range(-2, 3):
            for c in range(-2, 3):
                distance = abs(r) + abs(c)
                if 0 < distance <= 2:
                    wa[out, color, r + 2, c + 2] = 1
                elif distance > 2:
                    wa[out, color, r + 2, c + 2] = -1
    anchors2 = relu(conv("x0", wa, np.full(2, -11, np.float32)))
    selected_visible = sl(anchors2, 0, 1)
    anchors = total(anchors2)
    black_anchors = mul(anchors, sl("x0", 4, 5))

    # Detect filled circular disks, including unpainted disks and pickups.
    disk_colors = (0, 6, 7, 8, 9, 11, 12, 13, 14, 15)
    wd = np.zeros((1, 16, 5, 5), np.float32)
    for r in range(-2, 3):
        for c in range(-2, 3):
            if not (abs(r) == 2 and abs(c) == 2):
                for color in disk_colors:
                    wd[0, color, r + 2, c + 2] = 1
    disks = relu(conv("x0", wd, np.array([-20], np.float32)))

    colors = (6, 7, 8, 9, 11, 12, 13, 14, 15)
    C = len(colors)
    wc = np.zeros((C, 16), np.float32)
    for j, color in enumerate(colors):
        wc[j, color] = 1
    labels = linear("x0", wc)

    # Read all noncentral disk pixels, not just two samples. This also
    # recognizes rotated or partially painted disks.
    wp = np.zeros((C, C, 5, 5), np.float32)
    for j in range(C):
        for r in range(-2, 3):
            for c in range(-2, 3):
                if (r or c) and not (abs(r) == 2 and abs(c) == 2):
                    wp[j, j, r + 2, c + 2] = 1
    disk_palette = clip(conv(labels, wp))

    # Detect ring halves and combine them into colour sets.
    ring_points = [
        (-3, -1), (-3, 1), (-2, -2), (-2, 2), (-1, -3), (-1, 3),
        (1, -3), (1, 3), (2, -2), (2, 2), (3, -1), (3, 1),
    ]
    wr = np.zeros((4 * C, 16, 7, 7), np.float32)
    for side in range(4):
        for j, color in enumerate(colors):
            for r, c in ring_points:
                use = (
                    (side == 0 and c < 0) or
                    (side == 1 and c > 0) or
                    (side == 2 and r < 0) or
                    (side == 3 and r > 0)
                )
                if use:
                    wr[side * C + j, color, r + 3, c + 3] = 1
    halves = relu(conv(
        "x0", wr, np.full(4 * C, -5, np.float32)
    ))
    left, right = sl(halves, 0, C), sl(halves, C, 2 * C)
    top, bottom = sl(halves, 2 * C, 3 * C), sl(halves, 3 * C, 4 * C)
    horizontal = mul(total(left), total(right))
    vertical = mul(total(top), total(bottom))
    rings = maximum(horizontal, vertical)
    ring_palette = maximum(
        mul(maximum(left, right), horizontal),
        mul(maximum(top, bottom), vertical),
    )
    pair_ring = relu(scale(total(ring_palette), 1, -1))

    # Local directions of the visible anchor-to-barycentre lines.
    lines = sl("x0", 1, 2)
    wf = np.zeros((3, 1, 7, 7), np.float32)
    for r in range(-3, 4):
        for c in range(-3, 4):
            if max(abs(r), abs(c)) == 3:
                wf[:, 0, r + 3, c + 3] = (r, c, 1)
    flow = conv(lines, wf)
    fy, fx, fq = sl(flow, 0, 1), sl(flow, 1, 2), sl(flow, 2, 3)

    # Memory: active anchor records, completed anchor records, goal,
    # palette, validity, phase, selection, movement flag, click, group type.
    ma = sl("mem", 0, 24)
    md = sl("mem", 24, 72)
    mgxy = sl("mem", 72, 74)
    mgc = sl("mem", 74, 83)
    mgv = sl("mem", 83, 84)
    mi = sl("mem", 84, 85)
    msel = sl("mem", 85, 87)
    mmove = sl("mem", 87, 88)
    mclick = sl("mem", 88, 90)
    mplain = sl("mem", 90, 91)

    def unpack(records, count):
        outputs = []
        for component in range(3):
            w = np.zeros((count, 3 * count), np.float32)
            for j in range(count):
                w[j, 3 * j + component] = 1
            outputs.append(linear(records, w))
        return outputs

    def recall(records, count):
        ry, rx, valid = unpack(records, count)
        distance = add(absolute(sub(yy, ry)), absolute(sub(xx, rx)))
        return total(mul(le(distance, 0.10, 64), valid))

    def pack(mask, count):
        width = 3 * count
        shift = np.zeros((width, width), np.float32)
        shift[:width - 3, 3:] = np.eye(width - 3, dtype=np.float32)
        insert = np.zeros((width, 3), np.float32)
        insert[-3:, :] = np.eye(3, dtype=np.float32)

        def body(s):
            p = first(s["remaining"])
            record = cat(gather(xy, p), gmax(p))
            records = add(
                linear(s["records"], shift),
                linear(record, insert),
            )
            return {
                "remaining": relu(sub(s["remaining"], p)),
                "records": records,
            }

        return loop(count, {
            "remaining": mask,
            "records": [width],
        }, body)["records"]

    old_group = recall(ma, 8)
    old_done = recall(md, 16)

    observed_selection = first(selected_visible)
    observed_exists = gmax(observed_selection)
    observed_xy = gather(xy, observed_selection)

    # If the selected sprite is occluded, the last emitted click is its
    # predicted location. Valid visible observations override this.
    selected_xy = blend(observed_exists, observed_xy, mclick)
    selected_ok = clip(add(observed_exists, mi))
    selected_point = mul(point(selected_xy), selected_ok)

    selected_label = gather(labels, selected_point)
    raw_plain = total(selected_label)
    plain = blend(mi, mplain, raw_plain)
    black = inv(plain)
    plain_group = mul(anchors, dot_channels(labels, selected_label))

    connected_support = clip(add(
        lines, expand(add(anchors, disks), 5)
    ))

    def flood_body(s):
        return {
            "f": mul(expand(s["f"], 3), connected_support)
        }

    flood = loop(128, {"f": selected_point}, flood_body)["f"]
    flood_group = mul(flood, black_anchors)

    # A candidate disk is a barycentre only if the anchors pointing toward
    # it have the correct floored arithmetic mean.
    initial_ball_map = mul(disks, sl("x0", 6, 7))
    negative_best = constant([-100000])

    def discover_body(s):
        candidate = first(s["remaining"])
        candidate_xy = gather(xy, candidate)
        exists = gmax(candidate)

        delta = sub(candidate_xy, xy)
        dy, dx = sl(delta, 0, 1), sl(delta, 1, 2)
        radius = maximum(absolute(dy), absolute(dx))
        cross = absolute(sub(mul(dy, fx), mul(dx, fy)))
        projection = add(mul(dy, fy), mul(dx, fx))
        tolerance = scale(mul(radius, fq), 0.56, 0.05)
        aligned = mul(
            le(sub(cross, tolerance), 0, 512),
            positive(scale(projection, 1, -0.05)),
        )
        membership = mul(
            maximum(aligned, le(radius, 3.1)), black_anchors
        )

        count = scale(gmean(membership), 4096)
        sums = gather(xy, membership)
        residual = sub(sums, mul(count, candidate_xy))
        low = relu(scale(residual, -1, -0.08))
        high = relu(scale(sub(residual, count), 1, 0.08))
        violation = total(add(low, high))
        enough = positive(scale(count, 1, -1.5))
        contains_selected = gmax(mul(membership, selected_point))

        valid_ball = mul(
            mul(le(violation, 0, 128), enough), exists
        )
        ball_map = maximum(
            s["balls"], mul(candidate, valid_ball)
        )
        features = cat(
            violation, count, contains_selected, enough, exists
        )
        score = linear(
            features, [[-1, 0.01, 1000, 1000, 1000]], [-2900]
        )
        improvement = sub(score, s["best"])
        use = positive(improvement)
        return {
            "remaining": relu(sub(s["remaining"], candidate)),
            "best": blend(use, score, s["best"]),
            "group": blend(use, membership, s["group"]),
            "balls": ball_map,
        }

    discovery = loop(12, {
        "remaining": disks,
        "best": negative_best,
        "group": [1, 64, 64],
        "balls": initial_ball_map,
    }, discover_body)

    confident = positive(scale(discovery["best"], 1, -70))
    black_group = blend(confident, discovery["group"], flood_group)
    initial_group = blend(plain, plain_group, black_group)

    previous_member = mul(old_group, point(msel))
    update = mul(mmove, mi)
    reconciled = add(
        sub(old_group, mul(previous_member, update)),
        mul(selected_point, update),
    )
    group = blend(mi, reconciled, initial_group)
    count = scale(gmean(group), 4096)
    sums = gather(xy, group)

    offsets = linear(
        count, np.ones((9, 1), np.float32),
        -np.arange(9, dtype=np.float32)
    )
    hats = relu(scale(absolute(offsets), -4, 1))
    reciprocal = linear(
        hats, [[0] + [1.0 / j for j in range(1, 9)]]
    )
    mean = mul(sums, reciprocal)
    has_group = positive(scale(count, 1, -0.5))

    centred = absolute(sub(xy, scale(mean, 1, -0.4)))
    centre_error = maximum(sl(centred, 0, 1), sl(centred, 1, 2))
    ball_point = mul(le(centre_error, 0.5, 64), has_group)
    plain_color = blend(mi, mgc, selected_label)
    ball_colors = blend(
        plain, plain_color, gather(disk_palette, ball_point)
    )

    other_balls = mul(discovery["balls"], inv(ball_point))
    pickups = mul(
        disks, inv(maximum(discovery["balls"], ball_point))
    )
    available_colors = gmax(mul(disk_palette, disks))
    other_colors = gmax(mul(disk_palette, other_balls))

    ball_color_count = total(ball_colors)
    overlap = dot_channels(ring_palette, ball_colors)
    compatible = le(sub(ball_color_count, overlap), 0.1)
    available_goal = le(
        dot_channels(ring_palette, inv(available_colors)), 0.1
    )
    unclaimed = le(dot_channels(ring_palette, other_colors), 0.1)
    black_eligible = mul(mul(compatible, available_goal), unclaimed)
    plain_eligible = dot_channels(ring_palette, selected_label)
    eligible = mul(
        rings, blend(plain, plain_eligible, black_eligible)
    )

    from_ball = total(absolute(sub(xy, mean)))
    goal_score = add(
        scale(mul(pair_ring, black), 100),
        scale(from_ball, -0.01),
    )
    goal_score = sub(goal_score, scale(inv(eligible), 1000))
    chosen_goal = winner(goal_score, eligible)
    goal_xy = blend(mi, mgxy, gather(xy, chosen_goal))
    goal_colors = blend(
        mi, mgc, gather(ring_palette, chosen_goal)
    )
    goal_valid = blend(mi, mgv, gmax(chosen_goal))

    missing = mul(goal_colors, inv(ball_colors))
    needs_pickup = positive(total(missing))
    pickup_eligible = mul(
        pickups, clip(dot_channels(disk_palette, missing))
    )
    pickup_score = sub(
        scale(from_ball, -1),
        scale(inv(pickup_eligible), 1000),
    )
    chosen_pickup = winner(pickup_score, pickup_eligible)
    target_xy = blend(
        mul(needs_pickup, gmax(chosen_pickup)),
        gather(xy, chosen_pickup),
        goal_xy,
    )

    matched = le(
        total(absolute(sub(ball_colors, goal_colors))), 0.1
    )
    close_to_goal = le(
        total(absolute(sub(mean, goal_xy))), 3.0
    )
    complete = mul(
        has_group,
        add(
            inv(goal_valid),
            mul(goal_valid, mul(matched, close_to_goal)),
        ),
    )
    running = mul(mul(has_group, selected_ok), inv(complete))

    # Conservative pixel collision masks.
    orange = sl("x0", 10, 11)
    red = sl("x0", 2, 3)
    repaired_orange = maximum(
        orange, mul(lines, expand(orange, 3))
    )
    hazard_bad = expand(maximum(repaired_orange, red), 5)
    wanted_pickup = clip(dot_channels(disk_palette, goal_colors))
    wrong_pickups = mul(mul(pickups, inv(wanted_pickup)), black)
    wrong_region = expand9(wrong_pickups)
    other_region = expand9(other_balls)
    ball_bad = clip(add(add(hazard_bad, wrong_region), other_region))

    diamond_kernel = np.zeros((1, 1, 5, 5), np.float32)
    for r in range(-2, 3):
        for c in range(-2, 3):
            if abs(r) + abs(c) <= 2:
                diamond_kernel[0, 0, r + 2, c + 2] = 1
    wall_bad = clip(conv(red, diamond_kernel))

    # A radius-four exclusion prevents overlapping anchor diamonds,
    # not merely clicks inside their current bounding boxes.
    anchor_boxes = expand9(maximum(anchors, group))
    nonmoving_disks = mul(disks, inv(ball_point))
    disk_exclusion = expand9(nonmoving_disks)
    edge_bad = clip(add(
        add(
            positive(scale(xx, -1, 2.9)),
            positive(scale(xx, 1, -61.1)),
        ),
        add(
            positive(scale(yy, -1, 2.9)),
            positive(scale(yy, 1, -61.1)),
        ),
    ))
    anchor_bad = clip(add(
        add(wall_bad, anchor_boxes),
        add(edge_bad, disk_exclusion),
    ))

    # Positive-valued max-plus iteration: zero padding is now worse than
    # every valid spatial value and cannot create artificial border goals.
    target_distance = total(absolute(sub(xy, target_xy)))
    seed = sub(
        scale(target_distance, -1, 256),
        scale(ball_bad, 1000),
    )

    def shift_max(x, axis, step):
        k = 2 * step + 1
        w = np.zeros((3, 1, k, k), np.float32)
        for j, displacement in enumerate((-step, 0, step)):
            r = step + displacement if axis == 0 else step
            c = step + displacement if axis == 1 else step
            w[j, 0, r, c] = 1
        z = conv(x, w)
        return maximum(
            maximum(sl(z, 0, 1), sl(z, 1, 2)), sl(z, 2, 3)
        )

    def value_body(s):
        z = s["value"]
        for _ in range(2):
            for axis in (0, 1):
                z = shift_max(z, axis, 1)
                z = shift_max(z, axis, 3)
        improved = maximum(s["value"], scale(z, 1, -1))
        return {
            "value": sub(improved, scale(ball_bad, 1000))
        }

    value = loop(16, {"value": seed}, value_body)["value"]
    policy_field = add(value, scale(target_distance, -0.006))

    # Enumerate all active anchors in parallel. Each of eight channels
    # predicts the barycentre produced by moving that particular anchor
    # to each of the 4096 possible click positions.
    active_records = pack(group, 8)
    ay, ax, av = unpack(active_records, 8)
    sum_y, sum_x = sl(sums, 0, 1), sl(sums, 1, 2)
    predicted_y = mul(add(yy, sub(sum_y, ay)), reciprocal)
    predicted_x = mul(add(xx, sub(sum_x, ax)), reciprocal)

    source_w = np.zeros((16, 2), np.float32)
    source_w[:8, 0] = 1
    source_w[8:, 1] = 1
    source = linear(cat(ball_bad, policy_field), source_w)

    # Separable scan-and-gather for all eight predicted positions.
    # Channels 0..7 carry collision flags; 8..15 carry spatial values.
    def warp_axis(source_field, query, axis):
        source_coordinate = yy if axis == 0 else xx
        reduction = "colmax" if axis == 0 else "rowmax"

        def body(s):
            source_line = le(
                absolute(sub(source_coordinate, s["i"])), 0.1
            )
            sample = node(
                reduction, [mul(source_field, source_line)]
            )
            fractional = scale(sub(query, s["i"]), 1, -0.45)
            destination_line = le(absolute(fractional), 0.5, 64)
            doubled_line = cat(destination_line, destination_line)
            accumulated = maximum(
                s["out"], mul(sample, doubled_line)
            )
            return {
                "i": add(s["i"], one),
                "out": accumulated,
            }

        return loop(64, {
            "i": zero,
            "out": [16, 64, 64],
        }, body)["out"]

    warped = warp_axis(source, predicted_y, 0)
    warped = warp_axis(warped, predicted_x, 1)
    predicted_bad = sl(warped, 0, 8)
    predicted_value = sl(warped, 8, 16)

    # Keep the moved anchor clear of its newly positioned disk.
    hidden_anchor = mul(
        le(absolute(sub(yy, predicted_y)), 5.1),
        le(absolute(sub(xx, predicted_x)), 5.1),
    )
    move_bad = clip(add(
        add(anchor_bad, predicted_bad), hidden_anchor
    ))
    move_score = sub(predicted_value, scale(move_bad, 10000))
    move_score = sub(move_score, scale(inv(av), 20000))

    target_y, target_x = sl(target_xy, 0, 1), sl(target_xy, 1, 2)
    fractional_distance = add(
        absolute(sub(predicted_y, target_y)),
        absolute(sub(predicted_x, target_x)),
    )
    move_score = add(
        move_score, scale(fractional_distance, -0.0001)
    )

    sy, sx = sl(selected_xy, 0, 1), sl(selected_xy, 1, 2)
    selected_index = mul(
        av,
        le(add(absolute(sub(ay, sy)), absolute(sub(ax, sx))), 0.1),
    )
    best_values = gmax(move_score)
    current_value = gather(policy_field, ball_point)
    makes_progress = positive(
        scale(sub(best_values, current_value), 1, -0.01), 1000
    )
    current_bonus = scale(mul(selected_index, makes_progress), 0.04)
    index_bias = constant(-0.0002 * np.arange(8, dtype=np.float32))
    channel_bias = add(current_bonus, index_bias)
    ranked = add(best_values, channel_bias)

    # Exact finite choice between vector elements; unlike subtract-and-add
    # maxima, this does not lose precision when invalid scores are large.
    def vector_maximum(a, b):
        choose_b = positive(sub(b, a), 1.0e7)
        return blend(choose_b, b, a)

    best = ranked
    for width in (4, 2, 1):
        best = vector_maximum(
            sl(best, 0, width), sl(best, width, 2 * width)
        )
    choice = mul(
        av, le(absolute(sub(ranked, best)), 0, 1.0e7)
    )
    chosen_anchor_xy = cat(
        total(mul(ay, choice)), total(mul(ax, choice))
    )
    chosen_is_selected = clip(total(mul(choice, selected_index)))
    do_move = mul(running, chosen_is_selected)

    chosen_move_map = total(mul(
        add(move_score, channel_bias), choice
    ))
    selection_score = scale(
        total(absolute(sub(xy, chosen_anchor_xy))), -1
    )
    active_score = blend(
        chosen_is_selected, chosen_move_map, selection_score
    )

    done_map = maximum(old_done, mul(group, complete))
    new_mask = mul(anchors, inv(done_map))
    new_score = sub(
        scale(key, 1.0 / 4096),
        scale(inv(new_mask), 10000),
    )
    final_score = blend(running, active_score, new_score)
    scale(final_score, 1, name="click")

    emitted_point = winner(final_score)
    emitted_xy = gather(xy, emitted_point)
    done_records = pack(done_map, 16)
    memory = cat(
        active_records, done_records,
        goal_xy, goal_colors, goal_valid, running,
        selected_xy, do_move, emitted_xy, plain,
    )
    scale(memory, 1, name="mem_next")

    bias = np.zeros(8, np.float32)
    bias[6] = 1
    linear(
        "avail", np.zeros((8, 8), np.float32),
        bias, name="logits"
    )

    # Pure structural accounting, without examples or game simulation.
    def counts(nodes):
        node_count = evaluations = macs = 0
        for nd in nodes:
            node_count += 1
            if nd["op"] == "loop":
                n2, e2, m2 = counts(nd["body"])
                node_count += n2
                evaluations += 1 + nd["T"] * e2
                macs += nd["T"] * m2
                continue
            evaluations += 1
            op = nd["op"]
            if op == "conv":
                macs += int(params[nd["w"]].size) * 4096
            elif op == "dense":
                macs += int(params[nd["w"]].size)
            elif op in ("add", "sub", "mul", "max2", "min2"):
                macs += shapes[nd["id"]][1] * 4096
            elif op in (
                "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"
            ):
                macs += shapes[nd["in"][0]][1] * 4096
            elif op in ("vadd", "vmul"):
                macs += shapes[nd["id"]][1]
        return node_count, evaluations, macs

    nc, ne, macs = counts(graph)
    assert nc <= 1000
    assert ne <= 10000
    assert macs <= 1_500_000_000
    assert sum(a.size for a in params.values()) <= 150000
    assert all(np.isfinite(a).all() for a in params.values())
    assert all(
        channels <= (512 if kind == "s" else 4096)
        for kind, channels in shapes.values()
    )
    assert sum(
        channels for kind, channels in shapes.values() if kind == "s"
    ) <= 4096
    assert shapes["mem_next"] == ("v", 91)

    return {
        "graph": graph,
        "memory_size": 91,
        "params": params,
    }
