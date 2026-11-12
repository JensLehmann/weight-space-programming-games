import numpy as np


def build():
    graph, params = [], {}
    pcache, root_cache = {}, {}
    serial = [0]
    target, cache = [graph], [root_cache]

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in pcache:
            name = fresh("p")
            pcache[key] = name
            params[name] = a
        return pcache[key]

    def node(op, inputs, name=None, **fields):
        key = (op, tuple(inputs), tuple(sorted(fields.items())))
        if name is None and key in cache[0]:
            return cache[0][key]
        name = name or fresh()
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(fields)
        target[0].append(d)
        if name not in ("logits", "click", "mem_next"):
            cache[0][key] = name
        return name

    def conv(x, w, b=None, name=None):
        f = {"w": parameter(w)}
        if b is not None:
            f["b"] = parameter(b)
        return node("conv", [x], name, **f)

    def dense(x, w, b=None, name=None):
        f = {"w": parameter(w)}
        if b is not None:
            f["b"] = parameter(b)
        return node("dense", [x], name, **f)

    def const(values):
        a = np.asarray(values, np.float32).reshape(-1)
        old_target, old_cache = target[0], cache[0]
        target[0], cache[0] = graph, root_cache
        z = dense("avail", np.zeros((len(a), 8), np.float32), a)
        target[0], cache[0] = old_target, old_cache
        cache[0].update(root_cache)
        return z

    constants = {}

    def k(value):
        value = float(value)
        if value not in constants:
            constants[value] = const([value])
        return constants[value]

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def mn(a, b):
        return node("min2", [a, b])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

    def relu(x):
        return node("relu", [x])

    def vrelu(x):
        return node("vrelu", [x])

    def gm(x):
        return node("gmax", [x])

    def affine(x, scale, bias=0, channels=1, name=None):
        w = np.eye(channels, dtype=np.float32)[:, :, None, None] * scale
        return conv(x, w, np.full(channels, bias, np.float32), name)

    def vaffine(x, scale, bias=0, size=1):
        return dense(
            x, np.eye(size, dtype=np.float32) * scale,
            np.full(size, bias, np.float32)
        )

    def subv(a, b, size):
        return va(a, dense(b, -np.eye(size, dtype=np.float32)))

    def inv(x):
        return affine(x, -1, 1)

    def invv(x):
        return vaffine(x, -1, 1)

    def clip1(x):
        return mn(relu(x), k(1))

    def positive(x):
        return mn(x, k(1))

    def sum_channels(x, count):
        return conv(x, np.ones((1, count, 1, 1), np.float32))

    def sum_vector(x, count):
        return dense(x, np.ones((1, count), np.float32))

    def choose(x, selector, count):
        return sum_channels(mul(x, selector), count)

    def shift(x, dy, dx, channels=1, bias=None):
        if dy == 0 and dx == 0 and bias is None:
            return x
        extent = max(abs(dy), abs(dx))
        if extent > 3:
            cy, cx = max(-3, min(3, dy)), max(-3, min(3, dx))
            return shift(
                shift(x, cy, cx, channels),
                dy - cy, dx - cx, channels, bias
            )
        size = 2 * extent + 1
        w = np.zeros((channels, channels, size, size), np.float32)
        for c in range(channels):
            w[c, c, size // 2 + dy, size // 2 + dx] = 1
        b = None if bias is None else np.full(channels, bias, np.float32)
        return conv(x, w, b)

    def correlate(x, input_channels, patterns):
        points = [p for pattern in patterns for p in pattern]
        ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
        xmin, xmax = min(p[2] for p in points), max(p[2] for p in points)

        def center(v, lo, hi):
            if lo >= -3 and hi <= 3:
                return 0
            return lo + 3 + 7 * ((v - lo) // 7)

        groups = {}
        outputs = len(patterns)
        for o, pattern in enumerate(patterns):
            for c, dy, dx, coefficient in pattern:
                sy, sx = center(dy, ymin, ymax), center(dx, xmin, xmax)
                if (sy, sx) not in groups:
                    groups[sy, sx] = np.zeros(
                        (outputs, input_channels, 7, 7), np.float32
                    )
                groups[sy, sx][o, c, dy - sy + 3, dx - sx + 3] += coefficient
        result = None
        for (sy, sx), w in sorted(groups.items()):
            part = conv(shift(x, sy, sx, input_channels), w)
            result = part if result is None else add(result, part)
        return result

    def exact_count(x, counts):
        counts = np.asarray(counts, np.float32)
        n = len(counts)
        return mn(relu(conv(
            x, 2 * np.eye(n, dtype=np.float32)[:, :, None, None],
            1 - 2 * counts
        )), k(1))

    def zero_count(x, channels):
        return relu(affine(x, -1, 1, channels))

    for value in (1, 5, 1000):
        k(value)

    coord = sl("coords", 0, 2)
    one_plane = sl("coords", 2, 3)
    zero_plane = affine(one_plane, 0)
    directions = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    tie_map = conv(
        coord, np.asarray([-.5, -.02], np.float32).reshape(1, 2, 1, 1),
        np.asarray([1000], np.float32)
    )

    def position(marker):
        return vaffine(node("gmean", [mul(coord, marker)]), 4096, size=2)

    def distance_map(reference):
        return sum_channels(node("abs", [sub(coord, reference)]), 2)

    def marker_at(reference):
        return relu(affine(distance_map(reference), -10000, 1))

    def pick_score(score, eligible):
        return mul(
            clip1(affine(sub(score, gm(score)), 1000000, 1)), eligible
        )

    def pick_near(eligible, reference):
        return pick_score(
            mul(eligible, sub(
                tie_map, affine(distance_map(reference), 100)
            )), eligible
        )

    def max_channels(x, size):
        best = sl(x, 0, 1)
        for i in range(1, size):
            best = mx(best, sl(x, i, i + 1))
        return best

    def winner_vector(values, size):
        best = gm(max_channels(add(zero_plane, values), size))
        diff = va(values, dense(best, -np.ones((size, 1), np.float32)))
        return vrelu(vaffine(diff, 10000, 1, size))

    def extract_values(maps, marker, size, biases):
        values = node("gmean", [mul(maps, marker)])
        return va(vaffine(values, 4096, size=size), const(biases))

    def displacement(direction, amount):
        return dense(direction, np.asarray([
            [-amount, amount, 0, 0],
            [0, 0, -amount, amount]
        ], np.float32))

    # Exact row and column residue planes for the three-pixel action lattice.
    residue_w = np.zeros((128, 2, 1, 1), np.float32)
    residue_b = np.zeros(128, np.float32)
    for axis in range(2):
        for t in range(64):
            j = axis * 64 + t
            residue_w[j, axis, 0, 0] = 63
            residue_b[j] = -t
    residue_difference = node(
        "abs", [conv(coord, residue_w, residue_b)]
    )
    residue_tents = relu(sub(k(1), residue_difference))
    residue_sum = np.zeros((6, 128, 1, 1), np.float32)
    for axis in range(2):
        for t in range(64):
            residue_sum[axis * 3 + t % 3, axis * 64 + t, 0, 0] = 1
    residue_planes = conv(residue_tents, residue_sum)

    def lattice(marker):
        selected = gm(mul(residue_planes, marker))
        matched = mul(residue_planes, selected)
        row = sum_channels(sl(matched, 0, 3), 3)
        col = sum_channels(sl(matched, 3, 6), 3)
        # Round the analytically binary result against coordinate roundoff.
        return relu(affine(mul(row, col), 2, -1))

    dimensions = [(3, 3), (3, 6), (6, 3), (6, 6)]
    footprints = []
    for height, width in dimensions:
        ay, ax = (height - 1) // 2, (width - 1) // 2
        footprints.append([
            (y - ay, x - ax)
            for y in range(height) for x in range(width)
        ])
    footprints.append([
        (y - 4, x - 4)
        for y in range(9) for x in range(9)
        if 3 <= y <= 5 or 3 <= x <= 5
    ])
    footprint_patterns = [
        [(0, dy, dx, 1) for dy, dx in fp] for fp in footprints
    ]

    def footprint_counts(pixel_map):
        return correlate(pixel_map, 1, footprint_patterns)

    w = np.zeros((3, 16, 1, 1), np.float32)
    w[0, [0, 14], 0, 0] = 1
    w[1, [0, 4, 5], 0, 0] = 1
    w[2, 0, 0, 0] = 1
    categories = conv("x0", w)

    patterns, counts = [], []
    for active_only in (False, True):
        for height, width in dimensions:
            ay, ax = (height - 1) // 2, (width - 1) // 2
            center_rows = [1] if height == 3 else [2, 3]
            center_cols = [1] if width == 3 else [2, 3]
            pattern = []
            for y in range(height):
                for x in range(width):
                    interior = y in center_rows and x in center_cols
                    c = (2 if active_only else 1) if interior else 0
                    pattern.append((c, y - ay, x - ax, 1))
            patterns.append(pattern)
            counts.append(height * width)
    detected = exact_count(correlate(categories, 3, patterns), counts)
    yellows = sl(detected, 0, 4)
    active_map = sum_channels(sl(detected, 4, 8), 4)
    cross = exact_count(
        correlate(sl("x0", 11, 12), 1, [footprint_patterns[4]]),
        [len(footprints[4])]
    )
    objects = cat(yellows, cross)
    yellow_anchors = clip1(sum_channels(yellows, 4))

    def object_type(marker):
        return gm(mul(objects, marker))

    reconstruction = [[]]
    for c, fp in enumerate(footprints):
        reconstruction[0].extend((c, -dy, -dx, 1) for dy, dx in fp)

    def pixels_of(marker):
        return positive(correlate(mul(objects, marker), 5, reconstruction))

    all_pixels = positive(correlate(objects, 5, reconstruction))

    goal_patterns, goal_counts = [], []
    for height, width in dimensions:
        gh, gw = height + 2, width + 2
        ay, ax = 1 + (height - 1) // 2, 1 + (width - 1) // 2
        pattern = [
            (0, y - ay, x - ax, 1)
            for y in range(gh) for x in range(gw)
            if y in (0, gh - 1) or x in (0, gw - 1)
        ]
        goal_patterns.append(pattern)
        goal_counts.append(len(pattern))
    plus_outline = [
        "00011111000", "00010001000", "00010001000",
        "11110001111", "10000000001", "10000000001",
        "10000000001", "11110001111", "00010001000",
        "00010001000", "00011111000"
    ]
    pattern = [
        (0, y - 5, x - 5, 1)
        for y, row in enumerate(plus_outline)
        for x, bit in enumerate(row) if bit == "1"
    ]
    goal_patterns.append(pattern)
    goal_counts.append(len(pattern))
    goals = exact_count(
        correlate(sl("x0", 4, 5), 1, goal_patterns), goal_counts
    )
    unfinished = relu(sub(goals, objects))
    unfinished_any = clip1(sum_channels(unfinished, 5))
    priority = conv(
        unfinished,
        np.asarray([1000, 1010, 1020, 1030, 1100], np.float32)
        .reshape(1, 5, 1, 1)
    )
    fresh_goal = pick_score(
        mul(unfinished_any, add(priority, sub(tie_map, k(1000)))),
        unfinished_any
    )
    fresh_type = gm(mul(goals, fresh_goal))

    memory_position = vs("mem", 0, 2)
    memory_goal = vs("mem", 2, 4)
    memory_flag = vs("mem", 4, 5)
    memory_type = vs("mem", 5, 10)
    remembered_goal = marker_at(memory_goal)
    remembered_done = gm(mul(
        remembered_goal, choose(objects, memory_type, 5)
    ))
    retain_goal = vm(memory_flag, invv(remembered_done))
    selected_goal = add(
        mul(remembered_goal, retain_goal),
        mul(fresh_goal, invv(retain_goal))
    )
    goal_type = va(
        vm(memory_type, retain_goal),
        vm(fresh_type, invv(retain_goal))
    )
    goal_position = position(selected_goal)
    reference = va(
        vm(memory_position, retain_goal),
        vm(goal_position, invv(retain_goal))
    )
    primary = pick_near(
        clip1(choose(relu(sub(objects, goals)), goal_type, 5)),
        reference
    )
    primary_type = object_type(primary)
    primary_position = position(primary)
    primary_pixels = pixels_of(primary)
    primary_cross = vs(primary_type, 4, 5)
    primary_yellow = invv(primary_cross)

    original_gap = sl("x0", 15, 16)
    closed_gap = original_gap
    for vertical in (False, True):
        line = [
            (0, t if vertical else 0, 0 if vertical else t, 1)
            for t in range(-6, 7)
        ]
        dilated = positive(correlate(original_gap, 1, [line]))
        closed = exact_count(correlate(dilated, 1, [line]), [13])
        closed_gap = mx(closed_gap, closed)
    gap = mx(original_gap, mul(closed_gap, all_pixels))

    wall_w = np.zeros((1, 16, 1, 1), np.float32)
    wall_w[0, [2, 12, 13], 0, 0] = 1
    base_b = zero_count(footprint_counts(conv("x0", wall_w)), 5)
    footprint_gap = positive(footprint_counts(gap))
    base_free = mul(base_b, affine(footprint_gap, -1, 1, channels=5))

    def layout(typ, pixel_map, controlled=True):
        clear = zero_count(
            footprint_counts(relu(sub(all_pixels, pixel_map))), 5
        )
        b = choose(base_b, typ, 5)
        a = choose(mul(base_free, clear), typ, 5)
        f = choose(footprint_gap, typ, 5)
        if not controlled:
            a = mul(a, primary_yellow)
        return b, a, f

    def begin_body():
        saved = target[0], cache[0]
        body = []
        target[0], cache[0] = body, dict(root_cache)
        return body, saved

    def end_body(saved):
        target[0], cache[0] = saved

    direction_kernel = np.zeros((4, 4, 3, 3), np.float32)
    for j, (dy, dx) in enumerate(directions):
        direction_kernel[j, j, 1 + dy // 3, 1 + dx // 3] = 1

    def directional_shift(x):
        # Three one-pixel shifts equal one three-pixel shift.
        for _ in range(3):
            x = conv(x, direction_kernel)
        return x

    def four_shifts(x, bias=0):
        w = np.zeros((4, 1, 7, 7), np.float32)
        for j, (dy, dx) in enumerate(directions):
            w[j, 0, 3 + dy, 3 + dx] = 1
        return conv(x, w, np.full(4, bias, np.float32))

    def virtual_planner(goal, b, a, f, iterations):
        bnext = four_shifts(b)
        normal_limit = affine(mul(a, four_shifts(a)), 1000, channels=4)
        passive_limit = affine(bnext, 1000, channels=4)
        blocked = affine(bnext, -1, 1, channels=4)
        continuation = mul(f, bnext)
        continuation_limit = affine(continuation, 1000, channels=4)
        stopped = affine(continuation, -1, 1, channels=4)
        seed = affine(goal, 1000)

        lid, ds, gs = fresh("L"), fresh("s"), fresh("s")
        rs = [fresh("s") for _ in range(4)]
        states = [
            {"id": ds, "init": seed},
            {"id": gs, "zeros": [4, 64, 64]}
        ] + [{"id": s, "zeros": [4, 64, 64]} for s in rs]

        body, saved = begin_body()
        wall_stop = mul(ds, blocked)
        minimum_stop = mul(ds, stopped)
        gshift = directional_shift(gs)
        next_states = {
            gs: mx(minimum_stop, mn(gshift, continuation_limit))
        }
        incoming = gshift
        for stage in range(5):
            if stage:
                incoming = directional_shift(rs[stage - 1])
            endpoint = mx(wall_stop, mn(incoming, passive_limit))
            if stage < 4:
                next_states[rs[stage]] = endpoint
        passive_best = sub(max_channels(endpoint, 4), k(5))
        normal_best = max_channels(
            mn(four_shifts(ds, -1), normal_limit), 4
        )
        next_states[ds] = mul(mx(mx(ds, passive_best), normal_best), b)
        end_body(saved)
        graph.append({
            "id": lid, "op": "loop", "T": iterations,
            "state": states, "body": body, "next": next_states
        })
        final = lid + "." + ds
        endpoint = mx(
            mul(final, blocked),
            mn(directional_shift(lid + "." + rs[3]), passive_limit)
        )
        return cat(
            mn(four_shifts(final, -1), normal_limit),
            sub(endpoint, k(5))
        )

    pb, pa, pf = layout(primary_type, primary_pixels, controlled=False)
    primary_values = virtual_planner(selected_goal, pb, pa, pf, 64)
    primary_q = extract_values(
        primary_values, primary, 8,
        [.07, .06, .05, .04, .03, .02, .01, 0]
    )
    penalty = np.zeros((8, 1), np.float32)
    penalty[:4, 0] = -2000
    primary_q = va(primary_q, dense(primary_cross, penalty))
    primary_winner = winner_vector(primary_q, 8)
    pn_dir = vs(primary_winner, 0, 4)
    pi_dir = vs(primary_winner, 4, 8)
    pn_flag, pi_flag = sum_vector(pn_dir, 4), sum_vector(pi_dir, 4)

    def push_sites(target_pixels, helper_type, direction, helper):
        collision = positive(choose(
            footprint_counts(target_pixels), helper_type, 5
        ))
        free = choose(base_free, helper_type, 5)
        sites = mul(
            mul(four_shifts(collision), inv(collision)),
            mul(free, four_shifts(free))
        )
        return mul(
            clip1(choose(sites, direction, 4)), lattice(helper)
        )

    behind_primary = subv(
        primary_position, displacement(pi_dir, 6 / 63), 2
    )
    helper_candidates = relu(sub(yellow_anchors, primary))
    helper_exists = gm(helper_candidates)
    helper = add(
        pick_near(helper_candidates, behind_primary),
        mul(primary, vm(invv(helper_exists), primary_yellow))
    )
    helper_type = object_type(helper)
    helper_pixels = pixels_of(helper)
    helper_position = position(helper)
    helper_goal = push_sites(primary_pixels, helper_type, pi_dir, helper)
    helper_at_goal = gm(mul(helper_goal, helper))
    hb, ha, hf = layout(helper_type, helper_pixels)
    helper_values = virtual_planner(helper_goal, hb, ha, hf, 64)
    helper_q = extract_values(
        helper_values, helper, 8,
        [.07, .06, .05, .04, .03, .02, .01, 0]
    )
    helper_winner = winner_vector(helper_q, 8)
    hn_dir, hi_dir = vs(helper_winner, 0, 4), vs(helper_winner, 4, 8)
    hn_flag, hi_flag = sum_vector(hn_dir, 4), sum_vector(hi_dir, 4)
    launch_direction = va(
        vm(hi_dir, helper_exists), vm(pi_dir, invv(helper_exists))
    )

    behind_helper = subv(
        helper_position, displacement(launch_direction, 6 / 63), 2
    )
    assistant_candidates = relu(sub(yellow_anchors, helper))
    assistant_exists = gm(assistant_candidates)
    assistant = add(
        pick_near(assistant_candidates, behind_helper),
        mul(helper, invv(assistant_exists))
    )
    assistant_type = object_type(assistant)
    assistant_pixels = pixels_of(assistant)
    assistant_push_goal = mul(
        push_sites(
            helper_pixels, assistant_type, launch_direction, assistant
        ), assistant_exists
    )
    _, aa, _ = layout(assistant_type, assistant_pixels)

    bomb_w = np.zeros((1, 16, 1, 1), np.float32)
    bomb_w[0, [12, 13], 0, 0] = 1
    bomb_anchors = exact_count(correlate(
        conv("x0", bomb_w), 1, [[
            (0, y - 2, x - 2, 1)
            for y in range(6) for x in range(6)
        ]]
    ), [36])

    staging_patterns = []
    for height, width in dimensions:
        horizontal = [-1, 2] if width == 3 else [0]
        vertical = [-1, 2] if height == 3 else [0]
        up = [-4, -7, -10] if height == 3 else [-6, -9, -12]
        down = [5, 8, 11] if height == 3 else [6, 9, 12]
        left = [-4, -7, -10] if width == 3 else [-6, -9, -12]
        right = [5, 8, 11] if width == 3 else [6, 9, 12]
        groups = [
            [(y, x) for y in up for x in horizontal],
            [(y, x) for y in down for x in horizontal],
            [(y, x) for y in vertical for x in left],
            [(y, x) for y in vertical for x in right]
        ]
        staging_patterns.extend(
            [(0, -dy, -dx, 1) for dy, dx in group]
            for group in groups
        )
    staging = positive(correlate(bomb_anchors, 1, staging_patterns))
    selected_staging = [
        choose(
            sl(staging, 4 * shape, 4 * shape + 4),
            launch_direction, 4
        ) for shape in range(4)
    ]
    bomb_goal = mul(
        mul(clip1(choose(
            cat(*selected_staging), vs(assistant_type, 0, 4), 4
        )), aa),
        lattice(assistant)
    )
    final_seed = mx(
        affine(assistant_push_goal, 1000), affine(bomb_goal, 900)
    )

    def normal_planner(seed, valid, iterations):
        limits = affine(
            mul(valid, four_shifts(valid)), 1000, channels=4
        )
        lid, sid = fresh("L"), fresh("s")
        body, saved = begin_body()
        best = mx(
            sid, max_channels(mn(four_shifts(sid, -1), limits), 4)
        )
        final = mul(best, valid)
        end_body(saved)
        graph.append({
            "id": lid, "op": "loop", "T": iterations,
            "state": [{"id": sid, "init": seed}],
            "body": body, "next": {sid: final}
        })
        return mn(four_shifts(lid + "." + sid, -1), limits)

    final_values = normal_planner(final_seed, aa, 64)
    final_q = extract_values(
        final_values, assistant, 4, [.03, .02, .01, 0]
    )
    assistant_at_goal = gm(mul(
        assistant, clip1(add(assistant_push_goal, bomb_goal))
    ))
    assistant_direction = va(
        vm(launch_direction, assistant_at_goal),
        vm(winner_vector(final_q, 4), invv(assistant_at_goal))
    )

    not_helper_goal = invv(helper_at_goal)
    helper_normal = vm(not_helper_goal, hn_flag)
    helper_impulse = vm(not_helper_goal, hi_flag)
    primary_phase = pn_flag
    helper_phase = vm(
        pi_flag, vm(helper_exists, va(helper_at_goal, helper_normal))
    )
    assistant_phase = vm(
        pi_flag,
        va(invv(helper_exists), vm(helper_exists, helper_impulse))
    )
    helper_direction = va(
        vm(pi_dir, helper_at_goal), vm(hn_dir, not_helper_goal)
    )
    desired_piece = add(
        add(mul(primary, primary_phase), mul(helper, helper_phase)),
        mul(assistant, assistant_phase)
    )
    desired_direction = va(
        va(vm(pn_dir, primary_phase), vm(helper_direction, helper_phase)),
        vm(assistant_direction, assistant_phase)
    )

    # A detector/task failure must not produce a click at the empty corner.
    desired_exists = gm(desired_piece)
    fallback = pick_near(yellow_anchors, goal_position)
    desired_piece = add(
        desired_piece, mul(fallback, invv(desired_exists))
    )
    desired_is_active = gm(mul(desired_piece, active_map))
    executed_direction = vm(desired_direction, desired_is_active)

    output_w = np.zeros((8, 5), np.float32)
    for j in range(4):
        output_w[j + 1, j] = 10
    output_w[6, 4] = -5
    output_b = np.full(8, -1000, np.float32)
    output_b[1:5] = 0
    output_b[6] = 2
    dense(
        vcat(executed_direction, desired_is_active),
        output_w, output_b, name="logits"
    )
    affine(desired_piece, 1, name="click")

    ordinary_displacement = vm(
        displacement(pn_dir, 3 / 63),
        vm(primary_phase, desired_is_active)
    )
    launch_displacement = vm(
        displacement(pi_dir, 15 / 63),
        vm(vm(helper_phase, helper_at_goal), desired_is_active)
    )
    next_position = va(
        primary_position, va(ordinary_displacement, launch_displacement)
    )
    node(
        "vconcat",
        [next_position, goal_position, gm(selected_goal), goal_type],
        name="mem_next"
    )

    # Static dead-code elimination.
    needed = {"logits", "click", "mem_next"}
    kept = []
    for d in reversed(graph):
        if d["op"] == "loop":
            aliases = {d["id"] + "." + s["id"] for s in d["state"]}
            if not aliases.intersection(needed):
                continue
            local_ids = {s["id"] for s in d["state"]}
            local_ids.update(n["id"] for n in d["body"])
            for s in d["state"]:
                if "init" in s:
                    needed.add(s["init"])
            for n in d["body"]:
                needed.update(x for x in n["in"] if x not in local_ids)
            kept.append(d)
        elif d["id"] in needed:
            needed.update(d["in"])
            kept.append(d)
    graph = list(reversed(kept))

    used_params = set()
    for d in graph:
        for n in d["body"] if d["op"] == "loop" else [d]:
            for field in ("w", "b"):
                if field in n:
                    used_params.add(n[field])
    params = {name: a for name, a in params.items() if name in used_params}

    # Shape and cost accounting only; no game or observation evaluation.
    shapes = {
        "x0": (16, 64, 64), "coords": (3, 64, 64),
        "mem": (10,), "aprev": (8,), "avail": (8,)
    }
    loop_costs = []
    root_macs = root_evals = node_count = 0
    spatial_channels = 19

    def infer(d, env):
        op = d["op"]
        xs = [env[x] for x in d["in"]]
        a = xs[0]
        cost = 0
        if op == "conv":
            w = params[d["w"]]
            out, cost = (w.shape[0], 64, 64), int(w.size) * 4096
        elif op == "dense":
            w = params[d["w"]]
            out, cost = (w.shape[0],), int(w.size)
        elif op in ("concat", "vconcat"):
            out = (sum(s[0] for s in xs),) + a[1:]
        elif op in ("slice", "vslice"):
            out = (d["stop"] - d["start"],) + a[1:]
        elif op in ("gmean", "gmax"):
            out, cost = (a[0],), a[0] * 4096
        elif op in ("rowmax", "colmax", "rowsum", "colsum"):
            out, cost = a, a[0] * 4096
        elif op in ("add", "sub", "mul", "max2", "min2"):
            out = (max(s[0] for s in xs), 64, 64)
            cost = out[0] * 4096
        elif op in ("vadd", "vmul"):
            out = (max(s[0] for s in xs),)
            cost = out[0]
        else:
            out = a
        env[d["id"]] = out
        return cost, out

    for d in graph:
        node_count += 1
        root_evals += 1
        if d["op"] != "loop":
            cost, shape = infer(d, shapes)
            root_macs += cost
            if len(shape) == 3:
                spatial_channels += shape[0]
            continue
        local = dict(shapes)
        for s in d["state"]:
            shape = shapes[s["init"]] if "init" in s else tuple(s["zeros"])
            local[s["id"]] = shape
            if len(shape) == 3:
                spatial_channels += shape[0]
        body_cost = 0
        for n in d["body"]:
            node_count += 1
            cost, shape = infer(n, local)
            body_cost += cost
            if len(shape) == 3:
                spatial_channels += shape[0]
        for s in d["state"]:
            shapes[d["id"] + "." + s["id"]] = local[d["next"][s["id"]]]
        loop_costs.append((d, body_cost, len(d["body"])))

    while True:
        macs = root_macs + sum(d["T"] * c for d, c, e in loop_costs)
        evaluations = root_evals + sum(d["T"] * e for d, c, e in loop_costs)
        if macs <= 1480000000 and evaluations <= 9800:
            break
        candidates = [z for z in loop_costs if z[0]["T"] > 1]
        if macs > 1480000000:
            chosen = max(candidates, key=lambda z: z[1] * z[0]["T"])
        else:
            chosen = max(candidates, key=lambda z: z[2] * z[0]["T"])
        chosen[0]["T"] -= 1

    assert node_count <= 1000
    assert spatial_channels <= 4096
    assert sum(a.size for a in params.values()) <= 150000
    return {"graph": graph, "memory_size": 10, "params": params}
