import numpy as np


def build():
    graph, params, cache = [], {}, {}
    shapes = {
        "x0": ("s", 16), "coords": ("s", 3),
        "mem": ("v", 60), "aprev": ("v", 8), "avail": ("v", 8)
    }
    counter, destination, macs = [0], [graph], [0]

    def param(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p" + str(len(params))
            cache[key] = name
            params[name] = a
        return cache[key]

    def node(op, inputs, name=None, **fields):
        if name is None:
            name = "n" + str(counter[0])
            counter[0] += 1
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        destination[0].append(nd)
        ss = [shapes[x] for x in inputs]
        if op == "conv":
            w = params[fields["w"]]
            shape = ("s", w.shape[0])
            macs[0] += w.size * 4096
        elif op == "dense":
            w = params[fields["w"]]
            shape = ("v", w.shape[0])
            macs[0] += w.size
        elif op in ("concat", "vconcat"):
            shape = (ss[0][0], sum(s[1] for s in ss))
        elif op in ("slice", "vslice"):
            shape = (ss[0][0], fields["stop"] - fields["start"])
        elif op in ("gmax", "gmean"):
            shape = ("v", ss[0][1])
            macs[0] += ss[0][1] * 4096
        elif op in ("rowmax", "colmax", "rowsum", "colsum"):
            shape = ss[0]
            macs[0] += shape[1] * 4096
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = ("s", max(s[1] for s in ss))
            macs[0] += shape[1] * 4096
        elif op in ("vadd", "vmul"):
            shape = ("v", max(s[1] for s in ss))
            macs[0] += shape[1]
        else:
            shape = ss[0]
        shapes[name] = shape
        return name

    def conv(x, w, b=None, name=None):
        fields = {"w": param(w)}
        if b is not None and np.any(np.asarray(b) != 0):
            fields["b"] = param(b)
        return node("conv", [x], name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": param(w)}
        if b is not None and np.any(np.asarray(b) != 0):
            fields["b"] = param(b)
        return node("dense", [x], name, **fields)

    def sl(x, start, stop):
        return node(
            "slice" if shapes[x][0] == "s" else "vslice",
            [x], start=start, stop=stop
        )

    def cat(xs, vector=False):
        if len(xs) == 1:
            return xs[0]
        return node("vconcat" if vector else "concat", xs)

    def sa(x, scale=1, bias=0, name=None):
        c = shapes[x][1]
        return conv(
            x, (np.eye(c, dtype=np.float32) * scale)[:, :, None, None],
            np.full(c, bias, np.float32), name
        )

    def va(x, scale=1, bias=0, name=None):
        c = shapes[x][1]
        return dense(
            x, np.eye(c, dtype=np.float32) * scale,
            np.full(c, bias, np.float32), name
        )

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

    def sumv(xs):
        result = xs[0]
        for x in xs[1:]:
            result = vadd(result, x)
        return result

    def vnot(x):
        return va(x, -1, 1)

    def vmax(a, b):
        return vadd(b, node("vrelu", [vadd(a, va(b, -1))]))

    def vabs(x):
        return vadd(node("vrelu", [x]), node("vrelu", [va(x, -1)]))

    def constant(values):
        values = np.asarray(values, np.float32).reshape(-1)
        return dense(
            "avail", np.zeros((len(values), 8), np.float32), values
        )

    one = sl("coords", 2, 3)
    vone = node("gmax", [one])
    vzero = constant([0])
    row = sa(sl("coords", 0, 1), 63)
    col = sa(sl("coords", 1, 2), 63)

    def clip(x):
        return node("min2", [node("relu", [x]), one])

    def vclip(x):
        y = node("vrelu", [x])
        return va(node("vrelu", [va(y, -1, 1)]), -1, 1)

    def equal(a, b):
        return node("relu", [
            sa(node("abs", [sub(a, b)]), -1, 1)
        ])

    def vequal(a, b):
        return node("vrelu", [
            va(vabs(vadd(a, va(b, -1))), -1, 1)
        ])

    def channel_sum(x):
        return conv(x, np.ones((1, shapes[x][1], 1, 1), np.float32))

    def vector_sum(x):
        return dense(x, np.ones((1, shapes[x][1]), np.float32))

    def repeat_spatial(x, count):
        return conv(x, np.ones((count, 1, 1, 1), np.float32))

    def repeat_vector(x, count):
        return dense(x, np.ones((count, 1), np.float32))

    def shift(x, dy, dx):
        c = shapes[x][1]
        w = np.zeros((c, c, 7, 7), np.float32)
        for i in range(c):
            w[i, i, 3 - 3 * dy, 3 - 3 * dx] = 1
        return conv(conv(x, w), w)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def shift_four(x, reverse=False):
        w = np.zeros((4, 4, 7, 7), np.float32)
        for i, (dy, dx) in enumerate(directions):
            if reverse:
                dy, dx = -dy, -dx
            w[i, i, 3 - 3 * dy, 3 - 3 * dx] = 1
        return conv(conv(x, w), w)

    # Locate the board from solid background-four patches.
    main = clip(sa(row, -100, 5250))
    w = np.zeros((1, 16, 3, 3), np.float32)
    w[0, 4] = 1
    interior = mul(node("relu", [conv("x0", w, [-8])]), main)

    w = np.zeros((4, 3, 1, 1), np.float32)
    w[0, 1, 0, 0], w[0, 2, 0, 0] = -1, 1
    w[1, 0, 0, 0], w[1, 2, 0, 0] = -1, 1
    w[2, 1, 0, 0], w[3, 0, 0, 0] = 1, 1
    extrema = node("gmax", [mul(conv("coords", w), interior)])
    bounds = dense(
        extrema, np.diag([-63, -63, 63, 63]).astype(np.float32),
        [62, 62, -2, -2]
    )
    ox, oy, mx, my = [sl(bounds, i, i + 1) for i in range(4)]

    def bands(coordinate, origin):
        z = conv(
            sub(coordinate, origin),
            np.ones((7, 1, 1, 1), np.float32),
            -np.arange(7, dtype=np.float32) * 6 - 2
        )
        z = clip(sa(node("abs", [z]), -100, 50))
        return [sl(z, i, i + 1) for i in range(7)]

    cb, rb = bands(col, ox), bands(row, oy)
    lattice = mul(channel_sum(cat(cb)), channel_sum(cat(rb)))
    board = mul(
        lattice,
        mul(
            clip(sa(sub(mx, col), 100, 50)),
            clip(sa(sub(my, row), 100, 50))
        )
    )

    # Four bead corners avoid the black match-marker pixels.
    colours = [i for i in range(1, 16) if i not in (4, 5)]
    corners = [(-1, -1), (-1, 2), (2, -1), (2, 2)]
    w = np.zeros((len(colours) + 1, 16, 7, 7), np.float32)
    for k, colour in enumerate(colours + [5]):
        for dy, dx in corners:
            w[k, colour, dy + 3, dx + 3] = 1
    patches = node("relu", [
        conv("x0", w, np.full(len(colours) + 1, -3, np.float32))
    ])
    colour_image = conv(
        sl(patches, 0, len(colours)),
        np.asarray(colours, np.float32).reshape(1, -1, 1, 1)
    )
    root_tiles = mul(colour_image, board)
    walls = mul(sl(patches, len(colours), len(colours) + 1), board)

    # Handle cores and frame corners.
    w = np.zeros((len(colours), 16, 7, 7), np.float32)
    for k, colour in enumerate(colours):
        frame_colour = 5 if colour == 11 else colour
        for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            w[k, colour, dy + 3, dx + 3] += 1
        for dy, dx in [(-2, -2), (-2, 3), (3, -2), (3, 3)]:
            w[k, frame_colour, dy + 3, dx + 3] += 1
    handles = node("relu", [
        conv("x0", w, np.full(len(colours), -7, np.float32))
    ])
    all_handles = channel_sum(handles)
    handle_colours = conv(
        handles, np.asarray(colours, np.float32).reshape(1, -1, 1, 1)
    )
    w = np.zeros((1, 16, 7, 7), np.float32)
    for dy, dx in corners:
        w[0, 0, dy + 3, dx + 3] = 1
    selected_ring = node("relu", [conv("x0", w, [-3])])
    anchor = mul(mul(all_handles, selected_ring), main)
    selected_colour = node("gmax", [mul(handle_colours, anchor)])

    # Rod openings identify orientation and selection.
    w = np.zeros((4, 16, 7, 7), np.float32)
    specifications = [
        [(0, 3, 1), (1, 3, 2)],
        [(0, 3, 2), (1, 3, 3)],
        [(3, 0, 2), (3, 1, 1)],
        [(3, 0, 3), (3, 1, 2)],
    ]
    for k, spec in enumerate(specifications):
        for dy, dx, colour in spec:
            w[k, colour, dy + 3, dx + 3] = 1
    notches = node("relu", [conv("x0", w, [-1] * 4)])
    hs = channel_sum(sl(notches, 0, 2))
    vs = channel_sum(sl(notches, 2, 4))
    horizontal = node("gmax", [mul(anchor, hs)])
    vertical = vnot(horizontal)

    hr = mul(mul(node("rowmax", [anchor]), hs), board)
    vr = mul(mul(node("colmax", [anchor]), vs), board)
    rod = add(anchor, add(mul(hr, horizontal), mul(vr, vertical)))
    other_h = node("relu", [sub(mul(hs, board), mul(rod, horizontal))])
    other_v = node("relu", [sub(mul(vs, board), mul(rod, vertical))])
    cross_rods = add(mul(other_v, horizontal), mul(other_h, vertical))

    axis = add(mul(col, horizontal), mul(row, vertical))
    transverse = add(mul(row, horizontal), mul(col, vertical))
    axis_bands = [
        add(mul(cb[i], horizontal), mul(rb[i], vertical))
        for i in range(7)
    ]
    dimensions = dense(
        bounds,
        np.asarray([
            [-1 / 6, 0, 1 / 6, 0],
            [0, -1 / 6, 0, 1 / 6]
        ], np.float32),
        [4 / 6, 4 / 6]
    )
    dimension = vadd(
        vmul(sl(dimensions, 0, 1), horizontal),
        vmul(sl(dimensions, 1, 2), vertical)
    )
    cross_end = add(
        mul(node("colmax", [mul(other_v, row)]), horizontal),
        mul(node("rowmax", [mul(other_h, col)]), vertical)
    )
    exit_distance = mul(
        node("relu", [sa(sub(cross_end, transverse), 1 / 6, 1)]),
        cross_rods
    )
    transverse_lower = vadd(
        vmul(va(oy, 1, 2), horizontal),
        vmul(va(ox, 1, 2), vertical)
    )
    transverse_upper = vadd(
        vmul(my, horizontal), vmul(mx, vertical)
    )

    # Read seven interface positions, stopping after the first empty one.
    ui = clip(sa(row, 100, -5350))
    ui_anchor = mul(
        mul(all_handles, equal(handle_colours, selected_colour)), ui
    )
    zero_goals = constant(np.zeros(7, np.float32))
    goal_body = []
    goal_before = macs[0]
    destination[0] = goal_body
    shapes["gp"], shapes["gg"], shapes["gc"] = (
        ("s", 1), ("v", 7), ("v", 1)
    )
    gp_next = shift("gp", 0, 1)
    detected_goal = node("gmax", [mul(gp_next, colour_image)])
    gc_next = vmul("gc", vclip(detected_goal))
    detected_goal = vmul(detected_goal, gc_next)
    w = np.zeros((7, 7), np.float32)
    for i in range(6):
        w[i, i + 1] = 1
    shifted_goals = dense("gg", w)
    w = np.zeros((7, 1), np.float32)
    w[6, 0] = 1
    gg_next = vadd(shifted_goals, dense(detected_goal, w))
    goal_cost = macs[0] - goal_before
    destination[0] = graph
    graph.append({
        "id": "read", "op": "loop", "T": 7,
        "state": [
            {"id": "gp", "init": ui_anchor},
            {"id": "gg", "init": zero_goals},
            {"id": "gc", "init": vone}
        ],
        "body": goal_body,
        "next": {"gp": gp_next, "gg": gg_next, "gc": gc_next}
    })
    shapes["read.gg"] = ("v", 7)
    goals = "read.gg"
    present = vclip(goals)
    goal_length = vector_sum(present)

    clickable_w = np.ones((1, len(colours), 1, 1), np.float32)
    clickable_w[0, colours.index(11), 0, 0] = 0
    inactive_handles = mul(
        node("relu", [sub(conv(handles, clickable_w), anchor)]), main
    )
    click_exists = node("gmax", [inactive_handles])

    def repeated_channels(x, n):
        c = shapes[x][1]
        w = np.zeros((c * n, c, 1, 1), np.float32)
        for i in range(c):
            w[i * n:(i + 1) * n, i, 0, 0] = 1
        return conv(x, w)

    def repeat_goals(x, c):
        return dense(x, np.tile(np.eye(7, dtype=np.float32), (c, 1)))

    def reduce_seven(x, c, spatial=False):
        w = np.zeros((c, c * 7), np.float32)
        for i in range(c):
            w[i, i * 7:(i + 1) * 7] = 1
        return conv(x, w[:, :, None, None]) if spatial else dense(x, w)

    def statistics(tiles, current_rod):
        c = shapes[tiles][1]
        occupied = node("min2", [tiles, one])
        held = mul(occupied, current_rod)
        length = va(node("gmean", [current_rod]), 4096)
        held_count = va(node("gmean", [held]), 4096)

        packed_bands = cat(axis_bands * c)
        packed_held = repeated_channels(held, 7)
        band_counts = node("gmax", [mul(packed_held, packed_bands)])
        w = np.zeros((c * 7, c * 7), np.float32)
        for candidate in range(c):
            for i in range(7):
                for j in range(i):
                    w[candidate * 7 + i, candidate * 7 + j] = 1
        before = dense(band_counts, w)
        ranks = reduce_seven(mul(packed_bands, before), c, True)

        rank_offsets = -np.tile(np.arange(7, dtype=np.float32), c)
        rank_test = node("relu", [
            sa(
                node("abs", [
                    conv(
                        repeated_channels(ranks, 7),
                        np.eye(c * 7, dtype=np.float32)[:, :, None, None],
                        rank_offsets
                    )
                ]), -1, 1
            )
        ])
        colour_test = equal(
            repeated_channels(tiles, 7), repeat_goals(goals, c)
        )
        matched = mul(mul(packed_held, colour_test), rank_test)
        correct = node("gmax", [matched])

        prefix = constant(np.ones(c, np.float32))
        prefix_parts = []
        for i in range(7):
            w = np.zeros((c, c * 7), np.float32)
            for candidate in range(c):
                w[candidate, candidate * 7 + i] = 1
            prefix = vmul(prefix, dense(correct, w))
            prefix_parts.append(prefix)

        prefix_vector = cat(prefix_parts, vector=True)
        if c > 1:
            w = np.zeros((c * 7, c * 7), np.float32)
            for candidate in range(c):
                for i in range(7):
                    w[candidate * 7 + i, i * c + candidate] = 1
            prefix_vector = dense(prefix_vector, w)
        prefix_count = reduce_seven(prefix_vector, c)
        matched_prefix = reduce_seven(
            mul(matched, prefix_vector), c, True
        )
        return (
            occupied, length, held_count, prefix_vector,
            prefix_count, matched_prefix, colour_test
        )

    def phase_update(length, prefix_count, matched_prefix,
                     previous_prefix, previous_phase, extra_capture=None):
        overlap = va(
            node("gmean", [mul(matched_prefix, cross_rods)]), 4096
        )
        compact = vmul(
            vclip(va(vadd(length, va(prefix_count, -1)), -1, 2)),
            node("vrelu", [va(overlap, -1, 1)])
        )
        captured = vclip(vadd(prefix_count, va(previous_prefix, -1)))
        if extra_capture is not None:
            captured = vmax(captured, extra_capture)
        return vmul(vmax(previous_phase, captured), vnot(compact))

    same_handle = vequal(selected_colour, sl("mem", 2, 3))
    old_prefix = vmul(sl("mem", 0, 1), same_handle)
    old_phase = vmul(sl("mem", 1, 2), same_handle)
    old_held = vmul(sl("mem", 3, 4), same_handle)
    root_stats = statistics(root_tiles, rod)
    root_prefix = root_stats[4]
    acquired = vclip(vadd(root_stats[2], va(old_held, -1)))
    root_phase = phase_update(
        root_stats[1], root_prefix, root_stats[5],
        old_prefix, old_phase, acquired
    )

    # A 7-by-8 table records recently visited lane/length configurations.
    def configuration(cross_position, length):
        c = shapes[length][1]
        lane_index = va(
            vadd(cross_position, va(transverse_lower, -1)), 1 / 6
        )
        lane_w = np.zeros((56 * c, c), np.float32)
        length_w = np.zeros((56 * c, c), np.float32)
        lane_b = np.zeros(56 * c, np.float32)
        length_b = np.zeros(56 * c, np.float32)
        for candidate in range(c):
            for lane in range(7):
                for size in range(8):
                    i = candidate * 56 + lane * 8 + size
                    lane_w[i, candidate] = 1
                    length_w[i, candidate] = 1
                    lane_b[i] = -lane
                    length_b[i] = -(size + 1)
        distance = vadd(
            vabs(dense(lane_index, lane_w, lane_b)),
            vabs(dense(length, length_w, length_b))
        )
        return node("vrelu", [va(distance, -1, 1)])

    root_cross = node("gmax", [mul(anchor, transverse)])
    root_configuration = configuration(root_cross, root_stats[1])
    unchanged_prefix = vequal(root_prefix, sl("mem", 0, 1))
    visits = va(
        vmul(
            sl("mem", 4, 60),
            vmul(same_handle, unchanged_prefix)
        ), 0.98
    )
    visits_next = vadd(visits, root_configuration)
    va(cat([
        root_prefix, root_phase, selected_colour, root_stats[2],
        visits_next
    ], vector=True), name="mem_next")

    solved = vclip(
        va(vadd(goal_length, va(root_prefix, -1)), -1, 1)
    )
    compact_for_switch = vclip(
        va(vadd(root_stats[1], va(root_prefix, -1)), -1, 2)
    )
    should_click = vmul(
        vmul(solved, compact_for_switch), click_exists
    )

    # Static movement restrictions for four packed candidates.
    board4, walls4 = repeat_spatial(board, 4), repeat_spatial(walls, 4)
    blockers = cat([other_h, other_h, other_v, other_v])
    forbidden = node("max2", [
        node("max2", [
            sa(shift_four(board4, True), -1, 1),
            shift_four(walls4, True)
        ]),
        node("max2", [blockers, shift_four(blockers, True)])
    ])
    perpendicular = cat(
        [horizontal, horizontal, vertical, vertical], vector=True
    )
    extension = cat([vzero, vertical, vzero, horizontal], vector=True)
    retraction = vnot(vadd(perpendicular, extension))

    w = np.zeros((4, 8), np.float32)
    for i, opposite in enumerate([2, 1, 4, 3]):
        w[i, opposite] = 1
    inverse_action = dense("aprev", w)

    initial_first = constant([0, 0, 0, 0])
    initial_depth = constant([0])
    initial_best = constant([-1000000])
    state_specs = [
        ("pt", root_tiles), ("pr", rod), ("pa", anchor),
        ("pp", root_prefix), ("ph", root_phase), ("pv", vone),
        ("pf", initial_first), ("pd", initial_depth),
        ("pbs", initial_best), ("pba", initial_first)
    ]
    for sid, source in state_specs:
        shapes[sid] = shapes[source]

    body = []
    destination[0] = body
    before_body = macs[0]

    tiles4 = repeat_spatial("pt", 4)
    rod4 = repeat_spatial("pr", 4)
    anchor4 = repeat_spatial("pa", 4)
    previous_prefix = repeat_vector("pp", 4)
    previous_phase = repeat_vector("ph", 4)
    previous_valid = repeat_vector("pv", 4)

    moving_rod = sub(rod4, mul(anchor4, retraction))
    shifted_rod = shift_four(moving_rod)
    new_anchor = add(
        mul(shift_four(anchor4), perpendicular),
        mul(anchor4, vnot(perpendicular))
    )
    allowed = node("max2", [board4, new_anchor])
    spill = node("gmax", [mul(shifted_rod, sa(allowed, -1, 1))])
    rod_wall = vmul(
        node("gmax", [mul(shifted_rod, walls4)]),
        vnot(retraction)
    )
    new_rod = mul(node("max2", [shifted_rod, new_anchor]), allowed)

    occupied4 = node("min2", [tiles4, one])
    affected = mul(occupied4, node("max2", [rod4, shifted_rod]))
    for _ in range(6):
        affected = node("max2", [
            affected, mul(occupied4, shift_four(affected))
        ])
    blocked = mul(affected, forbidden)
    for _ in range(6):
        blocked = node("max2", [
            blocked, mul(affected, shift_four(blocked, True))
        ])
    movers = node("relu", [sub(affected, blocked)])
    moving_colours = mul(tiles4, movers)
    new_tiles = add(
        sub(tiles4, moving_colours), shift_four(moving_colours)
    )

    old_length = va(node("gmean", [rod4]), 4096)
    empty_retraction = vmul(
        node("vrelu", [va(old_length, -1, 2)]), retraction
    )
    blocked_lateral = vmul(node("gmax", [blocked]), perpendicular)
    new_cross = node("gmax", [mul(new_anchor, transverse)])
    outside_low = node("vrelu", [
        va(vadd(transverse_lower, va(new_cross, -1)), 1 / 6, -0.05)
    ])
    outside_high = node("vrelu", [
        va(vadd(new_cross, va(transverse_upper, -1)), 1 / 6, -0.05)
    ])
    invalid = sumv([
        spill, rod_wall, empty_retraction, blocked_lateral,
        outside_low, outside_high
    ])
    valid = vmul(
        node("vrelu", [va(invalid, -1, 1)]), previous_valid
    )

    st = statistics(new_tiles, new_rod)
    occupied, length, held_count, prefixes, prefix_count, matched, colour_eq = st
    phase = phase_update(
        length, prefix_count, matched, previous_prefix, previous_phase
    )

    # The first unmet sequence entry determines eligible next beads.
    w, b = np.zeros((28, 28), np.float32), np.zeros(28, np.float32)
    for candidate in range(4):
        b[candidate * 7] = 1
        for i in range(1, 7):
            w[candidate * 7 + i, candidate * 7 + i - 1] = 1
    predecessor = dense(prefixes, w, b)
    needed = vmul(
        vadd(predecessor, va(prefixes, -1)), repeat_goals(present, 4)
    )
    next_colour_map = reduce_seven(mul(colour_eq, needed), 4, True)
    eligible = node("relu", [sub(occupied, matched)])
    next_map = mul(next_colour_map, eligible)
    needs_target = reduce_seven(needed, 4)

    endpoint = node("gmax", [mul(new_rod, axis)])
    anchor_axis = node("gmax", [mul(new_anchor, axis)])
    anchor_cross = node("gmax", [mul(new_anchor, transverse)])
    lateral_distance = sa(
        node("abs", [sub(transverse, anchor_cross)]), 1 / 6
    )
    extension_distance = node("relu", [
        sa(sub(axis, endpoint), 1 / 6)
    ])
    collecting_cost = add(lateral_distance, extension_distance)

    obstructing_beads = mul(
        eligible, sa(next_colour_map, -1, 1)
    )
    lane_obstructions = add(
        mul(node("rowsum", [obstructing_beads]), horizontal),
        mul(node("colsum", [obstructing_beads]), vertical)
    )
    compact_extension = node("relu", [
        sub(sa(sub(axis, anchor_axis), 1 / 6), prefix_count)
    ])
    compact_free = vadd(dimension, va(prefix_count, -1))
    excess_length = node("vrelu", [
        va(vadd(length, va(prefix_count, -1)), 1, -1)
    ])
    compacting_cost = add(
        add(lateral_distance, excess_length),
        add(
            mul(sa(cross_rods, -1, 1), compact_free),
            mul(cross_rods, compact_extension)
        )
    )
    pickup_cost_map = add(
        add(
            add(
                mul(collecting_cost, vnot(phase)),
                mul(compacting_cost, phase)
            ),
            exit_distance
        ),
        sa(lane_obstructions, 5)
    )
    pickup_value = node("gmax", [
        mul(sa(pickup_cost_map, -1, 100), next_map)
    ])
    pickup_cost = vadd(va(needs_target, 100), va(pickup_value, -1))
    exit_cost = va(node("gmean", [mul(matched, exit_distance)]), 4096)
    unwanted = vadd(held_count, va(prefix_count, -1))

    # Lookup visit counts through one-hot arithmetic, not external code.
    candidate_configurations = configuration(new_cross, length)
    w = np.tile(np.eye(56, dtype=np.float32), (4, 1))
    packed_visits = dense(visits_next, w)
    weighted_visits = vmul(candidate_configurations, packed_visits)
    w = np.zeros((4, 224), np.float32)
    for i in range(4):
        w[i, i * 56:(i + 1) * 56] = 1
    candidate_visits = dense(weighted_visits, w)
    novelty_penalty = vmul(candidate_visits, vnot(vclip(prefix_count)))

    first_iteration = node("vrelu", [va("pd", -1, 1)])
    score = sumv([
        va(prefix_count, 100),
        va(unwanted, -1),
        va(pickup_cost, -2),
        va(length, -0.5),
        va(exit_cost, -4),
        va(novelty_penalty, -1.5),
        va(vnot(valid), -100000),
        va(vmul(inverse_action, first_iteration), -0.3)
    ])
    score = vadd(score, constant([0, -0.01, -0.02, -0.03]))

    # Finite four-way maximum-selection circuit.
    w = np.zeros((16, 4), np.float32)
    for i in range(4):
        for j in range(4):
            w[i * 4 + j, j] += 1
            w[i * 4 + j, i] -= 1
    differences = node("vrelu", [dense(score, w)])
    w = np.zeros((4, 16), np.float32)
    for i in range(4):
        w[i, i * 4:(i + 1) * 4] = 1
    selection = node("vrelu", [va(dense(differences, w), -1000, 1)])

    def select_s(x):
        return channel_sum(mul(x, selection))

    def select_v(x):
        return vector_sum(vmul(x, selection))

    next_tiles = select_s(new_tiles)
    next_rod = select_s(new_rod)
    next_anchor = select_s(new_anchor)
    next_prefix = select_v(prefix_count)
    next_phase = select_v(phase)
    next_valid = select_v(valid)

    first_action = vadd(
        vmul(selection, first_iteration),
        vmul("pf", vnot(first_iteration))
    )
    chosen_score = vadd(select_v(score), va("pd", -0.15))
    improvement = vclip(
        va(vadd(chosen_score, va("pbs", -1)), 1000)
    )
    best_score = vadd(
        vmul(chosen_score, improvement),
        vmul("pbs", vnot(improvement))
    )
    best_action = vadd(
        vmul(first_action, improvement),
        vmul("pba", vnot(improvement))
    )
    next_depth = va("pd", 1, 1)
    body_cost = macs[0] - before_body

    destination[0] = graph
    loop = {
        "id": "plan", "op": "loop", "T": 1,
        "state": [{"id": sid, "init": source} for sid, source in state_specs],
        "body": body,
        "next": {
            "pt": next_tiles, "pr": next_rod, "pa": next_anchor,
            "pp": next_prefix, "ph": next_phase, "pv": next_valid,
            "pf": first_action, "pd": next_depth,
            "pbs": best_score, "pba": best_action
        }
    }
    graph.append(loop)
    shapes["plan.pba"] = ("v", 4)

    w = np.zeros((8, 4), np.float32)
    for i in range(4):
        w[i + 1, i] = 10
    action_logits = dense(
        "plan.pba", w, [-1000, 0, 0, 0, 0, -1000, -20, -500]
    )
    w = np.zeros((8, 1), np.float32)
    w[6, 0] = 50
    node("vadd", [action_logits, dense(should_click, w)], name="logits")
    sa(inactive_handles, name="click")

    # Fixed rollout depth chosen only by static arithmetic budget accounting.
    fixed_cost = macs[0] - body_cost + 6 * goal_cost
    max_mac_depth = int((1450000000 - fixed_cost) // body_cost)
    fixed_evaluations = len(graph) + 7 * len(goal_body)
    max_eval_depth = int((9800 - fixed_evaluations) // len(body))
    loop["T"] = max(1, min(10, max_mac_depth, max_eval_depth))

    return {"graph": graph, "memory_size": 60, "params": params}
