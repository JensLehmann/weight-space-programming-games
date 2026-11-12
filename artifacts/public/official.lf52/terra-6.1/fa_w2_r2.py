import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]
    affine_cache = {}
    shift_cache = {}

    def fresh(prefix="n"):
        serial[0] += 1
        return "%s%d" % (prefix, serial[0])

    def param(value):
        name = fresh("p")
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def op(kind, inputs, out=None, **fields):
        name = out or fresh()
        nd = {"id": name, "op": kind, "in": list(inputs)}
        nd.update(fields)
        graph.append(nd)
        return name

    def conv(x, w, b=None, out=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return op("conv", [x], out=out, **fields)

    def dense(x, w, b=None, out=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return op("dense", [x], out=out, **fields)

    def sl(x, a, b):
        return op("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return op("vslice", [x], start=a, stop=b)

    def add(a, b):
        return op("add", [a, b])

    def sub(a, b):
        return op("sub", [a, b])

    def mul(a, b):
        return op("mul", [a, b])

    def mx(a, b):
        return op("max2", [a, b])

    def mn(a, b):
        return op("min2", [a, b])

    def vadd(a, b):
        return op("vadd", [a, b])

    def vmul(a, b):
        return op("vmul", [a, b])

    def maximum(xs):
        y = xs[0]
        for x in xs[1:]:
            y = mx(y, x)
        return y

    def total(xs):
        y = xs[0]
        for x in xs[1:]:
            y = add(y, x)
        return y

    def affine(x, weight, bias=0.0):
        key = (float(weight), float(bias))
        if key not in affine_cache:
            affine_cache[key] = (
                param(np.array([[[[weight]]]], dtype=np.float32)),
                param(np.array([bias], dtype=np.float32)),
            )
        w, b = affine_cache[key]
        return conv(x, w, b)

    def clip01(x):
        return mn(op("relu", [x]), one)

    def vone_minus(x):
        return dense(
            x,
            param(np.array([[-1.0]], dtype=np.float32)),
            param(np.array([1.0], dtype=np.float32)),
        )

    def shift_weight(channels, dy, dx):
        key = (channels, dy, dx)
        if key not in shift_cache:
            w = np.zeros((channels, channels, 7, 7), dtype=np.float32)
            for c in range(channels):
                w[c, c, 3 + 3 * dy, 3 + 3 * dx] = 1.0
            shift_cache[key] = param(w)
        return shift_cache[key]

    def shift6(x, direction, channels=1):
        dy, dx = direction
        w = shift_weight(channels, dy, dx)
        return conv(conv(x, w), w)

    def shift12(x, direction, channels=1):
        return shift6(shift6(x, direction, channels), direction, channels)

    def reverse(direction):
        return (-direction[0], -direction[1])

    def detect(colour, pattern):
        w = np.zeros((1, 16, 7, 7), dtype=np.float32)
        for y, x in pattern:
            w[0, colour, 3 + y - 2, 3 + x - 2] = 1.0
        y = conv(
            "x0",
            param(w),
            param(np.array([-len(pattern) + 0.5], dtype=np.float32)),
        )
        return clip01(affine(op("relu", [y]), 2.0))

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    white_interior = sl("x0", 1, 2)
    black = sl("x0", 5, 6)
    orange = sl("x0", 12, 13)

    def coordinate_map(r, c):
        dr = op("abs", [sub(row, r)])
        dc = op("abs", [sub(col, c)])
        mr = clip01(affine(dr, -63.0, 1.0))
        mc = clip01(affine(dc, -63.0, 1.0))
        return mul(mr, mc)

    dot_pattern = [
        (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 2), (4, 3),
    ]
    bottle_pattern = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 1), (1, 4),
        (2, 1), (2, 4),
        (3, 1), (3, 2), (3, 3), (3, 4),
    ]
    ring_pattern = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 1), (1, 4), (1, 5),
        (2, 0), (2, 5),
        (3, 0), (3, 5),
        (4, 0), (4, 1), (4, 4), (4, 5),
        (5, 1), (5, 2), (5, 3), (5, 4),
    ]
    platform_pattern = [(0, 0), (0, 5), (5, 0), (5, 5)]

    green = detect(14, dot_pattern)
    red = detect(8, dot_pattern)
    blue = detect(9, dot_pattern)
    bottle = detect(15, bottle_pattern)
    selected_ring = detect(3, ring_pattern)
    platform = detect(11, platform_pattern)

    pieces = clip01(total([green, red, blue]))
    occupied = clip01(add(pieces, bottle))
    empty = clip01(add(white_interior, orange))

    selected_flag = op("gmax", [selected_ring])
    not_selected = vone_minus(selected_flag)
    selected_gate = clip01(add(selected_ring, not_selected))
    eligible_source = mul(pieces, selected_gate)

    wt = np.zeros((1, 1, 3, 3), dtype=np.float32)
    wt[0, 0, 1, 1] = 1.0
    wt[0, 0, 1, 2] = 1.0
    wt[0, 0, 2, 1] = 1.0
    wt[0, 0, 2, 2] = 1.0
    track_black = conv(
        black, param(wt), param(np.array([-3.5], dtype=np.float32))
    )
    track_black = clip01(affine(op("relu", [track_black]), 2.0))
    track = clip01(add(track_black, platform))

    piece_on_platform = mul(pieces, platform)
    bottle_on_platform = mul(bottle, platform)
    piece_aboard = op("gmax", [piece_on_platform])
    bottle_aboard = op("gmax", [bottle_on_platform])
    green_aboard = op("gmax", [mul(green, platform)])
    red_aboard = op("gmax", [mul(red, platform)])
    no_piece_aboard = vone_minus(piece_aboard)
    platform_visible = op("gmax", [platform])
    platform_missing = vone_minus(platform_visible)

    previous_row = vsl("mem", 4, 5)
    previous_col = vsl("mem", 5, 6)
    previous_source = coordinate_map(previous_row, previous_col)

    failed_sources = []
    failed_targets = []
    for k in range(3):
        base = 12 + 4 * k
        failed_sources.append(
            coordinate_map(
                vsl("mem", base, base + 1),
                vsl("mem", base + 1, base + 2),
            )
        )
        failed_targets.append(
            coordinate_map(
                vsl("mem", base + 2, base + 3),
                vsl("mem", base + 3, base + 4),
            )
        )

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    features = op("concat", [occupied, green, red])
    stationary_occupied = clip01(sub(occupied, piece_on_platform))

    mids = []
    ends = []
    track_ends = []
    old_ends = []
    followup_parts = []
    exit_parts = []

    for direction in directions:
        mid = shift6(features, direction, 3)
        mid_occupied = sl(mid, 0, 1)
        mid_green = sl(mid, 1, 2)
        mid_red = sl(mid, 2, 3)
        end = shift12(empty, direction)
        track_end = shift12(track, direction)
        old_end = shift12(previous_source, direction)

        mids.append((mid_occupied, mid_green, mid_red))
        ends.append(end)
        track_ends.append(track_end)
        old_ends.append(old_end)
        followup_parts.append(mul(mid_occupied, end))

        stationary_mid = shift6(stationary_occupied, direction)
        same_colour = clip01(
            add(mul(mid_green, green_aboard), mul(mid_red, red_aboard))
        )
        forbidden_return = mul(old_end, clip01(sub(one, same_colour)))
        usable_end = mul(end, clip01(sub(one, forbidden_return)))
        exit_parts.append(mul(stationary_mid, usable_end))

    followups = total(followup_parts)
    exit_followups = total(exit_parts)

    wtie = np.zeros((1, 3, 1, 1), dtype=np.float32)
    wtie[0, 0, 0, 0] = -0.05
    wtie[0, 1, 0, 0] = -0.00078
    coordinate_tie = conv("coords", param(wtie))

    oscillator = vsl("mem", 0, 4)
    direction_preferences = dense(
        oscillator,
        param(np.eye(4, dtype=np.float32) * 0.025),
        param(np.array([-0.003, -0.002, -0.001, 0.0], dtype=np.float32)),
    )

    scores = []
    loading_goals = []
    bottle_goals = []

    for i, direction in enumerate(directions):
        mid_occupied, mid_green, mid_red = mids[i]
        capture = clip01(add(mul(green, mid_green), mul(red, mid_red)))
        noncapture = clip01(sub(one, capture))

        return_to_old = mul(old_ends[i], noncapture)
        landing_platform = shift12(platform, direction)
        return_to_old = mul(
            return_to_old, clip01(sub(one, landing_platform))
        )

        failed_here = []
        for k in range(3):
            target_here = shift12(failed_targets[k], direction)
            failed_here.append(mul(failed_sources[k], target_here))
        failed = clip01(maximum(failed_here))

        legal = mul(eligible_source, mul(mid_occupied, ends[i]))
        legal = mul(legal, clip01(sub(one, return_to_old)))
        legal = mul(legal, clip01(sub(one, failed)))

        future_at_end = shift12(followups, direction)
        value = affine(capture, 3.0, 10.0)
        value = add(value, affine(future_at_end, 1.1))
        value = add(value, coordinate_tie)
        value = add(value, vsl(direction_preferences, i, i + 1))
        scores.append(mul(legal, value))

        load_source = mul(pieces, mul(mid_occupied, track_ends[i]))
        loading_goals.append(shift12(load_source, reverse(direction)))

        bottle_source = mul(pieces, ends[i])
        bottle_midpoint = shift6(bottle_source, reverse(direction))
        bottle_goals.append(mul(bottle_midpoint, track))

    best_source = maximum(scores)
    best_value = op("gmax", [best_source])
    has_jump = op("gmax", [clip01(affine(best_source, 0.2))])

    landing_scores = [
        shift12(scores[i], reverse(directions[i]))
        for i in range(4)
    ]
    best_landing = maximum(landing_scores)
    normal_click = add(
        mul(best_source, not_selected),
        mul(best_landing, selected_flag),
    )

    wi = np.zeros((1, 16, 7, 7), dtype=np.float32)
    for yy in (2, 3, 4):
        for xx in (1, 2):
            wi[0, 15, 3 + yy - 4, 3 + xx - 4] = 1.0
    for yy, xx in ((3, 3), (4, 3)):
        wi[0, 5, 3 + yy - 4, 3 + xx - 4] = 1.0
    for yy, xx in ((4, 4), (4, 5)):
        wi[0, 0, 3 + yy - 4, 3 + xx - 4] = 1.0
    icon = conv(
        "x0", param(wi), param(np.array([-9.5], dtype=np.float32))
    )
    icon = clip01(affine(op("relu", [icon]), 2.0))
    lower = clip01(affine(row, 63.0, -48.0))
    left = clip01(affine(col, -63.0, 16.0))
    icon = mul(icon, mul(lower, left))
    icon_flag = op("gmax", [icon])

    op("relu", [normal_click], out="click")

    loading_goal = clip01(maximum(loading_goals))
    bottle_goal = clip01(maximum(bottle_goals))
    exit_goal = mul(track, clip01(exit_followups))

    has_bottle_goal = op("gmax", [bottle_goal])
    use_bottle = vmul(no_piece_aboard, vmul(bottle_aboard, has_bottle_goal))
    use_empty = vmul(no_piece_aboard, vone_minus(use_bottle))

    goal = clip01(total([
        mul(exit_goal, piece_aboard),
        mul(bottle_goal, use_bottle),
        mul(loading_goal, use_empty),
    ]))
    carrier = clip01(total([
        mul(piece_on_platform, piece_aboard),
        mul(bottle_on_platform, use_bottle),
        mul(platform, use_empty),
    ]))

    wf = np.zeros((4, 1, 7, 7), dtype=np.float32)
    ws = np.zeros((4, 4, 7, 7), dtype=np.float32)
    for i, (dy, dx) in enumerate(directions):
        wf[i, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
        ws[i, i, 3 + 3 * dy, 3 + 3 * dx] = 1.0
    pf = param(wf)
    ps = param(ws)
    pd = param(np.array([[[[0.94]]]], dtype=np.float32))

    loop_name = fresh("loop")
    state_name = fresh("state")
    body = []

    def bop(kind, inputs, **fields):
        name = fresh("b")
        nd = {"id": name, "op": kind, "in": list(inputs)}
        nd.update(fields)
        body.append(nd)
        return name

    q1 = bop("conv", [state_name], w=pf)
    q2 = bop("conv", [q1], w=ps)
    qs = [bop("slice", [q2], start=i, stop=i + 1) for i in range(4)]
    qmax = bop("max2", [qs[0], qs[1]])
    qmax = bop("max2", [qmax, qs[2]])
    qmax = bop("max2", [qmax, qs[3]])
    qdiscount = bop("conv", [qmax], w=pd)
    qtrack = bop("mul", [qdiscount, track])
    qnext = bop("max2", [qtrack, goal])

    graph.append({
        "id": loop_name,
        "op": "loop",
        "T": 40,
        "state": [{"id": state_name, "zeros": [1, 64, 64]}],
        "body": body,
        "next": {state_name: qnext},
    })
    wave = loop_name + "." + state_name

    recent_maps = []
    for k in range(4):
        base = 24 + 2 * k
        recent_maps.append(
            coordinate_map(
                vsl("mem", base, base + 1),
                vsl("mem", base + 1, base + 2),
            )
        )
    recent = maximum(recent_maps)

    valid_dirs = []
    wave_dirs = []
    revisited_dirs = []
    for direction in directions:
        nearby_track = shift6(track, direction)
        nearby_platform = shift6(platform, direction)
        available_track = mul(
            nearby_track, clip01(sub(one, nearby_platform))
        )
        nearby_wave = shift6(wave, direction)
        nearby_recent = shift6(recent, direction)
        valid = op("gmax", [mul(carrier, available_track)])
        potential = op("gmax", [mul(carrier, nearby_wave)])
        valid_dirs.append(valid)
        wave_dirs.append(vmul(valid, potential))
        revisited_dirs.append(op("gmax", [mul(carrier, nearby_recent)]))

    edge_maps = [
        clip01(affine(row, -63.0, 6.0)),
        clip01(affine(row, 63.0, -57.0)),
        clip01(affine(col, -63.0, 6.0)),
        clip01(affine(col, 63.0, -57.0)),
    ]
    edge_flags = [
        op("gmax", [mul(carrier, edge_maps[i])]) for i in range(4)
    ]
    previous_dirs = [vsl("aprev", i + 1, i + 2) for i in range(4)]
    edge_continuations = [
        vmul(edge_flags[i], previous_dirs[i]) for i in range(4)
    ]

    platform_row = op("gmean", [mul(platform, row)])
    platform_col = op("gmean", [mul(platform, col)])
    position = op("vconcat", [platform_row, platform_col])
    position = dense(position, param(np.eye(2, dtype=np.float32) * 4096.0))
    old_position = vsl("mem", 8, 10)

    neg_identity = param(-np.eye(2, dtype=np.float32))
    delta = vadd(position, dense(old_position, neg_identity))
    positive = op("vrelu", [delta])
    negative = op("vrelu", [dense(delta, neg_identity)])
    magnitude = vadd(positive, negative)
    unchanged = dense(
        magnitude,
        param(np.array([[-100.0, -100.0]], dtype=np.float32)),
        param(np.array([1.0], dtype=np.float32)),
    )
    unchanged = op("vrelu", [unchanged])
    blocked_dirs = [vmul(unchanged, previous_dirs[i]) for i in range(4)]

    moved_action = dense(
        "aprev",
        param(np.array([[0, 1, 1, 1, 1, 0, 0, 0]], dtype=np.float32)),
    )
    lost_platform = vmul(platform_missing, moved_action)
    offscreen_blocks = [vsl("mem", 34 + i, 35 + i) for i in range(4)]
    selected_without_jump = vmul(selected_flag, vone_minus(has_jump))

    action_features = op(
        "vconcat",
        valid_dirs
        + wave_dirs
        + edge_continuations
        + blocked_dirs
        + previous_dirs
        + revisited_dirs
        + offscreen_blocks
        + [best_value, icon_flag, selected_without_jump, lost_platform],
    )
    wa = np.zeros((8, 32), dtype=np.float32)
    ba = np.full(8, -10.0, dtype=np.float32)
    opposites = [1, 0, 3, 2]

    for i in range(4):
        wa[i + 1, i] = 2.0
        wa[i + 1, 4 + i] = 4.0
        wa[i + 1, 8 + i] = 1.0
        wa[i + 1, 12 + i] = -3.0
        wa[i + 1, 16 + i] = 0.03
        wa[i + 1, 16 + opposites[i]] = -0.15
        wa[i + 1, 20 + i] = -0.3
        wa[i + 1, 24 + i] = -5.0
        ba[i + 1] = -1.0

    ba[4] += 0.08
    wa[6, 28] = 1.0
    ba[6] = -1.0
    wa[7, 29] = 100.0
    wa[7, 30] = 20.0
    wa[7, 31] = 15.0
    ba[7] = 0.0
    dense(action_features, param(wa), param(ba), out="logits")

    def winner_map(score, maximum_value):
        difference = sub(score, maximum_value)
        return clip01(
            affine(
                op("relu", [affine(difference, 1.0, 0.000004)]),
                1000000.0,
            )
        )

    source_winner = winner_map(best_source, best_value)
    landing_max = op("gmax", [best_landing])
    landing_winner = winner_map(best_landing, landing_max)
    winner_row = op("gmax", [mul(source_winner, row)])
    winner_col = op("gmax", [mul(source_winner, col)])
    landing_row = op("gmax", [mul(landing_winner, row)])
    landing_col = op("gmax", [mul(landing_winner, col)])

    wo = np.zeros((4, 4), dtype=np.float32)
    wo[0, :] = -1.0
    wo[1, 0] = 1.0
    wo[2, 1] = 1.0
    wo[3, 2] = 1.0
    next_oscillator = dense(
        oscillator,
        param(wo),
        param(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
    )

    current_row = vsl("mem", 6, 7)
    current_col = vsl("mem", 7, 8)
    old_target_row = vsl("mem", 10, 11)
    old_target_col = vsl("mem", 11, 12)

    save_current = vmul(has_jump, not_selected)
    keep_current = vone_minus(save_current)
    commit_jump = vmul(has_jump, selected_flag)
    keep_jump = vone_minus(commit_jump)

    next_current_row = vadd(
        vmul(save_current, winner_row), vmul(keep_current, current_row)
    )
    next_current_col = vadd(
        vmul(save_current, winner_col), vmul(keep_current, current_col)
    )
    next_previous_row = vadd(
        vmul(commit_jump, current_row), vmul(keep_jump, previous_row)
    )
    next_previous_col = vadd(
        vmul(commit_jump, current_col), vmul(keep_jump, previous_col)
    )
    next_target_row = vadd(
        vmul(commit_jump, landing_row), vmul(keep_jump, old_target_row)
    )
    next_target_col = vadd(
        vmul(commit_jump, landing_col), vmul(keep_jump, old_target_col)
    )

    previous_click = vsl("aprev", 6, 7)
    save_failure = vmul(icon_flag, previous_click)
    keep_failures = vone_minus(save_failure)
    attempted_pair = op(
        "vconcat", [previous_row, previous_col, old_target_row, old_target_col]
    )
    failure_blocks = []
    for k in range(3):
        old_block = vsl("mem", 12 + 4 * k, 16 + 4 * k)
        incoming = attempted_pair if k == 0 else vsl(
            "mem", 12 + 4 * (k - 1), 16 + 4 * (k - 1)
        )
        failure_blocks.append(
            vadd(vmul(save_failure, incoming), vmul(keep_failures, old_block))
        )

    carrier_row = op("gmax", [mul(carrier, row)])
    carrier_col = op("gmax", [mul(carrier, col)])
    carrier_position = op("vconcat", [carrier_row, carrier_col])
    old_carrier_position = vsl("mem", 32, 34)
    keep_visits = vone_minus(moved_action)
    visit_blocks = []
    for k in range(4):
        old_block = vsl("mem", 24 + 2 * k, 26 + 2 * k)
        incoming = old_carrier_position if k == 0 else vsl(
            "mem", 24 + 2 * (k - 1), 26 + 2 * (k - 1)
        )
        visit_blocks.append(
            vadd(vmul(moved_action, incoming), vmul(keep_visits, old_block))
        )

    previous_dir_vector = vsl("aprev", 1, 5)
    old_offscreen = vsl("mem", 34, 38)
    visible_direction_action = vmul(platform_visible, moved_action)
    retain_offscreen = vmul(
        vone_minus(lost_platform), vone_minus(visible_direction_action)
    )
    next_offscreen = vadd(
        vmul(lost_platform, previous_dir_vector),
        vmul(retain_offscreen, old_offscreen),
    )

    op(
        "vconcat",
        [
            next_oscillator,
            next_previous_row,
            next_previous_col,
            next_current_row,
            next_current_col,
            position,
            next_target_row,
            next_target_col,
        ] + failure_blocks + visit_blocks + [carrier_position, next_offscreen],
        out="mem_next",
    )

    return {"graph": graph, "memory_size": 38, "params": params}
