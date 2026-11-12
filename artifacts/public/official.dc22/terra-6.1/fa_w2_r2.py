import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]
    pcounter = [0]

    def fresh(prefix="n"):
        i = counter[0]
        counter[0] += 1
        return "%s%d" % (prefix, i)

    def param(a):
        name = "p%d" % pcounter[0]
        pcounter[0] += 1
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs, out=None, dst=None, **fields):
        if dst is None:
            dst = graph
        if out is None:
            out = fresh()
        nd = {"id": out, "op": op, "in": list(inputs)}
        nd.update(fields)
        dst.append(nd)
        return out

    def conv(x, w, b=None, out=None, dst=None):
        kw = {"w": param(w)}
        if b is not None:
            kw["b"] = param(b)
        return node("conv", [x], out=out, dst=dst, **kw)

    def dense(x, w, b=None, out=None, dst=None):
        kw = {"w": param(w)}
        if b is not None:
            kw["b"] = param(b)
        return node("dense", [x], out=out, dst=dst, **kw)

    def sl(x, start, stop, dst=None):
        return node("slice", [x], dst=dst, start=start, stop=stop)

    def vsl(x, start, stop, dst=None):
        return node("vslice", [x], dst=dst, start=start, stop=stop)

    def op1(op, x, dst=None):
        return node(op, [x], dst=dst)

    def op2(op, x, y, dst=None):
        return node(op, [x, y], dst=dst)

    def max_many(xs, dst=None):
        z = xs[0]
        for x in xs[1:]:
            z = op2("max2", z, x, dst)
        return z

    def affine(x, scale, bias=0.0, dst=None):
        return conv(
            x,
            np.asarray([[[[scale]]]], np.float32),
            np.asarray([bias], np.float32),
            dst=dst,
        )

    def vaffine(x, scale, bias=0.0, dst=None):
        return dense(
            x,
            np.asarray([[scale]], np.float32),
            np.asarray([bias], np.float32),
            dst=dst,
        )

    def vadd(x, y):
        return node("vadd", [x, y])

    def vmul(x, y):
        return node("vmul", [x, y])

    def vsub(x, y):
        return vadd(x, vaffine(y, -1.0))

    def vclip(x):
        return vadd(
            op1("vrelu", x),
            vaffine(op1("vrelu", vaffine(x, 1.0, -1.0)), -1.0),
        )

    def vnot(x):
        return vaffine(x, -1.0, 1.0)

    def vabs(x):
        return vadd(op1("vrelu", x), op1("vrelu", vaffine(x, -1.0)))

    def veq(x, y):
        return vclip(vaffine(vabs(vsub(x, y)), -63.0, 1.0))

    def vgreater(x, y):
        return vclip(vaffine(vsub(x, y), 63.0))

    def clip01(x, dst=None):
        return op2("min2", op1("relu", x, dst), one, dst)

    def complement(x, dst=None):
        return op2("sub", one, x, dst)

    def read_shift(x, dy, dx, dst=None):
        z = x
        ry, rx = int(dy), int(dx)
        while ry or rx:
            sy = max(-3, min(3, ry))
            sx = max(-3, min(3, rx))
            w = np.zeros((1, 1, 7, 7), np.float32)
            w[0, 0, 3 + sy, 3 + sx] = 1.0
            z = conv(z, w, dst=dst)
            ry -= sy
            rx -= sx
        return z

    def translate(x, dy, dx, dst=None):
        return read_shift(x, -dy, -dx, dst)

    def anchor(mask):
        left = read_shift(mask, 0, -1)
        up = read_shift(mask, -1, 0)
        return op1("relu", op2("sub", op2("sub", mask, left), up))

    def count_2x2(x):
        w = np.zeros((1, 1, 3, 3), np.float32)
        w[0, 0, 1:3, 1:3] = 1.0
        return conv(x, w)

    one = sl("coords", 2, 3)
    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    colors = [sl("x0", i, i + 1) for i in range(16)]
    ui = clip01(affine(col, 63.0, -39.0))
    play = complement(ui)

    def button(color):
        p = colors[color]
        left = read_shift(p, 0, -5)
        right = read_shift(p, 0, 5)
        return op2("mul", ui, op2("mul", p, op2("mul", left, right)))

    button8 = button(8)
    button9 = button(9)
    button15 = button(15)
    button6 = button(6)
    has8 = op1("gmax", button8)
    has9 = op1("gmax", button9)
    has15 = op1("gmax", button15)
    has6 = op1("gmax", button6)

    p8 = op2("mul", colors[8], play)
    p9 = op2("mul", colors[9], play)
    p15 = op2("mul", colors[15], play)
    p13 = op2("mul", colors[13], play)
    p12 = op2("mul", colors[12], play)
    player_pixels = op2("mul", colors[14], play)
    player = anchor(player_pixels)

    def marker_coords(marker):
        return [
            op1("gmax", op2("mul", marker, row)),
            op1("gmax", op2("mul", marker, col)),
        ]

    def coordinate_marker(r, c):
        dr = op1("abs", op2("sub", row, r))
        dc = op1("abs", op2("sub", col, c))
        er = clip01(affine(dr, -63.0, 1.0))
        ec = clip01(affine(dc, -63.0, 1.0))
        return op2("mul", er, ec)

    # Remember both portal positions.
    p67 = op2("mul", max_many([colors[6], colors[7]]), play)
    portal_seen = anchor(p67)
    rank = conv(
        "coords",
        np.asarray([[[[4032.0]], [[63.0]], [[0.0]]]], np.float32),
        np.asarray([1.0], np.float32),
    )
    reverse_rank = affine(rank, -1.0, 4097.0)

    def select_extreme(score):
        best = op1("gmax", score)
        selected = op1("relu", affine(op2("sub", score, best), 1.0, 1.0))
        return op2("mul", selected, portal_seen)

    high_marker = select_extreme(op2("mul", portal_seen, rank))
    low_marker = select_extreme(op2("mul", portal_seen, reverse_rank))
    low_rc = marker_coords(low_marker)
    high_rc = marker_coords(high_marker)
    observed_four = node("vconcat", low_rc + high_rc)
    observed_valid = dense(
        observed_four,
        np.zeros((1, 4), np.float32),
        np.ones(1, np.float32),
    )
    observed_mem = node("vconcat", [observed_four, observed_valid])
    portal_count = vaffine(op1("gmean", portal_seen), 4096.0)
    at_least_two = vadd(
        op1("vrelu", vaffine(portal_count, 1.0, -1.0)),
        vaffine(op1("vrelu", vaffine(portal_count, 1.0, -2.0)), -1.0),
    )
    old_portals = vsl("mem", 0, 5)
    portal_mem = vadd(
        vmul(observed_mem, at_least_two),
        vmul(old_portals, vnot(at_least_two)),
    )
    low_portal = coordinate_marker(vsl(portal_mem, 0, 1), vsl(portal_mem, 1, 2))
    high_portal = coordinate_marker(vsl(portal_mem, 2, 3), vsl(portal_mem, 3, 4))
    portal_valid = vsl(portal_mem, 4, 5)
    low_portal = op2("mul", low_portal, portal_valid)
    high_portal = op2("mul", high_portal, portal_valid)
    portal = op2("mul", max_many([portal_seen, low_portal, high_portal]), play)

    # Recover a bridge pivot even when the player covers part of it.
    pivot_rows = op1("rowmax", p13)
    pivot_cols = op1("colmax", p13)
    pivot_box = op2("mul", pivot_rows, pivot_cols)
    filled_pivot = max_many([p13, op2("mul", player_pixels, pivot_box)])
    pivot = anchor(filled_pivot)
    bridge_probe = max_many([p8, player_pixels])
    horizontal_test = read_shift(bridge_probe, 0, -2)
    large_test = max_many(
        [read_shift(bridge_probe, 0, 4), read_shift(bridge_probe, 4, 0)]
    )
    h = op2("mul", pivot, horizontal_test)
    v = op2("mul", pivot, complement(horizontal_test))
    hs = op2("mul", h, complement(large_test))
    hl = op2("mul", h, large_test)
    vs = op2("mul", v, complement(large_test))
    vl = op2("mul", v, large_test)

    def bridge_line(seed, horizontal, large):
        w = np.zeros((1, 1, 7, 7), np.float32)
        if horizontal:
            w[0, 0, 3, 1] = 1.0
            w[0, 0, 3, 3] = 1.0
            w[0, 0, 3, 5] = 1.0
        else:
            w[0, 0, 1, 3] = 1.0
            w[0, 0, 3, 3] = 1.0
            w[0, 0, 5, 3] = 1.0
        wp = param(w)
        z = seed
        for _ in range(4 if large else 2):
            z = clip01(node("conv", [z], w=wp))
        if large:
            z = max_many(
                [z, translate(z, 0, 2) if horizontal else translate(z, 2, 0)]
            )
        else:
            z = translate(z, 0, -2) if horizontal else translate(z, -2, 0)
        z = max_many(
            [z, translate(z, 2, 0) if horizontal else translate(z, 0, 2)]
        )
        return z

    bridge_current = op2(
        "mul",
        max_many(
            [
                bridge_line(hs, True, False),
                bridge_line(hl, True, True),
                bridge_line(vs, False, False),
                bridge_line(vl, False, True),
                filled_pivot,
            ]
        ),
        play,
    )
    bridge_flipped = op2(
        "mul",
        max_many(
            [
                bridge_line(hs, False, False),
                bridge_line(hl, False, True),
                bridge_line(vs, True, False),
                bridge_line(vl, True, True),
                filled_pivot,
            ]
        ),
        play,
    )

    def switch_masks(p):
        count = count_2x2(p)
        active = clip01(affine(count, 1.0, -3.0))
        at_least = clip01(affine(count, 1.0, -1.0))
        inactive = op1("relu", op2("sub", at_least, active))
        return active, inactive, count

    a9, b9, count9 = switch_masks(p9)
    a15, b15, count15 = switch_masks(p15)
    key_shape = op2(
        "min2",
        op1("relu", affine(count15, 1.0, -2.0)),
        op1("relu", affine(count15, -1.0, 4.0)),
    )
    key_shape = op2("mul", key_shape, play)
    key = op2("mul", key_shape, vnot(has15))
    has_key = op1("gmax", key)
    goal = op2("mul", colors[11], play)
    ordinary_target = max_many([key, op2("mul", goal, vnot(has_key))])

    # Match the empty and loaded crane pictures.
    crane1 = [
        "...8888",
        "...8...",
        ".8888..",
        "8888...",
        "8888...",
        ".8888..",
        "...8...",
    ]
    crane2 = [
        ".......",
        "...8888",
        ".8888..",
        "8888...",
        "8888...",
        ".8888..",
        "...8888",
    ]
    cw = np.empty((2, 1, 7, 7), np.float32)
    cb = np.empty(2, np.float32)
    for k, pattern in enumerate([crane1, crane2]):
        positives = 0
        for yy in range(7):
            for xx in range(7):
                if pattern[yy][xx] == "8":
                    cw[k, 0, yy, xx] = 1.0
                    positives += 1
                else:
                    cw[k, 0, yy, xx] = -1.0
        cb[k] = -float(positives) + 0.5
    crane_detect = op1("relu", conv(p8, cw, cb))
    crane_detect = conv(
        crane_detect,
        np.asarray([[[[2.0]], [[0.0]]], [[[0.0]], [[2.0]]]], np.float32),
    )
    empty_crane = sl(crane_detect, 0, 1)
    loaded_crane = sl(crane_detect, 1, 2)
    crane_marker = max_many([empty_crane, loaded_crane])
    crane_visible = op1("gmax", crane_marker)
    loaded = op1("gmax", loaded_crane)
    crane_r, crane_c = marker_coords(crane_marker)
    player_r, player_c = marker_coords(player)

    # The black track gives home even when the episode starts mid-move.
    left_region = clip01(affine(col, -63.0, 32.0))
    above_status = clip01(affine(row, -63.0, 62.0))
    track = op2("mul", colors[0], op2("mul", left_region, above_status))
    track_bottom = op1("gmax", op2("mul", track, row))
    track_left_rev = op1("gmax", op2("mul", track, complement(col)))
    inferred_origin_r = vaffine(track_bottom, 1.0, -2.0 / 63.0)
    inferred_origin_c = vaffine(vnot(track_left_rev), 1.0, 2.0 / 63.0)

    object_present = op1("gmax", p12)
    object_max_c = op1("gmax", op2("mul", p12, col))
    object_reverse_r = op1("gmax", op2("mul", p12, complement(row)))
    object_min_r = vnot(object_reverse_r)
    object_target_r = vaffine(object_min_r, 1.0, 5.0 / 63.0)
    object_target_c = vaffine(object_max_c, 1.0, -8.0 / 63.0)

    # If already loaded, infer pickup from the track rather than the
    # object's now-moved position.
    default_pickup_r = vaffine(inferred_origin_r, 1.0, -12.0 / 63.0)
    default_pickup_c = vaffine(inferred_origin_c, 1.0, 12.0 / 63.0)
    initial_pickup_r = vadd(
        vmul(object_target_r, vnot(loaded)), vmul(default_pickup_r, loaded)
    )
    initial_pickup_c = vadd(
        vmul(object_target_c, vnot(loaded)), vmul(default_pickup_c, loaded)
    )

    old_crane_mem = vsl("mem", 5, 10)
    old_valid = vsl(old_crane_mem, 4, 5)
    capture = vmul(vnot(old_valid), vmul(crane_visible, object_present))
    candidate_crane_mem = node(
        "vconcat",
        [
            inferred_origin_r,
            inferred_origin_c,
            initial_pickup_r,
            initial_pickup_c,
            capture,
        ],
    )
    crane_mem = vadd(old_crane_mem, vmul(candidate_crane_mem, capture))
    origin_r = vsl(crane_mem, 0, 1)
    origin_c = vsl(crane_mem, 1, 2)
    pickup_r = vsl(crane_mem, 2, 3)
    pickup_c = vsl(crane_mem, 3, 4)
    crane_valid = vsl(crane_mem, 4, 5)

    eq_origin_r = veq(crane_r, origin_r)
    eq_origin_c = veq(crane_c, origin_c)
    eq_pickup_r = veq(crane_r, pickup_r)
    eq_pickup_c = veq(crane_c, pickup_c)

    at_first_placement = vmul(loaded, vmul(eq_origin_r, eq_pickup_c))
    first_done = vclip(vadd(vsl("mem", 10, 11), at_first_placement))

    # Cross from the lower-right platform to the upper-right platform.
    safe_r = vaffine(origin_r, 1.0, -7.0 / 63.0)
    safe_c = vaffine(pickup_c, 1.0, 9.0 / 63.0)
    safe_marker = coordinate_marker(safe_r, safe_c)
    at_safe = vmul(veq(player_r, safe_r), veq(player_c, safe_c))
    middle_done = vclip(
        vadd(vsl("mem", 11, 12), vmul(first_done, at_safe))
    )

    at_top = vmul(
        middle_done, vmul(loaded, vmul(eq_pickup_r, eq_pickup_c))
    )
    top_done = vclip(vadd(vsl("mem", 12, 13), at_top))

    left_limit = vaffine(origin_c, 1.0, 6.0 / 63.0)
    lower_limit = vaffine(origin_r, 1.0, 4.0 / 63.0)
    player_on_left = vmul(
        vgreater(left_limit, player_c), vgreater(player_r, lower_limit)
    )
    left_done = vclip(
        vadd(vsl("mem", 13, 14), vmul(top_done, player_on_left))
    )
    at_left_placement = vmul(
        left_done, vmul(eq_origin_r, eq_origin_c)
    )
    bottom_done = vclip(
        vadd(vsl("mem", 14, 15), at_left_placement)
    )
    node(
        "vconcat",
        [portal_mem, crane_mem, first_done, middle_done, top_done,
         left_done, bottom_done],
        out="mem_next",
    )

    # Use intermediate targets while the crane route is underway.
    safe_phase = vmul(crane_valid, vmul(first_done, vnot(top_done)))
    portal_phase = vmul(crane_valid, vmul(top_done, vnot(left_done)))
    special_phase = vclip(vadd(safe_phase, portal_phase))
    target = max_many(
        [
            op2("mul", ordinary_target, vnot(special_phase)),
            op2("mul", safe_marker, safe_phase),
            op2("mul", low_portal, portal_phase),
        ]
    )

    # Crane phase 0: go to the object and attach it.
    phase0 = vmul(crane_valid, vmul(vnot(loaded), vnot(first_done)))
    up0 = vmul(phase0, vgreater(crane_r, pickup_r))
    right0 = vmul(
        phase0, vmul(eq_pickup_r, vgreater(pickup_c, crane_c))
    )
    left0 = vmul(
        phase0, vmul(eq_pickup_r, vgreater(crane_c, pickup_c))
    )
    grab0 = vmul(phase0, vmul(eq_pickup_r, eq_pickup_c))

    # Carry it to the lower-right gap.
    phase1 = vmul(crane_valid, vmul(loaded, vnot(first_done)))
    above_origin = vgreater(origin_r, crane_r)
    right_of_origin = vgreater(crane_c, origin_c)
    left1 = vmul(phase1, vmul(above_origin, right_of_origin))
    down1 = vmul(phase1, vmul(above_origin, eq_origin_c))
    right1 = vmul(
        phase1, vmul(eq_origin_r, vgreater(pickup_c, crane_c))
    )

    # Once the player is safely across, move it back to the upper gap.
    phase2 = vmul(
        crane_valid, vmul(middle_done, vmul(loaded, vnot(top_done)))
    )
    up2 = vmul(phase2, vgreater(crane_r, pickup_r))

    # After teleportation, route the object left and then down.
    phase3 = vmul(
        crane_valid, vmul(left_done, vmul(loaded, vnot(bottom_done)))
    )
    left3 = vmul(phase3, vgreater(crane_c, origin_c))
    down3 = vmul(
        phase3, vmul(eq_origin_c, vgreater(origin_r, crane_r))
    )

    up_cmd = vclip(vadd(up0, up2))
    down_cmd = vclip(vadd(down1, down3))
    left_cmd = vclip(vadd(vadd(left0, left1), left3))
    right_cmd = vclip(vadd(right0, right1))
    grab_cmd = grab0

    control10 = op2("mul", colors[10], ui)
    max_control_c = op1("gmax", op2("mul", control10, col))
    reverse_control_c = op1("gmax", op2("mul", control10, complement(col)))
    min_control_c = vnot(reverse_control_c)
    midpoint = vaffine(vadd(max_control_c, min_control_c), 0.5)
    left_half = clip01(affine(op2("sub", midpoint, col), 63.0))
    right_half = clip01(affine(op2("sub", col, midpoint), 63.0))
    left_control = op2("mul", control10, left_half)
    right_control = op2("mul", control10, right_half)
    up_control = translate(left_control, 0, 5)
    down_control = translate(right_control, 0, 5)
    grab_control = translate(left_control, 6, 5)
    crane_click = affine(
        max_many(
            [
                op2("mul", up_control, up_cmd),
                op2("mul", down_control, down_cmd),
                op2("mul", left_control, left_cmd),
                op2("mul", right_control, right_cmd),
                op2("mul", grab_control, grab_cmd),
            ]
        ),
        3.0,
    )

    base = max_many(
        [
            op2("mul", colors[2], play),
            op2("mul", colors[1], play),
            p12,
            p67,
            portal,
            goal,
            key,
            player_pixels,
        ]
    )
    walks = []
    for s in range(8):
        bridge = bridge_flipped if (s & 1) else bridge_current
        floor9 = b9 if (s & 2) else a9
        floor15 = b15 if (s & 4) else a15
        walks.append(
            op2("mul", max_many([base, bridge, floor9, floor15]), play)
        )
    walk_all = node("concat", walks)

    # Discounted reachability across movement, switch and portal edges.
    body = []
    state_ids = ["state%d" % i for i in range(8)]
    movement_kernel = np.zeros((4, 1, 5, 5), np.float32)
    movement_kernel[0, 0, 0, 2] = 1.0
    movement_kernel[1, 0, 4, 2] = 1.0
    movement_kernel[2, 0, 2, 0] = 1.0
    movement_kernel[3, 0, 2, 4] = 1.0
    movement_param = param(movement_kernel)
    moved = []
    for sid in state_ids:
        q = node("conv", [sid], dst=body, w=movement_param)
        moved.append(max_many([sl(q, i, i + 1, body) for i in range(4)], body))
    moved_all = node("concat", moved, dst=body)
    states_all = node("concat", state_ids, dst=body)
    flips = []
    for bit, available in [(1, has8), (2, has9), (4, has15)]:
        w = np.zeros((8, 8, 1, 1), np.float32)
        for s in range(8):
            w[s, s ^ bit, 0, 0] = 1.0
        flips.append(op2("mul", conv(states_all, w, dst=body), available, body))
    flip_best = max_many(flips, body)
    portal_values = op2("mul", states_all, portal, body)
    portal_best = op1("gmax", portal_values, body)
    portal_offer = op2("mul", portal, portal_best, body)
    portal_offer = op2("mul", portal_offer, has6, body)
    candidates = max_many([moved_all, flip_best, portal_offer], body)
    candidates = op2("mul", candidates, walk_all, body)
    discount = np.zeros((8, 8, 1, 1), np.float32)
    for i in range(8):
        discount[i, i, 0, 0] = 0.97
    candidates = conv(candidates, discount, dst=body)
    updated = op2("max2", candidates, target, body)
    next_ids = [sl(updated, i, i + 1, body) for i in range(8)]
    graph.append(
        {
            "id": "reach",
            "op": "loop",
            "T": 96,
            "state": [{"id": sid, "init": target} for sid in state_ids],
            "body": body,
            "next": {sid: n for sid, n in zip(state_ids, next_ids)},
        }
    )
    values = ["reach." + sid for sid in state_ids]
    current_value = values[0]

    def at_player(spatial):
        return op1("gmax", op2("mul", spatial, player))

    offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    direction_scores = [
        at_player(read_shift(current_value, dy, dx)) for dy, dx in offsets
    ]
    click8_score = vmul(at_player(values[1]), has8)
    click9_score = vmul(at_player(values[2]), has9)
    click15_score = vmul(at_player(values[4]), has15)
    player_on_portal = at_player(portal)
    portal_destination = op1("gmax", op2("mul", current_value, portal))
    click6_score = vmul(vmul(player_on_portal, portal_destination), has6)
    normal_click = max_many(
        [
            op2("mul", button8, click8_score),
            op2("mul", button9, click9_score),
            op2("mul", button15, click15_score),
            op2("mul", button6, click6_score),
        ]
    )
    click_map = max_many([normal_click, crane_click])
    node("mul", [click_map, one], out="click")
    click_score = op1("gmax", click_map)

    value_here = at_player(current_value)
    fallback_gate = op1("vrelu", vaffine(value_here, -1000.0, 1.0))
    target_row = op1("gmax", op2("mul", target, row))
    target_col = op1("gmax", op2("mul", target, col))
    dr = op1("abs", op2("sub", row, target_row))
    dc = op1("abs", op2("sub", col, target_col))
    closeness = affine(op2("add", dr, dc), -1.0, 2.0)
    fallback_field = op2("mul", closeness, walks[0])
    adjusted = []
    for primary, (dy, dx) in zip(direction_scores, offsets):
        local = at_player(read_shift(fallback_field, dy, dx))
        local = vmul(vaffine(local, 0.001), fallback_gate)
        adjusted.append(vadd(primary, local))

    action_features = node("vconcat", adjusted + [click_score])
    output_weights = np.zeros((8, 5), np.float32)
    for i in range(4):
        output_weights[i + 1, i] = 1.0
    output_weights[6, 4] = 1.0
    output_bias = np.asarray([-5, 0, 0, 0, 0, -5, 0, -5], np.float32)
    dense(action_features, output_weights, output_bias, out="logits")

    return {"graph": graph, "memory_size": 15, "params": params}
