import numpy as np


def build():
    graph = []
    params = {}
    node_number = 0
    param_number = 0

    def node(kind, inputs, out=None, **fields):
        nonlocal node_number
        if out is None:
            out = "n%d" % node_number
            node_number += 1
        item = {"id": out, "op": kind, "in": list(inputs)}
        item.update(fields)
        graph.append(item)
        return out

    def parameter(value):
        nonlocal param_number
        name = "p%d" % param_number
        param_number += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def dense(x, weights, bias=None, out=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("dense", [x], out=out, **fields)

    def conv(x, weights, bias=None):
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim == 2:
            weights = weights[:, :, None, None]
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("conv", [x], **fields)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vs(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs, out=None):
        return node("vconcat", xs, out=out)

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

    def affine(x, scale, bias=0.0):
        return dense(x, [[scale]], [bias])

    def one_minus(x):
        return affine(x, -1.0, 1.0)

    def vabs(x, size=1):
        pos = node("vrelu", [x])
        neg = node("vrelu", [
            dense(x, -np.eye(size, dtype=np.float32))
        ])
        return vadd(pos, neg)

    def clip(x, size=1):
        pos = node("vrelu", [x])
        excess = node("vrelu", [
            dense(
                pos,
                np.eye(size, dtype=np.float32),
                -np.ones(size, dtype=np.float32),
            )
        ])
        return dense(
            vcat(pos, excess),
            np.concatenate(
                [
                    np.eye(size, dtype=np.float32),
                    -np.eye(size, dtype=np.float32),
                ],
                axis=1,
            ),
        )

    def choose(old, new, gate):
        return vadd(vmul(old, one_minus(gate)), vmul(new, gate))

    def vsum(*xs):
        result = xs[0]
        for x in xs[1:]:
            result = vadd(result, x)
        return result

    def vproduct(*xs):
        result = xs[0]
        for x in xs[1:]:
            result = vmul(result, x)
        return result

    # Memory:
    # 0 initialized; 1 masking mode; 2 planner mode;
    # 3 fallback reveal; 4 fallback spin; 5 fallback anchor;
    # 6:10 fallback target; 10:14 tracked marker bounds;
    # 14:16 previous core maxima; 16:22 planner phases;
    # 22:26 A endpoints; 26:30 B endpoints; 30:34 C endpoints;
    # 34:38 B bounds; 38:42 C bounds; 42 alternative candidate;
    # 43:47 A bounds; 47 second half-turn step.
    initialized = vs("mem", 0, 1)
    old_mode = vs("mem", 1, 2)
    old_special = vs("mem", 2, 3)
    fallback_reveal = vs("mem", 3, 4)
    fallback_spin = vs("mem", 4, 5)
    old_anchor = vs("mem", 5, 6)
    old_target = vs("mem", 6, 10)
    old_selected = vs("mem", 10, 14)
    old_core_max = vs("mem", 14, 16)
    phases = vs("mem", 16, 22)
    old_a = vs("mem", 22, 26)
    old_b = vs("mem", 26, 30)
    old_c = vs("mem", 30, 34)
    old_b_box = vs("mem", 34, 38)
    old_c_box = vs("mem", 38, 42)
    retry = vs("mem", 42, 43)
    old_a_box = vs("mem", 43, 47)
    second_turn = vs("mem", 47, 48)

    p1 = vs(phases, 0, 1)
    p2 = vs(phases, 1, 2)
    p3 = vs(phases, 2, 3)
    p4 = vs(phases, 3, 4)
    p5 = vs(phases, 4, 5)
    p6 = vs(phases, 5, 6)

    ones = sl("coords", 2, 3)
    row = sl("coords", 0, 1)
    valid_row = node(
        "min2",
        [node("relu", [conv(row, [[63.0]])]), ones],
    )

    # Estimate the background from colour frequency.
    colour_mean = node("gmean", ["x0"])
    bg_raw = dense(
        colour_mean,
        100.0 * np.eye(16, dtype=np.float32),
        -25.0 * np.ones(16, dtype=np.float32),
    )
    bg_flags = clip(bg_raw, 16)
    bg_pixel = conv(mul("x0", bg_flags), [np.ones(16)])
    not_bg = sub(ones, bg_pixel)

    black = sl("x0", 0, 1)
    green = sl("x0", 3, 4)
    grey = sl("x0", 4, 5)
    cyan = sl("x0", 8, 9)

    grey_pixels = mul(mul(grey, not_bg), valid_row)
    grey_count = affine(node("gmean", [grey_pixels]), 4096.0)
    grey_guess = clip(grey_count)
    mode = choose(grey_guess, old_mode, initialized)
    not_mode = one_minus(mode)

    # Selected-body and nearby-connector masks.
    ordinary = conv(
        "x0",
        [[-float(i in (3, 4, 8)) for i in range(16)]],
        [1.0],
    )
    grey_body = mul(mul(ordinary, not_bg), valid_row)
    black_body = mul(mul(black, not_bg), valid_row)
    core = add(mul(grey_body, mode), mul(black_body, not_mode))
    has_core = clip(affine(node("gmean", [core]), 4096.0))

    dilated = node(
        "conv",
        [core],
        w=parameter(np.ones((1, 1, 7, 7), dtype=np.float32)),
    )
    near_core = node("min2", [node("relu", [dilated]), ones])
    visible_cyan = mul(mul(cyan, not_bg), valid_row)
    selected_cyan = mul(visible_cyan, near_core)
    selected_green = mul(mul(green, valid_row), near_core)
    selected_markers = add(selected_cyan, selected_green)
    other_cyan = mul(visible_cyan, sub(ones, near_core))
    has_cyan = node("gmax", [selected_cyan])
    has_green = node("gmax", [selected_green])

    bounds_basis = conv(
        "coords",
        [
            [-63.0, 0.0, 0.0],
            [63.0, 0.0, 0.0],
            [0.0, -63.0, 0.0],
            [0.0, 63.0, 0.0],
        ],
        [64.0, 1.0, 64.0, 1.0],
    )

    def bounds(mask):
        return node("gmax", [mul(bounds_basis, mask)])

    seen_selected = bounds(selected_markers)
    seen_other = bounds(other_cyan)
    core_box = bounds(core)
    core_max = node(
        "gmax",
        [cat(
            mul(sl(bounds_basis, 1, 2), core),
            mul(sl(bounds_basis, 3, 4), core),
        )],
    )

    # Extract the first and last marker block centres in row-major order.
    key_first = conv(
        "coords",
        [[-4032.0, -63.0, 0.0]],
        [4096.0],
    )
    key_last = conv(
        "coords",
        [[4032.0, 63.0, 0.0]],
        [1.0],
    )
    rowcol = conv(
        "coords",
        [[63.0, 0.0, 0.0], [0.0, 63.0, 0.0]],
        [1.0, 1.0],
    )

    def endpoint_at(key, adjustment):
        best = node("gmax", [mul(key, selected_markers)])
        distance = node("abs", [sub(key, best)])
        point = mul(
            node("relu", [conv(distance, [[-1.0]], [1.0])]),
            selected_markers,
        )
        raw = node("gmax", [mul(rowcol, point)])
        return dense(
            raw,
            np.eye(2, dtype=np.float32),
            np.full(2, adjustment, dtype=np.float32),
        )

    seen_endpoints = vcat(
        endpoint_at(key_first, 0.0),
        endpoint_at(key_last, -2.0),
    )

    marker_pixels = affine(
        node("gmean", [selected_markers]), 4096.0
    )
    exactly_two = clip(
        affine(vabs(affine(marker_pixels, 1.0, -18.0)), -0.25, 1.0)
    )
    special_guess = vproduct(mode, exactly_two, grey_guess, has_core)
    special = choose(special_guess, old_special, initialized)
    fallback = one_minus(special)

    initial = vproduct(one_minus(initialized), has_core)
    start_special = vmul(initial, special)
    start_fallback = vmul(initial, fallback)

    # Track selected bounds through translations; refresh after clicks
    # and rotations.
    displacement = dense(
        vcat(core_max, old_core_max),
        [
            [-1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, -1.0],
        ],
    )
    predicted_selected = vadd(old_selected, displacement)
    previous_rotation = vs("aprev", 5, 6)
    previous_click = vs("aprev", 6, 7)
    refresh = clip(vsum(initial, previous_rotation, previous_click))
    tracked_selected = choose(
        predicted_selected, seen_selected, refresh
    )

    piece_a = choose(old_a, seen_endpoints, start_special)
    piece_b = choose(old_b, seen_endpoints, p1)
    piece_c = choose(old_c, seen_endpoints, p2)
    all_pieces = vcat(piece_a, piece_b, piece_c)

    # Enumerate the attachment endpoint and quarter-turn for B.
    matrices = {
        name: np.zeros((8, 12), dtype=np.float32)
        for name in (
            "mr", "mc", "dr", "dc", "br2", "bc2",
            "cr2", "cc2", "vr", "vc"
        )
    }
    rotations = [
        (1, 0, 0, 1),
        (0, 1, -1, 0),
        (-1, 0, 0, -1),
        (0, -1, 1, 0),
    ]

    for i in range(8):
        end = i // 4
        rot = i % 4
        ar = 0 if end == 0 else 2
        ac = ar + 1
        other_r = 2 if end == 0 else 0
        other_c = other_r + 1
        rr, rc, cr, cc = rotations[rot]

        vrow = np.zeros(12, dtype=np.float32)
        vcol = np.zeros(12, dtype=np.float32)
        vrow[6] += rr
        vrow[4] -= rr
        vrow[7] += rc
        vrow[5] -= rc
        vcol[6] += cr
        vcol[4] -= cr
        vcol[7] += cc
        vcol[5] -= cc

        attach_r = np.zeros(12, dtype=np.float32)
        attach_c = np.zeros(12, dtype=np.float32)
        other_row = np.zeros(12, dtype=np.float32)
        other_col = np.zeros(12, dtype=np.float32)
        attach_r[ar] = 1.0
        attach_c[ac] = 1.0
        other_row[other_r] = 1.0
        other_col[other_c] = 1.0

        mr = attach_r + vrow
        mc = attach_c + vcol
        matrices["mr"][i] = mr
        matrices["mc"][i] = mc
        matrices["dr"][i] = mr - other_row
        matrices["dc"][i] = mc - other_col
        matrices["br2"][i] = 2.0 * attach_r + vrow
        matrices["bc2"][i] = 2.0 * attach_c + vcol
        matrices["cr2"][i] = mr + other_row
        matrices["cc2"][i] = mc + other_col
        matrices["vr"][i] = vrow
        matrices["vc"][i] = vcol

    candidates = {
        name: dense(all_pieces, matrix)
        for name, matrix in matrices.items()
    }

    # Test whether the remaining gap matches a rotation of C.
    pair_input = vcat(candidates["dr"], candidates["dc"], piece_c)
    compare = np.zeros((64, 20), dtype=np.float32)
    for i in range(8):
        for rot, (rr, rc, cr, cc) in enumerate(rotations):
            j = i * 4 + rot
            compare[2 * j, i] = 1.0
            compare[2 * j + 1, 8 + i] = 1.0
            compare[2 * j, 16] += rr
            compare[2 * j, 18] -= rr
            compare[2 * j, 17] += rc
            compare[2 * j, 19] -= rc
            compare[2 * j + 1, 16] += cr
            compare[2 * j + 1, 18] -= cr
            compare[2 * j + 1, 17] += cc
            compare[2 * j + 1, 19] -= cc

    differences = dense(pair_input, compare)
    absolute_differences = vabs(differences, 64)
    pair_sum = np.zeros((32, 64), dtype=np.float32)
    for j in range(32):
        pair_sum[j, 2 * j:2 * j + 2] = 1.0
    errors = dense(absolute_differences, pair_sum)
    matches = clip(
        dense(
            errors,
            -0.5 * np.eye(32, dtype=np.float32),
            np.ones(32, dtype=np.float32),
        ),
        32,
    )
    group = np.zeros((8, 32), dtype=np.float32)
    for i in range(8):
        group[i, 4 * i:4 * i + 4] = 1.0
    eligible = clip(dense(matches, group), 8)

    for name, scale, bias in (
        ("mr", 1.0, -3.0),
        ("mr", -1.0, 60.0),
        ("mc", 1.0, -3.0),
        ("mc", -1.0, 60.0),
    ):
        edge = clip(
            dense(
                candidates[name],
                scale * np.eye(8, dtype=np.float32),
                np.full(8, bias, dtype=np.float32),
            ),
            8,
        )
        eligible = vmul(eligible, edge)

    prefix_matrix = np.tril(
        np.ones((8, 8), dtype=np.float32), k=-1
    )
    prefix = clip(dense(eligible, prefix_matrix), 8)
    first_choice = vmul(
        eligible,
        dense(
            prefix,
            -np.eye(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
        ),
    )
    after_first = vmul(eligible, prefix)
    second_prefix = clip(dense(after_first, prefix_matrix), 8)
    second_choice = vmul(
        after_first,
        dense(
            second_prefix,
            -np.eye(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
        ),
    )
    choice = vadd(
        vmul(first_choice, one_minus(retry)),
        vmul(second_choice, retry),
    )

    def selected_candidate(name):
        return dense(vmul(candidates[name], choice), [np.ones(8)])

    b_row2 = selected_candidate("br2")
    b_col2 = selected_candidate("bc2")
    c_row2 = selected_candidate("cr2")
    c_col2 = selected_candidate("cc2")
    b_vr = selected_candidate("vr")
    b_vc = selected_candidate("vc")
    c_vr = selected_candidate("dr")
    c_vc = selected_candidate("dc")

    current_vector = dense(
        seen_endpoints,
        [[-1, 0, 1, 0], [0, -1, 0, 1]],
    )

    def orientation_ok(want_r, want_c):
        inp = vcat(current_vector, want_r, want_c)
        diff = dense(
            inp,
            [
                [1, 0, -1, 0],
                [0, 1, 0, -1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
        )
        absolute = vabs(diff, 4)
        errors2 = dense(
            absolute,
            [[1, 1, 0, 0], [0, 0, 1, 1]],
        )
        matches2 = clip(
            dense(
                errors2,
                -0.5 * np.eye(2, dtype=np.float32),
                np.ones(2, dtype=np.float32),
            ),
            2,
        )
        return clip(dense(matches2, [[1, 1]]))

    b_oriented = orientation_ok(b_vr, b_vc)
    c_oriented = orientation_ok(c_vr, c_vc)

    def centre_delta(target_row2, target_col2):
        return dense(
            vcat(target_row2, target_col2, tracked_selected),
            [
                [1, 0, 1, -1, 0, 0],
                [0, 1, 0, 0, 1, -1],
            ],
            [-63.0, -63.0],
        )

    b_delta = centre_delta(b_row2, b_col2)
    c_delta = centre_delta(c_row2, c_col2)

    def aligned(delta):
        total = dense(vabs(delta, 2), [[1.0, 1.0]])
        return clip(affine(total, -0.25, 1.0))

    b_aligned = aligned(b_delta)
    c_aligned = aligned(c_delta)
    b_ready = vsum(vmul(p3, b_oriented), p4)
    c_ready = vsum(vmul(p5, c_oriented), p6)
    b_done = vmul(b_ready, b_aligned)
    c_done = vmul(c_ready, c_aligned)
    b_moving = vmul(b_ready, one_minus(b_aligned))
    c_moving = vmul(c_ready, one_minus(c_aligned))

    # A geometrically aligned B must also have a green connector.
    good_b = vmul(b_done, has_green)
    bad_b = vmul(b_done, one_minus(has_green))

    # At a completed C alignment, two quarter-turns try the other
    # connector assignment. Afterwards the centre is realigned.
    rotate = vsum(
        vmul(p3, one_minus(b_oriented)),
        vmul(p5, one_minus(c_oriented)),
        bad_b,
        c_done,
        second_turn,
    )
    planner_click = vsum(start_special, p1, p2, good_b)
    planner_delta = vadd(
        vmul(b_delta, b_moving),
        vmul(c_delta, c_moving),
    )

    next_p1 = start_special
    next_p2 = p1
    next_p3 = vsum(p2, vmul(p3, one_minus(b_oriented)), bad_b)
    next_p4 = b_moving
    next_p5 = vsum(good_b, vmul(p5, one_minus(c_oriented)), second_turn)
    next_p6 = c_moving
    next_phases = vcat(
        next_p1, next_p2, next_p3, next_p4, next_p5, next_p6
    )
    next_second_turn = c_done

    a_box = choose(old_a_box, core_box, start_special)
    b_box = choose(old_b_box, core_box, clip(vsum(p1, good_b)))
    c_box = choose(old_c_box, core_box, p2)
    next_retry = clip(vadd(retry, bad_b))

    def box_mask(box, padding):
        spatial = conv(
            "coords",
            [
                [63, 0, 0],
                [-63, 0, 0],
                [0, 63, 0],
                [0, -63, 0],
            ],
        )
        adjusted = add(spatial, box)
        shifted = conv(
            adjusted,
            np.eye(4, dtype=np.float32),
            [
                -63.5 + padding,
                -0.5 + padding,
                -63.5 + padding,
                -0.5 + padding,
            ],
        )
        clipped = node("min2", [node("relu", [shifted]), ones])
        return mul(
            mul(sl(clipped, 0, 1), sl(clipped, 1, 2)),
            mul(sl(clipped, 2, 3), sl(clipped, 3, 4)),
        )

    current_b_padded = box_mask(core_box, 4.0)
    current_b_strict = box_mask(core_box, 0.0)
    stored_a_padded = box_mask(old_a_box, 4.0)
    stored_a_strict = box_mask(old_a_box, 0.0)
    stored_b_padded = box_mask(old_b_box, 4.0)
    stored_c_padded = box_mask(old_c_box, 4.0)

    remaining_grey = mul(
        mul(grey_pixels, sub(ones, current_b_padded)),
        sub(ones, stored_a_padded),
    )
    click_b = mul(
        mul(grey_pixels, stored_b_padded),
        sub(ones, stored_a_strict),
    )
    click_c = mul(
        mul(
            mul(grey_pixels, stored_c_padded),
            sub(ones, stored_a_strict),
        ),
        sub(ones, current_b_strict),
    )
    planner_click_map = add(
        add(
            mul(grey_pixels, start_special),
            mul(remaining_grey, p1),
        ),
        add(mul(click_b, p2), mul(click_c, good_b)),
    )

    # Bounding-box fallback for other layouts.
    reveal_now = vproduct(start_fallback, mode)
    fallback_ready = vproduct(
        initialized, fallback, one_minus(fallback_reveal)
    )
    fb_target_update = vsum(
        vproduct(start_fallback, not_mode), fallback_reveal
    )
    fb_new_target = vadd(
        vmul(seen_other, vproduct(start_fallback, not_mode)),
        vmul(seen_selected, fallback_reveal),
    )
    fallback_target = vadd(
        vmul(old_target, one_minus(clip(fb_target_update))),
        fb_new_target,
    )

    current_anchor = node("gmax", [mul(key_first, core)])
    anchor = choose(old_anchor, current_anchor, start_fallback)
    anchor_diff = node("abs", [sub(key_first, old_anchor)])
    anchor_point = node(
        "relu", [conv(anchor_diff, [[-1.0]], [1.0])]
    )

    fallback_geometry = dense(
        vcat(fallback_target, tracked_selected),
        [
            [-1, 1, 0, 0, 1, -1, 0, 0],
            [0, 0, -1, 1, 0, 0, 1, -1],
            [0, 0, 0, 0, -1, -1, 1, 1],
            [-1, -1, 1, 1, 0, 0, 0, 0],
        ],
    )
    fb_dy = vs(fallback_geometry, 0, 1)
    fb_dx = vs(fallback_geometry, 1, 2)
    fb_orientation = vmul(
        vs(fallback_geometry, 2, 3),
        vs(fallback_geometry, 3, 4),
    )
    fb_rotate_shape = clip(affine(fb_orientation, -1.0 / 3.0))
    fb_spin_start = vproduct(
        fallback_ready,
        one_minus(has_cyan),
        has_green,
        one_minus(fallback_spin),
    )
    fb_rotate = vsum(fb_rotate_shape, fallback_spin, fb_spin_start)

    no_core_click = vproduct(fallback, one_minus(has_core))
    fallback_click = vsum(reveal_now, fallback_reveal, no_core_click)
    foreground = mul(mul(not_bg, valid_row), sub(ones, cyan))
    fallback_click_map = add(
        add(
            mul(grey_pixels, reveal_now),
            mul(anchor_point, fallback_reveal),
        ),
        mul(foreground, no_core_click),
    )

    planner_features = vcat(planner_delta, rotate, planner_click)
    planner_logits = dense(
        planner_features,
        [
            [0, 0, 0, 0],
            [-1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 10000, 0],
            [0, 0, 0, 20000],
            [0, 0, 0, 0],
        ],
        [-1000, 0, 0, 0, 0, -0.5, -1000, -1000],
    )
    fallback_features = vcat(fb_dy, fb_dx, fb_rotate, fallback_click)
    fallback_logits = dense(
        fallback_features,
        [
            [0, 0, 0, 0],
            [-1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1000, 0],
            [0, 0, 0, 10000],
            [0, 0, 0, 0],
        ],
        [-1000, 0, 0, 0, 0, -0.5, -1000, -1000],
    )
    logits = vadd(
        vmul(planner_logits, special),
        vmul(fallback_logits, fallback),
    )
    dense(logits, np.eye(8, dtype=np.float32), out="logits")

    click_map = add(
        mul(planner_click_map, special),
        mul(fallback_click_map, fallback),
    )
    node("slice", [click_map], out="click", start=0, stop=1)

    next_initialized = clip(vadd(initialized, initial))
    next_mode = choose(old_mode, mode, initial)
    next_special = choose(old_special, special, initial)

    vcat(
        next_initialized,
        next_mode,
        next_special,
        reveal_now,
        fb_spin_start,
        anchor,
        fallback_target,
        tracked_selected,
        core_max,
        next_phases,
        piece_a,
        piece_b,
        piece_c,
        b_box,
        c_box,
        next_retry,
        a_box,
        next_second_turn,
        out="mem_next",
    )

    return {
        "graph": graph,
        "memory_size": 48,
        "params": params,
    }
