import numpy as np


def build():
    graph = []
    params = {}
    node_number = 0
    param_number = 0

    def parameter(value):
        nonlocal param_number
        param_number += 1
        name = "p%d" % param_number
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, name=None, **fields):
        nonlocal node_number
        if name is None:
            node_number += 1
            name = "n%d" % node_number
        item = {"id": name, "op": op, "in": list(inputs)}
        item.update(fields)
        graph.append(item)
        return name

    def conv(x, weights, bias=None, name=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("conv", [x], name=name, **fields)

    def dense(x, weights, bias=None, name=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("dense", [x], name=name, **fields)

    def unary(op, x):
        return node(op, [x])

    def binary(op, x, y):
        return node(op, [x, y])

    def sslice(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vslice(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def saffine(x, scale, bias):
        return conv(
            x,
            np.array([[[[scale]]]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
        )

    def vaffine(xs, weights, bias=0.0):
        x = xs[0] if len(xs) == 1 else node("vconcat", xs)
        return dense(
            x,
            np.asarray([weights], dtype=np.float32),
            np.asarray([bias], dtype=np.float32),
        )

    def clip01(x):
        a = unary("vrelu", x)
        b = unary("vrelu", vaffine([x], [1.0], -1.0))
        return vaffine([a, b], [1.0, -1.0])

    def vand(*xs):
        out = xs[0]
        for x in xs[1:]:
            out = binary("vmul", out, x)
        return out

    def vor(x, y):
        return clip01(binary("vadd", x, y))

    def spatial_equal(x, y, one):
        difference = unary("abs", binary("sub", x, y))
        near = unary("relu", saffine(difference, -2.0, 2.0))
        return binary("min2", near, one)

    one = sslice("coords", 2, 3)
    one_v = unary("gmax", one)

    # Locate target diamonds and uncovered moving marker centers.
    d = sslice("x0", 13, 14)
    cross = np.zeros((1, 1, 3, 3), dtype=np.float32)
    cross[0, 0, 0, 1] = 1.0
    cross[0, 0, 1, 0] = 1.0
    cross[0, 0, 1, 2] = 1.0
    cross[0, 0, 2, 1] = 1.0

    goal = unary(
        "relu",
        conv(d, cross, np.array([-3.0], dtype=np.float32)),
    )
    occupied = binary("mul", goal, d)
    open_goal = binary("mul", goal, binary("sub", one, d))

    dilation = cross.copy()
    dilation[0, 0, 1, 1] = 1.0
    goal_area = binary("min2", conv(goal, dilation), one)
    markers = binary("mul", d, binary("sub", one, goal_area))

    # Manhattan distance to the nearest uncovered target.
    flood_init = saffine(open_goal, 128.0, 0.0)
    shifts = np.zeros((4, 1, 3, 3), dtype=np.float32)
    shifts[0, 0, 0, 1] = 1.0
    shifts[1, 0, 1, 0] = 1.0
    shifts[2, 0, 1, 2] = 1.0
    shifts[3, 0, 2, 1] = 1.0
    shift_w = parameter(shifts)
    shift_b = parameter(np.full(4, -1.0, dtype=np.float32))

    graph.append({
        "id": "flood",
        "op": "loop",
        "T": 128,
        "state": [{"id": "f_state", "init": flood_init}],
        "body": [
            {
                "id": "f_shift",
                "op": "conv",
                "in": ["f_state"],
                "w": shift_w,
                "b": shift_b,
            },
            {
                "id": "f_a",
                "op": "slice",
                "in": ["f_shift"],
                "start": 0,
                "stop": 1,
            },
            {
                "id": "f_b",
                "op": "slice",
                "in": ["f_shift"],
                "start": 1,
                "stop": 2,
            },
            {
                "id": "f_c",
                "op": "slice",
                "in": ["f_shift"],
                "start": 2,
                "stop": 3,
            },
            {
                "id": "f_d",
                "op": "slice",
                "in": ["f_shift"],
                "start": 3,
                "stop": 4,
            },
            {"id": "f_ab", "op": "max2", "in": ["f_a", "f_b"]},
            {"id": "f_cd", "op": "max2", "in": ["f_c", "f_d"]},
            {"id": "f_near", "op": "max2", "in": ["f_ab", "f_cd"]},
            {"id": "f_next", "op": "max2", "in": ["f_state", "f_near"]},
        ],
        "next": {"f_state": "f_next"},
    })

    distance = saffine("flood.f_state", -1.0, 128.0)
    marker_cost = binary("mul", markers, distance)
    reward = vaffine(
        [unary("gmean", occupied), unary("gmean", marker_cost)],
        [4096.0 * 1024.0, -4096.0],
    )

    # Detect resize-button halves and rotate buttons.
    small_left = [
        "22222",
        "2???4",
        "2?444",
        "2???4",
        "22222",
    ]
    small_right = [
        "22222",
        "4???2",
        "444?2",
        "4???2",
        "22222",
    ]
    large_left = [
        "2222222",
        "2444443",
        "24???43",
        "24?4443",
        "24???43",
        "2444443",
        "2222222",
    ]
    large_right = [
        "2222222",
        "3444442",
        "34???42",
        "3444?42",
        "34???42",
        "3444442",
        "2222222",
    ]
    rotate_pattern = [
        "22222",
        "24?42",
        "2???2",
        "24?42",
        "22222",
    ]

    def transpose(pattern):
        return [
            "".join(row[j] for row in pattern)
            for j in range(len(pattern[0]))
        ]

    def pattern_weights(patterns):
        k = len(patterns[0])
        weights = np.zeros((len(patterns), 16, k, k), dtype=np.float32)
        bias = np.zeros(len(patterns), dtype=np.float32)
        for i, pattern in enumerate(patterns):
            count = 0
            for y, row in enumerate(pattern):
                for x, char in enumerate(row):
                    if char != "?":
                        weights[i, int(char), y, x] = 1.0
                        count += 1
            bias[i] = 1.0 - count
        return weights, bias

    sw, sb = pattern_weights([
        small_left,
        small_right,
        transpose(small_left),
        transpose(small_right),
        rotate_pattern,
    ])
    lw, lb = pattern_weights([
        large_left,
        large_right,
        transpose(large_left),
        transpose(large_right),
    ])
    small = unary("relu", conv("x0", sw, sb))
    large = unary("relu", conv("x0", lw, lb))
    rotate = sslice(small, 4, 5)
    candidates = node("concat", [sslice(small, 0, 4), large, rotate])

    def combine_candidate_channels(indices):
        weights = np.zeros((1, 9, 1, 1), dtype=np.float32)
        weights[0, indices, 0, 0] = 1.0
        return binary("min2", conv(candidates, weights), one)

    candidate = combine_candidate_channels(list(range(9)))
    horizontal = combine_candidate_channels([0, 1, 4, 5])
    vertical = combine_candidate_channels([2, 3, 6, 7])
    grow = combine_candidate_channels([1, 3, 5, 7])

    # Determine which arm colours occur next to uncovered marker centers.
    neighbor_w = np.zeros((16, 16, 3, 3), dtype=np.float32)
    usable_colours = [0, 1, 6, 7, 8, 9, 10, 11, 12, 14]
    for colour in usable_colours:
        neighbor_w[colour, colour, 0, 1] = 1.0
        neighbor_w[colour, colour, 1, 0] = 1.0
        neighbor_w[colour, colour, 1, 2] = 1.0
        neighbor_w[colour, colour, 2, 1] = 1.0
    neighbor_colours = conv("x0", neighbor_w)
    marker_neighbours = binary("mul", neighbor_colours, markers)
    marker_colour_raw = unary("gmax", marker_neighbours)
    marker_colour_positive = unary("vrelu", marker_colour_raw)
    marker_colour_above_one = unary(
        "vrelu",
        dense(
            marker_colour_raw,
            np.eye(16, dtype=np.float32),
            -np.ones(16, dtype=np.float32),
        ),
    )
    marker_colours = node(
        "vadd",
        [
            marker_colour_positive,
            dense(
                marker_colour_above_one,
                -np.eye(16, dtype=np.float32),
            ),
        ],
    )

    # Read each button's colour from its glyph. Horizontal and vertical
    # controls use different offsets; rotate controls use their centers.
    top_w = np.zeros((16, 16, 3, 3), dtype=np.float32)
    left_w = np.zeros((16, 16, 3, 3), dtype=np.float32)
    center_w = np.zeros((16, 16, 1, 1), dtype=np.float32)
    for colour in usable_colours:
        top_w[colour, colour, 0, 1] = 1.0
        left_w[colour, colour, 1, 0] = 1.0
        center_w[colour, colour, 0, 0] = 1.0

    top_colour = binary("mul", conv("x0", top_w), marker_colours)
    left_colour = binary("mul", conv("x0", left_w), marker_colours)
    center_colour = binary("mul", conv("x0", center_w), marker_colours)
    sum_colours = np.ones((1, 16, 1, 1), dtype=np.float32)
    top_match = binary("min2", conv(top_colour, sum_colours), one)
    left_match = binary("min2", conv(left_colour, sum_colours), one)
    center_match = binary("min2", conv(center_colour, sum_colours), one)

    active_h = binary("mul", horizontal, top_match)
    active_v = binary("mul", vertical, left_match)
    active_r = binary("mul", rotate, center_match)
    active = binary(
        "min2",
        binary("add", binary("add", active_h, active_v), active_r),
        one,
    )
    passive = binary("mul", candidate, binary("sub", one, active))

    # Unique row-major rank for control selection.
    rank_w = np.zeros((1, 3, 1, 1), dtype=np.float32)
    rank_w[0, 0, 0, 0] = 4032.0
    rank_w[0, 1, 0, 0] = 63.0
    rank = conv("coords", rank_w, np.array([1.0], dtype=np.float32))

    # Passive controls are ordered with shrinking halves first, then
    # growing halves, then rotations.
    passive_rank = binary("add", rank, saffine(grow, 4096.0, 0.0))
    passive_rank = binary(
        "add", passive_rank, saffine(rotate, 8192.0, 0.0)
    )

    # Area and first moments form a compact image-change fingerprint.
    fingerprint_colours = node("concat", [
        sslice("x0", 1, 2),
        sslice("x0", 7, 13),
        sslice("x0", 14, 15),
    ])
    row_coord = saffine(sslice("coords", 0, 1), 63.0, 0.0)
    col_coord = saffine(sslice("coords", 1, 2), 63.0, 0.0)
    fingerprint_planes = node("concat", [
        fingerprint_colours,
        binary("mul", fingerprint_colours, row_coord),
        binary("mul", fingerprint_colours, col_coord),
    ])
    fingerprint = dense(
        unary("gmean", fingerprint_planes),
        np.eye(24, dtype=np.float32) * 4096.0,
    )

    # Memory layout:
    # last click, prior reward, undo phase, active scan cursor,
    # passive scan cursor, whether the last click was a normal passive
    # click, and the 24-element previous fingerprint.
    last = vslice("mem", 0, 1)
    prior_reward = vslice("mem", 1, 2)
    phase = vslice("mem", 2, 3)
    active_cursor = vslice("mem", 3, 4)
    passive_cursor = vslice("mem", 4, 5)
    last_passive = vslice("mem", 5, 6)
    prior_fingerprint = vslice("mem", 6, 30)

    fp_pair = node("vconcat", [fingerprint, prior_fingerprint])
    fp_w = np.zeros((48, 48), dtype=np.float32)
    for i in range(24):
        fp_w[i, i] = 1.0
        fp_w[i, 24 + i] = -1.0
        fp_w[24 + i, i] = -1.0
        fp_w[24 + i, 24 + i] = 1.0
    fp_difference = unary("vrelu", dense(fp_pair, fp_w))
    fp_total = dense(
        fp_difference, np.ones((1, 48), dtype=np.float32)
    )
    changed = clip01(vaffine([fp_total], [2.0], -1.0))

    phase_active = clip01(phase)
    phase_many = clip01(vaffine([phase], [1.0], -1.0))
    phase_none = vaffine([phase_active], [-1.0], 1.0)

    improvement = vaffine([reward, prior_reward], [1.0, -1.0])
    deterioration = vaffine([prior_reward, reward], [1.0, -1.0])
    better = vand(clip01(improvement), phase_none)
    worse = vand(clip01(deterioration), phase_none)

    equal_last = spatial_equal(rank, last, one)

    # Find the opposite half of the last resize button.
    offsets = np.array(
        [6, -6, 384, -384, 6, -6, 384, -384, 0],
        dtype=np.float32,
    )
    offset_vector = dense(one_v, offsets.reshape(9, 1))
    partner_rank = binary("add", rank, offset_vector)
    partner_delta = unary("abs", binary("sub", partner_rank, last))
    eq_w = -2.0 * np.eye(9, dtype=np.float32).reshape(9, 9, 1, 1)
    eq_b = np.full(9, 2.0, dtype=np.float32)
    partner_equal = binary(
        "min2", unary("relu", conv(partner_delta, eq_w, eq_b)), one
    )
    partner_matches = binary("mul", candidates, partner_equal)
    inverse = binary(
        "min2",
        conv(partner_matches, np.ones((1, 9, 1, 1), dtype=np.float32)),
        one,
    )
    inverse_exists = unary("gmax", inverse)
    was_rotation = unary("gmax", binary("mul", rotate, equal_last))

    undo = vand(worse, inverse_exists)
    undo_phase = vand(vaffine([was_rotation], [2.0], 1.0), undo)
    continuing_phase = vand(vaffine([phase], [1.0], -1.0), phase_many)
    next_phase = binary("vadd", undo_phase, continuing_phase)

    repeat = vor(better, phase_many)
    forced = binary(
        "add",
        binary("mul", equal_last, repeat),
        binary("mul", inverse, undo),
    )
    forced_flag = vor(repeat, undo)
    normal_flag = vaffine([forced_flag], [-1.0], 1.0)

    # If a passive adjustment changed the board, leave its cursor just
    # before that control so it can be tried again after marker controls.
    rewind_passive = vand(last_passive, changed, phase_none)
    effective_passive_cursor = vaffine(
        [passive_cursor, rewind_passive], [1.0, -1.0]
    )

    active_after = unary("relu", binary("sub", rank, active_cursor))
    active_after = binary("min2", active_after, one)
    active_forward = binary("mul", active, active_after)

    passive_after = unary(
        "relu", binary("sub", passive_rank, effective_passive_cursor)
    )
    passive_after = binary("min2", passive_after, one)

    # Active forward candidates outrank all passive candidates. Passive
    # candidates wrap around after the end of their scan.
    active_priority = binary(
        "mul", active_forward, saffine(rank, -1.0, 84096.0)
    )
    passive_priority = binary(
        "mul",
        passive,
        binary(
            "add",
            saffine(passive_rank, -1.0, 13000.0),
            saffine(passive_after, 20000.0, 0.0),
        ),
    )
    scores = binary("add", active_priority, passive_priority)
    scores = binary("add", scores, saffine(forced, 200000.0, 0.0))
    node("add", [scores, one], name="click")

    best = unary("gmax", scores)
    selected = spatial_equal(scores, best, one)
    selected = binary("mul", selected, candidate)
    selected_rank = unary("gmax", binary("mul", selected, rank))
    selected_passive_rank = unary(
        "gmax", binary("mul", selected, passive_rank)
    )
    selected_active = unary("gmax", binary("mul", selected, active))
    selected_passive = unary("gmax", binary("mul", selected, passive))

    # Forced repeats and undo clicks do not advance either scan cursor.
    next_active_cursor = binary(
        "vadd",
        vand(forced_flag, active_cursor),
        vand(normal_flag, selected_active, selected_rank),
    )
    update_passive = vand(normal_flag, selected_passive)
    keep_passive = vaffine([update_passive], [-1.0], 1.0)
    next_passive_cursor = binary(
        "vadd",
        vand(keep_passive, effective_passive_cursor),
        vand(update_passive, selected_passive_rank),
    )
    next_last_passive = update_passive

    node(
        "vconcat",
        [
            selected_rank,
            reward,
            next_phase,
            next_active_cursor,
            next_passive_cursor,
            next_last_passive,
            fingerprint,
        ],
        name="mem_next",
    )

    logits_w = np.zeros((8, 1), dtype=np.float32)
    logits_b = np.zeros(8, dtype=np.float32)
    logits_b[6] = 1.0
    dense(one_v, logits_w, logits_b, name="logits")

    return {
        "graph": graph,
        "memory_size": 30,
        "params": params,
    }
