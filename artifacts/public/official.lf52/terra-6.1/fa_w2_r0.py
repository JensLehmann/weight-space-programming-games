import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]

    def nd(op, xs=None, name=None, **fields):
        if name is None:
            name = "n%d" % counter[0]
            counter[0] += 1
        x = {"id": name, "op": op}
        if xs is not None:
            x["in"] = list(xs)
        x.update(fields)
        graph.append(x)
        return name

    def par(a):
        name = "p%d" % len(params)
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def conv(x, w, b=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return nd("conv", [x], **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return nd("dense", [x], name=name, **fields)

    def sl(x, a, b):
        return nd("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return nd("vslice", [x], start=a, stop=b)

    def cat(*xs):
        return nd("concat", xs)

    def vc(*xs):
        return nd("vconcat", xs)

    def op2(op, a, b):
        return nd(op, [a, b])

    def add(a, b):
        return op2("add", a, b)

    def sub(a, b):
        return op2("sub", a, b)

    def mul(a, b):
        return op2("mul", a, b)

    def mx(a, b):
        return op2("max2", a, b)

    def mn(a, b):
        return op2("min2", a, b)

    def va(a, b):
        return op2("vadd", a, b)

    def vm(a, b):
        return op2("vmul", a, b)

    def relu(x):
        return nd("relu", [x])

    def vrelu(x):
        return nd("vrelu", [x])

    def gm(x):
        return nd("gmax", [x])

    def ss(x, n, factor, bias=0.0):
        w = np.zeros((n, n, 1, 1), np.float32)
        for i in range(n):
            w[i, i, 0, 0] = factor
        return conv(x, w, np.full(n, bias, np.float32))

    def sv(x, n, factor, bias=0.0):
        return dense(
            x,
            np.eye(n, dtype=np.float32) * factor,
            np.full(n, bias, np.float32),
        )

    def inv(x):
        return dense(x, [[-1.0]], [1.0])

    def clamp(x, n):
        a = vrelu(x)
        b = vrelu(sv(x, n, 1.0, -1.0))
        return va(a, sv(b, n, -1.0))

    def sumc(x, n):
        return conv(x, np.ones((1, n, 1, 1), np.float32))

    def max4(x):
        a, b, c, d = [sl(x, i, i + 1) for i in range(4)]
        return mx(mx(a, b), mx(c, d))

    def detect(x, offsets, threshold):
        w = np.zeros((1, 1, 7, 7), np.float32)
        for dy, dx in offsets:
            w[0, 0, dy + 3, dx + 3] = 1.0
        return ss(relu(conv(x, w, [-threshold])), 1, 2.0)

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    first_w = np.zeros((4, 1, 7, 7), np.float32)
    next_w = np.zeros((4, 4, 7, 7), np.float32)
    reverse_w = np.zeros((4, 4, 7, 7), np.float32)
    for d, (dy, dx) in enumerate(directions):
        first_w[d, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
        next_w[d, d, 3 + 3 * dy, 3 + 3 * dx] = 1.0
        reverse_w[d, d, 3 - 3 * dy, 3 - 3 * dx] = 1.0

    def shift6(x):
        return conv(conv(x, first_w), next_w)

    def shift12(x):
        return conv(conv(shift6(x), next_w), next_w)

    def scatter12(x):
        for _ in range(4):
            x = conv(x, reverse_w)
        return x

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    rowcol = mul(row, col)

    gp = sl("x0", 14, 15)
    rp = sl("x0", 8, 9)
    bp = sl("x0", 9, 10)
    pp = sl("x0", 15, 16)
    wp = sl("x0", 1, 2)
    orange = sl("x0", 12, 13)
    yellow = sl("x0", 11, 12)
    gray = sl("x0", 2, 3)
    black = sl("x0", 5, 6)

    peg_shape = [
        (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1), (0, 2),
        (1, -1), (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
    ]
    obstacle_shape = [
        (-2, -1), (-2, 0), (-2, 1), (-2, 2),
        (-1, -1), (-1, 2),
        (0, -1), (0, 2),
        (1, -1), (1, 0), (1, 1), (1, 2),
    ]
    floor_shape = [(y, x) for y in range(-1, 3) for x in range(-1, 3)]
    marker_shape = [
        (-1, 0), (-1, 1),
        (0, -1), (0, 2),
        (1, -1), (1, 2),
        (2, 0), (2, 1),
    ]

    green = detect(gp, peg_shape, 11.5)
    red = detect(rp, peg_shape, 11.5)
    blue = detect(bp, peg_shape, 11.5)
    peg = mx(mx(green, red), blue)
    obstacle = detect(pp, obstacle_shape, 11.5)
    occupied = mx(peg, obstacle)
    empty = mx(
        detect(wp, floor_shape, 15.5),
        detect(orange, floor_shape, 15.5),
    )
    marker = detect(gray, marker_shape, 7.5)
    marker_flag = gm(marker)
    platform = detect(
        yellow, [(-2, -2), (-2, 3), (3, -2), (3, 3)], 3.5
    )
    carrier = mul(platform, peg)
    carrier_flag = gm(carrier)

    weighted = conv(
        cat(green, red, blue),
        np.array([1.0, 1.17, 1.31], np.float32).reshape(1, 3, 1, 1),
    )
    sig_maps = cat(
        weighted,
        mul(weighted, row),
        mul(weighted, col),
        mul(weighted, rowcol),
    )
    signature = sv(nd("gmean", [sig_maps]), 4, 4096.0)

    # Jump records: board signature, source row/column and direction.
    J = 8
    jump_sig = vs("mem", 0, 4 * J)
    jump_rows = vs("mem", 4 * J, 5 * J)
    jump_cols = vs("mem", 5 * J, 6 * J)
    jump_dirs = vs("mem", 6 * J, 10 * J)
    old_target = vs("mem", 10 * J, 10 * J + 2)
    pending = vs("mem", 10 * J + 2, 10 * J + 3)
    old_undo_count = vs("mem", 10 * J + 3, 10 * J + 4)

    undo_count = vm(
        dense(old_undo_count, [[1.0]], [1.0]),
        vs("aprev", 7, 8),
    )
    clear_jump = clamp(dense(undo_count, [[1.0]], [-5.0]), 1)

    compare = vc(signature, jump_sig)
    w = np.zeros((8 * J, 4 + 4 * J), np.float32)
    for i in range(J):
        for k in range(4):
            q = 4 * i + k
            w[2 * q, k] = 1.0
            w[2 * q, 4 + q] = -1.0
            w[2 * q + 1, k] = -1.0
            w[2 * q + 1, 4 + q] = 1.0
    differences = vrelu(dense(compare, w))
    w = np.zeros((J, 8 * J), np.float32)
    for i in range(J):
        w[i, 8 * i:8 * i + 8] = 1.0
    sig_distance = dense(differences, w)
    matches = vrelu(sv(sig_distance, J, -1000.0, 1.0))
    matches = vm(matches, inv(clear_jump))

    source_distance = add(
        nd("abs", [sub(row, jump_rows)]),
        nd("abs", [sub(col, jump_cols)]),
    )
    near = ss(relu(ss(source_distance, J, -63.0, 0.5)), J, 2.0)
    near = mul(near, matches)
    taboo_parts = []
    for d in range(4):
        taboo_parts.append(
            sumc(mul(near, vs(jump_dirs, d * J, (d + 1) * J)), J)
        )
    taboo = cat(*taboo_parts)

    # Immediately reversing a jump over a non-removable obstacle cannot
    # make progress. Gate this rule on the previous click.
    target_row = vs(old_target, 0, 1)
    target_col = vs(old_target, 1, 2)
    target_distance = add(
        nd("abs", [sub(row, target_row)]),
        nd("abs", [sub(col, target_col)]),
    )
    at_target = ss(relu(ss(target_distance, 1, -63.0, 0.5)), 1, 2.0)
    prev_click = vs("aprev", 6, 7)
    reverse_parts = []
    for d in range(4):
        opposite = (d + 2) % 4
        last_dir = vs(jump_dirs, opposite * J, opposite * J + 1)
        reverse_parts.append(mul(mul(at_target, last_dir), prev_click))
    taboo = mx(taboo, cat(*reverse_parts))

    middle = shift6(occupied)
    landing = shift12(empty)
    raw_jump = relu(ss(add(add(peg, middle), landing), 4, 1.0, -2.0))
    legal = relu(sub(raw_jump, taboo))
    legal_flag = clamp(dense(gm(legal), np.ones((1, 4), np.float32)), 1)

    captures = mx(mn(green, shift6(green)), mn(red, shift6(red)))
    captures = mul(captures, legal)
    degree = sumc(middle, 4)
    degree_landing = shift12(degree)
    opportunity = relu(ss(add(middle, landing), 4, 1.0, -1.0))
    opportunity_landing = shift12(sumc(opportunity, 4))

    features = cat(
        legal,
        captures,
        mul(degree_landing, legal),
        mul(degree, legal),
        mul(opportunity_landing, legal),
    )
    w = np.zeros((4, 20, 1, 1), np.float32)
    for d in range(4):
        w[d, d, 0, 0] = 15.0
        w[d, 4 + d, 0, 0] = 2.5
        w[d, 8 + d, 0, 0] = 0.20
        w[d, 12 + d, 0, 0] = -0.16
        w[d, 16 + d, 0, 0] = 0.30
    scores = conv(
        features,
        w,
        np.array([-10.0, -10.00001, -10.00002, -10.00003], np.float32),
    )
    tie = conv(
        cat(row, col),
        np.array([-0.25, -0.0035], np.float32).reshape(1, 2, 1, 1),
    )
    scores = add(scores, tie)
    best_map = max4(scores)
    best = gm(best_map)
    peaks = relu(ss(sub(scores, best), 4, 1.0, 0.000004))
    peaks = mn(ss(peaks, 4, 10000000.0), one)
    chosen_dirs = gm(peaks)
    chosen_row = dense(gm(mul(peaks, row)), np.ones((1, 4), np.float32))
    chosen_col = dense(gm(mul(peaks, col)), np.ones((1, 4), np.float32))
    chosen_info = vc(chosen_row, chosen_col, chosen_dirs)

    w = np.zeros((2, 6), np.float32)
    w[0, 0] = 1.0
    w[1, 1] = 1.0
    for d, (dy, dx) in enumerate(directions):
        w[0, 2 + d] = 12.0 * dy / 63.0
        w[1, 2 + d] = 12.0 * dx / 63.0
    new_target = dense(chosen_info, w)

    marker_score = add(
        ss(marker, 1, 15.0, -10.0),
        ss(target_distance, 1, -1.5),
    )
    normal_click = add(
        mul(best_map, inv(marker_flag)),
        mul(marker_score, marker_flag),
    )

    key_run = detect(pp, [(0, dx) for dx in range(-3, 4)], 6.5)
    key_region = mul(
        mn(relu(ss(row, 1, 63.0, -49.0)), one),
        mn(relu(ss(col, 1, -63.0, 16.0)), one),
    )
    key_flag = gm(mul(key_run, key_region))
    constant = dense(marker_flag, [[0.0]], [1.0])
    key_distance = add(
        nd("abs", [sub(row, sv(constant, 1, 52.0 / 63.0))]),
        nd("abs", [sub(col, sv(constant, 1, 4.0 / 63.0))]),
    )
    key_point = ss(relu(ss(key_distance, 1, -63.0, 0.5)), 1, 2.0)
    click_map = add(
        mul(normal_click, inv(key_flag)),
        mul(key_point, key_flag),
    )
    nd("add", [click_map, ss(one, 1, 0.0)], name="click")

    first_param = par(first_w)
    next_param = par(next_w)

    graph.append({
        "id": "lattice",
        "op": "loop",
        "T": 11,
        "state": [{"id": "ls", "init": platform}],
        "body": [
            {"id": "la", "op": "conv", "in": ["ls"], "w": first_param},
            {"id": "lb", "op": "conv", "in": ["la"], "w": next_param},
            {"id": "l0", "op": "slice", "in": ["lb"], "start": 0, "stop": 1},
            {"id": "l1", "op": "slice", "in": ["lb"], "start": 1, "stop": 2},
            {"id": "l2", "op": "slice", "in": ["lb"], "start": 2, "stop": 3},
            {"id": "l3", "op": "slice", "in": ["lb"], "start": 3, "stop": 4},
            {"id": "lm0", "op": "max2", "in": ["l0", "l1"]},
            {"id": "lm1", "op": "max2", "in": ["l2", "l3"]},
            {"id": "lm2", "op": "max2", "in": ["lm0", "lm1"]},
            {"id": "ln", "op": "max2", "in": ["ls", "lm2"]},
        ],
        "next": {"ls": "ln"},
    })
    rail = detect(black, [(0, 0), (0, 1), (1, 0), (1, 1)], 3.5)
    track = mx(mul(rail, "lattice.ls"), platform)

    pickup = relu(ss(add(add(peg, middle), shift12(track)), 4, 1.0, -2.0))
    pickup = relu(sub(pickup, taboo))
    pickup_goal = mul(max4(scatter12(pickup)), track)
    drop = relu(ss(add(middle, landing), 4, 1.0, -1.0))
    drop = relu(sub(drop, taboo))
    drop_goal = mul(max4(drop), track)
    goals = mn(
        add(mul(pickup_goal, inv(carrier_flag)),
            mul(drop_goal, carrier_flag)),
        one,
    )

    penalty_w = par(np.ones((1, 1, 1, 1), np.float32))
    penalty_b = par(np.array([-0.035], np.float32))
    graph.append({
        "id": "flood",
        "op": "loop",
        "T": 20,
        "state": [{"id": "fs", "init": goals}],
        "body": [
            {"id": "fa", "op": "conv", "in": ["fs"], "w": first_param},
            {"id": "fb", "op": "conv", "in": ["fa"], "w": next_param},
            {"id": "f0", "op": "slice", "in": ["fb"], "start": 0, "stop": 1},
            {"id": "f1", "op": "slice", "in": ["fb"], "start": 1, "stop": 2},
            {"id": "f2", "op": "slice", "in": ["fb"], "start": 2, "stop": 3},
            {"id": "f3", "op": "slice", "in": ["fb"], "start": 3, "stop": 4},
            {"id": "fm0", "op": "max2", "in": ["f0", "f1"]},
            {"id": "fm1", "op": "max2", "in": ["f2", "f3"]},
            {"id": "fm2", "op": "max2", "in": ["fm0", "fm1"]},
            {"id": "fp", "op": "conv", "in": ["fm2"],
             "w": penalty_w, "b": penalty_b},
            {"id": "fr", "op": "relu", "in": ["fp"]},
            {"id": "ft", "op": "mul", "in": ["fr", track]},
            {"id": "fn", "op": "max2", "in": ["fs", "ft"]},
        ],
        "next": {"fs": "fn"},
    })

    active_platform = add(
        mul(platform, inv(carrier_flag)),
        mul(carrier, carrier_flag),
    )
    platform_moves = mul(active_platform, shift6(track))
    move_available = gm(platform_moves)
    move_value = gm(add(
        platform_moves,
        mul(platform_moves, ss(shift6("flood.fs"), 4, 10.0)),
    ))
    move_any = clamp(dense(move_available, np.ones((1, 4), np.float32)), 1)

    # Recent platform-state/action records.
    R = 12
    base = 10 * J + 4
    nav_sig_mem = vs("mem", base, base + 6 * R)
    nav_dir_mem = vs("mem", base + 6 * R, base + 10 * R)
    previous_nav_sig = vs("mem", base + 10 * R, base + 10 * R + 6)
    nav_maps = cat(
        platform,
        mul(platform, row),
        mul(platform, col),
        mul(platform, rowcol),
        weighted,
        mul(weighted, rowcol),
    )
    nav_signature = sv(nd("gmean", [nav_maps]), 6, 4096.0)

    nav_compare = vc(nav_signature, nav_sig_mem)
    w = np.zeros((12 * R, 6 + 6 * R), np.float32)
    for i in range(R):
        for k in range(6):
            q = 6 * i + k
            w[2 * q, k] = 1.0
            w[2 * q, 6 + q] = -1.0
            w[2 * q + 1, k] = -1.0
            w[2 * q + 1, 6 + q] = 1.0
    nav_diff = vrelu(dense(nav_compare, w))
    w = np.zeros((R, 12 * R), np.float32)
    for i in range(R):
        w[i, 12 * i:12 * i + 12] = 1.0
    nav_distance = dense(nav_diff, w)
    nav_match = vrelu(sv(nav_distance, R, -1000.0, 1.0))
    repeats = []
    for d in range(4):
        selected = vm(nav_match, vs(nav_dir_mem, d * R, (d + 1) * R))
        repeats.append(dense(selected, np.ones((1, R), np.float32)))
    repeat_flags = clamp(vc(*repeats), 4)

    # A jump or marker has absolute priority over rail navigation.
    action_features = vc(
        marker_flag, legal_flag, pending, key_flag,
        move_any, move_available, move_value, repeat_flags, "aprev"
    )
    w = np.zeros((8, 25), np.float32)
    b = np.full(8, -100.0, np.float32)
    action_for_direction = [1, 4, 2, 3]
    opposite_action = [2, 3, 1, 4]
    bias = [0.02, 0.12, 0.04, -0.02]
    for d, action in enumerate(action_for_direction):
        b[action] = -202.0 + bias[d]
        w[action, 0] = -100.0
        w[action, 1] = -100.0
        w[action, 2] = -100.0
        w[action, 3] = -100.0
        w[action, 5 + d] = 200.0
        w[action, 9 + d] = 3.0
        w[action, 13 + d] = -40.0
        w[action, 17 + opposite_action[d]] = -0.6
    b[6] = -2.0
    w[6, 0] = 6.0
    w[6, 1] = 6.0
    w[6, 2] = -100.0
    w[6, 3] = 40.0
    b[7] = 1.0
    w[7, 0] = -5.0
    w[7, 1] = -5.0
    w[7, 2] = 100.0
    w[7, 3] = -100.0
    w[7, 4] = -50.0
    dense(action_features, w, b, name="logits")

    write_jump = vm(legal_flag, inv(marker_flag))
    write_jump = vm(write_jump, inv(pending))
    write_jump = vm(write_jump, inv(key_flag))
    new_jump_sig = vc(signature, vs(jump_sig, 0, 4 * (J - 1)))
    new_jump_rows = vc(chosen_row, vs(jump_rows, 0, J - 1))
    new_jump_cols = vc(chosen_col, vs(jump_cols, 0, J - 1))
    new_jump_dirs = []
    for d in range(4):
        new_jump_dirs.append(
            vc(vs(chosen_dirs, d, d + 1),
               vs(jump_dirs, d * J, d * J + J - 1))
        )
    new_jump_records = vc(
        new_jump_sig, new_jump_rows, new_jump_cols, *new_jump_dirs
    )
    old_jump_records = vs("mem", 0, 10 * J)
    jump_records = va(
        vm(new_jump_records, write_jump),
        vm(old_jump_records, inv(write_jump)),
    )
    jump_records = vm(jump_records, inv(clear_jump))
    target = va(vm(new_target, write_jump), vm(old_target, inv(write_jump)))

    next_pending = vm(inv(marker_flag), inv(legal_flag))
    next_pending = vm(next_pending, inv(move_any))
    next_pending = vm(next_pending, inv(pending))
    next_pending = vm(next_pending, inv(key_flag))

    previous_dirs = vc(
        vs("aprev", 1, 2),
        vs("aprev", 4, 5),
        vs("aprev", 2, 3),
        vs("aprev", 3, 4),
    )
    write_nav = clamp(dense(previous_dirs, np.ones((1, 4), np.float32)), 1)
    new_nav_sig = vc(previous_nav_sig, vs(nav_sig_mem, 0, 6 * (R - 1)))
    new_nav_dirs = []
    for d in range(4):
        new_nav_dirs.append(
            vc(vs(previous_dirs, d, d + 1),
               vs(nav_dir_mem, d * R, d * R + R - 1))
        )
    new_nav_records = vc(new_nav_sig, *new_nav_dirs)
    old_nav_records = vs("mem", base, base + 10 * R)
    nav_records = va(
        vm(new_nav_records, write_nav),
        vm(old_nav_records, inv(write_nav)),
    )

    nd(
        "vconcat",
        [jump_records, target, next_pending, undo_count,
         nav_records, nav_signature],
        name="mem_next",
    )
    return {
        "graph": graph,
        "memory_size": base + 10 * R + 6,
        "params": params,
    }
