import numpy as np


def build():
    graph, params = [], {}
    serial = [0]

    def p(name, value):
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def n(name, op, inputs, **fields):
        graph.append({"id": name, "op": op, "in": inputs, **fields})
        return name

    def cv(name, x, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return n(name, "conv", [x], **fields)

    def dn(name, x, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return n(name, "dense", [x], **fields)

    def sl(name, x, a, b):
        return n(name, "slice", [x], start=a, stop=b)

    def vs(name, x, a, b):
        return n(name, "vslice", [x], start=a, stop=b)

    def prop(source, weight, op, steps):
        k = serial[0]
        serial[0] += 1
        lid, state = "L%d" % k, "S%d" % k
        shifted, updated = "D%d" % k, "U%d" % k
        graph.append({
            "id": lid,
            "op": "loop",
            "T": steps,
            "state": [{"id": state, "init": source}],
            "body": [
                {"id": shifted, "op": "conv",
                 "in": [state], "w": weight},
                {"id": updated, "op": op,
                 "in": [source, shifted]},
            ],
            "next": {state: updated},
        })
        return lid + "." + state

    p("neg1", [[-1.0]])
    p("bias1", [1.0])
    p("neg9", -np.eye(9, dtype=np.float32))
    p("bias9", np.ones(9, dtype=np.float32))

    def clip_vector(name, x, width=1):
        w = "neg1" if width == 1 else "neg9"
        b = "bias1" if width == 1 else "bias9"
        t = dn(name + "_inv", x, w, b)
        t = n(name + "_relu", "vrelu", [t])
        return dn(name, t, w, b)

    def vector_max(name, old, current, width):
        key = "vector_max_w_%d" % width
        if key not in params:
            w = np.zeros((width, 2 * width), np.float32)
            for i in range(width):
                w[i, i] = 1.0
                w[i, width + i] = -1.0
            p(key, w)
        pair = n(name + "_pair", "vconcat", [current, old])
        diff = dn(name + "_diff", pair, key)
        positive = n(name + "_positive", "vrelu", [diff])
        return n(name, "vadd", [old, positive])

    one = sl("one", "coords", 2, 3)
    coord_pair = sl("coord_pair", "coords", 0, 2)

    # Legal play band and goal-icon area.
    mw = np.zeros((4, 3, 1, 1), np.float32)
    mw[0, 0, 0, 0] = 63
    mw[1, 0, 0, 0] = -63
    mw[2, 0, 0, 0] = -63
    mw[3, 1, 0, 0] = 63
    p("mask_w", mw)
    p("mask_b", [-9, 63, 10, -23])
    mask = cv("mask_raw", "coords", "mask_w", "mask_b")
    mask = n("mask_positive", "relu", [mask])
    mask = n("masks", "min2", [mask, one])
    lo = sl("lo", mask, 0, 1)
    hi = sl("hi", mask, 1, 2)
    header = sl("header", mask, 2, 3)
    icon_x = sl("icon_x", mask, 3, 4)
    play = n("play", "min2", [lo, hi])
    icon_area = n("icon_area", "min2", [header, icon_x])

    colors = [10, 6, 15, 11, 12, 8, 9, 7, 14]
    sizes = [1, 2, 3, 4, 5, 7, 8, 9, 10]
    patterns = [
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0]],
        [[1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 1, 1, 1, 0]],
    ]

    # Fruit, ring, and enemy-glyph templates.
    w1 = np.zeros((29, 16, 7, 7), np.float32)
    for i, (color, size) in enumerate(zip(colors, sizes)):
        if size <= 7:
            start = -(size // 2)
            for dy in range(start, start + size):
                for dx in range(start, start + size):
                    w1[i, color, dy + 3, dx + 3] = 1
        else:
            w1[i, color, :, :] = 1

    for k in range(3):
        for j, (sy, sx) in enumerate(
            [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        ):
            w1[9 + 4 * k + j, colors[6 + k],
               3 + 3 * sy, 3 + 3 * sx] = 1

    for d in range(-2, 3):
        w1[21, 9, 0, d + 3] = 1
        w1[22, 9, 6, d + 3] = 1
        w1[23, 9, d + 3, 0] = 1
        w1[24, 9, d + 3, 6] = 1
    for dy in (-3, 3):
        for dx in (-3, 3):
            w1[25, 9, dy + 3, dx + 3] = 1

    counts = []
    for k, pattern in enumerate(patterns):
        count = 0
        for y, row in enumerate(pattern):
            for x, bit in enumerate(row):
                w1[26 + k, [7, 14, 13][k], y + 1, x + 1] = (
                    1 if bit else -1
                )
                count += bit
        counts.append(count)
    p("template1_w", w1)
    t1 = cv("template1", "x0", "template1_w")

    w2 = np.zeros((13, 29, 7, 7), np.float32)
    b2 = np.zeros(13, np.float32)
    for i, size in enumerate(sizes):
        w2[i, i, 3, 3] = 2
        if size <= 7:
            b2[i] = -2 * (size * size - 0.5)
        else:
            a = -(size // 2)
            b = a + size - 1
            for j, (dy, dx) in enumerate(
                [(a, a), (a, b), (b, a), (b, b)]
            ):
                sy = -1 if dy < 0 else 1
                sx = -1 if dx < 0 else 1
                w2[i, 9 + 4 * (i - 6) + j,
                   dy - 3 * sy + 3, dx - 3 * sx + 3] = 2
            b2[i] = -2 * 52.5

    w2[9, 21, 2, 3] = 2
    w2[9, 22, 4, 3] = 2
    w2[9, 23, 3, 2] = 2
    w2[9, 24, 3, 4] = 2
    w2[9, 25, 3, 3] = 2
    b2[9] = -2 * 21.5
    for k, count in enumerate(counts):
        w2[10 + k, 26 + k, 3, 3] = 2
        b2[10 + k] = -2 * (count - 0.5)

    p("template2_w", w2)
    p("template2_b", b2)
    t2 = cv("template2", t1, "template2_w", "template2_b")
    t2 = n("template_positive", "relu", [t2])
    detected = n("detected", "min2", [t2, one])
    fruit_all = sl("fruit_all", detected, 0, 9)
    goal_all = sl("goal_all", detected, 9, 10)
    enemy_all = sl("enemy_all", detected, 10, 13)

    fruit_icons = n("fruit_icons", "mul", [fruit_all, icon_area])
    fruit_wanted = n("fruit_wanted", "gmax", [fruit_icons])
    enemy_icons = n("enemy_icons", "mul", [enemy_all, icon_area])
    enemy_wanted = n("enemy_wanted", "gmax", [enemy_icons])
    fruit = n("fruit", "mul", [fruit_all, play])
    enemy = n("enemy", "mul", [enemy_all, play])
    goal_seen = n("goal_seen", "mul", [goal_all, play])

    # Partial glyph matches are used only for danger estimation.
    loose_source = sl("loose_source", t1, 26, 29)
    iw3 = np.zeros((3, 3, 1, 1), np.float32)
    for i in range(3):
        iw3[i, i, 0, 0] = 1
    p("identity3", iw3)
    p("loose_b", [-2.5, -2.5, -2.5])
    loose = cv("loose_raw", loose_source, "identity3", "loose_b")
    loose = n("loose_positive", "relu", [loose])
    loose = n("loose_cap", "min2", [loose, one])
    loose = n("loose_play", "mul", [loose, play])
    p("sum3", np.ones((1, 3, 1, 1)))
    p("sum9", np.ones((1, 9, 1, 1)))
    loose_sum = cv("loose_sum", loose, "sum3")
    enemy_seed = n("enemy_seed", "min2", [loose_sum, one])

    # Store one goal coordinate and a presence flag per quadrant.
    qw = np.zeros((4, 3, 1, 1), np.float32)
    qw[0, 0, 0, 0] = -63
    qw[1, 0, 0, 0] = 63
    qw[2, 1, 0, 0] = -63
    qw[3, 1, 0, 0] = 63
    p("quadrant_w", qw)
    p("quadrant_b", [32, -31, 32, -31])
    q = cv("quadrant_raw", "coords", "quadrant_w", "quadrant_b")
    q = n("quadrant_positive", "relu", [q])
    q = n("quadrant_axes", "min2", [q, one])
    qtop = sl("qtop", q, 0, 1)
    qbottom = sl("qbottom", q, 1, 2)
    qleft = sl("qleft", q, 2, 3)
    qright = sl("qright", q, 3, 4)
    quadrants = [
        n("quad0", "min2", [qtop, qleft]),
        n("quad1", "min2", [qtop, qright]),
        n("quad2", "min2", [qbottom, qleft]),
        n("quad3", "min2", [qbottom, qright]),
    ]
    observed_coords, observed_flags = [], []
    for i, quadrant in enumerate(quadrants):
        goal_q = n("goal_q%d" % i, "mul", [goal_seen, quadrant])
        goal_q_coords = n(
            "goal_qcoords%d" % i, "mul", [coord_pair, goal_q]
        )
        observed_coords.append(
            n("goal_qcoord_v%d" % i, "gmax", [goal_q_coords])
        )
        observed_flags.append(n("goal_qflag_v%d" % i, "gmax", [goal_q]))

    coords_now = n("coords_now", "vconcat", observed_coords)
    flags_now = n("flags_now", "vconcat", observed_flags)
    old_coords = vs("old_coords", "mem", 0, 8)
    old_flags = vs("old_flags", "mem", 8, 12)
    saved_coords = vector_max("saved_coords", old_coords, coords_now, 8)
    saved_flags = vector_max("saved_flags", old_flags, flags_now, 4)

    p("goal_point_w", np.array([[[[-63.0]]]], np.float32))
    p("goal_point_b", [1.0])
    points = []
    for i in range(4):
        stored_coord = vs("stored_coord%d" % i, saved_coords, 2 * i, 2 * i + 2)
        stored_flag = vs("stored_flag%d" % i, saved_flags, i, i + 1)
        delta = n("goal_delta%d" % i, "sub", [coord_pair, stored_coord])
        delta = n("goal_abs%d" % i, "abs", [delta])
        dy = sl("goal_dy%d" % i, delta, 0, 1)
        dx = sl("goal_dx%d" % i, delta, 1, 2)
        distance = n("goal_distance%d" % i, "max2", [dy, dx])
        point = cv(
            "goal_point_raw%d" % i, distance,
            "goal_point_w", "goal_point_b"
        )
        point = n("goal_point_pos%d" % i, "relu", [point])
        point = n("goal_point%d" % i, "mul", [point, stored_flag])
        points.append(point)
    goals01 = n("goals01", "max2", points[:2])
    goals23 = n("goals23", "max2", points[2:])
    goals = n("goals", "max2", [goals01, goals23])

    p("box7", np.ones((1, 1, 7, 7)))
    p("box3", np.ones((1, 1, 3, 3)))
    goal_box = cv("goal_box7", goals, "box7")
    goal_box = cv("goal_box9", goal_box, "box3")
    goal_box = n("goal_region", "min2", [goal_box, one])

    protected_fruit = n(
        "protected_fruit_type", "mul", [goal_box, fruit_wanted]
    )
    protected_fruit = n("protected_fruit", "mul", [protected_fruit, fruit])
    active_fruit = n("active_fruit", "sub", [fruit, protected_fruit])
    protected_enemy = n(
        "protected_enemy_type", "mul", [goal_box, enemy_wanted]
    )
    protected_enemy = n("protected_enemy", "mul", [protected_enemy, enemy])
    active_enemy = n("active_enemy", "sub", [enemy, protected_enemy])

    below = np.zeros((9, 9), np.float32)
    above = np.zeros((9, 9), np.float32)
    for i in range(9):
        for j in range(9):
            below[i, j] = float(i < j)
            above[i, j] = float(i > j)
    p("below_w", below)
    p("above_w", above)
    below_goal = dn("below_goal", fruit_wanted, "below_w")
    above_goal_raw = dn("above_goal_raw", fruit_wanted, "above_w")
    above_goal = clip_vector("above_goal", above_goal_raw, 9)

    enemy_below_w = np.zeros((3, 3), np.float32)
    for i in range(3):
        for j in range(3):
            enemy_below_w[i, j] = float(i < j)
    p("enemy_below_w", enemy_below_w)
    enemy_below = dn("enemy_below", enemy_wanted, "enemy_below_w")

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for k, (dy, dx) in enumerate(directions):
        for name, channels, decay in [
            ("fruit", 9, 0.90),
            ("enemy", 3, 0.90),
            ("scalar", 1, 0.96),
        ]:
            w = np.zeros((channels, channels, 3, 3), np.float32)
            for c in range(channels):
                w[c, c, dy + 1, dx + 1] = decay
            p("%s_dir_%d" % (name, k), w)

    def matching_fields(source, prefix, stem, steps):
        left = prop(source, prefix + "_dir_2", "add", steps)
        right = prop(source, prefix + "_dir_3", "add", steps)
        h0 = n(stem + "_h0", "add", [left, right])
        h = n(stem + "_h", "sub", [h0, source])
        up = prop(h, prefix + "_dir_0", "add", steps)
        down = prop(h, prefix + "_dir_1", "add", steps)
        total0 = n(stem + "_total0", "add", [up, down])
        total = n(stem + "_total", "sub", [total0, h])
        maximum = source
        for k in range(4):
            maximum = prop(
                maximum, prefix + "_dir_%d" % k, "max2", steps
            )
        other = n(stem + "_other_raw", "sub", [total, maximum])
        return n(stem + "_other", "relu", [other])

    other_fruit = matching_fields(
        active_fruit, "fruit", "fruit_match", 30
    )
    other_enemy = matching_fields(
        active_enemy, "enemy", "enemy_match", 24
    )

    goal_field = goals
    for k in range(4):
        goal_field = prop(
            goal_field, "scalar_dir_%d" % k, "max2", 63
        )
    enemy_field = enemy_seed
    for k in range(4):
        enemy_field = prop(
            enemy_field, "scalar_dir_%d" % k, "max2", 63
        )
    fruit_seed_sum = cv("fruit_seed_sum", fruit, "sum9")
    fruit_seed = n("fruit_seed", "min2", [fruit_seed_sum, one])
    fruit_field = fruit_seed
    for k in range(4):
        fruit_field = prop(
            fruit_field, "scalar_dir_%d" % k, "max2", 63
        )

    # Radius-eight Manhattan reach around object centers.
    objects = n("objects", "concat", [active_fruit, active_enemy])
    rw = np.zeros((48, 12, 3, 3), np.float32)
    for k, (dy, dx) in enumerate(directions):
        for c in range(12):
            rw[k * 12 + c, c, dy + 1, dx + 1] = 1
    p("reach_w", rw)
    graph.append({
        "id": "reach_loop",
        "op": "loop",
        "T": 8,
        "state": [{"id": "reach_state", "init": objects}],
        "body": [
            {"id": "reach_shift", "op": "conv",
             "in": ["reach_state"], "w": "reach_w"},
            {"id": "reach_a", "op": "slice",
             "in": ["reach_shift"], "start": 0, "stop": 12},
            {"id": "reach_b", "op": "slice",
             "in": ["reach_shift"], "start": 12, "stop": 24},
            {"id": "reach_c", "op": "slice",
             "in": ["reach_shift"], "start": 24, "stop": 36},
            {"id": "reach_d", "op": "slice",
             "in": ["reach_shift"], "start": 36, "stop": 48},
            {"id": "reach_ab", "op": "max2",
             "in": ["reach_a", "reach_b"]},
            {"id": "reach_cd", "op": "max2",
             "in": ["reach_c", "reach_d"]},
            {"id": "reach_all", "op": "max2",
             "in": ["reach_ab", "reach_cd"]},
            {"id": "reach_next", "op": "max2",
             "in": ["reach_state", "reach_all"]},
        ],
        "next": {"reach_state": "reach_next"},
    })
    fr = sl("fruit_reach", "reach_loop.reach_state", 0, 9)
    er = sl("enemy_reach", "reach_loop.reach_state", 9, 12)
    fr_count = cv("fruit_reach_count", fr, "sum9")
    fr_any = n("fruit_reach_any", "min2", [fr_count, one])
    no_fr = n("no_fruit_reach", "sub", [one, fr_any])

    scalar_reach_w = np.zeros((4, 1, 3, 3), np.float32)
    for k, (dy, dx) in enumerate(directions):
        scalar_reach_w[k, 0, dy + 1, dx + 1] = 1
    p("scalar_reach_w", scalar_reach_w)
    graph.append({
        "id": "loose_reach_loop",
        "op": "loop",
        "T": 8,
        "state": [{"id": "loose_reach_state", "init": enemy_seed}],
        "body": [
            {"id": "loose_shift", "op": "conv",
             "in": ["loose_reach_state"], "w": "scalar_reach_w"},
            {"id": "loose_a", "op": "slice",
             "in": ["loose_shift"], "start": 0, "stop": 1},
            {"id": "loose_b", "op": "slice",
             "in": ["loose_shift"], "start": 1, "stop": 2},
            {"id": "loose_c", "op": "slice",
             "in": ["loose_shift"], "start": 2, "stop": 3},
            {"id": "loose_d", "op": "slice",
             "in": ["loose_shift"], "start": 3, "stop": 4},
            {"id": "loose_ab", "op": "max2",
             "in": ["loose_a", "loose_b"]},
            {"id": "loose_cd", "op": "max2",
             "in": ["loose_c", "loose_d"]},
            {"id": "loose_all", "op": "max2",
             "in": ["loose_ab", "loose_cd"]},
            {"id": "loose_reach_next", "op": "max2",
             "in": ["loose_reach_state", "loose_all"]},
        ],
        "next": {"loose_reach_state": "loose_reach_next"},
    })
    loose_reach = "loose_reach_loop.loose_reach_state"

    pair = n("pair_eligible", "mul", [other_fruit, below_goal])
    pair = n("pair_reachable", "mul", [pair, fr])
    pair = cv("pair_score", pair, "sum9")
    delivery = n("wanted_fruit_reach", "mul", [fr, fruit_wanted])
    delivery = cv("wanted_fruit_sum", delivery, "sum9")
    delivery = n("fruit_delivery", "mul", [delivery, goal_field])
    enemy_delivery = n("wanted_enemy_reach", "mul", [er, enemy_wanted])
    enemy_delivery = cv("wanted_enemy_sum", enemy_delivery, "sum3")
    enemy_delivery = n(
        "enemy_delivery", "mul", [enemy_delivery, goal_field]
    )
    enemy_pair = n(
        "enemy_pair_eligible", "mul", [other_enemy, enemy_below]
    )
    enemy_pair = n("enemy_pair_reach", "mul", [enemy_pair, er])
    enemy_pair = cv("enemy_pair_sum", enemy_pair, "sum3")
    enemy_pair = n("enemy_pair", "mul", [enemy_pair, no_fr])

    oversized_reach = n("oversized_reach", "mul", [fr, above_goal])
    oversized_reach = cv(
        "oversized_reach_sum", oversized_reach, "sum9"
    )
    not_loose_reach = n(
        "not_loose_reach", "sub", [one, loose_reach]
    )
    approach = n("approach0", "mul", [oversized_reach, enemy_field])
    approach = n("approach", "mul", [approach, not_loose_reach])

    oversized = n("oversized", "mul", [fruit, above_goal])
    vulnerable = n("vulnerable", "sub", [fruit, oversized])
    threat_channels = n(
        "threat_channels", "mul", [vulnerable, enemy_field]
    )
    threat_map = cv("threat_map", threat_channels, "sum9")
    threat_value = n("threat_value", "gmax", [threat_map])
    p("threat_w", [[5.0]])
    p("threat_b", [-2.0])
    threat = dn("threat_linear", threat_value, "threat_w", "threat_b")
    threat = n("threat_positive", "vrelu", [threat])
    threat = clip_vector("threat", threat)
    away = n("away_from_fruit", "sub", [one, fruit_field])
    escape = n("escape_reach", "mul", [loose_reach, away])
    escape = n("escape_safe", "mul", [escape, no_fr])
    escape = n("escape", "mul", [escape, threat])

    near_channels = n("near_channels", "mul", [oversized, enemy_field])
    near_map = cv("near_map", near_channels, "sum9")
    near_value = n("near_value", "gmax", [near_map])
    p("near_w", [[10.0]])
    p("near_b", [-5.5])
    wait = dn("wait_linear", near_value, "near_w", "near_b")
    wait = n("wait_positive", "vrelu", [wait])
    wait = clip_vector("wait_gate", wait)
    occupied = n("occupied_seed", "max2", [fruit_seed, enemy_seed])
    for i in range(5):
        occupied = cv("occupied_raw_%d" % i, occupied, "box7")
        occupied = n("occupied_%d" % i, "min2", [occupied, one])
    idle = n("safe_idle", "sub", [one, occupied])
    idle = n("idle_score", "mul", [idle, wait])

    fallback = n("fallback", "mul", [fr_count, goal_field])
    mixed = n("mixed_raw", "sub", [fr_count, one])
    mixed = n("mixed", "relu", [mixed])

    # Recent click locations are retained across undo.
    click1 = vs("click1", "mem", 12, 14)
    click2 = vs("click2", "mem", 14, 16)
    prev_undo = vs("prev_undo", "aprev", 7, 8)
    p("near_click_w", np.array([[[[-8.0]]]], np.float32))
    p("near_click_b", [1.0])

    def click_neighborhood(name, saved):
        delta = n(name + "_delta", "sub", [coord_pair, saved])
        delta = n(name + "_abs", "abs", [delta])
        dy = sl(name + "_dy", delta, 0, 1)
        dx = sl(name + "_dx", delta, 1, 2)
        distance = n(name + "_distance", "max2", [dy, dx])
        raw = cv(name + "_raw", distance, "near_click_w", "near_click_b")
        return n(name, "relu", [raw])

    retry1 = click_neighborhood("retry1", click1)
    retry2 = click_neighborhood("retry2", click2)
    retry = n("retry_max", "max2", [retry1, retry2])
    retry = n("retry", "mul", [retry, prev_undo])

    scores = n(
        "score_inputs", "concat",
        [pair, delivery, enemy_delivery, enemy_pair, approach, escape,
         idle, fallback, mixed, retry],
    )
    p("score_w", np.array(
        [4.0, 2.4, 1.8, 3.0, 2.3, 24.0, 8.0, 0.05, -6.0, -4.0],
        np.float32,
    ).reshape(1, 10, 1, 1))
    score = cv("score", scores, "score_w")
    illegal = n("illegal", "sub", [one, play])
    p("illegal_w", np.array([[[[-50.0]]]], np.float32))
    illegal = cv("illegal_penalty", illegal, "illegal_w")
    score = n("legal_score", "add", [score, illegal])
    tie_w = np.zeros((1, 3, 1, 1), np.float32)
    tie_w[0, 0, 0, 0] = -0.01
    tie_w[0, 1, 0, 0] = -0.00015
    p("tie_w", tie_w)
    tie = cv("tie", "coords", "tie_w")
    click = n("click", "add", [score, tie])

    click_max = n("click_max", "gmax", [click])
    gap = n("click_gap", "sub", [click_max, click])
    p("peak_w", np.array([[[[-1000000.0]]]], np.float32))
    p("peak_b", [1.0])
    peak = cv("peak_raw", gap, "peak_w", "peak_b")
    peak = n("peak", "relu", [peak])
    peak_coords = n("peak_coords", "mul", [coord_pair, peak])
    new_click = n("new_click", "gmax", [peak_coords])

    red = sl("red", "x0", 2, 3)
    red_header = n("red_header", "mul", [red, header])
    red_present = n("red_present", "gmax", [red_header])
    undo_flag = vs("undo_flag", "mem", 16, 17)
    second_undo = n("second_undo", "vmul", [undo_flag, prev_undo])
    undo_sum = n("undo_sum", "vadd", [red_present, second_undo])
    undo = clip_vector("undo", undo_sum)
    not_undo = dn("not_undo", undo, "neg1", "bias1")

    c1_old = n("c1_old", "vmul", [click1, undo])
    c1_new = n("c1_new", "vmul", [new_click, not_undo])
    saved_click1 = n("saved_click1", "vadd", [c1_old, c1_new])
    c2_old = n("c2_old", "vmul", [click2, undo])
    c2_new = n("c2_new", "vmul", [click1, not_undo])
    saved_click2 = n("saved_click2", "vadd", [c2_old, c2_new])
    n("mem_next", "vconcat",
      [saved_coords, saved_flags, saved_click1, saved_click2, red_present])

    logit_w = np.zeros((8, 1), np.float32)
    logit_b = np.full(8, -100.0, np.float32)
    logit_b[6] = 0
    logit_b[7] = -0.5
    logit_w[7, 0] = 1
    p("logit_w", logit_w)
    p("logit_b", logit_b)
    dn("logits", undo, "logit_w", "logit_b")

    return {"graph": graph, "memory_size": 17, "params": params}
