import numpy as np


def build():
    graph = []
    params = {}

    def node(name, op, inputs=None, **fields):
        d = {"id": name, "op": op}
        if inputs is not None:
            d["in"] = list(inputs)
        d.update(fields)
        graph.append(d)
        return name

    def param(name, value):
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def conv(name, inp, weights, bias=None):
        kw = {"w": param(name + "_w", weights)}
        if bias is not None:
            kw["b"] = param(name + "_b", bias)
        return node(name, "conv", [inp], **kw)

    def dense(name, inp, weights, bias=None):
        kw = {"w": param(name + "_w", weights)}
        if bias is not None:
            kw["b"] = param(name + "_b", bias)
        return node(name, "dense", [inp], **kw)

    def sl(name, inp, start, stop):
        return node(name, "slice", [inp], start=start, stop=stop)

    def vsl(name, inp, start, stop):
        return node(name, "vslice", [inp], start=start, stop=stop)

    def mul(name, a, b):
        return node(name, "mul", [a, b])

    def add(name, a, b):
        return node(name, "add", [a, b])

    def vmul(name, a, b):
        return node(name, "vmul", [a, b])

    def vadd(name, a, b):
        return node(name, "vadd", [a, b])

    def vnot(name, x):
        return dense(name, x, [[-1.0]], [1.0])

    def scale(name, x, value):
        return conv(name, x, np.array([[[[value]]]], dtype=np.float32))

    def vclip(name, x):
        t = dense(name + "_shift", x, [[1.0]], [-1.0])
        t = node(name + "_excess", "vrelu", [t])
        t = dense(name + "_negative", t, [[-1.0]])
        return vadd(name, x, t)

    def sclip(name, x, one):
        x = node(name + "_positive", "relu", [x])
        return node(name, "min2", [x, one])

    def at_player(name, field):
        overlap = mul(name + "_overlap", field, "player")
        return node(name, "gmax", [overlap])

    one = sl("one", "coords", 2, 3)
    row = sl("row", "coords", 0, 1)
    col = sl("col", "coords", 1, 2)

    # Player.
    w = np.zeros((1, 16, 5, 5), dtype=np.float32)
    w[0, 9, :, :] = 1
    w[0, 9, 2, 2] = 0
    p = conv("player_raw", "x0", w, [-23.5])
    p = node("player_positive", "relu", [p])
    player = scale("player", p, 2.0)

    # Goal.
    w = np.zeros((1, 16, 7, 7), dtype=np.float32)
    w[0, 9, 3, 3] = 10
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        w[0, 5, 3 + dy, 3 + dx] = 1
    for dy, dx in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
        w[0, 9, 3 + dy, 3 + dx] = 1
    g = conv("goal_raw", "x0", w, [-16.5])
    g = node("goal_positive", "relu", [g])
    goal = scale("goal", g, 2.0)

    # Doors and switches.
    w = np.zeros((1, 16, 5, 5), dtype=np.float32)
    w[0, 8, :, :] = 2
    w[0, 11, :, :] = 2
    door = sclip("door", conv("door_raw", "x0", w, [-35]), one)

    w = np.zeros((3, 16, 3, 3), dtype=np.float32)
    for i, color in enumerate((8, 11, 15)):
        w[i, color, :, :] = 2
    sw = conv("switch_raw", "x0", w, [-17, -17, -17])
    sw = node("switch_positive", "relu", [sw])
    blue = node("blue", "min2", [sl("blue_raw", sw, 0, 1), one])
    purple = node("purple", "min2", [sl("purple_raw", sw, 1, 2), one])
    white = node("white", "min2", [sl("white_raw", sw, 2, 3), one])
    bp = node("blue_purple", "max2", [blue, purple])
    all_sw = node("all_switches", "max2", [bp, white])
    sw_sub = node("switch_minus_door", "sub", [all_sw, door])
    switches = node("switches", "relu", [sw_sub])

    # White-bordered portals.
    w = np.zeros((1, 16, 7, 7), dtype=np.float32)
    for y in range(7):
        for x in range(7):
            if y in (0, 6) or x in (0, 6):
                w[0, 15, y, x] = 2
    portal = sclip("portal", conv("portal_raw", "x0", w, [-47]), one)

    # Memory:
    # 0:4 blue arrival flags; 4:8 white arrival flags
    # 8:14 two remembered switch locations
    # 14 portal mode; 15:19 detour state
    # 19:21 previous player position; 21:23 failed destination; 23 age.
    old_blue_dirs = vsl("old_blue_dirs", "mem", 0, 4)
    old_white_dirs = vsl("old_white_dirs", "mem", 4, 8)
    old_flag1 = vsl("old_flag1", "mem", 8, 9)
    old_row1 = vsl("old_row1", "mem", 9, 10)
    old_col1 = vsl("old_col1", "mem", 10, 11)
    old_flag2 = vsl("old_flag2", "mem", 11, 12)
    old_row2 = vsl("old_row2", "mem", 12, 13)
    old_col2 = vsl("old_col2", "mem", 13, 14)
    old_portal = vsl("old_portal", "mem", 14, 15)
    delay1 = vsl("delay1", "mem", 15, 16)
    delay2 = vsl("delay2", "mem", 16, 17)
    delay3 = vsl("delay3", "mem", 17, 18)
    delay4 = vsl("delay4", "mem", 18, 19)
    old_position = vsl("old_position", "mem", 19, 21)
    old_block = vsl("old_block", "mem", 21, 23)
    old_age = vsl("old_age", "mem", 23, 24)

    prev_dirs = vsl("prev_dirs", "aprev", 1, 5)
    prev_rewind = vsl("prev_rewind", "aprev", 5, 6)
    not_prev_rewind = vnot("not_prev_rewind", prev_rewind)

    blue_hit_vec = vmul("blue_hit_vec", old_blue_dirs, prev_dirs)
    white_hit_vec = vmul("white_hit_vec", old_white_dirs, prev_dirs)
    blue_hit = dense("blue_hit", blue_hit_vec, [[1, 1, 1, 1]])
    white_hit = dense("white_hit", white_hit_vec, [[1, 1, 1, 1]])
    rewind_request = vclip("rewind_request", vadd("hit_sum", blue_hit, white_hit))

    player_row = node("player_row", "gmax", [mul("player_row_map", player, row)])
    player_col = node("player_col", "gmax", [mul("player_col_map", player, col)])
    position = node("position", "vconcat", [player_row, player_col])

    # Record a directional action that did not move the player.
    position_pair = node("position_pair", "vconcat", [position, old_position])
    diff = dense(
        "position_diff",
        position_pair,
        [[1, 0, -1, 0], [0, 1, 0, -1]],
    )
    pos_diff = node("positive_diff", "vrelu", [diff])
    neg_diff = dense("negative_diff", diff, [[-1, 0], [0, -1]])
    neg_diff = node("negative_diff_positive", "vrelu", [neg_diff])
    abs_diff = vadd("abs_diff", pos_diff, neg_diff)
    stationary_raw = dense("stationary_raw", abs_diff, [[-32, -32]], [1])
    stationary = node("stationary", "vrelu", [stationary_raw])
    prev_move = dense("prev_move", prev_dirs, [[1, 1, 1, 1]])
    failed = vmul("failed", stationary, prev_move)
    not_failed = vnot("not_failed", failed)
    delta = dense(
        "previous_delta",
        prev_dirs,
        [
            [-6 / 63, 6 / 63, 0, 0],
            [0, 0, -6 / 63, 6 / 63],
        ],
    )
    attempted = vadd("attempted", old_position, delta)
    old_block_keep = vmul("old_block_keep", old_block, not_failed)
    new_block_part = vmul("new_block_part", attempted, failed)
    block_next = vadd("block_next", old_block_keep, new_block_part)
    age_decrement = dense("age_decrement", old_age, [[1]], [-1])
    age_decrement = node("age_positive", "vrelu", [age_decrement])
    age_keep0 = vmul("age_keep0", age_decrement, not_failed)
    age_keep = vmul("age_keep", age_keep0, not_prev_rewind)
    age_new = dense("age_new", failed, [[12]])
    age_next = vadd("age_next", age_keep, age_new)
    block_active = vclip("block_active", age_next)
    block_row = vsl("block_row", block_next, 0, 1)
    block_col = vsl("block_col", block_next, 1, 2)

    def location(name, flag, saved_row, saved_col):
        dr = node(name + "_dr", "sub", [row, saved_row])
        dc = node(name + "_dc", "sub", [col, saved_col])
        dr = node(name + "_adr", "abs", [dr])
        dc = node(name + "_adc", "abs", [dc])
        dr = scale(name + "_dr63", dr, 63)
        dc = scale(name + "_dc63", dc, 63)
        rr = node(name + "_rr", "sub", [one, dr])
        cc = node(name + "_cc", "sub", [one, dc])
        rr = node(name + "_rr_pos", "relu", [rr])
        cc = node(name + "_cc_pos", "relu", [cc])
        return mul(name, mul(name + "_both", rr, cc), flag)

    blocked_tile = location("blocked_tile", block_active, block_row, block_col)

    # Estimate walkable cells and apply the temporary failed-move mask.
    w = np.zeros((1, 16, 5, 5), dtype=np.float32)
    for color in (2, 5, 8, 9, 11, 14, 15):
        w[0, color, :, :] = 2
    filled = sclip("filled", conv("filled_raw", "x0", w, [-39]), one)
    w = np.zeros((1, 16, 1, 1), dtype=np.float32)
    for color in (2, 5, 8, 9, 11, 15):
        w[0, color, 0, 0] = 1
    center = conv("center_allowed", "x0", w)
    open0 = mul("open0", filled, center)
    closed_center = node("closed_center", "sub", [center, door])
    closed_center = node("closed_center_positive", "relu", [closed_center])
    closed0 = mul("closed0", filled, closed_center)
    open_sub = node("open_sub", "sub", [open0, blocked_tile])
    closed_sub = node("closed_sub", "sub", [closed0, blocked_tile])
    floor_open_01 = node("floor_open_01", "relu", [open_sub])
    floor_closed_01 = node("floor_closed_01", "relu", [closed_sub])
    floor_open = scale("floor_open", floor_open_01, 100)
    floor_closed = scale("floor_closed", floor_closed_01, 100)
    goal_seed = scale("goal_seed", goal, 100)
    player_seed = scale("player_seed", player, 100)

    for i, (dy, dx) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        w = np.zeros((1, 1, 7, 7), dtype=np.float32)
        w[0, 0, 3 + 3 * dy, 3 + 3 * dx] = 1
        param("shift_%d" % i, w)
    param("wave_decay", [-1])

    def wave(name, seed, floor):
        state = name + "_state"
        body = []
        shifts = []
        for i in range(4):
            a = name + "_shift3_%d" % i
            b = name + "_shift6_%d" % i
            body.append({"id": a, "op": "conv", "in": [state], "w": "shift_%d" % i})
            body.append({
                "id": b, "op": "conv", "in": [a],
                "w": "shift_%d" % i, "b": "wave_decay",
            })
            shifts.append(b)
        a = name + "_max01"
        b = name + "_max23"
        c = name + "_maxall"
        d = name + "_keep"
        e = name + "_next"
        body.extend([
            {"id": a, "op": "max2", "in": [shifts[0], shifts[1]]},
            {"id": b, "op": "max2", "in": [shifts[2], shifts[3]]},
            {"id": c, "op": "max2", "in": [a, b]},
            {"id": d, "op": "max2", "in": [state, c]},
            {"id": e, "op": "min2", "in": [d, floor]},
        ])
        graph.append({
            "id": name,
            "op": "loop",
            "T": 100,
            "state": [{"id": state, "init": seed}],
            "body": body,
            "next": {state: e},
        })
        return name + "." + state

    goal_closed = wave("goal_closed", goal_seed, floor_closed)
    goal_open = wave("goal_open", goal_seed, floor_open)
    player_closed = wave("player_closed", player_seed, floor_closed)
    player_open = wave("player_open", player_seed, floor_open)
    closed_value = at_player("closed_value", goal_closed)
    closed_reachable = vclip("closed_reachable", closed_value)
    goal_blocked = vnot("goal_blocked", closed_reachable)

    saved1 = location("saved1", old_flag1, old_row1, old_col1)
    saved2 = location("saved2", old_flag2, old_row2, old_col2)
    saved = node("saved", "max2", [saved1, saved2])
    available_raw = node("available_raw", "sub", [switches, saved])
    available = node("available", "relu", [available_raw])
    player_closed_reach = node("player_closed_reach", "min2", [player_closed, one])
    reachable_switches = mul("reachable_switches", available, player_closed_reach)

    goal_open_reach = node("goal_open_reach", "min2", [goal_open, one])
    player_open_reach = node("player_open_reach", "min2", [player_open, one])
    eligible0 = mul("eligible0", door, goal_open_reach)
    eligible = mul("eligible", eligible0, player_open_reach)
    door_value = add("door_value", goal_open, player_open)

    w = np.zeros((1, 16, 1, 1), dtype=np.float32)
    for color in (8, 11, 15):
        w[0, color, 0, 0] = 1
    wire = conv("wire", "x0", w)
    cross = np.zeros((1, 1, 3, 3), dtype=np.float32)
    for y, x in [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]:
        cross[0, 0, y, x] = 1
    param("wire_cross", cross)

    def wire_fill(name, seed):
        state = name + "_state"
        spread = name + "_spread"
        nxt = name + "_next"
        graph.append({
            "id": name,
            "op": "loop",
            "T": 128,
            "state": [{"id": state, "init": seed}],
            "body": [
                {"id": spread, "op": "conv", "in": [state], "w": "wire_cross"},
                {"id": nxt, "op": "min2", "in": [spread, wire]},
            ],
            "next": {state: nxt},
        })
        return name + "." + state

    remaining = eligible
    candidates = []
    flags = []
    for i in range(4):
        score = mul("door_score_%d" % i, door_value, remaining)
        best = node("door_best_%d" % i, "gmax", [score])
        threshold = dense("door_threshold_%d" % i, best, [[1]], [-1])
        rel = node("door_relative_%d" % i, "sub", [score, threshold])
        rel = node("door_relative_pos_%d" % i, "relu", [rel])
        winner = mul("door_winner_%d" % i, rel, remaining)
        winner = node("door_winner_clip_%d" % i, "min2", [winner, one])
        rem = node("door_remaining_raw_%d" % i, "sub", [remaining, winner])
        remaining = node("door_remaining_%d" % i, "relu", [rem])
        seed = mul("wire_seed_%d" % i, winner, wire)
        connected = wire_fill("wire_fill_%d" % i, seed)
        candidate = mul("wire_candidate_%d" % i, connected, reachable_switches)
        candidates.append(candidate)
        flags.append(node("wire_flag_%d" % i, "gmax", [candidate]))

    none_before = dense("initial_none_before", flags[0], [[0]], [1])
    parts = []
    for i in range(4):
        parts.append(mul("selected_part_%d" % i, candidates[i], none_before))
        no_this = vnot("not_wire_flag_%d" % i, flags[i])
        none_before = vmul("none_before_%d" % i, none_before, no_this)
    selected = parts[0]
    for i in range(1, 4):
        selected = node("selected_union_%d" % i, "max2", [selected, parts[i]])
    fallback = mul("fallback", reachable_switches, none_before)
    targets = node("targets", "max2", [selected, fallback])
    has_target = node("has_target", "gmax", [targets])
    no_target = vnot("no_target", has_target)
    target_wave = wave("target_wave", scale("target_seed", targets, 100), floor_closed)

    no_flag1 = vnot("no_flag1", old_flag1)
    no_flag2 = vnot("no_flag2", old_flag2)
    capture1 = vmul("capture1", rewind_request, no_flag1)
    capture2a = vmul("capture2a", rewind_request, old_flag1)
    capture2 = vmul("capture2", capture2a, no_flag2)
    clear = vmul("clear", capture2a, old_flag2)
    keep = vnot("keep", clear)

    portal_kept = vmul("portal_kept", old_portal, keep)
    portal_mode = vclip("portal_mode", vadd("portal_mode_sum", portal_kept, white_hit))
    goal_closed_reach = node("goal_closed_reach", "min2", [goal_closed, one])
    goal_portals = mul("goal_portals", portal, goal_closed_reach)
    has_goal_portal = node("has_goal_portal", "gmax", [goal_portals])
    source0 = mul("source0", portal, player_closed_reach)
    sources = mul("sources", source0, has_goal_portal)
    has_source = node("has_source", "gmax", [sources])
    portal_wave = wave("portal_wave", scale("portal_seed", sources, 100), floor_closed)
    portal_enabled0 = vmul("portal_enabled0", portal_mode, has_source)
    portal_enabled = vmul("portal_enabled", portal_enabled0, goal_blocked)
    no_portal = vnot("no_portal", portal_enabled)
    target_mode0 = vmul("target_mode0", goal_blocked, has_target)
    target_mode = vmul("target_mode", target_mode0, no_portal)
    optimistic0 = vmul("optimistic0", goal_blocked, no_target)
    optimistic = vmul("optimistic", optimistic0, no_portal)

    chosen_goal = mul("chosen_goal", goal_closed, closed_reachable)
    chosen_target = mul("chosen_target", target_wave, target_mode)
    chosen_open = mul("chosen_open", goal_open, optimistic)
    chosen_portal = mul("chosen_portal", portal_wave, portal_enabled)
    chosen01 = add("chosen01", chosen_goal, chosen_target)
    chosen23 = add("chosen23", chosen_open, chosen_portal)
    chosen = add("chosen", chosen01, chosen23)
    chosen_baseline = add("chosen_baseline", chosen, one)

    def shift6(name, x, direction):
        first = node(name + "_first", "conv", [x], w="shift_%d" % direction)
        return node(name, "conv", [first], w="shift_%d" % direction)

    target_blue = mul("target_blue", mul("target_blue0", targets, blue), target_mode)
    target_white = mul("target_white", mul("target_white0", targets, white), target_mode)
    scores = []
    legal_flags = []
    blue_next = []
    white_next = []
    for i in range(4):
        value = shift6("neighbor_value_%d" % i, chosen_baseline, i)
        legal = shift6("neighbor_legal_%d" % i, floor_closed_01, i)
        scores.append(at_player("direction_score_%d" % i, mul("legal_value_%d" % i, value, legal)))
        legal_flags.append(at_player("legal_flag_%d" % i, legal))
        b = shift6("neighbor_blue_%d" % i, target_blue, i)
        h = shift6("neighbor_white_%d" % i, target_white, i)
        blue_next.append(at_player("blue_direction_%d" % i, b))
        white_next.append(at_player("white_direction_%d" % i, h))
    scores = node("scores", "vconcat", scores)
    legal_flags = node("legal_flags", "vconcat", legal_flags)
    blue_next = node("blue_dirs_next", "vconcat", blue_next)
    white_next = node("white_dirs_next", "vconcat", white_next)

    # When a blue switch is recorded on a board with white portals,
    # record two out-and-back moves before proceeding toward white.
    has_white = node("has_white", "gmax", [white])
    has_portal = node("has_portal", "gmax", [portal])
    delay_start0 = vmul("delay_start0", blue_hit, has_white)
    delay_start = vmul("delay_start", delay_start0, has_portal)
    delay_away = vadd("delay_away", delay1, delay3)
    delay_back = vadd("delay_back", delay2, delay4)
    away_bonus0 = vmul("away_bonus0", legal_flags, delay_away)
    away_bonus = dense(
        "away_bonus", away_bonus0,
        np.eye(4, dtype=np.float32) * 200,
    )
    reverse = dense(
        "reverse_previous",
        prev_dirs,
        [[0, 1, 0, 0], [1, 0, 0, 0],
         [0, 0, 0, 1], [0, 0, 1, 0]],
    )
    reverse_legal = vmul("reverse_legal", reverse, legal_flags)
    back_bonus0 = vmul("back_bonus0", reverse_legal, delay_back)
    back_bonus = dense(
        "back_bonus", back_bonus0,
        np.eye(4, dtype=np.float32) * 200,
    )
    scores = vadd("scores_with_away", scores, away_bonus)
    scores = vadd("scores_with_back", scores, back_bonus)

    def update_slot(name, flag, r, c, capture):
        f = vadd(name + "_flag", vmul(name + "_kept_flag", flag, keep), capture)
        nr = vadd(
            name + "_row",
            vmul(name + "_kept_row", r, keep),
            vmul(name + "_captured_row", player_row, capture),
        )
        nc = vadd(
            name + "_col",
            vmul(name + "_kept_col", c, keep),
            vmul(name + "_captured_col", player_col, capture),
        )
        return f, nr, nc

    flag1, row1, col1 = update_slot("slot1", old_flag1, old_row1, old_col1, capture1)
    flag2, row2, col2 = update_slot("slot2", old_flag2, old_row2, old_col2, capture2)

    node(
        "mem_next",
        "vconcat",
        [
            blue_next, white_next,
            flag1, row1, col1, flag2, row2, col2,
            portal_mode,
            delay_start, delay1, delay2, delay3,
            position, block_next, age_next,
        ],
    )

    features = node("action_features", "vconcat", [scores, rewind_request])
    w = np.zeros((8, 5), dtype=np.float32)
    for i in range(4):
        w[i + 1, i] = 1
    w[5, 4] = 1000
    dense(
        "logits", features, w,
        [-100, 0, 0, 0, 0, -2, -100, -100],
    )
    sl("click", "coords", 2, 3)

    return {"graph": graph, "memory_size": 24, "params": params}
