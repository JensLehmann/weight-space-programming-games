import numpy as np


def build():
    P = {}
    graph = []
    out = graph
    serial = [0]

    def emit(op, ins=None, **kw):
        name = "n%d" % serial[0]
        serial[0] += 1
        nd = {"id": name, "op": op}
        if ins is not None:
            nd["in"] = list(ins)
        nd.update(kw)
        out.append(nd)
        return name

    def parameter(a):
        name = "p%d" % len(P)
        P[name] = np.asarray(a, dtype=np.float32)
        return name

    def dense(x, w, b=None):
        kw = {"w": w if isinstance(w, str) else parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return emit("dense", [x], **kw)

    def conv(x, w, b=None):
        kw = {"w": w if isinstance(w, str) else parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return emit("conv", [x], **kw)

    def sl(x, a, b):
        return emit("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return emit("vslice", [x], start=a, stop=b)

    def cat(*xs):
        return emit("concat", xs)

    def vcat(*xs):
        return emit("vconcat", xs)

    def add(a, b):
        return emit("add", [a, b])

    def sub(a, b):
        return emit("sub", [a, b])

    def mul(a, b):
        return emit("mul", [a, b])

    def va(a, b):
        return emit("vadd", [a, b])

    def vm(a, b):
        return emit("vmul", [a, b])

    def rel(x):
        return emit("relu", [x])

    def vr(x):
        return emit("vrelu", [x])

    def gm(x):
        return emit("gmax", [x])

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    spatial_one = sl("coords", 2, 3)
    one = gm(spatial_one)
    constants = {}

    def c(v):
        key = float(v)
        if key not in constants:
            constants[key] = dense(one, [[key]])
        return constants[key]

    def cv(v):
        return dense(one, np.asarray(v, np.float32).reshape(-1, 1))

    def neg(x):
        return vm(x, c(-1))

    def vsub(a, b):
        return va(a, neg(b))

    def vscale(x, k):
        return vm(x, c(k))

    def vshift(x, k):
        return va(x, c(k))

    def vabs(x):
        return va(vr(x), vr(neg(x)))

    def clip(x):
        y = vr(x)
        return vsub(y, vr(vshift(y, -1)))

    def veq(x, y):
        return vr(vshift(neg(vabs(vsub(x, y))), 1))

    def eqk(x, k):
        return veq(x, c(k))

    def vmax(a, b):
        return va(b, vr(vsub(a, b)))

    def sscale(x, k):
        return mul(x, c(k))

    def sshift(x, k):
        return add(x, c(k))

    def sclip(x):
        return emit("min2", [rel(x), one])

    def seq(a, b):
        d = emit("abs", [sub(a, b)])
        return rel(sshift(sscale(d, -1), 1))

    def sum_vectors(xs):
        z = xs[0]
        for x in xs[1:]:
            z = va(z, x)
        return z

    def point_mask(y, x):
        dy = emit("abs", [sub(row, c(y / 63.0))])
        dx = emit("abs", [sub(col, c(x / 63.0))])
        return rel(sshift(sscale(add(dy, dx), -128), 1))

    # Five-pixel cells, with their centers at (2+5r, 6+5c).
    H = W = 12
    N = H * W
    rr = np.repeat(np.arange(H), W).astype(np.float32)
    cc = np.tile(np.arange(W), H).astype(np.float32)
    cy = cv((2 + 5 * rr) / 63.0)
    cx = cv((6 + 5 * cc) / 63.0)
    dy = emit("abs", [sub(row, cy)])
    dx = emit("abs", [sub(col, cx)])
    cell_masks = rel(sshift(sscale(add(dx, dy), -128), 1))

    def gather(x):
        return gm(mul(cell_masks, x))

    scatter_w = parameter(np.ones((1, N, 1, 1), np.float32))

    def scatter(x):
        return conv(mul(cell_masks, x), scatter_w)

    sum_w = parameter(np.ones((1, N), np.float32))

    def total(x):
        return dense(x, sum_w)

    # Neighbor-reading matrices: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    shift_w = []
    boundary = []
    for dr, dc in directions:
        a = np.zeros((N, N), np.float32)
        b = np.zeros(N, np.float32)
        for r in range(H):
            for q in range(W):
                i = r * W + q
                nr, nq = r + dr, q + dc
                if 0 <= nr < H and 0 <= nq < W:
                    a[i, nr * W + nq] = 1
                else:
                    b[i] = 1
        shift_w.append(parameter(a))
        boundary.append(cv(b))

    def shift(x, d):
        return dense(x, shift_w[d])

    # Shape recognition, independent of the glyph's color.
    palette = [12, 9, 14, 8]
    w = np.zeros((1, 16, 1, 1), np.float32)
    w[0, palette, 0, 0] = 1
    colored = conv("x0", w)
    glyph_input = cat(colored, sl("x0", 5, 6))

    shapes = [
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
    ]

    wg = np.zeros((24, 2, 3, 3), np.float32)
    wu = np.zeros((24, 2, 5, 5), np.float32)
    for s in range(6):
        for r in range(4):
            a = np.rot90(np.asarray(shapes[s]), -r)
            k = 4 * s + r
            for y in range(3):
                for x in range(3):
                    channel = 0 if a[y, x] else 1
                    wg[k, channel, y, x] = 1
                    wu[k, channel, 2 * y, 2 * x] = 1

    goal_glyphs = rel(conv(glyph_input, wg, np.full(24, -8)))
    ui_glyphs = rel(conv(glyph_input, wu, np.full(24, -8)))
    ui_mask = point_mask(57, 5)
    ui24 = gm(mul(ui_glyphs, ui_mask))

    a = np.zeros((10, 24), np.float32)
    for s in range(6):
        for r in range(4):
            a[s, 4 * s + r] = 1
            a[6 + r, 4 * s + r] = 1
    ui_sr = dense(ui24, a)

    a = np.zeros((4, 16, 7, 7), np.float32)
    for k, color in enumerate(palette):
        a[k, color] = 1
    ui_colors = gm(mul(sclip(conv("x0", a)), ui_mask))
    ui_state = vcat(ui_sr, ui_colors)

    a = np.zeros((4, 16, 3, 3), np.float32)
    for k, color in enumerate(palette):
        a[k, color] = 1
    goal_colors = sclip(conv("x0", a))

    goal_present_visual = conv(
        goal_glyphs, np.ones((1, 24, 1, 1), np.float32)
    )
    a = np.zeros((1, 24, 1, 1), np.float32)
    for s in range(6):
        for r in range(4):
            a[0, 4 * s + r, 0, 0] = 16 + 16 * s + r
    goal_base_code = conv(goal_glyphs, a)
    goal_color_code = conv(
        goal_colors,
        np.asarray([0, 4, 8, 12], np.float32).reshape(1, 4, 1, 1),
    )
    visual_goal_code = add(
        goal_base_code, mul(goal_present_visual, goal_color_code)
    )

    # Transformation-tile detectors.
    station_patterns = [
        [(0, -1, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0)],
        [
            (9, -1, -1), (14, -1, 0), (14, -1, 1),
            (9, 0, -1), (0, 0, 0), (8, 0, 1),
            (12, 1, -1), (12, 1, 0), (8, 1, 1),
        ],
        [
            (0, -1, 0), (0, 0, 0), (0, 0, 1),
            (1, 0, -1), (1, 1, 0),
        ],
    ]
    a = np.zeros((3, 16, 3, 3), np.float32)
    b = []
    for k, pat in enumerate(station_patterns):
        for color, y, x in pat:
            a[k, color, y + 1, x + 1] = 1
        b.append(1 - len(pat))
    stations = rel(conv("x0", a, b))

    a = np.ones((1, 1, 3, 3), np.float32)
    a[0, 0, 1, 1] = 0
    recharge = rel(conv(sl("x0", 11, 12), a, [-7]))

    a = np.zeros((4, 1, 7, 7), np.float32)
    for t in range(-2, 3):
        a[0, 0, 6, t + 3] = 1
        a[1, 0, 0, t + 3] = 1
        a[2, 0, t + 3, 6] = 1
        a[3, 0, t + 3, 0] = 1
    boosters = rel(conv(sl("x0", 1, 2), a, [-4] * 4))

    a = np.zeros((1, 16, 5, 5), np.float32)
    a[0, 12, :2, :] = 1
    a[0, 9, 2:, :] = 1
    player_spatial = rel(conv("x0", a, [-24]))
    player = gather(player_spatial)

    # Fuel and remaining lives.
    timer_row = rel(sshift(
        sscale(emit("abs", [sub(row, c(61 / 63.0))]), -128), 1
    ))
    left = sclip(sshift(sscale(col, 126), -25))
    right = sclip(sshift(sscale(col, -126), 109))
    timer_mask = mul(timer_row, mul(left, right))
    fuel = vscale(
        emit("gmean", [mul(sl("x0", 11, 12), timer_mask)]), 4096
    )
    lives = vscale(
        emit("gmean", [mul(sl("x0", 8, 9), timer_row)]), 2048
    )
    old_fuel = vs("mem", 154, 155)
    old_lives = vs("mem", 188, 189)
    same_life = veq(lives, old_lives)
    has_memory = clip(old_lives)

    # Board alphabet:
    # 0 unknown, 1 wall, 2 floor, 3..5 transformations,
    # 6 recharge, 7..10 boosters, 16..111 goal attributes.
    a = np.full((1, 16, 1, 1), 2, np.float32)
    a[0, 4, 0, 0] = 1
    a[0, 5, 0, 0] = 0
    base = conv("x0", a)
    features = cat(stations, recharge, boosters, goal_present_visual)
    feature_presence = conv(
        features, np.ones((1, 9, 1, 1), np.float32)
    )
    feature_codes = conv(
        features,
        np.asarray([3, 4, 5, 6, 7, 8, 9, 10, 0],
                   np.float32).reshape(1, 9, 1, 1),
    )
    raw_map_spatial = add(
        add(mul(base, sub(spatial_one, feature_presence)), feature_codes),
        visual_goal_code,
    )
    raw_map = gather(raw_map_spatial)
    old_map = vs("mem", 0, N)
    seen = clip(raw_map)
    board = va(vm(seen, raw_map), vm(vsub(one, seen), old_map))

    old_ge3 = clip(vshift(old_map, -2))
    old_le10 = clip(vsub(c(11), old_map))
    preserve = vm(player, vm(old_ge3, old_le10))
    board = va(vm(vsub(one, preserve), board), vm(preserve, old_map))

    # A changed HUD attribute identifies a tile concealed by the player.
    previous_ui = vs("mem", 156, 170)
    same_w = np.zeros((3, 14), np.float32)
    same_w[0, :6] = 1
    same_w[1, 10:14] = 1
    same_w[2, 6:10] = 1
    same_attributes = dense(vm(ui_state, previous_ui), same_w)
    previous_ui_valid = clip(dense(
        previous_ui, np.ones((1, 14), np.float32)
    ))
    observation_valid = vm(previous_ui_valid, same_life)
    attribute_changes = vm(
        clip(vsub(one, same_attributes)), observation_valid
    )

    station_cells = []
    map_corrections = []
    for k in range(3):
        cached = eqk(board, 3 + k)
        visible = gather(sl(stations, k, k + 1))
        changed = vs(attribute_changes, k, k + 1)
        replace = clip(va(total(visible), changed))
        observed = va(visible, vm(changed, player))
        cells = va(vm(vsub(one, replace), cached), observed)
        station_cells.append(cells)
        map_corrections.append(vscale(vsub(cells, cached), k + 1))
    board = va(board, sum_vectors(map_corrections))

    wall = eqk(board, 1)
    unknown = eqk(board, 0)
    refill_cells = eqk(board, 6)
    boost_cells = [eqk(board, k) for k in (7, 8, 9, 10)]
    goals = clip(vshift(board, -15))
    solid_for_rays = va(wall, goals)

    a = np.asarray(
        [16 * s + r for s in range(6) for r in range(4)],
        np.float32,
    ).reshape(1, 24)
    current_code = vshift(
        va(dense(ui24, a), dense(ui_colors, [[0, 4, 8, 12]])), 16
    )
    matching_goals = vm(goals, veq(board, current_code))
    blocked = va(wall, vsub(goals, matching_goals))
    traversable = vsub(one, blocked)

    # Cached goal decoding.
    board_spatial = scatter(board)
    codes = np.arange(16, 112, dtype=np.float32)
    goal_decode = conv(
        board_spatial, np.ones((96, 1, 1, 1), np.float32), -codes
    )
    goal_decode = emit("abs", [goal_decode])
    goal_decode = rel(sshift(sscale(goal_decode, -1), 1))
    cached_goal_presence = conv(
        goal_decode, np.ones((1, 96, 1, 1), np.float32)
    )

    # Forward cyclic edit counts for every goal identity.
    a = np.zeros((288, 14), np.float32)
    for s in range(6):
        for color in range(4):
            for rotation in range(4):
                k = 16 * s + 4 * color + rotation
                for old in range(6):
                    a[k, old] = (s - old) % 6
                for old in range(4):
                    a[96 + k, 10 + old] = (color - old) % 4
                    a[192 + k, 6 + old] = (rotation - old) % 4
    edits = dense(ui_state, a)
    ds = vs(edits, 0, 96)
    dc = vs(edits, 96, 192)
    dr = vs(edits, 192, 288)
    edit_cost = vscale(va(va(ds, dc), dr), 8)
    edit_cost_spatial = conv(
        mul(goal_decode, edit_cost),
        np.ones((1, 96, 1, 1), np.float32),
    )

    player_position = gm(mul("coords", player_spatial))
    py = vs(player_position, 0, 1)
    px = vs(player_position, 1, 2)
    distance = sscale(
        add(emit("abs", [sub(row, py)]),
            emit("abs", [sub(col, px)])),
        63.0 / 5.0,
    )
    previous_goal = vs("mem", 153, 154)
    same_goal = mul(seq(board_spatial, previous_goal), same_life)
    score = sub(
        sshift(sscale(same_goal, 256), 256),
        add(edit_cost_spatial, distance),
    )
    tie = conv(
        "coords",
        np.asarray([63 * 12 / (5 * 1024),
                    63 / (5 * 1024), 0],
                   np.float32).reshape(1, 3, 1, 1),
    )
    score = mul(add(score, tie), cached_goal_presence)
    best_score = gm(score)
    winner = rel(sscale(
        sshift(sub(score, best_score), 1 / 8192), 8192
    ))
    winner = mul(winner, cached_goal_presence)
    selected96 = gm(mul(goal_decode, winner))
    selected_code = dense(selected96, codes.reshape(1, 96))
    selected_cells = gather(winner)

    count_parts = [
        dense(vm(selected96, d), np.ones((1, 96), np.float32))
        for d in (ds, dc, dr)
    ]
    need = clip(vcat(*count_parts))
    needed = [vs(need, k, k + 1) for k in range(3)]
    any_needed = clip(sum_vectors(needed))

    # Track-learning circuit, vectorized over all three tile types.
    # Each track stores velocity and min/max row/column bounds.
    pos_w = parameter(np.stack([rr, cc]))
    positions = [dense(x, pos_w) for x in station_cells]
    presences = [clip(total(x)) for x in station_cells]
    position6 = vcat(*positions)
    present3 = vcat(*presences)
    old_position6 = vs("mem", 144, 150)
    old_present3 = vs("mem", 150, 153)
    old_velocity6 = vm(vs("mem", 170, 176), same_life)

    pair_sum = np.zeros((3, 6), np.float32)
    repeat_pair = np.zeros((6, 3), np.float32)
    group_sum = np.zeros((3, 12), np.float32)
    repeat_four = np.zeros((12, 3), np.float32)
    direction_matrix = np.zeros((12, 6), np.float32)
    duplicate_position = np.zeros((12, 6), np.float32)
    perpendicular = np.zeros((12, 12), np.float32)
    for k in range(3):
        pair_sum[k, 2*k:2*k+2] = 1
        repeat_pair[2*k:2*k+2, k] = 1
        group_sum[k, 4*k:4*k+4] = 1
        repeat_four[4*k:4*k+4, k] = 1
        direction_matrix[4*k, 2*k] = -1
        direction_matrix[4*k+1, 2*k] = 1
        direction_matrix[4*k+2, 2*k+1] = -1
        direction_matrix[4*k+3, 2*k+1] = 1
        duplicate_position[4*k:4*k+2, 2*k] = 1
        duplicate_position[4*k+2:4*k+4, 2*k+1] = 1
        perpendicular[4*k:4*k+2, 4*k+2:4*k+4] = 1
        perpendicular[4*k+2:4*k+4, 4*k:4*k+2] = 1

    pair_sum_w = parameter(pair_sum)
    repeat_pair_w = parameter(repeat_pair)
    group_sum_w = parameter(group_sum)
    repeat_four_w = parameter(repeat_four)
    direction_w = parameter(direction_matrix)
    duplicate_w = parameter(duplicate_position)

    delta6 = vsub(position6, old_position6)
    unit_step3 = eqk(dense(vabs(delta6), pair_sum_w), 1)
    unit_step3 = vm(
        unit_step3, vm(vm(present3, old_present3), same_life)
    )
    unit_step6 = dense(unit_step3, repeat_pair_w)
    velocity6 = va(
        vm(unit_step6, delta6),
        vm(vsub(one, unit_step6), old_velocity6),
    )
    old_direction12 = vr(dense(old_velocity6, direction_w))
    direction12 = vr(dense(velocity6, direction_w))

    reverse = [1, 0, 3, 2]
    left_turn = [3, 2, 0, 1]
    right_turn = [2, 3, 1, 0]

    def direction_permutation(mapping):
        a = np.zeros((12, 12), np.float32)
        for k in range(3):
            for d in range(4):
                a[4*k + mapping[d], 4*k + d] = 1
        return parameter(a)

    reverse_w = direction_permutation(reverse)
    left_w = direction_permutation(left_turn)
    right_w = direction_permutation(right_turn)

    same_direction3 = dense(
        vm(old_direction12, direction12), group_sum_w
    )
    old_moving3 = dense(old_direction12, group_sum_w)
    turn3 = vm(unit_step3, vsub(old_moving3, same_direction3))
    reversal3 = vm(
        unit_step3,
        dense(
            vm(old_direction12, dense(direction12, reverse_w)),
            group_sum_w,
        ),
    )

    initial_bounds = cv([1, 12, 1, 12] * 3)
    bounds12 = vshift(
        va(vs("mem", 176, 188),
           vm(vsub(one, has_memory), initial_bounds)),
        -1,
    )

    # Turning out of a direction identifies that side's boundary.
    # A reversal additionally identifies a one-cell-wide shuttle track.
    boundary_updates = vm(
        dense(turn3, repeat_four_w), old_direction12
    )
    line_updates = vm(
        dense(reversal3, repeat_four_w),
        dense(old_direction12, perpendicular),
    )
    update12 = va(boundary_updates, line_updates)
    bounds12 = va(
        bounds12,
        vm(update12,
           vsub(dense(old_position6, duplicate_w), bounds12)),
    )
    stored_bounds12 = vshift(bounds12, 1)

    # Valid directions within the learned rectangle.
    signed_clearance = vm(
        vsub(dense(position6, duplicate_w), bounds12),
        cv([1, -1, 1, -1] * 3),
    )
    possible12 = clip(signed_clearance)

    # Continue straight, then turn in the simulator's priority order.
    chosen12 = vm(direction12, possible12)
    for perm in (left_w, right_w, reverse_w):
        remaining3 = vsub(one, dense(chosen12, group_sum_w))
        candidate12 = vm(
            vm(dense(direction12, perm), possible12),
            dense(remaining3, repeat_four_w),
        )
        chosen12 = va(chosen12, candidate12)

    chosen_moving3 = dense(chosen12, group_sum_w)
    predicted_stations = []
    for k in range(3):
        cells = station_cells[k]
        moving = vs(chosen_moving3, k, k + 1)
        pred = vm(vsub(one, moving), cells)
        for d in range(4):
            direction_flag = vs(chosen12, 4*k+d, 4*k+d+1)
            pred = va(
                pred, vm(direction_flag, shift(cells, reverse[d]))
            )
        legal_pred = vm(pred, vsub(one, wall))
        legal = clip(total(legal_pred))
        pred = va(vm(legal, legal_pred), vm(vsub(one, legal), cells))
        predicted_stations.append(pred)

    # Goal-conditioned navigation sources and traversal costs.
    useful_sources = [
        vm(predicted_stations[k], needed[k]) for k in range(3)
    ]
    goal_source = vm(selected_cells, vsub(one, any_needed))
    task_sources = clip(sum_vectors(useful_sources + [goal_source]))
    sources_exist = clip(total(task_sources))
    explore = vm(unknown, vsub(one, sources_exist))
    task_sources = clip(va(task_sources, explore))
    task_seed = vscale(task_sources, 1000)
    refill_seed = vscale(refill_cells, 1000)

    harmful = [
        vm(predicted_stations[k], vsub(one, needed[k]))
        for k in range(3)
    ]
    task_cost = vshift(vscale(sum_vectors(harmful), 20), 1)
    negative_task_cost = neg(task_cost)
    negative_refill_cost = c(-1)

    ray_stop = []
    ray_continue = []
    for d in range(4):
        stop = va(shift(solid_for_rays, d), boundary[d])
        ray_stop.append(stop)
        ray_continue.append(vsub(one, stop))
    no_boost = vsub(one, sum_vectors(boost_cells))

    # Parallel value propagation, with rays for forced-motion landings.
    body = []
    old_out = out
    out = body

    def bellman(value, rays, seed, negative_cost):
        adjacent = [shift(value, d) for d in range(4)]
        z = adjacent[0]
        for x in adjacent[1:]:
            z = vmax(z, x)
        z = vr(vmax(va(z, negative_cost), seed))

        next_rays = []
        for d in range(4):
            propagated = vm(ray_continue[d], shift(rays[d], d))
            terminal = vm(ray_stop[d], value)
            next_rays.append(va(propagated, terminal))

        result = vm(no_boost, z)
        for d in range(4):
            result = va(result, vm(boost_cells[d], next_rays[d]))
        return vm(result, traversable), next_rays

    nv, nr = bellman(
        "nv", ["nr0", "nr1", "nr2", "nr3"],
        task_seed, negative_task_cost,
    )
    fv, fr = bellman(
        "fv", ["fr0", "fr1", "fr2", "fr3"],
        refill_seed, negative_refill_cost,
    )
    out = old_out

    states = [
        {"id": "nv", "zeros": [N]},
        {"id": "fv", "zeros": [N]},
    ]
    states += [{"id": "nr%d" % d, "zeros": [N]} for d in range(4)]
    states += [{"id": "fr%d" % d, "zeros": [N]} for d in range(4)]
    nxt = {"nv": nv, "fv": fv}
    nxt.update({"nr%d" % d: nr[d] for d in range(4)})
    nxt.update({"fr%d" % d: fr[d] for d in range(4)})
    graph.append({
        "id": "planning",
        "op": "loop",
        "T": 80,
        "state": states,
        "body": body,
        "next": nxt,
    })

    # Infer the fuel decrement, retaining the last reliable measurement.
    old_dec = vs("mem", 155, 156)
    prior_dec = va(old_dec, vscale(vsub(one, clip(old_dec)), 2))
    delta = vsub(old_fuel, fuel)
    plausible = vm(
        clip(vscale(vshift(delta, -0.5), 2)),
        clip(vscale(vsub(c(2.5), delta), 2)),
    )
    plausible = vm(plausible, clip(old_fuel))
    decrement = va(
        vm(plausible, delta), vm(vsub(one, plausible), prior_dec)
    )
    inverse_dec = vsub(c(1.5), vscale(decrement, 0.5))
    actions_left = vm(fuel, inverse_dec)

    refill_value_here = total(vm(player, "planning.fv"))
    refill_distance = vsub(c(1000), refill_value_here)
    refill_reachable = clip(refill_value_here)
    refuel_now = vm(
        refill_reachable,
        clip(vsub(vshift(refill_distance, 1), actions_left)),
    )
    field = va(
        vm(vsub(one, refuel_now), "planning.nv"),
        vm(refuel_now, "planning.fv"),
    )

    action_values = []
    for d in range(4):
        value = total(vm(player, shift(field, d)))
        legal = total(vm(player, shift(traversable, d)))
        action_values.append(
            vsub(value, vscale(vsub(one, legal), 2000))
        )
    four_values = vcat(*action_values)

    a = np.zeros((8, 4), np.float32)
    a[1:5] = np.eye(4, dtype=np.float32)
    b = np.full(8, -1000000, np.float32)
    b[1:5] = 0
    graph.append({
        "id": "logits", "op": "dense", "in": [four_values],
        "w": parameter(a), "b": parameter(b),
    })
    graph.append({
        "id": "click", "op": "conv", "in": [spatial_one],
        "w": parameter(np.zeros((1, 1, 1, 1), np.float32)),
    })
    graph.append({
        "id": "mem_next", "op": "vconcat",
        "in": [board] + positions + presences
              + [selected_code, fuel, decrement, ui_state,
                 velocity6, stored_bounds12, lives],
    })

    # Pure graph bookkeeping: share identical arrays and expressions.
    parameter_alias = {}
    unique_arrays = {}
    new_params = {}
    for name, array in P.items():
        key = (array.shape, array.tobytes())
        if key in unique_arrays:
            parameter_alias[name] = unique_arrays[key]
        else:
            unique_arrays[key] = name
            parameter_alias[name] = name
            new_params[name] = array

    def freeze(x):
        if isinstance(x, dict):
            return tuple(sorted((k, freeze(v)) for k, v in x.items()))
        if isinstance(x, list):
            return tuple(freeze(v) for v in x)
        return x

    def simplify(nodes, aliases, cache):
        result = []
        for original in nodes:
            nd = dict(original)
            name = nd["id"]
            if nd["op"] == "loop":
                nd["state"] = [dict(s) for s in nd["state"]]
                for s in nd["state"]:
                    if "init" in s:
                        s["init"] = aliases.get(s["init"], s["init"])
                local_aliases = dict(aliases)
                local_cache = dict(cache)
                nd["body"] = simplify(
                    nd["body"], local_aliases, local_cache
                )
                nd["next"] = {
                    k: local_aliases.get(v, v)
                    for k, v in nd["next"].items()
                }
                result.append(nd)
                continue

            nd["in"] = [aliases.get(x, x) for x in nd["in"]]
            for field in ("w", "b"):
                if field in nd:
                    nd[field] = parameter_alias[nd[field]]
            key = freeze({k: v for k, v in nd.items() if k != "id"})
            if name not in ("logits", "click", "mem_next") and key in cache:
                aliases[name] = cache[key]
            else:
                cache[key] = name
                result.append(nd)
        return result

    graph = simplify(graph, {}, {})
    used = set()

    def collect(nodes):
        for nd in nodes:
            for field in ("w", "b"):
                if field in nd:
                    used.add(nd[field])
            if nd["op"] == "loop":
                collect(nd["body"])

    collect(graph)
    new_params = {k: v for k, v in new_params.items() if k in used}
    return {"graph": graph, "memory_size": 189, "params": new_params}
