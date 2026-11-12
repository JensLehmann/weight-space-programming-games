import numpy as np


def build():
    graph = []
    params = {}
    cache = {}
    serial = [0]

    def uid(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def par(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = uid("p")
            params[name] = a
            cache[key] = name
        return cache[key]

    def node(op, ins=None, name=None, **fields):
        name = name or uid()
        d = {"id": name, "op": op}
        if ins is not None:
            d["in"] = list(ins)
        d.update(fields)
        graph.append(d)
        return name

    def conv(x, w, b=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return node("conv", [x], **fields)

    def shared_conv(x, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return node("conv", [x], **fields)

    def sa(x, w, b=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b)

    def va(x, w, b=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return node("dense", [x], **fields)

    def scalar(x, a=1, b=0):
        return sa(x, [[a]], [b])

    def vscalar(x, a=1, b=0):
        return va(x, [[a]], [b])

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vl(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def vadd(a, b):
        return node("vadd", [a, b])

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

    def relu(x):
        return node("relu", [x])

    def vrelu(x):
        return node("vrelu", [x])

    def clip_sp(x, c=1):
        y = relu(x)
        excess = relu(sa(y, np.eye(c), -np.ones(c)))
        return sub(y, excess)

    def clip_v(x, c=1):
        y = vrelu(x)
        excess = vrelu(va(y, np.eye(c), -np.ones(c)))
        return vadd(y, va(excess, -np.eye(c)))

    def complement(x, c=1):
        return va(x, -np.eye(c), np.ones(c))

    def const_v(values):
        values = np.asarray(values, np.float32).reshape(-1)
        return va("avail", np.zeros((len(values), 8)), values)

    def sum_sp(x, c):
        return sa(x, np.ones((1, c)))

    def sum_v(x, c):
        return va(x, np.ones((1, c)))

    def count_sp(x, c):
        return va(node("gmean", [x]), np.eye(c) * 4096)

    def clamp_delta(delta):
        plus = vrelu(va(delta, np.eye(2), [5, 5]))
        minus = vrelu(va(delta, np.eye(2), [-5, -5]))
        return va(
            vcat(plus, minus),
            np.concatenate([np.eye(2), -np.eye(2)], 1),
            [-5, -5],
        )

    one_v = const_v([1])
    row = sa("coords", [[63, 0, 0]])
    col = sa("coords", [[0, 63, 0]])
    xy_sp = sl("coords", 0, 2)

    lower = sub(relu(scalar(row, 1, -9)),
                relu(scalar(row, 1, -10)))
    upper = sub(relu(scalar(row, -1, 63)),
                relu(scalar(row, -1, 62)))
    board = mul(lower, upper)

    header_rows = sub(relu(scalar(row, -1, 10)),
                      relu(scalar(row, -1, 9)))
    header_cols = sub(relu(scalar(col, 1, -23)),
                      relu(scalar(col, 1, -24)))
    header = mul(header_rows, header_cols)

    colors = [10, 6, 15, 11, 12, 8, 9, 7, 14]
    sizes = [1, 2, 3, 4, 5, 7, 8, 9, 10]

    # Square-fruit top-left candidates.
    w = np.zeros((9, 16, 3, 3), np.float32)
    b = np.zeros(9, np.float32)
    for t, color in enumerate(colors):
        if t == 0:
            w[t, color, 1, 1] = 1
        else:
            for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                w[t, color, 1 + dr, 1 + dc] += 1
            for dr, dc in [(-1, 0), (0, -1), (-1, 1), (1, -1)]:
                w[t, color, 1 + dr, 1 + dc] -= 4
            b[t] = -3
    tl_candidate = clip_sp(conv("x0", w, b), 9)

    # Verify the full tier-specific extent.
    cpw = np.zeros((9, 16), np.float32)
    for t, color in enumerate(colors):
        cpw[t, color] = 1
    color_maps = sa("x0", cpw)
    corner_samples = []
    for direction in [(0, 1), (1, 0), (1, 1)]:
        shifted = color_maps
        for stage in range(3):
            sw = np.zeros((9, 9, 7, 7), np.float32)
            for t, n in enumerate(sizes):
                amount = min(3, max(0, n - 1 - 3 * stage))
                dr = amount * direction[0]
                dc = amount * direction[1]
                sw[t, t, 3 + dr, 3 + dc] = 1
            shifted = conv(shifted, sw)
        corner_samples.append(shifted)
    corner_sum = add(add(corner_samples[0], corner_samples[1]),
                     corner_samples[2])
    tl = clip_sp(sa(add(tl_candidate, corner_sum),
                    np.eye(9), -3 * np.ones(9)), 9)

    # Simulator-style integer centers.
    w = np.zeros((9, 9, 7, 7), np.float32)
    w2 = np.zeros((9, 9, 5, 5), np.float32)
    for t, n in enumerate(sizes):
        d = n // 2
        first = min(d, 3)
        second = d - first
        w[t, t, 3 - first, 3 - first] = 1
        w2[t, t, 2 - second, 2 - second] = 1
    fruit_centers = conv(conv(tl, w), w2)

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
    ew = np.zeros((3, 16, 5, 5), np.float32)
    eb = np.zeros(3, np.float32)
    for t, (pattern, color) in enumerate(zip(patterns, [7, 14, 13])):
        total = 0
        for r in range(4):
            for c in range(5):
                if pattern[r][c]:
                    ew[t, color, r, c] = 1
                    total += 1
                else:
                    ew[t, color, r, c] = -2
        eb[t] = 1 - total
    enemy_centers = clip_sp(conv("x0", ew, eb), 3)

    raw_types = cat(fruit_centers, enemy_centers)
    types = mul(raw_types, board)
    requirements = count_sp(mul(raw_types, header), 12)
    counts = count_sp(types, 12)
    req_present = clip_v(requirements, 12)
    red_header = node("gmax", [mul(sl("x0", 2, 3), header)])

    # Four orientations of a goal-circle edge detector.
    def rotate(dr, dc, t):
        for _ in range(t):
            dr, dc = dc, -dr
        return dr, dc

    gw = np.zeros((4, 16, 7, 7), np.float32)
    positive = [(0, dc) for dc in range(-2, 3)] + [(1, -3), (1, 3)]
    negative = [(0, -3), (0, 3), (-1, 0)]
    for t in range(4):
        for color in [9, 2]:
            for dr, dc in positive:
                dr, dc = rotate(dr, dc, t)
                gw[t, color, 3 + dr, 3 + dc] = 1
            for dr, dc in negative:
                dr, dc = rotate(dr, dc, t)
                gw[t, color, 3 + dr, 3 + dc] = -8
    caps = clip_sp(conv("x0", gw, [-6] * 4), 4)

    sw = np.zeros((4, 4, 7, 7), np.float32)
    sw2 = np.zeros((4, 4, 3, 3), np.float32)
    for t in range(4):
        dr, dc = rotate(3, 0, t)
        sw[t, t, 3 - dr, 3 - dc] = 1
        dr, dc = rotate(1, 0, t)
        sw2[t, t, 1 - dr, 1 - dc] = 1
    live_goals = mul(
        clip_sp(sum_sp(conv(conv(caps, sw), sw2), 4)),
        board,
    )

    # Persistent goal locations indexed by 8x8 spatial bins.
    bin_w = np.zeros((16, 3), np.float32)
    bin_b = np.zeros(16, np.float32)
    for i in range(8):
        bin_w[i, 0] = 63
        bin_w[8 + i, 1] = 63
        bin_b[i] = 1 - 8 * i
        bin_b[8 + i] = 1 - 8 * i
    bin_low = clip_sp(sa("coords", bin_w, bin_b), 16)
    bin_high = clip_sp(sa("coords", bin_w, bin_b - 8), 16)
    intervals = sub(bin_low, bin_high)
    join_w = np.zeros((64, 16), np.float32)
    for r in range(8):
        for c in range(8):
            join_w[8 * r + c, r] = 1
            join_w[8 * r + c, 8 + c] = 1
    bins = relu(sa(intervals, join_w, -np.ones(64)))
    observed_bins = mul(bins, live_goals)
    observed_presence = clip_v(count_sp(observed_bins, 64), 64)
    observed_rows = count_sp(
        mul(observed_bins, sl("coords", 0, 1)), 64
    )
    observed_cols = count_sp(
        mul(observed_bins, sl("coords", 1, 2)), 64
    )

    previous_presence = vl("mem", 10, 74)
    empty_bins = complement(previous_presence, 64)
    goal_presence = vadd(
        previous_presence, vmul(observed_presence, empty_bins)
    )
    goal_rows = vadd(
        vl("mem", 74, 138), vmul(observed_rows, empty_bins)
    )
    goal_cols = vadd(
        vl("mem", 138, 202), vmul(observed_cols, empty_bins)
    )
    goal_r_pixels = va(goal_rows, 63 * np.eye(64))
    goal_c_pixels = va(goal_cols, 63 * np.eye(64))
    gr = node("abs", [sub(row, goal_r_pixels)])
    gc = node("abs", [sub(col, goal_c_pixels)])
    remembered_goals = mul(
        clip_sp(sa(add(gr, gc), -4 * np.eye(64), np.ones(64)), 64),
        goal_presence,
    )
    goals = clip_sp(sum_sp(remembered_goals, 64))

    goal_box = clip_sp(conv(goals, np.ones((1, 1, 7, 7))))
    goal_box = clip_sp(conv(goal_box, np.ones((1, 1, 3, 3))))
    inside_types = mul(types, goal_box)
    inside_counts = count_sp(inside_types, 12)
    reserved_centers = mul(inside_types, req_present)
    candidates = relu(sub(types, reserved_centers))
    enemies = clip_sp(sum_sp(sl(types, 9, 12), 3))

    # Max-plus propagation: 200 minus Manhattan distance to the seed.
    dw = np.zeros((5, 1, 3, 3), np.float32)
    for i, (dr, dc) in enumerate(
        [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    ):
        dw[i, 0, 1 + dr, 1 + dc] = 1
    dw_name = par(dw)
    db_name = par([-1, -1, -1, -1, 0])

    def distance_field(seed):
        initial = scalar(seed, 200)
        loop_id = uid("D")
        state_id = uid("s")
        conv_id = uid()
        body = [{
            "id": conv_id, "op": "conv", "in": [state_id],
            "w": dw_name, "b": db_name,
        }]
        pieces = []
        for i in range(5):
            q = uid()
            body.append({
                "id": q, "op": "slice", "in": [conv_id],
                "start": i, "stop": i + 1,
            })
            pieces.append(q)
        current = pieces[0]
        for q in pieces[1:]:
            nxt = uid()
            body.append({
                "id": nxt, "op": "max2", "in": [current, q],
            })
            current = nxt
        graph.append({
            "id": loop_id, "op": "loop", "T": 126,
            "state": [{"id": state_id, "init": initial}],
            "body": body, "next": {state_id: current},
        })
        return loop_id + "." + state_id

    goal_distance = distance_field(goals)
    enemy_distance = distance_field(enemies)

    # Transformation and delivery requirements.
    missing = clip_v(va(
        vcat(requirements, counts),
        np.concatenate([np.eye(12), -np.eye(12)], 1),
    ), 12)
    excess_in = vcat(counts, requirements)
    excess = clip_v(va(
        excess_in, np.concatenate([np.eye(12), -np.eye(12)], 1),
    ), 12)
    pair_supply = clip_v(va(
        excess_in, np.concatenate([np.eye(12), -np.eye(12)], 1),
        -np.ones(12),
    ), 12)

    above_f = np.zeros((9, 12), np.float32)
    below_f = np.zeros((9, 12), np.float32)
    above_e = np.zeros((3, 12), np.float32)
    for t in range(9):
        above_f[t, t + 1:9] = 1
        below_f[t, :t] = 1
    for t in range(3):
        above_e[t, 10 + t:12] = 1

    fruit_merge = vmul(
        vl(pair_supply, 0, 9), clip_v(va(missing, above_f), 9)
    )
    enemy_merge = vmul(
        vl(pair_supply, 9, 12), clip_v(va(missing, above_e), 3)
    )
    enemy_exists = clip_v(sum_v(vl(counts, 9, 12), 3))
    downgrade = vmul(
        vl(excess, 0, 9), clip_v(va(missing, below_f), 9)
    )
    downgrade = vmul(downgrade, enemy_exists)

    transport_missing = clip_v(va(
        vcat(requirements, inside_counts),
        np.concatenate([np.eye(12), -np.eye(12)], 1),
    ), 12)
    transport = vmul(transport_missing, clip_v(counts, 12))

    reverse9 = np.eye(9, dtype=np.float32)[::-1]
    reverse3 = np.eye(3, dtype=np.float32)[::-1]
    options = vcat(
        va(fruit_merge, reverse9),
        va(enemy_merge, reverse3),
        downgrade,
        va(vl(transport, 0, 9), reverse9),
        va(vl(transport, 9, 12), reverse3),
    )
    priority = np.eye(33, dtype=np.float32)
    priority -= np.tril(np.ones((33, 33), np.float32), -1)
    chosen_option = vrelu(va(options, priority))

    option_types = (
        list(range(8, -1, -1)) +
        list(range(11, 8, -1)) +
        list(range(9)) +
        list(range(8, -1, -1)) +
        list(range(11, 8, -1))
    )
    aw = np.zeros((12, 33), np.float32)
    for i, t in enumerate(option_types):
        aw[t, i] = 1
    active = va(chosen_option, aw)

    def option_flag(a, b):
        w = np.zeros((1, 33), np.float32)
        w[0, a:b] = 1
        return va(chosen_option, w)

    merging = option_flag(0, 12)
    lowering = option_flag(12, 21)
    delivering = option_flag(21, 33)
    not_merging = complement(merging)
    not_lowering = complement(lowering)
    fruit_phase = sum_v(vl(active, 0, 9), 9)
    enemy_phase = sum_v(vl(active, 9, 12), 3)
    no_task = complement(sum_v(chosen_option, 33))
    active_candidates = clip_sp(sum_sp(mul(candidates, active), 12))

    # Bounded retry preference; unchanged on the first attempt.
    retry_count = vadd(vl("mem", 203, 204), vl("aprev", 0, 1))
    retry_preference = clip_v(vscalar(retry_count, .5))
    retry_geometry = mul(
        sa("coords", [[.7, 1.4, 0]]), retry_preference
    )

    # Piecewise-linear reciprocal for spatial-point extraction.
    knots = np.arange(129, dtype=np.float32)
    reciprocal_w = par(np.ones((129, 1), np.float32))
    reciprocal_b = par(-knots)
    values = np.ones(130, np.float64)
    values[1:] = 1 / np.arange(1, 130)
    slopes = np.diff(values)
    rw = np.zeros(129, np.float32)
    rw[0] = slopes[0]
    for i in range(1, 128):
        rw[i] = slopes[i] - slopes[i - 1]
    rw[128] = -slopes[127]
    reciprocal_out = par(rw[None])
    reciprocal_bias = par([1])
    lex = sa("coords", [[-.0126, -.000126, 0]])

    def select_point(score, mask):
        score = add(score, lex)
        maximum = node("gmax", [score])
        near = clip_sp(scalar(sub(score, maximum), 1048576, 1))
        mark = mul(near, mask)
        count = count_sp(mark, 1)
        hinges = node(
            "dense", [count], w=reciprocal_w, b=reciprocal_b
        )
        hinges = vrelu(hinges)
        reciprocal = node(
            "dense", [hinges],
            w=reciprocal_out, b=reciprocal_bias,
        )
        coordinates = count_sp(mul(mark, xy_sp), 2)
        return mark, vmul(coordinates, reciprocal)

    def point_distance(point):
        rp = vscalar(vl(point, 0, 1), 63)
        cp = vscalar(vl(point, 1, 2), 63)
        dr = node("abs", [sub(row, rp)])
        dc = node("abs", [sub(col, cp)])
        return add(dr, dc)

    source_score = scalar(active_candidates, 10)
    source_score = add(
        source_score, mul(scalar(goal_distance, .01), not_lowering)
    )
    source_score = add(
        source_score, mul(scalar(enemy_distance, .01), lowering)
    )
    risk = relu(scalar(enemy_distance, 1, -184))
    safe_fruit = vmul(fruit_phase, not_lowering)
    source_score = sub(
        source_score, mul(scalar(risk, .08), safe_fruit)
    )
    source_score = add(source_score, retry_geometry)
    source_mark, source_point = select_point(
        source_score, active_candidates
    )

    distance_source = point_distance(source_point)
    partner_mask = clip_sp(sub(active_candidates, source_mark))
    partner_score = sub(
        scalar(partner_mask, 10), scalar(distance_source, .01)
    )
    partner_mark, partner_point = select_point(
        partner_score, partner_mask
    )

    enemy_score = sub(
        scalar(enemies, 10), scalar(distance_source, .01)
    )
    enemy_mark, enemy_point = select_point(enemy_score, enemies)

    done = clip_sp(sum_sp(inside_types, 12))
    occupied = clip_sp(conv(done, np.ones((1, 1, 7, 7))))
    occupied = clip_sp(conv(occupied, np.ones((1, 1, 3, 3))))
    goal_score = sub(
        scalar(goals, 10), scalar(distance_source, .01)
    )
    goal_score = sub(goal_score, scalar(occupied, 5))
    goal_risk = relu(scalar(enemy_distance, 1, -180))
    goal_score = sub(
        goal_score, mul(scalar(goal_risk, .04), fruit_phase)
    )
    goal_score = add(goal_score, retry_geometry)
    goal_mark, goal_point = select_point(goal_score, goals)

    mover_point = vadd(
        vmul(partner_point, merging),
        vmul(source_point, not_merging),
    )
    mover_mark = add(
        mul(partner_mark, merging),
        mul(source_mark, not_merging),
    )
    destination = vadd(
        vadd(vmul(source_point, merging),
             vmul(enemy_point, lowering)),
        vmul(goal_point, delivering),
    )
    delta = va(
        vcat(destination, mover_point),
        np.concatenate([63 * np.eye(2), -63 * np.eye(2)], 1),
    )
    desired_point = va(
        vadd(va(mover_point, 63 * np.eye(2)), clamp_delta(delta)),
        np.eye(2) / 63,
    )
    desired_distance = point_distance(desired_point)

    # Bounding-box reconstruction.
    bw1 = np.zeros((9, 9, 7, 7), np.float32)
    bw2 = np.zeros((9, 9, 5, 5), np.float32)
    for t, n in enumerate(sizes):
        first = min(n, 7)
        second = n - first + 1
        for dr in range(-(first // 2), (first - 1) // 2 + 1):
            for dc in range(-(first // 2), (first - 1) // 2 + 1):
                bw1[t, t, 3 - dr, 3 - dc] = 1
        for dr in range(-(second // 2), (second - 1) // 2 + 1):
            for dc in range(-(second // 2), (second - 1) // 2 + 1):
                bw2[t, t, 2 - dr, 2 - dc] = 1
    bw1_name, bw2_name = par(bw1), par(bw2)

    def fruit_boxes(centers):
        x = clip_sp(shared_conv(centers, bw1_name), 9)
        return clip_sp(shared_conv(x, bw2_name), 9)

    fruit_pixels = fruit_boxes(sl(types, 0, 9))
    reserved_pixels = fruit_boxes(sl(reserved_centers, 0, 9))

    ebox_w = np.zeros((3, 3, 5, 5), np.float32)
    single_ebox = np.zeros((1, 1, 5, 5), np.float32)
    for dr in range(-2, 2):
        for dc in range(-2, 3):
            single_ebox[0, 0, 2 - dr, 2 - dc] = 1
            for t in range(3):
                ebox_w[t, t, 2 - dr, 2 - dc] = 1
    ebox_name = par(ebox_w)
    enemy_pixels = clip_sp(
        shared_conv(sl(types, 9, 12), ebox_name), 3
    )

    bad_fruit = add(
        mul(fruit_pixels, complement(vl(active, 0, 9), 9)),
        reserved_pixels,
    )
    bad_enemy = mul(
        mul(enemy_pixels, complement(vl(active, 9, 12), 3)),
        enemy_phase,
    )
    bad_enemy = add(bad_enemy, mul(enemy_pixels, lowering))
    bad_pixels = clip_sp(add(
        sum_sp(bad_fruit, 9), sum_sp(bad_enemy, 3)
    ))

    # Equal-tier objects are not merge partners during delivery.
    selected_types = mul(types, mover_mark)
    other_active = relu(sub(mul(types, active), selected_types))
    equal_fruit_pixels = fruit_boxes(sl(other_active, 0, 9))
    equal_enemy_pixels = clip_sp(
        shared_conv(sl(other_active, 9, 12), ebox_name), 3
    )
    equal_obstacles = clip_sp(add(
        sum_sp(equal_fruit_pixels, 9),
        sum_sp(equal_enemy_pixels, 3),
    ))
    bad_pixels = clip_sp(add(
        bad_pixels, mul(equal_obstacles, delivering)
    ))

    all_fruit_pixels = clip_sp(sum_sp(fruit_pixels, 9))
    all_enemy_pixels = clip_sp(sum_sp(enemy_pixels, 3))
    all_pixels = clip_sp(add(all_fruit_pixels, all_enemy_pixels))

    dilation_cache = {}

    def dilate8(x, conservative=False):
        radii = [
            (3.2 if conservative else 3, 7),
            (3.2 if conservative else 3, 7),
            (2.3 if conservative else 2, 5),
        ]
        for radius, k in radii:
            key = (radius, k)
            if key not in dilation_cache:
                w = np.zeros((1, 1, k, k), np.float32)
                h = k // 2
                for dr in range(-h, h + 1):
                    for dc in range(-h, h + 1):
                        if dr * dr + dc * dc <= radius * radius:
                            w[0, 0, h + dr, h + dc] = 1
                dilation_cache[key] = par(w)
            x = clip_sp(shared_conv(x, dilation_cache[key]))
        return x

    foreign_circle = dilate8(bad_pixels, True)
    all_circle = dilate8(all_pixels, True)
    mover_circle = dilate8(mover_mark)

    near_enemy_value = count_sp(mul(enemy_distance, source_mark), 1)
    near_enemy = clip_v(vscalar(near_enemy_value, 1, -184))
    waiting = clip_v(vadd(vmul(near_enemy, lowering), no_task))

    # Protective displacement of a threatening enemy.
    thresholds = 13 + np.asarray(sizes, np.float32) // 2
    near_each = sa(enemy_distance, np.ones((9, 1)), thresholds - 200)
    cooldown_active = clip_v(vl("mem", 9, 10))
    near_each = sub(near_each, vscalar(cooldown_active, 3))
    near_each = clip_sp(near_each, 9)
    protected = complement(downgrade, 9)
    threatened = mul(mul(sl(types, 0, 9), protected), near_each)
    threat_mask = clip_sp(sum_sp(threatened, 9))
    emergency = node("gmax", [threat_mask])

    threat_score = add(
        scalar(threat_mask, 10), scalar(enemy_distance, .01)
    )
    threat_mark, threat_point = select_point(threat_score, threat_mask)
    threat_distance = point_distance(threat_point)
    flee_enemy_score = sub(
        scalar(enemies, 10), scalar(threat_distance, .01)
    )
    flee_enemy_mark, flee_enemy_point = select_point(
        flee_enemy_score, enemies
    )
    flee_delta = va(
        vcat(flee_enemy_point, threat_point),
        np.concatenate([63 * np.eye(2), -63 * np.eye(2)], 1),
    )
    flee_abs = vrelu(vcat(flee_delta, va(flee_delta, -np.eye(2))))
    coincident = clip_v(vscalar(sum_v(flee_abs, 4), -1, 1))
    fallback = va(coincident, [[0], [5]])
    flee_step = vadd(clamp_delta(flee_delta), fallback)
    flee_target = va(
        vadd(va(flee_enemy_point, 63 * np.eye(2)), flee_step),
        np.eye(2) / 63,
    )
    flee_distance = point_distance(flee_target)
    flee_circle = dilate8(flee_enemy_mark)

    selected_enemy_pixels = clip_sp(conv(flee_enemy_mark, single_ebox))
    other_enemy_pixels = clip_sp(
        sub(all_enemy_pixels, selected_enemy_pixels)
    )
    other_enemy_circle = dilate8(other_enemy_pixels, True)
    fruit_circle = dilate8(all_fruit_pixels, True)

    emergency_score = sub(scalar(flee_circle, 1000), flee_distance)
    emergency_score = sub(
        emergency_score, scalar(other_enemy_circle, 500)
    )
    emergency_score = sub(
        emergency_score, scalar(fruit_circle, 200)
    )

    # State fingerprints recognize unchanged and automatically undone clicks.
    weighted = sa(types, [np.arange(1, 13, dtype=np.float32)])
    fingerprint_sp = cat(
        weighted,
        mul(weighted, sl("coords", 0, 1)),
        mul(weighted, sl("coords", 1, 2)),
    )
    fingerprint = va(node("gmean", [fingerprint_sp]), 64 * np.eye(3))
    difference = va(
        vcat(fingerprint, vl("mem", 0, 3)),
        np.concatenate([np.eye(3), -np.eye(3)], 1),
    )
    absolute_parts = vrelu(
        vcat(difference, va(difference, -np.eye(3)))
    )
    discrepancy = sum_v(absolute_parts, 6)
    unchanged = clip_v(vscalar(discrepancy, -100000, 1))
    unchanged = vmul(unchanged, vl("mem", 5, 6))
    unchanged = vmul(unchanged, vl("aprev", 6, 7))
    unchanged = vmul(unchanged, complement(waiting))

    ban_trigger = clip_v(vadd(red_header, unchanged))
    banned_point = vadd(
        vmul(vl("mem", 3, 5), ban_trigger),
        vmul(vl("mem", 6, 8), complement(ban_trigger)),
    )
    decayed_ban = vrelu(vscalar(vl("mem", 8, 9), 1, -.5))
    ban_strength = vadd(
        vscalar(ban_trigger, 4),
        vmul(decayed_ban, complement(ban_trigger)),
    )
    banned_distance = point_distance(banned_point)
    ban_penalty = mul(
        relu(scalar(banned_distance, -1, 6)), ban_strength
    )

    # A second taboo region prevents alternating between two bad clicks.
    older_point = vadd(
        vmul(vl("mem", 6, 8), ban_trigger),
        vmul(vl("mem", 204, 206), complement(ban_trigger)),
    )
    older_strength = vadd(
        vmul(vl("mem", 8, 9), ban_trigger),
        vmul(vrelu(vscalar(vl("mem", 206, 207), 1, -.5)),
             complement(ban_trigger)),
    )
    older_distance = point_distance(older_point)
    older_penalty = mul(
        relu(scalar(older_distance, -1, 6)), older_strength
    )

    main_score = scalar(desired_distance, -1)
    main_score = add(main_score, scalar(mover_circle, 200))
    strict_interference = clip_v(vadd(delivering, enemy_phase))
    interference_weight = vscalar(strict_interference, 410, 90)
    main_score = sub(
        main_score, mul(foreign_circle, interference_weight)
    )

    wait_score = sa("coords", [[-.063, -.0063, 0]])
    wait_score = sub(wait_score, scalar(all_circle, 100))

    ordinary = add(
        mul(main_score, complement(waiting)),
        mul(wait_score, waiting),
    )
    mixed = add(
        mul(ordinary, complement(emergency)),
        mul(emergency_score, emergency),
    )
    mixed = sub(mixed, scalar(add(ban_penalty, older_penalty), 6))
    mixed = add(mixed, lex)
    valid_penalty = scalar(board, 2000, -2000)
    node("add", [mixed, valid_penalty], name="click")

    click_mark, click_point = select_point("click", board)
    stored_click = vadd(
        vmul(vl("mem", 3, 5), red_header),
        vmul(click_point, complement(red_header)),
    )
    cooldown = vadd(
        vscalar(emergency, 2),
        vmul(vrelu(vscalar(vl("mem", 9, 10), 1, -1)),
             complement(emergency)),
    )

    # Recovery after repeated unsuccessful undo/retry transactions.
    previous_red_count = vmul(
        vl("mem", 202, 203), complement(vl("aprev", 0, 1))
    )
    red_count = vrelu(vadd(
        previous_red_count, vscalar(red_header, 1.15, -.15)
    ))
    reset_request = vmul(
        red_header, clip_v(vscalar(red_count, 2, -5))
    )

    node(
        "vconcat",
        [fingerprint, stored_click, one_v,
         banned_point, ban_strength, cooldown,
         goal_presence, goal_rows, goal_cols, red_count,
         retry_count, older_point, older_strength],
        name="mem_next",
    )

    base = const_v([-4, -4, -4, -4, -4, -4, 2, 0])
    undo_w = np.zeros((8, 1), np.float32)
    undo_w[7, 0] = 10
    reset_w = np.zeros((8, 1), np.float32)
    reset_w[0, 0] = 30
    action_scores = vadd(
        vadd(base, va(red_header, undo_w)),
        va(reset_request, reset_w),
    )
    node("vadd", [action_scores, const_v([0] * 8)], name="logits")

    return {"graph": graph, "memory_size": 207, "params": params}
