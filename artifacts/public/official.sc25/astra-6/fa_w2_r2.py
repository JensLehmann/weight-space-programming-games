import numpy as np


def build():
    params = {}
    graph = []
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (8,),
        "aprev": (8,),
        "avail": (8,),
    }
    serial = [0]
    destination = [graph]

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def node(op, ins, name=None, **fields):
        name = name or fresh()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        destination[0].append(nd)
        ss = [shapes[i] for i in ins]

        if op == "conv":
            sh = (params[fields["w"]].shape[0], 64, 64)
        elif op == "dense":
            sh = (params[fields["w"]].shape[0],)
        elif op in ("slice", "vslice"):
            sh = (fields["stop"] - fields["start"],) + ss[0][1:]
        elif op in ("concat", "vconcat"):
            sh = (sum(s[0] for s in ss),) + ss[0][1:]
        elif op in ("gmax", "gmean"):
            sh = (ss[0][0],)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            sh = (max(s[0] for s in ss), 64, 64)
        elif op in ("vadd", "vmul"):
            sh = (max(s[0] for s in ss),)
        else:
            sh = ss[0]
        shapes[name] = sh
        return name

    def cv(x, w, b=None, name=None):
        ident = name or fresh()
        wk = ident + "_w"
        params[wk] = np.asarray(w, np.float32)
        kw = {"w": wk}
        if b is not None:
            bk = ident + "_b"
            a = np.asarray(b, np.float32)
            if a.ndim == 0:
                a = np.full(params[wk].shape[0], a, np.float32)
            params[bk] = a
            kw["b"] = bk
        return node("conv", [x], name=ident, **kw)

    def linear(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if len(shapes[x]) == 3:
            return cv(x, w[:, :, None, None], b, name)
        ident = name or fresh()
        wk = ident + "_w"
        params[wk] = w
        kw = {"w": wk}
        if b is not None:
            bk = ident + "_b"
            a = np.asarray(b, np.float32)
            if a.ndim == 0:
                a = np.full(w.shape[0], a, np.float32)
            params[bk] = a
            kw["b"] = bk
        return node("dense", [x], name=ident, **kw)

    def sl(x, a, b):
        return node(
            "slice" if len(shapes[x]) == 3 else "vslice",
            [x], start=a, stop=b
        )

    def cat(xs, name=None):
        return node(
            "concat" if len(shapes[xs[0]]) == 3 else "vconcat",
            xs, name=name
        )

    def relu(x):
        return node("relu" if len(shapes[x]) == 3 else "vrelu", [x])

    def add(a, b):
        spatial = max(len(shapes[a]), len(shapes[b])) == 3
        return node("add" if spatial else "vadd", [a, b])

    def mul(a, b):
        spatial = max(len(shapes[a]), len(shapes[b])) == 3
        return node("mul" if spatial else "vmul", [a, b])

    def scale(x, value, bias=0.0):
        w = np.eye(shapes[x][0], dtype=np.float32) * value
        return linear(x, w, bias)

    def minus(a, b):
        if max(len(shapes[a]), len(shapes[b])) == 3:
            return node("sub", [a, b])
        return add(a, scale(b, -1))

    def complement(x):
        return scale(x, -1, 1)

    def maximum(a, b):
        return node("max2", [a, b])

    def minimum(a, b):
        return node("min2", [a, b])

    def total(x):
        return linear(x, np.ones((1, shapes[x][0]), np.float32))

    def gmax(x):
        return node("gmax", [x])

    def cap_vector(x):
        return minus(x, relu(scale(x, 1, -1)))

    def pool_four(x):
        z = sl(x, 0, 1)
        for j in range(1, 4):
            z = maximum(z, sl(x, j, j + 1))
        return z

    def shift(x, offsets):
        c = shapes[x][0]
        o = len(offsets)
        radius = max(max(abs(y), abs(xx)) for y, xx in offsets)
        k = 2 * radius + 1
        w = np.zeros((o, c, k, k), np.float32)
        for i, (dy, dx) in enumerate(offsets):
            w[i, 0 if c == 1 else i, radius + dy, radius + dx] = 1
        return cv(x, w)

    def sum_pixels(x, offsets):
        r = max(max(abs(y), abs(xx)) for y, xx in offsets)
        w = np.zeros((1, 1, 2 * r + 1, 2 * r + 1), np.float32)
        for dy, dx in offsets:
            w[0, 0, r + dy, r + dx] = 1
        return cv(x, w)

    def all_pixels(x, offsets):
        return relu(scale(sum_pixels(x, offsets), 1, 1 - len(offsets)))

    def expand_anchor(x, offsets):
        return sum_pixels(x, [(-dy, -dx) for dy, dx in offsets])

    def first_one(x):
        c = shapes[x][0]
        w = np.eye(c, dtype=np.float32)
        w -= np.tril(np.ones((c, c), np.float32), -1)
        return relu(linear(x, w))

    ones = sl("coords", 2, 3)
    zero = scale(ones, 0)

    def cap_space(x):
        return minimum(x, ones)

    def colour(c):
        return sl("x0", c, c + 1)

    def colours(cs):
        w = np.zeros((1, 16), np.float32)
        w[0, cs] = 1
        return linear("x0", w)

    # Detect spell icons by their local shapes, not their screen positions.
    fire_icon_offsets = [
        (y - 3, x) for y in (0, 1, 3, 4, 6) for x in (0, 1)
    ]
    resize_icon_offsets = [
        (y - 3, x - 3)
        for y, x in (
            (0, 3), (0, 4), (1, 3), (1, 4),
            (3, 0), (3, 1), (4, 0), (4, 1),
            (3, 6), (4, 6), (6, 3), (6, 4),
        )
    ]
    teleport_icon_offsets = [
        (y - 2, x - 2)
        for y, x in (
            (0, 0), (0, 1), (0, 3), (0, 4),
            (1, 0), (1, 1), (1, 3), (1, 4),
            (3, 3), (3, 4), (4, 3), (4, 4),
        )
    ]
    can_fire = gmax(all_pixels(colour(6), fire_icon_offsets))
    can_resize = gmax(all_pixels(colour(15), resize_icon_offsets))
    can_teleport = gmax(all_pixels(colour(11), teleport_icon_offsets))

    # Exit recognition removes exit-coloured pixels from player detection.
    exit_template = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ], dtype=np.int32)

    nine_ten = sl("x0", 9, 11)
    ew = np.zeros((4, 2, 7, 7), np.float32)
    ex = np.zeros((1, 4, 7, 7), np.float32)
    for r in range(4):
        t = np.rot90(exit_template, -r)
        for y in range(t.shape[0]):
            for x in range(t.shape[1]):
                dy, dx = y - 2, x - 2
                ew[r, t[y, x] - 9, 3 + dy, 3 + dx] = 1
                ex[0, r, 3 - dy, 3 - dx] = 1
    exit_anchors = relu(cv(nine_ten, ew, -29))
    exit_pixels = cap_space(cv(exit_anchors, ex))
    player_colours = mul(nine_ten, complement(exit_pixels))

    # Player anchor is one pixel below/right of its upper-left corner.
    rotations = [180, 0, 90, 270]
    base_player = np.array([[10, 10], [9, 9]], np.int32)

    def player_templates(size):
        w = np.zeros((4, 2, 7, 7), np.float32)
        for d, rot in enumerate(rotations):
            t = np.rot90(base_player, -(rot // 90))
            t = np.repeat(np.repeat(t, size, 0), size, 1)
            for y in range(t.shape[0]):
                for x in range(t.shape[1]):
                    w[d, t[y, x] - 9, y + 2, x + 2] = 1
        return relu(cv(player_colours, w, 1 - 4 * size * size))

    big_orient = player_templates(2)
    big_player = total(big_orient)
    is_big = gmax(big_player)
    small_orient = mul(player_templates(1), complement(is_big))
    small_player = total(small_orient)
    is_small = gmax(small_player)
    player_maps = [small_player, big_player]

    # Exclude the canonical spell-entry and spell-icon interface regions.
    q = linear("coords", np.array([
        [-126, 0, 93],
        [0, 126, -19],
        [126, 0, -65],
    ], np.float32))
    q = cap_space(relu(q))
    play_area = mul(
        sl(q, 0, 1),
        maximum(sl(q, 1, 2), sl(q, 2, 3))
    )

    footprint = {
        2: [(y, x) for y in range(-1, 1) for x in range(-1, 1)],
        4: [(y, x) for y in range(-1, 3) for x in range(-1, 3)],
    }

    def key_detector(centre_colour):
        w = np.zeros((1, 16, 5, 5), np.float32)
        for y in range(4):
            for x in range(4):
                c = centre_colour if y in (1, 2) and x in (1, 2) else 6
                w[0, c, y + 1, x + 1] = 1
        a = relu(cv("x0", w, -15))
        return cap_space(expand_anchor(a, footprint[4]))

    keys = [key_detector(13), key_detector(12)]
    gate_colours = [colour(13), colour(12)]
    groups = [maximum(keys[t], gate_colours[t]) for t in range(2)]
    gates = [
        mul(gate_colours[t], complement(keys[t])) for t in range(2)
    ]

    c11 = colour(11)
    c15 = colour(15)
    corners4 = [(-1, -1), (-1, 2), (2, -1), (2, 2)]
    corners_big_indicator = [(-2, -2), (-2, 3), (3, -2), (3, 3)]
    corners_small_indicator = [(-3, -3), (-3, 2), (2, -3), (2, 2)]

    big_pad = all_pixels(c11, corners4)
    big_indicator = all_pixels(c11, corners_big_indicator)
    near_small = cap_space(sum_pixels(
        c15, [(y, x) for y in range(-2, 4) for x in range(-2, 4)]
    ))
    big_pad = mul(big_pad, complement(near_small))
    big_pad = mul(big_pad, play_area)
    big_indicator = mul(big_indicator, play_area)

    small_pad = mul(all_pixels(c15, footprint[2]), play_area)
    small_indicator = mul(
        all_pixels(c15, corners_small_indicator), play_area
    )

    def prefer_indicator(pad, indicator):
        return add(indicator, mul(pad, complement(gmax(indicator))))

    portals = [
        mul(prefer_indicator(small_pad, small_indicator), can_teleport),
        mul(prefer_indicator(big_pad, big_indicator), can_teleport),
    ]

    # Separate movement, ray, and growth collision semantics.
    all_blocking = colours([0, 1, 3, 4, 5, 6, 7, 8, 12, 13])
    fixed_movement = add(
        minus(minus(all_blocking, groups[0]), groups[1]),
        exit_pixels
    )
    fixed_ray_growth = colours([0, 1, 3, 5, 7, 8])

    def free_footprint(obstacles, width):
        inp = cat([obstacles, ones])
        k = 3 if width == 2 else 5
        r = k // 2
        w = np.zeros((1, 2, k, k), np.float32)
        for dy, dx in footprint[width]:
            w[0, 0, r + dy, r + dx] = -1
            w[0, 1, r + dy, r + dx] = 1
        return relu(cv(inp, w, 1 - width * width))

    dirs2 = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    dirs1 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    grow_offsets = [
        (dy, dx) for dx in range(-2, 1) for dy in range(-2, 1)
    ]

    free_maps = [[], []]
    full_free = []
    full_blocked = []
    grow_selectors = []

    # Flag bits represent the future removal of the two switch/gate groups.
    for f in range(4):
        m = fixed_movement
        gr = fixed_ray_growth
        for t in range(2):
            if not (f & (1 << t)):
                m = add(m, groups[t])
                gr = add(gr, gates[t])
        fs = free_footprint(m, 2)
        fb = free_footprint(m, 4)
        free_maps[0].append(fs)
        free_maps[1].append(fb)

        ff = shift(shift(fb, dirs2), dirs2)
        full_free.append(ff)
        full_blocked.append(complement(ff))

        grow_ok = free_footprint(gr, 4)
        grow_selectors.append(first_one(shift(grow_ok, grow_offsets)))

    free_vectors = [cat(free_maps[s]) for s in range(2)]
    goal_sources = [
        scale(cap_space(sum_pixels(exit_pixels, footprint[w])), 100)
        for w in (2, 4)
    ]

    # First-hit directional rays stop at walls, gates, or another switch.
    ray_cases = [(0, 0), (0, 1), (1, 1), (2, 0)]
    ray_initial = {}
    ray_free = {}
    for f, t in ray_cases:
        ray_initial[(f, t)] = linear(
            keys[t], np.ones((4, 1), np.float32)
        )
        blockers = fixed_ray_growth
        for u in range(2):
            if not (f & (1 << u)):
                blockers = add(blockers, groups[u])
        ray_free[(f, t)] = relu(complement(blockers))

    ray_states = {}
    ray_state_specs = []
    for case in ray_cases:
        ident = fresh("ray_state")
        shapes[ident] = (4, 64, 64)
        ray_states[case] = ident
        ray_state_specs.append({"id": ident, "init": ray_initial[case]})

    ray_body = []
    destination[0] = ray_body
    ray_next = {}
    for case in ray_cases:
        z = shift(ray_states[case], dirs1)
        z = mul(z, ray_free[case])
        z = maximum(z, ray_initial[case])
        ray_next[ray_states[case]] = z
    destination[0] = graph

    ray_loop_id = fresh("rays")
    graph.append({
        "id": ray_loop_id,
        "op": "loop",
        "T": 64,
        "state": ray_state_specs,
        "body": ray_body,
        "next": ray_next,
    })
    ray_results = {}
    for case in ray_cases:
        ref = ray_loop_id + "." + ray_states[case]
        shapes[ref] = (4, 64, 64)
        ray_results[case] = ref

    ray_samples = {}
    ray_any = {}
    for s, width in enumerate((2, 4)):
        origins = [
            (-1, -1), (width - 2, -1),
            (-1, -1), (-1, width - 2),
        ]
        for case in ray_cases:
            sample = shift(ray_results[case], origins)
            ray_samples[(s,) + case] = sample
            ray_any[(s,) + case] = mul(pool_four(sample), can_fire)

    fire_masks = {}
    for s in range(2):
        for t in range(2):
            fire_masks[(s, t)] = cat([
                ray_any.get((s, f, t), zero) for f in range(4)
            ])

    # Recurrent value propagation over position, size and switch flags.
    def movement_values(v, size, flag):
        half = shift(v, dirs2)
        if size == 0:
            return half
        full = shift(half, dirs2)
        return add(
            mul(full, full_free[flag]),
            mul(half, full_blocked[flag])
        )

    def growth_value(vbig, flag, cost):
        z = shift(vbig, grow_offsets)
        z = mul(z, grow_selectors[flag])
        return linear(z, np.ones((1, 9), np.float32), -cost)

    vs_state = fresh("small_values")
    vb_state = fresh("big_values")
    shapes[vs_state] = (4, 64, 64)
    shapes[vb_state] = (4, 64, 64)

    value_body = []
    destination[0] = value_body
    states = [vs_state, vb_state]
    parts = [
        [sl(states[s], f, f + 1) for f in range(4)] for s in range(2)
    ]

    candidates = []
    for s in range(2):
        candidates.append(cat([
            pool_four(movement_values(parts[s][f], s, f))
            for f in range(4)
        ]))

    grow = cat([growth_value(parts[1][f], f, 4) for f in range(4)])
    candidates[0] = maximum(candidates[0], mul(grow, can_resize))
    candidates[1] = maximum(
        candidates[1], mul(scale(vs_state, 1, -4), can_resize)
    )

    for s in range(2):
        tele_value = gmax(mul(states[s], portals[s]))
        candidates[s] = maximum(
            candidates[s], scale(tele_value, 1, -3)
        )
        for t in range(2):
            w = np.zeros((4, 4), np.float32)
            for f in range(4):
                if not (f & (1 << t)):
                    w[f, f | (1 << t)] = 1
            fire = relu(linear(states[s], w, -3))
            fire = mul(fire, fire_masks[(s, t)])
            candidates[s] = maximum(candidates[s], fire)

    next_values = []
    for s in range(2):
        v = relu(scale(candidates[s], 1, -1))
        v = mul(v, free_vectors[s])
        next_values.append(maximum(v, goal_sources[s]))

    destination[0] = graph
    value_loop_id = fresh("values")
    value_loop = {
        "id": value_loop_id,
        "op": "loop",
        "T": 1,
        "state": [
            {"id": vs_state, "zeros": [4, 64, 64]},
            {"id": vb_state, "zeros": [4, 64, 64]},
        ],
        "body": value_body,
        "next": {
            vs_state: next_values[0],
            vb_state: next_values[1],
        },
    }
    graph.append(value_loop)
    values = [
        value_loop_id + "." + vs_state,
        value_loop_id + "." + vb_state,
    ]
    for ref in values:
        shapes[ref] = (4, 64, 64)

    vp = [
        [sl(values[s], f, f + 1) for f in range(4)] for s in range(2)
    ]

    def at_player(field, s):
        return gmax(mul(field, player_maps[s]))

    move_scores = []
    for s in range(2):
        qmove = relu(scale(movement_values(vp[s][0], s, 0), 1, -1))
        move_scores.append(at_player(qmove, s))
    move_scores = add(move_scores[0], move_scores[1])

    fire_scores = []
    hit_vectors = []
    for t in range(2):
        fs, hs = [], []
        for s in range(2):
            qf = relu(scale(vp[s][1 << t], 1, -4))
            qf = mul(qf, ray_any[(s, 0, t)])
            fs.append(at_player(qf, s))
            hs.append(at_player(ray_samples[(s, 0, t)], s))
        fire_scores.append(add(fs[0], fs[1]))
        hit_vectors.append(add(hs[0], hs[1]))

    resize_score = mul(add(
        at_player(relu(growth_value(vp[1][0], 0, 5)), 0),
        at_player(relu(scale(vp[0][0], 1, -5)), 1)
    ), can_resize)

    tele_scores = []
    for s, present in enumerate((is_small, is_big)):
        tv = gmax(mul(vp[s][0], portals[s]))
        tv = relu(scale(tv, 1, -4))
        tele_scores.append(mul(tv, present))
    teleport_score = add(tele_scores[0], tele_scores[1])

    scores = cat([
        move_scores, fire_scores[0], fire_scores[1],
        resize_score, teleport_score
    ], name="macro_scores")

    # Winner gates for integer-valued macro scores, with deterministic ties.
    ranked = linear(
        scores, np.eye(8, dtype=np.float32),
        np.arange(8, 0, -1, dtype=np.float32) * 0.02
    )
    comparisons = np.zeros((56, 8), np.float32)
    reduction = np.zeros((8, 56), np.float32)
    k = 0
    for i in range(8):
        for j in range(8):
            if i != j:
                comparisons[k, j] = 1
                comparisons[k, i] = -1
                reduction[i, k] = -1000
                k += 1
    losses = relu(linear(ranked, comparisons))
    winner = relu(linear(losses, reduction, 1))

    # Read the spell-entry cells at their canonical UI coordinates.
    centres = [
        (50 + 5 * y, 25 + 5 * x)
        for y in range(3) for x in range(3)
    ]
    w = np.zeros((18, 3), np.float32)
    b = np.zeros(18, np.float32)
    for i, (y, x) in enumerate(centres):
        w[i, 0] = 63
        b[i] = -y
        w[9 + i, 1] = 63
        b[9 + i] = -x
    errors = node("abs", [linear("coords", w, b)])
    w = np.zeros((9, 18), np.float32)
    for i in range(9):
        w[i, i] = -4
        w[i, 9 + i] = -4
    points = cap_space(relu(linear(errors, w, 2)))
    bits = gmax(mul(points, colours([6, 11, 14, 15])))
    board_count = total(bits)
    board_nonempty = cap_vector(board_count)

    # Memory stores intent, facing, and previous selected-cell count.
    old_intent = sl("mem", 0, 3)
    old_face = sl("mem", 3, 7)
    old_count = sl("mem", 7, 8)

    previous_direction = sl("aprev", 1, 5)
    moved_last = total(previous_direction)
    face = add(
        previous_direction,
        mul(old_face, complement(moved_last))
    )

    intent_w = np.zeros((3, 8), np.float32)
    intent_w[0, 4:6] = 1
    intent_w[1, 6] = 1
    intent_w[2, 7] = 1
    raw_intent = linear(winner, intent_w)

    # Casting clears a board with at least two already-entered cells.
    # Unlike the prior controller, commitment also covers the facing move.
    previous_ready = cap_vector(relu(scale(old_count, 1, -1)))
    completed = mul(
        mul(previous_ready, complement(board_nonempty)),
        sl("aprev", 6, 7)
    )
    latch = mul(total(old_intent), complement(completed))
    intent = add(
        mul(raw_intent, complement(latch)),
        mul(old_intent, latch)
    )

    aim = add(
        mul(hit_vectors[0], sl(winner, 4, 5)),
        mul(hit_vectors[1], sl(winner, 5, 6))
    )
    aim = first_one(aim)
    aim = add(mul(aim, complement(latch)), mul(face, latch))
    facing_correct = total(mul(face, aim))

    fire_intent = sl(intent, 0, 1)
    fire_now = mul(fire_intent, facing_correct)
    turning = mul(aim, mul(fire_intent, complement(facing_correct)))
    ordinary_move = mul(sl(winner, 0, 4), complement(latch))
    movement_output = add(ordinary_move, turning)
    click_output = add(total(sl(intent, 1, 3)), fire_now)

    outputs = cat([movement_output, click_output])
    out_w = np.zeros((8, 5), np.float32)
    for d in range(4):
        out_w[d + 1, d] = 10
    out_w[6, 4] = 10
    out_b = np.array([-20, 0, 0, 0, 0, -20, 0, -20], np.float32)
    linear(outputs, out_w, out_b, name="logits")

    # Correct extraneous cells before filling missing recipe cells.
    recipes = np.zeros((9, 3), np.float32)
    recipes[[1, 4, 7], 0] = 1
    recipes[[1, 3, 5, 7], 1] = 1
    recipes[[0, 1, 4], 2] = 1
    desired_bits = linear(intent, recipes)
    missing = relu(minus(desired_bits, bits))
    extra = relu(minus(bits, desired_bits))
    cell_scores = add(missing, scale(extra, 2))
    weighted_points = mul(points, cell_scores)
    linear(
        weighted_points, np.ones((1, 9), np.float32),
        name="click"
    )
    cat([intent, face, board_count], name="mem_next")

    # Static resource accounting; no game or observation evaluation.
    def macs(nodes):
        result = 0
        for nd in nodes:
            op = nd["op"]
            if op == "loop":
                result += nd["T"] * macs(nd["body"])
            elif op == "conv":
                result += int(np.prod(params[nd["w"]].shape)) * 4096
            elif op == "dense":
                result += int(np.prod(params[nd["w"]].shape))
            elif op in (
                "add", "sub", "mul", "max2", "min2", "vadd", "vmul"
            ):
                result += int(np.prod(shapes[nd["id"]]))
            elif op in (
                "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"
            ):
                result += int(np.prod(shapes[nd["in"][0]]))
        return result

    def evaluations(nodes):
        result = 0
        for nd in nodes:
            if nd["op"] == "loop":
                result += 1 + nd["T"] * evaluations(nd["body"])
            else:
                result += 1
        return result

    body_macs = macs(value_body)
    body_evals = evaluations(value_body)
    base_macs = macs(graph) - body_macs
    base_evals = evaluations(graph) - body_evals
    iterations = min(
        64,
        int((1_450_000_000 - base_macs) // body_macs),
        int((9800 - base_evals) // body_evals),
    )
    value_loop["T"] = max(1, iterations)

    return {
        "graph": graph,
        "memory_size": 8,
        "params": params,
    }
