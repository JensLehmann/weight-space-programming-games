import numpy as np


def build():
    G, P = [], {}
    ctx = [G]
    counter = [0]
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    N = 4

    def nd(op, xs=None, name=None, **kw):
        if name is None:
            counter[0] += 1
            name = "n%d" % counter[0]
        d = {"id": name, "op": op}
        if xs is not None:
            d["in"] = list(xs)
        d.update(kw)
        ctx[0].append(d)
        return name

    def par(a):
        k = "p%d" % len(P)
        P[k] = np.asarray(a, np.float32)
        return k

    def cv(x, w, b=None, name=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return nd("conv", [x], name, **kw)

    def de(x, w, b=None, name=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return nd("dense", [x], name, **kw)

    def mx(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        return cv(x, w[:, :, None, None], b, name)

    def aff(x, a=1, b=0, c=1, name=None):
        return mx(
            x,
            np.eye(c, dtype=np.float32) * a,
            np.full(c, b, np.float32),
            name,
        )

    def va(x, a=1, b=0, c=1, name=None):
        return de(
            x,
            np.eye(c, dtype=np.float32) * a,
            np.full(c, b, np.float32),
            name,
        )

    def const(v):
        v = np.asarray(v, np.float32).reshape(-1)
        return de("aprev", np.zeros((len(v), 8), np.float32), v)

    one = const([1])

    def add(a, b):
        return nd("add", [a, b])

    def sub(a, b):
        return nd("sub", [a, b])

    def mul(a, b):
        return nd("mul", [a, b])

    def vmax(a, b):
        return nd("max2", [a, b])

    def vmin(a, b):
        return nd("min2", [a, b])

    def r(x):
        return nd("relu", [x])

    def vr(x):
        return nd("vrelu", [x])

    def clip(x):
        return vmin(r(x), one)

    def vc(x, c=1):
        q = vr(x)
        return va(vr(va(q, -1, 1, c)), -1, 1, c)

    def inv(x):
        return sub(one, x)

    def vi(x, c=1):
        return va(x, -1, 1, c)

    def cat(*xs):
        return nd("concat", xs)

    def vcat(*xs):
        return nd("vconcat", xs)

    def sl(x, a, b):
        return nd("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return nd("vslice", [x], start=a, stop=b)

    def gm(x):
        return nd("gmax", [x])

    def mean(x):
        return nd("gmean", [x])

    def vp(x, k):
        w = np.zeros((4, 4), np.float32)
        for d in range(4):
            w[d, (d + k) % 4] = 1
        return de(x, w)

    def perm(x, k, c=1):
        w = np.zeros((4 * c, 4 * c), np.float32)
        for d in range(4):
            for j in range(c):
                w[d * c + j, ((d + k) % 4) * c + j] = 1
        return mx(x, w)

    def neighbors(x, c=1):
        w = np.zeros((4 * c, c, 3, 3), np.float32)
        for d, (dy, dx) in enumerate(directions):
            for j in range(c):
                w[d * c + j, j, 1 + dy, 1 + dx] = 1
        return cv(x, w)

    def sum_dirs(x, c=1):
        w = np.zeros((c, 4 * c), np.float32)
        for d in range(4):
            for j in range(c):
                w[j, d * c + j] = 1
        return mx(x, w)

    def pick(x, selector):
        return sum_dirs(mul(x, selector))

    def pixel_shift(x, selector):
        return pick(neighbors(x), selector)

    def safe_shift(x, selector):
        return add(pixel_shift(sub(x, one), selector), one)

    def tile(x, count=N):
        return mx(x, np.ones((count, 1), np.float32))

    def square(x, radius, c=1):
        k = 2 * radius + 1
        w = np.zeros((c, c, k, k), np.float32)
        for j in range(c):
            w[j, j] = 1
        return clip(cv(x, w))

    p1 = sl("x0", 1, 2)
    p4 = sl("x0", 4, 5)
    p6 = sl("x0", 6, 7)
    p8 = sl("x0", 8, 9)
    p9 = sl("x0", 9, 10)
    p11 = sl("x0", 11, 12)
    p12 = sl("x0", 12, 13)
    p15 = sl("x0", 15, 16)
    xy = sl("coords", 0, 2)

    # Infer gravity and the perpendicular direction from the source.
    gravity = vp(gm(mul(neighbors(p4), p6)), 2)
    tangent = vp(gravity, 1)
    anti_g = vp(gravity, 2)
    anti_t = vp(tangent, 2)
    vertical = de(gravity, [[1, 0, 1, 0]])
    horizontal = vi(vertical)

    # Recognize rendering scales 1, 2, 3, and 4.
    w = np.zeros((8, 1, 7, 7), np.float32)
    b = np.zeros(8, np.float32)
    for axis in range(2):
        for s in range(1, 5):
            o = axis * 4 + s - 1
            start = -((s - 1) // 2)
            for q in range(start, start + s):
                if axis == 0:
                    w[o, 0, 3, 3 + q] = 1
                else:
                    w[o, 0, 3 + q, 3] = 1
            for q in (start - 1, start + s):
                if axis == 0:
                    w[o, 0, 3, 3 + q] = -1
                else:
                    w[o, 0, 3 + q, 3] = -1
            b[o] = 1 - s

    runs = gm(r(cv(p6, w, b)))
    gates = de(
        vcat(vertical, horizontal),
        np.array([[1, 0]] * 4 + [[0, 1]] * 4, np.float32),
    )
    runs = nd("vmul", [runs, gates])
    w = np.zeros((4, 8), np.float32)
    for s in range(4):
        w[s, s] = w[s, s + 4] = 1
    scale = de(runs, w)
    size = de(scale, [[1, 2, 3, 4]])

    w = np.zeros((16, 4), np.float32)
    for s in range(4):
        w[4 * s:4 * s + 4, s] = 1
    stride_gate = de(scale, w)

    def cells(x, c=1):
        w = np.zeros((12 * c, c, 7, 7), np.float32)
        for s in range(1, 4):
            for d, (dy, dx) in enumerate(directions):
                for j in range(c):
                    w[(4 * (s - 1) + d) * c + j, j,
                      3 + s * dy, 3 + s * dx] = 1
        q = cv(x, w)
        q2 = sl(q, 4 * c, 8 * c)
        w4 = np.zeros((4 * c, 4 * c, 5, 5), np.float32)
        for d, (dy, dx) in enumerate(directions):
            for j in range(c):
                w4[d * c + j, d * c + j,
                   2 + 2 * dy, 2 + 2 * dx] = 1
        q4 = cv(q2, w4)

        if c == 1:
            gate = stride_gate
        else:
            wg = np.zeros((16 * c, 16), np.float32)
            for a in range(16):
                for j in range(c):
                    wg[a * c + j, a] = 1
            gate = de(stride_gate, wg)

        q = mul(cat(q, q4), gate)
        ws = np.zeros((4 * c, 16 * c), np.float32)
        for s in range(4):
            for d in range(4):
                for j in range(c):
                    ws[d * c + j, (4 * s + d) * c + j] = 1
        return mx(q, ws)

    def cell_shift(x, selector):
        return pick(cells(x), selector)

    # Determine selected-piece dimensions and whether it is an elbow.
    sx = gm(nd("rowsum", [p9]))
    sy = gm(nd("colsum", [p9]))
    width = nd("vadd", [
        nd("vmul", [sx, vertical]),
        nd("vmul", [sy, horizontal]),
    ])
    height = nd("vadd", [
        nd("vmul", [sy, vertical]),
        nd("vmul", [sx, horizontal]),
    ])
    angle = vc(de(vcat(height, size), [[2, -2]], [-1]))
    bar = vi(angle)

    free_removed = clip(add(add(p12, p6), p9))

    # Bucket mouth cells for each incoming direction.
    bucket_n = cells(p11)
    mouths = mul(
        r(aff(
            add(bucket_n,
                add(perm(bucket_n, -1), perm(bucket_n, 1))),
            1, -2, 4,
        )),
        free_removed,
    )
    mouth_union = clip(sum_dirs(mouths))

    # Oriented corner of the selected piece's bounding rectangle.
    box = mul(nd("rowmax", [p9]), nd("colmax", [p9]))
    w = np.zeros((4, 1, 3, 3), np.float32)
    for d in range(4):
        w[d, 0, 1, 1] = 1
        gy, gx = directions[d]
        ty, tx = directions[(d - 1) % 4]
        w[d, 0, 1 - gy, 1 - gx] = -1
        w[d, 0, 1 - ty, 1 - tx] = -1
    anchor = pick(r(cv(box, w)), gravity)
    current_xy = gm(mul(xy, anchor))

    # Expand a cell-aligned placement lattice from this corner.
    body = []
    old = ctx[0]
    ctx[0] = body
    lattice_next = vmax(
        "lattice_state", clip(sum_dirs(cells("lattice_state"))))
    ctx[0] = old
    nd(
        "loop", name="Lattice", T=64,
        state=[{"id": "lattice_state", "init": anchor}],
        body=body, next={"lattice_state": lattice_next},
    )
    lattice = "Lattice.lattice_state"

    # Exclusion masks: bucket margins and the top three game rows.
    bucket_box = clip(add(p11, mouth_union))
    w = np.zeros((3, 1, 7, 7), np.float32)
    for rad in range(1, 4):
        w[rad - 1, 0,
          3 - rad:4 + rad, 3 - rad:4 + rad] = 1
    margins = clip(cv(bucket_box, w))
    margin4 = square(sl(margins, 2, 3), 1)
    cup_margin = mx(
        mul(cat(margins, margin4), scale),
        np.ones((1, 4), np.float32),
    )
    source_line = clip(add(
        mul(nd("rowmax", [p4]), vertical),
        mul(nd("colmax", [p4]), horizontal),
    ))
    line1 = cell_shift(source_line, anti_g)
    line2 = cell_shift(line1, anti_g)
    forbidden = clip(add(add(source_line, line1), line2))

    allowed_final = r(sub(sub(free_removed, cup_margin), forbidden))
    allowed_move = r(sub(sub(
        clip(add(add(add(p12, p6), add(p8, p9)), p15)),
        cup_margin,
    ), forbidden))
    angle_allowed = vmin(
        allowed_final, cell_shift(allowed_final, gravity))
    placement_allowed = add(
        mul(allowed_final, bar), mul(angle_allowed, angle))

    projected_mouths = clip(add(
        mul(nd("colmax", [mouth_union]), vertical),
        mul(nd("rowmax", [mouth_union]), horizontal),
    ))
    projected_sources = clip(add(
        mul(nd("colmax", [p6]), vertical),
        mul(nd("rowmax", [p6]), horizontal),
    ))

    # Transverse distance-to-catchment field for proposal generation.
    landing_init = vmax(projected_mouths, p1)
    body = []
    old = ctx[0]
    ctx[0] = body
    lr = safe_shift("landing_state", tangent)
    ll = safe_shift("landing_state", anti_t)
    landing_next = vmax(
        "landing_state", r(aff(vmax(lr, ll), 1, -0.012)))
    ctx[0] = old
    nd(
        "loop", name="Landing", T=64,
        state=[{"id": "landing_state", "init": landing_init}],
        body=body, next={"landing_state": landing_next},
    )
    landing = "Landing.landing_state"

    # Directional approach value for side-facing bucket mouths.
    side_goal = pick(mouths, tangent)
    body = []
    old = ctx[0]
    ctx[0] = body
    side_next = vmax(
        "side_state",
        r(aff(pixel_shift("side_state", tangent), 1, -0.003)),
    )
    ctx[0] = old
    nd(
        "loop", name="SideValue", T=64,
        state=[{"id": "side_state", "init": side_goal}],
        body=body, next={"side_state": side_next},
    )
    side_value = "SideValue.side_state"

    left_value = safe_shift(landing, anti_t)
    angle_input = cell_shift(projected_sources, tangent)
    angle_out = cell_shift(cell_shift(side_value, tangent), tangent)
    angle_proposal = sub(
        aff(angle_out, 1.2, 0), aff(inv(angle_input), 3, 0))

    lengths = np.arange(1, 33, dtype=np.float32)
    length_abs = nd("vadd", [
        vr(de(width, np.ones((32, 1), np.float32), -lengths)),
        vr(de(width, -np.ones((32, 1), np.float32), lengths)),
    ])
    length_selector = vr(va(length_abs, -1, 1, 32))
    selector_shift = np.zeros((32, 32), np.float32)
    for j in range(31):
        selector_shift[j, j + 1] = 1

    # Scan possible piece widths to score every candidate anchor.
    body = []
    old = ctx[0]
    ctx[0] = body
    right_next = safe_shift("proposal_right", tangent)
    source_next = pixel_shift("proposal_source", tangent)
    source_max_next = vmax("proposal_source_max", source_next)
    allow_next = pixel_shift("proposal_allow", tangent)
    allow_min_next = vmin("proposal_allow_min", allow_next)

    quality = add(
        vmin(left_value, right_next),
        aff(vmax(left_value, right_next), 0.1, 0),
    )
    bar_proposal = sub(
        quality, aff(inv("proposal_source_max"), 0.12, 0))
    score_here = add(
        mul(bar_proposal, bar), mul(angle_proposal, angle))
    score_here = sub(
        score_here, aff(inv("proposal_allow_min"), 4, 0))
    accumulated = add(
        "proposal_acc",
        mul(score_here, vs("proposal_selector", 0, 1)),
    )
    selector_next = de("proposal_selector", selector_shift)
    ctx[0] = old
    nd(
        "loop", name="Proposals", T=32,
        state=[
            {"id": "proposal_right", "init": landing},
            {"id": "proposal_source", "init": projected_sources},
            {"id": "proposal_source_max", "init": projected_sources},
            {"id": "proposal_allow", "init": placement_allowed},
            {"id": "proposal_allow_min", "init": placement_allowed},
            {"id": "proposal_selector", "init": length_selector},
            {"id": "proposal_acc", "zeros": [1, 64, 64]},
        ],
        body=body,
        next={
            "proposal_right": right_next,
            "proposal_source": source_next,
            "proposal_source_max": source_max_next,
            "proposal_allow": allow_next,
            "proposal_allow_min": allow_min_next,
            "proposal_selector": selector_next,
            "proposal_acc": accumulated,
        },
    )

    raw_proposal = "Proposals.proposal_acc"
    delta_xy = nd("abs", [sub(xy, current_xy)])
    distance = mx(delta_xy, [[1, 1]])
    rank = sub(raw_proposal, aff(distance, 0.01, 0))
    rank = sub(rank, aff(inv(lattice), 10, 0))
    key_tie = mx(xy, [[1, 1 / 64]])
    rank = sub(rank, aff(key_tie, 0.002, 0))

    projection_w = np.array(
        [[-1, 0, 1, 0], [0, 1, 0, -1]], np.float32)
    gvec = de(gravity, projection_w)
    tvec = de(tangent, projection_w)
    abs_tvec = de(tangent, np.abs(projection_w))

    targets = [current_xy]
    qualities = [gm(mul(raw_proposal, anchor))]
    work_rank = rank
    quality_offset = const([10])

    # Three distinct proposals, plus the unchanged layout.
    for _ in range(N - 1):
        best = gm(work_rank)
        chosen = clip(aff(sub(work_rank, best), 1e8, 1))
        target = gm(mul(xy, chosen))
        targets.append(target)
        qualities.append(
            va(gm(mul(add(raw_proposal, quality_offset), chosen)),
               1, -10)
        )
        transverse_distance = mx(
            mul(nd("abs", [sub(xy, target)]), abs_tvec),
            [[63, 63]],
        )
        exclusion_edge = sub(
            transverse_distance, va(size, 1.5, 0))
        outside = clip(aff(exclusion_edge, 100, 0))
        work_rank = sub(work_rank, aff(inv(outside), 20, 0))

    gfield = mx(mul(xy, gvec), [[63, 63]])
    tfield = mx(mul(xy, tvec), [[63, 63]])

    def scalar_projection(v, axis):
        return de(nd("vmul", [v, axis]), [[63, 63]])

    def shape_at(target):
        dg = sub(gfield, scalar_projection(target, gvec))
        dt = sub(tfield, scalar_projection(target, tvec))
        lower_g = clip(aff(dg, 100, 10))
        lower_t = clip(aff(dt, 100, 10))
        upper_g = clip(
            aff(sub(va(height, 1, -0.9), dg), 100, 0))
        upper_t = clip(
            aff(sub(va(width, 1, -0.9), dt), 100, 0))
        rectangle = mul(
            mul(lower_g, lower_t), mul(upper_g, upper_t))
        top_band = clip(
            aff(sub(va(size, 1, -0.9), dg), 100, 0))
        far_band = clip(
            aff(sub(dt, va(size, 1, -0.1)), 100, 0))
        missing_corner = mul(mul(top_band, far_band), angle)
        return mul(rectangle, inv(missing_corner))

    shapes = [p9] + [shape_at(t) for t in targets[1:]]
    candidate_shapes = cat(*shapes)
    bars = clip(add(tile(p8), mul(candidate_shapes, bar)))
    elbows = clip(add(tile(p15), mul(candidate_shapes, angle)))
    empty = r(sub(tile(free_removed), candidate_shapes))

    # Four scenes, each carrying water in four directions.
    bn = neighbors(bars, N)
    cn = neighbors(tile(p11), N)
    an = neighbors(elbows, N)
    ac = cells(elbows, N)
    ec = cells(empty, N)
    mouth_bank = cat(
        *[tile(sl(mouths, d, d + 1)) for d in range(4)]
    )
    turn_r = clip(mul(
        mul(an, perm(ac, 1, N)), perm(ec, -1, N)))
    turn_l = clip(mul(
        mul(an, perm(ac, -1, N)), perm(ec, 1, N)))
    turn_any = clip(add(turn_r, turn_l))
    split = clip(add(
        add(bn, mul(cn, inv(mouth_bank))),
        mul(an, inv(turn_any)),
    ))
    straight_keep = r(sub(inv(mouth_bank), turn_any))

    sr = [sl(turn_r, d * N, (d + 1) * N) for d in range(4)]
    st = [sl(turn_l, d * N, (d + 1) * N) for d in range(4)]
    sp = [sl(split, d * N, (d + 1) * N) for d in range(4)]
    sk = [
        sl(straight_keep, d * N, (d + 1) * N)
        for d in range(4)
    ]
    init = [
        tile(mul(p6, vs(gravity, d, d + 1)))
        for d in range(4)
    ]

    def fixed_shift(x, direction):
        dy, dx = directions[direction]
        w = np.zeros((N, N, 3, 3), np.float32)
        for j in range(N):
            w[j, j, 1 - dy, 1 - dx] = 1
        return cv(x, w)

    body = []
    old = ctx[0]
    ctx[0] = body
    states = ["water_u", "water_r", "water_d", "water_l"]
    next_states = {}
    for d in range(4):
        outgoing = add(
            add(
                mul(states[d], sk[d]),
                mul(states[(d + 1) % 4], sr[(d + 1) % 4]),
            ),
            mul(states[(d - 1) % 4], st[(d - 1) % 4]),
        )
        straight_arrival = fixed_shift(outgoing, d)
        splitting = mul(states[d], sp[d])
        right_arrival = fixed_shift(splitting, (d - 1) % 4)
        left_arrival = fixed_shift(splitting, (d + 1) % 4)
        arrival = clip(mul(
            add(add(straight_arrival, right_arrival), left_arrival),
            empty,
        ))
        next_states[states[d]] = vmax(states[d], arrival)
    ctx[0] = old
    nd(
        "loop", name="Counterfactual", T=96,
        state=[
            {"id": states[d], "init": init[d]}
            for d in range(4)
        ],
        body=body, next=next_states,
    )
    flows = ["Counterfactual." + s for s in states]

    caught_parts = [
        mul(flows[d], tile(sl(mouths, d, d + 1)))
        for d in range(4)
    ]
    caught = add(
        add(caught_parts[0], caught_parts[1]),
        add(caught_parts[2], caught_parts[3]),
    )
    reached = square(caught, 3, N)
    captured = mul(reached, tile(mouth_union))
    missing = gm(mul(tile(mouth_union), inv(reached)))

    wall_neighbors = neighbors(p1)
    leaks = [
        mul(flows[d], tile(sl(wall_neighbors, d, d + 1)))
        for d in range(4)
    ]
    leak_map = clip(add(
        add(leaks[0], leaks[1]), add(leaks[2], leaks[3])))
    leaking = gm(leak_map)
    leak_badness = mul(leak_map, tile(inv(landing)))

    capture_score = va(mean(captured), 8192, 0, N)
    leak_score = va(mean(leak_map), -61440, 0, N)
    distance_score = va(mean(leak_badness), -20480, 0, N)
    quality_vector = vcat(*qualities)
    quality_score = va(quality_vector, 5, 0, N)
    scores = nd("vadd", [
        nd("vadd", [capture_score, leak_score]),
        nd("vadd", [distance_score, quality_score]),
    ])

    travel_costs = []
    for target in targets:
        q = de(
            vcat(target, current_xy),
            [[1, 0, -1, 0], [0, 1, 0, -1]],
        )
        aq = nd("vadd", [vr(q), vr(va(q, -1, 0, 2))])
        travel_costs.append(de(aq, [[-0.063, -0.063]]))
    scores = nd("vadd", [scores, vcat(*travel_costs)])
    scores = nd("vadd", [
        scores,
        const([0.001 * (N - 1 - j) for j in range(N)]),
    ])

    # ReLU implementation of a deterministic vector maximum.
    best_score = vs(scores, 0, 1)
    for j in range(1, N):
        q = vs(scores, j, j + 1)
        difference = de(vcat(q, best_score), [[1, -1]])
        best_score = nd("vadd", [best_score, vr(difference)])

    winners = []
    for j in range(N):
        difference = de(
            vcat(vs(scores, j, j + 1), best_score), [[1, -1]])
        winners.append(vc(va(difference, 1e5, 1)))
    weighted_targets = [
        nd("vmul", [targets[j], winners[j]])
        for j in range(N)
    ]
    proposed_target = weighted_targets[0]
    for q in weighted_targets[1:]:
        proposed_target = nd("vadd", [proposed_target, q])

    # Destination memory avoids replanning during a successful move.
    old_xy = vs("mem", 0, 2)
    displacement = de(
        vcat(current_xy, old_xy),
        [[1, 0, -1, 0], [0, 1, 0, -1]],
    )
    absolute_displacement = nd("vadd", [
        vr(displacement), vr(va(displacement, -1, 0, 2)),
    ])
    unchanged = vc(de(
        absolute_displacement, [[-10000, -10000]], [1]))
    previous_moves = vs("aprev", 1, 5)
    was_moving = de(previous_moves, [[1, 1, 1, 1]])
    keep_target = nd("vmul", [was_moving, vi(unchanged)])
    old_target = vs("mem", 2, 4)
    target_xy = nd("vadd", [
        nd("vmul", [old_target, keep_target]),
        nd("vmul", [proposed_target, vi(keep_target)]),
    ])

    movement = de(
        vcat(current_xy, target_xy),
        np.array([
            [63, 0, -63, 0],
            [-63, 0, 63, 0],
            [0, 63, 0, -63],
            [0, -63, 0, 63],
        ], np.float32),
    )
    moved_piece = perm(cells(p9), 2)
    illegal = gm(mul(moved_piece, inv(allowed_move)))
    w = np.zeros((4, 4), np.float32)
    for j, d in enumerate([0, 2, 3, 1]):
        w[j, d] = -100
    penalties = de(illegal, w)
    failed_move = nd("vmul", [previous_moves, unchanged])
    movement = nd("vadd", [movement, penalties])
    movement = nd("vadd", [
        movement, va(failed_move, -100, 0, 4),
    ])

    # Select the next non-selected piece in display order.
    other = clip(add(p8, p15))
    other_exists = gm(other)
    key = mx(xy, [[64, 1]])
    current_key = de(current_xy, [[64, 1]])
    after = clip(aff(sub(key, current_key), 100000, -0.1))
    click_score = sub(
        add(aff(other, 1000, 0), aff(after, 100, 0)), key)
    aff(click_score, 1, 0, name="click")
    click_logit = de(other_exists, [[100.25]], [-100])

    current_missing = vs(missing, 0, 1)
    current_leaking = vs(leaking, 0, 1)
    success = nd("vmul", [
        vi(current_missing), vi(current_leaking),
    ])
    success_logit = de(success, [[105]], [-5])

    # After a complete settled-piece cycle, permit a real spill test.
    count = vs("mem", 4, 5)
    next_count = nd("vmul", [
        va(count, 1, 1), vs("aprev", 6, 7),
    ])
    cycle_done = vc(va(next_count, 1, -5))
    fallback = vc(nd("vadd", [cycle_done, vi(other_exists)]))
    fallback_logit = de(fallback, [[5.5]], [-5])
    difference = de(
        vcat(success_logit, fallback_logit), [[1, -1]])
    spill_logit = nd("vadd", [
        fallback_logit, vr(difference),
    ])

    features = vcat(movement, spill_logit, click_logit)
    w = np.zeros((8, 6), np.float32)
    w[1:5, :4] = np.eye(4, dtype=np.float32)
    w[5, 4] = 1
    w[6, 5] = 1
    b = np.zeros(8, np.float32)
    b[0] = b[7] = -1000
    de(features, w, b, name="logits")
    nd(
        "vconcat", [current_xy, target_xy, next_count],
        name="mem_next",
    )

    return {"graph": G, "memory_size": 5, "params": P}
