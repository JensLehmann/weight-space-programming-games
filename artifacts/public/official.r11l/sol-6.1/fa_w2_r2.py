import numpy as np


def build():
    P, G, W = {}, [], {}
    serial = [0]
    pcache, ncaches = {}, {}

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def param(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in pcache:
            name = "p" + str(len(P))
            P[name] = a
            pcache[key] = name
        return pcache[key]

    def node(op, xs, width=None, name=None, **fields):
        xs = list(xs)
        if width is None:
            if op in ("concat", "vconcat"):
                width = sum(W[x] for x in xs)
            elif op in ("slice", "vslice"):
                width = fields["stop"] - fields["start"]
            else:
                width = max(W[x] for x in xs)
        cache = ncaches.setdefault(id(G), {})
        key = (op, tuple(xs), tuple(sorted(fields.items())))
        if name is None and key in cache:
            return cache[key]
        name = name or fresh()
        nd = {"id": name, "op": op, "in": xs}
        nd.update(fields)
        G.append(nd)
        W[name] = width
        cache[key] = name
        return name

    W.update(x0=16, coords=3, aprev=8, avail=8, mem=237)

    def sl(x, a, b):
        if a == 0 and b == W[x]:
            return x
        return node("slice", [x], start=a, stop=b)

    def vl(x, a, b):
        if a == 0 and b == W[x]:
            return x
        return node("vslice", [x], start=a, stop=b)

    def cat(*xs):
        return xs[0] if len(xs) == 1 else node("concat", xs)

    def vc(*xs):
        return xs[0] if len(xs) == 1 else node("vconcat", xs)

    def conv(x, w, b=None):
        w = np.asarray(w, np.float32)
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], width=w.shape[0], **fields)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], width=w.shape[0], name=name, **fields)

    def smat(x, w, b=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b)

    def sa(x, scale=1.0, bias=0.0):
        if scale == 1.0 and bias == 0.0:
            return x
        return smat(
            x, np.eye(W[x], dtype=np.float32) * scale,
            np.full(W[x], bias, np.float32)
        )

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def ab(a):
        return node("abs", [a])

    def vm(a, b):
        return node("vmul", [a, b])

    def vv(a, b):
        return node("vadd", [a, b])

    ONE = sl("coords", 2, 3)
    ZERO = sa(ONE, 0.0)
    VONE = node("gmean", [ONE])
    VZERO = dense(VONE, [[0.0]])
    VNEG = dense(VONE, [[-1.0]])

    def va(x, scale=1.0, bias=0.0):
        if scale == 1.0 and bias == 0.0:
            return x
        if W[x] <= 16:
            return dense(
                x, np.eye(W[x], dtype=np.float32) * scale,
                np.full(W[x], bias, np.float32)
            )
        out = x
        if scale != 1.0:
            out = vm(out, dense(VONE, [[0.0]], [scale]))
        if bias != 0.0:
            out = vv(out, dense(VONE, [[0.0]], [bias]))
        return out

    def vd(a, b):
        return vv(a, vm(b, VNEG))

    def vab(a):
        return vv(
            node("vrelu", [a]),
            node("vrelu", [vm(a, VNEG)])
        )

    def vmax(a, b):
        return vv(node("vrelu", [vd(a, b)]), b)

    def clip(a):
        p = node("vrelu", [a])
        return vd(p, node("vrelu", [vv(p, VNEG)]))

    def hard(a, threshold=0.0, gain=32.0):
        return node("min2", [
            node("relu", [sa(a, gain, -threshold * gain)]), ONE
        ])

    def vh(a, threshold=0.0, gain=32.0):
        return clip(va(a, gain, -threshold * gain))

    def ss(a):
        return smat(a, np.ones((1, W[a]), np.float32))

    def vs(a):
        return dense(a, np.ones((1, W[a]), np.float32))

    def winner(scores):
        n = W[scores]
        scores = dense(
            scores, np.eye(n, dtype=np.float32),
            -0.002 * np.arange(n, dtype=np.float32)
        )
        w = np.zeros((n * n, n), np.float32)
        for i in range(n):
            for j in range(n):
                w[i * n + j, j] += 1.0
                w[i * n + j, i] -= 1.0
        positive = node("vrelu", [dense(scores, w)])
        reduction = np.zeros((n, n * n), np.float32)
        for i in range(n):
            reduction[i, i * n:(i + 1) * n] = -8192.0
        return node("vrelu", [
            dense(positive, reduction, np.ones(n, np.float32))
        ])

    Y = sa(sl("coords", 0, 1), 63.0)
    X = sa(sl("coords", 1, 2), 63.0)
    INDEX = add(sa(Y, 64.0), X)

    def point_winner(score, mask):
        score = add(score, sa(mask, 10000.0, -10000.0))
        best = node("gmax", [score])
        # Reject every strictly lower float32 score.
        maximal = mul(hard(sub(score, best), -1e-7, 2e7), mask)
        rank = add(sa(INDEX, -1.0), sa(maximal, 10000.0, -10000.0))
        rb = node("gmax", [rank])
        return mul(hard(sub(rank, rb), -0.25, 8.0), maximal)

    def extract(point, features):
        return va(node("gmean", [mul(point, features)]), 4096.0)

    def dist(x, y):
        return add(ab(sub(X, x)), ab(sub(Y, y)))

    def linf(x, y):
        return node("max2", [ab(sub(X, x)), ab(sub(Y, y))])

    def point_at(x, y):
        return hard(sa(linf(x, y), -1.0, 0.25), 0.0, 8.0)

    def floor_v(values):
        c = W[values]
        w = np.zeros((63 * c, c), np.float32)
        b = np.zeros(63 * c, np.float32)
        gain = 131072.0
        for i in range(c):
            for k in range(1, 64):
                j = i * 63 + k - 1
                w[j, i] = gain
                b[j] = -(k - 0.00001) * gain
        steps = clip(dense(values, w, b))
        reduction = np.zeros((c, 63 * c), np.float32)
        for i in range(c):
            reduction[i, i * 63:(i + 1) * 63] = 1.0
        return dense(steps, reduction)

    circle = [
        (dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)
        if not (abs(dy) == 2 and abs(dx) == 2)
    ]
    diamond = [
        (dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)
        if abs(dy) + abs(dx) <= 2
    ]
    outer = [p for p in diamond if p != (0, 0)]
    extra = [p for p in circle if p not in diamond]
    ring = [
        (-3, -1), (-3, 1), (-2, -2), (-2, 2),
        (-1, -3), (-1, 3), (1, -3), (1, 3),
        (2, -2), (2, 2), (3, -1), (3, 1)
    ]
    shell = [
        (dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)
        if max(abs(dy), abs(dx)) == 3
    ]
    hues = [7, 8, 9, 11, 12, 13, 14, 15]
    occupied = [c for c in range(16) if c not in (1, 2, 3, 5, 10)]

    def template(offsets, colors, k):
        w = np.zeros((1, 16, k, k), np.float32)
        r = k // 2
        for dy, dx in offsets:
            for c in colors:
                w[0, c, r + dy, r + dx] = 1.0
        return w

    circles = hard(
        conv("x0", template(circle, occupied, 5)), 20.5, 4.0
    )
    regular = mul(circles, sl("x0", 6, 7))
    black_circle = hard(
        conv("x0", template(circle, [0], 5)), 20.5, 4.0
    )
    nearby_lines = hard(
        conv("x0", template(shell, [1], 7)), 1.5, 4.0
    )
    wc = np.zeros((8, 16, 5, 5), np.float32)
    wr = np.zeros((8, 16, 7, 7), np.float32)
    for i, color in enumerate(hues):
        for dy, dx in circle:
            wc[i, color, dy + 2, dx + 2] = 1.0
        for dy, dx in ring:
            wr[i, color, dy + 3, dx + 3] = 1.0
    color_counts = conv("x0", wc)
    circle_sig = hard(color_counts, 7.5, 4.0)
    raw_ring_sig = hard(conv("x0", wr), 0.5, 4.0)
    raw_rings = hard(
        conv("x0", template(ring, occupied, 7)), 11.5, 4.0
    )
    black_rings = mul(
        raw_rings, sub(ONE, hard(ss(raw_ring_sig), 0.5, 4.0))
    )

    wm = template(outer, [0, 3], 5)
    for dy, dx in extra:
        wm[0, 0, dy + 2, dx + 2] = -2.0
        wm[0, 3, dy + 2, dx + 2] = -2.0
    markers = hard(conv("x0", wm), 11.5, 4.0)
    selected = mul(
        markers,
        hard(conv("x0", template(outer, [0], 5)), 11.5, 4.0)
    )
    marker_hues = cat(*(sl("x0", c, c + 1) for c in hues))
    marker_features = cat(X, Y, sl("x0", 4, 5), marker_hues, selected)

    candidates = mul(
        circles,
        node("max2", [
            regular, node("max2", [black_circle, nearby_lines])
        ])
    )
    priority = add(
        add(sa(regular, 20000.0), sa(black_circle, 10000.0)),
        sa(INDEX, -1.0)
    )
    ball_features = cat(X, Y, regular, ONE, circle_sig)

    def stage(index, n):
        diff = dense(
            index, np.ones((n, 1), np.float32),
            -np.arange(n, dtype=np.float32)
        )
        return vh(vm(vab(diff), VNEG), -0.25, 8.0)

    def stage_record(record, gates, n, size):
        copy_w = np.tile(np.eye(size, dtype=np.float32), (n, 1))
        gate_w = np.zeros((n * size, n), np.float32)
        for i in range(n):
            gate_w[i * size:(i + 1) * size, i] = 1.0
        return vm(dense(record, copy_w), dense(gates, gate_w))

    enclosing = G
    body = []
    G = body
    br, bd, bi = [fresh("s") for _ in range(3)]
    W.update({br: 1, bd: 48, bi: 1})
    bp = point_winner(priority, br)
    record = extract(bp, ball_features)
    bd_next = vv(bd, stage_record(record, stage(bi, 4), 4, 12))
    br_next = sub(br, bp)
    bi_next = va(bi, 1.0, 1.0)
    G = enclosing
    bid = fresh("L")
    G.append({
        "id": bid, "op": "loop", "T": 4,
        "state": [
            {"id": br, "init": candidates},
            {"id": bd, "zeros": [48]},
            {"id": bi, "init": VZERO}
        ],
        "body": body,
        "next": {br: br_next, bd: bd_next, bi: bi_next}
    })
    initial_records = bid + "." + bd
    W[initial_records] = 48
    balls = [vl(initial_records, i * 12, (i + 1) * 12) for i in range(4)]
    ibx = vc(*(vl(v, 0, 1) for v in balls))
    iby = vc(*(vl(v, 1, 2) for v in balls))
    ireg = vc(*(vl(v, 2, 3) for v in balls))
    ipresent = vc(*(vl(v, 3, 4) for v in balls))
    isig = [vl(v, 4, 12) for v in balls]
    ibdata = vc(*(vc(vl(v, 2, 3), vl(v, 4, 12)) for v in balls))

    enclosing = G
    body = []
    G = body
    sr, sh, se, si = [fresh("s") for _ in range(4)]
    W.update({sr: 1, sh: 84, se: 12, si: 1})
    mp = point_winner(sa(INDEX, -1.0), sr)
    mv = extract(mp, marker_features)
    mx, my = vl(mv, 0, 1), vl(mv, 1, 2)
    neutral = vl(mv, 2, 3)
    mh = vl(mv, 3, 11)
    mselected = vl(mv, 11, 12)
    mpresent = node("gmax", [mp])

    dx, dy = vd(ibx, mx), vd(iby, my)
    adx, ady = vab(dx), vab(dy)
    length = vmax(adx, ady)
    rx, ry = sub(X, mx), sub(Y, my)
    cross = sub(mul(rx, dy), mul(ry, dx))
    band = hard(sub(va(length, 0.51, 0.1), ab(cross)))
    xerr = sub(add(ab(rx), ab(sub(rx, dx))), adx)
    yerr = sub(add(ab(ry), ab(sub(ry, dy))), ady)
    ray = mul(
        band,
        mul(hard(sa(xerr, -1.0, 1.1)),
            hard(sa(yerr, -1.0, 1.1)))
    )
    line_count = va(
        node("gmean", [mul(ray, sl("x0", 1, 2))]), 4096.0
    )
    hue_match = vc(*(vs(vm(mh, s)) for s in isig))
    wanted_regular = va(neutral, -1.0, 1.0)
    score = vv(
        vd(line_count, length),
        vv(
            va(vab(vd(ireg, wanted_regular)), -1000.0),
            vv(
                va(vd(wanted_regular, hue_match), -1000.0),
                va(ipresent, 10000.0, -10000.0)
            )
        )
    )
    group = vm(winner(score), mpresent)
    record = vc(mx, my, group, mpresent)
    gates = stage(si, 12)
    sh_next = vv(sh, stage_record(record, gates, 12, 7))
    se_next = vv(se, vm(gates, mselected))
    sr_next = sub(sr, mp)
    si_next = va(si, 1.0, 1.0)
    G = enclosing
    aid = fresh("L")
    G.append({
        "id": aid, "op": "loop", "T": 12,
        "state": [
            {"id": sr, "init": markers},
            {"id": sh, "zeros": [84]},
            {"id": se, "zeros": [12]},
            {"id": si, "init": VZERO}
        ],
        "body": body,
        "next": {sr: sr_next, sh: sh_next, se: se_next, si: si_next}
    })
    ihdata, iselected = aid + "." + sh, aid + "." + se
    W[ihdata], W[iselected] = 84, 12

    near_initial = hard(sa(linf(ibx, iby), -1.0, 5.25), 0.0, 8.0)
    expand = np.zeros((32, 4), np.float32)
    reduce = np.zeros((8, 32), np.float32)
    for i in range(4):
        for c in range(8):
            expand[i * 8 + c, i] = 1.0
            reduce[c, i * 8 + c] = 1.0
    inferred_sig = smat(mul(smat(near_initial, expand), vc(*isig)), reduce)
    initial_goal_sig = hard(
        add(raw_ring_sig, mul(black_rings, inferred_sig)), 0.5, 4.0
    )
    goal_features = cat(X, Y, ONE, initial_goal_sig)

    enclosing = G
    body = []
    G = body
    gr, gg, gi = [fresh("s") for _ in range(3)]
    W.update({gr: 1, gg: 88, gi: 1})
    gp = point_winner(sa(INDEX, -1.0), gr)
    gv = extract(gp, goal_features)
    gg_next = vv(gg, stage_record(gv, stage(gi, 8), 8, 11))
    gr_next = sub(gr, gp)
    gi_next = va(gi, 1.0, 1.0)
    G = enclosing
    gid = fresh("L")
    G.append({
        "id": gid, "op": "loop", "T": 8,
        "state": [
            {"id": gr, "init": raw_rings},
            {"id": gg, "zeros": [88]},
            {"id": gi, "init": VZERO}
        ],
        "body": body,
        "next": {gr: gr_next, gg: gg_next, gi: gi_next}
    })
    igdata = gid + "." + gg
    W[igdata] = 88

    have_memory = vl("mem", 220, 221)
    no_memory = va(have_memory, -1.0, 1.0)

    def initialize(initial, start, stop):
        return vv(
            vm(vl("mem", start, stop), have_memory),
            vm(initial, no_memory)
        )

    hdata = initialize(ihdata, 0, 84)
    bdata = initialize(ibdata, 84, 120)
    gdata = initialize(igdata, 120, 208)
    selected_id = initialize(iselected, 208, 220)
    original = initialize(vc(ibx, iby), 225, 233)
    moved = vm(vl("mem", 233, 237), have_memory)

    def hcolumn(column):
        w = np.zeros((12, 84), np.float32)
        for i in range(12):
            w[i, i * 7 + column] = 1.0
        return dense(hdata, w)

    hx, hy = hcolumn(0), hcolumn(1)
    hg = [hcolumn(i + 2) for i in range(4)]
    hp = hcolumn(6)

    observed = extract(selected, cat(X, Y))
    visible = node("gmax", [selected])
    replace = vm(selected_id, visible)
    hx = vv(
        vm(hx, va(replace, -1.0, 1.0)),
        vm(vl(observed, 0, 1), replace)
    )
    hy = vv(
        vm(hy, va(replace, -1.0, 1.0)),
        vm(vl(observed, 1, 2), replace)
    )

    group_rows = vc(*hg)
    replicate = np.tile(np.eye(12, dtype=np.float32), (4, 1))
    reduction = np.zeros((4, 48), np.float32)
    for i in range(4):
        reduction[i, i * 12:(i + 1) * 12] = 1.0
    groups = vm(group_rows, dense(hp, replicate))
    counts = dense(groups, reduction)
    sumx = dense(vm(groups, dense(hx, replicate)), reduction)
    sumy = dense(vm(groups, dense(hy, replicate)), reduction)
    group_selected = dense(vm(groups, dense(selected_id, replicate)), reduction)
    present = vh(counts, 0.5, 4.0)

    diff_w = np.zeros((96, 4), np.float32)
    diff_b = np.zeros(96, np.float32)
    for i in range(4):
        for n in range(1, 13):
            j = i * 24 + 2 * (n - 1)
            diff_w[j, i], diff_w[j + 1, i] = 1.0, -1.0
            diff_b[j], diff_b[j + 1] = -n, n
    positive = node("vrelu", [dense(counts, diff_w, diff_b)])
    hats_w = np.zeros((48, 96), np.float32)
    for i in range(48):
        hats_w[i, 2 * i:2 * i + 2] = -1.0
    hats = node("vrelu", [
        dense(positive, hats_w, np.ones(48, np.float32))
    ])
    reciprocal_w = np.zeros((4, 48), np.float32)
    for i in range(4):
        for n in range(1, 13):
            reciprocal_w[i, i * 12 + n - 1] = 1.0 / n - 1.0
    reciprocal = dense(hats, reciprocal_w, np.ones(4, np.float32))
    bx, by = floor_v(vm(sumx, reciprocal)), floor_v(vm(sumy, reciprocal))
    single = vh(vm(vab(va(counts, 1.0, -1.0)), VNEG), -0.25, 8.0)
    keep_original = vm(single, va(moved, -1.0, 1.0))
    bx = vv(
        vm(bx, va(keep_original, -1.0, 1.0)),
        vm(vl(original, 0, 4), keep_original)
    )
    by = vv(
        vm(by, va(keep_original, -1.0, 1.0)),
        vm(vl(original, 4, 8), keep_original)
    )

    regs = vc(*(vl(bdata, i * 9, i * 9 + 1) for i in range(4)))
    old_sigs = [vl(bdata, i * 9 + 1, i * 9 + 9) for i in range(4)]

    # Observe current ball colours at the remembered centroid.
    ball_points = mul(point_at(bx, by), present)
    expand = np.zeros((32, 4), np.float32)
    color_expand = np.tile(np.eye(8, dtype=np.float32), (4, 1))
    for i in range(4):
        expand[i * 8:(i + 1) * 8, i] = 1.0
    observed_colors = va(node("gmean", [
        mul(smat(ball_points, expand), smat(circle_sig, color_expand))
    ]), 4096.0)
    sigs = [
        vmax(old_sigs[i], vl(observed_colors, i * 8, (i + 1) * 8))
        for i in range(4)
    ]

    def gcolumn(column):
        w = np.zeros((8, 88), np.float32)
        for i in range(8):
            w[i, i * 11 + column] = 1.0
        return dense(gdata, w)

    goal_points = mul(
        point_at(gcolumn(0), gcolumn(1)), gcolumn(2)
    )
    rings = ss(goal_points)
    sig_w = np.zeros((64, 88), np.float32)
    expand = np.zeros((64, 8), np.float32)
    reduce = np.zeros((8, 64), np.float32)
    for i in range(8):
        for c in range(8):
            sig_w[i * 8 + c, i * 11 + 3 + c] = 1.0
            expand[i * 8 + c, i] = 1.0
            reduce[c, i * 8 + c] = 1.0
    ring_sig = smat(
        mul(smat(goal_points, expand), dense(gdata, sig_w)), reduce
    )

    ball_distance = linf(bx, by)
    ball_near = mul(
        hard(sa(ball_distance, -1.0, 3.25), 0.0, 8.0), present
    )
    exclusion = hard(ss(ball_near), 0.5, 4.0)
    black_near = mul(
        black_rings,
        hard(sa(ball_distance, -1.0, 5.25), 0.0, 8.0)
    )
    done_now = node("gmax", [black_near])
    done = vm(
        clip(vv(done_now, vm(vl("mem", 221, 225), have_memory))),
        present
    )
    unsolved = vm(present, va(done, -1.0, 1.0))
    selected_unsolved = vm(group_selected, unsolved)
    use_selected = vh(vs(selected_unsolved), 0.5, 4.0)
    first = winner(dense(
        unsolved, 100.0 * np.eye(4, dtype=np.float32),
        -np.arange(4, dtype=np.float32)
    ))
    active = vv(
        vm(selected_unsolved, use_selected),
        vm(first, va(use_selected, -1.0, 1.0))
    )

    ax, ay = vs(vm(bx, active)), vs(vm(by, active))
    areg = vs(vm(regs, active))
    active_sig = vm(sigs[0], vl(active, 0, 1))
    active_handles = vm(hg[0], vl(active, 0, 1))
    for i in range(1, 4):
        active_sig = vv(active_sig, vm(sigs[i], vl(active, i, i + 1)))
        active_handles = vv(
            active_handles, vm(hg[i], vl(active, i, i + 1))
        )
    active_handles = vm(active_handles, hp)
    active_sumx, active_sumy = vs(vm(sumx, active)), vs(vm(sumy, active))
    active_reciprocal = vs(vm(reciprocal, active))

    pickups = mul(circles, sub(ONE, exclusion))
    pickup_supply = node("gmax", [mul(pickups, circle_sig)])
    ball_supply = sigs[0]
    for i in range(1, 4):
        ball_supply = vmax(ball_supply, sigs[i])
    supply = vmax(ball_supply, pickup_supply)

    color_count = ss(ring_sig)
    unavailable = ss(mul(ring_sig, va(supply, -1.0, 1.0)))
    valid = mul(
        mul(rings, sub(ONE, hard(unavailable, 0.5, 4.0))),
        hard(color_count, 0.5, 4.0)
    )
    multiple = mul(valid, hard(color_count, 1.5, 4.0))
    multiple_colors = node("gmax", [mul(multiple, ring_sig)])
    suppress = mul(
        mul(
            sub(ONE, hard(color_count, 1.5, 4.0)),
            hard(ss(mul(ring_sig, multiple_colors)), 0.5, 4.0)
        ),
        va(areg, -1.0, 1.0)
    )
    usable = mul(valid, sub(ONE, suppress))
    missing = node("relu", [sub(ring_sig, active_sig)])
    extras = node("relu", [sub(active_sig, ring_sig)])
    missing_count = ss(missing)
    impossible = ss(mul(missing, va(pickup_supply, -1.0, 1.0)))
    goal_error = add(
        add(ss(extras), impossible), mul(missing_count, areg)
    )
    goal_score = add(
        sa(dist(ax, ay), -1.0),
        add(sa(missing_count, -20.0), sa(goal_error, -10000.0))
    )
    goal_point = point_winner(goal_score, usable)
    goal = extract(goal_point, cat(X, Y, ring_sig))
    gx, gy = vl(goal, 0, 1), vl(goal, 1, 2)
    goal_sig = vl(goal, 2, 10)
    need = node("vrelu", [vd(goal_sig, active_sig)])
    eligible = mul(
        mul(pickups, hard(ss(mul(circle_sig, need)), 0.5, 4.0)),
        sub(ONE, hard(
            ss(mul(circle_sig, va(goal_sig, -1.0, 1.0))), 0.5, 4.0
        ))
    )
    pickup_score = add(sa(dist(ax, ay), -1.0), sa(dist(gx, gy), -0.2))
    pickup_point = point_winner(pickup_score, eligible)
    pickup_xy = extract(pickup_point, cat(X, Y))
    use_pickup = vm(vh(vs(need), 0.5, 4.0), node("gmax", [pickup_point]))
    target = vv(
        vm(pickup_xy, use_pickup),
        vm(vl(goal, 0, 2), va(use_pickup, -1.0, 1.0))
    )
    tx, ty = vl(target, 0, 1), vl(target, 1, 2)

    dilate_w = np.zeros((8, 8, 5, 5), np.float32)
    for i in range(8):
        for dy, dx in circle:
            dilate_w[i, i, dy + 2, dx + 2] = 1.0
    contacts = hard(
        conv(hard(conv(mul(pickups, circle_sig), dilate_w), 0.5, 4.0),
             dilate_w),
        0.5, 4.0
    )
    footprint = np.zeros((1, 1, 5, 5), np.float32)
    for dy, dx in circle:
        footprint[0, 0, dy + 2, dx + 2] = 1.0
    unsafe_ball = hard(
        conv(add(sl("x0", 10, 11), sl("x0", 2, 3)), footprint), 0.5, 4.0
    )
    wrong_contact = hard(
        ss(mul(contacts, va(goal_sig, -1.0, 1.0))), 0.5, 4.0
    )

    seed = sa(point_at(tx, ty), 256.0, -256.0)
    ceiling = sa(unsafe_ball, -1024.0)
    enclosing = G
    body = []
    G = body
    ps = fresh("s")
    W[ps] = 1
    shifts = np.zeros((4, 1, 3, 3), np.float32)
    for i, (dy, dx) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        shifts[i, 0, 1 + dy, 1 + dx] = 1.0
    shifted = conv(ps, shifts)
    edges = cat(
        hard(Y, 0.5, 4.0), hard(sa(Y, -1.0, 63.0), 0.5, 4.0),
        hard(X, 0.5, 4.0), hard(sa(X, -1.0, 63.0), 0.5, 4.0)
    )
    shifted = add(shifted, sa(edges, 1024.0, -1024.0))
    best = node("max2", [sl(shifted, 0, 1), sl(shifted, 1, 2)])
    best = node("max2", [best, sl(shifted, 2, 3)])
    best = node("max2", [best, sl(shifted, 3, 4)])
    pn = node("min2", [
        node("max2", [sa(best, 1.0, -1.0), seed]), ceiling
    ])
    G = enclosing
    pid = fresh("L")
    G.append({
        "id": pid, "op": "loop", "T": 128,
        "state": [{"id": ps, "init": seed}],
        "body": body, "next": {ps: pn}
    })
    path = pid + "." + ps
    W[path] = 1
    objective = add(
        path,
        add(
            sa(dist(tx, ty), -0.01),
            add(sa(unsafe_ball, -10000.0), sa(wrong_contact, -1000.0))
        )
    )
    predicted_x = mul(add(X, vd(active_sumx, hx)), active_reciprocal)
    predicted_y = mul(add(Y, vd(active_sumy, hy)), active_reciprocal)

    def warp(field, coordinate, predicted, reduction_op):
        nonlocal G
        enclosing = G
        body = []
        G = body
        acc, idx = fresh("s"), fresh("s")
        W[acc], W[idx] = 12, 1
        input_mask = hard(
            sa(ab(sub(coordinate, idx)), -1.0, 0.25), 0.0, 8.0
        )
        gathered = node(reduction_op, [mul(field, input_mask)])
        low = hard(sub(predicted, idx), -0.00002, 131072.0)
        high = hard(
            add(sa(predicted, -1.0), va(idx, 1.0, 0.99998)),
            0.0, 131072.0
        )
        an = add(acc, mul(gathered, mul(low, high)))
        ni = va(idx, 1.0, 1.0)
        G = enclosing
        lid = fresh("L")
        G.append({
            "id": lid, "op": "loop", "T": 64,
            "state": [
                {"id": acc, "zeros": [12, 64, 64]},
                {"id": idx, "init": VZERO}
            ],
            "body": body, "next": {acc: an, idx: ni}
        })
        result = lid + "." + acc
        W[result] = 12
        return result

    rows = warp(objective, Y, predicted_y, "colsum")
    move_objectives = warp(rows, X, predicted_x, "rowsum")

    handle_footprint = np.zeros((1, 1, 5, 5), np.float32)
    for dy, dx in diamond:
        handle_footprint[0, 0, dy + 2, dx + 2] = 1.0
    unsafe_handle = hard(
        conv(sl("x0", 2, 3), handle_footprint), 0.5, 4.0
    )
    visible_boxes = hard(
        conv(markers, np.ones((1, 1, 5, 5), np.float32)), 0.5, 4.0
    )
    remembered_boxes = hard(ss(mul(
        hard(sa(linf(hx, hy), -1.0, 2.25), 0.0, 8.0), hp
    )), 0.5, 4.0)
    boxes = node("max2", [visible_boxes, remembered_boxes])
    border = node("max2", [
        hard(sa(X, -1.0, 1.25)),
        node("max2", [
            hard(sa(Y, -1.0, 1.25)),
            node("max2", [
                hard(sa(X, 1.0, -61.25)),
                hard(sa(Y, 1.0, -61.25))
            ])
        ])
    ])
    blocked = node("max2", [
        border, node("max2", [unsafe_handle, boxes])
    ])
    move_scores = add(
        move_objectives,
        add(
            sa(blocked, -10000.0),
            add(
                sa(dist(hx, hy), -0.001),
                va(active_handles, 20000.0, -20000.0)
            )
        )
    )
    best_handle = winner(vv(
        node("gmax", [move_scores]), va(selected_id, 0.02)
    ))
    execute_move = vh(vs(vm(best_handle, selected_id)), 0.5, 4.0)
    chosen_score = ss(mul(move_scores, best_handle))
    chosen_x, chosen_y = vs(vm(hx, best_handle)), vs(vm(hy, best_handle))
    selection_score = sa(linf(chosen_x, chosen_y), -1.0)
    final_click = add(
        mul(chosen_score, execute_move),
        mul(selection_score, va(execute_move, -1.0, 1.0))
    )
    node("add", [final_click, ZERO], name="click")

    # Commit the actual decoded maximum, even if the moved handle becomes hidden.
    planned_point = point_winner(chosen_score, ONE)
    planned_xy = extract(planned_point, cat(X, Y))
    position_gate = vm(best_handle, execute_move)
    new_hx = vv(
        vm(hx, va(position_gate, -1.0, 1.0)),
        vm(vl(planned_xy, 0, 1), position_gate)
    )
    new_hy = vv(
        vm(hy, va(position_gate, -1.0, 1.0)),
        vm(vl(planned_xy, 1, 2), position_gate)
    )
    columns = vc(new_hx, new_hy, *hg, hp)
    pack = np.zeros((84, 84), np.float32)
    for i in range(12):
        for j in range(7):
            pack[i * 7 + j, j * 12 + i] = 1.0
    new_handles = dense(columns, pack)
    new_ball_data = vc(*(
        vc(vl(regs, i, i + 1), sigs[i]) for i in range(4)
    ))
    new_moved = clip(vv(moved, vm(active, execute_move)))
    memory_out = vc(
        new_handles, new_ball_data, gdata, best_handle,
        VONE, done, original, new_moved
    )
    node("vadd", [memory_out, VZERO], name="mem_next")

    logits_bias = np.zeros(8, np.float32)
    logits_bias[6] = 1.0
    dense("avail", np.zeros((8, 8), np.float32),
          logits_bias, name="logits")

    def prune(nodes, roots):
        needed, kept = set(roots), []
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                exports = {nd["id"] + "." + s["id"] for s in nd["state"]}
                if not (exports & needed):
                    continue
                nd = dict(nd)
                nd["body"] = prune(nd["body"], set(nd["next"].values()))
                internal = {s["id"] for s in nd["state"]}
                internal.update(n["id"] for n in nd["body"])
                external = set()
                for n in nd["body"]:
                    external.update(n.get("in", []))
                needed.update(external - internal)
                for s in nd["state"]:
                    if "init" in s:
                        needed.add(s["init"])
                kept.append(nd)
            elif nd["id"] in needed:
                needed.update(nd["in"])
                kept.append(nd)
        return list(reversed(kept))

    G = prune(G, {"click", "logits", "mem_next"})

    def fuse(nodes, roots):
        for n in nodes:
            if n["op"] == "loop":
                n["body"] = fuse(n["body"], set(n["next"].values()))
        uses = {}
        for n in nodes:
            for src in n.get("in", []):
                uses[src] = uses.get(src, 0) + 1
            if n["op"] == "loop":
                for s in n["state"]:
                    if "init" in s:
                        src = s["init"]
                        uses[src] = uses.get(src, 0) + 1
                internal = {s["id"] for s in n["state"]}
                internal.update(b["id"] for b in n["body"])
                for b in n["body"]:
                    for src in b.get("in", []):
                        if src not in internal:
                            uses[src] = uses.get(src, 0) + 1
        for root in roots:
            uses[root] = uses.get(root, 0) + 1
        by_id = {n["id"]: n for n in nodes}
        removed = set()
        for n in nodes:
            if n["op"] not in ("conv", "dense"):
                continue
            src = n["in"][0]
            parent = by_id.get(src)
            if parent is None or uses.get(src, 0) != 1:
                continue
            if parent["op"] != n["op"]:
                continue
            w2, w1 = P[n["w"]], P[parent["w"]]
            if n["op"] == "conv" and w2.shape[-1] != 1:
                continue
            b1 = P[parent["b"]] if "b" in parent else None
            b2 = P[n["b"]] if "b" in n else None
            if n["op"] == "dense":
                new_w = w2 @ w1
                new_b = (w2 @ b1 if b1 is not None
                         else np.zeros(w2.shape[0], np.float32))
            else:
                matrix = w2[:, :, 0, 0]
                new_w = np.einsum(
                    "oc,cijk->oijk", matrix, w1
                ).astype(np.float32)
                new_b = (matrix @ b1 if b1 is not None
                         else np.zeros(w2.shape[0], np.float32))
            if b2 is not None:
                new_b = new_b + b2
            n["in"] = list(parent["in"])
            n["w"], n["b"] = param(new_w), param(new_b)
            removed.add(src)
        return [n for n in nodes if n["id"] not in removed]

    G = fuse(G, {"click", "logits", "mem_next"})
    used = set()

    def collect(nodes):
        for n in nodes:
            for key in ("w", "b"):
                if key in n:
                    used.add(n[key])
            if n["op"] == "loop":
                collect(n["body"])

    collect(G)
    P = {k: v for k, v in P.items() if k in used}
    return {"graph": G, "memory_size": 237, "params": P}
