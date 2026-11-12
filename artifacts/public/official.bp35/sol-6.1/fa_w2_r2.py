import numpy as np


def build():
    P, G = {}, []
    dst = G
    dims = {"x0": 16, "coords": 3, "mem": 17,
            "aprev": 8, "avail": 8}
    serial = [0]
    pserial = [0]

    def param(a):
        pserial[0] += 1
        k = "p%d" % pserial[0]
        P[k] = np.asarray(a, dtype=np.float32)
        return k

    def node(op, ins, dim=None, name=None, **kw):
        serial[0] += 1
        ident = name or ("n%d" % serial[0])
        nd = {"id": ident, "op": op, "in": list(ins)}
        nd.update(kw)
        dst.append(nd)
        if dim is None:
            if op in ("concat", "vconcat"):
                dim = sum(dims[x] for x in ins)
            elif op in ("slice", "vslice"):
                dim = kw["stop"] - kw["start"]
            elif len(ins) == 2:
                dim = max(dims[x] for x in ins)
            else:
                dim = dims[ins[0]]
        dims[ident] = dim
        return ident

    def cv(x, w, b=None):
        w = np.asarray(w, np.float32)
        kw = {"w": param(w)}
        if b is not None:
            kw["b"] = param(b)
        return node("conv", [x], dim=w.shape[0], **kw)

    def sp(x, w, b=None):
        w = np.asarray(w, np.float32).reshape(-1, dims[x])
        return cv(x, w[:, :, None, None], b)

    def dn(x, w, b=None, name=None):
        w = np.asarray(w, np.float32).reshape(-1, dims[x])
        kw = {"w": param(w)}
        if b is not None:
            kw["b"] = param(b)
        return node("dense", [x], dim=w.shape[0], name=name, **kw)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def cat(*xs):
        return node("concat", xs)

    def vc(*xs):
        return node("vconcat", xs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def rel(x):
        return node("relu", [x])

    def gm(x):
        return node("gmax", [x])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def vr(x):
        return node("vrelu", [x])

    def shift(x, dy=0, dx=0):
        c = dims[x]
        w = np.zeros((c, c, 7, 7), np.float32)
        for i in range(c):
            w[i, i, 3 + dy // 2, 3 + dx // 2] = 1
        k = param(w)
        a = node("conv", [x], dim=c, w=k)
        return node("conv", [a], dim=c, w=k)

    def vertical(x):
        c = dims[x]
        w = np.zeros((c, c, 7, 7), np.float32)
        for i in range(c):
            w[i, i, 0 if i % 2 == 0 else 6, 3] = 1
        k = param(w)
        a = node("conv", [x], dim=c, w=k)
        return node("conv", [a], dim=c, w=k)

    def horizontal_pair(x):
        c = dims[x]
        w = np.zeros((2 * c, c, 7, 7), np.float32)
        for i in range(c):
            w[i, i, 3, 0] = 1
            w[c + i, i, 3, 6] = 1
        a = cv(x, w)
        w = np.zeros((2 * c, 2 * c, 7, 7), np.float32)
        for i in range(2 * c):
            w[i, i, 3, 0 if i < c else 6] = 1
        b = cv(a, w)
        return sl(b, 0, c), sl(b, c, 2 * c)

    def twin(x):
        return sp(x, np.ones((2, 1), np.float32))

    def opposite(x):
        c = dims[x]
        w = np.zeros((c, c), np.float32)
        for i in range(c):
            w[i, i ^ 1] = 1
        return sp(x, w)

    ones = sl("coords", 2, 3)
    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    ones2 = twin(ones)
    onev = gm(ones)
    zerov = dn(onev, [[0]])

    def cap(x):
        return node("min2", [x, ones])

    def sconst(c, v):
        return sp(ones, np.zeros((c, 1), np.float32),
                  np.full(c, v, np.float32))

    def positive(x):
        return cap(rel(sp(x, 1000 * np.eye(dims[x], dtype=np.float32))))

    # Player center, recognizing both orientations.
    w = np.zeros((2, 16, 7, 7), np.float32)
    for k, eye_dx in enumerate((1, -1)):
        for dy in (-2, -1, 0, 1):
            w[k, 9, 3 + dy, 3] = 1
        for dy in (-1, 0):
            w[k, 11, 3 + dy, 3 + eye_dx] = 1
    player = cap(sp(rel(cv("x0", w, [-5, -5])), [[1, 1]]))
    player_rows = node("rowmax", [player])

    # Recover the tile lattice from the player.
    body = []
    dims["phase_state"] = 1
    dst = body
    pn = cap(add("phase_state",
                 add(shift("phase_state", -6, 0),
                     shift("phase_state", 6, 0))))
    dst = G
    G.append({
        "id": "phase_loop", "op": "loop", "T": 11,
        "state": [{"id": "phase_state", "init": player_rows}],
        "body": body, "next": {"phase_state": pn},
    })
    dims["phase_loop.phase_state"] = 1

    centers = np.arange(3, 64, 6, dtype=np.float32)
    w = np.zeros((len(centers), 3), np.float32)
    w[:, 1] = 63
    xc = node("abs", [sp("coords", w, -centers)])
    peaks = rel(sp(xc, -np.eye(len(centers), dtype=np.float32),
                   np.ones(len(centers), np.float32)))
    xm = cap(rel(sp(peaks, np.ones((1, len(centers)), np.float32),
                    [-0.5])))
    xm = cap(sp(xm, [[4]]))
    play_area = cap(rel(sp("coords", [[-63, 0, 0]], [62])))
    anchors = mul(mul(xm, "phase_loop.phase_state"), play_area)

    def color(c):
        return mul(sl("x0", c, c + 1), anchors)

    w = np.zeros((1, 16, 7, 7), np.float32)
    for dy in (-1, 0, 1):
        w[0, 12, 3 + dy, 3] = 1
    purple = mul(rel(cv("x0", w, [-2])), anchors)
    ghost = rel(sub(color(12), purple))

    w = np.zeros((1, 16, 7, 7), np.float32)
    w[0, 15, 3, 3] = 1
    for dy in (-2, 2):
        for dx in (-1, 1):
            w[0, 11, 3 + dy, 3 + dx] = 1
    spike = cap(mul(rel(cv("x0", w, [-2])), anchors))
    orange = rel(sub(color(15), spike))
    empty = color(10)
    red, green, goal, wall = color(14), color(8), color(7), color(5)

    passing = cap(sp(cat(empty, ghost, color(9), goal),
                     [[1, 1, 1, 1]]))
    breaking = cap(sp(cat(red, purple, orange), [[1, 1, 1]]))
    allowed = cap(add(passing, breaking))
    resting = cap(add(add(wall, green), breaking))

    py = gm(mul(player, row))
    px = gm(mul(player, col))
    down = gm(cap(mul(ones, vr(dn(py, [[-63]], [34])))))
    gravity = dn(down, [[-1], [1]], [1, 0])
    green_exists = gm(green)
    cautious = gm(cap(mx(orange, mul(ones, vs("mem", 16, 17)))))
    blind_ok = dn(cautious, [[-1]], [1])

    # Diffuse possible orange growth and its source coordinates.
    orange_seed = cat(sp(orange, [[30]]),
                      mul(orange, row), mul(orange, col))
    body = []
    dims["growth_state"] = 3
    dst = body
    neighbors = [
        shift("growth_state", 0, 6),
        shift("growth_state", -6, 0),
        shift("growth_state", 6, 0),
        shift("growth_state", 0, -6),
    ]
    ns = [sl(x, 0, 1) for x in neighbors]
    best = mx(mx(ns[0], ns[1]), mx(ns[2], ns[3]))
    remaining, gates = ones, []
    for score in ns:
        eq = cap(rel(sp(sub(score, best), [[1000]], [1])))
        gate = mul(eq, remaining)
        gates.append(gate)
        remaining = sub(remaining, gate)
    sr = mul(gates[0], sl(neighbors[0], 1, 2))
    sc = mul(gates[0], sl(neighbors[0], 2, 3))
    for i in range(1, 4):
        sr = add(sr, mul(gates[i], sl(neighbors[i], 1, 2)))
        sc = add(sc, mul(gates[i], sl(neighbors[i], 2, 3)))
    new_score = mul(rel(sp(best, [[1]], [-1])), empty)
    reached = positive(new_score)
    gn = add(orange_seed,
             cat(new_score, mul(sr, reached), mul(sc, reached)))
    dst = G
    G.append({
        "id": "growth_loop", "op": "loop", "T": 10,
        "state": [{"id": "growth_state", "init": orange_seed}],
        "body": body, "next": {"growth_state": gn},
    })
    dims["growth_loop.growth_state"] = 3
    growth = "growth_loop.growth_state"
    growth_score = sl(growth, 0, 1)
    growable = mul(positive(growth_score), empty)
    grow_v = vertical(twin(growable))
    growcost_v = vertical(twin(sp(growth_score, [[-1]], [30])))
    growrow_v = vertical(twin(sl(growth, 1, 2)))
    growcol_v = vertical(twin(sl(growth, 2, 3)))

    pass_v = vertical(twin(passing))
    break_v = vertical(twin(breaking))
    ghost_v = vertical(twin(ghost))
    rest_v = vertical(twin(resting))
    allow2 = twin(allowed)
    break_l = shift(breaking, 0, -6)
    break_r = shift(breaking, 0, 6)
    allow_l = shift(allowed, 0, -6)
    allow_r = shift(allowed, 0, 6)
    pass_l = shift(passing, 0, -6)
    pass_r = shift(passing, 0, 6)

    top = cap(rel(sp("coords", [[-63, 0, 0]], [6])))
    bottom = cap(rel(sp("coords", [[63, 0, 0]], [-56])))
    frontier = mul(cat(top, bottom), allow2)
    frontier = mul(frontier, blind_ok)
    seed = mx(sp(twin(goal), 100 * np.eye(2, dtype=np.float32)),
              sp(frontier, 50 * np.eye(2, dtype=np.float32)))

    # Only distant supports receive exploration reward. Current-height
    # supports therefore cannot create a back-and-forth reward plateau.
    directed_distance = sp(
        sub(twin(row), py), [[-63, 0], [0, 63]], [-17, -17])
    distant = cap(rel(directed_distance))
    remote_reward = mul(
        sp(mul(distant, allow2), 50 * np.eye(2, dtype=np.float32)),
        cautious)

    cost_l = add(ones2, break_l)
    cost_r = add(ones2, break_r)
    c_one2 = sconst(2, 1)
    c_fall2 = sconst(2, 0.35)

    def bellman_parts(value, flip=None):
        lv, rv = horizontal_pair(value)
        vv = vertical(value)
        stand = mx(rel(sub(lv, cost_l)), rel(sub(rv, cost_r)))
        stand = mx(stand, mul(rel(sub(vv, c_one2)), break_v))
        if flip is not None:
            stand = mx(stand, flip)
        stand = mx(stand, remote_reward)
        fall = mul(rel(sub(vv, c_fall2)), pass_v)
        gs = mul(rel(sub(stand, c_one2)), ghost_v)
        os = mul(rel(sub(stand, growcost_v)), grow_v)
        result = mul(
            mx(seed, mx(fall, mx(mul(stand, rest_v), mx(gs, os)))),
            allow2)
        return result, fall, gs, os, lv, rv, vv

    # Plan with no switch remaining.
    body = []
    dims["base_state"] = 2
    dst = body
    bn, _, _, _, _, _, _ = bellman_parts("base_state")
    dst = G
    G.append({
        "id": "base_loop", "op": "loop", "T": 64,
        "state": [{"id": "base_state", "zeros": [2, 64, 64]}],
        "body": body, "next": {"base_state": bn},
    })
    dims["base_loop.base_state"] = 2
    base_value = "base_loop.base_state"
    _, bf, bgs, bos, _, _, _ = bellman_parts(base_value)
    opposite_base = opposite(base_value)
    flip_candidate = mul(rel(sub(opposite_base, c_one2)), green_exists)

    # Plan with at most one switch: its destination uses the no-switch map.
    body = []
    dims["value_state"] = 2
    dst = body
    vn, _, _, _, _, _, _ = bellman_parts("value_state", flip_candidate)
    dst = G
    G.append({
        "id": "value_loop", "op": "loop", "T": 64,
        "state": [{"id": "value_state", "init": base_value}],
        "body": body, "next": {"value_state": vn},
    })
    dims["value_loop.value_state"] = 2
    value = "value_loop.value_state"
    _, fall, gs, os, lv, rv, vv = bellman_parts(value, flip_candidate)

    support_row = sp("coords", [[1, 0, 0], [1, 0, 0]],
                     [-6 / 63, 6 / 63])
    support_col = sp("coords", [[0, 1, 0], [0, 1, 0]])
    repeat = np.zeros((6, 2), np.float32)
    for i in range(6):
        repeat[i, i % 2] = 1

    def label_map(prefix, fall_field, ghost_stop, growth_stop):
        nonlocal dst
        gc = mul(positive(sub(ghost_stop, mx(fall_field, growth_stop))),
                 ghost_v)
        oc = mul(positive(sub(growth_stop, mx(fall_field, ghost_stop))),
                 grow_v)
        choice = cap(add(gc, oc))
        continuation = mul(mul(pass_v, sub(ones2, choice)), allow2)
        lr = add(mul(gc, support_row), mul(oc, growrow_v))
        lc = add(mul(gc, support_col), mul(oc, growcol_v))
        source = cat(choice, lr, lc)
        cont6 = sp(continuation, repeat)
        sid = prefix + "_state"
        lid = prefix + "_loop"
        dims[sid] = 6
        body = []
        dst = body
        ln = add(source, mul(vertical(sid), cont6))
        dst = G
        G.append({
            "id": lid, "op": "loop", "T": 11,
            "state": [{"id": sid, "zeros": [6, 64, 64]}],
            "body": body, "next": {sid: ln},
        })
        out = lid + "." + sid
        dims[out] = 6
        return out

    base_labels = label_map("base_label", bf, bgs, bos)
    labels = label_map("route_label", fall, gs, os)
    label_l = shift(labels, 0, -6)
    label_r = shift(labels, 0, 6)
    label_v = vertical(labels)
    label_o = opposite(base_labels)

    # Actual unmodified falling safety, used for low-value exploration.
    safety_seed = mx(frontier, twin(goal))
    body = []
    dims["safe_state"] = 2
    dst = body
    sn = cap(mx(
        safety_seed,
        mul(allow2, add(rest_v, mul(pass_v, vertical("safe_state"))))))
    dst = G
    G.append({
        "id": "safe_loop", "op": "loop", "T": 11,
        "state": [{"id": "safe_state", "zeros": [2, 64, 64]}],
        "body": body, "next": {"safe_state": sn},
    })
    dims["safe_loop.safe_state"] = 2
    safe_l = shift("safe_loop.safe_state", 0, -6)
    safe_r = shift("safe_loop.safe_state", 0, 6)

    def pick(x):
        return gm(mul(sp(mul(x, gravity), [[1, 1]]), player))

    def pick_single(x):
        return gm(mul(x, player))

    fvals = vc(pick(lv), pick(rv), pick(vv), pick(opposite_base))
    bl, br, bs = pick_single(break_l), pick_single(break_r), pick(break_v)
    valid = vc(pick_single(allow_l), pick_single(allow_r),
               bs, green_exists)
    scores = va(
        dn(fvals, np.eye(4, dtype=np.float32),
           [-1, -0.98, -0.96, -0.94]),
        dn(vc(bl, br, zerov, zerov), -np.eye(4, dtype=np.float32)))

    flags, rows, cols = [], [], []
    for lab in (label_l, label_r, label_v, label_o):
        flags.append(pick(sl(lab, 0, 2)))
        rows.append(pick(sl(lab, 2, 4)))
        cols.append(pick(sl(lab, 4, 6)))
    flag4 = vc(*flags)
    noflag = dn(flag4, -np.eye(4, dtype=np.float32),
                np.ones(4, np.float32))
    step_y = dn(gravity, [[-6 / 63, 6 / 63]])
    direct_rows = vc(py, py, va(py, step_y), zerov)
    direct_cols = vc(dn(px, [[1]], [-6 / 63]),
                     dn(px, [[1]], [6 / 63]), px, zerov)
    target_rows = va(vm(flag4, vc(*rows)), vm(noflag, direct_rows))
    target_cols = va(vm(flag4, vc(*cols)), vm(noflag, direct_cols))

    raw_click = va(vc(bl, br, onev, zerov), flag4)
    overflow = vr(dn(raw_click, np.eye(4, dtype=np.float32),
                      -np.ones(4, np.float32)))
    click_eligible = vm(
        va(raw_click, dn(overflow, -np.eye(4, dtype=np.float32))), valid)

    row4 = sp("coords", np.tile([[1, 0, 0]], (4, 1)))
    col4 = sp("coords", np.tile([[0, 1, 0]], (4, 1)))
    distance = add(node("abs", [sub(row4, target_rows)]),
                   node("abs", [sub(col4, target_cols)]))
    invalid_click = dn(click_eligible, 1000 * np.eye(4, dtype=np.float32),
                       np.full(4, -1000, np.float32))
    point_maps = add(sp(distance, -6300 * np.eye(4, dtype=np.float32)),
                     va(scores, invalid_click))
    point_click = mx(mx(sl(point_maps, 0, 1), sl(point_maps, 1, 2)),
                     mx(sl(point_maps, 2, 3), sl(point_maps, 3, 4)))

    flip_score = gm(mx(mul(ones, vs(scores, 3, 4)), sconst(1, 0.20)))
    green_map = add(
        sp(green, [[1000]], [-1000]),
        va(flip_score, dn(flags[3], [[-1000]])))
    click_base = mx(point_click, green_map)

    wr = np.zeros((4, 17), np.float32)
    wc = np.zeros((4, 17), np.float32)
    for i in range(4):
        wr[i, 2 * i] = 1
        wc[i, 2 * i + 1] = 1
    old_dist = add(
        node("abs", [sub(row4, dn("mem", wr))]),
        node("abs", [sub(col4, dn("mem", wc))]))
    hits = rel(sp(old_dist, -126 * np.eye(4, dtype=np.float32),
                  np.ones(4, np.float32)))
    click = node("add",
                 [click_base, sp(hits, [[-0.08, -0.08, -0.08, -0.08]])],
                 name="click")

    move_pass = vc(pick_single(pass_l), pick_single(pass_r))
    safe_moves = vc(pick(safe_l), pick(safe_r))
    move_eligible = vm(vm(move_pass, vs(noflag, 0, 2)), safe_moves)
    move_scores = va(
        vs(scores, 0, 2),
        dn(move_eligible, 1000 * np.eye(2, dtype=np.float32),
           [-1000, -1000]))
    inertia_w = np.zeros((2, 8), np.float32)
    inertia_w[0, 3] = inertia_w[1, 4] = 0.20
    move_scores = va(move_scores, dn("aprev", inertia_w))
    click_peak = gm(click)

    wo = np.zeros((8, 3), np.float32)
    bo = np.full(8, -10000, np.float32)
    wo[3, 0] = wo[4, 1] = wo[6, 2] = 1
    bo[3] = bo[4] = bo[6] = 0
    bo[7] = -1500
    raw_logits = dn(vc(move_scores, click_peak), wo, bo)

    undone = vm(vs("aprev", 7, 8), vs("mem", 8, 16))
    undo_w = np.zeros((8, 8), np.float32)
    undo_w[3, 3] = undo_w[4, 4] = undo_w[6, 6] = -1000
    node("vadd", [raw_logits, dn(undone, undo_w)], name="logits")

    winner = cap(rel(sp(sub(click, click_peak), [[1000]], [1])))
    new_row = gm(mul(winner, row))
    new_col = gm(mul(winner, col))
    node("vconcat",
         [new_row, new_col, vs("mem", 0, 6), "aprev", cautious],
         name="mem_next")

    return {"graph": G, "memory_size": 17, "params": P}
