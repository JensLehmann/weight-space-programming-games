import numpy as np


def build():
    P = {}
    graph = []
    G = graph
    serial = 0
    shift_params = {}

    def node(op, ins, **fields):
        nonlocal serial
        serial += 1
        name = "n%d" % serial
        d = {"id": name, "op": op, "in": list(ins)}
        d.update(fields)
        G.append(d)
        return name

    def conv(x, w, b=None):
        name = "w%d" % (serial + 1)
        P[name] = np.asarray(w, np.float32)
        fields = {"w": name}
        if b is not None:
            bn = "b%d" % (serial + 1)
            P[bn] = np.asarray(b, np.float32)
            fields["b"] = bn
        return node("conv", [x], **fields)

    def aff(x, a=1.0, b=0.0):
        return conv(x, np.array([[[[a]]]], np.float32), [b])

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def relu(a):
        return node("relu", [a])

    def pool(a):
        return node("gmax", [a])

    def sl(a, start, stop):
        return node("slice", [a], start=start, stop=stop)

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    zero = aff(one, 0.0)
    c1 = aff(one, 0.0, 1.0)
    c3 = aff(one, 0.0, 3.0)
    c100 = aff(one, 0.0, 100.0)
    row_px = aff(row, 63.0)
    col_px = aff(col, 63.0)

    def clip(a):
        return node("min2", [relu(a), one])

    def inv(a):
        return sub(one, a)

    def positive(a):
        return clip(aff(a, 4.0, -0.01))

    def greater(a, b):
        return clip(aff(sub(a, b), 20.0, -0.02))

    def summ(xs):
        z = xs[0]
        for a in xs[1:]:
            z = add(z, a)
        return z

    def maxs(xs):
        z = xs[0]
        for a in xs[1:]:
            z = mx(z, a)
        return z

    def broadcast(a):
        return add(zero, a)

    def shift(a, dy=0, dx=0):
        key = (dy, dx)
        if key not in shift_params:
            assert abs(dy) + abs(dx) == 6
            w = np.zeros((1, 1, 7, 7), np.float32)
            w[0, 0, 3 + dy // 2, 3 + dx // 2] = 1.0
            name = "shift_%d_%d" % (dy + 6, dx + 6)
            P[name] = w
            shift_params[key] = name
        wname = shift_params[key]
        a = node("conv", [a], w=wname)
        return node("conv", [a], w=wname)

    # Reconstruct central tile colours on the UI-overwritten row.
    # The central columns of the canonical tiles repeat two rows up.
    ui_row = clip(aff(row_px, 1.0, -62.0))
    w = np.zeros((16, 16, 5, 5), np.float32)
    for i in range(16):
        w[i, i, 0, 2] = 1.0
    above_ui = conv("x0", w)
    vision = add(
        mul("x0", inv(ui_row)),
        mul(above_ui, ui_row),
    )

    # Lower central yellow pixel of either player orientation.
    w = np.zeros((1, 16, 3, 3), np.float32)
    w[0, 9, 0, 1] = 1
    w[0, 9, 1, 1] = 1
    w[0, 9, 2, 1] = 1
    w[0, 5, 2, 0] = 1
    w[0, 5, 2, 2] = 1
    player = clip(conv("x0", w, [-4]))

    # Six-pixel tile lattice.
    centres = np.arange(3, 64, 6, dtype=np.float32)
    w = np.full((len(centres), 1, 1, 1), 63.0, np.float32)
    distances = node("abs", [conv(col, w, -centres)])
    w = np.zeros((len(centres), len(centres), 1, 1), np.float32)
    for i in range(len(centres)):
        w[i, i, 0, 0] = -4
    phase = relu(conv(distances, w, np.full(len(centres), 2)))
    phase = node("min2", [phase, one])
    x_phase = conv(phase, np.ones((1, len(centres), 1, 1)))

    player_row_mask = node("rowmax", [player])
    outer = G
    body = []
    G = body
    w = np.zeros((1, 1, 7, 7), np.float32)
    w[0, 0, 0, 3] = 1
    w[0, 0, 6, 3] = 1
    P["phase_kernel"] = w
    a = node("conv", ["phase_state"], w="phase_kernel")
    a = node("conv", [a], w="phase_kernel")
    phase_next = clip(mx(a, "phase_state"))
    G = outer
    G.append({
        "id": "phase_loop",
        "op": "loop",
        "T": 11,
        "state": [{"id": "phase_state", "init": player_row_mask}],
        "body": body,
        "next": {"phase_state": phase_next},
    })
    lattice = mul("phase_loop.phase_state", x_phase)

    colours = {
        c: sl(vision, c, c + 1)
        for c in (7, 8, 9, 10, 14, 15)
    }

    def detect(taps, bias):
        w = np.zeros((1, 16, 5, 5), np.float32)
        for c, dy, dx, weight in taps:
            w[0, c, 2 + dy, 2 + dx] += weight
        return relu(conv(vision, w, [bias]))

    purple_raw = detect([(12, 0, 0, 1), (12, -2, 0, 1)], -1)
    ghost_raw = detect([(12, 0, 0, 1), (12, -2, 0, -1)], 0)
    white_raw = detect([(15, 0, 0, 1), (3, -2, -2, 1)], -1)

    red = mul(lattice, colours[14])
    green = mul(lattice, colours[8])
    purple = mul(lattice, purple_raw)
    ghost = mul(lattice, ghost_raw)
    white = mul(lattice, white_raw)
    goal = mul(lattice, colours[7])
    spike = mul(lattice, relu(sub(colours[15], white_raw)))
    empty = mul(lattice, summ([
        colours[10], colours[9], ghost_raw, colours[7]
    ]))
    plain_empty = mul(lattice, colours[10])
    removable = summ([red, purple, white])
    walk = add(empty, removable)
    floor = relu(sub(sub(lattice, empty), spike))

    has_green = broadcast(pool(green))
    has_white = broadcast(pool(white))
    has_ghost = broadcast(pool(ghost))
    careful = mx(has_white, has_ghost)

    row_code = aff(row_px, 1, 1)
    col_code = aff(col_px, 1, 1)
    player_row = pool(mul(player, row_px))
    normal = clip(aff(broadcast(player_row), 1, -34))
    inverse = inv(normal)

    def choose(u, d):
        return add(mul(normal, u), mul(inverse, d))

    prefer_u = mul(normal, inv(has_green))
    prefer_d = inv(prefer_u)

    frontier_u = clip(aff(row_px, -1, 6))
    frontier_d = clip(aff(row_px, 1, -54))

    ordinary_local_u = aff(row, -3, 6)
    ordinary_local_d = aff(row, 3, 3)
    safe_local_u = mul(
        aff(row, -40, 60), aff(prefer_u, 0.5, 0.5)
    )
    safe_local_d = mul(
        aff(row, 40, 20), aff(prefer_d, 0.5, 0.5)
    )
    local_u = add(
        ordinary_local_u,
        mul(careful, sub(safe_local_u, ordinary_local_u)),
    )
    local_d = add(
        ordinary_local_d,
        mul(careful, sub(safe_local_d, ordinary_local_d)),
    )

    goal_value = aff(goal, 240)
    base_u = mul(walk, mx(local_u, goal_value))
    base_d = mul(walk, mx(local_d, goal_value))

    ghost_only = mul(has_ghost, inv(has_white))
    exit_scale = sub(
        aff(has_white, -0.68, 1.0),
        aff(ghost_only, 0.58),
    )
    exit_u = mul(
        mul(mul(walk, frontier_u), aff(prefer_u, 55, 55)),
        exit_scale,
    )
    exit_d = mul(
        mul(mul(walk, frontier_d), aff(prefer_d, 55, 55)),
        exit_scale,
    )

    walk_l = shift(walk, dx=-6)
    walk_r = shift(walk, dx=6)
    rem_l = shift(removable, dx=-6)
    rem_r = shift(removable, dx=6)
    rem_u = shift(removable, dy=-6)
    rem_d = shift(removable, dy=6)
    floor_u = shift(floor, dy=-6)
    floor_d = shift(floor, dy=6)
    empty_u = shift(empty, dy=-6)
    empty_d = shift(empty, dy=6)
    ghost_u = shift(ghost, dy=-6)
    ghost_d = shift(ghost, dy=6)

    deletion_cost = add(removable, white)
    cost_l = add(c1, shift(deletion_cost, dx=-6))
    cost_r = add(c1, shift(deletion_cost, dx=6))

    # White expansion wavefront and pointers to the next existing
    # white block to click.
    white_seed_score = aff(white, 80)
    white_seed_row = mul(white, row_code)
    white_seed_col = mul(white, col_code)

    outer = G
    body = []
    G = body
    best = "spread_score"
    ptr_r = "spread_row"
    ptr_c = "spread_col"
    for dy, dx in ((-6, 0), (6, 0), (0, -6), (0, 6)):
        candidate = mul(
            relu(aff(shift("spread_score", dy, dx), 1, -1)),
            plain_empty,
        )
        win = greater(candidate, best)
        nr = shift("spread_row", dy, dx)
        nc = shift("spread_col", dy, dx)
        ptr_r = add(ptr_r, mul(win, sub(nr, ptr_r)))
        ptr_c = add(ptr_c, mul(win, sub(nc, ptr_c)))
        best = mx(best, candidate)
    G = outer
    G.append({
        "id": "spread",
        "op": "loop",
        "T": 10,
        "state": [
            {"id": "spread_score", "init": white_seed_score},
            {"id": "spread_row", "init": white_seed_row},
            {"id": "spread_col", "init": white_seed_col},
        ],
        "body": body,
        "next": {
            "spread_score": best,
            "spread_row": ptr_r,
            "spread_col": ptr_c,
        },
    })

    spread_score = "spread.spread_score"
    spread_row = "spread.spread_row"
    spread_col = "spread.spread_col"
    buildable = mul(plain_empty, positive(spread_score))
    build_cost = aff(spread_score, -1.35, 109.0)
    build_u = shift(buildable, dy=-6)
    build_d = shift(buildable, dy=6)
    build_cost_u = shift(build_cost, dy=-6)
    build_cost_d = shift(build_cost, dy=6)

    # Safe arrival means a visible non-spike landing, a goal, or a
    # landing that can be prepared using a visible ghost/white block.
    outer = G
    body = []
    G = body
    safe_u_next = mul(clip(maxs([
        goal, floor_u, ghost_u, build_u,
        mul(shift("safe_u", dy=-6), empty_u),
    ])), walk)
    safe_d_next = mul(clip(maxs([
        goal, floor_d, ghost_d, build_d,
        mul(shift("safe_d", dy=6), empty_d),
    ])), walk)
    G = outer
    G.append({
        "id": "safety",
        "op": "loop",
        "T": 11,
        "state": [
            {"id": "safe_u", "zeros": [1, 64, 64]},
            {"id": "safe_d", "zeros": [1, 64, 64]},
        ],
        "body": body,
        "next": {
            "safe_u": safe_u_next,
            "safe_d": safe_d_next,
        },
    })
    safe_u = "safety.safe_u"
    safe_d = "safety.safe_d"

    def move_value(f, cost, allowed):
        return mul(relu(sub(f, cost)), allowed)

    # Bellman-style maps: decision and compulsory-fall values for both
    # gravities, with either zero or one future switch.
    outer = G
    body = []
    G = body

    def decision(f, base, support_rem, dy, flip=None, safe_flip=None):
        left = move_value(shift(f, dx=-6), cost_l, walk_l)
        right = move_value(shift(f, dx=6), cost_r, walk_r)
        below = shift(f, dy=dy)
        delete = mul(relu(sub(below, c1)), support_rem)
        candidates = [base, left, right, delete]
        if flip is not None:
            flip_value = mul(
                mul(relu(sub(flip, c3)), has_green),
                safe_flip,
            )
            candidates.append(flip_value)
        return mul(maxs(candidates), walk), below

    def falling(v, below, fl, em, gh, bu, bc, ex):
        ordinary = add(mul(v, fl), mul(below, em))
        ghost_stop = mul(relu(sub(v, c1)), gh)
        white_stop = mul(relu(sub(v, bc)), bu)
        return mul(maxs([
            ordinary, ghost_stop, white_stop, ex, goal_value
        ]), walk)

    v0u_next, f0un = decision("f0u", base_u, rem_u, -6)
    v0d_next, f0dn = decision("f0d", base_d, rem_d, 6)
    f0u_next = falling(
        "v0u", f0un, floor_u, empty_u, ghost_u,
        build_u, build_cost_u, exit_u,
    )
    f0d_next = falling(
        "v0d", f0dn, floor_d, empty_d, ghost_d,
        build_d, build_cost_d, exit_d,
    )
    vu_next, fun = decision(
        "fu", base_u, rem_u, -6, "f0d", safe_d
    )
    vd_next, fdn = decision(
        "fd", base_d, rem_d, 6, "f0u", safe_u
    )
    fu_next = falling(
        "vu", fun, floor_u, empty_u, ghost_u,
        build_u, build_cost_u, exit_u,
    )
    fd_next = falling(
        "vd", fdn, floor_d, empty_d, ghost_d,
        build_d, build_cost_d, exit_d,
    )

    G = outer
    G.append({
        "id": "plan",
        "op": "loop",
        "T": 52,
        "state": [
            {"id": "v0u", "init": base_u},
            {"id": "v0d", "init": base_d},
            {"id": "f0u", "zeros": [1, 64, 64]},
            {"id": "f0d", "zeros": [1, 64, 64]},
            {"id": "vu", "init": base_u},
            {"id": "vd", "init": base_d},
            {"id": "fu", "zeros": [1, 64, 64]},
            {"id": "fd", "zeros": [1, 64, 64]},
        ],
        "body": body,
        "next": {
            "v0u": v0u_next, "v0d": v0d_next,
            "f0u": f0u_next, "f0d": f0d_next,
            "vu": vu_next, "vd": vd_next,
            "fu": fu_next, "fd": fd_next,
        },
    })

    vu, vd = "plan.vu", "plan.vd"
    fu, fd = "plan.fu", "plan.fd"
    f0u, f0d = "plan.f0u", "plan.f0d"

    # Identify required landing preparations and carry their actual
    # click coordinates backward along each fall.
    def preparation(v, f, dy, fl, em, gh, bu, bc):
        below = shift(f, dy=dy)
        ordinary = add(mul(v, fl), mul(below, em))
        ghost_value = mul(relu(sub(v, c1)), gh)
        white_value = mul(relu(sub(v, bc)), bu)
        ghost_stop = mul(
            mul(gh, greater(ghost_value, ordinary)), inv(goal)
        )
        white_stop = mul(
            mul(bu, greater(white_value, ordinary)), inv(goal)
        )
        stop = add(ghost_stop, white_stop)
        rr = add(
            mul(aff(row_code, 1, dy), ghost_stop),
            mul(shift(spread_row, dy=dy), white_stop),
        )
        cc = add(
            mul(col_code, ghost_stop),
            mul(shift(spread_col, dy=dy), white_stop),
        )
        continuation = mul(mul(em, inv(stop)), inv(goal))
        return rr, cc, continuation

    ur, uc, cont_u = preparation(
        vu, fu, -6, floor_u, empty_u, ghost_u,
        build_u, build_cost_u,
    )
    dr, dc, cont_d = preparation(
        vd, fd, 6, floor_d, empty_d, ghost_d,
        build_d, build_cost_d,
    )
    u0r, u0c, cont_0u = preparation(
        "plan.v0u", f0u, -6, floor_u, empty_u, ghost_u,
        build_u, build_cost_u,
    )
    d0r, d0c, cont_0d = preparation(
        "plan.v0d", f0d, 6, floor_d, empty_d, ghost_d,
        build_d, build_cost_d,
    )

    outer = G
    body = []
    G = body
    updates = {}
    for tag, dy, rr, cc, cont in (
        ("u", -6, ur, uc, cont_u),
        ("d", 6, dr, dc, cont_d),
        ("zu", -6, u0r, u0c, cont_0u),
        ("zd", 6, d0r, d0c, cont_0d),
    ):
        for suffix, seed in (("r", rr), ("c", cc)):
            sid = "ray_" + tag + suffix
            updates[sid] = add(seed, mul(shift(sid, dy=dy), cont))
    G = outer
    G.append({
        "id": "rays",
        "op": "loop",
        "T": 11,
        "state": [
            {"id": sid, "zeros": [1, 64, 64]}
            for sid in updates
        ],
        "body": body,
        "next": updates,
    })

    current_f = choose(fu, fd)
    opposite_f = choose(f0d, f0u)
    opposite_safe = choose(safe_d, safe_u)
    support_f = choose(shift(fu, dy=-6), shift(fd, dy=6))
    support_rem = choose(rem_u, rem_d)

    score_l = move_value(shift(current_f, dx=-6), cost_l, walk_l)
    score_r = move_value(shift(current_f, dx=6), cost_r, walk_r)
    score_s = mul(relu(sub(support_f, c1)), support_rem)
    score_flip = mul(
        mul(relu(sub(opposite_f, c3)), has_green),
        opposite_safe,
    )

    ray_r = choose("rays.ray_ur", "rays.ray_dr")
    ray_c = choose("rays.ray_uc", "rays.ray_dc")
    flip_ray_r = choose("rays.ray_zdr", "rays.ray_zur")
    flip_ray_c = choose("rays.ray_zdc", "rays.ray_zuc")

    lr, lc = shift(ray_r, dx=-6), shift(ray_c, dx=-6)
    rr, rc = shift(ray_r, dx=6), shift(ray_c, dx=6)
    sr = choose(
        shift("rays.ray_ur", dy=-6),
        shift("rays.ray_dr", dy=6),
    )
    sc = choose(
        shift("rays.ray_uc", dy=-6),
        shift("rays.ray_dc", dy=6),
    )

    prep_l, prep_r = positive(lr), positive(rr)
    prep_s, prep_flip = positive(sr), positive(flip_ray_r)

    move_l = mul(
        mul(relu(sub(walk_l, rem_l)), inv(prep_l)),
        positive(score_l),
    )
    move_r = mul(
        mul(relu(sub(walk_r, rem_r)), inv(prep_r)),
        positive(score_r),
    )
    left_value = pool(mul(player, mul(score_l, move_l)))
    right_value = pool(mul(player, mul(score_r, move_r)))
    left_valid = pool(mul(player, move_l))
    right_valid = pool(mul(player, move_r))

    pleft = shift(player, dx=6)
    pright = shift(player, dx=-6)
    psupport = choose(
        shift(player, dy=6), shift(player, dy=-6)
    )

    def scored(target, value):
        return sub(mul(target, add(c100, value)), c100)

    maps = []
    for target, value, preparation_mask in (
        (mul(pleft, removable), score_l, prep_l),
        (mul(pright, removable), score_r, prep_r),
        (mul(psupport, removable), score_s, prep_s),
    ):
        value = pool(mul(
            player, mul(value, inv(preparation_mask))
        ))
        maps.append(scored(target, value))

    def pointer_click(pointer_r, pointer_c, value):
        r = pool(mul(player, pointer_r))
        c = pool(mul(player, pointer_c))
        rm = clip(aff(node("abs", [sub(row_code, r)]), -1, 1))
        cm = clip(aff(node("abs", [sub(col_code, c)]), -1, 1))
        target = mul(mul(rm, cm), add(ghost, white))
        return scored(target, pool(mul(player, value)))

    maps.extend([
        pointer_click(lr, lc, score_l),
        pointer_click(rr, rc, score_r),
        pointer_click(sr, sc, score_s),
        pointer_click(flip_ray_r, flip_ray_c, score_flip),
    ])

    no_flip_prep = pool(mul(player, inv(prep_flip)))
    flip_target = mul(green, no_flip_prep)
    flip_value = pool(mul(
        player, mul(score_flip, inv(prep_flip))
    ))
    flip_map = scored(flip_target, flip_value)
    pc = pool(mul(player, col_px))
    proximity = add(
        node("abs", [sub(col_px, pc)]),
        node("abs", [sub(row_px, player_row)]),
    )
    flip_map = sub(flip_map, aff(proximity, 0.0005))
    maps.append(flip_map)

    click_map = maxs(maps)
    graph.append({
        "id": "click",
        "op": "add",
        "in": [click_map, c100],
    })
    click_value = pool(click_map)

    features = node("vconcat", [
        left_value, right_value, click_value,
        left_valid, right_valid, "aprev",
    ])
    w = np.zeros((8, 13), np.float32)
    b = np.full(8, -100, np.float32)
    b[0] = -40
    b[7] = -15

    w[3, 0] = 1
    w[3, 3] = 20
    w[3, 9] = -0.25
    b[3] = -19.99

    w[4, 1] = 1
    w[4, 4] = 20
    w[4, 8] = -0.25
    b[4] = -19.98

    w[6, 2] = 1
    b[6] = 0.03
    P["output_w"] = w
    P["output_b"] = b
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [features],
        "w": "output_w",
        "b": "output_b",
    })

    return {"graph": graph, "memory_size": 0, "params": P}
