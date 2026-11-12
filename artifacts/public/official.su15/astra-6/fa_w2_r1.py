import math
import numpy as np


def build():
    N = 12
    B = 8
    M = 3 * B + 9

    graph = []
    params = {}
    target = graph
    serial = 0
    pserial = 0
    constants = {}
    shared = {}

    def parameter(a):
        nonlocal pserial
        name = "p%d" % pserial
        pserial += 1
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, ins=None, name=None, **kw):
        nonlocal serial
        if name is None:
            name = "n%d" % serial
            serial += 1
        d = {"id": name, "op": op}
        if ins is not None:
            d["in"] = list(ins)
        d.update(kw)
        target.append(d)
        return name

    def dense(x, w, b=None, name=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("dense", [x], name=name, **kw)

    def conv(x, w, b=None, name=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("conv", [x], name=name, **kw)

    def constant(v):
        nonlocal target
        a = np.asarray(v, dtype=np.float32).reshape(-1)
        key = a.tobytes()
        if key not in constants:
            saved = target
            target = graph
            constants[key] = dense(
                "avail", np.zeros((len(a), 8), np.float32), a
            )
            target = saved
        return constants[key]

    def c(z):
        return constant([z])

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def scale(a, z):
        return a if z == 1 else mul(a, c(z))

    def bias(a, z):
        return a if z == 0 else add(a, c(z))

    def ab(a):
        return node("abs", [a])

    def relu(a):
        return node("relu", [a])

    def minimum(a, b):
        return node("min2", [a, b])

    def maximum(a, b):
        return node("max2", [a, b])

    def clip(a):
        return minimum(relu(a), c(1))

    def complement(a):
        return sub(c(1), a)

    def vadd(a, b):
        return node("vadd", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def vs(a, z):
        return a if z == 1 else vmul(a, c(z))

    def vb(a, z):
        return a if z == 0 else vadd(a, c(z))

    def vsub(a, b):
        return vadd(a, vs(b, -1))

    def vr(a):
        return node("vrelu", [a])

    def va(a):
        return vadd(vr(a), vr(vs(a, -1)))

    def vc(a):
        r = vr(a)
        return vsub(r, vr(vb(a, -1)))

    def veq(a, b):
        return vr(vb(vs(va(vsub(a, b)), -1), 1))

    def vcomp(a):
        return vb(vs(a, -1), 1)

    def sl(a, i, j):
        return node("slice", [a], start=i, stop=j)

    def vl(a, i, j):
        return node("vslice", [a], start=i, stop=j)

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

    def sum_channels(x, n):
        key = ("sum", n)
        if key not in shared:
            shared[key] = parameter(np.ones((1, n, 1, 1), np.float32))
        return node("conv", [x], w=shared[key])

    def sum_vector(x, n):
        key = ("vsum", n)
        if key not in shared:
            shared[key] = parameter(np.ones((1, n), np.float32))
        return node("dense", [x], w=shared[key])

    def rotate_vector(x, n):
        key = ("rotv", n)
        if key not in shared:
            w = np.zeros((n, n), np.float32)
            for i in range(n):
                w[i, (i + 1) % n] = 1
            shared[key] = parameter(w)
        return node("dense", [x], w=shared[key])

    def rotate_spatial(x, n):
        key = ("rots", n)
        if key not in shared:
            w = np.zeros((n, n, 1, 1), np.float32)
            for i in range(n):
                w[i, (i + 1) % n, 0, 0] = 1
            shared[key] = parameter(w)
        return node("conv", [x], w=shared[key])

    def rank_value(x, n):
        # At integer ranks 0..9: 0,1,2,4,8,...,256.
        w = np.zeros((7 * n, n), np.float32)
        b = np.zeros(7 * n, np.float32)
        out = np.zeros((n, 7 * n), np.float32)
        for i in range(7):
            for j in range(n):
                w[i * n + j, j] = 1
                b[i * n + j] = -(i + 2)
                out[j, i * n + j] = 2 ** i
        return vadd(x, dense(vr(dense(x, w, b)), out))

    # Pixel coordinates and a spatial constant.
    wxy = np.zeros((3, 3, 1, 1), np.float32)
    wxy[0, 1, 0, 0] = 63
    wxy[1, 0, 0, 0] = 63
    wxy[2, 2, 0, 0] = 1
    xy1 = conv("coords", wxy)
    X = sl(xy1, 0, 1)
    Y = sl(xy1, 1, 2)
    one = sl(xy1, 2, 3)

    def broadcast(v):
        return mul(one, v)

    index = add(X, scale(Y, 64))
    play = clip(scale(bias(Y, -9.5), 2))
    header = mul(complement(play), clip(scale(bias(X, -25), 2)))
    legal = mul(play, clip(scale(sub(c(62.5), Y), 2)))

    colors = [10, 6, 15, 11, 12, 8, 9, 7, 14, 13]
    sizes = [1, 2, 3, 4, 5, 7, 8, 9, 10]
    color_index = {v: i for i, v in enumerate(colors)}

    # Sparse shifted banks extend the detector's receptive field.
    shifts = [(-3, -3), (0, 0), (3, 3)]
    wb = np.zeros((30, 16, 7, 7), np.float32)
    for k, (dy, dx) in enumerate(shifts):
        for q, color in enumerate(colors):
            wb[k * 10 + q, color, dy + 3, dx + 3] = 1
    bank = conv("x0", wb)

    wd = np.zeros((13, 30, 7, 7), np.float32)
    bd = np.zeros(13, np.float32)

    def tap(out, color, dy, dx, value):
        for k, (sy, sx) in enumerate(shifts):
            yy, xx = dy - sy, dx - sx
            if -3 <= yy <= 3 and -3 <= xx <= 3:
                wd[out, k * 10 + color_index[color],
                   yy + 3, xx + 3] += value
                return
        raise ValueError("Detector offset outside shifted banks")

    # Channels 0..8 identify fruit centers and ranks.
    for r, size in enumerate(sizes):
        h = size // 2
        t = size - 1 - h
        color = colors[r]
        tap(r, color, -h, -h, 1)
        tap(r, color, t, t, 1)
        tap(r, color, 0, 0, 1)
        tap(r, color, -h - 1, -h, -3)
        tap(r, color, -h, -h - 1, -3)
        bd[r] = -2

    patterns = [
        [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ],
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
    ]
    # Ghost templates tolerate up to two missing foreground pixels.
    for k, pat in enumerate(patterns):
        count = 0
        color = [7, 14, 13][k]
        for yy in range(4):
            for xx in range(5):
                value = pat[yy][xx]
                tap(9 + k, color, yy - 2, xx - 2,
                    1 if value else -1)
                count += value
        bd[9 + k] = 3 - count

    # Goal detector ignores the goal interior.
    for dy, dx in [(-4, 0), (4, 0), (0, -4), (0, 4)]:
        tap(12, 9, dy, dx, 1)
    tap(12, 9, -4, -4, -4)
    bd[12] = -3
    det = clip(conv(bank, wd, bd))

    # Decode requested types and counts from the header.
    ui = mul(sl(det, 0, 12), header)
    ui_count = vs(node("gmean", [ui]), 4096)
    fw = np.zeros((1, 12, 1, 1), np.float32)
    gw = np.zeros_like(fw)
    fw[0, :9, 0, 0] = np.arange(1, 10)
    gw[0, 9:, 0, 0] = np.arange(1, 4)
    F = node("gmax", [conv(ui, fw)])
    G = node("gmax", [conv(ui, gw)])
    rf = np.zeros((1, 12), np.float32)
    rg = np.zeros_like(rf)
    rf[0, :9] = 1
    rg[0, 9:] = 1
    Fcount = dense(ui_count, rf)
    Gcount = dense(ui_count, rg)
    Gexists = vc(G)

    objects = mul(sl(det, 0, 12), play)
    presence = clip(sum_channels(objects, 12))
    ghosts_map = sum_channels(sl(objects, 9, 12), 3)

    # Inventory test uses every detection, not only the object slots.
    inventory = vs(node("gmean", [objects]), 4096)
    fruit_inventory = vl(inventory, 0, 9)
    ghost_inventory = vl(inventory, 9, 12)
    ranks = constant(np.arange(1, 10, dtype=np.float32))
    capped_rank = vsub(ranks, vr(vsub(ranks, F)))
    available_mass = sum_vector(
        vmul(rank_value(capped_rank, 9), fruit_inventory), 9
    )
    required_mass = vmul(rank_value(F, 1), Fcount)
    fruit_deficit = vc(vs(vb(vsub(required_mass, available_mass), -0.5), 2))

    ghost_tiers = constant([1, 2, 3])
    usable_ghosts = vc(vb(vsub(G, ghost_tiers), 1))
    available_ghost_mass = sum_vector(
        vmul(vmul(ghost_inventory, usable_ghosts), constant([1, 2, 4])), 3
    )
    required_ghost_mass = vmul(Gcount, vadd(G, vr(vb(G, -2))))
    ghost_deficit = vc(vs(
        vb(vsub(required_ghost_mass, available_ghost_mass), -0.5), 2
    ))
    red_header = node("gmax", [mul(sl("x0", 2, 3), header)])
    need_undo = vc(vadd(vadd(fruit_deficit, ghost_deficit), red_header))

    # Metadata: fruit rank, ghost tier, rectangle radii, center offsets.
    wm = np.zeros((6, 12, 1, 1), np.float32)
    for r, size in enumerate(sizes):
        wm[0, r, 0, 0] = r + 1
        wm[2, r, 0, 0] = (size - 1) / 2
        wm[3, r, 0, 0] = (size - 1) / 2
        wm[4, r, 0, 0] = size // 2 - (size - 1) / 2
        wm[5, r, 0, 0] = size // 2 - (size - 1) / 2
    for k in range(3):
        wm[1, 9 + k, 0, 0] = k + 1
        wm[2, 9 + k, 0, 0] = 2
        wm[3, 9 + k, 0, 0] = 1.5
        wm[5, 9 + k, 0, 0] = 0.5
    metadata = conv(objects, wm)
    features = cat(X, Y, metadata, presence)
    extraction_score = add(
        mul(presence, sub(c(20000), index)),
        scale(ghosts_map, 30000)
    )

    def extract(prefix, score, fmap, k, n):
        nonlocal target
        outer = target
        body = []
        target = body
        rs = prefix + "_remaining"
        maximum_score = node("gmax", [rs])
        at_maximum = clip(bias(sub(rs, maximum_score), 1))
        chosen = mul(at_maximum, clip(rs))
        remaining = sub(rs, mul(rs, chosen))
        values = vs(node("gmean", [mul(fmap, chosen)]), 4096)

        shift = np.zeros((n, n), np.float32)
        for i in range(n - 1):
            shift[i, i + 1] = 1
        sw = parameter(shift)
        states = [{"id": rs, "init": score}]
        nxt = {rs: remaining}
        names = []
        for j in range(k):
            sid = prefix + "_s%d" % j
            states.append({"id": sid, "zeros": [n]})
            shifted = node("dense", [sid], w=sw)
            w = np.zeros((n, k), np.float32)
            w[-1, j] = 1
            inserted = dense(values, w)
            nxt[sid] = vadd(shifted, inserted)
            names.append(sid)

        target = outer
        lid = node("loop", name=prefix, T=n, state=states,
                   body=body, next=nxt)
        return [lid + "." + z for z in names]

    extracted = extract("objects_extract", extraction_score, features, 9, N)
    ox, oy, fr, gr, rx, ry, exx, exy, present_v = extracted
    fruit = vc(fr)
    ghost = vc(gr)
    present_v = vc(present_v)
    low_f = vmul(vc(vsub(F, fr)), fruit)
    good_f = vmul(veq(fr, F), fruit)
    big_f = vmul(vc(vsub(fr, F)), fruit)
    low_g = vmul(vc(vsub(G, gr)), ghost)
    good_g = vmul(veq(gr, G), ghost)
    tier1 = vmul(veq(gr, c(1)), ghost)
    tier3 = vmul(veq(gr, c(3)), ghost)

    geom_x = vsub(ox, exx)
    geom_y = vsub(oy, exy)
    left_x = vadd(rx, exx)
    right_x = vsub(rx, exx)
    left_y = vb(vadd(ry, exy), 10)
    right_y = vsub(ry, exy)
    max_x = vsub(c(63), right_x)
    max_y = vsub(c(63), right_y)

    # Goal extraction and memory for temporary occlusions.
    goal_mask = mul(sl(det, 12, 13), play)
    goal_features = cat(X, Y, goal_mask)
    goal_score = mul(goal_mask, sub(c(20000), index))
    gx0, gy0, gp0 = extract("goals_extract", goal_score, goal_features, 3, B)
    old_gx = vl("mem", 0, B)
    old_gy = vl("mem", B, 2 * B)
    old_gp = vl("mem", 2 * B, 3 * B)
    fresh_count = sum_vector(gp0, B)
    saved_count = sum_vector(old_gp, B)
    refresh = vc(vb(vsub(fresh_count, saved_count), 1))
    keep = vcomp(refresh)
    gx = vadd(vmul(gx0, refresh), vmul(old_gx, keep))
    gy = vadd(vmul(gy0, refresh), vmul(old_gy, keep))
    gp = vadd(vmul(gp0, refresh), vmul(old_gp, keep))

    # Candidate clicks are evaluated at every spatial cell.
    ddx = relu(sub(ab(sub(X, geom_x)), rx))
    ddy = relu(sub(ab(sub(Y, geom_y)), ry))
    d2 = add(mul(ddx, ddx), mul(ddy, ddy))
    selected = mul(clip(sub(c(65), d2)), present_v)
    not_selected = complement(selected)

    snap_x = minimum(maximum(X, left_x), max_x)
    snap_y = minimum(maximum(Y, left_y), max_y)
    snap_dx = sub(snap_x, ox)
    snap_dy = sub(snap_y, oy)
    fruit_x = add(mul(selected, snap_dx), ox)
    fruit_y = add(mul(selected, snap_dy), oy)

    # Tier-one ghosts move along their initial direction and overshoot.
    raw_dx = sub(X, ox)
    raw_dy = sub(Y, oy)
    abs_pair = cat(ab(raw_dx), ab(raw_dy))
    wt = np.zeros((6 * N, 2 * N, 1, 1), np.float32)
    ratios = [
        math.sqrt(0.25 / 11.31),
        math.sqrt(2.25 / 9.31),
        math.sqrt(6.25 / 5.31),
    ]
    for j in range(N):
        for k, ratio in enumerate(ratios):
            wt[k * N + j, j, 0, 0] = 128
            wt[k * N + j, N + j, 0, 0] = -128 * ratio
            wt[(3 + k) * N + j, N + j, 0, 0] = 128
            wt[(3 + k) * N + j, j, 0, 0] = -128 * ratio
    thresholds = clip(conv(abs_pair, wt))
    ws = np.zeros((2 * N, 6 * N, 1, 1), np.float32)
    for j in range(N):
        for k in range(3):
            ws[j, k * N + j, 0, 0] = 4
            ws[N + j, (3 + k) * N + j, 0, 0] = 4
    magnitudes = conv(thresholds, ws)
    sign_x = minimum(maximum(raw_dx, c(-1)), c(1))
    sign_y = minimum(maximum(raw_dy, c(-1)), c(1))
    slide_dx = mul(sl(magnitudes, 0, N), sign_x)
    slide_dy = mul(sl(magnitudes, N, 2 * N), sign_y)
    ghost_selected_dx = add(snap_dx, mul(sub(slide_dx, snap_dx), tier1))
    ghost_selected_dy = add(snap_dy, mul(sub(slide_dy, snap_dy), tier1))

    # Project chasing ghosts toward the nearest fruit.
    body = []
    target = body
    cx = sl("chase_px", 0, 1)
    cy = sl("chase_py", 0, 1)
    cf = vc(vl("chase_f", 0, 1))
    dx = sub(cx, ox)
    dy = sub(cy, oy)
    dist = add(
        add(mul(dx, dx), mul(dy, dy)),
        scale(broadcast(vcomp(cf)), 100000)
    )
    improve = clip(scale(sub("chase_best", dist), 100))
    tx = add("chase_tx", mul(improve, sub(cx, "chase_tx")))
    ty = add("chase_ty", mul(improve, sub(cy, "chase_ty")))
    best = minimum("chase_best", dist)
    chase_next = {
        "chase_px": rotate_spatial("chase_px", N),
        "chase_py": rotate_spatial("chase_py", N),
        "chase_f": rotate_vector("chase_f", N),
        "chase_best": best,
        "chase_tx": tx,
        "chase_ty": ty,
    }
    target = graph
    huge = broadcast(constant(np.full(N, 1000000, np.float32)))
    chase_init_x = broadcast(ox)
    chase_init_y = broadcast(oy)
    node(
        "loop", name="chase", T=N,
        state=[
            {"id": "chase_px", "init": fruit_x},
            {"id": "chase_py", "init": fruit_y},
            {"id": "chase_f", "init": fr},
            {"id": "chase_best", "init": huge},
            {"id": "chase_tx", "init": chase_init_x},
            {"id": "chase_ty", "init": chase_init_y},
        ],
        body=body, next=chase_next
    )

    speed = vb(vs(tier3, 4), 4)
    neg_speed = vs(speed, -1)
    chase_dx = minimum(maximum(sub("chase.chase_tx", ox), neg_speed), speed)
    chase_dy = minimum(maximum(sub("chase.chase_ty", oy), neg_speed), speed)
    ghost_x = add(
        ox, add(mul(selected, ghost_selected_dx),
                mul(not_selected, chase_dx))
    )
    ghost_y = add(
        oy, add(mul(selected, ghost_selected_dy),
                mul(not_selected, chase_dy))
    )
    px = add(fruit_x, mul(sub(ghost_x, fruit_x), ghost))
    py = add(fruit_y, mul(sub(ghost_y, fruit_y), ghost))
    px = minimum(maximum(px, left_x), max_x)
    py = minimum(maximum(py, left_y), max_y)
    pgx = sub(px, exx)
    pgy = sub(py, exy)

    # Goal distances and occupancy-aware parking.
    body = []
    target = body
    qx = vl("goal_x", 0, 1)
    qy = vl("goal_y", 0, 1)
    qvalid = vl("goal_p", 0, 1)
    ax0 = ab(sub(ox, broadcast(qx)))
    ay0 = ab(sub(oy, broadcast(qy)))
    ax1 = ab(sub(px, qx))
    ay1 = ab(sub(py, qy))
    old_cost = add(relu(bias(ax0, -3)), relu(bias(ay0, -3)))
    new_cost = add(relu(bias(ax1, -3)), relu(bias(ay1, -3)))
    inside0 = mul(
        clip(sub(c(1), add(relu(bias(ax0, -4)), relu(bias(ay0, -4))))),
        qvalid
    )
    inside1 = mul(
        clip(sub(c(1), add(relu(bias(ax1, -4)), relu(bias(ay1, -4))))),
        qvalid
    )
    occupied_self = mul(inside0, good_f)
    occupied = sum_channels(occupied_self, N)
    other_occupied = relu(sub(occupied, occupied_self))
    occupancy_penalty = add(
        scale(mul(other_occupied, fruit), 20),
        scale(mul(occupied, ghost), 12)
    )
    invalid = scale(broadcast(vcomp(qvalid)), 1000)
    old_cost = add(add(old_cost, occupancy_penalty), invalid)
    new_cost = add(add(new_cost, occupancy_penalty), invalid)
    goal_next = {
        "goal_x": rotate_vector("goal_x", B),
        "goal_y": rotate_vector("goal_y", B),
        "goal_p": rotate_vector("goal_p", B),
        "goal_old": minimum("goal_old", old_cost),
        "goal_new": minimum("goal_new", new_cost),
        "goal_in_old": maximum("goal_in_old", inside0),
        "goal_in_new": maximum("goal_in_new", inside1),
    }
    target = graph
    far = broadcast(constant(np.full(N, 200, np.float32)))
    node(
        "loop", name="goals", T=B,
        state=[
            {"id": "goal_x", "init": gx},
            {"id": "goal_y", "init": gy},
            {"id": "goal_p", "init": gp},
            {"id": "goal_old", "init": far},
            {"id": "goal_new", "init": far},
            {"id": "goal_in_old", "zeros": [N, 64, 64]},
            {"id": "goal_in_new", "zeros": [N, 64, 64]},
        ],
        body=body, next=goal_next
    )

    goal_old = "goals.goal_old"
    goal_new = "goals.goal_new"
    in_old = "goals.goal_in_old"
    in_new = "goals.goal_in_new"
    parked = sum_vector(vmul(node("gmax", [in_old]), good_f), N)
    fruit_done = vc(vb(vsub(parked, Fcount), 1))
    ghost_goal_weight = vb(vs(fruit_done, 1.7), 0.3)
    ids = constant(np.arange(N, dtype=np.float32))

    # Pairwise progress, merging, and swept collision estimates.
    body = []
    target = body
    sx = vl("pair_x", 0, 1)
    sy = vl("pair_y", 0, 1)
    sf = vl("pair_f", 0, 1)
    sg = vl("pair_g", 0, 1)
    srx = vl("pair_rx", 0, 1)
    sry = vl("pair_ry", 0, 1)
    si = vl("pair_id", 0, 1)
    sx1 = sl("pair_px", 0, 1)
    sy1 = sl("pair_py", 0, 1)
    ssel = sl("pair_sel", 0, 1)
    sgcost = sl("pair_gc", 0, 1)
    not_self = vc(va(vsub(ids, si)))
    scan_fruit = vc(sf)
    scan_ghost = vc(sg)
    same_f = vmul(vmul(veq(fr, sf), fruit), scan_fruit)
    same_g = vmul(vmul(veq(gr, sg), ghost), scan_ghost)
    matched = vmul(vadd(same_f, same_g), not_self)

    dx0 = sub(broadcast(geom_x), sx)
    dy0 = sub(broadcast(geom_y), sy)
    dx1 = sub(pgx, sx1)
    dy1 = sub(pgy, sy1)
    distance0 = add(ab(dx0), ab(dy0))
    distance1 = add(ab(dx1), ab(dy1))
    dxmid = scale(add(dx0, dx1), 0.5)
    dymid = scale(add(dy0, dy1), 0.5)
    distance_mid = add(ab(dxmid), ab(dymid))
    ghost_pair_distance = minimum(distance1, distance_mid)
    pair_distance1 = add(
        distance1, mul(sub(ghost_pair_distance, distance1), same_g)
    )
    invalid_match = scale(broadcast(vcomp(matched)), 1000)
    near0 = minimum("pair_near0", add(distance0, invalid_match))
    near1 = minimum("pair_near1", add(pair_distance1, invalid_match))
    invalid_ghost = scale(broadcast(vcomp(scan_ghost)), 1000)
    ng0 = minimum("pair_ghost0", add(distance0, invalid_ghost))
    ng1 = minimum("pair_ghost1", add(distance1, invalid_ghost))

    radius_x = vadd(rx, srx)
    radius_y = vadd(ry, sry)
    gap_end = add(
        relu(sub(ab(dx1), radius_x)), relu(sub(ab(dy1), radius_y))
    )
    gap_mid = add(
        relu(sub(ab(dxmid), radius_x)), relu(sub(ab(dymid), radius_y))
    )
    collision = clip(sub(c(1), minimum(gap_end, gap_mid)))
    attacking = mul(mul(collision, complement(ssel)), scan_ghost)
    hit = maximum("pair_hit", mul(attacking, fruit))

    ghost_pair = vmul(vmul(ghost, scan_ghost), not_self)
    equal_tier = veq(gr, sg)
    useful_merge = vmul(equal_tier, low_g)
    mixed = vcomp(equal_tier)
    excessive = vmul(equal_tier, vcomp(low_g))
    excessive_price = vb(vs(Gexists, -300), -10)
    pair_value = vadd(
        vadd(vs(useful_merge, 70), vs(mixed, -1000)),
        vmul(excessive, excessive_price)
    )
    pair_value = vmul(pair_value, ghost_pair)
    ghost_reward = add("pair_reward", mul(collision, pair_value))

    tie = vs(vsub(ids, si), 0.001)
    better = clip(scale(add(sub(goal_old, sgcost), tie), 1000))
    better_count = add("pair_better", mul(better, matched))
    next_pair = {
        "pair_x": rotate_vector("pair_x", N),
        "pair_y": rotate_vector("pair_y", N),
        "pair_f": rotate_vector("pair_f", N),
        "pair_g": rotate_vector("pair_g", N),
        "pair_rx": rotate_vector("pair_rx", N),
        "pair_ry": rotate_vector("pair_ry", N),
        "pair_id": rotate_vector("pair_id", N),
        "pair_px": rotate_spatial("pair_px", N),
        "pair_py": rotate_spatial("pair_py", N),
        "pair_sel": rotate_spatial("pair_sel", N),
        "pair_gc": rotate_spatial("pair_gc", N),
        "pair_near0": near0,
        "pair_near1": near1,
        "pair_ghost0": ng0,
        "pair_ghost1": ng1,
        "pair_hit": hit,
        "pair_reward": ghost_reward,
        "pair_better": better_count,
    }
    target = graph
    pair_states = []
    for sid, initial in [
        ("pair_x", geom_x), ("pair_y", geom_y),
        ("pair_f", fr), ("pair_g", gr),
        ("pair_rx", rx), ("pair_ry", ry), ("pair_id", ids),
        ("pair_px", pgx), ("pair_py", pgy),
        ("pair_sel", selected), ("pair_gc", goal_old),
        ("pair_near0", far), ("pair_near1", far),
        ("pair_ghost0", far), ("pair_ghost1", far),
    ]:
        pair_states.append({"id": sid, "init": initial})
    for sid in ["pair_hit", "pair_reward", "pair_better"]:
        pair_states.append({"id": sid, "zeros": [N, 64, 64]})
    node("loop", name="pairs", T=N, state=pair_states,
         body=body, next=next_pair)

    needed = vadd(vmul(fruit, Fcount), vmul(ghost, Gcount))
    active = clip(sub(needed, "pairs.pair_better"))
    good_weight = vadd(good_f, vmul(good_g, ghost_goal_weight))
    active_good = mul(active, good_weight)
    pair_progress = sub("pairs.pair_near0", "pairs.pair_near1")
    pairing_weight = vadd(vs(low_f, 0.6), vs(low_g, 0.25))
    object_score = mul(pair_progress, pairing_weight)
    object_score = add(
        object_score,
        scale(mul(sub("pairs.pair_ghost0", "pairs.pair_ghost1"), big_f), 0.35)
    )
    object_score = add(
        object_score, mul(sub(goal_old, goal_new), active_good)
    )
    goal_progress = sub(in_new, in_old)
    object_score = add(
        object_score, scale(mul(goal_progress, active_good), 30)
    )
    unwanted_good = mul(complement(active), vadd(good_f, good_g))
    object_score = sub(
        object_score, scale(mul(goal_progress, unwanted_good), 50)
    )
    hit_value = vadd(vs(big_f, 100), vs(vsub(fruit, big_f), -500))
    object_score = add(object_score, mul("pairs.pair_hit", hit_value))
    object_score = add(object_score, scale("pairs.pair_reward", 0.5))
    score = sum_channels(object_score, N)

    # Prevent mixed-rank, overlarge, and target-destroying fruit merges.
    selected_fruit = mul(selected, fruit)
    count_selected = sum_channels(selected_fruit, N)
    rank_sum = sum_channels(mul(selected_fruit, fr), N)
    rank_square_sum = sum_channels(mul(selected_fruit, vmul(fr, fr)), N)
    variance = relu(sub(
        mul(count_selected, rank_square_sum), mul(rank_sum, rank_sum)
    ))
    mixed_penalty = scale(clip(variance), 1000)
    crowded_penalty = scale(relu(bias(count_selected, -2)), 1000)
    multi = clip(bias(count_selected, -1))
    high_selected = clip(sum_channels(mul(selected, vadd(good_f, big_f)), N))
    bad_merge = scale(mul(multi, high_selected), 1000)
    merge_bonus = scale(mul(multi, complement(high_selected)), 60)
    score = add(score, merge_bonus)
    score = sub(score, add(add(mixed_penalty, crowded_penalty), bad_merge))
    score = sub(score, scale(count_selected, 0.02))

    # State checksum and persistent failed-click memory.
    hash_input = vcat(ox, oy, fr, gr)
    hw = np.zeros((1, 4 * N), np.float32)
    hw[0, :N] = 1
    hw[0, N:2 * N] = 65
    hw[0, 2 * N:3 * N] = 4096
    hw[0, 3 * N:] = 55000
    state_hash = dense(hash_input, hw)

    base = 3 * B
    prev_x = vl("mem", base, base + 1)
    prev_y = vl("mem", base + 1, base + 2)
    prev_hash = vl("mem", base + 2, base + 3)
    old_bad_x = vl("mem", base + 3, base + 4)
    old_bad_y = vl("mem", base + 4, base + 5)
    old_bad_hash = vl("mem", base + 5, base + 6)
    old_bad_valid = vl("mem", base + 6, base + 7)
    old_failures = vl("mem", base + 7, base + 8)
    old_undo_streak = vl("mem", base + 8, base + 9)

    was_reset = vl("aprev", 0, 1)
    was_click = vl("aprev", 6, 7)
    was_undo = vl("aprev", 7, 8)
    not_reset = vcomp(was_reset)
    unchanged = vmul(veq(state_hash, prev_hash), was_click)
    new_failure = vmul(was_click, vc(vadd(need_undo, unchanged)))
    retain_failure = vcomp(new_failure)

    bad_x = vadd(vmul(new_failure, prev_x),
                 vmul(retain_failure, old_bad_x))
    bad_y = vadd(vmul(new_failure, prev_y),
                 vmul(retain_failure, old_bad_y))
    bad_hash = vadd(vmul(new_failure, prev_hash),
                    vmul(retain_failure, old_bad_hash))
    bad_valid = vmul(vc(vadd(old_bad_valid, new_failure)), not_reset)
    failures = vmul(vadd(old_failures, new_failure), not_reset)
    undo_streak = vmul(was_undo, vb(old_undo_streak, 1))
    apply_taboo = vmul(bad_valid, veq(state_hash, bad_hash))
    bad_distance = add(ab(sub(X, bad_x)), ab(sub(Y, bad_y)))
    taboo_region = clip(sub(c(1), scale(bad_distance, 0.25)))
    score = sub(score, scale(mul(taboo_region, apply_taboo), 1000))
    score = sub(score, scale(complement(legal), 100000))

    # Stable secondary ordering also supplies the remembered click.
    peak = node("gmax", [score])
    eligible = clip(sub(c(1), scale(sub(peak, score), 10000)))
    lex = sub(scale(eligible, 10000), index)
    click = node("add", [lex, c(0)], name="click")
    click_peak = node("gmax", [click])
    winner = clip(sub(c(1), scale(sub(click_peak, click), 1000)))
    click_x = vs(node("gmean", [mul(winner, X)]), 4096)
    click_y = vs(node("gmean", [mul(winner, Y)]), 4096)

    reset_needed = vc(vadd(
        vr(vb(failures, -5)),
        vmul(need_undo, vr(vb(undo_streak, -4)))
    ))
    control = vcat(need_undo, reset_needed)
    wl = np.zeros((8, 2), np.float32)
    bl = np.full(8, -100, np.float32)
    bl[6] = 5
    bl[7] = 0
    wl[7, 0] = 100
    wl[0, 1] = 300
    dense(control, wl, bl, name="logits")
    node(
        "vconcat",
        [gx, gy, gp, click_x, click_y, state_hash,
         bad_x, bad_y, bad_hash, bad_valid, failures, undo_streak],
        name="mem_next"
    )

    # Remove unreachable graph construction intermediates and parameters.
    def prune(nodes, required):
        needed = set(required)
        kept = []
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                prefix = nd["id"] + "."
                if not any(s.startswith(prefix) for s in needed):
                    continue
                body, external = prune(nd["body"], nd["next"].values())
                d = dict(nd)
                d["body"] = body
                kept.append(d)
                state_names = {s["id"] for s in nd["state"]}
                needed.update(external - state_names)
                for state in nd["state"]:
                    if "init" in state:
                        needed.add(state["init"])
            elif nd["id"] in needed:
                kept.append(nd)
                needed.update(nd.get("in", []))
        kept.reverse()
        local_ids = {nd["id"] for nd in nodes}
        external = {
            s for s in needed
            if s not in local_ids and s.split(".", 1)[0] not in local_ids
        }
        return kept, external

    graph, _ = prune(graph, ["logits", "click", "mem_next"])
    used_params = set()

    def collect(nodes):
        for nd in nodes:
            for field in ("w", "b"):
                if field in nd:
                    used_params.add(nd[field])
            if nd["op"] == "loop":
                collect(nd["body"])

    collect(graph)
    params = {k: v for k, v in params.items() if k in used_params}
    return {"graph": graph, "memory_size": M, "params": params}
