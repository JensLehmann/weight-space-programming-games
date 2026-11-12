import numpy as np


def build():
    P, G = {}, []
    target = G
    serial = 0
    parameter_cache = {}
    node_caches = {}

    def par(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key in parameter_cache:
            return parameter_cache[key]
        name = "p%d" % len(P)
        P[name] = a
        parameter_cache[key] = name
        return name

    def nd(kind, xs=None, name=None, **kw):
        nonlocal serial
        cache = node_caches.setdefault(id(target), {})
        signature = None
        if name is None and kind != "loop":
            signature = (kind, tuple(xs or ()), tuple(sorted(kw.items())))
            if signature in cache:
                return cache[signature]
        serial += 1
        ident = name or ("n%d" % serial)
        d = {"id": ident, "op": kind}
        if xs is not None:
            d["in"] = list(xs)
        d.update(kw)
        target.append(d)
        if signature is not None:
            cache[signature] = ident
        return ident

    def cv(x, w, b=None, name=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return nd("conv", [x], name=name, **fields)

    def lin(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        return cv(x, w[:, :, None, None], b, name)

    def dn(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return nd("dense", [x], name=name, **fields)

    def op(kind, *xs):
        return nd(kind, xs)

    def cat(*xs):
        return op("concat", *xs)

    def vc(*xs):
        return op("vconcat", *xs)

    def ch(x, i, j=None):
        return nd("slice", [x], start=i, stop=i + 1 if j is None else j)

    def gm(x):
        return op("gmax", x)

    colours = [ch("x0", i) for i in range(16)]
    one = cv("coords", np.zeros((1, 3, 1, 1)), [1])

    def inv(x):
        return op("sub", one, x)

    def vi(x):
        return dn(x, [-1], [1])

    def cap(x):
        return op("min2", x, one)

    def binary(x, bias=0.0):
        return cap(op("relu", lin(x, [1000], [1000 * bias])))

    def vt(x, w, bias=0.0, gain=1000):
        return op("vsigmoid",
                  dn(x, np.asarray(w, np.float32) * gain, [bias * gain]))

    def repeat(x, n):
        return lin(x, np.ones((n, 1), np.float32))

    # Rotation from the edge carrying the progress strip.
    ew = np.zeros((4, 3, 1, 1), np.float32)
    ew[0, 0, 0, 0] = -63
    ew[1, 1, 0, 0] = -63
    ew[2, 0, 0, 0] = 63
    ew[3, 1, 0, 0] = 63
    edge = op("relu", cv("coords", ew, [1, 1, -62, -62]))
    strip = op("add", colours[0], colours[14])
    rotation = op(
        "vsigmoid",
        dn(op("gmean", op("mul", edge, strip)),
           128000 * np.eye(4), [-500] * 4))

    wr = lin("coords", [[1, 0, 0], [0, 1, 0],
                       [-1, 0, 1], [0, -1, 1]])
    wc = lin("coords", [[0, 1, 0], [-1, 0, 1],
                       [0, -1, 1], [1, 0, 0]])
    r = lin(op("mul", wr, rotation), [1, 1, 1, 1])
    c = lin(op("mul", wc, rotation), [1, 1, 1, 1])
    vertical = dn(rotation, [1, 0, 1, 0])

    # Direction channels are physical down, up, right, left.
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    p1, p2 = [3, 3, 1, 1], [2, 2, 0, 0]

    def direction(indices):
        w = np.zeros((4, 4), np.float32)
        for k, i in enumerate(indices):
            w[i, k] = 1
        return dn(rotation, w)

    down = direction([0, 2, 1, 3])
    right = direction([2, 1, 3, 0])
    left = direction([3, 0, 2, 1])
    sideways = op("vadd", right, left)

    def choose(x, d):
        return lin(op("mul", x, d), [1, 1, 1, 1])

    raw_selected = colours[9]
    raw_source = colours[6]
    sx = gm(op("rowsum", raw_source))
    sy = gm(op("colsum", raw_source))
    unit = op("vadd", op("vmul", sy, vertical),
              op("vmul", sx, vi(vertical)))
    s2 = vt(unit, [-1], 2.5)
    s4 = vt(unit, [1], -3.5)
    s3 = dn(vc(s2, s4), [-1, -1], [1])
    scales = vc(s2, s3, s4)

    bg = colours[12]
    bg_top = dn(gm(op("mul", bg, inv(r))), [-1], [1])
    bg_left = dn(gm(op("mul", bg, inv(c))), [-1], [1])
    zero_pad = vt(bg_top, [-63], 1.5)
    pad_r = op("vmul", bg_top, vi(zero_pad))
    pad_c = bg_left

    # Piecewise-linear periodic combs mark logical-cell centres.
    def lattice_axis(axis, pad):
        relative = lin(op("sub", axis, pad), [63])
        knots = np.arange(-1, 65, dtype=np.float32)
        ramps = op("relu", cv(
            relative, np.ones((66, 1, 1, 1)), -knots))
        weights = np.zeros((3, 66), np.float32)
        for a, scale in enumerate((2, 3, 4)):
            def value(t):
                return float(t >= 0 and t % scale == scale // 2)
            for j, t in enumerate(range(-1, 65)):
                weights[a, j] = (
                    value(t + 1) - 2 * value(t) + value(t - 1))
        combs = lin(ramps, weights)
        mixed = lin(op("mul", combs, scales), [1, 1, 1])
        return binary(mixed, -.5)

    lattice = op("mul", lattice_axis(r, pad_r),
                 lattice_axis(c, pad_c))
    row_low = binary(op("sub", r, pad_r), .00001)
    col_low = binary(op("sub", c, pad_c), .00001)
    row_high = binary(op("sub", dn(pad_r, [-1], [1]), r), .00001)
    col_high = binary(op("sub", dn(pad_c, [-1], [1]), c), .00001)
    inside_r = op("mul", row_low, row_high)
    inside_c = op("mul", col_low, col_high)
    inside = op("mul", inside_r, inside_c)
    cells = op("mul", lattice, inside)

    first_r = dn(gm(op("mul", cells, inv(r))), [-1], [1])
    first_c = dn(gm(op("mul", cells, inv(c))), [-1], [1])
    last_r = gm(op("mul", cells, r))
    last_c = gm(op("mul", cells, c))

    selected = op("mul", raw_selected, cells)
    blue = op("mul", colours[8], cells)
    elbow = op("mul", colours[15], cells)
    cups = op("mul", op("add", colours[11], colours[13]), cells)
    walls = op("mul", colours[1], cells)
    source = op("mul", raw_source, cells)
    emitter = op("mul", colours[4], cells)

    row_extent = gm(op("rowsum", raw_selected))
    col_extent = gm(op("colsum", raw_selected))
    extents = vc(row_extent, col_extent)
    ho = dn(extents, [[0, 1], [1, 0], [0, 1], [1, 0]])
    wo = dn(extents, [[1, 0], [0, 1], [1, 0], [0, 1]])
    height = dn(op("vmul", ho, rotation), [1, 1, 1, 1])
    width = dn(op("vmul", wo, rotation), [1, 1, 1, 1])
    is_elbow = vt(vc(height, width), [1, -.6], -.5)
    is_bar = vi(is_elbow)
    current_r = gm(op("mul", selected, r))
    current_c = dn(gm(op("mul", selected, inv(c))), [-1], [1])
    selected_exists = gm(selected)

    # Four repeated one-pixel shifts yield scale-2/3/4 neighbours.
    # This is substantially cheaper than large dense kernels.
    shift_params = {}

    def logical_shift(x, deltas):
        key = tuple(deltas)
        if key not in shift_params:
            w = np.zeros((4, 4, 3, 3), np.float32)
            for j, (dy, dx) in enumerate(deltas):
                w[j, j, 1 + dy, 1 + dx] = 1
            shift_params[key] = par(w)
        kernel = shift_params[key]
        z1 = nd("conv", [x], w=kernel)
        z2 = nd("conv", [z1], w=kernel)
        z3 = nd("conv", [z2], w=kernel)
        z4 = nd("conv", [z3], w=kernel)
        return op(
            "add",
            op("add", op("mul", z2, s2), op("mul", z3, s3)),
            op("mul", z4, s4))

    def ahead(x):
        return logical_shift(x, dirs)

    def backwards(x):
        return logical_shift(x, [(-dy, -dx) for dy, dx in dirs])

    def perp(x, indices):
        return logical_shift(x, [dirs[i] for i in indices])

    def remap(x, indices, emit=False):
        w = np.zeros((4, 4), np.float32)
        for i, j in enumerate(indices):
            if emit:
                w[j, i] += 1
            else:
                w[i, j] = 1
        return lin(x, w)

    def neighbours(x):
        return ahead(repeat(x, 4))

    world_params = {}

    def world_logical(x, dy, dx):
        key = (dy, dx)
        if key not in world_params:
            ds = [(dy, dx), (-dx, dy), (-dy, -dx), (dx, -dy)]
            initial = np.zeros((4, 1, 3, 3), np.float32)
            carry = np.zeros((4, 4, 3, 3), np.float32)
            for j, (yy, xx) in enumerate(ds):
                initial[j, 0, 1 + yy, 1 + xx] = 1
                carry[j, j, 1 + yy, 1 + xx] = 1
            world_params[key] = (par(initial), par(carry))
        initial, carry = world_params[key]
        z1 = nd("conv", [x], w=initial)
        z2 = nd("conv", [z1], w=carry)
        z3 = nd("conv", [z2], w=carry)
        z4 = nd("conv", [z3], w=carry)
        z = op(
            "add",
            op("add", op("mul", z2, s2), op("mul", z3, s3)),
            op("mul", z4, s4))
        return choose(z, rotation)

    def geometry(bars, elbows):
        solid = op("add", cups, op("add", bars, elbows))
        free = op(
            "sub", cells,
            cap(op("add", op("add", solid, walls), emitter)))
        cn = neighbours(cups)
        mouth = op(
            "mul", free,
            op("mul", cn,
               op("min2", remap(cn, p1), remap(cn, p2))))
        en = neighbours(elbows)
        e1, e2 = remap(en, p1), remap(en, p2)
        t1 = op("mul", op("mul", en, e1), inv(e2))
        t2 = op("mul", op("mul", en, e2), inv(e1))
        return solid, free, mouth, t1, t2

    current_bars = op("add", blue, op("mul", selected, is_bar))
    current_elbows = op("add", elbow, op("mul", selected, is_elbow))
    solid_f, free_f, mouth_f, turn_f1, turn_f2 = geometry(
        current_bars, current_elbows)
    solid_ahead = neighbours(solid_f)
    wall_ahead = neighbours(walls)
    ordinary = inv(cap(op(
        "add", mouth_f, op("add", turn_f1, turn_f2))))
    seed = op("mul", repeat(source, 4), down)

    # Accumulating forward directional reachability.
    body = []
    target = body
    fs = "flow_state"
    usable = op("mul", fs, ordinary)
    straight = backwards(op("mul", usable, inv(solid_ahead)))
    branching = op("mul", usable, solid_ahead)
    b1, b2 = perp(branching, p1), perp(branching, p2)
    turns = op(
        "add", remap(op("mul", fs, turn_f1), p2, True),
        remap(op("mul", fs, turn_f2), p1, True))
    reflected = backwards(turns)
    reach = op("add", op("add", straight, b1), op("add", b2, reflected))
    reach = op("mul", cap(op("add", reach, seed)), free_f)
    next_flow = op("max2", fs, reach)
    target = G
    forward_id = nd(
        "loop", name="forward", T=40,
        state=[{"id": fs, "zeros": [4, 64, 64]}],
        body=body, next={fs: next_flow})
    flow = forward_id + "." + fs

    missed = op("mul", mouth_f, inv(flow))
    missing = gm(lin(missed, [1, 1, 1, 1]))
    leaking = gm(lin(op("mul", flow, wall_ahead), [1, 1, 1, 1]))
    solved = vt(vc(missing, leaking), [-1, -1], .5)

    # Backward fields exclude the selected piece.
    solid_b, free_b, mouth_b, turn_b1, turn_b2 = geometry(blue, elbow)
    branch_solid = op("add", blue, cups)
    outside_solid = inv(solid_b)

    right_wall = op(
        "mul", walls,
        op("mul", inside_r,
           binary(op("sub", c, gm(op("mul", bg, c))), -.0001)))
    above_floor = binary(
        op("sub", gm(op("mul", bg, r)), r), .0001)
    vertical_wall = gm(op("mul", right_wall, above_floor))
    outside_safe = vi(vertical_wall)
    ghost = op(
        "mul", op("mul", inside_r, inv(inside_c)), outside_safe)
    ghost4 = repeat(ghost, 4)

    boundary = inv(ahead(repeat(one, 4)))
    safe_boundary = op(
        "mul", op("mul", boundary, sideways), outside_safe)
    endpoint_edge1 = op(
        "mul", inv(perp(repeat(one, 4), p1)), outside_safe)
    endpoint_edge2 = op(
        "mul", inv(perp(repeat(one, 4), p2)), outside_safe)

    seed_good = cap(op("add", mouth_b, ghost4))
    seed_value = op(
        "add",
        op("mul", mouth_b, dn("avail", np.zeros((1, 8)), [.3])),
        op("mul", missed, dn("avail", np.zeros((1, 8)), [.7])))
    half = dn("avail", np.zeros((1, 8)), [.5])

    state_names = ["good_state", "good_left", "good_right",
                   "need_state", "need_left", "need_right"]
    gs, gl, gr, ns, nl, nr = state_names
    body = []
    target = body
    gout = op("mul", gs, outside_solid)
    nout = op("mul", ns, outside_solid)
    ngl = op(
        "max2", perp(op("add", gout, op("mul", gl, branch_solid)), p1),
        endpoint_edge1)
    ngr = op(
        "max2", perp(op("add", gout, op("mul", gr, branch_solid)), p2),
        endpoint_edge2)
    nnl = perp(op("add", nout, op("mul", nl, branch_solid)), p1)
    nnr = perp(op("add", nout, op("mul", nr, branch_solid)), p2)
    ga = op("max2", ahead(gs), safe_boundary)
    na = ahead(ns)
    gt = op(
        "max2", op("mul", remap(ga, p2), turn_b1),
        op("mul", remap(ga, p1), turn_b2))
    nt = op(
        "max2", op("mul", remap(na, p2), turn_b1),
        op("mul", remap(na, p1), turn_b2))
    gi = op("mul", branch_solid, op("min2", gl, gr))
    ni = op("mul", gi, op("mul", op("add", nl, nr), half))
    ng = op(
        "max2", seed_good,
        op("max2", gi, op("mul", free_b, op("max2", ga, gt))))
    nn = op(
        "max2", seed_value,
        op("max2", ni, op("mul", free_b, op("max2", na, nt))))
    target = G
    backward_id = nd(
        "loop", name="backward", T=40,
        state=[{"id": s, "zeros": [4, 64, 64]} for s in state_names],
        body=body,
        next={gs: ng, gl: ngl, gr: ngr, ns: nn, nl: nnl, nr: nnr})
    good = backward_id + "." + gs
    need = backward_id + "." + ns
    gd, vd = choose(good, down), choose(need, down)
    gright, vright = choose(good, right), choose(need, right)
    gleft, vleft = choose(good, left), choose(need, left)

    left_good = world_logical(gd, 0, -1)
    left_need = world_logical(vd, 0, -1)
    left_out = binary(
        op("sub", dn(vc(first_c, unit), [1, 1 / 63]), c), -.0001)
    left_good = op(
        "max2", left_good, op("mul", left_out, outside_safe))

    rg = world_logical(gright, -1, 0)
    rv = world_logical(vright, -1, 0)
    lg = world_logical(world_logical(gleft, -1, 0), 0, -1)
    lv = world_logical(world_logical(vleft, -1, 0), 0, -1)

    stripe = op(
        "add", op("mul", op("colmax", source), vertical),
        op("mul", op("rowmax", source), vi(vertical)))

    # One-cell exclusion margin around cups.
    cup_box = cups
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        cup_box = op("max2", cup_box, world_logical(cup_box, dy, dx))
    elbow_cup_box = op(
        "max2", cup_box, world_logical(cup_box, -1, 0))
    collision = op(
        "add", op("mul", cup_box, is_bar),
        op("mul", elbow_cup_box, is_elbow))
    candidate_base = cat(gd, vd, rg, rv, collision, stripe)

    # A shared pixel scan measures the selected footprint and its endpoint.
    pixel_w = np.zeros((8, 2, 3, 3), np.float32)
    offsets = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    for j in range(2):
        for k, (dy, dx) in enumerate(offsets):
            pixel_w[4 * j + k, j, 1 + dy, 1 + dx] = 1
    pixel_kernel = par(pixel_w)
    pixel_merge = np.zeros((2, 8, 1, 1), np.float32)
    pixel_merge[0, :4, 0, 0] = 1
    pixel_merge[1, 4:, 0, 0] = 1
    pixel_merge_name = par(pixel_merge)
    pixel_gate = dn(rotation, np.tile(np.eye(4), (2, 1)))

    def pixel_right(x):
        outputs = []
        for i in range(3):
            z = ch(x, 2 * i, 2 * i + 2)
            z = nd("conv", [z], w=pixel_kernel)
            z = op("mul", z, pixel_gate)
            outputs.append(nd("conv", [z], w=pixel_merge_name))
        return cat(*outputs)

    bs, count = "scan_base", "scan_count"
    cs, ss = "scan_cups", "scan_sources"
    body = []
    target = body
    active = vt(vc(width, count), [1, -1], -.5)
    moved = pixel_right(bs)
    nb = op("add", op("mul", moved, active), op("mul", bs, vi(active)))
    nc = op("max2", cs, op("mul", ch(bs, 4), active))
    nss = op("add", ss, op("mul", ch(bs, 5), active))
    ncount = dn(count, [1], [1])
    target = G
    scan_id = nd(
        "loop", name="scan", T=64,
        state=[{"id": bs, "init": candidate_base},
               {"id": count, "zeros": [1]},
               {"id": cs, "zeros": [1, 64, 64]},
               {"id": ss, "zeros": [1, 64, 64]}],
        body=body, next={bs: nb, count: ncount, cs: nc, ss: nss})
    scanned = scan_id + "." + bs

    right_out = binary(
        op("sub", op("add", c, dn(width, [1 / 63])), last_c),
        -.0001)
    right_good = op(
        "max2", ch(scanned, 0), op("mul", right_out, outside_safe))
    bar_quality = op("min2", left_good, right_good)
    bar_need = op("add", left_need, ch(scanned, 1))

    middle = dn(vc(current_c, width, unit),
                [1, 1 / 126, -1 / 126])
    moment = op(
        "gmean", op("mul", raw_selected, op("sub", c, middle)))
    gap_right = vt(moment, [-1], 0, 1000000)
    gap_left = vi(gap_right)
    elbow_quality = op(
        "add", op("mul", ch(scanned, 2), gap_right),
        op("mul", lg, gap_left))
    elbow_need = op(
        "add", op("mul", ch(scanned, 3), gap_right),
        op("mul", lv, gap_left))
    elbow_source = op(
        "add", op("mul", world_logical(stripe, 0, 1), gap_right),
        op("mul", stripe, gap_left))

    quality = op(
        "add", op("mul", bar_quality, is_bar),
        op("mul", elbow_quality, is_elbow))
    utility = op(
        "add", op("mul", bar_need, is_bar),
        op("mul", elbow_need, is_elbow))
    coverage = op(
        "add", op("mul", scan_id + "." + ss, is_bar),
        op("mul", elbow_source, is_elbow))

    minimum_row = dn(
        vc(first_r, unit, height), [1, 2 / 63, 1 / 63])
    final_offset = dn(vc(width, unit), [1 / 63, -1 / 63])
    bad_top = binary(op("sub", minimum_row, r), -.0001)
    bad_left = binary(op("sub", first_c, c), -.0001)
    bad_right = binary(
        op("sub", op("add", c, final_offset), last_c), -.0001)
    bad_bottom = binary(op("sub", r, last_r), -.0001)
    invalid = cap(op(
        "add", inv(cells),
        op("add", scan_id + "." + cs,
           op("add", op("add", bad_top, bad_bottom),
              op("add", bad_left, bad_right)))))
    valid = inv(invalid)
    feasible_map = op("mul", quality, valid)
    feasible = gm(feasible_map)

    dr, dc = op("sub", r, current_r), op("sub", c, current_c)
    distance = op(
        "add", op("mul", dc, dc), lin(op("mul", dr, dr), [2]))
    elbow_bias = op(
        "mul", is_elbow,
        op("sub", op("mul", c, gap_right), op("mul", c, gap_left)))
    score = lin(
        cat(feasible_map, op("mul", utility, valid),
            op("mul", coverage, valid), distance, invalid,
            elbow_bias, r, c),
        [8, 2, .3, -.8, -30, .12, -.0002, -.0003])
    peak = gm(score)
    winner = cap(op(
        "relu", lin(op("sub", score, peak), [1000000], [1])))
    tr = dn(gm(op("mul", winner, lin(r, [1], [1]))), [1], [-1])
    tc = dn(gm(op("mul", winner, lin(c, [1], [1]))), [1], [-1])

    moves = op("vrelu", dn(
        vc(tr, current_r, tc, current_c, unit),
        [[-1, 1, 0, 0, -.45 / 63],
         [1, -1, 0, 0, -.45 / 63],
         [0, 0, -1, 1, -.45 / 63],
         [0, 0, 1, -1, -.45 / 63]]))
    moving = vt(
        dn(moves, [1, 1, 1, 1]), [1], -.0001, 1000000)

    # Lower-left logical cells provide deterministic click anchors.
    others = op("add", blue, elbow)
    anchors = op(
        "mul", others,
        op("mul", inv(world_logical(others, 1, 0)),
           inv(world_logical(others, 0, -1))))
    label = lin(cat(r, c), [4032, 63], [1])
    current_label = dn(vc(current_r, current_c), [4032, 63], [1])
    match_distance = op("abs", op("sub", label, "mem"))
    matches = cap(op("relu", lin(
        match_distance, -10 * np.eye(8), [1] * 8)))
    remembered = cap(lin(matches, [1] * 8))
    unvisited = op("mul", anchors, inv(remembered))
    other_exists = gm(anchors)
    bad_anchor = op("mul", unvisited, inv(gd))
    lower_bad = gm(op("mul", bad_anchor, r))
    prefer_lower = vt(
        vc(lower_bad, current_r, unit), [1, -1, -1 / 63],
        -.0001, 1000000)

    move_gate = op(
        "vmul", op("vmul", moving, feasible), selected_exists)
    move_gate = op("vmul", move_gate, vi(prefer_lower))
    move_gate = op("vmul", move_gate, vi(solved))
    click_gate = op(
        "vmul", op("vmul", vi(move_gate), vi(solved)), other_exists)
    pour_gate = op(
        "vadd", solved,
        op("vmul", op("vmul", vi(move_gate), vi(other_exists)),
           vi(solved)))

    click_score = lin(
        cat(anchors, unvisited, bad_anchor, label, r),
        [200, 30, 10, .00001, .2])
    click_score = op(
        "add", click_score, lin(inv(anchors), [-1000]))
    nd("add", [click_score, one], name="click")

    # Convert world-relative U,D,L,R to display-relative actions.
    mw, rw = np.zeros((16, 4), np.float32), np.zeros((16, 4), np.float32)
    for i in range(4):
        for k in range(4):
            mw[4 * i + k, i] = 1
            rw[4 * i + k, k] = 1
    products = op("vmul", dn(moves, mw), dn(rotation, rw))
    mappings = [[1, 0, 3, 2], [3, 2, 0, 1],
                [0, 1, 2, 3], [2, 3, 1, 0]]
    output_map = np.zeros((4, 16), np.float32)
    for i in range(4):
        for k in range(4):
            output_map[mappings[k][i], 4 * i + k] = 1
    physical_moves = op("vmul", dn(products, output_map), move_gate)
    policy = vc(physical_moves, click_gate, pour_gate)
    ow = np.zeros((8, 6), np.float32)
    ow[1, 1] = ow[2, 0] = ow[3, 3] = ow[4, 2] = 150
    ow[5, 5], ow[6, 4] = 100, 50
    dn(policy, ow, [-100, 0, 0, 0, 0, -2, -1, -100],
       name="logits")

    fifo = np.zeros((8, 8), np.float32)
    for i in range(1, 8):
        fifo[i, i - 1] = 1
    insertion = np.zeros((8, 1), np.float32)
    insertion[0, 0] = 1
    proposed = op(
        "vadd", dn("mem", fifo), dn(current_label, insertion))
    new_mem = op("vmul", proposed, click_gate)
    old_mem = op("vmul", "mem", vi(click_gate))
    nd("vadd", [new_mem, old_mem], name="mem_next")

    # Remove graph computations not needed by the three outputs.
    def prune(nodes, roots):
        needed = set(roots)
        kept = []
        for d in reversed(nodes):
            if d["op"] == "loop":
                exported = {
                    d["id"] + "." + s["id"] for s in d["state"]}
                if not (needed & exported):
                    continue
                body_kept, external = prune(
                    d["body"], set(d["next"].values()))
                d["body"] = body_kept
                state_ids = {s["id"] for s in d["state"]}
                needed.update(external - state_ids)
                for s in d["state"]:
                    if "init" in s:
                        needed.add(s["init"])
                needed.difference_update(exported)
                kept.append(d)
            elif d["id"] in needed:
                needed.remove(d["id"])
                needed.update(d["in"])
                kept.append(d)
        return list(reversed(kept)), needed

    G, _ = prune(G, {"logits", "click", "mem_next"})

    def used_parameters(nodes):
        result = set()
        for d in nodes:
            for field in ("w", "b"):
                if field in d:
                    result.add(d[field])
            if d["op"] == "loop":
                result.update(used_parameters(d["body"]))
        return result

    used = used_parameters(G)
    P = {k: v for k, v in P.items() if k in used}

    # Purely static shape and operation accounting.
    def resources(nodes, shapes):
        macs, evaluations = 0, 0
        env = dict(shapes)
        for d in nodes:
            kind = d["op"]
            if kind == "loop":
                local = dict(env)
                for s in d["state"]:
                    local[s["id"]] = (
                        env[s["init"]] if "init" in s
                        else tuple(s["zeros"]))
                bm, be, result = resources(d["body"], local)
                macs += bm * d["T"]
                evaluations += 1 + be * d["T"]
                for s in d["state"]:
                    env[d["id"] + "." + s["id"]] = result[
                        d["next"][s["id"]]]
                continue
            evaluations += 1
            xs = [env[x] for x in d["in"]]
            a = xs[0]
            if kind == "conv":
                w = P[d["w"]]
                shape = (w.shape[0], 64, 64)
                macs += int(np.prod(w.shape)) * 4096
            elif kind == "dense":
                w = P[d["w"]]
                shape = (w.shape[0],)
                macs += int(np.prod(w.shape))
            elif kind in ("concat", "vconcat"):
                shape = (sum(s[0] for s in xs),) + a[1:]
            elif kind in ("slice", "vslice"):
                shape = (d["stop"] - d["start"],) + a[1:]
            elif kind in ("gmax", "gmean"):
                shape = (a[0],)
                macs += a[0] * 4096
            elif kind in ("rowmax", "colmax", "rowsum", "colsum"):
                shape = a
                macs += a[0] * 4096
            elif kind in ("add", "sub", "mul", "max2", "min2"):
                shape = (max(s[0] for s in xs), 64, 64)
                macs += shape[0] * 4096
            elif kind in ("vadd", "vmul"):
                shape = (max(s[0] for s in xs),)
                macs += shape[0]
            else:
                shape = a
            env[d["id"]] = shape
        return macs, evaluations, env

    inputs = {
        "x0": (16, 64, 64), "coords": (3, 64, 64),
        "mem": (8,), "aprev": (8,), "avail": (8,)}
    planning_loops = [
        d for d in G if d["op"] == "loop" and d["id"] != "scan"]
    while True:
        macs, evaluations, _ = resources(G, inputs)
        if macs <= 1480000000 and evaluations <= 9900:
            break
        adjustable = [d for d in planning_loops if d["T"] > 1]
        if not adjustable:
            break
        max(adjustable, key=lambda d: d["T"])["T"] -= 1

    return {"graph": G, "memory_size": 8, "params": P}
