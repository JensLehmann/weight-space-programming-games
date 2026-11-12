import numpy as np


def build():
    M, NT, NP = 102, 16, 8
    P, graph = {}, []
    out = graph
    serial = 0
    parameter_cache, expression_cache = {}, {}
    shapes = {
        "x0": ("s", 16), "coords": ("s", 3),
        "mem": ("v", M), "aprev": ("v", 8), "avail": ("v", 8),
    }

    def uid(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def param(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = uid("p")
            parameter_cache[key] = name
            P[name] = a
        return parameter_cache[key]

    def n(op, ins, name=None, **kw):
        key = (op, tuple(ins), tuple(sorted(kw.items())))
        if name is None and key in expression_cache:
            return expression_cache[key]
        name = name or uid()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(kw)
        out.append(nd)
        if op == "conv":
            sh = ("s", P[kw["w"]].shape[0])
        elif op == "dense":
            sh = ("v", P[kw["w"]].shape[0])
        elif op in ("slice", "vslice"):
            sh = (shapes[ins[0]][0], kw["stop"] - kw["start"])
        elif op in ("concat", "vconcat"):
            sh = ("s" if op == "concat" else "v",
                  sum(shapes[x][1] for x in ins))
        elif op in ("gmax", "gmean"):
            sh = ("v", shapes[ins[0]][1])
        elif op in ("add", "sub", "mul", "max2", "min2"):
            sh = ("s", max(shapes[x][1] for x in ins))
        elif op in ("vadd", "vmul"):
            sh = ("v", max(shapes[x][1] for x in ins))
        else:
            sh = shapes[ins[0]]
        shapes[name] = sh
        expression_cache[key] = name
        return name

    def C(x):
        return shapes[x][1]

    def conv(x, w, b=None, name=None):
        kw = {"w": w if isinstance(w, str) else param(w)}
        if b is not None:
            kw["b"] = param(b)
        return n("conv", [x], name=name, **kw)

    def dense(x, w, b=None, name=None):
        kw = {"w": w if isinstance(w, str) else param(w)}
        if b is not None:
            kw["b"] = param(b)
        return n("dense", [x], name=name, **kw)

    def const(a):
        a = np.asarray(a, np.float32).reshape(-1)
        return dense("avail", np.zeros((len(a), 8), np.float32), a)

    def ss(x, a, b):
        return n("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return n("vslice", [x], start=a, stop=b)

    def sc(xs):
        return n("concat", xs)

    def vc(xs, name=None):
        return n("vconcat", xs, name=name)

    def sa(x, a=1, b=0):
        if a == 1 and b == 0:
            return x
        return conv(x, (np.eye(C(x), dtype=np.float32) * a)[:, :, None, None],
                    np.full(C(x), b, np.float32))

    def va(x, a=1, b=0):
        if a == 1 and b == 0:
            return x
        return dense(x, np.eye(C(x), dtype=np.float32) * a,
                     np.full(C(x), b, np.float32))

    def vadd(a, b):
        return n("vadd", [a, b])

    def vmul(a, b):
        return n("vmul", [a, b])

    def vsub(a, b):
        return vadd(a, va(b, -1))

    def vinv(x):
        return va(x, -1, 1)

    def vrelu(x):
        return n("vrelu", [x])

    def vabs(x):
        return vadd(vrelu(x), vrelu(va(x, -1)))

    def vsum(x):
        return dense(x, np.ones((1, C(x)), np.float32))

    def vtotal(xs):
        x = xs[0]
        for y in xs[1:]:
            x = vadd(x, y)
        return x

    def vclip(x):
        c = C(x)
        eye = np.eye(c, dtype=np.float32)
        z = vrelu(dense(
            x, np.concatenate([eye, eye], axis=0),
            np.concatenate([np.zeros(c), -np.ones(c)])))
        return dense(z, np.concatenate([eye, -eye], axis=1))

    def vgt(x, threshold=0.5):
        c = C(x)
        eye = np.eye(c, dtype=np.float32)
        z = vrelu(dense(
            x, np.concatenate([eye, eye], axis=0) * 1024,
            np.concatenate([
                np.full(c, -1024 * threshold),
                np.full(c, -1024 * threshold - 1)])))
        return dense(z, np.concatenate([eye, -eye], axis=1))

    def veq(a, b):
        ca, cb = C(a), C(b)
        c = max(ca, cb)
        w = np.zeros((4 * c, ca + cb), np.float32)
        bias = np.empty(4 * c, np.float32)
        for k, offset in enumerate((0.5, 0.25, -0.25, -0.5)):
            for i in range(c):
                w[k * c + i, 0 if ca == 1 else i] = 1
                w[k * c + i, ca + (0 if cb == 1 else i)] = -1
                bias[k * c + i] = offset
        z = vrelu(dense(vc([a, b]), w, bias))
        eye = np.eye(c, dtype=np.float32)
        return dense(z, np.concatenate(
            [4 * eye, -4 * eye, -4 * eye, 4 * eye], axis=1))

    def vmax2(a, b):
        return vadd(b, vrelu(vsub(a, b)))

    def vmin2(a, b):
        return vsub(a, vrelu(vsub(a, b)))

    def winner(x):
        c = C(x)
        w = np.zeros((c * c, c), np.float32)
        summation = np.zeros((c, c * c), np.float32)
        for i in range(c):
            for j in range(c):
                w[i * c + j, j] += 1
                w[i * c + j, i] -= 1
                summation[i, i * c + j] = -8192
        costs = vrelu(dense(x, w))
        flags = vrelu(dense(costs, summation, np.ones(c, np.float32)))
        first = np.eye(c, dtype=np.float32) - np.tril(
            np.ones((c, c), np.float32), -1)
        return vrelu(dense(flags, first))

    def vmax(x):
        return vsum(vmul(x, winner(x)))

    def vblend(a, b, gate):
        return vadd(vmul(a, gate), vmul(b, vinv(gate)))

    def add(a, b):
        return n("add", [a, b])

    def sub(a, b):
        return n("sub", [a, b])

    def mul(a, b):
        return n("mul", [a, b])

    def ab(x):
        return n("abs", [x])

    def relu(x):
        return n("relu", [x])

    def gmax(x):
        return n("gmax", [x])

    def stotal(xs):
        x = xs[0]
        for y in xs[1:]:
            x = add(x, y)
        return x

    def channelsum(x):
        return conv(x, np.ones((1, C(x), 1, 1), np.float32))

    one = ss("coords", 2, 3)
    zero = sub(one, one)

    def clip(x):
        return n("min2", [relu(x), one])

    def inv(x):
        return sub(one, x)

    def near(a, b):
        return clip(sa(ab(sub(a, b)), -4, 2))

    def within(distance, radius):
        return clip(sa(sub(radius, distance), 4, 2))

    def sblend(a, b, gate):
        return add(mul(a, gate), mul(b, vinv(gate)))

    def begin_loop(T, specs):
        nonlocal out, expression_cache
        lid = uid("L")
        states, ids = [], {}
        for key, init in specs.items():
            sid = uid("s")
            ids[key] = sid
            if isinstance(init, str):
                states.append({"id": sid, "init": init})
                shapes[sid] = shapes[init]
            else:
                kind, c = init
                states.append({
                    "id": sid,
                    "zeros": [c, 64, 64] if kind == "s" else [c]
                })
                shapes[sid] = (kind, c)
        ctx = (out, expression_cache, lid, T, states, ids)
        out = []
        expression_cache = expression_cache.copy()
        return ctx, ids

    def end_loop(ctx, nxt):
        nonlocal out, expression_cache
        body = out
        out, expression_cache, lid, T, states, ids = ctx
        out.append({
            "id": lid, "op": "loop", "T": T,
            "state": states, "body": body,
            "next": {ids[k]: value for k, value in nxt.items()},
        })
        result = {}
        for key, sid in ids.items():
            result[key] = lid + "." + sid
            shapes[result[key]] = shapes[sid]
        return result

    shift_parameters = {}

    def shift(x, dy, dx):
        key = (C(x), dy, dx)
        if key not in shift_parameters:
            w = np.zeros((C(x), C(x), 3, 3), np.float32)
            for j in range(C(x)):
                w[j, j, 1 + dy, 1 + dx] = 1
            shift_parameters[key] = param(w)
        return conv(x, shift_parameters[key])

    row = conv("coords", np.array([[[[63]], [[0]], [[0]]]], np.float32))
    col = conv("coords", np.array([[[[0]], [[63]], [[0]]]], np.float32))
    row1, col1 = add(row, one), add(col, one)
    ordering = add(add(sa(row, 64), col), one)
    old_flag = vs("mem", 80, 81)
    old_running = vs("mem", 81, 82)
    old_locked = vs("mem", 82, 83)
    old_best = vs("mem", 86, 98)
    previous_switch = vs("aprev", 5, 6)
    palette = const(np.arange(6, 15, dtype=np.float32))

    def point_at(x, y):
        d = add(ab(sub(col, x)), ab(sub(row, y)))
        return clip(sa(d, -4, 2))

    def residues(coordinate):
        h = relu(conv(coordinate, np.ones((64, 1, 1, 1), np.float32),
                      -np.arange(64, dtype=np.float32)))
        w = np.zeros((3, 64, 1, 1), np.float32)
        b = np.zeros(3, np.float32)
        for r in range(3):
            f = np.array([float(j % 3 == r) for j in range(65)])
            b[r] = f[0]
            w[r, 0, 0, 0] = f[1] - f[0]
            for j in range(1, 64):
                w[r, j, 0, 0] = f[j + 1] - 2 * f[j] + f[j - 1]
        return clip(sa(conv(h, w, b), 4, -2))

    rr, cr = residues(row), residues(col)

    def reachable(point):
        r = gmax(mul(rr, point))
        c = gmax(mul(cr, point))
        return mul(channelsum(mul(rr, r)), channelsum(mul(cr, c)))

    def extract(mask, count, fields):
        specs = {"remaining": mask}
        for j in range(len(fields)):
            specs["f" + str(j)] = ("v", count)
        ctx, s = begin_loop(count, specs)
        best = gmax(mul(s["remaining"], ordering))
        pick = mul(s["remaining"], near(ordering, best))
        fifo = np.zeros((count, count), np.float32)
        fifo[:count - 1, 1:] = np.eye(count - 1, dtype=np.float32)
        insert = np.zeros((count, 1), np.float32)
        insert[-1, 0] = 1
        nxt = {"remaining": sub(s["remaining"], pick)}
        for j, field in enumerate(fields):
            value = gmax(mul(pick, field))
            nxt["f" + str(j)] = vadd(
                dense(s["f" + str(j)], fifo), dense(value, insert))
        result = end_loop(ctx, nxt)
        return [result["f" + str(j)] for j in range(len(fields))]

    # Target centres are recognized from either intact diagonal or axial borders.
    four = ss("x0", 4, 5)
    wc = np.zeros((1, 1, 3, 3), np.float32)
    wa = np.zeros_like(wc)
    for yy, xx in ((0, 0), (0, 2), (2, 0), (2, 2)):
        wc[0, 0, yy, xx] = 1
    for yy, xx in ((0, 1), (1, 0), (1, 2), (2, 1)):
        wa[0, 0, yy, xx] = 1
    detector = mul(n("max2", [
        clip(conv(four, wc, [-2])),
        clip(conv(four, wa, [-2]))
    ]), inv(four))
    fresh_tx, fresh_ty = extract(detector, NT, [col1, row1])
    tx = vblend(vs("mem", 0, 16), fresh_tx, old_flag)
    ty = vblend(vs("mem", 16, 32), fresh_ty, old_flag)
    target_dx, target_dy = sub(col1, tx), sub(row1, ty)
    target_ax, target_ay = ab(target_dx), ab(target_dy)
    target_points = clip(sa(add(target_ax, target_ay), -4, 2))

    colours = ss("x0", 6, 15)
    wn = np.zeros((9, 9, 3, 3), np.float32)
    for j in range(9):
        wn[j, j] = 1
        wn[j, j, 1, 1] = 0
    neighbours = conv(colours, wn)
    adjacent = clip(neighbours)
    lines = mul(colours, adjacent)
    isolated = mul(colours, inv(adjacent))
    number_weights = np.arange(6, 15, dtype=np.float32).reshape(1, 9, 1, 1)
    colour_numbers = conv(colours, number_weights)
    revealed = conv(isolated, number_weights)
    fresh_tc = gmax(mul(target_points, revealed))
    tc = vblend(fresh_tc, vs("mem", 32, 48), vgt(fresh_tc))

    # Detect and remember paint-box interiors and their dimensions.
    border = ss("x0", 2, 3)
    wp = np.zeros((1, 1, 3, 3), np.float32)
    wp[0, 0, 0, 1] = 1
    wp[0, 0, 1, 0] = 1
    anchors = clip(sa(add(conv(border, wp), channelsum(colours)), 1, -2))
    width_test = np.zeros((1, 1, 7, 7), np.float32)
    width_test[0, 0, 3, 6] = 1
    width_field = sa(conv(border, width_test), -1, 6)
    fresh_pads = extract(anchors, NP, [col1, row1, colour_numbers, width_field])
    pads = [
        vblend(vs("mem", 48 + 8 * j, 56 + 8 * j), fresh_pads[j], old_flag)
        for j in range(4)
    ]
    pad_x, pad_y, pad_colour, pad_width = pads
    pad_exists = vgt(pad_colour)
    pad_cx = vadd(pad_x, va(pad_width, 0.5, -2.5))
    pad_cy = vadd(pad_y, va(pad_width, 0.5, -2.5))
    pad_half = va(pad_width, 0.5, -0.5)
    near_border = clip(conv(border, np.ones((1, 1, 7, 7), np.float32)))
    interior = mul(mul(colours, clip(sa(neighbours, 1, -2))), near_border)
    available_colours = gmax(interior)
    clean_lines = mul(lines, inv(clip(channelsum(interior))))

    # Visible selected centre, with a short occlusion predictor.
    wx, wy = np.zeros((1, 8), np.float32), np.zeros((1, 8), np.float32)
    wx[0, 3], wx[0, 4] = -3, 3
    wy[0, 1], wy[0, 2] = -3, 3
    predicted_x = vadd(vs("mem", 100, 101), dense("aprev", wx))
    predicted_y = vadd(vs("mem", 101, 102), dense("aprev", wy))
    visible = ss("x0", 0, 1)
    centre = add(visible, mul(
        point_at(predicted_x, predicted_y),
        vmul(old_flag, vinv(gmax(visible)))))
    cx, cy = gmax(mul(centre, col)), gmax(mul(centre, row))
    reach = reachable(centre)

    # Eight rays measure continuous arms and first perimeter intersections.
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    specs = {
        "time": ("v", 1), "live": const(np.ones(72)),
        "length": ("v", 72), "seen": ("v", 72), "first": ("v", 72),
    }
    for j in range(8):
        specs["ray" + str(j)] = centre
    ctx, s = begin_loop(28, specs)
    t = va(s["time"], 1, 1)
    hits, nxt = [], {"time": t}
    for j, (yy, xx) in enumerate(directions):
        ray = shift(s["ray" + str(j)], -yy, -xx)
        nxt["ray" + str(j)] = ray
        hits.append(gmax(mul(ray, clean_lines)))
    hit = vc(hits)
    live = vmul(s["live"], hit)
    new_hit = vrelu(vsub(hit, s["seen"]))
    nxt.update({
        "live": live,
        "length": vadd(s["length"], live),
        "seen": vadd(s["seen"], new_hit),
        "first": vadd(s["first"], vmul(new_hit, t)),
    })
    probes = end_loop(ctx, nxt)

    total_lengths = dense(probes["length"], np.tile(np.eye(9), (1, 8)))
    first_costs = vadd(probes["first"], va(probes["seen"], -32, 32))
    fp = [vs(first_costs, 9 * j, 9 * (j + 1)) for j in range(8)]
    sp = [vs(probes["seen"], 9 * j, 9 * (j + 1)) for j in range(4)]
    shell_score = vtotal([
        va(vabs(vsub(fp[0], fp[1])), -1),
        va(vabs(vsub(fp[2], fp[3])), -1),
        va(vtotal(fp[:4]), -0.05),
        va(vtotal(sp), 30, -120),
    ])
    connected_colour = vgt(vmax(total_lengths), 2.5)
    colour_choice = winner(vblend(total_lengths, shell_score, connected_colour))
    actual_colour = vsum(vmul(colour_choice, palette))
    repeat_colour = dense(colour_choice, np.tile(np.eye(9), (8, 1)))
    sum_colour = np.zeros((8, 72), np.float32)
    for j in range(8):
        sum_colour[j, 9 * j:9 * (j + 1)] = 1
    lens = dense(vmul(probes["length"], repeat_colour), sum_colour)
    firsts = dense(vmul(probes["first"], repeat_colour), sum_colour)
    costs = dense(vmul(first_costs, repeat_colour), sum_colour)
    L = [vs(lens, j, j + 1) for j in range(8)]
    F = [vs(firsts, j, j + 1) for j in range(8)]
    FC = [vs(costs, j, j + 1) for j in range(8)]

    axial_length = vsum(vs(lens, 0, 4))
    diagonal_length = vsum(vs(lens, 4, 8))
    connected = vgt(vadd(axial_length, diagonal_length), 2.5)
    diagonal = vgt(vsub(diagonal_length, axial_length))
    is_x = vmul(connected, diagonal)
    is_axis = vmul(connected, vinv(diagonal))
    disconnected = vinv(connected)
    rectangle_test = vmul(
        veq(FC[4], vmin2(FC[0], FC[2])),
        veq(FC[7], vmin2(FC[1], FC[3])))
    is_rect = vmul(disconnected, rectangle_test)
    is_diamond = vmul(disconnected, vinv(rectangle_test))
    flags = [is_axis, is_x, is_rect, is_diamond]

    axis_rx, axis_ry = vmax2(L[2], L[3]), vmax2(L[0], L[1])
    xr = vmax(vs(lens, 4, 8))
    rect_rx, rect_ry = vmax2(F[2], F[3]), vmax2(F[0], F[1])
    dr = vmax2(rect_rx, rect_ry)
    rx = vtotal([vmul(axis_rx, is_axis), vmul(xr, is_x),
                 vmul(rect_rx, is_rect), vmul(dr, is_diamond)])
    ry = vtotal([vmul(axis_ry, is_axis), vmul(xr, is_x),
                 vmul(rect_ry, is_rect), vmul(dr, is_diamond)])

    # Evaluate translated copies against remembered target centres.
    zero_x, zero_y = near(target_dx, zero), near(target_dy, zero)
    equal_diagonals = near(target_ax, target_ay)
    manhattan = add(target_ax, target_ay)
    axis_cover = n("max2", [
        mul(zero_x, within(target_ay, ry)),
        mul(zero_y, within(target_ax, rx))
    ])
    x_cover = mul(equal_diagonals, within(target_ax, rx))
    rect_cover = n("max2", [
        mul(near(target_ax, rx), within(target_ay, ry)),
        mul(near(target_ay, ry), within(target_ax, rx))
    ])
    diamond_cover = near(manhattan, rx)
    cover = clip(stotal([
        mul(axis_cover, is_axis), mul(x_cover, is_x),
        mul(rect_cover, is_rect), mul(diamond_cover, is_diamond)
    ]))

    colour_masks, target_maps = [], []
    for j in range(9):
        mask = veq(tc, const([j + 6]))
        colour_masks.append(mask)
        target_maps.append(channelsum(mul(target_points, mask)))
    target_maps = sc(target_maps)
    matched = channelsum(mul(target_maps, lines))
    covered = gmax(mul(target_points, matched))
    own_geometry = gmax(mul(cover, centre))
    own = vmul(own_geometry, veq(tc, actual_colour))
    at_centre = gmax(mul(target_points, centre))
    centre_credit = vmul(
        vmul(at_centre, veq(tc, actual_colour)), vadd(is_axis, is_x))
    covered = vclip(vadd(covered, centre_credit))
    weights = vclip(vadd(vinv(covered), own))
    known = vgt(tc)
    valid_targets = vgt(tx)
    unknown = vmul(valid_targets, vinv(known))
    unknown_overlap = channelsum(mul(cover, unknown))

    scores = sc([
        channelsum(mul(cover, vmul(weights, mask)))
        for mask in colour_masks
    ])
    same_colour_vector = veq(palette, actual_colour)
    allowed = vclip(vadd(available_colours, same_colour_vector))
    distance = add(ab(sub(col, cx)), ab(sub(row, cy)))
    escaping = vgt(vs("mem", 99, 100))
    distance_coefficient = va(escaping, 0.005, -0.002)
    scores = add(scores, mul(distance, distance_coefficient))
    scores = sub(scores, sa(unknown_overlap, 0.15))
    scores = sub(scores, va(vinv(same_colour_vector), 0.05))
    scores = sub(scores, va(vinv(allowed), 1000))
    scores = sub(scores, sa(inv(reach), 1000))

    maxima = gmax(scores)
    desired_colour_vector = winner(maxima)
    best_score = vsum(vmul(maxima, desired_colour_vector))
    scalar_score = channelsum(mul(scores, desired_colour_vector))
    candidates = clip(sa(sub(scalar_score, best_score), 65536, 1))
    candidates = mul(clip(sa(candidates, 100, -90)), reach)
    best_rank = gmax(mul(candidates, ordering))
    chosen_position = mul(candidates, near(ordering, best_rank))
    goal_x = gmax(mul(chosen_position, col))
    goal_y = gmax(mul(chosen_position, row))
    goal_colour = vsum(vmul(desired_colour_vector, palette))
    baseline = vsub(vsum(own), va(vsum(vmul(own_geometry, unknown)), 0.15))
    quality = vsub(best_score, baseline)
    proposal = vc([
        quality, cx, cy, goal_x, goal_y, goal_colour,
        is_axis, is_x, is_rect, is_diamond, rx, ry
    ])

    # Survey cycle and persistent execution plan.
    source_match = vmul(
        veq(cx, vs(old_best, 1, 2)), veq(cy, vs(old_best, 2, 3)))
    selected = vclip(vadd(old_locked, source_match))
    completed = vmul(
        vmul(old_running, selected),
        vmul(
            vmul(veq(cx, vs(old_best, 3, 4)),
                 veq(cy, vs(old_best, 4, 5))),
            veq(actual_colour, vs(old_best, 5, 6))))
    timed_out = vmul(old_running, vgt(vs("mem", 98, 99), 79.5))
    restart = vclip(vadd(completed, timed_out))
    survey = vclip(vadd(vinv(old_running), restart))
    first_survey_step = vclip(vadd(vinv(old_flag), restart))
    ref_x = vblend(cx, vs("mem", 83, 84), first_survey_step)
    ref_y = vblend(cy, vs("mem", 84, 85), first_survey_step)
    count = vmul(
        vadd(vs("mem", 85, 86), vmul(previous_switch, survey)),
        vinv(first_survey_step))
    returned = vmul(
        vmul(veq(cx, ref_x), veq(cy, ref_y)), vgt(count))
    wrap = vmul(survey, vclip(vadd(returned, vgt(count, 5.5))))
    improve = vgt(vsub(quality, vs(old_best, 0, 1)), 0.0005)
    update = vmul(survey, vclip(vadd(first_survey_step, improve)))
    best = vblend(proposal, old_best, update)
    useful = vgt(vs(best, 0, 1), 0.02)
    launch = vmul(wrap, useful)
    running = vclip(vadd(vmul(old_running, vinv(restart)), launch))
    new_source_match = vmul(
        veq(cx, vs(best, 1, 2)), veq(cy, vs(best, 2, 3)))
    locked = vmul(running, vclip(vadd(
        vmul(old_locked, vinv(restart)), new_source_match)))
    recycle = vmul(wrap, vinv(useful))
    best = vblend(proposal, best, recycle)
    ref_x = vblend(cx, ref_x, recycle)
    ref_y = vblend(cy, ref_y, recycle)
    count = vmul(count, vinv(wrap))
    age = vmul(va(vs("mem", 98, 99), 1, 1),
               vmul(running, vinv(launch)))
    stall = vmul(vadd(vs("mem", 99, 100), recycle), vinv(launch))

    # Analytic paint-box intersections for the cached primitive.
    navigation_flags = [vs(best, j, j + 1) for j in range(6, 10)]
    nrx, nry = vs(best, 10, 11), vs(best, 11, 12)
    planned_colour = vs(best, 5, 6)
    bdx, bdy = sub(col, pad_cx), sub(row, pad_cy)
    bax, bay = ab(bdx), ab(bdy)
    min_x, min_y = relu(sub(bax, pad_half)), relu(sub(bay, pad_half))
    max_x, max_y = add(bax, pad_half), add(bay, pad_half)
    plus_hit = n("max2", [
        mul(within(bax, pad_half), within(min_y, nry)),
        mul(within(bay, pad_half), within(min_x, nrx))
    ])
    diagonal_strip = n("max2", [
        within(ab(sub(bdx, bdy)), va(pad_half, 2)),
        within(ab(add(bdx, bdy)), va(pad_half, 2))
    ])
    x_hit = mul(diagonal_strip,
                mul(within(min_x, nrx), within(min_y, nrx)))
    rect_hit = n("max2", [
        mul(mul(within(min_x, nrx), within(nrx, max_x)), within(min_y, nry)),
        mul(mul(within(min_y, nry), within(nry, max_y)), within(min_x, nrx))
    ])
    diamond_hit = mul(
        within(add(min_x, min_y), nrx), within(nrx, add(max_x, max_y)))
    paint_hits = mul(clip(stotal([
        mul(plus_hit, navigation_flags[0]),
        mul(x_hit, navigation_flags[1]),
        mul(rect_hit, navigation_flags[2]),
        mul(diamond_hit, navigation_flags[3])
    ])), pad_exists)
    right_pads = vmul(veq(pad_colour, planned_colour), pad_exists)
    current_pads = vmul(veq(pad_colour, actual_colour), pad_exists)
    wrong_pads = vmul(vinv(vclip(vadd(right_pads, current_pads))), pad_exists)
    paint_goal = clip(channelsum(mul(paint_hits, right_pads)))
    blocked = clip(channelsum(mul(paint_hits, wrong_pads)))
    walkable = n("max2", [inv(blocked), centre])
    destination = point_at(vs(best, 3, 4), vs(best, 4, 5))
    goal = mul(sblend(destination, paint_goal,
                      veq(actual_colour, planned_colour)), reach)

    # Value propagation on the three-pixel translation lattice.
    step_kernel = np.zeros((4, 1, 7, 7), np.float32)
    for j, (yy, xx) in enumerate(directions[:4]):
        step_kernel[j, 0, 3 + 3 * yy, 3 + 3 * xx] = 1
    step_weight = param(step_kernel)
    ctx, s = begin_loop(64, {"value": sa(goal, 96)})
    neighbour_values = conv(s["value"], step_weight, [-1, -1, -1, -1])
    value = s["value"]
    for j in range(4):
        value = n("max2", [value, ss(neighbour_values, j, j + 1)])
    value = mul(relu(value), walkable)
    navigation = end_loop(ctx, {"value": value})
    movement = gmax(mul(conv(navigation["value"], step_weight), centre))

    wmove = np.zeros((8, 4), np.float32)
    wmove[1:5] = np.eye(4, dtype=np.float32)
    move_logits = dense(
        movement, wmove, [-10000, 0, 0, 0, 0, -1, -10000, -10000])
    wswitch = np.zeros((8, 1), np.float32)
    wswitch[5, 0] = 200
    switch = vinv(vmul(running, locked))
    n("vadd", [move_logits, dense(switch, wswitch)], name="logits")
    conv(zero, np.zeros((1, 1, 1, 1), np.float32), name="click")
    vc([
        tx, ty, tc, pad_x, pad_y, pad_colour, pad_width,
        const([1]), running, locked, ref_x, ref_y, count,
        best, age, stall, cx, cy
    ], name="mem_next")

    # Static dead-code elimination; this does not execute the actor.
    def prune(nodes, roots):
        needed = set(roots)
        kept = []
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                outputs = {nd["id"] + "." + s["id"] for s in nd["state"]}
                if not outputs.intersection(needed):
                    continue
                body, dependencies = prune(nd["body"], nd["next"].values())
                nd = dict(nd)
                nd["body"] = body
                states = {s["id"] for s in nd["state"]}
                needed.update(dependencies - states)
                for state in nd["state"]:
                    if "init" in state:
                        needed.add(state["init"])
                kept.append(nd)
            elif nd["id"] in needed:
                needed.update(nd["in"])
                kept.append(nd)
        kept.reverse()
        defined = {nd["id"] for nd in nodes}
        for nd in nodes:
            if nd["op"] == "loop":
                defined.update(nd["id"] + "." + s["id"] for s in nd["state"])
        return kept, needed - defined

    graph, _ = prune(graph, {"logits", "click", "mem_next"})
    used = set()

    def collect(nodes):
        for nd in nodes:
            if nd["op"] == "loop":
                collect(nd["body"])
            else:
                for key in ("w", "b"):
                    if key in nd:
                        used.add(nd[key])

    collect(graph)
    P = {key: value for key, value in P.items() if key in used}

    # Check the constructed artifact, without evaluating any game state.
    def budget(nodes, factor=1):
        count = evaluations = spatial = mac = 0
        for nd in nodes:
            count += 1
            if nd["op"] == "loop":
                evaluations += factor
                for state in nd["state"]:
                    kind, c = shapes[state["id"]]
                    if kind == "s":
                        spatial += c
                a, b, c, d = budget(nd["body"], factor * nd["T"])
                count += a
                evaluations += b
                spatial += c
                mac += d
                continue
            evaluations += factor
            kind, channels = shapes[nd["id"]]
            if kind == "s":
                spatial += channels
            op = nd["op"]
            local = 0
            if op == "conv":
                local = int(P[nd["w"]].size) * 4096
            elif op == "dense":
                local = int(P[nd["w"]].size)
            elif op in ("add", "sub", "mul", "max2", "min2"):
                local = channels * 4096
            elif op in ("gmax", "gmean", "rowmax", "colmax", "rowsum", "colsum"):
                local = shapes[nd["in"][0]][1] * 4096
            elif op in ("vadd", "vmul"):
                local = channels
            mac += factor * local
        return count, evaluations, spatial, mac

    node_count, evaluations, spatial, mac = budget(graph)
    entries = sum(a.size for a in P.values())
    assert node_count <= 1000, ("nodes", node_count)
    assert evaluations <= 10000, ("evaluations", evaluations)
    assert spatial + 19 <= 4096, ("spatial channels", spatial + 19)
    assert entries <= 150000, ("parameters", entries)
    assert mac <= 1500000000, ("MACs", mac)

    return {"graph": graph, "memory_size": M, "params": P}
