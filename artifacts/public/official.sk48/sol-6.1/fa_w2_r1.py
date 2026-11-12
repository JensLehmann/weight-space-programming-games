import numpy as np


def build():
    # Resource-based architecture sizing, not weight optimization.
    for n in (6, 5, 4, 3, 2, 1):
        artifact, usage = _make_actor(n)
        channels, macs, nodes, evaluations, entries = usage
        if (channels <= 4096 and macs <= 1500000000
                and nodes <= 1000 and evaluations <= 10000
                and entries <= 150000):
            return artifact
    return _fallback()


def _make_actor(N):
    K = 6
    W = 1
    M = 9
    graph = []
    params = {}
    shapes = {
        "x0": (16, True), "coords": (3, True),
        "mem": (M, False), "aprev": (8, False), "avail": (8, False),
    }
    current = graph
    serial = 0
    p_cache = {}
    n_cache = {}
    constants = {}

    def parameter(value):
        a = np.asarray(value, np.float32)
        key = (a.shape, a.tobytes())
        if key not in p_cache:
            name = "p%d" % len(params)
            params[name] = a
            p_cache[key] = name
        return p_cache[key]

    def node(op, inputs, name=None, destination=None, **fields):
        nonlocal serial
        dst = current if destination is None else destination
        key = (id(dst), op, tuple(inputs),
               tuple(sorted((k, str(v)) for k, v in fields.items())))
        if name is None and key in n_cache:
            return n_cache[key]
        if name is None:
            name = "n%d" % serial
            serial += 1
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        dst.append(nd)

        if op in ("conv", "dense"):
            shape = (params[fields["w"]].shape[0], op == "conv")
        elif op in ("gmax", "gmean"):
            shape = (shapes[inputs[0]][0], False)
        elif op in ("concat", "vconcat"):
            shape = (sum(shapes[x][0] for x in inputs), op == "concat")
        elif op in ("slice", "vslice"):
            shape = (fields["stop"] - fields["start"], op == "slice")
        elif op in ("add", "sub", "mul", "max2", "min2"):
            if not any(shapes[x][1] for x in inputs):
                raise ValueError("Spatial operation requires a spatial input")
            shape = (max(shapes[x][0] for x in inputs), True)
        elif op in ("vadd", "vmul"):
            if any(shapes[x][1] for x in inputs):
                raise ValueError("Vector operation requires vector inputs")
            shape = (max(shapes[x][0] for x in inputs), False)
        else:
            shape = shapes[inputs[0]]
        shapes[name] = shape
        n_cache[key] = name
        return name

    def linear(x, weights, bias=None, name=None):
        a = np.asarray(weights, np.float32)
        spatial = shapes[x][1]
        kw = {"w": parameter(a[:, :, None, None] if spatial else a)}
        if bias is not None:
            kw["b"] = parameter(bias)
        return node("conv" if spatial else "dense", [x], name=name, **kw)

    def convolution(x, weights, bias=None):
        kw = {"w": parameter(weights)}
        if bias is not None:
            kw["b"] = parameter(bias)
        return node("conv", [x], **kw)

    def constant(value):
        value = float(value)
        if value not in constants:
            constants[value] = node(
                "dense", ["avail"], destination=graph,
                w=parameter(np.zeros((1, 8), np.float32)),
                b=parameter([value]))
        return constants[value]

    def sl(x, start, stop=None):
        return node(
            "slice" if shapes[x][1] else "vslice", [x],
            start=start, stop=start + 1 if stop is None else stop)

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

    def add(a, b):
        return node(
            "add" if shapes[a][1] or shapes[b][1] else "vadd", [a, b])

    def mul(a, b):
        return node(
            "mul" if shapes[a][1] or shapes[b][1] else "vmul", [a, b])

    def scale(a, k):
        return mul(a, constant(k))

    def sub(a, b):
        if shapes[a][1] or shapes[b][1]:
            return node("sub", [a, b])
        return add(a, scale(b, -1))

    def affine(a, k=1, b=0):
        if k == 1 and b == 0:
            return a
        if b == 0:
            return scale(a, k)
        c = shapes[a][0]
        return linear(
            a, np.eye(c, dtype=np.float32) * k,
            np.full(c, b, np.float32))

    def relu(a):
        return node("relu" if shapes[a][1] else "vrelu", [a])

    def broadcast(a):
        return a if shapes[a][1] else add(zero, a)

    def absolute(a):
        if shapes[a][1]:
            return node("abs", [a])
        return add(relu(a), relu(scale(a, -1)))

    def maximum(a, b):
        if not shapes[a][1] and not shapes[b][1]:
            a = broadcast(a)
        return node("max2", [a, b])

    def clip(a):
        return node("min2", [relu(broadcast(a)), one])

    def positive(a):
        return clip(scale(a, 64))

    def equal_zero(a):
        return clip(affine(absolute(a), -64, 3.2))

    def total(a):
        return linear(a, np.ones((1, shapes[a][0]), np.float32))

    def finish_loop(name, states, body, next_values):
        graph.append({
            "id": name, "op": "loop",
            "T": 1, "state": states, "body": body, "next": next_values,
        })
        for state in states:
            if "init" in state:
                shapes[state["id"]] = shapes[state["init"]]
            else:
                z = state["zeros"]
                shapes[state["id"]] = (z[0], len(z) == 3)
            shapes[name + "." + state["id"]] = shapes[state["id"]]
        return graph[-1]

    one = sl("coords", 2)
    zero = linear("coords", np.zeros((1, 3), np.float32))
    pos = linear("coords", [
        [0, 63 / 6, -2 / 6],
        [63 / 6, 0, -2 / 6],
    ])
    px, py = sl(pos, 0), sl(pos, 1)
    pixel_row = linear("coords", [[63, 0, 0]])
    upper = positive(sub(constant(52.5), pixel_row))
    lower = sub(one, upper)
    priority = linear("coords", [[-4032, -63, 4096]])

    # Marker position is sprite top-left plus (2,2).
    # Sample an interior corner, which is not hidden by target-match marks.
    ck = np.zeros((1, 16, 3, 3), np.float32)
    ck[0, :, 0, 0] = np.arange(16, dtype=np.float32)
    colour_number = convolution("x0", ck)
    object_features = cat(pos, colour_number, one)

    colours = [7, 8, 9, 10, 12, 13, 14]
    bw = np.zeros((len(colours), 16, 5, 5), np.float32)
    for o, c in enumerate(colours):
        for dy, dx in [(-1, -1), (-1, 2), (2, -1), (2, 2)]:
            bw[o, c, dy + 2, dx + 2] = 1
    ball_mark = clip(total(relu(
        convolution("x0", bw, [-3] * len(colours)))))

    extraction_number = 0

    def extract(mark, count):
        nonlocal current, extraction_number
        number = extraction_number
        extraction_number += 1
        lname = "Extract%d" % number
        mask_name = "extract_mask%d" % number
        pack_name = "extract_pack%d" % number
        states = [
            {"id": mask_name, "init": mark},
            {"id": pack_name, "zeros": [4 * count]},
        ]
        shapes[mask_name] = (1, True)
        shapes[pack_name] = (4 * count, False)

        body = []
        current = body
        score = mul(mask_name, priority)
        best = node("gmax", [score])
        winner = node(
            "min2",
            [relu(affine(sub(score, best), b=1)), mask_name])
        feature = scale(
            node("gmean", [mul(object_features, winner)]), 4096)
        packed_input = vcat(pack_name, feature)
        shift = np.zeros((4 * count, 4 * count + 4), np.float32)
        if count > 1:
            shift[:4 * (count - 1), 4:4 * count] = np.eye(
                4 * (count - 1), dtype=np.float32)
        shift[-4:, -4:] = np.eye(4, dtype=np.float32)
        pack_next = linear(packed_input, shift)
        mask_next = sub(mask_name, winner)
        current = graph
        loop = finish_loop(lname, states, body, {
            mask_name: mask_next, pack_name: pack_next})
        loop["T"] = count

        packed = lname + "." + pack_name
        result = []
        for component in range(4):
            a = np.zeros((count, 4 * count), np.float32)
            for i in range(count):
                a[i, 4 * i + component] = 1
            result.append(linear(packed, a))
        return result

    ball_xv, ball_yv, ball_cv, ball_ev = extract(
        mul(ball_mark, upper), N)
    target_xv, target_yv, target_cv, target_ev = extract(
        mul(ball_mark, lower), K)

    # Controllable head families have rims 6 and 15.
    # A fixed head has a two-by-two colour-11 centre.
    hw = np.zeros((3, 16, 7, 7), np.float32)
    for o, c in enumerate([6, 15]):
        for dy, dx in [
            (-2, -2), (-2, 3), (3, -2),
            (3, 3), (0, 0), (1, 1),
        ]:
            hw[o, c, dy + 3, dx + 3] = 1
    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        hw[2, 11, dy + 3, dx + 3] = 1
    heads_all = relu(convolution("x0", hw, [-5, -5, -3]))
    heads = [mul(sl(heads_all, i), upper) for i in range(3)]

    faces = [
        [(0, 3), (1, 3)],
        [(3, 0), (3, 1)],
        [(0, -2), (1, -2)],
        [(-2, 0), (-2, 1)],
    ]
    ow = np.zeros((13, 16, 7, 7), np.float32)
    ob = np.zeros(13, np.float32)
    for family, c in enumerate([6, 15]):
        for face in range(4):
            o = 4 * family + face
            ob[o] = 1
            for dy, dx in faces[face]:
                ow[o, c, dy + 3, dx + 3] = -.5
    for face in range(4):
        for dy, dx in faces[face]:
            for c in [1, 2, 3]:
                ow[8 + face, c, dy + 3, dx + 3] = .5
    ow[12, 0, 2, 2] = 1
    openings = relu(convolution("x0", ow, ob))

    head_vectors = []
    for family in range(3):
        d = np.zeros((2, 13), np.float32)
        d[0, 4 * family] = 1
        d[0, 4 * family + 2] = -1
        d[1, 4 * family + 1] = 1
        d[1, 4 * family + 3] = -1
        features = cat(pos, linear(openings, d), sl(openings, 12), one)
        head_vectors.append(scale(
            node("gmean", [mul(features, heads[family])]), 4096))

    # Rod endpoints remain visible at the edges of coloured ball interiors.
    sw = np.zeros((4, 16, 7, 7), np.float32)
    for active in range(2):
        a, b = (2, 3) if active == 0 else (1, 2)
        for dy, dx, c in [
            (0, -2, b), (1, -2, a), (0, 3, a), (1, 3, b),
        ]:
            sw[active, c, dy + 3, dx + 3] = 1
        for dy, dx, c in [
            (-2, 0, a), (-2, 1, b), (3, 0, b), (3, 1, a),
        ]:
            sw[2 + active, c, dy + 3, dx + 3] = 1
    strips = relu(convolution("x0", sw, [-3] * 4))
    horizontal_strip = maximum(sl(strips, 0), sl(strips, 1))
    vertical_strip = maximum(sl(strips, 2), sl(strips, 3))

    head_data = []
    for i, hv in enumerate(head_vectors):
        hx, hy, dx, dy, selected, exists = [
            sl(hv, j) for j in range(6)]
        mask = add(
            mul(mul(horizontal_strip, node("rowmax", [heads[i]])),
                absolute(dx)),
            mul(mul(vertical_strip, node("colmax", [heads[i]])),
                absolute(dy)))
        projection = add(mul(px, dx), mul(py, dy))
        end = sub(node("gmax", [
            mul(affine(projection, b=20), mask)]), constant(20))
        start = add(mul(hx, dx), mul(hy, dy))
        length = mul(add(relu(sub(end, start)), constant(1)), exists)
        head_data.append((hx, hy, dx, dy, selected, exists, length))

    # Extensive floor-colour rows and columns identify the board rectangle.
    floor = mul(sl("x0", 4), upper)
    row_mask = positive(sub(node("rowsum", [floor]), constant(12.5)))
    col_mask = positive(sub(node("colsum", [floor]), constant(12.5)))

    def extrema(coordinate, mask):
        high = node("gmax", [mul(affine(coordinate, b=20), mask)])
        low = node("gmax", [mul(affine(coordinate, -1, 20), mask)])
        return sub(constant(20), low), sub(high, constant(20))

    minx, maxx = extrema(px, col_mask)
    miny, maxy = extrema(py, row_mask)
    minx, miny = add(minx, constant(2 / 6)), add(miny, constant(2 / 6))
    maxx, maxy = sub(maxx, constant(3 / 6)), sub(maxy, constant(3 / 6))

    def within_x(x):
        return mul(
            positive(affine(sub(x, minx), b=.05)),
            positive(affine(sub(maxx, x), b=.05)))

    def within_y(y):
        return mul(
            positive(affine(sub(y, miny), b=.05)),
            positive(affine(sub(maxy, y), b=.05)))

    def inside(x, y):
        return mul(within_x(x), within_y(y))

    ww = np.zeros((1, 16, 7, 7), np.float32)
    for dy, dx in [(-2, -2), (-2, 3), (3, -2), (3, 3)]:
        ww[0, 5, dy + 3, dx + 3] = 1
    wall_mark = mul(
        relu(convolution("x0", ww, [-3])), inside(px, py))
    wall_xv, wall_yv, _, wall_ev = extract(wall_mark, W)

    bottom_second = mul(sl(heads_all, 1), lower)
    bottom_x = scale(node("gmean", [mul(px, bottom_second)]), 4096)
    bottom_exists = scale(node("gmean", [bottom_second]), 4096)

    bx0, by0 = broadcast(ball_xv), broadcast(ball_yv)
    ball_colour, ball_exists = broadcast(ball_cv), broadcast(ball_ev)
    target_exists = broadcast(target_ev)
    target_family = mul(
        positive(sub(broadcast(target_xv), add(bottom_x, constant(.5)))),
        bottom_exists)
    family0 = sub(one, target_family)
    target0, target1 = (
        mul(target_exists, family0),
        mul(target_exists, target_family))
    triangle = np.tril(np.ones((K, K), np.float32))
    target_rank = add(
        mul(linear(target0, triangle), family0),
        mul(linear(target1, triangle), target_family))
    count0, count1 = total(target0), total(target1)
    wall_x, wall_y, wall_exists = (
        broadcast(wall_xv), broadcast(wall_yv), broadcast(wall_ev))

    def pair_ball(x, count):
        c = shapes[x][0]
        a = np.zeros((c * count, c), np.float32)
        for i in range(c):
            a[i * count:(i + 1) * count, i] = 1
        return linear(x, a)

    def pair_other(x, count):
        c = shapes[x][0]
        return linear(x, np.tile(np.eye(c, dtype=np.float32), (count, 1)))

    def pair_difference(x):
        a = np.zeros((N * N, N), np.float32)
        for i in range(N):
            for j in range(N):
                a[i * N + j, i] += 1
                a[i * N + j, j] -= 1
        return linear(x, a)

    def project(x, y, hx, hy, dx, dy):
        xx, yy = sub(x, hx), sub(y, hy)
        return (
            add(mul(xx, dx), mul(yy, dy)),
            sub(mul(xx, dy), mul(yy, dx)))

    def along(t, length):
        return mul(
            positive(affine(t, b=.05)),
            positive(sub(affine(length, b=-.95), t)))

    def rod_mask(t, q, length):
        return mul(along(t, length), equal_zero(q))

    wall_reduce = np.zeros((N, N * W), np.float32)
    for i in range(N):
        wall_reduce[i, i * W:(i + 1) * W] = 1

    def wall_collision(x, y):
        dx = sub(pair_ball(x, W), pair_other(wall_x, N))
        dy = sub(pair_ball(y, W), pair_other(wall_y, N))
        contact = equal_zero(add(absolute(dx), absolute(dy)))
        return positive(linear(
            mul(contact, pair_other(wall_exists, N)), wall_reduce))

    # The image grid is repurposed as 4,096 independent candidate rollouts.
    def decode_seed(seed):
        value = scale(seed, 5)
        thresholds = linear(
            value, np.ones((4, 1), np.float32), [-1, -2, -3, -4])
        floors = clip(affine(thresholds, 100000, 1))
        a = np.zeros((5, 4), np.float32)
        a[0, 0] = -1
        for i in range(1, 4):
            a[i, i - 1] = 1
            a[i, i] = -1
        a[4, 3] = 1
        return (
            linear(floors, a, [1, 0, 0, 0, 0]),
            sub(value, total(floors)))

    seed0 = linear("coords", [[4032 / 4096, 63 / 4096, .5 / 4096]])
    first_choices, _ = decode_seed(seed0)
    h0, h1, hf = head_data
    initial_pose_vector = vcat(
        h0[0], h0[1], h0[6], h1[0], h1[1], h1[6])
    selected0 = broadcast(h1[4])
    click_possible = mul(h0[5], h1[5])

    states = [
        {"id": "psx", "init": bx0},
        {"id": "psy", "init": by0},
        {"id": "ppose", "init": broadcast(initial_pose_vector)},
        {"id": "pselected", "init": selected0},
        {"id": "pseed", "init": seed0},
        {"id": "pcost", "zeros": [1, 64, 64]},
        {"id": "phold", "zeros": [5, 64, 64]},
    ]
    for s in states:
        shapes[s["id"]] = (
            shapes[s["init"]] if "init" in s
            else (s["zeros"][0], True))

    body = []
    current = body
    decoded, decoded_next = decode_seed("pseed")
    held = total("phold")
    fresh = sub(one, held)
    choices = add(mul(decoded, fresh), "phold")
    seed_next = add(mul(decoded_next, fresh), mul("pseed", held))
    move_x = linear(choices, [[0, 0, -1, 1, 0]])
    move_y = linear(choices, [[-1, 1, 0, 0, 0]])
    switch = sl(choices, 4)
    selected = "pselected"
    not_selected = sub(one, selected)
    bx, by = "psx", "psy"

    def blend(a, b):
        return add(mul(a, not_selected), mul(b, selected))

    def other_blend(a, b):
        return add(mul(a, selected), mul(b, not_selected))

    p = [sl("ppose", i) for i in range(6)]
    hx, hy, length = [blend(p[i], p[i + 3]) for i in range(3)]
    dx, dy = blend(h0[2], h1[2]), blend(h0[3], h1[3])
    ox, oy, olen = [other_blend(p[i], p[i + 3]) for i in range(3)]
    odx, ody = other_blend(h0[2], h1[2]), other_blend(h0[3], h1[3])

    parallel = add(mul(move_x, dx), mul(move_y, dy))
    extension = positive(sub(parallel, constant(.5)))
    retraction = positive(sub(scale(parallel, -1), constant(.5)))
    sideways = sub(sub(one, absolute(parallel)), switch)
    nhx = add(hx, mul(sideways, move_x))
    nhy = add(hy, mul(sideways, move_y))
    nlen = add(length, parallel)
    tipx, tipy = add(hx, mul(length, dx)), add(hy, mul(length, dy))

    valid = add(
        add(mul(extension, inside(tipx, tipy)),
            mul(retraction, positive(sub(length, constant(1.5))))),
        mul(sideways, add(
            mul(absolute(dx), within_y(nhy)),
            mul(absolute(dy), within_x(nhx)))))

    wt, wq = project(wall_x, wall_y, nhx, nhy, dx, dy)
    rod_wall = positive(total(
        mul(rod_mask(wt, wq, nlen), wall_exists)))
    valid = mul(
        valid, sub(one, mul(add(extension, sideways), rod_wall)))

    t, q = project(bx, by, hx, hy, dx, dy)
    cross_move = sub(mul(move_x, dy), mul(move_y, dx))
    touched = mul(
        mul(along(t, add(length, extension)),
            maximum(equal_zero(q), equal_zero(sub(q, cross_move)))),
        ball_exists)

    nbx, nby = add(bx, move_x), add(by, move_y)
    blocked = maximum(
        sub(one, inside(nbx, nby)), wall_collision(nbx, nby))
    for rx, ry, rdx, rdy, rlen in [
        (ox, oy, odx, ody, olen),
        (hf[0], hf[1], hf[2], hf[3], hf[6]),
    ]:
        rt, rq = project(bx, by, rx, ry, rdx, rdy)
        rp = add(mul(move_x, rdx), mul(move_y, rdy))
        rc = sub(mul(move_x, rdy), mul(move_y, rdx))
        contact = maximum(equal_zero(rq), equal_zero(add(rq, rc)))
        blocked = maximum(blocked, mul(
            mul(along(rt, rlen), contact), sub(one, absolute(rp))))

    contact = equal_zero(add(
        absolute(add(pair_difference(bx), move_x)),
        absolute(add(pair_difference(by), move_y))))
    contact_twice = cat(contact, contact)
    flags = cat(touched, blocked)
    expansion = np.zeros((2 * N * N, 2 * N), np.float32)
    aggregation = np.zeros((2 * N, 2 * N * N), np.float32)
    for i in range(N):
        for j in range(N):
            z = i * N + j
            expansion[z, i] = 1
            expansion[N * N + z, N + j] = 1
            aggregation[j, z] = 1
            aggregation[N + i, N * N + z] = 1
    for _ in range(2):
        propagated = linear(
            mul(linear(flags, expansion), contact_twice), aggregation)
        flags = maximum(flags, clip(propagated))

    eligible, blocked_final = sl(flags, 0, N), sl(flags, N, 2 * N)
    failed_push = positive(total(mul(eligible, blocked_final)))
    valid = mul(valid, sub(one, mul(sideways, failed_push)))
    ball_move = mul(mul(eligible, sub(one, blocked_final)), valid)
    bx_next, by_next = (
        add(bx, mul(ball_move, move_x)),
        add(by, mul(ball_move, move_y)))
    head_dx = mul(mul(sideways, move_x), valid)
    head_dy = mul(mul(sideways, move_y), valid)
    length_delta = mul(parallel, valid)
    pose_next = cat(
        add(p[0], mul(head_dx, not_selected)),
        add(p[1], mul(head_dy, not_selected)),
        add(p[2], mul(length_delta, not_selected)),
        add(p[3], mul(head_dx, selected)),
        add(p[4], mul(head_dy, selected)),
        add(p[5], mul(length_delta, selected)))

    valid_switch = mul(switch, click_possible)
    selected_next = add(
        selected, mul(affine(selected, -2, 1), valid_switch))
    wasted = sub(sub(one, valid), valid_switch)
    cost_next = add(
        "pcost", add(scale(wasted, .25), scale(valid_switch, .025)))
    tip_ball = positive(total(mul(
        mul(equal_zero(sub(bx, tipx)), equal_zero(sub(by, tipy))),
        ball_exists)))
    hold_next = mul(
        choices, mul(mul(extension, valid), sub(one, tip_ball)))

    current = graph
    plan_loop = finish_loop("Plan", states, body, {
        "psx": bx_next, "psy": by_next, "ppose": pose_next,
        "pselected": selected_next, "pseed": seed_next,
        "pcost": cost_next, "phold": hold_next,
    })
    plan_loop["T"] = 12

    bx, by = "Plan.psx", "Plan.psy"
    selected = "Plan.pselected"
    p = [sl("Plan.ppose", i) for i in range(6)]
    rank_reduce = np.zeros((N, N * N), np.float32)
    for i in range(N):
        rank_reduce[i, i * N:(i + 1) * N] = 1

    on_rods, projections, transverse, ranks, counts = [], [], [], [], []
    for family, head in enumerate([h0, h1]):
        hx, hy, length = p[3 * family:3 * family + 3]
        t, q = project(bx, by, hx, hy, head[2], head[3])
        on = mul(rod_mask(t, q, length), ball_exists)
        earlier = mul(
            clip(affine(pair_difference(t), 4, -.2)),
            pair_other(on, N))
        rank = linear(earlier, rank_reduce, np.ones(N, np.float32))
        on_rods.append(on)
        projections.append(t)
        transverse.append(q)
        ranks.append(rank)
        counts.append(total(on))

    # A token-scoring loop avoids N-by-K spatial intermediates.
    target_family_v = node("gmax", [target_family])
    target_rank_v = node("gmax", [target_rank])
    token_columns = vcat(
        target_cv, target_ev, target_family_v, target_rank_v)
    reorder = np.zeros((4 * K, 4 * K), np.float32)
    for i in range(K):
        for j in range(4):
            reorder[4 * i + j, j * K + i] = 1
    token_queue = linear(token_columns, reorder)

    score_states = [
        {"id": "gqueue", "init": token_queue},
        {"id": "gscores", "zeros": [4, 64, 64]},
        {"id": "gneeded", "zeros": [N, 64, 64]},
    ]
    shapes["gqueue"] = (4 * K, False)
    shapes["gscores"] = (4, True)
    shapes["gneeded"] = (N, True)

    score_body = []
    current = score_body
    tc, te, tf, tr = [sl("gqueue", i) for i in range(4)]
    ntf = sub(constant(1), tf)

    def token_blend(a, b):
        return add(mul(a, ntf), mul(b, tf))

    token_on = token_blend(on_rods[0], on_rods[1])
    token_ball_rank = token_blend(ranks[0], ranks[1])
    collected_count = token_blend(counts[0], counts[1])
    desired_count = token_blend(count0, count1)
    prefix_difference = sub(token_ball_rank, tr)
    suffix_difference = add(
        prefix_difference, sub(desired_count, collected_count))
    assignment = mul(
        mul(equal_zero(cat(prefix_difference, suffix_difference)),
            cat(token_on, token_on)), te)
    same_colour = equal_zero(sub(ball_colour, tc))
    same_twice = cat(same_colour, same_colour)
    good = mul(assignment, same_twice)

    reward = np.asarray([10] * N + [8] * N, np.float32)
    penalty = np.asarray([7] * N + [5] * N, np.float32)
    token_weight = linear(
        same_twice, np.diag(reward + penalty), -penalty)
    reduction = np.zeros((2, 2 * N), np.float32)
    reduction[0, :N] = 1
    reduction[1, N:] = 1
    token_scores = linear(mul(assignment, token_weight), reduction)
    scores_next = add(
        "gscores", cat(mul(token_scores, ntf), mul(token_scores, tf)))

    unmatched = sub(one, positive(total(good)))
    needed_next = maximum(
        "gneeded", mul(mul(same_colour, te), unmatched))
    queue_shift = np.zeros((4 * K, 4 * K), np.float32)
    queue_shift[:-4, 4:] = np.eye(4 * K - 4, dtype=np.float32)
    queue_next = linear("gqueue", queue_shift)
    current = graph
    goal_loop = finish_loop("Goal", score_states, score_body, {
        "gqueue": queue_next,
        "gscores": scores_next,
        "gneeded": needed_next,
    })
    goal_loop["T"] = K

    gs = "Goal.gscores"
    score = add(
        maximum(sl(gs, 0), sl(gs, 1)),
        maximum(sl(gs, 2), sl(gs, 3)))
    score = sub(sub(score, "Plan.pcost"), scale(add(p[2], p[5]), .015))

    not_selected = sub(one, selected)
    active_q = add(
        mul(transverse[0], not_selected), mul(transverse[1], selected))
    active_t = add(
        mul(projections[0], not_selected), mul(projections[1], selected))
    active_length = add(mul(p[2], not_selected), mul(p[5], selected))
    active_on = add(
        mul(on_rods[0], not_selected), mul(on_rods[1], selected))
    available_ball = mul(
        mul("Goal.gneeded", ball_exists), sub(one, active_on))
    distance = add(
        absolute(active_q),
        add(relu(sub(active_t, affine(active_length, b=-1))),
            relu(scale(active_t, -1))))
    merit = sub(scale(available_ball, 30), distance)
    best_merit = sl(merit, 0)
    for i in range(1, N):
        best_merit = maximum(best_merit, sl(merit, i))
    score = add(score, scale(relu(best_merit), .1))

    values = node("gmax", [
        add(score, affine(first_choices, 10000, -10000))])
    out = np.zeros((8, 5), np.float32)
    bias = np.full(8, -1000000, np.float32)
    for action, choice in [(1, 0), (2, 1), (3, 2), (4, 3), (6, 4)]:
        out[action, choice] = 1
        bias[action] = 0
    raw_logits = linear(values, out, bias)

    fingerprint = vcat(
        initial_pose_vector, h1[4], total(ball_xv), total(ball_yv))
    unchanged = relu(affine(total(absolute(sub(fingerprint, "mem"))),
                            -1000, 1))
    stalled_penalty = scale(mul("aprev", unchanged), -10000)
    tie_matrix = np.zeros((8, 8), np.float32)
    for action, inverse in [(1, 2), (2, 1), (3, 4), (4, 3)]:
        tie_matrix[inverse, action] = -.015
        tie_matrix[action, action] = .002
    tie_matrix[6, 6] = -.02
    tie_bias = linear("aprev", tie_matrix)
    click_w = np.zeros((8, 1), np.float32)
    click_b = np.zeros(8, np.float32)
    click_w[6, 0], click_b[6] = 100000, -100000
    click_guard = linear(click_possible, click_w, click_b)
    logits_sum = add(
        add(raw_logits, stalled_penalty), add(tie_bias, click_guard))
    linear(logits_sum, np.eye(8, dtype=np.float32), name="logits")
    linear(fingerprint, np.eye(M, dtype=np.float32), name="mem_next")
    other_head = add(
        mul(heads[0], selected0),
        mul(heads[1], sub(one, selected0)))
    linear(other_head, [[1]], name="click")

    # Retain only dependencies of the three output tensors.
    top = {nd["id"]: nd for nd in graph}
    needed = set()
    kept_bodies = {}
    input_names = {"x0", "coords", "mem", "aprev", "avail"}

    def visit(identifier):
        if identifier in input_names or identifier in needed:
            return
        if "." in identifier:
            visit(identifier.split(".", 1)[0])
            return
        nd = top[identifier]
        needed.add(identifier)
        if nd["op"] != "loop":
            for x in nd["in"]:
                visit(x)
            return
        local = {b["id"]: b for b in nd["body"]}
        state_names = {s["id"] for s in nd["state"]}
        local_needed = set()

        def local_visit(x):
            if x in state_names:
                return
            if x not in local:
                visit(x)
                return
            if x in local_needed:
                return
            local_needed.add(x)
            for y in local[x]["in"]:
                local_visit(y)

        for x in nd["next"].values():
            local_visit(x)
        for s in nd["state"]:
            if "init" in s:
                visit(s["init"])
        kept_bodies[identifier] = [
            b for b in nd["body"] if b["id"] in local_needed]

    for output in ("logits", "click", "mem_next"):
        visit(output)
    result = []
    for nd in graph:
        if nd["id"] in needed:
            copied = dict(nd)
            if copied["op"] == "loop":
                copied["body"] = kept_bodies[copied["id"]]
            result.append(copied)

    used = set()

    def gather(nodes):
        for nd in nodes:
            for field in ("w", "b"):
                if field in nd:
                    used.add(nd[field])
            if nd["op"] == "loop":
                gather(nd["body"])

    gather(result)
    final_params = {k: params[k] for k in sorted(used)}
    entries = sum(a.size for a in final_params.values())

    def account():
        channels, macs, count, evaluations = 19, 0, 0, 0

        def walk(nodes, multiplier=1):
            nonlocal channels, macs, count, evaluations
            for nd in nodes:
                count += 1
                evaluations += multiplier
                if nd["op"] == "loop":
                    # Conservatively count carried and final spatial states.
                    channels += 2 * sum(
                        shapes[s["id"]][0] for s in nd["state"]
                        if shapes[s["id"]][1])
                    walk(nd["body"], multiplier * nd["T"])
                    continue
                c, spatial = shapes[nd["id"]]
                if spatial:
                    channels += c
                op = nd["op"]
                if op == "conv":
                    cost = int(np.prod(params[nd["w"]].shape)) * 4096
                elif op == "dense":
                    cost = int(np.prod(params[nd["w"]].shape))
                elif op in ("add", "sub", "mul", "max2", "min2"):
                    cost = c * 4096
                elif op in (
                    "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"
                ):
                    cost = shapes[nd["in"][0]][0] * 4096
                elif op in ("vadd", "vmul"):
                    cost = c
                else:
                    cost = 0
                macs += multiplier * cost

        walk(result)
        return channels, macs, count, evaluations, entries

    retained_plan = next(nd for nd in result if nd["id"] == "Plan")
    usage = account()
    while (usage[1] > 1500000000 or usage[3] > 10000
           ) and retained_plan["T"] > 1:
        retained_plan["T"] -= 1
        usage = account()
    return {
        "graph": result, "memory_size": M, "params": final_params,
    }, usage


def _fallback():
    # Small, analytically wired reactive fallback, always within the budget.
    # It moves toward the mean coloured-ball row and alternates extension
    # with retraction when aligned. It is not a support-level replay.
    p = {}
    g = []

    def conv(name, source, weights, bias=None):
        p[name + "_w"] = np.asarray(weights, np.float32)
        nd = {"id": name, "op": "conv", "in": [source], "w": name + "_w"}
        if bias is not None:
            p[name + "_b"] = np.asarray(bias, np.float32)
            nd["b"] = name + "_b"
        g.append(nd)

    def dense(name, source, weights, bias=None):
        p[name + "_w"] = np.asarray(weights, np.float32)
        nd = {"id": name, "op": "dense", "in": [source], "w": name + "_w"}
        if bias is not None:
            p[name + "_b"] = np.asarray(bias, np.float32)
            nd["b"] = name + "_b"
        g.append(nd)

    def op(name, operation, inputs, **fields):
        nd = {"id": name, "op": operation, "in": inputs}
        nd.update(fields)
        g.append(nd)

    ball_w = np.zeros((1, 16, 5, 5), np.float32)
    for c in [7, 8, 9, 10, 12, 13, 14]:
        for dy, dx in [(-1, -1), (-1, 2), (2, -1), (2, 2)]:
            ball_w[0, c, dy + 2, dx + 2] = 1
    conv("fb", "x0", ball_w, [-3])
    op("fbr", "relu", ["fb"])
    conv("fu", "coords",
         np.asarray([[-64, 0, 52.5]], np.float32)[:, :, None, None])
    op("fur", "relu", ["fu"])
    op("fball", "mul", ["fbr", "fur"])
    op("frow", "slice", ["coords"], start=0, stop=1)
    op("fweighted", "mul", ["fball", "frow"])
    op("fsum", "gmean", ["fweighted"])
    op("fcount", "gmean", ["fball"])

    head_w = np.zeros((1, 16, 7, 7), np.float32)
    for dy, dx in [
        (-2, -2), (-2, 3), (3, -2), (3, 3), (0, 0), (1, 1)
    ]:
        head_w[0, 6, dy + 3, dx + 3] = 1
    conv("fh", "x0", head_w, [-5])
    op("fhr", "relu", ["fh"])
    op("fhead", "mul", ["fhr", "fur"])
    op("fhweighted", "mul", ["fhead", "frow"])
    op("fhsum", "gmean", ["fhweighted"])
    op("fhcount", "gmean", ["fhead"])
    op("fterm1", "vmul", ["fsum", "fhcount"])
    op("fterm2", "vmul", ["fhsum", "fcount"])
    op("ffeatures", "vconcat", ["fterm1", "fterm2", "aprev"])

    weights = np.zeros((8, 10), np.float32)
    bias = np.full(8, -1000, np.float32)
    weights[1, 0], weights[1, 1] = -1e8, 1e8
    weights[2, 0], weights[2, 1] = 1e8, -1e8
    bias[1], bias[2] = 0, 0
    bias[3], bias[4] = -.1, .1
    weights[3, 6] = .4
    weights[4, 6] = -.4
    dense("logits", "ffeatures", weights, bias)
    conv("click", "fhead", np.ones((1, 1, 1, 1), np.float32))
    return {"graph": g, "memory_size": 0, "params": p}
