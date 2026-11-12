import numpy as np


def build():
    N, F, K, H = 8, 6, 7, 16
    COLORS = [7, 8, 9, 10, 11, 12, 14]
    M = H + 1
    params, param_cache = {}, {}
    shapes = {
        "x0": (16, 64, 64), "coords": (3, 64, 64),
        "avail": (8,), "aprev": (8,), "mem": (M,),
    }
    graph = []
    destination = graph
    serial = 0

    def name(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def parameter(value):
        a = np.asarray(value, np.float32)
        key = (a.shape, a.tobytes())
        if key not in param_cache:
            p = "p" + str(len(params))
            params[p] = a
            param_cache[key] = p
        return param_cache[key]

    def emit(op, inputs, shape, **fields):
        ident = name()
        nd = {"id": ident, "op": op, "in": list(inputs)}
        nd.update(fields)
        destination.append(nd)
        shapes[ident] = tuple(shape)
        return ident

    def linear(x, w, b=None):
        w = np.asarray(w, np.float32)
        spatial_input = len(shapes[x]) == 3
        if spatial_input:
            if w.ndim == 2:
                w = w[:, :, None, None]
            shape, op = (w.shape[0], 64, 64), "conv"
        else:
            shape, op = (w.shape[0],), "dense"
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(np.broadcast_to(
                np.asarray(b, np.float32), (w.shape[0],)).copy())
        return emit(op, [x], shape, **fields)

    def constant(value):
        v = np.asarray(value, np.float32).reshape(-1)
        return linear("avail", np.zeros((len(v), 8), np.float32), v)

    def unary(op, x):
        if len(shapes[x]) == 1:
            op = {"relu": "vrelu", "sigmoid": "vsigmoid",
                  "tanh": "vtanh"}.get(op, op)
        return emit(op, [x], shapes[x])

    def add(a, b):
        spatial_input = len(shapes[a]) == 3 or len(shapes[b]) == 3
        c = max(shapes[a][0], shapes[b][0])
        return emit("add" if spatial_input else "vadd", [a, b],
                    (c, 64, 64) if spatial_input else (c,))

    def mul(a, b):
        spatial_input = len(shapes[a]) == 3 or len(shapes[b]) == 3
        c = max(shapes[a][0], shapes[b][0])
        return emit("mul" if spatial_input else "vmul", [a, b],
                    (c, 64, 64) if spatial_input else (c,))

    def affine(x, a=1.0, b=0.0):
        if len(shapes[x]) == 3:
            if a == 1:
                return add(x, constant([b]))
            if b == 0:
                return mul(x, constant([a]))
        return linear(x, np.eye(shapes[x][0], dtype=np.float32)*a, b)

    def sub(a, b):
        if len(shapes[a]) == 3 or len(shapes[b]) == 3:
            return emit("sub", [a, b],
                        (max(shapes[a][0], shapes[b][0]), 64, 64))
        return add(a, affine(b, -1))

    def cat(xs):
        spatial_input = len(shapes[xs[0]]) == 3
        c = sum(shapes[x][0] for x in xs)
        return emit("concat" if spatial_input else "vconcat", xs,
                    (c, 64, 64) if spatial_input else (c,))

    def sl(x, start, stop=None):
        stop = start+1 if stop is None else stop
        spatial_input = len(shapes[x]) == 3
        return emit("slice" if spatial_input else "vslice", [x],
                    (stop-start, 64, 64) if spatial_input else (stop-start,),
                    start=start, stop=stop)

    def pool(x, op="gmax"):
        return emit(op, [x], (shapes[x][0],))

    def total(x):
        return linear(x, np.ones((1, shapes[x][0]), np.float32))

    def absolute(x):
        if len(shapes[x]) == 3:
            return unary("abs", x)
        return add(unary("relu", x), unary("relu", affine(x, -1)))

    one, zero = constant([1]), constant([0])

    def cap(x):
        if len(shapes[x]) == 3:
            return emit("min2", [x, one], shapes[x])
        return affine(unary("relu", affine(x, -1, 1)), -1, 1)

    def clip(x):
        return cap(unary("relu", x))

    def minimum(a, b):
        if len(shapes[a]) == 3 or len(shapes[b]) == 3:
            return emit("min2", [a, b],
                        (max(shapes[a][0], shapes[b][0]), 64, 64))
        return sub(a, unary("relu", sub(a, b)))

    def maximum(a, b):
        if len(shapes[a]) == 3 or len(shapes[b]) == 3:
            return emit("max2", [a, b],
                        (max(shapes[a][0], shapes[b][0]), 64, 64))
        return add(a, unary("relu", sub(b, a)))

    def mean_sum(x):
        return affine(pool(x, "gmean"), 4096)

    def begin_loop(initial):
        nonlocal destination
        parent = destination
        body, states, refs = [], [], {}
        for key, init in initial.items():
            sid = name("s")
            states.append({"id": sid, "init": init})
            shapes[sid] = shapes[init]
            refs[key] = sid
        destination = body
        return parent, body, states, refs

    def finish_loop(context, next_values, iterations):
        nonlocal destination
        parent, body, states, refs = context
        destination = parent
        lid = name("L")
        nd = {
            "id": lid, "op": "loop", "T": int(iterations),
            "state": states, "body": body,
            "next": {refs[k]: v for k, v in next_values.items()},
        }
        destination.append(nd)
        result = {}
        for key, sid in refs.items():
            result[key] = lid + "." + sid
            shapes[result[key]] = shapes[sid]
        return result, nd

    X = linear("coords", [[0, 63, 0]])
    Y = linear("coords", [[63, 0, 0]])
    XY = cat([X, Y])
    blank = linear("coords", [[0, 0, 0]])
    rank = linear("coords", [[-4032, -63, 4096]])
    reverse_rank = linear("coords", [[4032, 63, 1]])

    def spatial(v):
        return add(blank, v)

    def pick(mask, ranking=rank):
        score = mul(mask, ranking)
        chosen = mul(mask, unary(
            "relu", affine(absolute(sub(score, pool(score))), -2, 1)))
        return (chosen, pool(mul(chosen, X)),
                pool(mul(chosen, Y)), pool(mask))

    def box(cx, cy, width, height):
        gx = clip(affine(sub(affine(width, .5),
                            absolute(sub(X, cx))), 2))
        gy = clip(affine(sub(affine(height, .5),
                            absolute(sub(Y, cy))), 2))
        return minimum(gx, gy)

    # Control centers: horizontal resize, vertical resize, rotation.
    wd = np.zeros((3*K, 16, 5, 5), np.float32)
    bd = np.zeros(3*K, np.float32)
    for j, color in enumerate(COLORS):
        wd[j, 3, 2, 2] = 1
        for yy, xx in [(1, 0), (1, 4), (3, 0), (3, 4)]:
            wd[j, color, yy, xx] = 1
        wd[j, 4, 2, 1] = wd[j, 4, 2, 3] = 1
        bd[j] = -6

        t = K+j
        wd[t, 3, 2, 2] = 1
        for yy, xx in [(0, 1), (0, 3), (4, 1), (4, 3)]:
            wd[t, color, yy, xx] = 1
        wd[t, 4, 1, 2] = wd[t, 4, 3, 2] = 1
        bd[t] = -6

        t = 2*K+j
        for yy, xx in [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]:
            wd[t, color, yy, xx] = 1
        for yy, xx in [(1, 1), (1, 3), (3, 1), (3, 3)]:
            wd[t, 4, yy, xx] = 1
        for yy, xx in [(0, 2), (4, 2), (2, 0), (2, 4)]:
            wd[t, 2, yy, xx] = 1
        bd[t] = -12

    ui = unary("relu", linear("x0", wd, bd))
    uh, uv, ur = sl(ui, 0, K), sl(ui, K, 2*K), sl(ui, 2*K, 3*K)

    def shift_planes(x, dx, dy):
        c = shapes[x][0]
        w = np.zeros((c, c, 7, 7), np.float32)
        for i in range(c):
            w[i, i, 3-dy, 3-dx] = 1
        return linear(x, w)

    minus_clicks = add(shift_planes(uh, -3, 0), shift_planes(uv, 0, -3))
    plus_clicks = add(shift_planes(uh, 3, 0), shift_planes(uv, 0, 3))
    click_options = cat([minus_clicks, plus_clicks, ur])
    allowed = pool(click_options)
    cumulative = linear(allowed, np.tril(np.ones((21, 21), np.float32)))
    count_actions = total(allowed)

    # Rod bases and directions.
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    wa = np.zeros((4*K, 16, 3, 3), np.float32)
    for j, color in enumerate(COLORS):
        for d, (dx, dy) in enumerate(dirs):
            px, py = -dy, dx
            q = 4*j+d
            for side in [-1, 1]:
                wa[q, 3, 1-dy+side*py, 1-dx+side*px] += 1
                wa[q, color, 1+dy+side*py, 1+dx+side*px] += 1

    anchors = unary("relu", linear("x0", wa, -3))
    wu = np.zeros((1, 16, 3, 3), np.float32)
    wu[0, 2] = wu[0, 4] = 1
    anchors = mul(anchors, unary("relu", linear("x0", -wu, 1)))
    anchor_features = cat([anchors, sl("x0", 1)])
    candidates = total(anchor_features)

    wc = np.zeros((K+1, 16, 1, 1), np.float32)
    for i, color in enumerate(COLORS+[1]):
        wc[i, color, 0, 0] = wc[i, 13, 0, 0] = 1
    color_occupancy = linear("x0", wc)

    feature_matrix = np.zeros((14, 4*K+1), np.float32)
    for j in range(K):
        for d, (dx, dy) in enumerate(dirs):
            q = 4*j+d
            feature_matrix[j, q] = 1
            feature_matrix[8, q], feature_matrix[9, q] = dx, dy
            feature_matrix[10+d, q] = 1
    feature_matrix[7, -1] = 1

    def ray_lengths(px, py, occupancy):
        delta = sub(XY, cat([px, py]))
        along = linear(delta, [[1, 0], [0, 1], [-1, 0], [0, -1]])
        across = linear(delta, [[0, 1], [1, 0], [0, 1], [1, 0]])
        line = unary("relu", affine(absolute(across), -1, 1))
        forward = clip(affine(along, 1, 1))
        bad = mul(mul(line, forward), affine(occupancy, -1, 1))
        distance = affine(pool(mul(bad, affine(along, -1, 128))), -1, 128)
        edge = linear(cat([px, py]),
                      [[-1, 0], [0, -1], [1, 0], [0, 1]],
                      [64, 64, 1, 1])
        return minimum(distance, edge)

    object_keys = ["x", "y", "w", "h", "dx", "dy", "color", "present"]
    zeros_n = constant(np.zeros(N, np.float32))
    initial = {key: zeros_n for key in object_keys}
    initial["remaining"] = candidates
    context = begin_loop(initial)
    s = context[3]
    chosen, px, py, exists = pick(s["remaining"])
    features = linear(mean_sum(mul(anchor_features, chosen)), feature_matrix)
    color_onehot = sl(features, 0, 8)
    dx, dy = sl(features, 8), sl(features, 9)
    direction_onehot = sl(features, 10, 14)
    passive = sl(color_onehot, 7)
    occupancy = total(mul(color_occupancy, color_onehot))
    distances = ray_lengths(px, py, occupancy)
    estimated_length = affine(total(mul(distances, direction_onehot)), 1, 1)
    thresholds = np.arange(1, 22, dtype=np.float32)*3 + 1.5
    steps = cap(unary("relu", linear(
        estimated_length, np.full((21, 1), 2, np.float32), -2*thresholds)))
    length = affine(total(steps), 3, 3)
    extension = affine(length, 1, -3)
    aw = affine(mul(absolute(dx), extension), 1, 3)
    ah = affine(mul(absolute(dy), extension), 1, 3)
    width = add(aw, mul(passive, sub(sl(distances, 0), aw)))
    height = add(ah, mul(passive, sub(sl(distances, 1), ah)))
    ax = add(px, affine(mul(dx, extension), .5))
    ay = add(py, affine(mul(dy, extension), .5))
    cx = add(ax, mul(passive, sub(add(px, affine(width, .5, -.5)), ax)))
    cy = add(ay, mul(passive, sub(add(py, affine(height, .5, -.5)), ay)))
    color = linear(color_onehot, [COLORS+[1]])
    covered = box(cx, cy, width, height)
    values = dict(x=cx, y=cy, w=width, h=height, dx=dx, dy=dy,
                  color=color, present=exists)
    next_values = {
        key: cat([values[key], sl(s[key], 0, N-1)])
        for key in object_keys
    }
    next_values["remaining"] = mul(s["remaining"], affine(covered, -1, 1))
    objects, _ = finish_loop(context, next_values, N)

    # Full-width bottom walls become a boundary, not a rectangle slot.
    wall_pixels = sl("x0", 15)
    row_wall = emit("rowsum", [wall_pixels], (1, 64, 64))
    full_bottom = mul(unary("relu", affine(row_wall, 1, -63)),
                      clip(affine(Y, 1, -31)))
    bottom = affine(pool(mul(full_bottom, affine(Y, -1, 64))), -1, 64)
    bottom_center = affine(bottom, .5, -.5)
    bottom_half = affine(bottom, .5)
    wall_remaining = mul(wall_pixels, affine(full_bottom, -1, 1))

    # Fixed wall rectangle extraction.
    zeros_f = constant(np.zeros(F, np.float32))
    wall_keys = ["x", "y", "w", "h", "present"]
    initial = {key: zeros_f for key in wall_keys}
    initial["remaining"] = wall_remaining
    context = begin_loop(initial)
    s = context[3]
    chosen, px, py, exists = pick(s["remaining"])
    rx, ry = sub(X, px), sub(Y, py)
    on_row = unary("relu", affine(absolute(ry), -1, 1))
    right = clip(affine(rx, 1, 1))
    missing = affine(s["remaining"], -1, 1)
    bad = mul(mul(on_row, right), missing)
    width = minimum(
        affine(pool(mul(bad, affine(rx, -1, 128))), -1, 128),
        affine(px, -1, 64))
    cx = add(px, affine(width, .5, -.5))
    inside_x = clip(affine(sub(affine(width, .5),
                              absolute(sub(X, cx))), 2))
    bad_row = emit("rowmax", [mul(missing, inside_x)], (1, 64, 64))
    bad = mul(bad_row, clip(affine(ry, 1, 1)))
    height = minimum(
        affine(pool(mul(bad, affine(ry, -1, 128))), -1, 128),
        affine(py, -1, 64))
    cy = add(py, affine(height, .5, -.5))
    covered = box(cx, cy, width, height)
    values = dict(x=cx, y=cy, w=width, h=height, present=exists)
    next_values = {
        key: cat([values[key], sl(s[key], 0, F-1)])
        for key in wall_keys
    }
    next_values["remaining"] = mul(s["remaining"], affine(covered, -1, 1))
    walls, _ = finish_loop(context, next_values, F)

    # Purple dots and crosses.
    wm = np.zeros((2, 16, 3, 3), np.float32)
    wm[0, 13, 1, 1] = 1
    for yy, xx in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        wm[0, 13, yy, xx] = -1
    for yy, xx in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        wm[1, 13, yy, xx] = 1
    purple = unary("relu", linear("x0", wm, [0, -3]))

    def two_points(mask):
        first, x1, y1, p1 = pick(mask)
        last, x2, y2, _ = pick(mask, reverse_rank)
        p2 = clip(affine(mean_sum(mask), 1, -1))
        return (cat([first, mul(last, p2)]), cat([x1, x2]),
                cat([y1, y2]), cat([p1, p2]))

    marker_masks, marker_x, marker_y, marker_present = two_points(sl(purple, 0))
    goal_masks, goal_x, goal_y, _ = two_points(sl(purple, 1))

    # Obstacle-aware distance fields on the three-pixel lattice.
    wf = np.zeros((1, 16, 3, 3), np.float32)
    wf[0, 15] = -1
    free = unary("relu", linear("x0", wf, 1))
    wb = np.zeros((1, 3, 3, 3), np.float32)
    wb[0, 2] = 1
    free = mul(free, unary("relu", linear("coords", wb, -8)))
    seeds = affine(goal_masks, 96)
    wshift = np.zeros((8, 2, 7, 7), np.float32)
    for d, (ddx, ddy) in enumerate(dirs):
        for c in range(2):
            wshift[2*d+c, c, 3+3*ddy, 3+3*ddx] = 1

    context = begin_loop({"field": seeds})
    field = context[3]["field"]
    shifted = linear(field, wshift)
    parts = [sl(shifted, 2*i, 2*i+2) for i in range(4)]
    neighbor = maximum(maximum(parts[0], parts[1]),
                       maximum(parts[2], parts[3]))
    result, _ = finish_loop(context, {
        "field": mul(maximum(field, affine(neighbor, 1, -1)), free)
    }, 64)
    field = result["field"]

    at_first = pool(mul(field, sl(marker_masks, 0)))
    at_second = pool(mul(field, sl(marker_masks, 1)))
    advantage = linear(cat([at_first, at_second]), [[1, -1, -1, 1]])
    direct = clip(affine(advantage, 100, 1))
    swap2 = np.array([[0, 1], [1, 0]], np.float32)
    indirect = affine(direct, -1, 1)
    assigned_field = add(mul(field, direct), mul(linear(field, swap2), indirect))
    assigned_x = add(mul(goal_x, direct), mul(linear(goal_x, swap2), indirect))
    assigned_y = add(mul(goal_y, direct), mul(linear(goal_y, swap2), indirect))

    wt = np.zeros((2, 2, 7, 7), np.float32)
    for c in range(2):
        wt[c, c, 3, 3] = 1
        for ddx, ddy in dirs:
            wt[c, c, 3+3*ddy, 3+3*ddx] = 1
    trace_score = add(assigned_field,
                      linear("coords", [[.4032, .0063, 0]]))
    context = begin_loop({"pointer": marker_masks})
    trace_candidates = linear(context[3]["pointer"], wt)
    scored = mul(trace_candidates, trace_score)
    result, _ = finish_loop(context, {
        "pointer": mul(trace_candidates, unary("relu", affine(
            absolute(sub(scored, pool(scored))), -100000, 1)))
    }, 5)
    waypoint_x = pool(mul(result["pointer"], X))
    waypoint_y = pool(mul(result["pointer"], Y))

    # Geometric attachment graph.
    ox, oy = objects["x"], objects["y"]
    ow, oh = objects["w"], objects["h"]
    odx, ody = objects["dx"], objects["dy"]
    presence = objects["present"]
    passive = mul(presence, unary("relu", affine(
        absolute(affine(objects["color"], 1, -1)), -1, 1)))
    active = sub(presence, passive)
    half_extension = affine(add(ow, oh), .5, -3)
    bx = sub(ox, mul(odx, half_extension))
    by = sub(oy, mul(ody, half_extension))
    tx = add(ox, mul(odx, affine(half_extension, 1, 3)))
    ty = add(oy, mul(ody, affine(half_extension, 1, 3)))

    repeat_parent = np.zeros((N*N, N), np.float32)
    repeat_child = np.zeros((N*N, N), np.float32)
    reduce_child = np.zeros((N, N*N), np.float32)
    for i in range(N):
        for j in range(N):
            q = i*N+j
            repeat_parent[q, i] = repeat_child[q, j] = reduce_child[j, q] = 1

    ex = absolute(sub(linear(tx, repeat_parent), linear(bx, repeat_child)))
    ey = absolute(sub(linear(ty, repeat_parent), linear(by, repeat_child)))
    tolx = linear(mul(passive, affine(ow, .5, -1.5)), repeat_child)
    toly = linear(mul(passive, affine(oh, .5, -1.5)), repeat_child)
    error = add(unary("relu", sub(ex, tolx)), unary("relu", sub(ey, toly)))
    adjacency = unary("relu", affine(error, -10, 1))
    pair_presence = mul(linear(active, repeat_parent),
                        linear(presence, repeat_child))
    adjacency = mul(mul(adjacency, pair_presence),
                    constant((1-np.eye(N, dtype=np.float32)).reshape(-1)))
    parent_exists = clip(linear(adjacency, reduce_child))
    parent_dx = linear(mul(linear(odx, repeat_parent), adjacency), reduce_child)
    parent_dy = linear(mul(linear(ody, repeat_parent), adjacency), reduce_child)
    relative_cross = sub(mul(odx, parent_dy), mul(ody, parent_dx))
    phase = add(unary("relu", relative_cross),
                affine(unary("relu", affine(relative_cross, -1)), 2))

    rep_obj = np.zeros((2*N, N), np.float32)
    rep_mark = np.zeros((2*N, 2), np.float32)
    reduce_mark = np.zeros((2, 2*N), np.float32)
    for i in range(N):
        for j in range(2):
            q = 2*i+j
            rep_obj[q, i] = rep_mark[q, j] = reduce_mark[j, q] = 1

    mxerr = absolute(sub(linear(ox, rep_obj), linear(marker_x, rep_mark)))
    myerr = absolute(sub(linear(oy, rep_obj), linear(marker_y, rep_mark)))
    mxin = clip(affine(sub(linear(ow, .5*rep_obj), mxerr), 2))
    myin = clip(affine(sub(linear(oh, .5*rep_obj), myerr), 2))
    membership = mul(mul(mxin, myin),
                     mul(linear(presence, rep_obj),
                         linear(marker_present, rep_mark)))

    # One shared loop computes all eight descendant rows.
    D = N + 2
    root_init = constant(np.eye(N, dtype=np.float32)[-1])
    shift_root = np.zeros((N, N), np.float32)
    for i in range(N-1):
        shift_root[i, i+1] = 1
    context = begin_loop({
        "root": root_init, "reach": root_init, "clock": zero,
        "table": constant(np.zeros(N*D, np.float32))
    })
    s = context[3]
    propagated = linear(
        mul(linear(s["reach"], repeat_parent), adjacency), reduce_child)
    reach = cap(add(s["root"], propagated))
    marker_reach = cap(linear(
        mul(linear(reach, rep_obj), membership), reduce_mark))
    flush = unary("relu", affine(s["clock"], 1, -3))
    next_root = add(s["root"], mul(
        flush, sub(linear(s["root"], shift_root), s["root"])))
    next_reach = add(reach, mul(flush, sub(next_root, reach)))
    shifted_table = cat([reach, marker_reach, sl(s["table"], 0, (N-1)*D)])
    table = add(s["table"], mul(flush, sub(shifted_table, s["table"])))
    clock = sub(affine(s["clock"], 1, 1), affine(flush, 5))
    result, _ = finish_loop(context, {
        "root": next_root, "reach": next_reach, "clock": clock, "table": table
    }, 5*N)
    reach_rows = [sl(result["table"], i*D, (i+1)*D) for i in range(N)]

    def transmit(value):
        terms = [mul(sl(value, i), reach_rows[i]) for i in range(N)]
        while len(terms) > 1:
            terms = [add(terms[i], terms[i+1])
                     for i in range(0, len(terms), 2)]
        return terms[0]

    # Collision matrices.
    dynamic_pairs = N*(N-1)//2
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    pairs += [(i, N+j) for i in range(N) for j in range(F)]
    difference = np.zeros((len(pairs), N+F), np.float32)
    half_sum = np.zeros_like(difference)
    for q, (i, j) in enumerate(pairs):
        difference[q, i], difference[q, j] = 1, -1
        half_sum[q, i] = half_sum[q, j] = .5
    dyn_reduce = np.zeros((1, len(pairs)), np.float32)
    dyn_reduce[0, :dynamic_pairs] = 1
    wall_reduce = 1-dyn_reduce
    missing_object_width = affine(presence, 1000, -1000)
    fixed_x, fixed_y = spatial(walls["x"]), spatial(walls["y"])
    fixed_w = spatial(add(walls["w"], affine(walls["present"], 1000, -1000)))
    fixed_h = spatial(walls["h"])

    # Recent-configuration signature.
    hash_weights = np.array(
        [[1 + ((i*73 + i*i*19 + 17) % 251) for i in range(6*N)]],
        np.float32)
    current_hash = linear(cat([ox, oy, ow, oh, odx, ody]), hash_weights)
    history = sl("mem", 0, H)

    action_features = np.array(
        [COLORS*3, [-1]*K+[1]*K+[0]*K, [0]*(2*K)+[1]*K], np.float32)
    bin_matrix = -np.eye(21, dtype=np.float32)
    for i in range(1, 21):
        bin_matrix[i, i-1] = 1
    bin_bias = np.zeros(21, np.float32)
    bin_bias[0] = 1
    compact_matrix = action_features @ bin_matrix
    compact_bias = action_features @ bin_bias

    old_seed = sl("mem", H)
    seed_sum = affine(old_seed, 1, .1234567)
    next_seed = sub(seed_sum, clip(affine(seed_sum, 100000, -99999)))
    initial_random = add(
        linear("coords", [[4032/4096, 63/4096, .5/4096]]), old_seed)
    initial_random = sub(
        initial_random, clip(affine(initial_random, 100000, -99999)))

    def draw_action(random_value, compact=False):
        crossed = clip(affine(
            sub(mul(random_value, count_actions), cumulative), 100000, 1))
        if compact:
            return linear(crossed, compact_matrix, compact_bias)
        return unary("relu", linear(crossed, bin_matrix, bin_bias))

    # Parallel rollouts: one candidate sequence at each spatial position.
    first_action = draw_action(initial_random)
    initial = {
        "x": spatial(cat([ox, marker_x])), "y": spatial(cat([oy, marker_y])),
        "w": spatial(ow), "h": spatial(oh), "dx": spatial(odx), "dy": spatial(ody),
        "phase": spatial(phase), "random": initial_random,
        "action": linear(first_action, action_features),
        "best": add(blank, constant([-100000])),
        "bad_first": blank, "penalty": blank, "first_visit": blank, "time": zero,
    }
    context = begin_loop(initial)
    s = context[3]
    action = s["action"]
    action_color, sign, turn = sl(action, 0), sl(action, 1), sl(action, 2)
    selected = mul(active, unary("relu", affine(
        absolute(sub(action_color, objects["color"])), -1, 1)))
    contractible = clip(affine(add(s["w"], s["h"]), 1/3, -2))
    positive, negative = unary("relu", sign), unary("relu", affine(sign, -1))
    ds = affine(mul(selected, sub(positive, mul(negative, contractible))), 3)
    rotation = mul(selected, turn)
    skip = mul(parent_exists, unary("relu", affine(
        absolute(affine(s["phase"], 1, -1)), -1, 1)))
    a = affine(mul(rotation, affine(skip, 1, 1)), -1)
    b = mul(rotation, affine(skip, -1, 1))
    lh = affine(add(s["w"], s["h"]), .5, -3)
    pivot_x = sub(sl(s["x"], 0, N), mul(s["dx"], lh))
    pivot_y = sub(sl(s["y"], 0, N), mul(s["dy"], lh))
    resize_x, resize_y = mul(s["dx"], ds), mul(s["dy"], ds)
    trans_x = add(affine(add(mul(a, pivot_x), mul(b, pivot_y)), -1), resize_x)
    trans_y = add(sub(mul(b, pivot_x), mul(a, pivot_y)), resize_y)

    A, B = transmit(a), transmit(b)
    TX, TY = transmit(trans_x), transmit(trans_y)
    pad2 = linear(blank, np.zeros((2, 1), np.float32))
    px = sub(add(add(s["x"], mul(A, s["x"])), add(mul(B, s["y"]), TX)),
             cat([affine(resize_x, .5), pad2]))
    py = sub(add(add(s["y"], mul(A, s["y"])), sub(TY, mul(B, s["x"]))),
             cat([affine(resize_y, .5), pad2]))
    ao, bo = sl(A, 0, N), sl(B, 0, N)
    pw = add(add(s["w"], mul(bo, sub(s["h"], s["w"]))),
             mul(absolute(s["dx"]), ds))
    ph = add(add(s["h"], mul(bo, sub(s["w"], s["h"]))),
             mul(absolute(s["dy"]), ds))
    pdx = add(add(s["dx"], mul(ao, s["dx"])), mul(bo, s["dy"]))
    pdy = sub(add(s["dy"], mul(ao, s["dy"])), mul(bo, s["dx"]))
    pp = add(s["phase"], mul(rotation, affine(
        unary("relu", affine(s["phase"], 1, -1)), -3, 1)))

    all_x = cat([sl(px, 0, N), fixed_x])
    all_y = cat([sl(py, 0, N), fixed_y])
    all_w = cat([add(pw, missing_object_width), fixed_w])
    all_h = cat([ph, fixed_h])
    overlap_x = sub(linear(all_w, half_sum, -.25),
                    absolute(linear(all_x, difference)))
    overlap_y = sub(linear(all_h, half_sum, -.25),
                    absolute(linear(all_y, difference)))
    overlap = unary("relu", minimum(overlap_x, overlap_y))
    dynamic_penalty = linear(overlap, dyn_reduce)
    fixed_penalty = linear(overlap, wall_reduce)

    outside_x = unary("relu", affine(
        add(absolute(affine(sl(px, 0, N), 1, -31.5)), affine(pw, .5)), 1, -32))
    outside_y = unary("relu", sub(
        add(absolute(sub(sl(py, 0, N), bottom_center)), affine(ph, .5)),
        bottom_half))
    outside = total(mul(add(outside_x, outside_y), presence))
    hard_valid = unary("relu", affine(add(fixed_penalty, outside), -4, 1))
    dynamic_valid = unary("relu", affine(dynamic_penalty, -4, 1))
    changed = total(add(
        add(absolute(sub(pw, s["w"])), absolute(sub(ph, s["h"]))),
        add(absolute(sub(pdx, s["dx"])), absolute(sub(pdy, s["dy"])))))
    hard_valid = mul(hard_valid, clip(changed))
    is_first = unary("relu", affine(s["time"], -1, 1))
    actual_valid = mul(hard_valid, dynamic_valid)
    accept = mul(hard_valid, affine(
        mul(is_first, affine(dynamic_valid, -1, 1)), -1, 1))

    proposed = dict(x=px, y=py, w=pw, h=ph, dx=pdx, dy=pdy, phase=pp)
    accepted = {
        key: add(s[key], mul(accept, sub(value, s[key])))
        for key, value in proposed.items()
    }
    penalty = add(s["penalty"], mul(accept, sub(dynamic_penalty, s["penalty"])))
    bad_first = add(s["bad_first"], mul(is_first, affine(actual_valid, -1, 1)))
    elapsed = add(s["time"], one)
    mx, my = sl(accepted["x"], N, N+2), sl(accepted["y"], N, N+2)
    local_distance = add(absolute(sub(mx, waypoint_x)), absolute(sub(my, waypoint_y)))
    global_distance = add(absolute(sub(mx, assigned_x)), absolute(sub(my, assigned_y)))
    distance = total(mul(add(local_distance, affine(global_distance, .25)),
                         marker_present))
    global_total = total(mul(global_distance, marker_present))

    pose_hash = linear(cat([
        sl(accepted["x"], 0, N), sl(accepted["y"], 0, N),
        accepted["w"], accepted["h"], accepted["dx"], accepted["dy"]
    ]), hash_weights)
    visits = cap(total(unary("relu", affine(
        absolute(sub(pose_hash, history)), -8, 1))))
    first_visit = add(s["first_visit"], mul(is_first, visits))
    cost = add(
        add(affine(penalty, 2/3), affine(bad_first, 1000)),
        add(add(affine(visits, 1.5), affine(first_visit, .5)),
            affine(elapsed, .045)))
    score = sub(affine(distance, -1/3), cost)
    solved = mul(unary("relu", affine(global_total, -1, 1)),
                 unary("relu", affine(penalty, -4, 1)))
    best_score = maximum(s["best"], add(score, affine(solved, 50)))

    z = affine(s["random"], 17, .137)
    cuts = clip(linear(
        z, np.full((17, 1), 100000, np.float32),
        -100000*np.arange(1, 18, dtype=np.float32)+1))
    rng = clip(sub(z, total(cuts)))
    fresh = draw_action(rng, compact=True)
    repeat = mul(clip(affine(rng, -100000, 50000)), actual_valid)
    next_action = add(mul(action, repeat), mul(fresh, affine(repeat, -1, 1)))
    next_values = dict(accepted)
    next_values.update(
        random=rng, action=next_action, best=best_score, penalty=penalty,
        bad_first=bad_first, first_visit=first_visit, time=elapsed)
    rollouts, rollout_node = finish_loop(context, next_values, 32)

    values = rollouts["best"]
    near_best = unary("relu", affine(
        absolute(sub(values, pool(values))), -100000, 1))
    tie_scores = mul(near_best, reverse_rank)
    winner = mul(near_best, unary("relu", affine(
        absolute(sub(tie_scores, pool(tie_scores))), -2, 1)))
    chosen_action = pool(mul(first_action, winner))
    final_click = total(mul(click_options, chosen_action))
    emit("add", [final_click, blank], (1, 64, 64))
    graph[-1]["id"] = "click"
    shapes["click"] = (1, 64, 64)

    logits = np.full(8, -10, np.float32)
    logits[0], logits[6] = -100, 10
    linear("avail", np.zeros((8, 8), np.float32), logits)
    graph[-1]["id"] = "logits"
    shapes["logits"] = (8,)
    cat([current_hash, sl("mem", 0, H-1), next_seed])
    graph[-1]["id"] = "mem_next"
    shapes["mem_next"] = (M,)

    # Static graph simplification and resource accounting.
    protected = {"click", "logits", "mem_next"}

    def prune(nodes, required):
        kept, needed = [], set(required)
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                outputs = [nd["id"]+"."+st["id"] for st in nd["state"]]
                if not any(x in needed for x in outputs):
                    continue
                body, external = prune(nd["body"], set(nd["next"].values()))
                nd["body"] = body
                needed.update(external-{st["id"] for st in nd["state"]})
                for st in nd["state"]:
                    if "init" in st:
                        needed.add(st["init"])
                kept.append(nd)
            elif nd["id"] in needed:
                needed.update(nd["in"])
                kept.append(nd)
        kept.reverse()
        defined = {nd["id"] for nd in kept}
        for nd in kept:
            if nd["op"] == "loop":
                defined.update(nd["id"]+"."+st["id"] for st in nd["state"])
        return kept, needed-defined

    def share(nodes, inherited_alias=None, inherited_memo=None):
        alias, memo = dict(inherited_alias or {}), dict(inherited_memo or {})
        kept = []

        def resolve(x):
            while x in alias:
                x = alias[x]
            return x

        for nd in nodes:
            if nd["op"] == "loop":
                for st in nd["state"]:
                    if "init" in st:
                        st["init"] = resolve(st["init"])
                nd["body"], body_alias = share(nd["body"], alias, memo)
                for key, value in list(nd["next"].items()):
                    while value in body_alias:
                        value = body_alias[value]
                    nd["next"][key] = value
                kept.append(nd)
                continue
            nd["in"] = [resolve(x) for x in nd["in"]]
            key = (nd["op"], tuple(nd["in"]), nd.get("w"), nd.get("b"),
                   nd.get("start"), nd.get("stop"))
            if key in memo and nd["id"] not in protected:
                alias[nd["id"]] = memo[key]
            else:
                memo[key] = nd["id"]
                kept.append(nd)
        return kept, alias

    graph, _ = prune(graph, protected)
    graph, _ = share(graph)
    graph, _ = prune(graph, protected)

    def count_uses(nodes, counts):
        for nd in nodes:
            if nd["op"] == "loop":
                for st in nd["state"]:
                    if "init" in st:
                        counts[st["init"]] = counts.get(st["init"], 0)+1
                for value in nd["next"].values():
                    counts[value] = counts.get(value, 0)+1
                count_uses(nd["body"], counts)
            else:
                for value in nd["in"]:
                    counts[value] = counts.get(value, 0)+1

    uses = {}
    count_uses(graph, uses)

    def fuse(nodes):
        defined = {}
        for nd in nodes:
            if nd["op"] == "loop":
                fuse(nd["body"])
                continue
            if nd["op"] in ("dense", "conv") and len(nd["in"]) == 1:
                parent = defined.get(nd["in"][0])
                if (parent is not None and parent["op"] == nd["op"]
                        and uses.get(parent["id"], 0) == 1
                        and parent["id"] not in protected):
                    w1, w2 = params[parent["w"]], params[nd["w"]]
                    pointwise = (nd["op"] == "dense" or
                                 (w1.shape[-2:] == (1, 1)
                                  and w2.shape[-2:] == (1, 1)))
                    if pointwise:
                        a = w1.reshape(w1.shape[0], -1)
                        b = w2.reshape(w2.shape[0], -1)
                        if b.shape[0]*a.shape[1] <= 1024:
                            w = b @ a
                            bias1 = (params[parent["b"]] if "b" in parent
                                     else np.zeros(a.shape[0], np.float32))
                            bias2 = (params[nd["b"]] if "b" in nd
                                     else np.zeros(b.shape[0], np.float32))
                            bias = b @ bias1 + bias2
                            if nd["op"] == "conv":
                                w = w[:, :, None, None]
                            nd["w"], nd["b"] = parameter(w), parameter(bias)
                            nd["in"] = list(parent["in"])
            defined[nd["id"]] = nd

    fuse(graph)
    graph, _ = prune(graph, protected)
    graph, _ = share(graph)
    graph, _ = prune(graph, protected)

    def statistics(nodes):
        macs = evaluations = node_count = channels = 0
        used = set()
        for nd in nodes:
            node_count += 1
            evaluations += 1
            op = nd["op"]
            if op == "loop":
                bm, be, bn, bc, bu = statistics(nd["body"])
                macs += nd["T"]*bm
                evaluations += nd["T"]*be
                node_count += bn
                channels += bc
                for st in nd["state"]:
                    if len(shapes[st["id"]]) == 3:
                        channels += shapes[st["id"]][0]
                used.update(bu)
                continue
            sh = shapes[nd["id"]]
            if len(sh) == 3:
                channels += sh[0]
            if op in ("conv", "dense"):
                used.add(nd["w"])
                macs += params[nd["w"]].size*(4096 if op == "conv" else 1)
                if "b" in nd:
                    used.add(nd["b"])
            elif op in ("add", "sub", "mul", "max2", "min2"):
                macs += sh[0]*4096
            elif op in ("vadd", "vmul"):
                macs += sh[0]
            elif op in ("rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"):
                macs += int(np.prod(shapes[nd["in"][0]]))
        return macs, evaluations, node_count, channels, used

    rollout_node["T"] = 1
    base_macs, base_evals, _, _, _ = statistics(graph)
    body_macs, body_evals, _, _, _ = statistics(rollout_node["body"])
    horizon_mac = 1 + int((1_470_000_000-base_macs)//max(1, body_macs))
    horizon_eval = 1 + int((9800-base_evals)//max(1, body_evals))
    rollout_node["T"] = max(1, min(32, horizon_mac, horizon_eval))
    _, _, _, _, used = statistics(graph)
    return {
        "graph": graph,
        "memory_size": M,
        "params": {key: value for key, value in params.items() if key in used},
    }
