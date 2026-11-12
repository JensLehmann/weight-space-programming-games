import numpy as np


def build():
    class Graph:
        def __init__(self):
            self.top = []
            self.nodes = self.top
            self.params = {}
            self.param_cache = {}
            self.constants = {}
            self.shapes = {
                "x0": (16, 64, 64),
                "coords": (3, 64, 64),
                "mem": (25,),
                "aprev": (8,),
                "avail": (8,),
            }
            self.macs = {}
            self.serial = 0

        def fresh(self, prefix="n"):
            self.serial += 1
            return prefix + str(self.serial)

        def param(self, value):
            a = np.asarray(value, dtype=np.float32)
            key = (a.shape, a.tobytes())
            if key not in self.param_cache:
                name = "p" + str(len(self.params))
                self.param_cache[key] = name
                self.params[name] = a
            return self.param_cache[key]

        def emit(self, op, inputs, name=None, **kw):
            name = name or self.fresh()
            nd = {"id": name, "op": op, "in": list(inputs)}
            nd.update(kw)
            ss = [self.shapes[x] for x in inputs]
            mac = 0

            if op == "conv":
                w = self.params[kw["w"]]
                shape = (w.shape[0], 64, 64)
                mac = int(w.size) * 4096
            elif op == "dense":
                w = self.params[kw["w"]]
                shape = (w.shape[0],)
                mac = int(w.size)
            elif op in ("slice", "vslice"):
                shape = (kw["stop"] - kw["start"],) + ss[0][1:]
            elif op in ("concat", "vconcat"):
                shape = (sum(s[0] for s in ss),) + ss[0][1:]
            elif op in ("gmax", "gmean"):
                shape = (ss[0][0],)
                mac = ss[0][0] * 4096
            elif op in ("add", "sub", "mul", "max2", "min2"):
                shape = (max(s[0] for s in ss), 64, 64)
                mac = shape[0] * 4096
            elif op in ("vadd", "vmul"):
                shape = (max(s[0] for s in ss),)
                mac = shape[0]
            else:
                shape = ss[0]

            self.shapes[name] = shape
            self.macs[name] = mac
            self.nodes.append(nd)
            return name

        def conv(self, x, w, bias=None, name=None):
            kw = {"w": self.param(w)}
            if bias is not None:
                kw["b"] = self.param(bias)
            return self.emit("conv", [x], name=name, **kw)

        def linear(self, x, w, bias=None, name=None):
            kw = {"w": self.param(w)}
            if bias is not None:
                kw["b"] = self.param(bias)
            return self.emit("dense", [x], name=name, **kw)

        def constant(self, value):
            a = np.asarray(value, np.float32).reshape(-1)
            key = (a.shape, a.tobytes())
            if key not in self.constants:
                saved = self.nodes
                self.nodes = self.top
                z = self.linear(
                    "avail", np.zeros((len(a), 8), np.float32), a
                )
                self.nodes = saved
                self.constants[key] = z
            return self.constants[key]

        def start_loop(self, initial):
            saved = self.nodes
            body = []
            self.nodes = body
            states, specs = {}, []
            for key, init in initial.items():
                sid = self.fresh("s")
                states[key] = sid
                self.shapes[sid] = self.shapes[init]
                specs.append({"id": sid, "init": init})
            return saved, body, states, specs

        def finish_loop(self, ctx, updates, T):
            saved, body, states, specs = ctx
            self.nodes = saved
            lid = self.fresh("L")
            nd = {
                "id": lid,
                "op": "loop",
                "T": T,
                "state": specs,
                "body": body,
                "next": {states[k]: v for k, v in updates.items()},
            }
            self.nodes.append(nd)
            result = {}
            for key, sid in states.items():
                out = lid + "." + sid
                self.shapes[out] = self.shapes[sid]
                result[key] = out
            return result, nd

    g = Graph()
    P, A = 8, 4
    N = P * A

    def c(x):
        return g.constant(x)

    def op(kind, *xs):
        return g.emit(kind, xs)

    def sl(x, start, stop):
        kind = "slice" if len(g.shapes[x]) == 3 else "vslice"
        return g.emit(kind, [x], start=start, stop=stop)

    def cat(xs):
        kind = "concat" if len(g.shapes[xs[0]]) == 3 else "vconcat"
        return g.emit(kind, xs)

    def va(a, b):
        return op("vadd", a, b)

    def vm(a, b):
        return op("vmul", a, b)

    def vs(a, scale):
        return vm(a, c(scale))

    def vn(a):
        return vs(a, -1)

    def vd(a, b):
        return va(a, vn(b))

    def vr(a):
        return op("vrelu", a)

    def vab(a):
        return va(vr(a), vr(vn(a)))

    def vc(a):
        return vd(vr(a), vr(va(a, c(-1))))

    def vmax(a, b):
        return va(b, vr(vd(a, b)))

    def blend(a, b, choose_b):
        return va(a, vm(vd(b, a), choose_b))

    def ss(a, scale):
        return op("mul", a, c(scale))

    def sc(a):
        return op("sub", op("relu", a),
                  op("relu", op("sub", a, c(1))))

    def spatial_abs(a):
        return op("abs", a)

    def vmax_all(x):
        parts = [sl(x, i, i + 1) for i in range(g.shapes[x][0])]
        while len(parts) > 1:
            parts = [
                vmax(parts[i], parts[i + 1])
                if i + 1 < len(parts) else parts[i]
                for i in range(0, len(parts), 2)
            ]
        return parts[0]

    def channel_sum(x, weights=None):
        n = g.shapes[x][0]
        if weights is None:
            weights = np.ones(n, np.float32)
        return g.conv(x, np.asarray(weights, np.float32).reshape(1, n, 1, 1))

    def coord(channel):
        w = np.zeros((1, 3, 1, 1), np.float32)
        w[0, channel, 0, 0] = 63
        return g.conv("coords", w)

    row, col = coord(0), coord(1)
    one_sp = g.conv("coords", np.array([0, 0, 1], np.float32).reshape(1, 3, 1, 1))

    def point_mask(y, x):
        ry = op("relu", op("sub", c(1), spatial_abs(op("sub", row, c(y)))))
        rx = op("relu", op("sub", c(1), spatial_abs(op("sub", col, c(x)))))
        return op("mul", ry, rx)

    bottom = sc(op("sub", row, c(31)))
    right = sc(op("sub", col, c(31)))
    top = op("sub", one_sp, bottom)
    left = op("sub", one_sp, right)
    quadrants = [
        op("mul", top, left),
        op("mul", top, right),
        op("mul", bottom, left),
        op("mul", bottom, right),
    ]

    # Infer the four rendered wall/padding palette regions.
    bg_parts = []
    for (y, x), q in zip(
        [(1, 0), (1, 63), (62, 0), (62, 63)], quadrants
    ):
        sample = op("gmax", op("mul", "x0", point_mask(y, x)))
        bg_parts.append(op("mul", q, sample))
    bg = op("add", op("add", bg_parts[0], bg_parts[1]),
            op("add", bg_parts[2], bg_parts[3]))
    background_pixels = op("mul", "x0", bg)
    wall = channel_sum(background_pixels)
    objects = op("sub", "x0", background_pixels)

    avatar = channel_sum(objects, [
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
    ])
    gray = op("gmax", sl(objects, 1, 2))
    mode = vc(gray)

    box9 = sl(objects, 9, 10)
    box11 = sl(objects, 11, 12)
    boxes = op("add", box9, box11)
    dilation3 = np.ones((1, 1, 3, 3), np.float32)
    boxes_full = sc(g.conv(boxes, dilation3))
    selected_full = sc(g.conv(box11, dilation3))

    # Largest fully occupied avatar square estimates the camera's integer scale.
    sw = np.zeros((6, 1, 7, 7), np.float32)
    sb = np.zeros(6, np.float32)
    for i, size in enumerate(range(2, 8)):
        sw[i, 0, :size, :size] = 1
        sb[i] = 1 - size * size
    sizes = op("gmax", op("relu", g.conv(avatar, sw, sb)))
    scale = g.linear(sizes, np.ones((1, 6), np.float32), [1])
    half = vs(va(scale, c(-1)), 0.5)

    # Detect avatar top-left corners.
    cw = np.zeros((1, 1, 3, 3), np.float32)
    cw[0, 0, 1, 1] = 1
    cw[0, 0, 0, 1] = -1
    cw[0, 0, 1, 0] = -1
    corners = op("relu", g.conv(avatar, cw))

    # Distinguish long colored barriers from single-cell pressure switches.
    colors = cat([sl(objects, i, i + 1) for i in (12, 14, 15)])
    rw = np.zeros((6, 3, 7, 7), np.float32)
    dw = np.zeros((3, 6, 7, 7), np.float32)
    for j in range(3):
        rw[2*j, j, 3, [0, 3, 6]] = 1
        rw[2*j+1, j, [0, 3, 6], 3] = 1
        dw[j, 2*j, 3, :] = 1
        dw[j, 2*j+1, :, 3] = 1
    long_runs = op("relu", g.conv(colors, rw, [-2] * 6))
    gates = op("mul", colors, sc(g.conv(long_runs, dw)))
    pads = op("sub", colors, gates)
    gate_sum = channel_sum(gates)

    unknown_weights = np.zeros(16, np.float32)
    unknown_weights[[0, 2, 3, 4, 6, 7, 8, 13]] = 1
    unknown = sc(g.conv(channel_sum(objects, unknown_weights), dilation3))
    hard_wall = sc(op("add", wall, unknown))
    collision_map = sc(op("add", hard_wall, boxes_full))

    # Memory: y[4], x[4], sy[4], sx[4], alive[4],
    # initialized, random state, selected-box age, cooldown, previous mode.
    old_y = sl("mem", 0, 4)
    old_x = sl("mem", 4, 8)
    old_sy = sl("mem", 8, 12)
    old_sx = sl("mem", 12, 16)
    old_alive = sl("mem", 16, 20)
    initialized = sl("mem", 20, 21)
    random_memory = sl("mem", 21, 22)
    box_age = sl("mem", 22, 23)
    cooldown = sl("mem", 23, 24)
    previous_mode = sl("mem", 24, 25)

    rank = op("add", ss(row, 64.0 / 8192), ss(col, 1.0 / 8192))
    remaining = corners
    ys, xs, presences, peaks = [], [], [], []
    for i in range(4):
        py = vd(sl(old_y, i, i + 1), half)
        px = vd(sl(old_x, i, i + 1), half)
        distance = op(
            "add",
            spatial_abs(op("sub", row, py)),
            spatial_abs(op("sub", col, px)),
        )
        score = op("sub", op("sub", ss(remaining, 1024), distance), rank)
        best = op("gmax", score)
        present = vc(vs(va(best, c(-500)), 0.01))
        peak = op("relu", op(
            "sub", c(1),
            ss(spatial_abs(op("sub", score, best)), 32768)
        ))
        peak = op("mul", op("mul", peak, remaining), present)
        yy = va(vs(op("gmean", op("mul", peak, row)), 4096), half)
        xx = va(vs(op("gmean", op("mul", peak, col)), 4096), half)
        ys.append(yy)
        xs.append(xx)
        presences.append(present)
        peaks.append(peak)
        initial_removal = op("mul", peak, vd(c(1), initialized))
        remaining = op("sub", remaining, initial_removal)

    y = cat(ys)
    x = cat(xs)
    initial_alive = cat(presences)
    alive = blend(initial_alive, old_alive, initialized)

    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    da4 = np.zeros((6, 4), np.float32)
    db4 = np.zeros((6, 4), np.float32)
    inc4 = np.zeros((4, 6), np.float32)
    for k, (i, j) in enumerate(pairs):
        da4[k, i] = db4[k, j] = 1
        inc4[i, k] = inc4[j, k] = 1
    diff4 = da4 - db4
    sep = va(vab(g.linear(y, diff4)), vab(g.linear(x, diff4)))
    duplicate = vc(va(c(1), vs(sep, -4)))
    duplicate = vm(duplicate, vm(g.linear(alive, da4), g.linear(alive, db4)))
    newly_done = vc(g.linear(duplicate, inc4))
    alive = vm(alive, vd(c(1), newly_done))

    total_initial = g.linear(initial_alive, np.ones((1, 4), np.float32))
    four_piece = vc(va(total_initial, c(-2)))
    default_sx = va(c(-1), vs(vc(vd(c(32), x)), 2))
    upper_sign = va(c(1), vs(vc(vd(c(32), y)), -2))
    default_sy = blend(c([1] * 4), upper_sign, four_piece)
    base_sx = blend(default_sx, old_sx, initialized)
    base_sy = blend(default_sy, old_sy, initialized)

    previous_dx = g.linear("aprev", [[0, 0, 0, -1, 1, 0, 0, 0]])
    previous_dy = g.linear("aprev", [[0, -1, 1, 0, 0, 0, 0, 0]])
    learn_ok = vm(initialized, vd(c(1), previous_mode))

    def infer_sign(current, previous, action_delta, base):
        evidence = vm(vd(current, previous), action_delta)
        pos, neg = vc(evidence), vc(vn(evidence))
        changed = vm(va(pos, neg), learn_ok)
        learned = vd(pos, neg)
        return blend(base, learned, changed)

    sx = infer_sign(x, old_x, previous_dx, base_sx)
    sy = infer_sign(y, old_y, previous_dy, base_sy)

    # Four relaxed geodesic fields: channel i is sourced by all other live pieces.
    source_points = op("mul", cat(peaks), alive)
    source_w = (np.ones((4, 4), np.float32) - np.eye(4, dtype=np.float32))
    source = ss(sc(g.conv(source_points, source_w[:, :, None, None])), 256)
    cost = op("add", c(1), op(
        "add", ss(hard_wall, 1024),
        op("add", ss(boxes_full, 3), gate_sum)
    ))

    shift = np.zeros((16, 4, 3, 3), np.float32)
    for d, (dy, dx) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        for a in range(4):
            shift[4*d+a, a, 1+dy, 1+dx] = 1
    ctx = g.start_loop({"value": source})
    v = ctx[2]["value"]
    shifted = g.conv(v, shift)
    neighbors = [sl(shifted, 4*i, 4*i+4) for i in range(4)]
    nv = op("max2", op("max2", neighbors[0], neighbors[1]),
            op("max2", neighbors[2], neighbors[3]))
    nv = op("max2", source, op("sub", nv, cost))
    fields_out, field_loop = g.finish_loop(ctx, {"value": nv}, 192)
    fields = fields_out["value"]

    # Coordinate-based nearest-cell lookup, expressed only with allowed operations.
    def lookup(map_tensor, yy, xx, repetitions=1):
        ey = op("relu", op(
            "sub", spatial_abs(op("sub", row, yy)), c(0.51)
        ))
        ex = op("relu", op(
            "sub", spatial_abs(op("sub", col, xx)), c(0.51)
        ))
        penalty = ss(op("add", ey, ex), 1000)
        if repetitions != 1:
            penalty = cat([penalty] * repetitions)
        return op("gmax", op("sub", map_tensor, penalty))

    root_field4 = lookup(fields, y, x)

    replicate = np.zeros((N, 4), np.float32)
    sum_actors = np.zeros((P, N), np.float32)
    for a in range(4):
        for p in range(P):
            replicate[a*P+p, a] = 1
            sum_actors[p, a*P+p] = 1

    Y = g.linear(y, replicate)
    X = g.linear(x, replicate)
    SY = g.linear(sy, replicate)
    SX = g.linear(sx, replicate)
    F = g.linear(alive, replicate)
    root_values = g.linear(root_field4, replicate)

    def potential(values, live):
        per_actor = va(vm(values, live), vs(vd(c(1), live), 512))
        return g.linear(per_actor, sum_actors)

    root_score = potential(root_values, F)

    field_repeated = cat([
        sl(fields, a, a + 1) for a in range(4) for _ in range(P)
    ])
    map_parts = [field_repeated, cat([collision_map] * N)]
    map_parts += [cat([sl(gates, k, k + 1)] * N) for k in range(3)]
    map_parts += [cat([sl(pads, k, k + 1)] * N) for k in range(3)]
    rollout_maps = cat(map_parts)

    pa = np.zeros((6*P, N), np.float32)
    pb = np.zeros_like(pa)
    incidence = np.zeros((N, 6*P), np.float32)
    for k, (i, j) in enumerate(pairs):
        for p in range(P):
            z = k*P+p
            pa[z, i*P+p] = pb[z, j*P+p] = 1
            incidence[i*P+p, z] = incidence[j*P+p, z] = 1
    pdiff = pa - pb

    gate_add = np.concatenate([np.eye(N, dtype=np.float32)] * 3, axis=1)
    pad_reduce = np.zeros((3*P, 3*N), np.float32)
    open_repeat = np.zeros((3*N, 3*P), np.float32)
    for k in range(3):
        pad_reduce[k*P:(k+1)*P, k*N:(k+1)*N] = sum_actors
        for a in range(4):
            for p in range(P):
                open_repeat[k*N+a*P+p, k*P+p] = 1

    def lcg(r, length):
        expanded = va(vs(r, 5), c(1))
        w = np.concatenate([np.eye(length, dtype=np.float32)] * 4, axis=0)
        bias = np.concatenate([
            np.full(length, 1 - 1024*k, np.float32) for k in range(1, 5)
        ])
        wraps = vc(g.linear(expanded, w, bias))
        summer = np.concatenate([np.eye(length, dtype=np.float32)] * 4, axis=1)
        return vd(expanded, vs(g.linear(wraps, summer), 1024))

    seed = va(random_memory, c([3, 137, 271, 407, 541, 677, 811, 947]))
    seed = vd(seed, vs(vc(va(seed, c(-1023))), 1024))

    root_dy = c([-1, 1, 0, 0] * 2)
    root_dx = c([0, 0, -1, 1] * 2)
    temperatures = c([0, 0, 0, 0, 2, 6, 12, 24])

    def boundary_penalty(yy, xx):
        return va(
            va(vr(vn(yy)), vr(va(yy, c(-63)))),
            va(vr(vn(xx)), vr(va(xx, c(-63))))
        )

    initial = {
        "y": Y, "x": X, "live": F,
        "values": root_values,
        "pads": c(np.zeros(3*N, np.float32)),
        "open": c(np.zeros(3*P, np.float32)),
        "score": root_score, "best": root_score,
        "random": seed, "time": c(0),
    }
    ctx = g.start_loop(initial)
    s = ctx[2]

    random_next = lcg(s["random"], P)
    qw = np.concatenate([np.eye(P, dtype=np.float32)] * 3, axis=0)
    qb = np.concatenate([
        np.full(P, 1 - threshold, np.float32)
        for threshold in (256, 512, 768)
    ])
    quantized = vc(g.linear(random_next, qw, qb))
    wy = np.concatenate([
        2*np.eye(P, dtype=np.float32),
        -np.eye(P, dtype=np.float32),
        np.zeros((P, P), np.float32)
    ], axis=1)
    wx = np.concatenate([
        np.zeros((P, P), np.float32),
        -np.eye(P, dtype=np.float32),
        2*np.eye(P, dtype=np.float32)
    ], axis=1)
    random_dy = g.linear(quantized, wy, -np.ones(P, np.float32))
    random_dx = g.linear(quantized, wx)
    first = vc(vd(c(1), s["time"]))
    dy = blend(random_dy, root_dy, first)
    dx = blend(random_dx, root_dx, first)
    dy_n = vm(vm(cat([dy] * 4), SY), scale)
    dx_n = vm(vm(cat([dx] * 4), SX), scale)

    proposed_y = va(s["y"], vm(dy_n, s["live"]))
    proposed_x = va(s["x"], vm(dx_n, s["live"]))
    samples = lookup(rollout_maps, proposed_y, proposed_x, 8)
    sampled_value = sl(samples, 0, N)
    sampled_wall = sl(samples, N, 2*N)
    sampled_gate = sl(samples, 2*N, 5*N)
    sampled_pad = sl(samples, 5*N, 8*N)

    closed = vd(c(1), g.linear(s["open"], open_repeat))
    hit_gate = g.linear(vm(sampled_gate, closed), gate_add)
    blocked = vc(va(
        va(sampled_wall, hit_gate),
        boundary_penalty(proposed_y, proposed_x)
    ))
    can_move = vm(vd(c(1), blocked), s["live"])
    ny = blend(s["y"], proposed_y, can_move)
    nx = blend(s["x"], proposed_x, can_move)
    nvalues = blend(s["values"], sampled_value, can_move)
    npads = blend(s["pads"], sampled_pad, cat([can_move] * 3))

    # Exact overlap and horizontal crossing of adjacent pieces.
    new_dx = g.linear(nx, pdiff)
    new_dy = g.linear(ny, pdiff)
    abs_dy = vab(new_dy)
    equal = vc(va(c(1), vs(va(vab(new_dx), abs_dy), -4)))
    old_dx = g.linear(s["x"], pdiff)
    crossing = vm(
        vc(vn(vm(old_dx, new_dx))),
        vc(va(c(1), vs(abs_dy, -4)))
    )
    paired = vc(va(equal, crossing))
    paired = vm(paired, vm(
        g.linear(s["live"], pa), g.linear(s["live"], pb)
    ))
    removed = vc(g.linear(paired, incidence))
    nlive = vm(s["live"], vd(c(1), removed))
    nopen = vc(g.linear(vm(npads, cat([nlive] * 3)), pad_reduce))
    nscore = potential(nvalues, nlive)

    accept = vc(va(c(1), va(vd(nscore, s["score"]), temperatures)))
    accept = vmax(accept, first)
    accept_n = cat([accept] * 4)
    accept_pad = cat([accept_n] * 3)
    accept_open = cat([accept] * 3)

    discounted = vd(nscore, vs(s["time"], 0.2))
    updates = {
        "y": blend(s["y"], ny, accept_n),
        "x": blend(s["x"], nx, accept_n),
        "live": blend(s["live"], nlive, accept_n),
        "values": blend(s["values"], nvalues, accept_n),
        "pads": blend(s["pads"], npads, accept_pad),
        "open": blend(s["open"], nopen, accept_open),
        "score": blend(s["score"], nscore, accept),
        "best": vmax(s["best"], discounted),
        "random": random_next,
        "time": va(s["time"], c(1)),
    }
    result, rollout_loop = g.finish_loop(ctx, updates, 1)

    best_four = vmax(sl(result["best"], 0, 4), sl(result["best"], 4, 8))
    gain_four = vd(best_four, sl(root_score, 0, 4))
    normal_scores = va(vs(gain_four, 0.01), c([0.003, 0.002, 0.001, 0]))
    maximum_gain = vmax_all(gain_four)

    # When blocked by a movable object, select it and move it off the
    # relaxed connection between the active pieces.
    near_avatar = sc(g.conv(avatar, np.ones((1, 1, 7, 7), np.float32)))
    near_box = op("gmax", op("mul", near_avatar, box9))
    stalled = vc(va(c(2), vn(maximum_gain)))
    cool_ok = vd(c(1), vc(cooldown))
    request_box = vm(vm(near_box, stalled), cool_ok)

    selected_y = op("gmax", op("mul", box11, row))
    selected_x = op("gmax", op("mul", box11, col))
    inner_half = vs(va(scale, c(-3)), 0.5)
    selected_y = vd(selected_y, inner_half)
    selected_x = vd(selected_x, inner_half)
    box_y = va(selected_y, vm(scale, c([-1, 1, 0, 0])))
    box_x = va(selected_x, vm(scale, c([0, 0, -1, 1])))

    other_boxes = op("relu", op("sub", boxes_full, selected_full))
    box_collision = sc(op(
        "add",
        op("add", hard_wall, other_boxes),
        op("add", avatar, op("add", gate_sum, channel_sum(pads)))
    ))
    sum_fields = channel_sum(fields)
    box_maps = cat([box_collision] * 4 + [sum_fields] * 4)
    box_samples = lookup(box_maps, box_y, box_x, 2)
    box_hit = vc(va(sl(box_samples, 0, 4), boundary_penalty(box_y, box_x)))
    box_legal = vd(c(1), box_hit)
    box_scores = va(
        va(vs(sl(box_samples, 4, 8), -0.01), vs(box_hit, -1000)),
        vs(sl("aprev", 1, 5), 0.05)
    )
    no_box_move = vd(c(1), vmax_all(box_legal))
    age_stop = vc(va(box_age, c(-2)))
    stop_box = vmax(age_stop, no_box_move)
    click_request = blend(request_box, stop_box, mode)

    move_scores = blend(normal_scores, box_scores, mode)
    out_w = np.zeros((8, 4), np.float32)
    out_w[1:5] = np.eye(4, dtype=np.float32)
    out_bias = np.full(8, -10000, np.float32)
    out_bias[1:5] = 0
    base_logits = g.linear(move_scores, out_w, out_bias)
    click_w = np.zeros((8, 1), np.float32)
    click_w[6, 0] = 20000
    click_logits = g.linear(click_request, click_w)
    g.emit("vadd", [base_logits, click_logits], name="logits")

    nearest_field = op(
        "max2",
        op("max2", sl(fields, 0, 1), sl(fields, 1, 2)),
        op("max2", sl(fields, 2, 3), sl(fields, 3, 4))
    )
    choose_box_map = op("mul", box9, op("add", nearest_field, c(1000)))
    deselect_map = ss(avatar, 2000)
    click_map = op("add",
                   op("mul", choose_box_map, vd(c(1), mode)),
                   op("mul", deselect_map, mode))
    g.emit("add", [click_map, c(0)], name="click")

    next_age = vm(mode, va(box_age, c(1)))
    next_cooldown = blend(vr(va(cooldown, c(-1))), c(8), mode)
    memory_parts = [
        y, x, sy, sx, alive, c(1),
        sl(result["random"], 0, 1),
        next_age, next_cooldown, mode
    ]
    g.emit("vconcat", memory_parts, name="mem_next")

    # Set the rollout depth by exact graph bookkeeping, not by fitting
    # weights or evaluating any game states.
    def accounting(nodes):
        macs = evaluations = node_count = 0
        for nd in nodes:
            node_count += 1
            if nd["op"] == "loop":
                bm, be, bn = accounting(nd["body"])
                macs += nd["T"] * bm
                evaluations += 1 + nd["T"] * be
                node_count += bn
            else:
                macs += g.macs[nd["id"]]
                evaluations += 1
        return macs, evaluations, node_count

    body_mac, body_eval, _ = accounting(rollout_loop["body"])
    total_mac, total_eval, _ = accounting(g.top)
    fixed_mac = total_mac - body_mac
    fixed_eval = total_eval - body_eval
    depth = min(
        64,
        int((1_480_000_000 - fixed_mac) // max(1, body_mac)),
        int((9900 - fixed_eval) // max(1, body_eval)),
    )
    rollout_loop["T"] = max(1, depth)

    return {
        "graph": g.top,
        "memory_size": 25,
        "params": g.params,
    }
