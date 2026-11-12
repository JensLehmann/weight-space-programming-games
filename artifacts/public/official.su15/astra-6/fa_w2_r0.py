import numpy as np


def build():
    NF, NE, NG = 12, 5, 6
    Q = NF + NE
    GOAL_MEM = 3 * NG
    OBJECT_MEM = 6 * Q
    M = GOAL_MEM + 2 * OBJECT_MEM

    graph = []
    current = [graph]
    params = {}
    param_cache = {}
    constant_cache = {}
    shapes = {
        "x0": (16, True),
        "coords": (3, True),
        "mem": (M, False),
        "aprev": (8, False),
        "avail": (8, False),
    }
    serial = [0]

    def uid(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in param_cache:
            name = "p" + str(len(params))
            param_cache[key] = name
            params[name] = a
        return param_cache[key]

    def emit(op, inputs, name=None, **fields):
        name = name or uid()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        current[0].append(nd)
        if op in ("conv", "dense"):
            c = params[fields["w"]].shape[0]
            spatial = op == "conv"
        elif op in ("concat", "vconcat"):
            c = sum(shapes[x][0] for x in inputs)
            spatial = op == "concat"
        elif op in ("slice", "vslice"):
            c = fields["stop"] - fields["start"]
            spatial = op == "slice"
        elif op in ("gmax", "gmean"):
            c, spatial = shapes[inputs[0]][0], False
        elif op in ("add", "sub", "mul", "max2", "min2",
                    "vadd", "vmul"):
            c = max(shapes[x][0] for x in inputs)
            spatial = any(shapes[x][1] for x in inputs)
        else:
            c, spatial = shapes[inputs[0]]
        shapes[name] = (c, spatial)
        return name

    def linear(x, w, b=None, name=None):
        spatial = shapes[x][1]
        w = np.asarray(w, np.float32)
        if spatial and w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv" if spatial else "dense", [x],
                    name=name, **fields)

    def fixed(a):
        a = np.asarray(a, np.float32).reshape(-1)
        return linear("avail", np.zeros((len(a), 8), np.float32), a)

    def const(value, size=1):
        key = (float(value), int(size))
        if key not in constant_cache:
            constant_cache[key] = fixed(
                np.full(size, value, np.float32))
        return constant_cache[key]

    def cat(*xs):
        return emit("concat" if shapes[xs[0]][1] else "vconcat", xs)

    def sl(x, start, stop):
        return emit("slice" if shapes[x][1] else "vslice", [x],
                    start=int(start), stop=int(stop))

    def add(a, b, name=None):
        spatial = shapes[a][1] or shapes[b][1]
        return emit("add" if spatial else "vadd", [a, b], name=name)

    def mul(a, b):
        spatial = shapes[a][1] or shapes[b][1]
        return emit("mul" if spatial else "vmul", [a, b])

    def affine(x, a=1.0, b=0.0):
        c, spatial = shapes[x]
        if a == 1.0 and b == 0.0:
            return x
        if c <= 24:
            bias = None if b == 0.0 else np.full(c, b, np.float32)
            return linear(x, np.eye(c, dtype=np.float32) * a, bias)
        y = x if a == 1.0 else mul(x, const(a))
        return y if b == 0.0 else add(y, const(b))

    def sub(a, b):
        if shapes[a][1] or shapes[b][1]:
            return emit("sub", [a, b])
        return add(a, affine(b, -1.0))

    def relu(x):
        return emit("relu" if shapes[x][1] else "vrelu", [x])

    def absolute(x):
        if shapes[x][1]:
            return emit("abs", [x])
        return add(relu(x), relu(affine(x, -1.0)))

    def clip01(x):
        if shapes[x][1]:
            return emit("min2", [relu(x), const(1)])
        return affine(relu(affine(relu(x), -1, 1)), -1, 1)

    def total(x, scale=1.0):
        return linear(x, np.full((1, shapes[x][0]), scale, np.float32))

    def eqcode(code, value):
        return relu(affine(absolute(sub(code, value)), -1, 1))

    def mix(old, new, gate):
        return add(old, mul(gate, sub(new, old)))

    def add_many(xs):
        y = xs[0]
        for x in xs[1:]:
            y = add(y, x)
        return y

    def loop(T, initial, body_fn):
        lid = uid("L")
        state_ids = [uid("s") for _ in initial]
        states = []
        for sid, initial_id in zip(state_ids, initial):
            shapes[sid] = shapes[initial_id]
            states.append({"id": sid, "init": initial_id})
        outer = current[0]
        body = []
        current[0] = body
        nxt = body_fn(*state_ids)
        current[0] = outer
        outer.append({
            "id": lid, "op": "loop", "T": int(T),
            "state": states, "body": body,
            "next": dict(zip(state_ids, nxt)),
        })
        results = []
        for sid in state_ids:
            out = lid + "." + sid
            shapes[out] = shapes[sid]
            results.append(out)
        return results

    def shift_matrix(n, k):
        w = np.zeros((n * k, n * k), np.float32)
        if n > 1:
            w[:(n - 1) * k, k:] = np.eye((n - 1) * k)
        return w

    def feature(v, index, n, k=6):
        w = np.zeros((n, n * k), np.float32)
        w[np.arange(n), np.arange(n) * k + index] = 1
        return linear(v, w)

    # Coordinates and the two semantically distinct image regions.
    Y = linear("coords", [[63, 0, 0]])
    X = linear("coords", [[0, 63, 0]])
    lower = clip01(affine(Y, 1, -9))
    upper = affine(clip01(affine(Y, 1, -62)), -1, 1)
    world = mul(lower, upper)
    header = emit("min2", [
        affine(lower, -1, 1),
        clip01(affine(X, 1, -25))
    ])

    colors = [10, 6, 15, 11, 12, 8, 9, 7, 14]
    sizes = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10], np.float32)
    halves = np.floor(sizes / 2)

    # Filled-square top-left detectors.
    wf = np.zeros((9, 16, 5, 5), np.float32)
    bf = np.zeros(9, np.float32)
    for r, color in enumerate(colors):
        side = min(int(sizes[r]), 3)
        positives = [(dy, dx) for dy in range(side)
                     for dx in range(side)]
        negatives = [] if r == 0 else [
            (-1, 0), (0, -1), (1, -1), (-1, 1)
        ]
        for dy, dx in positives:
            wf[r, color, dy + 2, dx + 2] += 1
        for dy, dx in negatives:
            wf[r, color, dy + 2, dx + 2] -= 1
        bf[r] = 1 - len(positives)
    raw_fruit = relu(linear("x0", wf, bf))
    field_fruit = mul(raw_fruit, world)
    header_fruit = mul(raw_fruit, header)

    # Hunter detectors anchored at the game-defined sprite centers.
    patterns = [
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0]],
        [[1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 1, 1, 1, 0]],
    ]
    we = np.zeros((3, 16, 7, 7), np.float32)
    be = np.zeros(3, np.float32)
    for j, (pat, color) in enumerate(zip(patterns, [7, 14, 13])):
        pat = np.asarray(pat)
        for yy in range(4):
            for xx in range(5):
                we[j, color, yy + 1, xx + 1] = (
                    1 if pat[yy, xx] else -1)
        be[j] = 1 - int(pat.sum())
    raw_enemy = relu(linear("x0", we, be))
    field_enemy = mul(raw_enemy, world)
    header_enemy = mul(raw_enemy, header)

    # Goal-circle boundary arcs.
    wc = np.zeros((1, 16), np.float32)
    wc[0, [9, 2]] = 1
    circle_color = linear("x0", wc)
    wa = np.zeros((4, 1, 7, 7), np.float32)
    dirs = []
    positives = [(0, -2), (0, 0), (0, 2), (1, -3), (1, 3)]
    negatives = [(-1, 0), (0, -3), (0, 3)]

    def rotate(dy, dx, times):
        for _ in range(times):
            dy, dx = dx, -dy
        return dy, dx

    for j in range(4):
        dirs.append(rotate(1, 0, j))
        for dy, dx in positives:
            yy, xx = rotate(dy, dx, j)
            wa[j, 0, yy + 3, xx + 3] += 1
        for dy, dx in negatives:
            yy, xx = rotate(dy, dx, j)
            wa[j, 0, yy + 3, xx + 3] -= 1
    arcs = relu(linear(circle_color, wa, np.full(4, -4)))
    w7 = np.zeros((4, 4, 7, 7), np.float32)
    w3 = np.zeros((4, 4, 3, 3), np.float32)
    for j, (dy, dx) in enumerate(dirs):
        w7[j, j, 3 - 3 * dy, 3 - 3 * dx] = 1
        w3[j, j, 1 - dy, 1 - dx] = 1
    goal_map = mul(
        clip01(total(linear(linear(arcs, w7), w3))), world)

    # Features: y, x, type code, half-width, width, presence.
    def object_features(det, object_sizes, object_halves, centered):
        nc = shapes[det][0]
        w = np.zeros((6, 3 + nc), np.float32)
        w[0, 0] = 63
        w[1, 1] = 63
        if not centered:
            w[0, 3:] = object_halves
            w[1, 3:] = object_halves
        w[2, 3:] = np.arange(1, nc + 1)
        w[3, 3:] = object_halves
        w[4, 3:] = object_sizes
        w[5, 3:] = 1
        return linear(cat("coords", det), w)

    ff = object_features(field_fruit, sizes, halves, False)
    ef = object_features(field_enemy, np.full(3, 5),
                         np.full(3, 2), True)
    wg = np.zeros((3, 4), np.float32)
    wg[0, 0] = 63
    wg[1, 1] = 63
    wg[2, 3] = 1
    gf = linear(cat("coords", goal_map), wg)
    priority = linear("coords", [[4032, 63, 1]])

    def extract(feats, mask, n):
        k = shapes[feats][0]
        score = mul(priority, mask)
        shift = shift_matrix(n, k)
        insert = np.zeros((n * k, k), np.float32)
        insert[-k:, :] = np.eye(k)

        def body(s, buf):
            high = emit("gmax", [s])
            win = emit("min2", [
                relu(affine(sub(s, high), 2, 1)), mask
            ])
            item = emit("gmax", [mul(feats, win)])
            return [
                sub(s, affine(win, 10000)),
                add(linear(buf, shift), linear(item, insert))
            ]

        return loop(n, [score, const(0, n * k)], body)[1]

    fv = extract(ff, sl(ff, 5, 6), NF)
    ev = extract(ef, sl(ef, 5, 6), NE)
    gv_observed = extract(gf, goal_map, NG)

    # Goal memory is invalidated on RESET. Object history is independent
    # of the timer bar, so a wasted action really is recognized as wasted.
    remembered_goals = sl("mem", 0, GOAL_MEM)
    previous_objects = sl("mem", GOAL_MEM, GOAL_MEM + OBJECT_MEM)
    older_objects = sl("mem", GOAL_MEM + OBJECT_MEM, M)
    previous_reset = sl("aprev", 0, 1)
    memory_valid = mul(
        clip01(total(remembered_goals)),
        affine(previous_reset, -1, 1))
    gv = add(mul(remembered_goals, memory_valid),
             mul(gv_observed, affine(memory_valid, -1, 1)))
    object_state = cat(fv, ev)
    emit("vconcat", [gv, object_state, previous_objects],
         name="mem_next")

    same_previous = clip01(affine(
        total(absolute(sub(object_state, previous_objects))), -4, 1))
    same_older = clip01(affine(
        total(absolute(sub(object_state, older_objects))), -4, 1))
    previous_was_move = total(sl("aprev", 6, 8))
    cycle = mul(
        clip01(add(same_previous, same_older)), previous_was_move)

    fruit_goal_code = emit("gmax", [
        linear(header_fruit, [np.arange(1, 10, dtype=np.float32)])
    ])
    enemy_goal_code = emit("gmax", [
        linear(header_enemy, [[1, 2, 3]])
    ])
    fruit_required = total(emit("gmean", [header_fruit]), 4096)
    enemy_required = total(emit("gmean", [header_enemy]), 4096)

    F = [feature(fv, j, NF) for j in range(6)]
    E = [feature(ev, j, NE) for j in range(6)]
    qy, qx, qcode, qhalf, qwidth, qpresent = [
        cat(F[j], E[j]) for j in range(6)
    ]
    is_fruit = fixed(np.r_[np.ones(NF), np.zeros(NE)])
    is_enemy = fixed(np.r_[np.zeros(NF), np.ones(NE)])
    qheight = sub(qwidth, is_enemy)
    rx = affine(qwidth, .5, -.5)
    ry = affine(qheight, .5, -.5)
    ox = sub(qhalf, rx)
    oy = sub(qhalf, ry)

    # Shared nearest-object selection circuit.
    def nearest(targets, n, k, compatibility):
        shift = shift_matrix(n, k)

        def body(t, bx, by, best, bs, iteration):
            ty, tx = sl(t, 0, 1), sl(t, 1, 2)
            tc = sl(t, 2, 3) if k == 6 else const(0)
            ts = sl(t, 4, 5) if k == 6 else const(0)
            tp = sl(t, k - 1, k)
            dx, dy = sub(qx, tx), sub(qy, ty)
            d2 = add(mul(dx, dx), mul(dy, dy))
            cost = d2
            if compatibility is not None:
                mismatch = sub(const(1), eqcode(qcode, tc))
                self_match = relu(affine(d2, -1, 1))
                penalty = mul(add(mismatch, self_match), compatibility)
                cost = add(cost, affine(penalty, 100000))
            cost = add(cost, affine(tp, -1000000, 1000000))
            cost = add(cost, affine(iteration, .01))
            win = clip01(affine(sub(best, cost), 1000, -.5))
            return [
                linear(t, shift),
                mix(bx, tx, win),
                mix(by, ty, win),
                mix(best, cost, win),
                mix(bs, ts, win),
                affine(iteration, 1, 1),
            ]

        out = loop(n, [
            targets, const(0, Q), const(0, Q),
            const(10000000, Q), const(0, Q), const(0)
        ], body)
        return out[1], out[2], out[3], out[4]

    near_f = nearest(fv, NF, 6, is_fruit)
    near_e = nearest(ev, NE, 6, is_enemy)
    near_g = nearest(gv, NG, 3, None)

    def goal_distance(x, y, gx, gy):
        return add(relu(affine(absolute(sub(x, gx)), 1, -4)),
                   relu(affine(absolute(sub(y, gy)), 1, -4)))

    old_goal_distance = goal_distance(qx, qy, near_g[0], near_g[1])
    in_goal = clip01(affine(old_goal_distance, -1, 1))

    def task_flags(code, present, requested, count, inside, n):
        high = mul(clip01(sub(code, requested)), present)
        low = mul(clip01(sub(requested, code)), present)
        same = sub(present, add(high, low))
        parked = mul(same, inside)
        unparked = sub(same, parked)
        free = relu(sub(count, total(parked)))
        prefix = linear(unparked, np.tril(np.ones((n, n), np.float32)))
        chosen = mul(unparked, clip01(sub(affine(free, 1, 1), prefix)))
        active = add(parked, chosen)
        create = clip01(sub(count, total(same)))
        return active, mul(low, create), mul(high, create), same

    af, lf, hf, mf = task_flags(
        F[2], F[5], fruit_goal_code, fruit_required,
        sl(in_goal, 0, NF), NF)
    ae, le, he, me = task_flags(
        E[2], E[5], enemy_goal_code, enemy_required,
        sl(in_goal, NF, Q), NE)
    has_f = clip01(affine(near_f[2], -1, 50000))
    has_e = clip01(affine(near_e[2], -1, 50000))
    lf_task = mul(lf, sl(has_f, 0, NF))
    le_task = mul(le, sl(has_e, NF, Q))
    hf_task = mul(hf, sl(has_e, 0, NF))

    def targets_for_axis(axis):
        ftarget = add_many([
            mul(af, sl(near_g[axis], 0, NF)),
            mul(lf_task, sl(near_f[axis], 0, NF)),
            mul(hf_task, sl(near_e[axis], 0, NF)),
        ])
        etarget = add(
            mul(ae, sl(near_g[axis], NF, Q)),
            mul(le_task, sl(near_e[axis], NF, Q)))
        return cat(ftarget, etarget)

    target_x = targets_for_axis(0)
    target_y = targets_for_axis(1)
    weights = cat(
        add_many([affine(af, 2), affine(lf_task, 1.5),
                  affine(hf_task, .25)]),
        add(affine(ae, 1.5), le_task))
    padding = affine(cat(af, ae), 4)

    # Parallel click selection and attraction geometry.
    def selection(cx, cy, radx, rady, offx, offy, present):
        dx, dy = sub(X, cx), sub(Y, cy)
        ax = relu(sub(absolute(add(dx, offx)), radx))
        ay = relu(sub(absolute(add(dy, offy)), rady))
        d2 = add(mul(ax, ax), mul(ay, ay))
        selected = mul(clip01(affine(d2, -2, 129)), present)
        return selected, dx, dy

    selected, dx, dy = selection(qx, qy, rx, ry, ox, oy, qpresent)
    sf = sl(selected, 0, NF)
    se = sl(selected, NF, Q)

    def clamp_position(px, py, half, width, height):
        low_y = affine(half, 1, 10)
        high_x = add(sub(const(64), width), half)
        high_y = add(sub(const(64), height), half)
        px = emit("min2", [emit("max2", [px, half]), high_x])
        py = emit("min2", [emit("max2", [py, low_y]), high_y])
        return px, py

    click_x, click_y = clamp_position(X, Y, qhalf, qwidth, qheight)
    normal_x = mix(qx, click_x, selected)
    normal_y = mix(qy, click_y, selected)
    fruit_x, fruit_y = sl(normal_x, 0, NF), sl(normal_y, 0, NF)

    # First-tier hunter drag: four rounded fixed-direction increments.
    edx, edy = sl(dx, NF, Q), sl(dy, NF, Q)
    ex2, ey2 = mul(edx, edx), mul(edy, edy)

    def rounded_motion(primary, a2, b2):
        w = np.zeros((3 * NE, 2 * NE), np.float32)
        for j in range(NE):
            for t, threshold in enumerate([25, 225, 625]):
                row = t * NE + j
                w[row, j] = 1156 - threshold
                w[row, NE + j] = -threshold
        bits = clip01(linear(cat(a2, b2), w,
                             np.full(3 * NE, .5)))
        reduce = np.zeros((NE, 3 * NE), np.float32)
        for j in range(NE):
            reduce[j, [j, NE + j, 2 * NE + j]] = 4
        magnitude = linear(bits, reduce)
        sign = emit("max2", [
            emit("min2", [primary, const(1)]), const(-1)
        ])
        return mul(magnitude, sign)

    emx = rounded_motion(edx, ex2, ey2)
    emy = rounded_motion(edy, ey2, ex2)
    drag1_x, drag1_y = clamp_position(
        add(E[1], emx), add(E[0], emy),
        E[3], E[4], affine(E[4], 1, -1))
    type1 = eqcode(E[2], const(1))
    drag_x = mix(sl(click_x, NF, Q), drag1_x, type1)
    drag_y = mix(sl(click_y, NF, Q), drag1_y, type1)

    # Unselected hunters approximately pursue the predicted nearest fruit.
    nfx, nfy = sl(near_f[0], NF, Q), sl(near_f[1], NF, Q)
    nfs = sl(near_f[3], NF, Q)
    nearest_selected, _, _ = selection(
        nfx, nfy, affine(nfs, .5), affine(nfs, .5),
        const(0), const(0), E[5])
    chase_target_x = mix(nfx, X, nearest_selected)
    chase_target_y = mix(nfy, Y, nearest_selected)
    chase_speed = affine(eqcode(E[2], const(3)), 4, 4)

    def chase_axis(target, origin):
        delta = sub(target, origin)
        delta = emit("min2", [delta, chase_speed])
        delta = emit("max2", [delta, affine(chase_speed, -1)])
        return add(origin, delta)

    chase_x, chase_y = clamp_position(
        chase_axis(chase_target_x, E[1]),
        chase_axis(chase_target_y, E[0]),
        E[3], E[4], affine(E[4], 1, -1))
    enemy_x = mix(chase_x, drag_x, se)
    enemy_y = mix(chase_y, drag_y, se)
    predicted_x = cat(fruit_x, enemy_x)
    predicted_y = cat(fruit_y, enemy_y)

    old_distance = add(
        relu(sub(absolute(sub(qx, target_x)), padding)),
        relu(sub(absolute(sub(qy, target_y)), padding)))
    new_distance = add(
        relu(sub(absolute(sub(predicted_x, target_x)), padding)),
        relu(sub(absolute(sub(predicted_y, target_y)), padding)))
    scores = [total(mul(sub(old_distance, new_distance), weights))]

    # Equal-rank merging, with penalties for destructive groupings.
    rank_counts = []
    for code in range(1, 10):
        rank_counts.append(total(mul(sf, eqcode(F[2], const(code)))))
    counts = cat(*rank_counts)
    number_selected = total(counts)
    different_types = total(clip01(counts))
    scores.append(total(relu(affine(different_types, 1, -1)), -180))
    scores.append(total(relu(affine(number_selected, 1, -2)), -180))
    merging = clip01(affine(number_selected, 1, -1))
    lower_selected = total(mul(sf, lf))
    scores.append(mul(merging, affine(lower_selected, 40, -60)))

    # Fruit/hunter collision and intentional downgrade potentials.
    NP = NF * NE
    pf = np.zeros((NP, NF), np.float32)
    pe = np.zeros((NP, NE), np.float32)
    for i in range(NF):
        for j in range(NE):
            row = i * NE + j
            pf[row, i] = 1
            pe[row, j] = 1
    difference = np.concatenate([pf, -pe], axis=1)
    pp = mul(linear(F[5], pf), linear(E[5], pe))
    pair_high = mul(linear(hf, pf), pp)
    pair_protect = sub(pp, pair_high)
    pair_rx = affine(linear(sl(rx, 0, NF), pf), 1, 2)
    pair_ry = affine(linear(sl(ry, 0, NF), pf), 1, 1.5)
    pair_ox = linear(sl(ox, 0, NF), pf)
    pair_oy = affine(linear(sl(oy, 0, NF), pf), 1, -.5)

    def pair_distance(px, py):
        xx = absolute(sub(linear(px, difference), pair_ox))
        yy = absolute(sub(linear(py, difference), pair_oy))
        return add(relu(sub(xx, pair_rx)), relu(sub(yy, pair_ry)))

    pd = pair_distance(predicted_x, predicted_y)
    old_pd = pair_distance(qx, qy)
    nearby = relu(affine(pd, -1, 5))
    contact = relu(affine(pd, -1, 1))
    not_dragged = affine(linear(se, pe), -1, 1)
    vulnerable = mul(not_dragged, pair_protect)
    scores.append(total(mul(nearby, vulnerable), -14))
    scores.append(total(mul(nearby, pp), -.3))

    old_pd_capped = affine(
        clip01(affine(old_pd, 1.0 / 24.0)), 24)
    pd_capped = emit("min2", [pd, const(24)])
    approach = sub(old_pd_capped, pd_capped)
    downgrade_value = add(approach, affine(contact, 16))
    downgrade_value = mul(
        mul(downgrade_value, not_dragged), pair_high)
    scores.append(total(downgrade_value))

    # Hunter/hunter compatibility.
    pairs = [(i, j) for i in range(NE) for j in range(i + 1, NE)]
    nh = len(pairs)
    hi = np.zeros((nh, NE), np.float32)
    hj = np.zeros((nh, NE), np.float32)
    for row, (i, j) in enumerate(pairs):
        hi[row, i] = 1
        hj[row, j] = 1
    hd = hi - hj
    hdx = absolute(linear(enemy_x, hd))
    hdy = absolute(linear(enemy_y, hd))
    separation = add(relu(affine(hdx, 1, -4)),
                     relu(affine(hdy, 1, -3)))
    hcontact = clip01(affine(separation, -1, 1))
    equal_type = relu(affine(absolute(linear(E[2], hd)), -1, 1))
    both_present = mul(linear(E[5], hi), linear(E[5], hj))
    both_lower = mul(linear(le, hi), linear(le, hj))
    both_matching = mul(linear(me, hi), linear(me, hj))
    hunter_value = add_many([
        affine(both_lower, 30),
        affine(both_matching, -60),
        affine(equal_type, 200, -200),
    ])
    scores.append(total(mul(
        hcontact, mul(hunter_value, both_present))))

    utility = add_many(scores)
    add(utility, affine(world, 10000, -10000), name="click")

    # Undo on explicit infeasibility; reset on a detected unproductive
    # fixed point or two-state cycle. RESET has higher recovery priority.
    red = emit("gmax", [sl("x0", 2, 3)])
    lw = np.zeros((8, 2), np.float32)
    lw[7, 0] = 20
    lw[0, 1] = 40
    lb = np.full(8, -10, np.float32)
    lb[0] = -5
    lb[6] = 1
    lb[7] = 0
    linear(cat(red, cycle), lw, lb, name="logits")

    return {"graph": graph, "memory_size": M, "params": params}
