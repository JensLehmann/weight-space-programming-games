import numpy as np


def build():
    # Canonical rendering: a 20x20 board, magnified threefold,
    # with block centres at display coordinates 3, 6, ..., 60.
    N = 5
    E = 8
    D = 3 * E + 7
    M = N * D + N + 6

    G = []
    P = {}
    serial = [0]
    pserial = [0]
    constants = {}

    def uid(prefix="n"):
        serial[0] += 1
        return "%s_%d" % (prefix, serial[0])

    def parameter(a):
        pserial[0] += 1
        name = "p_%d" % pserial[0]
        P[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, ins, dst=None, out=None, **kw):
        if dst is None:
            dst = G
        name = out if out is not None else uid()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(kw)
        dst.append(nd)
        return name

    def dense(x, w, b=None, dst=None, out=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("dense", [x], dst, out, **kw)

    def conv(x, w, b=None, dst=None, out=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("conv", [x], dst, out, **kw)

    def literal(values):
        a = np.asarray(values, dtype=np.float32).reshape(-1)
        return dense("aprev", np.zeros((len(a), 8), np.float32), a)

    def K(value):
        key = float(value)
        if key not in constants:
            constants[key] = literal([key])
        return constants[key]

    def vs(x, a, b, dst=None):
        return node("vslice", [x], dst, start=a, stop=b)

    def ss(x, a, b, dst=None):
        return node("slice", [x], dst, start=a, stop=b)

    def va(a, b, dst=None):
        return node("vadd", [a, b], dst)

    def vm(a, b, dst=None):
        return node("vmul", [a, b], dst)

    def sa(a, b, dst=None):
        return node("add", [a, b], dst)

    def sm(a, b, dst=None):
        return node("mul", [a, b], dst)

    def sd(a, b, dst=None):
        return node("sub", [a, b], dst)

    def vf(x, scale=1.0, bias=0.0, dst=None):
        y = x
        if scale != 1.0:
            y = vm(y, K(scale), dst)
        if bias != 0.0:
            y = va(y, K(bias), dst)
        return y

    def sf(x, scale=1.0, bias=0.0, dst=None):
        y = x
        if scale != 1.0:
            y = sm(y, K(scale), dst)
        if bias != 0.0:
            y = sa(y, K(bias), dst)
        return y

    def vn(x, dst=None):
        return vf(x, -1.0, 1.0, dst)

    def sn(x, dst=None):
        return sf(x, -1.0, 1.0, dst)

    def vh(x, threshold=0.5, gain=100.0, dst=None):
        y = node("vrelu", [vf(x, gain, -gain * threshold, dst)], dst)
        y = node("vrelu", [vn(y, dst)], dst)
        return vn(y, dst)

    def sh(x, threshold=0.5, gain=100.0, dst=None):
        y = node("relu", [sf(x, gain, -gain * threshold, dst)], dst)
        return node("min2", [y, K(1.0)], dst)

    def vsubtract(a, b, dst=None):
        return va(a, vf(b, -1.0, 0.0, dst), dst)

    def vabs(x, dst=None):
        a = node("vrelu", [x], dst)
        b = node("vrelu", [vf(x, -1.0, 0.0, dst)], dst)
        return va(a, b, dst)

    def vtotal(x, length, dst=None):
        return dense(x, np.ones((1, length), np.float32), dst=dst)

    def vacc(xs, dst=None):
        y = xs[0]
        for x in xs[1:]:
            y = va(y, x, dst)
        return y

    def sacc(xs, dst=None):
        y = xs[0]
        for x in xs[1:]:
            y = sa(y, x, dst)
        return y

    def scalar_max_from_vector(x, length, zero, dst=None):
        z = sa(zero, x, dst)
        y = ss(z, 0, 1, dst)
        for j in range(1, length):
            y = node("max2", [y, ss(z, j, j + 1, dst)], dst)
        return node("gmax", [y], dst)

    # Constant coordinate fields.
    xy = conv(
        "coords",
        np.array([[0, 21, 0], [21, 0, 0]], np.float32),
        [-1, -1],
    )
    gx = ss(xy, 0, 1)
    gy = ss(xy, 1, 2)
    one = ss("coords", 2, 3)
    zero = sm(one, K(0.0))

    # An analytic lattice sampler. Each axis is a sum of narrow
    # triangular hats, subsequently clipped to exact zero or one.
    hw = np.zeros((120, 2), np.float32)
    hb = np.zeros(120, np.float32)
    sw = np.zeros((2, 120), np.float32)
    for axis in range(2):
        for j in range(20):
            base = axis * 60 + j * 3
            for t, offset in enumerate((-0.2, 0.0, 0.2)):
                hw[base + t, axis] = 1
                hb[base + t] = -(j + offset)
            sw[axis, base:base + 3] = [5, -10, 5]
    hats = node("relu", [conv(xy, hw, hb)])
    axes = sh(conv(hats, sw), 0.5, 20)
    lattice = sm(ss(axes, 0, 1), ss(axes, 1, 2))

    # Background is the dominant colour. The UI is outside the lattice.
    colour_means = node("gmean", ["x0"])
    background_colour = vh(colour_means, 0.55, 100)
    background = conv(
        sm("x0", background_colour),
        np.ones((1, 16), np.float32),
    )
    c0 = ss("x0", 0, 1)
    c3 = ss("x0", 3, 4)
    c4 = ss("x0", 4, 5)
    c8 = ss("x0", 8, 9)
    terminal_colour = sa(c3, c8)

    grey = node("gmax", [sm(c4, lattice)])
    not_grey = vn(grey)

    all_body = node(
        "relu",
        [sd(sd(sd(one, background), c3), c8)],
    )
    grey_selected_body = node("relu", [sd(all_body, c4)])
    selected_body = sa(
        sm(grey_selected_body, grey),
        sm(c0, not_grey),
    )
    body_grid = sm(selected_body, lattice)
    body_present = node("gmax", [body_grid])
    all_body_grid = sm(all_body, lattice)

    # An endpoint belongs to the selected piece when it is next to
    # selected body colour. Green endpoints are included as well.
    adjacency_kernel = np.zeros((1, 1, 7, 7), np.float32)
    for r, c in ((0, 3), (6, 3), (3, 0), (3, 6), (3, 3)):
        adjacency_kernel[0, 0, r, c] = 1
    adjacency = sh(conv(selected_body, adjacency_kernel), 0.5, 10)
    terminal_grid = sm(sm(terminal_colour, adjacency), lattice)
    selected_occupancy = node(
        "min2", [sa(body_grid, terminal_grid), K(1.0)]
    )

    key = conv(xy, np.array([[-1, -20]], np.float32), [400])

    def measure(mask, dst=None):
        features = node(
            "concat",
            [sm(mask, gx, dst), sm(mask, gy, dst), mask],
            dst,
        )
        return dense(
            node("gmean", [features], dst),
            np.eye(3, dtype=np.float32) * 4096,
            dst=dst,
        )

    def pick(mask, priority, dst=None):
        score = sm(mask, priority, dst)
        best = node("gmax", [score], dst)
        equal = sh(sd(score, best, dst), -0.001, 2000, dst)
        return sm(equal, mask, dst)

    # Bounding box of the selected piece.
    extent_features = node(
        "concat",
        [
            sm(selected_occupancy, sf(gx, -1, 20)),
            sm(selected_occupancy, sf(gy, -1, 20)),
            sm(selected_occupancy, gx),
            sm(selected_occupancy, gy),
        ],
    )
    extrema = node("gmax", [extent_features])
    bw = np.array(
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        np.float32,
    )
    bbox = dense(extrema, bw, [20, 20, -19, -19])

    # Store an interior body pixel for later selection.
    degree = conv(selected_body, adjacency_kernel)
    body_priority = sa(sf(degree, 500), sf(key, 0.01))
    body_pick = pick(body_grid, body_priority)
    body_location = measure(body_pick)
    click_location = vs(body_location, 0, 2)

    # Extract up to eight terminals in row-major order.
    residual = terminal_grid
    tx, ty, tv = [], [], []
    for j in range(E):
        chosen = pick(residual, key)
        measured = measure(chosen)
        tx.append(vs(measured, 0, 1))
        ty.append(vs(measured, 1, 2))
        tv.append(vs(measured, 2, 3))
        residual = sd(residual, chosen)

    observed = node(
        "vconcat",
        tx + ty + tv + [bbox, click_location, body_present],
    )

    # A black end marker on a grey-masked selected piece denotes
    # the extensible family. Advance it until the marker disappears.
    marker = node("gmax", [sm(sm(c0, selected_body), lattice)])
    grow = vm(grey, marker)

    # Memory layout:
    # N records of [x[E], y[E], valid[E], bbox[4], click[2], valid[1]]
    # followed by selection[N], plan[4], phase[1], clock[1].
    old_slots = [vs("mem", i * D, (i + 1) * D) for i in range(N)]
    sel_start = N * D
    old_sel = vs("mem", sel_start, sel_start + N)
    old_plan = vs("mem", sel_start + N, sel_start + N + 4)
    old_phase = vs("mem", sel_start + N + 4, sel_start + N + 5)
    clock = vs("mem", sel_start + N + 5, sel_start + N + 6)

    e0 = literal([1] + [0] * (N - 1))
    sel = va(old_sel, vm(e0, vn(vtotal(old_sel, N))))
    sel_scalars = [vs(sel, i, i + 1) for i in range(N)]

    expected = vacc(
        [vm(old_slots[i], sel_scalars[i]) for i in range(N)]
    )

    # Detect a failed selection rather than corrupting the catalogue.
    expected_box = vs(expected, 3 * E, 3 * E + 4)
    box_error = vtotal(vabs(vsubtract(bbox, expected_box)), 4)
    selection_error = vm(
        vm(vh(box_error, 0.5), vs(expected, D - 1, D)),
        vs("aprev", 6, 7),
    )
    failure = vh(va(vn(body_present), selection_error), 0.5)
    healthy = vn(failure)

    slots = []
    for i in range(N):
        write = vm(sel_scalars[i], body_present)
        slots.append(
            va(vm(old_slots[i], vn(write)), vm(observed, write))
        )

    active = vacc([vm(slots[i], sel_scalars[i]) for i in range(N)])
    valid = [vs(s, D - 1, D) for s in slots]
    valid_vector = node("vconcat", valid)

    # Rectangles are used only for cataloguing and clicking, not
    # for collision constraints: sprites may pass through each other.
    bxv = node("vconcat", [vs(s, 3 * E, 3 * E + 1) for s in slots])
    byv = node("vconcat", [vs(s, 3 * E + 1, 3 * E + 2) for s in slots])
    wv = node("vconcat", [vs(s, 3 * E + 2, 3 * E + 3) for s in slots])
    hv = node("vconcat", [vs(s, 3 * E + 3, 3 * E + 4) for s in slots])

    left = sh(sd(gx, bxv), -0.5, 4)
    top = sh(sd(gy, byv), -0.5, 4)
    right = sh(sd(vf(va(bxv, wv), 1, -1), gx), -0.5, 4)
    bottom = sh(sd(vf(va(byv, hv), 1, -1), gy), -0.5, 4)
    rectangles = sm(sm(sm(sm(left, right), top), bottom), valid_vector)
    sum_rect = conv(rectangles, np.ones((1, N), np.float32))
    known = sh(sum_rect, 0.5, 10)
    unknown = sm(all_body_grid, sn(known))
    unknown_exists = node("gmax", [unknown])

    fresh_parts = []
    prefix = K(1.0)
    for i in range(N):
        fresh_parts.append(vm(prefix, vn(valid[i])))
        prefix = vm(prefix, valid[i])
    fresh = node("vconcat", fresh_parts)
    fresh_exists = vtotal(fresh, N)
    explore = vm(vm(vn(old_phase), unknown_exists), fresh_exists)
    optimize = vn(explore)

    # Select the next existing piece, skipping unused slots.
    next_parts = []
    found = K(0.0)
    for offset in range(1, N + 1):
        rw = np.zeros((N, N), np.float32)
        for i in range(N):
            rw[(i + offset) % N, i] = 1
        candidate = vm(dense(sel, rw), valid_vector)
        taken = vm(candidate, vn(found))
        next_parts.append(taken)
        found = va(found, vtotal(taken, N))
    next_sel = vacc(next_parts)
    next_scalars = [vs(next_sel, i, i + 1) for i in range(N)]
    next_data = vacc(
        [vm(slots[i], next_scalars[i]) for i in range(N)]
    )

    # Click a body pixel outside other catalogued rectangles if one
    # exists. Otherwise use the stored interior pixel.
    next_rect = conv(
        sm(rectangles, next_sel), np.ones((1, N), np.float32)
    )
    other_rect = conv(
        sm(rectangles, vn(next_sel)), np.ones((1, N), np.float32)
    )
    safe_body = sm(
        sm(all_body_grid, next_rect),
        sn(sh(other_rect, 0.5, 10)),
    )
    safe_exists = node("gmax", [safe_body])
    ncx = vs(next_data, 3 * E + 4, 3 * E + 5)
    ncy = vs(next_data, 3 * E + 5, 3 * E + 6)
    click_distance = sa(
        node("abs", [sd(gx, ncx)]),
        node("abs", [sd(gy, ncy)]),
    )
    near_click = sf(click_distance, -1, 100)
    safe_priority = sa(near_click, sf(key, 0.0001))
    next_click_map = sa(
        sm(sm(safe_body, safe_priority), safe_exists),
        sm(sm(lattice, near_click), vn(safe_exists)),
    )
    unknown_click_map = sm(unknown, key)
    recovery_click_map = sm(all_body_grid, key)

    # Fixed terminals: the selected object is removed from this list.
    FX = node("vconcat", [vs(s, 0, E) for s in slots])
    FY = node("vconcat", [vs(s, E, 2 * E) for s in slots])
    FV = node(
        "vconcat",
        [
            vm(
                vm(vs(slots[i], 2 * E, 3 * E), valid[i]),
                vn(sel_scalars[i]),
            )
            for i in range(N)
        ],
    )
    L = N * E
    groups = literal([i for i in range(N) for _ in range(E)])
    field_x = sa(zero, FX)
    field_y = sa(zero, FY)
    field_group = sa(zero, groups)

    # Both loops use a cyclic queue of four vectors of length L.
    roll = np.zeros((4 * L, 4 * L), np.float32)
    for field in range(4):
        for j in range(L):
            roll[field * L + j, field * L + (j + 1) % L] = 1
    roll_parameter = parameter(roll)

    # First loop: nearest terminal on another stationary object.
    q1 = node("vconcat", [FX, FY, FV, groups])
    near_initial = conv(
        zero, np.zeros((L, 1), np.float32),
        np.full(L, 80, np.float32),
    )
    q_state = uid("queue")
    n_state = uid("nearest")
    body = []
    qx = vs(q_state, 0, 1, body)
    qy = vs(q_state, L, L + 1, body)
    qv = vs(q_state, 2 * L, 2 * L + 1, body)
    qg = vs(q_state, 3 * L, 3 * L + 1, body)
    dist = sa(
        node("abs", [sd(field_x, qx, body)], body),
        node("abs", [sd(field_y, qy, body)], body),
        body,
    )
    same = node(
        "relu",
        [sf(node("abs", [sd(field_group, qg, body)], body),
            -1, 1, body)],
        body,
    )
    dist = sa(dist, sf(same, 1000, 0, body), body)
    dist = sa(dist, vf(qv, -1000, 1000, body), body)
    n_next = node("min2", [n_state, dist], body)
    q_next = node("dense", [q_state], body, w=roll_parameter)
    loop1 = uid("FixedDistances")
    G.append({
        "id": loop1,
        "op": "loop",
        "T": L,
        "state": [
            {"id": q_state, "init": q1},
            {"id": n_state, "init": near_initial},
        ],
        "body": body,
        "next": {q_state: q_next, n_state: n_next},
    })
    old_nearest = node("gmax", [loop1 + "." + n_state])

    # Candidate endpoint coordinates. Channels are terminal-major:
    # four consecutive channels represent its four quarter-turns.
    sxw = np.zeros((4 * E, D), np.float32)
    syw = np.zeros((4 * E, D), np.float32)
    sxb = np.zeros(4 * E, np.float32)
    syb = np.zeros(4 * E, np.float32)
    vrep = np.zeros((4 * E, D), np.float32)
    X0, Y0, W0, H0 = 3 * E, 3 * E + 1, 3 * E + 2, 3 * E + 3

    for j in range(E):
        for r in range(4):
            t = 4 * j + r
            vrep[t, 2 * E + j] = 1
            if r == 0:
                sxw[t, j] = 1
                sxw[t, X0] = -1
                syw[t, E + j] = 1
                syw[t, Y0] = -1
            elif r == 1:
                sxw[t, E + j] = -1
                sxw[t, Y0] = 1
                sxw[t, H0] = 1
                sxb[t] = -1
                syw[t, j] = 1
                syw[t, X0] = -1
            elif r == 2:
                sxw[t, j] = -1
                sxw[t, X0] = 1
                sxw[t, W0] = 1
                sxb[t] = -1
                syw[t, E + j] = -1
                syw[t, Y0] = 1
                syw[t, H0] = 1
                syb[t] = -1
            else:
                sxw[t, E + j] = 1
                sxw[t, Y0] = -1
                syw[t, j] = -1
                syw[t, X0] = 1
                syw[t, W0] = 1
                syb[t] = -1

    candidate_x = sa(
        conv(gx, np.ones((4 * E, 1), np.float32)),
        dense(active, sxw, sxb),
    )
    candidate_y = sa(
        conv(gy, np.ones((4 * E, 1), np.float32)),
        dense(active, syw, syb),
    )
    selected_valid = dense(active, vrep)
    selected_invalid_penalty = vf(selected_valid, -1000, 1000)

    # Second loop: symmetric nearest-neighbour energy, plus a penalty
    # when a new endpoint lands on an already paired stationary point.
    q2 = node("vconcat", [FX, FY, FV, old_nearest])
    q2_state = uid("pose_queue")
    selected_near_state = uid("selected_nearest")
    energy_state = uid("stationary_energy")
    body2 = []

    qx = vs(q2_state, 0, 1, body2)
    qy = vs(q2_state, L, L + 1, body2)
    qv = vs(q2_state, 2 * L, 2 * L + 1, body2)
    qo = vs(q2_state, 3 * L, 3 * L + 1, body2)

    distances = sa(
        node("abs", [sd(candidate_x, qx, body2)], body2),
        node("abs", [sd(candidate_y, qy, body2)], body2),
        body2,
    )
    query_distances = sa(
        distances, vf(qv, -1000, 1000, body2), body2
    )
    selected_near_next = node(
        "min2", [selected_near_state, query_distances], body2
    )

    eligible = sa(distances, selected_invalid_penalty, body2)
    closest_selected = ss(eligible, 0, 4, body2)
    for j in range(1, E):
        closest_selected = node(
            "min2",
            [closest_selected, ss(eligible, 4 * j, 4 * j + 4, body2)],
            body2,
        )

    fixed_cost = sm(
        node("min2", [closest_selected, qo], body2),
        qv,
        body2,
    )
    new_hit = node(
        "relu", [sf(closest_selected, -1, 0.5, body2)], body2
    )
    old_hit = node(
        "vrelu", [vf(qo, -1, 0.5, body2)], body2
    )
    collision = sm(sm(new_hit, old_hit, body2), qv, body2)
    fixed_cost = sa(fixed_cost, sf(collision, 80, 0, body2), body2)
    energy_next = sa(energy_state, fixed_cost, body2)
    q2_next = node("dense", [q2_state], body2, w=roll_parameter)

    selected_near_initial = conv(
        zero,
        np.zeros((4 * E, 1), np.float32),
        np.full(4 * E, 80, np.float32),
    )
    loop2 = uid("PoseEnergy")
    G.append({
        "id": loop2,
        "op": "loop",
        "T": L,
        "state": [
            {"id": q2_state, "init": q2},
            {"id": selected_near_state, "init": selected_near_initial},
            {"id": energy_state, "zeros": [4, 64, 64]},
        ],
        "body": body2,
        "next": {
            q2_state: q2_next,
            selected_near_state: selected_near_next,
            energy_state: energy_next,
        },
    })

    reduce_terminals = np.zeros((4, 4 * E), np.float32)
    for j in range(E):
        for r in range(4):
            reduce_terminals[r, 4 * j + r] = 1
    selected_cost = conv(
        sm(loop2 + "." + selected_near_state, selected_valid),
        reduce_terminals,
    )
    energy = sa(selected_cost, loop2 + "." + energy_state)

    # A small deterministic schedule resolves indistinguishable endpoint
    # labels and periodically tries a different orientation basin.
    hat_w = np.ones((96, 1), np.float32)
    hat_b = np.zeros(96, np.float32)
    schedule_w = np.zeros((5, 96), np.float32)
    for k in range(32):
        hat_b[3 * k:3 * k + 3] = [1 - k, -k, -1 - k]
        force = 1 if k > 0 and k % 6 == 0 else 0
        if k < 2:
            preferred = 0
        elif force:
            preferred = 1 if (k // 6) % 2 else 3
        else:
            preferred = 2 if (k * k + 3 * k) % 5 < 3 else 0
        schedule_w[preferred, 3 * k:3 * k + 3] = [1, -2, 1]
        if force:
            schedule_w[4, 3 * k:3 * k + 3] = [1, -2, 1]
    schedule_hats = node("vrelu", [dense(clock, hat_w, hat_b)])
    schedule = dense(schedule_hats, schedule_w)
    preferred = vs(schedule, 0, 4)
    force_orientation = vs(schedule, 4, 5)
    rotation_bias = va(
        literal([0, 0.001, 0.002, 0.003]),
        vf(preferred, -0.05),
    )
    rotation_bias = va(
        rotation_bias,
        vf(vm(vn(preferred), force_orientation), 10000),
    )

    widths_w = np.zeros((4, D), np.float32)
    heights_w = np.zeros((4, D), np.float32)
    for r in range(4):
        widths_w[r, W0 if r % 2 == 0 else H0] = 1
        heights_w[r, H0 if r % 2 == 0 else W0] = 1
    widths = dense(active, widths_w)
    heights = dense(active, heights_w)

    overflow_x = node("relu", [sf(sa(gx, widths), 1, -20)])
    overflow_y = node("relu", [sf(sa(gy, heights), 1, -20)])
    boundary_cost = sf(sa(overflow_x, overflow_y), 1000)
    sample_cost = sf(sn(lattice), 10000)

    current_x = vs(active, X0, X0 + 1)
    current_y = vs(active, Y0, Y0 + 1)
    travel = sa(
        node("abs", [sd(gx, current_x)]),
        node("abs", [sd(gy, current_y)]),
    )
    cost = sacc([
        energy,
        sf(travel, 0.0005),
        rotation_bias,
        boundary_cost,
        sample_cost,
    ])
    negative_cost = sf(cost, -1)
    per_rotation_best = node("gmax", [negative_cost])
    best_score = scalar_max_from_vector(per_rotation_best, 4, zero)
    best_mask = sm(
        sh(sd(negative_cost, best_score), -0.00015, 10000),
        lattice,
    )

    # Unique row-major/rotation tie-break, expressed without argmax.
    pose_priority = sa(sf(key, 10), literal([3, 2, 1, 0]))
    pose_score = sm(best_mask, pose_priority)
    pose_max = scalar_max_from_vector(
        node("gmax", [pose_score]), 4, zero
    )
    chosen_pose = sm(
        sh(sd(pose_score, pose_max), -0.001, 2000),
        best_mask,
    )
    pose_union = conv(chosen_pose, np.ones((1, 4), np.float32))
    pose_measure = measure(pose_union)
    new_goal_x = vs(pose_measure, 0, 1)
    new_goal_y = vs(pose_measure, 1, 2)
    new_goal_turns = dense(
        node("gmean", [chosen_pose]),
        np.array([[0, 4096, 8192, 12288]], np.float32),
    )

    # Execute one persistent pose plan before moving to another piece.
    plan_active = vs(old_plan, 0, 1)
    goal_x = va(
        vm(vs(old_plan, 1, 2), plan_active),
        vm(new_goal_x, vn(plan_active)),
    )
    goal_y = va(
        vm(vs(old_plan, 2, 3), plan_active),
        vm(new_goal_y, vn(plan_active)),
    )
    goal_turns = va(
        vm(vs(old_plan, 3, 4), plan_active),
        vm(new_goal_turns, vn(plan_active)),
    )

    rotate = vh(goal_turns, 0.5)
    dx = vsubtract(goal_x, current_x)
    dy = vsubtract(goal_y, current_y)
    right_move = vm(vh(dx, 0.5), vn(rotate))
    left_move = vm(vh(vf(dx, -1), 0.5), vn(rotate))
    vertical_gate = vn(vacc([rotate, right_move, left_move]))
    down_move = vm(vh(dy, 0.5), vertical_gate)
    up_move = vm(vh(vf(dy, -1), 0.5), vertical_gate)
    done = vn(vacc([
        rotate, right_move, left_move, down_move, up_move
    ]))

    not_grow = vn(grow)
    normal = vm(healthy, not_grow)
    movement_gate = vm(normal, optimize)
    ordinary_click = vm(normal, va(explore, vm(optimize, done)))

    reset_available = vs("avail", 0, 1)
    reset_action = vm(failure, reset_available)
    recovery_click = vm(failure, vn(reset_available))

    action_up = vm(up_move, movement_gate)
    action_down = vm(down_move, movement_gate)
    action_left = vm(left_move, movement_gate)
    action_right = vm(right_move, movement_gate)
    action_rotate = va(
        vm(rotate, movement_gate), vm(grow, healthy)
    )
    action_click = va(ordinary_click, recovery_click)

    action_vector = node(
        "vconcat",
        [
            reset_action,
            action_up,
            action_down,
            action_left,
            action_right,
            action_rotate,
            action_click,
            K(0.0),
        ],
    )
    dense(action_vector, np.eye(8, dtype=np.float32), out="logits")

    click_a = sm(unknown_click_map, vm(normal, explore))
    click_b = sm(
        next_click_map, vm(vm(normal, optimize), done)
    )
    click_c = sm(recovery_click_map, recovery_click)
    node("add", [sa(click_a, click_b), click_c], out="click")

    # Commit catalogue, selection, plan, phase, and schedule clock.
    switch_explore = vm(not_grow, explore)
    switch_next = vm(vm(not_grow, optimize), done)
    keep_selection = vn(va(switch_explore, switch_next))
    new_sel = vacc([
        vm(sel, keep_selection),
        vm(fresh, switch_explore),
        vm(next_sel, switch_next),
    ])

    keep_plan = vm(vm(not_grow, optimize), vn(done))
    turns_left = vsubtract(goal_turns, rotate)
    new_plan = node(
        "vconcat",
        [
            keep_plan,
            vm(goal_x, keep_plan),
            vm(goal_y, keep_plan),
            vm(turns_left, keep_plan),
        ],
    )
    phase_increment = vm(
        vm(vn(old_phase), vn(explore)), not_grow
    )
    new_phase = va(old_phase, phase_increment)

    tick = vm(vm(vm(done, optimize), not_grow), healthy)
    advanced_clock = va(clock, tick)
    wrapped_clock = va(
        advanced_clock,
        vf(vh(advanced_clock, 31.5), -32),
    )

    memory_value = node(
        "vconcat",
        slots + [new_sel, new_plan, new_phase, wrapped_clock],
    )
    node("vmul", [memory_value, healthy], out="mem_next")

    return {
        "graph": G,
        "memory_size": M,
        "params": P,
    }
