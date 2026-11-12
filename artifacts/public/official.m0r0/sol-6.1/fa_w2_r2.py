import math
import numpy as np


def build():
    P = {}
    graph = []
    dest = graph
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "mem": ("v", 18),
        "aprev": ("v", 8),
        "avail": ("v", 8),
    }
    serial = 0
    parameter_cache = {}
    global_cse = {}
    local_cse = global_cse
    constants = {}
    constant_values = {}
    repeat_weights = {}

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def par(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = fresh("p")
            parameter_cache[key] = name
            P[name] = a
        return parameter_cache[key]

    def emit(op, inputs, name=None, where=None, **fields):
        target = dest if where is None else where
        cache = global_cse if target is graph else local_cse
        key = (op, tuple(inputs), tuple(sorted(fields.items())))
        if name is None:
            if key in global_cse:
                return global_cse[key]
            if key in cache:
                return cache[key]

        name = name or fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        target.append(nd)

        ins = [shapes[x] for x in inputs]
        if op == "conv":
            sh = ("s", P[fields["w"]].shape[0])
        elif op == "dense":
            sh = ("v", P[fields["w"]].shape[0])
        elif op in ("slice", "vslice"):
            sh = (ins[0][0], fields["stop"] - fields["start"])
        elif op in ("concat", "vconcat"):
            sh = (ins[0][0], sum(s[1] for s in ins))
        elif op in ("gmax", "gmean"):
            sh = ("v", ins[0][1])
        elif op in ("add", "sub", "mul", "max2", "min2"):
            sh = ("s", max(s[1] for s in ins))
        elif op in ("vadd", "vmul"):
            sh = ("v", max(s[1] for s in ins))
        else:
            sh = ins[0]
        shapes[name] = sh
        cache[key] = name
        return name

    def conv(x, w):
        return emit("conv", [x], w=w)

    def dense(x, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return emit("dense", [x], **fields)

    def sl(x, start, stop):
        op = "slice" if shapes[x][0] == "s" else "vslice"
        return emit(op, [x], start=start, stop=stop)

    def cat(xs):
        return emit(
            "concat" if shapes[xs[0]][0] == "s" else "vconcat", xs
        )

    def vr(x):
        return emit("vrelu", [x])

    def sr(x):
        return emit("relu", [x])

    def va(a, b):
        return emit("vadd", [a, b])

    def vmul(a, b):
        return emit("vmul", [a, b])

    def sa(a, b):
        return emit("add", [a, b])

    def ss(a, b):
        return emit("sub", [a, b])

    def mul(a, b):
        return emit("mul", [a, b])

    def sab(x):
        return emit("abs", [x])

    one_s = sl("coords", 2, 3)
    one_v = emit("gmax", [one_s])
    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    constants[1.0] = one_v
    constant_values[one_v] = 1.0

    def const(value):
        value = float(value)
        if value not in constants:
            name = emit(
                "dense", [one_v], where=graph, w=par([[value]])
            )
            constants[value] = name
            constant_values[name] = value
        return constants[value]

    def vc(values):
        return dense(
            one_v, par(np.asarray(values, np.float32).reshape(-1, 1))
        )

    def neg(x):
        if x in constant_values:
            return const(-constant_values[x])
        return vmul(x, const(-1))

    def vs(a, b):
        return va(a, neg(b))

    def vm(x, value):
        return vmul(x, const(value))

    def sm(x, value):
        return mul(x, const(value))

    def vab(x):
        return va(vr(x), vr(neg(x)))

    def vmax(a, b):
        return va(b, vr(vs(a, b)))

    def vmin(a, b):
        return vs(a, vr(vs(a, b)))

    def affine(x, scale, bias):
        n = shapes[x][1]
        w = np.eye(n, dtype=np.float32) * scale
        b = np.full(n, bias, np.float32)
        return dense(x, par(w), par(b))

    def clipv(x):
        n = shapes[x][1]
        eye = np.eye(n, dtype=np.float32)
        w = np.concatenate([eye, eye], axis=0)
        b = np.concatenate([
            np.zeros(n, np.float32), -np.ones(n, np.float32)
        ])
        h = vr(dense(x, par(w), par(b)))
        return dense(h, par(np.concatenate([eye, -eye], axis=1)))

    def stepv(x):
        return vr(emit("vtanh", [affine(x, 256, 0)]))

    def equal_distance(x):
        return vr(affine(x, -63, 1))

    def clip_s(x):
        return emit("min2", [sr(x), one_s])

    def repeat_v(x, n):
        if shapes[x][1] == n:
            return x
        assert shapes[x][1] == 1
        if n not in repeat_weights:
            repeat_weights[n] = par(np.ones((n, 1), np.float32))
        return dense(x, repeat_weights[n])

    def vector_sum(x):
        return dense(x, par(np.ones((1, shapes[x][1]), np.float32)))

    def linear(xs, coefficients):
        n = max(shapes[x][1] for x in xs)
        xs = [repeat_v(x, n) for x in xs]
        eye = np.eye(n, dtype=np.float32)
        w = np.concatenate(
            [eye * coefficient for coefficient in coefficients], axis=1
        )
        return dense(cat(xs), par(w))

    def distance(r1, c1, r2, c2):
        xs = [r1, c1, r2, c2]
        n = max(shapes[x][1] for x in xs)
        xs = [repeat_v(x, n) for x in xs]
        eye = np.eye(n, dtype=np.float32)
        w = np.zeros((4 * n, 4 * n), np.float32)
        w[0:n, 0:n] = eye
        w[0:n, 2*n:3*n] = -eye
        w[n:2*n, 0:n] = -eye
        w[n:2*n, 2*n:3*n] = eye
        w[2*n:3*n, n:2*n] = eye
        w[2*n:3*n, 3*n:4*n] = -eye
        w[3*n:4*n, n:2*n] = -eye
        w[3*n:4*n, 3*n:4*n] = eye
        h = vr(dense(cat(xs), par(w)))
        return dense(h, par(np.concatenate([eye] * 4, axis=1)))

    def point_mask(r, c):
        d = sm(sa(sab(ss(row, r)), sab(ss(col, c))), 63)
        d = sr(ss(d, const(0.15)))
        return sr(ss(one_s, sm(d, 2)))

    def sample(mask, image):
        return emit("gmax", [mul(mask, image)])

    def sample_channels(mask, channels):
        values = [sample(mask, image) for image in channels]
        return values[0] if len(values) == 1 else cat(values)

    def plane(color):
        w = np.zeros((1, 16, 1, 1), np.float32)
        w[0, color, 0, 0] = 1
        return conv("x0", par(w))

    def channel_sum(x):
        return conv(x, par(np.ones(
            (1, shapes[x][1], 1, 1), np.float32
        )))

    def corners(image):
        w = np.zeros((1, 1, 3, 3), np.float32)
        w[0, 0, 1, 1] = 1
        w[0, 0, 0, 1] = -1
        w[0, 0, 1, 0] = -1
        return sr(conv(image, par(w)))

    def choose_point(image, offset, target_r, target_c, tracking):
        rr = sa(row, const(offset / 63))
        cc = sa(col, const(offset / 63))
        d = sm(sa(sab(ss(rr, target_r)), sab(ss(cc, target_c))), 63)
        priority = sa(sm(cc, 1.89), sm(rr, 0.063))
        initial = sa(sm(cc, 63), sm(rr, 0.63))
        cost = sa(
            mul(d, tracking), mul(initial, vs(one_v, tracking))
        )
        cost = sa(cost, priority)
        score = mul(image, ss(const(1000), cost))
        peak = emit("gmax", [score])
        win = mul(
            sr(sa(sm(ss(score, peak), 2048), one_s)), image
        )
        r = emit("gmax", [mul(win, rr)])
        c = emit("gmax", [mul(win, cc)])
        return r, c, score

    def make_loop(initials, builder, iterations):
        nonlocal dest, local_cse
        lid = fresh("L")
        specs = []
        state_names = []
        for init in initials:
            sid = fresh("s")
            shapes[sid] = shapes[init]
            specs.append({"id": sid, "init": init})
            state_names.append(sid)

        body = []
        old_dest, old_cse = dest, local_cse
        dest, local_cse = body, {}
        next_values = builder(state_names)
        dest, local_cse = old_dest, old_cse

        nd = {
            "id": lid, "op": "loop", "T": int(iterations),
            "state": specs, "body": body,
            "next": dict(zip(state_names, next_values)),
        }
        dest.append(nd)
        aliases = []
        for sid in state_names:
            alias = lid + "." + sid
            shapes[alias] = shapes[sid]
            aliases.append(alias)
        return aliases, nd

    # Detect wall/padding colours rather than hard-code level palettes.
    wall_a = sample(
        point_mask(const(1 / 63), const(1 / 63)), "x0"
    )
    wall_b = sample(
        point_mask(const(1 / 63), const(62 / 63)), "x0"
    )
    wall_flags = vmax(wall_a, wall_b)

    def wall_subset(colors):
        w = np.zeros((len(colors), 16), np.float32)
        for j, color in enumerate(colors):
            w[j, color] = 1
        return dense(wall_flags, par(w))

    aw = np.zeros((2, 16, 1, 1), np.float32)
    aw[0, 10, 0, 0] = 1
    aw[1, 1, 0, 0] = 1
    actor_planes = mul(
        conv("x0", par(aw)),
        vs(one_v, wall_subset([10, 1]))
    )
    actor_image = channel_sum(actor_planes)
    selected_mode = emit("gmax", [sl(actor_planes, 1, 2)])
    actor_corners = corners(actor_image)

    full5 = emit("gmax", [conv(
        actor_image, par(np.ones((1, 1, 5, 5), np.float32))
    )])
    is5 = clipv(va(full5, const(-24)))
    pitch = va(const(4), is5)
    delta = vm(pitch, 1 / 63)

    colors = [14, 12, 15]
    mw = np.zeros((3, 16, 1, 1), np.float32)
    for j, color in enumerate(colors):
        mw[j, color, 0, 0] = 1
    mechanisms = mul(
        conv("x0", par(mw)), vs(one_v, wall_subset(colors))
    )
    count_w = np.zeros((3, 3, 7, 7), np.float32)
    for j in range(3):
        count_w[j, j] = 1
    local_count = conv(mechanisms, par(count_w))
    isolated = clip_s(ss(
        va(const(17), vm(is5, 9)), local_count
    ))
    plates = mul(mechanisms, isolated)
    gates = ss(mechanisms, plates)
    plate_channels = [sl(plates, j, j + 1) for j in range(3)]
    gate_channels = [sl(gates, j, j + 1) for j in range(3)]

    block9 = plane(9)
    block_inflated = clip_s(conv(
        block9, par(np.ones((1, 1, 3, 3), np.float32))
    ))
    walk = sr(ss(clip_s(sa(
        sa(plane(5), actor_image), channel_sum(mechanisms)
    )), block_inflated))
    row_pixels = sm(row, 63)
    top_bar = sr(ss(one_s, row_pixels))
    bottom_bar = sr(ss(row_pixels, const(62)))
    walk = mul(walk, ss(ss(one_s, top_bar), bottom_bar))

    m = [sl("mem", j, j + 1) for j in range(18)]
    track = m[11]

    merged_distance = sa(
        sab(ss(sa(row, const(2 / 63)), m[8])),
        sab(ss(sa(col, const(2 / 63)), m[9]))
    )
    merged_near = sr(ss(
        one_s, sm(merged_distance, 63 / 3.1)
    ))
    actor_corners = mul(
        actor_corners, ss(one_s, mul(merged_near, m[10]))
    )
    ar, ac, _ = choose_point(
        actor_corners, 2, m[0], m[1], track
    )
    first_near = sr(ss(one_s, sm(sa(
        sab(ss(sa(row, const(2 / 63)), ar)),
        sab(ss(sa(col, const(2 / 63)), ac))
    ), 63)))
    br, bc, _ = choose_point(
        mul(actor_corners, ss(one_s, first_near)),
        2, m[2], m[3], track
    )

    horizontal_previous = vs(
        sl("aprev", 4, 5), sl("aprev", 3, 4)
    )
    vertical_previous = vs(
        sl("aprev", 2, 3), sl("aprev", 1, 2)
    )

    def learned_sign(position, old_position, old_sign, default, axis):
        fallback = va(
            vmul(old_sign, track),
            vmul(const(default), vs(one_v, track))
        )
        evidence = vm(vmul(vs(position, old_position), axis), 63)
        sign = emit("vtanh", [vm(evidence, 16)])
        changed = vmul(sign, sign)
        return va(sign, vmul(fallback, vs(one_v, changed)))

    ah = learned_sign(ac, m[1], m[4], 1, horizontal_previous)
    av = learned_sign(ar, m[0], m[5], 1, vertical_previous)
    bh = learned_sign(bc, m[3], m[6], -1, horizontal_previous)
    bv = learned_sign(br, m[2], m[7], 1, vertical_previous)

    # Geodesic landmark fields, including optimistic gate crossings.
    gw = np.zeros((4, 3, 1, 1), np.float32)
    for j, (ry, cx) in enumerate(
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    ):
        gw[j, 0, 0, 0] = 63 if ry else -63
        gw[j, 1, 0, 0] = 63 if cx else -63
        gw[j, 2, 0, 0] = 200 - 63 * (ry + cx)
    corner_scores = mul(conv("coords", par(gw)), walk)
    corner_peak = emit("gmax", [corner_scores])
    corner_seeds = mul(sr(sa(
        sm(ss(corner_scores, corner_peak), 1024), one_s
    )), walk)
    seeds = sm(cat([corner_seeds, plates]), 512)

    shift_weights = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        w = np.zeros((7, 7, 3, 3), np.float32)
        for j in range(7):
            w[j, j, 1 + dy, 1 + dx] = 1
        shift_weights.append(par(w))

    def field_body(states):
        h = states[0]
        neighbors = [conv(h, w) for w in shift_weights]
        q = emit("max2", [neighbors[0], neighbors[1]])
        q = emit("max2", [q, neighbors[2]])
        q = emit("max2", [q, neighbors[3]])
        q = mul(sr(ss(q, one_v)), walk)
        q = emit("max2", [q, h])
        return [emit("max2", [q, seeds])]

    field_out, field_loop = make_loop([seeds], field_body, 128)
    field_channels = [
        sl(field_out[0], j, j + 1) for j in range(7)
    ]

    # One counterfactual trajectory per possible first move.
    N = 4
    K = 4 * N
    starts = [repeat_v(x, N) for x in [ar, ac, br, bc]]
    old_starts = [
        repeat_v(va(
            vmul(old, track),
            vmul(current, vs(one_v, track))
        ), N)
        for old, current in zip(m[:4], [ar, ac, br, bc])
    ]
    drow = vc(np.repeat([-1, 1, 0, 0], N))
    dcol = vc(np.repeat([0, 0, -1, 1], N))
    adr = vmul(drow, vmul(delta, av))
    adc = vmul(dcol, vmul(delta, ah))
    bdr = vmul(drow, vmul(delta, bv))
    bdc = vmul(dcol, vmul(delta, bh))

    forced = vc(np.eye(4, dtype=np.float32).reshape(-1))
    forced_parts = [
        sl(forced, j * N, (j + 1) * N) for j in range(4)
    ]
    direction_priority = vc(np.repeat([0.09, 0.06, 0.03, 0], N))
    clock = vr(vs(
        one_v, vab(vs(vc(np.arange(32)), m[16]))
    ))
    clock_w = par(np.roll(np.eye(32, dtype=np.float32), 1, axis=0))
    noise_array = np.empty((K, 32), np.float32)
    for action in range(4):
        for trajectory in range(N):
            for t in range(32):
                noise_array[action * N + trajectory, t] = (
                    3.0 * math.sin(
                        (t + 1) * (trajectory + 2) * 1.61803398875
                        + (action + 1) * 2.41421356237
                    )
                )
    noise_w = par(noise_array)

    eye_k = np.eye(K, dtype=np.float32)
    closed_reduce_w = par(np.concatenate([eye_k] * 3, axis=1))

    def broadcast_actions(x):
        return cat([x] * 4)

    def split_actions(x):
        return [sl(x, j * N, (j + 1) * N) for j in range(4)]

    def choose_actions(values):
        parts = split_actions(values)
        peak = parts[0]
        for part in parts[1:]:
            peak = vmax(peak, part)
        return [
            vr(affine(vs(part, peak), 1024, 1))
            for part in parts
        ]

    def select_actions(values, winners):
        terms = [
            vmul(part, winner)
            for part, winner in zip(split_actions(values), winners)
        ]
        result = terms[0]
        for term in terms[1:]:
            result = va(result, term)
        return result

    initials = starts + old_starts + [
        repeat_v(const(0), N),
        repeat_v(const(-10000), N),
        one_v, const(0), clock
    ]

    def rollout_body(st):
        a_r, a_c, b_r, b_c = st[:4]
        previous = st[4:8]
        first_goal, best, first, time, phase_clock = st[8:]

        pa = sample_channels(point_mask(a_r, a_c), plate_channels)
        pb = sample_channels(point_mask(b_r, b_c), plate_channels)
        active = vmax(pa, pb)
        inactive = vs(one_v, cat([
            broadcast_actions(sl(active, j * N, (j + 1) * N))
            for j in range(3)
        ]))

        old = [broadcast_actions(x) for x in [a_r, a_c, b_r, b_c]]
        proposed = [
            va(old[0], adr), va(old[1], adc),
            va(old[2], bdr), va(old[3], bdc)
        ]

        def permission(mask):
            gv = sample_channels(mask, gate_channels)
            closed = dense(vmul(gv, inactive), closed_reduce_w)
            return vr(vs(sample(mask, walk), closed))

        allow_a = permission(point_mask(proposed[0], proposed[1]))
        allow_b = permission(point_mask(proposed[2], proposed[3]))
        new = [
            va(old[0], vmul(allow_a, vs(proposed[0], old[0]))),
            va(old[1], vmul(allow_a, vs(proposed[1], old[1]))),
            va(old[2], vmul(allow_b, vs(proposed[2], old[2]))),
            va(old[3], vmul(allow_b, vs(proposed[3], old[3])))
        ]

        fa = sample_channels(point_mask(new[0], new[1]), field_channels)
        fb = sample_channels(point_mask(new[2], new[3]), field_channels)
        fd = vab(vs(fa, fb))
        metric = sl(fd, 0, K)
        for j in range(1, 7):
            metric = vmax(metric, sl(fd, j * K, (j + 1) * K))
        pair_distance = distance(*new)
        metric = vmax(metric, vm(pair_distance, 63))
        coincident = equal_distance(pair_distance)

        row_equal = equal_distance(vab(vs(old[0], old[2])))
        old_dx = vs(old[1], old[3])
        new_dx = vs(new[1], new[3])
        crossing = vmul(
            clipv(vm(neg(vmul(old_dx, new_dx)), 4096)), row_equal
        )
        goal = vmax(coincident, crossing)

        previous = [broadcast_actions(x) for x in previous]
        backwards = equal_distance(va(
            distance(new[0], new[1], previous[0], previous[1]),
            distance(new[2], new[3], previous[2], previous[3])
        ))
        stationary = equal_distance(va(
            distance(new[0], new[1], old[0], old[1]),
            distance(new[2], new[3], old[2], old[3])
        ))
        quality = linear(
            [metric, goal, backwards, stationary,
             dense(phase_clock, noise_w)],
            [-1, 1000, -6, -4, 1]
        )
        quality = va(quality, direction_priority)
        greedy = choose_actions(quality)
        not_first = vs(one_v, first)
        winners = [
            va(vmul(first, f), vmul(not_first, g))
            for f, g in zip(forced_parts, greedy)
        ]
        selected = [select_actions(x, winners) for x in new]
        selected_goal = select_actions(goal, winners)
        selected_metric = select_actions(metric, winners)
        outcome = linear(
            [selected_goal, selected_metric, time], [1000, -1, -0.5]
        )
        saved_goal = va(
            vmul(first, selected_goal), vmul(not_first, first_goal)
        )
        return selected + [a_r, a_c, b_r, b_c] + [
            saved_goal, vmax(best, outcome), const(0),
            va(time, one_v), dense(phase_clock, clock_w)
        ]

    rollout_out, rollout_loop = make_loop(
        initials, rollout_body, 32
    )
    action_quality = va(
        rollout_out[9], vc([0.009, 0.006, 0.003, 0])
    )
    action_peak = sl(action_quality, 0, 1)
    for j in range(1, 4):
        action_peak = vmax(action_peak, sl(action_quality, j, j + 1))
    action_win = vr(affine(vs(action_quality, action_peak), 1024, 1))
    predicted_merge = vector_sum(vmul(rollout_out[8], action_win))

    # A bounded selectable-block clearing fallback.
    current_distance = vm(distance(ar, ac, br, bc), 63)
    not_improved = stepv(va(
        vs(current_distance, m[17]), const(0.25)
    ))
    stagnation = vmul(
        vmul(not_improved, track), va(m[12], one_v)
    )
    block_r, block_c, block_click_score = choose_point(
        corners(block9), 1, ar, ac, one_v
    )
    has_block = emit("gmax", [block9])
    timer_active = stepv(va(m[13], const(-0.5)))
    timer_finished = stepv(va(m[13], const(-5.5)))
    trigger = vmul(
        stepv(va(stagnation, const(-9.5))),
        vmul(has_block, vs(one_v, timer_active))
    )
    trigger = vmul(trigger, vs(one_v, selected_mode))
    block_control = vmul(
        timer_active,
        vmul(selected_mode, vs(one_v, timer_finished))
    )
    do_click = vmax(
        trigger, vmul(selected_mode, vs(one_v, block_control))
    )

    block_walk = sr(ss(ss(
        walk, actor_image
    ), channel_sum(mechanisms)))
    normal_dr = vmul(vc([-1, 1, 0, 0]), delta)
    normal_dc = vmul(vc([0, 0, -1, 1]), delta)
    bp_r = va(repeat_v(m[14], 4), normal_dr)
    bp_c = va(repeat_v(m[15], 4), normal_dc)
    bp_allow = sample(point_mask(bp_r, bp_c), block_walk)
    dr, dc = vs(bp_r, ar), vs(bp_c, ac)
    away_squared = va(vmul(dr, dr), vmul(dc, dc))
    block_quality = va(
        va(vm(bp_allow, 100), vm(away_squared, 20)),
        vc([0.009, 0.006, 0.003, 0])
    )
    block_peak = sl(block_quality, 0, 1)
    for j in range(1, 4):
        block_peak = vmax(block_peak, sl(block_quality, j, j + 1))
    block_win = vr(affine(vs(block_quality, block_peak), 1024, 1))
    bp_r = va(m[14], vmul(bp_allow, vs(bp_r, m[14])))
    bp_c = va(m[15], vmul(bp_allow, vs(bp_c, m[15])))
    block_next_r = vector_sum(vmul(bp_r, block_win))
    block_next_c = vector_sum(vmul(bp_c, block_win))

    blended_quality = va(
        vmul(action_quality, vs(one_v, block_control)),
        vmul(block_quality, block_control)
    )
    output_w = np.zeros((8, 4), np.float32)
    for j in range(4):
        output_w[j + 1, j] = 1
    movement_logits = va(
        dense(blended_quality, par(output_w)),
        vc([-10000, 0, 0, 0, 0, -10000, -10000, -10000])
    )
    click_logits = vc([
        -10000, -10000, -10000, -10000,
        -10000, -10000, 10000, -10000
    ])
    emit("vadd", [
        va(
            vmul(movement_logits, vs(one_v, do_click)),
            vmul(click_logits, do_click)
        ), const(0)
    ], name="logits")
    emit("add", [
        sa(
            mul(block_click_score, trigger),
            mul(actor_image, vs(one_v, trigger))
        ), const(0)
    ], name="click")

    actor_move = vmul(
        vs(one_v, do_click), vs(one_v, block_control)
    )
    merge = clipv(vmul(predicted_merge, actor_move))
    no_merge = vs(one_v, merge)
    merge_r = linear([ar, br], [0.5, 0.5])
    merge_c = linear([ac, bc], [0.5, 0.5])
    remembered_r = va(vmul(merge, merge_r), vmul(no_merge, m[8]))
    remembered_c = va(vmul(merge, merge_c), vmul(no_merge, m[9]))

    new_timer = va(
        vmul(block_control, va(m[13], one_v)), trigger
    )
    not_control = vs(one_v, block_control)
    not_trigger = vs(one_v, trigger)
    new_block_r = va(
        vmul(trigger, block_r),
        vmul(not_trigger, va(
            vmul(block_control, block_next_r),
            vmul(not_control, m[14])
        ))
    )
    new_block_c = va(
        vmul(trigger, block_c),
        vmul(not_trigger, va(
            vmul(block_control, block_next_c),
            vmul(not_control, m[15])
        ))
    )
    new_stagnation = vmul(
        stagnation, vs(one_v, vmax(trigger, timer_active))
    )
    next_phase = vs(
        va(m[16], one_v),
        vm(vr(va(m[16], const(-30))), 32)
    )
    memory = cat([
        ar, ac, br, bc, ah, av, bh, bv,
        remembered_r, remembered_c, vmax(m[10], merge), no_merge,
        new_stagnation, new_timer, new_block_r, new_block_c,
        next_phase, current_distance
    ])
    emit("vadd", [memory, const(0)], name="mem_next")

    # Dead-code elimination, including loop-body dependencies.
    def prune_body(nodes, wanted):
        wanted = set(wanted)
        kept = []
        for nd in reversed(nodes):
            if nd["id"] in wanted:
                kept.append(nd)
                wanted.update(nd["in"])
        kept.reverse()
        return kept, wanted

    wanted = {"logits", "click", "mem_next"}
    kept = []
    for nd in reversed(graph):
        if nd["op"] == "loop":
            prefix = nd["id"] + "."
            if not any(x.startswith(prefix) for x in wanted):
                continue
            body, dependencies = prune_body(
                nd["body"], nd["next"].values()
            )
            nd["body"] = body
            local_ids = {x["id"] for x in body}
            local_ids.update(x["id"] for x in nd["state"])
            wanted.update(dependencies - local_ids)
            wanted.update(x["init"] for x in nd["state"])
            kept.append(nd)
        elif nd["id"] in wanted:
            kept.append(nd)
            wanted.update(nd["in"])
    graph = list(reversed(kept))

    def flatten(nodes):
        result = []
        for nd in nodes:
            result.append(nd)
            if nd["op"] == "loop":
                result.extend(flatten(nd["body"]))
        return result

    all_nodes = flatten(graph)
    used = set()
    for nd in all_nodes:
        for field in ("w", "b"):
            if field in nd:
                used.add(nd[field])
    P = {name: value for name, value in P.items() if name in used}

    def cost(nd):
        op = nd["op"]
        if op == "loop":
            bm = sum(cost(x)[0] for x in nd["body"])
            be = sum(cost(x)[1] for x in nd["body"])
            return nd["T"] * bm, 1 + nd["T"] * be
        sh = shapes[nd["id"]]
        if op == "conv":
            return int(P[nd["w"]].size) * 4096, 1
        if op == "dense":
            return int(P[nd["w"]].size), 1
        if op in ("add", "sub", "mul", "max2", "min2"):
            return sh[1] * 4096, 1
        if op in (
            "gmax", "gmean", "rowmax", "colmax", "rowsum", "colsum"
        ):
            return shapes[nd["in"][0]][1] * 4096, 1
        if op in ("vadd", "vmul"):
            return sh[1], 1
        return 0, 1

    # Select the static unrolling count from architectural costs only.
    body_mac = sum(cost(x)[0] for x in rollout_loop["body"])
    body_eval = sum(cost(x)[1] for x in rollout_loop["body"])
    rollout_loop["T"] = 1
    base_mac = sum(cost(x)[0] for x in graph) - body_mac
    base_eval = sum(cost(x)[1] for x in graph) - body_eval
    horizon = min(
        32,
        int((1_490_000_000 - base_mac) // max(body_mac, 1)),
        int((9_950 - base_eval) // max(body_eval, 1))
    )
    assert horizon >= 1, ("horizon", base_mac, base_eval)
    rollout_loop["T"] = horizon

    spatial_channels = 19
    for nd in all_nodes:
        if nd["op"] == "loop":
            for state in nd["state"]:
                sh = shapes[state["id"]]
                if sh[0] == "s":
                    spatial_channels += sh[1]
        else:
            sh = shapes[nd["id"]]
            if sh[0] == "s":
                spatial_channels += sh[1]

    parameter_count = sum(a.size for a in P.values())
    macs = sum(cost(x)[0] for x in graph)
    evaluations = sum(cost(x)[1] for x in graph)
    assert len(all_nodes) <= 1000, ("nodes", len(all_nodes))
    assert spatial_channels <= 4096, ("channels", spatial_channels)
    assert parameter_count <= 150000, ("parameters", parameter_count)
    assert macs <= 1_500_000_000, ("MACs", macs)
    assert evaluations <= 10000, ("evaluations", evaluations)
    assert all(s[1] <= 512 for s in shapes.values() if s[0] == "s")
    assert all(s[1] <= 4096 for s in shapes.values() if s[0] == "v")
    assert all(np.isfinite(a).all() for a in P.values())

    return {"graph": graph, "memory_size": 18, "params": P}
