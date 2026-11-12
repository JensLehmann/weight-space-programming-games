import numpy as np


def build():
    P = {}
    cache = {}
    graph = []
    current = graph
    serial = 0
    shapes = {
        "x0": (16, True),
        "coords": (3, True),
        "mem": (8, False),
        "aprev": (8, False),
        "avail": (8, False),
    }
    costs = {}

    def parameter(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p" + str(len(P))
            cache[key] = name
            P[name] = a
        return cache[key]

    def node(op, ins, w=None, b=None, name=None, **fields):
        nonlocal serial
        if name is None:
            name = "n" + str(serial)
            serial += 1
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        if w is not None:
            w = np.asarray(w, np.float32)
            nd["w"] = parameter(w)
        if b is not None:
            nd["b"] = parameter(b)

        c, spatial = shapes[ins[0]]
        mac = 0
        if op == "conv":
            out = (int(w.shape[0]), True)
            mac = int(w.size) * 4096
        elif op == "dense":
            out = (int(w.shape[0]), False)
            mac = int(w.size)
        elif op in ("slice", "vslice"):
            out = (fields["stop"] - fields["start"], spatial)
        elif op in ("concat", "vconcat"):
            out = (sum(shapes[i][0] for i in ins), spatial)
        elif op in ("gmax", "gmean"):
            out = (c, False)
            mac = c * 4096
        elif op in ("rowmax", "colmax", "rowsum", "colsum"):
            out = (c, True)
            mac = c * 4096
        elif op in ("add", "sub", "mul", "max2", "min2"):
            out = (max(shapes[i][0] for i in ins), True)
            mac = out[0] * 4096
        elif op in ("vadd", "vmul"):
            out = (max(shapes[i][0] for i in ins), False)
            mac = out[0]
        else:
            out = (c, spatial)
        shapes[name] = out
        costs[name] = mac
        current.append(nd)
        return name

    def conv(x, w, b=None, name=None):
        return node("conv", [x], w=w, b=b, name=name)

    def lin(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        return conv(x, w[:, :, None, None], b, name)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        return node("dense", [x], w=w, b=b, name=name)

    def sl(x, a, b):
        return node("slice" if shapes[x][1] else "vslice",
                    [x], start=a, stop=b)

    def cat(xs):
        return node("concat" if shapes[xs[0]][1] else "vconcat", xs)

    def relu(x):
        return node("relu" if shapes[x][1] else "vrelu", [x])

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def affine(x, scale=1.0, bias=0.0):
        c = shapes[x][0]
        w = np.eye(c, dtype=np.float32) * scale
        b = np.full(c, bias, np.float32)
        return lin(x, w, b) if shapes[x][1] else dense(x, w, b)

    one = dense("avail", np.zeros((1, 8)), [1])
    negone = dense("avail", np.zeros((1, 8)), [-1])
    space_one = sl("coords", 2, 3)
    space_zero = affine(space_one, 0, 0)

    def clip(x):
        return node("min2", [relu(x), one])

    def vclip(x):
        r = relu(x)
        excess = relu(va(r, negone))
        return va(r, affine(excess, -1, 0))

    def vmaxsum(x):
        return dense(x, np.ones((1, shapes[x][0])))

    def maximum_channels(x):
        parts = [sl(x, i, i + 1) for i in range(shapes[x][0])]
        z = parts[0]
        for q in parts[1:]:
            z = mx(z, q)
        return z

    def shift_scalar(x, dy, dx):
        k = max(1, 2 * max(abs(dy), abs(dx)) + 1)
        w = np.zeros((1, 1, k, k), np.float32)
        w[0, 0, k // 2 + dy, k // 2 + dx] = 1
        return conv(x, w)

    colors = [sl("x0", i, i + 1) for i in range(16)]
    ycoord = lin("coords", [[63, 0, 0]])
    xcoord = lin("coords", [[0, 63, 0]])
    inv_y = affine(ycoord, -1, 63)
    inv_x = affine(xcoord, -1, 63)

    # Spell availability is read from the visible UI icons.
    near_ui = clip(conv(colors[3], np.ones((1, 1, 7, 7))))
    spell_present = {}
    for spell, color in (("grow", 15), ("tele", 11), ("fire", 6)):
        spell_present[spell] = node(
            "gmax", [mul(colors[color], near_ui)]
        )

    # Exact exit filters in all four orientations.
    base_exit = np.full((6, 5), 10, dtype=np.int32)
    base_exit[0, :] = 9
    base_exit[-1, :] = 9
    base_exit[:, 0] = 9
    ew = np.zeros((4, 16, 7, 7), np.float32)
    ep = np.zeros((1, 4, 7, 7), np.float32)
    for r in range(4):
        a = np.rot90(base_exit, -r)
        for yy in range(a.shape[0]):
            for xx in range(a.shape[1]):
                ew[r, a[yy, xx], yy + 1, xx + 1] = 1
                ep[0, r, 5 - yy, 5 - xx] = 1
    exit_anchors = relu(conv("x0", ew, np.full(4, -29)))
    exit_pixels = clip(conv(exit_anchors, ep))

    # Exact player filters, excluding the exit and duplicate small
    # matches within the large player.
    def player_detector(size, extras):
        k = 3 if size == 2 else 7
        h = k // 2
        base = np.empty((size, size), dtype=np.int32)
        base[:size // 2, :] = 10
        base[size // 2:, :] = 9
        w = np.zeros((4, 16 + len(extras), k, k), np.float32)
        for r in range(4):
            a = np.rot90(base, -r)
            for yy in range(size):
                for xx in range(size):
                    w[r, a[yy, xx], h + yy, h + xx] = 1
            for j in range(len(extras)):
                w[r, 16 + j, h, h] = -size * size
        inp = cat(["x0"] + extras)
        hits = relu(conv(inp, w, np.full(4, 1 - size * size)))
        return lin(hits, np.ones((1, 4)))

    player_large = player_detector(4, [exit_pixels])
    paint4 = np.zeros((1, 1, 7, 7), np.float32)
    for yy in range(4):
        for xx in range(4):
            paint4[0, 0, 3 - yy, 3 - xx] = 1
    large_pixels = clip(conv(player_large, paint4))
    player_small = player_detector(2, [exit_pixels, large_pixels])

    # Switches and their corresponding removable gates.
    sw = np.zeros((2, 16, 7, 7), np.float32)
    for g, inner in enumerate((13, 12)):
        for yy in range(4):
            for xx in range(4):
                color = inner if 1 <= yy <= 2 and 1 <= xx <= 2 else 6
                sw[g, color, 3 + yy, 3 + xx] = 1
    switch_anchors = relu(conv("x0", sw, [-15, -15]))
    sp = np.zeros((2, 2, 7, 7), np.float32)
    sp[0, 0] = paint4[0, 0]
    sp[1, 1] = paint4[0, 0]
    switches = conv(switch_anchors, sp)
    switch_masks = [sl(switches, g, g + 1) for g in range(2)]
    removable = [
        mx(colors[13], switch_masks[0]),
        mx(colors[12], switch_masks[1]),
    ]
    gate_only = [
        relu(sub(colors[13], switch_masks[0])),
        relu(sub(colors[12], switch_masks[1])),
    ]

    # Teleport destinations for both sizes.
    tw = np.zeros((1, 16, 3, 3), np.float32)
    tw[0, 15, 1:3, 1:3] = 1
    small_dots = relu(conv("x0", tw, [-3]))
    small_dots = mul(small_dots, affine(near_ui, -1, 1))

    def six_corners(color):
        w = np.zeros((1, 16, 7, 7), np.float32)
        for yy in (-2, 3):
            for xx in (-2, 3):
                w[0, color, 3 + yy, 3 + xx] = 1
        return relu(conv("x0", w, [-3]))

    small_active = six_corners(15)
    target_small = clip(mx(small_dots, small_active))

    lw = np.zeros((1, 16, 7, 7), np.float32)
    for yy in (0, 3):
        for xx in (0, 3):
            lw[0, 11, 3 + yy, 3 + xx] = 1
    lw[0, 15, 4, 4] = -4
    large_dots = relu(conv("x0", lw, [-3]))
    small_inside = shift_scalar(target_small, 1, 1)
    large_dots = relu(sub(large_dots, small_inside))
    large_active = shift_scalar(six_corners(11), 1, 1)
    target_large = clip(mx(large_dots, large_active))

    fence_w = np.zeros((1, 1, 5, 5), np.float32)
    for yy in (-1, 2):
        for xx in (-1, 2):
            fence_w[0, 0, 2 - yy, 2 - xx] = 1
    small_fence = mul(
        clip(conv(target_small, fence_w)), colors[11]
    )

    other_six = relu(
        lin(cat([colors[6]] + switch_masks), [[1, -1, -1]])
    )
    static_move = clip(lin(
        cat([colors[c] for c in (5, 4, 3, 1, 7, 8)]
            + [other_six, small_fence]),
        np.ones((1, 8))
    ))
    static_ray = clip(lin(
        cat([colors[5], colors[7], colors[8]]), [[1, 1, 1]]
    ))

    def free_rectangle(blocked, size):
        free_pixels = relu(affine(blocked, -1, 1))
        k = 3 if size == 2 else 7
        w = np.zeros((1, 1, k, k), np.float32)
        h = k // 2
        w[0, 0, h:h + size, h:h + size] = 1
        return relu(conv(free_pixels, w, [1 - size * size]))

    def overlaps_rectangle(field, size):
        k = 3 if size == 2 else 7
        w = np.zeros((1, 1, k, k), np.float32)
        h = k // 2
        w[0, 0, h:h + size, h:h + size] = 1
        return clip(conv(field, w))

    goals = [
        overlaps_rectangle(exit_pixels, 2),
        overlaps_rectangle(exit_pixels, 4),
    ]

    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    move2_w = np.zeros((4, 1, 5, 5), np.float32)
    move_again_w = np.zeros((4, 4, 5, 5), np.float32)
    for d, (dy, dx) in enumerate(directions):
        move2_w[d, 0, 2 + dy, 2 + dx] = 1
        move_again_w[d, d, 2 + dy, 2 + dx] = 1

    grow_offsets = [(dy, dx)
                    for dx in (-2, -1, 0)
                    for dy in (-2, -1, 0)]
    grow_w = np.zeros((9, 1, 5, 5), np.float32)
    for i, (dy, dx) in enumerate(grow_offsets):
        grow_w[i, 0, 2 + dy, 2 + dx] = 1
    first_w = np.eye(9, dtype=np.float32)
    for i in range(9):
        first_w[i, :i] = -1

    source_free = []
    first_free4 = []
    first_block4 = []
    grow_choice = []
    ray_blockers = []

    for bits in range(4):
        remaining = [removable[g] for g in range(2)
                     if not (bits & (1 << g))]
        mb = clip(lin(cat([static_move, exit_pixels] + remaining),
                      np.ones((1, 2 + len(remaining)))))
        fs = free_rectangle(mb, 2)
        fl = free_rectangle(mb, 4)
        source_free.extend([fs, fl])

        f2 = conv(fl, move2_w)
        f4 = conv(f2, move_again_w)
        first_free4.append(f4)
        first_block4.append(affine(f4, -1, 1))

        gates = [gate_only[g] for g in range(2)
                 if not (bits & (1 << g))]
        gb = clip(lin(cat([static_ray] + gates),
                      np.ones((1, 1 + len(gates)))))
        grow_fit = free_rectangle(gb, 4)
        grow_options = conv(grow_fit, grow_w)
        grow_choice.append(relu(lin(grow_options, first_w)))

        rb = clip(lin(cat([static_ray] + remaining),
                      np.ones((1, 1 + len(remaining)))))
        ray_blockers.append(rb)

    # Directional line-of-sight masks for the fire spell.
    axes = [inv_y, ycoord, inv_x, xcoord]
    reductions = ["colmax", "colmax", "rowmax", "rowmax"]
    ray_pre = {}
    for g in range(2):
        for d in range(4):
            t = axes[d]
            op = reductions[d]
            target = switch_masks[g]
            score = lin(cat([t, target]), [[-1, 128]], [-128])
            target_min = affine(node(op, [score]), -1, 0)
            before = clip(sub(target_min, t))
            present = node(op, [target])
            ray_pre[g, d] = (before, present)

    fire_dirs = {}
    fire_any = {}
    for bits in range(4):
        for g in range(2):
            if bits & (1 << g):
                continue
            ds = []
            for d in range(4):
                before, present = ray_pre[g, d]
                bb = mul(before, ray_blockers[bits])
                score = relu(lin(cat([axes[d], bb]),
                                 [[1, 128]], [-127]))
                boundary = node(reductions[d], [score])
                clear = clip(lin(cat([axes[d], boundary]),
                                 [[1, -1]], [1]))
                ds.append(mul(mul(clear, before), present))
            field = cat(ds)
            fire_dirs[bits, g] = mul(field, spell_present["fire"])
            fire_any[bits, g] = maximum_channels(field)

    free_all = cat(source_free)
    goal_all = affine(cat([goals[i % 2] for i in range(8)]), 256, 0)
    target_all = cat([target_small if i % 2 == 0 else target_large
                      for i in range(8)])

    fire_masks = []
    successor_w = []
    successor_b = []
    for g in range(2):
        ms = []
        w = np.zeros((8, 8), np.float32)
        b = np.zeros(8, np.float32)
        for bits in range(4):
            allowed = not (bits & (1 << g))
            for s in range(2):
                i = 2 * bits + s
                if allowed:
                    ms.append(fire_any[bits, g])
                    w[i, 2 * (bits | (1 << g)) + s] = 1
                    b[i] = -4
                else:
                    ms.append(space_zero)
        fire_masks.append(mul(cat(ms), spell_present["fire"]))
        successor_w.append(w)
        successor_b.append(b)

    def movement(value, bits, large):
        d2 = conv(value, move2_w)
        if not large:
            return d2
        d4 = conv(d2, move_again_w)
        return add(mul(d4, first_free4[bits]),
                   mul(d2, first_block4[bits]))

    def growth(value_large, bits):
        shifted = conv(value_large, grow_w)
        chosen = mul(shifted, grow_choice[bits])
        return lin(chosen, np.ones((1, 9)))

    def grouped_four_max(x, groups):
        a = np.zeros((groups * 2, groups * 4), np.float32)
        b = np.zeros_like(a)
        for g in range(groups):
            for j in range(2):
                a[2 * g + j, 4 * g + 2 * j] = 1
                b[2 * g + j, 4 * g + 2 * j + 1] = 1
        pairs = mx(lin(x, a), lin(x, b))
        a = np.zeros((groups, groups * 2), np.float32)
        b = np.zeros_like(a)
        for g in range(groups):
            a[g, 2 * g] = 1
            b[g, 2 * g + 1] = 1
        return mx(lin(pairs, a), lin(pairs, b))

    # Eight value planes: four gate states times two player sizes.
    body = []
    current = body
    shapes["value_state"] = (8, True)
    vs = [sl("value_state", i, i + 1) for i in range(8)]

    moves = [movement(vs[i], i // 2, bool(i % 2))
             for i in range(8)]
    move_best = grouped_four_max(cat(moves), 8)

    resize_fields = []
    for bits in range(4):
        resize_fields.extend([growth(vs[2 * bits + 1], bits),
                              vs[2 * bits]])
    resize_value = mul(
        affine(cat(resize_fields), 1, -4), spell_present["grow"]
    )

    tele_at_target = node(
        "gmax", [mul("value_state", target_all)]
    )
    tele_value = vm(
        affine(tele_at_target, 1, -3), spell_present["tele"]
    )

    fire_values = []
    for g in range(2):
        succ = lin("value_state", successor_w[g], successor_b[g])
        fire_values.append(mul(succ, fire_masks[g]))
    fire_value = mx(fire_values[0], fire_values[1])

    best = mx(mx(move_best, resize_value), mx(fire_value, tele_value))
    best = relu(affine(best, 1, -1))
    next_value = mx(mul(best, free_all), goal_all)

    current = graph
    loop = {
        "id": "plan",
        "op": "loop",
        "T": 1,
        "state": [{"id": "value_state", "init": goal_all}],
        "body": body,
        "next": {"value_state": next_value},
    }
    graph.append(loop)
    shapes["plan.value_state"] = (8, True)

    # Read out movement, resizing, teleporting, and four firing actions.
    final_v = [sl("plan.value_state", i, i + 1) for i in range(8)]
    utilities = []
    for s, player in enumerate((player_small, player_large)):
        mv = movement(final_v[s], 0, bool(s))
        rz = growth(final_v[1], 0) if s == 0 else final_v[0]
        rz = mul(affine(rz, 1, -4), spell_present["grow"])

        target = target_small if s == 0 else target_large
        tv = node("gmax", [mul(final_v[s], target)])
        tv = vm(affine(tv, 1, -3), spell_present["tele"])
        tv = mul(space_one, tv)

        fq = []
        for g in range(2):
            successor = final_v[2 * (1 << g) + s]
            fv = affine(successor, 1, -4)
            fq.append(mul(fv, fire_dirs[0, g]))
        fire_q = mx(fq[0], fq[1])

        q = cat([mv, rz, tv, fire_q])
        utilities.append(node("gmax", [mul(q, player)]))
    utility = va(utilities[0], utilities[1])

    # Deterministic winner circuit.
    utility = dense(utility, np.eye(10),
                    -0.01 * np.arange(10, dtype=np.float32))
    compare_w = np.zeros((100, 10), np.float32)
    for i in range(10):
        for j in range(10):
            compare_w[10 * i + j, j] += 128
            compare_w[10 * i + j, i] -= 128
    greater = vclip(dense(utility, compare_w))
    win_w = np.zeros((10, 100), np.float32)
    for i in range(10):
        win_w[i, 10 * i:10 * i + 10] = -1
    winner = relu(dense(greater, win_w, np.ones(10)))

    select_w = np.zeros((3, 10), np.float32)
    select_w[0, 4] = 1
    select_w[1, 5] = 1
    select_w[2, 6:10] = 1
    selected_spell = dense(winner, select_w)
    chosen_move = sl(winner, 0, 4)
    chosen_fire_direction = sl(winner, 6, 10)

    # Localize the nine canonical spell-grid click positions.
    points = [(50 + 5 * r, 25 + 5 * c)
              for r in range(3) for c in range(3)]
    pw = np.zeros((18, 3), np.float32)
    pb = np.zeros(18, np.float32)
    for i, (yy, xx) in enumerate(points):
        pw[2 * i, 0] = 63
        pw[2 * i + 1, 1] = 63
        pb[2 * i] = -yy
        pb[2 * i + 1] = -xx
    distances = node("abs", [lin("coords", pw, pb)])
    tent_w = np.zeros((9, 18), np.float32)
    for i in range(9):
        tent_w[i, 2 * i:2 * i + 2] = -2
    point_masks = relu(lin(distances, tent_w, np.ones(9)))

    active_color = lin(
        cat([colors[14], colors[6], colors[11], colors[15]]),
        [[1, 1, 1, 1]]
    )
    active = node("gmax", [mul(point_masks, active_color)])
    active = vclip(affine(active, 2, -0.5))
    any_active = vclip(vmaxsum(active))

    old_spell = sl("mem", 0, 3)
    old_direction = sl("mem", 3, 7)
    pending_fire = sl("mem", 7, 8)

    # Preserve a spell while its pattern is being entered, and also
    # immediately after its preparatory direction-setting movement.
    continue_spell = vclip(va(any_active, pending_fire))
    hold = vm(continue_spell, vmaxsum(old_spell))
    release = affine(hold, -1, 1)
    desired_spell = va(vm(hold, old_spell),
                       vm(release, selected_spell))

    previous_move = sl("aprev", 1, 5)
    keep_direction = affine(vmaxsum(previous_move), -1, 1)
    direction = va(previous_move, vm(keep_direction, old_direction))
    facing = vmaxsum(vm(direction, chosen_fire_direction))

    desired_fire = sl(desired_spell, 2, 3)
    turn_needed = vm(vm(desired_fire, release),
                     affine(facing, -1, 1))
    final_moves = va(vm(release, chosen_move),
                     vm(turn_needed, chosen_fire_direction))

    nonfire_ready = dense(desired_spell, [[1, 1, 0]])
    fire_ready = vm(
        desired_fire, va(hold, vm(release, facing))
    )
    click_ready = va(nonfire_ready, fire_ready)

    patterns = np.zeros((9, 3), np.float32)
    patterns[[1, 3, 5, 7], 0] = 1
    patterns[[0, 1, 4], 1] = 1
    patterns[[1, 4, 7], 2] = 1
    desired_bits = dense(desired_spell, patterns)
    common = vm(active, desired_bits)

    # Incorrect active bits are removed before missing bits are added.
    err_w = np.concatenate(
        [2 * np.eye(9), np.eye(9), -3 * np.eye(9)], axis=1
    )
    errors = dense(cat([active, desired_bits, common]), err_w)
    click_scores = mul(point_masks, errors)
    lin(click_scores, np.ones((1, 9)), name="click")

    action_input = cat([final_moves, click_ready])
    action_w = np.zeros((8, 5), np.float32)
    action_b = np.full(8, -100, np.float32)
    for d in range(4):
        action_w[d + 1, d] = 10
        action_b[d + 1] = 0
    action_w[6, 4] = 10
    action_b[6] = 0
    dense(action_input, action_w, action_b, name="logits")

    memory_input = cat([desired_spell, direction, turn_needed])
    dense(memory_input, np.eye(8), name="mem_next")

    # Choose a fixed propagation depth within both execution budgets.
    static_mac = sum(costs[n["id"]] for n in graph if n["op"] != "loop")
    body_mac = sum(costs[n["id"]] for n in body)
    static_evals = len(graph)
    depth_mac = int((1_490_000_000 - static_mac) // body_mac)
    depth_evals = int((9990 - static_evals) // len(body))
    loop["T"] = max(1, min(128, depth_mac, depth_evals))

    return {"graph": graph, "memory_size": 8, "params": P}
