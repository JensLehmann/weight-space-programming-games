import numpy as np


def build():
    H = 16
    HISTORY_START = 14
    PLATE_START = HISTORY_START + 4 * H
    MEMORY_SIZE = PLATE_START + 18

    graph = []
    current = graph
    params = {}
    parameter_cache = {}
    constant_cache = {}
    expression_cache = {}
    constants = {}
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (MEMORY_SIZE,),
        "aprev": (8,),
        "avail": (8,),
    }
    top_names = set(shapes)
    serial = 0

    def parameter(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def node(op, ins=(), name=None, **fields):
        nonlocal serial
        ins = list(ins)
        if op in ("add", "mul", "max2", "min2", "vadd", "vmul"):
            ins.sort()

        if name is None and ins and all(x in constants for x in ins):
            xs = [constants[x] for x in ins]
            result = None
            if op == "vadd":
                result = xs[0] + xs[1]
            elif op == "vmul":
                result = xs[0] * xs[1]
            elif op == "vconcat":
                result = np.concatenate(xs)
            elif op == "vslice":
                result = xs[0][fields["start"]:fields["stop"]]
            elif op == "vrelu":
                result = np.maximum(xs[0], 0)
            elif op == "vtanh":
                result = np.tanh(xs[0])
            elif op == "dense":
                result = params[fields["w"]] @ xs[0]
                if "b" in fields:
                    result = result + params[fields["b"]]
            if result is not None:
                return const(result)

        key = (op, tuple(ins), tuple(sorted(fields.items())))
        if name is None and key in expression_cache:
            return expression_cache[key]
        if name is None:
            name = "n%d" % serial
            serial += 1

        nd = {"id": name, "op": op, "in": ins}
        nd.update(fields)
        if all(x in top_names for x in ins):
            graph.append(nd)
            top_names.add(name)
        else:
            current.append(nd)

        if op in ("dense", "conv"):
            c = params[fields["w"]].shape[0]
            shape = (c,) if op == "dense" else (c, 64, 64)
        elif op in ("slice", "vslice"):
            shape = (fields["stop"] - fields["start"],) + shapes[ins[0]][1:]
        elif op in ("concat", "vconcat"):
            shape = (sum(shapes[x][0] for x in ins),) + shapes[ins[0]][1:]
        elif op in ("gmax", "gmean"):
            shape = (shapes[ins[0]][0],)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = (max(shapes[x][0] for x in ins), 64, 64)
        elif op in ("vadd", "vmul"):
            shape = (max(shapes[x][0] for x in ins),)
        else:
            shape = shapes[ins[0]]
        shapes[name] = shape
        expression_cache[key] = name
        return name

    def const(value):
        a = np.asarray(value, np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key not in constant_cache:
            name = node(
                "dense", ["avail"],
                w=parameter(np.zeros((len(a), 8), np.float32)),
                b=parameter(a))
            constant_cache[key] = name
            constants[name] = a
        return constant_cache[key]

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def cat(*xs):
        return node("concat", xs)

    def vc(*xs):
        return node("vconcat", xs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def sm(x, k):
        return mul(x, const(k))

    def sr(x):
        return node("relu", [x])

    def sa(x):
        return node("abs", [x])

    def smax(a, b):
        return node("max2", [a, b])

    def smin(a, b):
        return node("min2", [a, b])

    def sclip(x):
        return smin(sr(x), const(1))

    def gm(x):
        return node("gmax", [x])

    def va(a, b):
        for z, other in ((a, b), (b, a)):
            if z in constants and np.all(constants[z] == 0):
                if shapes[z][0] == 1 or shapes[z] == shapes[other]:
                    return other
        return node("vadd", [a, b])

    def vm(a, b):
        for z, other in ((a, b), (b, a)):
            if z in constants and np.all(constants[z] == 1):
                if shapes[z][0] == 1 or shapes[z] == shapes[other]:
                    return other
        return node("vmul", [a, b])

    def vmc(x, k):
        if k == 1:
            return x
        if k == 0:
            return const(np.zeros(shapes[x][0], np.float32))
        return vm(x, const(k))

    def aff(x, a=1.0, b=0.0):
        n = shapes[x][0]
        bias = np.asarray(b, np.float32).reshape(-1)
        m = max(n, len(bias))
        if m > 32:
            return va(vmc(x, a), const(bias))
        w = (np.full((m, 1), a, np.float32) if n == 1 else
             np.eye(n, dtype=np.float32) * np.float32(a))
        return dense(x, w, np.broadcast_to(bias, (m,)).copy())

    def neg(x):
        return vmc(x, -1)

    def vd(a, b):
        if b in constants:
            return aff(a, 1, -constants[b])
        if a in constants:
            return aff(b, -1, constants[a])
        return va(a, neg(b))

    def vr(x):
        return node("vrelu", [x])

    def vab(x):
        return vm(x, node("vtanh", [vmc(x, 32)]))

    def cap(x):
        return aff(vr(aff(x, -1, 1)), -1, 1)

    def sat(x):
        return cap(vr(x))

    def zero_test(nonnegative, sharpness=2):
        return vr(aff(nonnegative, -sharpness, 1))

    def vmax(a, b):
        return va(b, vr(vd(a, b)))

    def vmin(a, b):
        return vd(a, vr(vd(a, b)))

    def blend(new, old, gate):
        return va(old, vm(vd(new, old), gate))

    def take(x, indices):
        w = np.zeros((len(indices), shapes[x][0]), np.float32)
        w[np.arange(len(indices)), indices] = 1
        return dense(x, w)

    def reduce_max(x):
        n = shapes[x][0]
        while n > 1:
            left = take(x, list(range(0, n - 1, 2)))
            right = take(x, list(range(1, n, 2)))
            z = vmax(left, right)
            if n % 2:
                z = vc(z, vs(x, n - 1, n))
            x, n = z, (n + 1) // 2
        return x

    def color(c):
        return sl("x0", c, c + 1)

    def channel_sum(x):
        return conv(x, np.ones((1, shapes[x][0], 1, 1), np.float32))

    def diagonal_box(c, k):
        w = np.zeros((c, c, k, k), np.float32)
        for i in range(c):
            w[i, i] = 1
        return w

    one, zero = const(1), const(0)
    row = sm(sl("coords", 0, 1), 63)
    col = sm(sl("coords", 1, 2), 63)
    memory = [vs("mem", i, i + 1) for i in range(14)]

    def point(cx, cy):
        d = smax(sa(sub(col, cx)), sa(sub(row, cy)))
        return sclip(sub(const(2), sm(d, 2)))

    def square(cx, cy, radius):
        d = smax(sa(sub(col, cx)), sa(sub(row, cy)))
        return sclip(sub(aff(radius, 1, 1), d))

    def penalty(mask):
        return sm(sub(one, mask), 1024)

    def sample_bool(field, mask):
        return gm(smin(field, mask))

    def sample_value(field, pen):
        return gm(sub(field, pen))

    # Rendering scale and wall palette.
    ex = sclip(sub(sa(sub(col, const(31.5))), const(29.5)))
    ey = sclip(sub(sa(sub(row, const(31.5))), const(29.5)))
    palette = gm(mul("x0", mul(ex, ey)))
    pw = np.eye(16, dtype=np.float32)
    pw[5, 5] = 0
    wall_colour = channel_sum(mul("x0", dense(palette, pw)))
    not_wall_colour = sub(one, wall_colour)
    actors = add(color(10), color(1))

    width_w = np.zeros((2, 1, 7, 7), np.float32)
    width_w[0, 0, 3, :] = 1
    width_w[1, 0, :, 3] = 1
    widths = gm(conv(actors, width_w))
    scale = vmin(vs(widths, 0, 1), vs(widths, 1, 2))
    scale = vmin(vmax(scale, const(3)), const(7))
    radius = aff(scale, 0.5, -0.5)

    # Outlined selected blocks.
    bw = np.zeros((4, 16, 7, 7), np.float32)
    bb = np.zeros(4, np.float32)
    for i, size in enumerate((3, 4, 5, 6)):
        start = -((size - 1) // 2)
        for yy in range(size):
            for xx in range(size):
                border = yy in (0, size - 1) or xx in (0, size - 1)
                c = 5 if border else 11
                bw[i, c, 3 + start + yy, 3 + start + xx] = 1
        bb[i] = -size * size + 0.5

    block_centers = channel_sum(sm(sr(conv("x0", bw, bb)), 2))
    selected = gm(block_centers)
    selected_neighborhood = sclip(
        conv(block_centers, np.ones((1, 1, 7, 7), np.float32)))
    selected_pixels = mul(selected_neighborhood, color(11))
    ordinary_pixels = mul(color(9), not_wall_colour)
    ordinary_blocks = sclip(
        conv(ordinary_pixels, np.ones((1, 1, 3, 3), np.float32)))
    all_block_pixels = sclip(add(ordinary_pixels, selected_pixels))

    # Gate and pressure-plate colours.
    cw = np.zeros((3, 16, 1, 1), np.float32)
    for i, c in enumerate((14, 12, 15)):
        cw[i, c, 0, 0] = 1
    coloured = mul(conv("x0", cw), not_wall_colour)
    counts = conv(coloured, diagonal_box(3, 7))
    gate_core = sclip(sub(counts, aff(vm(scale, scale), 1, 0.5)))
    gate_neighborhood = sclip(conv(gate_core, diagonal_box(3, 7)))
    gates = smin(coloured, gate_neighborhood)
    visible_plates = sr(sub(coloured, gates))
    gates_any = sclip(channel_sum(gates))
    gate_maps = [sl(gates, i, i + 1) for i in range(3)]

    # Persistent plate locations: two extremal locations per colour.
    old_plate_x = vs("mem", PLATE_START, PLATE_START + 6)
    old_plate_y = vs("mem", PLATE_START + 6, PLATE_START + 12)
    old_plate_valid = vs("mem", PLATE_START + 12, PLATE_START + 18)
    remembered = mul(square(old_plate_x, old_plate_y, radius), old_plate_valid)
    combine_plates = np.zeros((3, 6, 1, 1), np.float32)
    for i in range(3):
        combine_plates[i, i, 0, 0] = 1
        combine_plates[i, i + 3, 0, 0] = 1
    plates = sclip(add(visible_plates, conv(remembered, combine_plates)))

    rank = sm(add(sm(row, 64), col), 0.0005)
    plate_score = sub(
        sm(cat(plates, plates), 100),
        mul(rank, const([1, 1, 1, -1, -1, -1])))
    plate_best = gm(plate_score)
    plate_extrema = smin(
        cat(plates, plates),
        sr(sub(one, sm(sub(plate_best, plate_score), 4096))))
    plate_valid = gm(cat(plates, plates))
    plate_offset = vm(radius, const([1, 1, 1, -1, -1, -1]))
    plate_x = va(gm(mul(col, plate_extrema)), plate_offset)
    plate_y = va(gm(mul(row, plate_extrema)), plate_offset)
    plate_maps = [sl(plates, i, i + 1) for i in range(3)]
    plates_any = sclip(channel_sum(plates))

    free = sclip(add(
        add(add(color(5), actors), all_block_pixels),
        channel_sum(coloured)))
    free = mul(free, mul(sclip(row), sclip(sub(const(63), row))))
    hard = sub(one, free)
    closed_free = mul(free, sub(one, gates_any))

    # Token lattice and recurrent identity tracking.
    anchor_x = vd(const(64), gm(mul(sub(const(64), col), actors)))
    anchor_y = vd(const(64), gm(mul(sub(const(64), row), actors)))
    offsets = dense(scale, np.arange(-21, 22, dtype=np.float32).reshape(43, 1))

    def lattice_axis(axis, anchor):
        d = sub(sub(axis, anchor), offsets)
        return channel_sum(sclip(sub(const(2), sm(sa(d), 4))))

    corners = mul(
        actors, smin(lattice_axis(col, anchor_x),
                     lattice_axis(row, anchor_y)))
    inert_mask = mul(point(memory[9], memory[10]), memory[11])
    available_corners = sr(sub(corners, inert_mask))

    def choose(candidates, old_x, old_y, tracking):
        distance = add(sa(sub(col, old_x)), sa(sub(row, old_y)))
        cost = add(mul(distance, tracking), rank)
        score = sub(sm(sub(one, candidates), -1000), cost)
        best = gm(score)
        mask = smin(
            candidates, sr(sub(one, sm(sub(best, score), 2048))))
        return gm(mul(col, mask)), gm(mul(row, mask)), mask

    ready = memory[8]
    ax, ay, am = choose(available_corners, memory[0], memory[1], ready)
    second = sr(sub(available_corners, mul(am, vd(one, ready))))
    bx, by, _ = choose(second, memory[2], memory[3], ready)
    actor_area = vmc(node("gmean", [actors]), 4096)
    area_decreased = sat(vd(memory[12], actor_area))
    duplicate = zero_test(va(vab(vd(ax, bx)), vab(vd(ay, by))))
    merged = vm(vm(duplicate, area_decreased), ready)

    remaining = sr(sub(available_corners, mul(am, merged)))
    nax, nay, nam = choose(remaining, zero, zero, zero)
    nbx, nby, _ = choose(sr(sub(remaining, nam)), zero, zero, zero)
    px1, py1 = blend(nax, ax, merged), blend(nay, ay, merged)
    px2, py2 = blend(nbx, bx, merged), blend(nby, by, merged)
    inert_x = blend(ax, memory[9], merged)
    inert_y = blend(ay, memory[10], merged)
    inert_valid = cap(va(memory[11], merged))
    tracking = vm(ready, vd(one, merged))

    h_action = vd(vs("aprev", 4, 5), vs("aprev", 3, 4))
    v_action = vd(vs("aprev", 2, 3), vs("aprev", 1, 2))
    dx = vd(vc(px1, px2), vc(memory[0], memory[2]))
    dy = vd(vc(py1, py2), vc(memory[1], memory[3]))
    sx_old = blend(vc(memory[4], memory[5]), const([1, -1]), tracking)
    sy_old = blend(vc(memory[6], memory[7]), const([1, 1]), tracking)

    def infer_sign(delta, action, previous):
        sign = node("vtanh", [vmc(vm(delta, action), 16)])
        return blend(sign, previous, vm(vm(sign, sign), tracking))

    sx = infer_sign(dx, h_action, sx_old)
    sy = infer_sign(dy, v_action, sy_old)
    movement = va(vab(dx), vab(dy))
    movement_total = va(vs(movement, 0, 1), vs(movement, 1, 2))
    stuck = vm(vm(zero_test(movement_total), tracking),
               vd(one, sat(memory[13])))

    cx1, cy1 = va(px1, radius), va(py1, radius)
    cx2, cy2 = va(px2, radius), va(py2, radius)
    centers_x, centers_y = vc(cx1, cx2), vc(cy1, cy2)
    seed1 = mul(square(cx1, cy1, radius), free)
    seed2 = mul(square(cx2, cy2, radius), free)

    def wavefront(seed, limit, channels, iterations):
        nonlocal current, serial
        loop_id = "wave%d" % serial
        serial += 1
        state_id = "ws%d" % serial
        serial += 1
        shapes[state_id] = shapes[seed]
        body = []
        previous_current = current
        current = body

        w = np.zeros((4 * channels, channels, 3, 3), np.float32)
        for d, (yy, xx) in enumerate(((0, 1), (2, 1), (1, 0), (1, 2))):
            for c in range(channels):
                w[d * channels + c, c, yy, xx] = 1
        shifted = conv(state_id, w)
        pieces = [sl(shifted, d * channels, (d + 1) * channels)
                  for d in range(4)]
        reach = smax(smax(pieces[0], pieces[1]),
                     smax(pieces[2], pieces[3]))
        reach = sr(smin(smax(sub(reach, one), seed), limit))

        current = previous_current
        current.append({
            "id": loop_id, "op": "loop", "T": iterations,
            "state": [{"id": state_id, "init": seed}],
            "body": body, "next": {state_id: reach},
        })
        out = loop_id + "." + state_id
        shapes[out] = shapes[seed]
        top_names.add(out)
        return out

    # Two optimistic distances and one closed-gate reachability field.
    fields = wavefront(
        sm(cat(seed1, seed2, seed1), 512),
        sm(cat(free, free, closed_free), 512), 3, 256)
    f1, f2 = sl(fields, 0, 1), sl(fields, 1, 2)
    connected = sat(sample_bool(sl(fields, 2, 3), point(cx2, cy2)))
    need_switch = vm(vd(one, connected), gm(gates_any))

    # Expand off-corridor targets so a block's cell centre can reach them.
    field_sum = add(f1, f2)
    parking_goal = sclip(sub(sub(gm(field_sum), field_sum), scale))
    parking_goal = sclip(
        conv(parking_goal, np.ones((1, 1, 5, 5), np.float32)))
    parking_free = mul(free, sub(one, ordinary_blocks))
    parking_free = mul(parking_free, sub(one, actors))
    parking_free = mul(parking_free, sub(one, gates_any))
    parking_free = mul(parking_free, sub(one, plates_any))
    parking_goal = mul(parking_goal, parking_free)

    auxiliary = wavefront(
        sm(cat(parking_goal, mul(plates_any, closed_free)), 512),
        sm(cat(parking_free, closed_free), 512), 2, 192)
    parking_field = sl(auxiliary, 0, 1)
    plate_field = sl(auxiliary, 1, 2)

    initial_mask = point(centers_x, centers_y)
    initial_penalty = penalty(initial_mask)
    initial_fields = [
        sample_value(f, initial_penalty) for f in (f1, f2, plate_field)
    ]
    initial_switches = [sample_bool(f, initial_mask) for f in plate_maps]

    B, C = 16, 32
    repeat_w = np.zeros((C, 2), np.float32)
    repeat_w[:B, 0] = 1
    repeat_w[B:, 1] = 1

    def repeat_pair(x):
        return dense(x, repeat_w)

    sx_lanes, sy_lanes = repeat_pair(sx), repeat_pair(sy)

    def pair_sum(x):
        return va(vs(x, 0, B), vs(x, B, C))

    def pair_difference(x):
        return vd(vs(x, 0, B), vs(x, B, C))

    def both(x):
        return vc(x, x)

    history = vs("mem", HISTORY_START, PLATE_START)
    history_replicate = np.zeros((B * H, H), np.float32)
    lane_replicate = np.zeros((B * H, B), np.float32)
    history_sum = np.zeros((B, B * H), np.float32)
    for b in range(B):
        for h in range(H):
            history_replicate[b * H + h, h] = 1
            lane_replicate[b * H + h, b] = 1
            history_sum[b, b * H + h] = 1
    history_coordinates = [
        dense(take(history, list(range(k, 4 * H, 4))), history_replicate)
        for k in range(4)
    ]

    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))
    first_x = np.zeros(C, np.float32)
    first_y = np.zeros(C, np.float32)
    change_x = np.zeros((C, 1), np.float32)
    change_y = np.zeros((C, 1), np.float32)
    for actor in range(2):
        for first in range(4):
            for second in range(4):
                i = actor * B + first * 4 + second
                first_x[i], first_y[i] = directions[first]
                change_x[i, 0] = directions[second][0] - directions[first][0]
                change_y[i, 0] = directions[second][1] - directions[first][1]

    states = {
        "rx": repeat_pair(centers_x),
        "ry": repeat_pair(centers_y),
        "rf1": repeat_pair(initial_fields[0]),
        "rf2": repeat_pair(initial_fields[1]),
        "rfp": repeat_pair(initial_fields[2]),
        "rs0": repeat_pair(initial_switches[0]),
        "rs1": repeat_pair(initial_switches[1]),
        "rs2": repeat_pair(initial_switches[2]),
        "rscore": const(np.zeros(B, np.float32)),
        "rgoal": const(np.zeros(B, np.float32)),
        "rt": zero,
    }
    for k, v in states.items():
        shapes[k] = shapes[v]

    body = []
    previous_current = current
    current = body

    turn = sat("rt")
    vel_x = dense(turn, change_x, first_x)
    vel_y = dense(turn, change_y, first_y)
    proposed_x = va("rx", vm(vm(vel_x, sx_lanes), scale))
    proposed_y = va("ry", vm(vm(vel_y, sy_lanes), scale))
    proposed_mask = point(proposed_x, proposed_y)
    proposed_penalty = penalty(proposed_mask)
    collision = sample_bool(hard, proposed_mask)
    old_switches = ("rs0", "rs1", "rs2")
    for gate, old_switch in zip(gate_maps, old_switches):
        hit = sample_bool(gate, proposed_mask)
        opened = both(cap(pair_sum(old_switch)))
        collision = va(collision, vm(hit, vd(one, opened)))

    accepted = vm(vd(one, cap(collision)), gm(proposed_mask))
    next_x = blend(proposed_x, "rx", accepted)
    next_y = blend(proposed_y, "ry", accepted)
    next_switches = [
        blend(sample_bool(plate, proposed_mask), old, accepted)
        for plate, old in zip(plate_maps, old_switches)
    ]
    next_fields = [
        blend(sample_value(f, proposed_penalty), old, accepted)
        for f, old in zip((f1, f2, plate_field), ("rf1", "rf2", "rfp"))
    ]

    phi = vd(next_fields[0], next_fields[1])
    heuristic = va(
        neg(vab(pair_difference(phi))),
        vmc(pair_sum(va(next_fields[0], next_fields[1])), 0.15))

    # Only one token needs to occupy a useful switch.
    closest_switch = vmax(
        vs(next_fields[2], 0, B), vs(next_fields[2], B, C))
    heuristic = va(
        heuristic, vm(closest_switch, aff(need_switch, 3, 0.03)))

    opened_count = cap(pair_sum(next_switches[0]))
    for sw in next_switches[1:]:
        opened_count = va(opened_count, cap(pair_sum(sw)))
    heuristic = va(
        heuristic, vm(opened_count, vm(scale, aff(need_switch, 12, 3))))

    pair_dx = pair_difference(next_x)
    pair_dy = pair_difference(next_y)
    coincidence = zero_test(va(vab(pair_dx), vab(pair_dy)))
    crossing = vm(
        sat(neg(vm(pair_difference("rx"), pair_dx))),
        zero_test(vab(pair_dy)))
    reached = cap(va("rgoal", cap(va(coincidence, crossing))))

    motion = pair_sum(va(vab(vd(next_x, "rx")), vab(vd(next_y, "ry"))))
    idle_cost = va(vmc(scale, 2), aff(turn, -1000, 1000))
    heuristic = vd(heuristic, vm(zero_test(motion), idle_cost))

    simulated_coordinates = (
        vs(next_x, 0, B), vs(next_y, 0, B),
        vs(next_x, B, C), vs(next_y, B, C))
    history_distance = None
    for pos, past in zip(simulated_coordinates, history_coordinates):
        d = vab(vd(dense(pos, lane_replicate), past))
        history_distance = d if history_distance is None else va(history_distance, d)
    visits = dense(zero_test(history_distance), history_sum)
    heuristic = vd(heuristic, vm(visits, vmc(scale, 20)))
    heuristic = va(heuristic, vmc(reached, 800))

    next_values = {
        "rx": next_x, "ry": next_y,
        "rf1": next_fields[0], "rf2": next_fields[1],
        "rfp": next_fields[2],
        "rs0": next_switches[0], "rs1": next_switches[1],
        "rs2": next_switches[2],
        "rscore": va("rscore", heuristic),
        "rgoal": reached, "rt": aff("rt", 1, 1),
    }
    current = previous_current
    current.append({
        "id": "predict", "op": "loop", "T": 10,
        "state": [{"id": k, "init": v} for k, v in states.items()],
        "body": body, "next": next_values,
    })
    for k, v in states.items():
        out = "predict." + k
        shapes[out] = shapes[v]
        top_names.add(out)

    action_scores = "predict.rscore"
    for n in (16, 8):
        action_scores = vmax(
            take(action_scores, list(range(0, n, 2))),
            take(action_scores, list(range(1, n, 2))))
    action_scores = vd(
        action_scores, vm(vs("aprev", 1, 5), vmc(stuck, 10000)))

    # Probe each immediate joint move.
    four_repeat = np.zeros((8, 2), np.float32)
    four_repeat[:4, 0] = 1
    four_repeat[4:, 1] = 1
    probe_x = dense(centers_x, four_repeat)
    probe_y = dense(centers_y, four_repeat)
    probe_sx = dense(sx, four_repeat)
    probe_sy = dense(sy, four_repeat)
    probe_x = va(probe_x, vm(vm(
        const([0, 0, -1, 1, 0, 0, -1, 1]), probe_sx), scale))
    probe_y = va(probe_y, vm(vm(
        const([-1, 1, 0, 0, -1, 1, 0, 0]), probe_sy), scale))
    probes = point(probe_x, probe_y)

    # After a block yields control, take a non-blocked player move first.
    block_hits = sample_bool(ordinary_pixels, probes)
    block_hits = va(vs(block_hits, 0, 4), vs(block_hits, 4, 8))
    just_released = vm(sat(memory[13]), vd(one, selected))
    action_scores = vd(
        action_scores, vm(block_hits, vmc(just_released, 2000)))

    action_scores = vd(action_scores, reduce_max(action_scores))
    action_scores = va(action_scores, const([0, -0.01, -0.02, -0.03]))
    winners = zero_test(vd(reduce_max(action_scores), action_scores), 1000)
    block_click_map = channel_sum(
        mul(mul(probes, vc(winners, winners)), ordinary_pixels))
    need_block_click = sat(gm(block_click_map))

    # Selected-block navigation.
    block_x = gm(mul(col, block_centers))
    block_y = gm(mul(row, block_centers))
    block_probe_x = va(block_x, vm(const([0, 0, -1, 1]), scale))
    block_probe_y = va(block_y, vm(const([-1, 1, 0, 0]), scale))
    block_probes = point(block_probe_x, block_probe_y)
    parking_scores = sample_value(parking_field, penalty(block_probes))
    parking_collision = cap(va(
        sample_bool(sub(one, parking_free), block_probes),
        vd(one, gm(block_probes))))
    parking_scores = vd(parking_scores, vmc(parking_collision, 2000))
    parking_scores = va(parking_scores, vmc(vs("aprev", 1, 5), 0.1))
    parking_scores = vd(parking_scores, reduce_max(parking_scores))

    block_current = point(block_x, block_y)
    parked = sample_bool(parking_goal, block_current)
    no_parking_route = zero_test(sample_bool(parking_field, block_current))
    done_parking = cap(va(parked, no_parking_route))
    block_counter = vm(selected, aff(memory[13], 1, 1))

    final_directions = blend(parking_scores, action_scores, selected)
    click_needed = blend(done_parking, need_block_click, selected)
    node("vconcat", [
        const(-1000000), final_directions, const(-1000000),
        aff(click_needed, 101, -100), const(-1000000)
    ], name="logits")

    click_map = add(
        mul(block_click_map, vd(one, selected)), mul(actors, selected))
    node("add", [click_map, zero], name="click")
    node("vconcat", [
        px1, py1, px2, py2, sx, sy, one,
        inert_x, inert_y, inert_valid, actor_area, block_counter,
        cx1, cy1, cx2, cy2,
        vs("mem", HISTORY_START, PLATE_START - 4),
        plate_x, plate_y, plate_valid
    ], name="mem_next")

    # Structural dead-code elimination.
    def prune(nodes, wanted):
        needed, kept = set(wanted), []
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                outputs = {nd["id"] + "." + s["id"] for s in nd["state"]}
                if not (needed & outputs):
                    continue
                needed.difference_update(outputs)
                new_body, external = prune(nd["body"], set(nd["next"].values()))
                needed.update(external - {s["id"] for s in nd["state"]})
                needed.update(s["init"] for s in nd["state"] if "init" in s)
                copy = dict(nd)
                copy["body"] = new_body
                kept.append(copy)
            elif nd["id"] in needed:
                needed.remove(nd["id"])
                needed.update(nd["in"])
                kept.append(nd)
        kept.reverse()
        return kept, needed

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
    return {
        "graph": graph,
        "memory_size": MEMORY_SIZE,
        "params": {k: v for k, v in params.items() if k in used},
    }
