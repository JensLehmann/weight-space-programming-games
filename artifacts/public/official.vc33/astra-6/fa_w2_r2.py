import numpy as np


def build():
    graph = []
    params = {}
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "avail": (8,),
        "aprev": (8,),
    }
    current = graph
    serial = 0
    parameter_cache = {}
    constant_cache = {}
    repeated_flags = {}
    N = 16

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def parameter(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = fresh("p")
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def emit(op, inputs, name=None, **fields):
        name = name or fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        current.append(nd)

        if op == "conv":
            shape = (params[fields["w"]].shape[0], 64, 64)
        elif op == "dense":
            shape = (params[fields["w"]].shape[0],)
        elif op in ("gmax", "gmean"):
            shape = (shapes[inputs[0]][0],)
        elif op in ("concat", "vconcat"):
            c = sum(shapes[x][0] for x in inputs)
            shape = (c, 64, 64) if op == "concat" else (c,)
        elif op in ("slice", "vslice"):
            c = fields["stop"] - fields["start"]
            shape = (c, 64, 64) if op == "slice" else (c,)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = (max(shapes[x][0] for x in inputs), 64, 64)
        elif op in ("vadd", "vmul"):
            shape = (max(shapes[x][0] for x in inputs),)
        else:
            shape = shapes[inputs[0]]
        shapes[name] = shape
        return name

    def const(value):
        a = np.asarray(value, np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key not in constant_cache:
            name = fresh("k")
            w = parameter(np.zeros((len(a), 8), np.float32))
            b = parameter(a)
            graph.append({
                "id": name, "op": "dense", "in": ["avail"],
                "w": w, "b": b,
            })
            shapes[name] = (len(a),)
            constant_cache[key] = name
        return constant_cache[key]

    def conv(x, w, b=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(
                np.asarray(b, np.float32).reshape(-1)
            )
        return emit("conv", [x], **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(np.asarray(w, np.float32))}
        if b is not None:
            fields["b"] = parameter(
                np.asarray(b, np.float32).reshape(-1)
            )
        return emit("dense", [x], name=name, **fields)

    def sl(x, a, b):
        op = "slice" if len(shapes[x]) == 3 else "vslice"
        return emit(op, [x], start=a, stop=b)

    def add(a, b):
        return emit("add", [a, b])

    def sub(a, b):
        return emit("sub", [a, b])

    def mul(a, b):
        return emit("mul", [a, b])

    def relu(x):
        return emit("relu", [x])

    def absolute(x):
        return emit("abs", [x])

    def scale(x, a):
        return mul(x, const(a))

    def vadd(a, b):
        return emit("vadd", [a, b])

    def vmul(a, b):
        return emit("vmul", [a, b])

    def vscale(x, a):
        return vmul(x, const(a))

    def vsub(a, b):
        return vadd(a, vscale(b, -1.0))

    def vmax(a, b):
        return vadd(emit("vrelu", [vsub(a, b)]), b)

    def vabs(x):
        return vadd(
            emit("vrelu", [x]),
            emit("vrelu", [vscale(x, -1.0)])
        )

    one = const(1.0)

    def cap(x):
        x = relu(x)
        return sub(one, relu(sub(one, x)))

    def vcap(x):
        x = emit("vrelu", [x])
        return vsub(one, emit("vrelu", [vsub(one, x)]))

    def total(x):
        return conv(x, np.ones((1, shapes[x][0]), np.float32))

    def vtotal(x):
        return dense(x, np.ones((1, shapes[x][0]), np.float32))

    def pool(x):
        return emit("gmax", [x])

    def concat(*xs):
        return emit("concat", xs)

    def maxsp(a, b):
        return emit("max2", [a, b])

    def repeat_flag(flag, c):
        if c == 1:
            return flag
        key = (flag, c)
        if key not in repeated_flags:
            k = shapes[flag][0]
            w = np.zeros((k * c, k), np.float32)
            for d in range(k):
                w[d * c:(d + 1) * c, d] = 1
            name = fresh()
            graph.append({
                "id": name, "op": "dense", "in": [flag],
                "w": parameter(w),
            })
            shapes[name] = (k * c,)
            repeated_flags[key] = name
        return repeated_flags[key]

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def shift(x, flag, sign=1):
        c = shapes[x][0]
        w = np.zeros((4 * c, c, 3, 3), np.float32)
        for d, (dy, dx) in enumerate(directions):
            for j in range(c):
                w[d * c + j, j, 1 + sign * dy, 1 + sign * dx] = 1
        z = mul(conv(x, w), repeat_flag(flag, c))
        merge = np.zeros((c, 4 * c), np.float32)
        for d in range(4):
            for j in range(c):
                merge[j, d * c + j] = 1
        return conv(z, merge)

    def loop(T, initial, function):
        nonlocal current
        lid = fresh("l")
        state_specs = []
        state_ids = []

        for init in initial:
            sid = fresh("s")
            if isinstance(init, str):
                state_specs.append({"id": sid, "init": init})
                shapes[sid] = shapes[init]
            else:
                state_specs.append({"id": sid, "zeros": list(init)})
                shapes[sid] = tuple(init)
            state_ids.append(sid)

        outer = current
        body = []
        current = body
        result = function(*state_ids)
        if isinstance(result, str):
            result = [result]
        current = outer
        current.append({
            "id": lid, "op": "loop", "T": T,
            "state": state_specs,
            "body": body,
            "next": dict(zip(state_ids, result)),
        })

        outputs = []
        for sid, src in zip(state_ids, result):
            out = lid + "." + sid
            shapes[out] = shapes[src]
            outputs.append(out)
        return outputs[0] if len(outputs) == 1 else outputs

    def prefix_sum(x, flag):
        return loop(
            64, [x], lambda s: add(shift(s, flag, -1), x)
        )

    def equality_masks(label):
        d = absolute(sub(label, const(np.arange(1, N + 1))))
        return relu(sub(one, d))

    # Palette features.
    white = sl("x0", 0, 1)
    yellow = sl("x0", 4, 5)
    wall_colour = sl("x0", 5, 6)
    buttons = sl("x0", 9, 10)

    wb = np.zeros((2, 16), np.float32)
    wb[0, 11] = 1
    wb[1, 14] = 1
    bases = conv("x0", wb)
    any_base = total(bases)

    # The top display row is UI, not evidence of world geometry.
    top_row = relu(sub(one, scale(sl("coords", 0, 1), 63.0)))
    world_visible = sub(one, top_row)
    yellow = mul(yellow, world_visible)

    # Infer gravity from each animal's body and coloured base.
    wd = np.zeros((4, 1, 7, 7), np.float32)
    for d, (dy, dx) in enumerate(directions):
        for distance in (1, 2, 3):
            wd[d, 0, 3 - distance * dy, 3 - distance * dx] = 1
    gravity = vcap(pool(mul(conv(yellow, wd), any_base)))

    wu = np.zeros((4, 4), np.float32)
    wu[0, 1] = wu[0, 3] = 1
    wu[1, 0] = wu[1, 2] = 1
    perpendicular = dense(gravity, wu)

    wa = np.zeros((2, 4), np.float32)
    wa[0, 0] = wa[0, 2] = 1
    wa[1, 1] = wa[1, 3] = 1
    axes = dense(gravity, wa)

    def gravity_axis_max(x):
        z = concat(emit("colmax", [x]), emit("rowmax", [x]))
        return total(mul(z, axes))

    # Canonical coordinate increases along gravity.
    tq = conv("coords", np.array([
        [63, 0, 0], [0, 63, 0],
        [-63, 0, 63], [0, -63, 63],
    ], np.float32))
    t = total(mul(tq, gravity))
    tp = add(t, one)
    pixel_index = conv(
        "coords", np.array([[4032, 63, 1]], np.float32)
    )

    nearby_yellow = shift(yellow, gravity, -1)
    y2 = shift(nearby_yellow, gravity, -1)
    y3 = shift(y2, gravity, -1)
    body_near = cap(add(add(nearby_yellow, y2), y3))
    animals = mul(bases, body_near)
    goals = mul(bases, sub(one, body_near))
    goal_end = mul(
        goals, sub(one, shift(total(goals), gravity, 1))
    )

    wg = np.zeros((1, 16), np.float32)
    wg[0, 1] = wg[0, 12] = 1
    gates = conv("x0", wg)
    gate_end = mul(gates, sub(one, shift(gates, gravity, 1)))

    # Long wall runs partition the perpendicular axis.
    wm = np.zeros((1, 16), np.float32)
    wm[0, [1, 5, 11, 12, 14]] = 1
    material = conv("x0", wm)
    wr = np.zeros((2, 1, 7, 7), np.float32)
    wr[0, 0, :, 3] = 1
    wr[1, 0, 3, :] = 1
    runs = relu(conv(material, wr, [-6, -6]))
    vertical_wall = emit("colmax", [sl(runs, 0, 1)])
    horizontal_wall = emit("rowmax", [sl(runs, 1, 2)])
    walls = total(
        mul(concat(vertical_wall, horizontal_wall), axes)
    )

    wall_minus = shift(walls, perpendicular, -1)
    wall_plus = shift(walls, perpendicular, 1)
    floors = mul(wall_colour, sub(one, walls))

    # Critical revision: the timer row receives no compartment label.
    # Otherwise an occluded partial divider creates a false extension
    # that corrupts the compartment's maximum height and fluid quantity.
    free = mul(
        sub(sub(one, walls), floors),
        world_visible,
    )

    stripe_edge = mul(walls, sub(one, wall_minus))
    floor_edge = mul(
        floors, sub(one, shift(floors, gravity, -1))
    )
    floor_end = mul(
        floors, sub(one, shift(floors, gravity, 1))
    )
    stripes = prefix_sum(stripe_edge, perpendicular)
    floor_data = prefix_sum(
        concat(floor_edge, mul(floor_end, tp)), gravity
    )
    floor_number = sl(floor_data, 0, 1)
    floor_bound_map = sl(floor_data, 1, 2)

    labels = mul(
        add(add(stripes, scale(floor_number, 8.0)), one),
        free,
    )
    region = equality_masks(labels)

    # Suppress decorative rails when measuring fluid.
    wide_wall = maxsp(walls, maxsp(wall_minus, wall_plus))
    wide_wall = maxsp(
        wide_wall,
        maxsp(
            shift(wide_wall, perpendicular, -1),
            shift(wide_wall, perpendicular, 1),
        ),
    )
    fluid_pixels = mul(white, sub(one, wide_wall))

    def side_labels(sign):
        z = shift(labels, perpendicular, sign)
        acc = z
        for _ in range(4):
            z = shift(z, perpendicular, sign)
            acc = maxsp(acc, z)
        return acc

    left_label = side_labels(-1)
    right_label = side_labels(1)
    left_region = equality_masks(left_label)
    right_region = equality_masks(right_label)
    sides = add(left_region, right_region)

    button_left = mul(buttons, wall_minus)
    button_right = mul(buttons, wall_plus)
    button_edge = add(button_left, button_right)
    other_label = add(
        mul(button_left, left_label),
        mul(button_right, right_label),
    )
    other_region = equality_masks(other_label)

    # Fluid surface positions and capacities.
    hmax = pool(mul(region, tp))
    depth = pool(mul(mul(sub(hmax, t), region), fluid_pixels))
    height = vsub(hmax, depth)

    wall_visible = mul(walls, material)
    reverse_t = sub(const(64.0), t)
    wall_start = mul(
        sub(
            const(64.0),
            gravity_axis_max(mul(wall_visible, reverse_t)),
        ),
        walls,
    )
    neighboring_start = maxsp(
        shift(wall_start, perpendicular, -1),
        shift(wall_start, perpendicular, 1),
    )
    wall_minimum = pool(mul(region, neighboring_start))
    floor_minimum = pool(mul(region, floor_bound_map))

    # Button width is one transfer increment.
    bminus = shift(buttons, perpendicular, -1)
    bplus = shift(buttons, perpendicular, 1)
    button_width = add(
        add(add(buttons, bminus), bplus),
        add(
            shift(bminus, perpendicular, -1),
            shift(bplus, perpendicular, 1),
        ),
    )
    step_size = pool(mul(button_width, buttons))

    # Conservation and structural checks for possible goal destinations.
    fluid_supply = vtotal(depth)
    destination_minimum = vmax(
        wall_minimum,
        vadd(
            floor_minimum,
            vmul(vcap(floor_minimum), vscale(step_size, 2.0)),
        ),
    )
    required_depth = sub(hmax, tp)
    enough_fluid = cap(
        add(one, scale(sub(fluid_supply, required_depth), 2.0))
    )
    within_upper_bound = cap(
        add(one, scale(sub(tp, destination_minimum), 2.0))
    )
    within_lower_bound = cap(
        add(one, scale(required_depth, 2.0))
    )
    destination_sides = mul(
        sides,
        mul(
            enough_fluid,
            mul(within_upper_bound, within_lower_bound),
        ),
    )

    animal_present = []
    goal_present = []
    goal_height = []
    for colour in range(2):
        animal_present.append(
            pool(mul(region, sl(animals, colour, colour + 1)))
        )
        annotation = mul(
            destination_sides,
            sl(goal_end, colour, colour + 1),
        )
        goal_present.append(pool(annotation))
        goal_height.append(pool(mul(annotation, tp)))

    has_animal = vcap(vadd(*animal_present))
    reserve = vmul(
        vmul(has_animal, vcap(floor_minimum)),
        vscale(step_size, 2.0),
    )
    hmin = vmax(wall_minimum, vadd(floor_minimum, reserve))

    # Stable priority among unsatisfied animals.
    unsolved_maps = []
    for colour in range(2):
        mismatch = vcap(
            vscale(vabs(vsub(height, goal_height[colour])), 2.0)
        )
        mismatch = vmax(mismatch, vsub(one, goal_present[colour]))
        unsolved = vmul(animal_present[colour], mismatch)
        unsolved_maps.append(
            mul(
                total(mul(region, unsolved)),
                sl(animals, colour, colour + 1),
            )
        )

    animal_priority = add(
        mul(unsolved_maps[0], sub(const(200.0), labels)),
        mul(unsolved_maps[1], sub(const(100.0), labels)),
    )

    def winner(score, sharpness):
        near = relu(
            sub(
                one,
                scale(absolute(sub(score, pool(score))), sharpness),
            )
        )
        order = mul(near, pixel_index)
        return relu(
            sub(
                one,
                scale(absolute(sub(order, pool(order))), 2.0),
            )
        )

    animal_winner = winner(animal_priority, 2.0)
    active_region = pool(mul(region, animal_winner))
    active_colour = pool(mul(bases, animal_winner))

    matching_goal = total(mul(goal_end, active_colour))
    roots = vadd(
        vmul(goal_present[0], sl(active_colour, 0, 1)),
        vmul(goal_present[1], sl(active_colour, 1, 2)),
    )
    roots = vscale(roots, 100.0)

    # Pair index i*N+j: receiving region i, donor region j.
    expand_row = np.zeros((N * N, N), np.float32)
    expand_col = np.zeros((N * N, N), np.float32)
    row_sum = np.zeros((N, N * N), np.float32)
    for i in range(N):
        for j in range(N):
            k = i * N + j
            expand_row[k, i] = 1
            expand_col[k, j] = 1
            row_sum[i, k] = 1

    gate_pairs = mul(
        mul(conv(sides, expand_row), conv(sides, expand_col)),
        gate_end,
    )
    gate_adjacency = pool(gate_pairs)

    button_pairs = mul(
        conv(region, expand_row),
        conv(other_region, expand_col),
    )
    fluid_adjacency = pool(button_pairs)

    def neighbor_max(x):
        width = N * N
        while width > N:
            count = width // 2
            difference = np.zeros((count, width), np.float32)
            second = np.zeros((count, width), np.float32)
            for j in range(count):
                difference[j, 2 * j] = 1
                difference[j, 2 * j + 1] = -1
                second[j, 2 * j + 1] = 1
            x = vadd(
                emit("vrelu", [dense(x, difference)]),
                dense(x, second),
            )
            width = count
        return x

    edge_cost = vadd(
        vscale(vsub(gate_adjacency, one), 100.0),
        const(-1.0),
    )

    def route_iteration(d):
        candidates = vadd(dense(d, expand_col), edge_cost)
        return vmax(roots, neighbor_max(candidates))

    route_value = loop(16, [roots], route_iteration)

    accessible = cap(total(mul(sides, active_region)))
    goal_accessible = cap(
        total(mul(destination_sides, active_region))
    )
    route_at_gate = total(mul(sides, route_value))
    effort = total(mul(sides, absolute(sub(height, tp))))

    direct_goal = mul(matching_goal, goal_accessible)
    reachable_gate = mul(gate_end, accessible)
    target_score = add(
        mul(
            direct_goal,
            sub(const(1000.0), scale(effort, 0.01)),
        ),
        mul(
            reachable_gate,
            sub(
                add(const(100.0), scale(route_at_gate, 2.0)),
                scale(effort, 0.01),
            ),
        ),
    )
    target_winner = winner(target_score, 1024.0)
    target_is_gate = pool(mul(gate_end, target_winner))
    target_height = pool(mul(tp, target_winner))
    target_sides = pool(mul(sides, target_winner))
    target_weight = vadd(
        vmul(target_sides, target_is_gate),
        vmul(active_region, vsub(one, target_is_gate)),
    )

    # Continuous capacitated flow relaxation.
    def flow_iteration(flow):
        predicted_height = vsub(height, dense(flow, row_sum))
        error = vmul(
            target_weight,
            vsub(predicted_height, target_height),
        )
        too_high = emit("vrelu", [vsub(predicted_height, hmax)])
        too_low = emit("vrelu", [vsub(hmin, predicted_height)])
        gradient = vadd(
            error, vscale(vsub(too_high, too_low), 4.0)
        )
        pressure_difference = vsub(
            dense(gradient, expand_row),
            dense(gradient, expand_col),
        )
        update = vmul(fluid_adjacency, pressure_difference)
        return vadd(
            vscale(flow, 0.9997),
            vscale(update, 0.03),
        )

    flow = loop(256, [(N * N,)], flow_iteration)

    can_grow = vcap(vsub(height, hmin))
    can_shrink = vcap(vsub(hmax, height))
    legal_button = mul(
        mul(
            total(mul(region, can_grow)),
            total(mul(other_region, can_shrink)),
        ),
        button_edge,
    )
    local_flow = total(mul(button_pairs, flow))
    pump_score = mul(
        legal_button,
        add(const(100.0), scale(local_flow, 10.0)),
    )

    alignment_error = vtotal(
        vmul(
            target_weight,
            vabs(vsub(height, target_height)),
        )
    )
    gate_ready = vmul(
        target_is_gate,
        vcap(vsub(one, vscale(alignment_error, 2.0))),
    )
    crossing_score = scale(
        mul(target_winner, gate_ready), 10000.0
    )
    emit("add", [pump_score, crossing_score], name="click")

    logits_bias = np.full(8, -100.0, np.float32)
    logits_bias[6] = 10.0
    dense(
        "avail",
        np.zeros((8, 8), np.float32),
        logits_bias,
        name="logits",
    )

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }
