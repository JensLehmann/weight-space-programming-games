import numpy as np


def build():
    graph = []
    params = {}
    parameter_cache = {}
    expression_cache = {}
    tensor_shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (205,),
        "aprev": (8,),
        "avail": (8,),
    }
    current = graph
    serial = 0

    N = 100
    INVENTORIES = 16
    L = N * INVENTORIES
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    offsets = [-10, 10, -1, 1]
    rows = np.repeat(np.arange(10), 10).astype(np.float32)
    cols = np.tile(np.arange(10), 10).astype(np.float32)

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def emit(op, inputs, name=None, **fields):
        nonlocal serial
        key = (op, tuple(inputs), tuple(sorted(fields.items())))
        if name is None and key in expression_cache:
            return expression_cache[key]
        if name is None:
            name = "n%d" % serial
            serial += 1

        ss = [tensor_shapes[x] for x in inputs]
        if op == "conv":
            shape = (params[fields["w"]].shape[0], 64, 64)
        elif op == "dense":
            shape = (params[fields["w"]].shape[0],)
        elif op in ("gmax", "gmean"):
            shape = (ss[0][0],)
        elif op in ("slice", "vslice"):
            shape = (fields["stop"] - fields["start"],) + ss[0][1:]
        elif op in ("concat", "vconcat"):
            shape = (sum(s[0] for s in ss),) + ss[0][1:]
        elif op in ("add", "sub", "mul", "min2", "max2"):
            shape = (max(s[0] for s in ss), 64, 64)
        elif op in ("vadd", "vmul"):
            shape = (max(s[0] for s in ss),)
        else:
            shape = ss[0]

        tensor_shapes[name] = shape
        node = {"id": name, "op": op, "in": list(inputs)}
        node.update(fields)
        current.append(node)
        expression_cache[key] = name
        return name

    def spatial(x):
        return len(tensor_shapes[x]) == 3

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("dense", [x], name=name, **fields)

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv", [x], name=name, **fields)

    one = dense("aprev", np.zeros((1, 8)), [1])
    constants = {np.asarray([1], np.float32).tobytes(): one}

    def const(a):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        key = a.tobytes()
        if key not in constants:
            constants[key] = dense(one, np.zeros((len(a), 1)), a)
        return constants[key]

    zero = const([0])
    minus_one = const([-1])

    def sl(x, start, stop):
        return emit(
            "slice" if spatial(x) else "vslice",
            [x], start=int(start), stop=int(stop)
        )

    def cat(xs, name=None):
        return emit(
            "concat" if spatial(xs[0]) else "vconcat", xs, name=name
        )

    def add(a, b):
        return emit(
            "add" if spatial(a) or spatial(b) else "vadd", [a, b]
        )

    def mul(a, b):
        return emit(
            "mul" if spatial(a) or spatial(b) else "vmul", [a, b]
        )

    def neg(a):
        return mul(a, minus_one)

    def sub(a, b):
        if spatial(a) or spatial(b):
            return emit("sub", [a, b])
        return add(a, neg(b))

    def scale(a, k):
        return mul(a, const([k]))

    def bias(a, k):
        return add(a, const([k]))

    def relu(a):
        return emit("relu" if spatial(a) else "vrelu", [a])

    def absolute(a):
        if spatial(a):
            return emit("abs", [a])
        return add(relu(a), relu(neg(a)))

    def clip(a):
        return sub(relu(a), relu(bias(a, -1)))

    def equal(a, value=0):
        if value != 0:
            a = bias(a, -value)
        return relu(bias(neg(absolute(a)), 1))

    def maximum(a, b):
        if spatial(a) or spatial(b):
            return emit("max2", [a, b])
        return add(relu(sub(a, b)), b)

    def sum_nodes(xs):
        result = xs[0]
        for x in xs[1:]:
            result = add(result, x)
        return result

    def total(x):
        return dense(x, np.ones((1, tensor_shapes[x][0]), np.float32))

    def roll(x, offset):
        length = tensor_shapes[x][0]
        offset %= length
        if offset == 0:
            return x
        return cat([sl(x, offset, length), sl(x, 0, offset)])

    def valid(dy, dx):
        return (
            (rows + dy >= 0) & (rows + dy < 10) &
            (cols + dx >= 0) & (cols + dx < 10)
        ).astype(np.float32)

    def repeat_tiles(x):
        return cat([x] * INVENTORIES)

    def shift_tiles(x, dy, dx, blocks=1):
        return mul(
            roll(x, dy * 10 + dx),
            cat([const(valid(dy, dx))] * blocks)
        )

    # Coordinate masks: one channel per playable 5x5 cell.
    cw = np.zeros((20, 3, 1, 1), np.float32)
    cb = np.zeros(20, np.float32)
    for i in range(10):
        cw[i, 0, 0, 0] = 63
        cb[i] = -(7 + 5 * i)
        cw[10 + i, 1, 0, 0] = 63
        cb[10 + i] = -(11 + 5 * i)
    distances = absolute(conv("coords", cw, cb))

    mw = np.zeros((N, 20, 1, 1), np.float32)
    for i in range(N):
        mw[i, i // 10, 0, 0] = -2
        mw[i, 10 + i % 10, 0, 0] = -2
    peaks = relu(conv(distances, mw, np.ones(N)))
    cell_masks = clip(bias(scale(peaks, 4), -1))

    def point_mask(y, x):
        w = np.zeros((2, 3, 1, 1), np.float32)
        w[0, 0, 0, 0] = 63
        w[1, 1, 0, 0] = 63
        d = absolute(conv("coords", w, [-y, -x]))
        p = relu(conv(d, -2 * np.ones((1, 2, 1, 1)), [1]))
        return clip(bias(scale(p, 4), -1))

    hud_mask = point_mask(57, 5)

    def extract(field):
        return emit("gmax", [mul(cell_masks, field)])

    def vector_max(x):
        length = tensor_shapes[x][0]
        masks = cell_masks if length == N else sl(cell_masks, 0, length)
        field = conv(
            mul(masks, bias(x, 32)),
            np.ones((1, length, 1, 1), np.float32)
        )
        return bias(emit("gmax", [field]), -32)

    # Glyph templates in their six canonical orientations.
    glyphs = np.asarray([
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
    ], np.float32)
    colours = [12, 9, 14, 8]

    gw = np.zeros((2, 16, 1, 1), np.float32)
    gw[0, colours, 0, 0] = 1
    gw[1, 5, 0, 0] = 1
    glyph_input = conv("x0", gw)

    def decode_glyph(spacing):
        k = 2 * spacing + 1
        w = np.zeros((24, 2, k, k), np.float32)
        code_w = np.zeros((1, 24, 1, 1), np.float32)
        shape_w = np.zeros_like(code_w)
        for shape in range(6):
            for rotation in range(4):
                j = 4 * shape + rotation
                pattern = np.rot90(glyphs[shape], -rotation)
                for y in range(3):
                    for x in range(3):
                        channel = 0 if pattern[y, x] else 1
                        w[j, channel, y * spacing, x * spacing] = 1
                code_w[0, j, 0, 0] = 16 * shape + rotation
                shape_w[0, j, 0, 0] = shape

        matches = relu(conv(glyph_input, w, -8 * np.ones(24)))
        presence = conv(matches, np.ones((1, 24, 1, 1)))
        code = conv(matches, code_w)
        shape_code = conv(matches, shape_w)

        colour_w = np.zeros((4, 16, k, k), np.float32)
        for j, colour in enumerate(colours):
            for y in range(3):
                for x in range(3):
                    colour_w[j, colour, y * spacing, x * spacing] = 1
        colour_presence = clip(conv("x0", colour_w))
        colour_code = conv(
            colour_presence,
            (4 * np.arange(4, dtype=np.float32)).reshape(1, 4, 1, 1)
        )
        return presence, add(code, mul(presence, colour_code)), shape_code

    goal_presence, goal_code, _ = decode_glyph(1)
    _, hud_field, hud_shape_field = decode_glyph(2)
    hud = emit("gmax", [mul(hud_field, hud_mask)])
    hud_shape = emit("gmax", [mul(hud_shape_field, hud_mask)])
    hud_shape_part = scale(hud_shape, 16)
    hud_colour_rotation = sub(hud, hud_shape_part)

    # Rotation, colour, shape, and refill templates.
    sw = np.zeros((4, 16, 3, 3), np.float32)
    for dy, dx in [(-1, 0), (0, 0), (0, 1)]:
        sw[0, 0, dy + 1, dx + 1] = 1
    for dy, dx in [(0, -1), (1, 0)]:
        sw[0, 1, dy + 1, dx + 1] = 1

    colour_pattern = [[9, 14, 14], [9, 0, 8], [12, 12, 8]]
    for y in range(3):
        for x in range(3):
            sw[1, colour_pattern[y][x], y, x] = 1
            sw[3, 3 if (y == 1 and x == 1) else 11, y, x] = 1
    for dy, dx in [(-1, -1), (0, 0), (0, 1), (1, 0)]:
        sw[2, 0, dy + 1, dx + 1] = 1
    station_field = relu(conv("x0", sw, [-4, -8, -3, -8]))

    # The backing line lies opposite the direction of a pusher.
    pw = np.zeros((4, 16, 7, 7), np.float32)
    for j, (dy, dx) in enumerate([(3, 0), (-3, 0), (0, 3), (0, -3)]):
        pw[j, 1, 3 + dy, 3 + dx] = 1
        pw[j, 3, 3, 3] = 1
    push_field = relu(conv("x0", pw, -np.ones(4)))

    aw = np.zeros((1, 16, 5, 5), np.float32)
    aw[0, 12, :2, :] = 1
    aw[0, 9, 2:, :] = 1
    player = extract(relu(conv("x0", aw, [-24])))

    # Map codes:
    # 0 unseen, 1 wall, 2 floor, 3 rotation, 4 colour, 5 shape,
    # 6 battery, 7..10 pushers, 11+inventory-code goal.
    base_w = np.full((1, 16, 1, 1), 2, np.float32)
    base_w[0, 5, 0, 0] = 0
    base_w[0, 4, 0, 0] = 1
    base = conv("x0", base_w)
    special_field = cat([station_field, push_field])
    increments = conv(
        special_field, np.arange(1, 9, dtype=np.float32).reshape(1, 8, 1, 1)
    )
    ordinary = mul(
        add(base, increments), sub(sl("coords", 2, 3), goal_presence)
    )
    observed = extract(add(
        ordinary, add(goal_code, scale(goal_presence, 11))
    ))

    old_map = sl("mem", 0, 100)
    old_visits = sl("mem", 100, 200)
    old_steps = sl("mem", 200, 201)
    old_factor = sl("mem", 201, 202)
    old_battery_mode = sl("mem", 202, 203)
    old_distance = sl("mem", 203, 204)
    old_hud = sl("mem", 204, 205)

    # Tool moments also serve to distinguish occlusion from tool movement.
    px = np.zeros((3, 3 * N), np.float32)
    py = np.zeros_like(px)
    mass_w = np.zeros_like(px)
    tool_repeat_w = np.zeros((3 * N, 3), np.float32)
    for j in range(3):
        px[j, j * N:(j + 1) * N] = cols
        py[j, j * N:(j + 1) * N] = rows
        mass_w[j, j * N:(j + 1) * N] = 1
        tool_repeat_w[j * N:(j + 1) * N, j] = 1

    tool_codes = const(np.repeat([3, 4, 5], N))
    old_tools = equal(sub(cat([old_map] * 3), tool_codes))
    visible_tools = equal(sub(cat([observed] * 3), tool_codes))
    missing_visible = sub(one, clip(dense(visible_tools, mass_w)))

    # Restore a covered stationary tool, but do not leave a ghost copy
    # behind when a moving tool has become visible elsewhere.
    hidden_parts = []
    for j in range(3):
        old_part = sl(old_tools, j * N, (j + 1) * N)
        absent = sl(missing_visible, j, j + 1)
        hidden_parts.append(scale(mul(old_part, absent), j + 1))
    observed = add(observed, mul(player, sum_nodes(hidden_parts)))
    board = add(observed, mul(old_map, sub(one, clip(observed))))

    codes = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    decoded = equal(sub(
        cat([board] * len(codes)), const(np.repeat(codes, N))
    ))
    fields = {
        code: sl(decoded, j * N, (j + 1) * N)
        for j, code in enumerate(codes)
    }
    unknown = fields[0]
    wall = fields[1]
    batteries = fields[6]
    goals = clip(bias(board, -10))
    base_walk = sub(sub(one, wall), goals)
    tools = cat([fields[3], fields[4], fields[5]])

    # One-step extrapolation for mobile tools.
    dx = sub(dense(tools, px), dense(old_tools, px))
    dy = sub(dense(tools, py), dense(old_tools, py))
    velocity = [relu(neg(dy)), relu(dy), relu(neg(dx)), relu(dx)]
    moving = mul(
        equal(sum_nodes(velocity), 1),
        clip(dense(old_tools, mass_w))
    )
    velocity = [mul(v, moving) for v in velocity]
    stationary = sub(one, sum_nodes(velocity))
    prediction_terms = [mul(tools, dense(stationary, tool_repeat_w))]
    for j, (ddy, ddx) in enumerate(directions):
        prediction_terms.append(mul(
            shift_tiles(tools, -ddy, -ddx, 3),
            dense(velocity[j], tool_repeat_w)
        ))
    predicted = mul(sum_nodes(prediction_terms), cat([base_walk] * 3))

    # If extrapolation crosses an obstacle, retain the observed position
    # rather than deleting the tool from the planning map.
    predicted_mass = clip(dense(predicted, mass_w))
    observed_mass = clip(dense(tools, mass_w))
    restore = mul(observed_mass, sub(one, predicted_mass))
    predicted = add(
        predicted, mul(tools, dense(restore, tool_repeat_w))
    )
    rotation_tile = sl(predicted, 0, N)
    colour_tile = sl(predicted, N, 2 * N)
    shape_tile = sl(predicted, 2 * N, 3 * N)

    # Select a shape subgoal only when no remaining goal has our shape.
    relative_goal_code = sub(bias(board, -11), hud_shape_part)
    same_shape_goals = mul(goals, mul(
        clip(bias(relative_goal_code, 1)),
        clip(bias(neg(relative_goal_code), 16))
    ))
    has_goal = clip(total(goals))
    has_shape_goal = clip(total(same_shape_goals))
    shape_needed = mul(has_goal, sub(one, has_shape_goal))

    # Read the energy bar and infer whether moves consume one or two units.
    rw = np.zeros((1, 3, 1, 1), np.float32)
    rw[0, 0, 0, 0] = 63
    row_distance = absolute(conv("coords", rw, [-61]))
    row_peak = relu(bias(scale(row_distance, -2), 1))
    row61 = clip(bias(scale(row_peak, 4), -1))
    steps = scale(
        emit("gmean", [mul(sl("x0", 11, 12), row61)]), 4096
    )
    consumed_units = relu(sub(old_steps, steps))
    consumed = clip(consumed_units)
    prior_factor = bias(relu(bias(old_factor, -0.5)), 0.5)
    new_factor = bias(scale(consumed_units, -0.5), 1.5)
    factor = add(
        mul(prior_factor, sub(one, consumed)),
        mul(new_factor, consumed)
    )
    energy = mul(steps, factor)
    refill = clip(relu(sub(steps, old_steps)))

    complex_tools = clip(add(total(fields[4]), total(fields[5])))
    reserve = bias(scale(complex_tools, 12), 7)
    low_energy = clip(sub(reserve, energy))
    insufficient = clip(bias(sub(old_distance, energy), 2))
    need_reserve = clip(add(insufficient, shape_needed))
    wanted_battery = mul(low_energy, need_reserve)
    battery_mode = mul(clip(total(batteries)), clip(add(
        mul(old_battery_mode, sub(one, refill)), wanted_battery
    )))

    # Intermediate shape goals and refills are terminals for this search.
    shape_terminal = mul(
        shape_tile, mul(shape_needed, sub(one, battery_mode))
    )
    battery_terminal = mul(batteries, battery_mode)
    terminal_extra = add(shape_terminal, battery_terminal)
    walk = relu(sub(sub(base_walk, terminal_extra), unknown))

    # Pusher rays carry the value of the cell immediately before a wall.
    blockers = clip(add(wall, goals))
    stop_masks = []
    go_masks = []
    push_masks = []
    neighbour_masks = []
    active_push_tiles = []
    for j, (ddy, ddx) in enumerate(directions):
        va = const(valid(ddy, ddx))
        next_blocked = add(
            mul(roll(blockers, offsets[j]), va), sub(one, va)
        )
        active = mul(fields[7 + j], sub(one, next_blocked))
        active_push_tiles.append(active)
        stop_masks.append(repeat_tiles(next_blocked))
        go_masks.append(repeat_tiles(sub(one, next_blocked)))
        push_masks.append(repeat_tiles(active))
        neighbour_masks.append(repeat_tiles(va))

    active_push = sum_nodes(active_push_tiles)
    tool_occupancy = sum_nodes([rotation_tile, colour_tile, shape_tile])
    normal_tile = relu(sub(sub(walk, active_push), tool_occupancy))
    # Shape changes do not affect the validity of reaching a battery.
    normal_tile = add(normal_tile, mul(shape_tile, battery_mode))
    normal_gate = repeat_tiles(normal_tile)

    rotation_gate = repeat_tiles(mul(rotation_tile, walk))
    colour_gate = repeat_tiles(mul(colour_tile, walk))

    inventory_ids = np.repeat(np.arange(INVENTORIES), N).astype(np.float32)
    inventory_ids_node = const(inventory_ids)
    rotation_last = const(np.repeat(
        [i % 4 == 3 for i in range(INVENTORIES)], N
    ).astype(np.float32))
    colour_last = const(np.repeat(
        [i // 4 == 3 for i in range(INVENTORIES)], N
    ).astype(np.float32))
    rotation_gates = [
        mul(rotation_gate, sub(one, rotation_last)),
        mul(rotation_gate, rotation_last)
    ]
    colour_gates = [
        mul(colour_gate, sub(one, colour_last)),
        mul(colour_gate, colour_last)
    ]

    goal_compatible = equal(sub(
        repeat_tiles(relative_goal_code), inventory_ids_node
    ))
    goal_compatible = mul(goal_compatible, repeat_tiles(goals))
    seeds = add(
        scale(goal_compatible, 200),
        repeat_tiles(add(scale(terminal_extra, 200), scale(unknown, 20)))
    )
    inventory_mask = equal(sub(inventory_ids_node, hud_colour_rotation))

    # Zero-initialized value iteration eliminates unsupported optimistic
    # cycles. The state is "value on entering this tile".
    outer_cache = expression_cache
    body = []
    current = body
    expression_cache = dict(outer_cache)
    state_names = ["h", "ray_up", "ray_down", "ray_left", "ray_right"]
    for name in state_names:
        tensor_shapes[name] = (L,)

    neighbours = [
        mul(roll("h", offsets[j]), neighbour_masks[j]) for j in range(4)
    ]
    best = maximum(
        maximum(neighbours[0], neighbours[1]),
        maximum(neighbours[2], neighbours[3])
    )
    ordinary_value = relu(bias(best, -1))

    terms = [
        mul(ordinary_value, normal_gate),
        mul(roll(ordinary_value, N), rotation_gates[0]),
        mul(roll(ordinary_value, -3 * N), rotation_gates[1]),
        mul(roll(ordinary_value, 4 * N), colour_gates[0]),
        mul(roll(ordinary_value, -12 * N), colour_gates[1]),
    ]
    next_states = {}
    for j in range(4):
        ray = state_names[j + 1]
        terms.append(mul(ray, push_masks[j]))
        next_states[ray] = add(
            mul("h", stop_masks[j]),
            mul(roll(ray, offsets[j]), go_masks[j])
        )
    next_states["h"] = add(sum_nodes(terms), seeds)

    current = graph
    expression_cache = outer_cache
    graph.append({
        "id": "planner",
        "op": "loop",
        "T": 1,
        "state": (
            [{"id": "h", "init": seeds}] +
            [{"id": name, "zeros": [L]} for name in state_names[1:]]
        ),
        "body": body,
        "next": next_states
    })
    for name in state_names:
        tensor_shapes["planner." + name] = (L,)

    selected_values = mul("planner.h", inventory_mask)

    # Mild visitation penalties only break near-equal route choices.
    same_inventory = equal(sub(hud, old_hud))
    visits = add(player, scale(mul(old_visits, same_inventory), 0.9))
    action_values = []
    for ddy, ddx in directions:
        destination = shift_tiles(player, -ddy, -ddx)
        value = total(mul(selected_values, repeat_tiles(destination)))
        revisit = total(mul(visits, destination))
        action_values.append(sub(value, scale(revisit, 0.12)))
    q = cat(action_values)

    estimated_distance = relu(bias(neg(vector_max(q)), 200))
    estimated_distance = add(
        mul(estimated_distance, sub(one, battery_mode)),
        mul(old_distance, battery_mode)
    )

    output_w = np.zeros((8, 4), np.float32)
    output_w[1:5] = np.eye(4, dtype=np.float32)
    output_b = np.full(8, -1000, np.float32)
    output_b[1:5] = 0
    dense(q, output_w, output_b, name="logits")
    emit("mul", [sl("coords", 2, 3), zero], name="click")
    cat([
        board, visits, steps, factor,
        battery_mode, estimated_distance, hud
    ], name="mem_next")

    # Remove construction-only expressions and parameters.
    def prune(nodes):
        needed = {"logits", "click", "mem_next"}
        kept = []
        for node in reversed(nodes):
            if node["op"] == "loop":
                exports = [
                    node["id"] + "." + s["id"] for s in node["state"]
                ]
                if not any(x in needed for x in exports):
                    continue
                kept.append(node)
                local_names = {b["id"] for b in node["body"]}
                local_names.update(s["id"] for s in node["state"])
                for state in node["state"]:
                    if "init" in state:
                        needed.add(state["init"])
                for b in node["body"]:
                    for x in b.get("in", []):
                        if x not in local_names:
                            needed.add(x)
            elif node["id"] in needed:
                kept.append(node)
                needed.update(node.get("in", []))
        return list(reversed(kept))

    graph = prune(graph)

    def macs(node):
        op = node["op"]
        if op == "conv":
            w = params[node["w"]]
            return int(np.prod(w.shape)) * 4096
        if op == "dense":
            return int(np.prod(params[node["w"]].shape))
        if op in (
            "add", "sub", "mul", "max2", "min2", "vadd", "vmul"
        ):
            return int(np.prod(tensor_shapes[node["id"]]))
        if op in (
            "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"
        ):
            return int(np.prod(tensor_shapes[node["in"][0]]))
        return 0

    fixed_macs = sum(macs(n) for n in graph if n["op"] != "loop")
    fixed_evaluations = len(graph) - 1
    for node in graph:
        if node["op"] == "loop":
            body_macs = sum(macs(n) for n in node["body"])
            by_evaluations = (9900 - fixed_evaluations) // len(node["body"])
            by_macs = (1490000000 - fixed_macs) // max(1, body_macs)
            node["T"] = int(max(1, min(128, by_evaluations, by_macs)))

    used_parameters = set()
    for node in graph:
        nodes = [node] + (
            node["body"] if node["op"] == "loop" else []
        )
        for item in nodes:
            for key in ("w", "b"):
                if key in item:
                    used_parameters.add(item[key])
    params = {k: v for k, v in params.items() if k in used_parameters}

    return {
        "graph": graph,
        "memory_size": 205,
        "params": params,
    }
