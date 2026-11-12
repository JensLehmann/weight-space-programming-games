import numpy as np


def build():
    params = {}
    graph = []
    scopes = [graph]
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "mem": ("v", 57),
        "aprev": ("v", 8),
        "avail": ("v", 8),
    }
    counter = [0]

    def fresh(prefix="n"):
        counter[0] += 1
        return prefix + str(counter[0])

    def parameter(value):
        name = fresh("p")
        params[name] = np.asarray(value, np.float32)
        return name

    def node(op, inputs, name=None, **fields):
        name = name or fresh()
        entry = {"id": name, "op": op, "in": list(inputs)}
        entry.update(fields)
        scopes[-1].append(entry)
        if op == "conv":
            shapes[name] = ("s", params[fields["w"]].shape[0])
        elif op == "dense":
            shapes[name] = ("v", params[fields["w"]].shape[0])
        elif op in ("gmax", "gmean"):
            shapes[name] = ("v", shapes[inputs[0]][1])
        elif op in ("concat", "vconcat"):
            shapes[name] = (
                "s" if op == "concat" else "v",
                sum(shapes[x][1] for x in inputs),
            )
        elif op in ("slice", "vslice"):
            shapes[name] = (
                shapes[inputs[0]][0],
                fields["stop"] - fields["start"],
            )
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shapes[name] = ("s", max(shapes[x][1] for x in inputs))
        elif op in ("vadd", "vmul"):
            shapes[name] = ("v", max(shapes[x][1] for x in inputs))
        else:
            shapes[name] = shapes[inputs[0]]
        return name

    def conv(x, weight, bias=None, name=None):
        fields = {
            "w": weight if isinstance(weight, str) else parameter(weight)
        }
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("conv", [x], name=name, **fields)

    def dense(x, weight, bias=None, name=None):
        weight = np.asarray(weight, np.float32)
        if weight.ndim == 1:
            weight = weight[None, :]
        fields = {"w": parameter(weight)}
        if bias is not None:
            fields["b"] = parameter(
                np.asarray(bias, np.float32).reshape(-1)
            )
        return node("dense", [x], name=name, **fields)

    def ss(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vs(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def cat(*xs):
        return node("concat", xs)

    def vc(*xs):
        return node("vconcat", xs)

    def va(*xs):
        result = xs[0]
        for x in xs[1:]:
            result = node("vadd", [result, x])
        return result

    def vm(a, b):
        return node("vmul", [a, b])

    def sm(a, b):
        return node("mul", [a, b])

    def sa(a, b):
        return node("add", [a, b])

    def vr(x):
        return node("vrelu", [x])

    def sr(x):
        return node("relu", [x])

    def vscale(x, scale, bias=0.0):
        size = shapes[x][1]
        return dense(
            x,
            np.eye(size, dtype=np.float32) * scale,
            np.full(size, bias, np.float32),
        )

    def sscale(x, scale, bias=0.0):
        size = shapes[x][1]
        return conv(
            x,
            (np.eye(size, dtype=np.float32) * scale)[:, :, None, None],
            np.full(size, bias, np.float32),
        )

    one = dense("aprev", np.zeros((1, 8), np.float32), [1.0])
    zero = vscale(one, 0.0)

    def vclip(x):
        positive = vr(x)
        excess = vr(vscale(positive, 1.0, -1.0))
        return va(positive, vscale(excess, -1.0))

    def vnot(x):
        return vscale(x, -1.0, 1.0)

    def sclip(x):
        return node("min2", [sr(x), one])

    def vmax(x):
        return node("gmax", [x])

    def vsum(x):
        return dense(x, np.ones((1, shapes[x][1]), np.float32))

    def ssum(x):
        return conv(
            x, np.ones((1, shapes[x][1], 1, 1), np.float32)
        )

    xy = ss("coords", 0, 2)
    spatial_zero = conv(
        "coords", np.zeros((1, 3, 1, 1), np.float32)
    )

    def position(mask):
        means = node("gmean", [sm(xy, mask)])
        return dense(
            means, np.eye(2, dtype=np.float32) * 4096.0
        )

    def point_mask(pos):
        delta = node("abs", [node("sub", [xy, pos])])
        return sclip(conv(
            delta,
            np.full((1, 2, 1, 1), -252.0, np.float32),
            [2.0],
        ))

    # Fixed movement lattice, independent of level layout.
    weight = np.zeros((20, 3, 1, 1), np.float32)
    bias = np.zeros(20, np.float32)
    for i in range(10):
        weight[i, 0, 0, 0] = 63.0
        weight[10 + i, 1, 0, 0] = 63.0
        bias[i] = bias[10 + i] = -(4 + 6 * i)
    distances = node("abs", [conv("coords", weight, bias)])
    triangles = sr(sscale(distances, -1.0, 1.1))
    weight = np.zeros((2, 20, 1, 1), np.float32)
    weight[0, :10, 0, 0] = 1.0
    weight[1, 10:, 0, 0] = 1.0
    axes = sclip(sscale(conv(triangles, weight), 4.0, -2.0))
    lattice = sm(ss(axes, 0, 1), ss(axes, 1, 2))

    # Central 5x5 terrain test. Decorative three-pixel-wide wires
    # cannot fill this window, including at ordinary wire crossings.
    weight = np.zeros((1, 16, 5, 5), np.float32)
    weight[0, 1:, :, :] = 1.0
    floor = sm(sclip(conv("x0", weight, [-24.0])), lattice)

    # Player, ghost and patrol centre detectors.
    weight = np.zeros((3, 16, 5, 5), np.float32)
    for i, colour in enumerate((9, 2, 14)):
        weight[i, colour, :, :] = 1.0
        weight[i, colour, 2, 2] = 0.0
        weight[i, 5, 2, 2] = 1.0
    actors = sm(
        sclip(conv("x0", weight, [-24.0] * 3)), lattice
    )
    player = ss(actors, 0, 1)
    enemy = ss(actors, 2, 3)
    player_pos = position(player)
    enemy_pos = position(enemy)

    colours = (8, 11, 15)

    # Small switches, allowing one wire extending from the centre.
    weight = np.zeros((3, 16, 5, 5), np.float32)
    for i, colour in enumerate(colours):
        weight[i, colour, 1:4, 1:4] = 1.0
        for row, col in ((0, 2), (4, 2), (2, 0), (2, 4)):
            weight[i, colour, row, col] = -1.0
    switches = sm(
        sclip(conv("x0", weight, [-7.0] * 3)), lattice
    )

    # Large solid gate bodies.
    weight = np.zeros((3, 16, 5, 5), np.float32)
    for i, colour in enumerate(colours):
        weight[i, colour, :, :] = 1.0
    gates = sm(
        sclip(conv("x0", weight, [-22.0] * 3)), lattice
    )
    all_gates = sclip(ssum(gates))

    # Exit dot.
    weight = np.zeros((1, 16, 3, 3), np.float32)
    weight[0, 9, 1, 1] = 1.0
    for row, col in ((0, 1), (2, 1), (1, 0), (1, 2)):
        weight[0, 5, row, col] = 1.0
    goal = sm(
        sclip(conv("x0", weight, [-4.0])), lattice
    )

    # Portal frames.
    weight = np.zeros((1, 16, 7, 7), np.float32)
    weight[0, 5, 3, 3] = 1.0
    for row, col in ((0, 3), (6, 3), (3, 0), (3, 6)):
        weight[0, 15, row, col] = 1.0
    portals = sm(
        sclip(conv("x0", weight, [-4.0])), lattice
    )

    weight = np.zeros((3, 16, 1, 1), np.float32)
    for i, colour in enumerate(colours):
        weight[i, colour, 0, 0] = 1.0
    wires = conv("x0", weight)

    offsets = ((0, 3), (6, 3), (3, 0), (3, 6))
    weight = np.zeros((4, 1, 7, 7), np.float32)
    for i, (row, col) in enumerate(offsets):
        weight[i, 0, row, col] = 1.0
    shift_key = parameter(weight)

    def max_directions(x, channels=1):
        parts = [
            ss(x, i * channels, (i + 1) * channels)
            for i in range(4)
        ]
        a = node("max2", parts[:2])
        b = node("max2", parts[2:])
        return node("max2", [a, b])

    def neighbours(x):
        intermediate = max_directions(conv(x, shift_key))
        return max_directions(conv(intermediate, shift_key))

    def wave(seed, allowed, teleport=False):
        loop_id = fresh("wave")
        state_id = fresh("state")
        shapes[state_id] = ("s", 1)
        body = []
        scopes.append(body)
        value = node("sub", [neighbours(state_id), one])
        if teleport:
            portal_value = vscale(
                vmax(sm(state_id, portals)), 1.0, -1.0
            )
            value = node(
                "max2", [value, sm(portals, portal_value)]
            )
        value = node("max2", [value, seed])
        next_value = sm(value, allowed)
        scopes.pop()
        scopes[-1].append({
            "id": loop_id,
            "op": "loop",
            "T": 96,
            "state": [{"id": state_id, "init": seed}],
            "body": body,
            "next": {state_id: next_value},
        })
        result = loop_id + "." + state_id
        shapes[result] = ("s", 1)
        return result

    # Connectivity ignores the patrol; hazards enter the action head.
    free = sm(floor, sscale(all_gates, -1.0, 1.0))
    free = node("max2", [free, player])
    reach = wave(sscale(player, 128.0), free)
    optimistic = wave(
        sscale(goal, 128.0), floor, teleport=True
    )
    goal_reachable = vclip(vmax(sm(reach, goal)))
    portal_reach_value = vmax(sm(reach, portals))
    portal_available = vclip(portal_reach_value)

    # Spread relevance from gates and portals along matching wires.
    source = sm(gates, optimistic)
    portal_source = sm(portals, optimistic)
    source = sa(
        source, cat(spatial_zero, spatial_zero, portal_source)
    )
    weight = np.zeros((3, 3, 7, 7), np.float32)
    for i in range(3):
        weight[i, i, :, :] = 1.0
    wire_drive = sm(conv(source, weight), wires)

    weight = np.zeros((12, 3, 3, 3), np.float32)
    for direction, (row, col) in enumerate(
        ((0, 1), (2, 1), (1, 0), (1, 2))
    ):
        for i in range(3):
            weight[direction * 3 + i, i, row, col] = 1.0
    wire_shift_key = parameter(weight)
    wire_cost = vscale(one, 0.01)

    loop_id = fresh("wire")
    state_id = fresh("wirestate")
    shapes[state_id] = ("s", 3)
    body = []
    scopes.append(body)
    value = max_directions(
        conv(state_id, wire_shift_key), channels=3
    )
    value = node("sub", [value, wire_cost])
    value = node("max2", [value, wire_drive])
    next_value = sm(value, wires)
    scopes.pop()
    scopes[-1].append({
        "id": loop_id,
        "op": "loop",
        "T": 128,
        "state": [{"id": state_id, "init": wire_drive}],
        "body": body,
        "next": {state_id: next_value},
    })
    wire_value = loop_id + "." + state_id
    shapes[wire_value] = ("s", 3)

    # Memory layout:
    # 0 init; 1 recording length; 2 active endpoint;
    # 3:5 endpoint position; 5:8 endpoint type;
    # 8 delay; 9 return-leg flag; 10 completed-endpoint count;
    # 11:27 eight positions; 27:51 eight type triples;
    # 51:53 previous player; 53 portal-recording length;
    # 54 portal-transfer flag; 55:57 previous patrol.
    initialized = vs("mem", 0, 1)
    old_run = vs("mem", 1, 2)
    old_active = vs("mem", 2, 3)
    old_target_pos = vs("mem", 3, 5)
    old_type = vs("mem", 5, 8)
    old_delay = vs("mem", 8, 9)
    old_return = vs("mem", 9, 10)
    completed_count = vs("mem", 10, 11)
    saved_rows = vs("mem", 11, 19)
    saved_cols = vs("mem", 19, 27)
    saved_red = vs("mem", 27, 35)
    saved_blue = vs("mem", 35, 43)
    saved_white = vs("mem", 43, 51)
    previous_pos = vs("mem", 51, 53)
    old_white_length = vs("mem", 53, 54)
    old_transferred = vs("mem", 54, 55)
    previous_enemy_pos = vs("mem", 55, 57)

    delta = dense(
        vc(player_pos, previous_pos),
        [[1, 0, -1, 0], [0, 1, 0, -1]],
    )
    absolute_delta = va(vr(delta), vr(vscale(delta, -1.0)))
    travelled = vsum(absolute_delta)
    previous_move = vsum(vs("aprev", 1, 5))
    moved = vm(
        previous_move,
        vclip(vscale(travelled, 126.0, -1.0)),
    )
    run = va(old_run, moved)

    predicted_enemy_pos = dense(
        vc(enemy_pos, previous_enemy_pos),
        [[2, 0, -1, 0], [0, 2, 0, -1]],
    )
    prediction_enabled = vm(
        vm(previous_move, initialized),
        vclip(vmax(enemy)),
    )
    predicted_enemy = sm(
        point_mask(predicted_enemy_pos), prediction_enabled
    )

    white_assigned = vclip(vsum(saved_white))
    jumped = vm(
        previous_move, vclip(vscale(travelled, 63.0, -8.0))
    )
    transferred = vclip(
        va(old_transferred, vm(jumped, white_assigned))
    )

    saved_flags = va(saved_red, saved_blue, saved_white)
    weight = np.zeros((16, 3, 1, 1), np.float32)
    weight[:8, 0, 0, 0] = 1.0
    weight[8:, 1, 0, 0] = 1.0
    duplicated_xy = conv("coords", weight)
    delta = node("abs", [node("sub", [
        duplicated_xy, vc(saved_rows, saved_cols)
    ])])
    weight = np.zeros((8, 16, 1, 1), np.float32)
    for i in range(8):
        weight[i, i, 0, 0] = -252.0
        weight[i, 8 + i, 0, 0] = -252.0
    saved_maps = sm(
        sclip(conv(delta, weight, [2.0] * 8)),
        saved_flags,
    )
    assigned_map = sclip(ssum(saved_maps))
    active_map = point_mask(old_target_pos)

    # UI-based recording capacity.
    weight = np.zeros((1, 3, 1, 1), np.float32)
    weight[0, 0, 0, 0] = -126.0
    top = sclip(conv("coords", weight, [7.0]))
    weight = np.zeros((1, 3, 1, 1), np.float32)
    weight[0, 1, 0, 0] = -126.0
    left = sclip(conv("coords", weight, [31.0]))
    weight = np.zeros((1, 16, 1, 1), np.float32)
    weight[0, [1, 2, 9], 0, 0] = 1.0
    icon_pixels = sm(
        conv("x0", weight), sm(top, left)
    )
    icon_count = vscale(
        node("gmean", [icon_pixels]), 4096.0 / 9.0
    )
    ghost_count = vsum(va(saved_red, saved_white))
    recording_allowed = vclip(dense(
        vc(icon_count, ghost_count),
        [[4.0, -4.0]], [-4.5],
    ))

    def pick(score, mask):
        maximum = vmax(score)
        difference = node("sub", [score, maximum])
        best = sclip(sscale(difference, 100000.0, 1.0))
        return sm(best, mask)

    weight = np.zeros((1, 3, 1, 1), np.float32)
    weight[0, 0, 0, 0] = 0.137
    weight[0, 1, 0, 0] = 0.001731
    tie_break = conv("coords", weight)

    portal_candidates = sm(portals, sclip(reach))
    portal_scores = sm(
        sa(sa(optimistic, reach), tie_break),
        portal_candidates,
    )
    selected_portal = pick(
        portal_scores, portal_candidates
    )

    goal_mode = vm(goal_reachable, vnot(old_active))
    portal_mode = vm(
        vm(
            vm(
                vm(white_assigned, vnot(transferred)),
                vnot(goal_mode),
            ),
            portal_available,
        ),
        vnot(old_active),
    )
    switch_mode = vm(vnot(goal_mode), vnot(portal_mode))

    white_allowed = vm(
        recording_allowed, portal_available
    )
    permitted_types = vc(
        recording_allowed, one, white_allowed
    )
    candidates = ssum(sm(switches, permitted_types))
    candidates = sm(candidates, sclip(reach))
    candidates = sm(
        candidates, sscale(assigned_map, -1.0, 1.0)
    )
    switch_interest = ssum(sm(wire_value, switches))
    candidates = sm(
        candidates, sclip(switch_interest)
    )
    scores = sm(
        sa(sa(switch_interest, reach), tie_break),
        candidates,
    )
    selected_switch = pick(scores, candidates)
    selection_exists = vclip(vmax(selected_switch))
    selected_pos = position(selected_switch)
    selected_type = vmax(
        sm(switches, selected_switch)
    )

    new_target = vm(
        vm(vnot(old_active), selection_exists),
        switch_mode,
    )
    target_exists = vclip(va(old_active, new_target))
    effective_pos = va(
        vm(old_target_pos, old_active),
        vm(selected_pos, vnot(old_active)),
    )
    effective_type = va(
        vm(old_type, old_active),
        vm(selected_type, vnot(old_active)),
    )
    switch_target = sa(
        sm(active_map, old_active),
        sm(selected_switch, vnot(old_active)),
    )

    # Portal-trigger recordings include a travel-time allowance.
    selected_reach = vmax(
        sm(reach, selected_switch)
    )
    required_delay = vr(dense(
        vc(selected_reach, portal_reach_value),
        [[1.0, -1.0]], [6.0],
    ))
    effective_delay = va(
        vm(old_delay, old_active),
        vm(
            vm(required_delay, new_target),
            vs(selected_type, 2, 3),
        ),
    )
    effective_return = vm(old_return, old_active)
    delay_positive = vclip(
        vscale(effective_delay, 100.0)
    )
    padding = vm(delay_positive, switch_mode)
    return_leg = vm(padding, effective_return)
    away_leg = vm(padding, vnot(effective_return))

    arrived = vm(
        vm(
            vclip(vmax(sm(player, switch_target))),
            switch_mode,
        ),
        vnot(delay_positive),
    )
    red_type = vs(effective_type, 0, 1)
    blue_type = vs(effective_type, 1, 2)
    white_type = vs(effective_type, 2, 3)
    record_now = vm(
        arrived, va(red_type, white_type)
    )
    white_record = vm(record_now, white_type)

    target = sa(
        sm(goal, goal_mode),
        sa(
            sm(selected_portal, portal_mode),
            sm(switch_target, switch_mode),
        ),
    )
    target_present = vclip(vmax(target))
    target_value = wave(
        sscale(target, 128.0), free
    )
    planning_value = sa(
        target_value,
        sm(optimistic, vnot(target_present)),
    )

    # Four direction-preserving six-pixel samples.
    weight = np.zeros((4, 4, 7, 7), np.float32)
    for i, (row, col) in enumerate(offsets):
        weight[i, i, row, col] = 1.0
    directional_second = parameter(weight)

    def directional_sample(x):
        shifted = conv(
            conv(x, shift_key), directional_second
        )
        return vmax(sm(shifted, player))

    next_values = directional_sample(planning_value)
    next_valid = directional_sample(free)
    next_danger = directional_sample(neighbours(enemy))
    next_enemy = directional_sample(enemy)
    next_prediction = directional_sample(predicted_enemy)
    switch_obstacles = sclip(
        sa(ssum(switches), switch_target)
    )
    next_switch = directional_sample(switch_obstacles)

    validity_penalty = vscale(
        next_valid, 10000.0, -10000.0
    )
    danger_penalty = va(
        vscale(next_danger, -0.05),
        vscale(next_enemy, -100.0),
        vscale(next_prediction, -3000.0),
    )
    normal_scores = va(
        next_values, validity_penalty, danger_penalty
    )
    away_scores = va(
        vscale(next_values, -1.0),
        validity_penalty,
        danger_penalty,
        vscale(next_switch, -500.0),
    )
    reverse_weight = np.zeros((4, 8), np.float32)
    reverse_weight[0, 2] = 2000.0
    reverse_weight[1, 1] = 2000.0
    reverse_weight[2, 4] = 2000.0
    reverse_weight[3, 3] = 2000.0
    return_scores = va(
        dense("aprev", reverse_weight),
        validity_penalty,
        danger_penalty,
    )
    move_scores = va(
        vm(normal_scores, vnot(padding)),
        vm(away_scores, away_leg),
        vm(return_scores, return_leg),
    )

    overdue = vclip(dense(
        vc(run, old_white_length),
        [[1.0, -1.0]], [-2.0],
    ))
    missed_portal = vm(
        vm(
            vm(overdue, white_assigned),
            vnot(transferred),
        ),
        vnot(goal_mode),
    )
    reset_request = vclip(
        va(vnot(initialized), missed_portal)
    )
    actual_reset = vm(
        reset_request, vs("avail", 0, 1)
    )

    weight = np.zeros((8, 6), np.float32)
    bias = np.full(8, -100000.0, np.float32)
    weight[0, 5] = 50000.0
    bias[0] = -20000.0
    for i in range(4):
        weight[i + 1, i] = 1.0
        bias[i + 1] = 0.0
    weight[5, 4] = 30000.0
    bias[5] = -20000.0
    dense(
        vc(move_scores, record_now, reset_request),
        weight, bias, name="logits",
    )

    # Endpoint-memory slot selection and update.
    lower = vclip(dense(
        completed_count,
        np.ones((8, 1), np.float32),
        1.0 - np.arange(8, dtype=np.float32),
    ))
    upper = vclip(dense(
        completed_count,
        np.ones((8, 1), np.float32),
        -np.arange(8, dtype=np.float32),
    ))
    slot = va(lower, vscale(upper, -1.0))
    write = vm(slot, arrived)
    retain = vnot(write)

    new_rows = va(
        vm(saved_rows, retain),
        vm(write, vs(effective_pos, 0, 1)),
    )
    new_cols = va(
        vm(saved_cols, retain),
        vm(write, vs(effective_pos, 1, 2)),
    )
    new_red = va(
        vm(saved_red, retain), vm(write, red_type)
    )
    new_blue = va(
        vm(saved_blue, retain), vm(write, blue_type)
    )
    new_white = va(
        vm(saved_white, retain), vm(write, white_type)
    )

    active_next = vm(
        vm(target_exists, switch_mode), vnot(arrived)
    )
    delay_next = vm(
        vr(va(
            effective_delay,
            vscale(return_leg, -2.0),
        )),
        vm(switch_mode, vnot(arrived)),
    )
    run_next = vm(run, vnot(record_now))
    white_length_next = va(
        vm(old_white_length, vnot(white_record)),
        vm(run, white_record),
    )
    transferred_next = vm(
        transferred, vnot(white_record)
    )

    memory_candidate = vc(
        zero,
        run_next,
        active_next,
        effective_pos,
        vm(effective_type, active_next),
        delay_next,
        away_leg,
        va(completed_count, arrived),
        new_rows,
        new_cols,
        new_red,
        new_blue,
        new_white,
        player_pos,
        white_length_next,
        transferred_next,
        enemy_pos,
    )
    kept_memory = vm(
        memory_candidate, vnot(actual_reset)
    )
    init_bias = np.zeros(57, np.float32)
    init_bias[0] = 1.0
    init_vector = dense(
        "aprev",
        np.zeros((57, 8), np.float32),
        init_bias,
    )
    node(
        "vadd", [kept_memory, init_vector],
        name="mem_next",
    )
    conv(
        "coords",
        np.zeros((1, 3, 1, 1), np.float32),
        name="click",
    )

    return {
        "graph": graph,
        "memory_size": 57,
        "params": params,
    }
