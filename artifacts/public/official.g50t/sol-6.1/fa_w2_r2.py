import numpy as np


def build():
    graph, params = [], {}
    channels = {
        "x0": 16, "coords": 3, "mem": 30,
        "aprev": 8, "avail": 8,
    }
    destination = graph
    serial = 0
    pserial = 0

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def parameter(a):
        nonlocal pserial
        pserial += 1
        name = "p" + str(pserial)
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, ins, ident=None, **fields):
        ident = fresh() if ident is None else ident
        d = {"id": ident, "op": op, "in": list(ins)}
        d.update(fields)
        destination.append(d)
        if op in ("concat", "vconcat"):
            channels[ident] = sum(channels[x] for x in ins)
        elif op in ("slice", "vslice"):
            channels[ident] = fields["stop"] - fields["start"]
        elif op in ("conv", "dense"):
            channels[ident] = params[fields["w"]].shape[0]
        elif op in (
            "add", "sub", "mul", "max2", "min2", "vadd", "vmul"
        ):
            channels[ident] = max(channels[x] for x in ins)
        else:
            channels[ident] = channels[ins[0]]
        return ident

    def cat(xs, vector=False, ident=None):
        if len(xs) == 1 and ident is None:
            return xs[0]
        return node(
            "vconcat" if vector else "concat", xs, ident=ident
        )

    def sl(x, a, b, vector=False):
        return node(
            "vslice" if vector else "slice", [x], start=a, stop=b
        )

    def conv(x, w, b=None, ident=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], ident=ident, **fields)

    def dense(x, w, b=None, ident=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], ident=ident, **fields)

    def S(xs, coefficients, bias=0.0):
        x = cat(xs)
        w = np.asarray(coefficients, np.float32).reshape(
            1, channels[x], 1, 1
        )
        return conv(x, w, [bias])

    def V(xs, coefficients, bias=0.0):
        x = cat(xs, True)
        w = np.asarray(coefficients, np.float32).reshape(
            1, channels[x]
        )
        return dense(x, w, [bias])

    def D(xs, w, b=None, ident=None):
        return dense(cat(xs, True), w, b, ident)

    def add(a, b):
        return node("add", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def maximum(xs):
        r = xs[0]
        for x in xs[1:]:
            r = node("max2", [r, x])
        return r

    one = dense("avail", np.zeros((1, 8), np.float32), [1])
    spatial_one = sl("coords", 2, 3)

    def spositive(x):
        return node("min2", [node("relu", [x]), one])

    def vclip(x):
        n = channels[x]
        eye = np.eye(n, dtype=np.float32)
        a = node("vrelu", [x])
        b = node("vrelu", [
            dense(x, eye, -np.ones(n, np.float32))
        ])
        return D([a, b], np.concatenate([eye, -eye], axis=1))

    def vnot(x):
        n = channels[x]
        return dense(
            x, -np.eye(n, dtype=np.float32),
            np.ones(n, np.float32)
        )

    def vge(a, b):
        return vclip(V([a, b], [100, -100], 1))

    def pool(x, mask=None, mean=False):
        if mask is not None:
            x = mul(x, mask)
        return node("gmean" if mean else "gmax", [x])

    def sample(x):
        return pool(x, avatar)

    def vector_max(x):
        lifted = mul(spatial_one, x)
        return pool(maximum([
            sl(lifted, i, i + 1) for i in range(channels[x])
        ]))

    # Six-pixel lattice neighbours, using two three-pixel shifts.
    offsets = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    w_first = np.zeros((4, 1, 7, 7), np.float32)
    for i, (dy, dx) in enumerate(offsets):
        w_first[i, 0, 3 + dy, 3 + dx] = 1
        w = np.zeros((1, 1, 7, 7), np.float32)
        w[0, 0, 3 + dy, 3 + dx] = 1
        params["shift_second_" + str(i)] = w
    params["shift_first"] = w_first

    def neighbors6(x):
        first = node("conv", [x], w="shift_first")
        return [
            node(
                "conv", [sl(first, i, i + 1)],
                w="shift_second_" + str(i)
            )
            for i in range(4)
        ]

    def detector(patterns, k):
        w = np.zeros((len(patterns), 16, k, k), np.float32)
        b = []
        for o, pattern in enumerate(patterns):
            positive = 0
            for dy, dx, colour, weight in pattern:
                w[o, colour, k // 2 + dy, k // 2 + dx] += weight
                if weight > 0:
                    positive += weight
            b.append(-positive + 0.5)
        z = node("relu", [conv("x0", w, b)])
        return conv(
            z,
            (2 * np.eye(len(patterns), dtype=np.float32))[
                :, :, None, None
            ]
        )

    block_patterns = []
    for colour in (9, 14):
        block_patterns.append([
            (dy, dx, 5 if dy == 0 and dx == 0 else colour, 1)
            for dy in range(-2, 3) for dx in range(-2, 3)
        ])
    blocks = detector(block_patterns, 5)
    avatar = sl(blocks, 0, 1)
    enemy = sl(blocks, 1, 2)

    goal = detector([[
        (dy, dx, 9 if dy == 0 and dx == 0 else 5, 1)
        for dy in range(-2, 3) for dx in range(-2, 3)
    ]], 5)

    plate_patterns = []
    for colour in (8, 11, 15):
        pattern = [
            (dy, dx, colour, 1)
            for dy in range(-1, 2) for dx in range(-1, 2)
        ]
        pattern += [
            (dy, dx, colour, -1)
            for dy in (-2, 2) for dx in (-2, 2)
        ]
        plate_patterns.append(pattern)
    plates = detector(plate_patterns, 5)
    red_plate = sl(plates, 0, 1)
    blue_plate = sl(plates, 1, 2)
    white_plate = sl(plates, 2, 3)
    plate_any = S([plates], [1, 1, 1])

    # The portal border survives an actor occupying its interior.
    portal_pattern = [(0, 0, 5, 1)]
    portal_pattern += [
        (dy, dx, 15, 1)
        for dy, dx in (
            (-3, -3), (-3, 0), (-3, 3),
            (0, -3), (0, 3),
            (3, -3), (3, 0), (3, 3)
        )
    ]
    portals = detector([portal_pattern], 7)

    floor_w = np.zeros((1, 16, 5, 5), np.float32)
    floor_w[0, 1:, :, :] = 1
    floor = S([
        node("relu", [conv("x0", floor_w, [-24.5])])
    ], [2])
    board = spositive(S([sl("coords", 0, 1)], [63], -6))
    floor = mul(floor, board)
    floor_neighbors = neighbors6(floor)

    red_template = np.array([
        [5, 5, 5, 5, 5, 5, 5],
        [5, 8, 5, 8, 5, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 5, 5, 8, 5, 5, 5],
    ], np.int32)
    blue_template = np.array([
        [5, 5, 5, 5, 5, 5, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 5, 5, 11, 5, 5, 5],
    ], np.int32)
    gate_patterns = []
    for template in (red_template, blue_template):
        for rotation in range(4):
            a = np.rot90(template, -rotation)
            gate_patterns.append([
                (y - 3, x - 3, int(a[y, x]), 1)
                for y in range(7) for x in range(7)
            ])
    gates = detector(gate_patterns, 7)
    red_gates = maximum([sl(gates, i, i + 1) for i in range(4)])
    blue_gates = maximum([
        sl(gates, i, i + 1) for i in range(4, 8)
    ])
    all_gates = maximum([red_gates, blue_gates])

    junction = spositive(S(floor_neighbors, [1, 1, 1, 1], -1))
    red_closed = mul(red_gates, junction)
    blue_closed = mul(blue_gates, junction)
    closed = maximum([red_closed, blue_closed])
    passable = mul(floor, S([all_gates], [-1], 1))

    # UI indicators provide the number of slots and recorded ghosts.
    ui_mask = spositive(S([sl("coords", 0, 1)], [-63], 6))
    ui_w = np.zeros((1, 16, 3, 3), np.float32)
    ghost_ui_w = np.zeros_like(ui_w)
    for y in range(3):
        for x in range(3):
            if (y, x) != (1, 1):
                ui_w[0, [1, 2, 9], y, x] = 1
                ghost_ui_w[0, 2, y, x] = 1
    ghost_ui_w[0, 0, 1, 1] = 1
    ui_icons = S([
        node("relu", [conv("x0", ui_w, [-7.5])])
    ], [2])
    ui_ghosts = S([
        node("relu", [conv("x0", ghost_ui_w, [-8.5])])
    ], [2])
    total_slots = V([pool(ui_icons, ui_mask, True)], [4096])
    ghost_count = V([pool(ui_ghosts, ui_mask, True)], [4096])

    red_hit_old = sl("mem", 1, 2, True)
    white_hit_old = sl("mem", 2, 3, True)
    white_active_old = sl("mem", 3, 4, True)
    padding_count = sl("mem", 4, 5, True)
    padding_parity = sl("mem", 5, 6, True)
    old_red_xy = sl("mem", 6, 14, True)
    old_blue_xy = sl("mem", 14, 30, True)
    coord_row = sl("coords", 0, 1)
    coord_col = sl("coords", 1, 2)

    def remembered_map(vector, count):
        maps = []
        for i in range(count):
            r = sl(vector, 2 * i, 2 * i + 1, True)
            c = sl(vector, 2 * i + 1, 2 * i + 2, True)
            dr = node("abs", [node("sub", [coord_row, r])])
            dc = node("abs", [node("sub", [coord_col, c])])
            maps.append(node("relu", [
                S([dr, dc], [-63, -63], 1)
            ]))
        return maximum(maps)

    remembered_red_soft = remembered_map(old_red_xy, 4)
    remembered_blue_soft = remembered_map(old_blue_xy, 8)

    # Thresholding removes floating-point residue at remembered cells.
    remembered_red = spositive(
        S([remembered_red_soft], [100], -50)
    )
    remembered_blue = spositive(
        S([remembered_blue_soft], [100], -50)
    )
    new_red = S([remembered_red], [-1], 1)
    new_blue = S([remembered_blue], [-1], 1)
    unrecorded_red_plate = mul(red_plate, new_red)
    unvisited_blue_plate = mul(blue_plate, new_blue)

    def distance(seed, valid, cost=None, portal_links=None,
                 iterations=72):
        nonlocal destination
        loop_id = fresh("L")
        state_id = fresh("s")
        channels[state_id] = 1
        body = []
        previous_destination = destination
        destination = body
        m = maximum(neighbors6(state_id))
        if portal_links is not None:
            at_portal = pool(mul(state_id, portal_links))
            jumped = mul(portal_links, at_portal)
            jumped = node("relu", [S([jumped], [1], -20)])
            m = maximum([m, jumped])
        if cost is None:
            z = S([m], [1], -1)
        else:
            z = S([m, cost], [1, -1])
        z = mul(node("relu", [z]), valid)
        z = maximum([z, seed])
        destination = previous_destination
        destination.append({
            "id": loop_id,
            "op": "loop",
            "T": iterations,
            "state": [{"id": state_id, "init": seed}],
            "body": body,
            "next": {state_id: z},
        })
        result = loop_id + "." + state_id
        channels[result] = 1
        return result

    goal_closed = distance(S([goal], [1000]), passable)
    gate_cost = S([closed], [8], 1)
    goal_open = distance(
        S([goal], [200]), floor, gate_cost, portals
    )
    reachable = distance(S([avatar], [200]), passable)

    goal_reachable = vclip(sample(goal_closed))
    white_active = vmul(white_active_old, vnot(goal_reachable))
    portal_goal_connected = vclip(pool(mul(goal_closed, portals)))
    entrance_portals = mul(
        portals, S([spositive(goal_closed)], [-1], 1)
    )

    reach_neighbor = maximum(neighbors6(reachable))
    frontier = spositive(reach_neighbor)
    open_connected = spositive(goal_open)
    gate_score = node("relu", [
        S([goal_open, reach_neighbor], [1, 1], -200)
    ])
    gate_score = mul(mul(gate_score, frontier), open_connected)
    red_source = mul(gate_score, red_closed)
    blue_source = mul(gate_score, blue_closed)

    portal_score = node("relu", [
        S([goal_open, reachable], [1, 1], -220)
    ])
    portal_score = mul(
        mul(portal_score, entrance_portals), portal_goal_connected
    )
    white_source = conv(
        portal_score, np.ones((1, 1, 7, 7), np.float32)
    )

    # Three independent colour-constrained wire propagation channels.
    wire_mask = cat([
        sl("x0", 8, 9), sl("x0", 11, 12), sl("x0", 15, 16)
    ])
    wire_seed = mul(
        cat([red_source, blue_source, white_source]), wire_mask
    )
    wire_w = np.zeros((12, 3, 3, 3), np.float32)
    for d, (dy, dx) in enumerate(
        ((-1, 0), (1, 0), (0, -1), (0, 1))
    ):
        for c in range(3):
            wire_w[3 * d + c, c, 1 + dy, 1 + dx] = 1
    params["wire_shift"] = wire_w

    wire_loop = fresh("L")
    wire_state = fresh("s")
    channels[wire_state] = 3
    wire_body = []
    previous_destination = destination
    destination = wire_body
    wn = node("conv", [wire_state], w="wire_shift")
    wm = maximum([sl(wn, 3 * i, 3 * i + 3) for i in range(4)])
    wm = conv(
        wm, np.eye(3, dtype=np.float32)[:, :, None, None],
        [-0.01, -0.01, -0.01]
    )
    wm = mul(node("relu", [wm]), wire_mask)
    wm = maximum([wm, wire_seed])
    destination = previous_destination
    destination.append({
        "id": wire_loop,
        "op": "loop",
        "T": 112,
        "state": [{"id": wire_state, "init": wire_seed}],
        "body": wire_body,
        "next": {wire_state: wm},
    })
    wire_result = wire_loop + "." + wire_state
    channels[wire_result] = 3
    wire_red = sl(wire_result, 0, 1)
    wire_blue = sl(wire_result, 1, 2)
    wire_white = sl(wire_result, 2, 3)

    queued = vclip(pool(
        mul(mul(wire_red, remembered_red), red_plate)
    ))
    not_queued = vnot(queued)

    red_priority = S(
        [new_red, wire_red, spositive(wire_red)], [100, 1, 300]
    )
    red_seed = mul(
        mul(unrecorded_red_plate, red_priority), not_queued
    )
    blue_priority = S(
        [new_blue, wire_blue, spositive(wire_blue)], [100, 1, 300]
    )
    blue_seed = mul(mul(blue_plate, blue_priority), not_queued)
    white_priority = S(
        [wire_white, spositive(wire_white)], [1, 600]
    )
    white_seed_before = mul(
        mul(mul(white_plate, white_priority), not_queued),
        portal_goal_connected
    )
    white_seed = add(
        mul(white_seed_before, vnot(white_active)),
        mul(S([entrance_portals], [1100]), white_active)
    )

    red_field = distance(red_seed, passable)
    blue_field = distance(blue_seed, passable)
    white_field = distance(white_seed, passable)
    navigation = S([goal_open], [0.05])
    all_field = maximum([
        goal_closed, red_field, blue_field, white_field, navigation
    ])

    red_current = sample(red_field)
    white_current = sample(white_field)
    red_other_current = sample(maximum([
        goal_closed, blue_field, white_field, navigation
    ]))
    white_other_current = sample(maximum([
        goal_closed, red_field, blue_field, navigation
    ]))
    red_task = vmul(
        vge(red_current, red_other_current),
        vclip(V([red_current], [1], -20))
    )
    white_task = vmul(
        vge(white_current, white_other_current),
        vclip(V([white_current], [1], -20))
    )
    white_wanted = vmul(white_task, vnot(white_active))

    # Reversible staging delays a portal ghost's switch arrival.
    padding_request = vclip(V(
        [white_wanted, padding_parity], [1, 1]
    ))
    padding_left = vclip(V([padding_count], [-1], 16))
    padding = vmul(
        vmul(padding_request, padding_left), vnot(white_active)
    )

    direction_values = cat([
        sample(x) for x in neighbors6(all_field)
    ], True)
    direction_valid = cat([
        sample(x) for x in neighbors6(passable)
    ], True)
    direction_plates = cat([
        sample(x) for x in neighbors6(plate_any)
    ], True)

    enemy_near = spositive(add(enemy, maximum(neighbors6(enemy))))
    danger = cat([
        sample(x) for x in neighbors6(enemy_near)
    ], True)

    reverse_w = np.zeros((4, 8), np.float32)
    reverse_w[0, 2] = 1
    reverse_w[1, 1] = 1
    reverse_w[2, 4] = 1
    reverse_w[3, 3] = 1
    reverse_action = dense("aprev", reverse_w)
    reverse_bonus = vmul(reverse_action, padding_parity)

    eye4 = np.eye(4, dtype=np.float32)
    padding_values = D(
        [direction_plates, reverse_bonus],
        np.concatenate([-20 * eye4, 30 * eye4], axis=1)
    )
    chosen_values = node("vadd", [
        vmul(direction_values, vnot(padding)),
        vmul(padding_values, padding)
    ])
    scores = D(
        [chosen_values, direction_valid, danger],
        np.concatenate(
            [eye4, 2000 * eye4, -1000 * eye4], axis=1
        ),
        [-1999.96, -1999.97, -1999.98, -1999.99]
    )

    highest_score = vector_max(scores)
    selection_w = np.concatenate(
        [100 * eye4, -100 * np.ones((4, 1), np.float32)],
        axis=1
    )
    selected = vclip(D(
        [scores, highest_score], selection_w,
        np.ones(4, np.float32)
    ))

    record = vclip(V(
        [red_hit_old, white_hit_old], [1, 1]
    ))
    movement_used = vnot(record)
    non_padding_move = vmul(movement_used, vnot(padding))

    def destination_hit(plate):
        at_destination = cat([
            sample(x) for x in neighbors6(plate)
        ], True)
        return V(
            [vmul(at_destination, selected)], [1, 1, 1, 1]
        )

    # Recording is only triggered by an unrecorded pressure plate.
    hit_red = vmul(
        vmul(destination_hit(unrecorded_red_plate), red_task),
        non_padding_move
    )
    hit_white = vmul(
        vmul(destination_hit(white_plate), white_task),
        non_padding_move
    )
    hit_blue = vmul(
        destination_hit(unvisited_blue_plate), movement_used
    )

    actor_row = V(
        [pool(mul(avatar, coord_row), mean=True)], [4096]
    )
    actor_col = V(
        [pool(mul(avatar, coord_col), mean=True)], [4096]
    )
    actor_xy = cat([actor_row, actor_col], True)
    displacement = dense(selected, np.array([
        [-6 / 63, 6 / 63, 0, 0],
        [0, 0, -6 / 63, 6 / 63],
    ], np.float32))
    next_actor_xy = node("vadd", [actor_xy, displacement])

    red_record = vmul(record, red_hit_old)
    white_record = vmul(record, white_hit_old)
    cycle = vmul(record, vclip(V(
        [ghost_count, total_slots], [1, -1], 2
    )))
    not_cycle = vnot(cycle)

    def push_coordinates(old, xy, update, length):
        pushed = cat([xy, sl(old, 0, length - 2, True)], True)
        result = node("vadd", [
            vmul(pushed, update),
            vmul(old, vnot(update))
        ])
        return vmul(result, not_cycle)

    next_red_xy = push_coordinates(
        old_red_xy, actor_xy, red_record, 8
    )
    next_blue_xy = push_coordinates(
        old_blue_xy, next_actor_xy, hit_blue, 16
    )
    next_white_active = vmul(vclip(V(
        [white_active, white_record], [1, 1]
    )), not_cycle)
    next_padding_count = vmul(
        node("vadd", [
            padding_count, vmul(padding, movement_used)
        ]),
        vnot(record)
    )
    next_padding_parity = vmul(
        vmul(vnot(padding_parity), padding), movement_used
    )

    node("vconcat", [
        one, hit_red, hit_white, next_white_active,
        next_padding_count, next_padding_parity,
        next_red_xy, next_blue_xy,
    ], ident="mem_next")

    output_w = np.zeros((8, 5), np.float32)
    output_b = np.full(8, -10000, np.float32)
    for i in range(4):
        output_w[i + 1, i] = 1
        output_b[i + 1] = 0
    output_w[5, 4] = 10000
    output_b[5] = -5000
    D([scores, record], output_w, output_b, ident="logits")
    conv(
        "x0", np.zeros((1, 16, 1, 1), np.float32),
        ident="click"
    )

    return {
        "graph": graph,
        "memory_size": 30,
        "params": params,
    }
