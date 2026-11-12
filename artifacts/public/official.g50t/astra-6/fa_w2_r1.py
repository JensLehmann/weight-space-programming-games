import numpy as np


def build():
    params = {}
    graph = []
    current = graph
    serial = 0
    pserial = 0
    constants = {}

    K = 8
    END_START = 200
    END_STOP = END_START + 2 * K
    POS_START = END_STOP
    ENEMY_START = POS_START + 2
    M = ENEMY_START + 2

    def parameter(a):
        nonlocal pserial
        name = "p%d" % pserial
        pserial += 1
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def emit(op, ins, name=None, **fields):
        nonlocal serial
        if name is None:
            name = "n%d" % serial
            serial += 1
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        current.append(nd)
        return name

    def conv_ref(x, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return emit("conv", [x], **fields)

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv", [x], name=name, **fields)

    def linear(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        return conv(x, w[:, :, None, None], b, name)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("dense", [x], name=name, **fields)

    def sl(x, a, b):
        return emit("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return emit("vslice", [x], start=a, stop=b)

    def relu(x):
        return emit("relu", [x])

    def vr(x):
        return emit("vrelu", [x])

    def add(a, b):
        return emit("add", [a, b])

    def sub(a, b):
        return emit("sub", [a, b])

    def mul(a, b):
        return emit("mul", [a, b])

    def mx(a, b):
        return emit("max2", [a, b])

    def va(a, b):
        return emit("vadd", [a, b])

    def vm(a, b):
        return emit("vmul", [a, b])

    def cat(*xs):
        return emit("concat", xs)

    def vcat(*xs):
        return emit("vconcat", xs)

    def gm(x):
        return emit("gmax", [x])

    def mean(x):
        return emit("gmean", [x])

    def vc(value):
        key = float(value)
        if key not in constants:
            constants[key] = dense(
                "avail", np.zeros((1, 8), np.float32), [value]
            )
        return constants[key]

    def vneg(x):
        return vm(x, vc(-1))

    def vsub(a, b):
        return va(a, vneg(b))

    def vclip(x, upper=1):
        return vsub(vr(x), vr(va(x, vc(-upper))))

    one = sl("coords", 2, 3)

    def clip(x):
        return emit("min2", [relu(x), one])

    def complement(x):
        return sub(one, x)

    def max_list(xs):
        result = xs[0]
        for x in xs[1:]:
            result = mx(result, x)
        return result

    xy = linear("coords", [[63, 0, 0], [0, 63, 0]])

    # Indicator channels for individual movement-lattice cells.
    w = np.zeros((40, 3), np.float32)
    b = np.zeros(40, np.float32)
    for axis in range(2):
        for i in range(10):
            q = axis * 20 + 2 * i
            center = 4 + 6 * i
            w[q, axis] = 63
            w[q + 1, axis] = -63
            b[q] = -center
            b[q + 1] = center
    axis_parts = relu(linear("coords", w, b))

    w = np.zeros((100, 40), np.float32)
    for r in range(10):
        for c in range(10):
            j = 10 * r + c
            w[j, 2*r:2*r+2] = -1
            w[j, 20+2*c:22+2*c] = -1
    cells = clip(mul(
        relu(linear(axis_parts, w, np.full(100, 0.5))),
        vc(4)
    ))

    lower_screen = clip(linear("coords", [[63, 0, 0]], [-6]))
    grid = mul(
        linear(cells, np.ones((1, 100), np.float32)),
        lower_screen
    )

    def gather(field):
        return gm(mul(cells, field))

    def scatter(vector):
        return linear(
            mul(cells, vector), np.ones((1, 100), np.float32)
        )

    def point_from_xy(vector):
        delta = emit("abs", [sub(xy, vector)])
        point = clip(mul(
            relu(linear(delta, [[-1, -1]], [0.5])),
            vc(4)
        ))
        return mul(point, grid)

    def get_xy(point):
        return dense(
            mean(mul(point, xy)),
            np.eye(2, dtype=np.float32) * 4096
        )

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def piece_detector(color):
        w = np.zeros((1, 16, 5, 5), np.float32)
        w[0, color, :, :] = 1
        w[0, color, 2, 2] = 0
        w[0, 5, 2, 2] = 1
        return mul(clip(conv("x0", w, [-24])), grid)

    player = piece_detector(9)
    ghost = piece_detector(2)
    enemy = piece_detector(14)
    position = get_xy(player)
    enemy_position = get_xy(enemy)

    # The exit's isolated orange center.
    w = np.zeros((1, 16, 3, 3), np.float32)
    w[0, 9, 1, 1] = 1
    for dr, dc in directions:
        w[0, 5, 1+dr, 1+dc] = 1
    goal = mul(clip(conv("x0", w, [-4])), grid)

    # White teleport-pad perimeter.
    w = np.zeros((1, 16, 7, 7), np.float32)
    w[0, 15, 0, :] = 1
    w[0, 15, 6, :] = 1
    w[0, 15, 1:6, 0] = 1
    w[0, 15, 1:6, 6] = 1
    portal = mul(clip(conv("x0", w, [-23])), grid)

    # The center and corners must all be non-background.
    w = np.zeros((1, 16, 5, 5), np.float32)
    for r, c in [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]:
        w[0, 1:, r, c] = 1
    floor = mul(clip(conv("x0", w, [-4])), grid)

    colors = [8, 11, 15]
    color_planes = cat(
        sl("x0", 8, 9), sl("x0", 11, 12), sl("x0", 15, 16)
    )

    # Momentary, toggle, and teleport-trigger switches.
    w = np.zeros((3, 16, 5, 5), np.float32)
    for j, color in enumerate(colors):
        w[j, color, 1:4, 1:4] = 1
        for r, c in [(0, 0), (0, 4), (4, 0), (4, 4)]:
            w[j, 5, r, c] = 1
    switch_seen = mul(clip(conv("x0", w, [-12]*3)), grid)

    # Recover covered switches using a continuing wire.
    w = np.zeros((12, 3, 7, 7), np.float32)
    for j, (dr, dc) in enumerate(directions):
        for c in range(3):
            w[3*j+c, c, 3+3*dr, 3+3*dc] = 1
    ray3 = conv(color_planes, w)

    w = np.zeros((12, 12, 3, 3), np.float32)
    for j, (dr, dc) in enumerate(directions):
        for c in range(3):
            q = 3*j+c
            w[q, q, 1+dr, 1+dc] = 1
    ray4 = conv(ray3, w)
    continuous_rays = mul(ray3, ray4)
    emerging_wire = max_list([
        sl(continuous_rays, 3*j, 3*j+3) for j in range(4)
    ])

    entities = clip(add(add(player, ghost), enemy))
    covered = mul(entities, complement(portal))
    switch_seen = mx(switch_seen, mul(emerging_wire, covered))

    old_switches = vs("mem", 0, 100)
    observed_codes = gather(linear(switch_seen, [[1, 2, 3]]))
    switch_memory = va(
        old_switches, vr(vsub(observed_codes, old_switches))
    )
    switch_codes = scatter(switch_memory)
    type_delta = emit("abs", [
        linear(switch_codes, np.ones((3, 1)), [-1, -2, -3])
    ])
    switch_types = clip(linear(
        type_delta, -np.eye(3, dtype=np.float32), [1, 1, 1]
    ))
    red_switch = sl(switch_types, 0, 1)
    blue_switch = sl(switch_types, 1, 2)
    white_switch = sl(switch_types, 2, 3)
    hold_switch = add(red_switch, white_switch)

    # Gate direction is its projecting colored pixel.
    w = np.zeros((4, 16, 7, 7), np.float32)
    for j, (dr, dc) in enumerate(directions):
        points = [
            (0, 0), (-2, -2), (-2, 2), (2, -2), (2, 2),
            (3*dr, 3*dc)
        ]
        for color in colors:
            for rr, cc in points:
                w[j, color, 3+rr, 3+cc] += 1
    gate_orientation = mul(
        clip(conv("x0", w, [-5]*4)), grid
    )
    gates = clip(linear(gate_orientation, [[1, 1, 1, 1]]))

    # Six-pixel directional shifts.
    w = np.zeros((4, 1, 7, 7), np.float32)
    for j, (dr, dc) in enumerate(directions):
        w[j, 0, 3+3*dr, 3+3*dc] = 1
    shift_a = parameter(w)

    w = np.zeros((4, 4, 7, 7), np.float32)
    for j, (dr, dc) in enumerate(directions):
        w[j, j, 3+3*dr, 3+3*dc] = 1
    shift_b = parameter(w)

    def shift6(x):
        return conv_ref(conv_ref(x, shift_a), shift_b)

    def split4(x):
        return [sl(x, j, j+1) for j in range(4)]

    # Current recording slot and unused slots.
    row_distance = emit("abs", [
        linear("coords", [[63, 0, 0]], [-2])
    ])
    top_strip = clip(linear(row_distance, [[-1]], [2]))
    yellow_ui = mul(sl("x0", 9, 10), top_strip)
    ui_x = mean(mul(
        yellow_ui, linear("coords", [[0, 63, 0]])
    ))
    slot_index = dense(ui_x, [[4096.0 / 32.0]], [-0.5])
    first_slot = vclip(va(vm(slot_index, vc(-2)), vc(1.5)))
    keep_endpoints = vsub(vc(1), first_slot)

    unused_ui = mean(mul(sl("x0", 1, 2), top_strip))
    capacity = vclip(dense(unused_ui, [[8192.0 / 9.0]], [-0.5]))
    previous_record = vs("aprev", 5, 6)
    old_position = vs("mem", POS_START, POS_START + 2)

    # Counts 0..20; 100 denotes a confirmed false floor.
    old_visits = vs("mem", 100, 200)
    old_bad = vclip(va(old_visits, vc(-90)))
    old_counts = vm(old_visits, vsub(vc(1), old_bad))

    pos_delta = vsub(position, old_position)
    abs_pos_delta = va(vr(pos_delta), vr(vneg(pos_delta)))
    traveled = dense(abs_pos_delta, [[1, 1]])
    did_not_move = vclip(vm(
        va(vneg(traveled), vc(0.5)), vc(4)
    ))
    previous_move = dense(
        "aprev", [[0, 1, 1, 1, 1, 0, 0, 0]]
    )
    did_not_move = vm(did_not_move, previous_move)

    displacement = np.zeros((2, 8), np.float32)
    for j, (dr, dc) in enumerate(directions):
        displacement[:, j+1] = [6*dr, 6*dc]
    attempted_xy = va(
        old_position, dense("aprev", displacement)
    )
    attempted_cell = point_from_xy(attempted_xy)
    new_bad_map = mul(
        mul(attempted_cell, did_not_move), complement(gates)
    )
    bad_map = mul(
        clip(add(scatter(old_bad), new_bad_map)),
        complement(player)
    )
    bad_memory = gather(bad_map)
    floor = mul(floor, complement(bad_map))

    counts = va(
        vm(old_counts, vsub(vc(1), previous_record)),
        gather(player)
    )
    counts = vclip(counts, 20)
    visit_memory = va(counts, vm(bad_memory, vc(100)))
    visit_map = scatter(counts)

    floor_neighbors = split4(shift6(floor))
    vertical_floor = mx(floor_neighbors[0], floor_neighbors[1])
    horizontal_floor = mx(floor_neighbors[2], floor_neighbors[3])
    side_floor = cat(
        horizontal_floor, horizontal_floor,
        vertical_floor, vertical_floor
    )
    closed_gates = clip(linear(
        mul(gate_orientation, side_floor), [[1, 1, 1, 1]]
    ))
    actual_free = mul(floor, complement(gates))

    # Store destinations of completed recordings.
    old_endpoints = vs("mem", END_START, END_STOP)
    d = dense(
        slot_index, np.ones((K, 1), np.float32),
        -np.arange(1, K+1, dtype=np.float32)
    )
    abs_d = va(vr(d), vr(vneg(d)))
    write_slot = vclip(vm(
        vr(va(vneg(abs_d), vc(0.5))), vc(4)
    ))
    write_slot = vm(write_slot, previous_record)

    repeat_pairs = np.zeros((2*K, K), np.float32)
    for j in range(K):
        repeat_pairs[2*j:2*j+2, j] = 1
    write_pair = dense(write_slot, repeat_pairs)
    previous_pairs = dense(
        old_position, np.tile(np.eye(2, dtype=np.float32), (K, 1))
    )
    endpoints = vm(
        va(
            vm(old_endpoints, vsub(vc(1), write_pair)),
            vm(previous_pairs, write_pair)
        ),
        keep_endpoints
    )

    xy_pairs = linear(
        xy, np.tile(np.eye(2, dtype=np.float32), (K, 1))
    )
    ep_delta = emit("abs", [sub(xy_pairs, endpoints)])
    w = np.zeros((K, 2*K), np.float32)
    for j in range(K):
        w[j, 2*j:2*j+2] = -1
    endpoint_planes = clip(mul(
        relu(linear(ep_delta, w, np.full(K, 0.5))), vc(4)
    ))
    endpoint_map = mul(
        clip(linear(endpoint_planes, np.ones((1, K)))),
        grid
    )
    held_source = mul(
        clip(add(endpoint_map, ghost)), hold_switch
    )
    held_types = mul(switch_types, held_source)

    # Separate conductive masks for the wire colors.
    wire_input = cat(color_planes, switch_types, portal)
    w = np.zeros((3, 7, 7, 7), np.float32)
    for j in range(3):
        w[j, j, 2:5, 2:5] = 1
        w[j, 3+j, 1:6, 1:6] = 1
    w[2, 6, :, :] = 1
    wire_mask = mul(clip(conv(wire_input, w)), lower_screen)

    # Boolean propagation over held circuits.
    w = np.zeros((3, 3, 3, 3), np.float32)
    for c in range(3):
        w[c, c, 1, 1] = 1
        for dr, dc in directions:
            w[c, c, 1+dr, 1+dc] = 1
    boolean_wire_kernel = parameter(w)

    body = []
    current = body
    held_next = mul(
        clip(conv_ref("held_state", boolean_wire_kernel)),
        wire_mask
    )
    current = graph
    graph.append({
        "id": "held_flood",
        "op": "loop",
        "T": 112,
        "state": [{"id": "held_state", "init": held_types}],
        "body": body,
        "next": {"held_state": held_next}
    })
    held_wires = "held_flood.held_state"
    held_any = clip(linear(held_wires, [[1, 1, 1]]))
    predicted_open = mul(closed_gates, held_any)
    initially_blocked = sub(gates, predicted_open)
    preliminary_floor = mul(floor, complement(initially_blocked))

    # Enemy-accessible floor component.
    w = np.zeros((1, 1, 7, 7), np.float32)
    w[0, 0, 3, 3] = 1
    for dr, dc in directions:
        w[0, 0, 3+3*dr, 3+3*dc] = 1
    reach_kernel = parameter(w)

    body = []
    current = body
    reach_half = conv_ref("enemy_state", reach_kernel)
    reach_full = conv_ref(reach_half, reach_kernel)
    enemy_next = mul(clip(reach_full), preliminary_floor)
    current = graph
    graph.append({
        "id": "enemy_reach",
        "op": "loop",
        "T": 64,
        "state": [{"id": "enemy_state", "init": enemy}],
        "body": body,
        "next": {"enemy_state": enemy_next}
    })

    enemy_switches = mul("enemy_reach.enemy_state", blue_switch)
    blue_wire_mask = sl(wire_mask, 1, 2)

    w = np.zeros((1, 1, 3, 3), np.float32)
    w[0, 0, 1, 1] = 1
    for dr, dc in directions:
        w[0, 0, 1+dr, 1+dc] = 1
    blue_flood_kernel = parameter(w)

    body = []
    current = body
    blue_next = mul(
        clip(conv_ref("blue_state", blue_flood_kernel)),
        blue_wire_mask
    )
    current = graph
    graph.append({
        "id": "future_blue",
        "op": "loop",
        "T": 112,
        "state": [{"id": "blue_state", "init": enemy_switches}],
        "body": body,
        "next": {"blue_state": blue_next}
    })

    future_open = mul(closed_gates, "future_blue.blue_state")
    predicted_open = clip(add(predicted_open, future_open))
    predicted_blocked = sub(gates, predicted_open)
    navigation_floor = mul(floor, complement(predicted_blocked))
    unresolved_gate = mul(
        mul(closed_gates, complement(held_any)),
        complement("future_blue.blue_state")
    )

    eligible_switches = mul(switch_types, complement(held_source))
    enabled_portals = mul(portal, sl(held_wires, 2, 3))
    goal_seed = mul(goal, vc(1000))

    # Max-plus circuit propagation.
    w = np.zeros((12, 3, 3, 3), np.float32)
    for j, (dr, dc) in enumerate(directions):
        for c in range(3):
            w[3*j+c, c, 1+dr, 1+dc] = 1
    wire_shift = parameter(w)

    def wire_max(x):
        shifted = conv_ref(x, wire_shift)
        neighbors = [sl(shifted, 3*j, 3*j+3) for j in range(4)]
        return max_list([x] + neighbors)

    body = []
    current = body

    nav_neighbors = split4(shift6("value_state"))
    best_neighbor = max_list(nav_neighbors)

    wire_neighbor = wire_max("wire_state")
    wire_decay = relu(linear(
        wire_neighbor, np.eye(3, dtype=np.float32), [-0.02]*3
    ))

    gate_signal = mul(best_neighbor, unresolved_gate)
    gate_signal = mul(gate_signal, wire_mask)

    portal_signal = mul("value_state", portal)
    portal_signal = linear(portal_signal, [[0], [0], [1]])
    circuit_seed = mx(gate_signal, portal_signal)
    wire_next = mul(mx(wire_decay, circuit_seed), wire_mask)

    switch_value = relu(linear(
        wire_next, np.eye(3, dtype=np.float32), [-20, -5, -20]
    ))
    switch_value = linear(
        mul(switch_value, eligible_switches), [[1, 1, 1]]
    )

    portal_value = relu(linear(sl(wire_next, 2, 3), [[1]], [-2]))
    portal_value = mul(portal_value, enabled_portals)

    walking_value = relu(linear(best_neighbor, [[1]], [-1]))
    value_next = max_list([
        walking_value, switch_value, portal_value, goal_seed
    ])
    value_next = mul(value_next, navigation_floor)

    current = graph
    graph.append({
        "id": "planner",
        "op": "loop",
        "T": 192,
        "state": [
            {"id": "value_state", "init": goal_seed},
            {"id": "wire_state", "zeros": [3, 64, 64]}
        ],
        "body": body,
        "next": {
            "value_state": value_next,
            "wire_state": wire_next
        }
    })

    # Short-horizon enemy avoidance.
    previous_enemy = vs("mem", ENEMY_START, ENEMY_START + 2)
    predicted_enemy_xy = va(
        vm(enemy_position, vc(2)), vneg(previous_enemy)
    )
    predicted_enemy = point_from_xy(predicted_enemy_xy)
    projection_valid = gm(mul(predicted_enemy, actual_free))
    seen_enemy_before = vclip(dense(
        previous_enemy, [[2, 2]], [-1]
    ))
    projection_valid = vm(projection_valid, seen_enemy_before)

    enemy_neighbors = split4(shift6(enemy))
    conservative_danger = max_list([enemy] + enemy_neighbors)
    danger = add(
        mul(predicted_enemy, projection_valid),
        mul(conservative_danger, vsub(vc(1), projection_valid))
    )
    danger = mul(danger, gm(enemy))

    successor_values = gm(mul(
        shift6("planner.value_state"), player
    ))
    successor_free = gm(mul(shift6(actual_free), player))
    successor_danger = gm(mul(shift6(danger), player))
    successor_visits = gm(mul(shift6(visit_map), player))

    directional = va(
        successor_values,
        dense(successor_free, np.eye(4) * 10000, [-10000]*4)
    )
    directional = va(
        directional, dense(successor_danger, np.eye(4) * -5000)
    )
    directional = va(
        directional, dense(successor_visits, np.eye(4) * -0.05)
    )

    clockwise = {0: 3, 3: 1, 1: 2, 2: 0}
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}
    preference = np.zeros((4, 8), np.float32)
    for old in range(4):
        right = clockwise[old]
        back = opposite[old]
        left = opposite[right]
        for new, score in [
            (right, .04), (old, .03), (left, .02), (back, .01)
        ]:
            preference[new, old+1] = score
    directional = va(
        directional,
        dense("aprev", preference, [.004, .003, .002, .001])
    )

    # Repeated trigger visits improve teleport replay synchronization.
    current_visits = gm(mul(visit_map, player))
    white_ready = vclip(vm(
        va(current_visits, vc(-3.5)), vc(2)
    ))
    on_red = gm(mul(player, red_switch))
    on_white = gm(mul(player, white_switch))
    record_here = va(on_red, vm(on_white, white_ready))

    already_held = gm(mul(player, held_source))
    record = vm(
        vm(record_here, vsub(vc(1), already_held)),
        capacity
    )

    # RESET wins only when aprev is the all-zero episode-start vector.
    action_features = vcat(directional, record, "aprev")
    w = np.zeros((8, 13), np.float32)
    b = np.full(8, -1e6, np.float32)
    b[0] = 100000
    w[0, 5:13] = -200000
    for j in range(4):
        w[j+1, j] = 1
        b[j+1] = 0
    w[5, 4] = 50000
    b[5] = -20000
    dense(action_features, w, b, name="logits")

    conv(one, np.zeros((1, 1, 1, 1), np.float32), name="click")
    emit(
        "vconcat",
        [switch_memory, visit_memory, endpoints, position, enemy_position],
        name="mem_next"
    )

    return {
        "graph": graph,
        "memory_size": M,
        "params": params
    }
