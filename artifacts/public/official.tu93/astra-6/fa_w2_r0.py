import numpy as np


def build():
    params = {}
    graph = []
    target = graph
    serial = 0
    cache = {}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]
    memory_size = 154

    GOAL_VALUE = 4096.0
    CAPTURE_VALUE = 2048.0
    LETHAL_COST = 8192.0

    def put(a, key=None):
        if key is not None and key in cache:
            return cache[key]
        name = "p_" + str(len(params))
        params[name] = np.asarray(a, dtype=np.float32)
        if key is not None:
            cache[key] = name
        return name

    def node(kind, ins, name=None, **fields):
        nonlocal serial
        if name is None:
            name = "n_" + str(serial)
            serial += 1
        d = {"id": name, "op": kind, "in": list(ins)}
        d.update(fields)
        target.append(d)
        return name

    def conv(x, w, b=None):
        fields = {"w": w if isinstance(w, str) else put(w)}
        if b is not None:
            fields["b"] = b if isinstance(b, str) else put(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": w if isinstance(w, str) else put(w)}
        if b is not None:
            fields["b"] = b if isinstance(b, str) else put(b)
        return node("dense", [x], name=name, **fields)

    def sl(x, lo, hi):
        return node("slice", [x], start=lo, stop=hi)

    def vsl(x, lo, hi):
        return node("vslice", [x], start=lo, stop=hi)

    def const(v):
        key = ("constant_node", float(v))
        if key not in cache:
            cache[key] = dense(
                "avail", np.zeros((1, 8), np.float32), [v]
            )
        return cache[key]

    def clamp_vector(x):
        z = dense(x, [[1.0], [1.0]], [0.0, -1.0])
        z = node("vrelu", [z])
        return dense(z, [[1.0, -1.0]])

    def diag_kernel(sign):
        key = ("diagonal_shift", sign)
        if key not in cache:
            w = np.zeros((4, 4, 7, 7), np.float32)
            for a, (dy, dx) in enumerate(dirs):
                w[a, a, 3 + sign * 3 * dy,
                  3 + sign * 3 * dx] = 1
            put(w, key)
        return cache[key]

    def scalar_kernel(sign):
        key = ("scalar_shift", sign)
        if key not in cache:
            w = np.zeros((4, 1, 7, 7), np.float32)
            for a, (dy, dx) in enumerate(dirs):
                w[a, 0, 3 + sign * 3 * dy,
                  3 + sign * 3 * dx] = 1
            put(w, key)
        return cache[key]

    def shift4(x, sign, final_bias=None):
        k = diag_kernel(sign)
        return conv(conv(x, k), k, final_bias)

    def shift1(x, sign):
        return conv(conv(x, scalar_kernel(sign)), diag_kernel(sign))

    def sum_channels(x, count, weights=None):
        if weights is None:
            weights = np.ones(count, np.float32)
        w = np.asarray(weights, np.float32).reshape(1, count, 1, 1)
        return conv(x, w)

    def at_player(x, player):
        z = node("mul", [x, player])
        z = node("gmean", [z])
        return node("vmul", [z, const(4096.0)])

    one = const(1.0)
    scale4096 = const(4096.0)
    penalty = const(LETHAL_COST)

    # Four directional channels per enemy type, then player,
    # exit, and general lattice-node centres.
    detector = np.zeros((15, 16, 3, 3), np.float32)
    for base, colour, markers in [
        (0, 8, (15, 11)),
        (4, 12, (15, 11)),
        (8, 13, (11,)),
    ]:
        for a, (dy, dx) in enumerate(dirs):
            detector[base + a, colour, :, :] = 1
            for marker in markers:
                detector[base + a, marker, 1 + dy, 1 + dx] = 1

    detector[12, 9, :, :] = 1
    detector[12, 4, :, :] = 1
    detector[13, 14, :, :] = 1
    for colour in (0, 4, 8, 9, 11, 12, 13, 14, 15):
        detector[14, colour, :, :] = 1

    detected = node(
        "relu", [conv("x0", detector, np.full(15, -8, np.float32))]
    )
    red = sl(detected, 0, 4)
    blue_visible = sl(detected, 4, 8)
    purple = sl(detected, 8, 12)
    player = sl(detected, 12, 13)
    goal = sl(detected, 13, 14)
    centres = sl(detected, 14, 15)

    road = sl("x0", 2, 3)
    edge_w = np.zeros((4, 1, 7, 7), np.float32)
    for a, (dy, dx) in enumerate(dirs):
        edge_w[a, 0, 3 + 3 * dy, 3 + 3 * dx] = 1
    edges = conv(road, edge_w)

    perm = np.zeros((4, 4, 1, 1), np.float32)
    for a in range(4):
        perm[a, opposite[a], 0, 0] = 1
    perm_name = put(perm)
    behind_edges = conv(edges, perm_name)

    def blue_forward(x):
        moved = shift4(x, -1)
        keep = node("mul", [moved, edges])
        turn = node("sub", [moved, keep])
        turn = conv(turn, perm_name)
        return node("add", [keep, turn])

    def blue_reverse(x):
        keep = node("mul", [x, behind_edges])
        turn = node("sub", [x, keep])
        turn = conv(turn, perm_name)
        before_move = node("add", [keep, turn])
        return shift4(before_move, 1)

    xy = sl("coords", 0, 2)
    player_xy = node("gmax", [node("mul", [xy, player])])

    # Spatial bins encode patrol observations in recurrent memory.
    # They are screen-coordinate bins, not stored level layouts.
    bin_w = np.zeros((242, 3, 1, 1), np.float32)
    bin_b = np.zeros(242, np.float32)
    for i in range(11):
        for j in range(11):
            k = i * 11 + j
            bin_w[k, 0, 0, 0] = 63
            bin_b[k] = -(6 * i + 2.5)
            bin_w[121 + k, 1, 0, 0] = 63
            bin_b[121 + k] = -(6 * j + 2.5)

    bin_delta = node("abs", [conv("coords", bin_w, bin_b)])
    by = sl(bin_delta, 0, 121)
    bx = sl(bin_delta, 121, 242)
    bd = node("max2", [by, bx])
    bd = node("mul", [bd, const(-4.0)])
    bd = node("add", [bd, const(12.0)])
    bins = node("min2", [node("relu", [bd]), one])

    old_codes = vsl("mem", 0, 121)
    eye = put(np.eye(121, dtype=np.float32))
    neg_two_eye = put(-2 * np.eye(121, dtype=np.float32))
    ramps = []
    for k in range(6):
        ramps.append(node(
            "vrelu",
            [dense(old_codes, eye, np.full(121, -k, np.float32))]
        ))

    remembered_channels = []
    for d in range(1, 5):
        middle = dense(ramps[d], neg_two_eye)
        indicator = node("vadd", [ramps[d - 1], middle])
        indicator = node("vadd", [indicator, ramps[d + 1]])
        plane = node("mul", [bins, indicator])
        remembered_channels.append(sum_channels(plane, 121))

    remembered = node("concat", remembered_channels)
    remembered = node("mul", [remembered, centres])

    # Moving onto an enemy captures it before enemy motion.
    not_player = node("sub", [one, player])
    remembered = node("mul", [remembered, not_player])

    old_xy = vsl("mem", 122, 124)
    negative_xy = dense(old_xy, -np.eye(2, dtype=np.float32))
    delta_xy = node("vadd", [player_xy, negative_xy])
    positive_delta = node("vrelu", [delta_xy])
    negative_delta = node(
        "vrelu", [dense(delta_xy, -np.eye(2, dtype=np.float32))]
    )
    distance_moved = node("vadd", [positive_delta, negative_delta])
    motion = dense(distance_moved, np.full((1, 2), 20, np.float32))
    motion = clamp_vector(motion)
    not_reset_w = np.zeros((1, 8), np.float32)
    not_reset_w[0, 0] = -1
    not_reset = dense("aprev", not_reset_w, [1.0])
    motion = node("vmul", [motion, not_reset])

    predicted = blue_forward(remembered)
    predicted_delta = node("sub", [predicted, remembered])
    predicted_delta = node("mul", [predicted_delta, motion])
    predicted = node("add", [remembered, predicted_delta])

    hidden_w = np.zeros((1, 16, 1, 1), np.float32)
    hidden_w[0, 8, 0, 0] = 1
    hidden_w[0, 13, 0, 0] = 1
    hidden_by_enemy = conv("x0", hidden_w)
    hidden_prediction = node("mul", [predicted, hidden_by_enemy])
    blue = node("add", [blue_visible, hidden_prediction])

    encoded_blue = sum_channels(blue, 4, [1, 2, 3, 4])
    encoded_blue = node("mul", [encoded_blue, bins])
    encoded_blue = node("gmean", [encoded_blue])
    encoded_blue = node("vmul", [encoded_blue, scale4096])

    # Red attack destinations. Leaving an attacker's own square
    # is safe from that attacker, because it has just been captured.
    red_threat = sum_channels(shift4(red, -1), 4)
    incoming_red = shift1(red_threat, 1)
    red_net = node("relu", [node("sub", [incoming_red, red])])
    red_cost = node("mul", [red_net, penalty])

    # Purple next-step prediction and lag-two movement constraints.
    purple_next_channels = shift4(purple, -1)
    purple_next = sum_channels(purple_next_channels, 4)
    purple_second_fresh = sum_channels(
        shift4(purple_next_channels, -1), 4
    )

    previous_direction = vsl("aprev", 1, 5)
    purple_second_choices = shift1(purple_next, -1)
    purple_second_choices = node(
        "mul", [purple_second_choices, previous_direction]
    )
    purple_second_old = sum_channels(purple_second_choices, 4)

    purple_count = node("gmean", [purple])
    purple_count = dense(
        purple_count, np.full((1, 4), 4096, np.float32)
    )
    old_purple_count = vsl("mem", 121, 122)
    negative_count = dense(old_purple_count, [[-1.0]])
    count_change = node("vadd", [purple_count, negative_count])
    newly_active = clamp_vector(count_change)

    second_difference = node(
        "sub", [purple_second_fresh, purple_second_old]
    )
    second_difference = node("mul", [second_difference, newly_active])
    purple_second = node(
        "add", [purple_second_old, second_difference]
    )

    first_displacement = shift1(purple_second, 1)
    pair_w1 = np.zeros((16, 4, 7, 7), np.float32)
    pair_w2 = np.zeros((16, 16, 7, 7), np.float32)
    for p in range(4):
        for a, (dy, dx) in enumerate(dirs):
            k = p * 4 + a
            pair_w1[k, p, 3 + 3 * dy, 3 + 3 * dx] = 1
            pair_w2[k, k, 3 + 3 * dy, 3 + 3 * dx] = 1
    pair_map = conv(conv(first_displacement, pair_w1), pair_w2)
    pair_hits = at_player(pair_map, player)

    pair_costs = []
    for a in range(4):
        w = np.zeros((4, 16), np.float32)
        for p in range(4):
            w[p, p * 4 + a] = LETHAL_COST
        pair_costs.append(dense(pair_hits, w))

    # Roll patrols forward, then propagate route values backward.
    blue_first = blue_forward(blue)
    forward_body = []
    target = forward_body
    forward_next = blue_forward("bf_state")
    target = graph
    graph.append({
        "id": "patrol",
        "op": "loop",
        "T": 63,
        "state": [{"id": "bf_state", "init": blue_first}],
        "body": forward_body,
        "next": {"bf_state": forward_next},
    })

    # Capturable red guards are intermediate objectives.
    # A safe complete route always outranks a capture-only route.
    red_targets = sum_channels(
        red, 4, np.full(4, CAPTURE_VALUE, np.float32)
    )
    exit_targets = conv(
        goal, np.full((1, 1, 1, 1), GOAL_VALUE, np.float32)
    )
    terminal_targets = node("max2", [exit_targets, red_targets])
    goal_values = conv(
        terminal_targets, np.ones((4, 1, 1, 1), np.float32)
    )

    backward_body = []
    target = backward_body
    blue_before = blue_reverse("blue_state")

    # Capture targets use patrol positions BEFORE this transition.
    # Lethal patrol destinations use positions AFTER the transition.
    blue_capture_values = sum_channels(
        blue_before, 4, np.full(4, CAPTURE_VALUE, np.float32)
    )
    continuation = node(
        "max2", ["value_state", blue_capture_values]
    )
    patrol_cost = sum_channels(
        "blue_state", 4, np.full(4, LETHAL_COST, np.float32)
    )
    safe_value = node("sub", [continuation, patrol_cost])
    q = shift4(safe_value, 1, np.full(4, -1, np.float32))
    q = node("sub", [q, red_cost])
    q = node("relu", [q])
    q = node("mul", [q, edges])

    alternatives = []
    for a in range(4):
        qa = sl(q, a, a + 1)
        alternatives.append(node("sub", [qa, pair_costs[a]]))
    value = node("max2", [alternatives[0], alternatives[1]])
    value = node("max2", [value, alternatives[2]])
    value = node("max2", [value, alternatives[3]])
    value = node("max2", [value, goal_values])
    target = graph
    graph.append({
        "id": "plan",
        "op": "loop",
        "T": 64,
        "state": [
            {"id": "value_state", "init": goal_values},
            {"id": "blue_state", "init": "patrol.bf_state"},
            {"id": "q_state", "zeros": [4, 64, 64]},
        ],
        "body": backward_body,
        "next": {
            "value_state": value,
            "blue_state": blue_before,
            "q_state": q,
        },
    })

    action_values = at_player("plan.q_state", player)

    # Independent current-turn safety filter.
    moving_threat = node(
        "add", [sum_channels(blue_first, 4), purple_next]
    )
    incoming_moving = shift1(moving_threat, 1)
    immediate_threat = node("add", [incoming_red, incoming_moving])
    safe_actions = node(
        "relu", [node("sub", [edges, immediate_threat])]
    )
    safe_actions = at_player(safe_actions, player)

    # Small tie breakers: exit proximity and recent visitation.
    goal_xy = node("gmax", [node("mul", [xy, goal])])
    goal_delta = node("abs", [node("sub", [xy, goal_xy])])
    proximity = sum_channels(goal_delta, 2, [-1, -1])
    proximity = at_player(shift1(proximity, 1), player)

    history = vsl("mem", 122, 154)
    history_w = np.zeros((32, 3, 1, 1), np.float32)
    for k in range(16):
        history_w[2 * k, 0, 0, 0] = 1
        history_w[2 * k + 1, 1, 0, 0] = 1
    history_coordinates = conv("coords", history_w)
    history_delta = node(
        "abs", [node("sub", [history_coordinates, history])]
    )
    visit_w = np.zeros((16, 32, 1, 1), np.float32)
    for k in range(16):
        visit_w[k, 2 * k, 0, 0] = -21
        visit_w[k, 2 * k + 1, 0, 0] = -21
    visits = node(
        "relu", [conv(history_delta, visit_w, np.ones(16, np.float32))]
    )
    visits = sum_channels(
        visits, 16, [0.85 ** k for k in range(16)]
    )
    visits = at_player(shift1(visits, 1), player)

    features = node(
        "vconcat",
        [action_values, safe_actions, proximity, visits, "aprev"]
    )
    output_w = np.zeros((8, 24), np.float32)
    output_b = np.full(8, -30000, np.float32)
    output_b[0] = -2000
    for a in range(4):
        out = a + 1
        output_w[out, a] = 1
        output_w[out, 4 + a] = 10000
        output_w[out, 8 + a] = 0.1
        output_w[out, 12 + a] = -0.04
        output_w[out, 16 + out] = 0.002
        output_b[out] = -10000
    dense(features, output_w, output_b, name="logits")

    retained_history = vsl("mem", 122, 152)
    node(
        "vconcat",
        [encoded_blue, purple_count, player_xy, retained_history],
        name="mem_next"
    )
    node("slice", ["x0"], name="click", start=0, stop=1)

    return {
        "graph": graph,
        "memory_size": memory_size,
        "params": params,
    }
