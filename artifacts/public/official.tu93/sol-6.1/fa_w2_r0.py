import numpy as np


def build():
    graph, params = [], {}
    serial = [0, 0]

    def param(a, name=None):
        if name is None:
            name = "p%d" % serial[1]
            serial[1] += 1
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(nodes, op, inputs, name=None, **kw):
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(kw)
        nodes.append(d)
        return name

    def conv(nodes, x, w, b=None, name=None):
        kw = {"w": w}
        if b is not None:
            kw["b"] = b
        return node(nodes, "conv", [x], name=name, **kw)

    def affine(nodes, x, a, b=0, channels=1):
        w = np.eye(channels, dtype=np.float32) * a
        return conv(nodes, x, param(w[:, :, None, None]),
                    param(np.full(channels, b, np.float32)))

    def sl(nodes, x, k):
        return node(nodes, "slice", [x], start=k, stop=k + 1)

    def dense(nodes, x, w, b=None, name=None):
        kw = {"w": param(w)}
        if b is not None:
            kw["b"] = param(b)
        return node(nodes, "dense", [x], name=name, **kw)

    def vmax4(nodes, x):
        q = [sl(nodes, x, k) for k in range(4)]
        a = node(nodes, "max2", [q[0], q[1]])
        b = node(nodes, "max2", [q[2], q[3]])
        return node(nodes, "max2", [a, b])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    read = np.zeros((4, 1, 7, 7), np.float32)
    diag = np.zeros((4, 4, 7, 7), np.float32)
    move = np.zeros_like(diag)
    for k, (dy, dx) in enumerate(directions):
        read[k, 0, 3 + 3 * dy, 3 + 3 * dx] = 1
        diag[k, k, 3 + 3 * dy, 3 + 3 * dx] = 1
        move[k, k, 3 - 3 * dy, 3 - 3 * dx] = 1

    Wread = param(read, "read_three")
    Wdiag = param(diag, "directional_read_three")
    Wmove = param(move, "directional_move_three")
    Wsum = param(np.ones((1, 4, 1, 1)), "sum_four")
    Wrepeat = param(np.ones((4, 1, 1, 1)), "repeat_four")
    Widentity = param(np.eye(4)[:, :, None, None], "identity_four")
    Bstep = param([-1] * 4, "one_move_cost")

    flip = np.zeros((4, 4, 1, 1), np.float32)
    for k in range(4):
        flip[opposite[k], k, 0, 0] = 1
    Wflip = param(flip, "reverse_patrol")

    def forward(nodes, x):
        return conv(nodes, conv(nodes, x, Wmove), Wmove)

    def sum4(nodes, x):
        return conv(nodes, x, Wsum)

    def read_scalar(nodes, x):
        return conv(nodes, conv(nodes, x, Wread), Wdiag)

    def read_directional(nodes, x):
        return conv(nodes, conv(nodes, x, Wdiag), Wdiag)

    # Exact centers of ordinary three-pixel-square objects.
    cw = np.zeros((5, 16, 3, 3), np.float32)
    for k, color in enumerate([9, 14, 8, 12, 13]):
        cw[k, color] = 1
        if color == 9:
            cw[k, 4] = 1
        elif color in (8, 12, 13):
            cw[k, 15] = 1
            cw[k, 11] = 1
    centers = conv(graph, "x0", param(cw), param([-8] * 5))
    centers = node(graph, "relu", [centers])
    player, goal, blue, cyan, purple = [
        sl(graph, centers, k) for k in range(5)
    ]

    mw = np.zeros((4, 16, 3, 3), np.float32)
    aw = np.zeros((1, 16, 3, 3), np.float32)
    for k, (dy, dx) in enumerate(directions):
        mw[k, 15, 1 + dy, 1 + dx] = 1
        mw[k, 11, 1 + dy, 1 + dx] = 1
        aw[0, 11, 1 + dy, 1 + dx] = 1
    markers = conv(graph, "x0", param(mw))
    active_marker = conv(graph, "x0", param(aw))
    purple_active = node(graph, "mul", [purple, active_marker])
    purple_idle = node(graph, "sub", [purple, purple_active])

    ww = np.zeros((1, 16, 1, 1), np.float32)
    for c in (0, 2, 4, 8, 9, 11, 12, 13, 14, 15):
        ww[0, c, 0, 0] = 1
    walk = conv(graph, "x0", param(ww))

    ew = np.zeros((4, 16, 7, 7), np.float32)
    for k, (dy, dx) in enumerate(directions):
        ew[k, 2, 3 + 3 * dy, 3 + 3 * dx] = 1
    edges = conv(graph, "x0", param(ew))

    bd = node(graph, "mul", [blue, markers])
    cd = node(graph, "mul", [cyan, markers])
    pd = node(graph, "mul", [purple_active, markers])
    idle_dirs = node(graph, "mul", [purple_idle, markers])

    blue_front = sum4(graph, forward(graph, bd))
    blue_allow = affine(graph, blue_front, -1, 1)
    blue_allow = node(graph, "relu", [blue_allow])
    blue_cost = affine(graph, blue_front, 10000)
    idle_front = sum4(graph, forward(graph, idle_dirs))
    idle_cost = affine(graph, idle_front, 8)
    fixed_cost = node(graph, "add", [blue_cost, idle_cost])

    # Count active followers and recognize their activation turn.
    active_mean = node(graph, "gmean", [purple_active])
    active_count = dense(
        graph, active_mean, [[4096]], name="mem_next"
    )
    pair = node(graph, "vconcat", [active_count, "mem"])
    difference = dense(graph, pair, [[1, -1]])
    difference = node(graph, "vrelu", [difference])
    excess = dense(graph, difference, [[1]], [-1])
    excess = node(graph, "vrelu", [excess])
    pair = node(graph, "vconcat", [difference, excess])
    activated = dense(graph, pair, [[1, -1]])
    not_activated = dense(graph, activated, [[-1]], [1])

    pursuing = node(graph, "gmax", [purple_active])
    not_pursuing = dense(graph, pursuing, [[-1]], [1])
    previous_dir = node(
        graph, "vslice", ["aprev"], start=1, stop=5
    )

    first_dirs = forward(graph, pd)
    first_purple = sum4(graph, first_dirs)
    second_a = node(graph, "mul", [first_dirs, activated])
    second_b = node(graph, "mul", [first_purple, previous_dir])
    second_b = node(graph, "mul", [second_b, not_activated])
    second_dirs = node(graph, "add", [second_a, second_b])
    second_purple = sum4(graph, forward(graph, second_dirs))

    # Shortest-path potential to the exit on the visible corridor graph.
    base_seed = affine(graph, goal, 128)
    Wscalar = param([[[[1]]]], "scalar_identity")
    Bhalf = param([-0.5], "half_edge_cost")
    body = []
    shifted = conv(body, "base_state", Wread)
    best = vmax4(body, shifted)
    best = conv(body, best, Wscalar, Bhalf)
    best = node(body, "relu", [best])
    best = node(body, "mul", [best, walk])
    best = node(body, "max2", [best, base_seed])
    graph.append({
        "id": "distance",
        "op": "loop",
        "T": 160,
        "state": [{"id": "base_state", "init": base_seed}],
        "body": body,
        "next": {"base_state": best}
    })
    base = "distance.base_state"

    # Removal subgoals approximate the route after the obstacle is gone.
    removal_value = affine(graph, base, 1, 392)
    removable = node(graph, "max2", [blue, purple_idle])
    removal_seed = node(graph, "mul", [removable, removal_value])
    exit_seed = affine(graph, goal, 512)
    terminal = node(graph, "max2", [exit_seed, removal_seed])
    terminal = node(graph, "mul", [terminal, blue_allow])
    terminal_four = conv(graph, terminal, Wrepeat)
    patrol_reward = affine(graph, base, 1, 388)

    def directional_best(nodes, q):
        # Output channel k is value given incoming direction k.
        v = [sl(nodes, q, k) for k in range(4)]
        vertical = node(nodes, "max2", [v[0], v[1]])
        horizontal = node(nodes, "max2", [v[2], v[3]])
        unrestricted = node(nodes, "max2", [vertical, horizontal])
        unrestricted = node(
            nodes, "mul", [unrestricted, not_pursuing]
        )
        restricted = [
            node(nodes, "max2", [v[0], horizontal]),
            node(nodes, "max2", [v[1], horizontal]),
            node(nodes, "max2", [v[2], vertical]),
            node(nodes, "max2", [v[3], vertical])
        ]
        restricted = node(nodes, "concat", restricted)
        return node(nodes, "max2", [restricted, unrestricted])

    def movement_update(nodes, destination_values):
        q = read_directional(nodes, destination_values)
        q = conv(nodes, q, Widentity, Bstep)
        q = node(nodes, "relu", [q])
        q = node(nodes, "mul", [q, edges])
        best = directional_best(nodes, q)
        best = node(nodes, "max2", [best, terminal_four])
        best = node(nodes, "sub", [best, idle_cost])
        best = node(nodes, "relu", [best])
        return node(nodes, "mul", [best, blue_allow])

    body = []
    best = movement_update(body, "route_state")
    graph.append({
        "id": "routes",
        "op": "loop",
        "T": 64,
        "state": [{"id": "route_state", "init": terminal_four}],
        "body": body,
        "next": {"route_state": best}
    })

    # Forecast both pre-move and post-move patrol positions.
    horizon = 16
    push = np.zeros((horizon, horizon, 1, 1), np.float32)
    drop = np.zeros_like(push)
    insert = np.zeros((horizon, 1, 1, 1), np.float32)
    insert[0, 0, 0, 0] = 1
    for k in range(horizon - 1):
        push[k + 1, k, 0, 0] = 1
        drop[k, k + 1, 0, 0] = 1
    Wpush = param(push, "push_time")
    Wdrop = param(drop, "drop_time")
    Winsert = param(insert, "insert_time")

    body = []
    before = sum4(body, "forecast_dirs")
    moved = forward(body, "forecast_dirs")
    after = sum4(body, moved)
    keep = node(body, "mul", [moved, edges])
    bounce = node(body, "sub", [moved, keep])
    bounce = conv(body, bounce, Wflip)
    new_dirs = node(body, "add", [keep, bounce])

    pre_old = conv(body, "pre_stack", Wpush)
    pre_new = conv(body, before, Winsert)
    pre_next = node(body, "add", [pre_old, pre_new])
    post_old = conv(body, "post_stack", Wpush)
    post_new = conv(body, after, Winsert)
    post_next = node(body, "add", [post_old, post_new])
    graph.append({
        "id": "forecast",
        "op": "loop",
        "T": horizon,
        "state": [
            {"id": "forecast_dirs", "init": cd},
            {"id": "pre_stack", "zeros": [horizon, 64, 64]},
            {"id": "post_stack", "zeros": [horizon, 64, 64]}
        ],
        "body": body,
        "next": {
            "forecast_dirs": new_dirs,
            "pre_stack": pre_next,
            "post_stack": post_next
        }
    })

    hazards = affine(
        graph, "forecast.post_stack", 10000, channels=horizon
    )
    first_w = np.zeros((horizon, 1, 1, 1), np.float32)
    second_w = np.zeros_like(first_w)
    first_w[horizon - 1, 0, 0, 0] = 10000
    second_w[horizon - 2, 0, 0, 0] = 10000
    h1 = conv(graph, first_purple, param(first_w))
    h2 = conv(graph, second_purple, param(second_w))
    hazards = node(graph, "add", [hazards, h1])
    hazards = node(graph, "add", [hazards, h2])

    # Backward planning with a terminal reward for removing a patrol
    # before it gets its move.
    def destination_values(nodes, values, post_head, pre_head):
        penalty = node(nodes, "add", [post_head, fixed_cost])
        safe = node(nodes, "sub", [values, penalty])
        safe = node(nodes, "relu", [safe])
        intercept = node(nodes, "mul", [pre_head, patrol_reward])
        safe = node(nodes, "max2", [safe, intercept])
        return node(nodes, "mul", [safe, blue_allow])

    body = []
    post_head = sl(body, "time_hazards", 0)
    pre_head = sl(body, "time_patrols", 0)
    safe = destination_values(
        body, "time_values", post_head, pre_head
    )
    next_values = movement_update(body, safe)
    next_hazards = conv(body, "time_hazards", Wdrop)
    next_patrols = conv(body, "time_patrols", Wdrop)
    graph.append({
        "id": "planning",
        "op": "loop",
        "T": horizon - 1,
        "state": [
            {"id": "time_values", "init": "routes.route_state"},
            {"id": "time_hazards", "init": hazards},
            {"id": "time_patrols", "init": "forecast.pre_stack"}
        ],
        "body": body,
        "next": {
            "time_values": next_values,
            "time_hazards": next_hazards,
            "time_patrols": next_patrols
        }
    })

    post_head = sl(graph, "planning.time_hazards", 0)
    pre_head = sl(graph, "planning.time_patrols", 0)
    safe = destination_values(
        graph, "planning.time_values", post_head, pre_head
    )
    scores = read_directional(graph, safe)

    legal_w = np.eye(4, dtype=np.float32)[:, :, None, None] * 100000
    legality = conv(
        graph, edges, param(legal_w), param([-100000] * 4)
    )
    scores = node(graph, "add", [scores, legality])

    # Activation permits the escape reversal; established pursuit
    # forbids immediately retracing the previous edge.
    established = node(graph, "vmul", [pursuing, not_activated])
    reverse_w = np.zeros((4, 8), np.float32)
    for k in range(4):
        reverse_w[k, opposite[k] + 1] = -10000
    reversal = dense(graph, "aprev", reverse_w)
    reversal = node(graph, "vmul", [reversal, established])
    scores = node(graph, "add", [scores, reversal])

    tie_w = np.zeros((4, 8), np.float32)
    for k in range(4):
        tie_w[k, opposite[k] + 1] = -0.05
    tie_cost = dense(graph, "aprev", tie_w)
    scores = node(graph, "add", [scores, tie_cost])
    player_mask = affine(graph, player, 200000, -200000)
    scores = node(graph, "add", [scores, player_mask])
    action_values = node(graph, "gmax", [scores])

    out_w = np.zeros((8, 4), np.float32)
    out_b = np.full(8, -1000000, np.float32)
    for k in range(4):
        out_w[k + 1, k] = 1
        out_b[k + 1] = 0
    dense(graph, action_values, out_w, out_b, name="logits")
    conv(
        graph, player,
        param(np.zeros((1, 1, 1, 1), np.float32)),
        name="click"
    )

    return {"graph": graph, "memory_size": 1, "params": params}
