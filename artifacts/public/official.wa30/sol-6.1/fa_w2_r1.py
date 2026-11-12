import numpy as np


def build():
    graph, params = [], {}
    current = graph
    serial = 0
    kernels, colours, constants = {}, {}, {}

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def parameter(a):
        name = "p" + str(len(params))
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs=(), **fields):
        name = fields.pop("id", None) or fresh()
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(fields)
        current.append(d)
        return name

    def conv(x, w):
        return node("conv", [x], w=w)

    def affine(x, matrix, bias=None, output_id=None):
        fields = {
            "w": parameter(np.asarray(matrix, np.float32)[:, :, None, None])
        }
        if bias is not None:
            fields["b"] = parameter(bias)
        if output_id is not None:
            fields["id"] = output_id
        return node("conv", [x], **fields)

    def dense(x, matrix, bias=None, output_id=None):
        fields = {"w": parameter(matrix)}
        if bias is not None:
            fields["b"] = parameter(bias)
        if output_id is not None:
            fields["id"] = output_id
        return node("dense", [x], **fields)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def maximum(a, b):
        return node("max2", [a, b])

    def minimum(a, b):
        return node("min2", [a, b])

    def relu(x):
        return node("relu", [x])

    def cat(xs):
        return node("concat", xs)

    def channel(x, i):
        return node("slice", [x], start=i, stop=i + 1)

    def sum_channels(x, c):
        return affine(x, np.ones((1, c), np.float32))

    def max_list(xs):
        result = xs[0]
        for x in xs[1:]:
            result = maximum(result, x)
        return result

    one = node("slice", ["coords"], start=2, stop=3)
    vone = node("gmax", [one])

    def constant(value):
        # Shared constants must never acquire loop-local identifiers.
        nonlocal current
        value = float(value)
        if value not in constants:
            saved = current
            current = graph
            constants[value] = dense(
                vone, np.array([[value]], np.float32)
            )
            current = saved
        return constants[value]

    def scale(x, value):
        return mul(x, constant(value))

    def complement(x):
        return sub(one, x)

    def threshold(x, level=0.5, slope=1000.0, c=1):
        return minimum(
            relu(affine(
                x,
                np.eye(c, dtype=np.float32) * slope,
                np.full(c, -level * slope, np.float32),
            )),
            one,
        )

    def vscale(x, value):
        return node("vmul", [x, constant(value)])

    def vscalar_affine(x, factor, bias=0.0):
        return dense(
            x,
            np.array([[factor]], np.float32),
            np.array([bias], np.float32),
        )

    def vclip01(x):
        positive = node("vrelu", [x])
        excess = node("vrelu", [vscalar_affine(x, 1.0, -1.0)])
        return node("vadd", [positive, vscale(excess, -1.0)])

    # Coordinate hats identify all 256 four-pixel grid anchors.
    hw = np.zeros((96, 3), np.float32)
    hb = np.zeros(96, np.float32)
    hs = np.zeros((32, 96), np.float32)
    for axis in range(2):
        for t in range(16):
            base = axis * 48 + 3 * t
            for j, offset in enumerate((-1, 0, 1)):
                hw[base + j, axis] = 63.0
                hb[base + j] = -(4 * t + offset)
            hs[axis * 16 + t, base:base + 3] = (1, -2, 1)

    hats = relu(affine("coords", hw, hb))
    lines = threshold(affine(hats, hs), c=32)

    bw = np.zeros((256, 32), np.float32)
    for r in range(16):
        for c in range(16):
            bw[16 * r + c, r] = 1
            bw[16 * r + c, 16 + c] = 1
    basis = relu(affine(lines, bw, -np.ones(256, np.float32)))
    lattice = sum_channels(basis, 256)

    # Memory stores 300 * barrier bit + 100 * goal bit.
    # Entries 0..3 additionally carry bounded progress statistics.
    def memory_bits(x, offset):
        shifted = node("vadd", [x, constant(-offset)])
        signed = node("vtanh", [shifted])
        return vscale(node("vadd", [signed, vone]), 0.5)

    old_dot_bits = memory_bits("mem", 200.0)
    residual_mem = node(
        "vadd", ["mem", vscale(old_dot_bits, -300.0)]
    )
    old_goal_bits = memory_bits(residual_mem, 50.0)
    old_goal_map = sum_channels(mul(basis, old_goal_bits), 256)
    old_dot_map = sum_channels(mul(basis, old_dot_bits), 256)

    old_aux = node("vslice", [residual_mem], start=0, stop=4)
    old_aux_goals = node("vslice", [old_goal_bits], start=0, stop=4)
    old_aux = node(
        "vadd", [old_aux, vscale(old_aux_goals, -100.0)]
    )

    def sample_kernel(c, dy, dx, k):
        key = (c, dy, dx, k)
        if key not in kernels:
            w = np.zeros((c, c, k, k), np.float32)
            for i in range(c):
                w[i, i, k // 2 + dy, k // 2 + dx] = 1
            kernels[key] = parameter(w)
        return kernels[key]

    def pixel(colour, dy=0, dx=0):
        if colour not in colours:
            colours[colour] = node(
                "slice", ["x0"], start=colour, stop=colour + 1
            )
        x = colours[colour]
        if dy == 0 and dx == 0:
            return x
        k = 2 * max(abs(dy), abs(dx)) + 1
        return conv(x, sample_kernel(1, dy, dx, k))

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def shift4(x, direction, c=1):
        dy, dx = directions[direction]
        w = sample_kernel(c, 2 * dy, 2 * dx, 5)
        return conv(conv(x, w), w)

    def neighbours(x, c=1):
        return [shift4(x, d, c) for d in range(4)]

    def neighbouring_channels(x):
        return cat(neighbours(x))

    def aligned_shift(x, reverse=False):
        opposite = (1, 0, 3, 2)
        return cat([
            shift4(channel(x, d), opposite[d] if reverse else d)
            for d in range(4)
        ])

    def anchor(colour):
        return mul(pixel(colour, 1, 1), lattice)

    player = anchor(14)
    blue = anchor(12)
    purple = anchor(15)
    boxes = anchor(9)
    walls = anchor(5)

    held_box = mul(boxes, pixel(0))
    holding = node("gmax", [held_box])
    not_holding = node("vadd", [vone, vscale(holding, -1)])
    has_purple = node("gmax", [purple])
    no_purple = node("vadd", [vone, vscale(has_purple, -1)])

    face_up = mul(player, pixel(0, 0, 1))
    face_left = mul(player, pixel(0, 1, 0))
    face_right = mul(player, pixel(0, 1, 3))
    face_down = relu(
        sub(player, add(face_up, add(face_left, face_right)))
    )
    face = node(
        "gmax", [cat([face_up, face_down, face_left, face_right])]
    )

    def facing_selection(x):
        return sum_channels(mul(x, face), 4)

    def pool_player(x):
        return node("gmax", [mul(x, player)])

    # Reconstruct goal rectangles from surviving paired edges.
    p00, p20 = pixel(9), pixel(9, 2, 0)
    p03, p23 = pixel(9, 0, 3), pixel(9, 2, 3)
    p30, p33 = pixel(9, 3, 0), pixel(9, 3, 3)
    edges = cat([
        mul(relu(sub(add(p00, p20), one)), lattice),
        mul(relu(sub(add(p03, p23), one)), lattice),
        mul(relu(sub(add(p00, p03), one)), lattice),
        mul(relu(sub(add(p30, p33), one)), lattice),
    ])
    any_edge = minimum(sum_channels(edges, 4), one)
    floor_two = mul(pixel(2, 1, 1), lattice)
    occupants = minimum(
        add(add(player, boxes), add(blue, purple)), one
    )
    goal_candidate = minimum(
        add(add(floor_two, occupants), any_edge), one
    )

    rw = np.zeros((4, 4, 5, 5), np.float32)
    for i, (dy, dx) in enumerate(((0, -2), (0, 2), (-2, 0), (2, 0))):
        rw[i, i, 2 + dy, 2 + dx] = 1
    ray_parameter = parameter(rw)

    ray_state = fresh("s")
    ray_body = []
    saved = current
    current = ray_body
    ray_step = conv(conv(ray_state, ray_parameter), ray_parameter)
    ray_next = mul(maximum(ray_step, edges), goal_candidate)
    current = saved

    ray_loop = fresh("L")
    current.append({
        "id": ray_loop,
        "op": "loop",
        "T": 16,
        "state": [{"id": ray_state, "init": edges}],
        "body": ray_body,
        "next": {ray_state: ray_next},
    })
    rays = ray_loop + "." + ray_state
    observed_goal = maximum(
        any_edge,
        maximum(
            minimum(channel(rays, 0), channel(rays, 1)),
            minimum(channel(rays, 2), channel(rays, 3)),
        ),
    )
    goals = maximum(observed_goal, old_goal_map)
    new_goal_bits = node("gmax", [mul(basis, goals)])

    observed_dots = mul(
        lattice, mul(pixel(2, 1, 1), pixel(1, 0, 1))
    )
    dots = maximum(observed_dots, old_dot_map)
    new_dot_bits = node("gmax", [mul(basis, dots)])

    blue_adjacent = minimum(
        sum_channels(neighbouring_channels(blue), 4), one
    )
    npc_boxes = mul(
        maximum(
            mul(boxes, pixel(5)),
            mul(boxes, blue_adjacent),
        ),
        complement(held_box),
    )
    active_npc_boxes = mul(npc_boxes, complement(goals))

    # Detect friendly-carried crates that have stopped progressing.
    npc_stats = vscale(
        node("gmean", [mul("coords", active_npc_boxes)]), 320
    )
    previous_stats = node("vslice", [old_aux], start=0, stop=3)
    previous_wait = node("vslice", [old_aux], start=3, stop=4)
    stat_pair = node("vconcat", [npc_stats, previous_stats])

    dw = np.zeros((6, 6), np.float32)
    for i in range(3):
        dw[i, i], dw[i, i + 3] = 1, -1
        dw[i + 3, i], dw[i + 3, i + 3] = -1, 1
    difference = node("vrelu", [dense(stat_pair, dw)])
    difference = dense(difference, np.ones((1, 6), np.float32))
    stable = vclip01(vscalar_affine(difference, -512, 1))
    next_wait = node(
        "vmul",
        [
            vscale(
                vclip01(vscalar_affine(previous_wait, 0.05, 0.05)),
                20,
            ),
            stable,
        ],
    )
    rescue_npc = vclip01(vscalar_affine(next_wait, 0.5, -2))

    auxiliary = node("vconcat", [npc_stats, next_wait])
    aw = np.zeros((256, 4), np.float32)
    aw[:4] = np.eye(4, dtype=np.float32)
    encoded_aux = dense(auxiliary, aw)
    encoded_map = node(
        "vadd",
        [vscale(new_goal_bits, 100), vscale(new_dot_bits, 300)],
    )
    node("vadd", [encoded_map, encoded_aux], id="mem_next")

    occupied = minimum(
        add(add(boxes, walls), add(blue, purple)), one
    )
    occupied = relu(sub(occupied, held_box))
    body_free = mul(lattice, complement(occupied))
    carrier_free = mul(body_free, complement(dots))
    pair_valid = mul(neighbouring_channels(body_free), carrier_free)
    goal_shift = neighbouring_channels(goals)
    gamma = constant(0.97)

    def plan(seed, valid, channels, iterations=44):
        nonlocal current
        seed = mul(seed, valid)
        seeds = [channel(seed, i) for i in range(channels)]
        masks = [channel(valid, i) for i in range(channels)]
        states = [fresh("s") for _ in range(channels)]

        body = []
        saved_list = current
        current = body
        updates = {}
        for i, state in enumerate(states):
            propagated = mul(max_list(neighbours(state)), gamma)
            updates[state] = mul(
                maximum(propagated, seeds[i]), masks[i]
            )
        current = saved_list

        loop_id = fresh("L")
        current.append({
            "id": loop_id,
            "op": "loop",
            "T": iterations,
            "state": [
                {"id": s, "init": initial}
                for s, initial in zip(states, seeds)
            ],
            "body": body,
            "next": updates,
        })
        return cat([loop_id + "." + s for s in states])

    direct_values = plan(goal_shift, pair_valid, 4)

    def proposal(values, seed):
        propagated = mul(max_list(neighbours(values, 4)), gamma)
        return mul(maximum(seed, propagated), carrier_free)

    direct_proposal = proposal(direct_values, goal_shift)

    # Restrict hypothetical regrasp positions to reachable compartments.
    blue_walkable = minimum(add(carrier_free, blue), one)
    blue_reach = threshold(
        plan(blue, blue_walkable, 1), level=0.0001, slope=10000
    )
    player_reach = threshold(
        plan(player, carrier_free, 1), level=0.0001, slope=10000
    )

    blue_delivery = mul(direct_proposal, blue_reach)
    blue_crate_values = aligned_shift(blue_delivery, reverse=True)
    blue_crate_value = max_list([
        channel(blue_crate_values, d) for d in range(4)
    ])
    helper_seed = scale(
        neighbouring_channels(blue_crate_value), 0.90
    )
    fixed_seed = maximum(goal_shift, helper_seed)

    # Explicitly predefine the orientation-switch discount.
    constant(0.85)

    # Coupled planner: translate a pair, or release/regrasp in an
    # accessible orientation. Friendly handoffs use direct delivery values.
    seeds = [channel(fixed_seed, d) for d in range(4)]
    masks = [channel(pair_valid, d) for d in range(4)]
    initial = mul(fixed_seed, pair_valid)
    initial_parts = [channel(initial, d) for d in range(4)]
    states = [fresh("s") for _ in range(4)]

    coupled_body = []
    saved = current
    current = coupled_body

    reachable_orientations = cat([
        mul(s, player_reach) for s in states
    ])
    crate_orientations = aligned_shift(
        reachable_orientations, reverse=True
    )
    crate_best = max_list([
        channel(crate_orientations, d) for d in range(4)
    ])
    switch_field = scale(neighbouring_channels(crate_best), 0.85)
    switch_parts = [channel(switch_field, d) for d in range(4)]

    updates = {}
    for d, state in enumerate(states):
        translation = mul(max_list(neighbours(state)), gamma)
        updates[state] = mul(
            maximum(
                maximum(translation, seeds[d]), switch_parts[d]
            ),
            masks[d],
        )
    current = saved

    coupled_loop = fresh("L")
    current.append({
        "id": coupled_loop,
        "op": "loop",
        "T": 44,
        "state": [
            {"id": s, "init": initial_parts[d]}
            for d, s in enumerate(states)
        ],
        "body": coupled_body,
        "next": updates,
    })
    carried_values = cat([
        coupled_loop + "." + s for s in states
    ])

    final_regrasp = aligned_shift(
        mul(carried_values, player_reach), reverse=True
    )
    final_crate_best = max_list([
        channel(final_regrasp, d) for d in range(4)
    ])
    regrasp_seed = scale(
        neighbouring_channels(final_crate_best), 0.85
    )
    transfer_seed = maximum(helper_seed, regrasp_seed)
    pickup_seed = maximum(goal_shift, transfer_seed)
    pickup_proposal = proposal(carried_values, pickup_seed)

    useful_pickup = threshold(
        sub(pickup_proposal, transfer_seed),
        level=0.001,
        slope=1000,
        c=4,
    )
    pickup_values = maximum(
        direct_proposal, mul(pickup_proposal, useful_pickup)
    )

    npc_permission = add(
        complement(npc_boxes), mul(npc_boxes, rescue_npc)
    )
    pickup_boxes = mul(
        mul(mul(boxes, complement(goals)), complement(held_box)),
        npc_permission,
    )
    pickup_targets = mul(
        pickup_values, neighbouring_channels(pickup_boxes)
    )

    # Remove stealing agents before starting deliveries.
    purple_targets = neighbouring_channels(purple)
    targets = add(
        mul(pickup_targets, no_purple),
        mul(purple_targets, has_purple),
    )
    navigation_seed = scale(
        max_list([channel(targets, d) for d in range(4)]), 0.97
    )
    navigation = plan(navigation_seed, carrier_free, 1)
    walking = scale(neighbouring_channels(navigation), 0.97)
    turning = scale(targets, 0.97)
    free_directions = maximum(walking, turning)

    selected_field = facing_selection(carried_values)
    carrying_directions = scale(
        neighbouring_channels(selected_field), 0.97
    )
    direction_scores = pool_player(add(
        mul(free_directions, not_holding),
        mul(carrying_directions, holding),
    ))

    reversal_w = np.zeros((4, 8), np.float32)
    for d, opposite_action in enumerate((2, 1, 4, 3)):
        reversal_w[d, opposite_action] = -0.004
    direction_scores = node(
        "vadd",
        [
            direction_scores,
            node("vmul", [
                dense("aprev", reversal_w), not_holding
            ]),
        ],
    )

    selected_goal = facing_selection(goal_shift)
    selected_direct = facing_selection(direct_values)
    selected_transfer = facing_selection(transfer_seed)
    selected_value = facing_selection(carried_values)

    transfer_better = threshold(
        sub(selected_transfer, selected_direct),
        level=0.003,
        slope=1000,
    )
    transfer_destination = threshold(
        sub(selected_transfer, selected_value),
        level=-0.002,
        slope=1000,
    )
    release_transfer = mul(transfer_better, transfer_destination)
    unreachable = threshold(
        sub(one, scale(selected_value, 10000)),
        level=0,
        slope=10,
    )
    release_condition = max_list([
        selected_goal, release_transfer, unreachable, has_purple
    ])
    release_score = pool_player(scale(release_condition, 2))

    front_objects = facing_selection(
        neighbouring_channels(minimum(add(boxes, purple), one))
    )
    idle_allowed = node(
        "vadd", [vone, vscale(pool_player(front_objects), -1)]
    )
    grab_score = node(
        "vadd",
        [
            pool_player(facing_selection(targets)),
            vscale(idle_allowed, 1e-5),
        ],
    )
    toggle_score = node(
        "vadd",
        [
            node("vmul", [grab_score, not_holding]),
            node("vmul", [release_score, holding]),
        ],
    )

    features = node("vconcat", [direction_scores, toggle_score])
    action_w = np.zeros((8, 5), np.float32)
    for d in range(4):
        action_w[1 + d, d] = 1
    action_w[5, 4] = 1
    action_b = np.array(
        [-10, 2e-6, 1e-6, 1e-6, 1e-6, 0, -10, -10],
        np.float32,
    )
    dense(features, action_w, action_b, output_id="logits")
    affine(one, np.zeros((1, 1), np.float32), output_id="click")

    # Static dependency bookkeeping only; no inference or simulation.
    declared = {"x0", "coords", "mem", "aprev", "avail"}
    used_parameters = set()

    def check_nodes(nodes, available):
        available = set(available)
        evaluations = 0
        count = 0
        for nd in nodes:
            assert nd["id"] not in declared
            declared.add(nd["id"])
            count += 1
            if nd["op"] == "loop":
                local = set(available)
                for state in nd["state"]:
                    assert state["id"] not in declared
                    declared.add(state["id"])
                    if "init" in state:
                        assert state["init"] in available
                    local.add(state["id"])
                body_env, body_evals, body_count = check_nodes(
                    nd["body"], local
                )
                assert set(nd["next"]) == {
                    s["id"] for s in nd["state"]
                }
                for source in nd["next"].values():
                    assert source in body_env
                for state in nd["state"]:
                    available.add(nd["id"] + "." + state["id"])
                evaluations += 1 + nd["T"] * body_evals
                count += body_count
            else:
                assert all(x in available for x in nd["in"])
                available.add(nd["id"])
                evaluations += 1
                for field in ("w", "b"):
                    if field in nd:
                        used_parameters.add(nd[field])
        return available, evaluations, count

    _, evaluations, node_count = check_nodes(graph, declared)
    assert evaluations <= 10000
    assert node_count <= 1000
    assert used_parameters == set(params)
    assert sum(a.size for a in params.values()) <= 150000
    assert all(np.isfinite(a).all() for a in params.values())

    return {
        "graph": graph,
        "memory_size": 256,
        "params": params,
    }
