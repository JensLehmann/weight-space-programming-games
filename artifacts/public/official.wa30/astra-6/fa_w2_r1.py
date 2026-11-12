import numpy as np


def build():
    params = {}
    graph = []
    serial = 0

    def uid(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def put(a):
        name = uid("p")
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def emit(op, ins, nid=None, **kw):
        name = nid or uid()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(kw)
        graph.append(nd)
        return name

    def cv(x, w, b=None, nid=None):
        fields = {"w": put(w)}
        if b is not None:
            fields["b"] = put(b)
        return emit("conv", [x], nid=nid, **fields)

    def affine(x, w, b=None, nid=None):
        w = np.asarray(w, np.float32)
        return cv(x, w[:, :, None, None], b, nid)

    def dense(x, w, b=None, nid=None):
        fields = {"w": put(w)}
        if b is not None:
            fields["b"] = put(b)
        return emit("dense", [x], nid=nid, **fields)

    def constant(values):
        a = np.asarray(values, np.float32).reshape(-1)
        return dense("aprev", np.zeros((len(a), 8), np.float32), a)

    def sl(x, a, b):
        return emit("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return emit("vslice", [x], start=a, stop=b)

    def cat(xs):
        return xs[0] if len(xs) == 1 else emit("concat", xs)

    def vcat(xs):
        return xs[0] if len(xs) == 1 else emit("vconcat", xs)

    def add(a, b):
        return emit("add", [a, b])

    def sub(a, b):
        return emit("sub", [a, b])

    def mul(a, b):
        return emit("mul", [a, b])

    def mx(a, b):
        return emit("max2", [a, b])

    def relu(x):
        return emit("relu", [x])

    def pool(x):
        return emit("gmax", [x])

    def scale(x, a, b=0.0, c=1):
        return affine(
            x,
            np.eye(c, dtype=np.float32) * a,
            np.full(c, b, np.float32),
        )

    def mix(xs, weights, bias=0.0):
        return affine(
            cat(xs),
            np.asarray(weights, np.float32).reshape(1, -1),
            [bias],
        )

    one = constant([1.0])
    negone = constant([-1.0])
    four = constant([4.0])
    increment = constant([1.0 / 16.0])
    ones = sl("coords", 2, 3)
    zero = affine(ones, [[0.0]])

    def sat(x):
        return emit("min2", [relu(x), one])

    def complement(x, c=1):
        return scale(x, -1.0, 1.0, c)

    def max_channels(x, c):
        result = sl(x, 0, 1)
        for i in range(1, c):
            result = mx(result, sl(x, i, i + 1))
        return result

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    def shift4(x, c=1):
        outputs = []
        for dy, dx in directions:
            w = np.zeros((c, c, 5, 5), np.float32)
            for j in range(c):
                w[j, j, 2 + 2 * dy, 2 + 2 * dx] = 1.0
            pn = put(w)
            half = emit("conv", [x], w=pn)
            outputs.append(emit("conv", [half], w=pn))
        return outputs

    def oriented_shift(x, permutation=None):
        if permutation is None:
            permutation = [0, 1, 2, 3]
        w1 = np.zeros((4, 4, 5, 5), np.float32)
        w2 = np.zeros_like(w1)
        for d, (dy, dx) in enumerate(directions):
            w1[d, permutation[d], 2 + 2 * dy, 2 + 2 * dx] = 1.0
            w2[d, d, 2 + 2 * dy, 2 + 2 * dx] = 1.0
        return cv(cv(x, w1), w2)

    def fill_tiles(points, c=1):
        w = np.zeros((c, c, 7, 7), np.float32)
        for ch in range(c):
            for dy in (-2, -1, 0, 1):
                for dx in (-2, -1, 0, 1):
                    w[ch, ch, 3 + dy, 3 + dx] = 1.0
        return cv(points, w)

    near_kernel = np.zeros((1, 1, 3, 3), np.float32)
    for dy, dx in directions:
        near_kernel[0, 0, 1 + dy, 1 + dx] = 1.0

    def reach(seed, allowed, iterations):
        nonlocal graph
        lid, sid = uid("reach"), uid("reachable")
        outer = graph
        graph = []
        w = near_kernel.copy()
        w[0, 0, 1, 1] = 1.0
        spread = cv(sid, w)
        spread = emit("min2", [spread, one])
        nxt = emit("min2", [spread, allowed])
        body = graph
        graph = outer
        graph.append({
            "id": lid,
            "op": "loop",
            "T": iterations,
            "state": [{"id": sid, "init": seed}],
            "body": body,
            "next": {sid: nxt},
        })
        return lid + "." + sid

    def value_field(seed, allowed, channels, iterations):
        nonlocal graph
        lid, sid = uid("value_loop"), uid("value_state")
        outer = graph
        graph = []

        w = np.zeros((4 * channels, channels, 3, 3), np.float32)
        for direction, (dy, dx) in enumerate(directions):
            for ch in range(channels):
                w[direction * channels + ch, ch, 1 + dy, 1 + dx] = 1.0
        neighbors = cv(sid, w)
        parts = [
            sl(neighbors, d * channels, (d + 1) * channels)
            for d in range(4)
        ]
        best = parts[0]
        for p in parts[1:]:
            best = mx(best, p)
        decayed = relu(sub(best, one))
        nxt = mx(seed, mul(decayed, allowed))

        body = graph
        graph = outer
        graph.append({
            "id": lid,
            "op": "loop",
            "T": iterations,
            "state": [{"id": sid, "init": seed}],
            "body": body,
            "next": {sid: nxt},
        })
        return lid + "." + sid

    # Four-pixel cell basis and interior sample points.
    def coordinate_indicators(center, intercept):
        w = np.zeros((32, 3), np.float32)
        b = np.zeros(32, np.float32)
        for i in range(16):
            w[i, 0] = 63.0
            w[16 + i, 1] = 63.0
            b[i] = -(4 * i + center)
            b[16 + i] = -(4 * i + center)
        a = emit("abs", [affine("coords", w, b)])
        return sat(affine(
            a,
            -10.0 * np.eye(32, dtype=np.float32),
            np.full(32, intercept, np.float32),
        ))

    cell_axes = coordinate_indicators(1.5, 20.0)
    w = np.zeros((256, 32), np.float32)
    for r in range(16):
        for c in range(16):
            w[16 * r + c, r] = 1.0
            w[16 * r + c, 16 + c] = 1.0
    tile_basis = relu(affine(
        cell_axes, w, -np.ones(256, np.float32)
    ))

    point_axes = coordinate_indicators(1.0, 2.5)
    anchors = relu(affine(
        point_axes, np.ones((1, 32), np.float32), [-1.0]
    ))

    # Static memory: goal=10, fence=3, with auxiliary fractions.
    memory_map = affine(
        mul(tile_basis, "mem"), np.ones((1, 256), np.float32)
    )
    old_goal = sat(scale(memory_map, 1.0, -7.0))
    old_static = sat(scale(memory_map, 2.0, -3.0))
    old_fence = relu(sub(old_static, old_goal))
    old_code = add(scale(old_goal, 10.0), scale(old_fence, 3.0))
    old_code_three = pool(mul(sl(tile_basis, 0, 3), old_code))
    aux = emit("vadd", [
        vs("mem", 0, 3),
        emit("vmul", [old_code_three, negone]),
    ])

    # Sprite interior and border detectors.
    colors = [9, 14, 12, 15, 5, 2]
    w = np.zeros((6, 16), np.float32)
    for i, color in enumerate(colors):
        w[i, color] = 1.0
    raw = affine("x0", w)
    raw_parts = [sl(raw, i, i + 1) for i in range(6)]

    black = sl("x0", 0, 1)
    background = sl("x0", 1, 2)
    stencil_input = cat([black, raw_parts[4], background])
    w = np.zeros((3, 3, 3, 3), np.float32)
    w[0, 0, 0, 0] = 1.0
    w[1, 1, 0, 0] = 1.0
    w[2, 2, 0, 1] = 1.0
    stencils = cv(stencil_input, w)
    black_corner = sl(stencils, 0, 1)
    gray_corner = sl(stencils, 1, 2)
    fence_hole = sl(stencils, 2, 3)

    held_raw = relu(mix([raw_parts[0], black_corner], [1, 1], -1))
    other_raw = relu(mix([raw_parts[0], gray_corner], [1, 1], -1))
    fence_raw = relu(mix([raw_parts[5], fence_hole], [1, 1], -1))

    object_points = mul(cat(
        raw_parts[:5] + [held_raw, other_raw, fence_raw]
    ), anchors)
    objects = fill_tiles(object_points, 8)
    crate, player, helper, hostile, wall, held, other, fence_seen = [
        sl(objects, i, i + 1) for i in range(8)
    ]
    player_point = sl(object_points, 1, 2)
    holding = pool(held)
    has_helper = pool(helper)
    has_hostile = pool(hostile)

    # Facing direction, including bottom-row UI occlusion handling.
    w = np.zeros((4, 1, 5, 5), np.float32)
    for i, (dy, dx) in enumerate([(-1, 0), (2, 0), (0, -1), (0, 2)]):
        w[i, 0, 2 + dy, 2 + dx] = 1.0
    face_observed = pool(mul(cv(black, w), player_point))
    w = np.eye(4, dtype=np.float32)
    w[1] = [-1, 0, -1, -1]
    facing = dense(face_observed, w, [0, 1, 0, 0])
    reverse_matrix = np.zeros((4, 4), np.float32)
    for d in range(4):
        reverse_matrix[d, opposite[d]] = 1.0
    reverse_facing = dense(facing, reverse_matrix)

    # Goal outline and interior reconstruction.
    near_two = sat(cv(raw_parts[5], near_kernel))
    goal_seed = mul(raw_parts[0], near_two)
    goal_allowed = mul(
        sat(add(raw_parts[0], raw_parts[5])),
        mul(complement(crate), complement(fence_seen)),
    )
    goal_seed = mul(goal_seed, goal_allowed)
    goal_pixels = reach(goal_seed, goal_allowed, 48)

    def tile_presence(x):
        w = np.zeros((1, 1, 7, 7), np.float32)
        for dy in (-1, 0, 1, 2):
            for dx in (-1, 0, 1, 2):
                w[0, 0, 3 + dy, 3 + dx] = 1.0
        points = mul(sat(cv(x, w)), anchors)
        return fill_tiles(points)

    observed_goal = tile_presence(goal_pixels)
    gs = shift4(observed_goal)
    horizontal = mx(gs[2], gs[3])
    vertical = mx(gs[0], gs[1])
    dynamic = sat(add(add(crate, player), add(helper, hostile)))
    observed_goal = mx(
        observed_goal, mul(mul(horizontal, vertical), dynamic)
    )

    fence = mx(old_fence, fence_seen)
    goal = mul(mx(old_goal, observed_goal), complement(fence))

    # Detect sustained absence of cargo motion.
    moments = emit("gmean", [mul(crate, sl("coords", 0, 2))])
    moment_difference = emit("vadd", [
        moments,
        emit("vmul", [vs(aux, 0, 2), negone]),
    ])
    abs_difference = emit("vadd", [
        emit("vrelu", [moment_difference]),
        emit("vrelu", [
            emit("vmul", [moment_difference, negone])
        ]),
    ])
    change_strength = dense(abs_difference, [[10000.0, 10000.0]])
    changed = sat(add(zero, change_strength))
    old_stall = add(zero, vs(aux, 2, 3))
    stall = mul(
        emit("min2", [add(old_stall, increment), one]),
        complement(changed),
    )
    recover_other_cargo = sat(scale(stall, 16.0, -6.0))

    new_code = add(scale(goal, 10.0), scale(fence, 3.0))
    code_vector = pool(mul(tile_basis, new_code))
    new_aux = vcat([moments, pool(stall)])
    w = np.zeros((256, 3), np.float32)
    w[:3, :3] = np.eye(3, dtype=np.float32)
    encoded_aux = dense(new_aux, w)
    emit("vadd", [code_vector, encoded_aux], nid="mem_next")

    # Player walkability and carrying configurations.
    fixed_obstacles = sat(add(wall, add(helper, hostile)))
    remaining_crates = relu(sub(crate, held))
    walk_free = complement(sat(add(
        fixed_obstacles, add(crate, fence)
    )))
    walk_neighbors = shift4(walk_free)
    walk_neighbor_bank = cat(walk_neighbors)

    carrier_free = complement(sat(add(
        fixed_obstacles,
        add(fence, mul(remaining_crates, holding)),
    )))
    cargo_free = complement(sat(add(
        fixed_obstacles, remaining_crates
    )))
    configuration_free = mul(
        carrier_free, cat(shift4(cargo_free))
    )

    # Helper reachability and orientation-specific delivery routes.
    receiver_obstacles = sat(add(
        wall, add(hostile, remaining_crates)
    ))
    receiver_cargo_free = complement(receiver_obstacles)
    receiver_walk_free = mul(receiver_cargo_free, complement(fence))
    receiver_config = mul(
        receiver_walk_free, cat(shift4(receiver_cargo_free))
    )
    goal_scores = scale(goal, 400.0)
    oriented_goal_scores = cat(shift4(goal_scores))
    receiver_seeds = mul(oriented_goal_scores, receiver_config)
    receiver_value = value_field(
        receiver_seeds, receiver_config, 4, 256
    )
    helper_reach = reach(helper, receiver_walk_free, 192)

    receiver_at_handoff = oriented_shift(receiver_value, opposite)
    receiver_can_arrive = cat(shift4(helper_reach))
    handoff_cargo_scores = mul(
        relu(scale(receiver_at_handoff, 1.0, -12.0, 4)),
        receiver_can_arrive,
    )
    handoff_cargo_scores = mul(
        handoff_cargo_scores, mul(fence, has_helper)
    )
    handoff_seeds = oriented_shift(handoff_cargo_scores)

    carrying_seeds = mul(
        mx(oriented_goal_scores, handoff_seeds),
        configuration_free,
    )
    carry_value = value_field(
        carrying_seeds, configuration_free, 4, 256
    )
    carry_neighbors = shift4(carry_value, 4)
    extension = carry_neighbors[0]
    for p in carry_neighbors[1:]:
        extension = mx(extension, p)
    extension = relu(sub(extension, four))

    # Ordinary pickup targets.
    handoff_region = mul(fence, has_helper)
    deferred_handoff = mul(
        handoff_region, complement(recover_other_cargo)
    )
    satisfied = sat(add(goal, deferred_handoff))
    eligible = mul(crate, complement(satisfied))
    leave_other = mul(other, complement(recover_other_cargo))
    eligible = mul(eligible, complement(leave_other))
    pickup_bank = mul(cat(shift4(eligible)), extension)

    # Low-priority staging pickups: stranded fence cargo with room
    # for a backward pull. These need not already have a delivery route.
    stranded = mul(
        mul(crate, fence),
        mul(complement(goal), recover_other_cargo),
    )
    backward_room = cat([walk_neighbors[opposite[d]] for d in range(4)])
    staging_bank = mul(cat(shift4(stranded)), backward_room)
    staging_bank = scale(staging_bank, 100.0, c=4)

    hostile_bank = scale(cat(shift4(hostile)), 1000.0, c=4)
    bank = mul(
        mx(mx(pickup_bank, staging_bank), hostile_bank),
        walk_free,
    )
    bank = fill_tiles(mul(bank, anchors), 4)

    navigation_seed = max_channels(bank, 4)
    navigation_value = value_field(
        navigation_seed, walk_free, 1, 192
    )
    navigation_neighbors = cat(shift4(navigation_value))
    walking_scores = sub(navigation_neighbors, four)
    turning_scores = scale(bank, 1.0, -2.0, 4)
    unheld_moves = pool(mul(
        mx(walking_scores, turning_scores), player_point
    ))

    front_target_map = affine(
        mul(bank, facing), np.ones((1, 4), np.float32)
    )
    front_target = pool(mul(front_target_map, player_point))
    front_crate_map = affine(
        mul(cat(shift4(crate)), facing),
        np.ones((1, 4), np.float32),
    )
    front_crate = pool(mul(front_crate_map, player_point))
    front_target_present = pool(sat(add(zero, front_target)))
    unheld_interact = dense(
        vcat([front_target, front_crate, front_target_present]),
        [[1.0, -10.0, 10.0]],
        [0.2],
    )

    # Normal carrying actions.
    held_move_maps = []
    for field in carry_neighbors:
        held_move_maps.append(affine(
            mul(field, facing), np.ones((1, 4), np.float32)
        ))
    held_moves = pool(mul(
        sub(cat(held_move_maps), four), player_point
    ))

    # If the fixed carrying orientation has no route, clear the
    # original fence by two cells before releasing for reorientation.
    current_carry_map = affine(
        mul(carry_value, facing), np.ones((1, 4), np.float32)
    )
    has_carry_route = pool(sat(mul(current_carry_map, player_point)))
    no_carry_route = dense(has_carry_route, [[-1.0]], [1.0])

    fence_ahead = affine(
        mul(cat(shift4(fence)), facing),
        np.ones((1, 4), np.float32),
    )
    needs_pull = pool(mul(held, sat(add(fence, fence_ahead))))
    backward_open_map = affine(
        mul(walk_neighbor_bank, reverse_facing),
        np.ones((1, 4), np.float32),
    )
    backward_open = pool(mul(backward_open_map, player_point))
    pull_possible = emit("vmul", [needs_pull, backward_open])
    recovery_pull = emit("vmul", [no_carry_route, pull_possible])
    pull_bonus = dense(recovery_pull, [[100.0]])
    held_moves = emit("vadd", [
        held_moves,
        emit("vmul", [reverse_facing, pull_bonus]),
    ])
    recovery_drop = emit("vmul", [
        no_carry_route,
        dense(pull_possible, [[-1.0]], [1.0]),
    ])

    goal_delivery = pool(mul(held, goal))
    valid_handoff_here = affine(
        mul(handoff_seeds, facing), np.ones((1, 4), np.float32)
    )
    handoff_delivery = pool(mul(
        sat(valid_handoff_here), player_point
    ))
    delivered = emit("vadd", [goal_delivery, handoff_delivery])
    held_interact = dense(
        vcat([delivered, has_hostile, recovery_drop]),
        [[10000.0, 10000.0, 10000.0]],
        [0.2],
    )

    not_holding = dense(holding, [[-1.0]], [1.0])
    moves = emit("vadd", [
        emit("vmul", [unheld_moves, not_holding]),
        emit("vmul", [held_moves, holding]),
    ])

    # Break equal-value pursuit cycles without overriding route gains.
    w = np.zeros((4, 8), np.float32)
    for d in range(4):
        w[d, opposite[d] + 1] = -0.5
        w[d, d + 1] = 0.05
    moves = emit("vadd", [moves, dense("aprev", w)])

    interact = emit("vadd", [
        emit("vmul", [unheld_interact, not_holding]),
        emit("vmul", [held_interact, holding]),
    ])
    action_values = vcat([moves, interact])
    w = np.zeros((8, 5), np.float32)
    for i in range(5):
        w[i + 1, i] = 1.0
    b = np.full(8, -10000.0, np.float32)
    b[1:6] = 0.0
    dense(action_values, w, b, nid="logits")
    affine(ones, [[0.0]], nid="click")

    return {
        "graph": graph,
        "memory_size": 256,
        "params": params,
    }
