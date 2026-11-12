import numpy as np


def build():
    graph = []
    params = {}
    target = graph
    serial = 0
    pserial = 0

    def fresh():
        nonlocal serial
        name = "n%d" % serial
        serial += 1
        return name

    def parameter(value):
        nonlocal pserial
        name = "p%d" % pserial
        pserial += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def emit(op, inputs=None, name=None, **fields):
        node = {"id": fresh() if name is None else name, "op": op}
        if op != "loop":
            node["in"] = list(inputs)
        node.update(fields)
        target.append(node)
        return node["id"]

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv", [x], name=name, **fields)

    def shared_conv(x, w):
        return emit("conv", [x], w=w)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("dense", [x], name=name, **fields)

    def affine(x, scale=1.0, bias=0.0):
        return conv(x, [[[[scale]]]], [bias])

    def va(x, scale=1.0, bias=0.0):
        return dense(x, [[scale]], [bias])

    def unary(op, x):
        return emit(op, [x])

    def binary(op, x, y):
        return emit(op, [x, y])

    def add(x, y):
        return binary("add", x, y)

    def sub(x, y):
        return binary("sub", x, y)

    def mul(x, y):
        return binary("mul", x, y)

    def maximum(x, y):
        return binary("max2", x, y)

    def minimum(x, y):
        return binary("min2", x, y)

    def relu(x):
        return unary("relu", x)

    def cat(xs):
        return emit("concat", xs)

    def channel(x, i):
        return emit("slice", [x], start=i, stop=i + 1)

    def sum_maps(xs):
        value = xs[0]
        for x in xs[1:]:
            value = add(value, x)
        return value

    def max_maps(xs):
        value = xs[0]
        for x in xs[1:]:
            value = maximum(value, x)
        return value

    def min_maps(xs):
        value = xs[0]
        for x in xs[1:]:
            value = minimum(value, x)
        return value

    def sum_channels(x, count):
        return conv(x, np.ones((1, count, 1, 1), np.float32))

    one = dense("aprev", np.zeros((1, 8), np.float32), [1.0])
    twenty = dense("aprev", np.zeros((1, 8), np.float32), [20.0])

    def cap_one(x):
        return minimum(x, one)

    def complement(x):
        return affine(x, -1.0, 1.0)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    shift_weights = []
    for dy, dx in directions:
        w = np.zeros((1, 1, 5, 5), np.float32)
        w[0, 0, 2 + 2 * dy, 2 + 2 * dx] = 1.0
        shift_weights.append(parameter(w))

    def shift(x, direction):
        w = shift_weights[direction]
        return shared_conv(shared_conv(x, w), w)

    def neighbours(x):
        return [shift(x, d) for d in range(4)]

    def propagate(seeds, masks, iterations, decrement, regrasp=None):
        nonlocal target
        outer = target
        body = []
        states = [fresh() for _ in seeds]
        target = body
        next_values = {}

        rotation_value = None
        if regrasp is not None:
            rotation_value = relu(
                affine(max_maps(states), 1.0, -6.5)
            )
            rotation_value = mul(rotation_value, regrasp)

        for state, seed, mask in zip(states, seeds, masks):
            value = max_maps(neighbours(state))
            if decrement is not None:
                value = relu(affine(value, 1.0, -decrement))
            if rotation_value is not None:
                value = maximum(value, rotation_value)
            value = mul(value, mask)
            value = maximum(value, seed)
            next_values[state] = value

        target = outer
        loop_id = emit(
            "loop",
            T=int(iterations),
            state=[
                {"id": state, "init": seed}
                for state, seed in zip(states, seeds)
            ],
            body=body,
            next=next_values,
        )
        return [loop_id + "." + state for state in states]

    # One spatial basis channel per four-pixel cell.
    w = np.zeros((32, 3, 1, 1), np.float32)
    b = np.zeros(32, np.float32)
    for i in range(16):
        w[i, 0, 0, 0] = 63.0
        w[16 + i, 1, 0, 0] = 63.0
        b[i] = b[16 + i] = -(4 * i + 1)

    distance = unary("abs", conv("coords", w, b))
    peaks = conv(
        distance,
        (-2.0 * np.eye(32, dtype=np.float32))[:, :, None, None],
        np.full(32, 1.01, np.float32),
    )
    peaks = cap_one(relu(peaks))

    w = np.zeros((256, 32, 1, 1), np.float32)
    for r in range(16):
        for c in range(16):
            k = 16 * r + c
            w[k, r, 0, 0] = 1.0
            w[k, 16 + c, 0, 0] = 1.0
    cells = relu(conv(peaks, w, np.full(256, -1.0, np.float32)))

    sum_cells_w = parameter(np.ones((1, 256, 1, 1), np.float32))
    lattice = shared_conv(cells, sum_cells_w)
    memory_map = shared_conv(mul(cells, "mem"), sum_cells_w)

    # A common negative memory offset encodes the interception timer.
    timer = unary("gmax", affine(memory_map, -1.0))
    remaining_timer = unary("vrelu", va(timer, 1.0, -1.0))
    timer_active = emit(
        "vadd", [timer, va(remaining_timer, -1.0)]
    )
    timer_inactive = va(timer_active, -1.0, 1.0)
    old_code = add(memory_map, mul(lattice, timer))

    # Terrain integer code plus a fractional stationary-box age.
    old_nonordinary = cap_one(relu(affine(old_code, 4.0, -2.0)))
    old_porous = cap_one(relu(affine(old_code, 4.0, -6.0)))
    old_goal = relu(sub(old_nonordinary, old_porous))
    old_terrain = add(old_goal, affine(old_porous, 2.0))
    old_age = relu(affine(sub(old_code, old_terrain), 100.0))

    planes = {}

    def plane(colour):
        if colour not in planes:
            planes[colour] = channel("x0", colour)
        return planes[colour]

    def cell_colour(colour):
        return mul(lattice, plane(colour))

    player = cell_colour(14)
    boxes = cell_colour(9)
    friendly = cell_colour(12)
    hostile = cell_colour(15)
    walls = cell_colour(5)
    floor = cell_colour(2)

    corner_w = np.zeros((1, 1, 3, 3), np.float32)
    corner_w[0, 0, 0, 0] = 1.0
    corner_w = parameter(corner_w)
    held_boxes = mul(boxes, shared_conv(plane(0), corner_w))
    npc_boxes = mul(boxes, shared_conv(plane(5), corner_w))
    holding = unary("gmax", held_boxes)
    not_holding = va(holding, -1.0, 1.0)

    friend_neighbours = neighbours(friendly)
    hostile_neighbours = neighbours(hostile)
    near_friend = max_maps(friend_neighbours)
    near_hostile = max_maps(hostile_neighbours)
    friend_boxes = mul(boxes, near_friend)
    enemy_boxes = mul(boxes, near_hostile)

    stuck_boxes = mul(
        boxes, cap_one(relu(affine(old_age, 0.5, -2.0)))
    )
    protected_boxes = mul(friend_boxes, complement(stuck_boxes))

    hostile_halo2 = maximum(
        near_hostile, max_maps(neighbours(near_hostile))
    )
    hostile_halo3 = maximum(
        hostile_halo2, max_maps(neighbours(hostile_halo2))
    )
    local_hostile = unary("gmax", mul(player, hostile_halo3))
    local_wait = emit("vmul", [timer_active, local_hostile])

    # Facing from the black edge of the player sprite.
    face_w = np.zeros((3, 1, 5, 5), np.float32)
    face_w[0, 0, 1, 2] = 1.0
    face_w[1, 0, 2, 1] = 1.0
    face_w[2, 0, 2, 4] = 1.0
    face3 = unary("gmax", mul(conv(plane(0), face_w), player))
    facing = dense(
        face3,
        [[1, 0, 0], [-1, -1, -1], [0, 1, 0], [0, 0, 1]],
        [0, 1, 0, 0],
    )

    # Porous tiles have transparent pixels beside their upper-left corner.
    porous_input = cat([plane(2), plane(1)])
    porous_w = np.zeros((1, 2, 3, 3), np.float32)
    porous_w[0, 0, 1, 1] = 1.0
    porous_w[0, 0, 0, 0] = 1.0
    porous_w[0, 1, 0, 1] = 1.0
    porous_w[0, 1, 1, 0] = 1.0
    porous_seen = mul(
        lattice, relu(conv(porous_input, porous_w, [-3.0]))
    )
    porous = maximum(old_porous, porous_seen)
    pn = neighbours(porous)
    porous_bridge = maximum(
        minimum(pn[0], pn[1]), minimum(pn[2], pn[3])
    )
    porous = maximum(porous, mul(boxes, porous_bridge))
    nonporous = relu(sub(lattice, porous))

    # Detect rectangular goal borders and retain their occluded cells.
    border_w = np.zeros((2, 1, 3, 3), np.float32)
    border_w[0, 0, 1, :] = 1.0
    border_w[1, 0, :, 1] = 1.0
    borders = relu(conv(plane(9), border_w, [-2.0, -2.0]))
    horizontal = channel(borders, 0)
    vertical = channel(borders, 1)
    rectangle_support = mul(
        unary("colmax", horizontal),
        unary("rowmax", vertical),
    )
    near_border = cap_one(
        conv(borders, np.ones((1, 2, 5, 5), np.float32))
    )
    possible_surface = sum_maps(
        [floor, boxes, player, friendly, hostile]
    )
    goal_candidate = mul(rectangle_support, possible_surface)
    goal_candidate = mul(
        maximum(goal_candidate, old_goal), nonporous
    )
    goal_seed = mul(mul(near_border, floor), goal_candidate)
    goal_seed = maximum(goal_seed, mul(old_goal, nonporous))
    goal = propagate([goal_seed], [goal_candidate], 32, None)[0]
    goal = mul(maximum(goal, old_goal), nonporous)

    # Intercept a hostile behind its carried box, with a bounded wait.
    enemy_box_neighbours = neighbours(enemy_boxes)
    adjacent_enemy_box = max_maps(enemy_box_neighbours)
    hostile_adjacent = unary("gmax", mul(player, near_hostile))
    no_adjacent_hostile = va(hostile_adjacent, -1.0, 1.0)
    interception_enabled = emit(
        "vmul", [no_adjacent_hostile, timer_inactive]
    )
    interception_enabled = emit(
        "vmul", [interception_enabled, not_holding]
    )
    trigger = unary(
        "gmax",
        mul(mul(player, adjacent_enemy_box), interception_enabled),
    )
    continuing_timer = emit(
        "vmul", [remaining_timer, local_hostile]
    )
    next_timer = emit(
        "vadd", [continuing_timer, va(trigger, 7.0)]
    )

    new_age = mul(
        minimum(affine(old_age, 1.0, 1.0), twenty), boxes
    )
    terrain_code = add(goal, affine(porous, 2.0))
    memory_code = add(terrain_code, affine(new_age, 0.01))
    spatial_memory = unary("gmax", mul(cells, memory_code))
    emit(
        "vadd",
        [spatial_memory, va(next_timer, -1.0)],
        name="mem_next",
    )

    # Collision maps exclude the player's own attached box.
    occupied = sum_maps([walls, friendly, hostile, boxes])
    occupied = relu(sub(occupied, held_boxes))
    box_free = relu(sub(lattice, occupied))
    walkable = mul(box_free, nonporous)

    friend_value = propagate(
        [affine(friendly, 64.0)], [walkable], 48, 1.0
    )[0]
    friend_reachable = cap_one(friend_value)

    # Helper delivery fields reserve filled goals and relax moving cargo.
    moving_boxes = max_maps([friend_boxes, enemy_boxes, npc_boxes])
    moving_boxes = mul(moving_boxes, complement(goal))
    fixed_boxes = relu(
        sub(
            sub(sub(boxes, held_boxes), moving_boxes),
            mul(boxes, porous),
        )
    )
    helper_occupied = sum_maps([walls, hostile, fixed_boxes])
    helper_box_free = relu(sub(lattice, helper_occupied))
    helper_walkable = mul(helper_box_free, nonporous)

    helper_masks = []
    helper_seeds = []
    for d in range(4):
        pair = mul(
            helper_box_free, shift(helper_walkable, opposite[d])
        )
        helper_masks.append(pair)
        helper_seeds.append(mul(pair, affine(goal, 100.0)))

    helper_values = propagate(
        helper_seeds, helper_masks, 48, 1.0
    )

    transfer_options = []
    for d in range(4):
        reachable_carrier = shift(friend_reachable, opposite[d])
        feasible = mul(cap_one(helper_values[d]), reachable_carrier)
        helper_distance = shift(friend_value, opposite[d])
        value = add(
            affine(helper_values[d], 0.125),
            affine(helper_distance, 0.03125, 80.0),
        )
        transfer_options.append(mul(feasible, value))

    transfer_value = max_maps(transfer_options)
    handoff = mul(porous, cap_one(transfer_value))
    handoff = mul(handoff, complement(stuck_boxes))
    handoff_value = mul(handoff, transfer_value)
    destination_value = maximum(affine(goal, 100.0), handoff_value)
    release_terrain = maximum(goal, handoff)

    # Four box-centred carrying fields. Coupling implements regrasping.
    walk_neighbours = neighbours(walkable)
    regrasp_space = mul(box_free, min_maps(walk_neighbours))

    # Low-priority staging destinations recover cargo from dead handoffs.
    recovery_seed = affine(regrasp_space, 40.0)
    carry_masks = []
    carry_seeds = []
    for d in range(4):
        pair = mul(box_free, walk_neighbours[opposite[d]])
        carry_masks.append(pair)
        carry_seeds.append(
            maximum(mul(pair, destination_value), recovery_seed)
        )

    carry_values = propagate(
        carry_seeds, carry_masks, 48, 1.0,
        regrasp=regrasp_space,
    )

    eligible_boxes = relu(
        sub(sub(sub(boxes, goal), handoff), protected_boxes)
    )
    stale_boundary_boxes = mul(stuck_boxes, porous)
    eligible_boxes = maximum(eligible_boxes, stale_boundary_boxes)
    moving_npc = mul(npc_boxes, complement(stuck_boxes))
    interaction_values = []

    for d in range(4):
        first_transport = max_maps(neighbours(carry_values[d]))
        quality = relu(
            sub(
                affine(first_transport, 1.0, -3.0),
                affine(moving_npc, 12.0),
            )
        )
        pickup_at_box = mul(quality, eligible_boxes)
        pickup = mul(shift(pickup_at_box, d), walkable)
        remove_enemy = affine(
            mul(hostile_neighbours[d], walkable), 180.0
        )
        interaction_values.append(maximum(pickup, remove_enemy))

    free_seed = max_maps(interaction_values)
    free_value = propagate([free_seed], [walkable], 56, 1.0)[0]

    # Idle clear of goals, cargo and carriers.
    goal_halo1 = maximum(goal, max_maps(neighbours(goal)))
    goal_halo2 = maximum(
        goal_halo1, max_maps(neighbours(goal_halo1))
    )
    box_halo = max_maps(neighbours(boxes))
    idle_hazard = max_maps(
        [goal_halo2, near_friend, near_hostile, box_halo]
    )
    safe_idle = mul(walkable, complement(idle_hazard))
    idle_seed = affine(safe_idle, 16.0)
    idle_value = propagate(
        [idle_seed], [walkable], 8, 1.0
    )[0]

    free_neighbours = neighbours(free_value)
    idle_neighbours = neighbours(idle_value)
    free_moves = []

    for d in range(4):
        movement = relu(affine(free_neighbours[d], 1.0, -1.0))
        turning = relu(
            affine(interaction_values[d], 1.0, -0.1)
        )
        interception = affine(
            mul(enemy_box_neighbours[d], interception_enabled),
            181.0,
        )
        immediate_enemy_turn = affine(hostile_neighbours[d], 300.0)
        idle_movement = affine(
            relu(affine(idle_neighbours[d], 1.0, -1.0)), 0.1
        )
        fallback = affine(walk_neighbours[d], 0.001)
        free_moves.append(
            max_maps([
                movement, turning, interception,
                immediate_enemy_turn, idle_movement, fallback,
            ])
        )

    faced_interaction = sum_channels(
        mul(cat(interaction_values), facing), 4
    )
    faced_enemy = sum_channels(
        mul(cat(hostile_neighbours), facing), 4
    )
    immediate_remove = affine(faced_enemy, 301.0)
    idle_wait = affine(idle_seed, 0.1)
    forced_wait = mul(lattice, va(local_wait, 250.0))
    free_use = max_maps(
        [faced_interaction, immediate_remove, idle_wait, forced_wait]
    )
    free_actions = unary(
        "gmax", mul(cat(free_moves + [free_use]), player)
    )

    active_carry_value = sum_channels(
        mul(cat(carry_values), facing), 4
    )
    carry_moves = [
        relu(affine(v, 1.0, -1.0))
        for v in neighbours(active_carry_value)
    ]

    best_orientation = max_maps(carry_values)
    regrasp_drop = mul(
        regrasp_space,
        relu(affine(best_orientation, 1.0, -6.4)),
    )
    staging_drop = affine(regrasp_space, 39.5)

    reached_destination = unary(
        "gmax", mul(held_boxes, release_terrain)
    )
    carry_reachable = unary(
        "gmax", mul(held_boxes, cap_one(active_carry_value))
    )
    carry_unreachable = va(carry_reachable, -1.0, 1.0)
    immediate_drop = emit(
        "vadd", [reached_destination, carry_unreachable]
    )
    immediate_drop = mul(lattice, va(immediate_drop, 200.0))
    carry_use = max_maps(
        [regrasp_drop, staging_drop, immediate_drop]
    )
    carry_actions = unary(
        "gmax", mul(cat(carry_moves + [carry_use]), held_boxes)
    )

    free_actions = emit("vmul", [free_actions, not_holding])
    carry_actions = emit("vmul", [carry_actions, holding])
    action_values = emit("vadd", [free_actions, carry_actions])

    output_w = np.zeros((8, 5), np.float32)
    for i in range(5):
        output_w[i + 1, i] = 1.0
    output_b = np.array(
        [-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1000.0, -1000.0],
        dtype=np.float32,
    )
    dense(action_values, output_w, output_b, name="logits")
    conv(
        lattice,
        np.zeros((1, 1, 1, 1), np.float32),
        name="click",
    )

    return {
        "graph": graph,
        "memory_size": 256,
        "params": params,
    }
