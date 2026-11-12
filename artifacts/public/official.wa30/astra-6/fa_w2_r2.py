import numpy as np


def build():
    G = []
    params = {}
    serial = [0]
    constants = {}
    shift_weights = {}

    def fresh(prefix="n"):
        serial[0] += 1
        return "%s_%d" % (prefix, serial[0])

    def node(op, inputs=(), out=None, into=None, **fields):
        name = out if out is not None else fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        (G if into is None else into).append(nd)
        return name

    def conv(x, w, b=None, out=None, into=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        wn = fresh("w")
        params[wn] = w
        fields = {"w": wn}
        if b is not None:
            bn = fresh("b")
            params[bn] = np.asarray(b, dtype=np.float32)
            fields["b"] = bn
        return node("conv", [x], out=out, into=into, **fields)

    def dense(x, w, b=None, out=None):
        wn = fresh("w")
        params[wn] = np.asarray(w, dtype=np.float32)
        fields = {"w": wn}
        if b is not None:
            bn = fresh("b")
            params[bn] = np.asarray(b, dtype=np.float32)
            fields["b"] = bn
        return node("dense", [x], out=out, **fields)

    def C(value):
        value = float(value)
        if value not in constants:
            constants[value] = dense(
                "avail", np.zeros((1, 8), np.float32), [value]
            )
        return constants[value]

    def sl(x, a, b, vector=False):
        return node(
            "vslice" if vector else "slice",
            [x], start=int(a), stop=int(b)
        )

    def cat(xs, vector=False, out=None):
        return node("vconcat" if vector else "concat", xs, out=out)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def relu(a):
        return node("relu", [a])

    def vscale(a, value):
        return node("vmul", [a, C(value)])

    def vsub(a, b):
        return node("vadd", [a, vscale(b, -1)])

    def vmax(a, b):
        return node("vadd", [b, node("vrelu", [vsub(a, b)])])

    def vone_minus(a):
        return vsub(C(1), a)

    def vclip(a):
        positive = node("vrelu", [a])
        excess = node("vrelu", [
            node("vadd", [positive, C(-1)])
        ])
        return vsub(positive, excess)

    def maxs(xs, into=None):
        result = xs[0]
        for x in xs[1:]:
            result = node("max2", [result, x], into=into)
        return result

    one = sl("coords", 2, 3)
    xy = sl("coords", 0, 2)
    C(1)

    def clip(a):
        return node("min2", [relu(a), one])

    def complement(a):
        return sub(one, a)

    def total_channels(a, channels):
        return conv(a, np.ones((1, channels), np.float32))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    def shift4(x, channels, direction, into=None):
        key = (channels, direction)
        if key not in shift_weights:
            dy, dx = directions[direction]
            w = np.zeros((channels, channels, 5, 5), np.float32)
            for c in range(channels):
                w[c, c, 2 + 2 * dy, 2 + 2 * dx] = 1
            wn = fresh("shift")
            params[wn] = w
            shift_weights[key] = wn
        wn = shift_weights[key]
        z = node("conv", [x], w=wn, into=into)
        return node("conv", [z], w=wn, into=into)

    def neighbors(x, channels=1, into=None):
        return [
            shift4(x, channels, d, into=into) for d in range(4)
        ]

    def neighbor_stack(x):
        return cat(neighbors(x))

    def value_iteration(seed, valid, channels, iterations=64):
        # Independent scalar states for the rigid carrying orientations.
        if channels == 1:
            seeds = [seed]
            valids = [valid]
        else:
            seeds = [sl(seed, i, i + 1) for i in range(channels)]
            valids = [sl(valid, i, i + 1) for i in range(channels)]

        body = []
        lid = fresh("loop")
        states = []
        next_states = {}
        state_names = []
        unit = C(1)

        for i in range(channels):
            state = fresh("state")
            state_names.append(state)
            states.append({"id": state, "init": seeds[i]})
            ns = neighbors(state, into=body)
            best = maxs(ns, into=body)
            best = node("sub", [best, unit], into=body)
            best = node("relu", [best], into=body)
            best = node("max2", [best, seeds[i]], into=body)
            best = node("mul", [best, valids[i]], into=body)
            next_states[state] = best

        G.append({
            "id": lid,
            "op": "loop",
            "T": int(iterations),
            "state": states,
            "body": body,
            "next": next_states,
        })
        outputs = [lid + "." + s for s in state_names]
        return outputs[0] if channels == 1 else cat(outputs)

    # Four-pixel lattice, sampled at local sprite pixel (1,1).
    w = np.zeros((32, 3), np.float32)
    b = np.zeros(32, np.float32)
    for i in range(16):
        w[i, 0] = 63
        b[i] = -(4 * i + 1)
        w[16 + i, 1] = 63
        b[16 + i] = -(4 * i + 1)
    distances = node("abs", [conv("coords", w, b)])

    cells = (
        [(r, 2 * c) for r in range(16) for c in range(8)]
        + [(r, 2 * c + 1) for r in range(16) for c in range(8)]
    )
    w = np.zeros((256, 32), np.float32)
    for i, (r, c) in enumerate(cells):
        w[i, r] = -2
        w[i, 16 + c] = -2

    basis = clip(conv(
        distances, w, np.full(256, 1.5, np.float32)
    ))
    lattice = total_channels(basis, 256)

    # Memory: packed terrain pairs, failed-crate position and orientations.
    packed = sl("mem", 0, 128, vector=True)
    remainder = packed
    bits = {}
    for weight in (8, 4, 2):
        bit = vclip(node("vadd", [
            remainder, C(1 - weight)
        ]))
        bits[weight] = bit
        remainder = vsub(remainder, vscale(bit, weight))
    bits[1] = remainder

    old_goal_vector = cat([bits[1], bits[4]], vector=True)
    old_barrier_vector = cat([bits[2], bits[8]], vector=True)
    old_goal = total_channels(mul(basis, old_goal_vector), 256)
    old_barrier = total_channels(mul(basis, old_barrier_vector), 256)
    old_bad_position = sl("mem", 128, 130, vector=True)
    old_bad_directions = sl("mem", 130, 134, vector=True)

    # Sprite and terrain feature detectors.
    w = np.zeros((14, 16, 5, 5), np.float32)
    for out_channel, color in enumerate((9, 14, 15, 12, 5)):
        w[out_channel, color, 2, 2] = 1

    w[5, 2, 1:5, 1:5] = 1
    w[6, 9, 1:5, 1:5] = 1
    w[7, 1, 1:5, 1:5] = 1
    w[8, 0, 1, 1] = 1
    w[9, 5, 1, 1] = 1
    w[10, 0, 1, 2] = 1
    w[11, 0, 2, 1] = 1
    w[12, 0, 2, 4] = 1
    w[13, 3, 1, 1] = 1

    raw_features = conv("x0", w)
    f = [
        mul(node("min2", [
            sl(raw_features, i, i + 1), one
        ]), lattice)
        for i in range(14)
    ]

    box, player, enemy, helper, wall = f[:5]
    held = mul(box, f[8])
    robot_border = mul(box, f[9])
    highlighted_box = mul(box, f[13])
    other_box = relu(sub(box, held))
    holding = node("gmax", [held])
    not_holding = vone_minus(holding)
    helper_present = node("gmax", [helper])

    face_up = mul(player, f[10])
    face_left = mul(player, f[11])
    face_right = mul(player, f[12])
    face_down = relu(sub(sub(sub(
        player, face_up), face_left), face_right))
    face_map = cat([face_up, face_down, face_left, face_right])
    face = node("gmax", [face_map])

    observed_goal = mul(f[5], f[6])
    observed_barrier = mul(f[5], f[7])
    barrier = maxs([old_barrier, observed_barrier])

    # Recover patterned barrier cells hidden by crates.
    for _ in range(2):
        ns = neighbors(barrier)
        vertical = node("min2", [ns[0], ns[1]])
        horizontal = node("min2", [ns[2], ns[3]])
        gap = mul(box, node("max2", [vertical, horizontal]))
        barrier = node("max2", [barrier, gap])

    # Reconstruct rectangular goals from their visible borders.
    known_goal = maxs([old_goal, observed_goal])
    red_pass = relu(sub(f[5], barrier))
    occupied = maxs([box, player, enemy, helper])
    hole_pass = relu(sub(occupied, barrier))

    body = []
    state = fresh("terrain_state")
    lid = fresh("terrain_loop")
    ns = neighbors(state, into=body)
    adjacent = maxs(ns, into=body)
    spread = node("min2", [adjacent, red_pass], into=body)
    vertical = node("max2", [ns[0], ns[1]], into=body)
    horizontal = node("max2", [ns[2], ns[3]], into=body)
    corner = node("min2", [vertical, horizontal], into=body)
    corner = node("min2", [corner, hole_pass], into=body)
    grown = maxs([state, spread, corner], into=body)
    grown = node("mul", [grown, lattice], into=body)
    G.append({
        "id": lid,
        "op": "loop",
        "T": 16,
        "state": [{"id": state, "init": known_goal}],
        "body": body,
        "next": {state: grown},
    })
    goal = lid + "." + state

    bad_delta = node("abs", [sub(xy, old_bad_position)])
    bad_point = clip(conv(
        bad_delta, np.array([[-126, -126]], np.float32), [1.5]
    ))

    target_free = relu(sub(lattice, other_box))
    free_goal = mul(goal, target_free)

    # Helper reachability and four helper transport orientations.
    helper_neighbors = maxs(neighbors(helper))
    helper_space = relu(sub(sub(lattice, wall), enemy))
    helper_walkable = relu(sub(sub(
        helper_space, other_box), barrier))
    helper_reach_seed = mul(helper, C(100))
    helper_reach = value_iteration(
        helper_reach_seed, helper_walkable, 1, iterations=40
    )
    helper_reachable = clip(helper_reach)

    helper_pair_valids = []
    for d in range(4):
        own_cargo = mul(
            box, shift4(helper, 1, opposite[d])
        )
        blocking_boxes = relu(sub(other_box, own_cargo))
        foot_free = relu(sub(sub(
            helper_space, blocking_boxes), barrier))
        cargo_blockers = relu(sub(blocking_boxes, barrier))
        cargo_free = relu(sub(helper_space, cargo_blockers))
        valid = node("min2", [
            foot_free, shift4(cargo_free, 1, d)
        ])
        helper_pair_valids.append(valid)

    helper_pair_valid = cat(helper_pair_valids)
    helper_seed = mul(
        neighbor_stack(mul(free_goal, C(1000))),
        helper_pair_valid
    )
    helper_delivery = value_iteration(
        helper_seed, helper_pair_valid, 4, iterations=48
    )

    handoff_candidates = []
    useful_helper_cargo = []
    for d in range(4):
        field = sl(helper_delivery, d, d + 1)
        quality_at_robot = mul(
            relu(sub(field, C(800))), helper_reachable
        )
        handoff_candidates.append(
            shift4(quality_at_robot, 1, opposite[d])
        )
        useful_at_robot = mul(clip(field), helper)
        useful_helper_cargo.append(
            shift4(useful_at_robot, 1, opposite[d])
        )

    handoff_quality = mul(
        maxs(handoff_candidates), barrier
    )
    useful_helper_box = mul(
        box, clip(maxs(useful_helper_cargo))
    )

    robot_attached = maxs([
        robot_border,
        mul(highlighted_box, helper_neighbors)
    ])
    waiting_handoff = mul(
        clip(handoff_quality), complement(robot_attached)
    )

    # Player transport fields.
    robots = maxs([enemy, helper])
    unblocked = relu(sub(sub(lattice, wall), robots))
    actual_free = relu(sub(sub(unblocked, other_box), barrier))

    planning_boxes = mul(other_box, holding)
    cargo_free = relu(sub(unblocked, planning_boxes))
    feet_free = relu(sub(cargo_free, barrier))
    pair_valid = cat([
        node("min2", [
            feet_free, shift4(cargo_free, 1, d)
        ])
        for d in range(4)
    ])

    target_reward = add(
        mul(free_goal, C(1000)),
        mul(handoff_quality, target_free)
    )
    carrying_seed = mul(
        neighbor_stack(target_reward), pair_valid
    )
    direct_carrying_value = value_iteration(
        carrying_seed, pair_valid, 4, iterations=56
    )

    # Reorientation sites need a clear 3x3 neighborhood of game cells.
    # At least one pickup orientation there must have a direct route.
    turn_neighbors = neighbors(feet_free)
    turning_clearance = feet_free
    for q in turn_neighbors:
        turning_clearance = node("min2", [turning_clearance, q])
    for vertical_neighbor in turn_neighbors[:2]:
        for horizontal_direction in (2, 3):
            diagonal = shift4(
                vertical_neighbor, 1, horizontal_direction
            )
            turning_clearance = node("min2", [
                turning_clearance, diagonal
            ])

    future_direct_routes = []
    for d in range(4):
        orientation_field = sl(direct_carrying_value, d, d + 1)
        future_direct_routes.append(
            shift4(orientation_field, 1, opposite[d])
        )
    reorientation_sites = mul(
        turning_clearance, clip(maxs(future_direct_routes))
    )

    # This secondary field is deliberately lower priority than ordinary
    # deliveries and handoffs, and only covers short repositioning paths.
    staging_seed = mul(
        neighbor_stack(mul(reorientation_sites, C(100))),
        pair_valid
    )
    staging_value = value_iteration(
        staging_seed, pair_valid, 4, iterations=8
    )
    carrying_value = node("max2", [
        direct_carrying_value, staging_value
    ])

    # Pickup poses and hostile-robot attack poses.
    eligible_box = relu(sub(sub(sub(
        other_box, goal), useful_helper_box), waiting_handoff))
    pickup_poses = neighbor_stack(eligible_box)
    forbidden_poses = mul(
        neighbor_stack(bad_point), old_bad_directions
    )
    pickup_poses = relu(sub(pickup_poses, forbidden_poses))
    pickup_value = mul(carrying_value, pickup_poses)

    attack_value = mul(neighbor_stack(enemy), C(2000))
    interaction_value = node("max2", [
        pickup_value, attack_value
    ])
    best_interaction = maxs([
        sl(interaction_value, i, i + 1) for i in range(4)
    ])

    navigation_seed = mul(
        relu(sub(best_interaction, C(2))), actual_free
    )
    navigation_value = value_iteration(
        navigation_seed, actual_free, 1, iterations=64
    )

    # Low-priority parking away from goals, handoffs, crates and robots.
    workspace = maxs([goal, barrier, box, robots])
    workspace_ring = maxs([workspace] + neighbors(workspace))
    parking_cells = relu(sub(actual_free, workspace_ring))
    parking_seed = mul(
        mul(parking_cells, helper_present), C(50)
    )
    parking_value = value_iteration(
        parking_seed, actual_free, 1, iterations=16
    )
    parking_moves = node("gmax", [
        mul(player, neighbor_stack(parking_value))
    ])
    parking_moves = node("vmul", [
        parking_moves, not_holding
    ])
    parking_scores = node("vadd", [
        parking_moves, C(-1)
    ])
    parked_wait = node("vmul", [
        node("gmax", [mul(player, parking_seed)]),
        not_holding
    ])

    # Directional movement readout.
    navigation_moves = node("gmax", [
        mul(player, neighbor_stack(navigation_value))
    ])

    carry_move_maps = []
    for d in range(4):
        shifted = shift4(carrying_value, 4, d)
        selected = mul(shifted, face)
        carry_move_maps.append(total_channels(selected, 4))
    carrying_moves = node("gmax", [
        mul(player, cat(carry_move_maps))
    ])

    moves = node("vadd", [
        node("vmul", [navigation_moves, not_holding]),
        node("vmul", [carrying_moves, holding]),
    ])
    moves = node("vadd", [moves, C(-1)])

    # Blocked movement toward an interaction target sets facing.
    current_interactions = node("gmax", [
        mul(player, interaction_value)
    ])
    turn_scores = node("vadd", [
        node("vmul", [current_interactions, not_holding]),
        C(-2)
    ])
    movement_scores = vmax(moves, turn_scores)

    # Vacate a helper's immediate workspace even without a parking route.
    escape_map = mul(
        neighbor_stack(actual_free),
        complement(neighbor_stack(helper_neighbors))
    )
    escape_map = mul(escape_map, helper_neighbors)
    escape_map = mul(escape_map, not_holding)
    escape_values = node("gmax", [
        mul(escape_map, player)
    ])
    escape_scores = node("vadd", [
        vscale(escape_values, 2), C(-1)
    ])
    movement_scores = vmax(movement_scores, escape_scores)
    movement_scores = vmax(movement_scores, parking_scores)

    # Safe fallback waiting avoids repeatedly stealing a helper's cargo.
    blocked = relu(sub(lattice, actual_free))
    bump_values = node("gmax", [
        mul(player, neighbor_stack(blocked))
    ])
    bump_values = node("vmul", [
        bump_values, not_holding
    ])
    bump_scores = node("vadd", [
        vscale(bump_values, 0.5), C(-0.25)
    ])
    movement_scores = vmax(movement_scores, bump_scores)

    # Sub-unit tie-break discourages immediate pursuit reversals.
    w = np.zeros((4, 8), np.float32)
    for d in range(4):
        w[d, d + 1] = 0.03125
        w[d, opposite[d] + 1] = -0.25
    motion_tie_break = dense("aprev", w)
    movement_scores = node("vadd", [
        movement_scores, motion_tie_break
    ])

    # ACTION5: intentional interaction or safe parked waiting.
    faced_interaction = total_channels(
        mul(interaction_value, face), 4
    )
    grab_value = node("gmax", [
        mul(player, faced_interaction)
    ])
    grab_value = node("vmul", [
        grab_value, not_holding
    ])
    action5 = node("vrelu", [
        node("vadd", [grab_value, C(-1)])
    ])
    action5 = vmax(action5, parked_wait)

    current_carry_map = total_channels(
        mul(carrying_value, face), 4
    )
    current_carry_value = node("gmax", [
        mul(current_carry_map, player)
    ])

    current_direct_map = total_channels(
        mul(direct_carrying_value, face), 4
    )
    current_direct_value = node("gmax", [
        mul(current_direct_map, player)
    ])
    has_goal_route = vclip(node("vadd", [
        current_direct_value, C(-500)
    ]))

    # Release at a staging site only when reorientation beats continuing
    # directly in the present rigid orientation.
    current_staging_map = total_channels(
        mul(staging_seed, face), 4
    )
    current_staging_seed = node("gmax", [
        mul(current_staging_map, player)
    ])
    staging_advantage = vclip(vsub(
        current_staging_seed, current_direct_value
    ))
    staging_release = node("vmul", [
        holding, staging_advantage
    ])

    crate_on_goal = node("gmax", [mul(held, goal)])
    viable_handoff_here = node("gmax", [
        mul(held, clip(handoff_quality))
    ])
    release_handoff = node("vmul", [
        viable_handoff_here, vone_minus(has_goal_route)
    ])
    release = node("vadd", [
        crate_on_goal, release_handoff
    ])
    release = node("vadd", [release, staging_release])
    action5 = vmax(action5, vscale(release, 3000))

    action_features = cat(
        [movement_scores, action5], vector=True
    )
    w = np.zeros((8, 5), np.float32)
    for i in range(4):
        w[i + 1, i] = 1
    w[5, 4] = 1
    b = np.array(
        [-10000, 0, 0, 0, 0, 0, -10000, -10000],
        np.float32
    )
    dense(action_features, w, b, out="logits")
    node("sub", [lattice, lattice], out="click")

    # Packed terrain memory update.
    goal_bits = node("gmax", [mul(basis, goal)])
    barrier_bits = node("gmax", [mul(basis, barrier)])
    ge = sl(goal_bits, 0, 128, vector=True)
    go = sl(goal_bits, 128, 256, vector=True)
    he = sl(barrier_bits, 0, 128, vector=True)
    ho = sl(barrier_bits, 128, 256, vector=True)
    next_packed = node("vadd", [
        node("vadd", [ge, vscale(he, 2)]),
        node("vadd", [vscale(go, 4), vscale(ho, 8)])
    ])

    # Remember failed or deliberately abandoned pickup orientations.
    failure = node("vmul", [
        holding, vone_minus(vclip(current_carry_value))
    ])
    failure = vmax(failure, staging_release)
    keep_old = vone_minus(failure)
    held_position = node("gmax", [mul(xy, held)])
    same_bad_crate = node("gmax", [mul(held, bad_point)])

    accumulated_bad = vclip(node("vadd", [
        face,
        node("vmul", [
            old_bad_directions, same_bad_crate
        ])
    ]))
    next_bad_position = node("vadd", [
        node("vmul", [held_position, failure]),
        node("vmul", [old_bad_position, keep_old])
    ])
    next_bad_directions = node("vadd", [
        node("vmul", [accumulated_bad, failure]),
        node("vmul", [old_bad_directions, keep_old])
    ])

    cat(
        [next_packed, next_bad_position, next_bad_directions],
        vector=True, out="mem_next"
    )

    return {
        "graph": G,
        "memory_size": 134,
        "params": params,
    }
