import numpy as np


def build():
    graph, params = [], {}
    serial = [0]
    pserial = [0]

    def uid(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def param(a):
        pserial[0] += 1
        name = "p" + str(pserial[0])
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, ins, name=None, L=None, **fields):
        if L is None:
            L = graph
        name = name or uid()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        L.append(nd)
        return name

    def conv(x, w, b=None, name=None, L=None):
        fields = {"w": w if isinstance(w, str) else param(w)}
        if b is not None:
            fields["b"] = b if isinstance(b, str) else param(b)
        return node("conv", [x], name=name, L=L, **fields)

    def linear(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b, name=name)

    def dense(x, w, b=None, name=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], name=name, **fields)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vcat(xs):
        return node("vconcat", xs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def relu(x):
        return node("relu", [x])

    one = sl("coords", 2, 3)

    def clip(x):
        return node("min2", [relu(x), one])

    constants = {}

    def constant(v):
        v = float(v)
        if v not in constants:
            constants[v] = linear(one, [[v]])
        return constants[v]

    def channel_max(x, count):
        y = sl(x, 0, 1)
        for i in range(1, count):
            y = node("max2", [y, sl(x, i, i + 1)])
        return y

    def vnot(x):
        return dense(x, [[-1]], [1])

    def vclip(x):
        a = node("vrelu", [x])
        b = node("vrelu", [dense(a, [[-1]], [1])])
        return dense(b, [[-1]], [1])

    def sum4(x):
        return dense(x, np.ones((1, 4), np.float32))

    # Exact tile-anchor basis, generated from coordinate hinge functions.
    wh = np.zeros((96, 3), np.float32)
    bh = np.zeros(96, np.float32)
    wt = np.zeros((32, 96), np.float32)
    for axis in range(2):
        for k in range(16):
            q = axis * 48 + 3 * k
            for t, shift in enumerate((-1, 0, 1)):
                wh[q + t, axis] = 63
                bh[q + t] = -(4 * k + shift)
            wt[axis * 16 + k, q:q + 3] = (1, -2, 1)

    hinges = relu(linear("coords", wh, bh))
    triangles = linear(hinges, wt)
    stripes = clip(linear(
        triangles, 4 * np.eye(32, dtype=np.float32),
        -np.ones(32, np.float32)
    ))
    wb = np.zeros((256, 32), np.float32)
    for r in range(16):
        for c in range(16):
            wb[16 * r + c, r] = 1
            wb[16 * r + c, 16 + c] = 1
    basis = relu(linear(stripes, wb, -np.ones(256, np.float32)))
    lattice = linear(basis, np.ones((1, 256), np.float32))

    # Sprite detectors and the four player-facing edge detectors.
    wd = np.zeros((12, 16, 7, 7), np.float32)
    bd = np.zeros(12, np.float32)
    inner = [(r, c) for r in (1, 2) for c in (1, 2)]
    border = [(r, c) for r in range(4) for c in range(4)
              if r in (0, 3) or c in (0, 3)]
    footprint = [(r, c) for r in range(4) for c in range(4)]

    def marks(out, colour, positions):
        for r, c in positions:
            wd[out, colour, 3 + r, 3 + c] += 1

    marks(0, 9, inner)
    bd[0] = -3
    marks(1, 9, inner)
    marks(1, 0, border)
    bd[1] = -15
    for out, colour in ((2, 5), (3, 12), (4, 15)):
        marks(out, colour, inner)
        bd[out] = -3
    marks(5, 14, footprint)
    bd[5] = -11
    marks(6, 9, inner)
    marks(6, 5, border)
    bd[6] = -15

    holes = {(0, 1), (1, 0), (2, 3), (3, 2)}
    marks(7, 1, sorted(holes))
    marks(7, 2, [p for p in footprint if p not in holes])
    bd[7] = -15

    edges = [
        [(0, c) for c in range(4)],
        [(3, c) for c in range(4)],
        [(r, 0) for r in range(4)],
        [(r, 3) for r in range(4)],
    ]
    for d in range(4):
        marks(8 + d, 0, edges[d])
        bd[8 + d] = -3

    detections = mul(clip(conv("x0", wd, bd)), lattice)
    crate, own, wall, blue, purple, player, helper_crate, new_barrier = [
        sl(detections, i, i + 1) for i in range(8)
    ]
    faces = sl(detections, 8, 12)
    held = node("gmax", [own])
    not_held = vnot(held)
    blue_present = node("gmax", [blue])

    # Goal rectangles: remove crate interiors, then intersect border rays.
    stamp_w = np.zeros((1, 1, 7, 7), np.float32)
    stamp_w[0, 0, :4, :4] = 1
    crate_cover = conv(crate, stamp_w)
    visible_nine = relu(sub(sl("x0", 9, 10), crate_cover))
    ray_init = linear(visible_nine, np.ones((4, 1), np.float32))
    ray_w = np.zeros((4, 4, 3, 3), np.float32)
    for d, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        ray_w[d, d, 1 + dr, 1 + dc] = 1
    ray_param = param(ray_w)
    ray_state = uid("raystate")
    ray_body = []
    ray_shift = conv(ray_state, ray_param, L=ray_body)
    ray_next = node("max2", [ray_shift, ray_init], L=ray_body)
    ray_loop = uid("rays")
    graph.append({
        "id": ray_loop, "op": "loop", "T": 64,
        "state": [{"id": ray_state, "init": ray_init}],
        "body": ray_body, "next": {ray_state: ray_next}
    })
    rays = ray_loop + "." + ray_state
    vertical = node("min2", [sl(rays, 0, 1), sl(rays, 1, 2)])
    horizontal = node("min2", [sl(rays, 2, 3), sl(rays, 3, 4)])
    filled = node("min2", [vertical, horizontal])
    sample_w = np.zeros((1, 1, 3, 3), np.float32)
    sample_w[0, 0, 2, 2] = 1
    new_goal = mul(conv(filled, sample_w), lattice)

    # Tile memory uses three bits: goal, barrier, completed handoff.
    memory_map = linear(mul(basis, "mem"),
                        np.ones((1, 256), np.float32))
    old_handoff = clip(linear(memory_map, [[1]], [-3]))
    old_base = sub(memory_map, mul(old_handoff, constant(4)))
    old_barrier = clip(linear(old_base, [[1]], [-1]))
    old_goal = clip(sub(old_base, mul(old_barrier, constant(2))))
    barrier = node("max2", [old_barrier, new_barrier])
    new_goal = relu(sub(new_goal, barrier))
    goal = relu(sub(node("max2", [old_goal, new_goal]), barrier))

    # Shared four-pixel shifts, implemented as two two-pixel shifts.
    dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    neighbor_first_w = np.zeros((4, 1, 5, 5), np.float32)
    neighbor_second_w = np.zeros((4, 4, 5, 5), np.float32)
    reverse_w = np.zeros((4, 4, 5, 5), np.float32)
    carry_shift_params, scalar_shift_params = [], []
    for d, (dr, dc) in enumerate(dirs):
        neighbor_first_w[d, 0, 2 + dr, 2 + dc] = 1
        neighbor_second_w[d, d, 2 + dr, 2 + dc] = 1
        reverse_w[d, d, 2 - dr, 2 - dc] = 1
        wc = np.zeros((4, 4, 5, 5), np.float32)
        for ch in range(4):
            wc[ch, ch, 2 + dr, 2 + dc] = 1
        carry_shift_params.append(param(wc))
        ws = np.zeros((1, 1, 5, 5), np.float32)
        ws[0, 0, 2 + dr, 2 + dc] = 1
        scalar_shift_params.append(param(ws))

    neighbor_first = param(neighbor_first_w)
    neighbor_second = param(neighbor_second_w)
    reverse_param = param(reverse_w)

    def neighbors(x):
        return conv(conv(x, neighbor_first), neighbor_second)

    def shifted_field(x, d, L=None):
        w = carry_shift_params[d]
        return conv(conv(x, w, L=L), w, L=L)

    def propagate(seed, valid, channels, T):
        state = uid("valuestate")
        body = []
        weights = carry_shift_params if channels == 4 else scalar_shift_params
        shifted = [conv(conv(state, w, L=body), w, L=body)
                   for w in weights]
        z = shifted[0]
        for q in shifted[1:]:
            z = node("max2", [z, q], L=body)
        z = node("sub", [z, one], L=body)
        z = node("relu", [z], L=body)
        z = node("max2", [z, seed], L=body)
        z = node("mul", [z, valid], L=body)
        loop_id = uid("values")
        graph.append({
            "id": loop_id, "op": "loop", "T": T,
            "state": [{"id": state, "init": seed}],
            "body": body, "next": {state: z}
        })
        return loop_id + "." + state

    # Far-side goal placement reduces obstruction of later deliveries.
    xy = sl("coords", 0, 2)
    goal_count = node("gmean", [goal])
    crate_count = node("gmean", [crate])
    goal_xy = node("gmean", [mul(goal, xy)])
    crate_xy = node("gmean", [mul(crate, xy)])
    displacement = node("vadd", [
        node("vmul", [crate_xy, goal_count]),
        dense(node("vmul", [goal_xy, crate_count]),
              -np.eye(2, dtype=np.float32))
    ])
    positive_r = vclip(dense(displacement, [[100000000, 0]], [0]))
    negative_r = vclip(dense(displacement, [[-100000000, 0]], [0]))
    row_sign = node("vadd", [positive_r, dense(negative_r, [[-1]])])
    positive_c = vclip(dense(displacement, [[0, 100000000]], [0]))
    negative_c = vclip(dense(displacement, [[0, -100000000]], [0]))
    col_sign = node("vadd", [positive_c, dense(negative_c, [[-1]])])
    sign = vcat([row_sign, col_sign])
    centred_xy = linear(xy, np.eye(2, dtype=np.float32), [-0.5, -0.5])
    preference = linear(mul(centred_xy, sign), [[-48, -48]], [160])
    goal_reward = mul(goal, preference)

    # Four channels correspond to the four fixed crate offsets.
    other_crates = relu(sub(crate, own))
    settled_other = mul(other_crates, goal)
    loose_other = relu(sub(other_crates, settled_other))
    occupied = add(add(wall, blue), purple)
    occupied_primary = add(
        occupied, add(settled_other, mul(loose_other, held))
    )
    partner_free = relu(sub(lattice, occupied_primary))
    actor_free_primary = relu(sub(partner_free, barrier))
    pair_valid = node("min2", [actor_free_primary, neighbors(partner_free)])
    primary_seed = mul(neighbors(goal_reward), pair_valid)
    primary = propagate(primary_seed, pair_valid, 4, 64)

    # Blue-helper reachability, including its barrier restriction.
    helper_obstacles = add(add(wall, purple), other_crates)
    helper_free = relu(sub(
        relu(sub(lattice, helper_obstacles)), barrier
    ))
    helper_reach_seed = mul(blue, constant(128))
    helper_reach = propagate(helper_reach_seed, helper_free, 1, 64)

    # Match retrieval orientation with the helper's reachable carrier cell.
    retrieval_fields = conv(conv(primary, reverse_param), reverse_param)
    opposite = np.zeros((4, 4), np.float32)
    for d, e in enumerate((1, 0, 3, 2)):
        opposite[d, e] = 1
    helper_carrier_access = linear(neighbors(clip(helper_reach)), opposite)
    retrieval_possible = mul(retrieval_fields, helper_carrier_access)
    drop_potential = mul(channel_max(retrieval_possible, 4), blue_present)
    secondary_seed = mul(neighbors(drop_potential), pair_valid)
    secondary = propagate(secondary_seed, pair_valid, 4, 24)

    facing = node("gmax", [mul(faces, player)])
    attached_orientation = node("gmax", [mul(neighbors(own), player)])

    def read4(field):
        return node("gmax", [mul(field, player)])

    primary_here = sum4(node("vmul", [
        read4(primary), attached_orientation
    ]))
    use_primary = vclip(primary_here)
    use_secondary = vnot(use_primary)
    carrying_field = add(
        mul(primary, use_primary), mul(secondary, use_secondary)
    )
    carrying_seed = add(
        mul(primary_seed, use_primary), mul(secondary_seed, use_secondary)
    )
    carrying_moves = []
    for d in range(4):
        score = sum4(node("vmul", [
            read4(shifted_field(carrying_field, d)), attached_orientation
        ]))
        carrying_moves.append(dense(score, [[1]], [-1]))
    carrying_moves_vector = vcat(carrying_moves)
    carrying_release = sum4(node("vmul", [
        read4(carrying_seed), attached_orientation
    ]))
    carrying_release = dense(carrying_release, [[1]], [0.25])
    carrying_scores = vcat([carrying_moves_vector, carrying_release])

    # Record secondary releases so the empty actor cannot undo its handoff.
    carry_move_map = add(constant(0), carrying_moves_vector)
    best_carry_move = node("gmax", [channel_max(carry_move_map, 4)])
    will_release = vclip(dense(
        vcat([carrying_release, best_carry_move]), [[4, -4]], [0]
    ))
    handoff_now = node("vmul", [
        node("vmul", [will_release, held]), use_secondary
    ])
    handoff_map = node("max2", [old_handoff, mul(own, handoff_now)])
    encoded = add(
        add(goal, mul(barrier, constant(2))),
        mul(handoff_map, constant(4))
    )
    node("gmax", [mul(basis, encoded)], name="mem_next")

    # Ownership-aware pickup sources.
    direct_reachable = clip(primary)
    direct_pickup = mul(
        linear(primary, np.eye(4, dtype=np.float32),
               np.full(4, 160, np.float32)),
        direct_reachable
    )
    pickup_quality = node("max2", [direct_pickup, secondary])
    pickup_quality = relu(sub(
        pickup_quality, mul(neighbors(helper_crate), constant(80))
    ))

    blue_adjacent = channel_max(neighbors(blue), 4)
    protected_crates = clip(add(
        blue_adjacent, mul(barrier, blue_present)
    ))
    unfinished = relu(sub(crate, mul(crate, goal)))
    unfinished = mul(unfinished, sub(one, protected_crates))
    unfinished = mul(unfinished, sub(one, old_handoff))
    pickup_targets = mul(pickup_quality, neighbors(unfinished))
    thief_targets = mul(neighbors(purple), constant(640))

    actual_free = relu(sub(
        relu(sub(lattice, add(occupied, crate))), barrier
    ))
    targets = mul(node("max2", [pickup_targets, thief_targets]), actual_free)
    free_seed = channel_max(targets, 4)
    free_field = propagate(free_seed, actual_free, 1, 96)
    empty_move_maps = node("max2", [
        sub(neighbors(free_field), one), targets
    ])
    empty_moves = read4(empty_move_maps)

    # A small anti-reversal preference breaks moving-target oscillations.
    reverse_penalty = np.zeros((4, 8), np.float32)
    reverse_penalty[0, 2] = -0.15
    reverse_penalty[1, 1] = -0.15
    reverse_penalty[2, 4] = -0.15
    reverse_penalty[3, 3] = -0.15
    empty_moves = node("vadd", [
        empty_moves, dense("aprev", reverse_penalty)
    ])

    front_quality = sum4(node("vmul", [read4(targets), facing]))
    front_exists = vclip(front_quality)
    empty_grab = dense(
        vcat([front_quality, front_exists]),
        [[1, 1024]], [-1023.75]
    )
    empty_scores = vcat([empty_moves, empty_grab])
    scores = node("vadd", [
        node("vmul", [carrying_scores, held]),
        node("vmul", [empty_scores, not_held])
    ])

    output_w = np.zeros((8, 5), np.float32)
    for k in range(5):
        output_w[k + 1, k] = 1
    output_b = np.zeros(8, np.float32)
    output_b[[0, 6, 7]] = -10000
    dense(scores, output_w, output_b, name="logits")
    linear(lattice, [[0]], name="click")

    return {"graph": graph, "memory_size": 256, "params": params}
