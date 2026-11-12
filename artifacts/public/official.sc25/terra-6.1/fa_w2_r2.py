import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]
    current = [graph]

    def fresh():
        s = "n%d" % counter[0]
        counter[0] += 1
        return s

    def param(a):
        name = "p%d" % len(params)
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, *inputs, **fields):
        name = fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        current[0].append(nd)
        return name

    def conv(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", x, **fields)

    def conv_ref(x, wname):
        return node("conv", x, w=wname)

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", x, **fields)

    def sl(x, a, b):
        return node("slice", x, start=a, stop=b)

    def vsl(x, a, b):
        return node("vslice", x, start=a, stop=b)

    def cat(*xs):
        return node("concat", *xs)

    def vcat(*xs):
        return node("vconcat", *xs)

    def va(x, n, scale=1.0, bias=0.0):
        return dense(
            x,
            np.eye(n, dtype=np.float32) * scale,
            np.full(n, bias, dtype=np.float32),
        )

    def vclip(x, n=1):
        x = node("vrelu", x)
        t = va(x, n, -1.0, 1.0)
        t = node("vrelu", t)
        return va(t, n, -1.0, 1.0)

    def vnot(x):
        return va(x, 1, -1.0, 1.0)

    def vand(a, b):
        return node("vmul", a, b)

    def vor(a, b):
        return vclip(node("vadd", a, b))

    def positive(x):
        return vclip(node("vrelu", va(x, 1, 20.0, -10.0)))

    def spatial_clip(x):
        return node("min2", one, node("relu", x))

    def spatial_not(x):
        return node("sub", one, x)

    def sum_channels(x, channels):
        return conv(x, np.ones((1, channels, 1, 1), np.float32))

    def shift_weight(channels, dy, dx, k):
        w = np.zeros((channels, channels, k, k), np.float32)
        h = k // 2
        for i in range(channels):
            w[i, i, h + dy, h + dx] = 1.0
        return w

    one = sl("coords", 2, 3)
    onev = node("gmax", one)

    colour = {
        c: sl("x0", c, c + 1)
        for c in (2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)
    }

    # Exit template, in all four rotations.
    door_base = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ], dtype=np.int32)
    door_patterns = [np.rot90(door_base, -i) for i in range(4)]

    w = np.zeros((4, 16, 7, 7), np.float32)
    for i, pattern in enumerate(door_patterns):
        for y in range(pattern.shape[0]):
            for x in range(pattern.shape[1]):
                w[i, pattern[y, x], 3 + y - 2, 3 + x - 2] = 2.0
    doors = node("relu", conv("x0", w, np.full(4, -59.0, np.float32)))

    w = np.zeros((1, 4, 7, 7), np.float32)
    for i, pattern in enumerate(door_patterns):
        for y in range(pattern.shape[0]):
            for x in range(pattern.shape[1]):
                w[0, i, 3 + 2 - y, 3 + 2 - x] = 1.0
    door_mask = spatial_clip(conv(doors, w))

    # Player templates at both scales and all orientations.
    player_base = np.array([[10, 10], [9, 9]], dtype=np.int32)
    player_patterns = [np.rot90(player_base, -i) for i in range(4)]

    wsmall = np.zeros((4, 16, 7, 7), np.float32)
    wbig = np.zeros((4, 16, 7, 7), np.float32)
    for i, p in enumerate(player_patterns):
        for y in range(2):
            for x in range(2):
                wsmall[i, p[y, x], 3 + y, 3 + x] = 2.0
        q = np.repeat(np.repeat(p, 2, axis=0), 2, axis=1)
        for y in range(4):
            for x in range(4):
                wbig[i, q[y, x], 3 + y - 1, 3 + x - 1] = 2.0

    psmall0 = node(
        "relu", conv("x0", wsmall, np.full(4, -7.0, np.float32))
    )
    pbig0 = node(
        "relu", conv("x0", wbig, np.full(4, -31.0, np.float32))
    )

    pbig = conv(pbig0, shift_weight(4, 1, 1, 3))
    pbig_mask = spatial_clip(sum_channels(pbig, 4))
    big = vclip(node("gmax", pbig_mask))

    psmall = node("mul", psmall0, spatial_not(door_mask))
    psmall = node("mul", psmall, vnot(big))
    psmall_mask = spatial_clip(sum_channels(psmall, 4))
    small = vclip(node("gmax", psmall_mask))
    player = spatial_clip(node("add", psmall_mask, pbig_mask))

    orientations = node("gmax", node("add", psmall, pbig))
    ow = np.zeros((4, 4), np.float32)
    for out_i, in_i in enumerate((2, 0, 1, 3)):
        ow[out_i, in_i] = 1.0
    facing = dense(orientations, ow)

    # Visible obstacles and fixed interface boundaries.
    boundary_w = np.zeros((2, 3, 1, 1), np.float32)
    boundary_w[0, 0, 0, 0] = 63.0
    boundary_w[1, 1, 0, 0] = 63.0
    boundary = node(
        "relu", conv("coords", boundary_w, [-46.0, -61.0])
    )
    boundary = spatial_clip(sum_channels(boundary, 2))

    obstacle_w = np.zeros((1, 16, 1, 1), np.float32)
    for c in (3, 4, 5, 6, 12, 13):
        obstacle_w[0, c, 0, 0] = 1.0
    obstacle = spatial_clip(node("add", conv("x0", obstacle_w), boundary))

    ray_obstacle_w = np.zeros((1, 16, 1, 1), np.float32)
    for c in (5, 6, 12, 13):
        ray_obstacle_w[0, c, 0, 0] = 1.0
    ray_obstacle = spatial_clip(
        node("add", conv("x0", ray_obstacle_w), boundary)
    )
    ray_free = spatial_not(ray_obstacle)

    def rectangle_sum(x, size):
        w = np.zeros((1, 1, 7, 7), np.float32)
        w[0, 0, 3:3 + size, 3:3 + size] = 1.0
        return conv(x, w)

    legal_small = node(
        "relu", node("sub", one, rectangle_sum(obstacle, 2))
    )
    legal_big = node(
        "relu", node("sub", one, rectangle_sum(obstacle, 4))
    )
    goal_small = node(
        "mul", spatial_clip(rectangle_sum(door_mask, 2)), legal_small
    )
    goal_big = node(
        "mul", spatial_clip(rectangle_sum(door_mask, 4)), legal_big
    )

    # Destructible targets.
    crate_patterns = []
    for center in (13, 12):
        p = np.full((4, 4), 6, np.int32)
        p[1:3, 1:3] = center
        crate_patterns.append(p)

    w = np.zeros((2, 16, 7, 7), np.float32)
    for i, p in enumerate(crate_patterns):
        for y in range(4):
            for x in range(4):
                w[i, p[y, x], 3 + y - 1, 3 + x - 1] = 2.0
    crates = node("relu", conv("x0", w, [-31.0, -31.0]))

    w = np.zeros((1, 2, 7, 7), np.float32)
    for i in range(2):
        for y in range(4):
            for x in range(4):
                w[0, i, 3 + 1 - y, 3 + 1 - x] = 1.0
    crate_mask = spatial_clip(conv(crates, w))
    ray_seed = cat(crate_mask, crate_mask, crate_mask, crate_mask)

    # Directional fireball visibility.
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rw = np.zeros((4, 4, 3, 3), np.float32)
    for i, (dy, dx) in enumerate(dirs):
        rw[i, i, 1 + dy, 1 + dx] = 1.0
    ray_shift = param(rw)

    ray_state = fresh()
    ray_loop_id = fresh()
    ray_body = []
    old = current[0]
    current[0] = ray_body
    ray_shifted = conv_ref(ray_state, ray_shift)
    ray_next = node(
        "max2", ray_seed, node("mul", ray_shifted, ray_free)
    )
    current[0] = old
    current[0].append({
        "id": ray_loop_id,
        "op": "loop",
        "T": 64,
        "state": [{"id": ray_state, "init": ray_seed}],
        "body": ray_body,
        "next": {ray_state: ray_next},
    })
    rays = ray_loop_id + "." + ray_state

    sw = np.zeros((4, 4, 5, 5), np.float32)
    for i, (dy, dx) in enumerate(((-1, 0), (2, 0), (0, -1), (0, 2))):
        sw[i, i, 2 + dy, 2 + dx] = 1.0
    aim_small = conv(rays, sw)

    bw = np.zeros((4, 4, 5, 5), np.float32)
    for i, (dy, dx) in enumerate(((0, 0), (2, 0), (0, 0), (0, 2))):
        bw[i, i, 2 + dy, 2 + dx] = 1.0
    aim_big = conv(aim_small, bw)

    def max_four(x):
        a = sl(x, 0, 1)
        for i in range(1, 4):
            a = node("max2", a, sl(x, i, i + 1))
        return a

    fire_small_seed = node("mul", max_four(aim_small), legal_small)
    fire_big_seed = node("mul", max_four(aim_big), legal_big)

    # Main planner: small/big routes to exit and firing positions.
    seed = cat(goal_small, goal_big, fire_small_seed, fire_big_seed)
    legal = cat(legal_small, legal_big, legal_small, legal_big)

    small_selector = dense(onev, [[1.0], [0.0], [1.0], [0.0]])
    big_selector = dense(onev, [[0.0], [1.0], [0.0], [1.0]])
    decay = dense(onev, [[0.005], [0.005], [0.005], [0.005]])

    shift_keys = []
    primary_masks = []
    fallback_masks = []
    for dy, dx in dirs:
        key = param(shift_weight(4, 2 * dy, 2 * dx, 5))
        shift_keys.append(key)
        l2 = conv_ref(legal, key)
        l4 = conv_ref(l2, key)
        primary = node(
            "add",
            node("mul", l2, small_selector),
            node("mul", l4, big_selector),
        )
        fallback = node(
            "mul",
            node("mul", l2, node("sub", one, l4)),
            big_selector,
        )
        primary_masks.append(primary)
        fallback_masks.append(fallback)

    def directional_values(state):
        result = []
        for i in range(4):
            h2 = conv_ref(state, shift_keys[i])
            h4 = conv_ref(h2, shift_keys[i])
            selected = node(
                "add",
                node("mul", h2, small_selector),
                node("mul", h4, big_selector),
            )
            q = node(
                "add",
                node("mul", selected, primary_masks[i]),
                node("mul", h2, fallback_masks[i]),
            )
            result.append(q)
        return result

    plan_state = fresh()
    plan_loop_id = fresh()
    plan_body = []
    old = current[0]
    current[0] = plan_body
    qs = directional_values(plan_state)
    best = qs[0]
    for q in qs[1:]:
        best = node("max2", best, q)
    best = node("relu", node("sub", best, decay))
    plan_next = node("max2", seed, best)
    current[0] = old
    current[0].append({
        "id": plan_loop_id,
        "op": "loop",
        "T": 80,
        "state": [{"id": plan_state, "init": seed}],
        "body": plan_body,
        "next": {plan_state: plan_next},
    })
    plan = plan_loop_id + "." + plan_state
    qmaps = directional_values(plan)

    at_player = node("gmax", node("mul", plan, player))
    gs, gb, fs, fb = [vsl(at_player, i, i + 1) for i in range(4)]
    qvectors = [node("gmax", node("mul", q, player)) for q in qmaps]

    def gather_plan_channel(channel):
        return vcat(*[vsl(q, channel, channel + 1) for q in qvectors])

    qgs = gather_plan_channel(0)
    qgb = gather_plan_channel(1)
    qfs = gather_plan_channel(2)
    qfb = gather_plan_channel(3)

    # A small player can grow if any of the permitted offsets fits a 4x4
    # player. Propagate routes from these positions through small-player space.
    grow_candidates = None
    for dy in range(-2, 1):
        for dx in range(-2, 1):
            shifted = conv(legal_big, shift_weight(1, dy, dx, 5))
            grow_candidates = (
                shifted if grow_candidates is None
                else node("max2", grow_candidates, shifted)
            )
    grow_seed = node("mul", grow_candidates, legal_small)

    grow_shift_keys = [
        param(shift_weight(1, 2 * dy, 2 * dx, 5))
        for dy, dx in dirs
    ]
    grow_decay = dense(onev, [[0.005]])

    grow_state = fresh()
    grow_loop_id = fresh()
    grow_body = []
    old = current[0]
    current[0] = grow_body
    grow_qs = [conv_ref(grow_state, k) for k in grow_shift_keys]
    grow_best = grow_qs[0]
    for q in grow_qs[1:]:
        grow_best = node("max2", grow_best, q)
    grow_best = node("relu", node("sub", grow_best, grow_decay))
    grow_best = node("mul", grow_best, legal_small)
    grow_next = node("max2", grow_seed, grow_best)
    current[0] = old
    current[0].append({
        "id": grow_loop_id,
        "op": "loop",
        "T": 80,
        "state": [{"id": grow_state, "init": grow_seed}],
        "body": grow_body,
        "next": {grow_state: grow_next},
    })
    grow_plan = grow_loop_id + "." + grow_state
    grow_value = node("gmax", node("mul", grow_plan, player))
    grow_here = vclip(node("gmax", node("mul", grow_seed, player)))
    grow_moves = vcat(*[
        node("gmax", node("mul", conv_ref(grow_plan, k), player))
        for k in grow_shift_keys
    ])

    # Spell icons.
    region_w = np.zeros((1, 3, 1, 1), np.float32)
    region_w[0, 1, 0, 0] = -63.0
    left_region = spatial_clip(conv("coords", region_w, [22.0]))

    iw = np.zeros((1, 16, 7, 7), np.float32)
    for y in range(5):
        for x in range(2):
            c = 2 if y == 2 else 6
            iw[0, c, 3 + y - 2, 3 + x] = 2.0
    fire_icon_map = node("relu", conv("x0", iw, [-19.0]))
    has_fire = vclip(node("gmax", node("mul", fire_icon_map, left_region)))

    iw = np.zeros((1, 16, 7, 7), np.float32)
    for dy, dx in ((-3, 0), (0, -3), (0, 3), (3, 0)):
        iw[0, 15, 3 + dy, 3 + dx] = 2.0
    resize_icon_map = node("relu", conv("x0", iw, [-7.0]))
    has_resize = vclip(
        node("gmax", node("mul", resize_icon_map, left_region))
    )

    iw = np.zeros((1, 16, 7, 7), np.float32)
    for oy, ox in ((0, 0), (0, 3), (3, 3)):
        for y in range(2):
            for x in range(2):
                iw[0, 11, 3 + oy + y - 1, 3 + ox + x - 1] = 2.0
    tele_icon_map = node("relu", conv("x0", iw, [-23.0]))
    has_tele = vclip(node("gmax", node("mul", tele_icon_map, left_region)))

    # Test whether the player occupies the size-specific teleport marker.
    corner_nw = conv(colour[11], shift_weight(1, -1, -1, 3))
    corner_2 = conv(colour[11], shift_weight(1, 2, 2, 5))
    corner_4 = conv(corner_2, shift_weight(1, 2, 2, 5))
    here_small = vclip(node(
        "gmax", node("mul", player, node("mul", corner_nw, corner_2))
    ))
    here_big = vclip(node(
        "gmax", node("mul", player, node("mul", corner_nw, corner_4))
    ))
    at_own_marker = vor(vand(small, here_small), vand(big, here_big))

    goal_route = vor(vand(small, positive(gs)), vand(big, positive(gb)))
    resize_for_goal = vand(
        vand(vand(big, vnot(goal_route)), positive(gs)),
        has_resize,
    )
    fire_route = vor(vand(small, positive(fs)), vand(big, positive(fb)))
    fire_mode = vand(
        vand(vand(vnot(goal_route), vnot(resize_for_goal)), fire_route),
        has_fire,
    )
    no_route = vand(
        vand(vnot(goal_route), vnot(resize_for_goal)),
        vnot(fire_mode),
    )

    # Retain a growth plan after leaving the marker and while entering
    # its resize spell. Clear it once the player is large or another
    # objective becomes available.
    growth_context = vand(
        vand(no_route, has_tele),
        has_resize,
    )
    growth_active = vor(at_own_marker, vclip("mem"))
    small_growth = vand(
        vand(vand(growth_context, small), growth_active),
        positive(grow_value),
    )
    grow_travel = vand(small_growth, vnot(grow_here))
    resize_small = vand(small_growth, grow_here)
    resize_big = vand(vand(growth_context, big), at_own_marker)
    resize_at_marker = vor(resize_small, resize_big)
    want_resize = vor(resize_for_goal, resize_at_marker)

    growth_in_progress = vor(small_growth, resize_big)
    want_tele = vand(
        vand(no_route, has_tele),
        vnot(growth_in_progress),
    )

    graph.append({
        "id": "mem_next",
        "op": "dense",
        "in": [small_growth],
        "w": param([[1.0]]),
    })

    # Select the first visible firing direction.
    aims = node(
        "vadd",
        node("vmul", node("gmax", node("mul", aim_small, player)), small),
        node("vmul", node("gmax", node("mul", aim_big, player)), big),
    )
    aim_parts = [vclip(vsl(aims, i, i + 1)) for i in range(4)]
    chosen_parts = []
    remaining = onev
    for a in aim_parts:
        chosen_parts.append(vand(remaining, a))
        remaining = vand(remaining, vnot(a))
    chosen = vcat(*chosen_parts)
    any_aim = vnot(remaining)

    face_product = node("vmul", chosen, facing)
    face_ok = vclip(dense(face_product, np.ones((1, 4), np.float32)))
    previous_exists = vclip(dense("aprev", np.ones((1, 8), np.float32)))
    face_ok = vand(face_ok, previous_exists)

    want_fire = vand(vand(fire_mode, any_aim), face_ok)
    face_move = vand(vand(fire_mode, any_aim), vnot(face_ok))
    fire_travel = vand(fire_mode, vnot(any_aim))

    goal_moves = node(
        "vadd", node("vmul", qgs, small), node("vmul", qgb, big)
    )
    fire_moves = node(
        "vadd", node("vmul", qfs, small), node("vmul", qfb, big)
    )
    moves = node(
        "vadd",
        node("vmul", goal_moves, goal_route),
        node("vmul", fire_moves, fire_travel),
    )
    moves = node(
        "vadd", moves, node("vmul", grow_moves, grow_travel)
    )
    moves = node(
        "vadd",
        moves,
        va(node("vmul", chosen, face_move), 4, 2.0, 0.0),
    )
    want_click = vor(vor(want_resize, want_fire), want_tele)

    # Spell-grid tile masks and selected-state readings.
    rows = np.array([50 + 5 * (i // 3) for i in range(9)], np.float32)
    cols = np.array([25 + 5 * (i % 3) for i in range(9)], np.float32)
    wr = np.zeros((9, 3, 1, 1), np.float32)
    wc = np.zeros((9, 3, 1, 1), np.float32)
    wr[:, 0, 0, 0] = 63.0
    wc[:, 1, 0, 0] = 63.0
    dr = node("abs", conv("coords", wr, -rows))
    dc = node("abs", conv("coords", wc, -cols))
    distance = node("add", dr, dc)
    tile_w = -np.eye(9, dtype=np.float32)[:, :, None, None]
    tiles = node("relu", conv(distance, tile_w, np.ones(9, np.float32)))

    selected = node("gmax", node("mul", tiles, colour[14]))
    selected = vclip(node("vrelu", va(selected, 9, 10.0, -5.0)), 9)

    patterns = np.zeros((9, 3), np.float32)
    for i in (1, 3, 5, 7):
        patterns[i, 0] = 1.0
    for i in (1, 4, 7):
        patterns[i, 1] = 1.0
    for i in (0, 1, 4):
        patterns[i, 2] = 1.0
    desired = dense(vcat(want_resize, want_fire, want_tele), patterns)
    both = node("vmul", selected, desired)
    mismatch = node(
        "vadd",
        node("vadd", selected, desired),
        va(both, 9, -2.0, 0.0),
    )
    extras = node("vmul", selected, va(desired, 9, -1.0, 1.0))
    priority = np.diag(np.arange(10, 1, -1, dtype=np.float32))
    tile_scores = node(
        "vadd",
        dense(mismatch, priority),
        va(extras, 9, 30.0, 0.0),
    )
    weighted_tiles = node("mul", tiles, tile_scores)
    graph.append({
        "id": "click",
        "op": "conv",
        "in": [weighted_tiles],
        "w": param(np.ones((1, 9, 1, 1), np.float32)),
    })

    head_input = vcat(moves, want_click)
    head_w = np.zeros((8, 5), np.float32)
    for i in range(4):
        head_w[i + 1, i] = 1.0
    head_w[6, 4] = 4.0
    head_b = np.array([-5, 0, 0, 0, 0, -5, -1, -5], np.float32)
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [head_input],
        "w": param(head_w),
        "b": param(head_b),
    })

    return {"graph": graph, "memory_size": 1, "params": params}
