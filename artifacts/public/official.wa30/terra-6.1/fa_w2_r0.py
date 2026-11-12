import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]
    pserial = [0]

    def fresh():
        name = "n%d" % serial[0]
        serial[0] += 1
        return name

    def param(value):
        name = "p%d" % pserial[0]
        pserial[0] += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def emit(op, inputs=None, dest=None, name=None, **fields):
        if dest is None:
            dest = graph
        if name is None:
            name = fresh()
        node = {"id": name, "op": op}
        if inputs is not None:
            node["in"] = list(inputs)
        node.update(fields)
        dest.append(node)
        return name

    def cv(x, w, bias=None, dest=None):
        fields = {"w": param(w)}
        if bias is not None:
            fields["b"] = param(bias)
        return emit("conv", [x], dest=dest, **fields)

    def dn(x, w, bias=None, dest=None):
        fields = {"w": param(w)}
        if bias is not None:
            fields["b"] = param(bias)
        return emit("dense", [x], dest=dest, **fields)

    def u(op, x, dest=None):
        return emit(op, [x], dest=dest)

    def b(op, x, y, dest=None):
        return emit(op, [x, y], dest=dest)

    def sl(x, start, stop, dest=None):
        return emit("slice", [x], dest=dest, start=start, stop=stop)

    def vsl(x, start, stop):
        return emit("vslice", [x], start=start, stop=stop)

    def scalar_map(x, weight, bias=0.0):
        return cv(
            x,
            np.array([[[[weight]]]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
        )

    def sum_channels(x, count):
        return cv(x, np.ones((1, count, 1, 1), dtype=np.float32))

    ones = emit("slice", ["coords"], start=2, stop=3)
    one_vector = u("gmax", ones)

    def threshold(x, t):
        z = scalar_map(x, 2.0, -2.0 * t)
        return b("min2", u("relu", z), ones)

    def inv(x):
        return b("sub", ones, x)

    def vand(x, y):
        return b("vmul", x, y)

    def vconst(value):
        return dn(one_vector, [[0.0]], [value])

    def vscale(x, value):
        return vand(x, vconst(value))

    def vnot(x):
        return dn(x, [[-1.0]], [1.0])

    def vclip01(x):
        positive = u("vrelu", x)
        excess = u("vrelu", b("vadd", x, vconst(-1.0)))
        return b("vadd", positive, vscale(excess, -1.0))

    def vsum4(x):
        return dn(x, np.ones((1, 4), dtype=np.float32))

    # Indicators for tile anchors at multiples of four pixels.
    w = np.zeros((32, 3, 1, 1), dtype=np.float32)
    bias = np.zeros(32, dtype=np.float32)
    for i in range(16):
        w[i, 0, 0, 0] = 63.0
        bias[i] = -4.0 * i
        w[16 + i, 1, 0, 0] = 63.0
        bias[16 + i] = -4.0 * i
    axis_distance = u("abs", cv("coords", w, bias))

    w = np.zeros((32, 32, 1, 1), dtype=np.float32)
    for i in range(32):
        w[i, i, 0, 0] = -100.0
    stripes = b(
        "min2",
        u("relu", cv(axis_distance, w, np.full(32, 50.0))),
        ones,
    )

    w = np.zeros((256, 32, 1, 1), dtype=np.float32)
    for r in range(16):
        for c in range(16):
            k = 16 * r + c
            w[k, r, 0, 0] = 1.0
            w[k, 16 + c, 0, 0] = 1.0
    tiles = u("relu", cv(stripes, w, np.full(256, -1.0)))
    anchor = sum_channels(tiles, 256)

    # Count colours in each 4x4 tile and identify the crate centre.
    colours = [2, 9, 14, 12, 5, 0, 15, 1]
    w = np.zeros((9, 16, 7, 7), dtype=np.float32)
    for k, colour in enumerate(colours):
        for dy in range(4):
            for dx in range(4):
                w[k, colour, 3 + dy, 3 + dx] = 1.0
    for dy in (1, 2):
        for dx in (1, 2):
            w[8, 9, 3 + dy, 3 + dx] = 1.0
    counts = cv("x0", w)

    c2 = sl(counts, 0, 1)
    c9 = sl(counts, 1, 2)
    c14 = sl(counts, 2, 3)
    c12 = sl(counts, 3, 4)
    c5 = sl(counts, 4, 5)
    c0 = sl(counts, 5, 6)
    c15 = sl(counts, 6, 7)
    c1 = sl(counts, 7, 8)
    core9 = sl(counts, 8, 9)

    box = b("mul", anchor, threshold(core9, 3.5))
    player = b("mul", anchor, threshold(c14, 2.5))
    blue = b("mul", anchor, threshold(c12, 7.5))
    purple = b("mul", anchor, threshold(c15, 3.5))
    wall = b("mul", anchor, threshold(c5, 9.5))
    wall = b("mul", wall, inv(box))
    grip = b("mul", box, threshold(c0, 0.5))

    has2 = threshold(c2, 0.5)
    has9 = threshold(c9, 0.5)
    hazard_now = b(
        "mul",
        anchor,
        b("mul", threshold(c2, 8.5), threshold(c1, 2.5)),
    )

    # Shared convolutions read the four neighbours one tile away.
    w1 = np.zeros((4, 1, 5, 5), dtype=np.float32)
    w2 = np.zeros((4, 4, 5, 5), dtype=np.float32)
    offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    for k, (dy, dx) in enumerate(offsets):
        w1[k, 0, 2 + dy, 2 + dx] = 1.0
        w2[k, k, 2 + dy, 2 + dx] = 1.0
    shift_w1 = param(w1)
    shift_w2 = param(w2)

    def shift4(x, dest=None):
        first = emit("conv", [x], dest=dest, w=shift_w1)
        return emit("conv", [first], dest=dest, w=shift_w2)

    def max4(x, dest=None):
        a = sl(x, 0, 1, dest)
        c = sl(x, 1, 2, dest)
        d = sl(x, 2, 3, dest)
        e = sl(x, 3, 4, dest)
        return b(
            "max2",
            b("max2", a, c, dest),
            b("max2", d, e, dest),
            dest,
        )

    # Infer goal rectangles from their visible borders.
    border = b("mul", anchor, b("mul", has2, has9))
    border = b("mul", border, inv(box))
    goal_extent = b("mul", u("rowmax", border), u("colmax", border))
    goal_extent = b("mul", goal_extent, anchor)
    visible_goal = b("mul", goal_extent, has2)
    visible_goal = b("mul", visible_goal, inv(hazard_now))
    adjacent_visible_goal = max4(shift4(visible_goal))
    covered_goal = b("mul", goal_extent, box)
    covered_goal = b("mul", covered_goal, adjacent_visible_goal)
    goal_now = b("max2", visible_goal, covered_goal)

    # Per-tile memory: goal + 2*hazard + 4*old_player + 8*old_box.
    # Tile zero also contains 16 times an inactivity counter.
    mem0 = vsl("mem", 0, 1)
    clock_limit = 12
    w = np.ones((2 * clock_limit, 1), dtype=np.float32)
    bias = np.zeros(2 * clock_limit, dtype=np.float32)
    for i in range(clock_limit):
        t = 16.0 * (i + 1)
        bias[2 * i] = -(t - 0.5)
        bias[2 * i + 1] = -t
    clock_parts = u("vrelu", dn(mem0, w, bias))
    w = np.zeros((1, 2 * clock_limit), dtype=np.float32)
    for i in range(clock_limit):
        w[0, 2 * i] = 2.0
        w[0, 2 * i + 1] = -2.0
    clock = dn(clock_parts, w)

    w = np.zeros((256, 1), dtype=np.float32)
    w[0, 0] = -16.0
    clean_mem = b("vadd", "mem", dn(clock, w))

    def extract_bit(vector, threshold_value):
        a = u("vrelu", b(
            "vadd", vector, vconst(-(threshold_value - 0.5))
        ))
        c = u("vrelu", b(
            "vadd", vector, vconst(-threshold_value)
        ))
        return vscale(b("vadd", a, vscale(c, -1.0)), 2.0)

    old_box_vec = extract_bit(clean_mem, 8.0)
    terrain_and_player = b(
        "vadd", clean_mem, vscale(old_box_vec, -8.0)
    )
    old_player_vec = extract_bit(terrain_and_player, 4.0)
    terrain = b(
        "vadd", terrain_and_player, vscale(old_player_vec, -4.0)
    )
    old_h_vec = extract_bit(terrain, 2.0)
    old_g_vec = b("vadd", terrain, vscale(old_h_vec, -2.0))

    old_h = sum_channels(b("mul", tiles, old_h_vec), 256)
    old_g = sum_channels(b("mul", tiles, old_g_vec), 256)
    old_player = sum_channels(b("mul", tiles, old_player_vec), 256)
    old_box = sum_channels(b("mul", tiles, old_box_vec), 256)

    hazard = b("max2", hazard_now, old_h)
    goal = b("max2", goal_now, old_g)
    box_changed = u("gmax", u("abs", b("sub", box, old_box)))

    # The player's black edge indicates its facing direction.
    black = emit("slice", ["x0"], start=0, stop=1)
    w = np.zeros((4, 1, 7, 7), dtype=np.float32)
    samples = [(0, 1), (3, 1), (1, 0), (1, 3)]
    for k, (dy, dx) in enumerate(samples):
        w[k, 0, 3 + dy, 3 + dx] = 1.0
    face_samples = cv(black, w)
    raw_face = u("gmax", b("mul", face_samples, player))
    face_w = np.array([
        [1, 0, 0, 0],
        [-1, 0, -1, -1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    face = dn(raw_face, face_w, np.array([0, 1, 0, 0]))

    attached = u("gmax", grip)
    unattached = vnot(attached)
    grip_direction = u("gmax", b("mul", shift4(grip), player))
    blue_exists = u("gmax", blue)

    def selected_neighbour(m):
        return sum_channels(b("mul", shift4(m), grip_direction), 4)

    def at_player(m):
        return u("gmax", b("mul", m, player))

    def neighbour_scores(m):
        return u("gmax", b("mul", shift4(m), player))

    occupied = b("max2", b("max2", wall, box), b("max2", blue, purple))
    free = b("sub", anchor, occupied)
    walk_free = b("mul", free, inv(hazard))
    pair_free = b("max2", free, grip)
    pair_player_free = b("mul", pair_free, inv(hazard))
    pair_valid = b("mul", pair_player_free, selected_neighbour(pair_free))

    # Let blue helpers handle adjacent or visibly carried crates.
    adjacent_blue = max4(shift4(blue))
    adjacent_purple = max4(shift4(purple))
    grey_box = b("mul", box, threshold(c5, 6.5))
    contested = b(
        "max2",
        grey_box,
        b("mul", box, b("max2", adjacent_blue, adjacent_purple)),
    )
    staged = b("mul", hazard, blue_exists)
    wanted_box = b("mul", box, inv(goal))
    wanted_box = b("mul", wanted_box, inv(staged))
    wanted_box = b("mul", wanted_box, inv(contested))
    targets = b("max2", wanted_box, purple)

    # Purple adversaries have a higher initial routing value.
    primary_seed = b(
        "add",
        scalar_map(wanted_box, 116.0),
        scalar_map(purple, 128.0),
    )
    fallback_boxes = b("mul", box, inv(goal))
    fallback_boxes = b("mul", fallback_boxes, inv(grip))
    fallback_seed = scalar_map(fallback_boxes, 128.0)

    goal_neighbour = selected_neighbour(goal)
    hazard_neighbour = selected_neighbour(hazard)
    goal_seed = b("mul", goal_neighbour, pair_valid)
    stage_seed = b("mul", hazard_neighbour, pair_valid)
    stage_seed = b("mul", stage_seed, blue_exists)

    def wavefront(initial, passable):
        state_id = fresh()
        body = []
        shifted = shift4(state_id, body)
        best = max4(shifted, body)
        decayed = u("relu", b("sub", best, ones, body), body)
        propagated = b("mul", decayed, passable, body)
        new_state = b("max2", state_id, propagated, body)
        loop_id = fresh()
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 112,
            "state": [{"id": state_id, "init": initial}],
            "body": body,
            "next": {state_id: new_state},
        })
        return loop_id + "." + state_id

    seek_field = wavefront(primary_seed, walk_free)
    fallback_field = wavefront(fallback_seed, walk_free)
    goal_field = wavefront(scalar_map(goal_seed, 128.0), pair_valid)
    stage_field = wavefront(scalar_map(stage_seed, 128.0), pair_valid)

    seek_scores = neighbour_scores(seek_field)
    fallback_scores = neighbour_scores(fallback_field)
    goal_scores = neighbour_scores(goal_field)
    stage_scores = neighbour_scores(stage_field)

    primary_available = vclip01(at_player(seek_field))
    no_primary = vnot(primary_available)
    fallback_reachable = vclip01(at_player(fallback_field))
    fallback_ready = vclip01(b("vadd", clock, vconst(-7.0)))
    use_fallback = vand(
        vand(no_primary, fallback_ready), fallback_reachable
    )
    idle = vand(no_primary, vnot(use_fallback))

    has_goal_path = vclip01(at_player(goal_field))
    no_goal_path = vnot(has_goal_path)
    no_stage_path = vnot(vclip01(at_player(stage_field)))

    carry_scores = b(
        "vadd",
        vand(goal_scores, has_goal_path),
        vand(stage_scores, no_goal_path),
    )
    walking_scores = b(
        "vadd",
        seek_scores,
        vand(fallback_scores, use_fallback),
    )
    direction_scores = b(
        "vadd",
        vand(walking_scores, unattached),
        vand(carry_scores, attached),
    )

    # Discourage immediate reversal, but allow it in a dead end.
    previous_neighbours = neighbour_scores(old_player)
    direction_scores = b(
        "vadd", direction_scores, vscale(previous_neighbours, -2.1)
    )

    front_targets = neighbour_scores(targets)
    front_fallback = neighbour_scores(fallback_boxes)
    front_target = vsum4(vand(front_targets, face))
    front_fallback_target = vsum4(vand(front_fallback, face))
    front_any_box = vsum4(vand(neighbour_scores(box), face))
    front_any_purple = vsum4(vand(neighbour_scores(purple), face))
    front_object = vclip01(b("vadd", front_any_box, front_any_purple))

    # After releasing an unusable attachment, prefer a side approach.
    side_w = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ], dtype=np.float32)
    side_directions = dn(face, side_w)
    free_neighbours = neighbour_scores(walk_free)
    free_sides = vand(side_directions, free_neighbours)
    has_side = vclip01(vsum4(free_sides))
    previous_five = vsl("aprev", 5, 6)
    escape = vand(
        vand(vand(previous_five, unattached), front_target),
        has_side,
    )
    side_bonus = vand(free_sides, escape)

    grab_primary = vand(front_target, primary_available)
    grab_fallback = vand(front_fallback_target, use_fallback)
    grab = vand(
        vand(vclip01(b("vadd", grab_primary, grab_fallback)), unattached),
        vnot(escape),
    )

    on_goal = at_player(goal_neighbour)
    on_stage = at_player(hazard_neighbour)
    stage_release = vand(no_goal_path, on_stage)
    impossible_release = vand(no_goal_path, no_stage_path)
    release = vand(
        attached,
        vclip01(b(
            "vadd", on_goal, b("vadd", stage_release, impossible_release)
        )),
    )

    # ACTION5 can wait while helpers work when no object is faced.
    idle_wait = vand(vand(idle, unattached), vnot(front_object))
    idle_move = vand(vand(idle, unattached), front_object)
    idle_move_scores = vand(free_neighbours, idle_move)
    action_five = vclip01(b(
        "vadd", grab, b("vadd", release, idle_wait)
    ))

    policy_features = emit(
        "vconcat",
        [direction_scores, side_bonus, idle_move_scores, action_five],
    )
    w = np.zeros((8, 13), dtype=np.float32)
    bias = np.zeros(8, dtype=np.float32)
    bias[0] = -1000.0
    bias[5] = -1.0
    bias[6] = -1000.0
    bias[7] = -1000.0
    for k in range(4):
        w[1 + k, k] = 1.0
        w[1 + k, 4 + k] = 300.0
        w[1 + k, 8 + k] = 10.0
    w[5, 12] = 512.0
    emit(
        "dense",
        [policy_features],
        name="logits",
        w=param(w),
        b=param(bias),
    )

    # Count no-target steps only while crate positions remain unchanged.
    clock_plus_one = b("vadd", clock, vconst(1.0))
    overflow = u("vrelu", b("vadd", clock_plus_one, vconst(-12.0)))
    capped_clock = b("vadd", clock_plus_one, vscale(overflow, -1.0))
    next_clock = vand(
        vand(vand(capped_clock, no_primary), vnot(box_changed)),
        unattached,
    )

    next_g = u("gmax", b("mul", tiles, goal))
    next_h = u("gmax", b("mul", tiles, hazard))
    next_p = u("gmax", b("mul", tiles, player))
    next_b = u("gmax", b("mul", tiles, box))
    memory_base = b("vadd", next_g, vscale(next_h, 2.0))
    memory_base = b("vadd", memory_base, vscale(next_p, 4.0))
    memory_base = b("vadd", memory_base, vscale(next_b, 8.0))
    w = np.zeros((256, 1), dtype=np.float32)
    w[0, 0] = 16.0
    emit("vadd", [memory_base, dn(next_clock, w)], name="mem_next")

    emit("slice", ["coords"], name="click", start=2, stop=3)

    return {"graph": graph, "memory_size": 256, "params": params}
