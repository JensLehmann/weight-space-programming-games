import numpy as np


def build():
    params = {}
    graph = []
    nodes = graph
    counter = 0
    pcounter = 0

    def fresh(prefix="n"):
        nonlocal counter
        counter += 1
        return prefix + str(counter)

    def parameter(a):
        nonlocal pcounter
        pcounter += 1
        key = "p" + str(pcounter)
        params[key] = np.asarray(a, dtype=np.float32)
        return key

    def emit(op, inputs, name=None, **fields):
        ident = name if name is not None else fresh()
        nd = {"id": ident, "op": op, "in": list(inputs)}
        nd.update(fields)
        nodes.append(nd)
        return ident

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            b = np.broadcast_to(
                np.asarray(b, dtype=np.float32), (w.shape[0],)
            ).copy()
            fields["b"] = parameter(b)
        return emit("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            b = np.broadcast_to(
                np.asarray(b, dtype=np.float32), (w.shape[0],)
            ).copy()
            fields["b"] = parameter(b)
        return emit("dense", [x], name=name, **fields)

    def ss(x, start, stop):
        return emit("slice", [x], start=start, stop=stop)

    def vs(x, start, stop):
        return emit("vslice", [x], start=start, stop=stop)

    def sa(a, b):
        return emit("add", [a, b])

    def sm(a, b):
        return emit("mul", [a, b])

    def sx(a, b):
        return emit("max2", [a, b])

    def va(a, b):
        return emit("vadd", [a, b])

    def vm(a, b):
        return emit("vmul", [a, b])

    def vr(a):
        return emit("vrelu", [a])

    one = ss("coords", 2, 3)

    def sr(a):
        return emit("relu", [a])

    def clip(a):
        return emit("min2", [sr(a), one])

    def inv(a):
        return emit("sub", [one, a])

    def vneg(a, length):
        return dense(a, -np.eye(length, dtype=np.float32))

    def vmax(a, b, length=1):
        return va(vr(va(a, vneg(b, length))), b)

    def vmax_all(a, length):
        result = vs(a, 0, 1)
        for i in range(1, length):
            result = vmax(result, vs(a, i, i + 1))
        return result

    def vclip(a, length):
        r = vr(a)
        excess = vr(dense(r, np.eye(length), -1.0))
        return va(r, vneg(excess, length))

    def vbinary(a, length):
        return vclip(dense(a, 10.0 * np.eye(length), -5.0), length)

    def spatial_max_group(a, start, count):
        result = ss(a, start, start + 1)
        for i in range(start + 1, start + count):
            result = sx(result, ss(a, i, i + 1))
        return result

    def two_direction_maxima(a):
        s = spatial_max_group(a, 0, 4)
        b = spatial_max_group(a, 4, 4)
        return emit("concat", [s, b])

    def colours(indices):
        w = np.zeros((len(indices), 16), np.float32)
        for i, c in enumerate(indices):
            w[i, c] = 1.0
        return conv("x0", w)

    def colour_sum(indices):
        w = np.zeros((1, 16), np.float32)
        for c in indices:
            w[0, c] = 1.0
        return conv("x0", w)

    def box(xmin, xmax, ymin, ymax):
        w = np.array([
            [63.0, 0.0, 0.0],
            [-63.0, 0.0, 0.0],
            [0.0, 63.0, 0.0],
            [0.0, -63.0, 0.0],
        ], np.float32)
        b = np.array([
            1.0 - ymin, ymax + 1.0,
            1.0 - xmin, xmax + 1.0,
        ], np.float32)
        f = conv("coords", w, b)
        result = ss(f, 0, 1)
        for i in range(1, 4):
            result = emit("min2", [result, ss(f, i, i + 1)])
        return clip(result)

    # Detect the four rotations of the exit's exact 5-by-6 pattern.
    door_base = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ], dtype=np.int32)
    door_w = np.zeros((4, 16, 7, 7), np.float32)
    door_fill_w = np.zeros((1, 4, 7, 7), np.float32)
    for rotation in range(4):
        pattern = np.rot90(door_base, rotation)
        for y in range(pattern.shape[0]):
            for x in range(pattern.shape[1]):
                dy, dx = y - 2, x - 2
                door_w[rotation, int(pattern[y, x]), 3 + dy, 3 + dx] = 1
                door_fill_w[0, rotation, 3 - dy, 3 - dx] = 1
    door_centres = clip(conv("x0", door_w, -29.0))
    door_mask = clip(conv(door_centres, door_fill_w))

    # Player templates: size, facing. Anchors are one pixel from top-left.
    player_colours = colours([9, 10])
    clean_player_colours = sm(player_colours, inv(door_mask))
    player_w = np.zeros((8, 2, 7, 7), np.float32)
    player_b = np.zeros(8, np.float32)
    player_base = np.array([[10, 10], [9, 9]], dtype=np.int32)
    rotations = [2, 0, 3, 1]  # up, down, left, right
    for size_index, scale in enumerate([1, 2]):
        side = 2 * scale
        for direction, rotation in enumerate(rotations):
            oi = 4 * size_index + direction
            pattern = np.rot90(player_base, rotation)
            pattern = np.repeat(np.repeat(pattern, scale, 0), scale, 1)
            for y in range(side):
                for x in range(side):
                    player_w[oi, int(pattern[y, x]) - 9, y + 2, x + 2] = 1
            for dy in range(-2, side):
                for dx in range(-2, side):
                    if not (-1 <= dy <= side - 2 and
                            -1 <= dx <= side - 2):
                        player_w[oi, :, 3 + dy, 3 + dx] = -2
            player_b[oi] = 1.0 - side * side
    player_patterns = clip(conv(clean_player_colours, player_w, player_b))
    player_merge = np.zeros((2, 8), np.float32)
    player_merge[0, :4] = 1
    player_merge[1, 4:] = 1
    player = clip(conv(player_patterns, player_merge))
    player_sizes = emit("gmax", [player])
    face_merge = np.zeros((4, 8), np.float32)
    for d in range(4):
        face_merge[d, d] = 1
        face_merge[d, d + 4] = 1
    facing = dense(emit("gmax", [player_patterns]), face_merge)

    # Exact 10-by-10 spell-panel detection from its four corners.
    c3 = ss("x0", 3, 4)
    corner_w = np.zeros((4, 1, 3, 3), np.float32)
    for i, (dy, dx) in enumerate([(1, 1), (1, -1),
                                  (-1, 1), (-1, -1)]):
        corner_w[i, 0, 1, 1] = 1
        corner_w[i, 0, 1 + dy, 1] = 1
        corner_w[i, 0, 1, 1 + dx] = 1
        corner_w[i, 0, 1 + dy, 1 + dx] = -1
    panel_corners = clip(conv(c3, corner_w, -2.0))
    align_w = np.zeros((4, 4, 7, 7), np.float32)
    offsets = [(0, 0), (0, 3), (3, 0), (3, 3)]
    for i, (dy, dx) in enumerate(offsets):
        align_w[i, i, 3 + dy, 3 + dx] = 1
    aligned = panel_corners
    for _ in range(3):
        aligned = conv(aligned, align_w)
    panel_tl = clip(conv(aligned, np.ones((1, 4)), -3.0))
    expand_w = np.zeros((1, 1, 7, 7), np.float32)
    expand_w[0, 0, :4, :4] = 1
    panel_mask = panel_tl
    for _ in range(3):
        panel_mask = clip(conv(panel_mask, expand_w))
    spell_available = emit("gmax", [
        sm(colours([6, 11, 15]), panel_mask)
    ])
    fire_available = vs(spell_available, 0, 1)
    teleport_available = vs(spell_available, 1, 2)
    resize_available = vs(spell_available, 2, 3)

    grid_box = box(22, 38, 47, 63)
    action_strip = box(62, 63, 0, 63)
    ui_mask = clip(sa(panel_mask, grid_box))

    # Teleport anchors, preferring the currently displayed destination outline.
    portal_colours = sm(colours([11, 15]), inv(ui_mask))
    portal_w = np.zeros((4, 2, 7, 7), np.float32)
    specifications = [
        (1, [(-1, -1), (-1, 0), (0, -1), (0, 0)]),
        (0, [(-1, -1), (-1, 2), (2, -1), (2, 2)]),
        (1, [(-3, -3), (-3, 2), (2, -3), (2, 2)]),
        (0, [(-2, -2), (-2, 3), (3, -2), (3, 3)]),
    ]
    for i, (channel, points) in enumerate(specifications):
        for dy, dx in points:
            portal_w[i, channel, 3 + dy, 3 + dx] = 1
    portals = clip(conv(portal_colours, portal_w, -3.0))
    portal_targets = []
    for size in range(2):
        ordinary = ss(portals, size, size + 1)
        outline = ss(portals, size + 2, size + 3)
        has_outline = emit("gmax", [outline])
        no_outline = dense(has_outline, [[-1.0]], [1.0])
        portal_targets.append(sx(outline, sm(ordinary, no_outline)))
    teleport_targets = emit("concat", portal_targets)

    # Keys are solid four-by-four frames with either pink or cyan centres.
    key_w = np.zeros((2, 16, 7, 7), np.float32)
    key_fill_w = np.zeros((1, 2, 7, 7), np.float32)
    for key, centre_colour in enumerate([13, 12]):
        for y in range(4):
            for x in range(4):
                c = centre_colour if 1 <= y <= 2 and 1 <= x <= 2 else 6
                key_w[key, c, y + 2, x + 2] = 1
                key_fill_w[0, key, 4 - y, 4 - x] = 1
    key_centres = clip(conv("x0", key_w, -15.0))
    key_mask = clip(conv(key_centres, key_fill_w))

    strip_obstacle = sm(ss("x0", 14, 15), action_strip)
    movement_obstacles = colour_sum([0, 1, 3, 4, 5, 6, 7, 8, 12, 13])
    movement_obstacles = clip(sa(
        sa(sa(movement_obstacles, strip_obstacle), ui_mask), door_mask
    ))
    movement_cell_free = inv(movement_obstacles)

    # Growth ignores the yellow brackets, as the simulator's growth test does.
    growth_obstacles = colour_sum([0, 1, 5, 6, 7, 8, 12, 13])
    growth_obstacles = clip(sa(
        sa(growth_obstacles, strip_obstacle), ui_mask
    ))
    growth_cell_free = inv(growth_obstacles)

    footprint_w = np.zeros((2, 1, 7, 7), np.float32)
    for size, side in enumerate([2, 4]):
        for dy in range(-1, side - 1):
            for dx in range(-1, side - 1):
                footprint_w[size, 0, 3 + dy, 3 + dx] = 1
    free = clip(conv(movement_cell_free, footprint_w, [-3.0, -15.0]))
    door_overlap = clip(conv(door_mask, footprint_w))
    door_seed = conv(door_overlap, 1024.0 * np.eye(2))
    grow_free = clip(conv(
        growth_cell_free, footprint_w[1:2], -15.0
    ))

    # Select precisely the first fitting growth offset, in simulator order.
    growth_offsets = [(dy, dx) for dx in [-2, -1, 0]
                      for dy in [-2, -1, 0]]
    growth_shift_w = np.zeros((9, 1, 5, 5), np.float32)
    for i, (dy, dx) in enumerate(growth_offsets):
        growth_shift_w[i, 0, 2 + dy, 2 + dx] = 1
    growth_tests = conv(grow_free, growth_shift_w)
    first_w = np.zeros((9, 9), np.float32)
    for i in range(9):
        first_w[i, i] = 1
        first_w[i, :i] = -1
    first_growth = sr(conv(growth_tests, first_w))

    # Fire visibility propagates through ray-transparent cells in four directions.
    ray_obstacles = clip(sa(
        colour_sum([0, 1, 5, 6, 12, 13]), strip_obstacle
    ))
    ray_free = inv(ray_obstacles)
    ray_seed = conv(key_mask, np.ones((4, 1), np.float32))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ray_shift_w = np.zeros((4, 4, 3, 3), np.float32)
    for d, (dy, dx) in enumerate(directions):
        ray_shift_w[d, d, 1 + dy, 1 + dx] = 1

    ray_state = fresh("ray_state")
    ray_body = []
    outer_nodes = nodes
    nodes = ray_body
    propagated = conv(ray_state, ray_shift_w)
    propagated = sx(ray_seed, sm(propagated, ray_free))
    nodes = outer_nodes
    ray_loop = fresh("rays")
    nodes.append({
        "id": ray_loop,
        "op": "loop",
        "T": 64,
        "state": [{"id": ray_state, "init": ray_seed}],
        "body": ray_body,
        "next": {ray_state: propagated},
    })
    rays = ray_loop + "." + ray_state

    firing_w = np.zeros((8, 4, 7, 7), np.float32)
    firing_offsets = [
        [(-2, -1), (1, -1), (-1, -2), (-1, 1)],
        [(-2, -1), (3, -1), (-1, -2), (-1, 3)],
    ]
    for size in range(2):
        for d, (dy, dx) in enumerate(firing_offsets[size]):
            firing_w[4 * size + d, d, 3 + dy, 3 + dx] = 1
    firing = conv(rays, firing_w)

    replicate_w = np.zeros((8, 2), np.float32)
    replicate_w[:4, 0] = 1
    replicate_w[4:, 1] = 1
    firing = sm(sm(firing, conv(free, replicate_w)), fire_available)
    fire_seed = conv(two_direction_maxima(firing), 253.0 * np.eye(2))
    seed = sx(door_seed, fire_seed)

    # Fixed movement kernels and the large-player four-step/fallback test.
    step2_w = np.zeros((4, 1, 5, 5), np.float32)
    step2_diag_w = np.zeros((4, 4, 5, 5), np.float32)
    for d, (dy, dx) in enumerate(directions):
        step2_w[d, 0, 2 + 2 * dy, 2 + 2 * dx] = 1
        step2_diag_w[d, d, 2 + 2 * dy, 2 + 2 * dx] = 1
    big_destination_valid = sx(
        ss(free, 1, 2), ss(door_overlap, 1, 2)
    )
    big_valid2 = conv(big_destination_valid, step2_w)
    big_valid4 = conv(big_valid2, step2_diag_w)
    use_fallback = inv(big_valid4)

    def movement_values(value):
        eligible = sx(sm(value, free), door_seed)
        small_moves = conv(ss(eligible, 0, 1), step2_w)
        big_moves2 = conv(ss(eligible, 1, 2), step2_w)
        big_moves4 = conv(big_moves2, step2_diag_w)
        big_moves = sa(
            sm(big_moves4, big_valid4),
            sm(big_moves2, use_fallback),
        )
        result = emit("concat", [small_moves, big_moves])
        return sr(conv(result, np.eye(8), -1.0))

    def resize_values(value):
        big_value = ss(value, 1, 2)
        sampled = conv(big_value, growth_shift_w)
        grown = conv(sm(sampled, first_growth), np.ones((1, 9)))
        shrunk = ss(value, 0, 1)
        other = emit("concat", [grown, shrunk])
        return sm(sr(conv(other, np.eye(2), -4.0)), resize_available)

    def teleport_values(value):
        at_destinations = emit("gmax", [sm(value, teleport_targets)])
        result = vr(dense(at_destinations, np.eye(2), -3.0))
        return vm(result, teleport_available)

    # Bellman propagation over small and large player anchors.
    value_state = fresh("value_state")
    value_body = []
    outer_nodes = nodes
    nodes = value_body
    moves = movement_values(value_state)
    best_move = two_direction_maxima(moves)
    resized = resize_values(value_state)
    teleported_vector = teleport_values(value_state)
    teleported = sm(one, teleported_vector)
    value_next = sx(sx(sx(best_move, resized), teleported), seed)
    nodes = outer_nodes
    value_loop = fresh("navigation")
    nodes.append({
        "id": value_loop,
        "op": "loop",
        "T": 128,
        "state": [{"id": value_state, "init": seed}],
        "body": value_body,
        "next": {value_state: value_next},
    })
    value = value_loop + "." + value_state

    # Read movement and spell scores at the detected player anchor.
    player8 = conv(player, replicate_w)
    merge_directions = np.zeros((4, 8), np.float32)
    for d in range(4):
        merge_directions[d, d] = 1
        merge_directions[d, d + 4] = 1
    move_scores = dense(
        emit("gmax", [sm(movement_values(value), player8)]),
        merge_directions,
    )
    firing_here = dense(
        emit("gmax", [sm(firing, player8)]), merge_directions
    )
    fire_facing = vm(firing_here, facing)
    fire_score = dense(fire_facing, np.full((1, 4), 253.0))
    turn_scores = dense(firing_here, 252.5 * np.eye(4))
    move_scores = vmax(move_scores, turn_scores, 4)

    resize_score = dense(
        emit("gmax", [sm(resize_values(value), player)]),
        np.ones((1, 2)),
    )
    teleport_score = dense(
        vm(teleport_values(value), player_sizes), np.ones((1, 2))
    )
    spell_scores = emit("vconcat", [
        fire_score, teleport_score, resize_score
    ])
    spell_scores = dense(
        spell_scores, np.eye(3), [0.3, 0.2, 0.1]
    )
    best_spell = vmax_all(spell_scores, 3)
    spell_difference = va(
        spell_scores, dense(best_spell, -np.ones((3, 1)))
    )
    spell_choice = vbinary(
        vr(dense(spell_difference, 20.0 * np.eye(3), 1.0)), 3
    )

    # Fixed click centres and a robust readout of the nine selected cells.
    centres = [(50 + 5 * r, 25 + 5 * c)
               for r in range(3) for c in range(3)]
    coordinate_w = np.zeros((18, 3), np.float32)
    coordinate_b = np.zeros(18, np.float32)
    for i, (y, x) in enumerate(centres):
        coordinate_w[2 * i, 0] = 63
        coordinate_w[2 * i + 1, 1] = 63
        coordinate_b[2 * i] = -y
        coordinate_b[2 * i + 1] = -x
    coordinate_distance = emit("abs", [
        conv("coords", coordinate_w, coordinate_b)
    ])
    point_w = np.zeros((9, 18), np.float32)
    for i in range(9):
        point_w[i, 2 * i:2 * i + 2] = -1
    click_points = sr(conv(coordinate_distance, point_w, 1.0))
    selected_colour = colour_sum([6, 11, 14, 15])
    selected = vbinary(
        emit("gmax", [sm(click_points, selected_colour)]), 9
    )

    # Lock a spell only across consecutive clicks on a nonempty pattern.
    any_selected = vmax_all(selected, 9)
    any_memory = vmax_all("mem", 3)
    previous_click = vs("aprev", 6, 7)
    lock = vm(vm(any_selected, any_memory), previous_click)
    unlocked = dense(lock, [[-1.0]], [1.0])
    emit(
        "vadd",
        [vm("mem", lock), vm(spell_choice, unlocked)],
        name="mem_next",
    )

    # Columns: fire, teleport, resize.
    patterns = np.zeros((9, 3), np.float32)
    patterns[[1, 4, 7], 0] = 1
    patterns[[0, 1, 4], 1] = 1
    patterns[[1, 3, 5, 7], 2] = 1
    desired = dense("mem_next", patterns)
    difference = va(desired, vneg(selected, 9))
    missing = vr(difference)
    unwanted = vr(vneg(difference, 9))
    click_scores = va(
        dense(missing, 10.0 * np.eye(9)),
        dense(unwanted, 20.0 * np.eye(9)),
    )
    click_scores = dense(
        click_scores, np.eye(9),
        np.arange(9, 0, -1, dtype=np.float32) * 0.001,
    )
    conv(
        sm(click_points, click_scores),
        np.ones((1, 9)),
        name="click",
    )

    action_features = emit("vconcat", [move_scores, best_spell, lock])
    action_w = np.zeros((8, 6), np.float32)
    action_b = np.full(8, -10000.0, np.float32)
    for d in range(4):
        action_w[d + 1, d] = 1
        action_b[d + 1] = 0
    action_w[6, 4] = 1
    action_w[6, 5] = 2048
    action_b[6] = -0.5
    dense(action_features, action_w, action_b, name="logits")

    return {
        "graph": graph,
        "memory_size": 3,
        "params": params,
    }
