import numpy as np


def build():
    graph = []
    params = {}
    channels = {
        "x0": 16, "coords": 3, "mem": 3,
        "aprev": 8, "avail": 8,
    }
    current = graph
    counter = 0

    def put(op, inputs, n, name=None, **fields):
        nonlocal counter
        if name is None:
            name = "n%d" % counter
            counter += 1
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        current.append(nd)
        channels[name] = n
        return name

    def conv(a, w, b=None, name=None):
        nonlocal counter
        w = np.asarray(w, dtype=np.float32)
        pn = "w%d" % counter
        counter += 1
        params[pn] = w
        fields = {"w": pn}
        if b is not None:
            bn = "b%d" % counter
            counter += 1
            params[bn] = np.asarray(b, dtype=np.float32)
            fields["b"] = bn
        return put("conv", [a], w.shape[0], name, **fields)

    def ls(a, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        return conv(a, w[:, :, None, None], b, name)

    def vd(a, w, b=None, name=None):
        nonlocal counter
        w = np.asarray(w, dtype=np.float32)
        pn = "w%d" % counter
        counter += 1
        params[pn] = w
        fields = {"w": pn}
        if b is not None:
            bn = "b%d" % counter
            counter += 1
            params[bn] = np.asarray(b, dtype=np.float32)
            fields["b"] = bn
        return put("dense", [a], w.shape[0], name, **fields)

    def cat(*xs):
        return put("concat", xs, sum(channels[x] for x in xs))

    def vc(*xs):
        return put("vconcat", xs, sum(channels[x] for x in xs))

    def sl(a, start, stop):
        return put("slice", [a], stop - start, start=start, stop=stop)

    def vsl(a, start, stop):
        return put("vslice", [a], stop - start, start=start, stop=stop)

    def unary(op, a):
        return put(op, [a], channels[a])

    def pair(op, a, b, name=None):
        return put(op, [a, b], max(channels[a], channels[b]), name)

    def gm(a):
        return put("gmax", [a], channels[a])

    def aff(a, scale=1.0, bias=0.0):
        n = channels[a]
        return ls(a, np.eye(n) * scale, np.full(n, bias))

    def vaff(a, scale=1.0, bias=0.0):
        n = channels[a]
        return vd(a, np.eye(n) * scale, np.full(n, bias))

    def clip(a):
        r = unary("relu", a)
        excess = unary("relu", aff(r, 1, -1))
        return pair("sub", r, excess)

    def vclip(a):
        r = unary("vrelu", a)
        excess = unary("vrelu", vaff(r, 1, -1))
        return vsub(r, excess)

    def vsub(a, b):
        n = channels[a]
        return vd(vc(a, b), np.concatenate([np.eye(n), -np.eye(n)], axis=1))

    def vmax(a, b):
        return pair("vadd", b, unary("vrelu", vsub(a, b)))

    def shift(a, entries):
        # entries: (input channel, row displacement, column displacement)
        radius = max(max(abs(dy), abs(dx)) for _, dy, dx in entries)
        k = 2 * radius + 1
        w = np.zeros((len(entries), channels[a], k, k), np.float32)
        for o, (c, dy, dx) in enumerate(entries):
            w[o, c, radius + dy, radius + dx] = 1
        return conv(a, w)

    def sum_channels(a):
        return ls(a, np.ones((1, channels[a]), np.float32))

    def max_four_pairs(a):
        # Input ordering: four small-scale directions, then four large.
        outputs = []
        for d in range(4):
            w = np.zeros((2, 8), np.float32)
            w[0, d] = 1
            w[1, 4 + d] = 1
            outputs.append(ls(a, w))
        return pair(
            "max2",
            pair("max2", outputs[0], outputs[1]),
            pair("max2", outputs[2], outputs[3]),
        )

    one = sl("coords", 2, 3)
    xy = sl("coords", 0, 2)

    # Fixed spell-board geometry, as specified by the simulator.
    bw = np.zeros((4, 3), np.float32)
    bw[0, 1], bw[1, 1] = 126, -126
    bw[2, 0], bw[3, 0] = 126, -126
    halves = clip(ls("coords", bw, [-43, 77, -93, 127]))
    board = pair(
        "min2",
        pair("min2", sl(halves, 0, 1), sl(halves, 1, 2)),
        pair("min2", sl(halves, 2, 3), sl(halves, 3, 4)),
    )

    # Detect 10x10 spell panels by their one-pixel upper-left border.
    mw = np.zeros((1, 16, 7, 7), np.float32)
    mw[0, 3, 0, 0] = 1
    mw[0, 3, 0, 1] = 1
    mw[0, 3, 1, 0] = 1
    mw[0, 3, 1, 1] = -1
    menu_anchor = unary("relu", conv("x0", mw, [-2]))
    menu_anchor = pair("mul", menu_anchor, aff(board, -1, 1))
    menu7 = clip(conv(menu_anchor, np.ones((1, 1, 7, 7))))
    ew = np.zeros((1, 1, 7, 7), np.float32)
    ew[0, 0, :4, :4] = 1
    menus = clip(conv(menu7, ew))
    ui = pair("max2", menus, board)
    world = aff(ui, -1, 1)

    cw = np.zeros((3, 16), np.float32)
    cw[0, 6] = 1       # fire
    cw[1, 15] = 1      # resize
    cw[2, 11] = 1      # teleport
    capabilities = gm(pair("mul", ls("x0", cw), menus))
    cap_fire = vsl(capabilities, 0, 1)
    cap_resize = vsl(capabilities, 1, 2)
    cap_tele = vsl(capabilities, 2, 3)

    # Exit templates: anchors are two pixels below/right of top-left.
    exit_base = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ])
    exit_templates = [np.rot90(exit_base, k) for k in range(4)]
    exw = np.zeros((4, 16, 7, 7), np.float32)
    excover = np.zeros((1, 4, 7, 7), np.float32)
    for o, t in enumerate(exit_templates):
        h, w = t.shape
        for r in range(h):
            for c in range(w):
                exw[o, int(t[r, c]), r + 1, c + 1] = 1
                excover[0, o, 5 - r, 5 - c] = 1
    exits = unary("relu", conv("x0", exw, [-29] * 4))
    exit_pixels = clip(conv(exits, excover))

    # Player detectors, ordered up, down, left, right.
    player_patterns = [
        np.array([[9, 9], [10, 10]]),
        np.array([[10, 10], [9, 9]]),
        np.array([[9, 10], [9, 10]]),
        np.array([[10, 9], [10, 9]]),
    ]
    player_colors = pair(
        "mul", sl("x0", 9, 11), aff(exit_pixels, -1, 1)
    )

    pw_big = np.zeros((4, 2, 7, 7), np.float32)
    pw_small = np.zeros((4, 2, 3, 3), np.float32)
    for o, t in enumerate(player_patterns):
        for r in range(2):
            for c in range(2):
                pw_small[o, int(t[r, c]) - 9, r + 1, c + 1] = 1
        big = np.repeat(np.repeat(t, 2, axis=0), 2, axis=1)
        for r in range(4):
            for c in range(4):
                pw_big[o, int(big[r, c]) - 9, r + 3, c + 3] = 1

    player_big = unary("relu", conv(player_colors, pw_big, [-15] * 4))
    big_present = gm(sum_channels(player_big))
    player_small = unary("relu", conv(player_colors, pw_small, [-3] * 4))
    player_small = pair(
        "mul", player_small, vaff(big_present, -1, 1)
    )
    player = cat(sum_channels(player_small), sum_channels(player_big))
    facing = gm(pair("max2", player_small, player_big))

    # Recognize both torch types and fill their complete hit rectangles.
    tw = np.zeros((2, 16, 7, 7), np.float32)
    tcov = np.zeros((1, 2, 7, 7), np.float32)
    for o, center_color in enumerate((13, 12)):
        for r in range(4):
            for c in range(4):
                color = center_color if r in (1, 2) and c in (1, 2) else 6
                tw[o, color, r + 3, c + 3] = 1
                tcov[0, o, 3 - r, 3 - c] = 1
    torches = unary("relu", conv("x0", tw, [-15, -15]))
    torch_pixels = clip(conv(torches, tcov))

    # Ordinary movement obstacles. Unknown solid colours are conservative.
    allowed = {2, 9, 10, 11, 14, 15}
    ow = np.array([[0 if c in allowed else 1 for c in range(16)]],
                  dtype=np.float32)
    occupied = pair("max2", ls("x0", ow), ui)

    # Fire ignores the yellow decorative/collidable fixtures.
    ray_allowed = allowed | {4}
    rw = np.array([[0 if c in ray_allowed else 1 for c in range(16)]],
                  dtype=np.float32)
    ray_blocked = pair("max2", ls("x0", rw), ui)
    ray_pass = aff(ray_blocked, -1, 1)

    # Four directional visibility propagations from torch rectangles.
    ray_seed = ls(torch_pixels, np.ones((4, 1), np.float32))
    body = []
    current = body
    channels["ray_state"] = 4
    ray_shift = shift("ray_state", [
        (0, -1, 0), (1, 1, 0), (2, 0, -1), (3, 0, 1)
    ])
    ray_through = pair("mul", ray_shift, ray_pass)
    ray_next = pair("max2", ray_seed, ray_through)
    current = graph
    graph.append({
        "id": "R", "op": "loop", "T": 64,
        "state": [{"id": "ray_state", "init": ray_seed}],
        "body": body, "next": {"ray_state": ray_next},
    })
    channels["R.ray_state"] = 4

    fire_small = shift("R.ray_state", [
        (0, 0, 0), (1, 1, 0), (2, 0, 0), (3, 0, 1)
    ])
    fire_big = shift("R.ray_state", [
        (0, 0, 0), (1, 3, 0), (2, 0, 0), (3, 0, 3)
    ])
    fire_states = pair("mul", cat(fire_small, fire_big), cap_fire)

    # Footprint validity, including screen boundaries.
    obstacle_and_one = cat(occupied, one)

    def footprint_free(size):
        k = 3 if size == 2 else 7
        center = k // 2
        w = np.zeros((1, 2, k, k), np.float32)
        for r in range(size):
            for c in range(size):
                w[0, 0, center + r, center + c] = -1
                w[0, 1, center + r, center + c] = 1
        return unary("relu", conv(
            obstacle_and_one, w, [-(size * size - 1)]
        ))

    free_small = footprint_free(2)
    free_big = footprint_free(4)

    def overlap_exit(size):
        k = 3 if size == 2 else 7
        center = k // 2
        w = np.zeros((1, 1, k, k), np.float32)
        for r in range(size):
            for c in range(size):
                w[0, 0, center + r, center + c] = 1
        return clip(conv(exit_pixels, w))

    exit_small = overlap_exit(2)
    exit_big = overlap_exit(4)
    exit_states = cat(exit_small, exit_big)
    nav_pass = pair("max2", cat(free_small, free_big), exit_states)

    goal = pair(
        "max2",
        aff(exit_states, 1024),
        aff(max_four_pairs(fire_states), 512),
    )

    # Teleport-pad templates.
    def corner_detector(color, offsets, k=7):
        w = np.zeros((1, 16, k, k), np.float32)
        h = k // 2
        for dy, dx in offsets:
            w[0, color, h + dy, h + dx] = 1
        return unary("relu", conv("x0", w, [-3]))

    large_pads = corner_detector(11, [
        (0, 0), (0, 3), (3, 0), (3, 3)
    ])
    small_pads = corner_detector(15, [
        (0, 0), (0, 1), (1, 0), (1, 1)
    ], 3)
    large_pads = pair("mul", large_pads, world)
    small_pads = pair("mul", small_pads, world)

    large_marker_center = corner_detector(11, [
        (-2, -2), (-2, 3), (3, -2), (3, 3)
    ])
    large_marker = pair(
        "mul", shift(large_marker_center, [(0, 1, 1)]), world
    )
    small_marker = pair("mul", corner_detector(15, [
        (-2, -2), (-2, 3), (3, -2), (3, 3)
    ]), world)

    # Remember the visible large destination while it is hidden at small size.
    marker_present = gm(large_marker)
    marker_xy = gm(pair("mul", xy, large_marker))
    observed_memory = vc(marker_xy, marker_present)
    retained_memory = pair(
        "vmul", "mem", vaff(marker_present, -1, 1)
    )
    mem_next = pair(
        "vadd", retained_memory, observed_memory, name="mem_next"
    )
    remembered_xy = vsl(mem_next, 0, 2)
    remembered_valid = vsl(mem_next, 2, 3)
    coordinate_error = unary(
        "abs", pair("sub", xy, remembered_xy)
    )
    remembered_point = unary("relu", ls(
        coordinate_error, [[-126, -126]], [1]
    ))
    remembered_point = clip(aff(remembered_point, 2, -0.5))
    remembered_point = pair("mul", remembered_point, remembered_valid)

    target_big = pair(
        "max2",
        remembered_point,
        pair("mul", large_pads, vaff(remembered_valid, -1, 1)),
    )
    small_marker_present = gm(small_marker)
    target_small = pair(
        "max2",
        small_marker,
        pair("mul", small_pads, vaff(small_marker_present, -1, 1)),
    )
    teleport_targets = cat(target_small, target_big)

    # Static movement destination masks and the large-player fallback rule.
    dirs2 = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    small_valid = shift(sl(nav_pass, 0, 1),
                        [(0, dy, dx) for dy, dx in dirs2])
    big_valid2 = shift(sl(nav_pass, 1, 2),
                       [(0, dy, dx) for dy, dx in dirs2])
    big_valid4 = shift(big_valid2,
                      [(d, dy, dx) for d, (dy, dx) in enumerate(dirs2)])
    big_fallback = pair(
        "mul", big_valid2, aff(big_valid4, -1, 1)
    )

    # Growth uses the simulator's lexicographically first fitting offset.
    growth_offsets = [(dy, dx)
                      for dx in range(-2, 1)
                      for dy in range(-2, 1)]
    growth_valid = shift(
        free_big, [(0, dy, dx) for dy, dx in growth_offsets]
    )
    first_w = np.eye(9, dtype=np.float32)
    for j in range(9):
        first_w[j, :j] = -1
    growth_first = unary("relu", ls(growth_valid, first_w))

    def transitions(value, update=True):
        small_v = sl(value, 0, 1)
        big_v = sl(value, 1, 2)

        sm = pair("mul", shift(
            small_v, [(0, dy, dx) for dy, dx in dirs2]
        ), small_valid)
        b2 = shift(big_v, [(0, dy, dx) for dy, dx in dirs2])
        b4 = shift(b2, [
            (d, dy, dx) for d, (dy, dx) in enumerate(dirs2)
        ])
        bm = pair(
            "add",
            pair("mul", b4, big_valid4),
            pair("mul", b2, big_fallback),
        )
        movement = cat(sm, bm)

        grown = sum_channels(pair(
            "mul",
            shift(big_v, [(0, dy, dx) for dy, dx in growth_offsets]),
            growth_first,
        ))
        resize = unary("relu", aff(cat(grown, small_v), 1, -5))
        resize = pair("mul", resize, cap_resize)

        destination_values = gm(pair("mul", value, teleport_targets))
        teleport_v = unary("vrelu", vaff(destination_values, 1, -4))
        teleport_v = pair("vmul", teleport_v, cap_tele)
        teleport = pair("mul", nav_pass, teleport_v)

        if not update:
            return movement, resize, teleport

        moving_value = unary(
            "relu", aff(max_four_pairs(movement), 1, -1)
        )
        nxt = pair("max2", moving_value, resize)
        nxt = pair("max2", nxt, teleport)
        nxt = pair("max2", nxt, goal)
        nxt = pair("mul", nxt, nav_pass)
        return nxt

    # Two-scale, weighted value propagation.
    body = []
    current = body
    channels["value_state"] = 2
    value_next = transitions("value_state")
    current = graph
    graph.append({
        "id": "V", "op": "loop", "T": 160,
        "state": [{"id": "value_state", "zeros": [2, 64, 64]}],
        "body": body, "next": {"value_state": value_next},
    })
    channels["V.value_state"] = 2
    movement, resize, teleport = transitions("V.value_state", False)

    # Read action values at the player's current top-left.
    repeat_player = np.zeros((8, 2), np.float32)
    repeat_player[:4, 0] = 1
    repeat_player[4:, 1] = 1
    player8 = ls(player, repeat_player)
    combine_directions = np.concatenate(
        [np.eye(4), np.eye(4)], axis=1
    )

    movement_root = gm(pair("mul", movement, player8))
    moves = unary("vrelu", vd(
        movement_root, combine_directions, [-1] * 4
    ))
    resize_root = vd(
        gm(pair("mul", resize, player)), [[1, 1]]
    )
    teleport_root = vd(
        gm(pair("mul", teleport, player)), [[1, 1]]
    )

    visible_directions = vd(
        gm(pair("mul", fire_states, player8)), combine_directions
    )
    facing_visible = vd(
        pair("vmul", visible_directions, facing), [[1, 1, 1, 1]]
    )
    turn_needed = pair(
        "vmul", visible_directions, vaff(facing_visible, -1, 1)
    )
    moves = vmax(moves, vaff(turn_needed, 512))
    fire_root = vaff(facing_visible, 512.5)

    spell_values = vc(fire_root, resize_root, teleport_root)
    click_value = vmax(vmax(fire_root, resize_root), teleport_root)

    # Exact pairwise winner selection, with deterministic spell tie-breaking.
    priorities = [0.375, 0.25, 0.125]
    comparisons = [(0, 1), (0, 2), (1, 0),
                   (1, 2), (2, 0), (2, 1)]
    cmp_w = np.zeros((6, 3), np.float32)
    cmp_b = np.zeros(6, np.float32)
    for k, (i, j) in enumerate(comparisons):
        cmp_w[k, i] = 8
        cmp_w[k, j] = -8
        cmp_b[k] = 8 * (priorities[i] - priorities[j])
    wins = vclip(vd(spell_values, cmp_w, cmp_b))
    winner_w = np.zeros((3, 6), np.float32)
    for i in range(3):
        winner_w[i, 2 * i:2 * i + 2] = 1
    spell_winner = unary("vrelu", vd(wins, winner_w, [-1] * 3))

    # Nine sharply localized slot-centre maps.
    slot_w = np.zeros((18, 3), np.float32)
    slot_b = np.zeros(18, np.float32)
    for r in range(3):
        for c in range(3):
            j = 3 * r + c
            slot_w[2 * j, 0] = 63
            slot_w[2 * j + 1, 1] = 63
            slot_b[2 * j] = -(50 + 5 * r)
            slot_b[2 * j + 1] = -(25 + 5 * c)
    slot_errors = unary("abs", ls("coords", slot_w, slot_b))
    slot_sum_w = np.zeros((9, 18), np.float32)
    for j in range(9):
        slot_sum_w[j, 2 * j:2 * j + 2] = -2
    slots = unary("relu", ls(slot_errors, slot_sum_w, [1] * 9))

    selected_w = np.zeros((1, 16), np.float32)
    selected_w[0, [6, 11, 14, 15]] = 1
    selected_bits = gm(pair("mul", slots, ls("x0", selected_w)))
    selected_bits = vclip(vaff(selected_bits, 2, -0.5))

    patterns = np.zeros((3, 9), np.float32)
    patterns[0, [1, 4, 7]] = 1          # fire
    patterns[1, [1, 3, 5, 7]] = 1       # resize
    patterns[2, [0, 1, 4]] = 1          # teleport
    desired = sum_channels(pair(
        "mul", ls(slots, patterns), spell_winner
    ))
    selected = sum_channels(pair("mul", slots, selected_bits))
    remove_cells = unary("relu", pair("sub", selected, desired))
    add_cells = unary("relu", pair("sub", desired, selected))
    pair("add", aff(remove_cells, 10), add_cells, name="click")

    action_features = vc(moves, click_value)
    action_w = np.zeros((8, 5), np.float32)
    for d in range(4):
        action_w[d + 1, d] = 1
    action_w[6, 4] = 1
    action_b = np.zeros(8, np.float32)
    action_b[[0, 5, 7]] = -10000
    vd(action_features, action_w, action_b, name="logits")

    return {
        "graph": graph,
        "memory_size": 3,
        "params": params,
    }
