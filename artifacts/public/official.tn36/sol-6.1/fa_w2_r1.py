import numpy as np


def build():
    graph = []
    params = {}
    cache = {}
    constants = {}
    serial = [0]

    def node(op, inputs=None, out=None, target=None, **fields):
        serial[0] += 1
        ident = out if out is not None else "n%d" % serial[0]
        nd = {"id": ident, "op": op}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        (graph if target is None else target).append(nd)
        return ident

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p%d" % len(params)
            cache[key] = name
            params[name] = a
        return cache[key]

    def conv(x, w, b=None, target=None, out=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], target=target, out=out, **fields)

    def dense(x, w, b=None, out=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], out=out, **fields)

    def affine(x, w, b=None, target=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        return conv(x, w[:, :, None, None], b, target=target)

    def sl(x, start, stop, vector=False):
        return node("vslice" if vector else "slice", [x],
                    start=int(start), stop=int(stop))

    def cat(xs, vector=False):
        return node("vconcat" if vector else "concat", xs)

    def add(a, b, vector=False, target=None):
        return node("vadd" if vector else "add", [a, b], target=target)

    def mul(a, b, vector=False, target=None):
        return node("vmul" if vector else "mul", [a, b], target=target)

    def relu(x, vector=False, target=None):
        return node("vrelu" if vector else "relu", [x], target=target)

    def maximum(a, b, target=None):
        return node("max2", [a, b], target=target)

    def const(values):
        values = np.asarray(values, np.float32).reshape(-1)
        key = values.tobytes()
        if key not in constants:
            constants[key] = dense(
                "avail", np.zeros((len(values), 8), np.float32), values)
        return constants[key]

    def shift(x, dy, dx, channels=1, target=None):
        # Output at (r,c) reads input at (r-dy,c-dx).
        dy, dx = int(dy), int(dx)
        while dy or dx:
            def part(d):
                if abs(d) in (4, 8):
                    return 2 if d > 0 else -2
                return max(-3, min(3, d))

            sy, sx = part(dy), part(dx)
            radius = max(abs(sy), abs(sx))
            k = 2 * radius + 1
            w = np.zeros((channels, channels, k, k), np.float32)
            for c in range(channels):
                w[c, c, radius - sy, radius - sx] = 1
            x = conv(x, w, target=target)
            dy -= sy
            dx -= sx
        return x

    def complement(v):
        return dense(v, -np.eye(1, dtype=np.float32), [1])

    def snap_vector(v, length):
        # Exact 0/1 output for inputs near zero or one.
        z = dense(v, np.eye(length, dtype=np.float32) * 4,
                  np.full(length, -1, np.float32))
        a = relu(z, vector=True)
        b = relu(dense(z, np.eye(length, dtype=np.float32),
                       np.full(length, -1, np.float32)), vector=True)
        return dense(cat([a, b], vector=True),
                     np.concatenate([np.eye(length), -np.eye(length)], axis=1))

    one = sl("coords", 2, 3)
    pix = affine("coords", [[63, 0, 0], [0, 63, 0]])
    zero4 = const([0, 0, 0, 0])

    # Binary colour occupancy used for object-template recognition.
    colours = [11, 9, 8, 15]
    cw = np.zeros((1, 16), np.float32)
    cw[0, colours] = 1
    occupied = affine("x0", cw)

    player_template = np.ones((4, 4), np.float32)
    player_template[3, 1:3] = 0
    target_template = np.zeros((6, 6), np.float32)
    target_template[1:5, 0] = 1
    target_template[1:5, 5] = 1
    target_template[4, 2:4] = 1
    target_template[5, :] = 1
    checkpoint_template = (
        (np.indices((4, 4)).sum(axis=0) % 2) == 0
    ).astype(np.float32)

    templates = []
    for base in (player_template, target_template):
        for r in range(4):
            templates.append(np.rot90(base, -r).copy())
    templates.append(checkpoint_template)

    # Restrict object origins to the world, excluding the bottom controls.
    wy = affine("coords", [[-63, 0, 0]], [41])
    wy = relu(wy)
    wy_tail = relu(affine(wy, [[1]], [-1]))
    world_origin = node("sub", [wy, wy_tail])

    players = []
    goals = []
    checkpoints = []
    player_rots = []
    goal_rots = []

    for scale in range(1, 5):
        if scale == 1:
            block, groups, centre, k = 6, 1, 3, 7
        elif scale == 2:
            block, groups, centre, k = 3, 2, 2, 5
        elif scale == 3:
            block, groups, centre, k = 3, 2, 3, 7
        else:
            block, groups, centre, k = 2, 3, 2, 5

        shifted = []
        for by in range(groups):
            for bx in range(groups):
                shifted.append(shift(
                    occupied,
                    -(by * block * scale + centre),
                    -(bx * block * scale + centre)))
        inp = shifted[0] if len(shifted) == 1 else cat(shifted)

        w = np.zeros((9, len(shifted), k, k), np.float32)
        expected = np.zeros(9, np.float32)
        radius = k // 2

        for t, pattern in enumerate(templates):
            expected[t] = pattern.sum()
            h, width = pattern.shape
            for r in range(h):
                for c in range(width):
                    by, bx = r // block, c // block
                    channel = by * groups + bx
                    rr = (r % block) * scale - centre
                    cc = (c % block) * scale - centre
                    w[t, channel, radius + rr, radius + cc] = (
                        1 if pattern[r, c] else -1)

        matched = relu(conv(inp, w, -expected + 0.5))
        matched = affine(matched, np.eye(9, dtype=np.float32) * 2)

        # A scaled template produces a small plateau. Keep its top-left origin.
        aw = np.zeros((9, 9, 3, 3), np.float32)
        for c in range(9):
            aw[c, c, 1, 1] = 1
            aw[c, c, 0, 1] = -1
            aw[c, c, 1, 0] = -1
        anchored = mul(relu(conv(matched, aw)), world_origin)

        pr = sl(anchored, 0, 4)
        gr = sl(anchored, 4, 8)
        ck = sl(anchored, 8, 9)
        player_rots.append(pr)
        goal_rots.append(gr)
        players.append(affine(pr, np.ones((1, 4), np.float32)))
        goals.append(shift(
            affine(gr, np.ones((1, 4), np.float32)), scale, scale))
        checkpoints.append(ck)

    player_map = cat(players)
    goal_map = cat(goals)
    checkpoint_map = cat(checkpoints)
    player_rot_map = cat(player_rots)
    goal_rot_map = cat(goal_rots)

    player_presence = affine(player_map, np.ones((1, 4), np.float32))
    player_xy = dense(
        node("gmean", [mul(player_presence, pix)]),
        np.eye(2, dtype=np.float32) * 4096)
    player_scale = node("gmax", [player_map])
    p16 = node("gmax", [player_rot_map])
    g16 = node("gmax", [goal_rot_map])

    rotation_reduce = np.zeros((4, 16), np.float32)
    for s in range(4):
        for r in range(4):
            rotation_reduce[r, 4 * s + r] = 1
    player_rotation = dense(p16, rotation_reduce)
    goal_rotation = dense(g16, rotation_reduce)

    # Read the goal colour at a known occupied pixel in each rotated template.
    goal_origin_mask = affine(goal_rot_map, np.ones((1, 16), np.float32))
    goal_xy = dense(
        node("gmean", [mul(goal_origin_mask, pix)]),
        np.eye(2, dtype=np.float32) * 4096)
    offsets = np.zeros((2, 16), np.float32)
    for s in range(1, 5):
        for r in range(4):
            pattern = templates[4 + r]
            rr, cc = np.argwhere(pattern > 0)[0]
            offsets[:, 4 * (s - 1) + r] = [rr * s, cc * s]
    goal_sample_xy = add(goal_xy, dense(g16, offsets), vector=True)

    sample_offset = dense(
        player_scale, np.asarray([[1, 2, 3, 4], [1, 2, 3, 4]], np.float32))
    player_sample_xy = add(player_xy, sample_offset, vector=True)

    colour_w = np.zeros((4, 16), np.float32)
    for i, c in enumerate(colours):
        colour_w[i, c] = 1
    colour_planes = affine("x0", colour_w)

    def point_mask(xy):
        difference = node("sub", [pix, xy])
        distance = node("abs", [difference])
        return relu(affine(distance, [[-1, -1]], [1]))

    goal_colour = snap_vector(
        node("gmax", [mul(point_mask(goal_sample_xy), colour_planes)]), 4)
    player_colour = snap_vector(
        node("gmax", [mul(point_mask(player_sample_xy), colour_planes)]), 4)

    # Relative rotation: indices 0,1,2,3 mean 0,90,180,270 degrees.
    relative_parts = []
    for delta in range(4):
        perm = np.zeros((4, 4), np.float32)
        for old in range(4):
            perm[(old + delta) % 4, old] = 1
        overlap = mul(dense(player_rotation, perm), goal_rotation, vector=True)
        relative_parts.append(dense(overlap, np.ones((1, 4), np.float32)))
    relative_rotation = cat(relative_parts, vector=True)
    need_rotation = dense(relative_rotation, [[0, 1, 1, 1]])

    same_colour = dense(
        mul(player_colour, goal_colour, vector=True),
        np.ones((1, 4), np.float32))
    changeable_colour = dense(goal_colour, [[0, 1, 1, 1]])
    need_colour = mul(changeable_colour, complement(same_colour), vector=True)
    colour_without_rotation = mul(
        need_colour, complement(need_rotation), vector=True)
    colour_after_rotation = mul(need_colour, need_rotation, vector=True)

    def bit_matrix(codes):
        return np.asarray(
            [[(code >> bit) & 1 for code in codes] for bit in range(6)],
            np.float32)

    rotation_bits = dense(relative_rotation, bit_matrix([0, 5, 7, 6]))
    colour_bits = dense(goal_colour, bit_matrix([0, 14, 15, 63]))

    # Read horizontal and vertical toggle centres on black program panels.
    mark_w = np.zeros((1, 16), np.float32)
    mark_w[0, 1] = mark_w[0, 5] = 1
    mark_colour = affine("x0", mark_w)
    black = sl("x0", 0, 1)
    toggle_input = cat([mark_colour, black])
    tw = np.zeros((2, 2, 5, 5), np.float32)

    for dc in (-1, 0, 1):
        tw[0, 0, 2, 2 + dc] = 1
    for dr, dc in ((-1, 0), (1, 0), (0, -2), (0, 2)):
        tw[0, 1, 2 + dr, 2 + dc] = 1

    for dr in (-1, 0, 1):
        tw[1, 0, 2 + dr, 2] = 1
    for dr, dc in ((0, -1), (0, 1), (-2, 0), (2, 0)):
        tw[1, 1, 2 + dr, 2 + dc] = 1

    toggle = relu(conv(toggle_input, tw, [-6.5, -6.5]))
    toggle = affine(toggle, [[2, 2]])

    band = node("abs", [affine("coords", [[63, 0, 0]], [-40.5])])
    band = relu(affine(band, [[-1]], [10.5]))
    band = node("sub", [band, relu(affine(band, [[1]], [-1]))])
    toggle = mul(toggle, band)
    current_bits = mul(toggle, sl("x0", 5, 6))

    columns = node("colmax", [toggle])
    rows = node("rowmax", [toggle])

    # Parallel prefix counts rank columns from left to right and bits top down.
    prefix_input = cat([columns, rows])
    prefix_body = []
    pw = np.zeros((2, 2, 3, 3), np.float32)
    pw[0, 0, 1, 0] = 1
    pw[1, 1, 0, 1] = 1
    carried = conv("prefix_carry", pw, target=prefix_body)
    accumulated = add("prefix_total", carried, target=prefix_body)
    node(
        "loop", out="prefix_loop", T=63,
        state=[
            {"id": "prefix_carry", "init": prefix_input},
            {"id": "prefix_total", "init": prefix_input},
        ],
        body=prefix_body,
        next={"prefix_carry": carried, "prefix_total": accumulated})
    ranked = "prefix_loop.prefix_total"
    column_rank = sl(ranked, 0, 1)
    row_rank = sl(ranked, 1, 2)
    counts = node("gmax", [ranked])
    number_columns = dense(counts, [[1, 0]])
    six_bits = dense(counts, [[0, 0.25]], [-0.5])

    # One-hot number of program columns, for lengths 1 through 6.
    nw = np.ones((18, 1), np.float32)
    nb = []
    for n in range(1, 7):
        nb.extend([-(n - 1), -n, -(n + 1)])
    nr = relu(dense(number_columns, nw, nb), vector=True)
    nreduce = np.zeros((6, 18), np.float32)
    for i in range(6):
        nreduce[i, 3 * i:3 * i + 3] = [1, -2, 1]
    length_onehot = dense(nr, nreduce)

    rw = np.ones((6, 1), np.float32)
    rd = affine(row_rank, rw, -np.arange(1, 7, dtype=np.float32))
    rd = node("abs", [rd])
    bit_rows = mul(
        relu(affine(rd, -np.eye(6, dtype=np.float32),
                    np.ones(6, np.float32))), rows)

    # Square-footprint wall masks for scales 1,2,3,4.
    wall = sl("x0", 6, 7)

    def enlarge(square, amount):
        horizontal = maximum(square, shift(square, 0, -amount))
        return maximum(horizontal, shift(horizontal, -amount, 0))

    square2 = enlarge(wall, 1)
    square4 = enlarge(square2, 2)
    square8 = enlarge(square4, 4)
    square12 = enlarge(square8, 4)
    square16 = enlarge(square8, 8)
    wall_footprints = cat([square4, square8, square12, square16])

    # Keep the whole player footprint within the represented screen.
    border_w = np.zeros((8, 3), np.float32)
    border_b = np.zeros(8, np.float32)
    for s in range(1, 5):
        border_w[2 * (s - 1), 0] = 63
        border_w[2 * (s - 1) + 1, 1] = 63
        border_b[2 * (s - 1):2 * s] = -(64 - 4 * s)
    excess = relu(affine("coords", border_w, border_b))
    br = np.zeros((4, 8), np.float32)
    for s in range(4):
        br[s, 2 * s:2 * s + 2] = 1
    border = affine(excess, br)
    invalid = maximum(wall_footprints, border)
    valid = relu(affine(
        invalid, -np.eye(4, dtype=np.float32), np.ones(4, np.float32)))

    up_scale_values = np.zeros((4, 4), np.float32)
    down_scale_values = np.zeros((4, 4), np.float32)
    for s in range(4):
        up_scale_values[s, min(3, s + 1)] = 1
        down_scale_values[s, max(0, s - 1)] = 1

    nav_codes = [0, 1, 2, 3, 33, 10, 12, 8, 9]
    minus_one_w = np.eye(4, dtype=np.float32)
    minus_one_b = np.full(4, -1, np.float32)

    def successors(value, seed, target=None):
        left = shift(value, 0, 4, 4, target)
        right = shift(value, 0, -4, 4, target)
        down = shift(value, -4, 0, 4, target)
        up = shift(value, 4, 0, 4, target)
        right8 = shift(right, 0, -4, 4, target)
        left8 = shift(left, 0, 4, 4, target)
        grow = affine(value, up_scale_values, target=target)
        shrink = affine(value, down_scale_values, target=target)
        sources = [value, left, right, down, up, right8, left8, grow, shrink]

        result = []
        for i, src in enumerate(sources):
            q = relu(affine(
                src, minus_one_w, minus_one_b, target=target), target=target)
            if i == 0:
                q = maximum(q, seed, target=target)
            elif i >= 4:
                q = mul(q, six_bits, target=target)
            result.append(q)
        return result

    def reduce_successors(values, target=None):
        result = values[0]
        for v in values[1:]:
            result = maximum(result, v, target=target)
        return mul(result, valid, target=target)

    # Long-range navigation potential.
    goal_seed = affine(goal_map, np.eye(4, dtype=np.float32) * 100)
    goal_seed = mul(goal_seed, valid)
    bellman_body = []
    bellman_next = reduce_successors(
        successors("bellman_value", goal_seed, bellman_body), bellman_body)
    node(
        "loop", out="navigation_loop", T=32,
        state=[{"id": "bellman_value", "init": goal_seed}],
        body=bellman_body,
        next={"bellman_value": bellman_next})
    potential = "navigation_loop.bellman_value"

    # Checkpoint scores favour endpoints with a shorter remaining route.
    checkpoint_seed = mul(checkpoint_map, potential)
    checkpoint_seed = affine(
        checkpoint_seed, np.eye(4, dtype=np.float32) * 100)
    finish_seed = affine(goal_map, np.eye(4, dtype=np.float32) * 20000)
    endpoint_seed = mul(maximum(checkpoint_seed, finish_seed), valid)

    # Backward finite-horizon tables. Each table contains nine action values.
    action_tables = []
    finite_value = endpoint_seed
    for horizon in range(6):
        values = successors(finite_value, endpoint_seed)
        action_tables.append(cat(values))
        if horizon < 5:
            finite_value = reduce_successors(values)

    # Deterministic winner selection for nine integer-valued action scores.
    # Ties use the listed command order, with no-op preferred at an endpoint.
    compare_w = np.zeros((81, 9), np.float32)
    compare_b = np.zeros(81, np.float32)
    win_w = np.zeros((9, 81), np.float32)
    for i in range(9):
        for j in range(9):
            q = 9 * i + j
            if i != j:
                compare_w[q, j] = 1
                compare_w[q, i] = -1
                compare_b[q] = 0.25 if j < i else -0.25
                win_w[i, q] = -4
            else:
                compare_b[q] = -1

    score_reduce = np.zeros((9, 36), np.float32)
    repeat_scale = np.zeros((36, 4), np.float32)
    for a in range(9):
        score_reduce[a, 4 * a:4 * a + 4] = 1
        repeat_scale[4 * a:4 * a + 4, :] = np.eye(4)

    motion_w = np.asarray([
        [0, 0, 0, 4, -4, 0, 0, 0, 0],
        [0, -4, 4, 0, 0, 8, -8, 0, 0],
    ], np.float32)
    grow_transition = up_scale_values.T.copy()
    shrink_transition = down_scale_values.T.copy()
    grow_transition[:, 3] = 0
    grow_transition[3, 3] = 1
    shrink_transition[:, 0] = 0
    shrink_transition[0, 0] = 1

    xy = player_xy
    scale_state = player_scale
    desired_program = affine(one, [[0]])

    first_indicators = []
    active_indicators = []
    for step in range(6):
        f = np.zeros((1, 6), np.float32)
        f[0, 5 - step] = 1
        first_indicators.append(dense(length_onehot, f))
        a = np.zeros((1, 6), np.float32)
        a[0, 5 - step:] = 1
        active_indicators.append(dense(length_onehot, a))

    previous_first = const([0])
    for step in range(6):
        ghost = mul(point_mask(xy), scale_state)
        repeated_ghost = affine(ghost, repeat_scale)
        local = mul(action_tables[5 - step], repeated_ghost)
        scores = dense(node("gmax", [local]), score_reduce)
        violations = relu(dense(scores, compare_w, compare_b), vector=True)
        winner = relu(dense(violations, win_w, np.ones(9, np.float32)),
                      vector=True)

        first = first_indicators[step]
        rotate_here = mul(first, need_rotation, vector=True)
        colour_here = add(
            mul(first, colour_without_rotation, vector=True),
            mul(previous_first, colour_after_rotation, vector=True),
            vector=True)
        navigation_here = dense(
            cat([active_indicators[step], rotate_here, colour_here], vector=True),
            [[1, -1, -1]])
        selected = mul(winner, navigation_here, vector=True)

        xy = add(xy, dense(selected, motion_w), vector=True)

        grow_choice = sl(selected, 7, 8, vector=True)
        shrink_choice = sl(selected, 8, 9, vector=True)
        stay_choice = dense(selected, [[0, 0, 0, 0, 0, 0, 0, -1, -1]], [1])
        grown = mul(dense(scale_state, grow_transition),
                    grow_choice, vector=True)
        shrunk = mul(dense(scale_state, shrink_transition),
                     shrink_choice, vector=True)
        stayed = mul(scale_state, stay_choice, vector=True)
        scale_state = add(add(grown, shrunk, vector=True), stayed, vector=True)

        command_bits = dense(selected, bit_matrix(nav_codes))
        command_bits = add(
            command_bits, mul(rotation_bits, rotate_here, vector=True),
            vector=True)
        command_bits = add(
            command_bits, mul(colour_bits, colour_here, vector=True),
            vector=True)

        # Leading inactive planning steps are omitted from shorter programs.
        column_difference = affine(column_rank, [[1]], [5 - step])
        column_difference = node("sub", [column_difference, number_columns])
        column_difference = node("abs", [column_difference])
        selected_column = mul(
            relu(affine(column_difference, [[-1]], [1])), columns)

        row_values = affine(
            mul(bit_rows, command_bits), np.ones((1, 6), np.float32))
        desired_program = add(
            desired_program, mul(row_values, selected_column))
        previous_first = first

    differences = node("abs", [node("sub", [desired_program, current_bits])])

    # Recognize the filled execution button, not the hollow preset buttons.
    button_w = np.zeros((1, 16, 7, 7), np.float32)
    button_w[0, 9, :, :] = 1
    button = relu(conv("x0", button_w, [-48.5]))
    button = affine(button, [[2]])

    lower = relu(affine("coords", [[63, 0, 0]], [-50]))
    lower = node("sub", [lower, relu(affine(lower, [[1]], [-1]))])
    button = mul(button, lower)

    conv(cat([differences, button]),
         np.asarray([[[[100]], [[1]]]], np.float32), out="click")
    dense("avail", np.zeros((8, 8), np.float32),
          [-10, -10, -10, -10, -10, -10, 10, -10], out="logits")

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }
