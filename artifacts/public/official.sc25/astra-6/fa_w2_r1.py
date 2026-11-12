import numpy as np


def build():
    params = {}
    cache = {}
    graph = []
    G = graph
    serial = [0]

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p%d" % len(params)
            cache[key] = name
            params[name] = a
        return cache[key]

    def node(op, *inputs, **fields):
        name = fields.pop("name", None)
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(fields)
        G.append(d)
        return name

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", x, **fields)

    def lin(x, w, b=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        if name is not None:
            fields["name"] = name
        return node("dense", x, **fields)

    def sl(x, a, b):
        return node("slice", x, start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", x, start=a, stop=b)

    def cat(*xs):
        return node("concat", *xs)

    def vcat(*xs):
        return node("vconcat", *xs)

    def relu(x):
        return node("relu", x)

    def vr(x):
        return node("vrelu", x)

    def affine(x, n, a=1.0, b=0.0):
        return lin(x, np.eye(n, dtype=np.float32) * a,
                   np.full(n, b, np.float32))

    def va(x, n, a=1.0, b=0.0):
        return dense(x, np.eye(n, dtype=np.float32) * a,
                     np.full(n, b, np.float32))

    def clip01(x, n=1):
        return affine(relu(affine(x, n, -1, 1)), n, -1, 1)

    def vmax(a, b):
        return node("vadd", b, vr(node("vadd", a, va(b, 1, -1))))

    def vector_max(x, n):
        z = vs(x, 0, 1)
        for i in range(1, n):
            z = vmax(z, vs(x, i, i + 1))
        return z

    def channel_max(x, n):
        z = sl(x, 0, 1)
        for i in range(1, n):
            z = node("max2", z, sl(x, i, i + 1))
        return z

    def shift1(x, dy, dx):
        k = 2 * max(abs(dy), abs(dx)) + 1
        w = np.zeros((1, 1, k, k), np.float32)
        w[0, 0, k // 2 + dy, k // 2 + dx] = 1
        return conv(x, w)

    colors = {}

    def color(c):
        if c not in colors:
            colors[c] = sl("x0", c, c + 1)
        return colors[c]

    def color_sum(cs):
        w = np.zeros((1, 16), np.float32)
        w[0, cs] = 1
        return lin("x0", w)

    one = sl("coords", 2, 3)
    zero = lin(one, [[0]])

    def rectangle(xlo, xhi, ylo, yhi):
        w = np.array([[0, -63, 0], [0, 63, 0],
                      [-63, 0, 0], [63, 0, 0]], np.float32)
        b = np.array([xlo - .25, -xhi - .25,
                      ylo - .25, -yhi - .25], np.float32)
        violations = relu(lin("coords", w, b))
        return relu(lin(violations, -4 * np.ones((1, 4)), [1]))

    menu = node("max2", rectangle(0, 9, 0, 32),
                rectangle(11, 20, 50, 59))
    entry_region = rectangle(22, 38, 47, 63)
    nonworld = node("max2", menu, entry_region)
    world = affine(nonworld, 1, -1, 1)

    spell_colors = cat(color(6), color(11), color(15))
    spell_flags = node("gmax", node("mul", spell_colors, menu))
    fire_flag = vs(spell_flags, 0, 1)
    teleport_flag = vs(spell_flags, 1, 2)
    resize_flag = vs(spell_flags, 2, 3)

    # Exit recognition, independent of player recognition.
    # Each detector is centered at exit top-left + (2, 2).
    goal_template = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ])
    w = np.zeros((4, 2, 7, 7), np.float32)
    paint = np.zeros((1, 4, 7, 7), np.float32)
    for r in range(4):
        template = np.rot90(goal_template, -r)
        h, ww = template.shape
        for y in range(h):
            for x in range(ww):
                w[r, int(template[y, x] == 10), y + 1, x + 1] = 1
                paint[0, r, 5 - y, 5 - x] = 1
    goal_centers = relu(conv(cat(color(9), color(10)), w, [-29] * 4))
    goal_pixels = clip01(conv(goal_centers, paint))

    clean9 = relu(node("sub", color(9), goal_pixels))
    clean10 = relu(node("sub", color(10), goal_pixels))
    player_colors = cat(clean9, clean10)

    # Eight exact player detectors: two sizes times four directions.
    w = np.zeros((8, 2, 7, 7), np.float32)
    b = np.zeros(8, np.float32)
    for si, size in enumerate((2, 4)):
        for direction in range(4):
            o = 4 * si + direction
            for y in range(size):
                for x in range(size):
                    if direction == 0:
                        is10 = y >= size // 2
                    elif direction == 1:
                        is10 = y < size // 2
                    elif direction == 2:
                        is10 = x >= size // 2
                    else:
                        is10 = x < size // 2
                    w[o, int(is10), y + 2, x + 2] = 1
            for y in range(-1, size + 1):
                for x in range(-1, size + 1):
                    if 0 <= y < size and 0 <= x < size:
                        continue
                    w[o, :, y + 2, x + 2] = -1
            b[o] = 1 - size * size
    pd_center = relu(conv(player_colors, w, b))
    w = np.zeros((8, 8, 3, 3), np.float32)
    for i in range(8):
        w[i, i, 2, 2] = 1
    pd = conv(pd_center, w)

    w = np.zeros((2, 8), np.float32)
    w[0, :4] = 1
    w[1, 4:] = 1
    anchors = lin(pd, w)
    anchor_small = sl(anchors, 0, 1)
    anchor_big = sl(anchors, 1, 2)
    anchor_any = lin(anchors, [[1, 1]])
    size_flags = node("gmax", anchors)
    facing = dense(node("gmax", pd),
                   np.concatenate([np.eye(4), np.eye(4)], axis=1))

    # Teleport destinations: active outer corners, inactive inner corners,
    # and small filled pads.
    corner6 = np.zeros((1, 1, 7, 7), np.float32)
    for y in (-2, 3):
        for x in (-2, 3):
            corner6[0, 0, y + 3, x + 3] = 1

    outer_big_center = relu(conv(color(11), corner6, [-3]))
    outer_big = node("mul", shift1(outer_big_center, 1, 1), world)
    outer_big_present = node("gmax", outer_big)

    pad_features = cat(color(11), color(15), color(9), color(10))
    w = np.zeros((1, 4, 7, 7), np.float32)
    for y in (0, 3):
        for x in (0, 3):
            w[0, 0, y + 3, x + 3] = 1
    for y in (1, 2):
        for x in (1, 2):
            w[0, 1:, y + 3, x + 3] = -1
    inner_big = node("mul", relu(conv(pad_features, w, [-3])), world)
    inner_big = node("mul", inner_big, va(outer_big_present, 1, -1, 1))
    pad_big = node("max2", outer_big, inner_big)

    w = np.zeros((1, 1, 3, 3), np.float32)
    w[0, 0, 1:3, 1:3] = 1
    small_block = relu(conv(color(15), w, [-3]))
    small_outer = relu(conv(color(15), corner6, [-3]))
    pad_small = node("mul", node("max2", small_block, small_outer), world)

    # Keys and the doors removed by their corresponding fire hit.
    dilation = np.ones((1, 1, 3, 3), np.float32)
    adjacent13 = clip01(conv(color(13), dilation))
    adjacent12 = clip01(conv(color(12), dilation))
    adjacent6 = clip01(conv(color(6), dilation))
    red_key = node("mul", node("mul", color(6), adjacent13), world)
    cyan_key = node("mul", node("mul", color(6), adjacent12), world)
    red_core = node("mul", color(13), adjacent6)
    cyan_core = node("mul", color(12), adjacent6)

    red_group = node("add", color(13), red_key)
    cyan_group = node("add", color(12), cyan_key)
    red_door = relu(node("sub", color(13), red_core))
    cyan_door = relu(node("sub", color(12), cyan_core))

    common = node("add", color_sum([0, 1, 3, 4, 5, 7, 8]), color(6))
    common = node("sub", node("sub", common, red_key), cyan_key)
    ray_common = node("sub", common, color(4))

    # Four hypothetical door states, each with two player sizes.
    remaining = np.array([[1, 1, 1], [1, 0, 1],
                          [1, 1, 0], [1, 0, 0]], np.float32)
    obstacle4 = lin(cat(common, red_group, cyan_group), remaining)
    obstacle4 = node("add", obstacle4, goal_pixels)
    growth_obstacle4 = lin(cat(ray_common, red_door, cyan_door), remaining)

    def free_anchors(obstacles, size):
        x = cat(obstacles, one)
        w = np.zeros((4, 5, 7, 7), np.float32)
        for c in range(4):
            for y in range(size):
                for xx in range(size):
                    w[c, c, 3 + y, 3 + xx] = -1
                    w[c, 4, 3 + y, 3 + xx] = 1
        return relu(conv(x, w, np.full(4, 1 - size * size)))

    free_small4 = free_anchors(obstacle4, 2)
    free_big4 = free_anchors(obstacle4, 4)
    growth_free4 = free_anchors(growth_obstacle4, 4)

    free_parts = []
    for state in range(4):
        free_parts.extend([sl(free_small4, state, state + 1),
                           sl(free_big4, state, state + 1)])
    free8 = cat(*free_parts)

    w = np.zeros((2, 1, 7, 7), np.float32)
    for si, size in enumerate((2, 4)):
        w[si, 0, 3:3 + size, 3:3 + size] = 1
    goal_contact = clip01(conv(goal_pixels, w), 2)
    gs, gb = sl(goal_contact, 0, 1), sl(goal_contact, 1, 2)
    goal_seed = affine(cat(*([gs, gb] * 4)), 8, 128)

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    step2 = np.zeros((4, 1, 5, 5), np.float32)
    step2_again = np.zeros((4, 4, 5, 5), np.float32)
    ray_step = np.zeros((4, 4, 3, 3), np.float32)
    for d, (dy, dx) in enumerate(dirs):
        step2[d, 0, 2 + 2 * dy, 2 + 2 * dx] = 1
        step2_again[d, d, 2 + 2 * dy, 2 + 2 * dx] = 1
        ray_step[d, d, 1 + dy, 1 + dx] = 1

    # Large movement first tries four pixels, then falls back to two.
    prefer4, fallback2 = [], []
    for state in range(4):
        f = conv(conv(free_parts[2 * state + 1], step2), step2_again)
        prefer4.append(f)
        fallback2.append(affine(f, 4, -1, 1))

    shift_nw = np.zeros((1, 1, 3, 3), np.float32)
    shift_nw[0, 0, 0, 0] = 1
    neighborhood9 = np.zeros((9, 1, 3, 3), np.float32)
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for i, (dx, dy) in enumerate(offsets):
        neighborhood9[i, 0, dy + 1, dx + 1] = 1

    def growth_neighborhood(x):
        return conv(conv(x, shift_nw), neighborhood9)

    first_matrix = np.eye(9, dtype=np.float32)
    for i in range(9):
        first_matrix[i, :i] = -1
    growth_choices = []
    for state in range(4):
        f = growth_neighborhood(sl(growth_free4, state, state + 1))
        growth_choices.append(relu(lin(f, first_matrix)))
    growth_choices36 = cat(*growth_choices)

    # Directional fire visibility, including the hypothetical door states.
    ray_boards = lin(
        cat(ray_common, red_group, cyan_group),
        [[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1]])
    ray_free = relu(affine(ray_boards, 4, -1, 1))
    ray_free_parts = [sl(ray_free, i, i + 1) for i in range(4)]
    ray_seed_red = lin(red_key, np.ones((4, 1)))
    ray_seed_cyan = lin(cyan_key, np.ones((4, 1)))
    ray_seeds = [ray_seed_red, ray_seed_cyan,
                 ray_seed_red, ray_seed_cyan]

    ray_body = []
    old_G = G
    G = ray_body
    ray_next = {}
    for i in range(4):
        s = "ray_state%d" % i
        moved = conv(s, ray_step)
        continued = node("mul", moved, ray_free_parts[i])
        ray_next[s] = node("max2", continued, ray_seeds[i])
    G = old_G
    G.append({
        "id": "visibility",
        "op": "loop",
        "T": 64,
        "state": [{"id": "ray_state%d" % i, "init": ray_seeds[i]}
                  for i in range(4)],
        "body": ray_body,
        "next": ray_next,
    })
    rays = ["visibility.ray_state%d" % i for i in range(4)]
    any_ray = [channel_max(r, 4) for r in rays]

    red_gate = cat(any_ray[0], any_ray[0], zero, zero,
                   any_ray[2], any_ray[2], zero, zero)
    cyan_gate = cat(any_ray[1], any_ray[1], any_ray[3], any_ray[3],
                    zero, zero, zero, zero)
    red_gate = node("mul", red_gate, fire_flag)
    cyan_gate = node("mul", cyan_gate, fire_flag)
    teleport_targets8 = node(
        "mul", cat(*([pad_small, pad_big] * 4)), teleport_flag)

    red_transition = np.zeros((8, 8), np.float32)
    cyan_transition = np.zeros((8, 8), np.float32)
    for state in (0, 2):
        for size in range(2):
            red_transition[2 * state + size,
                           2 * (state | 1) + size] = 1
    for state in (0, 1):
        for size in range(2):
            cyan_transition[2 * state + size,
                            2 * (state | 2) + size] = 1

    def grouped_max4(x):
        sum1 = np.zeros((16, 32), np.float32)
        diff1 = np.zeros((16, 32), np.float32)
        for i in range(16):
            sum1[i, 2 * i:2 * i + 2] = 1
            diff1[i, 2 * i] = 1
            diff1[i, 2 * i + 1] = -1
        z = node("add", lin(x, sum1), node("abs", lin(x, diff1)))
        sum2 = np.zeros((8, 16), np.float32)
        diff2 = np.zeros((8, 16), np.float32)
        for i in range(8):
            sum2[i, 2 * i:2 * i + 2] = 1
            diff2[i, 2 * i] = 1
            diff2[i, 2 * i + 1] = -1
        z = node("add", lin(z, sum2), node("abs", lin(z, diff2)))
        return affine(z, 8, .25)

    def movement_fields(parts):
        result = []
        for state in range(4):
            result.append(conv(parts[2 * state], step2))
            q2 = conv(parts[2 * state + 1], step2)
            q4 = conv(q2, step2_again)
            result.append(node(
                "add", node("mul", q4, prefer4[state]),
                node("mul", q2, fallback2[state])))
        return result

    collapse_growth = np.zeros((4, 36), np.float32)
    select_small = np.zeros((4, 8), np.float32)
    scale_reorder = np.zeros((8, 8), np.float32)
    for state in range(4):
        collapse_growth[state, 9 * state:9 * state + 9] = 1
        select_small[state, 2 * state] = 1
        scale_reorder[2 * state, state] = 1
        scale_reorder[2 * state + 1, 4 + state] = 1

    # Max-plus value propagation over movement and spell transitions.
    planning_body = []
    old_G = G
    G = planning_body
    vp = [sl("value_state", i, i + 1) for i in range(8)]
    movement = grouped_max4(cat(*movement_fields(vp)))
    value = affine(movement, 8, 1, -1)

    grow_fields = [growth_neighborhood(vp[2 * s + 1]) for s in range(4)]
    grow_values = lin(
        node("mul", cat(*grow_fields), growth_choices36), collapse_growth)
    shrink_values = lin("value_state", select_small)
    scale_values = lin(cat(grow_values, shrink_values),
                       scale_reorder, np.full(8, -4))
    scale_values = node("mul", scale_values, resize_flag)
    value = node("max2", value, scale_values)

    # Reserve two units beyond the three clicks for future fire alignment.
    # The action head removes that reserve when the player already faces
    # the target; thus it does not walk away from an executable fire spell.
    red_values = node(
        "mul", lin("value_state", red_transition, np.full(8, -5)), red_gate)
    cyan_values = node(
        "mul", lin("value_state", cyan_transition, np.full(8, -5)), cyan_gate)
    value = node("max2", node("max2", value, red_values), cyan_values)

    teleport_values = node(
        "gmax", node("mul", "value_state", teleport_targets8))
    teleport_values = va(teleport_values, 8, 1, -3)
    value = node("max2", value, teleport_values)
    value = node("mul", relu(value), free8)
    value = node("max2", value, goal_seed)
    G = old_G

    planning_loop = {
        "id": "planning",
        "op": "loop",
        "T": 64,
        "state": [{"id": "value_state", "init": goal_seed}],
        "body": planning_body,
        "next": {"value_state": value},
    }
    G.append(planning_loop)
    vp = [sl("planning.value_state", i, i + 1) for i in range(8)]

    def at_anchor(field, anchor):
        return node("gmax", node("mul", field, anchor))

    small_moves = affine(conv(vp[0], step2), 4, 1, -1)
    big2 = conv(vp[1], step2)
    big4 = conv(big2, step2_again)
    big_moves = node("add", node("mul", big4, prefer4[0]),
                     node("mul", big2, fallback2[0]))
    big_moves = affine(big_moves, 4, 1, -1)
    move_q = node("vadd", at_anchor(small_moves, anchor_small),
                  at_anchor(big_moves, anchor_big))

    red_future = node("vadd", at_anchor(vp[2], anchor_small),
                      at_anchor(vp[3], anchor_big))
    cyan_future = node("vadd", at_anchor(vp[4], anchor_small),
                       at_anchor(vp[5], anchor_big))
    red_future = vr(va(red_future, 1, 1, -3))
    cyan_future = vr(va(cyan_future, 1, 1, -3))
    red_los = node("vmul", at_anchor(rays[0], anchor_any), fire_flag)
    cyan_los = node("vmul", at_anchor(rays[1], anchor_any), fire_flag)
    red_fire = node("vmul", red_los, red_future)
    cyan_fire = node("vmul", cyan_los, cyan_future)
    fire_q = node(
        "vadd", cyan_fire,
        vr(node("vadd", red_fire, va(cyan_fire, 4, -1))))

    firing_now = node("vmul", fire_q, facing)
    turn_q = node("vmul", fire_q, va(facing, 4, -1, 1))
    turn_q = vr(va(turn_q, 4, 1, -.9))
    move_q = node("vadd", move_q,
                  vr(node("vadd", turn_q, va(move_q, 4, -1))))
    fire_click_q = vector_max(firing_now, 4)

    ts = node("gmax", node("mul", vp[0], pad_small))
    tb = node("gmax", node("mul", vp[1], pad_big))
    teleport_q = node("vmul", vcat(ts, tb), size_flags)
    teleport_q = vr(dense(teleport_q, [[1, 1]], [-3]))
    teleport_q = node("vmul", teleport_q, teleport_flag)

    grown = lin(node("mul", growth_neighborhood(vp[1]), growth_choices[0]),
                np.ones((1, 9)))
    resize_q = node("vadd", at_anchor(grown, anchor_small),
                     at_anchor(vp[0], anchor_big))
    resize_q = node("vmul", vr(va(resize_q, 1, 1, -4)), resize_flag)

    spell_q = dense(vcat(fire_click_q, teleport_q, resize_q),
                    np.eye(3), [.003, .002, .001])
    click_q = vector_max(spell_q, 3)

    # Winner gates for fire, teleport, and resize.
    pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    w = np.zeros((6, 3), np.float32)
    for k, (i, j) in enumerate(pairs):
        w[k, i] = 10000
        w[k, j] = -10000
    comparisons = vr(dense(spell_q, w))
    comparisons = node(
        "vadd", comparisons, va(vr(va(comparisons, 6, 1, -1)), 6, -1))
    w = np.zeros((3, 6), np.float32)
    for k, (i, j) in enumerate(pairs):
        w[i, k] = 1
    chosen_spell = vr(dense(comparisons, w, [-1, -1, -1]))

    # Point masks for the nine entry cells.
    points = [(25 + 5 * x, 50 + 5 * y)
              for y in range(3) for x in range(3)]
    w = np.zeros((18, 3), np.float32)
    b = np.zeros(18, np.float32)
    for i, (x, y) in enumerate(points):
        w[2 * i, 0] = 63
        b[2 * i] = -y
        w[2 * i + 1, 1] = 63
        b[2 * i + 1] = -x
    distances = node("abs", lin("coords", w, b))
    w = np.zeros((9, 18), np.float32)
    for i in range(9):
        w[i, 2 * i:2 * i + 2] = -2
    point_masks = relu(lin(distances, w, np.ones(9)))

    selected_color = color_sum([14, 6, 11, 15])
    selected = node("gmax", node("mul", point_masks, selected_color))
    selected = vr(dense(selected, 4 * np.eye(9), np.full(9, -1)))
    selected = node(
        "vadd", selected, va(vr(va(selected, 9, 1, -1)), 9, -1))

    patterns = np.zeros((3, 9), np.float32)
    patterns[0, [1, 4, 7]] = 1
    patterns[1, [0, 1, 4]] = 1
    patterns[2, [1, 3, 5, 7]] = 1
    desired = dense(chosen_spell, patterns.T)
    delta = node("vadd", selected, va(desired, 9, -1))
    extras = vr(delta)
    differences = node("vadd", extras, vr(va(delta, 9, -1)))

    # Detect toggles that would complete an unintended spell.
    w = np.zeros((27, 9), np.float32)
    b = np.zeros(27, np.float32)
    for spell in range(3):
        p = patterns[spell]
        base_w = 1 - 2 * p
        base_b = float(p.sum())
        for bit in range(9):
            k = 9 * spell + bit
            mismatch_w = base_w.copy()
            mismatch_w[bit] *= -1
            mismatch_b = base_b + 1 - 2 * p[bit]
            w[k] = -mismatch_w
            b[k] = 1 - mismatch_b
    accidental = vr(dense(selected, w, b))
    w = np.zeros((27, 3), np.float32)
    for spell in range(3):
        w[9 * spell:9 * spell + 9, spell] = -1
    other_spell = dense(chosen_spell, w, np.ones(27))
    accidental = node("vmul", accidental, other_spell)
    w = np.zeros((9, 27), np.float32)
    for bit in range(9):
        for spell in range(3):
            w[bit, 9 * spell + bit] = 1
    forbidden = dense(accidental, w)

    click_scores = node("vadd", va(differences, 9, 100),
                        va(extras, 9, 10))
    click_scores = node("vadd", click_scores, va(forbidden, 9, -500))
    click_scores = dense(
        click_scores, np.eye(9), -np.arange(9, dtype=np.float32) * .01)
    click_field = node("mul", point_masks, click_scores)
    click_sum = lin(click_field, np.ones((1, 9)))
    node("add", click_sum, zero, name="click")

    action_features = vcat(move_q, click_q)
    w = np.zeros((8, 5), np.float32)
    for i in range(4):
        w[i + 1, i] = 1
    w[6, 4] = 1
    b = np.array([-1000, .04, .03, .02, .01, -1000, .10, -1000],
                 np.float32)
    dense(action_features, w, b, name="logits")

    # Static resource accounting only; no game execution or weight fitting.
    loop_body_costs = {}

    def account(nodes, shapes):
        macs = 0
        evaluations = 0
        node_count = 0
        channels = 0
        for nd in nodes:
            op = nd["op"]
            if op == "loop":
                local = dict(shapes)
                carried_channels = 0
                for s in nd["state"]:
                    shape = (shapes[s["init"]] if "init" in s
                             else tuple(s["zeros"]))
                    local[s["id"]] = shape
                    if len(shape) == 3:
                        carried_channels += shape[0]
                bm, be, bn, bc = account(nd["body"], local)
                loop_body_costs[nd["id"]] = (bm, be)
                macs += nd["T"] * bm
                evaluations += 1 + nd["T"] * be
                node_count += 1 + bn
                channels += carried_channels + bc
                for s in nd["state"]:
                    shapes[nd["id"] + "." + s["id"]] = local[
                        nd["next"][s["id"]]]
                continue

            ins = [shapes[i] for i in nd["in"]]
            x = ins[0]
            cost = 0
            if op == "conv":
                ww = params[nd["w"]]
                shape = (ww.shape[0], 64, 64)
                cost = int(np.prod(ww.shape)) * 4096
            elif op == "dense":
                ww = params[nd["w"]]
                shape = (ww.shape[0],)
                cost = int(np.prod(ww.shape))
            elif op in ("slice", "vslice"):
                shape = (nd["stop"] - nd["start"],) + x[1:]
            elif op in ("concat", "vconcat"):
                shape = (sum(a[0] for a in ins),) + x[1:]
            elif op in ("gmax", "gmean"):
                shape = (x[0],)
                cost = x[0] * 4096
            elif op in ("rowmax", "colmax", "rowsum", "colsum"):
                shape = x
                cost = x[0] * 4096
            elif op in ("add", "sub", "mul", "max2", "min2"):
                shape = (max(a[0] for a in ins), 64, 64)
                cost = shape[0] * 4096
            elif op in ("vadd", "vmul"):
                shape = (max(a[0] for a in ins),)
                cost = shape[0]
            else:
                shape = x
            shapes[nd["id"]] = shape
            macs += cost
            evaluations += 1
            node_count += 1
            if len(shape) == 3:
                channels += shape[0]
        return macs, evaluations, node_count, channels

    planning_loop["T"] = 1
    initial_shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (0,),
        "aprev": (8,),
        "avail": (8,),
    }
    mac1, ev1, count, channels = account(graph, dict(initial_shapes))
    body_mac, body_ev = loop_body_costs["planning"]
    by_mac = 1 + int((1_490_000_000 - mac1) // body_mac)
    by_evaluations = 1 + int((9900 - ev1) // body_ev)
    planning_loop["T"] = max(1, min(64, by_mac, by_evaluations))

    return {"graph": graph, "memory_size": 0, "params": params}
