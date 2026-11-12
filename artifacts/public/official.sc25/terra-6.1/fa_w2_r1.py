import numpy as np


def build():
    graph = []
    params = {}
    node_counter = [0]
    param_counter = [0]
    current = [graph]

    def node(op, inputs=None, name=None, **fields):
        if name is None:
            node_counter[0] += 1
            name = "n%d" % node_counter[0]
        result = {"id": name, "op": op}
        if inputs is not None:
            result["in"] = list(inputs)
        result.update(fields)
        current[0].append(result)
        return name

    def parameter(value):
        param_counter[0] += 1
        name = "p%d" % param_counter[0]
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def conv(x, weights, bias=None, name=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("conv", [x], name=name, **fields)

    def dense(x, weights, bias=None, name=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("dense", [x], name=name, **fields)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def sp_sum(x, channels):
        return conv(x, np.ones((1, channels, 1, 1), dtype=np.float32))

    def vneg(x, n, bias=0.0):
        return dense(
            x,
            -np.eye(n, dtype=np.float32),
            np.full(n, bias, dtype=np.float32),
        )

    def vsub(a, b, n):
        return node("vadd", [a, vneg(b, n)])

    def vclip(x, n):
        x = node("vrelu", [x])
        x = vneg(x, n, 1.0)
        x = node("vrelu", [x])
        return vneg(x, n, 1.0)

    def vstep(x, n):
        x = dense(x, 100.0 * np.eye(n, dtype=np.float32))
        return vclip(x, n)

    one = node("gmax", [sl("coords", 2, 3)])

    def sclip(x):
        x = node("relu", [x])
        x = node("sub", [one, x])
        x = node("relu", [x])
        return node("sub", [one, x])

    def soneminus(x):
        return node("sub", [one, x])

    def scalar_one_minus(x):
        return vneg(x, 1, 1.0)

    def shift2(x, dy, dx):
        w = np.zeros((1, 1, 5, 5), dtype=np.float32)
        w[0, 0, 2 + 2 * dy, 2 + 2 * dx] = 1.0
        return conv(x, w)

    # Door detection and reconstruction in all four orientations.
    door_base = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ], dtype=np.int32)
    doors = [np.rot90(door_base, -k) for k in range(4)]

    w = np.zeros((4, 16, 7, 7), dtype=np.float32)
    for k, template in enumerate(doors):
        for r in range(template.shape[0]):
            for c in range(template.shape[1]):
                expected = int(template[r, c])
                other = 10 if expected == 9 else 9
                w[k, expected, r, c] = 2.0
                w[k, other, r, c] = -2.0
    door_points = node("relu", [
        conv("x0", w, np.full(4, -59.0, dtype=np.float32))
    ])

    w = np.zeros((1, 4, 7, 7), dtype=np.float32)
    for k, template in enumerate(doors):
        for r in range(template.shape[0]):
            for c in range(template.shape[1]):
                w[0, k, 6 - r, 6 - c] = 1.0
    door = sclip(conv(door_points, w))

    # Player detection. The anchor is its top-left plus (1, 1).
    colors_9_10 = sl("x0", 9, 11)
    player_colors = node("relu", [node("sub", [colors_9_10, door])])
    player_base = np.array([[10, 10], [9, 9]], dtype=np.int32)

    w = np.zeros((8, 2, 7, 7), dtype=np.float32)
    biases = np.zeros(8, dtype=np.float32)
    for size_index, scale in enumerate((1, 2)):
        base = np.repeat(np.repeat(player_base, scale, axis=0), scale, axis=1)
        for k in range(4):
            template = np.rot90(base, -k)
            out = 4 * size_index + k
            h, width = template.shape
            for r in range(-1, h + 1):
                for c in range(-1, width + 1):
                    kr, kc = r + 2, c + 2
                    if 0 <= r < h and 0 <= c < width:
                        expected = int(template[r, c]) - 9
                        w[out, expected, kr, kc] = 2.0
                        w[out, 1 - expected, kr, kc] = -2.0
                    else:
                        w[out, 0, kr, kc] = -2.0
                        w[out, 1, kr, kc] = -2.0
            biases[out] = 1.0 - 2.0 * h * width
    oriented = node("relu", [conv(player_colors, w, biases)])

    w = np.zeros((2, 8, 1, 1), dtype=np.float32)
    w[0, :4, 0, 0] = 1.0
    w[1, 4:, 0, 0] = 1.0
    by_size = conv(oriented, w)
    small_map = sl(by_size, 0, 1)
    big_map = sl(by_size, 1, 2)
    player = node("max2", [small_map, big_map])
    small = node("gmax", [small_map])
    big = node("gmax", [big_map])

    # Direction order: up, down, left, right.
    rotation_for_direction = [2, 0, 1, 3]
    w = np.zeros((4, 8, 1, 1), dtype=np.float32)
    for d, k in enumerate(rotation_for_direction):
        w[d, k, 0, 0] = 1.0
        w[d, 4 + k, 0, 0] = 1.0
    player_directions = conv(oriented, w)

    w = np.zeros((1, 16, 1, 1), dtype=np.float32)
    for color in (3, 4, 5, 6, 12, 13):
        w[0, color, 0, 0] = 1.0
    wall = sclip(node("add", [conv("x0", w), door]))

    def valid_for_size(size):
        w = np.zeros((1, 1, 7, 7), dtype=np.float32)
        for r in range(size):
            for c in range(size):
                w[0, 0, r + 2, c + 2] = 1.0
        occupied = conv(wall, w)
        free = node("relu", [soneminus(occupied)])

        w = np.zeros((4, 3, 1, 1), dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        w[0, 0, 0, 0] = 126.0
        b[0] = -1.0
        w[1, 0, 0, 0] = -126.0
        b[1] = 2.0 * (65 - size) + 1.0
        w[2, 1, 0, 0] = 126.0
        b[2] = -1.0
        w[3, 1, 0, 0] = -126.0
        b[3] = 2.0 * (65 - size) + 1.0
        bounds = sclip(conv("coords", w, b))
        for i in range(4):
            free = node("mul", [free, sl(bounds, i, i + 1)])
        return free

    valid_small = valid_for_size(2)
    valid_big = valid_for_size(4)
    valid = node("add", [
        node("mul", [valid_small, small]),
        node("mul", [valid_big, big]),
    ])

    def door_overlap(size):
        w = np.zeros((1, 1, 7, 7), dtype=np.float32)
        for r in range(size):
            for c in range(size):
                w[0, 0, r + 2, c + 2] = 1.0
        return sclip(conv(door, w))

    overlap = node("add", [
        node("mul", [door_overlap(2), small]),
        node("mul", [door_overlap(4), big]),
    ])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    gates = []
    goal_hits = []
    for dy, dx in directions:
        valid4 = shift2(shift2(valid, dy, dx), dy, dx)
        gate = node("mul", [valid4, big])
        gates.append(gate)
        hit2 = shift2(overlap, dy, dx)
        hit = node("mul", [
            valid,
            node("mul", [hit2, soneminus(gate)]),
        ])
        goal_hits.append(hit)
    goal_dirs = node("concat", goal_hits)
    goal_seed = sclip(sp_sum(goal_dirs, 4))

    # Normal and special destructible target detection.
    target_templates = []
    for center in (13, 12):
        template = np.full((4, 4), 6, dtype=np.int32)
        template[1:3, 1:3] = center
        target_templates.append(template)

    w = np.zeros((2, 16, 7, 7), dtype=np.float32)
    for k, template in enumerate(target_templates):
        for r in range(4):
            for c in range(4):
                w[k, int(template[r, c]), r + 2, c + 2] = 2.0
    target_points = node("relu", [
        conv("x0", w, np.full(2, -31.0, dtype=np.float32))
    ])
    target_presence = vclip(node("gmax", [target_points]), 2)

    w = np.zeros((1, 2, 7, 7), dtype=np.float32)
    for k in range(2):
        for r in range(4):
            for c in range(4):
                w[0, k, 4 - r, 4 - c] = 1.0
    target = sclip(conv(target_points, w))

    # Straight obstacle-limited firing rays.
    w = np.zeros((1, 16, 1, 1), dtype=np.float32)
    for color in (5, 6, 12, 13):
        w[0, color, 0, 0] = 1.0
    ray_pass = soneminus(sclip(conv("x0", w)))
    ray_init = conv(target, np.ones((4, 1, 1, 1), dtype=np.float32))

    ray_body = []
    current[0] = ray_body
    w = np.zeros((4, 4, 3, 3), dtype=np.float32)
    for d, (dy, dx) in enumerate(directions):
        w[d, d, 1 + dy, 1 + dx] = 1.0
    ray_shifted = conv("ray_state", w)
    ray_advanced = node("mul", [ray_shifted, ray_pass])
    ray_next = node("max2", ["ray_state", ray_advanced])
    current[0] = graph
    graph.append({
        "id": "ray_loop",
        "op": "loop",
        "T": 64,
        "state": [{"id": "ray_state", "init": ray_init}],
        "body": ray_body,
        "next": {"ray_state": ray_next},
    })
    rays = "ray_loop.ray_state"

    def ray_at_player_front(size):
        offsets = [
            (-2, -1),
            (size - 1, -1),
            (-1, -2),
            (-1, size - 1),
        ]
        w = np.zeros((4, 4, 7, 7), dtype=np.float32)
        for d, (dy, dx) in enumerate(offsets):
            w[d, d, 3 + dy, 3 + dx] = 1.0
        return conv(rays, w)

    shot_dirs = node("add", [
        node("mul", [ray_at_player_front(2), small]),
        node("mul", [ray_at_player_front(4), big]),
    ])
    shot_dirs = node("mul", [shot_dirs, valid])
    shot_seed = sclip(sp_sum(shot_dirs, 4))

    # Two simultaneous distance fields: the door and firing positions.
    seeds = node("concat", [goal_seed, shot_seed])

    def field_choices(field):
        w = np.zeros((8, 2, 5, 5), dtype=np.float32)
        for d, (dy, dx) in enumerate(directions):
            for c in range(2):
                w[2 * d + c, c, 2 + 2 * dy, 2 + 2 * dx] = 1.0
        two_steps = conv(field, w)
        choices = []
        for d, (dy, dx) in enumerate(directions):
            at2 = sl(two_steps, 2 * d, 2 * d + 2)
            w2 = np.zeros((2, 2, 5, 5), dtype=np.float32)
            for c in range(2):
                w2[c, c, 2 + 2 * dy, 2 + 2 * dx] = 1.0
            at4 = conv(at2, w2)
            choices.append(node("add", [
                node("mul", [at2, soneminus(gates[d])]),
                node("mul", [at4, gates[d]]),
            ]))
        return choices

    dist_body = []
    current[0] = dist_body
    choices = field_choices("dist_state")
    best = choices[0]
    for choice in choices[1:]:
        best = node("max2", [best, choice])
    w = np.eye(2, dtype=np.float32).reshape(2, 2, 1, 1)
    decayed = node("relu", [
        conv(best, w, np.full(2, -0.008, dtype=np.float32))
    ])
    reachable = node("mul", [decayed, valid])
    dist_next = node("max2", [seeds, reachable])
    current[0] = graph
    graph.append({
        "id": "dist_loop",
        "op": "loop",
        "T": 96,
        "state": [{"id": "dist_state", "init": seeds}],
        "body": dist_body,
        "next": {"dist_state": dist_next},
    })
    distances = "dist_loop.dist_state"
    next_choices = field_choices(distances)

    # Spell availability from glyphs adjacent to framed UI elements.
    frame3 = sl("x0", 3, 4)
    near_frame = sclip(
        conv(frame3, np.ones((1, 1, 7, 7), dtype=np.float32))
    )

    def icon_available(color):
        return vclip(node("gmax", [
            node("mul", [sl("x0", color, color + 1), near_frame])
        ]), 1)

    fire_available = icon_available(6)
    teleport_available = icon_available(11)
    resize_available = icon_available(15)

    values = node("gmax", [node("mul", [distances, player])])
    goal_present = vstep(vsl(values, 0, 1), 1)
    shot_present = vstep(vsl(values, 1, 2), 1)
    goal_mode = goal_present
    shot_mode = node("vmul", [
        scalar_one_minus(goal_mode),
        node("vmul", [shot_present, fire_available]),
    ])
    fallback = node("vmul", [
        scalar_one_minus(goal_mode),
        scalar_one_minus(shot_mode),
    ])

    facing_rays = node("gmax", [
        node("mul", [player_directions, shot_dirs])
    ])
    facing_any = vclip(
        dense(facing_rays, np.ones((1, 4), dtype=np.float32)), 1
    )
    cast_fire = node("vmul", [shot_mode, facing_any])

    # A destroyed target permits another large-player teleport.
    old_presence = vsl("mem", 2, 4)
    disappeared = node("vrelu", [vsub(old_presence, target_presence, 2)])
    any_disappeared = vclip(
        dense(disappeared, np.ones((1, 2), dtype=np.float32)), 1
    )
    old_big = vsl("mem", 0, 1)
    mem_big = node("vmul", [old_big, scalar_one_minus(any_disappeared)])
    mem_small = vsl("mem", 1, 2)

    unused_teleport = node("vadd", [
        node("vmul", [big, scalar_one_minus(mem_big)]),
        node("vmul", [small, scalar_one_minus(mem_small)]),
    ])
    cast_teleport = node("vmul", [
        fallback,
        node("vmul", [teleport_available, unused_teleport]),
    ])
    cast_resize = node("vmul", [
        fallback,
        node("vmul", [
            scalar_one_minus(cast_teleport),
            resize_available,
        ]),
    ])
    cast_total = vclip(node("vadd", [
        node("vadd", [cast_fire, cast_teleport]),
        cast_resize,
    ]), 1)

    modes = node("vconcat", [goal_mode, shot_mode])
    move_values = []
    for choice in next_choices:
        at_player = node("gmax", [node("mul", [choice, player])])
        selected = node("vmul", [at_player, modes])
        move_values.append(dense(
            selected, np.ones((1, 2), dtype=np.float32)
        ))
    moves = node("vconcat", move_values)
    hit_values = node("gmax", [node("mul", [goal_dirs, player])])
    shot_values = node("gmax", [node("mul", [shot_dirs, player])])
    moves = node("vadd", [
        moves,
        dense(
            node("vmul", [hit_values, goal_mode]),
            2.0 * np.eye(4, dtype=np.float32),
        ),
    ])
    moves = node("vadd", [
        moves,
        dense(
            node("vmul", [shot_values, shot_mode]),
            2.0 * np.eye(4, dtype=np.float32),
        ),
    ])

    action_features = node("vconcat", [moves, cast_total])
    w = np.zeros((8, 5), dtype=np.float32)
    b = np.full(8, -10.0, dtype=np.float32)
    for d in range(4):
        w[1 + d, d] = 1.0
        b[1 + d] = 0.0
    w[6, 4] = 11.0
    b[6] = -1.0
    dense(action_features, w, b, name="logits")

    # Masks and desired patterns for the nine clickable spell-grid cells.
    cells = [(50 + 5 * r, 25 + 5 * c)
             for r in range(3) for c in range(3)]
    w = np.zeros((18, 3, 1, 1), dtype=np.float32)
    b = np.zeros(18, dtype=np.float32)
    for i, (r, c) in enumerate(cells):
        w[2 * i, 0, 0, 0] = 63.0
        b[2 * i] = -float(r)
        w[2 * i + 1, 1, 0, 0] = 63.0
        b[2 * i + 1] = -float(c)
    offsets = node("abs", [conv("coords", w, b)])
    w = np.zeros((9, 18, 1, 1), dtype=np.float32)
    for i in range(9):
        w[i, 2 * i, 0, 0] = -1.0
        w[i, 2 * i + 1, 0, 0] = -1.0
    cell_masks = node("relu", [
        conv(offsets, w, np.ones(9, dtype=np.float32))
    ])
    selected_cells = node("gmax", [
        node("mul", [cell_masks, sl("x0", 14, 15)])
    ])

    spell_flags = node("vconcat", [
        cast_teleport, cast_resize, cast_fire
    ])
    patterns = np.zeros((9, 3), dtype=np.float32)
    patterns[[0, 1, 4], 0] = 1.0
    patterns[[1, 3, 5, 7], 1] = 1.0
    patterns[[1, 4, 7], 2] = 1.0
    desired_cells = dense(spell_flags, patterns)
    extra = node("vrelu", [vsub(selected_cells, desired_cells, 9)])
    missing = node("vrelu", [vsub(desired_cells, selected_cells, 9)])
    priorities = node("vadd", [
        dense(extra, 2.0 * np.eye(9, dtype=np.float32)),
        missing,
    ])
    click_channels = node("mul", [cell_masks, priorities])
    conv(
        click_channels,
        np.ones((1, 9, 1, 1), dtype=np.float32),
        name="click",
    )

    # Record teleport use on its final pattern toggle.
    hamming_w = np.ones((1, 9), dtype=np.float32)
    hamming_w[0, [0, 1, 4]] = -1.0
    hamming = dense(
        selected_cells, hamming_w, np.array([3.0], dtype=np.float32)
    )
    almost_complete = vstep(
        dense(
            hamming,
            np.array([[-1.0]], dtype=np.float32),
            np.array([1.5], dtype=np.float32),
        ),
        1,
    )
    teleport_finishing = node("vmul", [cast_teleport, almost_complete])
    mark_big = node("vmul", [teleport_finishing, big])
    mark_small = node("vmul", [teleport_finishing, small])
    marks = node("vconcat", [mark_big, mark_small])
    previous_flags = node("vconcat", [mem_big, mem_small])
    updated_flags = vclip(node("vadd", [previous_flags, marks]), 2)
    next_memory = node("vconcat", [updated_flags, target_presence])
    dense(next_memory, np.eye(4, dtype=np.float32), name="mem_next")

    return {"graph": graph, "memory_size": 4, "params": params}
