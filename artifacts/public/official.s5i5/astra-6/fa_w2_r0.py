import numpy as np


def build():
    params = {}
    cache = {}
    graph = []
    target = graph
    shapes = {}
    counter = 0

    colors = [7, 8, 9, 10, 11, 12, 14]
    C = len(colors)
    N = C + 1
    A = 3 * C
    S = 3 * N + 4
    M = 4 * A + C + S

    shapes.update({
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (M,),
        "aprev": (8,),
        "avail": (8,),
    })

    def parameter(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p" + str(len(params))
            params[name] = a.copy()
            cache[key] = name
        return cache[key]

    def fresh(prefix="n"):
        nonlocal counter
        counter += 1
        return prefix + str(counter)

    def node(op, inputs, shape, name=None, **fields):
        name = name or fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        target.append(nd)
        shapes[name] = tuple(shape)
        return name

    def linear(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        spatial = len(shapes[x]) == 3
        kw = {"w": parameter(w[:, :, None, None] if spatial else w)}
        if b is not None:
            b = np.asarray(b, np.float32)
            if b.ndim == 0:
                b = np.full(w.shape[0], b, np.float32)
            kw["b"] = parameter(b)
        return node(
            "conv" if spatial else "dense",
            [x],
            (w.shape[0], 64, 64) if spatial else (w.shape[0],),
            name=name,
            **kw
        )

    def conv(x, w, b=None):
        w = np.asarray(w, np.float32)
        kw = {"w": parameter(w)}
        if b is not None:
            b = np.asarray(b, np.float32)
            if b.ndim == 0:
                b = np.full(w.shape[0], b, np.float32)
            kw["b"] = parameter(b)
        return node("conv", [x], (w.shape[0], 64, 64), **kw)

    def affine(x, a=1.0, b=0.0, name=None):
        n = shapes[x][0]
        return linear(
            x,
            np.eye(n, dtype=np.float32) * a,
            np.full(n, b, np.float32),
            name=name,
        )

    def cat(xs, name=None):
        spatial = len(shapes[xs[0]]) == 3
        n = sum(shapes[x][0] for x in xs)
        return node(
            "concat" if spatial else "vconcat",
            xs,
            (n, 64, 64) if spatial else (n,),
            name=name,
        )

    def sl(x, start, stop):
        spatial = len(shapes[x]) == 3
        return node(
            "slice" if spatial else "vslice",
            [x],
            (stop - start, 64, 64) if spatial else (stop - start,),
            start=int(start),
            stop=int(stop),
        )

    def relu(x):
        return node(
            "relu" if len(shapes[x]) == 3 else "vrelu",
            [x], shapes[x]
        )

    def add(x, y):
        spatial = len(shapes[x]) == 3 or len(shapes[y]) == 3
        n = max(shapes[x][0], shapes[y][0])
        return node(
            "add" if spatial else "vadd",
            [x, y],
            (n, 64, 64) if spatial else (n,),
        )

    def mul(x, y):
        spatial = len(shapes[x]) == 3 or len(shapes[y]) == 3
        n = max(shapes[x][0], shapes[y][0])
        return node(
            "mul" if spatial else "vmul",
            [x, y],
            (n, 64, 64) if spatial else (n,),
        )

    def sub(x, y):
        if len(shapes[x]) == 1 and len(shapes[y]) == 1:
            return add(x, affine(y, -1))
        return node(
            "sub", [x, y],
            (max(shapes[x][0], shapes[y][0]), 64, 64)
        )

    def maximum(x, y):
        if len(shapes[x]) == 1 and len(shapes[y]) == 1:
            return add(x, relu(sub(y, x)))
        return node(
            "max2", [x, y],
            (max(shapes[x][0], shapes[y][0]), 64, 64)
        )

    def absolute(x):
        if len(shapes[x]) == 3:
            return node("abs", [x], shapes[x])
        n = shapes[x][0]
        e = np.eye(n, dtype=np.float32)
        z = relu(linear(x, np.concatenate([e, -e])))
        return linear(z, np.concatenate([e, e], axis=1))

    def clip(x):
        n = shapes[x][0]
        e = np.eye(n, dtype=np.float32)
        z = relu(linear(
            x,
            np.concatenate([e, e]),
            np.concatenate([np.zeros(n), -np.ones(n)]),
        ))
        return linear(z, np.concatenate([e, -e], axis=1))

    def total(x):
        return linear(x, np.ones((1, shapes[x][0]), np.float32))

    def gmax(x):
        return node("gmax", [x], (shapes[x][0],))

    def gsum(x):
        z = node("gmean", [x], (shapes[x][0],))
        return affine(z, 4096)

    def loop(init, count, function):
        nonlocal target
        parent = target
        body = []
        target = body
        state = fresh("s")
        shapes[state] = shapes[init]
        result = function(state)
        target = parent
        lid = fresh("L")
        parent.append({
            "id": lid,
            "op": "loop",
            "T": int(count),
            "state": [{"id": state, "init": init}],
            "body": body,
            "next": {state: result},
        })
        out = lid + "." + state
        shapes[out] = shapes[result]
        return out

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def neighbors(x, sign=1):
        w = np.zeros((4, 1, 7, 7), np.float32)
        for d, (dy, dx) in enumerate(directions):
            w[d, 0, 3 + sign * 3 * dy, 3 + sign * 3 * dx] = 1
        return conv(x, w)

    def shift4(x, sign=1, collapse=False):
        w = np.zeros((1 if collapse else 4, 4, 7, 7), np.float32)
        for d, (dy, dx) in enumerate(directions):
            w[0 if collapse else d, d,
              3 - sign * 3 * dy, 3 - sign * 3 * dx] = 1
        return conv(x, w)

    # Detect centers of the three-pixel game cells.
    centers = list(range(1, 64, 3))
    w = np.zeros((2 * len(centers), 3), np.float32)
    b = np.zeros(2 * len(centers), np.float32)
    for axis in range(2):
        for j, p in enumerate(centers):
            k = axis * len(centers) + j
            w[k, axis] = 63
            b[k] = -p

    hats = relu(affine(absolute(linear("coords", w, b)), -2, 1))
    w = np.zeros((2, 2 * len(centers)), np.float32)
    w[0, :len(centers)] = 1
    w[1, len(centers):] = 1
    axes = clip(affine(linear(hats, w), 4, -2))
    grid = mul(sl(axes, 0, 1), sl(axes, 1, 2))

    xy = linear("coords", [[63, 0, 0], [0, 63, 0]])
    py, px = sl(xy, 0, 1), sl(xy, 1, 2)

    # Rod and obstacle cells; reject UI interiors.
    w = np.zeros((N + 1, 16, 3, 3), np.float32)
    for i, color in enumerate(colors + [1, 15]):
        w[i, color] = 1
    cell_colors = clip(conv("x0", w))

    w = np.zeros((1, 16, 3, 3), np.float32)
    w[0, 2] = w[0, 4] = 1
    no_ui = affine(clip(conv("x0", w)), -1, 1)
    cell_colors = mul(cell_colors, mul(grid, no_ui))

    main = sl(cell_colors, 0, N)
    rods = sl(main, 0, C)
    neutral = sl(main, C, C + 1)
    fixed = sl(cell_colors, N, N + 1)
    occupied = total(cell_colors)
    rod_present = total(rods)
    color_code = linear(
        cell_colors, np.arange(1, N + 2, dtype=np.float32)[None]
    )

    # Four anchor-cap directions, ordered down/right/up/left.
    w = np.zeros((4, 1, 3, 3), np.float32)
    w[0, 0, 0, :] = 1
    w[1, 0, :, 0] = 1
    w[2, 0, 2, :] = 1
    w[3, 0, :, 2] = 1
    cap_pixels = conv(sl("x0", 3, 4), w)
    anchors = mul(clip(affine(cap_pixels, 1, -1)), rod_present)
    anchor_any = total(anchors)

    same_previous = relu(affine(
        absolute(sub(color_code, neighbors(color_code, -1))), -1, 1
    ))
    continuation = mul(
        same_previous, mul(rod_present, affine(anchor_any, -1, 1))
    )

    def reach_body(s):
        return add(mul(shift4(s), continuation), anchors)

    reached = loop(anchors, 21, reach_body)
    orientation = clip(reached)
    tips = relu(sub(orientation, shift4(orientation, -1)))

    # Crucial distinction: reached is binary, not a length counter.
    # A tip is removable precisely when it is not the anchor cell.
    non_anchor = affine(anchor_any, -1, 1)
    removable_cells = mul(orientation, non_anchor)
    direction_code = linear(orientation, [[1, 2, 3, 4]])

    # Axis masks used to propagate labels along child objects.
    nn = neighbors(neutral)
    nh = clip(add(sl(nn, 1, 2), sl(nn, 3, 4)))
    nv = clip(add(sl(nn, 0, 1), sl(nn, 2, 3)))
    horizontal = add(sl(orientation, 1, 2), sl(orientation, 3, 4))
    vertical = add(sl(orientation, 0, 1), sl(orientation, 2, 3))
    hmask = cat([mul(rods, horizontal), mul(neutral, nh)])
    vmask = cat([mul(rods, vertical), mul(neutral, nv)])
    accepts_child = clip(add(anchor_any, neutral))
    child_colors = mul(main, accepts_child)

    # Dots and goal rings.
    purple = sl("x0", 13, 14)
    dots = mul(purple, grid)
    w = np.zeros((1, 1, 3, 3), np.float32)
    for y, x in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        w[0, 0, y, x] = 1
    goals = mul(relu(affine(conv(purple, w), 1, -3)), grid)
    matched = mul(dots, goals)
    unsatisfied = mul(dots, affine(goals, -1, 1))
    open_goals = mul(goals, affine(dots, -1, 1))

    rank = mul(
        unsatisfied, linear("coords", [[-4032, -63, 0]], [4096])
    )
    active = mul(
        unsatisfied,
        relu(affine(absolute(sub(rank, gmax(rank))), -2, 1)),
    )
    dot_position = gsum(mul(active, xy))

    # Max-plus distance propagation around fixed obstacles.
    free = mul(grid, affine(fixed, -1, 1))
    seed_value = affine(open_goals, 96)

    def distance_body(s):
        nb = neighbors(s)
        z = maximum(sl(nb, 0, 1), sl(nb, 1, 2))
        z = maximum(z, sl(nb, 2, 3))
        z = maximum(z, sl(nb, 3, 4))
        return maximum(seed_value, mul(relu(affine(z, 1, -1)), free))

    goal_value = loop(seed_value, 96, distance_body)
    current_value = gsum(mul(active, goal_value))

    # Trace ancestors of the active dot.
    links = add(
        sub(orientation, tips), mul(tips, neighbors(accepts_child))
    )

    def ancestor_body(s):
        return maximum(active, total(mul(neighbors(s), links)))

    ancestors = loop(active, 64, ancestor_body)
    selected_anchors = mul(anchor_any, ancestors)

    arriving = shift4(tips)
    perm = np.zeros((4, 4), np.float32)
    for d in range(4):
        perm[d, (d - 1) % 4] = 1
    skip = total(mul(anchors, linear(arriving, perm)))
    base_features = cat([xy, skip, sl("coords", 2, 3)])

    # Glyph detectors for horizontal and vertical resize controls.
    patterns = [
        {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2)},
        {(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)},
        {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)},
        {(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)},
    ]
    cross = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
    wg = np.zeros((4 * C, 16, 3, 3), np.float32)
    wr = np.zeros((C, 16, 3, 3), np.float32)
    for c, color in enumerate(colors):
        for d, pattern in enumerate(patterns):
            for y in range(3):
                for x in range(3):
                    wg[4 * c + d,
                       color if (y, x) in pattern else 4, y, x] = 1
        for y in range(3):
            for x in range(3):
                wr[c, color if (y, x) in cross else 4, y, x] = 1

    glyphs = relu(conv("x0", wg, -8))
    rotate_glyphs = relu(conv("x0", wr, -8))
    wb = np.zeros((A, 5 * C), np.float32)
    for c in range(C):
        wb[3 * c, 4 * c + 1] = wb[3 * c, 4 * c + 3] = 1
        wb[3 * c + 1, 4 * c] = wb[3 * c + 1, 4 * c + 2] = 1
        wb[3 * c + 2, 4 * C + c] = 1

    buttons = linear(cat([glyphs, rotate_glyphs]), wb)
    present_buttons = gmax(buttons)

    def choose(score, output_name=None):
        painted = total(mul(buttons, score))
        if output_name is not None:
            painted = affine(painted, name=output_name)
        peak = gmax(painted)
        equality = relu(affine(
            absolute(sub(painted, peak)), -10000, 1
        ))
        selected = gmax(mul(buttons, equality))
        return selected, painted

    group_maps = []
    descendant_maps = []
    collision_maps = []
    source_collision_maps = []
    root_action_maps = []
    records = []

    fixed_forward = neighbors(fixed)
    fixed_backward = neighbors(fixed, -1)

    # Per-color pooled features:
    # progress 2, matched loss 2, collisions 2, missing cells 2,
    # valid lengths 2, root size 1, carried dots 1, pivot features 4.
    F = 16

    for ci in range(C):
        root = sl(rods, ci, ci + 1)
        initial_labels = mul(root, direction_code)

        def descendant_body(s):
            incoming = shift4(mul(s, tips), collapse=True)
            labeled = mul(incoming, child_colors)
            rh = node("rowmax", [labeled], shapes[labeled])
            cv = node("colmax", [labeled], shapes[labeled])
            spread = total(maximum(mul(rh, hmask), mul(cv, vmask)))
            return maximum(initial_labels, spread)

        labels = loop(initial_labels, 8, descendant_body)
        group_directions = relu(affine(absolute(linear(
            labels,
            np.ones((4, 1), np.float32),
            -np.arange(1, 5, dtype=np.float32),
        )), -1, 1))

        descendants = mul(group_directions, affine(root, -1, 1))
        root_directions = mul(root, orientation)
        root_tips = mul(root, tips)
        removable_tips = mul(root_tips, removable_cells)
        shrink_valid = gmax(removable_tips)
        shrinking_descendants = mul(descendants, shrink_valid)

        dsum = total(descendants)
        ssum = total(shrinking_descendants)
        old_grow = total(add(descendants, root_tips))
        old_shrink = add(ssum, total(removable_tips))
        grow_to = shift4(add(descendants, root_tips), collapse=True)
        shrink_to = shift4(shrinking_descendants, -1, collapse=True)

        removed = cat([dsum, old_shrink])
        added = cat([grow_to, shrink_to])
        collisions = relu(affine(
            add(sub(occupied, removed), added), 1, -1
        ))

        affected = add(descendants, root_directions)
        ag = mul(affected, active)
        ash = mul(ag, shrink_valid)
        progress_maps = mul(cat([
            sub(shift4(ag, collapse=True), total(ag)),
            sub(shift4(ash, -1, collapse=True), total(ash)),
        ]), goal_value)

        mg = mul(affected, matched)
        ms = mul(mg, shrink_valid)
        matched_loss = cat([
            sub(total(mg), mul(shift4(mg, collapse=True), goals)),
            sub(total(ms), mul(shift4(ms, -1, collapse=True), goals)),
        ])
        missing = cat([
            sub(old_grow, grow_to),
            sub(ssum, shrink_to),
        ])

        group_any = add(root, dsum)
        group_maps.append(group_any)
        descendant_maps.append(dsum)
        collision_maps.append(collisions)
        root_action_maps.append(cat([root, root, root]))

        source_collision_maps.append(cat([
            total(mul(descendants, fixed_forward)),
            total(mul(shrinking_descendants, fixed_backward)),
        ]))

        pivot = mul(mul(root, selected_anchors), base_features)
        records.append(gsum(cat([
            progress_maps,
            matched_loss,
            collisions,
            missing,
            total(root_tips),
            total(removable_tips),
            root,
            mul(group_any, dots),
            pivot,
        ])))

    features = cat(records)
    groups = cat(group_maps)
    descendants_all = cat(descendant_maps)
    collision_all = cat(collision_maps)
    collision_sources = cat(source_collision_maps)
    root_actions = cat(root_action_maps)
    D = F * C

    def pick(start, count):
        w = np.zeros((C * count, D), np.float32)
        for c in range(C):
            for j in range(count):
                w[c * count + j, c * F + start + j] = 1
        return linear(features, w)

    progress = pick(0, 2)
    matched_loss = pick(2, 2)
    collision_counts = pick(4, 2)
    missing_counts = relu(pick(6, 2))
    resize_flags = clip(pick(8, 2))
    sizes = pick(10, 1)
    carried_dots = clip(pick(11, 1))
    bases = pick(12, 4)

    resize_to_actions = np.zeros((A, 2 * C), np.float32)
    rotations_to_actions = np.zeros((A, C), np.float32)
    actions_to_resize = np.zeros((2 * C, A), np.float32)
    actions_to_color = np.zeros((C, A), np.float32)
    color_to_grow = np.zeros((A, C), np.float32)
    color_to_shrink = np.zeros((A, C), np.float32)
    color_to_all = np.zeros((A, C), np.float32)

    for c in range(C):
        resize_to_actions[3 * c, 2 * c] = 1
        resize_to_actions[3 * c + 1, 2 * c + 1] = 1
        actions_to_resize[2 * c, 3 * c] = 1
        actions_to_resize[2 * c + 1, 3 * c + 1] = 1
        rotations_to_actions[3 * c + 2, c] = 1
        color_to_grow[3 * c, c] = 1
        color_to_shrink[3 * c + 1, c] = 1
        for a in range(3):
            actions_to_color[c, 3 * c + a] = 1
            color_to_all[3 * c + a, c] = 1

    # Predicted endpoints after up to three counterclockwise turns.
    rotation_input = cat([dot_position, bases])
    width = 2 + 4 * C
    turn = np.zeros((2 * C, width), np.float32)
    extra = np.zeros_like(turn)
    skips = np.zeros_like(turn)
    exists = np.zeros((C, 4 * C), np.float32)

    for c in range(C):
        y, x = 2 * c, 2 * c + 1
        by, bx, sk = 2 + 4 * c, 3 + 4 * c, 4 + 4 * c
        turn[y, by] = turn[y, bx] = 1
        turn[y, 1] = -1
        turn[x, bx] = turn[x, 0] = 1
        turn[x, by] = -1
        extra[y, by] = extra[y, 1] = 1
        extra[y, bx] = extra[y, 0] = -1
        extra[x, by] = extra[x, bx] = 1
        extra[x, 0] = extra[x, 1] = -1
        skips[y, sk] = skips[x, sk] = 1
        exists[c, 4 * c + 3] = 1

    endpoint1 = add(
        linear(rotation_input, turn),
        mul(linear(rotation_input, extra), linear(rotation_input, skips)),
    )
    rotate_again = np.zeros((2 * C, 6 * C), np.float32)
    for c in range(C):
        y, x = 2 * c, 2 * c + 1
        by, bx = 2 * C + 4 * c, 2 * C + 4 * c + 1
        rotate_again[y, by] = rotate_again[y, bx] = 1
        rotate_again[y, x] = -1
        rotate_again[x, bx] = rotate_again[x, y] = 1
        rotate_again[x, by] = -1

    endpoint2 = linear(cat([endpoint1, bases]), rotate_again)
    endpoint3 = linear(cat([endpoint2, bases]), rotate_again)
    endpoints = cat([endpoint1, endpoint2, endpoint3])

    repeat_xy = np.zeros((6 * C, 2), np.float32)
    sum_pairs = np.zeros((3 * C, 6 * C), np.float32)
    for i in range(3 * C):
        repeat_xy[2 * i, 0] = 1
        repeat_xy[2 * i + 1, 1] = 1
        sum_pairs[i, 2 * i:2 * i + 2] = -1

    location_error = absolute(sub(linear(xy, repeat_xy), endpoints))
    endpoint_masks = mul(
        relu(linear(location_error, sum_pairs, np.ones(3 * C))), grid
    )
    endpoint_values = gsum(mul(endpoint_masks, goal_value))
    r1 = sub(sl(endpoint_values, 0, C), current_value)
    r2 = affine(
        sub(sl(endpoint_values, C, 2 * C), current_value), 0.5, -0.5
    )
    r3 = affine(
        sub(sl(endpoint_values, 2 * C, 3 * C), current_value),
        1 / 3, -1,
    )
    rotation_progress = mul(
        maximum(maximum(r1, r2), r3),
        clip(linear(bases, exists)),
    )

    grow_pick = np.zeros((C, 2 * C), np.float32)
    shrink_pick = np.zeros_like(grow_pick)
    for c in range(C):
        grow_pick[c, 2 * c] = 1
        shrink_pick[c, 2 * c + 1] = 1

    rotation_flags = linear(resize_flags, grow_pick)
    valid = mul(present_buttons, add(
        linear(resize_flags, resize_to_actions),
        linear(rotation_flags, rotations_to_actions),
    ))

    resize_goal_score = add(
        affine(progress, 2), affine(matched_loss, -5)
    )
    rotation_goal_score = add(
        affine(rotation_progress, 1.4, -0.2),
        affine(linear(matched_loss, grow_pick), -5),
    )
    goal_scores = add(
        linear(resize_goal_score, resize_to_actions),
        linear(rotation_goal_score, rotations_to_actions),
    )

    # Timer-independent observation signature for failed-click detection.
    signature = gsum(cat([
        main,
        mul(main, py),
        mul(main, px),
        mul(anchors, color_code),
    ]))
    last = sl("mem", 0, A)
    previous_task = sl("mem", A, 2 * A)
    old_blocked = sl("mem", 2 * A, 3 * A)
    old_random = sl("mem", 3 * A, 4 * A)
    old_direction = sl("mem", 4 * A, 4 * A + C)
    old_signature = sl("mem", 4 * A + C, M)

    changed = clip(affine(
        total(absolute(sub(signature, old_signature))), 4
    ))
    blocked_memory = clip(mul(
        add(old_blocked, last), affine(changed, -1, 1)
    ))

    # Simplify dot-free assemblies before advancing a goal-bearing arm.
    shrink_possible = linear(resize_flags, shrink_pick)
    simple_color = mul(shrink_possible, affine(carried_dots, -1, 1))
    simple_actions = linear(simple_color, color_to_shrink)
    simple_priority = linear(
        mul(simple_color, affine(sizes, 0.1, 20)),
        color_to_shrink,
    )
    keep_task = mul(previous_task, simple_actions)
    task_scores = add(goal_scores, simple_priority)
    task_scores = add(task_scores, affine(keep_task, 60))
    task_scores = add(task_scores, affine(valid, 200, -200))
    task_scores = linear(
        task_scores,
        np.eye(A, dtype=np.float32),
        300 + 0.002 * (A - np.arange(A, dtype=np.float32)),
    )
    task, _ = choose(task_scores)

    geometric_block = clip(add(collision_counts, missing_counts))
    resize_block = linear(geometric_block, resize_to_actions)
    blocked = clip(add(resize_block, blocked_memory))
    legal = mul(valid, affine(blocked, -1, 1))

    fixed_hit_resize = clip(add(
        gsum(mul(collision_all, fixed)), missing_counts
    ))
    fixed_hit = clip(add(
        linear(fixed_hit_resize, resize_to_actions), blocked_memory
    ))
    stationary_collisions = mul(
        collision_all, affine(fixed, -1, 1)
    )

    # Repair inertia is separate from direct goal preferences.
    grow_preference = affine(old_direction, 0.35, 0.95)
    shrink_preference = affine(old_direction, -0.35, 1.00)
    rotation_preference = clip(affine(rotation_progress, 0.015, 0.50))
    helper_preferences = add(
        add(
            linear(grow_preference, color_to_grow),
            linear(shrink_preference, color_to_shrink),
        ),
        linear(rotation_preference, rotations_to_actions),
    )
    helper_preferences = mul(helper_preferences, valid)

    # Transfer blocked requests to relevant obstruction/ancestor controls.
    def repair_body(request):
        failed = mul(request, blocked)
        failed_resize = linear(failed, actions_to_resize)
        fixed_request = mul(failed, fixed_hit)

        obstruction_field = total(mul(
            stationary_collisions, failed_resize
        ))
        ancestor_field = total(mul(root_actions, fixed_request))
        child_field = total(mul(collision_sources, failed_resize))

        obstruction_helpers = clip(gmax(mul(
            groups, obstruction_field
        )))
        ancestor_helpers = clip(gmax(mul(
            descendants_all, ancestor_field
        )))
        child_helpers = clip(gmax(mul(rods, child_field)))

        generic = maximum(obstruction_helpers, ancestor_helpers)
        generic_actions = mul(
            linear(generic, color_to_all), helper_preferences
        )
        child_actions = linear(
            mul(child_helpers, shrink_possible), color_to_shrink
        )
        child_actions = mul(child_actions, present_buttons)

        self_turn = linear(
            linear(fixed_request, actions_to_color),
            rotations_to_actions,
        )
        self_turn = mul(self_turn, present_buttons)

        helpers = maximum(
            affine(generic_actions, 1.35),
            affine(child_actions, 1.75),
        )
        helpers = maximum(helpers, affine(self_turn, 0.65))
        helpers = mul(helpers, affine(blocked_memory, -1, 1))

        return maximum(mul(request, legal), helpers)

    requests = loop(task, 7, repair_body)

    # Bounded deterministic exploration and immediate reversal penalty.
    has_previous = clip(total(last))
    seeds = 0.1 + 0.8 * (
        ((np.arange(A) + 1) * 0.6180339887498949) % 1
    )
    random_next = add(
        affine(mul(old_random, affine(old_random, -1, 1)), 4),
        linear(affine(has_previous, -1, 1), seeds[:, None]),
    )

    reverse = np.zeros((A, A), np.float32)
    for c in range(C):
        reverse[3 * c, 3 * c + 1] = -0.12
        reverse[3 * c + 1, 3 * c] = -0.12
        reverse[3 * c + 2, 3 * c + 2] = -0.04

    scores = add(affine(requests, 10), affine(goal_scores, 0.015))
    scores = add(scores, linear(last, reverse))
    scores = add(scores, affine(random_next, 0.04))
    scores = add(scores, affine(legal, 100, -100))
    scores = add(scores, affine(blocked_memory, -200))
    scores = linear(
        scores,
        np.eye(A, dtype=np.float32),
        1000 + 0.003 * (A - np.arange(A, dtype=np.float32)),
    )
    winner, _ = choose(scores, output_name="click")

    direction_update = np.zeros((C, A), np.float32)
    resized = np.zeros((C, A), np.float32)
    for c in range(C):
        direction_update[c, 3 * c] = 1
        direction_update[c, 3 * c + 1] = -1
        resized[c, 3 * c] = resized[c, 3 * c + 1] = 1

    next_direction = add(
        mul(old_direction, affine(linear(winner, resized), -1, 1)),
        linear(winner, direction_update),
    )
    cat([
        winner,
        task,
        blocked_memory,
        random_next,
        next_direction,
        signature,
    ], name="mem_next")

    bias = np.full(8, -1, np.float32)
    bias[6] = 1
    linear(
        "avail", np.zeros((8, 8), np.float32),
        bias, name="logits"
    )

    # Pure graph bookkeeping: discard unreachable intermediates.
    table = {}
    for nd in graph:
        table[nd["id"]] = nd
        if nd["op"] == "loop":
            for state in nd["state"]:
                table[nd["id"] + "." + state["id"]] = nd

    required = set()
    inputs = {"x0", "coords", "mem", "aprev", "avail"}

    def require(name):
        if name in inputs:
            return
        nd = table[name]
        if nd["id"] in required:
            return
        required.add(nd["id"])

        if nd["op"] != "loop":
            for src in nd["in"]:
                require(src)
            return

        local = {s["id"] for s in nd["state"]}
        local.update(n["id"] for n in nd["body"])
        for state in nd["state"]:
            if "init" in state:
                require(state["init"])
        for bn in nd["body"]:
            for src in bn["in"]:
                if src not in local:
                    require(src)

    for output in ["click", "logits", "mem_next"]:
        require(output)
    graph = [nd for nd in graph if nd["id"] in required]

    used_params = set()

    def collect(nodes):
        for nd in nodes:
            if nd["op"] == "loop":
                collect(nd["body"])
            else:
                for key in ("w", "b"):
                    if key in nd:
                        used_params.add(nd[key])

    collect(graph)
    params = {
        name: value for name, value in params.items()
        if name in used_params
    }

    return {
        "graph": graph,
        "memory_size": M,
        "params": params,
    }
