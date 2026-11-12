import numpy as np


def build():
    graph = []
    params = {}
    dst = graph
    serial = 0
    parameter_cache = {}
    node_caches = {id(graph): {}}
    constants = {}
    M = 149
    shapes_meta = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "avail": (8,),
        "aprev": (8,),
        "mem": (M,),
    }

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return "%s_%d" % (prefix, serial)

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def node(op, inputs=None, name=None, **fields):
        inputs = [] if inputs is None else inputs
        cache = node_caches.setdefault(id(dst), {})
        key = None
        if op != "loop" and name is None:
            key = (op, tuple(inputs), tuple(sorted(fields.items())))
            if key in cache:
                return cache[key]
            if dst is not graph and key in node_caches[id(graph)]:
                return node_caches[id(graph)][key]

        ident = fresh() if name is None else name
        nd = {"id": ident, "op": op}
        if op != "loop":
            nd["in"] = list(inputs)
        nd.update(fields)
        dst.append(nd)
        if key is not None:
            cache[key] = ident

        if op == "loop":
            for s in fields["state"]:
                shapes_meta[ident + "." + s["id"]] = shapes_meta[s["id"]]
        elif op == "conv":
            shapes_meta[ident] = (params[fields["w"]].shape[0], 64, 64)
        elif op == "dense":
            shapes_meta[ident] = (params[fields["w"]].shape[0],)
        elif op in ("gmax", "gmean"):
            shapes_meta[ident] = (shapes_meta[inputs[0]][0],)
        elif op in ("concat", "vconcat"):
            count = sum(shapes_meta[x][0] for x in inputs)
            shapes_meta[ident] = (
                (count, 64, 64) if op == "concat" else (count,)
            )
        elif op in ("slice", "vslice"):
            count = fields["stop"] - fields["start"]
            shapes_meta[ident] = (
                (count, 64, 64) if op == "slice" else (count,)
            )
        elif op in ("add", "sub", "mul", "max2", "min2"):
            count = max(shapes_meta[x][0] for x in inputs)
            shapes_meta[ident] = (count, 64, 64)
        elif op in ("vadd", "vmul"):
            shapes_meta[ident] = (max(shapes_meta[x][0] for x in inputs),)
        else:
            shapes_meta[ident] = shapes_meta[inputs[0]]
        return ident

    def conv(x, w, b=None, name=None):
        fields = {"w": w if isinstance(w, str) else parameter(w)}
        if b is not None:
            fields["b"] = b if isinstance(b, str) else parameter(b)
        return node("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": w if isinstance(w, str) else parameter(w)}
        if b is not None:
            fields["b"] = b if isinstance(b, str) else parameter(b)
        return node("dense", [x], name=name, **fields)

    def constant(value):
        nonlocal dst
        value = float(value)
        if value not in constants:
            save = dst
            dst = graph
            constants[value] = dense("avail", np.zeros((1, 8)), [value])
            dst = save
        return constants[value]

    def fixed_vector(values):
        values = np.asarray(values, np.float32)
        return dense("avail", np.zeros((len(values), 8)), values)

    def state(ident, init=None, zeros=None):
        if init is not None:
            shapes_meta[ident] = shapes_meta[init]
            return {"id": ident, "init": init}
        shapes_meta[ident] = tuple(zeros)
        return {"id": ident, "zeros": list(zeros)}

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def affine(a, scale=1, bias=0):
        if scale == 1 and bias == 0:
            return a
        if shapes_meta[a] == (1,):
            return dense(a, [[scale]], [bias] if bias else None)
        if scale != 1:
            a = vm(a, constant(scale))
        if bias != 0:
            a = va(a, constant(bias))
        return a

    def neg(a):
        return affine(a, -1)

    def sub(a, b):
        return va(a, neg(b))

    def vr(a):
        return node("vrelu", [a])

    def clip(a):
        p = vr(a)
        return sub(p, vr(affine(p, 1, -1)))

    def one_minus(a):
        return affine(a, -1, 1)

    def vmax(a, b):
        return va(b, vr(sub(a, b)))

    def eq(a, b):
        d = sub(a, b) if isinstance(b, str) else affine(a, 1, -float(b))
        absolute = va(vr(d), vr(neg(d)))
        return vr(affine(absolute, -4, 1))

    def above(a, threshold):
        return clip(affine(a, 2, -2 * threshold))

    def hard_reach(a):
        return clip(affine(a, 2, -0.5))

    def sl(a, start, stop):
        return node("vslice", [a], start=start, stop=stop)

    def ss(a, start, stop):
        return node("slice", [a], start=start, stop=stop)

    def sa(a, b):
        return node("add", [a, b])

    def sm(a, b):
        return node("mul", [a, b])

    def sr(a):
        return node("relu", [a])

    def sclip(a):
        return node("min2", [sr(a), constant(1)])

    def sconcat(xs):
        return node("concat", xs)

    def vtotal(xs):
        a = xs[0]
        for b in xs[1:]:
            a = va(a, b)
        return a

    sum_parameters = {}

    def vsum(a, length):
        if length not in sum_parameters:
            sum_parameters[length] = parameter(np.ones((1, length)))
        return dense(a, sum_parameters[length])

    def dot(a, b, length):
        return vsum(vm(a, b), length)

    N = 144
    T = 56
    indices = fixed_vector(np.arange(N))
    state_indices = fixed_vector(np.arange(96))

    # Pixel-aligned logical cells.
    w = np.zeros((24, 3, 1, 1), np.float32)
    b = np.zeros(24, np.float32)
    for i in range(12):
        w[i, 0, 0, 0] = 63
        b[i] = -(2 + 5 * i)
        w[12 + i, 1, 0, 0] = 63
        b[12 + i] = -(6 + 5 * i)
    coordinate_distances = node("abs", [conv("coords", w, b)])
    boxes = sclip(conv(
        coordinate_distances,
        (-2 * np.eye(24)).reshape(24, 24, 1, 1),
        np.full(24, 5.5)
    ))
    points = sclip(conv(
        coordinate_distances,
        (-4 * np.eye(24)).reshape(24, 24, 1, 1),
        np.full(24, 2)
    ))

    w = np.zeros((N, 24, 1, 1), np.float32)
    for r in range(12):
        for c in range(12):
            w[12 * r + c, r, 0, 0] = 1
            w[12 * r + c, 12 + c, 0, 0] = 1
    tiles = sr(conv(boxes, w, np.full(N, -1)))

    row_w = np.zeros((1, 24, 1, 1), np.float32)
    col_w = row_w.copy()
    row_w[0, :12, 0, 0] = 1
    col_w[0, 12:, 0, 0] = 1
    grid_points = sm(conv(points, row_w), conv(points, col_w))

    id_w = np.zeros((1, 24, 1, 1), np.float32)
    id_w[0, :12, 0, 0] = 12 * np.arange(12)
    id_w[0, 12:, 0, 0] = np.arange(12)
    index_image = conv(boxes, id_w)
    map_sum = parameter(np.ones((1, N, 1, 1)))

    def to_map(v):
        return conv(sm(tiles, v), map_sum)

    def pool_tiles(image):
        return affine(node("gmean", [sm(tiles, image)]), 4096 / 25)

    def pool_centres(image):
        return affine(
            node("gmean", [sm(tiles, sm(image, grid_points))]), 4096
        )

    def spatial_equal(image, value):
        d = node("abs", [conv(image, np.ones((1, 1, 1, 1)), [-value])])
        return sr(conv(d, np.full((1, 1, 1, 1), -4), [1]))

    def point_or_box(row, col, row_bias=2, col_bias=2):
        w = np.zeros((2, 3, 1, 1), np.float32)
        w[0, 0, 0, 0] = 63
        w[1, 1, 0, 0] = 63
        d = node("abs", [conv("coords", w, [-row, -col])])
        q = sclip(conv(
            d, np.diag([-4, -4]).reshape(2, 2, 1, 1),
            [row_bias, col_bias]
        ))
        return sm(ss(q, 0, 1), ss(q, 1, 2))

    # State order: ((shape * 4 + rotation) * 4 + colour).
    glyph_shapes = [
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
    ]
    colours = [12, 9, 14, 8]
    glyphs = [
        np.rot90(np.asarray(s, np.float32), -r)
        for s in glyph_shapes for r in range(4)
    ]
    w = np.zeros((1, 16, 1, 1), np.float32)
    w[0, colours, 0, 0] = 1
    coloured = conv("x0", w)
    background_five = ss("x0", 5, 6)

    # Goal glyph and its surrounding frame.
    gw = np.zeros((24, 2, 5, 5), np.float32)
    gb = np.zeros(24, np.float32)
    for k, glyph in enumerate(glyphs):
        gw[k, 0, 1:4, 1:4] = 2 * glyph - 1
        gw[k, 1, :, :] = 1
        gw[k, 1, 1:4, 1:4] = 0
        gb[k] = 1 - glyph.sum() - 16
    goal_patterns = sr(conv(
        sconcat([coloured, background_five]), gw, gb
    ))
    goal_visible = conv(goal_patterns, np.ones((1, 24, 1, 1)))
    goal_base = conv(
        goal_patterns, (10 + 4 * np.arange(24)).reshape(1, 24, 1, 1)
    )
    cw = np.zeros((4, 16, 3, 3), np.float32)
    for k, colour in enumerate(colours):
        cw[k, colour] = 1
    goal_colours = sclip(conv("x0", cw))
    colour_number = conv(
        goal_colours, np.arange(4).reshape(1, 4, 1, 1)
    )
    goal_colour_number = sm(colour_number, goal_visible)

    # Current glyph in the doubled-scale HUD.
    uw = np.zeros((24, 1, 7, 7), np.float32)
    ub = np.zeros(24, np.float32)
    for k, glyph in enumerate(glyphs):
        for r in range(3):
            for c in range(3):
                uw[k, 0, 1 + 2 * r, 1 + 2 * c] = 2 * glyph[r, c] - 1
        ub[k] = 1 - glyph.sum()
    ui_point = point_or_box(57, 5)
    ui_shapes = node("gmax", [sm(sr(conv(coloured, uw, ub)), ui_point)])

    ucw = np.zeros((4, 16, 7, 7), np.float32)
    for k, colour in enumerate(colours):
        for r in range(3):
            for c in range(3):
                ucw[k, colour, 1 + 2 * r, 1 + 2 * c] = 1
    ui_colours = node("gmax", [sm(sclip(conv("x0", ucw)), ui_point)])
    ui_features = node("vconcat", [ui_shapes, ui_colours])
    uw = np.zeros((96, 28), np.float32)
    for t in range(24):
        for c in range(4):
            uw[4 * t + c, t] = 1
            uw[4 * t + c, 24 + c] = 1
    current_state = vr(dense(ui_features, uw, np.full(96, -1)))

    # Shape, colour, rotation, and refill icons.
    sw = np.zeros((4, 16, 3, 3), np.float32)
    sb = np.array([-3, -8, -4, -7], np.float32)
    sw[0, 0] = -1
    sw[2, 0] = -1
    for r, c in [(-1, -1), (0, 0), (0, 1), (1, 0)]:
        sw[0, 0, r + 1, c + 1] = 1
    for r, c in [(-1, 0), (0, 0), (0, 1)]:
        sw[2, 0, r + 1, c + 1] = 1
    for r, c in [(0, -1), (1, 0)]:
        sw[2, 1, r + 1, c + 1] = 1
    colour_icon = [[9, 14, 14], [9, 0, 8], [12, 12, 8]]
    for r in range(3):
        for c in range(3):
            sw[1, colour_icon[r][c], r, c] = 1
            if (r, c) != (1, 1):
                sw[3, 11, r, c] = 1
    modifiers = sr(conv("x0", sw, sb))

    pw = np.zeros((4, 1, 7, 7), np.float32)
    pw[0, 0, 6, 1:6] = 1
    pw[1, 0, 0, 1:6] = 1
    pw[2, 0, 1:6, 6] = 1
    pw[3, 0, 1:6, 0] = 1
    pushers = sr(conv(ss("x0", 1, 2), pw, np.full(4, -4)))

    wall_pixels = sclip(conv(
        ss("x0", 4, 5), np.ones((1, 1, 5, 5))
    ))
    player_w = np.zeros((1, 16, 5, 5), np.float32)
    player_w[0, 12, :2] = 1
    player_w[0, 9, 2:] = 1
    player_image = sr(conv("x0", player_w, [-24]))

    clear_background = sr(sa(
        conv(goal_visible, np.full((1, 1, 1, 1), -1), [1]),
        conv(wall_pixels, np.full((1, 1, 1, 1), -1))
    ))
    unknown_pixels = sm(background_five, clear_background)

    feature_stack = sconcat([
        wall_pixels, modifiers, pushers, player_image,
        unknown_pixels, goal_base, goal_colour_number
    ])
    feature_weights = np.array(
        [-11, -9, -8, -7, -6, -5, -4, -3, -2, -13, -12, 1, 1],
        np.float32
    ).reshape(1, 13, 1, 1)
    observed_codes = pool_centres(conv(
        feature_stack, feature_weights, [10]
    ))

    # Codes: 0 unknown, -1 wall, 10 floor, 1..8 objects, 20..115 goals.
    old_map = sl("mem", 0, N)
    old_target = sl("mem", N, N + 1)
    old_bar = sl("mem", N + 1, N + 2)
    old_factor = sl("mem", N + 2, N + 3)
    old_refill_index = sl("mem", N + 3, N + 4)
    old_refill_mode = sl("mem", N + 4, N + 5)

    unseen = eq(observed_codes, -2)
    player = eq(observed_codes, -3)
    ordinary_observation = sub(one_minus(unseen), player)
    keep_object = vm(
        above(old_map, 0.5),
        vm(one_minus(above(old_map, 8.5)), one_minus(eq(old_map, 4)))
    )
    occupied_code = va(
        constant(10), vm(keep_object, affine(old_map, 1, -10))
    )
    cached = vtotal([
        vm(observed_codes, ordinary_observation),
        vm(old_map, unseen),
        vm(occupied_code, player)
    ])

    seeds = [eq(cached, k) for k in (1, 2, 3)]
    refill_cells = eq(cached, 4)
    walls = above(neg(cached), 0.5)
    goals = above(cached, 19.5)
    unknown = eq(cached, 0)
    walk = one_minus(walls)
    all_modifiers = vtotal(seeds)
    no_goals = one_minus(goals)

    cached_image = to_map(cached)
    wall_image = sclip(conv(
        cached_image, np.full((1, 1, 1, 1), -2), [-1]
    ))
    goal_image = sclip(conv(
        cached_image, np.full((1, 1, 1, 1), 2), [-39]
    ))
    unknown_image = spatial_equal(cached_image, 0)
    ray_floor = sa(
        conv(wall_image, np.full((1, 1, 1, 1), -1), [1]),
        conv(goal_image, np.full((1, 1, 1, 1), -1))
    )

    # Nearest obstacle along launcher rays.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ray_w = np.zeros((4, 4, 3, 3), np.float32)
    for k, (dr, dc) in enumerate(directions):
        ray_w[k, k, 1 + dr, 1 + dc] = 1
    ray_state = fresh("ray")
    ray_definition = state(ray_state, zeros=[4, 64, 64])
    ray_body = []
    dst = ray_body
    ray_next = sm(sa(conv(ray_state, ray_w), constant(1)), ray_floor)
    dst = graph
    ray_loop = node(
        "loop", T=64, state=[ray_definition],
        body=ray_body, **{"next": {ray_state: ray_next}}
    )
    ray_distances = ray_loop + "." + ray_state

    pc = conv(cached_image, np.ones((4, 1, 1, 1)), -np.arange(5, 9))
    pc = node("abs", [pc])
    pc = sr(conv(
        pc, (-4 * np.eye(4)).reshape(4, 4, 1, 1), np.ones(4)
    ))
    displacement = sm(sa(ray_distances, constant(-3)), pc)
    displacement = conv(
        displacement,
        (np.array([-12, 12, -1, 1], np.float32) / 5).reshape(1, 4, 1, 1)
    )
    landing_indices = pool_centres(sa(index_image, displacement))
    landing_difference = node("abs", [
        node("sub", [index_image, landing_indices])
    ])
    landing_matrix = sr(conv(
        landing_difference,
        (-np.eye(N)).reshape(N, N, 1, 1), np.ones(N)
    ))

    adjacency = np.zeros((N, N), np.float32)
    for r in range(12):
        for c in range(12):
            i = 12 * r + c
            for dr, dc in directions:
                rr, cc = r + dr, c + dc
                if 0 <= rr < 12 and 0 <= cc < 12:
                    adjacency[i, 12 * rr + cc] = 1
    adjacency = parameter(adjacency)

    def backwards_update(reach, transit, seed):
        image = to_map(vm(reach, transit))
        settled = affine(
            node("gmean", [sm(landing_matrix, image)]), 4096 / 25
        )
        neighbours = dense(settled, adjacency)
        new_reach = hard_reach(va(seed, vm(neighbours, walk)))
        return new_reach, settled

    # Forward player distances and three modifier distance fields.
    transits = [
        vm(no_goals, va(one_minus(all_modifiers), seed))
        for seed in seeds
    ]
    wave_states = []
    reach_ids, acc_ids, nav_ids = [], [], []
    for seed in [player] + seeds:
        r, a = fresh("reach"), fresh("acc")
        reach_ids.append(r)
        acc_ids.append(a)
        wave_states.extend([
            state(r, init=seed), state(a, zeros=[N])
        ])
    for _ in seeds:
        q = fresh("nav")
        nav_ids.append(q)
        wave_states.append(state(q, zeros=[N]))

    wave_body = []
    dst = wave_body
    outgoing = dense(vm(reach_ids[0], no_goals), adjacency)
    spread_image = conv(sm(landing_matrix, outgoing), map_sum)
    arrived = pool_tiles(spread_image)
    nxt = {
        reach_ids[0]: hard_reach(va(player, vm(arrived, walk))),
        acc_ids[0]: va(acc_ids[0], reach_ids[0])
    }
    for k in range(3):
        rnext, settled = backwards_update(
            reach_ids[k + 1], transits[k], seeds[k]
        )
        nxt[reach_ids[k + 1]] = rnext
        nxt[acc_ids[k + 1]] = va(acc_ids[k + 1], reach_ids[k + 1])
        nxt[nav_ids[k]] = va(nav_ids[k], settled)
    dst = graph
    wave_loop = node(
        "loop", T=T, state=wave_states, body=wave_body, **{"next": nxt}
    )
    player_acc = wave_loop + "." + acc_ids[0]
    modifier_acc = [wave_loop + "." + x for x in acc_ids[1:]]
    modifier_nav = [wave_loop + "." + x for x in nav_ids]

    score_vector = va(affine(indices, -0.001, 1), player_acc)
    score_image = to_map(score_vector)
    old_index_match = eq(indices, old_target)

    def select_cells(candidate, candidate_image):
        best = node("gmax", [sm(score_image, candidate_image)])
        chosen = vm(
            candidate, vr(affine(sub(score_vector, best), 4000, 1))
        )
        held = vm(candidate, old_index_match)
        keep = vsum(held, N)
        return va(held, vm(one_minus(keep), chosen))

    chosen_goal = select_cells(goals, goal_image)
    chosen_unknown = select_cells(unknown, unknown_image)
    has_goal = clip(vsum(goals, N))
    unknown_reachable = clip(dot(unknown, player_acc, N))
    goal_code = affine(dot(cached, chosen_goal, N), 1, -20)
    goal_state = eq(state_indices, goal_code)

    attr_w = np.zeros((14, 96), np.float32)
    for s in range(6):
        for r in range(4):
            for c in range(4):
                i = (4 * s + r) * 4 + c
                attr_w[s, i] = 1
                attr_w[6 + c, i] = 1
                attr_w[10 + r, i] = 1
    attr_w = parameter(attr_w)
    current_attributes = dense(current_state, attr_w)
    goal_attributes = dense(goal_state, attr_w)
    matches = vm(current_attributes, goal_attributes)
    attribute_match = [
        vsum(sl(matches, 0, 6), 6),
        vsum(sl(matches, 6, 10), 4),
        vsum(sl(matches, 10, 14), 4)
    ]
    missing = clip(vtotal([
        vm(
            one_minus(attribute_match[k]),
            one_minus(clip(vsum(seeds[k], N)))
        )
        for k in range(3)
    ]))
    explore = va(
        one_minus(has_goal),
        vm(has_goal, vm(missing, unknown_reachable))
    )

    def reverse_field(target, transit):
        nonlocal dst
        r, a, n = fresh("gr"), fresh("ga"), fresh("gn")
        definitions = [
            state(r, init=target),
            state(a, zeros=[N]),
            state(n, zeros=[N])
        ]
        body = []
        dst = body
        next_r, settled = backwards_update(r, transit, target)
        update = {
            r: next_r,
            a: va(a, r),
            n: va(n, settled)
        }
        dst = graph
        loop = node(
            "loop", T=T, state=definitions,
            body=body, **{"next": update}
        )
        return loop + "." + a, loop + "." + n

    goal_transit = vm(
        one_minus(all_modifiers), clip(va(no_goals, chosen_goal))
    )
    goal_acc, goal_nav = reverse_field(chosen_goal, goal_transit)

    def distance(acc, source, minimum=0):
        value = dot(acc, source, N)
        negative_value = neg(value)
        unreachable = vr(va(negative_value, constant(1)))
        d = va(
            va(negative_value, constant(T)),
            affine(unreachable, 512)
        )
        if minimum:
            d = vmax(d, constant(minimum))
        return d

    inter_cost = []
    for j in range(3):
        inter_cost.append([
            constant(2) if j == k
            else distance(modifier_acc[k], seeds[j])
            for k in range(3)
        ])
    goal_cost = [distance(goal_acc, seed) for seed in seeds]
    player_cost = [
        distance(modifier_acc[k], player, 2) for k in range(3)
    ]
    player_goal_cost = distance(goal_acc, player)

    permutations = []
    for kind in range(3):
        w = np.zeros((96, 96), np.float32)
        for s in range(6):
            for r in range(4):
                for c in range(4):
                    ns, nr, nc = s, r, c
                    if kind == 0:
                        ns = (s + 1) % 6
                    elif kind == 1:
                        nc = (c + 1) % 4
                    else:
                        nr = (r + 1) % 4
                    i = (4 * s + r) * 4 + c
                    j = (4 * ns + nr) * 4 + nc
                    w[i, j] = 1
        permutations.append(parameter(w))

    # Bellman updates on modifier type x 96 attribute states.
    reward = affine(goal_state, 256)
    terminal = [vr(sub(reward, c)) for c in goal_cost]
    hs = [fresh("h") for _ in range(3)]
    hdefinitions = [state(hs[k], init=terminal[k]) for k in range(3)]
    bellman_body = []
    dst = bellman_body
    shifted = [dense(hs[k], permutations[k]) for k in range(3)]
    bellman_next = {}
    for j in range(3):
        value = terminal[j]
        for k in range(3):
            value = vmax(value, sub(shifted[k], inter_cost[j][k]))
        bellman_next[hs[j]] = vr(value)
    dst = graph
    bellman_loop = node(
        "loop", T=24, state=hdefinitions,
        body=bellman_body, **{"next": bellman_next}
    )

    q_values = [
        sub(dot(current_state, reward, 96), player_goal_cost)
    ]
    for k in range(3):
        h = dense(bellman_loop + "." + hs[k], permutations[k])
        q_values.append(affine(
            sub(dot(current_state, h, 96), player_cost[k]),
            1, -0.01 * (k + 1)
        ))
    best_q = q_values[0]
    for q in q_values[1:]:
        best_q = vmax(best_q, q)
    choices = [
        vr(affine(sub(q, best_q), 400, 1)) for q in q_values
    ]
    normal_navigation = vtotal([
        vm(choices[0], goal_nav),
        *[vm(choices[k + 1], modifier_nav[k]) for k in range(3)]
    ])

    # Read energy and infer its decrement from consecutive observations.
    bar_region = point_or_box(61, 33.5, 2, 84)
    bar = affine(
        node("gmean", [sm(ss("x0", 11, 12), bar_region)]), 4096
    )
    delta = sub(old_bar, bar)
    factor = va(old_factor, affine(eq(old_factor, 0), 0.5))
    dec_one, dec_two = eq(delta, 1), eq(delta, 2)
    factor = vtotal([
        vm(factor, sub(one_minus(dec_one), dec_two)),
        dec_one,
        affine(dec_two, 0.5)
    ])
    energy_budget = vm(bar, factor)

    # Prefer a refill that also advances the current symbolic plan.
    onward_value = vtotal([
        vm(choices[0], goal_acc),
        *[vm(choices[k + 1], modifier_acc[k]) for k in range(3)]
    ])
    refill_score = vtotal([
        player_acc, onward_value, affine(indices, -0.001, 1)
    ])
    reachable_refills = vm(refill_cells, clip(player_acc))
    feasible_refills = vm(
        reachable_refills,
        above(va(player_acc, energy_budget), T - 0.5)
    )
    any_feasible = clip(vsum(feasible_refills, N))
    refill_candidates = va(
        feasible_refills,
        vm(one_minus(any_feasible), reachable_refills)
    )
    best_refill_score = node("gmax", [
        to_map(vm(refill_score, refill_candidates))
    ])
    new_refill = vm(
        refill_candidates,
        vr(affine(sub(refill_score, best_refill_score), 4000, 1))
    )

    # Commitment survives changes in distances or in the chosen modifier.
    held_refill = vm(
        old_refill_mode,
        vm(reachable_refills, eq(indices, old_refill_index))
    )
    holding_refill = clip(vsum(held_refill, N))
    chosen_refill = va(
        held_refill, vm(one_minus(holding_refill), new_refill)
    )
    refill_distance = distance(player_acc, chosen_refill)
    refill_exists = clip(vsum(chosen_refill, N))

    remaining_plan = affine(best_q, -1, 256)
    can_finish = vm(
        one_minus(explore),
        above(sub(energy_budget, remaining_plan), 1)
    )
    refill_urgent = above(
        sub(affine(refill_distance, 1, 2), energy_budget), 0
    )
    request_refill = vm(
        refill_exists, vm(refill_urgent, one_minus(can_finish))
    )
    use_refill = vmax(holding_refill, request_refill)
    special_mode = clip(va(use_refill, explore))

    # Special routes permit useful incidental modifier crossings.
    special_target = va(
        vm(use_refill, chosen_refill),
        vm(one_minus(use_refill), chosen_unknown)
    )
    special_transit = clip(va(no_goals, special_target))
    _, special_navigation = reverse_field(special_target, special_transit)
    navigation = va(
        vm(one_minus(special_mode), normal_navigation),
        vm(special_mode, special_navigation)
    )

    # Read the four settled successor values at the player's location.
    nav_image = to_map(navigation)
    first_shift = np.zeros((4, 1, 7, 7), np.float32)
    second_shift = np.zeros((4, 4, 5, 5), np.float32)
    for k, (dr, dc) in enumerate(directions):
        first_shift[k, 0, 3 + 3 * dr, 3 + 3 * dc] = 1
        second_shift[k, k, 2 + 2 * dr, 2 + 2 * dc] = 1
    neighbour_values = conv(conv(nav_image, first_shift), second_shift)
    action_values = node("gmax", [sm(neighbour_values, player_image)])

    output_w = np.zeros((8, 4), np.float32)
    output_b = np.full(8, -1000, np.float32)
    for k in range(4):
        output_w[k + 1, k] = 1
        output_b[k + 1] = 0
    dense(action_values, output_w, output_b, name="logits")
    conv("coords", np.zeros((1, 3, 1, 1)), name="click")

    remembered_target = va(
        vm(has_goal, chosen_goal),
        vm(one_minus(has_goal), chosen_unknown)
    )
    target_index = dot(indices, remembered_target, N)
    refill_index = dot(indices, chosen_refill, N)
    node(
        "vconcat",
        [cached, target_index, bar, factor, refill_index, use_refill],
        name="mem_next"
    )

    # Remove dead graph construction intermediates and unused parameters.
    top_by_id = {nd["id"]: nd for nd in graph}
    needed = set()

    def require(ref):
        root = ref.split(".", 1)[0]
        if root in needed or root not in top_by_id:
            return
        needed.add(root)
        nd = top_by_id[root]
        if nd["op"] != "loop":
            for x in nd["in"]:
                require(x)
            return
        local = {s["id"] for s in nd["state"]}
        local.update(bn["id"] for bn in nd["body"])
        for s in nd["state"]:
            if "init" in s:
                require(s["init"])
        for bn in nd["body"]:
            for x in bn.get("in", []):
                if x not in local:
                    require(x)

    for output in ("logits", "click", "mem_next"):
        require(output)
    graph = [nd for nd in graph if nd["id"] in needed]

    used_parameters = set()
    for nd in graph:
        members = nd["body"] if nd["op"] == "loop" else [nd]
        for member in members:
            for field in ("w", "b"):
                if field in member:
                    used_parameters.add(member[field])
    params = {k: v for k, v in params.items() if k in used_parameters}

    return {"graph": graph, "memory_size": M, "params": params}
