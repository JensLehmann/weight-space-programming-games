import numpy as np


def build():
    PLAN = 24
    HISTORY = 32
    MEMORY = PLAN + 1 + 3 * HISTORY

    params = {}
    parameter_cache = {}
    constants = {}
    graph = []
    body = []
    current = graph
    serial = 0
    outer_cache = {}
    inner_cache = {}

    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "mem": ("v", MEMORY),
        "aprev": ("v", 8),
        "avail": ("v", 8),
    }

    def parameter(value):
        value = np.asarray(value, dtype=np.float32)
        key = (value.shape, value.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = value
        return parameter_cache[key]

    def node(op, inputs, name=None, **fields):
        nonlocal serial
        inputs = list(inputs)
        key = (op, tuple(inputs), tuple(sorted(fields.items())))
        cache = inner_cache if current is body else outer_cache

        if name is None:
            if key in outer_cache:
                return outer_cache[key]
            if key in cache:
                return cache[key]
            name = "n%d" % serial
            serial += 1
            cache[key] = name

        nd = {"id": name, "op": op, "in": inputs}
        nd.update(fields)
        current.append(nd)

        if op in ("conv", "dense"):
            weight = params[fields["w"]]
            shape = ("s" if op == "conv" else "v", weight.shape[0])
        elif op in ("gmax", "gmean"):
            shape = ("v", shapes[inputs[0]][1])
        elif op in ("concat", "vconcat"):
            shape = (
                shapes[inputs[0]][0],
                sum(shapes[i][1] for i in inputs),
            )
        elif op in ("slice", "vslice"):
            shape = (
                shapes[inputs[0]][0],
                fields["stop"] - fields["start"],
            )
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = ("s", max(shapes[i][1] for i in inputs))
        elif op in ("vadd", "vmul"):
            shape = ("v", max(shapes[i][1] for i in inputs))
        else:
            shape = shapes[inputs[0]]

        shapes[name] = shape
        return name

    def lin(x, weight, bias=None, name=None):
        weight = np.asarray(weight, np.float32)
        if weight.ndim == 1:
            weight = weight[None, :]
        spatial = shapes[x][0] == "s"
        if spatial:
            weight = weight[:, :, None, None]
        fields = {"w": parameter(weight)}
        if bias is not None:
            bias = np.asarray(bias, np.float32)
            if bias.ndim == 0:
                bias = np.full(weight.shape[0], float(bias), np.float32)
            fields["b"] = parameter(bias)
        return node(
            "conv" if spatial else "dense", [x], name=name, **fields
        )

    def scale(x, a=1.0, b=0.0):
        if a == 1 and b == 0:
            return x
        n = shapes[x][1]
        return lin(
            x,
            np.eye(n, dtype=np.float32) * a,
            None if b == 0 else np.full(n, b, np.float32),
        )

    def const(value):
        nonlocal current
        value = float(value)
        if value not in constants:
            saved = current
            current = graph
            constants[value] = lin("avail", np.zeros((1, 8)), [value])
            current = saved
        return constants[value]

    def sl(x, start, stop):
        return node(
            "slice" if shapes[x][0] == "s" else "vslice",
            [x], start=start, stop=stop,
        )

    def cat(*xs):
        return node(
            "concat" if shapes[xs[0]][0] == "s" else "vconcat", xs
        )

    def add(x, y, name=None):
        spatial = shapes[x][0] == "s" or shapes[y][0] == "s"
        return node("add" if spatial else "vadd", [x, y], name=name)

    def mul(x, y):
        spatial = shapes[x][0] == "s" or shapes[y][0] == "s"
        return node("mul" if spatial else "vmul", [x, y])

    def sub(x, y):
        if shapes[x][0] == "s" or shapes[y][0] == "s":
            return node("sub", [x, y])
        return add(x, scale(y, -1))

    def relu(x):
        return node("relu" if shapes[x][0] == "s" else "vrelu", [x])

    def hinges(x, offsets, coefficients):
        n = shapes[x][1]
        eye = np.eye(n, dtype=np.float32)
        expanded = lin(
            x,
            np.concatenate([eye] * len(offsets), axis=0),
            np.concatenate([
                np.full(n, a, np.float32) for a in offsets
            ]),
        )
        return lin(
            relu(expanded),
            np.concatenate([a * eye for a in coefficients], axis=1),
        )

    def clip(x):
        if shapes[x][0] == "s":
            return node("min2", [relu(x), const(1)])
        return hinges(x, (0, -1), (1, -1))

    def absolute(x):
        if shapes[x][0] == "s":
            return node("abs", [x])
        return add(relu(x), relu(scale(x, -1)))

    def eq(x, y=None, hard=False):
        if y is not None:
            x = sub(x, y)
        if shapes[x][0] == "v":
            if hard:
                return hinges(
                    x, (.75, .25, -.25, -.75), (2, -2, -2, 2)
                )
            return hinges(x, (1, 0, -1), (1, -2, 1))
        if hard:
            return clip(scale(absolute(x), -2, 1.5))
        return relu(scale(absolute(x), -1, 1))

    def onehot(x, values, hard=False):
        values = np.asarray(list(values), np.float32)
        if shapes[x][0] == "v":
            offsets = (
                (.75, .25, -.25, -.75) if hard else (1, 0, -1)
            )
            coefficients = (2, -2, -2, 2) if hard else (1, -2, 1)
            n = len(values)
            expanded = lin(
                x,
                np.ones((n * len(offsets), 1), np.float32),
                np.concatenate([a - values for a in offsets]),
            )
            eye = np.eye(n, dtype=np.float32)
            return lin(
                relu(expanded),
                np.concatenate([
                    a * eye for a in coefficients
                ], axis=1),
            )
        z = lin(x, np.ones((len(values), 1), np.float32), -values)
        return eq(z, hard=hard)

    def sumc(x):
        return lin(x, np.ones((1, shapes[x][1]), np.float32))

    def gm(x):
        return node("gmax", [x])

    def rs(x):
        return node("rowsum", [x])

    def cs(x):
        return node("colsum", [x])

    def maximum(x, y):
        return node("max2", [x, y])

    def mix(a, b, select_b):
        return add(
            mul(a, scale(select_b, -1, 1)),
            mul(b, select_b),
        )

    def quantize(x):
        return lin(
            onehot(x, range(9), hard=True),
            np.arange(9, dtype=np.float32)[None, :],
        )

    # Board and control recognition.
    one = lin("coords", [[0, 0, 1]])
    zero = scale(one, 0)
    image_xy = lin("coords", [[63, 0, 0], [0, 63, 0]])
    image_y = sl(image_xy, 0, 1)
    image_x = sl(image_xy, 1, 2)
    upper = clip(scale(image_y, -1, 53))

    w = np.zeros((1, 16, 3, 3), np.float32)
    for dy0 in (0, 1):
        for dx0 in (0, 1):
            w[0, 4, 1 + dy0, 1 + dx0] = 1
    background = relu(node(
        "conv", ["x0"], w=parameter(w), b=parameter([-3])
    ))
    background = mul(background, upper)

    bx = scale(
        gm(mul(background, scale(image_x, -1, 64))), -1, 64
    )
    by = scale(
        gm(mul(background, scale(image_y, -1, 64))), -1, 64
    )
    ex = scale(gm(mul(background, image_x)), 1, 2)
    ey = scale(gm(mul(background, image_y)), 1, 2)
    width = quantize(scale(sub(ex, bx), 1 / 6))
    height = quantize(scale(sub(ey, by), 1 / 6))

    w = np.zeros((3, 16, 3, 3), np.float32)
    for out, color in enumerate((6, 15, 11)):
        for dy0 in (0, 1):
            for dx0 in (0, 1):
                w[out, color, 1 + dy0, 1 + dx0] = 1
    centers = relu(node(
        "conv", ["x0"], w=parameter(w),
        b=parameter([-3, -3, -3]),
    ))
    all_centers = sumc(centers)
    active_centers = sumc(sl(centers, 0, 2))
    center_color = lin(centers, [[6, 15, 11]])

    hcenter = mul(
        mul(active_centers, clip(sub(bx, image_x))), upper
    )
    vcenter = mul(
        mul(all_centers, clip(sub(by, image_y))), upper
    )
    has_h = gm(hcenter)
    has_v = gm(mul(vcenter, active_centers))
    hcolor = gm(mul(hcenter, center_color))
    vcolor = gm(mul(vcenter, center_color))

    hr = quantize(scale(sub(
        scale(gm(mul(hcenter, image_y)), 1, -2), by
    ), 1 / 6))
    vc = quantize(scale(sub(
        scale(gm(mul(vcenter, image_x)), 1, -2), bx
    ), 1 / 6))

    w = np.zeros((1, 16, 3, 3), np.float32)
    w[0, 0, 0, 1] = 1
    ring_zero = node("conv", ["x0"], w=parameter(w))
    selected_v = gm(mul(
        mul(vcenter, active_centers), ring_zero
    ))

    palette = np.zeros((1, 16), np.float32)
    for color in (7, 8, 9, 10, 12, 13, 14):
        palette[0, color] = color
    color_image = lin("x0", palette)

    lower_triangle = np.tril(np.ones((8, 8), np.float32))
    bottom = clip(scale(image_y, 1, -53))
    target_row = onehot(image_y, [57], hard=True)

    def read_target(control_color, enabled):
        marker = mul(
            mul(all_centers, bottom),
            eq(center_color, control_color, hard=True),
        )
        origin = gm(mul(marker, image_x))
        columns = onehot(
            sub(image_x, origin),
            [5 + 6 * k for k in range(8)],
            hard=True,
        )
        colors = gm(mul(mul(columns, target_row), color_image))
        prefix = relu(lin(
            clip(colors), lower_triangle,
            1 - np.arange(1, 9, dtype=np.float32),
        ))
        prefix = mul(prefix, enabled)
        return mul(colors, prefix), sumc(prefix)

    target_h, nh = read_target(hcolor, has_h)
    target_v, nv = read_target(vcolor, has_v)
    enabled_h = clip(nh)
    enabled_v = clip(nv)

    # Flattened canonical 8x8 board, broadcast over hypothesis rows.
    column_basis = onehot(image_x, range(64), hard=True)
    row_basis = onehot(image_y, range(64), hard=True)
    gx_values = np.arange(64) % 8
    gy_values = np.arange(64) // 8
    gx = lin(column_basis, gx_values[None, :])
    gy = lin(column_basis, gy_values[None, :])
    time_columns = lin(column_basis, np.arange(64)[None, :])

    x_basis = lin(
        column_basis,
        (np.arange(8)[:, None] ==
         gx_values[None, :]).astype(np.float32),
    )
    y_basis = lin(
        column_basis,
        (np.arange(8)[:, None] ==
         gy_values[None, :]).astype(np.float32),
    )

    local_xy = cat(sub(image_y, by), sub(image_x, bx))
    aw = np.zeros((16, 2), np.float32)
    aw[:8, 0] = 1
    aw[8:, 1] = 1
    stripes = eq(lin(
        local_xy, aw,
        -np.tile(2 + 6 * np.arange(8), 2),
    ), hard=True)

    cw = np.zeros((64, 16), np.float32)
    for j in range(64):
        cw[j, j // 8] = 1
        cw[j, 8 + j % 8] = 1
    cell_centers = relu(lin(stripes, cw, -1))

    def extract(image):
        values = gm(mul(cell_centers, image))
        return sumc(mul(column_basis, values))

    valid_cells = mul(
        clip(sub(width, gx)), clip(sub(height, gy))
    )
    initial_b = mul(extract(color_image), valid_cells)
    obstacle = mul(extract(sl("x0", 5, 6)), valid_cells)
    free = mul(valid_cells, scale(obstacle, -1, 1))

    w = np.zeros((2, 16, 5, 5), np.float32)
    for color in (1, 2, 3):
        w[0, color, 2, 0] = 1
        w[1, color, 0, 2] = 1
    rail_pixels = node("conv", ["x0"], w=parameter(w))
    hrail = mul(extract(sl(rail_pixels, 0, 1)), valid_cells)
    vrail = mul(extract(sl(rail_pixels, 1, 2)), valid_cells)

    initial_hr = add(zero, hr)
    initial_vc = add(zero, vc)
    initial_hl = rs(mul(hrail, eq(gy, hr)))
    initial_vl = rs(mul(vrail, eq(gx, vc)))

    # A unique shared piece provides a geometric anchor for both sequences.
    common_root = mul(
        clip(initial_b),
        mul(
            clip(sumc(eq(initial_b, target_h))),
            clip(sumc(eq(initial_b, target_v))),
        ),
    )
    common_count = gm(rs(common_root))
    geometric_enabled = eq(common_count, const(1))
    anchor_color = gm(rs(mul(common_root, initial_b)))

    target_indices = lin(
        "avail", np.zeros((8, 8), np.float32),
        np.arange(8, dtype=np.float32),
    )

    def role_geometry(target):
        pivot_index = sumc(mul(
            eq(target, anchor_color), target_indices
        ))
        offsets = sub(target_indices, pivot_index)
        return (
            offsets,
            clip(scale(offsets, -1)),
            clip(offsets),
            clip(target),
        )

    geometry_h = role_geometry(target_h)
    geometry_v = role_geometry(target_v)

    # Weaken greedy prefix commitment when a shared-anchor constraint exists.
    value_weights = lin(
        geometric_enabled,
        np.array([-440, 0, 40, 0, -4, 0, 0], np.float32)[:, None],
        [600, 40, -80, -2, -4, -6, -1],
    )
    history_weight = lin(
        cat(gm(obstacle), geometric_enabled),
        [[220, 100]], [90],
    )

    # Eight moves are expanded from each of eight retained hypotheses.
    macro = np.arange(64) % 8
    direction = macro % 4
    vflag = lin(
        row_basis, (macro >= 4).astype(np.float32)[None, :]
    )
    hflag = scale(vflag, -1, 1)
    dx = lin(
        row_basis,
        np.array([0, 0, -1, 1], np.float32)[direction][None, :],
    )
    dy = lin(
        row_basis,
        np.array([-1, 1, 0, 0], np.float32)[direction][None, :],
    )
    hdx, hdy = mul(dx, hflag), mul(dy, hflag)
    vdx, vdy = mul(dx, vflag), mul(dy, vflag)
    action_code = lin(row_basis, (macro + 1)[None, :])
    enabled_action = add(mul(hflag, has_h), mul(vflag, has_v))

    dw = np.stack([
        (direction == d).astype(np.float32)
        for d in (2, 3, 0, 1)
    ])
    direction_flags = lin(row_basis, dw)
    reverse_flags = lin(row_basis, dw[[1, 0, 3, 2]])
    horizontal = sumc(sl(direction_flags, 0, 2))
    vertical = sumc(sl(direction_flags, 2, 4))
    parallel = add(
        mul(hflag, horizontal), mul(vflag, vertical)
    )
    perpendicular = scale(parallel, -1, 1)
    lock_h_factor = mul(vflag, vertical)
    lock_v_factor = mul(hflag, horizontal)

    direction_bounds = sumc(mul(cat(
        clip(gx),
        clip(scale(sub(width, gx), 1, -1)),
        clip(gy),
        clip(scale(sub(height, gy), 1, -1)),
    ), direction_flags))

    groups = lin(
        row_basis,
        (np.arange(8)[:, None] ==
         (np.arange(64) // 8)[None, :]).astype(np.float32),
    )
    tie_break = lin(
        row_basis, (-np.arange(64) / 128.0)[None, :]
    )

    wh = np.zeros((2, 1, 3, 3), np.float32)
    wh[0, 0, 1, 0] = 1
    wh[1, 0, 1, 2] = 1
    wv1 = np.zeros((2, 1, 7, 7), np.float32)
    wv1[0, 0, 3, 0] = 1
    wv1[1, 0, 3, 6] = 1
    wv2 = np.zeros((2, 2, 7, 7), np.float32)
    wv2[0, 0, 3, 0] = 1
    wv2[1, 1, 3, 6] = 1
    wv3 = np.zeros((2, 2, 5, 5), np.float32)
    wv3[0, 0, 2, 0] = 1
    wv3[1, 1, 2, 4] = 1

    def gather(x, reverse=False):
        h = node("conv", [x], w=parameter(wh))
        v = node("conv", [x], w=parameter(wv1))
        v = node("conv", [v], w=parameter(wv2))
        v = node("conv", [v], w=parameter(wv3))
        return sumc(mul(
            cat(h, v),
            reverse_flags if reverse else direction_flags,
        ))

    def rods(r, lh, c, lv):
        return (
            mul(eq(gy, r), clip(sub(lh, gx))),
            mul(eq(gx, c), clip(sub(lv, gy))),
        )

    def signatures(beads, r, lh, c, lv):
        a = rs(mul(beads, scale(gx, 1, 1)))
        b = rs(mul(beads, scale(gy, 1, 1)))
        return (
            lin(cat(a, r, lh, c, lv),
                [[1, 131, 977, 67, 139]]),
            lin(cat(b, r, lh, c, lv),
                [[1, 503, 53, 89, 431]]),
        )

    history_a = sl("mem", PLAN + 1, PLAN + 1 + HISTORY)
    history_b = sl(
        "mem", PLAN + 1 + HISTORY, PLAN + 1 + 2 * HISTORY
    )
    history_valid = sl(
        "mem", PLAN + 1 + 2 * HISTORY, MEMORY
    )
    root_a, root_b = signatures(
        initial_b, initial_hr, initial_hl, initial_vc, initial_vl
    )
    root_a_v, root_b_v = gm(root_a), gm(root_b)

    states = [
        {"id": "sb", "init": initial_b},
        {"id": "shr", "init": initial_hr},
        {"id": "shl", "init": initial_hl},
        {"id": "svc", "init": initial_vc},
        {"id": "svl", "init": initial_vl},
        {"id": "spath", "init": zero},
        {"id": "sdepth", "init": zero},
        {"id": "sactive", "init": add(zero, selected_v)},
        {"id": "scost", "init": zero},
        {"id": "sbestpath", "init": zero},
        {"id": "sbestlen", "init": zero},
        {"id": "sbestscore", "init": scale(one, 0, -200000)},
    ]
    for state in states:
        shapes[state["id"]] = shapes[state["init"]]

    current = body

    # Simultaneous hypothetical transitions.
    r1, lh1 = add("shr", hdy), add("shl", hdx)
    c1, lv1 = add("svc", vdx), add("svl", vdy)
    old_h, old_v = rods("shr", "shl", "svc", "svl")
    new_h, new_v = rods(r1, lh1, c1, lv1)
    old_rod = add(mul(old_h, hflag), mul(old_v, vflag))
    new_rod = add(mul(new_h, hflag), mul(new_v, vflag))

    violations = [
        clip(scale(lh1, -1)),
        clip(sub(lh1, width)),
        clip(scale(lv1, -1)),
        clip(sub(lv1, height)),
        clip(scale(r1, -1)),
        clip(scale(sub(r1, height), 1, 1)),
        clip(scale(c1, -1)),
        clip(scale(sub(c1, width), 1, 1)),
        clip(rs(mul(new_rod, obstacle))),
    ]
    bad = violations[0]
    for value in violations[1:]:
        bad = maximum(bad, value)

    occupied = clip("sb")
    seed = mul(occupied, maximum(old_rod, new_rod))
    lock = add(
        mul(old_h, lock_h_factor), mul(old_v, lock_v_factor)
    )
    destination_ok = mul(gather(free), direction_bounds)
    destination_ok = mul(
        destination_ok, scale(lock, -1, 1)
    )
    destination_ok = mul(
        destination_ok, scale(gather(lock), -1, 1)
    )
    blocked = mul(occupied, scale(destination_ok, -1, 1))

    for _ in range(7):
        blocked = maximum(
            blocked, mul(occupied, gather(blocked))
        )

    bad = maximum(
        bad, mul(clip(rs(mul(seed, blocked))), perpendicular)
    )
    legal = mul(scale(bad, -1, 1), enabled_action)
    moving = mul(seed, scale(blocked, -1, 1))

    for _ in range(7):
        moving = maximum(
            moving, mul(occupied, gather(moving, reverse=True))
        )

    proposed_b = add(
        mul("sb", scale(moving, -1, 1)),
        gather(mul("sb", moving), reverse=True),
    )
    next_b = mix("sb", proposed_b, legal)
    next_r = mix("shr", r1, legal)
    next_lh = mix("shl", lh1, legal)
    next_c = mix("svc", c1, legal)
    next_lv = mix("svl", lv1, legal)
    next_active = mix("sactive", vflag, legal)
    next_cost = add(
        "scost", scale(absolute(sub(vflag, "sactive")), 1, 1)
    )

    def belt_value(beads, belt, other, position, length,
                   other_length, target, count, enabled,
                   is_vertical):
        along = gy if is_vertical else gx
        across = gx if is_vertical else gy
        basis = y_basis if is_vertical else x_basis
        on = mul(clip(beads), belt)

        occupancy = rs(mul(on, basis))
        rank = sumc(mul(lin(occupancy, lower_triangle), basis))
        rank_hot = onehot(rank, range(1, 9))
        wanted_here = sumc(mul(rank_hot, target))
        matched = mul(on, eq(beads, wanted_here))
        matched_rank = rs(mul(matched, rank_hot))
        prefix = sumc(relu(lin(
            matched_rank, lower_triangle,
            1 - np.arange(1, 9, dtype=np.float32),
        )))

        wanted_next = sumc(mul(
            onehot(scale(prefix, 1, 1), range(1, 9)), target
        ))
        candidates = mul(
            mul(eq(beads, wanted_next), clip(beads)),
            scale(belt, -1, 1),
        )

        transverse = absolute(sub(across, position))
        extending = relu(scale(sub(along, length), 1, 1))
        shortening = mul(
            relu(sub(length, along)),
            scale(eq(across, position), -1, 1),
        )
        pinned = mul(other, relu(sub(other_length, across)))
        distance = add(add(transverse, extending), shortening)
        distance = add(distance, scale(pinned, 2))
        nearest = node("rowmax", [
            mul(scale(distance, -1, 100), candidates)
        ])
        acquisition = mul(
            scale(nearest, -1, 100), clip(sub(count, prefix))
        )

        features = cat(
            prefix,
            rs(matched),
            sub(rs(on), prefix),
            rs(mul(matched, along)),
            acquisition,
            rs(mul(on, other)),
            length,
        )
        value = sumc(mul(features, value_weights))
        return mul(value, enabled), prefix

    value_h, prefix_h = belt_value(
        next_b, new_h, new_v, r1, lh1, lv1,
        target_h, nh, enabled_h, False,
    )
    value_v, prefix_v = belt_value(
        next_b, new_v, new_h, c1, lv1, lh1,
        target_v, nv, enabled_v, True,
    )

    # Geometric distance to an ordered cross through the shared piece.
    anchor_mask = mul(
        clip(next_b), eq(next_b, anchor_color)
    )
    anchor_x = rs(mul(anchor_mask, gx))
    anchor_y = rs(mul(anchor_mask, gy))

    def formation_cost(target, geometry, is_vertical):
        offsets, before, after, valid_roles = geometry
        along = gy if is_vertical else gx
        across = gx if is_vertical else gy
        anchor_along = anchor_y if is_vertical else anchor_x
        anchor_across = anchor_x if is_vertical else anchor_y

        delta = add(sub(anchor_along, along), offsets)
        order_cost = add(
            mul(relu(delta), after),
            mul(relu(scale(delta, -1)), before),
        )
        distance = add(
            order_cost, absolute(sub(across, anchor_across))
        )
        candidates = mul(
            eq(next_b, target), clip(next_b)
        )
        nearest = node("rowmax", [
            mul(scale(distance, -1, 32), candidates)
        ])
        return sumc(mul(
            scale(nearest, -1, 32), valid_roles
        ))

    formation = add(
        formation_cost(target_h, geometry_h, False),
        formation_cost(target_v, geometry_v, True),
    )
    formation = mul(formation, geometric_enabled)

    signature_a, signature_b = signatures(
        next_b, next_r, next_lh, next_c, next_lv
    )
    revisited = mul(
        eq(signature_a, history_a),
        eq(signature_b, history_b),
    )
    revisited = clip(sumc(mul(revisited, history_valid)))

    depth1 = scale("sdepth", 1, 1)
    done_h = scale(clip(sub(nh, prefix_h)), -1, 1)
    done_v = scale(clip(sub(nv, prefix_v)), -1, 1)
    score = add(value_h, value_v)
    score = sub(score, scale(formation, 80))
    score = add(score, scale(mul(done_h, done_v), 20000))
    score = sub(score, next_cost)
    score = sub(score, mul(revisited, history_weight))
    score = sub(score, scale(scale(legal, -1, 1), 100000))
    score = add(score, tie_break)

    candidate_path = mix(
        "spath", action_code, eq(time_columns, "sdepth")
    )

    def score_equal(x, maximum_value):
        return relu(scale(
            absolute(sub(x, maximum_value)), -512, 1
        ))

    global_maximum = gm(score)
    global_choice = score_equal(score, global_maximum)
    chosen_path = cs(mul(candidate_path, global_choice))
    improved = clip(scale(
        sub(global_maximum, "sbestscore"), 512
    ))
    best_path = mix("sbestpath", chosen_path, improved)
    best_length = mix("sbestlen", depth1, improved)
    best_score = maximum("sbestscore", global_maximum)

    # Retain one winner in each of eight geometry niches.
    hash_value = lin(
        cat(next_r, next_lh, next_c, next_lv), [[1, 2, 3, 1]]
    )
    quotient = sumc(clip(lin(
        hash_value, np.ones((7, 1), np.float32),
        -(7 + 8 * np.arange(7, dtype=np.float32)),
    )))
    remainder = sub(hash_value, scale(quotient, 8))
    niches = mul(onehot(remainder, range(8)), legal)

    niche_score = sub(
        score, scale(scale(niches, -1, 1), 200000)
    )
    niche_choice = score_equal(niche_score, gm(niche_score))
    selection = mix(global_choice, niche_choice, gm(niches))

    def select_parent(field):
        return sumc(mul(cs(mul(field, selection)), groups))

    parent_fields = [
        select_parent(field)
        for field in (
            next_b, next_r, next_lh, next_c, next_lv,
            candidate_path, next_active, next_cost,
        )
    ]
    next_map = {
        "sb": parent_fields[0],
        "shr": parent_fields[1],
        "shl": parent_fields[2],
        "svc": parent_fields[3],
        "svl": parent_fields[4],
        "spath": parent_fields[5],
        "sactive": parent_fields[6],
        "scost": parent_fields[7],
        "sdepth": depth1,
        "sbestpath": best_path,
        "sbestlen": best_length,
        "sbestscore": best_score,
    }

    current = graph
    loop = {
        "id": "plan",
        "op": "loop",
        "T": 1,
        "state": states,
        "body": body,
        "next": next_map,
    }
    graph.append(loop)
    for state in states:
        shapes["plan." + state["id"]] = shapes[state["id"]]

    # Plan execution and recurrent history.
    fresh_plan = gm(mul(
        sl(column_basis, 0, PLAN), "plan.sbestpath"
    ))
    fresh_length = gm("plan.sbestlen")
    stored_plan = sl("mem", 0, PLAN)
    stored_length = sl("mem", PLAN, PLAN + 1)

    same_a = eq(root_a_v, sl(history_a, 0, 1))
    same_b = eq(root_b_v, sl(history_b, 0, 1))
    previous_direction = sumc(sl("aprev", 1, 5))
    failed_move = mul(
        mul(mul(same_a, same_b), sl(history_valid, 0, 1)),
        previous_direction,
    )
    follow = mul(
        clip(stored_length), scale(failed_move, -1, 1)
    )
    chosen_plan = mix(fresh_plan, stored_plan, follow)
    chosen_length = mix(fresh_length, stored_length, follow)

    action_hot = onehot(
        sl(chosen_plan, 0, 1), range(1, 9), hard=True
    )
    want_v = sumc(sl(action_hot, 4, 8))
    switch = add(
        mul(want_v, scale(selected_v, -1, 1)),
        mul(scale(want_v, -1, 1), selected_v),
    )

    output_weight = np.zeros((8, 9), np.float32)
    for k in range(8):
        output_weight[1 + k % 4, k] = 5
    output_weight[6, 8] = 10
    lin(
        cat(action_hot, switch), output_weight,
        [-100, 0, 0, 0, 0, -100, 0, -100],
        name="logits",
    )
    add(
        mul(hcenter, scale(want_v, -1, 1)),
        mul(vcenter, want_v),
        name="click",
    )

    shift_plan = np.zeros((PLAN, PLAN), np.float32)
    for k in range(PLAN - 1):
        shift_plan[k, k + 1] = 1
    remaining_plan = mix(
        lin(chosen_plan, shift_plan), chosen_plan, switch
    )
    remaining_length = mix(
        relu(scale(chosen_length, 1, -1)),
        chosen_length, switch,
    )

    history_shift = np.zeros((HISTORY, HISTORY), np.float32)
    for k in range(1, HISTORY):
        history_shift[k, k - 1] = 1
    insert = np.zeros((HISTORY, 1), np.float32)
    insert[0, 0] = 1

    def append_history(old, value):
        return add(
            lin(old, history_shift), lin(value, insert)
        )

    node("vconcat", [
        remaining_plan,
        remaining_length,
        append_history(history_a, root_a_v),
        append_history(history_b, root_b_v),
        append_history(history_valid, const(1)),
    ], name="mem_next")

    # Algebraic compilation only: compose adjacent affine maps.
    all_nodes = graph + body
    node_by_id = {nd["id"]: nd for nd in all_nodes}
    uses = {}
    for nd in all_nodes:
        if nd["op"] != "loop":
            for source in nd["in"]:
                uses[source] = uses.get(source, 0) + 1
    for state in states:
        uses[state["init"]] = uses.get(state["init"], 0) + 1
    for source in next_map.values():
        uses[source] = uses.get(source, 0) + 1

    protected = {"logits", "click", "mem_next"}
    for nd in all_nodes:
        if nd["op"] not in ("conv", "dense"):
            continue
        while True:
            source = nd["in"][0]
            previous = node_by_id.get(source)
            if (
                previous is None
                or previous["op"] != nd["op"]
                or uses.get(source, 0) != 1
                or source in protected
            ):
                break

            wa = params[previous["w"]]
            wb = params[nd["w"]]
            spatial = nd["op"] == "conv"
            if spatial:
                if wa.shape[-2:] != (1, 1) or wb.shape[-2:] != (1, 1):
                    break
                ma = wa[:, :, 0, 0]
                mb = wb[:, :, 0, 0]
            else:
                ma, mb = wa, wb

            ba = params[previous["b"]] if "b" in previous else np.zeros(
                ma.shape[0], np.float32
            )
            bb = params[nd["b"]] if "b" in nd else np.zeros(
                mb.shape[0], np.float32
            )
            wc = mb @ ma
            bc = mb @ ba + bb
            if spatial:
                wc = wc[:, :, None, None]
            nd["w"] = parameter(wc)
            if np.any(bc):
                nd["b"] = parameter(bc)
            else:
                nd.pop("b", None)
            nd["in"] = list(previous["in"])
            uses[source] = 0

    # Remove unreachable nodes and unused arrays after composition.
    outer_nodes = {nd["id"]: nd for nd in graph}
    inner_nodes = {nd["id"]: nd for nd in body}
    state_nodes = {state["id"]: state for state in states}
    live_outer = set()
    live_inner = set()
    inputs = {"x0", "coords", "mem", "aprev", "avail"}

    def mark_inner(name):
        if name in state_nodes:
            mark_outer(state_nodes[name]["init"])
            return
        if name not in inner_nodes:
            mark_outer(name)
            return
        if name in live_inner:
            return
        live_inner.add(name)
        for source in inner_nodes[name]["in"]:
            mark_inner(source)

    def mark_outer(name):
        if name in inputs:
            return
        if name.startswith("plan."):
            name = "plan"
        if name in live_outer:
            return
        live_outer.add(name)
        nd = outer_nodes[name]
        if nd["op"] == "loop":
            for state in states:
                mark_outer(state["init"])
            for source in next_map.values():
                mark_inner(source)
        else:
            for source in nd["in"]:
                mark_outer(source)

    for output in ("logits", "click", "mem_next"):
        mark_outer(output)

    graph = [nd for nd in graph if nd["id"] in live_outer]
    body = [nd for nd in body if nd["id"] in live_inner]
    loop["body"] = body

    used_parameters = set()
    for nd in graph + body:
        for field in ("w", "b"):
            if field in nd:
                used_parameters.add(nd[field])
    params = {
        name: value for name, value in params.items()
        if name in used_parameters
    }

    def cost(nd):
        op = nd["op"]
        if op == "loop":
            return 0
        if op == "conv":
            return int(params[nd["w"]].size) * 4096
        if op == "dense":
            return int(params[nd["w"]].size)
        if op in ("gmax", "gmean"):
            return shapes[nd["in"][0]][1] * 4096
        if op in (
            "add", "sub", "mul", "max2", "min2",
            "rowmax", "colmax", "rowsum", "colsum",
        ):
            return shapes[nd["id"]][1] * 4096
        if op in ("vadd", "vmul"):
            return shapes[nd["id"]][1]
        return 0

    outer_cost = sum(cost(nd) for nd in graph)
    inner_cost = sum(cost(nd) for nd in body)
    node_limit = (9800 - len(graph)) // max(1, len(body))
    mac_limit = (
        (1_450_000_000 - outer_cost) // max(1, inner_cost)
    )
    loop["T"] = int(max(
        1, min(PLAN, node_limit, mac_limit)
    ))

    spatial_channels = 19
    spatial_channels += sum(
        shapes[nd["id"]][1]
        for nd in graph + body
        if nd["op"] != "loop"
        and shapes[nd["id"]][0] == "s"
    )
    spatial_channels += 2 * sum(
        shapes[state["id"]][1]
        for state in states
        if shapes[state["id"]][0] == "s"
    )

    assert len(graph) + len(body) <= 1000
    assert spatial_channels <= 4096
    assert sum(value.size for value in params.values()) <= 150000
    assert outer_cost + loop["T"] * inner_cost <= 1500000000
    assert len(graph) + loop["T"] * len(body) <= 10000

    return {
        "graph": graph,
        "memory_size": MEMORY,
        "params": params,
    }
