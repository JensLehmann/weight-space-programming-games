import numpy as np


def build():
    graph = []
    params = {}
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (72,),
        "aprev": (8,),
        "avail": (8,),
    }
    current = graph
    serial = 0
    parameter_cache = {}
    constant_cache = {}
    expression_cache = {}

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p" + str(len(params))
            params[name] = a
            parameter_cache[key] = name
        return parameter_cache[key]

    def node(op, inputs, shape, name=None, where=None, **fields):
        destination = current if where is None else where
        key = (
            id(destination), op, tuple(inputs),
            tuple(sorted(fields.items()))
        )
        if name is None and key in expression_cache:
            return expression_cache[key]
        identifier = fresh() if name is None else name
        record = {"id": identifier, "op": op, "in": list(inputs)}
        record.update(fields)
        destination.append(record)
        shapes[identifier] = tuple(shape)
        if name is None:
            expression_cache[key] = identifier
        return identifier

    def conv(x, w, b=None, name=None, where=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node(
            "conv", [x], (w.shape[0], 64, 64),
            name=name, where=where, **fields
        )

    def dense(x, w, b=None, name=None, where=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node(
            "dense", [x], (w.shape[0],),
            name=name, where=where, **fields
        )

    def constant(value):
        a = np.asarray(value, np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key not in constant_cache:
            constant_cache[key] = dense(
                "aprev", np.zeros((len(a), 8), np.float32),
                a, where=graph
            )
        return constant_cache[key]

    def spatial(x):
        return len(shapes[x]) == 3

    def concat(xs):
        xs = list(xs)
        if len(xs) == 1:
            return xs[0]
        n = sum(shapes[x][0] for x in xs)
        sp = spatial(xs[0])
        return node(
            "concat" if sp else "vconcat", xs,
            (n, 64, 64) if sp else (n,)
        )

    def sl(x, start, stop):
        if start == 0 and stop == shapes[x][0]:
            return x
        sp = spatial(x)
        return node(
            "slice" if sp else "vslice", [x],
            (stop - start, 64, 64) if sp else (stop - start,),
            start=start, stop=stop
        )

    def binary(op, a, b):
        sp = spatial(a) or spatial(b)
        n = max(shapes[a][0], shapes[b][0])
        return node(
            op if sp else "v" + op, [a, b],
            (n, 64, 64) if sp else (n,)
        )

    def add(a, b):
        return binary("add", a, b)

    def mul(a, b):
        return binary("mul", a, b)

    def affine(x, scale=1.0, bias=0.0):
        if scale == 1.0 and bias == 0.0:
            return x
        n = shapes[x][0]
        if spatial(x):
            if n > 16:
                y = mul(x, constant(scale))
                return add(y, constant(bias)) if bias else y
            return conv(
                x,
                (scale * np.eye(n, dtype=np.float32))[:, :, None, None],
                np.full(n, bias, np.float32) if bias else None
            )
        return dense(
            x, scale * np.eye(n, dtype=np.float32),
            np.full(n, bias, np.float32) if bias else None
        )

    def sub(a, b):
        if spatial(a) or spatial(b):
            n = max(shapes[a][0], shapes[b][0])
            return node("sub", [a, b], (n, 64, 64))
        return add(a, affine(b, -1))

    def total(xs):
        xs = list(xs)
        result = xs[0]
        for x in xs[1:]:
            result = add(result, x)
        return result

    def relu(x):
        return node(
            "relu" if spatial(x) else "vrelu", [x], shapes[x]
        )

    def absolute(x):
        if spatial(x):
            return node("abs", [x], shapes[x])
        n = shapes[x][0]
        eye = np.eye(n, dtype=np.float32)
        positive = relu(dense(x, np.concatenate([eye, -eye], axis=0)))
        return dense(positive, np.concatenate([eye, eye], axis=1))

    def clip(x):
        if spatial(x):
            return node("min2", [relu(x), constant(1)], shapes[x])
        return relu(affine(relu(affine(x, -1, 1)), -1, 1))

    def gt(x, threshold=0.5):
        if spatial(x):
            return clip(affine(x, 4, -4 * threshold))
        return relu(affine(
            relu(affine(x, -4, 1 + 4 * threshold)), -1, 1
        ))

    def ge(x):
        return gt(x, -0.5)

    def eq(x):
        # Flat-topped equality test, tolerant of coordinate roundoff.
        if spatial(x):
            return clip(affine(absolute(x), -8, 2))
        n = shapes[x][0]
        eye = np.eye(n, dtype=np.float32)
        z = relu(dense(x, np.concatenate([eye, -eye], axis=0)))
        z = relu(dense(
            z, 8 * np.concatenate([eye, eye], axis=1),
            -np.ones(n, np.float32)
        ))
        return relu(dense(z, -eye, np.ones(n, np.float32)))

    def inverse(x):
        return affine(x, -1, 1)

    def mix(a, b, gate):
        return add(mul(a, gate), mul(b, inverse(gate)))

    def maximum(a, b):
        if spatial(a) or spatial(b):
            n = max(shapes[a][0], shapes[b][0])
            return node("max2", [a, b], (n, 64, 64))
        return add(a, relu(sub(b, a)))

    def minimum(a, b):
        return sub(a, relu(sub(a, b)))

    def gmax(x):
        return node("gmax", [x], (shapes[x][0],))

    def pool(x):
        return affine(node("gmean", [x], (shapes[x][0],)), 4096)

    def channel_sum(x):
        return conv(
            x, np.ones((1, shapes[x][0], 1, 1), np.float32)
        )

    def vector_sum(x):
        return dense(x, np.ones((1, shapes[x][0]), np.float32))

    def dot(x, values):
        return dense(x, np.asarray(values, np.float32).reshape(1, -1))

    def channel_max(x):
        y = sl(x, 0, 1)
        for i in range(1, shapes[x][0]):
            y = maximum(y, sl(x, i, i + 1))
        return y

    def winner(x):
        n = shapes[x][0]
        x = dense(
            x, np.eye(n, dtype=np.float32),
            -0.125 * np.arange(n, dtype=np.float32)
        )
        w = np.zeros((n * n, n), np.float32)
        for i in range(n):
            for j in range(n):
                w[i * n + j, j] += 1
                w[i * n + j, i] -= 1
        differences = relu(dense(x, w))
        w = np.zeros((n, n * n), np.float32)
        for i in range(n):
            w[i, i * n:(i + 1) * n] = -1024
        return relu(dense(differences, w, np.ones(n, np.float32)))

    def loop(iterations, initial, function):
        nonlocal current
        carried = {}
        descriptions = []
        for key, value in initial.items():
            state_id = fresh("s")
            carried[key] = state_id
            if isinstance(value, str):
                descriptions.append({"id": state_id, "init": value})
                shapes[state_id] = shapes[value]
            else:
                descriptions.append({"id": state_id, "zeros": list(value)})
                shapes[state_id] = tuple(value)

        outer = current
        body = []
        current = body
        updates = function(carried)
        current = outer
        loop_id = fresh("L")
        outer.append({
            "id": loop_id,
            "op": "loop",
            "T": iterations,
            "state": descriptions,
            "body": body,
            "next": {carried[k]: updates[k] for k in carried},
        })
        result = {}
        for key, state_id in carried.items():
            result[key] = loop_id + "." + state_id
            shapes[result[key]] = shapes[state_id]
        return result

    def shift(x, dx, dy):
        n = shapes[x][0]
        w = np.zeros((n, n, 3, 3), np.float32)
        for i in range(n):
            w[i, i, 1 + dy, 1 + dx] = 1
        return conv(x, w)

    # Coordinates and recurrent descriptors.
    yy = conv(
        "coords", np.array([[[[63]], [[0]], [[0]]]], np.float32)
    )
    xx = conv(
        "coords", np.array([[[[0]], [[63]], [[0]]]], np.float32)
    )
    rank = conv(
        "coords", np.array([[[[4032]], [[63]], [[0]]]], np.float32),
        np.array([1], np.float32)
    )

    initialized = sl("mem", 69, 70)
    old_geometry = sl("mem", 60, 67)
    old_goal_colour = sl("mem", 67, 68)
    old_goal_rank = sl("mem", 68, 69)
    cache_gate = mul(initialized, inverse(sl("aprev", 5, 6)))

    marker = sl("x0", 0, 1)
    marker_visible = gt(pool(marker))
    observed_y = pool(mul(marker, yy))
    observed_x = pool(mul(marker, xx))
    moved_y = dense(
        "aprev", np.array([[0, -3, 3, 0, 0, 0, 0, 0]], np.float32)
    )
    moved_x = dense(
        "aprev", np.array([[0, 0, 0, -3, 3, 0, 0, 0]], np.float32)
    )
    cy = mix(
        observed_y, add(sl("mem", 70, 71), moved_y), marker_visible
    )
    cx = mix(
        observed_x, add(sl("mem", 71, 72), moved_x), marker_visible
    )
    dy, dx = sub(yy, cy), sub(xx, cx)
    ay, ax = absolute(dy), absolute(dx)
    yzero, xzero = eq(dy), eq(dx)
    center = mul(yzero, xzero)

    def pick(score):
        winners = eq(sub(score, gmax(score)))
        winning_rank = gmax(mul(winners, rank))
        return mul(winners, eq(sub(rank, winning_rank)))

    # Palette-tile and target-frame detection.
    colours = sl("x0", 6, 15)
    colour_ids = np.arange(6, 15, dtype=np.float32)
    w = np.zeros((9, 9, 3, 3), np.float32)
    for c in range(9):
        w[c, c] = 1
    cores = gt(conv(colours, w), 8.5)
    button_fill = gt(conv(
        channel_sum(cores), np.ones((1, 1, 5, 5), np.float32)
    ))
    buttons = clip(add(button_fill, sl("x0", 2, 3)))

    border = sl("x0", 4, 5)
    surround = np.ones((1, 1, 3, 3), np.float32)
    surround[0, 0, 1, 1] = 0
    target = mul(gt(conv(border, surround), 3.5), inverse(border))
    near_lines = gt(conv(channel_sum(colours), surround))
    known_colours = mul(colours, mul(target, inverse(near_lines)))

    # Infer obscured target colours from the nearest unobscured markers.
    spread_w = np.zeros((9, 9, 3, 3), np.float32)
    for c in range(9):
        spread_w[c, c] = 1

    def spread_colours(s):
        r = clip(conv(s["r"], spread_w))
        unassigned = inverse(clip(channel_sum(s["f"])))
        return {
            "r": r,
            "f": add(s["f"], mul(r, unassigned)),
        }

    filled = loop(
        32, {"r": known_colours, "f": known_colours}, spread_colours
    )["f"]
    fill_scores = mul(filled, constant(colour_ids))
    fill_choice = mul(
        eq(sub(fill_scores, channel_max(fill_scores))),
        clip(channel_sum(filled))
    )
    inferred = channel_sum(mul(fill_choice, constant(colour_ids)))
    observed = channel_sum(mul(colours, constant(colour_ids)))
    target_colour = mul(target, mix(
        inferred, observed, clip(channel_sum(filled))
    ))

    # Store twenty target locations and their inferred colours.
    roll = np.zeros((20, 20), np.float32)
    roll[1:, :-1] = np.eye(19, dtype=np.float32)
    insert = np.zeros((20, 1), np.float32)
    insert[0, 0] = 1

    def extract_targets(s):
        p = mul(pick(mul(s["remaining"], rank)), s["remaining"])
        location = pool(mul(p, rank))
        colour = pool(mul(p, target_colour))
        return {
            "remaining": sub(s["remaining"], p),
            "ranks": add(
                dense(s["ranks"], roll), dense(location, insert)
            ),
            "colours": add(
                dense(s["colours"], roll), dense(colour, insert)
            ),
        }

    extracted = loop(20, {
        "remaining": target, "ranks": (20,), "colours": (20,)
    }, extract_targets)

    ranks = mix(sl("mem", 0, 20), extracted["ranks"], initialized)
    target_colours = mix(
        sl("mem", 20, 40), extracted["colours"], initialized
    )
    slots = eq(sub(rank, ranks))
    all_targets = channel_sum(slots)
    colour_sum = channel_sum(mul(slots, target_colours))
    colour_square = channel_sum(
        mul(slots, mul(target_colours, target_colours))
    )

    # Estimate selected colour, using local rays for hollow shapes.
    shape_colours = mul(colours, mul(inverse(buttons), inverse(target)))
    near_w = np.zeros((9, 9, 3, 3), np.float32)
    for c in range(9):
        near_w[c, c] = 1
        near_w[c, c, 1, 1] = 0
    near_count = pool(mul(conv(shape_colours, near_w), center))

    rays = [
        mul(yzero, ge(affine(dx, -1, -1))),
        mul(yzero, ge(affine(dx, 1, -1))),
        mul(xzero, ge(affine(dy, -1, -1))),
        mul(xzero, ge(affine(dy, 1, -1))),
    ]
    nearest = []
    for ray, distance in zip(rays, [ax, ax, ay, ay]):
        weighted = mul(
            mul(shape_colours, ray), affine(distance, -1, 64)
        )
        nearest.append(affine(gmax(weighted), -1, 64))

    ring_cost = add(total(nearest), affine(total([
        absolute(sub(nearest[0], nearest[1])),
        absolute(sub(nearest[2], nearest[3])),
    ]), 8))
    fresh_colour_score = sub(affine(near_count, 10000), ring_cost)

    old_rx = sl(old_geometry, 5, 6)
    old_ry = sl(old_geometry, 6, 7)
    old_box = mul(ge(sub(old_rx, ax)), ge(sub(old_ry, ay)))
    cached_colour_score = affine(
        pool(mul(shape_colours, old_box)), 100
    )
    active_colour_hot = winner(mix(
        cached_colour_score, fresh_colour_score, cache_gate
    ))
    active_colour = dot(active_colour_hot, colour_ids)
    active_pixels = channel_sum(mul(colours, active_colour_hot))

    # Four independent straight-line floods measure shape radii.
    support = clip(add(active_pixels, center))
    ray_initial = conv(center, np.ones((4, 1, 1, 1), np.float32))
    ray_w = np.zeros((4, 4, 5, 5), np.float32)
    for c, (ux, uy) in enumerate([(1, 0), (0, 1), (1, 1), (1, -1)]):
        for t in range(-2, 3):
            ray_w[c, c, 2 + t * uy, 2 + t * ux] = 1

    def flood_rays(s):
        return {"r": mul(clip(conv(s["r"], ray_w)), support)}

    reach = loop(24, {"r": ray_initial}, flood_rays)["r"]
    radii = gmax(mul(reach, concat([ax, ay, ax, ax])))
    diagonal_radius = maximum(sl(radii, 2, 3), sl(radii, 3, 4))

    local_w = np.zeros((3, 1, 3, 3), np.float32)
    local_w[0, 0, 1, 0] = local_w[0, 0, 1, 2] = 1
    local_w[1, 0, 0, 1] = local_w[1, 0, 2, 1] = 1
    for i in (0, 2):
        for j in (0, 2):
            local_w[2, 0, i, j] = 1
    local = pool(mul(conv(active_pixels, local_w), center))
    fresh_h = gt(sl(local, 0, 1))
    fresh_v = gt(sl(local, 1, 2))
    fresh_x = mul(
        gt(sl(local, 2, 3)), mul(inverse(fresh_h), inverse(fresh_v))
    )
    fresh_ring = inverse(clip(total([fresh_h, fresh_v, fresh_x])))

    scalar_nearest = [
        vector_sum(mul(n, active_colour_hot)) for n in nearest
    ]

    def usable_pair(a, b):
        high, low = maximum(a, b), minimum(a, b)
        return mix(low, high, gt(high, 32))

    ring_rx = usable_pair(scalar_nearest[0], scalar_nearest[1])
    ring_ry = usable_pair(scalar_nearest[2], scalar_nearest[3])
    tips = mul(eq(sub(ax, ring_rx)), yzero)
    tip_kind = pool(mul(conv(active_pixels, local_w[[1, 2]]), tips))
    fresh_d = mul(
        fresh_ring, gt(sub(sl(tip_kind, 1, 2), sl(tip_kind, 0, 1)))
    )
    fresh_b = sub(fresh_ring, fresh_d)

    fresh_rx = total([
        mul(fresh_h, sl(radii, 0, 1)),
        mul(fresh_x, diagonal_radius),
        mul(fresh_ring, ring_rx),
    ])
    fresh_ry = total([
        mul(fresh_v, sl(radii, 1, 2)),
        mul(fresh_x, diagonal_radius),
        mul(fresh_ring, ring_ry),
    ])
    geometry = mix(
        old_geometry,
        concat([fresh_h, fresh_v, fresh_x, fresh_d, fresh_b,
                fresh_rx, fresh_ry]),
        cache_gate
    )
    h, v, cross_x, diamond, box, rx, ry = [
        sl(geometry, i, i + 1) for i in range(7)
    ]
    ring = add(diamond, box)

    horizontal = mul(mul(yzero, ge(sub(rx, ax))), h)
    vertical = mul(mul(xzero, ge(sub(ry, ay))), v)
    diagonal = mul(
        mul(mul(eq(sub(ax, ay)), ge(sub(rx, ax))), inverse(center)),
        cross_x
    )
    diamond_mask = mul(eq(sub(add(ax, ay), rx)), diamond)
    box_mask = mul(add(
        mul(eq(sub(ax, rx)), ge(sub(ry, ay))),
        mul(eq(sub(ay, ry)), ge(sub(rx, ax)))
    ), box)
    footprint = clip(total([
        horizontal, vertical, diagonal, diamond_mask, box_mask
    ]))

    # Ownership prevents repeatedly repositioning already useful shapes.
    family = total([h, affine(v, 2), affine(cross_x, 4), affine(ring, 5)])
    shape_id = add(
        affine(family, 4096, 1),
        mul(add(affine(rx, 64), ry), inverse(ring))
    )
    owners = sl("mem", 40, 60)
    self_owned = eq(sub(owners, shape_id))
    unowned = inverse(gt(owners))
    needed_slots = clip(add(self_owned, unowned))
    needed_map = channel_sum(mul(slots, needed_slots))

    old_desired_hot = eq(dense(
        old_goal_colour, np.ones((9, 1), np.float32), -colour_ids
    ))
    desired_cores = channel_sum(mul(cores, old_desired_hot))
    desired_buttons = gt(conv(
        desired_cores, np.ones((1, 1, 5, 5), np.float32)
    ))

    fields = concat([
        all_targets, colour_sum, colour_square,
        needed_map, buttons, desired_buttons
    ])

    # Translate six scoring fields to the starts of four shape segments.
    offsets = concat([
        affine(rx, -1),
        affine(mul(add(cross_x, box), ry), -1),
        mul(add(cross_x, box), rx),
        affine(ry, -1),
        mul(ring, rx),
        mul(box, ry),
        affine(mul(box, rx), -1),
        mul(ring, ry),
    ])
    gate_w = np.zeros((16, 9), np.float32)
    for i in range(8):
        gate_w[2 * i, i] = 4
        gate_w[2 * i + 1, i] = -4
        gate_w[2 * i:2 * i + 2, 8] = -4

    def initialize_segments(s):
        gates = clip(dense(
            concat([offsets, s["t"]]), gate_w,
            np.full(16, -2, np.float32)
        ))
        out = {}
        for edge in range(4):
            a = s["e" + str(edge)]
            for axis in range(2):
                p = 4 * edge + 2 * axis
                positive_gate = sl(gates, p, p + 1)
                negative_gate = sl(gates, p + 1, p + 2)
                positive = shift(a, int(axis == 0), int(axis == 1))
                negative = shift(a, -int(axis == 0), -int(axis == 1))
                a = add(a, add(
                    mul(sub(positive, a), positive_gate),
                    mul(sub(negative, a), negative_gate)
                ))
            out["e" + str(edge)] = a
        out["t"] = affine(s["t"], 1, 1)
        return out

    starts = loop(24, {
        "e0": fields, "e1": fields, "e2": fields, "e3": fields, "t": (1,)
    }, initialize_segments)

    length0 = total([
        mul(add(h, cross_x), affine(rx, 2, 1)),
        mul(diamond, rx), mul(box, affine(rx, 2))
    ])
    length1 = total([
        mul(v, affine(ry, 2, 1)),
        mul(cross_x, affine(rx, 2, 1)),
        mul(diamond, rx), mul(box, affine(ry, 2))
    ])
    length2 = add(mul(diamond, rx), mul(box, affine(rx, 2)))
    length3 = add(mul(diamond, rx), mul(box, affine(ry, 2)))
    lengths = concat([length0, length1, length2, length3])

    directions = [
        [(1, 0, add(h, box)), (1, 1, cross_x), (1, -1, diamond)],
        [(0, 1, add(v, box)), (-1, 1, cross_x), (1, 1, diamond)],
        [(-1, 0, box), (-1, 1, diamond)],
        [(0, -1, box), (-1, -1, diamond)],
    ]
    length_w = np.zeros((4, 5), np.float32)
    length_w[:, :4] = 4 * np.eye(4, dtype=np.float32)
    length_w[:, 4] = -4
    both_bars = mul(h, v)

    def integrate_segments(s):
        live = clip(dense(
            concat([lengths, s["t"]]), length_w,
            np.full(4, -2, np.float32)
        ))
        skips = eq(dense(
            concat([s["t"], rx, ry]),
            np.array([[1, -1, 0], [1, 0, -1]], np.float32)
        ))
        skip_x = mul(cross_x, sl(skips, 0, 1))
        skip_cross = mul(both_bars, sl(skips, 1, 2))
        contributions = []
        out = {}
        for edge in range(4):
            a = s["e" + str(edge)]
            gate = sl(live, edge, edge + 1)
            if edge == 0:
                gate = mul(gate, inverse(skip_x))
            elif edge == 1:
                gate = mul(gate, inverse(add(skip_x, skip_cross)))
            contributions.append(mul(a, gate))
            out["e" + str(edge)] = total([
                mul(shift(a, sx, sy), enable)
                for sx, sy, enable in directions[edge]
            ])
        out["acc"] = add(s["acc"], total(contributions))
        out["t"] = affine(s["t"], 1, 1)
        return out

    correlation = loop(45, {
        "e0": starts["e0"], "e1": starts["e1"],
        "e2": starts["e2"], "e3": starts["e3"],
        "acc": (6, 64, 64), "t": (1,)
    }, integrate_segments)["acc"]
    count, csum, csquare, need, paint, desired_paint = [
        sl(correlation, i, i + 1) for i in range(6)
    ]

    # Only positions on the selected shape's movement lattice are reachable.
    def residues(axis):
        w = np.zeros((64, 3, 1, 1), np.float32)
        w[:, axis, 0, 0] = 63
        impulses = eq(conv(
            "coords", w, -np.arange(64, dtype=np.float32)
        ))
        r = np.zeros((3, 64, 1, 1), np.float32)
        for i in range(64):
            r[i % 3, i, 0, 0] = 1
        return conv(impulses, r)

    row_residue, col_residue = residues(0), residues(1)
    row_choice = pool(mul(row_residue, center))
    col_choice = pool(mul(col_residue, center))
    lattice = mul(
        channel_sum(mul(row_residue, row_choice)),
        channel_sum(mul(col_residue, col_choice))
    )

    variance = sub(mul(count, csquare), mul(csum, csum))
    same_colour = eq(sub(csum, mul(count, active_colour)))
    history = mul(eq(sub(rank, old_goal_rank)), cache_gate)
    score = total([
        affine(need, 10000),
        affine(count, 100),
        affine(add(ax, ay), -1),
        affine(same_colour, 15),
        affine(history, 3),
        affine(variance, -100000),
        affine(inverse(lattice), -1000000),
        affine(gt(paint), -1000000),
    ])
    goal = pick(score)
    goal_count = pool(mul(goal, count))
    goal_sum = pool(mul(goal, csum))
    goal_need = pool(mul(goal, need))
    goal_rank = pool(mul(goal, rank))
    colour_w = np.stack([-colour_ids, np.ones(9, np.float32)], axis=1)
    goal_hot = mul(
        eq(dense(concat([goal_count, goal_sum]), colour_w)),
        gt(goal_count)
    )
    goal_colour = dot(goal_hot, colour_ids)

    # Route either to the selected placement or to the required paint tile.
    consistent_goal = eq(sub(old_goal_colour, goal_colour))
    wrong_colour = inverse(eq(sub(active_colour, goal_colour)))
    desired_hit = gt(desired_paint)
    other_paint = gt(sub(paint, desired_paint))
    paint_goal = mul(desired_hit, inverse(other_paint))
    use_paint = mul(
        wrong_colour, mul(consistent_goal, gmax(paint_goal))
    )
    destination = mix(paint_goal, goal, use_paint)
    blocked = mix(other_paint, gt(paint), consistent_goal)
    traversable = inverse(blocked)

    movement_w = np.zeros((4, 1, 7, 7), np.float32)
    for i, (sx, sy) in enumerate([(0, -3), (0, 3), (-3, 0), (3, 0)]):
        movement_w[i, 0, 3 + sy, 3 + sx] = 1

    def navigate(s):
        neighbours = conv(s["value"], movement_w)
        advanced = relu(affine(channel_max(neighbours), 1, -1))
        return {
            "value": mul(maximum(s["value"], advanced), traversable)
        }

    navigation = loop(
        42, {"value": affine(destination, 200)}, navigate
    )["value"]
    direction_values = pool(mul(conv(navigation, movement_w), center))
    goal_dy = pool(mul(goal, dy))
    goal_dx = pool(mul(goal, dx))
    greedy = dense(
        concat([goal_dy, goal_dx]),
        np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], np.float32)
    )
    movement = total([
        affine(direction_values, 1000),
        greedy,
        affine(sl("aprev", 1, 5), 0.125),
    ])

    at_goal = pool(mul(goal, center))
    correct_colour = eq(sub(active_colour, goal_colour))
    has_work = gt(goal_need)
    claim = mul(mul(at_goal, correct_colour), has_work)
    switch = clip(add(mul(at_goal, correct_colour), inverse(has_work)))
    switch_logit = affine(switch, 2000000, -1000000)

    output_w = np.zeros((8, 5), np.float32)
    output_w[1:5, :4] = np.eye(4, dtype=np.float32)
    output_w[5, 4] = 1
    output_b = np.zeros(8, np.float32)
    output_b[[0, 6, 7]] = -1000000
    dense(
        concat([movement, switch_logit]), output_w, output_b,
        name="logits"
    )
    conv(
        "coords", np.zeros((1, 3, 1, 1), np.float32), name="click"
    )

    covered_slots = gt(mul(
        pool(mul(slots, footprint)),
        eq(sub(target_colours, active_colour))
    ))
    cleared = mul(owners, inverse(self_owned))
    claimed_owners = add(
        mul(cleared, inverse(covered_slots)),
        mul(covered_slots, shape_id)
    )
    next_owners = mix(claimed_owners, owners, claim)
    memory_output = concat([
        ranks, target_colours, next_owners, geometry,
        goal_colour, goal_rank, constant(1), cy, cx
    ])
    node(
        "vadd", [memory_output, constant(0)], (72,), name="mem_next"
    )

    # Static dead-code elimination, including unused parameter arrays.
    def trim(nodes, required):
        needed = set(required)
        kept = []
        for nd in reversed(nodes):
            if nd["op"] != "loop":
                if nd["id"] not in needed:
                    continue
                needed.remove(nd["id"])
                needed.update(nd["in"])
                kept.append(nd)
                continue

            outputs = {
                nd["id"] + "." + state["id"] for state in nd["state"]
            }
            if not (outputs & needed):
                continue
            needed.difference_update(outputs)
            body, dependencies = trim(nd["body"], nd["next"].values())
            state_names = {state["id"] for state in nd["state"]}
            needed.update(dependencies - state_names)
            for state in nd["state"]:
                if "init" in state:
                    needed.add(state["init"])
            nd = dict(nd)
            nd["body"] = body
            kept.append(nd)
        kept.reverse()
        return kept, needed

    graph, _ = trim(graph, {"logits", "click", "mem_next"})
    used_parameters = set()

    def collect(nodes):
        for nd in nodes:
            if nd["op"] == "loop":
                collect(nd["body"])
            else:
                for key in ("w", "b"):
                    if key in nd:
                        used_parameters.add(nd[key])

    collect(graph)
    params = {key: value for key, value in params.items()
              if key in used_parameters}
    return {"graph": graph, "memory_size": 72, "params": params}
