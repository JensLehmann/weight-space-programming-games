import math
import numpy as np


def build():
    graph = []
    params = {}
    shapes = {
        "x0": (16, True),
        "coords": (3, True),
        "avail": (8, False),
        "aprev": (8, False),
        "mem": (43, False),
    }
    scopes = [graph]
    serial = [0]
    parameter_cache = {}

    def fresh():
        serial[0] += 1
        return "n%d" % serial[0]

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def emit(op, inputs, shape, name=None, **fields):
        name = fresh() if name is None else name
        node = {"id": name, "op": op, "in": list(inputs)}
        node.update(fields)
        scopes[-1].append(node)
        shapes[name] = shape
        return name

    def affine(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        c, spatial_flag = shapes[x]
        assert w.shape[1] == c
        if spatial_flag:
            op = "conv"
            fields = {"w": parameter(w[:, :, None, None])}
        else:
            op = "dense"
            fields = {"w": parameter(w)}
        if b is not None:
            b = np.asarray(b, np.float32)
            if b.ndim == 0:
                b = np.full(w.shape[0], b, np.float32)
            fields["b"] = parameter(b)
        return emit(op, [x], (w.shape[0], spatial_flag),
                    name=name, **fields)

    def conv(x, w, b=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(np.asarray(b, np.float32))
        return emit("conv", [x], (w.shape[0], True), **fields)

    def scale(x, a, b=0.0, name=None):
        c = shapes[x][0]
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        if a.ndim == 0:
            a = np.full(c, a, np.float32)
        if b.ndim == 0:
            b = np.full(c, b, np.float32)
        return affine(x, np.diag(a), b, name=name)

    def constant(values):
        v = np.asarray(values, np.float32).reshape(-1)
        return affine("avail", np.zeros((len(v), 8), np.float32), v)

    zero_v = constant([0.0])
    one_v = constant([1.0])
    zero = affine("coords", [[0, 0, 0]])
    one = emit("slice", ["coords"], (1, True), start=2, stop=3)

    def spatial(x):
        if shapes[x][1]:
            return x
        return emit("add", [zero, x], (shapes[x][0], True))

    def add(a, b):
        ca, sa = shapes[a]
        cb, sb = shapes[b]
        return emit("add" if sa or sb else "vadd", [a, b],
                    (max(ca, cb), sa or sb))

    def mul(a, b):
        ca, sa = shapes[a]
        cb, sb = shapes[b]
        return emit("mul" if sa or sb else "vmul", [a, b],
                    (max(ca, cb), sa or sb))

    def sub(a, b):
        ca, sa = shapes[a]
        cb, sb = shapes[b]
        if not sa and not sb:
            return add(a, scale(b, -1))
        return emit("sub", [a, b], (max(ca, cb), True))

    def relu(x):
        return emit("relu" if shapes[x][1] else "vrelu",
                    [x], shapes[x])

    def absolute(x):
        x = spatial(x)
        return emit("abs", [x], shapes[x])

    def maximum(a, b):
        if not shapes[a][1] and not shapes[b][1]:
            a = spatial(a)
        return emit("max2", [a, b],
                    (max(shapes[a][0], shapes[b][0]), True))

    def minimum(a, b):
        if not shapes[a][1] and not shapes[b][1]:
            a = spatial(a)
        return emit("min2", [a, b],
                    (max(shapes[a][0], shapes[b][0]), True))

    def clip01(x):
        return minimum(maximum(spatial(x), zero_v), one_v)

    def inv(x):
        return scale(x, -1, 1)

    def sl(x, start, stop):
        return emit("slice" if shapes[x][1] else "vslice",
                    [x], (stop - start, shapes[x][1]),
                    start=start, stop=stop)

    def cat(*xs):
        spatial_flag = any(shapes[x][1] for x in xs)
        if spatial_flag:
            xs = tuple(spatial(x) for x in xs)
        return emit("concat" if spatial_flag else "vconcat", xs,
                    (sum(shapes[x][0] for x in xs), spatial_flag))

    def gmean(x):
        return emit("gmean", [x], (shapes[x][0], False))

    def gmax(x):
        return emit("gmax", [x], (shapes[x][0], False))

    def sum_channels(x):
        return affine(x, np.ones((1, shapes[x][0]), np.float32))

    def dot(x, weights):
        return affine(x, np.asarray(weights, np.float32)[None, :])

    def max_xy(x):
        return maximum(sl(x, 0, 1), sl(x, 1, 2))

    def cheb(a, b):
        return max_xy(absolute(sub(a, b)))

    def weighted(items, bias=0.0):
        return affine(cat(*(x for x, _ in items)),
                      [[w for _, w in items]], [bias])

    def product(*xs):
        z = xs[0]
        for x in xs[1:]:
            z = mul(z, x)
        return z

    def gated(score, gate):
        return sub(score, scale(inv(gate), 10000))

    def repeat(T, specifications, body_fn):
        states = []
        ids = []
        for initial in specifications:
            sid = fresh()
            ids.append(sid)
            if isinstance(initial, str):
                shapes[sid] = shapes[initial]
                states.append({"id": sid, "init": initial})
            else:
                c, sp = initial
                shapes[sid] = (c, sp)
                states.append({
                    "id": sid,
                    "zeros": [c, 64, 64] if sp else [c],
                })
        body = []
        scopes.append(body)
        results = body_fn(*ids)
        scopes.pop()
        lid = fresh()
        scopes[-1].append({
            "id": lid,
            "op": "loop",
            "T": int(T),
            "state": states,
            "body": body,
            "next": dict(zip(ids, results)),
        })
        outputs = []
        for sid in ids:
            ref = lid + "." + sid
            shapes[ref] = shapes[sid]
            outputs.append(ref)
        return outputs

    # A reset invalidates every cached field.
    previous_reset = sl("aprev", 0, 1)
    history = mul("mem", inv(previous_reset))

    xy = affine("coords", [[0, 63, 0], [63, 0, 0]])
    xmap = sl(xy, 0, 1)
    ymap = sl(xy, 1, 2)
    index = affine("coords", [[64 * 63, 63, 0]])
    index_plus_one = scale(index, 1, 1)

    board = mul(clip01(scale(ymap, 2, -19)),
                clip01(scale(ymap, -2, 125)))
    header = mul(clip01(scale(ymap, -2, 19)),
                 clip01(scale(xmap, 2, -51)))

    colours = [10, 6, 15, 11, 12, 8, 9, 7, 14]
    sizes = np.asarray([1, 2, 3, 4, 5, 7, 8, 9, 10], np.float32)
    halves = np.floor(sizes / 2)
    rights = sizes - 1 - halves
    extents = (sizes - 1) / 2
    centre_offsets = extents - halves

    # Fruit detection: top-left corner, complete top edge, and size.
    local_w = np.zeros((9, 16, 3, 3), np.float32)
    local_b = np.zeros(9, np.float32)
    for t, colour in enumerate(colours):
        if t == 0:
            local_w[t, colour, 1, 1] = 1
        else:
            local_w[t, colour, 1:3, 1:3] = 1
            local_b[t] = -3
        local_w[t, colour, 0, 1] = -1
        local_w[t, colour, 1, 0] = -1
    local = relu(conv("x0", local_w, local_b))

    colour_w = np.zeros((9, 16), np.float32)
    for t, colour in enumerate(colours):
        colour_w[t, colour] = 1
    colour_planes = affine("x0", colour_w)

    def diagonal_shift(x, dxs, dys):
        c = shapes[x][0]
        w = np.zeros((c, c, 7, 7), np.float32)
        for j in range(c):
            w[j, j, 3 + int(dys[j]), 3 + int(dxs[j])] = 1
        return conv(x, w)

    far = colour_planes
    remaining = sizes.astype(int) - 1
    for _ in range(3):
        shift = np.minimum(remaining, 3)
        far = diagonal_shift(far, shift, np.zeros(9, int))
        remaining -= shift

    outside_w = np.zeros((9, 9, 3, 3), np.float32)
    for j in range(9):
        outside_w[j, j, 1, 2] = 1
    outside = conv(far, outside_w)
    edge_w = np.concatenate(
        [np.eye(9), np.eye(9), -np.eye(9)], axis=1
    ).astype(np.float32)
    corners = relu(affine(cat(local, far, outside),
                         edge_w, -np.ones(9, np.float32)))

    fruit_all = corners
    remaining = halves.astype(int)
    for _ in range(2):
        shift = np.minimum(remaining, 3)
        fruit_all = diagonal_shift(fruit_all, -shift, -shift)
        remaining -= shift

    enemy_patterns = [
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0]],
        [[1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 1, 1, 1, 0]],
    ]
    enemy_w = np.zeros((3, 16, 7, 7), np.float32)
    enemy_b = np.zeros(3, np.float32)
    for t, (pat, colour) in enumerate(
            zip(enemy_patterns, [7, 14, 13])):
        pat = np.asarray(pat, np.float32)
        enemy_w[t, colour, 1:5, 1:6] = 2 * pat - 1
        enemy_b[t] = 1 - pat.sum()
    enemy_all = relu(conv("x0", enemy_w, enemy_b))

    # Goal detection also accepts the game's red warning rendering.
    goal_w = np.zeros((1, 16, 7, 7), np.float32)
    for colour in [9, 2]:
        for dx in range(-2, 3):
            goal_w[0, colour, 3, 3 + dx] = 1
        for dy in [1, 2]:
            for dx in [-3, 3]:
                goal_w[0, colour, 3 + dy, 3 + dx] = 1
        goal_w[0, colour, 3, 0] = -1
        goal_w[0, colour, 3, 6] = -1
        goal_w[0, colour, 2, 3] = -1
    observed_goals = relu(conv("x0", goal_w, [-8]))
    observed_goals = diagonal_shift(observed_goals, [0], [-3])
    observed_goals = diagonal_shift(observed_goals, [0], [-1])
    observed_goals = mul(observed_goals, board)

    fruits = mul(fruit_all, board)
    enemies = mul(enemy_all, board)
    fruit_mask = sum_channels(fruits)
    enemy_mask = sum_channels(enemies)

    def extract(mask, feature_planes, number):
        width = shapes[feature_planes][0]

        def body(rem, records):
            score = mul(rem, index_plus_one)
            top = gmax(score)
            selected = mul(
                rem, clip01(scale(sub(score, top), 4, 1))
            )
            record = scale(
                gmean(mul(feature_planes, selected)), 4096
            )
            records_next = cat(
                sl(records, width, width * number), record
            )
            return [relu(sub(rem, selected)), records_next]

        return repeat(
            number, [mask, (number * width, False)], body
        )[1]

    fruit_records = extract(fruit_mask, cat(fruits, xy, one), 16)
    enemy_records = extract(enemy_mask, cat(enemies, xy, one), 8)
    observed_goal_records = extract(
        observed_goals, cat(xy, one), 8
    )

    requirements_observed = scale(
        gmean(mul(cat(fruit_all, enemy_all), header)), 4096
    )
    previous_goals = sl(history, 0, 24)
    previous_requirements = sl(history, 24, 36)

    def record_count(records, stride, number):
        w = np.zeros(stride * number, np.float32)
        w[stride - 1::stride] = 1
        return dot(records, w)

    def cache_more(new, old, new_count, old_count):
        flag = gmax(clip01(scale(
            sub(new_count, old_count), 2, -0.5
        )))
        return add(mul(new, flag), mul(old, inv(flag)))

    goal_records = cache_more(
        observed_goal_records, previous_goals,
        record_count(observed_goal_records, 3, 8),
        record_count(previous_goals, 3, 8),
    )
    requirements = cache_more(
        requirements_observed, previous_requirements,
        sum_channels(requirements_observed),
        sum_channels(previous_requirements),
    )

    def rebuild_goals(records, accumulated):
        point = sl(records, 0, 2)
        active = sl(records, 2, 3)
        point_mask = mul(
            clip01(scale(cheb(xy, point), -4, 2)), active
        )
        rotated = cat(sl(records, 3, 24), sl(records, 0, 3))
        return [rotated, maximum(accumulated, point_mask)]

    goal_map = repeat(
        8, [goal_records, (1, True)], rebuild_goals
    )[1]

    fruit_counts = scale(gmean(fruits), 4096)
    enemy_counts = scale(gmean(enemies), 4096)
    fruit_req = sl(requirements, 0, 9)
    enemy_req = sl(requirements, 9, 12)

    def highest_required(req, number, rank_values):
        w = np.zeros((number, number), np.float32)
        for i in range(number):
            w[i, i + 1:] = -1
        no_higher = relu(affine(
            req, w, np.ones(number, np.float32)
        ))
        chosen = gmax(mul(clip01(req), no_higher))
        rank = dot(chosen, rank_values)
        count = sum_channels(mul(req, chosen))
        return chosen, rank, count

    fruit_target, target_rank, fruit_need = highest_required(
        fruit_req, 9, np.arange(9)
    )
    enemy_target, target_enemy_rank, enemy_need = highest_required(
        enemy_req, 3, [1, 2, 3]
    )

    ready_fruit_count = sum_channels(mul(fruit_counts, fruit_target))
    ready_enemy_count = sum_channels(mul(enemy_counts, enemy_target))
    ready_fruit_map = sum_channels(mul(fruits, fruit_target))
    ready_enemy_map = sum_channels(mul(enemies, enemy_target))

    def dilate_four(x):
        w = np.ones((1, 1, 5, 5), np.float32)
        return clip01(conv(conv(x, w), w))

    goal_region = dilate_four(goal_map)
    fruit_occupied = mul(goal_map, dilate_four(ready_fruit_map))
    enemy_occupied = mul(goal_map, dilate_four(ready_enemy_map))
    empty_fruit_goals = relu(sub(goal_map, fruit_occupied))
    empty_enemy_goals = relu(sub(goal_map, enemy_occupied))

    parked_fruit_count = scale(gmean(fruit_occupied), 4096)
    parked_enemy_count = scale(gmean(enemy_occupied), 4096)
    need_fabrication = clip01(sub(fruit_need, ready_fruit_count))
    need_fruit_delivery = clip01(sub(fruit_need, parked_fruit_count))
    need_enemy_fabrication = clip01(sub(enemy_need, ready_enemy_count))
    need_enemy_delivery = clip01(sub(enemy_need, parked_enemy_count))
    fruit_finished = inv(need_fruit_delivery)
    has_enemies = clip01(sum_channels(enemy_counts))

    rank_axis = constant(np.arange(9))
    rank_difference = sub(spatial(target_rank), rank_axis)
    lower_types = clip01(rank_difference)
    matching_types = clip01(scale(absolute(rank_difference), -1, 1))
    protected_types = gmax(add(
        matching_types, mul(lower_types, need_fabrication)
    ))
    protected_map = sum_channels(mul(fruits, protected_types))
    has_protected = gmax(protected_map)

    enemy_minimum = constant([2, 12])
    enemy_maximum = constant([61, 62])
    enemy_extent = constant([2, 1.5])
    enemy_offset = constant([0, -0.5])

    def box_selection(point, offset, extent):
        centre = add(point, offset)
        nearest = relu(sub(absolute(sub(xy, centre)), extent))
        distance_squared = sum_channels(mul(nearest, nearest))
        return clip01(scale(distance_squared, -2, 129))

    def fruit_geometry(kind, point):
        offset = dot(kind, centre_offsets)
        extent = dot(kind, extents)
        selected = box_selection(point, offset, extent)
        lo = affine(kind, np.stack([halves, halves]), [0, 10])
        hi = affine(kind, np.stack([-rights, -rights]), [63, 63])
        endpoint = minimum(maximum(xy, lo), hi)
        return selected, endpoint

    def enemy_selection(point):
        return box_selection(point, enemy_offset, enemy_extent)

    def incidences(records, ntypes, number, is_fruit):
        stride = ntypes + 3

        def body(packed, accumulated):
            kind = sl(packed, 0, ntypes)
            point = sl(packed, ntypes, ntypes + 2)
            active = sl(packed, ntypes + 2, ntypes + 3)
            if is_fruit:
                offset = dot(kind, centre_offsets)
                extent = dot(kind, extents)
                selected = box_selection(point, offset, extent)
            else:
                selected = enemy_selection(point)
            contribution = product(selected, active, kind)
            rotated = cat(
                sl(packed, stride, stride * number),
                sl(packed, 0, stride),
            )
            return [rotated, add(accumulated, contribution)]

        return repeat(
            number, [records, (ntypes, True)], body
        )[1]

    fruit_incidence = incidences(fruit_records, 9, 16, True)
    enemy_incidence = incidences(enemy_records, 3, 8, False)
    selected_fruit_total = sum_channels(fruit_incidence)
    selected_enemy_total = sum_channels(enemy_incidence)

    def mixing_penalty(incidence, total):
        squares = sum_channels(mul(incidence, incidence))
        mixed = clip01(sub(mul(total, total), squares))
        crowded = clip01(scale(total, 1, -2))
        return add(mixed, crowded)

    fruit_mixing = mixing_penalty(fruit_incidence, selected_fruit_total)
    enemy_mixing = mixing_penalty(enemy_incidence, selected_enemy_total)
    selected_ready_fruit = sum_channels(mul(fruit_incidence, fruit_target))
    consuming_ready = clip01(scale(selected_ready_fruit, 1, -1))

    def nearest(mask, l1_distance, extra_penalty=None):
        score = weighted([
            (mask, 1048576),
            (l1_distance, -4096),
            (index, -1),
        ])
        if extra_penalty is not None:
            score = sub(score, scale(extra_penalty, 600000))
        winner = clip01(scale(sub(score, gmax(score)), 2, 1))
        return scale(gmean(mul(xy, winner)), 4096)

    def origin_fields(point):
        delta = absolute(sub(xy, point))
        dmax = max_xy(delta)
        l1 = sum_channels(delta)
        self_mask = clip01(scale(dmax, -4, 2))
        inside_goal = gmax(product(
            goal_map, clip01(scale(dmax, -2, 9))
        ))
        return l1, self_mask, inside_goal

    initial_best = scale(zero, 1, -5000)

    def fruit_tasks(packed, best, danger):
        kind = sl(packed, 0, 9)
        point = sl(packed, 9, 11)
        active = sl(packed, 11, 12)
        rank = dot(kind, np.arange(9))
        half = dot(kind, halves)

        difference = sub(spatial(target_rank), rank)
        below = clip01(difference)
        above = clip01(scale(difference, -1))
        ready = clip01(scale(absolute(difference), -1, 1))
        protected = add(ready, mul(below, need_fabrication))

        l1, self_mask, inside_goal = origin_fields(point)
        same_kind = sum_channels(mul(fruits, kind))
        partner = nearest(relu(sub(same_kind, self_mask)), l1)
        destination = nearest(empty_fruit_goals, l1)
        predator = nearest(enemy_mask, l1)

        own_count = sum_channels(mul(fruit_counts, kind))
        has_partner = clip01(scale(own_count, 1, -1))
        selected, endpoint = fruit_geometry(kind, point)
        own_selected = sum_channels(mul(fruit_incidence, kind))

        pair_before = cheb(point, partner)
        pair_after = cheb(endpoint, partner)
        goal_before = cheb(point, destination)
        goal_after = cheb(endpoint, destination)
        pred_before = cheb(point, predator)
        pred_after_move = cheb(endpoint, predator)
        pred_after = add(
            pred_before,
            mul(selected, sub(pred_after_move, pred_before)),
        )
        predator_selected = enemy_selection(predator)

        collision_margin = relu(
            sub(scale(half, 1, 7), pred_after)
        )
        risk = product(
            collision_margin, inv(predator_selected),
            has_enemies, protected, active,
        )
        unpark = product(
            selected, ready, inside_goal, inv(goal_region), active
        )
        danger_next = add(danger, add(risk, scale(unpark, 2)))

        merge_bonus = clip01(scale(own_selected, 1, -1))
        merge_score = weighted([
            (sub(pair_before, pair_after), 2),
            (pair_after, -0.15),
            (pair_before, -0.10),
            (merge_bonus, 45),
        ], 30)
        merge_score = gated(merge_score, product(
            active, below, need_fabrication, has_partner, selected
        ))

        arrival = clip01(scale(goal_after, -2, 9))
        delivery_score = weighted([
            (sub(goal_before, goal_after), 2),
            (goal_after, -0.15),
            (arrival, 30),
        ], 18)
        delivery_score = gated(delivery_score, product(
            active, ready, need_fruit_delivery,
            inv(inside_goal), selected
        ))
        delivery_score = sub(
            delivery_score, scale(merge_bonus, 10000)
        )

        shrink_score = weighted([
            (sub(pred_before, pred_after), 1.5),
            (pred_after, -0.20),
            (predator_selected, -10000),
            (clip01(scale(selected_fruit_total, 1, -1)), -10000),
        ], 42)
        shrink_score = gated(shrink_score, product(
            active, above, need_fabrication, has_enemies
        ))

        # Rescue does not require a merge partner or a finished rank.
        threatened = clip01(sub(scale(half, 1, 11), pred_before))
        escape_score = weighted([
            (sub(pred_after_move, pred_before), 1),
            (sub(goal_before, goal_after), 0.15),
        ], 6)
        escape_score = gated(escape_score, product(
            active, protected, has_enemies, threatened, selected
        ))

        best_next = maximum(
            best,
            maximum(
                maximum(merge_score, delivery_score),
                maximum(shrink_score, escape_score),
            ),
        )
        rotated = cat(sl(packed, 12, 192), sl(packed, 0, 12))
        return [rotated, best_next, danger_next]

    fruit_result = repeat(
        16, [fruit_records, initial_best, (1, True)], fruit_tasks
    )
    best_fruit = fruit_result[1]
    fruit_danger = fruit_result[2]

    # Analytic piecewise-linear inverse square root.
    knots = np.asarray([
        0, 1, 2, 3, 4, 5, 8, 10, 13, 16, 20, 25, 32, 40,
        50, 64, 80, 100, 125, 160, 200, 256, 512, 1024, 8192,
    ], np.float32)
    values = np.zeros_like(knots)
    values[1:] = 1.0 / np.sqrt(knots[1:])
    slopes = np.diff(values) / np.diff(knots)
    hinge_coefficients = np.concatenate(
        [slopes[:1], np.diff(slopes)]
    ).astype(np.float32)

    round_w = np.zeros((12, 2), np.float32)
    round_b = np.zeros(12, np.float32)
    displacement_w = np.zeros((2, 12), np.float32)
    for axis in range(2):
        for sign_index, sign in enumerate([1, -1]):
            for k in range(3):
                j = axis * 6 + sign_index * 3 + k
                round_w[j, axis] = 32 * sign
                round_b[j] = 0.5 - 32 * (k + 0.5)
                displacement_w[axis, j] = 4 * sign

    def enemy_endpoint(kind, point):
        delta = sub(xy, point)
        r2 = sum_channels(mul(delta, delta))
        hinges = relu(affine(
            r2, np.ones((len(knots) - 1, 1), np.float32),
            -knots[:-1],
        ))
        reciprocal = dot(hinges, hinge_coefficients)
        velocity = scale(mul(delta, reciprocal), 3.4)
        rounded = clip01(affine(velocity, round_w, round_b))
        displacement = affine(rounded, displacement_w)
        flying_endpoint = add(point, displacement)
        first_type = sl(kind, 0, 1)
        endpoint = add(
            mul(flying_endpoint, first_type),
            mul(xy, inv(first_type)),
        )
        return minimum(maximum(endpoint, enemy_minimum),
                       enemy_maximum)

    def enemy_tasks(packed, best):
        kind = sl(packed, 0, 3)
        point = sl(packed, 3, 5)
        active = sl(packed, 5, 6)
        rank = dot(kind, [1, 2, 3])

        difference = sub(spatial(target_enemy_rank), rank)
        below = clip01(difference)
        ready = clip01(scale(absolute(difference), -1, 1))

        l1, self_mask, inside_goal = origin_fields(point)
        same_kind = sum_channels(mul(enemies, kind))
        partner = nearest(relu(sub(same_kind, self_mask)), l1)
        destination = nearest(empty_enemy_goals, l1, fruit_occupied)
        vulnerable_fruit = nearest(protected_map, l1)

        own_count = sum_channels(mul(enemy_counts, kind))
        has_partner = clip01(scale(own_count, 1, -1))
        selected = enemy_selection(point)
        endpoint = enemy_endpoint(kind, point)
        own_selected = sum_channels(mul(enemy_incidence, kind))
        merge_bonus = clip01(scale(own_selected, 1, -1))

        pair_before = cheb(point, partner)
        pair_after = cheb(endpoint, partner)
        goal_before = cheb(point, destination)
        goal_after = cheb(endpoint, destination)

        merge_score = weighted([
            (sub(pair_before, pair_after), 2),
            (pair_after, -0.20),
            (merge_bonus, 45),
            (fruit_finished, 35),
        ], 22)
        merge_score = gated(merge_score, product(
            active, below, need_enemy_fabrication,
            has_partner, selected
        ))

        arrival = clip01(scale(goal_after, -2, 9))
        delivery_score = weighted([
            (sub(goal_before, goal_after), 2),
            (goal_after, -0.15),
            (arrival, 30),
            (fruit_finished, 35),
            (merge_bonus, -10000),
        ], 12)
        delivery_score = gated(delivery_score, product(
            active, ready, need_enemy_delivery,
            inv(inside_goal), selected
        ))

        threat_before = cheb(point, vulnerable_fruit)
        threat_after = cheb(endpoint, vulnerable_fruit)
        nearby = clip01(scale(threat_before, -1, 18))
        repel_score = weighted([
            (sub(threat_after, threat_before), 2),
        ], 12)
        repel_score = gated(repel_score, product(
            active, has_protected, nearby, selected
        ))

        best_next = maximum(
            best, maximum(repel_score,
                          maximum(merge_score, delivery_score))
        )
        rotated = cat(sl(packed, 6, 48), sl(packed, 0, 6))
        return [rotated, best_next]

    best = repeat(
        8, [enemy_records, best_fruit], enemy_tasks
    )[1]

    # Recovery supervision. Two corrective undos are allowed; another
    # detected failure prefers reset over continuing an undo cycle.
    required_mass = dot(fruit_req, [2 ** i for i in range(9)])
    current_mass = dot(fruit_counts, [2 ** i for i in range(9)])
    deficient = clip01(scale(
        sub(required_mass, current_mass), 2, -0.5
    ))
    red_header = gmax(mul(sl("x0", 2, 3), header))
    bad = gmax(maximum(deficient, red_header))

    no_task = clip01(scale(gmax(best), -1, -1000))
    unreadable_spec = clip01(inv(sum_channels(requirements)))
    forced_reset = gmax(maximum(no_task, unreadable_spec))
    recovery = gmax(maximum(bad, forced_reset))

    failure_count = sl(history, 42, 43)
    failed_twice = gmax(clip01(scale(failure_count, 1, -1)))
    beginning = affine("aprev", -np.ones((1, 8), np.float32), [1])
    previous_undo = sl("aprev", 7, 8)
    reset_preferred = gmax(clip01(add(
        add(beginning, previous_undo),
        add(failed_twice, forced_reset),
    )))
    reset_bad = mul(recovery, reset_preferred)

    previous_point = sl(history, 36, 38)
    previous_valid = sl(history, 38, 39)
    banned_point = sl(history, 39, 41)
    banned_strength = sl(history, 41, 42)

    repetition = mul(
        clip01(scale(cheb(xy, previous_point), -1, 4)),
        previous_valid,
    )
    rejected_region = mul(
        clip01(scale(cheb(xy, banned_point), -0.25, 1)),
        banned_strength,
    )

    final_score = weighted([
        (best, 1),
        (fruit_danger, -120),
        (fruit_mixing, -10000),
        (enemy_mixing, -3000),
        (consuming_ready, -10000),
        (inv(board), -1000000),
        (repetition, -1.5),
        (rejected_region, -1000),
        (index, -0.0001),
    ])
    click = scale(final_score, 1, name="click")

    # Deterministic click-coordinate extraction for recurrent memory.
    top_click = gmax(click)
    maximal = clip01(scale(sub(click, top_click), 1.0e7, 1))
    tie_score = mul(maximal, scale(index, -1, 4096))
    tie_max = gmax(tie_score)
    chosen_click = mul(
        maximal, clip01(scale(sub(tie_score, tie_max), 2, 1))
    )
    candidate_point = scale(gmean(mul(xy, chosen_click)), 4096)

    good = inv(recovery)
    remembered_point = add(
        mul(candidate_point, good), mul(previous_point, recovery)
    )
    remembered_valid = add(good, mul(previous_valid, recovery))
    previous_was_click = sl("aprev", 6, 7)
    reject_now = mul(bad, previous_was_click)
    new_banned_point = add(
        mul(previous_point, reject_now),
        mul(banned_point, inv(reject_now)),
    )
    new_banned_strength = add(
        reject_now,
        mul(scale(banned_strength, 0.85), inv(reject_now)),
    )
    new_failure_count = add(failure_count, bad)

    mem_value = cat(
        goal_records, requirements,
        remembered_point, remembered_valid,
        new_banned_point, new_banned_strength,
        new_failure_count,
    )
    affine(mem_value, np.eye(43, dtype=np.float32), name="mem_next")

    logit_w = np.zeros((8, 2), np.float32)
    logit_b = np.full(8, -100, np.float32)
    logit_b[0] = -20
    logit_b[6] = 10
    logit_b[7] = -10
    logit_w[0] = [110, 30]
    logit_w[7] = [110, 0]
    affine(cat(recovery, reset_bad), logit_w, logit_b, name="logits")

    # Static accounting only; this does not execute the network.
    def graph_statistics(nodes):
        node_count = 0
        evaluations = 0
        macs = 0
        spatial_channels = 0
        for nd in nodes:
            node_count += 1
            if nd["op"] == "loop":
                a, b, c, d = graph_statistics(nd["body"])
                node_count += a
                evaluations += 1 + nd["T"] * b
                macs += nd["T"] * c
                spatial_channels += d
                for state in nd["state"]:
                    ch, sp = shapes[state["id"]]
                    if sp:
                        spatial_channels += ch
                continue
            evaluations += 1
            ch, sp = shapes[nd["id"]]
            if sp:
                spatial_channels += ch
            op = nd["op"]
            if op == "conv":
                macs += int(np.prod(params[nd["w"]].shape)) * 4096
            elif op == "dense":
                macs += int(np.prod(params[nd["w"]].shape))
            elif op in ("add", "sub", "mul", "max2", "min2"):
                macs += ch * 4096
            elif op in ("vadd", "vmul"):
                macs += ch
            elif op in ("rowmax", "colmax", "rowsum", "colsum",
                        "gmax", "gmean"):
                macs += shapes[nd["in"][0]][0] * 4096
        return node_count, evaluations, macs, spatial_channels

    nodes, evaluations, macs, channels = graph_statistics(graph)
    assert sum(a.size for a in params.values()) <= 150000
    assert nodes <= 1000
    assert evaluations <= 10000
    assert macs <= 1500000000
    assert channels + 19 <= 4096
    assert all(c <= (512 if sp else 4096)
               for c, sp in shapes.values())
    assert all(np.isfinite(a).all() for a in params.values())

    return {
        "graph": graph,
        "memory_size": 43,
        "params": params,
    }
