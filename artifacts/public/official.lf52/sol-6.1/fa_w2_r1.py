import numpy as np


def build():
    M = 101
    P = {}
    graph = []
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (M,),
        "aprev": (8,),
        "avail": (8,),
    }
    serial = [0]
    scope = [{"nodes": graph, "cache": {}, "shifts": {}, "zeros": {}}]
    parameter_cache = {}
    shift_parameters = {}
    zero_ids = set()

    def uid(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(value):
        a = np.asarray(value, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = uid("p")
            P[name] = a
            parameter_cache[key] = name
        return parameter_cache[key]

    def infer(op, inputs, fields):
        ss = [shapes[x] for x in inputs]
        if op == "conv":
            return (P[fields["w"]].shape[0], 64, 64)
        if op == "dense":
            return (P[fields["w"]].shape[0],)
        if op in ("concat", "vconcat"):
            return (sum(s[0] for s in ss),) + ss[0][1:]
        if op in ("slice", "vslice"):
            return (fields["stop"] - fields["start"],) + ss[0][1:]
        if op in ("gmax", "gmean"):
            return (ss[0][0],)
        if op in ("add", "sub", "mul", "max2", "min2"):
            return (max(s[0] for s in ss), 64, 64)
        if op in ("vadd", "vmul"):
            return (max(s[0] for s in ss),)
        return ss[0]

    def zero(shape):
        key = tuple(shape)
        if key not in scope[0]["zeros"]:
            if len(shape) == 3:
                z = node(
                    "conv", ["coords"],
                    w=parameter(np.zeros((shape[0], 3, 1, 1), np.float32))
                )
            else:
                z = node(
                    "dense", ["avail"],
                    w=parameter(np.zeros((shape[0], 8), np.float32))
                )
            zero_ids.add(z)
            scope[0]["zeros"][key] = z
        return scope[0]["zeros"][key]

    def node(op, inputs, name=None, **fields):
        inputs = list(inputs)
        out_shape = infer(op, inputs, fields)
        if name is None:
            if op in ("mul", "vmul") and any(x in zero_ids for x in inputs):
                return zero(out_shape)
            if op == "sub" and inputs[0] == inputs[1]:
                return zero(out_shape)
            if op in ("add", "vadd"):
                for i in range(2):
                    other = inputs[1 - i]
                    if inputs[i] in zero_ids and shapes[other] == out_shape:
                        return other
            if op in ("relu", "vrelu", "abs", "tanh", "vtanh"):
                if inputs[0] in zero_ids:
                    return inputs[0]
            if op in ("conv", "dense") and inputs[0] in zero_ids:
                if "b" not in fields or not np.any(P[fields["b"]]):
                    return zero(out_shape)
            if op in ("slice", "vslice"):
                if fields["start"] == 0 and fields["stop"] == shapes[inputs[0]][0]:
                    return inputs[0]
                if inputs[0] in zero_ids:
                    return zero(out_shape)
            key = (op, tuple(inputs), tuple(sorted(fields.items())))
            if key in scope[0]["cache"]:
                return scope[0]["cache"][key]
        else:
            key = None

        name = name or uid()
        nd = {"id": name, "op": op, "in": inputs}
        nd.update(fields)
        scope[0]["nodes"].append(nd)
        shapes[name] = out_shape
        if key is not None:
            scope[0]["cache"][key] = name
        return name

    def cat(xs):
        xs = list(xs)
        return xs[0] if len(xs) == 1 else node("concat", xs)

    def vcat(xs):
        xs = list(xs)
        return xs[0] if len(xs) == 1 else node("vconcat", xs)

    def sl(x, start, stop):
        return node(
            "slice" if len(shapes[x]) == 3 else "vslice",
            [x], start=start, stop=stop
        )

    def linear(x, weights, bias=None, name=None):
        w = np.asarray(weights, np.float32)
        spatial = len(shapes[x]) == 3
        if spatial and w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node(
            "conv" if spatial else "dense", [x], name=name, **fields
        )

    def aff(xs, coefficients, bias=0):
        c = shapes[xs[0]][0]
        terms = {}
        for x, a in zip(xs, coefficients):
            assert shapes[x] == (c, 64, 64)
            if x not in zero_ids:
                terms[x] = terms.get(x, 0.0) + float(a)
        terms = {x: a for x, a in terms.items() if a != 0}
        if not terms:
            if bias == 0:
                return zero((c, 64, 64))
            return linear(
                "coords", np.zeros((c, 3), np.float32),
                np.full(c, bias, np.float32)
            )
        xx = list(terms)
        aa = list(terms.values())
        if bias == 0:
            if len(xx) == 1 and aa[0] == 1:
                return xx[0]
            if len(xx) == 2:
                if aa == [1, 1]:
                    return node("add", xx)
                if aa == [1, -1]:
                    return node("sub", xx)
                if aa == [-1, 1]:
                    return node("sub", [xx[1], xx[0]])
        w = np.zeros((c, c * len(xx)), np.float32)
        ii = np.arange(c)
        for j, a in enumerate(aa):
            w[ii, j * c + ii] = a
        return linear(cat(xx), w, np.full(c, bias, np.float32))

    def va(xs, coefficients, bias=0):
        terms = {}
        for x, a in zip(xs, coefficients):
            assert shapes[x] == (1,)
            if x not in zero_ids:
                terms[x] = terms.get(x, 0.0) + float(a)
        terms = {x: a for x, a in terms.items() if a != 0}
        if not terms:
            return linear("avail", np.zeros((1, 8), np.float32), [bias])
        if len(terms) == 1 and bias == 0:
            x, a = next(iter(terms.items()))
            if a == 1:
                return x
        return linear(
            vcat(list(terms)), [list(terms.values())], [bias]
        )

    def scale(x, a, b=0):
        if len(shapes[x]) == 3:
            return aff([x], [a], b)
        if a == 1 and b == 0:
            return x
        if a == 0 and b == 0:
            return zero(shapes[x])
        c = shapes[x][0]
        return linear(
            x, np.eye(c, dtype=np.float32) * a,
            np.full(c, b, np.float32)
        )

    def relu(x):
        return node("relu" if len(shapes[x]) == 3 else "vrelu", [x])

    def clip(x):
        if len(shapes[x]) == 3:
            return node("min2", [relu(x), one])
        a = relu(x)
        b = relu(scale(a, 1, -1))
        c = shapes[x][0]
        return linear(
            vcat([a, b]),
            np.concatenate([np.eye(c), -np.eye(c)], axis=1)
        )

    def mul(a, b):
        spatial = len(shapes[a]) == 3 or len(shapes[b]) == 3
        return node("mul" if spatial else "vmul", [a, b])

    def pool(x):
        return node("gmax", [x])

    def total(x):
        return scale(node("gmean", [x]), 4096)

    def maximum(xs):
        xs = list(dict.fromkeys(xs))
        y = xs[0]
        for x in xs[1:]:
            y = node("max2", [y, x])
        return y

    def vmax(xs):
        y = xs[0]
        for x in xs[1:]:
            y = node("vadd", [relu(va([y, x], [1, -1])), x])
        return y

    def shift(x, dx, dy):
        if dx == 0 and dy == 0:
            return x
        if x in zero_ids:
            return x
        assert dx == 0 or dy == 0
        assert (abs(dx) + abs(dy)) % 3 == 0
        c = shapes[x][0]
        sx = 0 if dx == 0 else (3 if dx > 0 else -3)
        sy = 0 if dy == 0 else (3 if dy > 0 else -3)
        pk = (c, sx, sy)
        if pk not in shift_parameters:
            w = np.zeros((c, c, 7, 7), np.float32)
            for i in range(c):
                w[i, i, 3 + sy, 3 + sx] = 1
            shift_parameters[pk] = parameter(w)
        previous = x
        for t in range(1, (abs(dx) + abs(dy)) // 3 + 1):
            key = (x, sx * t, sy * t)
            if key not in scope[0]["shifts"]:
                scope[0]["shifts"][key] = node(
                    "conv", [previous], w=shift_parameters[pk]
                )
            previous = scope[0]["shifts"][key]
        return previous

    def pattern(conditions, k):
        w = np.zeros((1, 16, k, k), np.float32)
        h = k // 2
        for dy, dx, colors in conditions:
            for color in colors:
                w[0, color, h + dy, h + dx] = 1
        return relu(linear("x0", w, [1 - len(conditions)]))

    def loop(initials, iterations, body_function, prefix):
        outer = scope[0]
        scope[0] = {"nodes": [], "cache": {}, "shifts": {}, "zeros": {}}
        states, state_ids = [], []
        for initial in initials:
            sid = uid("s")
            shapes[sid] = shapes[initial]
            states.append({"id": sid, "init": initial})
            state_ids.append(sid)
        next_values = body_function(state_ids)
        body = scope[0]["nodes"]
        scope[0] = outer
        lid = uid(prefix)
        nd = {
            "id": lid, "op": "loop", "T": iterations,
            "state": states, "body": body,
            "next": dict(zip(state_ids, next_values)),
        }
        scope[0]["nodes"].append(nd)
        outputs = []
        for sid, initial in zip(state_ids, initials):
            out = lid + "." + sid
            shapes[out] = shapes[initial]
            outputs.append(out)
        return outputs, nd

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    action_ids = [1, 4, 2, 3]

    ball_points = [
        (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1), (0, 2),
        (1, -1), (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
    ]

    def ball(color):
        return pattern([(y, x, [color]) for y, x in ball_points], 5)

    green, red, blue = ball(14), ball(8), ball(9)
    gray = pool(ball(2))
    box_points = (
        [(-2, x) for x in (-1, 0, 1, 2)]
        + [(-1, -1), (-1, 2), (0, -1), (0, 2)]
        + [(1, x) for x in (-1, 0, 1, 2)]
    )
    box = pattern([(y, x, [15]) for y, x in box_points], 5)
    border = [
        (y - 2, x - 2, [0, 11, 3])
        for y in range(6) for x in range(6)
        if y in (0, 5) or x in (0, 5)
    ]
    corners = [
        (y - 2, x - 2, [1, 12, 3, 15, 7, 5])
        for y in (1, 4) for x in (1, 4)
    ]
    tile = pattern(border + corners, 7)
    orange = pattern([
        (y - 2, x - 2, [11])
        for y in (0, 5) for x in (0, 5)
    ], 7)
    live = aff([green, red], [1, 1])
    occupied = clip(aff([green, red, blue, box], [1, 1, 1, 1]))
    floor = maximum([tile, occupied])

    phase = sl("mem", 0, 1)
    unphase = va([phase], [-1], 1)

    def extract(indices):
        w = np.zeros((len(indices), M), np.float32)
        for i, j in enumerate(indices):
            w[i, j] = 1
        return linear("mem", w)

    def matches(rv, cv, validity=None):
        c = shapes[rv][0]
        dr = node("abs", [node("sub", [row, rv])])
        dc = node("abs", [node("sub", [col, cv])])
        distance = node("add", [dr, dc])
        match = clip(linear(
            distance, -6300 * np.eye(c, dtype=np.float32),
            np.full(c, 50, np.float32)
        ))
        if validity is not None:
            match = mul(match, validity)
        return linear(match, np.ones((1, c), np.float32))

    history = clip(matches(
        extract([5 + 2 * i for i in range(12)]),
        extract([6 + 2 * i for i in range(12)])
    ))
    taboo = clip(matches(
        extract([29 + 2 * i for i in range(8)]),
        extract([30 + 2 * i for i in range(8)]),
        extract([45 + i for i in range(8)])
    ))
    carrier_visits = matches(
        extract([53 + 2 * i for i in range(16)]),
        extract([54 + 2 * i for i in range(16)]),
        extract([85 + i for i in range(16)])
    )
    carrier_history = clip(carrier_visits)
    novel = aff([one, history], [1, -1])
    permitted = aff([one, taboo], [1, -1])
    zero_spatial = zero((1, 64, 64))
    zero_four = zero((4,))

    def lattice_body(states):
        s = states[0]
        return [maximum([s] + [
            shift(s, 6 * dx, 6 * dy) for dx, dy in dirs
        ])]

    lattice, _ = loop([floor], 22, lattice_body, "lattice")
    black = pattern([
        (y, x, [5]) for y in (0, 1) for x in (0, 1)
    ], 3)
    track = maximum([mul(black, lattice[0]), orange])
    movable_track = relu(aff([track, orange], [1, -1]))
    empty = mul(relu(aff([floor, occupied], [1, -1])), permitted)

    mids = [shift(occupied, 6 * dx, 6 * dy) for dx, dy in dirs]
    destinations = [
        shift(empty, 12 * dx, 12 * dy) for dx, dy in dirs
    ]
    capture, legal = [], []
    for i, (dx, dy) in enumerate(dirs):
        cg = mul(green, shift(green, 6 * dx, 6 * dy))
        cr = mul(red, shift(red, 6 * dx, 6 * dy))
        cap = mul(aff([cg, cr], [1, 1]), destinations[i])
        base = mul(mul(live, mids[i]), destinations[i])
        fresh = shift(novel, 12 * dx, 12 * dy)
        legal.append(mul(base, maximum([fresh, cap])))
        capture.append(cap)

    future_green = maximum([
        mul(shift(green, 6 * dx, 6 * dy), destinations[i])
        for i, (dx, dy) in enumerate(dirs)
    ])
    future_red = maximum([
        mul(shift(red, 6 * dx, 6 * dy), destinations[i])
        for i, (dx, dy) in enumerate(dirs)
    ])

    boarding = maximum([
        mul(mids[i], shift(track, 12 * dx, 12 * dy))
        for i, (dx, dy) in enumerate(dirs)
    ])
    transport = maximum([orange, boarding])
    alignment = node("add", [
        node("rowsum", [live]), node("colsum", [live])
    ])
    surface = aff([alignment, transport], [4, 20])
    priorities = [
        aff([
            shift(surface, 12 * dx, 12 * dy),
            shift(novel, 12 * dx, 12 * dy)
        ], [1, 8], 100)
        for dx, dy in dirs
    ]

    def encode(scores, stride=8192):
        c = shapes[scores][0]
        w = np.zeros((c, c + 2), np.float32)
        w[np.arange(c), np.arange(c)] = stride
        w[:, c] = -4032
        w[:, c + 1] = -63
        return linear(cat([scores, row, col]), w)

    def winner(scores, mask):
        return mul(relu(scale(
            node("sub", [scores, pool(scores)]), 4, 1
        )), mask)

    seeds, exists, seed_captures = [], [], []
    seed_targets, projected_green, projected_red = [], [], []
    for i, (dx, dy) in enumerate(dirs):
        quality = aff([priorities[i], capture[i]], [1, 150])
        scores = encode(mul(quality, legal[i]))
        seed = winner(scores, legal[i])
        seeds.append(seed)
        exists.append(pool(legal[i]))
        seed_captures.append(total(mul(seed, capture[i])))
        seed_targets.append(shift(seed, -12 * dx, -12 * dy))
        projected_green.append(shift(
            mul(seed, green), -12 * dx, -12 * dy
        ))
        projected_red.append(shift(
            mul(seed, red), -12 * dx, -12 * dy
        ))

    continuation = pool(aff([
        mul(cat(projected_green), future_green),
        mul(cat(projected_red), future_red)
    ], [1, 1]))
    target_transport = pool(mul(cat(seed_targets), transport))
    first_captures = vcat(seed_captures)
    initial_count = total(live)
    count_weights = np.zeros((4, 9), np.float32)
    count_weights[:, 0] = 1
    for i in range(4):
        count_weights[i, 1 + i] = -1
        count_weights[i, 5 + i] = -0.55
    predicted_counts = linear(
        vcat([initial_count, first_captures, continuation]),
        count_weights
    )

    values = []
    for i in range(4):
        values.append(va([
            sl(predicted_counts, i, i + 1),
            sl(target_transport, i, i + 1),
            seed_captures[i], exists[i]
        ], [-100, 15, 20, 6000], -6000 - 0.25 * i))

    value_maps = mul(one, vcat(values))
    best_value = pool(maximum([
        sl(value_maps, i, i + 1) for i in range(4)
    ]))
    direction_choice = mul(relu(scale(
        node("sub", [value_maps, best_value]), 16, 1
    )), vcat(exists))
    source = linear(
        mul(cat(seeds), direction_choice),
        np.ones((1, 4), np.float32)
    )
    target = linear(
        mul(cat(seed_targets), direction_choice),
        np.ones((1, 4), np.float32)
    )
    has_jump = pool(source)
    chosen_capture = linear(
        mul(first_captures, pool(direction_choice)),
        np.ones((1, 4), np.float32)
    )

    static_mask = aff([one, orange], [1, -1])
    sg = mul(green, static_mask)
    sr = mul(red, static_mask)
    sb = mul(box, static_mask)
    static_colors = cat([sg, sr])
    static_live = aff([sg, sr], [1, 1])
    static_obstacles = clip(
        aff([sg, sr, blue, sb], [1, 1, 1, 1])
    )
    static_empty = mul(empty, static_mask)

    carrier_maps = [
        mul(orange, green),
        mul(orange, red),
        mul(orange, box),
        mul(orange, relu(aff([one, occupied], [1, -1]))),
    ]
    carriers = cat(carrier_maps)
    carrier_union = linear(carriers, np.ones((1, 4), np.float32))

    def pair_aff(a, b, wa, wb):
        w = np.zeros((2, 4), np.float32)
        for i in range(2):
            w[i, i] = wa[i]
            w[i, 2 + i] = wb[i]
        return linear(cat([a, b]), w)

    land_goals, board_goals, box_goals = [], [], []
    for dx, dy in dirs:
        h = shift(static_obstacles, 6 * dx, 6 * dy)
        e = shift(static_empty, 12 * dx, 12 * dy)
        fresh = shift(novel, 12 * dx, 12 * dy)
        matching = shift(static_colors, 6 * dx, 6 * dy)
        ordinary = mul(h, maximum([fresh, matching]))
        land_goals.append(mul(
            pair_aff(ordinary, matching, [240, 165], [120, 120]), e
        ))

        boarding_source = shift(static_colors, -12 * dx, -12 * dy)
        boarding_mid = shift(static_obstacles, -6 * dx, -6 * dy)
        boarding_match = mul(
            boarding_source,
            shift(static_colors, -6 * dx, -6 * dy)
        )
        boarded = mul(
            mul(boarding_source, boarding_mid),
            maximum([novel, boarding_match])
        )
        board_goals.append(pair_aff(
            boarded, boarding_match, [235, 170], [75, 75]
        ))
        box_goals.append(mul(
            shift(static_live, -6 * dx, -6 * dy),
            mul(
                shift(static_empty, 6 * dx, 6 * dy),
                shift(novel, 6 * dx, 6 * dy)
            )
        ))

    land_pair = maximum(land_goals)
    board_pair = maximum(board_goals)
    board_goal = maximum([sl(board_pair, 0, 1), sl(board_pair, 1, 2)])
    box_goal = scale(maximum(box_goals), 195)

    frontier = maximum([
        scale(clip(scale(col, 63, -55)), 100),
        scale(clip(scale(col, -63, 7)), 45),
        scale(clip(scale(row, 63, -55)), 85),
        scale(clip(scale(row, -63, 7)), 40),
    ])
    frontier = mul(frontier, aff([one, carrier_history], [1, -1]))

    raw_goals = [
        maximum([sl(land_pair, 0, 1), scale(frontier, 1.5)]),
        maximum([sl(land_pair, 1, 2), frontier]),
        maximum([box_goal, scale(frontier, 0.7)]),
        maximum([board_goal, scale(frontier, 0.75)]),
    ]
    allowed_track = mul(track, permitted)
    goals = [
        mul(relu(aff([goal, carrier_visits], [1, -30])), allowed_track)
        for goal in raw_goals
    ]
    routes = []
    route_nodes = []

    for goal in goals:
        def route_body(states, goal=goal):
            propagated = maximum([
                shift(states[0], 6 * dx, 6 * dy)
                for dx, dy in dirs
            ])
            propagated = mul(
                relu(aff([propagated, carrier_visits], [1, -3], -2)),
                track
            )
            return [maximum([goal, propagated])]

        out, loop_node = loop([goal], 64, route_body, "route")
        routes.append(out[0])
        route_nodes.append(loop_node)

    packed_routes = cat(routes)
    potential_w = np.zeros((4, 5), np.float32)
    potential_w[np.arange(4), np.arange(4)] = 1
    potential_w[:, 4] = -8
    potential_b = np.asarray([230, 200, 200, 205], np.float32)

    direction_fields, direction_masks = [], []
    for dx, dy in dirs:
        neighbor_track = shift(movable_track, 6 * dx, 6 * dy)
        valid = mul(carriers, neighbor_track)
        potential = linear(cat([
            shift(packed_routes, 6 * dx, 6 * dy),
            shift(carrier_visits, 6 * dx, 6 * dy)
        ]), potential_w, potential_b)
        direction_fields.append(linear(
            mul(valid, potential), np.ones((1, 4), np.float32)
        ))
        direction_masks.append(mul(carrier_union, neighbor_track))

    arrows, arrow_winners, arrow_valid = [], [], []
    for i in range(4):
        encoded = encode(direction_fields[i])
        present = pool(direction_masks[i])
        arrow_valid.append(present)
        arrow_winners.append(winner(encoded, direction_masks[i]))
        score = scale(pool(encoded), 1 / 8192)
        previous = sl("aprev", action_ids[i], action_ids[i] + 1)
        opposite_index = action_ids[(i + 2) % 4]
        opposite = sl("aprev", opposite_index, opposite_index + 1)
        arrows.append(va(
            [score, present, previous, opposite],
            [1, 10000, 0.25, -0.5],
            -10000 - 0.03125 * i
        ))

    best_arrow = vmax(arrows)
    movable = vmax(arrow_valid)
    dead = mul(
        mul(va([movable], [-1], 1), va([has_jump], [-1], 1)),
        unphase
    )
    recover = clip(va([gray, dead], [1, 1]))
    healthy = va([recover], [-1], 1)

    jump_value = va([has_jump, chosen_capture], [10650, 550], -10000)
    click_value = va(
        [mul(jump_value, unphase), phase, recover],
        [1, 10000, -30000]
    )
    select_gate = mul(mul(
        unphase, clip(va([click_value, best_arrow], [1, -1]))
    ), healthy)
    execute_gate = mul(phase, healthy)
    arrow_gate = mul(mul(
        unphase, clip(va([best_arrow, click_value], [1, -1]))
    ), healthy)

    remembered_target = matches(extract([3]), extract([4]))
    node("add", [
        mul(source, unphase), mul(remembered_target, phase)
    ], name="click")

    output_w = np.zeros((8, 6), np.float32)
    output_b = np.full(8, -1000000, np.float32)
    for i, action in enumerate(action_ids):
        output_w[action, i] = 1
        output_b[action] = 0
    output_w[6, 4] = 1
    output_b[6] = 0
    output_w[7, 5] = 1030000
    linear(
        vcat(arrows + [click_value, recover]),
        output_w, output_b, name="logits"
    )

    arrow_choice = relu(scale(node(
        "sub", [mul(one, vcat(arrows)), best_arrow]
    ), 16, 1))
    mover = linear(
        mul(cat(arrow_winners), arrow_choice),
        np.ones((1, 4), np.float32)
    )

    coordinates = cat([row, col])
    source_coords = total(mul(source, coordinates))
    target_coords = total(mul(target, coordinates))
    mover_coords = total(mul(mover, coordinates))

    select_w = np.zeros((M, M + 4), np.float32)
    select_w[:, :M] = np.eye(M, dtype=np.float32)
    select_w[0] = 0
    for i in range(4):
        select_w[1 + i] = 0
        select_w[1 + i, M + i] = 1
    select_b = np.zeros(M, np.float32)
    select_b[0] = 1
    select_memory = linear(
        vcat(["mem", source_coords, target_coords]), select_w, select_b
    )

    execute_w = np.eye(M, dtype=np.float32)
    execute_w[0] = 0
    for i in range(12):
        execute_w[5 + 2 * i] = 0
        execute_w[6 + 2 * i] = 0
        if i == 0:
            execute_w[5, 1] = 1
            execute_w[6, 2] = 1
        elif i == 1:
            execute_w[7, 3] = 1
            execute_w[8, 4] = 1
        else:
            execute_w[5 + 2 * i, 5 + 2 * (i - 2)] = 1
            execute_w[6 + 2 * i, 6 + 2 * (i - 2)] = 1
    execute_memory = linear("mem", execute_w)

    recovery_w = np.eye(M, dtype=np.float32)
    recovery_w[0] = 0
    recovery_b = np.zeros(M, np.float32)
    for i in range(8):
        recovery_w[29 + 2 * i] = 0
        recovery_w[30 + 2 * i] = 0
        recovery_w[45 + i] = 0
        if i == 0:
            recovery_w[29, 3] = 1
            recovery_w[30, 4] = 1
            recovery_b[45] = 1
        else:
            recovery_w[29 + 2 * i, 29 + 2 * (i - 1)] = 1
            recovery_w[30 + 2 * i, 30 + 2 * (i - 1)] = 1
            recovery_w[45 + i, 45 + i - 1] = 1
    recovery_memory = linear("mem", recovery_w, recovery_b)

    move_w = np.zeros((M, M + 2), np.float32)
    move_w[:, :M] = np.eye(M, dtype=np.float32)
    move_b = np.zeros(M, np.float32)
    for i in range(16):
        move_w[53 + 2 * i] = 0
        move_w[54 + 2 * i] = 0
        move_w[85 + i] = 0
        if i == 0:
            move_w[53, M] = 1
            move_w[54, M + 1] = 1
            move_b[85] = 1
        else:
            move_w[53 + 2 * i, 53 + 2 * (i - 1)] = 1
            move_w[54 + 2 * i, 54 + 2 * (i - 1)] = 1
            move_w[85 + i, 85 + i - 1] = 1
    move_memory = linear(vcat(["mem", mover_coords]), move_w, move_b)

    hold_gate = va(
        [select_gate, execute_gate, recover, arrow_gate],
        [-1, -1, -1, -1], 1
    )
    parts = [
        mul(select_memory, select_gate),
        mul(execute_memory, execute_gate),
        mul(recovery_memory, recover),
        mul(move_memory, arrow_gate),
        mul("mem", hold_gate),
    ]
    result = parts[0]
    for part in parts[1:-1]:
        result = node("vadd", [result, part])
    node("vadd", [result, parts[-1]], name="mem_next")

    def prune(nodes, required):
        needed = set(required)
        kept = []
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                exports = {
                    nd["id"] + "." + s["id"] for s in nd["state"]
                }
                if not (exports & needed):
                    continue
                body, external = prune(nd["body"], set(nd["next"].values()))
                nd["body"] = body
                state_names = {s["id"] for s in nd["state"]}
                needed.update(external - state_names)
                for s in nd["state"]:
                    if "init" in s:
                        needed.add(s["init"])
                kept.append(nd)
            elif nd["id"] in needed:
                needed.update(nd["in"])
                kept.append(nd)
        kept.reverse()
        defined = {nd["id"] for nd in kept}
        for nd in kept:
            if nd["op"] == "loop":
                defined.update(
                    nd["id"] + "." + s["id"] for s in nd["state"]
                )
        return kept, needed - defined

    def replace_inputs(nodes, replacements):
        for nd in nodes:
            if nd["op"] == "loop":
                for s in nd["state"]:
                    if "init" in s:
                        s["init"] = replacements.get(s["init"], s["init"])
                replace_inputs(nd["body"], replacements)
                nd["next"] = {
                    k: replacements.get(v, v)
                    for k, v in nd["next"].items()
                }
            else:
                nd["in"] = [replacements.get(x, x) for x in nd["in"]]

    def count_nodes(nodes):
        return sum(
            1 + (count_nodes(nd["body"]) if nd["op"] == "loop" else 0)
            for nd in nodes
        )

    # Only graph bookkeeping occurs here. All surviving loops retain
    # their actual non-empty propagation bodies.
    original_graph = graph
    graph, _ = prune(original_graph, {"logits", "click", "mem_next"})
    if count_nodes(graph) > 1000:
        replace_inputs(original_graph, {continuation: zero_four})
        graph, _ = prune(original_graph, {"logits", "click", "mem_next"})
    if count_nodes(graph) > 1000:
        replace_inputs(original_graph, {frontier: zero_spatial})
        graph, _ = prune(original_graph, {"logits", "click", "mem_next"})
    if count_nodes(graph) > 1000:
        replace_inputs(original_graph, {carrier_visits: zero_spatial})
        graph, _ = prune(original_graph, {"logits", "click", "mem_next"})

    def mac(nd):
        op = nd["op"]
        if op == "loop":
            return nd["T"] * sum(mac(x) for x in nd["body"])
        if op == "conv":
            return int(np.prod(P[nd["w"]].shape)) * 4096
        if op == "dense":
            return int(np.prod(P[nd["w"]].shape))
        if op in ("add", "sub", "mul", "max2", "min2", "vadd", "vmul"):
            return int(np.prod(shapes[nd["id"]]))
        if op in ("rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"):
            return int(np.prod(shapes[nd["in"][0]]))
        return 0

    total_mac = sum(mac(nd) for nd in graph)
    if total_mac > 1450000000:
        present_routes = [
            nd for nd in route_nodes if any(nd is x for x in graph)
        ]
        per_iteration = sum(
            sum(mac(x) for x in nd["body"]) for nd in present_routes
        )
        fixed_mac = total_mac - 64 * per_iteration
        iterations = max(
            1, min(64, int((1450000000 - fixed_mac) // per_iteration))
        )
        for nd in present_routes:
            nd["T"] = iterations

    def used_parameters(nodes):
        used = set()
        for nd in nodes:
            if nd["op"] == "loop":
                used.update(used_parameters(nd["body"]))
            else:
                for field in ("w", "b"):
                    if field in nd:
                        used.add(nd[field])
        return used

    used = used_parameters(graph)
    P = {name: value for name, value in P.items() if name in used}
    return {"graph": graph, "memory_size": M, "params": P}
