import numpy as np


def build():
    graph = []
    params = {}
    destination = graph
    serial = [0]
    parameter_cache = {}
    constant_cache = {}

    def parameter(value):
        a = np.asarray(value, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p_" + str(len(params))
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def node(op, inputs, **fields):
        serial[0] += 1
        name = "n_" + str(serial[0])
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        destination.append(nd)
        return name

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def linear(x, w, b=None):
        a = np.asarray(w, dtype=np.float32)
        return conv(x, a[:, :, None, None], b)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def maximum(a, b):
        return node("max2", [a, b])

    def relu(a):
        return node("relu", [a])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def vr(a):
        return node("vrelu", [a])

    def gmax(a):
        return node("gmax", [a])

    one = sl("coords", 2, 3)
    unit = gmax(one)

    def const(value):
        nonlocal destination
        a = np.asarray(value, dtype=np.float32).reshape(-1)
        key = tuple(float(v) for v in a)
        if key not in constant_cache:
            previous = destination
            destination = graph
            constant_cache[key] = dense(
                unit, np.zeros((len(a), 1), np.float32), a
            )
            destination = previous
        return constant_cache[key]

    def scale(a, factor):
        return vm(a, const(factor))

    def vsub(a, b):
        return va(a, scale(b, -1))

    def vmax(a, b):
        return va(vr(vsub(a, b)), b)

    def affine(a, factor, bias):
        return dense(a, [[factor]], [bias])

    def complement(a):
        return affine(a, -1, 1)

    def cap(a):
        return complement(vr(complement(a)))

    def binary_scalar(a):
        return cap(vr(affine(a, 1000, -500)))

    def binary_spatial(a):
        return node("min2", [
            relu(linear(a, [[1000]], [-500])), const(1)
        ])

    def total(a):
        return scale(node("gmean", [a]), 4096)

    def sum_vector(a, length, bias=0):
        return dense(a, np.ones((1, length), np.float32), [bias])

    # Exact masks for the interior five-pixel movement lattice.
    rw = np.zeros((10, 3), np.float32)
    cw = np.zeros((10, 3), np.float32)
    rw[:, 0] = 63
    cw[:, 1] = 63
    row_delta = node("abs", [
        linear("coords", rw, -np.arange(5, 51, 5))
    ])
    col_delta = node("abs", [
        linear("coords", cw, -np.arange(9, 55, 5))
    ])
    eqw = -1000 * np.eye(10, dtype=np.float32)
    rows = node("min2", [
        relu(linear(row_delta, eqw, np.full(10, 500))),
        const(1),
    ])
    cols = node("min2", [
        relu(linear(col_delta, eqw, np.full(10, 500))),
        const(1),
    ])
    lift_rows = np.zeros((100, 10), np.float32)
    lift_cols = np.zeros((100, 10), np.float32)
    for r in range(10):
        for c in range(10):
            lift_rows[10 * r + c, r] = 1
            lift_cols[10 * r + c, c] = 1
    cells = mul(
        linear(rows, lift_rows), linear(cols, lift_cols)
    )
    lattice = linear(cells, np.ones((1, 100), np.float32))
    coordinates = sl("coords", 0, 2)

    def vectorize(field):
        return total(mul(cells, field))

    def spatialize(vector):
        return linear(
            mul(cells, vector), np.ones((1, 100), np.float32)
        )

    def position_mask(rc):
        delta = node("abs", [sub(coordinates, rc)])
        return node("min2", [
            relu(linear(delta, [[-63000, -63000]], [500])),
            const(1),
        ])

    def position_vector(mask):
        return total(mul(coordinates, mask))

    def fixed_point(row, col):
        w = np.zeros((2, 3), np.float32)
        w[0, 0] = 63
        w[1, 1] = 63
        delta = node("abs", [
            linear("coords", w, [-row, -col])
        ])
        return node("min2", [
            relu(linear(delta, [[-1000, -1000]], [500])),
            const(1),
        ])

    anchor_w = np.zeros((1, 1, 5, 5), np.float32)
    anchor_w[0, 0, 4, 4] = 1
    anchor_parameter = parameter(anchor_w)

    def anchor(x):
        return node("conv", [x], w=anchor_parameter)

    shapes = [
        [[0, 0, -1], [-1, 0, 0], [0, -1, 0]],
        [[-1, 0, -1], [-1, 0, -1], [0, 0, 0]],
        [[0, -1, 0], [0, -1, 0], [0, 0, 0]],
        [[-1, 0, 0], [0, -1, 0], [-1, 0, -1]],
        [[-1, 0, -1], [0, 0, -1], [-1, 0, 0]],
        [[0, 0, 0], [-1, -1, 0], [0, -1, 0]],
    ]
    colours = [12, 9, 14, 8]
    configurations = [
        (s, c, r)
        for s in range(6)
        for c in range(4)
        for r in range(4)
    ]
    patterns = []
    for s, c, r in configurations:
        shape = np.rot90(np.asarray(shapes[s]), k=(-r) % 4)
        patterns.append(
            np.where(shape < 0, 5, colours[c]).astype(np.int32)
        )

    # Only these five colours occur in the HUD and goal templates.
    detector_colours = [5, 8, 9, 12, 14]
    colour_index = {c: i for i, c in enumerate(detector_colours)}
    project = np.zeros((5, 16), np.float32)
    for i, c in enumerate(detector_colours):
        project[i, c] = 1
    compact_colours = linear("x0", project)

    def configuration_detector(source, kernel=3, spacing=1):
        w = np.zeros((96, 5, kernel, kernel), np.float32)
        for i, pattern in enumerate(patterns):
            for r in range(3):
                for c in range(3):
                    w[i, colour_index[int(pattern[r, c])],
                      r * spacing, c * spacing] = 1
        return relu(conv(source, w, np.full(96, -8, np.float32)))

    hud_detection = configuration_detector(
        compact_colours, kernel=5, spacing=2
    )
    hud = gmax(mul(hud_detection, fixed_point(57, 5)))

    shifted_input_w = np.zeros((5, 5, 5, 5), np.float32)
    for c in range(5):
        shifted_input_w[c, c, 4, 4] = 1
    shifted_input = conv(compact_colours, shifted_input_w)
    goal_configurations = mul(
        configuration_detector(shifted_input), lattice
    )
    observed_goals = binary_spatial(linear(
        goal_configurations, np.ones((1, 96), np.float32)
    ))

    compress = np.zeros((14, 96), np.float32)
    expand = np.zeros((96, 14), np.float32)
    for i, (s, c, r) in enumerate(configurations):
        indices = [s, 6 + c, 10 + r]
        compress[indices, i] = 1
        expand[i, indices] = 1
    compress_parameter = parameter(compress)
    expand_parameter = parameter(expand)
    hud_components = node("dense", [hud], w=compress_parameter)

    player_w = np.zeros((1, 16, 5, 5), np.float32)
    for r in range(5):
        for c in range(5):
            player_w[0, 12 if r < 2 else 9, r, c] = 1
    player = mul(
        anchor(relu(conv("x0", player_w, [-24]))), lattice
    )
    player_vector = vectorize(player)

    def detect(pattern_list):
        w = np.zeros((len(pattern_list), 16, 3, 3), np.float32)
        for i, pattern in enumerate(pattern_list):
            for r in range(3):
                for c in range(3):
                    w[i, int(pattern[r, c]), r, c] = 1
        return relu(conv(
            "x0", w, np.full(len(pattern_list), -8, np.float32)
        ))

    station_patterns = [
        np.array([[0, 3, 3], [3, 0, 0], [3, 0, 3]]),
        np.array([[9, 14, 14], [9, 0, 8], [12, 12, 8]]),
        np.array([[3, 0, 3], [1, 0, 0], [3, 1, 3]]),
    ]
    station_detection = detect(station_patterns)
    observed_stations = [
        mul(anchor(sl(station_detection, i, i + 1)), lattice)
        for i in range(3)
    ]
    observed_refills = mul(anchor(detect([
        np.array([[11, 11, 11], [11, 3, 11], [11, 11, 11]])
    ])), lattice)

    # Memory: terrain, stations, selected goal, refills, counter history.
    station_records = []
    station_masks = []
    for i in range(3):
        old = vs("mem", 100 + 3 * i, 103 + 3 * i)
        visible = gmax(observed_stations[i])
        rc = va(
            vm(vs(old, 0, 2), complement(visible)),
            position_vector(observed_stations[i]),
        )
        present = cap(va(vs(old, 2, 3), visible))
        station_records.append(vcat(rc, present))
        station_masks.append(mul(position_mask(rc), present))

    station_all = maximum(
        maximum(station_masks[0], station_masks[1]),
        station_masks[2],
    )
    station_vectors = [vectorize(m) for m in station_masks]
    station_all_vector = vectorize(station_all)

    old_goal = vs("mem", 109, 126)
    old_goal_mask = position_mask(vs(old_goal, 0, 2))
    old_components = vs(old_goal, 2, 16)
    old_agreement = vr(sum_vector(
        vm(old_components, hud_components), 14, -2
    ))
    old_occupied = gmax(mul(player, old_goal_mask))
    keep_goal = vm(
        vs(old_goal, 16, 17),
        complement(vm(old_agreement, old_occupied)),
    )

    ranking = linear("coords", [[-4032, -63, 0]], [4096])
    ranked_goals = mul(observed_goals, ranking)
    peak = gmax(ranked_goals)
    selected = node("min2", [
        relu(linear(sub(ranked_goals, peak), [[1000]], [500])),
        const(1),
    ])
    selected = mul(
        mul(selected, observed_goals), complement(keep_goal)
    )
    new_goal_configuration = gmax(
        mul(goal_configurations, selected)
    )
    new_components = node(
        "dense", [new_goal_configuration], w=compress_parameter
    )
    goal_record = va(
        vm(old_goal, keep_goal),
        vcat(position_vector(selected), new_components, gmax(selected)),
    )
    goal_components = vs(goal_record, 2, 16)
    goal_mask = mul(
        position_mask(vs(goal_record, 0, 2)),
        vs(goal_record, 16, 17),
    )
    goal_state = vr(node(
        "dense", [goal_components],
        w=expand_parameter,
        b=parameter(np.full(96, -2, np.float32)),
    ))
    goal_match = sum_vector(vm(goal_state, hud), 96)
    goal_vector = vectorize(goal_mask)

    # Observe cell centres; palette 5 represents unobserved fog or goals.
    centre_w = np.zeros((2, 16, 5, 5), np.float32)
    centre_w[0, 5, 4, 4] = -1
    centre_w[1, 4, 4, 4] = 1
    centre = conv("x0", centre_w, [1, 0])
    seen = sl(centre, 0, 1)
    wall = sl(centre, 1, 2)
    terrain_observed = linear(cat(seen, wall), [[1, -2]])
    seen_vector = vectorize(seen)
    unseen_vector = vsub(const(1), seen_vector)
    terrain_memory = va(
        vm(vs("mem", 0, 100), unseen_vector),
        vectorize(terrain_observed),
    )
    refill_memory = va(
        vm(vs("mem", 126, 226), unseen_vector),
        vectorize(observed_refills),
    )

    remembered_floor = vectorize(maximum(station_all, goal_mask))
    terrain = vmax(terrain_memory, remembered_floor)
    open_vector = vr(terrain)
    absolute_terrain = va(vr(terrain), vr(scale(terrain, -1)))
    unknown_vector = vr(vsub(const(1), absolute_terrain))

    # Shared sparse lattice-neighbour matrices.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    shift_parameters = []
    for dr, dc in directions:
        w = np.zeros((100, 100), np.float32)
        for r in range(10):
            for c in range(10):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 10 and 0 <= cc < 10:
                    w[10 * r + c, 10 * rr + cc] = 1
        shift_parameters.append(parameter(w))

    def shift(vector, direction):
        return node("dense", [vector], w=shift_parameters[direction])

    unknown_neighbours = [shift(unknown_vector, d) for d in range(4)]
    adjacent_unknown = unknown_neighbours[0]
    for d in range(1, 4):
        adjacent_unknown = va(adjacent_unknown, unknown_neighbours[d])
    # All inputs here are binary; cap the neighbour count at one.
    frontier_vector = vm(
        vsub(const(1), vr(vsub(const(1), adjacent_unknown))),
        open_vector,
    )

    # Identify which direction a blue boundary pushes.
    blue = sl("x0", 1, 2)
    blue_w = np.zeros((4, 1, 7, 7), np.float32)
    for d, (dr, dc) in enumerate(
        [(3, 2), (-1, 2), (2, 3), (2, -1)]
    ):
        blue_w[d, 0, 3 + dr, 3 + dc] = 1
    blue_w2 = np.zeros((4, 4, 5, 5), np.float32)
    blue_w2[0, 0, 4, 2] = 1
    blue_w2[1, 1, 2, 2] = 1
    blue_w2[2, 2, 2, 4] = 1
    blue_w2[3, 3, 2, 2] = 1
    push = mul(conv(conv(blue, blue_w), blue_w2), lattice)
    push_vectors = [
        vectorize(sl(push, d, d + 1)) for d in range(4)
    ]
    push_any = push_vectors[0]
    for d in range(1, 4):
        push_any = va(push_any, push_vectors[d])
    nonpush = vr(vsub(const(1), push_any))

    next_open = [shift(open_vector, d) for d in range(4)]
    next_closed = [vsub(const(1), n) for n in next_open]

    all_goals_vector = vectorize(maximum(observed_goals, goal_mask))
    transit = vr(vsub(
        vsub(open_vector, station_all_vector), all_goals_vector
    ))
    refill_transit = vr(vsub(open_vector, all_goals_vector))
    targets = station_vectors + [
        goal_vector, frontier_vector, refill_memory
    ]
    seeds = [scale(target, 128) for target in targets]
    gates = [
        vmax(refill_transit if i == 5 else transit, targets[i])
        for i in range(6)
    ]

    # h: distance value. ray_d: value at the last floor cell before
    # the next wall in direction d, with no intervening activations.
    wave_body = []
    destination = wave_body
    wave_states = []
    wave_next = {}
    wave_names = []

    for i in range(6):
        h = "wave_value_" + str(i)
        ray_names = ["ray_" + str(i) + "_" + str(d) for d in range(4)]
        wave_names.append(h)
        wave_states.append({"id": h, "init": seeds[i]})
        for ray in ray_names:
            wave_states.append({"id": ray, "zeros": [100]})

        gated = vm(h, gates[i])
        neighbours = [shift(gated, d) for d in range(4)]
        best = neighbours[0]
        for d in range(1, 4):
            best = vmax(best, neighbours[d])
        ordinary_value = vr(va(best, const(-1)))

        forced_terms = []
        for d in range(4):
            ray = ray_names[d]
            wave_next[ray] = va(
                vm(shift(ray, d), next_open[d]),
                vm(h, next_closed[d]),
            )
            forced_terms.append(vm(ray, push_vectors[d]))
        forced_value = forced_terms[0]
        for d in range(1, 4):
            forced_value = va(forced_value, forced_terms[d])

        propagated = vm(va(
            vm(ordinary_value, nonpush), forced_value
        ), open_vector)
        wave_next[h] = vmax(propagated, seeds[i])

    destination = graph
    graph.append({
        "id": "waves",
        "op": "loop",
        "T": 26,
        "state": wave_states,
        "body": wave_body,
        "next": wave_next,
    })
    fields = ["waves." + name for name in wave_names]

    def value_at(field, source):
        return sum_vector(vm(field, source), 100)

    def distance_cost(field, source):
        value = value_at(field, source)
        missing = vr(complement(value))
        return va(
            affine(value, 1, -128),
            affine(missing, -1000, 0),
        )

    goal_terminal = va(scale(goal_state, 1000), const(-1000))
    costs = []
    bases = []
    for i in range(3):
        row = [
            const(-2) if i == j else
            distance_cost(fields[j], station_vectors[i])
            for j in range(5)
        ]
        costs.append(row)
        bases.append(vmax(
            va(goal_terminal, row[3]),
            affine(row[4], 1, -100),
        ))

    # Station-entry transitions for the 96 discrete configurations.
    configuration_index = {
        q: i for i, q in enumerate(configurations)
    }
    permutation_parameters = []
    for station in range(3):
        permutation = np.zeros((96, 96), np.float32)
        for i, (s, c, r) in enumerate(configurations):
            if station == 0:
                successor = ((s + 1) % 6, c, r)
            elif station == 1:
                successor = (s, (c + 1) % 4, r)
            else:
                successor = (s, c, (r + 1) % 4)
            permutation[i, configuration_index[successor]] = 1
        permutation_parameters.append(parameter(permutation))

    abstract_body = []
    destination = abstract_body
    abstract_names = ["shape_value", "colour_value", "rotation_value"]
    abstract_next = {}
    for i in range(3):
        best = bases[i]
        for j in range(3):
            best = vmax(best, va(abstract_names[j], costs[i][j]))
        abstract_next[abstract_names[i]] = node(
            "dense", [best], w=permutation_parameters[i]
        )
    destination = graph
    abstract_initial = const(np.full(96, -2000, np.float32))
    graph.append({
        "id": "abstract",
        "op": "loop",
        "T": 28,
        "state": [
            {"id": name, "init": abstract_initial}
            for name in abstract_names
        ],
        "body": abstract_body,
        "next": abstract_next,
    })

    station_values = [
        sum_vector(vm("abstract." + name, hud), 96)
        for name in abstract_names
    ]
    selected_goal_value = affine(goal_match, 1000, -1000)

    # Counter occupies rows 61--62 and columns 13--54.
    band_delta = node("abs", [linear(
        "coords", [[63, 0, 0], [0, 63, 0]], [-61.5, -33.5]
    )])
    bar_rows = node("min2", [
        relu(linear(sl(band_delta, 0, 1), [[-1000]], [900])),
        const(1),
    ])
    bar_cols = node("min2", [
        relu(linear(sl(band_delta, 1, 2), [[-1000]], [20900])),
        const(1),
    ])
    bar = mul(mul(bar_rows, bar_cols), sl("x0", 11, 12))
    ticks = scale(node("gmean", [bar]), 2048)
    difference = vsub(vs("mem", 226, 227), ticks)

    def near_scalar(value, target):
        delta = affine(value, 1, -target)
        absolute = va(vr(delta), vr(scale(delta, -1)))
        return binary_scalar(complement(absolute))

    is_one = near_scalar(difference, 1)
    is_two = near_scalar(difference, 2)
    one_tick_flag = va(
        is_one,
        vm(vs("mem", 227, 228), complement(va(is_one, is_two))),
    )
    decrement = affine(one_tick_flag, -1, 2)
    refill_value_here = value_at(fields[5], player_vector)
    refill_distance = affine(refill_value_here, -1, 128)
    urgency = cap(vr(va(
        vsub(
            vm(affine(refill_distance, 1, 3), decrement), ticks
        ),
        const(1),
    )))
    urgency = vm(urgency, binary_scalar(refill_value_here))
    refill_bonus = affine(urgency, 900, -500)

    terminal_values = station_values + [
        selected_goal_value, const(-100), refill_bonus
    ]

    combined = None
    for i, field in enumerate(fields):
        missing = vr(vsub(const(1), field))
        score = va(
            va(field, const(-128)), scale(missing, -1000)
        )
        score = va(score, terminal_values[i])
        if i < 3:
            others = vr(vsub(station_all_vector, station_vectors[i]))
            score = va(score, scale(others, -10000))
        elif i < 5:
            score = va(score, scale(station_all_vector, -10000))
        combined = score if combined is None else vmax(combined, score)

    wrong_goals = vr(vsub(
        all_goals_vector, vm(goal_vector, goal_match)
    ))
    combined = va(combined, scale(wrong_goals, -10000))
    combined = va(
        combined, scale(vsub(const(1), open_vector), -10000)
    )

    # Missing neighbours must be very unattractive, not zero-valued.
    boundary_neighbours = [
        shift(const(np.ones(100, np.float32)), d)
        for d in range(4)
    ]
    action_values = []
    for d in range(4):
        candidate = va(
            shift(combined, d),
            scale(vsub(const(1), boundary_neighbours[d]), -20000),
        )
        action_values.append(value_at(candidate, player_vector))
    action_scores = vcat(*action_values)

    inertia_w = np.zeros((4, 8), np.float32)
    for d in range(4):
        inertia_w[d, d + 1] = 0.03
    action_scores = va(action_scores, dense("aprev", inertia_w))

    action_w = np.zeros((8, 4), np.float32)
    action_b = np.full(8, -1000000, np.float32)
    for d in range(4):
        action_w[d + 1, d] = 1
        action_b[d + 1] = 0

    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [action_scores],
        "w": parameter(action_w),
        "b": parameter(action_b),
    })
    graph.append({
        "id": "click",
        "op": "conv",
        "in": ["coords"],
        "w": parameter(np.zeros((1, 3, 1, 1), np.float32)),
    })
    memory_value = vcat(
        terrain_memory,
        *station_records,
        goal_record,
        refill_memory,
        ticks,
        one_tick_flag,
    )
    graph.append({
        "id": "mem_next",
        "op": "vslice",
        "in": [memory_value],
        "start": 0,
        "stop": 228,
    })

    # Account for graph evaluations without evaluating any game state.
    nonwave_evaluations = 0
    for nd in graph:
        if nd["id"] == "waves":
            continue
        if nd["op"] == "loop":
            nonwave_evaluations += 1 + nd["T"] * len(nd["body"])
        else:
            nonwave_evaluations += 1
    wave_iterations = min(
        26, (9990 - nonwave_evaluations - 1) // len(wave_body)
    )
    for nd in graph:
        if nd["id"] == "waves":
            nd["T"] = max(1, int(wave_iterations))

    return {
        "graph": graph,
        "memory_size": 228,
        "params": params,
    }
