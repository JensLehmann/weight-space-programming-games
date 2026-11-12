import numpy as np


def build():
    N = 12
    graph = []
    params = {}
    active = [graph]
    serial = [0]
    parameter_cache = {}
    constant_cache = {}

    def fresh():
        name = "n%d" % serial[0]
        serial[0] += 1
        return name

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a.copy()
        return parameter_cache[key]

    def node(op, inputs=None, name=None, **fields):
        name = fresh() if name is None else name
        nd = {"id": name, "op": op}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        active[0].append(nd)
        return name

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], name=name, **fields)

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    one = dense("avail", np.zeros((1, 8)), np.ones(1))
    constant_cache[((1,), np.ones(1, np.float32).tobytes())] = one

    def constant(value):
        a = np.asarray(value, np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key not in constant_cache:
            old = active[0]
            active[0] = graph
            constant_cache[key] = dense(one, a[:, None])
            active[0] = old
        return constant_cache[key]

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def scale(x, a):
        return x if a == 1 else mul(x, constant(a))

    def affine(x, a=1, b=0):
        y = scale(x, a)
        return y if b == 0 else add(y, constant(b))

    def clip(x):
        return node("min2", [node("relu", [x]), one])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def vs(x, a):
        return x if a == 1 else vm(x, constant(a))

    def vf(x, a=1, b=0):
        y = vs(x, a)
        return y if b == 0 else va(y, constant(b))

    def vsub(a, b):
        return va(a, vs(b, -1))

    def vr(x):
        return node("vrelu", [x])

    def vclip(x):
        p = vr(x)
        return vsub(p, vr(vf(p, 1, -1)))

    def vabs(x):
        return va(vr(x), vr(vs(x, -1)))

    def veq_zero(x):
        return vclip(vf(vabs(x), -4, 2))

    def vpositive(x):
        return vclip(vf(x, 2, -1))

    def nonnegative(x):
        # Integer-valued quantities with tolerance for float32 roundoff.
        return vclip(vf(x, 2, 2))

    def vmax(a, b):
        return va(b, vr(vsub(a, b)))

    def vsum(x, length=N):
        return dense(x, np.ones((1, length)))

    def vdot(a, b, length=N):
        return vsum(vm(a, b), length)

    def pool(x):
        return node("gmax", [x])

    def spatial_sum(x, channels):
        return conv(x, np.ones((1, channels, 1, 1)))

    # Restore the image row covered by the step-counter interface.
    row = sl("coords", 0, 1)
    below_w = np.zeros((16, 16, 3, 3), np.float32)
    for c in range(16):
        below_w[c, c, 2, 1] = 1
    below = conv("x0", below_w)
    not_top = clip(scale(row, 63))
    clean = add(
        mul("x0", not_top),
        mul(below, affine(not_top, -1, 1)),
    )

    colours = {}

    def colour(c):
        if c not in colours:
            colours[c] = sl(clean, c, c + 1)
        return colours[c]

    yellow = colour(4)
    blue = colour(11)
    pink = colour(14)
    bases = add(blue, pink)
    buttons = colour(9)
    gates = add(colour(1), colour(12))
    raw_wall = add(colour(5), gates)

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def directional_samples(x, sign):
        w = np.zeros((4, 1, 7, 7), np.float32)
        for d, (dy, dx) in enumerate(directions):
            for t in (1, 2, 3):
                w[d, 0, 3 + sign * dy * t,
                  3 + sign * dx * t] = 1
        return conv(x, w)

    gravity = vclip(pool(mul(directional_samples(bases, 1), yellow)))
    horizontal = dense(gravity, [[0, 0, 1, 1]])
    vertical = dense(gravity, [[1, 1, 0, 0]])

    qw = np.array([
        [63, 0, 0],
        [-63, 0, 63],
        [0, 63, 0],
        [0, -63, 63],
    ], np.float32).reshape(4, 3, 1, 1)
    q = spatial_sum(mul(conv("coords", qw), gravity), 4)
    q1 = affine(q, 1, 1)
    reverse_q = affine(q, -1, 128)

    def axis_planes(x, channels, radius, offsets=None):
        if offsets is None:
            offsets = list(range(-radius, radius + 1))
        k = 2 * radius + 1
        w = np.zeros((2 * channels, channels, k, k), np.float32)
        for c in range(channels):
            for d in offsets:
                w[c, c, radius, radius + d] = 1
                w[channels + c, c, radius + d, radius] = 1
        z = conv(x, w)
        return sl(z, 0, channels), sl(z, channels, 2 * channels)

    def perpendicular_sum(x, channels, radius, offsets=None):
        hx, vy = axis_planes(x, channels, radius, offsets)
        return add(mul(hx, vertical), mul(vy, horizontal))

    def perpendicular_dilate(x, channels, radius=3):
        return clip(perpendicular_sum(x, channels, radius))

    def gravity_max(x):
        hx = node("rowmax", [x])
        vy = node("colmax", [x])
        return add(mul(hx, horizontal), mul(vy, vertical))

    # Extend longitudinal separators, but not transverse floors.
    hx, vy = axis_planes(raw_wall, 1, 3)
    hx = node("rowmax", [node("relu", [affine(hx, 1, -6)])])
    vy = node("colmax", [node("relu", [affine(vy, 1, -6)])])
    walls = clip(add(mul(hx, horizontal), mul(vy, vertical)))
    barriers = node("max2", [walls, colour(5)])
    free = affine(barriers, -1, 1)

    # Label chambers by masked max propagation.
    label_w = np.array([4032, 63, 1], np.float32).reshape(1, 3, 1, 1)
    initial_labels = mul(conv("coords", label_w), free)
    label_ceiling = scale(free, 8192)

    shift_w = np.zeros((8, 1, 5, 5), np.float32)
    shifts = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
    ]
    for i, (dy, dx) in enumerate(shifts):
        shift_w[i, 0, 2 + dy, 2 + dx] = 1

    label_state = fresh()
    label_body = []
    old = active[0]
    active[0] = label_body
    shifted = conv(label_state, shift_w)
    maximum = label_state
    for i in range(8):
        maximum = node("max2", [maximum, sl(shifted, i, i + 1)])
    label_next = node("min2", [maximum, label_ceiling])
    active[0] = old
    label_loop = node(
        "loop", T=64,
        state=[{"id": label_state, "init": initial_labels}],
        body=label_body,
        next={label_state: label_next},
    )
    labels = label_loop + "." + label_state

    remaining = labels
    room_list = []
    for _ in range(N):
        largest = pool(remaining)
        difference = node("abs", [sub(labels, largest)])
        room = mul(node("relu", [affine(difference, -1, 1)]), free)
        room_list.append(room)
        remaining = node("relu", [
            sub(remaining, scale(room, 8192))
        ])
    rooms = node("concat", room_list)

    def room_pool(field, masks=rooms):
        return pool(mul(masks, field))

    room_min_q = vf(room_pool(reverse_q), -1, 128)
    room_max_q = room_pool(q)
    bottom = vf(room_max_q, 1, 1)

    # Remove narrow decorative bracket pixels from the water mask.
    water = node("relu", [
        affine(perpendicular_sum(colour(0), 1, 1), 1, -2)
    ])
    water_present = room_pool(water)
    observed_surface = vf(room_pool(mul(water, reverse_q)), -1, 128)
    surface = va(
        vm(observed_surface, water_present),
        vm(bottom, vf(water_present, -1, 1)),
    )

    # Water transfers conserve depth, not rendered pixel area.
    water_volume = vr(vsub(bottom, surface))
    total_water = vsum(water_volume)

    behind_yellow = spatial_sum(
        mul(directional_samples(yellow, -1), gravity), 4
    )
    behind_yellow = clip(behind_yellow)
    char_blue = room_pool(mul(blue, behind_yellow))
    char_pink = room_pool(mul(pink, behind_yellow))
    char_any = room_pool(yellow)

    raw_wall_q = mul(mul(raw_wall, walls), reverse_q)
    wall_start = mul(affine(gravity_max(raw_wall_q), -1, 128), walls)
    adjacent_wall_start = perpendicular_sum(
        wall_start, 1, 1, offsets=[-1, 1]
    )
    wall_capacity = room_pool(adjacent_wall_start)

    yellow_min_encoding = room_pool(mul(yellow, reverse_q))
    character_depth = va(vf(surface, 1, -128), yellow_min_encoding)
    bounded_above = vpositive(room_min_q)
    floor_capacity = va(
        room_min_q,
        vm(vm(bounded_above, char_any), character_depth),
    )
    capacity = vmax(wall_capacity, floor_capacity)

    halos = perpendicular_dilate(rooms, N, 3)
    own_buttons = mul(rooms, buttons)
    button_halos = mul(perpendicular_dilate(own_buttons, N, 3), walls)

    target_blue_map = mul(blue, walls)
    target_pink_map = mul(pink, walls)
    target_blue = room_pool(target_blue_map, halos)
    target_pink = room_pool(target_pink_map, halos)
    height_blue = room_pool(mul(target_blue_map, q1), halos)
    height_pink = room_pool(mul(target_pink_map, q1), halos)

    # A marker may border two chambers with different floor offsets.
    # Retain only sides whose required water depth is achievable.
    blue_required = vr(vsub(bottom, height_blue))
    pink_required = vr(vsub(bottom, height_pink))
    target_blue = vm(
        target_blue, nonnegative(vsub(total_water, blue_required))
    )
    target_pink = vm(
        target_pink, nonnegative(vsub(total_water, pink_required))
    )

    # Directed button graph and gate graph.
    pump_rows = []
    gate_height_rows = []
    for i in range(N):
        hi = sl(halos, i, i + 1)
        bi = sl(button_halos, i, i + 1)
        off_diagonal = np.ones(N, np.float32)
        off_diagonal[i] = 0
        off_diagonal = constant(off_diagonal)

        pump_rows.append(vm(pool(mul(halos, bi)), off_diagonal))
        shared_gate = mul(mul(halos, hi), gates)
        gh = pool(mul(shared_gate, q1))
        gate_height_rows.append(vm(gh, off_diagonal))

    pump_matrix = node("vconcat", pump_rows)
    gate_heights = node("vconcat", gate_height_rows)

    expand_row_w = np.zeros((N * N, N), np.float32)
    expand_col_w = np.zeros((N * N, N), np.float32)
    reduce_row_w = np.zeros((N, N * N), np.float32)
    reduce_col_w = np.zeros((N, N * N), np.float32)
    for i in range(N):
        for j in range(N):
            k = i * N + j
            expand_row_w[k, i] = 1
            expand_col_w[k, j] = 1
            reduce_row_w[i, k] = 1
            reduce_col_w[j, k] = 1

    def expand_row(x):
        return dense(x, expand_row_w)

    def expand_col(x):
        return dense(x, expand_col_w)

    def reduce_row(x):
        return dense(x, reduce_row_w)

    def reduce_col(x):
        return dense(x, reduce_col_w)

    def matrix_vector(matrix, x):
        return reduce_row(vm(matrix, expand_col(x)))

    # Gate alignment requires filling both bordering chambers.
    gate_required = va(
        vr(vsub(expand_row(bottom), gate_heights)),
        vr(vsub(expand_col(bottom), gate_heights)),
    )
    gate_matrix = vm(
        vclip(gate_heights),
        nonnegative(vsub(total_water, gate_required)),
    )

    difference_w = expand_col_w - expand_row_w
    tie_bias = (
        np.arange(N)[:, None] - np.arange(N)[None, :]
    ).astype(np.float32).reshape(-1)

    def winner(scores):
        differences = vr(dense(scores, difference_w, tie_bias))
        return vr(dense(differences, -reduce_row_w, np.ones(N)))

    def distances(matrix, seeds, upstream=True):
        initial_distance = vf(seeds, -1, 1)
        reach_state = fresh()
        distance_state = fresh()
        body = []
        previous = active[0]
        active[0] = body

        if upstream:
            neighbours = reduce_col(
                vm(matrix, expand_row(reach_state))
            )
        else:
            neighbours = reduce_row(
                vm(matrix, expand_col(reach_state))
            )
        counts = va(neighbours, reach_state)
        missing = vr(vf(counts, -1, 1))
        new_reach = vf(missing, -1, 1)
        new_distance = va(distance_state, missing)

        active[0] = previous
        loop = node(
            "loop", T=N,
            state=[
                {"id": reach_state, "init": seeds},
                {"id": distance_state, "init": initial_distance},
            ],
            body=body,
            next={
                reach_state: new_reach,
                distance_state: new_distance,
            },
        )
        return loop + "." + distance_state

    # Retain an unfinished goal colour across actions.
    solved_blue = vm(
        target_blue, veq_zero(vsub(surface, height_blue))
    )
    solved_pink = vm(
        target_pink, veq_zero(vsub(surface, height_pink))
    )
    unfinished_blue = vm(char_blue, vf(solved_blue, -1, 1))
    unfinished_pink = vm(char_pink, vf(solved_pink, -1, 1))
    any_blue = vclip(vsum(unfinished_blue))
    any_pink = vclip(vsum(unfinished_pink))

    hold_blue = vm(vsl("mem", 0, 1), any_blue)
    hold_pink = vm(vsl("mem", 1, 2), any_pink)
    released = vf(va(hold_blue, hold_pink), -1, 1)
    choose_blue = va(hold_blue, vm(released, any_blue))
    choose_pink = va(
        hold_pink,
        vm(vm(released, vf(any_blue, -1, 1)), any_pink),
    )
    node("vconcat", [choose_blue, choose_pink], name="mem_next")

    candidates = va(
        vm(unfinished_blue, choose_blue),
        vm(unfinished_pink, choose_pink),
    )
    character = vm(winner(vs(candidates, 1024)), candidates)
    targets = va(
        vm(target_blue, choose_blue), vm(target_pink, choose_pink)
    )
    target_heights = va(
        vm(height_blue, choose_blue), vm(height_pink, choose_pink)
    )

    gate_distance = distances(gate_matrix, targets)
    current_distance = vdot(gate_distance, character)
    neighbouring_rooms = matrix_vector(gate_matrix, character)
    closer = vr(vsub(current_distance, gate_distance))
    next_candidates = vm(neighbouring_rooms, closer)
    next_room = vm(winner(vs(next_candidates, 1024)), next_candidates)

    already_adjacent = vdot(character, targets)
    use_gate = vf(already_adjacent, -1, 1)
    next_room_for_gate = vm(next_room, use_gate)

    selected_gate_pair = vm(
        expand_row(character), expand_col(next_room)
    )
    gate_height = vdot(gate_heights, selected_gate_pair, N * N)
    final_height = vdot(target_heights, character)
    checkpoint_height = va(
        vm(already_adjacent, final_height),
        vm(use_gate, gate_height),
    )
    checkpoint_rooms = va(character, next_room_for_gate)

    extra_blue = vm(char_blue, target_blue)
    extra_pink = vm(
        vm(char_pink, target_pink), vf(extra_blue, -1, 1)
    )
    extra_rooms = va(extra_blue, extra_pink)
    extra_heights = va(
        vm(extra_blue, height_blue), vm(extra_pink, height_pink)
    )
    outside_checkpoint = vf(checkpoint_rooms, -1, 1)
    constrained = va(
        checkpoint_rooms, vm(extra_rooms, outside_checkpoint)
    )
    desired = va(
        vm(checkpoint_rooms, checkpoint_height),
        vm(extra_heights, outside_checkpoint),
    )

    error = vsub(surface, desired)
    need_raise = vm(constrained, vpositive(error))
    need_lower = vm(constrained, vpositive(vs(error, -1)))
    unconstrained = vf(constrained, -1, 1)
    has_water = vpositive(vsub(bottom, surface))
    has_capacity = vpositive(vsub(surface, capacity))
    raising = vclip(vsum(need_raise))

    sources_when_raising = vm(va(unconstrained, need_lower), has_water)
    sources_when_lowering = vm(need_lower, has_water)
    sources = va(
        vm(sources_when_raising, raising),
        vm(sources_when_lowering, vf(raising, -1, 1)),
    )
    sinks = va(
        vm(need_raise, raising),
        vm(vm(unconstrained, has_capacity), vf(raising, -1, 1)),
    )

    # Route liquid along a shortest source-to-demand path.
    sink_distance = distances(pump_matrix, sinks, upstream=True)
    reachable = vclip(vf(sink_distance, -2, 2 * N + 1))
    sources = vm(sources, reachable)
    source_scores = va(vs(sources, 1024), vs(sink_distance, -16))
    source = vm(winner(source_scores), sources)
    total_distance = vdot(sink_distance, source)
    source_distance = distances(pump_matrix, source, upstream=False)

    on_path = veq_zero(
        vsub(va(source_distance, sink_distance), total_distance)
    )
    downhill = vr(dense(sink_distance, difference_w))
    eligible = vm(pump_matrix, downhill)
    eligible = vm(eligible, expand_col(vm(on_path, has_water)))
    eligible = vm(eligible, expand_row(has_capacity))

    donor_candidates = vclip(reduce_col(eligible))
    donor_scores = va(
        vs(donor_candidates, 1024), vs(sink_distance, -16)
    )
    donor = vm(winner(donor_scores), donor_candidates)
    receiver_candidates = vclip(
        reduce_row(vm(eligible, expand_col(donor)))
    )
    receiver = vm(
        winner(vs(receiver_candidates, 1024)), receiver_candidates
    )

    donor_map = spatial_sum(mul(rooms, donor), N)
    receiver_map = spatial_sum(mul(rooms, receiver), N)
    donor_near = perpendicular_dilate(donor_map, 1, 3)
    donor_near = perpendicular_dilate(donor_near, 1, 3)
    pump_click = mul(mul(buttons, receiver_map), donor_near)

    aligned = veq_zero(vsub(surface, checkpoint_height))
    current_aligned = vdot(aligned, character)
    next_aligned = vdot(aligned, next_room)
    gate_ready = vm(vm(current_aligned, next_aligned), use_gate)

    current_halo = spatial_sum(mul(halos, character), N)
    next_halo = spatial_sum(mul(halos, next_room), N)
    gate_click = mul(mul(gates, current_halo), next_halo)
    gate_click = mul(gate_click, gate_ready)
    gate_click = mul(gate_click, affine(q, 1, 200))

    click_scores = add(
        add(scale(pump_click, 100), gate_click),
        scale(buttons, 0.1),
    )
    node("add", [click_scores, constant(0)], name="click")

    action_weights = np.zeros((8, 1), np.float32)
    action_weights[6, 0] = 10
    dense(one, action_weights, name="logits")

    return {
        "graph": graph,
        "memory_size": 2,
        "params": params,
    }
