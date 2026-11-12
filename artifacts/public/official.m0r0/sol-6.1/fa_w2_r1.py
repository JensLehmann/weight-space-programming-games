import numpy as np


def build():
    M = 33
    graph = []
    params = {}
    shape = {
        "x0": (16, True),
        "coords": (3, True),
        "mem": (M, False),
        "aprev": (8, False),
        "avail": (8, False),
    }
    serial = [0]
    pcache = {}
    ncache = {}

    def par(value):
        value = np.asarray(value, np.float32)
        key = (value.shape, value.tobytes())
        if key not in pcache:
            name = "p%d" % len(pcache)
            pcache[key] = name
            params[name] = value
        return pcache[key]

    def nd(op, inputs, **fields):
        key = (op, tuple(inputs), tuple(sorted(fields.items())))
        if key in ncache:
            return ncache[key]
        name = "n%d" % serial[0]
        serial[0] += 1
        item = {"id": name, "op": op, "in": list(inputs)}
        item.update(fields)
        graph.append(item)
        ncache[key] = name

        c, spatial = shape[inputs[0]]
        if op == "conv":
            c, spatial = params[fields["w"]].shape[0], True
        elif op == "dense":
            c, spatial = params[fields["w"]].shape[0], False
        elif op in ("concat", "vconcat"):
            c = sum(shape[x][0] for x in inputs)
        elif op in ("slice", "vslice"):
            c = fields["stop"] - fields["start"]
        elif op in ("gmax", "gmean"):
            spatial = False
        elif op in ("add", "sub", "mul", "max2", "min2",
                    "vadd", "vmul"):
            c = max(shape[x][0] for x in inputs)
            spatial = any(shape[x][1] for x in inputs)
        shape[name] = (c, spatial)
        return name

    def cat(xs):
        if len(xs) == 1:
            return xs[0]
        return nd("concat" if shape[xs[0]][1] else "vconcat", xs)

    def sl(x, a, b):
        if a == 0 and b == shape[x][0]:
            return x
        return nd("slice" if shape[x][1] else "vslice",
                  [x], start=a, stop=b)

    def aff(x, weight, bias=None):
        weight = np.asarray(weight, np.float32)
        spatial = shape[x][1]
        fields = {
            "w": par(weight[:, :, None, None] if spatial else weight)
        }
        if bias is not None:
            fields["b"] = par(np.broadcast_to(
                np.asarray(bias, np.float32), (weight.shape[0],)
            ).copy())
        return nd("conv" if spatial else "dense", [x], **fields)

    def scale(x, a=1.0, b=None):
        return aff(x, np.eye(shape[x][0], dtype=np.float32) * a, b)

    def lin(xs, coefficients, bias=None):
        c = shape[xs[0]][0]
        blocks = [
            np.eye(c, dtype=np.float32) * a
            for a in coefficients
        ]
        return aff(cat(xs), np.concatenate(blocks, axis=1), bias)

    def const(value, count=1):
        return aff("avail", np.zeros((count, 8), np.float32), value)

    def add(x, y):
        spatial = shape[x][1] or shape[y][1]
        return nd("add" if spatial else "vadd", [x, y])

    def sub(x, y):
        if shape[x][1] or shape[y][1]:
            return nd("sub", [x, y])
        return add(x, scale(y, -1.0))

    def mul(x, y):
        spatial = shape[x][1] or shape[y][1]
        return nd("mul" if spatial else "vmul", [x, y])

    def relu(x):
        return nd("relu" if shape[x][1] else "vrelu", [x])

    def absolute(x):
        if shape[x][1]:
            return nd("abs", [x])
        return add(relu(x), relu(scale(x, -1.0)))

    def clip(x):
        p = relu(x)
        return sub(p, relu(scale(p, 1.0, -1.0)))

    def inv(x):
        return scale(x, -1.0, 1.0)

    def eq_integer(x, value=0.0):
        d = scale(x, 1.0, -value)
        return relu(scale(mul(d, d), -1.0, 1.0))

    def gmax(x):
        return nd("gmax", [x])

    def total(x):
        return scale(nd("gmean", [x]), 4096.0)

    def sumc(x):
        return aff(x, np.ones((1, shape[x][0]), np.float32))

    def smax(xs):
        out = xs[0]
        for x in xs[1:]:
            out = nd("max2", [out, x])
        return out

    def vmax(x, y):
        return add(y, relu(sub(x, y)))

    def shift(x, dy, dx):
        r = max(abs(dy), abs(dx))
        k = 2 * r + 1
        c = shape[x][0]
        w = np.zeros((c, c, k, k), np.float32)
        for i in range(c):
            w[i, i, r + dy, r + dx] = 1.0
        return nd("conv", [x], w=par(w))

    def box(x, k):
        c = shape[x][0]
        w = np.zeros((c, c, k, k), np.float32)
        for i in range(c):
            w[i, i] = 1.0
        return nd("conv", [x], w=par(w))

    def corners(x):
        return relu(lin(
            [x, shift(x, -1, 0), shift(x, 0, -1)],
            [1.0, -1.0, -1.0]
        ))

    one = const(1.0)
    pixels = aff("coords", [[63, 0, 0], [0, 63, 0]])
    row = sl(pixels, 0, 1)
    col = sl(pixels, 1, 2)
    spatial_one = sl("coords", 2, 3)
    lex = aff(pixels, [[64, 1]])
    small_lex = scale(lex, -1.0 / 8192.0)

    def point_at(position):
        d = sumc(absolute(sub(pixels, position)))
        return relu(scale(d, -1.0, 1.0))

    def position_of(point):
        return total(mul(point, pixels))

    def choose(points, score):
        masked = sub(score, scale(inv(points), 10000.0))
        selected = relu(scale(sub(masked, gmax(masked)),
                              32768.0, 1.0))
        return mul(selected, points)

    def peaks(score):
        return mul(
            relu(scale(sub(score, gmax(score)), 32768.0, 1.0)),
            clip(score)
        )

    # Token paint and logical rendering scale.
    paint_w = np.zeros((1, 16), np.float32)
    paint_w[0, 1] = paint_w[0, 10] = 1.0
    paint = aff("x0", paint_w)
    area = total(paint)
    top_edges = total(relu(sub(paint, shift(paint, -1, 0))))
    left_edges = total(relu(sub(paint, shift(paint, 0, -1))))

    scale_flags = []
    statistics = cat([area, top_edges, left_edges])
    for s in (3, 4, 5, 6):
        errors = aff(statistics, [
            [1, -s, 0],
            [1, 0, -s],
        ])
        squared = mul(errors, errors)
        product = mul(sl(squared, 0, 1), sl(squared, 1, 2))
        scale_flags.append(relu(scale(product, -1.0, 1.0)))

    found_scale = clip(aff(cat(scale_flags), [[1, 1, 1, 1]]))
    scale_flags[1] = add(scale_flags[1], inv(found_scale))
    render_scale = aff(cat(scale_flags), [[3, 4, 5, 6]])

    def centers(x, interior=False):
        offsets = (0, 1, 1, 2) if interior else (1, 2, 2, 3)
        terms = [
            mul(shift(x, -offset, -offset), flag)
            for offset, flag in zip(offsets, scale_flags)
        ]
        out = terms[0]
        for term in terms[1:]:
            out = add(out, term)
        return out

    def neighbor(x, dy, dx):
        base = shift(x, 3 * dy, 3 * dx)
        terms = [mul(base, scale_flags[0])]
        for extra, flag in zip((1, 2, 3), scale_flags[1:]):
            terms.append(mul(shift(base, extra * dy, extra * dx), flag))
        out = terms[0]
        for term in terms[1:]:
            out = add(out, term)
        return out

    raw_points = centers(corners(paint))
    adjacent = add(neighbor(raw_points, 0, -1),
                   neighbor(raw_points, -1, 0))
    raw_points = clip(add(raw_points, mul(adjacent, paint)))
    visible_count = total(raw_points)

    # Two live tracks, two frozen rendezvous locations, and controller state.
    old_positions = [sl("mem", 0, 2), sl("mem", 2, 4)]
    old_signs = [sl("mem", 4, 6), sl("mem", 6, 8)]
    initialized = sl("mem", 8, 9)
    old_step = sl("mem", 9, 10)
    old_phase = sl("mem", 10, 11)
    frozen_positions = [sl("mem", 11, 13), sl("mem", 14, 16)]
    frozen_flags = [sl("mem", 13, 14), sl("mem", 16, 17)]
    old_block_start = sl("mem", 17, 19)
    old_block_mode = sl("mem", 19, 20)
    old_count = sl("mem", 20, 21)
    old_repick = sl("mem", 21, 22)
    old_visited = sl("mem", 22, 25)
    old_gate_exists = sl("mem", 25, 28)
    old_hold_timer = sl("mem", 28, 29)
    old_hold_position = sl("mem", 29, 31)
    old_switch_goal_position = sl("mem", 31, 33)

    fresh = inv(initialized)
    repick = clip(add(fresh, old_repick))
    continuing = inv(repick)
    lost_component = mul(clip(sub(old_count, visible_count)), initialized)

    pool = raw_points
    for position, flag in zip(frozen_positions, frozen_flags):
        pool = mul(pool, inv(mul(point_at(position), flag)))

    points = []
    positions = []
    for i in range(2):
        distance = sumc(absolute(sub(pixels, old_positions[i])))
        tracking_score = add(scale(distance, -1.0), small_lex)
        initial_score = scale(lex, -1.0 if i == 0 else 1.0)
        score = add(mul(tracking_score, continuing),
                    mul(initial_score, repick))
        point = choose(pool, score)
        points.append(point)
        positions.append(position_of(point))
        pool = mul(pool, inv(mul(point, inv(lost_component))))

    # Successful displacements identify each token's reflected controls.
    controls = aff("aprev", [
        [0, -1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 1, 0, 0, 0],
    ])
    control_present = absolute(controls)
    signs = []
    for i in range(2):
        displacement = sub(positions[i], old_positions[i])
        signed_displacement = nd(
            "vtanh", [scale(displacement, 10.0)]
        )
        empirical = mul(signed_displacement, controls)
        confidence = mul(
            mul(clip(absolute(displacement)), control_present),
            continuing
        )
        guess = aff(
            "avail", np.zeros((2, 8), np.float32),
            [1.0, 1.0 if i == 0 else -1.0]
        )
        prior = add(mul(old_signs[i], continuing), mul(guess, repick))
        signs.append(add(
            mul(empirical, confidence), mul(prior, inv(confidence))
        ))

    merged = mul(
        mul(gmax(mul(points[0], points[1])), lost_component),
        continuing
    )
    sources = [mul(point, inv(merged)) for point in points]

    first_slot = inv(frozen_flags[0])
    assignments = [
        mul(merged, first_slot),
        mul(merged, frozen_flags[0])
    ]
    new_frozen_flags = []
    new_frozen_positions = []
    for position, flag, assignment in zip(
            frozen_positions, frozen_flags, assignments):
        new_frozen_positions.append(add(
            mul(position, inv(assignment)),
            mul(positions[0], assignment)
        ))
        new_frozen_flags.append(clip(add(flag, assignment)))

    # Distinguish coloured sprites from frame-connected wall-colour regions.
    colour_w = np.zeros((5, 16), np.float32)
    for i, colour in enumerate((9, 11, 12, 14, 15)):
        colour_w[i, colour] = 1.0
    colour_planes = aff("x0", colour_w)
    border = clip(add(
        add(eq_integer(row, 1.0), eq_integer(col, 0.0)),
        eq_integer(col, 63.0)
    ))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def loop_begin(channels):
        sid = "s%d" % serial[0]
        serial[0] += 1
        lid = "L%d" % serial[0]
        serial[0] += 1
        shape[sid] = (channels, True)
        saved_graph = graph[:]
        saved_cache = dict(ncache)
        graph[:] = []
        return sid, lid, saved_graph, saved_cache

    def loop_end(sid, lid, seed, nxt, iterations, outer, cache):
        body = graph[:]
        graph[:] = outer
        ncache.clear()
        ncache.update(cache)
        graph.append({
            "id": lid, "op": "loop", "T": iterations,
            "state": [{"id": sid, "init": seed}],
            "body": body,
            "next": {sid: nxt},
        })
        result = lid + "." + sid
        shape[result] = shape[sid]
        return result

    def boolean_flood(seed, mask):
        c = shape[seed][0]
        sid, lid, outer, cache = loop_begin(c)
        w = np.zeros((c, c, 3, 3), np.float32)
        for i in range(c):
            for y, x in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)):
                w[i, i, y, x] = 1.0
        spread = nd("conv", [sid], w=par(w))
        nxt = mul(clip(spread), mask)
        return loop_end(sid, lid, seed, nxt, 96, outer, cache)

    def distance_flood(seed, passability):
        c = shape[seed][0]
        sid, lid, outer, cache = loop_begin(c)
        w = np.zeros((4 * c, c, 3, 3), np.float32)
        for d, (dy, dx) in enumerate(directions):
            for i in range(c):
                w[d * c + i, i, 1 + dy, 1 + dx] = 1.0
        shifted = nd("conv", [sid], w=par(w))
        pieces = [sl(shifted, d * c, (d + 1) * c) for d in range(4)]
        propagated = relu(scale(smax(pieces), 1.0, -1.0))
        nxt = mul(nd("max2", [propagated, seed]), passability)
        return loop_end(sid, lid, seed, nxt, 224, outer, cache)

    walls = boolean_flood(mul(colour_planes, border), colour_planes)
    isolated = relu(sub(colour_planes, walls))
    blocker9 = sl(isolated, 0, 1)
    selected11 = sl(isolated, 1, 2)
    gate_colours = sl(isolated, 2, 5)

    block_mode = gmax(selected11)
    token_mode = inv(block_mode)
    selected_point = choose(
        centers(corners(selected11), True), scale(lex, -1.0)
    )
    selected_position = position_of(selected_point)
    just_selected = mul(block_mode, inv(old_block_mode))
    block_start = add(
        mul(selected_position, just_selected),
        mul(old_block_start, inv(just_selected))
    )

    gate_corners = corners(gate_colours)
    extending = clip(add(
        neighbor(gate_colours, 0, 1),
        neighbor(gate_colours, 1, 0)
    ))
    switch_corners = mul(gate_corners, inv(extending))
    switch_points = centers(switch_corners)
    switch_fill = mul(
        gate_colours, clip(box(shift(switch_corners, -3, -3), 7))
    )
    gates = relu(sub(gate_colours, switch_fill))
    gate_exists = gmax(gates)
    newly_opened_colours = relu(sub(old_gate_exists, gate_exists))
    visited = clip(add(old_visited, newly_opened_colours))
    newly_opened = clip(aff(newly_opened_colours, [[1, 1, 1]]))

    # Remember a recently occupied switch even though the token hides its paint.
    hold_position = add(
        mul(old_switch_goal_position, newly_opened),
        mul(old_hold_position, inv(newly_opened))
    )
    hold_timer = add(
        scale(newly_opened, 6.0),
        mul(relu(scale(old_hold_timer, 1.0, -1.0)), inv(newly_opened))
    )
    hold_active = mul(clip(hold_timer), token_mode)
    hold_point = mul(point_at(hold_position), hold_active)

    blocker_full = clip(box(add(blocker9, selected11), 3))
    selected_full = clip(box(selected11, 3))
    floor5 = sl("x0", 5, 6)
    normal_floor = clip(sub(
        clip(add(add(floor5, paint), clip(sumc(switch_fill)))),
        blocker_full
    ))
    normal_pass = relu(scale(box(normal_floor, 3), 1.0, -8.0))
    block_floor = clip(sub(
        clip(add(floor5, selected_full)),
        relu(sub(blocker_full, selected_full))
    ))
    block_pass = relu(scale(box(block_floor, 3), 1.0, -8.0))

    distance_fields = distance_flood(
        scale(cat(sources + [selected_point]), 512.0),
        cat([normal_pass, normal_pass, block_pass])
    )
    field_a = sl(distance_fields, 0, 1)
    field_b = sl(distance_fields, 1, 2)
    block_field = sl(distance_fields, 2, 3)
    closest_field = nd("max2", [field_a, field_b])

    common = nd("min2", [field_a, field_b])
    can_meet = clip(gmax(common))
    meeting_goal = peaks(common)

    useful_switches = mul(switch_points, gate_exists)
    switch_priority = add(closest_field, scale(inv(visited), 128.0))
    switch_score = sumc(mul(
        mul(useful_switches, switch_priority), clip(closest_field)
    ))
    has_switch = clip(gmax(switch_score))
    switch_goal = choose(peaks(switch_score), small_lex)
    switch_goal_position = position_of(switch_goal)

    switch_value_a = gmax(mul(field_a, switch_goal))
    switch_value_b = gmax(mul(field_b, switch_goal))
    switch_best = vmax(switch_value_a, switch_value_b)
    switch_weight_a = eq_integer(sub(switch_value_a, switch_best))
    switch_weight_b = inv(switch_weight_a)

    # Route selected movable blocks toward reachable space off their start cell.
    outside_start = clip(sub(
        sumc(absolute(sub(pixels, block_start))), render_scale
    ))
    parking_valid = mul(
        mul(block_pass, clip(block_field)), outside_start
    )
    roominess = scale(box(block_floor, 7), 5.0)
    parking_score = mul(add(block_field, roominess), parking_valid)
    parking_goal = peaks(parking_score)
    has_parking = clip(gmax(parking_score))

    meeting_mode = mul(token_mode, can_meet)
    switching_mode = mul(
        mul(token_mode, inv(can_meet)), has_switch
    )
    goal = add(
        add(mul(meeting_goal, meeting_mode),
            mul(switch_goal, switching_mode)),
        mul(parking_goal, block_mode)
    )
    navigation_pass = add(
        mul(normal_pass, token_mode), mul(block_pass, block_mode)
    )
    navigation = distance_flood(scale(goal, 512.0), navigation_pass)

    # Four potential channels: goal distance, legality, chase, switch retention.
    sampled_maps = cat([
        navigation, navigation_pass, field_a, hold_point
    ])
    nav_deltas = []
    chase_deltas = []
    legal_maps = []
    hold_deltas = []
    for dy, dx in directions:
        sampled = neighbor(sampled_maps, dy, dx)
        legal = sl(sampled, 1, 2)
        nav_deltas.append(mul(
            sub(sl(sampled, 0, 1), navigation), legal
        ))
        chase_deltas.append(mul(
            sub(sl(sampled, 2, 3), field_a), legal
        ))
        legal_maps.append(legal)
        hold_deltas.append(mul(
            sub(sl(sampled, 3, 4), hold_point), legal
        ))
    move_maps = cat(
        nav_deltas + chase_deltas + legal_maps + hold_deltas
    )

    swap = np.array([
        [0, 1, 0, 0], [1, 0, 0, 0],
        [0, 0, 0, 1], [0, 0, 1, 0]
    ], np.float32)
    sign_repeat = np.array([
        [1, 0], [1, 0], [0, 1], [0, 1]
    ], np.float32)

    def reflect(values, sign):
        positive = scale(aff(sign, sign_repeat), 0.5, 0.5)
        return add(
            mul(values, positive),
            mul(aff(values, swap), inv(positive))
        )

    normal_terms = []
    legal_terms = []
    hold_terms = []
    chase_scores = const(0.0, 4)
    switch_weights = [switch_weight_a, switch_weight_b]
    for i in range(2):
        sampled = total(mul(move_maps, sources[i]))
        nav_values = reflect(sl(sampled, 0, 4), signs[i])
        legal_values = reflect(sl(sampled, 8, 12), signs[i])
        retention_values = reflect(sl(sampled, 12, 16), signs[i])
        weight = add(
            meeting_mode, mul(switching_mode, switch_weights[i])
        )
        here = total(mul(navigation, sources[i]))
        distance_weight = scale(here, -1.0 / 512.0, 1.5)
        normal_terms.append(mul(
            nav_values, mul(weight, distance_weight)
        ))
        legal_terms.append(legal_values)
        hold_terms.append(retention_values)
        if i == 1:
            chase_scores = reflect(sl(sampled, 4, 8), signs[i])

    retention_scores = scale(add(hold_terms[0], hold_terms[1]), 8.0)
    normal_scores = add(
        add(normal_terms[0], normal_terms[1]), retention_scores
    )
    progress = clip(scale(
        aff(relu(normal_scores), [[1, 1, 1, 1]]), 400.0
    ))
    use_chase = mul(inv(progress), meeting_mode)
    token_scores = add(
        mul(normal_scores, inv(use_chase)),
        mul(add(chase_scores, retention_scores), use_chase)
    )

    block_sampled = total(mul(move_maps, selected_point))
    movement_scores = add(
        mul(token_scores, token_mode),
        mul(sl(block_sampled, 0, 4), block_mode)
    )
    mobility = add(
        mul(add(legal_terms[0], legal_terms[1]), token_mode),
        mul(sl(block_sampled, 8, 12), block_mode)
    )
    movement_scores = add(movement_scores, scale(clip(mobility), 0.15))

    # Bounded exploration phase and mild action continuity.
    ge8 = clip(scale(old_phase, 1.0, -7.0))
    ge16 = clip(scale(old_phase, 1.0, -15.0))
    ge24 = clip(scale(old_phase, 1.0, -23.0))
    phase_priority = aff(cat([
        inv(ge8), sub(ge8, ge16), sub(ge16, ge24), ge24
    ]), [
        [0, 1, 0, 0], [0, 0, 0, 1],
        [0, 0, 1, 0], [1, 0, 0, 0]
    ])
    previous_moves = aff("aprev", [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
    ])
    momentum = lin(
        [previous_moves, aff(previous_moves, swap), phase_priority],
        [0.15, -0.20, 0.05]
    )
    movement_scores = add(movement_scores, momentum)

    first_probe = mul(repick, token_mode)
    second_probe = mul(eq_integer(old_step, 1.0), token_mode)
    probing = clip(add(first_probe, second_probe))
    movement_scores = add(
        movement_scores,
        aff(cat([first_probe, second_probe]), [
            [0, 64], [0, 0], [0, 0], [64, 0]
        ])
    )

    # Prefer blocks next to both disconnected token components.
    reach_a = clip(box(clip(field_a), 7))
    reach_b = clip(box(clip(field_b), 7))
    bridge = mul(reach_a, reach_b)
    blocker_click = mul(
        blocker9,
        lin(
            [bridge, add(reach_a, reach_b),
             scale(box(closest_field, 7), 1.0 / 49.0)],
            [1000.0, 100.0, 1.0], 1.0
        )
    )
    need_blocker = mul(
        mul(inv(can_meet), inv(has_switch)), gmax(blocker9)
    )
    need_blocker = mul(
        mul(need_blocker, token_mode), inv(probing)
    )

    block_here = total(mul(navigation, selected_point))
    near_parking = clip(add(
        scale(block_here, 1.0, -511.0), scale(render_scale, 0.5)
    ))
    moved_from_start = clip(add(
        aff(absolute(sub(selected_position, block_start)), [[1, 1]]),
        scale(render_scale, -1.0, 1.0)
    ))
    parked = mul(mul(near_parking, moved_from_start), has_parking)
    release_block = mul(
        block_mode,
        clip(add(add(can_meet, parked), inv(has_parking)))
    )
    click_required = clip(add(need_blocker, release_block))

    # A token is an in-board, non-block click target and safely deselects.
    click_value = add(
        mul(blocker_click, token_mode), mul(paint, block_mode)
    )
    graph.append({
        "id": "click", "op": "add",
        "in": [click_value, scale(spatial_one, 0.0)]
    })
    shape["click"] = (1, True)

    action_w = np.zeros((8, 4), np.float32)
    action_w[1:5] = np.eye(4, dtype=np.float32)
    base_logits = aff(
        movement_scores, action_w,
        [-10000, 0, 0, 0, 0, -10000, -10000, -10000]
    )
    click_w = np.zeros((8, 1), np.float32)
    click_w[6, 0] = 20000.0
    logits_value = add(base_logits, aff(click_required, click_w))
    graph.append({
        "id": "logits", "op": "vadd",
        "in": [logits_value, const(0.0, 8)]
    })
    shape["logits"] = (8, False)

    wrap = clip(scale(old_phase, 1.0, -30.0))
    next_phase = lin([old_phase, wrap], [1.0, -32.0], 1.0)
    memory_value = cat([
        positions[0], positions[1], signs[0], signs[1],
        one, scale(old_step, 1.0, 1.0), next_phase,
        new_frozen_positions[0], new_frozen_flags[0],
        new_frozen_positions[1], new_frozen_flags[1],
        block_start, block_mode, visible_count, merged,
        visited, gate_exists,
        hold_timer, hold_position, switch_goal_position
    ])
    graph.append({
        "id": "mem_next", "op": "vadd",
        "in": [memory_value, const(0.0, M)]
    })
    shape["mem_next"] = (M, False)

    # Static pruning of unused construction intermediates and parameters.
    by_id = {item["id"]: item for item in graph}
    loop_outputs = {}
    for item in graph:
        if item["op"] == "loop":
            for state in item["state"]:
                loop_outputs[item["id"] + "." + state["id"]] = item["id"]

    needed = set()

    def visit(name):
        name = loop_outputs.get(name, name)
        if name not in by_id or name in needed:
            return
        needed.add(name)
        item = by_id[name]
        if item["op"] != "loop":
            for source in item["in"]:
                visit(source)
            return
        local = {s["id"] for s in item["state"]}
        local.update(n["id"] for n in item["body"])
        for state in item["state"]:
            if "init" in state:
                visit(state["init"])
        for body_item in item["body"]:
            for source in body_item["in"]:
                if source not in local:
                    visit(source)

    for output in ("logits", "click", "mem_next"):
        visit(output)
    graph = [item for item in graph if item["id"] in needed]

    used = set()

    def collect(nodes):
        for item in nodes:
            if item["op"] == "loop":
                collect(item["body"])
            else:
                for field in ("w", "b"):
                    if field in item:
                        used.add(item[field])

    collect(graph)
    params = {name: value for name, value in params.items() if name in used}
    return {"graph": graph, "memory_size": M, "params": params}
