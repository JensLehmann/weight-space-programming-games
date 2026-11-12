import numpy as np


def build():
    N = 40
    M = 5 * N + 11
    graph = []
    params = {}
    pcache = {}
    ccache = {}
    serial = [0]

    def fresh():
        name = "n%d" % serial[0]
        serial[0] += 1
        return name

    def node(op, inputs=None, **fields):
        name = fresh()
        nd = {"id": name, "op": op}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        graph.append(nd)
        return name

    def parameter(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in pcache:
            name = "p%d" % len(params)
            pcache[key] = name
            params[name] = a
        return pcache[key]

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def pw(x, w, b=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b)

    def c(value):
        a = np.asarray(value, np.float32).reshape(-1)
        key = (id(graph), len(a), a.tobytes())
        if key not in ccache:
            ccache[key] = dense(
                "avail", np.zeros((len(a), 8), np.float32), a
            )
        return ccache[key]

    def sl(x, a, b, spatial=False):
        return node(
            "slice" if spatial else "vslice", [x],
            start=int(a), stop=int(b)
        )

    def vs(x, a):
        return node("vmul", [x, c(a)])

    def va(x, a=1, b=0):
        y = x if a == 1 else vs(x, a)
        return y if b == 0 else node("vadd", [y, c(b)])

    def vsub(a, b):
        return node("vadd", [a, vs(b, -1)])

    def vabs(x):
        return node("vadd", [
            node("vrelu", [x]), node("vrelu", [vs(x, -1)])
        ])

    def vclip(x):
        return vsub(
            node("vrelu", [x]),
            node("vrelu", [va(x, b=-1)])
        )

    def vmax(a, b):
        return node("vadd", [a, node("vrelu", [vsub(b, a)])])

    def vmax_vector(x, n):
        y = sl(x, 0, 1)
        for i in range(1, n):
            y = vmax(y, sl(x, i, i + 1))
        return y

    def replace(old, new, mask):
        return node("vadd", [
            node("vmul", [old, va(mask, -1, 1)]),
            node("vmul", [new, mask])
        ])

    def smul(x, a):
        return node("mul", [x, c(a)])

    def sadd(x, b):
        return node("add", [x, c(b)])

    def saff(x, a, b):
        return pw(
            x, np.array([[a]], np.float32),
            np.array([b], np.float32)
        )

    def sclip(x):
        return node("min2", [node("relu", [x]), c(1)])

    def channel_sum(x, n):
        return pw(x, np.ones((1, n), np.float32))

    def pool_sum(x, n, factor=4096):
        return dense(
            node("gmean", [x]),
            np.eye(n, dtype=np.float32) * factor
        )

    def first_match(x, n):
        prefix = np.tril(np.ones((n, n), np.float32), -1)
        return node("vrelu", [vsub(x, dense(x, prefix))])

    def decode_index(x, n):
        d = dense(
            x, np.ones((n, 1), np.float32),
            -np.arange(n, dtype=np.float32)
        )
        return vclip(va(vabs(d), -4, 3))

    def dot(a, b, n):
        return dense(
            node("vmul", [a, b]), np.ones((1, n), np.float32)
        )

    def enumerate_points(mask, count, key):
        nonlocal graph
        initial_selector = np.zeros(count, np.float32)
        initial_selector[0] = 1
        initial_selector = c(initial_selector)

        rotation = np.zeros((count, count), np.float32)
        for i in range(count):
            rotation[(i + 1) % count, i] = 1

        remaining_state = fresh()
        result_state = fresh()
        selector_state = fresh()

        outer = graph
        graph = []
        body = graph

        largest = node("gmax", [
            node("mul", [remaining_state, key])
        ])
        distance = node("abs", [node("sub", [key, largest])])
        winner = node("mul", [
            remaining_state,
            node("relu", [saff(distance, -2, 1)])
        ])
        remaining_next = node("sub", [remaining_state, winner])
        result_next = node("add", [
            result_state, node("mul", [winner, selector_state])
        ])
        selector_next = dense(selector_state, rotation)

        graph = outer
        loop_id = node(
            "loop", T=count,
            state=[
                {"id": remaining_state, "init": mask},
                {"id": result_state, "zeros": [count, 64, 64]},
                {"id": selector_state, "init": initial_selector}
            ],
            body=body,
            next={
                remaining_state: remaining_next,
                result_state: result_next,
                selector_state: selector_next
            }
        )
        return loop_id + "." + result_state

    rowpx = pw("coords", np.array([[63, 0, 0]], np.float32))
    colpx = pw("coords", np.array([[0, 63, 0]], np.float32))
    key = pw(
        "coords", np.array([[-8064, -63, 10000]], np.float32)
    )

    # Detect completely filled tile squares at either rendering scale.
    colours = [0, 1, 2, 6, 7, 9, 10, 11, 12, 13, 15]
    nc = len(colours)
    select = np.zeros((nc, 16), np.float32)
    for i, colour in enumerate(colours):
        select[i, colour] = 1
    planes = pw("x0", select)

    def square_anchors(size):
        w = np.zeros((nc, nc, 7, 7), np.float32)
        for i in range(nc):
            for dy in range(size):
                for dx in range(size):
                    w[i, i, 3 + dy, 3 + dx] = 1
            w[i, i, 2, 3] = -1
            w[i, i, 3, 2] = -1
        return node("relu", [
            conv(
                planes, w,
                np.full(nc, 1 - size * size, np.float32)
            )
        ])

    small = square_anchors(2)
    large = square_anchors(4)
    scale2 = vclip(node("gmax", [channel_sum(large, nc)]))
    scale1 = va(scale2, -1, 1)
    anchors = node("add", [
        node("mul", [small, scale1]),
        node("mul", [large, scale2])
    ])
    tile_mask = sclip(channel_sum(anchors, nc))

    def corners(offsets):
        w = np.zeros((2, 16, 7, 7), np.float32)
        for k, colour in enumerate([11, 12]):
            for dy, dx in offsets:
                w[k, colour, 3 + dy, 3 + dx] = 1
        return node("relu", [
            conv("x0", w, np.full(2, -3, np.float32))
        ])

    tsmall = corners([(-1, -1), (-1, 2), (2, -1), (2, 2)])
    tlarge = corners([(-3, -3), (-3, 3), (3, -3), (3, 3)])
    edge = np.zeros((2, 2, 3, 3), np.float32)
    shift = np.zeros_like(edge)
    for k in range(2):
        edge[k, k, 1, 1] = 1
        edge[k, k, 0, 1] = -1
        edge[k, k, 1, 0] = -1
        shift[k, k, 2, 2] = 1
    tlarge = conv(node("relu", [conv(tlarge, edge)]), shift)
    targets = node("add", [
        node("mul", [tsmall, scale1]),
        node("mul", [tlarge, scale2])
    ])
    tile_mask = node("max2", [
        tile_mask, sclip(channel_sum(targets, 2))
    ])

    points = enumerate_points(tile_mask, N, key)
    occupancy = pool_sum(points, N)
    colour_field = pw(
        "x0", (np.arange(16, dtype=np.float32) + 1)[None, :]
    )
    colours_now = pool_sum(node("mul", [points, colour_field]), N)
    xs = pool_sum(node("mul", [points, colpx]), N)
    ys = pool_sum(node("mul", [points, rowpx]), N)
    tb = pool_sum(node("mul", [
        points, sl(targets, 0, 1, True)
    ]), N)
    to = pool_sum(node("mul", [
        points, sl(targets, 1, 2, True)
    ]), N)

    # Physical control components, excluding the horizontal and vertical UI.
    bw = np.zeros((2, 16, 5, 5), np.float32)
    for k, colour in enumerate([8, 14]):
        bw[k, colour, 2, 2] = 1
        bw[k, colour, 2, 1] = -1
        for dx in range(-2, 3):
            bw[k, colour, 1, 2 + dx] = -1
    raw_buttons = node("relu", [conv("x0", bw)])
    crop = node("mul", [
        sclip(saff(rowpx, 1, -1)), sclip(colpx)
    ])
    left_candidates = node("mul", [
        sl(raw_buttons, 0, 1, True), crop
    ])
    right_candidates = node("mul", [
        sl(raw_buttons, 1, 2, True), crop
    ])
    right_maps = enumerate_points(right_candidates, 4, key)
    right_exists = pool_sum(right_maps, 4)
    right_x = pool_sum(node("mul", [right_maps, colpx]), 4)
    right_y = pool_sum(node("mul", [right_maps, rowpx]), 4)

    pdx = node("abs", [node("sub", [colpx, right_x])])
    pdy = node("abs", [node("sub", [rowpx, right_y])])
    palign = node("min2", [pdx, pdy])
    partner_score = node("add", [
        smul(palign, -10),
        smul(node("add", [pdx, pdy]), -1)
    ])
    partner_tie = pw(
        "coords",
        np.array([[-63 / 256, -63 / 16384, 0]], np.float32)
    )
    partner_score = node("add", [partner_score, partner_tie])
    partner_score = node("add", [
        partner_score, saff(left_candidates, 2048, -2048)
    ])
    partner_best = node("gmax", [partner_score])
    partner_gap = node("abs", [
        node("sub", [partner_score, partner_best])
    ])
    left_maps = node("mul", [
        left_candidates,
        node("relu", [sadd(smul(partner_gap, -100), 1)])
    ])
    left_maps = node("mul", [left_maps, right_exists])
    action_maps = node("concat", [right_maps, left_maps])

    # An artificial index lattice stores pairwise tile relations spatially.
    def index_basis(axis):
        w = np.zeros((N, 3), np.float32)
        w[:, axis] = 63
        delta = pw(
            "coords", w, -np.arange(N, dtype=np.float32)
        )
        shaped = pw(
            node("abs", [delta]),
            -4 * np.eye(N, dtype=np.float32),
            np.full(N, 3, np.float32)
        )
        return node("min2", [node("relu", [shaped]), c(1)])

    rb = index_basis(0)
    cb = index_basis(1)

    def rfield(v):
        return channel_sum(node("mul", [rb, v]), N)

    def cfield(v):
        return channel_sum(node("mul", [cb, v]), N)

    lattice_valid = node("mul", [
        channel_sum(rb, N), channel_sum(cb, N)
    ])
    diagonal = node("relu", [
        saff(node("abs", [node("sub", [rowpx, colpx])]), -1, 1)
    ])

    # Memory: previous colours, four permutations, readiness flags,
    # previous control/direction, initialization flag, exploration phase.
    old_colours = sl("mem", 0, N)
    old_models = sl("mem", N, 5 * N)
    old_ready = sl("mem", 5 * N, 5 * N + 4)
    previous_model = sl("mem", 5 * N + 4, 5 * N + 5)
    previous_inverse = sl("mem", 5 * N + 5, 5 * N + 6)
    seen = sl("mem", 5 * N + 6, 5 * N + 7)
    old_phase = sl("mem", 5 * N + 7, 5 * N + 11)

    expand_model = np.zeros((4 * N, 4), np.float32)
    collect_model = np.zeros((N, 4 * N), np.float32)
    replicate_model = np.zeros((4 * N, N), np.float32)
    for k in range(4):
        expand_model[k * N:(k + 1) * N, k] = 1
        collect_model[:, k * N:(k + 1) * N] = np.eye(N)
        replicate_model[k * N:(k + 1) * N] = np.eye(N)

    identity_models = np.tile(
        np.arange(1, N + 1, dtype=np.float32), 4
    )
    old_default_models = node("vadd", [
        old_models,
        node("vmul", [
            va(dense(old_ready, expand_model), -1, 1),
            c(identity_models)
        ])
    ])
    previous_selector = node("vmul", [
        decode_index(va(previous_model, b=-1), 4), seen
    ])
    previous_block_gate = dense(previous_selector, expand_model)
    previous_perm = dense(node("vmul", [
        old_default_models, previous_block_gate
    ]), collect_model)
    previous_known = dot(previous_selector, old_ready, 4)

    # Colour-compatible matching. Local moves, inactive identities, and
    # the previous estimate supply preferences for ambiguous equal colours.
    old_colour_rows = rfield(old_colours)
    new_colour_cols = cfield(colours_now)
    colour_match = node("relu", [
        saff(node("abs", [
            node("sub", [old_colour_rows, new_colour_cols])
        ]), -1, 1)
    ])
    rx, cx = rfield(xs), cfield(xs)
    ry, cy = rfield(ys), cfield(ys)
    pair_distance = node("add", [
        node("abs", [node("sub", [rx, cx])]),
        node("abs", [node("sub", [ry, cy])])
    ])

    changed = vclip(vs(vabs(vsub(old_colours, colours_now)), 4))
    changed_rows = rfield(changed)
    changed_cols = cfield(changed)
    radius = va(scale2, 6, 6.5)
    nearby = sclip(node("sub", [radius, pair_distance]))
    changed_neighbours = node("rowsum", [
        node("mul", [nearby, changed_cols])
    ])
    inferred_active = sclip(saff(changed_neighbours, 2, -2))
    active = node("max2", [changed_rows, inferred_active])

    previous_dest_rows = rfield(va(previous_perm, b=-1))
    previous_dest_cols = cfield(va(previous_perm, b=-1))
    prior_forward = node("relu", [
        saff(node("abs", [
            node("sub", [colpx, previous_dest_rows])
        ]), -1, 1)
    ])
    prior_inverse = node("relu", [
        saff(node("abs", [
            node("sub", [rowpx, previous_dest_cols])
        ]), -1, 1)
    ])
    prior = node("add", [
        node("mul", [prior_forward, va(previous_inverse, -1, 1)]),
        node("mul", [prior_inverse, previous_inverse])
    ])
    prior = node("mul", [prior, previous_known])

    static_identity = node("mul", [
        diagonal, sadd(smul(active, -1), 1)
    ])
    active_identity = node("mul", [diagonal, active])
    match_score = node("sub", [
        smul(colour_match, 1024), pair_distance
    ])
    match_score = node("add", [
        match_score, smul(static_identity, 64)
    ])
    match_score = node("sub", [
        match_score,
        node("mul", [active_identity, va(scale2, 16, 16)])
    ])
    match_score = node("add", [
        match_score, smul(prior, 128)
    ])
    match_tie = pw(
        "coords",
        np.array([[-63 / 64, -63 / 4096, 0]], np.float32)
    )
    match_score = node("add", [match_score, match_tie])
    match_score = node("add", [
        match_score, saff(lattice_valid, 1000000, -1000000)
    ])

    outer = graph
    graph = []
    matching_body = graph
    edge_best = node("gmax", ["matching_scores"])
    edge_gap = node("abs", [
        node("sub", ["matching_scores", edge_best])
    ])
    selected_edge = node("relu", [saff(edge_gap, -8192, 1)])
    blocked = node("add", [
        node("rowmax", [selected_edge]),
        node("colmax", [selected_edge])
    ])
    scores_next = node("sub", [
        "matching_scores", smul(blocked, 4096)
    ])
    matches_next = node("add", ["matching_result", selected_edge])
    graph = outer
    graph.append({
        "id": "matching", "op": "loop", "T": N,
        "state": [
            {"id": "matching_scores", "init": match_score},
            {"id": "matching_result", "zeros": [1, 64, 64]}
        ],
        "body": matching_body,
        "next": {
            "matching_scores": scores_next,
            "matching_result": matches_next
        }
    })

    destination_rows = node("rowsum", [
        node("mul", ["matching.matching_result", sadd(colpx, 1)])
    ])
    source_columns = node("colsum", [
        node("mul", ["matching.matching_result", sadd(rowpx, 1)])
    ])
    observed_forward = pool_sum(
        node("mul", [rb, destination_rows]), N, 64
    )
    observed_inverse = pool_sum(
        node("mul", [cb, source_columns]), N, 64
    )
    observed_canonical = node("vadd", [
        node("vmul", [
            observed_forward, va(previous_inverse, -1, 1)
        ]),
        node("vmul", [observed_inverse, previous_inverse])
    ])
    models_next = replace(
        old_models,
        dense(observed_canonical, replicate_model),
        previous_block_gate
    )
    ready_next = replace(
        old_ready, c(np.ones(4, np.float32)), previous_selector
    )
    models = node("vadd", [
        models_next,
        node("vmul", [
            va(dense(ready_next, expand_model), -1, 1),
            c(identity_models)
        ])
    ])

    # Identify goal locations and their type-compatible target sets.
    def colour_is(encoded_colour):
        d = dense(
            colours_now, np.eye(N, dtype=np.float32),
            np.full(N, -encoded_colour, np.float32)
        )
        return node("vrelu", [va(vabs(d), -1, 1)])

    gb = colour_is(12)
    go = colour_is(13)
    goal_union = node("vadd", [gb, go])
    goal_one = first_match(goal_union, N)
    goal_two_raw = vsub(goal_union, goal_one)
    has_two = vclip(dense(
        goal_two_raw, np.ones((1, N), np.float32)
    ))
    goal_two = node("vadd", [
        goal_two_raw,
        node("vmul", [goal_one, va(has_two, -1, 1)])
    ])
    first_is_b = dot(goal_one, gb, N)
    second_is_b = dot(goal_two, gb, N)
    target_one = node("vadd", [
        node("vmul", [tb, first_is_b]),
        node("vmul", [to, va(first_is_b, -1, 1)])
    ])
    target_two_typed = node("vadd", [
        node("vmul", [tb, second_is_b]),
        node("vmul", [to, va(second_is_b, -1, 1)])
    ])
    target_two = node("vadd", [
        node("vmul", [target_two_typed, has_two]),
        node("vmul", [occupancy, va(has_two, -1, 1)])
    ])
    terminal = node("mul", [
        rfield(target_one), cfield(target_two)
    ])
    terminal = node("mul", [
        terminal,
        sadd(smul(node("mul", [diagonal, has_two]), -1), 1)
    ])
    seed = smul(terminal, 128)

    # Joint-goal value propagation under each estimated permutation and
    # its inverse. A rotating selector shares the routing computation.
    cycle4 = np.zeros((4, 4), np.float32)
    for k in range(4):
        cycle4[(k + 1) % 4, k] = 1

    def route_rows(value, take, put):
        extracted = node("colmax", [node("mul", [value, take])])
        return channel_sum(node("mul", [extracted, put]), N)

    def route_cols(value, take, put):
        extracted = node("rowmax", [node("mul", [value, take])])
        return channel_sum(node("mul", [extracted, put]), N)

    def permutation_basis(axis, destinations):
        d = node("abs", [
            node("sub", [axis, destinations])
        ])
        return sclip(sadd(smul(d, -4), 3))

    outer = graph
    graph = []
    planning_body = graph
    planning_gate = dense("planning_selector", expand_model)
    planning_perm = dense(node("vmul", [
        models, planning_gate
    ]), collect_model)
    planning_dest = va(planning_perm, b=-1)
    perm_rows = permutation_basis(rowpx, planning_dest)
    perm_cols = permutation_basis(colpx, planning_dest)
    vf = route_cols(
        route_rows("planning_value", perm_rows, rb),
        perm_cols, cb
    )
    vi = route_cols(
        route_rows("planning_value", rb, perm_rows),
        cb, perm_cols
    )
    candidate_value = node("relu", [
        sadd(node("max2", [vf, vi]), -1)
    ])
    planning_next = node("max2", [
        "planning_value", candidate_value
    ])
    selector_next = dense("planning_selector", cycle4)
    graph = outer
    initial_planning_selector = c([1, 0, 0, 0])
    graph.append({
        "id": "planner", "op": "loop", "T": 224,
        "state": [
            {"id": "planning_value", "init": seed},
            {
                "id": "planning_selector",
                "init": initial_planning_selector
            }
        ],
        "body": planning_body,
        "next": {
            "planning_value": planning_next,
            "planning_selector": selector_next
        }
    })

    indices = np.arange(N, dtype=np.float32)[None, :]
    first_index = dense(goal_one, indices)
    second_index = dense(goal_two, indices)

    def inverse_destination(perm, index):
        delta = vsub(perm, va(index, b=1))
        gate = vclip(va(vabs(delta), -4, 3))
        return dense(gate, indices)

    def read_value(i, j):
        distance = node("add", [
            node("abs", [node("sub", [rowpx, i])]),
            node("abs", [node("sub", [colpx, j])])
        ])
        mask = sclip(saff(distance, -4, 3))
        return node("gmax", [
            node("mul", ["planner.planning_value", mask])
        ])

    forward_values = []
    inverse_values = []
    for k in range(4):
        perm = sl(models, k * N, (k + 1) * N)
        destinations = va(perm, b=-1)
        fi = dot(goal_one, destinations, N)
        fj = dot(goal_two, destinations, N)
        ii = inverse_destination(perm, first_index)
        ij = inverse_destination(perm, second_index)
        forward_values.append(read_value(fi, fj))
        inverse_values.append(read_value(ii, ij))
    values = node("vconcat", forward_values + inverse_values)

    # Integer-value threshold bank removes small routing roundoff errors.
    expand_values = np.zeros((8 * 128, 8), np.float32)
    reduce_values = np.zeros((8, 8 * 128), np.float32)
    thresholds = np.empty(8 * 128, np.float32)
    for a in range(8):
        expand_values[a * 128:(a + 1) * 128, a] = 20
        reduce_values[a, a * 128:(a + 1) * 128] = 1
        thresholds[a * 128:(a + 1) * 128] = (
            -20 * (np.arange(128, dtype=np.float32) + 0.5) + 0.5
        )
    rounded_values = dense(
        vclip(dense(values, expand_values, thresholds)),
        reduce_values
    )

    # Probe unseen controls; thereafter choose the largest propagated value.
    unknown = node("vmul", [
        right_exists, va(ready_next, -1, 1)
    ])
    probe = first_match(unknown, 4)
    probe8 = node("vconcat", [probe, c(np.zeros(4, np.float32))])
    phase = replace(old_phase, c([1, 0, 0, 0]), va(seen, -1, 1))
    phase8 = node("vconcat", [phase, c(np.zeros(4, np.float32))])
    reachable = vclip(vmax_vector(rounded_values, 8))
    fallback = node("vmul", [phase8, va(reachable, -1, 1)])
    available_controls = node("vconcat", [
        right_exists, right_exists
    ])

    scores = node("vadd", [rounded_values, vs(probe8, 1024)])
    scores = node("vadd", [scores, vs(fallback, 4)])
    scores = node("vadd", [
        scores, va(available_controls, 4096, -4096)
    ])
    scores = node("vadd", [
        scores, c(-0.01 * np.arange(8, dtype=np.float32))
    ])
    best = vmax_vector(scores, 8)
    chosen = first_match(
        vclip(va(vabs(vsub(scores, best)), -500, 3)), 8
    )

    scored_clicks = node("mul", [
        action_maps, va(scores, b=8192)
    ])
    graph.append({
        "id": "click", "op": "conv", "in": [scored_clicks],
        "w": parameter(np.ones((1, 8, 1, 1), np.float32))
    })

    chosen_model = node("vadd", [
        sl(chosen, 0, 4), sl(chosen, 4, 8)
    ])
    next_previous_model = dense(
        chosen_model, np.arange(1, 5, dtype=np.float32)[None, :]
    )
    next_previous_inverse = dense(
        sl(chosen, 4, 8), np.ones((1, 4), np.float32)
    )
    next_phase = dense(phase, cycle4)
    graph.append({
        "id": "mem_next", "op": "vconcat",
        "in": [
            colours_now, models_next, ready_next,
            next_previous_model, next_previous_inverse,
            c(1), next_phase
        ]
    })
    bias = np.zeros(8, np.float32)
    bias[6] = 1
    graph.append({
        "id": "logits", "op": "dense", "in": ["avail"],
        "w": parameter(np.zeros((8, 8), np.float32)),
        "b": parameter(bias)
    })

    # Static common-subexpression elimination.
    def simplify(nodes, inherited_alias=None, inherited_cache=None):
        aliases = dict(inherited_alias or {})
        cache = dict(inherited_cache or {})
        result = []

        def resolve(x):
            while x in aliases:
                x = aliases[x]
            return x

        for original in nodes:
            nd = dict(original)
            if nd["op"] == "loop":
                states = []
                for state in nd["state"]:
                    state = dict(state)
                    if "init" in state:
                        state["init"] = resolve(state["init"])
                    states.append(state)
                body, body_aliases = simplify(
                    nd["body"], aliases, cache
                )
                nd["state"] = states
                nd["body"] = body
                next_values = {}
                for state, value in nd["next"].items():
                    while value in body_aliases:
                        value = body_aliases[value]
                    next_values[state] = value
                nd["next"] = next_values
                result.append(nd)
                continue

            nd["in"] = [resolve(x) for x in nd["in"]]
            signature = repr(sorted(
                (k, v) for k, v in nd.items()
                if k not in ("id", "comment")
            ))
            if nd["id"] not in ("click", "logits", "mem_next"):
                if signature in cache:
                    aliases[nd["id"]] = cache[signature]
                    continue
                cache[signature] = nd["id"]
            result.append(nd)
        return result, aliases

    graph, _ = simplify(graph)

    used = set()

    def collect(nodes):
        for nd in nodes:
            if nd["op"] == "loop":
                collect(nd["body"])
            else:
                for field in ("w", "b"):
                    if field in nd:
                        used.add(nd[field])

    collect(graph)
    params = {k: v for k, v in params.items() if k in used}

    # Architecture-only bookkeeping selects a budget-safe fixed loop length.
    spatial_channels = [19]
    node_count = [0]
    shapes = {
        "x0": (16, 64, 64), "coords": (3, 64, 64),
        "mem": (M,), "aprev": (8,), "avail": (8,)
    }

    def tally(nodes, env):
        macs = 0
        evaluations = 0
        for nd in nodes:
            node_count[0] += 1
            op = nd["op"]
            if op == "loop":
                local = dict(env)
                for state in nd["state"]:
                    shape = (
                        env[state["init"]] if "init" in state
                        else tuple(state["zeros"])
                    )
                    local[state["id"]] = shape
                    if len(shape) == 3:
                        spatial_channels[0] += shape[0]
                body_macs, body_evals = tally(nd["body"], local)
                nd["_body_cost"] = (body_macs, body_evals)
                macs += nd["T"] * body_macs
                evaluations += 1 + nd["T"] * body_evals
                for state, source in nd["next"].items():
                    env[nd["id"] + "." + state] = local[source]
                continue

            ins = [env[x] for x in nd["in"]]
            x = ins[0]
            cost = 0
            if op == "conv":
                w = params[nd["w"]]
                shape = (w.shape[0], 64, 64)
                cost = int(np.prod(w.shape)) * 4096
            elif op == "dense":
                w = params[nd["w"]]
                shape = (w.shape[0],)
                cost = int(np.prod(w.shape))
            elif op in ("concat", "vconcat"):
                shape = (sum(s[0] for s in ins),) + x[1:]
            elif op in ("slice", "vslice"):
                shape = (nd["stop"] - nd["start"],) + x[1:]
            elif op in ("gmax", "gmean"):
                shape = (x[0],)
                cost = int(np.prod(x))
            elif op in ("add", "sub", "mul", "max2", "min2"):
                shape = (max(s[0] for s in ins), 64, 64)
                cost = int(np.prod(shape))
            elif op in ("vadd", "vmul"):
                shape = (max(s[0] for s in ins),)
                cost = shape[0]
            else:
                shape = x
                if op in ("rowmax", "colmax", "rowsum", "colsum"):
                    cost = int(np.prod(x))
            env[nd["id"]] = shape
            if len(shape) == 3:
                spatial_channels[0] += shape[0]
            macs += cost
            evaluations += 1
        return macs, evaluations

    total_macs, total_evals = tally(graph, shapes)
    for nd in graph:
        if nd["id"] == "planner":
            body_macs, body_evals = nd["_body_cost"]
            outside_macs = total_macs - nd["T"] * body_macs
            outside_evals = total_evals - nd["T"] * body_evals
            safe = min(
                nd["T"],
                (1500000000 - outside_macs) // max(1, body_macs),
                (10000 - outside_evals) // max(1, body_evals)
            )
            nd["T"] = int(max(1, (safe // 4) * 4))

    def clean(nodes):
        for nd in nodes:
            nd.pop("_body_cost", None)
            if nd["op"] == "loop":
                clean(nd["body"])

    clean(graph)

    return {
        "graph": graph,
        "memory_size": M,
        "params": params
    }
