import numpy as np


def build():
    params = {}
    graph = []
    sink = graph
    serial = [0]
    pserial = [0]
    HISTORY = 32
    STACK = 10
    MEMORY = HISTORY + 5

    def parameter(a):
        name = "p%d" % pserial[0]
        pserial[0] += 1
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs, name=None, **kwargs):
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(kwargs)
        sink.append(d)
        return name

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("conv", [x], name=name, **kw)

    def dense(x, w, b=None, name=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("dense", [x], name=name, **kw)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vs(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

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

    def minimum(a, b):
        return node("min2", [a, b])

    def relu(x):
        return node("relu", [x])

    def absolute(x):
        return node("abs", [x])

    def gmean(x):
        return node("gmean", [x])

    def gmax(x):
        return node("gmax", [x])

    def vadd(a, b):
        return node("vadd", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def constant(values):
        values = np.atleast_1d(np.asarray(values, np.float32))
        return conv("coords", np.zeros((len(values), 3)), values)

    zero = constant([0])
    one = constant([1])

    def affine(x, scale, bias, channels=1):
        return conv(
            x,
            np.eye(channels, dtype=np.float32) * scale,
            np.full(channels, bias, np.float32)
        )

    def clip(x):
        return minimum(relu(x), one)

    def total(x, channels):
        return conv(x, np.ones((1, channels), np.float32))

    def lift(x):
        return add(zero, x)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def shifts(x, channels=1, step=1):
        size = 2 * step + 1
        w = np.zeros((4 * channels, channels, size, size), np.float32)
        for d, (dy, dx) in enumerate(directions):
            for c in range(channels):
                w[d * channels + c, c,
                  step + dy * step, step + dx * step] = 1
        return conv(x, w)

    def directional_shift(x):
        w = np.zeros((4, 4, 3, 3), np.float32)
        for d, (dy, dx) in enumerate(directions):
            w[d, d, 1 + dy, 1 + dx] = 1
        return conv(x, w)

    def deposit(x, step):
        size = 2 * step + 1
        w = np.zeros((1, 4, size, size), np.float32)
        for d, (dy, dx) in enumerate(directions):
            w[0, d, step - dy * step, step - dx * step] = 1
        return conv(x, w)

    def channel_max(x, groups=4, width=1):
        parts = [sl(x, i * width, (i + 1) * width)
                 for i in range(groups)]
        while len(parts) > 1:
            parts = [
                maximum(parts[i], parts[i + 1])
                if i + 1 < len(parts) else parts[i]
                for i in range(0, len(parts), 2)
            ]
        return parts[0]

    def vector_max(x, length):
        while length > 1:
            half = length // 2
            wd = np.zeros((half, length), np.float32)
            wb = np.zeros((half, length), np.float32)
            for i in range(half):
                wd[i, 2 * i] = 1
                wd[i, 2 * i + 1] = -1
                wb[i, 2 * i + 1] = 1
            diff = node("vrelu", [dense(x, wd)])
            x = vadd(diff, dense(x, wb))
            length = half
        return x

    # Original coordinate construction, without fusing its threshold layer.
    wr = np.zeros((132, 3), np.float32)
    br = np.zeros(132, np.float32)
    for axis in range(2):
        for t, j in enumerate(range(-1, 65)):
            wr[axis * 66 + t, axis] = 63
            br[axis * 66 + t] = -j
    ramps = relu(conv("coords", wr, br))
    linew = np.zeros((56, 132), np.float32)

    def add_hat(out, axis, j):
        base = axis * 66
        linew[out, base + j] += 1
        linew[out, base + j + 1] -= 2
        linew[out, base + j + 2] += 1

    for axis in range(2):
        base = axis * 28
        for b in range(11):
            for j in range(6 * b, min(6 * b + 6, 64)):
                add_hat(base + b, axis, j)
        for phase in range(6):
            for j in range(phase, 64, 6):
                add_hat(base + 11 + phase, axis, j)
        for b in range(11):
            add_hat(base + 17 + b, axis, 24 + b)

    lines = clip(affine(conv(ramps, linew), 8, -4, 56))
    rb = sl(lines, 0, 11)
    rp = sl(lines, 11, 17)
    rc = sl(lines, 17, 28)
    cb = sl(lines, 28, 39)
    cp = sl(lines, 39, 45)
    cc = sl(lines, 45, 56)

    rep_r = np.zeros((121, 11), np.float32)
    rep_c = np.zeros((121, 11), np.float32)
    for i in range(121):
        rep_r[i, i // 11] = 1
        rep_c[i, i % 11] = 1

    bins = mul(conv(rb, rep_r), conv(cb, rep_c))
    compact_masks = mul(conv(rc, rep_r), conv(cc, rep_c))
    domain = total(compact_masks, 121)

    compact_xy = conv(
        compact_masks,
        np.array([[i // 11 for i in range(121)],
                  [i % 11 for i in range(121)]], np.float32)
    )
    compact_x = sl(compact_xy, 1, 2)

    base_index = conv(
        compact_masks,
        np.array([[4 * i + 1 for i in range(121)]], np.float32)
    )
    action_index = conv(
        base_index, np.ones((4, 1), np.float32),
        np.arange(4, dtype=np.float32)
    )
    jump_tie = affine(action_index, -1.0 / 1024.0, 0, 4)

    edge_w = np.zeros((4, 121), np.float32)
    right_w = np.zeros((1, 121), np.float32)
    for i in range(121):
        y, x = divmod(i, 11)
        edge_w[0, i] = float(y == 0)
        edge_w[1, i] = float(y == 10)
        edge_w[2, i] = float(x == 0)
        edge_w[3, i] = float(x == 10)
        right_w[0, i] = float(x >= 9)
    edge_directions = conv(compact_masks, edge_w)
    right_edge = conv(compact_masks, right_w)

    # Extend the integer x coordinate one cell beyond either side.
    wx = np.zeros((1, 4, 3, 3), np.float32)
    wx[0, 2, 1, 2] = -1
    wx[0, 3, 1, 0] = 11
    transport_x = add(compact_x, conv(edge_directions, wx))

    # Orb-only lattice detection: no floor or border anchors.
    orb_pixels = [
        (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 2), (4, 3)
    ]
    colors = [14, 8, 9, 2]
    ow = np.zeros((4, 16, 5, 5), np.float32)
    for c, color in enumerate(colors):
        for y, x in orb_pixels:
            ow[c, color, y, x] = 1
    detected_orbs = relu(conv("x0", ow, [-11] * 4))
    orb_centres = total(detected_orbs, 4)
    gray_any = gmax(sl(detected_orbs, 3, 4))

    row_phase = gmax(mul(rp, orb_centres))
    col_phase = gmax(mul(cp, orb_centres))
    lattice = mul(
        total(mul(rp, row_phase), 6),
        total(mul(cp, col_phase), 6)
    )
    sample_masks = mul(bins, lattice)

    phase_w = np.zeros((2, 12), np.float32)
    phase_w[0, :6] = np.arange(6)
    phase_w[1, 6:] = np.arange(6)
    phase_xy = dense(vcat(row_phase, col_phase), phase_w)

    # Occupant codes: green=1, red=2, blue=3, fixed purple jumper=4.
    w = np.zeros((1, 16), np.float32)
    w[0, 14], w[0, 8], w[0, 9], w[0, 7], w[0, 2] = 1, 2, 3, 4, 1
    occupant_pixels = conv("x0", w)

    w = np.zeros((2, 16, 7, 7), np.float32)
    w[0, 0, 6, 1] = 1
    w[1, 11, 6, 1] = 1
    floor_platform_pixels = conv("x0", w)
    floor_pixels = sl(floor_platform_pixels, 0, 1)
    platform_pixels = sl(floor_platform_pixels, 1, 2)

    w = np.zeros((1, 16), np.float32)
    w[0, 5] = 1
    rail_pixels = maximum(conv("x0", w), platform_pixels)

    def compactify(feature):
        values = gmean(mul(sample_masks, affine(feature, 4096, 0)))
        return total(mul(compact_masks, values), 121)

    O0 = compactify(occupant_pixels)
    F = compactify(floor_pixels)
    P0 = compactify(platform_pixels)
    R = compactify(rail_pixels)

    w = np.zeros((1, 16, 5, 5), np.float32)
    w[0, 3, 2, 0] = 1
    ring = conv("x0", w)
    w = np.zeros((1, 16), np.float32)
    w[0, colors] = 1
    original_selected = mul(mul(ring, conv("x0", w)), lattice)

    def decode(o):
        dif = absolute(conv(o, np.ones((4, 1)), [-1, -2, -3, -4]))
        types = relu(conv(dif, -np.eye(4), np.ones(4)))
        nb = conv(types, [[1, 1, 0, 0]])
        purple = sl(types, 3, 4)
        occ = minimum(o, one)
        moving = sub(occ, purple)
        empty = sub(one, occ)
        return types, nb, purple, occ, moving, empty

    types0, nb0, purple0, occ0, moving0, empty0 = decode(O0)
    blue0 = sl(types0, 2, 3)

    special_now = gmax(conv(types0, [[0, 1, 1, 1]]))
    special = minimum(
        maximum(lift(special_now),
                lift(vs("mem", HISTORY + 4, HISTORY + 5))),
        one
    )
    ordinary = sub(one, special)
    special_v, ordinary_v = gmax(special), gmax(ordinary)

    mid0 = minimum(shifts(O0, 1, 1), one)
    empty2 = relu(sub(one, shifts(O0, 1, 2)))
    f2 = shifts(F, 1, 2)

    # Rail receiving, unloading, and bridge-positioning goal fields.
    receive_weights = add(
        constant([64, 64, 64, 64]),
        mul(special, constant([32, 0, 48, -16]))
    )
    receive = mul(
        add(mul(shifts(nb0, 1, 2), receive_weights),
            affine(shifts(blue0, 1, 2), 20, 0, 4)),
        mid0
    )
    receive = mul(channel_max(receive), R)

    unload = conv(
        shifts(nb0, 1, 1), np.eye(4) * 32,
        [32, 36, 32, 48]
    )
    unload = add(unload, mul(special, constant([8, 28, 0, 32])))
    unload = mul(mul(mul(unload, mid0), f2), empty2)
    unload = mul(channel_max(unload), R)

    reverse = np.zeros((4, 4), np.float32)
    for i, j in enumerate([1, 0, 3, 2]):
        reverse[i, j] = 1
    source1 = conv(shifts(nb0, 1, 1), reverse)
    free_floor1 = shifts(mul(F, empty0), 1, 1)
    bridge_weights = add(
        constant([32, 40, 32, 64]),
        mul(special, constant([-8, 0, -8, 0]))
    )
    bridge = mul(mul(source1, free_floor1), bridge_weights)
    bridge = mul(channel_max(bridge), R)
    rail_seed = cat(receive, unload, bridge)

    rail_body = []
    sink = rail_body
    neighbors = shifts("rail_v", 3, 1)
    propagated = channel_max(neighbors, 4, 3)
    propagated = mul(relu(affine(propagated, 1, -1, 3)), R)
    rail_next = maximum(propagated, rail_seed)
    sink = graph
    graph.append({
        "id": "Rail",
        "op": "loop",
        "T": 48,
        "state": [{"id": "rail_v", "init": rail_seed}],
        "body": rail_body,
        "next": {"rail_v": rail_next}
    })

    E = sl("Rail.rail_v", 0, 1)
    L = sl("Rail.rail_v", 1, 2)
    B = sl("Rail.rail_v", 2, 3)
    E1, L1, B1 = shifts(E), shifts(L), shifts(B)
    R1 = shifts(R)
    empty_weight = affine(special, 3, 1)
    passenger_weight = affine(special, 11, 1)

    # Integer fingerprints. Geometry is included to distinguish scrolling.
    hash_numbers = []
    for i in range(121):
        z = ((i + 1) * 1103515245 + 12345) & 0x7fffffff
        z ^= z >> 11
        hash_numbers.append(1 + z % 509)
    hash_field = conv(
        compact_masks,
        np.array([hash_numbers], np.float32) * 4096
    )
    geometry_hash = gmean(mul(
        add(affine(F, 16, 0), affine(R, 32, 0)), hash_field
    ))

    def board_hash(o, p):
        h = gmean(mul(add(o, affine(p, 8, 0)), hash_field))
        return vadd(h, geometry_hash)

    h0 = board_hash(O0, P0)
    changed = minimum(absolute(sub(lift(h0), vs("mem", 0, 1))), one)
    changed_v = gmax(changed)
    unchanged_v = dense(changed_v, [[-1]], [1])

    history_old = vs("mem", 0, HISTORY)
    history_shift = vcat(h0, vs("mem", 0, HISTORY - 1))
    history_updated = vadd(
        vmul(history_shift, changed_v),
        vmul(history_old, unchanged_v)
    )

    failed_sum = vadd(
        vs("mem", HISTORY, HISTORY + 4),
        vs("aprev", 1, 5)
    )
    failed_field = mul(clip(lift(failed_sum)), unchanged_v)
    failed_v = gmax(failed_field)
    node("vconcat",
         [history_updated, failed_v, special_v],
         name="mem_next")

    zero_stack = constant(np.zeros(STACK - 1))
    initial_os = cat(O0, zero_stack)
    initial_ps = cat(P0, zero_stack)
    huge_v = dense(h0, [[0]], [1000000])
    zero_v = dense(h0, [[0]], [0])
    initial_ts = dense(
        h0, np.zeros((STACK, 1)),
        np.array([1000000] + [0] * (STACK - 1), np.float32)
    )
    very_negative = constant([-1000000])
    jump_progress = constant([0, 0, -16, 16])
    blue_progress = constant([0, 0, -1, 1])
    rail_codes = np.arange(485, 489, dtype=np.float32)

    body = []
    sink = body

    o = sl("plan_os", 0, 1)
    p = sl("plan_ps", 0, 1)
    cutoff = vs("plan_ts", 0, 1)
    depth_positive = minimum("plan_depth", one)
    types, nb, purple, occ, moving, empty = decode(o)
    green = sl(types, 0, 1)
    blue = sl(types, 2, 3)

    o1 = shifts(o, 1, 1)
    o2 = shifts(o, 1, 2)
    middle_occupied = minimum(o1, one)
    destination_empty = relu(sub(one, o2))
    destination_floor = shifts(maximum(F, p), 1, 2)

    legal_jump = mul(
        mul(mul(moving, middle_occupied), destination_empty),
        destination_floor
    )
    legal_jump = mul(legal_jump, domain)
    equal_middle = relu(affine(absolute(sub(o, o1)), -1, 1, 4))
    eats = mul(equal_middle, nb)

    jump_score = add(
        add(affine(eats, 80, 0, 4),
            conv(nb, np.full((4, 1), 20), [10, 12, 10, 14])),
        jump_tie
    )
    jump_score = add(
        jump_score,
        mul(special, add(mul(nb, jump_progress),
                         mul(blue, blue_progress)))
    )

    # Simultaneous front-to-back platform motion.
    p1 = shifts(p)
    visible_base = mul(p, R1)
    edge_probe = mul(mul(mul(p, green), edge_directions), special)
    platform_base = maximum(visible_base, edge_probe)
    free_neighbor = sub(one, p1)
    can_move = mul(platform_base, free_neighbor)
    for _ in range(3):
        can_move = mul(
            platform_base,
            minimum(add(free_neighbor, directional_shift(can_move)), one)
        )

    empty_potential = mul(empty, E)
    loaded_potential = mul(moving, L)
    crate_potential = mul(purple, B)
    current_unweighted = add(
        add(empty_potential, loaded_potential), crate_potential
    )
    current_potential = add(
        add(mul(empty_potential, empty_weight),
            mul(loaded_potential, passenger_weight)),
        crate_potential
    )
    neighbor_potential = add(
        add(mul(mul(empty, E1), empty_weight),
            mul(mul(moving, L1), passenger_weight)),
        mul(purple, B1)
    )
    platform_delta = gmean(
        mul(sub(neighbor_potential, current_potential), can_move)
    )

    rail_score_v = dense(
        platform_delta, np.eye(4) * (3 * 4096),
        np.array([8, 8, 8, 9], np.float32) - rail_codes / 1024
    )
    rail_progress_v = dense(
        gmean(mul(can_move, nb)),
        np.diag([0, 0, -24, 24]).astype(np.float32) * 4096
    )
    rail_score_v = vadd(
        rail_score_v, vmul(rail_progress_v, special_v)
    )
    rail_score = lift(rail_score_v)
    legal_rail = lift(gmax(can_move))

    # An unchanged picture must not prohibit movement along a visible rail:
    # a scrolling camera can keep its passenger at the same screen location.
    only_probe = sub(one, gmax(visible_base))
    root_failures = mul(
        mul(failed_field, only_probe), sub(one, depth_positive)
    )
    legal_rail = mul(legal_rail, sub(one, root_failures))

    def below_cutoff(score):
        return clip(affine(sub(cutoff, score), 2048, -0.5, 4))

    jvalid = mul(legal_jump, below_cutoff(jump_score))
    rvalid = mul(legal_rail, below_cutoff(rail_score))

    def masked_score(score, valid):
        return add(mul(score, valid), affine(valid, 4096, -4096, 4))

    jscore = masked_score(jump_score, jvalid)
    rscore = masked_score(rail_score, rvalid)
    best_score_v = vector_max(vcat(gmax(jscore), gmax(rscore)), 8)

    jchoose = clip(affine(sub(jscore, best_score_v), 2048, 1, 4))
    rchoose = clip(affine(sub(rscore, best_score_v), 2048, 1, 4))
    existence = dense(
        vcat(gmax(jvalid), gmax(rvalid)), np.ones((1, 8))
    )
    has_action = minimum(lift(existence), one)

    # Fixed real history plus the current path's ancestors.
    h = board_hash(o, p)
    all_history = vcat(history_updated, "plan_hs")
    hd = absolute(sub(lift(h), all_history))
    same_hash = relu(sub(one, hd))
    repeated = minimum(
        lift(dense(gmax(same_hash),
                   np.ones((1, HISTORY + STACK)))), one
    )
    repeated = mul(repeated, depth_positive)

    depth_limit = minimum(
        relu(affine("plan_depth", 1, -(STACK - 2))), one
    )
    pruned = maximum(repeated, depth_limit)
    forward = mul(
        mul("plan_active", has_action), sub(one, pruned)
    )
    backward = mul(
        mul("plan_active", sub(one, forward)), depth_positive
    )
    keep = sub(sub(one, forward), backward)
    active_next = mul(
        "plan_active", maximum(forward, depth_positive)
    )

    mobile_nb = mul(nb, minimum(total(legal_jump, 4), one))
    riding_nb = mul(nb, p)
    utility_features = vcat(
        gmean(nb),
        gmean(mobile_nb),
        gmean(riding_nb),
        gmean(mul(current_unweighted, p)),
        gmean(mul(nb, transport_x)),
        gmean(mul(blue, transport_x)),
        gmean("plan_depth")
    )

    # All utility increments are integers. This makes improve exactly
    # zero or one, rather than an interpolation between action codes.
    utilities = dense(
        utility_features,
        np.array([
            [-680 * 4096, 80 * 4096, 80 * 4096,
             4096, 4096, 0, -1],
            [-1400 * 4096, 24 * 4096, 8 * 4096,
             0, 32 * 4096, 2 * 4096, -2]
        ], np.float32)
    )
    utility_v = vadd(
        vmul(vs(utilities, 0, 1), ordinary_v),
        vmul(vs(utilities, 1, 2), special_v)
    )
    cumulative_value = dense(vs("plan_cs", 0, 1), [[4]])
    utility_v = vadd(
        utility_v, vmul(cumulative_value, special_v)
    )
    utility = sub(lift(utility_v), affine(repeated, 160, 0))

    improve = clip(sub(utility, "plan_best"))
    improve = mul(mul(improve, "plan_active"), depth_positive)
    improve = mul(improve, sub(one, mul(repeated, ordinary)))

    best_next = add(
        "plan_best", mul(improve, sub(utility, "plan_best"))
    )
    best_action_next = add(
        "plan_best_action",
        mul(improve, sub("plan_root", "plan_best_action"))
    )

    selected_code_v = dense(
        vcat(gmean(mul(jchoose, action_index)), gmax(rchoose)),
        [([4096] * 4) + rail_codes.tolist()]
    )
    root_update = mul(forward, sub(one, depth_positive))
    root_next = add(
        "plan_root",
        mul(root_update, sub(lift(selected_code_v), "plan_root"))
    )

    # Jump transition.
    source = total(jchoose, 4)
    landing = deposit(jchoose, 2)
    captured = deposit(mul(jchoose, eats), 1)
    moving_type = gmax(mul(o, source))
    jump_delta = sub(
        mul(landing, moving_type),
        mul(o, add(source, captured))
    )

    # Platform transition, carrying its occupant.
    rail_motion = mul(can_move, rchoose)
    rail_source = total(rail_motion, 4)
    rail_destination = deposit(rail_motion, 1)
    new_p = add(p, sub(rail_destination, rail_source))
    carried_destination = deposit(mul(rail_motion, o), 1)
    rail_delta = sub(carried_destination, mul(o, rail_source))
    new_o = add(add(o, jump_delta), rail_delta)

    def update_board_stack(old, new):
        pushed = cat(new, sl(old, 0, STACK - 1))
        popped = cat(sl(old, 1, STACK), zero)
        return add(
            add(mul(pushed, forward), mul(popped, backward)),
            mul(old, keep)
        )

    os_next = update_board_stack("plan_os", new_o)
    ps_next = update_board_stack("plan_ps", new_p)
    fv, bv, kv = gmax(forward), gmax(backward), gmax(keep)

    def update_vector_stack(old, pushed, popped):
        return vadd(
            vadd(vmul(pushed, fv), vmul(popped, bv)),
            vmul(old, kv)
        )

    pushed_t = vcat(
        huge_v, best_score_v, vs("plan_ts", 1, STACK - 1)
    )
    popped_t = vcat(vs("plan_ts", 1, STACK), zero_v)
    ts_next = update_vector_stack("plan_ts", pushed_t, popped_t)

    rail_gain_v = dense(
        vmul(platform_delta, gmax(rchoose)),
        np.ones((1, 4), np.float32) * 4096
    )
    new_cumulative = vadd(vs("plan_cs", 0, 1), rail_gain_v)
    pushed_c = vcat(new_cumulative, vs("plan_cs", 0, STACK - 1))
    popped_c = vcat(vs("plan_cs", 1, STACK), zero_v)
    cs_next = update_vector_stack("plan_cs", pushed_c, popped_c)

    pushed_h = vcat(h, vs("plan_hs", 0, STACK - 1))
    popped_h = vcat(vs("plan_hs", 1, STACK), zero_v)
    hs_next = update_vector_stack("plan_hs", pushed_h, popped_h)
    depth_next = sub(add("plan_depth", forward), backward)

    sink = graph
    planner = {
        "id": "Plan",
        "op": "loop",
        "T": 1,
        "state": [
            {"id": "plan_os", "init": initial_os},
            {"id": "plan_ps", "init": initial_ps},
            {"id": "plan_ts", "init": initial_ts},
            {"id": "plan_cs", "zeros": [STACK]},
            {"id": "plan_hs", "zeros": [STACK]},
            {"id": "plan_depth", "init": zero},
            {"id": "plan_root", "init": zero},
            {"id": "plan_best", "init": very_negative},
            {"id": "plan_best_action", "init": zero},
            {"id": "plan_active", "init": one}
        ],
        "body": body,
        "next": {
            "plan_os": os_next,
            "plan_ps": ps_next,
            "plan_ts": ts_next,
            "plan_cs": cs_next,
            "plan_hs": hs_next,
            "plan_depth": depth_next,
            "plan_root": root_next,
            "plan_best": best_next,
            "plan_best_action": best_action_next,
            "plan_active": active_next
        }
    }
    graph.append(planner)

    # Return a lone loaded ferry from an exhausted right-hand viewport.
    initial_jumps = mul(
        mul(mul(moving0, mid0), empty2),
        shifts(maximum(F, P0), 1, 2)
    )
    any_jump = gmax(channel_max(initial_jumps))
    other_movers = gmax(mul(moving0, sub(one, P0)))
    passenger = gmax(mul(nb0, P0))
    frontier = gmax(mul(maximum(F, R), right_edge))
    visible_rail_directions = gmax(mul(P0, R1))
    probe_failures = mul(
        failed_field, sub(one, visible_rail_directions)
    )

    scan_left = mul(special, passenger)
    scan_left = mul(scan_left, sub(one, other_movers))
    scan_left = mul(scan_left, sub(one, frontier))
    scan_left = mul(scan_left, sub(one, any_jump))
    scan_left = mul(scan_left, sub(one, sl(probe_failures, 2, 3)))
    answer = add(
        "Plan.plan_best_action",
        mul(scan_left,
            sub(constant([487]), "Plan.plan_best_action"))
    )

    # Selection and destination clicks use the recovered original lattice.
    chosen_jump = mul(
        relu(affine(absolute(sub(action_index, answer)), -1, 1, 4)),
        domain
    )
    chosen_source = total(chosen_jump, 4)
    chosen_landing = deposit(chosen_jump, 2)
    pixel_xy = conv(
        "coords",
        np.array([[63, 0, 0], [0, 63, 0]], np.float32)
    )

    def point_to_pixels(point):
        xy = gmean(mul(compact_xy, point))
        xy = dense(xy, np.eye(2) * (6 * 4096))
        return vadd(xy, phase_xy)

    source_xy = point_to_pixels(chosen_source)
    source_distance = total(absolute(sub(pixel_xy, source_xy)), 2)
    source_pixel = relu(affine(source_distance, -1, 1))
    already_selected = gmax(mul(source_pixel, original_selected))
    already_selected = clip(affine(lift(already_selected), 4, -2))

    click_point = add(
        chosen_source,
        mul(already_selected, sub(chosen_landing, chosen_source))
    )
    target_xy = point_to_pixels(click_point)
    conv(absolute(sub(pixel_xy, target_xy)),
         [[-1, -1]], name="click")

    rail_code_field = constant(rail_codes)
    chosen_rail = relu(
        affine(absolute(sub(rail_code_field, answer)), -1, 1, 4)
    )
    output_features = vcat(
        gmax(chosen_rail), gmax(chosen_source), gray_any
    )
    wo = np.zeros((8, 6), np.float32)
    bo = np.full(8, -20, np.float32)
    bo[1:5] = -1
    bo[6] = -1
    bo[7] = 0
    for d in range(4):
        wo[d + 1, d] = 100
    wo[6, 4] = 100
    wo[7, 5] = 500
    dense(output_features, wo, bo, name="logits")

    # Static resource accounting only.
    loop_costs = {}

    def account(nodes, initial_shapes):
        shapes = dict(initial_shapes)
        macs = 0
        evaluations = 0
        spatial_channels = 0
        node_count = 0

        for nd in nodes:
            node_count += 1
            op = nd["op"]

            if op == "loop":
                local = dict(shapes)
                state_channels = 0
                for st in nd["state"]:
                    shape = (
                        shapes[st["init"]] if "init" in st
                        else tuple(st["zeros"])
                    )
                    local[st["id"]] = shape
                    if len(shape) == 3:
                        state_channels += shape[0]

                bm, be, bc, bn, local_out = account(nd["body"], local)
                loop_costs[nd["id"]] = (bm, be)
                macs += nd["T"] * bm
                evaluations += 1 + nd["T"] * be
                node_count += bn
                spatial_channels += state_channels + bc

                for st in nd["state"]:
                    shape = local_out[nd["next"][st["id"]]]
                    shapes[nd["id"] + "." + st["id"]] = shape
                    if len(shape) == 3:
                        spatial_channels += shape[0]
                continue

            evaluations += 1
            ins = [shapes[x] for x in nd["in"]]
            x = ins[0]

            if op == "conv":
                w = params[nd["w"]]
                shape = (w.shape[0], 64, 64)
                macs += int(np.prod(w.shape)) * 4096
            elif op == "dense":
                w = params[nd["w"]]
                shape = (w.shape[0],)
                macs += int(np.prod(w.shape))
            elif op in ("slice", "vslice"):
                shape = (nd["stop"] - nd["start"],) + x[1:]
            elif op in ("concat", "vconcat"):
                shape = (sum(s[0] for s in ins),) + x[1:]
            elif op in ("gmax", "gmean"):
                shape = (x[0],)
                macs += x[0] * 4096
            elif op in ("rowmax", "colmax", "rowsum", "colsum"):
                shape = x
                macs += x[0] * 4096
            elif op in ("add", "sub", "mul", "max2", "min2"):
                shape = (max(s[0] for s in ins), 64, 64)
                macs += shape[0] * 4096
            elif op in ("vadd", "vmul"):
                shape = (max(s[0] for s in ins),)
                macs += shape[0]
            else:
                shape = x

            shapes[nd["id"]] = shape
            if len(shape) == 3:
                spatial_channels += shape[0]

        return macs, evaluations, spatial_channels, node_count, shapes

    initial_shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (MEMORY,),
        "aprev": (8,),
        "avail": (8,)
    }
    m, e, c, n, shapes = account(graph, initial_shapes)
    bm, be = loop_costs["Plan"]
    base_m = m - bm
    base_e = e - be
    planner["T"] = max(
        1,
        min(
            128,
            int((9900 - base_e) // be),
            int((1450000000 - base_m) // max(1, bm))
        )
    )

    assert sum(a.size for a in params.values()) <= 150000
    assert n <= 1000
    assert c + 19 <= 4096
    assert base_e + planner["T"] * be <= 10000
    assert base_m + planner["T"] * bm <= 1500000000
    assert shapes["click"] == (1, 64, 64)
    assert shapes["logits"] == (8,)
    assert shapes["mem_next"] == (MEMORY,)

    return {
        "graph": graph,
        "memory_size": MEMORY,
        "params": params
    }
