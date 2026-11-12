import numpy as np


def build():
    graph, params = [], {}
    ns, ps = [0], [0]

    def name(prefix="n"):
        ns[0] += 1
        return "%s%d" % (prefix, ns[0])

    def par(a):
        ps[0] += 1
        k = "p%d" % ps[0]
        params[k] = np.asarray(a, dtype=np.float32)
        return k

    def node(op, inputs, out=None, **kw):
        out = out or name()
        graph.append({"id": out, "op": op, "in": list(inputs), **kw})
        return out

    def conv(x, w, b=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return node("conv", [x], **kw)

    def linear(x, w, b=None):
        w = np.asarray(w, dtype=np.float32)
        return conv(x, w[:, :, None, None], b)

    def dense(x, w, b=None, out=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return node("dense", [x], out, **kw)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def relu(x):
        return node("relu", [x])

    def vrelu(x):
        return node("vrelu", [x])

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def maximum(a, b):
        return node("max2", [a, b])

    def vadd(a, b):
        return node("vadd", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def gm(x):
        return node("gmax", [x])

    one = sl("coords", 2, 3)
    xy = sl("coords", 0, 2)

    def clip(x):
        return node("min2", [relu(x), one])

    def crisp(x):
        return clip(linear(x, [[2.0]], [-1.0]))

    def vone_minus(x):
        return dense(x, [[-1.0]], [1.0])

    def vsub(a, b):
        return dense(node("vconcat", [a, b]), [[1.0, -1.0]])

    def vcap(x):
        return vone_minus(vrelu(vone_minus(x)))

    def vmax(a, b):
        return vadd(a, vrelu(vsub(b, a)))

    def point_map(pos):
        d = node("abs", [sub(xy, pos)])
        return crisp(relu(linear(d, [[-63.0, -63.0]], [1.0])))

    def equal_pos(a, b):
        d = dense(
            node("vconcat", [a, b]),
            [[1.0, 0.0, -1.0, 0.0],
             [0.0, 1.0, 0.0, -1.0]],
        )
        dn = dense(d, -np.eye(2))
        absolute = vadd(vrelu(d), vrelu(dn))
        distance = dense(absolute, [[63.0, 63.0]])
        return vrelu(vone_minus(distance))

    # One channel for each center on the game's six-pixel lattice.
    points = [(y, x) for y in range(4, 64, 6)
                      for x in range(4, 64, 6)]
    count = len(points)
    w = np.zeros((2 * count, 3), np.float32)
    b = np.zeros(2 * count, np.float32)
    for i, (y, x) in enumerate(points):
        w[2 * i, 0] = 63.0
        w[2 * i + 1, 1] = 63.0
        b[2 * i] = -y
        b[2 * i + 1] = -x
    delta = node("abs", [linear("coords", w, b)])

    w = np.zeros((count, 2 * count), np.float32)
    for i in range(count):
        w[i, 2 * i:2 * i + 2] = -1.0
    spots0 = relu(linear(delta, w, np.ones(count)))
    spots = relu(linear(
        spots0, 2.0 * np.eye(count), -np.ones(count)
    ))
    lattice = clip(linear(spots, np.ones((1, count))))

    c0 = sl("x0", 0, 1)
    c2 = sl("x0", 2, 3)
    c5 = sl("x0", 5, 6)
    c8 = sl("x0", 8, 9)
    c9 = sl("x0", 9, 10)
    c11 = sl("x0", 11, 12)
    c15 = sl("x0", 15, 16)

    yellow7 = conv(c9, np.ones((1, 1, 7, 7)))
    yellow3 = conv(c9, np.ones((1, 1, 3, 3)))
    background7 = conv(c0, np.ones((1, 1, 7, 7)))

    player = mul(
        mul(clip(linear(yellow7, [[1.0]], [-23.0])), c5),
        lattice,
    )
    goal = mul(
        mul(
            mul(clip(linear(yellow7, [[1.0]], [-17.0])),
                clip(linear(yellow3, [[-1.0]], [3.0]))),
            c9,
        ),
        lattice,
    )
    raw_floor = mul(
        clip(linear(background7, [[-1.0]], [13.0])),
        lattice,
    )

    colors = node("concat", [c8, c11, c15])
    k3 = np.zeros((3, 3, 3, 3), np.float32)
    k5 = np.zeros((3, 3, 5, 5), np.float32)
    for i in range(3):
        k3[i, i] = 1.0
        k5[i, i] = 1.0
    n3 = conv(colors, k3)
    n5 = conv(colors, k5)
    small = clip(linear(n3, np.eye(3), [-8.0] * 3))
    not_large = clip(linear(n5, -np.eye(3), [14.0] * 3))
    observed_switch = mul(
        linear(mul(small, not_large), [[1.0, 2.0, 3.0]]),
        lattice,
    )
    large = clip(linear(n5, np.eye(3), [-18.0] * 3))
    gate = crisp(maximum(sl(large, 0, 1), sl(large, 1, 2)))

    # White outlines are teleport pads.
    border = np.ones((1, 1, 7, 7), np.float32)
    border[0, 0, 1:6, 1:6] = 0.0
    border_count = conv(c15, border)
    pads = mul(clip(linear(border_count, [[1.0]], [-23.0])), lattice)
    pad_vector = gm(mul(spots, pads))
    pad_number = dense(pad_vector, np.ones((1, count)), [-1.0])
    has_pair = vcap(vrelu(pad_number))

    # Memory layout:
    # 0:100    known switch types
    # 100:200  claimed or visited switches
    # 200:202  previous player coordinates
    # 202      recorded trace length
    # 203      remaining replay length
    # 204:220  eight failed destination coordinates
    # 220      a pad has been visited on the current trace
    # 221      exploration timer
    # 222:242  ten recent positions
    known_old = vsl("mem", 0, 100)
    claimed_old = vsl("mem", 100, 200)
    previous = vsl("mem", 200, 202)
    trace_old = vsl("mem", 202, 203)
    remaining_old = vsl("mem", 203, 204)
    bad_old = vsl("mem", 204, 220)
    pad_visited_old = vsl("mem", 220, 221)
    explore_old = vsl("mem", 221, 222)
    history_old = vsl("mem", 222, 242)

    diff_w = np.concatenate(
        [-np.eye(100, dtype=np.float32),
         np.eye(100, dtype=np.float32)],
        axis=1,
    )
    diff_param = par(diff_w)

    def vector_max(a, b):
        z = node("vconcat", [a, b])
        d = node("dense", [z], w=diff_param)
        return vadd(a, vrelu(d))

    observed_vector = gm(mul(spots, observed_switch))
    known = vector_max(known_old, observed_vector)
    switch_map = linear(mul(spots, known), np.ones((1, count)))

    def type_mask(value):
        target = linear(one, [[float(value)]])
        d = node("abs", [sub(switch_map, target)])
        return crisp(relu(sub(one, d)))

    blue = type_mask(1)
    purple = type_mask(2)
    white = type_mask(3)
    any_switch = clip(linear(switch_map, [[2.0]], [-1.0]))

    a5 = vsl("aprev", 5, 6)
    not_a5 = vone_minus(a5)
    attempted = dense(
        "aprev",
        [[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
    )
    player_pos = gm(mul(xy, player))

    pos_pair = node("vconcat", [player_pos, previous])
    pd = dense(
        pos_pair,
        [[1.0, 0.0, -1.0, 0.0],
         [0.0, 1.0, 0.0, -1.0]],
    )
    pd_neg = dense(pd, -np.eye(2))
    pd_abs = vadd(vrelu(pd), vrelu(pd_neg))
    changed = vcap(dense(pd_abs, [[100.0, 100.0]]))
    failed = vmul(attempted, vone_minus(changed))
    moved = vmul(attempted, changed)

    step = 6.0 / 63.0
    offset = dense(
        "aprev",
        [[0.0, -step, step, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, -step, step, 0.0, 0.0, 0.0]],
    )
    failed_destination = vadd(previous, offset)
    shifted_bad = node(
        "vconcat", [failed_destination, vsl(bad_old, 0, 14)]
    )
    bad_new = vadd(
        vmul(shifted_bad, failed),
        vmul(bad_old, vone_minus(failed)),
    )
    bad_new = vmul(bad_new, not_a5)

    bad_maps = []
    for i in range(8):
        bad_maps.append(point_map(vsl(bad_new, 2 * i, 2 * i + 2)))
    bad_map = bad_maps[0]
    for m in bad_maps[1:]:
        bad_map = maximum(bad_map, m)

    floor = mul(raw_floor, sub(one, bad_map))
    passable = mul(floor, sub(one, gate))

    trace = vmul(vadd(trace_old, moved), not_a5)
    countdown = vrelu(vsub(remaining_old, moved))
    remaining = vadd(
        vmul(countdown, not_a5),
        vmul(trace_old, a5),
    )
    pending = vcap(remaining)

    prev_point = point_map(previous)
    ghost_claim = vmul(gm(mul(spots, prev_point)), a5)
    purple_visit = gm(mul(spots, mul(player, purple)))
    white_visit = vmul(
        gm(mul(spots, mul(player, white))),
        vone_minus(has_pair),
    )
    claimed = vector_max(claimed_old, ghost_claim)
    claimed = vector_max(claimed, purple_visit)
    claimed = vector_max(claimed, white_visit)

    hud_rows = clip(linear(xy, [[-63.0, 0.0]], [4.0]))
    hud_red = vcap(gm(mul(c2, hud_rows)))
    cycle = vmul(a5, vone_minus(hud_red))
    claimed = vmul(claimed, vone_minus(cycle))

    claim_map = crisp(linear(
        mul(spots, claimed), np.ones((1, count))
    ))
    inactive_claim = mul(claim_map, vone_minus(pending))
    available_switches = mul(any_switch, sub(one, inactive_claim))

    on_pad = gm(mul(player, pads))
    pad_visited = vmax(pad_visited_old, on_pad)
    pad_visited = vmul(pad_visited, not_a5)

    white_claimed = gm(mul(claim_map, white))
    white_exists = gm(white)
    need_prepare = vmul(
        vmul(has_pair, white_exists),
        vmul(vone_minus(pad_visited), vone_minus(white_claimed)),
    )
    # Do not record a white-switch replay before the trace has visited a pad.
    white_block = mul(white, need_prepare)
    switch_targets = mul(available_switches, sub(one, white_block))

    replay_to_pad = vmul(vmul(white_claimed, has_pair), pending)
    active_pads = mul(pads, replay_to_pad)

    # A two-step repeated position starts a short exploration period.
    repeated = vmul(
        equal_pos(player_pos, vsl(history_old, 2, 4)),
        moved,
    )
    timer_decay = vrelu(dense(explore_old, [[1.0]], [-1.0]))
    timer_start = dense(repeated, [[12.0]])
    explore = vmul(vmax(timer_decay, timer_start), not_a5)
    exploring = vcap(explore)
    history = node(
        "vconcat", [player_pos, vsl(history_old, 0, 18)]
    )
    history = vmul(history, not_a5)

    node(
        "vconcat",
        [known, claimed, player_pos, trace, remaining, bad_new,
         pad_visited, explore, history],
        out="mem_next",
    )

    shift1 = np.zeros((4, 1, 7, 7), np.float32)
    shift2 = np.zeros((4, 4, 7, 7), np.float32)
    for i, (dy, dx) in enumerate(
        [(-3, 0), (3, 0), (0, -3), (0, 3)]
    ):
        shift1[i, 0, 3 + dy, 3 + dx] = 1.0
        shift2[i, i, 3 + dy, 3 + dx] = 1.0
    shift1_p = par(shift1)
    shift2_p = par(shift2)
    decay_p = par(np.array([[[[0.98]]]], np.float32))

    def shifted4(x):
        a = node("conv", [x], w=shift1_p)
        return node("conv", [a], w=shift2_p)

    def wave(seed, allowed, tag, teleport=False):
        loop_id = "wave_" + tag
        state = "state_" + tag
        a = "shift_a_" + tag
        b = "shift_b_" + tag
        ds = ["dir_%s_%d" % (tag, i) for i in range(4)]
        m1, m2, m3 = ["merge%d_%s" % (i, tag) for i in range(1, 4)]
        d = "decay_" + tag
        p = "permit_" + tag
        nxt = "updated_" + tag

        body = [
            {"id": a, "op": "conv", "in": [state], "w": shift1_p},
            {"id": b, "op": "conv", "in": [a], "w": shift2_p},
        ]
        for i in range(4):
            body.append({
                "id": ds[i], "op": "slice", "in": [b],
                "start": i, "stop": i + 1,
            })
        body.extend([
            {"id": m1, "op": "max2", "in": [ds[0], ds[1]]},
            {"id": m2, "op": "max2", "in": [ds[2], ds[3]]},
            {"id": m3, "op": "max2", "in": [m1, m2]},
            {"id": d, "op": "conv", "in": [m3], "w": decay_p},
            {"id": p, "op": "mul", "in": [d, allowed]},
        ])
        incoming = p
        if teleport:
            at_pad = "at_pad_" + tag
            best_pad = "best_pad_" + tag
            other_pad = "other_pad_" + tag
            merged = "with_portal_" + tag
            body.extend([
                {"id": at_pad, "op": "mul", "in": [state, active_pads]},
                {"id": best_pad, "op": "gmax", "in": [at_pad]},
                {"id": other_pad, "op": "mul",
                 "in": [active_pads, best_pad]},
                {"id": merged, "op": "max2", "in": [p, other_pad]},
            ])
            incoming = merged
        body.append({"id": nxt, "op": "max2", "in": [state, incoming]})
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 64,
            "state": [{"id": state, "init": seed}],
            "body": body,
            "next": {state: nxt},
        })
        return loop_id + "." + state

    goal_field = wave(goal, passable, "goal", teleport=True)
    switch_field = wave(switch_targets, passable, "switch")
    open_field = wave(goal, floor, "open", teleport=True)
    pad_field = wave(pads, passable, "pad")

    goal_reach = vcap(dense(gm(mul(goal_field, player)), [[100.0]]))
    switch_reach = vcap(dense(gm(mul(switch_field, player)), [[100.0]]))
    pad_reach = vcap(dense(gm(mul(pad_field, player)), [[100.0]]))

    # Pad preparation is used when the pad is actually reachable.
    prepare_now = vmul(need_prepare, pad_reach)
    pad_mode = vmax(prepare_now, replay_to_pad)
    ordinary = vone_minus(pad_mode)
    no_goal = vone_minus(goal_reach)
    use_goal = vmul(ordinary, goal_reach)
    use_switch = vmul(ordinary, vmul(no_goal, switch_reach))
    use_open = vmul(
        ordinary, vmul(no_goal, vone_minus(switch_reach))
    )
    field = add(
        add(mul(goal_field, use_goal),
            mul(switch_field, use_switch)),
        add(mul(open_field, use_open),
            mul(pad_field, pad_mode)),
    )

    neighbor_values = shifted4(field)
    neighbor_allowed = shifted4(passable)
    scores = gm(mul(mul(neighbor_values, neighbor_allowed), player))

    reverse_penalty = dense(
        "aprev",
        [[0.0, 0.0, -0.025, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, -0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, -0.025, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, -0.025, 0.0, 0.0, 0.0, 0.0]],
    )
    scores = vadd(scores, reverse_penalty)

    # Penalize frequently repeated recent cells only after oscillation.
    visit_map = point_map(vsl(history_old, 0, 2))
    for i in range(1, 10):
        visit_map = add(
            visit_map,
            point_map(vsl(history_old, 2 * i, 2 * i + 2)),
        )
    neighbor_visits = shifted4(visit_map)
    visit_counts = gm(mul(neighbor_visits, player))
    penalty = dense(visit_counts, -0.045 * np.eye(4))
    scores = vadd(scores, vmul(penalty, exploring))

    # While standing on a pad during replay, take a legal step away;
    # the pad field brings the player back on the next step.
    wait_cycle = vmul(replay_to_pad, on_pad)
    legal_scores = gm(mul(neighbor_allowed, player))
    waiting_scores = dense(legal_scores, 0.5 * np.eye(4))
    waiting_scores = vadd(waiting_scores, reverse_penalty)
    scores = vadd(
        vmul(scores, vone_minus(wait_cycle)),
        vmul(waiting_scores, wait_cycle),
    )

    rewindable = maximum(blue, mul(white, has_pair))
    unclaimed = mul(rewindable, sub(one, claim_map))
    standing = gm(mul(player, unclaimed))
    rewind = vmul(standing, vcap(trace))

    choices = node("vconcat", [scores, rewind])
    out_w = np.zeros((8, 5), np.float32)
    out_b = np.full(8, -2.0, np.float32)
    for i in range(4):
        out_w[1 + i, i] = 1.0
        out_b[1 + i] = 0.0
    out_w[5, 4] = 2.0
    out_b[5] = 0.0
    dense(choices, out_w, out_b, out="logits")
    node("slice", ["coords"], out="click", start=2, stop=3)

    return {"graph": graph, "memory_size": 242, "params": params}
