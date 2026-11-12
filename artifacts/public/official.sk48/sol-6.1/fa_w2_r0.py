import numpy as np


def build():
    graph = []
    params = {}
    shapes = {
        "x0": (16, True), "coords": (3, True),
        "mem": (14, False), "aprev": (8, False), "avail": (8, False)
    }
    counter = [0]

    def fresh():
        counter[0] += 1
        return "n%d" % counter[0]

    def node(op, inputs, width, spatial=False, name=None, **fields):
        name = fresh() if name is None else name
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(fields)
        graph.append(d)
        shapes[name] = (width, spatial)
        return name

    def conv(x, w, b=None, name=None):
        name = fresh() if name is None else name
        w = np.asarray(w, np.float32)
        params[name + "_w"] = w
        fields = {"w": name + "_w"}
        if b is not None:
            params[name + "_b"] = np.asarray(b, np.float32).reshape(w.shape[0])
            fields["b"] = name + "_b"
        return node("conv", [x], w.shape[0], True, name, **fields)

    def dense(x, w, b=None, name=None):
        name = fresh() if name is None else name
        w = np.asarray(w, np.float32)
        params[name + "_w"] = w
        fields = {"w": name + "_w"}
        if b is not None:
            params[name + "_b"] = np.asarray(b, np.float32).reshape(w.shape[0])
            fields["b"] = name + "_b"
        return node("dense", [x], w.shape[0], False, name, **fields)

    def cat(xs):
        spatial = shapes[xs[0]][1]
        return node(
            "concat" if spatial else "vconcat", xs,
            sum(shapes[x][0] for x in xs), spatial
        )

    def sl(x, start, stop):
        return node(
            "slice" if shapes[x][1] else "vslice",
            [x], stop - start, shapes[x][1], start=start, stop=stop
        )

    def unary(op, x):
        return node(op, [x], shapes[x][0], shapes[x][1])

    def binary(op, a, b):
        return node(
            op, [a, b], max(shapes[a][0], shapes[b][0]),
            shapes[a][1] or shapes[b][1]
        )

    def pool(op, x):
        return node(op, [x], shapes[x][0], False)

    def sa(x, w, b=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b)

    def va(x, w, b=None):
        return dense(x, w, b)

    def V(terms=(), b=0):
        if not terms:
            return dense("aprev", np.zeros((1, 8), np.float32), [b])
        xs = [x for x, coefficient in terms]
        x = cat(xs) if len(xs) > 1 else xs[0]
        weights = []
        for tensor, coefficient in terms:
            weights.extend([coefficient] * shapes[tensor][0])
        return dense(x, [weights], [b])

    def S(terms=(), b=0):
        if not terms:
            return sa("coords", np.zeros((1, 3), np.float32), [b])
        xs = [x for x, coefficient in terms]
        x = cat(xs) if len(xs) > 1 else xs[0]
        weights = []
        for tensor, coefficient in terms:
            weights.extend([coefficient] * shapes[tensor][0])
        return sa(x, [weights], [b])

    def vadd(a, b):
        return binary("vadd", a, b)

    def vmul(a, b):
        return binary("vmul", a, b)

    def smul(a, b):
        return binary("mul", a, b)

    def sadd(a, b):
        return binary("add", a, b)

    def ssub(a, b):
        return binary("sub", a, b)

    def vc(x):
        n = shapes[x][0]
        eye = np.eye(n, dtype=np.float32)
        r = unary("vrelu", x)
        excess = unary("vrelu", va(r, eye, -np.ones(n)))
        return va(cat([r, excess]), np.concatenate([eye, -eye], axis=1))

    def sc(x):
        n = shapes[x][0]
        r = unary("relu", x)
        excess = unary(
            "relu", sa(r, np.eye(n, dtype=np.float32), -np.ones(n))
        )
        return ssub(r, excess)

    def vabs(x):
        return vadd(
            unary("vrelu", x),
            unary("vrelu", va(x, -np.eye(shapes[x][0], dtype=np.float32)))
        )

    def veq(a, b):
        return vc(V([(vabs(V([(a, 1), (b, -1)])), -4)], b=2))

    def vgt(x):
        n = shapes[x][0]
        return vc(va(x, 2 * np.eye(n, dtype=np.float32), -np.ones(n)))

    def vge(x):
        n = shapes[x][0]
        return vc(va(x, 2 * np.eye(n, dtype=np.float32), np.ones(n)))

    def sge(x):
        n = shapes[x][0]
        return sc(sa(x, 2 * np.eye(n, dtype=np.float32), np.ones(n)))

    def seq_zero(x):
        n = shapes[x][0]
        return sc(sa(
            unary("abs", x), -4 * np.eye(n, dtype=np.float32),
            2 * np.ones(n)
        ))

    def complement(x):
        n = shapes[x][0]
        return va(x, -np.eye(n, dtype=np.float32), np.ones(n))

    # Coordinates and play/reference regions.
    row = sa("coords", [[63, 0, 0]])
    col = sa("coords", [[0, 63, 0]])
    world = sc(S([(row, -2)], b=105))
    ui = S([(world, -1)], b=1)

    # Board-cell centre limits from empty interior patches.
    w = np.zeros((1, 16, 5, 5), np.float32)
    w[0, 4] = 1
    core = smul(unary("relu", conv("x0", w, [-24])), world)
    ymin = V([
        (pool("gmax", smul(S([(row, -1)], b=64), core)), -1)
    ], b=64)
    xmin = V([
        (pool("gmax", smul(S([(col, -1)], b=64), core)), -1)
    ], b=64)
    ymax = V([(pool("gmax", smul(row, core)), 1)], b=-1)
    xmax = V([(pool("gmax", smul(col, core)), 1)], b=-1)

    # Frame corners identify rod bases; black inset identifies selection.
    frame_colours = [6, 7, 8, 9, 10, 12, 13, 14, 15]
    corners = [(-2, -2), (-2, 3), (3, -2), (3, 3)]
    bw = np.zeros((len(frame_colours), 16, 7, 7), np.float32)
    aw = np.zeros_like(bw)
    for j, c in enumerate(frame_colours):
        for yy, xx in corners:
            bw[j, c, yy + 3, xx + 3] = 1
            aw[j, c, yy + 3, xx + 3] = 1
        aw[j, 0, 2, 2] = 1
    frames = unary("relu", conv(
        "x0", bw, -3 * np.ones(len(frame_colours))
    ))
    active_frames = unary("relu", conv(
        "x0", aw, -4 * np.ones(len(frame_colours))
    ))
    frame_sum = sa(frames, np.ones((1, len(frame_colours))))
    active_sum = sa(active_frames, np.ones((1, len(frame_colours))))
    base = smul(active_sum, world)
    other_base = smul(unary("relu", ssub(frame_sum, active_sum)), world)
    bx = pool("gmax", smul(col, base))
    by = pool("gmax", smul(row, base))

    # Direction from the striped opening in the selected base.
    directions = []
    for yy, xx in [(0, 3), (3, 0), (0, -2), (-2, 0)]:
        kw = np.zeros((1, 16, 7, 7), np.float32)
        for c in [1, 2, 3]:
            kw[0, c, yy + 3, xx + 3] = 1
        directions.append(pool("gmax", smul(conv("x0", kw), base)))
    right, down, left, up = directions
    horizontal = V([(right, 1), (left, 1)])
    vertical = V([(down, 1), (up, 1)])
    dx = V([(right, 1), (left, -1)])
    dy = V([(down, 1), (up, -1)])

    parallel_field = sadd(smul(col, dx), smul(row, dy))
    perpendicular_field = sadd(smul(row, horizontal), smul(col, vertical))
    base_parallel = vadd(vmul(bx, dx), vmul(by, dy))
    base_perpendicular = vadd(vmul(by, horizontal), vmul(bx, vertical))
    board_last_parallel = V([
        (vmul(xmax, right), 1), (vmul(ymax, down), 1),
        (vmul(xmin, left), -1), (vmul(ymin, up), -1)
    ])
    first_perpendicular = vadd(vmul(ymin, horizontal), vmul(xmin, vertical))
    last_perpendicular = vadd(vmul(ymax, horizontal), vmul(xmax, vertical))

    # Stripe endpoints remain visible around the coloured block interiors.
    patterns = [
        [(0, -2, 2), (1, -2, 1), (0, 3, 1), (1, 3, 2)],
        [(-2, 0, 1), (-2, 1, 2), (3, 0, 2), (3, 1, 1)],
        [(0, -2, 3), (1, -2, 2), (0, 3, 2), (1, 3, 3)],
        [(-2, 0, 2), (-2, 1, 3), (3, 0, 3), (3, 1, 2)]
    ]
    sw = np.zeros((4, 16, 7, 7), np.float32)
    for j, pattern in enumerate(patterns):
        for yy, xx, c in pattern:
            sw[j, c, yy + 3, xx + 3] = 1
    stripes = unary("relu", conv("x0", sw, [-3] * 4))
    lit_h = sl(stripes, 0, 1)
    lit_v = sl(stripes, 1, 2)
    inactive_h = sl(stripes, 2, 3)
    inactive_v = sl(stripes, 3, 4)
    selected_segments = smul(
        sc(S([(lit_h, 1), (lit_v, 1)])), world
    )
    inactive_segments = smul(
        sc(S([(inactive_h, 1), (inactive_v, 1)])), world
    )
    crossing_segments = sadd(
        smul(smul(inactive_v, horizontal), world),
        smul(smul(inactive_h, vertical), world)
    )
    tip_mask = sc(S([(selected_segments, 1), (base, 1)]))
    tip = V([
        (pool("gmax",
              smul(S([(parallel_field, 1)], b=64), tip_mask)), 1)
    ], b=-64)

    # Block-border detectors avoid central match markers.
    colours = [7, 8, 9, 10, 12, 13, 14]
    C = len(colours)
    cw = np.zeros((C, 16, 5, 5), np.float32)
    border = set(
        [(-1, x) for x in range(-1, 3)] +
        [(y, -1) for y in range(-1, 3)]
    )
    for j, c in enumerate(colours):
        for yy, xx in border:
            cw[j, c, yy + 2, xx + 2] = 1
        cw[j, c, 0, 1] -= 7
        cw[j, c, 1, 0] -= 7
    blocks = unary("relu", conv("x0", cw, [-6] * C))
    block_any = sa(blocks, np.ones((1, C)))
    held = smul(block_any, selected_segments)
    reference = smul(smul(block_any, lit_h), ui)
    count_map = S([(held, 1), (reference, 1)])
    attached = pool("gmax", smul(held, crossing_segments))

    # Directional prefix scans recover block ranks along rods.
    scan_input = sa(count_map, np.ones((4, 1)))
    scan_w = np.zeros((4, 4, 3, 3), np.float32)
    for j, (yy, xx) in enumerate([(1, 0), (0, 1), (1, 2), (2, 1)]):
        scan_w[j, j, yy, xx] = 1
    params["scan_w"] = scan_w
    graph.append({
        "id": "scan", "op": "loop", "T": 64,
        "state": [{"id": "scan_state", "zeros": [4, 64, 64]}],
        "body": [
            {
                "id": "scan_shift", "op": "conv",
                "in": ["scan_state"], "w": "scan_w"
            },
            {
                "id": "scan_update", "op": "add",
                "in": ["scan_shift", scan_input]
            }
        ],
        "next": {"scan_state": "scan_update"}
    })
    shapes["scan.scan_state"] = (4, True)
    scans = [sl("scan.scan_state", j, j + 1) for j in range(4)]
    held_rank_raw = S([
        (smul(scans[0], right), 1), (smul(scans[1], down), 1),
        (smul(scans[2], left), 1), (smul(scans[3], up), 1)
    ])
    held_rank = smul(held_rank_raw, held)
    reference_rank = smul(scans[0], reference)

    slots = 8
    rank_pair = cat([held_rank, reference_rank])
    rw = np.zeros((2 * slots, 2), np.float32)
    rb = np.zeros(2 * slots, np.float32)
    for context in range(2):
        for j in range(slots):
            q = context * slots + j
            rw[q, context] = 1
            rb[q] = -(j + 1)
    rank_equal = seq_zero(sa(rank_pair, rw, rb))

    repeat_rank = np.zeros((2 * slots * C, 2 * slots), np.float32)
    repeat_colour = np.zeros((2 * slots * C, C), np.float32)
    for q in range(2 * slots):
        for c in range(C):
            repeat_rank[q * C + c, q] = 1
            repeat_colour[q * C + c, c] = 1
    ordered = pool("gmax", smul(
        sa(rank_equal, repeat_rank), sa(blocks, repeat_colour)
    ))
    held_order = sl(ordered, 0, slots * C)
    goal_order = sl(ordered, slots * C, 2 * slots * C)

    match_w = np.zeros((slots, slots * C), np.float32)
    for j in range(slots):
        match_w[j, j * C:(j + 1) * C] = 1
    matches = va(vmul(held_order, goal_order), match_w)
    prefix_flags = unary("vrelu", va(
        matches, np.tril(np.ones((slots, slots), np.float32)),
        -np.arange(slots, dtype=np.float32)
    ))
    k = va(prefix_flags, np.ones((1, slots), np.float32))
    N = V([(pool("gmean", reference), 4096)])
    complete = vmul(vge(V([(k, 1), (N, -1)])), vgt(N))

    # A matched block at a crossing anchors the current prefix.
    matched_crossing = smul(
        smul(held, crossing_segments),
        sge(ssub(k, held_rank))
    )
    anchor = vmul(vgt(k), pool("gmax", matched_crossing))
    not_anchor = complement(anchor)

    next_w = np.zeros((slots, slots), np.float32)
    for j in range(slots):
        next_w[j, j] = -1
        if j:
            next_w[j, j - 1] = 1
    next_b = np.zeros(slots, np.float32)
    next_b[0] = 1
    next_slot = unary("vrelu", va(prefix_flags, next_w, next_b))
    slot_repeat = np.zeros((slots * C, slots), np.float32)
    colour_reduce = np.zeros((C, slots * C), np.float32)
    for j in range(slots):
        for c in range(C):
            slot_repeat[j * C + c, j] = 1
            colour_reduce[c, j * C + c] = 1
    desired = va(
        vmul(goal_order, va(next_slot, slot_repeat)), colour_reduce
    )
    desired_map = sa(smul(blocks, desired), np.ones((1, C)))
    external = smul(unary("relu", ssub(block_any, held)), world)
    source = smul(
        smul(desired_map, world), S([(held, -1)], b=1)
    )

    # Memory: phase, locked target, old geometry/prefix, discouraged target.
    old_phases = sl("mem", 0, 6)
    old_tp = sl("mem", 6, 7)
    old_tq = sl("mem", 7, 8)
    old_base = sl("mem", 8, 9)
    old_tip = sl("mem", 9, 10)
    old_k = sl("mem", 10, 11)
    ban_p = sl("mem", 11, 12)
    ban_q = sl("mem", 12, 13)
    ban_valid = sl("mem", 13, 14)

    previous_move = dense("aprev", [[0, 1, 1, 1, 1, 0, 0, 0]])
    unchanged = vmul(
        vmul(veq(base_perpendicular, old_base), veq(tip, old_tip)),
        veq(k, old_k)
    )
    stuck = vmul(previous_move, unchanged)
    changed_prefix = complement(veq(k, old_k))

    # Anchored prefixes seek a new block beyond their current tip.
    ordinary_bound = V([(base_parallel, 1), (k, 6)], b=6)
    anchored_bound = V([(tip, 1)], b=6)
    lower_bound = vadd(
        vmul(ordinary_bound, not_anchor),
        vmul(anchored_bound, anchor)
    )
    eligible = smul(source, sge(ssub(parallel_field, lower_bound)))
    eligible_found = pool("gmax", eligible)
    candidates = sadd(
        eligible, smul(source, complement(eligible_found))
    )
    target_found = pool("gmax", candidates)
    ban_match = smul(
        seq_zero(ssub(parallel_field, ban_p)),
        seq_zero(ssub(perpendicular_field, ban_q))
    )
    ban_match = smul(ban_match, ban_valid)
    axial_preference = V([(horizontal, 100), (vertical, -100)])
    target_score = smul(candidates, S([
        (smul(parallel_field, axial_preference), 1),
        (perpendicular_field, 1),
        (ban_match, -50000)
    ], b=100000))
    target_max = pool("gmax", target_score)
    chosen = smul(candidates, seq_zero(ssub(target_score, target_max)))
    dynamic_tp = V([
        (pool("gmax",
              smul(S([(parallel_field, 1)], b=64), chosen)), 1)
    ], b=-64)
    dynamic_tq = pool("gmax", smul(perpendicular_field, chosen))

    # Staging lines lie on the board lattice and avoid external clutter.
    crop = smul(
        smul(
            sge(ssub(row, V([(ymin, 1)], b=-2))),
            sge(ssub(V([(ymax, 1)], b=3), row))
        ),
        smul(
            sge(ssub(col, V([(xmin, 1)], b=-2))),
            sge(ssub(V([(xmax, 1)], b=3), col))
        )
    )
    holes = smul(sl("x0", 5, 6), crop)
    clutter = sc(S([
        (external, 1), (holes, 1), (inactive_segments, 1)
    ]))
    row_clutter = node("rowmax", [clutter], 1, True)
    col_clutter = node("colmax", [clutter], 1, True)
    line_clutter = sadd(
        smul(row_clutter, horizontal), smul(col_clutter, vertical)
    )
    lattice = sadd(
        sa(perpendicular_field, np.ones((8, 1), np.float32),
           -6 * np.arange(8)),
        va(first_perpendicular, -np.ones((8, 1), np.float32))
    )
    lattice = seq_zero(lattice)
    line_valid = vge(va(
        V([(last_perpendicular, 1), (first_perpendicular, -1)]),
        np.ones((8, 1), np.float32), -6 * np.arange(8)
    ))
    line_candidates = sa(
        smul(lattice, line_valid), np.ones((1, 8), np.float32)
    )
    staging_distance = unary(
        "abs", ssub(perpendicular_field, base_perpendicular)
    )
    workspace_score = smul(line_candidates, S([
        (smul(perpendicular_field, horizontal), 100),
        (smul(perpendicular_field, vertical), 1),
        (smul(staging_distance, vertical), -100),
        (line_clutter, -10000)
    ], b=100000))
    workspace_max = pool("gmax", workspace_score)
    workspace_mask = smul(
        line_candidates, seq_zero(ssub(workspace_score, workspace_max))
    )
    workspace = pool("gmax", smul(perpendicular_field, workspace_mask))

    # Ordinary phases: detach, pack, stage, set length, align, capture.
    init_w = np.eye(6, dtype=np.float32)
    init_w[0] -= 1
    init_b = np.zeros(6, np.float32)
    init_b[0] = 1
    phases = va(old_phases, init_w, init_b)
    approach_phase = va(phases, [[0, 0, 0, 0, 1, 1]])
    no_source_phase = vmul(
        va(phases, [[0, 0, 0, 1, 1, 1]]),
        complement(target_found)
    )
    at_end = vge(V([(tip, 1), (board_last_parallel, -1)]))
    abort = vmul(vc(V([
        (vmul(stuck, approach_phase), 1),
        (vmul(at_end, sl(phases, 5, 6)), 1),
        (no_source_phase, 1)
    ])), not_anchor)
    restart = vc(V([(abort, 1), (changed_prefix, 1)]))
    phases = vadd(
        vmul(phases, complement(restart)),
        va(restart, [[1], [0], [0], [0], [0], [0]])
    )
    pf = [sl(phases, j, j + 1) for j in range(6)]

    locked = V([(pf[4], 1), (pf[5], 1)])
    locked = vmul(locked, not_anchor)
    tp = vadd(vmul(old_tp, locked),
              vmul(dynamic_tp, complement(locked)))
    tq = vadd(vmul(old_tq, locked),
              vmul(dynamic_tq, complement(locked)))

    workspace_delta = V([(workspace, 1), (base_perpendicular, -1)])
    workspace_done = veq(workspace, base_perpendicular)
    detach_done = complement(attached)
    pack_retract = vgt(V([
        (tip, 1), (base_parallel, -1), (k, -6)
    ]))
    desired_tip = V([(tp, 1)], b=-6)
    pre_extend = vgt(V([(desired_tip, 1), (tip, -1)]))
    pre_retract = vgt(V([(tip, 1), (desired_tip, -1)]))
    pre_done = V([(pre_extend, -1), (pre_retract, -1)], b=1)
    align_delta = V([(tq, 1), (base_perpendicular, -1)])
    align_done = veq(tq, base_perpendicular)

    # Advance through satisfied phases combinationally.
    f0 = pf[0]
    d0 = vmul(f0, detach_done)
    a0 = vmul(f0, complement(detach_done))
    f1 = vadd(pf[1], d0)
    d1 = vmul(f1, complement(pack_retract))
    a1 = vmul(f1, pack_retract)
    f2 = vadd(pf[2], d1)
    d2 = vmul(f2, workspace_done)
    a2 = vmul(f2, complement(workspace_done))
    f3 = vadd(pf[3], d2)
    d3 = vmul(f3, pre_done)
    a3e = vmul(f3, pre_extend)
    a3r = vmul(f3, pre_retract)
    f4 = vadd(pf[4], d3)
    d4 = vmul(f4, align_done)
    a4 = vmul(f4, complement(align_done))
    f5 = vadd(pf[5], d4)

    extension = V([(a3e, 1), (f5, 1)])
    retraction = V([(a1, 1), (a3r, 1)])
    workspace_action = V([(a0, 1), (a2, 1)])
    perpendicular_positive = V([
        (vmul(workspace_action, vgt(workspace_delta)), 1),
        (vmul(a4, vgt(align_delta)), 1)
    ])
    perpendicular_negative = V([
        (vmul(workspace_action, vgt(V([(workspace_delta, -1)]))), 1),
        (vmul(a4, vgt(V([(align_delta, -1)]))), 1)
    ])

    # Shorten after a failed ordinary transverse staging move.
    shorten_fallback = vmul(stuck, workspace_action)
    keep_action = complement(shorten_fallback)
    extension = vmul(extension, keep_action)
    retraction = vadd(vmul(retraction, keep_action), shorten_fallback)
    perpendicular_positive = vmul(perpendicular_positive, keep_action)
    perpendicular_negative = vmul(perpendicular_negative, keep_action)

    # Anchored mode slides along the crossing rod, then extends without
    # retracting the shared correctly matched block.
    active_anchor = vmul(anchor, target_found)
    normal_control = complement(active_anchor)
    extension = vadd(
        vmul(extension, normal_control),
        vmul(active_anchor, align_done)
    )
    retraction = vmul(retraction, normal_control)
    perpendicular_positive = vadd(
        vmul(perpendicular_positive, normal_control),
        vmul(active_anchor, vgt(align_delta))
    )
    perpendicular_negative = vadd(
        vmul(perpendicular_negative, normal_control),
        vmul(active_anchor, vgt(V([(align_delta, -1)])))
    )

    click_action = vmul(complete, pool("gmax", other_base))
    moving = complement(click_action)
    action_features = cat([
        vmul(extension, right), vmul(extension, down),
        vmul(extension, left), vmul(extension, up),
        vmul(retraction, right), vmul(retraction, down),
        vmul(retraction, left), vmul(retraction, up),
        vmul(perpendicular_positive, horizontal),
        vmul(perpendicular_positive, vertical),
        vmul(perpendicular_negative, horizontal),
        vmul(perpendicular_negative, vertical),
        click_action
    ])
    action_w = np.zeros((8, 13), np.float32)
    for action, columns in {
        1: [3, 5, 10], 2: [1, 7, 8],
        3: [2, 4, 11], 4: [0, 6, 9]
    }.items():
        action_w[action, columns] = 20
    action_w[6, 12] = 50
    dense(
        action_features, action_w,
        [-100, 0, 0, 0, .01, -100, -10, -100],
        name="logits"
    )
    conv(other_base, np.ones((1, 1, 1, 1), np.float32), name="click")

    next_phase = cat([
        V([(f0, 1), (d0, -1)]),
        V([(f1, 1), (d1, -1)]),
        V([(f2, 1), (d2, -1)]),
        V([(f3, 1), (d3, -1)]),
        V([(f4, 1), (d4, -1)]),
        f5
    ])
    next_phase = vmul(vmul(next_phase, moving), normal_control)
    remember_target = vc(V([(f3, 1), (f4, 1), (f5, 1)]))
    next_tp = vadd(
        vmul(tp, remember_target),
        vmul(old_tp, complement(remember_target))
    )
    next_tq = vadd(
        vmul(tq, remember_target),
        vmul(old_tq, complement(remember_target))
    )
    failed_target = vmul(abort, approach_phase)
    next_ban_p = vadd(
        vmul(old_tp, failed_target),
        vmul(ban_p, complement(failed_target))
    )
    next_ban_q = vadd(
        vmul(old_tq, failed_target),
        vmul(ban_q, complement(failed_target))
    )
    next_ban_valid = vc(V([(ban_valid, 1), (failed_target, 1)]))
    next_ban_valid = vmul(
        vmul(next_ban_valid, complement(changed_prefix)), moving
    )
    memory_out = cat([
        next_phase, next_tp, next_tq,
        base_perpendicular, tip, k,
        next_ban_p, next_ban_q, next_ban_valid
    ])
    dense(memory_out, np.eye(14, dtype=np.float32), name="mem_next")

    return {"graph": graph, "memory_size": 14, "params": params}
