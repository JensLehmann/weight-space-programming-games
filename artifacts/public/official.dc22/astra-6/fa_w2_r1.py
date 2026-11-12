import numpy as np


def build():
    P = {}
    graph = []
    active = graph
    serial = [0]
    pcache = {}

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in pcache:
            name = "p%d" % len(P)
            P[name] = a
            pcache[key] = name
        return pcache[key]

    def node(op, ins=None, name=None, **kw):
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        d = {"id": name, "op": op}
        if ins is not None:
            d["in"] = list(ins)
        d.update(kw)
        active.append(d)
        return name

    def conv(x, w, b=None, name=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("conv", [x], name=name, **kw)

    def lin(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        return conv(x, w[:, :, None, None], b, name)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("dense", [x], name=name, **kw)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def cat(*xs):
        return node("concat", xs)

    def vc(*xs):
        return node("vconcat", xs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def mn(a, b):
        return node("min2", [a, b])

    def relu(x):
        return node("relu", [x])

    def vr(x):
        return node("vrelu", [x])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def gm(x):
        return node("gmax", [x])

    def affine(x, a, b=0.0):
        return lin(x, [[a]], [b])

    def vaff(x, a, b=0.0):
        return dense(x, [[a]], [b])

    def negv(x, n):
        return dense(x, -np.eye(n, dtype=np.float32))

    def vmax(a, b, n):
        return va(vr(va(a, negv(b, n))), b)

    one = lin("coords", [[0, 0, 1]])
    zero4 = lin(one, np.zeros((4, 1), np.float32))
    pix = lin("coords", [[63, 0, 0], [0, 63, 0]])
    row = sl(pix, 0, 1)
    col = sl(pix, 1, 2)

    def clip(x):
        return mn(relu(x), one)

    def inv(x):
        return sub(one, x)

    def vclip(x, n=1):
        z = vr(x)
        excess = vr(dense(z, np.eye(n), -np.ones(n)))
        return va(z, negv(excess, n))

    def vinv(x):
        return vaff(x, -1, 1)

    def shift(x, dy, dx):
        while dy or dx:
            sy = max(-3, min(3, dy))
            sx = max(-3, min(3, dx))
            r = max(abs(sy), abs(sx))
            w = np.zeros((1, 1, 2 * r + 1, 2 * r + 1), np.float32)
            w[0, 0, r + sy, r + sx] = 1
            x = conv(x, w)
            dy -= sy
            dx -= sx
        return x

    def stencil(x, offsets, bias=0):
        r = max(max(abs(y), abs(z)) for y, z, _ in offsets)
        w = np.zeros((1, 1, 2 * r + 1, 2 * r + 1), np.float32)
        for y, z, v in offsets:
            w[0, 0, r + y, r + z] += v
        return conv(x, w, [bias])

    def spread(x, h, w):
        return stencil(
            x, [(-y, -z, 1) for y in range(h) for z in range(w)]
        )

    def erode2(x, channels=1):
        w = np.zeros((channels, channels, 3, 3), np.float32)
        for c in range(channels):
            w[c, c, 1:3, 1:3] = 1
        return clip(conv(x, w, np.full(channels, -3, np.float32)))

    def max_channels(x, n):
        while n > 1:
            m = n // 2
            z = mx(sl(x, 0, m), sl(x, m, 2 * m))
            if n % 2:
                z = cat(z, sl(x, 2 * m, n))
            x = z
            n = m + n % 2
        return x

    def sample(x, marker):
        return gm(mul(x, marker))

    def sample_signed(x, marker):
        return gm(add(x, affine(marker, 10000, -10000)))

    def min_coords(mask):
        penalty = affine(mask, 1000, -1000)
        q = add(lin(pix, -np.eye(2)), penalty)
        return dense(gm(q), -np.eye(2))

    def coords_at(mask):
        return gm(mul(pix, mask))

    # Separate the board, sidebar, and bottom status bar.
    raw = [sl("x0", c, c + 1) for c in range(16)]
    black_columns = node("colsum", [raw[5]])
    panel = clip(affine(black_columns, 1, -15))
    board = mul(inv(panel), clip(affine(row, -1, 63)))
    ui = [mul(x, panel) for x in raw]
    observed = [mul(x, board) for x in raw]

    player = relu(stencil(
        raw[14],
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
        -3
    ))

    # Infer the material covered by the 2x2 player.
    perimeter = [(-1, 0), (-1, 1), (2, 0), (2, 1),
                 (0, -1), (1, -1), (0, 2), (1, 2)]
    w = np.zeros((16, 16, 5, 5), np.float32)
    for c in range(16):
        for y, x in perimeter:
            w[c, c, y + 2, x + 2] = 1
    counts = sample(conv("x0", w), player)

    corners = [
        [(-1, 0), (-1, 1), (0, -1), (1, -1), (-1, -1)],
        [(-1, 0), (-1, 1), (0, 2), (1, 2), (-1, 2)],
        [(2, 0), (2, 1), (0, -1), (1, -1), (2, -1)],
        [(2, 0), (2, 1), (0, 2), (1, 2), (2, 2)],
    ]
    w = np.zeros((64, 16, 5, 5), np.float32)
    for q, offsets in enumerate(corners):
        for c in range(16):
            for y, x in offsets:
                w[q * 16 + c, c, y + 2, x + 2] = 1
    corner_hits = relu(conv("x0", w, -4 * np.ones(64)))
    corner_hits = mx(
        mx(sl(corner_hits, 0, 16), sl(corner_hits, 16, 32)),
        mx(sl(corner_hits, 32, 48), sl(corner_hits, 48, 64))
    )
    corner_evidence = sample(corner_hits, player)

    material_order = [7, 6, 11, 1, 15, 9, 8, 12, 13, 2]
    w = np.zeros((16, 32), np.float32)
    b = np.full(16, -100, np.float32)
    for i, c in enumerate(material_order):
        w[c, c] = 1
        w[c, c + 16] = 32
        b[c] = 0.02 * (i + 1)
    material_scores = dense(vc(counts, corner_evidence), w, b)
    material_maps = mul(one, material_scores)
    best_material = max_channels(material_maps, 16)
    fill_flags = gm(relu(lin(
        sub(material_maps, best_material),
        1024 * np.eye(16), np.ones(16)
    )))
    filled = clip(add("x0", mul(raw[14], fill_flags)))
    colors = [mul(sl(filled, c, c + 1), board) for c in range(16)]

    # Detect hat-shaped toggle switches.
    hat_offsets = set((y, 0) for y in range(-2, 3))
    hat_offsets.update((0, x) for x in range(-3, 4))
    hats = {}
    present = {}
    for color in (8, 9, 6, 15):
        h = relu(stencil(
            ui[color],
            [(y, x, 1) for y, x in hat_offsets],
            1 - len(hat_offsets)
        ))
        hats[color] = clip(h)
        present[color] = gm(hats[color])
    present_e = gm(ui[2])
    present_f = gm(ui[11])

    # Key position persists through occlusion until its switch is unlocked.
    kp = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3)]
    kn = [(0, 0), (0, 3), (-1, 1), (-1, 2)]
    key_marker = relu(stencil(
        observed[15],
        [(y, x, 1) for y, x in kp] +
        [(y, x, -1) for y, x in kn],
        -5
    ))
    old_key_position = vs("mem", 19, 21)
    old_key_flag = vs("mem", 21, 22)
    key_distance = node("abs", [sub(pix, old_key_position)])
    remembered_key = relu(lin(key_distance, [[-2, -2]], [1]))
    key_marker = mx(key_marker, mul(remembered_key, old_key_flag))
    key_position = coords_at(key_marker)
    key_box = clip(spread(key_marker, 2, 4))
    key_exists = gm(key_marker)
    seek_key = vm(key_exists, vinv(present[15]))
    goal = clip(add(
        mul(key_box, seek_key),
        mul(observed[11], vinv(seek_key))
    ))

    # Detect and remember the two teleport endpoints.
    portal_w = np.zeros((2, 16, 3, 3), np.float32)
    for q, pattern in enumerate(((7, 6, 6, 7), (6, 7, 7, 6))):
        for (y, x), c in zip(
            ((0, 0), (0, 1), (1, 0), (1, 1)), pattern
        ):
            portal_w[q, c, y + 1, x + 1] = 1
    portal_now = relu(conv("x0", portal_w, [-3, -3]))
    portal_now = mul(lin(portal_now, [[1, 1]]), board)

    old_portal_flag = vs("mem", 4, 5)
    portal = portal_now
    for a in (0, 2):
        old_pos = vs("mem", a, a + 2)
        distance = node("abs", [sub(pix, old_pos)])
        marker = relu(lin(distance, [[-2, -2]], [1]))
        portal = mx(portal, mul(marker, old_portal_flag))
    portal = clip(portal)

    rank = lin(pix, [[64, 1]])
    selected = []
    for sign in (-1, 1):
        ranking = add(
            affine(rank, sign), affine(portal, 10000, -10000)
        )
        peak = gm(ranking)
        delta = node("abs", [sub(ranking, peak)])
        marker = mul(portal, relu(affine(delta, -2, 1)))
        selected.append(coords_at(marker))
    portal_memory = vc(selected[0], selected[1], gm(portal))
    portal_floor = clip(spread(portal, 2, 2))

    # Distinguish solid paths from their checkerboard alternatives.
    def gate_pair(c):
        neighbors = stencil(c, [
            (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)
        ])
        solid = clip(spread(erode2(c), 2, 2))
        cover = mx(c, clip(affine(neighbors, 1, -1)))
        alternate = relu(sub(cover, solid))
        return solid, alternate

    b0, b1 = gate_pair(colors[9])
    purple = relu(sub(colors[15], key_box))
    d0, d1 = gate_pair(purple)

    # Detect rotating bridge pivots and transpose their arms.
    pivot = colors[13]
    pivot_anchor = relu(stencil(
        pivot,
        [(y, x, 1) for y in range(4) for x in range(4)],
        -15
    ))
    near_pivot = pivot
    cross = [(0, x, 1) for x in range(-3, 4)]
    cross += [(y, 0, 1) for y in range(-3, 4) if y]
    for _ in range(3):
        near_pivot = clip(stencil(near_pivot, cross))
    a0 = mul(colors[8], near_pivot)
    a1 = lin(one, [[0]])
    for distance in (-8, -6, -4, -2, 4, 6, 8, 10):
        h = mul(pivot_anchor, shift(colors[8], 0, distance))
        h = shift(h, -distance, 0)
        h = clip(spread(h, 2, 4))
        v = mul(pivot_anchor, shift(colors[8], distance, 0))
        v = shift(v, 0, -distance)
        v = clip(spread(v, 4, 2))
        a1 = mx(a1, mx(h, v))

    # Detect the tall movable object's rectangular core.
    tall_seed = relu(stencil(
        colors[12], [(y, 0, 1) for y in range(-2, 3)], -4
    ))
    object_raw = clip(stencil(
        tall_seed, [(y, 0, 1) for y in range(-2, 3)]
    ))

    # Exclude the complete object, including its two-pixel left protrusion,
    # from thin-bridge detection. Keep object_raw unchanged for pose geometry.
    object_envelope = mx(object_raw, shift(object_raw, 0, 2))
    thin12 = relu(sub(colors[12], object_envelope))

    e_black = observed[5]
    wide12 = relu(stencil(
        thin12, [(0, x, 1) for x in range(-3, 4)], -6
    ))
    e_rows = node("rowmax", [mx(e_black, wide12)])
    e_current = clip(mul(add(colors[1], thin12), e_rows))
    e_potential = clip(add(e_black, e_current))
    f_current = clip(mul(add(colors[1], thin12), inv(e_rows)))
    f_last = gm(mul(thin12, inv(e_rows)))

    e_exists = gm(e_potential)
    e_min = vs(min_coords(e_potential), 1, 2)
    f_min = vs(min_coords(f_current), 1, 2)
    f_min = va(f_min, vaff(f_last, -10))
    f_origin = va(vm(e_min, e_exists), vm(f_min, vinv(e_exists)))
    f_center = vaff(f_origin, 1, 7.5)
    f_dist = node("abs", [sub(col, f_center)])
    f_span = clip(affine(f_dist, -2, 16))
    f_rows = node("rowmax", [f_current])
    f_potential = mul(f_rows, f_span)
    e_potential = mul(e_potential, present_e)
    f_potential = mul(f_potential, present_f)

    # Track extension versus retraction of the expanding bridge.
    e_count = dense(node("gmean", [e_current]), [[4096]])
    old_count = vs("mem", 17, 18)
    old_growing = vs("mem", 18, 19)
    delta = va(e_count, vaff(old_count, -1))
    increasing = vclip(delta)
    decreasing = vclip(vaff(delta, -1))
    unchanged = vinv(va(increasing, decreasing))
    full_e = vclip(vaff(e_count, 1, -31))
    empty_e = vclip(vaff(e_count, -1, 1))
    growing = va(increasing, vm(unchanged, old_growing))
    growing = va(vm(growing, vinv(full_e)), empty_e)
    growing = vclip(growing)

    ep_l = shift(e_potential, 0, -2)
    ep_r = shift(e_potential, 0, 2)
    ec_l = shift(e_current, 0, -2)
    ec_r = shift(e_current, 0, 2)
    e_edges = mul(e_potential, inv(mul(ep_l, ep_r)))
    e_grow = mul(
        e_potential, mx(e_edges, mx(ec_l, mx(ec_r, e_current)))
    )
    keep_l = mx(ec_l, inv(ep_l))
    keep_r = mx(ec_r, inv(ep_r))
    e_shrink = mul(e_current, mul(keep_l, keep_r))
    outer6 = mul(e_potential, inv(mul(
        shift(e_potential, 0, -6), shift(e_potential, 0, 6)
    )))
    e_shrink = add(
        mul(e_shrink, vinv(full_e)), mul(outer6, full_e)
    )
    e_future = add(
        mul(e_grow, growing), mul(e_shrink, vinv(growing))
    )
    f_future = add(
        mul(shift(f_current, 0, -2), vinv(f_last)),
        mul(shift(f_current, 0, 10), f_last)
    )

    permanent = colors[2]
    for x in (
        colors[6], colors[7], observed[11], pivot, key_box, portal_floor
    ):
        permanent = mx(permanent, x)
    permanent = mul(clip(permanent), board)

    # Locate the crane from the invariant part of its silhouette.
    cp = [(2, x) for x in range(1, 5)]
    cp += [(5, x) for x in range(1, 5)]
    cp += [(y, x) for y in (3, 4) for x in range(4)]
    cp += [(1, 3), (6, 3)]
    cn = [(3, 4), (4, 4), (3, 5), (4, 5), (2, 0), (5, 0)]
    crane_marker = relu(stencil(
        observed[8],
        [(y - 3, x - 3, 1) for y, x in cp] +
        [(y - 3, x - 3, -1) for y, x in cn],
        1 - len(cp)
    ))
    crane_marker = clip(crane_marker)
    has_crane = gm(crane_marker)
    has_object = gm(object_raw)
    closed = sample(shift(observed[8], -2, 1), crane_marker)
    crane_anchor = dense(coords_at(crane_marker), np.eye(2), [1, 6])
    object_anchor = dense(min_coords(object_raw), np.eye(2), [6, 1])
    rail = dense(min_coords(observed[0]), np.eye(2), [2, 8])

    poses = [
        (0, 12), (0, 8), (0, 4), (0, 0), (4, 0),
        (8, 0), (12, 0), (12, 4), (12, 8), (12, 12)
    ]

    def pose_onehot(anchor):
        w = np.zeros((20, 4), np.float32)
        b = np.zeros(20, np.float32)
        for i, (dy, dx) in enumerate(poses):
            w[2 * i, 0] = 1
            w[2 * i, 2] = -1
            w[2 * i + 1, 1] = 1
            w[2 * i + 1, 3] = -1
            b[2 * i:2 * i + 2] = [-dy, -dx]
        z = dense(vc(anchor, rail), w, b)
        absolute = va(vr(z), vr(negv(z, 20)))
        w2 = np.zeros((10, 20), np.float32)
        for i in range(10):
            w2[i, 2 * i:2 * i + 2] = -2
        return vr(dense(absolute, w2, np.ones(10)))

    object_pose = pose_onehot(object_anchor)
    crane_pose = pose_onehot(crane_anchor)
    op_exists = dense(object_pose, np.ones((1, 10)))
    cp_exists = dense(crane_pose, np.ones((1, 10)))
    crane_mode = vm(
        vm(has_crane, has_object), vm(op_exists, cp_exists)
    )
    coincident = dense(vm(object_pose, crane_pose), np.ones((1, 10)))
    attached = vm(closed, coincident)

    relative = sub(pix, rail)
    pose_masks = []
    for dy, dx in poses:
        centered = lin(
            relative, np.eye(2), [-dy + 0.5, -dx - 0.5]
        )
        absolute = node("abs", [centered])
        rectangle_axes = clip(lin(
            absolute, -2 * np.eye(2), [12, 4]
        ))
        pose_masks.append(mn(
            sl(rectangle_axes, 0, 1), sl(rectangle_axes, 1, 2)
        ))

    # Eight switch configurations plus forty object/gate configurations.
    terrain = cat(
        permanent, a0, a1, b0, b1, d0, d1,
        e_potential, f_potential, object_raw, *pose_masks
    )
    wt = np.zeros((48, 20), np.float32)
    for s in range(48):
        wt[s, 0] = 1
        wt[s, 7:9] = 1
        if s < 8:
            wt[s, 1 + ((s >> 2) & 1)] = 1
            wt[s, 3 + (s & 1)] = 1
            wt[s, 5 + ((s >> 1) & 1)] = 1
            wt[s, 9] = 1
        else:
            q = s - 8
            wt[s, 1] = 1
            wt[s, 3 + (q & 1)] = 1
            wt[s, 5 + ((q >> 1) & 1)] = 1
            wt[s, 10 + q // 4] = 1
    raw_valid = clip(lin(terrain, wt))
    valid = erode2(raw_valid, 48)

    wa = np.zeros((48, 4), np.float32)
    ba = np.ones(48, np.float32)
    for s in range(48):
        bits = s if s < 8 else (s - 8) % 4
        if bits & 1:
            wa[s, 1] = 1
            ba[s] -= 1
        if bits & 2:
            wa[s, 2] = 1
            ba[s] -= 1
        if s < 8 and bits & 4:
            wa[s, 0] = 1
            ba[s] -= 1
        if s >= 8:
            wa[s, 3] = 1
            ba[s] -= 1
    allow = vr(dense(
        vc(present[8], present[9], present[15], crane_mode),
        wa, ba
    ))
    valid = mul(valid, allow)
    initial_reach = mul(valid, goal)

    # Accumulated Boolean wavefronts yield distance-to-target scores.
    body = []
    active = body
    chunks = []
    move_kernel = np.zeros((4, 4, 3, 3), np.float32)
    for s in range(4):
        for y, x in ((0, 1), (2, 1), (1, 0), (1, 2), (1, 1)):
            move_kernel[s, s, y, x] = 1
        move_kernel[s, s ^ 1, 1, 1] = 1
        move_kernel[s, s ^ 2, 1, 1] = 1
    for i in range(12):
        chunks.append(conv(
            sl("reach_state", 4 * i, 4 * i + 4), move_kernel
        ))
    reached_neighbors = cat(*chunks)

    a_edges = cat(
        sl("reach_state", 4, 8), sl("reach_state", 0, 4)
    )
    previous_pose = cat(zero4, sl("reach_state", 8, 44))
    next_pose = cat(sl("reach_state", 12, 48), zero4)
    crane_edges = add(previous_pose, next_pose)
    configuration_edges = cat(a_edges, crane_edges)
    total = add(reached_neighbors, configuration_edges)
    portal_reach = gm(mul("reach_state", portal))
    total = add(total, mul(portal, portal_reach))
    total = add(total, goal)
    reach_next = mul(mn(total, one), valid)
    sum_next = add("distance_state", reach_next)
    active = graph
    graph.append({
        "id": "plan",
        "op": "loop",
        "T": 128,
        "state": [
            {"id": "reach_state", "init": initial_reach},
            {"id": "distance_state", "zeros": [48, 64, 64]},
        ],
        "body": body,
        "next": {
            "reach_state": reach_next,
            "distance_state": sum_next
        },
    })
    field = add("plan.distance_state", initial_reach)

    current_pose_weights = vm(object_pose, crane_mode)
    wc = np.zeros((48, 11), np.float32)
    bc = np.zeros(48, np.float32)
    wc[0, 0] = -1
    bc[0] = 1
    for i in range(10):
        wc[8 + 4 * i, i + 1] = 1
    current_weights = dense(
        vc(crane_mode, current_pose_weights), wc, bc
    )
    player_values = sample(field, player)
    player_validity = sample(valid, player)

    def evaluate_weights(weights):
        val = dense(
            vm(weights, player_values), np.ones((1, 48))
        )
        safe = dense(
            vm(weights, player_validity), np.ones((1, 48))
        )
        return va(val, vaff(safe, 1000, -1000))

    def next_weights(kind):
        w = np.zeros((48, 48), np.float32)
        for s in range(48):
            target = None
            if kind == "b":
                target = s ^ 1
            elif kind == "d":
                target = s ^ 2
            elif kind == "a" and s < 8:
                target = s ^ 4
            elif kind in ("prev", "next") and s >= 8:
                pose = (s - 8) // 4
                bits = (s - 8) % 4
                p = pose + (-1 if kind == "prev" else 1)
                if 0 <= p < 10:
                    target = 8 + 4 * p + bits
            if target is not None:
                w[target, s] = 1
        return dense(current_weights, w)

    qa = evaluate_weights(next_weights("a"))
    qb = evaluate_weights(next_weights("b"))
    qd = evaluate_weights(next_weights("d"))
    qp = evaluate_weights(next_weights("prev"))
    qn = evaluate_weights(next_weights("next"))

    selected_field = lin(
        mul(field, current_weights), np.ones((1, 48))
    )
    teleport_value = gm(mul(selected_field, portal))
    on_portal = sample(portal, player)
    qc = va(
        teleport_value,
        vaff(vm(on_portal, present[6]), 1000, -1000)
    )

    # Actual support constrains the optimistic thin-bridge planner.
    actual_raw = permanent
    for x in (a0, b0, d0, e_current, f_current, object_raw):
        actual_raw = mx(actual_raw, x)
    actual_raw = clip(actual_raw)
    actual = erode2(actual_raw)

    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    sw = np.zeros((4, 1, 5, 5), np.float32)
    for i, (y, x) in enumerate(directions):
        sw[i, 0, 2 + y, 2 + x] = 1
    neighboring_values = conv(selected_field, sw)
    neighboring_actual = conv(actual, sw)
    movement = sample_signed(add(
        neighboring_values,
        lin(neighboring_actual, 1000 * np.eye(4),
            -1000 * np.ones(4))
    ), player)

    def preparation_score(potential, current, future):
        candidate_locations = mul(
            conv(potential, sw), sub(one, neighboring_actual)
        )
        candidate_locations = clip(candidate_locations)
        candidate_score = add(
            neighboring_values,
            lin(candidate_locations, 1000 * np.eye(4),
                -1000 * np.ones(4))
        )
        candidate_score = add(
            candidate_score, affine(player, 1000, -1000)
        )
        q = gm(max_channels(candidate_score, 4))
        without = relu(sub(actual_raw, current))
        after = erode2(clip(add(without, future)))
        safe = sample(after, player)
        return va(q, vaff(safe, 1000, -1000))

    qe = preparation_score(e_potential, e_current, e_future)
    qf = preparation_score(f_potential, f_current, f_future)

    # Translate desired object motion into crane controls.
    object_index = dense(object_pose, [list(range(10))])
    crane_index = dense(crane_pose, [list(range(10))])
    index_difference = va(object_index, vaff(crane_index, -1))

    inc = np.zeros((5, 10), np.float32)
    dec = np.zeros((5, 10), np.float32)
    for i in range(10):
        if i < 3:
            inc[0, i] = 1
        elif i < 6:
            inc[3, i] = 1
        elif i < 9:
            inc[2, i] = 1
        if 1 <= i <= 3:
            dec[2, i] = 1
        elif 4 <= i <= 6:
            dec[1, i] = 1
        elif 7 <= i <= 9:
            dec[0, i] = 1
    increasing_controls = dense(crane_pose, inc)
    decreasing_controls = dense(crane_pose, dec)

    def robot_request(direction, q):
        difference = va(
            index_difference, vaff(attached, direction)
        )
        positive = vclip(difference)
        negative = vclip(vaff(difference, -1))
        equal = vinv(va(positive, negative))
        flags = va(
            vm(increasing_controls, positive),
            vm(decreasing_controls, negative)
        )
        grab = dense(equal, [[0], [0], [0], [0], [1]])
        flags = va(flags, grab)
        flags = vm(flags, crane_mode)
        return va(
            dense(q, np.ones((5, 1))),
            dense(flags, 1000 * np.eye(5), -1000 * np.ones(5))
        )

    robot_scores = vmax(
        robot_request(-1, qp), robot_request(1, qn), 5
    )

    # Small deterministic tie-breaking cycle.
    clock = vs("mem", 5, 17)
    cw = np.zeros((12, 12), np.float32)
    cb = np.zeros(12, np.float32)
    cw[0, :11] = -1
    cb[0] = 1
    for i in range(1, 12):
        cw[i, i - 1] = 1
    next_clock = dense(clock, cw, cb)

    click_scores = vc(qa, qb, qc, qd, qe, qf, robot_scores)
    click_scores = va(
        click_scores, dense(next_clock, 0.08 * np.eye(11, 12))
    )

    suppressed = {}
    for c in (8, 9):
        r = node("rowmax", [hats[c]])
        r = clip(stencil(
            r, [(y, 0, 1) for y in range(-3, 4)]
        ))
        suppressed[c] = mul(ui[c], inv(r))

    targets = [
        hats[8], hats[9], hats[6], hats[15], ui[2], ui[11],
        ui[10], suppressed[9], ui[10], suppressed[9], suppressed[8]
    ]
    maps = add(
        lin(cat(*targets), 10000 * np.eye(11),
            -10000 * np.ones(11)),
        click_scores
    )

    spatial_tie = np.zeros((11, 2), np.float32)
    spatial_tie[6, 1] = -0.0001
    spatial_tie[7, 1] = -0.0001
    spatial_tie[8, 1] = 0.0001
    spatial_tie[9, 1] = 0.0001
    maps = add(maps, lin(pix, spatial_tie))
    click_map = max_channels(maps, 11)
    lin(click_map, [[1]], name="click")
    click_value = gm(click_map)

    movement = dense(
        movement, np.eye(4), [0.003, 0.002, 0.001, 0]
    )
    output_features = vc(movement, click_value)
    wo = np.zeros((8, 5), np.float32)
    bo = np.full(8, -100000, np.float32)
    for i in range(4):
        wo[i + 1, i] = 1
        bo[i + 1] = 0
    wo[6, 4] = 1
    bo[6] = 0
    dense(output_features, wo, bo, name="logits")

    node(
        "vconcat",
        [portal_memory, next_clock, e_count, growing,
         key_position, seek_key],
        name="mem_next"
    )

    return {"graph": graph, "memory_size": 22, "params": P}
