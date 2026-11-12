import numpy as np


def build():
    G, P, cache = [], {}, {}
    serial = [0]

    def uid(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p" + str(len(P))
            cache[key] = name
            P[name] = a
        return cache[key]

    def node(op, ins, body=None, name=None, **fields):
        name = uid() if name is None else name
        d = {"id": name, "op": op, "in": list(ins)}
        d.update(fields)
        (G if body is None else body).append(d)
        return name

    def dense(x, w, b=None, body=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], body, name, **fields)

    def conv(x, w, b=None, body=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], body, name, **fields)

    def lin(x, w, b=None, body=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b, body)

    def sl(x, a, b, body=None):
        return node("slice", [x], body, start=a, stop=b)

    def vsl(x, a, b, body=None):
        return node("vslice", [x], body, start=a, stop=b)

    def add(a, b, body=None):
        return node("add", [a, b], body)

    def sub(a, b, body=None):
        return node("sub", [a, b], body)

    def mul(a, b, body=None):
        return node("mul", [a, b], body)

    def va(a, b, body=None):
        return node("vadd", [a, b], body)

    def vm(a, b, body=None):
        return node("vmul", [a, b], body)

    def vs(x, scale, n=1, bias=None, body=None):
        b = None if bias is None else np.full(n, bias, np.float32)
        return dense(x, scale*np.eye(n, dtype=np.float32), b, body)

    def neg(x, n=1, body=None):
        return vs(x, -1, n, body=body)

    def vd(a, b, n=1, body=None):
        return va(a, neg(b, n, body), body)

    def vabs(x, n=1, body=None):
        p = node("vrelu", [x], body)
        q = node("vrelu", [neg(x, n, body)], body)
        return va(p, q, body)

    def vclip(x, n=1, ceiling=1.0, body=None):
        p = node("vrelu", [x], body)
        q = dense(p, np.eye(n, dtype=np.float32),
                  np.full(n, -ceiling, np.float32), body)
        q = node("vrelu", [q], body)
        return va(p, neg(q, n, body), body)

    def total(xs, body=None, vector=False):
        z = xs[0]
        for x in xs[1:]:
            z = va(z, x, body) if vector else add(z, x, body)
        return z

    def vmax(x, n):
        while n > 1:
            m = (n+1)//2
            a = np.zeros((m, n), np.float32)
            b = np.zeros((m, n), np.float32)
            for j in range(m):
                a[j, 2*j] = 1
                b[j, min(2*j+1, n-1)] = 1
            p = node("vrelu", [dense(x, a-b)])
            x = va(p, dense(x, b))
            n = m
        return x

    def winner(x, n, priority=0.001):
        x = dense(x, np.eye(n, dtype=np.float32),
                  priority*np.arange(n, dtype=np.float32))
        best = vmax(x, n)
        delta = va(x, dense(best, -np.ones((n, 1), np.float32)))
        return vclip(vs(delta, 1.0e7, n, bias=1), n)

    one = dense("avail", np.zeros((1, 8), np.float32), [1])

    def clip(x, body=None):
        p = node("relu", [x], body)
        return node("min2", [p, one], body)

    def affine(x, scale=1.0, bias=0.0, body=None):
        return lin(x, [[scale]], [bias], body)

    def eqzero(x, body=None):
        z = node("abs", [x], body)
        return clip(affine(z, -2, 1, body), body)

    def veqzero(x, body=None):
        return vclip(vs(vabs(x, body=body), -2, bias=1,
                        body=body), body=body)

    C, K = 9, 16
    R = 3*K
    M = R+14

    pixels = lin("coords", [[63, 0, 0], [0, 63, 0]])
    rank = lin("coords", [[4032, 63, 1]])
    valid = clip(lin("coords", [[-63, 0, 0]], [63]))

    cw = np.zeros((C, 16), np.float32)
    for j, colour in enumerate(range(6, 15)):
        cw[j, colour] = 1
    colours = mul(lin("x0", cw), valid)
    fg = lin(colours, np.ones((1, C), np.float32))
    black = sl("x0", 0, 1)
    has_black = node("gmax", [black])
    center = node("gmax", [mul(black, pixels)])
    position = center
    relative = add(pixels, neg(position, 2))
    absolute = node("abs", [relative])
    ay, ax = sl(absolute, 0, 1), sl(absolute, 1, 2)
    dy, dx = sl(relative, 0, 1), sl(relative, 1, 2)

    old_records = vsl("mem", 0, R)
    old_geometry = vsl("mem", R, R+6)
    old_position = vsl("mem", R+6, R+8)
    old_timer = vsl("mem", R+8, R+9)
    old_detour = vsl("mem", R+9, R+13)
    initialized = vsl("mem", R+13, R+14)
    previous_switch = vsl("aprev", 5, 6)

    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    edge_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diamond_directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    corner_pairs = [(4, 5), (6, 7), (4, 6), (5, 7)]
    diamond_corner_indices = [3, 1, 2, 0]

    # Radius sweep: five independent fit scores and their best radii.
    # Channels are horizontal line, vertical line, X, rectangle, diamond.
    ray_w = np.zeros((8, 8, 3, 3), np.float32)
    for j, (y, x) in enumerate(directions):
        ray_w[j, j, y+1, x+1] = 1
    rect_w = np.zeros((4, 4, 3, 3), np.float32)
    diam_w = np.zeros_like(rect_w)
    for j, (y, x) in enumerate(edge_directions):
        rect_w[j, j, y+1, x+1] = 1
    for j, (y, x) in enumerate(diamond_directions):
        diam_w[j, j, y+1, x+1] = 1
    rect_corners = np.zeros((4, 8), np.float32)
    for j, pair in enumerate(corner_pairs):
        rect_corners[j, list(pair)] = 1
    diam_corners = np.zeros((4, 8), np.float32)
    for j, index in enumerate(diamond_corner_indices):
        diam_corners[j, index] = 1

    detection_w = np.zeros((5, 24), np.float32)
    detection_w[0, [2, 3]] = 1
    detection_w[1, [0, 1]] = 1
    detection_w[2, 4:8] = 1
    detection_w[3, 8:12] = 1
    detection_w[3, 20:24] = -1
    detection_w[4, 12:16] = 1
    detection_w[4, 16:20] = -1

    initial_rays = lin(fg, np.ones((8, 1), np.float32))
    initial_edges = lin(fg, np.ones((4, 1), np.float32))
    outside = affine(black, 10000, -10000)
    db = []
    ds = [uid("detect_state") for _ in range(7)]
    rays_s, accum_s, rect_s, diam_s, radius_s, score_s, radii_s = ds
    rays_n = conv(rays_s, ray_w, body=db)
    accum_n = add(accum_s, rays_n, db)
    rect_n = add(conv(rect_s, rect_w, body=db),
                 lin(rays_n, rect_corners, body=db), db)
    diam_n = add(conv(diam_s, diam_w, body=db),
                 lin(rays_n, diam_corners, body=db), db)
    radius_n = vs(radius_s, 1, bias=1, body=db)
    joined = node("concat", [accum_n, rect_n, diam_n, rays_n], db)
    overlap = lin(joined, 1.5*detection_w, body=db)
    penalty = dense(radius_n, -np.array([[1], [1], [2], [4], [2]],
                                       np.float32), body=db)
    fit_map = add(add(overlap, penalty, db), outside, db)
    fit = node("gmax", [fit_map], db)
    improvement = vd(fit, score_s, 5, db)
    better = vclip(vs(improvement, 1000, 5, body=db), 5, body=db)
    score_n = va(score_s, node("vrelu", [improvement], db), db)
    radius_vector = dense(radius_n, np.ones((5, 1)), body=db)
    radii_n = va(vm(radius_vector, better, db),
                 vm(radii_s, vs(better, -1, 5, bias=1, body=db), db), db)
    detect_loop = uid("detect_loop")
    G.append({
        "id": detect_loop, "op": "loop", "T": 21,
        "state": [
            {"id": rays_s, "init": initial_rays},
            {"id": accum_s, "zeros": [8, 64, 64]},
            {"id": rect_s, "init": initial_edges},
            {"id": diam_s, "init": initial_edges},
            {"id": radius_s, "zeros": [1]},
            {"id": score_s, "zeros": [5]},
            {"id": radii_s, "zeros": [5]}
        ],
        "body": db,
        "next": {
            rays_s: rays_n, accum_s: accum_n,
            rect_s: rect_n, diam_s: diam_n,
            radius_s: radius_n, score_s: score_n, radii_s: radii_n
        }
    })
    scores = detect_loop+"."+score_s
    radii = detect_loop+"."+radii_s
    family_scores = dense(scores, [
        [1, 1, 0, 0, 0], [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]
    ])
    family = winner(family_scores, 4)
    rx_candidates = dense(radii, [
        [1, 0, 0, 0, 0], [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]
    ])
    ry_candidates = dense(radii, [
        [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]
    ])
    detected_rx = dense(vm(rx_candidates, family), [[1, 1, 1, 1]])
    detected_ry = dense(vm(ry_candidates, family), [[1, 1, 1, 1]])
    detected_geometry = node("vconcat", [detected_rx, detected_ry, family])

    rigid = dense(old_geometry, [[0, 0, 1, 1, 0, 0]])
    keep = vm(vm(vclip(rigid), initialized),
              vs(previous_switch, -1, bias=1))
    geometry = va(vm(old_geometry, keep),
                  vm(detected_geometry, vs(keep, -1, bias=1)))
    rx, ry = vsl(geometry, 0, 1), vsl(geometry, 1, 2)
    cross_f, x_f = vsl(geometry, 2, 3), vsl(geometry, 3, 4)
    rect_f, diam_f = vsl(geometry, 4, 5), vsl(geometry, 5, 6)
    outline = va(rect_f, diam_f)

    bx = clip(affine(sub(mul(valid, rx), ax), 2, 1))
    by = clip(affine(sub(mul(valid, ry), ay), 2, 1))
    box = mul(bx, by)

    local = mul(fg, box)
    col_strength = add(node("colsum", [local]), affine(ax, -0.01))
    row_strength = add(node("rowsum", [local]), affine(ay, -0.01))
    col_peak, row_peak = node("gmax", [col_strength]), node("gmax", [row_strength])
    col_gate = clip(affine(add(col_strength, neg(col_peak)), 1.0e7, 1))
    row_gate = clip(affine(add(row_strength, neg(row_peak)), 1.0e7, 1))
    col_coordinate = node("gmax", [mul(col_gate, sl(pixels, 1, 2))])
    row_coordinate = node("gmax", [mul(row_gate, sl(pixels, 0, 1))])
    offset_x = vm(vd(col_coordinate, vsl(position, 1, 2)), cross_f)
    offset_y = vm(vd(row_coordinate, vsl(position, 0, 1)), cross_f)

    axes = clip(add(eqzero(sub(dx, mul(valid, offset_x))),
                    eqzero(sub(dy, mul(valid, offset_y)))))
    cross_mask = mul(box, axes)
    x_mask = mul(box, eqzero(sub(ax, ay)))
    rect_mask = mul(box, clip(add(
        eqzero(sub(ax, mul(valid, rx))),
        eqzero(sub(ay, mul(valid, ry)))
    )))
    diam_mask = eqzero(sub(add(ax, ay), mul(valid, rx)))
    selected = clip(total([
        mul(cross_mask, cross_f), mul(x_mask, x_f),
        mul(rect_mask, rect_f), mul(diam_mask, diam_f)
    ]))
    colour_counts = vs(node("gmean", [mul(colours, selected)]), 4096, C)
    current_colour = winner(colour_counts, C)

    # Small colour-parallel shift operators.
    def colour_shift(x, y, z, kernel=3, body=None):
        parts = []
        half = kernel//2
        for a in range(0, C, 3):
            b = min(a+3, C)
            n = b-a
            w = np.zeros((n, n, kernel, kernel), np.float32)
            for j in range(n):
                w[j, j, half+y, half+z] = 1
            part = sl(x, a, b, body)
            parts.append(conv(part, w, body=body))
        return node("concat", parts, body)

    # Memory records consist only of row, column, and colour id.
    slot_expand = np.zeros((R, K), np.float32)
    record_reduce = np.zeros((3, R), np.float32)
    for k in range(K):
        slot_expand[3*k:3*k+3, k] = 1
        record_reduce[:, 3*k:3*k+3] = np.eye(3)
    slot_shift = np.zeros((K, K), np.float32)
    for k in range(K-1):
        slot_shift[k+1, k] = 1
    slot_initial = dense(one, np.zeros((K, 1)), [1]+[0]*(K-1))

    mb = []
    read_slot, remembered_s = uid("read_slot"), uid("remembered_state")
    record = dense(
        vm(old_records, dense(read_slot, slot_expand, body=mb), mb),
        record_reduce, body=mb
    )
    record_position = vsl(record, 0, 2, mb)
    record_colour = vsl(record, 2, 3, mb)
    distances = node("abs", [add(pixels, neg(record_position, 2, mb), mb)], mb)
    mask = clip(lin(distances, [[-2, -2]], [1], mb), mb)
    colour_delta = dense(record_colour, np.ones((C, 1)),
                         -np.arange(6, 15, dtype=np.float32), mb)
    record_labels = vclip(
        vs(vabs(colour_delta, C, mb), -2, C, bias=1, body=mb),
        C, body=mb
    )
    remembered_n = add(remembered_s, mul(mask, record_labels, mb), mb)
    read_slot_n = dense(read_slot, slot_shift, body=mb)
    read_loop = uid("read_loop")
    G.append({
        "id": read_loop, "op": "loop", "T": K,
        "state": [
            {"id": read_slot, "init": slot_initial},
            {"id": remembered_s, "zeros": [C, 64, 64]}
        ],
        "body": mb,
        "next": {read_slot: read_slot_n, remembered_s: remembered_n}
    })
    remembered = clip(read_loop+"."+remembered_s)
    remembered_presence = clip(lin(remembered, np.ones((1, C))))

    ring_w = np.ones((1, 1, 3, 3), np.float32)
    ring_w[0, 0, 1, 1] = -8
    ring = conv(sl("x0", 4, 5), ring_w)
    observed_goals = mul(clip(affine(ring, 1, -3)),
                         clip(add(fg, black)))
    visible_goals = mul(colours, clip(affine(ring, 1, -7)))
    any_goal = node("max2", [observed_goals, remembered_presence])

    # Fill labels of initially obscured markers from the nearest known marker.
    seeds = node("max2", [visible_goals, remembered])
    seed_values = lin(seeds, 200*np.eye(C, dtype=np.float32))
    decay = dense(one, np.zeros((C, 1)), -np.ones(C))
    vb, vor_s = [], uid("voronoi_state")
    spread = vor_s
    for y, x in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbour = colour_shift(vor_s, y, x, body=vb)
        spread = node("max2", [spread, neighbour], vb)
    vor_n = node("max2", [add(spread, decay, vb), seed_values], vb)
    vor_loop = uid("voronoi_loop")
    G.append({
        "id": vor_loop, "op": "loop", "T": 64,
        "state": [{"id": vor_s, "init": seed_values}],
        "body": vb, "next": {vor_s: vor_n}
    })
    label_score = add(vor_loop+"."+vor_s, add(
        lin(remembered, 500*np.eye(C, dtype=np.float32)),
        lin(visible_goals, 250*np.eye(C, dtype=np.float32))
    ))
    label_score = lin(label_score, np.eye(C, dtype=np.float32),
                      0.01*np.arange(C, dtype=np.float32))
    label_max = sl(label_score, 0, 1)
    for c in range(1, C):
        label_max = node("max2", [label_max, sl(label_score, c, c+1)])
    labels = clip(lin(sub(label_score, label_max),
                      1.0e6*np.eye(C, dtype=np.float32), np.ones(C)))
    original_goals = mul(labels, any_goal)

    correct = lin(mul(original_goals, colours), np.ones((1, C)))
    elsewhere = mul(correct, affine(selected, -1, 1))
    goals = mul(original_goals, affine(elsewhere, -1, 1))

    # Write all marker records, in deterministic spatial order.
    wb = []
    remaining_s, records_s, write_slot = (
        uid("remaining_state"), uid("records_state"), uid("write_slot")
    )
    marker_score = mul(remaining_s, rank, wb)
    marker_peak = node("gmax", [marker_score], wb)
    marker_gate = eqzero(
        add(marker_score, neg(marker_peak, body=wb), wb), wb
    )
    marker_gate = mul(marker_gate, remaining_s, wb)
    marker_position = node("gmax", [mul(marker_gate, pixels, wb)], wb)
    marker_labels = node("gmax", [mul(marker_gate, original_goals, wb)], wb)
    marker_colour = dense(marker_labels,
                          [np.arange(6, 15, dtype=np.float32)], body=wb)
    marker_record = node("vconcat", [marker_position, marker_colour], wb)
    repeated_record = dense(marker_record,
                            np.tile(np.eye(3, dtype=np.float32), (K, 1)),
                            body=wb)
    record_value = vm(repeated_record,
                      dense(write_slot, slot_expand, body=wb), wb)
    records_n = va(records_s, record_value, wb)
    remaining_n = sub(remaining_s, marker_gate, wb)
    write_slot_n = dense(write_slot, slot_shift, body=wb)
    write_loop = uid("write_loop")
    G.append({
        "id": write_loop, "op": "loop", "T": K,
        "state": [
            {"id": remaining_s, "init": any_goal},
            {"id": records_s, "zeros": [R]},
            {"id": write_slot, "init": slot_initial}
        ],
        "body": wb,
        "next": {
            remaining_s: remaining_n, records_s: records_n,
            write_slot: write_slot_n
        }
    })

    paint_near = clip(conv(sl("x0", 2, 3),
                           np.ones((1, 1, 3, 3), np.float32)))
    paint = mul(colours, paint_near)
    paint_present = node("gmax", [paint])

    # Reachable centres on the current three-pixel translation lattice.
    rb, reach_s = [], uid("reach_state")
    reach_w = np.zeros((1, 1, 7, 7), np.float32)
    for y, x in [(3, 3), (0, 3), (6, 3), (3, 0), (3, 6)]:
        reach_w[0, 0, y, x] = 1
    reach_n = node("min2", [conv(reach_s, reach_w, body=rb), one], rb)
    reach_loop = uid("reach_loop")
    G.append({
        "id": reach_loop, "op": "loop", "T": 42,
        "state": [{"id": reach_s, "init": black}],
        "body": rb, "next": {reach_s: reach_n}
    })
    reachable = reach_loop+"."+reach_s

    # Colour-specific correlations of the selected shape with target points.
    cb = []
    ray_states = [uid("coverage_ray") for _ in range(8)]
    rect_states = [uid("coverage_rect") for _ in range(4)]
    diam_states = [uid("coverage_diamond") for _ in range(4)]
    other_s, h_s, v_s, radius_s = (
        uid("other_coverage"), uid("horizontal_coverage"),
        uid("vertical_coverage"), uid("coverage_radius")
    )
    initial_other = mul(goals, x_f)
    states = (
        [{"id": s, "init": goals} for s in ray_states] +
        [{"id": s, "init": goals} for s in rect_states] +
        [{"id": s, "init": goals} for s in diam_states] +
        [
            {"id": other_s, "init": initial_other},
            {"id": h_s, "zeros": [C, 64, 64]},
            {"id": v_s, "zeros": [C, 64, 64]},
            {"id": radius_s, "zeros": [1]}
        ]
    )
    ray_values = [
        colour_shift(s, y, x, body=cb)
        for s, (y, x) in zip(ray_states, directions)
    ]
    rect_values = []
    for s, (y, x), pair in zip(rect_states, edge_directions, corner_pairs):
        rect_values.append(add(
            colour_shift(s, y, x, body=cb),
            add(ray_values[pair[0]], ray_values[pair[1]], cb), cb
        ))
    diam_values = []
    for s, (y, x), corner in zip(
        diam_states, diamond_directions, diamond_corner_indices
    ):
        diam_values.append(add(
            colour_shift(s, y, x, body=cb), ray_values[corner], cb
        ))
    radius_n = vs(radius_s, 1, bias=1, body=cb)

    def radius_gate(r):
        return vclip(vs(vd(r, radius_n, body=cb),
                         1, bias=1, body=cb), body=cb)

    gx, gy = radius_gate(rx), radius_gate(ry)
    equal_radius = vclip(vs(
        vabs(vd(rx, radius_n, body=cb), body=cb),
        -1, bias=1, body=cb
    ), body=cb)
    horizontal_points = add(ray_values[2], ray_values[3], cb)
    vertical_points = add(ray_values[0], ray_values[1], cb)
    diagonal_points = total(ray_values[4:8], cb)
    rectangle_points = sub(total(rect_values, cb), diagonal_points, cb)
    diamond_points = sub(total(diam_values, cb),
                         total(ray_values[0:4], cb), cb)
    other_n = add(other_s, total([
        mul(diagonal_points, vm(x_f, gx, cb), cb),
        mul(rectangle_points, vm(rect_f, equal_radius, cb), cb),
        mul(diamond_points, vm(diam_f, equal_radius, cb), cb)
    ], cb), cb)
    h_n = add(h_s, mul(horizontal_points, gx, cb), cb)
    v_n = add(v_s, mul(vertical_points, gy, cb), cb)
    nexts = {radius_s: radius_n, other_s: other_n, h_s: h_n, v_s: v_n}
    nexts.update(zip(ray_states, ray_values))
    nexts.update(zip(rect_states, rect_values))
    nexts.update(zip(diam_states, diam_values))
    coverage_loop = uid("coverage_loop")
    G.append({
        "id": coverage_loop, "op": "loop", "T": 21,
        "state": states, "body": cb, "next": nexts
    })
    hcov = coverage_loop+"."+h_s
    vcov = coverage_loop+"."+v_s
    ocov = coverage_loop+"."+other_s

    # Compensate for cross arms moved by the central obstacle.
    sb = []
    shift_states = [uid("shift_state") for _ in range(4)]
    sr, sh, sv = uid("shift_radius"), uid("shift_h"), uid("shift_v")
    shifted = [
        colour_shift(shift_states[0], 3, 0, 7, sb),
        colour_shift(shift_states[1], -3, 0, 7, sb),
        colour_shift(shift_states[2], 0, 3, 7, sb),
        colour_shift(shift_states[3], 0, -3, 7, sb)
    ]
    nr = vs(sr, 1, bias=3, body=sb)
    hp = veqzero(vd(offset_y, nr, body=sb), sb)
    hn = veqzero(va(offset_y, nr, sb), sb)
    vp = veqzero(vd(offset_x, nr, body=sb), sb)
    vn = veqzero(va(offset_x, nr, sb), sb)
    shn = add(sh, add(mul(shifted[0], hp, sb),
                       mul(shifted[1], hn, sb), sb), sb)
    svn = add(sv, add(mul(shifted[2], vp, sb),
                       mul(shifted[3], vn, sb), sb), sb)
    hz, vz = veqzero(offset_y), veqzero(offset_x)
    hi, vi = mul(hcov, hz), mul(vcov, vz)
    shift_next = dict(zip(shift_states, shifted))
    shift_next.update({sr: nr, sh: shn, sv: svn})
    shift_loop = uid("shift_loop")
    G.append({
        "id": shift_loop, "op": "loop", "T": 7,
        "state": [
            {"id": shift_states[0], "init": hcov},
            {"id": shift_states[1], "init": hcov},
            {"id": shift_states[2], "init": vcov},
            {"id": shift_states[3], "init": vcov},
            {"id": sr, "zeros": [1]},
            {"id": sh, "init": hi},
            {"id": sv, "init": vi}
        ],
        "body": sb, "next": shift_next
    })
    cross_coverage = total([
        shift_loop+"."+sh, shift_loop+"."+sv,
        mul(goals, vclip(va(hz, vz)))
    ])
    coverage = add(ocov, mul(cross_coverage, cross_f))

    allowed = vclip(va(current_colour, paint_present), C)
    colour_bias = va(vs(allowed, 100, C, bias=-100),
                     vs(current_colour, 0.25, C, bias=-0.25))
    placement = add(coverage, colour_bias)
    placement = add(placement, lin(absolute, [[-0.003, -0.003]]))
    placement = add(placement, affine(reachable, 1000, -1000))
    desired_colour = winner(node("gmax", [placement]), C, 0.0001)
    placement_score = lin(mul(placement, desired_colour), np.ones((1, C)))

    best = node("gmax", [placement_score])
    target_gate = clip(affine(add(placement_score, neg(best)), 1.0e7, 1))
    ranked = mul(target_gate, rank)
    rank_best = node("gmax", [ranked])
    target_gate = mul(target_gate, eqzero(add(ranked, neg(rank_best))))
    target_position = node("gmax", [mul(target_gate, pixels)])

    same_colour = dense(vm(current_colour, desired_colour), np.ones((1, C)))
    needs_paint = vclip(vs(same_colour, -1, bias=1))
    desired_paint = lin(mul(paint, desired_colour), np.ones((1, C)))
    paint_position = node("gmax", [mul(desired_paint, pixels)])
    py, px = vsl(paint_position, 0, 1), vsl(paint_position, 1, 2)
    left_station = vclip(vs(px, -1, bias=32))
    inward = vs(left_station, 2, bias=-1)
    px = va(px, vm(vm(inward, rx), outline))
    paint_target = vclip(node("vconcat", [py, px]), 2, ceiling=63)
    destination = va(vm(paint_target, needs_paint),
                     vm(target_position, vs(needs_paint, -1, bias=1)))
    delta = vd(destination, position, 2)
    moves = dense(delta, [[-1, 0], [1, 0], [0, -1], [0, 1]])

    displacement = dense(vabs(vd(position, old_position, 2), 2), [[1, 1]])
    unchanged = vclip(vs(displacement, -10, bias=1))
    previous_move_w = np.zeros((1, 8), np.float32)
    previous_move_w[0, 1:5] = 1
    previous_move = dense("aprev", previous_move_w)
    stuck = vm(vm(vm(unchanged, previous_move), initialized), has_black)
    previous_vertical_w = np.zeros((1, 8), np.float32)
    previous_vertical_w[0, 1:3] = 1
    previous_vertical = dense("aprev", previous_vertical_w)
    previous_horizontal = vs(previous_vertical, -1, bias=1)
    upper = vclip(vs(vsl(position, 0, 1), -2, bias=64))
    left = vclip(vs(vsl(position, 1, 2), -2, bias=64))
    fallback = node("vconcat", [
        vm(previous_horizontal, upper),
        vm(previous_horizontal, vs(upper, -1, bias=1)),
        vm(previous_vertical, left),
        vm(previous_vertical, vs(left, -1, bias=1))
    ])
    detour = va(vm(fallback, stuck), vm(old_detour, vs(stuck, -1, bias=1)))
    next_timer = vclip(va(
        node("vrelu", [vs(old_timer, 1, bias=-1)]),
        vs(stuck, 4)
    ), ceiling=6)
    active_detour = vclip(va(old_timer, stuck))
    moves = va(moves, vs(vm(detour, active_detour), 100, 4))

    output_w = np.zeros((8, 4), np.float32)
    output_w[1:5] = np.eye(4, dtype=np.float32)
    output_b = np.array([-1000, 0, 0, 0, 0, 1.4, -1000, -1000], np.float32)
    logits0 = dense(moves, output_w, output_b)
    missing_black = vs(has_black, -1, bias=1)
    recovery_w = np.zeros((8, 1), np.float32)
    recovery_w[1, 0] = 10000
    node("vadd", [logits0, dense(missing_black, recovery_w)], name="logits")
    conv(black, np.zeros((1, 1, 1, 1), np.float32), name="click")
    node("vconcat", [
        write_loop+"."+records_s, geometry, position,
        next_timer, detour, one
    ], name="mem_next")

    # Static graph bookkeeping; no game execution or weight fitting.
    used = set()
    for nd in G:
        for item in nd["body"] if nd["op"] == "loop" else [nd]:
            for field in ("w", "b"):
                if field in item:
                    used.add(item[field])
    P = {k: v for k, v in P.items() if k in used}

    return {"graph": G, "memory_size": M, "params": P}
