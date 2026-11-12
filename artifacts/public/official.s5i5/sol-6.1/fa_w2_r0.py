import math
import numpy as np


def build():
    params = {}
    graph = []
    current = graph
    serial = 0
    constants = {}
    caches = {}

    def emit(op, ins=None, name=None, **fields):
        nonlocal serial
        ins = None if ins is None else tuple(ins)
        cache = caches.setdefault(id(current), {})
        reusable = name is None and op != "loop"
        if reusable:
            key = (op, ins, tuple(sorted(fields.items())))
            if key in cache:
                return cache[key]
        if name is None:
            serial += 1
            name = "n%d" % serial
        node = {"id": name, "op": op}
        if ins is not None:
            node["in"] = list(ins)
        node.update(fields)
        current.append(node)
        if reusable:
            cache[key] = name
        return name

    def parameter(name, value):
        if name not in params:
            params[name] = np.asarray(value, dtype=np.float32)
        return name

    def dense(x, w, name=None, bias=None, out=None):
        name = name or "p%d" % (len(params) + 1)
        fields = {"w": parameter(name, w)}
        if bias is not None:
            fields["b"] = parameter(name + "_b", bias)
        return emit("dense", [x], name=out, **fields)

    def conv(x, w, name=None, bias=None, out=None):
        name = name or "p%d" % (len(params) + 1)
        fields = {"w": parameter(name, w)}
        if bias is not None:
            fields["b"] = parameter(name + "_b", bias)
        return emit("conv", [x], name=out, **fields)

    def const(value):
        nonlocal current
        value = float(value)
        if value not in constants:
            saved = current
            current = graph
            constants[value] = dense(
                "avail", np.zeros((1, 8), np.float32),
                bias=np.array([value], np.float32)
            )
            current = saved
        return constants[value]

    def vector_constant(values):
        nonlocal current
        values = np.asarray(values, np.float32).reshape(-1)
        saved = current
        current = graph
        result = dense(
            "avail", np.zeros((len(values), 8), np.float32),
            bias=values
        )
        current = saved
        return result

    def cat(xs, spatial=False):
        return emit("concat" if spatial else "vconcat", xs)

    def sl(x, a, b, spatial=False):
        return emit("slice" if spatial else "vslice", [x],
                    start=a, stop=b)

    def add(a, b, spatial=False):
        return emit("add" if spatial else "vadd", [a, b])

    def mul(a, b, spatial=False):
        return emit("mul" if spatial else "vmul", [a, b])

    def scale(a, value, spatial=False):
        return mul(a, const(value), spatial)

    def neg(a, spatial=False):
        return scale(a, -1, spatial)

    def sub(a, b, spatial=False):
        if spatial:
            return emit("sub", [a, b])
        return add(a, neg(b))

    def relu(a, spatial=False):
        return emit("relu" if spatial else "vrelu", [a])

    def absolute(a, spatial=False):
        if spatial:
            return emit("abs", [a])
        return add(relu(a), relu(neg(a)))

    def total(xs, spatial=False):
        result = xs[0]
        for x in xs[1:]:
            result = add(result, x, spatial)
        return result

    def step(a, threshold=0.0, spatial=False):
        z = add(scale(a, 4096, spatial),
                const(1 - 4096 * threshold), spatial)
        r = relu(z, spatial)
        return sub(r, relu(sub(r, const(1), spatial), spatial), spatial)

    def equal(a, b, tolerance=0.01, spatial=False):
        return step(
            neg(absolute(sub(a, b, spatial), spatial), spatial),
            -tolerance, spatial
        )

    def vector_max(x, n):
        while n > 1:
            m = (n + 1) // 2
            wa = np.zeros((m, n), np.float32)
            wb = np.zeros_like(wa)
            for i in range(m):
                wa[i, 2 * i] = 1
                wb[i, min(2 * i + 1, n - 1)] = 1
            a = dense(x, wa)
            b = dense(x, wb)
            x = add(b, relu(sub(a, b)))
            n = m
        return x

    colors = [0, 1, 6, 7, 8, 9, 10, 11, 12, 14]
    control_colors = [c for c in colors if c != 1]
    C = len(colors)
    A = len(control_colors)
    N = 2 * C
    K = 3 * A
    MARKERS = 2
    HISTORY = 45
    carrier_index = colors.index(1)

    # Coordinates measured in three-pixel mechanical cells.
    wx = np.zeros((1, 3, 1, 1), np.float32)
    wy = wx.copy()
    wx[0, 1, 0, 0] = 21
    wy[0, 0, 0, 0] = 21
    gx = conv("coords", wx)
    gy = conv("coords", wy)
    raster_key = add(
        add(scale(gy, 64, True), gx, True), const(1), True
    )

    def select_last(mask):
        scores = mul(mask, raster_key, True)
        best = emit("gmax", [scores])
        chosen = mul(equal(scores, best, 0.01, True), mask, True)
        return chosen, relu(sub(mask, chosen, True), True)

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    # Find green pads and the canonical endpoints of plain carrier bars.
    arm_w = np.zeros((4 * C, 16, 5, 5), np.float32)
    arm_b = np.full(4 * C, -6, np.float32)
    allowed_w = np.zeros((4 * C, 16, 3, 3), np.float32)

    for d, (dx, dy) in enumerate(directions):
        px, py = -dy, dx
        for ci, color in enumerate(colors):
            o = d * C + ci
            for t in (-1, 1):
                xx, yy = -dx + t * px, -dy + t * py
                arm_w[o, 3, yy + 2, xx + 2] = 1
            arm_w[o, 3, 2 - dy, 2 - dx] = 1
            arm_w[o, 13, 2 - dy, 2 - dx] = 1
            for u in (0, 1):
                for t in (-1, 1):
                    xx, yy = u * dx + t * px, u * dy + t * py
                    arm_w[o, color, yy + 2, xx + 2] = 1
                    arm_w[o, 13, yy + 2, xx + 2] = 1
            for t in (-1, 1):
                xx, yy = t * px, t * py
                allowed_w[o, color, yy + 1, xx + 1] = 1
                allowed_w[o, 13, yy + 1, xx + 1] = 1

            if color == 1:
                arm_w[o] = 0
                if d < 2:
                    arm_b[o] = -8
                    for u in (-1, 0, 1):
                        for t in (-1, 0, 1):
                            xx, yy = u * dx + t * px, u * dy + t * py
                            arm_w[o, 1, yy + 2, xx + 2] = 1
                            arm_w[o, 13, yy + 2, xx + 2] = 1
                    for t in (-1, 0, 1):
                        xx, yy = -2 * dx + t * px, -2 * dy + t * py
                        arm_w[o, 1, yy + 2, xx + 2] = -9
                    for t in (-2, 2):
                        xx, yy = t * px, t * py
                        arm_w[o, 1, yy + 2, xx + 2] = -9
                else:
                    arm_b[o] = -100

    seeds = relu(conv("x0", arm_w, "arm_detector", arm_b), True)
    allowed = relu(conv(
        "x0", allowed_w, "axis_detector",
        np.full(4 * C, -1, np.float32)
    ), True)

    mix_w = np.zeros((C, 4 * C, 1, 1), np.float32)
    dx_w = mix_w.copy()
    dy_w = mix_w.copy()
    expand_w = np.zeros((4 * C, C, 1, 1), np.float32)
    for d, (dx, dy) in enumerate(directions):
        for c in range(C):
            mix_w[c, d * C + c, 0, 0] = 1
            dx_w[c, d * C + c, 0, 0] = dx
            dy_w[c, d * C + c, 0, 0] = dy
            expand_w[d * C + c, c, 0, 0] = 1

    all_pivots = emit("min2", [
        conv(seeds, mix_w, "color_mix"), const(1)
    ])

    shift_names = []
    for d, (dx, dy) in enumerate(directions):
        w = np.zeros((C, C, 3, 3), np.float32)
        for c in range(C):
            w[c, c, 1 - dy, 1 - dx] = 1
        shift_names.append(parameter("axis_shift_%d" % d, w))

    body = []
    current = body
    shifted = []
    for d in range(4):
        part = sl("axis_state", d * C, (d + 1) * C, True)
        shifted.append(emit("conv", [part], w=shift_names[d]))
    axis_next = emit("min2", [
        add(cat(shifted, True), seeds, True), allowed
    ])
    current = graph
    emit("loop", name="axis_flood", T=63,
         state=[{"id": "axis_state", "init": seeds}],
         body=body, next={"axis_state": axis_next})

    # Purple dots have solid corners; target crosses have four purple arms.
    marker_w = np.zeros((1, 16, 3, 3), np.float32)
    marker_w[0, 13, 1, 1] = 1
    for yy in (0, 2):
        for xx in (0, 2):
            for color in colors + [3]:
                marker_w[0, color, yy, xx] = 1
    dots = relu(conv(
        "x0", marker_w, "dot_detector", np.array([-4], np.float32)
    ), True)

    target_w = np.zeros((1, 16, 3, 3), np.float32)
    for yy, xx in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        target_w[0, 13, yy, xx] = 1
    targets = relu(conv(
        "x0", target_w, "target_detector", np.array([-3], np.float32)
    ), True)

    marker_maps = []
    marker_x = []
    marker_y = []
    marker_exists = []
    remainder = dots
    for _ in range(MARKERS):
        chosen, remainder = select_last(remainder)
        marker_maps.append(chosen)
        marker_exists.append(emit("gmax", [chosen]))
        marker_x.append(scale(
            emit("gmean", [mul(chosen, gx, True)]), 4096
        ))
        marker_y.append(scale(
            emit("gmean", [mul(chosen, gy, True)]), 4096
        ))

    xs, ys, dxs, dys, lengths, exists = [], [], [], [], [], []
    direct_parts = [[] for _ in range(MARKERS)]
    remaining = all_pivots
    for _ in range(2):
        chosen, remaining = select_last(remaining)
        selected = mul(
            seeds, conv(chosen, expand_w, "color_expand"), True
        )
        ex = emit("gmax", [chosen])
        bx0 = scale(emit("gmean", [mul(chosen, gx, True)]), 4096)
        by0 = scale(emit("gmean", [mul(chosen, gy, True)]), 4096)
        vx0 = scale(emit("gmean", [
            conv(selected, dx_w, "direction_x")
        ]), 4096)
        vy0 = scale(emit("gmean", [
            conv(selected, dy_w, "direction_y")
        ]), 4096)

        projections = []
        for d in range(4):
            part = sl(selected, d * C, (d + 1) * C, True)
            projections.append(emit(
                "rowmax" if d in (0, 2) else "colmax", [part]
            ))
        arm_line = conv(
            mul("axis_flood.axis_state", cat(projections, True), True),
            mix_w, "color_mix"
        )
        along = add(
            mul(sub(gx, bx0, True), vx0, True),
            mul(sub(gy, by0, True), vy0, True), True
        )
        extent = emit("gmax", [mul(arm_line, along, True)])
        ll = mul(add(extent, const(2 / 3)), ex)

        xs.append(bx0)
        ys.append(by0)
        dxs.append(vx0)
        dys.append(vy0)
        lengths.append(ll)
        exists.append(ex)
        for m in range(MARKERS):
            direct_parts[m].append(emit("gmax", [
                mul(arm_line, marker_maps[m], True)
            ]))

    bx, by = cat(xs), cat(ys)
    vx, vy = cat(dxs), cat(dys)
    length, present = cat(lengths), cat(exists)
    direct = [cat(parts) for parts in direct_parts]
    tx = add(bx, mul(sub(length, const(1)), vx))
    ty = add(by, mul(sub(length, const(1)), vy))

    parent_gather = np.zeros((N * N, N), np.float32)
    child_gather = np.zeros_like(parent_gather)
    diagonal = np.zeros(N * N, np.float32)
    for j in range(N):
        for i in range(N):
            r = j * N + i
            parent_gather[r, i] = 1
            child_gather[r, j] = 1
            diagonal[r] = float(i == j)

    parameter("gather_parent", parent_gather)
    parameter("gather_child", child_gather)
    parameter("sum_parent", parent_gather.T)
    parameter("sum_child", child_gather.T)

    def pg(x):
        return emit("dense", [x], w="gather_parent")

    def cg(x):
        return emit("dense", [x], w="gather_child")

    def ps(x):
        return emit("dense", [x], w="sum_parent")

    def cs(x):
        return emit("dense", [x], w="sum_child")

    def sum_n(x):
        return dense(x, np.ones((1, N), np.float32), "sum_objects")

    bare_array = np.zeros(N, np.float32)
    bare_array[carrier_index] = 1
    bare_array[C + carrier_index] = 1
    bare = vector_constant(bare_array)

    # Geometric attachment inference.
    ex_pair = sub(pg(add(tx, vx)), cg(bx))
    ey_pair = sub(pg(add(ty, vy)), cg(by))
    base_error = add(absolute(ex_pair), absolute(ey_pair))
    child_dx, child_dy = cg(vx), cg(vy)
    along = add(mul(ex_pair, child_dx), mul(ey_pair, child_dy))
    across = sub(mul(ex_pair, child_dy), mul(ey_pair, child_dx))
    carrier_error = total([
        absolute(across),
        relu(neg(along)),
        relu(sub(along, sub(cg(length), const(1))))
    ])
    child_bare = cg(bare)
    attachment_error = add(
        mul(base_error, sub(const(1), child_bare)),
        mul(carrier_error, child_bare)
    )
    adjacency = step(neg(attachment_error), -0.01)
    adjacency = mul(adjacency, vector_constant(1 - diagonal))
    adjacency = mul(adjacency, mul(pg(present), cg(present)))
    pair_dot = add(mul(pg(vx), child_dx), mul(pg(vy), child_dy))
    adjacency = mul(adjacency, step(pair_dot, -0.5))

    parent_dx = cs(mul(adjacency, pg(vx)))
    parent_dy = cs(mul(adjacency, pg(vy)))
    rotated_parent_dot = sub(mul(vy, parent_dx), mul(vx, parent_dy))
    skip_quarter = step(neg(rotated_parent_dot), 0.5)
    shrinkable = step(length, 1.5)

    initial_columns = []
    for i in range(N):
        z = np.zeros(N, np.float32)
        z[i] = 1
        initial_columns.append(vector_constant(z))

    body = []
    current = body
    states, next_states = [], {}
    for i in range(N):
        sid = "reach_object_%d" % i
        states.append({"id": sid, "init": initial_columns[i]})
        next_states[sid] = add(
            initial_columns[i], ps(mul(adjacency, cg(sid)))
        )
    for m in range(MARKERS):
        sid = "reach_marker_%d" % m
        states.append({"id": sid, "init": direct[m]})
        next_states[sid] = add(direct[m], ps(mul(adjacency, cg(sid))))
    current = graph
    emit("loop", name="ownership", T=6,
         state=states, body=body, next=next_states)

    reach = cat(["ownership.reach_object_%d" % i for i in range(N)])
    strict_reach = mul(reach, vector_constant(1 - diagonal))
    marker_owner = [
        "ownership.reach_marker_%d" % m for m in range(MARKERS)
    ]

    # Detect the resize and rotation widgets.
    resize_w = np.zeros((2 * A, 16, 5, 5), np.float32)
    for orient in range(2):
        for ci, color in enumerate(control_colors):
            o = orient * A + ci
            points = [(0, t, 3) for t in (-1, 0, 1)]
            points += [(-1, 0, 4), (1, 0, 4)]
            points += [
                (xx, yy, color) for xx in (-2, 2) for yy in (-1, 1)
            ]
            for xx, yy, cc in points:
                if orient:
                    xx, yy = yy, xx
                resize_w[o, cc, yy + 2, xx + 2] += 1
    resize_centres = relu(conv(
        "x0", resize_w, "resize_detector",
        np.full(2 * A, -8, np.float32)
    ), True)

    resize_click_w = np.zeros((2 * A, 2 * A, 5, 5), np.float32)
    for sign_index, sign in enumerate((-1, 1)):
        for ci in range(A):
            o = sign_index * A + ci
            resize_click_w[o, ci, 2, 2 - 2 * sign] = 1
            resize_click_w[o, A + ci, 2 - 2 * sign, 2] = 1
    resize_clicks = conv(resize_centres, resize_click_w, "resize_clicks")

    rotation_w = np.zeros((A, 16, 5, 5), np.float32)
    for ci, color in enumerate(control_colors):
        for yy, xx in [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]:
            rotation_w[ci, color, yy, xx] = 1
        for yy, xx in [(1, 1), (1, 3), (3, 1), (3, 3)]:
            rotation_w[ci, 4, yy, xx] = 1
        for yy, xx in [(0, 2), (4, 2), (2, 0), (2, 4)]:
            rotation_w[ci, 2, yy, xx] = 1
    rotation_clicks = relu(conv(
        "x0", rotation_w, "rotation_detector",
        np.full(A, -12, np.float32)
    ), True)
    click_maps = cat([resize_clicks, rotation_clicks], True)
    button_available = emit("gmax", [click_maps])

    # Wall-aware distance field, translated in three-pixel increments.
    wall = sl("x0", 15, 16, True)
    wall_neighborhood = conv(
        wall, np.ones((1, 1, 3, 3), np.float32), "dilate"
    )
    passable = sub(const(1), step(wall_neighborhood, 0.5, True), True)
    goal_seed = scale(targets, 128, True)

    field_shift_w = np.zeros((4, 1, 7, 7), np.float32)
    for d, (xx, yy) in enumerate(directions):
        field_shift_w[d, 0, 3 + 3 * yy, 3 + 3 * xx] = 1
    parameter("field_shift", field_shift_w)

    body = []
    current = body
    neighbors = emit("conv", ["distance_state"], w="field_shift")
    best_neighbor = sl(neighbors, 0, 1, True)
    for i in range(1, 4):
        best_neighbor = emit("max2", [
            best_neighbor, sl(neighbors, i, i + 1, True)
        ])
    field_next = mul(emit("max2", [
        relu(sub(best_neighbor, const(1), True), True), goal_seed
    ]), passable, True)
    current = graph
    emit("loop", name="distance_field", T=48,
         state=[{"id": "distance_state", "init": goal_seed}],
         body=body, next={"distance_state": field_next})

    def bounds(a, b):
        difference = sub(a, b)
        high = add(b, relu(difference))
        low = sub(a, relu(difference))
        return sub(low, const(1 / 3)), add(high, const(1 / 3))

    def rasterize(xx, yy, tip_x, tip_y):
        lx, hx = bounds(xx, tip_x)
        ly, hy = bounds(yy, tip_y)
        outside = total([
            relu(sub(lx, gx, True), True),
            relu(sub(gx, hx, True), True),
            relu(sub(ly, gy, True), True),
            relu(sub(gy, hy, True), True)
        ], True)
        rectangles = mul(
            step(neg(outside, True), -0.01, True), present, True
        )
        return rectangles, lx, hx, ly, hy

    current_rectangles, _, _, _, _ = rasterize(bx, by, tx, ty)
    current_occupancy = conv(
        current_rectangles,
        np.ones((1, N, 1, 1), np.float32), "occupancy_sum"
    )

    # Per-marker free space ignores its own attachment chain. Value can
    # propagate along a horizontal or vertical extension corridor.
    ray_passes = []
    for m in range(MARKERS):
        own_occupancy = conv(
            mul(current_rectangles, marker_owner[m], True),
            np.ones((1, N, 1, 1), np.float32), "occupancy_sum"
        )
        obstacles = add(
            relu(sub(current_occupancy, own_occupancy, True), True),
            wall, True
        )
        dilated = conv(
            obstacles, np.ones((1, 1, 3, 3), np.float32), "dilate"
        )
        ray_passes.append(
            sub(const(1), step(dilated, 0.5, True), True)
        )
    ray_passes = cat(ray_passes, True)
    ray_passes = cat([ray_passes, ray_passes], True)
    ray_drive = mul(
        scale("distance_field.distance_state", 2, True),
        ray_passes, True
    )

    R = 2 * MARKERS
    ray_shift_w = np.zeros((2 * R, R, 3, 3), np.float32)
    for c in range(R):
        for sign_index, sign in enumerate((-1, 1)):
            o = sign_index * R + c
            if c < MARKERS:
                ray_shift_w[o, c, 1, 1 + sign] = 1
            else:
                ray_shift_w[o, c, 1 + sign, 1] = 1
    parameter("ray_shift", ray_shift_w)

    body = []
    current = body
    ray_neighbors = emit("conv", ["ray_state"], w="ray_shift")
    ray_best = emit("max2", [
        sl(ray_neighbors, 0, R, True),
        sl(ray_neighbors, R, 2 * R, True)
    ])
    ray_next = mul(emit("max2", [
        relu(sub(ray_best, const(1 / 6), True), True),
        ray_drive
    ]), ray_passes, True)
    current = graph
    emit("loop", name="extension_field", T=48,
         state=[{"id": "ray_state", "init": ray_drive}],
         body=body, next={"ray_state": ray_next})

    # Symmetric configuration fingerprint.
    for k in range(5):
        values = []
        for i in range(N):
            c = i % C
            value = 0.11 + 0.07 * math.sqrt(2 + c + 3 * k)
            if k >= 3 and c == carrier_index:
                value = 0
            values.append(value)
        parameter("hash_%d" % k, np.array([values], np.float32))

    def fingerprint(xx, yy, ll, ddx, ddy):
        span = sub(ll, const(1))
        centre_x = add(xx, scale(mul(span, ddx), 0.5))
        centre_y = add(yy, scale(mul(span, ddy), 0.5))
        terms = []
        for k, value in enumerate((centre_x, centre_y, ll, ddx, ddy)):
            terms.append(emit("dense", [value], w="hash_%d" % k))
        return total(terms)

    current_hash = fingerprint(bx, by, length, vx, vy)
    previous_hash = sl("mem", 0, 1)
    previous_choice = sl("mem", 1, 1 + K)
    previous_failed_hashes = sl("mem", 1 + K, 1 + 2 * K)
    history = sl("mem", 1 + 2 * K, 1 + 2 * K + HISTORY)

    unchanged = equal(current_hash, previous_hash, 0.003)
    failed_update = mul(previous_choice, unchanged)
    failed_hashes = add(
        mul(previous_failed_hashes, sub(const(1), failed_update)),
        mul(failed_update, current_hash)
    )
    known_failed = equal(current_hash, failed_hashes, 0.003)

    pbx, pby = pg(bx), pg(by)
    cbx, cby = cg(bx), cg(by)
    cvx, cvy = cg(vx), cg(vy)
    quarter_x = total([pbx, neg(pby), cby, neg(cbx)])
    quarter_y = total([pby, pbx, neg(cbx), neg(cby)])
    half_x = scale(sub(pbx, cbx), 2)
    half_y = scale(sub(pby, cby), 2)
    direction_qx = sub(cvy, cvx)
    direction_qy = neg(add(cvx, cvy))
    direction_hx = scale(cvx, -2)
    direction_hy = scale(cvy, -2)

    marker_coefficients = []
    for m in range(MARKERS):
        marker_coefficients.append((
            total([bx, neg(by), marker_y[m], neg(marker_x[m])]),
            total([by, bx, neg(marker_x[m]), neg(marker_y[m])]),
            scale(sub(bx, marker_x[m]), 2),
            scale(sub(by, marker_y[m]), 2)
        ))

    selector_down = np.zeros((N, K), np.float32)
    selector_up = selector_down.copy()
    selector_rotate = selector_down.copy()
    for i in range(N):
        color = colors[i % C]
        if color in control_colors:
            c = control_colors.index(color)
            selector_down[i, c] = 1
            selector_up[i, A + c] = 1
            selector_rotate[i, 2 * A + c] = 1
    parameter("select_down", selector_down)
    parameter("select_up", selector_up)
    parameter("select_rotate", selector_rotate)

    initial_action = np.zeros(K, np.float32)
    initial_action[0] = 1
    action_init = vector_constant(initial_action)
    action_shift = np.zeros((K, K), np.float32)
    for i in range(K):
        action_shift[(i + 1) % K, i] = 1
    parameter("action_shift", action_shift)
    index_bias = vector_constant(-0.005 * np.arange(K, dtype=np.float32))
    marker_presence = cat(marker_exists)

    # Evaluate each colour/control pair analytically.
    body = []
    current = body
    down = emit("dense", ["candidate_action"], w="select_down")
    up = emit("dense", ["candidate_action"], w="select_up")
    rotate = emit("dense", ["candidate_action"], w="select_rotate")

    # Crucially, absent object slots cannot change.
    delta_length = mul(sub(up, mul(down, shrinkable)), present)
    rotate = mul(rotate, present)
    rq = mul(rotate, sub(const(1), skip_quarter))
    rh = mul(rotate, skip_quarter)
    ddx, ddy = mul(delta_length, vx), mul(delta_length, vy)
    rotation_q = mul(pg(rq), reach)
    rotation_h = mul(pg(rh), reach)

    next_x = total([
        bx, cs(mul(pg(ddx), strict_reach)),
        cs(mul(rotation_q, quarter_x)), cs(mul(rotation_h, half_x))
    ])
    next_y = total([
        by, cs(mul(pg(ddy), strict_reach)),
        cs(mul(rotation_q, quarter_y)), cs(mul(rotation_h, half_y))
    ])
    next_dx = total([
        vx, cs(mul(rotation_q, direction_qx)),
        cs(mul(rotation_h, direction_hx))
    ])
    next_dy = total([
        vy, cs(mul(rotation_q, direction_qy)),
        cs(mul(rotation_h, direction_hy))
    ])
    next_length = add(length, delta_length)
    next_tip_x = add(next_x, mul(sub(next_length, const(1)), next_dx))
    next_tip_y = add(next_y, mul(sub(next_length, const(1)), next_dy))

    rectangles, lx, hx, ly, hy = rasterize(
        next_x, next_y, next_tip_x, next_tip_y
    )
    occupancy = conv(
        rectangles, np.ones((1, N, 1, 1), np.float32), "occupancy_sum"
    )
    arm_collision = emit("gmax", [
        relu(sub(occupancy, const(1), True), True)
    ])
    wall_collision = emit("gmax", [mul(occupancy, wall, True)])
    boundary_violation = sum_n(mul(total([
        relu(neg(lx)), relu(neg(ly)),
        relu(sub(hx, const(21))), relu(sub(hy, const(21)))
    ]), present))
    collision = total([
        arm_collision, wall_collision, boundary_violation
    ])

    predicted_x, predicted_y = [], []
    horizontal, vertical = [], []
    for m in range(MARKERS):
        own = marker_owner[m]
        mq, mh = mul(rq, own), mul(rh, own)
        qx, qy, hx0, hy0 = marker_coefficients[m]
        predicted_x.append(total([
            marker_x[m], sum_n(mul(ddx, own)),
            sum_n(mul(mq, qx)), sum_n(mul(mh, hx0))
        ]))
        predicted_y.append(total([
            marker_y[m], sum_n(mul(ddy, own)),
            sum_n(mul(mq, qy)), sum_n(mul(mh, hy0))
        ]))
        horizontal.append(absolute(sum_n(mul(next_dx, direct[m]))))
        vertical.append(absolute(sum_n(mul(next_dy, direct[m]))))

    mx, my = cat(predicted_x), cat(predicted_y)
    point_distance = add(
        absolute(sub(gx, mx, True), True),
        absolute(sub(gy, my, True), True), True
    )
    points = mul(
        step(neg(point_distance, True), -0.01, True),
        marker_presence, True
    )
    marker_values = emit("gmax", [
        mul(points, "distance_field.distance_state", True)
    ])
    basic_potential = dense(
        marker_values, np.ones((1, MARKERS), np.float32),
        "sum_marker_values"
    )
    extension_values = emit("gmax", [
        mul(cat([points, points], True),
            "extension_field.ray_state", True)
    ])
    extension_potential = dense(
        mul(extension_values, cat(horizontal + vertical)),
        np.ones((1, R), np.float32), "sum_extension_values"
    )
    potential = add(
        scale(basic_potential, 0.2),
        scale(extension_potential, 0.4)
    )

    point_union = emit("min2", [
        conv(points, np.ones((1, MARKERS, 1, 1), np.float32),
             "point_union_sum"),
        const(1)
    ])
    coverage = scale(emit("gmean", [
        mul(point_union, targets, True)
    ]), 4096)

    future_hash = fingerprint(
        next_x, next_y, next_length, next_dx, next_dy
    )
    revisit_count = dense(
        equal(future_hash, history, 0.003),
        np.ones((1, HISTORY), np.float32), "count_revisits"
    )
    no_change = equal(future_hash, current_hash, 0.003)
    unavailable = sub(const(1), dense(
        mul("candidate_action", button_available),
        np.ones((1, K), np.float32), "sum_actions"
    ))
    failed = dense(
        mul("candidate_action", known_failed),
        np.ones((1, K), np.float32), "sum_actions"
    )

    score = total([
        potential,
        scale(coverage, 20),
        scale(sum_n(next_length), -0.03),
        scale(collision, -10000),
        scale(revisit_count, -8),
        scale(no_change, -2000),
        scale(unavailable, -100000),
        scale(failed, -20000)
    ])
    written_scores = add(
        "candidate_scores", mul("candidate_action", score)
    )
    next_action = emit("dense", ["candidate_action"], w="action_shift")
    current = graph
    emit("loop", name="evaluate_controls", T=K,
         state=[
             {"id": "candidate_action", "init": action_init},
             {"id": "candidate_scores", "zeros": [K]}
         ],
         body=body,
         next={
             "candidate_action": next_action,
             "candidate_scores": written_scores
         })

    scores = add("evaluate_controls.candidate_scores", index_bias)
    maximum = vector_max(scores, K)
    chosen_action = equal(scores, maximum, 0.0003)

    click_raw = conv(
        mul(click_maps, scores, True),
        np.ones((1, K, 1, 1), np.float32), "click_reduce"
    )
    mapped = conv(
        click_maps, np.ones((1, K, 1, 1), np.float32), "click_reduce"
    )
    blank = relu(sub(const(1), mapped, True), True)
    emit("sub", [click_raw, scale(blank, 10000000, True)], name="click")

    dense(
        "avail", np.zeros((8, 8), np.float32), "action_logits",
        bias=np.array([0, -1, -1, -1, -1, -1, 10, -1], np.float32),
        out="logits"
    )
    next_history = cat([current_hash, sl(history, 0, HISTORY - 1)])
    emit("vconcat", [
        current_hash, chosen_action, failed_hashes, next_history
    ], name="mem_next")

    # Static compilation: move loop-invariant nodes outside their loops.
    known = {"x0", "coords", "mem", "aprev", "avail"}
    compiled = []
    for node in graph:
        if node["op"] != "loop":
            compiled.append(node)
            known.add(node["id"])
            continue
        loop_body = []
        for inner in node["body"]:
            if all(x in known for x in inner.get("in", [])):
                compiled.append(inner)
                known.add(inner["id"])
            else:
                loop_body.append(inner)
        copied = dict(node)
        copied["body"] = loop_body
        compiled.append(copied)
        known.update(
            copied["id"] + "." + s["id"] for s in copied["state"]
        )
    graph = compiled

    def prune(nodes, needed):
        kept = []
        for node in reversed(nodes):
            if node["op"] == "loop":
                prefix = node["id"] + "."
                if not any(x.startswith(prefix) for x in needed):
                    continue
                new_body, dependencies = prune(
                    node["body"], set(node["next"].values())
                )
                copied = dict(node)
                copied["body"] = new_body
                state_ids = {s["id"] for s in copied["state"]}
                needed.update(dependencies - state_ids)
                for state in copied["state"]:
                    if "init" in state:
                        needed.add(state["init"])
                kept.append(copied)
            elif node["id"] in needed:
                needed.update(node.get("in", []))
                kept.append(node)
        kept.reverse()
        defined = {n["id"] for n in kept}
        for node in kept:
            if node["op"] == "loop":
                defined.update(
                    node["id"] + "." + s["id"] for s in node["state"]
                )
        return kept, needed - defined

    graph, _ = prune(graph, {"logits", "click", "mem_next"})

    # Bound execution count using only the static graph structure.
    def evaluations(nodes):
        return sum(
            1 + n["T"] * evaluations(n["body"])
            if n["op"] == "loop" else 1
            for n in nodes
        )

    for loop_name in ("extension_field", "distance_field", "axis_flood"):
        count = evaluations(graph)
        if count <= 9900:
            break
        for node in graph:
            if node["id"] == loop_name:
                cost = evaluations(node["body"])
                reduction = (count - 9900 + cost - 1) // cost
                node["T"] = max(1, node["T"] - reduction)
                break

    used = set()

    def collect(nodes):
        for node in nodes:
            for field in ("w", "b"):
                if field in node:
                    used.add(node[field])
            if node["op"] == "loop":
                collect(node["body"])

    collect(graph)
    params = {name: value for name, value in params.items() if name in used}

    return {
        "graph": graph,
        "memory_size": 1 + 2 * K + HISTORY,
        "params": params
    }
