import numpy as np
import math


def build():
    G, P = [], {}
    serial = [0]
    constants, sums = {}, {}

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        key = fresh("p")
        P[key] = np.asarray(a, dtype=np.float32)
        return key

    def node(kind, inputs, B=None, **fields):
        key = fresh()
        nd = {"id": key, "op": kind, "in": list(inputs)}
        nd.update(fields)
        (G if B is None else B).append(nd)
        return key

    def conv(a, w, b=None, B=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [a], B, **fields)

    def dense(a, w, b=None, B=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [a], B, **fields)

    def linear(a, w, B=None):
        return node("dense", [a], B, w=w)

    def sv(a, start, stop, B=None):
        return node("vslice", [a], B, start=start, stop=stop)

    def ss(a, start, stop, B=None):
        return node("slice", [a], B, start=start, stop=stop)

    def vc(xs, B=None):
        return node("vconcat", xs, B)

    def sc(xs, B=None):
        return node("concat", xs, B)

    def K(value):
        value = float(value)
        if value not in constants:
            constants[value] = dense(
                "aprev", np.zeros((1, 8), np.float32), [value]
            )
        return constants[value]

    def add(a, b, B=None):
        return node("vadd", [a, b], B)

    def mul(a, b, B=None):
        return node("vmul", [a, b], B)

    def scale(a, factor, B=None):
        return mul(a, K(factor), B)

    def neg(a, B=None):
        return scale(a, -1, B)

    def sub(a, b, B=None):
        return add(a, neg(b, B), B)

    def ab(a, B=None):
        return add(
            node("vrelu", [a], B),
            node("vrelu", [neg(a, B)], B), B
        )

    def vmax(a, b, B=None):
        return add(b, node("vrelu", [sub(a, b, B)], B), B)

    def vmin(a, b, B=None):
        return sub(a, node("vrelu", [sub(a, b, B)], B), B)

    def boolean(a, B=None):
        return node(
            "vtanh", [scale(node("vrelu", [a], B), 100, B)], B
        )

    def total(a, length, B=None):
        if length not in sums:
            sums[length] = parameter(
                np.ones((1, length), dtype=np.float32)
            )
        return linear(a, sums[length], B)

    def affine_s(a, factor, bias=0, B=None):
        result = node("mul", [a, K(factor)], B)
        if bias:
            result = node("add", [result, K(bias)], B)
        return result

    x = conv(
        "coords", np.array([[[[0]], [[63]], [[0]]]], np.float32)
    )
    y = conv(
        "coords", np.array([[[[63]], [[0]], [[0]]]], np.float32)
    )
    position = conv(
        "coords",
        np.array([[[[4032]], [[63]], [[0]]]], np.float32), [1]
    )
    one = conv("coords", np.zeros((1, 3, 1, 1), np.float32), [1])

    colors = [0, 6, 7, 8, 9, 10, 11, 12, 14]
    C = len(colors)
    duplicate_color = colors.index(12)
    N = 14
    Q = 3 * C
    H = 40

    border = ss("x0", 2, 3)
    purple = ss("x0", 13, 14)
    wall = ss("x0", 15, 16)

    ui = conv(border, np.ones((1, 1, 7, 7), np.float32))
    ui = node("min2", [ui, one])
    not_ui = node("sub", [one, ui])

    cw = np.zeros((C, 16, 1, 1), np.float32)
    for j, color in enumerate(colors):
        cw[j, color, 0, 0] = 1
    color_planes = conv("x0", cw)
    bare = node("mul", [color_planes, not_ui])

    # Resize and rotation button detectors.
    bw = np.zeros((3 * C, 16, 7, 7), np.float32)
    bb = np.zeros(3 * C, np.float32)
    for j, color in enumerate(colors):
        bw[j, 3, 3, 3] = 1
        for dy in (-1, 1):
            for dx in (-3, 3):
                bw[j, color, 3 + dy, 3 + dx] = 1
        bb[j] = -4

        k = C + j
        bw[k, 3, 3, 3] = 1
        for dy in (-3, 3):
            for dx in (-1, 1):
                bw[k, color, 3 + dy, 3 + dx] = 1
        bb[k] = -4

        k = 2 * C + j
        bw[k, color, 3, 3] = 1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            bw[k, color, 3 + dy, 3 + dx] = 1
        for dy in (-2, 2):
            for dx in (-2, 2):
                bw[k, 2, 3 + dy, 3 + dx] = 1
        bb[k] = -8

    buttons = node("relu", [conv("x0", bw, bb)])
    horizontal = node("mul", [ss(buttons, 0, C), ui])
    vertical = node("mul", [ss(buttons, C, 2 * C), ui])
    rotation_buttons = node(
        "mul", [ss(buttons, 2 * C, 3 * C), ui]
    )

    def shifted(a, dy, dx):
        w = np.zeros((C, C, 7, 7), np.float32)
        for j in range(C):
            w[j, j, 3 - dy, 3 - dx] = 1
        return conv(a, w)

    decrease_map = node("add", [
        shifted(horizontal, 0, -3), shifted(vertical, -3, 0)
    ])
    increase_map = node("add", [
        shifted(horizontal, 0, 3), shifted(vertical, 3, 0)
    ])
    controls = sc([decrease_map, increase_map, rotation_buttons])
    control_present = node("gmax", [controls])

    # Cap orientation order: bottom, left, top, right.
    capw = np.zeros((4 * C, 16, 3, 3), np.float32)
    capb = np.full(4 * C, -5, np.float32)
    for direction in range(4):
        for j, color in enumerate(colors):
            k = direction * C + j
            capw[k, color, 1, 1] = 1
            capw[k, 13, 1, 1] = 1
            if direction == 0:
                cap_pixels = [(1, -1), (1, 0), (1, 1)]
                color_pixels = [(-1, -1), (-1, 1)]
            elif direction == 1:
                cap_pixels = [(-1, -1), (0, -1), (1, -1)]
                color_pixels = [(-1, 1), (1, 1)]
            elif direction == 2:
                cap_pixels = [(-1, -1), (-1, 0), (-1, 1)]
                color_pixels = [(1, -1), (1, 1)]
            else:
                cap_pixels = [(-1, 1), (0, 1), (1, 1)]
                color_pixels = [(-1, -1), (1, -1)]
            for dy, dx in cap_pixels:
                capw[k, 3, 1 + dy, 1 + dx] += 1
            for dy, dx in color_pixels:
                capw[k, color, 1 + dy, 1 + dx] += 1

    caps = node("mul", [
        node("relu", [conv("x0", capw, capb)]), not_ui
    ])

    def pick_points(mask):
        peak = node("gmax", [node("mul", [mask, position])])
        gap = node("abs", [node("sub", [position, peak])])
        chosen = node("relu", [affine_s(gap, -2, 1)])
        chosen = node("mul", [chosen, mask])
        px = node("gmax", [node("mul", [chosen, x])])
        py = node("gmax", [node("mul", [chosen, y])])
        present = node("gmax", [chosen])
        return chosen, px, py, present

    def parse_active(capfield, colorfield, count):
        collapse = np.zeros((count, 4 * count, 1, 1), np.float32)
        repeat = np.zeros((4 * count, count, 1, 1), np.float32)
        for direction in range(4):
            for j in range(count):
                collapse[j, direction * count + j, 0, 0] = 1
                repeat[direction * count + j, j, 0, 0] = 1
        mask = conv(capfield, collapse)
        chosen, px, py, present = pick_points(mask)
        orientations = node("gmax", [
            node("mul", [capfield, conv(chosen, repeat)])
        ])

        wx = np.zeros((count, 4 * count), np.float32)
        wy = np.zeros_like(wx)
        for j in range(count):
            wx[j, count + j] = 1
            wx[j, 3 * count + j] = -1
            wy[j, j] = -1
            wy[j, 2 * count + j] = 1
        dx = dense(orientations, wx)
        dy = dense(orientations, wy)
        ax, ay = ab(dx), ab(dy)

        gx = node("abs", [node("sub", [x, px])])
        gy = node("abs", [node("sub", [y, py])])
        orthogonal = node("add", [
            node("mul", [gx, ay]), node("mul", [gy, ax])
        ])
        strip = node("relu", [affine_s(orthogonal, -1, 2)])
        strip = node("min2", [strip, one])
        strip = node("mul", [strip, colorfield])

        projection = node("add", [
            node("mul", [x, dx]), node("mul", [y, dy])
        ])
        projection = node("add", [projection, K(128)])
        far = node("gmax", [node("mul", [projection, strip])])
        pp = add(mul(px, dx), mul(py, dy))
        length = mul(add(sub(far, pp), K(-126)), present)
        extension = add(length, K(-3))
        width = mul(add(mul(ax, extension), K(3)), present)
        height = mul(add(mul(ay, extension), K(3)), present)
        cx = mul(add(px, mul(dx, scale(extension, 0.5))), present)
        cy = mul(add(py, mul(dy, scale(extension, 0.5))), present)
        return {
            "chosen": chosen, "px": px, "py": py,
            "dx": dx, "dy": dy, "length": length,
            "cx": cx, "cy": cy, "w": width, "h": height,
            "present": present
        }

    active = parse_active(caps, bare, C)

    secondw = np.zeros((4, 4 * C, 1, 1), np.float32)
    for d in range(4):
        secondw[d, d * C + duplicate_color, 0, 0] = 1
    second_caps = conv(caps, secondw)
    first_duplicate = ss(
        active["chosen"], duplicate_color, duplicate_color + 1
    )
    second_caps = node("mul", [
        second_caps, node("sub", [one, first_duplicate])
    ])
    second_active = parse_active(
        second_caps, ss(bare, duplicate_color, duplicate_color + 1), 1
    )

    # Passive colour-1 rods.
    blue = node("mul", [ss("x0", 1, 2), not_ui])
    cornerw = np.zeros((1, 1, 5, 5), np.float32)
    cornerw[0, 0, 1:4, 1:4] = 1
    cornerw[0, 0, 0, 2] = -1
    cornerw[0, 0, 2, 0] = -1
    blue_corners = node("relu", [conv(blue, cornerw, [-8])])

    def parse_passive(mask):
        chosen, px, py, present = pick_points(mask)
        gx = node("abs", [node("sub", [x, px])])
        gy = node("abs", [node("sub", [y, py])])
        vertical_strip = node("min2", [
            node("relu", [affine_s(gx, -1, 2)]), one
        ])
        horizontal_strip = node("min2", [
            node("relu", [affine_s(gy, -1, 2)]), one
        ])
        xmax = node("gmax", [
            node("mul", [node("mul", [blue, horizontal_strip]), x])
        ])
        ymax = node("gmax", [
            node("mul", [node("mul", [blue, vertical_strip]), y])
        ])
        width = mul(add(sub(xmax, px), K(2)), present)
        height = mul(add(sub(ymax, py), K(2)), present)
        cx = mul(add(px, scale(add(width, K(-3)), 0.5)), present)
        cy = mul(add(py, scale(add(height, K(-3)), 0.5)), present)
        return {
            "chosen": chosen, "cx": cx, "cy": cy,
            "w": width, "h": height, "present": present
        }

    passive1 = parse_passive(blue_corners)
    passive2 = parse_passive(node("mul", [
        blue_corners, node("sub", [one, passive1["chosen"]])
    ]))

    # Target petals have diagonal purple neighbours; marker centres
    # do not, including when a marker is centred on a target.
    markerw = np.zeros((1, 1, 3, 3), np.float32)
    markerw[0, 0, 1, 1] = 1
    for dy in (-1, 1):
        for dx in (-1, 1):
            markerw[0, 0, 1 + dy, 1 + dx] = -1
    markers = node("relu", [conv(purple, markerw)])

    targetw = np.zeros((1, 1, 3, 3), np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        targetw[0, 0, 1 + dy, 1 + dx] = 1
    targets = node("relu", [conv(purple, targetw, [-3])])

    marker1, mx1, my1, mp1 = pick_points(markers)
    marker2, mx2, my2, mp2 = pick_points(node("mul", [
        markers, node("sub", [one, marker1])
    ]))
    target1, _, _, _ = pick_points(targets)
    target2, _, _, _ = pick_points(node("mul", [
        targets, node("sub", [one, target1])
    ]))

    zeros4 = vc([K(0)] * 4)
    cx = vc([
        active["cx"], second_active["cx"],
        passive1["cx"], passive2["cx"], mx1, mx2
    ])
    cy = vc([
        active["cy"], second_active["cy"],
        passive1["cy"], passive2["cy"], my1, my2
    ])
    widths = vc([
        active["w"], second_active["w"],
        passive1["w"], passive2["w"], scale(mp1, 3), scale(mp2, 3)
    ])
    heights = vc([
        active["h"], second_active["h"],
        passive1["h"], passive2["h"], scale(mp1, 3), scale(mp2, 3)
    ])
    presence = vc([
        active["present"], second_active["present"],
        passive1["present"], passive2["present"], mp1, mp2
    ])
    actuated = vc([
        active["present"], second_active["present"], zeros4
    ])
    collidable = vc([
        active["present"], second_active["present"],
        passive1["present"], passive2["present"], K(0), K(0)
    ])
    pivot_x = vc([active["px"], second_active["px"], zeros4])
    pivot_y = vc([active["py"], second_active["py"], zeros4])
    dir_x = vc([active["dx"], second_active["dx"], zeros4])
    dir_y = vc([active["dy"], second_active["dy"], zeros4])
    lengths = vc([active["length"], second_active["length"], zeros4])

    # Flattened parent-child matrix operators.
    er = np.zeros((N * N, N), np.float32)
    ec = np.zeros_like(er)
    sr = np.zeros((N, N * N), np.float32)
    scol = np.zeros_like(sr)
    for i in range(N):
        for j in range(N):
            k = i * N + j
            er[k, i] = 1
            ec[k, j] = 1
            sr[i, k] = 1
            scol[j, k] = 1
    Wer, Wec, Wsr, Wsc = [
        parameter(a) for a in (er, ec, sr, scol)
    ]

    def row(a, B=None):
        return linear(a, Wer, B)

    def col(a, B=None):
        return linear(a, Wec, B)

    def row_total(a, B=None):
        return linear(a, Wsr, B)

    def col_total(a, B=None):
        return linear(a, Wsc, B)

    def inside(gx, gy, B=None):
        return boolean(add(vmin(gx, gy, B), K(-0.2), B), B)

    attach_x = add(pivot_x, mul(dir_x, lengths))
    attach_y = add(pivot_y, mul(dir_y, lengths))
    gx = sub(scale(col(widths), 0.5), ab(sub(row(attach_x), col(cx))))
    gy = sub(scale(col(heights), 0.5), ab(sub(row(attach_y), col(cy))))
    touching_endpoint = inside(gx, gy)

    gx = sub(scale(row(widths), 0.5), ab(sub(row(cx), col(cx))))
    gy = sub(scale(row(heights), 0.5), ab(sub(row(cy), col(cy))))
    contained = inside(gx, gy)

    marker_columns_array = np.zeros(N * N, np.float32)
    for i in range(N):
        marker_columns_array[i * N + 12:i * N + 14] = 1
    marker_columns = dense(
        "aprev", np.zeros((N * N, 8), np.float32),
        marker_columns_array
    )
    adjacency = add(
        mul(touching_endpoint, sub(K(1), marker_columns)),
        mul(contained, marker_columns)
    )
    adjacency = mul(adjacency, mul(row(actuated), col(presence)))

    parent_dx = col_total(mul(adjacency, row(dir_x)))
    parent_dy = col_total(mul(adjacency, row(dir_y)))
    extra_turn = boolean(add(
        sub(mul(dir_x, parent_dy), mul(dir_y, parent_dx)), K(-0.5)
    ))
    extra_turn = mul(extra_turn, actuated)

    identity = dense(
        "aprev", np.zeros((N * N, 8), np.float32),
        np.eye(N, dtype=np.float32).reshape(-1)
    )
    reach_initial = add(adjacency, identity)
    counter_initial = dense(
        "aprev", np.zeros((N, 8), np.float32),
        np.eye(N, dtype=np.float32)[0]
    )
    counter_shift = np.zeros((N, N), np.float32)
    for i in range(N - 1):
        counter_shift[i + 1, i] = 1
    Wcounter = parameter(counter_shift)

    closure_body = []
    sa, sk = fresh("s"), fresh("s")
    aik = row_total(
        mul(sa, col(sk, closure_body), closure_body), closure_body
    )
    akj = col_total(
        mul(sa, row(sk, closure_body), closure_body), closure_body
    )
    path = mul(
        row(aik, closure_body), col(akj, closure_body), closure_body
    )
    anext = boolean(add(sa, path, closure_body), closure_body)
    knext = linear(sk, Wcounter, closure_body)
    closure_id = fresh("loop")
    G.append({
        "id": closure_id, "op": "loop", "T": N,
        "state": [
            {"id": sa, "init": reach_initial},
            {"id": sk, "init": counter_initial}
        ],
        "body": closure_body, "next": {sa: anext, sk: knext}
    })
    reach = closure_id + "." + sa

    # Static-obstacle distance fields on the three-pixel lattice.
    inflated = conv(wall, np.ones((1, 1, 3, 3), np.float32))
    inflated = node("min2", [inflated, one])
    free_value = node("mul", [
        node("sub", [one, inflated]), K(128)
    ])
    goals = node("mul", [sc([target1, target2]), K(128)])

    navw = np.zeros((8, 2, 7, 7), np.float32)
    for d, (dy, dx) in enumerate(
        [(-3, 0), (3, 0), (0, -3), (0, 3)]
    ):
        for c in range(2):
            navw[2 * d + c, c, 3 + dy, 3 + dx] = 1
    Wnav = parameter(navw)
    Wdecay = parameter(
        np.eye(2, dtype=np.float32).reshape(2, 2, 1, 1)
    )
    Bdecay = parameter([-1, -1])

    nav_body = []
    sd = fresh("s")
    neighbors = node("conv", [sd], nav_body, w=Wnav)
    near = ss(neighbors, 0, 2, nav_body)
    for i in range(1, 4):
        near = node("max2", [
            near, ss(neighbors, 2 * i, 2 * i + 2, nav_body)
        ], nav_body)
    near = node("conv", [near], nav_body, w=Wdecay, b=Bdecay)
    near = node("relu", [near], nav_body)
    near = node("max2", [near, sd], nav_body)
    near = node("min2", [near, free_value], nav_body)
    nav_id = fresh("loop")
    G.append({
        "id": nav_id, "op": "loop", "T": 96,
        "state": [{"id": sd, "init": goals}],
        "body": nav_body, "next": {sd: near}
    })
    distance_fields = nav_id + "." + sd

    upper_array = np.zeros(N * N, np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            upper_array[i * N + j] = 1
    upper = dense(
        "aprev", np.zeros((N * N, 8), np.float32), upper_array
    )
    pair_mask = mul(mul(row(collidable), col(collidable)), upper)

    # Four geometric fingerprints.
    fw = np.empty((4, 4 * N), np.float32)
    for r in range(4):
        for j in range(4 * N):
            fw[r, j] = (
                0.012 * math.sin((r + 1.37) * (j + 1.11))
                + 0.007 * math.cos((r + 2.19) * (j + 0.73))
            )
    Wfinger = parameter(fw)
    Bfinger = parameter(np.full(4, 10, np.float32))

    def fingerprint(a, b, c, d, B=None):
        return node(
            "dense", [vc([a, b, c, d], B)], B,
            w=Wfinger, b=Bfinger
        )

    current_hash = fingerprint(cx, cy, widths, heights)

    # Memory: 40 records of [hash0..hash3, action_id],
    # followed by 27 failed-control flags and the last control one-hot.
    old_hash = sv("mem", 0, 4)
    old_bad = sv("mem", 200, 227)
    old_action = sv("mem", 227, 254)
    same_distance = total(ab(sub(current_hash, old_hash)), 4)
    same_state = boolean(add(K(0.005), neg(same_distance)))
    bad_controls = mul(
        same_state, vmin(add(old_bad, old_action), K(1))
    )

    extract_hash = np.zeros((4 * H, 254), np.float32)
    repeat_hash = np.zeros((4 * H, 4), np.float32)
    sum_hash = np.zeros((H, 4 * H), np.float32)
    extract_action = np.zeros((H, 254), np.float32)
    for i in range(H):
        for j in range(4):
            extract_hash[4 * i + j, 5 * i + j] = 1
            repeat_hash[4 * i + j, j] = 1
            sum_hash[i, 4 * i + j] = 1
        extract_action[i, 5 * i + 4] = 1

    old_hashes = dense("mem", extract_hash)
    repeated_current = dense(current_hash, repeat_hash)
    context_difference = ab(sub(repeated_current, old_hashes))
    context_distance = dense(context_difference, sum_hash)
    matching_context = boolean(
        add(K(0.005), neg(context_distance))
    )
    past_action_ids = dense("mem", extract_action)

    action_numbers = np.arange(1, Q + 1, dtype=np.float32)[None, :]
    Waction_number = parameter(action_numbers)

    root_selector = np.zeros((N, Q), np.float32)
    for q in range(Q):
        c = q % C
        root_selector[c, q] = 1
        if c == duplicate_color:
            root_selector[9, q] = 1
    Wroot = parameter(root_selector)
    Wsign = parameter(np.array(
        [[-1] * C + [1] * C + [0] * C], np.float32
    ))
    Wrotate = parameter(np.array(
        [[0] * (2 * C) + [1] * C], np.float32
    ))
    Wdecrease = parameter(np.array(
        [[1] * C + [0] * (2 * C)], np.float32
    ))
    qinitial = dense(
        "aprev", np.zeros((Q, 8), np.float32),
        np.eye(Q, dtype=np.float32)[0]
    )
    qshift = np.zeros((Q, Q), np.float32)
    for i in range(Q - 1):
        qshift[i + 1, i] = 1
    Wqshift = parameter(qshift)

    can_shrink = boolean(add(lengths, K(-3.5)))
    root_ax, root_ay = ab(dir_x), ab(dir_y)
    initial_click = conv(
        "coords", np.zeros((1, 3, 1, 1), np.float32), [-1e6]
    )
    Wcontrol_sum = parameter(np.ones((1, Q, 1, 1), np.float32))

    B = []
    sq, sbest = fresh("s"), fresh("s")
    selected = mul(linear(sq, Wroot, B), actuated, B)
    sign = linear(sq, Wsign, B)
    rotate = linear(sq, Wrotate, B)
    decrease = linear(sq, Wdecrease, B)
    shrink_gate = add(
        sub(K(1), decrease, B), mul(decrease, can_shrink, B), B
    )
    roots = mul(selected, shrink_gate, B)

    def propagate(v):
        return col_total(mul(reach, row(v, B), B), B)

    rdx, rdy = mul(roots, dir_x, B), mul(roots, dir_y, B)
    all_dx, all_dy = propagate(rdx), propagate(rdy)
    step_size = scale(sign, 3, B)

    resize_x = add(cx, mul(
        sub(all_dx, scale(rdx, 0.5, B), B), step_size, B
    ), B)
    resize_y = add(cy, mul(
        sub(all_dy, scale(rdy, 0.5, B), B), step_size, B
    ), B)
    resize_w = add(
        widths, mul(mul(roots, root_ax, B), step_size, B), B
    )
    resize_h = add(
        heights, mul(mul(roots, root_ay, B), step_size, B), B
    )

    affected = boolean(propagate(roots), B)
    px = propagate(mul(roots, pivot_x, B))
    py = propagate(mul(roots, pivot_y, B))
    double_turn = boolean(
        total(mul(roots, extra_turn, B), N, B), B
    )

    x90 = add(add(
        cx, mul(affected, sub(cy, cx, B), B), B
    ), sub(px, py, B), B)
    y90 = add(add(
        cy, mul(affected, neg(add(cx, cy, B), B), B), B
    ), add(px, py, B), B)
    w90 = add(
        widths, mul(affected, sub(heights, widths, B), B), B
    )
    h90 = add(
        heights, mul(affected, sub(widths, heights, B), B), B
    )
    x180 = add(cx, scale(
        sub(px, mul(affected, cx, B), B), 2, B
    ), B)
    y180 = add(cy, scale(
        sub(py, mul(affected, cy, B), B), 2, B
    ), B)

    def blend(a, b, flag):
        return add(a, mul(sub(b, a, B), flag, B), B)

    nx = blend(resize_x, blend(x90, x180, double_turn), rotate)
    ny = blend(resize_y, blend(y90, y180, double_turn), rotate)
    nw = blend(resize_w, blend(w90, widths, double_turn), rotate)
    nh = blend(resize_h, blend(h90, heights, double_turn), rotate)

    gx = sub(
        scale(add(row(nw, B), col(nw, B), B), 0.5, B),
        ab(sub(row(nx, B), col(nx, B), B), B), B
    )
    gy = sub(
        scale(add(row(nh, B), col(nh, B), B), 0.5, B),
        ab(sub(row(ny, B), col(ny, B), B), B), B
    )
    object_hits = total(
        mul(inside(gx, gy, B), pair_mask, B), N * N, B
    )

    sx = node("abs", [node("sub", [x, nx], B)], B)
    sy = node("abs", [node("sub", [y, ny], B)], B)
    sx = node("sub", [scale(nw, 0.5, B), sx], B)
    sy = node("sub", [scale(nh, 0.5, B), sy], B)
    rectangle = node("min2", [sx, sy], B)
    rectangle = node("relu", [rectangle], B)
    rectangle = node("mul", [rectangle, K(2)], B)
    rectangle = node("min2", [rectangle, one], B)
    rectangle = node("mul", [rectangle, collidable], B)
    wall_overlap = node("mul", [rectangle, wall], B)
    wall_means = node("gmean", [wall_overlap], B)
    wall_hits = scale(total(wall_means, N, B), 4096, B)

    xlow = node("vrelu", [
        add(sub(scale(nw, 0.5, B), nx, B), K(-0.5), B)
    ], B)
    ylow = node("vrelu", [
        add(sub(scale(nh, 0.5, B), ny, B), K(-0.5), B)
    ], B)
    xhigh = node("vrelu", [
        add(add(nx, scale(nw, 0.5, B), B), K(-63.5), B)
    ], B)
    yhigh = node("vrelu", [
        add(add(ny, scale(nh, 0.5, B), B), K(-63.5), B)
    ], B)
    boundary = total(mul(
        add(add(xlow, ylow, B), add(xhigh, yhigh, B), B),
        collidable, B
    ), N, B)

    marker_values = []
    for index in (12, 13):
        qx = sv(nx, index, index + 1, B)
        qy = sv(ny, index, index + 1, B)
        ax = node("abs", [node("sub", [x, qx], B)], B)
        ay = node("abs", [node("sub", [y, qy], B)], B)
        distance = node("add", [ax, ay], B)
        penalty = node("mul", [distance, K(-1024)], B)
        samples = node("add", [distance_fields, penalty], B)
        marker_values.append(node("gmax", [samples], B))

    a00 = sv(marker_values[0], 0, 1, B)
    a01 = sv(marker_values[0], 1, 2, B)
    a10 = sv(marker_values[1], 0, 1, B)
    a11 = sv(marker_values[1], 1, 2, B)
    navigation = vmax(add(a00, a11, B), add(a01, a10, B), B)
    compactness = total(
        mul(add(nw, nh, B), actuated, B), N, B
    )

    predicted_hash = fingerprint(nx, ny, nw, nh, B)
    unchanged_distance = total(
        ab(sub(predicted_hash, current_hash, B), B), 4, B
    )
    unchanged = boolean(
        add(K(0.001), neg(unchanged_distance, B), B), B
    )

    exists = total(mul(sq, control_present, B), Q, B)
    missing = sub(K(1), exists, B)
    known_failure = total(mul(sq, bad_controls, B), Q, B)

    candidate_action_id = linear(sq, Waction_number, B)
    action_difference = ab(
        sub(past_action_ids, candidate_action_id, B), B
    )
    same_action = boolean(
        add(K(0.25), neg(action_difference, B), B), B
    )
    visits = total(
        mul(same_action, matching_context, B), H, B
    )

    score = sub(
        scale(navigation, 5, B), scale(compactness, 0.08, B), B
    )
    score = sub(score, scale(visits, 160, B), B)
    invalidity = add(
        add(object_hits, wall_hits, B),
        add(boundary, add(unchanged, missing, B), B), B
    )
    score = sub(score, scale(invalidity, 10000, B), B)
    score = sub(score, scale(known_failure, 40000, B), B)

    chosen_control = node("mul", [controls, sq], B)
    chosen_control = node(
        "conv", [chosen_control], B, w=Wcontrol_sum
    )
    click_penalty = node("sub", [chosen_control, one], B)
    click_penalty = node("mul", [click_penalty, K(100000)], B)
    candidate_click = node("add", [click_penalty, score], B)
    best_next = node("max2", [sbest, candidate_click], B)
    qnext = linear(sq, Wqshift, B)

    candidate_id = fresh("loop")
    G.append({
        "id": candidate_id, "op": "loop", "T": Q,
        "state": [
            {"id": sq, "init": qinitial},
            {"id": sbest, "init": initial_click}
        ],
        "body": B, "next": {sq: qnext, sbest: best_next}
    })
    G.append({
        "id": "click", "op": "add",
        "in": [candidate_id + "." + sbest, K(0)]
    })

    # Identify the first maximum cell, matching row-major decoding.
    best_value = node("gmax", ["click"])
    best_gap = node("abs", [node("sub", ["click", best_value])])
    maximum_cells = node("relu", [affine_s(best_gap, -100, 1)])
    reverse_position = affine_s(position, -1, 4097)
    first_key = node("mul", [maximum_cells, reverse_position])
    first_peak = node("gmax", [first_key])
    first_gap = node(
        "abs", [node("sub", [reverse_position, first_peak])]
    )
    first_cell = node("relu", [affine_s(first_gap, -2, 1)])
    first_cell = node("mul", [first_cell, maximum_cells])
    selected_control = node("gmax", [
        node("mul", [controls, first_cell])
    ])
    selected_action_id = linear(selected_control, Waction_number)

    G.append({
        "id": "mem_next", "op": "vconcat",
        "in": [
            current_hash, selected_action_id,
            sv("mem", 0, 195), bad_controls, selected_control
        ]
    })

    logits_bias = np.zeros(8, np.float32)
    logits_bias[6] = 1
    G.append({
        "id": "logits", "op": "dense", "in": ["aprev"],
        "w": parameter(np.zeros((8, 8), np.float32)),
        "b": parameter(logits_bias)
    })

    return {"graph": G, "memory_size": 254, "params": P}
