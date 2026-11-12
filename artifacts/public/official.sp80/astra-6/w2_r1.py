import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    movable = (8, 9, 15)
    receivers = (11, 13)

    def c1(o, colors, value=1.0, dy=0, dx=0):
        if isinstance(colors, int):
            colors = (colors,)
        for color in colors:
            p["W1"][o, color, dy + 2, dx + 2] += value

    def c2(o, channel, value=1.0, dy=0, dx=0):
        p["W2"][o, channel, dy + 1, dx + 1] += value

    def local(o, channel, value=1.0):
        p["W3"][o, channel] += value

    def axial(o, channel, axis, value=1.0):
        p["W3"][o, (32 if axis == "row" else 64) + channel] += value

    # Materials and display orientation.
    c1(0, movable)
    c1(1, 9)
    c1(2, receivers)
    c1(3, 12)
    c1(4, 6)

    for o, coordinate, sign, tangent in [
        (5, 16, -1, (0, 1)),
        (6, 16, 1, (0, 1)),
        (7, 17, -1, (1, 0)),
        (8, 17, 1, (1, 0)),
    ]:
        dy, dx = tangent
        c1(o, (0, 14), 1, dy, dx)
        c1(o, (0, 14), 1, -dy, -dx)
        c1(o, coordinate, 64 * sign)
        p["b1"][o] = -1 if sign < 0 else -65

    # Concave selected corners distinguish turners from straight bars.
    notch_neighbors = [
        ((0, -1), (1, 0)),
        ((1, 0), (0, 1)),
        ((0, 1), (-1, 0)),
        ((-1, 0), (0, -1)),
    ]
    for o, neighbors in zip(range(9, 13), notch_neighbors):
        for dy, dx in neighbors:
            c1(o, 9, 1, dy, dx)
        c1(o, 9, -2)
        p["b1"][o] = -1

    for o, (dy, dx) in zip(
        range(13, 17), [(0, -2), (0, 2), (-2, 0), (2, 0)]
    ):
        c1(o, receivers, 1, dy, dx)

    c1(17, 16)
    c1(18, 17)

    # Cell-size code, measured from a source square.
    c1(19, 6)
    c1(19, 6, -1, -1, 0)
    c1(19, 6, -1, 0, -1)
    c1(20, 6, 1, 0, 2)

    # Euler corner features count visible movable components.
    c1(21, movable)
    c1(21, movable, -1, -1, 0)
    c1(21, movable, -1, 0, -1)
    c1(22, movable)
    c1(22, movable, 1, -1, 0)
    c1(22, movable, 1, 0, -1)
    c1(22, movable, -3, -1, -1)
    p["b1"][22] = -2

    for d, (dy, dx) in enumerate(directions):
        ty, tx = -dx, dy
        c1(23 + d, receivers, 1, dy, dx)
        c1(23 + d, receivers, 1, 2 * ty, 2 * tx)
        c1(23 + d, receivers, 1, -2 * ty, -2 * tx)
        c1(23 + d, receivers, -3)
        p["b1"][23 + d] = -2

    c1(31, 15)

    for o, source in [(0, 0), (1, 1), (2, 3), (3, 4)]:
        c2(o, source)
    for d in range(4):
        c2(4 + d, 5 + d)
        c2(8 + d, 9 + d)

    # Receiver mouth markers for cell sizes two, three and four.
    for d, (dy, dx) in enumerate(directions):
        c2(12 + d, 2, 1, dy, dx)
        side_terms = (
            [(13, 0, -1), (14, 0, 1)]
            if dx == 0 else [(15, -1, 0), (16, 1, 0)]
        )
        for source, sy, sx in side_terms:
            c2(12 + d, source, 1, sy, sx)
            c2(21 + d, source, -1, sy, sx)
        c2(12 + d, 2, -3)
        p["b2"][12 + d] = -2
        c2(21 + d, 23 + d, 3)
        p["b2"][21 + d] = -1

    c2(16, 17)
    c2(17, 18)
    c2(18, 19, 2)
    c2(18, 20)
    c2(18, 20, 1, 0, 1)
    p["b2"][18] = -2
    c2(19, 21)
    c2(20, 22)
    c2(25, 31)

    gauge = [(4, "col"), (5, "col"), (6, "row"), (7, "row")]

    def gravity(o, d, value=1.0):
        channel, axis = gauge[d]
        axial(o, channel, axis, value)

    for d in range(4):
        gravity(d, d)

    # Selected bounding box: x max, complemented x min, y equivalents.
    for o, coordinate, sign in [
        (4, 17, 1), (5, 17, -1),
        (6, 16, 1), (7, 16, -1),
    ]:
        local(o, 1)
        local(o, coordinate, sign)
        p["b3"][o] = -1 if sign > 0 else 0

    # Main receiver moments, grouped by the gravity axis.
    for axis in range(2):
        ds = (0, 1) if axis == 0 else (2, 3)
        coordinate = 17 if axis == 0 else 16
        for sign, o in [(1, 8 + 2 * axis), (-1, 9 + 2 * axis)]:
            for d in ds:
                local(o, 12 + d)
                local(o, 21 + d)
                gravity(o, d)
            local(o, coordinate, sign)
            p["b3"][o] = -2 if sign > 0 else -1

    for o, coordinate, sign in [
        (12, 17, 1), (13, 17, -1),
        (14, 16, 1), (15, 16, -1),
    ]:
        local(o, 2)
        local(o, coordinate, sign)
        p["b3"][o] = -1 if sign > 0 else 0

    axial(16, 1, "row")
    axial(17, 1, "col")
    local(18, 0)
    local(18, 1, -1)

    for o, coordinate in [(19, 16), (20, 17)]:
        local(o, 0)
        local(o, 1, -1)
        local(o, coordinate)
        p["b3"][o] = -1

    local(21, 19)
    local(22, 20)
    local(23, 18)
    for source in range(8, 12):
        local(24, source)
    local(25, 25)

    for o, coordinate, sign in [
        (26, 17, 1), (27, 17, -1),
        (28, 16, 1), (29, 16, -1),
    ]:
        local(o, 3)
        local(o, coordinate, sign)
        p["b3"][o] = -1 if sign > 0 else 0

    # Perpendicular receiver coordinate for turners.
    for o, mouth_dirs, gravity_dirs, coordinate in [
        (30, (2, 3), (0, 1), 16),
        (31, (0, 1), (2, 3), 17),
    ]:
        for d in mouth_dirs:
            local(o, 12 + d)
            local(o, 21 + d)
        for d in gravity_dirs:
            gravity(o, d)
        local(o, coordinate)
        p["b3"][o] = -2

    # Recurrent directional prefix sums create downstream half-planes.
    for d, (dy, dx) in enumerate(directions):
        stripe = 16 if d < 2 else 17
        coordinate = 19 if d < 2 else 20
        sign = -1 if d in (0, 2) else 1
        offset = 1 if sign < 0 else 0

        p["Wh"][d, d, 1 - dy, 1 - dx] = 1
        p["Uh"][d, stripe] = 1
        p["Uh"][d, d] = 128
        p["bh"][d] = -128

        p["Wh"][4 + d, d, 1, 1] = 1
        p["Uh"][4 + d, stripe] = -128
        p["Uh"][4 + d, 18] = 128
        p["Uh"][4 + d, d] = 128
        p["Uh"][4 + d, coordinate] = 0.01 * sign
        p["bh"][4 + d] = -256 + 0.01 * offset

        p["Uh"][8 + d, 18] = 2
        p["Uh"][8 + d, d] = 2
        p["Uh"][8 + d, coordinate] = sign
        p["bh"][8 + d] = -4 + offset

        p["Wc"][0, 32 + 4 + d] = 100
        p["Wc"][0, 32 + 8 + d] = 1
    p["Wc"][0, 18] = 0.001

    # Affine expressions over pooled features.
    def const(value):
        return np.zeros(96, np.float64), float(value)

    def feat(channel, mean=False):
        v = np.zeros(96, np.float64)
        v[(64 if mean else 32) + channel] = 1
        return v, 0.0

    def add(*expressions):
        v = np.zeros(96, np.float64)
        b = 0.0
        for ev, eb in expressions:
            v += ev
            b += eb
        return v, b

    def mul(expression, coefficient):
        v, b = expression
        return v * coefficient, b * coefficient

    def sub(a, b):
        return add(a, mul(b, -1))

    next_z = 0

    def zunit(expression):
        nonlocal next_z
        i = next_z
        next_z += 1
        p["Wz"][i] = expression[0]
        p["bz"][i] = expression[1]
        return i

    def bipolar_action(memory_unit, action_id, gain):
        p["Wo"][action_id, 128 + memory_unit] += gain
        p["bo"][action_id] += gain

    p["bo"][:] = -1000
    p["bo"][1:5] = 0
    p["bo"][5:7] = 5

    cell = add(const(2), feat(23))
    gs = [zunit(feat(d)) for d in range(4)]
    selected_elbow = zunit(feat(24))
    other_elbow = zunit(feat(25))
    scale_pos = zunit(add(feat(23), const(-1)))
    scale_neg = zunit(sub(const(1), feat(23)))

    selected_max = [
        mul(feat(4), 63),
        mul(feat(6), 63),
    ]
    selected_min = [
        add(const(63), mul(feat(5), -63)),
        add(const(63), mul(feat(7), -63)),
    ]
    source_max = [
        mul(feat(26), 63),
        mul(feat(28), 63),
    ]
    source_min = [
        add(const(63), mul(feat(27), -63)),
        add(const(63), mul(feat(29), -63)),
    ]

    # Exact splitter-width / receiver-gap matches.
    # Gates are applied separately, avoiding large-number cancellation.
    next_mem = 0
    for axis in range(2):
        ds = (0, 1) if axis == 0 else (2, 3)
        width = add(sub(selected_max[axis], selected_min[axis]), const(1))
        cp = feat(8 + 2 * axis)
        cm = feat(9 + 2 * axis)
        cpm = feat(8 + 2 * axis, True)
        cmm = feat(9 + 2 * axis, True)

        for marker_width in (2, 3):
            low = add(
                const(63 + (marker_width - 1) / 2),
                mul(cm, -63),
            )
            high = add(
                mul(cp, 63), const(-(marker_width - 1) / 2)
            )
            total = mul(cpm, 4096 * 63 / marker_width)
            middle = sub(sub(total, low), high)
            count_error = add(
                mul(add(cpm, cmm), 4096 / marker_width), const(-3)
            )
            count_pos = zunit(count_error)
            count_neg = zunit(mul(count_error, -1))

            for left, right, needs_three in [
                (low, high, False),
                (low, middle, True),
                (middle, high, True),
            ]:
                error = sub(add(width, cell), sub(right, left))
                residual = sub(
                    add(left, mul(add(cell, const(1)), 0.5)),
                    selected_min[axis],
                )
                ep = zunit(error)
                en = zunit(mul(error, -1))
                rp = zunit(residual)
                rn = zunit(mul(residual, -1))

                def match_gates(m):
                    p["Wmz"][m, ep] -= 1000
                    p["Wmz"][m, en] -= 1000
                    p["Wmz"][m, selected_elbow] -= 1000
                    p["bm"][m] -= 1000
                    for gd in ds:
                        p["Wmz"][m, gs[gd]] += 1000
                    if marker_width == 3:
                        p["Wmz"][m, scale_pos] -= 1000
                        p["Wmz"][m, scale_neg] -= 1000
                    else:
                        p["bm"][m] -= 1000
                        p["Wmz"][m, scale_pos] += 1000
                        p["Wmz"][m, scale_neg] += 1000
                    if needs_three:
                        p["Wmz"][m, count_pos] -= 1000
                        p["Wmz"][m, count_neg] -= 1000

                acts = (3, 4) if axis == 0 else (1, 2)
                for sign, act in [(-1, acts[0]), (1, acts[1])]:
                    m = next_mem
                    next_mem += 1
                    p["Wmz"][m, rp] = 4 * sign
                    p["Wmz"][m, rn] = -4 * sign
                    p["bm"][m] = -3.6
                    # 0.45 cell dead zone: 1.8*(2 + scale_code).
                    p["Wmz"][m, scale_pos] -= 1.8
                    p["Wmz"][m, scale_neg] += 1.8
                    p["bm"][m] -= 1.8
                    match_gates(m)
                    bipolar_action(m, act, 20)

                m = next_mem
                next_mem += 1
                p["Wmz"][m, rp] = -4
                p["Wmz"][m, rn] = -4
                p["bm"][m] = 4.8
                p["Wmz"][m, scale_pos] += 1.6
                p["Wmz"][m, scale_neg] -= 1.6
                match_gates(m)
                for act in range(1, 5):
                    bipolar_action(m, act, -100)

    assert next_mem == 36

    # Without a turner, unmatched bars can terminate one branch at an edge.
    for axis in range(2):
        ds = (0, 1) if axis == 0 else (2, 3)
        bound_max = mul(feat(12 + 2 * axis), 63)
        bound_min = add(
            const(63), mul(feat(13 + 2 * axis), -63)
        )
        dl = sub(selected_min[axis], bound_min)
        dh = sub(bound_max, selected_max[axis])
        zl = zunit(dl)
        zh = zunit(dh)
        nl = zunit(sub(dl, dh))
        nh = zunit(sub(dh, dl))

        for distance, nearer, act in [
            (zl, nl, 3 if axis == 0 else 1),
            (zh, nh, 4 if axis == 0 else 2),
        ]:
            m = next_mem
            next_mem += 1
            p["Wmz"][m, distance] = 4
            p["Wmz"][m, nearer] = -20
            p["Wmz"][m, selected_elbow] = -1000
            p["Wmz"][m, other_elbow] = -1000
            p["bm"][m] = -1005.6
            for gd in ds:
                p["Wmz"][m, gs[gd]] += 1000
            bipolar_action(m, act, 10)

    assert next_mem == 40

    # With a turner, unmatched straight bars become upstream distributors.
    # The remaining source is opposite the source used by the turner.
    branch_mem = 50
    for gd in range(4):
        axis = 0 if gd < 2 else 1
        other_axis = 1 - axis
        remaining_is_min = gd in (0, 3)

        lateral = sub(
            source_min[axis] if remaining_is_min else source_max[axis],
            selected_min[axis] if remaining_is_min else selected_max[axis],
        )

        if gd in (0, 2):
            upstream = sub(
                selected_min[other_axis],
                add(source_max[other_axis], cell, const(1)),
            )
            upstream_action = 1 if gd == 0 else 3
        else:
            upstream = sub(
                add(source_min[other_axis], mul(cell, -1), const(-1)),
                selected_max[other_axis],
            )
            upstream_action = 2 if gd == 1 else 4

        acts = (3, 4) if axis == 0 else (1, 2)
        decisions = [
            (mul(lateral, -1), acts[0]),
            (lateral, acts[1]),
            (upstream, upstream_action),
        ]

        for residual, act in decisions:
            drive = add(
                residual,
                mul(cell, -0.45),
                mul(sub(const(1), feat(gd)), -1000),
                mul(sub(const(1), feat(25)), -1000),
                mul(feat(24), -1000),
            )
            z = zunit(drive)
            m = branch_mem
            branch_mem += 1
            p["Wmz"][m, z] = 8
            p["bm"][m] = -4
            bipolar_action(m, act, 10)

    assert branch_mem == 62

    # Turner inlet and outlet alignment.
    for gd in range(4):
        axis = 0 if gd < 2 else 1
        other_axis = 1 - axis

        if gd in (0, 3):
            lateral_target = add(
                source_max[axis], mul(cell, -2), const(1)
            )
        else:
            lateral_target = source_min[axis]

        side_coordinate = mul(feat(30 if gd < 2 else 31), 63)
        outlet_target = add(
            side_coordinate,
            mul(cell, -1 if gd in (0, 2) else -2),
            const(1),
        )

        for control_axis, target in [
            (axis, lateral_target),
            (other_axis, outlet_target),
        ]:
            residual = sub(target, selected_min[control_axis])
            acts = (3, 4) if control_axis == 0 else (1, 2)
            for sign, act in [(-1, acts[0]), (1, acts[1])]:
                z = zunit(add(
                    mul(residual, sign),
                    mul(cell, -0.49),
                    mul(sub(const(1), feat(gd)), -1000),
                    mul(sub(const(1), feat(24)), -1000),
                ))
                p["Wo"][act, z] += 10

    # Retain the largest observed component count through temporary overlaps.
    count = zunit(add(
        mul(feat(21, True), 4096),
        mul(feat(22, True), -4096),
    ))
    for k in range(1, 9):
        m = 39 + k
        p["Wmz"][m, count] = 8
        p["bm"][m] = -8 * (k - 0.5)
        p["Wm"][m, m] = 32
        p["Wm"][m, 62] = 32

    # One selection pass, then test the routing.
    p["Wm"][48, 48] = 0.999
    p["Wma"][48, 6] = 0.01
    p["Wm"][49, 48] = 300
    p["Wma"][49, 6] = 3
    p["Wm"][49, 40:48] = -1.5
    p["bm"][49] = -7.8
    p["Wo"][5, 128 + 49] = 4

    # A single initial reset; subsequent decisions remain observation-based.
    p["bm"][62] = 10
    p["Wm"][63, 62] = 10
    p["bo"][0] = 1000
    p["Wo"][0, 128 + 63] = -2000

    assert next_z <= 128
    return p
