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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    def c1(o, i, y=0, x=0, v=1.0):
        p["W1"][o, i, y + 2, x + 2] += v

    def c2(o, i, y=0, x=0, v=1.0):
        p["W2"][o, i, y + 1, x + 1] += v

    def ch(o, i, y=0, x=0, v=1.0):
        p["Wh"][o, i, y + 1, x + 1] += v

    def f3(o, i, v=1.0, axial=None):
        off = 0 if axial is None else (32 if axial == "row" else 64)
        p["W3"][o, off + i] += v

    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    forward = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    corner_arms = [
        ((-1, 0), (0, -1)),
        ((-1, 0), (0, 1)),
        ((1, 0), (0, -1)),
        ((1, 0), (0, 1)),
    ]

    # Basic colors and coordinates.
    for o, i in enumerate((9, 8, 11, 12, 6, 16, 17, 18)):
        c1(o, i)
    c1(1, 15)

    # Gravity, inferred from the one-pixel timer strip.
    ui_specs = [
        ((0, 1), (1, 0), 16, -64.0, 0.0),
        ((0, 1), (-1, 0), 16, 64.0, -64.0),
        ((1, 0), (0, 1), 17, -64.0, 0.0),
        ((1, 0), (0, -1), 17, 64.0, -64.0),
    ]
    for d, (along, inward, coordinate, gain, bias) in enumerate(ui_specs):
        for color in (0, 14):
            c1(8 + d, color)
            c1(8 + d, color, *along)
            c1(8 + d, color, *inward, v=-1)
        c1(8 + d, 18, *along, v=-1)
        c1(8 + d, coordinate, v=gain)
        p["b1"][8 + d] = bias

    # Broad receiver-mouth masks.
    for d, (dy, dx) in enumerate(forward):
        py, px = -dx, dy
        c1(12 + d, 12, v=3)
        c1(12 + d, 11, 2 * dy, 2 * dx, 2)
        c1(12 + d, 11, 2 * py, 2 * px)
        c1(12 + d, 11, -2 * py, -2 * px)
        p["b1"][12 + d] = -5

    # Convex corners of the visible union of ordinary bars.
    for j, arms in enumerate(corner_arms):
        for color in (8, 9):
            c1(16 + j, color)
        for dy, dx in arms:
            for color in (8, 9):
                c1(16 + j, color, dy, dx, -1)
            c1(16 + j, 18, dy, dx)
        p["b1"][16 + j] = -2

    # Pixels immediately outside the selected piece.
    for j, (dy, dx) in enumerate(cardinal):
        c1(20 + j, 9, -dy, -dx)
        c1(20 + j, 9, v=-1)

    # Shifted receiver pixels for strict, binary mouth detection.
    for o, (dy, dx) in zip((24, 25, 30, 31), cardinal):
        c1(o, 11, 2 * dy, 2 * dx)

    # Concave corners of turning pieces.
    for j, arms in enumerate(corner_arms):
        c1(26 + j, 12)
        for dy, dx in arms:
            c1(26 + j, 9, dy, dx, 1.125)
            c1(26 + j, 15, dy, dx)
        p["b1"][26 + j] = -2.25

    # Second layer.
    for d in range(4):
        c2(d, 8 + d)
        c2(4 + d, 12 + d, v=-1)
        p["b2"][4 + d] = 1
        c2(8 + d, 16 + d)
        c2(12 + d, 20 + d)

    for o, i in ((16, 0), (17, 1), (18, 4), (19, 5), (20, 6)):
        c2(o, i)

    # Contact between selected and unselected pieces.
    for i in range(20, 24):
        c2(22, i)
    c2(22, 1)
    p["b2"][22] = -1

    # Selected and general turning-piece markers.
    for j, arms in enumerate(corner_arms):
        sy = arms[0][0] + arms[1][0]
        sx = arms[0][1] + arms[1][1]
        c2(23, 26 + j, sy, sx, 4)
        c2(27, 26 + j, sy, sx)
    p["b2"][23] = -3

    strict_channels = (21, 24, 25, 26)
    strict_samples = [
        [(25, 1, 0), (30, 0, -1), (31, 0, 1)],
        [(24, -1, 0), (30, 0, -1), (31, 0, 1)],
        [(31, 0, 1), (24, -1, 0), (25, 1, 0)],
        [(30, 0, -1), (24, -1, 0), (25, 1, 0)],
    ]
    for o, samples in zip(strict_channels, strict_samples):
        c2(o, 3)
        for i, dy, dx in samples:
            c2(o, i, dy, dx)
        p["b2"][o] = -3

    # Source coordinate moments.
    c2(28, 5)
    c2(28, 4)
    p["b2"][28] = -1
    c2(29, 4)
    c2(29, 5, v=-1)
    c2(30, 6)
    c2(30, 4)
    p["b2"][30] = -1
    c2(31, 4)
    c2(31, 6, v=-1)

    def ui(o, d, value=1.0):
        f3(o, d, value, "col" if d < 2 else "row")

    # Axial representation.
    for d in range(4):
        f3(d, 17)
        f3(d, 4 + d, -1)
        ui(d, d, 4)
        p["b3"][d] = -3

        for i in ((14, 15) if d < 2 else (12, 13)):
            f3(4 + d, i)
        ui(4 + d, d)
        p["b3"][4 + d] = -1
        ui(12 + d, d)

    corner_gravities = [(1, 3), (1, 2), (0, 3), (0, 2)]
    for j, directions in enumerate(corner_gravities):
        f3(8 + j, 8 + j)
        for d in directions:
            ui(8 + j, d)
        p["b3"][8 + j] = -1

    # Uncovered sources, plus a separately decodable contact marker.
    f3(16, 18)
    f3(16, 16, -1, "row")
    f3(16, 16, -1, "col")
    f3(16, 22, 2)
    f3(17, 16)

    # Unaligned turning pieces receive click priority.
    f3(18, 17, 0.01)
    f3(18, 27, 4.0 / 3.0)
    f3(18, 23, -4.0 / 3.0)
    for i in strict_channels[2:]:
        f3(18, i, -1, "row")
    for i in strict_channels[:2]:
        f3(18, i, -1, "col")

    # Selected bounding-box extrema.
    f3(19, 19)
    f3(19, 16)
    p["b3"][19] = -1
    f3(20, 16)
    f3(20, 19, -1)
    f3(21, 20)
    f3(21, 16)
    p["b3"][21] = -1
    f3(22, 16)
    f3(22, 20, -1)

    # Receiver-mouth coordinate extrema.
    f3(23, 19)
    f3(24, 19, -1)
    for i in strict_channels[2:]:
        f3(23, i)
        f3(24, i)
    p["b3"][23] = -1

    f3(25, 20)
    f3(26, 20, -1)
    for i in strict_channels[:2]:
        f3(25, i)
        f3(26, i)
    p["b3"][25] = -1

    for j in range(4):
        f3(27 + j, 28 + j)
    f3(31, 23)

    # Binary downstream rays: q[d] = h[d] - h[4+d].
    for d, (dy, dx) in enumerate(forward):
        for o in (d, 4 + d):
            ch(o, d, dy, dx)
            ch(o, 4 + d, dy, dx, -1)
            p["Uh"][o, d] = 1
        p["bh"][4 + d] = -1

    # Smooth guidance field, kept separate from binary validity.
    for dy, dx in cardinal:
        ch(8, 8, dy, dx, 0.2474)
    for d in range(4):
        ch(8, d, v=0.0104)
        ch(8, 4 + d, v=-0.0104)

    # Source deficits have value one; piece contacts have value three.
    p["Uh"][9, 16] = 4
    p["Uh"][9, 27] = -3
    p["Uh"][9, 28] = -3
    for d in range(4):
        ch(9, d, v=-1)
        ch(9, 4 + d)

    # Signed guidance derivatives, encoded around an outlet baseline.
    for o, positive, negative in (
        (10, (0, 1), (0, -1)),
        (11, (1, 0), (-1, 0)),
    ):
        ch(o, 8, *positive, v=0.5)
        ch(o, 8, *negative, v=-0.5)
        p["Uh"][o, 4:8] = 2
        p["bh"][o] = -1

    # Four independent OUTWARD corner tests.
    # An unresolved selected corner has value two, another corner one.
    outward = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for j, (dy, dx) in enumerate(outward):
        o = 12 + j
        p["Uh"][o, 8 + j] = 2
        p["Uh"][o, 17] = 1
        p["bh"][o] = -1
        for d in range(4):
            ch(o, d, dy, dx, -3)
            ch(o, 4 + d, dy, dx, 3)

    p["Wc"][0, 18] = 200
    p["Wc"][0, 17] = -1000
    for j in range(4):
        p["Wc"][0, 44 + j] = 100

    # Affine forms in the pooled network representation.
    def const(value):
        a = np.zeros(97, np.float64)
        a[-1] = value
        return a

    def feature(index):
        a = const(0)
        a[index] = 1
        return a

    def hm(i):
        return feature(i)

    def ha(i):
        return feature(16 + i)

    def fm(i):
        return feature(32 + i)

    def fa(i):
        return feature(64 + i)

    used = 0

    def unit(form, outputs=None):
        nonlocal used
        if used >= 128:
            raise RuntimeError("Controller exceeds its fixed width")
        j = used
        used += 1
        p["Wz"][j] = form[:96]
        p["bz"][j] = form[96]
        if outputs:
            for action, weight in outputs.items():
                p["Wo"][action, j] += weight
        return j

    def clipped(form):
        return unit(form), unit(form - const(1))

    def memory_feature(m, pair, weight):
        p["Wmz"][m, pair[0]] += weight
        p["Wmz"][m, pair[1]] -= weight

    p["bo"][:] = -10000
    p["bo"][1:5] = 0
    p["bo"][5] = 10
    p["bo"][6] = 5

    big = 100000.0
    hook = fm(31)
    zhook = unit(hook, {6: 1})
    zui = [unit(fm(12 + d)) for d in range(4)]

    zrisk = []
    for j in range(4):
        zrisk.append(unit(hm(12 + j) - const(1) - 2 * hook,
                          {6: -100}))
        unit(hm(12 + j), {5: -1000})
    unit(hm(9), {5: -1000})
    unit(fm(18) - const(0.1), {6: 500, 5: -1000})

    sy_max = fm(19)
    sy_min = const(1) - fm(20)
    sx_max = fm(21)
    sx_min = const(1) - fm(22)

    src_y_max = fm(27)
    src_y_min = const(1) - fm(28)
    src_x_max = fm(29)
    src_x_min = const(1) - fm(30)

    cup_y = 0.5 * (fm(23) + const(1) - fm(24))
    cup_x = 0.5 * (fm(25) + const(1) - fm(26))

    outlet_count = sum((fa(i) for i in range(4, 8)), const(0))
    grad_x = 8192 * (ha(10) - outlet_count)
    grad_y = 8192 * (ha(11) - outlet_count)

    perpendicular = [(3, 4), (3, 4), (1, 2), (1, 2)]
    downstream_corners = [(2, 3), (0, 1), (1, 3), (0, 2)]
    upstream_actions = [1, 2, 3, 4]

    # Movement is enabled by a selected-corner deficit.
    # Guidance is bounded, so a settled piece cannot drift.
    for d in range(4):
        orientation = fm(12 + d)
        gate = big * (orientation - const(1) - hook)
        neg, pos = perpendicular[d]

        for j in downstream_corners[d]:
            unit(hm(12 + j) - const(1) + gate, {neg: 20, pos: 20})

        gradient = 20 * (grad_x if d < 2 else grad_y)
        unit(gradient + const(1) + gate, {pos: 3, neg: -3})
        unit(gradient - const(1) + gate, {pos: -3, neg: 3})
        unit(orientation - hook, {pos: -3, neg: 3})

    # Preserve nearby coverage when there is a single source.
    # Source spread perpendicular to gravity disables this constraint
    # when upstream pieces can redistribute several sources.
    half_cell = const(1.5 / 63.0)
    source_axes = [
        (
            fm(12) + fm(13),
            sx_min, sx_max,
            src_x_min + half_cell, src_x_max - half_cell,
            (src_x_max - src_x_min) - (src_y_max - src_y_min),
            3, 4,
        ),
        (
            fm(14) + fm(15),
            sy_min, sy_max,
            src_y_min + half_cell, src_y_max - half_cell,
            (src_y_max - src_y_min) - (src_x_max - src_x_min),
            1, 2,
        ),
    ]
    for orientation, low, high, source_low, source_high, spread, neg, pos in source_axes:
        gate = big * (orientation - const(1) - hook - spread)
        for t, toward, away in (
            (63 * (low - source_low), neg, pos),
            (63 * (source_high - high), pos, neg),
        ):
            for threshold, coefficient in (
                (-3, 1), (-2, -1), (6, -1), (7, 1)
            ):
                unit(t - const(threshold) + gate,
                     {away: -100 * coefficient})
            for threshold, coefficient in (
                (0, 1), (1, -1), (6, -1), (7, 1)
            ):
                unit(t - const(threshold) + gate,
                     {toward: 30 * coefficient})

    # Turning-piece placement.
    hook_positions = [
        (0.75 * sy_min + 0.25 * sy_max,
         0.25 * sx_min + 0.75 * sx_max),
        (0.25 * sy_min + 0.75 * sy_max,
         0.75 * sx_min + 0.25 * sx_max),
        (0.75 * sy_min + 0.25 * sy_max,
         0.75 * sx_min + 0.25 * sx_max),
        (0.25 * sy_min + 0.75 * sy_max,
         0.25 * sx_min + 0.75 * sx_max),
    ]
    for d, (hy, hx) in enumerate(hook_positions):
        gate = big * (fm(12 + d) + hook - const(2))
        if d == 0:
            dy = cup_y - hy
            dx = src_x_max - half_cell - hx
        elif d == 1:
            dy = cup_y - hy
            dx = src_x_min + half_cell - hx
        elif d == 2:
            dx = cup_x - hx
            dy = src_y_min + half_cell - hy
        else:
            dx = cup_x - hx
            dy = src_y_max - half_cell - hy
        tolerance = const(0.030)
        unit(1000 * (dy - tolerance) + gate, {2: 1})
        unit(1000 * (-dy - tolerance) + gate, {1: 1})
        unit(1000 * (dx - tolerance) + gate, {4: 1})
        unit(1000 * (-dx - tolerance) + gate, {3: 1})

    # Long splitters are routed toward the nearest non-floor boundary.
    width = sx_max - sx_min
    height = sy_max - sy_min
    long_specs = [
        (fm(12) + fm(13), width, height, sx_min, sx_max, 3, 4),
        (fm(14) + fm(15), height, width, sy_min, sy_max, 1, 2),
    ]
    middle_flags = []
    mslot = 24
    for orientation, length, thickness, low, high, neg, pos in long_specs:
        gate = big * (orientation - const(1) - hook)
        long_flag = clipped(
            100 * (length - 5.5 * thickness - const(4.5 / 63.0)) + gate
        )
        p["Wo"][6, long_flag[0]] += 200
        p["Wo"][6, long_flag[1]] -= 200

        middle_flags.append(clipped(
            100 * (length - 4.5 * thickness - const(3.5 / 63.0)) + gate
        ))

        center = 0.5 * (low + high)
        for side, distance, action in (
            (const(0.5) - center, low, neg),
            (center - const(0.5), const(1) - high, pos),
        ):
            side_flag = clipped(100 * side)
            moving_flag = clipped(100 * (distance - const(0.012)))
            memory_feature(mslot, long_flag, 10)
            memory_feature(mslot, side_flag, 10)
            memory_feature(mslot, moving_flag, 10)
            p["bm"][mslot] = -25
            p["Wo"][action, 128 + mslot] = 200
            p["bo"][action] += 200
            mslot += 1

    # Separate a broad upstream splitter from a contacted lower shelf.
    contact_flag = clipped(hm(9) - const(2))
    for d in range(4):
        m = 28 + d
        memory_feature(m, contact_flag, 10)
        memory_feature(m, middle_flags[0 if d < 2 else 1], 10)
        p["Wmz"][m, zui[d]] = 10
        p["bm"][m] = -25
        action = upstream_actions[d]
        p["Wo"][action, 128 + m] = 250
        p["bo"][action] += 250

    # Avoid repeated commands into an image boundary.
    for distance, action in (
        (sx_min, 3), (const(1) - sx_max, 4),
        (sy_min, 1), (const(1) - sy_max, 2),
    ):
        unit(63 * (const(0.02) - distance), {action: -100})

    # Blocked-motion detection from successive selected-box positions.
    zy = unit(sy_max)
    zx = unit(sx_max)
    p["Wmz"][0, zy] = 0.1
    p["Wmz"][1, zx] = 0.1

    for j, (action, pm, pz, sign) in enumerate((
        (1, 0, zy, 1), (2, 0, zy, -1),
        (3, 1, zx, 1), (4, 1, zx, -1),
    )):
        m = 4 + j
        p["Wm"][m, pm] = -1000 * sign
        p["Wmz"][m, pz] = 100 * sign
        p["Wma"][m, action] = 1002
        p["bm"][m] = -1000
        p["Wo"][6, 128 + m] = 120
        p["bo"][6] += 120

    actions = (1, 2, 3, 4)
    opposite = (1, 0, 3, 2)
    for j, action in enumerate(actions):
        p["Wma"][10 + j, action] = 4

    # A lateral reversal on an unresolved corner triggers one upstream
    # adjustment. Each detector tests one corner, avoiding count errors.
    mslot = 40
    for j, action in enumerate(actions):
        directions = (2, 3) if action in (1, 2) else (0, 1)
        for d in directions:
            for corner in downstream_corners[d]:
                m = mslot
                mslot += 1
                p["Wm"][m, 10 + opposite[j]] = 6
                p["Wma"][m, action] = 6
                p["Wmz"][m, zui[d]] = 6
                p["Wmz"][m, zrisk[corner]] = 6
                p["Wmz"][m, zhook] = -30
                p["bm"][m] = -23
                upstream = upstream_actions[d]
                p["Wo"][upstream, 128 + m] = 100
                p["bo"][upstream] += 100

    # A failed spill must be followed by an actual configuration change.
    p["Wma"][22, 5] = 3
    p["Wo"][5, 128 + 22] = -1000
    for d in range(4):
        m = 56 + d
        p["Wma"][m, 5] = 10
        p["Wmz"][m, zui[d]] = 10
        p["bm"][m] = -15
        action = upstream_actions[d]
        p["Wo"][action, 128 + m] = 100
        p["bo"][action] += 100

    return p
