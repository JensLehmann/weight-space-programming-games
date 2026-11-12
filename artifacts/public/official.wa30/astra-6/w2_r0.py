import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}

    # ------------------------------------------------------------
    # Local sprite and terrain detectors.
    # Ordinary object markers are at sprite top-left + (1, 1).
    # ------------------------------------------------------------

    def match(out, requirements, shift=(0, 0)):
        sy, sx = shift
        for dy, dx, colors in requirements:
            if isinstance(colors, int):
                colors = (colors,)
            for color in colors:
                p["W1"][out, color, 2 + dy + sy, 2 + dx + sx] += 1
        p["b1"][out] = 1 - len(requirements)

    # Player orientations: north, south, west, east.
    for direction in range(4):
        req = []
        for r in range(4):
            for c in range(4):
                black = (
                    (direction == 0 and r == 0)
                    or (direction == 1 and r == 3)
                    or (direction == 2 and c == 0)
                    or (direction == 3 and c == 3)
                )
                color = 0 if black else 14
                colors = (color,)
                if r == 3:
                    colors = tuple(sorted({color, 4, 7}))
                req.append((r - 1, c - 1, colors))
        match(direction, req)

    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    corners = [(-1, -1), (-1, 2), (2, -1), (2, 2)]

    def bordered(out, border, center):
        req = [(y, x, (center,)) for y, x in centers]
        for y, x in corners:
            colors = set(border)
            if y == 2:
                colors.update((4, 7))
            req.append((y, x, tuple(sorted(colors))))
        match(out, req)

    bordered(4, (3, 4), 9)       # Loose or highlighted cargo.
    bordered(6, (3,), 9)         # Cargo directly in front.
    bordered(7, (11,), 15)       # Thief directly in front.
    bordered(8, (11, 15), 15)    # Thief.
    bordered(29, (5,), 9)        # Robot-carried cargo.

    held_req = [(y, x, (9,)) for y, x in centers]
    held_req += [(-1, 0, (0,)), (0, -1, (0,))]
    match(5, held_req)

    # Goal-border pixels: palette 9 adjacent to palette 2.
    for out, bias in [(9, -4), (10, -5)]:
        p["W1"][out, 9, 2, 2] = 4
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            p["W1"][out, 2, 2 + dy, 2 + dx] = 1
        p["b1"][out] = bias

    p["W1"][11, 2, :, :] = 1
    p["W1"][28, 2, 2, 2] = 1

    def obstacle_rectangle(out, ys, xs, colors):
        p["b1"][out] = len(ys) * len(xs)
        for dy in ys:
            for dx in xs:
                p["W1"][out, 18, 2 + dy, 2 + dx] -= 1
                for color in colors:
                    p["W1"][out, color, 2 + dy, 2 + dx] += 1

    obstacle_rectangle(12, range(-1, 3), range(-1, 3), (5, 12))
    obstacle_rectangle(
        13, range(-1, 3), range(-2, 3), (3, 4, 5, 11, 12, 15)
    )
    obstacle_rectangle(
        14, range(-2, 3), range(-1, 3), (3, 4, 5, 11, 12, 15)
    )

    holes = {(0, 1), (1, 0), (2, 3), (3, 2)}
    fence_req = []
    for r in range(4):
        for c in range(4):
            color = 1 if (r, c) in holes else 2
            colors = (color,)
            if r == 3:
                colors = tuple(sorted({color, 4, 7}))
            fence_req.append((r - 1, c - 1, colors))

    for out, shift in [
        (15, (0, 0)), (16, (0, -1)),
        (17, (-1, 0)), (18, (-1, -1)),
    ]:
        match(out, fence_req, shift)

    # Visible fragments immediately beyond an occluding object.
    # Opposite matching fragments identify cargo sitting on a fence.
    last_edge = (2, 2, 1, 2)
    first_edge = (2, 1, 2, 2)
    offsets = (-1, 0, 1, 2)
    match(19, [(-2, x, color)
               for x, color in zip(offsets, last_edge)])
    match(20, [(2, x, color)
               for x, color in zip(offsets, first_edge)])
    match(21, [(y, -2, color)
               for y, color in zip(offsets, last_edge)])
    match(22, [(y, 2, color)
               for y, color in zip(offsets, first_edge)])

    p["W1"][25, 16, 2, 2] = 1
    p["W1"][26, 17, 2, 2] = 1
    p["W1"][31, 5, 2, 2] = 1

    match(27, [
        (0, 0, 9), (0, 1, 9), (1, 0, 9), (1, 1, 2)
    ])

    # ------------------------------------------------------------
    # Morphology and local coordinate readings.
    # ------------------------------------------------------------

    def w2(out, inp, value=1, dy=0, dx=0):
        p["W2"][out, inp, 1 + dy, 1 + dx] += value

    def spread(out, inp, value=1):
        p["W2"][out, inp, :, :] += value

    for i in range(4):
        w2(0, i)
    for i in (2, 3):
        spread(1, i)
    for i in (0, 1):
        spread(2, i)

    w2(3, 27)

    w2(4, 19)
    w2(4, 20, dy=1)
    p["b2"][4] = -1
    w2(5, 21)
    w2(5, 22, dx=1)
    p["b2"][5] = -1

    w2(6, 4)
    spread(7, 4)
    spread(8, 5)
    w2(9, 5)
    w2(10, 6)
    w2(11, 7)
    spread(12, 8)

    w2(13, 9)
    w2(13, 10, -1)

    w2(14, 28)
    w2(14, 9)
    w2(14, 10, -1)

    spread(15, 11)
    spread(30, 11)
    p["b2"][30] = -1

    w2(16, 12)
    for dx in (-1, 0, 1):
        w2(17, 13, dx=dx)
    for dy in (-1, 0, 1):
        w2(18, 14, dy=dy)

    for inp in range(15, 19):
        spread(19, inp)
        spread(20, inp)
    p["b2"][20] = -1

    spread(21, 5)

    w2(23, 4)
    w2(23, 29)

    for out, coordinate, markers in [
        (24, 25, (0, 1, 2, 3)),
        (25, 26, (0, 1, 2, 3)),
        (26, 25, (5, 6)),
        (27, 26, (5, 6)),
    ]:
        w2(out, coordinate)
        for inp in markers:
            w2(out, inp)
        p["b2"][out] = -1

    w2(28, 25)
    w2(29, 26)
    spread(31, 31)
    p["b2"][31] = -8

    # ------------------------------------------------------------
    # Axial context, object sources, and coordinate extrema.
    # ------------------------------------------------------------

    def f3(out, terms, bias=0):
        for inp, value in terms:
            p["W3"][out, inp] += value
        p["b3"][out] = bias

    ROW, COL = 32, 64

    f3(0, [
        (ROW + 13, 1), (COL + 13, 1), (14, 1),
        (19, -1), (20, 1),
    ], -2)
    f3(1, [(16, 1), (19, 1), (20, -1)])
    f3(2, [(17, 1)])
    f3(3, [(18, 1)])

    f3(4, [
        (6, 3), (ROW + 13, -1), (COL + 13, -1),
        (15, -1), (30, 1), (4, -4), (5, -4),
    ])
    f3(5, [(21, 1)])
    f3(6, [(9, 1)])
    f3(7, [(0, 1), (ROW + 9, -1), (COL + 9, -1)])
    f3(8, [(ROW + 1, 1)])
    f3(9, [(COL + 2, 1)])
    f3(10, [(12, 1)])
    f3(11, [(10, 1), (4, -1), (5, -1)])
    f3(12, [(11, 1)])
    f3(17, [(19, 1), (20, -1)])
    f3(18, [(3, 1)])

    for out, inp in [(22, 24), (23, 25), (24, 26), (25, 27)]:
        f3(out, [(inp, 1)])
    f3(31, [(31, 1)])

    # 1-x_min, x_max, 1-y_min, y_max of visible goal borders.
    for out, coord, sign, constant in [
        (13, 29, -1, 1), (14, 29, 1, 0),
        (15, 28, -1, 1), (16, 28, 1, 0),
    ]:
        f3(out, [(13, 4), (coord, sign)], constant - 4)

    K = 128.0
    extrema = [
        (19, 20, 29, 28, -1, 1),
        (21, 26, 29, 28, 1, 0),
        (27, 28, 28, 29, -1, 1),
        (29, 30, 28, 29, 1, 0),
    ]
    for base, paired, major, minor, sign, constant in extrema:
        f3(base, [(23, 4), (major, sign)], constant - 4)
        f3(paired, [
            (23, K + 2), (major, K * sign), (minor, 1)
        ], K * constant - (K + 2))

    # ------------------------------------------------------------
    # Tied spatial fields.
    # ------------------------------------------------------------

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def wh(out, inp, value, dy=0, dx=0):
        p["Wh"][out, inp, 1 + dy, 1 + dx] += value

    def drive(out, terms, bias=0):
        for inp, value in terms:
            p["Uh"][out, inp] += value
        p["bh"][out] = bias

    # Horizontal and vertical carrying-footprint potentials.
    for channel, obstacle in [(0, 2), (1, 3)]:
        for dy, dx in directions:
            wh(channel, channel, 0.2495, dy, dx)
        drive(channel, [(0, 0.02), (17, 0.004), (obstacle, -16)])

    # Cargo attraction, mildly favoring side-on approaches.
    wh(2, 2, 0.4, dx=-1)
    wh(2, 2, 0.4, dx=1)
    drive(2, [(4, 1), (1, -64)])

    wh(3, 0, 1)
    drive(3, [(5, 4), (8, 4)], -8)
    wh(4, 1, 1)
    drive(4, [(5, 4), (9, 4)], -8)

    for dy, dx in directions:
        wh(5, 5, 0.2495, dy, dx)
    wh(5, 2, 0.025, dx=-1)
    wh(5, 2, 0.025, dx=1)
    drive(5, [(10, 0.5), (1, -64)])

    for i, (dy, dx) in enumerate(directions):
        wh(6 + i, 5, 1, dy, dx)
        drive(6 + i, [(7, 64)], -64)
        wh(10 + i, 3, 1, dy, dx)
        wh(10 + i, 4, 1, dy, dx)
        drive(10 + i, [(6, 8)], -8)

    drive(14, [(0, 1), (17, 1)])
    for dy, dx in directions:
        wh(15, 14, 1, dy, dx)
    drive(15, [(5, 16)], -16)

    # ------------------------------------------------------------
    # Pooled feature decoding.
    # ------------------------------------------------------------

    def zf(out, feature, value=1):
        p["Wz"][out, 32 + feature] += value

    for i in range(4):
        p["Wz"][i, 6 + i] = 1
        p["Wz"][i, 10 + i] = 1

    zf(4, 6)
    p["Wz"][5, 15] = 1
    zf(6, 11)
    zf(7, 12)

    # Unusable carrying orientation in a walled, non-fence region.
    zf(8, 6)
    zf(8, 31)
    zf(8, 17, -4)
    p["bz"][8] = -1
    for hchannel in range(10, 14):
        p["Wz"][8, hchannel] = -1e20

    for out, feature in [(9, 22), (10, 23), (11, 24), (12, 25)]:
        zf(out, feature)
    zf(13, 10)
    zf(14, 4)
    zf(27, 31)
    zf(28, 17)

    # Visible goal-corner count.
    p["Wz"][29, 64 + 18] = -4096
    p["bz"][29] = 1.5
    p["Wz"][30, 64 + 18] = 4096
    p["bz"][30] = -1.5

    def expr(terms=(), bias=0):
        a = np.zeros(96, np.float32)
        for feature, value in terms:
            a[32 + feature] += value
        return a, float(bias)

    targets = [
        (expr([(19, -1)], 1), expr([(20, 1), (19, -K)])),
        (expr([(21, 1)]), expr([(26, 1), (21, -K)])),
        (expr([(28, 1), (27, -K)]), expr([(27, -1)], 1)),
        (expr([(30, 1), (29, -K)]), expr([(29, 1)])),
    ]
    for t, (xexpr, yexpr) in enumerate(targets):
        for axis, (a, b) in enumerate((xexpr, yexpr)):
            p["Wz"][15 + 2 * t + axis] = a
            p["bz"][15 + 2 * t + axis] = b

    player_y = expr([(22, 1)])
    player_x = expr([(23, 1)])

    for t, (tx, ty) in enumerate(targets):
        for d in range(4):
            target = ty if d < 2 else tx
            player = player_y if d < 2 else player_x
            sign = -1 if d in (0, 2) else 1
            a = sign * (target[0] - player[0]) * 63
            b = sign * (target[1] - player[1]) * 63
            base = 92 + 8 * t + 2 * d
            p["Wz"][base] = a
            p["bz"][base] = b
            p["Wz"][base + 1] = a
            p["bz"][base + 1] = b - 1

    for d in range(4):
        cargo_feature = 24 if d < 2 else 25
        player_feature = 22 if d < 2 else 23
        sign = -1 if d in (0, 2) else 1

        # Avoid repeatedly walking into an ignored front object.
        zf(23 + d, cargo_feature, 15.75 * sign)
        zf(23 + d, player_feature, -15.75 * sign)
        zf(23 + d, 11, 32)
        p["bz"][23 + d] = -32
        p["Wo"][1 + d, 23 + d] = -8

        # A fence permits cargo, but not the carrier, to pass.
        # Continue forward until the carrier itself is stopped.
        zf(124 + d, cargo_feature, 15.75 * sign)
        zf(124 + d, player_feature, -15.75 * sign)
        zf(124 + d, 6, 32)
        zf(124 + d, 17, 32)
        p["bz"][124 + d] = -64
        p["Wo"][1 + d, 124 + d] = 8

    # ------------------------------------------------------------
    # Bistable memory of initial goal bounds.
    # ------------------------------------------------------------

    goal_tile_exprs = [
        expr([(13, -15.75)], 15.75),
        expr([(14, 15.75)], -0.75),
        expr([(15, -15.75)], 15.75),
        expr([(16, 15.75)], -0.75),
    ]
    for bound, (a, b) in enumerate(goal_tile_exprs):
        base = 32 + 15 * bound
        for k in range(15):
            p["Wz"][base + k] = a
            p["bz"][base + k] = b - k

        for bit in range(4):
            unit = 4 * bound + bit
            values = [1 if ((k >> bit) & 1) else -1 for k in range(16)]
            slopes = [values[k + 1] - values[k] for k in range(15)]
            p["Wmz"][unit, base] = 2 * slopes[0]
            for k in range(1, 15):
                p["Wmz"][unit, base + k] = 2 * (slopes[k] - slopes[k - 1])
            p["bm"][unit] = 2 * values[0]
            p["Wm"][unit, unit] = 20

    p["Wmz"][63, 29] = 8
    p["Wmz"][63, 30] = -8
    p["Wm"][63, 63] = 20

    p["Wmz"][16, 5] = 16
    p["Wmz"][17, 4] = 32
    p["Wmz"][19, 9] = 0.02
    p["Wmz"][20, 10] = 0.02

    p["Wm"][18, 63] = 32
    p["Wmz"][18, 4] = 32
    p["bm"][18] = -48

    # Successful movement flags: south, north, east, west.
    for unit, old_coord, current_z, sign in [
        (25, 19, 9, 1), (26, 19, 9, -1),
        (27, 20, 10, 1), (28, 20, 10, -1),
    ]:
        p["Wm"][unit, old_coord] = -10000 * sign
        p["Wmz"][unit, current_z] = 200 * sign
        p["bm"][unit] = -4

    # Inactive during an ordinary carry, active when empty or just grabbed.
    p["Wmz"][29, 4] = -32
    p["Wma"][29, 5] = 32
    p["bm"][29] = 16

    # Release cooldown.
    p["Wm"][30, 17] = 32
    p["Wmz"][30, 4] = -32
    p["bm"][30] = -16
    p["Wm"][62, 30] = 32
    p["Wm"][62, 63] = -32
    p["bm"][62] = -48

    # Pairwise comparisons preserve small spatial gradients.
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for k, (i, j) in enumerate(pairs):
        unit = 31 + k
        p["Wmz"][unit, i] = 1e30
        p["Wmz"][unit, j] = -1e30
        p["Wo"][1 + i, 128 + unit] = 1
        p["Wo"][1 + j, 128 + unit] = -1

    def outside_comparison(unit, bound, coord_z, lower, gain=16):
        sign = 1 if lower else -1
        for bit in range(4):
            p["Wm"][unit, 4 * bound + bit] += (
                gain * sign * 0.5 * (2 ** bit)
            )
        p["Wmz"][unit, coord_z] += -gain * sign * 15.75
        p["bm"][unit] += gain * (7.25 if lower else -8.25)

    validity_specs = [
        (0, 15, True), (1, 17, False),
        (2, 20, True), (3, 22, False),
    ]
    for t, (bound, coord_z, lower) in enumerate(validity_specs):
        outside_comparison(37 + t, bound, coord_z, lower)

    # Open-area target selection.
    for t in range(4):
        for d in range(4):
            unit = 41 + 4 * t + d
            base = 92 + 8 * t + 2 * d
            p["Wmz"][unit, base] = 16
            p["Wmz"][unit, base + 1] = -16
            p["Wm"][unit, 37 + t] = 16
            for earlier in range(t):
                p["Wm"][unit, 37 + earlier] = -16
            p["bm"][unit] = -24 - 16 * t
            p["Wmz"][unit, 4] = -256
            p["Wmz"][unit, 13] = -256
            p["Wmz"][unit, 27] = -256
            p["Wo"][1 + d, 128 + unit] = 6
            p["bo"][1 + d] += 6

    bound_specs = [
        (0, 12, True), (1, 12, False),
        (2, 11, True), (3, 11, False),
    ]

    # Single-goal inside tests.
    for k, (bound, coord_z, lower) in enumerate(bound_specs):
        unit = 21 + k
        outside_comparison(unit, bound, coord_z, lower)
        p["Wm"][unit] *= -1
        p["Wmz"][unit] *= -1
        p["bm"][unit] *= -1
        p["Wmz"][unit, 4] += 1000
        p["Wm"][unit, 63] += 500
        p["bm"][unit] -= 1500

    # Pickup eligibility.
    for k, (bound, coord_z, lower) in enumerate(bound_specs):
        unit = 57 + k
        outside_comparison(unit, bound, coord_z, lower)
        p["Wmz"][unit, 6] += 2000
        p["Wmz"][unit, 4] -= 2000
        p["Wm"][unit, 63] -= 500
        p["bm"][unit] -= 1500

    # Non-fence, multiple-goal handoff.
    p["Wm"][61, 16] = 32
    p["Wmz"][61, 4] = 32
    p["Wm"][61, 63] = -32
    p["Wmz"][61, 28] = -64
    p["Wma"][61, 5] = -64
    p["bm"][61] = -80

    # ------------------------------------------------------------
    # Action head.
    # ------------------------------------------------------------

    p["bo"][[0, 6, 7]] = -1e6

    # Weak geometric fallback when a held object obscures the goal.
    for action, unit in [(1, 24), (2, 23), (3, 22), (4, 21)]:
        p["Wo"][action, 128 + unit] -= 0.2

    # Avoid immediately undoing a successful move.
    for action, unit in [(1, 25), (2, 26), (3, 27), (4, 28)]:
        p["Wo"][action, 128 + unit] -= 4
        p["bo"][action] -= 4

    # Failed ordinary carrying moves favor releasing.
    for unit in range(25, 29):
        p["Wo"][5, 128 + unit] = -20
    p["Wo"][5, 128 + 29] = -60
    p["bo"][5] = -110

    # Release once all four remembered goal inequalities are satisfied.
    for unit in range(21, 25):
        p["Wo"][5, 128 + unit] = 10
    p["bo"][5] += 40
    p["Wo"][5, 128 + 18] = -30
    p["bo"][5] -= 30

    # Grab a front object satisfying at least one outside-goal test.
    for unit in range(57, 61):
        p["Wo"][5, 128 + unit] = 80
    p["bo"][5] += 320

    p["Wo"][5, 128 + 61] = 10
    p["bo"][5] += 10

    for unit in (30, 62):
        p["Wo"][5, 128 + unit] = -500
        p["bo"][5] -= 500

    p["Wo"][5, 7] = 200
    p["Wo"][5, 8] = 150

    return p
