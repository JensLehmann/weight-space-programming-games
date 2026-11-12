import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5),
        "b1": (32,),
        "W2": (32, 32, 3, 3),
        "b2": (32,),
        "W3": (32, 96),
        "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32),
        "bh": (16,),
        "Wz": (128, 96),
        "bz": (128,),
        "Wm": (64, 64),
        "Wmz": (64, 128),
        "Wma": (64, 8),
        "bm": (64,),
        "Wo": (8, 192),
        "bo": (8,),
        "Wc": (1, 48),
        "bc": (1,),
    }
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }
    B = float(2 ** 20)

    def c1(out, terms, bias=0.0):
        for channel, dy, dx, weight in terms:
            p["W1"][out, channel, dy + 2, dx + 2] += weight
        p["b1"][out] = bias

    def c2(out, terms, bias=0.0):
        for channel, dy, dx, weight in terms:
            p["W2"][out, channel, dy + 1, dx + 1] += weight
        p["b2"][out] = bias

    def point(out, terms, bias=0.0):
        for channel, weight in terms:
            p["W3"][out, channel] += weight
        p["b3"][out] = bias

    def rec(out, channel, dy=0, dx=0, weight=1.0):
        p["Wh"][out, channel, dy + 1, dx + 1] += weight

    def drive(out, channel, weight):
        p["Uh"][out, channel] += weight

    def gate(dy, dx, weight):
        return [(1, dy, dx, weight), (12, dy, dx, weight)]

    def coloured(dy, dx, weight):
        return [(11, dy, dx, weight), (14, dy, dx, weight)]

    # ------------------------------------------------------------
    # First visual layer.
    # ------------------------------------------------------------

    # Water/control coordinate extrema for gravity orientation.
    c1(0, [(0, 0, 0, 1), (9, 0, 0, 1),
           (16, 0, 0, 1)], -1)
    c1(1, [(0, 0, 0, 1), (9, 0, 0, 1),
           (16, 0, 0, -1)])
    c1(2, [(0, 0, 0, 1), (9, 0, 0, 1),
           (17, 0, 0, 1)], -1)
    c1(3, [(0, 0, 0, 1), (9, 0, 0, 1),
           (17, 0, 0, -1)])

    # Player: yellow near a coloured base.
    terms = [(4, 0, 0, 20)]
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            terms += coloured(dy, dx, 1)
    c1(4, terms, -20)

    # Goal marks embedded in vertical/horizontal separators.
    c1(5, coloured(0, 0, 10) +
       [(5, -2, 0, 1), (5, 2, 0, 1)], -11)
    c1(6, coloured(0, 0, 10) +
       [(5, 0, -2, 1), (5, 0, 2, 1)], -11)

    # Gate orientation.
    c1(7, gate(0, 0, 4) +
       gate(-2, 0, 1) + gate(2, 0, 1) +
       gate(0, -2, -1) + gate(0, 2, -1), -4)
    c1(8, gate(0, 0, 4) +
       gate(0, -2, 1) + gate(0, 2, 1) +
       gate(-2, 0, -1) + gate(2, 0, -1), -4)

    def orange_h(out, sx):
        c1(out, [(9, 0, sx, 1), (9, -1, sx, 1),
                 (9, -2, sx, -1)], -1)

    def orange_v(out, sy):
        c1(out, [(9, sy, 0, 1), (9, sy, -1, 1),
                 (9, sy, -2, -1)], -1)

    orange_h(9, -2)
    orange_h(10, 2)
    orange_v(11, -2)
    orange_v(12, 2)
    orange_h(29, 0)
    orange_v(31, 0)

    c1(13, [(5, 0, 0, 1)])
    c1(14, gate(0, 0, 1))
    c1(15, [(3, 0, 1, 1), (3, 1, 0, 1)])
    c1(16, [(0, 0, 0, 1)])
    c1(17, [(3, 0, -1, 1), (3, -1, 0, 1)])
    c1(18, [(9, 0, 0, 1)])

    # Two-pixel endpoint strips.
    for out, (dy, dx) in enumerate(
        [(2, 0), (-2, 0), (0, 2), (0, -2)], 19
    ):
        c1(out, gate(0, 0, 1) + gate(dy, dx, -1))

    # Air outside the water-boundary neighbourhood.
    terms = [(3, 0, 0, 9)]
    for d in (-2, -1, 1, 2):
        terms += [(0, d, 0, -1), (0, 0, d, -1)]
    c1(23, terms, -8)

    # UI row.
    c1(24, [(16, 0, 0, -64)], 1)

    # Wall-facing orange edges.
    for out, (dy, dx) in enumerate(
        [(0, -1), (0, 1), (-1, 0), (1, 0)], 25
    ):
        c1(out, [(9, 0, 0, 1), (5, dy, dx, 1)], -1)

    c1(30, [(12, 0, 0, 1)])

    # ------------------------------------------------------------
    # Second visual layer.
    # ------------------------------------------------------------

    for i in range(9):
        c2(i, [(i, 0, 0, 1)])

    # Paired controls on a single interior scan line.
    c2(9, [(13, 0, 0, 10), (9, 0, -1, 1),
           (10, 0, 1, 1)], -11)
    c2(10, [(13, 0, 0, 10), (11, -1, 0, 1),
            (12, 1, 0, 1)], -11)

    for out, source in [
        (11, 13), (12, 14), (13, 15),
        (14, 24), (15, 17), (16, 18),
    ]:
        c2(out, [(source, 0, 0, 1)])

    # Raise/lower probes beside goals and gate endpoints.
    for orientation in range(4):
        vertical = orientation < 2
        endpoint = 19 + orientation
        goal_channel = 5 if vertical else 6
        neighbours = (
            [(0, -1), (0, 1)]
            if vertical else [(-1, 0), (1, 0)]
        )
        for lower in range(2):
            eligible = 16 if lower else 23
            terms = [(eligible, 0, 0, 10)]
            for dy, dx in neighbours:
                terms += [
                    (endpoint, dy, dx, 1),
                    (goal_channel, dy, dx, 1),
                ]
            c2(17 + 2 * orientation + lower, terms, -10)

    # Edge signs; axial features restore orientation.
    c2(25, [(25, 0, 0, 1), (27, 0, 0, 1)])
    c2(26, [(26, 0, 0, 1), (28, 0, 0, 1)])

    # Shifted water planes for axial donor-presence tests.
    c2(27, [(16, -1, -1, 1)])
    c2(28, [(16, 1, 1, 1)])
    c2(29, [(30, 0, 0, 1)])

    horizontal = [(29, 0, 0, 1)]
    vertical = [(31, 0, 0, 1)]
    for dx in (-1, 0, 1):
        horizontal += [(9, 0, dx, 1), (10, 0, dx, 1)]
    for dy in (-1, 0, 1):
        vertical += [(11, dy, 0, 1), (12, dy, 0, 1)]
    c2(30, horizontal)
    c2(31, vertical)

    # ------------------------------------------------------------
    # Axial masks.
    # ------------------------------------------------------------

    # Local c, row maximum 32+c, column maximum 64+c.
    barrier = [
        (11, 1), (12, 1), (14, 1), (5, 1), (6, 1),
        (69, 1), (71, 1), (73, 1),
        (38, 1), (40, 1), (42, 1),
    ]
    point(0, barrier)

    # Goal reachability may pass through gates and goal marks.
    point(1, barrier + [(12, -32), (5, -32), (6, -32)])

    # Hydraulic paths require both local and axial permission.
    point(2, barrier + [(30, -256), (31, -256)])
    point(31, barrier + [(41, -256), (74, -256)])

    point(3, [(11, -1), (16, -1)], 1)
    point(4, [(41, -1), (74, -1)], 1)
    point(5, [(4, 1)])
    point(6, [(5, 1), (6, 1)])
    point(7, [(12, 1), (5, 1), (6, 1)])

    # Empty-donor seeds, with axial correction for offset controls.
    point(8, [(15, 4), (59, -1), (91, -1)], -2)
    point(9, [(13, 4), (60, -1), (92, -1)], -2)

    # Invalid click colour.
    point(10, [(16, -1), (29, -1)], 1)

    # Complementary action-edge masks.
    point(11, [(25, -1), (74, 1)], 1)
    point(12, [(26, -1), (74, 1)], 1)
    point(13, [(25, -1), (41, 1)], 1)
    point(14, [(26, -1), (41, 1)], 1)

    # Clipped probe eligibility: ReLU(v) - ReLU(v-1).
    for i in range(8):
        orientation = i // 2
        positive = orientation in (0, 2)
        sign = 1 if positive else -1
        hi, lo = (64, 65) if orientation < 2 else (34, 35)
        terms = [
            (17 + i, 1024),
            (hi, sign * 256),
            (lo, -sign * 256),
        ]
        bias = -1024 + (1 if positive else 0)
        point(15 + 2 * i, terms, bias)
        point(16 + 2 * i, terms, bias - 1)

    # ------------------------------------------------------------
    # Recurrent spatial circuit.
    # ------------------------------------------------------------

    n4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    n8 = [
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx
    ]

    # 0-1: clipped player-compartment flood.
    for out in (0, 1):
        for dy, dx in n8:
            rec(out, 0, dy, dx, 1)
            rec(out, 1, dy, dx, -1)
        drive(out, 5, 1)
        drive(out, 0, -B)
    p["bh"][1] = -1

    # 2-3: clipped goal reachability outside player compartments.
    for out in (2, 3):
        for dy, dx in n4:
            rec(out, 2, dy, dx, 1)
            rec(out, 3, dy, dx, -1)
        rec(out, 0, weight=-100)
        rec(out, 1, weight=100)
        drive(out, 6, 1)
        drive(out, 1, -B)
    p["bh"][3] = -1

    # 4-5: selected-gate relevance.
    # Goal marks retain relevance only on the player-facing edge.
    for out in (4, 5):
        for dy, dx in n8:
            rec(out, 4, dy, dx, 1)
            rec(out, 5, dy, dx, -1)
        for dy, dx in n4:
            rec(out, 0, dy, dx, 10)
            rec(out, 1, dy, dx, -10)
        rec(out, 2, weight=B)
        rec(out, 3, weight=-B)
        drive(out, 7, B)
        drive(out, 6, -8)
        p["bh"][out] = -2 * B
    p["bh"][5] -= 1

    # 6-7: signed demand beside relevant targets.
    for out, parity in [(6, 0), (7, 1)]:
        for dy, dx in n4:
            rec(out, 4, dy, dx, 1)
            rec(out, 5, dy, dx, -1)
        for i in range(parity, 8, 2):
            drive(out, 15 + 2 * i, B)
            drive(out, 16 + 2 * i, -B)
        p["bh"][out] = -B

    # 8-9: positive and negative hydraulic pressure.
    for out, source in [(8, 6), (9, 7)]:
        for dy, dx in n8:
            rec(out, out, dy, dx, 0.13)
        rec(out, source, weight=1)
        drive(out, 2, -B)
        drive(out, 31, -B)

    # 10-11: directed donor scans.
    #
    # Large empty-donor signals survive the short scan corridor.
    # Small negative-pressure signals survive only inside the
    # player compartment; the per-pixel threshold removes them
    # on entering a separator or another compartment.
    scan_threshold = 32.0
    empty_amplitude = 32.0 * scan_threshold

    rec(10, 10, 0, -1, 1)
    rec(10, 10, -1, 0, 1)
    drive(10, 8, empty_amplitude)

    rec(11, 11, 0, 1, 1)
    rec(11, 11, 1, 0, 1)
    drive(11, 9, empty_amplitude)

    for out in (10, 11):
        rec(out, 0, weight=scan_threshold)
        rec(out, 1, weight=-scan_threshold)
        rec(out, 9, weight=1)
        p["bh"][out] = -scan_threshold
        drive(out, 3, -B)
        drive(out, 4, -B)

    # 12-15: directed transfer scores.
    configurations = [
        (12, 11, 10, (0, -1), (0, 1)),
        (13, 12, 11, (0, 1), (0, -1)),
        (14, 13, 10, (-1, 0), (1, 0)),
        (15, 14, 11, (1, 0), (-1, 0)),
    ]
    for out, edge_mask, scan, wall, own in configurations:
        rec(out, 8, own[0], own[1], 1)
        rec(out, 8, wall[0], wall[1], -1)
        rec(out, 9, own[0], own[1], -1)
        rec(out, 9, wall[0], wall[1], 1)

        # Hard donor veto at the separator-facing neighbour.
        rec(out, scan, wall[0], wall[1], -B)

        # Soft penalty for returning surplus to the player reservoir.
        rec(out, scan, own[0], own[1], -1)

        drive(out, edge_mask, -B)
        drive(out, 4, -B)

    # Click-only policy. The global trunk and memory remain unused.
    p["bo"][6] = 10
    p["Wc"][0, 10] = -B

    # Ready, relevant gates outrank pressure-driven transfers.
    p["Wc"][0, 32 + 4] = 4096
    p["Wc"][0, 32 + 5] = -4096
    for channel in range(12, 16):
        p["Wc"][0, 32 + channel] = 1

    return p
