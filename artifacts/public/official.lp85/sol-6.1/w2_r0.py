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

    def k1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def k2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def kh(out, inp, dy=0, dx=0, value=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    ordinary = (1, 2, 9, 10, 15)
    tile_colours = ordinary + (11, 12)

    # First layer: tile colours and coordinate planes.
    for colour in tile_colours:
        k1(0, colour)
    for colour in ordinary:
        k1(1, colour)
    k1(2, 11)
    k1(3, 12)

    k1(4, 17)
    k1(5, 16)
    k1(6, 17, value=-1.0)
    p["b1"][6] = 1.0
    k1(7, 16, value=-1.0)
    p["b1"][7] = 1.0

    # Unscaled corner-marker detectors.
    for out, colour in ((8, 11), (9, 12)):
        for dy in (-1, 2):
            for dx in (-1, 2):
                k1(out, colour, dy, dx)
        k1(out, colour, -1, 0, -2.0)
        k1(out, colour, 0, -1, -2.0)
        p["b1"][out] = -3.0

    # Shifted colour planes for scaled corner-marker detectors.
    quadrants = ((-2, -2), (-2, 2), (2, -2), (2, 2))
    for base, colour in ((10, 11), (14, 12)):
        for q, (dy, dx) in enumerate(quadrants):
            k1(base + q, colour, dy, dx)

    # Filled 2x2 and 3x3 goal-colour patches.
    for out, colour in ((18, 11), (19, 12)):
        for dy in (0, 1):
            for dx in (0, 1):
                k1(out, colour, dy, dx)
        p["b1"][out] = -3.0

    for out, colour in ((20, 11), (21, 12)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                k1(out, colour, dy, dx)
        p["b1"][out] = -8.0

    # Exact unscaled ordinary tiles provide rendering-scale evidence.
    for colour in ordinary:
        for dy in (0, 1):
            for dx in (0, 1):
                k1(22, colour, dy, dx)
        for dy, dx in ((-1, 0), (0, -1), (0, 2), (2, 0)):
            k1(22, colour, dy, dx, -4.0)
    p["b1"][22] = -3.0

    # Filled button patches exclude thin interface bars.
    for out, colour in ((23, 8), (24, 14)):
        for dy in (0, 1):
            for dx in (0, 1):
                k1(out, colour, dy, dx)
        p["b1"][out] = -3.0

    # Correctly occupied unscaled markers.
    for out, colour in ((25, 11), (26, 12)):
        for dy in (-1, 2):
            for dx in (-1, 2):
                k1(out, colour, dy, dx)
        for dy in (0, 1):
            for dx in (0, 1):
                k1(out, colour, dy, dx)
        k1(out, colour, -1, 0, -2.0)
        k1(out, colour, 0, -1, -2.0)
        p["b1"][out] = -7.0

    # Second layer: expanded track.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            k2(0, 0, dy, dx)

    k2(1, 8)
    k2(2, 9)

    shifts = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    # Scaled markers, excluding solid goal blocks.
    for out, base in ((3, 10), (4, 14)):
        for q, (dy, dx) in enumerate(shifts):
            k2(out, base + q, dy, dx)
        k2(out, base, -1, 1, -2.0)
        k2(out, base, 1, -1, -2.0)
        p["b2"][out] = -3.0

    for out, inp in (
        (5, 18), (6, 19), (7, 20), (8, 21),
        (10, 4), (11, 5),
    ):
        k2(out, inp)

    # Scale evidence expanded over neighbouring pixel rows/columns.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            k2(9, 22, dy, dx)

    # Track-conditioned coordinate extrema.
    for out, coord in ((12, 4), (13, 5), (14, 6), (15, 7)):
        k2(out, coord)
        k2(out, 0)
        p["b2"][out] = -1.0

    # Button masks reconstructed over filled button pixels.
    for out, inp in ((16, 23), (17, 24)):
        for dy in (0, -1):
            for dx in (0, -1):
                k2(out, inp, dy, dx)

    # Binary exclusion masks for filled goal patches.
    for out, small, large in ((18, 18, 20), (19, 19, 21)):
        p["b2"][out] = 1.0
        k2(out, small, value=-1.0)
        k2(out, large, value=-1.0)

    for out, large in ((20, 20), (21, 21)):
        p["b2"][out] = 1.0
        k2(out, large, value=-1.0)

    # Correctly occupied scaled markers.
    for out, base, fill in ((22, 10, 20), (23, 14, 21)):
        for q, (dy, dx) in enumerate(shifts):
            k2(out, base + q, dy, dx)
        k2(out, base, -1, 1, -2.0)
        k2(out, base, 1, -1, -2.0)
        k2(out, fill)
        p["b2"][out] = -4.0

    k2(24, 25)
    k2(25, 26)

    W3 = p["W3"]
    b3 = p["b3"]

    # f3[0]: outside the expanded tile network.
    b3[0] = 1.0
    W3[0, 0] = -1.0

    # f3[1:3]: unmatched marker sources, separately by colour.
    W3[1, 1] = 1.0
    W3[1, 3] = 1.0
    W3[1, 22] = -2.0
    W3[1, 24] = -2.0

    W3[2, 2] = 1.0
    W3[2, 4] = 1.0
    W3[2, 23] = -2.0
    W3[2, 25] = -2.0

    # Clipped valid-goal indicators:
    # colour 11 = f3[3] - f3[4];
    # colour 12 = f3[5] - f3[6].
    for raw, overflow, small_gate, large_gate, ms, mu in (
        (3, 4, 18, 20, 22, 24),
        (5, 6, 19, 21, 23, 25),
    ):
        b3[raw] = 1.0
        W3[raw, small_gate] = -2.0
        W3[raw, large_gate] = -1.0
        W3[raw, 32 + 9] = 1.0
        W3[raw, 64 + 9] = 1.0
        W3[raw, ms] = -4.0
        W3[raw, mu] = -4.0
        W3[overflow] = W3[raw]
        b3[overflow] = b3[raw] - 1.0

    # Geometric side tests. The offset gives degenerate straight
    # tracks a consistent eastward/northward forward convention.
    offset = 1.1 / 63.0

    # f3[7]: below the horizontal midpoint.
    W3[7, 11] = 1.0
    W3[7, 64 + 13] = -0.5
    W3[7, 64 + 15] = 0.5
    b3[7] = -0.5 - offset

    # f3[8]: above the horizontal midpoint.
    W3[8] = -W3[7]
    b3[8] = -b3[7]

    # f3[9]: right of the vertical midpoint.
    W3[9, 10] = 1.0
    W3[9, 32 + 12] = -0.5
    W3[9, 32 + 14] = 0.5
    b3[9] = -0.5 - offset

    # f3[10]: left of the vertical midpoint.
    W3[10] = -W3[9]
    b3[10] = -b3[9]

    # Button masks and click exclusion.
    W3[11, 16] = 1.0                 # L button pixels
    W3[12, 17] = 1.0                 # R button pixels
    b3[13] = 1.0
    W3[13, 16] = -1.0
    W3[13, 17] = -1.0

    # A modest forward preference on tall, narrow scaled routes.
    W3[14, 64 + 13] = 1.0
    W3[14, 64 + 15] = 1.0
    W3[14, 32 + 12] = -2.0
    W3[14, 32 + 14] = -2.0
    W3[14, 20] = -1024.0
    W3[14, 21] = -1024.0
    # A large patch of either colour is enough.
    W3[14, 7] = 1024.0
    W3[14, 8] = 1024.0
    for inp in (22, 23, 24, 25):
        W3[14, inp] = -4096.0
    b3[14] = 1.0 - 1024.0 - 0.1

    # h[0]-h[1]: bounded colour-11 flood; h[2]: arrival age.
    # h[3]-h[4]: bounded colour-12 flood; h[5]: arrival age.
    for raw, clipped, age, seed in (
        (0, 1, 2, 1),
        (3, 4, 5, 2),
    ):
        for out in (raw, clipped):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    kh(out, raw, dy, dx)
                    kh(out, clipped, dy, dx, -1.0)
            p["Uh"][out, 0] = -10.0
            p["Uh"][out, seed] = 1.0
        p["bh"][clipped] = -1.0

        kh(age, age)
        kh(age, raw)
        kh(age, clipped, value=-1.0)

    # Four quadrant-specific clockwise comparisons per goal colour.
    # Their common baseline permits signed gradients through ReLU.
    baseline = 256.0
    gate_strength = 1.0e6
    goal_strength = 32768.0

    # (east/west sign, south/north sign, wrong-y gate, wrong-x gate)
    quadrants = (
        (1.0, -1.0, 7, 9),   # upper left
        (1.0, 1.0, 7, 10),   # upper right
        (-1.0, -1.0, 8, 9),  # lower left
        (-1.0, 1.0, 8, 10),  # lower right
    )

    for first, age, goal_raw, goal_overflow in (
        (6, 2, 3, 4),
        (10, 5, 5, 6),
    ):
        for q, (sx, sy, gy, gx) in enumerate(quadrants):
            out = first + q
            kh(out, age, 0, 1, 0.5 * sx)
            kh(out, age, 0, -1, -0.5 * sx)
            kh(out, age, 1, 0, 0.5 * sy)
            kh(out, age, -1, 0, -0.5 * sy)

            p["bh"][out] = -goal_strength
            p["Uh"][out, goal_raw] = goal_strength + baseline
            p["Uh"][out, goal_overflow] = -goal_strength - baseline
            p["Uh"][out, gy] = -gate_strength
            p["Uh"][out, gx] = -gate_strength

    # h[14] and h[15]: clockwise/counterclockwise request fields.
    # Attenuation removes the initial baseline transient and makes
    # button ownership local rather than globally pooled.
    propagation = 0.8 / 9.0
    for out in (14, 15):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                kh(out, out, dy, dx, propagation)

    scale = 0.01
    for source in range(6, 14):
        kh(14, source, value=scale)
        kh(15, source, value=-scale)

    for raw, overflow in ((3, 4), (5, 6)):
        p["Uh"][14, raw] = -scale * baseline + 1.0e-5
        p["Uh"][14, overflow] = scale * baseline - 1.0e-5
        p["Uh"][15, raw] = scale * baseline
        p["Uh"][15, overflow] = -scale * baseline

    p["Uh"][14, 14] = 0.005

    # Block clockwise requests at L buttons and vice versa.
    p["Uh"][14, 11] = -10000.0
    p["Uh"][15, 12] = -10000.0

    p["Wc"][0, 32 + 14] = 1.0
    p["Wc"][0, 32 + 15] = 1.0
    p["Wc"][0, 13] = -1.0e6

    # Always select the available click action.
    p["bo"][6] = 1.0
    return p
