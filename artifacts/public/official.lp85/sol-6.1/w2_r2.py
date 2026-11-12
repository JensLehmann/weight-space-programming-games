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

    tile_colours = (0, 1, 2, 6, 7, 9, 10, 11, 12, 13, 15)

    def w1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def w2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def wh(out, inp, dy=0, dx=0, value=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    def rectangle(out, colour, ys, xs):
        for dy in ys:
            for dx in xs:
                w1(out, colour, dy, dx)
        p["b1"][out] = -len(ys) * len(xs) + 0.5

    def pool(out, inp, ys=(-1, 0, 1), xs=(-1, 0, 1)):
        for dy in ys:
            for dx in xs:
                w2(out, inp, dy, dx)

    def affine(out, terms, bias=0.0):
        for inp, value in terms.items():
            p["W3"][out, inp] += value
        p["b3"][out] = bias

    def clipped(out, terms, bias=0.0):
        # Output difference equals clip(affine expression, 0, 1).
        affine(out, terms, bias)
        affine(out + 1, terms, bias - 1.0)

    def drive_difference(out, first, value):
        p["Uh"][out, first] += value
        p["Uh"][out, first + 1] -= value

    # ------------------------------------------------------------
    # W1: local geometry and coordinates.
    # ------------------------------------------------------------

    for colour in tile_colours:
        w1(0, colour)

    # Small goals, anchored at the NW pixel.
    rectangle(1, 11, (0, 1), (0, 1))
    rectangle(2, 12, (0, 1), (0, 1))

    # Large goals, anchored at the NW inner pixel.
    rectangle(3, 11, (-1, 0, 1, 2), (-1, 0, 1, 2))
    rectangle(4, 12, (-1, 0, 1, 2), (-1, 0, 1, 2))

    # Scale-two witnesses.
    for out, dy, xs in (
        (5, 2, (0, 1, 2)),
        (6, -2, (-2, -1, 0)),
    ):
        for dx in xs:
            for colour in tile_colours:
                w1(out, colour, dy, dx)
        p["b1"][out] = -2.5

    # Small target frames.
    for out, colour in ((7, 11), (8, 12)):
        for dy, dx in ((-1, -1), (-1, 2), (2, -1), (2, 2)):
            w1(out, colour, dy, dx)
        p["b1"][out] = -3.5

    # Shifted planes for large target-frame detection.
    corners = ((-2, -2), (-2, 2), (2, -2), (2, 2))
    for start, colour in ((9, 11), (13, 12)):
        for j, (dy, dx) in enumerate(corners):
            w1(start + j, colour, dy, dx)

    w1(17, 16)
    w1(18, 17)

    # Foreground-gated coordinates and complements.
    for out, coordinate, sign, bias in (
        (19, 16, 1.0, -1.0),
        (20, 16, -1.0, 0.0),
        (21, 17, 1.0, -1.0),
        (22, 17, -1.0, 0.0),
    ):
        w1(out, coordinate, value=sign)
        for colour in tile_colours:
            w1(out, colour)
        p["b1"][out] = bias

    rectangle(23, 8, (-1, 0), (-1, 0))
    rectangle(24, 14, (-1, 0), (-1, 0))

    # Binary colour neighbourhoods, represented by ReLU differences.
    for first, colour in ((25, 11), (27, 12)):
        for out in (first, first + 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    w1(out, colour, dy, dx)
        p["b1"][first + 1] = -1.0

    # ------------------------------------------------------------
    # W2: corridors, symmetric goal cores and frame centres.
    # ------------------------------------------------------------

    pool(0, 0)

    for out, inp in ((1, 1), (2, 2), (3, 3), (4, 4)):
        pool(out, inp, (-1, 0), (-1, 0))

    w2(7, 25)
    w2(7, 26, value=-1.0)
    w2(8, 27)
    w2(8, 28, value=-1.0)

    pool(9, 5)
    pool(9, 6)

    pool(10, 7, (-1, 0), (-1, 0))
    pool(11, 8, (-1, 0), (-1, 0))

    for out, start in ((12, 9), (13, 13)):
        for inp, ys, xs in (
            (start, (-1, 0), (-1, 0)),
            (start + 1, (-1, 0), (0, 1)),
            (start + 2, (0, 1), (-1, 0)),
            (start + 3, (0, 1), (0, 1)),
        ):
            pool(out, inp, ys, xs)
        p["b2"][out] = -8.5

    for out, inp in (
        (14, 17), (15, 18),
        (16, 19), (17, 20),
        (18, 21), (19, 22),
        (20, 23), (21, 24),
    ):
        w2(out, inp)

    # ------------------------------------------------------------
    # W3: masks, scale selection and approximate loop orientation.
    # ------------------------------------------------------------

    row = lambda c: 32 + c
    col = lambda c: 64 + c
    scale = row(9)

    clipped(0, {0: 1.0})

    # Target seeds.
    affine(2, {10: 2.0, scale: -128.0})
    affine(3, {11: 2.0, scale: -128.0})
    affine(4, {12: 2.0})
    affine(5, {13: 2.0})

    # Colour-matched sampling neighbourhoods.
    affine(8, {7: 1.0})
    affine(9, {8: 1.0})

    # Unsatisfied goal cores.
    clipped(10, {1: 2.0, 10: -4.0, scale: -128.0})
    clipped(12, {2: 2.0, 11: -4.0, scale: -128.0})
    clipped(14, {3: 2.0, 12: -4.0})
    clipped(16, {4: 2.0, 13: -4.0})

    sharpness = 256.0

    # Complementary half-plane tests. The offset makes an isolated
    # horizontal path use R=right, and a vertical path use R=up.
    offset = 2.5 / 63.0
    top = {
        col(16): sharpness,
        col(17): -sharpness,
        14: -2.0 * sharpness,
    }
    left = {
        row(18): sharpness,
        row(19): -sharpness,
        15: -2.0 * sharpness,
    }

    clipped(18, top, sharpness * (1.0 + offset))
    clipped(
        20,
        {c: -v for c, v in top.items()},
        -sharpness * (1.0 + offset),
    )
    clipped(22, left, sharpness * (1.0 + offset))
    clipped(
        24,
        {c: -v for c, v in left.items()},
        -sharpness * (1.0 + offset),
    )

    clipped(26, {20: 2.0})  # L button interiors
    clipped(28, {21: 2.0})  # R button interiors

    # ------------------------------------------------------------
    # Wh: bounded wavefronts and arrival-time accumulators.
    # ------------------------------------------------------------

    def wavefront(first, accumulator, source_channels):
        # q = h[first] - h[first+1] is a binary reached mask.
        # ReLU(s) - ReLU(s-1) clips the neighbour sum to [0,1].
        for out in (first, first + 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    wh(out, first, dy, dx, 1.0)
                    wh(out, first + 1, dy, dx, -1.0)

            drive_difference(out, 0, 64.0)
            for source in source_channels:
                p["Uh"][out, source] += 1.0

        p["bh"][first] = -64.0
        p["bh"][first + 1] = -65.0

        # After T iterations, this is approximately T-arrival_time.
        wh(accumulator, accumulator)
        wh(accumulator, first)
        wh(accumulator, first + 1, value=-1.0)

    wavefront(0, 2, (2, 4))
    wavefront(3, 5, (3, 5))

    # h6,h7: sample only the field matching the goal colour.
    for out, field, mask in ((6, 2, 8), (7, 5, 9)):
        wh(out, field)
        p["Uh"][out, mask] = 256.0
        p["bh"][out] = -256.0

    # h8..h11: signed directional derivatives plus a small baseline.
    # With complementary orientation masks, two components are
    # present at each goal core.
    baseline = 4.0
    gradient_gate = 4096.0

    components = (
        (8, 18, (0, 1), (0, -1)),   # top: R=right
        (9, 20, (0, -1), (0, 1)),   # bottom: R=left
        (10, 22, (-1, 0), (1, 0)),  # left: R=up
        (11, 24, (1, 0), (-1, 0)),  # right: R=down
    )

    for out, orientation, positive, negative in components:
        for matched in (6, 7):
            wh(out, matched, *positive, value=1.0)
            wh(out, matched, *negative, value=-1.0)

        drive_difference(out, orientation, gradient_gate)
        for core in (10, 12, 14, 16):
            drive_difference(out, core, gradient_gate)
        p["bh"][out] = baseline - 2.0 * gradient_gate

    # h12-h13 is a signed spatial broadcast. ReLU(s)-ReLU(-s)
    # preserves its sign without exponential amplification.
    for out, sign in ((12, 1.0), (13, -1.0)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                weight = 0.2 if (dy == 0 and dx == 0) else 0.1
                wh(out, 12, dy, dx, sign * weight)
                wh(out, 13, dy, dx, -sign * weight)

        for component in (8, 9, 10, 11):
            wh(out, component, value=sign)

        for core in (10, 12, 14, 16):
            drive_difference(out, core, -sign * 2.0 * baseline)

    # h14,h15: expose the signed preference on matching controls.
    button_gate = 65536.0

    wh(14, 12)
    wh(14, 13, value=-1.0)
    drive_difference(14, 28, button_gate)
    p["bh"][14] = -button_gate

    wh(15, 13)
    wh(15, 12, value=-1.0)
    drive_difference(15, 26, button_gate)
    p["bh"][15] = -button_gate

    p["Wc"][0, 46] = 1.0
    p["Wc"][0, 47] = 1.0

    # A zero-preference tie still selects a valid button.
    p["Wc"][0, 26] = 1.0e-12
    p["Wc"][0, 27] = -1.0e-12

    p["bo"][6] = 1.0
    return p
