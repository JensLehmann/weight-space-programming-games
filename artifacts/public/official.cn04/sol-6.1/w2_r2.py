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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    colour_groups = (
        (1, 10),
        (2, 11),
        (6, 12),
        (7, 14),
        (5, 9, 15),
    )

    def c1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def c2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def c3(out, inp, value=1.0):
        p["W3"][out, inp] += value

    # Basic colour planes.
    for out, colour in enumerate((8, 3, 4, 0)):
        c1(out, colour)

    # Exactly one on display row zero.
    c1(4, 18)
    c1(4, 16, value=-63.0)

    # Clipped proximity to a black wire, gated by a terminal.
    # Channel 5 minus channel 6 is a binary proximity test.
    for out, bias in ((5, -9.0), (6, -10.0)):
        c1(out, 8, value=9.0)
        c1(out, 3, value=9.0)
        for dy, dx in directions:
            c1(out, 0, dy, dx)
            c1(out, 0, 2 * dy, 2 * dx)
        p["b1"][out] = bias

    # Candidate terminal masks translated by two pixels here,
    # and by the third pixel in layer two.
    shifted_channels = (7, 8, 30, 31)
    for channel, (dy, dx) in zip(shifted_channels, directions):
        c1(channel, 8, -2 * dy, -2 * dx)
        c1(channel, 3, -2 * dy, -2 * dx)

    # Grey skeleton neighbours at cell-scale offsets.
    for k, (dy, dx) in enumerate(directions):
        c1(9 + k, 4, 2 * dy, 2 * dx)
        c1(9 + k, 3, 2 * dy, 2 * dx)

    # Coloured-wire proximity tests and homogeneous-region tests.
    # Homogeneous regions identify the background colour group.
    for k, group in enumerate(colour_groups):
        a, b, homogeneous = 13 + 3 * k, 14 + 3 * k, 15 + 3 * k
        for out, bias in ((a, -9.0), (b, -10.0)):
            c1(out, 8, value=9.0)
            c1(out, 3, value=9.0)
            for colour in group:
                for dy, dx in directions:
                    c1(out, colour, dy, dx)
                    c1(out, colour, 2 * dy, 2 * dx)
            p["b1"][out] = bias

        for colour in group:
            p["W1"][homogeneous, colour, :, :] = 1.0
        p["b1"][homogeneous] = -24.0

    c1(28, 16)
    c1(29, 17)

    # Layer two: raw red and green terminals.
    c2(0, 0)
    c2(1, 1)

    # Grey cells with exactly one cardinal skeleton neighbour.
    for out, bias in ((2, -8.0), (3, -9.0)):
        c2(out, 2, value=10.0)
        c2(out, 4, value=-10.0)
        for k, (dy, dx) in enumerate(directions):
            c2(out, 9 + k, dy, dx, -1.0)
        p["b2"][out] = bias

    # Expand the black-wire proximity test over each 3x3 terminal.
    for out, bias in ((4, -10.0), (5, -11.0)):
        c2(out, 0, value=10.0)
        c2(out, 1, value=10.0)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                c2(out, 5, dy, dx)
                c2(out, 6, dy, dx, -1.0)
        p["b2"][out] = bias

    c2(6, 4)

    # Remove the progress display from grey and black planes.
    c2(8, 2)
    c2(8, 4, value=-1.0)
    c2(9, 3)
    c2(9, 4, value=-1.0)

    for k, (channel, (dy, dx)) in enumerate(
            zip(shifted_channels, directions)):
        c2(10 + k, channel, -dy, -dx)

    c2(14, 0)
    c2(14, 1)
    c2(15, 28)
    c2(16, 29)

    # Expand each coloured-wire proximity test over the terminal.
    for k in range(5):
        a1, b1, hom1 = 13 + 3 * k, 14 + 3 * k, 15 + 3 * k
        a2, b2 = 17 + 2 * k, 18 + 2 * k

        for out, bias in ((a2, -10.0), (b2, -11.0)):
            c2(out, 0, value=10.0)
            c2(out, 1, value=10.0)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    c2(out, a1, dy, dx)
                    c2(out, b1, dy, dx, -1.0)
            p["b2"][out] = bias

        c2(27 + k, hom1)

    # Layer three semantic masks.
    c3(0, 2)
    c3(0, 3, -2.0)                  # Unconnected grey endpoints.

    c3(1, 14)                       # All visible terminals.

    c3(2, 4)
    c3(2, 5, -1.0)                  # Terminals beside selected black wire.

    c3(3, 0)
    c3(3, 4)
    c3(3, 5, -1.0)
    p["b3"][3] = -1.0              # Selected red terminals.

    c3(5, 8)                        # Grey-masking evidence.
    c3(6, 9)                        # Black growth marker.

    for k in range(5):
        out = 7 + k
        c3(out, 17 + 2 * k)
        c3(out, 18 + 2 * k, -1.0)
        c3(out, 32 + 27 + k, -1.0)
        c3(out, 64 + 27 + k, -1.0)

    c3(12, 1)                       # Green contacts.
    c3(13, 0)                       # Red terminals.

    for k in range(4):
        c3(27 + k, 10 + k)          # Whole-cell candidate terminal masks.

    c3(19, 15)
    c3(20, 16)
    c3(21, 8)

    # Aggregate coloured-wire endpoints. In unmasked play these
    # belong to the other pieces, even when near the black piece.
    for out in (4, 14):
        for k in range(5):
            c3(out, 17 + 2 * k)
            c3(out, 18 + 2 * k, -1.0)
            c3(out, 32 + 27 + k, -1.0)
            c3(out, 64 + 27 + k, -1.0)

    # Coordinate-weighted selected masks.
    for axis, coordinate in enumerate((15, 16)):
        out = 15 + axis
        c3(out, coordinate)
        c3(out, 4)
        c3(out, 5, -1.0)
        p["b3"][out] = -1.0

    # Coordinate-weighted coloured target masks.
    for axis, coordinate in enumerate((15, 16)):
        out = 17 + axis
        c3(out, coordinate)
        for k in range(5):
            c3(out, 17 + 2 * k)
            c3(out, 18 + 2 * k, -1.0)
            c3(out, 32 + 27 + k, -1.0)
            c3(out, 64 + 27 + k, -1.0)
        p["b3"][out] = -1.0

    # Recurrent fields use A-B as clipped binary reachability.
    # The integral equals approximately 127 minus Manhattan distance.
    def make_field(base, seeds):
        a, b, integral = base, base + 1, base + 2
        for out in (a, b):
            for dy, dx in directions:
                p["Wh"][out, a, dy + 1, dx + 1] = 1.0
                p["Wh"][out, b, dy + 1, dx + 1] = -1.0
            for feature in seeds:
                p["Uh"][out, feature] += 2.0
        p["bh"][b] = -1.0

        p["Wh"][integral, integral, 1, 1] = 1.0
        p["Wh"][integral, a, 1, 1] = 1.0
        p["Wh"][integral, b, 1, 1] = -1.0

    make_field(0, (0, 12))           # Grey targets and green anchors.
    make_field(8, (1,))             # Distance from visible terminals.

    # Same-age samples of the target field under current and candidate
    # masks. Whole-cell sampling avoids one-pixel overshoot estimates.
    for out, feature in (
        (3, 1), (4, 27), (5, 28), (6, 29), (7, 30)
    ):
        p["Wh"][out, 2, 1, 1] = 1.0
        p["Uh"][out, feature] = 1000.0
        p["bh"][out] = -1000.0

    # Masked translation scores.
    # Fixed green terminals appear in all translated masks; compensate
    # most, but not all, of their fixed-anchor distance penalty.
    for k in range(4):
        p["Wz"][k, 16 + 4 + k] = 4096.0
        p["Wz"][k, 16 + 3] = -4096.0
        p["Wz"][k, 64 + 12] = 2.5 * 4096.0
        p["Wz"][k, 32 + 5] = 100000.0
        p["bz"][k] = -100000.5
        p["Wo"][1 + k, k] = 1.0

    # Switch a fully connected masked piece if free grey ends remain.
    p["Wz"][8, 32 + 5] = 1.0
    p["Wz"][8, 32 + 0] = 1.0
    p["Wz"][8, 32 + 13] = -1.0
    p["bz"][8] = -1.0
    p["Wo"][6, 8] = 500.0

    # Expansion is requested only when grey masking and a black
    # growth marker are both present.
    p["Wz"][10, 32 + 5] = 1.0
    p["Wz"][10, 32 + 6] = 1.0
    p["bz"][10] = -1.0
    p["Wo"][5, 10] = 10000.0

    # Objective and mode for detecting unsuccessful masked movement.
    p["Wz"][19, 16 + 3] = 4096.0 * 1e-6 / 9.0
    p["Wz"][19, 32 + 5] = 1.0
    p["bz"][19] = -1.0
    p["Wz"][20, 32 + 5] = 1.0

    coordinate_small = 4096.0 * 63.0 * 1e-6 / 9.0
    for axis in range(2):
        p["Wz"][21 + axis, 64 + 15 + axis] = coordinate_small
        p["Wz"][21 + axis, 32 + 5] = -100.0

    # Recover selection in unmasked play.
    p["Wz"][23, 32 + 4] = 1.0
    p["Wz"][23, 32 + 2] = -1.0
    p["Wz"][23, 32 + 5] = -1.0
    p["Wo"][6, 23] = 1000.0

    # Avoid switching to nonexistent unconnected grey endpoints.
    p["Wz"][24, 32 + 5] = 1.0
    p["Wz"][24, 32 + 0] = -1.0
    p["Wo"][6, 24] = -3000.0

    # Global matched-terminal count, scaled to remain in tanh's
    # approximately linear region.
    p["Wz"][25, 64 + 12] = 4096.0 * 0.001 / 9.0

    # Analytic count-normalized centroid guidance for unmasked pieces.
    # Paired ReLUs gate coordinate sums by integer terminal count.
    # Telescoping coefficients supply reciprocal counts 1/q.
    next_unit = 30
    count_scale = 4096.0 / 9.0
    coordinate_scale = 63.0 * count_scale

    for count_feature, row_feature, col_feature, polarity in (
        (2, 15, 16, 1.0),
        (14, 17, 18, -1.0),
    ):
        for n in range(1, 11):
            coefficient = (
                1.0 / n - 1.0 / (n + 1) if n < 10 else 0.1
            )
            base = next_unit
            row_unit = next_unit + 1
            col_unit = next_unit + 2
            next_unit += 3

            for unit in (base, row_unit, col_unit):
                p["Wz"][unit, 64 + count_feature] = -1000.0 * count_scale
                p["Wz"][unit, 32 + 5] = -20000.0
                p["bz"][unit] = 1000.0 * (n + 0.5)

            p["Wz"][row_unit, 64 + row_feature] = coordinate_scale
            p["Wz"][col_unit, 64 + col_feature] = coordinate_scale

            for action, unit, sign in (
                (1, row_unit, 1.0),
                (2, row_unit, -1.0),
                (3, col_unit, 1.0),
                (4, col_unit, -1.0),
            ):
                weight = polarity * sign * coefficient
                p["Wo"][action, unit] += weight
                p["Wo"][action, base] -= weight

    # Learn, through episode memory only, which revealed colour group
    # belongs to a piece that actually grows.
    for k in range(5):
        activate, expanded = 90 + k, 95 + k

        p["Wz"][activate, 32 + 5] = 1.0
        p["Wz"][activate, 32 + 6] = 1.0
        p["Wz"][activate, 32 + 7 + k] = 1.0
        p["bz"][activate] = -2.0

        p["Wz"][expanded, 32 + 5] = 1.0
        p["Wz"][expanded, 32 + 7 + k] = 1.0
        p["Wz"][expanded, 32 + 6] = -1.0
        p["bz"][expanded] = -1.0

        latch = 2 + k
        p["Wm"][latch, latch] = 3.0
        p["bm"][latch] = -1.0
        p["Wmz"][latch, activate] = 10.0

        conjunction = 10 + 2 * k
        baseline = 11 + 2 * k
        for unit in (conjunction, baseline):
            p["Wm"][unit, latch] = 5.0
            p["bm"][unit] = -10.0
        p["Wmz"][conjunction, expanded] = 5.0

        p["Wo"][6, 128 + conjunction] = 1500.0
        p["Wo"][6, 128 + baseline] = -1500.0

    # Orientation exploration counter.
    p["Wm"][0, 0] = 0.95
    p["Wma"][0, 5] = 0.35
    p["Wma"][0, 1:5] = -0.10
    p["Wma"][0, 6] = -1.0
    p["bm"][0] = -0.01
    p["Wo"][6, 128] = 5.0

    # Detect a masked move with no improvement in the observed potential.
    p["Wmz"][8, 19] = 1.0
    p["Wm"][9, 8] = 2000000.0
    p["Wmz"][9, 19] = -2000000.0
    p["Wmz"][9, 20] = 10000.0
    p["bm"][9] = -9999.0
    for action in (0, 5, 6, 7):
        p["Wma"][9, action] = -20000.0

    p["Wo"][5, 128 + 9] = 50.0
    p["Wo"][6, 128 + 9] = 50.0
    p["bo"][5] = 50.1
    p["bo"][6] = 47.0

    # Unmasked blocked-move detection using selected coordinate sums.
    for axis in range(2):
        p["Wmz"][20 + axis, 21 + axis] = 1.0

    for k, action in enumerate((1, 2, 3, 4)):
        axis = 0 if action in (1, 2) else 1
        sign = 1.0 if action in (1, 3) else -1.0
        first, baseline = 22 + 2 * k, 23 + 2 * k

        for unit in (first, baseline):
            p["bm"][unit] = -2.0
            p["Wm"][unit, 20 + axis] = -sign * 2000000.0
            p["Wmz"][unit, 21 + axis] = sign * 2000000.0
            p["Wmz"][unit, 20] = -100.0

        p["Wma"][first, action] = 6.0
        p["Wo"][action, 128 + first] = -100.0
        p["Wo"][action, 128 + baseline] = 100.0
        p["Wo"][5, 128 + first] = 100.0
        p["Wo"][5, 128 + baseline] = -100.0

    # Soft immediate-reversal suppression.
    for k, opposite in enumerate((2, 1, 4, 3)):
        p["Wma"][30 + k, opposite] = 10.0
        p["Wo"][1 + k, 128 + 30 + k] = -10.0

    p["bo"][1:5] = -1.5

    # A newly formed contact promotes switching, rather than repeatedly
    # moving a piece that has just acquired an anchor.
    p["Wmz"][34, 25] = 1.0
    p["Wmz"][35, 25] = 20000.0
    p["Wm"][35, 34] = -20000.0
    p["bm"][35] = -5.0
    p["Wo"][6, 128 + 35] = 250.0
    p["bo"][6] += 250.0

    # Do not immediately click back after changing selection.
    p["Wma"][36, 6] = 10.0
    p["Wo"][6, 128 + 36] = -2000.0

    # Click unconnected grey endpoints, preferring those far from the
    # currently visible terminals. Unmasked clicks use coloured endpoints.
    p["Wc"][0, 0] = 1000.0
    p["Wc"][0, 4] = 500.0
    p["Wc"][0, 21] = 0.05
    p["Wc"][0, 32 + 10] = -1.0
    p["Wc"][0, 19] = -0.001
    p["Wc"][0, 20] = -0.0001

    p["bo"][0] = -100000.0
    p["bo"][7] = -100000.0
    return p
