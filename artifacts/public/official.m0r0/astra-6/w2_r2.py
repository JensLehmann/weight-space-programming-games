import math
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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]
    special_colours = (12, 14, 15)

    def c1(out, inp, weight=1.0, dy=0, dx=0):
        p["W1"][out, inp, dy + 2, dx + 2] += weight

    def c2(out, inp, weight=1.0, dy=0, dx=0):
        p["W2"][out, inp, dy + 1, dx + 1] += weight

    def rec(out, inp, weight=1.0, dy=0, dx=0):
        p["Wh"][out, inp, dy + 1, dx + 1] += weight

    def avatar(out, weight=1.0, dy=0, dx=0):
        c1(out, 1, weight, dy, dx)
        c1(out, 10, weight, dy, dx)

    # ------------------------------------------------------------
    # Local visual features.
    # ------------------------------------------------------------
    avatar(0)
    c1(1, 1)

    # Traversability, excluding the two UI rows.
    for colour in (1, 5, 9, 10):
        c1(2, colour)
    c1(2, 18, 1, -1, 0)
    c1(2, 18, 1, 1, 0)
    p["b1"][2] = -2
    c1(3, 9)

    # A binary left-half mask obtained by differencing rectifiers.
    c1(4, 17, -126)
    c1(5, 17, -126)
    p["b1"][4] = 63.5
    p["b1"][5] = 62.5

    avatar(6)
    c1(6, 17)
    p["b1"][6] = -1
    avatar(7)
    c1(7, 16)
    p["b1"][7] = -1

    for d, (dy, dx) in enumerate(directions):
        avatar(8 + d)
        avatar(8 + d, -1, dy, dx)

    # Long coloured strips versus isolated switch patches.
    for offset in (-2, 0, 2):
        for colour in special_colours:
            c1(12, colour, 1, 0, offset)
            c1(13, colour, 1, offset, 0)
    p["b1"][12:14] = -2
    for colour in special_colours:
        c1(14, colour)

    # Compact selected-obstacle interior, rejecting large walls.
    c1(16, 11, 2)
    c1(16, 11, -1, 0, -2)
    c1(16, 11, -1, 0, 2)
    c1(17, 11, 2)
    c1(17, 11, -1, -2, 0)
    c1(17, 11, -1, 2, 0)

    # Saturated dilation reconstructs bordered obstacle footprints.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c1(18, 9, 1, dy, dx)
            c1(19, 9, 1, dy, dx)
    p["b1"][19] = -1

    for d, (dy, dx) in enumerate(directions):
        c1(20 + d, 5, 1, 2 * dy, 2 * dx)
        avatar(24 + d, 1, 2 * dy, 2 * dx)

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            avatar(29, 1, dy, dx)
            avatar(30, 1, dy, dx)
    p["b1"][30] = -1

    # ------------------------------------------------------------
    # Second visual layer.
    # ------------------------------------------------------------
    for out, inp in ((0, 2), (1, 0), (2, 1), (3, 3)):
        c2(out, inp)

    c2(4, 4)
    c2(4, 5, -1)
    c2(5, 18)
    c2(5, 19, -1)

    for d in range(4):
        c2(6 + d, 8 + d)

    for offset in (-1, 0, 1):
        c2(10, 12, 1, 0, offset)
        c2(11, 13, 1, offset, 0)
    p["b2"][10:12] = -2

    c2(12, 14)
    c2(13, 6)
    c2(14, 7)

    for d, (dy, dx) in enumerate(directions):
        c2(15 + d, 16)
        c2(15 + d, 17)
        c2(15 + d, 20 + d, 1, dy, dx)
        p["b2"][15 + d] = -4

        od = opposite[d]
        oy, ox = directions[od]
        c2(19 + d, 16)
        c2(19 + d, 17)
        c2(19 + d, 24 + od, 1, oy, ox)
        p["b2"][19 + d] = -4

    c2(23, 18)
    c2(23, 19, -1)
    c2(23, 29)
    c2(23, 30, -1)
    p["b2"][23] = -1

    c2(24, 16)
    c2(24, 17)
    p["b2"][24] = -3

    for d, (dy, dx) in enumerate(directions):
        c2(25 + d, 8 + d)
        c2(25 + d, 2, 1, dy, dx)
        c2(25 + d, 14, 1, dy, dx)
        p["b2"][25 + d] = -1

    c2(29, 4, 1, 0, -1)
    c2(29, 5, -1, 0, -1)
    c2(29, 4, -1)
    c2(29, 5, 1)

    c2(30, 0)
    c2(30, 4)
    c2(30, 5, -1)
    p["b2"][30] = -1

    c2(31, 0)
    c2(31, 4, -1)
    c2(31, 5)

    # ------------------------------------------------------------
    # Axial layer.
    # ------------------------------------------------------------
    w = p["W3"]
    w[0, 0] = 1
    w[1, 30] = 1
    w[2, 31] = 1

    for d in range(4):
        w[3 + d, 25 + d] = 1
        w[3 + d, 4] = 1
        p["b3"][3 + d] = -1
        w[7 + d, 25 + d] = 1
        w[7 + d, 4] = -1

    w[11, 5] = 1

    w[12, 12] = 1
    w[12, 32 + 10] = -1
    w[12, 64 + 11] = -1

    w[13, 3] = 1
    w[14, 23] = 1
    w[15, 2] = 1

    for out, feature in ((16, 13), (17, 14)):
        w[out, feature] = 1
        w[out, 4] = 1
        p["b3"][out] = -1
    for out, feature in ((18, 13), (19, 14)):
        w[out, feature] = 1
        w[out, 4] = -1

    for d in range(4):
        axial = 64 if d < 2 else 32
        w[20 + d, 15 + d] = 3
        w[20 + d, axial + 19 + d] = -1
        w[20 + d, axial + 19 + opposite[d]] = -3
        w[24 + d, 15 + d] = 1

    # Gate-area measurement; static coloured walls cancel temporally.
    w[29, 12] = 1
    w[29, 32 + 10] = 1
    w[29, 64 + 11] = 1
    p["b3"][29] = -1

    w[30, 24] = 1
    w[31, 29] = 1
    w[31, 0] = 1
    p["b3"][31] = -1

    # ------------------------------------------------------------
    # Integer cardinal wavefronts.
    #
    # 0..2 and 3..5: two clipped waves and arrival accumulators.
    # 6..13: directional gradients at the two avatars.
    # 14: clock. 15: delayed passage/switch sources.
    # ------------------------------------------------------------
    rec(14, 14)
    p["bh"][14] = 1

    rec(15, 14)
    p["Uh"][15, 31] = 256
    p["Uh"][15, 12] = 240
    p["bh"][15] = -272

    neighbourhood = [(0, 0)] + directions
    for lo, hi, accumulated, source in (
        (0, 1, 2, 2), (3, 4, 5, 1)
    ):
        for out in (lo, hi):
            for dy, dx in neighbourhood:
                rec(out, lo, 1, dy, dx)
                rec(out, hi, -1, dy, dx)
            rec(out, 15)
            p["Uh"][out, 0] = 8
            p["Uh"][out, 12] = 8
            p["Uh"][out, source] += 1
        p["bh"][lo] = -8
        p["bh"][hi] = -9
        rec(accumulated, accumulated)
        rec(accumulated, lo)
        rec(accumulated, hi, -1)

    for group, field, first_edge in ((0, 2, 3), (1, 5, 7)):
        for d, (dy, dx) in enumerate(directions):
            out = 6 + 4 * group + d
            rec(out, field, 1, dy, dx)
            rec(out, field, -1)
            p["Uh"][out, first_edge + d] = 256
            p["bh"][out] = -255

    # ------------------------------------------------------------
    # Pooled summaries.
    # ------------------------------------------------------------
    for i in range(8):
        p["Wz"][i, 6 + i] = 1
        p["Wz"][8 + i, 32 + 3 + i] = 1

    for i in range(4):
        p["Wz"][16 + i, 32 + 16 + i] = 1
        p["Wz"][22 + i, 32 + 20 + i] = 1
        p["Wz"][26 + i, 32 + 24 + i] = 1

    p["Wz"][20, 32 + 14] = 1
    p["Wz"][21, 32 + 15] = 1
    p["Wz"][30, 32 + 13] = 1
    p["Wz"][31, 32 + 12] = 1
    p["Wz"][32, 64 + 29] = 1

    # Exactly one avatar has a free vertical edge:
    # ReLU(s) - 2*ReLU(s-1), for s in {0,1,2}.
    for d, base in ((0, 34), (1, 36)):
        for zunit in (base, base + 1):
            p["Wz"][zunit, 32 + 3 + d] = 1
            p["Wz"][zunit, 32 + 7 + d] = 1
        p["bz"][base + 1] = -1

    # Additional right-avatar gradient priority only on switch boards.
    for d in range(4):
        for zunit, sign in ((40 + d, 1), (44 + d, -1)):
            p["Wz"][zunit, 10 + d] = sign
            p["Wz"][zunit, 32 + 7 + d] = -sign
            p["Wz"][zunit, 32 + 12] = 8
            p["bz"][zunit] = -8

    def utility_memory(unit, indices, factor):
        for i in indices:
            p["Wmz"][unit, i] += 3 * factor
            p["Wmz"][unit, 8 + i] -= 2.8 * factor
            if i >= 4:
                p["Wmz"][unit, 40 + i - 4] += 3 * factor
                p["Wmz"][unit, 44 + i - 4] -= 3 * factor

    def utility_output(action, indices):
        for i in indices:
            p["Wo"][action, i] += 3
            p["Wo"][action, 8 + i] -= 2.8
            if i >= 4:
                p["Wo"][action, 40 + i - 4] += 3
                p["Wo"][action, 44 + i - 4] -= 3

    up, down = (0, 4), (1, 5)
    left, right = (2, 7), (3, 6)

    # ------------------------------------------------------------
    # Temporal motion and polarity feedback.
    # ------------------------------------------------------------
    coordinate_scale = 0.025
    for i in range(4):
        p["Wmz"][i, 16 + i] = coordinate_scale

    for d in range(4):
        coordinates = (1, 3) if d < 2 else (0, 2)
        for k, coordinate in enumerate(coordinates):
            for side, sign in enumerate((1, -1)):
                unit = 4 + 4 * d + 2 * k + side
                p["Wmz"][unit, 16 + coordinate] = sign * 120
                p["Wm"][unit, coordinate] = (
                    -sign * 120 / coordinate_scale
                )
                p["bm"][unit] = 18.5
                p["Wma"][unit, d + 1] = -20
        p["Wma"][20 + d, d + 1] = 8

    for unit, action, sign, gate_sign in (
        (24, 4, 1, 1), (25, 4, 1, -1),
        (26, 3, -1, 1), (27, 3, -1, -1),
    ):
        p["Wmz"][unit, 16] = sign * 20
        p["Wmz"][unit, 18] = -sign * 20
        p["Wm"][unit, 0] = -sign * 20 / coordinate_scale
        p["Wm"][unit, 2] = sign * 20 / coordinate_scale
        p["bm"][unit] = gate_sign * 64
        p["Wma"][unit, action] = -gate_sign * 64

    p["Wm"][28, 28] = 4
    p["Wm"][28, 24:28] = 6
    p["bm"][28] = 0.5

    # Obstacle selection requires adjacency and stalled motion.
    p["Wmz"][29, 20] = 100
    p["Wmz"][29, 21] = -200
    p["Wm"][29, 4:20] = -5
    p["bm"][29] = -55

    p["Wmz"][30, 21] = 8
    p["bm"][30] = -4
    for unit, previous in ((31, 30), (32, 31), (33, 32)):
        p["Wmz"][unit, 21] = 8
        p["Wm"][unit, previous] = 8
        p["bm"][unit] = -12

    # Crucially, the dead-end stop cannot fire after deselection.
    p["Wmz"][35, 21] = 20
    p["Wmz"][35, 26:30] = -5
    p["Wm"][35, 30] = 5
    p["bm"][35] = -18

    for d in range(4):
        p["Wmz"][36 + d, 22 + d] = 0.2
        p["Wmz"][36 + d, 21] = 8
        p["bm"][36 + d] = -8
        p["Wmz"][40 + d, 21] = 5
        p["Wma"][40 + d, d + 1] = 5
        p["bm"][40 + d] = -7.5

    epsilon = 0.04
    for unit, indices, sign in (
        (44, right, 1), (45, left, -1),
        (46, left, 1), (47, right, -1),
    ):
        utility_memory(unit, indices, epsilon)
        p["Wm"][unit, 28] = 5 * sign
        p["bm"][unit] -= 5

    for base, angle, gain, injection in (
        (48, 0.27, 1.04, 0.02),
        (50, 0.11, 1.025, 0.008),
    ):
        co, si = gain * math.cos(angle), gain * math.sin(angle)
        p["Wm"][base, base] = co
        p["Wm"][base, base + 1] = -si
        p["Wm"][base + 1, base] = si
        p["Wm"][base + 1, base + 1] = co
        p["bm"][base] = injection

    # ------------------------------------------------------------
    # Gate-opening feedback and reversal fatigue.
    # ------------------------------------------------------------
    p["bm"][52] = 8
    p["Wmz"][53, 32] = 0.01

    p["Wm"][54, 54] = 4
    p["Wm"][54, 53] = 80000
    p["Wmz"][54, 32] = -800

    for unit, zbase in ((55, 34), (56, 36)):
        p["Wm"][unit, 54] = 4
        p["Wm"][unit, 53] = 80000
        p["Wmz"][unit, 32] = -800
        p["Wmz"][unit, zbase] = 80
        p["Wmz"][unit, zbase + 1] = -160
        p["Wmz"][unit, 31] = 100
        p["Wmz"][unit, 21] = -200
        p["bm"][unit] = -182

    # Detect actual reversals rather than penalizing long straight runs.
    p["Wm"][60, 23] = 6
    p["Wma"][60, 3] = 6
    p["bm"][60] = -9
    p["Wm"][61, 22] = 6
    p["Wma"][61, 4] = 6
    p["bm"][61] = -9

    p["Wm"][57, 57] = 0.92
    p["Wm"][57, 60:62] = 0.35
    p["Wm"][57, 52] = 0.7
    p["bm"][57] = -0.02

    # Complementary gates cancel this signal on boards without switches.
    for unit, sign in ((58, 1), (59, -1)):
        p["Wm"][unit, 57] = 0.4
        p["Wmz"][unit, 31] = sign * 8
        p["bm"][unit] = -sign * 8

    # ------------------------------------------------------------
    # Action decoder.
    # ------------------------------------------------------------
    p["bo"][:] = -1000
    p["bo"][1:5] = 0
    utility_output(1, up)
    utility_output(2, down)

    for action, units in ((4, (44, 45)), (3, (46, 47))):
        for unit in units:
            p["Wo"][action, 128 + unit] = 1 / epsilon
        p["bo"][action] += 1 / epsilon
        p["Wo"][action, 128 + 58] -= 10
        p["Wo"][action, 128 + 59] -= 10

    for d in range(4):
        action = d + 1
        start = 128 + 4 + 4 * d
        p["Wo"][action, start:start + 4] += 0.4
        p["Wo"][action, 128 + 20 + d] += 0.4 * 5.8
        p["bo"][action] -= 1.6
        p["Wo"][action, 128 + 20 + d] += 0.12
        p["Wo"][action, 128 + 20 + opposite[d]] -= 0.08
        p["Wo"][action, 128 + 36 + d] += 120
        p["bo"][action] += 120

    for action, unit in ((1, 55), (2, 56)):
        p["Wo"][action, 128 + unit] += 4
        p["bo"][action] += 4

    clockwise = {0: 3, 3: 1, 1: 2, 2: 0}
    counterclockwise = {v: k for k, v in clockwise.items()}
    for previous in range(4):
        preferences = {
            previous: 2.0,
            opposite[previous]: -8.0,
            clockwise[previous]: 3.5,
            counterclockwise[previous]: 3.0,
        }
        for direction, value in preferences.items():
            p["Wo"][direction + 1, 128 + 40 + previous] += value / 2

    p["bo"][1] += 0.04
    for action, unit, weight in (
        (1, 48, 0.20), (2, 48, -0.20),
        (3, 49, 0.20), (4, 49, -0.20),
        (1, 50, 0.08), (2, 50, -0.08),
        (3, 51, -0.08), (4, 51, 0.08),
    ):
        p["Wo"][action, 128 + unit] += weight

    p["Wo"][6, 128 + 29] = 20
    p["Wo"][6, 128 + 33] = 180
    p["Wo"][6, 128 + 35] = 180
    p["bo"][6] = 360

    # Nearby obstacles for selection; inactive avatars for deselection.
    p["Wc"][0, 11] = 1000
    p["Wc"][0, 14] = 500
    p["Wc"][0, 15] = 2500
    p["Wc"][0, 32 + 2] = 0.1
    p["Wc"][0, 32 + 5] = 0.1

    return p
