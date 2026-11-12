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
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def w1(o, i, v, dy=0, dx=0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def w2(o, i, v, dy=0, dx=0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def wh(o, i, v, dy=0, dx=0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    def square(o, colors, weight):
        for color in colors:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    w1(o, color, weight, dy, dx)

    # Cell-centred cap detectors. For an integer neighbour count n,
    # relu(2-n) - 2*relu(1-n) identifies exactly one neighbour.
    for a, colors, body in ((0, (8, 3), 0), (2, (4,), 4)):
        for o, bias in ((a, -16), (a + 1, -17)):
            square(o, colors, 2)
            p["b1"][o] = bias
            for dy, dx in directions:
                w1(o, body, -1, 2 * dy, 2 * dx)

    for o, color in ((4, 8), (5, 3), (6, 4), (12, 0)):
        square(o, (color,), 1)
        p["b1"][o] = -8

    # Isolated black extension marker.
    square(7, (0,), 1)
    p["b1"][7] = -8
    for dy, dx in directions:
        w1(7, 0, -1, 2 * dy, 2 * dx)

    w1(8, 16, 1)
    w1(9, 17, 1)
    w1(10, 16, 0.5)
    w1(10, 17, 0.5)

    square(11, (8, 3), 1)
    p["b1"][11] = -9
    w1(11, 16, 0.4)
    w1(11, 17, 0.4 * math.sqrt(2.0))

    # Terminal masks, including one-pixel-displaced copies.
    offsets = [(0, 0)] + [(-dy, -dx) for dy, dx in directions]
    for j, (dy, dx) in enumerate(offsets):
        w2(j, 0, 1, dy, dx)
        w2(j, 1, -2, dy, dx)
        w2(5 + j, 4, 1, dy, dx)
        w2(5 + j, 5, 1, dy, dx)

    w2(10, 2, 1)
    w2(10, 3, -2)
    w2(11, 4, 1)
    w2(12, 5, 1)
    w2(13, 11, 1)
    w2(14, 0, 1)
    w2(14, 1, -2)
    w2(14, 4, 1)
    p["b2"][14] = -1
    w2(15, 7, 1)
    w2(16, 6, 1)
    for out, inp in ((17, 8), (18, 9), (19, 10), (20, 12)):
        w2(out, inp, 1)

    F = p["W3"]
    F[0, 0] = 1
    F[1, 5] = 1
    F[1, 0] = -1

    for j in range(4):
        F[2 + j, 1 + j] = 1
        F[6 + j, 6 + j] = 1
        F[6 + j, 1 + j] = -1

    F[10, 10] = 1
    F[11, 32 + 16] = 1
    F[11, 64 + 16] = 1

    for out, inp in (
        (12, 11), (13, 12), (14, 14), (15, 13),
        (16, 15), (17, 16),
    ):
        F[out, inp] = 1

    F[18, 11] = 1
    F[18, 14] = -1

    # Coordinate moments for the two terminal groups.
    for j, coord in enumerate((17, 18, 19)):
        F[19 + j, 0] = 1
        F[19 + j, coord] = 1
        p["b3"][19 + j] = -1

        F[22 + j, 5] = 1
        F[22 + j, 0] = -1
        F[22 + j, coord] = 1
        p["b3"][22 + j] = -1

    # Selected black-body extrema.
    for out, coord in ((25, 17), (26, 18)):
        F[out, 20] = 1
        F[out, coord] = 1
        p["b3"][out] = -1
    for out, coord in ((27, 17), (28, 18)):
        F[out, 20] = 1
        F[out, coord] = -1
    F[29, 20] = 1

    # H[0]-H[1]: saturating broadcast of grey-mask presence.
    for out in (0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                wh(out, 0, 1, dy, dx)
                wh(out, 1, -1, dy, dx)
        p["Uh"][out, 11] = 1
    p["bh"][1] = -1

    # Attraction sources: unselected terminals and attachment anchors.
    p["Uh"][2, 1] = 1
    p["Uh"][2, 10] = 2
    p["Uh"][2, 13] = 4
    wh(2, 0, -1)
    wh(2, 1, 1)

    # Separable exponential field.
    decay, k, gamma = 0.90, 0.03, 0.03
    wh(3, 3, decay, 0, -1)
    wh(3, 2, k)
    wh(4, 4, decay, 0, 1)
    wh(4, 2, k)
    for out, dy in ((5, -1), (6, 1)):
        wh(out, out, decay, dy, 0)
        wh(out, 3, gamma)
        wh(out, 4, gamma)
        wh(out, 2, -gamma * k)

    def field(out, dy=0, dx=0):
        wh(out, 5, 1, dy, dx)
        wh(out, 6, 1, dy, dx)
        wh(out, 3, -gamma, dy, dx)
        wh(out, 4, -gamma, dy, dx)
        wh(out, 2, gamma * k, dy, dx)

    # Two recurrent shifts and a displaced terminal mask evaluate
    # translations of three display pixels: one canonical game cell.
    for j, (dy, dx) in enumerate(directions):
        field(7 + j, dy, dx)
        wh(11 + j, 7 + j, 1, dy, dx)
        p["Uh"][11 + j, 2 + j] = 2
        p["Uh"][11 + j, 6 + j] = 1
        wh(11 + j, 0, 1)
        wh(11 + j, 1, -1)
        p["bh"][11 + j] = -2

    field(15)
    p["Uh"][15, 0] = 2
    p["Uh"][15, 1] = 1
    wh(15, 0, 1)
    wh(15, 1, -1)
    p["bh"][15] = -2

    def fm(c):
        return 32 + c

    def fa(c):
        return 64 + c

    Z = p["Wz"]

    # Translation improvements.
    for j in range(4):
        Z[j, 16 + 11 + j] = 409600
        Z[j, 16 + 15] = -409600

    Z[4, fm(16)] = 1
    Z[4, fm(17)] = 1
    p["bz"][4] = -1

    Z[5, fa(14)] = 1
    Z[5, fm(17)] = -1
    Z[6, fa(12)] = 1
    Z[6, fm(17)] = 1
    p["bz"][6] = -1

    Z[7, fm(18)] = 1
    Z[7, fm(17)] = -1
    Z[8, fm(10)] = 1
    Z[10, fa(15)] = 4.096

    for out, threshold in ((11, 3.5), (12, 4.5)):
        Z[out, fa(12)] = 4096
        Z[out, fa(13)] = 4096
        Z[out, fm(17)] = 100
        p["bz"][out] = -100 - threshold

    # Deviations from two terminals in each visible group.
    for out, channel, sign in (
        (20, 0, 1), (21, 0, -1),
        (22, 1, 1), (23, 1, -1),
    ):
        Z[out, fa(channel)] = sign * 4096
        p["bz"][out] = -sign * 2
    Z[24, fm(17)] = 1

    # Row, column and diagonal span mismatches.
    for j in range(3):
        for out, sign in ((30 + 2 * j, 1), (31 + 2 * j, -1)):
            Z[out, fm(19 + j)] = 2 * sign
            Z[out, fa(19 + j)] = -4096 * sign
            Z[out, fm(22 + j)] = -2 * sign
            Z[out, fa(22 + j)] = 4096 * sign

    # Signed centroid displacement, split into positive parts.
    for out, a, b, sign in (
        (36, 20, 23, 1), (37, 20, 23, -1),
        (38, 19, 22, 1), (39, 19, 22, -1),
    ):
        Z[out, fa(b)] = 2048 * sign
        Z[out, fa(a)] = -2048 * sign

    # Terminal-centroid displacement from the selected body's centre.
    for out, term, body, complement, sign in (
        (40, 19, 25, 27, 1),
        (41, 19, 25, 27, -1),
        (42, 20, 26, 28, 1),
        (43, 20, 26, 28, -1),
    ):
        Z[out, fa(term)] = 2048 * sign
        Z[out, fm(body)] = -0.5 * sign
        Z[out, fm(complement)] = 0.5 * sign
        p["bz"][out] = -0.5 * sign - 0.005

    direction_specs = [
        (19, 22, -1), (19, 22, 1),
        (20, 23, -1), (20, 23, 1),
    ]
    for j, (a, b, sign) in enumerate(direction_specs):
        for d in range(2):
            out = 44 + 2 * j + d
            Z[out, fa(b)] = 2048 * sign / 0.03
            Z[out, fa(a)] = -2048 * sign / 0.03
            p["bz"][out] = -d

    Z[60, fa(13)] = 4096

    # Memory 0..9: rotation counter, reset by piece selection.
    sharp = 8.0
    for i in range(5):
        hold, enter = 2 * i, 2 * i + 1
        predecessor = (i - 1) % 5
        p["Wm"][hold, 2 * i] = sharp
        p["Wm"][hold, 2 * i + 1] = sharp
        p["bm"][hold] = sharp if i == 0 else -sharp
        if i != 0:
            p["Wma"][hold, :] = 2 * sharp
        p["Wma"][hold, 5] = -2 * sharp if i == 0 else 0
        p["Wma"][hold, 6] = 2 * sharp if i == 0 else 0
        p["Wma"][hold, 0] = 2 * sharp if i == 0 else 0

        p["Wm"][enter, 2 * predecessor] = sharp
        p["Wm"][enter, 2 * predecessor + 1] = sharp
        p["bm"][enter] = -sharp
        p["Wma"][enter, 5] = 2 * sharp

    # Short observation history for blocked-move and cycle detection.
    p["Wmz"][10, 10] = 1
    for i in range(11, 14):
        p["Wm"][i, i - 1] = 1

    # Geometric-controller latch. Selection and rotation permit
    # re-evaluation after an initially deselected or obscured state.
    mode = 20
    p["Wm"][mode, mode] = 100
    p["bm"][mode] = 5
    for z in range(20, 24):
        p["Wmz"][mode, z] = -20
    p["Wmz"][mode, 24] = -200
    p["Wma"][mode, 6] = 100
    p["Wma"][mode, 5] = 100

    p["bm"][21] = 10
    p["Wm"][22, 21] = 10

    equality_pairs = []
    for j, previous in enumerate((10, 11, 13)):
        hi, lo = 14 + 2 * j, 15 + 2 * j
        equality_pairs.append((hi, lo))
        for out, sign in ((hi, 1), (lo, -1)):
            p["Wmz"][out, 10] = 2_000_000
            p["Wm"][out, previous] = -2_000_000
            p["bm"][out] = sign * 6 + 100_000
            p["Wm"][out, mode] = 100_000
            for action in (0, 5, 6, 7):
                p["Wma"][out, action] = 2_000_000

    # Approximately linear, gated translation-plan registers.
    # While translating, update by the previous action rather than
    # re-estimating terminal ownership from an overlapping image.
    translation_step = 0.01 * 3.0 / 63.0
    for base, positive, negative, plus_action, minus_action in (
        (23, 36, 37, 3, 4),
        (27, 38, 39, 1, 2),
    ):
        for d, sign in enumerate((1, -1)):
            live = base + d
            carry = base + 2 + d

            p["bm"][live] = -10
            p["Wm"][live, mode] = 10
            p["Wmz"][live, positive] = sign * 0.01
            p["Wmz"][live, negative] = -sign * 0.01

            p["bm"][carry] = -30
            p["Wm"][carry, mode] = 10
            for src, coef in (
                (base, 0.5), (base + 1, -0.5),
                (base + 2, 0.5), (base + 3, -0.5),
            ):
                p["Wm"][carry, src] = sign * coef

            for action in (1, 2, 3, 4):
                p["Wma"][live, action] = -20
                p["Wma"][carry, action] = 20
            p["Wma"][carry, plus_action] += sign * translation_step
            p["Wma"][carry, minus_action] -= sign * translation_step

    # Generic movement scores, gated off in geometric mode.
    for j in range(4):
        for d, sign in enumerate((1, -1)):
            out = 31 + 2 * j + d
            p["bm"][out] = -10
            p["Wm"][out, mode] = -10
            p["Wmz"][out, j] = sign * 0.01

    # Span and facing tests, evaluated between translation plans.
    for out, sign in ((43, 1), (44, -1)):
        p["bm"][out] = -10
        p["Wm"][out, mode] = 10
        for action in (1, 2, 3, 4):
            p["Wma"][out, action] = -20
        for z in range(30, 36):
            p["Wmz"][out, z] = sign * 0.02

    for j in range(4):
        for d, sign in enumerate((1, -1)):
            out = 45 + 2 * j + d
            p["bm"][out] = -30
            p["Wm"][out, mode] = 10
            p["Wmz"][out, 44 + 2 * j] = 20
            p["Wmz"][out, 45 + 2 * j] = -20
            p["Wmz"][out, 40 + j] = sign * 0.1
            for action in (1, 2, 3, 4):
                p["Wma"][out, action] = -20

    # Directional persistence is enabled only in generic mode and
    # only while the selected piece has no visible attachment.
    for j, action in enumerate((1, 2, 3, 4)):
        out = 54 + j
        p["bm"][out] = -32
        p["Wm"][out, mode] = -16
        p["Wma"][out, action] = 32
        p["Wmz"][out, 60] = -64

    p["bo"][:] = -10000
    p["bo"][5] = 1000
    p["Wo"][5, 128 + 22] = -1000
    p["Wo"][5, 4] = 1000

    for j, action in enumerate((1, 2, 3, 4)):
        p["bo"][action] = -0.07501 + 0.015
        p["Wo"][action, 128 + mode] = -0.075
        p["Wo"][action, 128 + 31 + 2 * j] = 50
        p["Wo"][action, 128 + 32 + 2 * j] = -50
        p["Wo"][action, 128 + 54 + j] = 0.015

    for action, base, sign in (
        (1, 27, -1), (2, 27, 1),
        (3, 23, -1), (4, 23, 1),
    ):
        for offset, coef in (
            (0, 500), (1, -500), (2, 500), (3, -500)
        ):
            p["Wo"][action, 128 + base + offset] = sign * coef

    for hi, lo in equality_pairs:
        p["Wo"][5, 128 + hi] = 100
        p["Wo"][5, 128 + lo] = -100

    p["Wo"][5, 128 + 43] = 5000
    p["Wo"][5, 128 + 44] = -5000
    for j in range(4):
        p["Wo"][5, 128 + 45 + 2 * j] = 5000
        p["Wo"][5, 128 + 46 + 2 * j] = -5000

    # Change pieces when attached, after exploring orientations,
    # or when a large branching piece is better left as an anchor.
    p["bo"][6] = 100.5 - 500 - 5000
    p["Wo"][6, 128 + mode] = -5000
    p["Wo"][6, 5] = -0.7 * 4096
    p["Wo"][6, 6] = -0.7 * 4096
    p["Wo"][6, 7] = 500
    p["Wo"][6, 8] = 500
    p["Wo"][6, 11] = 100
    p["Wo"][6, 12] = -100
    p["Wo"][6, 128 + 8] = 50
    p["Wo"][6, 128 + 9] = 50

    # Prefer exposed terminals on unselected, relatively isolated pieces.
    p["Wc"][0, 32 + 2] = 2
    p["Wc"][0, 13] = -20
    p["Wc"][0, 0] = -2
    p["Wc"][0, 32 + 5] = -0.005
    p["Wc"][0, 32 + 6] = -0.005

    return p
