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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}
    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    colors = [10, 6, 15, 11, 12, 8, 9, 7, 14]

    def first(o, c, dx=0, dy=0, v=1.0):
        W1[o, c, dy + 2, dx + 2] += v

    def second(o, c, dx=0, dy=0, v=1.0):
        W2[o, c, dy + 1, dx + 1] += v

    def recurrent(o, c, dx=0, dy=0, v=1.0):
        Wh[o, c, dy + 1, dx + 1] += v

    def rotate(point, n):
        x, y = point
        for _ in range(n):
            x, y = -y, x
        return x, y

    # Fruit anchors.
    for rank, color in enumerate(colors):
        required = [(0, 0)] if rank == 0 else [
            (0, 0), (1, 0), (0, 1), (1, 1)
        ]
        for dx, dy in required:
            first(rank, color, dx, dy)
        first(rank, color, -1, 0, -1)
        first(rank, color, 0, -1, -1)
        b1[rank] = 1 - len(required)

        if rank == 6:
            first(rank, color, -1, 1, -1)
            first(rank, color, 1, -1, -1)

    # Packed header-tier detector.
    for rank, color in enumerate(colors):
        for dx, dy in [(0, 0), (1, 0), (0, -1), (1, -1)]:
            first(9, color, dx, dy, 32 + rank)
    b1[9] = -127
    first(10, 10)

    # Crown centers.
    for out, color in ((11, 7), (12, 14)):
        for dx, dy in [(0, -2), (-1, -1), (1, -1)]:
            first(out, color, dx, dy)
        for dx, dy in [(0, -1), (-1, -2), (1, -2)]:
            first(out, color, dx, dy, -1)
        b1[out] = -2

    for dx, dy in [(-2, -2), (0, -2), (2, -2)]:
        first(13, 13, dx, dy)
    for dx, dy in [(-1, -2), (1, -2), (0, -1)]:
        first(13, 13, dx, dy, -1)
    b1[13] = -2

    # Rounded-goal signatures, moved inward.
    positive = [(0, 0), (1, -1), (-1, 1), (0, 1), (1, 0)]
    negative = [(0, -1), (-1, 0)]
    directions = []
    for rotation in range(4):
        out = 14 + rotation
        ux, uy = rotate((1, 1), rotation)
        directions.append((ux, uy))
        for point in positive:
            dx, dy = rotate(point, rotation)
            first(out, 9, dx - ux, dy - uy)
        for point in negative:
            dx, dy = rotate(point, rotation)
            first(out, 9, dx - ux, dy - uy, -1)
        b1[out] = 1 - len(positive)

    # Clipped nearby-goal presence.
    for out in (18, 19):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                first(out, 9, dx, dy)
    b1[19] = -1

    first(20, 16, v=-63)
    b1[20] = 10
    first(21, 0)
    first(22, 17, v=-63)
    b1[22] = 26
    first(23, 16, v=-63)
    b1[23] = 5
    first(24, 16, v=63)
    b1[24] = -5
    first(25, 2)

    # Crown proximity.
    for out in (26, 27):
        for color in (7, 14, 13):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    first(out, color, dx, dy)
    b1[27] = -1

    # Smaller-fruit proximity, for guarding occupied goals.
    for out in (28, 29):
        for color in (10, 6, 15, 11, 12):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    first(out, color, dx, dy)
    b1[29] = -1
    first(30, 17)

    # Factor a uniform seven-tap box into a positive five-tap
    # filter and a three-tap filter. Thus the first ReLU does not
    # disturb the factorization.
    factors = [
        np.array([1.0, -2.0 * np.cos(2.0 * np.pi * k / 7.0), 1.0])
        for k in (1, 2, 3)
    ]
    five = np.convolve(factors[1], factors[2])
    three = factors[0]
    W1[31, 10] = np.outer(five, five)

    # Priorities and shifted copies for vertical scans.
    for out, dy in ((0, 0), (1, -1), (2, 1)):
        for rank in range(9):
            second(out, rank, dy=dy, v=4096 - 128 * rank)
        for channel, priority in ((11, 2688), (12, 2816), (13, 2944)):
            second(out, channel, dy=dy, v=priority)
        second(out, 18, dy=dy, v=-2048)
        second(out, 19, dy=dy, v=2048)
        second(out, 6, dy=dy, v=2048)
        second(out, 20, dy=dy, v=-100000)

    # Header rank+1, restricted to the target-display region.
    second(3, 9, v=0.25)
    second(3, 10, v=0.25)
    second(3, 22, v=-100)
    second(3, 23, v=-100)
    second(3, 24, v=-100)
    b2[3] = 0.75

    for k, (dx, dy) in enumerate(directions):
        second(4, 14 + k, dx=-dx, dy=-dy)
    second(4, 20, v=-100)

    second(5, 25)
    second(5, 22, v=-100)
    second(5, 24, v=-100)

    second(6, 24, v=1 / 63)
    b2[6] = 5 / 63

    # Remaining-budget bar.
    second(8, 21)
    second(8, 24)
    b2[8] = -58

    for out in (9, 10):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                second(out, 26, dx, dy)
                second(out, 27, dx, dy, -1)
        second(out, 20, v=-1000)
    b2[10] = -1

    for out in (11, 12):
        second(out, 28)
        second(out, 29, v=-1)
        second(out, 20, v=-1000)
    b2[12] = -1

    second(13, 24)
    b2[13] = -57
    second(14, 20)
    second(15, 30)

    # N is the number of tier-zero fruits in a 7x7 neighborhood.
    # These hinges later produce one exactly when N=2.
    for out, threshold in ((16, 1), (17, 2), (18, 3)):
        W2[out, 31] = np.outer(three, three)
        b2[out] = -threshold
        second(out, 20, v=-1000)

    # A one-pixel offset from a crown center encourages a useful
    # fling rather than a zero-direction click on its exact center.
    for channel in (11, 12, 13):
        second(22, channel, dx=1)
    second(22, 20, v=-1000)

    for out in (23, 24):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                second(out, 18, dx, dy)
                second(out, 19, dx, dy, -1)
        second(out, 20, v=-1000)
    b2[24] = -1

    # Axial features.
    W3[0, 0] = 1
    W3[1, 32 + 0] = 1
    W3[2, 32 + 1] = 1
    W3[3, 32 + 2] = 1
    W3[4, 32 + 3] = 1
    W3[5, 4] = 1
    W3[6, 5] = 1
    W3[7, 64 + 8] = 1

    W3[8, 9] = 1
    W3[8, 10] = -1
    W3[8, 11] = 1
    W3[8, 12] = -1
    b3[8] = -1

    W3[9, 13] = 1
    W3[10, 14] = 1
    W3[11, 6] = 1
    W3[12, 15] = 1

    W3[13, 16] = 1
    W3[13, 17] = -2
    W3[13, 18] = 1

    # Danger involving a smaller fruit on or beside a goal.
    for c, v in ((9, 1), (10, -1), (11, 1), (12, -1),
                 (23, 1), (24, -1)):
        W3[14, c] = v
    b3[14] = -2

    W3[15, 22] = 1
    W3[15, 11] = 1
    W3[15, 12] = -1
    W3[15, 23] = 1
    W3[15, 24] = -1
    b3[15] = -2

    # Recurrent global-priority scans.
    recurrent(0, 0, dy=-1)
    Uh[0, 2] = 1
    Uh[0, 1] = -1

    recurrent(1, 1, dy=1)
    Uh[1, 3] = 1
    Uh[1, 1] = -1

    recurrent(11, 0)
    recurrent(11, 1, v=-1)

    # Header broadcast and clock.
    recurrent(2, 2, dy=-1)
    Uh[2, 4] = 1
    recurrent(3, 3)
    bh[3] = 1

    delay = 66
    Uh[4, 0] = 2
    Uh[4, 1] = -2
    recurrent(4, 0, v=-1)
    recurrent(4, 1, v=-1)
    recurrent(4, 3)
    bh[4] = -delay

    # Clipped expanding wave and its integral.
    cross = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    for out in (5, 6):
        for dx, dy in cross:
            recurrent(out, 5, dx, dy)
            recurrent(out, 6, dx, dy, -1)
        recurrent(out, 4)
    bh[6] = -1

    recurrent(7, 7)
    recurrent(7, 5)
    recurrent(7, 6, v=-1)

    recurrent(8, 7, v=-1)
    bh[8] = 128 - delay - 4 - 8

    # Crown/deferred-object class gate.
    recurrent(14, 1, v=-1 / 128)
    recurrent(14, 11, v=-1 / 128)
    Uh[14, 1] = -1 / 128
    bh[14] = 24

    # Merge suppression at or above the requested fruit tier.
    recurrent(12, 1, v=-1 / 128)
    recurrent(12, 11, v=-1 / 128)
    Uh[12, 1] = -1 / 128
    recurrent(12, 2, v=-1)
    recurrent(12, 14, v=-16)
    bh[12] = 34

    # Delayed occupancy rejects moving-wavefront artifacts.
    recurrent(13, 5)
    recurrent(13, 6, v=-1)

    # Diagonal curvature avoids the extra cardinal-curvature peak
    # at common three-source junctions.
    distance_cost = 0.05
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        recurrent(9, 7, dx, dy)
    recurrent(9, 7, v=-4 + distance_cost)
    recurrent(9, 3, v=-distance_cost)
    recurrent(9, 13, v=8)
    recurrent(9, 12, v=-32)
    Uh[9, 7] = -0.01
    Uh[9, 13] = 6
    bh[9] = distance_cost * (delay + 3) - 8

    # Shared attraction field.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            recurrent(10, 10, dx, dy, 1.01 / 9)
    recurrent(10, 9)
    Uh[10, 5] = 0.001

    # Oversized-fruit waiting. Equal output coefficients for this
    # channel and the radius violation make waiting approximately
    # flat outside the suction neighborhood, rather than preferring
    # an arbitrary maximally distant corner.
    recurrent(15, 1, v=-100 / 128)
    recurrent(15, 11, v=-100 / 128)
    Uh[15, 1] = -100 / 128
    recurrent(15, 2, v=-100)
    recurrent(15, 14, v=-1600)
    recurrent(15, 7, v=-1)
    Uh[15, 14] = 60
    Uh[15, 15] = 0.2
    bh[15] = 3300

    p["Wc"][0, 32 + 8] = -1000
    p["Wc"][0, 32 + 10] = 100
    p["Wc"][0, 32 + 15] = 1000
    p["Wc"][0, 8] = 0.05
    p["Wc"][0, 9] = -1e12
    p["Wc"][0, 10] = -1e12

    # Click normally; undo when the target display signals
    # insufficient remaining mass.
    p["Wz"][0, 32 + 6] = 1
    p["bo"][6] = 1
    p["Wo"][7, 0] = 4
    p["Wo"][0, 0] = 3

    return p
