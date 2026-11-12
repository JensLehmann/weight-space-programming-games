import math
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
    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    def pix(o, c, y, x, w=1.0):
        W1[o, c, y + 2, x + 2] += w

    def s2(o, c, w=1.0, y=0, x=0):
        W2[o, c, y + 1, x + 1] += w

    def s3(o, c, w=1.0):
        W3[o, c] += w

    circle = (
        [(-1, x) for x in (0, 1)]
        + [(y, x) for y in (0, 1) for x in (-1, 0, 1, 2)]
        + [(2, x) for x in (0, 1)]
    )
    corners = [(-1, -1), (-1, 2), (2, -1), (2, 2)]
    for out, color in ((0, 14), (1, 8), (2, 9), (20, 2)):
        for y, x in circle:
            pix(out, color, y, x)
        for y, x in corners:
            pix(out, color, y, x, -3.0)
        b1[out] = -11.0

    ring = [
        (-1, 0), (-1, 1), (0, -1), (0, 2),
        (1, -1), (1, 2), (2, 0), (2, 1),
    ]
    for y, x in ring:
        pix(3, 2, y, x)
    for y in (0, 1):
        for x in (0, 1):
            pix(3, 2, y, x, -3.0)
    b1[3] = -7.0

    outline = (
        [(-2, x) for x in (-1, 0, 1, 2)]
        + [(-1, x) for x in (-2, -1, 2)]
        + [(0, -2), (1, -2)]
        + [(2, x) for x in (-2, -1, 2)]
    )
    for y, x in outline:
        pix(4, 3, y, x)
    b1[4] = -(len(outline) - 1.0)

    for out, color in ((5, 1), (6, 12)):
        for y in (-1, 0, 1, 2):
            for x in (-1, 0, 1, 2):
                pix(out, color, y, x)
        b1[out] = -15.0

    for y in range(-2, 3):
        pix(7, 11, y, -2)
    pix(7, 11, -2, -1)
    pix(7, 15, -2, -1)
    b1[7] = -5.0

    post = [
        [15, 15, 15, 15],
        [15, 0, 7, 15],
        [15, 7, 7, 15],
        [15, 15, 15, 15],
        [5, 5, 5, 5],
    ]
    for yy, row in enumerate(post):
        for xx, color in enumerate(row):
            pix(8, color, yy - 2, xx - 1)
    b1[8] = -19.0

    # Four movement detectors, anchored just outside platform edges.
    for out, direction in enumerate(("U", "D", "L", "R"), 9):
        if direction == "U":
            yellow = [(1, x) for x in range(-2, 3)]
            track = [(-2, 0), (-2, 1)]
        elif direction == "D":
            yellow = [(-1, x) for x in range(-2, 3)]
            track = [(2, 0), (2, 1)]
        elif direction == "L":
            yellow = [(y, 1) for y in range(-2, 3)]
            track = [(0, -2), (1, -2)]
        else:
            yellow = [(y, -1) for y in range(-2, 3)]
            track = [(0, 2), (1, 2)]
        for y, x in yellow:
            pix(out, 11, y, x)
        for y, x in track:
            pix(out, 5, y, x)
        b1[out] = -6.0

    pix(14, 16, 0, 0)
    pix(15, 17, 0, 0)

    for c in range(13):
        if c not in (4, 7):
            s2(c, c)
    for c in (14, 15, 20):
        s2(c, c)

    for c in (4, 0, 1, 2):
        s2(4, c)
    b2[4] = -1.0

    s2(7, 7)
    s2(7, 7, -1.0, y=-1)

    for c in (0, 1, 2, 7):
        s2(13, c)
    b2[13] = -1.0

    # Stationary green targets, excluding circles riding platforms.
    s2(16, 0)
    s2(16, 7, -1.0)
    for out, coord in ((17, 15), (18, 14)):
        s2(out, coord)
        s2(out, 0)
        s2(out, 7, -1.0)
        b2[out] = -1.0
    for out, coord in ((24, 15), (25, 14)):
        s2(out, 0)
        s2(out, 7, -1.0)
        s2(out, coord, -1.0)

    s2(19, 4)
    s2(19, 2)
    b2[19] = -1.0
    b2[21] = 1.0

    for out, coord in ((22, 15), (23, 14)):
        s2(out, 3)
        s2(out, coord)
        b2[out] = -1.0

    for c in range(16):
        s3(c, c)
    s3(16, 17)
    s3(17, 18)
    s3(22, 20)
    s3(23, 21)
    s3(28, 22)
    s3(29, 23)

    for out, coord in ((30, 15), (31, 14)):
        s3(out, coord)
        s3(out, 7)
        b3[out] = -1.0

    # Directional attraction evaluated AT the legal-move detector.
    # Loaded features have a positive legality baseline.
    specs = (
        (14, 64 + 18, 64 + 25, 64 + 16, 64 + 13, 1.0),
        (14, 64 + 18, 64 + 25, 64 + 16, 64 + 13, -1.0),
        (15, 32 + 17, 32 + 24, 32 + 16, 32 + 13, 1.0),
        (15, 32 + 17, 32 + 24, 32 + 16, 32 + 13, -1.0),
    )
    for d, (coord, hi, inv, present, loaded, sign) in enumerate(specs):
        out = 24 + d
        s3(out, 9 + d, 4.0)
        s3(out, present, 4.0)
        s3(out, coord, sign)
        s3(out, hi, -0.5 * sign)
        s3(out, inv, 0.5 * sign)
        b3[out] = -8.0 - 0.5 * sign

        out = 18 + d
        s3(out, 9 + d, 4.0)
        s3(out, loaded, 4.0)
        s3(out, coord, 0.05 * sign)
        s3(out, hi, -0.025 * sign)
        s3(out, inv, 0.025 * sign)
        b3[out] = -7.0 - 0.025 * sign

    # Smooth local connectivity potential.
    for y, x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        Wh[12, 12, y + 1, x + 1] = 0.2
    Uh[12, 0] = 1.0
    Uh[12, 1] = 1.0
    Uh[12, 2] = 0.15
    Uh[12, 8] = 0.05
    Wh[13, 12, 1, 1] = 1.0
    for c in (3, 5, 6):
        Uh[13, c] = 4.0
    bh[13] = -4.0

    # Axial Green's functions: unit response six pixels from a seed.
    lam = 4.0 / 17.0
    seed = 4096.0 * 15.0 / 17.0
    for out, offsets in (
        (0, ((0, -1), (0, 1))),
        (1, ((-1, 0), (1, 0))),
    ):
        for y, x in offsets:
            Wh[out, out, y + 1, x + 1] = lam
        for c in (3, 5, 6):
            Uh[out, c] = seed
        Wh[out, 13, 1, 1] = seed * 20.0

    for gate, beam in ((2, 0), (3, 1)):
        Wh[gate, beam, 1, 1] = 1.0
        bh[gate] = -8192.97
        for c in (0, 1, 2, 8):
            Uh[gate, c] = 8192.0
        Uh[gate, 4] = -8192.0
        Uh[gate, 0] += 0.47
        Uh[gate, 1] += 0.47

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for d, (y, x) in enumerate(directions):
        ray = 4 + d
        pivot = 3 if d < 2 else 2
        Wh[ray, ray, y + 1, x + 1] = 0.25
        Wh[ray, pivot, y + 1, x + 1] = 1024.0
        out = 8 + d
        Wh[out, ray, 1, 1] = 1.0
        bh[out] = -4096.012
        penalty = (0.025, 0.0, 0.06, 0.0)[d]
        Uh[out, 0] = 4096.0 - penalty
        Uh[out, 1] = 4096.0 - penalty

    for ray in (4, 5, 7):
        Wh[14, ray, 1, 1] = 1.0
    Uh[14, 2] = 16384.0
    bh[14] = -16384.012

    Wh[15, 13, 1, 1] = 1.0
    Uh[15, 3] = 2.0
    bh[15] = -2.0

    Wz, bz = p["Wz"], p["bz"]
    for d in range(4):
        Wz[d, 8 + d] = 1.0
    Wz[4, 14] = 1.0

    for z, feature in ((5, 3), (6, 13), (19, 22), (21, 4)):
        Wz[z, 32 + feature] = 1.0
    for d in range(4):
        Wz[7 + d, 32 + 9 + d] = 1.0
        Wz[11 + d, 32 + 18 + d] = 1.0
        Wz[15 + d, 32 + 24 + d] = 1.0
        Wz[15 + d, 32 + 13] = -2.0
    bz[20] = 1.0

    for z in (23, 24):
        for d in range(4):
            Wz[z, 8 + d] = 1.0
    bz[24] = -0.008
    for z in (25, 26):
        Wz[z, 14] = 1.0
    bz[26] = -0.004

    # A small position checksum, kept in the near-linear tanh regime.
    Wz[30, 64 + 30] = 4.096
    Wz[30, 64 + 31] = 4.096 * 0.371

    # When one green and a red are visible, deliver the green first.
    for d, (target, platform, sign) in enumerate(
        ((17, 31, -1.0), (17, 31, 1.0),
         (16, 30, -1.0), (16, 30, 1.0))
    ):
        z = 40 + d
        Wz[z, 32 + target] = sign
        Wz[z, 32 + platform] = -sign
        Wz[z, 32] = 10.0
        Wz[z, 64] = -5.0 * 4096.0
        Wz[z, 33] = 5.0
        Wz[z, 45] = -5.0
        bz[z] = -10.0

    # Heading survives intervening click actions.
    movement_actions = [1, 2, 3, 4]
    for i in range(4):
        p["Wm"][i, i] = 2.0
        for j, action in enumerate(movement_actions):
            p["Wma"][i, action] = 6.0 if i == j else -6.0
        p["Wma"][i, 0] = 6.0 if i == 3 else -6.0

    angle, gain = 0.17, 1.04
    ca, sa = math.cos(angle), math.sin(angle)
    p["Wm"][4, 4] = gain * ca
    p["Wm"][4, 5] = -gain * sa
    p["Wm"][5, 4] = gain * sa
    p["Wm"][5, 5] = gain * ca
    p["bm"][4] = 0.015

    p["Wmz"][6, 6] = 4.0
    p["Wma"][6, 6] = 4.0
    p["bm"][6] = -6.0

    p["Wmz"][8, 30] = 1.0
    for d, action in enumerate(movement_actions):
        for j, offset in enumerate((-4.0, 4.0)):
            unit = 9 + 2 * d + j
            p["Wmz"][unit, 30] = 400000.0
            p["Wm"][unit, 8] = -400000.0
            p["Wma"][unit, action] = -100000.0
            p["bm"][unit] = 100000.0 + offset

    Wo, bo = p["Wo"], p["bo"]
    bo[:] = -30.0
    bo[0] = -4.0
    bo[5] = -100.0
    bo[7] = -4.5
    Wo[0, 19] = 200.0
    Wo[7, 19] = 150.0

    clockwise = {0: 3, 3: 1, 1: 2, 2: 0}
    reverse = {0: 1, 1: 0, 2: 3, 3: 2}
    for d, action in enumerate(movement_actions):
        bo[action] = -100.0
        Wo[action, 7 + d] = 100.0
        Wo[action, 11 + d] = 100.0
        Wo[action, 6] = -90.0
        Wo[action, 15 + d] = 15.0
        Wo[action, 40 + d] = 12.0

        scores = []
        for previous in range(4):
            if d == clockwise[previous]:
                score = 1.05
            elif d == previous:
                score = 0.90
            elif d == reverse[previous]:
                score = -1.0
            else:
                score = 0.25
            scores.append(score)
            Wo[action, 128 + previous] = 0.5 * score
        bo[action] += 0.5 * sum(scores)

        Wo[action, 128 + 9 + 2 * d] = 20.0
        Wo[action, 128 + 10 + 2 * d] = -20.0

    bo[4] += 0.04
    Wo[1, 132] = 0.25
    Wo[2, 132] = -0.25
    Wo[3, 133] = -0.25
    Wo[4, 133] = 0.25

    bo[6] = -35.0
    for d in range(4):
        Wo[6, d] = 40.0
    Wo[6, 4] = 40.0
    Wo[6, 5] = 100.0
    Wo[6, 23] = 3750.0
    Wo[6, 24] = -3750.0
    Wo[6, 25] = 7500.0
    Wo[6, 26] = -7500.0
    Wo[6, 134] = -30.0

    Wc = p["Wc"][0]
    Wc[0] = 0.005
    Wc[3] = 20.0
    Wc[28] = 0.20
    Wc[29] = -0.001
    Wc[15] = 0.00008
    Wc[14] = -0.00002
    for h in range(8, 12):
        Wc[32 + h] = 1.0
    Wc[32 + 14] = 0.20
    Wc[32 + 15] = 5.0
    return p
