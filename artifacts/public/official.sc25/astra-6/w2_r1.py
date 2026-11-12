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

    def c1(o, i, dy=0, dx=0, v=1.0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, dy=0, dx=0, v=1.0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def ch(o, i, dy=0, dx=0, v=1.0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    def f3(o, terms, bias=0.0):
        for i, v in terms.items():
            p["W3"][o, i] += v
        p["b3"][o] = bias

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rotations = [180, 0, 90, 270]

    # Exact striped-player detectors. Small anchors are top-left;
    # large anchors are top-left+(1,1).
    base = np.array([[10, 10], [9, 9]], dtype=np.int32)
    for size in (1, 2):
        for d, rot in enumerate(rotations):
            a = np.rot90(base, k=(-rot // 90) % 4)
            a = np.repeat(np.repeat(a, size, axis=0), size, axis=1)
            o = (0 if size == 1 else 4) + d
            offset = 0 if size == 1 else -1
            for y in range(a.shape[0]):
                for x in range(a.shape[1]):
                    c1(o, int(a[y, x]), y + offset, x + offset)

            # Only small templates need length checks, to distinguish
            # them from a stripe inside the large player or an exit.
            # Check along the stripe, not across it toward the exit.
            if size == 1:
                outside = (
                    [(0, -1), (0, 2)] if d < 2
                    else [(-1, 0), (2, 0)]
                )
                for y, x in outside:
                    c1(o, 9, y, x, -1)
                    c1(o, 10, y, x, -1)
            p["b1"][o] = 1 - a.size

    # Configuration-space clearance for the two footprints.
    solid = (0, 1, 3, 4, 5, 6, 7, 8, 12, 13)
    for o, extent in ((8, range(0, 2)), (9, range(-1, 3))):
        p["b1"][o] = 1
        for y in extent:
            for x in extent:
                for colour in solid:
                    c1(o, colour, y, x, -1)

    # Grates block movement but not firing.
    for colour in (3, 5, 8, 12, 13):
        c1(10, colour)

    # Exit-corner detectors.
    for o, (dy, dx) in enumerate(
        [(-1, -1), (-1, 1), (1, -1), (1, 1)], start=11
    ):
        c1(o, 10)
        c1(o, 9, dy, 0)
        c1(o, 9, 0, dx)
        c1(o, 9, dy, dx)
        p["b1"][o] = -3

    # Switch interior corners, for either gate colour.
    for o, sign in ((15, -1), (16, 1)):
        c1(o, 12)
        c1(o, 13)
        c1(o, 6, sign, 0)
        c1(o, 6, 0, sign)
        p["b1"][o] = -2

    c1(17, 16, v=63)
    c1(18, 17, v=63)

    # Coordinate ramps for the nine rune-cell centres.
    for j, row in enumerate((50, 55, 60)):
        c1(19 + j, 16, v=63)
        p["b1"][19 + j] = -row
    for j, col in enumerate((25, 30, 35)):
        c1(22 + j, 17, v=63)
        p["b1"][22 + j] = -col

    for colour in (6, 11, 14, 15):
        c1(25, colour)

    # Resize icon.
    for y, x in ((-2, 1), (-1, 2), (1, -2), (2, -1)):
        c1(26, 15, y, x)
    c1(26, 2)
    p["b1"][26] = -4

    # Teleport icon.
    tele_icon = np.full((5, 5), 2, dtype=np.int32)
    tele_icon[0:2, 0:2] = 11
    tele_icon[0:2, 3:5] = 11
    tele_icon[3:5, 3:5] = 11
    for y in range(5):
        for x in range(5):
            c1(27, int(tele_icon[y, x]), y - 2, x - 2)
    p["b1"][27] = -24

    # Second-layer local features.
    for d in range(4):
        c2(d, d)
        c2(d, d + 4)
        c2(4, d)
        c2(5, d + 4)

    c2(6, 8)
    c2(7, 10)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for i in range(11, 15):
                c2(8, i, dy, dx)
            c2(9, 15, dy, dx)
            c2(9, 16, dy, dx)
            c2(10, 15, dy, dx)
            c2(11, 15, dy, dx, 128)
            c2(12, 15, dy, dx, 128)

    c2(11, 17)
    c2(12, 18)
    p["b2"][11:13] = -128
    c2(13, 17)
    c2(14, 18)

    for j in range(6):
        o, i = 15 + j, 19 + j
        c2(o, i, v=-8.0 / 3.0)
        if j < 3:
            c2(o, i, -1, 0, 4.0 / 3.0)
            c2(o, i, 1, 0, 4.0 / 3.0)
        else:
            c2(o, i, 0, -1, 4.0 / 3.0)
            c2(o, i, 0, 1, 4.0 / 3.0)
        p["b2"][o] = -1.0 / 3.0

    c2(21, 25)
    for d, (dy, dx) in enumerate(dirs):
        for i in range(4):
            c2(22 + d, i)
            c2(26 + d, i + 4)
        c2(22 + d, 8, dy, dx, -1)
        c2(26 + d, 9, dy, dx, -1)

    c2(30, 26)
    c2(31, 27)

    # Local masks and axial relations.
    f3(0, {6: -1}, 1)
    f3(1, {7: 1, 9: -1})
    f3(2, {8: 1})
    f3(3, {9: 1})
    f3(4, {4: -1, 5: -1}, 1)
    f3(5, {5: -1}, 1)
    f3(6, {4: -1}, 1)

    for d in range(4):
        f3(7 + d, {22 + d: 1, 26 + d: 1})

    f3(11, {i: 1 for i in range(15, 21)}, -1.5)

    # Resize-pattern mismatch via three-bit parity.
    parity_terms = {16: 1, 19: 1, 21: 1}
    for o, bias in ((12, 0), (13, -1), (14, -2)):
        f3(o, parity_terms, bias)

    # Fire-pattern mismatch.
    f3(15, {19: 1, 21: -1})
    f3(16, {21: 1, 19: -1})

    # Teleport-pattern mismatch.
    f3(17, {15: 2, 16: 1, 19: 1, 20: -1, 21: -4}, -1)
    f3(18, {15: -2, 16: -1, 19: -1, 20: 1, 21: 4}, -2)

    # Facing an axially aligned switch.
    K = 1024.0
    front_terms = [
        {13: 1, 64 + 11: -1, 0: K, 64 + 10: K},
        {64 + 11: 1, 13: -1, 1: K, 64 + 10: K},
        {14: 1, 32 + 12: -1, 2: K, 32 + 10: K},
        {32 + 12: 1, 14: -1, 3: K, 32 + 10: K},
    ]
    for d, terms in enumerate(front_terms):
        f3(19 + 2 * d, terms, -2 * K)
        f3(20 + 2 * d, terms, -2 * K - 1)

    f3(27, {30: 1})
    f3(28, {31: 1})
    f3(29, {5: 1})
    f3(30, {32 + 10: 1, 64 + 10: 1})
    f3(31, {}, 1)

    # Obstacle-masked navigation diffusion.
    for dy, dx in dirs:
        ch(0, 0, dy, dx, 0.24)
    ch(0, 1, v=0.02)
    ch(0, 2, v=0.02)
    p["Uh"][0, 2] = 1
    p["Uh"][0, 0] = -10000

    # Horizontal and vertical firing fields.
    for dx in (-1, 1):
        ch(1, 1, 0, dx, 0.49)
    for dy in (-1, 1):
        ch(2, 2, dy, 0, 0.49)
    for o in (1, 2):
        p["Uh"][o, 3] = 1
        p["Uh"][o, 1] = -1000

    # Directional samples at the player.
    for d, (dy, dx) in enumerate(dirs):
        o = 3 + d
        ch(o, 0, dy, dx)
        ray = 2 if d < 2 else 1
        ch(o, ray, dy, dx, 50)
        p["Uh"][o, 4] = -100000

    ch(7, 1)
    ch(7, 2)
    p["Uh"][7, 4] = -100000

    def flood_pair(plus, minus):
        for o in (plus, minus):
            for dy, dx in dirs:
                ch(o, plus, dy, dx, 1)
                ch(o, minus, dy, dx, -1)
        p["bh"][minus] = -1

    # Saturating global objective-presence flag.
    flood_pair(8, 9)
    for o in (8, 9):
        p["Uh"][o, 30] = 1
        for q in range(3, 7):
            ch(o, q, v=0.01)

    # Saturating large-player flag.
    flood_pair(10, 11)
    p["Uh"][10, 29] = 1
    p["Uh"][11, 29] = 1

    # Resize mismatch, gated by the size/objective relation.
    for i, v in ((8, 1), (9, -1), (10, 1), (11, -1)):
        ch(12, i, v=v)
        ch(13, i, v=-v)
    p["bh"][12] = -2
    for o in (12, 13):
        p["Uh"][o, 12] = 1
        p["Uh"][o, 13] = -2
        p["Uh"][o, 14] = 2

    # Fire-ready flood, with rune mismatch encoded in its local cap.
    flood_pair(14, 15)
    ch(14, 7)
    ch(15, 7)
    p["Uh"][15, 15] = -1
    p["Uh"][15, 16] = -1

    for d in range(4):
        p["Wz"][d, 3 + d] = 1

    # Fire only when a ray reaches the player and facing is correct.
    p["Wz"][4, 7] = 1
    for d in range(4):
        p["Wz"][4, 32 + 19 + 2 * d] = 128
        p["Wz"][4, 32 + 20 + 2 * d] = -128
    p["bz"][4] = -128.0001

    p["Wz"][5, 32 + 27] = 1
    p["Wz"][5, 32 + 28] = 1
    p["Wz"][6, 7] = 1
    p["Wz"][7, 32 + 29] = 1

    for d in range(4):
        p["Wz"][8 + d, 32 + 7 + d] = 1
        p["Wz"][8 + d, 7] = -1e6

    # Bounded pairwise comparisons preserve tiny potential gradients.
    unit = 0
    for i in range(4):
        for j in range(i + 1, 4):
            p["Wmz"][unit, i] = 1e32
            p["Wmz"][unit, j] = -1e32
            p["Wo"][1 + i, 128 + unit] = 1
            p["Wo"][1 + j, 128 + unit] = -1
            unit += 1

    for d in range(4):
        p["Wmz"][unit, 8 + d] = 20
        p["Wo"][1 + d, 128 + unit] = -5
        unit += 1

    p["bo"][:] = -1000
    p["bo"][1:5] = 0
    p["bo"][6] = -18.4

    p["Wmz"][unit, 5] = 20
    p["Wo"][6, 128 + unit] = 20
    unit += 1

    p["Wmz"][unit, 4] = 1e6
    p["Wo"][6, 128 + unit] = 30

    # Rune-cell restriction, then spell-dependent mismatch priority.
    p["Wc"][0, 11] = 10000
    p["Wc"][0, 17] = 10
    p["Wc"][0, 18] = 10
    p["Wc"][0, 32 + 12] = 100
    p["Wc"][0, 32 + 13] = 100
    p["Wc"][0, 32 + 14] = 1000
    p["Wc"][0, 32 + 15] = -1000

    return p
