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
    p = {k: np.zeros(s, dtype=np.float32)
         for k, s in shapes.items()}

    def c1(o, i, dy=0, dx=0, w=1.0):
        p["W1"][o, i, dy + 2, dx + 2] += w

    def c2(o, i, dy=0, dx=0, w=1.0):
        p["W2"][o, i, dy + 1, dx + 1] += w

    def rec(o, i, dy=0, dx=0, w=1.0):
        p["Wh"][o, i, dy + 1, dx + 1] += w

    def feature(o, terms, bias=0.0):
        for i, w in terms.items():
            p["W3"][o, i] += w
        p["b3"][o] = bias

    def unit(o, hs=None, fs=None, bias=0.0):
        for i, w in (hs or {}).items():
            p["Wz"][o, i] += w
        for i, w in (fs or {}).items():
            p["Wz"][o, 32 + i] += w
        p["bz"][o] = bias

    # Tile-center texture detectors.
    for dy, dx in ((-2, -1), (1, -2), (2, 1)):
        c1(0, 3, dy, dx)
    p["b1"][0] = -2.0

    for dy, dx in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
        c1(1, 3, dy, dx)
    p["b1"][1] = -3.0

    # Air, vertical occupancy, obstacles, and object colors.
    for color in (10, 9, 7):
        c1(2, color)
        c1(3, color)
        c1(16, color)
    for color in (14, 12, 15):
        c1(3, color)
        c1(17, color)
    for color in (14, 12):
        c1(16, color)
    for color in (5, 8, 14, 15):
        c1(4, color)

    c1(5, 8)
    c1(6, 7)
    c1(21, 15)
    c1(22, 12)

    # Player-center detectors and walking-query offsets.
    for out, shift in ((7, 0), (8, 2), (9, -2)):
        for dy in (-2, -1, 0, 1):
            c1(out, 9, dy, shift)
        c1(out, 5, 2, shift)
        p["b1"][out] = -4.0

    # Spike center, independent of its orientation.
    c1(10, 15)
    c1(10, 0, -2, 0)
    c1(10, 0, 2, 0)
    p["b1"][10] = -1.0

    # Platform boundaries.
    for dx in (-2, 2):
        c1(11, 5, -2, dx)
        c1(12, 5, 2, dx)
    p["b1"][11] = -1.0
    p["b1"][12] = -1.0

    # Blocked destinations. Transparent green markers are excluded.
    for out, shift in ((13, -2), (14, 2)):
        for color in (5, 8, 14, 15, 12):
            c1(out, color, 0, shift)
        c1(out, 10, -1, shift, -1.0)

    c1(15, 16, w=63.0)
    c1(18, 17, w=63.0)

    # Second layer: clipped masks and geometric markers.
    c2(0, 0)
    c2(0, 1)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c2(1, 2, dy, dx)
            c2(2, 2, dy, dx)
            c2(3, 3, dy, dx)
            c2(4, 3, dy, dx)
    p["b2"][2] = -1.0
    p["b2"][4] = -1.0

    c2(5, 4)
    c2(6, 10)
    for dx in (-1, 1):
        c2(7, 11, -1, dx)
        c2(8, 12, 1, dx)
    p["b2"][7] = -1.0
    p["b2"][8] = -1.0

    c2(9, 7)
    c2(10, 8, 0, 1)
    c2(11, 9, 0, -1)

    # Gravity markers from stable camera-relative player heights.
    c2(12, 7, w=20.0)
    c2(12, 15, w=0.2)
    p["b2"][12] = -26.8
    c2(13, 7, w=20.0)
    c2(13, 15, w=-0.2)
    p["b2"][13] = -13.2

    c2(14, 6)
    c2(15, 5)
    c2(16, 1)
    c2(16, 21)
    p["b2"][16] = -1.0
    c2(17, 17)
    c2(18, 15)
    c2(19, 18)
    c2(20, 16)
    c2(21, 13, 0, -1)
    c2(22, 14, 0, 1)

    c2(23, 15)
    p["b2"][23] = -33.0
    c2(24, 15, w=-1.0)
    p["b2"][24] = 35.0

    for out, threshold in (
        (25, 59.0), (26, 60.0), (27, 62.0),
        (28, 31.0), (29, 34.0), (30, 37.0),
    ):
        c2(out, 15)
        p["b2"][out] = -threshold
    c2(31, 22)

    # Axial features: local indices 0:32, row maxima 32:64,
    # column maxima 64:96.
    feature(0, {1: 1, 2: -1})
    feature(1, {3: 1, 4: -1})
    feature(2, {64: 1})
    feature(3, {32: 1})

    # Only non-removable center obstacles block vertical planning.
    feature(4, {
        64: 1, 32: 1, 5: 1, 17: -1, 6: 1,
    }, -2.0)

    feature(5, {64: 1, 7: -1})
    feature(6, {64: 1, 8: -1})
    feature(7, {12: 1})
    feature(8, {13: 1})
    feature(9, {9: 1})
    feature(10, {14: 1, 27: -1000})

    # Solid clickable centers, not transparent markers or spikes.
    feature(11, {
        64: 1, 32: 1, 17: 1, 6: -1, 1: -1, 2: 1,
    }, -2.0)
    feature(12, {18: 1})

    # Landing and removable-platform exploration targets.
    feature(13, {
        20: 1000, 64: 1000, 32: 1000, 7: 1000,
        18: -0.8, 23: -1000, 41: -1000,
    }, -3900.0)

    feature(14, {
        20: 1000, 64: 1000, 32: 1000, 8: 1000,
        18: 0.8, 24: -1000, 26: -1000, 41: -1000,
    }, -3950.0)

    feature(15, {
        20: 1000, 64: 1000, 18: -120,
    }, -1880.0)

    feature(16, {
        20: 1000, 64: 1000, 25: 120, 26: -240,
    }, -2000.0)

    # Direct-support click mask.
    feature(17, {
        64: 6, 32: 6, 17: 6, 73: 6,
        6: -6, 1: -6, 2: 6,
        28: 2, 29: -4, 30: 2,
    }, -27.0)

    feature(18, {64: 2, 32: 2, 15: 2}, -5.0)
    feature(19, {16: 1})
    feature(20, {15: 1})
    feature(21, {16: 1})
    feature(22, {16: 20, 19: 0.002}, -19.0)

    feature(23, {10: 1, 21: -1})
    feature(24, {11: 1, 22: -1})
    feature(25, {45: 1})
    feature(26, {44: 1})
    feature(27, {6: 1, 64: 1, 32: 1}, -2.0)

    # Center walls stop danger diffusion but removable blocks do not.
    feature(29, {
        64: 1, 32: 1, 5: 1, 17: -1,
    }, -2.0)

    # Removable blocks on the player's current row.
    feature(30, {
        64: 1, 32: 1, 17: 1, 41: 1,
        6: -1, 1: -1, 2: 1,
    }, -3.0)

    # Contractive spike-danger signal.
    rec(14, 14, -1, 0, 0.45)
    rec(14, 14, 1, 0, 0.45)
    p["Uh"][14, 27] = 1.0
    p["Uh"][14, 2] = 100.0
    p["Uh"][14, 29] = -100.0
    p["bh"][14] = -100.0

    # Five channels per planning bank:
    # D: positive left-minus-right; R: right continuation;
    # E: positive horizontal-minus-raw-vertical;
    # C: clipped vertical continuation; Q: E+C+terminal seed.
    #
    # E and C share the same raw vertical comparison value.
    # Since all continuation drives are nonpositive, E+C is
    # bounded by the largest source value.
    B = 2000.0
    for base, dy, unsupported, inactive, landing, frontier in (
        (0, -1, 5, 25, 13, 15),
        (5, 1, 6, 26, 14, 16),
    ):
        d, r, e, c, q = range(base, base + 5)

        rec(d, q, 0, -1)
        rec(d, q, 0, 1, -1.0)
        rec(r, q, 0, 1)
        rec(e, d)
        rec(e, r)
        rec(e, q, dy, 0, -1.0)
        rec(c, q, dy, 0)
        rec(q, e)
        rec(q, c)
        rec(q, 14, w=-80000.0)

        for out in (d, r):
            p["Uh"][out, 3] = B
            p["Uh"][out, unsupported] = -B
            p["Uh"][out, 4] = -B
            p["Uh"][out, 11] = -3.0
            p["Uh"][out, inactive] = -B
            p["bh"][out] = -B - 0.16

        # No positive material exception: traversability is
        # represented solely by subtracting a masked obstacle.
        p["Uh"][c, 2] = B
        p["Uh"][c, 4] = -B
        p["Uh"][c, 11] = -3.0
        p["bh"][c] = -B - 0.06

        for out in (d, r, e, c):
            p["Uh"][out, landing] -= 10000.0
            p["Uh"][out, frontier] -= 10000.0
            p["Uh"][out, 10] -= 10000.0

        p["Uh"][e, inactive] -= B
        p["Uh"][q, landing] = 1.0
        p["Uh"][q, frontier] = 1.0
        p["Uh"][q, 10] = 500.0

    # Safe walking-query readouts.
    for out, marker in ((10, 23), (11, 24)):
        rec(out, 4)
        rec(out, 9)
        p["Uh"][out, marker] = 2000.0
        p["bh"][out] = -2000.0

    # Independent upward and downward fall readouts.
    for out, source in ((12, 3), (13, 8)):
        rec(out, source)
        p["Uh"][out, 9] = 2000.0
        p["bh"][out] = -2000.0

    # Values at support and side-click candidates.
    rec(15, 4)
    rec(15, 9)
    p["Uh"][15, 17] = 2000.0
    p["Uh"][15, 30] = 2000.0
    p["bh"][15] = -2000.0

    # Action comparisons.
    unit(0, {10: 1}, {23: 0.3})
    unit(1, {11: 1}, {24: 0.3})

    G = 2048.0

    # Support removal uses only the active gravity's fall value.
    unit(2, {12: 1}, {7: G, 17: G, 20: -G},
         -2 * G + 0.7)
    unit(3, {13: 1}, {8: G, 17: G, 20: -G},
         -2 * G + 0.7)

    # Reversal uses the opposite gravity's fall value.
    unit(4, {13: 1}, {7: G, 20: G},
         -2 * G + 0.8)
    unit(5, {12: 1}, {8: G, 20: G},
         -2 * G + 0.8)

    # Clearing a side block is safe and can open a new walking edge.
    unit(6, {15: 1}, {30: G, 20: -G},
         -G + 0.6)

    # Low-priority expansion when no other continuation is valuable.
    unit(7, {}, {21: G, 17: -G, 30: -G, 20: -G},
         -G + 20.0)

    p["Wm"][0, 0] = 1.0
    p["Wma"][0, 3] = -2.0
    p["Wma"][0, 4] = 2.0

    p["bo"][:] = -1000.0
    p["Wo"][3, 0] = 1.0
    p["Wo"][4, 1] = 1.0
    p["bo"][3] = 0.0
    p["bo"][4] = 0.015
    p["Wo"][3, 128] = -0.025
    p["Wo"][4, 128] = 0.025

    p["Wo"][6, 2:8] = 1.0
    p["bo"][6] = -0.01

    # Switches, safe side clearing, support removal, then expansion.
    p["Wc"][0, 18] = 10000.0
    p["Wc"][0, 30] = 2000.0
    p["Wc"][0, 17] = 1000.0
    p["Wc"][0, 22] = 20.0
    p["Wc"][0, 32 + 15] = 0.1

    return p
