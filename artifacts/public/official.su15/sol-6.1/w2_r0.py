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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    def c1(o, i, dy, dx, v):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, dy, dx, v):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def ch(o, i, dy, dx, v):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    colours = (10, 6, 15, 11, 12, 8)
    priority = np.array([16.0 ** (5 - r) for r in range(6)],
                        dtype=np.float32)

    # F1[0:6]: unique corner-derived square markers.
    c1(0, 10, 0, 0, 1)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        c1(0, 10, dy, dx, -1)

    for r in range(1, 6):
        offsets = ((-1, 0) if r == 1 else
                   ((-1, 0, 1) if r == 2 else (-1, 0, 1, 2)))
        n = len(offsets) ** 2
        for dy in offsets:
            for dx in offsets:
                c1(r, colours[r], dy, dx, 1)
        c1(r, colours[r], -2, -1, -n)
        c1(r, colours[r], -1, -2, -n)
        p["b1"][r] = -(n - 1)

    # Rounded-goal staircase markers.
    for dy, dx in ((-2, -1), (-1, -2), (-1, -1)):
        c1(6, 9, dy, dx, 1)
    c1(6, 9, -2, -2, -3)
    p["b1"][6] = -2

    # Coordinate hinges.
    c1(7, 16, 0, 0, -1)
    p["b1"][7] = 10.0 / 63.0
    c1(8, 17, 0, 0, -1)
    p["b1"][8] = 24.0 / 63.0
    c1(9, 16, 0, 0, 1)
    p["b1"][9] = -9.0 / 63.0
    c1(10, 16, 0, 0, 1)
    p["b1"][10] = -62.0 / 63.0

    # Square exclusion footprints, approximately centred on the markers.
    footprint_offsets = (
        (-2, -1, 0, 1, 2),
        (-2, -1, 0, 1),
        (-1, 0, 1),
        (-1, 0),
        (0,),
        (1,),
    )
    for colour, offsets in zip(colours, footprint_offsets):
        for dy in offsets:
            for dx in offsets:
                c1(11, colour, dy, dx, 1)

    # Capped goal-colour counts on four surrounding stripes.
    stripes = (
        tuple((dy, -2) for dy in range(-2, 3)),
        tuple((dy, 2) for dy in range(-2, 3)),
        tuple((-2, dx) for dx in range(-2, 3)),
        tuple((2, dx) for dx in range(-2, 3)),
    )
    stripe_shifts = ((0, -1), (0, 1), (-1, 0), (1, 0))
    for k, stripe in enumerate(stripes):
        a = 12 + 2 * k
        for dy, dx in stripe:
            c1(a, 9, dy, dx, 1)
            c1(a + 1, 9, dy, dx, 1)
        p["b1"][a + 1] = -1

    c1(20, 2, 0, 0, 1)

    # Exact polygon markers for the three creature types.
    glyphs = (
        (7, (
            (-2, 0), (-1, -1), (-1, 1),
            (0, -2), (0, 2),
            (1, -1), (1, 0), (1, 1),
        ), (-2, -2)),
        (14, (
            (-2, 0), (-1, -1), (-1, 1),
            (0, -2), (0, 0), (0, 2),
            (1, -1), (1, 1),
        ), (-2, -2)),
        (13, (
            (-2, -2), (-2, 0), (-2, 2),
            (-1, -1), (-1, 1),
            (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
            (1, -1), (1, 0), (1, 1),
        ), (-2, -1)),
    )
    for k, (colour, points, hole) in enumerate(glyphs):
        out = 21 + k
        for dy, dx in points:
            c1(out, colour, dy, dx, 1)
        c1(out, colour, hole[0], hole[1], -len(points))
        p["b1"][out] = -(len(points) - 1)

    # Local creature-colour presence, represented as a capped count pair.
    for colour in (7, 14, 13):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                c1(24, colour, dy, dx, 1)
                c1(25, colour, dy, dx, 1)
    p["b1"][25] = -1

    def subtract_goal_stripes(out, scale):
        for k, (dy, dx) in enumerate(stripe_shifts):
            a = 12 + 2 * k
            c2(out, a, dy, dx, -scale)
            c2(out, a + 1, dy, dx, scale)

    # F2[0:6]: raw play-area square markers.
    # F2[6:12]: markers discounted by surrounding goal colour.
    for r in range(6):
        shift = -1 if r >= 3 else 0
        c2(r, r, shift, shift, 1)
        c2(r, 7, 0, 0, -1000)

        c2(6 + r, r, shift, shift, 1)
        c2(6 + r, 7, 0, 0, -1000)
        subtract_goal_stripes(6 + r, 0.25)

        # HUD rank source, excluding the palette and play area.
        c2(12, r, shift, shift, float(priority[r]))

    c2(12, 9, 0, 0, -1.0e10)
    c2(12, 8, 0, 0, -1.0e10)

    c2(13, 6, -1, -1, 1)

    # Empty-square-footprint indicator.
    p["b2"][14] = 1
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c2(14, 11, dy, dx, -1)

    c2(15, 7, 0, 0, 1)
    c2(16, 10, 0, 0, 1)

    # Red requirement cue.
    c2(17, 20, 0, 0, 1)
    c2(17, 9, 0, 0, -1000)
    c2(17, 8, 0, 0, -1000)

    # Raw and goal-discounted creature markers.
    for k in range(3):
        c2(18 + k, 21 + k, 0, 0, 1)
        c2(18 + k, 7, 0, 0, -1000)

        c2(21 + k, 21 + k, 0, 0, 1)
        c2(21 + k, 7, 0, 0, -1000)
        subtract_goal_stripes(21 + k, 0.25)

        c2(24, 21 + k, 0, 0, 1)

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                c2(25, 21 + k, dy, dx, 1)

    c2(24, 7, 0, 0, -1000)
    c2(25, 7, 0, 0, -10000)

    # F2[26] is one only outside the short-range creature neighbourhood.
    p["b2"][26] = 1
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c2(26, 24, dy, dx, -1)
            c2(26, 25, dy, dx, 1)

    # F3[0:4]: local and axial rank-priority sources.
    for r in range(6):
        pr = float(priority[r])
        p["W3"][0, r] = pr
        p["W3"][1, 6 + r] = pr
        p["W3"][2, 64 + 6 + r] = pr
        p["W3"][8, r] = 1

    p["W3"][3, 32 + 12] = 1
    p["W3"][4, 13] = 1

    p["W3"][5, 14] = -1
    p["b3"][5] = 1
    p["W3"][6, 15] = 1
    p["W3"][6, 16] = 1
    p["W3"][7, 17] = 1

    for k in range(3):
        p["W3"][9 + k, 21 + k] = 1

    p["W3"][12, 24] = 1
    p["W3"][13, 64 + 22] = 16
    p["W3"][13, 64 + 23] = 256
    p["W3"][14, 25] = 1

    # Urgent vulnerable-square rescue markers.
    for r in range(3):
        p["W3"][15, r] = 1
    p["W3"][15, 26] = -1

    # H[0:2]: horizontal prefix scans of play-area priorities.
    ch(0, 0, 0, -1, 1)
    ch(1, 1, 0, 1, 1)
    p["Uh"][0, 2] = 1
    p["Uh"][1, 2] = 1

    # H[2]: downward prefix scan of the row-broadcast HUD requirement.
    ch(2, 2, -1, 0, 1)
    p["Uh"][2, 3] = 1

    # H[3], H[15]: priority scans for higher creature types.
    ch(3, 3, 0, -1, 1)
    ch(15, 15, 0, 1, 1)
    p["Uh"][3, 13] = 1
    p["Uh"][15, 13] = 1

    # H[4]-H[5]: clipped, approximately binary selected-square sources.
    # Undelivered squares receive twelve times their rank priority.
    # Deeply delivered squares retain only one times that priority.
    for out in (4, 5):
        p["Uh"][out, 0] = 1
        p["Uh"][out, 1] = 11
        ch(out, 0, 0, 0, -1)
        ch(out, 1, 0, 0, -1)
        ch(out, 2, 0, 0, -1.2)
    p["bh"][5] = -1

    # H[6]-H[7]: binary Manhattan wavefront from goal seeds.
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for out in (6, 7):
        for dy, dx in cardinal:
            ch(out, 6, dy, dx, 1)
            ch(out, 7, dy, dx, -1)
        p["Uh"][out, 4] = 1
    p["bh"][7] = -1

    # H[8]: time until arrival of the wavefront.
    ch(8, 8, 0, 0, 1)
    ch(8, 6, 0, 0, -1)
    ch(8, 7, 0, 0, 1)
    p["bh"][8] = 1

    # H[9]: square gathering potential.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ch(9, 9, dy, dx, 0.1)
    ch(9, 4, 0, 0, 1)
    ch(9, 5, 0, 0, -1)
    p["Uh"][9, 8] = -0.015
    p["Uh"][9, 15] = 2

    # H[11]: positive while lower ranks still require gathering.
    ch(11, 0, 0, 0, 1)
    ch(11, 1, 0, 0, 1)
    ch(11, 2, 0, 0, -4.5)

    # H[10]: cap the potential during delivery, but not gathering.
    ch(10, 9, 0, 0, 1)
    ch(10, 8, 0, 0, 0.00001)
    ch(10, 11, 0, 0, -1.0e8)
    p["bh"][10] = -0.005

    # H[12]: creature threat potential.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ch(12, 12, dy, dx, 0.1)
    p["Uh"][12, 12] = 1

    # H[13]: creature selection. Higher existing creature types suppress
    # lower ones; creature transport waits until square delivery is done.
    p["Uh"][13, 9] = 8
    p["Uh"][13, 10] = 128
    p["Uh"][13, 11] = 2048
    ch(13, 0, 0, 0, -100000)
    ch(13, 1, 0, 0, -100000)
    ch(13, 3, 0, 0, -1)
    ch(13, 15, 0, 0, -1)

    # H[14]: creature gathering/transport potential.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ch(14, 14, dy, dx, 0.1)
    ch(14, 13, 0, 0, 1)

    # Click score: gathering or clipped delivery, threat avoidance,
    # creature completion, and a global fallback goal direction.
    p["Wc"][0, 32 + 9] = 50000
    p["Wc"][0, 32 + 10] = -50000
    p["Wc"][0, 32 + 12] = -25000
    p["Wc"][0, 32 + 14] = 50000
    p["Wc"][0, 32 + 8] = -0.5
    p["Wc"][0, 5] = -1000000
    p["Wc"][0, 14] = -1000000
    p["Wc"][0, 6] = -10000000

    # Click normally; undo when the HUD reports insufficient material.
    p["bo"][6] = 1
    p["Wz"][0, 32 + 7] = 1
    p["Wo"][7, 0] = 100

    return p
