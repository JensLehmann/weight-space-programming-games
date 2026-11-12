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
        p["W1"][o, i, dy + 2, dx + 2] += np.float32(v)

    def c2(o, i, dy=0, dx=0, v=1.0):
        p["W2"][o, i, dy + 1, dx + 1] += np.float32(v)

    def c3(o, i, v=1.0):
        p["W3"][o, i] += np.float32(v)

    def rec(o, i, dy=0, dx=0, v=1.0):
        p["Wh"][o, i, dy + 1, dx + 1] += np.float32(v)

    def diffuse(o):
        for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            rec(o, o, dy, dx, .2)

    def zw(o, i, v=1.0):
        p["Wz"][o, i] += np.float32(v)

    # First layer: local templates and coordinate features.
    c1(0, 16, v=-63)
    c1(0, 17, v=-63)
    p["b1"][0] = 70

    blocked = (1, 3, 4, 5, 6, 12, 13)
    p["b1"][1] = 4
    for dy in (0, 1):
        for dx in (0, 1):
            c1(1, 18, dy, dx, -1)
            for colour in blocked:
                c1(1, colour, dy, dx)

    # A large player occupying a large teleport destination.
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            c1(2, 9, dy, dx)
            c1(2, 10, dy, dx)
    c1(2, 11, -2, -2)
    p["b1"][2] = -16

    # Row coordinate with a small, separately recoverable hint-black term.
    c1(3, 16, v=63)
    c1(3, 0, v=.125)

    def player_colour(d, y, x, size):
        half = size // 2
        if d == 0:
            return 9 if y < half else 10
        if d == 1:
            return 10 if y < half else 9
        if d == 2:
            return 9 if x < half else 10
        return 10 if x < half else 9

    for d in range(4):
        # Complementary small-player tests tolerate an adjacent exit edge.
        for alternative in range(2):
            out = 5 + 2 * d + alternative
            for y in range(2):
                for x in range(2):
                    c1(out, player_colour(d, y, x, 2), y, x)
            p["b1"][out] = -3
            edges = ((-1, 0), (2, 0)) if alternative == 0 else (
                (0, -1), (0, 2))
            for dy, dx in edges:
                c1(out, 9, dy, dx, -5)
                c1(out, 10, dy, dx, -5)

        # The complete large-player pattern needs no exterior rejection.
        out = 13 + d
        for y in range(4):
            for x in range(4):
                c1(out, player_colour(d, y, x, 4), y - 1, x - 1)
        p["b1"][out] = -15

    # Exit corners.
    for out, (sy, sx) in enumerate(
        ((-1, -1), (-1, 1), (1, -1), (1, 1)), 17
    ):
        c1(out, 9)
        c1(out, 9, sy, 0)
        c1(out, 9, 0, sx)
        c1(out, 10, sy, sx)
        p["b1"][out] = -3

    # Two anchors of each yellow-framed fire target.
    for dy in (1, 2):
        for dx in (1, 2):
            c1(21, 12, dy, dx)
            c1(21, 13, dy, dx)
    c1(21, 6, 0, 1)
    c1(21, 6, 1, 0)
    p["b1"][21] = -5

    for dy in (-1, 0):
        for dx in (-1, 0):
            c1(4, 12, dy, dx)
            c1(4, 13, dy, dx)
    c1(4, 6, -2, -1)
    c1(4, 6, -1, -2)
    p["b1"][4] = -5

    # Small teleport destination, excluding UI copies.
    for dy in (0, 1):
        for dx in (0, 1):
            c1(22, 15, dy, dx)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(22, 3, dy, dx, -10)
    p["b1"][22] = -3

    for out, inp, slope, intercept in (
        (23, 17, 63, -29), (24, 17, -63, 29),
        (25, 16, 63, -54), (26, 16, -63, 54),
    ):
        c1(out, inp, v=slope)
        p["b1"][out] = intercept

    # Spell-cell upper-left pixel and selected-cell colour.
    for colour in range(16):
        if colour != 3:
            c1(27, colour)
    c1(27, 3, -1, 0)
    c1(27, 3, 0, -1)
    p["b1"][27] = -2
    for colour in (6, 11, 14, 15):
        c1(28, colour)

    # Available spell icons.
    for out, colour in ((29, 15), (30, 11), (31, 6)):
        for dy in (0, 1):
            for dx in (0, 1):
                c1(out, colour, dy, dx)
        c1(out, 3, -1, 0)
        p["b1"][out] = -4

    # Second layer: masks anchored at actual player positions.
    for d in range(4):
        for raw in (5 + 2 * d, 6 + 2 * d):
            for out in (0, 1, 10, 11, 12 + d):
                c2(out, raw)
        c2(0, 13 + d, 1, 1, 2)
        c2(1, 13 + d, 1, 1, 2)
        c2(12 + d, 13 + d, 1, 1, 2)
        c2(27, 13 + d, 1, 1)

    p["b2"][1] = -1
    p["b2"][11] = -1

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            for raw in range(17, 21):
                c2(2, raw, dy, dx)
            for raw in (4, 21):
                c2(3, raw, dy, dx)
                c2(4, raw, dy, dx, -8)

    c2(4, 1)
    c2(5, 22)
    c2(6, 29)
    c2(7, 30)
    c2(8, 31)
    c2(29, 2, 1, 1)

    # Four small footprints cover a possible enlargement footprint.
    for dy in (-1, 1):
        for dx in (-1, 1):
            c2(9, 1, dy, dx)

    # Board cell anchors.
    c2(16, 27)
    c2(16, 0, v=-100)

    # Middle column and middle row.
    c2(17, 27)
    c2(17, 23, v=-.2)
    c2(17, 24, v=-.2)
    c2(18, 27)
    c2(18, 25, v=-.2)
    c2(18, 26, v=-.2)
    for out in (17, 18):
        c2(out, 0, v=-100)

    # Left column. Column = 29 + positive hinge - negative hinge.
    c2(19, 27, v=100)
    c2(19, 23, v=-.2)
    c2(19, 24, v=.2)
    c2(19, 0, v=-100)
    p["b2"][19] = -100

    # Top row.
    c2(20, 27, v=100)
    c2(20, 3, v=-.2)
    c2(20, 0, v=-100)
    p["b2"][20] = 10.8 - 100

    # Filled bit.
    c2(21, 28)
    c2(21, 27)
    c2(21, 0, v=-100)
    p["b2"][21] = -1

    # Hint-black at board cells.
    c2(22, 27)
    c2(22, 28, v=-1)
    c2(22, 0, v=-100)
    c2(22, 3, v=8)
    c2(22, 25, v=-8)
    c2(22, 26, v=8)
    p["b2"][22] = -433

    # Full target row/column coverage from the two diagonal 2x2 blocks.
    for dy in (0, -1):
        for dx in (0, -1):
            for raw in (4, 21):
                c2(28, raw, dy, dx)
                c2(23, raw, dy, dx, 64)
                c2(24, raw, dy, dx, 64)

    c2(23, 23)
    c2(23, 24, v=-1)
    p["b2"][23] = -35

    c2(24, 3)
    p["b2"][24] = -64
    c2(25, 3)

    c2(26, 23)
    c2(26, 24, v=-1)
    p["b2"][26] = 29

    # Third layer.
    copies = {
        0: 2, 1: 3, 2: 4, 3: 5,
        7: 9, 8: 6, 9: 7, 10: 8,
        11: 29, 16: 22, 29: 16,
    }
    for out, inp in copies.items():
        c3(out, inp)

    # Bounded player and small-player masks.
    c3(4, 0)
    c3(4, 1, -1)
    c3(5, 10)
    c3(5, 11, -1)
    c3(6, 27)

    # Margin-separated non-player and non-cell masks.
    c3(30, 0, -1)
    c3(30, 1)
    p["b3"][30] = .5
    c3(31, 16, -1)
    p["b3"][31] = .5

    # Axial aiming predicates: target in front on the firing row/column.
    aiming = (
        (12, 64 + 28, 25, 64 + 24, 1),
        (13, 64 + 28, 25, 64 + 24, -1),
        (14, 32 + 28, 26, 32 + 23, 1),
        (15, 32 + 28, 26, 32 + 23, -1),
    )
    for out, alignment, own, target, sign in aiming:
        c3(out, 0, 1000)
        c3(out, 1, -1000)
        c3(out, alignment, 1000)
        c3(out, own, sign)
        c3(out, target, -sign)
        p["b3"][out] = -2000

    # Positive outputs indicate bits already matching a spell.
    def match(out, terms, bias):
        for inp, coefficient in terms:
            c3(out, inp, coefficient)
        p["b3"][out] = bias

    cell, C, R, L, T, F = 16, 17, 18, 19, 20, 21

    # Size-change cross.
    match(17, [(cell, 1), (F, -1), (C, -1), (R, -1)], -.5)
    match(18, [(C, 1), (R, 1), (F, -1)], -1.5)
    match(19, [(F, 1), (C, 1), (R, -1)], -1.5)
    match(20, [(F, 1), (R, 1), (C, -1)], -1.5)

    # Fire column.
    match(21, [(cell, 1), (F, -1), (C, -1)], -.5)
    match(22, [(F, 1), (C, 1)], -1.5)

    # Teleport pattern.
    match(23, [(F, 1), (T, 1), (L, 1)], -2.5)
    match(24, [(F, 1), (T, 1), (C, 1)], -2.5)
    match(25, [(F, 1), (C, 1), (R, 1)], -2.5)
    match(26, [(cell, 1), (F, -1), (T, -1), (R, -1)], -.5)
    match(27, [(R, 1), (C, -1), (F, -1)], -.5)
    match(28, [(T, 1), (L, -1), (C, -1)], -.5)

    # Objective potential.
    diffuse(0)
    p["Uh"][0, 0] = .001
    p["Uh"][0, 1] = 1
    p["Uh"][0, 2] = -1e6

    # Portal-component potential.
    diffuse(1)
    p["Uh"][1, 3] = 1
    p["Uh"][1, 2] = -1e6

    # Growth-area potential within the portal's connected component.
    diffuse(6)
    rec(6, 1)
    p["Uh"][6, 7] = -1e6

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for out, (dy, dx) in enumerate(directions, 2):
        rec(out, 0, dy, dx)
        rec(out, 6, dy, dx, 1e-12)
        p["Uh"][out, 30] = -1e30

    # Shrink when pursuing an ordinary objective or after large teleport.
    # Do not shrink in the return-portal component.
    rec(7, 0)
    rec(7, 1, v=-1e25)
    p["Uh"][7, 11] = 1
    p["Uh"][7, 30] = -1e35
    p["Uh"][7, 5] = -1e35

    # Grow at a fitting position in the portal-connected component.
    rec(8, 1)
    rec(8, 0, v=-1e30)
    p["Uh"][8, 30] = -1e35
    p["Uh"][8, 6] = -1e35
    p["Uh"][8, 7] = -1e35

    # Iteration clock for unreachable-objective tests.
    rec(15, 15)
    p["bh"][15] = 1

    # Large teleport request.
    rec(9, 15)
    rec(9, 0, v=-1e30)
    p["bh"][9] = -48
    p["Uh"][9, 30] = -1e35
    p["Uh"][9, 5] = -1e35
    p["Uh"][9, 11] = -1e35

    # Small teleport request, except on the return-portal component.
    rec(14, 15)
    rec(14, 0, v=-1e30)
    rec(14, 1, v=-1e30)
    p["bh"][14] = -48
    p["Uh"][14, 30] = -1e35
    p["Uh"][14, 6] = -1e35

    for feature in range(12, 16):
        p["Uh"][10, feature] = 1

    # Spell request fields; matching cells absorb their respective field.
    for out in (11, 12, 13):
        diffuse(out)
    rec(11, 7, v=1e20)
    rec(11, 8, v=1e20)
    rec(12, 9, v=1e25)
    rec(12, 14, v=1e25)
    rec(13, 10, v=1e28)

    for out, matches in (
        (11, range(17, 21)),
        (12, range(23, 29)),
        (13, range(21, 23)),
    ):
        for feature in matches:
            p["Uh"][out, feature] = -1e32

    # Global navigation readout.
    for d in range(4):
        zw(d, 2 + d)
        p["Wo"][1 + d, d] = 1

    # Spell absence indicators.
    for unit, feature in ((20, 8), (21, 9), (22, 10)):
        zw(unit, 32 + feature, -1)
        p["bz"][unit] = 1

    for unit, hchannel in ((10, 7), (11, 8), (12, 9), (13, 14)):
        zw(unit, hchannel)
    zw(15, 32 + 16)

    p["Wmz"][0, 10] = 1e30
    p["Wmz"][0, 11] = 1e30
    p["Wmz"][0, 20] = -1e35

    p["Wmz"][1, 12] = 100
    p["Wmz"][1, 13] = 100
    p["Wmz"][1, 21] = -1e35

    p["Wmz"][3, 15] = 100

    for memunit, absence in ((0, 20), (1, 21)):
        p["Wo"][6, 128 + memunit] = 100
        p["Wo"][6, absence] = 100
    p["Wo"][6, 128 + 3] = 10000

    # Bounded aiming indicators, available only with the fire spell.
    for d in range(4):
        for unit in (30 + d, 34 + d):
            zw(unit, 32 + 12 + d)
            zw(unit, 32 + 10, 1000)
            p["bz"][unit] = -1000
        p["bz"][34 + d] -= 1

        p["Wo"][1 + d, 30 + d] = 500
        p["Wo"][1 + d, 34 + d] = -500
        p["Wo"][6, 30 + d] = -1000
        p["Wo"][6, 34 + d] = 1000

    # Persistent facing memory. Spell-grid clicks do not erase it.
    for d in range(4):
        p["Wm"][4 + d, 4 + d] = 10
        for other in range(4):
            p["Wma"][4 + d, 1 + other] = (
                100 if other == d else -100
            )

        p["Wmz"][8 + d, 30 + d] = 100
        p["Wmz"][8 + d, 34 + d] = -100
        p["Wm"][8 + d, 4 + d] = 100
        p["Wmz"][8 + d, 22] = -1e5
        p["bm"][8 + d] = -150
        p["Wo"][6, 128 + 8 + d] = 2000

    p["bo"][6] = 8000
    for action in (0, 5, 7):
        p["bo"][action] = -1e8

    # Click only board-cell anchors. Hint-black has first priority.
    p["Wc"][0, 31] = -1e35
    p["Wc"][0, 16] = 1e36
    for hchannel in (11, 12, 13):
        p["Wc"][0, 32 + hchannel] = 1

    return p
