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

    colours = [6, 8, 9, 11, 12, 14, 15]
    ci = {c: i for i, c in enumerate(colours)}
    function_colours = [8, 9, 14]

    def w1(o, i, v, dy=0, dx=0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def w2(o, i, v, dy=0, dx=0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def wh(o, i, v, dy=0, dx=0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    def local(o, i, v=1.0):
        p["W3"][o, i] += v

    def row(o, i, v=1.0):
        p["W3"][o, 32 + i] += v

    def col(o, i, v=1.0):
        p["W3"][o, 64 + i] += v

    # First layer: token perimeters, token interiors, horizontal bars.
    for c, i in ci.items():
        for dy in range(-1, 3):
            for dx in range(-1, 3):
                if dy in (-1, 2) or dx in (-1, 2):
                    w1(i, c, 1, dy, dx)
        p["b1"][i] = -11

        w1(7 + i, c, 1)

        for dx in range(-2, 3):
            w1(14 + i, c, 1, -2, dx)
        p["b1"][14 + i] = -4

    EMPTY1, SELECT1 = 21, 22
    X1, Y1 = 23, 24
    LOW1, UPPER1, BOTTOM1 = 25, 26, 27

    # Empty-slot sample is above the two-by-two central marker.
    for dy in (1, 2):
        for dx in (0, 1):
            w1(EMPTY1, 2, 1, dy, dx)
    p["b1"][EMPTY1] = -3

    outline = {(-2, dx) for dx in range(-2, 3)}
    outline |= {(dy, -2) for dy in range(-1, 3)}
    for dy, dx in outline:
        w1(SELECT1, 0, 1, dy, dx)
    p["b1"][SELECT1] = -8

    w1(X1, 17, 1)
    w1(Y1, 16, 63)

    w1(LOW1, 16, -63)
    p["b1"][LOW1] = 3

    w1(UPPER1, 16, -63)
    p["b1"][UPPER1] = 54

    w1(BOTTOM1, 16, 63)
    p["b1"][BOTTOM1] = -53

    # Piecewise-linear decreasing horizontal priority.
    knots = [0, 16, 24, 32, 40, 64]
    values = [(1.0 - k / 64.0) ** 4 for k in knots]
    slopes = [
        63.0 * (values[j + 1] - values[j])
        / (knots[j + 1] - knots[j])
        for j in range(len(knots) - 1)
    ]
    for j, k in enumerate(knots[1:-1]):
        w1(28 + j, 17, 1)
        p["b1"][28 + j] = -k / 63.0

    def add_order(o, scale=1.0):
        p["b2"][o] += scale
        w2(o, X1, scale * slopes[0])
        for j in range(4):
            w2(o, 28 + j, scale * (slopes[j + 1] - slopes[j]))

    # Second-layer feature allocation.
    FRAME = 0
    REF = 3
    SOURCE = 6
    TARGET = 13
    EMPTY2, SELECT2 = 20, 21
    X2, Y2 = 22, 23
    SELCOLOUR = 24
    RINGRANK = 27
    FRAMELEFT, FRAMERIGHT = 28, 29
    ORDER2, ANYSELRING = 30, 31

    for j, c in enumerate(function_colours):
        i = ci[c]

        # Seven-wide bar: a function frame, not a target glyph.
        for dx in (-1, 0, 1):
            w2(FRAME + j, 14 + i, 1, -1, dx)
        p["b2"][FRAME + j] = -2

        # Board call token, carrying its horizontal priority.
        w2(REF + j, i, 32, 1, 0)
        w2(REF + j, 7 + i, -32, 1, 0)
        w2(REF + j, BOTTOM1, -32)
        p["b2"][REF + j] = -32
        add_order(REF + j)

        # Selected token of this colour, solid or hollow.
        w2(SELCOLOUR + j, i, 1, 1, 0)
        w2(SELCOLOUR + j, SELECT1, 1, 1, 0)
        p["b2"][SELCOLOUR + j] = -1

    for c, i in ci.items():
        w2(SOURCE + i, i, 1, 1, 0)
        w2(SOURCE + i, UPPER1, -32)

        # Earliest target occurrence of each colour.
        w2(TARGET + i, 14 + i, 2)
        w2(TARGET + i, Y1, -2)
        w2(TARGET + i, X1, -1)
        w2(TARGET + i, LOW1, -10)
        p["b2"][TARGET + i] = 5

        w2(ANYSELRING, i, 1, 1, 0)
        w2(ANYSELRING, 7 + i, -1, 1, 0)

        # Place calls before solids.
        rank = 3 if c == 9 else (2 if c == 14 else 1)
        w2(RINGRANK, i, rank, 1, 0)
        w2(RINGRANK, 7 + i, -rank, 1, 0)

        for dx in (-1, 0, 1):
            w2(FRAMELEFT, 14 + i, 32, -1, dx)
            w2(FRAMERIGHT, 14 + i, 32, -1, dx)

    w2(ANYSELRING, SELECT1, 1, 1, 0)
    p["b2"][ANYSELRING] = -1
    w2(RINGRANK, UPPER1, -64)

    p["b2"][FRAMELEFT] = -95
    w2(FRAMELEFT, X1, -1)
    p["b2"][FRAMERIGHT] = -96
    w2(FRAMERIGHT, X1, 1)

    w2(EMPTY2, EMPTY1, 1)
    w2(EMPTY2, BOTTOM1, -32)
    w2(SELECT2, SELECT1, 1, 1, 0)
    w2(X2, X1, 1)
    w2(Y2, Y1, 1)
    add_order(ORDER2)

    # Axial feature allocation.
    FTARGET = 0
    ROUTE = 7
    AFFINITY = 10
    FSOURCE = 13
    FEMPTY, ROOTORDER = 20, 21
    FX, FSELECT, FRING = 22, 23, 24
    BOTTOM_A, BOTTOM_B = 25, 26
    DIST_R, DIST_L = 27, 28
    FRANK, FY, OCCUPIED = 29, 30, 31

    for i in range(7):
        row(FTARGET + i, TARGET + i)
        local(FSOURCE + i, SOURCE + i)

    for j in range(3):
        # Reference column crossed with matching function row.
        col(ROUTE + j, REF + j)
        row(ROUTE + j, FRAME + j, 32)
        p["b3"][ROUTE + j] = -32

        # Selected-colour affinity, strongest for short routines.
        # The extrema encode the horizontal span of frame interiors.
        col(AFFINITY + j, SELCOLOUR + j, 32)
        row(AFFINITY + j, FRAME + j, 32)
        row(AFFINITY + j, FRAMELEFT, -12.6)
        row(AFFINITY + j, FRAMERIGHT, -12.6)
        p["b3"][AFFINITY + j] = -47.9

        row(OCCUPIED, REF + j)

    local(FEMPTY, EMPTY2)

    # Compensate the root row for the stronger local ordering term.
    local_order_slope = 0.003 * 63.0
    local(ROOTORDER, ORDER2)
    local(ROOTORDER, X2, local_order_slope)
    local(ROOTORDER, EMPTY2, 2)
    row(ROOTORDER, FRAME + function_colours.index(8), 2)
    p["b3"][ROOTORDER] = -4

    local(FX, X2)
    col(FSELECT, SELECT2)
    col(FRING, ANYSELRING)

    local(BOTTOM_A, Y2)
    p["b3"][BOTTOM_A] = -56
    local(BOTTOM_B, Y2)
    p["b3"][BOTTOM_B] = -57

    centre_bias = 0.6 / 63.0
    local(DIST_R, X2)
    row(DIST_R, FRAMERIGHT, -0.5)
    row(DIST_R, FRAMELEFT, 0.5)
    p["b3"][DIST_R] = -0.5 + centre_bias

    local(DIST_L, X2, -1)
    row(DIST_L, FRAMERIGHT, 0.5)
    row(DIST_L, FRAMELEFT, -0.5)
    p["b3"][DIST_L] = 0.5 - centre_bias

    local(FRANK, RINGRANK)
    local(FY, Y2, 1.0 / 63.0)

    # Recurrent channels:
    # 0..6: colour priorities transported to inventory tokens
    # 7..8: horizontal route and affinity broadcasts
    # 9..10: selection broadcast
    # 11..12: selected-call broadcast
    # 13..15: inventory / solid destination / call destination scores
    G = 1024.0

    for i in range(7):
        wh(i, i, 1, -1, 0)
        p["Uh"][i, FTARGET + i] = 1
        p["Uh"][i, FSOURCE + i] = G
        p["Uh"][i, BOTTOM_A] = -G
        p["Uh"][i, BOTTOM_B] = G

    routing_features = list(range(ROUTE, ROUTE + 3))
    routing_features += list(range(AFFINITY, AFFINITY + 3))

    for h, dx in ((7, 1), (8, -1)):
        wh(h, h, 1, 0, dx)
        for f in routing_features:
            p["Uh"][h, f] = 1

    for h, dx in ((9, 1), (10, -1)):
        wh(h, h, 1, 0, dx)
        p["Uh"][h, FSELECT] = 1

    for h, dx in ((11, 1), (12, -1)):
        wh(h, h, 1, 0, dx)
        p["Uh"][h, FRING] = 1

    # Choose inventory token while no selection is active.
    for i in range(7):
        wh(13, i, 1)
        p["Uh"][13, FSOURCE + i] = G
    for h in (9, 10):
        wh(13, h, -100)
    p["Uh"][13, FSELECT] = 100
    p["Uh"][13, FRANK] = 10
    p["bh"][13] = 50 - G

    # Solid destination: expansion priority plus colour affinity.
    for h in (7, 8):
        wh(14, h, 1)
    for f in routing_features:
        p["Uh"][14, f] = -1
    for h in (9, 10):
        wh(14, h, 50)
    p["Uh"][14, FSELECT] = -50
    for h in (11, 12):
        wh(14, h, -100)
    p["Uh"][14, FRING] = 100
    p["Uh"][14, ROOTORDER] = 1
    p["Uh"][14, FX] = -local_order_slope
    p["Uh"][14, FEMPTY] = G
    p["bh"][14] = -G

    # Call destination: centred, avoiding self-calls and occupied rows.
    for h in (11, 12):
        wh(15, h, 50)
    p["Uh"][15, FRING] = -50
    for h in (7, 8):
        wh(15, h, -10)
    for f in routing_features:
        p["Uh"][15, f] = 10
    p["Uh"][15, OCCUPIED] = -20
    p["Uh"][15, DIST_R] = -1
    p["Uh"][15, DIST_L] = -1
    p["Uh"][15, FY] = -0.05
    p["Uh"][15, FEMPTY] = G
    p["bh"][15] = -G

    for h in (13, 14, 15):
        p["Wc"][0, 32 + h] = 1

    # Execute when no board slots remain empty.
    p["Wz"][0, 32 + FEMPTY] = 1
    p["bo"][:] = -10
    p["bo"][6] = 0
    p["bo"][5] = 1
    p["Wo"][5, 0] = -2

    # An unsuccessful execution returns control with ACTION5 previous.
    p["Wma"][0, 5] = 1
    p["bo"][0] = -2
    p["Wo"][0, 128] = 5

    return p
