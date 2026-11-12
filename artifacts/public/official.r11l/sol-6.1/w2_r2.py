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
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}

    w1, b1 = p["W1"], p["b1"]
    w2, b2 = p["W2"], p["b2"]
    w3 = p["W3"]
    wh, uh, bh = p["Wh"], p["Uh"], p["bh"]

    diamond = [(y, x) for y in range(-2, 3)
               for x in range(-2, 3) if abs(y) + abs(x) <= 2]
    shell = [(y, x) for y, x in diamond if y or x]
    disc = [(y, x) for y in range(-2, 3)
            for x in range(-2, 3)
            if not (abs(y) == 2 and abs(x) == 2)]

    def a1(out, color, y=0, x=0, value=1.0):
        w1[out, color, y + 2, x + 2] += value

    def a2(out, source, value=1.0, y=0, x=0):
        w2[out, source, y + 1, x + 1] += value

    def local(out, source, value=1.0):
        w3[out, source] += value

    def row(out, source, value=1.0):
        w3[out, 32 + source] += value

    def col(out, source, value=1.0):
        w3[out, 64 + source] += value

    def h(out, source, value=1.0, y=0, x=0):
        wh[out, source, y + 1, x + 1] += value

    # All anchor centres.
    for y, x in shell:
        a1(0, 0, y, x)
        a1(0, 3, y, x)
    for color in (4, 14, 15):
        a1(0, color)
    b1[0] = -12.0

    # Selected purple / green anchor centres.
    for out, color in ((1, 15), (2, 14)):
        for y, x in shell:
            a1(out, 0, y, x)
        a1(out, color)
        b1[out] = -12.0

    # Purple / green balls.
    for out, color in ((3, 15), (4, 14)):
        a1(out, 6)
        for y, x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a1(out, color, y, x)
        b1[out] = -4.0

    # Purple / green ring centres.
    for out, color in ((5, 15), (6, 14)):
        for y in (-2, 2):
            for x in (-2, 2):
                a1(out, color, y, x)
        b1[out] = -3.0

    # Anchor-wall collision field.
    for y, x in diamond:
        a1(7, 2, y, x)

    # Ball-hazard collision field.
    for y, x in disc:
        a1(8, 10, y, x)

    # Packed coordinate: (64*row + column)/4095.
    a1(9, 16, value=64.0 / 65.0)
    a1(9, 17, value=1.0 / 65.0)

    # Anchor bounding-box exclusion.
    for y in range(-2, 3):
        for x in range(-2, 3):
            a1(10, 0, y, x, 4.0)
            a1(10, 3, y, x, 4.0)
            for color in (4, 14, 15):
                a1(10, color, y, x)
    b1[10] = -20.0

    # Screen-margin exclusion.
    b1[11] = float(len(diamond))
    for y, x in diamond:
        a1(11, 18, y, x, -1.0)

    # Budget-edge triangle clock, with a peak every six display rows.
    # Outside the single budget-edge pixel these units remain zero.
    cuts = list(range(0, 60, 3))
    coefficients = []
    for j, cut in enumerate(cuts):
        out = 12 + j
        a1(out, 0, value=64.0)
        a1(out, 5, y=-1, value=64.0)
        a1(out, 16, value=63.0)
        a1(out, 17, value=-256.0)
        b1[out] = -128.0 - cut
        coefficients.append(
            1.0 if j == 0 else (-2.0 if j % 2 else 2.0)
        )

    # W2 local masks and sampled coordinates.
    for out, source in ((0, 0), (1, 1), (2, 2), (3, 3),
                        (4, 4), (5, 5), (6, 6), (7, 9),
                        (11, 7), (12, 11), (13, 10)):
        a2(out, source)

    # Coordinate at selected anchor.
    a2(8, 9)
    a2(8, 1, 2.0)
    a2(8, 2, 2.0)
    b2[8] = -2.0

    # Coordinate at any anchor, used to cancel coordinate tie-breaking
    # during reselection.
    a2(9, 9)
    a2(9, 0, 2.0)
    b2[9] = -2.0

    # Safe ball-centre pixels. Walls constrain anchors, not balls.
    a2(10, 8, -1.0)
    a2(10, 11, -1.0)
    b2[10] = 1.0

    # Two threshold units whose difference clips the clock to [0, 1].
    for j, coefficient in enumerate(coefficients):
        a2(14, 12 + j, coefficient)
        a2(15, 12 + j, coefficient)
    b2[14] = -2.0
    b2[15] = -3.0

    # F3 meanings:
    # 0 all anchors; 1 selected anchor; 2 selected family row-source;
    # 3/4 balls; 5/6 goals; 7 click coordinate; 8 selected coordinate;
    # 9 anchor coordinate; 10 ball-safe mask; 11 wall exclusion;
    # 12 margin exclusion; 13 occupied-box exclusion; 14 clock column.
    local(0, 0)
    local(1, 1)
    local(1, 2)
    row(2, 1, 2.0)  # Purple code 2.
    row(2, 2, 1.0)  # Green code 1.
    for out, source in ((3, 3), (4, 4), (5, 5), (6, 6),
                        (7, 7), (8, 8), (9, 9), (10, 10),
                        (11, 11), (12, 12), (13, 13)):
        local(out, source)
    col(14, 14)
    col(14, 15, -1.0)

    # h0/h1: vertical family-code prefix/suffix.
    h(0, 0, y=-1)
    h(1, 1, y=1)
    uh[0, 2] = 1.0
    uh[1, 2] = 1.0

    # Independent clipped goal waves:
    # purple active = h2-h3; green active = h5-h6.
    neighbours = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    for positive, negative, goal in ((2, 3, 5), (5, 6, 6)):
        for out in (positive, negative):
            for y, x in neighbours:
                h(out, positive, 1.0, y, x)
                h(out, negative, -1.0, y, x)
            uh[out, goal] = 1.0
            uh[out, 10] = 5.0
            bh[out] = -5.0
        bh[negative] -= 1.0

    # Arrival-time costs. Once reached by a wave, a cell's cost stops
    # increasing. Unreachable cells have cost 128.
    for out, positive, negative in ((4, 2, 3), (7, 5, 6)):
        h(out, out)
        h(out, positive, -1.0)
        h(out, negative, 1.0)
        bh[out] = 1.0

    # h8/h9: positive packed-coordinate corrections at the chosen ball.
    # A large local ball gate prevents global-code transients from
    # creating sources elsewhere in the image.
    baseline = 16.0
    displacement = 8.0
    for out, ball, cost, purple in (
        (8, 3, 4, True),
        (9, 4, 7, False),
    ):
        uh[out, ball] = 1000.0
        cx = displacement / 4095.0
        cy = displacement * 64.0 / 4095.0
        h(out, cost, cx, x=-1)
        h(out, cost, -cx, x=1)
        h(out, cost, cy, y=-1)
        h(out, cost, -cy, y=1)

        sign = 1.0 if purple else -1.0
        h(out, 0, sign * 200.0)
        h(out, 1, sign * 200.0)
        uh[out, 2] = -sign * 200.0

        # Purple: require code 2. Green: require code 1.
        bh[out] = baseline - (1400.0 if purple else 800.0)

    # h10/h11: horizontal prefix/suffix of the coordinate source.
    for out, x in ((10, -1), (11, 1)):
        h(out, out, x=x)
        h(out, 8)
        h(out, 9)
        uh[out, 8] = 1.0

    # h12/h13: vertical prefix/suffix of the complete horizontal sum.
    for out, y in ((12, -1), (13, 1)):
        h(out, out, y=y)
        h(out, 10)
        h(out, 11)
        h(out, 8, -1.0)
        h(out, 9, -1.0)
        uh[out, 8] = -1.0

    # h15: horizontal broadcast of the bounded reselection clock.
    h(15, 15, x=-1)
    uh[15, 14] = 1.0

    # Full coordinate sum:
    # h12+h13-h10-h11+h8+h9+selected_coordinate.
    # h14 is its positive click-coordinate error, with a gated
    # anchor-selection alternative.
    scale = 10000.0
    uh[14, 7] = scale
    uh[14, 8] = -scale
    for source, coefficient in (
        (12, -1.0), (13, -1.0),
        (10, 1.0), (11, 1.0),
        (8, -1.0), (9, -1.0),
    ):
        h(14, source, scale * coefficient)
    bh[14] = scale * baseline

    h(14, 15, 1.0e7)
    uh[14, 0] = -1.002e7
    uh[14, 1] = 4.0e7

    # Readout: packed-coordinate L1 minimisation during movement;
    # far-from-goal anchor preference during reselection.
    click = p["Wc"][0]
    click[7] = scale
    click[9] = -scale
    click[0] = -100000.0
    click[1] = -100000.0
    click[11] = -1.0e9
    click[12] = -1.0e9
    click[13] = -10000.0
    click[32 + 14] = -2.0
    click[32 + 4] = 1.0
    click[32 + 7] = 1.0

    p["bo"][:] = -10.0
    p["bo"][6] = 10.0
    return p
