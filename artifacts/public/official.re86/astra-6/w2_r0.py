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
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}

    ink = tuple(range(6, 15))
    groups = ((8,), (9,), (11,), (14,), (6, 7, 10, 12, 13), (0,))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    neighbours = diagonal + cardinal

    def w1(o, c, value=1.0, dy=0, dx=0):
        p["W1"][o, c, dy + 2, dx + 2] += value

    def w2(o, c, value=1.0, dy=0, dx=0):
        p["W2"][o, c, dy + 1, dx + 1] += value

    # Unit-valued marker detections. The outer-ring penalty rejects
    # detections displaced by one pixel from the actual marker centre.
    for j, group in enumerate(groups):
        for t in range(2):
            o = 2 * j + t
            for colour in group:
                w1(o, colour, 100)
            for dy, dx in neighbours:
                w1(o, 4, 1, dy, dx)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if max(abs(dy), abs(dx)) == 2:
                        w1(o, 4, -10, dy, dx)
            p["b1"][o] = -103 - t

    # Movable ink, excluding marker borders and paint-box interiors.
    for colour in ink:
        w1(12, colour, 20)
    for dy, dx in neighbours:
        w1(12, 4, -20, dy, dx)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            w1(12, 2, -20, dy, dx)
    p["b1"][12] = -19

    # Local signatures at the selected zero-coloured centre.
    for o in (13, 14, 15, 16):
        w1(o, 0, 100)
    for colour in ink:
        for dy, dx in diagonal:
            w1(13, colour, 1, dy, dx)
            w1(14, colour, -20, dy, dx)
        for dy, dx in cardinal:
            w1(13, colour, -20, dy, dx)
            w1(14, colour, 1, dy, dx)
        for dx in (-1, 1):
            w1(15, colour, 1, 0, dx)
        for dy in (-1, 1):
            w1(15, colour, -20, dy, 0)
        for dy, dx in neighbours:
            w1(16, colour, -100, dy, dx)
    p["b1"][13] = -103
    p["b1"][14] = -103
    p["b1"][15] = -101
    p["b1"][16] = -99

    w1(17, 0)
    w1(18, 17)
    w1(19, 16)

    # Saturating selected-colour indicators.
    for colour, a in ((9, 20), (11, 22), (8, 28)):
        for t in range(2):
            o = a + t
            w1(o, 0, 100)
            for dy, dx in neighbours:
                w1(o, colour, 1, dy, dx)
            p["b1"][o] = -100 - t

    # Completely unobscured target markers.
    for colour in (0,) + ink:
        w1(24, colour, 20)
    for dy, dx in neighbours:
        w1(24, 4, 1, dy, dx)
    p["b1"][24] = -27
    w1(27, 2)

    for j in range(6):
        w2(j, 2 * j)
        w2(j, 2 * j + 1, -1)

    w2(6, 12)
    for o, source in ((7, 13), (8, 14), (9, 15),
                      (10, 16), (11, 17)):
        w2(o, source)

    for o, coordinate in ((12, 18), (13, 19)):
        w2(o, 17, 2)
        w2(o, coordinate)
        p["b2"][o] = -2

    # Marker-coordinate features carry a one-unit presence offset.
    def marker_coordinate(o, j, coordinate):
        w2(o, 2 * j, 2)
        w2(o, 2 * j + 1, -2)
        w2(o, coordinate)
        p["b2"][o] = -1

    marker_coordinate(14, 1, 18)
    marker_coordinate(15, 1, 19)
    marker_coordinate(16, 2, 18)
    marker_coordinate(17, 2, 19)
    marker_coordinate(20, 0, 18)
    marker_coordinate(21, 0, 19)

    w2(22, 18)
    w2(23, 19)
    for o, a in ((24, 28), (25, 20), (30, 22)):
        w2(o, a)
        w2(o, a + 1, -1)

    for o, coordinate in ((26, 18), (27, 19)):
        for j in range(6):
            w2(o, 2 * j, 2)
            w2(o, 2 * j + 1, -2)
        w2(o, coordinate)
        p["b2"][o] = -1

    w2(28, 24)
    w2(29, 27)

    p["W3"][:, :32] = np.eye(32, dtype=np.float32)

    p["W3"][4] = 0
    p["W3"][4, :6] = 1
    p["W3"][5] = 0
    p["W3"][5, 28] = 1
    p["W3"][31] = 0
    p["W3"][31, 11] = 1

    # Each axial vote has a base score and a coordinate-tagged score.
    # Their spatial-max difference gives the winning row or column.
    def axial_colour(outputs, marker, mx, my):
        rb, ry, cb, cx = outputs
        for o in outputs:
            p["W3"][o] = 0
            p["W3"][o, marker] = 1000
            p["b3"][o] = -1000

        for o in (rb, ry):
            p["W3"][o, 32 + mx] = 100
            p["W3"][o, mx] = -100
        p["W3"][ry, my] += 1
        p["W3"][ry, marker] -= 1

        for o in (cb, cx):
            p["W3"][o, 64 + my] = 100
            p["W3"][o, my] = -100
        p["W3"][cx, mx] += 1
        p["W3"][cx, marker] -= 1

    axial_colour((0, 1, 2, 3), 0, 20, 21)
    axial_colour((14, 15, 16, 17), 1, 14, 15)
    axial_colour((18, 19, 20, 21), 2, 16, 17)

    # Colour-independent row and column votes.
    for o in (8, 26, 11):
        p["W3"][o] = 0
        p["W3"][o, 32 + 26] = 100
        p["W3"][o, 26] = -100
        p["W3"][o, :6] = 1000
        p["b3"][o] = -1000
    p["W3"][26, 27] += 1
    p["W3"][26, :6] -= 1
    p["W3"][11, 26] += 1
    p["W3"][11, :6] -= 1

    for o in (27, 28):
        p["W3"][o] = 0
        p["W3"][o, 64 + 27] = 100
        p["W3"][o, 27] = -100
        p["W3"][o, :6] = 1000
        p["b3"][o] = -1000
    p["W3"][28, 26] += 1
    p["W3"][28, :6] -= 1

    # Bounded ink diffusion for local improvement trials.
    for o in (0, 1):
        for dy, dx in cardinal:
            p["Wh"][o, 0, dy + 1, dx + 1] = .25
            p["Wh"][o, 1, dy + 1, dx + 1] = -.25
        p["Uh"][o, 6] = 1
    p["bh"][1] = -1
    p["Wh"][2, 0, 1, 1] = 1
    p["Wh"][2, 1, 1, 1] = -1
    p["Uh"][2, 4] = 1
    p["bh"][2] = -1

    # Discounted diagonal Hough votes for X-shaped figures.
    for o, (dy, dx) in zip(range(3, 7), diagonal):
        p["Wh"][o, o, dy + 1, dx + 1] = .90
        p["Uh"][o, 5] = 1

    for source, weight in ((3, 1), (6, 1), (4, -1), (5, -1)):
        p["Wh"][7, source, 1, 1] = weight

    eps = .0001
    for o in (8, 9, 10):
        for source, weight in ((3, 1.2), (6, 1.2),
                               (4, .2), (5, .2), (7, -1)):
            p["Wh"][o, source, 1, 1] = weight
        p["Uh"][o, 4] = -20
        p["Uh"][o, 22] = .02
        p["Uh"][o, 23] = .002
    p["Uh"][9, 22] += eps
    p["Uh"][10, 23] += eps

    # Two-scale annular field for hollow-shape proposals.
    for o, diffusion in ((11, .25), (12, .245)):
        for dy, dx in cardinal:
            p["Wh"][o, o, dy + 1, dx + 1] = diffusion
        p["Uh"][o, 5] = 1
    for o in (13, 14, 15):
        p["Wh"][o, 11, 1, 1] = 1
        p["Wh"][o, 12, 1, 1] = -3
        p["Uh"][o, 22] = .001
        p["Uh"][o, 23] = .002
    p["Uh"][14, 22] += eps
    p["Uh"][15, 23] += eps

    nz = 0

    def zunit(terms, bias=0):
        nonlocal nz
        o = nz
        nz += 1
        for index, weight in terms.items():
            p["Wz"][o, index] += weight
        p["bz"][o] = bias
        return o

    def fm(c):
        return 32 + c

    def fa(c):
        return 64 + c

    score = zunit({
        18: 4096,
        fa(4): .3 * 4096,
        fa(5): -.3 * 4096,
    })
    current_x = zunit({fm(12): 1})
    current_y = zunit({fm(13): 1})
    hollow = zunit({fm(10): 1})
    visible = zunit({fm(31): 1})

    eligible_terms = {
        fm(7): 1, fm(9): 1, fm(10): 1,
        fm(24): 1, fm(25): 1, fm(30): 1,
    }
    eligible_a = zunit(eligible_terms)
    eligible_b = zunit(eligible_terms, -1)

    goal_x, goal_y = [], []

    def gated_goal(xt, yt, gate, bias):
        xt, yt = dict(xt), dict(yt)
        for index, weight in gate.items():
            xt[index] = xt.get(index, 0) + weight
            yt[index] = yt.get(index, 0) + weight
        goal_x.append(zunit(xt, bias))
        goal_y.append(zunit(yt, bias))

    K = 10.0
    gated_goal(
        {9: 1 / eps, 8: -1 / eps},
        {10: 1 / eps, 8: -1 / eps},
        {fm(7): K}, -K,
    )
    gated_goal(
        {14: 1 / eps, 13: -1 / eps},
        {15: 1 / eps, 13: -1 / eps},
        {fm(10): K}, -K,
    )
    gated_goal(
        {fm(11): 1, fm(8): -.995},
        {fm(26): 1, fm(8): -1},
        {fm(9): K}, -K,
    )

    for rb, ry, cb, cx, hint in (
        (0, 1, 2, 3, 24),
        (14, 15, 16, 17, 25),
        (18, 19, 20, 21, 30),
    ):
        gated_goal(
            {fm(cx): 1, fm(cb): -1},
            {fm(ry): 1, fm(rb): -1},
            {fm(hint): K, fm(7): -K, fm(9): -K, fm(10): -K},
            -K,
        )

    INIT, GO, ARRIVE, NEW, HOLLOW = 0, 1, 2, 3, 4
    GX, GY = (5, 6, 7, 8), (9, 10, 11, 12)
    GN = (13, 14, 15, 16)
    Q, U = (17, 18, 19, 20), (21, 22, 23, 24)
    SWITCH, OLD_SCORE, HIDDEN = 25, 26, 27
    OLD_X, OLD_Y = 28, 29
    MOVED = (30, 31, 32, 33)
    STILL, STILL2, STILL3 = 34, 35, 36

    def mz(o, z, value):
        p["Wmz"][o, z] += value

    def mm(o, m, value):
        p["Wm"][o, m] += value

    def flag(o, m, value):
        mm(o, m, value / 2)
        mm(o, INIT, value / 2)

    def ma(o, action, value):
        p["Wma"][o, action] += value

    p["bm"][INIT] = 20
    p["bm"][HOLLOW] = -20
    mz(HOLLOW, hollow, 40)

    p["bm"][NEW] = 20
    mm(NEW, INIT, -40)
    ma(NEW, 5, 40)

    C = 10000.0
    p["bm"][GO] = 20 - 4 * C
    mm(GO, INIT, -40)
    mz(GO, eligible_a, 4 * C)
    mz(GO, eligible_b, -4 * C)
    flag(GO, GO, 40)
    flag(GO, ARRIVE, -C)
    ma(GO, 5, 3 * C)

    p["bm"][ARRIVE] = -20
    flag(ARRIVE, GO, 40)
    for o in GN:
        flag(ARRIVE, o, -100)
    flag(ARRIVE, NEW, -100)
    ma(ARRIVE, 5, -C)

    # Four-unit sample-and-hold, updated only on selection.
    scale = .001
    for slots, sources in ((GX, goal_x), (GY, goal_y)):
        a, b, c, d = slots
        for s, weight in zip(slots, (1, -1, 1, -1)):
            mm(a, s, weight)

        for o in (a, b):
            p["bm"][o] = 20
            mm(o, INIT, -20)
            ma(o, 5, 20)
        for o in (c, d):
            mm(o, INIT, 20)
            ma(o, 5, -20)
        for z in sources:
            mz(c, z, scale)

    nav_gain = 1000.0
    tolerance = 1.7 / 63.0
    directions = (
        (current_y, 1, GY), (current_y, -1, GY),
        (current_x, 1, GX), (current_x, -1, GX),
    )
    for d, (coordinate, sign, slots) in enumerate(directions):
        o = GN[d]
        p["bm"][o] = -nav_gain * tolerance - C
        mz(o, coordinate, sign * nav_gain)
        for s, weight in zip(slots, (1, -1, 1, -1)):
            mm(o, s, -sign * nav_gain * weight / scale)
        flag(o, GO, C)
        ma(o, 5, -C)

    # Directional improvement trials for unmodelled figures and
    # refinement after the hollow-shape proposal.
    mz(OLD_SCORE, score, 1e-5)
    M, gain, threshold = 1e7, 1e9, 12.0

    for d in range(4):
        o = Q[d]
        p["bm"][o] = -M - threshold
        mz(o, score, gain * 1e-5)
        mm(o, OLD_SCORE, -gain)
        flag(o, Q[d], M)
        flag(o, SWITCH if d == 0 else U[d - 1], 2 * M)
        if d == 0:
            p["bm"][o] += 2 * M
            mm(o, INIT, -2 * M)
        force = 2 * M if d == 0 else -4 * M
        flag(o, GO, force)
        flag(o, ARRIVE, force)
        ma(o, 5, force)

        o = U[d]
        p["bm"][o] = -M + threshold
        mz(o, score, -gain * 1e-5)
        mm(o, OLD_SCORE, gain)
        flag(o, Q[d], M)
        flag(o, GO, -4 * M)
        flag(o, ARRIVE, -4 * M)
        ma(o, 5, -4 * M)

    p["bm"][SWITCH] = -20
    flag(SWITCH, U[3], 40)
    flag(SWITCH, GO, -C)

    p["bm"][HIDDEN] = 20
    mz(HIDDEN, visible, -40)

    # Escape repeated blocked navigation.
    mz(OLD_X, current_x, scale)
    mz(OLD_Y, current_y, scale)
    for o, coordinate, old, sign in (
        (30, current_x, OLD_X, 1),
        (31, current_x, OLD_X, -1),
        (32, current_y, OLD_Y, 1),
        (33, current_y, OLD_Y, -1),
    ):
        p["bm"][o] = -10
        mz(o, coordinate, sign * 1e6 * scale)
        mm(o, old, -sign * 1e6)

    p["bm"][STILL] = -20
    flag(STILL, GO, 40)
    for o in MOVED:
        flag(STILL, o, -100)
    ma(STILL, 5, -C)

    p["bm"][STILL2] = -60
    flag(STILL2, STILL, 40)
    flag(STILL2, GO, 40)
    ma(STILL2, 5, -C)

    p["bm"][STILL3] = -100
    flag(STILL3, STILL2, 40)
    flag(STILL3, STILL, 40)
    flag(STILL3, GO, 40)
    ma(STILL3, 5, -C)

    def output_flag(action, memory, weight):
        p["Wo"][action, 128 + memory] += weight / 2
        p["Wo"][action, 128 + INIT] += weight / 2

    reverse = (2, 1, 4, 3)
    for d in range(4):
        output_flag(d + 1, Q[d], 1)
        output_flag(reverse[d], U[d], 1)
        output_flag(d + 1, GN[d], 300 if d < 2 else 400)

    output_flag(5, SWITCH, 200)
    output_flag(5, GO, 100)
    output_flag(5, HOLLOW, -100)
    output_flag(1, NEW, 10000)
    output_flag(5, HIDDEN, 30000)
    output_flag(5, STILL3, 40000)

    p["bo"][[0, 6, 7]] = -100000
    return p
