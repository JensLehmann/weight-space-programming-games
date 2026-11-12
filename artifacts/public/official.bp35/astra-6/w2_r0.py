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

    def c1(o, i, y=0, x=0, v=1.0):
        p["W1"][o, i, y + 2, x + 2] += v

    def box1(o, colors, radius, v=1.0):
        for color in colors:
            for y in range(-radius, radius + 1):
                for x in range(-radius, radius + 1):
                    c1(o, color, y, x, v)

    def c2(o, i, y=0, x=0, v=1.0):
        p["W2"][o, i, y + 1, x + 1] += v

    def box2(o, i, v=1.0):
        for y in (-1, 0, 1):
            for x in (-1, 0, 1):
                c2(o, i, y, x, v)

    def f(o, i, v=1.0):
        p["W3"][o, i] += v

    def h(o, i, y=0, x=0, v=1.0):
        p["Wh"][o, i, y + 1, x + 1] += v

    def z(o, i, v=1.0):
        p["Wz"][o, i] += v

    NOPEN, NSOFT, SU, ONE, SD, UI = range(6)
    E0, E1, E2, PU, PD, ROW, LOW, HIGH = range(6, 14)
    PN, PN1, PT, PB, YELLOW, CLICKABLE = range(14, 20)
    GREEN, GHOST, NGOAL, COL, OPENL, OPENR = range(20, 26)
    EYE_L, EYE_R, PTL, PTR, PBL, PBR = range(26, 32)

    # Remove thin decorative borders from the navigable-space mask.
    p["b1"][NOPEN] = 1.0
    box1(NOPEN, (10, 14, 12, 15, 9, 11, 7), 1, -1.0)
    p["b1"][NSOFT] = 1.0
    box1(NSOFT, (14, 12, 15), 1, -1.0)
    c1(ONE, 18)

    for out, dy in ((SU, -2), (SD, 2)):
        for color in (5, 3, 14, 8, 15, 12):
            for dx in (-1, 0, 1):
                c1(out, color, dy, dx)
        p["b1"][out] = -2.0

    c1(UI, 16, v=63.0)
    p["b1"][UI] = -62.0

    # The second difference of these counts detects isolated spike eyes.
    for out, bias in ((E0, 0.0), (E1, -1.0), (E2, -2.0)):
        box1(out, (0,), 2)
        p["b1"][out] = bias

    # Tile phase obtained from the repeated wall texture.
    c1(PU, 3, 0, -1)
    c1(PU, 3, -2, 1)
    p["b1"][PU] = -1.0
    c1(PD, 3, -1, -2)
    c1(PD, 3, 0, 1)
    p["b1"][PD] = -1.0

    c1(ROW, 16)
    c1(COL, 17)
    c1(LOW, 16, v=-63.0)
    p["b1"][LOW] = 34.0
    c1(HIGH, 16, v=63.0)
    p["b1"][HIGH] = -33.0

    for out, bias in ((PN, 0.0), (PN1, -1.0)):
        box1(out, (9,), 2)
        p["b1"][out] = bias

    # Vertical eye pairs distinguish the player from horizontal spike eyes.
    # Shifted detections extend interaction reach to both neighboring tiles.
    for out, dx in ((EYE_L, 2), (EYE_R, -2)):
        c1(out, 11, 0, dx)
        c1(out, 11, 1, dx)
        p["b1"][out] = -1.0

    def player_point(out, bottom=False, shift=0):
        for dy in ((0, -1, -2) if bottom else (0, 1, 2)):
            c1(out, 9, dy, shift)
        c1(out, 5, 1 if bottom else -1, shift)
        p["b1"][out] = -3.0

    player_point(PT)
    player_point(PB, True)
    player_point(PTL, shift=2)
    player_point(PTR, shift=-2)
    player_point(PBL, True, 2)
    player_point(PBR, True, -2)

    for out, dx in ((OPENL, -2), (OPENR, 2)):
        c1(out, 10, 0, dx)
        c1(out, 7, 0, dx)

    c1(YELLOW, 15)
    for dy in (-2, 2):
        for dx in (-2, 2):
            c1(YELLOW, 3, dy, dx)
    p["b1"][YELLOW] = -4.0

    for color in (14, 15, 12):
        c1(CLICKABLE, color)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        c1(CLICKABLE, 10, dy, dx, -1.0)

    c1(GREEN, 8)
    for dy, dx in ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        c1(GHOST, 12, dy, dx)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        c1(GHOST, 10, dy, dx)
    p["b1"][GHOST] = -8.0

    p["b1"][NGOAL] = 1.0
    box1(NGOAL, (7,), 2, -1.0)

    M = 256.0
    K = 65536.0
    B = 8192.0
    OBSTACLE = 4096.0

    c2(0, NOPEN)
    c2(0, UI, v=10.0)
    for feature, coefficient in ((E0, 1.0), (E1, -2.0), (E2, 1.0)):
        box2(0, feature, coefficient)

    c2(1, NSOFT, v=-1.0)
    p["b2"][1] = 1.0
    for out, support in ((2, SU), (3, SD)):
        c2(out, support, v=-1.0)
        p["b2"][out] = 1.0

    # Static values for supported landing positions and visible goals.
    for out, down, dy, dx in (
        (4, False, 0, 0), (5, False, 0, -1),
        (6, False, 0, 1), (7, False, -1, 0),
        (8, True, 0, 0), (9, True, 0, -1),
        (10, True, 0, 1), (11, True, 1, 0),
    ):
        c2(out, ONE, dy, dx, (36.0 if down else 100.0) + 1024.0)
        c2(out, ROW, dy, dx, 63.0 if down else -63.0)
        c2(out, SD if down else SU, dy, dx, M)
        c2(out, NGOAL, dy, dx, -1024.0)
        c2(out, NOPEN, dy, dx, -K)
        c2(out, UI, dy, dx, -K)
        for feature, coefficient in ((E0, -K), (E1, 2.0 * K), (E2, -K)):
            c2(out, feature, dy, dx, coefficient)
        p["b2"][out] = -M

    c2(12, PU)
    c2(13, PU, -1)
    c2(14, PD)
    c2(15, PD, 1)

    for out, gate, excess in (
        (16, LOW, False), (17, LOW, True),
        (18, HIGH, False), (19, HIGH, True),
    ):
        box2(out, PN)
        box2(out, PN1, -1.0)
        box2(out, EYE_L)
        box2(out, EYE_R)
        c2(out, gate, v=-K)
        p["b2"][out] = -1.0 if excess else 0.0

    c2(20, CLICKABLE)
    for feature, coefficient in ((E0, -100.0), (E1, 200.0), (E2, -100.0)):
        box2(20, feature, coefficient)
    c2(21, GREEN)
    c2(22, GHOST)

    c2(23, YELLOW)
    c2(23, COL, v=0.001)
    p["b2"][23] = -1.0

    # Localized movement probes, augmented by destination legality.
    for out, player, py, px, openness, ox in (
        (24, PTL, 0, 1, OPENL, -1),
        (25, PTR, 0, -1, OPENR, 1),
        (27, PBL, -1, 1, OPENL, -1),
        (28, PBR, -1, -1, OPENR, 1),
    ):
        c2(out, player, py, px, 2.0)
        c2(out, openness, 0, ox)
        p["b2"][out] = -1.0

    c2(26, PT)
    c2(29, PB, -1)
    c2(30, PT)
    c2(30, COL, v=0.001)
    p["b2"][30] = -1.0
    c2(31, PT)
    c2(31, LOW, v=-K)

    BLOCK, COST, NGU, NGD = 0, 1, 2, 3
    U0, USL, USR, USV = 4, 5, 6, 7
    D0, DSL, DSR, DSV = 8, 9, 10, 11
    UL, UR, UC, DL, DR, DC = 12, 13, 14, 15, 16, 17
    CU, CD, GR, GP, YY, UP, PX = 18, 19, 20, 21, 22, 23, 24
    VUL, VUR, VDL, VDR = 25, 26, 27, 28

    f(BLOCK, 0)
    f(COST, 1)
    for out, unsupported, phase in ((NGU, 2, 12), (NGD, 3, 14)):
        f(out, unsupported)
        f(out, 32 + phase, -1.0)
        p["b3"][out] = 1.0

    for out, phase in (
        (U0, 12), (USL, 12), (USR, 12), (USV, 13),
        (D0, 14), (DSL, 14), (DSR, 14), (DSV, 15),
    ):
        f(out, out)
        f(out, 32 + phase, K)
        p["b3"][out] = -K

    for out, feature in (
        (UL, 24), (UR, 25), (UC, 26),
        (DL, 27), (DR, 28), (DC, 29),
        (GR, 21), (YY, 23), (UP, 31),
    ):
        f(out, feature)
    for out, feature in ((VUL, 24), (VUR, 25), (VDL, 27), (VDR, 28)):
        f(out, feature)
        p["b3"][out] = -1.0

    for out, raw, excess in ((CU, 16, 17), (CD, 18, 19)):
        f(out, raw)
        f(out, excess, -1.0)
        f(out, 20)
        p["b3"][out] = -1.0

    f(GP, 22)
    f(GP, 64 + 30, 1000.0)
    p["b3"][GP] = -1.0
    f(PX, 30, 1000.0)

    # ReLU implementation of directional max-plus value propagation.
    for base, vertical, nonground, sources in (
        (0, -1, NGU, (U0, USL, USR, USV)),
        (5, 1, NGD, (D0, DSL, DSR, DSV)),
    ):
        L, R, D, Q, V = [base + i for i in range(5)]
        current, left, right, vert = sources

        h(L, V, x=-1)
        h(L, V, x=1, v=-1.0)
        p["Uh"][L, left] = 1.0
        p["Uh"][L, right] = -1.0
        p["Uh"][L, nonground] = -K

        h(R, V, x=1)
        p["Uh"][R, right] = 1.0
        p["Uh"][R, nonground] = -K
        h(D, L)
        h(D, R)

        h(Q, V, y=vertical)
        h(Q, L, v=-1.0)
        h(Q, R, v=-1.0)
        p["Uh"][Q, vert] = 1.0

        h(V, D)
        h(V, Q)
        p["Uh"][V, current] = -1.0
        p["Uh"][V, BLOCK] = -K
        p["Uh"][V, COST] = -0.35
        p["bh"][V] = -0.10

    # Pack left and center probes: max gives left; sum minus max gives center.
    for out, value, source, marker, valid, center in (
        (10, 4, USL, UL, VUL, UC),
        (12, 9, DSL, DL, VDL, DC),
    ):
        h(out, value, x=-1)
        p["Uh"][out, source] = 1.0
        p["Uh"][out, marker] = K + B - OBSTACLE
        p["Uh"][out, valid] = -(K + B) + 2.0 * OBSTACLE
        p["Uh"][out, center] = K
        p["bh"][out] = -K

    for out, value, source, marker, valid in (
        (11, 4, USR, UR, VUR), (13, 9, DSR, DR, VDR)
    ):
        h(out, value, x=1)
        p["Uh"][out, source] = 1.0
        p["Uh"][out, marker] = K + B - OBSTACLE
        p["Uh"][out, valid] = -(K + B) + 2.0 * OBSTACLE
        p["bh"][out] = -K

    for out, value, source, candidate, baseline in (
        (14, 4, U0, CU, 63.0), (15, 9, D0, CD, 67.0)
    ):
        h(out, value)
        p["Uh"][out, source] = 1.0
        p["Uh"][out, candidate] = K
        p["bh"][out] = -K - baseline + 0.30

    for unit, probe, upward, baseline in (
        (0, 10, True, 63.0), (1, 11, True, 63.0),
        (2, 12, False, 67.0), (3, 13, False, 67.0),
    ):
        z(unit, probe)
        z(unit, 32 + UP, K if upward else -K)
        p["bz"][unit] = -B - baseline - (K if upward else 0.0)

    z(4, 14)
    z(4, 32 + UP, K)
    p["bz"][4] = -K
    z(5, 15)
    z(5, 32 + UP, -K)

    def center_sample(unit, probe):
        z(unit, 16 + probe, 4096.0)
        z(unit, probe, -1.0)

    center_sample(6, 12)
    z(6, 32 + UP, K)
    z(6, 32 + GR, K)
    p["bz"][6] = -76.75 - 2.0 * K

    center_sample(7, 10)
    z(7, 32 + UP, -K)
    z(7, 32 + GR, K)
    p["bz"][7] = -72.75 - K

    z(8, 32 + UP)
    z(9, 32 + PX)

    for unit in (10, 11):
        z(unit, 32 + YY, 100000.0)
    p["bz"][11] = -1.0

    for unit in (12, 13):
        for feature, coefficient in (
            (CU, 1.0), (CD, 1.0), (GR, 1.0),
            (YY, 100000.0), (GP, 100.0),
        ):
            z(unit, 32 + feature, coefficient)
    p["bz"][13] = -1.0

    for unit, probe, current_up in (
        (15, 12, True), (16, 13, True),
        (17, 10, False), (18, 11, False),
    ):
        z(unit, probe)
        z(unit, 32 + GR, K)
        z(unit, 32 + UP, K if current_up else -K)
        baseline = 77.0 if current_up else 73.0
        p["bz"][unit] = -B - baseline - K - (K if current_up else 0.0)

    for unit, feature, current_up in (
        (20, VUL, True), (21, VUR, True),
        (22, VDL, False), (23, VDR, False),
    ):
        z(unit, 32 + feature)
        z(unit, 32 + UP, 1.0 if current_up else -1.0)
        p["bz"][unit] = -1.0 if current_up else 0.0

    # Saturated presence of a ghost platform in the player's column.
    z(24, 32 + GP, 10.0)
    z(25, 32 + GP, 10.0)
    p["bz"][25] = -1.0

    p["bo"][:] = -100000.0
    p["bo"][3] = 0.0
    p["bo"][4] = 0.001
    p["bo"][6] = -100.0

    for unit in (0, 2, 15, 17):
        p["Wo"][3, unit] = 20.0
    for unit in (1, 3, 16, 18):
        p["Wo"][4, unit] = 20.0
    for unit in (20, 22):
        p["Wo"][3, unit] = 0.08
    for unit in (21, 23):
        p["Wo"][4, unit] = 0.08
    for unit in (4, 5, 6, 7):
        p["Wo"][6, unit] = 20.0

    p["Wo"][6, 10] = 0.4
    p["Wo"][6, 11] = -0.4
    p["Wo"][6, 12] = 99.95
    p["Wo"][6, 13] = -99.95
    p["Wo"][6, 24] = 0.30
    p["Wo"][6, 25] = -0.30

    # Memory only breaks nearly equal movement scores.
    p["Wm"][7, 7] = 2.0
    p["Wma"][7, 4] = 3.0
    p["Wma"][7, 3] = -3.0
    p["Wo"][3, 128 + 7] = -0.01
    p["Wo"][4, 128 + 7] = 0.01

    p["Wc"][0, 32 + 14] = 1.0
    p["Wc"][0, 32 + 15] = 1.0
    p["Wc"][0, CU] = 0.0001
    p["Wc"][0, CD] = 0.0001
    p["Wc"][0, GR] = 0.05
    p["Wc"][0, GP] = 1.0
    p["Wc"][0, YY] = 20.0

    return p
