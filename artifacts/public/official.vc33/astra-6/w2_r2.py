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
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    W1, W2, W3 = p["W1"], p["W2"], p["W3"]
    b1, b2, b3 = p["b1"], p["b2"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def c1(o, i, dy=0, dx=0, v=1.0):
        W1[o, i, 2 + dy, 2 + dx] += v

    def c2(o, i, dy=0, dx=0, v=1.0):
        W2[o, i, 1 + dy, 1 + dx] += v

    def r(o, i, dy=0, dx=0, v=1.0):
        Wh[o, i, 1 + dy, 1 + dx] += v

    def gate(o, dy=0, dx=0, v=1.0):
        c1(o, 1, dy, dx, v)
        c1(o, 12, dy, dx, v)

    # Local appearance.
    for out, color in enumerate((0, 3, 5, 9)):
        c1(out, color)
    gate(4)
    c1(5, 11)
    c1(5, 14)

    # Goal-colored pixels have no nearby yellow player pixels.
    c1(6, 11)
    c1(6, 14)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(6, 4, dy, dx, -1.0)

    # Gravity-facing player edges: down, right, left.
    player_dirs = [(1, 0), (0, 1), (0, -1)]
    for out, (dy, dx) in zip((7, 8, 9), player_dirs):
        c1(out, 11)
        c1(out, 14)
        c1(out, 4, -2 * dy, -2 * dx)
        c1(out, 11, dy, dx, -1.0)
        c1(out, 14, dy, dx, -1.0)
        b1[out] = -1.0

    # Long gate axes and passage endpoints.
    for dy in (-2, 0, 2):
        gate(10, dy, 0)
    b1[10] = -2.0
    for dx in (-2, 0, 2):
        gate(11, 0, dx)
    b1[11] = -2.0
    gate(12)
    gate(12, 1, 0, -1.0)
    gate(13)
    gate(13, 0, -1, -1.0)
    c1(14, 12)

    # Wall pixels adjacent to pump buttons.
    for d, (dy, dx) in enumerate(dirs):
        c1(15 + d, 5)
        c1(15 + d, 9, 2 * dy, 2 * dx)
        b1[15 + d] = -1.0

    # Shifted button/air endpoints. These also provide the local
    # Green-function correction used by the donor-availability field.
    for j, shift in enumerate((-1, 0, 1)):
        out = 19 + j
        c1(out, 9, 0, shift, 2.0)
        c1(out, 3, 0, shift - 1)
        c1(out, 3, 0, shift + 1)
        b1[out] = -2.0

        out = 22 + j
        c1(out, 9, shift, 0, 2.0)
        c1(out, 3, shift - 1, 0)
        c1(out, 3, shift + 1, 0)
        b1[out] = -2.0

    # Only button interiors may be clicked.
    c1(25, 9)
    for dy, dx in dirs:
        c1(25, 9, dy, dx)
    b1[25] = -4.0

    # Second layer.
    for out, inp in enumerate((0, 1, 2, 3, 4, 6)):
        c2(out, inp)
    for out, inp in zip((6, 7, 8), (7, 8, 9)):
        c2(out, inp)

    c2(9, 18, 0, -1)
    c2(9, 17, 0, 1)
    b2[9] = -1.0
    c2(10, 16, -1, 0)
    c2(10, 15, 1, 0)
    b2[10] = -1.0

    for out, inp in zip((11, 12, 13, 14, 15),
                        (10, 11, 12, 13, 14)):
        c2(out, inp)

    c2(16, 20)
    c2(17, 23)

    # Upper bound on the endpoint source's own contribution inside
    # its button: 1.25 * (1/2)**distance. Clickable interiors are at
    # most two pixels from each endpoint.
    c2(18, 20, v=1.25)
    c2(18, 23, v=1.25)
    for inp in (19, 21, 22, 24):
        c2(18, inp, v=0.625)
    c2(18, 19, 0, -1, 0.3125)
    c2(18, 21, 0, 1, 0.3125)
    c2(18, 22, -1, 0, 0.3125)
    c2(18, 24, 1, 0, 0.3125)
    c2(19, 25)

    # Axial features:
    # 0 barriers, 1 goals, 2 player-height rays, 3 player,
    # 4:7 oriented player edges, 7 fluid domain, 8 air,
    # 9 buttons, 10 ready gates, 11 pump bridge domain,
    # 12:14 empty-edge sources, 14 self-response correction,
    # 15 excluded click cells, 16:18 oriented ready gates.
    W3[0, 2] = 1.0
    W3[0, 64 + 9] = 1.0
    W3[0, 64 + 11] = 1.0
    W3[0, 32 + 10] = 1.0
    W3[0, 32 + 12] = 1.0
    for inp in (13, 14, 5):
        W3[0, inp] = -10.0

    W3[1, 5] = 1.0
    W3[2, 32 + 6] = 1.0
    W3[2, 64 + 7] = 1.0
    W3[2, 64 + 8] = 1.0
    for d in range(3):
        W3[3, 6 + d] = 1.0
        W3[4 + d, 6 + d] = 1.0
        W3[7, 6 + d] = 1.0
    W3[7, 0] = 1.0
    W3[7, 3] = 1.0
    W3[8, 1] = 1.0
    W3[9, 3] = 1.0
    W3[10, 15] = 1.0
    W3[11, 2] = 1.0
    W3[11, 3] = 1.0
    W3[12, 16] = 1.0
    W3[13, 17] = 1.0
    W3[14, 18] = 1.0

    b3[15] = 1.0
    W3[15, 19] = -1.0
    W3[15, 15] = -1.0

    W3[16, 15] = 1.0
    W3[16, 64 + 11] = 1.0
    b3[16] = -1.0
    W3[17, 15] = 1.0
    W3[17, 32 + 12] = 1.0
    b3[17] = -1.0

    # H0-H1: clipped goal flood; H2: arrival-time potential.
    for out in (0, 1):
        for dy, dx in dirs:
            r(out, 0, dy, dx)
            r(out, 1, dy, dx, -1.0)
        Uh[out, 1] = 1.0
        Uh[out, 0] = -8.0
    bh[1] = -1.0
    r(2, 2)
    r(2, 0)
    r(2, 1, v=-1.0)

    # H3: the player's height signal, confined to its axial ray.
    for dy, dx in dirs:
        r(3, 3, dy, dx, 0.5)
    Uh[3, 3] = 0.5
    Uh[3, 2] = 2048.0
    Uh[3, 0] = -2048.0
    bh[3] = -2048.0

    # H4-H6 serve two disjoint kinds of pixels:
    # player edges obtain height derivatives with baseline +2;
    # ready gates obtain directional crossing tests with baseline -2.
    gradient_dirs = [(1, 0), (0, -1), (0, 1)]
    gate_features = [17, 16, 16]
    for out, feat, gfeat, (dy, dx) in zip(
            (4, 5, 6), (4, 5, 6), gate_features, gradient_dirs):
        r(out, 2, dy, dx)
        r(out, 2, -dy, -dx, -1.0)
        r(out, 3, -dy, -dx, 0.1)
        r(out, 3, dy, dx, -0.1)
        bh[out] = -1024.0
        Uh[out, feat] = 1026.0
        Uh[out, gfeat] = 1022.0

    # H7,H9: empty-donor fields along the two pump axes.
    for dx in (-1, 1):
        r(7, 7, 0, dx, 0.4)
    for dy in (-1, 1):
        r(9, 9, dy, 0, 0.4)
    for out, source in ((7, 12), (9, 13)):
        Uh[out, source] = 1.0
        Uh[out, 11] = 8.0
        bh[out] = -8.0

    # H8: an air-side signal requesting a neighboring surface rise.
    for dy, dx in dirs:
        r(8, 8, dy, dx, 0.25)
        r(8, 3, dy, dx, 0.25)
    Uh[8, 8] = 2048.0
    Uh[8, 2] = -2048.0
    Uh[8, 0] = -2048.0
    bh[8] = -2048.0

    # H10 combines a five-iteration local settling delay with a
    # rectified empty-donor penalty. Local self-responses cancel.
    availability_scale = 262144.0
    r(10, 7, v=availability_scale)
    r(10, 9, v=availability_scale)
    r(10, 2, v=-1.0)
    Uh[10, 14] = -availability_scale
    bh[10] = 5.0

    # H11-H12: fill-request flood.
    # H13-H14: drain-request flood.
    for out, pos, neg in (
        (11, 11, 12), (12, 11, 12),
        (13, 13, 14), (14, 13, 14),
    ):
        for dy, dx in dirs:
            r(out, pos, dy, dx)
            r(out, neg, dy, dx, -1.0)

        Uh[out, 7] = 4096.0
        bh[out] = -4096.0 - 10000.0
        r(out, 0, v=10000.0)
        r(out, 1, v=-10000.0)
        r(out, 10, v=-10000.0)

    bh[12] -= 1.0
    bh[14] -= 1.0

    for out in (11, 12):
        r(out, 4, v=-1000.0)
        r(out, 5, v=1000.0)
        r(out, 6, v=1000.0)
        Uh[out, 4] += 3000.0
        Uh[out, 5] -= 1000.0
        Uh[out, 6] -= 1000.0
        for dy, dx in dirs:
            r(out, 8, dy, dx, 0.25)

    for out in (13, 14):
        r(out, 4, v=1000.0)
        r(out, 5, v=-1000.0)
        r(out, 6, v=-1000.0)
        Uh[out, 4] -= 1000.0
        Uh[out, 5] += 3000.0
        Uh[out, 6] += 3000.0
        r(out, 3)

    # H15: pressure across pump dividers. Fill requests resist
    # donating; drain requests encourage a neighboring receiver.
    for dy, dx in dirs:
        r(15, 15, dy, dx, 0.2)
    r(15, 13)
    r(15, 14, v=-1.0)
    r(15, 11, v=-2.0)
    r(15, 12, v=2.0)
    Uh[15, 9] = 1.0
    Uh[15, 11] = 4096.0
    bh[15] = -4096.0

    # Always select the click action; all decisions are spatial.
    p["bo"][6] = 1.0
    click = p["Wc"][0]
    click[15] = -100000000.0
    click[32 + 10] = -1000.0
    click[32 + 11] = 100.0
    click[32 + 12] = -100.0
    click[32 + 13] = -200.0
    click[32 + 14] = 200.0
    click[32 + 15] = 10.0
    for channel in (4, 5, 6):
        click[32 + channel] = 10000.0
    click[32 + 2] = 0.0001

    return p
