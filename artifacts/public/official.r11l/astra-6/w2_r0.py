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

    def w1(out, channel, dx=0, dy=0, value=1.0):
        p["W1"][out, channel, dy + 2, dx + 2] += value

    def w2(out, channel, dx=0, dy=0, value=1.0):
        p["W2"][out, channel, dy + 1, dx + 1] += value

    def wh(out, channel, dx=0, dy=0, value=1.0):
        p["Wh"][out, channel, dy + 1, dx + 1] += value

    def rotate(point, turns):
        x, y = point
        for _ in range(turns):
            x, y = -y, x
        return x, y

    # Solid-ball centre detectors.
    for out, colour in ((0, 15), (1, 14)):
        w1(out, 6)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            w1(out, colour, dx, dy)
        p["b1"][out] = -4.0

    # Four goal-ring quarter detectors per colour.
    quarters = []
    for bank, colour in enumerate((15, 14)):
        samples = []
        for turn in range(4):
            out = 2 + 4 * bank + turn
            sensor = rotate((-1, -1), turn)
            sx, sy = sensor

            for point in ((-1, -3), (-2, -2), (-3, -1), (1, -3)):
                x, y = rotate(point, turn)
                w1(out, colour, x - sx, y - sy)

            for point in ((-1, -2), (-2, -1)):
                x, y = rotate(point, turn)
                w1(out, colour, x - sx, y - sy, -2.0)

            p["b1"][out] = -3.0
            samples.append((out, sensor))
        quarters.append(samples)

    diamond = [
        (dx, dy)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if abs(dx) + abs(dy) <= 2
    ]
    ball_footprint = [
        (dx, dy)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if not (abs(dx) == 2 and abs(dy) == 2)
    ]

    # Selected and inactive anchor centres.
    for out, body, centre in (
        (10, 0, 15), (11, 0, 14),
        (12, 3, 15), (13, 3, 14),
    ):
        for dx, dy in diamond:
            if dx or dy:
                w1(out, body, dx, dy)
        w1(out, centre)
        p["b1"][out] = -12.0

    # Anchor destination footprint. Black also excludes the UI column
    # and discourages overlap with the currently selected anchor.
    p["b1"][14] = float(len(diamond))
    for dx, dy in diamond:
        w1(14, 2, dx, dy)
        w1(14, 0, dx, dy)
        w1(14, 18, dx, dy, -1.0)

    # Exclusion around existing anchor bodies.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w1(15, 0, dx, dy)
            w1(15, 3, dx, dy)

    # Shifted black planes for recognizing a nearby blackened goal rim.
    black_shifts = (
        (16, 2, 0), (17, -2, 0),
        (18, 0, 2), (19, 0, -2),
        (23, 2, 2), (24, -2, 2),
        (25, 2, -2), (26, -2, -2),
    )
    for channel, dx, dy in black_shifts:
        w1(channel, 0, dx, dy)

    w1(20, 16)
    w1(21, 17)

    # Exact 21-pixel ball footprint, rather than a 7x7 dilation.
    p["b1"][22] = float(len(ball_footprint))
    for dx, dy in ball_footprint:
        w1(22, 2, dx, dy)
        w1(22, 10, dx, dy)
        w1(22, 18, dx, dy, -1.0)

    # Extra clearance around coloured balls and goal rims.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            w1(31, 14, dx, dy)
            w1(31, 15, dx, dy)

    # At least three recognized quarters count as an unoccupied goal.
    # Paired ReLUs will clip the result to one in f3.
    for bank in range(2):
        for out, threshold in ((bank, 2.0), (14 + bank, 3.0)):
            for channel, (dx, dy) in quarters[bank]:
                w2(out, channel, dx, dy)
            p["b2"][out] = -threshold

    black_samples = {
        (dx, dy)
        for dy in range(-3, 4)
        for dx in range(-3, 4)
        if max(abs(dx), abs(dy)) == 3
    }
    black_samples.update(
        (dx, dy) for dx in (-2, 2) for dy in (-2, 2)
    )

    # Suppress balls beside completed, blackened goals.
    for out, ball_channel in ((2, 0), (3, 1)):
        w2(out, ball_channel)
        for dx, dy in sorted(black_samples):
            for channel, sx, sy in black_shifts:
                rx, ry = dx - sx, dy - sy
                if -1 <= rx <= 1 and -1 <= ry <= 1:
                    w2(out, channel, rx, ry, -1.0)
                    break

    w2(4, 12)
    w2(5, 13)

    STRIDE = 128.0
    CODE = 16384.0
    COORD_GATE = 8192.0

    # Encoded selected-anchor coordinate: x + 128*y + colour digit.
    w2(6, 10, value=COORD_GATE)
    w2(6, 11, value=COORD_GATE + CODE)
    w2(6, 20, value=63.0 * STRIDE)
    w2(6, 21, value=63.0)
    p["b2"][6] = -COORD_GATE

    w2(7, 14)
    p["W2"][7, 31, :, :] = 0.05
    w2(8, 15)
    w2(9, 22)
    w2(10, 20, value=63.0 * STRIDE)
    w2(10, 21, value=63.0)

    for out, anchor_channel in ((11, 12), (12, 13)):
        w2(out, anchor_channel, value=COORD_GATE)
        w2(out, 20, value=63.0 * STRIDE)
        w2(out, 21, value=63.0)
        p["b2"][out] = -COORD_GATE

    # f3 feature layout.
    for bank in range(2):
        p["W3"][bank, bank] = 1.0
        p["W3"][bank, 14 + bank] = -1.0

    direct = {
        3: 2, 4: 3, 5: 4, 6: 5,
        8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12,
    }
    for out, source in direct.items():
        p["W3"][out, source] = 1.0

    # Green-goal row stripe and selected-coordinate row stripe.
    p["W3"][2, 32 + 1] = 1.0
    p["W3"][2, 32 + 15] = -1.0
    p["W3"][7, 32 + 6] = 1.0

    # h0,h1: prefix scans broadcasting goal presence.
    BASE = 4.0
    FLAG_TOTAL = BASE * 65.0
    for out, dy in ((0, -1), (1, 1)):
        wh(out, out, dy=dy)
        p["Uh"][out, 2] = -2.0
        p["bh"][out] = BASE

    # h2-h3: Boolean obstacle-aware wavefront.
    # Green is seeded immediately; pink is seeded when no green goal remains.
    PROPAGATE = 1024.0
    for out, extra in ((2, 0.0), (3, -1.0)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    wh(out, 2, dx, dy, PROPAGATE)
                    wh(out, 3, dx, dy, -PROPAGATE)

        wh(out, 0)
        wh(out, 1)
        p["Uh"][out, 0] = 1.5
        p["Uh"][out, 1] = PROPAGATE
        p["Uh"][out, 2] = 2.0
        p["Uh"][out, 10] = -10000.0
        p["bh"][out] = -FLAG_TOTAL - 0.25 + extra

    # h4: integrated reachability, a distance-like routing potential.
    wh(4, 4)
    wh(4, 2)
    wh(4, 3, value=-1.0)

    # h5-h6 = 1 + clip(horizontal gradient, -1, 1).
    # h7-h8 = 1 + clip(vertical gradient, -1, 1).
    # Completed green balls are locally suppressed by the black-rim detector.
    CLIP = 1.0
    BANK_GATE = 256.0
    LOCAL_GATE = 2048.0

    for out, axis, offset in (
        (5, 0, CLIP), (6, 0, -CLIP),
        (7, 1, CLIP), (8, 1, -CLIP),
    ):
        if axis == 0:
            wh(out, 4, 1, 0)
            wh(out, 4, -1, 0, -1.0)
        else:
            wh(out, 4, 0, 1)
            wh(out, 4, 0, -1, -1.0)

        wh(out, 0, value=BANK_GATE)
        wh(out, 1, value=BANK_GATE)
        p["Uh"][out, 2] = 2.0 * BANK_GATE
        p["Uh"][out, 3] = LOCAL_GATE
        p["Uh"][out, 4] = LOCAL_GATE + 2.0 * BANK_GATE
        p["bh"][out] = (
            offset - LOCAL_GATE - BANK_GATE * FLAG_TOTAL
        )

    STEP = 12.0
    force = {
        5: STEP,
        6: -STEP,
        7: STEP * STRIDE,
        8: -STEP * STRIDE,
    }

    # h9,h10: horizontal scans of the encoded bounded displacement.
    for out, dx in ((9, -1), (10, 1)):
        wh(out, out, dx=dx)
        for source, value in force.items():
            wh(out, source, value=value)

    # h11,h12: vertical scans, also broadcasting the selected coordinate.
    for out, dy in ((11, -1), (12, 1)):
        wh(out, out, dy=dy)
        wh(out, 9)
        wh(out, 10)
        for source, value in force.items():
            wh(out, source, value=-value)
        p["Uh"][out, 7] = 1.0

    # Residual: proposed encoded coordinate minus candidate coordinate.
    rh = np.zeros(16, np.float32)
    ru = np.zeros(32, np.float32)
    rh[11] = rh[12] = 1.0
    rh[9] = rh[10] = -1.0
    for source, value in force.items():
        rh[source] = value

    rh[0] = rh[1] = CODE / 2.0
    ru[2] = CODE
    ru[7] = -1.0
    ru[11] = -1.0
    rb = -CLIP * STEP * (1.0 + STRIDE) - CODE * FLAG_TOTAL / 2.0

    # h13 supplies the positive half needed for -abs(residual).
    p["Wh"][13, :, 1, 1] = rh
    p["Uh"][13] = ru
    p["bh"][13] = rb

    WALL_COST = 1000.0
    p["Wc"][0, 32:] = rh
    p["Wc"][0, :32] = ru
    p["Wc"][0, 32 + 13] -= 2.0
    p["Wc"][0, 8] -= WALL_COST
    p["Wc"][0, 9] -= WALL_COST
    p["bc"][0] = rb

    # h14,h15: fallback selection. At eligible inactive anchors,
    # cancel the movement cost and substitute a small selection cost.
    cost_h = -rh.copy()
    cost_h[13] += 2.0
    cost_u = -ru.copy()
    cost_u[8] += WALL_COST
    cost_u[9] += WALL_COST

    SELECT_GATE = 65536.0
    SELECT_COST = 8.0

    for out, is_green in ((14, True), (15, False)):
        hh = cost_h.copy()
        uu = cost_u.copy()
        bb = -rb - SELECT_COST

        if is_green:
            hh[0] -= SELECT_GATE / 2.0
            hh[1] -= SELECT_GATE / 2.0
            uu[2] -= SELECT_GATE
            uu[6] += SELECT_GATE
            bb += SELECT_GATE * (FLAG_TOTAL / 2.0 - 2.0)
        else:
            hh[0] += SELECT_GATE / 2.0
            hh[1] += SELECT_GATE / 2.0
            uu[2] += SELECT_GATE
            uu[5] += SELECT_GATE
            bb -= SELECT_GATE * (FLAG_TOTAL / 2.0 + 1.0)

        p["Wh"][out, :, 1, 1] = hh
        p["Uh"][out] = uu
        p["bh"][out] = bb
        p["Wc"][0, 32 + out] = 1.0

    # Deterministic spatial tie-break among inactive anchors.
    p["Wc"][0, 12] -= 0.001
    p["Wc"][0, 13] -= 0.001

    p["bo"][6] = 1.0
    return p
