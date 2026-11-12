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

    def k1(o, c, dy, dx, v):
        p["W1"][o, c, dy + 2, dx + 2] += v

    def k2(o, c, dy, dx, v):
        p["W2"][o, c, dy + 1, dx + 1] += v

    def kh(o, c, dy, dx, v):
        p["Wh"][o, c, dy + 1, dx + 1] += v

    def f3(o, c, v, pool=None):
        offset = 0 if pool is None else (32 if pool == "row" else 64)
        p["W3"][o, offset + c] += v

    def exact_template(o, a, oy, ox):
        n = 0
        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                k1(o, int(a[y, x]), y + oy, x + ox, 1.0)
                n += 1
        p["b1"][o] = 1.0 - n

    # W1: player templates. Their reference point is top-left + (1,1).
    player = np.array([[10, 10], [9, 9]], dtype=np.int8)
    for r in range(4):
        a = np.rot90(player, r)
        exact_template(r, a, -1, -1)
        # An isolated 2x2 body, not a small patch of the exit.
        for y in range(-2, 2):
            for x in range(-2, 2):
                if not (-1 <= y <= 0 and -1 <= x <= 0):
                    k1(r, 9, y, x, -1.0)
                    k1(r, 10, y, x, -1.0)

        large = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)
        exact_template(4 + r, large, -1, -1)

    # Two opposite one-pixel-border corner templates cover all exit rotations.
    corner = np.array([
        [9, 9, 9],
        [9, 10, 10],
        [9, 10, 10],
    ], dtype=np.int8)
    exact_template(8, corner, -1, -1)
    exact_template(9, np.rot90(corner, 2), -1, -1)

    # Switches: a 6-coloured frame around either door colour.
    for o, colour in ((10, 13), (11, 12)):
        a = np.full((4, 4), 6, dtype=np.int8)
        a[1:3, 1:3] = colour
        exact_template(o, a, -1, -1)

    # Small-body collision footprint. Decorative corners are traversable here.
    for dy in (-1, 0):
        for dx in (-1, 0):
            for colour in (3, 4, 5):
                k1(12, colour, dy, dx, 1.0)

            for colour in (12, 13):
                k1(13, colour, dy, dx, 1.0)

    # Door-coloured pixels inside a switch are not navigation barriers.
    for dy in range(-2, 2):
        for dx in range(-2, 2):
            k1(13, 6, dy, dx, -5.0)

    # Raw spell colours.
    for o, colour in ((14, 6), (15, 11), (16, 15)):
        k1(o, colour, 0, 0, 1.0)

    # Coordinate features.
    k1(17, 16, 0, 0, 1.0)
    k1(18, 17, 0, 0, 1.0)
    p["b1"][17:19] = 1.0

    # Absolute-coordinate ramps for the nine canonical click cells.
    xs = (25, 30, 35)
    ys = (50, 55, 59)
    for i, c in enumerate(xs):
        o = 19 + 2 * i
        k1(o, 17, 0, 0, 63.0)
        p["b1"][o] = -float(c)
        k1(o + 1, 17, 0, 0, -63.0)
        p["b1"][o + 1] = float(c)

    for i, c in enumerate(ys):
        o = 25 + 2 * i
        k1(o, 16, 0, 0, 63.0)
        p["b1"][o] = -float(c)
        k1(o + 1, 16, 0, 0, -63.0)
        p["b1"][o + 1] = float(c)

    k1(31, 14, 0, 0, 1.0)

    # W2 feature numbering.
    P, BIG = 0, 1
    PROBES = (2, 3, 4, 5)  # up, down, left, right
    SOURCE, SWITCH, BAD = 6, 7, 8
    FIRE_COLOUR, TELE_COLOUR, SIZE_COLOUR = 9, 10, 11
    ROW, COL, SWITCH_Y, SWITCH_X = 12, 13, 14, 15
    ON = 16
    SLOT0 = 17
    ANCHOR, SMALL_TELE, LOW_X, HIGH_Y = 26, 27, 28, 29

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for c in range(8):
        k2(P, c, 0, 0, 1.0)
        if c >= 4:
            k2(BIG, c, 0, 0, 1.0)
        for out, (dy, dx) in zip(PROBES, directions):
            k2(out, c, -dy, -dx, 1.0)
        # Top-left reference used by the game's fire ray.
        k2(ANCHOR, c, 1, 1, 1.0)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for c in (8, 9, 10, 11):
                k2(SOURCE, c, dy, dx, 1.0)
            for c in (10, 11):
                k2(SWITCH, c, dy, dx, 1.0)

    k2(BAD, 12, 0, 0, 1.0)
    k2(BAD, 13, 0, 0, 1.0)

    # Vertical expansion meets the selected click rows of the spell panel.
    for out, inp in (
        (FIRE_COLOUR, 14),
        (TELE_COLOUR, 15),
        (SIZE_COLOUR, 16),
    ):
        k2(out, inp, -1, 0, 1.0)
        k2(out, inp, 1, 0, 1.0)

    k2(ROW, 17, 0, 0, 1.0)
    k2(COL, 18, 0, 0, 1.0)

    # Coordinates gated by a switch detector.
    for out, coord in ((SWITCH_Y, 17), (SWITCH_X, 18)):
        k2(out, coord, 0, 0, 1.0)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for c in (10, 11):
                    k2(out, c, dy, dx, 10.0)
        p["b2"][out] = -10.0

    k2(ON, 31, 0, 0, 1.0)

    for r in range(3):
        for c in range(3):
            out = SLOT0 + 3 * r + c
            for inp in (19 + 2*c, 20 + 2*c, 25 + 2*r, 26 + 2*r):
                k2(out, inp, 0, 0, -1.0)
            p["b2"][out] = 1.0

    # A small teleport target, excluding the fixed panel regions.
    for dy in (-1, 0):
        for dx in (-1, 0):
            k2(SMALL_TELE, 16, dy, dx, 1.0)
    p["b2"][SMALL_TELE] = -3.0

    k2(LOW_X, 18, 0, 0, -63.0)
    p["b2"][LOW_X] = 74.0
    k2(HIGH_Y, 17, 0, 0, 63.0)
    p["b2"][HIGH_Y] = -109.0

    # W3 numbering.
    F_PROBES = (0, 1, 2, 3)
    F_PLAYER, F_SOURCE, F_BAD, F_BIG, F_TELE = 4, 5, 6, 7, 8
    F_FLAGS = (9, 10, 11, 12)  # up, down, left, right
    F_SIZE_AV, F_TELE_AV, F_FIRE_AV = 13, 14, 15
    SIZE_MISS, SIZE_WRONG = 16, 17
    FIRE_MISS, FIRE_WRONG = 18, 19
    TELE_MISS, TELE_WRONG = 20, 21

    for out, inp in zip(F_PROBES, PROBES):
        f3(out, inp, 1.0)
    f3(F_PLAYER, P, 1.0)
    f3(F_SOURCE, SOURCE, 1.0)
    f3(F_BAD, BAD, 1.0)

    f3(F_BIG, BIG, 1.0, "row")
    f3(F_BIG, BIG, 1.0, "col")

    f3(F_TELE, SMALL_TELE, 1.0)
    f3(F_TELE, LOW_X, -100.0)
    f3(F_TELE, HIGH_Y, -100.0)

    # Aligned switch directions, evaluated at the player's ray origin.
    M = 10.0
    for out, coord, swcoord, pool, sign in (
        (F_FLAGS[0], ROW, SWITCH_Y, "col", -1.0),
        (F_FLAGS[1], ROW, SWITCH_Y, "col", 1.0),
        (F_FLAGS[2], COL, SWITCH_X, "row", -1.0),
        (F_FLAGS[3], COL, SWITCH_X, "row", 1.0),
    ):
        f3(out, swcoord, sign, pool)
        f3(out, coord, -sign)
        f3(out, ANCHOR, M)
        f3(out, SWITCH, M, pool)
        p["b3"][out] = -2.0 * M

    for out, inp in (
        (F_SIZE_AV, SIZE_COLOUR),
        (F_TELE_AV, TELE_COLOUR),
        (F_FIRE_AV, FIRE_COLOUR),
    ):
        f3(out, inp, 1.0)

    patterns = (
        ({1, 3, 5, 7}, SIZE_COLOUR, SIZE_MISS, SIZE_WRONG),
        ({1, 4, 7}, FIRE_COLOUR, FIRE_MISS, FIRE_WRONG),
        ({0, 1, 4}, TELE_COLOUR, TELE_MISS, TELE_WRONG),
    )
    for target, av, missing, wrong in patterns:
        for i in range(9):
            if i in target:
                f3(missing, SLOT0 + i, 10.0)
            else:
                f3(wrong, SLOT0 + i, 10.0)

        f3(missing, ON, -10.0)
        f3(missing, av, 1.0, "row")
        p["b3"][missing] = -10.0

        if wrong == TELE_WRONG:
            # Repair arbitrary extra bits, including the unused bottom row.
            p["W3"][wrong] *= 0.1
            f3(wrong, ON, 1.0)
            p["b3"][wrong] = -1.0
        else:
            f3(wrong, ON, 1.0)
            f3(wrong, av, 1.0, "row")
            p["b3"][wrong] = -11.0

    # Tied spatial block.
    Q = 0
    H_DIR = (1, 2, 3, 4)
    NO_LOCAL, BAD_SMALL_TELE = 5, 6
    BIG_FIELD = 7
    N_RAW, N_OVER = 8, 9
    T_RAW, T_OVER = 10, 11
    CLICK_SHRINK, CLICK_GROW, CLICK_TELE = 12, 13, 14
    CLOCK = 15

    # Discounted harmonic navigation potential.
    for dy, dx in directions:
        kh(Q, Q, dy, dx, 0.249)
    p["Uh"][Q, F_SOURCE] = 1.0
    p["Uh"][Q, F_BAD] = -1000.0

    # Four samples two pixels away from the detected reference point.
    for out, feature, (dy, dx) in zip(H_DIR, F_PROBES, directions):
        kh(out, Q, dy, dx, 1.0)
        p["Uh"][out, feature] = 1000.0
        p["bh"][out] = -1000.0

    # Clocked zero-connectivity tests avoid the initial propagation transient.
    kh(CLOCK, CLOCK, 0, 0, 1.0)
    p["bh"][CLOCK] = 1.0
    for out, marker in ((NO_LOCAL, F_PLAYER), (BAD_SMALL_TELE, F_TELE)):
        kh(out, CLOCK, 0, 0, 1.0)
        kh(out, Q, 0, 0, -1.0e34)
        p["Uh"][out, marker] = 100000.0
        p["bh"][out] = -100052.0

    # Static size information diffuses from an axial cross.
    for dy, dx in directions:
        kh(BIG_FIELD, BIG_FIELD, dy, dx, 0.249)
    p["Uh"][BIG_FIELD, F_BIG] = 1.0

    # Capped flood pairs: raw-overflow is a propagated 0/1 signal.
    for raw, over, source in (
        (N_RAW, N_OVER, NO_LOCAL),
        (T_RAW, T_OVER, BAD_SMALL_TELE),
    ):
        for out in (raw, over):
            for dy, dx in directions:
                kh(out, raw, dy, dx, 1.0)
                kh(out, over, dy, dx, -1.0)
            kh(out, source, 0, 0, 1.0)
        p["bh"][over] = -1.0

    def click_eligibility(out, missing, wrong, bias):
        gate = 1.0e12
        p["Uh"][out, missing] = gate + 1.0
        p["Uh"][out, wrong] = gate + 10.0
        p["bh"][out] = -gate + bias

    click_eligibility(CLICK_SHRINK, SIZE_MISS, SIZE_WRONG, -20.0)
    kh(CLICK_SHRINK, BIG_FIELD, 0, 0, 1.0e6)
    kh(CLICK_SHRINK, T_RAW, 0, 0, -1.0e10)
    kh(CLICK_SHRINK, T_OVER, 0, 0, 1.0e10)

    click_eligibility(CLICK_GROW, SIZE_MISS, SIZE_WRONG, -50.0)
    kh(CLICK_GROW, T_RAW, 0, 0, 100.0)
    kh(CLICK_GROW, T_OVER, 0, 0, -100.0)
    kh(CLICK_GROW, BIG_FIELD, 0, 0, -1.0e8)

    click_eligibility(CLICK_TELE, TELE_MISS, TELE_WRONG, -50.0)
    kh(CLICK_TELE, N_RAW, 0, 0, 100.0)
    kh(CLICK_TELE, N_OVER, 0, 0, -100.0)

    # These large eligibility gates must preserve their small priorities.
    # Use separate constant and eligibility scales below float32's integer limit.
    for out, missing, wrong, bias in (
        (CLICK_SHRINK, SIZE_MISS, SIZE_WRONG, -20.0),
        (CLICK_GROW, SIZE_MISS, SIZE_WRONG, -50.0),
        (CLICK_TELE, TELE_MISS, TELE_WRONG, -50.0),
    ):
        gate = 1.0e6
        p["Uh"][out, missing] = gate + 1.0
        p["Uh"][out, wrong] = gate + 10.0
        p["bh"][out] = -gate + bias

    # Keep all non-eligible shrink cells suppressed despite the size field.
    kh(CLICK_SHRINK, BIG_FIELD, 0, 0, -999000.0)
    kh(CLICK_SHRINK, T_RAW, 0, 0, 9999900000.0)
    kh(CLICK_SHRINK, T_OVER, 0, 0, -9999900000.0)
    kh(CLICK_GROW, BIG_FIELD, 0, 0, 99999000.0)

    # Effective click-control coefficients after the additions above:
    # shrink: 1000*size_field - 100000*target_bad
    # grow:   100*target_bad - 1000*size_field.

    # Global readout. g starts with h.max, then h.mean, then f3.max.
    def zg(o, c, v):
        p["Wz"][o, c] += v

    Z_BIG, Z_N, Z_T = 0, 1, 2
    Z_SHRINK, Z_GROW = 3, 4
    Z_NAV = (5, 6, 7, 8)
    Z_FLAGS = (9, 10, 11, 12)
    Z_AV = ((13, 14), (15, 16), (17, 18))

    zg(Z_BIG, 32 + F_BIG, 0.5)
    zg(Z_N, N_RAW, 1.0)
    zg(Z_N, N_OVER, -1.0)
    zg(Z_T, T_RAW, 1.0)
    zg(Z_T, T_OVER, -1.0)

    for z, sb, st in ((Z_SHRINK, 1.0, -1.0), (Z_GROW, -1.0, 1.0)):
        zg(z, 32 + F_BIG, 0.5 * sb)
        zg(z, T_RAW, st)
        zg(z, T_OVER, -st)

    for z, h in zip(Z_NAV, H_DIR):
        zg(z, h, 1.0)
    for z, f in zip(Z_FLAGS, F_FLAGS):
        zg(z, 32 + f, 1000.0)

    for (za, zb), f in zip(Z_AV, (F_SIZE_AV, F_TELE_AV, F_FIRE_AV)):
        zg(za, 32 + f, 1.0)
        zg(zb, 32 + f, 1.0)
        p["bz"][zb] = -1.0

    # Memory 0..3: retained facing. Memory 4..7: aligned facing tests.
    for i in range(4):
        p["Wm"][i, i] = 5.0
        for j in range(4):
            p["Wma"][i, 1 + j] = 20.0 if i == j else -20.0

        match = 4 + i
        p["Wm"][match, i] = 1000.0
        p["Wmz"][match, Z_FLAGS[i]] = 1.0
        p["bm"][match] = -1005.0
        for j in range(4):
            p["Wma"][match, 1 + j] = 2000.0 if i == j else -2000.0

    M_SIZE, M_TELE = 8, 9
    p["Wmz"][M_SIZE, Z_SHRINK] = 20.0
    p["Wmz"][M_SIZE, Z_GROW] = 20.0
    p["Wmz"][M_SIZE, Z_AV[0][0]] = 20.0
    p["Wmz"][M_SIZE, Z_AV[0][1]] = -20.0
    p["bm"][M_SIZE] = -30.0

    p["Wmz"][M_TELE, Z_N] = 20.0
    p["Wmz"][M_TELE, Z_AV[1][0]] = 20.0
    p["Wmz"][M_TELE, Z_AV[1][1]] = -20.0
    p["bm"][M_TELE] = -30.0

    # Monotone, unsaturated movement scores retain very small potentials.
    for i, z in enumerate(Z_NAV):
        p["Wmz"][10 + i, z] = 0.01
        p["Wo"][1 + i, 128 + 10 + i] = 1.0

    p["bo"][:] = -10000.0
    p["bo"][1:5] = 0.0

    p["Wo"][6, 128 + M_SIZE] = 8.0
    p["Wo"][6, 128 + M_TELE] = 6.0
    for i in range(4):
        p["Wo"][6, 128 + 4 + i] = 3.0
    p["Wo"][6, Z_AV[2][0]] = 3.0
    p["Wo"][6, Z_AV[2][1]] = -3.0
    p["bo"][6] = 20.0

    # Default fire pattern; spatial control channels override it for other spells.
    p["Wc"][0, FIRE_MISS] = 1.0
    p["Wc"][0, FIRE_WRONG] = 10.0
    p["Wc"][0, 32 + CLICK_SHRINK] = 10.0
    p["Wc"][0, 32 + CLICK_GROW] = 10.0
    p["Wc"][0, 32 + CLICK_TELE] = 1.0

    return p
