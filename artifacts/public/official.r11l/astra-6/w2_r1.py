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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    def k1(out, channels, dy=0, dx=0, value=1.0):
        for channel in channels:
            p["W1"][out, channel, dy + 2, dx + 2] += value

    def k2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def win2(out, inp, value=1.0):
        p["W2"][out, inp, :, :] += value

    colours = (9, 8, 14, 11, 15)
    markers = (4, 14, 15)
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    quadrants = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    # First layer: local visual features.
    SEL = 0
    ANCH = (1, 2, 3, 4)
    PASS = 5
    SRC = (6, 7, 8, 9, 10)
    RING = (11, 12, 13, 14)
    OUTER, WALL, BLACKSEL = 15, 16, 17
    PICK = (18, 19, 20, 21)
    EARLY, WHITEBODY, LOWER_RING = 22, 23, 24
    ROW, COL, DENS, FINGERPRINT = 25, 26, 27, 28

    k1(SEL, markers)
    k1(SEL, (0,), 0, -1)
    k1(SEL, (0,), 0, 1)
    p["b1"][SEL] = -2.0

    for channel, (dy, dx) in zip(ANCH, quadrants):
        k1(channel, markers, dy, dx)
        k1(channel, (0, 3), dy, dx - 1)
        k1(channel, (0, 3), dy, dx + 1)
        p["b1"][channel] = -2.0

    k1(PASS, (0, 1, 3, 4, 6, 8, 9, 11, 14, 15))

    interfaces = ((0, 8), (0, 9), (0, 11), (0, 14), (0,))
    for out, colour, adjacent in zip(SRC, colours, interfaces):
        k1(out, (colour,), value=4.0)
        for dy, dx in cardinal:
            k1(out, adjacent, dy, dx)
        p["b1"][out] = -4.0

    for out, colour in zip(RING, (8, 11, 14, 15)):
        k1(out, (colour,))
        for dy, dx in cardinal:
            k1(out, (1, 5, 10), dy, dx)
        p["b1"][out] = -4.0

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) + abs(dx) > 2:
                k1(OUTER, colours, dy, dx)
            if abs(dy) + abs(dx) <= 2:
                k1(WALL, (2,), dy, dx)
            k1(DENS, (0, 3), dy, dx)

    k1(BLACKSEL, (4,))
    k1(BLACKSEL, (0,), 0, -1)
    k1(BLACKSEL, (0,), 0, 1)
    p["b1"][BLACKSEL] = -2.0

    # Broad pigment patches; ordinary centre-6 discs are vetoed.
    for out, colour in zip(PICK, (9, 8, 14, 11)):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                k1(out, (colour,), dy, dx)
                k1(out, (6,), dy, dx, -25.0)
        p["b1"][out] = -8.0

    # A green disc high on the board favours the upper approach.
    k1(EARLY, (6,), value=64.0)
    k1(EARLY, (14,), 0, -1, 64.0)
    k1(EARLY, (16,), value=-63.0)
    p["b1"][EARLY] = -103.0

    # White disc centre: 2 - row/63.
    k1(WHITEBODY, (6,), value=4.0)
    k1(WHITEBODY, (15,), 0, -1, 4.0)
    k1(WHITEBODY, (16,), value=-1.0)
    p["b1"][WHITEBODY] = -6.0

    k1(LOWER_RING, (15,), -2, 0)
    k1(LOWER_RING, (1, 5, 10), -2, -1)
    k1(LOWER_RING, (1, 5, 10), -2, 1)
    p["b1"][LOWER_RING] = -2.0

    k1(ROW, (16,))
    k1(COL, (17,))

    # A position-weighted colour checksum. Excludes the budget-bar colours.
    k1(FINGERPRINT, (3, 4, 6, 8, 9, 11, 14, 15), value=2.0)
    k1(FINGERPRINT, (16,), value=0.83)
    k1(FINGERPRINT, (17,), value=1.17)
    p["b1"][FINGERPRINT] = -2.0

    # Second layer.
    RAW = (0, 1, 2, 3)
    CAP = (4, 5, 6)
    UNION_ABS = 7
    ABS = tuple(range(8, 16))
    SEL2, INACTIVE2, BLOCK2, PASS2 = 16, 17, 18, 19
    WALL2, NEAR2, PARK2, PARKCAP2 = 20, 21, 22, 23
    EARLY2, EARLYCAP2, BODY2 = 24, 25, 26
    WAY2, ROW2, COL2, DENS2, WAYCAP2 = 27, 28, 29, 30, 31

    grouped_sources = (
        (SRC[0], SRC[1]),
        (SRC[2], SRC[3]),
        (SRC[4],),
    )
    for raw, cap, inputs in zip(RAW[:3], CAP, grouped_sources):
        for inp in inputs:
            k2(raw, inp)
            k2(cap, inp)
        p["b2"][cap] = -1.0

    # BLACKSEL is already binary and needs no clipping companion.
    k2(RAW[3], BLACKSEL)

    for out, inp in zip(ABS, PICK + RING):
        win2(out, inp, -1.0)
        p["b2"][out] = 1.0

    for inp in PICK + RING:
        win2(UNION_ABS, inp, -1.0)
    p["b2"][UNION_ABS] = 1.0

    k2(SEL2, SEL)
    for inp, (dy, dx) in zip(ANCH, quadrants):
        k2(INACTIVE2, inp, -dy, -dx, 0.25)
        win2(BLOCK2, inp)
        k2(NEAR2, inp, -dy, -dx, 256.0)
    k2(INACTIVE2, SEL, value=-1.0)

    k2(PASS2, PASS)
    k2(WALL2, WALL)
    win2(DENS2, DENS, 1.0 / 9.0)

    k2(NEAR2, SEL, value=-1024.0)
    win2(NEAR2, OUTER)
    p["b2"][NEAR2] = -1024.0

    for out, extra in ((PARK2, 0.0), (PARKCAP2, 1.0)):
        k2(out, SEL, value=1024.0)
        win2(out, OUTER)
        p["b2"][out] = -1024.0 - extra

    k2(EARLY2, EARLY)
    k2(EARLYCAP2, EARLY)
    p["b2"][EARLYCAP2] = -1.0
    k2(BODY2, WHITEBODY)

    win2(WAY2, LOWER_RING)
    win2(WAYCAP2, LOWER_RING)
    p["b2"][WAYCAP2] = -1.0

    k2(ROW2, ROW)
    k2(ROW2, FINGERPRINT, value=0.01)
    k2(COL2, COL)

    # Third layer.
    FPASS, FSEL = 0, 1
    FSRC = (2, 3, 4, 5)
    FTARGET = tuple(range(6, 14))
    FINACTIVE, FFORBID, FWALL = 14, 15, 16
    FFAR, FUNION = 17, 18
    FBLOCK = (19, 20, 21, 22)
    FPARK, FEARLY, FBODY = 23, 24, 25
    FWAY, FCOORD, FROW, FCOL, FDENS, FWAYPARK = 26, 27, 28, 29, 30, 31

    p["W3"][FPASS, PASS2] = 1.0
    p["W3"][FSEL, SEL2] = 1.0
    for out, raw, cap in zip(FSRC[:3], RAW[:3], CAP):
        p["W3"][out, raw] = 1.0
        p["W3"][out, cap] = -1.0
    p["W3"][FSRC[3], RAW[3]] = 1.0

    for out, absence in zip(FTARGET, ABS):
        p["b3"][out] = 1.0
        p["W3"][out, absence] = -1.0
        p["W3"][out, INACTIVE2] = -1.0
        p["W3"][out, SEL2] = -1.0

    p["W3"][FINACTIVE, INACTIVE2] = 1.0
    p["W3"][FFORBID, BLOCK2] = 1.0
    p["W3"][FFORBID, INACTIVE2] = -64.0
    p["W3"][FWALL, WALL2] = 1.0

    p["W3"][FFAR, INACTIVE2] = 1.0
    p["W3"][FFAR, NEAR2] = -1.0

    p["b3"][FUNION] = 1.0
    p["W3"][FUNION, UNION_ABS] = -1.0
    p["W3"][FUNION, INACTIVE2] = -1.0
    p["W3"][FUNION, SEL2] = -1.0

    # Target order:
    # paint9, paint8, paint14, paint11, ring8, ring11, ring14, ring15.
    permitted = (
        (0, 1, 4),
        (2, 3, 5, 6),
        (7,),
        (0, 1, 2, 3),
    )
    for i, (out, targets) in enumerate(zip(FBLOCK, permitted)):
        for target in targets:
            p["b3"][out] += 1.0
            p["W3"][out, ABS[target]] -= 1.0
        p["W3"][out, INACTIVE2] += 1.0
        p["W3"][out, RAW[i]] -= 16.0
        if i < 3:
            p["W3"][out, CAP[i]] += 16.0

    p["W3"][FPARK, PARK2] = 1.0
    p["W3"][FPARK, PARKCAP2] = -1.0

    p["W3"][FEARLY, 64 + EARLY2] = 1.0
    p["W3"][FEARLY, 64 + EARLYCAP2] = -1.0
    p["W3"][FBODY, 64 + BODY2] = 1.0

    p["W3"][FWAY, 32 + WAY2] = 1.0
    p["W3"][FWAY, 32 + WAYCAP2] = -1.0
    p["W3"][FWAYPARK, SEL2] = 1.0
    p["W3"][FWAYPARK, 32 + WAY2] = 1.0
    p["W3"][FWAYPARK, 32 + WAYCAP2] = -1.0
    p["b3"][FWAYPARK] = -1.0

    p["W3"][FCOORD, ROW2] = 0.5
    p["W3"][FCOORD, COL2] = 0.5
    p["W3"][FROW, ROW2] = 1.0
    p["W3"][FCOL, COL2] = 1.0
    p["W3"][FDENS, DENS2] = 1.0
    p["W3"][FDENS, INACTIVE2] = -1000.0

    # Recurrent field encoded as a-b = clip(input, 0, 1).
    def flood(a, b):
        for out in (a, b):
            p["Wh"][out, a, :, :] = 1.0
            p["Wh"][out, b, :, :] = -1.0
        p["bh"][b] = -1.0

    def drive(a, b, feature, value):
        p["Uh"][a, feature] += value
        p["Uh"][b, feature] += value

    flood(0, 1)
    p["bh"][0] -= 12.0
    p["bh"][1] -= 12.0
    drive(0, 1, FPASS, 12.0)
    drive(0, 1, FSEL, 1.0)

    for i in range(4):
        a, b = 2 + 2 * i, 3 + 2 * i
        flood(a, b)
        p["bh"][a] -= 0.25
        p["bh"][b] -= 0.25
        for out in (a, b):
            p["Wh"][out, 0, 1, 1] += 0.25
            p["Wh"][out, 1, 1, 1] -= 0.25
        drive(a, b, FSRC[i], 0.25)
        drive(a, b, FBLOCK[i], -12.0)

    flood(10, 11)
    drive(10, 11, FPARK, 1.0)
    drive(10, 11, FWAYPARK, 1.0)
    drive(10, 11, FINACTIVE, -12.0)

    # Directed scans broadcast the geometric source stripes.
    p["Wh"][12, 12, 1, 0] = 1.0
    p["Uh"][12, FEARLY] = 1.0
    p["Wh"][13, 13, 1, 2] = 1.0
    p["Uh"][13, FBODY] = 1.0

    p["Uh"][14, FUNION] = 16.0
    p["Uh"][14, FCOORD] = 1.0
    p["Uh"][14, FTARGET[7]] = -16.0
    p["Wh"][14, 12, 1, 1] = -1.0
    p["Wh"][14, 10, 1, 1] = -16.0
    p["Wh"][14, 11, 1, 1] = 16.0
    p["bh"][14] = -16.0

    # Low-side waypoint when a selected white component is above its goal.
    p["Uh"][15, FWAY] = 16.0
    p["Uh"][15, FUNION] = -16.0
    p["Uh"][15, FROW] = 1.0
    p["Uh"][15, FCOL] = -0.15
    p["Wh"][15, 13, 1, 1] = 1.0
    p["Wh"][15, 6, 1, 1] = 16.0
    p["Wh"][15, 7, 1, 1] = -16.0
    p["Wh"][15, 10, 1, 1] = -16.0
    p["Wh"][15, 11, 1, 1] = 16.0
    p["bh"][15] = -34.2

    wc = p["Wc"][0]
    wc[FUNION] = 70.0
    for feature, bonus in zip(
        FTARGET, (20.4, 20.2, 20.3, 20.1, 6.0, 6.0, 6.0, 6.0)
    ):
        wc[feature] = bonus

    wc[FINACTIVE] = 50.0
    wc[FFAR] = 15.0
    wc[FFORBID] = -100000.0
    wc[FWALL] = -100000.0
    wc[FDENS] = -0.12
    wc[FCOORD] = -0.5

    def read_difference(a, b, value):
        wc[32 + a] += value
        wc[32 + b] -= value

    read_difference(0, 1, 3.0)
    for i in range(4):
        read_difference(2 + 2 * i, 3 + 2 * i, -24.0)
    read_difference(10, 11, -60.0)
    wc[32 + 14] = 2.0
    wc[32 + 15] = 1300.0

    # Extract the small checksum embedded in the row feature's mean.
    # The unmodified row plane has mean exactly 0.5.
    p["Wz"][0, 64 + FROW] = 0.1
    p["bz"][0] = -0.05

    # Two-step history and a smooth near-equality test.
    p["Wmz"][0, 0] = 1.0
    p["Wm"][1, 0] = 1.0

    gain = 1.0e7
    epsilon = 2.0e-7
    p["Wmz"][2, 0] = gain
    p["Wm"][2, 1] = -gain
    p["bm"][2] = -gain * epsilon

    p["Wmz"][3, 0] = -gain
    p["Wm"][3, 1] = gain
    p["bm"][3] = -gain * epsilon

    # Do not issue consecutive resets.
    p["Wma"][2, 0] = 1.0e5

    p["bo"][:] = -10.0
    p["bo"][6] = 10.0
    p["bo"][0] = -50.0
    p["Wo"][0, 128 + 2] = -50.0
    p["Wo"][0, 128 + 3] = -50.0

    return p
