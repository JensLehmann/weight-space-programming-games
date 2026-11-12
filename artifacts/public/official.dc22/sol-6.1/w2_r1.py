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

    MASK = np.float32(2.0 ** 100)
    FLOOR_MASK = np.float32(1024.0)
    BUTTON_BLOCK = np.float32(2.0 ** 30)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def c1(o, i, v, dy=0, dx=0):
        assert -2 <= dy <= 2 and -2 <= dx <= 2
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, v, dy=0, dx=0):
        assert -1 <= dy <= 1 and -1 <= dx <= 1
        p["W2"][o, i, dy + 1, dx + 1] += v

    def ch(o, i, v, dy=0, dx=0):
        assert -1 <= dy <= 1 and -1 <= dx <= 1
        p["Wh"][o, i, dy + 1, dx + 1] += v

    def linear3(o, terms, bias=0.0):
        for i, v in terms.items():
            p["W3"][o, i] += v
        p["b3"][o] = bias

    def footprint(o, colour, weight=1.0, dy=0, dx=0):
        for yy in (0, 1):
            for xx in (0, 1):
                c1(o, colour, weight, dy + yy, dx + xx)

    # ------------------------------------------------------------
    # First convolution: footprint and sprite predicates.
    # ------------------------------------------------------------
    p["b1"][0] = 1.0

    c1(1, 17, 63.0)
    p["b1"][1] = -38.0
    c1(2, 17, -63.0)
    p["b1"][2] = 40.0

    for colour in (1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15):
        footprint(3, colour)
    p["b1"][3] = -3.0

    # Saturated orange and magenta footprint occupancy.
    for low, high, colour in ((4, 5, 9), (6, 7, 15)):
        footprint(low, colour)
        footprint(high, colour)
        p["b1"][low] = -1.0
        p["b1"][high] = -2.0

    # Top-left anchors of the player and goal.
    for out, colour in ((8, 14), (9, 11)):
        c1(out, colour, 1.0)
        c1(out, colour, -1.0, -1, 0)
        c1(out, colour, -1.0, 0, -1)

    # Magenta unlock token.
    c1(10, 15, 1.0, 0, 1)
    c1(10, 15, 1.0, 1, 0)
    c1(10, 15, 1.0, 1, 1)
    c1(10, 15, -1.0)
    p["b1"][10] = -2.0

    footprint(11, 6)
    footprint(11, 7)
    p["b1"][11] = -3.0

    # Four edges of a blue bridge's pivot.
    for out in (12, 13, 14, 15):
        footprint(out, 13)
        p["b1"][out] = -3.0
    c1(12, 13, -1.0, -1, 0)
    c1(13, 13, -1.0, 0, -1)
    c1(14, 13, -1.0, 2, 0)
    c1(15, 13, -1.0, 0, 2)

    # A positive-side blue stem identifies a two-sided bridge.
    c1(16, 13, 1.0)
    c1(16, 8, 1.0, 0, 2)
    c1(16, 8, 1.0, 2, 0)
    p["b1"][16] = -1.0

    # Wide button interiors, excluding small crusher keys.
    for out, colour in ((17, 8), (18, 9), (19, 15), (20, 6)):
        for dx in range(-2, 3):
            c1(out, colour, 1.0, 0, dx)
        p["b1"][out] = -4.0

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(21, 14, 0.25, dy, dx)

    # Red-platform neighbours. The positive-side tests inspect
    # the adjacent edge rather than an out-of-kernel footprint.
    footprint(23, 2, dx=-2)
    p["b1"][23] = -3.0

    c1(24, 2, 1.0, 0, 2)
    c1(24, 2, 1.0, 1, 2)
    p["b1"][24] = -1.0

    footprint(25, 2, dy=-2)
    p["b1"][25] = -3.0

    c1(26, 2, 1.0, 2, 0)
    c1(26, 2, 1.0, 2, 1)
    p["b1"][26] = -1.0

    # ------------------------------------------------------------
    # Second convolution: panel exclusion and shifted predicates.
    # ------------------------------------------------------------
    c2(0, 0, 1.0)
    c2(1, 3, 1.0)
    c2(1, 1, -16.0)

    for out, low, high in ((2, 4, 5), (3, 6, 7)):
        c2(out, low, 1.0)
        c2(out, high, -1.0)
        c2(out, 1, -16.0)

    c2(4, 8, 1.0)

    for out, inp in ((5, 9), (6, 10), (7, 11)):
        c2(out, inp, 1.0)
        c2(out, 1, -16.0)

    for out, inp in (
        (8, 12), (9, 13), (10, 14), (11, 15), (12, 16)
    ):
        c2(out, inp, 1.0)
        c2(out, 1, -16.0)

    for out, inp in ((13, 17), (14, 18), (15, 19), (16, 20)):
        c2(out, inp, 1.0)
        c2(out, 2, -16.0)

    for i, (dy, dx) in enumerate(directions):
        # Midpoint between player and two-pixel destination.
        c2(17 + i, 8, 1.0, -dy, -dx)
        c2(21 + i, 3, 1.0, dy, dx)
        c2(21 + i, 1, -16.0, dy, dx)

    # Local requests from checkerboards, not solid bridges.
    for out, low, high in ((25, 4, 5), (26, 6, 7)):
        c2(out, low, 1.0)
        c2(out, high, -1.0)
        c2(out, 3, -1.0)
        c2(out, 21, 1.0)
        c2(out, 1, -16.0)
        p["b2"][out] = -1.0

    # A covered platform corner can indicate a covered portal.
    c2(27, 8, 8.0)
    for inp in (23, 24, 25, 26):
        c2(27, inp, 1.0)
    p["b2"][27] = -9.0

    for dy, dx in directions:
        for low, high in ((4, 5), (6, 7)):
            c2(27, low, -2.0, dy, dx)
            c2(27, high, 2.0, dy, dx)

    # ------------------------------------------------------------
    # Axial layer: geometry, buttons and action-validity masks.
    # ------------------------------------------------------------
    for out, inp in (
        (0, 1), (1, 2), (2, 3),
        (3, 5), (4, 6), (5, 7)
    ):
        linear3(out, {inp: 1.0})

    # Negative rays extend six pixels for one-sided bridges and
    # eight pixels for two-sided bridges.
    linear3(6, {8: 10.0, 32 + 12: 2.0}, -2.0)
    linear3(7, {9: 10.0, 64 + 12: 2.0}, -2.0)

    # Positive rays exist only for two-sided bridges.
    linear3(8, {10: 2.0, 32 + 12: 1.0}, -2.0)
    linear3(9, {11: 2.0, 64 + 12: 1.0}, -2.0)

    for out, inp in ((10, 13), (11, 14), (12, 15), (13, 16)):
        linear3(out, {inp: 1.0})

    for i in range(4):
        near = 17 + i
        destination = 21 + i
        linear3(14 + i, {near: 1.0})
        linear3(18 + i, {near: 1.0, destination: -1.0})
        linear3(22 + i, {near: 1.0, destination: 1.0}, -1.0)

    linear3(26, {25: 1.0})
    linear3(27, {26: 1.0})
    linear3(28, {4: 1.0})
    linear3(29, {27: 1.0})

    # One away from buttons; zero on a recognized button.
    linear3(30, {
        13: -1.0, 14: -1.0, 15: -1.0, 16: -1.0
    }, 1.0)
    linear3(31, {0: 1.0})

    # ------------------------------------------------------------
    # Recurrent spatial computation.
    #
    # 0/1: raw target potential / blocked-cell copy
    # 2..5: finite bridge rays
    # 6..9: directional potential samples
    # 10..13: colour-specific button fields
    # 14/15: magenta permission field / gated permission
    # ------------------------------------------------------------
    for channel, seed, dy, dx, strength in (
        (2, 6, 1, 0, 1.0),
        (3, 7, 0, 1, 1.0),
        (4, 8, -1, 0, 10.0),
        (5, 9, 0, -1, 10.0),
    ):
        ch(channel, channel, 1.0, dy, dx)
        p["Uh"][channel, seed] = strength
        p["bh"][channel] = -1.0

    # Effective potential is h[0] - h[1].
    for dy, dx in directions:
        ch(0, 0, 0.24, dy, dx)
        ch(0, 1, -0.24, dy, dx)

    p["Uh"][0, 3] = 1.0
    p["Uh"][0, 4] = 10.0
    p["Uh"][0, 5] = 1.0e-12

    p["Wh"][1] = p["Wh"][0]
    p["Uh"][1] = p["Uh"][0]

    # Identical values cancel at blocked cells. Planned support
    # suppresses the blocked copy without subtracting huge values
    # from the useful potential.
    p["Uh"][1, 0] -= FLOOR_MASK
    p["Uh"][1, 1] -= FLOOR_MASK
    for ray in range(2, 6):
        ch(1, ray, -FLOOR_MASK)
    ch(1, 15, -FLOOR_MASK)

    for i, (dy, dx) in enumerate(directions):
        ch(6 + i, 0, 1.0, dy, dx)
        ch(6 + i, 1, -1.0, dy, dx)
        p["Uh"][6 + i, 14 + i] = MASK
        p["bh"][6 + i] = -MASK

    # Each field is clamped on buttons of other colours.
    button_features = (10, 11, 12, 13)
    for channel, matching_button in zip(range(10, 14), button_features):
        for dy, dx in directions:
            ch(channel, channel, 0.249, dy, dx)
        ch(channel, channel, 0.003)
        for button in button_features:
            if button != matching_button:
                p["Uh"][channel, button] = -BUTTON_BLOCK

    p["Uh"][10, 28] = 1.0e-3
    p["Uh"][11, 26] = 1.0
    p["Uh"][12, 27] = 1.0
    p["Uh"][13, 29] = 1.0
    p["Uh"][13, 28] = 1.0e-9

    # Visible magenta buttons grant relaxed traversal permission.
    for dy, dx in directions:
        ch(14, 14, 0.3, dy, dx)
    p["Uh"][14, 12] = 1.0

    ch(15, 14, 1.0)
    p["Uh"][15, 2] = MASK
    p["bh"][15] = -MASK

    # ------------------------------------------------------------
    # Pooled action readout and memory.
    # ------------------------------------------------------------
    for i in range(4):
        p["Wz"][i, 6 + i] = 1.0
        p["Wz"][i, 32 + 18 + i] = -MASK

        p["Wz"][4 + i, 6 + i] = 1.0
        p["Wz"][4 + i, 32 + 22 + i] = -MASK

    # Two portal anchors are normally visible. Covering one with
    # the player leaves one, requesting a portal-button click.
    p["Wz"][8, 32 + 13] = 10.0
    p["Wz"][8, 64 + 5] = -4096.0
    p["bz"][8] = -8.0

    p["Wma"][0, 6] = 100.0

    for i, action in enumerate((1, 2, 3, 4)):
        p["Wo"][action, i] = 1.0

    p["Wo"][6, 4:8] = 0.99
    p["Wo"][6, 8] = 1024.0
    p["Wo"][6, 128] = -1024.0

    p["bo"][0] = -4096.0
    p["bo"][5] = -4096.0
    p["bo"][7] = -4096.0

    p["Wc"][0, 30] = -MASK
    for channel in range(10, 14):
        p["Wc"][0, 32 + channel] = 1.0

    return p
