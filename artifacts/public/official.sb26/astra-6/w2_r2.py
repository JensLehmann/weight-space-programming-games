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

    colours = (6, 8, 9, 11, 12, 14, 15)
    B = 1000.0
    M = 2048.0
    eps = 0.125

    def k1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def k2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def any_colour(out, dy, dx, value):
        for c in colours:
            k1(out, c, dy, dx, value)

    # W1:
    #  0:7   goal top-left anchors, carrying x+1
    #  7:14  selected token inner corners
    # 14:21  five-pixel horizontal frame-border tests
    for i, c in enumerate(colours):
        q = 256.0
        for channel, dy, dx in (
            (c, 0, 0), (c, 1, 0), (c, 0, 1),
            (5, -1, 0), (5, 0, -1),
        ):
            k1(i, channel, dy, dx, q)
        k1(i, 17, value=63.0)
        p["b1"][i] = 1.0 - 5.0 * q

        for channel, dy, dx in (
            (c, 0, 0), (0, -1, 0),
            (0, 0, -1), (0, -1, 2),
        ):
            k1(7 + i, channel, dy, dx)
        p["b1"][7 + i] = -3.0

        for dx in range(-2, 3):
            k1(14 + i, c, -2, dx)
        p["b1"][14 + i] = -4.0

    # Unselected inventory-token anchor.
    any_colour(21, 0, 0, 16.0)
    k1(21, 4, -1, 0, 16.0)
    k1(21, 4, 0, -1, 16.0)
    k1(21, 16, value=63.0)
    p["b1"][21] = -104.0

    # Hollow inventory token: the inner diagonal is background.
    any_colour(22, 0, 0, 16.0)
    for dy, dx in ((-1, 0), (0, -1), (1, 1)):
        k1(22, 4, dy, dx, 16.0)
    k1(22, 16, value=63.0)
    p["b1"][22] = -120.0

    # Empty-slot anchor immediately above its 2x2 marker.
    tests = (
        (2, 1, 0), (2, 1, 1),
        (2, 2, 0), (2, 2, 1),
        (4, 0, 0), (4, 1, -1), (4, 1, 2),
    )
    for c, dy, dx in tests:
        k1(23, c, dy, dx, 128.0)
    k1(23, 16, value=-63.0)
    p["b1"][23] = 53.0 - 128.0 * len(tests)

    # Frame corners encode 100-left_x and 1+right_x.
    # Both background and the entry-frame black bracket are accepted outside.
    for out, direction in ((24, 1), (25, -1)):
        any_colour(out, -2, 0, 256.0)
        any_colour(out, -2, direction, 256.0)
        any_colour(out, -1, 0, 256.0)
        for c in (0, 4):
            k1(out, c, -2, -direction, 256.0)

    k1(24, 17, value=-63.0)
    p["b1"][24] = 100.0 - 1024.0
    k1(25, 17, value=63.0)
    p["b1"][25] = 1.0 - 1024.0

    # Coordinate features.
    k1(26, 17, value=63.0)
    p["b1"][26] = -31.0
    k1(27, 17, value=63.0)
    k1(28, 16, value=63.0)

    # Two ramps whose difference gives an exact row-one indicator.
    k1(29, 16, value=-63.0)
    p["b1"][29] = 2.0
    k1(30, 16, value=-63.0)
    p["b1"][30] = 1.0

    # Source ordering: address-coloured literals first.
    # Hollow tokens receive a separate negative score later.
    priority = {14: 160.0, 9: 112.0}
    for c in colours:
        k1(31, c, value=1000.0 + priority.get(c, 0.0))
    k1(31, 4, -1, 0, 1000.0)
    k1(31, 4, 0, -1, 1000.0)
    k1(31, 16, value=6300.0)
    p["b1"][31] = -8700.0

    # W2: preserve anchors and extend horizontal border tests to seven pixels.
    for i in range(14):
        k2(i, i)

    for i in range(7):
        k2(14 + i, 14 + i, -1, -1)
        k2(14 + i, 14 + i, -1, 1)
        p["b2"][14 + i] = -1.0

    for i in (21, 22, 23, 26, 27, 28, 31):
        k2(i, i)

    k2(24, 24, -1, 0)
    k2(25, 25, -1, 0)

    # raw_empty - relu(raw_empty-1) is its binary indicator.
    k2(29, 23)
    p["b2"][29] = -1.0

    k2(30, 29)
    k2(30, 30, value=-2.0)

    # W3 uses local features and row maxima.
    # A default goal coordinate remains available for call colours absent
    # from the output string.
    for i, c in enumerate(colours):
        p["W3"][i, 32 + i] = 1.0
        p["W3"][i, 30] = B - 1.0

        # Colour 8 is the default entry-frame outline, not necessarily an
        # address association.
        if c != 8:
            p["W3"][i, 14 + i] = eps

        p["W3"][7 + i, 32 + 7 + i] = 1.0

    p["W3"][14, 23] = 1.0
    p["W3"][14, 29] = -1.0
    p["W3"][15, 21] = 1.0
    p["W3"][16, 22] = 1.0
    p["W3"][17, 27] = 1.0
    p["W3"][18, 28] = 1.0
    p["W3"][24, 31] = 1.0

    # Reserve the middle of short bodies when outer literal slots suffice.
    p["W3"][19, 32 + 24] = -1.0
    p["W3"][19, 32 + 25] = -1.0
    p["W3"][19, 26] = -2.0
    p["W3"][19, 27] = 1.0
    p["W3"][19, 23] = 200.0
    p["W3"][19, 29] = -200.0
    p["b3"][19] = 92.0 - 200.0

    # An exact integer-width window for three-slot frames.
    # With d=21-(right-left), these four ReLUs produce
    # max(80-2*x, 0) for d=0, and zero for other integer widths.
    width_scale = 128.0
    for out, shift, with_ramp in (
        (20, 0, True), (21, 0, False),
        (22, -1, True), (23, -1, False),
    ):
        p["W3"][out, 32 + 24] = -width_scale
        p["W3"][out, 32 + 25] = -width_scale
        p["b3"][out] = width_scale * (122.0 + shift)
        if with_ramp:
            p["W3"][out, 27] = -2.0
            p["b3"][out] += 80.0

    # Tied spatial block.
    # h[0:7]: selected-colour signals travelling upward.
    # h[7:14]: colour-gated goal coordinates / local frame preference.
    # h[14]: downward coordinate accumulator.
    # h[15]: positive half of the coordinate error.
    for i in range(7):
        p["Wh"][i, i, 2, 1] = 1.0
        p["Uh"][i, 7 + i] = 1.0

        p["Wh"][7 + i, i, 1, 1] = M
        p["Uh"][7 + i, i] = 1.0
        p["bh"][7 + i] = -M

        p["Wh"][14, 7 + i, 0, 1] = 1.0

    p["Wh"][14, 14, 0, 1] = 1.0
    p["Wh"][15, 14, 1, 1] = 1.0
    p["Uh"][15, 17] = -7.0 / 6.0
    p["Uh"][15, 15] = 750.0
    p["bh"][15] = 7.75 - B

    # Click score: valid glyphs only, then coordinate error and structure.
    valid_bonus = 2_000_000.0
    p["Wc"][0, 14] = valid_bonus
    p["Wc"][0, 15] = valid_bonus + 300.0
    p["Wc"][0, 16] = -200.0
    p["Wc"][0, 17] = -7.0 / 6.0
    p["Wc"][0, 18] = 0.02
    p["Wc"][0, 19] = -9.0
    p["Wc"][0, 20:24] = (1.0, -1.0, -1.0, 1.0)
    p["Wc"][0, 24] = 1.0

    for i in range(7):
        p["Wc"][0, 32 + 7 + i] = 100.0 / eps
    p["Wc"][0, 32 + 14] = 1.0
    p["Wc"][0, 32 + 15] = -2.0
    p["bc"][0] = -valid_bonus + 7.75

    # Global action selection.
    # g = [h_max, h_mean, f3_max, f3_mean].
    p["Wz"][0, 32 + 15] = 1.0
    for i in range(7):
        p["Wz"][1, 32 + 7 + i] = 1.0

    p["bo"][:] = -100.0
    p["bo"][0] = 10.0
    p["bo"][5] = 1.0
    p["bo"][6] = 0.0
    p["Wo"][6, 0] = 3.0
    p["Wo"][6, 1] = 3.0

    # One initial reset normalizes perturbed starts. Afterwards previous-action
    # presence suppresses RESET; memory never encodes a solution sequence.
    p["Wma"][0, :] = 3.0
    p["Wo"][0, 128] = -20.0

    return p
