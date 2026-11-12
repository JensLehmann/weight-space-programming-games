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
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }

    backgrounds = [1, 2, 5, 6, 9, 10, 11, 12, 14, 15]
    body_colours = [0] + backgrounds
    bg10 = backgrounds.index(10)

    for i, colour in enumerate(backgrounds):
        p["W1"][i, colour, 2, 2] = 1.0

    for i, colour in enumerate(body_colours):
        p["W1"][10 + i, colour, :, :] = 1.0 / 25.0

    p["W1"][21, 8, 2, 2] = 1.0
    p["W1"][22, 3, 2, 2] = 1.0

    for i, threshold in enumerate((2.0, 3.0, 4.0)):
        p["W1"][23 + i, 17, 2, 2] = 63.0
        p["b1"][23 + i] = -threshold
        p["W1"][26 + i, 16, 2, 2] = 63.0
        p["b1"][26 + i] = -threshold

    p["W1"][29, 16, 2, 2] = 1.0
    p["W1"][30, 17, 2, 2] = 1.0
    p["W1"][31, 4, 2, 2] = 1.0

    # Detect the background at the left edge.
    for i in range(len(backgrounds)):
        p["W2"][i, i, 1, 1] = 1.0
        p["W2"][i, 30, 1, 1] = -63.0

    for i in range(len(body_colours)):
        p["W2"][10 + i, 10 + i, 1, 1] = 1.0

    p["W2"][21, 21, 1, 1] = 1.0
    p["W2"][22, 22, 1, 1] = 1.0

    for dst, src in ((23, 23), (24, 26)):
        p["W2"][dst, src, 1, 1] = 2.0
        p["W2"][dst, src + 1, 1, 1] = -4.0
        p["W2"][dst, src + 2, 1, 1] = 2.0
        p["b2"][dst] = -1.0

    p["W2"][25, 29, 1, 1] = 1.0
    p["W2"][26, 30, 1, 1] = 1.0

    # A grey pixel with another grey pixel below it excludes the
    # one-pixel-high action counter.
    p["W2"][31, 31, 1, 1] = 1.0
    p["W2"][31, 31, 2, 1] = 1.0
    p["b2"][31] = -1.0
    p["W2"][27, 31, 1, 1] = 1.0
    p["W2"][27, 30, 1, 1] = -63.0

    p["W3"][0, 21] = 1.0
    p["W3"][1, 22] = 1.0

    # Candidate selected-body colours, excluding the background.
    for i, colour in enumerate(body_colours):
        out = 2 + i
        p["W3"][out, 10 + i] = 1.0
        if colour != 0:
            bg_index = backgrounds.index(colour)
            p["W3"][out, 32 + bg_index] -= 1.0
            p["W3"][out, 32 + bg10] -= 1.0

    p["W3"][13, 23] = 1.0
    p["W3"][14, 24] = 1.0

    # Blue/green-marker-gated coordinates.
    for out in (15, 16, 17, 18):
        p["W3"][out, 21] = 1.0
        p["W3"][out, 22] = 1.0
    p["W3"][15, 25] = 1.0
    p["b3"][15] = -1.0
    p["W3"][16, 25] = -1.0
    p["W3"][17, 26] = 1.0
    p["b3"][17] = -1.0
    p["W3"][18, 26] = -1.0

    # Grey-piece click candidates, excluding a grey background.
    p["W3"][24, 31] = 1.0
    p["W3"][24, 32 + 27] = -1.0
    p["Wc"][0, 24] = 1.0

    # Period-three signals identify display cell centers.
    p["Uh"][0, 13] = 1.0
    p["Wh"][0, 2, 1, 0] = 1.0
    p["Wh"][1, 0, 1, 0] = 1.0
    p["Wh"][2, 1, 1, 0] = 1.0

    p["Uh"][3, 14] = 1.0
    p["Wh"][3, 5, 0, 1] = 1.0
    p["Wh"][4, 3, 0, 1] = 1.0
    p["Wh"][5, 4, 0, 1] = 1.0

    # H[6] - H[7] is a saturated selected-body-nearby signal.
    for channel in range(2, 13):
        p["Uh"][6, channel] = 25.0
        p["Uh"][7, channel] = 25.0
    p["bh"][7] = -1.0

    # Source and target marker coordinate extremes. Green markers
    # can contribute to both sets.
    for k in range(4):
        source = 8 + k
        target = 12 + k
        coord = 15 + k

        p["Uh"][source, coord] = 1.0
        p["bh"][source] = -3.0
        p["Wh"][source, 0, 1, 1] = 1.0
        p["Wh"][source, 3, 1, 1] = 1.0
        p["Wh"][source, 6, 1, 1] = 1.0
        p["Wh"][source, 7, 1, 1] = -1.0

        p["Uh"][target, coord] = 1.0
        p["Uh"][target, 1] = 1.0
        p["bh"][target] = -2.0
        p["Wh"][target, 0, 1, 1] = 1.0
        p["Wh"][target, 3, 1, 1] = 1.0
        p["Wh"][target, 6, 1, 1] = -1.0
        p["Wh"][target, 7, 1, 1] = 1.0

    # g[56] is the maximum of F3 channel 24.
    mask_index = 32 + 24

    def center(unit, high, inverse_low, mask_weight=0.0):
        p["Wz"][unit, high] = 0.5
        p["Wz"][unit, inverse_low] = -0.5
        p["Wz"][unit, mask_index] = mask_weight
        p["bz"][unit] = 0.5

    def span(unit, high, inverse_low, mask_weight=0.0):
        p["Wz"][unit, high] = 1.0
        p["Wz"][unit, inverse_low] = 1.0
        p["Wz"][unit, mask_index] = mask_weight
        p["bz"][unit] = -1.0

    # Unmasked: store the other visible markers.
    center(8, 12, 13, -1.0)
    center(9, 14, 15, -1.0)
    span(10, 12, 13, -1.0)
    span(11, 14, 15, -1.0)

    # Current selected-marker measurements.
    span(12, 8, 9)
    span(13, 10, 11)
    center(14, 8, 9)
    center(15, 10, 11)

    # Masked: store the initially selected markers before clicking
    # another piece.
    center(18, 8, 9, 1.0)
    center(19, 10, 11, 1.0)
    p["bz"][18] -= 1.0
    p["bz"][19] -= 1.0
    span(20, 8, 9, 1.0)
    span(21, 10, 11, 1.0)
    p["bz"][20] -= 1.0
    p["bz"][21] -= 1.0

    p["Wz"][16, 64 + 1] = 1.0
    p["bz"][16] = -0.0033
    p["Wz"][17, mask_index] = 1.0

    # Thermometer-coded latched target centers.
    for axis, z_index in enumerate((8, 9)):
        for i in range(20):
            m = axis * 20 + i
            threshold = (i + 0.5) / 20.0
            p["Wm"][m, m] = 6.0
            p["Wmz"][m, z_index] = 2.0
            p["Wmz"][m, 18 + axis] = 2.0
            p["bm"][m] = -2.0 * threshold

    grid_step = 3.0 / 63.0
    for axis, z_index in enumerate((10, 11)):
        for i in range(6):
            m = 40 + axis * 6 + i
            threshold = (i + 0.5) * grid_step
            p["Wm"][m, m] = 6.0
            p["Wmz"][m, z_index] = 2.0
            p["Wmz"][m, 20 + axis] = 2.0
            p["bm"][m] = -2.0 * threshold

    half_step = grid_step / 2.0
    for axis in range(2):
        source_z = 12 + axis
        bit_start = 40 + axis * 6
        positive = 52 + axis * 2
        negative = positive + 1

        p["Wmz"][positive, source_z] = 40.0
        p["Wmz"][negative, source_z] = -40.0
        for i in range(6):
            p["Wm"][positive, bit_start + i] = -40.0 * half_step
            p["Wm"][negative, bit_start + i] = 40.0 * half_step
        p["bm"][positive] = -40.0 * 6.0 * half_step - 1.5
        p["bm"][negative] = 40.0 * 6.0 * half_step - 1.5

    # Two-step rotation pulse for a full green overlap.
    p["Wmz"][56, 16] = 1000.0
    p["Wm"][57, 56] = 3.0

    # One initial click on a grey piece.
    p["Wmz"][58, 17] = 5.0
    p["Wm"][59, 58] = 5.0

    p["bo"][:] = -100.0

    for m in range(20):
        p["Wo"][2, 128 + m] = 1.0 / 40.0
        p["Wo"][1, 128 + m] = -1.0 / 40.0
    p["Wo"][2, 14] = -1.0
    p["Wo"][1, 14] = 1.0
    p["bo"][2] = 0.5 - 0.025
    p["bo"][1] = -0.5 - 0.025

    for m in range(20, 40):
        p["Wo"][4, 128 + m] = 1.0 / 40.0
        p["Wo"][3, 128 + m] = -1.0 / 40.0
    p["Wo"][4, 15] = -1.0
    p["Wo"][3, 15] = 1.0
    p["bo"][4] = 0.5 - 0.025
    p["bo"][3] = -0.5 - 0.025

    p["bo"][5] = 3.6
    for m in range(52, 56):
        p["Wo"][5, 128 + m] = 1.0
    p["Wo"][5, 128 + 56] = 20.0
    p["Wo"][5, 128 + 57] = 20.0

    p["bo"][6] = -20.0
    p["Wo"][6, 128 + 58] = 50.0
    p["Wo"][6, 128 + 59] = -50.0

    return p
