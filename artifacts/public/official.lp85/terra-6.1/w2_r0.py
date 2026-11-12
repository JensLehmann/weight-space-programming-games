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

    # Pass colour-8 button pixels and normalized coordinates.
    p["W1"][0, 8, 2, 2] = 1
    p["W2"][0, 0, 1, 1] = 1
    p["W3"][0, 0] = 1

    p["W1"][29, 16, 2, 2] = 1
    p["W1"][30, 17, 2, 2] = 1
    p["W2"][8, 29, 1, 1] = 1
    p["W2"][9, 30, 1, 1] = 1
    p["W3"][8, 8] = 1
    p["W3"][9, 9] = 1

    # Detect whether selected rows of the left-edge step bar are colour 5.
    # Four ReLU features form two clipped half-planes for each row.
    thresholds = (4, 9, 14, 19, 24, 29, 34)
    for j, row in enumerate(thresholds):
        base = 1 + 4 * j
        for side in range(2):
            sign = 1 if side == 0 else -1
            for clipped in range(2):
                c = base + 2 * side + clipped
                p["W1"][c, 5, 2, 2] = 1
                p["W1"][c, 18, 2, 1] = -1
                p["W1"][c, 16, 2, 2] = sign * 63
                p["b1"][c] = -sign * row - clipped

        out = 1 + j
        p["W2"][out, base, 1, 1] = 2
        p["W2"][out, base + 1, 1, 1] = -2
        p["W2"][out, base + 2, 1, 1] = 2
        p["W2"][out, base + 3, 1, 1] = -2
        p["b2"][out] = -3

        # Row-max broadcasts each detected threshold across its row.
        p["W3"][out, 32 + out] = 1

        # A pair of recurrent channels spreads a saturated binary wave
        # vertically from that row.
        u, v = 2 * j, 2 * j + 1
        for dy in range(3):
            p["Wh"][u, u, dy, 1] += 1
            p["Wh"][u, v, dy, 1] -= 1
            p["Wh"][v, u, dy, 1] += 1
            p["Wh"][v, v, dy, 1] -= 1
        p["Uh"][u, out] = 1
        p["Uh"][v, out] = 1
        p["bh"][v] = -1

    # Alternating sums of threshold waves form a binary phase. Gate the
    # coordinate sum by that phase to reverse the button preference.
    for j in range(len(thresholds)):
        sign = 1 if j % 2 == 0 else -1
        u, v = 2 * j, 2 * j + 1
        p["Wh"][14, u, 1, 1] += 2 * sign
        p["Wh"][14, v, 1, 1] -= 2 * sign
    p["Uh"][14, 8] = 1
    p["Uh"][14, 9] = 1
    p["bh"][14] = -2

    p["bo"][6] = 1
    p["Wc"][0, 0] = 20
    p["Wc"][0, 8] = -1
    p["Wc"][0, 9] = -1
    p["Wc"][0, 32 + 14] = 2

    return p
