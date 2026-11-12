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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    support = (1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15)

    # Visible support and the player.
    for color in support:
        p["W1"][0, color, 2, 2] = 1.0
    p["W1"][1, 14, 2, 2] = 1.0

    # Top-left of an isolated 2x2 goal-colored square.
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        p["W1"][2, 11, 2 + dy, 2 + dx] = 1.0
    for dy, dx in ((-1, 0), (0, -1), (2, 0), (0, 2)):
        p["W1"][2, 11, 2 + dy, 2 + dx] = -4.0
    p["b1"][2] = -3.0

    # Player pixels shifted to each destination.
    for i, (dy, dx) in enumerate(dirs):
        p["W1"][3 + i, 14, 2 - 2 * dy, 2 - 2 * dx] = 1.0

    # Complete transparent cells of yellow and pink checkerboards.
    for color, first in ((9, 7), (15, 9)):
        for ch, bias in ((first, -5.0), (first + 1, -6.0)):
            p["W1"][ch, 4, 2, 2] = 4.0
            for dy, dx in dirs:
                p["W1"][ch, color, 2 + dy, 2 + dx] = 1.0
            p["b1"][ch] = bias

    # Solid control patches.
    for ch, color in ((11, 9), (12, 8)):
        p["W1"][ch, color, :, :] = 1.0
        p["b1"][ch] = -24.0

    p["W1"][13, 13, 2, 2] = 1.0

    # Visible half of a yellow checkerboard.
    for ch, bias in ((14, -5.0), (15, -6.0)):
        p["W1"][ch, 9, 2, 2] = 5.0
        for dy, dx in dirs:
            p["W1"][ch, 4, 2 + dy, 2 + dx] = 1.0
        p["b1"][ch] = bias

    # Pivot pixels within a 5x5 neighborhood, clipped by subtraction.
    p["W1"][16, 13, :, :] = 1.0
    p["W1"][17, 13, :, :] = 1.0
    p["b1"][17] = -1.0

    # Optimistic support for a 2x2 anchor.
    for dy in (0, 1):
        for dx in (0, 1):
            y, x = 1 + dy, 1 + dx
            p["W2"][0, 0, y, x] = 1.0
            p["W2"][0, 7, y, x] = 1.0
            p["W2"][0, 8, y, x] = -1.0
            p["W2"][0, 9, y, x] = 1.0
            p["W2"][0, 10, y, x] = -1.0
    p["b2"][0] = -3.0
    p["W2"][1, 2, 1, 1] = 1.0

    # Candidate anchors and currently usable destination support.
    for i in range(4):
        src = 3 + i
        for out, scale in ((2 + i, 1.0),
                           (6 + i, 5.0),
                           (13 + i, 4.0)):
            p["W2"][out, src, 1, 1] = scale
            p["W2"][out, src, 0, 1] = -scale
            p["W2"][out, src, 1, 0] = -scale

        for dy in (0, 1):
            for dx in (0, 1):
                y, x = 1 + dy, 1 + dx
                p["W2"][6 + i, 0, y, x] = 1.0

                # A checkerboard is currently blocking.
                for positive, negative in ((14, 15), (7, 8)):
                    p["W2"][6 + i, positive, y, x] -= 5.0
                    p["W2"][6 + i, negative, y, x] += 5.0
                    p["W2"][13 + i, positive, y, x] += 1.0
                    p["W2"][13 + i, negative, y, x] -= 1.0

        p["b2"][6 + i] = -8.0
        p["b2"][13 + i] = -4.0

    p["W2"][10, 13, 1, 1] = 1.0
    p["W2"][11, 12, 1, 1] = 1.0
    p["W2"][12, 11, 1, 1] = 1.0

    # Pivot close to the current player anchor.
    p["W2"][17, 1, 1, 1] = 1.0
    p["W2"][17, 1, 0, 1] = -1.0
    p["W2"][17, 1, 1, 0] = -1.0
    p["W2"][17, 16, 1, 1] = 1.0
    p["W2"][17, 17, 1, 1] = -1.0
    p["b2"][17] = -1.0

    for ch in range(13):
        p["W3"][ch, ch] = 1.0

    # Pivot row/column union, represented as two ReLU channels.
    for ch in (13, 14):
        p["W3"][ch, 32 + 10] = 1.0
        p["W3"][ch, 64 + 10] = 1.0
    p["b3"][14] = -1.0

    for i in range(4):
        p["W3"][15, 13 + i] = 1.0
    p["W3"][16, 17] = 1.0

    # h[0] - h[1] is a clipped goal field. Channels 2:8
    # implement the maximum of the four neighboring values.
    def field_weight(out, dy, dx, weight):
        p["Wh"][out, 0, 1 + dy, 1 + dx] += weight
        p["Wh"][out, 1, 1 + dy, 1 + dx] -= weight

    field_weight(2, -1, 0, 1.0)
    field_weight(2, 1, 0, -1.0)
    field_weight(3, 1, 0, 1.0)
    field_weight(4, 0, -1, 1.0)
    field_weight(4, 0, 1, -1.0)
    field_weight(5, 0, 1, 1.0)

    p["Wh"][6, 2, 1, 1] = 1.0
    p["Wh"][6, 3, 1, 1] = 1.0
    p["Wh"][6, 4, 1, 1] = -1.0
    p["Wh"][6, 5, 1, 1] = -1.0
    p["Wh"][7, 4, 1, 1] = 1.0
    p["Wh"][7, 5, 1, 1] = 1.0

    for out in (0, 1):
        p["Wh"][out, 6, 1, 1] = 0.98
        p["Wh"][out, 7, 1, 1] = 0.98
        p["Wh"][out, 12, 1, 1] = 1.0
        p["Wh"][out, 13, 1, 1] = -1.0
        p["Uh"][out, 0] = 1.0
        p["Uh"][out, 1] = 1.0
    p["bh"][0] = -1.0
    p["bh"][1] = -2.0

    # Read the field only at candidate destinations.
    for i in range(4):
        p["Wh"][8 + i, 0, 1, 1] = 1.0
        p["Wh"][8 + i, 1, 1, 1] = -1.0
        p["Uh"][8 + i, 2 + i] = 2.0
        p["bh"][8 + i] = -2.0

    # A bounded-distance signal around pivots.
    for dy, dx in dirs:
        p["Wh"][11, 11, 1 + dy, 1 + dx] = 0.24
    p["Uh"][11, 10] = 1.0

    # Virtual bridge support, restricted to pivot rows/columns
    # near a pivot and excluded where actual support exists.
    for out, bias in ((12, -10200.0), (13, -10201.0)):
        p["Wh"][out, 11, 1, 1] = 1000.0
        p["Uh"][out, 13] = 10000.0
        p["Uh"][out, 14] = -10000.0
        p["Uh"][out, 0] = -10000.0
        p["bh"][out] = bias

    # Carry nearby-checker information toward the UI.
    p["Wh"][14, 14, 1, 0] = 0.99
    p["Uh"][14, 15] = 1.0
    p["Wh"][15, 15, 0, 1] = 0.48
    p["Wh"][15, 15, 2, 1] = 0.48
    p["Wh"][15, 14, 1, 1] = 1.0

    # Legal and blocked portions of each directional value.
    for i in range(4):
        p["Wz"][i, 8 + i] = 1.0
        p["Wz"][i, 32 + 6 + i] = 1.0
        p["bz"][i] = -1.0

        p["Wz"][4 + i, 8 + i] = 1.0
        p["Wz"][4 + i, 32 + 6 + i] = -1.0

    # Clip the nearby-checker signal; also read pivot proximity.
    p["Wz"][8, 32 + 15] = 1.0
    p["Wz"][9, 32 + 15] = 1.0
    p["bz"][9] = -1.0
    p["Wz"][10, 32 + 16] = 1.0

    for i in range(4):
        p["Wo"][1 + i, i] = 1.0
        p["Wo"][6, 4 + i] = 0.7

    # Permit clicks when a checker or pivot is nearby.
    p["Wo"][6, 8] = 1.0
    p["Wo"][6, 9] = -1.0
    p["Wo"][6, 10] = 1.0
    p["bo"][6] = -1.0

    # Discourage immediate reversals.
    for i in range(4):
        p["Wma"][i, 1 + i] = 1.0
    for i, opposite in enumerate((1, 0, 3, 2)):
        p["Wo"][1 + opposite, 128 + i] = -0.005

    p["bo"][0] = -1.0
    p["bo"][5] = -1.0
    p["bo"][7] = -1.0

    # Restrict the click argmax to detected controls. Nearby yellow
    # checkers bias it toward yellow; otherwise blue wins ties.
    p["Wc"][0, 11] = 10000.1
    p["Wc"][0, 12] = 10000.0
    p["Wc"][0, 32 + 15] = 1.0

    return p
