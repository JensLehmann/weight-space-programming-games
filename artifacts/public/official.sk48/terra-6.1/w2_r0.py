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

    colors = (7, 8, 9, 10, 11, 12, 13, 14)

    # Four corners identify a 4x4 coloured block.
    for k, color in enumerate(colors):
        for dy, dx in ((-1, -1), (-1, 2), (2, -1), (2, 2)):
            p["W1"][k, color, dy + 2, dx + 2] = 1
        p["b1"][k] = -3

    # Selected head: four black inner corners around a coloured centre.
    for dy, dx in ((-1, -1), (-1, 2), (2, -1), (2, 2)):
        p["W1"][8, 0, dy + 2, dx + 2] = 1
    p["W1"][8, 6, 2, 2] = 1
    p["W1"][8, 15, 2, 2] = 1
    p["b1"][8] = -4

    p["W1"][9, 0, 2, 2] = 1
    p["W1"][10, 16, 2, 2] = -63
    p["b1"][10] = 53
    p["W1"][11, 16, 2, 2] = 63
    p["b1"][11] = -52
    p["W1"][12, 16, 2, 2] = 1
    p["W1"][13, 17, 2, 2] = 1
    p["W1"][14, 1, 2, 2] = 1

    # Belt-coloured pixels immediately above a block.
    for dx in (0, 1):
        p["W1"][15, 2, 0, dx + 2] = 1
        p["W1"][15, 3, 0, dx + 2] = 1
    p["b1"][15] = -1

    # Board blocks and unmatched goal blocks.
    for k in range(8):
        p["W2"][k, k, 1, 1] = 1
        p["W2"][k, 11, 1, 1] = -1

        p["W2"][8 + k, k, 1, 1] = 1
        p["W2"][8 + k, 10, 1, 1] = -1
        p["W2"][8 + k, 9, 1, 1] = -2

    p["W2"][16, 8, 1, 1] = 1
    p["W2"][16, 11, 1, 1] = -1
    p["W2"][17, 8, 1, 1] = 1
    p["W2"][17, 10, 1, 1] = -1
    p["W2"][18, 12, 1, 1] = 1
    p["W2"][19, 13, 1, 1] = 1
    p["W2"][20, 14, 1, 1] = 1
    p["W2"][20, 11, 1, 1] = -1
    p["W2"][21, 15, 1, 1] = 1

    # At a block centre, the active horizontal belt has a colour-1
    # pixel one column to the right.
    p["W2"][22, 14, 1, 2] = 1

    # Goal flags, uncarried block rows, and uncarried block flags.
    for k in range(8):
        p["W3"][k, 8 + k] = 1

        p["W3"][8 + k, k] = 1
        p["W3"][8 + k, 18] = 1
        p["W3"][8 + k, 22] = -1
        p["b3"][8 + k] = -1

        p["W3"][16 + k, k] = 1
        p["W3"][16 + k, 22] = -1

    # Board-head row and column; goal-head column.
    p["W3"][24, 16] = 1
    p["W3"][24, 18] = 1
    p["b3"][24] = -1

    p["W3"][25, 16] = 1
    p["W3"][25, 19] = 1
    p["b3"][25] = -1

    p["W3"][26, 17] = 1
    p["W3"][26, 19] = 1
    p["b3"][26] = -1

    # Rightmost active-belt pixel and unmatched-goal columns.
    p["W3"][27, 20] = 1
    p["W3"][27, 19] = 1
    p["b3"][27] = -1

    for k in range(8):
        p["W3"][28, 8 + k] = 1
    p["W3"][28, 19] = 1
    p["b3"][28] = -1

    # Block on both the selected horizontal belt and a vertical belt.
    for k in range(8):
        p["W3"][29, k] = 1
    p["W3"][29, 21] = 1
    p["W3"][29, 22] = 1
    p["W3"][29, 32 + 16] = 1
    p["b3"][29] = -3

    # Propagate goal counts left-to-right and retain the first unmatched
    # goal's colour and column.
    for k in range(8):
        p["Uh"][0, k] = 1
    p["Wh"][0, 0, 1, 0] = 1

    for k in range(8):
        p["Uh"][1 + k, k] = 1
        p["Wh"][1 + k, 0, 1, 0] = -1

    p["Uh"][9, 28] = 1
    p["Wh"][9, 0, 1, 0] = -2

    # Move vertically towards an uncarried block of the requested colour.
    for k in range(8):
        up = 2 * k
        down = up + 1

        p["Wz"][up, 32 + 24] = 1
        p["Wz"][up, 32 + 8 + k] = -1
        p["Wz"][up, 1 + k] = 2
        p["Wz"][up, 32 + 16 + k] = 2
        p["bz"][up] = -4.04

        p["Wz"][down, 32 + 8 + k] = 1
        p["Wz"][down, 32 + 24] = -1
        p["Wz"][down, 1 + k] = 2
        p["Wz"][down, 32 + 16 + k] = 2
        p["bz"][down] = -4.04

    p["Wz"][16, 9] = 1

    # Difference of these ReLUs is a clipped excess-length indicator.
    for unit in (17, 18):
        p["Wz"][unit, 32 + 27] = 63
        p["Wz"][unit, 32 + 25] = -63
        p["Wz"][unit, 9] = -63
        p["Wz"][unit, 32 + 26] = 63
    p["bz"][17] = 3
    p["bz"][18] = 2

    p["Wz"][20, 32 + 29] = 1

    # Memory 0 stores the previous first-unmatched-goal column.
    p["Wmz"][0, 16] = 0.1

    # Memory 1 is a retract-mode latch. A goal-column increase starts it;
    # it remains on while there is excess belt and turns off at the
    # matched-prefix length. Previous-action input suppresses the initial
    # comparison against zero memory.
    p["Wmz"][1, 16] = 50
    p["Wm"][1, 0] = -500
    p["Wm"][1, 1] = 3
    p["Wmz"][1, 17] = 6
    p["Wmz"][1, 18] = -6
    p["Wma"][1, :] = 100
    p["bm"][1] = -106.5

    for k in range(8):
        p["Wo"][1, 2 * k] = 100
        p["Wo"][2, 2 * k + 1] = 100

    # Extend by default; retract while the latch is on.
    p["bo"][4] = 1
    p["Wo"][3, 128 + 1] = 70
    p["bo"][3] = -30

    # Prefer moving down when carrying a block at a vertical crossing.
    p["Wo"][2, 20] = 200

    return p
