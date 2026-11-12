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

    colors = (8, 9, 11, 14)

    # W1: paired goal detectors, selected-color detectors, black
    # selection pixel, and row/column coordinates.
    for i, color in enumerate(colors):
        for part in range(2):
            unit = 2 * i + part
            p["W1"][unit, color, 2, 2] = 10.0
            for dy in (-1, 1):
                for dx in (-1, 1):
                    p["W1"][unit, 4, 2 + dy, 2 + dx] = 1.0
            p["b1"][unit] = -11.5 - 0.5 * part

        unit = 8 + i
        p["W1"][unit, 0, 2, 2] = 32.0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx or dy:
                    p["W1"][unit, color, 2 + dy, 2 + dx] = 1.0
        p["b1"][unit] = -32.0

    p["W1"][12, 0, 2, 2] = 1.0
    p["W1"][13, 16, 2, 2] = 1.0
    p["W1"][14, 17, 2, 2] = 1.0
    p["W1"][15, 18, 2, 2] = 1.0
    p["W1"][15, 16, 2, 2] = -1.0
    p["W1"][16, 18, 2, 2] = 1.0
    p["W1"][16, 17, 2, 2] = -1.0

    # W2: goal maps weighted by row, inverse row, column, and
    # inverse column. The two W1 goal detectors differ by 0.5
    # at a valid goal, yielding a binary mask after scaling.
    for i in range(4):
        for j, coord in enumerate((13, 15, 14, 16)):
            out = 4 * i + j
            p["W2"][out, 2 * i, 1, 1] = 2.0
            p["W2"][out, 2 * i + 1, 1, 1] = -2.0
            p["W2"][out, coord, 1, 1] = 1.0
            p["b2"][out] = -1.0

        # Two ReLUs whose difference clamps the selected-color
        # evidence to one.
        p["W2"][16 + 2 * i, 8 + i, 1, 1] = 1.0
        p["W2"][17 + 2 * i, 8 + i, 1, 1] = 1.0
        p["b2"][17 + 2 * i] = -1.0

    p["W2"][24, 12, 1, 1] = 1.0
    p["W2"][25, 13, 1, 1] = 1.0
    p["W2"][26, 14, 1, 1] = 1.0

    # W3:
    # 0..3: evidence for two same-color goals on a row.
    # 4..7: evidence for two same-color goals on a column.
    # 8..11: selected-color flags.
    # 12,13: selected row and column.
    # 14,15: unrestricted row and column coordinates.
    for i in range(4):
        p["W3"][i, 32 + 4 * i + 2] = 1.0
        p["W3"][i, 32 + 4 * i + 3] = 1.0
        p["b3"][i] = -1.0

        p["W3"][4 + i, 64 + 4 * i] = 1.0
        p["W3"][4 + i, 64 + 4 * i + 1] = 1.0
        p["b3"][4 + i] = -1.0

        p["W3"][8 + i, 16 + 2 * i] = 1.0
        p["W3"][8 + i, 17 + 2 * i] = -1.0

    p["W3"][12, 24] = 1.0
    p["W3"][12, 25] = 1.0
    p["b3"][12] = -1.0

    p["W3"][13, 24] = 1.0
    p["W3"][13, 26] = 1.0
    p["b3"][13] = -1.0

    p["W3"][14, 25] = 1.0
    p["W3"][15, 26] = 1.0

    # H: attach a coordinate to the strongest pair evidence.
    # A one-pixel difference in pair separation dominates the
    # coordinate term. Wh is zero, so the tied block is stable.
    for i in range(4):
        p["Uh"][i, i] = 128.0
        p["Uh"][i, 14] = 1.0
        p["bh"][i] = -1.0

        p["Uh"][4 + i, 4 + i] = 128.0
        p["Uh"][4 + i, 15] = 1.0
        p["bh"][4 + i] = -1.0

    # Reconstruct target coordinates from h maxima and pair maxima.
    # g[0:16] contains h maxima; g[32:64] contains f3 maxima.
    for i in range(4):
        p["Wz"][i, i] = 1.0
        p["Wz"][i, 32 + i] = -128.0
        p["bz"][i] = 1.0

        p["Wz"][4 + i, 4 + i] = 1.0
        p["Wz"][4 + i, 32 + 4 + i] = -128.0
        p["bz"][4 + i] = 1.0

        # Clamp positive pair evidence to a binary availability flag.
        for axis in range(2):
            pair_channel = i + 4 * axis
            z0 = 8 + 4 * i + 2 * axis
            p["Wz"][z0, 32 + pair_channel] = 128.0
            p["Wz"][z0 + 1, 32 + pair_channel] = 128.0
            p["bz"][z0 + 1] = -1.0

        p["Wz"][24 + i, 32 + 8 + i] = 1.0

    p["Wz"][28, 32 + 12] = 1.0
    p["Wz"][29, 32 + 13] = 1.0

    # Four directional units per color. A unit is active only for
    # the selected color and when its required marker pair exists.
    for i in range(4):
        for direction in range(4):
            unit = 4 * i + direction
            vertical = direction < 2
            sign = -1.0 if direction in (0, 2) else 1.0

            target = i if vertical else 4 + i
            current = 28 if vertical else 29
            pair_z = 8 + 4 * i + (0 if vertical else 2)

            p["Wmz"][unit, target] += 36.0 * sign
            p["Wmz"][unit, current] -= 36.0 * sign

            p["Wmz"][unit, 24 + i] += 40.0
            p["Wmz"][unit, pair_z] += 40.0
            p["Wmz"][unit, pair_z + 1] -= 40.0
            p["bm"][unit] -= 80.0

            p["Wo"][1 + direction, 128 + unit] = 1.0

    # Compensate for the three inactive color gates. Switch sprites
    # when all active movement scores are small.
    for action in range(1, 5):
        p["bo"][action] = 3.0
    p["bo"][5] = 0.55

    return p
