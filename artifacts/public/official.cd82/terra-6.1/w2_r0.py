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

    W1, b1 = p["W1"], p["b1"]
    W2, W3 = p["W2"], p["W3"]

    # Red-pixel extrema for estimating basket position.
    W1[0, 2, 2, 2] = 1
    W1[0, 16, 2, 2] = 1
    b1[0] = -1

    W1[1, 2, 2, 2] = 1
    W1[1, 16, 2, 2] = -1

    W1[2, 2, 2, 2] = 1
    W1[2, 17, 2, 2] = 1
    b1[2] = -1

    W1[3, 2, 2, 2] = 1
    W1[3, 17, 2, 2] = -1

    colors = [0, 15, 12, 11, 14, 8, 9]

    for i, color in enumerate(colors):
        target = 4 + i
        swatch = 11 + i
        selected = 18 + i

        # Color at a bottom-row target cell.
        W1[target, color, 0, 2] += 1
        W1[target, 5, 0, 0] += 1
        W1[target, 5, 1, 2] += 1
        W1[target, 5, 2, 2] += 1
        W1[target, 5, 3, 2] += 1
        W1[target, 4, 4, 2] += 1
        W1[target, 5, 0, 1] -= 1
        b1[target] = -5.5

        # Center of a palette swatch.
        W1[swatch, color, 2, 2] += 1
        W1[swatch, 4, 0, 2] += 1
        W1[swatch, 4, 4, 2] += 1
        W1[swatch, 4, 2, 0] += 1
        W1[swatch, 4, 2, 4] += 1
        b1[swatch] = -4.5

        # Selection underline below a swatch.
        W1[selected, color, 2, 2] += 1
        W1[selected, 4, 2, 0] += 1
        W1[selected, 4, 2, 4] += 1
        W1[selected, 4, 3, 2] += 1
        W1[selected, 0, 4, 2] += 1
        b1[selected] = -4.5

    # Coordinate ramps and a non-black indicator isolate canvas cell
    # (38, 29), which is not covered by the resting basket.
    W1[25, 16, 2, 2] = 63
    b1[25] = -38
    W1[26, 16, 2, 2] = -63
    b1[26] = 38
    W1[27, 17, 2, 2] = 63
    b1[27] = -29
    W1[28, 17, 2, 2] = -63
    b1[28] = 29
    W1[29, 0, 2, 2] = -1
    b1[29] = 1

    W2[25, 29, 1, 1] = 1
    for channel in (25, 26, 27, 28):
        W2[25, channel, 1, 1] = -1

    for i in range(4):
        W2[i, i, 1, 1] = 1
        W3[i, i] = 1

    for i in range(len(colors)):
        target = 4 + i
        swatch = 11 + i
        selected = 18 + i

        W2[target, target, 1, 1] = 2
        W2[swatch, swatch, 1, 1] = 2
        W2[selected, selected, 2, 1] = 2

        # Broadcast target color across its row.
        W3[target, 32 + target] = 1

        # Matching selected swatches have a larger value.
        W3[swatch, swatch] = 1
        W3[swatch, selected] = 0.3

    # Broadcast the upper-canvas repair signal across its row.
    W3[18, 32 + 25] = 1
    p["Uh"][14, 18] = 1
    p["Wh"][14, 14, 2, 1] = 1

    for i in range(len(colors)):
        target = 4 + i
        swatch = 11 + i
        propagation = i
        match = 7 + i

        # Move the bottom target color upwards to the palette.
        p["Wh"][propagation, propagation, 2, 1] = 1
        p["Uh"][propagation, target] = 1

        # Suppress bottom-color selection when upper repair is needed.
        p["Wh"][match, propagation, 2, 1] = 1
        p["Wh"][match, 14, 2, 1] = -2
        p["Uh"][match, swatch] = 1
        p["bh"][match] = -1.5
        p["Wc"][0, 32 + match] = 1

    # In repair mode, select the black swatch.
    p["Wh"][15, 14, 2, 1] = 1
    p["Uh"][15, 11] = 1
    p["bh"][15] = -1.5
    p["Wc"][0, 32 + 15] = 1

    centers = [
        (26.5, 31.5),
        (29.0, 41.0),
        (38.5, 43.5),
        (48.0, 41.0),
        (50.5, 31.5),
        (48.0, 22.0),
        (38.5, 19.5),
        (29.0, 22.0),
    ]

    Wz, bz = p["Wz"], p["bz"]
    for i, (cy, cx) in enumerate(centers):
        j = 4 * i

        Wz[j, 32] = 31.5
        Wz[j, 33] = -31.5
        bz[j] = 31.5 - cy

        Wz[j + 1, 32] = -31.5
        Wz[j + 1, 33] = 31.5
        bz[j + 1] = cy - 31.5

        Wz[j + 2, 34] = 31.5
        Wz[j + 2, 35] = -31.5
        bz[j + 2] = 31.5 - cx

        Wz[j + 3, 34] = -31.5
        Wz[j + 3, 35] = 31.5
        bz[j + 3] = cx - 31.5

        # Memory units 0..7: top-repair mode and basket position.
        p["bm"][i] = -6
        p["Wmz"][i, j:j + 4] = -2
        p["Wmz"][i, 100] = 10

        # Memory units 8..15: bottom-paint mode and basket position.
        p["bm"][8 + i] = 4
        p["Wmz"][8 + i, j:j + 4] = -2
        p["Wmz"][8 + i, 100] = -10

    Wz[100, 14] = 1

    # Selected-color indicators.
    for i in range(len(colors)):
        Wz[32 + i, 7 + i] = 1
        bz[32 + i] = -0.65
    Wz[39, 15] = 1
    bz[39] = -0.65

    # Routes to top position 0 or bottom position 4.
    top_routes = {
        1: [2, 3, 5, 6],
        3: [1],
        4: [4, 7],
        5: [0],
    }
    bottom_routes = {
        2: [1, 2, 6, 7],
        3: [3],
        4: [0, 5],
        5: [4],
    }

    p["bo"][:] = -10
    p["bo"][6] = 0

    for action in (1, 2, 3, 4, 5):
        units = top_routes.get(action, [])
        units = list(units) + [
            8 + i for i in bottom_routes.get(action, [])
        ]
        p["bo"][action] = len(units) - 3
        for unit in units:
            p["Wo"][action, 128 + unit] = 1
        for selected_unit in range(32, 40):
            p["Wo"][action, selected_unit] = 30

    return p
