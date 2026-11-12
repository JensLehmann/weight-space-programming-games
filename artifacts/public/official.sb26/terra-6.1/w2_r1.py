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
    p = {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}

    colors = (6, 8, 9, 11, 12, 14, 15)
    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]

    # Priority of each color's first occurrence in the target.
    for i, color in enumerate(colors):
        W1[i, color, 2, 2] = 1000
        W1[i, 16, 2, 2] = -63
        W1[i, 17, 2, 2] = -0.25
        b1[i] = -998.5
        W2[i, i, 1, 1] = 1

    # Inventory pixels of each color.
    for i, color in enumerate(colors):
        for ch, bias in ((7 + i, -1056), (14 + i, -1057)):
            W1[ch, color, 2, 2] = 1000
            W1[ch, 16, 2, 2] = 63
            b1[ch] = bias
        W2[7 + i, 7 + i, 1, 1] = 1
        W2[7 + i, 14 + i, 1, 1] = -1

    # Black selection outline in the inventory area.
    for ch, bias in ((21, -1055), (22, -1056)):
        W1[ch, 0, 2, 2] = 1000
        W1[ch, 16, 2, 2] = 63
        b1[ch] = bias
    W2[14, 21, 1, 1] = 1
    W2[14, 22, 1, 1] = -1

    # Red pixels restricted to the playing area.
    for ch, row_weight, bias in (
        (23, 63, -1008),
        (24, 63, -1009),
        (25, -63, -946),
        (26, -63, -947),
    ):
        W1[ch, 2, 2, 2] = 1000
        W1[ch, 16, 2, 2] = row_weight
        b1[ch] = bias
    W1[27, 2, 2, 2] = 1

    def slot_detector(out, scale):
        for ch, weight in ((23, 1), (24, -1), (25, 1), (26, -1)):
            W2[out, ch, 1, 1] += scale * weight
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            W2[out, 27, 1 + dy, 1 + dx] += scale
        W2[out, 27, 0, 1] -= scale
        W2[out, 27, 1, 0] -= scale

    slot_detector(15, 1)
    b2[15] = -5

    # Red inventory markers indicate a moved item.
    for ch, weight in ((23, 1), (24, -1), (25, -1), (26, 1)):
        W2[17, ch, 1, 1] = weight

    # Coordinate-dependent ordering for empty slots.
    W1[30, 18, 2, 2] = 1
    W1[30, 17, 2, 2] = -0.2
    W1[30, 16, 2, 2] = -0.01

    W1[31, 16, 2, 2] = 1
    W1[31, 17, 2, 2] = 1
    b1[31] = -1

    slot_detector(18, 2)
    W2[18, 30, 1, 1] = 1
    W2[18, 31, 1, 1] = 0.02
    b2[18] = -12

    # L-shaped templates for a black selection outline.
    for dx in range(5):
        W1[28, 0, 0, dx] = 1
    for dy in range(1, 5):
        W1[28, 0, dy, 0] = 1

    for dx in range(5):
        W1[29, 0, 4, dx] = 1
    for dy in range(4):
        W1[29, 0, dy, 4] = 1

    W2[19, 28, 1, 1] = 1
    W2[19, 29, 2, 2] = 1
    b2[19] = -17

    W3 = p["W3"]
    for i in range(7):
        W3[i, 32 + i] = 1
        W3[7 + i, 7 + i] = 1
    W3[14, 32 + 14] = 1
    W3[15, 15] = 1
    W3[17, 17] = 1
    W3[18, 18] = 1
    W3[19, 19] = 1

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Broadcast target priorities downward.
    for i in range(7):
        Wh[i, i, 0, 1] = 1
        Uh[i, i] = 1

    # Broadcast inventory-selection evidence vertically.
    Wh[14, 14, 2, 1] = 1
    Wh[15, 15, 0, 1] = 1
    Uh[14, 14] = 1
    Uh[15, 14] = 1

    # Score matching inventory colors, suppressing them when selected.
    for i in range(7):
        q = 7 + i
        Wh[q, i, 1, 1] = 1
        Wh[q, 14, 1, 1] = -10
        Wh[q, 15, 1, 1] = -10
        Uh[q, 7 + i] = 2
        bh[q] = -2

    blank_max = 32 + 15
    for i in range(7):
        p["Wz"][i, 7 + i] = 1
        p["Wz"][i, blank_max] = 1
        p["bz"][i] = -1

    p["Wz"][7, 14] = 1
    p["Wz"][7, blank_max] = 100
    p["bz"][7] = -100

    # Disturbance indicators: moved item or selection.
    p["Wz"][8, 32 + 17] = 1
    p["Wz"][9, 32 + 19] = 1
    p["Wz"][10, 32 + 14] = 1

    # Memory distinguishes the first step from subsequent steps.
    p["Wm"][0, 0] = 2
    p["bm"][0] = 1

    p["Wo"][0, 8] = 3
    p["Wo"][0, 9] = 3
    p["Wo"][0, 10] = 3
    p["Wo"][0, 128] = -30
    p["bo"][0] = 22.7

    # Click when placement appears possible; otherwise test.
    p["Wo"][6, :8] = 1
    p["bo"][5] = 0.01

    for i in range(7):
        p["Wc"][0, 32 + 7 + i] = 1
    p["Wc"][0, 18] = 0.05

    return p
