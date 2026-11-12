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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    W1, b1 = p["W1"], p["b1"]
    W2 = p["W2"]
    W3 = p["W3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    colors = [10, 6, 15, 11, 12, 8, 9, 7, 14]
    sizes = [1, 2, 3, 4, 5, 7, 8, 9, 10]

    def pixel(out, color, dy, dx, value):
        W1[out, color, dy + 2, dx + 2] += value

    # Solid-fruit corner markers.
    for i, (color, size) in enumerate(zip(colors, sizes)):
        if size == 1:
            pixel(i, color, 0, 0, 1)
            pixel(i, color, -1, 0, -1)
            pixel(i, color, 0, -1, -1)
            b1[i] = -0.5
        else:
            for dy, dx in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
                pixel(i, color, dy, dx, 1)
            for dy, dx in [(-2, -1), (-2, 0), (-1, -2), (0, -2)]:
                pixel(i, color, dy, dx, -1)
            b1[i] = -3.5

    # Goal-pad top-edge marker.
    for dy, dx in [(-1, 0), (-1, 1), (-1, 2), (0, -1)]:
        pixel(9, 9, dy, dx, 1)
    for dy, dx in [(-1, -1), (-2, 0)]:
        pixel(9, 9, dy, dx, -1)
    b1[9] = -3.5

    # Guards for the board, header legend, and bottom bar.
    W1[10, 16, 2, 2] = -63.0
    b1[10] = 10.0
    W1[16, 16, 2, 2] = 63.0
    b1[16] = -9.0
    W1[17, 2, 2, 2] = 1.0
    W1[18, 17, 2, 2] = -63.0
    b1[18] = 26.0
    W1[19, 16, 2, 2] = 63.0
    b1[19] = -62.0

    # Monster patterns.
    monster_patterns = [
        (7, [
            "..#..",
            ".#.#.",
            "#...#",
            ".###.",
        ]),
        (14, [
            "..#..",
            ".#.#.",
            "#.#.#",
            ".#.#.",
        ]),
        (13, [
            "#.#.#",
            ".#.#.",
            "#####",
            ".###.",
        ]),
    ]
    for j, (color, pattern) in enumerate(monster_patterns):
        out = 13 + j
        positives = 0
        for r, line in enumerate(pattern):
            for c, ch in enumerate(line):
                if ch == "#":
                    pixel(out, color, r - 1, c - 2, 1)
                    positives += 1
                else:
                    pixel(out, color, r - 1, c - 2, -1)
        b1[out] = -(positives - 0.5)

    # Board fruit, shifted toward the center for larger sprites.
    for i, size in enumerate(sizes):
        shift = 1 if size >= 4 else 0
        ky = 1 - shift
        kx = 1 - shift
        W2[i, i, ky, kx] = 2.0
        W2[i, 10, ky, kx] = -2.0

    # Header fruit targets, excluding the left-hand legend.
    for i in range(9):
        W2[9 + i, i, 1, 1] = 2.0
        W2[9 + i, 16, 1, 1] = -2.0
        W2[9 + i, 18, 1, 1] = -2.0

    # Goal marker.
    W2[18, 9, 0, 0] = 2.0
    W2[18, 10, 0, 0] = -2.0

    # Board monsters and header monster targets.
    for j in range(3):
        W2[19 + j, 13 + j, 0, 1] = 2.0
        W2[19 + j, 10, 0, 1] = -2.0
        W2[26 + j, 13 + j, 1, 1] = 2.0
        W2[26 + j, 16, 1, 1] = -2.0
        W2[26 + j, 18, 1, 1] = -2.0

    # Red header and invalid click regions.
    W2[25, 17, 1, 1] = 1.0
    W2[25, 16, 1, 1] = -2.0
    W2[29, 10, 1, 1] = 1.0
    W2[30, 19, 1, 1] = 1.0

    # f3 features.
    for i in range(9):
        W3[0, i] = 1.0
        W3[4, i] = float(i + 1)
        W3[3, 32 + 9 + i] = float(i + 1)

    W3[1, 64 + 18] = 1.0  # goal columns
    W3[2, 32 + 18] = 1.0  # goal rows
    W3[10, 18] = 1.0      # individual goal markers
    W3[11, 0] = 1.0       # smallest fruit
    W3[12, 1] = 1.0       # second-smallest fruit

    for j in range(3):
        W3[5, 19 + j] = 1.0
        W3[6, 32 + 26 + j] = 1.0

    W3[7, 29] = 1.0
    W3[8, 30] = 1.0
    W3[9, 25] = 1.0

    # Capped directional waves for goal-axis proximity.
    directions = [
        (1, 1, 0),
        (1, 1, 2),
        (2, 0, 1),
        (2, 2, 1),
    ]
    for k, (source, ky, kx) in enumerate(directions):
        a = 2 * k
        e = a + 1
        for out in (a, e):
            Wh[out, a, ky, kx] = 1.0
            Wh[out, e, ky, kx] = -1.0
            Uh[out, source] = 100.0
        bh[a] = -1.0
        bh[e] = -101.0

    neighbors = [(0, 1), (2, 1), (1, 0), (1, 2)]

    # Local two-dimensional goal field.
    for ky, kx in neighbors:
        Wh[8, 8, ky, kx] = 0.249
    Uh[8, 10] = 1.0

    # Fruit sources are suppressed near a goal.
    Uh[9, 0] = 1.0
    Wh[9, 8, 1, 1] = -2.0

    # Local attraction and its capped excess.
    for ky, kx in neighbors:
        Wh[10, 10, ky, kx] = 0.24
    Wh[10, 9, 1, 1] = 1.0
    Wh[10, 14, 1, 1] = 1.0
    Wh[11, 10, 1, 1] = 1.0
    bh[11] = -0.03

    # Separate fields for the two smallest fruit levels.
    for ky, kx in neighbors:
        Wh[12, 12, ky, kx] = 0.24
        Wh[13, 13, ky, kx] = 0.24
    Uh[12, 11] = 1.0
    Uh[13, 12] = 1.0

    # Broadcast a header monster target down the board.
    Wh[15, 15, 0, 1] = 1.0
    Uh[15, 6] = 1.0

    # Relevant board monsters become attraction sources.
    Uh[14, 5] = 2.0
    Wh[14, 15, 1, 1] = 1.0
    Wh[14, 8, 1, 1] = -2.0
    bh[14] = -2.0

    # Click map.
    Wc = p["Wc"]
    for a, e in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        Wc[0, 32 + a] += 1.0
        Wc[0, 32 + e] -= 1.0
    Wc[0, 1] = -100.0
    Wc[0, 2] = -100.0
    Wc[0, 32 + 8] = 5.0

    # Prefer valid clicks near movable objects.
    Wc[0, 32 + 10] = 10000.0
    Wc[0, 32 + 11] = -10000.0

    # Mild preference for clustering like-sized small fruit.
    Wc[0, 32 + 12] = 0.2
    Wc[0, 32 + 13] = 0.2
    Wc[0, 11] = -0.2
    Wc[0, 12] = -0.2

    Wc[0, 7] = -1000.0
    Wc[0, 8] = -1000.0

    # Reset if the goal indicator turns red.
    p["Wz"][0, 32 + 9] = 1.0
    p["Wmz"][0, 0] = 10.0
    p["bo"][0] = -5.0
    p["Wo"][0, 128] = 10.0

    return p
