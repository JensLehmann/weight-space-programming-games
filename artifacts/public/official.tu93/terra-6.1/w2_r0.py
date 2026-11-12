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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    # Player markers displaced one pixel in each action direction.
    for i, (dy, dx) in enumerate(directions):
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                ky = 2 - dy + oy
                kx = 2 - dx + ox
                p["W1"][i, 9, ky, kx] = 1.0
                p["W1"][i, 4, ky, kx] = 1.0
        p["b1"][i] = -8.0

    # Track, goal, and corridor.
    for color in (0, 2, 4, 8, 9, 11, 12, 13, 14, 15):
        p["W1"][4, color, 2, 2] = 1.0
    p["W1"][5, 14, 2, 2] = 1.0
    p["W1"][10, 2, 2, 2] = 1.0

    # Blue and purple enemy markers, one pixel ahead of their centers.
    for i, (dy, dx) in enumerate(directions):
        c = 6 + i
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                ky = 2 - dy + oy
                kx = 2 - dx + ox
                for color in (8, 12, 11, 15):
                    p["W1"][c, color, ky, kx] = 1.0
        p["W1"][c, 11, 2, 2] += 1.0
        p["W1"][c, 15, 2, 2] += 1.0
        p["b1"][c] = -9.0

        # Active delayed pursuers.
        c = 11 + i
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                ky = 2 - dy + oy
                kx = 2 - dx + ox
                p["W1"][c, 13, ky, kx] = 1.0
                p["W1"][c, 11, ky, kx] = 1.0
        p["W1"][c, 11, 2, 2] += 1.0
        p["b1"][c] = -9.0

    # Legal outgoing edges, two pixels from the player center.
    for i, (dy, dx) in enumerate(directions):
        p["W2"][i, i, 1 - dy, 1 - dx] = 1.0
        p["W2"][i, 10, 1, 1] = 1.0
        p["b2"][i] = -1.0

    p["W2"][4, 4, 1, 1] = 1.0
    p["W2"][5, 5, 1, 1] = 1.0

    # Enemy ray sources, two pixels ahead of their centers.
    for i, (dy, dx) in enumerate(directions):
        p["W2"][6 + i, 6 + i, 1 - dy, 1 - dx] = 1.0
        p["W2"][6 + i, 11 + i, 1 - dy, 1 - dx] = 1.0

    for i in range(10):
        p["W3"][i, i] = 1.0

    # Goal reachability wave: h[0] - h[1] is approximately binary.
    for out in (0, 1):
        for ky, kx in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)):
            p["Wh"][out, 0, ky, kx] = 1.0
            p["Wh"][out, 1, ky, kx] = -1.0
        p["Uh"][out, 4] = 6.0
        p["Uh"][out, 5] = 1.0
    p["bh"][0] = -6.0
    p["bh"][1] = -7.0

    # Earlier wave arrival yields a larger goal-distance score.
    p["Wh"][2, 2, 1, 1] = 1.0
    p["Wh"][2, 0, 1, 1] = 1.0
    p["Wh"][2, 1, 1, 1] = -1.0

    # Sample goal distance and local projected danger at each edge.
    for i, (dy, dx) in enumerate(directions):
        c = 3 + i
        p["Wh"][c, 2, 1, 1] = 1.0
        p["Wh"][c, 14, 1 + dy, 1 + dx] = -40000.0
        p["Uh"][c, i] = 330.0
        p["bh"][c] = -130.0

    # Directional rays. For an isolated enemy, the ray is 1/16 at
    # its projected next center.
    for i, (dy, dx) in enumerate(directions):
        c = 7 + i
        p["Wh"][c, c, 1 - dy, 1 - dx] = 0.5
        p["Uh"][c, 6 + i] = 1.0

    # Triangular band-pass centered at 1/16.
    for c in (11, 12, 13):
        for i in range(4):
            p["Wh"][c, 7 + i, 1, 1] = 1.0
    p["bh"][11] = -1.0 / 32.0
    p["bh"][12] = -2.0 / 32.0
    p["bh"][13] = -3.0 / 32.0

    # A sharply decaying local field around projected enemy centers.
    p["Wh"][14, 11, 1, 1] = 32.0
    p["Wh"][14, 12, 1, 1] = -64.0
    p["Wh"][14, 13, 1, 1] = 32.0
    for ky, kx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        p["Wh"][14, 14, ky, kx] = 0.1

    for i in range(4):
        p["Wz"][i, 3 + i] = 1.0
        p["Wz"][4 + i, 32 + i] = 1.0

    # Previous action and a decaying history of action directions.
    for i in range(4):
        p["Wma"][i, 1 + i] = 1.0
        p["Wm"][4 + i, 4 + i] = 0.6
        p["Wma"][4 + i, 1 + i] = 0.4

    p["bo"][:] = -1000.0
    for i in range(4):
        action = 1 + i
        p["bo"][action] = -500.0
        p["Wo"][action, i] = 1.0
        p["Wo"][action, 4 + i] = 500.0
        p["Wo"][action, 128 + opposite[i]] = -2.0
        p["Wo"][action, 128 + 4 + i] = -1.0

    return p
