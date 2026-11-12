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
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Wc = p["Wc"]

    def stencil(channel, color, positive, negative):
        for dx, dy in positive:
            W1[channel, color, dy + 2, dx + 2] += 1.0
        for dx, dy in negative:
            W1[channel, color, dy + 2, dx + 2] -= 1.0
        b1[channel] = 0.5 - len(positive)

    # Exact local square detectors for small fruit.
    colors = [10, 6, 15, 11, 12]
    bounds = [(0, 0), (-1, 0), (-1, 1), (-2, 1), (-2, 2)]
    for c, (color, (lo, hi)) in enumerate(zip(colors, bounds)):
        positive = [
            (x, y)
            for y in range(lo, hi + 1)
            for x in range(lo, hi + 1)
        ]
        negative = [
            (x, y)
            for y in range(max(-2, lo - 1), min(2, hi + 1) + 1)
            for x in range(max(-2, lo - 1), min(2, hi + 1) + 1)
            if not (lo <= x <= hi and lo <= y <= hi)
        ]
        stencil(c, color, positive, negative)
        W2[c, c, 1, 1] = 2.0

    # Solid 5x5 cores of larger fruit.
    core = [(x, y) for y in range(-2, 3) for x in range(-2, 3)]
    for c, color in [(5, 8), (6, 7), (7, 14)]:
        stencil(c, color, core, [])

    # Center of a 7x7 square.
    for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        W2[5, 5, dy + 1, dx + 1] = 2.0
    b2[5] = -4.5

    # A representative core of larger solid squares.
    for c in (6, 7):
        W2[c, c, 1, 1] = 2.0
        W2[c, c, 0, 1] = -2.0
        W2[c, c, 1, 0] = -2.0
        b2[c] = -0.5

    # Board row masks.
    W1[16, 16, 2, 2] = -63.0
    b1[16] = 10.0
    W1[17, 16, 2, 2] = 63.0
    b1[17] = -62.0
    W2[10, 16, 1, 1] = 1.0
    W2[11, 17, 1, 1] = 1.0

    # Enemy templates.
    enemy_patterns = [
        (7, [
            (2, 0), (1, 1), (3, 1), (0, 2),
            (4, 2), (1, 3), (2, 3), (3, 3),
        ]),
        (14, [
            (2, 0), (1, 1), (3, 1), (0, 2),
            (2, 2), (4, 2), (1, 3), (3, 3),
        ]),
        (13, [
            (0, 0), (2, 0), (4, 0), (1, 1), (3, 1),
            (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
            (1, 3), (2, 3), (3, 3),
        ]),
    ]
    for j, (color, points) in enumerate(enemy_patterns):
        occupied = set(points)
        positive = [(x - 2, y - 1) for x, y in points]
        negative = [
            (x - 2, y - 1)
            for y in range(4)
            for x in range(5)
            if (x, y) not in occupied
        ]
        stencil(18 + j, color, positive, negative)
        W2[12, 18 + j, 1, 1] = 2.0

    # Nearby same-colour pair evidence.
    pair_specs = [
        (22, 10, 1.5),
        (23, 6, 7.5),
        (26, 15, 9.5),
        (27, 11, 16.5),
    ]
    for j, (c, color, threshold) in enumerate(pair_specs):
        W1[c, color, :, :] = 1.0
        b1[c] = -threshold
        W2[13 + j, c, 1, 1] = 2.0

    # Color 9 in the board region: goal pixels.
    W1[21, 9, 2, 2] = 1.0
    W2[20, 21, 1, 1] = 1.0
    W2[20, 16, 1, 1] = -10.0
    W2[20, 17, 1, 1] = -10.0

    # Detect a rank-3 goal icon in the right-hand HUD.
    W1[24, 17, 2, 2] = -63.0
    b1[24] = 24.0
    W1[25, 16, 2, 2] = 63.0
    b1[25] = -9.0
    W2[21, 3, 1, 1] = 2.0
    W2[21, 24, 1, 1] = -4.0
    W2[21, 25, 1, 1] = -4.0

    # Movable-object sources.
    for c in range(5):
        W3[0, c] = 1.0
    for c in (5, 6, 7):
        W3[0, c] = 2.0
    W3[0, 12] = 1.0
    W3[0, 10] = -10.0
    W3[0, 11] = -10.0

    # Goal pixels, and a rectangular goal-interior mask made from
    # their row and column projections.
    W3[1, 20] = 1.0
    W3[4, 32 + 20] = 1.0
    W3[4, 64 + 20] = 1.0
    b3[4] = -1.5

    # Invalid click rows.
    W3[3, 10] = 1.0
    W3[3, 11] = 1.0

    # Pair features.
    for j in range(4):
        W3[6 + j, 13 + j] = 1.0
        W3[6 + j, 10] = -10.0
        W3[6 + j, 11] = -10.0

    # Broadcast the row containing the rank-3 HUD icon and the row
    # containing a detected rank-5 fruit.
    W3[11, 32 + 21] = 1.0
    W3[12, 32 + 5] = 2.0

    # Goal attraction: a two-dimensional diffusion field.
    Wh[0, 0, 0, 1] = 0.25
    Wh[0, 0, 2, 1] = 0.25
    Wh[0, 0, 1, 0] = 0.25
    Wh[0, 0, 1, 2] = 0.25
    Uh[0, 1] = 1.0

    # Broadcast HUD evidence down the board.
    Wh[1, 1, 0, 1] = 1.0
    Uh[1, 11] = 1.0

    # Broadcast rank-5 fruit evidence above and below its row.
    Wh[2, 2, 0, 1] = 1.0
    Uh[2, 12] = 1.0
    Wh[3, 3, 2, 1] = 1.0
    Uh[3, 12] = 1.0

    # Wait-mode conjunction: rank-3 target and rank-5 fruit present.
    Wh[15, 1, 1, 1] = 1.0
    Wh[15, 2, 1, 1] = 1.0
    Wh[15, 3, 1, 1] = 1.0
    bh[15] = -1.5

    # Effective source: suppress objects at goals and during wait mode.
    Uh[12, 0] = 1.0
    Uh[12, 4] = -2.0
    Wh[12, 15, 1, 1] = -2.0

    # Four-neighbour max-propagation circuit, with a distance decay.
    Wh[5, 4, 0, 1] = 1.0
    Wh[5, 4, 1, 2] = -1.0
    Wh[6, 4, 2, 1] = 1.0
    Wh[6, 4, 1, 0] = -1.0
    Wh[7, 4, 1, 2] = 1.0
    Wh[8, 4, 1, 0] = 1.0

    Wh[9, 5, 1, 1] = 1.0
    Wh[9, 7, 1, 1] = 1.0
    Wh[9, 6, 1, 1] = -1.0
    Wh[9, 8, 1, 1] = -1.0
    Wh[10, 6, 1, 1] = 1.0
    Wh[10, 8, 1, 1] = 1.0

    Wh[11, 9, 1, 1] = 1.0
    Wh[11, 10, 1, 1] = 1.0
    Wh[11, 12, 1, 1] = -1.0
    bh[11] = -0.1

    Wh[4, 11, 1, 1] = 1.0
    Wh[4, 12, 1, 1] = 1.0

    # Narrow clipped reachability bonus.
    Wh[13, 4, 1, 1] = 1.0
    Wh[14, 4, 1, 1] = 1.0
    bh[13] = -0.15
    bh[14] = -0.16

    Wc[0, 32 + 0] = 1.0
    Wc[0, 3] = -1000.0
    Wc[0, 32 + 13] = 20000.0
    Wc[0, 32 + 14] = -20000.0

    Wc[0, 6] = 500.0
    Wc[0, 7] = 400.0
    Wc[0, 8] = 300.0
    Wc[0, 9] = 200.0

    p["bo"][6] = 1.0
    return p
