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

    classes = (
        (1, 7, 8, 10, 14),
        (9, 11, 12, 13, 15),
    )
    border_colors = (0, 2, 3)
    directions = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),             (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    )

    W1, b1 = p["W1"], p["b1"]

    def sample(out, colors, dy, dx, weight=1.0):
        for color in colors:
            W1[out, color, dy + 2, dx + 2] += weight

    def gap(out, dy, dx):
        # Background 4 or outside the image.
        sample(out, (4,), dy, dx)
        W1[out, 18, dy + 2, dx + 2] -= 1.0
        b1[out] += 1.0

    # Tile anchors: ordinary class 0/1, then special class 0/1.
    for special in range(2):
        for state in range(2):
            out = 2 * special + state
            colors = classes[state]
            for dy, dx in ((0, 0), (0, -1), (-1, 0), (1, 1)):
                sample(out, colors, dy, dx)
            gap(out, 0, -2)
            gap(out, -2, 0)
            axis_colors = (6,) if special else colors
            sample(out, axis_colors, 0, 1)
            sample(out, axis_colors, 1, 0)
            b1[out] -= 7.0

    # Clue anchors, separated by the class of the center color.
    for state in range(2):
        out = 4 + state
        for dy, dx in ((0, 0), (0, -1), (-1, 0)):
            sample(out, border_colors, dy, dx)
        sample(out, classes[state], 1, 1)
        gap(out, 0, -2)
        gap(out, -2, 0)
        b1[out] -= 5.0

    # One sensor per clue-border position.
    for d, (dy, dx) in enumerate(directions):
        sy = -1 if dy < 0 else (1 if dy == 0 else 2)
        sx = -1 if dx < 0 else (1 if dx == 0 else 2)
        sample(6 + d, (0,), sy, sx)

    W2, b2 = p["W2"], p["b2"]

    for i in range(4):
        W2[i, i, 1, 1] = 1.0

    W2[4, 4, 1, 1] = 1.0
    W2[4, 5, 1, 1] = 1.0

    # For each direction, detect the cases requiring class 1:
    # class-0 center with nonzero border, or class-1 center with zero.
    for d, (dy, dx) in enumerate(directions):
        sy = 1 if dy > 0 else 0
        sx = 1 if dx > 0 else 0
        a = 5 + 2 * d
        b = a + 1
        W2[a, 4, 1, 1] = 1.0
        W2[a, 6 + d, 1 + sy, 1 + sx] = -1.0
        W2[b, 5, 1, 1] = 1.0
        W2[b, 6 + d, 1 + sy, 1 + sx] = 1.0
        b2[b] = -1.0

    W3 = p["W3"]

    # Channels 0:8 emit class-0 requirements, 8:16 class-1.
    for d in range(8):
        a = 5 + 2 * d
        b = a + 1
        W3[d, 4] = 2.25
        W3[d, a] = -2.25
        W3[d, b] = -2.25
        W3[8 + d, a] = 2.25
        W3[8 + d, b] = 2.25

    # Tile state channels and a combined anchor channel.
    for i in range(4):
        W3[16 + i, i] = 1.0
        W3[20, i] = 1.0

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    for state in range(2):
        for d, (dy, dx) in enumerate(directions):
            c = 8 * state + d
            Wh[c, c, 1 - dy, 1 - dx] = 1.0
            Uh[c, c] = 1.0
            bh[c] = -0.25

            # A matching tile suppresses the signal. A mismatching
            # tile gives it a small positive response at distance 8.
            matching = (16 + state, 18 + state)
            other = 1 - state
            mismatching = (16 + other, 18 + other)
            for f in matching:
                Uh[c, f] = -2.0
            for f in mismatching:
                Uh[c, f] = 0.125

    p["bo"][6] = 1.0

    # Restrict clicks to tile anchors and prefer constraint violations.
    p["Wc"][0, 20] = 1000.0
    p["Wc"][0, 18] = 0.05
    p["Wc"][0, 19] = 0.05
    for c in range(16):
        p["Wc"][0, 32 + c] = 10.0

    return p
