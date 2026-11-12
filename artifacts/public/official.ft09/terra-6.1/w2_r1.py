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

    groups = [
        (1, 7, 8, 10, 14),
        (9, 11, 12, 13, 15),
    ]
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),             (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3 = p["W3"]

    def add1(out, colors, dy, dx, value=1.0):
        for color in colors:
            W1[out, color, dy + 2, dx + 2] += value

    def add2(out, inp, dy=0, dx=0, value=1.0):
        W2[out, inp, dy + 1, dx + 1] += value

    # Clue centres.
    for g, colors in enumerate(groups):
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            add1(g, colors, dy, dx)
        add1(g, (0, 2, 3), -1, 0)
        add1(g, (0, 2, 3), 0, -1)
        b1[g] = -5

    # Eight zero-valued clue markers.
    for d, (dy, dx) in enumerate(directions):
        sy = -1 if dy < 0 else (2 if dy > 0 else 0)
        sx = -1 if dx < 0 else (2 if dx > 0 else 0)
        add1(2 + d, (0,), sy, sx)

    # Solid-colour tiles.
    for g, colors in enumerate(groups):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                add1(10 + g, colors, dy, dx)
        b1[10 + g] = -24

    # Background samples exclude the level-one demonstrations.
    add1(12, (4,), 0, -2)
    add1(13, (4,), -2, 0)

    # Cross-flipping tile centres.
    for g, colors in enumerate(groups):
        out = 14 + g
        for dy, dx in (
            (0, 0), (0, 1), (1, 0), (1, 1),
            (-1, -1), (-1, 2), (2, -1), (2, 2),
        ):
            add1(out, colors, dy, dx)
        for dy, dx in ((-1, 0), (0, -1), (0, 2), (2, 0)):
            add1(out, (6,), dy, dx)
        b1[out] = -11

    add2(0, 0)
    add2(1, 1)

    # Clue group AND nonzero marker.
    for d in range(8):
        for g in range(2):
            out = 2 + 2 * d + g
            add2(out, g)
            add2(out, 2 + d, value=-1)

    # Canonical solid-tile centres.
    for g in range(2):
        out = 18 + g
        s = 10 + g
        add2(out, s)
        add2(out, s, 0, -1, -1)
        add2(out, s, -1, 0, -1)
        add2(out, 12, 0, -1)
        add2(out, 13, -1, 0)
        b2[out] = -2

    add2(20, 14)
    add2(21, 15)

    # Directional ray terminators: immediately after a tile, and
    # immediately before another clue.
    for d, (dy, dx) in enumerate(directions):
        out = 22 + d
        for g in range(2):
            add2(out, 10 + g, -dy, -dx)
            add2(out, 10 + g, 0, 0, -1)
            add2(out, 14 + g, -dy, -dx)
            add2(out, g, dy, dx)

    # Desired colour group for each clue direction.
    for d in range(8):
        a_nonzero = 2 + 2 * d
        b_nonzero = a_nonzero + 1

        W3[2 * d, 0] = 1
        W3[2 * d, a_nonzero] = -1
        W3[2 * d, b_nonzero] = 1

        W3[2 * d + 1, 1] = 1
        W3[2 * d + 1, a_nonzero] = 1
        W3[2 * d + 1, b_nonzero] = -1

    for i in range(4):
        W3[16 + i, 18 + i] = 1
    for d in range(8):
        W3[20 + d, 22 + d] = 1

    # Propagate directional requirements. Signals reach a tile eight
    # display pixels away but decay before reaching a second tile.
    for d, (dy, dx) in enumerate(directions):
        for desired in range(2):
            ch = 2 * d + desired
            p["Wh"][ch, ch, 1 - dy, 1 - dx] = 1
            p["Uh"][ch, ch] = 100
            p["bh"][ch] = -10
            p["Uh"][ch, 20 + d] = -100

            for actual in range(2):
                match_penalty = 20 if actual == desired else 0
                p["Uh"][ch, 16 + actual] = -9 - match_penalty
                p["Uh"][ch, 18 + actual] = -5 - match_penalty

    # Click at a tile centre with the largest mismatch score.
    for i in range(16, 20):
        p["Wc"][0, i] = 10000
    p["Wc"][0, 32:48] = 1

    p["bo"][6] = 1
    return p
