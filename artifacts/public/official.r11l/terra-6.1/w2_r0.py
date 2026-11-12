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

    def control_filter(out, center_colors, outline_colors,
                       oy=0, ox=0, extra=()):
        for color in center_colors:
            W1[out, color, 2 + oy, 2 + ox] += 2.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                for color in outline_colors:
                    W1[out, color, 2 + oy + dy, 2 + ox + dx] += 2.0
        for dy, dx, color in extra:
            W1[out, color, 2 + oy + dy, 2 + ox + dx] += 2.0
        b1[out] = -2.0 * (9 + len(extra)) + 1.0

    # Selected and unselected colored controls.
    control_filter(0, [14], [0])
    control_filter(1, [15], [0])
    control_filter(2, [14], [3])
    control_filter(3, [15], [3])

    # Centers of simple goal outlines.
    for out, color in ((4, 14), (5, 15)):
        for dy in (-2, 2):
            for dx in (-2, 2):
                W1[out, color, 2 + dy, 2 + dx] = 2.0
        b1[out] = -7.0

    # Selected controls near goal outlines.
    slot_pairs = [
        [(-2, -1), (2, -1)],
        [(-2, 1), (2, 1)],
        [(-1, -2), (-1, 2)],
        [(1, -2), (1, 2)],
    ]
    for color, start in ((14, 6), (15, 10)):
        for k, pair in enumerate(slot_pairs):
            control_filter(
                start + k,
                [color],
                [0],
                extra=[(dy, dx, color) for dy, dx in pair],
            )

    # Offset control detectors for 5x5 selection boxes.
    k = 14
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            control_filter(k, [4, 14, 15], [0, 3], oy, ox)
            k += 1

    # Wall footprint, nearby ball and column coordinate.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) + abs(dx) <= 2:
                W1[23, 2, 2 + dy, 2 + dx] = 1.0
    W1[24, 6, :, :] = 1.0
    W1[25, 17, 2, 2] = 1.0

    for i in range(14):
        W2[i, i, 1, 1] = 1.0

    # Selected control exactly on a goal.
    for out, selected, goal in ((14, 0, 4), (15, 1, 5)):
        W2[out, selected, 1, 1] = 2.0
        W2[out, goal, 1, 1] = 2.0
        b2[out] = -3.0

    # Unoccupied goal centers.
    for out, goal, selected, unselected in (
        (16, 4, 0, 2),
        (17, 5, 1, 3),
    ):
        W2[out, goal, 1, 1] = 1.0
        W2[out, selected, 1, 1] = -1.0
        W2[out, unselected, 1, 1] = -1.0

    # Occupied goal, reported one column to its left.
    for out, goal, selected, unselected in (
        (18, 4, 0, 2),
        (19, 5, 1, 3),
    ):
        W2[out, goal, 1, 2] = 2.0
        W2[out, selected, 1, 2] = 2.0
        W2[out, unselected, 1, 2] = 2.0
        b2[out] = -3.0

    for inp in range(14, 23):
        W2[26, inp, :, :] = 1.0
    W2[27, 23, 1, 1] = 1.0
    W2[28, 24, :, :] = 1.0
    W2[29, 24, :, :] = 1.0
    b2[29] = -1.0
    W2[30, 25, 1, 1] = 1.0

    for out, inp in (
        (0, 0), (1, 1), (2, 2), (3, 3),
        (4, 16), (5, 17),
        (6, 18), (7, 19),
        (8, 26), (9, 27), (14, 30),
    ):
        W3[out, inp] = 1.0

    W3[10, 14] = 1.0
    W3[10, 6:10] = 1.0
    W3[11, 15] = 1.0
    W3[11, 10:14] = 1.0

    for out, exact, first in ((12, 14, 6), (13, 15, 10)):
        W3[out, exact] = 2.0
        W3[out, first:first + 4] = 2.0
        W3[out, 28] = 2.0
        W3[out, 29] = -2.0
        b3[out] = -3.0

    def broadcast(base, selected, near_goal, near_ball):
        sources = (selected, near_goal, near_ball)

        Wh[base, base, 1, 0] = 1.0
        Wh[base + 1, base + 1, 1, 2] = 1.0
        for feature in sources:
            Uh[base, feature] = 1.0
            Uh[base + 1, feature] = 1.0

        Wh[base + 2, base + 2, 0, 1] = 1.0
        Wh[base + 3, base + 3, 2, 1] = 1.0
        for out in (base + 2, base + 3):
            Wh[out, base, 1, 1] = 1.0
            Wh[out, base + 1, 1, 1] = 1.0
            for feature in sources:
                Uh[out, feature] = -1.0

    broadcast(0, 0, 10, 12)
    broadcast(4, 1, 11, 13)

    # Rays from occupied goals.
    Wh[8, 8, 1, 2] = 1.0
    Wh[9, 9, 1, 2] = 1.0
    Uh[8, 6] = 1.0
    Uh[9, 7] = 1.0

    def add_global(out, base, sources):
        Wh[out, base, 1, 1] = -1.0
        Wh[out, base + 1, 1, 1] = -1.0
        Wh[out, base + 2, 1, 1] = 1.0
        Wh[out, base + 3, 1, 1] = 1.0
        for feature in sources:
            Uh[out, feature] += 1.0

    for out, base, sources, vacant, ray in (
        (10, 0, (0, 10, 12), 4, 8),
        (11, 4, (1, 11, 13), 5, 9),
    ):
        add_global(out, base, sources)
        Uh[out, vacant] += 10.0
        Wh[out, ray, 1, 2] += 10.0
        Uh[out, 8] -= 100.0
        Uh[out, 9] -= 100.0
        bh[out] = -10.5

    for out, base, sources, unselected in (
        (12, 0, (0, 10, 12), 2),
        (13, 4, (1, 11, 13), 3),
    ):
        add_global(out, base, sources)
        Uh[out, unselected] += 10.0
        bh[out] = -11.5

    for out, base, sources, other_unselected in (
        (14, 0, (0, 10, 12), 3),
        (15, 4, (1, 11, 13), 2),
    ):
        add_global(out, base, sources)
        Uh[out, other_unselected] += 10.0
        bh[out] = -12.5

    p["bo"][6] = 1.0
    p["Wc"][0, 14] = 0.001
    p["Wc"][0, 32 + 10] = 2.0
    p["Wc"][0, 32 + 11] = 2.0
    p["Wc"][0, 32 + 12] = 8.0
    p["Wc"][0, 32 + 13] = 8.0
    p["Wc"][0, 32 + 14] = 40.0
    p["Wc"][0, 32 + 15] = 40.0
    return p
