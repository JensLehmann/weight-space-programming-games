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

    glyphs = [
        ["00100", "00100", "01110", "00100", "11111"],
        ["00001", "00101", "11111", "10100", "10000"],
        ["11111", "10100", "10000", "10100", "11111"],
        ["00100", "11111", "10101", "10101", "00100"],
        ["10001", "11111", "00100", "11111", "10001"],
        ["11111", "00100", "11100", "10000", "11111"],
        ["10001", "11111", "10001", "10001", "11011"],
        ["10000", "11110", "10010", "10010", "11111"],
        ["11111", "10001", "11101", "10101", "11111"],
        ["00111", "00101", "11111", "10100", "11100"],
        ["01110", "01010", "11111", "10001", "11111"],
        ["11111", "10001", "10001", "11111", "00100"],
        ["11110", "10011", "10001", "11001", "01111"],
        ["00100", "11111", "10101", "11111", "00100"],
    ]

    # Encode the counts on each orbit of the 5x5 rotation group.
    representatives = {}
    for r in range(5):
        for c in range(5):
            orbit = []
            y, x = r, c
            for _ in range(4):
                orbit.append((y, x))
                y, x = x, 4 - y
            key = min(orbit)
            if key not in representatives:
                representatives[key] = len(representatives)

    kernel = np.zeros((5, 5), dtype=np.float32)
    for r in range(5):
        for c in range(5):
            orbit = []
            y, x = r, c
            for _ in range(4):
                orbit.append((y, x))
                y, x = x, 4 - y
            kernel[r, c] = 9 ** representatives[min(orbit)]

    # Two ReLUs represent the absolute distance from each signature.
    for i, rows in enumerate(glyphs):
        mask = np.array([[int(v) for v in row] for row in rows],
                        dtype=np.float32)
        signature = float(np.sum(mask * kernel))
        p["W1"][2 * i, 5] = kernel
        p["b1"][2 * i] = -signature
        p["W1"][2 * i + 1, 5] = -kernel
        p["b1"][2 * i + 1] = signature

    # Restrict matching to the bottom part of the observation.
    p["W1"][28, 16, 2, 2] = -63.0
    p["b1"][28] = 38.0
    p["W1"][29, 16, 2, 2] = 63.0
    p["b1"][29] = -60.0

    # Five black pixels identify the center of a selection bracket.
    p["W1"][30, 0, 2, :] = 1.0
    p["b1"][30] = -4.5

    for i in range(14):
        p["W2"][i, 2 * i, 1, 1] = -1.0
        p["W2"][i, 2 * i + 1, 1, 1] = -1.0
        p["W2"][i, 28, 1, 1] = -100.0
        p["W2"][i, 29, 1, 1] = -100.0
        p["b2"][i] = 1.0
    p["W2"][14, 30, 1, 1] = 2.0

    # Associate glyph signatures with the bracket's column.
    for i in range(14):
        p["W3"][i, 64 + i] = 1.0
        p["W3"][i, 64 + 14] = 1.0
        p["b3"][i] = -1.0

    # Each z unit detects a pair of ordinal glyph signatures.
    for a in range(7):
        for b in range(7):
            j = 7 * a + b
            p["Wz"][j, 32 + a] = 1.0
            p["Wz"][j, 32 + 7 + b] = 1.0
            p["bz"][j] = -1.5

            delta = (a - b) % 7
            if delta == 0:
                action = 4
            elif delta <= 3:
                action = 2
            else:
                action = 1
            p["Wo"][action, j] = 4.0

    # Clock: after several unmatched increments, advance selection.
    for i in range(7):
        p["Wm"][i, (i - 1) % 7] = 6.0
        p["bm"][i] = -3.0
        p["Wma"][i, 4] = -6.0
    p["bm"][0] = 3.0
    p["Wma"][0, 4] = 6.0

    p["bo"][:] = -10.0
    p["bo"][1] = -0.25
    p["bo"][2] = 0.0
    p["bo"][4] = -0.5
    p["Wo"][4, 128 + 6] = 1.0

    return p
