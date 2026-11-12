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

    # Pass player color, normalized row, and normalized column.
    p["W1"][0, 10, 2, 2] = 1
    p["W1"][1, 16, 2, 2] = 1
    p["W1"][2, 17, 2, 2] = 1
    for i in range(3):
        p["W2"][i, i, 1, 1] = 1

    # At player pixels, produce x, 1-x, y, and 1-y.
    p["W3"][0, 0] = 1
    p["W3"][0, 2] = 1
    p["b3"][0] = -1

    p["W3"][1, 0] = 1
    p["W3"][1, 2] = -1

    p["W3"][2, 0] = 1
    p["W3"][2, 1] = 1
    p["b3"][2] = -1

    p["W3"][3, 0] = 1
    p["W3"][3, 1] = -1

    # The last 32 max-pooled entries of g are f3 maxima.
    for i in range(4):
        p["Wz"][i, 64 + i] = 1

    # Extent of the observed player pixels in each direction.
    p["Wz"][4, 64] = 1
    p["Wz"][4, 65] = 1
    p["bz"][4] = -1

    p["Wz"][5, 66] = 1
    p["Wz"][5, 67] = 1
    p["bz"][5] = -1

    # Store the current extrema and extents for the next actor step.
    for i in range(6):
        p["Wmz"][i, i] = 0.1

    # Two recurrent oscillators provide deterministic exploration.
    p["Wm"][6, 6] = 0.91
    p["Wm"][6, 7] = -0.50
    p["Wm"][7, 6] = 0.50
    p["Wm"][7, 7] = 0.91
    p["bm"][6] = 0.01

    p["Wm"][8, 8] = 1.00
    p["Wm"][8, 9] = -0.30
    p["Wm"][9, 8] = 0.30
    p["Wm"][9, 9] = 1.00
    p["bm"][8] = 0.01

    # Remember which directional action was taken.
    for j in range(4):
        p["Wma"][10 + j, 1 + j] = 5

    # For each previous action, compare all four extrema with their
    # stored values. Both signed tests are positive when unchanged.
    k = 500.0
    threshold = 0.0018
    for j in range(4):
        for d in range(4):
            for s_index, sign in enumerate((1.0, -1.0)):
                q = 14 + j * 8 + d * 2 + s_index
                p["Wmz"][q, d] = sign * k * 0.1
                p["Wm"][q, d] = -sign * k
                p["Wma"][q, 1 + j] = 15
                p["bm"][q] = k * threshold - 15

    # Compare the current and previous horizontal and vertical extents,
    # gated by which directional action was taken.
    k_gap = 250.0
    for j in range(4):
        for axis in range(2):
            for s_index, sign in enumerate((1.0, -1.0)):
                q = 46 + j * 4 + axis * 2 + s_index
                feature = 4 + axis
                p["Wmz"][q, feature] = sign * k_gap * 0.1
                p["Wm"][q, feature] = -sign * k_gap
                p["Wma"][q, 1 + j] = 15
                p["bm"][q] = -15

    p["bo"][:] = -10
    for j in range(4):
        action = 1 + j
        p["bo"][action] = 0

        # Penalize a repeated action if its previous use did not move
        # either observable player extremum.
        for d in range(8):
            p["Wo"][action, 128 + 14 + 8 * j + d] = -2
        p["Wo"][action, 128 + 10 + j] = 26.2

        # Prefer repeating an action that reduced player separation.
        for axis in range(2):
            q = 46 + j * 4 + axis * 2
            p["Wo"][action, 128 + q] = -0.8
            p["Wo"][action, 128 + q + 1] = 0.8

    # Discourage immediate reversals.
    opposites = (1, 0, 3, 2)
    for j in range(4):
        p["Wo"][1 + j, 128 + 10 + opposites[j]] -= 0.5

    # Oscillator projections into the four directional logits.
    p["Wo"][1, 128 + 6] += 1.4
    p["Wo"][1, 128 + 8] += 1.0
    p["Wo"][2, 128 + 6] -= 1.4
    p["Wo"][2, 128 + 8] -= 1.0
    p["Wo"][3, 128 + 7] += 1.4
    p["Wo"][3, 128 + 9] -= 1.0
    p["Wo"][4, 128 + 7] -= 1.4
    p["Wo"][4, 128 + 9] += 1.0

    # Small initial preferences.
    p["bo"][1] = 0.15
    p["bo"][4] = 0.05

    return p
